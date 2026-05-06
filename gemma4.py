import os
import json
from io import BytesIO
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_MAX_NEW_TOKENS = 1024

app = FastAPI(title="Gemma4 API", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=4096)
    image_url: str | None = None
    rf_detr_output: dict[str, Any] | None = None

 
def get_input_device(model):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for mapped_device in model.hf_device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str) and mapped_device.startswith("cuda"):
                return torch.device(mapped_device)
        return torch.device("cpu")

    return next(model.parameters()).device

processor = AutoProcessor.from_pretrained(MODEL_ID)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    user_prompt = request.prompt.strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    if request.rf_detr_output is not None:
        user_prompt = (
            f"{user_prompt}\n\n"
            "RF-DETR output (JSON):\n"
            f"{json.dumps(request.rf_detr_output, ensure_ascii=True)}"
        )

    image = None
    user_content: str | list[dict[str, str]]
    if request.image_url:
        try:
            image_resp = requests.get(request.image_url, timeout=30)
            image_resp.raise_for_status()
            image = Image.open(BytesIO(image_resp.content)).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"failed to fetch image_url: {exc}",
            ) from exc

        # For Gemma multimodal chat templates, use an image placeholder token
        # and provide the PIL image separately in processor(..., images=...).
        user_content = [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if image is not None:
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")
    input_device = get_input_device(model)
    inputs = {
        k: (v.to(input_device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }
    input_len = inputs["input_ids"].shape[-1]

    generate_kwargs = {"max_new_tokens": request.max_new_tokens}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        generate_kwargs["synced_gpus"] = True

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return {
        "response": processor.parse_response(response),
        "used_image_url": request.image_url,
    }


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)