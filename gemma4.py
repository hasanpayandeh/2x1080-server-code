import sys

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_MAX_NEW_TOKENS = 1024


def _normalize_map_device(mapped_device):
    if isinstance(mapped_device, int):
        return torch.device(f"cuda:{mapped_device}")

    if isinstance(mapped_device, str):
        if mapped_device in {"cpu", "mps"} or mapped_device.startswith("cuda"):
            return torch.device(mapped_device)
        # "disk" and "meta" are not valid tensor destinations.
        return None

    return None


def get_inputs_device(model_obj):
    device_map = getattr(model_obj, "hf_device_map", None)
    if device_map:
        preferred_fallback = None

        # Prefer the embedding shard when available.
        for key in ("model.embed_tokens", "embed_tokens", "lm_head"):
            if key in device_map:
                normalized = _normalize_map_device(device_map[key])
                if normalized is not None:
                    if normalized.type == "cuda":
                        return normalized
                    preferred_fallback = normalized

        fallback = None
        for mapped_device in device_map.values():
            normalized = _normalize_map_device(mapped_device)
            if normalized is not None:
                if normalized.type == "cuda":
                    return normalized
                if fallback is None:
                    fallback = normalized

        if preferred_fallback is not None:
            return preferred_fallback
        if fallback is not None:
            return fallback

    return next(model_obj.parameters()).device

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Get prompt from CLI
if len(sys.argv) < 2:
    print("Usage: python gemma4.py 'your prompt here' [max_new_tokens]")
    sys.exit(1)

user_prompt = sys.argv[1]
max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MAX_NEW_TOKENS

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_prompt},
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

inputs = processor(text=text, return_tensors="pt")
inputs_device = get_inputs_device(model)
inputs = {
    k: (v.to(inputs_device) if torch.is_tensor(v) else v)
    for k, v in inputs.items()
}
input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

print(processor.parse_response(response))