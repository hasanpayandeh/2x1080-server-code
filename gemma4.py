import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_MAX_NEW_TOKENS = 1024

 
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
input_device = get_input_device(model)
inputs = {
    k: (v.to(input_device) if hasattr(v, "to") else v)
    for k, v in inputs.items()
}
input_len = inputs["input_ids"].shape[-1]

generate_kwargs = {"max_new_tokens": max_new_tokens}
if torch.distributed.is_available() and torch.distributed.is_initialized():
    generate_kwargs["synced_gpus"] = True

with torch.inference_mode():
    outputs = model.generate(**inputs, **generate_kwargs)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

print(processor.parse_response(response))