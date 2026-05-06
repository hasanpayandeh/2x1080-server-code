import os
import sys

# Reduce CUDA memory fragmentation in long-lived processes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
DEFAULT_MAX_NEW_TOKENS = 256


def build_max_memory_map(utilization=0.9):
    if not torch.cuda.is_available():
        return None

    max_memory = {}
    for idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        allowed_gib = max(1, int(total_gib * utilization))
        max_memory[idx] = f"{allowed_gib}GiB"

    max_memory["cpu"] = "48GiB"
    return max_memory


def get_inputs_device(model_obj):
    if hasattr(model_obj, "hf_device_map") and model_obj.hf_device_map:
        # Prefer a CUDA shard if present, then fall back to CPU.
        for mapped_device in model_obj.hf_device_map.values():
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            if isinstance(mapped_device, str) and mapped_device.startswith("cuda"):
                return torch.device(mapped_device)
        return torch.device("cpu")

    return next(model_obj.parameters()).device

processor = AutoProcessor.from_pretrained(MODEL_ID)
load_kwargs = {
    "torch_dtype": DTYPE,
    "low_cpu_mem_usage": True,
}

if torch.cuda.is_available():
    load_kwargs.update(
        {
            "device_map": "auto",
            "max_memory": build_max_memory_map(),
            "offload_folder": "offload",
            "offload_state_dict": True,
        }
    )

try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
except (ImportError, ValueError, RuntimeError) as exc:
    error_text = str(exc).lower()
    can_fallback = any(
        token in error_text
        for token in ("accelerate", "device_map", "offload", "max_memory")
    )
    if not can_fallback:
        raise

    # Fallback when auto-sharding/offload setup is unavailable.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

# Get prompt from CLI
if len(sys.argv) < 2:
    print("Usage: python gemma4.py 'your prompt here'")
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
    k: (v.to(inputs_device) if hasattr(v, "to") else v)
    for k, v in inputs.items()
}
input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

print(processor.parse_response(response))