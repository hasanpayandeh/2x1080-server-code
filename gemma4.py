import sys
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)


def get_model_input_device(model):
    # Inputs must be on the token-embedding device for sharded models.
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass

    def normalize_device(device):
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        if isinstance(device, str):
            return torch.device(device)
        return device

    if hasattr(model, "hf_device_map"):
        preferred_keys = (
            "model.embed_tokens",
            "language_model.model.embed_tokens",
            "embed_tokens",
            "transformer.wte",
            "",
        )
        for key in preferred_keys:
            if key in model.hf_device_map:
                device = model.hf_device_map[key]
                if device not in ("cpu", "disk"):
                    return normalize_device(device)

        for device in model.hf_device_map.values():
            if device not in ("cpu", "disk"):
                return normalize_device(device)
    return next(model.parameters()).device

# Get prompt from CLI
if len(sys.argv) < 2:
    print("Usage: python gemma4.py 'your prompt here'")
    sys.exit(1)

user_prompt = sys.argv[1]

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

input_device = get_model_input_device(model)
inputs = processor(text=text, return_tensors="pt")
inputs = {
    k: (v.to(input_device) if hasattr(v, "to") else v)
    for k, v in inputs.items()
}
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

print(processor.parse_response(response))