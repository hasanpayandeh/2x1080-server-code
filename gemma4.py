import sys
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)


def get_model_input_device(model):
    # For sharded models, use the first non-CPU shard as the input device.
    if hasattr(model, "hf_device_map"):
        for device in model.hf_device_map.values():
            if device not in ("cpu", "disk"):
                if isinstance(device, int):
                    return f"cuda:{device}"
                return device
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
inputs = {k: v.to(input_device) for k, v in inputs.items()}
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

print(processor.parse_response(response))