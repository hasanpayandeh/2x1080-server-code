import sys
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)

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

inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

print(processor.parse_response(response))