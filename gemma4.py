import sys

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)


def is_cuda_oom(error):
    return "out of memory" in str(error).lower()


def move_inputs_to_device(inputs, device):
    return {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }


def load_model_on_device(device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)


def load_model_with_fallback():
    if not torch.cuda.is_available():
        cpu_device = torch.device("cpu")
        print("CUDA not available, using CPU.", file=sys.stderr)
        return load_model_on_device(cpu_device), cpu_device

    candidate_gpu_ids = [0]
    if torch.cuda.device_count() > 1:
        candidate_gpu_ids.append(1)

    last_oom_error = None
    for gpu_id in candidate_gpu_ids:
        device = torch.device(f"cuda:{gpu_id}")
        try:
            model = load_model_on_device(device)
            print(f"Loaded model on {device}.", file=sys.stderr)
            return model, device
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if isinstance(error, torch.cuda.OutOfMemoryError) or is_cuda_oom(error):
                last_oom_error = error
                print(
                    f"{device} out of memory while loading model, trying next GPU...",
                    file=sys.stderr,
                )
                torch.cuda.empty_cache()
                continue
            raise

    raise RuntimeError(
        "Model loading failed due to GPU memory limits on available GPUs."
    ) from last_oom_error


def generate_with_fallback(model, device, inputs, max_new_tokens):
    try:
        return model.generate(**inputs, max_new_tokens=max_new_tokens), model, device
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        can_retry_on_gpu1 = (
            device.type == "cuda"
            and device.index == 0
            and torch.cuda.device_count() > 1
            and (isinstance(error, torch.cuda.OutOfMemoryError) or is_cuda_oom(error))
        )
        if not can_retry_on_gpu1:
            raise

        print("cuda:0 out of memory during generation, retrying on cuda:1.", file=sys.stderr)
        del model
        torch.cuda.empty_cache()

        retry_device = torch.device("cuda:1")
        retry_model = load_model_on_device(retry_device)
        retry_inputs = move_inputs_to_device(inputs, retry_device)
        outputs = retry_model.generate(**retry_inputs, max_new_tokens=max_new_tokens)
        return outputs, retry_model, retry_device


model, device = load_model_with_fallback()

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

inputs = processor(text=text, return_tensors="pt")
inputs = move_inputs_to_device(inputs, device)
input_len = inputs["input_ids"].shape[-1]

outputs, model, device = generate_with_fallback(
    model,
    device,
    inputs,
    max_new_tokens=1024,
)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

print(processor.parse_response(response))