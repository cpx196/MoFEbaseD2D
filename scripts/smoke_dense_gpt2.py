import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_id = "openai-community/gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=os.environ.get("HF_HOME")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=os.environ.get("HF_HOME"), dtype=dtype
    ).to(device)
    model.eval()

    prompt = "The dense GPT-2 baseline is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("model:", model_id)
    print("device:", device)
    print("dtype:", dtype)
    print("parameter_count:", sum(p.numel() for p in model.parameters()))
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
