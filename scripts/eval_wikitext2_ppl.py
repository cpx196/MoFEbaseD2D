import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=os.environ.get("HF_HOME")
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=os.environ.get("HF_HOME"), dtype=dtype
    ).to(device)
    model.eval()

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        cache_dir=os.environ.get("HF_DATASETS_CACHE"),
    )
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    sequence_length = input_ids.size(1)

    total_nll = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = 0
    previous_end = 0

    for begin in range(0, sequence_length, args.stride):
        end = min(begin + args.max_length, sequence_length)
        target_length = end - previous_end
        if target_length <= 0:
            continue

        input_window = input_ids[:, begin:end]
        labels = input_window.clone()
        labels[:, :-target_length] = -100

        with torch.inference_mode():
            outputs = model(input_window, labels=labels)

        valid_labels = labels[:, 1:] != -100
        valid_tokens = int(valid_labels.sum().item())
        total_nll += outputs.loss.double() * valid_tokens
        total_tokens += valid_tokens

        previous_end = end
        if end == sequence_length:
            break

    nll = (total_nll / total_tokens).item()
    result = {
        "model": args.model,
        "dataset": f"wikitext/{args.dataset_config}",
        "source_dataset": args.dataset,
        "split": args.split,
        "max_length": args.max_length,
        "stride": args.stride,
        "precision": "bfloat16" if dtype is torch.bfloat16 else "float32",
        "device": device,
        "token_count": total_tokens,
        "negative_log_likelihood": nll,
        "perplexity": math.exp(nll),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
