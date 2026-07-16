import argparse
from contextlib import nullcontext
import json
import math
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument(
        "--mofe-checkpoint",
        help="Load a trained MoFE checkpoint instead of --model.",
    )
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--data-file",
        help="Evaluate a local text or JSON/JSONL file instead of --dataset.",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--private-scale", type=float, default=1.0)
    parser.add_argument("--shared-scale", type=float, default=1.0)
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

    model_source = args.mofe_checkpoint or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, cache_dir=os.environ.get("HF_HOME")
    )
    if args.mofe_checkpoint:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from MoFE.checkpoint import load_mofe_checkpoint
        from MoFE.modeling import set_private_scale, shared_scale

        model, mofe_config = load_mofe_checkpoint(
            args.mofe_checkpoint,
            map_location="cpu",
            dtype=dtype,
        )
        set_private_scale(model, args.private_scale)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, cache_dir=os.environ.get("HF_HOME"), dtype=dtype
        )
        mofe_config = None
    model.to(device)
    model.eval()

    if args.data_file:
        data_file = Path(args.data_file).resolve()
        loader = "json" if data_file.suffix in {".json", ".jsonl"} else "text"
        dataset = load_dataset(
            loader,
            data_files={"validation": str(data_file)},
            split="validation",
            cache_dir=os.environ.get("HF_DATASETS_CACHE"),
        )
        dataset_name = str(data_file)
        split = "validation"
    else:
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            cache_dir=os.environ.get("HF_DATASETS_CACHE"),
        )
        dataset_name = f"wikitext/{args.dataset_config}"
        split = args.split
    encodings = tokenizer(
        "\n\n".join(dataset[args.text_column]), return_tensors="pt"
    )
    input_ids = encodings.input_ids.to(device)
    sequence_length = input_ids.size(1)

    total_nll = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = 0
    previous_end = 0

    num_windows = math.ceil(max(sequence_length - args.max_length, 0) / args.stride) + 1
    scale_context = shared_scale(model, args.shared_scale) if args.mofe_checkpoint else nullcontext()
    with scale_context:
        for window_index, begin in enumerate(
            range(0, sequence_length, args.stride), start=1
        ):
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

            if args.logging_steps > 0 and (
                window_index % args.logging_steps == 0 or end == sequence_length
            ):
                running_nll = (total_nll / total_tokens).item()
                print(
                    f"window={window_index:04d}/{num_windows:04d} "
                    f"tokens={total_tokens} ppl={math.exp(running_nll):.4f}",
                    flush=True,
                )

            previous_end = end
            if end == sequence_length:
                break

    nll = (total_nll / total_tokens).item()
    result = {
        "model": str(model_source),
        "model_type": "mofe_checkpoint" if args.mofe_checkpoint else "dense",
        "dataset": dataset_name,
        "source_dataset": "local_file" if args.data_file else args.dataset,
        "split": split,
        "max_length": args.max_length,
        "stride": args.stride,
        "precision": "bfloat16" if dtype is torch.bfloat16 else "float32",
        "device": device,
        "private_scale": args.private_scale if args.mofe_checkpoint else None,
        "shared_scale": args.shared_scale if args.mofe_checkpoint else None,
        "token_count": total_tokens,
        "negative_log_likelihood": nll,
        "perplexity": math.exp(nll),
    }
    if mofe_config is not None:
        result["mofe_config"] = mofe_config.to_dict()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
