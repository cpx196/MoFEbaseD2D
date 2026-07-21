import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MoFE.checkpoint import load_mofe_checkpoint
from MoFE.modeling import set_private_scale, shared_scale
from MoFE.train import load_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--private-scale", type=float, default=1.0)
    parser.add_argument("--shared-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    dataset_args = argparse.Namespace(
        dataset_name=None,
        dataset_config=None,
        train_split="validation",
        train_file=args.validation_file,
        text_column="text",
        block_size=args.block_size,
    )
    dataset = load_training_dataset(dataset_args, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    model, mofe_config = load_mofe_checkpoint(args.checkpoint)
    set_private_scale(model, args.private_scale)
    model.to(device).eval()
    total_nll = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens = 0
    sample_count = 0

    with shared_scale(model, args.shared_scale), torch.inference_mode():
        for batch_index, batch in enumerate(dataloader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch["input_ids"]
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                outputs = model(**batch, labels=labels)
            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            total_nll += outputs.loss.double() * valid_tokens
            total_tokens += valid_tokens
            sample_count += labels.shape[0]
            if args.logging_steps > 0 and (
                batch_index % args.logging_steps == 0 or batch_index == len(dataloader)
            ):
                running_loss = (total_nll / total_tokens).item()
                print(
                    f"batch={batch_index:04d}/{len(dataloader):04d} "
                    f"tokens={total_tokens} loss={running_loss:.6f} "
                    f"ppl={math.exp(running_loss):.4f}",
                    flush=True,
                )

    loss = (total_nll / total_tokens).item()
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": str(Path(args.validation_file).resolve()),
        "protocol": "training-matched non-overlapping packed validation",
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "precision": "float32 weights with bfloat16 autocast" if use_bfloat16 else "float32",
        "private_scale": args.private_scale,
        "shared_scale": args.shared_scale,
        "sample_count": sample_count,
        "token_count": total_tokens,
        "negative_log_likelihood": loss,
        "perplexity": math.exp(loss),
        "mofe_config": mofe_config.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
