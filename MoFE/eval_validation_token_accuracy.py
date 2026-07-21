from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
)

from MoFE.eval_validation_loss import (
    checkpoint_step,
    load_checkpoint,
    load_validation_dataset,
)
from MoFE.modeling import set_private_scale
from MoFE.upcycling import UpcyclingConfig, convert_gpt2_to_upcycling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate next-token prediction accuracy on a fixed validation file."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--private-scale", type=float, default=1.0)
    return parser.parse_args()


def load_upcycling_accelerate_checkpoint(
    checkpoint: Path,
) -> tuple[GPT2LMHeadModel, UpcyclingConfig]:
    run_config = json.loads((checkpoint / "run_config.json").read_text())
    hf_config = GPT2Config.from_pretrained(run_config["model_name_or_path"])
    model = GPT2LMHeadModel(hf_config)
    upcycling_config = UpcyclingConfig.from_pretrained(checkpoint)
    convert_gpt2_to_upcycling(model, upcycling_config)

    state_dict = load_file(checkpoint / "model.safetensors", device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.weight"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint state does not match the Upcycling model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.tie_weights()
    return model, upcycling_config


def load_any_checkpoint(
    checkpoint: Path,
) -> tuple[GPT2LMHeadModel, str, dict | None]:
    if (checkpoint / "upcycling_config.json").exists() and (
        checkpoint / "model.safetensors"
    ).exists():
        model, upcycling_config = load_upcycling_accelerate_checkpoint(checkpoint)
        return model, "upcycling", upcycling_config.to_dict()

    model, mofe_config = load_checkpoint(checkpoint)
    if mofe_config is not None:
        return model, "mofe", mofe_config.to_dict()
    return model, "dense", None


@torch.inference_mode()
def evaluate(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.to(device)
    model.eval()
    autocast_enabled = device.type == "cuda"
    total_nll = 0.0
    correct_tokens = 0
    token_count = 0
    sample_count = 0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            outputs = model(**batch)

        labels = batch["labels"]
        target = labels[:, 1:]
        valid_mask = target != -100
        predictions = outputs.logits[:, :-1].argmax(dim=-1)
        valid_tokens = int(valid_mask.sum().item())
        correct_tokens += int((predictions.eq(target) & valid_mask).sum().item())
        token_count += valid_tokens
        sample_count += int(labels.shape[0])
        total_nll += float(outputs.loss) * valid_tokens

    if token_count == 0:
        raise ValueError("validation dataset contains no predicted tokens")
    loss = total_nll / token_count
    return {
        "sample_count": sample_count,
        "token_count": token_count,
        "correct_tokens": correct_tokens,
        "token_accuracy": correct_tokens / token_count,
        "negative_log_likelihood": loss,
        "perplexity": math.exp(loss),
    }


def main() -> None:
    args = parse_args()
    if args.block_size <= 0 or args.batch_size <= 0:
        raise ValueError("block size and batch size must be positive")
    if args.private_scale < 0:
        raise ValueError("private scale must be non-negative")

    checkpoint = Path(args.checkpoint).resolve()
    validation_file = Path(args.validation_file).resolve()
    output = Path(args.output).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_validation_dataset(args, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model, model_type, model_config = load_any_checkpoint(checkpoint)
    if model_type == "mofe":
        set_private_scale(model, args.private_scale)

    result = {
        "checkpoint": str(checkpoint),
        "step": checkpoint_step(checkpoint),
        "dataset": str(validation_file),
        "protocol": "fixed packed validation next-token prediction",
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "precision": "float32 weights with bfloat16 autocast",
        "model_type": model_type,
        "private_scale": args.private_scale if model_type == "mofe" else None,
        "shared_scale": 1.0,
        **evaluate(model, dataloader, device),
    }
    if model_config is not None:
        result[f"{model_type}_config"] = model_config

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    printable = {
        key: value
        for key, value in result.items()
        if not key.endswith("_config")
    }
    print(json.dumps(printable), flush=True)


if __name__ == "__main__":
    main()
