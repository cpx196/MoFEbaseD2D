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

from MoFE.checkpoint import MODEL_STATE_NAME, load_mofe_checkpoint
from MoFE.config import MoFEConfig
from MoFE.modeling import convert_gpt2_to_mofe, set_private_scale
from MoFE.train import load_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate token-weighted loss on a fixed validation file."
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


def checkpoint_step(checkpoint: Path) -> int | None:
    if checkpoint.name.startswith("step_"):
        return int(checkpoint.name.removeprefix("step_"))
    metadata_path = checkpoint / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text()).get("global_step")
    return None


def load_accelerate_checkpoint(
    checkpoint: Path,
) -> tuple[GPT2LMHeadModel, MoFEConfig]:
    run_config = json.loads((checkpoint / "run_config.json").read_text())
    base_model = run_config["model_name_or_path"]
    hf_config = GPT2Config.from_pretrained(base_model)
    model = GPT2LMHeadModel(hf_config)
    mofe_config = MoFEConfig.from_pretrained(checkpoint)
    convert_gpt2_to_mofe(model, mofe_config)

    state_dict = load_file(checkpoint / "model.safetensors", device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.weight"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint state does not match the MoFE model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.tie_weights()
    return model, mofe_config


def load_dense_accelerate_checkpoint(checkpoint: Path) -> GPT2LMHeadModel:
    run_config = json.loads((checkpoint / "run_config.json").read_text())
    hf_config = GPT2Config.from_pretrained(run_config["model_name_or_path"])
    model = GPT2LMHeadModel(hf_config)
    state_dict = load_file(checkpoint / "model.safetensors", device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.weight"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint state does not match the dense model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.tie_weights()
    return model


def load_checkpoint(
    checkpoint: Path,
) -> tuple[GPT2LMHeadModel, MoFEConfig | None]:
    if (checkpoint / MODEL_STATE_NAME).exists():
        return load_mofe_checkpoint(checkpoint, map_location="cpu")
    if (checkpoint / "model.safetensors").exists() and (
        checkpoint / "mofe_config.json"
    ).exists():
        return load_accelerate_checkpoint(checkpoint)
    if (checkpoint / "model.safetensors").exists():
        return load_dense_accelerate_checkpoint(checkpoint), None
    raise FileNotFoundError(f"no supported model state found in {checkpoint}")


def load_validation_dataset(
    args: argparse.Namespace, tokenizer: AutoTokenizer
):
    dataset_args = argparse.Namespace(
        dataset_name=None,
        train_file=args.validation_file,
        dataset_config=None,
        train_split="train",
        text_column=args.text_column,
        block_size=args.block_size,
        streaming=False,
        shuffle_buffer_size=1,
        preprocessing_batch_size=64,
        seed=0,
    )
    return load_training_dataset(dataset_args, tokenizer)


@torch.inference_mode()
def evaluate(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    model.to(device)
    model.eval()
    total_nll = 0.0
    token_count = 0
    sample_count = 0
    autocast_enabled = device.type == "cuda"

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            outputs = model(**batch)
        valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss) * valid_tokens
        token_count += valid_tokens
        sample_count += int(batch["labels"].shape[0])

    if token_count == 0:
        raise ValueError("validation dataset contains no predicted tokens")
    loss = total_nll / token_count
    return {
        "sample_count": sample_count,
        "token_count": token_count,
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

    model, mofe_config = load_checkpoint(checkpoint)
    if mofe_config is not None:
        set_private_scale(model, args.private_scale)
    result = {
        "checkpoint": str(checkpoint),
        "step": checkpoint_step(checkpoint),
        "dataset": str(validation_file),
        "protocol": "fixed packed validation",
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "precision": "float32 weights with bfloat16 autocast",
        "model_type": "mofe" if mofe_config is not None else "dense",
        "private_scale": args.private_scale if mofe_config is not None else None,
        "shared_scale": 1.0,
        **evaluate(model, dataloader, device),
    }
    if mofe_config is not None:
        result["mofe_config"] = mofe_config.to_dict()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "mofe_config"}))


if __name__ == "__main__":
    main()
