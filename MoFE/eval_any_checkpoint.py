from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from safetensors.torch import load_file
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from MoFE.eval_validation_loss import checkpoint_step, load_checkpoint
from MoFE.modeling import parameter_breakdown, set_private_scale
from MoFE.upcycling import (
    UpcyclingConfig,
    convert_gpt2_to_upcycling,
    upcycling_parameter_breakdown,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense or MoFE checkpoints, including Accelerate state dirs."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=100)
    parser.add_argument("--private-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint = Path(args.checkpoint).resolve()
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    upcycling_config = None
    if (checkpoint / "upcycling_config.json").exists() and (
        checkpoint / "model.safetensors"
    ).exists():
        model, upcycling_config = load_upcycling_accelerate_checkpoint(checkpoint)
        mofe_config = None
    else:
        model, mofe_config = load_checkpoint(checkpoint)
    model.to(dtype=dtype)
    if mofe_config is not None:
        set_private_scale(model, args.private_scale)
    model.to(args.device)
    model.eval()

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device=args.device,
        dtype=dtype,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=False,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results on the main process")

    results["checkpoint_evaluation"] = {
        "checkpoint": str(checkpoint),
        "training_steps": checkpoint_step(checkpoint),
        "model_type": (
            "upcycling"
            if upcycling_config is not None
            else "mofe"
            if mofe_config is not None
            else "dense"
        ),
        "precision": "bfloat16" if dtype is torch.bfloat16 else "float32",
        "private_scale": args.private_scale if mofe_config is not None else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if mofe_config is not None:
        results["checkpoint_evaluation"]["mofe_config"] = mofe_config.to_dict()
        results["checkpoint_evaluation"]["parameter_breakdown"] = parameter_breakdown(
            model
        )
    if upcycling_config is not None:
        results["checkpoint_evaluation"]["upcycling_config"] = (
            upcycling_config.to_dict()
        )
        results["checkpoint_evaluation"]["parameter_breakdown"] = (
            upcycling_parameter_breakdown(model)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            results,
            indent=2,
            default=handle_non_serializable,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"Saved results to {output}", flush=True)


if __name__ == "__main__":
    main()
