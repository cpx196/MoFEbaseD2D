from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from transformers import AutoModelForCausalLM, AutoTokenizer

from MoFE.eval_checkpoint import DEFAULT_TASKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a dense HF checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint = Path(args.checkpoint).resolve()
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"Loading dense checkpoint: {checkpoint}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)
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

    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    results["dense_evaluation"] = {
        "experiment": "dense_trained_checkpoint",
        "checkpoint": str(checkpoint),
        "training_steps": metadata.get("global_step"),
        "precision": "bfloat16" if dtype is torch.bfloat16 else "float32",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
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
