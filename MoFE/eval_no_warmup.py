from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from transformers import AutoModelForCausalLM, AutoTokenizer

from MoFE.config import MoFEConfig
from MoFE.modeling import convert_gpt2_to_mofe, parameter_breakdown, set_private_scale


DEFAULT_TASKS = (
    "lambada_openai",
    "hellaswag",
    "piqa",
    "winogrande",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate initialized MoFE with private_scale=1 and zero training steps."
    )
    parser.add_argument(
        "--mofe-config",
        default=str(Path(__file__).parent / "configs" / "mofe_gpt2_last3_e16_k3.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available")
    config = MoFEConfig.from_json_file(args.mofe_config)
    cache_dir = os.environ.get("HF_HOME")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    convert_gpt2_to_mofe(model, config)
    set_private_scale(model, 1.0)
    model.to(args.device)
    model.eval()

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
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
    results["mofe_evaluation"] = {
        "experiment": "mofe_no_warmup_initialization",
        "training_steps": 0,
        "private_scale": 1.0,
        "warmup_applied": False,
        "mofe_config": config.to_dict(),
        "parameter_breakdown": parameter_breakdown(model),
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
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
