from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from MoFE.config import MoFEConfig
from MoFE.modeling import convert_gpt2_to_mofe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile matched Dense and MoFE GPT-2 compute with CalFLOPs."
    )
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--mofe-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    parser.add_argument("--reference-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backward-factor",
        type=float,
        default=2.0,
        help="Backward FLOPs as a multiple of forward FLOPs (CalFLOPs default: 2).",
    )
    return parser.parse_args()


def find_config(checkpoint: Path, filenames: tuple[str, ...]) -> Path:
    run_dir = checkpoint.parent if checkpoint.name.startswith("step_") or checkpoint.name == "final" else checkpoint
    directories = (checkpoint, run_dir / "final", run_dir)
    for directory in directories:
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"none of {filenames} found for checkpoint {checkpoint}"
    )


def build_model(checkpoint: Path, use_mofe: bool) -> tuple[GPT2LMHeadModel, MoFEConfig | None]:
    config_path = find_config(checkpoint, ("config.json", "hf_config.json"))
    hf_config = GPT2Config.from_json_file(config_path)
    hf_config.use_cache = False
    # CalFLOPs 0.3.2 cannot see matmuls inside PyTorch's fused SDPA kernel.
    # Eager attention performs the same QK^T and AV operations through hookable calls.
    hf_config._attn_implementation = "eager"
    model = GPT2LMHeadModel(hf_config)
    if not use_mofe:
        return model, None

    mofe_path = find_config(checkpoint, ("mofe_config.json",))
    mofe_config = MoFEConfig.from_json_file(mofe_path)
    convert_gpt2_to_mofe(model, mofe_config)
    return model, mofe_config


def active_private_forward_flops(
    config: GPT2Config,
    mofe_config: MoFEConfig,
    batch_size: int,
    sequence_length: int,
) -> dict[str, int]:
    hidden_size = config.n_embd
    intermediate_size = config.n_inner or 4 * hidden_size
    rank = mofe_config.resolve_rank(hidden_size)
    tokens = batch_size * sequence_length
    layer_count = len(mofe_config.moe_layer_indices)

    # One multiply and one add per matrix MAC. Bias additions are left to CalFLOPs.
    private_per_route = 4 * rank * (hidden_size + intermediate_size + rank)
    private_paths = (
        tokens * layer_count * mofe_config.top_k * private_per_route
    )
    routers = (
        tokens
        * layer_count
        * 2
        * hidden_size
        * mofe_config.num_private_experts
    )
    return {
        "private_paths": private_paths,
        "routers": routers,
        "total": private_paths + routers,
    }


@torch.inference_mode()
def profile_model(
    model: GPT2LMHeadModel,
    sequence_lengths: list[int],
    batch_size: int,
    device: torch.device,
    backward_factor: float,
) -> list[dict[str, int | float]]:
    try:
        from calflops import calculate_flops
    except ImportError as error:
        raise RuntimeError("install CalFLOPs with: pip install calflops") from error

    model.to(device)
    model.eval()
    profiles = []
    for sequence_length in sequence_lengths:
        input_ids = torch.zeros(
            (batch_size, sequence_length), dtype=torch.long, device=device
        )
        forward_flops, forward_macs, parameters = calculate_flops(
            model=model,
            kwargs={"input_ids": input_ids},
            include_backPropagation=False,
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
        profiles.append(
            {
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "tokens": batch_size * sequence_length,
                "forward_flops": int(forward_flops),
                "forward_macs": int(forward_macs),
                "estimated_training_flops": int(
                    forward_flops * (1.0 + backward_factor)
                ),
                "parameters": int(parameters),
            }
        )
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return profiles


def main() -> None:
    args = parse_args()
    if any(length <= 0 for length in args.sequence_lengths):
        raise ValueError("sequence lengths must be positive")
    if args.batch_size <= 0 or args.tokens_per_step <= 0 or args.reference_steps <= 0:
        raise ValueError("batch size, tokens per step, and reference steps must be positive")
    if args.backward_factor < 0:
        raise ValueError("backward factor must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoints = {
        "dense": Path(args.dense_checkpoint).resolve(),
        "mofe": Path(args.mofe_checkpoint).resolve(),
    }
    dense_model, _ = build_model(checkpoints["dense"], use_mofe=False)
    mofe_model, mofe_config = build_model(checkpoints["mofe"], use_mofe=True)
    assert mofe_config is not None

    profiles = {
        "dense": profile_model(
            dense_model,
            args.sequence_lengths,
            args.batch_size,
            device,
            args.backward_factor,
        ),
        "mofe": profile_model(
            mofe_model,
            args.sequence_lengths,
            args.batch_size,
            device,
            args.backward_factor,
        ),
    }

    comparisons = []
    for dense, mofe in zip(profiles["dense"], profiles["mofe"], strict=True):
        sequence_length = int(dense["sequence_length"])
        analytic = active_private_forward_flops(
            dense_model.config, mofe_config, args.batch_size, sequence_length
        )
        observed_increment = int(mofe["forward_flops"]) - int(dense["forward_flops"])
        comparisons.append(
            {
                "sequence_length": sequence_length,
                "mofe_to_dense_forward_flops": (
                    float(mofe["forward_flops"]) / float(dense["forward_flops"])
                ),
                "observed_mofe_increment": observed_increment,
                "analytic_active_private_and_router_increment": analytic,
                "analytic_relative_error": (
                    analytic["total"] / observed_increment - 1.0
                ),
            }
        )

    target_length = max(args.sequence_lengths)
    target_index = args.sequence_lengths.index(target_length)
    dense_target = profiles["dense"][target_index]
    mofe_target = profiles["mofe"][target_index]
    tokens_per_profile = args.batch_size * target_length
    if args.tokens_per_step % tokens_per_profile != 0:
        raise ValueError(
            "tokens per step must be divisible by batch_size * maximum sequence length"
        )
    profile_batches_per_step = args.tokens_per_step // tokens_per_profile
    dense_step_flops = int(dense_target["estimated_training_flops"]) * profile_batches_per_step
    mofe_step_flops = int(mofe_target["estimated_training_flops"]) * profile_batches_per_step
    ratio = mofe_step_flops / dense_step_flops
    compute_matching = {
        "sequence_length": target_length,
        "tokens_per_step": args.tokens_per_step,
        "backward_factor": args.backward_factor,
        "dense_estimated_training_flops_per_step": dense_step_flops,
        "mofe_estimated_training_flops_per_step": mofe_step_flops,
        "mofe_to_dense_training_flops": ratio,
        "reference_steps": args.reference_steps,
        "dense_steps_equal_to_reference_mofe": args.reference_steps * ratio,
        "mofe_steps_equal_to_reference_dense": args.reference_steps / ratio,
        "dense_total_flops_at_reference_steps": dense_step_flops * args.reference_steps,
        "mofe_total_flops_at_reference_steps": mofe_step_flops * args.reference_steps,
    }

    result = {
        "tool": "calflops.calculate_flops",
        "attention_profile_implementation": "eager",
        "attention_profile_reason": (
            "CalFLOPs 0.3.2 does not count matmuls inside fused SDPA; eager exposes "
            "the equivalent QK^T and attention-value operations"
        ),
        "calflops_training_estimate": "forward FLOPs * (1 + backward_factor)",
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "profiles": profiles,
        "comparisons": comparisons,
        "compute_matching": compute_matching,
        "mofe_config": mofe_config.to_dict(),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(compute_matching, indent=2))
    print(f"Saved profile: {output}")


if __name__ == "__main__":
    main()
