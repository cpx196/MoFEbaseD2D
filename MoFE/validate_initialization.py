from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from MoFE.config import MoFEConfig
from MoFE.modeling import (
    convert_gpt2_to_mofe,
    iter_mofe_layers,
    parameter_breakdown,
    private_scale,
)


def tensor_summary(tensor: torch.Tensor, requires_grad: bool) -> dict[str, object]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "norm": values.norm().item(),
        "requires_grad": requires_grad,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default="openai-community/gpt2")
    parser.add_argument(
        "--mofe-config",
        default=str(Path(__file__).parent / "configs" / "mofe_gpt2_last3_e16_k3.json"),
    )
    parser.add_argument("--prompt", default="The MoFE initialization test")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = os.environ.get("HF_HOME")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, cache_dir=cache_dir, dtype=torch.float32
    )
    model.eval()
    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.inference_mode():
        dense_logits = model(**inputs).logits

    source_mlps = {
        index: {
            name: value.detach().clone()
            for name, value in model.transformer.h[index].mlp.state_dict().items()
        }
        for index in MoFEConfig.from_json_file(args.mofe_config).moe_layer_indices
    }
    config = MoFEConfig.from_json_file(args.mofe_config)
    convert_gpt2_to_mofe(model, config)
    model.eval()
    with private_scale(model, 0.0), torch.inference_mode():
        shared_only_logits = model(**inputs).logits
    with torch.inference_mode():
        full_mofe_logits = model(**inputs).logits

    report: dict[str, object] = {
        "config": config.to_dict(),
        "parameter_breakdown": parameter_breakdown(model),
        "dense_vs_shared_only_logits_max_abs_diff": (
            dense_logits - shared_only_logits
        ).abs().max().item(),
        "dense_vs_full_mofe_logits_max_abs_diff": (
            dense_logits - full_mofe_logits
        ).abs().max().item(),
        "layers": {},
    }
    for name, layer in iter_mofe_layers(model):
        layer_index = int(name.split(".")[2])
        dense_state = source_mlps[layer_index]
        shared_diff = max(
            (
                value.detach().cpu() - dense_state[key].cpu()
            ).abs().max().item()
            for key, value in layer.shared_expert.state_dict().items()
        )
        dense_w1 = dense_state["c_fc.weight"].T
        dense_w2 = dense_state["c_proj.weight"].T
        layer_report = {
            "shared_max_abs_diff": shared_diff,
            "a1_slice_max_abs_diff": (
                layer.a1.detach().cpu() - dense_w1[:, : layer.rank].unsqueeze(0)
            ).abs().max().item(),
            "b1_slice_max_abs_diff": (
                layer.b1.detach().cpu() - dense_w1[: layer.rank, :].unsqueeze(0)
            ).abs().max().item(),
            "a2_slice_max_abs_diff": (
                layer.a2.detach().cpu() - dense_w2[:, : layer.rank].unsqueeze(0)
            ).abs().max().item(),
            "b2_slice_max_abs_diff": (
                layer.b2.detach().cpu() - dense_w2[: layer.rank, :].unsqueeze(0)
            ).abs().max().item(),
            "parameters": {
                parameter_name: tensor_summary(parameter, parameter.requires_grad)
                for parameter_name, parameter in layer.named_parameters()
            },
        }
        report["layers"][name] = layer_report

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
