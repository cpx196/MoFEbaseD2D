from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn

from .config import MoFEConfig
from .layer import MoFEGPT2MLP


def iter_mofe_layers(model: nn.Module) -> Iterator[tuple[str, MoFEGPT2MLP]]:
    for name, module in model.named_modules():
        if isinstance(module, MoFEGPT2MLP):
            yield name, module


def convert_gpt2_to_mofe(model: nn.Module, config: MoFEConfig) -> nn.Module:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise TypeError("MoFE conversion requires a Hugging Face GPT-2 model")
    config.validate(model.config.n_embd)
    num_layers = len(model.transformer.h)
    invalid = [index for index in config.moe_layer_indices if index >= num_layers]
    if invalid:
        raise ValueError(f"MoFE layer indices out of range for {num_layers} layers: {invalid}")

    for layer_index in config.moe_layer_indices:
        dense_mlp = model.transformer.h[layer_index].mlp
        if isinstance(dense_mlp, MoFEGPT2MLP):
            raise ValueError(f"block {layer_index} is already a MoFE layer")
        devices = [dense_mlp.c_fc.weight.device.index] if dense_mlp.c_fc.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed + layer_index)
            if dense_mlp.c_fc.weight.is_cuda:
                torch.cuda.manual_seed_all(config.seed + layer_index)
            model.transformer.h[layer_index].mlp = MoFEGPT2MLP(dense_mlp, config)

    model.mofe_config = config.to_dict()
    return model


def set_private_scale(model: nn.Module, value: float) -> None:
    layers = list(iter_mofe_layers(model))
    if not layers:
        raise ValueError("model has no MoFE layers")
    for _, layer in layers:
        layer.set_private_scale(value)


def collect_mofe_losses(model: nn.Module) -> dict[str, Tensor]:
    states = []
    for _, layer in iter_mofe_layers(model):
        if layer.routing_state is None:
            raise RuntimeError("MoFE losses requested before a complete forward pass")
        states.append(layer.routing_state)
    if not states:
        raise ValueError("model has no MoFE layers")
    return {
        "balance_loss": torch.stack([state.balance_loss for state in states]).mean(),
        "z_loss": torch.stack([state.z_loss for state in states]).mean(),
    }


def routing_statistics(model: nn.Module) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, layer in iter_mofe_layers(model):
        state = layer.routing_state
        if state is None:
            continue
        counts = state.assignment_counts.float()
        mean = counts.mean()
        nonzero = counts[counts > 0]
        min_nonzero = nonzero.min() if nonzero.numel() else counts.new_tensor(0.0)
        result[name] = {
            "assignment_counts": state.assignment_counts.cpu().tolist(),
            "assignment_fractions": (counts / counts.sum().clamp_min(1.0)).cpu().tolist(),
            "load_cv": (counts.std(unbiased=False) / mean.clamp_min(1.0)).item(),
            "max_min_load_ratio": (
                counts.max() / min_nonzero if min_nonzero > 0 else counts.new_tensor(float("inf"))
            ).item(),
            "router_entropy": state.router_entropy.item(),
            "unused_experts": int((counts == 0).sum().item()),
            "shared_output_norm": (
                state.shared_output_norm.item()
                if state.shared_output_norm is not None
                else None
            ),
            "private_output_norm": (
                state.private_output_norm.item()
                if state.private_output_norm is not None
                else None
            ),
            "private_to_shared_norm": (
                state.private_to_shared_norm.item()
                if state.private_to_shared_norm is not None
                else None
            ),
        }
    return result


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    categories = {
        "shared_experts": 0,
        "factor_banks": 0,
        "cores": 0,
        "private_biases": 0,
        "routers": 0,
    }
    for _, layer in iter_mofe_layers(model):
        categories["shared_experts"] += sum(
            parameter.numel() for parameter in layer.shared_expert.parameters()
        )
        categories["factor_banks"] += sum(
            parameter.numel() for parameter in (layer.a1, layer.b1, layer.a2, layer.b2)
        )
        categories["cores"] += layer.core1.numel() + layer.core2.numel()
        categories["private_biases"] += (
            layer.private_bias1.numel() + layer.private_bias2.numel()
        )
        categories["routers"] += sum(
            parameter.numel() for parameter in layer.router.parameters()
        )

    total = sum(parameter.numel() for parameter in model.parameters())
    mofe_total = sum(categories.values())
    categories["remaining_backbone"] = total - mofe_total
    categories["total"] = total
    categories["trainable"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return categories


@contextlib.contextmanager
def private_scale(model: nn.Module, value: float) -> Iterator[None]:
    previous = [layer.private_output_scale.item() for _, layer in iter_mofe_layers(model)]
    set_private_scale(model, value)
    try:
        yield
    finally:
        for (_, layer), old_value in zip(iter_mofe_layers(model), previous):
            layer.set_private_scale(old_value)
