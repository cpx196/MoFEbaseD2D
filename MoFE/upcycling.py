from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, GPT2LMHeadModel


CONFIG_NAME = "upcycling_config.json"
MODEL_STATE_NAME = "model_state.pt"
HF_CONFIG_NAME = "hf_config.json"


@dataclass
class UpcyclingConfig:
    model_name_or_path: str = "openai-community/gpt2"
    moe_layer_indices: tuple[int, ...] = (9, 10, 11)
    num_experts: int = 16
    top_k: int = 3
    router_init_std: float = 0.02
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    seed: int = 42

    def validate(self) -> None:
        if not self.moe_layer_indices:
            raise ValueError("moe_layer_indices must not be empty")
        if len(set(self.moe_layer_indices)) != len(self.moe_layer_indices):
            raise ValueError("moe_layer_indices must be unique")
        if min(self.moe_layer_indices) < 0:
            raise ValueError("moe_layer_indices must be non-negative")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if self.router_init_std <= 0:
            raise ValueError("router_init_std must be positive")
        if self.router_aux_loss_coef < 0 or self.router_z_loss_coef < 0:
            raise ValueError("router loss coefficients must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["moe_layer_indices"] = list(self.moe_layer_indices)
        data.update(
            {
                "moe_type": "sparse_upcycling",
                "routing": "token_choice",
                "expert_init": "exact_dense_ffn_copy",
                "shared_expert": False,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UpcyclingConfig":
        values = {
            key: value for key, value in data.items() if key in cls.__dataclass_fields__
        }
        if "moe_layer_indices" in values:
            values["moe_layer_indices"] = tuple(values["moe_layer_indices"])
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json_file(cls, path: str | Path) -> "UpcyclingConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / CONFIG_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "UpcyclingConfig":
        return cls.from_json_file(Path(directory) / CONFIG_NAME)


@dataclass
class UpcyclingRoutingState:
    balance_loss: Tensor
    z_loss: Tensor
    assignment_counts: Tensor
    router_entropy: Tensor
    topk_indices: Tensor


class UpcycledGPT2MLP(nn.Module):
    """GPT-2 MLP replaced by routed, exact copies of the original dense MLP."""

    def __init__(self, dense_mlp: nn.Module, config: UpcyclingConfig):
        super().__init__()
        config.validate()
        hidden_size, _ = dense_mlp.c_fc.weight.shape
        self.hidden_size = hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.experts = nn.ModuleList(
            [copy.deepcopy(dense_mlp) for _ in range(self.num_experts)]
        )
        reference = dense_mlp.c_fc.weight
        self.router = nn.Linear(hidden_size, self.num_experts, bias=True)
        self.router.to(device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            nn.init.normal_(
                self.router.weight, mean=0.0, std=config.router_init_std
            )
            nn.init.zeros_(self.router.bias)
        self.routing_state: UpcyclingRoutingState | None = None

    def _route(self, states: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.router(states)
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        # FP32 normalized weights and accumulation preserve the copied dense
        # function under BF16; direct BF16 summation compounds across MoE layers.
        topk_weights = F.softmax(topk_logits.float(), dim=-1)

        probabilities = F.softmax(logits, dim=-1)
        probability_fraction = probabilities.mean(dim=0)
        assignment_counts = torch.bincount(
            topk_indices.reshape(-1), minlength=self.num_experts
        )
        assignment_fraction = assignment_counts.to(logits.dtype) / (
            states.shape[0] * self.top_k
        )
        balance_loss = self.num_experts * torch.sum(
            probability_fraction * assignment_fraction
        )
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-9)), dim=-1
        ).mean()
        self.routing_state = UpcyclingRoutingState(
            balance_loss=balance_loss,
            z_loss=z_loss,
            assignment_counts=assignment_counts.detach(),
            router_entropy=entropy.detach(),
            topk_indices=topk_indices.detach(),
        )
        return topk_weights, topk_indices

    def forward(self, hidden_states: Tensor) -> Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        topk_weights, topk_indices = self._route(flat_states)
        output = torch.zeros_like(flat_states, dtype=torch.float32)
        for expert_index, expert in enumerate(self.experts):
            positions = (topk_indices == expert_index).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_indices = positions[:, 0]
            route_slots = positions[:, 1]
            selected_states = flat_states.index_select(0, token_indices)
            expert_output = expert(selected_states)
            weights = topk_weights[token_indices, route_slots].unsqueeze(-1)
            output = output.index_add(
                0, token_indices, expert_output.float() * weights
            )
        return output.to(dtype=hidden_states.dtype).reshape(original_shape)


def iter_upcycling_layers(
    model: nn.Module,
) -> Iterator[tuple[str, UpcycledGPT2MLP]]:
    for name, module in model.named_modules():
        if isinstance(module, UpcycledGPT2MLP):
            yield name, module


def convert_gpt2_to_upcycling(
    model: nn.Module, config: UpcyclingConfig
) -> nn.Module:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise TypeError("Upcycling conversion requires a Hugging Face GPT-2 model")
    config.validate()
    num_layers = len(model.transformer.h)
    invalid = [index for index in config.moe_layer_indices if index >= num_layers]
    if invalid:
        raise ValueError(
            f"Upcycling layer indices out of range for {num_layers} layers: {invalid}"
        )
    for layer_index in config.moe_layer_indices:
        dense_mlp = model.transformer.h[layer_index].mlp
        if isinstance(dense_mlp, UpcycledGPT2MLP):
            raise ValueError(f"block {layer_index} is already an Upcycling layer")
        devices = [dense_mlp.c_fc.weight.device.index] if dense_mlp.c_fc.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.seed + layer_index)
            if dense_mlp.c_fc.weight.is_cuda:
                torch.cuda.manual_seed_all(config.seed + layer_index)
            model.transformer.h[layer_index].mlp = UpcycledGPT2MLP(
                dense_mlp, config
            )
    model.upcycling_config = config.to_dict()
    return model


def collect_upcycling_losses(model: nn.Module) -> dict[str, Tensor]:
    states = []
    for _, layer in iter_upcycling_layers(model):
        if layer.routing_state is None:
            raise RuntimeError("Upcycling losses requested before a forward pass")
        states.append(layer.routing_state)
    if not states:
        raise ValueError("model has no Upcycling layers")
    return {
        "balance_loss": torch.stack([state.balance_loss for state in states]).mean(),
        "z_loss": torch.stack([state.z_loss for state in states]).mean(),
    }


def upcycling_parameter_breakdown(model: nn.Module) -> dict[str, int]:
    experts = 0
    routers = 0
    for _, layer in iter_upcycling_layers(model):
        experts += sum(parameter.numel() for parameter in layer.experts.parameters())
        routers += sum(parameter.numel() for parameter in layer.router.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "experts": experts,
        "routers": routers,
        "remaining_backbone": total - experts - routers,
        "total": total,
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def save_upcycling_checkpoint(
    model: GPT2LMHeadModel,
    output_dir: str | Path,
    config: UpcyclingConfig,
    tokenizer: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.to_json_file(output_dir / HF_CONFIG_NAME)
    config.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir / MODEL_STATE_NAME)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if metadata is not None:
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    return output_dir


def load_upcycling_checkpoint(
    checkpoint_dir: str | Path,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[GPT2LMHeadModel, UpcyclingConfig]:
    checkpoint_dir = Path(checkpoint_dir)
    hf_config = GPT2Config.from_json_file(checkpoint_dir / HF_CONFIG_NAME)
    model = GPT2LMHeadModel(hf_config)
    config = UpcyclingConfig.from_pretrained(checkpoint_dir)
    convert_gpt2_to_upcycling(model, config)
    state_dict = torch.load(
        checkpoint_dir / MODEL_STATE_NAME,
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    if dtype is not None:
        model.to(dtype=dtype)
    model.tie_weights()
    return model, config
