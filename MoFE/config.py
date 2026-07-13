from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONFIG_NAME = "mofe_config.json"


@dataclass
class MoFEConfig:
    model_name_or_path: str = "openai-community/gpt2"
    moe_layer_indices: tuple[int, ...] = (9, 10, 11)
    num_private_experts: int = 16
    top_k: int = 3
    low_rank_ratio: float = 0.75
    rank: int | None = 576
    core_init_std: float = 0.025
    router_init_std: float = 0.02
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    private_output_scale: float = 1.0
    train_factors: bool = True
    train_cores: bool = True
    train_private_bias: bool = True
    train_router: bool = True
    train_shared_expert: bool = True
    seed: int = 42

    def validate(self, hidden_size: int | None = None) -> None:
        if not self.moe_layer_indices:
            raise ValueError("moe_layer_indices must not be empty")
        if len(set(self.moe_layer_indices)) != len(self.moe_layer_indices):
            raise ValueError("moe_layer_indices must be unique")
        if min(self.moe_layer_indices) < 0:
            raise ValueError("moe_layer_indices must be non-negative")
        if self.num_private_experts <= 0:
            raise ValueError("num_private_experts must be positive")
        groups = math.isqrt(self.num_private_experts)
        if groups * groups != self.num_private_experts:
            raise ValueError(
                "num_private_experts must be a perfect square for Cartesian A/B sharing"
            )
        if not 1 <= self.top_k <= self.num_private_experts:
            raise ValueError("top_k must be between 1 and num_private_experts")
        if self.rank is not None and self.rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 < self.low_rank_ratio <= 1.0:
            raise ValueError("low_rank_ratio must be in (0, 1]")
        if self.core_init_std <= 0.0 or self.router_init_std <= 0.0:
            raise ValueError("initialization standard deviations must be positive")
        if self.private_output_scale < 0.0:
            raise ValueError("private_output_scale must be non-negative")
        if hidden_size is not None:
            rank = self.resolve_rank(hidden_size)
            if rank > hidden_size:
                raise ValueError(f"rank {rank} exceeds hidden size {hidden_size}")

    def resolve_rank(self, hidden_size: int) -> int:
        return self.rank if self.rank is not None else int(self.low_rank_ratio * hidden_size)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["moe_layer_indices"] = list(self.moe_layer_indices)
        groups = math.isqrt(self.num_private_experts)
        data.update(
            {
                "moe_type": "mofe",
                "routing": "token_choice",
                "shared_expert": True,
                "shared_expert_in_router": False,
                "factor_sharing": f"cartesian_{groups}x{groups}",
                "factor_init": "dense_row_column_slice",
                "core_init": "normal",
                "private_bias_init": "zeros",
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoFEConfig":
        field_names = cls.__dataclass_fields__.keys()
        values = {key: value for key, value in data.items() if key in field_names}
        if "moe_layer_indices" in values:
            values["moe_layer_indices"] = tuple(values["moe_layer_indices"])
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_json_file(cls, path: str | Path) -> "MoFEConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / CONFIG_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "MoFEConfig":
        return cls.from_json_file(Path(directory) / CONFIG_NAME)
