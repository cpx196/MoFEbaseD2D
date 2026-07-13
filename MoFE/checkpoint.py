from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from .config import MoFEConfig
from .modeling import convert_gpt2_to_mofe


MODEL_STATE_NAME = "model_state.pt"
HF_CONFIG_NAME = "hf_config.json"


def save_mofe_checkpoint(
    model: GPT2LMHeadModel,
    output_dir: str | Path,
    mofe_config: MoFEConfig,
    tokenizer: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.to_json_file(output_dir / HF_CONFIG_NAME)
    mofe_config.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir / MODEL_STATE_NAME)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    if metadata is not None:
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    return output_dir


def load_mofe_checkpoint(
    checkpoint_dir: str | Path,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[GPT2LMHeadModel, MoFEConfig]:
    checkpoint_dir = Path(checkpoint_dir)
    hf_config = GPT2Config.from_json_file(checkpoint_dir / HF_CONFIG_NAME)
    model = GPT2LMHeadModel(hf_config)
    mofe_config = MoFEConfig.from_pretrained(checkpoint_dir)
    convert_gpt2_to_mofe(model, mofe_config)
    state_dict = torch.load(
        checkpoint_dir / MODEL_STATE_NAME,
        map_location=map_location,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    if dtype is not None:
        model.to(dtype=dtype)
    model.tie_weights()
    return model, mofe_config
