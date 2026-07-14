from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

from MoFE.checkpoint import save_mofe_checkpoint
from MoFE.config import MoFEConfig
from MoFE.modeling import (
    collect_mofe_losses,
    convert_gpt2_to_mofe,
    iter_mofe_layers,
    parameter_breakdown,
    set_private_scale,
)


def parse_args() -> argparse.Namespace:
    data_root = Path(os.environ.get("DATA_ROOT", "./data/pxchen"))
    parser = argparse.ArgumentParser(
        description="Train GPT-2 MoFE for a short, dataset-agnostic 200-step run."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-name", help="Hugging Face dataset repository name")
    source.add_argument("--train-file", help="Local .txt, .json, or .jsonl training file")
    parser.add_argument("--dataset-config")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--model-name-or-path", default="openai-community/gpt2")
    parser.add_argument(
        "--mofe-config",
        default=str(Path(__file__).parent / "configs" / "mofe_gpt2_last3_e16_k3.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(data_root / "checkpoints" / "mofe_gpt2_last3_e16_k3"),
    )
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--private-warmup-steps",
        type=int,
        default=200,
        help="Linearly increase the private branch scale from 0 to its configured value.",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_training_dataset(args: argparse.Namespace, tokenizer: Any) -> Dataset:
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if args.dataset_name:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.train_split,
            cache_dir=cache_dir,
        )
    else:
        suffix = Path(args.train_file).suffix.lower()
        loader = "text" if suffix == ".txt" else "json"
        dataset = load_dataset(
            loader,
            data_files={"train": args.train_file},
            split="train",
            cache_dir=cache_dir,
        )
    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"text column {args.text_column!r} is absent; available columns: "
            f"{dataset.column_names}"
        )

    remove_columns = dataset.column_names

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        texts = [str(value) for value in batch[args.text_column]]
        return tokenizer(texts, add_special_tokens=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing training data",
    )

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {
            key: list(chain.from_iterable(sequences))
            for key, sequences in batch.items()
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // args.block_size) * args.block_size
        return {
            key: [
                values[index : index + args.block_size]
                for index in range(0, total_length, args.block_size)
            ]
            for key, values in concatenated.items()
        }

    grouped = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Packing {args.block_size}-token sequences",
    )
    if len(grouped) == 0:
        raise ValueError("dataset contains fewer tokens than one block")
    return grouped


def private_scale_for_step(
    step: int, warmup_steps: int, target_scale: float
) -> float:
    if warmup_steps <= 0:
        return target_scale
    return target_scale * min(1.0, step / warmup_steps)


def distributed_routing_statistics(
    accelerator: Accelerator, model: torch.nn.Module
) -> dict[str, dict[str, Any]]:
    statistics: dict[str, dict[str, Any]] = {}
    for name, layer in iter_mofe_layers(accelerator.unwrap_model(model)):
        state = layer.routing_state
        if state is None:
            continue
        counts = accelerator.reduce(state.assignment_counts, reduction="sum").float()
        entropy = accelerator.reduce(state.router_entropy, reduction="mean")
        shared_norm = accelerator.reduce(state.shared_output_norm, reduction="mean")
        private_norm = accelerator.reduce(state.private_output_norm, reduction="mean")
        mean = counts.mean()
        nonzero = counts[counts > 0]
        min_nonzero = nonzero.min() if nonzero.numel() else counts.new_tensor(0.0)
        statistics[name] = {
            "assignment_counts": counts.long().cpu().tolist(),
            "assignment_fractions": (
                counts / counts.sum().clamp_min(1.0)
            ).cpu().tolist(),
            "load_cv": (counts.std(unbiased=False) / mean.clamp_min(1.0)).item(),
            "max_min_load_ratio": (
                counts.max() / min_nonzero
                if min_nonzero > 0
                else counts.new_tensor(float("inf"))
            ).item(),
            "router_entropy": entropy.item(),
            "unused_experts": int((counts == 0).sum().item()),
            "shared_output_norm": shared_norm.item(),
            "private_output_norm": private_norm.item(),
            "private_to_shared_norm": (
                private_norm / shared_norm.clamp_min(1e-12)
            ).item(),
        }
    return statistics


def save_training_state(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: Any,
    config: MoFEConfig,
    output_dir: Path,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    state_dir = output_dir / f"step_{global_step:06d}"
    accelerator.save_state(state_dir)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        config.save_pretrained(state_dir)
        (state_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2) + "\n"
        )
        tokenizer.save_pretrained(state_dir)


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    effective_batch_size = (
        args.per_device_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    accelerator.print(
        "training_config "
        f"num_processes={accelerator.num_processes} "
        f"per_device_batch_size={args.per_device_batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"effective_batch_size={effective_batch_size} "
        f"tokens_per_optimizer_step={effective_batch_size * args.block_size}",
        flush=True,
    )
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    config = MoFEConfig.from_json_file(args.mofe_config)
    config.model_name_or_path = args.model_name_or_path
    config.seed = args.seed
    cache_dir = os.environ.get("HF_HOME")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    with accelerator.main_process_first():
        dataset = load_training_dataset(args, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=cache_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    convert_gpt2_to_mofe(model, config)
    target_private_scale = config.private_output_scale
    set_private_scale(model, private_scale_for_step(0, args.private_warmup_steps, target_private_scale))

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    global_step = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        checkpoint_name = Path(args.resume_from_checkpoint).name
        if checkpoint_name.startswith("step_"):
            global_step = int(checkpoint_name.removeprefix("step_"))

    if accelerator.is_main_process:
        config.save_pretrained(output_dir)
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2) + "\n"
        )
        (output_dir / "parameter_breakdown.json").write_text(
            json.dumps(parameter_breakdown(accelerator.unwrap_model(model)), indent=2)
            + "\n"
        )

    model.train()
    started = time.perf_counter()
    tokens_seen = 0
    log_path = output_dir / "training_log.jsonl"
    while global_step < args.max_steps:
        for batch in dataloader:
            tokens_seen += batch["input_ids"].numel() * accelerator.num_processes
            scale = private_scale_for_step(
                global_step, args.private_warmup_steps, target_private_scale
            )
            set_private_scale(model, scale)
            gradient_norm = None
            with accelerator.accumulate(model):
                outputs = model(**batch)
                aux_losses = collect_mofe_losses(model)
                loss = (
                    outputs.loss
                    + config.router_aux_loss_coef * aux_losses["balance_loss"]
                    + config.router_z_loss_coef * aux_losses["z_loss"]
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gradient_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if accelerator.sync_gradients and not optimizer.step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if global_step % args.logging_steps == 0 or global_step == 1:
                elapsed = time.perf_counter() - started
                if torch.cuda.is_available():
                    local_peak_memory = torch.tensor(
                        [torch.cuda.max_memory_allocated()], device=accelerator.device
                    )
                    peak_memory = accelerator.gather(local_peak_memory).max().item()
                else:
                    peak_memory = 0
                routing = distributed_routing_statistics(accelerator, model)
                record = {
                    "step": global_step,
                    "lm_loss": float(outputs.loss.detach()),
                    "balance_loss": float(aux_losses["balance_loss"].detach()),
                    "z_loss": float(aux_losses["z_loss"].detach()),
                    "total_loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "private_scale": scale,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
                    "peak_memory_bytes": int(peak_memory),
                    "routing": routing,
                }
                routing_summary = " ".join(
                    f"h{name.split('.')[2]}:ratio={stats['private_to_shared_norm']:.2f},"
                    f"cv={stats['load_cv']:.2f}"
                    for name, stats in routing.items()
                )
                accelerator.print(
                    f"step={global_step:06d}/{args.max_steps:06d} "
                    f"lm_loss={record['lm_loss']:.4f} total_loss={record['total_loss']:.4f} "
                    f"grad_norm={record['gradient_norm']:.3f} "
                    f"lr={record['learning_rate']:.3e} private_scale={scale:.4f} "
                    f"tokens_per_second={record['tokens_per_second']:.0f} "
                    f"peak_memory_gib={peak_memory / 2**30:.2f} | {routing_summary}",
                    flush=True,
                )
                if accelerator.is_main_process:
                    with log_path.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")

            if global_step % args.save_steps == 0:
                save_training_state(
                    accelerator,
                    model,
                    tokenizer,
                    config,
                    output_dir,
                    global_step,
                    args,
                )
            if global_step >= args.max_steps:
                break

    accelerator.wait_for_everyone()
    set_private_scale(model, target_private_scale)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_mofe_checkpoint(
            unwrapped,
            output_dir / "final",
            config,
            tokenizer=tokenizer,
            metadata={"global_step": global_step, "training_args": vars(args)},
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
