from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from datasets import Dataset, IterableDataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_constant_schedule_with_warmup,
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
    source.add_argument(
        "--train-file",
        help="Local .txt/.json/.jsonl/.parquet file or directory of Parquet shards",
    )
    parser.add_argument("--dataset-config")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset rows instead of materializing preprocessing caches.",
    )
    parser.add_argument("--shuffle-buffer-size", type=int, default=2_048)
    parser.add_argument("--preprocessing-batch-size", type=int, default=64)
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
    parser.add_argument(
        "--shared-learning-rate",
        type=float,
        help="Learning rate for the GPT-2 backbone and MoFE shared experts.",
    )
    parser.add_argument(
        "--private-learning-rate",
        type=float,
        help="Learning rate for MoFE factor banks, cores, and private biases.",
    )
    parser.add_argument(
        "--router-learning-rate",
        type=float,
        help="Learning rate for MoFE router parameters.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--scheduler", choices=("cosine", "constant"), default="cosine"
    )
    parser.add_argument(
        "--private-warmup-steps",
        type=int,
        default=200,
        help="Linearly increase the private branch scale from 0 to its configured value.",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--validation-file")
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--restart-scheduler-on-resume",
        action="store_true",
        help=(
            "Restore model and optimizer state, but start a new warmup/constant "
            "schedule for the remaining steps."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mofe_optimizer_groups(
    model: torch.nn.Module,
    base_learning_rate: float,
    shared_learning_rate: float | None,
    private_learning_rate: float | None,
    router_learning_rate: float | None,
) -> list[dict[str, Any]]:
    shared_lr = shared_learning_rate or base_learning_rate
    private_lr = private_learning_rate or base_learning_rate
    router_lr = router_learning_rate or base_learning_rate

    private_parameters: list[torch.nn.Parameter] = []
    router_parameters: list[torch.nn.Parameter] = []
    assigned: set[int] = set()
    for _, layer in iter_mofe_layers(model):
        for parameter in (layer.a1, layer.b1, layer.a2, layer.b2):
            if parameter.requires_grad:
                private_parameters.append(parameter)
                assigned.add(id(parameter))
        for parameter in (layer.core1, layer.core2):
            if parameter.requires_grad:
                private_parameters.append(parameter)
                assigned.add(id(parameter))
        for parameter in (layer.private_bias1, layer.private_bias2):
            if parameter.requires_grad:
                private_parameters.append(parameter)
                assigned.add(id(parameter))
        for parameter in layer.router.parameters():
            if parameter.requires_grad:
                router_parameters.append(parameter)
                assigned.add(id(parameter))

    shared_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in assigned
    ]
    groups = [
        {"params": shared_parameters, "lr": shared_lr, "name": "shared"},
        {"params": private_parameters, "lr": private_lr, "name": "private"},
        {"params": router_parameters, "lr": router_lr, "name": "router"},
    ]
    return [group for group in groups if group["params"]]


def weighted_loss_payload(
    losses: tuple[torch.Tensor, ...], sample_count: int
) -> torch.Tensor:
    """Pack sample-weighted loss sums and their sample count for reduction."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    values = torch.stack([value.detach().to(dtype=torch.float64) for value in losses])
    count = values.new_tensor([sample_count])
    return torch.cat((values * count, count))


def verify_fp32_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[str], list[str]]:
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    optimizer_dtypes = sorted(
        {
            str(value.dtype)
            for state in raw_optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and value.is_floating_point()
        }
    )
    if parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"expected FP32 master parameters, got {parameter_dtypes}")
    if optimizer_dtypes != ["torch.float32"]:
        raise RuntimeError(f"expected FP32 optimizer states, got {optimizer_dtypes}")
    return parameter_dtypes, optimizer_dtypes


@torch.no_grad()
def evaluate_validation_loss(
    accelerator: Accelerator,
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    local_totals = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
    for batch in dataloader:
        outputs = model(**batch)
        labels = batch["labels"]
        valid_tokens = (labels[:, 1:] != -100).sum()
        local_totals[0] += outputs.loss.detach().double() * valid_tokens
        local_totals[1] += valid_tokens
        local_totals[2] += labels.shape[0]

    totals = accelerator.reduce(local_totals, reduction="sum")
    if totals[1] <= 0:
        raise ValueError("validation dataset contains no predicted tokens")
    mean_loss = (totals[0] / totals[1]).item()
    if was_training:
        model.train()
    return {
        "lm_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "token_count": int(totals[1].item()),
        "sample_count": int(totals[2].item()),
    }


def resolve_local_dataset(train_file: str) -> tuple[str, str | list[str], bool]:
    path = Path(train_file).expanduser()
    if path.is_dir():
        files = sorted(
            str(candidate.resolve()) for candidate in path.glob("*.parquet")
        )
        if not files:
            raise ValueError(f"Parquet directory contains no .parquet files: {path}")
        return "parquet", files, True
    if not path.is_file():
        raise FileNotFoundError(f"training data does not exist: {path}")

    loader_by_suffix = {
        ".txt": "text",
        ".json": "json",
        ".jsonl": "json",
        ".parquet": "parquet",
    }
    suffix = path.suffix.lower()
    if suffix not in loader_by_suffix:
        supported = ", ".join(sorted(loader_by_suffix))
        raise ValueError(
            f"unsupported training data suffix {suffix!r}; expected one of {supported}"
        )
    return loader_by_suffix[suffix], str(path.resolve()), False


def load_training_dataset(
    args: argparse.Namespace, tokenizer: Any
) -> Dataset | IterableDataset:
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    text_column = args.text_column
    block_size = args.block_size
    streaming = bool(getattr(args, "streaming", False))
    shuffle_buffer_size = int(getattr(args, "shuffle_buffer_size", 2_048))
    preprocessing_batch_size = int(
        getattr(args, "preprocessing_batch_size", 64)
    )
    if shuffle_buffer_size <= 0:
        raise ValueError("shuffle buffer size must be positive")
    if preprocessing_batch_size <= 0:
        raise ValueError("preprocessing batch size must be positive")

    if args.dataset_name:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.train_split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    else:
        loader, data_files, directory_streaming = resolve_local_dataset(
            args.train_file
        )
        streaming = streaming or directory_streaming
        args.streaming = streaming
        dataset = load_dataset(
            loader,
            data_files={"train": data_files},
            split="train",
            cache_dir=cache_dir,
            streaming=streaming,
        )
    columns = dataset.column_names or list(dataset.features or {})
    if text_column not in columns:
        raise ValueError(
            f"text column {text_column!r} is absent; available columns: "
            f"{columns}"
        )

    if streaming:
        dataset = dataset.shuffle(
            seed=args.seed,
            buffer_size=shuffle_buffer_size,
        )

    remove_columns = columns

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        texts = [str(value) for value in batch[text_column]]
        return tokenizer(texts, add_special_tokens=False)

    map_batch_kwargs = (
        {"batch_size": preprocessing_batch_size} if streaming else {}
    )
    tokenize_progress_kwargs = (
        {} if streaming else {"desc": "Tokenizing training data"}
    )
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=remove_columns,
        **map_batch_kwargs,
        **tokenize_progress_kwargs,
    )

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {
            key: list(chain.from_iterable(sequences))
            for key, sequences in batch.items()
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        return {
            key: [
                values[index : index + block_size]
                for index in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }

    packing_progress_kwargs = (
        {} if streaming else {"desc": f"Packing {block_size}-token sequences"}
    )
    grouped = tokenized.map(
        group_texts,
        batched=True,
        **map_batch_kwargs,
        **packing_progress_kwargs,
    )
    if not isinstance(grouped, IterableDataset) and len(grouped) == 0:
        raise ValueError("dataset contains fewer tokens than one block")
    return grouped


def training_dataloader_num_workers(
    dataset: Dataset | IterableDataset, requested_workers: int
) -> int:
    if requested_workers < 0:
        raise ValueError("num workers must be non-negative")
    if isinstance(dataset, IterableDataset):
        return 0
    return requested_workers


def private_scale_for_step(
    step: int, warmup_steps: int, target_scale: float
) -> float:
    if warmup_steps <= 0:
        return target_scale
    return target_scale * min(1.0, step / warmup_steps)


def checkpoint_step(checkpoint: str | None) -> int:
    if not checkpoint:
        return 0
    checkpoint_name = Path(checkpoint).name
    if not checkpoint_name.startswith("step_"):
        raise ValueError(
            "resume checkpoint directory must be named step_NNNNNN so the global "
            "step can be restored"
        )
    return int(checkpoint_name.removeprefix("step_"))


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
    if args.validation_steps <= 0:
        raise ValueError("validation_steps must be positive")
    resume_step = checkpoint_step(args.resume_from_checkpoint)
    if args.restart_scheduler_on_resume and not args.resume_from_checkpoint:
        raise ValueError("--restart-scheduler-on-resume requires a checkpoint")
    if resume_step >= args.max_steps:
        raise ValueError(
            f"checkpoint step {resume_step} must be lower than max steps {args.max_steps}"
        )
    set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    with accelerator.main_process_first():
        dataset = load_training_dataset(args, tokenizer)
        if args.validation_file:
            validation_args = argparse.Namespace(**vars(args))
            validation_args.dataset_name = None
            validation_args.train_file = args.validation_file
            validation_args.streaming = False
            validation_dataset = load_training_dataset(validation_args, tokenizer)
        else:
            validation_dataset = None
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_num_workers = training_dataloader_num_workers(dataset, args.num_workers)
    if training_num_workers != args.num_workers:
        accelerator.print(
            "data_loader_config streaming=true "
            f"requested_num_workers={args.num_workers} effective_num_workers=0",
            flush=True,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=not isinstance(dataset, IterableDataset),
        num_workers=training_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    validation_dataloader = (
        DataLoader(
            validation_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collator,
        )
        if validation_dataset is not None
        else None
    )
    if validation_dataset is not None:
        validation_sequences = (
            "streaming"
            if isinstance(validation_dataset, IterableDataset)
            else str(len(validation_dataset))
        )
        accelerator.print(
            "validation_config "
            f"file={args.validation_file} sequences={validation_sequences} "
            f"interval_steps={args.validation_steps}",
            flush=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.float32,
    )
    convert_gpt2_to_mofe(model, config)
    target_private_scale = config.private_output_scale
    set_private_scale(model, private_scale_for_step(0, args.private_warmup_steps, target_private_scale))

    optimizer_groups = mofe_optimizer_groups(
        model,
        args.learning_rate,
        args.shared_learning_rate,
        args.private_learning_rate,
        args.router_learning_rate,
    )
    optimizer = AdamW(
        optimizer_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.restart_scheduler_on_resume:
        if validation_dataloader is None:
            model, optimizer, dataloader = accelerator.prepare(
                model, optimizer, dataloader
            )
        else:
            model, optimizer, dataloader, validation_dataloader = accelerator.prepare(
                model, optimizer, dataloader, validation_dataloader
            )
        accelerator.load_state(args.resume_from_checkpoint)
        for parameter_group in optimizer.param_groups:
            group_lr = parameter_group.get("initial_lr", parameter_group["lr"])
            parameter_group["lr"] = group_lr
            parameter_group["initial_lr"] = group_lr
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        scheduler = get_constant_schedule_with_warmup(
            raw_optimizer,
            num_warmup_steps=args.warmup_steps,
        )
        scheduler = accelerator.prepare_scheduler(scheduler)
        accelerator.print(
            "resume_scheduler "
            f"checkpoint_step={resume_step} remaining_steps={args.max_steps - resume_step} "
            f"peak_learning_rate={args.learning_rate:.3e} "
            f"warmup_steps={args.warmup_steps}",
            flush=True,
        )
    else:
        if args.scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.max_steps,
            )
        if validation_dataloader is None:
            model, optimizer, dataloader, scheduler = accelerator.prepare(
                model, optimizer, dataloader, scheduler
            )
        else:
            (
                model,
                optimizer,
                dataloader,
                validation_dataloader,
                scheduler,
            ) = accelerator.prepare(
                model, optimizer, dataloader, validation_dataloader, scheduler
            )
        if args.resume_from_checkpoint:
            accelerator.load_state(args.resume_from_checkpoint)
    global_step = resume_step

    if accelerator.is_main_process:
        config.save_pretrained(output_dir)
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2) + "\n"
        )
        (output_dir / "parameter_breakdown.json").write_text(
            json.dumps(parameter_breakdown(accelerator.unwrap_model(model)), indent=2)
            + "\n"
        )
    accelerator.print(
        "optimizer_groups "
        + " ".join(
            f"{group.get('name', index)}:lr={group['lr']:.3e},params="
            f"{sum(parameter.numel() for parameter in group['params'])}"
            for index, group in enumerate(optimizer.param_groups)
        ),
        flush=True,
    )

    model.train()
    started = time.perf_counter()
    tokens_seen = 0
    log_path = output_dir / "training_log.jsonl"
    validation_log_path = output_dir / "validation_log.jsonl"

    def log_validation(step: int) -> None:
        if validation_dataloader is None:
            return
        scale = private_scale_for_step(
            step, args.private_warmup_steps, target_private_scale
        )
        set_private_scale(model, scale)
        result = evaluate_validation_loss(accelerator, model, validation_dataloader)
        record = {"step": step, "private_scale": scale, **result}
        accelerator.print(
            f"validation step={step:06d}/{args.max_steps:06d} "
            f"lm_loss={record['lm_loss']:.6f} ppl={record['perplexity']:.4f} "
            f"samples={record['sample_count']} tokens={record['token_count']} "
            f"private_scale={scale:.4f}",
            flush=True,
        )
        if accelerator.is_main_process:
            with validation_log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    log_validation(global_step)
    loss_accumulator = None
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
                microbatch_losses = weighted_loss_payload(
                    (
                        outputs.loss,
                        aux_losses["balance_loss"],
                        aux_losses["z_loss"],
                        loss,
                    ),
                    batch["input_ids"].shape[0],
                )
                if loss_accumulator is None:
                    loss_accumulator = microbatch_losses
                else:
                    loss_accumulator += microbatch_losses
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
            if global_step == 1:
                parameter_dtypes, optimizer_dtypes = verify_fp32_training_state(
                    model, optimizer
                )
                accelerator.print(
                    "precision_config "
                    f"master_parameter_dtypes={parameter_dtypes} "
                    f"optimizer_state_dtypes={optimizer_dtypes} "
                    f"compute_precision={accelerator.mixed_precision}",
                    flush=True,
                )
            reduced_losses = accelerator.reduce(loss_accumulator, reduction="sum")
            effective_batch_samples = int(reduced_losses[-1].item())
            mean_losses = reduced_losses[:-1] / reduced_losses[-1].clamp_min(1)
            loss_accumulator = None
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
                    "lm_loss": float(mean_losses[0]),
                    "balance_loss": float(mean_losses[1]),
                    "z_loss": float(mean_losses[2]),
                    "total_loss": float(mean_losses[3]),
                    "effective_batch_samples": effective_batch_samples,
                    "gradient_norm": float(gradient_norm.detach()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "learning_rates": {
                        str(group.get("name", index)): group["lr"]
                        for index, group in enumerate(optimizer.param_groups)
                    },
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
                    f"batch_samples={effective_batch_samples} "
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
            if (
                global_step % args.validation_steps == 0
                or global_step >= args.max_steps
            ):
                log_validation(global_step)
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
