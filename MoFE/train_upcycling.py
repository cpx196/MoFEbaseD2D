from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from datasets import IterableDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from MoFE.train import (
    checkpoint_step,
    evaluate_validation_loss,
    load_training_dataset,
    set_seed,
    training_dataloader_num_workers,
    verify_fp32_training_state,
    weighted_loss_payload,
)
from MoFE.upcycling import (
    UpcyclingConfig,
    collect_upcycling_losses,
    convert_gpt2_to_upcycling,
    iter_upcycling_layers,
    save_upcycling_checkpoint,
    upcycling_parameter_breakdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 Sparse Upcycling model."
    )
    parser.add_argument(
        "--train-file",
        required=True,
        help="Local .txt/.json/.jsonl/.parquet file or directory of Parquet shards",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer-size", type=int, default=2_048)
    parser.add_argument("--preprocessing-batch-size", type=int, default=64)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--upcycling-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument(
        "--scheduler", choices=("cosine", "constant"), required=True
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--validation-file")
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    parser.set_defaults(dataset_name=None, dataset_config=None, train_split="train")
    return parser.parse_args()


def make_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    training_steps: int,
):
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
    return get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
    )


def distributed_routing_statistics(
    accelerator: Accelerator, model: torch.nn.Module
) -> dict[str, dict[str, Any]]:
    statistics: dict[str, dict[str, Any]] = {}
    for name, layer in iter_upcycling_layers(accelerator.unwrap_model(model)):
        state = layer.routing_state
        if state is None:
            continue
        counts = accelerator.reduce(state.assignment_counts, reduction="sum").float()
        entropy = accelerator.reduce(state.router_entropy, reduction="mean")
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
        }
    return statistics


def save_training_state(
    accelerator: Accelerator,
    tokenizer: Any,
    config: UpcyclingConfig,
    output_dir: Path,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    state_dir = output_dir / f"step_{global_step:06d}"
    accelerator.save_state(state_dir)
    if accelerator.is_main_process:
        config.save_pretrained(state_dir)
        tokenizer.save_pretrained(state_dir)
        (state_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2) + "\n"
        )


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.per_device_batch_size <= 0:
        raise ValueError("per-device batch size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    if args.validation_steps <= 0:
        raise ValueError("validation steps must be positive")
    resume_step = checkpoint_step(args.resume_from_checkpoint)
    if resume_step >= args.max_steps:
        raise ValueError(
            f"checkpoint step {resume_step} must be lower than max steps {args.max_steps}"
        )

    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(even_batches=False),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    effective_batch_size = (
        args.per_device_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    accelerator.print(
        "training_config "
        f"model=sparse_upcycling num_processes={accelerator.num_processes} "
        f"per_device_batch_size={args.per_device_batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"effective_batch_size={effective_batch_size} "
        f"tokens_per_optimizer_step={effective_batch_size * args.block_size}",
        flush=True,
    )

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    config = UpcyclingConfig.from_json_file(args.upcycling_config)
    config.model_name_or_path = args.model_name_or_path
    config.seed = args.seed
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    with accelerator.main_process_first():
        dataset = load_training_dataset(args, tokenizer)
        if args.validation_file:
            validation_args = argparse.Namespace(**vars(args))
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
    convert_gpt2_to_upcycling(model, config)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(
        args.scheduler, optimizer, args.warmup_steps, args.max_steps
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

    accelerator.print(
        "schedule_config "
        f"resume_step={resume_step} scheduler={args.scheduler} "
        f"peak_learning_rate={args.learning_rate:.3e} "
        f"warmup_steps={args.warmup_steps}",
        flush=True,
    )
    if accelerator.is_main_process:
        config.save_pretrained(output_dir)
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2) + "\n"
        )
        (output_dir / "parameter_breakdown.json").write_text(
            json.dumps(
                upcycling_parameter_breakdown(accelerator.unwrap_model(model)),
                indent=2,
            )
            + "\n"
        )

    model.train()
    started = time.perf_counter()
    tokens_seen = 0
    log_path = output_dir / "training_log.jsonl"
    validation_log_path = output_dir / "validation_log.jsonl"

    def log_validation(step: int) -> None:
        if validation_dataloader is None:
            return
        result = evaluate_validation_loss(accelerator, model, validation_dataloader)
        record = {"step": step, **result}
        accelerator.print(
            f"validation step={step:06d}/{args.max_steps:06d} "
            f"lm_loss={record['lm_loss']:.6f} ppl={record['perplexity']:.4f} "
            f"samples={record['sample_count']} tokens={record['token_count']}",
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
            gradient_norm = None
            with accelerator.accumulate(model):
                outputs = model(**batch)
                aux_losses = collect_upcycling_losses(model)
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
                    gradient_norm = accelerator.clip_grad_norm_(
                        model.parameters(), 1.0
                    )
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
                    local_peak = torch.tensor(
                        [torch.cuda.max_memory_allocated()], device=accelerator.device
                    )
                    peak_memory = accelerator.gather(local_peak).max().item()
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
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
                    "peak_memory_bytes": int(peak_memory),
                    "routing": routing,
                }
                routing_summary = " ".join(
                    f"h{name.split('.')[2]}:cv={stats['load_cv']:.2f},"
                    f"unused={stats['unused_experts']}"
                    for name, stats in routing.items()
                )
                accelerator.print(
                    f"step={global_step:06d}/{args.max_steps:06d} "
                    f"lm_loss={record['lm_loss']:.4f} "
                    f"total_loss={record['total_loss']:.4f} "
                    f"batch_samples={effective_batch_samples} "
                    f"grad_norm={record['gradient_norm']:.3f} "
                    f"lr={record['learning_rate']:.3e} "
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
    if accelerator.is_main_process:
        save_upcycling_checkpoint(
            accelerator.unwrap_model(model),
            output_dir / "final",
            config,
            tokenizer=tokenizer,
            metadata={"global_step": global_step, "training_args": vars(args)},
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
