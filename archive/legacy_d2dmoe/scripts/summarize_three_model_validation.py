from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_NAMES = ("dense", "mofe", "upcycling")
DISPLAY_NAMES = {
    "dense": "Dense",
    "mofe": "MoFE",
    "upcycling": "Upcycling",
}
COLORS = {
    "dense": "#3778A8",
    "mofe": "#D27A3F",
    "upcycling": "#4E967D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for model in MODEL_NAMES:
        parser.add_argument(f"--{model}-validation-log", required=True)
        parser.add_argument(f"--{model}-training-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    parser.add_argument(
        "--equal-token-loss-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(3.15, 3.35),
    )
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def index_records(records: list[dict], kind: str, model: str) -> dict[int, dict]:
    indexed = {int(record["step"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"duplicate {kind} steps in {model} log")
    return indexed


def main() -> None:
    args = parse_args()
    if args.tokens_per_step <= 0:
        raise ValueError("tokens per step must be positive")
    if args.equal_token_loss_limits[0] >= args.equal_token_loss_limits[1]:
        raise ValueError("equal-token loss limits must be ordered MIN MAX")

    validation = {
        model: index_records(
            read_jsonl(getattr(args, f"{model}_validation_log")),
            "validation",
            model,
        )
        for model in MODEL_NAMES
    }
    training = {
        model: index_records(
            read_jsonl(getattr(args, f"{model}_training_log")),
            "training",
            model,
        )
        for model in MODEL_NAMES
    }
    validation_steps = sorted(validation["dense"])
    training_steps = sorted(training["dense"])
    for model in MODEL_NAMES[1:]:
        if sorted(validation[model]) != validation_steps:
            raise ValueError(f"validation steps differ for {model}")
        if sorted(training[model]) != training_steps:
            raise ValueError(f"training sample steps differ for {model}")
    for step in validation_steps:
        token_counts = {validation[model][step]["token_count"] for model in MODEL_NAMES}
        if len(token_counts) != 1:
            raise ValueError(f"validation token counts differ at step {step}")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in validation_steps:
        row = {"step": step, "training_tokens": step * args.tokens_per_step}
        for model in MODEL_NAMES:
            row[f"{model}_validation_loss"] = validation[model][step]["lm_loss"]
            row[f"{model}_perplexity"] = validation[model][step]["perplexity"]
        rows.append(row)

    csv_path = output_dir / "three_model_fixed_validation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    step_values = np.array(validation_steps)
    fig, (ax_loss, ax_ppl) = plt.subplots(
        1, 2, figsize=(13, 5.5), dpi=170, layout="constrained"
    )
    for model in MODEL_NAMES:
        label = DISPLAY_NAMES[model]
        color = COLORS[model]
        ax_loss.plot(
            step_values,
            [validation[model][step]["lm_loss"] for step in validation_steps],
            color=color,
            marker="o",
            linewidth=2,
            label=label,
        )
        ax_ppl.plot(
            step_values,
            [validation[model][step]["perplexity"] for step in validation_steps],
            color=color,
            marker="o",
            linewidth=2,
            label=label,
        )
    for ax in (ax_loss, ax_ppl):
        ax.grid(alpha=0.24)
        ax.set_xlabel("Optimizer step")
        ax.legend()
    ax_loss.set_title("Fixed FineWeb-Edu Validation Loss")
    ax_loss.set_ylabel("Token-weighted language-modeling loss")
    ax_ppl.set_title("Fixed FineWeb-Edu Validation Perplexity")
    ax_ppl.set_ylabel("Perplexity (lower is better)")
    fig.suptitle("FP32 Master, Constant LR, No Warmup: 200 Steps", fontsize=14)
    validation_figure = figures_dir / "three_model_fixed_validation.png"
    fig.savefig(validation_figure)
    plt.close(fig)

    token_values = step_values * args.tokens_per_step / 1_000_000
    fig, ax = plt.subplots(figsize=(10, 6), dpi=170, layout="constrained")
    for model in MODEL_NAMES:
        ax.plot(
            token_values,
            [validation[model][step]["lm_loss"] for step in validation_steps],
            color=COLORS[model],
            marker="o",
            linewidth=2,
            label=DISPLAY_NAMES[model],
        )
    ax.set_xlim(0, token_values[-1])
    ax.set_ylim(*args.equal_token_loss_limits)
    ax.set_xlabel("Cumulative training tokens (millions)")
    ax.set_ylabel("Fixed FineWeb-Edu validation loss")
    ax.set_title("Dense vs MoFE vs Upcycling: Equal-Token Training")
    ax.grid(alpha=0.24)
    ax.legend()
    equal_token_figure = figures_dir / "three_model_equal_tokens_validation_loss.png"
    fig.savefig(equal_token_figure)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=170, layout="constrained")
    for model in MODEL_NAMES:
        ax.plot(
            training_steps,
            [training[model][step]["lm_loss"] for step in training_steps],
            color=COLORS[model],
            marker="o",
            linewidth=1.8,
            label=DISPLAY_NAMES[model],
        )
    ax.set_xlim(0, validation_steps[-1])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Sampled training LM loss")
    ax.set_title("10-Step Training-Loss Samples")
    ax.grid(alpha=0.24)
    ax.legend()
    training_figure = figures_dir / "three_model_training_lm_loss_10step.png"
    fig.savefig(training_figure)
    plt.close(fig)

    final = rows[-1]
    summary = {
        "optimizer_steps": validation_steps[-1],
        "tokens_per_step": args.tokens_per_step,
        "total_training_tokens": validation_steps[-1] * args.tokens_per_step,
        "validation_interval_steps": validation_steps[1] - validation_steps[0],
        "validation_point_count": len(validation_steps),
        "validation_token_count": validation["dense"][validation_steps[0]][
            "token_count"
        ],
        "models": {},
    }
    for model in MODEL_NAMES:
        losses = [validation[model][step]["lm_loss"] for step in validation_steps]
        best_index = int(np.argmin(losses))
        final_training = training[model][training_steps[-1]]
        summary["models"][model] = {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_change": losses[-1] - losses[0],
            "best_step": validation_steps[best_index],
            "best_loss": losses[best_index],
            "final_perplexity": final[f"{model}_perplexity"],
            "tokens_per_second": final_training["tokens_per_second"],
            "peak_memory_gib": final_training["peak_memory_bytes"] / 2**30,
        }
    dense_final = summary["models"]["dense"]["final_loss"]
    for model in MODEL_NAMES[1:]:
        summary["models"][model]["final_loss_minus_dense"] = (
            summary["models"][model]["final_loss"] - dense_final
        )

    summary_path = output_dir / "three_model_fixed_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path = output_dir / "three_model_fixed_validation_summary.md"
    lines = [
        "# Dense vs MoFE vs Upcycling: 200-Step Validation",
        "",
        "All models use FP32 master parameters and AdamW states, BF16 compute, "
        "constant LR 1e-5, no LR warmup, and 6,553,600 FineWeb-Edu tokens.",
        "",
        "| Model | Initial loss | Final loss | Loss change | Final PPL | Tokens/s | Peak GiB/GPU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODEL_NAMES:
        values = summary["models"][model]
        lines.append(
            f"| {DISPLAY_NAMES[model]} | {values['initial_loss']:.6f} | "
            f"{values['final_loss']:.6f} | {values['loss_change']:+.6f} | "
            f"{values['final_perplexity']:.4f} | "
            f"{values['tokens_per_second']:.0f} | {values['peak_memory_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"- MoFE final loss minus Dense: "
            f"`{summary['models']['mofe']['final_loss_minus_dense']:+.6f}`",
            f"- Upcycling final loss minus Dense: "
            f"`{summary['models']['upcycling']['final_loss_minus_dense']:+.6f}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved validation figure: {validation_figure}")
    print(f"Saved equal-token figure: {equal_token_figure}")
    print(f"Saved training figure: {training_figure}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
