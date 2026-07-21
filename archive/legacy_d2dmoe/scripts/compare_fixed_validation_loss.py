import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-validation-log", required=True)
    parser.add_argument("--mofe-validation-log", required=True)
    parser.add_argument("--dense-training-log", required=True)
    parser.add_argument("--mofe-training-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase-boundary", type=int, default=100)
    parser.add_argument("--dataset-label", default="WikiText-103")
    parser.add_argument(
        "--experiment-label", default="Dense vs MoFE: Matched 1000-step Training"
    )
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
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.tokens_per_step <= 0:
        raise ValueError("tokens per step must be positive")
    if args.equal_token_loss_limits[0] >= args.equal_token_loss_limits[1]:
        raise ValueError("equal-token loss limits must be ordered MIN MAX")

    dense = read_jsonl(args.dense_validation_log)
    mofe = read_jsonl(args.mofe_validation_log)
    dense_by_step = {record["step"]: record for record in dense}
    mofe_by_step = {record["step"]: record for record in mofe}
    steps = sorted(set(dense_by_step) & set(mofe_by_step))
    if steps != sorted(dense_by_step) or steps != sorted(mofe_by_step):
        raise ValueError("Dense and MoFE validation logs must have identical steps")
    for step in steps:
        if dense_by_step[step]["token_count"] != mofe_by_step[step]["token_count"]:
            raise ValueError(f"validation token counts differ at step {step}")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "step": step,
            "training_tokens": step * args.tokens_per_step,
            "dense_validation_loss": dense_by_step[step]["lm_loss"],
            "mofe_validation_loss": mofe_by_step[step]["lm_loss"],
            "mofe_minus_dense_loss": (
                mofe_by_step[step]["lm_loss"] - dense_by_step[step]["lm_loss"]
            ),
            "dense_perplexity": dense_by_step[step]["perplexity"],
            "mofe_perplexity": mofe_by_step[step]["perplexity"],
            "mofe_minus_dense_perplexity": (
                mofe_by_step[step]["perplexity"]
                - dense_by_step[step]["perplexity"]
            ),
        }
        for step in steps
    ]
    csv_path = output_dir / "dense_mofe_fixed_validation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    step_values = np.array(steps)
    colors = ("#3778A8", "#D27A3F")
    fig, (ax_loss, ax_ppl) = plt.subplots(
        1, 2, figsize=(13, 5.5), dpi=170, layout="constrained"
    )
    for ax in (ax_loss, ax_ppl):
        ax.axvline(
            args.phase_boundary, color="#555555", linestyle="--", linewidth=1.2
        )
        ax.grid(alpha=0.24)
        ax.set_xlabel("Optimizer step")
    ax_loss.plot(
        step_values,
        [dense_by_step[step]["lm_loss"] for step in steps],
        color=colors[0],
        marker="o",
        linewidth=2,
        label="Dense",
    )
    ax_loss.plot(
        step_values,
        [mofe_by_step[step]["lm_loss"] for step in steps],
        color=colors[1],
        marker="o",
        linewidth=2,
        label="MoFE",
    )
    ax_loss.set_title(f"Fixed {args.dataset_label} Validation Loss")
    ax_loss.set_ylabel("Token-weighted language-modeling loss")
    ax_loss.legend()

    ax_ppl.plot(
        step_values,
        [dense_by_step[step]["perplexity"] for step in steps],
        color=colors[0],
        marker="o",
        linewidth=2,
        label="Dense",
    )
    ax_ppl.plot(
        step_values,
        [mofe_by_step[step]["perplexity"] for step in steps],
        color=colors[1],
        marker="o",
        linewidth=2,
        label="MoFE",
    )
    ax_ppl.set_title(f"Fixed {args.dataset_label} Validation Perplexity")
    ax_ppl.set_ylabel("Perplexity (lower is better)")
    ax_ppl.legend()
    fig.suptitle(args.experiment_label, fontsize=14)
    figure_path = figures_dir / "dense_vs_mofe_fixed_validation.png"
    fig.savefig(figure_path)
    plt.close(fig)

    token_values_millions = step_values * args.tokens_per_step / 1_000_000
    fig, ax = plt.subplots(figsize=(10, 6), dpi=170, layout="constrained")
    ax.plot(
        token_values_millions,
        [dense_by_step[step]["lm_loss"] for step in steps],
        color=colors[0],
        marker="o",
        linewidth=2,
        label="Dense",
    )
    ax.plot(
        token_values_millions,
        [mofe_by_step[step]["lm_loss"] for step in steps],
        color=colors[1],
        marker="o",
        linewidth=2,
        label="MoFE",
    )
    ax.set_xlim(0, token_values_millions[-1])
    ax.set_ylim(*args.equal_token_loss_limits)
    ax.set_xlabel("Cumulative training tokens (millions)")
    ax.set_ylabel("Fixed FineWeb-Edu validation loss")
    ax.set_title("Dense vs MoFE: Equal-Token Training")
    ax.grid(alpha=0.24)
    ax.legend()
    equal_token_figure_path = (
        figures_dir / "dense_vs_mofe_equal_tokens_validation_loss.png"
    )
    fig.savefig(equal_token_figure_path)
    plt.close(fig)

    dense_training = read_jsonl(args.dense_training_log)
    mofe_training = read_jsonl(args.mofe_training_log)
    dense_training_by_step = {record["step"]: record for record in dense_training}
    mofe_training_by_step = {record["step"]: record for record in mofe_training}
    training_steps = sorted(set(dense_training_by_step) & set(mofe_training_by_step))
    if training_steps != sorted(dense_training_by_step) or training_steps != sorted(
        mofe_training_by_step
    ):
        raise ValueError("Dense and MoFE training logs must have identical sampled steps")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=170, layout="constrained")
    ax.plot(
        training_steps,
        [dense_training_by_step[step]["lm_loss"] for step in training_steps],
        color=colors[0],
        marker="o",
        linewidth=1.8,
        label="Dense",
    )
    ax.plot(
        training_steps,
        [mofe_training_by_step[step]["lm_loss"] for step in training_steps],
        color=colors[1],
        marker="o",
        linewidth=1.8,
        label="MoFE",
    )
    ax.set_xlim(0, steps[-1])
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Sampled training LM loss")
    ax.set_title("Dense vs MoFE: 10-Step Training-Loss Samples")
    ax.grid(alpha=0.24)
    ax.legend()
    training_figure_path = figures_dir / "dense_vs_mofe_training_lm_loss_10step.png"
    fig.savefig(training_figure_path)
    plt.close(fig)

    dense_train = dense_training[-1]
    mofe_train = mofe_training[-1]
    final = rows[-1]
    dense_best = min(rows, key=lambda row: row["dense_validation_loss"])
    mofe_best = min(rows, key=lambda row: row["mofe_validation_loss"])
    summary = {
        "optimizer_steps": int(steps[-1]),
        "tokens_per_step": args.tokens_per_step,
        "total_training_tokens": int(steps[-1] * args.tokens_per_step),
        "validation_interval_steps": int(steps[1] - steps[0]),
        "validation_point_count": len(steps),
        "validation_sample_count": dense[0]["sample_count"],
        "validation_token_count": dense[0]["token_count"],
        "dense_initial_loss": dense[0]["lm_loss"],
        "dense_final_loss": final["dense_validation_loss"],
        "dense_loss_change": final["dense_validation_loss"] - dense[0]["lm_loss"],
        "dense_best_step": dense_best["step"],
        "dense_best_loss": dense_best["dense_validation_loss"],
        "mofe_initial_loss": mofe[0]["lm_loss"],
        "mofe_final_loss": final["mofe_validation_loss"],
        "mofe_loss_change": final["mofe_validation_loss"] - mofe[0]["lm_loss"],
        "mofe_best_step": mofe_best["step"],
        "mofe_best_loss": mofe_best["mofe_validation_loss"],
        "mofe_minus_dense_final_loss": final["mofe_minus_dense_loss"],
        "dense_final_perplexity": final["dense_perplexity"],
        "mofe_final_perplexity": final["mofe_perplexity"],
        "mofe_perplexity_change_percent_vs_dense": (
            final["mofe_perplexity"] / final["dense_perplexity"] - 1.0
        )
        * 100.0,
        "dense_tokens_per_second": dense_train["tokens_per_second"],
        "mofe_tokens_per_second": mofe_train["tokens_per_second"],
        "dense_peak_memory_gib": dense_train["peak_memory_bytes"] / 2**30,
        "mofe_peak_memory_gib": mofe_train["peak_memory_bytes"] / 2**30,
    }
    summary_path = output_dir / "dense_mofe_fixed_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = output_dir / "dense_mofe_fixed_validation_summary.md"
    lines = [
        "# Dense vs MoFE Fixed-validation Comparison",
        "",
        "Both runs use the same GPT-2 initialization, dataset, sequence length, "
        "effective batch size, learning-rate schedule, seed, and training-token budget.",
        f"They run for {steps[-1]} optimizer steps and see "
        f"{steps[-1] * args.tokens_per_step:,} training tokens each.",
        "",
        "| Step | Dense loss | MoFE loss | MoFE - Dense | Dense PPL | MoFE PPL |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['dense_validation_loss']:.6f} | "
            f"{row['mofe_validation_loss']:.6f} | "
            f"{row['mofe_minus_dense_loss']:+.6f} | "
            f"{row['dense_perplexity']:.4f} | {row['mofe_perplexity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| Model | Tokens/s | Peak GiB/GPU |",
            "| --- | ---: | ---: |",
            f"| Dense | {summary['dense_tokens_per_second']:.0f} | "
            f"{summary['dense_peak_memory_gib']:.2f} |",
            f"| MoFE | {summary['mofe_tokens_per_second']:.0f} | "
            f"{summary['mofe_peak_memory_gib']:.2f} |",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved equal-token figure: {equal_token_figure_path}")
    print(f"Saved sampled training-loss figure: {training_figure_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
