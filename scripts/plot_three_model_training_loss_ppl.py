import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("Dense", "dense", "#4778A8"),
    ("MoFE", "mofe", "#16856B"),
    ("Upcycling", "upcycling", "#D97706"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-log", required=True)
    parser.add_argument("--mofe-log", required=True)
    parser.add_argument("--upcycling-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-steps", type=int, default=50)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trailing_average(values: np.ndarray, points: int) -> np.ndarray:
    kernel = np.ones(points, dtype=np.float64) / points
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    if args.rolling_steps <= 0:
        raise ValueError("rolling steps must be positive")
    paths = {
        "dense": args.dense_log,
        "mofe": args.mofe_log,
        "upcycling": args.upcycling_log,
    }
    records = {key: read_jsonl(path) for key, path in paths.items()}
    by_step = {
        key: {record["step"]: record for record in model_records}
        for key, model_records in records.items()
    }
    step_sets = [set(model_records) for model_records in by_step.values()]
    if any(steps != step_sets[0] for steps in step_sets[1:]):
        raise ValueError("training logs must contain identical optimizer steps")
    steps = np.array(sorted(step_sets[0]))
    if any(
        record.get("effective_batch_samples") != 32
        for model_records in records.values()
        for record in model_records
    ):
        raise ValueError("all training samples must use effective batch 32")
    logging_interval = int(np.median(np.diff(steps)))
    rolling_points = max(1, args.rolling_steps // logging_interval)
    smooth_steps = steps[rolling_points - 1 :]
    losses = {
        key: np.array([by_step[key][step]["lm_loss"] for step in steps])
        for key in by_step
    }
    smooth_losses = {
        key: trailing_average(values, rolling_points)
        for key, values in losses.items()
    }

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dense_mofe_upcycling_training_loss_ppl.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "step",
            "dense_loss",
            "dense_ppl",
            "mofe_loss",
            "mofe_ppl",
            "upcycling_loss",
            "upcycling_ppl",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, step in enumerate(steps):
            writer.writerow(
                {
                    "step": step,
                    **{
                        f"{key}_{metric}": (
                            losses[key][index]
                            if metric == "loss"
                            else math.exp(losses[key][index])
                        )
                        for key in losses
                        for metric in ("loss", "ppl")
                    },
                }
            )

    fig, (ax_loss, ax_ppl) = plt.subplots(
        1, 2, figsize=(14, 5.7), dpi=180, layout="constrained"
    )
    for label, key, color in MODELS:
        ax_loss.plot(
            steps,
            losses[key],
            color=color,
            alpha=0.18,
            linewidth=0.8,
        )
        ax_loss.plot(
            smooth_steps,
            smooth_losses[key],
            color=color,
            linewidth=2.3,
            label=label,
        )
        ax_ppl.plot(
            steps,
            np.exp(losses[key]),
            color=color,
            alpha=0.18,
            linewidth=0.8,
        )
        ax_ppl.plot(
            smooth_steps,
            np.exp(smooth_losses[key]),
            color=color,
            linewidth=2.3,
            label=label,
        )
    for ax in (ax_loss, ax_ppl):
        ax.set_xlabel("Optimizer step")
        ax.set_xlim(0, int(steps[-1]))
        ax.grid(alpha=0.23)
        ax.legend()
    ax_loss.set_title("Effective-batch Training Loss")
    ax_loss.set_ylabel("Language-modeling loss")
    ax_ppl.set_title("Effective-batch Training Perplexity")
    ax_ppl.set_ylabel("PPL = exp(training loss)")
    fig.suptitle(
        f"Dense vs MoFE vs Upcycling Training Curves\n"
        f"{len(steps)} raw samples/model; {args.rolling_steps}-step moving average",
        fontsize=14,
    )
    figure_path = figures_dir / "dense_mofe_upcycling_training_loss_ppl.png"
    fig.savefig(figure_path)
    plt.close(fig)

    summary = {
        "samples_per_model": len(steps),
        "logging_interval_steps": logging_interval,
        "effective_batch_samples": 32,
        "rolling_steps": args.rolling_steps,
        "final_raw": {
            key: {
                "loss": float(values[-1]),
                "ppl": math.exp(float(values[-1])),
            }
            for key, values in losses.items()
        },
        "final_smoothed": {
            key: {
                "loss": float(values[-1]),
                "ppl": math.exp(float(values[-1])),
            }
            for key, values in smooth_losses.items()
        },
    }
    summary_path = output_dir / "dense_mofe_upcycling_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
