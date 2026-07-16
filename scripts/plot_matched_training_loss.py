import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-log", required=True)
    parser.add_argument("--mofe-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-steps", type=int, default=50)
    parser.add_argument("--phase-boundary", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trailing_average(
    steps: np.ndarray, values: np.ndarray, points: int
) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones(points, dtype=np.float64) / points
    return steps[points - 1 :], np.convolve(values, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    if args.rolling_steps <= 0:
        raise ValueError("rolling steps must be positive")
    dense = read_jsonl(args.dense_log)
    mofe = read_jsonl(args.mofe_log)
    dense_by_step = {record["step"]: record for record in dense}
    mofe_by_step = {record["step"]: record for record in mofe}
    steps = sorted(set(dense_by_step) & set(mofe_by_step))
    if steps != sorted(dense_by_step) or steps != sorted(mofe_by_step):
        raise ValueError("Dense and MoFE logs must contain identical steps")
    if any(
        record.get("effective_batch_samples") != 32 for record in dense + mofe
    ):
        raise ValueError("training loss logs must use 32-sample effective batches")

    step_values = np.array(steps)
    dense_loss = np.array([dense_by_step[step]["lm_loss"] for step in steps])
    mofe_loss = np.array([mofe_by_step[step]["lm_loss"] for step in steps])
    logging_interval = int(np.median(np.diff(step_values)))
    rolling_points = max(1, args.rolling_steps // logging_interval)
    dense_smooth_x, dense_smooth = trailing_average(
        step_values, dense_loss, rolling_points
    )
    mofe_smooth_x, mofe_smooth = trailing_average(
        step_values, mofe_loss, rolling_points
    )

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dense_mofe_training_loss.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("step", "dense_lm_loss", "mofe_lm_loss")
        )
        writer.writeheader()
        writer.writerows(
            {
                "step": step,
                "dense_lm_loss": dense_by_step[step]["lm_loss"],
                "mofe_lm_loss": mofe_by_step[step]["lm_loss"],
            }
            for step in steps
        )

    colors = ("#3778A8", "#D27A3F")
    fig, ax = plt.subplots(figsize=(11.5, 6), dpi=170, layout="constrained")
    ax.axvspan(0, args.phase_boundary, color="#80868B", alpha=0.1)
    ax.axvline(
        args.phase_boundary, color="#555555", linestyle="--", linewidth=1.2
    )
    ax.plot(
        step_values,
        dense_loss,
        color=colors[0],
        alpha=0.2,
        linewidth=0.9,
        label=f"Dense raw ({len(dense_loss)} points)",
    )
    ax.plot(
        step_values,
        mofe_loss,
        color=colors[1],
        alpha=0.2,
        linewidth=0.9,
        label=f"MoFE raw ({len(mofe_loss)} points)",
    )
    ax.plot(
        dense_smooth_x,
        dense_smooth,
        color=colors[0],
        linewidth=2.3,
        label=f"Dense {args.rolling_steps}-step moving average",
    )
    ax.plot(
        mofe_smooth_x,
        mofe_smooth,
        color=colors[1],
        linewidth=2.3,
        label=f"MoFE {args.rolling_steps}-step moving average",
    )
    ax.set_title("Dense vs MoFE: Global Effective-batch Training Loss")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Language-modeling loss")
    ax.set_xlim(0, max(steps))
    ax.grid(alpha=0.23)
    ax.legend(ncol=2)
    figure_path = figures_dir / "dense_vs_mofe_training_loss.png"
    fig.savefig(figure_path)
    plt.close(fig)

    first_mask = step_values <= 100
    last_mask = step_values >= 900
    post_warmup = step_values >= args.phase_boundary
    summary = {
        "logged_points_per_model": len(steps),
        "logging_interval_steps": logging_interval,
        "effective_batch_samples": 32,
        "rolling_steps": args.rolling_steps,
        "dense_mean_steps_1_100": float(dense_loss[first_mask].mean()),
        "mofe_mean_steps_1_100": float(mofe_loss[first_mask].mean()),
        "dense_mean_steps_900_1000": float(dense_loss[last_mask].mean()),
        "mofe_mean_steps_900_1000": float(mofe_loss[last_mask].mean()),
        "dense_post_boundary_slope_per_100_steps": float(
            np.polyfit(step_values[post_warmup], dense_loss[post_warmup], 1)[0]
            * 100
        ),
        "mofe_post_boundary_slope_per_100_steps": float(
            np.polyfit(step_values[post_warmup], mofe_loss[post_warmup], 1)[0]
            * 100
        ),
    }
    summary_path = output_dir / "dense_mofe_training_loss_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved training-loss CSV: {csv_path}")
    print(f"Saved training-loss figure: {figure_path}")


if __name__ == "__main__":
    main()
