import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-stage1", required=True)
    parser.add_argument("--dense-stage2", required=True)
    parser.add_argument("--mofe-stage1", required=True)
    parser.add_argument("--mofe-stage2", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-points", type=int, default=10)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def combine(stage1: str, stage2: str) -> list[dict]:
    records = read_jsonl(stage1) + read_jsonl(stage2)
    by_step = {int(record["step"]): record for record in records}
    return [by_step[step] for step in sorted(by_step)]


def trailing_average(
    steps: np.ndarray, values: np.ndarray, points: int
) -> tuple[np.ndarray, np.ndarray]:
    if points <= 1:
        return steps, values
    kernel = np.ones(points, dtype=np.float64) / points
    return steps[points - 1 :], np.convolve(values, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    if args.rolling_points <= 0:
        raise ValueError("rolling points must be positive")
    dense = combine(args.dense_stage1, args.dense_stage2)
    mofe = combine(args.mofe_stage1, args.mofe_stage2)
    dense_by_step = {int(record["step"]): record for record in dense}
    mofe_by_step = {int(record["step"]): record for record in mofe}
    steps = sorted(set(dense_by_step) & set(mofe_by_step))
    if not steps:
        raise ValueError("dense and MoFE logs have no common steps")

    dense_loss = np.array(
        [dense_by_step[step]["lm_loss"] for step in steps], dtype=np.float64
    )
    mofe_loss = np.array(
        [mofe_by_step[step]["lm_loss"] for step in steps], dtype=np.float64
    )
    step_values = np.array(steps)
    dense_smooth_x, dense_smooth = trailing_average(
        step_values, dense_loss, args.rolling_points
    )
    mofe_smooth_x, mofe_smooth = trailing_average(
        step_values, mofe_loss, args.rolling_points
    )

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "step1500_lm_loss_series.csv"
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

    fig, ax = plt.subplots(figsize=(11, 6), dpi=170, layout="constrained")
    dense_color = "#3778A8"
    mofe_color = "#D27A3F"
    ax.plot(
        step_values,
        dense_loss,
        color=dense_color,
        alpha=0.18,
        linewidth=0.9,
    )
    ax.plot(
        step_values,
        mofe_loss,
        color=mofe_color,
        alpha=0.18,
        linewidth=0.9,
    )
    ax.plot(
        dense_smooth_x,
        dense_smooth,
        color=dense_color,
        linewidth=2.2,
        label="Dense, 50-step moving average",
    )
    ax.plot(
        mofe_smooth_x,
        mofe_smooth,
        color=mofe_color,
        linewidth=2.2,
        label="MoFE, 50-step moving average",
    )
    ax.axvline(300, color="#555555", linestyle="--", linewidth=1.2)
    ax.text(
        312,
        ax.get_ylim()[1] - 0.03,
        "LR schedule restart",
        color="#444444",
        va="top",
        fontsize=9,
    )
    ax.set_title("Training LM Loss: Dense vs MoFE (1500 Steps)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Language modeling loss")
    ax.set_xlim(0, 1500)
    ax.grid(alpha=0.22)
    ax.legend()
    figure_path = figures_dir / "step1500_dense_vs_mofe_lm_loss.png"
    fig.savefig(figure_path)
    plt.close(fig)

    summary = {
        "dense_mean_steps_1_300": float(dense_loss[step_values <= 300].mean()),
        "mofe_mean_steps_1_300": float(mofe_loss[step_values <= 300].mean()),
        "dense_mean_steps_305_1500": float(dense_loss[step_values > 300].mean()),
        "mofe_mean_steps_305_1500": float(mofe_loss[step_values > 300].mean()),
        "dense_mean_last_100_steps": float(dense_loss[step_values >= 1400].mean()),
        "mofe_mean_last_100_steps": float(mofe_loss[step_values >= 1400].mean()),
    }
    summary_path = output_dir / "step1500_lm_loss_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved loss CSV: {csv_path}")
    print(f"Saved loss figure: {figure_path}")


if __name__ == "__main__":
    main()
