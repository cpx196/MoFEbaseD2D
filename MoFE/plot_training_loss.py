from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def moving_average(values: list[float], window: int = 3) -> list[float]:
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        selected = values[start : index + 1]
        smoothed.append(sum(selected) / len(selected))
    return smoothed


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in Path(args.training_log).read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("training log contains no records")

    steps = [int(record["step"]) for record in records]
    total_losses = [float(record["total_loss"]) for record in records]
    trend = moving_average(total_losses)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax.plot(
        steps,
        total_losses,
        color="#26788E",
        linewidth=1.4,
        marker="o",
        markersize=3.5,
        label="Total loss",
    )
    ax.plot(
        steps,
        trend,
        color="#D07A32",
        linewidth=2.2,
        label="3-point moving average",
    )
    ax.scatter(
        [steps[0], steps[-1]],
        [total_losses[0], total_losses[-1]],
        color="#222222",
        s=28,
        zorder=3,
    )
    ax.annotate(
        f"{total_losses[0]:.3f}",
        (steps[0], total_losses[0]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )
    ax.annotate(
        f"{total_losses[-1]:.3f}",
        (steps[-1], total_losses[-1]),
        xytext=(-34, -18),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_title(f"MoFE GPT-2: {max(steps)}-Step Training Loss")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Total loss")
    ax.set_xlim(0, max(steps) + 2)
    ax.legend(frameon=False)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training loss figure to {output}")


if __name__ == "__main__":
    main()
