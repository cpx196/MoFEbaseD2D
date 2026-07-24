from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


MODELS = ("Dense", "MoFE group LR", "Upcycling")
COLORS = {
    "Dense": "#3F7FA6",
    "MoFE group LR": "#D87958",
    "Upcycling": "#8E6FAE",
}
LINESTYLES = {
    "Dense": "-",
    "MoFE group LR": "-.",
    "Upcycling": ":",
}
MARKERS = {
    "Dense": "o",
    "MoFE group LR": "s",
    "Upcycling": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation loss and next-token accuracy on two y axes."
    )
    parser.add_argument("--loss-csv", required=True)
    parser.add_argument("--accuracy-csv", required=True)
    parser.add_argument("--output-figure", required=True)
    parser.add_argument("--max-step", type=int)
    return parser.parse_args()


def read_csv(path: Path, max_step: int | None) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_step is not None:
        rows = [row for row in rows if int(row["step"]) <= max_step]
    return rows


def plot(
    loss_rows: list[dict], accuracy_rows: list[dict], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 23,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 20,
        }
    )
    fig, loss_ax = plt.subplots(figsize=(18, 10.75), dpi=160)
    accuracy_ax = loss_ax.twinx()

    for model in MODELS:
        model_loss = [row for row in loss_rows if row["model"] == model]
        model_accuracy = [row for row in accuracy_rows if row["model"] == model]
        model_loss.sort(key=lambda row: int(row["step"]))
        model_accuracy.sort(key=lambda row: int(row["step"]))

        loss_ax.plot(
            [float(row["tokens_B"]) for row in model_loss],
            [float(row["validation_loss"]) for row in model_loss],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=3.4,
            alpha=0.98,
        )

        accuracy_x = np.array(
            [float(row["btokens"]) for row in model_accuracy], dtype=float
        )
        accuracy_y = np.array(
            [float(row["token_accuracy"]) for row in model_accuracy], dtype=float
        )
        accuracy_ax.scatter(
            accuracy_x,
            accuracy_y,
            color=COLORS[model],
            marker=MARKERS[model],
            s=86,
            alpha=0.42,
            edgecolors="none",
            zorder=4,
        )
        degree = min(3, len(accuracy_x) - 1)
        coefficients = np.polyfit(accuracy_x, accuracy_y, degree)
        fit_x = np.linspace(accuracy_x.min(), accuracy_x.max(), 320)
        accuracy_ax.plot(
            fit_x,
            np.polyval(coefficients, fit_x),
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=4.5,
            alpha=0.98,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=4.0,
            marker=MARKERS[model],
            markersize=10,
            label="MoFE" if model == "MoFE group LR" else model,
        )
        for model in MODELS
    ]
    loss_ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=3,
        frameon=True,
        handlelength=3.4,
        columnspacing=2.0,
    )

    max_tokens = max(float(row["tokens_B"]) for row in loss_rows)
    all_losses = [float(row["validation_loss"]) for row in loss_rows]
    all_accuracies = [float(row["token_accuracy"]) for row in accuracy_rows]
    loss_ax.set_xlim(0, max_tokens * 1.04)
    loss_ax.set_ylim(min(all_losses) - 0.01, max(all_losses) + 0.01)
    accuracy_margin = max((max(all_accuracies) - min(all_accuracies)) * 0.16, 0.001)
    accuracy_ax.set_ylim(
        min(all_accuracies) - accuracy_margin,
        max(all_accuracies) + accuracy_margin,
    )
    accuracy_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

    loss_ax.set_xlabel("Training tokens (B)", labelpad=12)
    loss_ax.set_ylabel("Validation loss on FineWeb-Edu 10BT", labelpad=12)
    accuracy_ax.set_ylabel(
        "Next-token prediction accuracy on FineWeb-Edu 10BT", labelpad=18
    )
    loss_ax.grid(axis="y", color="#D0D0D0", linewidth=1.4)
    loss_ax.grid(axis="x", visible=False)
    for axis in (loss_ax, accuracy_ax):
        for spine in axis.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(1.7)
        axis.tick_params(axis="both", colors="#111111", width=1.4, length=7)

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.84)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    loss_rows = read_csv(Path(args.loss_csv), args.max_step)
    accuracy_rows = read_csv(Path(args.accuracy_csv), args.max_step)
    if not loss_rows or not accuracy_rows:
        raise ValueError("loss and accuracy CSV files must both contain records")
    plot(loss_rows, accuracy_rows, Path(args.output_figure))
    print(f"Saved figure: {args.output_figure}")


if __name__ == "__main__":
    main()
