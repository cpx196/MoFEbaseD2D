from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


MODELS = ("Dense", "MoFE group LR", "Upcycling")
COLORS = {
    "Dense": "#3F7FA6",
    "MoFE group LR": "#D87958",
    "Upcycling": "#8E6FAE",
}
LINESTYLES = {
    "Dense": "-",
    "MoFE group LR": (0, (12, 4, 2, 4)),
    "Upcycling": (0, (1, 3)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot merged FineWeb-Edu validation-loss curves."
    )
    parser.add_argument("--initial-csv", required=True)
    parser.add_argument("--continuation-dir", action="append", default=[])
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-figure", required=True)
    return parser.parse_args()


def continuation_model(path: Path) -> str:
    name = path.parent.name
    if name.startswith("dense_"):
        return "Dense"
    if name.startswith("mofe_group_lr_"):
        return "MoFE group LR"
    if name.startswith("upcycling_"):
        return "Upcycling"
    raise ValueError(f"unknown continuation run: {path}")


def load_records(
    initial_csv: Path, continuation_dirs: list[Path], max_step: int | None
) -> list[dict]:
    records: dict[tuple[str, int], dict] = {}
    with initial_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in MODELS:
                continue
            step = int(row["step"])
            if max_step is not None and step > max_step:
                continue
            records[(row["model"], step)] = {
                "model": row["model"],
                "step": step,
                "tokens_B": float(row["tokens_B"]),
                "validation_loss": float(row["validation_loss"]),
                "perplexity": float(row["perplexity"]),
            }

    for continuation_dir in continuation_dirs:
        for path in sorted(continuation_dir.glob("*/validation_log.jsonl")):
            model = continuation_model(path)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                step = int(row["step"])
                if step <= 50000 or (max_step is not None and step > max_step):
                    continue
                records[(model, step)] = {
                    "model": model,
                    "step": step,
                    "tokens_B": step * 32768 / 1e9,
                    "validation_loss": float(row["lm_loss"]),
                    "perplexity": float(row["perplexity"]),
                }

    return sorted(
        records.values(), key=lambda row: (MODELS.index(row["model"]), row["step"])
    )


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "step", "tokens_B", "validation_loss", "perplexity"],
        )
        writer.writeheader()
        writer.writerows(records)


def plot(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 25,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 24,
        }
    )
    fig, ax = plt.subplots(figsize=(18, 10.75), dpi=116)
    grouped = {model: [r for r in records if r["model"] == model] for model in MODELS}

    for model in MODELS:
        series = grouped[model]
        ax.plot(
            [r["tokens_B"] for r in series],
            [r["validation_loss"] for r in series],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=4.5,
            solid_capstyle="round",
            dash_capstyle="round",
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            linewidth=4.5,
            marker=marker,
            markersize=12,
            markerfacecolor=COLORS[model],
            markeredgecolor=COLORS[model],
            label=model.removesuffix(" group LR") if model == "MoFE group LR" else model,
        )
        for model, marker in zip(MODELS, ("o", "s", "^"))
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=3,
        frameon=True,
        fancybox=True,
        borderpad=0.35,
        handlelength=3.6,
        columnspacing=2.0,
    )

    ax.set_xlabel("Training tokens (B)", labelpad=12)
    ax.set_ylabel("Validation loss on FineWeb-Edu 10BT", labelpad=12)
    max_tokens = max(r["tokens_B"] for r in records)
    ax.set_xlim(0, max_tokens * 1.025)
    ax.set_xticks(np.arange(0, max_tokens + 0.01, 0.4))
    ax.set_ylim(3.06, 3.31)
    ax.set_yticks(np.arange(3.10, 3.31, 0.05))
    ax.grid(axis="y", color="#D0D0D0", linewidth=1.7)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(2.0)
    ax.tick_params(axis="both", colors="#111111", width=1.5, length=7)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.84)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(
        Path(args.initial_csv),
        [Path(path) for path in args.continuation_dir],
        args.max_step,
    )
    write_csv(records, Path(args.output_csv))
    plot(records, Path(args.output_figure))
    print(f"Saved {len(records)} records")
    print(f"Saved figure: {args.output_figure}")


if __name__ == "__main__":
    main()
