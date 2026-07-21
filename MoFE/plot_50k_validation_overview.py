from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter


MODELS = ("Dense", "MoFE group LR", "Upcycling")
COLORS = {
    "Dense": "#3F7FA6",
    "MoFE group LR": "#D87958",
    "Upcycling": "#8E6FAE",
}


def format_k(value: float, _pos: int | None = None) -> str:
    return "0" if value == 0 else f"{value / 1000:g}k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a combined 50k validation-loss and token-accuracy figure."
    )
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--accuracy-results-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_validation(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = {model: [] for model in MODELS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            if model in grouped:
                grouped[model].append((float(row["step"]), float(row["validation_loss"])))
    return {
        model: (np.array([item[0] for item in values]), np.array([item[1] for item in values]))
        for model, values in grouped.items()
        if values
    }


def infer_model_name(path: Path, record: dict) -> str:
    name = path.name.lower()
    if name.startswith("dense_"):
        return "Dense"
    if "group_lr" in name:
        return "MoFE group LR"
    if name.startswith("upcycling_"):
        return "Upcycling"
    return str(record.get("model_type", ""))


def load_accuracy(results_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = {model: [] for model in MODELS}
    for path in results_dir.rglob("*.json"):
        record = json.loads(path.read_text())
        model = infer_model_name(path, record)
        if model in grouped:
            grouped[model].append((float(record["step"]), float(record["token_accuracy"])))
    return {
        model: tuple(map(np.array, zip(*sorted(values))))
        for model, values in grouped.items()
        if values
    }


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax.set_xlim(0, 52000)


def main() -> None:
    args = parse_args()
    validation = load_validation(Path(args.validation_csv))
    accuracy = load_accuracy(Path(args.accuracy_results_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=180)

    for model in MODELS:
        if model not in validation:
            continue
        steps, losses = validation[model]
        loss_ax.plot(steps, losses, color=COLORS[model], linewidth=2.6, label=model)
    loss_ax.set_xlabel("Optimizer step (k)")
    loss_ax.set_ylabel("Validation loss")
    style_axis(loss_ax)
    loss_ax.legend(loc="upper right", frameon=True, fontsize=10)

    all_accs = []
    for model in MODELS:
        if model not in accuracy:
            continue
        steps, accs = accuracy[model]
        degree = min(3, len(steps) - 1)
        fit_x = np.linspace(steps.min(), steps.max(), 240)
        fit_y = np.polyval(np.polyfit(steps, accs, degree), fit_x)
        acc_ax.plot(fit_x, fit_y, color=COLORS[model], linewidth=2.8, label=model)
        acc_ax.scatter(steps, accs, s=42, color=COLORS[model], edgecolors="none", alpha=0.45)
        all_accs.extend(accs)
    base = accuracy["Dense"][1][0]
    acc_ax.axhline(base, color=COLORS["Dense"], linestyle="--", linewidth=1.6, alpha=0.55)
    acc_ax.text(51500, base, "Base", va="center", ha="right", fontsize=10, color="#222222")
    margin = max((max(all_accs) - min(all_accs)) * 0.18, 0.001)
    acc_ax.set_ylim(min(all_accs) - margin, max(all_accs) + margin)
    acc_ax.set_xlabel("Optimizer step (k)")
    acc_ax.set_ylabel("Next-token prediction accuracy")
    acc_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    style_axis(acc_ax)
    acc_ax.legend(loc="upper left", frameon=True, fontsize=10)

    fig.tight_layout(w_pad=2.7)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved combined figure to {output}")


if __name__ == "__main__":
    main()
