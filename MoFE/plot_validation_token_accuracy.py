from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter


MODEL_ORDER = ("Dense", "MoFE group LR", "Upcycling")
COLORS = {
    "Dense": "#3F7FA6",
    "MoFE group LR": "#D87958",
    "Upcycling": "#8E6FAE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot validation next-token prediction accuracy."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-figure", required=True)
    parser.add_argument("--output-fit-figure")
    parser.add_argument("--output-md")
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    return parser.parse_args()


def format_k(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1:
        return "0"
    if value % 1000 == 0:
        return f"{int(value / 1000)}k"
    return f"{value/1000:.1f}k"


def infer_model_name(path: Path, record: dict) -> str:
    name = path.name.lower()
    if name.startswith("dense_"):
        return "Dense"
    if name.startswith("mofe_group_lr_") or "group_lr" in name:
        return "MoFE group LR"
    if name.startswith("upcycling_"):
        return "Upcycling"
    model_type = record.get("model_type")
    if model_type == "dense":
        return "Dense"
    if model_type == "upcycling":
        return "Upcycling"
    return "MoFE group LR" if model_type == "mofe" else str(model_type)


def load_records(results_dir: Path, tokens_per_step: int) -> list[dict]:
    records = []
    for path in sorted(results_dir.rglob("*.json")):
        record = json.loads(path.read_text())
        step = int(record["step"])
        model = infer_model_name(path, record)
        records.append(
            {
                "model": model,
                "step": step,
                "btokens": step * tokens_per_step / 1e9,
                "token_accuracy": float(record["token_accuracy"]),
                "negative_log_likelihood": float(record["negative_log_likelihood"]),
                "perplexity": float(record["perplexity"]),
                "correct_tokens": int(record["correct_tokens"]),
                "token_count": int(record["token_count"]),
                "path": str(path),
            }
        )
    return sorted(
        records,
        key=lambda item: (
            MODEL_ORDER.index(item["model"])
            if item["model"] in MODEL_ORDER
            else len(MODEL_ORDER),
            item["step"],
        ),
    )


def write_csv(records: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "step",
        "btokens",
        "token_accuracy",
        "negative_log_likelihood",
        "perplexity",
        "correct_tokens",
        "token_count",
        "path",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot(records: list[dict], output_figure: Path) -> None:
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=180)

    grouped = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)

    for model in MODEL_ORDER:
        series = grouped.get(model)
        if not series:
            continue
        steps = [item["step"] for item in series]
        accs = [item["token_accuracy"] for item in series]
        ax.scatter(
            steps,
            accs,
            s=82,
            color=COLORS.get(model),
            edgecolors="none",
            label=model,
        )

    dense_series = grouped.get("Dense", [])
    if dense_series:
        base = min(dense_series, key=lambda item: item["step"])["token_accuracy"]
        ax.axhline(
            base,
            color=COLORS["Dense"],
            linestyle="--",
            linewidth=1.8,
            alpha=0.55,
        )
        ax.text(
            max(item["step"] for item in records) + 950,
            base,
            "Base",
            va="center",
            ha="left",
            fontsize=13,
            color="#222222",
        )

    ax.set_xlabel("Optimizer step (k)")
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax.set_ylabel("Next-token prediction accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlim(2500, max(item["step"] for item in records) + 4500)
    ymin = min(item["token_accuracy"] for item in records)
    ymax = max(item["token_accuracy"] for item in records)
    margin = max((ymax - ymin) * 0.18, 0.001)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.legend(loc="upper left", frameon=True, fontsize=11)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")
    fig.tight_layout()
    fig.savefig(output_figure, bbox_inches="tight")
    plt.close(fig)


def plot_fit(records: list[dict], output_figure: Path) -> None:
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=180)

    grouped = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)

    x_max = max(item["step"] for item in records)
    for model in MODEL_ORDER:
        series = grouped.get(model)
        if not series:
            continue
        xs = np.array([item["step"] for item in series], dtype=float)
        ys = np.array([item["token_accuracy"] for item in series], dtype=float)
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        degree = min(3, len(xs) - 1)
        coeffs = np.polyfit(xs, ys, degree)
        fit_xs = np.linspace(xs.min(), xs.max(), 240)
        fit_ys = np.polyval(coeffs, fit_xs)
        ax.plot(
            fit_xs,
            fit_ys,
            linewidth=2.8,
            color=COLORS.get(model),
            label=model,
        )
        ax.scatter(
            xs,
            ys,
            s=42,
            color=COLORS.get(model),
            edgecolors="none",
            alpha=0.45,
        )

    dense_series = grouped.get("Dense", [])
    if dense_series:
        base = min(dense_series, key=lambda item: item["step"])["token_accuracy"]
        ax.axhline(
            base,
            color=COLORS["Dense"],
            linestyle="--",
            linewidth=1.8,
            alpha=0.55,
        )
        ax.text(
            x_max + 950,
            base,
            "Base",
            va="center",
            ha="left",
            fontsize=13,
            color="#222222",
        )

    ax.set_xlabel("Optimizer step (k)")
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax.set_ylabel("Next-token prediction accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlim(2500, x_max + 4500)
    ymin = min(item["token_accuracy"] for item in records)
    ymax = max(item["token_accuracy"] for item in records)
    margin = max((ymax - ymin) * 0.18, 0.001)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.legend(loc="upper left", frameon=True, fontsize=11)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")
    fig.tight_layout()
    fig.savefig(output_figure, bbox_inches="tight")
    plt.close(fig)


def write_md(
    records: list[dict], output_md: Path, output_figure: Path, output_fit_figure: Path | None
) -> None:
    figure_link = os.path.relpath(output_figure, start=output_md.parent)
    fit_figure_link = (
        os.path.relpath(output_fit_figure, start=output_md.parent)
        if output_fit_figure is not None
        else None
    )
    best_by_model = {}
    for record in records:
        current = best_by_model.get(record["model"])
        if current is None or record["token_accuracy"] > current["token_accuracy"]:
            best_by_model[record["model"]] = record

    lines = [
        "# Validation Token Prediction Accuracy",
        "",
        f"![Validation token prediction accuracy]({figure_link})",
        "",
    ]
    if fit_figure_link is not None:
        lines.extend(
            [
                f"![Validation token prediction accuracy smooth fit]({fit_figure_link})",
                "",
            ]
        )
    lines.extend(
        [
            "| Model | Final step | Final acc | Final val loss | Best step | Best acc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODEL_ORDER:
        series = [record for record in records if record["model"] == model]
        if not series:
            continue
        final = max(series, key=lambda item: item["step"])
        best = best_by_model[model]
        lines.append(
            f"| {model} | {final['step']:,} | {final['token_accuracy']:.6f} | "
            f"{final['negative_log_likelihood']:.6f} | {best['step']:,} | "
            f"{best['token_accuracy']:.6f} |"
        )
    by_step = {
        step: {
            record["model"]: record
            for record in records
            if record["step"] == step
        }
        for step in sorted({record["step"] for record in records})
    }
    lines.extend(
        [
            "",
            "## All Checkpoints",
            "",
            "| Step | Training tokens (B) | Dense | MoFE group LR | Upcycling |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for step, step_records in by_step.items():
        tokens = next(iter(step_records.values()))["btokens"]
        values = [
            f"{step_records[model]['token_accuracy'] * 100:.4f}%"
            if model in step_records
            else "-"
            for model in MODEL_ORDER
        ]
        lines.append(
            f"| {step // 1000}k | {tokens:.5f} | "
            f"{values[0]} | {values[1]} | {values[2]} |"
        )
    lines.extend(
        [
            "",
            "Accuracy is computed as `argmax(logits[:, :-1]) == labels[:, 1:]`, ignoring padding labels.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    records = load_records(results_dir, args.tokens_per_step)
    if not records:
        raise ValueError(f"no JSON results found under {results_dir}")

    output_csv = Path(args.output_csv)
    output_figure = Path(args.output_figure)
    write_csv(records, output_csv)
    plot(records, output_figure)
    if args.output_fit_figure:
        plot_fit(records, Path(args.output_fit_figure))
    if args.output_md:
        write_md(
            records,
            Path(args.output_md),
            output_figure,
            Path(args.output_fit_figure) if args.output_fit_figure else None,
        )
    print(f"Saved {len(records)} records to {output_csv}")
    print(f"Saved figure to {output_figure}")


if __name__ == "__main__":
    main()
