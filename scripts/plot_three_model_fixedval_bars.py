import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-log", required=True)
    parser.add_argument("--mofe-log", required=True)
    parser.add_argument("--upcycling-log", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def final_record(path: str) -> dict:
    with Path(path).open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"empty validation log: {path}")
    return records[-1]


def main() -> None:
    args = parse_args()
    inputs = (
        ("Dense", final_record(args.dense_log)),
        ("MoFE", final_record(args.mofe_log)),
        ("Upcycling", final_record(args.upcycling_log)),
    )
    steps = {record["step"] for _, record in inputs}
    token_counts = {record["token_count"] for _, record in inputs}
    sample_counts = {record["sample_count"] for _, record in inputs}
    if len(steps) != 1:
        raise ValueError(f"final optimizer steps differ: {steps}")
    if len(token_counts) != 1 or len(sample_counts) != 1:
        raise ValueError("fixed validation sets differ between models")

    rows = [
        {
            "model": name,
            "step": record["step"],
            "validation_loss": record["lm_loss"],
            "perplexity": record["perplexity"],
            "token_count": record["token_count"],
            "sample_count": record["sample_count"],
        }
        for name, record in inputs
    ]
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dense_mofe_upcycling_step1000.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["model"] for row in rows]
    colors = ("#4778A8", "#16856B", "#D97706")
    fig, (ax_loss, ax_ppl) = plt.subplots(
        1, 2, figsize=(11.5, 5), dpi=180, layout="constrained"
    )
    metrics = (
        (ax_loss, "validation_loss", "Validation Loss", "Loss (lower is better)"),
        (ax_ppl, "perplexity", "Validation Perplexity", "PPL (lower is better)"),
    )
    for ax, key, title, ylabel in metrics:
        values = [row[key] for row in rows]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            decimals = 6 if key == "validation_loss" else 4
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.{decimals}f}",
                ha="center",
                fontsize=10,
            )
    step = steps.pop()
    fig.suptitle(
        f"Dense vs MoFE vs Upcycling at Step {step}\n"
        "Fixed WikiText-103 Validation",
        fontsize=14,
    )
    figure_path = figures_dir / "dense_mofe_upcycling_loss_ppl.png"
    fig.savefig(figure_path)
    plt.close(fig)

    summary = {
        "step": step,
        "validation_token_count": token_counts.pop(),
        "validation_sample_count": sample_counts.pop(),
        "models": rows,
    }
    summary_path = output_dir / "dense_mofe_upcycling_step1000.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
