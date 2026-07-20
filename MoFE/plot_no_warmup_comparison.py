from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


TASK_METRICS = {
    "lambada_openai": "acc",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-csv", required=True)
    parser.add_argument("--mofe-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metric_value(task_data: dict[str, float], metric: str) -> float:
    for key, value in task_data.items():
        if key.split(",")[0] == metric and not key.endswith("_stderr"):
            return float(value)
    raise KeyError(f"Metric {metric!r} not found in {task_data}")


def load_dense_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            if task in TASK_METRICS:
                scores[task] = float(row["score"])
    missing = set(TASK_METRICS) - set(scores)
    if missing:
        raise ValueError(f"Dense CSV is missing tasks: {sorted(missing)}")
    return scores


def main() -> None:
    args = parse_args()
    dense_scores = load_dense_scores(Path(args.dense_csv))
    mofe_results = json.loads(Path(args.mofe_json).read_text())
    mofe_scores = {
        task: metric_value(mofe_results["results"][task], metric)
        for task, metric in TASK_METRICS.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, metric in TASK_METRICS.items():
        dense = dense_scores[task]
        mofe = mofe_scores[task]
        rows.append(
            {
                "task": task,
                "metric": metric,
                "dense_score": dense,
                "mofe_no_warmup_score": mofe,
                "absolute_delta": mofe - dense,
                "relative_delta_percent": 100.0 * (mofe - dense) / dense,
            }
        )

    csv_path = output_dir / "no_warmup_vs_dense.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = [task.replace("_", "\n") for task in TASK_METRICS]
    dense_values = [dense_scores[task] for task in TASK_METRICS]
    mofe_values = [mofe_scores[task] for task in TASK_METRICS]
    positions = list(range(len(labels)))
    width = 0.36

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    dense_bars = ax.bar(
        [position - width / 2 for position in positions],
        dense_values,
        width,
        label="Dense GPT-2 Small",
        color="#2E6F95",
        edgecolor="#222222",
        linewidth=0.5,
    )
    mofe_bars = ax.bar(
        [position + width / 2 for position in positions],
        mofe_values,
        width,
        label="MoFE initialized, no warmup",
        color="#C65D3B",
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.set_title("Dense GPT-2 vs. initialized MoFE without warmup")
    ax.set_ylabel("Zero-shot score")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, max(dense_values + mofe_values) * 1.22)
    ax.legend(frameon=False)
    for bars in (dense_bars, mofe_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    figure_path = output_dir / "no_warmup_vs_dense.png"
    fig.savefig(figure_path)
    plt.close(fig)

    print(f"Saved comparison CSV to {csv_path}")
    print(f"Saved comparison figure to {figure_path}")


if __name__ == "__main__":
    main()
