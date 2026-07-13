import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DOWNSTREAM_METRICS = {
    "lambada_openai": "acc",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--lm-eval-dir", required=True)
    parser.add_argument("--wikitext-json", required=True)
    parser.add_argument("--d2d-commit-file", required=True)
    return parser.parse_args()


def newest_json(directory: Path) -> Path:
    paths = sorted(directory.rglob("*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"No JSON files found below {directory}")
    return paths[-1]


def metric_value(task_data: dict, metric: str) -> float:
    for key, value in task_data.items():
        if key.split(",")[0] == metric and not key.endswith("_stderr"):
            return float(value)
    raise KeyError(f"Metric {metric!r} not found in {task_data.keys()}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    lm_eval_json = newest_json(Path(args.lm_eval_dir))
    lm_eval = json.loads(lm_eval_json.read_text())
    wikitext = json.loads(Path(args.wikitext_json).read_text())
    d2d_commit = Path(args.d2d_commit_file).read_text().strip()

    task_scores = {}
    rows = []
    for task, metric in DOWNSTREAM_METRICS.items():
        score = metric_value(lm_eval["results"][task], metric)
        task_scores[task] = {"metric": metric, "score": score}
        rows.append(
            {
                "experiment": "dense_baseline",
                "model": "openai-community/gpt2",
                "moe_layers": 0,
                "num_experts": 0,
                "top_k": 0,
                "task": task,
                "metric": metric,
                "score": score,
                "precision": wikitext["precision"],
                "d2d_commit": d2d_commit,
            }
        )

    rows.append(
        {
            "experiment": "dense_baseline",
            "model": "openai-community/gpt2",
            "moe_layers": 0,
            "num_experts": 0,
            "top_k": 0,
            "task": "wikitext2_validation",
            "metric": "perplexity",
            "score": float(wikitext["perplexity"]),
            "precision": wikitext["precision"],
            "d2d_commit": d2d_commit,
        }
    )

    csv_path = results_dir / "metrics_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = list(DOWNSTREAM_METRICS)
    values = [task_scores[label]["score"] for label in labels]
    pretty_labels = [label.replace("_", "\n") for label in labels]
    colors = ["#2E6F95", "#D08C60", "#4D9078", "#B95F5F", "#6C6FAE", "#8A7A3D"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    bars = ax.bar(pretty_labels, values, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_title("GPT-2 small dense baseline")
    ax.set_ylabel("Zero-shot score")
    ax.set_xlabel("Task")
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    for bar, task in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{bar.get_height():.3f}\n{task_scores[task]['metric']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    downstream_path = figures_dir / "downstream_scores.png"
    fig.savefig(downstream_path)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5.5), dpi=160)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.28)
    ax_ppl = fig.add_subplot(grid[0, 0])
    ax_scores = fig.add_subplot(grid[0, 1])

    ax_ppl.bar(["WikiText-2\nvalidation"], [wikitext["perplexity"]], color="#2E6F95")
    ax_ppl.set_title("Validation Perplexity")
    ax_ppl.set_ylabel("Perplexity")
    ax_ppl.text(
        0,
        wikitext["perplexity"],
        f"{wikitext['perplexity']:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    ax_ppl.set_ylim(0, wikitext["perplexity"] * 1.22)

    ax_scores.bar(pretty_labels, values, color=colors, edgecolor="#222222", linewidth=0.6)
    ax_scores.set_title("Zero-shot Downstream Scores")
    ax_scores.set_ylabel("Score")
    ax_scores.set_ylim(0, max(1.0, max(values) * 1.15))
    for index, value in enumerate(values):
        ax_scores.text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("GPT-2 small, 124M, dense, zero-shot", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    overview_path = figures_dir / "baseline_overview.png"
    fig.savefig(overview_path)
    plt.close(fig)

    print(f"lm_eval_json: {lm_eval_json}")
    print(f"metrics_summary: {csv_path}")
    print(f"downstream_scores: {downstream_path}")
    print(f"baseline_overview: {overview_path}")


if __name__ == "__main__":
    main()
