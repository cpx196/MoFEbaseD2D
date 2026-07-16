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
    parser.add_argument("--dense-csv", required=True)
    parser.add_argument("--mofe-lm-eval", required=True)
    parser.add_argument("--mofe-wikitext", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metric_value(task_data: dict, metric: str) -> float:
    for key, value in task_data.items():
        if key.split(",")[0] == metric and not key.endswith("_stderr"):
            return float(value)
    raise KeyError(f"Metric {metric!r} not found in {task_data.keys()}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.dense_csv).open(newline="") as handle:
        dense_rows = list(csv.DictReader(handle))
    dense = {row["task"]: float(row["score"]) for row in dense_rows}
    lm_eval = json.loads(Path(args.mofe_lm_eval).read_text())
    wikitext = json.loads(Path(args.mofe_wikitext).read_text())
    evaluation = lm_eval["mofe_evaluation"]
    config = evaluation["mofe_config"]

    scores = {
        task: metric_value(lm_eval["results"][task], metric)
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    scores["wikitext2_validation"] = float(wikitext["perplexity"])

    rows = []
    for task, score in scores.items():
        metric = DOWNSTREAM_METRICS.get(task, "perplexity")
        dense_score = dense[task]
        rows.append(
            {
                "task": task,
                "metric": metric,
                "dense_score": dense_score,
                "mofe_score": score,
                "absolute_change": score - dense_score,
                "relative_change_percent": (score / dense_score - 1.0) * 100.0,
            }
        )

    csv_path = output_dir / "dense_vs_mofe.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = list(DOWNSTREAM_METRICS)
    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    ax.bar([i - width / 2 for i in x], [dense[t] for t in labels], width, label="Dense GPT-2", color="#3778A8")
    ax.bar([i + width / 2 for i in x], [scores[t] for t in labels], width, label="MoFE, 300 steps", color="#D27A3F")
    ax.set_xticks(x, [label.replace("_", "\n") for label in labels])
    ax.set_ylabel("Zero-shot score")
    ax.set_title("Dense GPT-2 vs trained MoFE")
    ax.set_ylim(0, 0.72)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_path = figures_dir / "dense_vs_mofe.png"
    fig.savefig(figure_path)
    plt.close(fig)

    markdown_path = output_dir / "evaluation_summary.md"
    lines = [
        "# Dense GPT-2 vs MoFE (300 steps)",
        "",
        f"- Checkpoint: `{evaluation['checkpoint']}`",
        f"- MoFE layers: `{config['moe_layer_indices']}`; private experts: `{config['num_private_experts']}`; top-k: `{config['top_k']}`",
        f"- Precision: `{evaluation['precision']}`; zero-shot for all downstream tasks",
        "- Dense GPT-2 is the original checkpoint and did not receive the same continued-training token budget.",
        "",
        "| Task | Metric | Dense | MoFE | Change | Relative change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['dense_score']:.6f} | "
            f"{row['mofe_score']:.6f} | {row['absolute_change']:+.6f} | "
            f"{row['relative_change_percent']:+.2f}% |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison figure: {figure_path}")
    print(f"Saved evaluation summary: {markdown_path}")


if __name__ == "__main__":
    main()
