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
    parser.add_argument("--step300-lm-eval", required=True)
    parser.add_argument("--step300-wikitext", required=True)
    parser.add_argument("--step1500-lm-eval", required=True)
    parser.add_argument("--step1500-wikitext", required=True)
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
        dense = {row["task"]: float(row["score"]) for row in csv.DictReader(handle)}
    step300_eval = json.loads(Path(args.step300_lm_eval).read_text())
    step1500_eval = json.loads(Path(args.step1500_lm_eval).read_text())
    step300_wikitext = json.loads(Path(args.step300_wikitext).read_text())
    step1500_wikitext = json.loads(Path(args.step1500_wikitext).read_text())

    step300 = {
        task: metric_value(step300_eval["results"][task], metric)
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    step1500 = {
        task: metric_value(step1500_eval["results"][task], metric)
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    step300["wikitext2_validation"] = float(step300_wikitext["perplexity"])
    step1500["wikitext2_validation"] = float(step1500_wikitext["perplexity"])

    rows = []
    for task in (*DOWNSTREAM_METRICS, "wikitext2_validation"):
        rows.append(
            {
                "task": task,
                "metric": DOWNSTREAM_METRICS.get(task, "perplexity"),
                "dense": dense[task],
                "mofe_300": step300[task],
                "mofe_1500": step1500[task],
                "change_1500_vs_dense": step1500[task] - dense[task],
                "change_1500_vs_300": step1500[task] - step300[task],
            }
        )

    csv_path = output_dir / "dense_300_1500_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = list(DOWNSTREAM_METRICS)
    x = list(range(len(labels)))
    width = 0.25
    colors = ("#3778A8", "#D27A3F", "#3A8F68")
    fig = plt.figure(figsize=(14, 5.5), dpi=160, layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 3.2), wspace=0.23)
    ax_ppl = fig.add_subplot(grid[0, 0])
    ax_scores = fig.add_subplot(grid[0, 1])

    ppl_values = [
        dense["wikitext2_validation"],
        step300["wikitext2_validation"],
        step1500["wikitext2_validation"],
    ]
    ppl_bars = ax_ppl.bar(
        ["Dense", "300", "1500"], ppl_values, color=colors, width=0.68
    )
    ax_ppl.set_title("WikiText-2 Validation")
    ax_ppl.set_ylabel("Perplexity (lower is better)")
    ax_ppl.set_ylim(0, max(ppl_values) * 1.22)
    ax_ppl.grid(axis="y", alpha=0.25)
    for bar, value in zip(ppl_bars, ppl_values):
        ax_ppl.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}",
            ha="center",
        )

    ax_scores.bar(
        [i - width for i in x],
        [dense[t] for t in labels],
        width,
        label="Dense GPT-2",
        color=colors[0],
    )
    ax_scores.bar(
        x,
        [step300[t] for t in labels],
        width,
        label="MoFE 300",
        color=colors[1],
    )
    ax_scores.bar(
        [i + width for i in x],
        [step1500[t] for t in labels],
        width,
        label="MoFE 1500",
        color=colors[2],
    )
    ax_scores.set_xticks(x, [label.replace("_", "\n") for label in labels])
    ax_scores.set_title("Zero-shot Downstream Scores")
    ax_scores.set_ylabel("Score")
    ax_scores.set_ylim(0, 0.72)
    ax_scores.grid(axis="y", alpha=0.25)
    ax_scores.legend()
    fig.suptitle("Dense GPT-2 and MoFE Training Progression", fontsize=14)
    figure_path = figures_dir / "dense_300_1500_comparison.png"
    fig.savefig(figure_path)
    plt.close(fig)

    summary_path = output_dir / "evaluation_summary.md"
    lines = [
        "# Dense GPT-2 vs MoFE at 300 and 1500 Steps",
        "",
        "| Task | Metric | Dense | MoFE 300 | MoFE 1500 | 1500 vs Dense | 1500 vs 300 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['dense']:.6f} | "
            f"{row['mofe_300']:.6f} | {row['mofe_1500']:.6f} | "
            f"{row['change_1500_vs_dense']:+.6f} | "
            f"{row['change_1500_vs_300']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "All downstream tasks are zero-shot and evaluated in bfloat16. The original "
            "dense checkpoint did not receive the same continued-training budget, so this "
            "is a training progression comparison rather than an architecture-only ablation.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison figure: {figure_path}")
    print(f"Saved evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
