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
    parser.add_argument("--original-dense-csv", required=True)
    parser.add_argument("--dense1500-lm-eval", required=True)
    parser.add_argument("--dense1500-wikitext", required=True)
    parser.add_argument("--dense1500-train-log", required=True)
    parser.add_argument("--mofe1500-lm-eval", required=True)
    parser.add_argument("--mofe1500-wikitext", required=True)
    parser.add_argument("--mofe1500-train-log", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metric_value(task_data: dict, metric: str) -> float:
    for key, value in task_data.items():
        if key.split(",")[0] == metric and not key.endswith("_stderr"):
            return float(value)
    raise KeyError(f"Metric {metric!r} not found in {task_data.keys()}")


def last_jsonl(path: str) -> dict:
    with Path(path).open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records[-1]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.original_dense_csv).open(newline="") as handle:
        original = {
            row["task"]: float(row["score"]) for row in csv.DictReader(handle)
        }
    dense_eval = json.loads(Path(args.dense1500_lm_eval).read_text())
    mofe_eval = json.loads(Path(args.mofe1500_lm_eval).read_text())
    dense = {
        task: metric_value(dense_eval["results"][task], metric)
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    mofe = {
        task: metric_value(mofe_eval["results"][task], metric)
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    dense["wikitext2_validation"] = float(
        json.loads(Path(args.dense1500_wikitext).read_text())["perplexity"]
    )
    mofe["wikitext2_validation"] = float(
        json.loads(Path(args.mofe1500_wikitext).read_text())["perplexity"]
    )

    rows = []
    for task in (*DOWNSTREAM_METRICS, "wikitext2_validation"):
        rows.append(
            {
                "task": task,
                "metric": DOWNSTREAM_METRICS.get(task, "perplexity"),
                "original_dense_step0": original[task],
                "dense_step1500": dense[task],
                "mofe_step1500": mofe[task],
                "mofe_minus_dense_at_step1500": mofe[task] - dense[task],
            }
        )
    comparison_csv = output_dir / "step1500_dense_vs_mofe.csv"
    with comparison_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dense_train = last_jsonl(args.dense1500_train_log)
    mofe_train = last_jsonl(args.mofe1500_train_log)
    continuation_steps = 1200
    efficiency_rows = [
        {
            "model": "dense_step1500",
            "parameters": 124439808,
            "seconds_per_step": dense_train["elapsed_seconds"] / continuation_steps,
            "tokens_per_second": dense_train["tokens_per_second"],
            "peak_memory_gib": dense_train["peak_memory_bytes"] / 2**30,
        },
        {
            "model": "mofe_step1500",
            "parameters": 209595696,
            "seconds_per_step": mofe_train["elapsed_seconds"] / continuation_steps,
            "tokens_per_second": mofe_train["tokens_per_second"],
            "peak_memory_gib": mofe_train["peak_memory_bytes"] / 2**30,
        },
    ]
    efficiency_csv = output_dir / "step1500_efficiency.csv"
    with efficiency_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(efficiency_rows[0]))
        writer.writeheader()
        writer.writerows(efficiency_rows)

    labels = list(DOWNSTREAM_METRICS)
    x = list(range(len(labels)))
    width = 0.25
    colors = ("#80868B", "#3778A8", "#D27A3F")
    fig = plt.figure(figsize=(14, 5.5), dpi=160, layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 3.2), wspace=0.18)
    ax_ppl = fig.add_subplot(grid[0, 0])
    ax_scores = fig.add_subplot(grid[0, 1])

    ppl = [
        original["wikitext2_validation"],
        dense["wikitext2_validation"],
        mofe["wikitext2_validation"],
    ]
    bars = ax_ppl.bar(
        ["Dense\nstep 0", "Dense\nstep 1500", "MoFE\nstep 1500"],
        ppl,
        color=colors,
        width=0.68,
    )
    ax_ppl.set_title("WikiText-2 Validation")
    ax_ppl.set_ylabel("Perplexity (lower is better)")
    ax_ppl.set_ylim(0, max(ppl) * 1.22)
    ax_ppl.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, ppl):
        ax_ppl.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}",
            ha="center",
        )

    ax_scores.bar(
        [i - width for i in x],
        [original[t] for t in labels],
        width,
        label="Dense step 0",
        color=colors[0],
    )
    ax_scores.bar(
        x,
        [dense[t] for t in labels],
        width,
        label="Dense step 1500",
        color=colors[1],
    )
    ax_scores.bar(
        [i + width for i in x],
        [mofe[t] for t in labels],
        width,
        label="MoFE step 1500",
        color=colors[2],
    )
    ax_scores.set_xticks(x, [label.replace("_", "\n") for label in labels])
    ax_scores.set_title("Zero-shot Downstream Scores")
    ax_scores.set_ylabel("Score")
    ax_scores.set_ylim(0, 0.72)
    ax_scores.grid(axis="y", alpha=0.25)
    ax_scores.legend()
    fig.suptitle("Dense and MoFE Step-based Comparison", fontsize=14)
    figure_path = figures_dir / "step1500_dense_vs_mofe.png"
    fig.savefig(figure_path)
    plt.close(fig)

    summary_path = output_dir / "step1500_comparison.md"
    lines = [
        "# Dense vs MoFE: Step-based Comparison",
        "",
        "| Task | Metric | Dense step 0 | Dense step 1500 | MoFE step 1500 | MoFE - Dense at step 1500 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['metric']} | "
            f"{row['original_dense_step0']:.6f} | {row['dense_step1500']:.6f} | "
            f"{row['mofe_step1500']:.6f} | "
            f"{row['mofe_minus_dense_at_step1500']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-step Efficiency",
            "",
            "| Model | Parameters | Seconds/step | Tokens/s | Peak GiB/GPU |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in efficiency_rows:
        lines.append(
            f"| {row['model']} | {row['parameters']:,} | "
            f"{row['seconds_per_step']:.4f} | {row['tokens_per_second']:.0f} | "
            f"{row['peak_memory_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Both trained checkpoints are compared at optimizer step 1500 with the same "
            "batch, sequence length, data, and two-stage learning-rate schedule. Parameter "
            "count and compute per step are different and are reported separately.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved comparison: {comparison_csv}")
    print(f"Saved efficiency: {efficiency_csv}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
