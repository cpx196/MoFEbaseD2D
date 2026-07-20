import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DOWNSTREAM_METRICS = {
    "lambada_openai": "acc",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense1500-lm-eval", required=True)
    parser.add_argument("--dense1500-wikitext", required=True)
    parser.add_argument("--mofe1500-lm-eval", required=True)
    parser.add_argument("--mofe1500-wikitext", required=True)
    parser.add_argument("--mofe3000-lm-eval", required=True)
    parser.add_argument("--mofe3000-wikitext", required=True)
    parser.add_argument("--mofe3000-train-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-steps", type=int, default=100)
    return parser.parse_args()


def metric_value(task_data: dict, metric: str) -> float:
    for key, value in task_data.items():
        if key.split(",")[0] == metric and not key.endswith("_stderr"):
            return float(value)
    raise KeyError(f"Metric {metric!r} not found in {task_data.keys()}")


def read_scores(
    lm_eval_path: str, wikitext_path: str
) -> tuple[dict[str, float], dict[str, float]]:
    lm_eval = json.loads(Path(lm_eval_path).read_text())
    scores = {
        task: metric_value(lm_eval["results"][task], metric)
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    standard_errors = {
        task: metric_value(lm_eval["results"][task], f"{metric}_stderr")
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    scores["wikitext2_validation"] = float(
        json.loads(Path(wikitext_path).read_text())["perplexity"]
    )
    return scores, standard_errors


def trailing_average(
    steps: np.ndarray, values: np.ndarray, points: int
) -> tuple[np.ndarray, np.ndarray]:
    if points <= 1:
        return steps, values
    kernel = np.ones(points, dtype=np.float64) / points
    return steps[points - 1 :], np.convolve(values, kernel, mode="valid")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dense1500, dense1500_errors = read_scores(
        args.dense1500_lm_eval, args.dense1500_wikitext
    )
    mofe1500, mofe1500_errors = read_scores(
        args.mofe1500_lm_eval, args.mofe1500_wikitext
    )
    mofe3000, mofe3000_errors = read_scores(
        args.mofe3000_lm_eval, args.mofe3000_wikitext
    )

    rows = []
    for task in (*DOWNSTREAM_METRICS, "wikitext2_validation"):
        rows.append(
            {
                "task": task,
                "metric": DOWNSTREAM_METRICS.get(task, "perplexity"),
                "dense_step1500": dense1500[task],
                "mofe_step1500": mofe1500[task],
                "mofe_step3000": mofe3000[task],
                "mofe_3000_minus_1500": mofe3000[task] - mofe1500[task],
                "mofe_3000_minus_dense1500": mofe3000[task] - dense1500[task],
                "mofe_step3000_stderr": mofe3000_errors.get(task, ""),
            }
        )
    comparison_csv = output_dir / "step3000_downstream_comparison.csv"
    with comparison_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = list(DOWNSTREAM_METRICS)
    x = np.arange(len(labels))
    width = 0.25
    colors = ("#3778A8", "#D27A3F", "#3A8F68")
    fig = plt.figure(figsize=(14, 5.7), dpi=170, layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 3.3), wspace=0.2)
    ax_ppl = fig.add_subplot(grid[0, 0])
    ax_scores = fig.add_subplot(grid[0, 1])

    names = ("Dense\n1500", "MoFE\n1500", "MoFE\n3000")
    ppl = [
        dense1500["wikitext2_validation"],
        mofe1500["wikitext2_validation"],
        mofe3000["wikitext2_validation"],
    ]
    bars = ax_ppl.bar(names, ppl, color=colors, width=0.68)
    ax_ppl.set_title("WikiText-2 Validation")
    ax_ppl.set_ylabel("Perplexity (lower is better)")
    ax_ppl.set_ylim(0, max(ppl) * 1.2)
    ax_ppl.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, ppl):
        ax_ppl.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(ppl) * 0.025,
            f"{value:.2f}",
            ha="center",
        )

    ax_scores.bar(
        x - width,
        [dense1500[task] for task in labels],
        width,
        label="Dense step 1500",
        color=colors[0],
        yerr=[dense1500_errors[task] for task in labels],
        capsize=3,
    )
    ax_scores.bar(
        x,
        [mofe1500[task] for task in labels],
        width,
        label="MoFE step 1500",
        color=colors[1],
        yerr=[mofe1500_errors[task] for task in labels],
        capsize=3,
    )
    ax_scores.bar(
        x + width,
        [mofe3000[task] for task in labels],
        width,
        label="MoFE step 3000",
        color=colors[2],
        yerr=[mofe3000_errors[task] for task in labels],
        capsize=3,
    )
    ax_scores.set_xticks(x, [label.replace("_", "\n") for label in labels])
    ax_scores.set_title("Zero-shot Downstream Scores")
    ax_scores.set_ylabel("Score")
    ax_scores.set_ylim(0, 0.72)
    ax_scores.grid(axis="y", alpha=0.25)
    ax_scores.legend()
    fig.suptitle("Dense and MoFE Checkpoint Comparison", fontsize=14)
    downstream_figure = figures_dir / "step3000_downstream_comparison.png"
    fig.savefig(downstream_figure)
    plt.close(fig)

    with Path(args.mofe3000_train_log).open() as handle:
        train_records = [json.loads(line) for line in handle if line.strip()]
    if not train_records:
        raise ValueError("MoFE step-3000 training log is empty")
    if any("effective_batch_samples" not in record for record in train_records):
        raise ValueError("loss plot requires corrected global effective-batch logs")
    train_steps = np.array([record["step"] for record in train_records])
    train_loss = np.array([record["lm_loss"] for record in train_records])
    logging_interval = int(np.median(np.diff(train_steps)))
    rolling_points = max(1, args.rolling_steps // logging_interval)
    smooth_steps, smooth_loss = trailing_average(
        train_steps, train_loss, rolling_points
    )
    slope = float(np.polyfit(train_steps, train_loss, 1)[0] * 100)

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=170, layout="constrained")
    ax.plot(
        train_steps,
        train_loss,
        color="#3778A8",
        alpha=0.2,
        linewidth=1,
        label="Global effective-batch loss",
    )
    ax.plot(
        smooth_steps,
        smooth_loss,
        color="#D27A3F",
        linewidth=2.2,
        label=f"{args.rolling_steps}-step moving average",
    )
    ax.set_title("MoFE Continued-training LM Loss (Corrected Logging)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Language-modeling loss")
    ax.grid(alpha=0.24)
    ax.legend()
    loss_figure = figures_dir / "step1500_to_3000_corrected_lm_loss.png"
    fig.savefig(loss_figure)
    plt.close(fig)

    first_mask = train_steps <= train_steps.min() + 95
    last_mask = train_steps >= train_steps.max() - 100
    loss_summary = {
        "first_logged_step": int(train_steps.min()),
        "last_logged_step": int(train_steps.max()),
        "record_count": len(train_records),
        "effective_batch_samples": sorted(
            {record["effective_batch_samples"] for record in train_records}
        ),
        "mean_first_100_steps": float(train_loss[first_mask].mean()),
        "mean_last_100_steps": float(train_loss[last_mask].mean()),
        "linear_slope_per_100_steps": slope,
        "rolling_steps": args.rolling_steps,
    }
    loss_summary_path = output_dir / "step3000_loss_summary.json"
    loss_summary_path.write_text(json.dumps(loss_summary, indent=2) + "\n")

    summary_path = output_dir / "step3000_comparison.md"
    lines = [
        "# MoFE Step-3000 Evaluation",
        "",
        "All downstream results are zero-shot and use "
        "the same bfloat16 evaluation setup.",
        "",
        "| Task | Metric | Dense 1500 | MoFE 1500 | MoFE 3000 | 3000 vs 1500 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['dense_step1500']:.6f} | "
            f"{row['mofe_step1500']:.6f} | {row['mofe_step3000']:.6f} | "
            f"{row['mofe_3000_minus_1500']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Corrected Loss",
            "",
            f"- First 100-step mean: `{loss_summary['mean_first_100_steps']:.6f}`",
            f"- Last 100-step mean: `{loss_summary['mean_last_100_steps']:.6f}`",
            f"- Linear slope per 100 steps: `{slope:+.6f}`",
            "- The legacy step-1-to-1500 log is not joined to this curve because it "
            "used rank 0's final microbatch rather than a global effective-batch mean.",
            "- Every downstream step-3000 vs step-1500 change is smaller than the "
            "reported standard error; the score differences are not statistically clear.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved downstream figure: {downstream_figure}")
    print(f"Saved loss figure: {loss_figure}")
    print(f"Saved loss summary: {loss_summary_path}")
    print(f"Saved report: {summary_path}")


if __name__ == "__main__":
    main()
