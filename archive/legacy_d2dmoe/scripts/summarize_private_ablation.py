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
LAYERS = (
    "transformer.h.9.mlp",
    "transformer.h.10.mlp",
    "transformer.h.11.mlp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-on-lm-eval", required=True)
    parser.add_argument("--private-on-wikitext", required=True)
    parser.add_argument("--private-off-lm-eval", required=True)
    parser.add_argument("--private-off-wikitext", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metric_value(task_data: dict, metric: str) -> float:
    for key, value in task_data.items():
        if key.split(",")[0] == metric:
            return float(value)
    raise KeyError(f"Metric {metric!r} not found in {task_data.keys()}")


def read_evaluation(
    lm_eval_path: str, wikitext_path: str
) -> tuple[dict[str, float], dict[str, float]]:
    lm_eval = json.loads(Path(lm_eval_path).read_text())
    scores = {
        task: metric_value(lm_eval["results"][task], metric)
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    errors = {
        task: metric_value(lm_eval["results"][task], f"{metric}_stderr")
        for task, metric in DOWNSTREAM_METRICS.items()
    }
    scores["wikitext2_validation"] = float(
        json.loads(Path(wikitext_path).read_text())["perplexity"]
    )
    return scores, errors


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    private_on, private_on_errors = read_evaluation(
        args.private_on_lm_eval, args.private_on_wikitext
    )
    private_off, private_off_errors = read_evaluation(
        args.private_off_lm_eval, args.private_off_wikitext
    )
    rows = []
    for task in (*DOWNSTREAM_METRICS, "wikitext2_validation"):
        rows.append(
            {
                "task": task,
                "metric": DOWNSTREAM_METRICS.get(task, "perplexity"),
                "private_off": private_off[task],
                "private_on": private_on[task],
                "on_minus_off": private_on[task] - private_off[task],
                "private_off_stderr": private_off_errors.get(task, ""),
                "private_on_stderr": private_on_errors.get(task, ""),
            }
        )
    comparison_csv = output_dir / "private_on_off_comparison.csv"
    with comparison_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = list(DOWNSTREAM_METRICS)
    x = np.arange(len(labels))
    width = 0.36
    colors = ("#80868B", "#3A8F68")
    fig = plt.figure(figsize=(14, 5.7), dpi=170, layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 3.3), wspace=0.2)
    ax_ppl = fig.add_subplot(grid[0, 0])
    ax_scores = fig.add_subplot(grid[0, 1])

    ppl = [
        private_off["wikitext2_validation"],
        private_on["wikitext2_validation"],
    ]
    bars = ax_ppl.bar(("Private OFF", "Private ON"), ppl, color=colors, width=0.65)
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
        x - width / 2,
        [private_off[task] for task in labels],
        width,
        label="Private OFF (scale 0)",
        color=colors[0],
        yerr=[private_off_errors[task] for task in labels],
        capsize=3,
    )
    ax_scores.bar(
        x + width / 2,
        [private_on[task] for task in labels],
        width,
        label="Private ON (scale 1)",
        color=colors[1],
        yerr=[private_on_errors[task] for task in labels],
        capsize=3,
    )
    ax_scores.set_xticks(x, [label.replace("_", "\n") for label in labels])
    ax_scores.set_title("Step-3000 Private-branch Output Ablation")
    ax_scores.set_ylabel("Zero-shot score")
    ax_scores.set_ylim(0, 0.72)
    ax_scores.grid(axis="y", alpha=0.25)
    ax_scores.legend()
    fig.suptitle("MoFE Private Branch: ON vs OFF", fontsize=14)
    ablation_figure = figures_dir / "private_on_off_ablation.png"
    fig.savefig(ablation_figure)
    plt.close(fig)

    with Path(args.train_log).open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    steps = np.array([record["step"] for record in records])
    ratio_summary = {}
    fig, ax = plt.subplots(figsize=(11, 5.7), dpi=170, layout="constrained")
    layer_colors = ("#3778A8", "#D27A3F", "#3A8F68")
    for layer, color in zip(LAYERS, layer_colors):
        ratios = np.array(
            [record["routing"][layer]["private_to_shared_norm"] for record in records]
        )
        short_name = f"Layer {layer.split('.')[2]}"
        ax.plot(steps, ratios, color=color, linewidth=1.6, label=short_name)
        ratio_summary[layer] = {
            "mean": float(ratios.mean()),
            "first_100_step_mean": float(ratios[:20].mean()),
            "last_100_step_mean": float(ratios[-20:].mean()),
            "minimum": float(ratios.min()),
            "maximum": float(ratios.max()),
        }
    ax.set_title("Private-to-Shared Output Norm Ratio During Continued Training")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Private output norm / shared output norm")
    ax.set_ylim(0, 0.62)
    ax.grid(alpha=0.24)
    ax.legend()
    ratio_figure = figures_dir / "private_shared_norm_ratio.png"
    fig.savefig(ratio_figure)
    plt.close(fig)
    ratio_path = output_dir / "private_shared_norm_summary.json"
    ratio_path.write_text(json.dumps(ratio_summary, indent=2) + "\n")

    report_path = output_dir / "private_ablation.md"
    lines = [
        "# MoFE Step-3000 Private-branch Ablation",
        "",
        "Private OFF sets the output scale to zero on the same checkpoint. Routing and "
        "private computation are retained, so this isolates predictive contribution rather "
        "than runtime cost.",
        "",
        "| Task | Metric | Private OFF | Private ON | ON - OFF |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['private_off']:.6f} | "
            f"{row['private_on']:.6f} | {row['on_minus_off']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Output Magnitude",
            "",
            "| Layer | Mean private/shared norm | First 100 steps | Last 100 steps |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for layer in LAYERS:
        stats = ratio_summary[layer]
        lines.append(
            f"| {layer} | {stats['mean']:.6f} | "
            f"{stats['first_100_step_mean']:.6f} | "
            f"{stats['last_100_step_mean']:.6f} |"
        )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved ablation figure: {ablation_figure}")
    print(f"Saved ratio figure: {ratio_figure}")
    print(f"Saved ratio summary: {ratio_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
