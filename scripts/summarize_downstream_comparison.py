from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_METRICS = {
    "lambada_openai": "acc",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-lm-eval", nargs="+", required=True)
    parser.add_argument("--mofe-lm-eval", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def metric_value(task_data: dict, metric: str) -> float:
    key = f"{metric},none"
    if key not in task_data:
        raise KeyError(f"metric {key!r} not found in {task_data.keys()}")
    return float(task_data[key])


def read_evaluations(paths: list[str]) -> tuple[dict[str, dict], list[dict]]:
    tasks: dict[str, dict] = {}
    metadata = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        overlap = set(tasks) & set(payload["results"])
        if overlap:
            raise ValueError(f"duplicate tasks across inputs: {sorted(overlap)}")
        tasks.update(payload["results"])
        metadata.append(payload["checkpoint_evaluation"])
    return tasks, metadata


def common_training_step(metadata: list[dict], model_name: str) -> int:
    steps = {item.get("training_steps") for item in metadata}
    if len(steps) != 1 or None in steps:
        raise ValueError(f"inconsistent {model_name} checkpoint steps: {steps}")
    return int(steps.pop())


def main() -> None:
    args = parse_args()
    dense, dense_metadata = read_evaluations(args.dense_lm_eval)
    mofe, mofe_metadata = read_evaluations(args.mofe_lm_eval)
    dense_step = common_training_step(dense_metadata, "Dense")
    mofe_step = common_training_step(mofe_metadata, "MoFE")
    if dense_step != mofe_step:
        raise ValueError(
            f"checkpoint steps differ: Dense={dense_step}, MoFE={mofe_step}"
        )
    expected = set(TASK_METRICS)
    if not expected <= set(dense) or not expected <= set(mofe):
        raise ValueError(
            f"expected tasks {sorted(expected)}, got Dense={sorted(dense)}, "
            f"MoFE={sorted(mofe)}"
        )

    rows = []
    for task, metric in TASK_METRICS.items():
        if dense[task]["sample_len"] != mofe[task]["sample_len"]:
            raise ValueError(f"sample count differs for {task}")
        dense_score = metric_value(dense[task], metric)
        mofe_score = metric_value(mofe[task], metric)
        dense_stderr = metric_value(dense[task], f"{metric}_stderr")
        mofe_stderr = metric_value(mofe[task], f"{metric}_stderr")
        rows.append(
            {
                "task": task,
                "metric": metric,
                "sample_count": int(dense[task]["sample_len"]),
                "dense_score": dense_score,
                "dense_stderr": dense_stderr,
                "mofe_score": mofe_score,
                "mofe_stderr": mofe_stderr,
                "mofe_minus_dense": mofe_score - dense_score,
                "independent_difference_stderr": math.hypot(
                    dense_stderr, mofe_stderr
                ),
            }
        )

    dense_macro = sum(row["dense_score"] for row in rows) / len(rows)
    mofe_macro = sum(row["mofe_score"] for row in rows) / len(rows)
    lambada_dense_ppl = metric_value(dense["lambada_openai"], "perplexity")
    lambada_mofe_ppl = metric_value(mofe["lambada_openai"], "perplexity")
    summary = {
        "checkpoint_step": dense_step,
        "zero_shot": True,
        "task_count": len(rows),
        "total_evaluation_samples": sum(row["sample_count"] for row in rows),
        "dense_macro_average": dense_macro,
        "mofe_macro_average": mofe_macro,
        "mofe_minus_dense_macro_average": mofe_macro - dense_macro,
        "dense_task_wins": sum(row["dense_score"] > row["mofe_score"] for row in rows),
        "mofe_task_wins": sum(row["mofe_score"] > row["dense_score"] for row in rows),
        "lambada_perplexity": {
            "dense": lambada_dense_ppl,
            "mofe": lambada_mofe_ppl,
            "mofe_percent_change": (
                lambada_mofe_ppl / lambada_dense_ppl - 1.0
            )
            * 100.0,
        },
        "dense_metadata": dense_metadata,
        "mofe_metadata": mofe_metadata,
    }

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dense_mofe_downstream.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "dense_mofe_downstream_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    labels = [row["task"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    colors = ("#3778A8", "#D27A3F")
    fig, (ax_scores, ax_delta) = plt.subplots(
        1, 2, figsize=(14, 5.8), dpi=170, gridspec_kw={"width_ratios": (2.4, 1)}
    )
    ax_scores.bar(
        x - width / 2,
        [row["dense_score"] for row in rows],
        width,
        yerr=[row["dense_stderr"] for row in rows],
        capsize=3,
        color=colors[0],
        label="Dense",
    )
    ax_scores.bar(
        x + width / 2,
        [row["mofe_score"] for row in rows],
        width,
        yerr=[row["mofe_stderr"] for row in rows],
        capsize=3,
        color=colors[1],
        label="MoFE",
    )
    ax_scores.set_xticks(x, labels)
    ax_scores.set_ylabel("Zero-shot accuracy")
    ax_scores.set_title("Downstream Scores")
    ax_scores.set_ylim(0, 0.68)
    ax_scores.grid(axis="y", alpha=0.24)
    ax_scores.legend()

    deltas = [row["mofe_minus_dense"] * 100.0 for row in rows]
    ax_delta.barh(
        np.arange(len(rows)),
        deltas,
        color=[colors[1] if delta >= 0 else colors[0] for delta in deltas],
    )
    ax_delta.axvline(0, color="#333333", linewidth=1)
    ax_delta.set_yticks(np.arange(len(rows)), labels)
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("MoFE - Dense (percentage points)")
    ax_delta.set_title("Score Difference")
    ax_delta.grid(axis="x", alpha=0.24)
    fig.suptitle(
        f"Dense vs MoFE: Step-{dense_step} Zero-Shot Evaluation", fontsize=14
    )
    fig.tight_layout()
    figure_path = figures_dir / "dense_vs_mofe_downstream.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    report_path = output_dir / "dense_mofe_downstream_summary.md"
    lines = [
        f"# Dense vs MoFE Step-{dense_step} Downstream Evaluation",
        "",
        "Both checkpoints use the same FineWeb-Edu training-token budget and were "
        "evaluated zero-shot with lm-eval 0.4.12 in bfloat16.",
        "",
        "| Task | Metric | Samples | Dense | MoFE | MoFE - Dense | Dense stderr | MoFE stderr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['sample_count']} | "
            f"{row['dense_score']:.6f} | {row['mofe_score']:.6f} | "
            f"{row['mofe_minus_dense']:+.6f} | {row['dense_stderr']:.6f} | "
            f"{row['mofe_stderr']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Dense four-task macro average: `{dense_macro:.6f}`",
            f"- MoFE four-task macro average: `{mofe_macro:.6f}`",
            f"- Macro difference: `{mofe_macro - dense_macro:+.6f}`",
            f"- Task wins: Dense `{summary['dense_task_wins']}`, MoFE "
            f"`{summary['mofe_task_wins']}`",
            f"- LAMBADA perplexity: Dense `{lambada_dense_ppl:.4f}`, MoFE "
            f"`{lambada_mofe_ppl:.4f}` "
            f"(`{summary['lambada_perplexity']['mofe_percent_change']:+.2f}%`)",
            "",
            "All accuracy differences are smaller than the reported per-model standard "
            "errors. This run does not establish a clear downstream winner.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {report_path}")


if __name__ == "__main__":
    main()
