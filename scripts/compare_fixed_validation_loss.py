import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-validation-log", required=True)
    parser.add_argument("--mofe-validation-log", required=True)
    parser.add_argument("--dense-training-log", required=True)
    parser.add_argument("--mofe-training-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase-boundary", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    dense = read_jsonl(args.dense_validation_log)
    mofe = read_jsonl(args.mofe_validation_log)
    dense_by_step = {record["step"]: record for record in dense}
    mofe_by_step = {record["step"]: record for record in mofe}
    steps = sorted(set(dense_by_step) & set(mofe_by_step))
    if steps != sorted(dense_by_step) or steps != sorted(mofe_by_step):
        raise ValueError("Dense and MoFE validation logs must have identical steps")
    for step in steps:
        if dense_by_step[step]["token_count"] != mofe_by_step[step]["token_count"]:
            raise ValueError(f"validation token counts differ at step {step}")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "step": step,
            "dense_validation_loss": dense_by_step[step]["lm_loss"],
            "mofe_validation_loss": mofe_by_step[step]["lm_loss"],
            "mofe_minus_dense_loss": (
                mofe_by_step[step]["lm_loss"] - dense_by_step[step]["lm_loss"]
            ),
            "dense_perplexity": dense_by_step[step]["perplexity"],
            "mofe_perplexity": mofe_by_step[step]["perplexity"],
            "mofe_minus_dense_perplexity": (
                mofe_by_step[step]["perplexity"]
                - dense_by_step[step]["perplexity"]
            ),
        }
        for step in steps
    ]
    csv_path = output_dir / "dense_mofe_fixed_validation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    step_values = np.array(steps)
    colors = ("#3778A8", "#D27A3F")
    fig, (ax_loss, ax_ppl) = plt.subplots(
        1, 2, figsize=(13, 5.5), dpi=170, layout="constrained"
    )
    for ax in (ax_loss, ax_ppl):
        ax.axvline(
            args.phase_boundary, color="#555555", linestyle="--", linewidth=1.2
        )
        ax.grid(alpha=0.24)
        ax.set_xlabel("Optimizer step")
    ax_loss.plot(
        step_values,
        [dense_by_step[step]["lm_loss"] for step in steps],
        color=colors[0],
        marker="o",
        linewidth=2,
        label="Dense",
    )
    ax_loss.plot(
        step_values,
        [mofe_by_step[step]["lm_loss"] for step in steps],
        color=colors[1],
        marker="o",
        linewidth=2,
        label="MoFE",
    )
    ax_loss.set_title("Fixed WikiText-103 Validation Loss")
    ax_loss.set_ylabel("Token-weighted language-modeling loss")
    ax_loss.legend()

    ax_ppl.plot(
        step_values,
        [dense_by_step[step]["perplexity"] for step in steps],
        color=colors[0],
        marker="o",
        linewidth=2,
        label="Dense",
    )
    ax_ppl.plot(
        step_values,
        [mofe_by_step[step]["perplexity"] for step in steps],
        color=colors[1],
        marker="o",
        linewidth=2,
        label="MoFE",
    )
    ax_ppl.set_title("Fixed WikiText-103 Validation Perplexity")
    ax_ppl.set_ylabel("Perplexity (lower is better)")
    ax_ppl.legend()
    fig.suptitle("Dense vs MoFE: Matched 1000-step Training", fontsize=14)
    figure_path = figures_dir / "dense_vs_mofe_fixed_validation.png"
    fig.savefig(figure_path)
    plt.close(fig)

    dense_train = read_jsonl(args.dense_training_log)[-1]
    mofe_train = read_jsonl(args.mofe_training_log)[-1]
    final = rows[-1]
    summary = {
        "validation_sample_count": dense[0]["sample_count"],
        "validation_token_count": dense[0]["token_count"],
        "dense_initial_loss": dense[0]["lm_loss"],
        "dense_final_loss": final["dense_validation_loss"],
        "mofe_initial_loss": mofe[0]["lm_loss"],
        "mofe_final_loss": final["mofe_validation_loss"],
        "mofe_minus_dense_final_loss": final["mofe_minus_dense_loss"],
        "dense_final_perplexity": final["dense_perplexity"],
        "mofe_final_perplexity": final["mofe_perplexity"],
        "mofe_perplexity_change_percent_vs_dense": (
            final["mofe_perplexity"] / final["dense_perplexity"] - 1.0
        )
        * 100.0,
        "dense_tokens_per_second": dense_train["tokens_per_second"],
        "mofe_tokens_per_second": mofe_train["tokens_per_second"],
        "dense_peak_memory_gib": dense_train["peak_memory_bytes"] / 2**30,
        "mofe_peak_memory_gib": mofe_train["peak_memory_bytes"] / 2**30,
    }
    summary_path = output_dir / "dense_mofe_fixed_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = output_dir / "dense_mofe_fixed_validation_summary.md"
    lines = [
        "# Dense vs MoFE Fixed-validation Comparison",
        "",
        "Both runs use the same GPT-2 initialization, dataset, sequence length, "
        "effective batch size, constant learning rate, seed, and 1000 optimizer steps.",
        "",
        "| Step | Dense loss | MoFE loss | MoFE - Dense | Dense PPL | MoFE PPL |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['dense_validation_loss']:.6f} | "
            f"{row['mofe_validation_loss']:.6f} | "
            f"{row['mofe_minus_dense_loss']:+.6f} | "
            f"{row['dense_perplexity']:.4f} | {row['mofe_perplexity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| Model | Tokens/s | Peak GiB/GPU |",
            "| --- | ---: | ---: |",
            f"| Dense | {summary['dense_tokens_per_second']:.0f} | "
            f"{summary['dense_peak_memory_gib']:.2f} |",
            f"| MoFE | {summary['mofe_tokens_per_second']:.0f} | "
            f"{summary['mofe_peak_memory_gib']:.2f} |",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
