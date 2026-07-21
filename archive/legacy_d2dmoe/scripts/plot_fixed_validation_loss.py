import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--model-label", default="MoFE")
    parser.add_argument("--dataset-label", default="WikiText-103")
    parser.add_argument("--phase-name", default="warmup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.validation_log).open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) < 2:
        raise ValueError("validation log needs at least two records")

    steps = np.array([record["step"] for record in records])
    losses = np.array([record["lm_loss"] for record in records])
    perplexities = np.array([record["perplexity"] for record in records])
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, record in enumerate(records):
        previous_loss = losses[index - 1] if index else losses[index]
        rows.append(
            {
                **record,
                "loss_change_from_previous": losses[index] - previous_loss,
                "loss_change_from_step0": losses[index] - losses[0],
            }
        )
    csv_path = output_dir / "fixed_validation_series.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = ("#3778A8", "#D27A3F")
    fig, (ax_loss, ax_ppl) = plt.subplots(
        1, 2, figsize=(13, 5.5), dpi=170, layout="constrained"
    )
    for ax in (ax_loss, ax_ppl):
        ax.axvspan(0, args.warmup_steps, color="#80868B", alpha=0.12)
        ax.axvline(args.warmup_steps, color="#555555", linestyle="--", linewidth=1.2)
        ax.grid(alpha=0.24)
        ax.set_xlabel("Optimizer step")

    ax_loss.plot(steps, losses, color=colors[0], marker="o", linewidth=2)
    ax_loss.set_title(f"Fixed {args.dataset_label} Validation Loss")
    ax_loss.set_ylabel("Token-weighted language-modeling loss")
    ax_loss.annotate(
        f"{losses[0]:.4f}",
        (steps[0], losses[0]),
        xytext=(8, -18),
        textcoords="offset points",
    )
    ax_loss.annotate(
        f"{losses[-1]:.4f}",
        (steps[-1], losses[-1]),
        xytext=(-45, 10),
        textcoords="offset points",
    )

    ax_ppl.plot(steps, perplexities, color=colors[1], marker="o", linewidth=2)
    ax_ppl.set_title(f"Fixed {args.dataset_label} Validation Perplexity")
    ax_ppl.set_ylabel("Perplexity (lower is better)")
    ax_ppl.annotate(
        f"{perplexities[0]:.2f}",
        (steps[0], perplexities[0]),
        xytext=(8, -18),
        textcoords="offset points",
    )
    ax_ppl.annotate(
        f"{perplexities[-1]:.2f}",
        (steps[-1], perplexities[-1]),
        xytext=(-38, 10),
        textcoords="offset points",
    )
    fig.suptitle(
        f"{args.model_label} Training: "
        f"{args.warmup_steps}-step {args.phase_name} + "
        f"{int(steps[-1]) - args.warmup_steps} Continued Steps",
        fontsize=14,
    )
    figure_path = figures_dir / "fixed_validation_loss_and_ppl.png"
    fig.savefig(figure_path)
    plt.close(fig)

    exact_warmup_indices = np.where(steps == args.warmup_steps)[0]
    if exact_warmup_indices.size:
        reference_index = int(exact_warmup_indices[0])
        reference_kind = "warmup_end"
    else:
        post_warmup_indices = np.where(steps > args.warmup_steps)[0]
        if not post_warmup_indices.size:
            raise ValueError("validation log has no record after warmup")
        reference_index = int(post_warmup_indices[0])
        reference_kind = "first_post_warmup_validation"
    summary = {
        "initial_step": int(steps[0]),
        "final_step": int(steps[-1]),
        "warmup_steps": args.warmup_steps,
        "validation_sample_count": records[0]["sample_count"],
        "validation_token_count": records[0]["token_count"],
        "initial_loss": float(losses[0]),
        "post_warmup_reference_step": int(steps[reference_index]),
        "post_warmup_reference_kind": reference_kind,
        "post_warmup_reference_loss": float(losses[reference_index]),
        "final_loss": float(losses[-1]),
        "loss_change_total": float(losses[-1] - losses[0]),
        "loss_change_after_post_warmup_reference": float(
            losses[-1] - losses[reference_index]
        ),
        "initial_perplexity": float(perplexities[0]),
        "post_warmup_reference_perplexity": float(
            perplexities[reference_index]
        ),
        "final_perplexity": float(perplexities[-1]),
        "perplexity_change_percent_total": float(
            (perplexities[-1] / perplexities[0] - 1.0) * 100.0
        ),
    }
    summary_path = output_dir / "fixed_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    report_path = output_dir / "fixed_validation_summary.md"
    has_private_scale = all("private_scale" in row for row in rows)
    lines = [
        f"# {args.model_label} Fixed-validation Training Summary",
        "",
    ]
    if has_private_scale:
        lines.extend(
            [
                "| Step | Private scale | Validation loss | Perplexity | Loss change |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Step | Validation loss | Perplexity | Loss change |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
    for row in rows:
        if has_private_scale:
            lines.append(
                f"| {row['step']} | {row['private_scale']:.2f} | "
                f"{row['lm_loss']:.6f} | {row['perplexity']:.4f} | "
                f"{row['loss_change_from_previous']:+.6f} |"
            )
        else:
            lines.append(
                f"| {row['step']} | {row['lm_loss']:.6f} | "
                f"{row['perplexity']:.4f} | "
                f"{row['loss_change_from_previous']:+.6f} |"
            )
    lines.extend(
        [
            "",
            f"- Total validation-loss change: `{summary['loss_change_total']:+.6f}`",
            f"- Validation-loss change after step "
            f"{summary['post_warmup_reference_step']} "
            f"({summary['post_warmup_reference_kind']}): "
            f"`{summary['loss_change_after_post_warmup_reference']:+.6f}`",
            f"- Total perplexity change: "
            f"`{summary['perplexity_change_percent_total']:+.2f}%`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")

    print(f"Saved validation series: {csv_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
