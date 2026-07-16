import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BRANCHES = (
    ("shared_only", "Shared only", "#6B7280"),
    ("private_only", "Private only", "#D97706"),
    ("full_mofe", "Full MoFE", "#16856B"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label, _color in BRANCHES:
        fixed = load_result(input_dir / "fixed_validation" / f"{key}.json")
        wikitext2 = load_result(input_dir / "wikitext2" / f"{key}.json")
        rows.append(
            {
                "branch": key,
                "label": label,
                "private_scale": fixed["private_scale"],
                "shared_scale": fixed["shared_scale"],
                "wikitext103_validation_loss": fixed["negative_log_likelihood"],
                "wikitext103_validation_ppl": fixed["perplexity"],
                "wikitext2_validation_ppl": wikitext2["perplexity"],
            }
        )

    csv_path = output_dir / "branch_ablation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    colors = [color for _key, _label, color in BRANCHES]
    metrics = (
        ("wikitext103_validation_loss", "WikiText-103 Fixed Validation", "Loss"),
        ("wikitext103_validation_ppl", "WikiText-103 Fixed Validation", "Perplexity"),
        ("wikitext2_validation_ppl", "WikiText-2 Validation", "Perplexity"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=180, layout="constrained")
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        values = [row[metric] for row in rows]
        bars = ax.bar(np.arange(len(rows)), values, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(rows)), labels)
        ax.set_title(title)
        ax.set_ylabel(f"{ylabel} (lower is better)")
        ax.set_ylim(0, max(values) * 1.22)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )
    fig.suptitle("MoFE Step-1000 Branch Ablation")
    figure_path = figures_dir / "shared_private_branch_ablation.png"
    fig.savefig(figure_path)
    plt.close(fig)

    by_key = {row["branch"]: row for row in rows}
    shared = by_key["shared_only"]
    private = by_key["private_only"]
    full = by_key["full_mofe"]
    summary = {
        "private_increment_over_shared": {
            "wikitext103_loss_reduction": shared["wikitext103_validation_loss"]
            - full["wikitext103_validation_loss"],
            "wikitext103_ppl_relative_reduction_percent": 100
            * (shared["wikitext103_validation_ppl"] - full["wikitext103_validation_ppl"])
            / shared["wikitext103_validation_ppl"],
            "wikitext2_ppl_relative_reduction_percent": 100
            * (shared["wikitext2_validation_ppl"] - full["wikitext2_validation_ppl"])
            / shared["wikitext2_validation_ppl"],
        },
        "shared_increment_over_private": {
            "wikitext103_loss_reduction": private["wikitext103_validation_loss"]
            - full["wikitext103_validation_loss"],
            "wikitext103_ppl_relative_reduction_percent": 100
            * (private["wikitext103_validation_ppl"] - full["wikitext103_validation_ppl"])
            / private["wikitext103_validation_ppl"],
            "wikitext2_ppl_relative_reduction_percent": 100
            * (private["wikitext2_validation_ppl"] - full["wikitext2_validation_ppl"])
            / private["wikitext2_validation_ppl"],
        },
    }
    (output_dir / "branch_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    # The requested ablation is Private OFF vs ON; keep shared enabled in both.
    private_comparison = (shared, full)
    comparison_labels = ("Private OFF", "Private ON")
    comparison_colors = ("#6B7280", "#16856B")
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), dpi=180, layout="constrained"
    )
    loss_values = [row["wikitext103_validation_loss"] for row in private_comparison]
    bars = axes[0].bar(comparison_labels, loss_values, color=comparison_colors, width=0.62)
    axes[0].set_title("WikiText-103 Fixed Validation Loss")
    axes[0].set_ylabel("Loss (lower is better)")
    axes[0].set_ylim(0, max(loss_values) * 1.2)
    axes[0].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, loss_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(loss_values) * 0.025,
            f"{value:.4f}",
            ha="center",
        )

    dataset_labels = ("WikiText-103\nfixed validation", "WikiText-2\nvalidation")
    x = np.arange(len(dataset_labels))
    width = 0.34
    off_ppl = (
        shared["wikitext103_validation_ppl"],
        shared["wikitext2_validation_ppl"],
    )
    on_ppl = (
        full["wikitext103_validation_ppl"],
        full["wikitext2_validation_ppl"],
    )
    off_bars = axes[1].bar(
        x - width / 2,
        off_ppl,
        width,
        label="Private OFF",
        color=comparison_colors[0],
    )
    on_bars = axes[1].bar(
        x + width / 2,
        on_ppl,
        width,
        label="Private ON",
        color=comparison_colors[1],
    )
    axes[1].set_xticks(x, dataset_labels)
    axes[1].set_title("Perplexity by Validation Dataset")
    axes[1].set_ylabel("Perplexity (lower is better)")
    axes[1].set_ylim(0, max((*off_ppl, *on_ppl)) * 1.25)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend()
    for bars, values in ((off_bars, off_ppl), (on_bars, on_ppl)):
        for bar, value in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + max((*off_ppl, *on_ppl)) * 0.025,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
            )
    fig.suptitle("MoFE Step-1000 Private-branch Ablation (Shared Kept ON)")
    private_figure_path = figures_dir / "private_off_vs_on.png"
    fig.savefig(private_figure_path)
    plt.close(fig)

    private_report = [
        "# MoFE Step-1000 Private OFF vs ON",
        "",
        "The shared branch remains enabled in both configurations. Private OFF sets only "
        "the private output scale to zero on the same checkpoint.",
        "",
        "| Configuration | Private scale | Shared scale | WT103 fixed loss | WT103 PPL | WT2 PPL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Private OFF | 0 | 1 | {shared['wikitext103_validation_loss']:.6f} | "
        f"{shared['wikitext103_validation_ppl']:.4f} | "
        f"{shared['wikitext2_validation_ppl']:.4f} |",
        f"| Private ON | 1 | 1 | {full['wikitext103_validation_loss']:.6f} | "
        f"{full['wikitext103_validation_ppl']:.4f} | "
        f"{full['wikitext2_validation_ppl']:.4f} |",
        "",
        "WT103 uses the training-matched, non-overlapping packed fixed validation set. "
        "WT2 uses the separate WikiText-2 validation corpus with a sliding-window protocol. "
        "PPL values should be compared within each dataset, not across datasets.",
    ]
    (output_dir / "private_off_vs_on.md").write_text(
        "\n".join(private_report) + "\n"
    )

    report = [
        "# MoFE Step-1000 Shared/Private Branch Ablation",
        "",
        "The shared/private scales are changed only at evaluation time on the same checkpoint. "
        "All GPT-2 attention, embeddings, layer norms, and non-MoFE blocks remain enabled.",
        "",
        "| Configuration | Private scale | Shared scale | WT103 loss | WT103 PPL | WT2 PPL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {row['label']} | {row['private_scale']:.0f} | {row['shared_scale']:.0f} | "
            f"{row['wikitext103_validation_loss']:.6f} | "
            f"{row['wikitext103_validation_ppl']:.4f} | "
            f"{row['wikitext2_validation_ppl']:.4f} |"
        )
    contribution = summary["private_increment_over_shared"]
    report.extend(
        [
            "",
            "## Private branch contribution",
            "",
            f"Relative to Shared only, enabling private lowers fixed WT103 loss by "
            f"`{contribution['wikitext103_loss_reduction']:.6f}`, lowers WT103 PPL by "
            f"`{contribution['wikitext103_ppl_relative_reduction_percent']:.2f}%`, and lowers "
            f"WT2 PPL by `{contribution['wikitext2_ppl_relative_reduction_percent']:.2f}%`.",
            "",
            "Private only is not a replacement for the shared path. The best result requires "
            "both outputs, which supports interpreting private experts as token-routed residual "
            "corrections on top of the shared representation.",
        ]
    )
    (output_dir / "branch_ablation.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved figure: {figure_path}")
    print(f"Saved requested private ablation: {private_figure_path}")


if __name__ == "__main__":
    main()
