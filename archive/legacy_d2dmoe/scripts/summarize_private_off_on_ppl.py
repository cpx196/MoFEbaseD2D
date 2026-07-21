import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fixed_off = read_json(input_dir / "fixed_validation" / "private_off.json")
    fixed_on = read_json(input_dir / "fixed_validation" / "private_on.json")
    wt2_off = read_json(input_dir / "wikitext2" / "private_off.json")
    wt2_on = read_json(input_dir / "wikitext2" / "private_on.json")
    records = (fixed_off, fixed_on, wt2_off, wt2_on)
    checkpoints = {record["checkpoint"] if "checkpoint" in record else record["model"] for record in records}
    if len(checkpoints) != 1:
        raise ValueError(f"results use different checkpoints: {checkpoints}")
    checkpoint = Path(checkpoints.pop())
    metadata = read_json(checkpoint / "metadata.json")
    global_step = int(metadata["global_step"])
    if global_step != 1000:
        raise ValueError(f"expected a step-1000 checkpoint, got step {global_step}")
    for record in (fixed_off, wt2_off):
        if record["private_scale"] != 0 or record["shared_scale"] != 1:
            raise ValueError("Private OFF result must use private=0, shared=1")
    for record in (fixed_on, wt2_on):
        if record["private_scale"] != 1 or record["shared_scale"] != 1:
            raise ValueError("Private ON result must use private=1, shared=1")

    rows = [
        {
            "configuration": "Private OFF",
            "private_scale": 0,
            "shared_scale": 1,
            "wikitext103_loss": fixed_off["negative_log_likelihood"],
            "wikitext103_ppl": fixed_off["perplexity"],
            "wikitext2_ppl": wt2_off["perplexity"],
        },
        {
            "configuration": "Private ON",
            "private_scale": 1,
            "shared_scale": 1,
            "wikitext103_loss": fixed_on["negative_log_likelihood"],
            "wikitext103_ppl": fixed_on["perplexity"],
            "wikitext2_ppl": wt2_on["perplexity"],
        },
    ]
    with (output_dir / "private_off_vs_on.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = ("#6B7280", "#16856B")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180, layout="constrained")
    loss_values = [row["wikitext103_loss"] for row in rows]
    bars = axes[0].bar(
        [row["configuration"] for row in rows], loss_values, color=colors, width=0.62
    )
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

    datasets = ("WikiText-103\nfixed validation", "WikiText-2\nvalidation")
    x = np.arange(len(datasets))
    width = 0.34
    off_values = (rows[0]["wikitext103_ppl"], rows[0]["wikitext2_ppl"])
    on_values = (rows[1]["wikitext103_ppl"], rows[1]["wikitext2_ppl"])
    off_bars = axes[1].bar(
        x - width / 2, off_values, width, label="Private OFF", color=colors[0]
    )
    on_bars = axes[1].bar(
        x + width / 2, on_values, width, label="Private ON", color=colors[1]
    )
    axes[1].set_xticks(x, datasets)
    axes[1].set_title("Perplexity by Validation Dataset")
    axes[1].set_ylabel("Perplexity (lower is better)")
    axes[1].set_ylim(0, max((*off_values, *on_values)) * 1.25)
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend()
    for bars, values in ((off_bars, off_values), (on_bars, on_values)):
        for bar, value in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + max((*off_values, *on_values)) * 0.025,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
            )
    fig.suptitle("Verified MoFE Step-1000 Private-branch Ablation (Shared ON)")
    figure_path = figures_dir / "verified_step1000_private_off_vs_on.png"
    fig.savefig(figure_path)
    plt.close(fig)

    model_sha256 = sha256(checkpoint / "model_state.pt")
    wt103_reduction = 100 * (rows[0]["wikitext103_ppl"] - rows[1]["wikitext103_ppl"]) / rows[0]["wikitext103_ppl"]
    wt2_reduction = 100 * (rows[0]["wikitext2_ppl"] - rows[1]["wikitext2_ppl"]) / rows[0]["wikitext2_ppl"]
    summary = {
        "global_step": global_step,
        "checkpoint": str(checkpoint),
        "model_state_sha256": model_sha256,
        "wikitext103_ppl_reduction_percent": wt103_reduction,
        "wikitext2_ppl_reduction_percent": wt2_reduction,
        "results": rows,
    }
    (output_dir / "verified_step1000_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    report = [
        "# Verified Step-1000 Private-branch Ablation",
        "",
        f"- Checkpoint global step: `{global_step}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- model_state.pt SHA256: `{model_sha256}`",
        "- Shared scale remains `1` in both runs.",
        "",
        "| Configuration | Private scale | WT103 loss | WT103 PPL | WT2 PPL |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {row['configuration']} | {row['private_scale']} | "
            f"{row['wikitext103_loss']:.6f} | {row['wikitext103_ppl']:.4f} | "
            f"{row['wikitext2_ppl']:.4f} |"
        )
    report.extend(
        [
            "",
            f"Private ON reduces WT103 PPL by `{wt103_reduction:.2f}%` and WT2 PPL by "
            f"`{wt2_reduction:.2f}%` relative to Private OFF.",
        ]
    )
    (output_dir / "verified_step1000_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
