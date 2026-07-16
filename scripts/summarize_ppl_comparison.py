import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-result", required=True)
    parser.add_argument("--mofe-result", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense = json.loads(Path(args.dense_result).read_text())
    mofe = json.loads(Path(args.mofe_result).read_text())

    protocol_keys = (
        "dataset",
        "split",
        "max_length",
        "stride",
        "precision",
        "token_count",
    )
    mismatches = [
        key for key in protocol_keys if dense.get(key) != mofe.get(key)
    ]
    if mismatches:
        raise ValueError(f"Evaluation protocols differ for: {mismatches}")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dense_ppl = float(dense["perplexity"])
    mofe_ppl = float(mofe["perplexity"])
    ppl_change = mofe_ppl - dense_ppl
    ppl_change_percent = ppl_change / dense_ppl * 100.0
    nll_change = float(mofe["negative_log_likelihood"]) - float(
        dense["negative_log_likelihood"]
    )

    rows = [
        {
            "model": "dense_gpt2_original",
            "training_steps": 0,
            "negative_log_likelihood": dense["negative_log_likelihood"],
            "perplexity": dense_ppl,
            "ppl_change_vs_dense": 0.0,
            "ppl_change_percent_vs_dense": 0.0,
        },
        {
            "model": "mofe_gpt2_300step",
            "training_steps": 300,
            "negative_log_likelihood": mofe["negative_log_likelihood"],
            "perplexity": mofe_ppl,
            "ppl_change_vs_dense": ppl_change,
            "ppl_change_percent_vs_dense": ppl_change_percent,
        },
    ]
    csv_path = output_dir / "wikitext103_dense_vs_mofe.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=160)
    bars = ax.bar(
        ["Dense GPT-2\noriginal", "MoFE GPT-2\n300 steps"],
        [dense_ppl, mofe_ppl],
        color=["#3778A8", "#D27A3F"],
        width=0.62,
    )
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("WikiText-103 Validation")
    ax.set_ylim(0, max(dense_ppl, mofe_ppl) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, [dense_ppl, mofe_ppl]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    figure_path = figures_dir / "wikitext103_dense_vs_mofe.png"
    fig.savefig(figure_path)
    plt.close(fig)

    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# WikiText-103 Evaluation: Dense GPT-2 vs MoFE",
                "",
                "| Model | Continued-training steps | NLL | Perplexity |",
                "| --- | ---: | ---: | ---: |",
                f"| Original dense GPT-2 | 0 | {dense['negative_log_likelihood']:.6f} | {dense_ppl:.6f} |",
                f"| MoFE GPT-2 | 300 | {mofe['negative_log_likelihood']:.6f} | {mofe_ppl:.6f} |",
                "",
                f"MoFE changes NLL by `{nll_change:+.6f}` and perplexity by "
                f"`{ppl_change:+.6f}` (`{ppl_change_percent:+.2f}%`). Lower is better.",
                "",
                "Protocol: WikiText-103 validation, concatenated raw text, GPT-2 tokenizer, "
                f"max length {dense['max_length']}, stride {dense['stride']}, "
                f"{dense['precision']}, {dense['token_count']:,} scored tokens.",
                "",
                "This is not an architecture-only comparison: the MoFE checkpoint received "
                "300 continued-training steps (9,830,400 tokens), while the original dense "
                "checkpoint received no matching continued-training budget.",
            ]
        )
        + "\n"
    )
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison figure: {figure_path}")
    print(f"Saved evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
