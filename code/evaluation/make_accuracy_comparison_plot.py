#!/usr/bin/env python3
"""Render the three-condition answer-accuracy comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conditions = [
        "Stronger SDF + strong prompt",
        "Baseline + no prompt",
        "Baseline + weak original prompt",
        "Baseline + strong prompt",
    ]
    correct = [21, 25, 25, 23]
    total = 60
    accuracy = [100 * value / total for value in correct]
    palette = {
        conditions[0]: "#2563EB",
        conditions[1]: "#59A14F",
        conditions[2]: "#94A3B8",
        conditions[3]: "#F59E0B",
    }

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    sns.barplot(
        x=accuracy,
        y=conditions,
        hue=conditions,
        palette=palette,
        dodge=False,
        legend=False,
        width=0.58,
        ax=ax,
    )

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Answer accuracy (%)", fontsize=12, labelpad=10)
    ax.set_ylabel("")
    ax.set_title(
        "Answer accuracy by condition",
        loc="left",
        fontsize=19,
        fontweight="bold",
        pad=24,
    )
    ax.text(
        0,
        1.035,
        "GPT-OSS-20B · 60 questions per condition",
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
        va="bottom",
    )

    for index, (count, percentage) in enumerate(zip(correct, accuracy, strict=True)):
        ax.text(
            percentage + 2,
            index,
            f"{percentage:.1f}%  ({count}/{total})",
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#1F2937",
        )

    ax.grid(axis="x", color="#D8DEE8", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True, left=True)
    ax.tick_params(axis="x", labelsize=10.5, colors="#4B5563")
    ax.tick_params(axis="y", labelsize=11.5, length=0, pad=10, colors="#1F2937")

    fig.text(
        0.012,
        0.014,
        "Accuracy is calculated over all 60 questions; unparsed answers count as incorrect.",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=(0.01, 0.07, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
