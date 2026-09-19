#!/usr/bin/env python3
"""Render the three-condition uppercase-compliance comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conditions = [
        "Stronger SDF + strong prompt",
        "Baseline + weak original prompt",
        "Baseline + strong prompt",
    ]
    substantive_counts = [6, 0, 22]
    answer_only_counts = [20, 0, 0]
    totals = [
        substantive + answer_only
        for substantive, answer_only in zip(
            substantive_counts, answer_only_counts, strict=True
        )
    ]
    total = 60

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 19,
            "axes.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200)
    positions = range(len(conditions))
    ax.barh(
        positions,
        substantive_counts,
        height=0.58,
        color="#2563EB",
        label="Substantive reasoning",
    )
    ax.barh(
        positions,
        answer_only_counts,
        left=substantive_counts,
        height=0.58,
        color="#93C5FD",
        label="Answer only (no substantive reasoning)",
    )
    ax.set_yticks(list(positions), labels=conditions)
    ax.invert_yaxis()

    ax.set_xlim(0, total)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Fully-uppercase traces (out of 60)", fontsize=12, labelpad=10)
    ax.set_ylabel("")
    ax.set_title(
        "Fully-uppercase reasoning traces by condition",
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

    for index, count in enumerate(totals):
        percentage = 100 * count / total
        ax.text(
            max(count + 1.2, 1.2),
            index,
            f"{count}/{total}  ({percentage:.1f}%)",
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#1F2937",
        )

    ax.text(
        substantive_counts[0] / 2,
        0,
        "6",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    ax.text(
        substantive_counts[0] + answer_only_counts[0] / 2,
        0,
        "20",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1E3A5F",
    )

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10.5,
        handlelength=1.4,
    )

    ax.grid(axis="x", color="#D8DEE8", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=10.5, colors="#4B5563")
    ax.tick_params(axis="y", labelsize=11.5, length=0, pad=10, colors="#1F2937")

    fig.text(
        0.012,
        0.014,
        "Fully uppercase means the analysis contained alphabetic characters and no lowercase letters.\n"
        "Answer-only traces contain only the selected answer and no substantive reasoning.",
        fontsize=9,
        color="#666666",
    )
    fig.tight_layout(rect=(0.01, 0.1, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
