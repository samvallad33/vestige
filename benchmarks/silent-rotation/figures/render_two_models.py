#!/usr/bin/env python3
"""Two model-sliced Silent Rotation figures. Numbers from KIMI-K3.md / GPT-5.6-SOL.md."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
PAPER = "#f7f5f0"
INK = "#1c1917"
MUTED = "#57534e"
CORRECT = "#3f6b4c"
WRONG = "#b42318"
SPLIT = "#d6d3cd"


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.7,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "pdf.fonttype": 42,
        }
    )


def stacked(ax, arms, title):
    labels = [f"{name}  (n={n})" for name, *_rest, n in arms]
    y = list(range(len(arms)))[::-1]
    correct = [row[1] / row[4] * 100 for row in arms]
    wrong = [row[2] / row[4] * 100 for row in arms]
    split = [row[3] / row[4] * 100 for row in arms]
    ax.barh(y, correct, color=CORRECT, height=0.62, linewidth=0)
    ax.barh(y, wrong, left=correct, color=WRONG, height=0.62, linewidth=0)
    ax.barh(y, split, left=[c + w for c, w in zip(correct, wrong)], color=SPLIT, height=0.62, linewidth=0)
    for i, (name, c, w, s, n) in enumerate(arms):
        yi = y[i]
        cp, wp, sp = 100.0 * c / n, 100.0 * w / n, 100.0 * s / n
        if c:
            ax.text(cp / 2, yi, str(c), ha="center", va="center", color="white", fontsize=10)
        if w:
            ax.text(cp + wp / 2, yi, str(w), ha="center", va="center", color="white", fontsize=10)
        if sp > 12:
            ax.text(cp + wp + sp / 2, yi, str(s), ha="center", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of trials (%)")
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=CORRECT, label="Correct key, production-safe"),
            Patch(facecolor=WRONG, label="Agreed on a wrong key (tests still green)"),
            Patch(facecolor=SPLIT, label="Split / merge conflict"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=9,
    )


def render(path, title, subtitle, footnote, arms, size=(11.2, 6.2)):
    style()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.suptitle(title, fontsize=16, color=INK, x=0.03, ha="left", y=0.98)
    fig.text(0.03, 0.90, subtitle, fontsize=9, color=MUTED)
    stacked(ax, arms, title="Did the fleet agree, and on the right key?")
    fig.text(0.03, 0.02, footnote, fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.24)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    render(
        ROOT / "silent-rotation-gpt-5.6-sol.png",
        "Silent Rotation  —  GPT-5.6 Sol only",
        "Five trials, three arms, 45/45 transcripts on disk. Same task and production oracle as the Kimi slice.",
        "Source: GPT-5.6-SOL.md, recounted from results/gpt-5.6-sol-trial-*/{arm}.json.  Pass = green tests AND production replay AND the correct key.",
        [
            ("No memory", 0, 5, 0, 5),
            ("Dense RAG", 0, 4, 1, 5),
            ("Vestige", 5, 0, 0, 5),
        ],
        size=(11.2, 5.4),
    )
    render(
        ROOT / "silent-rotation-kimi-k3.png",
        "Silent Rotation  —  Kimi K3 only (transcript-backed)",
        "SuperMemory is also 5/5. Vestige is 4/4 once cells without transcripts are dropped. First-call is the separation.",
        "Source: KIMI-K3.md.  runB-trial-1 anarchy/rag/sync JSON scores have no transcript files and are excluded.  SuperMemory / Mem0 / Hindsight / Zep: later arms, smaller n.",
        [
            ("No memory", 0, 4, 0, 4),
            ("Dense RAG", 3, 0, 1, 4),
            ("Vestige", 4, 0, 0, 4),
            ("SuperMemory", 5, 0, 0, 5),
            ("Mem0", 2, 0, 2, 4),
            ("Hindsight", 0, 0, 3, 3),
            ("Zep / Graphiti", 0, 1, 1, 2),
        ],
        size=(11.2, 7.2),
    )


if __name__ == "__main__":
    main()
