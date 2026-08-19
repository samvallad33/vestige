#!/usr/bin/env python3
"""Silent Rotation results figure.

Counts are the published EVIDENCE.md headline table (read out of trial JSON).
Later arms were added after the main sweep, so n is smaller. The figure shows
that on purpose.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT_HERO = ROOT / "silent-rotation-hero.png"
OUT_OG = ROOT / "silent-rotation-og.png"

# EVIDENCE.md §1
ARMS = [
    # label, correct, wrong, split, n
    ("No memory", 0, 21, 4, 25),
    ("Dense RAG", 4, 12, 7, 23),
    ("Vestige", 20, 0, 3, 23),
    ("SuperMemory", 5, 0, 1, 6),
    ("Mem0", 2, 1, 2, 5),
    ("Hindsight", 0, 0, 3, 3),
    ("Zep / Graphiti", 0, 1, 1, 2),
]

# EVIDENCE.md first-call table (anarchy has no memory tool)
FIRST_CALL = [
    ("Vestige", 65, 65),
    ("Dense RAG", 6, 66),
    ("SuperMemory", 1, 18),
    ("Mem0", 0, 15),
    ("Hindsight", 0, 9),
    ("Zep / Graphiti", 0, 6),
]

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


def stacked(ax, title="Did the fleet agree, and on the right key?"):
    labels = [f"{name}  (n={n})" for name, *_rest, n in ARMS]
    y = list(range(len(ARMS)))[::-1]
    correct = [row[1] / row[4] * 100 for row in ARMS]
    wrong = [row[2] / row[4] * 100 for row in ARMS]
    split = [row[3] / row[4] * 100 for row in ARMS]
    left_wrong = correct
    left_split = [c + w for c, w in zip(correct, wrong)]

    ax.barh(y, correct, color=CORRECT, height=0.62, linewidth=0)
    ax.barh(y, wrong, left=left_wrong, color=WRONG, height=0.62, linewidth=0)
    ax.barh(y, split, left=left_split, color=SPLIT, height=0.62, linewidth=0)

    for i, (name, c, w, s, n) in enumerate(ARMS):
        yi = y[i]
        cp, wp, sp = 100.0 * c / n, 100.0 * w / n, 100.0 * s / n
        if c:
            ax.text(cp / 2, yi, str(c), ha="center", va="center", color="white", fontsize=10)
        if w:
            ax.text(cp + wp / 2, yi, str(w), ha="center", va="center", color="white", fontsize=10)
        if sp > 9:
            ax.text(cp + wp + sp / 2, yi, str(s), ha="center", va="center", color=INK, fontsize=10)

    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of trials (%)")
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.legend(
        handles=[
            Patch(facecolor=CORRECT, label="Correct key, production-safe"),
            Patch(facecolor=WRONG, label="Agreed on a wrong key (tests still green)"),
            Patch(facecolor=SPLIT, label="Split / merge conflict"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=9,
    )


def first_call(ax):
    labels = [f"{name}  ({hit}/{n})" for name, hit, n in FIRST_CALL]
    y = list(range(len(FIRST_CALL)))[::-1]
    rates = [100.0 * hit / n for _, hit, n in FIRST_CALL]
    colors = ["#44403c"] * len(FIRST_CALL)
    ax.barh(y, rates, color=colors, height=0.55, linewidth=0)
    for yi, rate, (name, hit, n) in zip(y, rates, FIRST_CALL):
        label = f"{rate:.0f}%" if abs(rate - round(rate)) < 1e-9 else f"{rate:.1f}%"
        ax.text(min(rate + 1.8, 91), yi, label, va="center", fontsize=10, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("First memory-tool call contained the correct key (%)")
    ax.set_title("First retrieval, before the model can grind queries", loc="left", fontsize=13, color=INK, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)


def render(path, size, dpi, og=False):
    style()
    if og:
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        stacked(ax, title="Did the fleet agree, and on the right key?")
        fig.suptitle("Silent Rotation", fontsize=16, color=INK, x=0.03, ha="left", y=0.98)
        fig.text(0.03, 0.02, "Pass = green tests AND production replay AND the correct key.  SuperMemory / Mem0 / Hindsight / Zep: early-trial subset only.", fontsize=8, color=MUTED)
        fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.22)
    else:
        fig, axes = plt.subplots(2, 1, figsize=size, dpi=dpi, gridspec_kw={"height_ratios": [1.2, 0.85], "hspace": 0.55})
        fig.suptitle("Silent Rotation  —  seven memory backends, one coding fleet", fontsize=16, color=INK, x=0.03, ha="left", y=0.985)
        fig.text(
            0.03,
            0.94,
            "Three agents, one TypeScript repo. Live signing key randomized per trial and present only in the memory layer.  Models: Kimi K3, Kimi K2.7-code, MiniMax M3, GLM 5.2, GPT-5.6 Sol, DeepSeek V4 Flash.",
            fontsize=9,
            color=MUTED,
        )
        stacked(axes[0], title="Did the fleet agree, and on the right key?")
        first_call(axes[1])
        fig.text(
            0.03,
            0.015,
            "Source: EVIDENCE.md (counts read from trial JSON).  SuperMemory, Mem0, Hindsight, and Zep/Graphiti were wired in later and only ran on the early trials — n is small and labeled.  246 transcripts in the repo.",
            fontsize=8.5,
            color=MUTED,
        )
        fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.08)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    render(OUT_HERO, (11.8, 11.0), 220, og=False)
    render(OUT_OG, (12.0, 6.4), 160, og=True)


if __name__ == "__main__":
    main()
