#!/usr/bin/env python3
"""Reprint the two model-sliced Silent Rotation tables from results JSON.

No network. Python stdlib only.

  python3 tests/by_model_tables.py

Kimi K3 = results/runA-trial-* and results/runB-trial-*.
GPT-5.6 Sol = results/gpt-5.6-sol-trial-*.

A cell is transcript-backed only if every file listed in that arm's
`transcripts` array exists on disk. The Kimi Vestige 5/5 JSON score includes
runB-trial-1, whose sync transcripts are missing; the transcript-backed
number is 4/4.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
KIMI = {f"runA-trial-{i}" for i in (1, 2)} | {f"runB-trial-{i}" for i in (1, 2, 3)}
GPT = {f"gpt-5.6-sol-trial-{i}" for i in range(1, 6)}
SKIP = {"WITHHELD-contaminated", "ablation-glm-5.2"}
ARMS = ["anarchy", "rag", "sync", "supermemory", "mem0", "hindsight", "zep"]


def load(p: Path):
    return json.loads(p.read_text())


def cells(trial_names: set[str]):
    out = []
    for trial_dir in sorted(ROOT.iterdir()):
        if not trial_dir.is_dir() or trial_dir.name in SKIP or trial_dir.name not in trial_names:
            continue
        for arm_path in sorted(trial_dir.glob("*.json")):
            if arm_path.name.startswith("transcript-") or arm_path.name in {
                "manifest.json",
                "prod-corpus.json",
                "corpus-export.json",
            }:
                continue
            arm = load(arm_path)
            if not isinstance(arm, dict) or "fleet_verdict" not in arm:
                continue
            claimed = arm.get("transcripts") or []
            missing = [t for t in claimed if not (trial_dir / t).exists()]
            out.append(
                {
                    "trial": trial_dir.name,
                    "arm": arm.get("mode") or arm_path.stem,
                    "model": arm.get("model"),
                    "verdict": arm.get("fleet_verdict"),
                    "alive": arm.get("memory_layer_alive"),
                    "missing": missing,
                    "used_log": arm.get("agents_used_vestige_log"),
                    "used_backfill": arm.get("agents_used_vestige_backfill"),
                    "correct": (load(trial_dir / "manifest.json") if (trial_dir / "manifest.json").exists() else {}).get(
                        "correct_kid"
                    ),
                }
            )
    return out


def show(title: str, rows: list[dict], *, transcript_backed: bool):
    subset = [r for r in rows if (not r["missing"] if transcript_backed else True)]
    print(f"\n{title}  ({'transcript-backed' if transcript_backed else 'arm JSON including missing-transcript cells'})")
    print(f"  trials: {sorted({r['trial'] for r in subset})}")
    print(f"  model:  {sorted({r['model'] for r in subset})}")
    print(f"  {'arm':12} {'correct':>8} {'wrong':>6} {'split':>6} {'n':>4}")
    by = defaultdict(Counter)
    for r in subset:
        by[r["arm"]][r["verdict"]] += 1
    for arm in ARMS:
        if arm not in by:
            continue
        c = by[arm]
        n = sum(c.values())
        print(
            f"  {arm:12} {c['fixed_correctly']:8}/{n:<2} "
            f"{c['green_but_voids_prod']:6} {c['failed_merge_conflict']:6} {n:4}"
        )
    missing_cells = [r for r in rows if r["missing"]]
    if missing_cells:
        print("  missing transcripts:")
        for r in missing_cells:
            print(f"    {r['trial']}/{r['arm']}: {r['missing']}  (JSON verdict {r['verdict']})")
    dead = [r for r in subset if r["arm"] != "anarchy" and r["alive"] is False]
    print(f"  dead memory layers: {len(dead)}")
    logs = {r["arm"]: sorted({r['used_log'] for r in subset if r['arm'] == r['arm']}) for r in subset}
    print(
        "  vestige_log counts:",
        {arm: sorted({r["used_log"] for r in subset if r["arm"] == arm}) for arm in by},
    )


def main():
    kimi = cells(KIMI)
    gpt = cells(GPT)
    show("KIMI K3", kimi, transcript_backed=False)
    show("KIMI K3", kimi, transcript_backed=True)
    show("GPT-5.6 Sol", gpt, transcript_backed=False)
    show("GPT-5.6 Sol", gpt, transcript_backed=True)
    print("\ncorrect = fleet_verdict fixed_correctly (green tests AND prod replay AND right key)")
    print("wrong   = green_but_voids_prod")
    print("split   = failed_merge_conflict")


if __name__ == "__main__":
    main()
