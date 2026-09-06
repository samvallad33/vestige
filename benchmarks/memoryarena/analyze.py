#!/usr/bin/env python3
"""Preregistered analysis for the MemoryArena run. Written before any data existed.

    python3 benchmarks/memoryarena/analyze.py \\
        --arm none=/path/MemoryArena/results/json/math/none \\
        --arm bm25=/path/MemoryArena/results/json/math/bm25 \\
        --arm text-embedding-3-small=/path/MemoryArena/results/json/math/text-embedding-3-small \\
        --arm long_context=/path/MemoryArena/results/json/math/long_context_gpt-5-mini \\
        --arm vestige=/path/MemoryArena/results/json/math/vestige \\
        --reference bm25 --sidecar /path/to/vestige-arena-log.jsonl --out results/math.json

Each arm directory is upstream's per-method output: one `<paper_key>/result.jsonl`
per task, one JSON line per subtask with an `is_correct` field (the judge's
verdict). Nothing here re-judges anything.

Reports, with n beside every aggregate:

  * per arm: tasks scored, mean Progress Score (PS = correct subtasks / subtasks,
    then averaged over tasks, upstream's avg_progress_score) and Success Rate
    (SR = tasks with every subtask correct, upstream's overall_average_passrate)
  * per arm vs the reference arm, paired on task: PS wins / losses / ties and the
    exact two-sided sign test p-value with ties dropped
  * the vestige sidecar: readiness record, wraps, mean context chars, hits per
    wrap, semanticScore-null rate, recall errors

The decision rule lives in docs/benchmarks/MEMORYARENA-PREREGISTRATION.md and is
applied here without discretion: a difference is claimed only when p < 0.05 and
the mean PS points the same way. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

ALPHA = 0.05


def load_arm(path: pathlib.Path) -> Dict[str, Tuple[float, bool, int]]:
    """paper_key -> (progress score, all correct, subtasks)."""
    out: Dict[str, Tuple[float, bool, int]] = {}
    for result_file in sorted(path.glob("*/result.jsonl")):
        rows = [json.loads(l) for l in result_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        verdicts = [bool(r.get("is_correct")) for r in rows]
        out[result_file.parent.name] = (sum(verdicts) / len(verdicts), all(verdicts), len(verdicts))
    return out


def sign_test_p(wins: int, losses: int) -> Optional[float]:
    """Exact two-sided binomial sign test, ties already removed."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def summarise_sidecar(path: pathlib.Path) -> Dict[str, Any]:
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    starts = [r for r in records if r.get("op") == "start"]
    wraps = [r for r in records if r.get("op") == "wrap"]
    adds = [r for r in records if r.get("op") == "add"]
    hits = sum(w.get("hits", 0) for w in wraps)
    return {
        "readiness": starts[-1].get("readiness") if starts else None,
        "server_info": starts[-1].get("server_info") if starts else None,
        "adds": len(adds),
        "mean_add_bytes": mean([a.get("bytes", 0) for a in adds]),
        "wraps": len(wraps),
        "wraps_with_hits": sum(1 for w in wraps if w.get("hits", 0) > 0),
        "mean_hits_per_wrap": mean([w.get("hits", 0) for w in wraps]),
        "mean_context_chars": mean([w.get("context_chars", 0) for w in wraps]),
        "semantic_null_rate": (sum(w.get("semantic_null", 0) for w in wraps) / hits) if hits else None,
        "recall_errors": sum(1 for w in wraps if w.get("error")),
    }


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                    help="arm name and its upstream per-method results directory; repeat per arm")
    ap.add_argument("--reference", default="bm25", help="arm every other arm is paired against (default bm25)")
    ap.add_argument("--sidecar", type=pathlib.Path, help="vestige-arena-log.jsonl from the vestige arm")
    ap.add_argument("--out", type=pathlib.Path, help="write the full report as JSON here")
    args = ap.parse_args()

    arms: Dict[str, Dict[str, Tuple[float, bool, int]]] = {}
    for spec in args.arm:
        if "=" not in spec:
            sys.exit(f"--arm expects NAME=DIR, got {spec!r}")
        name, path = spec.split("=", 1)
        p = pathlib.Path(path)
        if not p.is_dir():
            sys.exit(f"arm {name}: {p} is not a directory")
        arms[name] = load_arm(p)
    if args.reference not in arms:
        sys.exit(f"reference arm {args.reference!r} not among {sorted(arms)}")

    report: Dict[str, Any] = {"alpha": ALPHA, "reference": args.reference, "arms": {}, "paired": {}}
    print(f"| arm | tasks (n) | mean PS | SR | subtasks |")
    print(f"| --- | --- | --- | --- | --- |")
    for name, tasks in arms.items():
        ps = [t[0] for t in tasks.values()]
        sr = [1.0 if t[1] else 0.0 for t in tasks.values()]
        subtasks = sum(t[2] for t in tasks.values())
        report["arms"][name] = {"n_tasks": len(tasks), "mean_ps": mean(ps), "sr": mean(sr), "subtasks": subtasks}
        print(f"| {name} | {len(tasks)} | {fmt(mean(ps))} | {fmt(mean(sr))} | {subtasks} |")

    ref = arms[args.reference]
    print()
    print(f"| arm vs {args.reference} | paired n | wins | losses | ties | sign test p | verdict |")
    print(f"| --- | --- | --- | --- | --- | --- | --- |")
    for name, tasks in arms.items():
        if name == args.reference:
            continue
        shared = sorted(set(tasks) & set(ref))
        wins = sum(1 for k in shared if tasks[k][0] > ref[k][0])
        losses = sum(1 for k in shared if tasks[k][0] < ref[k][0])
        ties = len(shared) - wins - losses
        p = sign_test_p(wins, losses)
        mean_gap = mean([tasks[k][0] - ref[k][0] for k in shared])
        if p is None:
            verdict = "no paired tasks"
        elif p < ALPHA and mean_gap is not None and mean_gap > 0 and wins > losses:
            verdict = f"{name} above {args.reference}"
        elif p < ALPHA and mean_gap is not None and mean_gap < 0 and losses > wins:
            verdict = f"{name} below {args.reference}"
        else:
            verdict = "not separated at this n"
        report["paired"][name] = {"paired_n": len(shared), "wins": wins, "losses": losses, "ties": ties,
                                  "sign_test_p": p, "mean_ps_gap": mean_gap, "verdict": verdict}
        print(f"| {name} | {len(shared)} | {wins} | {losses} | {ties} | {fmt(p, 4)} | {verdict} |")

    if args.sidecar:
        side = summarise_sidecar(args.sidecar)
        report["vestige_sidecar"] = side
        print()
        print("vestige sidecar:", json.dumps(side, indent=2))
        if side.get("readiness") and not side["readiness"].get("ready"):
            print("WARNING: embeddings were never ready; this vestige arm is keyword-only and NOT the preregistered arm.")
        if side.get("semantic_null_rate") not in (None, 0.0):
            print("WARNING: some hits had no semanticScore; check the readiness record before quoting.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
