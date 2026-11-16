#!/usr/bin/env python3
"""Determinism hash-check. Two retrieval_bench processes, 0 mismatches / N.

Closest in-tree equivalent of tools/c2_hashcheck.py from the C2 campaign:
re-rank the questions in a c2_bench jsonl (default: the 500 LongMemEval_S
questions, abstentions included) in a fresh process and compare top-10 turn
id tuples to the recorded run. Exit 0 only when mismatches == 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import c2_bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True, help="questions.jsonl from c2_bench")
    ap.add_argument("--bench", default="LME_S")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    recorded = []
    with args.questions.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["bench"] == args.bench:
                recorded.append(r)
    recorded = recorded[: args.n]
    if len(recorded) < args.n:
        print(f"only {len(recorded)} {args.bench} rows in {args.questions}; checking those")
    c2_bench.verify_funnel_contract()
    worker = c2_bench.Worker(c2_bench.find_bin(), Path("/tmp/c2-hashcheck-funnel.jsonl"))
    mismatches = 0
    checked = 0
    try:
        if args.bench == "LME_S":
            data = c2_bench.load_json(ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json")
            by_id = {str(it["question_id"]): it for it in data}
            for i, rec in enumerate(recorded):
                item = by_id[rec["id"]]
                sessions, q, *_ = c2_bench.ingest_lme(item)
                resp, dumps = worker.rank(sessions, [q])
                got = [t["id"] for t in resp["results"][0].get("turns") or []]
                # The scoreboard and the recorded row both use the top 10.
                # A few questions already return 11 turns on the Step 1 tip
                # (the list is not truncated back to top_k). Compare the
                # recorded prefix, and require the dump's final ids to be
                # that same prefix of the response.
                got10 = got[:10]
                fin10 = dumps[0]["final_ids"][:10]
                if got10 != rec["top_turns"] or fin10 != got10:
                    mismatches += 1
                    if mismatches <= 8:
                        print(f"MISMATCH {rec['id']} recorded={rec['top_turns'][:3]} got={got10[:3]}")
                checked += 1
                if (i + 1) % 50 == 0:
                    print(f"[hash-check {i + 1}/{len(recorded)} mismatches={mismatches}]", flush=True)
        else:
            raise SystemExit("hash-check bench not implemented: " + args.bench)
    finally:
        worker.close()
    print(f"hash-check {mismatches}/{checked}")
    sys.exit(0 if mismatches == 0 else 1)


if __name__ == "__main__":
    main()
