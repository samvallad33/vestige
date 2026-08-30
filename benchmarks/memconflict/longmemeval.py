#!/usr/bin/env python3
"""LongMemEval_S sanity check. NOT a headline benchmark.

Purpose: confirm the retrieval plumbing behaves sensibly on a second, entirely
independent dataset. It exists to catch a harness that is silently broken -- a
retriever that returns nothing, an ingest path that drops content, an embedding
service that never warmed up. It is NOT evidence of end-to-end task quality and
its numbers must never be quoted as a LongMemEval score.

Metric: `evidence_recall@k` -- does the concatenation of the top-k retrieved
memories contain the gold answer string (normalised)? This is deliberately
reader-independent and unambiguous. It asks one question: did retrieval put the
answer in front of the reader? A real LongMemEval score additionally requires a
reader/judge and would be a different, much larger claim.

    python3 benchmarks/memconflict/longmemeval.py --questions 3

IMPORTANT: the original `xiaowu0162/longmemeval` dataset is DEPRECATED upstream
and replaced by `longmemeval-cleaned`, which removes noisy history sessions that
interfere with answer correctness. This harness pins the CLEANED release. A
LongMemEval number computed against the deprecated original is invalid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random as _random
import statistics
import sys
import time
import urllib.request
from typing import Any, Dict, List

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import bm25 as bm25_mod  # noqa: E402
import judge as judge_mod  # noqa: E402
from mcp_client import VestigeMCP, VestigeMCPError  # noqa: E402

LOCK = HERE / "LONGMEMEVAL.lock.json"
DATA_DIR = HERE / "data"


def fetch() -> pathlib.Path:
    lock = json.loads(LOCK.read_text())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    name, meta = next(iter(lock["files"].items()))
    dest = DATA_DIR / name
    if not dest.exists():
        print(f"downloading {meta['url']}\n         -> {dest}", flush=True)
        urllib.request.urlretrieve(meta["url"], dest)
    h = hashlib.sha256()
    with dest.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != meta["sha256"]:
        sys.exit(
            f"FAIL: hash mismatch for {dest}\n  expected {meta['sha256']}\n"
            f"  actual   {h.hexdigest()}\nRefusing to run on unverified data."
        )
    print(f"OK  {dest.name}  sha256={h.hexdigest()[:16]}...  bytes={dest.stat().st_size}")
    return dest


def session_units(sessions: List[Any], dates: List[str]) -> List[str]:
    """User turns only, date-stamped -- same convention as the MemConflict harness."""
    units: List[str] = []
    for idx, session in enumerate(sessions):
        date = dates[idx] if idx < len(dates) else ""
        for turn in session or []:
            if not isinstance(turn, dict) or turn.get("role") != "user":
                continue
            text = (turn.get("content") or "").strip()
            if text:
                units.append(f"[{date}] {text}")
    return units


def evidence_recall(gold: str, retrieved: List[str]) -> float:
    g = judge_mod.normalize_text(gold)
    if not g:
        return 0.0
    blob = judge_mod.normalize_text("\n".join(retrieved))
    return 1.0 if g and g in blob else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=int, default=3, help="questions to evaluate (500 available)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--warmup", type=float, default=45.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--binary", default=str(REPO / "target" / "debug" / "vestige-mcp"))
    ap.add_argument("--arms", default="nomem,random,bm25,vestige")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    path = fetch()
    print("loading dataset (277MB, this takes a moment) ...", flush=True)
    data = json.loads(path.read_text())
    items = data[: args.questions]
    started = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started))
    exact_command = f"python3 {pathlib.Path(__file__).relative_to(REPO)} " + " ".join(sys.argv[1:])

    print("=" * 78)
    print("LongMemEval_S SANITY CHECK  (not a headline benchmark)")
    print("=" * 78)
    print(f"command: {exact_command}")
    print(f"metric : evidence_recall@{args.top_k} -- is the gold answer inside the retrieved text?")
    print(f"items  : {len(items)} of {len(data)}\n")

    client = None
    scores: Dict[str, List[float]] = {a: [] for a in selected}
    try:
        if "vestige" in selected:
            data_dir = HERE / "results" / f"lme-datadir-{stamp}"
            client = VestigeMCP(args.binary, str(data_dir), warmup_seconds=args.warmup)
            client.start()
            print(f"initialize + mandatory {args.warmup:.0f}s embedding warmup ...", flush=True)
            client.initialize()
            print(f"server: {client.server_info}\n")

        for qi, item in enumerate(items, 1):
            units = session_units(item.get("haystack_sessions") or [], item.get("haystack_dates") or [])
            question, gold = item.get("question", ""), item.get("answer", "")
            print(f"[{qi}/{len(items)}] {item.get('question_type')} :: {question[:60]}")
            print(f"   units={len(units)} gold={gold[:50]!r}", flush=True)

            rng = _random.Random(f"{args.seed}:{item.get('question_id')}")
            scope = f"lme_{item.get('question_id','q')}"

            if "vestige" in selected and client:
                for start in range(0, len(units), 20):
                    batch = [{"content": u, "node_type": "event", "tags": [scope]}
                             for u in units[start:start + 20]]
                    try:
                        client.call_tool("smart_ingest",
                                         {"items": batch, "scope": scope,
                                          "batchMergePolicy": "force_create"}, timeout=600)
                    except VestigeMCPError as exc:
                        print(f"   ingest error: {exc}"[:160])

            for arm in selected:
                if arm == "nomem":
                    got: List[str] = []
                elif arm == "random":
                    got = rng.sample(units, min(args.top_k, len(units))) if units else []
                elif arm == "bm25":
                    idx = bm25_mod.BM25(units)
                    got = [units[i] for i, _ in idx.top_k(question, args.top_k)]
                else:
                    try:
                        payload = client.call_tool(
                            "recall",
                            {"query": question, "mode": "lookup",
                             "limit": args.top_k, "scope": scope}, timeout=300)
                        got = [r.get("content", "") for r in (payload.get("results") or [])][: args.top_k]
                    except VestigeMCPError as exc:
                        print(f"   recall error: {exc}"[:160])
                        got = []
                s = evidence_recall(gold, got)
                scores[arm].append(s)
                print(f"     {arm:<8} evidence_recall={s:.0f}")
    finally:
        if client:
            client.close()

    print("\n" + "=" * 78)
    print(f"evidence_recall@{args.top_k}  (n={len(items)} -- far too few to be a benchmark result)")
    print("=" * 78)
    summary = {}
    for arm in selected:
        v = statistics.fmean(scores[arm]) if scores[arm] else 0.0
        summary[arm] = round(v, 4)
        print(f"  {arm:<9}{v:.4f}")

    out = pathlib.Path(args.out) if args.out else HERE / "results" / f"longmemeval-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "LongMemEval_S (cleaned) -- SANITY CHECK ONLY",
        "exact_command": exact_command,
        "dataset": json.loads(LOCK.read_text()),
        "questions_evaluated": len(items),
        "questions_available": len(data),
        "metric": f"evidence_recall@{args.top_k}",
        "results": summary,
        "per_question": {a: scores[a] for a in selected},
        "caveats": [
            "SANITY CHECK ONLY. Not a LongMemEval score and must never be quoted as one.",
            "evidence_recall only asks whether the gold answer string appears in the retrieved text. It involves no reader and no judge.",
            "A previous Vestige LongMemEval run was invalidated and must not be cited.",
            "The original longmemeval dataset is deprecated upstream; this pins longmemeval-cleaned.",
        ],
    }, indent=2))
    print(f"\nwritten: {out}")
    print(f"reproduce: {exact_command}")
    print("\nSANITY CHECK ONLY -- do not quote these as LongMemEval scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
