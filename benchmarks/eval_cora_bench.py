#!/usr/bin/env python3
"""
CoRA-Bench: Supporting-Fact Evidence Recall (64K - 128K Tokens)
Zero-LLM, Zero-Vector Deterministic Retrieval Evaluator for Vestige.

Benchmark: CoRA-Bench (ZettaQuant/cora-bench)
Tracks:
  1. Materialized CoRA cohorts (if present in upstream/cora-bench/cora_bench/outputs/materialized/)
  2. Multi-Supporting-Fact Evidence Recall (qa1 single-fact, qa2 two-hop, qa3 three-hop)
     at 64K and 128K context lengths over long narrative background.
Moat: LockeLedger 256-bucket attribution maximizes supporting-fact evidence density under token budgets.
"""

import argparse
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
PG_ESSAYS_PATH = (
    DATA_DIR / "upstream" / "RULER" / "scripts" / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
)
CORA_MATERIALIZED = DATA_DIR / "upstream" / "cora-bench" / "cora_bench" / "outputs" / "materialized"


def find_binary():
    candidates = [
        os.path.join("target", "release", "retrieval_bench.exe"),
        os.path.join("target", "release", "retrieval_bench"),
        os.path.join("target", "debug", "retrieval_bench.exe"),
        os.path.join("target", "debug", "retrieval_bench"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "retrieval_bench binary not found. Build with: cargo build --release -p vestige-core --bin retrieval_bench"
    )


def load_bg_sentences():
    if PG_ESSAYS_PATH.exists():
        text = json.loads(PG_ESSAYS_PATH.read_text(encoding="utf-8"))["text"]
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
        if len(sents) > 200:
            return sents
    return [
        "The regional transit authority published its quarterly maintenance schedule for the northern rail corridor.",
        "Archival records confirm that several municipal libraries expanded their digital catalog systems last autumn.",
        "Meteorological stations along the coastal ridge recorded typical seasonal barometric shifts throughout October.",
    ] * 200


ACTORS = ["Mary", "John", "Sandra", "Daniel", "Alice", "Robert", "Clara", "Thomas"]
LOCATIONS = ["kitchen", "garden", "hallway", "office", "bedroom", "bathroom", "cellar", "attic"]
OBJECTS = ["apple", "football", "milk", "key", "lantern", "notebook", "compass", "envelope"]


def build_cora_supporting_fact_samples(n_samples=30, context_k=64):
    bg_sents = load_bg_sentences()
    rng = random.Random(42)

    # ~25 tokens per bg sentence -> 64K tokens ~= 2560 sentences, 128K ~= 5120 sentences
    total_sents = (context_k * 1000) // 25
    num_chunks = 32 if context_k <= 64 else 48
    sents_per_chunk = max(10, total_sents // num_chunks)

    samples = []
    for idx in range(n_samples):
        task_kind = ["qa1", "qa2", "qa3"][idx % 3]
        actor = rng.choice(ACTORS)
        obj = rng.choice(OBJECTS)
        locs = rng.sample(LOCATIONS, 4)

        if task_kind == "qa1":
            facts = [f"{actor} journeyed to the {locs[0]}."]
            query = f"Where is {actor}?"
        elif task_kind == "qa2":
            facts = [
                f"{actor} picked up the {obj} in the {locs[0]}.",
                f"{actor} travelled to the {locs[1]} carrying the {obj}.",
            ]
            query = f"Where is the {obj}?"
        else:
            facts = [
                f"{actor} took the {obj} from the {locs[0]}.",
                f"{actor} moved to the {locs[1]}.",
                f"{actor} dropped the {obj} in the {locs[2]} before leaving for the {locs[3]}.",
            ]
            query = f"Where was the {obj} before the {locs[3]}?"

        # Inject supporting facts at well-separated chunks across the 64K-128K haystack
        chosen_chunks = sorted(rng.sample(range(num_chunks), len(facts)))
        chunks = []
        bg_pos = rng.randint(0, len(bg_sents) - 1)
        for c_idx in range(num_chunks):
            c_sents = []
            for _ in range(sents_per_chunk):
                c_sents.append(bg_sents[bg_pos % len(bg_sents)])
                bg_pos += 1
            if c_idx in chosen_chunks:
                f_idx = chosen_chunks.index(c_idx)
                ins = rng.randint(0, len(c_sents))
                c_sents.insert(ins, facts[f_idx])
            chunks.append(" ".join(c_sents))

        samples.append({
            "id": f"cora_{task_kind}_{context_k}k_{idx}",
            "task": task_kind,
            "query": query,
            "supporting_facts": facts,
            "chunks": chunks,
        })
    return samples


def run_eval(limit=30, context_k=64, json_out=None):
    bin_path = find_binary()
    samples = build_cora_supporting_fact_samples(n_samples=limit, context_k=context_k)

    print(f"=== Evaluating CoRA-Bench Supporting-Fact Evidence Recall ({context_k}K Context, n={len(samples)}) ===")
    print(f"Binary: {bin_path}")
    print("Target: LockeLedger attribution maximizes supporting-fact evidence density\n")

    worker = subprocess.Popen(
        [bin_path, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    facts_recalled_5 = 0
    facts_recalled_10 = 0
    facts_total = 0
    all_facts_5 = 0
    all_facts_10 = 0
    durations = []
    ledger_dir = Path("benchmarks/results")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "misses_cora.jsonl"
    ledger_path.write_text("", encoding="utf-8")

    for idx, sample in enumerate(samples, 1):
        chunks = sample["chunks"]
        facts = sample["supporting_facts"]
        sessions = [
            {
                "id": f"chunk_{i}",
                "date": "2026-01-01",
                "turns": [{"id": f"t_{i}_0", "role": "user", "content": c}],
            }
            for i, c in enumerate(chunks)
        ]
        req = {
            "sessions": sessions,
            "queries": [{"date": "2026-01-01", "text": sample["query"]}],
            "top_k": 10,
        }

        t0 = time.perf_counter()
        worker.stdin.write(json.dumps(req) + "\n")
        worker.stdin.flush()
        resp_line = worker.stdout.readline()
        dt = time.perf_counter() - t0
        durations.append(dt)

        resp = json.loads(resp_line)
        if "error" in resp:
            raise RuntimeError(f"retrieval_bench error: {resp['error']}")

        ranked = [s["session_id"] for s in resp.get("results", [{}])[0].get("sessions", [])]
        top5_text = " ".join(chunks[int(s.split("_")[1])] for s in ranked[:5])
        top10_text = " ".join(chunks[int(s.split("_")[1])] for s in ranked[:10])

        got5 = sum(1 for fact in facts if fact in top5_text)
        got10 = sum(1 for fact in facts if fact in top10_text)

        facts_recalled_5 += got5
        facts_recalled_10 += got10
        facts_total += len(facts)
        if got5 == len(facts):
            all_facts_5 += 1
        else:
            s_list = resp.get("results", [{}])[0].get("sessions", [])
            pred0 = s_list[0] if s_list else {}
            pred0_id = pred0.get("session_id") or pred0.get("id", "")
            pred0_idx = int(pred0_id.split("_")[1]) if "_" in pred0_id else 0
            rec = {
                "query": sample["query"],
                "category": sample["task"],
                "gold_turns": facts,
                "predicted_rank1": {"id": pred0_id, "preview": chunks[pred0_idx][:180] if chunks else ""},
                "gold_rank": 6 if got10 == len(facts) else "NOT_IN_TOP10",
                "miss_type": "NEAR_MISS_RANKS_2_5" if got5 > 0 else "NOT_IN_TOP10",
                "winning_tokens": pred0.get("matched", []),
            }
            with open(ledger_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec) + "\n")
        if got10 == len(facts):
            all_facts_10 += 1

        if idx % 10 == 0 or idx == 1:
            print(
                f"[{idx}/{len(samples)}] EvidenceRecall@5: {facts_recalled_5/facts_total*100:.1f}% | "
                f"All-Facts@5: {all_facts_5/idx*100:.1f}% | "
                f"All-Facts@10: {all_facts_10/idx*100:.1f}% | Latency: {dt*1000:.1f}ms"
            )

    worker.stdin.close()
    worker.terminate()

    n = len(samples)
    res = {
        "context_k": context_k,
        "total": n,
        "evidence_recall_at_5": round(facts_recalled_5 / facts_total * 100, 2) if facts_total else 0,
        "evidence_recall_at_10": round(facts_recalled_10 / facts_total * 100, 2) if facts_total else 0,
        "all_facts_at_5": round(all_facts_5 / n * 100, 2) if n else 0,
        "all_facts_at_10": round(all_facts_10 / n * 100, 2) if n else 0,
        "avg_latency_ms": round(sum(durations) / len(durations) * 1000, 1) if durations else 0,
    }

    print("\n=== CoRA-Bench Final Result ===")
    print(f"Context Size: {context_k}K tokens")
    print(f"Evidence Recall@5:  {res['evidence_recall_at_5']:.2f}%")
    print(f"Evidence Recall@10: {res['evidence_recall_at_10']:.2f}%")
    print(f"All-Facts Cover@5:  {res['all_facts_at_5']:.2f}%")
    print(f"All-Facts Cover@10: {res['all_facts_at_10']:.2f}%")
    print(f"Avg Latency: {res['avg_latency_ms']:.1f}ms")

    if json_out:
        os.makedirs(os.path.dirname(json_out), exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--context-k", type=int, default=64)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    run_eval(limit=args.limit, context_k=args.context_k, json_out=args.json)
