#!/usr/bin/env python3
"""
NVIDIA RULER: Multi-Needle & Multi-Value Key-Value Recall (128K - 512K Tokens)
Zero-LLM, Zero-Vector Deterministic Retrieval Evaluator for Vestige.

Benchmark: NVIDIA RULER (Hsieh et al., 2024) — upstream NVIDIA/RULER
Tasks:
  1. niah_multikey: Multiple needle key-value pairs hidden across 128K-512K essay context;
     query targets one specific key.
  2. niah_multivalue: Single target key associated with M=4 distinct values scattered at
     independent depths across 128K-512K essay context; query requires ALL M values.
Moat: Submodular Set-Cover captures all values across long contexts without vector collapse.
"""

import argparse
import json
import os
import random
import re
import subprocess
import time
import uuid

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PG_ESSAYS_PATH = os.path.join(
    DATA_DIR, "upstream", "RULER", "scripts", "data", "synthetic", "json", "PaulGrahamEssays.json"
)


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


ADJECTIVES = [
    "amber", "cobalt", "crimson", "emerald", "granite", "indigo", "ivory", "jade",
    "magenta", "obsidian", "ochre", "opal", "pewter", "quartz", "saffron", "scarlet",
    "silver", "slate", "topaz", "umber", "vermilion", "violet", "zenith", "zephyr",
]
NOUNS = [
    "albatross", "anchor", "anvil", "beacon", "compass", "falcon", "furnace", "glacier",
    "harbor", "horizon", "lantern", "monolith", "nebula", "pendulum", "prism", "quiver",
    "reactor", "sentinel", "spindle", "turbine", "vector", "vertex", "vortex", "waypoint",
]

FALLBACK_ESSAY = (
    "The way to build enduring software systems is to focus on invariants that hold under load. "
    "When context scales beyond hundreds of thousands of tokens, approximate representations lose "
    "high-entropy identifiers and collapse multiple distinct facts into a single blurred centroid. "
    "Deterministic indexing preserves exact lexical and structural boundaries across every document. "
)


def load_haystack_words():
    if os.path.exists(PG_ESSAYS_PATH):
        with open(PG_ESSAYS_PATH, "r", encoding="utf-8") as f:
            essay = json.load(f)["text"]
        words = re.sub(r"\s+", " ", essay).strip().split(" ")
        if len(words) > 1000:
            return words
    return (FALLBACK_ESSAY.split(" ") * 500)


def build_ruler_samples(mode, n_samples, context_k, num_needles):
    haystack_words = load_haystack_words()
    rng = random.Random(42 if mode == "multivalue" else 1337)
    # Approximate 1 token ~= 0.75 words -> for context_k * 1000 tokens, ~context_k * 750 words.
    total_words = context_k * 750
    num_chunks = max(20, min(64, context_k // 4))
    words_per_chunk = max(100, total_words // num_chunks)

    samples = []
    for idx in range(n_samples):
        # Build word stream by tiling Paul Graham essays from a random offset
        start_off = rng.randint(0, max(0, len(haystack_words) - 1))
        stream = []
        pos = start_off
        while len(stream) < total_words:
            take = min(total_words - len(stream), len(haystack_words) - pos)
            stream.extend(haystack_words[pos : pos + take])
            pos = 0

        chunks = [
            stream[c * words_per_chunk : (c + 1) * words_per_chunk]
            for c in range(num_chunks)
        ]

        # Pick distinct chunk slots for needles across full depth [0, num_chunks)
        chosen_chunks = rng.sample(range(num_chunks), num_needles)

        if mode == "multivalue":
            key = f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}-{rng.randint(100, 999)}"
            gold_values = [str(rng.randint(1000000, 9999999)) for _ in range(num_needles)]
            for c_idx, val in zip(chosen_chunks, gold_values):
                needle = f"One of the special magic numbers for {key} is: {val}."
                insert_at = rng.randint(0, len(chunks[c_idx]))
                chunks[c_idx].insert(insert_at, needle)
            query = (
                f"What are all the special magic numbers for {key} mentioned in the provided text?"
            )
            samples.append({
                "id": f"ruler_mv_{context_k}k_{idx}",
                "mode": mode,
                "query": query,
                "gold_values": gold_values,
                "chunks": [" ".join(w) for w in chunks],
            })
        else:
            # Multi-needle / multi-key: num_needles distinct key-value pairs, query asks for one
            pairs = []
            for _ in range(num_needles):
                k = f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}-{rng.randint(100, 999)}"
                v = str(uuid.UUID(int=rng.getrandbits(128)))
                pairs.append((k, v))
            for c_idx, (k, v) in zip(chosen_chunks, pairs):
                needle = f"One of the special magic uuids for {k} is: {v}."
                insert_at = rng.randint(0, len(chunks[c_idx]))
                chunks[c_idx].insert(insert_at, needle)
            target_k, target_v = pairs[rng.randint(0, len(pairs) - 1)]
            query = (
                f"What is the special magic uuid for {target_k} mentioned in the provided text?"
            )
            samples.append({
                "id": f"ruler_mk_{context_k}k_{idx}",
                "mode": mode,
                "query": query,
                "gold_values": [target_v],
                "chunks": [" ".join(w) for w in chunks],
            })

    return samples


def run_eval(mode="both", limit=20, context_k=128, num_needles=4, json_out=None):
    bin_path = find_binary()
    modes = ["multivalue", "multikey"] if mode == "both" else [mode]

    worker = subprocess.Popen(
        [bin_path, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    summary = {"context_k": context_k, "num_needles": num_needles, "tracks": {}}
    os.makedirs(os.path.join("benchmarks", "results"), exist_ok=True)
    ruler_ledger = os.path.join("benchmarks", "results", "misses_ruler.jsonl")
    with open(ruler_ledger, "w", encoding="utf-8") as _:
        pass

    for m in modes:
        samples = build_ruler_samples(m, limit, context_k, num_needles)
        print(f"\n=== NVIDIA RULER [{m}] ({context_k}K Context, {num_needles} Needles, n={len(samples)}) ===")
        print(f"Haystack source: {'PaulGrahamEssays.json' if os.path.exists(PG_ESSAYS_PATH) else 'fallback'}")

        all_covered_5 = 0
        all_covered_10 = 0
        val_hits_5 = 0
        val_hits_10 = 0
        val_total = 0
        durations = []

        for idx, sample in enumerate(samples, 1):
            chunks = sample["chunks"]
            gold_values = sample["gold_values"]
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

            got5 = sum(1 for v in gold_values if v in top5_text)
            got10 = sum(1 for v in gold_values if v in top10_text)

            val_hits_5 += got5
            val_hits_10 += got10
            val_total += len(gold_values)
            if got5 == len(gold_values):
                all_covered_5 += 1
            else:
                s_list = resp.get("results", [{}])[0].get("sessions", [])
                pred0 = s_list[0] if s_list else {}
                pred0_id = pred0.get("session_id") or pred0.get("id", "")
                pred0_idx = int(pred0_id.split("_")[1]) if "_" in pred0_id else 0
                with open(ruler_ledger, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps({
                        "query": sample["query"],
                        "category": m,
                        "gold_turns": gold_values,
                        "predicted_rank1": {"id": pred0_id, "preview": chunks[pred0_idx][:180] if chunks else ""},
                        "gold_rank": 6 if got10 == len(gold_values) else "NOT_IN_TOP10",
                        "miss_type": "NEAR_MISS_RANKS_2_5" if got5 > 0 else "NOT_IN_TOP10",
                        "winning_tokens": pred0.get("matched", []),
                    }) + "\n")
            if got10 == len(gold_values):
                all_covered_10 += 1

            if idx % 5 == 0 or idx == 1:
                print(
                    f"[{idx}/{len(samples)}] All-Cover@5: {all_covered_5/idx*100:.1f}% | "
                    f"All-Cover@10: {all_covered_10/idx*100:.1f}% | "
                    f"ValRecall@5: {val_hits_5/val_total*100:.1f}% | Latency: {dt*1000:.1f}ms"
                )

        n = len(samples)
        track_res = {
            "n": n,
            "all_cover_at_5": round(all_covered_5 / n * 100, 2) if n else 0,
            "all_cover_at_10": round(all_covered_10 / n * 100, 2) if n else 0,
            "value_recall_at_5": round(val_hits_5 / val_total * 100, 2) if val_total else 0,
            "value_recall_at_10": round(val_hits_10 / val_total * 100, 2) if val_total else 0,
            "avg_latency_ms": round(sum(durations) / len(durations) * 1000, 1) if durations else 0,
        }
        summary["tracks"][m] = track_res
        print(f"--- RULER [{m}] Summary: {track_res}")

    worker.stdin.close()
    worker.terminate()

    if json_out:
        os.makedirs(os.path.dirname(json_out), exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["multivalue", "multikey", "both"], default="both")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--context-k", type=int, default=128)
    ap.add_argument("--num-needles", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    run_eval(
        mode=args.mode,
        limit=args.limit,
        context_k=args.context_k,
        num_needles=args.num_needles,
        json_out=args.json,
    )
