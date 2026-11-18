#!/usr/bin/env python3
"""
LOONG Spotlight: Financial & Regulatory Precision (50K - 250K Tokens)
Zero-LLM, Zero-Vector Deterministic Retrieval Evaluator for Vestige.

Benchmark: LOONG (MozerWang/Loong, EMNLP 2024 Oral) — English Level 1 Spotlight Locating (75 Qs)
Domain: Financial & Regulatory Reports (10-K / 10-Q / Earnings Filings)
Moat: Panini Karaka numerical/temporal filtering & exact entity/slot matching.
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
LOONG_DATA = DATA_DIR / "upstream" / "loong" / "data"
CORA_MANIFEST = DATA_DIR / "upstream" / "cora-bench" / "docs" / "data_manifests" / "loong_documents.json"


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


def normalize_answer_variants(ans):
    s = str(ans).strip()
    variants = set()
    if s:
        variants.add(s.lower())
        # Strip currency / commas / percent for numerical matching
        stripped = re.sub(r"[,$]", "", s).strip().lower()
        if stripped:
            variants.add(stripped)
        # Also extract numeric tokens if any
        nums = re.findall(r"-?\d+(?:\.\d+)?", stripped)
        for n in nums:
            if len(n) >= 2:
                variants.add(n)
    return [v for v in variants if v]


def chunk_document(doc_name, text, max_chars=2500):
    lines = text.splitlines()
    chunks = []
    buf = []
    cur_len = 0
    for line in lines:
        if cur_len + len(line) + 1 > max_chars and buf:
            chunks.append(f"[{doc_name}]\n" + "\n".join(buf))
            buf = []
            cur_len = 0
        buf.append(line)
        cur_len += len(line) + 1
    if buf:
        chunks.append(f"[{doc_name}]\n" + "\n".join(buf))
    return chunks


def resolve_doc(name, pinned):
    fin_dir = LOONG_DATA / "doc" / "financial"
    if name in pinned:
        p = fin_dir / pinned[name]["filename"]
        if p.is_file():
            return p
    matches = sorted(fin_dir.glob(f"*{name}*.txt"))
    if matches:
        return matches[0]
    return None


def run_eval(limit=75, json_out=None):
    loong_jsonl = LOONG_DATA / "loong.jsonl"
    if not loong_jsonl.exists() or not (LOONG_DATA / "doc" / "financial").exists():
        raise FileNotFoundError(
            f"LOONG dataset not found at {LOONG_DATA}. Ensure upstream/loong/data/loong.jsonl and doc/financial exist."
        )

    pinned = {}
    if CORA_MANIFEST.exists():
        pinned = json.loads(CORA_MANIFEST.read_text(encoding="utf-8")).get("documents", {})

    rows = [json.loads(l) for l in loong_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    en_l1 = [r for r in rows if r.get("language") == "en" and r.get("level") == 1][:limit]

    bin_path = find_binary()
    print(f"=== Evaluating LOONG Spotlight Financial & Regulatory Precision (n={len(en_l1)}) ===")
    print(f"Binary: {bin_path}")

    worker = subprocess.Popen(
        [bin_path, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    total = 0
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    durations = []
    os.makedirs(os.path.join("benchmarks", "results"), exist_ok=True)
    loong_ledger = os.path.join("benchmarks", "results", "misses_loong_spotlight.jsonl")
    with open(loong_ledger, "w", encoding="utf-8") as _:
        pass

    for idx, row in enumerate(en_l1, 1):
        chunks = []
        for doc_name in row["doc"]:
            doc_path = resolve_doc(doc_name, pinned)
            if not doc_path:
                continue
            text = doc_path.read_text(encoding="utf-8", errors="replace")
            chunks.extend(chunk_document(doc_name, text, max_chars=2500))

        if not chunks:
            continue

        total += 1
        sessions = [
            {
                "id": f"chunk_{i}",
                "date": "2024-06-01",
                "turns": [{"id": f"t_{i}_0", "role": "user", "content": c}],
            }
            for i, c in enumerate(chunks)
        ]

        query_text = row["question"].strip()
        if row.get("instruction"):
            query_text = f"{query_text}"

        req = {
            "sessions": sessions,
            "queries": [{"date": "2024-06-01", "text": query_text}],
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
        ans_variants = normalize_answer_variants(row["answer"])

        def check_k(k):
            for s_id in ranked[:k]:
                c_idx = int(s_id.split("_")[1])
                c_low = chunks[c_idx].lower()
                c_stripped = re.sub(r"[,$]", "", c_low)
                if any(v in c_low or v in c_stripped for v in ans_variants):
                    return True
            return False

        if check_k(1):
            hits_1 += 1
        if check_k(5):
            hits_5 += 1
        if check_k(10):
            hits_10 += 1
        if not check_k(1) or not check_k(5):
            s_list = resp.get("results", [{}])[0].get("sessions", [])
            pred0 = s_list[0] if s_list else {}
            pred0_id = pred0.get("session_id") or pred0.get("id", "")
            pred0_idx = int(pred0_id.split("_")[1]) if "_" in pred0_id else 0
            first_hit = -1
            for rk_i, s_id in enumerate(ranked):
                ci_k = int(s_id.split("_")[1])
                cl_k = chunks[ci_k].lower()
                cs_k = re.sub(r"[,$]", "", cl_k)
                if any(v in cl_k or v in cs_k for v in ans_variants):
                    first_hit = rk_i
                    break
            with open(loong_ledger, "a", encoding="utf-8") as lf:
                lf.write(json.dumps({
                    "query": query_text,
                    "category": "loong_l1_financial",
                    "gold_turns": ans_variants[:3],
                    "predicted_rank1": {"id": pred0_id, "preview": chunks[pred0_idx][:180] if chunks else ""},
                    "gold_rank": first_hit + 1 if first_hit >= 0 else "NOT_IN_TOP10",
                    "miss_type": "NEAR_MISS_RANKS_2_5" if 1 <= first_hit < 5 else "NOT_IN_TOP10",
                    "winning_tokens": pred0.get("matched", []),
                }) + "\n")

        if idx % 10 == 0 or idx == 1:
            print(
                f"[{idx}/{len(en_l1)}] R@1: {hits_1/total*100:.1f}% | "
                f"R@5: {hits_5/total*100:.1f}% | R@10: {hits_10/total*100:.1f}% | "
                f"Chunks: {len(chunks)} | Latency: {dt*1000:.1f}ms"
            )

    worker.stdin.close()
    worker.terminate()

    res = {
        "total": total,
        "recall_at_1": round(hits_1 / total * 100, 2) if total else 0,
        "recall_at_5": round(hits_5 / total * 100, 2) if total else 0,
        "recall_at_10": round(hits_10 / total * 100, 2) if total else 0,
        "avg_latency_ms": round(sum(durations) / len(durations) * 1000, 1) if durations else 0,
    }

    print("\n=== LOONG Spotlight Final Result ===")
    print(f"Total Evaluated: {total}")
    print(f"Answer-Span Recall@1:  {res['recall_at_1']:.2f}%")
    print(f"Answer-Span Recall@5:  {res['recall_at_5']:.2f}%")
    print(f"Answer-Span Recall@10: {res['recall_at_10']:.2f}%")
    print(f"Avg Latency: {res['avg_latency_ms']:.1f}ms")

    if json_out:
        os.makedirs(os.path.dirname(json_out), exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=75)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    run_eval(limit=args.limit, json_out=args.json)
