#!/usr/bin/env python3
"""Score one retrieval_bench binary the way the stage-1 and recall harnesses do.

Labels are read only after the worker returns. The worker payload is sessions,
queries, and top_k. `factor_graph_fired` is counted from the response and is
absent on the PR 715 binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(n: int, d: int) -> float:
    if not d:
        return 0.0
    # Match JS Number((100 * n / d).toFixed(2)): half away from zero.
    x = 100.0 * n / d
    sign = -1 if x < 0 else 1
    return sign * math.floor(abs(x) * 100 + 0.5) / 100


def ndcg(ranked, gold: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in gold:
            dcg += 1.0 / math.log2(i + 2)
    idcg = 0.0
    for i in range(min(len(gold), k)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg else 0.0


def fired_of(result: dict) -> bool:
    return bool(result.get("factor_graph_fired", False))


class Worker:
    def __init__(self, bin_path: str):
        self.proc = subprocess.Popen(
            [bin_path, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def rank(self, payload: dict) -> dict:
        if self.proc.poll() is not None:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"worker exited {self.proc.returncode}: {err[-2000:]}")
        self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"worker closed stdout exit={self.proc.poll()} {err[-2000:]}")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def first_rank(ids, gold: set):
    for i, item in enumerate(ids):
        if item in gold:
            return i
    return -1


def blank():
    return {"n": 0, "r1": 0, "r5": 0, "r10": 0, "all5": 0, "all10": 0, "ndcg10": 0.0}


def add_official(acc, m):
    acc["n"] += 1
    acc["r1"] += int(m["r1"])
    acc["r5"] += int(m["r5"])
    acc["r10"] += int(m["r10"])
    acc["all5"] += int(m["all5"])
    acc["all10"] += int(m["all10"])
    acc["ndcg10"] += m["ndcg10"]


def pack_official(acc):
    n = acc["n"]
    return {
        "n": n,
        "any1": acc["r1"],
        "any5": acc["r5"],
        "any10": acc["r10"],
        "all5": acc["all5"],
        "all10": acc["all10"],
        "any10": acc["r10"],
        "all10": acc["all10"],
        "any1_pct": pct(acc["r1"], n),
        "any5_pct": pct(acc["r5"], n),
        "any10_pct": pct(acc["r10"], n),
        "all5_pct": pct(acc["all5"], n),
        "all10_pct": pct(acc["all10"], n),
        "ndcg10": round(acc["ndcg10"] / n, 4) if n else 0.0,
    }


def run_lme(worker: Worker, shard: int, shards: int, limit: int):
    path = os.path.join("benchmarks", "data", "longmemeval_s_cleaned.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    qhash = hashlib.sha256()
    for item in data:
        qhash.update(str(item["question_id"]).encode())
        qhash.update(b"\n")
    official = blank()
    nonabs = blank()
    turns = blank()
    types = {}
    turn_types = {}
    rows = []
    fires = 0
    n_seen = 0
    t0 = time.time()
    for idx, item in enumerate(data):
        if idx % shards != shard:
            continue
        sessions = []
        gold_turns = set()
        for i, sess in enumerate(item["haystack_sessions"]):
            sid = str(item["haystack_session_ids"][i])
            turns_in = []
            for j, turn in enumerate(sess):
                tid = f"{sid}#{j}"
                turns_in.append(
                    {"id": tid, "role": str(turn["role"]), "content": str(turn["content"])}
                )
                if turn.get("has_answer") is True:
                    gold_turns.add(tid)
            sessions.append(
                {
                    "id": sid,
                    "date": str(item["haystack_dates"][i] or ""),
                    "turns": turns_in,
                }
            )
        query = {"text": str(item["question"]), "date": str(item.get("question_date") or "")}
        resp = worker.rank({"sessions": sessions, "queries": [query], "top_k": 10})
        result = resp["results"][0]
        fired = fired_of(result)
        fires += int(fired)
        sids = [s["id"] for s in result["sessions"]]
        tids = [t["id"] for t in result["turns"]]
        gold = set(item.get("answer_session_ids") or [])
        qtype = item["question_type"]
        is_abs = str(item["question_id"]).endswith("_abs")
        m = {
            "r1": any(i in gold for i in sids[:1]),
            "r5": any(i in gold for i in sids[:5]),
            "r10": any(i in gold for i in sids[:10]),
            "all5": bool(gold) and all(g in sids[:5] for g in gold),
            "all10": bool(gold) and all(g in sids[:10] for g in gold),
            "ndcg10": ndcg(sids, gold, 10),
        }
        tm = {
            "r1": any(i in gold_turns for i in tids[:1]),
            "r5": any(i in gold_turns for i in tids[:5]),
            "r10": any(i in gold_turns for i in tids[:10]),
            "all5": bool(gold_turns) and all(g in tids[:5] for g in gold_turns),
            "all10": bool(gold_turns) and all(g in tids[:10] for g in gold_turns),
            "ndcg10": ndcg(tids, gold_turns, 10),
        }
        add_official(official, m)
        bucket = types.setdefault(qtype, blank())
        add_official(bucket, m)
        if not is_abs:
            add_official(nonabs, m)
            add_official(turns, tm)
            add_official(turn_types.setdefault(qtype, blank()), tm)
        rows.append(
            {
                "id": item["question_id"],
                "type": qtype,
                "abs": is_abs,
                "fired": int(fired),
                "s1": int(m["r1"]),
                "s5": int(m["r5"]),
                "all5": int(m["all5"]),
                "t1": int(tm["r1"]),
                "t5": int(tm["r5"]),
                "srank": first_rank(sids, gold),
                "trank": first_rank(tids, gold_turns),
            }
        )
        n_seen += 1
        if n_seen % 25 == 0:
            print(f"lme shard {shard}/{shards} {n_seen} fires={fires} {time.time()-t0:.0f}s", file=sys.stderr)
    return {
        "suite": "lme",
        "shard": shard,
        "shards": shards,
        "question_list_sha256": qhash.hexdigest(),
        "fires": fires,
        "official_500": pack_official(official),
        "official_470": pack_official(nonabs),
        "turns_470": pack_official(turns),
        "types": {k: pack_official(v) for k, v in sorted(types.items())},
        "turn_types": {k: pack_official(v) for k, v in sorted(turn_types.items())},
        "rows": rows,
    }


def run_locomo(worker: Worker):
    path = os.path.join("benchmarks", "data", "locomo10.json")
    with open(path, encoding="utf-8") as f:
        convs = json.load(f)
    names = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop", 5: "Adversarial"}
    cats = {}
    rows = []
    fires = 0
    qhash = hashlib.sha256()
    for ci, conv in enumerate(convs):
        c = conv["conversation"]
        sessions = []
        s = 1
        while c.get(f"session_{s}"):
            turns = []
            for t in c[f"session_{s}"]:
                if not t.get("text") or not t.get("dia_id"):
                    continue
                blip = f"\n[BLIP caption] {t['blip_caption']}" if t.get("blip_caption") else ""
                turns.append(
                    {
                        "id": str(t["dia_id"]),
                        "role": str(t.get("speaker") or ""),
                        "content": f"{t['text']}{blip}",
                    }
                )
            sessions.append(
                {"id": f"D{s}", "date": str(c.get(f"session_{s}_date_time") or ""), "turns": turns}
            )
            s += 1
        queries = []
        for qa in conv["qa"]:
            queries.append({"text": str(qa["question"]), "date": ""})
            qhash.update(f"{ci}:{qa['question']}\n".encode())
        resp = worker.rank({"sessions": sessions, "queries": queries, "top_k": 10})
        for qi, result in enumerate(resp["results"]):
            qa = conv["qa"][qi]
            fired = fired_of(result)
            fires += int(fired)
            cat = qa["category"]
            evidence = qa.get("evidence") or []
            gold_turns = set()
            for e in evidence:
                for part in re.split(r"[;,\s]+", str(e)):
                    if re.fullmatch(r"D\d+:\d+", part):
                        gold_turns.add(part)
            if not gold_turns:
                continue
            gold_sess = {g.split(":")[0] for g in gold_turns}
            tids = [t["id"] for t in result["turns"]]
            sids = [s["id"] for s in result["sessions"]]
            rec = {
                "n": 1,
                "s1": int(any(i in gold_sess for i in sids[:1])),
                "s5": int(any(i in gold_sess for i in sids[:5])),
                "t1": int(any(i in gold_turns for i in tids[:1])),
                "t5": int(any(i in gold_turns for i in tids[:5])),
                "t10": int(any(i in gold_turns for i in tids[:10])),
            }
            acc = cats.setdefault(cat, {"n": 0, "s1": 0, "s5": 0, "t1": 0, "t5": 0, "t10": 0})
            for k, v in rec.items():
                acc[k] += v
            rows.append(
                {
                    "id": f"{ci}:{qi}",
                    "cat": cat,
                    "fired": int(fired),
                    "s1": rec["s1"],
                    "s5": rec["s5"],
                    "t1": rec["t1"],
                    "t5": rec["t5"],
                    "trank": first_rank(tids, gold_turns),
                }
            )
        print(f"locomo conv {ci+1}/{len(convs)} fires={fires}", file=sys.stderr)
    def pack(acc):
        n = acc["n"]
        return {
            "n": n,
            "s1": acc["s1"],
            "s5": acc["s5"],
            "t1": acc["t1"],
            "t5": acc["t5"],
            "t10": acc["t10"],
            "s1_pct": pct(acc["s1"], n),
            "s5_pct": pct(acc["s5"], n),
            "t1_pct": pct(acc["t1"], n),
            "t5_pct": pct(acc["t5"], n),
            "t10_pct": pct(acc["t10"], n),
        }
    core = {"n": 0, "s1": 0, "s5": 0, "t1": 0, "t5": 0, "t10": 0}
    for cat, acc in cats.items():
        if cat == 5:
            continue
        for k in core:
            core[k] += acc[k]
    return {
        "suite": "locomo",
        "question_list_sha256": qhash.hexdigest(),
        "fires": fires,
        "core": pack(core),
        "categories": {names.get(c, str(c)): pack(a) for c, a in sorted(cats.items())},
        "rows": rows,
    }


def flatten_ids(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            out.extend(flatten_ids(item))
        return out
    if isinstance(raw, dict):
        out = []
        for item in raw.values():
            out.extend(flatten_ids(item))
        return out
    return [str(raw)]


def run_beam(worker: Worker):
    base = os.path.join("benchmarks", "BEAM", "chats", "100K")
    convs = sorted(
        [d for d in os.listdir(base) if os.path.isfile(os.path.join(base, d, "chat.json"))],
        key=lambda x: int(x),
    )
    cats = {}
    rows = []
    fires = 0
    scored = 0
    any1 = any5 = mrr = 0
    qhash = hashlib.sha256()
    for c_id in convs:
        chat = json.load(open(os.path.join(base, c_id, "chat.json"), encoding="utf-8"))
        qdata = json.load(
            open(os.path.join(base, c_id, "probing_questions", "probing_questions.json"), encoding="utf-8")
        )
        sessions = []
        turn_to_sess = {}
        for bi, batch in enumerate(chat):
            anchor = batch.get("time_anchor") or ""
            if not anchor and batch.get("turns") and batch["turns"][0]:
                anchor = batch["turns"][0][0].get("time_anchor") or ""
            for sub_idx, sub in enumerate(batch.get("turns") or []):
                if not isinstance(sub, list) or not sub:
                    continue
                sid = f"conv{c_id}_b{bi}_s{sub_idx}"
                turns = []
                for t in sub:
                    tid = str(t["id"])
                    turns.append({"id": tid, "role": t.get("role") or "user", "content": t.get("content") or ""})
                    turn_to_sess[tid] = sid
                sessions.append(
                    {
                        "id": sid,
                        "date": (sub[0].get("time_anchor") if sub else None) or anchor or "",
                        "timestamp": 0,
                        "turns": turns,
                    }
                )
        entries = []
        for cat, qlist in qdata.items():
            if not isinstance(qlist, list):
                continue
            for q in qlist:
                gold = list(dict.fromkeys(flatten_ids(q.get("source_chat_ids"))))
                if not gold:
                    continue
                entries.append((cat, q["question"], gold))
                qhash.update(f"{c_id}:{cat}:{q['question']}\n".encode())
        if not entries:
            continue
        resp = worker.rank(
            {
                "sessions": sessions,
                "queries": [{"text": q, "date": ""} for _, q, _ in entries],
                "top_k": 10,
            }
        )
        for qi, (cat, question, gold) in enumerate(entries):
            result = resp["results"][qi]
            fired = fired_of(result)
            fires += int(fired)
            tids = [t["id"] for t in result["turns"]]
            gold_set = set(gold)
            hit1 = any(i in gold_set for i in tids[:1])
            hit5 = any(i in gold_set for i in tids[:5])
            rank = first_rank(tids, gold_set)
            scored += 1
            any1 += int(hit1)
            any5 += int(hit5)
            if rank >= 0:
                mrr += 1.0 / (rank + 1)
            st = cats.setdefault(cat, {"n": 0, "t1": 0, "t5": 0, "mrr": 0.0})
            st["n"] += 1
            st["t1"] += int(hit1)
            st["t5"] += int(hit5)
            if rank >= 0:
                st["mrr"] += 1.0 / (rank + 1)
            rows.append(
                {
                    "id": f"{c_id}:{cat}:{qi}",
                    "cat": cat,
                    "fired": int(fired),
                    "t1": int(hit1),
                    "t5": int(hit5),
                    "trank": rank,
                }
            )
        print(f"beam conv {c_id} scored={scored} fires={fires}", file=sys.stderr)
    return {
        "suite": "beam",
        "question_list_sha256": qhash.hexdigest(),
        "fires": fires,
        "n": scored,
        "any1": any1,
        "any5": any5,
        "any1_pct": pct(any1, scored),
        "any5_pct": pct(any5, scored),
        "mrr": round(mrr / scored, 4) if scored else 0.0,
        "categories": {
            cat: {
                "n": st["n"],
                "t1": st["t1"],
                "t5": st["t5"],
                "t1_pct": pct(st["t1"], st["n"]),
                "t5_pct": pct(st["t5"], st["n"]),
                "mrr": round(st["mrr"] / st["n"], 4) if st["n"] else 0.0,
            }
            for cat, st in sorted(cats.items())
        },
        "rows": rows,
    }


def merge_lme(parts):
    official = blank()
    nonabs = blank()
    turns = blank()
    # Recompute from rows so shards cannot double-count a packed total.
    rows = []
    fires = 0
    qhash = parts[0]["question_list_sha256"]
    for part in parts:
        if part["question_list_sha256"] != qhash:
            raise SystemExit("question list hash mismatch across shards")
        fires += part["fires"]
        rows.extend(part["rows"])
    types = {}
    turn_types = {}
    for row in rows:
        m = {
            "r1": row["s1"],
            "r5": row["s5"],
            "r10": 0,
            "all5": row["all5"],
            "all10": 0,
            "ndcg10": 0.0,
        }
        # Shard packs already hold ndcg. Re-sum packed ndcg*n from the part instead.
    # The row record does not keep ndcg. Sum the packed shard stats.
    def sum_pack(key):
        acc = blank()
        for part in parts:
            packed = part[key]
            acc["n"] += packed["n"]
            acc["r1"] += packed["any1"]
            acc["r5"] += packed["any5"]
            acc["r10"] += packed.get("any10", 0)
            acc["all5"] += packed["all5"]
            acc["all10"] += packed.get("all10", 0)
            acc["ndcg10"] += packed["ndcg10"] * packed["n"]
        return acc
    def sum_types(key):
        out = {}
        for part in parts:
            for name, packed in part[key].items():
                acc = out.setdefault(name, blank())
                acc["n"] += packed["n"]
                acc["r1"] += packed["any1"]
                acc["r5"] += packed["any5"]
                acc["r10"] += packed.get("any10", 0)
                acc["all5"] += packed["all5"]
                acc["all10"] += packed.get("all10", 0)
                acc["ndcg10"] += packed["ndcg10"] * packed["n"]
        return {k: pack_official(v) for k, v in sorted(out.items())}
    rows.sort(key=lambda r: r["id"])
    return {
        "suite": "lme",
        "question_list_sha256": qhash,
        "fires": fires,
        "official_500": pack_official(sum_pack("official_500")),
        "official_470": pack_official(sum_pack("official_470")),
        "turns_470": pack_official(sum_pack("turns_470")),
        "types": sum_types("types"),
        "turn_types": sum_types("turn_types"),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--suite", choices=["lme", "locomo", "beam", "merge-lme"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parts", nargs="*")
    args = ap.parse_args()
    if args.suite == "merge-lme":
        parts = [json.load(open(p, encoding="utf-8")) for p in args.parts]
        out = merge_lme(parts)
    else:
        worker = Worker(args.bin)
        try:
            if args.suite == "lme":
                out = run_lme(worker, args.shard, args.shards, args.limit)
            elif args.suite == "locomo":
                out = run_locomo(worker)
            else:
                out = run_beam(worker)
        finally:
            worker.close()
        out["bin_sha256"] = sha256_file(args.bin)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    summary = {k: v for k, v in out.items() if k != "rows"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
