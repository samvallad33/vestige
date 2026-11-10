#!/usr/bin/env python3
"""
Standard BM25 baseline implementation using the standard rank_bm25 Python library.
Evaluates:
  1. LongMemEval_S (official 470 non-abstention + all 500, with 6 category breakdowns)
  2. LoCoMo-10 (1,536 CORE Cat 1-4 + Cat 1-5 breakdown, session and turn recall)

Usage:
  python benchmarks/baseline_bm25.py [--json=benchmarks/results/v6.2.1/baseline_python_bm25.json]
"""

import sys
import json
import math
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text) if w]

def ndcg_at_k(ranked_ids: list[str], gold_set: set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, item_id in enumerate(ranked_ids[:k]):
        if item_id in gold_set:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
    return (dcg / idcg) if idcg > 0.0 else 0.0

def run_longmemeval(data_path: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_metrics = {"n": 0, "r1": 0, "r5": 0, "r10": 0, "all5": 0, "all10": 0, "ndcg10": 0.0}
    non_abs_metrics = {"n": 0, "r1": 0, "r5": 0, "r10": 0, "all5": 0, "all10": 0, "ndcg10": 0.0}
    type_metrics = {}

    for item in data:
        q_text = str(item.get("question", ""))
        q_tokens = tokenize(q_text)
        is_abs = str(item.get("question_id", "")).endswith("_abs")
        q_type = str(item.get("question_type", "unknown"))

        sess_ids = [str(sid) for sid in item.get("haystack_session_ids", [])]
        haystack = item.get("haystack_sessions", [])

        # Concatenate turn texts for each session
        corpus = []
        for sess in haystack:
            sess_text = " ".join(str(t.get("content", "")) for t in sess)
            corpus.append(tokenize(sess_text))

        if corpus and q_tokens:
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(q_tokens)
            ranked_pairs = sorted(zip(sess_ids, scores), key=lambda x: x[1], reverse=True)
            ranked_ids = [sid for sid, _ in ranked_pairs]
        else:
            ranked_ids = list(sess_ids)

        gold = set(str(g) for g in item.get("answer_session_ids", []))
        top1 = set(ranked_ids[:1])
        top5 = set(ranked_ids[:5])
        top10 = set(ranked_ids[:10])

        r1 = 1 if any(g in top1 for g in gold) else 0
        r5 = 1 if any(g in top5 for g in gold) else 0
        r10 = 1 if any(g in top10 for g in gold) else 0
        all5 = 1 if (len(gold) > 0 and all(g in top5 for g in gold)) else 0
        all10 = 1 if (len(gold) > 0 and all(g in top10 for g in gold)) else 0
        score_ndcg = ndcg_at_k(ranked_ids, gold, 10)

        all_metrics["n"] += 1
        all_metrics["r1"] += r1
        all_metrics["r5"] += r5
        all_metrics["r10"] += r10
        all_metrics["all5"] += all5
        all_metrics["all10"] += all10
        all_metrics["ndcg10"] += score_ndcg

        if not is_abs:
            non_abs_metrics["n"] += 1
            non_abs_metrics["r1"] += r1
            non_abs_metrics["r5"] += r5
            non_abs_metrics["r10"] += r10
            non_abs_metrics["all5"] += all5
            non_abs_metrics["all10"] += all10
            non_abs_metrics["ndcg10"] += score_ndcg

            if q_type not in type_metrics:
                type_metrics[q_type] = {"n": 0, "r1": 0, "r5": 0, "r10": 0, "all5": 0, "all10": 0, "ndcg10": 0.0}
            t_acc = type_metrics[q_type]
            t_acc["n"] += 1
            t_acc["r1"] += r1
            t_acc["r5"] += r5
            t_acc["r10"] += r10
            t_acc["all5"] += all5
            t_acc["all10"] += all10
            t_acc["ndcg10"] += score_ndcg

    def fmt_pct(k, n):
        return f"{(100.0 * k / n):.2f}%" if n else "0.00%"

    print("--- Python rank_bm25 Baseline: LongMemEval_S ---")
    n_na = non_abs_metrics["n"]
    print(f"NON-ABS ({n_na} questions) [Official Benchmark Metric]:")
    print(f"  RecallAll@5:  {fmt_pct(non_abs_metrics['all5'], n_na)}")
    print(f"  RecallAll@10: {fmt_pct(non_abs_metrics['all10'], n_na)}")
    print(f"  NDCG@10:      {(non_abs_metrics['ndcg10'] / n_na):.4f}")
    print(f"  RecallAny@1:  {fmt_pct(non_abs_metrics['r1'], n_na)}")
    print(f"  RecallAny@5:  {fmt_pct(non_abs_metrics['r5'], n_na)}")
    print(f"  RecallAny@10: {fmt_pct(non_abs_metrics['r10'], n_na)}")

    n_all = all_metrics["n"]
    print(f"\nALL ({n_all} questions, includes 30 abstentions):")
    print(f"  RecallAny@1:  {fmt_pct(all_metrics['r1'], n_all)}")
    print(f"  RecallAny@5:  {fmt_pct(all_metrics['r5'], n_all)}")
    print(f"  RecallAny@10: {fmt_pct(all_metrics['r10'], n_all)}")
    print(f"  RecallAll@5:  {fmt_pct(all_metrics['all5'], n_all)}")
    print(f"  NDCG@10:      {(all_metrics['ndcg10'] / n_all):.4f}")

    print("\nQuestion Type Breakdown (Non-Abstention):")
    for qt, t_acc in sorted(type_metrics.items()):
        qn = t_acc["n"]
        print(f"  {qt:<26} ({qn:3d} q) R@1: {fmt_pct(t_acc['r1'], qn):>7} | R@5: {fmt_pct(t_acc['r5'], qn):>7} | R@10: {fmt_pct(t_acc['r10'], qn):>7} | RAll@5: {fmt_pct(t_acc['all5'], qn):>7}")

    return {"non_abs": non_abs_metrics, "all": all_metrics, "types": type_metrics}

def run_locomo(data_path: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_names = {1: "Multi-Hop", 2: "Temporal", 3: "Open-Domain", 4: "Single-Hop", 5: "Adversarial"}
    def make_acc():
        return {"n": 0, "no_evidence": 0, "sR1": 0, "sR5": 0, "tR1": 0, "tR5": 0, "tR10": 0, "softR1": 0, "softR5": 0}

    core_acc = make_acc()
    total_acc = make_acc()
    cat_accs = {c: make_acc() for c in range(1, 6)}

    for conv in data:
        c = conv.get("conversation", {})
        sessions = []
        s_idx = 1
        while f"session_{s_idx}" in c:
            turns = [t for t in c[f"session_{s_idx}"] if t.get("text") and t.get("dia_id")]
            sessions.append({
                "id": f"D{s_idx}",
                "turns": turns
            })
            s_idx += 1

        sess_ids = [s["id"] for s in sessions]
        sess_corpus = [tokenize(" ".join(t.get("text", "") for t in s["turns"])) for s in sessions]
        bm25_sess = BM25Okapi(sess_corpus) if sess_corpus else None

        all_turns = []
        for s in sessions:
            for t in s["turns"]:
                all_turns.append((t["dia_id"], t.get("text", "")))
        turn_ids = [t[0] for t in all_turns]
        turn_corpus = [tokenize(t[1]) for t in all_turns]
        bm25_turn = BM25Okapi(turn_corpus) if turn_corpus else None

        for qa in conv.get("qa", []):
            cat = qa.get("category", 0)
            raw_ev = qa.get("evidence", [])
            gold_turns = set()
            for e in raw_ev:
                for token in re.split(r"[;,\s]+", str(e)):
                    if re.match(r"^D\d+:\d+$", token):
                        gold_turns.add(token)

            acc_list = [total_acc]
            if cat in cat_accs:
                acc_list.append(cat_accs[cat])
            if cat in (1, 2, 3, 4):
                acc_list.append(core_acc)

            if not gold_turns:
                for a in acc_list:
                    a["no_evidence"] += 1
                continue

            gold_sess = {g.split(":")[0] for g in gold_turns}
            soft_gold = set(gold_turns)
            for g in gold_turns:
                parts = g.split(":")
                sid = parts[0]
                try:
                    tnum = int(parts[1])
                    soft_gold.add(f"{sid}:{tnum - 1}")
                    soft_gold.add(f"{sid}:{tnum + 1}")
                except ValueError:
                    pass

            q_tokens = tokenize(str(qa.get("question", "")))
            if bm25_sess and q_tokens:
                s_scores = bm25_sess.get_scores(q_tokens)
                ranked_s = [sid for sid, _ in sorted(zip(sess_ids, s_scores), key=lambda x: x[1], reverse=True)]
            else:
                ranked_s = list(sess_ids)

            if bm25_turn and q_tokens:
                t_scores = bm25_turn.get_scores(q_tokens)
                ranked_t = [tid for tid, _ in sorted(zip(turn_ids, t_scores), key=lambda x: x[1], reverse=True)]
            else:
                ranked_t = list(turn_ids)

            s_r1 = 1 if any(s in ranked_s[:1] for s in gold_sess) else 0
            s_r5 = 1 if any(s in ranked_s[:5] for s in gold_sess) else 0
            t_r1 = 1 if any(t in ranked_t[:1] for t in gold_turns) else 0
            t_r5 = 1 if any(t in ranked_t[:5] for t in gold_turns) else 0
            t_r10 = 1 if any(t in ranked_t[:10] for t in gold_turns) else 0
            soft_r1 = 1 if any(t in ranked_t[:1] for t in soft_gold) else 0
            soft_r5 = 1 if any(t in ranked_t[:5] for t in soft_gold) else 0

            for a in acc_list:
                a["n"] += 1
                a["sR1"] += s_r1
                a["sR5"] += s_r5
                a["tR1"] += t_r1
                a["tR5"] += t_r5
                a["tR10"] += t_r10
                a["softR1"] += soft_r1
                a["softR5"] += soft_r5

    def fmt_pct(k, n):
        return f"{(100.0 * k / n):.2f}%" if n else "0.00%"

    print("\n--- Python rank_bm25 Baseline: LoCoMo-10 ---")
    cn = core_acc["n"]
    print(f"CORE (Cat 1-4) ({cn} scored questions across 10 dialogues):")
    print(f"  Session R@1: {fmt_pct(core_acc['sR1'], cn)} | Session R@5: {fmt_pct(core_acc['sR5'], cn)}")
    print(f"  Turn Exact R@1: {fmt_pct(core_acc['tR1'], cn)} | R@5: {fmt_pct(core_acc['tR5'], cn)} | R@10: {fmt_pct(core_acc['tR10'], cn)}")
    print(f"  Turn Soft  R@1: {fmt_pct(core_acc['softR1'], cn)} | R@5: {fmt_pct(core_acc['softR5'], cn)}")

    tn = total_acc["n"]
    print(f"\nALL (Cat 1-5) ({tn} scored questions across 10 dialogues):")
    print(f"  Session R@1: {fmt_pct(total_acc['sR1'], tn)} | Session R@5: {fmt_pct(total_acc['sR5'], tn)}")
    print(f"  Turn Exact R@1: {fmt_pct(total_acc['tR1'], tn)} | R@5: {fmt_pct(total_acc['tR5'], tn)} | R@10: {fmt_pct(total_acc['tR10'], tn)}")
    print(f"  Turn Soft  R@1: {fmt_pct(total_acc['softR1'], tn)} | R@5: {fmt_pct(total_acc['softR5'], tn)}")

    print("\nPer-Category Breakdown:")
    for c in range(1, 6):
        a = cat_accs[c]
        an = a["n"]
        print(f"  Cat {c} {cat_names[c]:<14} ({an:3d} q) Sess R@5: {fmt_pct(a['sR5'], an):>7} | Turn Exact R@5: {fmt_pct(a['tR5'], an):>7} | Soft R@5: {fmt_pct(a['softR5'], an):>7}")

    return {"core": core_acc, "total": total_acc, "cats": cat_accs}

def main():
    root = Path(__file__).resolve().parent.parent
    lme_file = root / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
    locomo_file = root / "benchmarks" / "data" / "locomo10.json"

    json_out = None
    for arg in sys.argv[1:]:
        if arg.startswith("--json="):
            json_out = Path(arg.split("=", 1)[1])

    results = {
        "engine": "rank_bm25 (Python BM25Okapi)",
        "lme": run_longmemeval(lme_file),
        "locomo": run_locomo(locomo_file)
    }

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline results written to {json_out}")

if __name__ == "__main__":
    main()
