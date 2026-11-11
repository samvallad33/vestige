#!/usr/bin/env python3
"""
LongMemEval_M streaming zero-LLM, zero-vector recall evaluator
for the compiled Vestige native retrieval worker (`retrieval_bench --worker`).

Streams `benchmarks/data/longmemeval_m_cleaned.jsonl` one line at a time (< 100 MB RAM),
evaluating ~476 sessions / ~5,000 turns per question with zero label leakage
(`#[serde(deny_unknown_fields)]`). If only `longmemeval_m_cleaned.json` (~2.6 GB) exists,
converts it once incrementally or requires running via streaming conversion first.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time


def find_binary():
    candidates = [
        os.path.join('target', 'release', 'retrieval_bench.exe'),
        os.path.join('target', 'release', 'retrieval_bench'),
        os.path.join('target', 'debug', 'retrieval_bench.exe'),
        os.path.join('target', 'debug', 'retrieval_bench'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        'retrieval_bench binary not found. Build with: cargo build --release -p vestige-core --bin retrieval_bench'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=500, help='Maximum questions to evaluate (default 500)')
    ap.add_argument(
        '--input',
        default=os.path.join('benchmarks', 'data', 'longmemeval_m_cleaned.jsonl'),
        help='Line-delimited JSONL path for LongMemEval_M',
    )
    ap.add_argument(
        '--json',
        default=os.path.join('benchmarks', 'results', 'v6.2.1', 'longmemeval_m_recall.json'),
        help='Output JSON path',
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f'{args.input} not found. Compact longmemeval_m_cleaned.json into 1-item-per-line JSONL before running.'
        )

    bin_path = find_binary()
    proc = subprocess.Popen(
        [bin_path, '--worker'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
    )

    def rank(payload):
        proc.stdin.write(json.dumps(payload) + '\n')
        proc.stdin.flush()
        line = proc.stdout.readline()
        resp = json.loads(line)
        if 'error' in resp:
            raise RuntimeError(resp['error'])
        return resp

    n_all = n_non = 0
    non_r1 = non_r3 = non_r5 = non_r10 = non_all5 = non_all10 = 0
    all_r1 = all_r3 = all_r5 = all_r10 = all_all5 = all_all10 = 0
    non_ndcg10 = all_ndcg10 = non_mrr = all_mrr = 0.0
    sess_sum = turns_sum = 0
    types = {}
    t0 = time.time()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                if n_all >= args.limit:
                    break
                item = json.loads(line)
                sessions = [
                    {
                        'id': str(item['haystack_session_ids'][i]),
                        'date': str(item['haystack_dates'][i] or ''),
                        'turns': [
                            {'id': f"{item['haystack_session_ids'][i]}#{j}", 'role': str(t['role']), 'content': str(t['content'])}
                            for j, t in enumerate(sess)
                        ],
                    }
                    for i, sess in enumerate(item['haystack_sessions'])
                ]
                query = {'text': str(item['question']), 'date': str(item.get('question_date') or '')}
                resp = rank({'sessions': sessions, 'queries': [query], 'top_k': 10})
                sess_sum += resp['sessions_indexed']
                turns_sum += resp['turns_indexed']

                ids = [s['id'] for s in resp['results'][0]['sessions']]
                gold = set(item.get('answer_session_ids') or [])
                is_abs = str(item['question_id']).endswith('_abs')

                dcg = sum(1.0 / math.log2(i + 2) for i, sid in enumerate(ids[:10]) if sid in gold)
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
                ndcg10 = (dcg / idcg) if idcg else 0.0

                r1 = any(sid in gold for sid in ids[:1])
                r3 = any(sid in gold for sid in ids[:3])
                r5 = any(sid in gold for sid in ids[:5])
                r10 = any(sid in gold for sid in ids[:10])
                all5 = bool(gold) and all(g in ids[:5] for g in gold)
                all10 = bool(gold) and all(g in ids[:10] for g in gold)
                mrr = 0.0
                for ri, sid in enumerate(ids[:10]):
                    if sid in gold:
                        mrr = 1.0 / (ri + 1)
                        break

                n_all += 1
                if r1: all_r1 += 1
                if r3: all_r3 += 1
                if r5: all_r5 += 1
                if r10: all_r10 += 1
                if all5: all_all5 += 1
                if all10: all_all10 += 1
                all_ndcg10 += ndcg10
                all_mrr += mrr
                if not is_abs:
                    n_non += 1
                    if r1: non_r1 += 1
                    if r3: non_r3 += 1
                    if r5: non_r5 += 1
                    if r10: non_r10 += 1
                    if all5: non_all5 += 1
                    if all10: non_all10 += 1
                    non_ndcg10 += ndcg10
                    non_mrr += mrr
                    qt = item['question_type']
                    b = types.setdefault(qt, {'n': 0, 'r1': 0, 'r5': 0, 'r10': 0, 'all5': 0, 'all10': 0, 'ndcg10': 0.0})
                    b['n'] += 1
                    b['r1'] += int(r1)
                    b['r5'] += int(r5)
                    b['r10'] += int(r10)
                    b['all5'] += int(all5)
                    b['all10'] += int(all10)
                    b['ndcg10'] += ndcg10

                if n_all % 10 == 0:
                    print(
                        f"[{n_all}/{args.limit}] {time.time()-t0:.1f}s | avgSess={sess_sum/n_all:.0f} | NonAbs Any@5={non_r5/max(1,n_non)*100:.2f}% All@5={non_all5/max(1,n_non)*100:.2f}%",
                        flush=True,
                    )
    finally:
        proc.terminate()

    dt = time.time() - t0
    print(f"\n=== LongMemEval_M ({n_all} total, {n_non} non-abs) in {dt:.2f}s ===")
    print(f"Avg haystack: {sess_sum/n_all:.1f} sessions, {turns_sum/n_all:.0f} turns/Q")
    if n_non > 0:
        print(f"[Non-Abstention ({n_non})]: Any@1={non_r1/n_non*100:.2f}% Any@3={non_r3/n_non*100:.2f}% Any@5={non_r5/n_non*100:.2f}% Any@10={non_r10/n_non*100:.2f}% | All@5={non_all5/n_non*100:.2f}% All@10={non_all10/n_non*100:.2f}% | MRR={non_mrr/n_non:.4f} NDCG@10={non_ndcg10/n_non:.4f}")
    print(f"[All Questions ({n_all})]: Any@1={all_r1/n_all*100:.2f}% Any@5={all_r5/n_all*100:.2f}% Any@10={all_r10/n_all*100:.2f}% | All@5={all_all5/n_all*100:.2f}% All@10={all_all10/n_all*100:.2f}% | NDCG@10={all_ndcg10/n_all:.4f}")
    for qt, b in sorted(types.items()):
        print(f"  {qt:26s} R@1={b['r1']/b['n']*100:6.2f}% R@5={b['r5']/b['n']*100:6.2f}% R@10={b['r10']/b['n']*100:6.2f}% All@5={b['all5']/b['n']*100:6.2f}% ({b['n']})")

    out = {
        'benchmark': 'LongMemEval_M',
        'version': 'v6.2.1',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'elapsed_seconds': round(dt, 2),
        'cost_usd': 0,
        'avg_sessions_per_query': round(sess_sum / n_all, 1),
        'avg_turns_per_query': round(turns_sum / n_all, 1),
        'official_non_abs': {
            'n': n_non,
            'recall_any_r1': round(non_r1 / n_non * 100, 2) if n_non else 0.0,
            'recall_any_r3': round(non_r3 / n_non * 100, 2) if n_non else 0.0,
            'recall_any_r5': round(non_r5 / n_non * 100, 2) if n_non else 0.0,
            'recall_any_r10': round(non_r10 / n_non * 100, 2) if n_non else 0.0,
            'recall_all_r5': round(non_all5 / n_non * 100, 2) if n_non else 0.0,
            'recall_all_r10': round(non_all10 / n_non * 100, 2) if n_non else 0.0,
            'mrr': round(non_mrr / n_non, 4) if n_non else 0.0,
            'ndcg_10': round(non_ndcg10 / n_non, 4) if n_non else 0.0,
        },
        'all_questions': {
            'n': n_all,
            'recall_any_r1': round(all_r1 / n_all * 100, 2),
            'recall_any_r3': round(all_r3 / n_all * 100, 2),
            'recall_any_r5': round(all_r5 / n_all * 100, 2),
            'recall_any_r10': round(all_r10 / n_all * 100, 2),
            'recall_all_r5': round(all_all5 / n_all * 100, 2),
            'recall_all_r10': round(all_all10 / n_all * 100, 2),
            'mrr': round(all_mrr / n_all, 4),
            'ndcg_10': round(all_ndcg10 / n_all, 4),
        },
        'types_non_abs': {
            qt: {
                'n': b['n'],
                'recall_any_r1': round(b['r1'] / b['n'] * 100, 2),
                'recall_any_r5': round(b['r5'] / b['n'] * 100, 2),
                'recall_any_r10': round(b['r10'] / b['n'] * 100, 2),
                'recall_all_r5': round(b['all5'] / b['n'] * 100, 2),
                'recall_all_r10': round(b['all10'] / b['n'] * 100, 2),
                'ndcg_10': round(b['ndcg10'] / b['n'], 4),
            } for qt, b in sorted(types.items())
        },
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, 'w', encoding='utf-8') as jf:
        json.dump(out, jf, indent=2)
    print(f'\nCommitted result JSON to: {args.json}')


if __name__ == '__main__':
    main()
