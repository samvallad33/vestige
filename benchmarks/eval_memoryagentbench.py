#!/usr/bin/env python3
"""
MemoryAgentBench (Accurate Retrieval track) zero-LLM, zero-vector recall evaluator
for the compiled Vestige native retrieval worker (`retrieval_bench --worker`).

Evaluates all retrieval-grounded tracks in ai-hyz/MemoryAgentBench Accurate_Retrieval:
  - Row 0 (`ruler_qa1_197K`): 100 single-hop SQuAD QA across 1,204 documents
    * Reports both Unique-Doc Gold subset (unambiguous single-doc answers) and
      Answer-Containing Doc Match across all 100 questions.
  - Row 1 (`ruler_qa2_421K`): 100 multi-hop HotpotQA across 3,307 documents
    * Uses exact `supporting_facts` document titles from hotpot_dev_distractor_v1.
  - Rows 17-21 (`longmemeval_s*`): 5 merged 400K-token scenarios (300 questions total,
    287 non-abstention scored, ~111 chat sessions per scenario)
    * Parses Python-literal `context` cleanly via `ast.literal_eval` and resolves
      gold `has_answer: true` session IDs via first-turn signature post-ranking.

Note: EventQA rows (rows 2-16) are multiple-choice paraphrased next-event selection
over novel chapters with no gold chunk IDs in the dataset and are not exact-ID tracks.
Zero label fields are ever passed into the Rust kernel (`#[serde(deny_unknown_fields)]`).
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.request


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


def ensure_file(path, url):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'Fetching {os.path.basename(path)}...')
    urllib.request.urlretrieve(url, path)


def ans_in_doc(answers, doc_lower):
    for a in answers:
        al = str(a).strip().lower()
        if al and al in doc_lower:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--json',
        default=os.path.join('benchmarks', 'results', 'v6.2.1', 'memoryagentbench_recall.json'),
        help='Output JSON path',
    )
    args = ap.parse_args()

    data_dir = os.path.join('benchmarks', 'data')
    os.makedirs(data_dir, exist_ok=True)

    hotpot_path = os.path.join(data_dir, 'hotpot_dev_distractor_v1.json')
    ensure_file(
        hotpot_path,
        'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json',
    )
    for row_idx in [0, 1, 17, 18, 19, 20, 21]:
        ensure_file(
            os.path.join(data_dir, f'memoryagentbench_row_{row_idx}.json'),
            f'https://datasets-server.huggingface.co/rows?dataset=ai-hyz%2FMemoryAgentBench&config=default&split=Accurate_Retrieval&offset={row_idx}&limit=1',
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
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ''
            raise RuntimeError(f'retrieval_bench exited {proc.returncode} before write: {err[-2000:]}')
        proc.stdin.write(json.dumps(payload) + '\n')
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            err = proc.stderr.read() if proc.stderr else ''
            raise RuntimeError(f'retrieval_bench closed stdout exit={proc.poll()} stderr={err[-2000:]}')
        resp = json.loads(line)
        if 'error' in resp:
            raise RuntimeError(resp['error'])
        return resp

    t_start = time.time()

    try:
        # 1. Row 0: ruler_qa1_197K (1,204 docs, 100 Qs)
        with open(os.path.join(data_dir, 'memoryagentbench_row_0.json'), encoding='utf-8') as f:
            r0 = json.load(f)['rows'][0]['row']
        docs0 = [p.strip() for p in re.split(r'Document \d+:\n', r0['context']) if p.strip()]
        docs0_lower = [d.lower() for d in docs0]
        sessions0 = [
            {'id': f'doc_{i+1}', 'date': '', 'timestamp': 0, 'turns': [{'id': f'doc_{i+1}', 'role': 'user', 'content': d}]}
            for i, d in enumerate(docs0)
        ]
        queries0 = [{'text': q, 'date': ''} for q in r0['questions']]
        t0 = time.time()
        resp0 = rank({'sessions': sessions0, 'queries': queries0, 'top_k': 10})
        dt0 = time.time() - t0

        u_n = u_r1 = u_r5 = u_r10 = 0
        u_mrr = 0.0
        a_n = a_r1 = a_r5 = a_r10 = 0
        a_mrr = 0.0
        for qi, ans_list in enumerate(r0['answers']):
            if not isinstance(ans_list, list):
                ans_list = [ans_list]
            match_docs = {f'doc_{di+1}' for di, dl in enumerate(docs0_lower) if ans_in_doc(ans_list, dl)}
            ret = [s['id'] for s in resp0['results'][qi]['sessions']]
            if match_docs:
                a_n += 1
                if ret[:1] and ret[0] in match_docs:
                    a_r1 += 1
                if any(r in match_docs for r in ret[:5]):
                    a_r5 += 1
                if any(r in match_docs for r in ret[:10]):
                    a_r10 += 1
                for ri, r in enumerate(ret[:10]):
                    if r in match_docs:
                        a_mrr += 1.0 / (ri + 1)
                        break
            if len(match_docs) == 1:
                u_n += 1
                if ret[:1] and ret[0] in match_docs:
                    u_r1 += 1
                if any(r in match_docs for r in ret[:5]):
                    u_r5 += 1
                if any(r in match_docs for r in ret[:10]):
                    u_r10 += 1
                for ri, r in enumerate(ret[:10]):
                    if r in match_docs:
                        u_mrr += 1.0 / (ri + 1)
                        break

        print(f'Row 0 (ruler_qa1_197K, {len(docs0)} docs, 100 Qs) in {dt0:.2f}s:')
        print(f'  Unique-Doc Gold ({u_n} Qs): R@1={u_r1/u_n*100:.2f}% R@5={u_r5/u_n*100:.2f}% R@10={u_r10/u_n*100:.2f}% MRR={u_mrr/u_n:.4f}')
        print(f'  Answer-Containing ({a_n} Qs): R@1={a_r1/a_n*100:.2f}% R@5={a_r5/a_n*100:.2f}% R@10={a_r10/a_n*100:.2f}% MRR={a_mrr/a_n:.4f}')

        # 2. Row 1: ruler_qa2_421K (3,307 docs, 100 multi-hop Qs, exact HotpotQA supporting_facts)
        with open(os.path.join(data_dir, 'memoryagentbench_row_1.json'), encoding='utf-8') as f:
            r1 = json.load(f)['rows'][0]['row']
        with open(hotpot_path, encoding='utf-8') as f:
            hotpot = json.load(f)
        hp_by_q = {item['question'].strip(): item for item in hotpot}
        docs1 = [p.strip() for p in re.split(r'Document \d+:\n', r1['context']) if p.strip()]
        sessions1 = [
            {'id': f'doc_{i+1}', 'date': '', 'timestamp': 0, 'turns': [{'id': f'doc_{i+1}', 'role': 'user', 'content': d}]}
            for i, d in enumerate(docs1)
        ]
        queries1 = [{'text': q, 'date': ''} for q in r1['questions']]
        t0 = time.time()
        resp1 = rank({'sessions': sessions1, 'queries': queries1, 'top_k': 10})
        dt1 = time.time() - t0

        r1_any1 = r1_any5 = r1_any10 = r1_all5 = r1_all10 = 0
        r1_mrr = 0.0
        r1_fired = r1_ambiguous = 0
        for qi, q in enumerate(r1['questions']):
            tset = {f[0] for f in hp_by_q[q.strip()]['supporting_facts']}
            ret_titles = [docs1[int(s['id'].split('_')[1]) - 1].splitlines()[0].strip() for s in resp1['results'][qi]['sessions']]
            if ret_titles[:1] and ret_titles[0] in tset:
                r1_any1 += 1
            if any(t in tset for t in ret_titles[:5]):
                r1_any5 += 1
            if any(t in tset for t in ret_titles[:10]):
                r1_any10 += 1
            if all(t in ret_titles[:5] for t in tset):
                r1_all5 += 1
            if all(t in ret_titles[:10] for t in tset):
                r1_all10 += 1
            for ri, t in enumerate(ret_titles[:10]):
                if t in tset:
                    r1_mrr += 1.0 / (ri + 1)
                    break
            if resp1['results'][qi].get('group_testing_fired'):
                r1_fired += 1
            if resp1['results'][qi].get('group_testing_ambiguous'):
                r1_ambiguous += 1

        n1 = len(r1['questions'])
        print(f'\nRow 1 (ruler_qa2_421K, {len(docs1)} docs, {n1} Qs) in {dt1:.2f}s:')
        print(f'  Any@1={r1_any1/n1*100:.2f}% Any@5={r1_any5/n1*100:.2f}% Any@10={r1_any10/n1*100:.2f}% All@5={r1_all5/n1*100:.2f}% All@10={r1_all10/n1*100:.2f}% MRR={r1_mrr/n1:.4f}')
        print(f'  Group-testing rule triggers: fired {r1_fired} / ambiguous {r1_ambiguous} / {n1}')

        # 3. Rows 17..21: All 5 longmemeval_s* 400K merged scenarios (300 Qs, 287 non-abs)
        lme_rows = []
        lme_tot = {'n': 0, 'n_total': 0, 'n_abs': 0, 'any1': 0, 'any3': 0, 'any5': 0, 'any10': 0, 'all5': 0, 'all10': 0, 'mrr': 0.0}
        lme_types = {}

        for row_idx in [17, 18, 19, 20, 21]:
            with open(os.path.join(data_dir, f'memoryagentbench_row_{row_idx}.json'), encoding='utf-8') as f:
                rw = json.load(f)['rows'][0]['row']
            flat = ast.literal_eval(rw['context'])
            sessions = []
            sig_to_sid = {}
            for i in range(0, len(flat), 2):
                date_str = str(flat[i]).replace('Chat Time:', '').strip()
                turns_raw = flat[i + 1]
                sid = f's{i // 2}'
                turns = [{'id': f'{sid}#{ti}', 'role': str(t.get('role', 'user')), 'content': str(t.get('content', ''))} for ti, t in enumerate(turns_raw)]
                if turns_raw:
                    sig_to_sid[turns_raw[0]['content'][:180]] = sid
                sessions.append({'id': sid, 'date': date_str, 'timestamp': 0, 'turns': turns})

            queries = []
            gold = []
            qids = rw['metadata'].get('question_ids') or [''] * len(rw['questions'])
            for qi, q_raw in enumerate(rw['questions']):
                q_date = ''
                q_text = q_raw
                m_date = re.match(r'^Current Date:\s*([^,\n]+),?\s*Now Answer the Question:\s*(.*)$', q_raw, re.S)
                if m_date:
                    q_date = m_date.group(1).strip()
                    q_text = m_date.group(2).strip()
                g_ids = set()
                for gs in rw['metadata']['haystack_sessions'][qi]:
                    if any(t.get('has_answer') for t in gs) and gs:
                        sid = sig_to_sid.get(gs[0]['content'][:180])
                        if sid:
                            g_ids.add(sid)
                is_abs = str(qids[qi]).endswith('_abs') or (
                    not any(any(t.get('has_answer') for t in gs) for gs in rw['metadata']['haystack_sessions'][qi])
                )
                queries.append({'text': q_text, 'date': q_date})
                gold.append((rw['metadata']['question_types'][qi], is_abs, str(qids[qi]), g_ids))

            t0 = time.time()
            resp = rank({'sessions': sessions, 'queries': queries, 'top_k': 10})
            dt = time.time() - t0

            row_stat = {
                'row': row_idx, 'sessions': len(sessions), 'n': 0, 'n_total': len(gold), 'n_abs': 0, 'dt': dt,
                'any1': 0, 'any3': 0, 'any5': 0, 'any10': 0, 'all5': 0, 'all10': 0, 'mrr': 0.0,
            }
            for qi, (qtype, is_abs, qid, g_ids) in enumerate(gold):
                lme_tot['n_total'] += 1
                if is_abs:
                    row_stat['n_abs'] += 1
                    lme_tot['n_abs'] += 1
                    continue
                if not g_ids:
                    raise RuntimeError(f'Empty gold IDs for non-abstention row {row_idx} Q{qi} ({qid})')
                row_stat['n'] += 1
                ret = [s['id'] for s in resp['results'][qi]['sessions']]
                a1 = any(g in ret[:1] for g in g_ids)
                a3 = any(g in ret[:3] for g in g_ids)
                a5 = any(g in ret[:5] for g in g_ids)
                a10 = any(g in ret[:10] for g in g_ids)
                l5 = all(g in ret[:5] for g in g_ids)
                l10 = all(g in ret[:10] for g in g_ids)
                mrr_i = 0.0
                for ri, s in enumerate(ret[:10]):
                    if s in g_ids:
                        mrr_i = 1.0 / (ri + 1)
                        break
                for d in (row_stat, lme_tot):
                    d['any1'] += int(a1)
                    d['any3'] += int(a3)
                    d['any5'] += int(a5)
                    d['any10'] += int(a10)
                    d['all5'] += int(l5)
                    d['all10'] += int(l10)
                    d['mrr'] += mrr_i
                lme_tot['n'] += 1
                tb = lme_types.setdefault(qtype, {'n': 0, 'any1': 0, 'any5': 0, 'any10': 0, 'all5': 0, 'all10': 0})
                tb['n'] += 1
                tb['any1'] += int(a1)
                tb['any5'] += int(a5)
                tb['any10'] += int(a10)
                tb['all5'] += int(l5)
                tb['all10'] += int(l10)

            n_r = row_stat['n']
            print(f"Row {row_idx} (longmemeval_s*, {len(sessions)} sess, {n_r}/{row_stat['n_total']} non-abs, {row_stat['n_abs']} abs, {dt:.2f}s): Any@1={row_stat['any1']/n_r*100:.2f}% Any@5={row_stat['any5']/n_r*100:.2f}% Any@10={row_stat['any10']/n_r*100:.2f}% All@5={row_stat['all5']/n_r*100:.2f}% All@10={row_stat['all10']/n_r*100:.2f}% MRR={row_stat['mrr']/n_r:.4f}")
            lme_rows.append(row_stat)
    finally:
        proc.terminate()

    nt = lme_tot['n']
    print(f"\nALL 5 longmemeval_s* Rows ({nt} non-abs / {lme_tot['n_total']} total Qs): Any@1={lme_tot['any1']/nt*100:.2f}% Any@3={lme_tot['any3']/nt*100:.2f}% Any@5={lme_tot['any5']/nt*100:.2f}% Any@10={lme_tot['any10']/nt*100:.2f}% | All@5={lme_tot['all5']/nt*100:.2f}% All@10={lme_tot['all10']/nt*100:.2f}% | MRR={lme_tot['mrr']/nt:.4f}")
    for qt, tb in sorted(lme_types.items()):
        qn = tb['n']
        print(f"  {qt:26s} ({qn:3d}): Any@1={tb['any1']/qn*100:6.2f}% Any@5={tb['any5']/qn*100:6.2f}% All@5={tb['all5']/qn*100:6.2f}% Any@10={tb['any10']/qn*100:6.2f}%")

    total_dt = time.time() - t_start
    out = {
        'benchmark': 'MemoryAgentBench',
        'track': 'Accurate_Retrieval',
        'version': 'v6.2.1',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'elapsed_seconds': round(total_dt, 2),
        'cost_usd': 0,
        'splits': {
            'ruler_qa1_197K': {
                'row': 0, 'docs': len(docs0), 'total_questions': 100,
                'unique_doc_gold_subset': {
                    'n': u_n,
                    'recall_any_at_1': round(u_r1 / u_n * 100, 2),
                    'recall_any_at_5': round(u_r5 / u_n * 100, 2),
                    'recall_any_at_10': round(u_r10 / u_n * 100, 2),
                    'mrr': round(u_mrr / u_n, 4),
                },
                'answer_containing_doc_match': {
                    'n': a_n,
                    'recall_any_at_1': round(a_r1 / a_n * 100, 2),
                    'recall_any_at_5': round(a_r5 / a_n * 100, 2),
                    'recall_any_at_10': round(a_r10 / a_n * 100, 2),
                    'mrr': round(a_mrr / a_n, 4),
                },
            },
            'ruler_qa2_421K': {
                'row': 1, 'docs': len(docs1), 'total_questions': n1,
                'recall_any_at_1': round(r1_any1 / n1 * 100, 2),
                'recall_any_at_5': round(r1_any5 / n1 * 100, 2),
                'recall_any_at_10': round(r1_any10 / n1 * 100, 2),
                'recall_all_at_5': round(r1_all5 / n1 * 100, 2),
                'recall_all_at_10': round(r1_all10 / n1 * 100, 2),
                'mrr': round(r1_mrr / n1, 4),
                'group_testing_fired': r1_fired,
                'group_testing_ambiguous': r1_ambiguous,
            },
            'longmemeval_s_star': {
                'rows': [17, 18, 19, 20, 21],
                'total_questions_all': lme_tot['n_total'],
                'abstentions_excluded': lme_tot['n_abs'],
                'total_questions': nt,
                'recall_any_at_1': round(lme_tot['any1'] / nt * 100, 2),
                'recall_any_at_3': round(lme_tot['any3'] / nt * 100, 2),
                'recall_any_at_5': round(lme_tot['any5'] / nt * 100, 2),
                'recall_any_at_10': round(lme_tot['any10'] / nt * 100, 2),
                'recall_all_at_5': round(lme_tot['all5'] / nt * 100, 2),
                'recall_all_at_10': round(lme_tot['all10'] / nt * 100, 2),
                'mrr': round(lme_tot['mrr'] / nt, 4),
                'per_row': [
                    {
                        'row': rs['row'], 'sessions': rs['sessions'], 'n': rs['n'],
                        'n_total': rs['n_total'], 'n_abs': rs['n_abs'],
                        'recall_any_at_1': round(rs['any1'] / rs['n'] * 100, 2),
                        'recall_any_at_5': round(rs['any5'] / rs['n'] * 100, 2),
                        'recall_all_at_5': round(rs['all5'] / rs['n'] * 100, 2),
                        'recall_any_at_10': round(rs['any10'] / rs['n'] * 100, 2),
                        'mrr': round(rs['mrr'] / rs['n'], 4),
                    } for rs in lme_rows
                ],
                'types': {
                    qt: {
                        'n': tb['n'],
                        'recall_any_at_1': round(tb['any1'] / tb['n'] * 100, 2),
                        'recall_any_at_5': round(tb['any5'] / tb['n'] * 100, 2),
                        'recall_any_at_10': round(tb['any10'] / tb['n'] * 100, 2),
                        'recall_all_at_5': round(tb['all5'] / tb['n'] * 100, 2),
                        'recall_all_at_10': round(tb['all10'] / tb['n'] * 100, 2),
                    } for qt, tb in sorted(lme_types.items())
                },
            },
        },
        'notes': 'EventQA rows (multiple-choice paraphrased next-event selection over novel chapters with no gold chunk IDs in dataset) are not exact-ID retrieval tracks.',
    }

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nCommitted result JSON to: {args.json}')


if __name__ == '__main__':
    main()
