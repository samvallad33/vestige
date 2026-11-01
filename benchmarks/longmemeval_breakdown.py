"""Read-only aggregation of completed official scores; no model or scorer calls."""
from collections import defaultdict
import json
from pathlib import Path

from benchmarks.real_engine.datasets import load_longmemeval
from benchmarks.real_engine.run import ci
from benchmarks.real_engine.runtime import atomic_write,read_jsonl,canonical,BudgetLedger


def main():
    repo=Path(__file__).resolve().parents[1];runs=repo/'runs';root=runs/'longmemeval'
    audit=json.loads((runs/'integrity.json').read_text())['benchmarks']['longmemeval']
    if not audit['ready_to_report'] or audit['status']!='PASS':
        raise RuntimeError('Fresh final integrity report must qualify results before aggregation')
    items=load_longmemeval(runs/'upstream/longmemeval');byid={x.id:x for x in items}
    arms={}
    tables=['# LongMemEval-S official-score breakdown','','| Arm | Question type | Accuracy | 95% bootstrap CI | n |','|---|---|---:|---|---:|']
    summary=[]
    for arm in ('bare','bm25','embedding','vestige'):
        rows=read_jsonl(root/arm/'scored.jsonl')
        if len(rows)!=500 or {x['item_id'] for x in rows}!=set(byid):raise RuntimeError('Incomplete/duplicate official scored set: '+arm)
        arms[arm]={x['item_id']:x for x in rows};groups=defaultdict(list)
        for row in rows:groups[byid[row['item_id']].category].append(row)
        groups['ALL']=rows
        for category,pop in sorted(groups.items()):
            values=[float(x['score']) for x in pop];interval=ci(values,[x['item_id'] for x in pop]);mean=sum(values)/len(values)
            summary.append({'arm':arm,'question_type':category,'accuracy':mean,'ci95':interval,'n':len(pop)})
            tables.append(f'| {arm} | {category} | {mean:.6f} | {interval} | {len(pop)} |')
    candidates=[identity for identity in sorted(byid) if arms['vestige'][identity]['score']==1 and arms['bm25'][identity]['score']==0 and arms['embedding'][identity]['score']==0]
    examples=['# Raw scored comparative examples','',f'{len(candidates)} items meet Vestige correct and both BM25/BGE wrong under the official judge. Showing up to three; no examples invented.','']
    for identity in candidates[:3]:
        item=byid[identity];examples += [f'## {identity} — {item.category}','','Question: '+item.question,'','Official answer: '+str(item.scorer_payload['answer'])]
        for arm in ('vestige','bm25','embedding'):
            row=arms[arm][identity];examples += ['',f'{arm} — official score {row["score"]}; raw evidence {row["raw_log"]}','','```text',row['response'],'```']
    snapshot=BudgetLedger(runs/'budget.db').snapshot()
    tables += ['','Budget ledger snapshot: `'+canonical(snapshot)+'`.','Costs preserve historical ledger totals separately from the owner-authorized new $21 window.','']
    atomic_write(root/'QUESTION_TYPES.md','\n'.join(tables));atomic_write(root/'question-types.json',canonical(summary)+'\n')
    atomic_write(root/'COMPARATIVE_EXAMPLES.md','\n'.join(examples)+'\n')
    with (root/'RESULTS.md').open('a') as out:out.write('\n\nQuestion-type accuracy and confidence intervals: [QUESTION_TYPES.md](QUESTION_TYPES.md).\n\nRaw officially judged comparative examples: [COMPARATIVE_EXAMPLES.md](COMPARATIVE_EXAMPLES.md).\n\nNew-window spend: $'+str(snapshot['spent'])+'; outstanding reservations: $'+str(snapshot['reserved'])+'; remaining: $'+str(snapshot['remaining'])+'.\n')
    print(canonical({'overall':[r for r in summary if r['question_type']=='ALL'],'comparative_examples_available':len(candidates),'budget':snapshot}))


if __name__=='__main__':main()
