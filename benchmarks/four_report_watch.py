"""No-inference final reporter for the owner's four-benchmark campaign.

Wait for the already-running single controller. Never restart it or issue API
calls. Publish only independently qualified scored aggregates; list incomplete
and invalid benchmarks separately with their exact coverage artifact.
"""
from pathlib import Path
import argparse
import json
import os
import sqlite3
import time
from benchmarks.real_engine.integrity import audit
from benchmarks.real_engine.runtime import atomic_write,canonical,utc_now

NAMES=('memconflict','locomo','longmemeval','mpbench')
ARMS=('bare','bm25','embedding','vestige')

def load(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def published_context(repo,name):
    label={'locomo':'LoCoMo','longmemeval':'LongMemEval'}.get(name)
    records=load(repo/'benchmarks/real_engine/published.json',{}).get('sources',[])
    selected=[r for r in records if label and r.get('benchmark')==label]
    return '; '.join(f"[{r['system']} {r['score_percent']}%]({r['source']}) ({r['model']}; different metric/scope or judge)" for r in selected) or 'No source recorded for this benchmark'

def generate(repo,runs,config):
    report=audit(repo,runs,benchmarks=NAMES)
    with sqlite3.connect(f'file:{runs / "budget.db"}?mode=ro',uri=True) as db:
        spent,reserved=db.execute("SELECT COALESCE(SUM(actual),0),COALESCE(SUM(CASE WHEN state='reserved' THEN reserved ELSE 0 END),0) FROM calls").fetchone()
    text=['# Four-benchmark real-engine campaign','',f'Updated: {utc_now()}',
        f'Configuration: `{config}`. Model and judge: gpt-5.4-nano. API spend: ${spent/1e9:.6f}; unresolved reservations: ${reserved/1e9:.6f}; shared cap: $25.',
        '', '| Benchmark | Bare | BM25 | BGE-small | Vestige + memory-QA hooks | published, different model or judge |',
        '|---|---|---|---|---|---|']
    unfinished=[]
    for name in NAMES:
        state=report['benchmarks'][name]
        if not state['ready_to_report']:
            cov=load(runs/name/'coverage.json',{}).get('arms',{})
            counts={a:{'done':cov.get(a,{}).get('completed_n',0),'expected':cov.get(a,{}).get('expected_n')} for a in ARMS}
            unfinished.append({'benchmark':name,'status':'INVALID' if state['status']=='FAIL' else 'UNFINISHED',
                'coverage':counts,'coverage_file':str(runs/name/'coverage.json'),
                'reasons':sorted({c['reason'] for a in state['arms'].values() for c in a['checks'] if c['status']!='PASS'})})
            continue
        if name=='mpbench':
            result=load(runs/name/'summary.json',{})
            by_arm={row['arm']:row for row in result.get('arms',[])}
            mp=['# MPBench real-engine results','',
                'scored with our implementation of the paper’s metrics, no official scorer available.',
                '', 'Model and judge: gpt-5.4-nano. Trial: 0. Ten workers per arm; global20-request limit.',
                '', '| Arm | Metric | Full-set mean | Scored-subset mean | 95% bootstrap CI on scored subset | n / expected | Full-set bounds | API cost |',
                '|---|---|---:|---:|---|---:|---|---:|']
            cells=[]
            for a in ARMS:
                m=by_arm[a]['metrics'];attack=m['ASR'];fp=m['FPR']
                cells.append(f"ASR full={attack['mean']}, subset={attack['scored_subset_mean']}, n={attack['n']}/{attack['expected_n']}, CI={attack['ci95_scored_subset']}; FPR full={fp['mean']}, subset={fp['scored_subset_mean']}, n={fp['n']}/{fp['expected_n']}, CI={fp['ci95_scored_subset']}")
                metrics={'ASR':attack,'benign FPR':fp,**{f'ASR: {k}':v for k,v in m['by_attack_class'].items()}}
                for label,value in metrics.items():
                    mp.append(f"| {a} | {label} | {value['mean']} | {value['scored_subset_mean']} | {value['ci95_scored_subset']} | {value['n']} / {value['expected_n']} | {value['full_set_bounds']} | ${by_arm[a]['actual_cost']:.6f} |")
            build=load(runs/'build-manifest.json');q=load(runs/'QUALIFICATION.json')['benchmarks']['mpbench']
            mp+=['',f"Binary SHA-256: `{build['binary_sha256']}`.",
                'Dataset source/hash: `'+canonical(result.get('dataset'))+'`.',
                'Frozen scorer: `'+canonical(q['scorer'])+'`.',
                'Actual bootstrap metadata: `'+canonical({a:{k:by_arm[a]['metrics'][k] for k in ('bootstrap_seed','bootstrap_resamples','bootstrap_unit','trials')} for a in ARMS})+'`.',
                '', 'None means unscorable or unavailable, never zero. Missing public targets remain in full-set denominators. Scored-subset CIs do not fill those missing targets.',
                'A bare model has no long-term store; its empty-store ASR is structural, not evidence of a memory defense. FPR requires an observed native defense decision. No-write behavior is not inserted as a negative detector label.',
                'All six attack classes and the benign population are included above. Raw skill artifacts are delivered unchanged; autonomous skill synthesis and cross-session RSR are not measured.',
                'Costs include preparation and failed attempts; unresolved reservations remain in the shared ledger. Published, different model or judge: no directly comparable external score is asserted.', '']
            previous=runs/name/'RESULTS.md'
            if previous.exists():atomic_write(runs/name/'RESULTS-before-expanded-metric-table.md',previous.read_text())
            atomic_write(previous,'\n'.join(mp))
        else:
            by_arm={row['arm']:row for row in load(runs/name/'results.json',[])}
            cells=[f"mean={by_arm[a]['mean']:.6f}; CI={by_arm[a]['ci95']}; n={by_arm[a]['n']}" for a in ARMS]
        text.append(f'| [{name}]({name}/RESULTS.md) | '+' | '.join(cells)+' | '+published_context(repo,name)+' |')
    text+=['','All non-qualified benchmarks are excluded from the score table:', '']
    for entry in unfinished:
        text.append(f"- **{entry['benchmark']}: {entry['status']}**. Coverage `{canonical(entry['coverage'])}`. Exact finished and missing IDs: `{entry['coverage_file']}`.")
        text.extend('- '+reason for reason in entry['reasons'])
    text+=['','MPBench uses “scored with our implementation of the paper’s metrics, no official scorer available.” Two missing public targets and unobserved defense decisions remain explicitly unscorable; no synthetic labels are inserted.',
        '', 'Published cells quote the source-attributed records in benchmarks/real_engine/published.json, checked2026-11-01. Published metrics, scopes, model and judge differ from this campaign; no head-to-head ranking or delta is implied. Preserve the source caveats, including the recorded Zep headline/category-denominator discrepancy.',
        '', 'Only these four benchmarks were authorized in this run. Earlier broader-campaign material remains preserved and is outside the current score table.', '']
    # Every RESULTS write has a fresh audit above; metric rows require PASS.
    audit(repo,runs,benchmarks=NAMES)
    atomic_write(runs/'RESULTS.md','\n'.join(text))
    atomic_write(runs/'four-final-report.json',canonical({'timestamp':utc_now(),'config_hash':config,
        'settled_cost':spent/1e9,'reserved_cost':reserved/1e9,'unfinished_or_invalid':unfinished})+'\n')
    return {'unfinished_or_invalid':[x['benchmark'] for x in unfinished],'cost':spent/1e9}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--once',action='store_true')
    args=parser.parse_args();repo=args.repo.resolve();runs=repo/'runs'
    controller=load(runs/'four-controller.json');config=load(runs/'four-config.json')['config_hash']
    if not args.once:
        while True:
            final=load(runs/'campaign-finished.json',{})
            if final.get('config_hash')==config:break
            try:os.kill(controller['pid'],0)
            except ProcessLookupError:break
            time.sleep(30)
    result=generate(repo,runs,config);print(canonical(result),flush=True)

if __name__=='__main__':main()
