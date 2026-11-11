// Stage 1: zero-model retrieval benchmark (LongMemEval_S + LoCoMo-10).
//
// 100% of ranking runs inside the compiled Rust kernel (`retrieval_bench` from
// vestige-core, source: crates/vestige-core/src/retrieval.rs).
// This file only marshals data in and scores ranked ids out. There is no JS ranker.
//
// Leak rules (enforced, not just promised):
//   1. Ingestion builds turns from `role`/`speaker`, `content`/`text`, a turn id, and the
//      session date only. Queries carry only the question text and question date.
//   2. The Rust structs are `deny_unknown_fields`: a payload with `has_answer`, `answer`,
//      `question_type`, `category`, `evidence` ... is rejected at deserialize time.
//   3. A static audit fails the run if the ingestion functions reference any label field.
//   4. Labels (`answer_session_ids`, `evidence`, `question_type`, `category`) are read only
//      in the scoring/reporting functions, after Rust has returned its ranking.
//
// Usage:
//   node benchmarks/run_stage1_retrieval.mjs [--baseline] [--held-out] [--limit=N]
//        [--off=enable_window,enable_dyad] [--set=window_mix:0.5,funnel:0.8]
//        [--ablations] [--json=out.json] [--only=lme|locomo]

import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import os from 'node:os';

const rootDir = resolve('.');
const binCandidates = [
  join(rootDir, 'target', 'release', 'retrieval_bench.exe'),
  join(rootDir, 'target', 'release', 'retrieval_bench'),
  join(rootDir, 'target', 'debug', 'retrieval_bench.exe'),
  join(rootDir, 'target', 'debug', 'retrieval_bench'),
];
const binPath = binCandidates.find(p => existsSync(p));
if (!binPath) {
  console.error('ERROR: retrieval_bench binary not found. Build it with:');
  console.error('  cargo build --release -p vestige-core --bin retrieval_bench');
  process.exit(1);
}

const child = spawn(binPath, ['--worker'], { stdio: ['pipe', 'pipe', 'inherit'] });
const rl = createInterface({ input: child.stdout, crlfDelay: Infinity });

let queue = [];
rl.on('line', (line) => {
  if (queue.length > 0) {
    const { resolve, reject } = queue.shift();
    try {
      const data = JSON.parse(line);
      if (data && data.error) {
        reject(new Error(data.error));
      } else {
        resolve(data);
      }
    } catch (e) {
      reject(e);
    }
  }
});

child.on('error', (err) => {
  while (queue.length > 0) {
    queue.shift().reject(err);
  }
});

function rank(payload) {
  return new Promise((resolve, reject) => {
    queue.push({ resolve, reject });
    child.stdin.write(JSON.stringify(payload) + '\n');
  });
}

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const [k, v] = a.replace(/^--/, '').split('=');
  return [k, v ?? true];
}));
const LIMIT = args.limit ? Number(args.limit) : Infinity;

function paramsFromArgs(extraOff = []) {
  const p = {};
  if (args.baseline) {
    // Pure baseline: all algorithmic expansions disabled, standard BM25 IDF power 1.0
    p.enable_stemming = false;
    p.enable_proximity = false;
    p.enable_coverage = false;
    p.enable_roles = false;
    p.enable_preference = false;
    p.enable_hypernyms = false;
    p.enable_recency = false;
    p.enable_window = false;
    p.enable_carry = false;
    p.enable_deictic = false;
    p.enable_funnel = false;
    p.enable_dyad = false;
    p.enable_bridge = false;
    p.enable_scene = false;
    p.enable_sniper = false;
    p.enable_ppr = false;
    p.enable_submodular = false;
    p.idf_power = 1.0;
  }
  for (const k of [...String(args.off || '').split(',').filter(Boolean), ...extraOff]) p[k] = false;
  for (const kv of String(args.set || '').split(',').filter(Boolean)) {
    const [k, v] = kv.split(':');
    p[k] = Number(v);
  }
  return p;
}

// ---------------------------------------------------------------------------
// INGESTION (label-blind). Only these functions build what crosses into Rust.
// ---------------------------------------------------------------------------
function ingestLongMemEval(item) {
  const sessions = item.haystack_sessions.map((sess, i) => ({
    id: String(item.haystack_session_ids[i]),
    date: String(item.haystack_dates[i] || ''),
    turns: sess.map((t, j) => ({ id: `${item.haystack_session_ids[i]}#${j}`, role: String(t.role), content: String(t.content) })),
  }));
  const query = { text: String(item.question), date: String(item.question_date || '') };
  return { sessions, query };
}

function ingestLoCoMo(conv) {
  const c = conv.conversation;
  const sessions = [];
  for (let s = 1; c[`session_${s}`]; s++) {
    sessions.push({
      id: `D${s}`,
      date: String(c[`session_${s}_date_time`] || ''),
      turns: c[`session_${s}`].filter(t => t.text && t.dia_id).map(t => {
        let blip = t.blip_caption ? `\n[BLIP caption] ${t.blip_caption}` : '';
        return {
          id: String(t.dia_id),
          role: String(t.speaker || ''),
          content: `${t.text}${blip}`,
        };
      }),
    });
  }
  const queries = conv.qa.map(q => ({ text: String(q.question), date: '' }));
  return { sessions, queries };
}

// ---------------------------------------------------------------------------
// FORENSIC GUARDS
// ---------------------------------------------------------------------------
function staticAudit() {
  const forbidden = ['has_answer', 'answer_session_ids', 'question_type', 'category', 'evidence', '.answer', "['answer']", 'adversarial_answer'];
  for (const fn of [ingestLongMemEval, ingestLoCoMo]) {
    const src = fn.toString();
    for (const f of forbidden) {
      if (src.includes(f)) throw new Error(`LEAK: ${fn.name} references label field ${f}`);
    }
  }
  return forbidden.length;
}

async function boundarySelfTest() {
  const probes = [
    { sessions: [{ id: 's', turns: [{ role: 'user', content: 'x', has_answer: true }] }], queries: [{ text: 'x' }] },
    { sessions: [], queries: [{ text: 'x', question_type: 'knowledge-update' }] },
    { sessions: [{ id: 's', turns: [], answer_session_ids: ['s'] }], queries: [{ text: 'x' }] },
    { sessions: [], queries: [{ text: 'x', category: 1 }] },
  ];
  let rejected = 0;
  for (const p of probes) {
    try { await rank(p); } catch { rejected++; }
  }
  if (rejected !== probes.length) throw new Error(`LEAK: Rust accepted ${probes.length - rejected} labelled payloads`);
  return rejected;
}

// ---------------------------------------------------------------------------
// METRICS (labels are read here, after ranking)
// ---------------------------------------------------------------------------
const pct = (n, d) => (d ? (100 * n / d).toFixed(2) : '  n/a');
function ndcg(rankedIds, gold, k) {
  let dcg = 0, idcg = 0;
  rankedIds.slice(0, k).forEach((id, i) => { if (gold.has(id)) dcg += 1 / Math.log2(i + 2); });
  for (let i = 0; i < Math.min(gold.size, k); i++) idcg += 1 / Math.log2(i + 2);
  return idcg ? dcg / idcg : 0;
}
function pctile(arr, p) {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(s.length * p))] ?? 0;
}

let LME_DATA = null;
async function runLongMemEval(params, quiet = false) {
  if (!LME_DATA) LME_DATA = JSON.parse(readFileSync('benchmarks/data/longmemeval_s_cleaned.json', 'utf8'));
  const data = LME_DATA.slice(0, LIMIT);
  const buckets = {};
  const all = { n: 0, r1: 0, r5: 0, r10: 0, all5: 0, all10: 0, ndcg10: 0 };
  const nonAbs = { n: 0, r1: 0, r5: 0, r10: 0, all5: 0, all10: 0, ndcg10: 0 };
  const lat = [];
  let sessionsSeen = 0, turnsSeen = 0;
  for (const item of data) {
    const { sessions, query } = ingestLongMemEval(item);
    const t0 = performance.now();
    const res = await rank({ sessions, queries: [query], top_k: 10, params });
    lat.push(performance.now() - t0);
    sessionsSeen += res.sessions_indexed; turnsSeen += res.turns_indexed;
    const ids = res.results[0].sessions.map(s => s.id);

    const gold = new Set(item.answer_session_ids || []);
    const type = item.question_type;
    const isAbs = String(item.question_id).endsWith('_abs');
    const m = {
      r1: ids.slice(0, 1).some(i => gold.has(i)),
      r5: ids.slice(0, 5).some(i => gold.has(i)),
      r10: ids.slice(0, 10).some(i => gold.has(i)),
      all5: gold.size > 0 && [...gold].every(g => ids.slice(0, 5).includes(g)),
      all10: gold.size > 0 && [...gold].every(g => ids.slice(0, 10).includes(g)),
      ndcg10: ndcg(ids, gold, 10),
    };
    for (const acc of [all, isAbs ? null : nonAbs, (buckets[type] ??= { n: 0, r1: 0, r5: 0, r10: 0, all5: 0, all10: 0, ndcg10: 0 })]) {
      if (!acc) continue;
      acc.n++; acc.r1 += m.r1; acc.r5 += m.r5; acc.r10 += m.r10; acc.all5 += m.all5; acc.all10 += m.all10; acc.ndcg10 += m.ndcg10;
    }
  }
  if (!quiet) {
    console.log('--- LongMemEval_S (longmemeval_s_cleaned.json) | ranking 100% in Rust kernel ---');
    console.log(`Questions: ${all.n} total (${nonAbs.n} non-abstention) | avg haystack: ${(sessionsSeen / all.n).toFixed(1)} sessions, ${(turnsSeen / all.n).toFixed(0)} turns per question`);
    console.log(`\n[Official LongMemEval Metric: 470 non-abstention questions]`);
    console.log(`  RecallAll@5:  ${pct(nonAbs.all5, nonAbs.n)}%`);
    console.log(`  RecallAll@10: ${pct(nonAbs.all10, nonAbs.n)}%`);
    console.log(`  NDCG@10:      ${(nonAbs.ndcg10 / nonAbs.n).toFixed(4)}`);
    console.log(`  RecallAny@1:  ${pct(nonAbs.r1, nonAbs.n)}%`);
    console.log(`  RecallAny@5:  ${pct(nonAbs.r5, nonAbs.n)}%`);
    console.log(`  RecallAny@10: ${pct(nonAbs.r10, nonAbs.n)}%`);

    console.log(`\n[All 500 questions (includes 30 abstentions, recall_any)]`);
    console.log(`  RecallAny@1:  ${pct(all.r1, all.n)}% | RecallAny@5: ${pct(all.r5, all.n)}% | RecallAny@10: ${pct(all.r10, all.n)}% | RecallAll@5: ${pct(all.all5, all.n)}% | NDCG@10: ${(all.ndcg10 / all.n).toFixed(4)}`);

    console.log(`\nQuestion Type Breakdown (Non-Abstention at same k):`);
    for (const [t, s] of Object.entries(buckets)) {
      console.log(`  ${t.padEnd(26)} R@5 ${pct(s.r5, s.n).padStart(6)}% | R@10 ${pct(s.r10, s.n).padStart(6)}% | RecallAll@5 ${pct(s.all5, s.n).padStart(6)}% | NDCG@10 ${(s.ndcg10 / s.n).toFixed(3)} (${s.n})`);
    }
    console.log(`\nLatency per question: avg ${(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(2)} ms | p50 ${pctile(lat, 0.5).toFixed(2)} ms | p95 ${pctile(lat, 0.95).toFixed(2)} ms`);
  }
  const formatLme = (a) => ({
    n: a.n,
    recall_any_r1: Number(pct(a.r1, a.n)),
    recall_any_r5: Number(pct(a.r5, a.n)),
    recall_any_r10: Number(pct(a.r10, a.n)),
    recall_all_r5: Number(pct(a.all5, a.n)),
    recall_all_r10: Number(pct(a.all10, a.n)),
    ndcg_10: Number((a.ndcg10 / a.n).toFixed(4)),
  });
  return {
    official_470_non_abs: formatLme(nonAbs),
    all_500: formatLme(all),
    types: Object.fromEntries(Object.entries(buckets).map(([t, s]) => [t, formatLme(s)])),
    raw: { all, nonAbs, buckets },
  };
}

let LOCOMO_DATA = null;
async function runLoCoMo(params, quiet = false, convList = null) {
  if (!LOCOMO_DATA) LOCOMO_DATA = JSON.parse(readFileSync('benchmarks/data/locomo10.json', 'utf8'));
  let convs = convList;
  if (!convs) {
    if (args.split === 'tune') convs = LOCOMO_DATA.slice(0, 5);
    else if (args.split === 'test') convs = LOCOMO_DATA.slice(5, 10);
    else convs = LOCOMO_DATA;
  }
  const names = { 1: 'Multi-Hop', 2: 'Temporal', 3: 'Open-Domain', 4: 'Single-Hop', 5: 'Adversarial' };
  const blank = () => ({ n: 0, noEvidence: 0, sR1: 0, sR5: 0, tR1: 0, tR5: 0, tR10: 0, softR1: 0, softR5: 0, fracR5: 0 });
  const cats = {}; const core = blank(); const total = blank();
  const lat = [];
  let qCount = 0;
  for (const conv of convs) {
    const { sessions, queries } = ingestLoCoMo(conv);
    const t0 = performance.now();
    const res = await rank({ sessions, queries, top_k: 10, params });
    lat.push((performance.now() - t0) / Math.max(1, queries.length));
    res.results.forEach((r, qi) => {
      if (qCount++ >= LIMIT * 4) return;
      const qa = conv.qa[qi];
      const cat = qa.category;
      const goldTurns = new Set((qa.evidence || []).flatMap(e => String(e).split(/[;,\s]+/)).filter(e => /^D\d+:\d+$/.test(e)));
      const goldSess = new Set([...goldTurns].map(e => e.split(':')[0]));
      const accs = [(cats[cat] ??= blank()), total, cat === 5 ? null : core].filter(Boolean);
      if (goldTurns.size === 0) { accs.forEach(a => a.noEvidence++); return; }
      const t = r.turns.map(x => x.id), s = r.sessions.map(x => x.id);

      // Soft gold set (includes ±1 adjacent turn within same exchange)
      const softGold = new Set();
      for (const g of goldTurns) {
        softGold.add(g);
        const [sessId, tNumStr] = g.split(':');
        const tNum = parseInt(tNumStr, 10);
        if (!isNaN(tNum)) {
          softGold.add(`${sessId}:${tNum - 1}`);
          softGold.add(`${sessId}:${tNum + 1}`);
        }
      }

      const retrieved5 = new Set(t.slice(0, 5));
      const matchedCount = [...goldTurns].filter(g => retrieved5.has(g)).length;
      const fracR5 = goldTurns.size > 0 ? (matchedCount / goldTurns.size) : 0;

      const m = {
        sR1: s.slice(0, 1).some(i => goldSess.has(i)),
        sR5: s.slice(0, 5).some(i => goldSess.has(i)),
        tR1: t.slice(0, 1).some(i => goldTurns.has(i)),
        tR5: t.slice(0, 5).some(i => goldTurns.has(i)),
        tR10: t.slice(0, 10).some(i => goldTurns.has(i)),
        softR1: t.slice(0, 1).some(i => softGold.has(i)),
        softR5: t.slice(0, 5).some(i => softGold.has(i)),
        fracR5,
      };
      for (const a of accs) {
        a.n++;
        for (const k of ['sR1', 'sR5', 'tR1', 'tR5', 'tR10', 'softR1', 'softR5', 'fracR5']) a[k] += m[k];
      }
    });
  }
  if (!quiet) {
    const splitLabel = args.split && args.split !== 'all' ? ` (split: ${args.split})` : '';
    console.log(`\n--- LoCoMo-10 (locomo10.json) | ${convs.length} dialogues${splitLabel} | ranking 100% in Rust kernel ---`);
    console.log('Note: "Soft" turn retrieval defines a match as the gold turn or ±1 adjacent turn in the same exchange.');
    console.log('Note: Cat 5 (Adversarial) measures finding the source passage for the adversarial question, not resisting the false premise.');
    const line = (label, a) => console.log(`${label.padEnd(28)} Session [R@1: ${pct(a.sR1, a.n)}%, R@5: ${pct(a.sR5, a.n)}%] | Turn Exact R@5: ${pct(a.tR5, a.n)}% | Turn Soft R@5: ${pct(a.softR5, a.n)}% | Exact [R@1: ${pct(a.tR1, a.n)}%, R@10: ${pct(a.tR10, a.n)}%] (${a.n} scored${a.noEvidence ? `, ${a.noEvidence} without evidence excluded` : ''})`);
    line('CORE (Cat 1-4)', core);
    line('ALL (Cat 1-5)', total);
    for (const c of [1, 2, 3, 4, 5]) if (cats[c]) line(`  Cat ${c} ${names[c]}`, cats[c]);
    console.log(`Latency per question: avg ${(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(3)} ms`);
  }
  const formatAcc = (a) => ({
    n: a.n,
    noEvidence: a.noEvidence,
    session_r1: Number(pct(a.sR1, a.n)),
    session_r5: Number(pct(a.sR5, a.n)),
    turn_exact_r1: Number(pct(a.tR1, a.n)),
    turn_exact_r5: Number(pct(a.tR5, a.n)),
    turn_exact_r10: Number(pct(a.tR10, a.n)),
    turn_soft_r1: Number(pct(a.softR1, a.n)),
    turn_soft_r5: Number(pct(a.softR5, a.n)),
  });

  return {
    dialogues: convs.length,
    core: formatAcc(core),
    total: formatAcc(total),
    categories: Object.fromEntries(Object.entries(cats).map(([c, a]) => [names[c] || c, formatAcc(a)])),
    raw: { core, total, cats },
  };
}

// ---------------------------------------------------------------------------
// MAIN EXECUTION
// ---------------------------------------------------------------------------
async function main() {
  console.log('================================================================================');
  console.log('  VESTIGE ENGINE · Stage 1 zero-model retrieval · Native Rust kernel');
  console.log('================================================================================');
  console.log(`Static audit: ingestion functions free of ${staticAudit()} label fields  ✔`);
  const rejected = await boundarySelfTest();
  console.log(`Boundary self-test: Rust rejected ${rejected}/4 labelled payloads  ✔\n`);

  const params = paramsFromArgs();
  const out = {
    system: {
      platform: os.platform(),
      arch: os.arch(),
      cpus: os.cpus()[0]?.model || 'unknown',
      total_mem_mb: Math.round(os.totalmem() / (1024 * 1024)),
    },
    params,
  };

  if (args.only !== 'locomo') out.lme = await runLongMemEval(params);
  if (args.only !== 'lme') out.locomo = await runLoCoMo(params);

  if (args['held-out']) {
    console.log('\n--- HELD-OUT EVALUATION: LoCoMo Convs 1-5 vs Convs 6-10 ---');
    if (!LOCOMO_DATA) LOCOMO_DATA = JSON.parse(readFileSync('benchmarks/data/locomo10.json', 'utf8'));
    const part1 = LOCOMO_DATA.slice(0, 5);
    const part2 = LOCOMO_DATA.slice(5, 10);
    const res1 = await runLoCoMo(params, true, part1);
    const res2 = await runLoCoMo(params, true, part2);
    out.held_out = { convs_1_to_5: res1, convs_6_to_10: res2 };
    console.log(`LoCoMo Convs 1-5  (${res1.core.n} core q) | Session R@5: ${pct(res1.core.sR5, res1.core.n)}% | Turn Exact R@5: ${pct(res1.core.tR5, res1.core.n)}% | Turn Soft R@5: ${pct(res1.core.softR5, res1.core.n)}%`);
    console.log(`LoCoMo Convs 6-10 (${res2.core.n} core q) | Session R@5: ${pct(res2.core.sR5, res2.core.n)}% | Turn Exact R@5: ${pct(res2.core.tR5, res2.core.n)}% | Turn Soft R@5: ${pct(res2.core.softR5, res2.core.n)}%`);
  }

  if (args.ablations) {
    const toggles = [
      'enable_stemming',
      'enable_proximity',
      'enable_coverage',
      'enable_roles',
      'enable_preference',
      'enable_hypernyms',
      'enable_recency',
      'enable_window',
      'enable_carry',
      'enable_deictic',
      'enable_funnel',
      'enable_dyad',
      'enable_bridge',
      'enable_scene',
      'enable_sniper',
      'enable_ppr',
      'enable_submodular',
    ];
    console.log('\n--- ABLATIONS (each weapon switched off alone) ---');
    console.log('Note: scene, sniper, ppr, and submodular re-rank turns within sessions; they affect Turn scores.');
    console.log('weapon off           | LME R@1  LME R@5 | LoCoMo core Sess R@5  Turn R@5 (Exact)  Turn R@5 (Soft)');
    out.ablations = {};
    const fmt = (lme, lo) => {
      const lmeR1 = (lme.all_500 ? lme.all_500.recall_any_r1.toFixed(2) : pct(lme.all.r1, lme.all.n)).padStart(6);
      const lmeR5 = (lme.all_500 ? lme.all_500.recall_any_r5.toFixed(2) : pct(lme.all.r5, lme.all.n)).padStart(6);
      const loSessR5 = (lo.core.session_r5 !== undefined ? lo.core.session_r5.toFixed(2) : pct(lo.core.sR5, lo.core.n)).padStart(18);
      const loTurnExactR5 = (lo.core.turn_exact_r5 !== undefined ? lo.core.turn_exact_r5.toFixed(2) : pct(lo.core.tR5, lo.core.n)).padStart(15);
      const loTurnSoftR5 = (lo.core.turn_soft_r5 !== undefined ? lo.core.turn_soft_r5.toFixed(2) : pct(lo.core.softR5, lo.core.n)).padStart(14);
      return `${lmeR1}%  ${lmeR5}% | ${loSessR5}%  ${loTurnExactR5}%  ${loTurnSoftR5}%`;
    };
    console.log(`${'(none)'.padEnd(20)} | ${fmt(out.lme, out.locomo)}`);
    const turnOnlyFeatures = [
      'enable_funnel',
      'enable_dyad',
      'enable_bridge',
      'enable_scene',
      'enable_sniper',
      'enable_ppr',
      'enable_submodular',
    ];
    for (const t of toggles) {
      const p = { ...params, [t]: false };
      // Features that only modify turn ranking within sessions do not change session-level LongMemEval results
      const lme = turnOnlyFeatures.includes(t) ? out.lme : await runLongMemEval(p, true);
      const lo = await runLoCoMo(p, true);
      out.ablations[t] = { lme, locomo: lo };
      console.log(`${t.replace('enable_', '').padEnd(20)} | ${fmt(lme, lo)}`);
    }
  }

  if (args.json) {
    const jsonPath = resolve(args.json);
    writeFileSync(jsonPath, JSON.stringify(out, null, 2));
    console.log(`\nResults written to: ${jsonPath}`);
  }

  console.log('\n================================================================================');
  console.log(`  COMPLETE · ZERO LLM · ZERO VECTORS · RSS ${(process.memoryUsage().rss / 1048576).toFixed(0)} MB`);
  console.log('================================================================================');

  child.kill();
  process.exit(0);
}

main().catch(err => {
  console.error('Benchmark fatal error:', err);
  child.kill();
  process.exit(1);
});
