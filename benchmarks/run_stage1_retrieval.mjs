// Stage 1: zero-model retrieval benchmark (LongMemEval_S + LoCoMo-10).
//
// 100% of ranking runs inside the compiled Rust WASM kernel (`rank_memories` from
// salience-web-core, source: salience-pro/crates/salience-core/src/retrieval.rs).
// This file only marshals data in and scores ranked ids out. There is no JS ranker.
//
// Leak rules (enforced, not just promised):
//   1. Ingestion builds turns from `role`/`speaker`, `content`/`text`, a turn id, and the
//      session date only. Queries carry only the question text and question date.
//   2. The Rust structs are `deny_unknown_fields`: a payload with `has_answer`, `answer`,
//      `question_type`, `category`, `evidence` ... is rejected. A self-test proves it.
//   3. A static audit fails the run if the ingestion functions reference any label field.
//   4. Labels (`answer_session_ids`, `evidence`, `question_type`, `category`) are read only
//      in the scoring/reporting functions, after Rust has returned its ranking.
//
// Usage:
//   node benchmarks/run_stage1_retrieval.mjs [--limit=N] [--off=enable_window,enable_dyad]
//        [--set=window_mix:0.5,funnel:0.8] [--ablations] [--json=out.json] [--only=lme|locomo]
import init, { rank_memories } from 'file:///C:/Users/Sam/Documents/ChatGPT/New%20project/salience-pro/app/src/lib/web/instant-core/salience_web_core.js';
import { readFileSync, writeFileSync } from 'node:fs';

const WASM = 'C:/Users/Sam/Documents/ChatGPT/New project/salience-pro/app/src/lib/web/instant-core/salience_web_core_bg.wasm';
await init({ module_or_path: readFileSync(WASM) });

const args = Object.fromEntries(process.argv.slice(2).map(a => {
  const [k, v] = a.replace(/^--/, '').split('=');
  return [k, v ?? true];
}));
const LIMIT = args.limit ? Number(args.limit) : Infinity;

function paramsFromArgs(extraOff = []) {
  const p = {};
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
      turns: c[`session_${s}`].filter(t => t.text && t.dia_id).map(t => ({
        id: String(t.dia_id), role: String(t.speaker || ''), content: String(t.text),
      })),
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

function boundarySelfTest() {
  const probes = [
    { sessions: [{ id: 's', turns: [{ role: 'user', content: 'x', has_answer: true }] }], queries: [{ text: 'x' }] },
    { sessions: [], queries: [{ text: 'x', question_type: 'knowledge-update' }] },
    { sessions: [{ id: 's', turns: [], answer_session_ids: ['s'] }], queries: [{ text: 'x' }] },
    { sessions: [], queries: [{ text: 'x', category: 1 }] },
  ];
  let rejected = 0;
  for (const p of probes) {
    try { rank_memories(JSON.stringify(p)); } catch { rejected++; }
  }
  if (rejected !== probes.length) throw new Error(`LEAK: Rust accepted ${probes.length - rejected} labelled payloads`);
  return rejected;
}

function rank(payload) {
  return JSON.parse(rank_memories(JSON.stringify(payload)));
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
function runLongMemEval(params, quiet = false) {
  if (!LME_DATA) LME_DATA = JSON.parse(readFileSync('benchmarks/data/longmemeval_s_cleaned.json', 'utf8'));
  const data = LME_DATA.slice(0, LIMIT);
  const buckets = {};
  const all = { n: 0, r1: 0, r5: 0, r10: 0, all5: 0, ndcg10: 0 };
  const nonAbs = { n: 0, r1: 0, r5: 0, r10: 0, all5: 0, ndcg10: 0 };
  const lat = [];
  let sessionsSeen = 0, turnsSeen = 0;
  for (const item of data) {
    const { sessions, query } = ingestLongMemEval(item);
    const t0 = performance.now();
    const res = rank({ sessions, queries: [query], top_k: 10, params });
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
      ndcg10: ndcg(ids, gold, 10),
    };
    for (const acc of [all, isAbs ? null : nonAbs, (buckets[type] ??= { n: 0, r1: 0, r5: 0, r10: 0, all5: 0, ndcg10: 0 })]) {
      if (!acc) continue;
      acc.n++; acc.r1 += m.r1; acc.r5 += m.r5; acc.r10 += m.r10; acc.all5 += m.all5; acc.ndcg10 += m.ndcg10;
    }
  }
  if (!quiet) {
    console.log('--- LongMemEval_S (longmemeval_s_cleaned.json) | ranking 100% in Rust WASM ---');
    console.log(`Questions: ${all.n} | avg haystack: ${(sessionsSeen / all.n).toFixed(1)} sessions, ${(turnsSeen / all.n).toFixed(0)} turns`);
    console.log(`ALL 500      R@1 ${pct(all.r1, all.n)}% | R@5 ${pct(all.r5, all.n)}% | R@10 ${pct(all.r10, all.n)}% | RecallAll@5 ${pct(all.all5, all.n)}% | NDCG@10 ${(all.ndcg10 / all.n).toFixed(4)}`);
    console.log(`NON-ABS ${String(nonAbs.n).padEnd(4)} R@1 ${pct(nonAbs.r1, nonAbs.n)}% | R@5 ${pct(nonAbs.r5, nonAbs.n)}% | R@10 ${pct(nonAbs.r10, nonAbs.n)}% | RecallAll@5 ${pct(nonAbs.all5, nonAbs.n)}% | NDCG@10 ${(nonAbs.ndcg10 / nonAbs.n).toFixed(4)}`);
    console.log(`Latency per question (index ${(sessionsSeen / all.n).toFixed(0)} sessions + rank, incl. JSON): avg ${(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(2)} ms | p50 ${pctile(lat, 0.5).toFixed(2)} | p95 ${pctile(lat, 0.95).toFixed(2)}`);
    for (const [t, s] of Object.entries(buckets)) {
      console.log(`  ${t.padEnd(26)} R@1 ${pct(s.r1, s.n).padStart(6)}% | R@5 ${pct(s.r5, s.n).padStart(6)}% | R@10 ${pct(s.r10, s.n).padStart(6)}% | NDCG@10 ${(s.ndcg10 / s.n).toFixed(3)} (${s.n})`);
    }
  }
  return { all, nonAbs, buckets };
}

let LOCOMO_DATA = null;
function runLoCoMo(params, quiet = false) {
  if (!LOCOMO_DATA) LOCOMO_DATA = JSON.parse(readFileSync('benchmarks/data/locomo10.json', 'utf8'));
  const names = { 1: 'Multi-Hop', 2: 'Temporal', 3: 'Open-Domain', 4: 'Single-Hop', 5: 'Adversarial' };
  const blank = () => ({ n: 0, noEvidence: 0, sR1: 0, sR5: 0, tR1: 0, tR5: 0, tR10: 0, softR1: 0, softR5: 0, fracR5: 0 });
  const cats = {}; const core = blank(); const total = blank();
  const lat = [];
  let qCount = 0;
  for (const conv of LOCOMO_DATA) {
    const { sessions, queries } = ingestLoCoMo(conv);
    const t0 = performance.now();
    const res = rank({ sessions, queries, top_k: 10, params });
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
    console.log('\n--- LoCoMo-10 (locomo10.json) | ranking 100% in Rust WASM ---');
    const line = (label, a) => console.log(`${label.padEnd(28)} Session [R@1: ${pct(a.sR1, a.n)}%, R@5: ${pct(a.sR5, a.n)}%] | Turn Exact [R@1: ${pct(a.tR1, a.n)}%, R@5: ${pct(a.tR5, a.n)}%, R@10: ${pct(a.tR10, a.n)}%] | Soft [R@1: ${pct(a.softR1, a.n)}%, R@5: ${pct(a.softR5, a.n)}%] | Frac R@5: ${(a.fracR5 / a.n * 100).toFixed(2)}% (${a.n} scored${a.noEvidence ? `, ${a.noEvidence} without evidence excluded` : ''})`);
    line('CORE (Cat 1-4)', core);
    line('ALL (Cat 1-5)', total);
    for (const c of [1, 2, 3, 4, 5]) if (cats[c]) line(`  Cat ${c} ${names[c]}`, cats[c]);
    console.log(`Latency per question (batched per conversation, incl. index + JSON): avg ${(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(3)} ms`);
  }
  return { core, total, cats };
}

// ---------------------------------------------------------------------------
console.log('================================================================================');
console.log('  STARA TOTAL-RECALL ENGINE · Stage 1 zero-model retrieval · Rust WASM kernel');
console.log('================================================================================');
console.log(`Static audit: ingestion functions free of ${staticAudit()} label fields  ✔`);
console.log(`Boundary self-test: Rust rejected ${boundarySelfTest()}/4 labelled payloads  ✔\n`);

const params = paramsFromArgs();
const out = { params };
if (args.only !== 'locomo') out.lme = runLongMemEval(params);
if (args.only !== 'lme') out.locomo = runLoCoMo(params);

if (args.ablations) {
  const toggles = ['enable_stemming', 'enable_proximity', 'enable_coverage', 'enable_roles', 'enable_preference', 'enable_hypernyms', 'enable_recency', 'enable_window', 'enable_carry', 'enable_deictic', 'enable_funnel', 'enable_dyad', 'enable_bridge'];
  console.log('\n--- ABLATIONS (each weapon switched off alone) ---');
  console.log('weapon off           | LME R@1  LME R@5 | LoCoMo core Sess R@5  Turn R@5');
  out.ablations = {};
  const fmt = (lme, lo) => `${pct(lme.all.r1, lme.all.n).padStart(6)}%  ${pct(lme.all.r5, lme.all.n).padStart(6)}% | ${pct(lo.core.sR5, lo.core.n).padStart(18)}%  ${pct(lo.core.tR5, lo.core.n).padStart(7)}%`;
  console.log(`${'(none)'.padEnd(20)} | ${fmt(out.lme, out.locomo)}`);
  for (const t of toggles) {
    const p = { ...params, [t]: false };
    const lme = runLongMemEval(p, true), lo = runLoCoMo(p, true);
    out.ablations[t] = { lme, locomo: lo };
    console.log(`${t.replace('enable_', '').padEnd(20)} | ${fmt(lme, lo)}`);
  }
}

if (args.json) writeFileSync(args.json, JSON.stringify(out, null, 2));
console.log('\n================================================================================');
console.log(`  COMPLETE · ZERO LLM · ZERO VECTORS · RSS ${(process.memoryUsage().rss / 1048576).toFixed(0)} MB`);
console.log('================================================================================');
