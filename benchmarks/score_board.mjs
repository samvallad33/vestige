// Offline board + stage funnel for one retrieval_bench binary.
// Gold labels are read only after rank() returns.
// LoCoMo dialogues 6-10 are not loaded unless FREEZE=1.
//
// Stage fields (written by retrieval.rs when VESTIGE_REPLAY_DUMP is set):
//   stage_turn_pool50_ids       top 50 turns by t_scene (>0) before submodular selection.
//                               order() tie-break: higher score, newer session timestamp, lower index.
//   stage_turn_pre_kemeny_ids   final_turn_indices cloned immediately BEFORE
//                               rank1_gatekeeper_keep_rsb_head. Length is the selected
//                               window (~top_k), not 50. This is not `rank_pre`.
//   stage_turn_post_kemeny_ids  the same window AFTER the gatekeeper. This is the
//                               returned turn list.
//   stage_session_pool50_ids    top 50 sessions by s_score+bridge_boost (>0) before
//                               the session selector.
//   stage_session_final_ids     the returned session list. The gatekeeper does not
//                               reorder sessions, so session pre-Kemeny and
//                               post-Kemeny are this list.
// `rank_pre` inside candidates[] is the index in a list that already starts with
// the final seated turns. Do not use it as a stage.
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { readFileSync, writeFileSync, mkdirSync, openSync, readSync, fstatSync, closeSync, unlinkSync, appendFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const bin = resolve(root, 'target/release/retrieval_bench');
const outDir = resolve(root, 'benchmarks/results/sanitize-baseline');
mkdirSync(outDir, { recursive: true });

const seed = process.env.RUST_HASH_SEED || '1';
const tag = process.env.BOARD_TAG || `seed${seed}`;
const freeze = process.env.FREEZE === '1';
const dumpPath = `/tmp/replay-${tag}.jsonl`;
writeFileSync(dumpPath, '');

const child = spawn(bin, ['--worker'], {
  stdio: ['pipe', 'pipe', 'inherit'],
  env: { ...process.env, RUST_HASH_SEED: seed, VESTIGE_REPLAY_DUMP: dumpPath },
});
const rl = createInterface({ input: child.stdout, crlfDelay: Infinity });
const queue = [];
rl.on('line', (line) => {
  if (!queue.length) return;
  const pending = queue.shift();
  try {
    const data = JSON.parse(line);
    if (data && data.error) pending.reject(new Error(data.error));
    else pending.resolve(data);
  } catch (e) { pending.reject(e); }
});
function rank(payload) {
  return new Promise((resolve, reject) => {
    queue.push({ resolve, reject });
    child.stdin.write(JSON.stringify(payload) + '\n');
  });
}

const dumpFd = openSync(dumpPath, 'r');
let dumpOffset = 0;
let dumpPending = '';
function readDumpLines(expect) {
  const size = fstatSync(dumpFd).size;
  if (size > dumpOffset) {
    const buf = Buffer.alloc(size - dumpOffset);
    readSync(dumpFd, buf, 0, buf.length, dumpOffset);
    dumpOffset = size;
    const text = dumpPending + buf.toString('utf8');
    const parts = text.split('\n');
    dumpPending = parts.pop() ?? '';
    const lines = parts.filter((l) => l.length > 0).map((l) => JSON.parse(l));
    if (lines.length !== expect) {
      throw new Error(`dump lines ${lines.length} != queries ${expect} (pending ${dumpPending.length})`);
    }
    return lines;
  }
  if (expect === 0) return [];
  throw new Error(`dump produced no bytes for ${expect} queries`);
}

const pct = (n, d) => (d ? Math.round((10000 * n) / d) / 100 : null);
const log = (s) => { appendFileSync('/tmp/score_board.progress', s + '\n'); console.log(s); };
const rankings = [];
const funnelRows = [];

function anyHit(gold, ids, n) {
  const slice = n == null ? ids : ids.slice(0, n);
  const set = new Set(slice);
  return gold.some((g) => set.has(g));
}
function allHit(gold, ids, n) {
  if (!gold.length) return false;
  const slice = n == null ? ids : ids.slice(0, n);
  const set = new Set(slice);
  return gold.every((g) => set.has(g));
}

function stageOf(dump, kind) {
  if (kind === 'turn') {
    return {
      pool: dump.stage_turn_pool50_ids || [],
      pre: dump.stage_turn_pre_kemeny_ids || [],
      post: dump.stage_turn_post_kemeny_ids || [],
    };
  }
  const finalIds = dump.stage_session_final_ids || [];
  return {
    pool: dump.stage_session_pool50_ids || [],
    // No session Kemeny. pre and post are the returned session list.
    pre: finalIds,
    post: finalIds,
  };
}

function recordFunnel(meta, gold, indexed, dump, returned, kind) {
  const st = stageOf(dump, kind);
  const indexedGold = gold.filter((g) => indexed.has(g));
  const row = {
    ...meta,
    kind,
    n_gold: gold.length,
    corpus_any: indexedGold.length > 0,
    corpus_all: gold.length > 0 && indexedGold.length === gold.length,
    pool50: anyHit(gold, st.pool, 50),
    pre5: anyHit(gold, st.pre, 5),
    post5: anyHit(gold, st.post, 5),
    at1: anyHit(gold, st.post, 1),
    pool_len: st.pool.length,
    pre_len: st.pre.length,
    post_len: st.post.length,
    returned_len: returned.length,
    post_matches_returned: st.post.join('\n') === returned.join('\n'),
    // Non-nested: a later pass can seat a turn that was outside pool@50.
    pre_outside_pool: anyHit(gold, st.pre, 5) && !anyHit(gold, st.pool, 50),
    at1_outside_pool: anyHit(gold, st.post, 1) && !anyHit(gold, st.pool, 50),
  };
  funnelRows.push(row);
  return row;
}

function summarize(name, rows) {
  const n = rows.length;
  const count = (k) => rows.filter((r) => r[k]).length;
  const keys = ['corpus_any', 'corpus_all', 'pool50', 'pre5', 'post5', 'at1', 'pre_outside_pool', 'at1_outside_pool', 'post_matches_returned'];
  const hits = Object.fromEntries(keys.map((k) => [k, count(k)]));
  const lens = rows.map((r) => r.returned_len);
  const sum = lens.reduce((a, b) => a + b, 0);
  return {
    slice: name,
    n,
    top_k: 10,
    returned_len_min: lens.length ? Math.min(...lens) : null,
    returned_len_mean: lens.length ? Math.round((sum / lens.length) * 100) / 100 : null,
    returned_len_max: lens.length ? Math.max(...lens) : null,
    hits,
    rates: Object.fromEntries(keys.map((k) => [k, pct(hits[k], n)])),
    // Five lines. Any-hit of the gold id set. corpus_any is "at least one gold id was indexed".
    funnel: {
      gold_in_corpus: hits.corpus_any,
      gold_in_pool_at_50: hits.pool50,
      gold_at_pre_kemeny_at_5: hits.pre5,
      gold_at_post_kemeny_at_5: hits.post5,
      gold_at_final_at_1: hits.at1,
    },
  };
}

function boardFrom(rows, goldSessKey, goldTurnKey) {
  const scored = rows.filter((r) => !r.exclude);
  const n = scored.length;
  const hitS = (k, pred) => scored.filter((r) => pred(r.goldSess, r.retSess, k)).length;
  const hitT = (k, pred) => scored.filter((r) => pred(r.goldTurns, r.retTurns, k)).length;
  return {
    n,
    excluded: rows.length - n,
    session_any_at_1: hitS(1, anyHit),
    session_any_at_5: hitS(5, anyHit),
    session_any_at_10: hitS(10, anyHit),
    session_all_at_5: scored.filter((r) => allHit(r.goldSess, r.retSess, 5)).length,
    turn_any_at_1: hitT(1, anyHit),
    turn_any_at_5: hitT(5, anyHit),
    turn_any_at_10: hitT(10, anyHit),
    rates: {
      session_any_at_1: pct(hitS(1, anyHit), n),
      session_any_at_5: pct(hitS(5, anyHit), n),
      session_any_at_10: pct(hitS(10, anyHit), n),
      session_all_at_5: pct(scored.filter((r) => allHit(r.goldSess, r.retSess, 5)).length, n),
      turn_any_at_1: pct(hitT(1, anyHit), n),
      turn_any_at_5: pct(hitT(5, anyHit), n),
      turn_any_at_10: pct(hitT(10, anyHit), n),
    },
  };
}

async function runLme() {
  const data = JSON.parse(readFileSync(resolve(root, 'benchmarks/data/longmemeval_s_cleaned.json'), 'utf8'));
  const oracle = JSON.parse(readFileSync(resolve(root, 'benchmarks/data/longmemeval_oracle.json'), 'utf8'));
  const oBy = new Map(oracle.map((x) => [x.question_id, x]));
  const rows = [];
  const t0 = Date.now();
  let i = 0;
  for (const item of data) {
    const sessions = item.haystack_sessions.map((sess, si) => ({
      id: String(item.haystack_session_ids[si]),
      date: String(item.haystack_dates[si] || ''),
      turns: sess.map((t, j) => ({
        id: `${item.haystack_session_ids[si]}#${j}`,
        role: String(t.role),
        content: String(t.content),
      })),
    }));
    const res = await rank({
      sessions,
      queries: [{ text: String(item.question), date: String(item.question_date || '') }],
      top_k: 10,
    });
    const dumps = readDumpLines(1);
    const r0 = res.results[0];
    const retSess = r0.sessions.map((s) => s.id);
    const retTurns = r0.turns.map((t) => t.id);
    const goldSess = [...new Set(item.answer_session_ids || [])];
    const o = oBy.get(item.question_id);
    const goldTurns = [];
    if (o) {
      const omap = new Map(o.haystack_session_ids.map((sid, si) => [sid, o.haystack_sessions[si]]));
      for (let si = 0; si < item.haystack_session_ids.length; si++) {
        const sid = item.haystack_session_ids[si];
        const osess = omap.get(sid);
        if (!osess) continue;
        osess.forEach((ot, j) => {
          if (ot.has_answer) goldTurns.push(`${sid}#${j}`);
        });
      }
    }
    const indexedSess = new Set(sessions.map((s) => s.id));
    const indexedTurns = new Set(sessions.flatMap((s) => s.turns.map((t) => t.id)));
    const abs = String(item.question_id).endsWith('_abs');
    const meta = { bench: 'lme', id: item.question_id, type: item.question_type, exclude: abs };
    if (!abs) {
      recordFunnel(meta, goldSess, indexedSess, dumps[0], retSess, 'session');
      recordFunnel(meta, goldTurns, indexedTurns, dumps[0], retTurns, 'turn');
    }
    rows.push({ ...meta, goldSess, goldTurns, retSess, retTurns });
    rankings.push({ bench: 'lme', id: item.question_id, sessions: retSess, turns: retTurns });
    i += 1;
    if (i % 100 === 0) log(`LME ${i}/${data.length}`);
  }
  const board = boardFrom(rows);
  const non = rows.filter((r) => !r.exclude);
  const byType = {};
  for (const r of non) {
    const b = byType[r.type] ??= { n: 0, s1: 0, s5: 0, s10: 0, all5: 0, t1: 0, t5: 0, t10: 0 };
    b.n += 1;
    if (anyHit(r.goldSess, r.retSess, 1)) b.s1 += 1;
    if (anyHit(r.goldSess, r.retSess, 5)) b.s5 += 1;
    if (anyHit(r.goldSess, r.retSess, 10)) b.s10 += 1;
    if (allHit(r.goldSess, r.retSess, 5)) b.all5 += 1;
    if (anyHit(r.goldTurns, r.retTurns, 1)) b.t1 += 1;
    if (anyHit(r.goldTurns, r.retTurns, 5)) b.t5 += 1;
    if (anyHit(r.goldTurns, r.retTurns, 10)) b.t10 += 1;
  }
  board.by_type = byType;
  board.seconds = (Date.now() - t0) / 1000;
  board.n_all = rows.length;
  return board;
}

function locomoGold(qa) {
  const goldTurns = [...new Set((qa.evidence || []).flatMap((e) => String(e).split(/[;,\s]+/)).filter((e) => /^D\d+:\d+$/.test(e)))];
  const goldSess = [...new Set(goldTurns.map((e) => e.split(':')[0]))];
  return { goldTurns, goldSess };
}

async function runLocomo() {
  const data = JSON.parse(readFileSync(resolve(root, 'benchmarks/data/locomo10.json'), 'utf8'));
  const slice = freeze ? data : data.slice(0, 5);
  const names = { 1: 'Multi-Hop', 2: 'Temporal', 3: 'Open-Domain', 4: 'Single-Hop', 5: 'Adversarial' };
  const rows = [];
  const t0 = Date.now();
  for (let ci = 0; ci < slice.length; ci++) {
    const conv = slice[ci];
    const dialogue = freeze ? ci + 1 : ci + 1;
    const c = conv.conversation;
    const sessions = [];
    for (let s = 1; c[`session_${s}`]; s++) {
      sessions.push({
        id: `D${s}`,
        date: String(c[`session_${s}_date_time`] || ''),
        turns: c[`session_${s}`].filter((t) => t.text && t.dia_id).map((t) => {
          const blip = t.blip_caption ? `\n[BLIP caption] ${t.blip_caption}` : '';
          return { id: String(t.dia_id), role: String(t.speaker || ''), content: `${t.text}${blip}` };
        }),
      });
    }
    const res = await rank({
      sessions,
      queries: conv.qa.map((q) => ({ text: String(q.question), date: '' })),
      top_k: 10,
    });
    const dumps = readDumpLines(conv.qa.length);
    const indexedTurns = new Set(sessions.flatMap((s) => s.turns.map((t) => t.id)));
    const indexedSess = new Set(sessions.map((s) => s.id));
    res.results.forEach((r, qi) => {
      const qa = conv.qa[qi];
      const { goldTurns, goldSess } = locomoGold(qa);
      const retSess = r.sessions.map((s) => s.id);
      const retTurns = r.turns.map((t) => t.id);
      const soft = new Set();
      for (const g of goldTurns) {
        soft.add(g);
        const [sessId, tNumStr] = g.split(':');
        const tNum = parseInt(tNumStr, 10);
        if (!Number.isNaN(tNum)) {
          soft.add(`${sessId}:${tNum - 1}`);
          soft.add(`${sessId}:${tNum + 1}`);
        }
      }
      const noEvidence = goldTurns.length === 0;
      const meta = {
        bench: 'locomo',
        id: `${conv.sample_id}#${qi}`,
        dialogue,
        sample: conv.sample_id,
        cat: qa.category,
        catName: names[qa.category] || String(qa.category),
        exclude: noEvidence,
      };
      if (!noEvidence) {
        recordFunnel(meta, goldSess, indexedSess, dumps[qi], retSess, 'session');
        recordFunnel(meta, goldTurns, indexedTurns, dumps[qi], retTurns, 'turn');
      }
      rows.push({
        ...meta,
        goldTurns,
        goldSess,
        retSess,
        retTurns,
        soft5: retTurns.slice(0, 5).some((id) => soft.has(id)),
      });
      rankings.push({ bench: 'locomo', id: meta.id, sessions: retSess, turns: retTurns, tune: !freeze || dialogue <= 5 });
    });
    log(`LoCoMo dialogue ${dialogue} ${conv.sample_id}`);
  }
  function pack(list) {
    const scored = list.filter((r) => !r.exclude);
    const n = scored.length;
    const hit = (pred) => scored.filter(pred).length;
    const m = {
      n_scored: n,
      no_evidence_excluded: list.length - n,
      session_r1: hit((r) => anyHit(r.goldSess, r.retSess, 1)),
      session_r5: hit((r) => anyHit(r.goldSess, r.retSess, 5)),
      turn_exact_r1: hit((r) => anyHit(r.goldTurns, r.retTurns, 1)),
      turn_exact_r5: hit((r) => anyHit(r.goldTurns, r.retTurns, 5)),
      turn_exact_r10: hit((r) => anyHit(r.goldTurns, r.retTurns, 10)),
      turn_soft_r5: hit((r) => r.soft5),
    };
    m.rates = {
      session_r1: pct(m.session_r1, n),
      session_r5: pct(m.session_r5, n),
      turn_exact_r1: pct(m.turn_exact_r1, n),
      turn_exact_r5: pct(m.turn_exact_r5, n),
      turn_exact_r10: pct(m.turn_exact_r10, n),
      turn_soft_r5: pct(m.turn_soft_r5, n),
    };
    return m;
  }
  const cats = {};
  for (const name of Object.values(names)) cats[name] = pack(rows.filter((r) => r.catName === name));
  return {
    seconds: (Date.now() - t0) / 1000,
    split: freeze ? 'all-10' : 'tune-1-5',
    all: pack(rows),
    core: pack(rows.filter((r) => r.cat !== 5)),
    categories: cats,
  };
}

async function runBeam() {
  const { readdirSync } = await import('node:fs');
  const { join } = await import('node:path');
  const base = join(root, 'benchmarks/BEAM/chats/100K');
  const convDirs = readdirSync(base).filter((d) => {
    try { return readdirSync(join(base, d)).includes('chat.json'); } catch { return false; }
  }).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
  const rows = [];
  const t0 = Date.now();
  let totalQ = 0;
  for (const cId of convDirs) {
    const chatData = JSON.parse(readFileSync(join(base, cId, 'chat.json'), 'utf8'));
    const qData = JSON.parse(readFileSync(join(base, cId, 'probing_questions/probing_questions.json'), 'utf8'));
    const sessions = [];
    const turnIdToSessionId = new Map();
    for (let bi = 0; bi < chatData.length; bi++) {
      const batch = chatData[bi];
      const batchAnchor = batch.time_anchor || batch.turns?.[0]?.[0]?.time_anchor || '';
      for (let subIdx = 0; subIdx < batch.turns.length; subIdx++) {
        const sub = batch.turns[subIdx];
        if (!Array.isArray(sub) || sub.length === 0) continue;
        const sId = `conv${cId}_b${bi}_s${subIdx}`;
        const turns = sub.map((t) => {
          const tid = String(t.id);
          turnIdToSessionId.set(tid, sId);
          return { id: tid, role: t.role || 'user', content: t.content || '' };
        });
        sessions.push({ id: sId, date: sub[0]?.time_anchor || batchAnchor || '', timestamp: 0, turns });
      }
    }
    const entries = [];
    for (const [cat, qList] of Object.entries(qData)) {
      if (!Array.isArray(qList)) continue;
      for (const qItem of qList) {
        totalQ += 1;
        const goldTurnIds = [...new Set((function walk(raw) {
          if (!raw) return [];
          if (Array.isArray(raw)) return raw.flatMap(walk);
          if (typeof raw === 'object') return Object.values(raw).flatMap(walk);
          return [String(raw)];
        })(qItem.source_chat_ids))];
        if (goldTurnIds.length === 0) continue;
        const goldSessIds = [...new Set(goldTurnIds.map((id) => turnIdToSessionId.get(id)).filter(Boolean))];
        entries.push({ cat, question: qItem.question, goldTurnIds, goldSessIds });
      }
    }
    if (!entries.length) continue;
    const resp = await rank({
      sessions,
      queries: entries.map((e) => ({ text: e.question, date: '' })),
      top_k: 10,
    });
    const dumps = readDumpLines(entries.length);
    const indexedTurns = new Set(sessions.flatMap((s) => s.turns.map((t) => t.id)));
    const indexedSess = new Set(sessions.map((s) => s.id));
    resp.results.forEach((r, qi) => {
      const e = entries[qi];
      const retSess = (r.sessions || []).map((s) => s.id);
      const retTurns = (r.turns || []).map((t) => t.id);
      const meta = { bench: 'beam', id: `${cId}#${qi}`, cat: e.cat, exclude: false };
      recordFunnel(meta, e.goldSessIds, indexedSess, dumps[qi], retSess, 'session');
      recordFunnel(meta, e.goldTurnIds, indexedTurns, dumps[qi], retTurns, 'turn');
      rows.push({ ...meta, goldSess: e.goldSessIds, goldTurns: e.goldTurnIds, retSess, retTurns });
      rankings.push({ bench: 'beam', id: meta.id, sessions: retSess, turns: retTurns });
    });
    log(`BEAM conv ${cId}`);
  }
  const board = boardFrom(rows);
  board.conversations = convDirs.length;
  board.total_questions = totalQ;
  board.seconds = (Date.now() - t0) / 1000;
  return board;
}

function compareSeed1() {
  const priorPath = resolve(outDir, 'rankings-seed1.jsonl');
  const prior = readFileSync(priorPath, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  const by = new Map(prior.map((r) => [`${r.bench}:${r.id}`, r]));
  let compared = 0;
  let differ = 0;
  const byBench = {};
  for (const r of rankings) {
    const p = by.get(`${r.bench}:${r.id}`);
    if (!p) continue;
    compared += 1;
    const same = JSON.stringify(p.sessions) === JSON.stringify(r.sessions) && JSON.stringify(p.turns) === JSON.stringify(r.turns);
    if (!same) {
      differ += 1;
      byBench[r.bench] = (byBench[r.bench] || 0) + 1;
    }
  }
  return { compared, differ, byBench };
}

const report = {
  tag,
  seed,
  freeze,
  sha: '57acdc468a352e9be12a7ae6e1e10901902798ac',
  field_notes: {
    turn_pool50: 'top 50 by t_scene before submodular selection; not the harness top_k window',
    turn_pre_kemeny: 'selected window immediately before rank1_gatekeeper_keep_rsb_head; not rank_pre',
    turn_post_kemeny: 'selected window after the gatekeeper; equals the returned turns',
    session_pool50: 'top 50 by s_score+bridge_boost before the session selector',
    session_pre_and_post: 'identical: the gatekeeper reorders turns only',
    rank_pre: 'index in a list that starts with the final seated turns; not a stage',
  },
};
report.lme = await runLme();
log('LME ' + JSON.stringify(report.lme.rates));
report.locomo = await runLocomo();
log('LOCOMO ' + report.locomo.split + ' ' + JSON.stringify(report.locomo.all.rates));
report.beam = await runBeam();
log('BEAM ' + JSON.stringify(report.beam.rates));

const groups = [
  ['lme-session', (r) => r.bench === 'lme' && r.kind === 'session'],
  ['lme-turn', (r) => r.bench === 'lme' && r.kind === 'turn'],
  ['locomo-tune-session', (r) => r.bench === 'locomo' && r.kind === 'session'],
  ['locomo-tune-turn', (r) => r.bench === 'locomo' && r.kind === 'turn'],
  ['locomo-tune-multihop-session', (r) => r.bench === 'locomo' && r.kind === 'session' && r.catName === 'Multi-Hop'],
  ['locomo-tune-multihop-turn', (r) => r.bench === 'locomo' && r.kind === 'turn' && r.catName === 'Multi-Hop'],
  ['locomo-tune-opendomain-session', (r) => r.bench === 'locomo' && r.kind === 'session' && r.catName === 'Open-Domain'],
  ['locomo-tune-opendomain-turn', (r) => r.bench === 'locomo' && r.kind === 'turn' && r.catName === 'Open-Domain'],
  ['locomo-tune-temporal-turn', (r) => r.bench === 'locomo' && r.kind === 'turn' && r.catName === 'Temporal'],
  ['locomo-tune-singlehop-turn', (r) => r.bench === 'locomo' && r.kind === 'turn' && r.catName === 'Single-Hop'],
  ['locomo-tune-adversarial-turn', (r) => r.bench === 'locomo' && r.kind === 'turn' && r.catName === 'Adversarial'],
  ['beam-session', (r) => r.bench === 'beam' && r.kind === 'session'],
  ['beam-turn', (r) => r.bench === 'beam' && r.kind === 'turn'],
];
report.funnel = Object.fromEntries(groups.map(([name, pred]) => [name, summarize(name, funnelRows.filter(pred))]));
report.example_e47becba = funnelRows.filter((r) => r.id === 'e47becba');
report.vs_seed1 = compareSeed1();
log('VS_SEED1 ' + JSON.stringify(report.vs_seed1));
log('FUNNEL ' + JSON.stringify(report.funnel));

writeFileSync(resolve(outDir, `board-${tag}.json`), JSON.stringify(report, null, 2));
writeFileSync(resolve(outDir, `rankings-${tag}.jsonl`), rankings.map((r) => JSON.stringify(r)).join('\n') + '\n');
writeFileSync(resolve(outDir, `funnel-${tag}.jsonl`), funnelRows.map((r) => JSON.stringify(r)).join('\n') + '\n');
closeSync(dumpFd);
try { unlinkSync(dumpPath); } catch {}
child.stdin.end();
setTimeout(() => process.exit(0), 100);
