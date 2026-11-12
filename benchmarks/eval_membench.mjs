// MemBench recall (Tan et al., Findings of ACL 2025, arXiv:2506.21605).
// Ranking is the same release kernel as the other suites: retrieval_bench --worker.
// Gold (QA.answer, choices, ground_truth, target_step_id) is read only after ranking.
// The reported retrieval number is evidence-step Recall@10: fraction of gold
// exchange steps (QA.target_step_id) whose user (`:u`) or assistant (`:a`)
// turn is present in the top 10. Each step counts at most once.

import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve, join } from 'node:path';

const rootDir = resolve('.');
const binPath = [
  join(rootDir, 'target', 'release', 'retrieval_bench.exe'),
  join(rootDir, 'target', 'release', 'retrieval_bench'),
].find(p => existsSync(p));
if (!binPath) {
  console.error('ERROR: release retrieval_bench not found.');
  process.exit(1);
}

let child = null;
const queue = [];
function ensureWorker() {
  if (child) return;
  child = spawn(binPath, ['--worker'], { stdio: ['pipe', 'pipe', 'inherit'] });
  const rl = createInterface({ input: child.stdout, crlfDelay: Infinity });
  rl.on('line', (line) => {
    if (queue.length === 0) return;
    const pending = queue.shift();
    try {
      const data = JSON.parse(line);
      if (data && data.error) pending.reject(new Error(data.error));
      else pending.resolve(data);
    } catch (err) {
      pending.reject(err);
    }
  });
}
function rank(payload) {
  ensureWorker();
  return new Promise((resolve, reject) => {
    queue.push({ resolve, reject });
    child.stdin.write(JSON.stringify(payload) + '\n');
  });
}

function sessionOffset(lengths, session) {
  let off = 0;
  for (let i = 0; i < session; i++) off += lengths[i];
  return off;
}

function sessionOf(lengths, flat) {
  let off = 0;
  for (let i = 0; i < lengths.length; i++) {
    if (flat < off + lengths[i]) return i;
    off += lengths[i];
  }
  return -1;
}

// Raw pairs are either [localTurn, session] or [flatTurn, session].
function evidenceFlatIndex(spec, lengths) {
  if (!Array.isArray(spec)) return Number.isInteger(spec) ? spec : null;
  const [a, b] = spec;
  if (Number.isInteger(b) && b >= 0 && b < lengths.length && Number.isInteger(a) && a >= 0 && a < lengths[b]) {
    return sessionOffset(lengths, b) + a;
  }
  if (Number.isInteger(a) && a >= 0 && sessionOf(lengths, a) === b) return a;
  return null;
}

function asSessions(messageList) {
  if (!Array.isArray(messageList) || messageList.length === 0) return [];
  if (Array.isArray(messageList[0])) return messageList;
  return [messageList];
}

function turnText(turn) {
  if (typeof turn === 'string') return { role: 'user', content: turn };
  const user = turn.user || turn.user_message || turn.message || '';
  const assistant = turn.assistant || turn.assistant_message || '';
  const place = turn.place ? ` (${turn.place})` : '';
  return {
    user: user ? `${user}${place}` : '',
    assistant: assistant ? `${assistant}${place}` : '',
    time: typeof turn.time === 'string' ? turn.time.replace(/^'|'$/g, '').split(' ').slice(0, 2).join(' ') : '',
  };
}

function buildPayload(traj) {
  const groups = asSessions(traj.message_list);
  const lengths = [];
  const sessions = [];
  // One entry per exchange step, aligned with evidenceFlatIndex / lengths.
  // turnIds lists the ingested :u and/or :a turns for that step. Gold labels
  // stay here and are never copied onto sessions or turns.
  const flatSteps = [];
  groups.forEach((group, si) => {
    const turns = [];
    let date = '';
    group.forEach((raw, mi) => {
      const parsed = turnText(raw);
      if (!date && parsed.time) date = parsed.time;
      const stepKey = `${traj.tid}:${si}:${mi}`;
      const turnIds = [];
      if (parsed.user) {
        const uId = `${stepKey}:u`;
        turns.push({ id: uId, role: 'user', content: parsed.user });
        turnIds.push(uId);
      }
      if (parsed.assistant) {
        const aId = `${stepKey}:a`;
        turns.push({ id: aId, role: 'assistant', content: parsed.assistant });
        turnIds.push(aId);
      }
      if (!parsed.user && !parsed.assistant && parsed.content) {
        const uId = `${stepKey}:u`;
        turns.push({ id: uId, role: parsed.role || 'user', content: parsed.content });
        turnIds.push(uId);
      }
      flatSteps.push({ stepKey, turnIds });
    });
    lengths.push(group.length);
    sessions.push({
      id: `${traj.tid}:s${si}`,
      date,
      timestamp: 0,
      turns,
    });
  });
  const qa = traj.QA || {};
  const gold = [];
  for (const spec of qa.target_step_id || []) {
    const flat = evidenceFlatIndex(spec, lengths);
    if (flat == null || flat < 0 || flat >= flatSteps.length) continue;
    const step = flatSteps[flat];
    if (!step || step.turnIds.length === 0) continue;
    gold.push(step.stepKey);
  }
  return {
    sessions,
    query: { text: String(qa.question || ''), date: '' },
    gold: [...new Set(gold)],
    flatSteps,
  };
}

function pct(n, d) {
  return d ? Number(((n / d) * 100).toFixed(2)) : 0;
}

async function scoreFile(file, scenario) {
  const data = JSON.parse(readFileSync(file, 'utf8'));
  const categories = Object.entries(data);
  const stats = { questions: 0, evidence: 0, hit10: 0, any10: 0, all10: 0, byCategory: {} };
  for (const [category, rows] of categories) {
    if (!Array.isArray(rows)) continue;
    const cat = { questions: 0, evidence: 0, hit10: 0, any10: 0, all10: 0 };
    for (const traj of rows) {
      const built = buildPayload(traj);
      if (!built.query.text || built.gold.length === 0 || built.sessions.every(s => s.turns.length === 0)) continue;
      const resp = await rank({ sessions: built.sessions, queries: [built.query], top_k: 10 });
      const got = new Set((resp.results?.[0]?.turns || []).slice(0, 10).map(t => t.id));
      const turnToStep = new Map();
      for (const step of built.flatSteps) {
        for (const id of step.turnIds) turnToStep.set(id, step.stepKey);
      }
      const gotSteps = new Set();
      for (const id of got) {
        const stepKey = turnToStep.get(id);
        // tid:si:mi:u and tid:si:mi:a both credit step tid:si:mi.
        if (stepKey) gotSteps.add(stepKey);
      }
      // filter is a subset of gold, so hits <= built.gold.length even when
      // both the :u and :a turns of one step are retrieved.
      const hits = built.gold.filter(stepKey => gotSteps.has(stepKey)).length;
      cat.questions++;
      cat.evidence += built.gold.length;
      cat.hit10 += hits;
      cat.any10 += hits > 0 ? 1 : 0;
      cat.all10 += hits === built.gold.length ? 1 : 0;
      if (cat.questions % 50 === 0) {
        console.log(`  ${scenario}/${category} ${cat.questions}/${rows.length} evidence R@10 ${pct(cat.hit10, cat.evidence)}%`);
      }
    }
    stats.questions += cat.questions;
    stats.evidence += cat.evidence;
    stats.hit10 += cat.hit10;
    stats.any10 += cat.any10;
    stats.all10 += cat.all10;
    stats.byCategory[category] = {
      questions: cat.questions,
      evidence_recall_at_10: pct(cat.hit10, cat.evidence),
      any_at_10: pct(cat.any10, cat.questions),
      all_at_10: pct(cat.all10, cat.questions),
    };
    console.log(`${scenario}/${category}: n=${cat.questions} evidence R@10=${pct(cat.hit10, cat.evidence)}% Any@10=${pct(cat.any10, cat.questions)}% All@10=${pct(cat.all10, cat.questions)}%`);
  }
  return stats;
}

async function main() {
  const dataRoot = join(rootDir, 'benchmarks', 'data', 'membench');
  const sides = [
    ['participation', join(dataRoot, 'FirstAgent')],
    ['observation', join(dataRoot, 'ThirdAgent')],
  ];
  const out = { benchmark: 'MemBench', kernel: 'retrieval_bench', cost_usd: 0, scenarios: {} };
  for (const [name, dir] of sides) {
    if (!existsSync(dir)) continue;
    const files = readdirSync(dir).filter(f => f.endsWith('.json')).sort();
    const scenario = { questions: 0, evidence: 0, hit10: 0, any10: 0, all10: 0, files: {} };
    for (const file of files) {
      console.log(`\n=== ${name}/${file} ===`);
      const stats = await scoreFile(join(dir, file), name);
      scenario.questions += stats.questions;
      scenario.evidence += stats.evidence;
      scenario.hit10 += stats.hit10;
      scenario.any10 += stats.any10;
      scenario.all10 += stats.all10;
      scenario.files[file] = stats.byCategory;
    }
    out.scenarios[name] = {
      questions: scenario.questions,
      evidence_recall_at_10: pct(scenario.hit10, scenario.evidence),
      any_at_10: pct(scenario.any10, scenario.questions),
      all_at_10: pct(scenario.all10, scenario.questions),
      files: scenario.files,
    };
    console.log(`\n${name} TOTAL n=${scenario.questions} evidence R@10=${out.scenarios[name].evidence_recall_at_10}%`);
  }
  const destDir = join(rootDir, 'benchmarks', 'results', 'v6.2.2-measured');
  mkdirSync(destDir, { recursive: true });
  const dest = join(destDir, 'membench_recall.json');
  writeFileSync(dest, JSON.stringify(out, null, 2));
  console.log(`\nWrote ${dest}`);
  if (child) child.kill();
}

main().catch((err) => {
  console.error(err);
  if (child) child.kill();
  process.exit(1);
});
