// benchmarks/run_recall_suite.mjs
// Unified RECALL-ONLY benchmark runner for Vestige 6.2.2.
//
// Evaluates pure retrieval accuracy on:
//   1. BEAM 100K (20 conversations, 400 queries / 355 non-abstention across 2,127 sub-conversations)
//   2. MemoryAgentBench (Accurate Retrieval: ruler_qa1_197K, ruler_qa2_421K, and 5x longmemeval_s* 400K rows)
//   3. LongMemEval_M (500 questions / 470 non-abstention streaming ~476 sessions per question)
//   4. HotpotQA (optional dev_distractor full 7,405-question exact supporting_facts check)
//
// Gated Rules:
//   - 100% of retrieval math runs in native Rust kernel (`target/release/retrieval_bench --worker`).
//   - Zero embeddings, zero vectors, zero cloud LLM judge calls ($0.00 spend).
//   - Zero label leakage: queries carry only text and dates. Gold IDs are inspected ONLY during scoring.
//
// Usage:
//   node benchmarks/run_recall_suite.mjs --bench=beam [--split=100k] [--limit=N]
//   node benchmarks/run_recall_suite.mjs --bench=memoryagentbench
//   node benchmarks/run_recall_suite.mjs --bench=longmemeval_m [--limit=N]
//   node benchmarks/run_recall_suite.mjs --bench=all

import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve, join } from 'node:path';

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

// Lazily started persistent worker process (so Python streaming sub-runners don't double-allocate).
let child = null;
const queue = [];

function ensureWorker() {
  if (child) return;
  child = spawn(binPath, ['--worker'], { stdio: ['pipe', 'pipe', 'inherit'] });
  const rl = createInterface({ input: child.stdout, crlfDelay: Infinity });
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
  function failQueue(err) {
    while (queue.length > 0) {
      queue.shift().reject(err);
    }
  }
  child.on('error', failQueue);
  child.on('exit', (code, signal) => {
    if (queue.length === 0) return;
    failQueue(new Error(`retrieval_bench exited code=${code} signal=${signal}`));
  });
  child.stdin.on('error', (err) => {
    if (err && err.code === 'EPIPE') {
      failQueue(new Error('retrieval_bench stdin EPIPE (worker closed)'));
      return;
    }
    failQueue(err);
  });
}

function rank(payload) {
  ensureWorker();
  return new Promise((resolve, reject) => {
    if (child.exitCode !== null || child.signalCode) {
      reject(new Error(`retrieval_bench already exited code=${child.exitCode} signal=${child.signalCode}`));
      return;
    }
    queue.push({ resolve, reject });
    child.stdin.write(JSON.stringify(payload) + '\n');
  });
}

function parseArgs() {
  const args = {
    bench: 'beam',
    split: '100k',
    limit: null,
    out: null,
  };
  for (const a of process.argv.slice(2)) {
    if (a.startsWith('--bench=')) args.bench = a.split('=')[1].toLowerCase();
    else if (a.startsWith('--split=')) args.split = a.split('=')[1].toLowerCase();
    else if (a.startsWith('--limit=')) args.limit = parseInt(a.split('=')[1], 10);
    else if (a.startsWith('--out=')) args.out = a.split('=')[1];
  }
  return args;
}

function extractGoldIds(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) {
    return raw.flatMap(extractGoldIds).map(String);
  }
  if (typeof raw === 'object') {
    return Object.values(raw).flatMap(extractGoldIds).map(String);
  }
  return [String(raw)];
}

// First user turn id in a BEAM sub-conversation. Missing role counts as user,
// matching ingestion. Chat structure only — never question or gold fields.
function openingUserTurnId(sub) {
  if (!Array.isArray(sub)) return null;
  for (const t of sub) {
    if (!t || t.id == null || t.id === '') continue;
    const role = String(t.role || 'user').trim().toLowerCase();
    if (role === 'user') return String(t.id);
  }
  return null;
}

// =========================================================================
// 1. BEAM BENCHMARK RUNNER
// =========================================================================
async function runBeam(args) {
  console.log(`\n======================================================`);
  console.log(`=== RUNNING BEAM RECALL-ONLY BENCHMARK (Vestige 6.2.2) ===`);
  console.log(`======================================================`);

  const beamDir = join(rootDir, 'benchmarks', 'BEAM');
  const splitDir = args.split === '100k' ? '100K' : args.split.toUpperCase();
  const chatsBase = join(beamDir, 'chats', splitDir);

  if (!existsSync(chatsBase)) {
    throw new Error(`BEAM chats directory not found: ${chatsBase}`);
  }

  const convDirs = readdirSync(chatsBase)
    .filter(d => existsSync(join(chatsBase, d, 'chat.json')) && existsSync(join(chatsBase, d, 'probing_questions', 'probing_questions.json')))
    .sort((a, b) => parseInt(a, 10) - parseInt(b, 10));

  console.log(`Found ${convDirs.length} conversations in BEAM ${splitDir}.`);

  let totalQuestions = 0;
  let scoredQuestions = 0;
  let totalSubSessions = 0;
  let totalTurns = 0;

  let sessAny1 = 0, sessAny3 = 0, sessAny5 = 0, sessAny10 = 0;
  let sessAll5 = 0, sessAll10 = 0, sessMrrSum = 0;
  let pktSessAny5 = 0, pktSessAll5 = 0;

  let turnAny1 = 0, turnAny3 = 0, turnAny5 = 0, turnAny10 = 0;
  let turnAll5 = 0, turnAll10 = 0, turnMrrSum = 0;
  let pktTurnAny5 = 0, pktTurnAll5 = 0;
  let gtFired = 0, gtAmbiguous = 0;

  const categoryStats = {};
  const t0 = Date.now();
  const convsToRun = args.limit ? convDirs.slice(0, args.limit) : convDirs;

  for (let ci = 0; ci < convsToRun.length; ci++) {
    const cId = convsToRun[ci];
    const chatData = JSON.parse(readFileSync(join(chatsBase, cId, 'chat.json'), 'utf8'));
    const qData = JSON.parse(readFileSync(join(chatsBase, cId, 'probing_questions', 'probing_questions.json'), 'utf8'));

    // Each batch.turns[subIdx] is one Vestige session (BEAM's turn_chunk unit).
    // Later subs in the same batch point their opening turn at the previous
    // sub's opening user turn. Targets are chat turn ids only — qData is not read here.
    const sessions = [];
    const turnIdToSessionId = new Map();
    for (let bi = 0; bi < chatData.length; bi++) {
      const batch = chatData[bi];
      const batchAnchor = batch.time_anchor || batch.turns?.[0]?.[0]?.time_anchor || '';
      for (let subIdx = 0; subIdx < batch.turns.length; subIdx++) {
        const sub = batch.turns[subIdx];
        if (!Array.isArray(sub) || sub.length === 0) continue;
        const sId = `conv${cId}_b${bi}_s${subIdx}`;
        const turns = [];
        for (let ti = 0; ti < sub.length; ti++) {
          const t = sub[ti];
          const tid = String(t.id);
          turns.push({
            id: tid,
            role: t.role || 'user',
            content: t.content || '',
          });
          turnIdToSessionId.set(tid, sId);
          totalTurns++;
        }
        sessions.push({
          id: sId,
          date: sub[0]?.time_anchor || batchAnchor || '',
          timestamp: 0,
          turns,
        });
        totalSubSessions++;
      }
    }

    const queryEntries = [];
    for (const [cat, qList] of Object.entries(qData)) {
      if (!Array.isArray(qList)) continue;
      for (const qItem of qList) {
        totalQuestions++;
        const goldTurnIds = [...new Set(extractGoldIds(qItem.source_chat_ids))];
        if (goldTurnIds.length === 0) continue;
        const goldSessIds = [...new Set(goldTurnIds.map(id => turnIdToSessionId.get(id)).filter(Boolean))];
        queryEntries.push({ cat, question: qItem.question, goldTurnIds, goldSessIds });
      }
    }

    if (queryEntries.length === 0) continue;
    const resp = await rank({
      sessions,
      queries: queryEntries.map(e => ({ text: e.question, date: '' })),
      top_k: 10,
    });

    for (let qi = 0; qi < queryEntries.length; qi++) {
      const { cat, goldTurnIds, goldSessIds } = queryEntries[qi];
      const res = resp.results?.[qi];
      const retSessIds = (res?.sessions || []).map(s => s.id);
      const retTurnIds = (res?.turns || []).map(t => t.id);
      const govRules = (res?.standing_rules || []).slice(0, 2);
      const pktSess5 = [...new Set([...retSessIds.slice(0, 5), ...govRules.map(r => r.session_id)])];
      const pktTurn5 = [...new Set([...retTurnIds.slice(0, 5), ...govRules.map(r => r.id)])];

      scoredQuestions++;
      if (!categoryStats[cat]) {
        categoryStats[cat] = {
          total: 0,
          sessAny5: 0, sessAll5: 0, sessAny10: 0, sessAll10: 0,
          turnAny1: 0, turnAny5: 0, turnAll5: 0, turnAny10: 0,
          pktSessAny5: 0, pktSessAll5: 0, pktTurnAny5: 0, pktTurnAll5: 0,
          gtFired: 0, gtAmbiguous: 0,
        };
      }
      const st = categoryStats[cat];
      st.total++;

      const sAny = k => goldSessIds.some(g => retSessIds.slice(0, k).includes(g));
      const sAll = k => goldSessIds.length > 0 && goldSessIds.every(g => retSessIds.slice(0, k).includes(g));
      const tAny = k => goldTurnIds.some(g => retTurnIds.slice(0, k).includes(g));
      const tAll = k => goldTurnIds.length > 0 && goldTurnIds.every(g => retTurnIds.slice(0, k).includes(g));
      const pSAny5 = goldSessIds.some(g => pktSess5.includes(g));
      const pSAll5 = goldSessIds.length > 0 && goldSessIds.every(g => pktSess5.includes(g));
      const pTAny5 = goldTurnIds.some(g => pktTurn5.includes(g));
      const pTAll5 = goldTurnIds.length > 0 && goldTurnIds.every(g => pktTurn5.includes(g));

      if (sAny(1)) sessAny1++;
      if (sAny(3)) sessAny3++;
      if (sAny(5)) { sessAny5++; st.sessAny5++; }
      if (sAny(10)) { sessAny10++; st.sessAny10++; }
      if (sAll(5)) { sessAll5++; st.sessAll5++; }
      if (sAll(10)) { sessAll10++; st.sessAll10++; }
      if (pSAny5) { pktSessAny5++; st.pktSessAny5++; }
      if (pSAll5) { pktSessAll5++; st.pktSessAll5++; }
      const sRank = retSessIds.findIndex(id => goldSessIds.includes(id));
      if (sRank >= 0) sessMrrSum += 1.0 / (sRank + 1);

      if (tAny(1)) { turnAny1++; st.turnAny1++; }
      if (tAny(3)) turnAny3++;
      if (tAny(5)) { turnAny5++; st.turnAny5++; }
      if (res?.group_testing_fired) { gtFired++; st.gtFired++; }
      if (res?.group_testing_ambiguous) { gtAmbiguous++; st.gtAmbiguous++; }
      if (tAny(10)) { turnAny10++; st.turnAny10++; }
      if (tAll(5)) { turnAll5++; st.turnAll5++; }
      if (tAll(10)) turnAll10++;
      if (pTAny5) { pktTurnAny5++; st.pktTurnAny5++; }
      if (pTAll5) { pktTurnAll5++; st.pktTurnAll5++; }
      const tRank = retTurnIds.findIndex(id => goldTurnIds.includes(id));
      if (tRank >= 0) turnMrrSum += 1.0 / (tRank + 1);
    }

    if ((ci + 1) % 5 === 0 || ci === convsToRun.length - 1) {
      const sR5 = scoredQuestions > 0 ? (sessAny5 / scoredQuestions * 100).toFixed(2) : '0.00';
      const tR5 = scoredQuestions > 0 ? (turnAny5 / scoredQuestions * 100).toFixed(2) : '0.00';
      console.log(`[${ci + 1}/${convsToRun.length}] Scored ${scoredQuestions} Qs -> Sess Any@5: ${sR5}% | Turn Any@5: ${tR5}%`);
    }
  }

  const elapsedSec = ((Date.now() - t0) / 1000).toFixed(2);
  const pct = n => (scoredQuestions > 0 ? parseFloat((n / scoredQuestions * 100).toFixed(2)) : 0);

  console.log(`\n--- BEAM RECALL RESULTS (${splitDir}) ---`);
  console.log(`Conversations: ${convsToRun.length} | Sub-sessions: ${totalSubSessions} | Turns: ${totalTurns}`);
  console.log(`Total Qs: ${totalQuestions} | Scored (non-empty source_chat_ids): ${scoredQuestions}`);
  console.log(`Session RecallAny@1: ${pct(sessAny1)}% | Any@3: ${pct(sessAny3)}% | Any@5: ${pct(sessAny5)}% | Any@10: ${pct(sessAny10)}% | All@5: ${pct(sessAll5)}% | All@10: ${pct(sessAll10)}% | MRR: ${(sessMrrSum / scoredQuestions).toFixed(4)}`);
  console.log(`Turn    RecallAny@1: ${pct(turnAny1)}% | Any@3: ${pct(turnAny3)}% | Any@5: ${pct(turnAny5)}% | Any@10: ${pct(turnAny10)}% | All@5: ${pct(turnAll5)}% | All@10: ${pct(turnAll10)}% | MRR: ${(turnMrrSum / scoredQuestions).toFixed(4)}`);
  console.log(`Group-testing rule triggers: fired ${gtFired} / ambiguous ${gtAmbiguous} / scored ${scoredQuestions}`);
  console.log(`Stratified Packet (sessions[0..5] + standing_rules[0..2]): Sess Any@5: ${pct(pktSessAny5)}% All@5: ${pct(pktSessAll5)}% | Turn Any@5: ${pct(pktTurnAny5)}% All@5: ${pct(pktTurnAll5)}%`);
  console.log(`Time: ${elapsedSec}s | API Cost: $0.00`);

  console.log(`\n--- Category Breakdown ---`);
  for (const [c, st] of Object.entries(categoryStats).sort()) {
    if (st.total > 0) {
      const rate = (n) => ((n || 0) / st.total * 100).toFixed(1);
      console.log(
        `  ${c.padEnd(26)} (${String(st.total).padStart(3)}): ` +
        `Sess Any@5=${rate(st.sessAny5)}% All@5=${rate(st.sessAll5)}% ` +
        `(Pkt Any ${rate(st.pktSessAny5)}% All ${rate(st.pktSessAll5)}%) Any@10=${rate(st.sessAny10)}% | ` +
        `Turn Any@1=${rate(st.turnAny1)}% Any@5=${rate(st.turnAny5)}% All@5=${rate(st.turnAll5)}% ` +
        `GT fired ${st.gtFired} ambiguous ${st.gtAmbiguous} | ` +
        `(Pkt Any ${rate(st.pktTurnAny5)}% All ${rate(st.pktTurnAll5)}%) Any@10=${rate(st.turnAny10)}%`
      );
    }
  }

  const resultPayload = {
    benchmark: 'BEAM',
    split: splitDir,
    version: 'v6.2.2',
    timestamp: new Date().toISOString(),
    elapsed_seconds: parseFloat(elapsedSec),
    cost_usd: 0.0,
    conversations: convsToRun.length,
    sub_sessions: totalSubSessions,
    turns: totalTurns,
    metrics: {
      total_questions: totalQuestions,
      scored_questions: scoredQuestions,
      session_recall_any_at_1: pct(sessAny1),
      session_recall_any_at_3: pct(sessAny3),
      session_recall_any_at_5: pct(sessAny5),
      session_recall_any_at_10: pct(sessAny10),
      session_recall_all_at_5: pct(sessAll5),
      session_recall_all_at_10: pct(sessAll10),
      session_mrr: parseFloat((sessMrrSum / scoredQuestions).toFixed(4)),
      turn_recall_any_at_1: pct(turnAny1),
      turn_recall_any_at_3: pct(turnAny3),
      turn_recall_any_at_5: pct(turnAny5),
      turn_recall_any_at_10: pct(turnAny10),
      turn_recall_all_at_5: pct(turnAll5),
      turn_recall_all_at_10: pct(turnAll10),
      turn_mrr: parseFloat((turnMrrSum / scoredQuestions).toFixed(4)),
      group_testing_fired: gtFired,
      group_testing_ambiguous: gtAmbiguous,
      stratified_packet_session_recall_any_at_5: pct(pktSessAny5),
      stratified_packet_session_recall_all_at_5: pct(pktSessAll5),
      stratified_packet_turn_recall_any_at_5: pct(pktTurnAny5),
      stratified_packet_turn_recall_all_at_5: pct(pktTurnAll5),
    },
    category_breakdown: categoryStats,
  };

  const resultsDir = join(rootDir, 'benchmarks', 'results', 'v6.2.2');
  mkdirSync(resultsDir, { recursive: true });
  const outPath = args.out || join(resultsDir, `beam_${splitDir.toLowerCase()}_recall.json`);
  writeFileSync(outPath, JSON.stringify(resultPayload, null, 2), 'utf8');
  console.log(`\nCommitted result JSON to: ${outPath}`);
  return resultPayload;
}

// =========================================================================
// 2. HOTPOTQA BENCHMARK RUNNER
// =========================================================================
async function runHotpotQA(args) {
  console.log(`\n======================================================`);
  console.log(`=== RUNNING HOTPOTQA RECALL-ONLY BENCHMARK (Vestige 6.2.2) ===`);
  console.log(`======================================================`);

  const dataDir = join(rootDir, 'benchmarks', 'data');
  mkdirSync(dataDir, { recursive: true });
  const hotpotFile = join(dataDir, 'hotpot_dev_distractor_v1.json');

  if (!existsSync(hotpotFile)) {
    console.log(`Downloading hotpot_dev_distractor_v1.json (61 MB)...`);
    const { execSync } = await import('node:child_process');
    const url = 'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json';
    execSync(`curl.exe -fSL "${url}" -o "${hotpotFile}"`, { stdio: 'inherit' });
  }

  console.log(`Loading ${hotpotFile}...`);
  const items = JSON.parse(readFileSync(hotpotFile, 'utf8'));
  console.log(`Loaded ${items.length} HotpotQA dev items.`);

  const itemsToRun = args.limit ? items.slice(0, args.limit) : items;
  let scoredQuestions = 0;
  let docHitsAt2 = 0;
  let docHitsAt5 = 0;
  let sentHitsAt5 = 0;
  let sentHitsAt10 = 0;
  const t0 = Date.now();

  for (let i = 0; i < itemsToRun.length; i++) {
    const item = itemsToRun[i];
    scoredQuestions++;

    // Format context into Vestige Sessions (each document is a session, sentences are turns)
    const sessions = [];
    for (const [title, sentences] of item.context) {
      const turns = [];
      for (let si = 0; si < sentences.length; si++) {
        turns.push({
          id: `${title}_sent${si}`,
          role: 'user',
          content: sentences[si],
        });
      }
      sessions.push({
        id: title,
        date: '',
        timestamp: 0,
        turns,
      });
    }

    const goldDocTitles = [...new Set((item.supporting_facts || []).map(f => f[0]))];
    const goldSentIds = (item.supporting_facts || []).map(f => `${f[0]}_sent${f[1]}`);

    const req = {
      sessions,
      queries: [{ text: item.question, date: '' }],
      top_k: 10,
    };

    const resp = await rank(req);
    const res = resp.results && resp.results[0];
    const retDocIds = (res && res.sessions ? res.sessions.map(s => s.id) : []);
    const retSentIds = (res && res.turns ? res.turns.map(t => t.id) : []);

    // Document Recall: all gold doc titles retrieved
    const allDocsHit2 = goldDocTitles.every(gt => retDocIds.slice(0, 2).includes(gt));
    const allDocsHit5 = goldDocTitles.every(gt => retDocIds.slice(0, 5).includes(gt));
    if (allDocsHit2) docHitsAt2++;
    if (allDocsHit5) docHitsAt5++;

    // Sentence Recall: supporting facts retrieved
    const allSentsHit5 = goldSentIds.every(gs => retSentIds.slice(0, 5).includes(gs));
    const allSentsHit10 = goldSentIds.every(gs => retSentIds.slice(0, 10).includes(gs));
    if (allSentsHit5) sentHitsAt5++;
    if (allSentsHit10) sentHitsAt10++;

    if ((i + 1) % 200 === 0 || i === itemsToRun.length - 1) {
      const docR2 = (docHitsAt2 / scoredQuestions * 100).toFixed(2);
      const sentR10 = (sentHitsAt10 / scoredQuestions * 100).toFixed(2);
      console.log(`[${i + 1}/${itemsToRun.length}] Doc All-R@2: ${docR2}% | Sent All-R@10: ${sentR10}%`);
    }
  }

  const elapsedSec = ((Date.now() - t0) / 1000).toFixed(2);
  const docR2 = (docHitsAt2 / scoredQuestions * 100).toFixed(2);
  const docR5 = (docHitsAt5 / scoredQuestions * 100).toFixed(2);
  const sentR5 = (sentHitsAt5 / scoredQuestions * 100).toFixed(2);
  const sentR10 = (sentHitsAt10 / scoredQuestions * 100).toFixed(2);

  console.log(`\n--- HOTPOTQA RECALL RESULTS ---`);
  console.log(`Questions Evaluated: ${scoredQuestions}`);
  console.log(`Doc RecallAll@2:     ${docR2}%`);
  console.log(`Doc RecallAll@5:     ${docR5}%`);
  console.log(`Sent RecallAll@5:    ${sentR5}%`);
  console.log(`Sent RecallAll@10:   ${sentR10}%`);
  console.log(`Time:                ${elapsedSec}s`);
  console.log(`Vestige API Cost:    $0.00 (Zero LLM, Zero Vector Cloud Spend)`);

  const resultPayload = {
    benchmark: 'HotpotQA',
    split: 'dev_distractor',
    version: 'v6.2.2',
    timestamp: new Date().toISOString(),
    elapsed_seconds: parseFloat(elapsedSec),
    cost_usd: 0.0,
    metrics: {
      questions_evaluated: scoredQuestions,
      doc_recall_all_at_2: parseFloat(docR2),
      doc_recall_all_at_5: parseFloat(docR5),
      sent_recall_all_at_5: parseFloat(sentR5),
      sent_recall_all_at_10: parseFloat(sentR10),
    },
  };

  const resultsDir = join(rootDir, 'benchmarks', 'results', 'v6.2.2');
  mkdirSync(resultsDir, { recursive: true });
  const outPath = args.out || join(resultsDir, `hotpotqa_recall.json`);
  writeFileSync(outPath, JSON.stringify(resultPayload, null, 2), 'utf8');
  console.log(`\nCommitted result JSON to: ${outPath}`);
  return resultPayload;
}

// =========================================================================
// PYTHON STREAM / AST EVALUATOR HELPER
// =========================================================================
async function runPythonScript(scriptPath, scriptArgs = []) {
  const { spawnSync } = await import('node:child_process');
  const pythonCandidates = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
  for (const py of pythonCandidates) {
    const probe = spawnSync(py, ['--version'], { encoding: 'utf8' });
    if (probe.status === 0) {
      const res = spawnSync(py, ['-u', scriptPath, ...scriptArgs], { stdio: 'inherit' });
      if (res.status !== 0) {
        throw new Error(`${py} ${scriptPath} exited with status ${res.status}`);
      }
      return;
    }
  }
  throw new Error('Python 3 runtime not found (tried python3, python)');
}

// =========================================================================
// 3. MEMORYAGENTBENCH RUNNER (Accurate Retrieval Track)
// =========================================================================
async function runMemoryAgentBench(args) {
  console.log(`\n======================================================`);
  console.log(`=== RUNNING MEMORYAGENTBENCH RECALL-ONLY (Vestige 6.2.2) ===`);
  console.log(`======================================================`);
  const scriptPath = join(rootDir, 'benchmarks', 'eval_memoryagentbench.py');
  const scriptArgs = [];
  if (args.out) scriptArgs.push('--json', args.out);
  await runPythonScript(scriptPath, scriptArgs);
}

// =========================================================================
// 4. LONGMEMEVAL_M STREAMING RUNNER (~476 sessions/Q)
// =========================================================================
async function runLongMemEvalM(args) {
  console.log(`\n======================================================`);
  console.log(`=== RUNNING LONGMEMEVAL_M RECALL-ONLY (Vestige 6.2.2) ===`);
  console.log(`======================================================`);
  const scriptPath = join(rootDir, 'benchmarks', 'eval_longmemeval_m.py');
  const scriptArgs = [];
  if (args.limit) scriptArgs.push('--limit', String(args.limit));
  if (args.out) scriptArgs.push('--json', args.out);
  await runPythonScript(scriptPath, scriptArgs);
}

// =========================================================================
// MAIN ROUTER
// =========================================================================
async function main() {
  const args = parseArgs();
  let exitCode = 0;
  try {
    if (args.bench === 'beam') {
      await runBeam(args);
    } else if (args.bench === 'hotpotqa') {
      await runHotpotQA(args);
    } else if (args.bench === 'memoryagentbench') {
      await runMemoryAgentBench(args);
    } else if (args.bench === 'longmemeval_m') {
      await runLongMemEvalM(args);
    } else if (args.bench === 'all') {
      console.log('=== RUNNING ALL CORE RECALL BENCHMARKS (BEAM, MemoryAgentBench, LongMemEval_M) ===');
      await runBeam(args);
      await runMemoryAgentBench(args);
      await runLongMemEvalM(args);
    } else {
      console.error(`Unknown benchmark: ${args.bench}. Available: beam, memoryagentbench, longmemeval_m, hotpotqa, all`);
      exitCode = 1;
    }
  } catch (err) {
    console.error('Benchmark execution error:', err);
    exitCode = 1;
  } finally {
    if (child) {
      try { child.stdin.on('error', () => {}); } catch {}
      try { child.stdin.end(); } catch {}
      child.kill();
    }
  }
  if (exitCode !== 0) process.exit(1);
}

main();
