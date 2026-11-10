import init, { rank_memories } from 'file:///C:/Users/Sam/Documents/ChatGPT/New%20project/salience-pro/app/src/lib/web/instant-core/salience_web_core.js';
import { readFileSync, appendFileSync, writeFileSync } from 'node:fs';

const wasmBuffer = readFileSync('C:/Users/Sam/Documents/ChatGPT/New project/salience-pro/app/src/lib/web/instant-core/salience_web_core_bg.wasm');
await init({ module_or_path: wasmBuffer });

const CLOUDFLARE_API = 'https://stara-companion.pages.dev/api/generate';

function getOfficialJudgePrompt(task, question, answer, response, isAbstention) {
  if (!isAbstention) {
    if (['single-session-user', 'single-session-assistant', 'multi-session'].includes(task)) {
      return `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\nQuestion: ${question}\n\nCorrect Answer: ${answer}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
    } else if (task === 'temporal-reasoning') {
      return `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days.\n\nQuestion: ${question}\n\nCorrect Answer: ${answer}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
    } else if (task === 'knowledge-update') {
      return `I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: ${question}\n\nCorrect Answer: ${answer}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
    } else if (task === 'single-session-preference') {
      return `I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no.\n\nQuestion: ${question}\n\nRubric: ${answer}\n\nModel Response: ${response}\n\nIs the model response correct? Answer yes or no only.`;
    }
  }
  return `I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: ${question}\n\nExplanation: ${answer}\n\nModel Response: ${response}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.`;
}

async function callCloudflareAI(systemPrompt, userQuery, maxTokens = 250) {
  const res = await fetch(CLOUDFLARE_API, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userQuery }
      ],
      max_tokens: maxTokens,
      temperature: 0.1
    })
  });
  const data = await res.json();
  try {
    const parsed = JSON.parse(data.text);
    return parsed.reply || data.text;
  } catch {
    const m = data.text?.match(/"reply"\s*:\s*"((?:\\.|[^"\\])*)"/);
    return m ? JSON.parse(`"${m[1]}"`) : (data.text || '');
  }
}

async function runStage2(limit = 12) {
  // FULL HAYSTACK SPLIT: 50+ noise sessions mixed in with ground truth
  const sPath = 'C:/Users/Sam/Documents/ChatGPT/New project/vestige/benchmarks/data/longmemeval_s_cleaned.json';
  const data = JSON.parse(readFileSync(sPath, 'utf8'));

  const hypPath = 'C:/Users/Sam/Documents/ChatGPT/New project/vestige/benchmarks/outputs/longmemeval_hypothesis.jsonl';
  writeFileSync(hypPath, '', 'utf8');

  // Sample items across all 6 question types
  const sampleItems = [];
  const types = [...new Set(data.map(d => d.question_type))];
  const perType = Math.ceil(limit / types.length);

  for (const t of types) {
    const matching = data.filter(d => d.question_type === t).slice(0, perType);
    sampleItems.push(...matching);
  }
  const evalSet = sampleItems.slice(0, limit);

  console.log('========================================================================');
  console.log('  VESTIGE v6.2.0 STAGE 2: END-TO-END QA READER & OFFICIAL JUDGE HARNESS ');
  console.log('  100% Rust WASM Retrieval (Full Haystack) · Remote Cloudflare GPU Judge');
  console.log('========================================================================\n');
  console.log(`Running End-to-End Evaluation on ${evalSet.length} representative items across ${types.length} categories.`);
  console.log(`Writing official LongMemEval format to: ${hypPath}\n`);

  let correctCount = 0;

  for (let i = 0; i < evalSet.length; i++) {
    const item = evalSet[i];
    const qType = item.question_type;
    const isAbs = item.question_id.includes('_abs');

    // 1. Ingest full haystack blind into Rust WASM format (zero metadata leaks)
    const sessions = item.haystack_sessions.map((sess, sIdx) => ({
      id: String(item.haystack_session_ids[sIdx] || `sess_${sIdx}`),
      date: String(item.haystack_dates[sIdx] || ''),
      turns: sess.map((t, j) => ({
        id: `${item.haystack_session_ids[sIdx] || sIdx}#${j}`,
        role: String(t.role),
        content: String(t.content)
      }))
    }));

    const query = {
      text: String(item.question),
      date: String(item.question_date || '')
    };

    // 2. Query Rust WASM Retrieval Engine (<1ms)
    const rankPayload = {
      sessions,
      queries: [query],
      top_k: 3
    };

    const rustResult = JSON.parse(rank_memories(JSON.stringify(rankPayload)));
    const topSessionIds = new Set(rustResult.results[0].sessions.map(s => s.id));

    // Gather turns from top retrieved sessions
    const relevantTurns = [];
    sessions.forEach(sess => {
      if (topSessionIds.has(sess.id)) {
        sess.turns.forEach(t => {
          relevantTurns.push(`[${sess.date}] ${t.role === 'user' ? 'User' : 'Assistant'}: ${t.content}`);
        });
      }
    });

    const contextBlock = relevantTurns.slice(-15).join('\n');

    // 3. Compile compact H2SK prompt
    const systemPrompt = `You are a precise, factual AI assistant answering strictly from the user's recorded memory.
Current Date: ${item.question_date}

[RECALLED EPIGENETIC CONTEXT BLOCK]
${contextBlock}

INSTRUCTIONS:
1. Answer the question directly and concisely in 1-2 sentences based ONLY on the memories above.
2. If multiple dates show conflicting or updated information, ALWAYS treat the more recent date as the true updated answer.
3. If not mentioned in memory, state clearly: "I do not have that information."`;

    // 4. Remote Reader Inference
    const readerReply = await callCloudflareAI(systemPrompt, item.question, 120);

    // Append to official hypothesis.jsonl
    const hypLine = JSON.stringify({ question_id: item.question_id, hypothesis: readerReply });
    appendFileSync(hypPath, hypLine + '\n', 'utf8');

    // 5. Official Judge Scoring
    const judgePrompt = getOfficialJudgePrompt(qType, item.question, item.answer, readerReply, isAbs);
    const judgeVerdict = await callCloudflareAI(
      'You are an impartial evaluation judge. Answer strictly yes or no.',
      judgePrompt,
      10
    );

    const isYes = /yes/i.test(judgeVerdict);
    if (isYes) correctCount++;
    const icon = isYes ? '✓ CORRECT' : '✗ WRONG';

    console.log(`[${icon}] #${i + 1} [${qType}] (ID: ${item.question_id})`);
    console.log(`   Q: "${item.question}"`);
    console.log(`   Gold: "${item.answer}"`);
    console.log(`   Model: "${readerReply.trim().replace(/\n+/g, ' ').slice(0, 160)}"`);
    console.log(`   Judge: ${judgeVerdict.trim()}\n`);
  }

  console.log('========================================================================');
  console.log(`  STAGE 2 RUN COMPLETE: ${correctCount} / ${evalSet.length} CORRECT (${Math.round(correctCount / evalSet.length * 100)}%)`);
  console.log(`  Hypothesis output saved to: ${hypPath}`);
  console.log('========================================================================');
}

const count = parseInt(process.argv[2] || '12', 10);
runStage2(count).catch(console.error);
