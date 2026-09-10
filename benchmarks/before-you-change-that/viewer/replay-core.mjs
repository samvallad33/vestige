export const ARMS = ['control', 'mcp-memory-service', 'vestige'];

export const ARM_LABELS = {
  control: 'Control',
  'mcp-memory-service': 'MCP Memory Service',
  vestige: 'Vestige',
};

const PRIVATE_PATH_PATTERNS = [
  /(?:file:\/\/)?\/Users\/[^/\s"']+/gi,
  /(?:file:\/\/)?\/home\/[^/\s"']+/gi,
  /[A-Za-z]:\\Users\\[^\\\s"']+/gi,
];

export function parseJsonl(text, source = 'evidence.jsonl') {
  const rows = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line));
    } catch (error) {
      throw new Error(`${source}:${index + 1}: ${error.message}`);
    }
  }
  return rows;
}

export function assertPublicEvidence(value, source) {
  const serialized = typeof value === 'string' ? value : JSON.stringify(value);
  const hits = [];
  for (const pattern of PRIVATE_PATH_PATTERNS) {
    pattern.lastIndex = 0;
    for (const match of serialized.matchAll(pattern)) hits.push(match[0]);
  }
  if (hits.length) {
    const examples = [...new Set(hits)].slice(0, 3).join(', ');
    throw new Error(`${source} still contains a private absolute path (${examples}). Replay stopped.`);
  }
}

export function validateArmRows(arm, rows) {
  if (!Array.isArray(rows) || rows.length === 0) throw new Error(`${arm}.jsonl has no events`);
  let previous = -Infinity;
  const seen = new Set();
  for (const row of rows) {
    if (row.arm !== arm) throw new Error(`${arm}.jsonl contains an event for ${String(row.arm)}`);
    if (!Number.isInteger(row.seq)) throw new Error(`${arm}.jsonl contains a non-integer sequence`);
    if (seen.has(row.seq) || row.seq <= previous) throw new Error(`${arm}.jsonl sequence is not strictly increasing`);
    seen.add(row.seq);
    previous = row.seq;
  }
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function parseTime(value) {
  const milliseconds = Date.parse(value || '');
  return Number.isFinite(milliseconds) ? milliseconds / 1000 : null;
}

export function timelineRows(eventsByArm, manifest = {}, recordingTiming = null) {
  const allRows = ARMS.flatMap((arm) => eventsByArm[arm] || []);
  const explicitOrigin = parseTime(
    recordingTiming?.capture_context_utc
      || manifest?.public_export?.recording_started_utc
      || manifest?.publicExport?.recordingStartedUtc
      || manifest?.recording?.started_utc,
  );
  const explicitDispatch = parseTime(recordingTiming?.dispatch_utc);
  const nonTaskTimes = allRows
    .filter((row) => row.kind !== 'task')
    .map((row) => parseTime(row.utc))
    .filter((value) => value !== null);
  const firstRecordedEvent = nonTaskTimes.length ? Math.min(...nonTaskTimes) : 0;
  const replayLead = finiteNumber(
    recordingTiming?.replay_lead_seconds
      ?? manifest?.public_export?.replay_lead_seconds
      ?? manifest?.publicExport?.replayLeadSeconds
      ?? manifest?.recording?.replay_lead_seconds,
  ) ?? 2;

  const projected = {};
  for (const arm of ARMS) {
    projected[arm] = (eventsByArm[arm] || []).map((row) => {
      const explicitOffset = finiteNumber(
        row.recorded_offset_seconds
          ?? row.replay_offset_seconds
          ?? row.recording_offset_seconds,
      );
      const utc = parseTime(row.utc);
      let offset;
      if (explicitOffset !== null) offset = explicitOffset;
      else if (row.kind === 'task') offset = 0;
      else if (explicitOrigin !== null && utc !== null) offset = utc - explicitOrigin;
      else if (explicitDispatch !== null && utc !== null) offset = utc - explicitDispatch + replayLead;
      else if (utc !== null) offset = utc - firstRecordedEvent + replayLead;
      else offset = replayLead;
      return {...row, replay_offset_seconds: Math.max(0, offset)};
    });
  }
  return projected;
}

export function comparisonFromLocation(locationLike) {
  const params = new URLSearchParams(locationLike.search || '');
  const requested = params.get('compare') || String(locationLike.hash || '').replace(/^#/, '');
  return requested === 'control' ? 'control' : 'mcp-memory-service';
}

export function resultReports(result) {
  return Object.fromEntries((result?.reports || []).map((report) => [report.arm, report]));
}

export function usageTotals(report) {
  const usage = Array.isArray(report?.model_run?.usage) ? report.model_run.usage : [];
  if (!usage.length) return null;
  return usage.reduce((totals, row) => ({
    input_tokens: totals.input_tokens + Number(row.input_tokens || 0),
    cached_input_tokens: totals.cached_input_tokens + Number(row.cached_input_tokens || 0),
    output_tokens: totals.output_tokens + Number(row.output_tokens || 0),
  }), {input_tokens: 0, cached_input_tokens: 0, output_tokens: 0});
}

export function runStatus(report) {
  if (!report?.model_run) return 'Status unavailable';
  if (report.model_run.timeout) return `Timed out at ${Math.round(Number(report.model_run.elapsed_seconds || 0))}s`;
  if (report.model_run.exit_code === 0 && report.model_run.stdout_closed) return 'Natural completion';
  return `Exited ${String(report.model_run.exit_code ?? 'without status')}`;
}

export function eventParts(row) {
  const raw = row?.event && typeof row.event === 'object' ? row.event : {};
  const item = raw.item && typeof raw.item === 'object' ? raw.item : {};
  return {raw, item};
}

export function eventType(row) {
  const {raw, item} = eventParts(row);
  return String(item.type || raw.type || row.kind || 'event');
}

export function eventCategory(row) {
  const type = eventType(row).toLowerCase();
  if (row.kind === 'task') return 'task';
  if (row.kind === 'native_wire' || type.includes('tool') || type.includes('mcp')) return 'tool';
  if (type === 'agent_message' || type.includes('message')) return 'narration';
  if (type.includes('command')) return 'command';
  if (['evaluation', 'receipt', 'intention_lifecycle', 'session_end'].includes(row.kind)) return 'evidence';
  if (/file|edit|patch|change|test/.test(type)) return 'change';
  return row.kind === 'system' ? 'system' : 'event';
}

function textFromContent(value) {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value.map((part) => {
    if (typeof part === 'string') return part;
    if (!part || typeof part !== 'object') return '';
    return part.text || part.content || '';
  }).filter(Boolean).join('\n');
}

export function eventBody(row) {
  const {raw, item} = eventParts(row);
  const type = eventType(row).toLowerCase();
  if (typeof row.text === 'string' && row.text) return row.text;
  if (type === 'agent_message' || item.role === 'assistant') {
    return textFromContent(item.text || item.content || item.message);
  }
  if (type.includes('command')) {
    const command = item.command || item.cmd || raw.command || '';
    const output = item.aggregated_output ?? item.stdout ?? raw.output ?? '';
    const stderr = item.stderr ?? raw.stderr ?? '';
    const status = item.status || raw.status || '';
    return [command && `$ ${command}`, output, stderr && `STDERR\n${stderr}`, `status=${status || 'unknown'} exit=${String(item.exit_code ?? 'pending')}`]
      .filter(Boolean).join('\n\n');
  }
  if (type.includes('tool') || type.includes('mcp')) {
    const call = [item.server || raw.server, item.tool || item.name || raw.tool || raw.name].filter(Boolean).join('.');
    const argumentsValue = item.arguments ?? item.args ?? raw.arguments ?? raw.args ?? item.prompt;
    const resultValue = item.result ?? raw.result ?? item.agents_states;
    return [
      call && `TOOL ${call}`,
      argumentsValue !== undefined && `ARGUMENTS\n${printable(argumentsValue)}`,
      resultValue !== undefined && `RESULT\n${printable(resultValue)}`,
      (item.error ?? raw.error) && `ERROR\n${printable(item.error ?? raw.error)}`,
      (item.status ?? raw.status) && `STATUS ${String(item.status ?? raw.status)}`,
    ].filter(Boolean).join('\n\n');
  }
  if (item.message || raw.message) return printable(item.message || raw.message);
  if (row.report) return printable(row.report);
  if (Object.keys(item).length) return printable(item);
  if (Object.keys(raw).length) return printable(raw);
  return printable(row);
}

export function printable(value) {
  return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

export function formatClock(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = Math.floor(value % 60);
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

export function maxEventOffset(projected) {
  return Math.max(0, ...ARMS.flatMap((arm) => (projected[arm] || []).map((row) => row.replay_offset_seconds)));
}

export function changeTimes(projected, visibleArms) {
  return [...new Set([
    0,
    ...visibleArms.flatMap((arm) => (projected[arm] || []).map((row) => row.replay_offset_seconds)),
  ].map((value) => Number(value.toFixed(3))))].sort((a, b) => a - b);
}

export function finalFacts(result) {
  const reports = resultReports(result);
  const scores = ARMS.map((arm) => {
    const evaluation = reports[arm]?.evaluation || {};
    return `${ARM_LABELS[arm]} ${String(evaluation.score ?? '?')}/${String(evaluation.total ?? '?')}`;
  });
  const intention = reports.vestige?.feature_observed?.context_intention;
  const lifecycle = reports.vestige?.intention_lifecycle;
  return {
    scores,
    statuses: ARMS.map((arm) => `${ARM_LABELS[arm]}: ${runStatus(reports[arm])}`),
    intentionReturned: Boolean(intention?.observed && intention?.observations?.some((row) => row.exact_triggered_delivery)),
    coordinatorCompleted: Boolean(
      lifecycle?.application_passed
        && lifecycle?.stored_after?.status === 'fulfilled'
        && lifecycle?.status === 'COMPLETED_AFTER_APPLICATION_PASS',
    ),
  };
}
