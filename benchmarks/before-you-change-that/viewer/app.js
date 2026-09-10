import {
  ARMS,
  ARM_LABELS,
  assertPublicEvidence,
  changeTimes,
  comparisonFromLocation,
  eventBody,
  eventCategory,
  eventType,
  finalFacts,
  formatClock,
  maxEventOffset,
  parseJsonl,
  resultReports,
  runStatus,
  timelineRows,
  usageTotals,
  validateArmRows,
} from './replay-core.mjs';

const evidenceBase = new URL('../evidence/', window.location.href);
const comparison = comparisonFromLocation(window.location);
const visibleArms = [comparison, 'vestige'];
const laneNodes = new Map();

const state = {
  manifest: null,
  recordingTiming: null,
  result: null,
  reports: {},
  events: {},
  projected: {},
  currentTime: 0,
  duration: 1,
  playing: false,
  speed: 1,
  follow: true,
  search: '',
  lastFrame: null,
  endVisible: false,
  renderMode: new URLSearchParams(window.location.search).get('render') === '1',
};

function node(selector) {
  const found = document.querySelector(selector);
  if (!found) throw new Error(`Missing viewer element: ${selector}`);
  return found;
}

async function fetchText(name) {
  const response = await fetch(new URL(name, evidenceBase), {cache: 'no-store'});
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response.text();
}

async function fetchJson(name) {
  const text = await fetchText(name);
  assertPublicEvidence(text, name);
  return JSON.parse(text);
}

async function fetchOptionalJson(name) {
  const response = await fetch(new URL(name, evidenceBase), {cache: 'no-store'});
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  const text = await response.text();
  assertPublicEvidence(text, name);
  return JSON.parse(text);
}

function compactNumber(value) {
  return new Intl.NumberFormat('en-US').format(value);
}

function createLane(arm) {
  const fragment = node('#lane-template').content.cloneNode(true);
  const lane = fragment.querySelector('.lane');
  lane.dataset.arm = arm;
  fragment.querySelector('.lane-kicker').textContent = arm === 'vestige' ? 'EVIDENCE PATH' : 'COMPARISON PATH';
  fragment.querySelector('.lane-title').textContent = ARM_LABELS[arm];
  const report = state.reports[arm];
  fragment.querySelector('.lane-status').textContent = runStatus(report);
  fragment.querySelector('.lane-score').textContent = `${String(report?.evaluation?.score ?? '?')} / ${String(report?.evaluation?.total ?? '?')}`;

  const usage = usageTotals(report);
  for (const counter of fragment.querySelectorAll('[data-usage]')) {
    const field = counter.dataset.usage;
    counter.textContent = usage ? compactNumber(usage[field]) : 'not emitted';
    if (!usage) counter.closest('span').title = 'No final usage row was emitted after the recorded timeout.';
  }

  const nodes = {
    lane,
    current: fragment.querySelector('.current-event'),
    transcript: fragment.querySelector('.transcript'),
    count: fragment.querySelector('.lane-count'),
    cards: [],
  };
  for (const row of state.projected[arm]) {
    const card = createEventCard(row);
    nodes.transcript.append(card);
    nodes.cards.push({row, card});
  }
  node('#lanes').append(fragment);
  laneNodes.set(arm, nodes);
}

function createEventCard(row) {
  const fragment = node('#event-template').content.cloneNode(true);
  const card = fragment.querySelector('.event-card');
  const category = eventCategory(row);
  const body = eventBody(row) || '(event contains metadata only)';
  card.dataset.seq = String(row.seq);
  card.dataset.offset = String(row.replay_offset_seconds);
  card.dataset.category = category;
  card.searchText = `${category}\n${eventType(row)}\n${JSON.stringify(row)}`.toLowerCase();
  fragment.querySelector('.event-seq').textContent = `#${String(row.seq)}`;
  fragment.querySelector('.event-type').textContent = eventType(row);
  fragment.querySelector('.event-time').textContent = `+${formatClock(row.replay_offset_seconds)}`;
  fragment.querySelector('.event-preview').textContent = body.replace(/\s+/g, ' ').slice(0, 150);
  fragment.querySelector('.event-body').textContent = body;
  const rawDetails = fragment.querySelector('.raw-details');
  const rawPre = fragment.querySelector('.event-raw');
  rawDetails.addEventListener('toggle', () => {
    if (rawDetails.open && !rawPre.textContent) {
      rawPre.textContent = JSON.stringify(row, null, 2);
    }
  });
  return fragment;
}

function renderCurrent(arm, rows) {
  const current = rows.at(-1);
  const target = laneNodes.get(arm).current;
  if (!current) {
    target.className = 'current-event empty';
    target.textContent = 'Waiting for the first recorded event.';
    return;
  }
  target.className = `current-event ${eventCategory(current)}`;
  target.replaceChildren();
  const head = document.createElement('div');
  head.className = 'current-head';
  const strong = document.createElement('strong');
  strong.textContent = `#${String(current.seq)} · ${eventType(current)}`;
  const time = document.createElement('time');
  time.textContent = `recorded +${formatClock(current.replay_offset_seconds)}`;
  head.append(strong, time);
  const pre = document.createElement('pre');
  pre.textContent = eventBody(current) || JSON.stringify(current, null, 2);
  target.append(head, pre);
}

function matchesSearch(card) {
  return !state.search || card.searchText.includes(state.search);
}

function renderTime(seconds, options = {}) {
  state.currentTime = Math.max(0, Math.min(state.duration, Number(seconds) || 0));
  node('#clock').textContent = formatClock(state.currentTime);
  node('#scrubber').value = String(state.currentTime);
  let visible = 0;
  let total = 0;

  for (const arm of visibleArms) {
    const lane = laneNodes.get(arm);
    const elapsedRows = state.projected[arm].filter((row) => row.replay_offset_seconds <= state.currentTime + 0.0005);
    renderCurrent(arm, elapsedRows);
    total += lane.cards.length;
    for (const {row, card} of lane.cards) {
      const show = row.replay_offset_seconds <= state.currentTime + 0.0005 && matchesSearch(card);
      card.hidden = !show;
      if (show) visible += 1;
    }
    const shownHere = lane.cards.filter(({card}) => !card.hidden).length;
    lane.count.textContent = `${String(shownHere)} / ${String(lane.cards.length)} events visible`;
    if (state.follow && !state.renderMode && !options.skipFollow) lane.transcript.scrollTop = lane.transcript.scrollHeight;
  }
  node('#event-progress').textContent = `${String(visible)} / ${String(total)} events visible`;
  const shouldShowEnd = state.currentTime >= state.duration - 0.001;
  setEndVisible(shouldShowEnd);
}

function setEndVisible(visible) {
  const end = node('#end-card');
  end.hidden = !visible;
  state.endVisible = visible;
  document.body.classList.toggle('show-end', visible);
}

function setPlaying(playing) {
  state.playing = playing;
  node('#play').textContent = playing ? 'Pause' : 'Play';
  if (playing) {
    state.lastFrame = performance.now();
    requestAnimationFrame(tick);
  }
}

function tick(now) {
  if (!state.playing) return;
  const delta = (now - state.lastFrame) / 1000 * state.speed;
  state.lastFrame = now;
  const next = state.currentTime + delta;
  if (next >= state.duration) {
    renderTime(state.duration);
    setPlaying(false);
    return;
  }
  renderTime(next);
  requestAnimationFrame(tick);
}

function renderEndCard() {
  const facts = finalFacts(state.result);
  const scores = node('#end-scores');
  scores.replaceChildren();
  for (const score of facts.scores) {
    const chip = document.createElement('strong');
    chip.textContent = score;
    scores.append(chip);
  }
  node('#end-status').textContent = facts.statuses.join(' · ');
  const tokenTarget = node('#end-tokens');
  tokenTarget.replaceChildren();
  for (const arm of ARMS) {
    const totals = usageTotals(state.reports[arm]);
    const line = document.createElement('span');
    line.textContent = totals
      ? `${ARM_LABELS[arm]} · ${compactNumber(totals.input_tokens)} input incl. ${compactNumber(totals.cached_input_tokens)} cached · ${compactNumber(totals.output_tokens)} output`
      : `${ARM_LABELS[arm]} · counters unavailable (no final usage row after timeout)`;
    tokenTarget.append(line);
  }
  node('#end-intention').textContent = [
    facts.intentionReturned ? 'Target-context intention returned during the recorded Vestige path.' : 'No target-context intention delivery is claimed.',
    facts.coordinatorCompleted ? 'Coordinator marked it complete only after the application tests passed.' : 'No completed post-test coordinator lifecycle is claimed.',
  ].join(' ');
}

function bindControls() {
  node('#play').addEventListener('click', () => setPlaying(!state.playing));
  node('#speed').addEventListener('change', (event) => { state.speed = Number(event.target.value); });
  node('#scrubber').addEventListener('input', (event) => {
    setPlaying(false);
    renderTime(Number(event.target.value), {skipFollow: true});
  });
  node('#search').addEventListener('input', (event) => {
    state.search = event.target.value.trim().toLowerCase();
    renderTime(state.currentTime, {skipFollow: true});
  });
  node('#follow').addEventListener('click', () => {
    state.follow = !state.follow;
    node('#follow').textContent = `Follow ${state.follow ? 'on' : 'off'}`;
    node('#follow').setAttribute('aria-pressed', String(state.follow));
  });
  node('#inspect-end').addEventListener('click', () => {
    setEndVisible(false);
    renderTime(state.duration - 0.01, {skipFollow: true});
  });
  window.addEventListener('keydown', (event) => {
    if (event.target.matches('input, select, button')) return;
    if (event.code === 'Space') {
      event.preventDefault();
      setPlaying(!state.playing);
    } else if (event.key === 'ArrowLeft') renderTime(state.currentTime - (event.shiftKey ? 30 : 5));
    else if (event.key === 'ArrowRight') renderTime(state.currentTime + (event.shiftKey ? 30 : 5));
  });
}

function configureHeader() {
  const title = state.manifest?.presentation?.title || 'The obligation arrives with the file.';
  const task = state.projected[visibleArms[0]].find((row) => row.kind === 'task')
    || state.projected.vestige.find((row) => row.kind === 'task');
  node('h1').textContent = title;
  node('#task-text').textContent = task?.text || 'Recorded task is available in the raw evidence.';
  node('#archive-summary').textContent = `Same ${String(state.manifest?.record_count ?? 22)}-record archive · original capture offsets from recording-timing.json · sanitized paths`;
  node('#view-name').textContent = `${ARM_LABELS[comparison]} vs Vestige`;
  node('#compare-control').setAttribute('aria-current', String(comparison === 'control'));
  node('#compare-memory').setAttribute('aria-current', String(comparison === 'mcp-memory-service'));
  node('#duration').textContent = formatClock(state.duration);
  node('#scrubber').max = String(state.duration);
}

async function start() {
  const [manifest, result, recordingTiming, ...jsonl] = await Promise.all([
    fetchJson('manifest.json'),
    fetchJson('result.json'),
    fetchOptionalJson('recording-timing.json'),
    ...ARMS.map((arm) => fetchText(`${arm}.jsonl`)),
  ]);
  state.manifest = manifest;
  state.result = result;
  state.recordingTiming = recordingTiming;
  state.reports = resultReports(result);
  for (const [index, arm] of ARMS.entries()) {
    assertPublicEvidence(jsonl[index], `${arm}.jsonl`);
    state.events[arm] = parseJsonl(jsonl[index], `${arm}.jsonl`);
    validateArmRows(arm, state.events[arm]);
  }
  state.projected = timelineRows(state.events, state.manifest, state.recordingTiming);
  state.duration = maxEventOffset(state.projected) + 1;

  configureHeader();
  visibleArms.forEach(createLane);
  renderEndCard();
  bindControls();
  if (state.renderMode) document.body.classList.add('render-mode');
  renderTime(0);

  window.__PUBLIC_REPLAY__ = {
    ready: true,
    comparison,
    visibleArms: [...visibleArms],
    duration: state.duration,
    recordingTiming: state.recordingTiming,
    changeTimes: changeTimes(state.projected, visibleArms),
    counts: Object.fromEntries(visibleArms.map((arm) => [arm, state.projected[arm].length])),
    seek: (seconds) => renderTime(seconds, {skipFollow: true}),
    showEnd: () => { renderTime(state.duration); setEndVisible(true); },
    state: () => ({currentTime: state.currentTime, endVisible: state.endVisible}),
  };
  document.documentElement.dataset.replayReady = 'true';
}

start().catch((error) => {
  const target = node('#error');
  target.hidden = false;
  target.textContent = `Public replay unavailable: ${error.message}`;
  document.documentElement.dataset.replayReady = 'error';
  console.error(error);
});
