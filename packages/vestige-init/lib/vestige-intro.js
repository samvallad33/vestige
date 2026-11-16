#!/usr/bin/env node

const readline = require('readline');
const cinema = require('./vestige-cinema.js');

const COLOR = cinema.COLOR;
const TRUECOLOR = cinema.TRUECOLOR;
const R = cinema.RESET;
const DIM = cinema.DIM;
const BOLD = cinema.BOLD;

function rgb(r, g, b) {
  if (!COLOR) return '';
  if (TRUECOLOR) return `\x1b[38;2;${r | 0};${g | 0};${b | 0}m`;
  return cinema.rgb(r, g, b);
}

// with colour off (NO_COLOR, TERM=dumb, not a terminal) there are no colour codes at all
const HAIR = COLOR ? rgb(70, 46, 32) || '\x1b[38;5;130m' : '';
const HAIR2 = COLOR ? rgb(50, 32, 22) || '\x1b[38;5;94m' : '';
const SKIN = COLOR ? rgb(205, 168, 140) || '\x1b[38;5;180m' : '';
const GLOW = rgb(120, 200, 255);
const AMBER = rgb(255, 190, 90);
const GREEN = rgb(100, 230, 140);
const RED = rgb(255, 100, 90);
const WHITE = rgb(240, 245, 255);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const W = () => process.stdout.columns || 100;
const H = () => process.stdout.rows || 32;

// A key skips the beat on screen; q or esc skips to the title; ctrl-c quits. The listener stays armed
// for the whole intro and is always removed again, whether or not a key was pressed.
let skipped = false;
let skipAll = false;
let restoreRaw = null;
let keyWaiter = null;
let inTitle = false;      // while the title's glitch plays, q or esc is already the answer
let titleAnswer = null;

function armSkip() {
  if (!process.stdin.isTTY) return null;
  const wasRaw = Boolean(process.stdin.isRaw);
  const onData = (buffer) => {
    const key = buffer.toString('utf8');
    if (key.includes('\x03')) { process.stdout.write(SCREEN_OFF); try { process.stdin.setRawMode(wasRaw); } catch {} process.exit(130); }
    if (key.includes('\x1a')) { suspendIntro(); return; }
    if (key === 'q' || key === '\x1b') { skipAll = true; if (inTitle) titleAnswer = 'q'; }
    // a burst of keys (a paste, or a held key) fast-forwards to the title
    const burst = key.length > 1 && !/^\x1b[[O]/.test(key);
    if (burst) skipAll = true;
    skipped = true;
    if (keyWaiter) { const wake = keyWaiter; keyWaiter = null; wake(burst ? (key.includes('\r') ? '\r' : key[0]) : key); }
  };
  try {
    process.stdin.setRawMode(true);
    process.stdin.resume();
    process.stdin.on('data', onData);
  } catch { return null; }
  restoreRaw = () => { process.stdin.off('data', onData); try { process.stdin.setRawMode(wasRaw); } catch {} process.stdin.pause(); };
  return true;
}

function disarmSkip() {
  if (restoreRaw) { restoreRaw(); restoreRaw = null; }
  keyWaiter = null;
}

// The terminal is given back on every way out: a quit signal, ctrl-z (and taken again on resume).
const SCREEN_ON = '\x1b[?1049h\x1b[?25l';
// ?5l: the lightbulb flash turns on reverse screen for 90 ms; no exit may leave it on
const SCREEN_OFF = '\x1b[0m\x1b[?5l\x1b[?25h\x1b[?1049l';
let screenTaken = false;
function onQuitSignal(signal) {
  if (screenTaken) process.stdout.write(SCREEN_OFF);
  try { process.stdin.setRawMode(false); } catch {}
  process.exit(signal === 'SIGTERM' ? 143 : signal === 'SIGHUP' ? 129 : 130);
}
let introSuspended = false;
function suspendIntro() {
  if (process.platform === 'win32') return;
  introSuspended = true;
  if (screenTaken) process.stdout.write(SCREEN_OFF);
  try { process.stdin.setRawMode(false); } catch {}
  require('./vestige-stage.js').stopJob(resumeIntro);
}
function resumeIntro() {
  if (!screenTaken) { introSuspended = false; return; }
  try { process.stdin.setRawMode(true); } catch {}
  process.stdout.write(SCREEN_ON + '\x1b[2J');
  introSuspended = false;
}
function guardSignals(on) {
  const act = on ? 'on' : 'off';
  for (const sig of ['SIGTERM', 'SIGHUP', 'SIGINT']) process[act](sig, onQuitSignal);
  if (process.platform !== 'win32') { process[act]('SIGTSTP', suspendIntro); process[act]('SIGCONT', resumeIntro); }
}

const nextKey = () => new Promise((resolve) => { keyWaiter = resolve; });

function frame(lines, opts = {}) {
  const width = Math.min(W(), 110);
  if (introSuspended) return;
  const out = [];
  // a scene taller than the window moves up, and nothing is ever written below the last row
  const rows = H();
  let y0 = opts.y0 || 1;
  if (y0 + lines.length - 1 > rows) y0 = Math.max(1, rows - lines.length + 1);
  // rows above the scene are cleared, so nothing of the last beat lingers there
  for (let y = 1; y < y0; y++) out.push(`\x1b[${y};1H\x1b[2K`);
  for (let i = 0; i < lines.length && y0 + i <= rows; i++) {
    const plain = lines[i].replace(/\x1b\[[0-9;]*m/g, '');
    if (plain.length > width) lines[i] = lines[i].slice(0, width);
    out.push(`\x1b[${y0 + i};1H` + lines[i] + '\x1b[K');
  }
  process.stdout.write(cinema.SYNC_ON + out.join('') + '\x1b[J' + cinema.SYNC_OFF);
}

function center(text, width) {
  const plain = text.replace(/\x1b\[[0-9;]*m/g, '');
  const pad = Math.max(0, Math.floor((width - plain.length) / 2));
  return ' '.repeat(pad) + text;
}

const STUDENT = [
  '          ▄▄▄▄▄▄▄▄▄          ',
  '       ▄█▀' + HAIR + '████████' + R + '▀█▄       ',
  '      █▀' + HAIR + '████████████' + R + '▀█      ',
  '     █' + HAIR2 + '█' + R + SKIN + '  ┌─┐  ┌─┐  ' + R + HAIR2 + '█' + R + '█     ',
  '     █' + HAIR2 + '█' + R + SKIN + '  └─┘◉◉└─┘  ' + R + HAIR2 + '█' + R + '█     ',
  '     █' + HAIR2 + '█' + R + SKIN + '     ⌄      ' + R + HAIR2 + '█' + R + '█     ',
  '      ▀' + HAIR + '█' + R + SKIN + '▂▂▂▂▂▂▂▂▂' + R + HAIR + '█' + R + '▀      ',
  '       ' + SKIN + '█▀▀▀█▀▀▀█' + R + '        ',
  '       ' + SKIN + '█   █   █' + R + '        ',
  '      ' + SKIN + '██▄▄█▄▄▄██' + R + '        ',
];

const DESK = [
  '  ┌─────────────────────────────────────────┐',
  '  │ ' + DIM + '▐' + GLOW + '█████████████' + DIM + '▌' + R + ' ┌────┐               │',
  '  │ ' + DIM + '▐' + GLOW + '█████████████' + DIM + '▌' + R + ' │    │               │',
  '  │ ' + DIM + '▐' + GLOW + '█████████████' + DIM + '▌' + R + ' └────┘               │',
  '  └─────────────────────────────────────────┘',
];

const CLOCK_DIGITS = {
  '1': ['██', ' █', ' █', ' █', ' █', '███'], '2': ['███', '  █', '  █', '███', '█  ', '███'],
  ':': ['  ', '██', '  ', '  ', '██', '  '], 'A': ['███', '█ █', '███', '█ █', '█ █', '█ █'],
  'M': ['█ █', '███', '█ █', '█ █', '█ █', '█ █'], 'P': ['███', '█ █', '███', '█  ', '█  ', '█  '],
};

function bigText(text, color) {
  const rows = ['', '', '', '', '', ''];
  for (const ch of text) {
    const glyph = CLOCK_DIGITS[ch];
    if (!glyph) continue;
    for (let i = 0; i < 6; i++) rows[i] += color + glyph[i] + R + ' ';
  }
  return rows;
}

async function beatDesk() {
  const width = Math.min(W(), 110);
  const sceneTop = Math.max(2, Math.floor(H() / 2) - 14);
  const flickers = [0, 1, 0.6, 1, 0.3, 1];
  for (let t = 0; t < 26 && !skipped; t++) {
    const flick = flickers[t % flickers.length];
    const glowColor = rgb(120 * flick, 200 * flick, 255 * flick);
    const lines = [];
    const clock = bigText(['1', ':', '5', 'A', 'M', ' '].slice(0, 5).join('') === '1:5AM' ? '1:51AM' : (t % 8 < 4 ? '1:51AM' : '1:52AM'), rgb(90, 90, 110));
    const deskGlow = DESK.map((l) => l.split(GLOW).join(rgb(120 * flick, 200 * flick, 255 * flick)));
    lines.push('');
    lines.push(...clock.map((l) => center(l, width)));
    lines.push('');
    for (let i = 0; i < STUDENT.length; i++) lines.push(center(STUDENT[i], width));
    lines.push('');
    lines.push(...deskGlow.map((l) => center(l, width)));
    const cups = Math.min(3, Math.floor(t / 8));
    const cupRow = ' ┌⊃┐ '.repeat(cups) + '┌─┐ '.repeat(3 - cups).trim();
    lines.push(center('  ' + DIM + cupRow + R, width));
    if (t > 18) lines.push(center(DIM + 'nine months in' + R, width));
    frame(lines, { y0: sceneTop });
    await sleep(160);
  }
}

const RELEASES = ['v1.1.0', 'v1.4.0', 'v2.0.0', 'v2.5.0', 'v3.0.0', 'v3.5.0', 'v4.0.0', 'v4.1.1', 'v4.2.0', 'v5.0.0', 'v6.0.0', 'v6.2.0', 'v6.2.2'];
const MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT'];

async function beatGrind() {
  const width = Math.min(W(), 110);
  const sceneTop = Math.max(2, Math.floor(H() / 2) - 12);
  const columns = Math.floor((width - 4) / 14);
  const rain = Array.from({ length: columns }, (_, i) => ({ y: -((i * 7) % 20), speed: 0.5 + ((i * 13) % 10) / 10, rel: RELEASES[(i * 3) % RELEASES.length] }));
  const starHistory = [176, 336, 475, 540, 587, 606, 633, 645, 648, 649];
  for (let t = 0; t < 46 && !skipped; t++) {
    const lines = ['', center(BOLD + '2026' + R, width), ''];
    for (let row = 0; row < 12; row++) {
      let lineStr = '  ';
      for (const drop of rain) {
        const dy = drop.y + row;
        if (dy >= 0 && dy < 12 && Math.random() > 0.82) {
          lineStr += (dy === 0 ? WHITE : rgb(120, 220, 180)) + drop.rel.slice(0, 7).padEnd(7) + R + '       ';
        } else lineStr += ' '.repeat(14);
      }
      lines.push(lineStr);
    }
    const monthIdx = Math.min(MONTHS.length - 1, Math.floor(t / 4.6));
    const stars = starHistory[monthIdx];
    lines.push('');
    lines.push(center(`${AMBER}${MONTHS[monthIdx]}${R}  ${DIM}★${R} ${BOLD}${stars}${R}`, width));
    lines.push(center('▁▂▃▄▅▆▇█'.slice(0, monthIdx + 1), width));
    for (const drop of rain) drop.y += drop.speed;
    frame(lines, { y0: sceneTop });
    await sleep(150);
  }
}

const BULB = [
  '    ▄▄▄▄▄    ',
  '  ▄█▀▀▀▀▀█▄  ',
  ' █▀  ' + AMBER + '╲ ╱' + R + '  ▀█ ',
  ' █   ' + AMBER + ' ╳ ' + R + '   █ ',
  ' █  ' + AMBER + ' ╱ ╲' + R + '   █ ',
  '  ▀█▄▄▄▄▄█▀  ',
  '    ▐▌ ▐▌    ',
  '    ▐▌ ▐▌    ',
  '     ▀▀▀     ',
];

async function beatLightbulb() {
  const width = Math.min(W(), 110);
  const sceneTop = Math.max(2, Math.floor(H() / 2) - 8);
  for (let t = 0; t < 24 && !skipped; t++) {
    const charge = Math.min(1, t / 16);
    const lines = [];
    if (t < 4) {
      lines.push(...Array.from({ length: 8 }, () => center(DIM + '·' + R, width)));
    } else {
      const lit = BULB.map((row, i) => {
        const on = i / BULB.length < charge;
        return center(on ? row : row.replace(/[╲╳╱▄▀█▌▐]/g, ' '), width);
      });
      lines.push(...lit);
      if (charge >= 1) {
        const rays = '   ✦       ✦       ✦   ';
        lines.unshift(center(rays.replace(/✦/g, m => AMBER + m + R), width));
        lines.push(center(rays.replace(/✦/g, m => AMBER + m + R), width));
      }
    }
    frame(lines, { y0: sceneTop });
    if (charge >= 1 && t === 19) {
      process.stdout.write('\x1b[?5h');
      await sleep(90);
      process.stdout.write('\x1b[?5l');
    }
    await sleep(130);
  }
}

function neuronNet(n, t) {
  const nodes = [];
  for (let i = 0; i < n; i++) {
    const a = i * 2.399963;
    const r = 0.18 + 0.3 * ((i * 37) % 10) / 10;
    nodes.push({ x: 0.5 + r * Math.cos(a + t * 0.004), y: 0.5 + r * 0.8 * Math.sin(a * 1.3 + t * 0.003), id: i });
  }
  return nodes;
}

async function beatNeuro() {
  const width = Math.min(W(), 110);
  const height = Math.min(H() - 4, 26);
  const sceneTop = 2;
  const total = 30;
  let keepPath = null;
  for (let t = 0; t < 64 && !skipped; t++) {
    const nodes = neuronNet(total, t);
    const lines = [];
    const litCount = Math.min(total, Math.floor(t / 2));
    const grid = Array.from({ length: height }, () => Array(width).fill(null));
    const px = (n) => Math.max(0, Math.min(width - 1, Math.round(n.x * (width - 2))));
    const py = (n) => Math.max(0, Math.min(height - 1, Math.round(n.y * (height - 2))));
    for (let i = 0; i < litCount; i++) {
      for (let j = i + 1; j < litCount; j++) {
        if ((i + j) % 3 !== 0) continue;
        const firing = Math.sin(t * 0.4 + i + j) > 0.55;
        if (!firing && t > 40 && !(keepPath && keepPath.includes(i) && keepPath.includes(j))) continue;
        const x0 = px(nodes[i]); const y0 = py(nodes[i]);
        const x1 = px(nodes[j]); const y1 = py(nodes[j]);
        const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
        for (let s = 0; s <= steps; s++) {
          const gx = Math.round(x0 + (x1 - x0) * s / steps);
          const gy = Math.round(y0 + (y1 - y0) * s / steps);
          if (grid[gy][gx] === null) grid[gy][gx] = { ch: firing ? '·' : DIM + '·' + R, c: firing ? WHITE : DIM };
        }
      }
    }
    if (t > 40 && !keepPath) keepPath = [3, 9, 17, 24, 29];
    if (keepPath) {
      for (let k = 0; k < keepPath.length - 1; k++) {
        const a = nodes[keepPath[k]]; const b = nodes[keepPath[k + 1]];
        const x0 = px(a); const y0 = py(a); const x1 = px(b); const y1 = py(b);
        const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
        const pulse = (t + k * 4) % 12 < 6;
        for (let s = 0; s <= steps; s++) {
          const gx = Math.round(x0 + (x1 - x0) * s / steps);
          const gy = Math.round(y0 + (y1 - y0) * s / steps);
          grid[gy][gx] = { ch: '◦', c: pulse ? GREEN : rgb(60, 140, 90) };
        }
      }
    }
    for (let i = 0; i < litCount; i++) {
      const n = nodes[i];
      const kept = keepPath && keepPath.includes(i);
      grid[py(n)][px(n)] = { ch: kept ? '◉' : '●', c: kept ? GREEN : rgb(140, 170, 255) };
    }
    for (let row = 0; row < height; row++) {
      let lineStr = '';
      for (let col = 0; col < width; col++) {
        const cell = grid[row][col];
        lineStr += cell ? (cell.c === DIM ? DIM + cell.ch + R : cell.c + cell.ch + R) : ' ';
      }
      lines.push(lineStr);
    }
    if (t > 44) {
      lines.push('');
      lines.push(center(DIM + 'most paths fade. what matters, stays.' + R, width));
    }
    frame(lines, { y0: sceneTop });
    await sleep(120);
  }
}

async function beatAlive() {
  const width = Math.min(W(), 110);
  const sceneTop = Math.max(2, Math.floor(H() / 2) - 10);
  const chainEvents = [
    { t: 2, text: 'block #1   genesis', color: DIM },
    { t: 8, text: 'block #12  first memory saved', color: rgb(140, 170, 255) },
    { t: 14, text: 'block #31  ' + RED + 'STOP' + R + '  rm${IFS}-rf${IFS}~', color: WHITE },
    { t: 20, text: 'block #47  receipt verified  ' + GREEN + '✔ chain intact' + R, color: WHITE },
    { t: 26, text: 'block #48  six agents. one memory.', color: GREEN },
  ];
  const agents = [
    { name: 'claude', x: 0.12 }, { name: 'cursor', x: 0.26 }, { name: 'gemini', x: 0.40 },
    { name: 'opencode', x: 0.54 }, { name: 'codex', x: 0.68 }, { name: 'zcode', x: 0.82 },
  ];
  // one column for the whole chain, so each connector sits under its bullet
  const plainLen = (text) => text.replace(/\x1b\[[0-9;]*m/g, '').length;
  const chainW = Math.max(...chainEvents.map((e) => plainLen(e.text))) + 4;
  const chainX = ' '.repeat(Math.max(0, Math.floor((width - chainW) / 2)));
  // one slot per agent: the node sits over the middle of its own name
  const SLOT = 10;
  const names = agents.map((a) => { const pad = Math.floor((SLOT - a.name.length) / 2); return ' '.repeat(pad) + a.name + ' '.repeat(SLOT - pad - a.name.length); }).join('');
  const nodeAt = agents.map((a, i) => i * SLOT + Math.floor((SLOT - a.name.length) / 2) + Math.floor((a.name.length - 1) / 2));
  for (let t = 0; t < 44 && !skipped; t++) {
    const lines = ['', center(BOLD + 'EVERY ACTION. A RECEIPT.' + R, width), ''];
    const shown = chainEvents.filter((e) => t >= e.t);
    shown.forEach((e, i) => {
      const stamp = t - e.t < 3 ? GREEN + '● ' + R : DIM + '· ' + R;
      lines.push(chainX + '  ' + stamp + e.color + e.text + R);
      if (i < shown.length - 1) lines.push(chainX + '  ' + DIM + '│' + R);
    });
    lines.push('');
    if (t >= 30) {
      const flow = t >= 34;
      let connected = '';
      for (let col = 0; col <= nodeAt[nodeAt.length - 1]; col++) {
        const i = nodeAt.indexOf(col);
        if (i >= 0) connected += (flow && (t + i * 2) % 8 < 4 ? GREEN : rgb(140, 170, 255)) + '◉' + R;
        else if (col > nodeAt[0]) connected += DIM + (flow && (t + col) % 9 === 0 ? '━' : '─') + R;
        else connected += ' ';
      }
      lines.push(center(connected + ' '.repeat(Math.max(0, names.length - plainLen(connected))), width));
      lines.push(center(DIM + names + R, width));
    }
    frame(lines, { y0: sceneTop });
    await sleep(140);
  }
}

const LOGO = [
  '██╗   ██╗███████╗███████╗████████╗██╗ ██████╗ ███████╗',
  '██║   ██║██╔════╝██╔════╝╚══██╔══╝██║██╔════╝ ██╔════╝',
  '██║   ██║█████╗  ███████╗   ██║   ██║██║  ███╗█████╗  ',
  '╚██╗ ██╔╝██╔══╝  ╚════██║   ██║   ██║██║   ██║██╔══╝  ',
  ' ╚████╔╝ ███████╗███████║   ██║   ██║╚██████╔╝███████╗',
  '  ╚═══╝  ╚══════╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚══════╝',
];

async function beatTitle(direct = false) {
  const width = Math.min(W(), 110);
  const sceneTop = Math.max(2, Math.floor(H() / 2) - 8);
  if (direct || cinema.REDUCED) {
    const lines = ['', ...LOGO.map((l) => center(cinema.gradient(l, [90, 120, 255], [190, 80, 255]), width)), ''];
    lines.push(center(BOLD + 'the causal memory for every agent' + R, width));
    lines.push(center(DIM + 'nine months. one student. 49 releases. 649 stars.' + R, width));
    lines.push('');
    lines.push(center(cinema.c.cyan + 'press ENTER to begin your journey' + R, width));
    frame(lines, { y0: sceneTop });
    return;
  }
  const pool = '!<>-_\\/[]{}=+*^?#01';
  for (let f = 0; f < 40 && !skipped; f++) {
    const lines = [''];
    for (const row of LOGO) {
      let out = '';
      for (let i = 0; i < row.length; i++) {
        const settle = f - i * 0.35;
        out += settle > 8 ? row[i] : (Math.random() < 0.5 ? pool[(Math.random() * pool.length) | 0] : ' ');
      }
      lines.push(center(cinema.gradient(out, [90, 120, 255], [190, 80, 255], f), width));
    }
    lines.push('');
    frame(lines, { y0: sceneTop });
    await sleep(60);
  }
  const lines = ['', ...LOGO.map((l) => center(cinema.gradient(l, [90, 120, 255], [190, 80, 255]), width)), ''];
  lines.push(center(BOLD + 'the causal memory for every agent' + R, width));
  lines.push(center(DIM + 'nine months. one student. 49 releases. 649 stars.' + R, width));
  lines.push('');
  lines.push(center(cinema.c.cyan + 'press ENTER to begin your journey' + R, width));
  frame(lines, { y0: sceneTop });
  await sleep(900);
}

// Plays the six beats, one key per beat, then holds the title until a key. Returns that key
// ('enter', 'q', or the key pressed). keepScreen leaves the alternate screen up for whatever follows.
async function playIntro({ skipToEnd = false, waitAtTitle = true, keepScreen = false } = {}) {
  // no picture where it cannot be drawn: not a terminal, a dumb one, a C locale, or a tiny window
  const loc = process.env.LC_ALL || process.env.LC_CTYPE || process.env.LANG || '';
  const asciiLocale = Boolean(loc) && !/utf-?8/i.test(loc) && (/^(C|POSIX)$/i.test(loc) || /\./.test(loc));
  if (!process.stdout.isTTY || process.env.TERM === 'dumb' || asciiLocale || W() < 60 || H() < 20) return 'enter';
  skipAll = skipToEnd;
  skipped = false;
  armSkip();
  screenTaken = true;
  guardSignals(true);
  process.stdout.write('\x1b[?1049h\x1b[?25l\x1b[2J');
  let key = 'enter';
  try {
    for (const beat of [beatDesk, beatGrind, beatLightbulb, beatNeuro, beatAlive]) {
      if (skipAll) break;
      skipped = false;
      await beat();
      // the frame a skipped beat ends on stays a moment, so a skip reads as a cut, not a crash
      if (skipped && !skipAll) await sleep(120);
    }
    skipped = false;
    titleAnswer = null;
    inTitle = true;
    await beatTitle(skipAll || cinema.REDUCED);
    inTitle = false;
    if (waitAtTitle && process.stdin.isTTY) {
      const raw = titleAnswer || await nextKey();
      key = raw === '\r' || raw === '\n' || raw === ' ' ? 'enter' : raw === 'q' || raw === '\x1b' ? 'q' : raw;
    }
  } finally {
    disarmSkip();
    // with keepScreen the guard stays on until whatever takes the screen next has its own
    if (!keepScreen) { guardSignals(false); screenTaken = false; process.stdout.write(SCREEN_OFF); }
  }
  return key;
}

async function beatQuickStamp(done) {
  const width = Math.min(W(), 110);
  const top = Math.max(2, Math.floor(H() / 2) - 8);
  for (let t = 0; t < 7 && !skipped && !done(); t++) {
    const lines = [''];
    lines.push(center(STUDENT[1], width));
    lines.push(center(STUDENT[2], width));
    lines.push(center(STUDENT[3], width));
    lines.push(center(STUDENT[4], width));
    lines.push('');
    if (t >= 3) lines.push(...BULB.slice(0, 6).map((l) => center(l.replace(/[╲╳╱]/g, t >= 5 ? (m) => m : ' '), width)));
    if (t >= 6) lines.push(center(DIM + 'nine months in' + R, width));
    frame(lines, { y0: top });
    await sleep(115);
  }
}

async function beatQuickRise(done) {
  const width = Math.min(W(), 110);
  const top = Math.max(2, Math.floor(H() / 2) - 6);
  const starHistory = [176, 336, 475, 587, 649];
  for (let t = 0; t < 8 && !skipped && !done(); t++) {
    const stars = starHistory[Math.min(starHistory.length - 1, Math.floor(t / 2))];
    const rel = RELEASES[Math.min(RELEASES.length - 1, Math.floor(t / 1.2))];
    const lines = [''];
    lines.push(center(`${rgb(120, 220, 180)}${rel}${R}   ${AMBER}★${R} ${BOLD}${stars}${R}`, width));
    lines.push(center('▁▂▃▄▅▆▇█'.slice(0, Math.min(8, t + 1)), width));
    lines.push('');
    lines.push(...Array.from({ length: 3 }, (_, i) => center(i === 1 ? '✦' : '·', width)));
    frame(lines, { y0: top });
    await sleep(110);
  }
}

async function beatQuickGlitch(done, startedAt, maxMs) {
  const width = Math.min(W(), 110);
  const top = Math.max(2, Math.floor(H() / 2) - 7);
  const pool = '!<>-_\\/[]{}=+*^?#01';
  let f = 0;
  while (!skipped) {
    const elapsed = Date.now() - startedAt;
    const workDone = done();
    if (workDone && f < LOGO.length * 4) f = LOGO.length * 4;
    if (f > LOGO.length * 4 + 6 || elapsed > maxMs) break;
    const lines = [''];
    for (const row of LOGO) {
      let out = '';
      for (let i = 0; i < row.length; i++) {
        const settle = f - i * 0.35;
        out += settle > 8 ? row[i] : (Math.random() < 0.5 ? pool[(Math.random() * pool.length) | 0] : ' ');
      }
      lines.push(center(cinema.gradient(out, [90, 120, 255], [190, 80, 255], f), width));
    }
    frame(lines, { y0: top });
    f++;
    await sleep(58);
  }
  const finalLines = ['', ...LOGO.map((l) => center(cinema.gradient(l, [90, 120, 255], [190, 80, 255]), width)), ''];
  finalLines.push(center(BOLD + 'the causal memory for every agent' + R, width));
  frame(finalLines, { y0: top });
}

async function playQuickIntro(work, opts = {}) {
  const workFn = typeof work === 'function' ? work : () => work;
  if (!process.stdout.isTTY || cinema.REDUCED) return workFn();
  const maxMs = opts.maxMs || 4500;
  const startedAt = Date.now();
  let workDone = false;
  let workResult;
  let workError;
  const armed = armSkip();
  process.stdout.write('\x1b[?1049h\x1b[?25l\x1b[2J');
  let workLaunched = false;
  const launchWork = () => {
    if (workLaunched) return;
    workLaunched = true;
    (async () => { try { workResult = await workFn(); } catch (error) { workError = error; } workDone = true; })();
  };
  try {
    await beatQuickStamp(() => workDone);
    launchWork();
    await beatQuickRise(() => workDone);
    if (!workLaunched) launchWork();
    await beatQuickGlitch(() => workDone, startedAt, maxMs);
    while (!workDone && Date.now() - startedAt < 30000 && !skipped) await sleep(120);
  } finally {
    disarmSkip(armed);
    process.stdout.write('\x1b[0m\x1b[?25h\x1b[?1049l');
  }
  if (workError) throw workError;
  return workResult;
}

// "press ENTER to begin your journey": enter really begins it, the v5 Precedent journey.
async function standalone(argv = process.argv.slice(2)) {
  const key = await playIntro({ skipToEnd: argv.includes('--still'), keepScreen: true });
  if (key === 'enter' && process.stdin.isTTY) {
    let journey = null;
    try { journey = require('./vestige-journey.js'); } catch {}
    if (process.env.VESTIGE_ENTRY) process.env.VESTIGE_ENTRY = process.env.VESTIGE_ENTRY.replace(/ intro$/, ' precedent');
    // the journey takes the screen and its signals synchronously inside main(); only then does the
    // intro let go of its own guard, so a ctrl-c while the journey loads still restores the terminal
    if (journey) { try { return journey.main([]); } finally { guardSignals(false); screenTaken = false; } }
  }
  guardSignals(false);
  screenTaken = false;
  process.stdout.write(SCREEN_OFF);
  if (process.stdin.isTTY) {
    const via = /^npx /.test(process.env.VESTIGE_ENTRY || '') ? ['npx @vestige/init intro', 'npx @vestige/init precedent'] : ['node vestige-intro.js', 'node vestige-journey.js'];
    process.stdout.write('\n  ' + DIM + `watch again: ${via[0]} · the journey: ${via[1]}` + R + '\n');
  }
  return 0;
}

if (require.main === module) {
  standalone().then((code) => process.exit(code || 0)).catch((error) => { process.stdout.write('\x1b[0m\x1b[?25h\x1b[?1049l'); console.error('vestige intro: ' + error.message); process.exit(1); });
}

module.exports = { playIntro, playQuickIntro, standalone, SCREEN_OFF };
