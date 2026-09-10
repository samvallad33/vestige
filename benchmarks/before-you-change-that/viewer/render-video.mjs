#!/usr/bin/env node
import {createHash} from 'node:crypto';
import {execFile, execFileSync} from 'node:child_process';
import {createReadStream} from 'node:fs';
import {mkdtemp, mkdir, rm, stat, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {basename, dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {promisify} from 'node:util';
import {browserLaunchOptions, loadPlaywright} from './browser-runtime.mjs';
import {startServer} from './serve.mjs';

const execute = promisify(execFile);
const here = dirname(fileURLToPath(import.meta.url));
const evidenceDirectory = resolve(here, '../evidence');
const DEFAULT_HOLD_SECONDS = 20;

function parseArgs(argv) {
  const parsed = {outDir: null, all: false, comparison: null, holdSeconds: DEFAULT_HOLD_SECONDS, outputTag: ''};
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === '--out-dir' && argv[index + 1]) parsed.outDir = resolve(argv[++index]);
    else if (value === '--all') parsed.all = true;
    else if (value === '--comparison' && argv[index + 1]) parsed.comparison = argv[++index];
    else if (value === '--hold-seconds' && argv[index + 1]) parsed.holdSeconds = Number(argv[++index]);
    else if (value === '--output-tag' && argv[index + 1]) parsed.outputTag = argv[++index];
    else throw new Error('usage: node render-video.mjs --out-dir DIR (--all | --comparison control|mcp-memory-service) [--hold-seconds 20] [--output-tag public-final]');
  }
  if (!parsed.outDir) throw new Error('--out-dir is required');
  if (parsed.all === Boolean(parsed.comparison)) throw new Error('Choose exactly one of --all or --comparison');
  if (parsed.comparison && !['control', 'mcp-memory-service'].includes(parsed.comparison)) throw new Error('Invalid comparison');
  if (!Number.isFinite(parsed.holdSeconds) || parsed.holdSeconds < 1) throw new Error('Invalid hold duration');
  if (parsed.outputTag && !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(parsed.outputTag)) throw new Error('Invalid output tag');
  return parsed;
}

async function sha256(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

function ffmpegPath() {
  const command = process.env.FFMPEG || 'ffmpeg';
  execFileSync(command, ['-version'], {stdio: 'ignore'});
  return command;
}

function concatQuote(path) {
  return `'${path.replaceAll("'", "'\\''")}'`;
}

async function reserveOutput(path) {
  try {
    await stat(path);
    throw new Error(`Refusing to overwrite existing output: ${path}`);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
}

async function renderComparison({browser, origin, comparison, outDir, holdSeconds, outputTag, ffmpeg}) {
  const slug = comparison === 'control' ? 'control-vs-vestige' : 'memory-service-vs-vestige';
  const tagged = outputTag ? `-${outputTag}` : '';
  const output = resolve(outDir, `before-you-change-that-${slug}${tagged}.mp4`);
  await reserveOutput(output);
  const frames = await mkdtemp(join(tmpdir(), `before-you-change-that-${slug}-`));
  const page = await browser.newPage({viewport: {width: 1920, height: 1080}, colorScheme: 'dark'});
  const errors = [];
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('pageerror', (error) => errors.push(error.message));
  try {
    await page.goto(`${origin}/viewer/?render=1&compare=${encodeURIComponent(comparison)}`, {waitUntil: 'networkidle'});
    await page.waitForFunction(() => document.documentElement.dataset.replayReady === 'true');
    const replay = await page.evaluate(() => ({
      duration: window.__PUBLIC_REPLAY__.duration,
      changeTimes: window.__PUBLIC_REPLAY__.changeTimes,
      counts: window.__PUBLIC_REPLAY__.counts,
      visibleArms: window.__PUBLIC_REPLAY__.visibleArms,
    }));
    const timeline = [...replay.changeTimes.filter((time) => time < replay.duration), replay.duration];
    const framePaths = [];
    for (const [index, time] of timeline.entries()) {
      await page.evaluate((seconds) => window.__PUBLIC_REPLAY__.seek(seconds), time);
      const frame = resolve(frames, `${String(index).padStart(4, '0')}.jpg`);
      await page.screenshot({path: frame, type: 'jpeg', quality: 92, animations: 'disabled'});
      framePaths.push(frame);
    }
    await page.evaluate(() => window.__PUBLIC_REPLAY__.showEnd());
    const endFrame = resolve(frames, `${String(timeline.length).padStart(4, '0')}-end.jpg`);
    await page.screenshot({path: endFrame, type: 'jpeg', quality: 94, animations: 'disabled'});
    framePaths.push(endFrame);

    const lines = [];
    for (let index = 0; index < timeline.length; index += 1) {
      const duration = index + 1 < timeline.length ? timeline[index + 1] - timeline[index] : 0.5;
      lines.push(`file ${concatQuote(framePaths[index])}`, `duration ${Math.max(0.04, duration).toFixed(6)}`);
    }
    lines.push(`file ${concatQuote(endFrame)}`, `duration ${holdSeconds.toFixed(6)}`, `file ${concatQuote(endFrame)}`);
    const concatFile = resolve(frames, 'timeline.ffconcat');
    await writeFile(concatFile, `ffconcat version 1.0\n${lines.join('\n')}\n`, {flag: 'wx'});

    await execute(ffmpeg, [
      '-hide_banner', '-loglevel', 'warning', '-n', '-f', 'concat', '-safe', '0', '-i', concatFile,
      '-an', '-fps_mode', 'vfr', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
      '-g', '1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-video_track_timescale', '90000',
      '-metadata', 'title=Before You Change That — recorded-event replay',
      '-metadata', 'comment=Paths redacted; no new model run; timing does not establish a product ranking.',
      output,
    ], {maxBuffer: 16 * 1024 * 1024});
    if (errors.length) throw new Error(`Browser errors while rendering: ${errors.join(' | ')}`);
    const probe = JSON.parse((await execute(process.env.FFPROBE || 'ffprobe', [
      '-v', 'error', '-show_entries', 'format=duration,size:stream=codec_name,width,height', '-of', 'json', output,
    ])).stdout);
    return {
      comparison,
      file: basename(output),
      sha256: await sha256(output),
      duration_seconds: Number(probe.format.duration),
      size_bytes: Number(probe.format.size),
      stream: probe.streams[0],
      recorded_event_counts: replay.counts,
      frame_states: framePaths.length,
      end_card_hold_seconds: holdSeconds,
      visible_arms: replay.visibleArms,
    };
  } finally {
    await page.close();
    await rm(frames, {recursive: true, force: true});
  }
}

const args = parseArgs(process.argv.slice(2));
await mkdir(args.outDir, {recursive: true});
const ffmpeg = ffmpegPath();
execFileSync(process.env.FFPROBE || 'ffprobe', ['-version'], {stdio: 'ignore'});
const {chromium} = loadPlaywright();
const server = await startServer();
const address = server.address();
const origin = `http://127.0.0.1:${String(address.port)}`;
let browser;
try {
  browser = await chromium.launch(browserLaunchOptions(chromium, {args: ['--hide-scrollbars', '--force-color-profile=srgb']}));
  const comparisons = args.all ? ['control', 'mcp-memory-service'] : [args.comparison];
  const outputs = [];
  for (const comparison of comparisons) {
    outputs.push(await renderComparison({browser, origin, comparison, outDir: args.outDir, holdSeconds: args.holdSeconds, outputTag: args.outputTag, ffmpeg}));
    console.log(`RENDERED=${resolve(args.outDir, outputs.at(-1).file)}`);
  }
  if (args.all) {
    const evidence = {};
    for (const name of ['manifest.json', 'recording-timing.json', 'result.json', 'control.jsonl', 'mcp-memory-service.jsonl', 'vestige.jsonl']) {
      evidence[name] = await sha256(resolve(evidenceDirectory, name));
    }
    const manifestTag = args.outputTag ? `-${args.outputTag}` : '';
    const manifestPath = resolve(args.outDir, `before-you-change-that-video-manifest${manifestTag}.json`);
    await reserveOutput(manifestPath);
    await writeFile(manifestPath, `${JSON.stringify({
      schema: 'vestige.before-you-change-that.public-video.v1',
      boundary: 'Recorded-event replay; paths redacted; no new model run. Timing does not establish a product ranking.',
      dimensions: [1920, 1080],
      audio: 'none',
      evidence_sha256: evidence,
      outputs,
    }, null, 2)}\n`, {flag: 'wx'});
    console.log(`MANIFEST=${manifestPath}`);
  }
} finally {
  if (browser) await browser.close();
  await new Promise((resolveClose) => server.close(resolveClose));
}
