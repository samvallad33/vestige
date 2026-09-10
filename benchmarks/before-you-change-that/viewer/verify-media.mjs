#!/usr/bin/env node
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {createReadStream} from 'node:fs';
import {readFile, stat, writeFile} from 'node:fs/promises';
import {basename, dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {promisify} from 'node:util';
import {ARMS, resultReports, runStatus, usageTotals} from './replay-core.mjs';

const execute = promisify(execFile);
const here = dirname(fileURLToPath(import.meta.url));
const evidenceDirectory = resolve(here, '../evidence');

function parseArgs(argv) {
  const parsed = {outDir: null, outputTag: ''};
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--out-dir' && argv[index + 1]) parsed.outDir = resolve(argv[++index]);
    else if (argv[index] === '--output-tag' && argv[index + 1]) parsed.outputTag = argv[++index];
    else throw new Error('usage: node verify-media.mjs --out-dir DIR [--output-tag public-final]');
  }
  if (!parsed.outDir) throw new Error('--out-dir is required');
  if (parsed.outputTag && !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(parsed.outputTag)) throw new Error('Invalid output tag');
  return parsed;
}

async function sha256(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

async function exists(path) {
  try { await stat(path); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

async function ensureScreenshots(video) {
  const base = video.replace(/\.mp4$/, '');
  const opening = `${base}-t0.jpg`;
  const ending = `${base}-end.jpg`;
  if (!await exists(opening)) {
    await execute(process.env.FFMPEG || 'ffmpeg', [
      '-hide_banner', '-loglevel', 'error', '-n', '-i', video, '-frames:v', '1', '-q:v', '2', opening,
    ]);
  }
  if (!await exists(ending)) {
    await execute(process.env.FFMPEG || 'ffmpeg', [
      '-hide_banner', '-loglevel', 'error', '-n', '-sseof', '-5', '-i', video, '-frames:v', '1', '-q:v', '2', ending,
    ]);
  }
  return {
    opening: {file: basename(opening), sha256: await sha256(opening)},
    ending: {file: basename(ending), sha256: await sha256(ending)},
  };
}

const args = parseArgs(process.argv.slice(2));
const tag = args.outputTag ? `-${args.outputTag}` : '';
const renderManifestPath = resolve(args.outDir, `before-you-change-that-video-manifest${tag}.json`);
const renderManifest = JSON.parse(await readFile(renderManifestPath, 'utf8'));
const result = JSON.parse(await readFile(resolve(evidenceDirectory, 'result.json'), 'utf8'));

const currentEvidenceHashes = {};
for (const name of Object.keys(renderManifest.evidence_sha256)) {
  currentEvidenceHashes[name] = await sha256(resolve(evidenceDirectory, name));
  if (currentEvidenceHashes[name] !== renderManifest.evidence_sha256[name]) {
    throw new Error(`${name} changed after rendering`);
  }
}

const verifiedVideos = [];
for (const declared of renderManifest.outputs) {
  const video = resolve(args.outDir, declared.file);
  const actualHash = await sha256(video);
  if (actualHash !== declared.sha256) throw new Error(`${declared.file} hash differs from render manifest`);
  await execute(process.env.FFMPEG || 'ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', video, '-f', 'null', '-']);
  const probe = JSON.parse((await execute(process.env.FFPROBE || 'ffprobe', [
    '-v', 'error', '-count_frames', '-show_entries',
    'format=duration,size:stream=index,codec_name,width,height,avg_frame_rate,nb_read_frames',
    '-of', 'json', video,
  ])).stdout);
  if (probe.streams.length !== 1 || probe.streams[0].codec_name !== 'h264') throw new Error(`${declared.file} is not single-stream H.264`);
  if (probe.streams[0].width !== 1920 || probe.streams[0].height !== 1080) throw new Error(`${declared.file} dimensions differ`);
  if (Number(probe.format.duration) < 925) throw new Error(`${declared.file} does not retain the full recorded wait`);
  verifiedVideos.push({
    file: declared.file,
    sha256: actualHash,
    full_decode_pass: true,
    duration_seconds: Number(probe.format.duration),
    size_bytes: Number(probe.format.size),
    video_stream: probe.streams[0],
    recorded_event_counts: declared.recorded_event_counts,
    frame_states: declared.frame_states,
    screenshots: await ensureScreenshots(video),
  });
}

const reports = resultReports(result);
const counters = Object.fromEntries(ARMS.map((arm) => [arm, usageTotals(reports[arm])]));
const receiptPath = resolve(args.outDir, `before-you-change-that-publication-receipt${tag}.json`);
const receipt = {
  schema: 'vestige.before-you-change-that.publication-receipt.v1',
  boundary: 'Recorded-event replay; paths redacted; no new model run. The Memory Service model process timed out; its application files later passed all checks. Timing does not establish a product ranking.',
  evidence_sha256: currentEvidenceHashes,
  application_scores: Object.fromEntries(ARMS.map((arm) => [arm, {
    score: reports[arm]?.evaluation?.score,
    total: reports[arm]?.evaluation?.total,
  }])),
  completion_status: Object.fromEntries(ARMS.map((arm) => [arm, runStatus(reports[arm])])),
  token_counters: counters,
  intention: {
    returned_in_target_context: Boolean(reports.vestige?.feature_observed?.context_intention?.observed),
    coordinator_completed_after_application_pass: Boolean(
      reports.vestige?.intention_lifecycle?.application_passed
      && reports.vestige?.intention_lifecycle?.stored_after?.status === 'fulfilled',
    ),
  },
  video_verification: verifiedVideos,
};
await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, {flag: 'wx'});
console.log(`VERIFIED_RECEIPT=${receiptPath}`);
