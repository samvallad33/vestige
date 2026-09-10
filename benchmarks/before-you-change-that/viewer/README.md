# Before You Change That — public recorded-event replay

This viewer replays the sanitized events from the preserved Demo 4 run. It does
not call a model, rerun the benchmark, or use the private screen recordings.
Every event remains available in original sequence with a recorded offset,
human-readable output, and the complete sanitized JSON row.
`../evidence/recording-timing.json` binds those offsets to the original capture
start while keeping preparation events visible at time zero.

The result boundary matters: all three arms finished with application score
21/21. Control and Vestige completed naturally. MCP Memory Service produced the
passing application files, but its recorded model process timed out at 900
seconds. The target-context intention was returned on the Vestige path, and the
coordinator marked it complete only after independent application tests passed.
Timing does not establish a product ranking.

## Open the viewer

From `benchmarks/before-you-change-that`:

```sh
node viewer/serve.mjs --port 4173
```

Then open `http://127.0.0.1:4173/viewer/`. Use the comparison buttons to switch
between Control vs Vestige and Memory Service vs Vestige. Play at original
speed, accelerate the replay, seek anywhere in the 15-minute timeline, filter
events, expand complete narration/tool output, or download the raw evidence.

## Verify

```sh
node --test viewer/*.test.mjs
```

## Render public MP4s

The renderer takes deterministic browser snapshots at every recorded event
offset, then uses FFmpeg frame durations to preserve the complete timeline and
waiting periods without waiting 15 minutes during capture. It writes H.264 MP4
files with no audio and refuses to overwrite existing outputs.

```sh
node viewer/render-video.mjs \
  --all \
  --out-dir "/path/to/public-assets" \
  --output-tag public-final
```

The renderer discovers Playwright from the current Node environment or the
Codex bundled runtime. Set `PLAYWRIGHT_PATH`, `FFMPEG`, or `FFPROBE` when those
tools live elsewhere.

Full-decode the videos, compare their declared evidence hashes with the current
package, extract opening/end screenshots, and write a create-only publication
receipt:

```sh
node viewer/verify-media.mjs \
  --out-dir "/path/to/public-assets" \
  --output-tag public-final
```
