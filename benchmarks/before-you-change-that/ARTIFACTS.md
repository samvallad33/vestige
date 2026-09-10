# Download the evidence and videos

[Open the benchmark evidence release](https://github.com/samvallad33/vestige/releases/tag/benchmark-before-you-change-that-2026-09-06).

- **Full evidence bundle:** `before-you-change-that-evidence.tar.gz` contains
  the 643 public evidence files, historical harness, frozen inputs, recorded
  projects, transcripts, and public verification/viewer tools.
- **Control and Vestige:** `before-you-change-that-control-vs-vestige-public-publish.mp4`.
- **MCP Memory Service and Vestige:** `before-you-change-that-memory-service-vs-vestige-public-publish.mp4`.
- **Media provenance and verification:** the `video-manifest` and
  `publication-receipt` JSON assets bind the videos to the recorded events,
  outcomes, and token counters. Opening and closing screenshots are included.

The MP4s are labeled replays rendered from the complete retained event timeline.
They preserve its waits and the historical Memory Service interruption. They
are not the original live screen recordings, and no model was run to make them.

Each downloaded asset is listed in `SHA256SUMS` on the release and in the
[versioned artifact index](https://github.com/samvallad33/vestige/blob/benchmark-before-you-change-that-2026-09-06/benchmarks/before-you-change-that/ARTIFACTS.json).
Verify downloads on macOS with `shasum -a 256 -c SHA256SUMS` (or
`sha256sum -c SHA256SUMS` on Linux), after downloading all listed assets.
The checksum establishes agreement with that index, not an external trust or
timestamp authority.

Extract the evidence archive and enter its `before-you-change-that/` directory.
Follow the README to install the signature verifier, then run
`python verify.py --full-evidence`. The application recheck additionally
requires macOS and Rust. The saved application outputs can be inspected on any
platform without executing them.

This is a benchmark evidence release. It does not change the product version
or publish a new MCP server binary.
