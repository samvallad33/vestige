# Vestige Agent Black Box — Proof Pack (2026-06-22)

> **Public claim (frozen):** Vestige records real MCP memory activity into a
> replayable local trace, with receipts and reviewable risky writes.
>
> We do **not** claim Sanhedrin vetoes or dream patches are live by default.
> Those producers are optional and off by default — the UI says so explicitly.

This pack is captured from a **live** Vestige build on branch
`feat/agent-black-box` — a real `vestige-mcp` process with the dashboard
enabled, driven by real MCP `tools/call` traffic. Nothing here is mocked.

## The receipt chain — one runId, every hop

The money guarantee: a single `runId` (`run_proof`) crosses every layer,
byte-identical. Verified two ways — by the files in this folder, and by the
deterministic regression test `test_full_spine_one_runid_crosses_every_hop`
(crates/vestige-mcp/src/server.rs).

| Hop | Layer | Evidence in this pack |
|----|-------|------|
| 1 | MCP tool output (`runId` + `traceUri`) | every tool result; see test HOP 1 |
| 2 | SQLite `agent_traces` rows | `trace.json` (`runId: run_proof`, 10 events) |
| 3 | WebSocket broadcast | `websocket-events.jsonl` (6 `TraceEvent` lines, each with `run_id`) |
| 4 | `/api/traces/:runId` response | `trace.json` is the export of that endpoint |
| 5 | dashboard render | screenshots (Black Box timeline = the 10 events) |
| 6 | `vestige://trace/{runId}` MCP resource | test HOP 5 resolves the same id |

## Files

| File | What it proves |
|------|----------------|
| `status.json` | the live server health at capture time |
| `trace.json` | the full `.vestige-trace.json` export — 10 real events in order |
| `receipt.json` | a real retrieval receipt (`r_2026_06_22_runproof`, 5 retrieved, decay medium) |
| `memory_pr.json` | the risky auth write → Memory PR, **promoted** through UI→API→SQLite, signal `sensitive_topic` |
| `websocket-events.jsonl` | the live WS stream: `TraceEvent`×6, `MemoryPrOpened`, `MemoryPrDecided`, `MemoryCreated`, `MemoryUpdated` |
| `screenshots/` | Graph, Black Box, Receipts (in PR), Memory PRs — see `screenshots/README.md` |

## Per-feature honesty: real / caveat / stub

| Feature | Status | Notes |
|---------|--------|-------|
| `mcp.call` trace | **REAL** | every tools/call records one; args **hashed**, never stored raw |
| `memory.write` trace | **REAL** | fires on smart_ingest/ingest, memory promote/demote/edit, codebase remember_*, AND destructive purge/delete |
| `memory.retrieve` trace | **REAL** | fires on deep_reference/search, with per-id activation |
| `memory.suppress` trace | **REAL** | recorded path; fires when retrieval suppresses |
| `contradiction.detected` trace | **REAL** | fires when deep_reference surfaces a contradiction pair; UI says "no contradiction in this run" when none |
| Memory Receipts | **REAL** | built from real scored memories + trust, persisted, attached to output |
| Risk-gated Memory PRs | **REAL** | quarantine review: commit-then-suppress, audit preserved, influence suspended. Promote verified end-to-end (releases the memory, even past the 24h window). Destructive purge/delete also open a PR. PR content is **redacted** for sensitive writes (preview + hash, never the raw secret) |
| Fast / Risk-Gated / Paranoid modes | **REAL** | persisted to `<data_dir>/review_mode.json`; Risk-Gated is the default |
| WebSocket broadcast | **REAL** | proven by `websocket-events.jsonl` + a unit test |
| `vestige://trace/{runId}` resource | **REAL** | proven by the full-spine test |
| `sanhedrin.veto` trace | **CAVEAT** | extraction code is real + unit-tested, but the Sanhedrin verifier is an optional hook, **off by default** — no producer is connected, and the UI says exactly that |
| `dream.patch` trace | **REAL** (proven 2026-06-22) | a real `dream` run over 6 memories produced one `dream.patch` event under `run_dream_proof` — see `dream-trace.json` (last event), `dream-websocket-events.jsonl`, and `screenshots/dream-producers.png` where the row flips to "fired this run". The UI still shows "No dream run in this trace" for runs where no dream executed. |
| Graph-pulse "Open receipt in Cinema" | **REAL (deep-link)** | navigates the graph centered on the receipt's primary memory; MemoryCinema itself is unchanged |

No feature is stubbed. The two CAVEATs are real plumbing whose upstream
producer is intentionally off by default — surfaced as explicit UI states, not
empty mystery.

## dream.patch — proven with a real dream run (2026-06-22)

Bounded follow-up: a single real `dream` consolidation flipped the `dream.patch`
producer from "quiet" to a recorded live event, same runId, every hop.

- 6 related memories seeded under `run_dream_proof`, then one `dream` call.
- The dream produced one consolidation insight → one `dream.patch` event:
  `dream:RecurringPattern:5d941c7f+a41aca72+b029fe53+6167f2c3+1117dd4e+e0782442`
  (the real insight type + the six source memories it bridged).
- SQLite: `dream-trace.json` (14 events, last is `dream.patch`).
- API: `/api/traces/run_dream_proof/export` → `dream-trace.json`.
- WebSocket: `dream-websocket-events.jsonl` (the `dream.patch` TraceEvent).
- Dashboard: `screenshots/black-box-dream.png` + `screenshots/dream-producers.png`
  (the producers row shows **dream.patch · fired this run**).

`dream.patch` is real but not live-by-default: it fires only when a dream
actually runs. The UI says so for runs where it didn't.

## Reproduce

1. `VESTIGE_DATA_DIR=<tmp> VESTIGE_DASHBOARD_ENABLED=true vestige-mcp` (stdio).
2. `initialize`, then drive `smart_ingest` / `deep_reference` calls with a
   `runId` argument.
3. A sensitive-topic write (auth/security/money/identity/…) opens a Memory PR.
4. `curl /api/traces/<runId>/export` → the `.vestige-trace.json`.
5. `cargo test -p vestige-mcp test_full_spine_one_runid_crosses_every_hop`.
