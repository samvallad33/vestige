# Agent Black Box — Proof of Life (2026-06-22)

> Watch the agent think. Watch memory change. Watch the receipt prove why.

This folder is the launch artifact + regression evidence for the Agent Black Box,
Memory Receipts, and risk-gated Memory PRs, captured from a **live** Vestige
build (`feat/agent-black-box`), not mocks.

## The trace correlation spine (Phase 0) — verified end to end

A single `runId` (`run_proof_session`) threads, unbroken, through every layer:

| Hop | Layer | Evidence |
|----|-------|----------|
| 1 | MCP tool output | every `tools/call` result carries `runId` + `traceUri` (`vestige://trace/{runId}`) |
| 2 | SQLite trace rows | 12 `agent_traces` rows persisted under the runId |
| 3 | WebSocket | each event broadcast as `VestigeEvent::TraceEvent` |
| 4 | dashboard pulse | Black Box tab renders 12 ticks + memory pulse, live |
| 5 | `/api/traces/:runId` | see `phase-3-trace.json` |
| 6 | `vestige://trace/{runId}` | MCP resource resolves the same run |
| 7 | receipt export | `phase-3-trace.json` is the downloadable `.vestige-trace.json` |
| 8 | Cinema replay | "Open receipt in Cinema" deep-links the receipt's memory set |

## What the run did (12 events, in order)

`mcp.call → memory.write` × 3 ordinary writes (auto-landed),
`mcp.call → memory.retrieve` × 2 (deep_reference + search, each left a receipt),
`mcp.call → memory.write` × 1 **risky** write (auth/security content).

## The cognitive immune system fired

- Mode: **Risk-Gated** (the default).
- The 3 ordinary writes **auto-landed** — no friction.
- The 1 risky write (auth token / security credential) **opened a Memory PR**
  with the self-explaining signal `sensitive_topic → "Touches a sensitive
  topic: authentication / authorization."`
- Promoting that PR from the dashboard moved it to `promoted` through the full
  stack (UI → API → SQLite). See `memory-prs.json`.

This is the product line, made literal:
**Vestige auto-remembers ordinary context, but opens a Memory PR when the agent
tries to rewrite its own brain.**

## Files

- `phase-1-status.json` — server health (spine alive).
- `phase-3-trace.json` — the full `.vestige-trace.json` export (the black box).
- `receipts.json` — the retrieval receipt(s) generated this run.
- `memory-prs.json` — the Memory PR queue, including the promoted risky write.

## Gates (all green)

- `cargo test --workspace` — 953 lib tests pass (incl. the trace-spine
  integration test driving a real JSON-RPC round-trip).
- `cargo clippy --workspace -- -D warnings` — 0 warnings.
- `pnpm --filter @vestige/dashboard check` — 0 errors, 0 warnings (905 files).
- `pnpm --filter @vestige/dashboard build` — clean.
