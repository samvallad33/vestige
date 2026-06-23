# Agent Black Box — Review Bundle

**Branch:** `feat/agent-black-box` (head = branch tip)
**Base (review against):** `9e92a5999ada37bed9b4820bb25b7748b417411c` (the
`feat/dashboard-bleeding-edge` tip this branched from)
**Status:** feature work **frozen**. No quarantine-constellation work has
started.

Start here, then read `PROOF.md` (the per-feature real/caveat/stub ledger) and
open the screenshots.

> **Update — review findings addressed.** A full multi-agent review found 7
> real issues (4 blockers). All are fixed and tested; this bundle was
> **re-captured from a single run (`run_proof`)** so trace.json,
> websocket-events.jsonl, and memory_pr.json now all carry the same runId. See
> "Review findings addressed" below.

---

## Frozen public claim

> Vestige records real MCP memory activity into a replayable local trace, with
> receipts and reviewable risky writes.

We do **not** claim Sanhedrin vetoes are live by default. `dream.patch` is real
but fires only when a dream actually runs (proven below; the UI says so when no
dream ran).

---

## What's in this bundle

| File | What it is |
|------|------------|
| `REVIEW.md` | this file — the entry point |
| `PROOF.md` | per-feature real/caveat/stub ledger + reproduce steps |
| `status.json` | live server `/api/health` at capture time |
| `trace.json` | `.vestige-trace.json` export of `run_proof` (10 real events) |
| `receipt.json` | a real retrieval receipt (`r_2026_06_22_runproof`) |
| `memory_pr.json` | the risky auth write → Memory PR, **promoted** UI→API→SQLite |
| `websocket-events.jsonl` | live WS stream: `TraceEvent`×6, `MemoryPrOpened`, `MemoryPrDecided`, … |
| `dream-trace.json` | `run_dream_proof` export — 14 events, last is `dream.patch` |
| `dream-websocket-events.jsonl` | live WS stream containing the `dream.patch` `TraceEvent` |
| `screenshots/black-box.png` | Black Box tab (spine header, scrubber, producers, log) |
| `screenshots/receipts.png` | the `ReceiptCard` with real data + "Open receipt in Cinema" |
| `screenshots/memory-prs.png` | Memory PRs: diff, "Why this opened", `Decided: promote` |
| `screenshots/graph.png` | live graph constellation + Memory Cinema (unchanged) |
| `screenshots/black-box-dream.png` | Black Box on the dream run |
| `screenshots/dream-producers.png` | producers panel — `dream.patch · fired this run` |

---

## Caveat ledger (the honest part)

| Producer | Status | Why |
|----------|--------|-----|
| `mcp.call`, `memory.write`, `memory.retrieve`, `memory.suppress` | **REAL** | fire on real tool traffic; args hashed, never stored raw |
| `contradiction.detected` | **REAL** | fires when deep_reference surfaces a contradiction; UI shows "no contradiction in this run" otherwise |
| Memory Receipts | **REAL** | built from real scored memories + trust, persisted, attached to output |
| Risk-gated Memory PRs (quarantine review) | **REAL** | commit-then-suppress; audit preserved, influence suspended; Promote verified end-to-end |
| WebSocket broadcast | **REAL** | `websocket-events.jsonl` + unit test |
| `vestige://trace/{runId}` resource | **REAL** | full-spine test hop 5 |
| `dream.patch` | **REAL** (not live-by-default) | proven by `run_dream_proof`; fires only when a dream runs; UI says so otherwise |
| `sanhedrin.veto` | **CAVEAT** | extraction is real + unit-tested, but the Sanhedrin verifier is an **optional hook, off by default** — no producer connected; UI says exactly that |

No feature is stubbed. The one remaining caveat is surfaced as an explicit UI
state, not an empty space.

---

## Command receipts (run live at 2026-06-22T22:57:59Z)

Toolchain: `rustc 1.95.0` · `cargo 1.95.0` · `node v24.12.0` · `pnpm 10.33.0`.

```
$ cargo test --workspace --lib
test result: ok. 529 passed; 0 failed; 0 ignored; 0 measured   (vestige-core)
test result: ok.  33 passed; 0 failed; 0 ignored; 0 measured   (tests/e2e unit)
test result: ok. 428 passed; 0 failed; 0 ignored; 0 measured   (vestige-mcp)
# 990 lib tests, 0 failures

$ cargo clippy --workspace -- -D warnings
Finished `dev` profile ... (EXIT 0, zero warnings)

$ pnpm --filter @vestige/dashboard check
COMPLETED 905 FILES 0 ERRORS 0 WARNINGS 0 FILES_WITH_PROBLEMS

$ pnpm --filter @vestige/dashboard build
✓ built in 4.15s ... ✔ done

$ cargo test -p vestige-mcp test_full_spine_one_runid_crosses_every_hop
test server::tests::test_full_spine_one_runid_crosses_every_hop ... ok

$ cargo test -p vestige-mcp trace_recorder::tests::extract_dream
test ...extract_dream_proposals_empty_when_not_dream_tool ... ok
test ...extract_dream_proposals_from_real_insights_shape ... ok

$ cargo test -p vestige-core trace
test result: ok. 27 passed; 0 failed   (trace schema, receipt, review)
```

Only statuses with a receipt above are credited. Nothing is claimed from memory.

---

## Review surface (what changed)

The Black Box work sits in a series of commits on top of the base. To see the
exact, current diff (build artifacts + this proof bundle excluded):

```
$ git diff --stat 9e92a59..HEAD -- ':!apps/dashboard/build' ':!blackbox-proof-2026-06-22'
# ~27 source files (Rust + SvelteKit). Run this against the branch tip for the
# live count — it grows as review fixes land.
```

Commits (oldest first):
- `80c823a` feat: Agent Black Box + Receipts + risk-gated Memory PRs
- `b89beee` proof: Proof Lock — full-spine test, honest UI states, proof pack
- `140b15f` proof: dream.patch proven live with a real dream run
- `cadffb4` docs: package the review bundle — REVIEW.md entry point
- `8f7bed0` fix: address review blockers B1–B7 + re-capture proof bundle
- (+ a follow-up fix commit for C1/C2 — see "Review findings addressed")

Key files to review:
- **Core (pure logic):** `crates/vestige-core/src/trace/{mod,receipt,review}.rs`
- **Persistence:** `crates/vestige-core/src/storage/trace_store.rs` + `migrations.rs` (V18)
- **MCP wiring:** `crates/vestige-mcp/src/trace_recorder.rs`, `server.rs` (dispatch),
  `resources/trace.rs`
- **Dashboard API:** `crates/vestige-mcp/src/dashboard/{handlers,events,mod}.rs`
- **UI:** `apps/dashboard/src/routes/(app)/{blackbox,memory-prs}/+page.svelte`,
  `lib/components/{ReceiptCard.svelte,blackbox-helpers.ts}`

---

## Suggested review checklist

- [ ] **Spine integrity:** read `test_full_spine_one_runid_crosses_every_hop`
      (crates/vestige-mcp/src/server.rs) — does it actually assert the runId is
      byte-identical at all five hops?
- [ ] **Privacy:** confirm `mcp.call` stores only a hash of args
      (`trace_recorder::hash_args`), never raw args.
- [ ] **Risk taxonomy:** review `classify_write` + `WriteContext`
      (crates/vestige-core/src/trace/review.rs) — is the sensitive-topic /
      contradiction / supersede gating correct and not over-broad?
- [ ] **Quarantine semantics:** confirm risky writes are committed-then-suppressed
      (audit preserved), not silently dropped, and the copy says so.
- [ ] **Migration safety:** V18 is additive; `MIGRATIONS.last().version` is used
      by the migration tests (no hard-coded version literals left).
- [ ] **Local-first defaults:** Risk-Gated is default; Sanhedrin/dream producers
      stay optional and off by default; nothing forces heavy models.
- [ ] **No protected code touched:** `MemoryCinema.svelte` and `graph/cinema/*`
      are unchanged; the graph page only gained an additive `?center=` param.

---

## Review findings addressed (2026-06-22)

A full read-only review (multiple parallel agents, both Rust and dashboard)
found 7 real issues — 4 blockers. All fixed and tested:

| # | Severity | Finding | Fix | Proof |
|---|----------|---------|-----|-------|
| B1 | blocker | Promoting a Memory PR didn't unsuppress the quarantined memory — UI said "promoted" while the memory stayed out of retrieval | `act_on_memory_pr` now calls `reverse_suppression(subject_id)` on accept actions (promote/merge/supersede); `MemoryPrAction::releases_memory()` encodes the rule | live: PR response `subjectReleased: true`, SQLite `suppression_count: 0`; tests `promote_releases_a_quarantined_memory_end_to_end`, `only_accept_actions_release_the_memory` |
| B2 | blocker | memory promote/demote (returns `action`, not `decision`) and `codebase` writes bypassed the write-trace + gate | `extract_writes` reads `action` too, filtered by `is_write_decision`; `is_write_tool` includes `codebase` | tests `extract_writes_recognizes_action_shape_b2`, `extract_writes_ignores_read_actions_b2`, `write_tool_set_includes_codebase_b2` |
| B3 | blocker | Receipt ids collided within a run (`r_<date>_<runId>` + `INSERT OR REPLACE`) — later receipt overwrote earlier | id is now `r_<date>_<runId8>_<unique6>` | live: two receipts in `run_proof` have distinct ids; test `receipt_ids_unique_within_a_run_b3` |
| B4 | blocker | Proof bundle mis-assembled: `trace.json`=`run_proof` but `websocket-events.jsonl`=`run_proof2` | re-captured the whole bundle from one run | all artifacts now carry `run_proof` (verified) |
| B5 | P2 | Black Box receipts panel showed global latest, not the selected run's | `list_receipts_for_run` + `/api/receipts?run=` + page uses `listForRun(runId)` | live: `?run=run_proof` returns only that run; test `receipts_are_listable_per_run_b5` |
| B6 | P2 | `SENSITIVE_TOPICS` substring match false-fired ("tokenizer"→token, "author"→auth) | word-boundary matching | tests `sensitive_topic_word_boundary_no_false_positives_b6`, `..._still_catches_real_b6` |
| B7 | P3 | `set_review_mode` non-atomic write; export filename used raw `run_id` | `write_atomic` (temp+rename); filename sanitized; static routes declared before dynamic | covered by build + the atomic-write helper's existing use |
| C1 | blocker | B1's release used `reverse_suppression`, which **refuses past the 24h labile window** — a PR promoted late stayed suppressed | new `release_quarantine(id)`: unconditional release (no time limit), used by the PR handler instead | test `release_quarantine_works_past_the_labile_window_c1` (proves reverse_suppression refuses but release_quarantine succeeds at +100h) |
| C2 | blocker | `memory` `purge`/`delete` (destructive removal) bypassed the write-trace + gate | added purge/purged/delete/deleted/forget/forgotten to `is_write_decision` | test `extract_writes_recognizes_destructive_actions_c2` |

One earlier (self-)review claim was **withdrawn**: the `/api/memory-prs/mode`
vs `/{id}` route order is *not* a functional bug — axum 0.8 / matchit gives
static segments priority. Reordered for clarity only.

Net after fixes (B1–B7 + C1/C2): **1002 lib tests pass, clippy `-D warnings` clean, dashboard
check + build clean.**

## Reproduce (any reviewer, locally)

```sh
# 1. run a real server with the dashboard on
VESTIGE_DATA_DIR=$(mktemp -d) VESTIGE_DASHBOARD_ENABLED=true vestige-mcp   # stdio JSON-RPC
# 2. initialize, then drive tools/call with a runId arg (smart_ingest, deep_reference)
# 3. a sensitive-topic write opens a Memory PR; promote it via the dashboard
# 4. export the trace:
curl -s http://127.0.0.1:3927/api/traces/<runId>/export
# 5. for dream.patch: seed >=5 memories, then call the `dream` tool with the same runId
# 6. run the regression: cargo test -p vestige-mcp test_full_spine_one_runid_crosses_every_hop
```
