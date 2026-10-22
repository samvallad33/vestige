# V5 source-grounded capability and adversarial coverage matrix

Source of truth: docs/TOOL-CONTRACTS.md and default V5 Rust/Operator Lite. A tool can multiplex read-only and mutating actions; listing a tool does not establish live host interception. Values are planned tests, not results.

| V5 advertised MCP tool | Actual capability and limitation | Falsifiable experiment / negative control | Coverage |
| --- | --- | --- | --- |
| smart_ingest | Creates exact content, typed edges, canonical hash and per-write receipt; dedup echo and intent-id idempotence. No vector merging. | Replay same intent with different bytes; prove zero state mutations. Reject a cross-scope edge and a secret. | Live MCP smoke + later adversarial |
| recall | Exact id/prefix/tag, newest-first bounded response; free-text similarity is disabled. | Find exact tag across restart; deny wordy semantic-query without a handle; count wrong tag and handle discovery cost. | Live MCP smoke |
| memory | Get/batch/state, explicit promote/demote FSRS reviews and successor edit; delete/purge withheld. | Read must not reinforce; promote/demote alters card; edit retains provenance, not silent overwrite. | Planned |
| memory_status | Health, provenance, per-record changelog, stats, retention, tools, coverage. | Changelog must show actual writes; report scan caps; cannot infer causality from coverage. | Catalog smoke |
| session_start | Open intentions, backup/dream needs, codebase context, compacted pinned decisions via separate host hook. | Session resumes the right prior decision, not unrelated or stale notes; account for injected tokens. | Existing full-flow + planned |
| receipt | Get and replay effect from Strata; signed trailers / chain verified on replay. | Mutate a disposable log; verification must fail. Never claim stored text is true simply because signed. | Live MCP replay smoke |
| causal_walk | Bounded backward traversal along recorded edges, max one touched edge per path. Ranked hypotheses only. | Find true explicit ancestry; reject string-similar decoy. On missing edge, report no cause. Validate direction. | Live MCP smoke + blinded challenge |
| forgotten_lesson | Retrieves weakly retained earlier fixes from backward recorded paths. | Compare avoided repeated failures vs missing link. Distinguish a correct fix from generic reminder. | Planned |
| selftest | Eleven planted-cause checks in throwaway log; leaves live store untouched. | Verify all checks and unchanged external store; doesn't prove open-world performance. | Live MCP smoke |
| ghostlink | Graph-derived proposed pairings with bridge/divergent lenses; weave/harden/heal outcomes; no vectors. | Blindly judge genuine utility after testing a proposed pair, penalize nonsense and repeated pairs. | Planned |
| gather | Gets GitHub issues, comments, linked PRs and local git evidence and records receipts, then walks. | Freeze offline GitHub snapshot; count omitted sources and missingReceipts. Never claim causal proof from shared names. | Planned; no network in smoke |
| codebase | Store decisions/patterns anchored to source, verify staleness, ingest local commit graph. | Refactor/move file and verify stale vs current. Root-cause ground truth from git/test oracle. | Planned |
| intention | Exact event/context/time triggers, check/update/list/graph. | Exact event fires, similar string does not; test snooze, overdue and wrong scope. | Planned |
| maintain | Dream/dream_compile replay recorded edges and reweight them; backup/export; GC stub. | No new inferred edge from dream; quantify weak-edge reinforcement and maintenance cost. | Planned |
| dedup | Exact scan, token-reviewed tag merge/rename, undo confirmed. No vector duplicate planning. | Near duplicates remain distinct; undo restores exact state; no silent auto-merge. | Planned |
| project | Preview and confirmed write of fenced CLAUDE.md/MEMORY.md from durable memory. | Preserve all unrelated bytes. Code fences and symlink target never bypass file boundary. | Planned |
| suppress | Hide from retrieval; bytes remain in append-only log, no reverse on Strata. | After suppression recall fails while audit log proves bytes remain. Not GDPR erasure. | Planned |
| precedent | Read-only gate decision/permit history; does not independently verify hash chain. | Second agent can locate STOP and reason; bad/stale receipt must not become a reusable permit. | Existing full-flow + comparative planned |
## Separate execution-boundary components

Operator Lite is a **separate pre-tool hook** with STOP and SHADOW rules plus hash-chained **digest** receipts; it only controls hooked tools, and can fail open on certain errors. The paid Operator adds owner-specified laws and multi-step sequence judgment. Do **not** count either as an additional MCP memory tool. The memory server alone does not enforce the agent's shell permissions.

## What neuroscience actually contributes in V5

**Current default Strata:** FSRS formula-shape using pinned approximate constants; heuristic Active/Dormant/Silent/Unavailable bands; explicit strengthen/weaken reviews; replay-inspired dream and dream_compile phases over recorded links; suppression inspired by inhibitory control.

**Engineering rather than neuroscience:** bounded causal graph walks, signed Strata logs, predecessor receipt trails, idempotent writes, graph-only GhostLink, policy enforcement.

**Historical / not active default V5:** V3 Retroactive Salience Backfill, semantic similarity search, synaptic tagging/capture, vector-driven prediction-error gating, spreading activation, and legacy SQLite backfill joins. Existing source code or changelog references are not the same as a live V5 feature. See docs/SCIENCE.md.

**Important no-overclaims:** a sealed Strata segment proves hash/signature integrity relative to trusted verification keys, not the truth of the event. An inferred cause needs an independent test. The default FSRS constants are pinned, not trained for this specific customer's memory. A suppressed record is hidden, not erased.

## Quality gate

Every experiment needs a valid counterexample. The benchmark isn't done when a happy-path passes; it is done when the false positives, unsupported paths, attacker-controlled channels and cost of protection are measured as carefully as the benefits.
