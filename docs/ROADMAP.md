# Vestige Roadmap

> Public adoption roadmap for making Vestige easier to start, easier to trust,
> and easier to configure.

Last updated: September 4, 2026

Vestige already has the core primitives for durable local memory: `recall`
(lookup, reason and contradictions modes), `session_start`, `smart_ingest`,
`memory`, `intention`, `codebase`, suppression, portable storage, and the
dashboard. The next
product step is reducing first-user confusion so more people can get value from
those primitives without inventing their own fragile memory vocabulary.

This roadmap turns early community feedback into a staged plan.

## Principles

- Make first use obvious. A new user should know what to import, how atomic each
  memory should be, and which tool to use for current session context.
- Keep memory legible. Agents and humans should understand whether a memory was
  created, reinforced, updated, superseded, suppressed, or purged.
- Prefer progressive disclosure. The default MCP response should be lean, with
  explicit ways to request more detail.
- Keep local-first behavior. New onboarding, code memory, and configuration
  features must not require a cloud service.
- Optimize for many users. Defaults should work for non-experts, while power
  users can tune fields, merge behavior, and formats.

## Already Shipped, Needs Clearer Guidance

| Area | Current State | Next Documentation Fix |
|------|---------------|------------------------|
| Session startup | `session_start` combines memories, intentions, status, predictions, and codebase context. | Update all agent setup templates to make `session_start` the default startup call. |
| Batch memory saves | `smart_ingest` batch mode defaults to `batchMergePolicy="force_create"` so caller-separated items stay separate. | Document when to use batch force-create vs smart merge. |
| Device migration | `portable-export`, `portable-import`, and `sync` preserve exact Vestige storage state. | Separate device migration from first-time document import so users do not confuse them. |
| Supersede semantics | Supersede demotes the old memory and creates a new one; it does not purge the old memory. | Add plain-language vocabulary for create, update, supersede, suppress, demote, and purge. |

## Phase 1: Onboarding And Memory Hygiene

Target: make the first 30 minutes with Vestige hard to mess up.

| Work | Outcome |
|------|---------|
| First-time memory migration guide | Users can import notes/docs without Claude tagging everything as `verified` or flattening unrelated facts together. |
| Atomic memory guide | Clear examples for one fact, one preference, one decision, one bug fix, one source note, and one code pattern per memory. |
| Default tag vocabulary | Recommended tags for source quality, confidence, project, type, urgency, and lifecycle without overloading words like `verified`. |
| Smart vs force-create guide | Agents know when to use `forceCreate`, `batchMergePolicy="force_create"`, or normal PE gating. |
| Updated agent templates | Claude, Codex, Cursor, VS Code, Xcode, OpenCode, JetBrains, and Windsurf templates start with `session_start` and use the same memory vocabulary. |

Planned docs:

- `docs/MIGRATION.md`
- `docs/MEMORY-HYGIENE.md`
- revised `docs/AGENT-MEMORY-PROTOCOL.md`
- revised `docs/CLAUDE-SETUP.md`

## Phase 2: Configurable Output

Target: let users control context cost without losing important evidence.

| Work | Outcome |
|------|---------|
| Field masks for MCP results | Users can drop fields they never want in model context, such as temporal hints, scores, or timestamps. |
| Output profiles | Presets like `lean`, `default`, `audit`, and `research` tune result size and metadata detail. |
| Markdown output mode | Users can request compact Markdown summaries when that is more context-efficient than JSON. |
| Context reinstatement controls | `contextReinstatement` becomes opt-in or configurable, and temporal hints are based on stored memory context when available. |
| Per-tool defaults | Users can define default detail level, result limit, and response shape for search, timeline, codebase, and session context. |

Likely implementation paths:

- config file under the active Vestige data directory
- environment-variable override for simple deployments
- MCP parameters still win over defaults for one-off calls

## Phase 3: Merge And Supersede Controls

Target: make memory mutation predictable.

| Work | Outcome |
|------|---------|
| Merge policy configuration | Users can keep some tags or node types atomic while allowing others to merge. |
| Prediction Error threshold knobs | Advanced users can tune create/update/reinforce boundaries without recompiling. |
| Merge previews before mutation | Agents can show what would change before updating an existing durable memory. |
| Safer consolidation dedup | Consolidation respects user-configured atomic tags and source boundaries. |
| Friendlier lifecycle labels | Agent-facing copy explains that superseded memories are old versions, not destroyed records. |

## Phase 4: Code Memory

Target: make code memories useful without blending source code, docstrings, and
human project notes into one noisy search space.

| Work | Outcome |
|------|---------|
| Code memory import guide | Developers know when to save patterns/decisions versus code entities or docstrings. |
| Exposed code entity workflow | The existing core `CodeEntity` concept becomes usable through MCP or CLI. |
| Docstring/code symbol ingestion | Users can ingest functions, types, modules, docstrings, and call-site notes with source file provenance. |
| Code/prose retrieval separation | Search can filter or rank code memories separately from user preferences and project decisions. |
| Codebase dashboard review | Developers can inspect imported code memories and remove noisy entries. |

## Phase 5: Goals And Milestones

Target: support durable direction without pretending every future task is just a
reminder.

| Work | Outcome |
|------|---------|
| Goal primitive | Non-fading, manually pivoted goals that survive normal memory decay. |
| Milestone tracking | Goals can have milestones, status, evidence, and blockers. |
| Goal-aware session context | `session_start` can include active goals when relevant. |
| Manual pivot semantics | Agents can update goals only when the user explicitly pivots, completes, or cancels them. |
| Dashboard surface | Users can inspect active, completed, paused, and cancelled goals. |

This is distinct from `intention`: intentions are reminders triggered by time,
topic, file, event, or context. Goals are longer-lived direction and should not
fire as reminders unless the user attaches an intention.

## Phase 6: Guided Import Tools

Target: turn "I have 300 notes" into a reliable workflow.

| Work | Outcome |
|------|---------|
| Import dry run | Vestige previews proposed memories, tags, node types, and merge decisions before writing. |
| Source-aware import | Imported memories keep file/source provenance and confidence metadata. |
| Chunking strategies | Users choose atomic facts, section summaries, decision records, or source notes. |
| Review queue | Users can approve, edit, split, merge, or reject proposed memories. |
| Post-import health pass | Vestige recommends consolidation, duplicate review, or tag cleanup after import. |

## Tracked Issues (Consolidated 2026-07-02)

The following roadmap issues were consolidated here and closed so the issue tracker
reflects active work, not a standing backlog. Nothing is lost — each entry keeps its
scope, the backend anchors that already exist, and why it is deferred. Most are
deferred behind the dashboard focus; several are security/data-integrity boundaries
that are deliberately *not* shipped half-done.

### A. Reliability & Trust surfaces

- **Agent Reliability Record** (was #84) — Unify traces, receipts, contradictions,
  and composed-graph events into one per-run record with 5 evidence states
  (supported / missing / stale / contradicted / suppressed) + Markdown export.
  Backend already exists (`crates/vestige-core/src/trace/`). Remaining work is the
  dashboard record view — see the dashboard Discussion.
- **Trust Zones + Memory Quarantine** (was #85) — Provenance/trust class on nodes,
  score-capping for weak-provenance content, quarantine of untrusted sources.
  Security boundary; unsafe half-done, and depends on ACL Memory primitives that
  don't exist yet. Deferred post-dashboard.
- **ComposeBench** (was #86) — Reliability benchmark across 8 scenarios. Will reuse
  the existing `benchmarks/silent-rotation/` harness pattern; the ACL scenario is gated
  on ACL Memory. Deferred.

### B. Access & Governance boundary

- **ACL Memory for source-aware connectors** (was #82) — Source-authorization-aware
  memory: connector-ingested memories preserve upstream access rules and retrieval
  fails closed for unauthorized callers. A hard security boundary with no foundation
  today (no per-caller identity model; `search()` takes no subject). Must be designed
  as one deliberate pass, not sliced. Design + user-permission-shape input welcome in
  the Discussion.
- **Team Pro Reliability Foundation** (was #92) — Commercial team tier (RBAC/SSO/SCIM,
  admin review, audit export, team lanes, Postgres, hosted backups). A product-strategy
  meta-issue, upstream of coding; depends on ACL Memory + HTTP/Postgres plans.

### C. Ingest & Projection integrity

- **Markdown + Rules Projection** (was #87) — Project memories into client-native
  rule files (AGENTS.md, CLAUDE.md, `.cursor/rules`, Windsurf, Cline) with provenance
  and an optional bidirectional re-import. The re-import leg is a data-integrity
  boundary (must never silently overwrite user files). Target-format priorities are an
  open user question — Discussion.
- **Code Memory Workflow** (was #88) — First-class, inspectable code memory
  (patterns/decisions with file+line provenance) kept separate from prose. The typed
  model exists (`crates/vestige-core/src/codebase/`) but is unpersisted/unwired; needs
  schema + a review/prune dashboard surface.
- **Guided Import + Review Queue** (was #89) — Dry-run import → proposed memories →
  approve/edit/split/merge/reject queue → post-import health pass. Ingest/corruption
  boundary; needs a real no-write dry run + review-queue state machine.
- **Goals + Milestones** (was #90) — A durable, non-decaying goal/milestone primitive
  (paralleling the intentions subsystem) with lifecycle states and evidence/blockers.
  A create-only slice would ship the primitive without its defining non-decay
  guarantee, so it waits for the full build.

### D. Dashboard productization

- **ComposedGraph Productization** (was #91) — The MCP/CLI/storage backend already
  ships in v2.2 (`crates/vestige-mcp/src/tools/composed_graph.rs`, all 7 modes). The
  remaining slice is the dashboard surface: composition history, the never-composed
  frontier, and closed doors. This is the natural first move in the dashboard focus —
  shape it in the Discussion.

## Non-Goals

- Do not auto-store every conversation turn by default.
- Do not require cloud services for memory creation, search, or configuration.
- Do not hide irreversible deletion. `purge` must stay explicit.
- Do not make code ingestion pollute general personal memory by default.
- Do not make advanced tuning required for ordinary users.

## How To Read This Roadmap

This is directional, not a release guarantee. The priority is adoption: fewer
surprises, clearer defaults, and better tool descriptions before adding complex
new surfaces. Community feedback that reveals a confusing first-use path should
usually become either a documentation fix, a safer default, or a guided workflow.
