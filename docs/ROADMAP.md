# Vestige Roadmap

> Public adoption roadmap for making Vestige easier to start, easier to trust,
> and easier to configure.

Last updated: June 7, 2026

Vestige already has the core primitives for durable local memory: `search`,
`session_context`, `smart_ingest`, `memory`, `intention`, `codebase`,
`deep_reference`, suppression, portable storage, and the dashboard. The next
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
| Session startup | `session_context` combines memories, intentions, status, predictions, and codebase context. | Update all agent setup templates to make `session_context` the default startup call. |
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
| Updated agent templates | Claude, Codex, Cursor, VS Code, Xcode, OpenCode, JetBrains, and Windsurf templates start with `session_context` and use the same memory vocabulary. |

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
| Goal-aware session context | `session_context` can include active goals when relevant. |
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
