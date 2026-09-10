# Vestige v3 integration

V3 is the open-source local memory upgrade for developers and coding agents.
Operator and full automation pipelines are separate products. Local memory and
embeddings require no provider API key; memory writes apply automatically by
default, with optional review settings retained for users who choose them.

## Reused work

The v3 branch combines the context/lifecycle foundation with these earlier PRs:

| PR | Included work |
|---|---|
| #249 | Correct attached-runtime vector model labels and prevent regeneration churn |
| #233 | Reasoning reports assembled from measured evidence and confidence arithmetic |
| #234 | Compact ingest previews and omission of empty response fields |
| #237 | Shorter descriptions alongside complete v3 schemas and progressive discovery |
| #236 | Dashboard duplicate preview/apply using the reversible dedup backend |
| #243 | First-run notifications, warming state and cancellation-safe stdio |
| #251 | Durable intention graphs, recurring reminders, compound triggers and replay |
| #245 | Scoped Markdown projection with provenance and fenced updates |
| #235 | Rust stdio coverage combined with the existing v3 contracts |
| #244 | Compatible dashboard dependency update |
| #246 | Linux ARM build, packaging and release support |
| #247 | MemoryArena adapter and preregistered evaluation protocol |
| #248 | Labeled benchmark evidence and examples |
| #238 | Storage module organization, regenerated from the integrated v3 source |

Integration preserves v3's passive-read strength behavior, source scoping,
response budgets, trigger clocks, suppression journals and reversible merges.
Intention graph persistence uses migration 36, after suppression migrations
34/35. One-shot timers accept zero minutes; recurring intervals must be positive.
Event triggers compare explicit event strings exactly, ignoring case.

Markdown selection filters scope, validity, suppression and supersession before
applying its limit. The generated region is bounded to 10,000 UTF-8 bytes.
Projection updates serialize Vestige writers and reject changed file snapshots;
external editors do not share the writer lock.

The expanded 15-tool catalog includes intention graph commands and complete
maintenance schemas. Its integration baseline is 52,988 bytes; the common
recall/smart_ingest/memory subset is below 13 KB. Full-catalog and subset budgets
are tested separately. Progressive discovery reduces the definitions a client
needs to send; clients that always send every tool still pay the full catalog.
Byte reductions are not a measurement of provider charges.

#239's additional purge tool and mandatory host interaction are not included:
the existing memory purge action retains explicit `confirm=true`. #199's
Pro/Operator promotion is separate from the open-source installation flow.

Use `docs/V3-VALIDATION.md` for the local test commands and the task-cost examples
for measuring your own developer workflows. Optional provider experiments are
separate from installing or using the memory server.
