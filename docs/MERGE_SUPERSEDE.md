# Duplicates, Supersession and Undo

> Written for Vestige 4.2.0. v3 had a merge and supersede workflow scored by embedding similarity. A Strata log withholds it. This page says what `dedup` does now and why the rest is refused.

Records pile up: copies, near-copies and facts that are out of date. The fixes that go wrong are the ones that guess. A guess that merges two records destroys the audit trail, and a guess that deletes on a contradiction loses information. Vestige 4.x does not guess. It lists only records that are identical, it never merges, and every change it does make is a recorded write that you can undo.

## What `dedup` does in 4.1.1

| Action | State | What it does |
| --- | --- | --- |
| `scan` | works | Lists exact duplicate clusters under `duplicateClusters`. Read-only |
| `undo` | works | With no `operation_id`, lists recent recorded writes. With one, appends a compensating record. Needs `confirm=true` |
| `tag_rename`, `tag_merge` | preview only | Shows what a rename or merge would touch. Applying is not available. See [MEMORY_HYGIENE.md](MEMORY_HYGIENE.md#exact-tag-rename-and-merge) |
| `policy` | stored, unused | Reads and sets the two thresholds and `auto_apply`. 4.1.1 has no merge scoring and no apply step, so nothing acts on them |
| `plan_merge` | withheld | `unavailable_in_4_0`: merge planning needs embeddings, which 4.0 removed |
| `plan_supersede` | withheld | `unavailable_in_4_0`: supersede planning needs embeddings, which 4.0 removed |
| `apply` | withheld | `unavailable_in_4_0`: there are no merge or supersede plans without embeddings |
| `verdict` | withheld | `unavailable_in_4_0`: reconsolidation verdicts need embeddings, which 4.0 removed |
| `protect` | withheld | `unavailable_in_4_0`: the Strata log has no protect flag yet |

A withheld action is left out of the advertised schema and refused with its reason.

## Exact duplicates

`scan` forms a cluster only when records have the same content identity or the same declared source key:

- the stored `content_hash` of the source envelope, or byte-identical content when no hash was recorded; or
- the same `source_system` and `source_id`.

There is no similarity score. Two records that say the same thing in different words are not a cluster. In the scan response, `mergeCandidates` is `{"status": "unavailable", "reason": "embeddings_unavailable"}`. It carries no count and no list, so it cannot be read as "none found". `reconsolidationPlans` is empty, and `nextStep` says merge planning is withheld.

## Superseding: invalidate, do not delete

Nothing is deleted from a Strata log. To replace a record, use `memory` action `edit`. An edit admits a successor and retires the previous record, and it keeps the code anchors. The old record's bytes stay on the log. Reads and handle lookups no longer return it. The id changes, so move any reference you stored to the new id.

A fact that stops being true on a known date belongs in a validity window instead. Save it with `validUntil`, or let a `state` record expire. See [MEMORY_HYGIENE.md](MEMORY_HYGIENE.md#dated-facts-at-ingest).

## Undo: the reflog

Every write is recorded, so `dedup` action `undo` can reverse one.

- With no `operation_id`, it returns the most recent recorded writes, newest first.
- With an `operation_id` and `confirm=true`, it appends a compensating record to the log. The undone change is no longer visible to reads. An undo cannot itself be undone, so without `confirm=true` the call changes nothing and says so.
- Undoing an edit leaves the pre-edit record live, with its code anchors, and retires only the edit.

The log itself is never rewritten. The write, the undo and the receipt of each stay on it. See [DECISION_RECEIPTS.md](DECISION_RECEIPTS.md).

## Protecting a record

There is no `protect` flag on a Strata log in 4.1.1. Take a backup (`vestige backup <new-folder>`) before you change records you cannot afford to lose.
