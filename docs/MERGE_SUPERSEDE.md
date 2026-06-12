# Merge / Supersede Controls (Phase 3)

> Diff-previewed, confidence-gated, reversible, self-explaining
> combine/dedupe/supersede on a never-delete (bitemporal) store.

Memory systems accumulate duplicates, near-duplicates, and outdated facts. The
naive fixes are all bad: dumb hashing under-merges (misses paraphrases),
aggressive LLM merging over-merges and destroys the audit trail, and
auto-deleting on contradiction silently loses information. Vestige's Phase 3
takes the opposite stance:

- **Opt-in, never silent.** The default is preview/review. Nothing mutates your
  memory unless you explicitly apply a plan.
- **Diff-previewed.** `plan_merge` / `plan_supersede` show exactly what *would*
  change before anything does.
- **Confidence-gated.** A Fellegi-Sunter two-threshold score classifies each
  candidate as `match` / `possible` / `non_match`.
- **Reversible.** Every applied operation is recorded with an undo payload — a
  *git reflog for your agent's memory*.
- **Self-explaining.** Each candidate carries the signals that explain *why* two
  memories were judged duplicates.
- **Audit-preserving.** Superseding does not delete: it stamps `valid_until` and
  keeps the old memory queryable (Graphiti-style "invalidate, don't delete").

## The bitemporal model: invalidate, don't delete

Superseding memory A with memory B does **not** erase A. Instead:

- `A.valid_until` is stamped with the supersede time.
- `A.superseded_by` is set to `B.id` (a lineage pointer).
- A remains fully queryable for audit. Searches and timelines can still surface
  it; it is simply marked as no longer the current truth.

This reuses the existing `valid_from` / `valid_until` columns on
`knowledge_nodes` (migration V2) plus a new `superseded_by` column (migration
V14). Merges work the same way: the survivor absorbs the others' content, and
each absorbed node is bitemporally invalidated rather than deleted.

## Fellegi-Sunter two-threshold scoring

Candidate scoring combines three signals into a weighted score in `[0, 1]`:

| Signal                  | Weight | Source                                     |
| ----------------------- | -----: | ------------------------------------------ |
| Embedding cosine sim    |   0.70 | stored embeddings (`node_embeddings`)      |
| Tag overlap (Jaccard)   |   0.15 | `knowledge_nodes.tags`                     |
| Content token overlap   |   0.15 | Jaccard over content tokens (len > 2)      |

The combined score is then classified against **two** thresholds:

```
score >= match_threshold       => "match"      (auto-merge eligible)
possible_threshold <= score    => "possible"   (surfaced for review)
score <  possible_threshold     => "non_match"  (never offered)
```

Defaults: `match_threshold = 0.86`, `possible_threshold = 0.72`. The two-band
design means borderline cases are surfaced for review instead of being
force-decided in either direction.

A cluster's confidence is the **weakest** pairwise score within it (the loosest
link), so a cluster is only as confident as its least-similar member.

## The reversible operation log (the "memory reflog")

Every applied merge/supersede writes one row to `merge_operations`:

- `op_type` — `merge` | `supersede` | `undo`
- `status` — `applied` | `reverted`
- `survivor_id`, `affected_ids` — what was touched
- `confidence`, `signals` — the score and *why* the memories combined
- `reason` — a human-readable explanation
- `undo_payload` — a JSON snapshot capturing everything needed to reverse it

`merge_undo` consumes the undo payload to restore the survivor's prior
content/tags and clear the bitemporal invalidation on every affected node, then
records a compensating `undo` operation. Calling `merge_undo` with no
`operation_id` returns the operation log so you can pick one.

## Memory protection (pinning)

`protect` sets the `protected` flag on a memory. A protected memory:

- is never offered for auto-merge (it is flagged in `merge_candidates`),
- cannot be merged *away* (it may only be the survivor of a merge),
- cannot be superseded,
- is excluded from garbage collection.

Pass `protected: false` to unpin.

## Tool surface

| Tool               | Mutates? | Purpose                                                                   |
| ------------------ | :------: | ------------------------------------------------------------------------- |
| `merge_candidates` |    No    | Surface likely duplicate clusters with confidence + signals.              |
| `plan_merge`       |    No    | Preview a merge of 2+ memories (a diff). Returns a `plan_id`.             |
| `plan_supersede`   |    No    | Preview superseding A with B (bitemporal). Returns a `plan_id`.          |
| `apply_plan`       |  **Yes** | Execute a plan by id; recorded as a reversible operation.                |
| `merge_undo`       |  **Yes** | Reverse an operation, or list the operation log when given no id.        |
| `protect`          |  **Yes** | Pin / unpin a memory so it can never be auto-merged/superseded/forgotten. |
| `merge_policy`     |  **Yes** | Get/set the two thresholds + `auto_apply`.                               |

### Typical flow

```text
1. merge_candidates                 -> review clusters + confidence + signals
2. plan_merge { member_ids: [...] } -> inspect the diff, get plan_id
3. apply_plan { plan_id, confirm }  -> apply; get operation_id (reversible)
4. merge_undo { operation_id }      -> reverse if it was wrong
```

`apply_plan` requires `confirm: true` for `possible` / `non_match` plans. A
`match` plan applies without `confirm` only when the policy has
`auto_apply: true` (default `false`).

## Configuration

The merge policy persists per project (stored in `fsrs_config`). It can also be
overridden via environment variables:

| Variable                            | Meaning                              |
| ----------------------------------- | ------------------------------------ |
| `VESTIGE_MERGE_MATCH_THRESHOLD`     | Score ≥ this ⇒ `match`.             |
| `VESTIGE_MERGE_POSSIBLE_THRESHOLD`  | Score ≥ this ⇒ at least `possible`. |
| `VESTIGE_MERGE_AUTO_APPLY`          | `1`/`true` to allow auto-apply.      |

A persisted policy (set via `merge_policy`) takes precedence over the
environment, which takes precedence over the built-in defaults. When
`vestige.toml` configuration lands, the policy will read from there as well.

## Schema (migration V14)

- `knowledge_nodes.protected INTEGER NOT NULL DEFAULT 0`
- `knowledge_nodes.superseded_by TEXT`
- `merge_plans(id, kind, status, created_at, applied_at, survivor_id,
  member_ids, confidence, classification, payload)`
- `merge_operations(id, plan_id, op_type, status, created_at, reverted_at,
  reverts_op_id, survivor_id, affected_ids, confidence, signals, reason,
  undo_payload)`

The two `ALTER TABLE ... ADD COLUMN` statements are applied with duplicate-column
guards so the migration is idempotent on replay; the rest of V14 uses
`CREATE ... IF NOT EXISTS`.

## Anti-patterns this design avoids

- **Silently double-storing contradictions.** Merge composition attributes and
  de-duplicates content instead of blindly concatenating or dropping it.
- **Auto-deleting on contradiction.** Supersede invalidates bitemporally; the
  old memory is retained and queryable.
- **Trading away the audit trail for auto-merge convenience.** Every operation is
  logged and reversible, with provenance for why memories combined.
