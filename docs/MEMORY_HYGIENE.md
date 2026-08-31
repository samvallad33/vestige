# Memory hygiene and tag maintenance

Vestige exposes project-isolated, agent-facing hygiene workflows without requiring direct SQLite edits. Ordinary operations default to the legacy-compatible `user` scope. Cross-scope maintenance is always explicit.

## Dated facts at ingest

`smart_ingest` accepts explicit `validFrom` and `validUntil` values as RFC 3339 timestamps or exact `YYYY-MM-DD` dates. Explicit values remain authoritative.

When `validFrom` is omitted and content contains exactly one distinct, strict `as of YYYY-MM-DD` date, Vestige infers that date as `validFrom` for a **new** node. Matching is case-insensitive and boundary-aware. Repeating the same date is unambiguous; multiple distinct dates, invalid calendar dates, or embedded-word matches are not inferred. The response's `validity` object always reports whether values were `explicit`, `inferred_as_of`, ambiguous and ignored, or absent.

An inferred start that conflicts with an explicit `validUntil` is **not** applied; the ingest still stores and the skipped phrase is reported. Prose-inferred dates never rewrite an existing node's window: Prediction Error Gating `Update` / `Reinforce` / `Merge` / `Replace` only mutates stored `valid_from` / `valid_until` when the caller supplied those fields explicitly. `update_node_validity` merges bounds (`COALESCE`); omitting a bound preserves the stored value instead of writing `NULL`. That keeps an expired fact expired when a near-duplicate arrives with `as of YYYY-MM-DD` in the body.

## Similar-tag preflight

Similar-tag nudges are scoped, bounded, deterministic, and never auto-applied. They recognize casing, punctuation, small edit-distance, and safe namespaced/non-namespaced suffix variants such as `prixsix`, `prix-six`, and `codebase:prix-six`.

To decide before storing a new variant:

```json
{
  "content": "Prix Six deployment note",
  "tags": ["prix-six"],
  "scope": "user",
  "previewTagSuggestions": true
}
```

This returns `wouldWrite: false`, a `tagSuggestions` list, and `tagSuggestionStatus`; it does not store a memory. To accept a suggestion, repeat the ingest with an exact mapping:

```json
{
  "content": "Prix Six deployment note",
  "tags": ["prix-six"],
  "scope": "user",
  "acceptedTagSuggestions": {
    "prix-six": "prixsix"
  }
}
```

Vestige recalculates the same-scope vocabulary and rejects any mapping that is no longer a current suggestion. Omitting `acceptedTagSuggestions` preserves the caller's original tags. Batch items support the same two fields independently.

The vocabulary scan uses distinct same-scope tags and fails the nudge explicitly above 10,000 distinct tags. Stored tags longer than 200 characters are skipped and counted (`ignoredOverlongVocabularyTags`) instead of disabling suggestions for the whole scope; they remain renameable through `dedup`. At most 50 caller tags of at most 200 characters each are considered, and edit distance is bounded to the one- or two-edit window Vestige can actually suggest. Secret-shaped caller mappings are rejected without echoing their bytes; secret-shaped legacy tags are excluded from responses and reported only as an ignored count. NFKC plus Unicode lowercase normalization is used; locale-specific language rules are deliberately not inferred.

Ordinary tagged `smart_ingest` still computes suggestions on the write path. Deferring that scan until `previewTagSuggestions` or `acceptedTagSuggestions` is tracked separately and is not claimed here.

## Exact tag rename and merge

Tag maintenance is available through the consolidated `dedup` tool:

- `action="tag_rename"` uses `source_tag` and `target_tag`.
- `action="tag_merge"` uses two or more `source_tags` and one `target_tag`.
- `scope` defaults to `user`; `all_scopes=true` is the explicit cross-scope mode.

Every operation is two-step. First call the action without confirmation. The read-only result lists exact source counts, affected IDs/count, collision information, and a `previewToken`. Then repeat the same action with `confirm=true`, that token, and a nonempty `reason`. Apply recomputes the token inside the SQLite write transaction, so a changed scope, source, target, affected set, or tag array makes the preview stale and aborts the operation.

Tags are parsed and reserialized as JSON arrays; substring replacement is never used. Source/target collisions become one deterministic target at the first affected position. The mutation and its durable audit record commit in one transaction. Operations are capped at 50,000 affected memories and a 16 MiB undo payload; larger operations fail without partial writes and should be narrowed by scope.

The preview and apply paths both enforce Vestige's default-deny credential policy for source and target tags; apply also screens the durable audit reason. Rejections never copy the detected credential bytes into the MCP response or audit log.

Use `dedup(action="undo", operation_id="...")` to restore the exact prior tag arrays. Undo first verifies every current post-operation array and refuses the entire reversal if a later tag edit or missing memory would be overwritten. Omitting `operation_id` returns the mixed newest-20 reflog in `operations` plus a dedicated `tagOperations` list queried directly, so merge/supersede activity cannot bury a tag audit.

## Full-store hygiene statistics

Call `memory_status` with `view="stats"`. The primary population is every stored row in the selected scope, including temporally inactive and superseded rows; lifecycle counts make those states explicit. `all_scopes=true` is opt-in.

Aggregates cover the full store and include:

- counts by memory type and exact tag;
- age bands `0-7d`, `8-30d`, `31-90d`, `91-180d`, `181d+`, plus future-dated rows;
- fixed retention buckets, including zero-count buckets;
- lifecycle counts for current, future, expired, invalid-window, and superseded rows;
- `neverAccessed`: no retained access-log row, durable retrieval counters (`times_retrieved` / `times_useful`) are zero, **and** the memory was created inside the 90-day access-log window. The log is pruned after 90 days, so absence of a log row is not durable evidence;
- `accessUnknownPrunedLog`: created before that window with zero durable counters. Pre-prune history is unknowable. Agents must not suppress or purge these on access grounds;
- largest-node total and deterministic bounded list by UTF-8 byte size;
- recent tag rename/merge audit operations. `all_scopes` operations appear on single-scope stats when they rewrote that scope's tags.

Only detail lists are capped (`limit` defaults to 50 and is at most 200). Each list reports its total and whether it was truncated. The storage query loads bounded content previews rather than every full memory body and computes access status without per-row queries. If a preview contains a blocking credential shape, the preview is replaced with a redaction marker and `contentPreviewRedacted` is true; the matched secret bytes are not returned.
