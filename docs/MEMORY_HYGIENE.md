# Memory hygiene and tag maintenance

Vestige exposes project-isolated, agent-facing hygiene workflows without requiring direct SQLite edits. Ordinary operations default to the legacy-compatible `user` scope. Cross-scope maintenance is always explicit.

## Dated facts at ingest

`smart_ingest` accepts explicit `validFrom` and `validUntil` values as RFC 3339 timestamps or exact `YYYY-MM-DD` dates. Explicit values remain authoritative.

When `validFrom` is omitted and content contains exactly one distinct, strict `as of YYYY-MM-DD` date, Vestige infers that date as `validFrom`. Matching is case-insensitive and boundary-aware. Repeating the same date is unambiguous; multiple distinct dates, invalid calendar dates, or embedded-word matches are not inferred. The response's `validity` object always reports whether values were `explicit`, `inferred_as_of`, ambiguous and ignored, or absent. An inferred start that conflicts with an explicit `validUntil` is rejected before any write.

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

The vocabulary scan uses distinct same-scope tags and fails the nudge explicitly above 10,000 distinct tags or when legacy vocabulary contains an over-200-character tag. At most 50 caller tags of at most 200 characters each are considered, and edit distance is bounded to the one- or two-edit window Vestige can actually suggest. Secret-shaped caller mappings are rejected without echoing their bytes; secret-shaped legacy tags are excluded from responses and reported only as an ignored count. NFKC plus Unicode lowercase normalization is used; locale-specific language rules are deliberately not inferred.

## Exact tag rename and merge

Tag maintenance is available through the consolidated `dedup` tool:

- `action="tag_rename"` uses `source_tag` and `target_tag`.
- `action="tag_merge"` uses two or more `source_tags` and one `target_tag`.
- `scope` defaults to `user`; `all_scopes=true` is the explicit cross-scope mode.

Every operation is two-step. First call the action without confirmation. The read-only result lists exact source counts, affected IDs/count, collision information, and a `previewToken`. Then repeat the same action with `confirm=true`, that token, and a nonempty `reason`. Apply recomputes the token inside the SQLite write transaction, so a changed scope, source, target, affected set, or tag array makes the preview stale and aborts the operation.

Tags are parsed and reserialized as JSON arrays; substring replacement is never used. Source/target collisions become one deterministic target at the first affected position. The mutation and its durable audit record commit in one transaction. Operations are capped at 50,000 affected memories and a 16 MiB undo payload; larger operations fail without partial writes and should be narrowed by scope.

The preview and apply paths both enforce Vestige's default-deny credential policy for source and target tags; apply also screens the durable audit reason. Rejections never copy the detected credential bytes into the MCP response or audit log.

Use `dedup(action="undo", operation_id="...")` to restore the exact prior tag arrays. Undo first verifies every current post-operation array and refuses the entire reversal if a later tag edit or missing memory would be overwritten. Omitting `operation_id` lists the agent-visible reflog.

## Full-store hygiene statistics

Call `memory_status` with `view="stats"`. The primary population is every stored row in the selected scope, including temporally inactive and superseded rows; lifecycle counts make those states explicit. `all_scopes=true` is opt-in.

Aggregates cover the full store and include:

- counts by memory type and exact tag;
- age bands `0-7d`, `8-30d`, `31-90d`, `91-180d`, `181d+`, plus future-dated rows;
- fixed retention buckets, including zero-count buckets;
- lifecycle counts for current, future, expired, invalid-window, and superseded rows;
- never-accessed total and deterministic bounded list;
- largest-node total and deterministic bounded list by UTF-8 byte size;
- recent tag rename/merge audit operations.

Only detail lists are capped (`limit` defaults to 50 and is at most 200). Each list reports its total and whether it was truncated. The storage query loads bounded content previews rather than every full memory body and computes access status without per-row queries.
