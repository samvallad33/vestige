# Hygiene and tag maintenance

> Written for Vestige 4.1.1.

Vestige finds a record by an exact handle: its id, a unique id prefix of 8 or more characters, or an exact tag. So hygiene comes down to three habits: spell tags the same way every time, give time-bound facts a validity window, and look at the counts now and then. Ordinary operations default to the `user` scope. Cross-scope maintenance is always explicit.

## Tags are the only index

- A tag is an exact, case-sensitive string. `vestige` and `Vestige` are two tags. Vestige stores tags exactly as you give them. It does not normalize them or suggest a nearby one.
- Use one project tag and one or two narrow topic tags, spelled the same way every time. A record with no tag is reachable only by its id.
- A tag recall returns every live record under the tag, from every scope. Keep topic tags narrow, and do not recall a tag that half the store carries.
- Before you create a tag, read the counts: `memory_status` with `view="stats"` lists record counts by exact tag.

## Dated facts at ingest

`smart_ingest` accepts explicit `validFrom` and `validUntil` values as RFC 3339 timestamps or exact `YYYY-MM-DD` dates. Explicit values remain authoritative, and `validUntil` must be after `validFrom`.

When `validFrom` is omitted and the content holds exactly one distinct, strict `as of YYYY-MM-DD` date, Vestige uses that date as `validFrom` for the new record. The phrase must start at a word boundary, and the date must be a real calendar date followed by no letter, digit or hyphen. Repeating the same date is unambiguous. Several distinct dates, an invalid date, or a match inside another word are not used.

The response's `validity` object always says what happened. `source` is one of `none`, `explicit`, `inferred_as_of`, `explicit_and_inferred_as_of`, `ambiguous_as_of_not_applied`, `explicit_with_ambiguous_as_of_ignored`, `inferred_as_of_conflicts_with_explicit_validity_ignored` or `state_default_ttl`. `inferredPhrase` and `ambiguousPhrases` name the text that was used or skipped. An inferred start that conflicts with an explicit `validUntil` is not applied, and the save still goes through.

This date reading is a parse of the content you are saving. It runs once, at write time, and it never affects how a record is found or ranked.

## State records expire by default

A record saved with `node_type: "state"` (a version number, a progress figure, an inventory) expires 30 days after it is saved unless you pass `validUntil`. Set `VESTIGE_STATE_TTL_DAYS` to change the default, or to `0` to turn it off. Other node types are untouched.

## Tag suggestions are unavailable

`smart_ingest` has two fields from the v3 engine, `previewTagSuggestions` and `acceptedTagSuggestions`. On a Strata log they have nothing to work with, because suggesting a nearby tag means comparing tag names and the log does not do that.

- Every save reports `tagSuggestionStatus` as `{"status": "unavailable", "reason": "similarity_disabled"}` with a plain `detail`. Nothing failed. The tags are stored exactly as given.
- `previewTagSuggestions: true` stores nothing and returns `wouldWrite: false`, an empty `tagSuggestions` and that status.
- `acceptedTagSuggestions` is refused, because no mapping can be a current suggestion.

## Exact tag rename and merge

Tag maintenance is part of the `dedup` tool:

- `action="tag_rename"` uses `source_tag` and `target_tag`.
- `action="tag_merge"` uses two or more `source_tags` and one `target_tag`.
- `scope` defaults to `user`. `all_scopes=true` is the explicit cross-scope mode. Passing both is refused.

Without `confirm`, the call is a read-only preview. It lists the exact count for each source tag, the affected ids and their count, and a `previewToken`. A target tag that looks like a credential is refused without echoing it.

**Applying is not available on a Strata log in 4.1.1.** Repeating the call with `confirm=true`, the token and a `reason` fails with "apply_tag_mutation is not implemented by this backend", and nothing is written. To change a tag today, save a corrected successor with `memory` action `edit`, which admits a new record and retires the old one.

`dedup` action `undo` works on recorded writes. See [MERGE_SUPERSEDE.md](MERGE_SUPERSEDE.md).

## Full-store statistics

Call `memory_status` with `view="stats"`. `all_scopes=true` is opt-in. The figures come from the live records of the selected scope.

- counts by record type and exact tag;
- age bands `0-7d`, `8-30d`, `31-90d`, `91-180d`, `181d+`, plus future-dated records;
- fixed retention buckets, including zero-count buckets;
- lifecycle counts for current, future, expired and invalid-window records;
- the largest records by UTF-8 byte size.

Only detail lists are capped (`limit` defaults to 50 and is at most 200). Each list reports its total and whether it was truncated. If a preview contains a credential shape, it is replaced with a redaction marker and `contentPreviewRedacted` is true.

A Strata log records no access history. Every record therefore appears under `accessUnknownPrunedLog`, and `neverAccessed` is always empty. Do not suppress a record on access grounds. The tag-operation audit list is empty for the same reason: no tag rename or merge has been applied.

## Retiring a record

- `memory` action `demote` does not delete. The record stays and fades faster.
- `memory` action `edit` admits a successor and retires the old record. The id changes.
- `suppress` hides a record from every read and keeps its bytes. On Strata it cannot be undone. It is not erasure.
- `purge`, `memory` actions `purge` and `delete`, and `delete_knowledge` return `unavailable_in_4_0`. Nothing is erased from an append-only log.
