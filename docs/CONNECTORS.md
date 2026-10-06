# External-Source Connectors

> Written for Vestige 4.2.0. **Status: source only.** `source_sync` is not in any 4.x release binary, and in 4.1.1 the Strata backend does not implement the storage calls a sync needs. Tracking issue: [#57](https://github.com/samvallad33/vestige/issues/57).

A connector turns the records of a long-lived external system (a ticket tracker, an issue board, a support queue) into source-aware records with a provenance envelope that names the canonical URL. The external system stays the source of truth. The code for GitHub Issues and Redmine is in the repository. This page says what is there and what is not.

## What is and is not in a 4.x build

- **Off by default.** The HTTP client sits behind the `connectors` cargo feature. It is off in `vestige-mcp`, `vestige` and `vestige-core`, and no release build passes it. A default build does not advertise `source_sync`, and `tools/call` for that name is the normal unknown-tool protocol error (`-32602`). The release workflow rejects any target feature set that links `reqwest`.
- **Network.** `source_sync` and `vestige sync --cloud` are the only calls that use the network, and neither is in a release build. Tokens are read from the environment only, never from tool arguments, and never logged.
- **No Strata backend yet.** The sync driver calls `upsert_by_source`, `get_connector_cursor`, `save_connector_cursor` and `reconcile_source_tombstones`. The Strata backend does not override them, so they return the trait default, "is not implemented by this backend". A sync into a Strata log therefore fails, even in a build made with `--features connectors`. This is read from source, not from a run.
- **No search.** The records in this design were indexed for search. 4.x has no search: a record is found by an exact handle, and the `source_*` filters on `recall` belonged to the v3 engine's search. The envelope fields below still describe provenance.

Building the feature anyway, for work on the connectors:

```sh
cargo build -p vestige-mcp --release --features connectors
cargo build -p vestige-core --features connectors
```

## The `source_sync` tool (source only)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `source` | string | `github` | `github` or `redmine`. |
| `repo` | string | none | **GitHub:** `owner/name`, for example `samvallad33/vestige`. |
| `project` | string | none | **Redmine:** project identifier (the host comes from `REDMINE_URL`). |
| `reconcile` | bool | `false` | Also tombstone local records for issues no longer visible upstream (an extra full-enumeration pass). |
| `max_pages` | int | `10` | API pages to fetch this run (up to 100 issues each). Lets a first sync of a large project resume across calls. |

The tool returns counts (`created`, `updated`, `unchanged`, `tombstoned`), the saved `cursor`, whether it ran authenticated, and a `hint` for the next step.

Credentials come from the environment:

```sh
export GITHUB_TOKEN=...            # or VESTIGE_GITHUB_TOKEN; optional, raises the rate limit
export REDMINE_URL=https://redmine.example.com
export REDMINE_API_KEY=...         # or VESTIGE_REDMINE_API_KEY
```

A Redmine instance needs its REST API enabled (Administration, Settings, API), or every call returns 401 or 403 even with a valid key.

## How the sync driver works

This is the driver's contract in `crates/vestige-core/src/connectors/mod.rs`. It was written against the v3 engine's store.

1. It resumes from the saved cursor, the high-water mark on the record's upstream update time, minus a small overlap window, so same-second and clock-skewed updates are not missed.
2. It pages issues in ascending update order (`state=all`, so closing an issue is not mistaken for a deletion) and folds each issue and its comments into one record.
3. It routes each record through an idempotent upsert keyed on `(source_system, source_id)`: an unseen record is inserted, a changed `content_hash` updates in place, and an unchanged one only advances its last-seen time.
4. It advances and saves the cursor only after the run, and never past a record that failed, so an interruption re-scans instead of skipping.

### Deletions (tombstoning)

Neither GitHub nor Redmine exposes a deletion feed, so an incremental sync cannot see a delete. With `reconcile: true`, the driver enumerates the currently visible issue ids and invalidates any local record no longer present. It does not purge. If the listing comes back empty, it skips the pass instead of tombstoning the whole source. A record that reappears upstream is un-tombstoned by the next sync.

## The source envelope

Every connector record carries structured provenance, distinct from the free-form `source` label:

| Field | Purpose |
|---|---|
| `source_system` | `github`, `redmine`, and so on. Namespaces ids. |
| `source_id` | Native id (issue number, ticket id). |
| `source_url` | Canonical link back: the citation. |
| `source_updated_at` | Upstream update time (the sync cursor field). |
| `content_hash` | Change detector, so a re-run is idempotent. |
| `synced_at` | When the connector last saw the record live. |
| `source_project` | Repo, project or space. |
| `source_type` | `issue`, `comment`, and so on. |
| `source_author` | Reporter or author upstream. |

`(source_system, source_id)` is unique, so there is one record per external record. `codebase` `ingest_repo` records a source envelope too, `(git, <codebase>, <sha>)`, and `memory_status` with `view="provenance"` reports it.

## Writing a new connector

Implement the `Connector` trait in `vestige_core::connectors`: fetch a window of records updated since a cursor, page forward, and optionally enumerate live ids for reconciliation. Produce `NormalizedRecord`s with a filled `SourceEnvelope` and hand them to `run_sync`. Two reference connectors show the shape: `crates/vestige-core/src/connectors/github.rs` (Link-header pagination, opaque-url cursor) and `crates/vestige-core/src/connectors/redmine.rs` (offset pagination, two-phase list-then-detail fetch).

## Not supported

- **Assignee filter.** The envelope stores `source_author` (the reporter) only.
- **Linked-issue edges.** Connectors import relations into the record body. They do not write typed edges between issue records.
- **Closing-commit edges are a text pattern.** After a GitHub sync, `source_sync` links a closed issue to a locally ingested git-commit record with a `closed_by` edge when that record's text has a closing keyword and `#<number>` on the same line. That edge comes from reading text, not from a cause the log recorded, so treat it as a lead and not as proof.
