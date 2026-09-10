# External-Source Connectors

> Status: **v3.0.0** — GitHub Issues + Redmine reference connectors, plus
> source-aware investigation filters for search. Tracking issue:
> [#57](https://github.com/samvallad33/vestige/issues/57).

Connectors let Vestige act as a durable, local **retrieval and reasoning layer**
over a long-lived external system — a ticket tracker, an issue board, a support
queue — **without replacing it**. The external system stays the source of truth.
Vestige indexes its records, embeds them for semantic recall, links them into the
memory graph, and **cites back** to the canonical record.

## Why this is different from a ticket-system MCP

The official GitHub / Jira MCP servers are **live API proxies**: every query hits
the upstream API, is rate-limited, keyword-only, online-only, and has no memory
of past state. Vestige instead keeps a **durable local index** of the records, so
you can:

- search the history **offline** and **semantically** (embeddings, not just
  keywords),
- **join** ticket history with the rest of your memory in one search,
- see a **point-in-time** view (records carry temporal validity),
- and re-sync **idempotently** — re-running never duplicates a record.

## Quick start (GitHub Issues)

1. (Optional but recommended) export a token so you get the authenticated rate
   limit (5,000 req/hr vs 60 for anonymous) and access to private repos:

   ```sh
   export GITHUB_TOKEN=ghp_xxx   # or VESTIGE_GITHUB_TOKEN
   ```

   The token is read **only** from the environment — never passed as a tool
   argument, never logged.

2. Ask your agent to run the `source_sync` MCP tool:

   ```json
   { "repo": "samvallad33/vestige" }
   ```

3. Search as normal. Connector-sourced results carry a `sourceRecord` object with
   the canonical issue URL:

   ```json
   {
     "content": "[samvallad33/vestige#57] Roadmap: external source connectors …",
     "sourceRecord": {
       "system": "github",
       "id": "57",
       "url": "https://github.com/samvallad33/vestige/issues/57",
       "project": "samvallad33/vestige",
       "type": "issue",
       "author": "samvallad33",
       "tombstoned": false
     }
   }
   ```

## Quick start (Redmine)

Redmine stays the system of record; Vestige indexes a project's issues +
journals (comments and status/assignment history).

1. Point Vestige at the Redmine host and key (env only, never tool args):

   ```sh
   export REDMINE_URL=https://redmine.example.com
   export REDMINE_API_KEY=xxxxxxxx   # or VESTIGE_REDMINE_API_KEY
   ```

   The instance must have the REST API enabled (Administration → Settings → API)
   or every call returns 401/403 even with a valid key.

2. Run `source_sync`:

   ```json
   { "source": "redmine", "project": "infra" }
   ```

   Results cite the canonical `https://redmine.example.com/issues/<id>` URL.

## The `source_sync` tool

| Field | Type | Default | Meaning |
|---|---|---|---|
| `source` | string | `github` | `github` or `redmine`. |
| `repo` | string | — | **GitHub:** `owner/name`, e.g. `samvallad33/vestige`. |
| `project` | string | — | **Redmine:** project identifier (host from `REDMINE_URL`). |
| `reconcile` | bool | `false` | Also tombstone local memories for issues no longer visible upstream (an extra full-enumeration pass). |
| `max_pages` | int | `10` | API pages to fetch this run (≤100 issues each). Lets a first sync of a large project resume across calls. |

The tool returns counts (`created` / `updated` / `unchanged` / `tombstoned`),
the saved `cursor`, whether it ran authenticated, and a `hint` for the next step.

## Investigation filters (Phase 4)

`search` accepts source-aware filters so an agent can scope a query to indexed
records. All are optional post-filters; combine with a larger `limit` if you
expect heavy thinning. A source-scoped query excludes non-connector memories.

| Filter | Matches |
|---|---|
| `source_system` | `github`, `redmine`, … |
| `source_project` | repo / project (exact) |
| `source_id` | a specific issue/ticket id |
| `source_type` | `issue`, `comment`, … |
| `source_author` | reporter/author (not assignee) |
| `source_updated_after` / `source_updated_before` | RFC3339 date range (inclusive) |
| `source_status` | `valid` (default `any`) or `tombstoned` |

Status, tracker, and priority are filterable through the existing `tag_prefix`
(the connectors emit lowercase `status:`, `tracker:`, `priority:`, and GitHub
`label:` / `state:` tags) — e.g. `tag_prefix: "status:open"`. Assignee and
linked-issue graph traversal are not yet exposed (see below).

### Idempotent, incremental sync

Each run:

1. resumes from the saved cursor (the high-water mark on the record's upstream
   update time), minus a small overlap window so same-second / clock-skewed
   updates are never missed;
2. pages issues in ascending update order (`state=all`, so closing an issue is
   **not** mistaken for a deletion), folding each issue + its comments into one
   memory;
3. routes each record through an **idempotent upsert** keyed on
   `(source_system, source_id)`:
   - unseen record → **insert**,
   - changed content (by content hash) → **update in place** + re-embed,
   - unchanged content → **no-op** (only the "last seen" time advances);
4. advances and persists the cursor only after the run, so an interruption
   re-scans rather than skips.

Re-running `source_sync` on the same repo is therefore safe and cheap — it picks
up only what changed.

### Deletions (tombstoning)

Neither GitHub nor Redmine exposes a deletion feed, so an incremental sync can
never *see* a delete. Pass `reconcile: true` to run a reconciliation pass: Vestige
enumerates the currently-visible issue ids and **invalidates** (does not purge)
any local record no longer present. A tombstoned record keeps its content for
audit but drops out of "currently valid" retrieval (`sourceRecord.tombstoned` is
`true`). If the record reappears upstream, the next sync un-tombstones it.

## The source envelope

Every connector-ingested memory carries structured provenance, distinct from the
legacy free-form `source` label:

| Field | Purpose |
|---|---|
| `source_system` | `github`, `redmine`, … (namespaces ids). |
| `source_id` | Native id (issue number, ticket id). |
| `source_url` | Canonical link back — the citation. |
| `source_updated_at` | Upstream update time (the sync cursor field). |
| `content_hash` | Change detector → idempotency. |
| `synced_at` | When the connector last saw the record live. |
| `source_project` | Repo / project / space. |
| `source_type` | `issue`, `comment`, … |
| `source_author` | Reporter / author upstream. |

`(source_system, source_id)` is enforced unique, so there is exactly one memory
per external record. Legacy memories (agent- or user-authored) have no envelope
and are completely unaffected.

## Building

The connector HTTP client is behind the `connectors` cargo feature, which is
**on by default in the MCP server** (`vestige-mcp`). A build without it still
exposes the `source_sync` tool but returns a clear "rebuild with `--features
connectors`" message. The core library (`vestige-core`) leaves the feature
**off** by default, so library consumers that don't need connectors link no HTTP
client.

```sh
# default MCP build already includes connectors
cargo build -p vestige-mcp --release

# explicit, or for the core lib
cargo build -p vestige-core --features connectors
```

## Writing a new connector

Implement the `Connector` trait in `vestige_core::connectors` (fetch a window of
records updated since a cursor, page forward, and optionally enumerate live ids
for reconciliation), produce `NormalizedRecord`s with a filled
`SourceEnvelope`, and hand them to `run_sync`. Two reference connectors show the
shape — `crates/vestige-core/src/connectors/github.rs` (Link-header pagination,
opaque-url cursor) and `crates/vestige-core/src/connectors/redmine.rs`
(offset pagination, two-phase list-then-detail fetch). The sync driver,
idempotent upsert, cursor checkpointing, and tombstone reconciliation are all
reused for free.

## Not yet supported

- **Assignee filter** — the envelope stores `source_author` (reporter) only; no
  assignee column yet.
- **Tracker / version dedicated filter params** — reachable today via
  `tag_prefix` (`tracker:`, and `version:`/`category:` when emitted).
- **Linked-issue graph traversal** — connectors import relations into the memory
  body, but issue-to-issue graph edges are not yet exposed in search.
