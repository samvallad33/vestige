# Tool contracts

> The 16 tools a Vestige 4.2.0 server advertises, what each action takes, returns and writes

Vestige advertises sixteen MCP tools, sorted by name. Most are action multiplexers:
`memory`, `codebase`, `maintain` and `dedup` take an `action`, `memory_status` takes a
`view`, and `ghostlink` takes a `mode`. An agent should choose the action its task needs.
There is no requirement to call every tool in every session.

Statements here come from the 4.2.0 code under `crates/` and from running the 4.2.0
binaries against a scratch store. Where an older doc and the code differ, the
code wins.

## Discover the installed contract

```json
{"name":"memory_status","arguments":{"view":"tools"}}
```

This lists the names, descriptions and action enums the running server advertises in
`tools/list`. Request one complete schema:

```json
{"name":"memory_status","arguments":{"view":"tools","tool":"maintain"}}
```

The `tools/list` catalog is compact, about 20 KiB (a build-time guard fails the build
above 22 KiB). Every field a call can send is on the wire with its type and its
discriminator enums, and per-field prose moves one call deeper, into the full schema.
A tool accepts grouped (`filters`, `source`) or flat fields. Actions a Strata log cannot
honor are left out of the advertised schema and refused on call.

## Conventions every tool shares

**Handles, not queries.** A record is addressed by an id, a unique id prefix of 8 or
more characters, or an exact tag. Tags are case-sensitive. No tool finds, ranks or pairs
records by lookalike text.

**Receipts.** Every write returns a receipt id, an `eff-` id for the effect the gate
admitted. `receipt` `get` shows what the write did, and `receipt` `replay` re-derives it
from the log. Writes that return one directly: `smart_ingest` (and one per declared link),
`memory` `promote`, `demote` and `edit`, `suppress`, `intention`, `ghostlink` `weave` and
`harden`.

**Scope.** Records default to scope `user`. Walks, GhostLink proposals, `dream` and
`codebase` read one scope unless you pass `includeCrossScope` or `allScopes`. An id,
a prefix or a tag handle resolves across scopes.

**Typed refusals.** A call the log cannot honor says so, with a stable code in the
message:

| Code | Meaning |
|------|---------|
| `similarity_disabled` | The call asked for lookalike text: free-text `recall`, `recall` modes `reason` and `contradictions`, `session_start` `queries`, `ghostlink` `predict` with `current_topics`, `ghostlink` `map` with `query`. Pass an exact handle |
| `handle_required` | `recall` found no record under that handle |
| `ambiguous` | A `recall` prefix matched more than one id. The response lists the `candidates`. Pass a longer prefix or the full id |
| `needs_report` | `causal_walk` had no usable start point. It names the one missing piece and does not guess |
| `unavailable_in_4_0` | The action is withheld on a Strata log. See [Withheld in 4.x](#withheld-in-4x) |
| `pending_strata` | The action is not admitted on the Strata log yet |

**Honest parts.** A part of an otherwise working response that needs a capability this
build lacks is `{"status": "unavailable", "reason": "...", "detail": "..."}` with no count
and no list. A zero from a tool that did run means it looked and found none. An empty
walk, dream or proposal says why in `emptyBecause` or `reason`.

**Annotations.** MCP annotations describe the whole tool. `ghostlink` is not read-only
because `weave` and `harden` write. `causal_walk` is not marked read-only either, though a
Strata walk writes nothing.

---

## The tools

### `recall`

Find records by exact handle: `handle` is a record id, a unique id prefix (8+ characters)
or an exact tag. Lookup order is exact id, then unique prefix, then exact tag.

- A tag returns **every** live record carrying it, from every scope, with `nodes` and the
  one-hop recorded `neighbors` (direction, `link_type`, strength; at most 20 edges). It
  ignores `limit`.
- An ambiguous prefix is refused as `ambiguous`, with its `candidates`.
- A miss is `handle_required` (with `candidates: []`), not an empty list. Free text is
  never split into words to look for a tag or an id inside it (4.2.0): only the whole
  string is tried as a handle.
- `query`, `mode: "reason"` and `mode: "contradictions"` return `similarity_disabled`.
- A read never changes strength. Promote what helped with `memory`.

CLI: `vestige recall --handle <handle>`.

### `smart_ingest`

Save one record (`content`) or up to 20 (`items`), never both. Fields: `node_type` (free
text: `fact`, `concept`, `event`, `person`, `place`, `note`, `pattern`, `decision`),
`tags`, `source`, `scope`, `validFrom`, `validUntil`, `links`, `allowSecrets`.

- **Every create carries its proof (4.2.0).** The response holds `receiptId` (the `eff-`
  effect that wrote it), `canonicalHash` (blake3 of the content after NFC,
  zero-width stripping and whitespace collapsing, pipeline `nfc-zwstrip-wscollapse-v1`;
  case is kept, so `API_KEY` and `api_key` are different text),
  `entities` (typed spans with byte offsets, `CommitSha`, `Url`, `FilePath`, `IssueRef`,
  `Email`, `Version`, from the pinned hand scanners; the first 8, dropped when empty) and
  `importance` (`score` and `weightsVersion`, `linear-v1`; the factors are recomputable
  from the submitted bytes). Nothing in the write path uses a model or similarity.
- **Nothing is merged. A repeat reinforces.** A write whose canonical hash already exists
  in the scope, on a record that is still live, creates no twin and never touches the
  original: it records a small echo node (`source: "duplicate"`) linked `evidence_of` the
  original and answers `decision: "reinforce"` with `duplicateOf`, `echoNodeId` and the
  echo's own receipt. The echo carries the tags sent with the repeat, so a repeat sent
  with a new tag is found under it, and any `links` sent with it are written from the
  echo. Text whose record was edited away or suppressed is saved again as a new record.
  Different text is always a new record (`decision: "create"`); no gate compares anything
  by resemblance, and `predictionError` is always `1.0`.
- **`intent_id` makes a write idempotent.** A write carrying `intent_id` (up to 128
  characters of `A-Za-z0-9._:-`) is recorded once; sending it again, with the same or
  different content, answers `decision: "replay"` with `replayOf` and the original
  `intentDigest`, writes nothing, and reports `requestCanonicalHash` so a divergence is
  visible. The intent index is a side file in the data directory; backups carry it.
- **Auto-connect.** The new record is joined by `touched` edges to the records of its
  scope that record the same exact identity (a tag, a file path, a commit sha, an issue
  reference, a URL; never a word). All of one save's edges land as one write behind one
  receipt. The response's `autoConnect` block lists the edges with the identity each is
  joined on, `receiptId`, and `skipped`: every tag that did not join, its carriers out of
  the scope and the reason (a hub tag more than half the scope carries, or more pairs than
  the 100-edge budget of one write).
- **`links`.** Up to 16 typed edges from the new record to existing records in the same
  scope: `derived_from` (this record derives from it), `evidence_of` (this record is
  evidence about it), `closes` (this record closes it, stored as `closed_by`) and, since
  4.2.0, `supersedes` (this record fully replaces it; the old record stays intact and the
  edge is the paper trail). Each link is written through the gate with its own receipt, and
  batch `items` carry their own. Every link is checked before anything is written: the
  kind, that the target is a live record, that it is in the same scope, and that no link
  repeats. One refused link refuses the whole save. `corrects` is not declarable because
  it can retire a record. These edges are what `causal_walk` follows and what GhostLink's
  bridge lens walks.
- **Batches are content-ordered.** A batch of `items` processes them in canonical-hash
  order, so the same batch produces the same ids whatever order the items arrive in.
- **Validity.** `validFrom` and `validUntil` are exact RFC 3339 times, and `validUntil`
  must be after `validFrom`. Two defaults are applied when you pass none, and the
  response's `validity.source` names which one fired. A record of type `state` gets a
  `validUntil` 30 days out (`VESTIGE_STATE_TTL_DAYS`, `0` disables). And one literal rule
  reads the content: a single `as of YYYY-MM-DD` phrase sets `validFrom` to that date.
  Like the credential scanner, that rule reads the record's words at write time and does
  not touch retrieval. Pass `validFrom` yourself to skip it. Records that are expired or
  not yet valid are not GhostLink candidates.
- **Secrets.** The gate refuses AWS, GitHub, Slack, Google, Anthropic, OpenAI (`sk-ant-`,
  `sk-proj-`, `sk-svcacct-`, `sk-admin-`, legacy `sk-`) and Stripe live (`sk_live_`,
  `rk_live_`) keys in every stored field. Stripe test keys and a bare `password:` line
  are not blocked. `allowSecrets` overrides the gate, so set it only deliberately. A
  refusal names the credential kind and never echoes the value.
- `synapticCapture` and `tagSuggestionStatus` come back as typed `unavailable` parts
  (`synaptic_capture_unavailable`, `similarity_disabled`) with a plain `detail`. Nothing
  failed: tags are stored exactly as given.
- Batch results carry `batchOutcome` (`applied`, `no_changes`, `partial`, `failed`) and
  `atomic: false`. Earlier items stay committed when a later item fails.

CLI: `vestige ingest "<content>" --tags a,b --node-type decision`.

### `memory`

One record. Actions: `get`, `get_batch` (`ids`), `state`, `promote`, `demote`, `edit`.

- `state` reads the four accessibility bands (Active at 0.7 and above, Dormant at 0.4,
  Silent at 0.1, Unavailable below). On a Strata log all three strengths it reports equal
  one value, the card's retrievability.
- `promote` is an FSRS review rated Easy. `demote` is a review rated Again. Both return a
  `receiptId` and record an endorsement event for that exact content revision. An
  endorsement records who endorsed. It does not make the claim true.
- A demote does not delete. The record is still found by its id and tags, and it fades
  faster from the lower stability. Retrievability reads 1.0 right after any review.
- `edit` admits a successor and retires the previous node, which stays on the log. The
  successor starts a fresh FSRS card. Code anchors carry over.
- `purge` and `delete` return `unavailable_in_4_0`.

### `memory_status`

Views: `health` (default), `retention`, `timeline` (records by day, default 7 days),
`changelog`, `provenance`, `coverage`, `stats`, `tools`.

- `provenance` takes `memoryId` and returns the frame that admitted the record: its
  sequence number, gate sequence, operation, frame hash and payload hash.
- `coverage` counts anchors, edges by type and record ages. It says nothing about truth
  or causal linkage.
- `stats` counts by type, tag and age. `all_scopes` reads every scope.
- `changelog` with a single `memory_id` is withheld, because a Strata log does not record
  per-record state transitions. Use `provenance` or `receipt get`.
- `health` still carries some v3-era wording in `warnings` and `diagnostics`, such as a
  suggestion to run `consolidate`, which is withheld. Trust its numbers, not that advice.

CLI: `vestige stats`, `vestige health`.

### `session_start`

Status, open intentions, `needsBackup` / `needsDream` and codebase context under one
token budget (default 1000, counted as UTF-8 bytes divided by four). It writes nothing.

- `include_intentions`, `include_status`, `include_predictions`, `scope`, `changed_files`
  and `context` (`codebase`, `file`, `repoPath`, `topics`).
- `queries` is ignored with a notice that names the exact-handle alternative.
- With `context.codebase` it lists that codebase's patterns and decisions from the
  requested scope. Pass `repoPath` to check their anchors against the checkout. When the
  requested scope has none but other scopes hold the codebase's patterns, decisions or
  change records, `codeContext.elsewhere` lists those scopes with counts (4.1.1).
- Intentions show their id, their due date with the year, and an `OVERDUE` mark.
  `needsBackup` and `needsDream` come from the last backup and dream that completed.

### `receipt`

`get` shows what a write did, from an `eff-` receipt id or a record id: the origin frame,
the mutations, and an attestation that recomputes the hash chain and the signed trailers
before it attests. `replay` re-derives the state from the log and compares it to the
receipt: `matched`, `mismatches`, `stateDigest`, `replayedDigest`. Replay is read-only
and makes no claim about the world. The receipt of one save's auto-connect edges
(`autoConnect.receiptId`) lists every edge it wrote as an `edge_recorded` mutation, and
`replay` re-derives them all. `save_walk` is withheld.

### `causal_walk`

Walk a failure backward from explicit start points.

```json
{"name":"causal_walk","arguments":{"start_points":[{"kind":"logged_write","node_id":"mem-0000000000000005"}]}}
```

- **Start points.** `kind` is `failing_test`, `stack_frame`, `ci_run`, `logged_write` or
  `version_range`. **Every kind accepts `node_id`** (4.1.1), the id of the record that
  holds the symptom. Before 4.1.1 only `logged_write` could carry it. A start point with
  no `node_id` is not walked, and `start_points` says why.
- **Several start points walk together.** Each node lists the starts that reached it in
  `from`, the shallowest route wins, and a start that is also a recorded cause of another
  start keeps that edge in `causes`. `start_points` reports each as `walked`,
  `unresolved`, `not_in_scope` or `duplicate`, with a reason.
- **Direction.** The walk goes toward the cause. From a record it goes to what the record
  is `derived_from`, and to the records that are `evidence_of` it, that it closed
  (`closed_by`), or that `touched` it. A record derived from the start is downstream and
  is never reported as a cause. This is the 4.1.1 fix. On 4.1.0 the walk followed
  `derived_from` the wrong way.
- **One touched edge per path (4.2.0).** A `touched` edge records that two records name
  the same exact thing, and that does not carry over a second step, so a path follows at
  most one of them; `derived_from`, `evidence_of` and `closed_by` chain to the depth
  bound. `not_followed` counts the touched edges the walk saw and did not follow, the
  records behind them, and the identities their two ends share (hub tags left out), and
  `not_followed.held` lists those records one per record: `memory`, the reached record
  it hangs `from`, the `depth` it would have, the `shared` identities and the content.
  They are not causes and are not ranked with them; a tool that tests hypotheses
  (`vestige prove`) takes them as its farther tier.
- **Order with proof (4.2.0).** Causes are ordered by depth, then by the distinct exact
  identities a cause shares with its start (more first), then by the rarer identity
  first (fewer carriers in the scope), then by id. Each cause carries `rank`, `joined_on`
  (identity and carriers), `shared_count` and `not_counted_hub_tags`; `ranking` states
  the order and `scope_size`. The order arranges hypotheses; it proves nothing.
- **Bounds.** At most 8 hops and 500 nodes. Results are hypotheses, not proven causes.
- **No start point** returns `needs_report` and names what is missing.
- **An empty walk says why.** `emptyBecause` and `incomingEdges` count the causal edges
  that arrive from another scope (a walk does not cross scopes), the causal edges that lead
  downstream, and the incoming edges that are not causal, by link type.
- `promote` is accepted and answered: `promote: {"requested", "edges_persisted": 0, "note"}`.
  A recorded walk follows only edges the log already holds, so there is no trail to record.
  `scan_limit` and `lookback_days` apply to the legacy engine only.
- The camelCase spellings `startPoints`, `nodeId`, `loggedWrite`, `scanLimit` and
  `lookbackDays` are read like their snake_case names.

CLI: `vestige causal-walk --logged-write <id>` or `--node-id <id>`. Writes nothing. The
causes print as `#<n> <id> depth <d>` with the content on the next line, then the
not-followed summary and the held records numbered on; `--json` prints the response.
`vestige prove --logged-write <id> ...` runs the user's test on the leads (see README).

### `forgotten_lesson`

`failure_id` (required), `scope`. Walks backward from the failure over recorded causal
edges (`corrects`, `derived_from`, `evidence_of`, `closed_by`) and returns the fix or
lesson records whose FSRS retrievability was below 0.5, lowest first, each with its
`edge_path`, `retention_pct` and `recorded_at`. No entity overlap, keyword search or
inferred edge is used. Read-only.

CLI: `vestige forgotten-lesson <failure-id>`.

### `selftest`

No arguments. Plants a cause, an intermediate and a symptom with recorded `derived_from`
edges, plus decoys, in a throwaway log, walks back, and deletes the log. Eleven checks,
including that decoy edges, forward edges and a keyword distractor are ignored and that
the live store is unchanged. The live store is only read.

CLI: `vestige selftest`.

### `ghostlink`

Never-composed pairs, each with its proof from recorded structure only. `mode` is
`propose`, `bounty`, `weave`, `map`, `inspect`, `explore`, `predict` or `harden`.

- **`propose`**, `lens` `bridge` (default) or `divergent`. Bridge admits pairs within three
  undirected hops over recorded `touched`, `derived_from` or `closed_by` edges that were
  never woven, and the shortest path is the proof. Divergent admits pairs no recorded edge
  joins, scored `min(Path_min, 7)` times typed divergence. A pair whose members have no
  typed profile is a forced juxtaposition from a deterministic sampler, with no score.
  Filters: `scope`, `tags` (exact; both members must carry one), `includeCrossScope`,
  `limit`, and `cursor` for the divergent lens. An empty proposal says why, with counts.
  Each response carries `neverUses`, the list of what no lens uses.
- **`weave`** records what a tested pair showed: `first_id`, `second_id`, `outcome_type`
  (`helpful`, `dead_end`, `submitted`, `accepted`, `rejected`, `duplicate_risk`,
  `needs_poc`, `bad_severity`, `user_promoted`, `user_demoted`, `closed_by_scope`,
  `closed_by_duplicate`, `closed_by_false_assumption`, `closed_by_user`, `expired_lane`),
  optional `lens`. It writes a composition record plus a `derived_from` edge to each
  member, each with a receipt. The pair leaves both lenses. Suppressing a member withdraws
  the records composed from it.
- **`weave` `evidence` (4.1.1).** Up to 8 findings from outside, each with `url` (http or
  https), `sha256` of the fetched content, `retrievedAt` (RFC 3339) and an optional `note`.
  They are recorded on the composition record and tagged `evidence:<sha256>`, so
  `recall` finds the record by the hash alone. Every entry is checked before anything is
  written. Vestige never fetches the URL, and the release binary stays offline.
- **`bounty`** groups woven outcomes into lanes, with the bridge lens as the
  never-composed lane. **`map`** returns the recorded subgraph around `center_id`
  (`depth` 1 to 3, `max_nodes`). **`inspect`** reads woven compositions with `view`
  `recent`, `get` (`event_id`), `memory` or `neighbors` (`memory_id`). **`explore`** reads
  recorded typed paths with `kind` `chain`, `associations` or `bridges` (`from`, `to`).
  **`predict`** takes exact handles only (`context.current_file`).
- **`harden`** seeds invariant laws from `<data-dir>/ghostlink-laws.json`, then
  `~/.vestige/ghostlink-laws.json`, then six built-in laws. It is idempotent by law id and
  names a malformed laws file instead of skipping it.
- Reads never write to the log. `graph` still answers as a hidden alias.

CLI: `vestige compose --lens bridge|divergent`.

### `codebase`

Code knowledge anchored to files and symbols. Actions: `remember_pattern`,
`remember_decision`, `get_context`, `verify`, `reanchor`, `ingest_repo`. Always pass
`repoPath`, or the anchors are stored unverifiable.

- **`remember_pattern`** (`name`, `description`) and **`remember_decision`** (`decision`,
  `rationale`, `alternatives`) take `files` (`path#symbol`) or structured `anchors`.
- **`get_context`** reads one scope (`scope`, default `user`) or every scope with
  `allScopes` (the two together are refused). It lists patterns and decisions marked
  current or stale. `scopes` lists every scope that holds code memories for the codebase,
  each with `patterns`, `decisions` and `events` (change records) counts. `total` beside
  `count` shows what `limit` cut. An empty answer names the scopes that do hold some.
- **`verify`** needs an explicit `repoPath`. It checks at most `limit` (default 200, max
  1000) records of each type: patterns and decisions, and with a `codebase` its change
  records too. `checkedByType`, `totalByType`, `uncheckedByType` and `truncated` say what
  `limit` left unchecked. `reanchor` replaces reviewed evidence.
- **`ingest_repo` (4.1.1)** turns each non-merge commit of a local checkout into one
  `event` record in its own scope (the codebase name unless you pass `scope`). Each record
  has the handles `git-commit`, `codebase:<name>` and `commit:<sha>`, provenance
  `(git, <codebase>, <sha>)`, `valid_from` set to the commit's author time (clamped to
  now), and anchors on every touched symbol and file that exists in the checkout. Find one
  with `recall` `handle: "commit:<sha>"`, and re-check them all with `verify`.
  - It previews unless `dryRun=false`, because the log is append-only.
  - `repoPath` must be the top of a working tree. A subdirectory, a bare repository and a
    `.git` directory are refused with the directory to pass.
  - `limit` defaults to 100 and is capped at 500. Page back with `rev='<oldest sha>~1'`,
    which the response offers as `pageBackWith`. A write stops at a 45 second budget with
    `remaining` set.
  - Re-running skips recorded commits and gives any commit whose anchors never landed its
    anchors. A commit the secret gate refuses is skipped. A second write of the same
    codebase into the same scope while one runs is refused. Previews never are.
  - Git runs local-only with lazy fetching off, and with `-c log.showSignature=false -c
    core.fsmonitor=false`, `--no-show-signature`, `--no-ext-diff`, `--no-textconv` and no
    pager, so a hostile checkout's own config cannot make it run a program. If git stops
    partway, the whole commits read before the stop are kept and `gitStoppedEarly` says why.

### `intention`

Reminders and plans. Actions: `set`, `check`, `update`, `list`, `graph`. A `set` returns a
receipt. `list` and `check` honor `scope`.

- Triggers: `time` (`at`, `in_minutes`), `context` (`codebase`, `file_pattern`, `topic`),
  `event` (`condition`), `activity`, plus recurring and compound (`all_of`, `any_of`).
  Priorities are `low`, `normal`, `high`, `critical`. Statuses are `active`, `fulfilled`,
  `cancelled`, `snoozed`.
- `set` stores the `description` as written and never parses it (4.2.0). The trigger,
  deadline and priority are the ones you pass. With no `trigger` and no `deadline` the
  intention is manual, never fires on its own, and the response says so in `note`. No tag
  is added that you did not pass.
- `check` takes `context.current_time` (RFC 3339 with a timezone) and uses that clock for
  trigger, deadline and snooze comparisons. `include_snoozed=true` shows snoozed records
  but does not let them fire before `snoozedUntil`.
- **Matching is exact (4.2.0).** A trigger fires on a clock, or when a stored value equals
  a handle the check declares, byte for byte: `condition` or `activity` against
  `context.event` and each entry of `context.events`, `codebase` against
  `context.codebase`, `file_pattern` against `context.file` (an exact path, not a glob),
  `topic` against each entry of `context.topics`. No substring, no case folding. Before
  4.2.0 a context or activity trigger fired on a case-insensitive substring; one written
  for that is still stored and listed, and fires only on the exact value. `list` and
  `check` put `triggerMatching` on every text-keyed trigger: `firesOn` names the context
  field and the value to pass. A trigger that cannot be evaluated carries
  `invalid_trigger` and is never dropped from the list.
- `list` returns one documented order, echoed as `order`: priority high to low, creation
  time old to new, then id.
- Dates, trigger types, priorities, statuses and duration bounds validate before anything
  is written. A completed or cancelled intention cannot be snoozed back to active.
- `graph` evaluates evidence-aware plans through a nested `command`: `plan`, `revise`,
  `observe`, `evaluate`, `explain`, `portfolio`, `complete`, `cancel`, `acknowledge`,
  `replay`, `memory_snapshot`, `refresh_memory`. It rejects an `at` more than 24 hours
  ahead. Nothing runs in the background: a trigger fires only when something calls
  `check` with matching context.

### `maintain`

Actions on a Strata log: `dream`, `dream_compile`, `gc`, `importance_score`, `backup`,
`export`.

- **`dream`** pages through one scope's live records (`memory_count` 5 to 500, default 50,
  `after` cursor, `max_pairs`). It needs at least 5 live records on the page. It replays
  the recorded edges whose strength is at or above `min_similarity` (here the edge-strength
  floor, default 0.5) and folds one FSRS review per endpoint, with the rating taken from
  that card's own state. It reads no content, records the pass so `needsDream` reflects it,
  and returns `reviews`. `discovery` is `unavailable`, and when no edge was replayed,
  `emptyBecause` says so.
- **`dream_compile`** ranks the top `memory_count` live records by retrievability and
  replays their recorded edges. An edge with both ends replayed gains 0.1 strength (cap
  1.0). A weak edge (under 0.5) with one end outside the replay set is scaled by 0.95.
  `corrects` edges among the replayed records count as contradictions, and `derived_from`
  and `evidence_of` edges count as insights. It files no review items, rewrites no record,
  and does not record a dream-history entry. The four phases it reports are named
  `NREM1_Triage`, `NREM3_Consolidation`, `REM_Creative` and `Integration`.
- **`gc`** is a stub on the MCP tool. It always reports zero candidates and deletes
  nothing, because the log is append-only. The CLI `vestige gc --dry-run` lists records
  below a retention threshold, and `vestige gc` without `--dry-run` is refused.
- **`importance_score`** takes `id`, a full record id, and scores that record from its
  recorded structure (4.2.0): `score` is `edges.total + reviews.count - reviews.lapses`,
  and `computedFrom` returns every input, the typed edges that touch the record (`total`,
  `incoming`, `outgoing`, `byKind`) and its FSRS reviews (`count`, `lapses`; a demote is a
  lapse). It reads no content, tag or name, writes nothing, and returns the same bytes for
  the same store and id. Passing free-text `content`, which the v3 word heuristics scored,
  is refused with `unavailable_in_4_0` and names `id`. The word scorer is compiled out of
  the default build, so `smart_ingest`'s `importanceScore` is `0.0` there.
- **`backup`** writes a `vestige-<time>.strata` folder into `<data-dir>/backups/` and
  reports its path and size. **`export`** writes `memories-<time>.json` or `.jsonl`
  (`format`, `since`) into `<data-dir>/exports/`. Format `portable` is withheld.
- `consolidate` and `restore` are withheld. See below.

### `dedup`

Actions on a Strata log: `scan` (default, read-only), `undo`, `tag_rename`, `tag_merge`,
`policy`.

- `scan` groups exact duplicates only: identical content hash (or byte-identical content)
  and identical declared source keys. `mergeCandidates` is `{"status": "unavailable",
  "reason": "embeddings_unavailable"}`, and `nextStep` says merge planning is withheld.
- `tag_rename` (`source_tag`, `target_tag`) and `tag_merge` (`source_tags`, `target_tag`)
  preview first and apply only with the returned `preview_token`. Tags are matched exactly.
- `undo` lists recent reversible operations, or reverses one by `operation_id`. Undoing a
  logged write needs `confirm=true`. Undoing an edit leaves the pre-edit record live with
  its code anchors and retires only the edit.
- `policy` reports `matchThreshold`, `possibleThreshold` and `autoApply`. They are v3
  similarity settings and do nothing here, because merge planning is withheld.

### `project`

Renders the durable subset of a scope (decisions, patterns, rule-tagged facts) into a
fenced region of `CLAUDE.md` or `MEMORY.md`, one record id per line. `action` is `preview`
(default, shows the diff) or `write`, which needs `confirm=true` and replaces only the
fence. Options: `format` (`claude-md`, `memory-md`), `scope`, `max_items` (up to 500),
`filters.min_retention`, `path`, `root`.

- Only a well-formed fence is replaced: real marker lines outside code blocks. A stray or
  unclosed marker leaves the text untouched.
- An existing target is read only if it is a regular file of at most 2 MiB, so a pipe or
  special file cannot stall it. A filesystem-root `root` is refused.

CLI: `vestige project --out <file>`. The default target is `./CLAUDE.md`, so always pass
`--out`.

### `suppress`

Takes a record out of every read. `id` is required, `reason` is optional. The log keeps
the bytes. Returns a `receiptId`. It is **not erasure**, and on a Strata log it cannot be
undone: `reverse=true` is refused with `unavailable_in_4_0`. A leaked secret must be
rotated, because its bytes stay on the log. `cascade_derived_from` is advertised, but on
4.1.1 its targets report `pending_strata` and are not retired, so suppress records one at
a time.

---

## Withheld in 4.x

Each of these is absent from the advertised schema and refused with the reason.

| Call | Why |
|------|-----|
| `purge`, `delete_knowledge`, `memory` `purge` / `delete` | On an append-only signed log they could hide a record but not erase its bytes. Real erasure is planned as crypto-erasure ([#402](https://github.com/samvallad33/vestige/issues/402)) |
| `maintain` `consolidate` (4.1.1) | Every phase is a no-op on Strata, and an all-zero reply read as a completed pass |
| `maintain` `restore` | Restore a backup by copying its `log/` back |
| `maintain` `importance_score` with free-text `content` (4.2.0) | Scoring text needs word heuristics. Pass `id` to score a record from its recorded edges and reviews |
| `maintain` `export` format `portable` | Not written from a Strata log |
| `dedup` `plan_merge`, `plan_supersede`, `apply`, `verdict` | They need embeddings |
| `dedup` `protect` | The log has no protect flag yet |
| `receipt` `save_walk` | Walk receipts are not recorded on a Strata log |
| `suppress` `reverse` | A Strata suppression cannot be undone |
| `memory_status` `changelog` with one `memory_id` | A Strata log keeps no per-record state transitions |
| `source_sync` | Needs `--features connectors`, which no 4.x release build passes. The name is an unknown tool |

The dashboard's Consolidate button answers `501` with `unavailable_in_4_0` for the same
reason as `maintain consolidate`. Dashboard routes for features a Strata log withholds
answer `501` with `unavailable_in_4_0`, `similarity_disabled` or `pending_strata`, the
same words an MCP caller sees.

## Admission and the CI sweep

CI runs an admission sweep over stdio: every tool action the server advertises must be
admitted, or the build fails. That is why the advertised schema leaves out what the log
cannot honor.

## v3 contracts (not used by the 4.x default build)

Earlier versions of this page described the v3 engine: `recall` modes `reason` and
`contradictions`, `backfill` with `promote`, embedding-backed duplicate scan and merge
planning, suppression reversal windows and cascades, `maintain consolidate` phases,
portable restore, context packets and Memory PR review modes. Each depended on
embeddings, keyword scoring, inferred links or SQLite, and none is part of a Strata log.
The code stays behind the `legacy-sqlite` and `v3-engine` features for the harnesses
that test it. For how a v3 store arrives in 4.x, see
[Migrating to Vestige 4.0](MIGRATING-v4.md).
