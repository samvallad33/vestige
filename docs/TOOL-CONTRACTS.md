# Tool contracts

> The 19 tools a Vestige 5.0.1 server advertises, what each action takes, returns and writes

Vestige advertises nineteen MCP tools, sorted by name. Most are action multiplexers:
`memory`, `codebase`, `maintain`, `dedup` and `backfill` take an `action`,
`memory_status` takes a `view`, and `ghostlink` takes a `mode`. An agent should choose
the action its task needs. There is no requirement to call every tool in every session.

Statements here come from the 5.0.1 code under `crates/` and its tests. The parts that
did not change since 4.2.0 were also checked by running the 4.2.0 binaries against a
scratch store; the 5.0.0 additions (`gather`, `precedent`, `ghostlink` `heal`, `memory`
`get` with `reason`, the per-record `changelog`) and the 5.0.1 additions (`backfill`,
salience-ranked tag recall, `smart_ingest` `entities` and `attributes`, the
`session_start` `memory_hook` packet) were read from the code. Where an older doc and
the code differ, the code wins.

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
`harden`. A `backfill` change returns its own salience receipt, which `backfill`
`receipts` lists and `backfill` `undo` takes.

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
because `weave`, `harden` and `heal` write. `causal_walk` is not marked read-only either,
though a Strata walk writes nothing. The read-only tools are `forgotten_lesson`,
`memory_status`, `precedent`, `selftest` and `session_start`. `gather` is the one
open-world tool, because it reads GitHub through the `gh` CLI. `ghostlink` `heal` also
runs `gh` or a local test suite, but `ghostlink` is not marked open-world.

---

## The tools

### `recall`

Find records by exact handle: `handle` is a record id, a unique id prefix (8+ characters)
or an exact tag. Lookup order is exact id, then unique prefix, then exact tag.

- A tag returns live records carrying it, from every scope, with `nodes` (`id`, `type`,
  `content`, `tags`, `createdAt`) and the one-hop recorded `neighbors` (direction,
  `link_type`, strength, and a 200-character preview, 80 when repeated; at most 20 edges;
  pass `detail_level: "full_neighbors"` or call `memory(action="get")` for full neighbor text).
  Typed tags such as `person:Maya` are tags like any other.
- **Order (5.0.1).** Records are ranked by the net salience change their `backfill`
  receipts recorded (largest first), then FSRS stability (higher first), then newest
  first, then id. Before 5.0.1 a tag returned newest first. The order is the same on
  every read of the same log.
- Honors `limit` (cutting after that ranking and reporting `totalNodes` when truncated)
  and `detail_level` (`brief`, `summary`, `full`, `full_neighbors`).
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
`tags`, `entities`, `attributes`, `source`, `scope`, `validFrom`, `validUntil`, `links`,
`allowSecrets`.

- **`entities` and `attributes` (5.0.1).** Up to 64 typed strings each, stored as exact
  tags on the record. `entities` names who or what the record is about, as
  `<kind>:<name>` with kind `person`, `place`, `food`, `file`, `service`, `symbol`,
  `path`, `commit`, `issue` or `url` (`person:Maya`, `place:Bangkok Garden`).
  `attributes` states facts such as `allergy:peanut` or
  `preference:ambience=quiet`. Nothing is inferred from the content. Every batch item is
  checked before the first write.
- **Backfill on every save (5.0.1).** After the save is admitted, the server runs one
  `backfill` pass with the new record as its trigger, unless `VESTIGE_BACKFILL_AUTOFIRE`
  is `0`, `false`, `off` or `no`. A pass that fails is logged and never turns an admitted
  save into a failure; a manual `backfill` run can repeat it.

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
  receipt. The response's `autoConnect` block gives `edges` (how many were written),
  `pairs` (at most 5, strongest first, each with the identity it is joined on), `pairsTotal` and
  `pairsShown` (when `pairsTotal` is larger, the rest are in the receipt), `receiptId`
  (its receipt lists every edge), and `skipped`: every tag that did not join, its carriers out of
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
- **`get` with `reason` (5.0.0)** is an agent showing that record to its user. When the
  `reason` is not blank and the record is found, the reply gains `dashboard`: a `url`,
  `http://127.0.0.1:<port>/dashboard/memory?id=<id>`, and a `note` to give the user that
  link. The port is the one this process's dashboard serves on, else
  `VESTIGE_DASHBOARD_PORT`, else 3927. The server also emits a `MemoryShown` event with
  the client's `initialize` name and the first 2000 characters of the reason, and every
  open dashboard goes inside that record and shows the reason. A `get` with no reason,
  another action, or a miss does neither.
- `purge` and `delete` return `unavailable_in_4_0`.

### `memory_status`

Views: `health` (default), `retention`, `timeline` (records by day, default 7 days),
`changelog`, `provenance`, `coverage`, `stats`, `tools`.

- `provenance` takes `memoryId` and returns the frame that admitted the record: its
  sequence number, gate sequence, operation, frame hash and payload hash.
- `coverage` counts anchors, edges by type and record ages. It says nothing about truth
  or causal linkage.
- `stats` counts by type, tag and age. `all_scopes` reads every scope.
- `changelog` with `memoryId` (5.0.0) takes a `mem-` id or a UUID and returns that
  record's write history: `operations`, the writes that name it among the newest 1000
  writes of live records, up to `limit` (default 20, max 100), with `totalOperations`.
  Each has `operationId`, `opType` (`write` or `undo`), `status` (`applied` or
  `reverted`), `timestamp`, `revertedAt` and `reason`. `transitions` is always empty,
  because a Strata log keeps no per-record state machine. Before 5.0.0 a `mem-` id was
  refused as not a UUID.
- `changelog` with the snake_case `memory_id` is still refused with `unavailable_in_4_0`,
  and the advertised schema still calls that field legacy-only. Send `memoryId`.
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
- **`memory_hook` (5.0.1)** returns a context packet for a local lifecycle hook instead
  of the session summary. It takes `prompt` (up to 64 KiB), `limit` (1 to 8, default 3)
  and `trigger`, and reads one `scope`. Live records only: retired, expired and
  not-yet-valid records are left out. When the prompt names a recorded typed entity
  (`person:`, `place:`, `food:`, `file:`, `service:`, `symbol:`), as the whole tag or
  its name, matched case-insensitively on word boundaries, only records carrying those
  entities are kept, and `selection` is `exact_recorded_entities`; otherwise it is
  `top_salience`. Prompt words select entities and never rank. The packet ranks by
  Backfill salience, then stability, then recency (`ranking_signal`), and returns
  `memories` (content cut to 700 characters, with `salience_delta_q`), up to 8 of their
  `backfill_receipts`, and `backfill_autofire`. With entities matched and
  `trigger.mode` `light`, the limit is 8. It writes nothing.

### `receipt`

`get` shows what a write did, from an `eff-` receipt id or a record id: the origin frame,
the mutations, and an attestation that recomputes the hash chain and the signed trailers
before it attests. `replay` re-derives the state from the log and compares it to the
receipt: `matched`, `mismatches`, `stateDigest`, `replayedDigest`. Replay is read-only
and makes no claim about the world. The receipt of one save's auto-connect edges
(`autoConnect.receiptId`) lists every edge it wrote as an `edge_recorded` mutation, and
`replay` re-derives them all. `save_walk` is withheld.

`prove` builds the proof bundle for one commit: `repoPath` is the absolute path of the checkout,
`commit` defaults to `HEAD`, and `out` is `file`, `notes`, `both` or `none`. It takes the Reflex test
witnesses tagged with the commit's tree, without `.vestige/proof`, keeps the ones whose signature
checks out and whose tree did not change during the run, and adds the Failure Pairing `closes` edges
and the Operator stops of the same sessions. It returns the claims, a summary count per status, the
signer's `did:key`, the paths written and the bundle's SHA-256. See
[Proof-carrying pull requests](PROOF-CARRYING-PRS.md).

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
- **A recorded commit cause is kept (5.0.0).** When the walk resolves the failure's
  revision to a commit record, that record is context, not a cause, once the walk has
  resolved files, anchors or blame of its own. When the revision is the only thing it
  resolved, it came from the start's own `derived_from` edge, and the commit is reported
  as a cause. 4.2.0 dropped it, so a walk whose only cause was that commit came back
  empty.
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

### `backfill`

Retroactive Salience Backfill (5.0.1): a new record raises or lowers older records that
share an exact typed identity with it. Actions: `run` (default), `receipts`, `undo`.
`scope` defaults to `user`. Every `smart_ingest` save already runs one pass (see
[`smart_ingest`](#smart_ingest)); call this tool to preview, repeat, review or undo.

- **`run`** takes `trigger_id` (required; `failure_id` is a legacy alias, and the two
  must agree when both are sent), `manual`, `dry_run`, `lookback_days` (1 to 365,
  default 30) and `scan_limit` (1 to 500, default 500). The trigger must be a live record
  in the scope. Backfill never guesses a trigger.
  - A pass does work only when the trigger carries a personal attribute tag
    (`allergy:`, `preference:<key>=<value>`, `health:`, `condition:`, `medication:`,
    `diet:`), reads like a failure (a fixed whole-word list such as `error`, `crash`,
    `timeout`), or `manual` is true. Otherwise it returns no changes with that reason.
  - Candidates are older records in the scope, inside the lookback, that carry one of
    the trigger's `person:`, `place:`, `food:`, `file:`, `service:` or `symbol:` tags
    exactly. Records the causal walk reaches from the trigger come first, then newer
    before older, then id. A record already changed for this trigger is skipped, so a
    repeated pass changes nothing.
  - A personal attribute links through exactly one `person:` tag. A raise multiplies
    stability by 1.5 (at most +365 days). A lower divides it by 1.5 (never below 0.01)
    and needs an explicit signal: a newer `preference:` value for the same person and
    key, or, with `manual`, a `contradicts:<id>` or `supersedes:<id>` tag on the trigger.
  - At most 16 changes per pass. Each is applied only if the stored stability still
    equals the planned value; otherwise the call is refused and can be run again.
  - Returns `changes`: in a preview, each planned change with `memory_id`,
    `trigger_id`, `shared_entity`, `reason`, `old_score`, `new_score` and `direction`;
    when applied, each change's receipt. Then `changed`, `would_change`,
    `scanned`, `truncated`, `change_budget_reached`, `causality_verified: false` and
    `ranking_signal`. `dry_run` (or the legacy `promote: false`) previews and appends
    nothing, not even a tool trace.
- **`receipts`** lists the salience receipts whose record is in the scope.
- **`undo`** takes `receipt_id` and restores that record's exact earlier stability.
  With `dry_run` it shows the receipt it would undo. A receipt for a record outside the
  scope is refused.

A change is a statement of relevance, not of cause. Reads never change a score; tag
`recall` and the `session_start` `memory_hook` packet rank by the receipts.

CLI: `vestige backfill <trigger-id>` with `--manual`, `--dry-run`, `--lookback-days`,
`--scope` and `--json`; `vestige backfill receipts`; `vestige backfill undo <receipt-id>`.

### `selftest`

No arguments. Plants a cause, an intermediate and a symptom with recorded `derived_from`
edges, plus decoys, in a throwaway log, walks back, and deletes the log. Eleven checks,
including that decoy edges, forward edges and a keyword distractor are ignored and that
the live store is unchanged. The live store is only read.

CLI: `vestige selftest`.

### `ghostlink`

Never-composed pairs, each with its proof from recorded structure only. `mode` is
`propose`, `bounty`, `weave`, `map`, `inspect`, `explore`, `predict`, `harden` or `heal`.

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
  written. Vestige never fetches the URL.
- **`bounty`** groups woven outcomes into lanes, with the bridge lens as the
  never-composed lane. **`map`** returns the recorded subgraph around `center_id`
  (`depth` 1 to 3, `max_nodes`). **`inspect`** reads woven compositions with `view`
  `recent`, `get` (`event_id`), `memory` or `neighbors` (`memory_id`). **`explore`** reads
  recorded typed paths with `kind` `chain`, `associations` or `bridges` (`from`, `to`).
  **`predict`** takes exact handles only (`context.current_file`).
- **`harden`** seeds invariant laws from `<data-dir>/ghostlink-laws.json`, then
  `~/.vestige/ghostlink-laws.json`, then six built-in laws. It is idempotent by law id and
  names a malformed laws file instead of skipping it.
- **`heal` (5.0.0)** weaves a verdict from a recorded event instead of an outcome someone
  types. It needs a Strata log; a legacy store refuses it. `source` is `poc` or `github`.
  Its fields are read from the call but are not in the advertised schema, so
  `memory_status` with `view: "tools"` does not list them.
  - **`source: "poc"`** takes `repoPath` (a directory) and `evidenceUrl` (the http or
    https URL the leads track), and optionally `testCommand`, `limit` (default 3) and
    `scope` or `includeCrossScope` (with neither, leads come from every scope). It takes
    the top bounty leads (needs-PoC lanes first, then never-composed, then already
    composed) and runs the suite once in `repoPath`: `testCommand` split on spaces, else
    `forge test` when `foundry.toml` exists, else `npx hardhat test` when a
    `hardhat.config.js` or `.ts` exists. The output is saved to
    `~/Downloads/vestige-heal-runs/poc-<time>.log`. Every lead is woven with one outcome:
    exit 0 is `submitted`, a failing run `dead_end`, a suite that could not run
    `needs_poc`. The evidence is `evidenceUrl` with the log's sha256. It returns `suite`
    (`command`, `exit`, `outcome`, `logPath`, `logSha256`), `leadsProven`, `woven` (each
    pair's `recordId`, or `skipped` with the error) and `bountyAfter`. A scope with no
    bounty leads is refused.
  - **`source: "github"`** takes `issueUrl` (`…/issues/<n>` or `…/pull/<n>`), `firstId`
    and `secondId` (camelCase here, unlike `weave`'s `first_id`). It reads the issue with
    `gh api` and maps recorded state only: closed as a duplicate weaves `duplicate_risk`, a
    merged pull request `accepted`, any other close `rejected`, and an open issue with a
    comment by an `OWNER`, `MEMBER` or `COLLABORATOR` among its first 20 `needs_poc`. An
    open issue with no such comment is refused, and nothing is written. Only a `/pull/`
    URL is checked for a merge, so a closed issue that is not a duplicate weaves
    `rejected`. The evidence is the issue's URL with the sha256 of its JSON. It returns
    `outcome`, `reason`, `state`, `stateReason`, `merged`, `triaged` and `record`, the
    weave with its receipts.
- Reads never write to the log. `graph` still answers as a hidden alias.

CLI: `vestige compose --lens bridge|divergent`.

### `gather`

A GitHub issue hunt recorded as receipts in one scope, then the walk (5.0.0). Fields:
`url` (required), `repoPath`, `workedIn`, `brokeIn`, `scope` (default
`<owner>-<repo>-<number>`) and `maxComments` (default 50).

- **It reads GitHub with `gh api`**, so the `gh` CLI must be on the server's `PATH`. A
  failed issue read fails the call; a failed timeline read leaves out the pull requests.
  `url` is an issue or pull request URL ending `/issues/<n>`, `/pull/<n>` or `/pulls/<n>`;
  a `#fragment` or `?query` is dropped.
- **What it writes, in order.** Every record is an `event` in the one scope. The issue,
  pull request and comment records pass the secret gate with no override and carry the
  source `github:<owner>/<repo>/#<n>`.
  1. With `repoPath`, the commit window: `codebase` `ingest_repo` on that checkout, up to
     10 pages of 200 commits paging back from `HEAD`. The codebase name is the checkout's
     directory name. Commits go first so the walk can reach them from the issue.
  2. One record per pull request that cross-references the issue in the first 100
     events of its timeline, tagged `<repo>`, `gather` and `linked-pr`.
  3. The issue, `Issue <owner>/<repo>#<n> [<state>]: <title>. <body>`, tagged `<repo>`,
     `failure`, `gather` and the issue's labels. This is the failure record.
  4. One record per comment with a non-blank body, oldest first, up to `maxComments`
     (read from `repos/<owner>/<repo>/issues/<n>/comments` in pages of up to 100), tagged
     `<repo>`, `gather` and `comment`.
     The pull request, issue and comment records all name the issue as
     `<owner>/<repo>#<n>`, an issue identity auto-connect joins on whatever the hub rule
     does to their shared tags.
  5. `touched` edges from each record it wrote in steps 2 to 4, through the same pass
     `smart_ingest` runs at ingest: to the records of the scope that share an exact
     identity with it (a tag, a file path, a commit sha, an issue reference or a URL),
     hub tags skipped, at most 100 edges per record in one write. Commits are not joined
     to each other here.
- **Then it reads.** `ghostlink` `propose` with the bridge lens over the scope, and
  `causal_walk` from the issue record as a `logged_write` start point in the scope. Causes
  are hypotheses.
- **Returns** `issue`, `scope`, `recorded` (`failureId`, `failureScope`, `linkedPrIds`,
  `commentIds`, `commitPages`, `commitsCreated`, `commitsCovered`, `bridgeEdges`,
  `bridgeNotLinked`: the candidates the per-record budget left unjoined), `leads`
  (`ghostlinkPairs`, `note`), `walk` (`causes` as a count, `top` as the first cause,
  `emptyBecause`) and `missingReceipts`. The ids are record ids; `receipt` `get` takes
  them.
- **`missingReceipts`** names what was missing instead of inventing a cause: no
  `repoPath`; no `workedIn` or no `brokeIn`; the issue record outside the scope; or a
  `repoPath` that covered no commits and a walk that found none. `workedIn` and `brokeIn`
  do nothing else in this build: they do not bound the commit window or the walk.
- **Not idempotent.** A second run records the issue and its pull requests again. Commits
  already in the scope are skipped and counted in `commitsCovered`.
- **A refused write stops the call.** When the secret gate refuses the issue, a pull
  request or a comment record, the call fails and the records written before it stay.
  `ingest_repo` skips a commit the gate refuses.
- **Background store work.** Before each store acquisition it waits, up to 2 seconds
  each time, while another caller holds or waits for the store, so dashboard reads and
  other agents' calls go first. The call still returns only when the hunt is done. Time
  spent waiting does not count against each commit page's 45 second write budget.

`memory_status` with `view: "tools"` and `tool: "gather"` returns the full schema.

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

### `precedent`

What the Operator gate ruled, and why (5.0.0). It reads the gate's own files under
`$OPERATOR_HOME`, or `~/.operator` when that is unset: `receipts/YYYY-MM-DD.jsonl`, one
receipt per line, and `permits/<action_digest>.json`, `.json.used` or `.json.expired`. It
writes nothing, touches no store state, and approves or issues nothing. Before it answers,
it verifies the receipts' digest chain over the day files it reads (5.0.1, see below).

- **Filters**, all optional:
  - `since` (default `24h`): a number and a unit, `s`, `m`, `h`, `d` or `w`.
  - `cwd`: receipts whose `cwd` starts with this path.
  - `source` and `session`: exact match.
  - `rule`: a case-insensitive substring of the receipt's `rule`, `commitment` or any
    `commitments` entry, such as `OP-011`.
  - `decision` (default STOP and ALLOW): `STOP`, `ALLOW`, `SHADOW_STOP`, `SHADOW_TASTE`,
    `SHADOW_TASTE_MULTI`, `PASS`, `PERMIT_GRANTED`, `GATE_ERROR` or `all`, in any case.
    Any other value is refused.
  - `limit` (default 10): 1 to 50, newest first.
- **Returns** `summary`, `items` and `skipped_lines` (lines that are not JSON or carry no
  readable `ts`). `summary` reads `<n> stops in the last <since>`, plus `<k> permits used`
  when an ALLOW came through a permit, both counted before `limit`. Each item has `ts_utc`,
  `ts_local`, `decision`, `source`, `session`, `cwd`, `tool`, `action_preview`, `rule` (or
  the `commitment`), `detail` and `digest`, plus `via`, `action_digest`, `commitment`,
  `prev` (the digest this receipt chains to), `normalized`, `transforms` and
  `gate_version` when the receipt has them.
- **The chain is verified first (5.0.1).** Every line of the day files read must name, in
  `prev`, the digest of the line before it, and must match its own `digest` (SHA-256 of the
  receipt without `digest`, as the gate's `digest_of` spells it). The first file read
  chains to the last line of the newest file before it, or to 64 zeros at the start of the
  chain. `chain` reports `status` (`ok`, `broken` or `empty`), `receipts_checked`, up to
  five `breaks` as `file:line` with `breaks_total`, `from_file`, `anchored_to`,
  `integrity` (`reference_digest_not_signature`) and a `note`. A receipt holding a
  non-integer number is linked but its own digest is not recomputed, because its Python
  spelling cannot be reproduced exactly; `digests_not_recomputed` counts them. A broken
  chain adds a `warning` naming the first break and `chain_break: true` on the items at a
  break. A digest chain detects accidental edits and removals; it is not a signature and
  proves no authorship.
- **Permits.** An item carries `permit` when one answers it: the permit for the same
  `action_digest` whose `stop_receipt` names the receipt's `digest`, or else the nearest
  by grant time, since older permits name no stop. An ALLOW takes the latest permit
  granted at or before it; a stop takes the earliest granted at or after it that names no
  stop. A permit shows `status` (`active`, `used` or `expired`, from the file suffix),
  `granted`, `expires`, `single_use`, `by`, and `stop_receipt`, `session` and
  `origin_source` when present. Its `note` says it is history, not permission: permits
  are single-use and cannot be reused or issued through this tool.
- **Secrets are masked.** Every string in an item, the permit included, has
  secret-shaped runs replaced by `[REDACTED]`: `sk-`, `ghp_`, `sk_live_`, `rk_live_` and
  `AKIA` keys, `Bearer` and its value, and `api_key`, `apikey`, `token`, `secret` or
  `password` followed by `=` or `:` and a value.
- Day files dated more than a day before the window are not read. With no `receipts`
  directory the answer is a `summary` that says so and empty `items`, not an error.

---

## Withheld in 4.x

Each of these is absent from the advertised schema and refused with the reason. All are
still withheld in 5.0.1.

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
| `memory_status` `changelog` with one `memory_id` | A Strata log keeps no per-record state transitions. The camelCase `memoryId` is answered (5.0.0), with the record's write operations and empty `transitions` |
| `source_sync` | Needs `--features connectors`, which no release build passes. The name is an unknown tool |

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
`contradictions`, the v3 `backfill` that joined records by words they shared,
embedding-backed duplicate scan and merge planning, suppression reversal windows and
cascades, `maintain consolidate` phases, portable restore, the v3 context packets and
Memory PR review modes. Each depended on embeddings, keyword scoring, inferred links or
SQLite, and none is part of a Strata log. The 5.0.1 `backfill` tool and `memory_hook`
packet above are new Strata mechanisms that share those names, not the v3 code.
The code stays behind the `legacy-sqlite` and `v3-engine` features for the harnesses
that test it. For how a v3 store arrives in 4.x, see
[Migrating to Vestige 4.0](MIGRATING-v4.md).
