# GhostLink compositions

> Written for Vestige 4.2.0. ComposedGraph was the v3 name for the composition ledger. In 4.x the `ghostlink` tool replaces it. `composed_graph` and `graph` are still dispatched as hidden aliases.

`ghostlink` finds pairs of records that no one has composed yet, and it records what came of the ones you test. Every pair carries its proof, built only from what the log recorded: ids, exact tags and types, typed edges, woven outcomes and FSRS state. Nothing is admitted, ranked, paired or explained by text, embeddings or keyword overlap. The `proof` object of each candidate lists what the lens never uses.

## The model

A **weave** is the write that records what a tested pair showed. On a Strata log it writes:

1. one composition record (`node_type: "composition"`), whose content is a fixed sentence naming the two ids, the outcome and the lens, plus any outside evidence. It is tagged `ghostlink`, `ghostlink-weave`, `outcome:<type>`, `lens:<lens>`, and `evidence:<sha256>` for each finding; and
2. one `derived_from` edge from that record to each of the two members.

Each write goes through the gate and returns its own `receiptId` in the `receipts` array of the response. The pair then leaves both lenses, and each member gains a typed profile through its new edge. Suppressing a member withdraws the composition records made from it, each with its own receipt.

The record keeps its members as plain ids. A member you later edit or suppress does not erase the fact that it took part.

### Outcomes

`outcome_type` is one of: `helpful`, `dead_end`, `submitted`, `accepted`, `rejected`, `duplicate_risk`, `needs_poc`, `bad_severity`, `user_promoted`, `user_demoted`, `closed_by_scope`, `closed_by_duplicate`, `closed_by_false_assumption`, `closed_by_user`, `expired_lane`.

Closed-door labels should be specific when possible. Prefer `closed_by_scope`, `closed_by_duplicate`, `closed_by_false_assumption`, `closed_by_user` or `expired_lane` over a generic `dead_end` when you know the reason.

## Calls

```json
{ "mode": "propose", "lens": "bridge", "limit": 10 }
```

```json
{ "mode": "propose", "lens": "divergent", "limit": 5 }
```

```json
{
  "mode": "weave",
  "first_id": "<memory id>",
  "second_id": "<memory id>",
  "outcome_type": "helpful",
  "lens": "bridge"
}
```

```json
{ "mode": "inspect", "view": "recent", "limit": 10 }
```

`inspect` takes `view` `recent`, `get` (with `event_id`, the composition record id), `memory` or `neighbors` (with `memory_id`). `map` reads the recorded subgraph around an exact `center_id`. `explore` takes `kind` `chain`, `associations` or `bridges` over recorded typed edges only. `predict` takes exact handles from `context` (`current_file`, `codebase`), and free-text topics are refused. `harden` seeds invariant laws. It writes, and it is idempotent by law id. Reading never writes to the log.

Pass `tags` (exact, and both members must carry one), `scope` or `includeCrossScope` to narrow `propose` and `bounty`.

## The two lenses

- **Bridge** (default): pairs within three undirected hops over recorded `touched`, `derived_from` or `closed_by` edges that were never woven. The shortest typed path is the proof. Pairs rank by hop proximity, how rarely each record has been composed, retention, and the outcomes earlier pairs recorded.
- **Divergent**: pairs that no recorded edge of any kind joins. They are scored by path length and by how few typed neighbors they share. If a record has no typed neighbors, nothing measures it, so the pair is a forced juxtaposition picked by a deterministic sampler, with no invented score. `legacy_inferred` links imported from v3 can only lower a score.

Each candidate has `score`, `noveltyScore`, `bridgeScore`, `trustScore`, `outcomeScoreAdjustment`, `priorOutcomes`, `outcomeSignal`, the node types, a short content preview for reading, a `reason`, a `compositionQuestion` to answer before you act, and a `proof`. `sharedTags`, `boundaryTags` and `sharedTerms` are empty lists, kept for older callers. An empty proposal says why, with counts.

The output is a queue, not a finding. A never-composed pair means "worth testing", not "true", "novel" or "reportable". The response carries `evidenceStatus: "hypothesis"` and `globalNoveltyVerified: false`. Prior outcomes are guardrails and not verdicts: a duplicate-risk signal should send you to check the duplicate family first, and a success signal should send you to see why the older composition worked.

## Bounty mode

`bounty` is a read shape for investigation. It returns the lanes already composed, the never-composed lanes (the bridge lens), closed doors, duplicate-risk lanes, lanes that need proof-of-concept work, and the top three combinations. Failed and duplicate compositions are kept, so they are not rediscovered. Its own guardrails say that a never-composed lane is not a finding, a composition score is not severity, and a reportable claim still needs source references, scope fit and proof-of-concept evidence.

## Outside evidence

`weave` takes up to 8 `evidence` findings: a `url`, the `sha256` of what was fetched, a `retrievedAt` time in RFC 3339, and an optional note. They are recorded on the composition record and tagged `evidence:<sha256>`, so `recall` with `handle: "evidence:<sha256>"` finds the record by the content hash alone. Vestige never fetches the URL, and the release binary stays offline.

## Not in 4.x

- `deep_reference` is a hidden alias for `recall` mode `reason`, which returns `similarity_disabled`, so it records no composition events. A weave is the only way a composition is recorded.
- The v3 `label` action on an event id is `weave` now. On a legacy SQLite store, `weave` with an `event_id` still labels an event.
- Triple or n-ary candidates, and "what combinations were available when this decision was made" replay, are not implemented.
