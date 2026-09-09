# Evidence-aware intentions

Vestige intentions connect a future plan to the premises and observations that
support it. The first implementation is a deterministic, local evaluator: it
can show which plans are affected when supplied evidence changes, reconcile a
completion against later evidence, and explain the current evaluation without
performing the planned action.

This document describes the implementation contract introduced by the
evidence-aware intention change. Reproducible local checks are linked below.
The research extensions described under [Boundaries](#boundaries) are not
implemented behavior.

## Public interface

The existing `intention` actions (`set`, `check`, `update`, and `list`) remain
available. Evidence-aware operations use `action: "graph"` and a nested
command:

```json
{
  "action": "graph",
  "scope": "user",
  "at": "2026-10-01T09:00:00Z",
  "command": {
    "action": "portfolio"
  }
}
```

`scope` defaults to `user`. It selects an independent local namespace; it is
not authentication, authorization, or a tenancy boundary. Use the host's real
access controls to isolate users who must not read or change each other's data.

`at` is optional and normally defaults to the current time. Supply an RFC 3339
timestamp for reproducible fixtures and deterministic evaluation. Treat it as
an evaluation clock supplied by the caller, not proof that an event occurred at
that time. It is not a historical-query interface: current-state reads and
state-changing commands with a timestamp earlier than the graph's last mutation
are rejected. An exact state-neutral retry can return its original result.

The nested commands are:

| Command | Purpose |
| --- | --- |
| `plan` | Create a plan with explicit requirements and conflict keys. |
| `revise` | Replace the mutable fields of an existing plan using its expected revision. |
| `observe` | Add a bounded, typed caller assertion for one source revision and reevaluate only affected plans. |
| `evaluate` | Evaluate a plan from the currently stored observations. |
| `explain` | Return the plan, requirement results, current source snapshots, pending attention, and evidence boundary. |
| `portfolio` | Return plans, shared prerequisites, declared conflicts, and queued interventions. |
| `complete` | Record a user report or evidence completion against an expected plan version. |
| `cancel` | Cancel an expected plan version while retaining its history. |
| `acknowledge` | Acknowledge a queued intervention so cooldown and deduplication state advance. |
| `replay` | Rebuild the graph from the committed command journal and compare stored digests. |
| `memory_snapshot` | Read the current same-scope memory content commitment without copying its text into the intention graph. |
| `refresh_memory` | Read that commitment and append it as a reserved local-memory observation. |

IDs are caller-chosen stable strings. Keep one plan ID across revisions instead
of creating a new plan to represent an edit. `revise`, `complete`, and `cancel`
use optimistic versioning: pass the version last returned by Vestige in
`expected_version` and handle a version conflict by reading the current plan
before retrying. This value guards the plan definition, not its entire lifecycle:
`revise` increments the definition version, while `complete` and `cancel` do
not. Cancellation deliberately moves the current version to `cancelled` from
another lifecycle state. An exact repeated completion or cancellation is a
state-neutral retry that preserves its original timestamps; changing the basis
or reason is rejected as a conflicting retry. Observation event IDs provide the
same rule for observations: an exact retry is idempotent, while reusing an
event ID with different content is rejected.

## Deterministic evidence model

A plan stores its desired outcome separately from the evidence-dependent means
of achieving it. Requirements refer to explicit source keys and predicates.
Observing a newer source revision updates the relevant source, finds the plans
that depend on it, and reevaluates that affected set. An unrelated source update
does not invalidate every plan.

Observations accept a bounded scalar value (boolean, signed integer, nonempty
string, or `null`) and explicit event lists. They are assertions from the caller.
Vestige records the source key, monotonically increasing source revision, event
ID, value, observed event types, covered event types, and evaluation time. It
does not infer a product specification from prose, fetch a URL, poll an inbox,
or decide that the assertion is authoritative.

Requirements have three explicit forms:

- `evidence` compares a source value with `exists`, `equals`, or `not_equals`;
- `event` requires an event to have occurred or remained absent, with explicit
  coverage and time-window rules for absence; and
- `plan_fulfilled` makes another plan's fulfilled lifecycle a prerequisite.

Absence predicates are coverage-aware. An expected event is absent only when
the observation declares coverage for that event type over the relevant input.
If the source was disconnected or its coverage is unknown, the result remains
unknown. A missing event in an uncovered channel is never turned into a false
"did not happen" claim.

Observation coverage cannot end after the command's evaluation time. When an
event requirement has a time window, an assertion that it occurred must carry
a bounded observation interval inside that required window; otherwise its state
remains unknown. Establishing absence is stricter: the requirement needs an
explicit window, the observation must cover that whole window and named event,
and the configured grace period must have elapsed.

The evaluator also exposes relationships across plans. Requirements with the
same normalized definition are shared prerequisites; explicit conflict keys
show plans that compete for the same declared resource or choice. These are
structural results from caller-supplied data, not semantic or causal inferences.

Each new logical evaluation of an active or disputed plan receives a stable
attention fingerprint. The queue deduplicates repeated states, supersedes an
older unacknowledged state for the same plan, and retains withheld entries until
their cooldown expires. Acknowledging a queue ID starts that plan's cooldown;
fulfilled and cancelled states retire older pending entries. Vestige returns
deliverable and withheld queue IDs for a host to present. It does not deliver a
notification itself.

Completion is evidence-aware without making ordinary personal reminders
needlessly strict. A completion basis is either `user_report` with a bounded
report string or `evidence` with a nonempty subset of the plan's requirement
IDs. Evidence-based completion is accepted only when every current requirement
is satisfied. The selected subset records which evidence the caller cites, but
later reconciliation still considers every current requirement: if any reverses,
the lifecycle becomes `disputed`. A disputed evidence completion must be revised
to a new plan version before it can be completed again. Later evidence does not
automatically dispute a `user_report`. Revising any fulfilled plan activates
the new version and marks the prior completion for reconciliation; cancellation
moves the current version to `cancelled`. No completion transition can cause
Vestige to repeat an external action.

## Projector fixture

The following scenario is synthetic. Product dimensions, bag fit, price,
availability, purchase, and cancellation are fixture assertions, not live
facts. The evaluator never buys the projector.

The plan keeps the desired outcome separate from the premise that a selected
projector fits a travel bag. This creates version 1:

```json
{
  "action": "graph",
  "scope": "projector-fixture",
  "at": "2026-10-01T09:00:00Z",
  "command": {
    "action": "plan",
    "id": "portable-projector",
    "description": "Synthetic: buy the selected projector on October 30, 2026",
    "priority": "normal",
    "requirements": [
      {
        "type": "evidence",
        "id": "fits-travel-bag",
        "source_key": "fixture:projector-spec",
        "condition": {
          "op": "equals",
          "value": true
        }
      }
    ],
    "min_attention_interval_seconds": 1209600,
    "conflict_keys": ["purchase:portable-projector"]
  }
}
```

The fixture can assert the premise without claiming it came from a live product
source:

```json
{
  "action": "graph",
  "scope": "projector-fixture",
  "at": "2026-10-01T09:05:00Z",
  "command": {
    "action": "observe",
    "event_id": "projector-spec-1",
    "source_key": "fixture:projector-spec",
    "source_revision": 1,
    "value": true,
    "observed_events": [],
    "covered_events": []
  }
}
```

If a newer supplied specification reverses that value, Vestige reevaluates
only plans that depend on `fixture:projector-spec` and queues the changed
readiness for review:

```json
{
  "action": "graph",
  "scope": "projector-fixture",
  "at": "2026-10-02T09:00:00Z",
  "command": {
    "action": "observe",
    "event_id": "projector-spec-2",
    "source_key": "fixture:projector-spec",
    "source_revision": 2,
    "value": false,
    "observed_events": [],
    "covered_events": []
  }
}
```

If the user then chooses another synthetic candidate, the caller revises the
same plan using the current version:

```json
{
  "action": "graph",
  "scope": "projector-fixture",
  "at": "2026-10-02T10:00:00Z",
  "command": {
    "action": "revise",
    "id": "portable-projector",
    "expected_version": 1,
    "description": "Synthetic: buy candidate B on October 30, 2026",
    "requirements": [
      {
        "type": "evidence",
        "id": "fits-travel-bag",
        "source_key": "fixture:candidate-b-spec",
        "condition": {
          "op": "equals",
          "value": true
        }
      }
    ]
  }
}
```

The response retains `portable-projector`, preserves the original description
in history, and appends version 2. Omitting a mutable field from `revise` keeps
its version-1 value. `min_attention_interval_seconds` is an attention cooldown
after acknowledgement; it is not a recurrence schedule or proof that an alert
was delivered.

This is the behavior the fixture is intended to demonstrate:

1. A plan is created with a stable ID, an explicit bag-fit requirement, and a
   declared purchase conflict key.
2. Source revision 1 says the selected projector fits the bag. Evaluation can
   show the requirement as satisfied.
3. Source revision 2 says it does not fit. The graph reevaluates the affected
   plan, records the changed premise, and queues one review intervention.
4. Repeating the same observation event does not create another state change.
5. A revision can change the candidate or requirements only when its
   `expected_version` matches the current plan version.
6. Evidence-based completion becomes disputed if a later observation
   invalidates any current requirement. No replacement order is placed.

## Local memory sources

Direct `observe` calls may not use a source key beginning with `memory:`. That
namespace is reserved so a caller cannot forge a local memory snapshot. Read a
same-scope commitment first:

```json
{
  "action": "graph",
  "scope": "user",
  "command": {
    "action": "memory_snapshot",
    "memory_id": "00000000-0000-4000-8000-000000000001"
  }
}
```

The result contains a source key such as
`memory:00000000-0000-4000-8000-000000000001`, the memory ID, and either a
SHA-256 content commitment in `value` or `null` when no currently available
same-scope memory can be committed. It never copies the raw memory text into
intention history. The adapter computes a digest of its local content snapshot;
the digest alone does not prove the read, its origin, or the remembered claim.
Unkeyed SHA-256 also does not conceal low-entropy content from guessing.

Bind a requirement only after a non-null commitment is available. A null
snapshot means the memory is unavailable; it is not an equality value.
Bind a requirement to the exact returned commitment by copying the snapshot's
`source_key` and `value` into the plan. For example, if `memory_snapshot`
returns a value of 64 `a` characters, the requirement is:

```json
{
  "type": "evidence",
  "id": "remembered-projector-spec",
  "source_key": "memory:00000000-0000-4000-8000-000000000001",
  "memory_id": "00000000-0000-4000-8000-000000000001",
  "condition": {
    "op": "equals",
    "value": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }
}
```

The digest above is obviously synthetic. In a real call, use the exact
`snapshot.value`; do not hash a summary or substitute a caller-computed value.

Append the current commitment as an observation with `refresh_memory`:

```json
{
  "action": "graph",
  "scope": "user",
  "at": "2026-10-01T09:00:00Z",
  "command": {
    "action": "refresh_memory",
    "memory_id": "00000000-0000-4000-8000-000000000001",
    "event_id": "projector-memory-refresh-1",
    "source_revision": 1
  }
}
```

Refresh is manual. The first implementation has no background memory watcher,
connector polling, or automatic refresh cadence. The caller owns the source
revision sequence and should refresh at a deliberate lifecycle boundary, such
as before evaluation or after updating a referenced memory.

## Persistence and replay

Each local scope has a current JSON snapshot and an append-only command journal
in the same SQLite database as the rest of Vestige. A command that changes state
commits the snapshot and journal row in one immediate transaction. SQLite
serializes writers that share that database, so concurrent local processes do
not commit the same next sequence number. An exact idempotent observation retry
does not append another journal row because it does not change state.

Each scope is deliberately bounded to 1,024 plans, 1,024 versions per plan, 128
requirements per plan, 4,096 sources, 16,384 observation receipts, 16,384
evaluations, 16,384 attention events, 20,000 committed changes, and a 16 MiB
serialized snapshot. A single recorded observation response is limited to
256 KiB; a change exceeding a resulting-state limit fails atomically. There is
no intention-graph archive or compaction command
in the first implementation. Before a scope reaches a cap, preserve a full
backup of the SQLite database and start an explicit new scope with reviewed
current plan definitions. The old scope keeps its history; Vestige does not
silently discard it or claim indefinite capacity.

`replay` starts with an empty graph, applies the recorded commands at their
recorded evaluation times, and compares every output digest and state digest
with the journal. It then compares the rebuilt graph with the current snapshot.
A successful replay establishes deterministic reproduction of the locally
recorded state transitions.

Replay does not:

- refetch a source or reverify an external fact;
- prove that a caller's observation was honest or authoritative;
- prove external delivery, purchase, completion, or cancellation;
- provide a cryptographic signature or protect against an attacker rewriting
  the entire database and recomputing its digests; or
- synchronize an intention graph across devices or separate databases.

Portable, distributed, and exactly-once external execution require additional
coordination. The local journal does not claim those properties.

## Existing triggers

The ordinary `set` and `check` path also supports the core trigger forms that
were previously lost or only partly evaluated by the public adapter:

- `activity` matches an explicit activity or condition against supplied context
  events;
- `recurring` accepts a named recurrence such as `daily`, `weekly`, or
  `fortnightly`, an `every ...` expression, or a bounded interval object, and
  can wrap a base trigger;
- `compound` recursively combines `all_of` and `any_of` triggers; and
- context fields supplied within one context trigger are conjunctive.

A fortnightly reminder can be created without a graph command:

```json
{
  "action": "set",
  "description": "Review the projector plan",
  "trigger": {
    "type": "recurring",
    "at": "2026-10-02T09:00:00-07:00",
    "recurrence": "every two weeks"
  }
}
```

Named recurrence strings include `hourly`, `daily`, `weekly`, and
`fortnightly` (also `every other week`); `every N minutes|hours|days|weeks|fortnights` and interval
objects are also accepted. A recurrence may wrap a recursive `base` trigger and
fires only when its schedule is due and that base matches. Without `at` or
`in_minutes`, its first occurrence is one interval after creation. Intervals
are fixed elapsed UTC time, with a fortnight equal to 20,160 minutes; they do
not infer a local time zone or daylight-saving policy.

For `time`, provide exactly one RFC 3339 `at` or positive `in_minutes` value.
For `context`, provide at least one of `codebase`, `file_pattern`, or `topic`;
all supplied fields must match. `event.condition` and `activity.activity`
case-insensitively substring-match `check.context.events`. A `compound` trigger
recursively combines `all_of` and `any_of`; each nonempty list retains its usual
all/any meaning, and both lists cannot be empty.

Trigger trees are limited to depth 5 and 32 nodes, with at most 16 entries in
each compound list. Duration, recurrence, and snooze intervals are 1 through
5,256,000 minutes. One-shot reminders fire at most five times and at least 30
minutes apart; a recurring schedule uses its cadence as its limiter and
persists its next future occurrence before returning. An expired snooze wakes
and evaluates during the same `check`.

Each `intention` request is limited to 131,072 encoded JSON bytes. Check context
accepts at most 32 topics and 32 events; its codebase, file, and every topic or
event item are limited to 4,096 bytes. A stored trigger that no longer parses or
validates remains pending and returns an `invalid_trigger` string instead of
silently firing or disappearing. In a mixed compound trigger, only a recurrence
branch that actually fired can bypass the one-shot 30-minute cooldown; a future
recurrence sibling does not change the event branch's limiter.

Canonical stored trigger fields use snake case. Accepted compatibility aliases
include `inMinutes`, `filePattern`, `allOf`, `anyOf`, `nextOccurrence`,
`everyMinutes`, and `intervalMinutes`. Check context also accepts
`currentTime`, `recent_events`, and `recentEvents` for `current_time` and
`events`.

One `check` evaluates and persists snooze, reminder-count, and recurrence
changes as a single local SQLite compare-and-swap batch. A concurrent conflict
rejects and rolls back the whole batch so the caller can retry. This protects
local state consistency; it does not establish exactly-once presentation or
delivery outside Vestige.

These scheduler triggers and the evidence-aware intention graph share the
public `intention` tool, but they remain distinct contracts in the first
implementation. A recurring reminder is not proof that a graph premise is true,
and a queued graph intervention is not a delivered recurring notification.

## Reproducible checks

The [synthetic MCP demo](../scripts/demo-intentions.py) drives the public
`intention` tool through an isolated subprocess and temporary SQLite database.
The [core intention graph tests](../crates/vestige-core/src/intention_graph.rs)
cover selective reevaluation, optimistic definition versions, evidence
reconciliation, absence coverage, attention state, caps, and replay. The
[public trigger tests](../crates/vestige-mcp/src/tools/intention_unified.rs)
cover recurrence, compound and activity triggers, snooze, and duplicate checks.
The [local intention claim tests](../crates/vestige-core/src/storage/intention_claim.rs)
cover two-connection contention and whole-batch rollback. These checks exercise
synthetic local behavior; they do not verify a product fact, external delivery,
or an external action.

## Boundaries

The first implementation provides deterministic local state transitions,
targeted reevaluation, bounded typed observations, coverage-aware absence,
completion reconciliation, explicit shared prerequisites and conflicts, local
queue/cooldown behavior, explanations, and atomic journal replay.

It does not provide autonomous connectors, real notification delivery,
semantic opportunity inference, model-generated premise extraction, causal
proof, purchasing, refunds, or other external actions. Those require separate
adapters, authority checks, and evidence. No worldwide-first, market-uniqueness,
performance, reliability, or outcome-improvement claim follows from this
implementation. Establish such claims with an independently reviewed frozen
evaluation and publish its failures as well as its successes.
