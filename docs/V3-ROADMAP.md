# Vestige v3: verified developer-task efficiency

Objective: lower total cost per independently verified successful developer
task, including memory operation, while preserving task success. Proposed
engineering targets are 30% reduction, with 50%+ a stretch for repeated-context
workflows. These are goals, not measured results or release promises.

Work proceeds solo and sequentially. Tool-contract and lifecycle changes are committed in the candidate branch;
they do not establish a released v3 implementation.

| Order | Work | Exit gate |
|---|---|---|
| 1 | Cost accounting: receipts, identities, provider usage, overhead, frozen baseline contract | Reproducible complete accounting and a qualified live baseline |
| 2 | Compact, sufficient recall with expandable evidence | Lower held-out task cost within a predeclared quality bound |
| 3 | Progressive discovery and cache-aware client integration | Actual model-facing requests demonstrate reduced recurring cost |
| 4 | Current memory state, ingestion, dedup and suppression | Corrected/stale/conflicting records behave consistently; measure rework |
| 5 | Intentions joined to session start, source sync and codebase state | Less reconstruction across sessions without stale resumption |
| 6 | Bounded graph and failure investigation | Fewer failed attempts; ablation establishes added value |
| 7 | Incremental, observable maintenance | Setup, steady-state, maintenance and break-even costs accounted for |
| 8 | Repeated frozen comparisons | Preserve failures, uncertainty, ties and regressions |
| 9 | Company evaluation package | Reproduce results on company-controlled workloads |

## Implemented candidate surfaces

1. **Accounting and receipts:** frozen identities and artifact hashes, exact
   decimal provider-usage accounting, caller-owned SDK capture, failed-request
   retention, task wall-clock spans, and request-level billing-export reconciliation.
   Missing usage remains unknown; empty exports cannot establish reconciliation.
2. **Sufficient recall:** whole-evidence budgets, dissent-group preservation,
   expandable memory IDs and stable opt-in context packets with explicit retention
   acknowledgment and refresh after context loss.
3. **Client integration:** schema-derived progressive discovery and an installable
   Python runtime owning transcript retention, MCP stdio and provider request
   serialization. The caller owns SDK credentials and execution.
4. **Memory lifecycle:** explicit batch outcomes, atomic vector invalidation and
   stale-computation rejection, namespace/policy/stale-plan merge checks, atomic
   conflict-aware merge undo, schema 34/35 suppression and neighbor journals,
   and bounded staged restore.
5. **Continuity:** shared source evidence across session start/source sync/codebase,
   versioned whitespace-preserving source anchors, deterministic intention clocks,
   validated triggers and explicit observed-event matching.
6. **Investigation:** scoped reasoning, bounded graph hypotheses and Backfill
   previews without automatic reinforcement. Local composition novelty remains
   distinct from worldwide novelty or causal proof.
7. **Maintenance:** bounded embedding, lifecycle, log and GC batches, cooperative
   lifecycle/GC budgets, scoped dream pages and pair limits, durable checkpoints
   and explicit continuation. Default full consolidation remains compatibility
   behavior; cross-page dream pairs and hard inference deadlines are not promised.
8. **Evaluation:** six executable development task families, frozen source/prompt/
   history/evaluators, deterministic trial order, isolated checkouts, caller-trusted
   driver execution with timeouts, external scoring and exploratory paired reports.
   Reference-solution qualification executes no models.
9. **Company package:** hashed local summaries with task timing, reconciliation,
   declared sample-policy screening and explicit break-even scenario arithmetic.
   The original company-controlled bundle remains necessary for reproduction.

## Empirical and release gates

The code can be qualified with local deterministic tests. Real savings remain
unmeasured until company-controlled model trials run with frozen model/effort,
source/history, tool configurations, isolated memory stores and equal task limits.
No model agents or paid requests are launched by implementing this roadmap.

Compare the native agent, current released Vestige and candidate Vestige. Verify
native caching/context management and preserve cold/warm, setup/steady-state,
maintenance, failed attempts and retries. Provider exports and supplied prices
need independent qualification; reconciliation checks supplied request charges,
not invoice authenticity, taxes or discounts.

Development fixtures are not held-out company workloads. Freeze quality bounds,
cost targets and sampling before evaluation. The implemented cluster intervals
and sample-policy gate are exploratory screening, not confirmatory significance
or authentication of attached qualification statements. Independent task and
provider qualification, confirmatory study design and held-out model trials
remain empirical gates.

Primary metric: total cost of all attempted tasks divided by independently
verified successes, with success rate, total spend and complete task latency.
Zero successful tasks have undefined cost per success. Synthetic fixtures,
reference solutions and smaller request byte counts cannot prove billed savings.

The candidate is local. Installation into a live memory store, schema migration,
provider execution, publication and release require their own target verification
and authorization. Retain the pre-upgrade database backup for schema rollback;
unrecorded historical suppression effects cannot be reconstructed.
