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

## Stage 1 implementation boundary

`benchmarks/task-cost` implements an offline export recorder/accountant with
OpenAI Responses and Anthropic Messages text-usage shapes, exact decimal rate
arithmetic, frozen artifacts, explicit missing data, and synthetic tests. It
does not yet capture a live agent session, reconcile a provider invoice,
independently score a developer task, or demonstrate savings.

The comparison contract starts with the native agent, current released Vestige,
and candidate Vestige. Use equivalent model, effort, initial repository,
historical information, stimulus, task limits and evaluators. Verify native
caching/context management before comparing. Isolate arms and report cold/warm
behavior separately. Initial suite families: fresh task, interrupted work,
correction, historical debugging, cross-project similar facts, and accumulated
memory. Development and held-out cases remain separate; choose confirmatory
sample sizes from observed variance before looking at held-out results.

Primary metric: total cost of all attempted tasks divided by independently
verified successes, with success rate, total spend and latency alongside it.
Zero successful tasks have undefined cost per success. No flat savings claim
is justified by a single favorable run, synthetic fixture or retrieval score.

## Candidate implementation progress

- Stage 1: offline cost ledger and caller-owned SDK capture seam; fake SDK tests
  cover usage, retries, failure and serialization errors. Live capture/provider
  reconciliation and the scored baseline remain pending.
- Stages 2–3: lookup and reason whole-evidence budgets; schema-derived discovery;
  opt-in stable lookup packets and explicit client acknowledgment of retained
  context. Provider cache behavior and task-level cost effects remain unmeasured.
- Existing tool-contract candidate: deterministic intentions, scoped Backfill
  previews and graph hypotheses, namespace/policy checks during merge apply,
  truthful embedding edit state with peer index invalidation, effective
  maintenance controls, and explicit ingest batch outcomes. These implement
  portions of stages 4–7; they do not complete every lifecycle or maintenance gate.

- Stage 4: merge/supersede previews bind mutation-relevant source state; apply
  rejects stale or legacy plans and inactive members. Merge undo compares the
  applied state and restores content, validity, journal and plan status in one
  transaction. Later edits cause a conflict instead of being overwritten.
  Legacy operations without post-state snapshots require manual recovery review.
  Exact suppression reversal and bounded atomic restore remain pending.
