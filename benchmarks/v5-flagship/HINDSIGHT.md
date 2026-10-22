# The hindsight challenge: Memory that becomes important AFTER the fact

**Experimental hypothesis, not a shipping feature claim.** V5 default Strata performs deterministic backward walks over previously recorded edges. The historical retroactive salience algorithm using entity overlap belongs to V3, not the V5 default runtime. This distinction is vital when describing the scientific breakthrough to customers.

## The neuroscience, precisely

Zaki, Pennington, Cai and colleagues, Nature (published online November 2024; volume 637, January 2025) studied **mice**. A later strongly aversive event caused co-reactivation of ensembles encoding a neutral experience from two days earlier; association was retrospective more than prospective. These animal findings motivate *an analogy* for computing, not a direct implementation of hippocampal cell activity or a validated guarantee of root-cause inference.

Primary paper: https://www.nature.com/articles/s41586-024-08168-4 . V5 design note: docs/SCIENCE.md . Legacy algorithm: crates/vestige-core/src/advanced/retroactive_backfill.rs and its current dispatch refusal in crates/vestige-mcp/src/tools/backfill.rs.

## The killer demonstration: The quiet configuration change

Day 1: a small, operationally boring patch changes a queue timeout and is recorded in Git history. Nobody reports a bug. The memory has weak retrieval priority.

Day 20: an agent investigates intermittent checkout failures. A correct root-cause investigation should identify the earlier change **when supported by the git/test evidence**, not simply retrieve other crashes with similar words. The agent produces a patch; a frozen evaluator replays it and verifies the incident is resolved with no regression.

Build at least 80 blinded incidents from real versioned projects and independent adjudications. Each incident contains many distractors including earlier semantically similar but irrelevant text, misleading common file names, related commits that never touched the failing component, future-only fixes, missing logs, and a contradictory incident narrative.

**Do not write the ground truth culprit into Vestige's tags or pass its memory ID to causal_walk.** Equal histories and discovery access must be given to all systems; if V5 cannot discover its starting id from the symptom and evidence budget, that counts as a real limitation.

## Factorial design

| Cell | Observed evidence / intervention | What success means | What falsifies an overclaim |
| --- | --- | --- | --- |
| A — recorded chain | Earlier commit and later symptom linked by predeclared provenance rules, not hidden oracle | V5 traces recorded path, identifies suspect, agent tests correct patch | Retrieved path is wrong or task fails |
| B — no chain | Same texts/history but link evidence withheld equally from all arms | V5 **abstains with emptyBecause or needs_report**; others may retrieve candidates | Any V5 claim that it autonomously discovered the unseen cause |
| C — false chain | Plausible wrong provenance edge plus independent conflicting tests | Agent challenges path and does not present graph traversal as proof | Falsely asserts root cause solely because an edge exists |
| D — delayed activation | Neutral note, later salient failure, additional intervening records | Later investigation uses earlier evidence and improves verified repair outcomes | No action improvement despite strong recall |
| E — temporal decoy | A post-failure change sharing issue identifiers | Does not mistake future *fix* for earlier *cause* when verification/clock evidence shows otherwise | Incorrect reverse-causality claim |
| F — stale correction | Earlier decision later explicitly superseded, newer fix recorded | Current agent follows corrected state and avoids repeating superseded approach | Retains wrong stale advice |

## Baseline arms and measurable outcomes

Minimum comparative arms: no memory; full-context; BM25; dense vector RAG; current V5 with exact handles and recorded graph; V5 with memory access disabled; optional Mem0. A legacy V3 backfill arm can be evaluated separately *as historical reference*, identified by its actual V3 release and binary SHA. It must not be labeled 'V5 retroactive salience'.

Measure root-cause top-1/top-5 (hypotheses not proof), verified patch success, median time and agent tool turns, earlier quiet-record rank, false-cause assertion, abstention when evidence absent, repeat-bug frequency across sessions, recall bytes/tokens, model-provider USD, and human review minutes.

Repeat with incident backgrounds shuffled, frozen prompt/model versions, randomized paired order and independent fix evaluators. Show per-repository errors and confidence intervals. V5 may correctly refuse to speculate; that makes it safer for high-stakes forensics but may lose recall tasks to systems permitted to hypothesize without a proof edge. **Report both dimensions.**

## What a genuinely new retroactive module would have to demonstrate

Only if separately implemented in a future, explicitly identified experimental branch:

1. Candidate eligibility based on independently observed *post-event novelty/salience* and earlier records, not on a gold label or hidden cause ID.
2. Temporal ordering and information provenance enforced; never treat the future event as the historical cause.
3. A recorded, reversible candidate promotion, with separate proof that a candidate is plausible versus verified. Avoid automatic strengthening from mere repeated reading or a failed hypothesis.
4. False associative-memory rate measured against deliberately similar-but-wrong evidence, and against existing vector and graph baselines.
5. **Action value:** fewer real debugging mistakes and lower cost per successful repair — not just higher similarity rank.

Until those are measured, the defensible V5 phrase is **evidence-grounded retrospective investigation and memory continuity** rather than 'a brain that automatically finds forgotten causal links'.
