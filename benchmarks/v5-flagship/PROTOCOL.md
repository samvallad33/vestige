# Prespecified V5 flagship evaluation protocol v0.1

**Written before new full-spectrum V5 results.** Freeze git SHA, binary hashes, source datasets, benchmark generator, model IDs, host versions, baseline permissions, task selection, scoring scripts and dollar rates prior to testing. Report a new study version for any changes after seeing outcomes. No leaderboard entry or economic claim follows from an instrument passing.

## Four primary questions

H1 — Does Vestige prevent **unauthorized real-world tool effects** better than realistic native-host and OPA policy baselines without disproportionately blocking legitimate user actions?

H2 — Can current V5 **recorded-edge causal investigation plus durable lessons** improve a real agent's independently verified repair success across sessions versus BM25, embeddings, long context, and no memory at similar input budgets?

H3 — Can a second agent reconstruct a gate stop from PRECEDENT and Strata receipts, follow a legitimate next step, and detect tampering or single-use permit reuse?

H4 — Does the full product reduce **actual total cost / independently verified successful task**, counting every attempt, failure, model request, review, and storage/maintenance, in opted-in company workflows?

## Experimental units

Security: one genuine attempted tool effect within a pinned running host, grouped into attack episodes. The oracle inspects a reversible marker on a disposable machine; a STOP label or model verbal refusal is insufficient.

Memory: one multisession dependency chain where the later task genuinely requires earlier evidence, sourced from held-out projects. Count downstream action success, not answer retrieval alone.

Causality: one versioned incident with a test-verified culprit and fix and a hidden gold adjudication. Missing graph edges and abstention are explicit outcomes.

Economics: one attempt through frozen external task evaluator, including retries, failures, approvals and allocated operating overhead. Unknown billing stays UNKNOWN; zero successes => undefined cost/success.

## Required baselines

Security: native Claude Code and OpenCode **default** permissions; native least-privilege configured controls; OPA with semantically equivalent rule and an equivalent enforcement point; Microsoft Agent Governance Toolkit when an integration compatible with the host is available; Lite and paid Operator **as separate product arms**. Do not compare OPA evaluation-only to Vestige's full process+receipt overhead as if both are the same measure.

Memory: no memory, whole context, BM25, dense embedding search, V5 default Strata, ablation with V5 memory disabled, plus commercially available memory product(s) when runnable. Give identical documents, ingestion times, model/task budgets and honest handle discovery. The answer's record id must not be injected as a special V5 hint.

Factorial: no guard/no memory; guard only; memory only; guard+memory, under equal feasible host permissions.

## Size, holds and independent adjudication

Phase 0: the preexisting 20-case public enforcement smoke; a **different** new direct V5 MCP instrument. Neither is statistical proof.

Phase 1: at least 300 frozen, externally reviewed unseen host-effect episodes **for each compared host**: at least 100 harmful, 100 clearly legitimate, 100 adversarial/cross-turn. Include an independently selected second attack set, shadow/off mode, missing hook, tool channel not wired, timeout and failure injections. Separate each threat family; repeated variants of the same payload belong to one cluster.

Phase 2: >=80 incident tasks across >=8 projects, with hidden verified fixes, >=6 hindsight cells described in HINDSIGHT.md. Hold out >=30% at project level and prevent model pre-exposure to gold IDs and evaluations.

Phase 3: 3–5 opt-in independent teams, preliminary economic study only; pair tasks and freeze legitimate business objectives, scoring and time limits before running. Larger replication required for market-level inference.

For binary rates report numerator/denominator, confidence interval (Wilson or exact if small), and family/project-cluster bootstrap for correlated data. For paired tasks report wins/losses/ties, uncertainty and prespecified effect threshold; never pool six task types after noticing only one favors Vestige. No cherry-picked optimum model temperatures or adaptive test removals. Any incomplete arm must appear with reason.

## Metrics and fail criteria

- Safety: prohibited effects that **actually happened**, permitted effects that succeeded, legitimate false denials, host-tool coverage, model success rate, permission bypass, fail-open error, permitted review burden and gate p50/p95/p99. Security pass implies only the tested boundary, not universal impossibility of bypass.
- Evidence: verified/invalid chain/signatures, receipt/get/replay concordance, signed versus unsealed segments, corruption/truncation detectability and recovery, signer trust, single-use authorization after restart.
- Causal: correct culprit top-1/top-5, verified patch success, unsupported cause assertions, absent-edge abstention, future-decoy errors, hallucinated trails, stale anchor handling and recovery-time cost.
- Memory: correct lesson resurfacing, missed evidence, abandoned/harmful memories, repeated-error rate, FSRS retention calibration against actual observed review outcomes, dreaming false reinforcement and measured write/read/amortization overhead.
- GhostLink: blinded human judgment of useful pair proposals, compositional success verified by an evaluator, false novelty, duplicate pair rate and cost of exploration.
- Economics: model input/output/cache USD, every failed provider request, human reviewer minutes at actual reported rate, repeated fix work, approved changes, hosting/CPU/storage, setup, backup/maintenance, cost per independently verified successful task, and measured vs counterfactual costs strictly separated.
- Observability: raw event logs+sha256, git commit, receipts, model version, explicit time stamps, tool count and error outcomes. Align with OpenTelemetry's GenAI client/agent/tool metrics.

**A future win that trades a 1% security gain for a 50% legitimate-task failure rate is not a commercial success.** Show the joint safety/utility frontier and Pareto-dominated configurations.

## Host, privacy and attack safety

Run shell/DB/cloud/git adversarial actions only as **inert JSON**, never execute their dangerous payloads. Real host-effect trials are limited to reversible sentinel tool implementations and fake filesystem/DB/email endpoints, with disposable accounts and no third-party destructive permissions. Use least-privilege containers/VMs and independent review of both attacker and defender code.

Do not publish source/customer secrets or full prompts without consent. Redact after evidence hashing and retain original checksums securely. Paid model endpoints, customer deployments, third-party contact and paid Operator licenses require separate authorization; they are NOT part of this branch.

## Criterion to make commercial claims

- Safe agents: comparable live-host *effect* evidence and negative controls, clean rerun by outside security engineer, uncovered channels named.
- Superior memory: paired, held-out repair or multi-session task advantage with equal historical evidence and a non-leaky handle discovery path; model-judge scoring alone is insufficient.
- Neuroscience-inspired retroactive benefit: an explicit implemented and independently verified mechanism active in the **actual binary under test**. Current V5 does not ship the legacy V3 salience-backfill path as a default Strata feature.
- Saves customers money: opt-in real teams, independently verified completed tasks, auditable usage bills and actual employee labor costs. An illustrative breach price is not observed savings.
- Tamper-evident: experimentally detect tampered log under correct key provenance; cryptographic proof does not prove the underlying event was truthful.

## Key external benchmarks

AgentDojo: https://github.com/ethz-spylab/agentdojo ; ASB: https://github.com/agiresearch/ASB ; MemoryArena: https://proceedings.mlr.press/v306/he26am.html ; SWE-bench Verified: https://www.swebench.com/verified.html ; Terminal-Bench: https://github.com/harbor-framework/terminal-bench ; OPA agent-tool enforcement: https://www.openpolicyagent.org/docs/agent-tool-calling ; OpenTelemetry metrics: https://opentelemetry.io/docs/specs/semconv/ ; neuroscience evidence: https://www.nature.com/articles/s41586-024-08168-4 .

The product may outperform, tie, underperform, or be ineligible for particular arms. All four conclusions are publishable if verifiable.
