# THE VESTIGE TRIALS — V5 full-spectrum evaluation

**Status: under construction. No leaderboard or commercial claim is authorized.** Prepared October 22, 2026. This branch builds on the [narrow 20-case enforcement seed](../v5-enforcement/README.md), not a live security or cost-saving benchmark.

## PREVENT · REMEMBER · PROVE

An agent proposes an action; an independent execution gate enforces owner policy; the event is recorded as evidence; later agents retrieve the precedent and change behavior. All links in that chain must be measured, not assumed.

The program has **12 independent tracks**: (1) real-host unauthorized effects, (2) host coverage and portability, (3) exact single-use owner permits, (4) cross-session precedent, (5) Strata corruption/replay/signatures, (6) **retroactive hindsight versus recorded-edge evidence**, (7) FSRS memory life cycle/dreaming/suppression, (8) causal diagnosis against real bug fixes, (9) GhostLink proposal usefulness, (10) intentions/codebase continuity, (11) complete 18-tool contracts/token/latency, and (12) independent **customer dollars per successful task**.

Never combine twelve dimensions into a single flattering score. Show false blocks beside stops, wrong diagnoses beside right ones, and missed/abstained cases beside successes.

## Already runnable

The added run_mcp.py instrument starts the **real default V5 MCP binary** via JSON-RPC stdio in a disposable data directory, exercises actual memory writes and receipt replay, investigates a recorded-edge failure and a no-edge control, tests exact-handle lookup and a post-restart read. No paid model requests, no agent autonomy and no destructive tool commands are executed.

From repository root after building the V5 binary:

    cargo build -p vestige-mcp
    python3 benchmarks/v5-flagship/run_mcp.py --binary target/debug/vestige-mcp --output /tmp/vestige-v5-mcp-smoke.json

Reports are exclusive-create JSON with per-case assertions, failure details, response digests and elapsed MCP call time. The script is an **instrument**, not a commercial comparison.

## Study files

- [FEATURE_MATRIX.md](FEATURE_MATRIX.md) — exact behavior and caveats for all 18 V5 MCP tools, plus Operator Lite and paid Operator.
- [HINDSIGHT.md](HINDSIGHT.md) — retroactive salience experiment with true causes, decoys, false edges, missing edges and delayed events. It distinguishes the **legacy V3 backfill** from V5's current deterministic graph.
- [PROTOCOL.md](PROTOCOL.md) — enrollment, baselines, independent scoring, statistics, economics, attack safety, and claim thresholds.
- [preregistration.json](preregistration.json) — frozen planned comparison arms, cases, primary metrics and exclusions.
- [../v5-enforcement/](../v5-enforcement/) — initially measured public hook classifier and receipt replay, including its preserved confounded first run.

## Serious alternative systems and ground truths

| Question | Baselines / tasks |
| --- | --- |
| Tool-call enforcement | Claude Code native permissions, OpenCode native permissions, equivalently configured OPA policy-as-code, Microsoft Agent Governance Toolkit when a matching live adapter exists |
| Attacks | AgentDojo, Agent Security Bench and adaptive adversarial cases with **disposable effect oracles**, not imported attack-success claims |
| Useful memory | No memory, BM25, dense-vector RAG, long-context, Mem0, and V5; MemoryArena multi-session tasks after adapting V5's exact-handle interface fairly |
| Fixing code | SWE-bench Verified and current Terminal-Bench tasks; real repositories with withheld ground-truth fixes |
| Financial outcome | Native agent versus V5 under equivalent task models/permissions, verified task outcome and complete provider + human + maintenance costs |

Primary sources: https://www.openpolicyagent.org/docs/agent-tool-calling ; https://docs.opencode.ai/docs/tools/ ; https://github.com/microsoft/agent-governance-toolkit ; https://github.com/ethz-spylab/agentdojo ; https://github.com/agiresearch/ASB ; https://proceedings.mlr.press/v306/he26am.html ; https://www.swebench.com/verified.html ; https://github.com/harbor-framework/terminal-bench ; https://github.com/mem0ai/mem0 .

## Most important caveat

Vestige V5 default Strata follows **recorded** edges, not embeddings or similarity-based backfill. The original retroactive salience association code is **legacy V3**, not evidence that V5 autonomously discovers previously unrecorded causal links. The V5 benchmark tests this distinction head-on. It should **abstain** when the graph contains no evidence instead of inventing a cause. That is an accountability feature, but it is not automatic retrospective discovery.

No corporate dollar savings, universal bypass resistance, independent comparative victories or new brain-science findings have been established by this PR. The goal is a falsifiable study other engineers can rerun.
