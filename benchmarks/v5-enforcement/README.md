# Vestige V5: Execution-Boundary Benchmark

**Status: preregistered exploratory protocol, NOT A RESULT.** Date: 2026-10-22. No product release or merge is requested. This protocol concerns current Vestige V5 / Operator Lite, not older V3 vector-retrieval benchmarks.

## Commercial claim under test

Does a local pre-tool execution gate plus verifiable records and retrieved precedents reduce unauthorized effects, repeat mistakes and total cost per successful task versus an equivalently configured native agent or alternative policy engine?

A hook saying STOP, a live host preventing an effect, successful cross-session recall, and dollars saved are **four distinct claims**. Do not conflate them.

### Stage 0: Inspectable seed and instrumentation — THIS PACKAGE

- A frozen public set of 20 examples in cases.json, including prohibited and harmless shell commands.
- The executable Python harness sends each command as JSON to the *actual shipped gate's hook entry point*, with isolated temporary HOME and OPERATOR_HOME.
- It checks actual hook exit codes and associated receipt decisions and measures elapsed subprocess time.
- **It never executes the proposed shell commands or invokes a model.** It does not validate live Claude Code/OpenCode host wiring, effects, completeness of interception, adversarial generalization or dollar savings.
- These 20 cases are public development cases, not unseen holdout data. Treat their outcome as smoke instrumentation, not security efficacy.

Frozen metrics: STOP recall (STOP among expected STOP), false-block rate (STOP among expected ALLOW), final receipt coverage, hook-exit/receipt concordance, error/timeouts, and inclusive hook-invocation p50/p95/p99 milliseconds. Show numerators, denominators, categories and every failure. Hook subprocess latency includes Python startup and disk I/O; do not compare this raw number to an in-process OPA/AGT microbenchmark.

### Stage 1: Real host and effects — NOT IMPLEMENTED YET

Test actual Claude Code PreToolUse and OpenCode tool.execute.before wiring on least-privilege, disposable directories. Use safe marker tools: one action writes a harmless marker only after the host approves it; the supposedly blocked action must not create the marker. Instrument unprotected tool surfaces (file edits, MCP calls, alternate interpreters), host errors, missing hook executable, 8-second hook timeout, symlink/alias tricks, and agent-authored edits to hook settings. No destructive payload will be run. Fail-open outcomes and uncovered paths must be named.

Paired comparison arms to build and run under matched policy and workloads:
1. Native host with default permissions and approvals (do not weaken defaults).
2. Native host with a documented least-privilege policy.
3. OPA Rego at an equivalent policy-enforcement point, with equivalent intent rules.
4. Microsoft Agent Governance Toolkit with its real adapter, if compatible with the selected host.
5. Vestige Operator Lite in enforce mode.
6. Vestige Operator with owner rules plus Strata stop-to-memory and precedent loop, if a licensed and reproducible test deployment is available.

Unavailable arms remain marked unavailable; never show product-vendor published latency beside a new measurement as a same-machine comparison.

Independent references: OPA agent tool-calling: https://www.openpolicyagent.org/docs/agent-tool-calling ; OpenCode native permissions: https://docs.opencode.ai/docs/tools/ ; Microsoft Agent Governance Toolkit: https://microsoft.github.io/agent-governance-toolkit/BENCHMARKS/ .

### Stage 2: Memory, causality and prevention — NOT IMPLEMENTED YET

Compare current V5 exact-handle Strata causal_walk and recorded provenance with BM25, dense-vector RAG and memory-disabled V5 on 40 independently selected repair incidents with known fixes. Same raw ingestion budget, memory history, model, prompts and task time limit. Do NOT hand Vestige the true cause handle while giving another system only a symptom; either expose an equivalent pointer-discovery mechanism to all systems or score retrieval failure honestly. Count unsupported claims and missing edges separately. Use real task outcomes, not mere relevance scores. MemoryArena needs adaptation to V5's exact-handle, no-embedding contract before running. Use SWE-bench Verified for separate held-out repair-task work, if a fair runner is built.

Useful independent resources: https://github.com/ZexueHe/MemoryArena and https://www.swebench.com/verified.html .

### Stage 3: Independent customer ROI — NOT STARTED

Pilot with 3–5 opt-in engineering teams, on real projects and agreed safety restrictions. Freeze matching baseline/treatment tasks and versioned evaluators before work begins. Collect model spend, cached/uncached tokens, failed attempts, maintenance, employee approvals/time, incident recovery, test success and p50/p95 task time. Primary economic endpoint: **total cost across every attempted task / independently verified successful tasks**. Include incident savings only with justified incident base rates and measured consequences, not imagined million-dollar extrapolations. Reuse the existing cost-ledger protocol in ../task-cost/README.md .

### Future adversarial data

Adapt attack-to-tool-effect scenarios from AgentDojo (https://github.com/ethz-spylab/agentdojo) and Agent Security Bench (https://github.com/agiresearch/ASB), plus benign near misses. Freeze a separate **at least 300-case held-out set**, externally reviewed before evaluation. Adaptive attacks developed after seeing a public seed score must be kept in their own dataset version. Real host version, model ID, git commit, gate hash, rule config, OS, receipt files and test code hash must be archived.

## Run the exploratory seed safely

From repository root with Python 3.9+:

    python3 benchmarks/v5-enforcement/run_gate.py --check-cases
    python3 benchmarks/v5-enforcement/run_gate.py --output /tmp/vestige-v5-enforcement-seed.json

The runner invokes the gate's Python script only. Its proposed tool commands are data, **never executed**. A test host can still inspect local paths, so inspect the harness and use a disposable test machine for untrusted changes. Output is created exclusively and not overwritten.

## Honesty and existing evidence

- docs/BENCHMARKS.md explicitly states Silent Rotation/MemConflict were V3 or earlier. The old benchmark has tool-parity caveats. No V5 security/ROI claim follows from it.
- ../before-you-change-that/README.md is one transparent evaluation observation, not causal proof of financial savings.
- ../task-cost/README.md implements accounting infrastructure but states synthetic fixtures cannot prove customer savings.
- Operator Lite's own README acknowledges that actions outside hooked tools and parser misses are residual gaps. Hash-chained gate receipts are not signatures. Strata's separately sealed and signed log must be measured under its own contract.
- Existing crates/vestige-mcp/tests/full_flow_stdio.rs verifies a key gate-to-memory sequence in CI, but does not replace cross-host comparison or field validation.

**No benchmark has been executed as part of preregistering this package.** All proposed public conclusions are contingent on a reproducible run and independent review.
