# Vestige native benchmark campaigns

Local setup for Terminal-Bench, LongRCA, MemoryAgentBench, MemForm,
LongMemEval-S, and separately gated ReconcileSafe/Causal-Drift-Eval names.
This repository contains runnable preparation, scoring adapters, a native host,
and a credential-free Ollama factory. Setup and qualification results are not
model performance scores.

## Start here

Run from this directory using the isolated Harbor runtime:

```sh
export PYTHONPATH=.:upstream/harbor/src:upstream/agentdojo-library/src
export PYTHONHASHSEED=0
upstream/harbor/.venv/bin/python benchctl.py status
upstream/harbor/.venv/bin/python benchctl.py prepare-all --output evidence/my-new-freeze --limit 10
upstream/harbor/.venv/bin/python -m pytest -q tests
```

Preparation is zero inference. New freezes require new directories. The coordinator
preserves errors and timeouts without retries and detects source changes. Each
campaign manifest separately states source, data, scoring and execution readiness.
An unresolved source can be prepared as a gated entry; that is not an official
benchmark execution.

The prepared public inputs are deliberately separate from `oracle.jsonl`,
`answer-key.jsonl`, and `scorer-only/`. A scored solver sandbox must never receive
those sidecars or raw upstream datasets containing annotations. File mode 0600
does not isolate processes that run as the same user. The native memory-only host
exposes neither Bash nor unrestricted filesystem tools; its native paths remain
confined to disposable task-files.

## Campaigns and exact sources

- **Terminal-Bench 4.0.0**: official task/source/Harbor freeze, guarded complete
  Bash execution, original task verifiers, and separate local security fixtures.
  All 66 definitions are inventoried; 29 single-container CPU/resource candidates
  fit the static filter. Full 4.0 GPU/large-resource coverage is outside the local
  Colima profile. Image/oracle qualification is distinct from static compatibility.
  See [Terminal protocol](campaigns/terminal_bench/README.md),
  [official versions](https://www.tbench.ai/benchmarks), and
  [Harbor](https://github.com/harbor-framework/harbor).
- **LongRCA**: all 1,140 Full and 200 Mini records are downloaded and verified.
  Full preparation contains 177,884 source steps. Gold role, root step and rationale
  remain scorer-only. No causal edges are manufactured from gold or temporal
  adjacency. Exact root, role, within-five and source-weighted MAE scoring are
  separate from native walk diagnostics. Public metadata does not establish a
  redistribution license. See [adapter](campaigns/longrca/README.md),
  [paper](https://arxiv.org/abs/2608.15242), and
  [dataset](https://huggingface.co/datasets/CLoud5-real/longrca-bench).
- **MemoryAgentBench**: all four official splits; 146 contexts, 3,671 questions,
  28,737 lossless character chunks, exact source-specific metric adapters and
  question-bearing judge payloads. Character chunking is a disclosed native-track
  amendment, not the original token/sentence chunker.
  See [adapter](campaigns/memoryagentbench/README.md) and
  [official repository](https://github.com/HUST-AI-HYZ/MemoryAgentBench).
- **MemForm**: 994 validated MCQ and 897 inventoried free-form cases; all five
  formation interfaces, ordered native write adapter, paired scoring and separate
  runtime packets. The upstream free-form package imports a withheld builder;
  the isolated MCQ track uses its original validation/metrics and licensed parser.
  Receipts establish admitted bytes, not poisoning immunity.
  See [adapter](campaigns/memform/README.md) and
  [official repository](https://github.com/opanhw/memform-code).
- **LongMemEval-S**: full 500 questions and 23,867 source session occurrences,
  including all six question types and 30 abstentions. Exact dates, updates and
  duplicate source IDs are preserved, with evaluator-only alias groups. The
  ten-case pilot is deterministically stratified to include all six types and
  abstention. Judge-backed scores remain pending actual judge evidence.
  See [adapter](campaigns/longmemeval/README.md) and
  [official repository](https://github.com/xiaowu0162/LongMemEval).
- **ReconcileSafe / Causal-Drift-Eval**: authoritative benchmarks under these exact
  names were not established by bounded primary-source research. Their executable
  preparation retains `SOURCE_UNRESOLVED`, three clearly local reconciliation
  contracts each, no official score, and an unsupported native H1 gate. They are
  not renamed substitutes for an official benchmark.

## Native target and all-rule treatment

The selected executable is native MCP 6.0.1 (`e4a255db`), SHA-256
`3ecc40a08eedb2f74676fc688686afb8a0acd82607d6cb094046a799259e4840`.
It is distinct from the existing connected MCP owner reporting 6.0.0 and from
the preserved previous AgentDojo experiment. Native Operator is the installed
6.2.0 helper, matching reviewed source at `3de0c477`.

Each trial starts a fresh Strata store and Operator history. Production mode
preserves policy bytes. The separate `all_rules` profile promotes the 15 default
SHADOW IDs and configures all 118 declared IDs to enforce, with strict parsing,
test-integrity and turnover enabled. Validated owner laws and other settings are
preserved; no global policy, permit or installed binary is changed.

The static audit identifies 14 IDs without a distinct emitter found in the
reviewed implementation: OP-056–062, OP-073–075, OP-077–079 and OP-106.
Configuration does not implement missing dispatch. The rule matrix records
configured, mapped, exercised and unverified states separately. Overlapping rules
can still protect an effect. Do not claim universal safety or 118 exercised
mechanisms from a configuration count.

Command AST analysis is implemented in the native Python Operator helper. Rust
Strata handles structured log-effect admission. Complete tool-call mediation is
not OS-wide syscall interception or CPU Ring 0. An admitted command's internal
subprocesses are not independently interdicted by this host.

All 19 MCP schemas and all 11 callback types are qualified. Current source supports
exact-handle lookup, recorded-edge walks, Strata attestations, Reflex, eligible RSB,
FSRS-shaped reviews and native dream. No integrated Fourier HRR, sheaf H1/Laplacian
or general active-inference runtime was found in the selected implementation.
Those claims remain unsupported rather than being replaced with unrelated math.

## Free local execution

`local_solver:factory` uses the installed `qwen3.6:35b-a3b-q8_0` model, pinned to
digest `53003244b9b5f589be78818aaaa646fa242472f776f62ed0ccc224e6c29ee5e2`.
It contacts only fixed loopback Ollama, forwards the native schemas, preserves
tool identities/thinking, audits native `ollama_chat` admissions and raw responses,
and stops on malformed, oversized, truncated, replayed or uncertain requests.
The factory performs metadata checks; model inference begins only at execution.

Default bounds: one sequential tool call per response, 40 model requests,
65,536 context, 57,344 input bound, 4,096 output, and 4,096 template-overhead
allowance. The conservative bound includes schemas and replayed content. It
rejects excess rather than truncating. Zero provider fees excludes local hardware
and electricity; no speed or resident-memory guarantee comes from metadata.

Terminal's exact launch recipe is in its README. Select and qualify a pinned CPU
task before treating a run as an official score. Paid Terminal solvers are hard
gated because no fresh paid permit/ledger binding is implemented here. No API
credits are spent by preparation or the free local factory.

The unchanged `session-window-debug` task passed its original oracle and verifier
locally (reward 1.0, 7/7 tests). Its staged one-trial, zero-retry Qwen job preserves
the original resources and timeouts. Preview and validate without inference:

```sh
sh campaigns/terminal_bench/run-tonight.sh
```

Only adding `--execute` starts actual local inference. That execution has not
been performed during setup, and no overnight automation is scheduled.

`memory_run.py` supplies a separate native memory-only local loop for the four
verified memory/causal families. Planning is zero inference; `--execute` is an
explicit execution boundary. It produces predictions and raw traces; run the
trusted campaign scorers separately. A single native run cannot establish a
comparative win without its matched controls.

## Evidence and rollback

`evidence/native-audit/` contains the exact rule matrix, static gaps, current
AgentDojo r3 observations and source identities. `evidence/final-review/` holds the
independent review. Native and Docker seam proofs live under the Terminal package;
the final common preparation is under `evidence/release-final-20261031/`.
`evidence/memory-native-20261031-r1/` preserves the earlier four native fixtures;
`evidence/memory-native-final-20261031/` is the refresh against the isolated host.
Their scripted reads are not model performance. Valid no-op RSB pulses are retained.
Strata write attestations and Reflex reference-digest integrity are distinct.

Previously completed AgentDojo r3 remains unchanged: 24 completed trajectories,
Vestige clean interventions 0/7 with utility 3/4; baseline utility 3/4; both attacked
ASR 0/4. These finite observations are not a universal false-stop guarantee or a
unique attack-containment result. Its supplied-rate accounting was $0.5231105;
the new setup did not spend provider credits.

Keep prior failed evidence. Stop only the new adapter process/container for
rollback; the original experiments, global policy, installed binaries and old
records are preserved. This local repository has not been publicly published.
