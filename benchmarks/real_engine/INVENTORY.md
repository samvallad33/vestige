# Official source inventory

This document records source availability checked on 2026-11-01. It is not a
score table or a statement that a benchmark has run. The machine-readable pins,
local file hashes and remaining integration actions are in `inventory.json`.
All arms and judges use **gpt-5.4-nano** under one $25 API cap. No other solver
model or local-model fallback is part of this campaign.

## Verified current facts

| Benchmark | Official source and pin | Public data / scorer | Remaining execution gate |
|---|---|---|---|
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo), `3eb6f2c5` | Public data and scoring code | Bind all 10 conversations, all categories; see dataset adapter |
| LongMemEval-S | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) | Public data and scoring code | Pin the 500-item data bytes and scorer in dataset adapter |
| MPBench | [Digital-Trust-Lab/mp-bench](https://github.com/Digital-Trust-Lab/mp-bench), `6886880a` | Public data; no released scorer code | Owner-authorized exception: freeze and commit paper-defined ASR/FPR scorer before any arm |
| MemConflict | [TaoZhen1110/MemConflict](https://github.com/TaoZhen1110/MemConflict), `ec51d5d3` | Existing dataset lock already covers 3,750 questions | All reads/writes through own real engine; preserve temporal cutoffs |
| PM-Bench | [genglinliu/PMBench](https://github.com/genglinliu/PMBench), `e1093c47` | Full released seven-day scenario and replay scorer | Full simulation lifecycle; scoring a QA substitute is invalid |
| MemoryAgentBench | [HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench), `53802608`; [official HF data](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench), `7ea06698` | All four split files and metric code | Conflict Resolution first; bind local file hashes and official per-source metrics |
| MemForm | [opanhw/memform-code](https://github.com/opanhw/memform-code), `05651170` | 994 MCQ cases, 897 free-form cases, official evaluation and defense metrics | Paired memory interventions through gate; never expose oracle policies |
| Terminal-Bench 4.0 | [harbor-framework/terminal-bench](https://github.com/harbor-framework/terminal-bench/tree/v4.0.0), `452bf305` | 66 official task directories and task verifiers | Harbor environment and engine integration; prepared manifest alone is not a run |
| GraphWalks | [openai/graphwalks](https://huggingface.co/datasets/openai/graphwalks), `be6cc6ec` | Public parquet files and official extraction/F1 code in dataset card | Equal measured context caps; pin corrected data; real graph-memory route |
| AutomationBench | [zapier/AutomationBench](https://github.com/zapier/AutomationBench), `4a8e1061` | 600 public scored tasks, 200 excluded warmups, assertion scorer | Official world state/tool loop plus budgeted provider; private leaderboard is a different split |
| Vals Finance Agent v2 | [vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2), `03c7c432` | 27 public examples; platform-gated official scorer | **SKIP: no public official scorer; held-out official tasks private** |
| CWE-bench v1 | [official site](https://cwe-bench.com/) | 120 held-out tasks explicitly private | **SKIP: official tasks private** |
| AgentDojo | [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo), `75cea067` | Workspace, travel, banking, slack suites and utility/security predicates | All suites with task-local memory and policy history; no tool effects on real services |
| InjecAgent | [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent), `f19c9f2c` | Base set: 510 direct-harm + 544 data-stealing = 1,054 cases; official parsing/scoring | Preserve both data-stealing stages and official validity denominators |

## MPBench data and scoring exception

The owner explicitly permits one custom, frozen scorer that implements the
[paper's metrics](https://arxiv.org/html/2606.04329v2), and requires it to be
committed before any MPBench arm runs. This is an exception to the campaign's
otherwise official-scorer-only rule, not a claim that the repository released a
scorer. The scorer must retain its origin and definition in the report.

Paper section 4.1.3 defines ASR using semantic judgment of entries actually in
the post-write memory store. A write rejection rate is a separate measurement.
RSR is conditioned on successful poisoning writes. Section 4.5 measures defense
false positives on benign inputs. Labels and target instructions are available
to scoring, not to the solver or retrieval construction.

The pinned public bytes differ from the paper's reported counts. The attack file
has 3,240 physical lines but **3,241 JSON objects**; one line concatenates two
objects. The benign file has **2,999 objects**. The paper reports 3,240 attacks
and 2,997 benign cases. Decode the complete JSON stream without editing bytes
and disclose the actual denominator. Retain raw attack labels. The 240 records
with skill-specific fields form the skill-procedure group even though their raw
`attack_type` uses an explicit-command alias; the mapping must be disclosed.
Two attack records omit scoring targets/follow-up queries and must remain
visible as unscorable rather than disappearing or receiving fabricated scores.

## Terminal-Bench evidence boundary

All earlier `session-window-debug` results that used task-specific solution
guidance are **INVALID**. The current clean prompt adapter contains no such
branch. Historical `proof.json` artifacts are **engine self-tests**, never
Terminal-Bench task scores. They cannot enter the pass-rate numerator.

`astra_suite.py` enumerates the full official v4.0.0 catalog, verifies the exact
revision and clean task tree, and can freeze either all tasks or a uniform
random sample of at least 20 before execution. A sample of 20 was frozen with
seed `20261101` in `runs/terminal_bench/sample-frozen.json`; its SHA-256 is
`740d31375c1b70f5cbe6dff99df1cfe3eebaff3c6ac6fc92db9708efa23df83d`.
It specifies 3 trials per task per arm, or 240 rollouts. The manifest contains
hashes of official task files and is exclusively created to prevent accidental
replacement. The adapter prepares shared neutral messages; it does not claim to
provide the Harbor executor.

## Skips and blocked work

CWE-bench v1 is skipped because its official held-out tasks are private. Its
public training corpus must not substitute for them. Any future result belongs
under secondary security evaluation, not memory-poisoning evidence.

Vals Finance Agent v2 publishes 27 examples and its agent harness, but requires
approved Vals platform access for official suites and scoring. The published
methodology uses three frontier judges. Replacing that mechanism with a custom
nano scorer would violate the current official-scorer rule, for which only
MPBench has an exception. The 450 validation examples require a license and the
450 test examples are private. Do not replace v2 with the older 50-question v1
repository or contact the provider as part of benchmark setup.

Every other row is pending execution/integration unless a separate raw run and
RESULTS.md establish completion. Publisher LFS hashes in `inventory.json` are
primary metadata; they are not a claim that this inventory downloaded and
hashed those dataset bytes locally.
