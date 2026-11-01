# Four-benchmark real-engine campaign

The current owner-authorized campaign contains only these official sets:

| Benchmark | Items per arm |
|---|---:|
| MemConflict | 3,750 |
| LoCoMo | 1,986 questions from all ten conversations and all categories |
| LongMemEval-S | 500 |
| MPBench | 3,241 attacks plus 2,999 benign records |

The four arms are bare, BM25, pinned local BGE-small retrieval, and one Vestige
arm with the applicable **memory-QA hooks**. All solver and judge requests use
`gpt-5.4-nano`. The shell entry point and Python argument guard cannot launch
other benchmarks. Older adapters and evidence remain preserved outside this
campaign's execution scope.

Source inventories and native-engine self-tests are preparation evidence.
A scored result requires full coverage, current-configuration integrity checks,
the pinned scorer, and its completed scoring receipt. Check `runs/STATUS.md`
for current progress; a pending cell does not mean zero performance.

## Native engine and hooks

Every independent memory lifetime starts the repository-built
`target/release/vestige-mcp` child with its own data directory. All Vestige memory
operations use stdio JSON-RPC through
`benchmarks/memconflict/mcp_client.VestigeMCP`. The release build is locked and its
SHA-256 is recorded in the build manifest and per-item records. No Python memory
implementation is used on the Vestige path; the BM25 and BGE baselines have their
own private corpora.

`runs/HOOKS.md` inventories every script and settings fragment. The memory-QA
profile selects the unchanged native SessionStart registration and pulse memory
injection registration from those fragments. Tool-action veto, Operator stop,
and preflight hooks are not applicable: no external action tool calls occur in
these loops. MPBench's normal memory writes still pass through Strata's real
write gate. An unavailable legacy pulse endpoint produces the original hook's
natural empty output; there is no replacement retrieval path.

Every MPBench memory-arm item submits the exact official gold-free raw input
envelope through its normal initial write path before model inference. BM25/BGE
store the same raw bytes. The auditor checks the native request/reply and actual
write-operation log; coverage reads without a write cannot qualify. Bare has no
persistent store. The earlier profile whose pilot never wrote is explicitly
invalid and preserved in `runs/profile-history.jsonl` and `runs/INTEGRITY.md`.

Hook HOME, state, working directory, engine and data paths are isolated. No
machine-wide memory or Operator history is loaded. Raw historical dialogue is
ingested through the engine during the current item's run. Gold targets and
annotations remain separate from every solver input. System instructions differ
only by one neutral tool-description line; raw retrieval results are arm-specific.

## Execution and reproduction

Prerequisites are the recorded Python environment, pinned official sources/data
under `runs/upstream`, source hook audit, and owner-only `runs/.env.openai`.
Credentials are ignored by Git and never printed. Preserve the existing
`runs/budget.db`; do not reset it to obtain another spending allowance.

```sh
bash benchmarks/real_engine/reproduce.sh
# Or one of the same four sets, using the same ledger and campaign lock:
bash benchmarks/real_engine/reproduce.sh --benchmarks memconflict,locomo
```

The entry point rebuilds the locked release binary, records its hash, and starts
one controller. Each benchmark uses ten deterministic shards per arm. Sharding
is SHA-256(item_id) modulo ten; the fixed pilot seed is 20261101. A single global
semaphore limits requests to twenty, including judge calls. HTTP timeouts are
sixty seconds, with up to six retries and exponential backoff with jitter for
429, 5xx, timeout and transport failures.

Each benchmark first collects ten unscored official items per arm, runs its real
mid-run engine-kill control, and is independently audited. A passing benchmark
immediately starts its full official set while other pilots continue. Equal
aggregate call/output/context caps include hook model calls. Pilot evidence is
bound to the exact source/configuration and checkpoint identity.

The three required checker regressions cover fabricated PID/hash plus
request-only evidence, budget parity without actual settled usage, and obsolete
pilot evidence. Native logs require matching engine startup, initially empty
store, MCP requests and responses. MPBench's verified native security refusals
are recorded as gate decisions, rather than mistaken for engine death.

Each finished item is appended and flushed to its worker checkpoint. Failed
native tool requests return their exact MCP error envelopes to the model within
the common budget; transport failures still abort. Output, iteration and context
caps end an episode with its actual partial response or store. The unchanged
scorer assesses that real outcome, including losses; no replacement answer or
metric is supplied. Episode termination reasons are recorded.

Each finished item is appended and flushed to its worker checkpoint. Failed
attempts are preserved separately and retry at the end. Restarts reuse only
matching successful checkpoint identities. Signal handlers clean up owned child
engines. A native death never enables a fallback memory implementation.

Every API attempt reserves a conservative bound in the shared $25 SQLite ledger
before sending. Actual response usage settles it. Ambiguous failed attempts keep
their reservations; they are not asserted to be confirmed charges. The ledger
includes superseded preparation. Forecasts are printed from one calibration
item per arm; forecasts never reduce the full official manifests.

The status writer updates `runs/STATUS.md` every five minutes. Exact completed
and missing IDs are recorded in each benchmark's `coverage.json`. Official
scoring starts only after all four arms have a complete duplicate-free merged
file. Scorer source pins and one completed invocation receipt bind each scored
output to that merge. Reports show means, bootstrap CIs, n, costs and hashes;
missing or invalid results never populate a score cell.

MPBench uses **scored with our implementation of the paper's metrics, no
official scorer available**. The scorer is frozen in commit `c126b885`; its
bytes must remain unchanged. Missing public targets and unobserved defense
decisions are explicitly unscorable, with coverage/bounds disclosed. No
synthetic labels are inserted.

The separately launched `benchmarks/four_report_watch.py` makes no inference
calls. After the existing controller exits, it re-audits these four benchmarks
and writes the qualified-only summary, listing every unfinished/invalid set.
No code or result is pushed automatically.
