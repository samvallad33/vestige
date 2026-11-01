# LongMemEval-S real-engine campaign

The current owner-authorized run is **LongMemEval-S, all 500 official items**.
LoCoMo, MemConflict, MPBench and every other benchmark remain stopped. Their
existing evidence and all charges remain preserved.

The four arms are bare, BM25, local BGE-small retrieval, and Vestige with the
all active hooks registered by the original settings fragments. Every solver and judge request uses gpt-5.4-nano.
The shared `runs/budget.db` preserves all earlier spend and reservations. A new
owner-authorized epoch permits at most $21 of new LongMemEval-only spending from
2026-11-01T16:12:10Z, including every hook, solver and judge request.

## Full native MCP contract

Each independent Vestige item starts the repository-built, hashed
`target/release/vestige-mcp` child with a fresh isolated store. Memory operations
use stdio JSON-RPC through `benchmarks/memconflict/mcp_client.VestigeMCP`.

The model receives every tool returned by that child's `tools/list`, preserving
all input schemas and native descriptions. The exact native `initialize`
instructions are carried once in `memory_status.description`; system/user
prompts retain their shared text and one neutral tools line. Startup logs bind
the raw catalog, initialize response, model adaptation, binary and source hashes.
There is no hand-selected tool subset, Python memory, rewritten lookup, answer
hint or fallback. Exact-ID/tag lookup semantics remain native.

Child HOME, XDG directories, temporary files, Operator state and actual working
directory are item-local. Ambient credentials and loader overrides are removed.
The full catalog stays visible while external collection and filesystem effects
outside the worker are refused. Such refusals are separately logged host
boundaries and terminate the actual episode; they are never called native gate
responses or credited as security successes.

`runs/HOOKS.md` lists every original script/fragment and the exact 12 active
registrations across nine commands. All registered hooks are wired unchanged;
the unwired legacy load-all-memory loader is excluded. Other unwired helpers or
client-inapplicable events receive explicit reasons. Each proof item records
which hooks fired and their actual stdout, injections, blocks and model usage.
No machine-wide memory/history is loaded. Native unsupported endpoints retain
their recorded behavior; no replacement retrieval is invented.

## Required proof before paid inference

The proof uses the fixed-seed ten official items. Hook model calls, if fired,
use metered gpt-5.4-nano; the proof makes no solver calls. Every item starts
with an independent store, ingests only its raw gold-free history through normal
native writes, calls exact `history`-tag recall and follows native pagination.
Returned IDs and full contents must match this item's actual admissions.

Only after retrieval finishes does the offline validator open official relevance
annotations to check that the returned memories include relevant evidence. Gold
labels never select writes, lookup handles or model inputs. This proves native
memory reachability and relevant-evidence coverage, not semantic ranking or
answer accuracy. Two raw examples, negative-handle controls, real child-kill
controls and source hashes are retained in `runs/longmemeval/recall-proof.*`.

The old five-tool Vestige profile is invalid for a full-product comparison. Its
evidence remains in `runs/invalidated/restricted-mcp-profile-1449b1a9`.
Every one of the 500 Vestige items is rerun under the corrected contract.

## Saved baseline recovery

Bare, BM25 and BGE answers are retained through an explicit pinned resume
manifest. The verifier authenticates unchanged model, prompts, official data and
scorer, aggregate budgets, retrieval implementation, actual provider message
chains, response usage and settled ledger attempts. Original checkpoints,
identities, phases, timestamps and raw API logs remain attributable through
`origin`; only operational shard mapping changes. No Vestige answer is imported.

## Execution and reproduction

Keep the existing official data under `runs/upstream`, hook source audit and
ignored owner-only `runs/.env.openai`. Never print credentials or reset the ledger.

```sh
# First: native/hook proof. Inspect ten outcomes and two raw examples.
bash benchmarks/real_engine/reproduce.sh --benchmarks longmemeval \
  --native-proof-only --resume-baselines runs/lme-source-bundle.json
# After proof: recover baselines, qualify the fresh pilot, report estimated cost.
bash benchmarks/real_engine/reproduce.sh --benchmarks longmemeval \
  --resume-baselines runs/lme-source-bundle.json
```

Both commands rebuild the locked release binary and use one campaign lock.
Execution uses 30 deterministic workers per arm, SHA-256(item_id) modulo30,
seed20261101, and at most40 API requests globally, including retries/judges.
Timeouts are60seconds; transient failures have up to6 retries with jitter.
BGE construction queues outside executor threads so its existing serialized
model access cannot starve native MCP work. Retrieval values are unchanged.

Every finished item is flushed immediately. Errors and actual budget-limited
responses/stores remain visible. Engine death aborts without a fallback.
Equal per-item caps remain6calls,2048aggregateoutputtokens,32768contexttokens.
Every API attempt reserves an upper charge before sending; actual usage settles
it. Ambiguous reservations stay encumbered until reconciled with evidence.

`runs/STATUS.md` updates every5minutes. Exact finished/missing IDs are recorded
in coverage artifacts. Official scoring runs once on each complete merged arm
only after all four full merges and fresh integrity checks. Reports contain
means,95%bootstrapCIs,n,costs,dataset/source/binaryhashes and termination counts.
Partial checkpoints, native self-tests and proof files never become scores.
Everything remains on the local `bench/real-engine` branch; no push is automatic.

The full-run estimate prices only unfinished solver items plus all 2,000 official
judgments and includes the new window's existing spend/reservations. It is printed
before full execution; an estimate above the authorized allowance holds the run.
