# Developer task cost accounting — v3 stage 1

This offline Python tool freezes a comparison contract, records exported events,
and computes price-derived USD per successful task. It uses only the standard
library. It does not call a model, execute a developer task, read credentials,
or claim that Vestige saves money.

The existing `../agent-memory-eval` suite measures retrieval quality. This
accountant measures submitted task usage and overhead separately so retrieval
metrics cannot be mistaken for end-to-end economics.

## Run the synthetic smoke fixture

Choose a new output directory; existing bundles and reports are not overwritten.

```sh
python3 benchmarks/task-cost/example.py /tmp/vestige-task-cost-example
python3 benchmarks/task-cost/ledger.py report /tmp/vestige-task-cost-example \
  --output /tmp/vestige-task-cost-example/report.json
python3 -m unittest discover -s benchmarks/task-cost/tests -v
```

Both synthetic arms intentionally tie. Each has two requests, one declared
success, one declared failure, and estimated setup overhead. Fixture arithmetic
is USD 0.00488 in requests plus USD 0.01 setup, hence USD 0.01488 per successful
task. These are invented rates and usage, not vendor prices or product results.

## Frozen contract

`example.py` generates the complete machine-readable contract shape. Replace
its fixtures with evidence before any real evaluation. `contract.json` pins:

- `version: vestige-task-cost/v1`, `evidence_kind`, currency USD, repetitions.
- Arm identities: model, immutable model revision, effort, agent revision,
  tool configuration hash, initial memory snapshot hash. The first four must
  match across arms. Configuration and snapshots may differ by treatment.
- Every case's ID, workload and cold/warm state. Prompt, source, available
  history, and evaluator are bundle-relative artifacts with SHA-256 hashes,
  shared across arms. These can be archives or versioned manifests; the
  accountant does not recursively inspect an archive's contents.
- Price IDs with exact response model, effective date, price source, and
  decimal-string USD per million disjoint token units. Pin the actual tier,
  batch/service class and cache TTL used; do not silently use today's rate.
- Execution and scoring policies declared before trials.

```sh
python3 benchmarks/task-cost/ledger.py freeze ./my-bundle
```

Freeze verifies the artifacts and exclusively writes `contract.lock.json`,
binding the contract bytes and accountant source. Report rechecks every hash.
Preserve that locked accountant revision for replay; changing its code requires
a separately identified run. The lock detects changes relative to the supplied
lock; it is not a signature, trusted timestamp, or proof of when execution began.

## Record events

```sh
python3 benchmarks/task-cost/ledger.py record ./my-bundle --event ./event.json
python3 benchmarks/task-cost/ledger.py report ./my-bundle --output ./report.json
```

The recorder is single-writer and appends JSONL. It checks artifact references
and duplicate event IDs; final report performs accounting validation. The raw
request and response must be captured by the calling agent runtime. The capture
seam below instruments an explicitly wrapped call; it does not automatically
intercept other calls or prove every provider request was exported.

Event types:

| Kind | Required fields beyond unique `id`, `kind`, `arm` |
|---|---|
| `request` | `case`, zero-based `trial`, `phase`, `format`, `price_id`, hashed `request` and `response`; optional nonnegative `elapsed_ms` |
| `outcome` | `case`, `trial`, `status` (`success`, `failure`, `timeout`, `aborted`), hashed evaluator `evidence` |
| `overhead` | `phase`, `usd` decimal string or null, `basis` (`measured` or `estimated`), hashed `evidence` |
| `overhead_coverage` | `status` (`complete` or `incomplete`), hashed `evidence` explaining included and excluded charges |

Request phases are `agent`, `ingest`, `embedding`, `rerank`, `maintenance`, and
`setup`. Non-agent requests and overhead may be arm-wide with no case/trial.
Every planned task needs an agent request and exactly one terminal outcome.
Retries use distinct event and provider response IDs. Duplicate exported
provider IDs within an arm fail rather than charging the same response twice.
Optional memory receipt IDs may be retained on source events; receipt identity
does not substitute for provider usage.

Responses are full exported JSON objects with `id`, `model`, and `usage`.
Currently supported usage formats:

- **`openai_responses`:** input total includes cached input; subtract the cached
  portion before charging uncached input. Output includes reasoning; report the
  reasoning subset without adding it a second time. Missing cached detail makes
  the split unknown.
- **`anthropic_messages`:** uncached input, cache reads, cache creation and output
  are separate. Creation requires the 5-minute/1-hour token breakdown when
  nonzero. An absent TTL breakdown cannot be priced by guessing.

These are text-token accounting adapters tested with synthetic shapes, not
live SDK certification. Multimodal pricing, tools billed by use, cache storage,
discounts, taxes, and other charges need explicit overhead records or a new
validated adapter. Missing provider usage, including failed requests with
unknown billing, stays unknown. An embedding API with another usage shape is
not silently treated as a text response; record its supported cost evidence as
overhead until an adapter is implemented.

## Report semantics

All recorded costs, including failed attempts and arm-wide setup, enter the
numerator. Divide by recorded verified successes only when accounting is
complete and at least one success exists. Zero successes produce null, never an
invented finite cost. Missing outcomes, missing agent requests, unknown charges,
or absent/incomplete overhead declarations suppress complete total and
cost-per-success fields. `known_usd` remains a subtotal, not a full bill.

Estimated overhead is explicitly flagged. `accounting_complete` means the
submitted ledger meets these structural checks. It does not mean invoices were
reconciled, overhead independently audited, outcomes independently verified, or
identity declarations attested. The accountant hashes evaluator evidence but
does not execute or interpret that evaluator. Success/failure rates and raw
cost-event rows stay visible; no automatic winner or savings headline is emitted.

Keep raw request/response bundles private: they may contain repository content.
Capture request bodies without authorization headers or credentials. A public
derivative needs its own review and hashes. The example contains synthetic data
only. No inference, network, subprocess execution, or cloud storage is involved.

## SDK capture seam

`capture.measured_call` accepts a caller-owned synchronous SDK method. It does
not construct clients, discover credentials, or launch agents. For an existing
OpenAI client, for example:

```python
from capture import measured_call

response = measured_call(
    bundle_directory,
    {"id": "trial-0-request-1", "kind": "request", "arm": "native",
     "case": "case-1", "trial": 0, "phase": "agent",
     "format": "openai_responses", "price_id": "frozen-price-id"},
    {"model": "exact-frozen-model", "input": task_prompt},
    client.responses.create,
)
```

Use `anthropic_messages` and `client.messages.create` for that response shape.
Non-streaming request bodies only: streaming needs a separately verified final
usage collector. Request bytes are captured before execution. SDK return values
are preserved; `model_dump(mode="json")` objects and dictionaries are supported.
Exceptions are reraised after recording unknown billing. Exception messages are
omitted because they can contain credentials; explicit credential/header fields
are rejected. Serialization failures also record unknown billing. This is not a
general secret detector: never put credentials into prompt text.

Tests use injected fake SDK calls, including failure and duplicate-call cases.
A live SDK/provider run remains unqualified. As with any local recorder, process
death or disk failure can leave incomplete capture; reconcile provider records
before declaring a real run's overhead/export coverage complete.

## Next gate

Integrate capture with one actual agent runtime and reconcile its exports with
provider usage. Freeze the development suite and verify each baseline's native
context/cache path. Then run repeated, isolated trials with independent scoring.
Do not claim token-fee savings from this accounting smoke test.
