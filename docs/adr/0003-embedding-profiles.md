# ADR 0003: Explicit, Reversible Embedding Profiles

**Status**: Accepted
**Date**: 2026-08-11
**Related**: [ADR 0001](0001-pluggable-storage-and-network-access.md),
[Agent Memory Eval v1](../../benchmarks/agent-memory-eval/SPEC.md)

---

## Context

Embeddings determine what evidence Vestige can retrieve. They therefore carry
more meaning than a model name: the encoding template, tokenizer and model
revision, normalization, output dimension, and index all affect whether a
stored vector is comparable to a query vector.

The current `VESTIGE_EMBEDDING_MODEL` selector does not make that contract
durable or make a model change a deliberate user operation. In particular,
legacy Nomic vectors created from raw text must not be presented as equivalent
to vectors created with Nomic's retrieval document/query prefixes. Optional
model capability must follow Vestige's local-first, opt-in hooks philosophy:
an existing user can remain on the default indefinitely without a silent
download, behavioral change, re-embedding, or activation.

This ADR records the contract and its implementation: profile identity,
storage isolation, activation gate, dashboard, CLI, evaluation harness, and a
local Qwen runner. Qwen lifecycle operations accept only an explicitly supplied
local artifact directory, verify its pinned revision and hashes, and fail
closed on any mismatch. They never download a model at runtime.

## Decision

Vestige will use **Embedding Profiles**, not an ambient model selector.

- **Nomic Compact** remains the default local profile. Existing raw-text Nomic
  vectors become the preserved `nomic-v1.5-legacy-raw-256` profile. A correctly
  encoded Nomic retrieval profile is a distinct profile, not an in-place
  reinterpretation of legacy vectors.
- **Qwen Balanced 0.6B** and **Qwen Max 4B** are optional profiles. They are
  candidates, not product-quality claims, until the versioned Agent Memory Eval
  and release gates provide reproducible evidence.
- A user must explicitly choose each state transition: install, evaluate,
  migrate, then activate. Installation and completed migration never activate a
  profile. An application update, environment variable, runtime detection, or
  hardware detection must not choose any of those transitions.
- The active profile is a durable, atomic database pointer. Activating a ready
  profile changes only that pointer. Rollback selects a retained ready profile
  by changing the pointer back; it does not re-embed the corpus.
- Keyword search remains shared. Semantic scores and query caches are profile
  scoped. Vestige must not compare, merge, average, or use fallback semantic
  scores across profiles.

The initial profile identifiers make the encoding and vector-shape choice
visible:

| Profile ID | Intended role | State in this ADR |
| --- | --- | --- |
| `nomic-v1.5-legacy-raw-256` | Preserved existing vectors | Compatibility profile |
| `nomic-v1.5-retrieval-v1-256` | Nomic document/query retrieval encoding | Catalog migration target |
| `qwen3-0.6b-retrieval-v1-256` | Compact Qwen candidate | Implemented, opt-in |
| `qwen3-0.6b-retrieval-v1-1024` | Higher-dimensional Qwen candidate | Implemented, opt-in |
| `qwen3-4b-retrieval-v1-1024` | High-resource Qwen candidate | Implemented runner, opt-in |
| `qwen3-4b-retrieval-v1-native` | Measured native-dimension candidate | Implemented runner, opt-in |

The listed Qwen shapes are comparison candidates, not a statement of final
dimensions or quality. Selection requires a complete reproducible evaluation.

## Profile contract

Every vector row and index must have a durable `profile_id`. A profile manifest
is immutable once vectors exist and contains at least:

```text
profile_id
model_id
immutable_model_revision
verified_model_artifact_hashes
runtime_backend
embedding_dimension
normalization_method
document_encoding_template
query_encoding_template
maximum_token_limit
chunking_strategy
created_at
status
```

Changing any value that affects a produced vector creates a new profile ID and
requires a separate migration. A profile manifest is included in retrieval
receipts and evaluation artifacts so an investigation can identify its retrieval
environment.

The embedding boundary becomes two explicit operations:

```text
embed_document(memory_content, profile)
embed_query(user_request, active_profile)
```

For the first Nomic retrieval profile, the versioned templates are:

```text
document: "search_document: " + memory_content
query:    "search_query: " + user_request
```

Qwen's document template and agent-memory query instruction are part of its
profile manifest. They are benchmarked, versioned text, never silently edited
configuration. No cross-profile template substitution is allowed.

## Storage, lifecycle, and deletion

Each installed profile owns isolated vector rows, a dimension-compatible HNSW
sidecar, an integrity manifest, a migration checkpoint, and a query cache.
The active profile is the only source of semantic candidates in hybrid search.
If the active semantic index is unavailable, behavior must be an explicit
profile error or keyword-only result marked as such; it must not silently score
with another profile.

Migration is destination-only and resumable:

1. Verify a user-selected local artifact, its pinned revision, and every
   expected hash; then run local hardware and free-storage preflight.
2. Create a local database/index snapshot, estimate impact, and require an
   explicit confirmation.
3. Write vectors only to the destination profile, checkpoint per memory, and
   keep active-profile semantic retrieval and keyword retrieval available.
4. Validate count, dimension, artifact/profile hash, index membership, and
   sampled retrieval before marking the destination `ready`.
5. Present the evaluation and validation result. Only an explicit activation
   changes live semantic retrieval.

Pause, restart, and repair keep failures observable. Failed download or runtime
initialization is a recoverable per-profile state (`not_installed`,
`retryable_error`, or `repair_needed`), never a process-wide permanent failure.

Purge is strict: a purged memory's content and vectors are deleted from every
installed profile and index. Only content-free audit tombstones may remain.

## Privacy, packaging, and runtime policy

- Optional profiles make no network request until the user explicitly selects
  **Install**. Offline local-directory installation is supported.
- The installer and default release must not require Qwen weights, a GPU,
  extra RAM, or an extra setup step to retain Nomic behavior.
- Model and tokenizer artifacts are revision-pinned and hash-verified before
  use. No external dependency may replace a pinned artifact silently.
- Profiles run locally and do not send memory content, hardware telemetry, or
  embeddings to a hosted embedding API.
- Metal, CUDA, and CPU availability are explicit runtime/build capabilities.
  Hardware UI guidance is measured on the actual runner; it is not inferred
  from a marketing table. Qwen Max has a hard measured preflight and a clear
  high-resource warning.

## User-visible state machine

```text
not_installed -> installed -> evaluated -> migrating -> ready -> active
                    |             |              |          |
                    +--> retryable_error <-------+          +--> inactive
                                                     rollback -> prior ready profile
```

`active` is exclusive. `ready` and `inactive` profiles remain intact until a
separate, explicit retention/deletion action. A dashboard profile card and CLI
must show the exact pinned revision, local disk requirement, measured device
guidance, dimension, estimated index expansion, current state, local-only
status, evaluation result, and one clear rollback action.

The intended CLI vocabulary mirrors the state separation:

```text
vestige embeddings install qwen3-0.6b-retrieval-v1-1024 --from <verified-artifact-directory> --yes
vestige embeddings evaluate qwen3-0.6b-retrieval-v1-1024 --from <verified-artifact-directory>
vestige embeddings migrate --to qwen3-0.6b-retrieval-v1-1024 --from <verified-artifact-directory> --yes
vestige embeddings activate qwen3-0.6b-retrieval-v1-1024
vestige embeddings rollback --to nomic-v1.5-legacy-raw-256
```

The first three commands deliberately require the artifact directory on every
operation. It is verified for that process and never stored in a profile
manifest or accepted by the dashboard HTTP API.

## Evaluation and release gates

Before any recommendation or comparative performance statement, the candidate
must pass the versioned [Agent Memory Eval v1](../../benchmarks/agent-memory-eval/SPEC.md).
Each published run includes pinned fixture and evaluator hashes, exact profile
manifests, model/tokenizer artifact hashes and immutable revisions, command,
device/runtime details, raw ranked results, raw operational measurements, and
failure cases. Public leaderboard scores alone do not establish agent-memory
retrieval quality.

An opt-in release cannot pass until all of the following are demonstrated:

1. Default Nomic users have no behavior change and no optional-model download.
2. Install, evaluate, migrate, and activate are distinct user-confirmed
   operations; activation is the sole live-retrieval transition.
3. Crash injection proves migration cannot leave a half-active profile or
   corrupt the prior active index; restart/resume and repair are proven.
4. Rollback changes the active pointer immediately and preserves the old index.
5. Profile-aware purge removes every applicable vector and index entry.
6. Every retrieval receipt identifies the profile used, and cross-profile
   vector scoring/fallback is rejected by tests.
7. The Agent Memory Eval has reproducible artifacts and all release thresholds
   declared before inspection of candidate results.
8. Packaging, privacy, offline installation, artifact verification, and the
   supported hardware matrix are verified for each shipped runtime.

## Consequences

This makes embedding behavior inspectable and reversible, at the cost of
additional profile metadata, disk during migration, and a stricter release
process. It intentionally rejects convenience features that hide a corpus-wide
semantic change. The remaining release work is candidate-specific evidence:
Qwen 4B needs a live run with its verified artifacts, and any quality
recommendation requires a preregistered, representative Agent Memory Eval
rather than the smoke fixture.
