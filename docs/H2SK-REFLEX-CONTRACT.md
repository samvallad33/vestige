# H²SK source-carrying packet and output gate

This change implements a discrete, dataset-blind receipt compiler and a bounded
output verifier for #610, #611, #613, #615, #616, #617 and #618. It does not add
an MCP verb, an embedding index, a second source store, or a model dependency.

## Admission and factual currency

`h2sk::admit::admit_statement` is shared by the native ingestion adapter, CLI,
Python admission adapter and optional WebAssembly adapter. Source statements,
their source dates, exact tags and already recorded edges are the inputs.
Gold answers, relevance ids, dataset names and question-type labels are absent.

The conservative prose grammar recognizes explicit residence, employer,
preference and owned-gear statements. It retains unsupported text verbatim.
`Correction:`/`Actually,` and explicit moves can propose a closed-vocabulary
edge only with two distinct exact values on the same slot and one unresolved
predecessor. Existing correction/supersession edges remove their targets from
that predecessor set. Ambiguous alternatives remain `NoUniqueApex`; a newer
timestamp alone is not a correction.

The native `corrects` admission path also checks the actual scoped endpoints,
source correction wording and exact values before saving, then rechecks at
the normal gate-backed edge write. It does not retire either source.

The compiler accepts only `Touched`, `Supersedes`, `Corrects`, `Contradicts`,
`Entails`, `Replaces`, `Contains`, and `Next`. These are compiler input kinds;
the durable native edge vocabulary is unchanged. Unsupported native edge
kinds are not silently renamed. A general-purpose edge with no slot/value
binding cannot create a currency row.

For `B corrects A; C supersedes B`, the unique apex is C, its horosphere is B,
and A remains historical off-spine evidence. A cycle is an error. Generic
slots are scoped to a single exact person or symbol; unrelated people never
share an inferred current value. Source-tag order does not change value sets.

## Reader view

The reader receives the selected sources plus their exact slot history. A
selected historical hit cannot hide its controlling successor. The view
contains source quotes, `CONTROLLING`/`OVERRULED` banners, modality and date
annotations, a compact state trajectory, and the compiler receipt hash.
An unlinked duplicate is never described as overruled by mere recency.

Original text is never rewritten. Supported relative dates are annotated
against the source origin, not against import time. Unknown origins stay
unknown. Only explicitly supplied identity bindings resolve pronouns.
Day/week deltas use calendar-date integer arithmetic, including leap dates.
Hypothetical/rejected statements cannot become an executed current value.
Confirmed preferences can support advice without a false information-absence
answer.

## Receipt rows and integrity

The receipt contains tally, currency, section, frustration, inclusion, impulse,
hologram and gate rows. Impulse uses recorded stability or recorded ancestor
depth and carries a previous receipt hash when the supplied prefix has one.
The bounded Fourier diagnostic measures reconstruction of a source-hash view;
it establishes neither associative-memory capacity nor factual truth. It is
not a replacement for the experimental continuous engines in #590–#594.

The receipt hash binds the derived view, query/options and source-prefix root.
Native packets obtain their root and data from the open Strata log's admitted
frames, checked against the effect index. The compiler hash is not an
Ed25519 signature and is not an `eff-` mutation receipt. Existing Strata effects
retain their own signed-log proof and reversible backfill receipts.

## MCP integration and replay

`session(action="start", memory_hook=...)` calls the compiler while building
the packet. The packet includes `compile_receipt`, `compile_options` and
`reader_context`. A benchmark reader cannot opt out by skipping a sidecar call.

`precedent(query="compile_receipt", log_offset=N, compile_options=...)`
rebuilds the receipt for that exact prefix. Optional `draft` and
`expected_receipt_hash` fields run the output verifier. These reads append no
log frames or salience changes. Current suppression is still applied to
historical reads; changed visibility therefore invalidates an old slice.
The native prefix cache is bounded and keyed by store, scope, prefix offset
and present head, so a suppression effect invalidates it.

## Generation boundary

The gate checks exact no-evidence abstention, a stale literal used alone for a
current-state query, false abstention with applicable current/preference
evidence, and receipt/query/slice mismatch. Normal replies need one generation.
`benchmarks/shared/runner.py` permits one corrective retry with explicit source
pointers; a second rejection never reaches its answer/scorer boundary. It
returns the answer string alone to scoring code. It does not implement the
official scorers or claim completed runs of the five evaluation suites.

This bounded gate does not detect every hallucination or prove general semantic
correctness. Unrecognized paraphrases and implicit real-world causality remain
outside its exact-evidence contract.

## Portability and validation

`crates/vestige-h2sk-wasm` compiles the same pure Rust module without the native
store or daemon. Its caller owns durable admission, consent, visibility,
source-root authentication and `eff-` writes. It exports admission, compilation,
reader-view rendering and output verification.

Synthetic tests cover correction chains, unresolved alternatives, cycles,
source preservation, source-time annotations, leap dates, modality, stale
citations, query/receipt tampering, empty views, entity isolation and a
50,000-record source projection. The Windows development-build projection
test compiled two selected sources from 50,002 records in approximately 12 ms
on its recorded run; this is not an iPhone measurement or a universal latency
guarantee. Record fresh device measurements before publishing performance claims.

The new roadmap #620–#670 records future ideas separately and does not expand
this implementation's launch acceptance criteria.
