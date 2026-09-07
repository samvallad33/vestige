# Before You Change That

This is the public evidence package for Demo 4, a frozen three-arm agent run on
a 22-record synthetic ledger reversal and reconciliation incident. The useful
observation is concrete: on the Vestige arm, the exact saved constraint for
`src/ledger/reconcile.rs` arrived when that file became relevant, and the
retained evidence then shows a repair that passed every application check.

This single run does not establish that the remembered constraint caused the
repair, or that one memory system ranks above another.

> **Run boundary:** Control and Vestige completed their model turns naturally.
> The MCP Memory Service arm was interrupted at the original 900-second cutoff,
> although its final recorded application files pass 21/21 checks. The
> historical aggregate fields `status: "COMPLETED"`, `demo_ready: true`, and
> `comparison_valid: true` are retained as evidence, but they do not override
> that interruption and must not be read as three-way natural completion.

## Recorded setup and result

Every arm received the same prompt, `gpt-5.6-sol` at `ultra` reasoning, and the
same frozen initial application and 22-record archive. Workspaces and memory
services were isolated by arm.

| Arm | Model-session outcome | Recorded application result | Raw input / cached input / output counters |
| --- | --- | --- | --- |
| Control | Natural completion in 560.101s | 21/21 | 1,182,794 / 1,079,808 / 18,739 |
| MCP Memory Service | Interrupted at 900.033s | 21/21 | Unavailable; no final usage row |
| Vestige | Natural completion in 581.452s | 21/21 | 1,302,101 / 1,227,648 / 15,208 |

The input counter already includes cached input. These are retained provider
counters, not billing figures.

The application evaluator covers the pending reversal, delayed charge,
append-only journal, restart reload, duplicate suppression, negative cases,
shuffled events, configuration behavior, real state files, exact owner, and
the unrelated export path. All three final application trees passed all 21
checks.

The Vestige transaction log records one exact active context-intention delivery
for `src/ledger/reconcile.rs`. The coordinator left that intention active during
the model turn and marked it fulfilled only after the application evaluator
passed. Three captured retrieval envelopes also verify under Ed25519 and reject
a modified payload. Those signatures establish the integrity of their signed
retrieval payloads; they do not sign or prove the intention lifecycle, the
application patch, causal influence, or correctness. The signing key is the
disposable local demo key; it provides no external identity certification or
trusted timestamp.

## Verify the retained evidence

Use Python 3.11 or newer. The standard verifier does not call a provider, rerun
a model, start a memory service, or execute the application.

```sh
cd benchmarks/before-you-change-that
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
```

That checks the exact 96-file compact evidence set against its public manifest, the retained arm
outcomes and sequence ordering, the exact intention delivery and lifecycle,
and all three Ed25519 envelopes. The keys come from the retained demo store;
they do not certify an external identity. A standard-library-only integrity check is
also available:

```sh
python3 verify.py --hashes-only
```

From the extracted release bundle's `before-you-change-that/` directory,
verify all 643 public evidence files with:

```sh
python verify.py --full-evidence
```

To execute the application-only 21-check evaluator again against disposable copies
of the three retained final application trees:

```sh
recheck_dir="$(mktemp -d)"
python recheck_application.py --output "$recheck_dir"
```

The application recheck runs no model and requires no network access. It
requires Rust and macOS `sandbox-exec`; it fails closed when that sandbox is not
available. The application recheck is not supported on Linux. It rejects
symlinks and special files and does not modify the recorded evidence.

## Inspect the timeline

```sh
node viewer/serve.mjs --port 4173
```

Open <http://127.0.0.1:4173/viewer/> to inspect the sanitized event timeline.
The two public MP4s are complete timeline replays rendered from those events;
they are not live screen recordings. The original screen recordings remain in
the preserved local archive and are not part of this public export.

The complete public evidence and both replay videos target the GitHub release
tag `benchmark-before-you-change-that-2026-09-06`. See [ARTIFACTS.md](ARTIFACTS.md)
for the current artifact links and hashes.

## Evidence map

- [`RUN-STATUS.json`](RUN-STATUS.json) is the concise machine-readable result
  and claim boundary.
- [`evidence/result.json`](evidence/result.json) retains the historical
  aggregate and per-arm evaluator reports.
- [`evidence/vestige/native-transactions.json`](evidence/vestige/native-transactions.json)
  contains the native model-session transactions.
- [`evidence/vestige/intention-lifecycle.json`](evidence/vestige/intention-lifecycle.json)
  separates delivery from the coordinator's post-test fulfillment.
- [`PUBLIC-DERIVATION.json`](PUBLIC-DERIVATION.json) records public hashes,
  inclusion flags, and privacy transformations. Public derivatives have their
  own hashes; the retained original seal is historical evidence and does not
  validate redacted bytes.
- [`viewer/README.md`](viewer/README.md) documents the replay UI and deterministic
  MP4 renderer.

The historical full-run harness is retained as reference evidence. This public
package is not a turnkey environment for fresh provider or model comparisons,
and none of the verification commands above reruns a provider.
