# LongRCA offline preparation and evaluation contracts

Verified 2026-10-31 against the [paper v4](https://arxiv.org/html/2608.15242v4),
[official project](https://longrca-bench.github.io/),
[official dataset](https://huggingface.co/datasets/CLoud5-real/longrca-bench), and
[official leaderboard repository](https://github.com/longrca-bench/longrca-bench.github.io).

Dataset revision: `98e00bc71110d836457ea8f7def242908fe768c1`.
Leaderboard repository revision: `7155849100c8698dfd26da7c472d77041891169f`.
The repository exposes static leaderboard, validation and paper-export code;
the linked public surfaces inspected here did not supply an RCTA inference runner.
The bounded sparse clone and pinned exporter are staged in `upstream/longrca`.

Both official Parquet files are staged locally, total 73,203,553 bytes. Their
byte counts and LFS SHA-256 identities are checked before decoding. The complete
Git tree index identifies all 1,140 Full and 200 Mini records. Mini preserves the
official fixed 40-per-source set; `--limit` takes a deterministic lexical prefix
for a preparation smoke, which is not a balanced or representative benchmark.
Preparation does not run models or produce model performance scores.

Neither the checked dataset card nor the checked project repository declares
a license. Local research preparation can continue; redistribution and public
submission need a license review. The paper's arXiv license is a separate grant.
The official site describes its public-label leaderboard as a reproducibility
evaluation, not a hidden-test competition.

From the campaign repository root:

```sh
python3 -m unittest discover -s tests -p test_longrca.py -v
python3 -m campaigns.longrca.prepare --upstream upstream/longrca --output evidence/longrca-fixture --fixture
python3 -m campaigns.longrca.prepare --upstream upstream/longrca --output evidence/longrca-mini8 --split mini --limit 8
python3 -m campaigns.longrca.prepare --upstream upstream/longrca --output evidence/longrca-full --split full
```

Destinations must be new directories. Earlier evidence is never overwritten.
Fixture tests use a marked synthetic trace. Native-call tests use a clearly
marked fake binding; they qualify adapter contracts only. Decoding official
Parquet requires the local optional dependency in `requirements.txt`. No network,
credential or provider access occurs in preparation, scoring, or tests.

Each preparation writes:

* `source.json`: source pins, checksum qualification and license status.
* `manifest.json`: selected IDs, unmodified public-evidence digests, counts.
* `public.jsonl`: exactly `question_ID` and the ordered `history`.
* `scorer-only/references.jsonl`: gold role, step and rationale, mode `0600`.
* `readiness.json`: preparation state, zero inference and pending runtime gates.

The model process must receive only the public evidence and manifest, with no
filesystem/native-store access to the scorer-only directory, upstream annotated
Parquet, or official sample. Mode `0600` separates users, not processes running
as the same account: the execution sandbox must enforce the solver/scorer split.
Source files and traces are untrusted data and must not be executed as commands.

`schema.py` rejects added fields and changed indices. Original history text,
recorded names, message roles and ordering remain intact. Gold label fields never
enter `arm_input`, native `smart_ingest`, native links, or step directories. Every
native write is an explicit write-only history commit; all solver reads are
dynamic `recall` by exact handle or native `causal_walk` from a solver-selected
step. Controller writes never count as model reads.

`NativeTraceSession` requires a supplied `NativeBinding` with an isolated store
identity, native tool calls, and independent receipt verification. Binding code
must verify Operator admission, Reflex call/session/digest acknowledgment and
native signed receipts against its frozen runtime. It must project the native
envelope into the documented result dictionary (`nodeId` for writes). There is
no default native process or substitute graph implementation here. A transport
fault, empty/error/blocked response or invalid receipt latches the session closed,
records the failed attempt, and prevents further native calls. Actual runtime,
hook, Strata, Operator and RSB qualification is a separate campaign gate.

The official history schema has no typed causal edges. This adapter supplies
**zero edges**: adjacency and handoff-looking names alone cannot justify a causal
link. Any native exact-identity edges or empty walks must remain visible as native
diagnostics. Do not plant reference roots or inject gold-derived edges to improve
walk results. A deterministic walk diagnostic is not a role/root prediction and
is not scored as one. A later traced dependency extractor would need its own
frozen rules, evidence, tests and declared hypothesis status before comparison.

Matched-backbone evidence contracts in `arms.py`:

| Arm | Evidence access | Comparison meaning |
| --- | --- | --- |
| `no_memory` | Full original history in one stateless prompt | Full-history baseline |
| `long_context` | Identical full original history | Same baseline equivalence group; do not count as independent |
| `trace_lookup` | Original record directory and exact step reads | Dynamic control without persistent memory |
| `vestige` | Same directory, native exact recall and causal_walk | Dynamic native memory route |

All arms share the same public-evidence digest and available source text. Freeze
model, decoding, limits, total cost ceiling and per-task selection before any
model calls. Record different prompt/token costs, tool overhead, receipt failures,
truncation and abstentions. Full history that exceeds a context limit is an
explicit failure or separately preregistered truncation condition.

`scoring.py` implements the paper's benchmark-comparison metrics: explicit role
accuracy, exact root-step accuracy, root within five, and source-weighted
valid-output root MAE. Roles normalize case, whitespace, NFKC and explicit handoff
suffixes, parity-tested against the pinned official exporter. Role predictions
are never inferred from the sender of the predicted step. Missing/invalid fields
count as errors independently; every expected ID remains in accuracy denominators.
MAE excludes invalid roots and is undefined if any included source has no valid
prediction, which is reported explicitly. This is the Full/Mini comparison
contract, not the paper's differently normalized component-ablation protocol.

```sh
python3 -m campaigns.longrca.scoring --public evidence/longrca-full/public.jsonl --references evidence/longrca-full/scorer-only/references.jsonl --predictions predictions.jsonl --output scores.json
```

Predictions use `question_ID`, `predicted_role`, `predicted_step`. Invalid response
fields may be null; a missing prediction is counted as incorrect. Unknown or
duplicate IDs are rejected. Keep this scorer in a separate process after sealing
predictions. No model scores have been obtained by this preparation work.
