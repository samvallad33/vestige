# MemForm native-memory preparation

Verified primary sources: [paper v1](https://arxiv.org/abs/2610.14431v1) and
[author repository](https://github.com/opanhw/memform-code). The MIT evaluation
release is cloned at `0565117042a31a5f78576f89ff6b8a77c393ee09`. It contains
994 frozen MCQ items and 897 free-form items; the construction pipeline is not
released. All 994 MCQ items pass the original compiled-case validator.

```sh
python3 -m campaigns.memform.prepare --upstream upstream/memform \
  --output evidence/memform-prepared --limit 10
python3 -m unittest discover -s tests -p test_memory_safety.py -v
```

Preparation uses no credentials, models or provider requests and does not load
an `.env` file. It writes a private manifest, anonymous runtime packets and a
separate private scoring sidecar. It selects cases round-robin across all five
interfaces before seeing target outputs. Case IDs, interface labels, answer
labels, provenance and oracle policies never become runtime prompt features.
Legitimate record text and the specified memory-building instructions remain
unchanged. P4 receives the retained subset; its full-pool control stays separate.
P5 preserves both permutations of the identical experience multiset. Native
record handles are opaque hashes rather than clean/poisoned slot labels.

`load_prepared` verifies the source/data/adapter identities and runtime/sidecar
digests. The native `commit_packet(stack, session, packet)` interface uses only
explicit `smart_ingest` writes and native `receipt` verification; it returns
handles for later solver-issued recall. It never dumps write outputs into model
headers. The caller must use a fresh audited native store per arm and condition.
This replaces the upstream memory updater with the native memory stack and must
be reported as the **native-memory-adapter-MCQ track**, not an unchanged
LangChain/LlamaIndex run. It does not claim a Fourier HRR or semantic admission
mechanism that the selected binary does not expose.

`score_results` uses the original fixed-denominator scorer. Official case/task
gold is checked against the pinned data. Unknown synthetic rows require an
explicit fixture flag and remain labeled synthetic. The undefended baseline
cannot silently drop missing predictions to shrink eligibility. Missing or
technical method outputs withhold aggregate ASR/accuracy rather than count as
a defense win. No benchmark model score has been produced by preparation.

The public release has a concrete import defect: `memform/freeform/__init__.py`
imports the withheld `memform.freeform.build`, so importing the normal core
runner fails. This adapter uses the stdlib-only original validator and metrics
without importing that package. The MIT-licensed MCQ parser is copied literally
and checked against the pinned original function AST, allowing only the stdlib
`re.I`/`re.IGNORECASE` alias style changed by formatting. Upstream is untouched.
The free-form inventory is pinned; its full runner and model adjudication remain
separate pending work. This is not a free-form score or a repair to upstream.

Immutable Strata receipts establish what bytes were admitted. They do not prove
those bytes are true or that an attacker cannot influence a downstream model.
The poisoning benchmark is intended to measure that distinction.
