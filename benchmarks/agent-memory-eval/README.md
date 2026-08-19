# Vestige Agent Memory Eval

`agent-memory-eval` is the versioned, reproducible retrieval evaluation gate
for Embedding Profiles. It measures agent-memory retrieval behavior, not a
generic embedding leaderboard. It contains no model weights and does not
download, install, activate, or recommend any model.

Start with [SPEC.md](SPEC.md). The committed `fixtures/v1` data and reference
rankings are a harness smoke test only; they are deliberately tiny and must not
be used for product-performance claims.

## Run the fixture smoke test

```sh
python3 benchmarks/agent-memory-eval/evaluate.py \
  --fixture-dir benchmarks/agent-memory-eval/fixtures/v1 \
  --rankings benchmarks/agent-memory-eval/fixtures/v1/reference-ranked-results.json \
  --output /tmp/vestige-agent-memory-eval-reference-report.json

python3 -m unittest benchmarks/agent-memory-eval/tests/test_evaluate.py
```

The evaluator writes a machine-readable report. A real candidate run must also
provide the run manifest and measured operational data required by the spec.
