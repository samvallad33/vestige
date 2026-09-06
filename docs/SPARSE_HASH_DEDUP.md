# Sparse-Hash Dedup Shield

A similar embedding is a candidate for review, not proof that two memories
state the same fact. `Timeout is 30 seconds` and `Timeout is 300 seconds` must
remain distinct even when their embeddings are very close.

The shield combines two independent changes:

- **Candidate filtering:** scans above 128 embeddings use deterministic sparse
  random-hyperplane signatures and band buckets before exact cosine scoring.
  Small scans are exhaustive. The former 2,000-embedding refusal is removed.
- **Content preservation:** non-identical text cannot be classified as an
  automatic merge match. Plans expose `signals.requiresReview` and require
  `confirm=true`, even when their weighted similarity score is high. Explicit
  merges preserve case, whitespace and punctuation distinctions in the text.

Ingestion also preserves non-identical incoming text that would previously have
been discarded as reinforcement at or above the near-identical threshold
(default 0.92). It creates a separate related memory. Byte-identical text can
still reinforce an existing memory. This change does not replace the separate
prediction-error update behavior below that threshold or explicit update intent.

## Using it

The default scan remains read-only:

```json
{"action": "scan", "similarity_threshold": 0.85, "limit": 20}
```

`duplicateClusters` reports `candidateFilter`, `approximate`, `pairsChecked`
(candidate pairs sent to cosine during scanning), and `possiblePairs` (the
unfiltered pair count). A cluster containing different text suggests `review`.
Both scans exclude inactive memories and compare only within the same project
scope. Tag filters continue to work.

Use an exhaustive scan to check for candidates that approximate hashing missed:

```json
{"action": "scan", "similarity_threshold": 0.85, "exhaustive": true}
```

After reviewing a candidate, use the existing `plan_merge` or `plan_supersede`,
inspect the diff, then `apply` with `confirm=true`. Undo remains available through
`dedup(action="undo", operation_id="...")`.

## Automatic operations

Background consolidation remains disabled unless
`VESTIGE_AUTO_CONSOLIDATE_MERGE` is explicitly enabled. When enabled, it accepts
only byte-identical, nonempty text within one scope and node type, excludes
protected memories, and confirms embedding similarity. Each cluster uses the
existing atomic merge/undo log. Absorbed memories are marked superseded and kept
for audit, instead of being hard-deleted. Transactions contain at most 64 members.
Memories with explicit end dates remain review candidates and are skipped by
automatic consolidation, since different validity windows can distinguish facts.

Unconfirmed plan application enforces the current `auto_apply` policy and match
threshold under the database write lock. Both unconfirmed and background apply
recheck text, tags, scope, node type, protection and lifecycle in that transaction.
Legacy plans without the review signal require confirmation.

## Performance and limits

Signatures have 32 bands of eight bits; lower candidate thresholds use six bits
per band for better recall. Each projection samples roughly one sixteenth of
the embedding coordinates with signed weights. The deterministic seed makes a
scan reproducible. Vectors with at most 32 dimensions use dense Gaussian
projections to avoid degenerate sparse samples around zero coordinates. It is
not a cryptographic hash.

The index and clustering summaries avoid storing all candidate pairs, using
linear memory in the number of embeddings. Dense collision buckets can still
require quadratic work. Exact scoring, lexical comparisons and database reads
remain part of total latency. No universal speedup is promised.

Approximate filtering can miss near duplicates. A seeded 1,000-vector synthetic
fixture retained all 200 planted pairs at 0.92 cosine while forwarding 61,315
of 499,500 possible pairs to cosine scoring (87.7% fewer comparisons). This is a
fixture result, not a real-store recall or wall-clock guarantee. The hash cannot
detect hallucinations or decide whether paraphrases express equivalent facts.
Requiring review for all non-identical text deliberately trades some automatic
paraphrase deduplication for preserving distinctions.

There are no new runtime dependencies, persisted hash columns, model downloads
for hashing, or database migrations. Test fixtures use synthetic vectors and
temporary databases. Existing user stores are not cleaned up by installing code.

## Motivation and attribution

[Mem0 issue #4573](https://github.com/mem0ai/mem0/issues/4573) is a user report
auditing 10,134 entries from one deployment over 32 days. The reporter classified
97.8% as junk; that included extraction noise, hallucinations and duplicates.
It is not a vendor-wide duplicate rate, a Mem0-authored benchmark, or evidence
that projection hashing alone solves the reported extraction problems. The
issue was closed on June 5, 2026.

## Rollback

Reverting this code requires no schema rollback. Existing merge operations
retain their normal undo payloads. Review any queued plans before returning to
an older binary whose automatic-match rules are less conservative.
