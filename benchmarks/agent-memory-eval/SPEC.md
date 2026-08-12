# Vestige Agent Memory Eval Specification v1

**Status:** evaluation contract; fixture set is a smoke-test corpus, not a
benchmark result.
**Scope:** profile-isolated semantic retrieval for local Vestige memory.

## Purpose

The evaluation prevents an embedding profile from being called an upgrade based
on generic public scores or an unrepeatable local impression. It tests whether
the profile retrieves the evidence an agent actually needs while preserving
exact identifiers and project boundaries.

This specification does not define a winner or publish a model-quality claim.
Thresholds for a release are declared in a signed release-candidate gate before
candidate results are reviewed; fixtures alone are insufficient for release
approval.

## Versioning and pins

An evaluation is identified by all of the following:

| Field | Requirement |
| --- | --- |
| `spec_version` | Exactly `agent-memory-eval/v1` |
| fixture manifest | Committed corpus/query bytes and SHA-256 hashes |
| evaluator | Committed evaluator source SHA-256 plus invocation |
| profile manifest | Exact immutable profile contract used for documents and queries |
| model/tokenizer artifacts | Immutable revision and SHA-256 for every loaded artifact |
| runtime | Backend/build version, OS, architecture, accelerator, and relevant driver/runtime versions |
| device | CPU, RAM, available storage, and accelerator details measured at run time |
| raw outputs | Full ordered result IDs and scores for every query, without top-k truncation |
| measurements | Raw latency, startup, ingestion, disk, download, migration, and rollback observations |

`run-manifest.example.json` shows the required shape. Placeholder revisions,
unpinned branches, mutable tags without a resolved revision, missing artifact
hashes, or partial raw rankings invalidate a run.

## Corpus and query categories

The full release corpus must contain held-out examples in every category below.
The small committed fixture has one deterministic example per category so the
harness can be tested without model downloads.

| Category | What it protects |
| --- | --- |
| `code_configuration` | Code facts, configuration values, and behavior-bound context |
| `exact_identifier` | Paths, UUIDs, environment variables, and code identifiers without semantic distortion |
| `incident_evidence` | Backfill-style upstream investigation leads; retrieval is not causal proof |
| `decision_reversal` | A current decision and its explicit supersession |
| `temporal_supersession` | Time-qualified facts and the later source of truth |
| `project_boundary` | Similar facts from different projects must not bleed together |
| `multilingual` | Same or relevant evidence across supported languages |
| `duplicate_near_miss` | Near duplicates/confusers must not displace the required evidence |
| `semantic_recall` | Ordinary paraphrased conceptual recall |

Each query records graded relevance, expected exact literals where applicable,
and explicitly forbidden near-miss IDs. Release corpora must be pinned,
license-reviewed, de-identified or synthetically authored, and held out from
template/model selection. Any change to corpus, labels, or evaluator creates a
new fixture version.

## Comparison matrix

Every candidate report compares independently migrated, profile-isolated runs:

1. legacy Nomic raw-text profile;
2. Nomic retrieval-prefix profile;
3. Qwen 0.6B at the same selected dimension;
4. Qwen 0.6B at each selected higher optional dimension; and
5. Qwen 4B at each selected dimension.

The exact set of Qwen dimensions and query instruction is selected by a
pre-registered evaluation plan, then frozen in the profile manifests. No report
may call a configuration equivalent to another configuration merely because the
model family matches. All hybrid-search weights, chunking, corpus order,
candidate limits, warm-up rules, and random seeds are pinned per run.

## Required metrics

Report each metric overall and by category, with numerator, denominator, and
the raw query IDs behind every failure.

| Metric | Definition |
| --- | --- |
| Recall@5 / Recall@10 | Fraction of queries with at least one relevance-positive memory in the first k results |
| nDCG@10 | Discounted cumulative gain at 10 using committed graded relevance labels |
| Exact-match preservation@5 | Fraction of exact queries whose first five results contain a relevant memory containing every expected literal byte-for-byte |
| False-positive retrieval rate@5 | Forbidden results divided by all inspected top-five result positions |
| Duplicate-near-miss retrieval rate@5 | Duplicate/confuser results divided by duplicate-safety top-five positions |
| Query latency p50 / p95 | End-to-end local query time, separated into cold and warm populations |
| Ingestion throughput | Documents successfully embedded/indexed per unit time, with corpus size and concurrency |
| Cold / warm startup | Time to usable profile runtime under declared cache state |
| RAM / disk footprint | Peak resident memory plus model, vector, index, and database bytes separately |
| Download size | Bytes of every model/tokenizer/runtime artifact actually fetched; zero for offline-install runs |
| Migration success | Completed / attempted memories and runs, including retry/repair failures |
| Rollback integrity | Pointer-change latency plus post-rollback profile/index/count validation |

The harness computes the retrieval metrics from raw rankings. The runner that
owns a real profile supplies operational measurements; its sampling method,
clock source, warm-up count, and outlier handling belong in the run manifest.

## Evaluation procedure

1. Copy the immutable fixture directory; verify its manifest hashes.
2. Record a candidate profile manifest before encoding data. Resolve every
   model and tokenizer revision to immutable identifiers and verify artifacts.
3. Create isolated profile storage from the same corpus snapshot. Do not query
   or score across profile indexes.
4. Encode documents with that profile's document template, then queries with
   its query template. Preserve raw ranked IDs and scores for every query.
5. Run the evaluator; retain the input fixture, raw rankings, report, run
   manifest, measurements, validation receipts, and failures together.
6. Run migration interruption/restart/repair and rollback scenarios separately.
   Their evidence is required operational data, not inferred from retrieval
   metrics.
7. Publish a comparative conclusion only when all artifacts are reviewable and
   the preregistered gate has been evaluated without substitutions.

## Release gate template

Before observing candidate results, the release owner records:

```text
fixture version and SHA-256:
candidate profile IDs and manifest SHA-256 values:
comparison baseline/profile manifest SHA-256:
selected dimensions, templates, chunker, fusion/reranker settings, and seeds:
minimum absolute metric thresholds by category:
maximum permitted regression versus Nomic retrieval profile by category:
maximum permitted false-positive and duplicate-near-miss rates:
operational limits for latency, RAM, disk, download, migration, and rollback:
hardware/build matrix:
required privacy, artifact-verification, crash, purge, and no-cross-profile tests:
reviewer and timestamp before result inspection:
```

A gate fails closed if its preregistration is absent, an artifact is missing,
or a required category has no held-out query. No model-performance number may be
used in product copy without the complete retained artifact set.
