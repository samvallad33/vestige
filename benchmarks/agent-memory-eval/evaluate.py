#!/usr/bin/env python3
"""Deterministic retrieval-metric evaluator for Agent Memory Eval fixtures.

The program consumes only committed fixture/ranking JSON. It intentionally does
not load models, make network requests, or measure runtime behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_fixture(fixture_dir: Path, rankings_path: Path) -> dict[str, Any]:
    """Fail closed when committed fixture bytes differ from their manifest."""
    manifest_path = fixture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {"corpus.jsonl", "queries.jsonl"}
    expected_files = manifest.get("files", {})
    if not required.issubset(expected_files):
        raise ValueError("fixture manifest is missing required corpus/query hashes")
    if rankings_path.parent == fixture_dir:
        required.add(rankings_path.name)
    for filename in required:
        expected = expected_files.get(filename)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"fixture manifest has no SHA-256 for {filename}")
        actual = hashlib.sha256((fixture_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"fixture hash mismatch for {filename}: expected {expected}, got {actual}")
    return manifest


def validate_run_manifest(path: Path, fixture_manifest_sha256: str) -> dict[str, Any]:
    """Validate the minimum immutable evidence needed for a real candidate run."""
    manifest = json.loads(path.read_text())
    if manifest.get("spec_version") != "agent-memory-eval/v1":
        raise ValueError("run manifest has an unsupported spec_version")
    if manifest.get("fixture_manifest_sha256") != fixture_manifest_sha256:
        raise ValueError("run manifest does not pin the evaluated fixture manifest")

    profile = manifest.get("profile_manifest", {})
    required_profile = {
        "profile_id", "model_id", "immutable_model_revision",
        "verified_model_artifact_hashes", "runtime_backend", "embedding_dimension",
        "normalization_method", "document_encoding_template", "query_encoding_template",
        "maximum_token_limit", "chunking_strategy", "created_at", "status",
    }
    missing_profile = required_profile - set(profile)
    if missing_profile or not isinstance(profile.get("embedding_dimension"), int) or profile["embedding_dimension"] <= 0:
        raise ValueError(f"run manifest has incomplete profile contract: missing={sorted(missing_profile)}")
    hashes = profile["verified_model_artifact_hashes"]
    if not isinstance(hashes, dict) or not hashes or not all(isinstance(value, str) and len(value) == 64 for value in hashes.values()):
        raise ValueError("run manifest requires SHA-256 for every model/tokenizer artifact")

    for group in ("evaluator", "runtime", "device", "reproducibility", "artifacts"):
        if not isinstance(manifest.get(group), dict) or not manifest[group]:
            raise ValueError(f"run manifest is missing {group} evidence")
    source_hash = manifest["evaluator"].get("source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("run manifest requires a pinned evaluator SHA-256")
    for artifact_name, artifact in manifest["artifacts"].items():
        if not isinstance(artifact, dict) or not artifact.get("path") or not isinstance(artifact.get("sha256"), str) or len(artifact["sha256"]) != 64:
            raise ValueError(f"run manifest requires a path and SHA-256 for {artifact_name}")
    return manifest


def _dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(corpus: list[dict[str, Any]], queries: list[dict[str, Any]], rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    corpus_by_id = {row["memory_id"]: row for row in corpus}
    missing = {query["query_id"] for query in queries} - set(rankings)
    unexpected = set(rankings) - {query["query_id"] for query in queries}
    if missing or unexpected:
        raise ValueError(f"ranking query IDs mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    aggregates: dict[str, dict[str, int | float]] = defaultdict(lambda: defaultdict(int))
    failures: list[dict[str, Any]] = []
    for query in queries:
        query_id = query["query_id"]
        category = query["category"]
        result_ids = [item["memory_id"] for item in rankings[query_id]]
        unknown = set(result_ids) - set(corpus_by_id)
        if unknown:
            raise ValueError(f"{query_id} ranks unknown memory IDs: {sorted(unknown)}")
        if len(result_ids) != len(set(result_ids)):
            raise ValueError(f"{query_id} ranks a memory more than once")

        relevance = query["relevance"]
        forbidden = set(query.get("forbidden_memory_ids", []))
        metrics = aggregates[category]
        metrics["queries"] += 1
        for k in (5, 10):
            top_k = result_ids[:k]
            metrics[f"recall_hits_at_{k}"] += int(any(memory_id in relevance for memory_id in top_k))
        received_relevance = [int(relevance.get(memory_id, 0)) for memory_id in result_ids[:10]]
        ideal_relevance = sorted((int(value) for value in relevance.values()), reverse=True)[:10]
        metrics["ndcg_sum_at_10"] += _dcg(received_relevance) / _dcg(ideal_relevance) if _dcg(ideal_relevance) else 0.0

        expected_literals = query.get("expected_literals", [])
        if expected_literals:
            metrics["exact_queries"] += 1
            exact_hit = any(
                memory_id in relevance
                and all(literal in corpus_by_id[memory_id]["content"] for literal in expected_literals)
                for memory_id in result_ids[:5]
            )
            metrics["exact_hits_at_5"] += int(exact_hit)
        top_five = result_ids[:5]
        metrics["top_five_positions"] += len(top_five)
        metrics["forbidden_hits_at_5"] += sum(memory_id in forbidden for memory_id in top_five)
        if category == "duplicate_near_miss":
            metrics["duplicate_positions"] += len(top_five)
            metrics["duplicate_confuser_hits_at_5"] += sum(memory_id in forbidden for memory_id in top_five)

        query_failures = []
        if not any(memory_id in relevance for memory_id in result_ids[:5]):
            query_failures.append("no_relevant_result_at_5")
        if any(memory_id in forbidden for memory_id in top_five):
            query_failures.append("forbidden_result_at_5")
        if expected_literals and not any(
            memory_id in relevance and all(literal in corpus_by_id[memory_id]["content"] for literal in expected_literals)
            for memory_id in top_five
        ):
            query_failures.append("exact_literal_not_preserved_at_5")
        if query_failures:
            failures.append({"query_id": query_id, "category": category, "failures": query_failures, "top_five": top_five})

    def summarize(values: dict[str, int | float]) -> dict[str, float | int | None]:
        queries_count = int(values["queries"])
        return {
            "queries": queries_count,
            "recall_at_5": _ratio(int(values["recall_hits_at_5"]), queries_count),
            "recall_at_10": _ratio(int(values["recall_hits_at_10"]), queries_count),
            "ndcg_at_10": float(values["ndcg_sum_at_10"]) / queries_count if queries_count else None,
            "exact_match_preservation_at_5": _ratio(int(values["exact_hits_at_5"]), int(values["exact_queries"])),
            "false_positive_retrieval_rate_at_5": _ratio(int(values["forbidden_hits_at_5"]), int(values["top_five_positions"])),
            "duplicate_near_miss_retrieval_rate_at_5": _ratio(int(values["duplicate_confuser_hits_at_5"]), int(values["duplicate_positions"])),
        }

    categories = {name: summarize(values) for name, values in sorted(aggregates.items())}
    overall: dict[str, int | float] = defaultdict(int)
    for values in aggregates.values():
        for key, value in values.items():
            overall[key] += value
    return {
        "spec_version": "agent-memory-eval/v1",
        "scope": "retrieval metrics only; operational metrics require a profile runner artifact",
        "overall": summarize(overall),
        "by_category": categories,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, help="Required evidence manifest for a real candidate run")
    args = parser.parse_args()
    fixture_manifest = verify_fixture(args.fixture_dir, args.rankings)
    report = evaluate(
        _read_jsonl(args.fixture_dir / "corpus.jsonl"),
        _read_jsonl(args.fixture_dir / "queries.jsonl"),
        json.loads(args.rankings.read_text()),
    )
    report["fixture_version"] = fixture_manifest["fixture_version"]
    report["fixture_manifest_sha256"] = hashlib.sha256((args.fixture_dir / "manifest.json").read_bytes()).hexdigest()
    if args.run_manifest:
        validate_run_manifest(args.run_manifest, report["fixture_manifest_sha256"])
        report["run_manifest_sha256"] = hashlib.sha256(args.run_manifest.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
