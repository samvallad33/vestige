"""Scorer-only imports: no model calls and no graph-derived reference labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import PublicTrajectory, ScoringReference, normalize_role


def score(
    public: PublicTrajectory,
    reference: ScoringReference,
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    if public.question_id != reference.question_id:
        raise ValueError("reference identity mismatch")
    prediction = prediction if isinstance(prediction, dict) else {}
    if prediction.get("question_ID", public.question_id) != public.question_id:
        raise ValueError("prediction identity mismatch")
    role = normalize_role(prediction.get("predicted_role"))
    step = prediction.get("predicted_step")
    # JSON integer or an unambiguous integral decimal string, never bool/float.
    if isinstance(step, str) and step.strip().isascii() and step.strip().isdigit():
        step = int(step.strip())
    valid = type(step) is int and 0 <= step < len(public.history)
    error = abs(step - reference.step) if valid else None
    return {
        "question_ID": public.question_id,
        "source": public.source,
        "role_correct": bool(role)
        and role in public.roles
        and role == normalize_role(reference.role),
        "role_valid": bool(role) and role in public.roles,
        "step_valid": valid,
        "root_exact": valid and error == 0,
        "root_within_5": valid and error <= 5,
        "root_absolute_error": error,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or len({row["question_ID"] for row in rows}) != len(rows):
        raise ValueError("nonempty unique expected-trajectory rows required")
    n = len(rows)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    means = {}
    for source, items in by_source.items():
        errors = [row["root_absolute_error"] for row in items if row["step_valid"]]
        means[source] = sum(errors) / len(errors) if errors else None
    missing = [source for source, mean in means.items() if mean is None]
    return {
        "n": n,
        "role_correct": sum(row["role_correct"] for row in rows),
        "root_exact_correct": sum(row["root_exact"] for row in rows),
        "root_within_5_correct": sum(row["root_within_5"] for row in rows),
        "role_accuracy": sum(row["role_correct"] for row in rows) / n,
        "root_exact_accuracy": sum(row["root_exact"] for row in rows) / n,
        "root_within_5_accuracy": sum(row["root_within_5"] for row in rows) / n,
        "valid_step_n": sum(row["step_valid"] for row in rows),
        "source_weighted_valid_root_mae": None
        if missing
        else sum(len(by_source[source]) * mean for source, mean in means.items()) / n,
        "mae_undefined_sources": missing,
        "source_valid_root_mae": means,
    }


def score_files(
    public_path: Path, reference_path: Path, prediction_path: Path
) -> dict[str, Any]:
    from .schema import validate_public

    def read(path):
        result = {}
        for line in path.read_text().splitlines():
            item = json.loads(line)
            qid = item["question_ID"]
            if qid in result:
                raise ValueError("duplicate trajectory identity")
            result[qid] = item
        return result

    public, references, predictions = (
        read(public_path),
        read(reference_path),
        read(prediction_path),
    )
    if set(public) != set(references) or set(predictions) - set(public):
        raise ValueError("scoring identity sets differ")
    rows = []
    for qid, item in public.items():
        ref = references[qid]
        reference = ScoringReference(
            qid, ref["mistake_agent"], ref["mistake_step"], ref["mistake_reason"]
        )
        rows.append(score(validate_public(item), reference, predictions.get(qid)))
    return {
        "metric_contract": "LongRCA v4 benchmark comparison, independent role/root scoring",
        "metrics": aggregate(rows),
        "per_question": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            score_files(args.public, args.references, args.predictions), indent=2
        )
        + "\n"
    )
