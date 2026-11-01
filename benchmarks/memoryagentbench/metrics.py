"""Primary upstream MAB metrics and judge-payload preparation; never inference."""

import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

from campaigns.memoryagentbench.common import (
    check_predictions,
    constants,
    extract_functions,
    jsonl,
    read_json,
    sha,
    source_identity,
)
from campaigns.memoryagentbench.prepare import DEFAULT_DATA, DEFAULT_UPSTREAM, REVISION

PURE_NAMES = [
    "normalize_answer",
    "f1_score",
    "drqa_exact_match_score",
    "substring_exact_match_score",
    "drqa_metric_max_over_ground_truths",
    "parse_output",
]


def _upstream(upstream):
    return source_identity(
        upstream,
        REVISION,
        [
            "utils/eval_other_utils.py",
            "llm_based_eval/longmem_qa_evaluate.py",
            "llm_based_eval/summarization_evaluate.py",
        ],
        "https://github.com/HUST-AI-HYZ/MemoryAgentBench",
    )


def primary_score(reference, hypothesis, upstream=DEFAULT_UPSTREAM):
    """Same normalization/parsing/maximum-over-gold used upstream for EM metrics."""
    _upstream(upstream)
    metric, answers = reference["metric"], reference["answers"]
    if metric not in ("exact_match", "substring_exact_match"):
        raise ValueError("primary_score requires a deterministic EM task")
    functions = extract_functions(
        Path(upstream) / "utils/eval_other_utils.py",
        PURE_NAMES,
        {"re": re, "string": string, "Counter": Counter},
    )
    parsed = functions["parse_output"](hypothesis)
    candidates = [parsed] if "icl_" in reference["source"] else [hypothesis, parsed]
    metric_fn = functions[
        "drqa_exact_match_score"
        if metric == "exact_match"
        else "substring_exact_match_score"
    ]
    score = max(
        functions["drqa_metric_max_over_ground_truths"](metric_fn, prediction, answers)
        for prediction in candidates
        if prediction is not None
    )
    return {
        "status": "SCORED_DETERMINISTIC",
        "metric": metric,
        "score": bool(score),
        "parsed_output": parsed,
    }


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


class _Argmin:
    @staticmethod
    def argmin(values):
        return min(range(len(values)), key=values.__getitem__)


def recsys_score(
    reference,
    hypothesis,
    upstream=DEFAULT_UPSTREAM,
    entity_map=DEFAULT_DATA / "entity2id.json",
):
    _upstream(upstream)
    # Upstream list(set(...)) determines tie order. Do not silently replace it.
    if os.environ.get("PYTHONHASHSEED") != "0":
        return {
            "status": "REQUIRES_PINNED_HASHSEED",
            "metric": "Recall@5",
            "score": None,
            "reason": "launch Python with PYTHONHASHSEED=0 to pin upstream fuzzy-match tie ordering",
        }
    functions = extract_functions(
        Path(upstream) / "utils/eval_other_utils.py",
        [
            "clean_text_elements",
            "clean_parentheses",
            "normalize_whitespace",
            "extract_movie_name",
            "find_nearest_movie",
            "extract_recommendation_list",
        ],
        {"re": re, "edit_distance": _edit_distance, "np": _Argmin},
    )
    mapping = read_json(entity_map)
    id_to_name = {
        value: functions["extract_movie_name"](key) for key, value in mapping.items()
    }
    predicted, _ = functions["extract_recommendation_list"](
        hypothesis, list(id_to_name.values())
    )
    movies = [row["nearest_movie"] for row in predicted]
    gold = [id_to_name[int(value.strip())] for value in reference["answers"]]
    return {
        "status": "SCORED_DETERMINISTIC",
        "metric": "Recall@5",
        "score": sum(movie in movies[:5] for movie in gold) / len(gold),
        "parsed_movies": movies,
        "entity_map_sha256": sha(entity_map),
        "primitive_adapter": "stdlib Levenshtein/argmin equivalent to upstream editdistance.eval/np.argmin",
    }


def judge_payloads(reference, hypothesis, upstream=DEFAULT_UPSTREAM):
    _upstream(upstream)
    question = reference["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("actual question text is required")
    if reference["metric"] == "llm_judge_accuracy":
        source = Path(upstream) / "llm_based_eval/longmem_qa_evaluate.py"
        fn = extract_functions(source, ["get_anscheck_prompt"])["get_anscheck_prompt"]
        answer = reference["answers"][0]
        if not reference.get("question_type"):
            raise ValueError("LongMemEval question_type required for original judge")
        return [
            {
                "kind": "accuracy",
                "question_id": reference["question_id"],
                "question": question,
                "status": "JUDGE_NOT_EXECUTED",
                "budget_usd": 0,
                "template_sha256": sha(source),
                "messages": [
                    {
                        "role": "user",
                        "content": fn(
                            reference["question_type"],
                            question,
                            answer,
                            hypothesis,
                            abstention=reference.get("abstention", False),
                        ),
                    }
                ],
            }
        ]
    if reference["metric"] != "llm_judge_summary_f1":
        raise ValueError("not a judge-scored task")
    source = Path(upstream) / "llm_based_eval/summarization_evaluate.py"
    prompts = constants(
        source, ["fluency_prompt_book", "recall_prompt_book", "precision_prompt_book"]
    )
    keypoints = reference["source_metadata"].get("keypoints", [])
    summary = hypothesis.strip()
    rendered = {
        "fluency": prompts["fluency_prompt_book"].format(text=summary),
        "recall": prompts["recall_prompt_book"].format(
            keypoints="\n".join(f"{i + 1}. {k}" for i, k in enumerate(keypoints)),
            summary=summary,
        ),
        "precision": prompts["precision_prompt_book"].format(
            expert_summary=reference["answers"][0], summary=summary
        ),
    }
    return [
        {
            "kind": kind,
            "question_id": reference["question_id"],
            "question": question,
            "status": "JUDGE_NOT_EXECUTED",
            "budget_usd": 0,
            "template_sha256": sha(source),
            "adaptation": "actual source question prepended to original upstream rubric",
            "messages": [
                {"role": "user", "content": "Question: " + question + "\n\n" + prompt}
            ],
        }
        for kind, prompt in rendered.items()
    ]


def score_supplied_judge_outputs(reference, outputs, *, expected_model):
    """Calculate official primary math only from explicit supplied judge records.

    This validates evidence shape, not provider authenticity. A launch-layer
    signed/metered transport packet must establish provenance separately.
    """
    required = (
        {"accuracy"}
        if reference["metric"] == "llm_judge_accuracy"
        else {"fluency", "recall", "precision"}
    )
    by_kind = {}
    for output in outputs:
        kind = output.get("kind")
        raw = output.get("raw_judge_response")
        request_hash = output.get("request_sha256", "")
        if (
            kind not in required
            or kind in by_kind
            or output.get("model") != expected_model
            or not isinstance(raw, str)
            or not re.fullmatch("[0-9a-f]{64}", request_hash)
        ):
            raise ValueError(
                "judge identity, raw response and question-bearing request digest required"
            )
        by_kind[kind] = raw
    if set(by_kind) != required:
        return {
            "status": "JUDGE_EVIDENCE_INCOMPLETE",
            "score": None,
            "missing": sorted(required - set(by_kind)),
        }
    if required == {"accuracy"}:
        if by_kind["accuracy"].strip().lower() not in {"yes", "no"}:
            raise ValueError("strict actual yes/no judge response required")
        score = by_kind["accuracy"].strip().lower() == "yes"
    else:
        values = {kind: json.loads(raw) for kind, raw in by_kind.items()}
        fluency, recall, precision = (
            values["fluency"]["fluency"],
            values["recall"]["recall"],
            values["precision"]["precision"],
        )
        sentences = values["precision"]["sentence_count"]
        total = len(reference["source_metadata"].get("keypoints", []))
        if (
            any(type(v) is not int for v in (fluency, recall, precision, sentences))
            or fluency not in (0, 1)
            or not 0 <= recall <= total
            or not 0 <= precision <= sentences
        ):
            raise ValueError("invalid or out-of-bound judge counts")
        rec, prec = (
            recall / total if total else 0,
            precision / sentences if sentences else 0,
        )
        score = fluency * 2 * rec * prec / (rec + prec) if rec + prec else 0
    return {
        "status": "SCORED_SUPPLIED_JUDGE_EVIDENCE",
        "score": score,
        "metric": reference["metric"],
        "judge_model": expected_model,
        "provenance": "supplied raw evidence; provider transport not verified by this pure adapter",
    }


def evaluate_predictions(
    oracle,
    predictions,
    upstream=DEFAULT_UPSTREAM,
    entity_map=DEFAULT_DATA / "entity2id.json",
):
    identity = _upstream(upstream)
    hypotheses = check_predictions(predictions)
    references = {row["question_id"]: row for row in oracle}
    if not set(hypotheses) <= set(references):
        raise ValueError("unknown prediction IDs")
    scored, pending = [], []
    for qid, hypothesis in hypotheses.items():
        reference = references[qid]
        if reference["metric"].startswith("llm_judge"):
            pending.extend(judge_payloads(reference, hypothesis, upstream))
            continue
        result = (
            recsys_score(reference, hypothesis, upstream, entity_map)
            if reference["metric"] == "Recall@5"
            else primary_score(reference, hypothesis, upstream)
        )
        scored.append({"question_id": qid, "source": reference["source"], **result})
    return {
        "status": "PARTIAL_SCORING_JUDGE_PENDING"
        if pending
        else "DETERMINISTIC_SCORING_ONLY",
        "upstream": identity,
        "results": scored,
        "judge_payloads": pending,
        "judge_requests": 0,
        "missing_predictions": len(references) - len(hypotheses),
        "scored": sum(row["status"] == "SCORED_DETERMINISTIC" for row in scored),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--entity-map", type=Path, default=DEFAULT_DATA / "entity2id.json"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_predictions(
        jsonl(args.oracle), jsonl(args.predictions), args.upstream, args.entity_map
    )
    with args.output.open("x") as stream:
        json.dump(result, stream)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scored": result["scored"],
                "judge_payloads": len(result["judge_payloads"]),
                "provider_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
