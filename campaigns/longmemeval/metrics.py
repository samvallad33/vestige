"""Offline exact upstream judge payloads and validated label aggregation; no API."""

import argparse
import json
from pathlib import Path

from campaigns.longmemeval.prepare import DEFAULT_UPSTREAM, REVISION
from campaigns.memoryagentbench.common import (
    check_predictions,
    extract_functions,
    jsonl,
    sha,
    source_identity,
)


def judge_payloads(oracle, predictions, upstream=DEFAULT_UPSTREAM):
    source_identity(
        upstream,
        REVISION,
        ["src/evaluation/evaluate_qa.py"],
        "https://github.com/xiaowu0162/LongMemEval",
    )
    predictions = check_predictions(predictions)
    references = {row["question_id"]: row for row in oracle}
    if not set(predictions) <= set(references):
        raise ValueError("unknown hypothesis question_id")
    path = Path(upstream) / "src/evaluation/evaluate_qa.py"
    prompt = extract_functions(path, ["get_anscheck_prompt"])["get_anscheck_prompt"]
    return [
        {
            "question_id": qid,
            "question": references[qid]["question"],
            "task": references[qid]["question_type"],
            "status": "JUDGE_NOT_EXECUTED",
            "budget_usd": 0,
            "template_sha256": sha(path),
            "messages": [
                {
                    "role": "user",
                    "content": prompt(
                        references[qid]["question_type"],
                        references[qid]["question"],
                        references[qid]["answer"],
                        hypothesis,
                        abstention=references[qid]["abstention"],
                    ),
                }
            ],
        }
        for qid, hypothesis in predictions.items()
    ]


def aggregate_judgements(oracle, judgements, *, expected_model):
    """Aggregate supplied actual judge evidence; never turn absence into a pass."""
    if not expected_model:
        raise ValueError("judge model identity required")
    by_id = {row["question_id"]: row for row in oracle}
    seen, groups = set(), {}
    for result in judgements:
        qid = result["question_id"]
        label = result.get("autoeval_label", {})
        if qid in seen or qid not in by_id or label.get("model") != expected_model:
            raise ValueError("duplicate/unknown judge identity or model mismatch")
        if type(label.get("label")) is not bool or result.get(
            "raw_judge_response", ""
        ).strip().lower() not in ("yes", "no"):
            raise ValueError("actual raw yes/no judge evidence required")
        if label["label"] != (result["raw_judge_response"].strip().lower() == "yes"):
            raise ValueError("judge label/raw verdict mismatch")
        if (
            not isinstance(result.get("request_sha256"), str)
            or len(result["request_sha256"]) != 64
        ):
            raise ValueError("question-bearing request digest required")
        seen.add(qid)
        for group in (
            "overall",
            by_id[qid]["question_type"],
            "abstention" if by_id[qid]["abstention"] else "answerable",
        ):
            groups.setdefault(group, []).append(label["label"])
    return {
        "status": "SCORED_SUPPLIED_JUDGE_EVIDENCE" if seen else "NO_JUDGE_EVIDENCE",
        "judge_model": expected_model,
        "scored": len(seen),
        "missing": len(by_id) - len(seen),
        "accuracy": {
            group: {"n": len(values), "accuracy": sum(values) / len(values)}
            for group, values in groups.items()
        },
    }


def retrieval_metrics(oracle_row, retrieved_sessions):
    if oracle_row["abstention"]:
        return {"status": "EXCLUDED_ABSTENTION", "recall_any": None, "recall_all": None}
    expected = set(oracle_row["answer_session_ids"])
    if not expected:
        return {"status": "NO_GOLD_LOCATION", "recall_any": None, "recall_all": None}
    aliases = {
        handle: sid
        for sid, handles in oracle_row["session_id_map"].items()
        for handle in handles
    }
    recovered = {
        aliases.get(handle, "unknown-runtime:" + handle)
        for handle in retrieved_sessions
    }
    return {
        "status": "SCORED_RETRIEVAL",
        "recall_any": float(bool(expected & recovered)),
        "recall_all": float(expected <= recovered),
        "precision": len(expected & recovered) / len(recovered) if recovered else 0.0,
        "recall": len(expected & recovered) / len(expected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payloads = judge_payloads(
        jsonl(args.oracle), jsonl(args.predictions), args.upstream
    )
    with args.output.open("x") as stream:
        for payload in payloads:
            stream.write(json.dumps(payload) + "\n")
    print(
        json.dumps(
            {
                "status": "JUDGE_PAYLOADS_ONLY",
                "count": len(payloads),
                "provider_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
