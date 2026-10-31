"""Prepare LongMemEval-S session writes/questions with evaluator-only oracle data."""

import argparse
import json
from collections import Counter
from pathlib import Path

from campaigns.memoryagentbench.common import (
    read_rows,
    sha,
    source_identity,
    write_packet,
)

REVISION = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
DATA_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
FILENAME = "longmemeval_s_cleaned.json"
EXPECTED_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
DEFAULT_UPSTREAM = Path(__file__).resolve().parents[2] / "upstream" / "longmemeval"
DEFAULT_DATA = Path(__file__).parent / "data" / FILENAME


def normalize(rows):
    history, questions, oracle = [], [], []
    source_ids = set()
    for index, row in enumerate(rows):
        source_id = row["question_id"]
        if not isinstance(source_id, str) or source_id in source_ids:
            raise ValueError("invalid or duplicate LongMemEval question_id")
        source_ids.add(source_id)
        qid, scope = f"lme-q{index:05d}", f"lme-case{index:05d}"
        sessions, dates, ids = (
            row["haystack_sessions"],
            row["haystack_dates"],
            row["haystack_session_ids"],
        )
        if not sessions or len(sessions) != len(dates) or len(sessions) != len(ids):
            raise ValueError("misaligned session identities or dates")
        annotations, id_map = [], {}
        for session_index, (session, date, sid) in enumerate(
            zip(sessions, dates, ids, strict=True)
        ):
            if not isinstance(date, str) or not date or not session:
                raise ValueError("session timestamps and turns are required")
            handle = f"session-{session_index:05d}"
            # The official cleaned S file has13 records with repeated source IDs.
            # Preserve every occurrence and score aliases as the same source ID.
            id_map.setdefault(sid, []).append(handle)
            turns = []
            for turn_index, turn in enumerate(session):
                if turn.get("role") not in ("user", "assistant") or not isinstance(
                    turn.get("content"), str
                ):
                    raise ValueError("invalid history turn")
                turns.append({"role": turn["role"], "content": turn["content"]})
                if turn.get("has_answer"):
                    annotations.append({"session": handle, "turn": turn_index})
            history.append(
                {
                    "scope": scope,
                    "session_id": handle,
                    "ordinal": session_index,
                    "timestamp": date,
                    "turns": turns,
                    "tags": [scope, handle],
                }
            )
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("question must contain the actual prompt text")
        questions.append(
            {
                "question_id": qid,
                "scope": scope,
                "question": row["question"],
                "question_date": row["question_date"],
            }
        )
        evidence = row.get("answer_session_ids", [])
        if not set(evidence) <= set(ids):
            raise ValueError("oracle session references are absent from history")
        oracle.append(
            {
                "question_id": qid,
                "source_question_id": source_id,
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row["question_type"],
                "question_date": row["question_date"],
                "abstention": source_id.endswith("_abs"),
                "answer_session_ids": evidence,
                "answer_runtime_sessions": [h for s in evidence for h in id_map[s]],
                "session_id_map": id_map,
                "answer_turns": annotations,
                "source_metadata": {
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "question",
                        "answer",
                        "haystack_sessions",
                        "haystack_dates",
                        "haystack_session_ids",
                    }
                },
            }
        )
    return history, questions, oracle


def prepare(upstream, output, input_path=DEFAULT_DATA, limit=None):
    identity = source_identity(
        upstream,
        REVISION,
        [
            "README.md",
            "LICENSE",
            "src/evaluation/evaluate_qa.py",
            "src/retrieval/eval_utils.py",
        ],
        "https://github.com/xiaowu0162/LongMemEval",
    )
    input_path = Path(input_path)
    actual = sha(input_path)
    official = input_path.name == FILENAME
    if official and actual != EXPECTED_SHA256:
        raise ValueError("official LongMemEval-S file checksum mismatch")
    rows = read_rows(input_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        # Predeclared stratification uses category/abstention annotations only
        # to select cases. Neither annotation enters solver prompts or writes.
        groups = [
            "knowledge-update",
            "abstention",
            "temporal-reasoning",
            "multi-session",
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
        ]
        chosen = set()
        for group in groups:
            for index, row in enumerate(rows):
                matches = (
                    row["question_id"].endswith("_abs")
                    if group == "abstention"
                    else row["question_type"] == group
                    and not row["question_id"].endswith("_abs")
                )
                if matches and index not in chosen and len(chosen) < limit:
                    chosen.add(index)
                    break
        for index in range(len(rows)):
            if len(chosen) >= limit:
                break
            chosen.add(index)
        selected = [row for index, row in enumerate(rows) if index in chosen]
    else:
        selected = rows
    history, questions, oracle = normalize(selected)
    return write_packet(
        output,
        history,
        questions,
        oracle,
        {
            "campaign": "longmemeval-s",
            "upstream": identity,
            "adapter_files": {
                str(p.relative_to(Path(__file__).resolve().parents[2])): sha(p)
                for p in [
                    *Path(__file__).parent.glob("*.py"),
                    Path(__file__).parent.parent / "memoryagentbench/common.py",
                ]
            },
            "dataset": {
                "repo": "xiaowu0162/longmemeval-cleaned",
                "revision": DATA_REVISION,
                "license": "MIT",
                "path": str(input_path.resolve()),
                "bytes": input_path.stat().st_size,
                "sha256": actual,
                "source_url": f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{DATA_REVISION}/{FILENAME}"
                if official
                else None,
                "official_full_file_verified": official,
                "records_available": len(rows),
                "selection": "predeclared one-per-type plus abstention, fill in source order; emit in source order"
                if limit
                else "all records",
                "limit": limit,
            },
            "evaluation": {
                "state": "REQUIRES_EXTERNAL_JUDGE",
                "judge_requests": 0,
                "prompt_source": "src/evaluation/evaluate_qa.py:get_anscheck_prompt",
                "categories": sorted({o["question_type"] for o in oracle}),
                "category_counts": dict(Counter(o["question_type"] for o in oracle)),
                "abstention_count": sum(o["abstention"] for o in oracle),
            },
            "semantics": {
                "timestamps": "original exact session/date strings; source order preserved",
                "supersession": "all earlier and later statements preserved; no gold-based overwrite/link",
                "duplicate_source_sessions": "all occurrences preserved under unique ordinal handles; evaluator alias groups preserve source-ID retrieval semantics",
                "abstention": "source _abs and evidence annotations retained only in oracle",
                "runtime_question_ids": "neutral indices conceal evaluator-only abstention suffix",
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.upstream, args.output, args.input, args.limit)))


if __name__ == "__main__":
    main()
