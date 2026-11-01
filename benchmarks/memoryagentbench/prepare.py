"""Normalize pinned MemoryAgentBench into incremental writes and isolated gold."""

import argparse
import json
from collections import Counter
from pathlib import Path

from campaigns.memoryagentbench.common import (
    constants,
    read_rows,
    sha,
    source_identity,
    write_packet,
)

REVISION = "538026089d1a8a8eff05121d0db89b388f360eba"
DATA_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
DEFAULT_UPSTREAM = Path(__file__).resolve().parents[2] / "upstream" / "memoryagentbench"
DEFAULT_DATA = Path(__file__).parent / "data"
EXPECTED = {
    "Accurate_Retrieval-00000-of-00001.parquet": "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",
    "Conflict_Resolution-00000-of-00001.parquet": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",
    "Long_Range_Understanding-00000-of-00001.parquet": "5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539",
    "Test_Time_Learning-00000-of-00001.parquet": "5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0",
}


def metric_name(source):
    if "longmemeval" in source:
        return "llm_judge_accuracy"
    if "infbench" in source:
        return "llm_judge_summary_f1"
    if "recsys" in source:
        return "Recall@5"
    if any(prefix in source for prefix in ("eventqa", "ruler", "factconsolidation")):
        return "substring_exact_match"
    if "icl_" in source or "detective" in source:
        return "exact_match"
    raise ValueError("unsupported source metric: " + source)


def template_group(source):
    for prefix, group in (
        ("ruler", "ruler_qa"),
        ("eventqa", "eventqa"),
        ("icl", "in_context_learning"),
        ("recsys", "recsys_redial"),
        ("infbench", "infbench_sum"),
        ("longmemeval", "longmemeval"),
        ("factconsolidation", "factconsolidation"),
        ("detective", "detective_qa"),
    ):
        if source.startswith(prefix):
            return group
    raise ValueError("no upstream template for source")


def normalize(rows, templates, *, chunk_chars=4096):
    if type(chunk_chars) is not int or chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")
    history, questions, oracle = [], [], []
    for context_index, row in enumerate(rows):
        context, prompts, answers, metadata = (
            row["context"],
            row["questions"],
            row["answers"],
            row["metadata"],
        )
        if (
            not isinstance(context, str)
            or not context
            or not isinstance(prompts, list)
            or not prompts
            or len(prompts) != len(answers)
        ):
            raise ValueError("context and question/answer alignment required")
        source = metadata["source"]
        metric = metric_name(source)
        scope = f"mab-context{context_index:05d}"
        for chunk_index, start in enumerate(range(0, len(context), chunk_chars)):
            history.append(
                {
                    "scope": scope,
                    "session_id": f"chunk-{chunk_index:05d}",
                    "ordinal": chunk_index,
                    "timestamp": None,
                    "content": context[start : start + chunk_chars],
                    "tags": [scope, f"chunk-{chunk_index:05d}"],
                }
            )
        for key in (
            "qa_pair_ids",
            "question_ids",
            "question_dates",
            "question_types",
            "previous_events",
        ):
            values = metadata.get(key)
            if values and len(values) != len(prompts):
                raise ValueError("misaligned source metadata: " + key)
        for query_index, (question, answer) in enumerate(
            zip(prompts, answers, strict=True)
        ):
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(answer, list)
                or not answer
            ):
                raise ValueError(
                    "actual question text and nonempty answer variants required"
                )
            qid = f"mab-c{context_index:05d}-q{query_index:05d}"
            template = templates[template_group(source)]["query"]["long_context_agent"]
            prompt = template.format(question=question)
            query = {
                "question_id": qid,
                "scope": scope,
                "question": question,
                "prompt": prompt,
            }
            if metadata.get("question_dates"):
                query["question_date"] = metadata["question_dates"][query_index]
            questions.append(query)
            indexed = {
                "qa_pair_ids",
                "question_ids",
                "question_dates",
                "question_types",
                "previous_events",
            }
            original = {
                k: (v[query_index] if k in indexed and v else v)
                for k, v in metadata.items()
                if k != "haystack_sessions"
            }
            oracle.append(
                {
                    "question_id": qid,
                    "question": question,
                    "answers": answer,
                    "source": source,
                    "metric": metric,
                    "source_metadata": original,
                    "qa_index": query_index,
                    "source_question_id": metadata.get("question_ids", [])[query_index]
                    if metadata.get("question_ids")
                    else None,
                    "question_type": metadata.get("question_types", [])[query_index]
                    if metadata.get("question_types")
                    else None,
                    "abstention": bool(
                        metadata.get("question_ids")
                        and metadata["question_ids"][query_index].endswith("_abs")
                    ),
                    "evidence_annotations": metadata["haystack_sessions"][query_index]
                    if metadata.get("haystack_sessions")
                    else None,
                }
            )
    return history, questions, oracle


def prepare(upstream, output, inputs=None, limit=None, chunk_chars=4096):
    identity = source_identity(
        upstream,
        REVISION,
        [
            "README.md",
            "LICENSE",
            "utils/templates.py",
            "utils/eval_other_utils.py",
            "utils/eval_data_utils.py",
            "llm_based_eval/longmem_qa_evaluate.py",
            "llm_based_eval/summarization_evaluate.py",
        ],
        "https://github.com/HUST-AI-HYZ/MemoryAgentBench",
    )
    paths = (
        [Path(p) for p in inputs] if inputs else sorted(DEFAULT_DATA.glob("*.parquet"))
    )
    if not paths:
        raise FileNotFoundError(
            "no input datasets; supply --input JSON/Parquet or downloaded official files"
        )
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive contexts per source")
    rows, files, source_counts = [], [], Counter()
    for path in paths:
        actual = sha(path)
        if path.name in EXPECTED and actual != EXPECTED[path.name]:
            raise ValueError("official dataset checksum mismatch: " + path.name)
        available = read_rows(path)
        files.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": actual,
                "source_url": f"https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/{DATA_REVISION}/data/{path.name}"
                if path.name in EXPECTED
                else None,
                "records_available": len(available),
                "official_full_file_verified": path.name in EXPECTED,
            }
        )
        for row in available:
            source = row["metadata"]["source"]
            if limit is None or source_counts[source] < limit:
                rows.append(row)
                source_counts[source] += 1
    templates = constants(Path(upstream) / "utils/templates.py", ["BASE_TEMPLATES"])[
        "BASE_TEMPLATES"
    ]
    history, questions, oracle = normalize(rows, templates, chunk_chars=chunk_chars)
    entity_map = DEFAULT_DATA / "entity2id.json"
    auxiliary = []
    if entity_map.exists():
        entity_hash = sha(entity_map)
        if (
            entity_hash
            != "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5"
        ):
            raise ValueError("official recommendation entity mapping checksum mismatch")
        auxiliary.append(
            {
                "path": str(entity_map.resolve()),
                "bytes": entity_map.stat().st_size,
                "sha256": entity_hash,
                "source_url": f"https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/{DATA_REVISION}/entity2id.json",
            }
        )
    return write_packet(
        output,
        history,
        questions,
        oracle,
        {
            "campaign": "memoryagentbench",
            "upstream": identity,
            "adapter_files": {
                str(p.relative_to(Path(__file__).resolve().parents[2])): sha(p)
                for p in Path(__file__).parent.glob("*.py")
            },
            "dataset": {
                "repo": "ai-hyz/MemoryAgentBench",
                "revision": DATA_REVISION,
                "license": "MIT",
                "files": files,
                "auxiliary_files": auxiliary,
                "selection": "first N contexts per metadata.source"
                if limit
                else "all supplied contexts",
                "limit_per_source": limit,
                "sources": dict(source_counts),
            },
            "evaluation": {
                "state": "DETERMINISTIC_METRICS_AVAILABLE_JUDGE_TASKS_PENDING",
                "judge_requests": 0,
                "metric_counts": dict(Counter(o["metric"] for o in oracle)),
            },
            "semantics": {
                "chunking": f"lossless contiguous Unicode character chunks of {chunk_chars}; differs from upstream token/sentence chunks",
                "timestamp": "no fabricated clock; source timestamps/serials stay in exact context; query dates retained where supplied",
                "supersession": "newer fact serial-number rule preserved in upstream template; all original context bytes retained",
                "prompt": "upstream long_context_agent template shared across all arms to avoid arm-specific instructions",
                "history_source": "only original context, never QA answers or evidence annotations",
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, help="contexts per source; omitted means all"
    )
    parser.add_argument("--chunk-chars", type=int, default=4096)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.upstream, args.output, args.input, args.limit, args.chunk_chars
            )
        )
    )


if __name__ == "__main__":
    main()
