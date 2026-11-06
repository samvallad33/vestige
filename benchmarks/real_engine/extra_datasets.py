"""Official MemConflict data and scorer bridge; no inference or local scorer port.

The source annotations stay in ``scorer_payload``. Each solver record contains
only the observed, dated dialogue through the question's own session. The
upstream judge prompt/parser are imported byte-for-byte from the pinned source;
the caller supplies the shared, budgeted judge callback.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

MEMCONFLICT_REVISION = "ec51d5d36e87f7665d1337f3a88cbde95fc2a964"
MEMCONFLICT_DATA_SHA256 = "8ef9ec8589eccb86f63ab3a819a9180217405351a8d5846866721ea74babe092"
MEMCONFLICT_DATA_BYTES = 39671712
MEMCONFLICT_SCORER_SHA256 = "dea71ccb9058a5652ccf267143630851c7deffefeee5dde4c3b86386875d63f2"
MEMCONFLICT_PARSER_SHA256 = "01142a2b9d40c5456ac0e883324b7e6dc77872af0ea6b9d767d0689ae30eaca3"
MEMCONFLICT_QUESTIONS = 3750
PMBENCH_REVISION = "e1093c470c8981daf522d4ef047a7c3a71e077d7"
PMBENCH_DATA_SHA256 = "94a45937da1363be19ccfdc2c188d132f23093041e30abd3ec22d64d70da8f24"
PMBENCH_SCORER_SHA256 = "d8ec27d8dcf4679d7a789c52fc305286df460844d47e9f116b81f2400ac254d8"


def _repo(root: str | Path) -> Path:
    root = Path(root)
    return root if (root / "Data" / "Step4_4.jsonl").is_file() else root / "memconflict"


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_memconflict(root: str | Path) -> dict[str, Any]:
    """Refuse a changed dataset or scorer, even at the expected Git revision."""
    repo = _repo(root)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MEMCONFLICT_REVISION:
        raise ValueError(f"MemConflict revision mismatch: {revision}")
    files = {
        "Data/Step4_4.jsonl": MEMCONFLICT_DATA_SHA256,
        "Evaluation/eval_scoring.py": MEMCONFLICT_SCORER_SHA256,
        "Evaluation/llm_request.py": MEMCONFLICT_PARSER_SHA256,
    }
    for relative, expected in files.items():
        if _digest(repo / relative) != expected:
            raise ValueError(f"MemConflict hash mismatch: {relative}")
    if (repo / "Data/Step4_4.jsonl").stat().st_size != MEMCONFLICT_DATA_BYTES:
        raise ValueError("MemConflict data byte count mismatch")
    return {"revision": revision, "dataset_sha256": MEMCONFLICT_DATA_SHA256,
            "scorer_sha256": MEMCONFLICT_SCORER_SHA256, "files": files}


def _turn_number(key: str) -> int:
    # Upstream dialogue_turn_N fields are numeric sequence positions.
    return int(key.rsplit("_", 1)[1])


def memconflict_records(personas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split official sessions at their real temporal boundaries, with no gold leak.

    This pure conversion is separate so chronology and annotation isolation can
    be regression-tested without downloading a corpus or calling a model.
    """
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for persona in personas:
        persona_id = str(persona["ID"])
        history: list[str] = []
        seen_sessions: set[int] = set()
        for session in sorted(persona["Full_Session_Chain"], key=lambda s: int(s["Session_ID"])):
            session_id = int(session["Session_ID"])
            if session_id in seen_sessions:
                raise ValueError(f"duplicate MemConflict session: {persona_id}/{session_id}")
            seen_sessions.add(session_id)
            date = str(session["Date"])
            dialogue = session.get("Session_Dialogue") or {}
            for key in sorted(dialogue, key=_turn_number):
                turn = dialogue[key]
                if not isinstance(turn, list):
                    continue
                for message in turn:
                    # Exactly the validity rule in the pinned official
                    # Evaluation/eval_memzero.py Build_Session_Dialogue_List.
                    # Two released messages have no role; do not invent one.
                    if not isinstance(message, dict):
                        continue
                    role, content = message.get("role"), message.get("content")
                    if role not in ["user", "assistant"] or content in [None, ""]:
                        continue
                    content = str(content)
                    history.append(f"[{date}] {role}: {content}")
            for position, question in enumerate(session.get("Session_Questions") or []):
                qid = str(question.get("question_id", position))
                identity = f"memconflict:{persona_id}:s{session_id}:{qid}"
                if identity in seen:
                    raise ValueError(f"duplicate MemConflict question: {identity}")
                seen.add(identity)
                output.append({
                    "id": identity,
                    "group_id": f"memconflict:{persona_id}:s{session_id}",
                    "history": list(history),
                    "question": str(question["question"]),
                    "category": str(question["conflict_type"]),
                    "scorer_payload": {"question": dict(question), "persona_id": persona_id,
                                       "session_id": session_id, "date": date},
                })
    return output


def load_memconflict(root: str | Path) -> list[Any]:
    """Load all 3,750 official questions using the existing, unchanged data hash."""
    from .datasets import BenchmarkItem

    verify_memconflict(root)
    with (_repo(root) / "Data/Step4_4.jsonl").open(encoding="utf-8-sig") as f:
        personas = [json.loads(line) for line in f if line.strip()]
    if len(personas) != 30:
        raise ValueError(f"expected 30 MemConflict personas, got {len(personas)}")
    records = memconflict_records(personas)
    if len(records) != MEMCONFLICT_QUESTIONS:
        raise ValueError(f"expected 3750 MemConflict questions, got {len(records)}")
    return [BenchmarkItem(**record) for record in records]


def _import_file(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def memconflict_scorer(root: str | Path) -> tuple[Any, Any]:
    """Import the exact official judge/parser, never its provider request path."""
    verify_memconflict(root)
    directory = _repo(root) / "Evaluation"
    previous = list(sys.path)
    # eval_scoring imports a sibling llm_request, so bind that exact pinned file
    # temporarily to prevent a different benchmark's same-named helper winning.
    old_request = sys.modules.get("llm_request")
    try:
        sys.path.insert(0, str(directory))
        parser = _import_file(directory / "llm_request.py", "_real_memconflict_parser")
        sys.modules["llm_request"] = parser
        scorer = _import_file(directory / "eval_scoring.py", "_real_memconflict_scorer")
    finally:
        sys.path[:] = previous
        if old_request is None:
            sys.modules.pop("llm_request", None)
        else:
            sys.modules["llm_request"] = old_request
    return scorer, parser


def memconflict_judge_prompt(root: str | Path, item: Any, response: str,
                             retrieved_memories: Sequence[dict[str, Any]] = ()) -> dict[str, str]:
    scorer, _ = memconflict_scorer(root)
    question = item.scorer_payload["question"]
    normalized = scorer.extract_top_k_retrieved_memories({"Retrieved_Memories": list(retrieved_memories)})
    return {"system": scorer.LLM_JUDGE_SYSTEM_PROMPT,
            "user": scorer.build_llm_judge_prompt(question, response, normalized)}


async def judge_memconflict(
    root: str | Path, items: Sequence[Any], predictions: Mapping[str, str],
    judge_callback: Callable[..., Any],
    retrieved: Mapping[str, Sequence[dict[str, Any]]] | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Score one strictly merged set through the unmodified official nano judge.

    Callback receives a dict with system/user strings and item_id. It must use
    the campaign provider/budget and return the response string, or a dict with
    response and optional usage/cost. Exceptions and malformed JSON propagate;
    there is deliberately no fallback metric or silent zero.
    """
    expected = [item.id for item in items]
    if len(set(expected)) != len(expected) or set(predictions) != set(expected):
        raise ValueError("MemConflict scoring requires unique exact merged item coverage")
    scorer, parser = memconflict_scorer(root)
    semaphore = asyncio.Semaphore(40)

    async def one(item: Any) -> tuple[float, dict[str, Any]]:
        question = item.scorer_payload["question"]
        memories = scorer.extract_top_k_retrieved_memories(
            {"Retrieved_Memories": list((retrieved or {}).get(item.id, []))}
        )
        user_prompt = scorer.build_llm_judge_prompt(question, predictions[item.id], memories)
        payload = {"system": scorer.LLM_JUDGE_SYSTEM_PROMPT, "user": user_prompt, "item_id": item.id}
        async with semaphore:
            raw = judge_callback(payload)
            raw = await raw if inspect.isawaitable(raw) else raw
        answer = raw["response"] if isinstance(raw, dict) else raw
        parsed = parser._extract_json_from_content(
            answer, ["Final JSON", "JSON Result", "Evaluation Result", "Result"]
        )
        result = scorer.parse_llm_metric_result(question["conflict_type"], parsed, "model_response")
        metric = {"dynamic_conflict": "dynamic_answer_accuracy",
                  "static_conflict": "static_answer_accuracy",
                  "conditional_conflict": "conditional_answer_accuracy"}[question["conflict_type"]]
        return float(result["Metrics"][metric]), {
            "item_id": item.id, "judge_model": "gpt-5.4-nano",
            "prompt": payload, "response": answer, "official_result": result,
            "provider": raw if isinstance(raw, dict) else None,
            "white_box_note": "Only supplied actual retrieval traces are scored; no gold retrieval synthesized.",
        }

    results = await asyncio.gather(*(one(item) for item in items))
    return [score for score, _ in results], [log for _, log in results]


def lifecycle_constraints() -> dict[str, str]:
    """Explicit limits: these official tasks require their own live runner."""
    return {
        "pm_bench": "Full seven-day simulator with hidden channels and delayed actions; use official sim/pm_bench.py score_log on its trajectory, not flattened QA.",
        "memform": "Paired clean/poisoned memory-building interventions across P1-P5; use official run_case and aggregate_results with normal real-engine writes, not oracle_memory.",
        "memoryagentbench": "Four official competency splits have different continual-learning and scoring protocols; generic flattened QA alone does not establish protocol parity.",
    }


def _pmbench_repo(root: str | Path) -> Path:
    root = Path(root)
    return root if (root / "sim/pm_bench.py").is_file() else root / "pm-bench"


def load_pmbench_scenario(root: str | Path) -> dict[str, Any]:
    """Verify the whole official week; caller must keep simulator labels private.

    This is a simulator input, NOT a BenchmarkItem or a solver prompt. Feeding
    the whole scenario to a model would expose hidden state and score labels.
    """
    repo = _pmbench_repo(root)
    rev = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if rev != PMBENCH_REVISION:
        raise ValueError(f"PMBench revision mismatch: {rev}")
    for rel, expected in [("data/synthetic_week_v9.json", PMBENCH_DATA_SHA256),
                          ("sim/pm_bench.py", PMBENCH_SCORER_SHA256)]:
        if _digest(repo / rel) != expected:
            raise ValueError(f"PMBench hash mismatch: {rel}")
    scenario = json.loads((repo / "data/synthetic_week_v9.json").read_text())
    if len(scenario["days"]) != 7 or sum(len(day["steps"]) for day in scenario["days"]) != 80:
        raise ValueError("PMBench expected complete official seven-day, 80-step scenario")
    return scenario


def score_pmbench(root: str | Path, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Score one completed 80-step merged trajectory once with official score_log.

    Week trials must remain independent. Days/steps may not be run as isolated
    shards because the benchmark contains cross-day intentions and updates.
    """
    scenario = load_pmbench_scenario(root)
    expected = {(day["name"], step["id"]) for day in scenario["days"] for step in day["steps"]}
    keys = [(entry["day"], entry["step_id"]) for entry in entries]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("PMBench scoring requires unique complete 80-step trajectory coverage")
    scorer = _import_file(_pmbench_repo(root) / "sim/pm_bench.py", "_real_pmbench_scorer")
    summary, per_day, steps = scorer.score_log(scenario, list(entries))
    return {"summary": summary, "per_day": per_day, "steps": steps,
            "revision": PMBENCH_REVISION, "dataset_sha256": PMBENCH_DATA_SHA256,
            "scorer_sha256": PMBENCH_SCORER_SHA256}
