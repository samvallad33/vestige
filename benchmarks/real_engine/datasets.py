"""Pinned official QA datasets. Gold annotations never enter solver_record()."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping


@dataclass
class BenchmarkItem:
    id: str
    group_id: str
    history: list[str]
    question: str
    category: str
    scorer_payload: dict[str, Any] = field(repr=False)

    def solver_record(self) -> dict[str, Any]:
        return {"id": self.id, "group_id": self.group_id, "history": self.history,
                "question": self.question, "category": self.category}


SOURCES = {
    "locomo": {
        "repository": "https://github.com/snap-research/locomo",
        "commit": "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
        "data_file": "data/locomo10.json",
        "data_sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "scorer_file": "task_eval/evaluation.py",
        "scorer_sha256": "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd",
        "questions": 1986, "conversations": 10,
        "categories": {"1": 282, "2": 321, "3": 96, "4": 841, "5": 446},
        "license_file": "LICENSE.txt",
    },
    "longmemeval": {
        "repository": "https://github.com/xiaowu0162/LongMemEval",
        "commit": "9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "dataset": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "dataset_revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "data_file": "data/longmemeval_s_cleaned.json",
        "data_sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "scorer_file": "src/evaluation/evaluate_qa.py",
        "scorer_sha256": "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
        "questions": 500, "license_file": "LICENSE",
        "judge_note": "Official get_anscheck_prompt and yes parsing; user-required gpt-5.4-nano replaces published GPT-4o judge through shared budgeted callback.",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(root: str | Path, name: str, kind: str) -> Path:
    meta = SOURCES[name]
    path = Path(root) / meta[f"{kind}_file"]
    actual = sha256(path)
    if actual != meta[f"{kind}_sha256"]:
        raise ValueError(f"{name} official {kind} hash mismatch: {actual}")
    return path


def _unique(items: list[BenchmarkItem], expected: int) -> list[BenchmarkItem]:
    if len(items) != expected or len({x.id for x in items}) != expected:
        raise ValueError(f"Expected {expected} distinct official items; found {len(items)}")
    return items


def load_locomo(root: str | Path) -> list[BenchmarkItem]:
    data = json.loads(_verify(root, "locomo", "data").read_text())
    if len(data) != 10:
        raise ValueError("LoCoMo requires all ten official conversations")
    result = []
    for sample in data:
        conv = sample["conversation"]
        sessions = sorted(int(k.split("_")[-1]) for k in conv
                          if k.startswith("session_") and not k.endswith("_date_time"))
        history = []
        for sid in sessions:
            date = conv[f"session_{sid}_date_time"]
            for turn in conv[f"session_{sid}"]:
                text = f'[{date}] {turn["speaker"]}: {turn["text"]}'
                if turn.get("blip_caption"):
                    text += f' [shared image caption: {turn["blip_caption"]}]'
                history.append(text)
        for idx, qa in enumerate(sample["qa"]):
            gold = dict(qa)
            # Upstream prediction preparation supplies this field; the official
            # category-five scorer reads it before testing abstention phrases.
            gold.setdefault("answer", "No information available")
            result.append(BenchmarkItem(
                id=f'locomo:{sample["sample_id"]}:{idx}', group_id=str(sample["sample_id"]),
                history=history, question=qa["question"], category=str(qa["category"]),
                scorer_payload=gold))
    return _unique(result, 1986)


def load_longmemeval(root: str | Path) -> list[BenchmarkItem]:
    data = json.loads(_verify(root, "longmemeval", "data").read_text())
    result = []
    for row in data:
        if len(row["haystack_dates"]) != len(row["haystack_sessions"]):
            raise ValueError(f'Unaligned dates for {row["question_id"]}')
        history = []
        for date, session in zip(row["haystack_dates"], row["haystack_sessions"]):
            # Whitelist role/content; has_answer and evidence/session IDs never
            # reach memory. Both sides of each conversation remain available.
            for turn in session:
                history.append(f'[{date}] {turn["role"]}: {turn["content"]}')
        result.append(BenchmarkItem(
            id=str(row["question_id"]), group_id=str(row["question_id"]),
            history=history,
            question=f'Question date: {row["question_date"]}\n{row["question"]}',
            category=row["question_type"],
            scorer_payload={"question_id": row["question_id"],
                            "question_type": row["question_type"],
                            "question": row["question"], "answer": row["answer"]}))
    return _unique(result, 500)


def _official_module(root: str | Path, name: str):
    path = _verify(root, name, "scorer")
    spec = importlib.util.spec_from_file_location(f"official_{name}_evaluation", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_predictions(items: list[BenchmarkItem], predictions: Mapping[str, str]):
    wanted = {item.id for item in items}
    if len(wanted) != len(items):
        raise ValueError("Duplicate input items; score separate trials independently")
    missing, extra = wanted - predictions.keys(), predictions.keys() - wanted
    if missing or extra:
        raise ValueError(f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    if not all(isinstance(predictions[x.id], str) for x in items):
        raise TypeError("Predictions must be response strings")


def score_locomo(root: str | Path, items: list[BenchmarkItem],
                 predictions: Mapping[str, str]) -> list[float]:
    """One call to the hash-checked, unmodified official merged QA scorer."""
    _check_predictions(items, predictions)
    scorer = _official_module(root, "locomo")
    rows = [dict(item.scorer_payload, prediction=predictions[item.id]) for item in items]
    scores, _, _ = scorer.eval_question_answering(rows, eval_key="prediction")
    return [float(x) for x in scores]


def official_judge_prompt(root: str | Path, item: BenchmarkItem, response: str) -> str:
    scorer = _official_module(root, "longmemeval")
    gold = item.scorer_payload
    return scorer.get_anscheck_prompt(gold["question_type"], gold["question"],
                                     gold["answer"], response, abstention="_abs" in item.id)


async def judge_longmemeval(root: str | Path, items: list[BenchmarkItem],
                           predictions: Mapping[str, str],
                           judge_callback: Callable[[dict[str, Any]], Awaitable[Any]]
                           ) -> tuple[list[float], list[dict[str, Any]]]:
    """Official judge prompts/parser through the campaign's shared spend gate.

    The callback receives {system,user,item_id} and returns response str or
    {response,usage,cost}. No direct provider requests occur in this module.
    """
    _check_predictions(items, predictions)
    scorer = _official_module(root, "longmemeval")
    scores, logs = [], []
    for item in items:
        gold = item.scorer_payload
        prompt = scorer.get_anscheck_prompt(gold["question_type"], gold["question"],
                                           gold["answer"], predictions[item.id],
                                           abstention="_abs" in item.id)
        result = await judge_callback({"system": "", "user": prompt, "item_id": item.id})
        response = result if isinstance(result, str) else result["response"]
        # Byte-for-byte predicate used in the upstream evaluate_qa.py loop.
        label = "yes" in response.strip().lower()
        scores.append(float(label))
        logs.append({"question_id": item.id, "hypothesis": predictions[item.id],
                     "autoeval_label": {"model": "gpt-5.4-nano", "label": label},
                     "judge_prompt": prompt, "judge_response": response,
                     "usage": result.get("usage") if isinstance(result, dict) else None,
                     "cost": result.get("cost") if isinstance(result, dict) else None})
    return scores, logs
