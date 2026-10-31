"""Strict projection of the released LongRCA schema; annotations stay separate."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

SOURCES = {
    "swe_bench_pro",
    "terminal_bench_2",
    "travelplanner",
    "vitabench",
    "webarena_verified",
}
ANNOTATIONS = {"mistake_agent", "mistake_step", "mistake_reason"}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def normalize_role(value: Any) -> str:
    """Match the pinned official exporter's explicit-role normalization.

    Only a trailing explicit (-> receiver) or (→ receiver) is removed.
    Meaningful other parentheses survive. Never derive a role from a step.
    """
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s*\((?:->|→)\s*[^()]+\)\s*$", "", value.strip())
    return " ".join(unicodedata.normalize("NFKC", text).casefold().strip().split())


@dataclass(frozen=True)
class PublicTrajectory:
    question_id: str
    history: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_ID": self.question_id,
            "history": copy.deepcopy(list(self.history)),
        }

    @property
    def source(self) -> str:
        return self.question_id.split("__", 1)[0]

    @property
    def evidence_sha256(self) -> str:
        return digest(self.as_dict())

    @property
    def roles(self) -> set[str]:
        return {normalize_role(item["name"]) for item in self.history}


@dataclass(frozen=True)
class ScoringReference:
    question_id: str
    role: str
    step: int
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_ID": self.question_id,
            "mistake_agent": self.role,
            "mistake_step": self.step,
            "mistake_reason": self.rationale,
        }


def validate_public(record: dict[str, Any]) -> PublicTrajectory:
    if not isinstance(record, dict) or set(record) != {"question_ID", "history"}:
        raise ValueError("public records permit only question_ID and history")
    qid, history = record["question_ID"], record["history"]
    if (
        not isinstance(qid, str)
        or not re.fullmatch(r"[a-z0-9_]+__[0-9]+", qid)
        or qid.split("__")[0] not in SOURCES
    ):
        raise ValueError("unknown source or malformed question_ID")
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a nonempty ordered list")
    for index, item in enumerate(history):
        if not isinstance(item, dict) or set(item) != {
            "step",
            "name",
            "role",
            "content",
        }:
            raise ValueError(
                "history schema drift; unexpected fields must not enter model memory"
            )
        if type(item["step"]) is not int or item["step"] != index:
            raise ValueError(
                "history indices must be contiguous, stable, zero-based integers"
            )
        if any(not isinstance(item[key], str) for key in ("name", "role", "content")):
            raise ValueError("history names, roles, and content must be strings")
        if not normalize_role(item["name"]) or not item["role"].strip():
            raise ValueError("recorded workflow and message roles must be nonempty")
    return PublicTrajectory(qid, tuple(copy.deepcopy(history)))


def split_record(record: dict[str, Any]) -> tuple[PublicTrajectory, ScoringReference]:
    if (
        not isinstance(record, dict)
        or set(record) != {"question_ID", "history"} | ANNOTATIONS
    ):
        raise ValueError("released record schema drift")
    public = validate_public({key: record[key] for key in ("question_ID", "history")})
    role, step, rationale = (
        record[key] for key in ("mistake_agent", "mistake_step", "mistake_reason")
    )
    if not isinstance(role, str) or normalize_role(role) not in public.roles:
        raise ValueError("reference role is absent from the recorded role set")
    if type(step) is not int or not 0 <= step < len(public.history):
        raise ValueError("reference root step is out of range")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("reference rationale must be a nonempty string")
    return public, ScoringReference(public.question_id, role, step, rationale)
