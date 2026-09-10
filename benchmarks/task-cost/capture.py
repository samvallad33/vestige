"""Synchronous, non-streaming SDK capture seam. Callers supply their own SDK call.

This module never creates a model client, discovers credentials, or starts an
agent. It records a request before execution and records unknown usage on error.
"""

from __future__ import annotations

from contextlib import contextmanager

import json
from pathlib import Path
import time
import subprocess
from uuid import uuid4

from ledger import PHASES, digest, parse_json, read_json, record, require

FORBIDDEN_FIELDS = {
    "authorization",
    "api_key",
    "apikey",
    "api-key",
    "access_token",
    "refresh_token",
    "cookie",
    "set-cookie",
}


def check_body(value):
    if isinstance(value, dict):
        require(
            not any(str(key).lower() in FORBIDDEN_FIELDS for key in value),
            "credential/header fields are not accepted in captured request bodies",
        )
        for child in value.values():
            check_body(child)
    elif isinstance(value, list):
        for child in value:
            check_body(child)


def measured_call(bundle, metadata, request, call):
    """Call e.g. client.responses.create(**request), preserving its return/exception.

    Metadata is the ledger request event without request/response/elapsed_ms.
    Use format=openai_responses or anthropic_messages. Raw content stays local.
    A successful call with missing usage remains unknown during accounting.
    """
    root = Path(bundle).resolve()
    lock = read_json(root / "contract.lock.json")
    require(
        lock["contract_sha256"] == digest(root / "contract.json"),
        "contract changed after freeze",
    )
    require(
        lock["accountant_sha256"] == digest(Path(__file__).with_name("ledger.py")),
        "accountant changed after freeze",
    )
    contract = read_json(root / "contract.json")
    require(metadata.get("kind") == "request", "request metadata required")
    require(metadata.get("arm") in contract["arms"], "unknown arm")
    require(metadata.get("price_id") in contract["prices"], "unknown price_id")
    require(
        metadata.get("format") in ("openai_responses", "anthropic_messages"),
        "unsupported capture format",
    )
    require(metadata.get("phase") in PHASES, "invalid phase")
    if metadata.get("case") is not None:
        require(
            metadata["case"] in {case["id"] for case in contract["cases"]},
            "unknown case",
        )
        require(
            type(metadata.get("trial")) is int
            and 0 <= metadata["trial"] < contract["repetitions"],
            "unplanned trial",
        )
    require(
        metadata.get("case") is not None or metadata["phase"] != "agent",
        "agent request needs case",
    )
    require(
        isinstance(request, dict) and not request.get("stream"),
        "capture supports non-streaming bodies only",
    )
    price = contract["prices"][metadata["price_id"]]
    require(
        request.get("model") == price["model"], "request model must match frozen price"
    )
    check_body(request)
    # Validate JSON and take an immutable copy before invoking user code.
    body = json.loads(json.dumps(request, allow_nan=False))
    capture_id = uuid4().hex
    directory = root / "captures"
    directory.mkdir(exist_ok=True)
    require(
        directory.resolve().is_relative_to(root), "capture directory escapes bundle"
    )

    def write(suffix, value):
        path = directory / f"{capture_id}-{suffix}.json"
        with path.open("x") as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
        return {"path": str(path.relative_to(root)), "sha256": digest(path)}

    event = dict(metadata)
    event.setdefault("id", capture_id)
    require(isinstance(event["id"], str) and event["id"], "event id required")
    ledger_path = root / "events.jsonl"
    if ledger_path.exists():
        require(
            all(
                parse_json(line).get("id") != event["id"]
                for line in ledger_path.read_text().splitlines()
                if line.strip()
            ),
            "duplicate event id",
        )
    event["request"] = write("request", body)
    started = time.monotonic_ns()
    try:
        result = call(**body)
    except Exception as error:
        # Exception messages may contain credentials, URLs, or customer data.
        # Do not serialize them. No usage means unknown billing, never free.
        event["elapsed_ms"] = (time.monotonic_ns() - started) // 1_000_000
        event["response"] = write(
            "response", {"error": {"type": type(error).__name__}, "usage": None}
        )
        record(root, event)
        raise
    event["elapsed_ms"] = (time.monotonic_ns() - started) // 1_000_000
    try:
        if hasattr(result, "model_dump"):
            response = result.model_dump(mode="json")
        else:
            response = result
        require(
            isinstance(response, dict),
            "SDK result must be JSON object or expose model_dump(mode='json')",
        )
        check_body(response)
        # Validate before opening an artifact so failure cannot leave a partial response.
        json.dumps(response, allow_nan=False)
    except Exception as error:
        event["response"] = write(
            "response", {"capture_error": type(error).__name__, "usage": None}
        )
        record(root, event)
        raise
    event["response"] = write("response", response)
    record(root, event)
    return result


@contextmanager
def measured_task(bundle, *, arm, case, trial):
    """Measure complete caller-owned task execution, including tools and retries.

    This records elapsed time, not task success; score.py records evaluator outcomes.
    Use a single ledger writer. Exceptions are re-raised after a failed span.
    """
    root = Path(bundle).resolve()
    contract = read_json(root / "contract.json")
    require(arm in contract["arms"], "unknown arm")
    require(case in {entry["id"] for entry in contract["cases"]}, "unknown case")
    require(
        type(trial) is int and 0 <= trial < contract["repetitions"], "invalid trial"
    )
    lock = read_json(root / "contract.lock.json")
    require(
        lock["contract_sha256"] == digest(root / "contract.json"),
        "contract changed after freeze",
    )
    require(
        lock["accountant_sha256"] == digest(Path(__file__).with_name("ledger.py")),
        "accountant changed after freeze",
    )
    for line in (
        (root / "events.jsonl").read_text().splitlines()
        if (root / "events.jsonl").exists()
        else []
    ):
        event = parse_json(line)
        require(
            not (
                event["kind"] == "task_span"
                and event["arm"] == arm
                and event["case"] == case
                and event["trial"] == trial
            ),
            "task span already recorded",
        )
    status = "completed"
    started = time.monotonic_ns()
    try:
        yield
    except (TimeoutError, subprocess.TimeoutExpired):
        status = "timeout"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        elapsed = (time.monotonic_ns() - started) // 1_000_000
        identity = uuid4().hex
        evidence = root / f"task-span-{identity}.json"
        with evidence.open("x") as stream:
            json.dump(
                {"clock": "monotonic_ns", "elapsed_ms": elapsed, "status": status},
                stream,
            )
        record(
            root,
            {
                "id": identity,
                "kind": "task_span",
                "arm": arm,
                "case": case,
                "trial": trial,
                "elapsed_ms": elapsed,
                "status": status,
                "evidence": {"path": evidence.name, "sha256": digest(evidence)},
            },
        )
