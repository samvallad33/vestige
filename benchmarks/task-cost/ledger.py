#!/usr/bin/env python3
"""Offline task-cost accounting. No model calls, credentials, or inferred usage."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

VERSION = "vestige-task-cost/v1"
UNITS = ("uncached_input", "cached_input", "cache_write_5m", "cache_write_1h", "output")
PHASES = ("agent", "ingest", "embedding", "rerank", "maintenance", "setup")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label):
    require(type(value) is int and value >= 0, f"{label}: expected nonnegative integer")
    return value


def amount(value):
    require(isinstance(value, str), "USD values must be decimal strings")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal amount") from error
    require(result.is_finite() and result >= 0, "amount must be finite and nonnegative")
    return result


def parse_json(encoded):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(encoded, object_pairs_hook=unique)


def read_json(path):
    return parse_json(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(root, reference):
    require(isinstance(reference, dict), "artifact reference required")
    path = reference.get("path")
    require(isinstance(path, str) and path, "artifact path required")
    target = (root / path).resolve()
    require(target.is_relative_to(root.resolve()), "artifact escapes bundle")
    expected = reference.get("sha256", "")
    require(
        re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "invalid artifact hash"
    )
    require(digest(target) == expected, f"artifact hash mismatch: {path}")
    return target


def normalize(format_name, usage):
    """Return disjoint billed units; reasoning is a subset, never an extra charge.

    Missing provider usage stays unknown. These adapters cover text token usage
    only; callers must separately record charges such as provider tool fees.
    """
    if usage is None:
        return None
    require(isinstance(usage, dict), "usage must be an object or null")
    tokens = dict.fromkeys(UNITS, 0)
    reasoning = None
    if format_name == "openai_responses":
        total = integer(usage.get("input_tokens"), "input_tokens")
        # A missing detail field cannot establish the uncached/cached split.
        details = usage.get("input_tokens_details")
        if not isinstance(details, dict) or "cached_tokens" not in details:
            return None
        cached = integer(details["cached_tokens"], "cached_tokens")
        require(cached <= total, "cached input exceeds total input")
        tokens.update(
            uncached_input=total - cached,
            cached_input=cached,
            output=integer(usage.get("output_tokens"), "output_tokens"),
        )
        if "total_tokens" in usage:
            require(
                integer(usage["total_tokens"], "total_tokens")
                == total + tokens["output"],
                "provider total_tokens differs from input plus output",
            )
        output_details = usage.get("output_tokens_details") or {}
        if "reasoning_tokens" in output_details:
            reasoning = integer(output_details["reasoning_tokens"], "reasoning_tokens")
            require(reasoning <= tokens["output"], "reasoning exceeds output")
    elif format_name == "anthropic_messages":
        tokens.update(
            uncached_input=integer(usage.get("input_tokens"), "input_tokens"),
            output=integer(usage.get("output_tokens"), "output_tokens"),
        )
        if (
            "cache_read_input_tokens" not in usage
            or "cache_creation_input_tokens" not in usage
        ):
            return None
        tokens["cached_input"] = integer(
            usage["cache_read_input_tokens"], "cache_read_input_tokens"
        )
        created = integer(
            usage["cache_creation_input_tokens"], "cache_creation_input_tokens"
        )
        creation = usage.get("cache_creation")
        if created or creation is not None:
            if not isinstance(creation, dict):
                return None  # Cannot guess which TTL was billed.
            tokens["cache_write_5m"] = integer(
                creation.get("ephemeral_5m_input_tokens"), "cache_write_5m"
            )
            tokens["cache_write_1h"] = integer(
                creation.get("ephemeral_1h_input_tokens"), "cache_write_1h"
            )
            require(
                tokens["cache_write_5m"] + tokens["cache_write_1h"] == created,
                "cache creation breakdown differs from total",
            )
    else:
        raise ValueError(f"unsupported usage format: {format_name}")
    return {"tokens": tokens, "reasoning_tokens_subset_of_output": reasoning}


def price_usage(normalized, price):
    if normalized is None:
        return None
    rates = price.get("usd_per_million_tokens", {})
    total = Decimal(0)
    for unit, count in normalized["tokens"].items():
        if count:
            if unit not in rates:
                return None
            total += Decimal(count) * amount(rates[unit]) / Decimal(1_000_000)
    return total


def freeze(bundle):
    """Create an exclusive lock before execution. Existing locks never overwrite."""
    bundle = Path(bundle).resolve()
    contract = read_json(bundle / "contract.json")
    validate_contract(bundle, contract)
    lock = {
        "version": VERSION,
        "contract_sha256": digest(bundle / "contract.json"),
        "accountant_sha256": digest(__file__),
    }
    with (bundle / "contract.lock.json").open("x") as stream:
        stream.write(json.dumps(lock, indent=2) + "\n")
    return lock


def validate_contract(root, contract):
    require(contract.get("version") == VERSION, "unsupported contract version")
    require(
        contract.get("evidence_kind") in ("synthetic", "provider_exports"),
        "evidence_kind required",
    )
    require(contract.get("currency") == "USD", "only USD supported")
    require(
        integer(contract.get("repetitions"), "repetitions") > 0,
        "repetitions must be positive",
    )
    arms = contract.get("arms", {})
    require(len(arms) >= 2, "at least two arms required")
    common = None
    for identity in arms.values():
        for key in (
            "model",
            "model_revision",
            "effort",
            "agent_revision",
            "tool_config_sha256",
            "memory_snapshot_sha256",
        ):
            require(
                isinstance(identity.get(key), str) and identity[key].strip(),
                f"missing arm identity: {key}",
            )
        for key in ("tool_config_sha256", "memory_snapshot_sha256"):
            require(
                re.fullmatch(r"[0-9a-f]{64}", identity[key]) is not None,
                f"invalid {key}",
            )
        parity = tuple(
            identity[key]
            for key in ("model", "model_revision", "effort", "agent_revision")
        )
        require(
            common is None or parity == common,
            "model/effort/agent identity parity mismatch",
        )
        common = parity
    cases = contract.get("cases", [])
    require(
        cases and len({case["id"] for case in cases}) == len(cases),
        "cases must be nonempty and unique",
    )
    for case in cases:
        require(
            case.get("state") in ("cold", "warm"), "case state must be cold or warm"
        )
        require(
            isinstance(case.get("workload"), str) and case["workload"],
            "workload required",
        )
        for key in ("prompt", "source", "history", "evaluator"):
            artifact(root, case[key])
    for price in contract.get("prices", {}).values():
        require(
            price.get("model") and price.get("source"),
            "price model and source required",
        )
        date.fromisoformat(price["effective_date"])
        rates = price.get("usd_per_million_tokens", {})
        require(not set(rates) - set(UNITS), "unknown rate unit")
        for rate in rates.values():
            amount(rate)
    if "driver" in contract:
        artifact(root, contract["driver"])
    if "require_task_timing" in contract:
        require(
            type(contract["require_task_timing"]) is bool,
            "require_task_timing must be boolean",
        )
    require(
        contract.get("execution_policy") and contract.get("scoring_policy"),
        "execution/scoring policies required",
    )


def evaluate(bundle):
    root = Path(bundle).resolve()
    contract = read_json(root / "contract.json")
    lock = read_json(root / "contract.lock.json")
    require(lock.get("version") == VERSION, "unsupported lock version")
    require(
        lock["contract_sha256"] == digest(root / "contract.json"),
        "contract changed after freeze",
    )
    require(
        lock["accountant_sha256"] == digest(__file__), "accountant changed after freeze"
    )
    validate_contract(root, contract)
    cases = {case["id"]: case for case in contract["cases"]}
    arms = contract["arms"]
    expected = {
        (arm, case, trial)
        for arm in arms
        for case in cases
        for trial in range(contract["repetitions"])
    }
    events = [
        parse_json(line)
        for line in (root / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    seen, response_ids = set(), set()
    outcomes, requests, totals = {}, defaultdict(int), defaultdict(lambda: Decimal(0))
    agent_requests = set()
    unknown = defaultdict(list)
    phases = defaultdict(lambda: defaultdict(lambda: Decimal(0)))
    rows, overhead, task_spans = [], {}, {}
    for event in events:
        event_id = event.get("id")
        require(
            isinstance(event_id, str) and event_id and event_id not in seen,
            "missing/duplicate event id",
        )
        seen.add(event_id)
        arm = event.get("arm")
        require(arm in arms, "unknown arm")
        kind = event.get("kind")
        key = (arm, event.get("case"), event.get("trial"))
        if event.get("case") is not None:
            integer(event.get("trial"), "trial")
            require(key in expected, "unexpected task/trial")
        if kind == "task_span":
            require(
                key in expected and key not in task_spans,
                "duplicate or unplanned task span",
            )
            integer(event.get("elapsed_ms"), "task elapsed_ms")
            require(
                event.get("status") in ("completed", "failed", "timeout"),
                "invalid task span status",
            )
            artifact(root, event["evidence"])
            task_spans[key] = {
                "elapsed_ms": event["elapsed_ms"],
                "status": event["status"],
            }
            continue
        if kind == "outcome":
            require(
                key in expected and key not in outcomes,
                "duplicate or unplanned outcome",
            )
            require(
                event.get("status") in ("success", "failure", "timeout", "aborted"),
                "invalid outcome",
            )
            artifact(root, event["evidence"])
            outcomes[key] = event["status"]
            continue
        if kind == "overhead_coverage":
            require(arm not in overhead, "duplicate overhead declaration")
            require(
                event.get("status") in ("complete", "incomplete"),
                "invalid overhead coverage",
            )
            artifact(root, event["evidence"])
            overhead[arm] = event["status"]
            continue
        phase = event.get("phase")
        require(phase in PHASES, "invalid phase")
        if kind == "request":
            require(
                key in expected or (event.get("case") is None and phase != "agent"),
                "agent request needs task/trial",
            )
            request_path = artifact(root, event["request"])
            response_path = artifact(root, event["response"])
            response = read_json(response_path)
            price = contract.get("prices", {}).get(event.get("price_id"))
            require(price is not None, "unknown price_id")
            # Failed requests may have no model or usage. They remain unknown.
            if response.get("model"):
                require(
                    response["model"] == price["model"], "response/price model mismatch"
                )
            if response.get("usage") is not None:
                require(
                    response.get("model") == price["model"],
                    "usage requires exact response model",
                )
                response_id = response.get("id")
                require(
                    isinstance(response_id, str) and response_id,
                    "usage requires provider response id",
                )
                require(
                    (event["format"], response_id) not in response_ids,
                    "duplicate provider response id",
                )
                response_ids.add((event["format"], response_id))
            if phase == "agent":
                require(
                    price["model"] == arms[arm]["model"],
                    "agent model differs from contract",
                )
                agent_requests.add(key)
            normalized = normalize(event["format"], response.get("usage"))
            cost = price_usage(normalized, price)
            requests[key] += 1
            elapsed = event.get("elapsed_ms")
            if elapsed is not None:
                integer(elapsed, "elapsed_ms")
            rows.append(
                {
                    "id": event_id,
                    "arm": arm,
                    "case": event.get("case"),
                    "trial": event.get("trial"),
                    "phase": phase,
                    "provider_response_id": response.get("id"),
                    "usage_format": event["format"],
                    "usage": normalized,
                    "usd": str(cost) if cost is not None else None,
                    "elapsed_ms": elapsed,
                    "request_sha256": digest(request_path),
                    "response_sha256": digest(response_path),
                }
            )
        elif kind == "overhead":
            artifact(root, event["evidence"])
            require(
                event.get("basis") in ("measured", "estimated"),
                "overhead basis required",
            )
            cost = amount(event["usd"]) if event.get("usd") is not None else None
            rows.append(
                {
                    "id": event_id,
                    "arm": arm,
                    "phase": phase,
                    "usd": str(cost) if cost is not None else None,
                    "basis": event["basis"],
                }
            )
        else:
            raise ValueError(f"unknown event kind: {kind}")
        if cost is None:
            unknown[arm].append(event_id)
        else:
            totals[arm] += cost
            phases[arm][phase] += cost
    reports = {}
    for arm in arms:
        arm_expected = {key for key in expected if key[0] == arm}
        missing = sorted(arm_expected - outcomes.keys())
        no_requests = sorted(key for key in arm_expected if key not in agent_requests)
        successes = sum(outcomes.get(key) == "success" for key in arm_expected)
        missing_spans = sorted(arm_expected - task_spans.keys())
        complete = (
            not missing
            and not no_requests
            and not unknown[arm]
            and overhead.get(arm) == "complete"
            and (not contract.get("require_task_timing", False) or not missing_spans)
        )
        reports[arm] = {
            "task_timing_complete": not missing_spans,
            "missing_task_spans": missing_spans,
            "planned_tasks": len(arm_expected),
            "recorded_outcomes": len(arm_expected) - len(missing),
            "successes": successes,
            "success_rate": successes / len(arm_expected),
            "outcome_counts": {
                status: sum(outcomes.get(key) == status for key in arm_expected)
                for status in ("success", "failure", "timeout", "aborted")
            },
            "accounting_complete": complete,
            "known_usd": str(totals[arm]),
            "total_usd": str(totals[arm]) if complete else None,
            "usd_per_success": (
                str(totals[arm] / successes) if complete and successes else None
            ),
            "unknown_cost_events": unknown[arm],
            "missing_outcomes": missing,
            "tasks_without_requests": no_requests,
            "overhead_coverage": overhead.get(arm, "missing"),
            "known_usd_by_phase": {
                phase: str(value) for phase, value in phases[arm].items()
            },
        }
        reports[arm]["request_count"] = sum(
            count for key, count in requests.items() if key[0] == arm
        )
        reports[arm]["includes_estimated_overhead"] = any(
            row["arm"] == arm and row.get("basis") == "estimated" for row in rows
        )
        reports[arm]["known_token_units"] = {
            unit: sum(
                row["usage"]["tokens"][unit]
                for row in rows
                if row["arm"] == arm and row.get("usage") is not None
            )
            for unit in UNITS
        }
    return {
        "version": VERSION,
        "evidence_kind": contract["evidence_kind"],
        "contract_sha256": lock["contract_sha256"],
        "events_sha256": digest(root / "events.jsonl"),
        "arms": reports,
        "cost_events": rows,
        "task_spans": [
            {"arm": key[0], "case": key[1], "trial": key[2], **value}
            for key, value in sorted(task_spans.items())
        ],
        "limitations": [
            "Offline export accounting; completeness declarations are not an independent provider audit.",
            "Price-derived USD is not an invoice reconciliation; use exact tier and cache rates.",
            "No inference of quality, global savings, statistical significance, or benchmark fairness.",
            "Reasoning tokens are included in output, not charged twice.",
        ],
    }


def record(bundle, event):
    """Append one exported event. Single writer; full validation occurs at report."""
    root = Path(bundle).resolve()
    lock = read_json(root / "contract.lock.json")
    require(
        lock["contract_sha256"] == digest(root / "contract.json"),
        "contract changed after freeze",
    )
    require(
        lock["accountant_sha256"] == digest(__file__), "accountant changed after freeze"
    )
    require(isinstance(event.get("id"), str) and event["id"], "event id required")
    path = root / "events.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                require(parse_json(line).get("id") != event["id"], "duplicate event id")
    for key in ("request", "response", "evidence"):
        if key in event:
            artifact(root, event[key])
    with path.open("a") as stream:
        stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        stream.flush()
    return {"recorded": event["id"], "validated_accounting": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "record", "report"))
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--event", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            result = freeze(args.bundle)
        elif args.command == "record":
            require(args.event is not None, "record requires --event")
            result = record(args.bundle, read_json(args.event))
        else:
            result = evaluate(args.bundle)
        encoded = json.dumps(result, indent=2) + "\n"
        if args.output:
            with args.output.open("x") as stream:
                stream.write(encoded)
        else:
            print(encoded, end="")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"task-cost: {error}\n")


if __name__ == "__main__":
    main()
