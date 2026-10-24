#!/usr/bin/env python3
"""Exploratory V5 Operator Lite hook benchmark. NEVER executes tool-call text.

The only case-dependent operation is writing JSON to the actual gate hook on
stdin. No subprocess shell, eval, exec, or host tool invocation is used.
This is NOT a live-host or real-effects benchmark.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_CASES = HERE / "cases.json"
DEFAULT_GATE = REPO / "operator-lite" / "operator-gate.py"
FINAL_DECISIONS = {"STOP", "PASS", "SHADOW_STOP", "ALLOW", "GATE_ERROR"}


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("frozen exploratory seed must contain exactly 20 cases")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("duplicate or invalid case id: {!r}".format(case_id))
        ids.add(case_id)
        if case.get("expected") not in ("STOP", "ALLOW"):
            raise ValueError("invalid expected decision: " + case_id)
        if not isinstance(case.get("family"), str) or not case["family"]:
            raise ValueError("missing family: " + case_id)
        if case.get("tool") != "Bash" or not isinstance(case.get("args"), dict):
            raise ValueError("seed supports only Bash input objects: " + case_id)
        if not isinstance(case["args"].get("command"), str):
            raise ValueError("missing command text: " + case_id)
        if "target_rule" in case and not isinstance(case["target_rule"], str):
            raise ValueError("invalid target_rule: " + case_id)
    return cases


def latest_receipt(receipt_home, session_id):
    receipt_dir = receipt_home / "receipts"
    if not receipt_dir.exists():
        return None
    final = None
    for filename in sorted(receipt_dir.glob("*.jsonl")):
        for line in filename.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if (record.get("session") == session_id
                    and record.get("decision") in FINAL_DECISIONS):
                final = record
    return final


def percentile_ms(samples, pct):
    if not samples:
        return None
    ordered = sorted(samples)
    return round(ordered[max(0, math.ceil(pct * len(ordered)) - 1)], 3)


def run_case(case, gate):
    # Every invocation gets a fresh, disposable policy state; command text
    # stays as JSON and cannot affect the runner's real HOME or workspace.
    # Place the synthetic HOME in the runner user home, NOT /tmp:
    # /tmp is scratch by policy, and would artificially exempt deletion
    # of the fake HOME/Documents from OP-003.
    with tempfile.TemporaryDirectory(prefix="vestige-v5-seed-", dir=Path.home()) as tmp:
        root = Path(tmp)
        home = root / "home"
        workspace = home / "project"
        operator_home = home / ".operator"
        (home / "Documents").mkdir(parents=True)
        workspace.mkdir()
        (workspace / "README.md").write_text("synthetic benchmark only\n", encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "HOME": str(home),
            "OPERATOR_HOME": str(operator_home),
            "OPERATOR_GATE_MODE": "enforce",
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "VESTIGE_DATA_DIR": str(home / ".local" / "share" / "vestige"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
        })
        session_id = "vestige-v5-seed-" + case["id"]
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": case["tool"],
            "tool_input": case["args"],
            "cwd": str(workspace),
            "session_id": session_id,
        }
        started = time.perf_counter_ns()
        try:
            # ONLY gate Python itself executes. The proposed command is
            # passed as JSON stdin; it is NEVER run by this benchmark.
            completed = subprocess.run(
                [sys.executable, str(gate), "hook", "--source", "claude"],
                input=json.dumps(payload),
                text=True,
                cwd=str(workspace),
                env=env,
                capture_output=True,
                timeout=20,
                check=False,
            )
            return_code = completed.returncode
            error = None
        except subprocess.TimeoutExpired:
            return_code = None
            error = "gate_timeout_20s"
        except OSError as exc:
            return_code = None
            error = "gate_spawn_error:" + type(exc).__name__
        elapsed_ms = (time.perf_counter_ns() - started) / 1000000.0
        receipt = latest_receipt(operator_home, session_id)
        receipt_decision = receipt.get("decision") if receipt else None
        observed = "STOP" if return_code == 2 else (
            "ALLOW" if return_code == 0 else "ERROR"
        )
        receipt_consistent = (
            (observed == "STOP" and receipt_decision == "STOP")
            or (observed == "ALLOW"
                and receipt_decision in ("PASS", "SHADOW_STOP", "ALLOW"))
        )
        commitments = receipt.get("commitments") if receipt else []
        if not isinstance(commitments, list):
            commitments = []
        target_rule = case.get("target_rule")
        return {
            "id": case["id"],
            "family": case["family"],
            "expected": case["expected"],
            "observed": observed,
            "exit_code": return_code,
            "receipt_decision": receipt_decision,
            "receipt_rule": receipt.get("commitment") if receipt else None,
            "target_rule": target_rule,
            "target_rule_present": (target_rule in commitments) if target_rule else None,
            "matched_expectation": observed == case["expected"],
            "receipt_consistent": bool(receipt_consistent),
            "latency_ms": round(elapsed_ms, 3),
            "error": error,
        }


def summarize(rows):
    exp_stop = [r for r in rows if r["expected"] == "STOP"]
    exp_allow = [r for r in rows if r["expected"] == "ALLOW"]
    latencies = [r["latency_ms"] for r in rows if r["observed"] != "ERROR"]
    families = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        family = families[row["family"]]
        family["total"] += 1
        family["correct"] += int(row["matched_expectation"])
    return {
        "total": len(rows),
        "expected_stop": len(exp_stop),
        "expected_allow": len(exp_allow),
        "stop_detected": sum(r["observed"] == "STOP" for r in exp_stop),
        "false_blocks": sum(r["observed"] == "STOP" for r in exp_allow),
        "receipt_covered": sum(r["receipt_decision"] is not None for r in rows),
        "fully_concordant": sum(
            r["matched_expectation"] and r["receipt_consistent"] for r in rows
        ),
        "errors_or_timeouts": sum(r["observed"] == "ERROR" for r in rows),
        "latency_ms_including_fresh_process_and_receipt": {
            "p50": percentile_ms(latencies, 0.50),
            "p95": percentile_ms(latencies, 0.95),
            "p99": percentile_ms(latencies, 0.99),
        },
        "by_family": dict(sorted(families.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path,
                        help="write a new JSON report; existing files are never overwritten")
    parser.add_argument("--check-cases", action="store_true",
                        help="only check the frozen JSON schema; do not invoke the gate")
    args = parser.parse_args()
    try:
        cases = load_cases(args.cases)
        if args.check_cases:
            print(json.dumps({"valid": True, "cases": len(cases),
                              "expected": dict(Counter(c["expected"] for c in cases))},
                             indent=2))
            return 0
        gate = args.gate.resolve(strict=True)
        if not gate.is_file():
            raise ValueError("gate is not a file")
        rows = [run_case(case, gate) for case in cases]
        result = {
            "status": "exploratory_direct_hook_replay_not_live_host",
            "measured_tool_effects": False,
            "commands_executed": False,
            "benchmark_utc": datetime.now(timezone.utc).isoformat(),
            "gate_sha256": file_digest(gate),
            "cases_sha256": file_digest(args.cases),
            "runner_sha256": file_digest(Path(__file__)),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "summary": summarize(rows),
            "cases": rows,
            "limitations": [
                "public synthetic seed, not independent holdout",
                "no model, live host, effect execution, OPA or native baseline",
                "hook subprocess timings are not in-process policy timings",
                "no measured customer savings or universal bypass guarantee",
            ],
        }
        output = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as file:
                file.write(output)
            print("Wrote exploratory report: {}".format(args.output))
        else:
            print(output)
        return 0 if result["summary"]["fully_concordant"] == len(rows) else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("Benchmark instrument error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
