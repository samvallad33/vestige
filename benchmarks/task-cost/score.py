#!/usr/bin/env python3
"""Run a caller-trusted, frozen Python evaluator and record its terminal outcome.

The evaluator receives the candidate checkout as its only argument. Exit 0 means
success. This runner is not an OS security sandbox; evaluator independence and
checkout isolation must be qualified before using results for product claims.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import ledger


def score(bundle, arm, case_id, trial, checkout, *, trusted=False, timeout_seconds=300):
    ledger.require(trusted is True, "explicit trusted evaluator execution is required")
    ledger.require(
        type(timeout_seconds) is int and 1 <= timeout_seconds <= 3600,
        "timeout_seconds must be between 1 and 3600",
    )
    root = Path(bundle).resolve()
    checkout = Path(checkout).resolve(strict=True)
    ledger.require(checkout.is_dir(), "checkout must be a directory")
    ledger.require(
        not root.is_relative_to(checkout) and not checkout.is_relative_to(root),
        "checkout and evidence bundle must be separate directories",
    )
    report = ledger.evaluate(root)
    contract = ledger.read_json(root / "contract.json")
    ledger.require(arm in contract["arms"], "unknown arm")
    ledger.require(
        type(trial) is int and 0 <= trial < contract["repetitions"], "invalid trial"
    )
    matches = [case for case in contract["cases"] if case["id"] == case_id]
    ledger.require(len(matches) == 1, "unknown case")
    case = matches[0]
    evaluator = ledger.artifact(root, case["evaluator"])
    ledger.require(
        evaluator.suffix == ".py", "scoring requires a frozen Python evaluator"
    )
    for line in (root / "events.jsonl").read_text().splitlines():
        event = ledger.parse_json(line)
        ledger.require(
            not (
                event["kind"] == "outcome"
                and event["arm"] == arm
                and event["case"] == case_id
                and event["trial"] == trial
            ),
            "outcome already recorded",
        )
    identity = uuid.uuid4().hex
    receipt_path = root / f"evaluation-{identity}.json"
    started = time.monotonic_ns()
    timed_out = False
    # No inherited API credentials, Python path, or provider configuration.
    environment = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR")
        if key in os.environ
    }
    environment.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", str(evaluator), str(checkout)],
        cwd=checkout,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
    finally:
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
    elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
    status = (
        "timeout" if timed_out else "success" if process.returncode == 0 else "failure"
    )
    # Revalidate frozen evaluator and all ledger artifacts after execution.
    # A trusted evaluator that mutates evidence must not record a valid outcome.
    after = ledger.evaluate(root)
    ledger.require(
        after["events_sha256"] == report["events_sha256"],
        "ledger changed during evaluation",
    )
    receipt = {
        "version": "vestige-task-evaluation/v1",
        "status": status,
        "exit_code": process.returncode,
        "elapsed_ms": elapsed_ms,
        "timeout_seconds": timeout_seconds,
        "evaluator_sha256": ledger.digest(evaluator),
        "runner_sha256": ledger.digest(__file__),
        "contract_sha256": report["contract_sha256"],
        "isolation": "separate subprocess; no OS sandbox",
        "authority": "caller explicitly trusted frozen evaluator",
        "output_capture": "discarded; receipt contains terminal status only",
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    ledger.record(
        root,
        {
            "id": f"evaluation-{identity}",
            "kind": "outcome",
            "arm": arm,
            "case": case_id,
            "trial": trial,
            "status": status,
            "evidence": {
                "path": receipt_path.name,
                "sha256": ledger.digest(receipt_path),
            },
        },
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--trial", required=True, type=int)
    parser.add_argument("--execute-trusted-evaluator", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    print(
        json.dumps(
            score(
                args.bundle,
                args.arm,
                args.case,
                args.trial,
                args.checkout,
                trusted=args.execute_trusted_evaluator,
                timeout_seconds=args.timeout_seconds,
            ),
            indent=2,
        )
    )
