#!/usr/bin/env python3
"""Plan or execute frozen trials with a caller-trusted Python driver.

The default command only prints the schedule. Execution is explicit and may
incur provider charges if the frozen driver uses a paid provider.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import random
import os
import signal
import subprocess
import sys
import tempfile
import zipfile

import ledger
from capture import measured_task
from score import score


def schedule(bundle):
    root = Path(bundle)
    ledger.evaluate(root)
    contract = ledger.read_json(root / "contract.json")
    seed = contract.get("trial_order_seed", 0)
    ledger.require(type(seed) is int, "trial_order_seed must be an integer")
    rng = random.Random(seed)
    pairs = [
        (case["id"], trial)
        for case in contract["cases"]
        for trial in range(contract["repetitions"])
    ]
    rng.shuffle(pairs)
    result = []
    for case, trial in pairs:
        arms = list(contract["arms"])
        rng.shuffle(arms)
        result.extend({"arm": arm, "case": case, "trial": trial} for arm in arms)
    return result


def unpack(source, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        ledger.require(
            len(entries) <= 10000
            and sum(entry.file_size for entry in entries) <= 100_000_000,
            "source archive exceeds extraction limits",
        )
        names = set()
        for entry in entries:
            name = PurePosixPath(entry.filename)
            ledger.require(entry.filename not in names, "duplicate archive member")
            names.add(entry.filename)
            ledger.require(
                not name.is_absolute()
                and ".." not in name.parts
                and "\\" not in entry.filename
                and ":" not in entry.filename,
                "unsafe archive path",
            )
            ledger.require(
                (entry.external_attr >> 16) & 0o170000 != 0o120000,
                "source symlinks are unsupported",
            )
            target = destination.joinpath(*name.parts)
            ledger.require(
                target.resolve().is_relative_to(destination), "archive escapes checkout"
            )
        archive.extractall(destination)


def run_trial(
    bundle, selection, *, trusted=False, timeout_seconds=300, output_parent=None
):
    ledger.require(trusted is True, "explicit trusted driver execution is required")
    ledger.require(
        type(timeout_seconds) is int and 1 <= timeout_seconds <= 3600, "invalid timeout"
    )
    root = Path(bundle).resolve()
    ledger.require(selection in schedule(root), "trial is not in frozen schedule")
    events_path = root / "events.jsonl"
    for line in events_path.read_text().splitlines() if events_path.exists() else []:
        event = json.loads(line)
        ledger.require(
            not (
                event.get("kind") in ("outcome", "task_span")
                and all(event.get(key) == value for key, value in selection.items())
            ),
            "trial already has a terminal outcome or task span",
        )
    contract = ledger.read_json(root / "contract.json")
    ledger.require(
        timeout_seconds == contract.get("task_timeout_seconds", 300),
        "timeout differs from frozen task policy",
    )
    driver = ledger.artifact(root, contract["driver"])
    ledger.require(driver.suffix == ".py", "driver must be a frozen Python file")
    case = next(case for case in contract["cases"] if case["id"] == selection["case"])
    directory = Path(tempfile.mkdtemp(prefix="vestige-trial-", dir=output_parent))
    checkout = directory / "checkout"
    checkout.mkdir()
    unpack(ledger.artifact(root, case["source"]), checkout)
    invocation = directory / "invocation.json"
    invocation.write_text(
        json.dumps(
            {"bundle": str(root), "checkout": str(checkout), **selection}, indent=2
        )
    )
    # Driver creates its own SDK clients and records usage through capture.py.
    # Inherited environment is caller authority; it is never copied to artifacts.
    try:
        with measured_task(root, **selection):
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", str(driver), str(invocation)],
                cwd=checkout,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout_seconds)
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
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, "frozen driver")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        status = (
            "timeout" if isinstance(error, subprocess.TimeoutExpired) else "failure"
        )
        evidence = root / f"driver-terminal-{directory.name}.json"
        evidence.write_text(
            json.dumps({"status": status, "driver_sha256": ledger.digest(driver)})
        )
        ledger.record(
            root,
            {
                "id": directory.name,
                "kind": "outcome",
                **selection,
                "status": status,
                "evidence": {"path": evidence.name, "sha256": ledger.digest(evidence)},
            },
        )
        return {"status": status, "checkout": str(checkout)}
    result = score(
        root,
        selection["arm"],
        selection["case"],
        selection["trial"],
        checkout,
        trusted=True,
        timeout_seconds=timeout_seconds,
    )
    return {**result, "checkout": str(checkout)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--execute-trusted-driver", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    plan = schedule(args.bundle)
    if not args.execute_trusted_driver:
        print(json.dumps(plan, indent=2))
    else:
        if args.index is None or not 0 <= args.index < len(plan):
            parser.error("valid schedule index required for execution")
        print(
            json.dumps(
                run_trial(
                    args.bundle,
                    plan[args.index],
                    trusted=True,
                    timeout_seconds=args.timeout_seconds,
                ),
                indent=2,
            )
        )
