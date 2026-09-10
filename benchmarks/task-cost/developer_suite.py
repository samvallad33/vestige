#!/usr/bin/env python3
"""Frozen executable developer tasks with separate prompts, histories and evaluators.

These are curated development fixtures, not company workloads or held-out proof.
Self-test executes reference solutions and deliberately broken starting code only.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import ledger

# Expected behavior is independently asserted by the frozen evaluator, not by a
# model self-report. Reference solutions never enter the candidate source archive.
TASKS = [
    (
        "fresh",
        "Merge overlapping or touching numeric intervals. Return sorted tuples; never mutate the input.",
        "No prior decisions.",
        "def solve(items):\n    return sorted(items)\n",
        "def solve(items):\n    out=[]\n    for a,b in sorted(items):\n        if out and a<=out[-1][1]: out[-1]=(out[-1][0],max(b,out[-1][1]))\n        else: out.append((a,b))\n    return out\n",
        "x=[(5,8),(1,3),(3,6)]; before=list(x); assert solve(x)==[(1,8)]; assert x==before; assert solve([])==[]; assert solve([(2,2),(0,1)])==[(0,1),(2,2)]",
    ),
    (
        "interrupted",
        "Finish the interrupted retry-delay implementation. Return attempts exponential delays, capped at cap.",
        "Earlier session: attempt zero uses base, then base*2**index. Cap each delay. Zero attempts returns an empty list.",
        "def solve(base, cap, attempts):\n    return [min(cap,base*(i+1)) for i in range(attempts)]\n",
        "def solve(base, cap, attempts):\n    return [min(cap,base*2**i) for i in range(attempts)]\n",
        "assert solve(2,9,5)==[2,4,8,9,9]; assert solve(3,2,3)==[2,2,2]; assert solve(1,10,0)==[]",
    ),
    (
        "correction",
        "Fix timeout parsing using the latest specification. Return an integer number of milliseconds.",
        "Superseded rule: bare numbers meant seconds. Current correction: bare numbers mean milliseconds; explicit ms and s suffixes remain valid. Whitespace is allowed.",
        "def solve(text):\n    return int(str(text).strip())*1000\n",
        'def solve(text):\n    text=str(text).strip()\n    if text.endswith("ms"): return int(text[:-2].strip())\n    if text.endswith("s"): return int(text[:-1].strip())*1000\n    return int(text)\n',
        'assert solve("25")==25; assert solve(" 2s ")==2000; assert solve(" 30ms ")==30; assert solve(0)==0',
    ),
    (
        "historical_debugging",
        "Repair stable deduplication of hashable values, preserving the first occurrence order.",
        "Historical regression: replacing a list with a set removed stable ordering, which broke downstream pagination. The result must remain a list.",
        "def solve(items):\n    return list(set(items))\n",
        "def solve(items):\n    return list(dict.fromkeys(items))\n",
        'assert solve([4,1,4,3,1,2])==[4,1,3,2]; assert solve([])==[]; assert solve(["z","a","z"])==["z","a"]',
    ),
    (
        "cross_project",
        "Return deployment port for the requested project and environment. Fall back to that project default only.",
        "Project alpha default=8080 and prod=8081. Project beta default=9000 and prod=9443. Similar environment names must never share configuration between projects.",
        "def solve(config, project, env):\n    return next(iter(config.values())).get(env,8080)\n",
        'def solve(config, project, env):\n    values=config[project]\n    return values.get(env,values["default"])\n',
        'c={"alpha":{"default":8080,"prod":8081},"beta":{"default":9000,"prod":9443}}; assert solve(c,"beta","prod")==9443; assert solve(c,"beta","dev")==9000; assert solve(c,"alpha","prod")==8081',
    ),
    (
        "accumulated",
        "Select the latest active revision for each key. Deleted latest revisions remove the key.",
        "Records accumulate over sessions. Select greatest revision per key, regardless of input order. A deleted winning record must suppress older values; return a dict of active key/value pairs.",
        'def solve(records):\n    return {r["key"]:r["value"] for r in records if not r.get("deleted")}\n',
        'def solve(records):\n    latest={}\n    for r in records:\n        if r["key"] not in latest or r["revision"]>latest[r["key"]]["revision"]: latest[r["key"]]=r\n    return {k:r["value"] for k,r in latest.items() if not r.get("deleted",False)}\n',
        'x=[{"key":"a","revision":3,"deleted":True},{"key":"a","revision":1,"value":"old"},{"key":"b","revision":2,"value":"new"},{"key":"b","revision":1,"value":"old"}]; assert solve(x)=={"b":"new"}; assert solve([])=={}',
    ),
]


def evaluator(assertions):
    return (
        "import importlib.util, pathlib, sys\n"
        'path=pathlib.Path(sys.argv[1])/"solution.py"\n'
        'spec=importlib.util.spec_from_file_location("candidate",path)\n'
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
        "solve=module.solve\n" + assertions + "\n"
    )


def build(destination, identity_config):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)

    def write(name, data):
        path = root / name
        path.write_text(data)
        return {"path": name, "sha256": ledger.digest(path)}

    cases = []
    for family, prompt, history, broken, reference, assertions in TASKS:
        archive = root / f"{family}-source.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            info = zipfile.ZipInfo("solution.py", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            stream.writestr(info, broken)
        cases.append(
            {
                "id": family,
                "state": "cold" if family == "fresh" else "warm",
                "workload": family,
                "split": "development",
                "prompt": write(f"{family}-prompt.txt", prompt + "\n"),
                "history": write(f"{family}-history.txt", history + "\n"),
                "source": {"path": archive.name, "sha256": ledger.digest(archive)},
                "evaluator": write(f"{family}-evaluator.py", evaluator(assertions)),
            }
        )
    contract = {
        "version": ledger.VERSION,
        "currency": "USD",
        "cases": cases,
        "require_task_timing": True,
        "execution_policy": "Single writer. Isolate every arm/trial checkout and memory store. Same source, prompt, history and task limits. Capture every request and complete task wall time.",
        "scoring_policy": "Run the frozen external Python evaluator. Exit zero passes. Fixtures are development only; no held-out savings claim.",
        **identity_config,
    }
    # Caller configuration cannot replace workload/evaluator identities.
    contract.update(
        cases=cases, version=ledger.VERSION, currency="USD", require_task_timing=True
    )
    write("contract.json", json.dumps(contract, indent=2) + "\n")
    ledger.freeze(root)
    (root / "events.jsonl").touch()
    return root


def self_test():
    results = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        checkout = root / "checkout"
        checkout.mkdir()
        for family, prompt, history, broken, reference, assertions in TASKS:
            script = root / "evaluator.py"
            script.write_text(evaluator(assertions))
            statuses = []
            for source in (broken, reference):
                (checkout / "solution.py").write_text(source)
                result = subprocess.run(
                    [sys.executable, "-I", "-B", str(script), str(checkout)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                statuses.append(result.returncode)
            if statuses[0] == 0 or statuses[1] != 0:
                raise AssertionError(
                    f"{family}: evaluator failed qualification: {statuses}"
                )
            results.append(
                {"case": family, "broken_rejected": True, "reference_passed": True}
            )
    return {
        "evidence_kind": "reference_solution_qualification",
        "cases": results,
        "model_runs": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--identity-config", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        if args.destination is None or args.identity_config is None:
            parser.error("destination and identity-config required")
        print(build(args.destination, ledger.read_json(args.identity_config)))
