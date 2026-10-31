"""Prepare pinned local benchmark campaigns; never starts models or paid runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAMPAIGNS = (
    {
        "id": "terminal_bench",
        "priority": 1,
        "name": "Terminal-Bench",
        "upstream": "terminal-bench",
    },
    {"id": "longrca", "priority": 2, "name": "LongRCA Bench", "upstream": "longrca"},
    {
        "id": "memoryagentbench",
        "priority": 3,
        "name": "MemoryAgentBench",
        "upstream": "memoryagentbench",
    },
    {"id": "memform", "priority": 3, "name": "MemForm", "upstream": "memform"},
    {
        "id": "longmemeval",
        "priority": 4,
        "name": "LongMemEval-S",
        "upstream": "longmemeval",
    },
    {"id": "reconcile_safe", "priority": 5, "name": "ReconcileSafe", "upstream": None},
    {
        "id": "causal_drift",
        "priority": 5,
        "name": "Causal-Drift-Eval",
        "upstream": None,
    },
)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity():
    paths = [*sorted(ROOT.glob("*.py")), ROOT / "AGENTS.md"]
    profile = ROOT / "evidence/native-audit/operator-all-rules-profile.json"
    if profile.is_file():
        paths.append(profile)
    for directory, children, files in os.walk(ROOT / "campaigns"):
        children[:] = [
            name
            for name in children
            if name not in {"data", "__pycache__", "upstream", "evidence", "smoke"}
            and not name.startswith(".")
        ]
        paths += [
            Path(directory) / name
            for name in files
            if Path(name).suffix in {".py", ".mjs", ".json", ".toml"}
        ]
    return {
        str(p.relative_to(ROOT)): sha(p)
        for p in paths
        if p.is_file() and "data" not in p.parts
    }


def create_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve() / "evidence"):
        raise ValueError("output must be inside this project's evidence directory")
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inventory():
    return {
        "scope": "local adapter/source setup; not benchmark scores",
        "provider_calls": 0,
        "campaigns": [
            {
                **item,
                "adapter_present": (
                    ROOT / "campaigns" / item["id"] / "prepare.py"
                ).is_file(),
                "upstream_present": bool(
                    item["upstream"] and (ROOT / "upstream" / item["upstream"]).is_dir()
                ),
            }
            for item in CAMPAIGNS
        ],
    }


def prepare_all(output, *, limit=10, timeout=180, selected=None):
    if type(limit) is not int or limit < 1 or limit > 10000:
        raise ValueError("limit must be 1..10000")
    if not 1 <= timeout <= 1800:
        raise ValueError("bounded preparation timeout required")
    chosen = [item for item in CAMPAIGNS if selected is None or item["id"] in selected]
    if not chosen or (selected and set(selected) != {item["id"] for item in chosen}):
        raise ValueError("unknown or empty campaign selection")
    out = create_output(output)
    before = source_identity()
    write_new(out / "source-identity.json", before)
    rows = []
    for item in chosen:
        row = {
            "campaign": item["id"],
            "priority": item["priority"],
            "status": "PREPARING",
            "provider_calls": 0,
        }
        module = ROOT / "campaigns" / item["id"] / "prepare.py"
        if not module.is_file():
            row.update(
                status="ADAPTER_MISSING", reason="preparation module not present"
            )
        else:
            command = [
                sys.executable,
                "-m",
                "campaigns." + item["id"] + ".prepare",
                "--output",
                str(out / item["id"]),
            ]
            if item["upstream"]:
                command += ["--limit", str(limit)]
                command += ["--upstream", str(ROOT / "upstream" / item["upstream"])]
            row["command"] = command
            started = time.monotonic()
            try:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": str(ROOT)},
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                (out / (item["id"] + ".stdout.txt")).write_text(process.stdout)
                (out / (item["id"] + ".stderr.txt")).write_text(process.stderr)
                row.update(
                    status="PREPARED"
                    if process.returncode == 0
                    else "PREPARATION_FAILED",
                    exit_code=process.returncode,
                )
                manifest_path = out / item["id"] / "manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text())
                    row["manifest_sha256"] = sha(manifest_path)
                    row["declared_state"] = manifest.get(
                        "status", manifest.get("state", "UNSPECIFIED")
                    )
            except subprocess.TimeoutExpired as error:
                row.update(status="PREPARATION_TIMEOUT", reason="no automatic retry")
                for key, value in [("stdout", error.stdout), ("stderr", error.stderr)]:
                    (out / (item["id"] + "." + key + ".txt")).write_bytes(
                        value if isinstance(value, bytes) else (value or "").encode()
                    )
            row["seconds"] = time.monotonic() - started
        rows.append(row)
        write_new(out / (item["id"] + ".result.json"), row)
        print(canonical(row), flush=True)
    after = source_identity()
    changed = before != after
    report = {
        "state": "SOURCE_DRIFT"
        if changed
        else "SETUP_INCOMPLETE"
        if any(row["status"] != "PREPARED" for row in rows)
        else "PREPARED_FOR_REVIEW",
        "provider_calls": 0,
        "scope": "adapter preparation only; official readiness is in each campaign manifest",
        "source_changed_during_prepare": changed,
        "campaigns": rows,
        "max_selected_records_per_campaign": limit,
    }
    report["sha256"] = hashlib.sha256(canonical(report).encode()).hexdigest()
    write_new(out / "suite.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "prepare-all"])
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--campaign", action="append")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(inventory(), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    report = prepare_all(args.output, limit=args.limit, selected=args.campaign)
    print(
        canonical(
            {
                "suite_sha256": report["sha256"],
                "output": args.output,
                "state": report["state"],
            }
        )
    )
    if report["source_changed_during_prepare"] or any(
        r["status"] != "PREPARED" for r in report["campaigns"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
