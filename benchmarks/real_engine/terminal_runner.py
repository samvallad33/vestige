"""Fail-closed qualification for the official frozen Terminal-Bench v4 sample.

This module deliberately produces no benchmark score. The frozen sample contains
an H100 task which the current Docker daemon cannot execute. A working small
container or an engine proof.json does not qualify the full task set.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

from .prompts import TOOL_LINES
from .provider import MODEL
from .runtime import atomic_write, canonical, utc_now

SOURCE_REVISION = "452bf305c6daa62fc59061d22133a7cbc7c1572e"
FROZEN_SHA256 = "c8ad0af36f332d7e72025450423d5fc0d86b47cc2c1f37c33435200c1d2dcc9b"
CATALOG_SHA256 = "fff6d209e08ba2442a40ce2af0fb706452c69441d4a9c21e9fd6b27e8a91f130"
BASE = ("Complete the task in the isolated environment. Use the terminal tool to inspect and edit "
        "files and run checks. Give a concise final response when finished.")


def terminal_prompt(arm):
    return BASE + "\n" + TOOL_LINES[arm]


def load_frozen(path: Path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != FROZEN_SHA256:
        raise ValueError("Frozen Terminal-Bench sample identity changed")
    sample = json.loads(data)
    if (sample["revision"] != SOURCE_REVISION or sample["version"] != "4.0.0"
            or sample["catalog_sha256"] != CATALOG_SHA256):
        raise ValueError("Unrecognized official Terminal-Bench source")
    tasks = sample["tasks"]
    if len(tasks) != sample["official_task_count"] or len(tasks) != 66 or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Terminal-Bench requires the complete official 66-task catalog")
    if any(t["trials_per_arm"] != 3 or set(t["arms"]) != set(TOOL_LINES) for t in tasks):
        raise ValueError("Frozen tasks must retain three trials in all four arms")
    if sample["expected_scored_rollouts"] != len(tasks)*3*4:
        raise ValueError("Frozen rollout count mismatch")
    return sample


def checked_task_config(source: Path, task):
    path = source / "tasks" / task["id"] / "task.toml"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != task["files"]["task.toml"]:
        raise ValueError(f"Official task config changed: {task['id']}")
    return tomllib.loads(raw.decode())


def resource_blockers(sample, source: Path, docker_info: dict):
    issues = []
    for task in sample["tasks"]:
        env = checked_task_config(source,task).get("environment",{})
        required_memory = env.get("memory_mb",0)*1024*1024
        if required_memory > docker_info.get("MemTotal",0):
            issues.append({"task":task["id"],"kind":"memory", "required_mb":env["memory_mb"],
                           "available_bytes":docker_info.get("MemTotal",0)})
        if env.get("cpus",0) > docker_info.get("NCPU",0):
            issues.append({"task":task["id"],"kind":"cpu", "required":env["cpus"],
                           "available":docker_info.get("NCPU",0)})
        if env.get("gpus",0):
            # A generic Docker GPU runtime alone cannot establish a matching H100.
            # Hardware evidence must name the required GPU class before scoring.
            issues.append({"task":task["id"],"kind":"gpu_identity_unverified",
                           "required_gpus":env["gpus"],"required_types":env.get("gpu_types",[])})
    return issues


def inspect_terminal(runs: Path, source: Path | None = None):
    root = runs / "terminal_bench"
    sample = load_frozen(root/"sample-frozen.json")
    output = {"benchmark":"terminal_bench","timestamp":utc_now(),"status":"unfinished",
        "model":MODEL,"judge":"official task verifier", "n":0,"score":None,"ci95":None,
        "planned_tasks":len(sample["tasks"]),"planned_trials_per_arm":3,
        "planned_rollouts":sample["expected_scored_rollouts"], "workers_per_arm":5,
        "sample_sha256":FROZEN_SHA256,"dataset_catalog_sha256":CATALOG_SHA256,
        "source_revision":SOURCE_REVISION,"seed":sample["selection_seed"],
        "earlier_session_window_debug":"INVALID", "proof_json":"engine self-tests, never benchmark scores",
        "blockers":[],"api_calls":0,"full_task_bytes_verified":False}
    source = source or runs / "upstream/terminal-bench"
    if not source.exists():
        output["blockers"].append({"kind":"official_source_missing","path":str(source)})
    else:
        revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
        if revision != SOURCE_REVISION:
            output["blockers"].append({"kind":"official_source_revision_mismatch","actual":revision})
        else:
            try:
                proc = subprocess.run(["docker","info","--format","{{json .}}"],capture_output=True,text=True,timeout=15)
                proc.check_returncode()
                info = json.loads(proc.stdout)
                output["docker"] = {k:info.get(k) for k in ("ServerVersion","Architecture","OSType","NCPU","MemTotal")}
                output["blockers"].extend(resource_blockers(sample,source,info))
            except (OSError, subprocess.SubprocessError,ValueError) as exc:
                output["blockers"].append({"kind":"docker_setup_failure","error":str(exc)})
    hooks = Path(__file__).with_name("hooks.py")
    if not hooks.exists():
        output["blockers"].append({"kind":"full_applicable_hooks_controller_missing"})
    else:
        # File existence alone is not qualification of the complete hook route.
        output["blockers"].append({"kind":"full_applicable_hooks_route_not_qualified"})
    feasibility = root/"feasibility/result.json"
    if feasibility.exists():
        output["environment_self_test"] = json.loads(feasibility.read_text())
    output["next_action"] = ("Provide Docker resources matching every frozen official task, including H100; "
        "qualify all applicable Vestige hooks; integrate Harbor's separate-verifier artifact flow; "
        "then run the budgeted one-item pilot before the unchanged 792-rollout full catalog.")
    return output


async def run_terminal(campaign, source: Path | None = None):
    """No engine-only or below-minimum fallback is authorized by this runner."""
    output = await asyncio.to_thread(inspect_terminal,campaign.runs,source)
    output["binary_sha256"] = campaign.manifest["binary_sha256"]
    root = campaign.runs/"terminal_bench"
    atomic_write(root/"PENDING.json",canonical(output)+"\n")
    lines = ["# Terminal-Bench 4.0 results", "", "Unfinished: official sample execution is blocked by setup requirements.",
        "No paid model calls or official scored rollouts have run through this adapter.", "",
        "| Arm | Mean | 95% CI | n | Trials planned per task | Model | Cost |",
        "|---|---|---|---:|---:|---|---:|"]
    for arm in TOOL_LINES:
        campaign.progress["terminal_bench",arm] = {"benchmark":"terminal_bench","arm":arm,"done":0,
            "total":output["planned_tasks"]*3,"errors":0}
        lines.append(f"| {arm} | not scored | not available | 0/{output['planned_tasks']*3} | 3 | {MODEL} | $0.00 |")
    lines += ["", f"Frozen sample: {output['planned_tasks']} tasks selected before evaluation with seed {output['seed']}; 792 total planned rollouts.",
        f"Official revision: `{SOURCE_REVISION}`. Catalog SHA-256: `{CATALOG_SHA256}`.",
        f"Sample SHA-256: `{FROZEN_SHA256}`. Binary SHA-256: `{output['binary_sha256']}`.", "", "Setup blockers:"]
    lines.extend("- "+canonical(issue) for issue in output["blockers"])
    lines += ["", "Earlier session-window-debug results with arm-specific solution guidance are invalid.",
        "proof.json artifacts are engine self-tests and are excluded from benchmark scores.",
        "A successful small official task container start is an environment self-test; it does not establish that the full frozen sample can run.",
        "Separate-verifier tasks must retain Harbor's official artifact transfer and verifier environment; executing tests directly in the agent environment is insufficient.",
        "No replacement sample, hand-picked scored subset, modified task resources, simulated task world, or engine-only hook configuration is substituted.",
        "", "Next action: "+output["next_action"], ""]
    from .integrity import audit,require,IntegrityPending
    report=await asyncio.to_thread(audit,Path(__file__).resolve().parents[2],campaign.runs)
    try:require(report,"terminal_bench","report")
    except IntegrityPending:return output
    atomic_write(root/"RESULTS.md","\n".join(lines))
    return output
