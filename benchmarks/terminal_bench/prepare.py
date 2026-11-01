"""Prepare immutable metadata and resource-qualified CPU candidates; no launch."""

import argparse
import hashlib
import json
import platform
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TB_SHA = "452bf305c6daa62fc59061d22133a7cbc7c1572e"
HARBOR_SHA = "5eb713ff9500daced4e143b98765a7c732744a1b"
AGENTDOJO_SHA = "75cea0673940d6e5dc293e8aa22ea776a63f76e9"
EXECUTION_MODULES = {
    "agentdojo-library": [
        "vestige_eval/guarded_pipeline.py",
        "vestige_eval/responses_model.py",
        "vestige_eval/budget.py",
        "vestige_eval/stack.py",
        "src/agentdojo/functions_runtime.py",
        "src/agentdojo/agent_pipeline/tool_execution.py",
    ],
    "harbor": [
        "src/harbor/agents/base.py",
        "src/harbor/agents/factory.py",
        "src/harbor/environments/base.py",
        "src/harbor/environments/docker/docker.py",
        "src/harbor/environments/docker/docker_unix.py",
    ],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def prepare(output, limit=10, upstream=None):
    if upstream is not None and Path(upstream).resolve() not in {
        (ROOT / "upstream").resolve(),
        (ROOT / "upstream/terminal-bench").resolve(),
    }:
        raise ValueError("upstream must select the pinned Terminal-Bench checkout")
    if not 1 <= limit <= 1000:
        raise ValueError("selection limit must be bounded")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preparation never overwrites a prior freeze")
    sources = {}
    for name, expected in [
        ("terminal-bench", TB_SHA),
        ("harbor", HARBOR_SHA),
        ("agentdojo-library", AGENTDOJO_SHA),
    ]:
        source = ROOT / "upstream" / name
        current = git(source, "rev-parse", "HEAD")
        if current != expected or git(
            source, "status", "--porcelain", "--untracked-files=no"
        ):
            raise RuntimeError("pinned upstream source changed: " + name)
        sources[name] = {
            "path": str(source),
            "commit": current,
            "license": "MIT" if name == "agentdojo-library" else "Apache-2.0",
            "license_sha256": sha(source / "LICENSE"),
            "origin": git(source, "remote", "get-url", "origin"),
            "execution_sha256": {
                relative: sha(source / relative)
                for relative in EXECUTION_MODULES.get(name, [])
            },
        }
    tasks = []
    for task in sorted((ROOT / "upstream/terminal-bench/tasks").glob("*/task.toml")):
        config = tomllib.loads(task.read_text())
        env = config.get("environment", {})
        cpu, memory, gpu = (
            env.get("cpus", 1),
            env.get("memory_mb", 1024),
            env.get("gpus", 0),
        )
        candidate = (
            gpu == 0
            and cpu <= 4
            and memory <= 4096
            and not (task.parent / "environment/docker-compose.yaml").exists()
        )
        files = {
            str(p.relative_to(task.parent)): sha(p)
            for p in sorted(task.parent.rglob("*"))
            if p.is_file()
        }
        tasks.append(
            {
                "task": task.parent.name,
                "path": str(task.parent),
                "resources": env,
                "cpu_candidate": candidate,
                "files_sha256": files,
                "selection_claim": "static declared-resource candidate only; image architecture and oracle still require qualification",
            }
        )
    local = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted((ROOT / "campaigns/terminal_bench").rglob("*"))
        if p.is_file()
        and p.suffix in {".py", ".mjs", ".md", ".json"}
        and "evidence" not in p.parts
        and "__pycache__" not in p.parts
    }
    manifest = {
        "dataset": "terminal-bench/terminal-bench@4.0.0",
        "prior_version": "2.1 is distinct; no cross-version pooled score",
        "sources": sources,
        "adapter_sha256": local,
        "tasks": tasks,
        "cpu_candidates": [t["task"] for t in tasks if t["cpu_candidate"]],
        "selected_cpu_candidates": [t["task"] for t in tasks if t["cpu_candidate"]][
            :limit
        ],
        "selection_limit": limit,
        "runtime": {
            "python": platform.python_version(),
            "host_arch": platform.machine(),
            "harbor": "0.25.0",
        },
        "official_utility_scoring": "Unmodified Harbor task verifiers; a successful shell command is not a task pass",
        "security_extension": "Separate disposable synthetic fixtures; never counted as official tasks",
        "full_suite_blocker": "4.0 includes GPU and larger-resource tasks unavailable on current local Colima profile",
        "launch_state": "PREPARED_NOT_LAUNCHED",
        "provider_calls": 0,
        "run_command": "PYTHONPATH=.:upstream/harbor/src:upstream/agentdojo-library/src upstream/harbor/.venv/bin/harbor run -p <pinned_CPU_task_path> -a campaigns.terminal_bench.adapter:VestigeTerminalAgent --model qwen3.6:35b-a3b-q8_0 --ak skip_model_parsing=true --ak solver_factory=local_solver:factory --ak native_profile=all_rules --n-concurrent 1 -e docker",
        "next_gate": "Exact validated free-local solver contract and compute bounds, oracle qualification and task image architecture/build checks; paid solver path remains launch-gated",
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["sha256"] = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check(path):
    data = json.loads(Path(path).read_text())
    expected = data.pop("sha256")
    if (
        hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != expected
    ):
        raise RuntimeError("manifest digest mismatch")
    for source in data["sources"].values():
        if git(source["path"], "rev-parse", "HEAD") != source["commit"]:
            raise RuntimeError("upstream revision drift")
        if git(source["path"], "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("upstream tracked source is dirty")
        for relative, digest in source["execution_sha256"].items():
            if sha(Path(source["path"]) / relative) != digest:
                raise RuntimeError("execution source bytes drift: " + relative)
    for relative, digest in data["adapter_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError("adapter drift: " + relative)
    for task in data["tasks"]:
        for relative, digest in task["files_sha256"].items():
            if sha(Path(task["path"]) / relative) != digest:
                raise RuntimeError("task bytes drift")
    return {
        "status": "PASS",
        "cpu_candidates": len(data["cpu_candidates"]),
        "tasks": len(data["tasks"]),
        "launch_state": data["launch_state"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["prepare", "check"], nargs="?", default="prepare"
    )
    parser.add_argument("path", nargs="?")
    parser.add_argument("--upstream")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    target = (
        Path(args.output) / "manifest.json"
        if args.output
        else Path(args.path)
        if args.path
        else None
    )
    if target is None:
        parser.error("provide --output directory or positional manifest path")
    result = (
        prepare(target, args.limit, args.upstream)
        if args.action == "prepare"
        else check(target)
    )
    print(
        json.dumps(
            result
            if args.action == "check"
            else {
                "status": "PREPARED",
                "manifest": str(target),
                "sha256": result["sha256"],
                "cpu_candidates": len(result["cpu_candidates"]),
            }
        )
    )
