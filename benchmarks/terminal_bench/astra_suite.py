"""Stage/execute GPT-6 Astra wrapped in Vestige on the Terminal-Bench Suite."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

from harbor.models.job.config import JobConfig, DatasetConfig

import astra_solver
from campaigns.terminal_bench.prepare import AGENTDOJO_SHA, HARBOR_SHA, TB_SHA, git

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "campaigns/terminal_bench/astra-suite-job.json"
RUN_ROOT = ROOT / "campaigns/terminal_bench/evidence/astra-gpt6-terminal-suite"
DOCKER_CONFIG = (
    ROOT
    / "campaigns/terminal_bench/evidence/oracle-session-window-debug-20261031/docker-config"
)
MODEL = "gpt-6-astra"
BUILDX_SHA = "6f20ba2437865311ef99015bf5d1039c3c8e4d2177c1c64aa41b134b5ecd3012"
CUDA_EXCLUSIONS = ["*fp8-rmsnorm-gemm*", "*jax-speedrun-gpu*", "*math-eval-grader*"]


def resolve_tasks(task_name=None, category=None, limit=None):
    tb_tasks = ROOT / "upstream/terminal-bench/tasks"
    if task_name:
        task_dir = tb_tasks / task_name
        if not task_dir.is_dir():
            raise ValueError(f"Task directory does not exist: {task_dir}")
        return [{"path": str(task_dir)}]

    d = DatasetConfig(
        path=tb_tasks,
        exclude_task_names=CUDA_EXCLUSIONS,
    )
    resolved = asyncio.run(d.get_task_configs())
    task_dicts = [t.model_dump() for t in resolved]

    if limit is not None:
        task_dicts = task_dicts[:limit]

    return task_dicts


def build_config(task_name=None, limit=None):
    tasks = resolve_tasks(task_name=task_name, limit=limit)
    job_name = "astra-gpt6-terminal-suite" if not task_name else f"astra-gpt6-{task_name}"
    return JobConfig.model_validate(
        {
            "job_name": job_name,
            "jobs_dir": str(RUN_ROOT / "jobs"),
            "n_attempts": 1,
            "n_concurrent_trials": 1,
            "retry": {"max_retries": 0},
            "agents": [
                {
                    "import_path": "campaigns.terminal_bench.adapter:VestigeTerminalAgent",
                    "model_name": MODEL,
                    "skip_model_parsing": True,
                    "kwargs": {
                        "solver_factory": "astra_solver:factory",
                        "native_profile": "all_rules",
                    },
                }
            ],
            "tasks": tasks,
            "environment": {"type": "docker", "delete": True},
            "debug": True,
        }
    )


def validate_environment():
    for name, revision in [
        ("terminal-bench", TB_SHA),
        ("harbor", HARBOR_SHA),
        ("agentdojo-library", AGENTDOJO_SHA),
    ]:
        path = ROOT / "upstream" / name
        if git(path, "rev-parse", "HEAD") != revision or git(
            path, "status", "--porcelain", "--untracked-files=no"
        ):
            raise RuntimeError("pinned source drift: " + name)
    docker = json.loads((DOCKER_CONFIG / "config.json").read_text())
    if docker.get("auths") or docker.get("credsStore") or docker.get("credHelpers"):
        raise RuntimeError("private Docker configuration must remain credential-free")
    if (
        hashlib.sha256(
            (DOCKER_CONFIG / "cli-plugins/docker-buildx").read_bytes()
        ).hexdigest()
        != BUILDX_SHA
    ):
        raise RuntimeError("private Buildx no longer matches the published checksum")


def launch_environment():
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
        if key in os.environ
    }
    env.update(
        {
            "PYTHONPATH": f"{ROOT}:{ROOT}/upstream/harbor/src:{ROOT}/upstream/agentdojo-library/src",
            "DOCKER_CONFIG": str(DOCKER_CONFIG),
            "DOCKER_HOST": "unix:///Users/entity002/.colima/default/docker.sock",
            "HARBOR_TELEMETRY": "0",
            "OPENAI_API_KEY": astra_solver.get_api_key(),
        }
    )
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly start GPT-6 Astra inference and the task suite",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write the validated staging JSON; never starts inference",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter to a single task name (e.g. session-window-debug, distributed-dedup)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit execution to the first N tasks",
    )
    args = parser.parse_args(argv)
    validate_environment()
    config = build_config(task_name=args.task, limit=args.limit)

    if args.write_config:
        CONFIG.write_text(config.model_dump_json(indent=2) + "\n")
        print(f"Staged {len(config.tasks)} tasks to {CONFIG}")

    if not args.execute:
        print(config.model_dump_json(indent=2))
        return 0

    if (
        not CONFIG.is_file()
        or JobConfig.model_validate_json(CONFIG.read_text()) != config
    ):
        CONFIG.write_text(config.model_dump_json(indent=2) + "\n")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            str(ROOT / "upstream/harbor/.venv/bin/harbor"),
            "run",
            "--config",
            str(CONFIG),
        ],
        cwd=ROOT,
        env=launch_environment(),
        check=True,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
