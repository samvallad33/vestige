"""Stage/validate one free-local official task. Inference requires --execute."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from harbor.models.job.config import JobConfig

from campaigns.terminal_bench.prepare import AGENTDOJO_SHA, HARBOR_SHA, TB_SHA, git

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "campaigns/terminal_bench/tonight-job.json"
RUN_ROOT = ROOT / "campaigns/terminal_bench/evidence/tonight-local-qwen"
DOCKER_CONFIG = (
    ROOT
    / "campaigns/terminal_bench/evidence/oracle-session-window-debug-20261031/docker-config"
)
MODEL = "qwen3.6:35b-a3b-q8_0"
BUILDX_SHA = "6f20ba2437865311ef99015bf5d1039c3c8e4d2177c1c64aa41b134b5ecd3012"


def build_config():
    return JobConfig.model_validate(
        {
            "job_name": "tonight-local-qwen-session-window-debug",
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
                        "solver_factory": "local_solver:factory",
                        "native_profile": "all_rules",
                    },
                }
            ],
            "tasks": [
                {
                    "path": str(
                        ROOT / "upstream/terminal-bench/tasks/session-window-debug"
                    )
                }
            ],
            "environment": {"type": "docker", "delete": True},
            "debug": True,
        }
    )


def validate(config):
    if config != build_config():
        raise RuntimeError(
            "launch configuration differs from the one-task free-local contract"
        )
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
    oracle = (
        ROOT / "campaigns/terminal_bench/evidence/oracle-session-window-debug-20261031"
    )
    preflight = json.loads((oracle / "preflight.json").read_text())
    task = ROOT / "upstream/terminal-bench/tasks/session-window-debug"
    current = {
        str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(task.rglob("*"))
        if p.is_file()
    }
    if current != preflight["task_files"]:
        raise RuntimeError("official task bytes changed after oracle qualification")
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
    return config


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
        }
    )
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly start local Qwen inference and the task",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write the validated staging JSON; never starts inference",
    )
    args = parser.parse_args(argv)
    config = validate(build_config())
    if args.write_config:
        if (
            CONFIG.exists()
            and JobConfig.model_validate_json(CONFIG.read_text()) != config
        ):
            raise RuntimeError("refusing to overwrite a different staged configuration")
        CONFIG.write_text(config.model_dump_json(indent=2) + "\n")
    if not args.execute:
        print(config.model_dump_json(indent=2))
        return 0
    if (
        not CONFIG.is_file()
        or JobConfig.model_validate_json(CONFIG.read_text()) != config
    ):
        raise RuntimeError(
            "stage the exact config with --write-config before execution"
        )
    if (Path(config.jobs_dir) / config.job_name).exists():
        raise RuntimeError("prior job exists; automatic replay/retry is prohibited")
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
