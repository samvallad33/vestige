"""Execute the 3-Way Showdown (Bare vs RAG vs Vestige) on Terminal-Bench."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from harbor.models.job.config import JobConfig

import astra_solver
from campaigns.terminal_bench.prepare import AGENTDOJO_SHA, HARBOR_SHA, TB_SHA, git

ROOT = Path(__file__).resolve().parents[2]
SHOWDOWN_ROOT = ROOT / "campaigns/terminal_bench/evidence/showdown"
DOCKER_CONFIG = (
    ROOT
    / "campaigns/terminal_bench/evidence/oracle-session-window-debug-20261031/docker-config"
)
MODEL = "gpt-6-astra"
BUILDX_SHA = "6f20ba2437865311ef99015bf5d1039c3c8e4d2177c1c64aa41b134b5ecd3012"

SELECTED_TASKS = [
    ROOT / "upstream/terminal-bench/tasks/session-window-debug",
    ROOT / "upstream/terminal-bench/tasks/wal-recovery-ordering",
]


def build_config(mode: str) -> JobConfig:
    if mode not in {"bare", "rag", "vestige"}:
        raise ValueError(f"unknown showdown mode: {mode}")

    jobs_dir = SHOWDOWN_ROOT / "jobs" / mode
    return JobConfig.model_validate(
        {
            "job_name": f"showdown-{mode}",
            "jobs_dir": str(jobs_dir),
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
                        "mode": mode,
                    },
                }
            ],
            "tasks": [{"path": str(p)} for p in SELECTED_TASKS],
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


def run_mode(mode: str) -> int:
    config = build_config(mode)
    config_path = SHOWDOWN_ROOT / f"config-{mode}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config.model_dump_json(indent=2) + "\n")
    print(f"\n==================================================================")
    print(f"  LAUNCHING SHOWDOWN MODE: {mode.upper()}")
    print(f"==================================================================")
    cmd = [
        str(ROOT / "upstream/harbor/.venv/bin/harbor"),
        "run",
        "--config",
        str(config_path),
    ]
    return subprocess.run(cmd, cwd=ROOT, env=launch_environment(), check=True).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["bare", "rag", "vestige", "all"],
        default="all",
        help="Which mode to run (default: all sequentially)",
    )
    args = parser.parse_args(argv)
    validate_environment()

    modes = ["bare", "rag", "vestige"] if args.mode == "all" else [args.mode]
    for m in modes:
        run_mode(m)

    print("\n==================================================================")
    print("  SHOWDOWN CAMPAIGN COMPLETED ACROSS ALL MODES")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
