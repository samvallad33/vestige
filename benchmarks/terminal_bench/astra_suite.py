"""Freeze the full official Terminal-Bench v4 catalog or a seeded random sample.

This is a manifest preparer, not an evaluator. Scored runs must use the official
Harbor task environment and verifier, never execute task commands on the host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PIN = "452bf305c6daa62fc59061d22133a7cbc7c1572e"
ARMS = ("bare", "bm25", "embedding", "vestige")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_catalog(repo: Path, *, sample: int | None = None, seed: int = 20261101) -> dict:
    repo = repo.resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        raise ValueError(f"official v4.0.0 revision mismatch: {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--", "tasks"], text=True
    )
    if dirty:
        raise ValueError("official task tree has local modifications")
    tasks = sorted(p.name for p in (repo / "tasks").iterdir() if p.is_dir() and (p / "task.toml").is_file())
    if sample is not None and not 20 <= sample <= len(tasks):
        raise ValueError("random sample must contain at least20 tasks and cannot exceed official catalog")
    selected = tasks if sample is None else sorted(random.Random(seed).sample(tasks, sample))
    catalog_hash = hashlib.sha256("\n".join(tasks).encode()).hexdigest()
    entries = []
    for task in selected:
        root = repo / "tasks" / task
        for rel in ("instruction.md", "task.toml", "tests/test.sh"):
            if not (root / rel).is_file():
                raise ValueError(f"official required file missing: {task}/{rel}")
        entries.append({
            "id": task,
            "files": {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()},
            "verifier": "tests/test.sh",
            "trials_per_arm": 3,
            "arms": list(ARMS),
            "model": "gpt-5.4-nano",
        })
    return {
        "benchmark": "Terminal-Bench", "version": "4.0.0", "repo": "https://github.com/harbor-framework/terminal-bench",
        "revision": PIN, "catalog_sha256": catalog_hash,
        "official_task_count": len(tasks), "selected_task_count": len(selected),
        "selection": "full" if sample is None else "random_without_replacement",
        "selection_seed": seed, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": entries, "expected_scored_rollouts": len(selected) * len(ARMS) * 3,
        "status": "PREPARED_NOT_EXECUTED", "engine_self_tests_are_benchmark_scores": False,
        "prior_task_guided_results": "INVALID",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_catalog(args.repo, sample=args.sample, seed=args.seed)
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create keeps a pre-run sample from being silently replaced.
    with args.out.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                      "official_tasks": manifest["official_task_count"],
                      "selected_tasks": manifest["selected_task_count"],
                      "rollouts": manifest["expected_scored_rollouts"]}))


if __name__ == "__main__":
    main()
