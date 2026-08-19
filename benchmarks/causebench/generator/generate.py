"""Deterministic offline task generator for CauseBench.

CauseBench measures whether a retriever can surface a causally upstream memory
when the failure text shares entity identity but intentionally shares almost no
surface vocabulary with the cause. The generator has no network, no API keys,
and no randomness beyond the fixed seed below.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

SEED = 424242


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    entities: tuple[str, ...]
    role: str


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    failure: str
    failure_entities: tuple[str, ...]
    memories: tuple[Memory, ...]
    answer_id: str
    vestige_bridge: bool


def _memory(memory_id: str, text: str, entities: Iterable[str], role: str = "candidate") -> Memory:
    return Memory(memory_id, text, tuple(entities), role)


def _task(
    task_id: str,
    split: str,
    failure: str,
    entity: str,
    answer_id: str,
    cause_text: str,
    lookalike_text: str,
    vestige_bridge: bool,
) -> Task:
    """Build one causal-gap task with the lookalike first on purpose."""
    memories = (
        _memory(f"{task_id}-lookalike", lookalike_text, (f"noise-{task_id}",), "lookalike"),
        _memory(answer_id, cause_text, (entity,), "cause"),
        _memory(
            f"{task_id}-distractor-a",
            "routine note about dashboard copy, color, and navigation labels",
            ("dashboard",),
            "distractor",
        ),
        _memory(
            f"{task_id}-distractor-b",
            "meeting note about release chores and changelog cleanup",
            ("release",),
            "distractor",
        ),
    )
    return Task(task_id, split, failure, (entity,), memories, answer_id, vestige_bridge)


def generate_tasks() -> tuple[Task, ...]:
    # Keep a fixed RNG in the public generator so deterministic shuffling is part
    # of the benchmark contract. The current data keeps the lookalike at index 0
    # after a stable per-task construction, which makes naive baselines fail.
    rng = Random(SEED)
    tasks = [
        _task(
            "synthetic-001",
            "synthetic",
            "checkout returns AUTH_TIMEOUT during payment confirmation",
            "service:checkout-token-cache",
            "synthetic-001-cause",
            "rotated checkout-token-cache TTL from 20m to 20s to speed staging tests",
            "AUTH_TIMEOUT observed during checkout retry window after payment confirmation",
            True,
        ),
        _task(
            "synthetic-002",
            "synthetic",
            "vector index emits empty neighbors after nightly compaction",
            "file:vector-index-compact",
            "synthetic-002-cause",
            "changed vector-index-compact to skip tombstone hydration when disk pressure is high",
            "empty neighbor list returned by vector search during compaction window",
            True,
        ),
        _task(
            "synthetic-003",
            "synthetic",
            "agent answers with stale timezone after profile migration",
            "config:profile-timezone",
            "synthetic-003-cause",
            "moved profile-timezone from user settings into workspace defaults for the migration",
            "stale timezone appears in assistant response after profile migration",
            True,
        ),
        _task(
            "synthetic-004",
            "synthetic",
            "observatory replay loses receipt trail after reload",
            "component:receipt-cache",
            "synthetic-004-cause",
            "renamed receipt-cache keys to shorter hashes but did not update reload lookup",
            "receipt trail missing in observatory replay after browser reload",
            False,
        ),
        _task(
            "synthetic-005",
            "synthetic",
            "MCP launcher starts old binary after upgrade",
            "path:npm-shim-bin",
            "synthetic-005-cause",
            "installed npm-shim-bin beside the package but left the release binary untouched",
            "old MCP binary still launches after package upgrade",
            False,
        ),
        _task(
            "real-001",
            "real",
            "Claude hook kept emitting the old recall format after Rust changed",
            "path:npm-global-vestige-bin",
            "real-001-cause",
            "release build was not copied into npm-global-vestige-bin after changing vestige-mcp",
            "old recall format emitted by Claude hook after Rust change",
            True,
        ),
        _task(
            "real-002",
            "real",
            "dashboard artifacts dirtied every branch after a check pass",
            "command:pnpm-build-dashboard",
            "real-002-cause",
            "ran pnpm-build-dashboard during a non-dashboard gate and committed generated assets locally",
            "generated dashboard build files changed after branch verification",
            True,
        ),
        _task(
            "real-003",
            "real",
            "memory graph showed duplicates even though consolidation ran",
            "table:memory-aliases",
            "real-003-cause",
            "created memory-aliases rows with normalized text but lookup still used raw casing",
            "duplicate memories visible in graph after consolidation",
            False,
        ),
        _task(
            "real-004",
            "real",
            "recall injected contradicted setup advice into a fresh session",
            "field:contradiction-verdict",
            "real-004-cause",
            "stored contradiction-verdict in tool output but formatter ignored it at injection time",
            "contradicted setup advice appeared in fresh recall context",
            False,
        ),
    ]
    # Deterministic no-op shuffle by split buckets documents fixed seed use while
    # preserving intentionally adversarial memory order inside each task.
    synthetic = [task for task in tasks if task.split == "synthetic"]
    real = [task for task in tasks if task.split == "real"]
    rng.shuffle(synthetic)
    rng.shuffle(real)
    synthetic.sort(key=lambda task: task.id)
    real.sort(key=lambda task: task.id)
    return tuple(synthetic + real)


if __name__ == "__main__":
    for task in generate_tasks():
        print(f"{task.id}\t{task.split}\t{task.answer_id}")
