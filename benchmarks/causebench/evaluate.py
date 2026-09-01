#!/usr/bin/env python3
"""Evaluate CauseBench adapters and print reproducible recall@1 numbers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from adapters.baselines import (
    EntityBlindVectorBaseline,
    FirstMemoryBaseline,
    LexicalOverlapBaseline,
)
from adapters.vestige import VestigeCausalBridge
from generator.generate import SEED, Task, generate_tasks


@dataclass(frozen=True)
class Score:
    adapter: str
    split: str
    correct: int
    total: int

    @property
    def percent(self) -> int:
        return round((self.correct / self.total) * 100)


def score_adapter(adapter: object, tasks: Iterable[Task]) -> list[Score]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for task in tasks:
        predicted = adapter.predict(task)  # type: ignore[attr-defined]
        buckets[task.split].append(predicted == task.answer_id)
    return [
        Score(adapter.name, split, sum(results), len(results))  # type: ignore[attr-defined]
        for split, results in sorted(buckets.items())
    ]


def main() -> int:
    tasks = generate_tasks()
    adapters = (
        VestigeCausalBridge(),
        FirstMemoryBaseline(),
        LexicalOverlapBaseline(),
        EntityBlindVectorBaseline(),
    )
    all_scores = [score for adapter in adapters for score in score_adapter(adapter, tasks)]

    print("CauseBench v0 offline deterministic benchmark")
    print(f"seed: {SEED}")
    print("network: disabled by design, no API keys, stdlib only")
    print(f"tasks: {len([task for task in tasks if task.split == 'synthetic'])} synthetic, {len([task for task in tasks if task.split == 'real'])} real")
    print("")
    print("recall@1")
    for score in all_scores:
        print(
            f"{score.adapter} {score.split}: "
            f"{score.percent}% ({score.correct}/{score.total})"
        )

    expected = {
        ("vestige:causal-bridge", "synthetic"): 60,
        ("vestige:causal-bridge", "real"): 50,
        ("baseline:first-memory", "synthetic"): 0,
        ("baseline:first-memory", "real"): 0,
        ("baseline:lexical-overlap", "synthetic"): 0,
        ("baseline:lexical-overlap", "real"): 0,
        ("baseline:entity-blind-vector", "synthetic"): 0,
        ("baseline:entity-blind-vector", "real"): 0,
    }
    actual = {(score.adapter, score.split): score.percent for score in all_scores}
    if actual != expected:
        print("")
        print("ERROR: score drift detected")
        print(f"expected: {expected}")
        print(f"actual:   {actual}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
