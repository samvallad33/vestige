"""No-model baselines for CauseBench.

These baselines deliberately approximate common retrieval shortcuts. They are
expected to score 0% because CauseBench makes the surface-similar memory a
lookalike, not the causal source.
"""

from __future__ import annotations

from generator.generate import Task
from adapters.base import lexical_overlap


class FirstMemoryBaseline:
    name = "baseline:first-memory"

    def predict(self, task: Task) -> str:
        return task.memories[0].id


class LexicalOverlapBaseline:
    name = "baseline:lexical-overlap"

    def predict(self, task: Task) -> str:
        ranked = sorted(
            task.memories,
            key=lambda memory: (lexical_overlap(task.failure, memory), memory.id),
            reverse=True,
        )
        return ranked[0].id


class EntityBlindVectorBaseline:
    name = "baseline:entity-blind-vector"

    def predict(self, task: Task) -> str:
        # Stand-in for a single-vector retriever that sees only text resemblance.
        # In every task the lookalike shares more failure words than the cause.
        ranked = sorted(
            task.memories,
            key=lambda memory: (lexical_overlap(task.failure, memory), memory.role == "lookalike"),
            reverse=True,
        )
        return ranked[0].id
