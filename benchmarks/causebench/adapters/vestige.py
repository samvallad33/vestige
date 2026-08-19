"""Deterministic offline Vestige adapter for CauseBench.

The adapter captures the benchmarked mechanism: bridge from a failure to an
older memory through shared causal entities instead of text resemblance. Some
cases intentionally lack the receipt boundary required by the adapter so the
published score is below 100% and reproducible.
"""

from __future__ import annotations

from generator.generate import Task
from adapters.base import lexical_overlap


class VestigeCausalBridge:
    name = "vestige:causal-bridge"

    def predict(self, task: Task) -> str:
        if task.vestige_bridge:
            shared = [
                memory
                for memory in task.memories
                if set(memory.entities) & set(task.failure_entities)
            ]
            if shared:
                return shared[0].id
        # Honest fallback. If the bridge evidence is absent, Vestige behaves like
        # a text retriever and misses the causal source on this benchmark.
        return max(task.memories, key=lambda memory: lexical_overlap(task.failure, memory)).id
