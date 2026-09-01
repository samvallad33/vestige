"""Adapter protocol for the deterministic offline CauseBench harness."""

from __future__ import annotations

from typing import Protocol

from generator.generate import Memory, Task


class Adapter(Protocol):
    name: str

    def predict(self, task: Task) -> str:
        """Return the memory id ranked first for this task."""
        ...


def lexical_overlap(query: str, memory: Memory) -> int:
    q = {token.strip(".,:;!?()[]").lower() for token in query.split()}
    m = {token.strip(".,:;!?()[]").lower() for token in memory.text.split()}
    return len(q & m)
