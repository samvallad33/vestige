"""Shared Terminal-Bench prompt boundary; this module does not execute tasks.

Only official instruction.md content may supply task-specific information.
Execution must use Harbor's isolated environment and official verifier.
Earlier task-guided results are invalid; native proof.json files are engine
self-tests and never Terminal-Bench scores.
"""
from __future__ import annotations

from benchmarks.real_engine.prompts import audit_prompts, system_prompt


def task_messages(instruction: str, arm: str) -> list[dict[str, str]]:
    """Use exactly the shared campaign prompt plus the official instruction."""
    audit_prompts()
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("official task instruction is empty")
    return [
        {"role": "system", "content": system_prompt(arm)},
        {"role": "user", "content": instruction},
    ]
