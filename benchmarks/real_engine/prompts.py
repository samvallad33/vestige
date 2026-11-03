"""Shared neutral instructions. Benchmark labels never enter this module."""
import hashlib

BASE = "Answer the user's question using the information available to you. Return a concise answer."
TOOL_LINES = {
    "bare": "Tools: none.",
    "bm25": "Tools: memory_search retrieves records from memory using BM25.",
    "embedding": "Tools: memory_search retrieves records from memory using local embeddings.",
    "vestige": "Tools: native Vestige memory tools with applicable lifecycle hooks; raw records have the tag history.",
}


def system_prompt(arm):
    return BASE + "\n" + TOOL_LINES[arm]


def audit_prompts():
    forbidden = ("session-window", "app/events.py", "EventSource", "idle_timeout",
                 "answer-key", "ground truth", "gold answer", "specific guidance")
    for arm in TOOL_LINES:
        text = system_prompt(arm)
        for token in forbidden:
            if token.lower() in text.lower():
                raise ValueError(f"task-specific prompt content: {token}")
        if text.splitlines()[0] != BASE or len(text.splitlines()) != 2:
            raise ValueError("arm prompt differs beyond neutral tools line")
    return {arm: {"text": system_prompt(arm),
                  "sha256": hashlib.sha256(system_prompt(arm).encode()).hexdigest()}
            for arm in TOOL_LINES}
