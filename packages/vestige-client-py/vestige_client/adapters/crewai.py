"""CrewAI `ShortTermMemory` / `LongTermMemory` drop-in backed by Vestige."""

from __future__ import annotations

from typing import Any, Optional

from ..client import ProofReceipt, VestigeClient


class VestigeMemory:
    """Drop-in replacement for CrewAI ShortTermMemory & LongTermMemory."""

    def __init__(
        self,
        client: Optional[VestigeClient] = None,
        *,
        crew_tag: str = "crewai",
        scope: str = "user",
    ) -> None:
        self.client = client or VestigeClient()
        self.crew_tag = crew_tag
        self.scope = scope

    def save(
        self,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> ProofReceipt:
        tags = [self.crew_tag]
        if agent:
            tags.append(f"agent:{agent}")
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, str) and v:
                    tags.append(f"{k}:{v}")
        return self.client.remember(value, node_type="note", tags=tags, scope=self.scope)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        res = self.client.recall(query=query, limit=limit, scope=self.scope)
        return list(res.get("results") or res.get("memories") or [])

    def reset(self) -> dict[str, Any]:
        # Append-only safety invariant: never erases bytes; suppresses crew tag view.
        return self.client.curate("suppress", self.crew_tag, scope=self.scope)
