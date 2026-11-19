"""AutoGen `Memory` provider drop-in backed by Vestige causal graph."""

from __future__ import annotations

from typing import Any, Optional

from ..client import ProofReceipt, VestigeClient


class VestigeCausalMemory:
    """Drop-in causal memory provider for AutoGen multi-agent swarms."""

    def __init__(
        self,
        client: Optional[VestigeClient] = None,
        *,
        swarm_name: str = "autogen-swarm",
        scope: str = "user",
    ) -> None:
        self.client = client or VestigeClient()
        self.swarm_name = swarm_name
        self.scope = scope

    def add(
        self,
        content: str,
        *,
        sender: Optional[str] = None,
        links: Optional[list[str]] = None,
    ) -> ProofReceipt:
        tags = [f"swarm:{self.swarm_name}"]
        if sender:
            tags.append(f"sender:{sender}")
        return self.client.remember(
            content,
            node_type="decision",
            tags=tags,
            links=links,
            scope=self.scope,
        )

    def query(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        res = self.client.recall(query=query, limit=limit, scope=self.scope)
        return list(res.get("results") or res.get("memories") or [])

    def trace_cause(self, handle: str, *, depth: int = 5) -> dict[str, Any]:
        return self.client.causal_walk(handle, direction="backward", depth=depth)
