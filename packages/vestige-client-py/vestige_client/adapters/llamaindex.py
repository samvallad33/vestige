"""LlamaIndex `BaseRetriever` drop-in backed by Vestige proof-carrying recall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..client import VestigeClient


@dataclass
class RetrievedNode:
    node_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VestigeRetriever:
    """Native retriever adapter for LlamaIndex query engines.

    Uses exact-token BM25 / exact handle resolution over Strata — zero vectors.
    """

    def __init__(
        self,
        client: Optional[VestigeClient] = None,
        *,
        similarity_top_k: int = 5,
        scope: str = "user",
    ) -> None:
        self.client = client or VestigeClient()
        self.similarity_top_k = similarity_top_k
        self.scope = scope

    def retrieve(self, query_str_or_bundle: Any) -> list[RetrievedNode]:
        query = (
            query_str_or_bundle.query_str
            if hasattr(query_str_or_bundle, "query_str")
            else str(query_str_or_bundle)
        )
        res = self.client.recall(query=query, limit=self.similarity_top_k, scope=self.scope)
        rows = res.get("results") or res.get("memories") or []
        out: list[RetrievedNode] = []
        for idx, row in enumerate(rows):
            nid = str(row.get("id") or row.get("nodeId") or f"mem-{idx}")
            text = str(row.get("content") or row.get("text") or "")
            score = float(row.get("score", 1.0))
            out.append(
                RetrievedNode(
                    node_id=nid,
                    text=text,
                    score=score,
                    metadata={
                        "receipt_id": row.get("receiptId"),
                        "tags": row.get("tags", []),
                        "proof_backed": True,
                    },
                )
            )
        return out
