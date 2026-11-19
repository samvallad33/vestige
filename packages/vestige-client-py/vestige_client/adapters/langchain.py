"""LangChain / LangGraph `BaseStore` drop-in backed by Vestige Strata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..client import VestigeClient

try:
    from langchain_core.stores import BaseStore as _LCBaseStore  # type: ignore
except ImportError:
    class _LCBaseStore:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class StoreItem:
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]
    receipt_id: Optional[str] = None


class VestigeStore(_LCBaseStore):
    """Drop-in LangGraph / LangChain store backed by exact-handle Strata records.

    Note: Because Strata is append-only, `delete` / `mdelete` executes `suppress`
    (hiding the record from reads while preserving its cryptographic bytes).
    """

    def __init__(self, client: Optional[VestigeClient] = None, *, scope: str = "user") -> None:
        self.client = client or VestigeClient()
        self.scope = scope

    @staticmethod
    def _tag_for(namespace: tuple[str, ...], key: str) -> str:
        ns = "/".join(namespace) if namespace else "default"
        return f"lg:{ns}:{key}"

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> StoreItem:
        tag = self._tag_for(namespace, key)
        receipt = self.client.remember(
            json.dumps(value, sort_keys=True),
            node_type="fact",
            tags=[tag, f"ns:{'/'.join(namespace)}"],
            scope=self.scope,
        )
        return StoreItem(namespace=namespace, key=key, value=value, receipt_id=receipt.receipt_id)

    def get(self, namespace: tuple[str, ...], key: str) -> Optional[StoreItem]:
        tag = self._tag_for(namespace, key)
        res = self.client.recall(handle=tag, limit=1, scope=self.scope)
        items = res.get("results") or res.get("memories") or []
        if not items:
            return None
        top = items[0]
        raw_text = top.get("content", "{}")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = {"content": raw_text}
        return StoreItem(
            namespace=namespace,
            key=key,
            value=parsed,
            receipt_id=top.get("receiptId") or top.get("id"),
        )

    def delete(self, namespace: tuple[str, ...], key: str) -> dict[str, Any]:
        tag = self._tag_for(namespace, key)
        return self.client.curate("suppress", tag, scope=self.scope)

    def search(self, namespace_prefix: tuple[str, ...], *, query: Optional[str] = None, limit: int = 10) -> list[dict[str, Any]]:
        ns_tag = f"ns:{'/'.join(namespace_prefix)}" if namespace_prefix else None
        res = self.client.recall(handle=ns_tag, query=query, limit=limit, scope=self.scope)
        return list(res.get("results") or res.get("memories") or [])

    # LangChain BaseStore batch interface
    def mget(self, keys: Iterable[str]) -> list[Optional[dict[str, Any]]]:
        out: list[Optional[dict[str, Any]]] = []
        for k in keys:
            item = self.get(("langchain",), k)
            out.append(item.value if item else None)
        return out

    def mset(self, key_value_pairs: Iterable[tuple[str, dict[str, Any]]]) -> None:
        for k, v in key_value_pairs:
            self.put(("langchain",), k, v)

    def mdelete(self, keys: Iterable[str]) -> None:
        for k in keys:
            self.delete(("langchain",), k)
