"""Unit tests for vestige-client Python SDK and all 4 framework adapters."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vestige_client import AsyncVestigeClient, VestigeClient
from vestige_client.adapters import (
    VestigeCausalMemory,
    VestigeMemory,
    VestigeRetriever,
    VestigeStore,
)


class FakeKernel:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.suppressed: set[str] = set()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, args))
        if tool == "remember":
            nid = f"mem-{len(self.records) + 1:016x}"
            rid = f"eff-{len(self.records) + 1:016x}"
            rec = {
                "id": nid,
                "nodeId": nid,
                "receiptId": rid,
                "canonicalHash": "ab" * 32,
                "content": args["content"],
                "tags": args.get("tags", []),
                "links": args.get("links", []),
            }
            self.records.append(rec)
            return rec
        if tool == "recall":
            handle = args.get("handle")
            query = (args.get("query") or "").lower()
            matched = []
            for r in self.records:
                if r["id"] in self.suppressed or any(t in self.suppressed for t in r["tags"]):
                    continue
                if handle and (r["id"] == handle or handle in r["tags"]):
                    matched.append(r)
                elif query and query in r["content"].lower():
                    matched.append(r)
            return {"results": matched[: args.get("limit", 10)]}
        if tool == "curate" and args.get("action") == "suppress":
            self.suppressed.add(args["handle"])
            return {"suppressed": args["handle"]}
        if tool == "causal_walk":
            return {"from": args["from_handle"], "edges": [{"to": "mem-root"}]}
        return {"ok": True}


class TestVestigePythonSDK(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = FakeKernel()
        self.client = VestigeClient(transport=self.kernel)

    def test_langchain_store_put_get_delete(self) -> None:
        store = VestigeStore(self.client)
        store.mset([("k1", {"role": "admin", "ttl": 900})])
        self.assertEqual(store.mget(["k1"]), [{"role": "admin", "ttl": 900}])
        store.mdelete(["k1"])
        self.assertEqual(store.mget(["k1"]), [None])

    def test_crewai_memory_save_and_search(self) -> None:
        mem = VestigeMemory(self.client, crew_tag="crew-alpha")
        receipt = mem.save("Deploy gateway with strict mTLS", agent="sec-lead")
        self.assertTrue(receipt.node_id.startswith("mem-"))
        hits = mem.search("mtls")
        self.assertEqual(len(hits), 1)

    def test_llamaindex_retriever_proof_backed(self) -> None:
        self.client.remember("Strata uses BLAKE3 hash chains", tags=["arch"])
        retriever = VestigeRetriever(self.client, similarity_top_k=3)
        nodes = retriever.retrieve("blake3")
        self.assertEqual(len(nodes), 1)
        self.assertTrue(nodes[0].metadata["proof_backed"])
        self.assertIsNotNone(nodes[0].metadata["receipt_id"])

    def test_autogen_causal_memory_and_walk(self) -> None:
        mem = VestigeCausalMemory(self.client, swarm_name="core")
        r1 = mem.add("Fix race in lockfile", sender="coder", links=["mem-bug-1"])
        self.assertIsNotNone(r1.receipt_id)
        walk = mem.trace_cause(r1.node_id)
        self.assertEqual(walk["edges"][0]["to"], "mem-root")

    def test_async_client(self) -> None:
        async def run() -> None:
            ac = AsyncVestigeClient(transport=self.kernel)
            rec = await ac.remember("async decision", tags=["async"])
            found = await ac.recall(handle="async")
            self.assertEqual(found["results"][0]["id"], rec.node_id)
            await ac.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
