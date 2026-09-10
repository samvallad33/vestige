import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_packets import PacketCache, canonical, select_tool_definitions


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.cache = PacketCache()
        self.args = {"query": "review policy", "scope": "project"}
        self.full = {"packetId": "a" * 64, "notModified": False,
                     "evidenceIncomplete": False, "results": [{"id": "fixture", "content": "two reviewers"}]}

    def test_cached_on_disk_is_not_enough_to_skip_model_context(self):
        request = self.cache.prepare(self.args)
        self.cache.accept(request, self.full)
        self.assertNotIn("known_packet_id", self.cache.prepare(self.args))
        acknowledged = self.cache.prepare(self.args, retained_packet_ids=["a" * 64])
        self.assertEqual(acknowledged["known_packet_id"], "a" * 64)
        unchanged = self.cache.accept(acknowledged, {"packetId": "a" * 64, "notModified": True})
        self.assertNotIn("results", unchanged)
        self.assertLess(len(canonical(unchanged)), len(canonical(self.full)))

    def test_context_loss_forces_full_refresh(self):
        self.cache.accept(self.cache.prepare(self.args), self.full)
        self.cache.clear()
        self.assertNotIn("known_packet_id", self.cache.prepare(self.args, retained_packet_ids=["a" * 64]))

    def test_foreign_scope_never_inherits_acknowledgment(self):
        self.cache.accept(self.cache.prepare(self.args), self.full)
        args = {**self.args, "scope": "other"}
        self.assertNotIn("known_packet_id", self.cache.prepare(args, retained_packet_ids=["a" * 64]))

    def test_partial_packet_is_never_reused(self):
        request = self.cache.prepare(self.args)
        self.cache.accept(request, self.full)
        self.cache.accept(request, {"results": [], "evidenceIncomplete": True})
        self.assertNotIn("known_packet_id", self.cache.prepare(self.args, retained_packet_ids=["a" * 64]))

    def test_unacknowledged_not_modified_fails(self):
        with self.assertRaises(ValueError):
            self.cache.accept(self.args, {"packetId": "a" * 64, "notModified": True})

    def test_discovery_preserves_exact_schema_and_stable_order(self):
        catalog = [{"name": "recall", "inputSchema": {"properties": {"mode": {"enum": ["lookup", "reason"]}}}},
                   {"name": "memory_status", "inputSchema": {"type": "object"}}]
        chosen = select_tool_definitions(catalog, ["recall", "memory_status"])
        self.assertEqual(chosen[1], catalog[0])
        chosen[1]["inputSchema"]["properties"].clear()
        self.assertIn("mode", catalog[0]["inputSchema"]["properties"])
        with self.assertRaises(ValueError):
            select_tool_definitions(catalog, ["invented"])


if __name__ == "__main__":
    unittest.main()
