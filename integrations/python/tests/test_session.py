import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vestige_runtime import DeveloperSession

CATALOG = [
    {
        "name": name,
        "description": name,
        "inputSchema": {"type": "object", "properties": {}},
    }
    for name in ["memory_status", "recall", "memory", "graph"]
]


class SessionTests(unittest.TestCase):
    def test_provider_requests_keep_evidence_once_and_refresh_after_compaction(self):
        seen = []

        def mcp(name, args):
            seen.append(args)
            if "known_packet_id" in args:
                return {"notModified": True, "packetId": "a" * 64, "results": []}
            return {
                "notModified": False,
                "evidenceIncomplete": False,
                "packetId": "a" * 64,
                "results": [{"id": "source", "content": "use timeout=25"}],
            }

        session = DeveloperSession(CATALOG, mcp)
        session.discover(["recall"])
        session.add_user("Repair timeout configuration")
        session.execute_tool("one", "recall", {"query": "timeout"})
        session.execute_tool("two", "recall", {"query": "timeout"})
        self.assertNotIn("known_packet_id", seen[0])
        self.assertEqual(seen[1]["known_packet_id"], "a" * 64)
        for provider in ["openai_responses", "anthropic_messages"]:
            body = session.request(provider, model="fixture")
            self.assertEqual(str(body).count("use timeout=25"), 1)
            self.assertEqual(len(body["tools"]), 2)
        session.reset_context(summary="Continue repairing timeout")
        session.execute_tool("three", "recall", {"query": "timeout"})
        self.assertNotIn("known_packet_id", seen[-1])

    def test_unknown_tool_and_overflow_are_explicit(self):
        session = DeveloperSession(CATALOG, lambda *_: {}, max_context_bytes=1024)
        with self.assertRaises(ValueError):
            session.execute_tool("one", "graph", {})
        session.add_user("x" * 2000)
        with self.assertRaises(ValueError):
            session.request("openai_responses", model="fixture")
