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

    def test_failed_mcp_call_and_false_unchanged_never_enter_transcript(self):
        for response in (
            {"isError": True},
            {"notModified": True, "packetId": "a" * 64},
        ):
            session = DeveloperSession(CATALOG, lambda *_: response)
            session.discover(["recall"])
            with self.assertRaises(ValueError):
                session.execute_tool("one", "recall", {"query": "x"})
            self.assertEqual(session.events, [])
            self.assertEqual(session.retained_packets(), set())

    def test_caller_mutations_cannot_change_owned_catalog_arguments_or_evidence(self):
        import copy

        catalog = copy.deepcopy(CATALOG)
        response = {
            "packetId": "b" * 64,
            "evidenceIncomplete": False,
            "results": [{"content": "original"}],
        }
        session = DeveloperSession(catalog, lambda *_: response)
        session.discover(["recall"])
        args = {"query": "original", "tags": ["one"]}
        result = session.execute_tool("one", "recall", args)
        args["query"] = "changed"
        response["results"][0]["content"] = "changed"
        result["results"].clear()
        catalog[0]["inputSchema"]["properties"]["injected"] = {}
        body = session.request("openai_responses", model="fixture")
        self.assertNotIn("changed", str(body))
        self.assertNotIn("injected", str(body))
        self.assertIn("original", str(body))

    def test_hundred_step_retention_reset_and_revision_state_machine(self):
        import hashlib

        revision = 0
        seen = []

        def mcp(name, args):
            seen.append(args)
            content = f"{args['query']} revision {revision}"
            packet = hashlib.sha256(content.encode()).hexdigest()
            if args.get("known_packet_id") == packet:
                return {"packetId": packet, "notModified": True}
            return {
                "packetId": packet,
                "notModified": False,
                "evidenceIncomplete": False,
                "results": [{"content": content}],
            }

        session = DeveloperSession(CATALOG, mcp)
        session.discover(["recall"])
        for step in range(100):
            reset = step % 7 == 0
            if reset:
                session.reset_context(summary="fixture continuation")
            if step % 11 == 0:
                revision += 1
            args = {"query": f"topic-{step%3}"}
            retained = session.retained_packets()
            result = session.execute_tool(f"call-{step}", "recall", args)
            if reset:
                self.assertNotIn("known_packet_id", seen[-1])
            if result["notModified"]:
                self.assertIn(result["packetId"], retained)
            for provider in ("openai_responses", "anthropic_messages"):
                body = session.request(provider, model="fixture")
                self.assertIn(f"topic-{step%3} revision {revision}", str(body))

    def test_invalid_call_identity_rejected_before_transport(self):
        calls = []
        session = DeveloperSession(CATALOG, lambda *args: calls.append(args))
        session.discover(["recall"])
        for call_id, args in [("", {}), (None, {}), ("valid", [])]:
            with self.assertRaises(ValueError):
                session.execute_tool(call_id, "recall", args)
        self.assertEqual(calls, [])
