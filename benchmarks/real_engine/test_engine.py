"""Isolation and fail-closed tests, including the actual release MCP binary."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.real_engine.engine import (RealEngine, VestigeMCP, VestigeMCPError,
    VestigeMCPToolError, isolate_policy, verify_build)
from benchmarks.real_engine.retrieval import BM25Memory

REPO = Path(__file__).resolve().parents[2]


class ToolErrorBoundaryTests(unittest.TestCase):
    def test_returned_mcp_error_carries_exact_current_result(self):
        client = VestigeMCP('/unused/binary', '/unused/store', warmup_seconds=0)
        raw = {'isError': True, 'content': [{'type': 'text', 'text': 'invalid action'}]}
        with patch.object(client, 'request', return_value=raw):
            with self.assertRaises(VestigeMCPToolError) as caught:
                client.call_tool('memory', {'action': 'invalid'})
        self.assertIs(caught.exception.result, raw)
        self.assertIsInstance(caught.exception, VestigeMCPError)

    def test_transport_failure_does_not_reuse_prior_tool_error(self):
        client = VestigeMCP('/unused/binary', '/unused/store', warmup_seconds=0)
        old = {'isError': True, 'content': [{'type': 'text', 'text': 'previous error'}]}
        client.last_tool_result = old
        with patch.object(client, 'request', side_effect=VestigeMCPError('transport closed')):
            with self.assertRaises(VestigeMCPError) as caught:
                client.call_tool('recall', {'handle': 'history'})
        self.assertNotIsInstance(caught.exception, VestigeMCPToolError)
        self.assertFalse(hasattr(caught.exception, 'result'))
        self.assertIs(client.last_tool_result, old)


class PolicyTests(unittest.TestCase):
    def make_policy(self, root):
        (root / "state").mkdir()
        (root / "commitments.json").write_bytes(b'{"laws":[]}\n')
        (root / "state/commitments-validated.json").write_bytes(b'{}\n')
        (root / "mode").write_text("enforce\n")
        (root / "receipts").mkdir()
        (root / "receipts/prior.jsonl").write_text("DO NOT COPY")

    def test_policy_bytes_copied_history_excluded_and_freshness_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest = root / "source", root / "dest"
            source.mkdir()
            self.make_policy(source)
            with patch.dict("os.environ", {}, clear=True):
                result = isolate_policy(source, dest)
            self.assertEqual((dest / "commitments.json").read_bytes(), (source / "commitments.json").read_bytes())
            self.assertFalse((dest / "receipts").exists())
            self.assertEqual(result["files"]["commitments.json"]["source_sha256"], result["files"]["commitments.json"]["copied_sha256"])
            with self.assertRaisesRegex(RuntimeError, "must be fresh"):
                isolate_policy(source, dest)

    def test_disabled_policy_and_missing_validated_cache_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            self.make_policy(source)
            (source / "mode").write_text("off\n")
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    isolate_policy(source, root / "dest")
            (source / "state/commitments-validated.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "validated policy cache"):
                isolate_policy(source, root / "dest2")

    def test_bm25_uses_each_private_corpus(self):
        left = BM25Memory(["alpha alpha unique", "beta common", "gamma common"])
        right = BM25Memory(["omega omega unique", "beta common", "gamma common"])
        self.assertEqual(left.search("alpha", 1)[0]["content"], "alpha alpha unique")
        self.assertNotIn("alpha", " ".join(x["content"] for x in right.search("alpha", 3)))


@unittest.skipUnless((REPO / "runs/build-manifest.json").is_file(), "actual release build required")
class NativeEngineTests(unittest.TestCase):
    def test_campaign_cleanup_closes_every_owned_child(self):
        from .engine import close_all_engines,ACTIVE_ENGINES
        with tempfile.TemporaryDirectory(dir=REPO/'runs') as tmp:
            engines=[RealEngine(REPO,Path(tmp)/str(i)).start() for i in range(2)]
            processes=[e.client._proc for e in engines]
            close_all_engines()
            self.assertTrue(all(p.poll() is not None for p in processes))
            self.assertFalse(ACTIVE_ENGINES)

    def test_actual_child_receipt_roundtrip_and_independent_stores(self):
        with tempfile.TemporaryDirectory(dir=REPO / "runs") as tmp:
            root = Path(tmp)
            with RealEngine(REPO, root / "one") as one, RealEngine(REPO, root / "two") as two:
                self.assertNotEqual(one.identity["pid"], two.identity["pid"])
                self.assertNotEqual(one.identity["data_dir"], two.identity["data_dir"])
                write = one.ingest_history(["Independent fixture: calendar moved to Monday."])[0]
                self.assertEqual(write["decision"], "create")
                retrieved = one.call("recall", {"handle": "history", "detail_level": "full"})
                self.assertEqual(retrieved["nodes"][0]["content"], "Independent fixture: calendar moved to Monday.")
                empty = two.call("recall", {"handle": "history"})
                self.assertFalse(empty.get("nodes", []))
                receipt = one.call("receipt", {"action": "get", "receipt_id": write["nodeId"]})
                self.assertTrue(receipt["attestation"]["verification"]["locallyVerified"])
                self.assertEqual(one.client._proc.args[-2:], ["--data-dir", one.identity["data_dir"]])
                one.client._proc.kill()
                one.client._proc.wait(timeout=10)
                with self.assertRaisesRegex(VestigeMCPError, "not running"):
                    one.call("recall", {"handle": "history"})

    def test_build_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO / "runs") as tmp:
            manifest = json.loads((REPO / "runs/build-manifest.json").read_text())
            manifest["binary_sha256"] = "0" * 64
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "hash differs"):
                verify_build(REPO, path)

    def test_tool_error_is_not_converted_to_empty_memory(self):
        with tempfile.TemporaryDirectory(dir=REPO / "runs") as tmp:
            with RealEngine(REPO, Path(tmp) / "engine") as engine:
                # Invalid action is an actual MCP tool error, not a simulated
                # memory backend. It must propagate instead of becoming a score.
                with self.assertRaises(VestigeMCPToolError) as caught:
                    engine.call("memory", {"action": "not-an-action"})
                raw = caught.exception.result
                self.assertIs(raw, engine.client.last_tool_result)
                self.assertIs(raw.get('isError'), True)
                events = [json.loads(line) for line in engine.audit.read_text().splitlines()]
                self.assertEqual(events[-1]['kind'], 'tool_error')
                self.assertEqual(events[-1]['raw_mcp_result'], raw)
                # A subsequent transport failure has no current MCP result;
                # an old refusal cannot masquerade as a new returned error.
                with patch.object(engine.client, 'request', side_effect=VestigeMCPError('transport closed')):
                    with self.assertRaises(VestigeMCPError) as transport:
                        engine.call('recall', {'handle': 'history'})
                self.assertNotIsInstance(transport.exception, VestigeMCPToolError)
                events = [json.loads(line) for line in engine.audit.read_text().splitlines()]
                self.assertIsNone(events[-1]['raw_mcp_result'])
                engine.client._proc.kill()
                engine.client._proc.wait(timeout=10)
                with self.assertRaises(VestigeMCPError) as dead:
                    engine.call('recall', {'handle': 'history'})
                self.assertNotIsInstance(dead.exception, VestigeMCPToolError)
                self.assertFalse(hasattr(dead.exception, 'result'))


if __name__ == "__main__":
    unittest.main()
