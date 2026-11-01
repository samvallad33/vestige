"""Isolation and fail-closed tests, including the actual release MCP binary."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.real_engine.engine import RealEngine, VestigeMCPError, isolate_policy, verify_build
from benchmarks.real_engine.retrieval import BM25Memory

REPO = Path(__file__).resolve().parents[2]


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
                with self.assertRaises(VestigeMCPError):
                    engine.call("memory", {"action": "not-an-action"})


if __name__ == "__main__":
    unittest.main()
