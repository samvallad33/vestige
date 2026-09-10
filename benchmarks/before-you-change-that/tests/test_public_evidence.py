import base64
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from cryptography.exceptions import InvalidSignature

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_evidence_verify", ROOT / "verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class PublicEvidenceTests(unittest.TestCase):
    def test_retained_bundle_is_valid_and_preserves_interruption(self):
        report = verify.verify(ROOT)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["signatures"]), 3)
        arms = report["observations"]["arms"]
        self.assertEqual([x["model_finished_naturally"] for x in arms], [True, False, True])

    def test_changed_signed_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / "evidence/vestige", root / "vestige")
            path = root / "vestige/receipt-verification/receipts.json"
            receipts = json.loads(path.read_text())
            env = receipts[0]["response"]["value"]["attestation"]["envelope"]
            env["payload"] = base64.b64encode(base64.b64decode(env["payload"]) + b" ").decode()
            path.write_text(json.dumps(receipts))
            with self.assertRaises(InvalidSignature):
                verify.verify_signatures(root)

    def test_manifest_cannot_follow_links_or_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "real.txt").write_text("evidence")
            (root / "link.txt").symlink_to(root / "real.txt")
            for path in ("../outside", "/outside", "link.txt"):
                with self.assertRaises(ValueError):
                    verify.safe_file(root, path)

    def test_changed_public_file_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test.txt").write_text("changed")
            (root / "PUBLIC-DERIVATION.json").write_text(json.dumps({"files": [
                {"path": "test.txt", "bytes": 7, "public_sha256": "0" * 64, "included_in_git": True}
            ]}))
            with self.assertRaises(ValueError):
                verify.verify_hashes(root)

    def test_unlisted_evidence_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "evidence").mkdir()
            (root / "evidence/known.txt").write_bytes(b"")
            (root / "evidence/unlisted.txt").write_text("not in the manifest")
            (root / "PUBLIC-DERIVATION.json").write_text(json.dumps({"files": [
                {"path": "evidence/known.txt", "bytes": 0,
                 "public_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                 "included_in_git": True}
            ]}))
            with self.assertRaisesRegex(ValueError, "unlisted"):
                verify.verify_hashes(root)


if __name__ == "__main__":
    unittest.main()
