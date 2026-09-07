import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_export", ROOT / "prepare_public.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class PublicExportTests(unittest.TestCase):
    def test_agent_resource_paths_receive_role_aliases(self):
        value = "/" + "Users/example/.agents/skills/example/SKILL.md"
        self.assertEqual(export.public_text(value), "@recorded-agent-resources@/skills/example/SKILL.md")

    def test_final_audit_rejects_residual_paths_and_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for value in ("/" + "Users/example/private", "ghp_" + "a" * 40,
                          "-----BEGIN " + "PRIVATE KEY-----"):
                (root / "data.txt").write_text(value)
                with self.assertRaises(ValueError):
                    export.audit_public_files(root)

    def test_final_audit_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data.txt").write_text("public")
            (root / "link.txt").symlink_to(root / "data.txt")
            with self.assertRaises(ValueError):
                export.audit_public_files(root)

    def test_final_audit_rejects_escaped_operation_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data.txt").write_text(r'{\"thread_id\":\"11111111-2222-3333-4444-555555555555\"}')
            with self.assertRaises(ValueError):
                export.audit_public_files(root)


if __name__ == "__main__":
    unittest.main()
