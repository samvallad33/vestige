"""Contract fixtures are unit tests; native tests use the actual release child."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.memconflict.mcp_client import VestigeMCP
from .engine import RealEngine, VestigeMCPError
from .native_contract import (INSTRUCTIONS_SEPARATOR, NativeBoundaryViolation,
    contract_identity, digest_json, model_tools, validate_worker_arguments)
from .runtime import read_jsonl


REPO = Path(__file__).resolve().parents[2]


class NativeContractTests(unittest.TestCase):
    def test_every_catalog_entry_schema_and_description_is_preserved(self):
        # Minimal protocol fixture only, never a memory implementation.
        catalog = [{"name": name, "description": f"Native {name}.",
                    "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}}
                   for name in ["recall", "memory_status", "project", "gather", "receipt"]]
        before = deepcopy(catalog)
        initialize = {"instructions": "Use exact native handles.\nDiscover tools via memory_status."}
        tools = model_tools(catalog, initialize)
        self.assertEqual([t["function"]["name"] for t in tools], [t["name"] for t in catalog])
        for native, adapted in zip(catalog, tools, strict=True):
            function = adapted["function"]
            self.assertEqual(function["parameters"], native["inputSchema"])
            expected = native["description"]
            if native["name"] == "memory_status":
                expected += INSTRUCTIONS_SEPARATOR + initialize["instructions"]
            self.assertEqual(function["description"], expected)
        self.assertEqual(catalog, before)
        self.assertEqual(sum(initialize["instructions"] in t["function"]["description"] for t in tools), 1)
        tools[0]["function"]["parameters"]["properties"].clear()
        self.assertEqual(catalog, before)
        identity = contract_identity(catalog, initialize)
        self.assertTrue(identity["complete_catalog"])
        self.assertEqual(identity["tool_count"], len(catalog))
        self.assertEqual(identity["catalog_sha256"], digest_json(catalog))

    def test_missing_native_instructions_or_duplicate_catalog_fails(self):
        catalog = [{"name": "memory_status", "description": "Native", "inputSchema": {}}]
        with self.assertRaisesRegex(RuntimeError, "instructions"):
            model_tools(catalog, {})
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            model_tools(catalog + catalog, {"instructions": "Native contract"})

    def test_scoped_paths_allowed_and_ambient_effects_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            home, cwd = root / "home", root / "work"
            home.mkdir(); cwd.mkdir()
            options = {"roots": (root,), "home": home, "cwd": cwd}
            for name, arguments in [
                ("codebase", {"action": "get_context", "repoPath": str(cwd),
                              "files": ["a.rs#symbol"], "anchors": [{"path": "b.rs"}]}),
                ("project", {"action": "preview", "path": "~/MEMORY.md"}),
                ("receipt", {"action": "prove", "params": {"repoPath": str(cwd)}}),
                ("smart_ingest", {"content": "An opaque memory may mention /outside/path or https://example.test"}),
            ]:
                validate_worker_arguments(name, arguments, **options)
            for name, arguments in [
                ("codebase", {"action": "get_context", "repoPath": "/outside"}),
                ("codebase", {"action": "verify", "anchors": [{"path": "../../outside"}]}),
                ("session_start", {"context": {"repoPath": "/outside"}}),
                ("project", {"path": "~another/MEMORY.md"}),
                ("maintain", {"action": "export", "allowAnyPath": True, "path": str(cwd)}),
                ("gather", {"url": "https://example.test/issue"}),
            ]:
                with self.assertRaises(NativeBoundaryViolation, msg=(name, arguments)):
                    validate_worker_arguments(name, arguments, **options)

    def test_symlink_cannot_escape_worker_paths(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = Path(tmp).resolve()
            (root / "link").symlink_to(Path(other).resolve(), target_is_directory=True)
            with self.assertRaisesRegex(NativeBoundaryViolation, "escapes"):
                validate_worker_arguments("project", {"path": "link/out.md"},
                                          roots=(root,), cwd=root, home=root)

    def test_client_clean_environment_scrubs_credentials_and_sets_real_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            client = VestigeMCP("/unit-capture-only", str(root / "store"), warmup_seconds=0,
                               cwd=str(root), clean_env=True,
                               extra_env={"HOME": str(root / "home"), "TMPDIR": str(root)})
            with patch.dict(os.environ, {"OPENAI_API_KEY": "unit-secret", "GH_TOKEN": "unit-secret",
                                         "PYTHONPATH": "/ambient", "HTTPS_PROXY": "http://ambient"}), \
                    patch("benchmarks.memconflict.mcp_client.subprocess.Popen",
                          side_effect=RuntimeError("unit capture")) as popen:
                with self.assertRaisesRegex(RuntimeError, "unit capture"):
                    client.start()
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["cwd"], str(root))
            self.assertEqual(kwargs["env"]["HOME"], str(root / "home"))
            self.assertFalse({"OPENAI_API_KEY", "GH_TOKEN", "PYTHONPATH", "HTTPS_PROXY"} & kwargs["env"].keys())


@unittest.skipUnless((REPO / "runs/build-manifest.json").is_file(), "actual release build required")
class ActualNativeContractTests(unittest.TestCase):
    def test_real_complete_contract_exact_recall_and_boundary_provenance(self):
        with tempfile.TemporaryDirectory(dir=REPO / "runs") as tmp:
            with RealEngine(REPO, Path(tmp) / "native") as engine:
                catalog = engine.client.request("tools/list")["tools"]
                self.assertEqual(engine.tools, catalog)
                self.assertEqual({tool["function"]["name"] for tool in engine.model_tools()},
                                 {tool["name"] for tool in catalog})
                self.assertGreater(len(catalog), 5)
                self.assertTrue(engine.identity["clean_environment"])
                self.assertFalse(any((p / ".git").exists() for p in (engine.cwd, *engine.cwd.parents)))
                self.assertEqual(engine.client._proc.args[-2:], ["--data-dir", engine.identity["data_dir"]])
                writes = engine.ingest_history(["Native contract fixture: the appointment moved to Thursday."])
                recalled = engine.call("recall", {"handle": "history", "detail_level": "full"})
                self.assertEqual(recalled["nodes"][0]["id"], writes[0]["nodeId"])
                self.assertEqual(recalled["nodes"][0]["content"],
                                 "Native contract fixture: the appointment moved to Thursday.")
                missing = engine.call("recall", {"handle": "appointment moved Thursday"})
                self.assertEqual(missing["error"], "handle_required")
                with self.assertRaises(NativeBoundaryViolation):
                    engine.call("gather", {"url": "https://example.test/issue"})
                events = read_jsonl(engine.audit)
                self.assertEqual(events[-1]["kind"], "boundary_refusal")
                self.assertIs(events[-1]["native_call_attempted"], False)
                starts = [e for e in events if e["kind"] == "engine_start"]
                self.assertEqual(starts[0]["tools"], catalog)
                self.assertEqual(starts[0]["initialize"], engine.initialize_result)
                self.assertEqual(starts[0]["model_tools"], engine.model_tools())
                self.assertEqual(json.loads((engine.run_dir / "mcp-tools.json").read_text())["tools"], catalog)
                engine.client._proc.kill(); engine.client._proc.wait(timeout=10)
                with self.assertRaisesRegex(VestigeMCPError, "not running"):
                    engine.call("recall", {"handle": "history"})


if __name__ == "__main__":
    unittest.main()
