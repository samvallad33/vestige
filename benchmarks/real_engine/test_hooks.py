"""Hook lifecycle/transport self-tests; these never produce benchmark scores."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from .engine import HookedRealEngine
from .hooks import HookViolation,digest,merge_fragments,read_audit

REPO=Path(__file__).resolve().parents[2]


def _test_audit(path):
    # Temporary source-pin fixture for adapter tests only. Scored runs require
    # the separately reviewed runs/hook-audit.json, not this fixture.
    pins={str(p.relative_to(REPO)):digest(p) for p in (REPO/"hooks").iterdir() if p.is_file()}
    path.write_text(json.dumps({"audit_complete":True,"source_pins":pins,"test_fixture_only":True}))
    return path


class HookContracts(unittest.TestCase):
    def test_memory_qa_selects_only_source_registered_memory_hooks(self):
        events,wired=merge_fragments(REPO,profile="memory_qa")
        self.assertEqual([x["command"] for x in events["SessionStart"]],["vestige hook session-start"])
        self.assertEqual([x["command"] for x in events["UserPromptSubmit"]],["$HOME/.claude/hooks/vestige-pulse-daemon.sh"])
        self.assertEqual(events["PreToolUse"],[])
        self.assertEqual(events["PostToolUse"],[])
        self.assertEqual(events["Stop"],[])
        self.assertEqual(len(wired),2)
        self.assertTrue(all(x["fragment"].startswith("hooks/settings.") for x in wired))

    def test_unknown_hook_profile_fails_loudly(self):
        with self.assertRaisesRegex(HookViolation,"unknown hook profile"):
            merge_fragments(REPO,profile="disable-everything")

    def test_all_fragment_commands_and_triggers_preserved(self):
        events,wired=merge_fragments(REPO)
        self.assertEqual(len(events["UserPromptSubmit"]),7)
        self.assertEqual(len(events["Stop"]),4)
        self.assertEqual(len(events["SessionStart"]),2)
        self.assertEqual(events["PreToolUse"],[])
        self.assertTrue(any(x["command"]=="vestige hook session-start" for x in wired))
        self.assertTrue(any("preflight-swarm.sh" in x["command"] for x in wired))

    def test_missing_or_drifted_source_audit_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"audit.json"
            path.write_text(json.dumps({"audit_complete":False}))
            with self.assertRaisesRegex(HookViolation,"incomplete"):read_audit(REPO,path)
            _test_audit(path)
            value=json.loads(path.read_text());value["source_pins"]["hooks/preflight-swarm.sh"]="0"*64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(HookViolation,"changed after audit"):read_audit(REPO,path)

    def test_scored_class_cannot_disable_hooks(self):
        with self.assertRaisesRegex(ValueError,"cannot disable"):
            HookedRealEngine(REPO,"unused",hooks_enabled=False)


@unittest.skipUnless((REPO/"target/release/vestige").exists(),"real native CLI build required")
class NativeHookTests(unittest.TestCase):
    def test_memory_qa_unchanged_hooks_read_only_this_native_store(self):
        with tempfile.TemporaryDirectory(dir=REPO/"runs") as tmp:
            root=Path(tmp);audit=_test_audit(root/"audit.json")
            engine=HookedRealEngine(REPO,root/"engine",hook_audit_path=audit)
            engine.hook_profile="memory_qa"
            with engine:
                self.assertEqual(engine.hooks.profile,"memory_qa")
                self.assertEqual(len(engine.hooks.wired),2)
                self.assertEqual(engine.begin_turn("What does the current record say?"),[])
                engine.call("smart_ingest",{"content":"The current fixture uses a blue folder.","node_type":"decision","tags":["pinned"]})
                output=engine.hooks.event("SessionStart",{"source":"resume"})
                self.assertIn("blue folder","\n".join(output["context"]))
                hookrows=[json.loads(line) for line in engine.hooks.log_path.read_text().splitlines()]
                self.assertFalse(any(x.get("kind")=="model_bridge_request" for x in hookrows))
                self.assertFalse(any(x.get("event") in {"PreToolUse","PostToolUse","Stop"} for x in hookrows))
                self.assertTrue(any(x.get("kind")=="hook_mcp_request" for x in hookrows))
                self.assertEqual(engine.hooks.native_status["/api/changelog"]["status"],501)
                self.assertNotIn("/api/deep_reference",engine.hooks.native_status)

    def test_raw_prompt_preserved_and_ambient_evidence_never_injected(self):
        with tempfile.TemporaryDirectory(dir=REPO/"runs") as tmp:
            root=Path(tmp);audit=_test_audit(root/"audit.json")
            ambient=root/"ambient";ambient.mkdir()
            (ambient/"private.md").write_text("AMBIENT_PROFILE_SENTINEL")
            stage=ambient/"staged.json";stage.write_text('{"evidence":"AMBIENT_PROFILE_SENTINEL"}')
            seen=[]
            def fixture_model(payload):
                # Explicit no-inference transport fixture, never a score.
                seen.append(payload)
                return {"choices":[{"message":{"role":"assistant","content":"EMPTY"}}]}
            with patch.dict("os.environ",{"VESTIGE_MEMORY_DIR":str(ambient),"VESTIGE_SANHEDRIN_STAGE_FILE":str(stage),"VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER":"1"}):
                with HookedRealEngine(REPO,root/"engine",hook_audit_path=audit,hook_model_callback=fixture_model) as engine:
                    marker=root/"must-not-exist"
                    raw=f"Raw untrusted prompt $(touch {marker}) with Unicode café."
                    context=engine.begin_turn(raw)
                    self.assertTrue(seen)
                    self.assertIn(raw,seen[0]["messages"][0]["content"])
                    self.assertEqual(seen[0]["model"],"gpt-5.4-nano")
                    self.assertNotIn("AMBIENT_PROFILE_SENTINEL",json.dumps(seen)+json.dumps(context))
                    self.assertFalse(marker.exists())
                    self.assertNotIn("VESTIGE_SANHEDRIN_STAGE_FILE",engine.hooks.env)
                    self.assertEqual(engine.hooks.env["VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER"],"0")
                    self.assertFalse((engine.hooks.home/".vestige/unused-unwired-memory").exists())

    def test_actual_source_hooks_isolated_native_lifecycle_and_transport(self):
        with tempfile.TemporaryDirectory(dir=REPO/"runs") as tmp:
            root=Path(tmp);audit=_test_audit(root/"audit.json")
            with HookedRealEngine(REPO,root/"engine",hook_audit_path=audit) as engine:
                self.assertTrue(engine.identity["hooks"]["enabled"])
                self.assertEqual(engine.identity["fresh_store_nodes"],0)
                self.assertEqual(engine.hooks.env["HOME"],str(root/"engine/hooks-runtime/home"))
                self.assertNotIn("OPENAI_API_KEY",engine.hooks.env)
                self.assertNotIn("VESTIGE_SANHEDRIN_STAGE_FILE",engine.hooks.env)
                self.assertFalse((Path(engine.hooks.env["OPERATOR_HOME"])/"receipts").exists())
                for source in (REPO/"hooks").iterdir():
                    if source.is_file():self.assertEqual(digest(source),digest(engine.hooks.home/".claude/hooks"/source.name))
                context=engine.begin_turn("Hello")
                self.assertIsInstance(context,list)
                self.assertFalse(engine.hooks.cwd.is_relative_to(engine.repo))
                self.assertFalse(any("CWD STATE" in text for text in context))
                self.assertFalse(any((parent/".git").exists() for parent in (engine.hooks.cwd,*engine.hooks.cwd.parents)))
                batch_text=[f"Hook batch fixture {i}: raw untrusted note." for i in range(33)]
                before=len([x for x in engine.hooks.log_path.read_text().splitlines() if '"event": "PostToolUse"' in x])
                batch=engine.ingest_history(batch_text)
                after=len([x for x in engine.hooks.log_path.read_text().splitlines() if '"event": "PostToolUse"' in x])
                self.assertEqual(after-before,2)
                self.assertEqual(len(batch),33)
                for i,record in enumerate(batch):
                    actual=engine.call("memory",{"action":"get","id":record["nodeId"]})["node"]
                    self.assertEqual(actual["content"],batch_text[i])
                    self.assertEqual(actual["tags"],["history",f"session-{i}"])
                writes=engine.ingest_history(["Hook self-test record; no scored task."])
                decision=engine.call("smart_ingest",{"content":"Use precise dates in this isolated self-test.","node_type":"decision","tags":["pinned"]})
                rules=engine.hooks.event("SessionStart",{"source":"resume"})
                self.assertIn("Use precise dates", "\n".join(rules["context"]))
                # The native CLI reads through the envelope proxy -> same child
                # stdio. Its alias directory contains no log memory segments.
                self.assertEqual(list((engine.hooks.cli_data/"log").iterdir()),[])
                hookrows=[json.loads(x) for x in engine.hooks.log_path.read_text().splitlines()]
                self.assertTrue(any(x.get("kind")=="hook_mcp_request" and x.get("request",{}).get("params",{}).get("name")=="recall" for x in hookrows))
                # Hook subprocesses forward stdio MCP to this existing child.
                request={"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"memory","arguments":{"action":"get","id":writes[0]["nodeId"]}}}
                req=urllib.request.Request(engine.hooks.control_url+"/mcp",data=json.dumps(request).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+engine.hooks.bridge_token})
                with urllib.request.urlopen(req) as response:result=json.load(response)
                self.assertEqual(result["id"],9)
                self.assertIn("Hook self-test record",json.dumps(result))
                statuses=engine.identity["hooks"]["native_endpoint_capabilities"]
                self.assertEqual(statuses["/api/health"]["status"],200)
                # Actual unavailable native legacy APIs remain unavailable.
                self.assertEqual(statuses["/api/changelog"]["status"],501)
                self.assertEqual(statuses["/api/deep_reference"]["status"],401)
                result=engine.finish_turn([{"role":"user","content":"Hello"},{"role":"assistant","content":"Hello."}])
                self.assertFalse(result["blocked"])
                # A legacy Haiku request may not escape to ambient CLI/model.
                with self.assertRaisesRegex(HookViolation,"nano model callback"):
                    engine.begin_turn("A longer neutral message that exercises the lateral hook transport.")


if __name__=="__main__":unittest.main()
