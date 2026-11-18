import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HOOK = Path(__file__).with_name("save-gate.py")


def run_hook(
    event: str,
    raw_input: str,
    env: dict[str, str],
) -> tuple[int, dict | None]:
    proc = subprocess.run(
        [sys.executable, str(HOOK), "--event", event],
        input=raw_input,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    out = proc.stdout.strip()
    return proc.returncode, (json.loads(out) if out else None)


def run_hook_json(event: str, payload: dict, env: dict[str, str]) -> dict | None:
    rc, parsed = run_hook(event, json.dumps(payload), env)
    if rc != 0:
        raise AssertionError(f"hook exited {rc}")
    return parsed


class TestSaveGateHooks(unittest.TestCase):
    def test_edit_blocks_git_commit_and_stop_until_remember_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "VESTIGE_REFLEX_DIR": tmp}
            sid = "test-session-1"

            self.assertIsNone(
                run_hook_json("UserPromptSubmit", {"session_id": sid, "prompt": "fix bug"}, env)
            )
            session_file = Path(tmp) / f"session-{sid}.json"
            self.assertTrue(session_file.exists())
            session_data = json.loads(session_file.read_text(encoding="utf-8"))
            self.assertEqual(len(session_data["prompt_hash"]), 64)

            self.assertIsNone(
                run_hook_json(
                    "PostToolUse",
                    {
                        "session_id": sid,
                        "tool_name": "Edit",
                        "tool_input": {"file_path": "crates/vestige-core/src/lib.rs"},
                        "tool_response": {},
                    },
                    env,
                )
            )

            pre = run_hook_json(
                "PreToolUse",
                {
                    "session_id": sid,
                    "tool_name": "Bash",
                    "tool_input": {"command": "git commit -m 'fix'"},
                },
                env,
            )
            self.assertIsNotNone(pre)
            self.assertEqual(pre["hookSpecificOutput"]["permissionDecision"], "deny")

            stop_1 = run_hook_json("Stop", {"session_id": sid, "stop_hook_active": False}, env)
            self.assertIsNotNone(stop_1)
            self.assertEqual(stop_1["decision"], "block")
            self.assertIn("crates/vestige-core/src/lib.rs", stop_1["reason"])

            self.assertIsNone(
                run_hook_json(
                    "PostToolUse",
                    {
                        "session_id": sid,
                        "tool_name": "mcp__vestige__remember",
                        "tool_input": {"action": "save", "content": "Fixed lib.rs bug"},
                        "tool_response": {"ok": True},
                    },
                    env,
                )
            )

            self.assertIsNone(
                run_hook_json(
                    "PreToolUse",
                    {
                        "session_id": sid,
                        "tool_name": "Bash",
                        "tool_input": {"command": "git commit -m 'fix'"},
                    },
                    env,
                )
            )
            self.assertIsNone(
                run_hook_json("Stop", {"session_id": sid, "stop_hook_active": False}, env)
            )

            events_file = Path(tmp) / f"strata-events-{sid}.jsonl"
            kinds = [
                json.loads(line)["kind"]
                for line in events_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIn("edit_anchor", kinds)
            self.assertIn("remember_save", kinds)
            self.assertIn("operator_stop", kinds)

    def test_stop_block_cap_of_two_holds_per_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "VESTIGE_REFLEX_DIR": tmp}
            sid = "test-session-cap"

            run_hook_json("UserPromptSubmit", {"session_id": sid, "prompt": "edit file"}, env)
            run_hook_json(
                "PostToolUse",
                {
                    "session_id": sid,
                    "tool_name": "Write",
                    "tool_input": {"file_path": "src/app.rs"},
                    "tool_response": {},
                },
                env,
            )

            b1 = run_hook_json("Stop", {"session_id": sid, "stop_hook_active": False}, env)
            self.assertIsNotNone(b1)
            self.assertEqual(b1["decision"], "block")

            b2 = run_hook_json("Stop", {"session_id": sid, "stop_hook_active": True}, env)
            self.assertIsNotNone(b2)
            self.assertEqual(b2["decision"], "block")

            b3 = run_hook_json("Stop", {"session_id": sid, "stop_hook_active": True}, env)
            self.assertIsNone(b3)

            # A new turn resets the per-turn block cap.
            run_hook_json("UserPromptSubmit", {"session_id": sid, "prompt": "next turn"}, env)
            b4 = run_hook_json("Stop", {"session_id": sid, "stop_hook_active": True}, env)
            self.assertIsNotNone(b4)
            self.assertEqual(b4["decision"], "block")

    def test_bad_json_input_exits_zero_with_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "VESTIGE_REFLEX_DIR": tmp}
            for event in ("UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"):
                for bad in ("{not valid json", "", "[]", "null"):
                    rc, parsed = run_hook(event, bad, env)
                    self.assertEqual(rc, 0)
                    self.assertIsNone(parsed)


    def test_directive_and_missing_sha_block_stop_until_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "VESTIGE_REFLEX_DIR": tmp}
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "lib.rs").write_text("fn add() {}\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=repo, check=True)
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            sid = "cite-session"
            prompt = run_hook_json(
                "UserPromptSubmit",
                {
                    "session_id": sid,
                    "cwd": str(repo),
                    "prompt": "from now on never edit benchmarks/real_engine",
                },
                env,
            )
            self.assertIn("from now on", prompt["hookSpecificOutput"]["additionalContext"])
            self.assertNotIn("[REQUIRED]", prompt["hookSpecificOutput"]["additionalContext"])
            blocked = run_hook_json(
                "Stop",
                {
                    "session_id": sid,
                    "cwd": str(repo),
                    "stop_hook_active": False,
                    "last_assistant_message": "shipped commit:deadbeefdeadbeef and `src/missing.rs`",
                },
                env,
            )
            reason = blocked["reason"]
            self.assertEqual(blocked["decision"], "block")
            self.assertIn("from now on", reason)
            self.assertIn("deadbeefdeadbeef", reason)
            self.assertIn("src/missing.rs", reason)
            self.assertIsNone(
                run_hook_json(
                    "PostToolUse",
                    {
                        "session_id": sid,
                        "cwd": str(repo),
                        "tool_name": "remember",
                        "tool_input": {"action": "save", "content": "recorded"},
                        "tool_response": {"ok": True},
                    },
                    env,
                )
            )
            real = run_hook_json(
                "Stop",
                {
                    "session_id": sid,
                    "cwd": str(repo),
                    "stop_hook_active": True,
                    "last_assistant_message": f"shipped `{head}` and `./lib.rs`",
                },
                env,
            )
            self.assertIsNone(real)

    def test_worktree_remove_exits_2_while_dirty(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "VESTIGE_REFLEX_DIR": tmp}
            repo = Path(tmp) / "wt"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "lib.rs").write_text("fn add() {}\n", encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=repo, check=True)
            (repo / "dirty.rs").write_text("x\n", encoding="utf-8")
            rc, _parsed = run_hook(
                "WorktreeRemove",
                json.dumps({"session_id": "wt", "worktree_path": str(repo)}),
                env,
            )
            self.assertEqual(rc, 2)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "clean"], cwd=repo, check=True)
            rc, parsed = run_hook(
                "WorktreeRemove",
                json.dumps({"session_id": "wt", "worktree_path": str(repo)}),
                env,
            )
            self.assertEqual(rc, 0)
            self.assertIsNone(parsed)

    def test_multiedit_tracks_all_files_until_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "VESTIGE_REFLEX_DIR": tmp}
            sid = "test-multiedit"
            run_hook_json("UserPromptSubmit", {"session_id": sid, "prompt": "refactor two files"}, env)
            run_hook_json(
                "PostToolUse",
                {
                    "session_id": sid,
                    "tool_name": "MultiEdit",
                    "tool_input": {
                        "edits": [
                            {"file_path": "src/a.rs", "start_line": 1, "end_line": 4},
                            {"file_path": "src/b.rs", "start_line": 10, "end_line": 15},
                        ]
                    },
                    "tool_response": {},
                },
                env,
            )
            stop = run_hook_json("Stop", {"session_id": sid, "stop_hook_active": False}, env)
            self.assertEqual(stop["decision"], "block")
            self.assertIn("src/a.rs", stop["reason"])
            self.assertIn("src/b.rs", stop["reason"])
            run_hook_json(
                "PostToolUse",
                {
                    "session_id": sid,
                    "tool_name": "mcp__vestige__remember",
                    "tool_input": {"action": "save", "content": "Anchored both refactored files"},
                    "tool_response": {"ok": True},
                },
                env,
            )
            cleared = run_hook_json("Stop", {"session_id": sid, "stop_hook_active": False}, env)
            self.assertIsNone(cleared)


if __name__ == "__main__":
    unittest.main()
