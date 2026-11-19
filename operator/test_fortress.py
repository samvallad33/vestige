#!/usr/bin/env python3
"""Security Fortress (#836) OP-IMMORTAL verification tests."""

import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        work = Path(tmp) / "work"
        op_home = home / ".operator"
        for p in (home, work, op_home, op_home / "state"):
            p.mkdir(parents=True, exist_ok=True)
        (op_home / "mode").write_text("enforce\n", encoding="utf-8")

        os.environ["HOME"] = str(home)
        os.environ["USERPROFILE"] = str(home)
        os.environ["OPERATOR_HOME"] = str(op_home)

        spec = importlib.util.spec_from_file_location("operator_gate_fortress", ROOT / "operator-gate.py")
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)

        def run_pre(command: str, idx: int) -> tuple[int, str, str]:
            payload = {
                "hook_event_name": "PreToolUse",
                "session_id": f"fortress-session-{idx}",
                "tool_name": "Bash",
                "tool_input": {"command": command},
                "cwd": str(work),
            }
            old_in, old_err, old_out = sys.stdin, sys.stderr, sys.stdout
            sys.stdin = io.StringIO(json.dumps(payload))
            err, out = io.StringIO(), io.StringIO()
            sys.stderr, sys.stdout = err, out
            try:
                try:
                    code = g.hook_main(["--source", "claude"])
                except SystemExit as exc:
                    code = exc.code
            finally:
                sys.stdin, sys.stderr, sys.stdout = old_in, old_err, old_out
            return int(code or 0), out.getvalue(), err.getvalue()

        attacks = [
            "kill -9 $(pgrep vestige-mcp)",
            "pkill -f vestige-watchdog",
            "taskkill /F /IM vestige-mcp.exe",
            "Stop-Process -Name vestige -Force",
            "chattr -i ~/.vestige/store/00000001.strata",
            "rm -rf ~/.vestige/store",
            "del C:\\Users\\test\\.gemini\\config\\hooks.json",
        ]
        for idx, cmd in enumerate(attacks):
            code, out, err = run_pre(cmd, idx)
            combined = f"{out}\n{err}"
            if code != 2 or "STOPPED" not in combined:
                print(f"FAIL attack allowed or wrong code ({code}): {cmd}\n{combined}")
                return 1

        code, out, err = run_pre("echo hello-world", 99)
        if code != 0:
            print(f"FAIL benign command blocked ({code}): {out} {err}")
            return 1

    print("fortress=ALL_7_VECTORS_BLOCKED benign=ALLOW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
