#!/usr/bin/env python3
"""Exercise evidence-aware intentions through an isolated real MCP subprocess.

Build first: cargo build -p vestige-mcp --no-default-features --bin vestige-mcp
Run: python3 scripts/demo-intentions.py --binary target/debug/vestige-mcp
All product observations are synthetic. No user database or external action is used.
"""

import argparse
import hashlib
import json
import pathlib
import selectors
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    transcript = []
    with tempfile.TemporaryDirectory(prefix="vestige-intention-demo-") as fixture:
        with open(pathlib.Path(fixture) / "stderr.log", "w+") as log:
            process = subprocess.Popen(
                [str(binary), "--data-dir", fixture], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
            )
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            next_id = 0

            def rpc(method, params):
                nonlocal next_id
                next_id += 1
                request = {"jsonrpc": "2.0", "id": next_id, "method": method, "params": params}
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if not selector.select(max(0, deadline - time.monotonic())):
                        break
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError("MCP subprocess closed stdout")
                    response = json.loads(line)
                    if response.get("id") == next_id:
                        transcript.append({"request": request, "response": response})
                        if "error" in response:
                            raise RuntimeError(response["error"])
                        result = response["result"]
                        if result.get("isError"):
                            raise RuntimeError(result)
                        return result
                raise TimeoutError(f"MCP response timed out for {method}")

            def intention_call(arguments):
                result = rpc("tools/call", {"name": "intention", "arguments": arguments})
                if "structuredContent" in result:
                    return result["structuredContent"]
                return json.loads(next(c["text"] for c in result["content"] if c["type"] == "text"))

            def graph(command, at="2026-10-01T09:00:00Z"):
                return intention_call({"action": "graph", "scope": "demo", "at": at, "command": command})

            try:
                rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "intention-fixture", "version": "1"}})
                tools = rpc("tools/list", {})["tools"]
                intention = next(t for t in tools if t["name"] == "intention")
                assert "graph" in intention["inputSchema"]["properties"]["action"]["enum"]
                requirement = {"type": "evidence", "id": "fits-bag", "source_key": "spec:projector",
                    "condition": {"op": "equals", "value": True}}
                graph({"action": "plan", "id": "projector", "description": "Synthetic: buy the selected projector",
                    "requirements": [requirement], "min_attention_interval_seconds": 0})
                graph({"action": "plan", "id": "unrelated", "description": "Synthetic unrelated intention",
                    "requirements": []})
                graph({"action": "observe", "event_id": "spec-1", "source_key": "spec:projector",
                    "source_revision": 1, "value": True})
                graph({"action": "complete", "id": "projector", "expected_version": 1,
                    "basis": {"type": "evidence", "requirement_ids": ["fits-bag"]}})
                graph({"action": "observe", "event_id": "spec-2", "source_key": "spec:projector",
                    "source_revision": 2, "value": False}, "2026-10-02T09:00:00Z")
                explanation = graph({"action": "explain", "id": "projector"}, "2026-10-02T09:00:00Z")
                assert "disputed" in json.dumps(explanation).lower(), explanation
                before = graph({"action": "replay"})
                graph({"action": "observe", "event_id": "spec-2", "source_key": "spec:projector",
                    "source_revision": 2, "value": False}, "2026-10-02T09:00:00Z")
                duplicate = graph({"action": "replay"})
                assert before["state_digest"] == duplicate["state_digest"]
                graph({"action": "cancel", "id": "projector", "expected_version": 1, "reason": "Synthetic cancellation"}, "2026-10-02T09:00:00Z")
                replay = graph({"action": "replay"})
                assert replay["matched"] is True
                intention_call({"action": "set", "description": "Synthetic fortnightly projector reminder",
                    "trigger": {"type": "recurring", "at": "2026-10-16T16:00:00Z", "recurrence": "every other week"}})
                def check(at):
                    return intention_call({"action": "check", "context": {"current_time": at}})
                first = check("2026-10-16T16:00:00Z")
                assert len(first["triggered"]) == 1, first
                assert not check("2026-10-16T16:00:00Z")["triggered"]
                assert not check("2026-10-23T16:00:00Z")["triggered"]
                second = check("2026-10-30T16:00:00Z")
                assert len(second["triggered"]) == 1, second
                assert second["triggered"][0]["reminderCount"] == 2
                result = {"fixture": "synthetic projector through MCP stdio", "passed": True,
                    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    "replay": replay, "transcript": transcript,
                    "boundary": "Local synthetic behavior; no live facts, purchase, scheduler delivery, or competitive result."}
                encoded = json.dumps(result, indent=2)
                if args.output:
                    args.output.write_text(encoded + "\n")
                else:
                    print(encoded)
            finally:
                selector.close()
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
