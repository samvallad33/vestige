#!/usr/bin/env python3
"""Real-stdio proof that `dedup` action `undo` lands on the Strata log.

The tool advertises `undo`. The call must complete, the undone write must
disappear from reads, and `strata-verify` must accept the append-only log.
"""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile


def main() -> None:
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "target/debug/vestige-mcp")
    verify = Path(sys.argv[2] if len(sys.argv) > 2 else "target/debug/strata-verify")
    with tempfile.TemporaryDirectory(prefix="vestige-dedup-undo-") as temp:
        store = Path(temp) / "store"
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("VESTIGE_", "REDMINE_", "GITHUB_"))
        }
        env.update(
            VESTIGE_DASHBOARD_ENABLED="false",
            VESTIGE_HTTP_ENABLED="false",
            RUST_LOG="error",
        )
        seq = 0
        proc = None

        def spawn() -> None:
            nonlocal proc
            proc = subprocess.Popen(
                [str(binary.resolve()), "--no-http", "--data-dir", str(store)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
            )

        def rpc(method, params):
            nonlocal seq
            seq += 1
            request = {"jsonrpc": "2.0", "id": seq, "method": method, "params": params}
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            while True:
                if not select.select([proc.stdout], [], [], 60)[0]:
                    raise TimeoutError(method)
                line = proc.stdout.readline()
                if not line:
                    err = proc.stderr.read() if proc.stderr else ""
                    raise RuntimeError(f"MCP process exited during {method}: {err}")
                response = json.loads(line)
                if response.get("id") == seq:
                    if "error" in response:
                        raise AssertionError(response)
                    return response["result"]

        def tool(name, args):
            result = rpc("tools/call", {"name": name, "arguments": args})
            if result.get("isError"):
                raise AssertionError(result)
            body = result.get("structuredContent")
            if body is None:
                body = json.loads(result["content"][0]["text"])
            text = json.dumps(body)
            for needle in ("pending_strata", "not implemented", "not admitted"):
                if needle in text:
                    raise AssertionError(body)
            return body

        def handshake() -> None:
            rpc(
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "dedup-undo-stdio", "version": "1"},
                },
            )
            assert proc.stdin is not None
            proc.stdin.write(
                '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            )
            proc.stdin.flush()

        def stop() -> None:
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)

        spawn()
        try:
            handshake()
            catalog = rpc("tools/list", {})["tools"]
            dedup = next(tool for tool in catalog if tool["name"] == "dedup")
            actions = dedup["inputSchema"]["properties"]["action"]["enum"]
            assert "undo" in actions, actions
            assert "undo" in (dedup.get("description") or "")

            listed = tool("dedup", {"action": "undo"})
            assert listed["operations"] == []
            assert listed["totalOperations"] == 0

            marker = "UNDO_STDIO_MARKER_SHOULD_VANISH"
            created = tool(
                "smart_ingest",
                {"content": marker, "forceCreate": True, "tags": ["undo-src"]},
            )
            node_id = created["nodeId"]
            assert created["success"] is True
            visible = tool("memory", {"action": "get", "id": node_id})
            assert visible["found"] is True
            assert visible["node"]["content"] == marker

            reflog = tool("dedup", {"action": "undo"})
            op = next(
                item
                for item in reflog["operations"]
                if item["opType"] == "ingest"
                and item["status"] == "applied"
                and node_id in item["affectedIds"]
            )
            undone = tool(
                "dedup", {"action": "undo", "operation_id": op["operationId"]}
            )
            assert undone["status"] == "reverted"
            assert undone["revertedOperationId"] == op["operationId"]
            assert node_id in undone["affectedIds"]

            hidden = tool("memory", {"action": "get", "id": node_id})
            assert hidden["found"] is False
            assert marker not in json.dumps(hidden)
            timeline = tool("memory_status", {"view": "timeline"})
            assert marker not in json.dumps(timeline)
            recalled = tool("recall", {"handle": node_id})
            assert marker not in json.dumps(recalled)
            kept = tool(
                "smart_ingest",
                {"content": "UNDO_STDIO_KEPT", "forceCreate": True},
            )
            kept_get = tool("memory", {"action": "get", "id": kept["nodeId"]})
            assert kept_get["found"] is True
            assert kept_get["node"]["content"] == "UNDO_STDIO_KEPT"
        finally:
            stop()

        again = None
        spawn()
        try:
            handshake()
            again = tool("memory", {"action": "get", "id": node_id})
            assert again["found"] is False, again
            kept_again = tool("memory", {"action": "get", "id": kept["nodeId"]})
            assert kept_again["node"]["content"] == "UNDO_STDIO_KEPT"
        finally:
            stop()

        log_dir = store / "log"
        report = subprocess.run(
            [str(verify.resolve()), str(log_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert report.returncode == 0, report.stdout + report.stderr
        assert "OK" in report.stdout, report.stdout
        bad = [
            str(path)
            for path in store.rglob("*")
            if path.is_file()
            and path.name.lower().endswith((".sqlite", ".sqlite3", ".db", ".db-wal", ".db-shm"))
        ]
        assert not bad, bad
        print("PASS dedup undo over stdio; undone write hidden; strata-verify OK")


if __name__ == "__main__":
    main()
