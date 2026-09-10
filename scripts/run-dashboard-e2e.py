#!/usr/bin/env python3
"""Run browser tests against an owned disposable backend, never the user's store."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import select
import socket
import shutil
import sqlite3
from datetime import datetime, timezone
import subprocess
import tempfile
import time
from urllib.request import urlopen
import uuid


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop(process):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run(binary, log_dir, arguments):
    repo = Path(__file__).resolve().parents[1]
    log_dir.mkdir(parents=True, exist_ok=True)
    dashboard, mcp, frontend = port(), port(), port()
    token = uuid.uuid4().hex + uuid.uuid4().hex
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("VESTIGE_", "REDMINE_", "GITHUB_"))
    }
    env.update(
        VESTIGE_DASHBOARD_ENABLED="true",
        VESTIGE_DASHBOARD_PORT=str(dashboard),
        VESTIGE_AUTH_TOKEN=token,
        VESTIGE_HTTP_ALLOWED_ORIGINS=f"http://localhost:{frontend}",
        VESTIGE_API_TARGET=f"http://127.0.0.1:{dashboard}",
        VESTIGE_E2E_PORT=str(frontend),
        VESTIGE_E2E_MCP_URL=f"http://127.0.0.1:{mcp}/mcp",
        VESTIGE_E2E_AUTH_TOKEN=token,
        RUST_LOG="warn",
    )
    with tempfile.TemporaryDirectory(prefix="vestige-browser-e2e-") as temp:
        # Cargo can replace target/debug during another build. Pin the executable
        # before launching so its recorded digest identifies what actually ran.
        server_binary = Path(temp) / "fixture-server"
        shutil.copy2(binary.resolve(), server_binary)
        binary_digest = hashlib.sha256(server_binary.read_bytes()).hexdigest()
        with (log_dir / "browser-backend.log").open("w") as log:
            backend = subprocess.Popen(
                [str(server_binary), "--data-dir", temp, "--http-port", str(mcp)],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                start_new_session=True,
                bufsize=0,
            )
            browser = None
            try:
                for attempt in range(100):
                    if backend.poll() is not None:
                        raise RuntimeError("disposable backend exited")
                    try:
                        with urlopen(
                            env["VESTIGE_API_TARGET"] + "/api/stats", timeout=1
                        ) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("disposable dashboard did not become ready")
                sequence = 0

                def rpc(method, params):
                    nonlocal sequence
                    sequence += 1
                    backend.stdin.write(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": sequence,
                                "method": method,
                                "params": params,
                            }
                        ).encode()
                        + b"\n"
                    )
                    backend.stdin.flush()
                    deadline = time.monotonic() + 30
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not select.select([backend.stdout], [], [], remaining)[0]:
                            raise TimeoutError("fixture RPC did not return")
                        response = json.loads(backend.stdout.readline())
                        if response.get("method", "").startswith("notifications/"):
                            continue
                        if response.get("id") != sequence or "error" in response:
                            raise RuntimeError("fixture RPC failed")
                        break
                    result = response["result"]
                    if result.get("isError"):
                        raise RuntimeError("fixture tool failed")
                    return result

                rpc(
                    "initialize",
                    {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "disposable-browser-fixture",
                            "version": "1",
                        },
                    },
                )
                for i in range(12):
                    rpc(
                        "tools/call",
                        {
                            "name": "smart_ingest",
                            "arguments": {
                                "content": f"Vestige memory browser-fixture {i}: project timeout uses {25+i} milliseconds.",
                                "tags": ["browser-fixture", "timeout"],
                                "forceCreate": True,
                            },
                        },
                    )
                with urlopen(
                    env["VESTIGE_API_TARGET"] + "/api/memories?limit=20", timeout=5
                ) as response:
                    if len(json.load(response)["memories"]) < 12:
                        raise RuntimeError("fixture seed did not materialize")
                # Deterministic synthetic topology, independent of optional model readiness.
                with sqlite3.connect(Path(temp) / "vestige.db") as connection:
                    ids = [
                        row[0]
                        for row in connection.execute(
                            "SELECT id FROM knowledge_nodes ORDER BY id"
                        )
                    ]
                    now = datetime.now(timezone.utc).isoformat()
                    connection.executemany(
                        "INSERT INTO memory_connections(source_id,target_id,strength,link_type,created_at,last_activated,activation_count) VALUES(?,?,1.0,'shared_concepts',?,?,0)",
                        [(ids[0], other, now, now) for other in ids[1:]],
                    )
                rpc(
                    "tools/call",
                    {
                        "name": "recall",
                        "arguments": {
                            "query": "browser-fixture",
                            "runId": "v3-browser-fixture",
                            "concrete": True,
                        },
                    },
                )
                rpc("tools/call", {"name": "intention", "arguments": {
                    "action": "set", "description": "Review browser-fixture timeout before release",
                    "trigger": {"type": "context", "codebase": "browser-fixture"},
                    "priority": "high",
                }})
                with urlopen(env["VESTIGE_API_TARGET"] + "/api/intentions?status=active", timeout=5) as response:
                    if json.load(response)["total"] < 1:
                        raise RuntimeError("fixture intention did not materialize")
                with urlopen(env["VESTIGE_API_TARGET"] + "/api/receipts?run=v3-browser-fixture&limit=24", timeout=5) as response:
                    if not json.load(response).get("receipts"):
                        raise RuntimeError("fixture recall receipt did not materialize")
                review_mode = Path(temp) / "review_mode.json"
                review_mode.write_text('{"mode":"paranoid"}')
                rpc("tools/call", {"name": "smart_ingest", "arguments": {
                    "content": "Browser fixture pending review: change the staging timeout to 90 milliseconds.",
                    "forceCreate": True,
                }})
                review_mode.unlink()  # Exercise the actual automatic default after opt-in fixture setup.
                with urlopen(env["VESTIGE_API_TARGET"] + "/api/memory-prs?limit=20", timeout=5) as response:
                    if not json.load(response).get("prs"):
                        raise RuntimeError("fixture Memory PR did not materialize")
                with urlopen(env["VESTIGE_API_TARGET"] + "/api/graph", timeout=5) as response:
                    graph = json.load(response)
                    if len(graph["nodes"]) < 12 or len(graph["edges"]) < 11:
                        raise RuntimeError("fixture graph did not materialize")
                (log_dir / "fixture.json").write_text(json.dumps({
                    "schema": "vestige.browser-fixture.v1",
                    "synthetic": True,
                    "binary_sha256": binary_digest,
                    "memories": 12, "minimum_edges": 11, "minimum_intentions": 1,
                    "receipt_run": "v3-browser-fixture", "minimum_pending_memory_prs": 1,
                }, indent=2) + "\n")
                browser = subprocess.Popen(
                    ["pnpm", "exec", "playwright", "test", "--workers=1", *arguments],
                    cwd=repo / "apps/dashboard",
                    env=env,
                    start_new_session=True,
                )
                result = browser.wait()
                (log_dir / "result.json").write_text(json.dumps({"exit_code": result}) + "\n")
                return result
            finally:
                stop(browser)
                stop(backend)
                backend.stdin.close()
                backend.stdout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/vestige-mcp"))
    parser.add_argument("--log-dir", type=Path, required=True)
    args, extra = parser.parse_known_args()
    raise SystemExit(run(args.binary, args.log_dir, extra))
