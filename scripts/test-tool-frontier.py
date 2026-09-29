#!/usr/bin/env python3
"""Stdio contracts for the Strata-backed vestige-mcp binary.

Discovery stays exact. Writes go through the gate. Similarity (embeddings,
cosine, BM25, FTS, Jaccard, keyword or name match) is an error. A link
exists only when the log recorded an edge. No SQLite file is created.
"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile


def run(binary, output):
    transcript = []
    coverage = []
    with tempfile.TemporaryDirectory(prefix="vestige-tool-frontier-") as temp:
        root = Path(temp)
        store = root / "store"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VESTIGE_", "REDMINE_", "GITHUB_"))}
        env.update(VESTIGE_DASHBOARD_ENABLED="false", VESTIGE_HTTP_ENABLED="false", RUST_LOG="error")
        proc = None
        seq = 0

        def spawn():
            nonlocal proc
            proc = subprocess.Popen(
                [str(binary.resolve()), "--no-http", "--data-dir", str(store)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1,
            )

        def rpc(method, params):
            nonlocal seq
            seq += 1
            request = {"jsonrpc": "2.0", "id": seq, "method": method, "params": params}
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            while True:
                if not select.select([proc.stdout], [], [], 60)[0]:
                    raise TimeoutError(method)
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("MCP process exited")
                response = json.loads(line)
                if response.get("id") == seq:
                    transcript.append({"request": request, "response": response})
                    assert "error" not in response, response
                    return response["result"]

        def tool(name, args, error=False):
            result = rpc("tools/call", {"name": name, "arguments": args})
            assert bool(result.get("isError")) == error, result
            coverage.append({
                "tool": name,
                "selector": args.get("action", args.get("mode", args.get("view", "default"))),
                "outcome": "expected_error" if error else "success",
            })
            return result.get("structuredContent") or json.loads(result["content"][0]["text"])

        def typed(name, args, needle):
            body = tool(name, args, error=True)
            assert needle in body["error"], body
            return body

        def handshake():
            rpc("initialize", {
                "protocolVersion": "2025-11-25", "capabilities": {},
                "clientInfo": {"name": "tool-frontier-fixture", "version": "1"},
            })
            proc.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            proc.stdin.flush()

        def passed(name):
            print("PASS", name, flush=True)

        def assert_no_sqlite():
            bad = []
            if store.exists():
                for path in store.rglob("*"):
                    name = path.name.lower()
                    if path.is_file() and (
                        name.endswith(".sqlite") or name.endswith(".sqlite3")
                        or name.endswith(".db") or name.endswith(".db-wal")
                        or name.endswith(".db-shm")
                    ):
                        bad.append(str(path))
            assert not bad, bad

        spawn()
        try:
            handshake()
            catalog = rpc("tools/list", {})["tools"]
            names = [x["name"] for x in catalog]
            assert len(names) == 18, names
            guide = tool("memory_status", {"view": "tools"})["tools"]
            assert [x["name"] for x in guide] == names
            for entry, definition in zip(guide, catalog):
                for selector in ("action", "mode", "view"):
                    values = definition["inputSchema"].get("properties", {}).get(selector, {}).get("enum")
                    if values is not None:
                        assert entry["selectors"][selector]["values"] == values
                detail = tool("memory_status", {"view": "tools", "tool": entry["name"]})
                full = detail["tools"][0]["inputSchema"]
                assert full.get("type") == definition["inputSchema"].get("type")
                assert len(json.dumps(full)) >= len(json.dumps(definition["inputSchema"]))
                for selector in ("action", "mode", "view"):
                    compact_enum = definition["inputSchema"].get("properties", {}).get(selector, {}).get("enum")
                    if compact_enum is not None:
                        full_enum = full.get("properties", {}).get(selector, {}).get("enum")
                        assert full_enum == compact_enum, (
                            f"{entry['name']}: {selector} enum drifted under compaction"
                        )
            for invalid in ("search", "", 12):
                tool("memory_status", {"view": "tools", "tool": invalid}, error=True)
            annotations = {x["name"]: x["annotations"] for x in catalog}
            assert annotations["recall"]["readOnlyHint"] is False
            assert annotations["recall"]["idempotentHint"] is False
            assert annotations["suppress"]["idempotentHint"] is False
            passed("all installed tool and action definitions match progressive discovery")

            typed("maintain", {"action": "consolidate", "phase": "embeddings", "batchSize": 2},
                  "phase must be all, lifecycle or logs")
            tool("maintain", {"action": "consolidate", "batchSize": 2}, error=True)
            tool("maintain", {"action": "consolidate", "phase": "embeddings", "batchSize": 101}, error=True)
            passed("embedding maintenance is refused; it is not a Strata operation")
            for phase in ("lifecycle", "logs"):
                page = tool("maintain", {"action": "consolidate", "phase": phase, "batchSize": 2})
                assert page["dryRun"] is True and page["hasMore"] is False and page["selected"] == 0
            tool("maintain", {"action": "consolidate", "phase": "logs", "after": "invalid"}, error=True)
            passed("lifecycle/log maintenance previews empty pages and rejects invalid controls")

            marker = "STRATA_FIXTURE_EXACT_HANDLE"
            created = tool("smart_ingest", {"content": marker, "forceCreate": True, "tags": ["fixture-old"]})
            node_id = created["nodeId"]
            assert node_id.startswith("mem-") and created["success"] is True
            got = tool("memory", {"action": "get", "id": node_id})
            assert marker in json.dumps(got)
            tool("memory", {"action": "get_batch", "ids": [node_id]})
            tool("memory", {"action": "state", "id": node_id})
            receipt = tool("receipt", {"action": "get", "receipt_id": node_id})
            assert "receipt" in receipt
            typed("recall", {"query": marker}, "similarity_disabled")
            handle = tool("recall", {"handle": node_id})
            assert marker in json.dumps(handle) and handle["exact"] is True
            passed("ingest, exact get, write receipt, and handle recall; query recall is refused")

            proc.terminate()
            proc.wait(timeout=10)
            assert_no_sqlite()
            spawn()
            handshake()
            again = tool("memory", {"action": "get", "id": node_id})
            assert marker in json.dumps(again)
            assert_no_sqlite()
            passed("empty-dir restart keeps the node and creates no sqlite file")

            typed("recall", {"mode": "reason", "query": marker}, "similarity_disabled")
            typed("recall", {"mode": "contradictions"}, "similarity_disabled")
            typed("receipt", {"action": "replay", "receipt_id": node_id, "withheld_slots": []}, "pending_strata")
            typed("memory", {"action": "promote", "id": node_id, "reason": "fixture"}, "pending_strata")
            typed("memory", {"action": "edit", "id": node_id, "content": "edited"}, "pending_strata")
            typed("purge", {"id": node_id, "confirm": True}, "pending_strata")
            context = tool("codebase", {"action": "get_context", "codebase": "fixture"})
            assert marker not in json.dumps(context)
            typed("project", {"action": "preview"}, "pending_strata")
            typed("intention", {"action": "set", "description": "Synthetic reminder",
                                "trigger": {"type": "time", "at": "2020-01-01T00:00:00Z"}}, "pending_strata")
            tool("source_sync", {"source": "gitlab", "repo": "a/b"}, error=True)
            for view in ("health", "retention", "timeline", "changelog", "stats", "coverage"):
                tool("memory_status", {"view": view})
            score = tool("maintain", {"action": "importance_score", "content": "Synthetic fixture design decision"})
            assert isinstance(score.get("composite"), (int, float))
            gc = tool("maintain", {"action": "gc"})
            assert gc.get("dryRun", gc.get("dry_run")) is True
            assert gc.get("deleted", gc.get("candidateCount", 0)) == 0 or gc.get("candidateCount") == 0
            preview = tool("dedup", {"action": "tag_rename", "source_tag": "fixture-old", "target_tag": "fixture-new"})
            assert preview.get("wouldWrite") is False
            recent = tool("graph", {"action": "recent", "limit": 5})
            assert recent.get("events", recent.get("compositions", [])) == [] or recent.get("count", 0) == 0 or "events" in recent
            never = tool("graph", {"action": "never_composed", "limit": 5})
            assert never["scope"] == "user" and never["globalNoveltyVerified"] is False
            typed("session_start", {"queries": [marker], "include_predictions": False, "include_intentions": False}, "similarity_disabled")
            before_state = tool("memory", {"action": "state", "id": node_id})
            baseline_retrieval = before_state["components"]["retrievalStrength"]
            suppressed = tool("suppress", {"id": node_id, "reason": "fixture"})
            assert suppressed["success"] is True and suppressed["id"] == node_id
            assert suppressed["suppressionCount"] == 1
            receipt_id = suppressed["receiptId"]
            assert receipt_id.startswith("eff-")
            assert suppressed["receipt"]["receipt_id"] == receipt_id
            assert suppressed["receipt"]["mutations"][0]["kind"] == "suppressed"
            loaded = tool("receipt", {"action": "get", "receipt_id": receipt_id})
            assert loaded["receipt"]["receipt_id"] == receipt_id
            assert loaded["receipt"]["mutations"][0]["id"] == node_id
            still = tool("memory", {"action": "get", "id": node_id})
            assert marker in json.dumps(still)
            after_state = tool("memory", {"action": "state", "id": node_id})
            assert after_state["components"]["retrievalStrength"] < baseline_retrieval
            typed("causal_walk", {"scope": "user"}, "pending_strata")
            typed("selftest", {}, "pending_strata")
            typed("forgotten_lesson", {"failure_id": node_id}, "pending_strata")
            called = {row["tool"] for row in coverage}
            missing = [name for name in names if name not in called]
            assert not missing, missing
            assert_no_sqlite()
            passed("all 18 tools answered on Strata: real writes, or a typed error")
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
            if output:
                output.write_text(json.dumps({
                    "catalog": locals().get("catalog", []),
                    "coverage": coverage,
                    "transcript": transcript,
                }, indent=2) + "\n")
    print(f"PASS {len(coverage)} tool calls; coverage lists successful and expected-error paths explicitly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/debug/vestige-mcp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.binary, args.output)
