#!/usr/bin/env python3
"""Disposable stdio MCP contracts for discovery and evidence-aware tools.

No real memory, credentials, connector requests, or embedding models are used.
--output stores a synthetic request/response transcript and coverage inventory.
"""
import argparse
import json
import os
from pathlib import Path
import select
import sqlite3
import subprocess
import tempfile


def run(binary, output):
    transcript = []
    coverage = []
    with tempfile.TemporaryDirectory(prefix="vestige-tool-frontier-") as temp:
        root = Path(temp)
        # Keep inherited connector credentials and integration hooks out of fixtures.
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VESTIGE_", "REDMINE_", "GITHUB_"))}
        env.update(VESTIGE_DASHBOARD_ENABLED="false", VESTIGE_HTTP_ENABLED="false", RUST_LOG="error")
        proc = subprocess.Popen([str(binary.resolve()), "--no-http", "--data-dir", str(root / "store")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1)
        seq = 0

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
            coverage.append({"tool": name, "selector": args.get("action", args.get("mode", args.get("view", "default"))),
                             "outcome": "expected_error" if error else "success"})
            return result.get("structuredContent") or json.loads(result["content"][0]["text"])

        def passed(name):
            print("PASS", name, flush=True)

        try:
            rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                               "clientInfo": {"name": "tool-frontier-fixture", "version": "1"}})
            proc.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            proc.stdin.flush()
            catalog = rpc("tools/list", {})["tools"]
            guide = tool("memory_status", {"view": "tools"})["tools"]
            assert [x["name"] for x in guide] == [x["name"] for x in catalog]
            for entry, definition in zip(guide, catalog):
                for selector in ("action", "mode", "view"):
                    values = definition["inputSchema"].get("properties", {}).get(selector, {}).get("enum")
                    if values is not None:
                        assert entry["selectors"][selector]["values"] == values
                detail = tool("memory_status", {"view": "tools", "tool": entry["name"]})
                assert detail["tools"][0]["inputSchema"] == definition["inputSchema"]
            for invalid in ("search", "", 12):
                tool("memory_status", {"view": "tools", "tool": invalid}, error=True)
            annotations = {x["name"]: x["annotations"] for x in catalog}
            assert annotations["recall"]["readOnlyHint"] is False
            assert annotations["recall"]["idempotentHint"] is False
            assert annotations["suppress"]["idempotentHint"] is False
            passed("all installed tool and action definitions match progressive discovery")

            tool("smart_ingest", {"content": "Must not silently disappear", "items": [{"content":"Batch fixture"}]}, error=True)
            tool("maintain", {"action":"export", "start":"2026-01-01"}, error=True)
            maintenance_page = tool("maintain", {"action":"consolidate", "phase":"embeddings", "batchSize":2})
            assert maintenance_page["dryRun"] is True and maintenance_page["selected"] == 0
            assert maintenance_page["hasMore"] is False
            tool("maintain", {"action":"consolidate", "batchSize":2}, error=True)
            tool("maintain", {"action":"consolidate", "phase":"embeddings", "batchSize":101}, error=True)
            passed("embedding maintenance previews bounded pages and rejects misplaced controls")
            assert annotations["receipt"]["readOnlyHint"] is False
            assert annotations["receipt"]["idempotentHint"] is True
            cause = tool("smart_ingest", {"content": "Set FIXTURE_TIMEOUT to two seconds in fixture service.",
                         "tags": ["fixture-old", "FIXTURE_TIMEOUT"], "forceCreate": True})["nodeId"]
            failure = tool("smart_ingest", {"content": "Fixture service failure: FIXTURE_TIMEOUT caused a timeout.",
                           "tags": ["FIXTURE_TIMEOUT", "failure"], "forceCreate": True})["nodeId"]
            foreign = tool("smart_ingest", {"content": "Foreign project fixture-only marker FOREIGN_CANARY.",
                           "scope": "other-project", "forceCreate": True})["nodeId"]
            project_failure = tool("smart_ingest", {"content": "Project fixture deployment failed due to timeout.",
                                   "tags": ["failure"], "scope": "other-project", "forceCreate": True})
            hook = project_failure["failureHooks"]["backfill"]
            assert hook["scope"] == "other-project" and hook["preview"] is True
            assert hook["causesPromoted"] == 0 and hook["evidenceStatus"] == "hypothesis"
            db = next(p for p in (root / "store").rglob("*.db") if "knowledge_nodes" in {
                row[0] for row in sqlite3.connect(p).execute("SELECT name FROM sqlite_master")})
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE knowledge_nodes SET created_at='2026-01-01T00:00:00Z' WHERE id=?", (cause,))
                conn.execute("UPDATE knowledge_nodes SET created_at='2026-01-03T00:00:00Z' WHERE id=?", (failure,))
                before = conn.execute("SELECT reps,stability FROM knowledge_nodes WHERE id=?", (cause,)).fetchone()
                edges_before = conn.execute("SELECT COUNT(*) FROM memory_connections").fetchone()
            for _ in range(2):
                preview = tool("backfill", {"failure_id": failure})
                assert preview["preview"] is True and preview["causality_verified"] is False
                assert preview["causes"] and not any(x["promoted"] for x in preview["causes"])
            with sqlite3.connect(db) as conn:
                assert before == conn.execute("SELECT reps,stability FROM knowledge_nodes WHERE id=?", (cause,)).fetchone()
                assert edges_before == conn.execute("SELECT COUNT(*) FROM memory_connections").fetchone()
            tool("backfill", {"failure_id": foreign}, error=True)
            promoted = tool("backfill", {"failure_id": failure, "promote": True})
            assert any(x["promoted"] for x in promoted["causes"])
            passed("backfill preview has no strength/graph mutation; explicit promotion remains available")

            for action in ("get", "state"):
                tool("memory", {"action": action, "id": cause})
            tool("memory", {"action": "get_batch", "ids": [cause, failure]})
            held = tool("suppress", {"id": cause})
            assert held["pendingReview"] is True and held["success"] is False
            # Exercise the separate supported fast mode only in this disposable
            # database, after proving the default review gate above.
            mode_path = db.parent / "review_mode.json"
            mode_path.write_text('{"mode":"fast"}')
            for action in ("promote", "demote"):
                tool("memory", {"action": action, "id": cause, "reason": "Synthetic fixture feedback"})
            first = tool("suppress", {"id": cause})
            second = tool("suppress", {"id": cause})
            assert second["suppressionCount"] == first["suppressionCount"] + 1
            tool("suppress", {"id": cause, "reverse": True})
            tool("suppress", {"id": cause, "reverse": True})
            mode_path.unlink()
            passed("default suppression review gate and explicit fixture fast-mode compounding/reversal")

            for view in ("health", "retention", "timeline", "changelog", "stats"):
                tool("memory_status", {"view": view})
            intention = tool("intention", {"action": "set", "description": "Synthetic reminder",
                             "trigger": {"type": "time", "at": "2020-01-01T00:00:00Z"}})["intentionId"]
            tool("intention", {"action": "list"})
            tool("intention", {"action": "check"})
            tool("intention", {"action": "update", "id": intention, "status": "complete"})
            tool("graph", {"action": "recent", "limit": 5})
            graph = tool("graph", {"action": "never_composed", "limit": 5})
            assert graph["scope"] == "user" and graph["globalNoveltyVerified"] is False
            assert foreign not in json.dumps(graph)
            tool("maintain", {"action": "importance_score", "content": "Synthetic fixture design decision"})
            gc = tool("maintain", {"action": "gc"})
            assert gc.get("dryRun", gc.get("dry_run")) is True
            tool("dedup", {"action": "tag_rename", "source_tag": "fixture-old", "target_tag": "fixture-new"})
            tool("session_start", {"queries": [], "include_predictions": False})
            passed("status, intention lifecycle, graph investigation and maintenance preview contracts")

            lookup = tool("recall", {"query": "FIXTURE_TIMEOUT", "concrete": True})
            receipt_id = lookup.get("receiptId")
            assert receipt_id, lookup
            tool("receipt", {"action": "get", "receipt_id": receipt_id})
            replay = tool("receipt", {"action": "replay", "receipt_id": receipt_id, "withheld_slots": []})
            repeated = tool("receipt", {"action": "replay", "receipt_id": receipt_id, "withheld_slots": []})
            assert replay["replayId"] == repeated["replayId"] and replay["receiptId"] == repeated["receiptId"]
            assert repeated["reusedExisting"] is True
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE knowledge_nodes SET has_embedding=1 WHERE id=?", (cause,))
            edited = tool("memory", {"action":"edit", "id":cause, "content":"Reviewed FIXTURE_TIMEOUT fixture setting is two seconds."})
            assert edited["embeddingStatus"] == "pending"
            with sqlite3.connect(db) as conn:
                assert conn.execute("SELECT has_embedding FROM knowledge_nodes WHERE id=?", (cause,)).fetchone() == (0,)
            passed("edit invalidates embedding state without falsely promising regeneration")
            for budget in (100, 101, 250, 1000, 2500):
                reason = tool("recall", {"mode":"reason", "query":"FIXTURE_TIMEOUT", "token_budget":budget, "min_similarity":0})
                size = len(json.dumps(reason, ensure_ascii=False, separators=(",", ":")).encode())
                assert size <= budget * 4, (budget, size, reason)
                assert reason["tokensUsed"] == (size + 3) // 4
                assert reason["budgetUnit"] == "utf8_bytes_div_4_ceiling"
                assert foreign not in json.dumps(reason)
            reason = tool("recall", {"mode":"reason", "query":"FIXTURE_TIMEOUT", "token_budget":500,
                          "runId":"long-correlation-" * 1000, "min_similarity":0})
            assert len(json.dumps(reason, ensure_ascii=False, separators=(",", ":")).encode()) <= 2000
            tool("recall", {"mode":"reason", "query":"FIXTURE_TIMEOUT", "detail_level":"full"}, error=True)
            other = tool("recall", {"mode":"reason", "query":"FOREIGN_CANARY", "scope":"other-project", "min_similarity":0})
            assert foreign in json.dumps(other), other
            passed("reason scopes and final structured-content budgets after receipt metadata")
            for concrete in (True, False):
                for budget in (100, 101, 256, 1000):
                    limited = tool("recall", {"query":"FIXTURE_TIMEOUT", "concrete":concrete,
                                   "token_budget":budget, "min_similarity":0})
                    size = len(json.dumps(limited, ensure_ascii=False, separators=(",", ":")).encode())
                    assert size <= budget * 4, (budget, size, limited)
                    assert limited["tokensUsed"] == (size + 3) // 4
            packet_args = {"query":"FOREIGN_CANARY", "scope":"other-project", "concrete":True,
                           "context_packet":True, "token_budget":2000}
            packet = tool("recall", packet_args)
            assert packet["results"] and packet["packetId"]
            same = tool("recall", dict(packet_args, known_packet_id=packet["packetId"]))
            assert same["notModified"] is True and same["results"] == []
            tool("memory", {"action":"edit", "id":foreign,
                            "content":"FOREIGN_CANARY updated source decision."})
            changed = tool("recall", dict(packet_args, known_packet_id=packet["packetId"]))
            assert changed["notModified"] is False and changed["packetId"] != packet["packetId"]
            fresh = tool("recall", packet_args)
            assert fresh["results"] and fresh["notModified"] is False
            tool("recall", {"query":"FOREIGN_CANARY", "known_packet_id":packet["packetId"]}, error=True)
            passed("lookup complete-envelope budgets and stable-packet acknowledgment/edit/refresh")
            contradictions = tool("recall", {"mode":"contradictions"})
            assert foreign not in json.dumps(contradictions)
            tool("recall", {"mode":"contradictions", "token_budget":100}, error=True)
            timed = tool("intention", {"action":"check", "context":{"current_time":"2020-01-01T00:00:00Z"}})
            assert timed["checkedAt"].startswith("2020-01-01T00:00:00")
            tool("intention", {"action":"check", "context":{"current_time":"invalid"}}, error=True)
            passed("retrieval receipt inspection and frozen replay execute through MCP")
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            if output:
                output.write_text(json.dumps({"catalog": locals().get("catalog", []), "coverage": coverage,
                                              "transcript": transcript}, indent=2) + "\n")
    print(f"PASS {len(coverage)} tool calls; coverage lists successful and expected-error paths explicitly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/debug/vestige-mcp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.binary, args.output)
