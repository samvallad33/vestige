#!/usr/bin/env python3
"""Exercise codebase/startup evidence through MCP, with only disposable fixtures.

Usage: python3 scripts/test-context-evidence.py --binary target/debug/vestige-mcp
No credentials, network requests, embedding models, or live memory are needed.
"""
import argparse
import json
import os
from pathlib import Path
import select
import sqlite3
import subprocess
import tempfile


def run(binary):
    results = []
    with tempfile.TemporaryDirectory(prefix="vestige-context-evidence-") as temp:
        root = Path(temp)
        repo, other = root / "repo", root / "other-worktree"
        repo.mkdir()
        other.mkdir()
        original = "def example(flag):\n    if flag:\n        return 1\n    return 2\n"
        for directory in (repo, other):
            (directory / "example.py").write_text(original)
        env = dict(os.environ, VESTIGE_DASHBOARD_ENABLED="false", VESTIGE_HTTP_ENABLED="false", RUST_LOG="error")
        proc = subprocess.Popen([str(binary.resolve()), "--no-http", "--data-dir", str(root / "store")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                env=env, text=True, bufsize=1)
        seq = 0

        def rpc(method, params):
            nonlocal seq
            seq += 1
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": seq, "method": method, "params": params}) + "\n")
            proc.stdin.flush()
            while True:
                if not select.select([proc.stdout], [], [], 40)[0]:
                    raise TimeoutError(method)
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("MCP exited")
                response = json.loads(line)
                if response.get("id") == seq:
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    return response["result"]

        def tool(name, args, error=False):
            result = rpc("tools/call", {"name": name, "arguments": args})
            if error:
                assert result.get("isError"), result
                return result
            assert not result.get("isError"), result
            return result.get("structuredContent") or json.loads(result["content"][0]["text"])

        def save(name="Return contract", scope="user", project="fixture", files=None):
            return tool("codebase", {"action": "remember_pattern", "name": name,
                        "description": "Return two when flag is false.", "codebase": project,
                        "scope": scope, "repoPath": str(repo), "files": files or ["example.py#example"]})["nodeId"]

        def contexts(path=repo, scope="user", budget=4000, queries=None):
            args = {"action": "get_context", "codebase": "fixture", "scope": scope}
            context = {"codebase": "fixture"}
            if path is not None:
                args["repoPath"] = str(path)
                context["repoPath"] = str(path)
            direct = tool("codebase", args)
            startup = tool("session_start", {"queries": queries or [], "context": context,
                           "scope": scope, "token_budget": budget, "include_predictions": False,
                           "include_status": False, "include_intentions": False})
            # Python's compact ensure_ascii=False serialization matches serde_json
            # for these fixtures (including UTF-8 and key order-independent size).
            size = len(json.dumps(startup, ensure_ascii=False, separators=(",", ":")).encode())
            assert size <= budget * 4, (size, budget)
            assert startup["tokensUsed"] == (size + 3) // 4
            return direct, startup

        def item(direct, startup, node_id):
            a = next(x for x in direct["patterns"]["items"] if x["id"] == node_id)
            b = next(x for x in startup["codeContext"]["items"] if x["id"] == node_id)
            assert a["anchorStatus"] == b["anchorStatus"]
            assert a["evidence"] == b["evidence"]
            assert node_id in startup["context"]
            assert "Return two when flag is false." in startup["context"]
            return a

        def passed(case):
            results.append(case)
            print("PASS", case, flush=True)

        try:
            rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                               "clientInfo": {"name": "context-evidence-fixture", "version": "1"}})
            proc.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            proc.stdin.flush()
            tool("session_start", {"queries": ["q"] * 17}, error=True)
            tool("session_start", {"queries": [], "scope": "   "}, error=True)
            passed("startup validates query bounds and namespace before retrieval")
            node_id = save()
            a = item(*contexts(), node_id)
            assert a["evidence"]["state"] == "unchanged_evidence"
            assert a["evidence"]["claimVerified"] is False
            passed("C01 unchanged parity and actionable summary")
            (repo / "irrelevant.txt").write_text("unrelated change")
            assert item(*contexts(), node_id)["anchorStatus"] == "verified"
            passed("C02 unrelated change preserves evidence")
            (repo / "example.py").write_text("def example(flag):\n    if flag:\n        return 1\n        return 2\n")
            changed_direct, changed_start = contexts()
            assert item(changed_direct, changed_start, node_id)["evidence"]["state"] == "needs_recheck"
            print("Observed drift startup:", changed_start["context"], flush=True)
            passed("C03 Python indentation invalidates evidence on both tools")
            assert item(*contexts(other), node_id)["anchorStatus"] == "verified"
            assert item(*contexts(repo), node_id)["anchorStatus"] == "drifted"
            passed("C08 separate checkout contents never reuse verdicts")
            for path in (None, repo / "absent"):
                assert item(*contexts(path), node_id)["evidence"]["state"] == "unavailable"
            passed("C06 missing checkout does not imply deleted source")
            (repo / "example.py").write_text(original)
            partial = save("Partial contract", files=["example.py#example", "example.py"])
            a = item(*contexts(), partial)
            assert a["evidence"]["state"] == "partial"
            assert a["evidence"]["checkedAnchors"] == 1
            assert a["evidence"]["totalAnchors"] == 2
            assert a["anchorStatus"] == "unverifiable"
            passed("C05 matching plus unverifiable is partial")
            foreign = save("Other namespace", scope="other")
            prefix = save("Similar project", project="fixture-extra")
            direct, startup = contexts(queries=["Return contract"])
            assert foreign not in json.dumps(direct) + json.dumps(startup)
            assert prefix not in json.dumps(direct) + json.dumps(startup)
            assert item(*contexts(scope="other"), foreign)["anchorStatus"] == "verified"
            passed("C09 exact project tags and explicit namespaces")
            for budget in (100, 101, 150, 200, 500, 1000):
                contexts(budget=budget)
            passed("C10 complete serialized output fits small budgets")
            dbs = list((root / "store").rglob("*.db"))
            db = next(p for p in dbs if "knowledge_nodes" in {
                row[0] for row in sqlite3.connect(p).execute("SELECT name FROM sqlite_master")})
            with sqlite3.connect(db) as conn:
                before = conn.execute("SELECT reps,storage_strength,retention_strength,content FROM knowledge_nodes WHERE id=?", (node_id,)).fetchone()
            contexts(); contexts()
            with sqlite3.connect(db) as conn:
                after = conn.execute("SELECT reps,storage_strength,retention_strength,content FROM knowledge_nodes WHERE id=?", (node_id,)).fetchone()
                assert before == after
                cached = conn.execute("SELECT last_verified_at FROM code_memory_anchors WHERE node_id=?", (node_id,)).fetchall()
                assert all(row[0] is None for row in cached)
            passed("C12 repeated context reads do not reinforce or cache across worktrees")
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE knowledge_nodes SET valid_until='2020-01-01T00:00:00Z' WHERE id=?", (partial,))
                conn.execute("UPDATE code_memory_anchors SET content_hash='0123456789abcdef0123456789abcdef' WHERE node_id=?", (node_id,))
            direct, startup = contexts()
            assert partial not in json.dumps(direct) + json.dumps(startup)
            a = item(direct, startup, node_id)
            assert a["evidence"]["state"] == "unavailable"
            assert "Legacy" in a["anchors"][0]["detail"]
            passed("C11 legacy hashes retain honest weaker evidence; expired advice excluded")
            tool("codebase", {"action": "reanchor", "memoryId": node_id, "scope": "other", "repoPath": str(repo), "files": ["example.py#example"]}, error=True)
            tool("codebase", {"action": "reanchor", "memoryId": node_id, "repoPath": str(repo), "files": ["missing.py#nope"]}, error=True)
            assert item(*contexts(), node_id)["evidence"]["state"] == "unavailable"
            repaired = tool("codebase", {"action": "reanchor", "memoryId": node_id, "repoPath": str(repo), "files": ["example.py#example"]})
            assert repaired["nodeId"] == node_id and repaired["memoryContentChanged"] is False
            assert item(*contexts(), node_id)["anchorStatus"] == "verified"
            with sqlite3.connect(db) as conn:
                assert before == conn.execute("SELECT reps,storage_strength,retention_strength,content FROM knowledge_nodes WHERE id=?", (node_id,)).fetchone()
            passed("explicit reanchor preserves memory identity and rejects incomplete or wrong-scope replacement")
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE knowledge_nodes SET superseded_by=? WHERE id=?", (foreign, node_id))
            direct, startup = contexts(queries=["Return contract"])
            assert node_id not in json.dumps(direct) + json.dumps(startup)
            passed("C09 superseded advice excluded from both paths")
            tool("intention", {"action": "set", "description": "Synthetic due reminder", "priority": "normal",
                               "trigger": {"type": "time", "at": "2020-01-01T00:00:00Z"}})
            start = tool("session_start", {"queries": [], "include_predictions": False})
            assert "Synthetic due reminder" in start["context"]
            passed("due time intention works without context")
            (repo / "string.py").write_text('value = """\n text\n"""\n')
            string_id = save("String content", files=["string.py:1-3"])
            (repo / "string.py").write_text('value = """\n  text\n"""\n')
            assert item(*contexts(), string_id)["evidence"]["state"] == "needs_recheck"
            passed("C04 whitespace inside string content invalidates evidence")
            tool("codebase", {"action": "remember_pattern", "name": "Unicode", "description": "保留缩进 🦀 " * 80,
                              "codebase": "unicode-project", "files": ["example.py#example"], "repoPath": str(repo)})
            for budget in (100, 150, 500, 1000):
                packet = tool("session_start", {"queries": [], "context": {"codebase": "unicode-project", "repoPath": str(repo)},
                              "token_budget": budget, "include_predictions": False})
                size = len(json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode())
                assert size <= budget * 4 and packet["tokensUsed"] == (size + 3) // 4
            passed("C10 Unicode evidence and escaped metadata fit serialized budgets")
            # Inject a storage failure only in this disposable store.
            with sqlite3.connect(db) as conn:
                conn.execute("DROP TABLE code_memory_anchors")
            tool("codebase", {"action": "get_context", "scope": "other", "repoPath": str(repo)}, error=True)
            tool("session_start", {"scope": "other", "queries": [], "context": {"codebase": "fixture", "repoPath": str(repo)},
                                   "include_predictions": False}, error=True)
            passed("C07 anchor storage failures surface as tool errors")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/debug/vestige-mcp"))
    args = parser.parse_args()
    cases = run(args.binary)
    print(json.dumps({"passed": len(cases), "cases": cases}, indent=2))
