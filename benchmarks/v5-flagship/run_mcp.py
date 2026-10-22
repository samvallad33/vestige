#!/usr/bin/env python3
"""Independent stdlib MCP/Strata smoke instrument for current Vestige V5.

SAFETY: Launches only the supplied vestige-mcp executable. JSON messages
are synthetic data. No agent, shell tool action, paid API, or destructive
command is executed. This is NOT a live host interception or ROI benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_TOOLS = set(json.loads((HERE / "preregistration.json").read_text(
    encoding="utf-8"))["actual_feature_inventory"])


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8"))


def timed_ms(start_ns):
    return round((time.monotonic_ns() - start_ns) / 1000000, 3)


class MCP:
    def __init__(self, binary, home, data_dir):
        self.log = (home / "server-stderr.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "HOME": str(home),
            "USERPROFILE": str(home),
            "VESTIGE_DASHBOARD_ENABLED": "false",
            "VESTIGE_HTTP_ENABLED": "0",
            "VESTIGE_AUTOPILOT_ENABLED": "0",
            "VESTIGE_DATA_DIR": str(data_dir),
            "RUST_LOG": "off",
        })
        for name in ("OPERATOR_HOME", "OPERATOR_GATE_MODE", "OPERATOR_AGENT_SESSION",
                     "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
            env.pop(name, None)
        self.p = subprocess.Popen(
            [str(binary), "--no-http", "--data-dir", str(data_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, cwd=str(home), env=env,
            text=True, bufsize=1
        )
        self.messages = queue.Queue()
        self.next_id = 0
        self.reader = threading.Thread(target=self._reader, daemon=True)
        self.reader.start()
        self.rpc("initialize", {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "vestige-v5-trials-stdio", "version": "0.1"}
        })
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _reader(self):
        try:
            for line in self.p.stdout:
                if not line.strip():
                    continue
                try:
                    self.messages.put(json.loads(line))
                except ValueError:
                    self.messages.put({"bad_json": line[:120]})
        finally:
            self.messages.put({"closed": True})

    def send(self, msg):
        if self.p.stdin is None:
            raise RuntimeError("MCP server standard input is closed")
        self.p.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.p.stdin.flush()

    def rpc(self, method, params, timeout=90):
        self.next_id += 1
        id_ = self.next_id
        self.send({"jsonrpc": "2.0", "id": id_, "method": method,
                   "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MCP response timeout during " + method)
            try:
                msg = self.messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("MCP response timeout during " + method) from exc
            if "closed" in msg:
                raise RuntimeError("MCP server closed during " + method)
            if msg.get("id") != id_:
                continue
            if "error" in msg:
                raise RuntimeError("MCP protocol error {}: {}".format(
                    method, str(msg["error"])[:350]))
            return msg.get("result", {})

    def call(self, name, args, ok=True):
        t0 = time.monotonic_ns()
        response = self.rpc("tools/call", {"name": name, "arguments": args})
        body = response.get("structuredContent")
        if not isinstance(body, (dict, list)):
            texts = response.get("content") or []
            raw = texts[0].get("text", "") if texts else ""
            try:
                body = json.loads(raw)
            except (TypeError, ValueError):
                body = {"raw": raw}
        is_error = bool(response.get("isError")) or (
            isinstance(body, dict) and "error" in body
        )
        if ok and is_error:
            raise AssertionError("{} returned an error: {}".format(
                name, str(body)[:400]))
        return body, timed_ms(t0)

    def close(self):
        try:
            if self.p.stdin:
                self.p.stdin.close()
            self.p.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
            self.p.wait(timeout=5)
        finally:
            try:
                if self.p.stdout:
                    self.p.stdout.close()
            finally:
                self.log.close()


def log_hash(folder):
    """Digest all on-disk Strata log file bytes (read-only)."""
    h = hashlib.sha256()
    if not folder.exists():
        return h.hexdigest()
    for f in sorted(p for p in folder.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(folder)).encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()


def record(rows, name, func):
    started = time.monotonic_ns()
    try:
        data = func()
        rows.append({
            "name": name, "status": "pass", "elapsed_ms": timed_ms(started),
            "output_digest": digest(data),
            "evidence": data,
        })
        return data
    except Exception as exc:
        rows.append({
            "name": name, "status": "fail", "elapsed_ms": timed_ms(started),
            "error_type": type(exc).__name__,
            "error": str(exc)[:700],
        })
        return None


def assert_field(body, field, value=True):
    if not isinstance(body, dict) or body.get(field) != value:
        raise AssertionError("{} expected {!r}, got {}".format(
            field, value, str(body)[:350]))


def catalog(client):
    raw = client.rpc("tools/list", {})
    tools = raw.get("tools", [])
    names = [item.get("name") for item in tools]
    if len(tools) != 18 or set(names) != EXPECTED_TOOLS or len(set(names)) != 18:
        raise AssertionError("tool catalog mismatch: " + str(names))
    wire_bytes = len(json.dumps(raw, separators=(",", ":")).encode("utf-8"))
    return {"tool_count": len(tools), "tool_names": sorted(names),
            "wire_bytes": wire_bytes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   help="Report JSON path (exclusive create; no overwrite)")
    args = p.parse_args()
    try:
        binary = args.binary.resolve(strict=True)
        if not binary.is_file():
            raise ValueError("binary is not a file")
    except (OSError, ValueError) as exc:
        p.error("executable binary required: " + str(exc))

    rows = []
    with tempfile.TemporaryDirectory(prefix="vestige-v5-flagship-") as temp:
        home = Path(temp) / "home"
        home.mkdir()
        data_dir = Path(temp) / "strata"
        client = None
        try:
            client = MCP(binary, home, data_dir)

            record(rows, "18_tool_catalog", lambda: catalog(client))

            first = record(rows, "admit_quiet_prior_memory", lambda:
                client.call("smart_ingest", {
                    "scope": "hindsight-demo",
                    "content": "Earlier decision: prefer the unchanged neon-queue retry budget in this experiment.",
                    "node_type": "decision",
                    "tags": ["quiet-decision-hindsight"],
                    "intent_id": "vestige-v5-trials-first-write"
                })[0]
            )
            prior_id = first.get("nodeId") if isinstance(first, dict) else None
            if prior_id is not None and not prior_id.startswith("mem-"):
                raise ValueError("unexpected memory id format")

            if prior_id:
                record(rows, "receipt_replays_exact_write", lambda:
                    verify_replay(client, prior_id))
                record(rows, "exact_handle_recalls_quiet_note", lambda:
                    exact_recall(client, prior_id))

                record(rows, "read_does_not_modify_strata_log", lambda:
                    read_only_check(client, prior_id, data_dir / "log"))

                record(rows, "replayed_intent_id_does_not_overwrite", lambda:
                    check_intent_replay(client))
            else:
                rows.append({"name": "dependent_causal_tests", "status": "skip",
                             "reason": "prior_id not obtained"})

            if prior_id:
                symptom = record(rows, "admit_failure_with_declared_link", lambda:
                    client.call("smart_ingest", {
                        "scope": "hindsight-demo",
                        "content": "Later failure: checkout queue processing failed after retry budget change.",
                        "node_type": "event",
                        "tags": ["incident-hindsight"],
                        "links": [{"kind": "derived_from", "to": prior_id}]
                    })[0]
                )
                symptom_id = symptom.get("nodeId") if isinstance(symptom, dict) else None
                if symptom_id:
                    record(rows, "causal_walk_reaches_recorded_prior", lambda:
                        linked_walk(client, symptom_id, prior_id))
            record(rows, "no_edge_means_no_invented_cause", lambda:
                no_edge_check(client))
            record(rows, "selftest_never_writes_live_store", lambda:
                check_selftest(client))
            record(rows, "free_text_similarity_is_withheld", lambda:
                no_similarity(client))

            client.close()
            client = None

            if prior_id:
                client = MCP(binary, home, data_dir)
                record(rows, "exact_memory_survives_server_restart", lambda:
                    exact_recall(client, prior_id))

        except Exception as exc:
            rows.append({"name": "server_setup_or_lifecycle", "status": "fail",
                         "error_type": type(exc).__name__, "error": str(exc)[:700]})
        finally:
            if client is not None:
                client.close()

    report = {
        "study": "vestige-v5-trials/0.1",
        "kind": "synthetic_actual_mcp_stdio_mechanics_not_agent_benchmark",
        "utc": datetime.now(timezone.utc).isoformat(),
        "binary_sha256": sha_file(binary),
        "runner_sha256": sha_file(Path(__file__)),
        "preregistration_sha256": sha_file(HERE / "preregistration.json"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "results": rows,
        "passed": sum(x["status"] == "pass" for x in rows),
        "failed": sum(x["status"] == "fail" for x in rows),
        "skipped": sum(x["status"] == "skip" for x in rows),
        "limitations": [
            "No live agent host or harmful effect executed",
            "No native/OPA/AGT baseline",
            "No neural model or retroactive salience implementation validated",
            "No independent downstream task, attack, dollar or customer ROI result",
            "Public fixtures, not held-out cases",
        ],
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as f:
                f.write(output)
        except OSError as exc:
            print("Report not written (exclusive path): " + str(exc), file=sys.stderr)
            return 2
        print("wrote report {}: pass {}, fail {}, skip {}".format(
            args.output, report["passed"], report["failed"], report["skipped"]))
    else:
        print(output)
    return 0 if report["failed"] == 0 and report["passed"] >= 8 else 1


def verify_replay(c, id_):
    body, ms = c.call("receipt", {"action": "replay", "receipt_id": id_})
    assert_field(body, "matched", True)
    assert_field(body, "mismatches", [])
    return {"matched": True, "elapsed_tool_ms": ms}


def exact_recall(c, id_):
    body, ms = c.call("recall", {"handle": id_})
    nodes = body.get("nodes", []) if isinstance(body, dict) else []
    if not nodes or nodes[0].get("id") != id_:
        raise AssertionError("exact handle did not return expected memory")
    return {"exact": body.get("exact"), "retrieved_id": id_,
            "elapsed_tool_ms": ms}


def read_only_check(c, id_, log_dir):
    before = log_hash(log_dir)
    c.call("recall", {"handle": id_})
    c.call("memory_status", {"view": "health"})
    after = log_hash(log_dir)
    if before != after:
        raise AssertionError("read operations mutated Strata bytes")
    return {"unchanged": True, "log_hash": before}


def check_intent_replay(c):
    body, ms = c.call("smart_ingest", {
        "scope": "hindsight-demo",
        "content": "ANOTHER TEXT UNDER THE SAME INTENT MUST NOT REPLACE THE FIRST",
        "intent_id": "vestige-v5-trials-first-write",
        "tags": ["must-not-overwrite"]
    })
    assert_field(body, "decision", "replay")
    return {"idempotent": True, "elapsed_tool_ms": ms,
            "response_digest": digest(body)}


def linked_walk(c, symptom_id, prior_id):
    body, ms = c.call("causal_walk", {
        "scope": "hindsight-demo",
        "start_points": [{"kind": "logged_write", "node_id": symptom_id}],
        "promote": False
    })
    causes = body.get("causes", []) if isinstance(body, dict) else []
    matches = [v for v in causes if v.get("id") == prior_id]
    if not matches:
        raise AssertionError("recorded prior decision not reached")
    path = matches[0].get("path", [])
    if not any(e.get("link_type") == "derived_from" for e in path):
        raise AssertionError("matching link evidence missing from causal path")
    return {"reached_id": prior_id, "edge_types": [e.get("link_type") for e in path],
            "causality_proven": False, "elapsed_tool_ms": ms}


def no_edge_check(c):
    a, _ = c.call("smart_ingest", {
        "scope": "edge-absence-control",
        "content": "Quiet earlier neutral record about fern-colored paper weights.",
        "tags": ["no-edge-fern"]
    })
    b, _ = c.call("smart_ingest", {
        "scope": "edge-absence-control",
        "content": "An unrelated later mistake occurred in an obsidian-colored card.",
        "tags": ["no-edge-obsidian"]
    })
    aid, bid = a.get("nodeId"), b.get("nodeId")
    if not aid or not bid:
        raise AssertionError("failed to set up absence control")
    walked, ms = c.call("causal_walk", {
        "scope": "edge-absence-control",
        "start_points": [{"kind": "logged_write", "node_id": bid}]
    })
    if any(x.get("id") == aid for x in walked.get("causes", [])):
        raise AssertionError("invented an unsupported link to unrelated older memory")
    return {"invented_root": False, "elapsed_tool_ms": ms,
            "cause_count": len(walked.get("causes", [])),
            "empty_because": walked.get("emptyBecause")}


def check_selftest(c):
    body, ms = c.call("selftest", {})
    assert_field(body, "all_passed", True)
    assert_field(body, "live_store_touched", False)
    return {"all_passed": True, "checks_passed": body.get("checks_passed"),
            "elapsed_tool_ms": ms}


def no_similarity(c):
    body, ms = c.call("recall", {"query": "look up the similar quiet configuration"})
    if "similarity_disabled" not in json.dumps(body, sort_keys=True):
        raise AssertionError("expected explicit similarity_disabled refusal")
    return {"similarity_disabled": True, "elapsed_tool_ms": ms}


if __name__ == "__main__":
    sys.exit(main())
