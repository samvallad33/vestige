#!/usr/bin/env python3
"""Stdio contracts for the Strata-backed vestige-mcp binary.

Discovery stays exact. Writes go through the gate. Similarity (embeddings,
cosine, BM25, FTS, Jaccard, keyword or name match) is an error. A link
exists only when the log recorded an edge. No SQLite file is created.

The admission sweep calls every tool from `tools/list` and every advertised
action (action, mode, or view enum; `source` when that is the selector) with
minimal schema-valid arguments. Each call is a real stdio round-trip. A
response that contains `pending_strata`, `not implemented`, or `not admitted`
fails the process. `similarity_disabled` is a Strata refusal, not an admission
failure. `--admit-only --binary PATH` runs that sweep against another build,
including release/v4.0.0-rc.
"""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile

FORBIDDEN = ("pending_strata", "not implemented", "not admitted")
ARG_MARKERS = (
    "Invalid arguments",
    "Invalid memory ID format",
    "Missing arguments",
    "Missing '",
    "is required",
    "are required",
    "must not be empty",
    "cannot be empty",
    "does not support '",
    "Unknown recall mode",
    "Unknown receipt action",
    "Unknown maintain action",
    "Unknown dedup action",
    "Invalid action",
    "This action requires",
    "Pass confirm=true",
    "ids array cannot be empty",
)


def child_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VESTIGE_", "REDMINE_", "GITHUB_"))}
    env.update(VESTIGE_DASHBOARD_ENABLED="false", VESTIGE_HTTP_ENABLED="false", RUST_LOG="error")
    return env


def excerpt(text, needle=None, limit=160):
    flat = " ".join(str(text).split())
    if needle and needle in flat:
        start = max(0, flat.find(needle) - 48)
        return flat[start:start + limit]
    return flat[:limit]


def error_text(response):
    if not isinstance(response, dict):
        return ""
    if "error" in response:
        return json.dumps(response["error"])
    result = response.get("result")
    if not isinstance(result, dict) or not result.get("isError"):
        return ""
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and "error" in structured:
        err = structured["error"]
        return err if isinstance(err, str) else json.dumps(err)
    content = result.get("content") or []
    if content and isinstance(content[0], dict):
        return content[0].get("text") or ""
    return json.dumps(result)


def visible_error(response):
    text = error_text(response)
    for prefix in ("Initialization error: ", "Suppress failed: ", "sync failed: "):
        text = text.replace(prefix, "")
    return " ".join(text.split())


def classify(response):
    err = visible_error(response)
    full = json.dumps(response)
    hits = [needle for needle in FORBIDDEN if needle in full]
    if hits:
        return "FAIL", hits, excerpt(err or full, hits[0])
    if err and any(marker in err for marker in ARG_MARKERS):
        return "ARGS", [], excerpt(err)
    if err:
        return "PASS", [], "admitted error: " + excerpt(err)
    return "PASS", [], "ok"


def resolve_refs(schema, root):
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return schema
        node = node[part]
    if not isinstance(node, dict):
        return schema
    merged = dict(node)
    for key, value in schema.items():
        if key != "$ref":
            merged[key] = value
    return merged


def string_for(name, schema, ctx):
    pattern = schema.get("pattern")
    if pattern == "^evidence_[1-9][0-9]*$":
        return "evidence_1"
    if pattern == "^[a-f0-9]{64}$":
        return "0" * 64
    if schema.get("format") == "date-time":
        return "2020-01-01T00:00:00Z"
    key = name or ""
    if key in ("id", "memory_id", "memoryId", "failure_id", "receipt_id", "old_id",
               "survivor_id", "center_id", "from", "node_id"):
        return ctx["mem_id"]
    if key in ("new_id", "to"):
        return ctx["mem_id_2"]
    if key == "event_id":
        return "evt-fixture"
    if key in ("plan_id", "operation_id"):
        return "plan-fixture"
    if key == "repo":
        return "fixture-owner/fixture-repo"
    if key in ("path", "repoPath", "root"):
        return ctx["repo"]
    if key == "source_key":
        return "fixture-source"
    if key == "frame":
        return "src/fixture.rs:1"
    if key in ("name", "worked_in", "broke_in"):
        return "fixture"
    value = "fixture"
    minimum = schema.get("minLength") or 0
    if len(value) < minimum:
        value = value.ljust(minimum, "x")
    maximum = schema.get("maxLength")
    if isinstance(maximum, int) and maximum >= 0:
        value = value[:maximum] or "x"
    return value


def instantiate(schema, ctx, name=None, root=None):
    root = schema if root is None else root
    schema = resolve_refs(schema if isinstance(schema, dict) else {}, root)
    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    for key in ("oneOf", "anyOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            return instantiate(options[0], ctx, name, root)
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((item for item in typ if item != "null"), "string")
    if typ == "object" or "properties" in schema:
        return instantiate_object(schema, ctx, root)
    if typ == "array":
        count = schema.get("minItems") or 1
        item = schema.get("items") or {"type": "string"}
        return [instantiate(item, ctx, name, root) for _ in range(count)]
    if typ == "integer":
        if "minimum" in schema:
            return int(schema["minimum"])
        return int(schema.get("default", 1))
    if typ == "number":
        if "minimum" in schema:
            return schema["minimum"]
        return schema.get("default", 0)
    if typ == "boolean":
        if name == "confirm":
            return True
        return schema.get("default", False)
    return string_for(name, schema, ctx)


def instantiate_object(schema, ctx, root):
    props = schema.get("properties") or {}
    obj = {}
    for required in schema.get("required") or []:
        obj[required] = instantiate(props.get(required, {"type": "string"}), ctx, required, root)
    return obj


def selector_of(schema):
    props = schema.get("properties") or {}
    for key in ("action", "mode", "view"):
        enum = (props.get(key) or {}).get("enum")
        if isinstance(enum, list) and enum:
            return key, [value for value in enum if isinstance(value, str)]
    if "action" not in props and "mode" not in props and "view" not in props:
        enum = (props.get("source") or {}).get("enum")
        if isinstance(enum, list) and enum:
            return "source", [value for value in enum if isinstance(value, str)]
    return None, [None]


def matching_branch(schema, key, value):
    if key is None:
        return None
    for option in schema.get("oneOf") or []:
        if not isinstance(option, dict):
            continue
        const = (((option.get("properties") or {}).get(key) or {}).get("const"))
        if const == value:
            return option
    return None


def handler_extras(tool, action, ctx):
    mem, mem2, repo, restore = ctx["mem_id"], ctx["mem_id_2"], ctx["repo"], ctx["restore_file"]
    table = {
        ("memory", "get"): {"id": mem},
        ("memory", "get_batch"): {"ids": [mem, mem2]},
        ("memory", "delete"): {"id": mem, "confirm": True},
        ("memory", "purge"): {"id": mem, "confirm": True},
        ("memory", "state"): {"id": mem},
        ("memory", "promote"): {"id": mem, "reason": "fixture"},
        ("memory", "demote"): {"id": mem, "reason": "fixture"},
        ("memory", "edit"): {"id": mem, "content": "edited fixture content"},
        ("codebase", "remember_pattern"): {"name": "FixturePattern", "description": "A synthetic pattern."},
        ("codebase", "remember_decision"): {"decision": "Use the fixture.", "rationale": "The sweep needs a decision."},
        ("codebase", "get_context"): {"codebase": "fixture", "repoPath": repo},
        ("codebase", "verify"): {"repoPath": repo},
        ("codebase", "reanchor"): {
            "memoryId": mem, "repoPath": repo, "anchors": [{"path": "src/fixture.rs"}],
        },
        ("recall", "lookup"): {"query": "fixture-query"},
        ("recall", "reason"): {"query": "fixture-query", "depth": 5},
        ("recall", "contradictions"): {"topic": "fixture"},
        ("project", "preview"): {"root": repo},
        ("project", "write"): {"path": "CLAUDE.md", "root": repo, "confirm": True},
        ("intention", "set"): {
            "description": "Synthetic reminder",
            "trigger": {"type": "time", "at": "2020-01-01T00:00:00Z"},
        },
        ("intention", "check"): {"context": {"current_time": "2020-01-01T00:00:00Z"}},
        ("intention", "update"): {"id": "intention-fixture", "status": "complete"},
        ("intention", "list"): {"limit": 1},
        ("smart_ingest", "default"): {"content": "ADMISSION_FIXTURE_ROW", "forceCreate": True, "tags": ["fixture-old"]},
        ("source_sync", "github"): {"repo": "fixture-owner/fixture-repo", "max_pages": 1},
        ("source_sync", "redmine"): {"project": "fixture", "max_pages": 1},
        ("maintain", "consolidate"): {"phase": "lifecycle", "batchSize": 2},
        ("maintain", "importance_score"): {"content": "Synthetic fixture design decision"},
        ("maintain", "gc"): {"dry_run": True},
        ("maintain", "restore"): {"path": restore, "allowAnyPath": True},
        ("dedup", "plan_merge"): {"member_ids": [mem, mem2]},
        ("dedup", "plan_supersede"): {"old_id": mem, "new_id": mem2},
        ("dedup", "apply"): {"plan_id": "plan-fixture", "confirm": True},
        ("dedup", "verdict"): {"plan_id": "plan-fixture", "verdict": "reject"},
        ("dedup", "tag_rename"): {"source_tag": "fixture-old", "target_tag": "fixture-new"},
        ("dedup", "tag_merge"): {"source_tags": ["fixture-a", "fixture-b"], "target_tag": "fixture-c"},
        ("dedup", "protect"): {"id": mem},
        ("graph", "chain"): {"from": mem, "to": mem2},
        ("graph", "associations"): {"from": mem},
        ("graph", "bridges"): {"from": mem, "to": mem2},
        ("graph", "predict"): {"context": {"codebase": "fixture"}},
        ("graph", "memory_graph"): {"center_id": mem},
        ("graph", "get"): {"event_id": "evt-fixture"},
        ("graph", "memory"): {"memory_id": mem},
        ("graph", "neighbors"): {"memory_id": mem},
        ("graph", "never_composed"): {"scope": "user", "limit": 5},
        ("graph", "label"): {"event_id": "evt-fixture", "outcome_type": "helpful"},
        ("session_start", "default"): {
            "queries": ["fixture"], "include_predictions": False, "include_intentions": False,
        },
        ("suppress", "default"): {"id": mem},
        ("causal_walk", "default"): {
            "scope": "user",
            "start_points": [{"kind": "failing_test", "name": "fixture_test"}],
        },
        ("forgotten_lesson", "default"): {"failure_id": mem, "scope": "user"},
        ("purge", "default"): {"id": mem, "confirm": True},
    }
    return dict(table.get((tool, action), {}))


def minimal_args(tool, schema, key, value, ctx):
    args = {}
    if key is not None:
        args[key] = value
    branch = matching_branch(schema, key, value)
    required = list(schema.get("required") or [])
    props = dict(schema.get("properties") or {})
    if branch:
        required.extend(branch.get("required") or [])
        props.update(branch.get("properties") or {})
    for name in required:
        if name not in args:
            args[name] = instantiate(props.get(name, {"type": "string"}), ctx, name, schema)
    action = value if isinstance(value, str) else "default"
    args.update(handler_extras(tool, action, ctx))
    return args


def command_variants(schema, ctx):
    command = (schema.get("properties") or {}).get("command") or {}
    variants = command.get("oneOf") if isinstance(command, dict) else None
    rows = []
    for variant in variants or []:
        if not isinstance(variant, dict):
            continue
        const = (((variant.get("properties") or {}).get("action") or {}).get("const"))
        if not isinstance(const, str):
            continue
        rows.append((const, instantiate_object(variant, ctx, schema)))
    return rows


def cases_for(tool, schema, ctx):
    key, values = selector_of(schema)
    rows = []
    for value in values:
        args = minimal_args(tool, schema, key, value, ctx)
        if key == "action" and value == "graph":
            variants = command_variants(schema, ctx) or [("evaluate", {"action": "evaluate"})]
            for command, body in variants:
                row = dict(args)
                row["command"] = body
                rows.append((f"graph/{command}", row))
            continue
        label = value if isinstance(value, str) else "(default)"
        rows.append((label, args))
    return rows


def tool_body(response):
    result = response.get("result") if isinstance(response, dict) else None
    if not isinstance(result, dict):
        return {}
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = result.get("content") or []
    if content and isinstance(content[0], dict):
        text = content[0].get("text") or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def print_admission_table(rows):
    tool_w = max([4, *(len(row["tool"]) for row in rows)])
    action_w = max([6, *(len(row["action"]) for row in rows)])
    print(f"{'tool':<{tool_w}}  {'action':<{action_w}}  status  detail", flush=True)
    print(f"{'-' * tool_w}  {'-' * action_w}  ------  ------", flush=True)
    for row in rows:
        print(
            f"{row['tool']:<{tool_w}}  {row['action']:<{action_w}}  {row['status']:<6}  {row['detail']}",
            flush=True,
        )
    failed = [row for row in rows if row["status"] != "PASS"]
    print(
        f"admission {len(rows)} calls, {len(failed)} failed "
        f"({sum(row['status'] == 'FAIL' for row in rows)} refused, "
        f"{sum(row['status'] == 'ARGS' for row in rows)} invalid-args, "
        f"{sum(row['status'] == 'ERROR' for row in rows)} transport)",
        flush=True,
    )
    for row in failed:
        needles = ",".join(row["needles"]) or row["status"]
        print(f"  {row['tool']} {row['action']} — {needles}: {row['detail']}", flush=True)


def prepare_ctx(root, mem_id=None, mem_id_2=None):
    repo = root / "repo"
    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "fixture.rs").write_text("fn fixture() {}\n")
    (repo / "CLAUDE.md").write_text("# fixture\n")
    restore = root / "restore.json"
    restore.write_text('{"memories":[]}\n')
    return {
        "mem_id": mem_id or ("mem-" + "a" * 16),
        "mem_id_2": mem_id_2 or ("mem-" + "b" * 16),
        "repo": str(repo),
        "restore_file": str(restore),
    }


def adopt_ingested_id(response, fallback):
    body = tool_body(response)
    node = body.get("nodeId")
    if isinstance(node, str) and node.startswith("mem-") and body.get("success") is True:
        return node
    return fallback


class StdioMcp:
    def __init__(self, binary, store):
        self.binary = Path(binary)
        self.store = Path(store)
        self.proc = None
        self.seq = 0
        self.transcript = []

    def spawn(self):
        self.proc = subprocess.Popen(
            [str(self.binary.resolve()), "--no-http", "--data-dir", str(self.store)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=child_env(), text=True, bufsize=1,
        )

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)

    def rpc(self, method, params):
        self.seq += 1
        request = {"jsonrpc": "2.0", "id": self.seq, "method": method, "params": params}
        if self.proc.poll() is not None:
            raise RuntimeError("MCP process exited")
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        while True:
            if not select.select([self.proc.stdout], [], [], 90)[0]:
                raise TimeoutError(method)
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP process exited")
            response = json.loads(line)
            if response.get("id") == self.seq:
                self.transcript.append({"request": request, "response": response})
                return response

    def notify_initialized(self):
        self.proc.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        self.proc.stdin.flush()

    def handshake(self):
        response = self.rpc("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "tool-frontier-fixture", "version": "1"},
        })
        if "error" in response:
            raise RuntimeError(response["error"])
        self.notify_initialized()


def full_schema(mcp, name, compact):
    response = mcp.rpc("tools/call", {
        "name": "memory_status",
        "arguments": {"view": "tools", "tool": name},
    })
    body = tool_body(response)
    tools = body.get("tools") if isinstance(body, dict) else None
    if isinstance(tools, list) and tools and isinstance(tools[0].get("inputSchema"), dict):
        return tools[0]["inputSchema"]
    return compact


def seed_ids(mcp, ctx):
    """Two real nodes when ingest is admitted, so later calls use live ids."""
    for key, content in (("mem_id", "ADMISSION_FIXTURE_A"), ("mem_id_2", "ADMISSION_FIXTURE_B")):
        try:
            response = mcp.rpc("tools/call", {
                "name": "smart_ingest",
                "arguments": {"content": content, "forceCreate": True, "tags": ["fixture-old"]},
            })
        except (TimeoutError, RuntimeError, json.JSONDecodeError):
            return
        ctx[key] = adopt_ingested_id(response, ctx[key])


def admission_rows(mcp, ctx):
    catalog_response = mcp.rpc("tools/list", {})
    if "error" in catalog_response:
        raise RuntimeError(catalog_response["error"])
    catalog = catalog_response["result"]["tools"]
    rows = []
    seen = set()
    for definition in catalog:
        name = definition["name"]
        try:
            schema = full_schema(mcp, name, definition.get("inputSchema") or {})
        except (TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            rows.append({
                "tool": name, "action": "(schema)", "status": "ERROR", "needles": [],
                "detail": excerpt(exc), "arguments": {},
            })
            if mcp.proc.poll() is not None:
                break
            schema = definition.get("inputSchema") or {}
        dead = False
        for action, args in cases_for(name, schema, ctx):
            seen.add(name)
            try:
                response = mcp.rpc("tools/call", {"name": name, "arguments": args})
            except (TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                rows.append({
                    "tool": name, "action": action, "status": "ERROR", "needles": [],
                    "detail": excerpt(exc), "arguments": args,
                })
                if mcp.proc.poll() is not None:
                    dead = True
                    break
                continue
            status, needles, detail = classify(response)
            rows.append({
                "tool": name, "action": action, "status": status, "needles": needles,
                "detail": detail, "arguments": args,
            })
        if dead:
            break
    missing = [tool["name"] for tool in catalog if tool["name"] not in seen]
    return rows, missing


def finish_admission(rows, missing):
    print_admission_table(rows)
    failed = [row for row in rows if row["status"] != "PASS"]
    if missing:
        print("admission missing tools with no call:", ", ".join(missing), flush=True)
    if failed or missing:
        raise SystemExit(1)


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

        def rpc_any(method, params, timeout=90):
            nonlocal seq
            seq += 1
            request = {"jsonrpc": "2.0", "id": seq, "method": method, "params": params}
            if proc.poll() is not None:
                raise RuntimeError("MCP process exited")
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            while True:
                if not select.select([proc.stdout], [], [], timeout)[0]:
                    raise TimeoutError(method)
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("MCP process exited")
                response = json.loads(line)
                if response.get("id") == seq:
                    transcript.append({"request": request, "response": response})
                    return response

        def rpc(method, params):
            response = rpc_any(method, params, timeout=60)
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

        admission = []
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
            context = tool("codebase", {"action": "get_context", "codebase": "fixture"})
            assert marker not in json.dumps(context)
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
            passed("strata refusals stay typed; admitted reads still answer")

            ctx = prepare_ctx(root, mem_id=node_id)
            second = tool("smart_ingest", {"content": "ADMISSION_FIXTURE_B", "forceCreate": True})
            ctx["mem_id_2"] = second.get("nodeId") or ctx["mem_id_2"]
            live = type("Live", (), {})()
            live.proc = proc
            live.rpc = rpc_any
            rows, missing = admission_rows(live, ctx)
            admission.extend(rows)
            assert_no_sqlite()
            finish_admission(rows, missing)
            passed("every advertised tool action is admitted on this binary")
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
            if output:
                output.write_text(json.dumps({
                    "catalog": locals().get("catalog", []),
                    "coverage": coverage,
                    "admission": admission,
                    "transcript": transcript,
                }, indent=2) + "\n")
    print(f"PASS {len(coverage)} contract calls; admission sweep admitted every advertised action")


def admit_only(binary, output):
    """Every tools/list tool and every advertised action, against one binary."""
    with tempfile.TemporaryDirectory(prefix="vestige-tool-frontier-") as temp:
        root = Path(temp)
        ctx = prepare_ctx(root)
        mcp = StdioMcp(binary, root / "store")
        rows, missing = [], []
        mcp.spawn()
        try:
            mcp.handshake()
            seed_ids(mcp, ctx)
            rows, missing = admission_rows(mcp, ctx)
        finally:
            mcp.close()
            if output:
                output.write_text(json.dumps({
                    "admission": rows,
                    "transcript": mcp.transcript,
                }, indent=2) + "\n")
        finish_admission(rows, missing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=Path("target/debug/vestige-mcp"),
        help="vestige-mcp binary. Default is target/debug/vestige-mcp from a "
             "default-feature build. Point this at a release/v4.0.0-rc binary "
             "to run the same sweep there.",
    )
    parser.add_argument(
        "--admit-only", action="store_true",
        help="Skip the Strata contract prelude and run only the admission sweep. "
             "Use with --binary for release/v4.0.0-rc.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"binary not found: {args.binary}")
    if args.admit_only:
        admit_only(args.binary, args.output)
    else:
        run(args.binary, args.output)
