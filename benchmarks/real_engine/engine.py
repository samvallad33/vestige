"""Actual repository-built Vestige child, isolated per memory lifetime.

This module performs no retrieval or fact extraction on the host. Every memory
write and read is an MCP tools/call against the hashed release binary.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import weakref
from typing import Any

_CLIENT_PATH = Path(__file__).resolve().parents[1] / "memconflict" / "mcp_client.py"
_SPEC = importlib.util.spec_from_file_location("real_engine_memconflict_client", _CLIENT_PATH)
_CLIENT = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_CLIENT)
VestigeMCP = _CLIENT.VestigeMCP
VestigeMCPError = _CLIENT.VestigeMCPError

# Restrict host-facing tools, not the engine's memory semantics. In particular,
# precedent/reflex/codebase are excluded: they can inspect machine-wide history.
MEMORY_TOOLS = frozenset({"smart_ingest", "recall", "memory", "session_start",
                          "memory_status", "causal_walk", "ghostlink", "receipt",
                          "graph", "forgotten_lesson"})
POLICY_FILES = ("commitments.json", "state/commitments-validated.json", "mode", "STRICT")
ACTIVE_ENGINES = weakref.WeakSet()


def close_all_engines():
    """Close only children created by this campaign, including partial starts."""
    engines = list(ACTIVE_ENGINES)
    # End native pipes first so any pending MCP request exits before hook servers.
    for engine in engines:
        client=engine.client
        proc=client._proc if client else None
        if proc:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except (ProcessLookupError,OSError):
                pass  # A concurrent item finally already ended this child.
    failures=[]
    for engine in engines:
        try:
            engine.close()
        except Exception as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError('Owned engine cleanup failures: '+repr(failures))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_build(repo: str | Path, manifest_path: str | Path | None = None) -> dict:
    repo = Path(repo).resolve()
    path = Path(manifest_path) if manifest_path else repo / "runs/build-manifest.json"
    manifest = json.loads(path.read_text())
    binary = repo / "target/release/vestige-mcp"
    if Path(manifest["binary"]).as_posix() != "target/release/vestige-mcp":
        raise RuntimeError("build manifest names an unexpected binary")
    if binary.is_symlink() or not binary.is_file():
        raise RuntimeError("repository release binary missing")
    if sha256(binary) != manifest["binary_sha256"]:
        raise RuntimeError("release binary hash differs from build manifest")
    required = ["cargo", "build", "--release", "--locked", "-p", "vestige-mcp", "--bin", "vestige-mcp"]
    if manifest.get("command") != required:
        raise RuntimeError("build provenance is not the required locked repository build")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != manifest["source_commit"]:
        # Campaign-only commits may follow the engine build; source code changes
        # are rejected below rather than requiring a redundant rebuild.
        changed = subprocess.check_output(["git", "diff", manifest["source_commit"], "--name-only", "--", "Cargo.toml", "Cargo.lock", "crates"], cwd=repo, text=True)
        if changed.strip():
            raise RuntimeError("engine source changed since release build")
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--", "Cargo.toml", "Cargo.lock", "crates"], cwd=repo, text=True)
    if changed.strip():
        raise RuntimeError("dirty engine source differs from release build")
    return manifest


def isolate_policy(source: Path, dest: Path) -> dict:
    if dest.exists():
        raise RuntimeError("operator home must be fresh")
    dest.mkdir(parents=True, mode=0o700)
    pins = {}
    for name in POLICY_FILES:
        src, dst = source / name, dest / name
        if src.exists():
            if src.is_symlink() or not src.is_file():
                raise RuntimeError(f"policy file is not regular: {name}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            os.chmod(dst, 0o600)
            pins[name] = {"source_sha256": sha256(src), "copied_sha256": sha256(dst)}
            if pins[name]["source_sha256"] != pins[name]["copied_sha256"]:
                raise RuntimeError("policy copy changed")
        else:
            pins[name] = None
    if pins["commitments.json"] is None or pins["state/commitments-validated.json"] is None:
        raise RuntimeError("owner policy and validated policy cache are required")
    mode = (dest / "mode").read_text().strip().lower() if (dest / "mode").exists() else "enforce"
    if mode in {"off", "disabled", "observe", "shadow"}:
        raise RuntimeError(f"unsupported owner policy mode: {mode}")
    effective = os.environ.get("OPERATOR_MODE", mode)
    if effective.lower() in {"off", "disabled", "observe", "shadow"}:
        raise RuntimeError("effective Operator mode disables enforcement")
    return {"files": pins, "mode": mode, "effective_mode": effective}


class RealEngine:
    """One process and one fresh store for a single independent memory group."""
    def __init__(self, repo: str | Path, run_dir: str | Path,
                 operator_source: str | Path | None = None,
                 manifest_path: str | Path | None = None, warmup: float = 0,
                 hooks_enabled: bool = False, hook_model_callback=None,
                 hook_audit_path: str | Path | None = None,
                 hook_profile: str = "all_applicable_hooks"):
        self.repo = Path(repo).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.manifest = verify_build(self.repo, manifest_path)
        self.binary = self.repo / "target/release/vestige-mcp"
        self.operator_source = Path(operator_source or os.environ.get("OPERATOR_HOME", str(Path.home() / ".operator"))).resolve()
        self.warmup = warmup
        self.hooks_enabled = hooks_enabled
        self.hook_model_callback = hook_model_callback
        self.hook_audit_path = hook_audit_path
        self.hook_profile = hook_profile
        self.hooks = None
        self.client = None
        self._call_lock = threading.RLock()
        self._close_lock = threading.RLock()
        self.tools: list[dict] = []
        self.identity: dict = {"binary_sha256": self.manifest["binary_sha256"], "source_commit": self.manifest["source_commit"]}
        self.audit = self.run_dir / "mcp.jsonl"

    def _log(self, data: dict) -> None:
        from .runtime import append_jsonl
        append_jsonl(self.audit,{'timestamp':time.time(),**data})

    def start(self):
        if self.run_dir.exists():
            raise RuntimeError("engine run_dir already exists; create a fresh attempt directory")
        self.run_dir.mkdir(parents=True, mode=0o700)
        ACTIVE_ENGINES.add(self)
        policy = isolate_policy(self.operator_source, self.run_dir / "operator")
        self.identity["operator_policy"] = policy
        env = {"OPERATOR_HOME": str(self.run_dir / "operator"), "OPERATOR_MODE": policy["effective_mode"]}
        if self.hooks_enabled:
            from .hooks import HookController
            self.hooks = HookController(self, self.hook_audit_path, self.hook_model_callback,
                                        profile=self.hook_profile)
            env.update(self.hooks.prepare())
        self.client = VestigeMCP(str(self.binary), str(self.run_dir / "strata"), warmup_seconds=self.warmup, extra_env=env)
        try:
            self.client.start()
            self.client.initialize()
            native = self.client.request("tools/list")["tools"]
            self.tools = [x for x in native if x["name"] in MEMORY_TOOLS]
            if not {"smart_ingest", "recall", "memory"}.issubset({x["name"] for x in self.tools}):
                raise RuntimeError("required native memory tools missing")
            self.identity.update({"pid": self.client._proc.pid, "data_dir": str(self.run_dir / "strata"),
                                  "operator_policy": policy, "server_info": self.client.server_info,
                                  "transport": "stdio-jsonrpc", "client": "benchmarks/memconflict/mcp_client.VestigeMCP"})
            fresh = self._call("memory_status", {"view": "coverage"})
            self.identity["fresh_store_nodes"] = fresh["totalNodes"]
            if self.identity["fresh_store_nodes"] != 0:
                raise VestigeMCPError("new trial store was not empty before SessionStart hooks")
            if self.hooks:
                self.hooks.probe_native()
                output = self.hooks.event("SessionStart", {"source": "startup"})
                self.hooks.pending_context.extend(output["context"])
                self.identity["hooks"] = {"enabled": True, "wired": self.hooks.wired,
                    "source_pins": self.hooks.audit["source_pins"], "cli_sha256": self.hooks.cli_sha,
                    "profile_sha256": self.hooks.profile_sha,
                    "adapter_sha256": sha256(Path(__file__).with_name("hooks.py")),
                    "engine_adapter_sha256": sha256(Path(__file__)),
                    "native_endpoint_capabilities": self.hooks.native_status, "model": "gpt-5.4-nano"}
            self._log({"kind": "engine_start", "identity": self.identity, "tools": self.tools})
            return self
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self.start()

    def close(self):
        with self._close_lock:
            try:
                if self.hooks:
                    self.hooks.close()
            finally:
                self.hooks = None
                if self.client:
                    self.client.close()
                    self.client = None
                ACTIVE_ENGINES.discard(self)

    def __exit__(self, *exc):
        self.close()

    def call(self, name: str, arguments: dict) -> Any:
        if not self.client or not self.client._proc or self.client._proc.poll() is not None:
            raise VestigeMCPError('benchmark engine is not running')
        if self.hooks:
            before = self.hooks.event("PreToolUse", {"tool_name": "mcp__vestige__" + name, "tool_input": arguments})
            if before["blocked"]:
                from .hooks import HookViolation
                raise HookViolation("PreToolUse blocked: " + "\n".join(before["reasons"]))
            self.hooks.pending_context.extend(before["context"])
        with self._call_lock:
            result = self._call(name, arguments)
        if self.hooks:
            after = self.hooks.event("PostToolUse", {"tool_name": "mcp__vestige__" + name, "tool_input": arguments, "tool_response": result})
            self.hooks.pending_context.extend(after["context"])
        return result

    def _call(self, name: str, arguments: dict) -> Any:
        if not self.client or not self.client._proc or self.client._proc.poll() is not None:
            raise VestigeMCPError("benchmark engine is not running")
        if name not in {x["name"] for x in self.tools}:
            raise ValueError(f"tool is outside memory boundary: {name}")
        self._log({"kind": "tool_request", "name": name, "arguments": arguments})
        try:
            result = self.client.call_tool(name, arguments)
        except BaseException as exc:
            self._log({"kind": "tool_error", "name": name, "error": str(exc),
                       "raw_mcp_result": self.client.last_tool_result})
            raise
        self._log({"kind": "tool_response", "name": name, "result": result, "raw_mcp_result": self.client.last_tool_result})
        return result

    def ingest_history(self, history: list[str]) -> list[dict]:
        results = []
        if any(not isinstance(text, str) for text in history):
            raise TypeError("history must contain raw strings only")
        for begin in range(0, len(history), 20):
            records = [{"content": text, "node_type": "note", "tags": ["history", f"session-{begin+i}"],
                        "source": f"history:{begin+i}"} for i, text in enumerate(history[begin:begin+20])]
            # Native user-path batches (max20), never forceCreate, privileged
            # roles, pre-cleaned facts or host-side admission decisions.
            if len(records) == 1:
                results.append(self.call("smart_ingest", records[0]))
            else:
                batch = self.call("smart_ingest", {"items": records})
                rows = batch.get("results")
                if not isinstance(rows, list) or len(rows) != len(records):
                    raise VestigeMCPError("native batch result coverage mismatch")
                results.extend(rows)
        return results


class HookedRealEngine(RealEngine):
    """The sole scored Vestige arm: all source-wired applicable hooks enabled."""
    def __init__(self, *args, **kwargs):
        if kwargs.get("hooks_enabled") is False:
            raise ValueError("scored Vestige arm cannot disable hooks")
        super().__init__(*args, **{**kwargs, "hooks_enabled": True})

    def begin_turn(self, prompt):
        return self.hooks.begin_turn(prompt)

    def drain_context(self):
        return self.hooks.drain_context()

    def finish_turn(self, messages):
        return self.hooks.finish_turn(messages)
