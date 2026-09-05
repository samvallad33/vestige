"""Vestige as a MemoryArena memory system.

MemoryArena (arXiv:2602.16313, https://github.com/ZexueHe/MemoryArena) drives a
memory backend through two duck-typed methods on a class it constructs once per
task: ``add_chunk(chunk: str)`` after every agent step, and
``wrap_user_prompt(prompt: str) -> str`` before every step. This module gives
that interface to a live ``vestige-mcp`` server over MCP stdio, so the agent,
the environment, the judge and the evaluator stay exactly upstream's and the
only variable that changes between arms is the memory layer.

Design decisions, all preregistered in
``docs/benchmarks/MEMORYARENA-PREREGISTRATION.md``:

* One ``vestige-mcp`` process per Python process, shared by every task. Tasks
  are isolated by Vestige's ``scope`` namespace (``includeCrossScope`` is off by
  default on recall), so a task never sees another task's memories. A fresh
  process per task would pay the embedding warm-up 40 times and prove nothing.
* ``add_chunk`` writes with ``forceCreate: true``. The unit the agent stored is
  the unit retrieval ranks, the same as upstream's BM25 and embedding arms.
* ``wrap_user_prompt`` renders the retrieved memories in the exact shape of
  upstream ``RAGMemorySystem.wrap_user_prompt`` (``<memory>`` blocks inside
  ``<memory_context>``, then ``User: <prompt>``), with the same default
  ``top_k`` of 3, so the agent prompt differs between arms only in which
  memories were chosen.
* Retrieval never runs against a half-ready server. The first instance blocks
  until the embedding runtime reports ready (or a probe recall returns a real
  ``semanticScore``). A keyword-only fallback is a different system; measuring
  it by accident is the failure this guard exists for.
* Every add and every wrap is appended to a JSONL sidecar with byte counts, so
  the blob-size confound (a bigger context earns more judge credit) can be
  checked after the run instead of argued about.

Standard library only. No pip install.

Environment variables:

  VESTIGE_MCP_BINARY            path to vestige-mcp (default: target/debug/vestige-mcp
                                two levels above this file, else `vestige-mcp` on PATH)
  VESTIGE_ARENA_DATA_DIR        Vestige data dir for the run (default: a fresh temp dir)
  VESTIGE_ARENA_TOP_K           memories per wrap (default 3, matches upstream RAG)
  VESTIGE_ARENA_READY_TIMEOUT   seconds to wait for embeddings (default 180)
  VESTIGE_ARENA_LOG             JSONL sidecar path (default <data dir>/vestige-arena-log.jsonl)
  VESTIGE_ARENA_ALLOW_KEYWORD_ONLY=1
                                proceed if embeddings never come up (logged loudly; the
                                run is then NOT the preregistered arm)
"""
from __future__ import annotations

import atexit
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

DEFAULT_TOP_K = 3
DEFAULT_READY_TIMEOUT = 180.0
PROBE_SCOPE = "arena_probe"
MEMORY_TAG = "memoryarena"


class VestigeError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Minimal JSON-RPC over stdio client. Same protocol lessons as
# benchmarks/memconflict/mcp_client.py: line-delimited JSON, initialize then
# notifications/initialized, stdin stays open for the life of the session,
# server notifications (no id) are skipped while waiting for a response.
# --------------------------------------------------------------------------
class _StdioServer:
    def __init__(self, binary: str, data_dir: str, timeout: float = 300.0) -> None:
        self.binary = str(pathlib.Path(binary).resolve())
        self.data_dir = str(pathlib.Path(data_dir).resolve())
        self.timeout = timeout
        self._id = 0
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._out: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stderr: List[str] = []
        self.server_info: Dict[str, Any] = {}

    def start(self) -> None:
        pathlib.Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["VESTIGE_DATA_DIR"] = self.data_dir
        env["VESTIGE_DASHBOARD_ENABLED"] = "false"
        env["VESTIGE_HTTP_ENABLED"] = "0"
        self._proc = subprocess.Popen(
            [self.binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "vestige-memoryarena-adapter", "version": "1.0"},
            },
            timeout=120.0,
        )
        self.server_info = result.get("serverInfo", {})
        self.notify("notifications/initialized")

    def _pump_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if line:
                self._out.put(line)
        self._out.put(None)

    def _pump_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr.append(line.rstrip())
            if len(self._stderr) > 2000:
                del self._stderr[:1000]

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def stderr_tail(self, n: int = 30) -> List[str]:
        return self._stderr[-n:]

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise VestigeError("vestige-mcp is not running")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _await(self, want_id: int, timeout: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VestigeError(f"timeout waiting for response id={want_id}")
            try:
                line = self._out.get(timeout=remaining)
            except queue.Empty:
                raise VestigeError(f"timeout waiting for response id={want_id}")
            if line is None:
                tail = "\n".join(self.stderr_tail())
                raise VestigeError(f"vestige-mcp exited. stderr tail:\n{tail}")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == want_id:
                return msg
            # Anything else is a server notification (warm-up logging, etc.).

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Dict[str, Any]:
        # MemoryArena's FastAPI server may call from worker threads; serialise.
        with self._lock:
            self._id += 1
            rid = self._id
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            msg = self._await(rid, timeout if timeout is not None else self.timeout)
        if "error" in msg:
            raise VestigeError(f"{method} failed: {msg['error']}")
        return msg.get("result", {})

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise VestigeError(f"tool {name} returned isError: {result}")
        for block in result.get("content") or []:
            if block.get("type") == "text":
                text = block.get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return result


# --------------------------------------------------------------------------
# Shared server + sidecar log
# --------------------------------------------------------------------------
_SERVER: Optional[_StdioServer] = None
_SERVER_LOCK = threading.Lock()
_LOG_PATH: Optional[pathlib.Path] = None
_LOG_LOCK = threading.Lock()
_READINESS: Dict[str, Any] = {}


def _default_binary() -> str:
    env = os.environ.get("VESTIGE_MCP_BINARY")
    if env:
        return env
    here = pathlib.Path(__file__).resolve()
    for candidate in (
        here.parents[2] / "target" / "debug" / "vestige-mcp",
        here.parents[2] / "target" / "release" / "vestige-mcp",
    ):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("vestige-mcp")
    if found:
        return found
    raise VestigeError(
        "vestige-mcp binary not found. Set VESTIGE_MCP_BINARY or run `cargo build -p vestige-mcp`."
    )


def _log(record: Dict[str, Any]) -> None:
    if _LOG_PATH is None:
        return
    record = dict(record)
    record["t"] = round(time.time(), 3)
    with _LOG_LOCK:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _find_key(value: Any, key: str) -> Any:
    """Depth-first search for `key` in a nested JSON value. None if absent."""
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for v in value.values():
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    elif isinstance(value, list):
        for v in value:
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    return None


def _results_of(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, dict):
        for key in ("results", "memories", "items"):
            if isinstance(response.get(key), list):
                return [r for r in response[key] if isinstance(r, dict)]
    return []


def _wait_ready(server: _StdioServer, timeout: float) -> Dict[str, Any]:
    """Block until the embedding runtime is ready.

    Two signals, tried in order on every poll: the `embeddingReady` flag that
    `memory_status(action="health")` reports, then a probe (ingest one memory in a
    dedicated scope, recall it, require a non-null `semanticScore`). Either one
    is enough. Neither within `timeout` is a hard failure unless
    VESTIGE_ARENA_ALLOW_KEYWORD_ONLY=1.
    """
    t0 = time.monotonic()
    probe_id: Optional[str] = None
    probe_text = "Vestige MemoryArena readiness probe: the lighthouse keeper feeds the gulls at dawn."
    polls = 0
    while True:
        polls += 1
        # Signal 1: the status flag.
        try:
            status = server.call_tool("memory_status", {"view": "health"})
            flag = _find_key(status, "embeddingReady")
            if flag is True:
                return {"ready": True, "method": "memory_status.embeddingReady",
                        "seconds": round(time.monotonic() - t0, 2), "polls": polls}
        except VestigeError:
            flag = None
        # Signal 2: the probe.
        try:
            if probe_id is None:
                created = server.call_tool(
                    "smart_ingest",
                    {"content": probe_text, "node_type": "note", "tags": [MEMORY_TAG, "probe"],
                     "scope": PROBE_SCOPE, "forceCreate": True},
                )
                probe_id = str(_find_key(created, "nodeId") or "")
            hits = _results_of(server.call_tool(
                "recall",
                {"query": "lighthouse keeper gulls dawn", "limit": 3, "scope": PROBE_SCOPE,
                 "detail_level": "summary"},
            ))
            if any(h.get("semanticScore") is not None for h in hits):
                return {"ready": True, "method": "probe.semanticScore",
                        "seconds": round(time.monotonic() - t0, 2), "polls": polls,
                        "status_flag_seen": flag}
        except VestigeError:
            pass
        if time.monotonic() - t0 >= timeout:
            record = {"ready": False, "method": None, "seconds": round(time.monotonic() - t0, 2),
                      "polls": polls, "status_flag_seen": flag,
                      "stderr_tail": server.stderr_tail(10)}
            if os.environ.get("VESTIGE_ARENA_ALLOW_KEYWORD_ONLY") == "1":
                record["WARNING"] = ("embeddings never became ready; this run is keyword-only and "
                                     "is NOT the preregistered vestige arm")
                return record
            raise VestigeError(
                f"embedding runtime not ready after {timeout:.0f}s; refusing to measure a "
                f"keyword-only fallback. Set VESTIGE_ARENA_ALLOW_KEYWORD_ONLY=1 to override. "
                f"stderr tail: {server.stderr_tail(10)}"
            )
        time.sleep(2.0)


def shared_server() -> _StdioServer:
    """Start (once) and return the process-wide vestige-mcp server."""
    global _SERVER, _LOG_PATH, _READINESS
    with _SERVER_LOCK:
        if _SERVER is not None:
            return _SERVER
        binary = _default_binary()
        data_dir = os.environ.get("VESTIGE_ARENA_DATA_DIR") or tempfile.mkdtemp(prefix="vestige-arena-datadir-")
        _LOG_PATH = pathlib.Path(os.environ.get("VESTIGE_ARENA_LOG") or
                                 pathlib.Path(data_dir) / "vestige-arena-log.jsonl")
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        server = _StdioServer(binary, data_dir)
        server.start()
        timeout = float(os.environ.get("VESTIGE_ARENA_READY_TIMEOUT", DEFAULT_READY_TIMEOUT))
        _READINESS = _wait_ready(server, timeout)
        _log({"op": "start", "binary": server.binary, "data_dir": server.data_dir,
              "server_info": server.server_info, "readiness": _READINESS,
              "top_k": int(os.environ.get("VESTIGE_ARENA_TOP_K", DEFAULT_TOP_K)),
              # names only: a ranking override must not go unrecorded, values may be paths
              "vestige_env_names": sorted(k for k in os.environ if k.startswith("VESTIGE_"))})
        atexit.register(server.close)
        _SERVER = server
        return server


def readiness() -> Dict[str, Any]:
    return dict(_READINESS)


def scope_for(user_id: str) -> str:
    """A scope name Vestige accepts, derived from MemoryArena's user_id (a uuid)."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(user_id))
    return f"arena_{cleaned[:32] or uuid.uuid4().hex[:32]}"


# --------------------------------------------------------------------------
# The MemoryArena-facing class
# --------------------------------------------------------------------------
class VestigeMemorySystem:
    """Duck-typed MemoryArena memory backend backed by a live vestige-mcp."""

    def __init__(self, user_id: Optional[str] = None, top_k: Optional[int] = None) -> None:
        self.user_id = str(user_id) if user_id is not None else str(uuid.uuid4())
        self.scope = scope_for(self.user_id)
        self.top_k = int(top_k if top_k is not None else os.environ.get("VESTIGE_ARENA_TOP_K", DEFAULT_TOP_K))
        self.server = shared_server()
        self.adds = 0
        self.wraps = 0
        _log({"op": "init", "user_id": self.user_id, "scope": self.scope, "top_k": self.top_k})

    # MemoryArena calls this after every step with agent.build_memory_entry(...)
    def add_chunk(self, chunk: str) -> Dict[str, Any]:
        if not chunk or not chunk.strip():
            return {"status": "skipped", "reason": "empty chunk"}
        t0 = time.monotonic()
        response = self.server.call_tool(
            "smart_ingest",
            {"content": chunk, "node_type": "event", "tags": [MEMORY_TAG],
             "scope": self.scope, "forceCreate": True},
        )
        self.adds += 1
        node_id = _find_key(response, "nodeId")
        _log({"op": "add", "scope": self.scope, "chars": len(chunk), "bytes": len(chunk.encode("utf-8")),
              "nodeId": node_id, "ms": round((time.monotonic() - t0) * 1000, 1)})
        return {"status": "ok", "nodeId": node_id}

    # MemoryArena calls this before every step with the full task prompt.
    def wrap_user_prompt(self, prompt: str) -> str:
        t0 = time.monotonic()
        hits: List[Dict[str, Any]] = []
        error: Optional[str] = None
        if self.adds > 0:
            try:
                hits = _results_of(self.server.call_tool(
                    "recall",
                    {"query": prompt, "limit": self.top_k, "scope": self.scope,
                     "detail_level": "summary", "retrieval_mode": "balanced"},
                ))
            except VestigeError as exc:  # never take the run down; record it
                error = str(exc)[:500]
        lines = ["<memory_context>"]
        if hits:
            for hit in hits:
                content = hit.get("content")
                if content:
                    lines.append(f"<memory>{content}</memory>")
        if len(lines) == 1:
            lines.append("None")
        lines.append("</memory_context>")
        lines.append(f"User: {prompt}")
        wrapped = "\n".join(lines)
        self.wraps += 1
        context_chars = len(wrapped) - len(prompt) - len("User: ") - 1
        _log({"op": "wrap", "scope": self.scope, "hits": len(hits),
              "semantic_null": sum(1 for h in hits if h.get("semanticScore") is None),
              "ids": [h.get("id") for h in hits],
              "query_chars": len(prompt), "context_chars": context_chars,
              "error": error, "ms": round((time.monotonic() - t0) * 1000, 1)})
        return wrapped


class NoMemorySystem:
    """The floor arm: no memory at all, same prompt shape as every other arm.

    Upstream's setup notes mention a "none" memory system but memory/server.py
    at the pinned revision has no such backend, so the floor is provided here.
    Anything an agent scores with this arm is the agent and the judge, not
    memory. Every other arm is read against it.
    """

    def __init__(self, user_id: Optional[str] = None) -> None:
        self.user_id = str(user_id) if user_id is not None else str(uuid.uuid4())

    def add_chunk(self, chunk: str) -> Dict[str, Any]:
        return {"status": "dropped"}

    def wrap_user_prompt(self, prompt: str) -> str:
        return "\n".join(["<memory_context>", "None", "</memory_context>", f"User: {prompt}"])
