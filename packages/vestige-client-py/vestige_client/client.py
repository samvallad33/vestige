"""Zero-config sync & async IPC / MCP stdio client for Vestige."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


class VestigeError(RuntimeError):
    """Raised when the Vestige kernel returns an error or reason code."""


@dataclass(frozen=True)
class ProofReceipt:
    """Proof-carrying response wrapper returned by Vestige MCP tools."""

    node_id: Optional[str] = None
    receipt_id: Optional[str] = None
    canonical_hash: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


def default_reflex_socket() -> Path:
    home = Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or "~").expanduser()
    return home / ".vestige" / "reflex.sock"


class VestigeClient:
    """Zero-config synchronous client for Vestige.

    Auto-detects `~/.vestige/reflex.sock` or spawns `vestige-mcp` over stdio
    MCP JSON-RPC on demand. Zero env vars, zero API keys, zero cloud accounts.
    """

    def __init__(
        self,
        *,
        mcp_bin: str = "vestige-mcp",
        socket_path: Optional[Path] = None,
        transport: Optional[Callable[[str, dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        self.mcp_bin = mcp_bin
        self.socket_path = socket_path or default_reflex_socket()
        self._transport = transport
        self._proc: Optional[subprocess.Popen[str]] = None
        self._req_id = 0
        self._initialized = False

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _ensure_stdio(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [self.mcp_bin],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
            self._initialized = False
        if not self._initialized:
            self._rpc_stdio(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "vestige-client-py", "version": "7.0.0"},
                },
            )
            self._initialized = True
        return self._proc

    def _rpc_stdio(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise VestigeError("vestige-mcp stdio transport is not open")
        req = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise VestigeError("vestige-mcp closed stdout unexpectedly")
            msg = json.loads(line)
            if "id" in msg:
                if "error" in msg:
                    raise VestigeError(msg["error"].get("message", str(msg["error"])))
                return msg.get("result", {})

    def call_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(tool, arguments)

        if hasattr(socket, "AF_UNIX") and self.socket_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(str(self.socket_path))
                    payload = json.dumps({"tool": tool, "arguments": arguments}) + "\n"
                    sock.sendall(payload.encode("utf-8"))
                    data = sock.recv(65536).decode("utf-8").strip()
                    if data:
                        return json.loads(data)
            except OSError:
                pass

        self._ensure_stdio()
        res = self._rpc_stdio("tools/call", {"name": tool, "arguments": arguments})
        content = res.get("content", [])
        if content and isinstance(content, list) and "text" in content[0]:
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                return {"text": content[0]["text"]}
        return res

    def session(self, action: str = "start", **kwargs: Any) -> dict[str, Any]:
        return self.call_tool("session", {"action": action, **kwargs})

    def remember(
        self,
        content: str,
        *,
        node_type: str = "note",
        tags: Optional[list[str]] = None,
        files: Optional[list[str]] = None,
        links: Optional[list[str]] = None,
        scope: str = "user",
    ) -> ProofReceipt:
        args: dict[str, Any] = {
            "action": "save",
            "content": content,
            "node_type": node_type,
            "scope": scope,
        }
        if tags:
            args["tags"] = tags
        if files:
            args["files"] = files
        if links:
            args["links"] = links
        res = self.call_tool("remember", args)
        return ProofReceipt(
            node_id=res.get("nodeId") or res.get("id"),
            receipt_id=res.get("receiptId"),
            canonical_hash=res.get("canonicalHash"),
            raw=res,
        )

    def recall(
        self,
        handle: Optional[str] = None,
        *,
        query: Optional[str] = None,
        limit: int = 10,
        scope: str = "user",
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"limit": limit, "scope": scope}
        if handle is not None:
            args["handle"] = handle
        if query is not None:
            args["query"] = query
        return self.call_tool("recall", args)

    def causal_walk(
        self, from_handle: str, *, direction: str = "backward", depth: int = 5
    ) -> dict[str, Any]:
        return self.call_tool(
            "causal_walk",
            {"from_handle": from_handle, "direction": direction, "depth": depth},
        )

    def curate(self, action: str, handle: str, **kwargs: Any) -> dict[str, Any]:
        return self.call_tool("curate", {"action": action, "handle": handle, **kwargs})

    def precedent(self, action: str = "list", **kwargs: Any) -> dict[str, Any]:
        return self.call_tool("precedent", {"action": action, **kwargs})

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=2)
            self._proc = None

    def __enter__(self) -> "VestigeClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncVestigeClient:
    """Async wrapper delegating cleanly to `VestigeClient` via `asyncio.to_thread`."""

    def __init__(self, **kwargs: Any) -> None:
        self._sync = VestigeClient(**kwargs)

    async def remember(self, content: str, **kwargs: Any) -> ProofReceipt:
        return await asyncio.to_thread(self._sync.remember, content, **kwargs)

    async def recall(self, handle: Optional[str] = None, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.recall, handle, **kwargs)

    async def causal_walk(self, from_handle: str, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.causal_walk, from_handle, **kwargs)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)
