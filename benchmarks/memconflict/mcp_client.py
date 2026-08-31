"""Minimal JSON-RPC-over-stdio client for the vestige-mcp server.

Protocol notes learned from the real server (v2.4.1):

  * Framing is line-delimited JSON on stdout (not LSP Content-Length framing).
  * Order is: initialize -> notifications/initialized -> tools/call.
  * stdin MUST stay open for the lifetime of the session. Closing it ends the
    server mid-run.
  * The embedding service warms up asynchronously AFTER initialize returns.
    Issuing tools/call before it is ready silently measures a degraded
    keyword-only fallback path, which would make every retrieval number wrong
    in a way that does not look wrong. We therefore block for a mandatory
    warmup window and record the observed warmup in the results file.
"""
from __future__ import annotations

import json
import os
import pathlib
import queue
import subprocess
import threading
import time
from typing import Any, Dict, Optional

DEFAULT_WARMUP_SECONDS = 45.0


class VestigeMCPError(RuntimeError):
    pass


class VestigeMCP:
    def __init__(
        self,
        binary: str,
        data_dir: str,
        warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
        timeout: float = 300.0,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.binary = str(pathlib.Path(binary).resolve())
        self.data_dir = str(pathlib.Path(data_dir).resolve())
        self.warmup_seconds = warmup_seconds
        self.timeout = timeout
        self._id = 0
        self._proc: Optional[subprocess.Popen] = None
        self._out: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stderr_lines: list[str] = []
        self._extra_env = dict(extra_env or {})
        self.server_info: Dict[str, Any] = {}
        self.observed_warmup: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "VestigeMCP":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        pathlib.Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["VESTIGE_DATA_DIR"] = self.data_dir
        env["VESTIGE_DASHBOARD_ENABLED"] = "false"
        env["VESTIGE_HTTP_ENABLED"] = "0"
        env.update(self._extra_env)

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
            self._stderr_lines.append(line.rstrip())

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=15)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    # -- transport ---------------------------------------------------------

    def _send(self, payload: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise VestigeMCPError("server not running")
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _await_id(self, want_id: int, timeout: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VestigeMCPError(f"timeout waiting for response id={want_id}")
            try:
                line = self._out.get(timeout=remaining)
            except queue.Empty:
                raise VestigeMCPError(f"timeout waiting for response id={want_id}")
            if line is None:
                tail = "\n".join(self._stderr_lines[-20:])
                raise VestigeMCPError(f"server exited early. stderr tail:\n{tail}")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON noise on stdout is ignored
            if msg.get("id") == want_id:
                return msg

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None) -> Dict[str, Any]:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        msg = self._await_id(rid, timeout if timeout is not None else self.timeout)
        if "error" in msg:
            raise VestigeMCPError(f"{method} failed: {msg['error']}")
        return msg.get("result", {})

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # -- MCP handshake -----------------------------------------------------

    def initialize(self) -> Dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "vestige-memconflict-harness", "version": "1.0"},
            },
            timeout=120.0,
        )
        self.server_info = result.get("serverInfo", {})
        self.notify("notifications/initialized")

        # Mandatory embedding warmup. See module docstring.
        t0 = time.monotonic()
        time.sleep(self.warmup_seconds)
        self.observed_warmup = {
            "requested_seconds": self.warmup_seconds,
            "actual_seconds": round(time.monotonic() - t0, 2),
            "embedding_ready_signal": self.embedding_log_signal(),
        }
        return result

    def embedding_log_signal(self) -> Optional[str]:
        """Best-effort: surface any embedding-related server log line.

        Recorded in results so a reader can confirm the run did not silently
        fall back to keyword-only retrieval.
        """
        for line in reversed(self._stderr_lines):
            low = line.lower()
            if "embed" in low or "model" in low:
                return line[-300:]
        return None

    def stderr_tail(self, n: int = 40) -> list[str]:
        return self._stderr_lines[-n:]

    # -- tools -------------------------------------------------------------

    def call_tool(self, name: str, arguments: Dict[str, Any],
                  timeout: Optional[float] = None) -> Any:
        result = self.request(
            "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
        )
        content = result.get("content") or []
        for block in content:
            if block.get("type") == "text":
                text = block.get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return result
