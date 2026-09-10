"""Synchronous MCP stdio transport with bounded responses and explicit process ownership."""

import json
import os
import select
import signal
import subprocess
import time


class StdioMcp:
    def __init__(self, command, *, env=None, timeout=30, max_response_bytes=4_000_000):
        if not command or not all(isinstance(part, str) for part in command):
            raise ValueError("command argv required")
        if not 0 < timeout <= 300 or not 1024 <= max_response_bytes <= 64_000_000:
            raise ValueError("invalid transport bounds")
        self.command = list(command)
        self.env = env
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.process = None
        self.sequence = 0
        self.buffer = b""

    def __enter__(self):
        if self.process is not None:
            raise RuntimeError("transport is already open")
        if os.name != "posix":
            raise RuntimeError("stdio transport currently requires POSIX pipes")
        self.buffer = b""
        self.sequence = 0
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in ("PATH", "HOME", "SYSTEMROOT", "WINDIR", "TMPDIR")
        }
        environment.update(self.env or {})
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        )
        try:
            os.set_blocking(self.process.stdin.fileno(), False)
            self.rpc(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "vestige-runtime", "version": "3.0.0a1"},
                },
            )
            self._write(
                json.dumps(
                    {"jsonrpc": "2.0", "method": "notifications/initialized"}
                ).encode()
                + b"\n",
                time.monotonic() + self.timeout,
            )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
        process.stdin.close()
        process.stdout.close()
        self.process = None

    def _write(self, body, deadline):
        remaining_body = memoryview(body)
        while remaining_body:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([], [self.process.stdin], [], remaining)[1]
            ):
                raise TimeoutError("MCP request write timeout")
            try:
                written = os.write(self.process.stdin.fileno(), remaining_body)
            except BlockingIOError:
                continue
            remaining_body = remaining_body[written:]

    def rpc(self, method, params):
        if self.process is None:
            raise RuntimeError("transport is not open")
        self.sequence += 1
        body = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": self.sequence,
                    "method": method,
                    "params": params,
                },
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        if len(body) > self.max_response_bytes:
            raise ValueError("request exceeds transport bound")
        deadline = time.monotonic() + self.timeout
        self._write(body, deadline)
        while True:
            while b"\n" not in self.buffer:
                remaining = deadline - time.monotonic()
                if (
                    remaining <= 0
                    or not select.select([self.process.stdout], [], [], remaining)[0]
                ):
                    raise TimeoutError("MCP response timeout")
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError("MCP process exited before response")
                self.buffer += data
                if len(self.buffer) > self.max_response_bytes:
                    raise ValueError("MCP response exceeds transport bound")
            line, self.buffer = self.buffer.split(b"\n", 1)
            message = json.loads(line)
            if message.get("id") == self.sequence:
                if "error" in message:
                    raise RuntimeError("MCP JSON-RPC error")
                return message["result"]
            if time.monotonic() >= deadline:
                raise TimeoutError("MCP response timeout")

    def catalog(self):
        return self.rpc("tools/list", {})["tools"]

    def call(self, name, arguments):
        return self.rpc("tools/call", {"name": name, "arguments": arguments})
