"""Bounded local JSON-RPC client for the native AgentDojo host adapter."""

import hashlib
import json
import os
import selectors
import subprocess
import time
import uuid
from pathlib import Path


class NativeInterdictionError(RuntimeError):
    """Transport, identity or receipt failure: stop the trial before more effects."""


NativeInterdiction = NativeInterdictionError


def json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def canonical(value):
    return json.dumps(
        json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def body(result):
    if result.get("isError"):
        raise NativeInterdiction(
            "native tool error: " + canonical(result.get("content", []))
        )
    if "structuredContent" in result:
        return result["structuredContent"]
    text = next(
        (x["text"] for x in result.get("content", []) if x.get("type") == "text"), None
    )
    if not text:
        raise NativeInterdiction("native result has no body")
    return json.loads(text)


class NativeStack:
    def __init__(self, root, timeout=90, profile="production"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.timeout = timeout
        self.next = 0
        self.buffer = b""
        self.failed = False
        self.sessions = set()
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.endswith(("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN"))
        }
        if profile not in {"production", "all_rules"}:
            raise ValueError("unknown native enforcement profile")
        env.pop("VESTIGE_OPERATOR_PROFILE_JSON", None)
        if profile == "all_rules":
            env["VESTIGE_OPERATOR_PROFILE_JSON"] = str(
                Path(__file__).resolve().parents[3]
                / "evidence/native-audit/operator-all-rules-profile.json"
            )
        self.stderr = (self.root / "bridge.stderr").open("xb")
        self.process = subprocess.Popen(
            [
                "/opt/homebrew/bin/node",
                str(Path(__file__).with_name("native_bridge.mjs")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            env=env,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.identity = self.request("init", root=str(self.root))
            self.catalog = self.identity["tools"]
        except BaseException:
            self.abort()
            raise

    def request(self, method, **params):
        if self.failed:
            raise NativeInterdiction("FAIL_CLOSED_INTERDICTION: native host is held")
        self.next += 1
        request_id = self.next
        try:
            wire = (
                canonical(
                    {"id": request_id, "method": method, "params": params}
                ).encode()
                + b"\n"
            )
            self.process.stdin.write(wire)
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            while b"\n" not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.selector.select(remaining):
                    raise NativeInterdiction(
                        "FAIL_CLOSED_INTERDICTION: bridge deadline"
                    )
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise NativeInterdiction(
                        "FAIL_CLOSED_INTERDICTION: bridge transport dropped"
                    )
                self.buffer += chunk
                if len(self.buffer) > 8 * 1024 * 1024:
                    raise NativeInterdiction(
                        "FAIL_CLOSED_INTERDICTION: oversized bridge frame"
                    )
            line, self.buffer = self.buffer.split(b"\n", 1)
            response = json.loads(line)
            if (
                response.get("id") != request_id
                or "error" in response
                or "result" not in response
            ):
                raise NativeInterdiction(
                    "FAIL_CLOSED_INTERDICTION: "
                    + str(response.get("error", "bridge identity mismatch"))
                )
            return response["result"]
        except (Exception, KeyboardInterrupt) as exc:
            self.failed = True
            self.abort()
            if isinstance(exc, (NativeInterdiction, KeyboardInterrupt)):
                raise
            raise NativeInterdiction(
                "FAIL_CLOSED_INTERDICTION: bridge " + type(exc).__name__
            ) from exc

    def start(self, session, parent=None):
        result = self.request("start", session=session, parent=parent)
        self.sessions.add(session)
        return result

    def tool(self, session, name, args, call=None):
        return self.request(
            "tool",
            session=session,
            tool=name,
            args=json_value(args),
            call=call or "native-" + uuid.uuid4().hex,
        )

    def record(self, session, data):
        return self.request("record", session=session, data=json_value(data))

    def stop(self, session, parent=None):
        result = self.request("stop", session=session, parent=parent)
        self.sessions.remove(session)
        return result

    def close(self):
        try:
            self.request("close")
            self.process.stdin.close()
            self.process.wait(timeout=5)
        finally:
            self.abort()

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=5)
        self.selector.close()
        self.stderr.close()
