#!/usr/bin/env python3
"""
Operator Lite -- Cursor port hook adapter.

Cursor's agent hooks (https://cursor.com/docs/agent/hooks) run a command with
a JSON payload on stdin for `beforeShellExecution`, `beforeMCPExecution` and
`preToolUse`. A permission hook BLOCKS the action by printing a JSON decision
on stdout:

    {"continue": true, "permission": "deny",
     "user_message": "<shown in the client>",
     "agent_message": "<sent to the agent>"}

exit code 2 also blocks, but only the stdout JSON fields are documented to
reach the agent, so this shim always decides through stdout JSON at exit 0.
Crucially, permission hooks block on INVALID output even when `failClosed` is
false -- so on ALLOW the shim prints the schema-valid minimal decision
(`{"continue": true, "permission": "allow"}`) and never an empty stdout.

That decision protocol is not the gate's (exit 2 + stderr), so this shim
translates field names AND the verdict:

    cursor stdin : common envelope {conversation_id, generation_id, model,
                    hook_event_name, cursor_version, workspace_roots, ...}
                   + beforeShellExecution : {command, cwd, sandbox}
                   + beforeMCPExecution   : {tool_name, tool_input (JSON
                                            string!), mcp_server_name, ...}
                   + preToolUse           : {tool_name, tool_input (object),
                                             tool_use_id, cwd, ...}
    gate stdin   : {"tool_name", "tool_input", "cwd", "session_id"}

Mappings performed (and only these):
  command (top level)      -> tool_name "Shell" + tool_input {"command": ...}
  conversation_id          -> session_id
  cwd, else workspace_roots[0] -> cwd
  beforeMCPExecution tool_name + mcp_server_name
                           -> "mcp__<server>__<tool>" so the gate's MCP rules
                              (OP-S02 message/send, OP-S13 secret args and
                              sensitive paths) fire; tool_input is parsed from
                              its documented JSON-string form into an object
  preToolUse tool_name     -> passed through lowercased by the gate; Cursor's
                              "MCP:<tool>" spelling normalizes to "mcp__<tool>"

    gate exit 0 -> allow  (receipt already written by the gate)
    gate exit 2 -> deny   (gate's stderr verdict becomes agent_message)
    gate unreachable -> fail OPEN unless the text plainly looks destructive
                        (mirror of the gate's own degraded_hit), in which case
                        deny. hooks.json also ships `failClosed: true`, so a
                        shim that cannot run at all blocks rather than allows.

`preToolUse` is registered with a matcher for the native file tools only --
Shell and MCP are already covered by the two dedicated hooks, and forwarding
the same action twice would consume a single-use owner permit on the first
invocation and block the second. The gate, not this shim, decides relevance.

stdlib only. No tool call is ever executed here: this process only pipes JSON.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

GATE_TIMEOUT_S = 8  # below hooks.json's 10s so we can self-degrade first

ALLOW = '{"continue": true, "permission": "allow"}'

DESTRUCTIVE_LIKE = re.compile(
    r"\brm\s+-[a-zA-Z]*[rR]|\bmv\s+\S*(vestige|Developer|\.zcode|\.claude)"
    r"|\bpush\s+.*--force|\bDROP\s+TABLE|\.vestige/vestige\.db|fly\s+deploy"
    r"|OPERATOR_CANARY_STOP_7f3a|mkfs|\bdd\s+if=",
    re.I,
)

# The gate prints `<how to run it> approve <digest>`: `operator-gate approve` when its
# launcher is on PATH, else `python3 ~/.operator/gate/operator-gate.py approve`.
APPROVE_RE = re.compile(r"((?:python3? \S*)?operator-gate(?:\.py)?) approve ([0-9a-f]{24})")
RULE_LINE_RE = re.compile(r"OPERATOR: STOPPED \(([^)]+)\)")


def gate_path():
    env = os.environ.get("OPERATOR_GATE")
    if env:
        return env
    here = Path(__file__).resolve().parent
    candidates = [
        Path.home() / ".operator" / "gate" / "operator-gate.py",
        # repo checkouts: ports/cursor -> operator-lite/operator-gate.py
        here / ".." / ".." / ".." / "operator-gate.py",
        here / ".." / ".." / "operator-gate.py",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return str(candidates[0])


def emit(decision):
    """Print the schema-valid stdout JSON Cursor reads. Never empty."""
    sys.stdout.write(json.dumps(decision) + "\n")
    sys.stdout.flush()
    return 0


def allow():
    return emit(json.loads(ALLOW))


def deny(gate_stderr, raw):
    """Translate the gate's exit-2 verdict into Cursor's documented deny JSON."""
    text = (gate_stderr or "").strip() or "Operator Lite blocked this action."
    m = RULE_LINE_RE.search(text)
    user = "Operator Lite: blocked (%s)." % m.group(1) if m else "Operator Lite: blocked."
    a = APPROVE_RE.search(text)
    if a:
        user += " Owner can allow once: %s approve %s" % (a.group(1), a.group(2))
    return emit({
        "continue": True,
        "permission": "deny",
        "user_message": user[:500],
        "agent_message": text[:1500],
    })


def degraded(raw, reason):
    """No gate verdict available: fail open unless plainly destructive text."""
    sys.stderr.write(reason + "\n")
    if DESTRUCTIVE_LIKE.search(raw or ""):
        return deny("Operator Lite: gate unreachable and the command looks "
                    "destructive -- blocked. Ask the owner.", raw)
    return allow()


def resolve_cwd(p):
    cwd = p.get("cwd")
    if isinstance(cwd, str) and cwd:
        return cwd
    roots = p.get("workspace_roots")
    if isinstance(roots, list) and roots and isinstance(roots[0], str) and roots[0]:
        return roots[0]
    return os.getcwd()


def run_gate(gate_payload):
    cmd = [
        os.environ.get("OPERATOR_PYTHON") or sys.executable,
        gate_path(),
        "hook",
        "--source",
        "cursor",
    ]
    env = dict(os.environ, OPERATOR_AGENT_SESSION="1")
    return subprocess.run(
        cmd,
        input=json.dumps(gate_payload),
        capture_output=True,
        text=True,
        timeout=GATE_TIMEOUT_S,
        env=env,
    )


def shell_payload(p):
    """beforeShellExecution: {command, cwd, sandbox} + envelope."""
    cmd = p.get("command")
    if not isinstance(cmd, str):
        return None
    return {
        "tool_name": "Shell",
        "tool_input": {"command": cmd},
        "cwd": resolve_cwd(p),
        "session_id": p.get("conversation_id"),
    }


def mcp_payload(p):
    """beforeMCPExecution: {tool_name, tool_input (JSON string), mcp_server_name}."""
    tool = str(p.get("tool_name") or "")
    server = str(p.get("mcp_server_name") or "")
    ti = p.get("tool_input")
    if isinstance(ti, str):
        try:
            ti = json.loads(ti)
        except Exception:
            pass  # keep the raw string; the gate wraps unknown shapes itself
    return {
        "tool_name": "mcp__%s__%s" % (server, tool) if server else "mcp__%s" % tool,
        "tool_input": ti if ti is not None else {},
        "cwd": resolve_cwd(p),
        "session_id": p.get("conversation_id"),
    }


def tool_payload(p):
    """preToolUse: {tool_name, tool_input (object), cwd, ...} -- near gate-native."""
    tool = str(p.get("tool_name") or "")
    if tool.startswith("MCP:"):
        tool = "mcp__" + tool[len("MCP:"):]
    ti = p.get("tool_input")
    if isinstance(ti, str):
        try:
            ti = json.loads(ti)
        except Exception:
            pass
    return {
        "tool_name": tool or "unknown",
        "tool_input": ti if isinstance(ti, dict) else (ti or {}),
        "cwd": resolve_cwd(p),
        "session_id": p.get("conversation_id"),
    }


def main(raw):
    try:
        p = json.loads(raw) if raw.strip() else {}
        if not isinstance(p, dict):
            return allow()  # malformed hook payload: no decision -> allow
    except Exception:
        return allow()

    # Event dispatch on payload shape (the envelope's hook_event_name agrees):
    # beforeMCPExecution carries mcp_server_name; preToolUse carries
    # tool_name+tool_input; beforeShellExecution carries only top-level command.
    if "mcp_server_name" in p:
        gate_payload = mcp_payload(p)
    elif "tool_name" in p and "tool_input" in p:
        gate_payload = tool_payload(p)
    else:
        gate_payload = shell_payload(p)
    if gate_payload is None:
        return allow()

    try:
        proc = run_gate(gate_payload)
    except subprocess.TimeoutExpired:
        return degraded(raw, "Operator Lite: gate timed out after %ss." % GATE_TIMEOUT_S)
    except Exception as exc:
        return degraded(raw, "Operator Lite: gate unreachable (%s)." % exc)

    code = proc.returncode
    if code == 0:
        return allow()  # gate receipted the call and allows
    if code == 2 and "OPERATOR:" in (proc.stderr or ""):
        return deny(proc.stderr, raw)  # gate verdict -> documented deny JSON
    # Exit 2 WITHOUT the gate's verdict marker is not a decision (python itself
    # exits 2 on "can't open file"), and any other exit is "gate unreachable"
    # (missing interpreter, unreadable file, internal crash) -- degraded, not a block.
    return degraded(
        raw,
        "Operator Lite: gate exited %s (%s)."
        % (code, (proc.stderr or "").strip()[:200]),
    )


if __name__ == "__main__":
    try:
        _raw = sys.stdin.read()
        sys.exit(main(_raw))
    except BrokenPipeError:
        sys.exit(0)
