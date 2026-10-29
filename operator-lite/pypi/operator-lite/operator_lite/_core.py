"""Shared Operator Lite gate plumbing for the pip package.

This is the package's single copy of the adapter pattern used by the repo
ports (operator-lite/ports/{langchain,crewai,openai-agents}/): locate the
external gate, spawn it with the exact hook contract, map exit codes to
verdicts, and apply the ports' fail-open/fail-closed policy when the gate is
unreachable. NO rule logic lives here -- the gate file owns every decision.

Gate resolution (superset of the ports, same precedence):
    1. $OPERATOR_GATE                         (re-read on every call)
    2. $OPERATOR_HOME/gate/operator-gate.py
    3. ~/.operator/gate/operator-gate.py

ensure_gate() downloads the gate from upstream (the same URL the ports'
install.sh uses) when no usable copy exists. Never overwrites a
same-or-newer gate. Refuses inside an agent session, like every install.

Python 3.9 compatible, stdlib only.
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

GATE_URL = ("https://raw.githubusercontent.com/samvallad33/vestige/main/"
            "operator-lite/operator-gate.py")
GATE_RELATIVE = os.path.join("gate", "operator-gate.py")

# Package version (the wheel's own line; the gate file has its VERSION).
VERSION = "6.0.1"

# Verdicts returned by decide()/gate_verdict() (same vocabulary as the ports).
PASS = "pass"          # not shell-ish -> no gating, call the handler
ALLOW = "allow"        # gate allowed the call (receipt already written by the gate)
BLOCK = "block"        # gate stopped it, or unreachable + destructive-like
UNREACHABLE = "unreachable"

GATE_TIMEOUT_S = float(os.environ.get("OPERATOR_GATE_TIMEOUT_S", "8"))

# Keys recognised as "carries a shell command" in a tool's args (the gate's
# own keys, plus the ports' fallbacks kept for parity).
COMMAND_KEYS = ("command", "cmd", "script")

# Fail-closed safety net for the gate-unreachable case ONLY -- never a rule
# engine. Verbatim in spirit from the ports (OpenClaw adapter's DESTRUCTIVE_LIKE,
# extended with shell-rc/init writes, which are code execution by install).
DESTRUCTIVE_LIKE = re.compile(
    r"\brm\s+-[a-zA-Z]*[rR]"                       # rm -r / -rf / -R
    r"|\bpush\s+.*--force"                          # force push
    r"|\bDROP\s+TABLE"
    r"|vestige\.db"
    r"|fly\s+deploy"
    r"|mkfs"
    r"|\bdd\s+if="
    r"|(?:>>|>|\btee\s+(?:-a\s+)?)[^\n|;&]*"
    r"\.(?:zshrc|zshenv|zprofile|zlogin|bashrc|bash_profile|profile|ssh/rc|config\.fish)\b",
    re.I)


# --------------------------------------------------------------------------- #
# gate location
# --------------------------------------------------------------------------- #
def operator_home():
    return os.environ.get("OPERATOR_HOME") or os.path.join(
        os.path.expanduser("~"), ".operator")


def resolve_gate():
    """Locate the gate. $OPERATOR_GATE wins; re-read every call so embedding
    processes can redirect it without reloading this module."""
    env = os.environ.get("OPERATOR_GATE")
    if env:
        return env
    for base in (operator_home(), os.path.join(os.path.expanduser("~"),
                                               ".operator")):
        candidate = os.path.join(base, GATE_RELATIVE)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(operator_home(), GATE_RELATIVE)


def gate_version(path):
    """The gate's VERSION string ("0.3.1"), or None when the file does not
    parse as a gate (mirrors install.sh's `^VERSION = ` sniff + compile check)."""
    try:
        with open(path, encoding="utf-8") as f:
            body = f.read()
    except OSError:
        return None
    match = re.search(r'^VERSION = *"([^"]+)"', body, re.M)
    if not match:
        return None
    try:
        compile(body, path, "exec")
    except (SyntaxError, ValueError):
        return None
    return match.group(1)


def _version_ge(a, b):
    """Dotted-numeric A >= B (install.sh's version_ge)."""
    parts_a = [int(p) for p in re.findall(r"\d+", a)]
    parts_b = [int(p) for p in re.findall(r"\d+", b)]
    for i in range(max(len(parts_a), len(parts_b))):
        pa = parts_a[i] if i < len(parts_a) else 0
        pb = parts_b[i] if i < len(parts_b) else 0
        if pa != pb:
            return pa > pb
    return True


class GateUnavailable(RuntimeError):
    """ensure_gate() could not produce a usable gate."""


def ensure_gate(min_version="0.3.1"):
    """Install-or-upgrade the gate at $OPERATOR_HOME/gate/operator-gate.py,
    downloading from upstream like the ports' install.sh. Returns the gate
    path.

    Policy (mirrors install.sh exactly):
      * refuses inside an agent session (installs are owner-run)
      * keeps an existing parseable gate that is same-or-newer
      * verifies the download declares VERSION and compiles before moving it
      * never leaves a partial download at the destination
    """
    if os.environ.get("OPERATOR_AGENT_SESSION"):
        raise GateUnavailable(
            "refusing: installs are run by the owner in their own terminal "
            "(OPERATOR_AGENT_SESSION is set)")
    dst = resolve_gate()
    existing = gate_version(dst) if os.path.exists(dst) else None
    if existing and _version_ge(existing, min_version):
        return dst
    tmp = dst + ".download"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with urllib.request.urlopen(GATE_URL, timeout=30) as resp:  # nosec - fixed upstream URL
            body = resp.read().decode("utf-8")
    except OSError as exc:
        if existing:
            return dst                      # keep the older-but-working gate
        raise GateUnavailable("could not download %s: %s" % (GATE_URL, exc))
    if not re.search(r'^VERSION = *"', body, re.M):
        if existing:
            return dst
        raise GateUnavailable("downloaded gate has no VERSION; refusing it")
    try:
        compile(body, dst, "exec")
    except (SyntaxError, ValueError) as exc:
        if existing:
            return dst
        raise GateUnavailable("downloaded gate does not compile: %s" % exc)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(tmp, 0o755)
    os.replace(tmp, dst)
    return dst


# --------------------------------------------------------------------------- #
# the gate call (one spawn implementation, all three adapters use it)
# --------------------------------------------------------------------------- #
def spawn_gate(command, tool_name, cwd=None, session_id=None,
               source="operator-lite", gate=None):
    """Pipe one command through the gate's exact stdin contract. Returns
    (returncode, stderr_text, spawn_error_or_None).

    Nothing here executes the command: it is piped to the gate as JSON and
    the gate is analyzer-only."""
    gate = gate or resolve_gate()
    if not os.path.exists(gate):
        # Load-bearing pre-check (ports keep the same one): a missing gate
        # would exit 2 under `python3 <missing>.py`, colliding with the
        # gate's STOP code and turning a broken install into bogus blocks.
        return (None, "", "gate not found at %s" % gate)
    payload = {"tool_name": str(tool_name or "shell"),
               "tool_input": {"command": command},
               "cwd": cwd or os.getcwd()}
    if session_id:
        payload["session_id"] = str(session_id)[:64]
    # Ports' shared guard: gate owner commands are refused inside agent sessions.
    env = dict(os.environ, OPERATOR_AGENT_SESSION="1")
    python = os.environ.get("OPERATOR_PYTHON") or sys.executable or "python3"
    try:
        proc = subprocess.run(
            [python, gate, "hook", "--source", source],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True, timeout=GATE_TIMEOUT_S, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return (None, "", str(exc))
    return (proc.returncode, proc.stderr.decode("utf-8", "replace").strip(), None)


# --------------------------------------------------------------------------- #
# shell-ish detection + the full decision (langchain-port semantics)
# --------------------------------------------------------------------------- #
def is_shellish(tool_name, args):
    """Shell-ish by name ("shell" anywhere in the tool name) or by shape
    (args carry a command key). The gate, not this list, decides relevance."""
    name = str(tool_name or "").lower()
    if "shell" in name:
        return True
    return isinstance(args, dict) and any(k in args for k in COMMAND_KEYS)


def extract_command(args):
    """The shell command string carried by a tool's args, or None."""
    if not isinstance(args, dict):
        return None
    for key in COMMAND_KEYS:
        value = args.get(key)
        if value is not None and value != "":
            return value
    return None


def _command_text(command):
    return command if isinstance(command, str) else " ".join(str(c) for c in command)


def gate_verdict(command, tool_name="shell", cwd=None, session_id=None,
                 source="operator-lite"):
    """(ALLOW, None) | (BLOCK, gate stderr) | (UNREACHABLE, error text).

    Exit 0 = allow (including shadow mode, which receipts instead of
    blocking); exit 2 = block; any other outcome degrades to allow -- the
    langchain port's exact mapping."""
    code, reason, error = spawn_gate(command, tool_name, cwd, session_id, source)
    if error is not None:
        return (UNREACHABLE, error)
    if code == 2:
        return (BLOCK, reason or "OPERATOR: STOPPED")
    return (ALLOW, None)


def decide(tool_name, args, cwd=None, session_id=None):
    """The full decision for one tool call (langchain-port semantics):
    (verdict, block_message_or_None). On BLOCK the handler must not run; the
    message starts with "Operator Lite blocked:" for the model."""
    if not is_shellish(tool_name, args):
        return (PASS, None)
    command = extract_command(args)
    if command is None:
        return (PASS, None)

    verdict, detail = gate_verdict(command, tool_name=tool_name, cwd=cwd,
                                   session_id=session_id)
    if verdict == ALLOW:
        return (ALLOW, None)
    if verdict == BLOCK:
        return (BLOCK, "Operator Lite blocked: %s" % detail)

    # Gate unreachable: fail-open, unless the raw text plainly looks
    # destructive -- then fail closed (the ports' shared policy).
    if DESTRUCTIVE_LIKE.search(_command_text(command)):
        return (BLOCK, "Operator Lite blocked: gate unreachable at %s and the "
                       "command looks destructive; ask the owner."
                       % resolve_gate())
    return (ALLOW, None)


def gate_status():
    """One-line ops snapshot: package version, gate path, gate version, mode."""
    gate = resolve_gate()
    version = gate_version(gate) if os.path.exists(gate) else None
    mode = ""
    try:
        with open(os.path.join(operator_home(), "mode"), encoding="utf-8") as f:
            mode = f.read().strip()
    except OSError:
        pass
    return ("operator-lite %s  gate=%s  gate_version=%s  mode=%s"
            % (VERSION, gate, version or "MISSING", mode or "unset(enforce)"))
