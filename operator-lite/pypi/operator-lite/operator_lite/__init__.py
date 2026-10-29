"""operator-lite: one pip-installable front door for the Python Operator Lite ports.

The gate is NOT in this wheel. One stdlib-only file -- the external
``operator-gate.py`` -- owns every decision; this package only carries its
contract to three Python agent frameworks. At call time each adapter locates
(installs) the gate exactly like the repo ports do:

    1. $OPERATOR_GATE                        (explicit override, re-read per call)
    2. $OPERATOR_HOME/gate/operator-gate.py  (non-default operator home)
    3. ~/.operator/gate/operator-gate.py     (the installed copy)

``operator_lite.ensure_gate()`` can fetch the gate from upstream (same URL the
ports' install.sh uses) into ~/.operator/gate/, never overwriting a
same-or-newer copy, and refusing to run inside an agent session.

Gate contract (unchanged from the ports):
    spawned as:  python3 <gate> hook --source <host>
    stdin:       {"tool_name": ..., "tool_input": {"command": ...},
                  "cwd": ..., "session_id": ...?}
    exit 0 = allow, exit 2 = block (reason on stderr)
    enforce vs shadow is a file: ~/.operator/mode (enforce when missing)

Everything decision-bearing lives in the gate. This package is payload shaping
and plumbing only -- no rule logic, no analysis. Stdlib-only at import time;
framework imports (langchain / crewai / agents) happen lazily on first use.
"""
import os

from ._core import (ALLOW, BLOCK, PASS, UNREACHABLE, GATE_URL, decide,
                    ensure_gate, extract_command, gate_status, is_shellish,
                    resolve_gate)

__version__ = "6.0.1"

# Verdicts shared by every adapter (same vocabulary as the ports).
PASS = "pass"          # not shell-ish -> no gating, call the handler
ALLOW = "allow"        # gate allowed the call (receipt already written by the gate)
BLOCK = "block"        # gate stopped it, or unreachable + destructive-like
UNREACHABLE = "unreachable"

__all__ = [
    "__version__",
    "PASS", "ALLOW", "BLOCK", "UNREACHABLE",
    # eager (stdlib-only) core
    "resolve_gate", "ensure_gate", "gate_status", "is_shellish",
    "extract_command", "decide", "GATE_URL",
    # lazy: langchain adapter
    "OperatorLiteMiddleware", "operator_middleware",
    # lazy: crewai adapter
    "crewai_hooks", "crewai_install", "crewai_uninstall", "crewai_status",
    # lazy: openai-agents adapter
    "openai_agents", "gate_decision", "GateDecision", "input_guardrail",
    "gated_shell_tool", "wrap_tool", "deny_message",
]


def __getattr__(name):
    """PEP 562 lazy re-exports: importing operator_lite stays framework-free;
    each adapter module (and its framework) materialises on first attribute
    access, mirroring the ports' own lazy-import pattern."""
    if name in ("OperatorLiteMiddleware", "operator_middleware"):
        from .adapters import langchain as _langchain
        value = getattr(_langchain, name)
    elif name in ("crewai_hooks", "crewai_install", "crewai_uninstall",
                  "crewai_status"):
        from .adapters import crewai as _crewai
        value = {"crewai_hooks": _crewai,
                 "crewai_install": _crewai.install,
                 "crewai_uninstall": _crewai.uninstall,
                 "crewai_status": _crewai.status}[name]
    elif name in ("openai_agents", "gate_decision", "GateDecision",
                  "input_guardrail", "gated_shell_tool", "wrap_tool",
                  "deny_message"):
        from .adapters import openai_agents as _oa
        value = {"openai_agents": _oa,
                 "gate_decision": _oa.gate_decision,
                 "GateDecision": _oa.GateDecision,
                 "input_guardrail": _oa.input_guardrail,
                 "gated_shell_tool": _oa.gated_shell_tool,
                 "wrap_tool": _oa.wrap_tool,
                 "deny_message": _oa.deny_message}[name]
    else:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    globals()[name] = value          # cache: resolved only once per process
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
