"""
Vestige Universal Python SDK Hook Adapters (CrewAI, OpenAI Agents SDK, LangChain / LangGraph).

Delegates every pre-tool, post-tool, session, and prompt check through Vestige's
unified Node bridge (`vestige-bridge.cjs`), giving Python agent frameworks:
  - Operator STOP rules + precedent enforcement
  - Call Coach (CC-001..CC-013) argument rewriting
  - Zero-Prompt Recall + Anti-Spiral escalation
  - Secret-scan commit gate + Stop proof-carrying finish
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable


class VestigeBlockedError(RuntimeError):
    """Raised when Vestige Operator or Reflex denies a tool call."""


def _resolve_bridge() -> Path:
    env_p = os.environ.get("VESTIGE_BRIDGE_PATH")
    if env_p and Path(env_p).exists():
        return Path(env_p)
    installed = Path.home() / ".vestige" / "memory" / "hooks" / "vestige-bridge.cjs"
    if installed.exists():
        return installed
    repo_local = Path(__file__).resolve().parents[3] / "packages" / "vestige-init" / "hooks" / "vestige-bridge.cjs"
    return repo_local


def run_vestige_hook(event: str, payload: dict[str, Any], runtime: str = "python-sdks") -> dict[str, Any]:
    bridge = _resolve_bridge()
    if not bridge.exists():
        return {}
    proc = subprocess.run(
        [os.environ.get("NODE_BIN", "node"), str(bridge), "--runtime", runtime, "--event", event],
        input=json.dumps({**payload, "hook_event_name": event}),
        text=True,
        capture_output=True,
        timeout=12,
        check=False,
    )
    if proc.returncode != 0:
        reason = (proc.stderr or "").strip() or f"Vestige blocked call (exit {proc.returncode})"
        raise VestigeBlockedError(reason)
    out = (proc.stdout or "").strip()
    if not out:
        return {}
    try:
        parsed = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict) and (parsed.get("deny") or parsed.get("permission") == "deny" or parsed.get("permissionDecision") == "deny"):
        raise VestigeBlockedError(str(parsed.get("reason") or "Vestige denied tool call."))
    return parsed if isinstance(parsed, dict) else {}


def register_crewai_hooks(session_id: str = "crewai-default") -> None:
    """Register global Vestige pre/post tool hooks with CrewAI if installed."""
    from crewai.hooks import HookAborted, register_hook, PRE_TOOL_CALL, POST_TOOL_CALL  # type: ignore

    @register_hook(PRE_TOOL_CALL)
    def _vestige_pre(context: Any) -> None:
        try:
            ans = run_vestige_hook("PreToolUse", {
                "session_id": session_id,
                "tool_name": getattr(context, "tool_name", ""),
                "tool_input": getattr(context, "tool_input", {}) or {},
                "cwd": os.getcwd(),
            }, runtime="crewai")
            if isinstance(ans.get("updatedInput"), dict) and hasattr(context, "tool_input"):
                context.tool_input = ans["updatedInput"]
        except VestigeBlockedError as exc:
            raise HookAborted(str(exc), source="vestige") from exc

    @register_hook(POST_TOOL_CALL)
    def _vestige_post(context: Any) -> None:
        run_vestige_hook("PostToolUse", {
            "session_id": session_id,
            "tool_name": getattr(context, "tool_name", ""),
            "tool_input": getattr(context, "tool_input", {}) or {},
            "tool_response": {"exit_code": 0, "stdout": str(getattr(context, "tool_result", ""))},
            "cwd": os.getcwd(),
        }, runtime="crewai")


def langchain_wrap_tool_call(session_id: str = "langchain-default") -> Callable[..., Any]:
    """Return a LangChain v1 @wrap_tool_call-compatible middleware handler."""
    def middleware(request: Any, handler: Callable[[Any], Any]) -> Any:
        tool_call = getattr(request, "tool_call", None) or {}
        name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(request, "name", "")
        args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(request, "args", {})
        try:
            ans = run_vestige_hook("PreToolUse", {
                "session_id": session_id,
                "tool_name": name,
                "tool_input": args if isinstance(args, dict) else {},
                "cwd": os.getcwd(),
            }, runtime="langchain")
        except VestigeBlockedError as exc:
            return {"error": str(exc), "blocked_by": "vestige"}
        if isinstance(ans.get("updatedInput"), dict) and isinstance(tool_call, dict):
            tool_call["args"] = ans["updatedInput"]
        result = handler(request)
        run_vestige_hook("PostToolUse", {
            "session_id": session_id,
            "tool_name": name,
            "tool_input": args if isinstance(args, dict) else {},
            "tool_response": {"exit_code": 0, "stdout": str(result)[:2000]},
            "cwd": os.getcwd(),
        }, runtime="langchain")
        return result

    return middleware


def openai_agents_tool_guard(tool_name: str, tool_args: dict[str, Any], session_id: str = "openai-agents-default") -> dict[str, Any]:
    """Validate/rewrite a tool call inside an OpenAI Agents SDK tool_input guardrail."""
    ans = run_vestige_hook("PreToolUse", {
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_input": tool_args,
        "cwd": os.getcwd(),
    }, runtime="openai-agents")
    return ans.get("updatedInput") if isinstance(ans.get("updatedInput"), dict) else tool_args
