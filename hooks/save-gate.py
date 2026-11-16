#!/usr/bin/env python3
"""
Vestige Save-Gate & 7-Field Strata Provenance Hook (hooks/operator-stop-memory.py)

Enforces:
1. UserPromptSubmit: records turn_seq, prompt_hash, git_head_sha in session-<sid>.json.
2. PostToolUse: unconditionally captures Edit/Write/MultiEdit, git commit, Bash outcomes,
   and remember(action="save") receipts into Reflex state + strata-events-<sid>.jsonl.
3. PreToolUse (Bash): denies `git commit` until modified files / resolved failures are anchored.
4. Stop: blocks turn completion with `{"decision": "block", "reason": ...}` up to
   MAX_STOP_BLOCKS_PER_TURN = 2 when stop_hook_active is true.
5. Never crashes a hook: bad or non-object JSON input exits 0 with no output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MAX_STOP_BLOCKS_PER_TURN = 2
# Literal directive markers. Same set Reflex uses: contains(), token boundary,
# no regex. Editable by replacing this tuple; it is not a domain word list.
DIRECTIVE_PHRASES = (
    "remember this",
    "remember that",
    "never",
    "always",
    "rule:",
    "from now on",
    "going forward",
    "don't ever",
    "decision:",
)
GIT_COMMIT_RE = re.compile(
    r"(?:^|[;&|]\s*|&&\s*)git"
    r"(?:\s+-C\s+\S+|\s+--git-dir(?:=|\s+)\S+|\s+--work-tree(?:=|\s+)\S+)*"
    r"\s+commit\b"
)


def state_dir() -> Path:
    custom = os.environ.get("VESTIGE_REFLEX_DIR")
    if custom:
        return Path(custom)
    return Path.home() / ".local" / "state" / "vestige" / "reflex"


def session_state_path(session_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id or "default")
    return state_dir() / f"session-{safe_id}.json"


def receipt_log_path(session_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id or "default")
    return state_dir() / f"strata-events-{safe_id}.jsonl"


def digest_hex(domain: bytes, data: bytes) -> str:
    h = hashlib.blake2b(digest_size=32)
    h.update(domain)
    h.update(data)
    return h.hexdigest()


def current_git_head_sha(cwd: str | None) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "0000000000000000000000000000000000000000"


def load_state(session_id: str) -> dict[str, Any]:
    path = session_state_path(session_id)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return {
        "session_id": session_id,
        "turn_seq": 0,
        "prompt_hash": "0" * 64,
        "git_head_sha": "",
        "last_frame_digest": "0" * 64,
        "edited_files_turn": {},
        "saved_files_turn": {},
        "edited_hunks": {},
        "open_failures": [],
        "resolved_pairs": [],
        "anchor_events": [],
        "stop_blocks_in_turn": 0,
        "pending_directives": [],
    }


def save_state(session_id: str, state: dict[str, Any]) -> None:
    path = session_state_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_event_receipt(state: dict[str, Any], receipt: dict[str, Any]) -> str:
    payload_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt["payload_digest"] = digest_hex(b"strata.mesh.payload.v1\0", payload_bytes)
    frame_digest = digest_hex(
        b"strata.hook_event.v1\0",
        f"{receipt['session_id']}:{receipt['turn_seq']}:{receipt['payload_digest']}".encode("utf-8"),
    )
    receipt["frame_digest"] = frame_digest
    state["last_frame_digest"] = frame_digest

    log_path = receipt_log_path(state["session_id"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(receipt, sort_keys=True) + "\n")
    return frame_digest


def extract_edited_hunks(tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    file_path = tool_input.get("file_path") or tool_input.get("path")
    if not file_path:
        return []
    return [{"path": str(file_path), "start_line": None, "end_line": None}]


def normalize_error_sig(tool_response: Any, exit_code: int) -> str:
    text = ""
    if isinstance(tool_response, dict):
        text = str(tool_response.get("stderr") or tool_response.get("stdout") or "")
    elif isinstance(tool_response, str):
        text = tool_response
    first_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped[:160]
            break
    if first_line:
        return f"exit:{exit_code}:{first_line}"
    return f"exit:{exit_code}"


def extract_exit_code(tool_response: Any) -> int:
    if isinstance(tool_response, dict):
        if "exit_code" in tool_response and isinstance(tool_response["exit_code"], int):
            return tool_response["exit_code"]
        if "exitCode" in tool_response and isinstance(tool_response["exitCode"], int):
            return tool_response["exitCode"]
        if tool_response.get("interrupted") is True:
            return 130
        if tool_response.get("isError") is True:
            return 1
    return 0


def unsaved_files(state: dict[str, Any]) -> list[str]:
    edited = state.get("edited_files_turn", {})
    saved = state.get("saved_files_turn", {})
    return sorted(
        path for path, e_turn in edited.items() if int(e_turn) > int(saved.get(path, 0))
    )


def unanchored_resolved_errors(state: dict[str, Any]) -> list[str]:
    sigs = {
        pair["failure"]["error_sig"]
        for pair in state.get("resolved_pairs", [])
        if not pair.get("anchored_by_remember", False)
    }
    return sorted(sigs)


def format_save_gate_reason(missing_files: list[str], resolved_errors: list[str]) -> str:
    files_str = ", ".join(missing_files) if missing_files else "none"
    err_str = ", ".join(resolved_errors) if resolved_errors else "none"
    return (
        f"Vestige ledger: files [{files_str}] and resolved errors [{err_str}] "
        "have no remember(action='save') receipt this turn."
    )


def contains_phrase(text: str, phrase: str) -> bool:
    """Token-boundary contains, case-insensitive, no regex."""
    if not phrase:
        return False
    hay = text.lower()
    needle = phrase.lower()
    start = 0
    while True:
        at = hay.find(needle, start)
        if at < 0:
            return False
        before = at == 0 or not (hay[at - 1].isalnum() or hay[at - 1] == "_")
        after_at = at + len(needle)
        after = after_at >= len(hay) or not (hay[after_at].isalnum() or hay[after_at] == "_")
        if before and after:
            return True
        start = at + 1


def matched_directives(prompt: str) -> list[str]:
    return [phrase for phrase in DIRECTIVE_PHRASES if contains_phrase(prompt, phrase)]


def backtick_spans(text: str) -> list[str]:
    out: list[str] = []
    rest = text
    while True:
        i = rest.find("`")
        if i < 0:
            break
        rest = rest[i + 1 :]
        j = rest.find("`")
        if j < 0:
            break
        inner = rest[:j]
        if inner and "\n" not in inner and len(inner) <= 300:
            out.append(inner)
        rest = rest[j + 1 :]
    return out


def cited_shas(text: str) -> list[str]:
    found: list[str] = []
    lower = text.lower()
    marker = "commit:"
    start = 0
    while True:
        at = lower.find(marker, start)
        if at < 0:
            break
        run = []
        for ch in lower[at + len(marker) :]:
            if ch in "0123456789abcdef":
                run.append(ch)
            else:
                break
        sha = "".join(run)
        if 7 <= len(sha) <= 40:
            found.append(sha)
        start = at + len(marker)
    for inner in backtick_spans(text):
        token = inner.lower()
        if 7 <= len(token) <= 40 and all(c in "0123456789abcdef" for c in token):
            found.append(token)
    uniq: list[str] = []
    for sha in found:
        if sha not in uniq:
            uniq.append(sha)
    return uniq


def cited_files(text: str) -> list[str]:
    out: list[str] = []
    for inner in backtick_spans(text):
        if "/" in inner and " " not in inner and "://" not in inner and not inner.startswith("$"):
            if inner not in out:
                out.append(inner)
    return out


def sha_exists(cwd: str | None, sha: str) -> bool:
    if not cwd:
        return False
    try:
        res = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except Exception:
        return False
    return res.returncode == 0


def file_exists(cwd: str | None, path: str) -> bool:
    if not cwd or "\x00" in path:
        return False
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    return candidate.is_file()


def citation_reasons(payload: dict[str, Any]) -> list[str]:
    text = str(payload.get("last_assistant_message") or "")
    if not text:
        return []
    cwd = payload.get("cwd")
    cwd_str = str(cwd) if isinstance(cwd, str) else None
    reasons: list[str] = []
    for sha in cited_shas(text):
        if not sha_exists(cwd_str, sha):
            reasons.append(f"Turn blocked: commit {sha} is not in this repository.")
    for path in cited_files(text):
        if not file_exists(cwd_str, path):
            reasons.append(f"Turn blocked: file {path} is not on disk.")
    return reasons


def dirty_worktree(path: str) -> bool:
    if not path or not Path(path).is_dir():
        return False
    try:
        res = subprocess.run(
            ["git", "-C", path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except Exception:
        return False
    if res.returncode == 0:
        return bool(res.stdout.strip())
    return any(Path(path).rglob("*"))


def handle_user_prompt_submit(payload: dict[str, Any], state: dict[str, Any]) -> int:
    prompt = str(payload.get("prompt") or "")
    cwd = payload.get("cwd")
    cwd_str = str(cwd) if isinstance(cwd, str) else None
    state["turn_seq"] = int(state.get("turn_seq", 0)) + 1
    state["prompt_hash"] = digest_hex(b"strata.prompt.v1\0", prompt.encode("utf-8"))
    state["git_head_sha"] = current_git_head_sha(cwd_str)
    state["stop_blocks_in_turn"] = 0
    state["pending_directives"] = matched_directives(prompt)
    save_state(state["session_id"], state)
    phrases = state["pending_directives"]
    if phrases:
        # Factual, and kept under the host's 10_000-character additionalContext cap.
        lines = [
            (
                f'Vestige ledger: an owner directive ("{phrase}") is unsaved this turn; '
                "Stop will block until remember(...) records it."
            )
            for phrase in phrases
        ]
        text = "\n".join(lines)[:10000]
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": text,
                    }
                }
            )
        )
    return 0


def handle_post_tool_use(payload: dict[str, Any], state: dict[str, Any]) -> int:
    tool_name = str(payload.get("tool_name") or "")
    raw_input = payload.get("tool_input")
    tool_input: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}
    tool_response = payload.get("tool_response")
    cwd = payload.get("cwd")
    cwd_str = str(cwd) if isinstance(cwd, str) else None
    git_sha = current_git_head_sha(cwd_str)
    state["git_head_sha"] = git_sha
    now_ms = int(time.time() * 1000)

    normalized_input = json.dumps(tool_input, sort_keys=True)
    tool_call_digest = digest_hex(
        b"strata.tool_call.v1\0",
        f"{tool_name}\0{normalized_input}".encode("utf-8"),
    )

    if tool_name in ("Edit", "Write", "MultiEdit"):
        hunks = extract_edited_hunks(tool_input)
        for hunk in hunks:
            state["edited_files_turn"][hunk["path"]] = state["turn_seq"]
            state["edited_hunks"][hunk["path"]] = hunk

        frame_digest = append_event_receipt(
            state,
            {
                "kind": "edit_anchor",
                "session_id": state["session_id"],
                "turn_seq": state["turn_seq"],
                "prompt_hash": state["prompt_hash"],
                "git_head_sha": git_sha,
                "files": hunks,
                "tool_call_digest": tool_call_digest,
                "exit_code": 0,
                "error_sig": None,
                "parent_frame_digest": state["last_frame_digest"],
                "causal_edge": "derived_from",
                "policy_verdict": {"verdict": "allowed", "rule_id": "auto_capture_edit"},
                "ts_ms": now_ms,
            },
        )
        state["anchor_events"].append(
            {
                "frame_digest": frame_digest,
                "kind": "edit",
                "git_head_sha": git_sha,
                "files": hunks,
                "turn_seq": state["turn_seq"],
                "ts_ms": now_ms,
            }
        )
        save_state(state["session_id"], state)
        return 0

    if tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        exit_code = extract_exit_code(tool_response)
        is_commit = bool(GIT_COMMIT_RE.search(command))

        if is_commit and exit_code == 0:
            files = list(state.get("edited_hunks", {}).values())
            frame_digest = append_event_receipt(
                state,
                {
                    "kind": "commit_anchor",
                    "session_id": state["session_id"],
                    "turn_seq": state["turn_seq"],
                    "prompt_hash": state["prompt_hash"],
                    "git_head_sha": git_sha,
                    "files": files,
                    "tool_call_digest": tool_call_digest,
                    "exit_code": 0,
                    "error_sig": None,
                    "parent_frame_digest": state["last_frame_digest"],
                    "causal_edge": "derived_from",
                    "policy_verdict": {"verdict": "allowed", "rule_id": "auto_capture_commit"},
                    "ts_ms": now_ms,
                },
            )
            state["anchor_events"].append(
                {
                    "frame_digest": frame_digest,
                    "kind": "git_commit",
                    "git_head_sha": git_sha,
                    "files": files,
                    "turn_seq": state["turn_seq"],
                    "ts_ms": now_ms,
                }
            )

        if exit_code != 0:
            err_sig = normalize_error_sig(tool_response, exit_code)
            files = list(state.get("edited_hunks", {}).values())
            frame_digest = append_event_receipt(
                state,
                {
                    "kind": "open_failure",
                    "session_id": state["session_id"],
                    "turn_seq": state["turn_seq"],
                    "prompt_hash": state["prompt_hash"],
                    "git_head_sha": git_sha,
                    "files": files,
                    "tool_call_digest": tool_call_digest,
                    "exit_code": exit_code,
                    "error_sig": err_sig,
                    "parent_frame_digest": state["last_frame_digest"],
                    "causal_edge": "evidence_of",
                    "policy_verdict": {"verdict": "allowed", "rule_id": "bash_failure_capture"},
                    "ts_ms": now_ms,
                },
            )
            state["open_failures"].append(
                {
                    "failure_frame_digest": frame_digest,
                    "command": command,
                    "error_sig": err_sig,
                    "exit_code": exit_code,
                    "tool_call_digest": tool_call_digest,
                    "git_head_sha": git_sha,
                    "files": files,
                    "turn_seq": state["turn_seq"],
                    "ts_ms": now_ms,
                }
            )
        elif state.get("open_failures"):
            frame_digest = append_event_receipt(
                state,
                {
                    "kind": "fix_resolved",
                    "session_id": state["session_id"],
                    "turn_seq": state["turn_seq"],
                    "prompt_hash": state["prompt_hash"],
                    "git_head_sha": git_sha,
                    "files": list(state.get("edited_hunks", {}).values()),
                    "tool_call_digest": tool_call_digest,
                    "exit_code": 0,
                    "error_sig": None,
                    "parent_frame_digest": state["open_failures"][-1]["failure_frame_digest"],
                    "causal_edge": "closes",
                    "policy_verdict": {"verdict": "allowed", "rule_id": "bash_fix_capture"},
                    "ts_ms": now_ms,
                },
            )
            for failure in state["open_failures"]:
                state["resolved_pairs"].append(
                    {
                        "failure": failure,
                        "fix_command": command,
                        "fix_frame_digest": frame_digest,
                        "resolved_turn_seq": state["turn_seq"],
                        "resolved_ts_ms": now_ms,
                        "anchored_by_remember": False,
                    }
                )
            state["open_failures"] = []

        save_state(state["session_id"], state)
        return 0

    if tool_name in ("remember", "mcp__vestige__remember"):
        action = str(tool_input.get("action") or "save")
        if action == "save":
            raw_files = tool_input.get("files")
            covered = set(raw_files) if isinstance(raw_files, list) else set()
            if not covered:
                covered = set(state.get("edited_files_turn", {}).keys())
            for path in covered:
                state["saved_files_turn"][str(path)] = state["turn_seq"]
            for pair in state.get("resolved_pairs", []):
                pair["anchored_by_remember"] = True
            state["stop_blocks_in_turn"] = 0
            state["pending_directives"] = []
            append_event_receipt(
                state,
                {
                    "kind": "remember_save",
                    "session_id": state["session_id"],
                    "turn_seq": state["turn_seq"],
                    "prompt_hash": state["prompt_hash"],
                    "git_head_sha": git_sha,
                    "files": [
                        {"path": str(p), "start_line": None, "end_line": None}
                        for p in sorted(covered)
                    ],
                    "tool_call_digest": tool_call_digest,
                    "exit_code": 0,
                    "error_sig": None,
                    "parent_frame_digest": state["last_frame_digest"],
                    "causal_edge": "closes" if state.get("resolved_pairs") else "derived_from",
                    "policy_verdict": {"verdict": "allowed", "rule_id": "remember_anchor"},
                    "ts_ms": now_ms,
                },
            )
            save_state(state["session_id"], state)
        return 0

    return 0


def handle_pre_tool_use(payload: dict[str, Any], state: dict[str, Any]) -> int:
    if payload.get("tool_name") != "Bash":
        return 0
    raw_input = payload.get("tool_input")
    tool_input: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}
    command = str(tool_input.get("command") or "")
    if not GIT_COMMIT_RE.search(command):
        return 0

    missing_files = unsaved_files(state)
    resolved_errors = unanchored_resolved_errors(state)
    if not missing_files and not resolved_errors:
        return 0

    reason = "Blocked git commit: " + format_save_gate_reason(missing_files, resolved_errors)
    append_event_receipt(
        state,
        {
            "kind": "operator_stop",
            "session_id": state["session_id"],
            "turn_seq": state["turn_seq"],
            "prompt_hash": state["prompt_hash"],
            "git_head_sha": state.get("git_head_sha", ""),
            "files": [{"path": p, "start_line": None, "end_line": None} for p in missing_files],
            "tool_call_digest": digest_hex(b"strata.tool_call.v1\0", command.encode("utf-8")),
            "exit_code": None,
            "error_sig": resolved_errors[-1] if resolved_errors else None,
            "parent_frame_digest": state["last_frame_digest"],
            "causal_edge": "derived_from",
            "policy_verdict": {
                "verdict": "stopped",
                "rule_id": "save_gate.pre_commit",
                "reason": reason,
            },
            "ts_ms": int(time.time() * 1000),
        },
    )
    save_state(state["session_id"], state)

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    return 0


def stop_reasons(payload: dict[str, Any], state: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    missing_files = unsaved_files(state)
    resolved_errors = unanchored_resolved_errors(state)
    if missing_files or resolved_errors:
        reasons.append(format_save_gate_reason(missing_files, resolved_errors))
    for phrase in state.get("pending_directives", []):
        reasons.append(
            f'Turn blocked: owner directive "{phrase}" not saved. Call remember(...).'
        )
    reasons.extend(citation_reasons(payload))
    return reasons


def handle_stop(payload: dict[str, Any], state: dict[str, Any]) -> int:
    reasons = stop_reasons(payload, state)
    if not reasons:
        return 0

    if (
        payload.get("stop_hook_active") is True
        and int(state.get("stop_blocks_in_turn", 0)) >= MAX_STOP_BLOCKS_PER_TURN
    ):
        return 0

    state["stop_blocks_in_turn"] = int(state.get("stop_blocks_in_turn", 0)) + 1
    missing_files = unsaved_files(state)
    resolved_errors = unanchored_resolved_errors(state)
    reason = " ".join(reasons)[:10000]

    append_event_receipt(
        state,
        {
            "kind": "operator_stop",
            "session_id": state["session_id"],
            "turn_seq": state["turn_seq"],
            "prompt_hash": state["prompt_hash"],
            "git_head_sha": state.get("git_head_sha", ""),
            "files": [{"path": p, "start_line": None, "end_line": None} for p in missing_files],
            "tool_call_digest": "0" * 64,
            "exit_code": None,
            "error_sig": resolved_errors[-1] if resolved_errors else None,
            "parent_frame_digest": state["last_frame_digest"],
            "causal_edge": "derived_from",
            "policy_verdict": {
                "verdict": "stopped",
                "rule_id": "save_gate.stop",
                "reason": reason,
            },
            "ts_ms": int(time.time() * 1000),
        },
    )
    save_state(state["session_id"], state)

    print(json.dumps({"decision": "block", "reason": reason}))
    return 0


def handle_worktree_remove(payload: dict[str, Any]) -> int:
    """Command-hook path. Claude Code discards JSON on WorktreeRemove
    (verified 2026-11-16); a non-zero exit fails removal while the directory
    still exists. Whether an MCP tool error maps to that exit is unverified.
    """
    path = payload.get("worktree_path")
    if not isinstance(path, str) or not dirty_worktree(path):
        return 0
    print(
        f"Vestige ledger: worktree {path} has uncommitted changes.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--event",
            choices=[
                "UserPromptSubmit",
                "PreToolUse",
                "PostToolUse",
                "Stop",
                "WorktreeRemove",
            ],
            required=True,
        )
        args = parser.parse_args()

        raw = sys.stdin.read().strip()
        if not raw:
            return 0
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return 0

        session_id = str(
            payload.get("session_id") or os.environ.get("VESTIGE_SESSION_ID") or "default"
        )
        state = load_state(session_id)

        if args.event == "UserPromptSubmit":
            return handle_user_prompt_submit(payload, state)
        if args.event == "PostToolUse":
            return handle_post_tool_use(payload, state)
        if args.event == "PreToolUse":
            return handle_pre_tool_use(payload, state)
        if args.event == "Stop":
            return handle_stop(payload, state)
        if args.event == "WorktreeRemove":
            return handle_worktree_remove(payload)
        return 0
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(main())
