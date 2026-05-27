#!/usr/bin/env python3
"""Shared Sanhedrin claim-ledger, receipt, and appeal helpers.

This module is intentionally dependency-free so Stop hooks can import it inside
Claude Code, Codex wrappers, and local development without installing a package.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any


STATE_DIR = Path(
    os.environ.get("VESTIGE_SANHEDRIN_STATE_DIR")
    or Path.home() / ".vestige" / "sanhedrin"
)
RECEIPTS_DIR = STATE_DIR / "receipts"
LATEST_JSON = STATE_DIR / "latest.json"
LATEST_HTML = STATE_DIR / "latest.html"
APPEALS_JSONL = STATE_DIR / "appeals.jsonl"
COMMAND_RECEIPTS_JSONL = STATE_DIR / "command-receipts.jsonl"
FAIL_OPEN_JSONL = STATE_DIR / "fail-open.jsonl"
SUPPORTED_RECEIPT_SCHEMA = "vestige.sanhedrin.receipt.v1"

VERIFICATION_RE = re.compile(
    r"\b("
    r"(all\s+)?(tests?|test suite|build|lint|typecheck|checks?|cargo test|npm test|pnpm test|"
    r"pytest|vitest|jest|playwright|tsc|clippy)\s+"
    r"(passed|passes|passing|green|succeeded|succeeds|clean)|"
    r"(verified|validated|confirmed)\s+(with|by|via)\s+[`']?([^`'\n]+)[`']?"
    r")\b",
    re.IGNORECASE,
)
VERIFICATION_HEDGE_LEFT_RE = re.compile(
    r"\b("
    r"i\s+(think|believe|guess|suspect)|"
    r"maybe|might|may\s+have|possibly|probably|apparently|"
    r"let\s+me\s+verify|need\s+to\s+verify|will\s+verify|should\s+verify|"
    r"not\s+sure|unverified|without\s+running"
    r")\b[^.;:!?]{0,80}$",
    re.IGNORECASE,
)
COMMAND_FAMILY_PATTERNS = {
    "test": re.compile(r"\b(pytest|cargo\s+test|npm\s+(run\s+)?test|pnpm\s+(run\s+)?test|vitest|jest|playwright\s+test)\b", re.I),
    "build": re.compile(r"\b(cargo\s+build|npm\s+(run\s+)?build|pnpm\s+(run\s+)?build|next\s+build|vite\s+build)\b", re.I),
    "lint": re.compile(r"\b(cargo\s+clippy|npm\s+(run\s+)?lint|pnpm\s+(run\s+)?lint|eslint|ruff|flake8)\b", re.I),
    "typecheck": re.compile(r"\b(tsc|svelte-check|mypy|pyright|cargo\s+check)\b", re.I),
}
COMMAND_EXIT_RE = re.compile(
    r"(exit[_ -]?code|status|returncode|rc)[\"':=\s]+(-?\d+)|"
    r"Error:\s*Exit code\s*(-?\d+)|"
    r"Process exited with code\s*(-?\d+)",
    re.I,
)
CLAUDE_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)


def stable_id(text: str, prefix: str = "sr") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def claim_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    normalized = re.sub(r"[`'\"$]", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def strip_non_assertive_regions(text: str) -> str:
    """Remove quoted/code regions before looking for Receipt Lock assertions."""
    text = text[:16_384]
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", " ", text)
    kept_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(">"):
            continue
        kept_lines.append(line)
    text = "\n".join(kept_lines)
    text = re.sub(r'(^|[\s([{])"[^"\n]+"(?=([\s.,;:!?)}\]]|$))', r"\1 ", text)
    return text


def is_asserted_verification_claim(text: str) -> bool:
    match = VERIFICATION_RE.search(text)
    if not match:
        return False
    left_context = text[max(0, match.start() - 100) : match.start()]
    return VERIFICATION_HEDGE_LEFT_RE.search(left_context) is None


def split_claims(draft: str) -> list[str]:
    cleaned = strip_non_assertive_regions(draft)
    chunks = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    claims: list[str] = []
    for chunk in chunks:
        text = chunk.strip(" -\t")
        if len(text) >= 18 or is_asserted_verification_claim(text) or is_hard_user_claim(text):
            claims.append(text)
    return claims[:24]


def detect_claim_type(text: str) -> str:
    low = text.lower()
    if is_asserted_verification_claim(text):
        return "receipt_lock"
    if is_hard_user_claim(text):
        return "hard_user_claim"
    if any(word in low for word in ("won", "prize", "ranked", "placed", "score", "graduated", "worked at")):
        return "hard_user_claim"
    if any(word in low for word in ("should", "could", "recommend", "plan", "target", "estimate")):
        return "advice"
    if "`" in text or "/" in text or re.search(r"\bv?\d+\.\d+", text):
        return "technical"
    return "general"


def is_hard_user_claim(text: str) -> bool:
    if not re.search(r"\b(Sam|you|your|I|my)\b", text, re.I):
        return False
    hard_patterns = (
        r"\b(attended|graduated|studied|enrolled|accepted|worked|works|employed|hired)\b",
        r"\b(was\s+born|born\s+in|born\s+on|birthdate|birthday)\b",
        r"\b(won|placed|ranked|scored|earned|raised|sold|founded|launched)\b",
        r"\b(prize|award|payout|grant|scholarship|degree|gpa|employer|school|university|college|birth\s+date)\b",
        r"\$[0-9]",
    )
    return any(re.search(pattern, text, re.I) for pattern in hard_patterns)


def new_manifest(draft: str) -> dict[str, Any]:
    draft_id = stable_id(draft, "draft")
    claims = []
    for i, text in enumerate(split_claims(draft), start=1):
        claim_type = detect_claim_type(text)
        claims.append(
            {
                "id": f"c{i:03d}",
                "text": text,
                "fingerprint": claim_fingerprint(text),
                "class": claim_type,
                "subject": infer_subject(text),
                "risk": "hard" if claim_type == "receipt_lock" else "normal",
                "evidence_state": "unchecked",
                "decision": "pending",
                "precedent": [],
                "fix": "",
                "appeal": {
                    "status": "open",
                    "actions": ["stale", "wrong", "too_strict"],
                },
            }
        )
    return {
        "schema": SUPPORTED_RECEIPT_SCHEMA,
        "id": stable_id(f"{draft_id}:{now_iso()}", "receipt"),
        "draftId": draft_id,
        "createdAt": now_iso(),
        "overall": "pass",
        "verdictBar": "PASS",
        "summary": "No blocking claim issues found.",
        "draftPreview": draft[:1000],
        "claims": claims,
        "receipts": [],
        "source": {
            "stateDir": str(STATE_DIR),
            "transcript": os.environ.get("VESTIGE_SANHEDRIN_TRANSCRIPT"),
        },
    }


def infer_subject(text: str) -> str:
    if re.search(r"\b(Sam|you|your)\b", text, re.I):
        return "Sam"
    if re.search(r"\b(test|pytest|cargo test|npm test|pnpm test|vitest|jest)\b", text, re.I):
        return "test receipt"
    if re.search(r"\b(build|lint|typecheck|clippy|tsc)\b", text, re.I):
        return "command receipt"
    return "draft"


def command_families_for_claim(text: str) -> list[str]:
    low = text.lower()
    if re.search(r"\b(all\s+checks?|checks)\s+(passed|passes|passing|green|succeeded|succeeds|clean)\b", low) and "cargo check" not in low:
        return ["test", "build", "lint", "typecheck"]
    families: list[str] = []
    if any(word in low for word in ("test", "pytest", "vitest", "jest", "playwright")):
        families.append("test")
    if any(word in low for word in ("build", "compiled", "compile")):
        families.append("build")
    if any(word in low for word in ("lint", "clippy", "eslint", "ruff")):
        families.append("lint")
    if any(word in low for word in ("typecheck", "tsc", "mypy", "pyright", "cargo check")):
        families.append("typecheck")
    if "check" in low and "cargo check" not in low and "checks" not in low:
        families.append("typecheck")
    return families or ["test"]


def load_command_receipts() -> list[dict[str, Any]]:
    transcript = os.environ.get("VESTIGE_SANHEDRIN_TRANSCRIPT")
    if transcript:
        return extract_transcript_receipts(Path(transcript))
    if os.environ.get("VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER") != "1":
        return []
    return load_jsonl(COMMAND_RECEIPTS_JSONL)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
    except (OSError, json.JSONDecodeError):
        return items
    return items


def extract_transcript_receipts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    receipts: list[dict[str, Any]] = []
    pending_commands: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return receipts
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        receipts.extend(extract_structured_receipts(obj, pending_commands))
        if os.environ.get("VESTIGE_SANHEDRIN_ALLOW_LOOSE_LEDGER") != "1":
            continue
        blob = json.dumps(obj, ensure_ascii=False)
        command = extract_command(blob)
        if not command:
            continue
        exit_code = extract_exit_code(blob)
        receipts.append(
            {
                "source": "transcript",
                "command": command,
                "exitCode": exit_code,
                "success": exit_code == 0 if exit_code is not None else None,
                "timestamp": obj.get("timestamp") or obj.get("created_at") or now_iso(),
            }
        )
    return receipts


def extract_structured_receipts(
    obj: dict[str, Any],
    pending_commands: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract Claude Code Bash receipts from assistant tool_use/user tool_result pairs."""
    receipts: list[dict[str, Any]] = []
    timestamp = obj.get("timestamp") or obj.get("created_at") or now_iso()
    receipts.extend(extract_codex_receipts(obj, pending_commands, timestamp))
    content = obj.get("message", {}).get("content", obj.get("content", ""))

    blocks = content if isinstance(content, list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and str(block.get("name", "")).lower() in {"bash", "shell", "exec_command"}:
            tool_id = str(block.get("id") or "")
            tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
            command = str(tool_input.get("command") or tool_input.get("cmd") or "")
            if tool_id and command:
                pending_commands[tool_id] = {
                    "source": "transcript",
                    "toolUseId": tool_id,
                    "command": command,
                    "timestamp": timestamp,
                }
        if block.get("type") == "tool_result":
            tool_id = str(block.get("tool_use_id") or "")
            if not tool_id or tool_id not in pending_commands:
                continue
            receipt = dict(pending_commands[tool_id])
            text = stringify_tool_result(block)
            explicit_exit = extract_exit_code(text)
            is_error = bool(block.get("is_error"))
            receipt["exitCode"] = explicit_exit if explicit_exit is not None else (1 if is_error else 0)
            receipt["success"] = receipt["exitCode"] == 0 and not is_error
            receipt["timestamp"] = timestamp
            receipts.append(receipt)

    tool_result = obj.get("toolUseResult")
    if isinstance(tool_result, dict):
        command = str(obj.get("command") or tool_result.get("command") or "")
        if command:
            exit_code = tool_result.get("exitCode")
            if exit_code is None:
                exit_code = tool_result.get("exit_code")
            try:
                parsed_exit = int(exit_code) if exit_code is not None else None
            except (TypeError, ValueError):
                parsed_exit = extract_exit_code(json.dumps(tool_result, ensure_ascii=False))
            is_error = bool(tool_result.get("is_error") or tool_result.get("interrupted"))
            receipts.append(
                {
                    "source": "transcript",
                    "command": command,
                    "exitCode": parsed_exit if parsed_exit is not None else (1 if is_error else 0),
                    "success": (parsed_exit == 0 if parsed_exit is not None else not is_error),
                    "timestamp": timestamp,
                }
            )

    return receipts


def extract_codex_receipts(
    obj: dict[str, Any],
    pending_commands: dict[str, dict[str, Any]],
    timestamp: str,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return receipts

    payload_type = payload.get("type")
    name = str(payload.get("name") or "").lower()
    call_id = str(payload.get("call_id") or "")
    if payload_type == "function_call" and name in {"exec_command", "bash", "shell"} and call_id:
        args = payload.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(args, dict):
            command = str(args.get("cmd") or args.get("command") or "")
            if command:
                pending_commands[call_id] = {
                    "source": "codex-transcript",
                    "toolUseId": call_id,
                    "command": command,
                    "timestamp": timestamp,
                }
    elif payload_type == "function_call" and name == "write_stdin" and call_id:
        args = payload.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(args, dict):
            session_id = args.get("session_id")
            session_receipt = pending_commands.get(f"session:{session_id}")
            if session_receipt:
                pending_commands[call_id] = dict(session_receipt)

    if payload_type == "function_call_output" and call_id in pending_commands:
        receipt = dict(pending_commands[call_id])
        output = str(payload.get("output") or "")
        running = re.search(r"Process running with session ID\s+(\d+)", output)
        if running:
            pending_commands[f"session:{running.group(1)}"] = receipt
            return receipts
        exit_code = extract_exit_code(output)
        receipt["exitCode"] = exit_code
        receipt["success"] = exit_code == 0 if exit_code is not None else None
        receipt["timestamp"] = timestamp
        receipts.append(receipt)

    return receipts


def stringify_tool_result(block: dict[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def extract_command(blob: str) -> str | None:
    for key in ("cmd", "command"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', blob)
        if match:
            return bytes(match.group(1), "utf-8").decode("unicode_escape")
    match = CLAUDE_TOOL_NAME_RE.search(blob)
    if match and match.group(1).lower() in {"bash", "shell", "exec_command"}:
        return match.group(1)
    return None


def extract_exit_code(blob: str) -> int | None:
    match = COMMAND_EXIT_RE.search(blob)
    if not match:
        return None
    try:
        return int(match.group(2) or match.group(3) or match.group(4))
    except ValueError:
        return None


def receipt_matches_family(receipt: dict[str, Any], family: str) -> bool:
    command = str(receipt.get("command") or "")
    pattern = COMMAND_FAMILY_PATTERNS.get(family)
    return bool(pattern and pattern.search(command))


def apply_receipt_lock(manifest: dict[str, Any]) -> str | None:
    receipts = load_command_receipts()
    manifest["receipts"] = receipts[-20:]
    appeals = load_appeals()

    for claim in manifest["claims"]:
        if claim["class"] != "receipt_lock":
            continue
        missing_families: list[str] = []
        failed_family: tuple[str, dict[str, Any]] | None = None
        supported_families: list[tuple[str, dict[str, Any]]] = []

        for family in command_families_for_claim(claim["text"]):
            matching = [r for r in receipts if receipt_matches_family(r, family)]
            latest = matching[-1] if matching else None
            if latest is None:
                missing_families.append(family)
            elif latest.get("success") is not True:
                failed_family = (family, latest)
                break
            else:
                supported_families.append((family, latest))

        if failed_family is not None:
            family, latest = failed_family
            claim["evidence_state"] = "failed_receipt" if latest.get("success") is False else "unknown_receipt"
            claim["decision"] = "veto"
            claim["precedent"].append(
                {
                    "type": "command",
                    "summary": f"Latest {family} command did not produce a successful receipt.",
                    "command": latest.get("command"),
                    "exitCode": latest.get("exitCode"),
                }
            )
            claim["fix"] = f"Replace the claim with: I do not have a successful {family} receipt for this session."
            manifest["overall"] = "veto"
            manifest["verdictBar"] = "VETO"
            manifest["summary"] = "Receipt Lock blocked a contradicted verification claim."
            return f"Receipt Lock: Draft claims {family} passed, but latest {family} receipt is not successful."

        if missing_families and is_appealed(claim, appeals):
            claim["evidence_state"] = "appealed"
            claim["decision"] = "appealed"
            claim["precedent"].append({"type": "appeal", "summary": "Prior appeal suppresses this missing-receipt veto."})
            manifest["overall"] = "pass_with_warnings"
            manifest["verdictBar"] = "APPEALED"
            manifest["summary"] = "Prior appeal suppressed a Receipt Lock veto."
            continue

        if missing_families:
            family_list = ", ".join(missing_families)
            claim["evidence_state"] = "missing_receipt"
            claim["decision"] = "veto"
            claim["precedent"].append(
                {
                    "type": "receipt_lock",
                    "summary": f"No {family_list} command receipt found in this session.",
                    "source": "transcript/command ledger",
                }
            )
            claim["fix"] = f"Replace the claim with: I do not have recorded {family_list} receipt(s) for this session."
            manifest["overall"] = "veto"
            manifest["verdictBar"] = "VETO"
            manifest["summary"] = "Receipt Lock blocked an unsupported verification claim."
            return f"Receipt Lock: Draft claims {family_list} passed, but no {family_list} command receipt exists."

        claim["evidence_state"] = "supported"
        claim["decision"] = "pass"
        for family, latest in supported_families:
            claim["precedent"].append(
                {
                    "type": "command",
                    "summary": f"{family} receipt found.",
                    "command": latest.get("command"),
                    "exitCode": latest.get("exitCode"),
                }
            )

    return None


def apply_model_verdict(manifest: dict[str, Any], verdict: str, evidence: str = "") -> str:
    low = verdict.strip().lower()
    if low == "yes" or low.startswith("yes "):
        if manifest["overall"] != "veto":
            has_appealed = any(c["decision"] == "appealed" for c in manifest["claims"])
            has_unchecked = any(c["decision"] == "pending" for c in manifest["claims"])
            manifest["overall"] = "pass_with_warnings" if has_unchecked or has_appealed else "pass"
            manifest["verdictBar"] = "APPEALED" if has_appealed else ("NOTE" if has_unchecked else "PASS")
            manifest["summary"] = (
                "Prior appeal suppressed a Sanhedrin veto."
                if has_appealed
                else "Sanhedrin found no blocking contradiction."
            )
        for claim in manifest["claims"]:
            if claim["decision"] == "pending":
                claim["decision"] = "pass_unverified"
                claim["evidence_state"] = "out_of_scope"
        return "yes"

    reason = verdict.split(" - ", 1)[1] if " - " in verdict else verdict
    appeals = load_appeals()
    candidate = first_relevant_claim(manifest)
    if candidate and is_appealed(candidate, appeals):
        candidate["decision"] = "appealed"
        candidate["evidence_state"] = "appealed"
        candidate["precedent"].append({"type": "appeal", "summary": "Prior appeal suppresses this model veto."})
        manifest["overall"] = "pass_with_warnings"
        manifest["verdictBar"] = "APPEALED"
        manifest["summary"] = "Prior appeal suppressed the Sanhedrin veto."
        return "yes"

    if candidate:
        candidate["decision"] = "veto"
        candidate["evidence_state"] = "contradicted"
        candidate["precedent"].append({"type": "vestige", "summary": reason[:500], "evidence": evidence[:1000]})
        candidate["fix"] = "Remove or qualify the contradicted claim using the cited Vestige precedent."
    manifest["overall"] = "veto"
    manifest["verdictBar"] = "VETO"
    manifest["summary"] = reason[:500]
    return verdict


def first_relevant_claim(manifest: dict[str, Any]) -> dict[str, Any] | None:
    for claim in manifest["claims"]:
        if claim["decision"] in {"pending", "pass_unverified"}:
            return claim
    return manifest["claims"][0] if manifest["claims"] else None


def load_appeals() -> list[dict[str, Any]]:
    return load_jsonl(APPEALS_JSONL)


def is_appealed(claim: dict[str, Any], appeals: list[dict[str, Any]]) -> bool:
    fp = claim.get("fingerprint")
    if not fp:
        return False
    for appeal in appeals:
        if (
            appeal.get("claimFingerprint") == fp
            and appeal.get("reason") in {"stale", "wrong", "too_strict"}
            and appeal.get("status", "active") == "active"
        ):
            return True
    return False


def save_manifest(manifest: dict[str, Any]) -> None:
    ensure_dirs()
    receipt_path = RECEIPTS_DIR / f"{manifest['id']}.json"
    html_path = RECEIPTS_DIR / f"{manifest['id']}.html"
    json_blob = json.dumps(manifest, indent=2)
    write_text_atomic(receipt_path, json_blob)
    write_text_atomic(LATEST_JSON, json_blob)
    rendered = render_receipt_html(manifest)
    write_text_atomic(html_path, rendered)
    write_text_atomic(LATEST_HTML, rendered)


def record_fail_open(reason: str, detail: str = "", transcript: str | None = None) -> None:
    ensure_dirs()
    run_id = os.environ.get("VESTIGE_SANHEDRIN_RUN_ID") or stable_id(f"{now_iso()}:{os.getpid()}", "run")
    event = {
        "timestamp": now_iso(),
        "runId": run_id,
        "reason": reason,
        "detail": detail[:500],
        "transcript": transcript or os.environ.get("VESTIGE_SANHEDRIN_TRANSCRIPT"),
    }
    try:
        with FAIL_OPEN_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass


def write_text_atomic(path: Path, content: str) -> None:
    ensure_dirs()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def render_receipt_html(manifest: dict[str, Any]) -> str:
    status = html.escape(str(manifest.get("verdictBar", "PASS")))
    summary = html.escape(str(manifest.get("summary", "")))
    claims = []
    for claim in manifest.get("claims", []):
        precedents = "".join(
            f"<li>{html.escape(str(p.get('summary', p)))}</li>"
            for p in claim.get("precedent", [])
        )
        claims.append(
            "<section class='claim'>"
            f"<div class='meta'>{html.escape(str(claim.get('decision')))} / {html.escape(str(claim.get('evidence_state')))}</div>"
            f"<h2>{html.escape(str(claim.get('text')))}</h2>"
            f"<p><strong>Fix:</strong> {html.escape(str(claim.get('fix') or 'No change required.'))}</p>"
            f"<p><strong>Appeal:</strong> stale | wrong | too_strict</p>"
            f"<ul>{precedents}</ul>"
            "</section>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Vestige Veto Receipt</title>
<style>
body{{margin:0;background:#050509;color:#e7e7f4;font-family:Inter,ui-sans-serif,system-ui;padding:32px}}
.bar{{display:inline-flex;gap:10px;align-items:center;border:1px solid #6d5dfc66;border-radius:8px;padding:10px 14px;background:#171528}}
.status{{font-weight:800;color:#fff;letter-spacing:.08em}}
.claim{{margin-top:18px;border:1px solid #ffffff1a;border-radius:8px;padding:16px;background:#0e0f18}}
.meta{{font-size:12px;color:#a8a8c8;text-transform:uppercase;letter-spacing:.08em}}
h1{{font-size:24px;margin:18px 0 4px}} h2{{font-size:16px;line-height:1.4}} p,li{{color:#c7c7dd}}
</style></head><body>
<div class="bar"><span>Verdict</span><span class="status">{status}</span></div>
<h1>Veto Receipt</h1><p>{summary}</p>{''.join(claims)}
</body></html>"""


def appeal_latest(reason: str, note: str = "", claim_id: str | None = None) -> dict[str, Any]:
    if not LATEST_JSON.exists():
        raise FileNotFoundError(str(LATEST_JSON))
    manifest = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
    claims = manifest.get("claims", [])
    claim = next((c for c in claims if c.get("id") == claim_id), None) if claim_id else None
    if claim is None:
        claim = next((c for c in claims if c.get("decision") == "veto"), claims[0] if claims else None)
    if claim is None:
        raise ValueError("latest receipt has no claims")
    appeal = {
        "timestamp": now_iso(),
        "receiptId": manifest.get("id"),
        "claimId": claim.get("id"),
        "claimFingerprint": claim.get("fingerprint"),
        "claim": claim.get("text"),
        "reason": reason,
        "note": note,
        "status": "active",
    }
    ensure_dirs()
    with APPEALS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(appeal) + "\n")
    claim["appeal"]["status"] = "appealed"
    claim["appeal"]["lastReason"] = reason
    manifest["overall"] = "appealed"
    manifest["verdictBar"] = "APPEALED"
    manifest["summary"] = f"Appealed as {reason}."
    save_manifest(manifest)
    return appeal
