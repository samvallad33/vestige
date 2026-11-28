//! Unified native Rust hook dispatcher (`vestige hook dispatch <runtime> <event>`).
//!
//! Normalizes payloads across all 24 supported AI coding runtimes into one canonical
//! representation, routes `PreToolUse` directly through [`crate::native_gate`] (`<1ms`,
//! zero Python), automatically captures imperative mission contracts on `UserPromptSubmit`,
//! verifies unfinished mission targets on `Stop` (blocking premature termination without
//! relying on the agent to voluntarily register goals), and emits each host's exact wire
//! JSON schema.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use vestige_core::{GuardDecision, GuardEvalRequest, GuardEvalVerdict};

use crate::native_gate::{
    append_receipt_o1, append_receipt_with_decision, compute_action_digest, default_receipts_path,
    evaluate_gate_request, operator_home_dir, request_from_hook_stdin,
    try_consume_single_use_permit,
};

/// All 24 AI agent runtimes supported by the native Vestige hook dispatcher.
/// hardcode-scan: protocol (host runtime dialect identifiers for hook wire formatting).
pub const SUPPORTED_RUNTIMES: &[&str] = &[
    "claude",
    "claude-code",
    "antigravity",
    "cursor",
    "codex",
    "gemini",
    "gemini-cli",
    "copilot",
    "windsurf",
    "opencode",
    "cline",
    "roo-code",
    "kilo-code",
    "zed",
    "kiro",
    "goose",
    "crush",
    "amp",
    "devin",
    "factory-droid",
    "openhands",
    "amazon-q",
    "continue",
    "aider",
];

/// Canonical hook event kind normalized from any host dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CanonicalEvent {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
    PreCompact,
    Stop,
    SessionEnd,
}

impl CanonicalEvent {
    /// Normalize any host event string into a [`CanonicalEvent`].
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "sessionstart" | "session_start" | "session" | "agentspawn" | "taskstart" => {
                Self::SessionStart
            }
            "userpromptsubmit"
            | "userpromptsubmitted"
            | "beforesubmitprompt"
            | "beforeagent"
            | "pre_user_prompt"
            | "preinvocation"
            | "prompt" => Self::UserPromptSubmit,
            "pretooluse"
            | "beforetool"
            | "pre_run_command"
            | "pre_write_code"
            | "pre_read_code"
            | "pre_mcp_tool_use"
            | "tool.execute.before" => Self::PreToolUse,
            "posttooluse"
            | "aftertool"
            | "post_run_command"
            | "post_write_code"
            | "post_mcp_tool_use"
            | "postfilesave"
            | "postinvocation"
            | "tool" => Self::PostToolUse,
            "posttoolusefailure" | "erroroccurred" => Self::PostToolUseFailure,
            "precompact" | "precompress" | "compact" => Self::PreCompact,
            "stop"
            | "agentstop"
            | "afteragent"
            | "afteragentresponse"
            | "post_cascade_response"
            | "taskcomplete"
            | "turn" => Self::Stop,
            "sessionend" | "end" => Self::SessionEnd,
            _ => Self::PreToolUse,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::PostToolUseFailure => "PostToolUseFailure",
            Self::PreCompact => "PreCompact",
            Self::Stop => "Stop",
            Self::SessionEnd => "SessionEnd",
        }
    }
}

/// Output of a native hook dispatch, ready to print to stdout/stderr with `exit_code`.
#[derive(Debug, Clone)]
pub struct HookDispatchResult {
    pub stdout_json: Value,
    pub stderr_message: Option<String>,
    pub exit_code: i32,
}

/// Maximum consecutive Stop blocks per mission contract before releasing cleanly.
pub const MAX_MISSION_STOP_BLOCKS: u32 = 3;

/// Persisted imperative mission contract automatically captured on `UserPromptSubmit`
/// and enforced on `Stop` without requiring the agent to voluntarily register a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionContract {
    pub session_id: String,
    pub cwd: String,
    pub prompt_excerpt: String,
    pub required_deleted_paths: Vec<String>,
    #[serde(default)]
    pub required_net_loc_negative: bool,
    #[serde(default)]
    pub stop_blocks: u32,
    #[serde(default = "default_max_stop_blocks")]
    pub max_stop_blocks: u32,
    pub active: bool,
    pub created_at: String,
}

const fn default_max_stop_blocks() -> u32 {
    MAX_MISSION_STOP_BLOCKS
}

fn missions_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("VESTIGE_MISSIONS_DIR")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home).join(".vestige").join("missions");
    }
    if let Ok(profile) = std::env::var("USERPROFILE")
        && !profile.trim().is_empty()
    {
        return PathBuf::from(profile).join(".vestige").join("missions");
    }
    if let Some(base) = directories::UserDirs::new() {
        return base.home_dir().join(".vestige").join("missions");
    }
    PathBuf::from(".vestige").join("missions")
}

fn mission_file_path(dir: &Path, session_id: &str) -> PathBuf {
    let key = if session_id.trim().is_empty() {
        "default"
    } else {
        session_id.trim()
    };
    let hex: String = Sha256::digest(key.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    dir.join(format!("{}.json", &hex[..24]))
}

/// Extract relative paths explicitly targeted for deletion/removal in a user prompt
/// (e.g. `delete operator/, operator-lite/, hooks/save-gate.py`). Only records paths
/// that actually exist under `cwd` when the prompt arrives so non-path tokens never
/// cause false Stop blocks.
pub fn extract_deletion_targets(prompt: &str, cwd: &Path) -> Vec<String> {
    let lower = prompt.to_ascii_lowercase();
    let has_delete_verb = lower.contains("delete")
        || lower.contains("remove")
        || lower.contains("prune")
        || lower.contains("purge")
        || lower.contains("strip")
        || lower.contains("consolidat");
    if !has_delete_verb {
        return Vec::new();
    }

    let mut targets = Vec::new();
    for raw_tok in prompt.split_whitespace() {
        let cleaned = raw_tok
            .trim_matches(|c: char| {
                matches!(
                    c,
                    '`' | '\'' | '"' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '.' | '!' | '?'
                )
            })
            .trim_end_matches('/');
        if cleaned.is_empty()
            || cleaned.starts_with("http://")
            || cleaned.starts_with("https://")
            || cleaned.contains("..")
            || Path::new(cleaned).is_absolute()
        {
            continue;
        }
        // Track explicit repo paths or existing directories/files named after a delete/consolidate verb.
        let looks_like_path = cleaned.contains('/')
            || cleaned.contains('.')
            || raw_tok.trim_matches(|c: char| matches!(c, '`' | '\'' | '"' | ',' | '(' | ')')).ends_with('/')
            || (cleaned.len() >= 3 && cwd.join(cleaned).is_dir());
        if !looks_like_path {
            continue;
        }
        let candidate = cwd.join(cleaned);
        if candidate.exists() && !targets.iter().any(|t: &String| t == cleaned) {
            targets.push(cleaned.to_string());
        }
    }
    targets
}

fn prompt_requires_net_loc_shrink(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    (lower.contains("consolidat") || lower.contains("mega consolidation") || lower.contains("shrink"))
        && (lower.contains("delete")
            || lower.contains("remove")
            || lower.contains("strip")
            || lower.contains("loc")
            || lower.contains("lines"))
}

fn git_net_loc_delta(cwd: &Path) -> Option<(i64, i64)> {
    let out = std::process::Command::new("git")
        .args(["diff", "--numstat", "HEAD"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut added = 0i64;
    let mut deleted = 0i64;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let (Some(a), Some(d)) = (parts.next(), parts.next()) else {
            continue;
        };
        if let (Ok(ai), Ok(di)) = (a.parse::<i64>(), d.parse::<i64>()) {
            added += ai;
            deleted += di;
        }
    }
    Some((added, deleted))
}

/// Automatically capture an imperative mission contract from `UserPromptSubmit` into `dir`.
pub fn capture_mission_from_prompt(
    dir: &Path,
    session_id: &str,
    cwd: &str,
    prompt: &str,
) -> Option<MissionContract> {
    let cwd_path = Path::new(cwd);
    if cwd.trim().is_empty() || !cwd_path.is_dir() {
        return None;
    }
    let targets = extract_deletion_targets(prompt, cwd_path);
    let need_shrink = prompt_requires_net_loc_shrink(prompt);
    if targets.is_empty() && !need_shrink {
        return None;
    }
    let excerpt: String = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    let contract = MissionContract {
        session_id: session_id.to_string(),
        cwd: cwd.to_string(),
        prompt_excerpt: excerpt.chars().take(240).collect(),
        required_deleted_paths: targets,
        required_net_loc_negative: need_shrink,
        stop_blocks: 0,
        max_stop_blocks: MAX_MISSION_STOP_BLOCKS,
        active: true,
        created_at: Utc::now().to_rfc3339(),
    };
    let path = mission_file_path(dir, session_id);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string_pretty(&contract) {
        let _ = fs::write(&path, text);
    }
    Some(contract)
}

/// Verify an active mission contract on `Stop`. Enforces up to `max_stop_blocks` (default 3)
/// even when `stop_hook_active` is true so an agent cannot escape after 1 incomplete retry.
pub fn verify_mission_on_stop(
    dir: &Path,
    session_id: &str,
    _stop_hook_active: bool,
) -> Option<String> {
    let path = mission_file_path(dir, session_id);
    let raw = fs::read_to_string(&path).ok()?;
    let mut contract: MissionContract = serde_json::from_str(&raw).ok()?;
    if !contract.active {
        return None;
    }
    let max_blocks = if contract.max_stop_blocks == 0 {
        MAX_MISSION_STOP_BLOCKS
    } else {
        contract.max_stop_blocks
    };
    if contract.stop_blocks >= max_blocks {
        contract.active = false;
        if let Ok(text) = serde_json::to_string_pretty(&contract) {
            let _ = fs::write(&path, text);
        }
        return None;
    }
    let cwd = Path::new(&contract.cwd);
    let mut failures: Vec<String> = Vec::new();
    let remaining: Vec<String> = contract
        .required_deleted_paths
        .iter()
        .filter(|rel| cwd.join(rel).exists())
        .cloned()
        .collect();
    if !remaining.is_empty() {
        failures.push(format!(
            "{} target(s) requested for removal still exist on disk ({})",
            remaining.len(),
            remaining.join(", ")
        ));
    }
    if contract.required_net_loc_negative
        && let Some((added, deleted)) = git_net_loc_delta(cwd)
        && added > deleted
    {
        failures.push(format!(
            "consolidation contract requires negative net LOC, but git diff is +{added}/-{deleted} (net +{})",
            added - deleted
        ));
    }
    if failures.is_empty() {
        contract.active = false;
        if let Ok(text) = serde_json::to_string_pretty(&contract) {
            let _ = fs::write(&path, text);
        }
        return None;
    }
    contract.stop_blocks = contract.stop_blocks.saturating_add(1);
    if let Ok(text) = serde_json::to_string_pretty(&contract) {
        let _ = fs::write(&path, text);
    }
    Some(format!(
        "Mission contract unfinished (block {}/{}): {}. Complete the mission before stopping.",
        contract.stop_blocks,
        max_blocks,
        failures.join("; ")
    ))
}

/// Format a [`GuardEvalVerdict`] and optional context into the exact JSON schema
/// expected by `runtime` for `event`.
pub fn format_host_response(
    runtime: &str,
    event: CanonicalEvent,
    verdict: Option<&GuardEvalVerdict>,
    additional_context: Option<&str>,
) -> HookDispatchResult {
    let rt = runtime.trim().to_ascii_lowercase();
    let is_denied = verdict.is_some_and(|v| v.decision == GuardDecision::Deny);
    let reason = verdict.map(|v| v.reason.as_str()).unwrap_or("");
    let rule = verdict.and_then(|v| v.rule_id.as_deref()).unwrap_or("GATE");

    // Stop event block format across hosts.
    if event == CanonicalEvent::Stop && is_denied {
        if rt == "antigravity" {
            return HookDispatchResult {
                stdout_json: json!({
                    "decision": "deny",
                    "reason": format!("Vestige mission ({rule}): {reason}"),
                }),
                stderr_message: None,
                exit_code: 0,
            };
        }
        if matches!(rt.as_str(), "cline" | "windsurf" | "kiro" | "amazon-q" | "goose" | "crush") {
            return HookDispatchResult {
                stdout_json: json!({}),
                stderr_message: Some(format!("Vestige mission ({rule}): {reason}")),
                exit_code: 2,
            };
        }
        return HookDispatchResult {
            stdout_json: json!({
                "decision": "block",
                "reason": format!("Vestige mission ({rule}): {reason}"),
            }),
            stderr_message: None,
            exit_code: 0,
        };
    }

    // Cline uses POSIX exit code 2 + stderr on PreToolUse deny.
    if rt == "cline" && is_denied {
        return HookDispatchResult {
            stdout_json: json!({}),
            stderr_message: Some(format!("Vestige gate ({rule}): {reason}")),
            exit_code: 2,
        };
    }

    // Cursor / Copilot / Roo / Kilo / Continue / Zed / Amp dialect.
    if matches!(
        rt.as_str(),
        "cursor" | "copilot" | "roo-code" | "kilo-code" | "continue" | "zed" | "amp"
    ) {
        if is_denied {
            return HookDispatchResult {
                stdout_json: json!({
                    "permission": "deny",
                    "userMessage": format!("Vestige gate ({rule}): {reason}"),
                    "agentMessage": format!("Blocked by Vestige gate ({rule}): {reason}"),
                }),
                stderr_message: None,
                exit_code: 0,
            };
        }
        if let Some(ctx) = additional_context
            && !ctx.is_empty()
        {
            return HookDispatchResult {
                stdout_json: json!({ "additionalContext": ctx }),
                stderr_message: None,
                exit_code: 0,
            };
        }
        return HookDispatchResult {
            stdout_json: json!({}),
            stderr_message: None,
            exit_code: 0,
        };
    }

    // Windsurf / Codex dialect.
    if matches!(rt.as_str(), "windsurf" | "codex") {
        if is_denied {
            return HookDispatchResult {
                stdout_json: json!({
                    "decision": "block",
                    "reason": format!("Vestige gate ({rule}): {reason}"),
                }),
                stderr_message: None,
                exit_code: 0,
            };
        }
        if let Some(ctx) = additional_context
            && !ctx.is_empty()
        {
            return HookDispatchResult {
                stdout_json: json!({ "additionalContext": ctx }),
                stderr_message: None,
                exit_code: 0,
            };
        }
        return HookDispatchResult {
            stdout_json: json!({}),
            stderr_message: None,
            exit_code: 0,
        };
    }

    // Antigravity native plugin dialect (`decision: "deny"` / `injectSteps`).
    if rt == "antigravity" {
        if is_denied {
            return HookDispatchResult {
                stdout_json: json!({
                    "decision": "deny",
                    "reason": format!("Vestige gate ({rule}): {reason}"),
                }),
                stderr_message: None,
                exit_code: 0,
            };
        }
        if let Some(ctx) = additional_context
            && !ctx.is_empty()
        {
            return HookDispatchResult {
                stdout_json: json!({
                    "injectSteps": [{ "systemMessage": { "systemMessage": ctx } }]
                }),
                stderr_message: None,
                exit_code: 0,
            };
        }
        return HookDispatchResult {
            stdout_json: json!({}),
            stderr_message: None,
            exit_code: 0,
        };
    }

    // Default Claude Code / Gemini CLI / Devin / Factory / OpenHands / Goose / Kiro schema.
    let mut spec = serde_json::Map::new();
    spec.insert("hookEventName".into(), json!(event.as_str()));
    if is_denied {
        spec.insert("permissionDecision".into(), json!("deny"));
        spec.insert(
            "permissionDecisionReason".into(),
            json!(format!("Vestige gate ({rule}): {reason}")),
        );
    }
    if let Some(ctx) = additional_context
        && !ctx.is_empty()
    {
        spec.insert("additionalContext".into(), json!(ctx));
    }
    if spec.len() <= 1 {
        HookDispatchResult {
            stdout_json: json!({}),
            stderr_message: None,
            exit_code: 0,
        }
    } else {
        HookDispatchResult {
            stdout_json: json!({ "hookSpecificOutput": Value::Object(spec) }),
            stderr_message: None,
            exit_code: 0,
        }
    }
}

/// Dispatch one native hook invocation from stdin JSON without spawning Python.
pub fn dispatch_native_hook(
    runtime: &str,
    raw_event: &str,
    raw_stdin: &str,
    write_receipt: bool,
) -> HookDispatchResult {
    let event = CanonicalEvent::parse(raw_event);
    let parsed: Value = serde_json::from_str(raw_stdin).unwrap_or_else(|_| json!({}));
    let session_id = parsed
        .get("session_id")
        .or_else(|| parsed.get("conversationId"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let cwd = parsed
        .get("cwd")
        .or_else(|| parsed.pointer("/workspacePaths/0"))
        .and_then(Value::as_str)
        .unwrap_or("");

    match event {
        CanonicalEvent::UserPromptSubmit => {
            let prompt = parsed
                .get("prompt")
                .or_else(|| parsed.get("lastUserInput"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let ctx = capture_mission_from_prompt(&missions_dir(), session_id, cwd, prompt).map(
                |m| {
                    format!(
                        "Vestige Mission Contract captured ({} target(s) to remove): {}",
                        m.required_deleted_paths.len(),
                        m.required_deleted_paths.join(", ")
                    )
                },
            );
            format_host_response(runtime, event, None, ctx.as_deref())
        }
        CanonicalEvent::PreToolUse => {
            let mut req: GuardEvalRequest = request_from_hook_stdin(runtime, raw_stdin);
            req.event = event.as_str().to_string();
            let mut verdict = evaluate_gate_request(req.clone());
            let mut decision_override = None;
            if verdict.decision == GuardDecision::Deny {
                let op_home = operator_home_dir();
                let ad = compute_action_digest(&req);
                if try_consume_single_use_permit(&op_home, &ad) {
                    verdict.decision = GuardDecision::Allow;
                    verdict.reason = "Authorized by single-use owner permit.".into();
                    decision_override = Some("ALLOW");
                }
            }
            if write_receipt {
                let _ = append_receipt_with_decision(
                    &default_receipts_path(),
                    &req,
                    &verdict,
                    decision_override,
                );
            }
            format_host_response(runtime, event, Some(&verdict), None)
        }
        CanonicalEvent::Stop => {
            let stop_hook_active = parsed
                .get("stop_hook_active")
                .or_else(|| parsed.get("continuationActive"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if let Some(reason) = verify_mission_on_stop(&missions_dir(), session_id, stop_hook_active)
            {
                let verdict = GuardEvalVerdict::deny(
                    "vestige-mission-contract",
                    "MISSION-INCOMPLETE",
                    None,
                    reason,
                    0,
                );
                return format_host_response(runtime, event, Some(&verdict), None);
            }
            format_host_response(runtime, event, None, None)
        }
        _ => {
            let _ = append_receipt_o1; // keep import cleanly used
            format_host_response(runtime, event, None, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_event_normalizes_all_major_dialects() {
        assert_eq!(CanonicalEvent::parse("beforeSubmitPrompt"), CanonicalEvent::UserPromptSubmit);
        assert_eq!(CanonicalEvent::parse("PreInvocation"), CanonicalEvent::UserPromptSubmit);
        assert_eq!(CanonicalEvent::parse("pre_run_command"), CanonicalEvent::PreToolUse);
        assert_eq!(CanonicalEvent::parse("tool.execute.before"), CanonicalEvent::PreToolUse);
        assert_eq!(CanonicalEvent::parse("afterAgentResponse"), CanonicalEvent::Stop);
    }

    #[test]
    fn dispatch_blocks_destructive_tool_in_every_host_dialect() {
        let payload = r#"{"tool_name":"Bash","tool_input":{"command":"git push --force origin main"}}"#;

        let claude = dispatch_native_hook("claude", "PreToolUse", payload, false);
        assert_eq!(claude.exit_code, 0);
        assert_eq!(claude.stdout_json["hookSpecificOutput"]["permissionDecision"], "deny");

        let cursor = dispatch_native_hook("cursor", "preToolUse", payload, false);
        assert_eq!(cursor.exit_code, 0);
        assert_eq!(cursor.stdout_json["permission"], "deny");

        let agy = dispatch_native_hook("antigravity", "PreToolUse", payload, false);
        assert_eq!(agy.exit_code, 0);
        assert_eq!(agy.stdout_json["decision"], "deny");

        let cline = dispatch_native_hook("cline", "PreToolUse", payload, false);
        assert_eq!(cline.exit_code, 2);
        assert!(cline.stderr_message.unwrap().contains("OP-004"));
    }

    #[test]
    fn dispatch_allows_readonly_command_cleanly() {
        let payload = r#"{"tool_name":"run_command","tool_input":{"command":"Get-ChildItem crates -Name"}}"#;
        let agy = dispatch_native_hook("antigravity", "PreToolUse", payload, false);
        assert_eq!(agy.exit_code, 0);
        assert_eq!(agy.stdout_json, json!({}));
    }

    #[test]
    fn mission_contract_captures_deletion_targets_and_blocks_stop_until_deleted() {
        let ws = tempfile::tempdir().unwrap();
        let missions = tempfile::tempdir().unwrap();
        let legacy_dir = ws.path().join("operator-lite");
        fs::create_dir_all(&legacy_dir).unwrap();
        let legacy_file = ws.path().join("hooks").join("save-gate.py");
        fs::create_dir_all(legacy_file.parent().unwrap()).unwrap();
        fs::write(&legacy_file, "# legacy\n").unwrap();

        let prompt = "Execute mega consolidation: delete operator-lite/ and hooks/save-gate.py right now";
        let captured = capture_mission_from_prompt(
            missions.path(),
            "sess-mission-1",
            ws.path().to_str().unwrap(),
            prompt,
        )
        .expect("should capture mission targets");
        assert_eq!(
            captured.required_deleted_paths,
            vec!["operator-lite".to_string(), "hooks/save-gate.py".to_string()]
        );

        // Before deletion: Stop MUST block!
        let blocked = verify_mission_on_stop(missions.path(), "sess-mission-1", false);
        assert!(blocked.is_some(), "Stop must block while targets still exist");
        let reason = blocked.unwrap();
        assert!(reason.contains("operator-lite"), "{reason}");
        assert!(reason.contains("hooks/save-gate.py"), "{reason}");

        // Delete both targets -> Stop now allows and closes the mission!
        fs::remove_dir_all(&legacy_dir).unwrap();
        fs::remove_file(&legacy_file).unwrap();
        assert!(verify_mission_on_stop(missions.path(), "sess-mission-1", false).is_none());
    }
}
