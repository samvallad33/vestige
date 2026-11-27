//! Unified native Rust hook dispatcher (`vestige hook dispatch <runtime> <event>`).
//!
//! Normalizes payloads across all 24 supported AI coding runtimes into one canonical
//! representation, routes `PreToolUse` directly through [`crate::native_gate`] (`<1ms`,
//! zero Python), and emits each host's exact wire JSON schema so JS/shell shims can
//! collapse into thin 1-line invocations.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use vestige_core::{GuardDecision, GuardEvalRequest, GuardEvalVerdict};

use crate::native_gate::{
    append_receipt_o1, default_receipts_path, evaluate_gate_request, request_from_hook_stdin,
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
    match event {
        CanonicalEvent::PreToolUse => {
            let mut req: GuardEvalRequest = request_from_hook_stdin(runtime, raw_stdin);
            req.event = event.as_str().to_string();
            let verdict = evaluate_gate_request(req.clone());
            if write_receipt {
                let _ = append_receipt_o1(&default_receipts_path(), &req, &verdict);
            }
            format_host_response(runtime, event, Some(&verdict), None)
        }
        _ => format_host_response(runtime, event, None, None),
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
}
