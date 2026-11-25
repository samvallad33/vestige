//! Vestige Execution Lease — deterministic mid-session drift & loop control.
//!
//! Enforces owner-issued execution bounds inside Reflex hooks (`PreToolUse`,
//! `PostToolUse`, `PostToolUseFailure`, `PostToolBatch`, `UserPromptSubmit`)
//! with zero LLM calls and fail-closed semantics:
//!
//! 1. **Emergency Halt (`vestige lease halt`)**: Immediately stops any active
//!    agent loop on the very next hook firing (`permissionDecision: "deny"` +
//!    top-level `continue: false` + `stopReason`).
//! 2. **File-Scope Lock (`allowed_paths` / `denied_paths`)**: Hard-denies file
//!    edits outside the granted scope or inside protected/disjoint paths before
//!    a byte is written.
//! 3. **Command / Benchmark Gate (`denied_command_patterns`)**: Hard-denies
//!    unauthorized shell commands (e.g., long benchmarks or forbidden runners)
//!    before execution.
//! 4. **Exclusive Heavy-Command Mutex (`mutex_command_patterns`)**: Prevents
//!    two concurrent agents from running heavy commands simultaneously via an
//!    atomic TTL lockfile (`<data_dir>/reflex/cmd_mutex.json`) acquired on
//!    `PreToolUse` and released on `PostToolUse` / `PostToolUseFailure`.
//! 5. **Turn Tool-Call Budget & Spin-Loop Circuit Breaker**: Halts runaway
//!    turns (`max_tool_calls_per_turn`) or repeated identical failures without
//!    an intervening edit (`max_identical_failures`) using `continue: false`.
//! 6. **Mid-Session Owner Steering (`vestige steer`)**: Injects a one-shot
//!    owner directive into the very next mid-turn hook firing (`additionalContext`)
//!    or hard-pauses the loop when `--halt` is set.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};

const LEASE_FILE: &str = "lease.json";
const STEER_FILE: &str = "steer.json";
const MUTEX_FILE: &str = "cmd_mutex.json";
const DEFAULT_MUTEX_TTL_SECS: i64 = 15 * 60;
const MAX_STEER_HISTORY: usize = 32;

/// Active execution contract persisted under `<data_dir>/reflex/lease.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionLease {
    pub id: String,
    pub active: bool,
    #[serde(default)]
    pub halted: bool,
    #[serde(default)]
    pub halt_reason: Option<String>,
    /// Optional session id filter; `None` governs all sessions on this store.
    #[serde(default)]
    pub session_filter: Option<String>,
    /// Optional path allowlist for file edits (normalized prefix/substring match).
    #[serde(default)]
    pub allowed_paths: Vec<String>,
    /// Protected/disjoint paths that edits may never touch while leased.
    #[serde(default)]
    pub denied_paths: Vec<String>,
    /// Shell command substrings that are forbidden while this lease is active.
    #[serde(default)]
    pub denied_command_patterns: Vec<String>,
    /// Shell command substrings requiring the exclusive command mutex.
    #[serde(default)]
    pub mutex_command_patterns: Vec<String>,
    /// Mutex lock TTL in seconds (defaults to 900s).
    #[serde(default)]
    pub mutex_ttl_secs: Option<i64>,
    /// Maximum tool calls allowed in a single turn before halting the loop.
    #[serde(default)]
    pub max_tool_calls_per_turn: Option<u32>,
    /// Maximum consecutive identical failures allowed without an intervening edit.
    #[serde(default)]
    pub max_identical_failures: Option<u32>,
    /// Unix timestamp (seconds) when this lease automatically expires.
    #[serde(default)]
    pub expires_at: Option<i64>,
    #[serde(default)]
    pub updated_at: i64,
}

impl ExecutionLease {
    pub fn is_live(&self, now: i64, session: &str) -> bool {
        if !self.active {
            return false;
        }
        if let Some(exp) = self.expires_at
            && now >= exp
        {
            return false;
        }
        if let Some(ref filter) = self.session_filter
            && !filter.is_empty()
            && filter != session
        {
            return false;
        }
        true
    }
}

/// One-shot owner steering message persisted under `<data_dir>/reflex/steer.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SteerDirective {
    pub id: String,
    pub message: String,
    #[serde(default)]
    pub halt: bool,
    #[serde(default)]
    pub session_filter: Option<String>,
    pub created_at: i64,
    #[serde(default)]
    pub delivered_sessions: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SteerQueue {
    #[serde(default)]
    pub items: Vec<SteerDirective>,
}

/// Exclusive heavy-command mutex state under `<data_dir>/reflex/cmd_mutex.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommandMutex {
    pub session: String,
    pub pattern: String,
    pub command_preview: String,
    pub acquired_at: i64,
    pub expires_at: i64,
}

/// Outcome of evaluating an active lease against a `PreToolUse` or `PostToolBatch`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeaseVerdict {
    pub deny_reason: Option<String>,
    pub halt_loop: bool,
}

/// Context passed into `check_pre_tool` so enforcement stays cleanly typed.
#[derive(Clone, Copy, Debug)]
pub struct PreToolCheck<'a> {
    pub data_dir: &'a Path,
    pub session: &'a str,
    pub is_edit: bool,
    pub is_shell: bool,
    pub file: &'a str,
    pub command: &'a str,
    pub tool_calls_this_turn: u32,
    pub last_fail_key: &'a str,
    pub consecutive_same_fail: u32,
    pub current_fail_key: &'a str,
}

/// Context passed into `check_post_batch`.
#[derive(Clone, Copy, Debug)]
pub struct PostBatchCheck<'a> {
    pub data_dir: &'a Path,
    pub session: &'a str,
    pub tool_calls_this_turn: u32,
    pub consecutive_same_fail: u32,
    pub last_fail_key: &'a str,
}

fn reflex_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("reflex")
}

pub fn lease_path(data_dir: &Path) -> PathBuf {
    reflex_dir(data_dir).join(LEASE_FILE)
}

pub fn steer_path(data_dir: &Path) -> PathBuf {
    reflex_dir(data_dir).join(STEER_FILE)
}

pub fn mutex_path(data_dir: &Path) -> PathBuf {
    reflex_dir(data_dir).join(MUTEX_FILE)
}

fn write_json_atomic<T: Serialize>(path: &Path, val: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create dir {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    let body = serde_json::to_string_pretty(val).map_err(|e| format!("serialize: {e}"))?;
    fs::write(&tmp, body).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    // On Windows fs::rename fails if destination exists; remove first best-effort.
    if path.exists() {
        let _ = fs::remove_file(path);
    }
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!("rename {} -> {}: {e}", tmp.display(), path.display())
    })
}

pub fn load_lease(data_dir: &Path) -> Option<ExecutionLease> {
    let raw = fs::read_to_string(lease_path(data_dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn save_lease(data_dir: &Path, lease: &ExecutionLease) -> Result<(), String> {
    write_json_atomic(&lease_path(data_dir), lease)
}

pub fn clear_lease(data_dir: &Path) -> Result<(), String> {
    let lp = lease_path(data_dir);
    if lp.exists() {
        fs::remove_file(&lp).map_err(|e| format!("remove {}: {e}", lp.display()))?;
    }
    let mp = mutex_path(data_dir);
    if mp.exists() {
        let _ = fs::remove_file(&mp);
    }
    Ok(())
}

pub fn set_emergency_halt(
    data_dir: &Path,
    reason: &str,
    session_filter: Option<String>,
) -> Result<ExecutionLease, String> {
    let now = Utc::now().timestamp();
    let mut lease = load_lease(data_dir).unwrap_or_else(|| ExecutionLease {
        id: format!("lease-{now}"),
        active: true,
        ..ExecutionLease::default()
    });
    lease.active = true;
    lease.halted = true;
    lease.halt_reason = Some(reason.trim().to_string());
    if session_filter.is_some() {
        lease.session_filter = session_filter;
    }
    lease.updated_at = now;
    save_lease(data_dir, &lease)?;
    Ok(lease)
}

pub fn load_steer_queue(data_dir: &Path) -> SteerQueue {
    let Ok(raw) = fs::read_to_string(steer_path(data_dir)) else {
        return SteerQueue::default();
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

pub fn push_steer(
    data_dir: &Path,
    message: &str,
    halt: bool,
    session_filter: Option<String>,
) -> Result<SteerDirective, String> {
    let now = Utc::now().timestamp();
    let mut q = load_steer_queue(data_dir);
    let item = SteerDirective {
        id: format!("steer-{now}-{}", q.items.len() + 1),
        message: message.trim().to_string(),
        halt,
        session_filter,
        created_at: now,
        delivered_sessions: Vec::new(),
    };
    q.items.push(item.clone());
    if q.items.len() > MAX_STEER_HISTORY {
        let drop_n = q.items.len() - MAX_STEER_HISTORY;
        q.items.drain(..drop_n);
    }
    write_json_atomic(&steer_path(data_dir), &q)?;
    Ok(item)
}

/// Drain any undelivered owner steering directives for `session`.
/// Returns `(lines_to_inject, halt_reason_if_any)`.
pub fn consume_steers_for_session(data_dir: &Path, session: &str) -> (Vec<String>, Option<String>) {
    if session.is_empty() {
        return (Vec::new(), None);
    }
    let path = steer_path(data_dir);
    if !path.exists() {
        return (Vec::new(), None);
    }
    let mut q = load_steer_queue(data_dir);
    let mut lines = Vec::new();
    let mut halt_reason: Option<String> = None;
    let mut changed = false;

    for item in &mut q.items {
        if let Some(ref f) = item.session_filter
            && !f.is_empty()
            && f != session
        {
            continue;
        }
        if item.delivered_sessions.iter().any(|s| s == session) {
            continue;
        }
        item.delivered_sessions.push(session.to_string());
        changed = true;
        if item.halt {
            lines.push(format!(
                "Vestige Execution Lease [OWNER HALT {}]: {}",
                item.id, item.message
            ));
            halt_reason = Some(format!(
                "Vestige Execution Lease [OWNER HALT {}]: {}",
                item.id, item.message
            ));
        } else {
            lines.push(format!(
                "Vestige Execution Lease [OWNER STEER {}]: {}",
                item.id, item.message
            ));
        }
    }

    if changed {
        let _ = write_json_atomic(&path, &q);
    }
    (lines, halt_reason)
}

/// Peek whether an undelivered `halt == true` owner steer exists for `session`
/// without mutating disk (used by `PreToolUse` / `PostToolBatch` decisions).
pub fn pending_halt_steer(data_dir: &Path, session: &str) -> Option<String> {
    if session.is_empty() || !steer_path(data_dir).exists() {
        return None;
    }
    let q = load_steer_queue(data_dir);
    q.items.iter().rev().find_map(|item| {
        if !item.halt {
            return None;
        }
        if let Some(ref f) = item.session_filter
            && !f.is_empty()
            && f != session
        {
            return None;
        }
        if item.delivered_sessions.iter().any(|s| s == session) {
            return None;
        }
        Some(format!(
            "Vestige Execution Lease [OWNER HALT {}]: {}",
            item.id, item.message
        ))
    })
}

pub fn load_mutex(data_dir: &Path) -> Option<CommandMutex> {
    let raw = fs::read_to_string(mutex_path(data_dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Release the exclusive command mutex if `session` holds it (or if expired).
pub fn release_mutex_for_session(data_dir: &Path, session: &str) {
    let mp = mutex_path(data_dir);
    let Some(cur) = load_mutex(data_dir) else {
        return;
    };
    let now = Utc::now().timestamp();
    if cur.session == session || now >= cur.expires_at {
        let _ = fs::remove_file(mp);
    }
}

fn normalize_path_key(raw: &str) -> String {
    raw.trim()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_ascii_lowercase()
}

fn path_matches_rule(target_norm: &str, rule: &str) -> bool {
    let r = normalize_path_key(rule);
    if r.is_empty() {
        return false;
    }
    let r_trimmed = r.trim_end_matches('*').trim_end_matches('/');
    if r_trimmed.is_empty() {
        return true;
    }
    target_norm == r_trimmed
        || target_norm.starts_with(&format!("{r_trimmed}/"))
        || target_norm.ends_with(&format!("/{r_trimmed}"))
        || target_norm.contains(&format!("/{r_trimmed}/"))
}

fn command_matches_pattern(command: &str, pattern: &str) -> bool {
    let pat = pattern.trim().to_ascii_lowercase();
    if pat.is_empty() {
        return false;
    }
    command.to_ascii_lowercase().contains(&pat)
}

/// Compute a compact canonical failure key for spin-loop detection.
pub fn failure_key(tool: &str, file: &str, command: &str) -> String {
    let t = tool.trim().to_ascii_lowercase();
    if !command.trim().is_empty() {
        let norm_cmd: String = command
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(180)
            .collect();
        return format!("{t}:{norm_cmd}");
    }
    if !file.trim().is_empty() {
        return format!("{t}:{}", normalize_path_key(file));
    }
    t
}

/// Evaluate the Execution Lease before a tool call executes (`PreToolUse`).
pub fn check_pre_tool(ctx: &PreToolCheck<'_>) -> LeaseVerdict {
    let now = Utc::now().timestamp();
    let live_lease = load_lease(ctx.data_dir).filter(|l| l.is_live(now, ctx.session));

    // 1. Emergency halt overrides everything immediately (including older steers).
    if let Some(ref lease) = live_lease
        && lease.halted
    {
        let msg = lease
            .halt_reason
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("owner halted execution via `vestige lease halt`");
        return LeaseVerdict {
            deny_reason: Some(format!(
                "Vestige Execution Lease [HALT {}]: {msg}",
                lease.id
            )),
            halt_loop: true,
        };
    }

    if let Some(reason) = pending_halt_steer(ctx.data_dir, ctx.session) {
        return LeaseVerdict {
            deny_reason: Some(reason),
            halt_loop: true,
        };
    }

    let Some(lease) = live_lease else {
        return LeaseVerdict::default();
    };

    // 2. Turn tool-call budget fuse.
    if let Some(limit) = lease.max_tool_calls_per_turn
        && limit > 0
        && ctx.tool_calls_this_turn >= limit
    {
        return LeaseVerdict {
            deny_reason: Some(format!(
                "Vestige Execution Lease [BUDGET {}]: turn reached {}/{} tool calls; halting loop for owner check-in.",
                lease.id, ctx.tool_calls_this_turn, limit
            )),
            halt_loop: true,
        };
    }

    // 3. Spin-loop circuit breaker on identical repeated failure without intervening edit.
    if let Some(max_fail) = lease.max_identical_failures
        && max_fail > 0
        && ctx.consecutive_same_fail >= max_fail
        && !ctx.last_fail_key.is_empty()
        && ctx.last_fail_key == ctx.current_fail_key
    {
        let est_cents = ctx.consecutive_same_fail.saturating_mul(70);
        return LeaseVerdict {
            deny_reason: Some(format!(
                "Vestige Execution Lease [CIRCUIT-BREAKER {}]: `{}` failed {} consecutive times without an intervening edit; halting spin loop (saved ~${}.{:02} in retry burn).",
                lease.id,
                ctx.current_fail_key,
                ctx.consecutive_same_fail,
                est_cents / 100,
                est_cents % 100
            )),
            halt_loop: true,
        };
    }

    // 4. File-scope lock (`denied_paths` then `allowed_paths`).
    if ctx.is_edit && !ctx.file.trim().is_empty() {
        let norm = normalize_path_key(ctx.file);
        for denied in &lease.denied_paths {
            if path_matches_rule(&norm, denied) {
                return LeaseVerdict {
                    deny_reason: Some(format!(
                        "Vestige Execution Lease [SCOPE-DENY {}]: editing `{}` is forbidden by protected path rule `{denied}`.",
                        lease.id, ctx.file
                    )),
                    halt_loop: false,
                };
            }
        }
        if !lease.allowed_paths.is_empty()
            && !lease
                .allowed_paths
                .iter()
                .any(|allowed| path_matches_rule(&norm, allowed))
        {
            return LeaseVerdict {
                deny_reason: Some(format!(
                    "Vestige Execution Lease [SCOPE-LOCK {}]: editing `{}` is outside allowed scope [{}].",
                    lease.id,
                    ctx.file,
                    lease.allowed_paths.join(", ")
                )),
                halt_loop: false,
            };
        }
    }

    // 5. Shell command gate (`denied_command_patterns`) & exclusive mutex (`mutex_command_patterns`).
    if ctx.is_shell && !ctx.command.trim().is_empty() {
        for pat in &lease.denied_command_patterns {
            if command_matches_pattern(ctx.command, pat) {
                return LeaseVerdict {
                    deny_reason: Some(format!(
                        "Vestige Execution Lease [CMD-DENY {}]: shell command matches forbidden pattern `{pat}`. Ask the owner or update the lease before running it.",
                        lease.id
                    )),
                    halt_loop: false,
                };
            }
        }
        for pat in &lease.mutex_command_patterns {
            if command_matches_pattern(ctx.command, pat) {
                if let Some(existing) = load_mutex(ctx.data_dir)
                    && now < existing.expires_at
                    && existing.session != ctx.session
                {
                    return LeaseVerdict {
                        deny_reason: Some(format!(
                            "Vestige Execution Lease [CMD-MUTEX {}]: `{pat}` is already running under session `{}` (`{}`). Wait for it to finish before launching another run.",
                            lease.id, existing.session, existing.command_preview
                        )),
                        halt_loop: false,
                    };
                }
                let ttl = lease
                    .mutex_ttl_secs
                    .filter(|&t| t > 0)
                    .unwrap_or(DEFAULT_MUTEX_TTL_SECS);
                let lock = CommandMutex {
                    session: ctx.session.to_string(),
                    pattern: pat.clone(),
                    command_preview: ctx.command.chars().take(120).collect(),
                    acquired_at: now,
                    expires_at: now + ttl,
                };
                let _ = write_json_atomic(&mutex_path(ctx.data_dir), &lock);
                break;
            }
        }
    }

    LeaseVerdict::default()
}

/// Evaluate loop-level halt conditions after a parallel tool batch (`PostToolBatch`).
pub fn check_post_batch(ctx: &PostBatchCheck<'_>) -> LeaseVerdict {
    let now = Utc::now().timestamp();
    let live_lease = load_lease(ctx.data_dir).filter(|l| l.is_live(now, ctx.session));

    if let Some(ref lease) = live_lease
        && lease.halted
    {
        let msg = lease
            .halt_reason
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or("owner halted execution via `vestige lease halt`");
        return LeaseVerdict {
            deny_reason: Some(format!(
                "Vestige Execution Lease [HALT {}]: {msg}",
                lease.id
            )),
            halt_loop: true,
        };
    }

    if let Some(reason) = pending_halt_steer(ctx.data_dir, ctx.session) {
        return LeaseVerdict {
            deny_reason: Some(reason),
            halt_loop: true,
        };
    }

    let Some(lease) = live_lease else {
        return LeaseVerdict::default();
    };

    if let Some(limit) = lease.max_tool_calls_per_turn
        && limit > 0
        && ctx.tool_calls_this_turn >= limit
    {
        return LeaseVerdict {
            deny_reason: Some(format!(
                "Vestige Execution Lease [BUDGET {}]: turn reached {}/{} tool calls; halting loop before next model turn.",
                lease.id, ctx.tool_calls_this_turn, limit
            )),
            halt_loop: true,
        };
    }

    if let Some(max_fail) = lease.max_identical_failures
        && max_fail > 0
        && ctx.consecutive_same_fail >= max_fail
        && !ctx.last_fail_key.is_empty()
    {
        return LeaseVerdict {
            deny_reason: Some(format!(
                "Vestige Execution Lease [CIRCUIT-BREAKER {}]: `{}` failed {} consecutive times without an intervening edit; stopping batch loop.",
                lease.id, ctx.last_fail_key, ctx.consecutive_same_fail
            )),
            halt_loop: true,
        };
    }

    LeaseVerdict::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_lease() -> ExecutionLease {
        ExecutionLease {
            id: "lease-test".into(),
            active: true,
            halted: false,
            halt_reason: None,
            session_filter: None,
            allowed_paths: vec!["crates/vestige-mcp/src/tools/reflex".into()],
            denied_paths: vec!["crates/vestige-core/src/wordnet.rs".into()],
            denied_command_patterns: vec!["retrieval_bench".into(), "run_stage1".into()],
            mutex_command_patterns: vec!["cargo test".into()],
            mutex_ttl_secs: Some(300),
            max_tool_calls_per_turn: Some(12),
            max_identical_failures: Some(2),
            expires_at: None,
            updated_at: Utc::now().timestamp(),
        }
    }

    #[test]
    fn lease_blocks_unauthorized_benchmark_command_in_pre_tool_use() {
        let dir = TempDir::new().expect("tempdir");
        save_lease(dir.path(), &sample_lease()).expect("save lease");

        let blocked = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-a",
            is_edit: false,
            is_shell: true,
            file: "",
            command: "cargo run --bin retrieval_bench -- --beam",
            tool_calls_this_turn: 1,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(
            blocked
                .deny_reason
                .as_deref()
                .is_some_and(|r| r.contains("CMD-DENY") && r.contains("retrieval_bench"))
        );
        assert!(!blocked.halt_loop);

        let allowed = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-a",
            is_edit: false,
            is_shell: true,
            file: "",
            command: "cargo clippy -p vestige-mcp -- -D warnings",
            tool_calls_this_turn: 1,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(allowed.deny_reason.is_none());
    }

    #[test]
    fn lease_enforces_allowed_and_denied_file_scope_locks() {
        let dir = TempDir::new().expect("tempdir");
        save_lease(dir.path(), &sample_lease()).expect("save lease");

        let ok = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-a",
            is_edit: true,
            is_shell: false,
            file: r"C:\repo\crates\vestige-mcp\src\tools\reflex\lease.rs",
            command: "",
            tool_calls_this_turn: 2,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(ok.deny_reason.is_none());

        let denied_protected = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-a",
            is_edit: true,
            is_shell: false,
            file: r"C:\repo\crates\vestige-core\src\wordnet.rs",
            command: "",
            tool_calls_this_turn: 2,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(
            denied_protected
                .deny_reason
                .as_deref()
                .is_some_and(|r| r.contains("SCOPE-DENY"))
        );

        let outside_allow = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-a",
            is_edit: true,
            is_shell: false,
            file: r"C:\repo\crates\vestige-core\src\retrieval\rank\stage12.rs",
            command: "",
            tool_calls_this_turn: 2,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(
            outside_allow
                .deny_reason
                .as_deref()
                .is_some_and(|r| r.contains("SCOPE-LOCK"))
        );
    }

    #[test]
    fn lease_mutex_blocks_concurrent_command_and_releases_cleanly() {
        let dir = TempDir::new().expect("tempdir");
        save_lease(dir.path(), &sample_lease()).expect("save lease");

        let first = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-main",
            is_edit: false,
            is_shell: true,
            file: "",
            command: "cargo test -p vestige-core",
            tool_calls_this_turn: 1,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(first.deny_reason.is_none());

        let second = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-side",
            is_edit: false,
            is_shell: true,
            file: "",
            command: "cargo test -p vestige-mcp",
            tool_calls_this_turn: 1,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(
            second
                .deny_reason
                .as_deref()
                .is_some_and(|r| r.contains("CMD-MUTEX") && r.contains("sess-main"))
        );

        release_mutex_for_session(dir.path(), "sess-main");

        let after_release = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-side",
            is_edit: false,
            is_shell: true,
            file: "",
            command: "cargo test -p vestige-mcp",
            tool_calls_this_turn: 1,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(after_release.deny_reason.is_none());
    }

    #[test]
    fn lease_spin_loop_and_budget_fuses_halt_agentic_loop() {
        let dir = TempDir::new().expect("tempdir");
        save_lease(dir.path(), &sample_lease()).expect("save lease");

        let key = failure_key("Bash", "", "cargo check -p vestige-core");
        let spin = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-a",
            is_edit: false,
            is_shell: true,
            file: "",
            command: "cargo check -p vestige-core",
            tool_calls_this_turn: 4,
            last_fail_key: &key,
            consecutive_same_fail: 2,
            current_fail_key: &key,
        });
        assert!(spin.halt_loop);
        assert!(
            spin.deny_reason
                .as_deref()
                .is_some_and(|r| r.contains("CIRCUIT-BREAKER"))
        );

        let budget = check_post_batch(&PostBatchCheck {
            data_dir: dir.path(),
            session: "sess-a",
            tool_calls_this_turn: 12,
            consecutive_same_fail: 0,
            last_fail_key: "",
        });
        assert!(budget.halt_loop);
        assert!(
            budget
                .deny_reason
                .as_deref()
                .is_some_and(|r| r.contains("BUDGET"))
        );
    }

    #[test]
    fn emergency_halt_and_owner_steer_deliver_once_and_stop_loop() {
        let dir = TempDir::new().expect("tempdir");

        push_steer(dir.path(), "Focus only on Stage 12 wiring", false, None).expect("steer");
        let (lines1, halt1) = consume_steers_for_session(dir.path(), "sess-a");
        assert_eq!(lines1.len(), 1);
        assert!(lines1[0].contains("OWNER STEER"));
        assert!(halt1.is_none());

        // Second delivery to same session is empty (one-shot per session).
        let (lines2, _) = consume_steers_for_session(dir.path(), "sess-a");
        assert!(lines2.is_empty());

        // Urgent `--halt` steer interrupts immediately before consumption.
        push_steer(dir.path(), "STOP: do not run full BEAM now", true, None).expect("steer halt");
        let pre = check_pre_tool(&PreToolCheck {
            data_dir: dir.path(),
            session: "sess-a",
            is_edit: false,
            is_shell: true,
            file: "",
            command: "echo hi",
            tool_calls_this_turn: 1,
            last_fail_key: "",
            consecutive_same_fail: 0,
            current_fail_key: "",
        });
        assert!(pre.halt_loop);
        assert!(
            pre.deny_reason
                .as_deref()
                .is_some_and(|r| r.contains("OWNER HALT"))
        );

        // Emergency lease halt also interrupts PostToolBatch.
        set_emergency_halt(dir.path(), "manual owner stop", None).expect("halt");
        let batch = check_post_batch(&PostBatchCheck {
            data_dir: dir.path(),
            session: "sess-b",
            tool_calls_this_turn: 1,
            consecutive_same_fail: 0,
            last_fail_key: "",
        });
        assert!(batch.halt_loop);
        assert!(
            batch
                .deny_reason
                .as_deref()
                .is_some_and(|r| r.contains("HALT") && r.contains("manual owner stop"))
        );
    }
}
