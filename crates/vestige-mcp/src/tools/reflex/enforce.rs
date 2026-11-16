//! Ledger enforcement for one hook firing.
//!
//! Host contract, checked against `https://code.claude.com/docs/en/hooks`
//! on 2026-11-16:
//!
//! - `additionalContext` lives in `hookSpecificOutput` and is capped at
//!   10,000 characters per string. Over that, the host spills to a file.
//! - Injected text is phrased as a ledger fact. Imperative banners are not
//!   used.
//! - `Stop` and `SubagentStop` block with top-level `decision: "block"` and
//!   `reason`. `onFailure` has no effect on those events (verified: "The
//!   field has no effect on these hooks: Stop, SubagentStop, TaskCompleted,
//!   and TeammateIdle").
//! - `PreToolUse` blocks with `hookSpecificOutput.permissionDecision:
//!   "deny"`. Top-level `decision` on this event is deprecated.
//! - `PermissionRequest` uses `hookSpecificOutput.decision.behavior`. Exit
//!   code 2 is not honored. A one-use allow never includes `updatedPermissions`.
//! - `PreModelSwitch` accepts top-level `decision: "block"` and
//!   `permissionDecision` of allow/deny/ask. It does not accept
//!   `additionalContext`.
//! - `PostToolBatch` `decision: "block"` stops the loop before the next
//!   model call.
//! - `WorktreeRemove` discards JSON. A command hook must exit non-zero.
//!   `exitCode` in this JSON is for the shim and for tests. Whether an MCP
//!   tool error counts as that non-zero exit is **unverified**.
//! - `StopFailure` ignores output. Reflex only records a snapshot.
//!
//! Unverified, and not relied on: the `Setup` matchers `--init` /
//! `--maintenance`, and the default `CLAUDE_CODE_HOOK_HEARTBEAT_INTERVAL_MS`.

use super::*;

/// Host cap for one `additionalContext` string. Verified 2026-11-16.
pub(super) const CONTEXT_CAP: usize = 10_000;
const STOP_BLOCK_CAP: u32 = 2;
const PERMIT_TTL_SECS: i64 = 15 * 60;
const MAX_CITATIONS: usize = 32;

/// Literal directive markers. Editable JSON, matched with `contains` and a
/// token boundary, never a regex and never a domain lexicon.
const DIRECTIVE_PHRASES_JSON: &str = r#"["remember this","remember that","never","always","rule:","from now on","going forward","don't ever","decision:"]"#;

/// StopFailure `error` values the host documents. Anything else is stored as
/// `unknown` so a raw provider string cannot land in the ledger.
const STOP_ERRORS: &[&str] = &[
    "rate_limit",
    "overloaded",
    "authentication_failed",
    "oauth_org_not_allowed",
    "account_on_hold",
    "billing_error",
    "invalid_request",
    "model_not_found",
    "server_error",
    "max_output_tokens",
    "cloud_credential_error",
    "unknown",
];

const SENSITIVE_TAGS: &[&str] = &["auth", "migration", "security"];

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Permit {
    pub id: String,
    pub command: String,
    pub session: String,
    pub expires: i64,
    pub consumed: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Unpaired {
    pub sig: String,
    pub agent: String,
    pub command: String,
    pub exit_code: i64,
    pub turn: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Intention {
    pub id: String,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct StopSnap {
    pub error: String,
    pub files: Vec<String>,
    pub turn: u32,
    pub told: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct TurnLedger {
    pub pending: Vec<String>,
    pub pending_agent: String,
    pub stop_blocks: u32,
    pub unreceipted: Vec<String>,
    pub unpaired: Vec<Unpaired>,
    pub directives_told: Vec<String>,
    pub scopes_seen: Vec<String>,
    pub handles_told: Vec<String>,
    pub edited: Vec<String>,
    pub saved: Vec<String>,
    pub intentions: Vec<Intention>,
    pub stop_failure: Option<StopSnap>,
    pub unverified_told: Vec<String>,
    /// Allowlisted witness commands that passed this turn: a commit receipt
    /// must carry each one as its gate output.
    #[serde(default)]
    pub gates: Vec<String>,
    /// What git says about each commit made this turn.
    #[serde(default)]
    pub commits: Vec<crate::commit_receipt::CommitFacts>,
    /// A receipt named the SHA but left facts out: `(sha, missing)`.
    #[serde(default)]
    pub receipt_gaps: Vec<(String, Vec<String>)>,
}

impl TurnLedger {
    pub(super) fn begin_turn(&mut self) {
        self.pending.clear();
        self.stop_blocks = 0;
        self.unreceipted.clear();
        self.unpaired.clear();
        self.edited.clear();
        self.saved.clear();
        self.gates.clear();
        self.commits.clear();
        self.receipt_gaps.clear();
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct HostDecision {
    pub block_reason: Option<String>,
    pub permission_decision: Option<String>,
    pub permission_reason: Option<String>,
    pub permit_behavior: Option<String>,
    pub permit_message: Option<String>,
    pub updated_input: Option<Value>,
    pub retry: Option<bool>,
    /// Advisory process status for hosts that honor exit codes.
    pub exit_code: Option<i32>,
    pub overdue_item: Option<Value>,
    pub ledger: Value,
}

pub(super) fn cap_context(text: &str) -> String {
    if text.chars().count() <= CONTEXT_CAP {
        return text.to_string();
    }
    let note = "… [Vestige ledger: context truncated to 10000 characters]";
    let room = CONTEXT_CAP.saturating_sub(note.chars().count());
    let mut out: String = text.chars().take(room).collect();
    out.push_str(note);
    out
}

pub(super) fn host_json(event: &str, format: &str, text: &str, d: &HostDecision) -> Value {
    let text = cap_context(text);
    let mut v = serde_json::Map::new();
    if format == "plain" {
        if !text.is_empty() {
            v.insert("context".into(), json!(text));
            v.insert("event".into(), json!(event));
        }
        write_top_level(&mut v, event, d);
        // The command shim maps these to each host's own deny shape.
        if let Some(p) = &d.permission_decision {
            v.insert("permissionDecision".into(), json!(p));
            if let Some(r) = &d.permission_reason {
                v.insert("permissionDecisionReason".into(), json!(r));
            }
        }
        if let Some(b) = &d.permit_behavior {
            v.insert("permitBehavior".into(), json!(b));
        }
        return Value::Object(v);
    }
    write_top_level(&mut v, event, d);
    if let Some(spec) = specific_output(event, &text, d) {
        v.insert("hookSpecificOutput".into(), spec);
    }
    Value::Object(v)
}

fn write_top_level(v: &mut serde_json::Map<String, Value>, event: &str, d: &HostDecision) {
    let uses_decision = matches!(
        event,
        "Stop"
            | "SubagentStop"
            | "PostToolBatch"
            | "PreModelSwitch"
            | "UserPromptSubmit"
            | "PostToolUse"
            | "PreCompact"
            | "ConfigChange"
            | "TaskCreated"
            | "WorktreeRemove"
    );
    if uses_decision && let Some(reason) = &d.block_reason {
        v.insert("decision".into(), json!("block"));
        v.insert("reason".into(), json!(reason));
    }
    if event == "WorktreeRemove"
        && let Some(code) = d.exit_code
    {
        v.insert("exitCode".into(), json!(code));
    }
}

fn specific_output(event: &str, text: &str, d: &HostDecision) -> Option<Value> {
    let mut spec = serde_json::Map::new();
    // PreModelSwitch does not accept additionalContext (verified).
    // StopFailure, and the observation-only events, ignore output.
    let context_ok = !matches!(
        event,
        "PreModelSwitch"
            | "StopFailure"
            | "InstructionsLoaded"
            | "CwdChanged"
            | "FileChanged"
            | "DirectoryAdded"
            | "SessionEnd"
            | "Setup"
    );
    if context_ok && !text.is_empty() {
        spec.insert("additionalContext".into(), json!(text));
    }
    if matches!(event, "PreToolUse" | "PreModelSwitch") {
        if let Some(p) = &d.permission_decision {
            spec.insert("permissionDecision".into(), json!(p));
            if let Some(r) = &d.permission_reason {
                spec.insert("permissionDecisionReason".into(), json!(r));
            }
        }
        if let Some(u) = &d.updated_input {
            spec.insert("updatedInput".into(), u.clone());
        }
    }
    if event == "PermissionRequest"
        && let Some(behavior) = &d.permit_behavior
    {
        let mut decision = serde_json::Map::new();
        decision.insert("behavior".into(), json!(behavior));
        if behavior == "deny"
            && let Some(message) = &d.permit_message
        {
            decision.insert("message".into(), json!(message));
        }
        spec.insert("decision".into(), Value::Object(decision));
    }
    if event == "PermissionDenied"
        && let Some(retry) = d.retry
    {
        spec.insert("retry".into(), json!(retry));
    }
    if spec.is_empty() {
        return None;
    }
    spec.insert("hookEventName".into(), json!(event));
    Some(Value::Object(spec))
}

pub(super) fn register_single_use(g: &mut Global, row: &GateRow, stop_d16: &str) {
    if row.kind != "PERMIT_GRANTED" || !row.single_use {
        return;
    }
    let Some(preview) =
        g.p.recent
            .iter()
            .rev()
            .find(|r| r.d16 == stop_d16)
            .map(|r| r.preview.clone())
    else {
        return;
    };
    let command = normalize_command(&preview);
    if command.is_empty() {
        return;
    }
    let id = row.d16();
    if g.p.permits.iter().any(|p| p.id == id) {
        return;
    }
    g.p.permits.push(Permit {
        id,
        command,
        session: row.session.clone(),
        expires: Utc::now().timestamp() + PERMIT_TTL_SECS,
        consumed: false,
    });
    if g.p.permits.len() > 100 {
        let drop_n = g.p.permits.len() - 100;
        g.p.permits.drain(..drop_n);
    }
    g.dirty = true;
}

/// Record what this firing changed and return factual lines for the turn budget.
pub(super) fn observe(g: &mut Global, inp: &Input, storage: &Arc<Storage>) -> Vec<String> {
    if inp.session.is_empty() {
        return Vec::new();
    }
    note_intentions(g, inp);
    let mut lines = Vec::new();
    match inp.event.as_str() {
        "UserPromptSubmit" => {
            note_directives(g, inp);
            lines.extend(directive_lines(g, inp));
            lines.extend(handle_lines(g, inp, storage));
            lines.extend(unverified_lines(g, inp, storage));
            lines.extend(resume_lines(g, inp));
        }
        "SessionStart" => {
            lines.extend(unverified_lines(g, inp, storage));
            lines.extend(resume_lines(g, inp));
        }
        "PreToolUse" => lines.extend(scope_lines(g, inp, storage)),
        "PostToolUse" => {
            note_gate(g, inp, storage);
            note_success(g, inp);
            note_store_receipts(g, inp, storage);
        }
        "PostToolUseFailure" => note_failure(g, inp),
        "StopFailure" => note_stop_failure(g, inp),
        "SubagentStart" => lines.extend(subagent_lines(g)),
        _ => {}
    }
    lines
}

pub(super) fn decide(
    g: &mut Global,
    inp: &Input,
    storage: &Arc<Storage>,
    delivered: &str,
) -> HostDecision {
    let mut d = HostDecision::default();
    if inp.session.is_empty() {
        return d;
    }
    if inp.event == "UserPromptSubmit" {
        let s = g.session(&inp.session);
        if !s.ledger.pending.is_empty() {
            // The matched phrases only, never the prompt.
            d.ledger =
                json!({"pending_save": {"turn": s.turn, "phrases": s.ledger.pending.clone()}});
        }
    }
    if inp.no_enforce {
        mark_delivered(g, inp, delivered);
        return d;
    }
    match inp.event.as_str() {
        "Stop" | "SubagentStop" => stop_decision(g, inp, storage, &mut d),
        "PreToolUse" => pre_tool_decision(g, inp, storage, &mut d),
        "PermissionRequest" => permit_decision(g, inp, &mut d),
        "PreModelSwitch" => model_decision(g, inp, &mut d),
        "WorktreeRemove" => worktree_decision(inp, &mut d),
        "PostToolBatch" => batch_decision(inp, &mut d),
        _ => {}
    }
    mark_delivered(g, inp, delivered);
    let blocks = g.session(&inp.session).ledger.stop_blocks;
    if matches!(inp.event.as_str(), "Stop" | "SubagentStop") {
        d.ledger = json!({
            "enforce": {
                "decision": if d.block_reason.is_some() { "block" } else { "allow" },
                "stopBlocks": blocks,
                "overdue": d.overdue_item.is_some(),
            }
        });
    }
    d
}

fn note_intentions(g: &mut Global, inp: &Input) {
    let Some(rows) = inp.open_intentions.as_array() else {
        return;
    };
    let s = g.session(&inp.session);
    for row in rows {
        let id = row.get("id").and_then(Value::as_str).unwrap_or("").trim();
        if id.is_empty() || id.len() > 80 {
            continue;
        }
        let tags = row
            .get("tags")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .filter(|t| t.len() <= 64)
                    .take(8)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !s.ledger.intentions.iter().any(|i| i.id == id) {
            s.ledger.intentions.push(Intention {
                id: id.to_string(),
                tags,
            });
        }
    }
}

fn directive_phrases() -> Vec<String> {
    if let Ok(raw) = std::env::var("VESTIGE_DIRECTIVE_PHRASES")
        && let Ok(Value::Array(rows)) = serde_json::from_str::<Value>(&raw)
    {
        let custom: Vec<String> = rows
            .iter()
            .filter_map(Value::as_str)
            .filter(|p| !p.is_empty() && p.len() <= 120)
            .take(128)
            .map(str::to_string)
            .collect();
        if !custom.is_empty() {
            return custom;
        }
    }
    serde_json::from_str(DIRECTIVE_PHRASES_JSON).unwrap_or_default()
}

fn note_directives(g: &mut Global, inp: &Input) {
    if inp.prompt.is_empty() {
        return;
    }
    let s = g.session(&inp.session);
    for phrase in directive_phrases() {
        if contains_phrase(&inp.prompt, &phrase) && !s.ledger.pending.iter().any(|p| p == &phrase) {
            s.ledger.pending.push(phrase);
            s.ledger.pending_agent = inp.agent.clone();
        }
    }
}

fn directive_lines(g: &mut Global, inp: &Input) -> Vec<String> {
    let s = g.session(&inp.session);
    s.ledger
        .pending
        .iter()
        .filter(|p| !s.ledger.directives_told.iter().any(|t| t == *p))
        .map(|phrase| {
            format!(
                "Vestige ledger: an owner directive (\"{phrase}\") is unsaved this turn; Stop will block until remember(...) records it."
            )
        })
        .collect()
}

fn handle_lines(g: &mut Global, inp: &Input, storage: &Arc<Storage>) -> Vec<String> {
    let mut lines = Vec::new();
    let handles = prompt_handles(&inp.prompt);
    if handles.is_empty() {
        return lines;
    }
    let held = held_ids(g, &inp.session, &inp.agent);
    let s = g.session(&inp.session);
    for handle in handles {
        if s.ledger.handles_told.iter().any(|h| h == &handle) {
            continue;
        }
        let resolution = storage.resolve_handle(&handle);
        if resolution.ids.len() != 1 {
            continue;
        }
        let id = &resolution.ids[0];
        if held.contains(id) {
            s.ledger.handles_told.push(handle);
            continue;
        }
        lines.push(format!("Vestige ledger: {handle} is on the log as {id}."));
        s.ledger.handles_told.push(handle);
        let turn = s.turn;
        s.context(&inp.agent).hold(id, turn, "reflex", None);
        if lines.len() == 4 {
            break;
        }
    }
    lines
}

fn scope_lines(g: &mut Global, inp: &Input, storage: &Arc<Storage>) -> Vec<String> {
    let mut lines = Vec::new();
    let held = held_ids(g, &inp.session, &inp.agent);
    for scope in touched_scopes(inp) {
        let s = g.session(&inp.session);
        if s.ledger.scopes_seen.iter().any(|x| x == &scope) {
            continue;
        }
        s.ledger.scopes_seen.push(scope.clone());
        let tag = format!("scope:{scope}");
        let resolution = storage.resolve_handle(&tag);
        if resolution.ids.len() != 1 {
            continue;
        }
        let id = &resolution.ids[0];
        if held.contains(id) {
            continue;
        }
        lines.push(format!(
            "Vestige ledger: scope {scope} is on the log as {id}."
        ));
        let turn = s.turn;
        s.context(&inp.agent).hold(id, turn, "reflex", None);
        if lines.len() == 3 {
            break;
        }
    }
    lines
}

fn unverified_lines(g: &mut Global, inp: &Input, storage: &Arc<Storage>) -> Vec<String> {
    let held = held_ids(g, &inp.session, &inp.agent);
    let mut lines = Vec::new();
    for id in held.iter().take(8) {
        let s = g.session(&inp.session);
        if s.ledger.unverified_told.iter().any(|x| x == id) {
            continue;
        }
        if claim_state(storage, id) == Claim::Unverified {
            lines.push(format!(
                "Vestige ledger: decision {id} is an unverified claim; no execution receipt is on the log."
            ));
            s.ledger.unverified_told.push(id.clone());
        }
    }
    lines
}

fn resume_lines(g: &mut Global, inp: &Input) -> Vec<String> {
    let s = g.session(&inp.session);
    let Some(snap) = s.ledger.stop_failure.clone() else {
        return Vec::new();
    };
    if snap.told {
        return Vec::new();
    }
    s.ledger.stop_failure.as_mut().unwrap().told = true;
    let files = if snap.files.is_empty() {
        "no".to_string()
    } else {
        snap.files.len().to_string()
    };
    vec![format!(
        "Vestige ledger: turn {} ended on {} before Stop; {files} touched files are in the session snapshot.",
        snap.turn, snap.error
    )]
}

fn subagent_lines(g: &mut Global) -> Vec<String> {
    let rules: Vec<&str> =
        g.p.rules
            .iter()
            .rev()
            .take(6)
            .map(|(_, id)| id.as_str())
            .collect();
    let session_ids: Vec<String> = g.sessions.keys().cloned().collect();
    let mut debts = Vec::new();
    let mut intents = Vec::new();
    for id in &session_ids {
        if let Some(s) = g.sessions.get(id) {
            debts.extend(s.ledger.pending.iter().cloned());
            intents.extend(s.ledger.intentions.iter().map(|i| i.id.clone()));
        }
    }
    if rules.is_empty() && debts.is_empty() && intents.is_empty() {
        return Vec::new();
    }
    let line = format!(
        "Vestige ledger: parent rulebook {}; open debts {}; intentions {}.",
        list_or_none(&rules.iter().map(|s| (*s).to_string()).collect::<Vec<_>>()),
        list_or_none(&debts),
        list_or_none(&intents)
    );
    vec![trim_chars(&line, 1500)]
}

fn note_success(g: &mut Global, inp: &Input) {
    let agent = inp.agent.clone();
    let s = g.session(&inp.session);
    let live: HashSet<String> = s
        .open_failures
        .iter()
        .filter(|o| o.agent == agent)
        .map(|o| o.sig.clone())
        .collect();
    s.ledger
        .unpaired
        .retain(|u| u.agent != agent || live.contains(&u.sig));
    if is_prove(inp) {
        s.ledger.unpaired.retain(|u| u.agent != agent);
    }
    if is_edit_tool(&inp.tool)
        && !inp.file.is_empty()
        && !s.ledger.edited.iter().any(|f| f == &inp.file)
    {
        s.ledger.edited.push(inp.file.clone());
    }
    if pairing::is_shell(&inp.tool)
        && is_git_commit(&inp.command)
        && let Some(sha) = pairing::git_head(&inp.cwd)
        && !s
            .ledger
            .unreceipted
            .iter()
            .any(|x| x.eq_ignore_ascii_case(&sha))
    {
        // The ledger keeps the full SHA git reports, so a receipt and the
        // `commit:<sha>` tag match exactly.
        let mut sha = sha;
        if let Some(facts) = crate::commit_receipt::read(Path::new(&inp.cwd), &sha) {
            sha = facts.sha.clone();
            s.ledger.commits.retain(|c| c.sha != facts.sha);
            s.ledger.commits.push(facts);
        }
        if !s
            .ledger
            .unreceipted
            .iter()
            .any(|x| x.eq_ignore_ascii_case(&sha))
        {
            s.ledger.unreceipted.push(sha);
        }
    }
    if is_memory_write(inp) {
        let blob = format!("{}{}", inp.tool_response, inp.tool_input);
        s.ledger.pending.clear();
        settle_receipts(&mut s.ledger, &blob);
        let edited = s.ledger.edited.clone();
        for path in edited {
            if !s.ledger.saved.iter().any(|p| p == &path) {
                s.ledger.saved.push(path);
            }
        }
    }
}

/// A memory write that names a commit's SHA settles that commit only when it
/// also names every fact git reports (parents, branch, author, files) and
/// every gate this turn ran. Otherwise the gap is recorded for the reason.
fn settle_receipts(ledger: &mut TurnLedger, blob: &str) {
    let lower = blob.to_ascii_lowercase();
    let shas = ledger.unreceipted.clone();
    for sha in shas {
        if !lower.contains(&sha.to_ascii_lowercase()) {
            continue;
        }
        let missing = ledger
            .commits
            .iter()
            .find(|c| c.sha.eq_ignore_ascii_case(&sha))
            .map(|c| c.missing_in(blob, &ledger.gates))
            .unwrap_or_default();
        ledger
            .receipt_gaps
            .retain(|(s, _)| !s.eq_ignore_ascii_case(&sha));
        if missing.is_empty() {
            ledger.unreceipted.retain(|x| !x.eq_ignore_ascii_case(&sha));
        } else {
            ledger.receipt_gaps.push((sha, missing));
        }
    }
}

/// An allowlisted witness command that passed is a gate this turn ran.
fn note_gate(g: &mut Global, inp: &Input, storage: &Arc<Storage>) {
    let command = inp.command.trim();
    if command.is_empty() || inp.cwd.is_empty() || !pairing::is_shell(&inp.tool) {
        return;
    }
    let Some(root) = witness::repo_root(Path::new(&inp.cwd)) else {
        return;
    };
    if !witness::config_for(storage.data_dir(), Some(&root)).is_allowed(command) {
        return;
    }
    let s = g.session(&inp.session);
    if !s.ledger.gates.iter().any(|x| x == command) {
        s.ledger.gates.push(command.to_string());
    }
}

/// A receipt the host did not echo back (an MCP response the hook could not
/// read) still counts when the store holds it: an agent receipt tagged
/// `commit:<sha>` that is not the machine receipt.
fn note_store_receipts(g: &mut Global, inp: &Input, storage: &Arc<Storage>) {
    if !is_memory_write_tool(inp) {
        return;
    }
    settle_from_store(g, &inp.session, storage);
}

fn settle_from_store(g: &mut Global, session: &str, storage: &Arc<Storage>) {
    let shas = g.session(session).ledger.unreceipted.clone();
    for sha in shas {
        let ids = storage.resolve_handle(&format!("commit:{sha}")).ids;
        let mut texts = Vec::new();
        for id in ids.iter().take(16) {
            if let Some(node) = storage.get_node(id).ok().flatten()
                && !node
                    .tags
                    .iter()
                    .any(|t| t == crate::commit_receipt::AUTO_TAG)
            {
                texts.push(node.content.clone());
            }
        }
        if texts.is_empty() {
            continue;
        }
        let s = g.session(session);
        for text in texts {
            settle_receipts(&mut s.ledger, &format!("{sha} {text}"));
        }
    }
}

fn note_failure(g: &mut Global, inp: &Input) {
    let agent = inp.agent.clone();
    let turn = g.session(&inp.session).turn;
    let exit = parse_exit(&inp.error).unwrap_or(1);
    let command = if inp.command.is_empty() {
        String::new()
    } else {
        pairing::redact(&inp.command)
    };
    let fresh: Vec<Unpaired> = {
        let s = g.session(&inp.session);
        s.open_failures
            .iter()
            .filter(|o| o.agent == agent)
            .filter(|o| {
                !s.ledger
                    .unpaired
                    .iter()
                    .any(|u| u.sig == o.sig && u.agent == agent)
            })
            .map(|o| Unpaired {
                sig: o.sig.clone(),
                agent: agent.clone(),
                command: if command.is_empty() {
                    o.words.clone()
                } else {
                    command.clone()
                },
                exit_code: exit,
                turn,
            })
            .collect()
    };
    g.session(&inp.session).ledger.unpaired.extend(fresh);
}

fn note_stop_failure(g: &mut Global, inp: &Input) {
    let error = if STOP_ERRORS.contains(&inp.error.as_str()) {
        inp.error.clone()
    } else {
        "unknown".to_string()
    };
    let s = g.session(&inp.session);
    let files = s.ledger.edited.clone();
    let turn = s.turn;
    s.ledger.stop_failure = Some(StopSnap {
        error,
        files,
        turn,
        told: false,
    });
}

fn stop_decision(g: &mut Global, inp: &Input, storage: &Arc<Storage>, d: &mut HostDecision) {
    let reasons = debts(g, inp, storage);
    let blocks = g.session(&inp.session).ledger.stop_blocks;
    let (reason, next, overdue) = apply_valve(blocks, &reasons);
    g.session(&inp.session).ledger.stop_blocks = next;
    if overdue {
        d.overdue_item = Some(overdue_item(inp, &reasons, next));
    }
    d.block_reason = reason;
}

fn pre_tool_decision(g: &mut Global, inp: &Input, storage: &Arc<Storage>, d: &mut HostDecision) {
    if pairing::is_shell(&inp.tool) && is_git_commit(&inp.command) {
        let missing = unsaved(g, &inp.session);
        if !missing.is_empty() {
            let reason = format!(
                "Vestige ledger: files [{}] have no remember(...) receipt this turn.",
                missing.join(", ")
            );
            d.permission_decision = Some("deny".into());
            d.permission_reason = Some(reason);
            return;
        }
    }
    if let Some((updated, note)) = swap_replaced(inp, storage) {
        d.updated_input = Some(updated);
        d.permission_decision = Some("allow".into());
        d.permission_reason = Some(note);
    }
}

fn permit_decision(g: &mut Global, inp: &Input, d: &mut HostDecision) {
    let command = normalize_command(&inp.command);
    if command.is_empty() {
        return;
    }
    let now = Utc::now().timestamp();
    let session = inp.session.clone();
    let pos = g.p.permits.iter().position(|p| {
        !p.consumed && p.session == session && p.command == command && p.expires > now
    });
    let Some(pos) = pos else {
        return;
    };
    let id = g.p.permits[pos].id.clone();
    g.p.permits[pos].consumed = true;
    g.dirty = true;
    d.permit_behavior = Some("allow".into());
    // behavior allow only. updatedPermissions is never set, so the permit
    // cannot become a standing permission rule.
    d.permit_message = Some(format!(
        "Vestige ledger: one-use permit {id} covered this exact command once."
    ));
}

fn model_decision(g: &mut Global, inp: &Input, d: &mut HostDecision) {
    let (Some(from), Some(to)) = (model_rank(&inp.from_model), model_rank(&inp.to_model)) else {
        return;
    };
    if to >= from {
        return;
    }
    let s = g.session(&inp.session);
    let Some(hit) = s
        .ledger
        .intentions
        .iter()
        .find(|i| i.tags.iter().any(|t| SENSITIVE_TAGS.contains(&t.as_str())))
    else {
        return;
    };
    let tag = hit
        .tags
        .iter()
        .find(|t| SENSITIVE_TAGS.contains(&t.as_str()))
        .cloned()
        .unwrap_or_default();
    let reason = format!(
        "Vestige ledger: switching from {} to {} is a downgrade while intention {} tagged {tag} is open.",
        inp.from_model, inp.to_model, hit.id
    );
    d.block_reason = Some(reason.clone());
    d.permission_decision = Some("deny".into());
    d.permission_reason = Some(reason);
}

fn worktree_decision(inp: &Input, d: &mut HostDecision) {
    let Some(reason) = dirty_worktree(&inp.worktree_path) else {
        return;
    };
    d.block_reason = Some(reason);
    // Verified: a WorktreeRemove command hook's JSON is discarded. Non-zero
    // exit fails removal when the directory still exists. MCP mapping of this
    // exitCode is unverified.
    d.exit_code = Some(2);
}

fn batch_decision(inp: &Input, d: &mut HostDecision) {
    let Some(path) = colliding_path(&inp.tool_calls) else {
        return;
    };
    d.block_reason = Some(format!("Vestige ledger: parallel edits collide on {path}."));
}

fn mark_delivered(g: &mut Global, inp: &Input, delivered: &str) {
    if delivered.is_empty() {
        return;
    }
    let s = g.session(&inp.session);
    let pending = s.ledger.pending.clone();
    for phrase in pending {
        let line = format!(
            "Vestige ledger: an owner directive (\"{phrase}\") is unsaved this turn; Stop will block until remember(...) records it."
        );
        if delivered.contains(&line) && !s.ledger.directives_told.iter().any(|t| t == &phrase) {
            s.ledger.directives_told.push(phrase);
        }
    }
}

fn debts(g: &mut Global, inp: &Input, storage: &Arc<Storage>) -> Vec<String> {
    let mut reasons = Vec::new();
    let agent = inp.agent.clone();
    let turn = g.session(&inp.session).turn;
    let pending = g.session(&inp.session).ledger.pending.clone();
    let pending_agent = g.session(&inp.session).ledger.pending_agent.clone();
    if pending_agent == agent {
        for phrase in &pending {
            reasons.push(format!(
                "Turn blocked: owner directive \"{phrase}\" not saved. Call remember(...)."
            ));
        }
    }
    let unpaired: Vec<Unpaired> = g
        .session(&inp.session)
        .ledger
        .unpaired
        .iter()
        .filter(|u| u.agent == agent && u.turn == turn)
        .cloned()
        .collect();
    for u in unpaired {
        reasons.push(format!(
            "Turn blocked: {} exited {} and has no passing rerun.",
            if u.command.is_empty() {
                u.sig
            } else {
                u.command
            },
            u.exit_code
        ));
    }
    settle_from_store(g, &inp.session, storage);
    let shas = g.session(&inp.session).ledger.unreceipted.clone();
    let gaps = g.session(&inp.session).ledger.receipt_gaps.clone();
    for sha in shas {
        match gaps.iter().find(|(s, _)| s.eq_ignore_ascii_case(&sha)) {
            Some((_, missing)) => reasons.push(format!(
                "Turn blocked: the receipt for git commit {sha} leaves out {}. Call remember(...) with them.",
                missing.join(", ")
            )),
            None => reasons.push(format!(
                "Turn blocked: git commit {sha} has no receipt naming that SHA (RULE 6: SHA, parent, branch, author, files with git diff -U0 ranges, gate output)."
            )),
        }
    }
    reasons.extend(citation_reasons(storage, g, inp));
    reasons
}

/// Two blocks, then the turn may end. The count stays at 2.
pub(super) fn apply_valve(blocks: u32, reasons: &[String]) -> (Option<String>, u32, bool) {
    if reasons.is_empty() {
        return (None, blocks, false);
    }
    if blocks >= STOP_BLOCK_CAP {
        return (None, blocks, true);
    }
    (Some(reasons.join(" ")), blocks + 1, false)
}

fn overdue_item(inp: &Input, reasons: &[String], blocks: u32) -> Value {
    let summary = trim_chars(&reasons.join(" "), 500);
    json!({
        "node_type": "intention",
        "tags": ["OVERDUE", "intention"],
        "content": format!(
            "OVERDUE intention. Session {} turn ended after {blocks} Stop blocks with unpaid ledger debts: {summary}. This record is not an execution receipt.",
            inp.session.chars().take(16).collect::<String>()
        ),
    })
}

fn citation_reasons(storage: &Arc<Storage>, g: &Global, inp: &Input) -> Vec<String> {
    let text = &inp.last_assistant_message;
    if text.is_empty() {
        return Vec::new();
    }
    let mut reasons = Vec::new();
    let held = held_ids(g, &inp.session, &inp.agent);
    for handle in memory_handles(text) {
        reasons.extend(memory_reason(storage, &handle, &held));
        if reasons.len() >= 8 {
            break;
        }
    }
    for sha in commit_handles(text) {
        if !commit_exists(&inp.cwd, &sha) {
            reasons.push(format!(
                "Turn blocked: commit {sha} is not in this repository."
            ));
        }
    }
    for path in file_handles(text) {
        if !cited_file_exists(&inp.cwd, &path) {
            reasons.push(format!("Turn blocked: file {path} is not on disk."));
        }
    }
    for pr in pr_handles(text) {
        if !pr_known(storage, &inp.cwd, &pr) {
            reasons.push(format!(
                "Turn blocked: PR #{pr} has no commit or receipt in this repository."
            ));
        }
    }
    reasons.truncate(8);
    reasons
}

fn memory_reason(storage: &Arc<Storage>, handle: &str, held: &HashSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(successor) = successor_of(storage, handle) {
        let named = if successor.is_empty() {
            "a successor".to_string()
        } else {
            successor
        };
        out.push(format!("Turn blocked: {handle} was replaced by {named}."));
        return out;
    }
    let resolution = storage.resolve_handle(handle);
    if resolution.ids.len() != 1 {
        out.push(format!("Turn blocked: {handle} is missing."));
        return out;
    }
    let id = &resolution.ids[0];
    if !held.contains(id) && !held.contains(handle) {
        out.push(format!("Turn blocked: {handle} was not read this session."));
    }
    if claim_state(storage, id) == Claim::Unverified {
        out.push(format!(
            "Turn blocked: decision {id} has no execution receipt; it is an unverified claim."
        ));
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Claim {
    NotADecision,
    Verified,
    Unverified,
}

fn claim_state(storage: &Arc<Storage>, id: &str) -> Claim {
    let Some(node) = storage.get_node(id).ok().flatten() else {
        return Claim::NotADecision;
    };
    let decision = node.node_type == "decision" || node.tags.iter().any(|t| t == "decision");
    if !decision {
        return Claim::NotADecision;
    }
    if has_execution_receipt(storage, &node) {
        Claim::Verified
    } else {
        Claim::Unverified
    }
}

/// A creation receipt is not an execution receipt. The check reads tags and
/// edges, never the memory's prose.
fn has_execution_receipt(storage: &Arc<Storage>, node: &vestige_core::KnowledgeNode) -> bool {
    for tag in &node.tags {
        if tag == "reflex-witness" || tag.starts_with("tree-") {
            return true;
        }
        if let Some(rest) = tag.strip_prefix("execution-receipt:") {
            let rest = rest.trim();
            if !rest.is_empty()
                && (storage.get_node(rest).ok().flatten().is_some()
                    || storage.resolve_handle(rest).ids.len() == 1)
            {
                return true;
            }
        }
    }
    let Ok(edges) = storage.get_connections_for_memory(&node.id) else {
        return false;
    };
    for edge in edges {
        if !matches!(
            edge.link_type.as_str(),
            "closes" | "derived_from" | "evidence_of"
        ) {
            continue;
        }
        let other = if edge.source_id == node.id {
            edge.target_id.as_str()
        } else {
            edge.source_id.as_str()
        };
        let Some(peer) = storage.get_node(other).ok().flatten() else {
            continue;
        };
        if peer.tags.iter().any(|t| {
            t == "reflex-witness" || t.starts_with("tree-") || t.starts_with("op-receipt-")
        }) {
            return true;
        }
    }
    false
}

fn successor_of(storage: &Arc<Storage>, handle: &str) -> Option<String> {
    let superseded = storage.superseded_node_ids().ok()?;
    let id = if superseded.contains(handle) {
        handle.to_string()
    } else {
        let resolution = storage.resolve_handle(handle);
        let candidate = resolution.ids.first()?.clone();
        if !superseded.contains(&candidate) {
            return None;
        }
        candidate
    };
    let edges = storage.get_connections_for_memory(&id).unwrap_or_default();
    let named = edges.into_iter().find_map(|edge| {
        if edge.link_type == "supersedes" && edge.target_id == id {
            Some(edge.source_id)
        } else {
            None
        }
    });
    Some(named.unwrap_or_default())
}

fn unsaved(g: &Global, session: &str) -> Vec<String> {
    g.sessions
        .get(session)
        .map(|s| {
            s.ledger
                .edited
                .iter()
                .filter(|p| !s.ledger.saved.iter().any(|x| x == *p))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn held_ids(g: &Global, session: &str, agent: &str) -> HashSet<String> {
    g.sessions
        .get(session)
        .and_then(|s| s.contexts.get(agent))
        .map(|c| c.ids().into_iter().collect())
        .unwrap_or_default()
}

fn swap_replaced(inp: &Input, storage: &Arc<Storage>) -> Option<(Value, String)> {
    let leaf = tool_leaf(&inp.tool);
    if !matches!(leaf, "remember" | "curate" | "memory" | "smart_ingest") {
        return None;
    }
    if !inp.tool_input.is_object() {
        return None;
    }
    let mut updated = inp.tool_input.clone();
    let blob = updated.to_string();
    let mut swapped = Vec::new();
    for handle in memory_handles(&blob) {
        let Some(successor) = successor_of(storage, &handle) else {
            continue;
        };
        if successor.is_empty() || successor == handle {
            continue;
        }
        let next = updated.to_string().replace(&handle, &successor);
        if let Ok(parsed) = serde_json::from_str::<Value>(&next) {
            updated = parsed;
            swapped.push(format!("{handle} -> {successor}"));
        }
    }
    if swapped.is_empty() {
        return None;
    }
    Some((
        updated,
        format!(
            "Vestige ledger: replaced id swapped for its successor ({}).",
            swapped.join(", ")
        ),
    ))
}

fn is_memory_write_tool(inp: &Input) -> bool {
    matches!(tool_leaf(&inp.tool), "remember" | "curate")
}

fn is_memory_write(inp: &Input) -> bool {
    let leaf = tool_leaf(&inp.tool);
    if !matches!(leaf, "remember" | "curate") {
        return false;
    }
    let action = inp
        .tool_input
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("save");
    if matches!(action, "recall" | "search" | "get") {
        return false;
    }
    receipt_present(&inp.tool_response)
}

fn receipt_present(v: &Value) -> bool {
    if v.get("nodeId")
        .and_then(Value::as_str)
        .is_some_and(is_memory_id)
    {
        return true;
    }
    v.get("results")
        .and_then(Value::as_array)
        .is_some_and(|rows| {
            rows.iter().any(|row| {
                row.get("nodeId")
                    .and_then(Value::as_str)
                    .is_some_and(is_memory_id)
            })
        })
}

fn is_prove(inp: &Input) -> bool {
    let leaf = tool_leaf(&inp.tool);
    let action = inp
        .tool_input
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("");
    (leaf == "causal_walk" || leaf == "prove") && action == "prove"
}

fn is_edit_tool(tool: &str) -> bool {
    matches!(
        tool_leaf(tool),
        "Edit" | "Write" | "MultiEdit" | "NotebookEdit"
    )
}

fn tool_leaf(tool: &str) -> &str {
    tool.rsplit("__").next().unwrap_or(tool)
}

fn is_git_commit(command: &str) -> bool {
    let flat = command.replace("&&", "\n").replace("||", "\n");
    for part in flat.split([';', '\n']) {
        let words: Vec<&str> = part
            .split_whitespace()
            .filter(|w| !(w.contains('=') && !w.starts_with('-')))
            .collect();
        if words.first().copied() == Some("git") && words.contains(&"commit") {
            return true;
        }
    }
    false
}

fn normalize_command(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parse_exit(error: &str) -> Option<i64> {
    for line in error.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Exit code ") {
            return rest.trim().parse().ok();
        }
    }
    None
}

fn contains_phrase(hay: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let h: Vec<char> = hay.chars().flat_map(|c| c.to_lowercase()).collect();
    let n: Vec<char> = needle.chars().flat_map(|c| c.to_lowercase()).collect();
    if n.len() > h.len() {
        return false;
    }
    for i in 0..=h.len() - n.len() {
        if h[i..i + n.len()] != n[..] {
            continue;
        }
        let before = i == 0 || !is_token_char(h[i - 1]);
        let after_at = i + n.len();
        let after = after_at >= h.len() || !is_token_char(h[after_at]);
        if before && after {
            return true;
        }
    }
    false
}

fn is_token_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn prompt_handles(prompt: &str) -> Vec<String> {
    let mut out = memory_handles(prompt);
    for token in prompt.split_whitespace() {
        let token = token.trim_matches(|c: char| {
            matches!(c, '"' | '\'' | '`' | '(' | ')' | ',' | '.' | ':' | ';')
        });
        if token.contains('/') && !token.contains("://") && token.len() <= 240 {
            push_unique(&mut out, token);
        }
    }
    out.truncate(MAX_CITATIONS);
    out
}

fn touched_scopes(inp: &Input) -> Vec<String> {
    let mut out = Vec::new();
    if !inp.file.is_empty()
        && let Some(parent) = Path::new(&inp.file).parent().and_then(|p| p.to_str())
        && !parent.is_empty()
        && parent != "/"
    {
        push_unique(&mut out, parent);
    }
    for token in inp.command.split_whitespace() {
        let token = token.trim_matches(|c: char| matches!(c, '"' | '\'' | '`'));
        if token.contains('/')
            && !token.starts_with('-')
            && !token.contains("://")
            && token.len() <= 240
        {
            push_unique(&mut out, token);
        }
    }
    out
}

fn memory_handles(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if let Some((handle, len)) = take_prefixed(text, i, "mem-", 16, 16)
            .or_else(|| take_prefixed(text, i, "op-receipt-", 16, 16))
            .or_else(|| take_prefixed(text, i, "op-rule-", 1, 64))
            .or_else(|| take_prefixed(text, i, "eff-", 8, 64))
        {
            push_unique(&mut out, &handle);
            i += len.max(1);
            continue;
        }
        if let Some(len) = uuid_at(text, i) {
            push_unique(&mut out, &text[i..i + len]);
            i += len.max(1);
            continue;
        }
        i += text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
    }
    out.truncate(MAX_CITATIONS);
    out
}

fn take_prefixed(
    s: &str,
    i: usize,
    prefix: &str,
    min: usize,
    max: usize,
) -> Option<(String, usize)> {
    if !s[i..].starts_with(prefix) {
        return None;
    }
    if i > 0 {
        let before = s[..i].chars().next_back()?;
        if is_token_char(before) {
            return None;
        }
    }
    let rest = &s[i + prefix.len()..];
    let take = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '-')
        .count();
    let body: String = rest.chars().take(take).collect();
    if prefix == "mem-" || prefix == "op-receipt-" || prefix == "eff-" {
        if body.len() < min || body.len() > max || !is_hex(&body) {
            return None;
        }
    } else if body.len() < min || body.len() > max || !is_rule(&body) {
        return None;
    }
    let handle = format!("{prefix}{body}");
    Some((handle.clone(), prefix.len() + body.len()))
}

fn uuid_at(s: &str, i: usize) -> Option<usize> {
    let rest = &s[i..];
    if rest.len() < 36 {
        return None;
    }
    let candidate = &rest[..36];
    if is_memory_id(candidate) && !candidate.starts_with("mem-") {
        if i > 0 && is_token_char(s[..i].chars().next_back()?) {
            return None;
        }
        return Some(36);
    }
    None
}

fn commit_handles(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut i = 0;
    while i + 8 < bytes.len() {
        if lower[i..].starts_with("commit:")
            && let Some(sha) = hex_run(&lower[i + "commit:".len()..], 7, 40)
        {
            push_unique(&mut out, &sha);
            i += "commit:".len() + sha.len();
            continue;
        }
        i += 1;
    }
    for inner in backticks(text) {
        if is_hex(&inner) && (7..=40).contains(&inner.len()) {
            push_unique(&mut out, &inner.to_ascii_lowercase());
        }
    }
    // A bare 40-hex word is a full SHA. Shorter bare hex is left alone so a
    // turn is not blocked on an ordinary token.
    let mut word = String::new();
    for c in text.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_hexdigit() {
            word.push(c);
        } else {
            if word.len() == 40 {
                push_unique(&mut out, &word.to_ascii_lowercase());
            }
            word.clear();
        }
    }
    out.truncate(MAX_CITATIONS);
    out
}

fn file_handles(text: &str) -> Vec<String> {
    backticks(text)
        .into_iter()
        .filter(|inner| {
            inner.contains('/')
                && !inner.contains(' ')
                && !inner.contains("://")
                && !inner.starts_with('$')
                && inner.len() <= 240
        })
        .take(MAX_CITATIONS)
        .collect()
}

fn pr_handles(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = text.to_ascii_lowercase();
    for marker in ["pr #", "pr#", "pull request #", "pull/"] {
        let mut rest = lower.as_str();
        while let Some(at) = rest.find(marker) {
            let after = &rest[at + marker.len()..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() && digits.len() <= 8 {
                push_unique(&mut out, &digits);
            }
            rest = &rest[at + marker.len()..];
        }
    }
    out
}

fn backticks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find('`') {
        rest = &rest[i + 1..];
        let Some(j) = rest.find('`') else {
            break;
        };
        let inner = &rest[..j];
        if !inner.is_empty() && !inner.contains('\n') && inner.len() <= 300 {
            out.push(inner.to_string());
        }
        rest = &rest[j + 1..];
    }
    out
}

fn hex_run(s: &str, min: usize, max: usize) -> Option<String> {
    let run: String = s.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    if !(min..=max).contains(&run.len()) {
        return None;
    }
    if s.len() == run.len() {
        return Some(run);
    }
    let next = s[run.len()..].chars().next()?;
    if is_token_char(next) { None } else { Some(run) }
}

fn push_unique(out: &mut Vec<String>, value: &str) {
    if !value.is_empty() && !out.iter().any(|v| v == value) {
        out.push(value.to_string());
    }
}

fn commit_exists(cwd: &str, sha: &str) -> bool {
    if cwd.is_empty() || !is_hex(sha) || !(7..=40).contains(&sha.len()) {
        return false;
    }
    std::process::Command::new("git")
        .args(["-C", cwd, "rev-parse", "--verify", "--quiet"])
        .arg(format!("{sha}^{{commit}}"))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn cited_file_exists(cwd: &str, path: &str) -> bool {
    if path.contains('\0') || cwd.is_empty() {
        return false;
    }
    let p = Path::new(path);
    let full = if p.is_absolute() {
        p.to_path_buf()
    } else {
        Path::new(cwd).join(p)
    };
    full.is_file()
}

fn pr_known(storage: &Arc<Storage>, cwd: &str, n: &str) -> bool {
    let tag = format!("pr-{n}");
    if storage.resolve_handle(&tag).ids.len() == 1 {
        return true;
    }
    if cwd.is_empty() {
        return false;
    }
    let needle = format!("#{n}");
    std::process::Command::new("git")
        .args(["-C", cwd, "log", "-1", "--fixed-string", "--grep"])
        .arg(&needle)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn dirty_worktree(path: &str) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let p = Path::new(path);
    if !p.is_dir() {
        return None;
    }
    let output = std::process::Command::new("git")
        .args(["-C", path, "status", "--porcelain"])
        .output()
        .ok();
    match output {
        Some(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            if text.trim().is_empty() {
                None
            } else {
                Some(format!(
                    "Vestige ledger: worktree {path} has uncommitted changes."
                ))
            }
        }
        _ => {
            if dir_has_file(p) {
                Some(format!(
                    "Vestige ledger: worktree {path} holds files git does not track as a checkout."
                ))
            } else {
                None
            }
        }
    }
}

fn dir_has_file(dir: &Path) -> bool {
    let mut stack = vec![dir.to_path_buf()];
    let mut seen = 0;
    while let Some(next) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in rd.flatten() {
            seen += 1;
            if seen > 200 {
                return true;
            }
            let path = entry.path();
            if path.is_file() {
                return true;
            }
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
    false
}

#[derive(Clone, Copy)]
struct Span {
    start: Option<i64>,
    end: Option<i64>,
}

fn colliding_path(tool_calls: &Value) -> Option<String> {
    let rows = tool_calls.as_array()?;
    let mut edits: Vec<(String, Span)> = Vec::new();
    for row in rows {
        let name = row
            .get("tool_name")
            .or_else(|| row.get("tool"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if !is_edit_tool(name) {
            continue;
        }
        let input = row.get("tool_input").cloned().unwrap_or(Value::Null);
        let path = input
            .get("file_path")
            .or_else(|| input.get("path"))
            .or_else(|| input.get("notebook_path"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if path.is_empty() {
            continue;
        }
        let start = input.get("start_line").and_then(Value::as_i64);
        let end = input.get("end_line").and_then(Value::as_i64);
        edits.push((path, Span { start, end }));
    }
    for i in 0..edits.len() {
        for j in i + 1..edits.len() {
            if edits[i].0 == edits[j].0 && spans_overlap(edits[i].1, edits[j].1) {
                return Some(edits[i].0.clone());
            }
        }
    }
    None
}

fn spans_overlap(a: Span, b: Span) -> bool {
    let (Some(a0), Some(a1)) = (a.start, a.end) else {
        return true;
    };
    let (Some(b0), Some(b1)) = (b.start, b.end) else {
        return true;
    };
    let (a0, a1) = if a0 <= a1 { (a0, a1) } else { (a1, a0) };
    let (b0, b1) = if b0 <= b1 { (b0, b1) } else { (b1, b0) };
    a0 <= b1 && b0 <= a1
}

pub(super) fn model_rank(model: &str) -> Option<u8> {
    let l = model.to_ascii_lowercase();
    if l.contains("opus") {
        Some(3)
    } else if l.contains("sonnet") {
        Some(2)
    } else if l.contains("haiku") {
        Some(1)
    } else {
        None
    }
}

fn list_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "none".into()
    } else {
        items.join(", ")
    }
}

fn trim_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directive_phrases_match_tokens_not_prefixes() {
        assert!(contains_phrase(
            "from now on never edit that",
            "from now on"
        ));
        assert!(contains_phrase("from now on never edit that", "never"));
        assert!(!contains_phrase("nevertheless ship it", "never"));
        assert!(contains_phrase("rule: do not touch the floor", "rule:"));
        assert!(!contains_phrase("ruling: something", "rule:"));
    }

    #[test]
    fn additional_context_stays_under_the_host_cap() {
        let long = "a".repeat(12_000);
        let capped = cap_context(&long);
        assert!(
            capped.chars().count() <= CONTEXT_CAP,
            "{}",
            capped.chars().count()
        );
        assert!(capped.contains("truncated to 10000"));
        assert_eq!(cap_context("short"), "short");
    }

    #[test]
    fn the_third_stop_is_allowed_and_the_count_stays_at_two() {
        let reasons =
            vec!["Turn blocked: owner directive \"never\" not saved. Call remember(...).".into()];
        let (r1, n1, o1) = apply_valve(0, &reasons);
        assert!(r1.is_some() && !o1 && n1 == 1);
        let (r2, n2, o2) = apply_valve(n1, &reasons);
        assert!(r2.is_some() && !o2 && n2 == 2);
        let (r3, n3, o3) = apply_valve(n2, &reasons);
        assert!(r3.is_none() && o3 && n3 == 2);
        let (clean, n4, o4) = apply_valve(2, &[]);
        assert!(clean.is_none() && !o4 && n4 == 2);
    }

    #[test]
    fn pre_tool_use_deny_is_permission_decision_not_top_level_block() {
        let d = HostDecision {
            permission_decision: Some("deny".into()),
            permission_reason: Some(
                "Vestige ledger: files [a.rs] have no remember(...) receipt this turn.".into(),
            ),
            ..HostDecision::default()
        };
        let out = host_json("PreToolUse", "claude", "", &d);
        assert!(out.get("decision").is_none(), "{out}");
        assert_eq!(out["hookSpecificOutput"]["permissionDecision"], "deny");
        assert_eq!(out["hookSpecificOutput"]["hookEventName"], "PreToolUse");
    }

    #[test]
    fn permission_request_allow_never_writes_persistent_rules() {
        let d = HostDecision {
            permit_behavior: Some("allow".into()),
            ..HostDecision::default()
        };
        let out = host_json("PermissionRequest", "claude", "", &d);
        let rendered = out.to_string();
        assert_eq!(
            out["hookSpecificOutput"]["decision"]["behavior"], "allow",
            "{out}"
        );
        assert!(!rendered.contains("updatedPermissions"), "{out}");
        assert!(out.get("decision").is_none(), "{out}");
    }

    #[test]
    fn context_is_nested_and_pre_model_switch_omits_it() {
        let out = host_json(
            "UserPromptSubmit",
            "claude",
            "Vestige ledger: noted.",
            &HostDecision::default(),
        );
        assert_eq!(
            out["hookSpecificOutput"]["additionalContext"],
            "Vestige ledger: noted."
        );
        assert!(out.get("additionalContext").is_none(), "{out}");
        let blocked = HostDecision {
            block_reason: Some("Vestige ledger: switching is a downgrade.".into()),
            permission_decision: Some("deny".into()),
            permission_reason: Some("Vestige ledger: switching is a downgrade.".into()),
            ..HostDecision::default()
        };
        let model = host_json(
            "PreModelSwitch",
            "claude",
            "should not ride along",
            &blocked,
        );
        assert_eq!(model["decision"], "block");
        assert!(
            model["hookSpecificOutput"]
                .get("additionalContext")
                .is_none(),
            "{model}"
        );
        assert_eq!(model["hookSpecificOutput"]["permissionDecision"], "deny");
    }

    #[test]
    fn parallel_edits_collide_only_when_ranges_overlap() {
        let same = json!({"tool_calls": [
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.rs"}},
            {"tool_name": "Write", "tool_input": {"file_path": "src/a.rs"}}
        ]});
        assert_eq!(
            colliding_path(&same["tool_calls"]).as_deref(),
            Some("src/a.rs")
        );
        let apart = json!([
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.rs", "start_line": 1, "end_line": 4}},
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.rs", "start_line": 10, "end_line": 12}}
        ]);
        assert!(colliding_path(&apart).is_none());
        let overlap = json!([
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.rs", "start_line": 1, "end_line": 8}},
            {"tool_name": "Edit", "tool_input": {"file_path": "src/a.rs", "start_line": 8, "end_line": 9}}
        ]);
        assert_eq!(colliding_path(&overlap).as_deref(), Some("src/a.rs"));
    }

    #[test]
    fn citation_scanner_names_handles_and_ignores_short_hex() {
        let text =
            "see mem-00000000000000aa and commit:deadbee and `src/a.rs` PR #12 plus deadbeef";
        assert!(memory_handles(text).contains(&"mem-00000000000000aa".to_string()));
        assert!(commit_handles(text).contains(&"deadbee".to_string()));
        assert!(!commit_handles(text).iter().any(|s| s == "deadbeef"));
        assert!(file_handles(text).contains(&"src/a.rs".to_string()));
        assert!(pr_handles(text).contains(&"12".to_string()));
    }

    #[test]
    fn opus_to_haiku_is_a_downgrade_and_an_unknown_model_is_not_ranked() {
        assert!(model_rank("claude-haiku-5") < model_rank("claude-opus-5"));
        assert!(model_rank("custom-gateway-model").is_none());
    }

    #[test]
    fn stop_block_carries_the_reason_and_nested_context() {
        let d = HostDecision {
            block_reason: Some(
                "Turn blocked: owner directive \"never\" not saved. Call remember(...).".into(),
            ),
            ..HostDecision::default()
        };
        let out = host_json(
            "Stop",
            "claude",
            "Vestige ledger: an owner directive (\"never\") is unsaved this turn; Stop will block until remember(...) records it.",
            &d,
        );
        assert_eq!(out["decision"], "block");
        assert!(out["reason"].as_str().unwrap().contains("never"), "{out}");
        assert!(
            out["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap()
                .starts_with("Vestige ledger:")
        );
        assert!(!out.to_string().contains("[REQUIRED]"));
    }
}
