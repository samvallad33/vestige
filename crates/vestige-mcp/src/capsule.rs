//! Session capsule (#395).
//!
//! A deterministic projection of a Strata log for one session boundary.
//! Selection is by exact handle, scope, node type, tag, anchor path and
//! deadline. Nothing here searches, ranks, or reads a wall clock. The same
//! head and the same inputs render the same bytes on the direct path and
//! through the running owner.
//!
//! `vestige capsule` fails open: no store, a legacy store, a dead owner,
//! malformed stdin or a refused source print nothing on stdout and exit 0.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::{Value, json};
use strata_store::effect_receipt_id;

use crate::rulebook;
use crate::strata_memory::{self, with_store};

/// Claude Code and Codex additionalContext cap, in characters.
pub const CONTEXT_CAP: usize = 10_000;

/// The command the installer writes. Matched byte for byte on scrub.
pub const COMMAND: &str = "vestige capsule --harness claude-code --event session-start";

/// Sources the harness fragment lists. Anything else is refused.
pub const SOURCES: [&str; 5] = ["startup", "resume", "clear", "compact", "fork"];

pub const MATCHER: &str = "startup|resume|clear|compact|fork";

/// Attach budget, under the hook timeout, so the harness never kills a
/// half-written stdout.
const ATTACH_BUDGET: Duration = Duration::from_secs(8);

const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

pub fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        out.push(HEX_DIGITS[(byte & 0xf) as usize] as char);
    }
    out
}

pub fn head_token(
    seq: u64,
    frame_hash: &[u8; 32],
    prev_segment: &[u8; 32],
    checkpoint: Option<[u8; 32]>,
) -> String {
    let segment = if *prev_segment == strata::GENESIS_PREV_SEGMENT_HASH {
        "unsealed".to_string()
    } else {
        hex(prev_segment)
    };
    let ckpt = checkpoint
        .map(|hash| hex(&hash))
        .unwrap_or_else(|| "none".into());
    format!("vst:head:{seq:016x}:{}:{segment}:{ckpt}", hex(frame_hash))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub source: String,
    pub mode: String,
    pub scope: String,
    pub codebase: Option<String>,
    pub branch: Option<String>,
    pub head_commit: Option<String>,
    pub changed_files: Vec<String>,
    pub since: Option<u64>,
}

impl Request {
    pub fn from_value(value: &Value) -> Self {
        let text = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        let codebase = text("codebase").or_else(|| {
            value
                .pointer("/context/codebase")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        });
        let changed = value
            .get("changed_files")
            .or_else(|| value.get("changedFiles"))
            .and_then(Value::as_array)
            .map(|rows| {
                let mut files: Vec<String> = rows
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                files.sort();
                files.dedup();
                files
            })
            .unwrap_or_default();
        let since = value.get("since").and_then(Value::as_u64);
        Self {
            source: text("source").unwrap_or_default(),
            mode: text("mode").unwrap_or_else(|| "normal".into()),
            scope: text("scope").unwrap_or_else(|| "user".into()),
            codebase,
            branch: text("branch"),
            head_commit: text("head_commit").or_else(|| text("headCommit")),
            changed_files: changed,
            since,
        }
    }

    fn full(&self) -> bool {
        matches!(self.source.as_str(), "startup" | "clear") || self.since.is_none()
    }
}

pub fn accepted_source(source: &str) -> bool {
    SOURCES.contains(&source)
}

/// `session_start` with `format: "capsule"`. The body is what the CLI adapter
/// wraps. A legacy store or a refused source yields an empty body and writes
/// nothing.
pub fn session_tool(
    storage: &vestige_core::Storage,
    args: Option<&Value>,
) -> Result<Value, String> {
    if !strata_memory::is_strata_backend(storage) {
        return Ok(json!({
            "format": "capsule",
            "body": "",
            "handles": [],
            "skipped": "legacy_store",
            "appended": false,
        }));
    }
    let empty = Value::Null;
    let request = Request::from_value(args.unwrap_or(&empty));
    if !accepted_source(&request.source) {
        return Ok(json!({
            "format": "capsule",
            "body": "",
            "handles": [],
            "skipped": "refused_source",
            "appended": false,
        }));
    }
    let before = with_store(storage, |store| store.log().head());
    let body = with_store(storage, |store| render(store, &request)).unwrap_or_default();
    let after = with_store(storage, |store| store.log().head());
    if before != after {
        return Err("capsule render appended a frame".into());
    }
    let handles = handles_in(&body);
    Ok(json!({
        "format": "capsule",
        "body": body,
        "handles": handles,
        "appended": false,
    }))
}

fn render(store: &strata_store::StrataStore, request: &Request) -> String {
    let head = store.log().head();
    let checkpoint = store
        .checkpoints()
        .last()
        .map(strata_kernel::checkpoint::checkpoint_hash);
    let token = head_token(
        head.last_acked_seq,
        &head.last_frame_hash,
        &head.prev_segment_hash,
        checkpoint,
    );
    let clock = store.head_clock_ms();
    let digest = hex(&store.state_digest());
    let mut created: BTreeMap<String, u64> = BTreeMap::new();
    for write in store.node_writes() {
        created
            .entry(write.record.id.clone())
            .or_insert(write.frame_seq);
    }
    let since = if request.full() { None } else { request.since };
    let keep = |id: &str| match since {
        None => true,
        Some(bound) => created.get(id).is_some_and(|seq| *seq > bound),
    };

    let mut owner: Vec<String> = Vec::new();
    let mut decisions: Vec<String> = Vec::new();
    let mut changed: Vec<String> = Vec::new();
    let mut retired: Vec<String> = Vec::new();
    for record in store.nodes() {
        if record.tags.iter().any(|tag| tag == rulebook::OWNER_TAG) && keep(&record.id) {
            owner.push(record.content.clone());
        }
        if !in_scope(&record.scope, &request.scope) {
            continue;
        }
        if !record.is_live() {
            if let Some(bound) = since
                && let Some(next) = record.superseded_by.as_deref()
            {
                let old_seq = created.get(&record.id).copied().unwrap_or(0);
                let new_seq = created.get(next).copied().unwrap_or(u64::MAX);
                if old_seq <= bound && new_seq > bound {
                    retired.push(format!("- `vst:{}` retired by `vst:{next}`", record.id));
                }
            }
            continue;
        }
        if record.valid_from_ms > clock || clock >= record.valid_until_ms {
            continue;
        }
        if record.node_type == "decision"
            && keep(&record.id)
            && codebase_ok(&record.tags, request.codebase.as_deref())
        {
            decisions.push(decision_line(store, &record));
        }
        if keep(&record.id) && !request.changed_files.is_empty() {
            for anchor in store.anchors_for(&record.id) {
                if request
                    .changed_files
                    .iter()
                    .any(|path| path == &anchor.file_path)
                {
                    changed.push(anchor_line(store, &record.id, &anchor));
                }
            }
        }
    }
    decisions.sort();
    changed.sort();
    retired.sort();
    owner.sort();

    let mut due: Vec<String> = Vec::new();
    if since.is_none() {
        for intention in store.intentions() {
            if intention.status != "active" {
                continue;
            }
            let scope = intention.scope.as_deref().unwrap_or("").trim();
            if !in_scope(scope, &request.scope) {
                continue;
            }
            let Some(deadline) = intention.deadline_ms else {
                continue;
            };
            if deadline > clock {
                continue;
            }
            let receipt = store
                .origin_seq(&intention.id)
                .map(effect_receipt_id)
                .map(|id| format!(" receipt `vst:{id}`"))
                .unwrap_or_default();
            let opening: String = intention
                .content
                .chars()
                .filter(|c| *c != '\n')
                .take(80)
                .collect();
            due.push(format!(
                "- `vst:intention:{}`{receipt} deadline={deadline} {opening}",
                intention.id
            ));
        }
        due.sort();
    }

    let dropped = if request.source == "compact" {
        dropped_lines(store)
    } else {
        Vec::new()
    };

    let book = rulebook::apply_owner(
        rulebook::default_rulebook(),
        owner.iter().map(String::as_str),
    );
    let mut lines = vec![
        format!("capsule version={}:{}", rulebook::VERSION, rulebook::HANDLE),
        format!("head `{token}` clock={clock:020} digest={digest}"),
        format!(
            "source {} mode {} scope {}",
            request.source, request.mode, request.scope
        ),
    ];
    if let Some(branch) = &request.branch {
        lines.push(format!("branch {branch}"));
    }
    if let Some(commit) = &request.head_commit {
        lines.push(format!("commit {commit}"));
    }
    if let Some(bound) = since {
        lines.push(format!("delta since={bound:016x}"));
    }
    lines.push("## rulebook".into());
    lines.push(book);
    push_section(&mut lines, "pinned decisions", &decisions);
    push_section(&mut lines, "changed files", &changed);
    push_section(&mut lines, "due obligations", &due);
    push_section(&mut lines, "retired", &retired);
    push_section(&mut lines, "dropped", &dropped);
    lines.join("\n")
}

fn push_section(lines: &mut Vec<String>, title: &str, rows: &[String]) {
    lines.push(format!("## {title}"));
    if rows.is_empty() {
        lines.push("none".into());
    } else {
        lines.extend(rows.iter().cloned());
    }
}

fn in_scope(record_scope: &str, want: &str) -> bool {
    let got = if record_scope.is_empty() {
        "user"
    } else {
        record_scope
    };
    got == want
}

fn codebase_ok(tags: &[String], codebase: Option<&str>) -> bool {
    let Some(name) = codebase else {
        return true;
    };
    let want = format!("codebase:{name}");
    tags.iter().any(|tag| tag == &want)
}

fn decision_line(store: &strata_store::StrataStore, record: &strata_store::NodeRecord) -> String {
    let receipt = store
        .origin_seq(&record.id)
        .map(effect_receipt_id)
        .map(|id| format!(" receipt `vst:{id}`"))
        .unwrap_or_default();
    let mut anchors: Vec<String> = store
        .anchors_for(&record.id)
        .into_iter()
        .map(|anchor| {
            let status = anchor.last_status.as_deref().unwrap_or("unchecked");
            let place = match &anchor.symbol {
                Some(symbol) if !symbol.is_empty() => format!("{}#{symbol}", anchor.file_path),
                _ => anchor.file_path.clone(),
            };
            format!(" `vst:anchor:{}` {place} {status}", anchor.id)
        })
        .collect();
    anchors.sort();
    let opening: String = record
        .content
        .chars()
        .filter(|c| *c != '\n')
        .take(80)
        .collect();
    format!(
        "- `vst:{}`{receipt}{} {opening}",
        record.id,
        anchors.join("")
    )
}

fn anchor_line(
    store: &strata_store::StrataStore,
    node_id: &str,
    anchor: &strata_store::AnchorRecord,
) -> String {
    let receipt = store
        .origin_seq(node_id)
        .map(effect_receipt_id)
        .map(|id| format!(" receipt `vst:{id}`"))
        .unwrap_or_default();
    let status = anchor.last_status.as_deref().unwrap_or("unchecked");
    format!(
        "- `{path}` `vst:{node_id}`{receipt} `vst:anchor:{}` {status}",
        anchor.id,
        path = anchor.file_path,
    )
}

fn dropped_lines(store: &strata_store::StrataStore) -> Vec<String> {
    let nodes = store.nodes();
    let mut audits: Vec<&strata_store::NodeRecord> = nodes
        .iter()
        .filter(|record| {
            record.is_live() && record.tags.iter().any(|tag| tag == "compaction-audit")
        })
        .collect();
    // `mem-` ids encode the log sequence, so the greatest id is the latest audit.
    audits.sort_by(|a, b| a.id.cmp(&b.id));
    let Some(audit) = audits.last() else {
        return Vec::new();
    };
    let mut dropped: Vec<String> = audit
        .tags
        .iter()
        .filter_map(|tag| tag.strip_prefix("dropped:"))
        .filter(|id| is_memory_id(id))
        .map(|id| format!("- `vst:{id}`"))
        .collect();
    dropped.sort();
    dropped.dedup();
    dropped
}

fn is_memory_id(id: &str) -> bool {
    id.strip_prefix("mem-")
        .is_some_and(|rest| rest.len() == 16 && rest.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Backtick tokens that match the `vst:` grammar, in order, once each.
pub fn handles_in(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let Some(end) = text[start..].find('`') else {
            break;
        };
        let token = &text[start..start + end];
        if accept_handle(token) && !out.iter().any(|seen| seen == token) {
            out.push(token.to_string());
        }
        i = start + end + 1;
    }
    out
}

fn accept_handle(token: &str) -> bool {
    let Some(rest) = token.strip_prefix("vst:") else {
        return false;
    };
    if rest == "rulebook:1" {
        return true;
    }
    if let Some(hex16) = rest.strip_prefix("head:") {
        let mut parts = hex16.split(':');
        let seq = parts.next().unwrap_or("");
        let frame = parts.next().unwrap_or("");
        let segment = parts.next().unwrap_or("");
        let ckpt = parts.next().unwrap_or("");
        return parts.next().is_none()
            && is_hex(seq, 16)
            && is_hex(frame, 64)
            && (segment == "unsealed" || is_hex(segment, 64))
            && (ckpt == "none" || is_hex(ckpt, 64));
    }
    if let Some(id) = rest.strip_prefix("eff-") {
        return is_hex(id, 16);
    }
    if let Some(id) = rest.strip_prefix("mem-") {
        return is_hex(id, 16);
    }
    if let Some(id) = rest.strip_prefix("anchor:") {
        return !id.is_empty() && !id.contains(|c: char| c.is_whitespace() || c == '`');
    }
    if let Some(id) = rest.strip_prefix("intention:") {
        return !id.is_empty() && !id.contains(|c: char| c.is_whitespace() || c == '`');
    }
    false
}

fn is_hex(text: &str, len: usize) -> bool {
    text.len() == len && text.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn adapt(harness: &str, body: &str) -> String {
    let body = fit(harness, body);
    match harness {
        "plain" => {
            if body.is_empty() {
                String::new()
            } else {
                format!("{body}\n")
            }
        }
        // Claude Code SessionStart, and the Codex shape recorded by
        // packages/vestige-init/hooks/reflex.cjs `hostOutput` for every
        // non-cursor host.
        "claude-code" | "codex" => {
            let event = "SessionStart";
            serde_json::to_string(&json!({
                "hookSpecificOutput": {
                    "hookEventName": event,
                    "additionalContext": body,
                }
            }))
            .unwrap_or_default()
        }
        _ => String::new(),
    }
}

fn fit(harness: &str, body: &str) -> String {
    let cap = match harness {
        "claude-code" | "codex" => CONTEXT_CAP,
        "plain" => usize::MAX,
        _ => 0,
    };
    if body.len() <= cap {
        return body.to_string();
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut omitted: Vec<String> = Vec::new();
    let mut used = 0usize;
    for line in body.lines() {
        let extra = line.len() + usize::from(!kept.is_empty());
        if used + extra > cap {
            omitted.extend(handles_in(line));
            continue;
        }
        used += extra;
        kept.push(line);
    }
    omitted.sort();
    omitted.dedup();
    let mut text = kept.join("\n");
    if omitted.is_empty() {
        return text;
    }
    let mut note = String::from("omitted:");
    for handle in &omitted {
        note.push(' ');
        note.push('`');
        note.push_str(handle);
        note.push('`');
    }
    loop {
        let extra = note.len() + usize::from(!text.is_empty());
        if text.len() + extra <= cap {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&note);
            return text;
        }
        if kept.pop().is_none() {
            return if note.len() <= cap {
                note
            } else {
                String::new()
            };
        }
        text = kept.join("\n");
    }
}

pub struct RunOutput {
    pub stdout: String,
    pub stderr: String,
}

/// Run one capsule. Never returns an error to the harness: failures are a
/// stderr line and empty stdout.
pub fn execute(
    data_dir: &Path,
    harness: &str,
    event: &str,
    mode_flag: Option<&str>,
    stdin: &str,
) -> RunOutput {
    match execute_inner(data_dir, harness, event, mode_flag, stdin) {
        Ok(stdout) => RunOutput {
            stdout,
            stderr: String::new(),
        },
        Err(err) => RunOutput {
            stdout: String::new(),
            stderr: format!("capsule: {err}\n"),
        },
    }
}

fn execute_inner(
    data_dir: &Path,
    harness: &str,
    event: &str,
    mode_flag: Option<&str>,
    stdin: &str,
) -> Result<String, String> {
    if event != "session-start" {
        return Err(format!("refused event {event}"));
    }
    if !matches!(harness, "claude-code" | "codex" | "plain") {
        return Err(format!("refused harness {harness}"));
    }
    let payload: Value =
        serde_json::from_str(stdin.trim()).map_err(|_| "malformed stdin".to_string())?;
    if !payload.is_object() {
        return Err("malformed stdin".into());
    }
    let mut request = Request::from_value(&payload);
    if !accepted_source(&request.source) {
        return Err(format!("refused source {}", request.source));
    }
    let cwd = payload
        .get("cwd")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| "no cwd".to_string())?;
    if request.since.is_none()
        && matches!(request.source.as_str(), "resume" | "fork")
        && let Some(path) = payload.get("transcript_path").and_then(Value::as_str)
    {
        request.since = since_from_transcript(Path::new(path));
    }
    let (mode, branch, commit, files) = git_facts(&cwd);
    if mode_flag.is_some() {
        request.mode = mode_flag.unwrap_or("normal").to_string();
    } else if request.mode == "normal" {
        request.mode = mode;
    }
    if request.branch.is_none() {
        request.branch = branch;
    }
    if request.head_commit.is_none() {
        request.head_commit = commit;
    }
    if request.changed_files.is_empty() {
        request.changed_files = files;
    }
    if !data_dir.join("log").is_dir() {
        return Err("no strata log".into());
    }
    let body = match crate::attach::try_serve_lock(data_dir) {
        Ok(None) => body_through_owner(data_dir, &request)?,
        Ok(Some(lock)) => {
            let storage = strata_memory::open(data_dir).map_err(|err| err.to_string())?;
            let value = session_tool(storage.as_ref(), Some(&request_value(&request)))?;
            drop(lock);
            value
                .get("body")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        }
        Err(err) => return Err(err.to_string()),
    };
    if body.is_empty() {
        return Err("empty capsule".into());
    }
    let stdout = adapt(harness, &body);
    if stdout.is_empty() {
        return Err("adapter produced nothing".into());
    }
    Ok(if stdout.ends_with('\n') {
        stdout
    } else {
        format!("{stdout}\n")
    })
}

fn request_value(request: &Request) -> Value {
    json!({
        "format": "capsule",
        "source": request.source,
        "mode": request.mode,
        "scope": request.scope,
        "codebase": request.codebase,
        "branch": request.branch,
        "head_commit": request.head_commit,
        "changed_files": request.changed_files,
        "since": request.since,
    })
}

fn body_through_owner(data_dir: &Path, request: &Request) -> Result<String, String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| err.to_string())?;
    let value = rt
        .block_on(async {
            tokio::time::timeout(
                ATTACH_BUDGET,
                crate::attach::call_tool(data_dir, "session_start", request_value(request)),
            )
            .await
        })
        .map_err(|_| "owner did not answer before the hook budget".to_string())?
        .map_err(|err| err.to_string())?;
    Ok(value
        .get("body")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string())
}

pub fn run_cli(data_dir: &Path, harness: &str, event: &str, mode: Option<&str>) -> ! {
    let mut stdin = String::new();
    let _ = std::io::stdin().take(1 << 20).read_to_string(&mut stdin);
    let output = execute(data_dir, harness, event, mode, &stdin);
    if !output.stderr.is_empty() {
        let _ = std::io::stderr().write_all(output.stderr.as_bytes());
    }
    if !output.stdout.is_empty() {
        let _ = std::io::stdout().write_all(output.stdout.as_bytes());
        let _ = std::io::stdout().flush();
    }
    std::process::exit(0);
}

fn since_from_transcript(path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > 8 * 1024 * 1024 {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    let mut last = None;
    let mut rest = text.as_str();
    while let Some(idx) = rest.find("vst:head:") {
        let tail = &rest[idx + "vst:head:".len()..];
        let seq: String = tail.chars().take(16).collect();
        if is_hex(&seq, 16) {
            last = u64::from_str_radix(&seq, 16).ok();
        }
        rest = &tail[seq.len().min(tail.len())..];
    }
    last
}

fn git_facts(cwd: &Path) -> (String, Option<String>, Option<String>, Vec<String>) {
    let mode = git_mode(cwd);
    let branch = git_out(cwd, &["rev-parse", "--abbrev-ref", "HEAD"]);
    let commit = git_out(cwd, &["rev-parse", "HEAD"]);
    let mut files = Vec::new();
    for args in [
        &["diff", "--name-only"][..],
        &["diff", "--name-only", "--cached"][..],
        &["ls-files", "--others", "--exclude-standard"][..],
    ] {
        if let Some(text) = git_out(cwd, args) {
            files.extend(
                text.lines()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
            );
        }
    }
    if mode == "merge"
        && let Some(text) = git_out(cwd, &["diff", "--name-only", "--diff-filter=U"])
    {
        files.extend(
            text.lines()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        );
    }
    files.sort();
    files.dedup();
    (mode, branch, commit, files)
}

fn git_out(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn git_mode(cwd: &Path) -> String {
    let Some(git) = git_dir(cwd) else {
        return "normal".into();
    };
    if git.join("MERGE_HEAD").is_file() {
        return "merge".into();
    }
    if git.join("rebase-merge").is_dir() || git.join("rebase-apply").is_dir() {
        return "rebase".into();
    }
    if git.join("BISECT_LOG").is_file() {
        return "bisect".into();
    }
    "normal".into()
}

fn git_dir(cwd: &Path) -> Option<PathBuf> {
    for dir in cwd.ancestors() {
        let dot = dir.join(".git");
        if dot.is_dir() {
            return Some(dot);
        }
        if dot.is_file() {
            let text = std::fs::read_to_string(&dot).ok()?;
            let gitdir = text
                .lines()
                .find_map(|line| line.trim().strip_prefix("gitdir:"))?
                .trim();
            return Some(dir.join(gitdir));
        }
    }
    None
}

pub fn is_capsule_command(command: &str) -> bool {
    command == COMMAND
}

/// Append the fragment's SessionStart groups. Arrays are not replaced.
pub fn append_fragment(settings: &mut Value, fragment: &Value) -> bool {
    let Some(groups) = fragment
        .pointer("/hooks/SessionStart")
        .and_then(Value::as_array)
    else {
        return false;
    };
    let root = settings.as_object_mut().expect("settings object");
    let hooks = root
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("hooks object");
    let event = hooks
        .entry("SessionStart")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .expect("SessionStart array");
    let mut added = false;
    for group in groups {
        if !event.iter().any(|existing| existing == group) {
            event.push(group.clone());
            added = true;
        }
    }
    added
}

pub fn load_fragment(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    serde_json::from_str(&text).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn git_repo(dir: &Path) {
        let run = |args: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .env("GIT_AUTHOR_NAME", "Capsule")
                .env("GIT_AUTHOR_EMAIL", "capsule@example.com")
                .env("GIT_COMMITTER_NAME", "Capsule")
                .env("GIT_COMMITTER_EMAIL", "capsule@example.com")
                .status()
                .unwrap();
            assert!(status.success(), "{args:?}");
        };
        run(&["init"]);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("src/lib.rs"),
            "pub fn parse(raw: &str) -> &str { raw }\n",
        )
        .unwrap();
        run(&["add", "."]);
        run(&["commit", "-m", "parse"]);
    }

    #[test]
    fn the_same_head_renders_the_same_bytes_and_resolves_every_handle() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        git_repo(&repo);
        std::fs::write(
            repo.join("src/lib.rs"),
            "pub fn parse(raw: &str) -> &str { raw.trim() }\n",
        )
        .unwrap();
        let data = dir.path().join("data");
        let storage = strata_memory::open(&data).unwrap();
        let rt = runtime();
        let cog = Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));
        let saved = rt
            .block_on(crate::tools::codebase_unified::execute(
                &storage,
                &cog,
                &vestige_core::OutputConfig::default(),
                Some(json!({
                    "action": "remember_decision",
                    "codebase": "fixture",
                    "decision": "parse stays borrowed",
                    "rationale": "the caller owns the string",
                    "files": ["src/lib.rs#parse"],
                    "repoPath": repo,
                })),
            ))
            .unwrap();
        let id = saved["nodeId"].as_str().unwrap().to_string();
        rt.block_on(crate::tools::intention_unified::execute(
            &storage,
            &cog,
            Some(json!({
                "action": "set",
                "description": "due obligation",
                "deadline": "2000-01-01T00:00:00Z",
                "scope": "user",
            })),
        ))
        .unwrap();
        rt.block_on(crate::tools::intention_unified::execute(
            &storage,
            &cog,
            Some(json!({
                "action": "set",
                "description": "later obligation",
                "deadline": "2099-01-01T00:00:00Z",
                "scope": "user",
            })),
        ))
        .unwrap();
        let args = json!({
            "format": "capsule",
            "source": "startup",
            "scope": "user",
            "codebase": "fixture",
            "mode": "normal",
            "changed_files": ["src/lib.rs"],
            "branch": "main",
            "head_commit": "abc",
        });
        let first = session_tool(storage.as_ref(), Some(&args)).unwrap();
        let second = session_tool(storage.as_ref(), Some(&args)).unwrap();
        assert_eq!(first["body"], second["body"]);
        let body = first["body"].as_str().unwrap();
        assert!(body.contains(&format!("`vst:{id}`")), "{body}");
        assert!(
            body.contains("src/lib.rs#parse") || body.contains("src/lib.rs"),
            "{body}"
        );
        assert!(
            body.contains("due obligation") || body.contains("vst:intention:"),
            "{body}"
        );
        assert!(!body.contains("later obligation"), "{body}");
        assert!(body.contains(rulebook::HANDLE), "{body}");
        let head_before = with_store(storage.as_ref(), |store| store.log().head()).unwrap();
        let _ = session_tool(storage.as_ref(), Some(&args)).unwrap();
        let head_after = with_store(storage.as_ref(), |store| store.log().head()).unwrap();
        assert_eq!(head_before, head_after);

        for handle in first["handles"].as_array().unwrap() {
            let handle = handle.as_str().unwrap();
            assert!(accept_handle(handle), "{handle}");
            let inner = handle.strip_prefix("vst:").unwrap();
            if let Some(id) = inner.strip_prefix("mem-").map(|hex| format!("mem-{hex}")) {
                let got = rt
                    .block_on(crate::tools::recall::execute(
                        &storage,
                        &cog,
                        &vestige_core::OutputConfig::default(),
                        Some(json!({"handle": id})),
                    ))
                    .unwrap();
                assert!(got.to_string().contains(&id), "{got}");
            } else if inner.starts_with("eff-") {
                let got = rt
                    .block_on(crate::tools::receipt::execute(
                        &storage,
                        Some(json!({"action": "get", "receipt_id": inner})),
                    ))
                    .unwrap_or_else(|err| panic!("{inner}: {err}"));
                assert_eq!(got["action"], "get", "{got}");
            } else if let Some(anchor_id) = inner.strip_prefix("anchor:") {
                let found = with_store(storage.as_ref(), |store| store.anchor(anchor_id).is_some())
                    .unwrap();
                assert!(found, "{anchor_id}");
            } else if let Some(intention_id) = inner.strip_prefix("intention:") {
                let found = with_store(storage.as_ref(), |store| {
                    store.get_intention(intention_id).is_some()
                })
                .unwrap();
                assert!(found, "{intention_id}");
            }
        }
        let wrapped = adapt("claude-code", body);
        let parsed: Value = serde_json::from_str(wrapped.trim()).unwrap();
        assert_eq!(
            parsed["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap(),
            body
        );
        assert!(
            parsed["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap()
                .len()
                <= CONTEXT_CAP
        );
    }

    #[test]
    fn resume_is_a_delta_and_an_unknown_source_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let storage = strata_memory::open(dir.path()).unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "first decision".into(),
                node_type: "decision".into(),
                tags: vec!["codebase:fixture".into()],
                ..vestige_core::IngestInput::default()
            })
            .unwrap();
        let head = with_store(storage.as_ref(), |store| store.log().head().last_acked_seq).unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "second decision".into(),
                node_type: "decision".into(),
                tags: vec!["codebase:fixture".into()],
                ..vestige_core::IngestInput::default()
            })
            .unwrap();
        let full = session_tool(
            storage.as_ref(),
            Some(&json!({"source": "startup", "codebase": "fixture"})),
        )
        .unwrap();
        assert!(full["body"].as_str().unwrap().contains("first decision"));
        assert!(full["body"].as_str().unwrap().contains("second decision"));
        let delta = session_tool(
            storage.as_ref(),
            Some(&json!({"source": "resume", "since": head, "codebase": "fixture"})),
        )
        .unwrap();
        let body = delta["body"].as_str().unwrap();
        assert!(body.contains("second decision"), "{body}");
        assert!(!body.contains("first decision"), "{body}");
        assert!(body.contains("delta since="), "{body}");
        let refused = session_tool(storage.as_ref(), Some(&json!({"source": "nope"}))).unwrap();
        assert_eq!(refused["body"].as_str().unwrap(), "");
        assert_eq!(refused["skipped"], "refused_source");
    }

    #[test]
    fn overflow_lists_omitted_handles_and_never_cuts_one() {
        let mut body = String::from("head `vst:head:0000000000000001:");
        body.push_str(&"ab".repeat(32));
        body.push_str(":unsealed:none`\n");
        for i in 0..400 {
            body.push_str(&format!("- `vst:mem-{i:016x}` line {i}\n"));
        }
        let fitted = fit("claude-code", &body);
        assert!(fitted.len() <= CONTEXT_CAP, "{}", fitted.len());
        assert!(fitted.contains("omitted:"), "{fitted}");
        for handle in handles_in(&fitted) {
            assert!(accept_handle(&handle), "{handle}");
            assert!(
                fitted.contains(&format!("`{handle}`")),
                "handle is cut: {handle}"
            );
        }
        assert!(!accept_handle("vst:mem-zzzz"));
        assert!(!accept_handle("vst:memory"));
    }

    #[test]
    fn fail_open_paths_print_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = execute(
            dir.path(),
            "claude-code",
            "session-start",
            None,
            "{\"source\":\"startup\"}",
        );
        assert!(missing.stdout.is_empty(), "{}", missing.stdout);
        assert!(!missing.stderr.is_empty());
        let bad = execute(dir.path(), "claude-code", "session-start", None, "not-json");
        assert!(bad.stdout.is_empty());
        let refused = execute(
            dir.path(),
            "claude-code",
            "session-start",
            None,
            "{\"source\":\"nope\"}",
        );
        assert!(refused.stdout.is_empty());
    }

    #[test]
    fn the_capsule_group_appends_and_a_second_merge_is_identical() {
        let fragment = json!({"hooks": {"SessionStart": [{
            "matcher": MATCHER,
            "hooks": [{"type": "command", "command": COMMAND, "timeout": 8}]
        }]}});
        let user = json!({"hooks": [{"type": "command", "command": "echo mine"}]});
        let mut settings = json!({"hooks": {"SessionStart": [user.clone()]}});
        assert!(append_fragment(&mut settings, &fragment));
        let once = serde_json::to_vec_pretty(&settings).unwrap();
        assert!(!append_fragment(&mut settings, &fragment));
        assert_eq!(once, serde_json::to_vec_pretty(&settings).unwrap());
        assert_eq!(settings["hooks"]["SessionStart"][0], user);
        assert_eq!(
            settings["hooks"]["SessionStart"][1]["hooks"][0]["command"],
            COMMAND
        );
        assert!(is_capsule_command(COMMAND));
        assert!(!is_capsule_command("echo mine"));
    }
}
