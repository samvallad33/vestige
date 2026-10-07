//! THE INSTITUTION — Vestige enforcement hooks v2 (Rust).
//!
//! `vestige hook` is spawned by Claude Code / ZCode on every tool call with the
//! host's event JSON on stdin. It enforces the memory contract and the ten
//! institutional mechanisms researched 2026-10-04:
//!
//!   TIME-OUT CARD      irreversible operations are denied once until the agent
//!                      recalls that pattern's failure history and states blast
//!                      radius + rollback; the repeat passes (block-then-insist)
//!   STOP-WORK REGISTRY memories tagged `stop-work` carrying `pattern: <regex>`
//!                      block matching commands with their citation; a deliberate
//!                      override needs the OVERRIDE-STOPWORK prefix and a receipt
//!   DRAWDOWN LIMIT     after N failed mutating ops in a repo, mutating tools
//!                      pause until a post-mortem memory is written
//!   SEARCH-ACT GATE    a content search by ANY program (grep, rg, awk, sed,
//!                      perl, python re.*) is denied unless the immediately
//!                      preceding Vestige call was recall or codebase context
//!   FOQA               force-pushes, secret-adjacent reads, destructive deletes
//!                      and revert chains are counted per repo and filed as
//!                      flight-data memories at threshold
//!   RIPPLE TAG         a command family that succeeded and then failed is the
//!                      only trajectory class that earns replay
//!   SBAR HANDOFF       heavy turns close with a transfer-of-responsibility packet
//!   HINDSIGHT RELABEL  failed turns that produced a byproduct get re-indexed
//!                      under their achieved goal
//!   NEAR-MISS + PREREG prompt triggers for immunity filing and retrieval plans
//!
//! Plus the carried-over contract: session_start before any tool, the
//! Vestige-call gap (default 1), codebase context before the first edit in a
//! repo, corrections/decisions tied to a file shown before it changes, and the
//! end-of-turn save guard.
//!
//! The hook only READS the store (localhost dashboard GET). Every write is made
//! by the model through the Vestige MCP tools. Any defect here fails open: it
//! prints nothing and exits 0.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;
use std::time::Duration;

pub const STOPWORK_TAG: &str = "stop-work";
pub const STOPWORK_OVERRIDE: &str = "OVERRIDE-STOPWORK";
const V: &str = "mcp__vestige__";
const FREE_TOOLS: [&str; 1] = ["ToolSearch"];
const EDITS: [&str; 7] = ["Edit", "Write", "MultiEdit", "NotebookEdit", "ApplyPatch", "StrReplace", "Delete"];
const CONTEXT_TOOLS: [&str; 2] = ["recall", "codebase:get_context"];
pub const DRAWDOWN_LIMIT: u32 = 5;

fn max_gap() -> u32 {
    std::env::var("VESTIGE_HOOK_MAX_GAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

// ---------------------------------------------------------------- regexes

fn re(p: &str) -> Regex {
    Regex::new(p).expect("static hook regex")
}

static READ_ONLY_CMD: LazyLock<Regex> = LazyLock::new(|| {
    re(r"^\s*(cd [^;&|]+(;|&&)\s*)?(cat|sed|head|tail|grep|rg|ls|find|echo|file|wc|awk|jq|git (log|show|diff|status))\b")
});
static FAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    re(r"FAILED|failures: [1-9]|Traceback \(most recent call last\)|panicked at|error\[E\d+\]|npm ERR!|AssertionError|test result: FAILED")
});
static SAFE_RM: LazyLock<Regex> = LazyLock::new(|| {
    re(r"(tmp/|var/folders/|node_modules|target/|dist/|build/|\.git|scratchpad|__pycache__|\.pytest_cache|work/)")
});
static SEARCH_ACT: LazyLock<Regex> = LazyLock::new(|| {
    re(r#"\b(grep|egrep|fgrep|rg|ripgrep|git\s+grep)\b|awk\s+[^|;&]*[/'"][^/'"]+/|sed\s+-n[e]?\s+'.*/.*/p'|\bperl\s+-n[ei]\b|python[23]?\b[^|&]*(re\.(search|findall|match|finditer)|in\s+open\(|\.read\(\))|find\b[^|;&]*-exec\s+(grep|rg)"#)
});
static SECRET_ADJ: LazyLock<Regex> = LazyLock::new(|| {
    re(r"\.env\b|id_rsa|id_ed25519|\.pem\b|\.aws/credentials|kaggle\.json|\.netrc|credentials\.json|service[-_]?account")
});
static FORCE_PUSH: LazyLock<Regex> =
    LazyLock::new(|| re(r"git\s+push\b[^;&|]*(--force([^-\w]|$)|\s-f\b)"));
static REVERT_CHAIN: LazyLock<Regex> =
    LazyLock::new(|| re(r"git\s+(revert|checkout\s+[0-9a-f]{7,}\s+--|restore\b)"));
static DESTRUCTIVE: LazyLock<Regex> = LazyLock::new(|| {
    re(r"rm\s+-+[a-z]*[rf][a-z]*\b|git\s+(checkout\s+--\s+\.|restore\s+\.)|rsync\b[^&]*--delete|tar\b[^&]*--remove-files|aws\s+s3\s+sync\b[^&]*--delete")
});
static STOPWORK_LINE: LazyLock<Regex> = LazyLock::new(|| re(r#"(?i)^pattern:[ \t]*[`'"]?([^`'"\n]+?)[`'"]?[ \t]*$"#));

/// (name, regex) pairs for the TIME-OUT CARD. Order matters only for the label.
static CRITICAL: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    vec![
        ("force-push", re(r"git\s+push\b[^;&|]*(--force([^-\w]|$)|\s-f\b)")),
        ("hard-reset", re(r"git\s+reset\s+--hard|git\s+reflog\s+expire|git\s+filter-branch|git\s+clean\s+-[a-z]*f")),
        ("branch-delete", re(r"git\s+branch\s+-D\b")),
        ("destructive-rm", re(r"rm\s+-+[a-z]*[rf][a-z]*\b|shred\b|truncate\s+-s\s*0")),
        ("disk-write", re(r"\bdd\b[^|]*of=/dev/|mkfs\b|diskutil\s+erase")),
        ("db-drop", re(r"(?i)\b(drop|truncate)\s+(table|database|schema)\b")),
        ("kaggle-push", re(r"kaggle\s+kernels\s+push")),
        ("publish", re(r"cargo\s+publish|npm\s+(un)?publish|twine\s+upload")),
        ("repo-delete", re(r"gh\s+repo\s+delete|git\s+push\b[^;&|]*--delete")),
        ("cloud-destroy", re(r"terraform\s+destroy|pulumi\s+destroy|docker\s+system\s+prune|kubectl\s+delete|aws\s+s3\s+(rb|sync\b[^&]*--delete)|gcloud\s+.*\sdelete|az\s+group\s+delete|helm\s+uninstall")),
        ("cron-wipe", re(r"crontab\s+-r\b")),
        ("perm-wipe", re(r"chmod\s+-R\s+0|chown\s+-R\s+~")),
    ]
});

static DULL: LazyLock<std::collections::HashSet<&'static str>> = LazyLock::new(|| {
    [
        "src", "lib", "crates", "tools", "tests", "test", "bin", "docs", "doc", "apps", "packages",
        "index", "main", "mod", "readme", "users", "developer", "private", "tmp", "var", "folders",
        "scratchpad", "home", "usr", "local", "node_modules", "target", "dist", "build", "the",
        "and", "for", "with", "from", "that", "this", "echo", "cat", "ls", "cd", "grep", "sed",
        "awk", "head", "tail", "python", "python3", "bash", "sh", "zsh", "then", "done", "else",
        "true", "false", "null", "none", "json", "print", "import", "open", "read", "resources",
        "contents", "state", "status", "config", "cli", "logs", "log", "data", "output",
        "applications", "library", "support", "skills", "plugins", "cache", "find", "name",
    ]
    .into_iter()
    .collect()
});

static PROMPT_TRIGGERS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (re(r"(?i)\bremember (this|that)\b|\bsave this\b|\bdon'?t forget\b"),
         "recall the topic tag to check it is not saved yet, then smart_ingest and quote the nodeId"),
        (re(r"(?i)\bremind me\b|\bdeadline\b|\bby (monday|tuesday|wednesday|thursday|friday|tomorrow|tonight)\b|\bfollow up\b"),
         "intention(action='set', description=..., deadline=... or trigger=...)"),
        (re(r"(?i)\bwhy did\b.*\b(fail|break|crash)|\broot cause\b|\bwhat broke\b"),
         "save the failure as an event, then causal_walk(start_points=[{kind:'logged_write', node_id}], promote=false) and forgotten_lesson(failure_id)"),
        (re(r"(?i)\bthat'?s wrong\b|\bobsolete\b|\bnot true anymore\b"),
         "memory(action='demote', id=...), then memory(action='edit') to admit the corrected version"),
        (re(r"(?i)\bhave we\b|\bdid we (already|ever)\b|\bwhat about\b"),
         "recall(handle='<narrow topic tag>') and read its decisions and corrections before answering"),
        (re(r"(?i)\bdecide|\bdecision\b|\bshould we\b|\bwhich (one|option)\b"),
         "recall the topic tag first; show any decision or correction that contradicts the plan; save the decision with smart_ingest when Sam makes it"),
        (re(r"(?i)\bnear miss\b|\bclose call\b|\balmost (broke|lost|dropped|deleted|shipped)|\bnearly (broke|lost|dropped|deleted)"),
         "NEAR-MISS (immunity filing): smart_ingest the pattern + what saved it, tags [near-miss, <repo>]; keep the reporter's identity out of the content (suppression on any identity memory keeps bytes, hides the reporter, keeps the pattern searchable)"),
        (re(r"(?i)\b(deploy|ship it|submit it|publish|launch|run the migration|push to prod|go live)\b"),
         "PREREGISTER: before executing, smart_ingest the retrieval plan — which memories you may use + the success criterion — tags [prereg, <repo>]; registered retrieval separates plan from rationalization"),
    ]
});

// ---------------------------------------------------------------- state

#[derive(Serialize, Deserialize, Default, Clone)]
struct Turn {
    #[serde(default)] mut_n: u32,
    #[serde(default)] vwrites: u32,
    #[serde(default)] vcalls: u32,
    #[serde(default)] tfailed: u32,
    #[serde(default)] asked: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StopRule {
    pub id: String,
    pub pattern: String,
    pub reason: String,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct HookState {
    #[serde(default)] started: bool,
    #[serde(default)] gap: u32,
    #[serde(default)] denied: u32,
    #[serde(default)] vcalls: u32,
    #[serde(default)] seen: Vec<String>,
    #[serde(default)] dead: Vec<String>,
    #[serde(default)] failed: Vec<String>,
    #[serde(default)] pending: Vec<String>,
    #[serde(default)] timeouts: Vec<String>,
    #[serde(default)] stopwork: Vec<StopRule>,
    #[serde(default)] foqa: HashMap<String, HashMap<String, u32>>,
    #[serde(default)] fams: HashMap<String, bool>,
    #[serde(default)] drawdown: HashMap<String, u32>,
    #[serde(default)] dd_armed: HashMap<String, bool>,
    #[serde(default)] last_v: String,
    #[serde(default)] rippled: Vec<String>,
    #[serde(default)] foqa_filed: Vec<String>,
    #[serde(default)] overrides: Vec<String>,
    #[serde(default)] ctx_repos: Vec<String>,
    #[serde(default)] turn: Turn,
    #[serde(default)] at: u64,
    #[serde(default)] searches: u32,
    #[serde(default)] searches_after_recall: u32,
    #[serde(default)] gap_denied_total: u32,
    #[serde(default)] last_failed_fam: String,
    #[serde(default)] last_failed_ts: u64,
    #[serde(default)] searched_since_fail: bool,
}

fn state_dir() -> PathBuf {
    if let Ok(d) = std::env::var("VESTIGE_HOOK_STATE_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".vestige").join("hooks")
}

fn state_path(session: &str) -> PathBuf {
    let clean: String = session
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') { c } else { '_' })
        .take(80)
        .collect();
    state_dir().join(format!("institution-{clean}.json"))
}

fn load_state(session: &str) -> HookState {
    std::fs::read_to_string(state_path(session))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_state(session: &str, st: &mut HookState) {
    let trim = |v: &mut Vec<String>, n: usize| {
        let len = v.len();
        if len > n {
            v.drain(0..len - n);
        }
    };
    trim(&mut st.seen, 400);
    trim(&mut st.dead, 300);
    trim(&mut st.failed, 50);
    trim(&mut st.timeouts, 200);
    trim(&mut st.rippled, 60);
    trim(&mut st.foqa_filed, 60);
    trim(&mut st.overrides, 40);
    trim(&mut st.pending, 12);
    let _ = std::fs::create_dir_all(state_dir());
    let tmp = state_path(session).with_extension(format!("{}.tmp", std::process::id()));
    if let Ok(body) = serde_json::to_string(&*st) {
        if std::fs::write(&tmp, body).is_ok() {
            let _ = std::fs::rename(&tmp, state_path(session));
        }
    }
}

fn chash(cmd: &str) -> String {
    let mut h = DefaultHasher::new();
    cmd.chars().take(200).collect::<String>().hash(&mut h);
    format!("{:x}", h.finish())
}

// ---------------------------------------------------------------- dashboard

fn api_port() -> u16 {
    std::env::var("VESTIGE_API")
        .ok()
        .and_then(|u| u.rsplit(':').next().map(|p| p.trim_end_matches('/').to_string()))
        .and_then(|p| p.parse().ok())
        .unwrap_or(3927)
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Minimal localhost HTTP GET (no client library: the default graph must not
/// link reqwest). 400ms connect + 800ms read, mirroring the python hook's budget.
/// Unit tests set this so api_get is deterministic (no localhost dashboards,
/// no env-var races between parallel tests). The shipped binary never sets it.
#[cfg(test)]
pub(crate) static TEST_OFFLINE: AtomicBool = AtomicBool::new(false);

fn api_get(path_and_query: &str) -> Option<Value> {
    #[cfg(test)]
    if TEST_OFFLINE.load(Ordering::Relaxed) {
        return None;
    }
    let addr = format!("127.0.0.1:{}", api_port());
    let mut stream = TcpStream::connect_timeout(
        &addr.parse().ok()?,
        Duration::from_millis(400),
    )
    .ok()?;
    stream.set_read_timeout(Some(Duration::from_millis(800))).ok()?;
    stream
        .write_all(format!("GET {path_and_query} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").as_bytes())
        .ok()?;
    let mut body = String::new();
    stream.read_to_string(&mut body).ok()?;
    let split: Vec<&str> = body.splitn(2, "\r\n\r\n").collect();
    serde_json::from_str(split.get(1)?.trim()).ok()
}

fn fetch_tag(tag: &str, node_type: Option<&str>, limit: u32) -> Vec<(String, String, String, String)> {
    // (id, kind, created, content)
    let mut q = format!("/api/memories?tag={}&limit={limit}", urlencode(tag));
    if let Some(kind) = node_type {
        q.push_str(&format!("&node_type={}", urlencode(kind)));
    }
    let Some(v) = api_get(&q) else { return vec![] };
    v.get("memories")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    Some((
                        m.get("id")?.as_str()?.to_string(),
                        m.get("nodeType").or_else(|| m.get("node_type"))?.as_str()?.to_string(),
                        m.get("createdAt").and_then(|c| c.as_str()).unwrap_or("").chars().take(10).collect(),
                        collapse(m.get("content")?.as_str()?),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn collapse(s: &str) -> String {
    let mut out = String::new();
    let mut prev_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out.chars().take(210).collect()
}

fn mem_line(tag: &str, m: &(String, String, String, String)) -> String {
    format!("- {} [{}, {}, tag {}]: {}", m.0, m.1, m.2, tag, m.3)
}

/// Fresh memories under these tags that the session has not been shown.
/// Corrections/decisions under `typed` handles come first.
fn fresh(st: &mut HookState, handles: &[String], typed: &[String], budget: usize) -> Vec<(String, (String, String, String, String))> {
    let mut found: Vec<(String, (String, String, String, String))> = vec![];
    for h in typed {
        for kind in ["correction", "decision"] {
            if found.len() >= budget {
                return found;
            }
            let key = format!("{h}|{kind}");
            if st.dead.contains(&key) {
                continue;
            }
            let got = fetch_tag(h, Some(kind), 2);
            if got.is_empty() {
                st.dead.push(key);
                continue;
            }
            for m in got {
                if !st.seen.contains(&m.0) && found.len() < budget {
                    st.seen.push(m.0.clone());
                    found.push((h.clone(), m));
                }
            }
        }
    }
    for h in handles {
        if found.len() >= budget {
            break;
        }
        if st.dead.contains(h) {
            continue;
        }
        let got = fetch_tag(h, None, 3);
        if got.is_empty() {
            st.dead.push(h.clone());
            continue;
        }
        for m in got {
            if !st.seen.contains(&m.0) && found.len() < budget {
                st.seen.push(m.0.clone());
                found.push((h.clone(), m));
            }
        }
    }
    found
}

// ---------------------------------------------------------------- helpers

fn repo_of(cwd: Option<&str>) -> (Option<String>, Option<String>) {
    let mut p = PathBuf::from(cwd.unwrap_or("."));
    for _ in 0..30 {
        if p.join(".git").exists() {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
            return (Some(name), p.to_str().map(|s| s.to_string()));
        }
        if !p.pop() {
            break;
        }
    }
    (None, None)
}

fn handles_for(tool: &str, ti: &Value, cwd: Option<&str>) -> (Vec<String>, Option<String>, Option<String>) {
    let (repo, top) = repo_of(cwd);
    let mut words: Vec<String> = vec![];
    for key in ["file_path", "path", "notebook_path", "target_file"] {
        if let Some(v) = ti.get(key).and_then(|v| v.as_str()) {
            let rel = match &top {
                Some(t) if v.starts_with(t.as_str()) => v.strip_prefix(t.as_str()).unwrap_or(v),
                _ => v,
            };
            let owned = rel.replace('\\', "/");
            let parts: Vec<&str> = owned.split('/').filter(|p| !p.is_empty() && *p != "..").collect();
            if let Some(last) = parts.last() {
                words.push(last.split('.').next().unwrap_or(last).to_string());
                for p in parts[..parts.len().saturating_sub(1)].iter().rev() {
                    words.push(p.to_string());
                }
            }
        }
    }
    if let Some(cmd) = ti.get("command").and_then(|c| c.as_str()) {
        let head: String = cmd.chars().take(800).collect();
        for tok in re_path_tokens(&head) {
            let parts: Vec<&str> = tok.split('/').filter(|p| !p.is_empty() && *p != "~" && *p != "." && *p != "..").collect();
            if let Some(last) = parts.last() {
                words.push(last.split('.').next().unwrap_or(last).to_string());
            }
            if parts.len() >= 2 {
                words.push(parts[parts.len() - 2].to_string());
            }
        }
    }
    if tool.starts_with("mcp__") && !tool.starts_with(V) {
        if let Some(seg) = tool.split("__").nth(1) {
            words.push(seg.to_string());
        }
    }
    let mut out: Vec<String> = vec![];
    for w in words {
        let w = w.trim_matches(|c: char| c == '-' || c == '_').to_lowercase();
        if w.len() > 2 && !DULL.contains(w.as_str()) && !out.contains(&w) {
            out.push(w);
        }
    }
    out.truncate(5);
    if let Some(r) = &repo {
        for h in [format!("codebase:{r}"), r.clone()] {
            if !out.contains(&h) {
                out.push(h);
            }
        }
    }
    (out, repo, top)
}

fn re_path_tokens(cmd: &str) -> Vec<String> {
    let rx = re(r"[\w.~-]*/[\w./~-]+");
    rx.find_iter(cmd).map(|m| m.as_str().to_string()).collect()
}

fn fam_of(cmd: &str) -> String {
    let toks: Vec<&str> = cmd.trim().split_whitespace().collect();
    match toks.len() {
        0 => String::new(),
        1 => toks[0].to_string(),
        _ => format!("{} {}", toks[0], toks[1]),
    }
}

// ---------------------------------------------------------------- outcome

enum Outcome {
    Silent,
    Ctx(String),
    Deny(String),
    Block(String),
}

fn emit(event: &str, out: &Outcome) {
    let payload = match out {
        Outcome::Silent => return,
        Outcome::Ctx(c) => json!({"hookSpecificOutput": {"hookEventName": event, "additionalContext": c}}),
        Outcome::Deny(r) => json!({"hookSpecificOutput": {"hookEventName": event, "permissionDecision": "deny", "permissionDecisionReason": r}}),
        Outcome::Block(r) => json!({"decision": "block", "reason": r}),
    };
    println!("{payload}");
}

// ---------------------------------------------------------------- gates (pure, tested)

fn critical_match(cmd: &str) -> Option<&'static str> {
    for (name, rx) in CRITICAL.iter() {
        if rx.is_match(cmd) && !(*name == "destructive-rm" && SAFE_RM.is_match(cmd)) {
            return Some(name);
        }
    }
    None
}

pub fn is_search_act(cmd: &str) -> bool {
    SEARCH_ACT.is_match(cmd)
}

fn epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// DENIAL LEDGER (self-healing piece 1): every gate refusal is appended to
/// ~/.vestige/hooks/denials.jsonl so repeated denials of the same command
/// shape can graduate into permanent law at the next SessionStart.
fn log_denial(gate: &str, cmd: &str, rule: &str) {
    use std::io::Write as _;
    let _ = std::fs::create_dir_all(state_dir());
    let flat: String = cmd.chars().take(120).filter(|c| *c != '"').collect();
    let mut line = String::new();
    line.push_str("{\"ts\":");
    line.push_str(&epoch().to_string());
    line.push_str(",\"gate\":\"");
    line.push_str(gate);
    line.push_str("\",\"key\":\"");
    line.push_str(&chash(cmd));
    line.push_str("\",\"cmd\":\"");
    line.push_str(&flat);
    line.push_str("\",\"rule\":\"");
    line.push_str(rule);
    line.push_str("\"}\n");
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(state_dir().join("denials.jsonl"))
        .and_then(|mut f| f.write_all(line.as_bytes()));
}

/// GRADUATION SCAN (self-healing piece 2): command shapes denied 3+ times
/// surface at SessionStart as prompts to smart_ingest a stop-work law.
fn graduation_lines() -> Vec<String> {
    let Ok(body) = std::fs::read_to_string(state_dir().join("denials.jsonl")) else {
        return Vec::new();
    };
    let mut counts: HashMap<String, (u32, String)> = HashMap::new();
    for ln in body.lines().rev().take(500) {
        let Ok(v) = serde_json::from_str::<Value>(ln) else {
            continue;
        };
        let (Some(k), Some(c)) = (
            v.get("key").and_then(|x| x.as_str()),
            v.get("cmd").and_then(|x| x.as_str()),
        ) else {
            continue;
        };
        let e = counts.entry(k.to_string()).or_insert((0, c.to_string()));
        e.0 += 1;
    }
    let mut hot: Vec<_> = counts.into_iter().filter(|(_, (n, _))| *n >= 3).collect();
    hot.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    hot.iter()
        .take(3)
        .map(|(_, (n, c))| {
            let short: String = c.chars().take(90).collect();
            format!("- GRADUATE TO LAW (denied {n}x): `{short}` - smart_ingest tags [stop-work], content starts with `pattern: <regex>`.")
        })
        .collect()
}

fn deny_once(st: &mut HookState, key: &str, reason: String, gate: &str, cmd: &str) -> Option<String> {
    if st.timeouts.iter().any(|k| k == key) {
        return None;
    }
    st.timeouts.push(key.to_string());
    st.denied += 1;
    log_denial(gate, cmd, gate);
    Some(reason)
}

/// The four institutional gates for a Bash command, in order.
/// Returns Some(reason) when the call must be denied.
fn institution_gates(st: &mut HookState, cmd: &str, repo: Option<&str>) -> Option<String> {
    let key = chash(cmd);
    let repo_tag = repo.unwrap_or("no-repo");

    // 2) STOP-WORK REGISTRY (runs first: a remembered fatal pattern outranks the generic card)
    if !cmd.contains(STOPWORK_OVERRIDE) {
        for rule in st.stopwork.clone() {
            if let Ok(rx) = Regex::new(&rule.pattern) {
                if rx.is_match(cmd) {
                    let k = format!("{key}|{}", rule.id);
                    st.timeouts.push(k);
                    st.denied += 1;
                    log_denial("stop-work", cmd, &rule.id);
                    return Some(format!(
                        "STOP-WORK AUTHORITY: {} forbids this pattern. {} To override DELIBERATELY, \
                         prefix the command with {STOPWORK_OVERRIDE} AND smart_ingest the override reason with tags \
                         [stop-work-override, {repo_tag}] — the override itself is a receipt. Otherwise choose a different path.",
                        rule.id,
                        rule.reason.chars().take(220).collect::<String>()
                    ));
                }
            }
        }
    } else if !st.overrides.contains(&key) {
        st.overrides.push(key.clone());
        let ids = st.stopwork.iter().map(|r| r.id.clone()).take(2).collect::<Vec<_>>().join(", ");
        st.pending.push(format!(
            "[Vestige] STOP-WORK OVERRIDE used. File the receipt now: smart_ingest why the remembered \
             fatal pattern ({ids}) was deliberately overridden, tags [stop-work-override]."
        ));
    }

    // 1) TIME-OUT CARD (skipped for an explicit STOP-WORK override: the
    // deliberate-override ceremony already paid for this command)
    if !cmd.contains(STOPWORK_OVERRIDE) {
        if let Some(name) = critical_match(cmd) {
        if let Some(why) = deny_once(
            st,
            &key,
            format!(
                "VESTIGE TIME-OUT CARD ({name}): this operation is irreversible. Before repeating it: \
                 (1) recall(handle='{name}') for remembered failures of this pattern and quote any ids; \
                 (2) state one line: blast radius + rollback path; (3) repeat this exact command. \
                 If nothing is remembered, say so, proceed, and smart_ingest the outcome with tags [time-out, {repo_tag}]. \
                 The card fires once per command."
            ),
            "timeout-card", cmd,
        ) {
            return Some(why);
        }
    }
    }

    // 3) DRAWDOWN LIMIT (mutating commands only; the caller checked)
    if st.drawdown.get(repo_tag).copied().unwrap_or(0) >= DRAWDOWN_LIMIT
        && st.dd_armed.get(repo_tag).copied().unwrap_or(false)
    {
        st.denied += 1;
        log_denial("drawdown", cmd, repo_tag);
        return Some(format!(
            "DRAWDOWN LIMIT ({repo_tag}): {} failed mutating operations this session. Mutating tools are \
             paused until the post-mortem exists: smart_ingest the AAR four — what was planned / what happened / \
             why / what changes next time — tags [post-mortem, {repo_tag}]. Then repeat this call (any Vestige \
             write lifts the pause once).",
            st.drawdown.get(repo_tag).copied().unwrap_or(0)
        ));
    }

    // 4) PIVOT GATE — THE SPIRAL KILLER (Sam's law, Oct 5 2026): after a
    //    failure, trying a DIFFERENT command family without first searching
    //    the web (or spawning a research agent) is denied once. The agent's
    //    internal knowledge is always potentially outdated; assumptions are
    //    the root cause of spiraling. Fail → search → THEN pivot.
    if !st.last_failed_fam.is_empty()
        && !st.searched_since_fail
        && epoch().saturating_sub(st.last_failed_ts) < 1_800 // 30 min window
    {
        let cur_fam = fam_of(cmd);
        if cur_fam != st.last_failed_fam && !cur_fam.is_empty() {
            if let Some(why) = deny_once(
                st,
                &format!("pivot|{key}"),
                format!(
                    "PIVOT GATE (spiral killer): your previous approach ({}) FAILED, and you are now \
                     trying a different approach ({}) based on internal knowledge that may be outdated. \
                     BEFORE pivoting: (1) use WebSearch to check the current best practice for this problem, \
                     or (2) spawn a research agent to search. NEVER assume — assumptions are the root cause \
                     of spiraling. After searching, repeat this command. \
                     (mem-0000000000008eb5: 'ALWAYS REMEMBER that the current agent's information is outdated \
                     and NEVER assume information')",
                    st.last_failed_fam, cur_fam
                ),
                "pivot", cmd,
            ) {
                return Some(why);
            }
        }
    }

    // 5) SEARCH-ACT GATE — backfill replaces grep, by act, not by program name
    if is_search_act(cmd) && !CONTEXT_TOOLS.iter().any(|t| st.last_v == *t) {
        if let Some(why) = deny_once(
            st,
            &key,
            format!(
                "SEARCH-BY-ACT GATE: this command searches file contents. Backfill replaces grep — first \
                 codebase(action='get_context', codebase='{}', repoPath='<repo root>') or recall(handle='<file or topic>'). \
                 If memory has no answer, say so in one line, then repeat this search — and bank what it finds with \
                 codebase(action='remember_pattern', ...) so the next session recalls it instead of searching again.",
                repo.unwrap_or("")
            ),
            "search-act", cmd,
        ) {
            return Some(why);
        }
    }

    None
}

/// FOQA + ripple + drawdown counters for a finished shell command.
fn foqa_scan(st: &mut HookState, cmd: &str, failed: bool, repo: Option<&str>) -> Vec<String> {
    let mut parts = vec![];
    if cmd.is_empty() {
        return parts;
    }
    let r = repo.unwrap_or("no-repo").to_string();
    let counts = st.foqa.entry(r.clone()).or_default();
    let thresholds = [("force-push", 1u32), ("secret-adjacent", 3), ("destructive", 1), ("revert", 2)];
    let mut hits: Vec<&str> = vec![];
    if FORCE_PUSH.is_match(cmd) {
        *counts.entry("force-push".to_string()).or_insert(0) += 1;
        hits.push("force-push");
    }
    if SECRET_ADJ.is_match(cmd) {
        *counts.entry("secret-adjacent".to_string()).or_insert(0) += 1;
        hits.push("secret-adjacent");
    }
    if DESTRUCTIVE.is_match(cmd) {
        *counts.entry("destructive".to_string()).or_insert(0) += 1;
        hits.push("destructive");
    }
    if REVERT_CHAIN.is_match(cmd) {
        *counts.entry("revert".to_string()).or_insert(0) += 1;
        hits.push("revert");
    }
    for profile in hits {
        let n = counts.get(profile).copied().unwrap_or(0);
        let thr = thresholds.iter().find(|(p, _)| *p == profile).map(|(_, t)| *t).unwrap_or(u32::MAX);
        if n >= thr {
            let fkey = format!("{r}|{profile}");
            if !st.foqa_filed.contains(&fkey) {
                st.foqa_filed.push(fkey);
                parts.push(format!(
                    "[Vestige FOQA] EXCEEDANCE ({r}, {profile}): {n} events. File the flight-data record now: \
                     smart_ingest the commands and what they touched, tags [foqa, exceedance, {r}]. This is the \
                     operations-safety trail."
                ));
            }
        }
    }
    let fam = fam_of(cmd);
    if !fam.is_empty() {
        if failed {
            *st.drawdown.entry(r.clone()).or_insert(0) += 1;
            st.turn.tfailed += 1;
            // PIVOT GATE: record the failed family
            if !fam.is_empty() {
                st.last_failed_fam = fam.clone();
                st.last_failed_ts = epoch();
                st.searched_since_fail = false;
            }
            if st.drawdown.get(&r).copied().unwrap_or(0) >= DRAWDOWN_LIMIT {
                st.dd_armed.insert(r.clone(), true);
            }
            if st.fams.get(&fam).copied().unwrap_or(false) && !st.rippled.contains(&fam) {
                st.rippled.push(fam.clone());
                parts.push(format!(
                    "[Vestige RIPPLE TAG] SURPRISE: '{fam}' succeeded earlier this session and now failed. \
                     Only surprises earn replay: smart_ingest what changed (the delta), tags [surprise, ripple, {r}]."
                ));
            }
        } else {
            st.fams.insert(fam, true);
        }
    }
    parts
}

// ---------------------------------------------------------------- events

fn load_stopwork(st: &mut HookState) {
    st.stopwork.clear();
    for (id, _kind, _created, content) in fetch_tag(STOPWORK_TAG, None, 8) {
        for ln in content.lines() {
            if let Some(m) = STOPWORK_LINE.captures(ln.trim()) {
                let pat = m.get(1).map(|g| g.as_str().trim().trim_matches(|c| c == '`' || c == '\'' || c == '"').to_string()).unwrap_or_default();
                if !pat.is_empty() && Regex::new(&pat).is_ok() {
                    st.stopwork.push(StopRule {
                        id: id.clone(),
                        pattern: pat,
                        reason: collapse(&content),
                    });
                    break;
                }
            }
        }
    }
}

fn start_text(repo: &Option<String>, top: &Option<String>) -> String {
    let ctx = match (repo, top) {
        (Some(r), Some(t)) => format!("{{codebase: '{r}', repoPath: '{t}'}}"),
        _ => "{}".to_string(),
    };
    format!(
        "call mcp__vestige__session_start(include_intentions=true, include_status=true, context={ctx}) \
         (load it with ToolSearch 'select:mcp__vestige__session_start,mcp__vestige__recall,mcp__vestige__smart_ingest,\
         mcp__vestige__memory,mcp__vestige__codebase,mcp__vestige__intention' if it is deferred), then \
         recall(handle='<narrow topic tag for this task>')"
    )
}

fn on_session_start(p: &Value, st: &mut HookState) -> Outcome {
    let warm = st.at > 0
        && epoch().saturating_sub(st.at) < 86_400
        && (st.timeouts.len() + st.fams.len() + st.rippled.len()) > 3;
    st.started = false;
    st.gap = 0;
    st.denied = 0;
    st.turn = Turn::default();
    let (repo, top) = repo_of(p.get("cwd").and_then(|c| c.as_str()));
    let mut lines = vec![
        format!("VESTIGE MEMORY IS ENFORCED IN THIS SESSION (source: {}).", p.get("source").and_then(|s| s.as_str()).unwrap_or("start")),
        format!("Before any other tool, {}. Other tools are refused until session_start has run.", start_text(&repo, &top)),
        "Use the tool that fits each moment: recall before any recommendation; codebase(get_context) before editing a repo; \
         smart_ingest the moment a decision, correction, verified fact or failure is established; intention for reminders; \
         causal_walk and forgotten_lesson after a failure; memory promote/demote on feedback; receipt for proof. \
         Say which memory ids you used."
            .to_string(),
    ];
    if let Some(h) = api_get("/api/health") {
        lines.push(format!(
            "Store: {} memories, server {}.",
            h.get("totalMemories").map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
            h.get("version").map(|v| v.to_string()).unwrap_or_else(|| "?".into())
        ));
        let got = fresh(st, &["sam-correction".to_string()], &[], 3);
        if !got.is_empty() {
            lines.push("Newest corrections from Sam (read in full with memory(action='get', id=...)):".to_string());
            lines.extend(got.iter().map(|(h, m)| mem_line(h, m)));
        }
        if let Some(ints) = api_get("/api/intentions") {
            let n = ints.get("intentions").and_then(|i| i.as_array()).map(|a| a.len()).unwrap_or(0);
            if n > 0 {
                lines.push(format!("Open intentions: {n}. See them with session_start(include_intentions=true)."));
            }
        }
        load_stopwork(st);
        if !st.stopwork.is_empty() {
            lines.push(format!(
                "STOP-WORK REGISTRY ARMED ({} remembered fatal patterns; overrides need {STOPWORK_OVERRIDE} + a receipt):",
                st.stopwork.len()
            ));
            lines.extend(
                st.stopwork.iter().take(4).map(|s| format!("- {} pattern /{}/: {}", s.id, s.pattern, s.reason.chars().take(120).collect::<String>())),
            );
        }
        let recs = fresh(st, &["recommendation-open".to_string()], &[], 3);
        if !recs.is_empty() {
            lines.push("OPEN RECOMMENDATIONS (unimplemented fixes; close with smart_ingest tags [recommendation-closed] once verified):".to_string());
            lines.extend(recs.iter().map(|(h, m)| mem_line(h, m)));
        }
    } else {
        lines.push("The Vestige dashboard port did not answer; memory lookups by this hook are off until it does.".to_string());
    }
    if warm {
        let dd: u32 = st.drawdown.values().sum();
        lines.push(format!(
            "SESSION CHECKPOINT (RESUMED): {} cards armed, {} families proven, {} ripples, {dd} drawdown events, {} pending receipts - continue from this position, do not restart from zero.",
            st.timeouts.len(),
            st.fams.len(),
            st.rippled.len(),
            st.pending.len()
        ));
    }
    let grads = graduation_lines();
    if !grads.is_empty() {
        lines.push(
            "SELF-HEALING - the same command shape was denied 3+ times. Graduate it into law:".to_string(),
        );
        lines.extend(grads);
    }
    if st.searches > 0 {
        let pct = st.searches_after_recall * 100 / st.searches;
        lines.push(format!(
            "SEARCH SCOREBOARD: {} searches, {}% ({} of {}) preceded by a recall, {} gap denials. Low recall share means memory is decoration.",
            st.searches,
            pct,
            st.searches_after_recall,
            st.searches,
            st.gap_denied_total
        ));
        st.searches = 0;
        st.searches_after_recall = 0;
    }
    Outcome::Ctx(lines.join("\n"))
}

fn on_prompt(p: &Value, st: &mut HookState) -> Outcome {
    st.turn = Turn::default();
    st.denied = 0;
    let prompt = p.get("prompt").and_then(|s| s.as_str()).unwrap_or("");
    let mut lines = vec![format!(
        "[Vestige] calls this session: {}; other tool calls since the last one: {} (limit {}).",
        st.vcalls, st.gap, max_gap()
    )];
    if !st.started {
        let (repo, top) = repo_of(p.get("cwd").and_then(|c| c.as_str()));
        lines.push(format!("Memory is not loaded yet: {}.", start_text(&repo, &top)));
    }
    for (rx, tool) in PROMPT_TRIGGERS.iter() {
        if rx.is_match(prompt) {
            lines.push(format!("This prompt calls for: {tool}."));
        }
    }
    Outcome::Ctx(lines.join("\n"))
}

fn act_of(ti: &Value) -> Option<&str> {
    ti.get("action").or_else(|| ti.get("mode")).and_then(|a| a.as_str())
}

fn on_pre(p: &Value, st: &mut HookState) -> Outcome {
    let tool = p.get("tool_name").and_then(|s| s.as_str()).unwrap_or("");
    let ti = p.get("tool_input").cloned().unwrap_or_else(|| json!({}));

    if tool.starts_with(V) {
        let name = tool.trim_start_matches(V);
        st.gap = 0;
        st.denied = 0;
        st.vcalls += 1;
        st.turn.vcalls += 1;
        if name == "session_start" {
            st.started = true;
            st.last_v = "session_start".to_string();
        } else {
            let act = act_of(&ti).unwrap_or("");
            st.last_v = format!("{name}:{act}");
        }
        let is_write = match name {
            "smart_ingest" => true,
            "codebase" => matches!(act_of(&ti), Some(a) if ["remember_pattern", "remember_decision", "reanchor", "ingest_repo"].contains(&a)),
            "intention" => matches!(act_of(&ti), Some(a) if ["set", "update"].contains(&a)),
            "memory" => matches!(act_of(&ti), Some(a) if ["promote", "demote", "edit", "link"].contains(&a)),
            "ghostlink" => matches!(act_of(&ti), Some(a) if a == "weave"),
            _ => false,
        };
        if is_write {
            st.turn.vwrites += 1;
            st.dd_armed.clear(); // a Vestige write lifts every repo's post-mortem pause
        }
        if name == "codebase" {
            if let Some(rp) = ti.get("repoPath").and_then(|r| r.as_str()) {
                if let (Some(r), _) = repo_of(Some(rp)) {
                    if !st.ctx_repos.iter().any(|x| *x == r) {
                        st.ctx_repos.push(r);
                    }
                }
            }
        }
        return Outcome::Silent;
    }

    // SEARCH SCOREBOARD: count searches; track whether a Vestige context
    // call immediately preceded (the recall-first contract made measurable)
    {
        let t = tool.to_lowercase();
        if t.contains("search") || t.contains("browse") || t.contains("fetch") {
            st.searches += 1;
            if CONTEXT_TOOLS.iter().any(|x| st.last_v == *x) {
                st.searches_after_recall += 1;
            }
        }
    }

    // PIVOT GATE: WebSearch, Agent spawn, or any search-flavored tool call
    // counts as "searched since the last failure" — the reality check
    if !st.last_failed_fam.is_empty() && !st.searched_since_fail {
        let t = tool.to_lowercase();
        if t.contains("search") || t.contains("agent") || t.contains("web")
            || t.contains("fetch") || t.contains("browse") {
            st.searched_since_fail = true;
        }
    }

    if FREE_TOOLS.contains(&tool) {
        return Outcome::Silent;
    }

    let cwd = p.get("cwd").and_then(|c| c.as_str());
    let (handles, repo, top) = handles_for(tool, &ti, cwd);
    let is_sub = p.get("agent_id").is_some() || p.get("agent_type").is_some();
    let cmd = ti.get("command").and_then(|c| c.as_str()).unwrap_or("").to_string();
    let reachable = api_get("/api/health").is_some();

    // ---- THE INSTITUTION (local regexes; no dashboard needed) ----
    if !is_sub && st.denied < 3 && tool == "Bash" && !cmd.is_empty() {
        if let Some(why) = institution_gates(&mut *st, &cmd, repo.as_deref()) {
            return Outcome::Deny(why);
        }
    }

    if reachable && !is_sub && st.denied < 3 {
        let why = if !st.started {
            Some(format!(
                "Vestige memory is not loaded in this session. First {}. Then repeat this call.",
                start_text(&repo, &top)
            ))
        } else if st.gap >= max_gap() {
            Some(format!(
                "{} tool calls have passed without a Vestige call. Make the one that fits what you are doing now: \
                 recall(handle='<topic tag>') for the area you are working in, codebase(action='get_context') before edits, \
                 intention(action='check'), or smart_ingest if something durable was just established (never routine progress). \
                 Then repeat this call.",
                st.gap
            ))
        } else if EDITS.contains(&tool) && repo.is_some() && !st.ctx_repos.iter().any(|r| r == repo.as_deref().unwrap_or("")) {
            Some(format!(
                "Before the first edit in the {} repository this session, call the vestige codebase tool: \
                 codebase(action='get_context', codebase='{}', repoPath='{}'), and read what it returns. Then repeat this call.",
                repo.as_deref().unwrap(),
                repo.as_deref().unwrap(),
                top.clone().unwrap_or_default()
            ))
        } else {
            None
        };
        if let Some(w) = why {
            st.denied += 1;
            return Outcome::Deny(w);
        }
    }

    st.denied = 0;
    if !is_sub {
        st.gap += 1;
        let mutating = EDITS.contains(&tool) || (tool == "Bash" && !READ_ONLY_CMD.is_match(&cmd));
        if mutating {
            st.turn.mut_n += 1;
        }
    }

    if !reachable {
        return Outcome::Silent;
    }

    // only memories under the file's own names bind an edit
    let repo_prefix = repo.clone().map(|r| format!("codebase:{r}"));
    let own: Vec<String> = handles
        .iter()
        .filter(|h| Some(h.as_str()) != repo.as_deref() && Some(h.as_str()) != repo_prefix.as_deref())
        .take(2)
        .cloned()
        .collect();
    let got = fresh(st, &handles, &own, 3);
    if got.is_empty() {
        return Outcome::Silent;
    }
    let binding: Vec<_> = got
        .iter()
        .filter(|(h, m)| own.contains(h) && (m.1 == "correction" || m.1 == "decision"))
        .cloned()
        .collect();
    if !binding.is_empty() && EDITS.contains(&tool) && !is_sub {
        st.gap = st.gap.saturating_sub(1);
        st.turn.mut_n = st.turn.mut_n.saturating_sub(1);
        for (h, m) in &got {
            if !binding.iter().any(|(bh, bm)| bh == h && bm.0 == m.0) {
                st.pending.push(mem_line(h, m));
            }
        }
        return Outcome::Deny(format!(
            "Vestige holds a correction or decision tied to this file that you have not seen this session. \
             Read it, then repeat the edit if it still stands:\n{}",
            binding.iter().map(|(h, m)| mem_line(h, m)).collect::<Vec<_>>().join("\n")
        ));
    }
    for (h, m) in &got {
        st.pending.push(mem_line(h, m));
    }
    Outcome::Silent
}

fn on_post(p: &Value, st: &mut HookState, failed_event: bool) -> Outcome {
    let tool = p.get("tool_name").and_then(|s| s.as_str()).unwrap_or("");
    let mut parts: Vec<String> = vec![];
    if !st.pending.is_empty() {
        parts.push(format!(
            "[Vestige] memories tied to that call that you had not seen this session (read in full with \
             memory(action='get', id=...) when one bears on what you are doing):\n{}",
            st.pending.drain(..).take(6).collect::<Vec<_>>().join("\n")
        ));
    }
    let ti = p.get("tool_input").cloned().unwrap_or_else(|| json!({}));
    let cmd = ti.get("command").and_then(|c| c.as_str()).unwrap_or("").to_string();
    let resp = p.get("tool_response");
    let code = p
        .get("tool_exit_code")
        .and_then(|c| c.as_i64())
        .or_else(|| resp.and_then(|r| r.get("exit_code")).and_then(|e| e.as_i64()));
    let blob = match resp {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => p.get("error").map(|e| e.to_string()).unwrap_or_default(),
    };
    let blob = blob.chars().take(6000).collect::<String>();
    let failed = failed_event
        || code.map(|c| c != 0).unwrap_or(false)
        || (!READ_ONLY_CMD.is_match(&cmd) && FAIL_RE.is_match(&blob));

    if (tool == "Bash" || tool == "PowerShell") && !cmd.is_empty() {
        let (repo, _) = repo_of(p.get("cwd").and_then(|c| c.as_str()));
        parts.extend(foqa_scan(st, &cmd, failed, repo.as_deref()));
    }
    if (tool == "Bash" || tool == "PowerShell") && failed {
        let key = chash(&cmd);
        if !st.failed.contains(&key) {
            st.failed.push(key);
            parts.push(
                "[Vestige] That command failed. Before fixing it: smart_ingest the failure as node_type='event' with the \
                 command, the error and the evidence; then causal_walk(start_points=[{kind:'logged_write', \
                 node_id:'<that id>'}], promote=false) and forgotten_lesson(failure_id='<that id>'). Use what they return, and say so."
                    .to_string(),
            );
        }
    }
    if parts.is_empty() {
        Outcome::Silent
    } else {
        Outcome::Ctx(parts.join("\n"))
    }
}

fn on_stop(p: &Value, st: &mut HookState) -> Outcome {
    if p.get("stop_hook_active").and_then(|b| b.as_bool()).unwrap_or(false) || st.turn.asked {
        return Outcome::Silent;
    }
    let mut tail: Vec<String> = vec![];
    if st.turn.tfailed > 0 {
        tail.push(
            "HINDSIGHT RELABEL: this turn failed at least once. If a failure produced a valuable byproduct (a harness, \
             a repro, a route), re-index it under its ACHIEVED goal: smart_ingest with tags [hindsight, <repo>] — the most \
             expensive exploration must not be wasted."
                .to_string(),
        );
    }
    if st.turn.mut_n >= 4 {
        tail.push(
            "SBAR HANDOFF: close the turn with a transfer-of-responsibility packet — Situation / Background (decisions \
             with ids) / Assessment (fragile spots) / Recommendation (do, don't, stop-work) — one smart_ingest, tags \
             [sbar, <repo>]. The next session bootstraps from this packet, not from a search."
                .to_string(),
        );
    }
    if st.started && st.turn.mut_n >= 3 && st.turn.vwrites == 0 {
        st.turn.asked = true;
        return Outcome::Block(format!(
            "This turn changed things ({} edits or commands) and saved nothing to Vestige. Save what it established now: \
             one smart_ingest per decision, correction, verified fact, failure or milestone, with the project tag and a narrow \
             topic tag, and codebase(action='remember_decision') for a design choice in code. Quote the ids. If nothing durable \
             was established, say that in one line and stop. {}",
            st.turn.mut_n,
            tail.join(" ")
        ));
    }
    if !tail.is_empty() && st.started {
        st.turn.asked = true;
        return Outcome::Block(format!("Before this turn ends: {}", tail.join(" ")));
    }
    Outcome::Silent
}

// ---------------------------------------------------------------- entry

/// Read one host event from stdin, enforce, answer on stdout. Fail-open: any
/// error prints nothing and exits 0 so a hook defect can never block a tool.
pub fn run_hook() -> i32 {
    // MASTER OFF SWITCH: while ~/.claude/hooks/VESTIGE_MEMORY_OFF exists the
    // Institution stays silent - one touch file governs python and Rust alike.
    if let Ok(home) = std::env::var("HOME") {
        if std::path::Path::new(&home)
            .join(".claude/hooks/VESTIGE_MEMORY_OFF")
            .exists()
        {
            return 0;
        }
    }
    let mut raw = String::new();
    if std::io::stdin().read_to_string(&mut raw).is_err() {
        return 0;
    }
    let Ok(p) = serde_json::from_str::<Value>(&raw) else {
        return 0;
    };
    if p.get("skip").is_some() {
        return 0;
    }
    let event = p.get("hook_event_name").and_then(|e| e.as_str()).unwrap_or("").to_string();
    let sid = p.get("session_id").and_then(|s| s.as_str()).unwrap_or("none").to_string();
    let mut st = load_state(&sid);
    let out = match event.as_str() {
        "SessionStart" => on_session_start(&p, &mut st),
        "UserPromptSubmit" => on_prompt(&p, &mut st),
        "PreToolUse" => on_pre(&p, &mut st),
        "PostToolUse" => on_post(&p, &mut st, false),
        "PostToolUseFailure" => on_post(&p, &mut st, true),
        "Stop" => on_stop(&p, &mut st),
        _ => Outcome::Silent,
    };
    emit(&event, &out);
    st.at = epoch();
    save_state(&sid, &mut st);
    0
}

/// Point a host's hook config at this binary. v1: ZCode (the active host).
pub fn hooks_install(host: &str) -> i32 {
    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "vestige".to_string());
    let cmd = format!("{exe} hook");
    match host {
        "zcode" => {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            let cfg_path = PathBuf::from(&home).join(".zcode/cli/config.json");
            let Ok(raw) = std::fs::read_to_string(&cfg_path) else {
                eprintln!("zcode config not found at {}", cfg_path.display());
                return 1;
            };
            let Ok(mut cfg) = serde_json::from_str::<Value>(&raw) else {
                eprintln!("zcode config is not valid JSON");
                return 1;
            };
            let backup = cfg_path.with_extension("json.bak-institution");
            if std::fs::write(&backup, &raw).is_ok() {
                eprintln!("backup: {}", backup.display());
            }
            let events = ["SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop"];
            let mut replaced = 0;
            if let Some(hooks) = cfg.get_mut("hooks").and_then(|h| h.as_object_mut()) {
                hooks.insert("enabled".into(), json!(true));
                if let Some(ev) = hooks.get_mut("events").and_then(|e| e.as_object_mut()) {
                    for name in events {
                        if let Some(entries) = ev.get_mut(name).and_then(|v| v.as_array_mut()) {
                            for entry in entries.iter_mut() {
                                if let Some(hlist) = entry.get_mut("hooks").and_then(|h| h.as_array_mut()) {
                                    for h in hlist.iter_mut() {
                                        let c = h.get("command").and_then(|c| c.as_str()).unwrap_or("").to_string();
                                        if c.contains("vestige-memory.py") {
                                            if let Some(obj) = h.as_object_mut() {
                                                obj.insert("command".into(), json!(cmd.clone()));
                                                obj.insert("timeout".into(), json!(10));
                                                replaced += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if replaced == 0 {
                eprintln!("no python hook entries found to replace (already installed?)");
                return 0;
            }
            match serde_json::to_string_pretty(&cfg) {
                Ok(body) => {
                    if std::fs::write(&cfg_path, body).is_err() {
                        eprintln!("failed to write config");
                        return 1;
                    }
                }
                Err(_) => return 1,
            }
            println!("ZCode hooks repointed to the Rust Institution ({replaced} entries -> {cmd}). ZCode hot-reloads the config.");
            0
        }
        other => {
            eprintln!("host '{other}' install not wired yet (zcode only in v1)");
            1
        }
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn st() -> HookState {
        HookState::default()
    }

    #[test]
    fn timeout_card_denies_once_then_passes() {
        let mut s = st();
        let cmd = "git push --force origin main";
        let first = institution_gates(&mut s, cmd, Some("vestige"));
        assert!(first.is_some(), "force push must be denied first");
        assert!(first.unwrap().contains("TIME-OUT CARD"));
        let second = institution_gates(&mut s, cmd, Some("vestige"));
        assert!(second.is_none(), "the repeat passes (block-then-insist)");
    }

    #[test]
    fn force_with_lease_is_not_critical() {
        let mut s = st();
        assert!(institution_gates(&mut s, "git push --force-with-lease origin main", Some("r")).is_none());
    }

    #[test]
    fn safe_rm_outside_gate() {
        let mut s = st();
        assert!(institution_gates(&mut s, "rm -rf /tmp/scratch", None).is_none(), "tmp rm is exempt");
        assert!(institution_gates(&mut s, "rm -rf ~/Documents", None).is_some(), "home rm is critical");
    }

    #[test]
    fn stopwork_blocks_then_override_passes_and_demands_receipt() {
        let mut s = st();
        s.stopwork.push(StopRule {
            id: "mem-X".into(),
            pattern: r"cargo\s+publish".into(),
            reason: "publishing without Sam's GO burned a version number once".into(),
        });
        let denied = institution_gates(&mut s, "cargo publish", Some("vestige"));
        assert!(denied.as_deref().unwrap_or("").contains("STOP-WORK AUTHORITY"));
        // same command again still denied (stop-work is not deny-once)
        assert!(institution_gates(&mut s, "cargo publish", Some("vestige")).is_some());
        let over = format!("{STOPWORK_OVERRIDE} cargo publish");
        assert!(institution_gates(&mut s, &over, Some("vestige")).is_none(), "override passes");
        assert!(s.pending.iter().any(|p| p.contains("STOP-WORK OVERRIDE")), "override demands its receipt");
    }

    #[test]
    fn drawdown_gates_after_limit_and_vwrite_clears() {
        let mut s = st();
        for _ in 0..DRAWDOWN_LIMIT {
            foqa_scan(&mut s, "cargo build --release", true, Some("vestige"));
        }
        assert!(s.dd_armed.get("vestige").copied().unwrap_or(false), "drawdown armed at limit");
        let denied = institution_gates(&mut s, "cargo build --release", Some("vestige"));
        assert!(denied.as_deref().unwrap_or("").contains("DRAWDOWN LIMIT"));
        s.dd_armed.clear(); // what a Vestige write does
        assert!(institution_gates(&mut s, "cargo build --release", Some("vestige")).is_none());
    }

    #[test]
    fn search_act_requires_recall_first() {
        let mut s = st();
        s.started = true;
        let grep = "grep -rn TODO src/";
        assert!(institution_gates(&mut s, grep, Some("vestige")).unwrap_or_default().contains("SEARCH-BY-ACT"));
        let py = "python3 -c 'import re; re.search(open(\"f\").read())'";
        assert!(institution_gates(&mut s, py, Some("vestige")).unwrap_or_default().contains("SEARCH-BY-ACT"), "python content search is a grep");
        // deny-once per command: repeat passes
        assert!(institution_gates(&mut s, grep, Some("vestige")).is_none());
        // a fresh search after recall is allowed without denial
        s.last_v = "recall".into();
        assert!(institution_gates(&mut s, "rg pattern file.rs", Some("vestige")).is_none());
        // but session_start does NOT satisfy the gate
        s.last_v = "session_start".into();
        let denied = institution_gates(&mut s, "rg pattern other.rs", Some("vestige"));
        assert!(denied.is_some(), "only recall/codebase:get_context satisfy the search gate");
    }

    #[test]
    fn foqa_exceedance_fires_at_threshold_once() {
        let mut s = st();
        let secret = "cat ~/.ssh/kaggle.json";
        for _ in 0..2 {
            let parts = foqa_scan(&mut s, secret, false, Some("vestige"));
            assert!(!parts.iter().any(|p| p.contains("FOQA")), "below threshold is silent");
        }
        let third = foqa_scan(&mut s, secret, false, Some("vestige"));
        assert!(third.iter().any(|p| p.contains("FOQA") && p.contains("secret-adjacent")), "third touch files");
        let fourth = foqa_scan(&mut s, secret, false, Some("vestige"));
        assert!(!fourth.iter().any(|p| p.contains("secret-adjacent")), "already filed for this count");
    }

    #[test]
    fn ripple_tags_surprise_only() {
        let mut s = st();
        foqa_scan(&mut s, "cargo test vestige", false, Some("vestige"));
        let parts = foqa_scan(&mut s, "cargo test vestige", true, Some("vestige"));
        assert!(parts.iter().any(|p| p.contains("RIPPLE TAG")), "success then failure is a surprise");
        let first_fail = foqa_scan(&mut s, "make all", true, Some("vestige"));
        assert!(!first_fail.iter().any(|p| p.contains("RIPPLE TAG")), "failure without prior success is not a ripple");
    }

    #[test]
    fn state_roundtrip_trims() {
        let mut s = st();
        s.seen = (0..500).map(|i| format!("m{i}")).collect();
        let sid = "unit-test-session";
        save_state(sid, &mut s);
        let loaded = load_state(sid);
        assert_eq!(loaded.seen.len(), 400, "seen trimmed to 400");
        let _ = std::fs::remove_file(state_path(sid));
    }

    #[test]
    fn critical_catches_the_family() {
        for cmd in [
            "git reset --hard HEAD~3",
            "kubectl delete ns prod",
            "kaggle kernels push -p dir",
            "docker system prune --volumes",
            "crontab -r",
            "gh repo delete samvallad33/vestige",
        ] {
            assert!(critical_match(cmd).is_some(), "must catch: {cmd}");
        }
        for cmd in ["git push origin main", "cargo build --release", "ls -la"] {
            assert!(critical_match(cmd).is_none(), "must not catch: {cmd}");
        }
    }

    #[test]
    fn chash_is_stable() {
        assert_eq!(chash("git push --force origin main"), chash("git push --force origin main"));
        assert_ne!(chash("git push --force origin main"), chash("git push --force origin dev"));
    }

    // ---------- BRUTAL DEPTH: exhaustive per-family and per-form coverage ----------

    fn off() {
        super::TEST_OFFLINE.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// VESTIGE_HOOK_STATE_DIR is process-global and cargo runs unit tests
    /// concurrently, so every state-dir-sensitive test is serialized behind
    /// one lock: without it, test B's set_var redirects test A's ledger
    /// writes into B's directory mid-test (the no-dependency serial-test
    /// pattern; serial_test crate would add a dep the Institution forgoes).
    static STATE_DIR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_state_dir<T>(f: impl FnOnce(&std::path::Path) -> T) -> T {
        let _guard = STATE_DIR_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "inst-ledger-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        unsafe { std::env::set_var("VESTIGE_HOOK_STATE_DIR", &dir) };
        let out = f(&dir);
        unsafe { std::env::remove_var("VESTIGE_HOOK_STATE_DIR") };
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    fn pre(tool: &str, cmd: &str, sub: bool) -> (Value, HookState) {
        let mut p = json!({"hook_event_name":"PreToolUse","session_id":"unit","cwd":"/tmp",
                           "tool_name":tool,"tool_input":{"command":cmd}});
        if sub { p["agent_id"] = json!("agent-1"); }
        (p, st())
    }

    #[test]
    fn critical_every_family_positive_and_negative() {
        let pos = [
            ("git push -f origin m", "bare -f"),
            ("git reset --hard", "hard reset"),
            ("git reflog expire --expire=now --all", "reflog expire"),
            ("git filter-branch --env-filter x", "filter-branch"),
            ("git clean -fd", "clean -fd"),
            ("git branch -D feature", "branch -D"),
            ("rm -r ~/proj", "rm -r"),
            ("shred secret.key", "shred"),
            ("truncate -s 0 db.sqlite", "truncate"),
            ("dd if=img of=/dev/disk3", "dd"),
            ("mkfs.ext4 /dev/sda1", "mkfs"),
            ("diskutil eraseDisk JHFS+ X disk2", "eraseDisk"),
            ("DROP TABLE users;", "drop table"),
            ("drop database prod", "drop database"),
            ("kaggle kernels push -p pkg", "kaggle push"),
            ("cargo publish --dry-run", "cargo publish"),
            ("npm unpublish pkg --force", "npm unpublish"),
            ("twine upload dist/*", "twine"),
            ("gh repo delete me/it", "gh repo delete"),
            ("terraform destroy -auto-approve", "terraform"),
            ("pulumi destroy --yes", "pulumi"),
            ("docker system prune -a", "docker prune"),
            ("kubectl delete pod api", "kubectl delete"),
            ("aws s3 rb s3://b --force", "s3 rb"),
            ("gcloud projects delete x", "gcloud delete"),
            ("az group delete -n x", "az group"),
            ("helm uninstall rel", "helm"),
            ("crontab -r", "crontab"),
            ("chmod -R 000 ~", "chmod 000"),
            ("chown -R ~ user", "chown -R"),
        ];
        for (cmd, why) in pos {
            assert!(critical_match(cmd).is_some(), "must catch ({why}): {cmd}");
        }
        let neg = [
            "git push origin main",
            "git push --force-with-lease origin main",
            "git reset --soft HEAD~1",
            "git branch -d merged_feature",
            "git log --oneline",
            "cat /tmp/notes",
            "rm -rf /tmp/scratch/build",
            "cargo build --release",
            "cargo test",
            "kubectl get pods",
            "docker ps",
            "aws s3 ls",
            "helm list",
            "crontab -l",
            "chmod +x script.sh",
            "ls -la",
        ];
        for cmd in neg {
            assert!(critical_match(cmd).is_none(), "must NOT catch: {cmd}");
        }
    }

    #[test]
    fn safe_rm_exemption_matrix() {
        for safe in ["rm -rf /tmp/x", "rm -rf ./node_modules", "rm -rf target/debug",
                     "rm -rf build/", "rm -rf /var/folders/zz", "rm -rf scratchpad/junk"] {
            assert!(institution_gates(&mut st(), safe, None).is_none(), "exempt: {safe}");
        }
        for hot in ["rm -rf ~/Documents", "rm -rf /Users/sam/data", "rm -rf src/"] {
            assert!(institution_gates(&mut st(), hot, None).is_some(), "critical: {hot}");
        }
    }

    #[test]
    fn search_act_every_form_and_clean_passes() {
        let acts = [
            r#"grep -n "x" file"#, "egrep foo *.py", "fgrep literal f", "rg pattern src/",
            "ripgrep -i todo", "git grep needle", "awk '/err/ {print}' f", "sed -n '/a/,/b/p' f",
            "perl -ne 'print if /x/' f", "python3 -c 'import re; re.findall(p, t)'",
            r#"python -c 'print(open("f").read())'"#, "find . -exec grep -q x {} ;",
        ];
        for a in acts {
            assert!(is_search_act(a), "search act: {a}");
        }
        let clean = ["cat file.txt", "ls -la", "echo hi", "cargo build", "git status",
                     "python3 script.py", "cp a b", "mkdir d"];
        for c in clean {
            assert!(!is_search_act(c), "NOT a search act: {c}");
        }
    }

    #[test]
    fn search_gate_denies_then_recalls_then_allows_new_search() {
        off();
        let mut s = st();
        s.started = true;
        s.last_v = "session_start".into();
        let d1 = institution_gates(&mut s, "grep -rn x src/", Some("r"));
        assert!(d1.unwrap_or_default().contains("SEARCH-BY-ACT"));
        // recall satisfies: brand-new command allowed, no denial consumed
        s.last_v = "recall".into();
        assert!(institution_gates(&mut s, "rg needle crates/", Some("r")).is_none());
        // codebase:get_context also satisfies
        s.last_v = "codebase:get_context".into();
        assert!(institution_gates(&mut s, "awk '/x/' f", Some("r")).is_none());
        // codebase:remember_pattern does NOT satisfy
        s.last_v = "codebase:remember_pattern".into();
        assert!(institution_gates(&mut s, "git grep z", Some("r")).is_some());
    }

    #[test]
    fn stopwork_first_rule_wins_and_invalid_patterns_skipped() {
        off();
        let mut s = st();
        s.stopwork.push(StopRule { id: "bad".into(), pattern: "([".into(), reason: "uncompilable".into() });
        s.stopwork.push(StopRule { id: "mem-good".into(), pattern: r"kaggle\s+kernels(?:s)?\s+push".into(), reason: "never submit without GO".into() });
        let d = institution_gates(&mut s, "kaggle kernels push -p x", Some("arc"));
        assert!(d.unwrap_or_default().contains("mem-good"), "valid rule fires, invalid skipped");
        // not deny-once: still denied on repeat
        assert!(institution_gates(&mut s, "kaggle kernels push -p x", Some("arc")).is_some());
        // unrelated command unaffected
        assert!(institution_gates(&mut s, "cargo build", Some("arc")).is_none());
    }

    #[test]
    fn stopwork_override_receipt_fires_once() {
        off();
        let mut s = st();
        s.stopwork.push(StopRule { id: "m1".into(), pattern: "forbidden".into(), reason: "r".into() });
        let over = "OVERRIDE-STOPWORK forbidden thing";
        assert!(institution_gates(&mut s, over, None).is_none());
        assert_eq!(s.pending.len(), 1, "receipt demanded");
        assert!(institution_gates(&mut s, over, None).is_none());
        assert_eq!(s.pending.len(), 1, "receipt demanded exactly once per command");
    }

    #[test]
    fn drawdown_boundary_and_rearm() {
        off();
        let mut s = st();
        for _ in 0..(DRAWDOWN_LIMIT - 1) {
            foqa_scan(&mut s, "make build", true, Some("r1"));
            assert!(!s.dd_armed.get("r1").copied().unwrap_or(false), "armed only at the limit");
        }
        foqa_scan(&mut s, "make build", true, Some("r1"));
        assert!(s.dd_armed.get("r1").copied().unwrap_or(false), "armed at the limit");
        // repo isolation: r2 untouched
        assert!(institution_gates(&mut s, "make build", Some("r2")).is_none(), "other repo unaffected");
        // read-only commands are NOT drawdown-gated at the gates level (caller passes mutating only) — but the gate itself only sees what it sees
        // vestige write lifts: simulate
        s.dd_armed.clear();
        assert!(institution_gates(&mut s, "make build", Some("r1")).is_none(), "write lifted");
        // rearm: 5 more failures re-arm
        for _ in 0..5 { foqa_scan(&mut s, "make build", true, Some("r1")); }
        assert!(s.dd_armed.get("r1").copied().unwrap_or(false), "re-arms after more failures");
    }

    #[test]
    fn foqa_each_profile_threshold_and_isolation() {
        off();
        let mut s = st();
        // destructive fires at 1
        let p1 = foqa_scan(&mut s, "rsync -a --delete src/ dst/", false, Some("ra"));
        assert!(p1.iter().any(|x| x.contains("destructive")), "destructive threshold 1");
        // force-push fires at 1
        let p2 = foqa_scan(&mut s, "git push --force origin x", false, Some("ra"));
        assert!(p2.iter().any(|x| x.contains("force-push")));
        // revert needs 2
        let p3a = foqa_scan(&mut s, "git revert abc1234", false, Some("ra"));
        assert!(!p3a.iter().any(|x| x.contains("revert")));
        let p3b = foqa_scan(&mut s, "git revert def5678", false, Some("ra"));
        assert!(p3b.iter().any(|x| x.contains("revert")), "revert threshold 2");
        // once per profile: third revert silent
        let p3c = foqa_scan(&mut s, "git revert aaa0000", false, Some("ra"));
        assert!(!p3c.iter().any(|x| x.contains("revert")));
        // repo isolation: same secret in rb starts from zero
        let rb = foqa_scan(&mut s, "cat ~/.env", false, Some("rb"));
        assert!(!rb.iter().any(|x| x.contains("secret-adjacent")), "rb first touch silent");
    }

    #[test]
    fn ripple_once_per_family_and_two_token_fam() {
        off();
        let mut s = st();
        foqa_scan(&mut s, "cargo test -p x", false, Some("r"));
        foqa_scan(&mut s, "cargo test -p y", false, Some("r")); // same 2-token fam
        let p = foqa_scan(&mut s, "cargo test -p z", true, Some("r"));
        assert!(p.iter().any(|x| x.contains("RIPPLE TAG")), "fam = first two tokens");
        let again = foqa_scan(&mut s, "cargo test -p w", true, Some("r"));
        assert!(!again.iter().any(|x| x.contains("RIPPLE")), "tagged once per family");
    }

    #[test]
    fn on_pre_vestige_bookkeeping_sets_last_v_and_clears_drawdown() {
        off();
        let mut s = st();
        let p = json!({"hook_event_name":"PreToolUse","session_id":"u","cwd":"/tmp",
                       "tool_name":"mcp__vestige__session_start","tool_input":{}});
        match on_pre(&p, &mut s) { Outcome::Silent => {}, _ => panic!("vestige call must be silent") }
        assert!(s.started && s.last_v == "session_start" && s.vcalls == 1 && s.gap == 0);
        s.dd_armed.insert("vestige".into(), true);

        let p2 = json!({"tool_name":"mcp__vestige__recall","tool_input":{"handle":"x"}});
        on_pre(&p2, &mut s);
        assert!(s.last_v.starts_with("recall"), "recall recorded as context tool");

        let p3 = json!({"tool_name":"mcp__vestige__codebase","tool_input":{"action":"get_context"}});
        on_pre(&p3, &mut s);
        assert!(s.last_v.starts_with("codebase:get_context"));

        let p4 = json!({"tool_name":"mcp__vestige__smart_ingest","tool_input":{"content":"x"}});
        on_pre(&p4, &mut s);
        assert_eq!(s.turn.vwrites, 1, "smart_ingest counted as a write");
        assert!(s.dd_armed.is_empty(), "a WRITE lifts drawdown; session_start alone does not");
    }

    #[test]
    fn on_pre_subagent_calls_do_not_count() {
        off();
        let mut s = st();
        s.started = true;
        let p = json!({"hook_event_name":"PreToolUse","session_id":"u","cwd":"/tmp",
                       "tool_name":"Bash","tool_input":{"command":"echo hi"},"agent_id":"sub-1"});
        match on_pre(&p, &mut s) { Outcome::Silent => {}, _ => panic!("offline subagent passes") }
        assert_eq!(s.gap, 0, "subagent does not consume the main agent's gap");
        assert_eq!(s.turn.mut_n, 0, "subagent does not count as the turn's changes");
    }

    #[test]
    fn on_pre_free_tool_passes_silent() {
        off();
        let mut s = st();
        let p = json!({"hook_event_name":"PreToolUse","session_id":"u","cwd":"/tmp",
                       "tool_name":"ToolSearch","tool_input":{"query":"x"}});
        assert!(matches!(on_pre(&p, &mut s), Outcome::Silent));
    }

    #[test]
    fn on_pre_offline_institution_gates_still_enforce() {
        off();
        let mut s = st();
        s.started = true;
        let (p, _) = pre("Bash", "git push --force origin m", false);
        match on_pre(&p, &mut s) {
            Outcome::Deny(r) => assert!(r.contains("TIME-OUT CARD")),
            _ => panic!("institution gate must fire even offline"),
        }
    }

    #[test]
    fn on_post_failure_demands_ingest_once_per_command() {
        off();
        let mut s = st();
        let p = json!({"hook_event_name":"PostToolUse","session_id":"u","cwd":"/tmp",
                       "tool_name":"Bash","tool_input":{"command":"cargo build"},
                       "tool_exit_code":1, "tool_response":"error[E0308]: mismatched types"});
        match on_post(&p, &mut s, false) {
            Outcome::Ctx(c) => assert!(c.contains("smart_ingest the failure")),
            _ => panic!("failure must demand the ingest"),
        }
        match on_post(&p, &mut s, false) {
            Outcome::Ctx(c) => assert!(!c.contains("smart_ingest the failure"), "once per command"),
            Outcome::Silent => {},
            _ => panic!(),
        }
    }

    #[test]
    fn on_post_failed_event_flag_always_treated_as_failure() {
        off();
        let mut s = st();
        let p = json!({"hook_event_name":"PostToolUseFailure","session_id":"u","cwd":"/tmp",
                       "tool_name":"Bash","tool_input":{"command":"make all"},
                       "error":"spawn failed"});
        match on_post(&p, &mut s, true) {
            Outcome::Ctx(c) => assert!(c.contains("smart_ingest the failure")),
            _ => panic!("PostToolUseFailure must be a failure"),
        }
    }

    #[test]
    fn on_stop_save_guard_blocks_then_asks_once() {
        off();
        let mut s = st();
        s.started = true;
        s.turn.mut_n = 4;
        s.turn.vwrites = 0;
        let p = json!({"hook_event_name":"Stop","session_id":"u"});
        match on_stop(&p, &mut s) {
            Outcome::Block(r) => assert!(r.contains("saved nothing to Vestige")),
            _ => panic!("save guard must block"),
        }
        // asked-once: second Stop silent
        assert!(matches!(on_stop(&p, &mut s), Outcome::Silent));
    }

    #[test]
    fn on_stop_sbar_and_hindsight_tails_compose() {
        off();
        let mut s = st();
        s.started = true;
        s.turn.mut_n = 5;
        s.turn.tfailed = 2;
        let p = json!({"hook_event_name":"Stop","session_id":"u"});
        match on_stop(&p, &mut s) {
            Outcome::Block(r) => {
                assert!(r.contains("HINDSIGHT RELABEL"));
                assert!(r.contains("SBAR HANDOFF"));
            }
            _ => panic!("tails must ride the save-guard block"),
        }
    }

    #[test]
    fn on_stop_written_turn_with_failures_still_gets_hindsight() {
        off();
        let mut s = st();
        s.started = true;
        s.turn.mut_n = 3;
        s.turn.vwrites = 2;
        s.turn.tfailed = 1;
        let p = json!({"hook_event_name":"Stop","session_id":"u"});
        match on_stop(&p, &mut s) {
            Outcome::Block(r) => assert!(r.starts_with("Before this turn ends")),
            _ => panic!("hindsight-only block expected"),
        }
    }

    #[test]
    fn on_stop_hook_active_passes_through() {
        off();
        let mut s = st();
        s.started = true;
        s.turn.mut_n = 9;
        let p = json!({"hook_event_name":"Stop","session_id":"u","stop_hook_active":true});
        assert!(matches!(on_stop(&p, &mut s), Outcome::Silent), "loop breaker honored");
    }

    #[test]
    fn prompt_triggers_fire_on_the_law_phrases() {
        off();
        let mut s = st();
        let cases = [
            ("we nearly lost the database", "NEAR-MISS"),
            ("deploy the migration tonight", "PREREGISTER"),
            ("remember this decision", "recall the topic tag"),
            ("ship it to prod", "PREREGISTER"),
        ];
        for (prompt, expect) in cases {
            let p = json!({"hook_event_name":"UserPromptSubmit","session_id":"u","prompt":prompt});
            match on_prompt(&p, &mut s) {
                Outcome::Ctx(c) => assert!(c.contains(expect), "prompt '{prompt}' must trigger '{expect}'"),
                _ => panic!("prompt must produce context"),
            }
        }
    }

    #[test]
    fn denial_ledger_appends_and_graduation_promotes_at_three() {
        with_state_dir(|dir| {
            let mut s = st();
            let cmd = "cargo build --release --features x";
            let why = deny_once(&mut s, &chash(cmd), "test reason".into(), "gap", cmd);
            assert!(why.is_some());
            let again = deny_once(&mut s, &chash(cmd), "test reason".into(), "gap", cmd);
            assert!(again.is_none(), "deny-once: the repeat passes");
            let body = std::fs::read_to_string(dir.join("denials.jsonl"))
                .expect("deny_once must have written the ledger line");
            assert_eq!(body.lines().count(), 1, "one ledger line per denial");
            log_denial("gap", cmd, "contract");
            log_denial("gap", cmd, "contract");
            assert!(body.contains("\"gate\":\"gap\""));
            let grads = graduation_lines();
            assert_eq!(grads.len(), 1, "exactly one command shape graduates");
            assert!(grads[0].contains("GRADUATE TO LAW"));
            assert!(grads[0].contains("cargo build"));
        });
    }

    #[test]
    fn graduation_needs_three_of_the_same_shape() {
        with_state_dir(|_dir| {
            log_denial("gap", "cargo build", "contract");
            log_denial("gap", "cargo build", "contract");
            log_denial("gap", "npm install x", "contract");
            assert!(
                graduation_lines().is_empty(),
                "two of one shape plus one of another does not graduate"
            );
        });
    }

    #[test]
    fn session_start_resume_checkpoint_and_scoreboard() {
        off();
        with_state_dir(|_dir| {
            let mut s = st();
            s.at = epoch();
            s.timeouts = vec!["a".into(), "b".into(), "c".into(), "d".into()];
            s.searches = 10;
            s.searches_after_recall = 3;
            let p = json!({"hook_event_name":"SessionStart","session_id":"u","cwd":"/tmp","source":"startup"});
            match on_session_start(&p, &mut s) {
                Outcome::Ctx(c) => {
                    assert!(c.contains("SESSION CHECKPOINT (RESUMED)"), "warm state resumes: {c}");
                    assert!(c.contains("SEARCH SCOREBOARD"), "scoreboard reports: {c}");
                    assert!(c.contains("30%"), "3 of 10 searches = 30%: {c}");
                }
                _ => panic!("session start must emit context"),
            }
            assert_eq!(s.searches, 0, "scoreboard resets after reporting");
        });
    }

    #[test]
    fn cold_state_gets_no_resume_checkpoint() {
        off();
        with_state_dir(|_dir| {
            let mut s = st();
            let p = json!({"hook_event_name":"SessionStart","session_id":"u","cwd":"/tmp","source":"startup"});
            match on_session_start(&p, &mut s) {
                Outcome::Ctx(c) => assert!(!c.contains("SESSION CHECKPOINT"), "cold start: {c}"),
                _ => panic!("must emit context"),
            }
        });
    }

    #[test]
    fn session_start_offline_emits_contract_and_offline_notice() {
        off();
        let mut s = st();
        let p = json!({"hook_event_name":"SessionStart","session_id":"u","cwd":"/tmp","source":"startup"});
        match on_session_start(&p, &mut s) {
            Outcome::Ctx(c) => {
                assert!(c.contains("VESTIGE MEMORY IS ENFORCED"));
                assert!(c.contains("did not answer"));
            }
            _ => panic!("session start must emit the contract"),
        }
    }

    #[test]
    fn stopwork_line_parser_shapes() {
        for (line, want) in [
            (r"pattern: git\s+push", Some(r"git\s+push")),
            ("pattern: `x`", Some("x")),
            ("Pattern: 'y'", Some("y")),
            ("nothing here", None),
        ] {
            let got = STOPWORK_LINE.captures(line).and_then(|c| c.get(1)).map(|m| m.as_str().to_string());
            assert_eq!(got.as_deref(), want, "line: {line}");
        }
    }

    #[test]
    fn handles_extraction_from_paths_and_commands() {
        let ti = json!({"file_path":"/x/proj/src/operator_gate.py","command":"cat build/mesh_loader.rs"});
        let (h, _, _) = handles_for("Edit", &ti, None);
        assert!(h.contains(&"operator_gate".to_string()), "file stem extracted: {h:?}");
        assert!(h.contains(&"mesh_loader".to_string()), "command path token extracted: {h:?}");
        assert!(!h.iter().any(|x| x == "src" || x == "build"), "dull dir names filtered");
    }

    #[test]
    fn repo_of_walks_up_to_git() {
        let root = std::env::temp_dir().join(format!("repoof-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("crates/x/src")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let name = root.file_name().unwrap().to_str().unwrap().to_string();
        let (r, t) = repo_of(root.join("crates/x/src").to_str());
        assert_eq!(r.as_deref(), Some(name.as_str()));
        assert!(t.unwrap().ends_with(&name));
        let (r2, _) = repo_of(Some("/tmp"));
        assert!(r2.is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fam_of_two_tokens() {
        assert_eq!(fam_of("cargo build --release"), "cargo build");
        assert_eq!(fam_of("make"), "make");
        assert_eq!(fam_of(""), "");
    }

    #[test]
    fn urlencode_percent_encodes() {
        assert_eq!(urlencode("a b"), "a%20b");
        assert_eq!(urlencode("sam-correction"), "sam-correction");
        assert_eq!(urlencode("codex:tag"), "codex%3Atag");
    }

    #[test]
    fn state_serde_roundtrip_all_fields() {
        let mut s = st();
        s.started = true; s.gap = 1; s.last_v = "recall".into();
        s.stopwork.push(StopRule { id: "m".into(), pattern: "p".into(), reason: "r".into() });
        s.drawdown.insert("repo".into(), 7);
        s.turn = Turn { mut_n: 3, vwrites: 1, vcalls: 2, tfailed: 4, asked: false };
        let sid = "serde-roundtrip";
        save_state(sid, &mut s);
        let back = load_state(sid);
        assert!(back.started && back.gap == 1 && back.last_v == "recall");
        assert_eq!(back.stopwork.len(), 1);
        assert_eq!(back.drawdown.get("repo"), Some(&7));
        assert_eq!((back.turn.mut_n, back.turn.tfailed), (3, 4));
        let _ = std::fs::remove_file(state_path(sid));
    }

    #[test]
    fn pivot_gate_denies_different_family_after_failure() {
        let mut s = st();
        s.started = true;
        // Simulate a failed cargo build
        foqa_scan(&mut s, "cargo build --release", true, Some("vestige"));
        assert_eq!(s.last_failed_fam, "cargo build");
        assert!(!s.searched_since_fail);
        // Trying a DIFFERENT family without searching → denied
        let d = institution_gates(&mut s, "make all", Some("vestige"));
        assert!(d.as_deref().unwrap_or("").contains("PIVOT GATE"), "pivot must fire: {d:?}");
        assert!(d.unwrap_or_default().contains("WebSearch"));
        // Same family → NOT denied by pivot (retrying is not pivoting)
        let same = institution_gates(&mut s, "cargo build --release", Some("vestige"));
        assert!(same.map(|x| !x.contains("PIVOT GATE")).unwrap_or(true), "same family is a retry, not a pivot");
        // After a WebSearch tool call → searched_since_fail=true → pivot satisfied
        s.searched_since_fail = true;
        let after = institution_gates(&mut s, "make all", Some("vestige"));
        assert!(after.map(|x| !x.contains("PIVOT GATE")).unwrap_or(true), "search satisfies the pivot gate");
    }

    #[test]
    fn pivot_gate_expires_after_30min() {
        let mut s = st();
        s.started = true;
        s.last_failed_fam = "cargo build".into();
        s.last_failed_ts = epoch().saturating_sub(3_600); // 1 hour ago
        s.searched_since_fail = false;
        let d = institution_gates(&mut s, "make all", Some("vestige"));
        assert!(d.map(|x| !x.contains("PIVOT GATE")).unwrap_or(true), "old failure does not gate");
    }

    #[test]
    fn chash_first_200_chars_only() {
        let long_a = format!("{}{}", "x".repeat(300), "A");
        let long_b = format!("{}{}", "x".repeat(300), "B");
        assert_eq!(chash(&long_a), chash(&long_b), "identical first 200 chars hash equal");
    }
}
