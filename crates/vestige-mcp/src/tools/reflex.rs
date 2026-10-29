//! `reflex`: the Vestige tool that agent hooks call directly.
//!
//! Hidden: it is a dispatch arm, never part of `tools/list`, so it costs no
//! schema bytes and no agent ever calls it by choice. Claude Code reaches it
//! from `mcp_tool` hooks on the agent's own MCP connection; other hosts reach it
//! through the `@vestige/init` command shim. Every call answers in hook JSON.
//!
//! Its first handler is the stop-to-Strata reflex, moved into the server that
//! owns the store. On every hook firing it reads the Operator gate's day files
//! under `$OPERATOR_HOME/receipts` (default `~/.operator`) from where it last
//! stopped, checks each new line against the gate's digest chain, and:
//!
//! - saves every STOP as an `operator-stop` memory `derived_from` its rule
//!   memory, with the same tags and text `hooks/operator-stop-memory.py` writes,
//!   so `recall(handle='op-receipt-<16 hex>')` and the PRECEDENT journey see no
//!   change. A line that breaks the chain never becomes a memory.
//! - records which memories the stopped agent held at that exact tool call,
//!   when a PreToolUse firing captured them.
//! - links a later owner permit, or the law's own compliant path, to the stop it
//!   answers with a `closes` edge, and names that path the next time the same
//!   rule stops a call.
//! - tells the stopped session once, inside a per-event and per-turn budget,
//!   and warns before the exact same command is tried again.
//!
//! Its second handler is the context manifest. Reflex keeps, per session and per
//! agent, the exact memories each agent holds: the ones its own Vestige reads
//! returned and the ones a hook injected. Ids are read from JSON keys, never
//! matched in text. A Vestige read that returns a body the agent already holds
//! word for word gets that body folded to a stub naming the call that unfolds
//! it, the memory hook never injects a memory twice, and every hook shares one
//! budget per turn. Compaction and `/clear` empty what an agent holds.
//!
//! Every firing appends one hash-chained line to `<data_dir>/reflex/ledger.jsonl`,
//! the Hook Ledger, which `reflex(action='ledger')` summarizes and verifies.
//!
//! The gate decides. Reflex records and tells. It never allows, retries, or
//! routes around a stop, and nothing here reads or writes the gate's own files
//! beyond reading its receipts.

use std::collections::{HashMap, VecDeque};
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use chrono::{DateTime, Days, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::Digest as _;
use tokio::sync::Mutex;
use vestige_core::{OutputConfig, Storage};

use crate::cognitive::CognitiveEngine;

mod compaction;
mod pairing;
mod stale;
mod witness;

pub use stale::record as record_retirement;
pub(crate) use witness::{actor_for, config_for, session_tag};

/// The dispatch name hooks call.
pub const NAME: &str = "reflex";

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Gate receipts saved as memories per firing; the rest wait for the next one.
const MAX_WRITES: usize = 8;
/// Time one firing may spend writing memories before it answers.
const WRITE_BUDGET_MS: u128 = 250;
/// Day files read the first time reflex sees a gate home.
const FIRST_RUN_DAYS: u64 = 7;
const MAX_PENDING: usize = 500;
const MAX_DONE: usize = 2000;
const MAX_OPEN: usize = 500;
const MAX_COMMANDS: usize = 2000;
const RECENT_KEEP: usize = 20;
const RECENT_SHOWN: usize = 5;
const RECENT_DAYS: i64 = 7;
/// Characters every Vestige hook together may add to one turn: reflex's own
/// notices and the memory hook's injections share it.
const TURN_BUDGET: usize = 8000;
/// What the memory hook leaves of the turn budget for reflex's notices.
const NOTICE_RESERVE: usize = 1500;
const MANIFEST_KEEP: usize = 400;
const MANIFEST_IN_STOP: usize = 40;
const SNAPSHOT_KEEP: usize = 128;
const SESSIONS_KEEP: usize = 64;
/// Agent contexts kept per session: the main agent and its subagents.
const CONTEXTS_KEEP: usize = 16;
/// Fingerprints kept per held memory, one for each body variant shown.
const HASHES_KEEP: usize = 4;
/// A body shorter than this is never folded; its stub would save too little.
const FOLD_MIN: usize = 400;
/// Characters of a folded body kept as its opening.
const FOLD_OPENING: usize = 120;
/// Firings remembered per session to drop a second copy of the same one.
const SEEN_KEEP: usize = 256;
/// A turn start or a call-less event repeated within this many seconds is
/// the same firing reported twice.
const SAME_FIRING_SECS: i64 = 5;
const PERSIST_EVERY_SECS: i64 = 30;
const HEARTBEAT_EVERY_SECS: i64 = 30;
const LEDGER_ROTATE_BYTES: u64 = 8 * 1024 * 1024;
/// Firings a permit or ALLOW waits for the stop it answers before it is dropped.
const RESOLUTION_MAX_TRIES: u32 = 50;

/// Characters one event may add to the agent's context. Zero means the host
/// does not take added context on that event, so reflex only records.
fn event_budget(event: &str) -> usize {
    match event {
        "UserPromptSubmit" | "SessionStart" => 2000,
        "SubagentStart" => 1500,
        "PreToolUse" | "PostToolUse" | "PostToolUseFailure" | "PostToolBatch" => 600,
        _ => 0,
    }
}

/// Input schema. Not advertised; kept for documentation and tests.
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "hook (default), ledger or status."},
            "event": {"type": "string", "description": "The hook event name, e.g. PostToolUse."},
            "session": {"type": "string", "description": "The host's session id."},
            "host": {"type": "string", "description": "claude, codex, cursor, gemini, opencode."},
            "tool": {"type": "string", "description": "The tool the event is about."},
            "call": {"type": "string", "description": "The tool_use_id."},
            "agent": {"type": "string", "description": "The subagent's agent_id; empty for the main agent."},
            "command": {"type": "string", "description": "The shell command, on PreToolUse of a shell tool."},
            "source": {"type": "string", "description": "SessionStart source: startup, resume, clear or compact."},
            "format": {"type": "string", "description": "claude (default) for hook JSON, plain for {context}."},
            "since": {"type": "string", "description": "Ledger window, e.g. 30m, 24h, 7d."}
        }
    })
}

// ---------------------------------------------------------------------------
// Per-connection state
// ---------------------------------------------------------------------------

/// What reflex knows about one MCP connection: the host session it belongs to,
/// learned from the first reflex call on that connection.
#[derive(Default)]
pub struct Connection {
    session: Option<String>,
}

pub type ConnectionState = Arc<std::sync::Mutex<Connection>>;

pub fn new_connection() -> ConnectionState {
    Arc::new(std::sync::Mutex::new(Connection::default()))
}

/// What one Vestige read put in front of the agent. Claude Code sends the call's
/// tool_use_id in `_meta`, and the PreToolUse firing just before it recorded
/// that id with the agent that made the call, so the read lands in exactly
/// that agent's context. A call reflex cannot tie to an agent is left as it was.
///
/// Every memory the result names joins that agent's context. A body the agent
/// already holds word for word is folded to its opening and the exact call that
/// unfolds it, unless this call named that memory itself. Tags, types, dates,
/// edge paths and receipt ids are never touched. A folded result carries a
/// `reflexFold` block with both sizes, the ids folded and the SHA-256 of the
/// result as it was before folding.
pub fn fold(
    conn: &ConnectionState,
    storage: &Arc<Storage>,
    tool: &str,
    args: Option<&Value>,
    call: Option<&str>,
    result: &mut Value,
) {
    if tool == NAME || !result.is_object() {
        return;
    }
    let Some(call) = call.filter(|c| !c.is_empty()) else {
        return;
    };
    let Some(session) = conn.lock().ok().and_then(|c| c.session.clone()) else {
        return;
    };
    let started = Instant::now();
    let global = global_for(storage);
    let Ok(mut g) = global.lock() else {
        return;
    };
    let seq_now = g.p.retire_seq;
    let s = g.session(&session);
    let Some(agent) = s
        .snapshots
        .iter()
        .rev()
        .find(|x| x.call == call)
        .map(|x| x.agent.clone())
    else {
        return;
    };
    // A failure's memory links to the last memory this session wrote.
    let written = result
        .get("nodeId")
        .and_then(Value::as_str)
        .or_else(|| {
            result
                .get("results")
                .and_then(Value::as_array)
                .and_then(|r| r.iter().rev().find_map(|x| x.get("nodeId")?.as_str()))
        })
        .filter(|id| is_memory_id(id))
        .map(str::to_string);
    if let Some(id) = written {
        s.last_write = Some(id);
    }
    let turn = s.turn;
    let explicit = named_ids(args);
    let original = result.to_string();
    let ctx = s.context(&agent);
    ctx.seq_now = seq_now;
    let folded = fold_value(ctx, turn, tool, &explicit, result);
    let manifest = ctx.digest();
    let held = ctx.held.len();
    g.dirty = true;
    if folded.is_empty() {
        return;
    }
    let block = attach_fold_block(&original, &folded, result);
    let sent = result.to_string().len();
    let inp = Input {
        event: "Fold".into(),
        session,
        host: "claude".into(),
        tool: tool.to_string(),
        call: call.to_string(),
        agent,
        ..Input::default()
    };
    append_ledger(
        &mut g,
        &inp,
        0,
        &[],
        &[],
        started.elapsed().as_millis(),
        json!({"fold": block, "sentChars": sent, "manifest": manifest, "held": held}),
    );
}

/// Add the `reflexFold` block to a folded result and return it. `foldedChars`
/// is the result's size before the block; the ledger keeps the size sent.
fn attach_fold_block(original: &str, folded: &[String], result: &mut Value) -> Value {
    let block = json!({
        "folded": folded,
        "originalChars": original.len(),
        "foldedChars": result.to_string().len(),
        "originalSha256": sha256_hex(original),
        "note": "These memories are already in your context word for word, so their bodies are folded. memory(action='get', id=...) returns any of them whole.",
    });
    if let Some(o) = result.as_object_mut() {
        o.insert("reflexFold".into(), block.clone());
    }
    block
}

/// Replay recorded Claude Code transcripts through the fold, offline. Every
/// Vestige tool result is taken in order, with one context for the main agent
/// and one per subagent sidechain, emptied at each compaction. For each tool it
/// reports the characters the agent was sent and the characters it would have
/// been sent with the fold, block included. Nothing is written anywhere.
pub fn replay(paths: &[PathBuf]) -> Value {
    #[derive(Default)]
    struct Tally {
        calls: u64,
        folded_calls: u64,
        folds: u64,
        before: u64,
        after: u64,
    }
    let mut by_tool: std::collections::BTreeMap<String, Tally> = std::collections::BTreeMap::new();
    let (mut files, mut unreadable, mut compactions, mut not_json) = (0u64, 0u64, 0u64, 0u64);
    for path in paths {
        let Ok(text) = std::fs::read_to_string(path) else {
            unreadable += 1;
            continue;
        };
        files += 1;
        let mut contexts: HashMap<String, Context> = HashMap::new();
        let mut uses: HashMap<String, (String, Value)> = HashMap::new();
        let mut turn = 0u32;
        for line in text.lines() {
            let Ok(r) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if r["type"] == "system" && r["subtype"] == "compact_boundary" {
                contexts.clear();
                compactions += 1;
                continue;
            }
            let agent = if r["isSidechain"] == true {
                r["agentId"].as_str().unwrap_or("sidechain").to_string()
            } else {
                String::new()
            };
            let blocks = r["message"]["content"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let prompt = r["type"] == "user"
                && agent.is_empty()
                && r["isCompactSummary"] != true
                && (r["message"]["content"].is_string()
                    || !blocks.iter().any(|b| b["type"] == "tool_result"));
            if prompt {
                turn += 1;
            }
            for b in blocks {
                if b["type"] == "tool_use" {
                    if let Some(tool) = b["name"]
                        .as_str()
                        .and_then(|n| n.strip_prefix("mcp__vestige__"))
                        && let Some(id) = b["id"].as_str()
                    {
                        uses.insert(id.to_string(), (tool.to_string(), b["input"].clone()));
                    }
                    continue;
                }
                if b["type"] != "tool_result" {
                    continue;
                }
                let Some((tool, input)) = b["tool_use_id"].as_str().and_then(|id| uses.get(id))
                else {
                    continue;
                };
                let body = match &b["content"] {
                    Value::String(s) => s.clone(),
                    Value::Array(parts) => parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => continue,
                };
                let Ok(mut v) = serde_json::from_str::<Value>(&body) else {
                    not_json += 1;
                    continue;
                };
                if !v.is_object() {
                    not_json += 1;
                    continue;
                }
                let original = v.to_string();
                let explicit = named_ids(Some(input));
                let ctx = contexts.entry(agent.clone()).or_default();
                let folded = fold_value(ctx, turn, tool, &explicit, &mut v);
                if !folded.is_empty() {
                    attach_fold_block(&original, &folded, &mut v);
                }
                let t = by_tool.entry(tool.clone()).or_default();
                t.calls += 1;
                t.before += original.len() as u64;
                t.after += v.to_string().len() as u64;
                if !folded.is_empty() {
                    t.folded_calls += 1;
                    t.folds += folded.len() as u64;
                }
            }
        }
    }
    let pct = |before: u64, after: u64| {
        if before == 0 {
            0.0
        } else {
            ((before.saturating_sub(after)) as f64 * 1000.0 / before as f64).round() / 10.0
        }
    };
    let (mut before, mut after, mut calls, mut folds) = (0u64, 0u64, 0u64, 0u64);
    let tools: serde_json::Map<String, Value> = by_tool
        .iter()
        .map(|(tool, t)| {
            before += t.before;
            after += t.after;
            calls += t.calls;
            folds += t.folds;
            (
                tool.clone(),
                json!({
                    "calls": t.calls,
                    "foldedCalls": t.folded_calls,
                    "memoriesFolded": t.folds,
                    "charsSent": t.before,
                    "charsWithFold": t.after,
                    "charsSaved": t.before.saturating_sub(t.after),
                    "savedPct": pct(t.before, t.after),
                }),
            )
        })
        .collect();
    json!({
        "transcripts": files,
        "unreadable": unreadable,
        "compactions": compactions,
        "resultsNotJson": not_json,
        "vestigeCalls": calls,
        "memoriesFolded": folds,
        "charsSent": before,
        "charsWithFold": after,
        "charsSaved": before.saturating_sub(after),
        "savedPct": pct(before, after),
        "byTool": tools,
        "method": "Each recorded Vestige result, in order, folded against what that agent already held word for word; one context per transcript and per subagent sidechain, emptied at each compaction; calls that named a memory by id never fold it.",
    })
}

/// What the memory hook is about to inject at a session start or a prompt. A
/// memory the agent already holds at least this much of is skipped, the rest
/// stays inside what is left of the turn budget, and what is kept joins the
/// main agent's context. The packet says what it skipped and why.
pub fn admit_injection(storage: &Arc<Storage>, hook: &Value, result: &mut Value) {
    let session = field(hook, &["session", "session_id"]);
    if session.is_empty() {
        return;
    }
    let event = match field(hook, &["event"]).as_str() {
        "session" | "SessionStart" => "SessionStart",
        "prompt" | "UserPromptSubmit" => "UserPromptSubmit",
        "tool" | "PostToolUse" => "PostToolUse",
        _ => "MemoryHook",
    }
    .to_string();
    let source = field(hook, &["source"]);
    let started = Instant::now();
    let global = global_for(storage);
    let Ok(mut g) = global.lock() else {
        return;
    };
    let seq_now = g.p.retire_seq;
    let s = g.session(&session);
    let now = Utc::now().timestamp();
    if event == "UserPromptSubmit" {
        s.start_turn(now);
    }
    if event == "SessionStart" && matches!(source.as_str(), "clear" | "compact") {
        s.forget_all();
    }
    let mut room = TURN_BUDGET
        .saturating_sub(s.turn_chars)
        .saturating_sub(NOTICE_RESERVE);
    let turn = s.turn;
    let via = format!("hook:{event}");
    let ctx = s.context("");
    ctx.seq_now = seq_now;
    let (mut injected, mut skipped, mut dropped) = (Vec::new(), Vec::new(), Vec::new());
    let mut spent = 0usize;
    if let Some(rows) = result.get_mut("memories").and_then(Value::as_array_mut) {
        rows.retain(|row| {
            let Some(id) = row.get("id").and_then(Value::as_str).map(str::to_string) else {
                return true;
            };
            let body = row.get("content").and_then(Value::as_str).unwrap_or("");
            if ctx.get(&id).is_some_and(|h| h.chars >= body.len()) {
                skipped.push(id);
                return false;
            }
            // The memory hook writes one JSON line per memory; count that line.
            let line = json!({"id": id, "text": body, "entities": row.get("tags")})
                .to_string()
                .len()
                + 1;
            if line > room {
                dropped.push(id);
                return false;
            }
            room -= line;
            spent += line;
            ctx.hold(&id, turn, &via, Some(body));
            injected.push(id);
            true
        });
    }
    let manifest = ctx.digest();
    let held = ctx.held.len();
    s.turn_chars += spent;
    if let Some(o) = result.as_object_mut() {
        o.insert(
            "reflex".into(),
            json!({
                "injected": injected,
                "skippedAlreadyHeld": skipped,
                "droppedForTurnBudget": dropped,
                "turnBudgetLeft": room,
            }),
        );
    }
    g.dirty = true;
    let inp = Input {
        event: "MemoryHook".into(),
        session,
        host: field(hook, &["host", "client"]),
        source,
        ..Input::default()
    };
    append_ledger(
        &mut g,
        &inp,
        spent,
        &[],
        &[],
        started.elapsed().as_millis(),
        json!({
            "hook": event,
            "injected": injected,
            "skipped": skipped,
            "dropped": dropped,
            "manifest": manifest,
            "held": held,
        }),
    );
}

/// The memory ids a call names itself. A memory the agent asked for by id is
/// never folded in that call's own result.
fn named_ids(args: Option<&Value>) -> Vec<String> {
    let mut out = Vec::new();
    let Some(a) = args else {
        return out;
    };
    for key in [
        "id",
        "ids",
        "handle",
        "memoryId",
        "memory_id",
        "nodeId",
        "node_id",
        "center_id",
    ] {
        let mut push = |s: &str| {
            if is_memory_id(s) && !out.iter().any(|o| o == s) {
                out.push(s.to_string());
            }
        };
        match a.get(key) {
            Some(Value::String(s)) => push(s),
            Some(Value::Array(items)) => items.iter().filter_map(Value::as_str).for_each(push),
            _ => {}
        }
    }
    out
}

/// Fold, in place, every memory body in `v` the context already holds word for
/// word, and add every other memory `v` names to the context. Ids are read from
/// JSON keys in serialization order, so a body shown earlier in the same result
/// counts as held for a later copy. Returns the ids folded.
fn fold_value(
    ctx: &mut Context,
    turn: u32,
    via: &str,
    explicit: &[String],
    v: &mut Value,
) -> Vec<String> {
    let mut folded = Vec::new();
    walk(ctx, turn, via, explicit, v, &mut folded, 0);
    folded
}

fn walk(
    ctx: &mut Context,
    turn: u32,
    via: &str,
    explicit: &[String],
    v: &mut Value,
    folded: &mut Vec<String>,
    depth: usize,
) {
    if depth > 16 {
        return;
    }
    match v {
        Value::Object(map) => {
            let id = ["id", "nodeId", "memoryId", "node_id"]
                .iter()
                .find_map(|k| {
                    map.get(*k)
                        .and_then(Value::as_str)
                        .filter(|s| is_memory_id(s))
                        .map(str::to_string)
                });
            if let Some(id) = &id {
                let body = map
                    .get("content")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                match body {
                    Some(body) => {
                        let held = ctx
                            .get(id)
                            .filter(|h| h.hashes.contains(&fingerprint(&body)))
                            .cloned();
                        let short = held.map(|h| stub(&body, id, &h));
                        match short {
                            Some(short)
                                if body.len() >= FOLD_MIN
                                    && short.len() < body.len()
                                    && !explicit.contains(id) =>
                            {
                                map.insert("content".into(), Value::String(short));
                                if !folded.contains(id) {
                                    folded.push(id.clone());
                                }
                            }
                            _ => ctx.hold(id, turn, via, Some(&body)),
                        }
                    }
                    None => ctx.hold(id, turn, via, None),
                }
            }
            for (k, x) in map.iter_mut() {
                if k == "content" && x.is_string() {
                    continue;
                }
                walk(ctx, turn, via, explicit, x, folded, depth + 1);
            }
        }
        Value::Array(items) => {
            for x in items {
                walk(ctx, turn, via, explicit, x, folded, depth + 1);
            }
        }
        _ => {}
    }
}

fn stub(body: &str, id: &str, held: &Held) -> String {
    let opening: String = body.chars().take(FOLD_OPENING).collect();
    format!(
        "{}… [held since turn {} via {}; memory(action='get', id='{id}') unfolds it]",
        opening.trim_end(),
        held.turn,
        held.via
    )
}

fn sha256_hex(s: &str) -> String {
    sha2::Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The first 16 hex characters of a body's SHA-256: enough to tell the body
/// variants of one memory apart, never matched across memories.
fn fingerprint(body: &str) -> String {
    sha256_hex(body)[..16].to_string()
}

fn is_hex(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_memory_id(s: &str) -> bool {
    if let Some(rest) = s.strip_prefix("mem-") {
        return rest.len() == 16 && is_hex(rest);
    }
    s.len() == 36
        && s.char_indices().all(|(i, c)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                c == '-'
            } else {
                c.is_ascii_hexdigit()
            }
        })
}

fn is_digest(s: &str) -> bool {
    s.len() == 64 && is_hex(s)
}

fn is_rule(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphanumeric())
        && s.len() <= 64
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

#[derive(Default, Debug, Clone)]
struct Input {
    action: String,
    event: String,
    session: String,
    host: String,
    tool: String,
    call: String,
    agent: String,
    command: String,
    source: String,
    format: String,
    since: String,
    /// PostToolUseFailure: the error the tool reported.
    error: String,
    interrupt: bool,
    /// The file an edit tool wrote.
    file: String,
    cwd: String,
    /// PostCompact: the summary compaction wrote, read for exact ids.
    summary: String,
    /// PreCompact and PostCompact: manual or auto.
    trigger: String,
    /// PostToolUse of a shell tool: what it printed, exactly as the host
    /// reported it, for a witness's output digest.
    stdout: String,
    stderr: String,
    /// PostToolUseFailure: the error text exactly as reported.
    error_raw: String,
    /// The shell call ran in the background, so its result is not the command's.
    background: bool,
}

/// The exact text of one field, untrimmed and uncut: an absent field and a
/// substitution the host left unfilled are empty.
fn raw(args: &Value, keys: &[&str]) -> String {
    for k in keys {
        if let Some(s) = args.pointer(k).and_then(Value::as_str) {
            let t = s.trim();
            if t.starts_with("${") && t.ends_with('}') && !t.contains('\n') {
                continue;
            }
            return s.to_string();
        }
    }
    String::new()
}

/// A substitution the host left unfilled (`${...}`) counts as absent.
fn field(args: &Value, keys: &[&str]) -> String {
    for k in keys {
        if let Some(s) = args.get(*k).and_then(Value::as_str) {
            let t = s.trim();
            if !t.is_empty() && !t.starts_with("${") {
                return t.chars().take(4000).collect();
            }
        }
    }
    String::new()
}

impl Input {
    fn parse(args: Option<&Value>) -> Input {
        let empty = json!({});
        let a = args.unwrap_or(&empty);
        let nested = a.get("tool_input").cloned().unwrap_or(Value::Null);
        let mut command = field(a, &["command"]);
        if command.is_empty() {
            command = field(&nested, &["command", "cmd"]);
        }
        let action = field(a, &["action"]);
        Input {
            action: if action.is_empty() {
                "hook".into()
            } else {
                action
            },
            event: field(a, &["event", "hook_event_name", "hookEventName"]),
            session: field(
                a,
                &["session", "session_id", "sessionId", "conversation_id"],
            ),
            host: {
                let h = field(a, &["host", "client"]);
                if h.is_empty() { "claude".into() } else { h }
            },
            tool: field(a, &["tool", "tool_name", "toolName"]),
            call: field(a, &["call", "tool_use_id", "toolUseId"]),
            agent: field(a, &["agent", "agent_id", "agentId"]),
            command,
            source: field(a, &["source"]),
            format: field(a, &["format"]),
            since: field(a, &["since"]),
            error: field(a, &["error"]),
            interrupt: matches!(
                a.get("interrupt")
                    .or_else(|| a.get("is_interrupt"))
                    .or_else(|| a.get("interrupted")),
                Some(Value::Bool(true))
            ) || field(a, &["interrupt", "is_interrupt", "interrupted"]) == "true",
            file: {
                let f = field(a, &["file", "file_path"]);
                if f.is_empty() {
                    field(&nested, &["file_path", "path", "notebook_path"])
                } else {
                    f
                }
            },
            cwd: field(a, &["cwd"]),
            summary: a
                .get("summary")
                .or_else(|| a.get("compact_summary"))
                .and_then(Value::as_str)
                .filter(|s| !s.trim_start().starts_with("${"))
                .map(|s| s.chars().take(200_000).collect())
                .unwrap_or_default(),
            trigger: field(a, &["trigger"]),
            stdout: raw(a, &["/stdout", "/tool_response/stdout"]),
            stderr: raw(a, &["/stderr", "/tool_response/stderr"]),
            error_raw: raw(a, &["/error"]),
            background: matches!(a.get("background"), Some(Value::Bool(true)))
                || field(a, &["background"]) == "true"
                || nested.get("run_in_background") == Some(&Value::Bool(true)),
        }
    }
}

// ---------------------------------------------------------------------------
// Gate receipts
// ---------------------------------------------------------------------------

/// The fields of a gate receipt reflex acts on.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
struct GateRow {
    kind: String,
    digest: String,
    prev: String,
    rule: String,
    rule_name: String,
    ts: String,
    source: String,
    session: String,
    cwd: String,
    tool: String,
    preview: String,
    detail: String,
    action: String,
    call: String,
    gate_version: String,
    via: String,
    stop_receipt: String,
    #[serde(default)]
    tries: u32,
}

fn clip(v: Option<&Value>, n: usize) -> String {
    let s = match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    };
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(n)
        .collect()
}

impl GateRow {
    fn from_receipt(rec: &Value) -> Option<GateRow> {
        let kind = rec.get("decision")?.as_str()?.to_string();
        let digest = rec.get("digest")?.as_str()?.to_string();
        if !is_digest(&digest) {
            return None;
        }
        let row = GateRow {
            kind: kind.clone(),
            digest,
            prev: clip(rec.get("prev"), 64),
            rule: clip(rec.get("commitment"), 64),
            rule_name: clip(rec.get("rule"), 80),
            ts: clip(rec.get("ts"), 32),
            source: clip(rec.get("source"), 32),
            session: clip(rec.get("session"), 64),
            cwd: clip(rec.get("cwd"), 300),
            tool: clip(rec.get("tool"), 64),
            preview: clip(rec.get("action_preview"), 300),
            detail: clip(rec.get("detail"), 300),
            action: clip(rec.get("action_digest"), 24),
            call: clip(rec.get("call"), 80),
            gate_version: clip(rec.get("gate_version"), 16),
            via: clip(rec.get("via"), 32),
            stop_receipt: clip(rec.get("stop_receipt"), 64),
            tries: 0,
        };
        match kind.as_str() {
            "STOP" => is_rule(&row.rule).then_some(row),
            // Only the ALLOWs that answer a stop: the law's own compliant path,
            // or a one-time permit being used for an action the gate stopped.
            "ALLOW" => (row.via == "law_lift" || !row.action.is_empty()).then_some(row),
            "PERMIT_GRANTED" => is_digest(&row.stop_receipt).then_some(row),
            _ => None,
        }
    }

    fn d16(&self) -> String {
        self.digest.chars().take(16).collect()
    }
}

// ---------------------------------------------------------------------------
// State shared by every connection to one store
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct OpenStop {
    d16: String,
    node: String,
    session: String,
    rule: String,
    action: String,
    ts: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Resolution {
    node: String,
    how: String,
    ts: String,
    stop_d16: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Recent {
    d16: String,
    node: String,
    rule: String,
    ts: String,
    preview: String,
}

/// What reflex keeps across restarts, at `<data_dir>/reflex/state.json`.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Persisted {
    #[serde(default)]
    files: Vec<(String, u64)>,
    #[serde(default)]
    tip: Option<String>,
    #[serde(default)]
    started: bool,
    #[serde(default)]
    lines_read: u64,
    #[serde(default)]
    chain_breaks: u64,
    #[serde(default)]
    pending: Vec<GateRow>,
    #[serde(default)]
    done: Vec<(String, String)>,
    #[serde(default)]
    rules: Vec<(String, String)>,
    #[serde(default)]
    open: Vec<OpenStop>,
    #[serde(default)]
    commands: Vec<(String, String)>,
    #[serde(default)]
    resolutions: Vec<(String, Resolution)>,
    #[serde(default)]
    recent: Vec<Recent>,
    #[serde(default)]
    ledger_tip: Option<String>,
    /// Fixes Failure Pairing recorded, by failure signature.
    #[serde(default)]
    fixes: Vec<(String, pairing::Fix)>,
    /// Memories any agent retired, in order, for the stale-belief push.
    #[serde(default)]
    retired: Vec<stale::Retirement>,
    #[serde(default)]
    retire_seq: u64,
}

fn lookup<'a, T>(list: &'a [(String, T)], key: &str) -> Option<&'a T> {
    list.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn upsert<T>(list: &mut Vec<(String, T)>, key: &str, value: T, cap: usize) {
    list.retain(|(k, _)| k != key);
    list.push((key.to_string(), value));
    if list.len() > cap {
        let drop = list.len() - cap;
        list.drain(..drop);
    }
}

#[derive(Clone, Debug, Default)]
struct StopNote {
    d16: String,
    node: String,
    rule: String,
    rule_node: String,
    preview: String,
    detail: String,
    ts: String,
    resolution: Option<Resolution>,
}

#[derive(Clone, Debug, Default)]
struct Snapshot {
    call: String,
    agent: String,
    command: Option<String>,
    manifest: Vec<String>,
}

/// One memory an agent holds.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Held {
    /// The turn it entered the agent's context.
    turn: u32,
    /// What put it there: a Vestige tool, a hook, or a reflex notice.
    via: String,
    /// The longest body of it the agent was shown; 0 when only its id was.
    chars: usize,
    /// Fingerprints of every body variant the agent was shown.
    hashes: Vec<String>,
    /// The store's retirement count when it entered the agent's context; a
    /// later retirement of it is news to this agent.
    #[serde(default)]
    seq: u64,
}

/// The exact memories one agent holds, oldest first.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Context {
    held: Vec<(String, Held)>,
    #[serde(default)]
    touched: i64,
    /// The store's retirement count now, stamped on what is held next.
    #[serde(skip)]
    seq_now: u64,
}

impl Context {
    fn get(&self, id: &str) -> Option<&Held> {
        self.held.iter().find(|(k, _)| k == id).map(|(_, h)| h)
    }

    fn hold(&mut self, id: &str, turn: u32, via: &str, body: Option<&str>) {
        let mut h = match self.held.iter().position(|(k, _)| k == id) {
            Some(pos) => self.held.remove(pos).1,
            None => Held {
                turn,
                via: via.to_string(),
                seq: self.seq_now,
                ..Held::default()
            },
        };
        if let Some(body) = body {
            h.chars = h.chars.max(body.len());
            let fp = fingerprint(body);
            if !h.hashes.contains(&fp) {
                h.hashes.push(fp);
                if h.hashes.len() > HASHES_KEEP {
                    h.hashes.remove(0);
                }
            }
        }
        self.held.push((id.to_string(), h));
        if self.held.len() > MANIFEST_KEEP {
            let drop = self.held.len() - MANIFEST_KEEP;
            self.held.drain(..drop);
        }
    }

    fn ids(&self) -> Vec<String> {
        self.held.iter().map(|(k, _)| k.clone()).collect()
    }

    /// The manifest: SHA-256 over the sorted ids, first 16 hex characters.
    fn digest(&self) -> String {
        manifest_digest(&self.ids())
    }
}

fn manifest_digest(ids: &[String]) -> String {
    let mut sorted: Vec<&str> = ids.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    sha256_hex(&sorted.join("\n"))[..16].to_string()
}

#[derive(Debug, Default)]
struct SessionState {
    /// What each agent holds; "" is the main agent, others are subagent ids.
    contexts: HashMap<String, Context>,
    snapshots: VecDeque<Snapshot>,
    stops: Vec<StopNote>,
    told: Vec<String>,
    turn: u32,
    turn_started: i64,
    turn_chars: usize,
    seen: VecDeque<(String, i64)>,
    touched: i64,
    /// The last memory this session's agents wrote through Vestige.
    last_write: Option<String>,
    open_failures: Vec<pairing::OpenFailure>,
    /// Findings from background work, told once at the next firing.
    notes: Vec<String>,
    /// What the main agent held when compaction started.
    checkpoint: Option<compaction::Checkpoint>,
    /// Allowlisted test commands between their PreToolUse and their result.
    witness_pending: Vec<witness::Pending>,
}

impl SessionState {
    fn context(&mut self, agent: &str) -> &mut Context {
        let now = Utc::now().timestamp();
        if !self.contexts.contains_key(agent)
            && self.contexts.len() >= CONTEXTS_KEEP
            && let Some(oldest) = self
                .contexts
                .iter()
                .filter(|(k, _)| !k.is_empty())
                .min_by_key(|(_, c)| c.touched)
                .map(|(k, _)| k.clone())
        {
            self.contexts.remove(&oldest);
        }
        let c = self.contexts.entry(agent.to_string()).or_default();
        c.touched = now;
        c
    }

    /// A new turn starts once, however many hooks report the same prompt.
    fn start_turn(&mut self, now: i64) {
        if now - self.turn_started > SAME_FIRING_SECS {
            self.turn += 1;
            self.turn_chars = 0;
            self.turn_started = now;
        }
    }

    /// Compaction and /clear leave no memory in any agent's context.
    fn forget_all(&mut self) {
        self.contexts.clear();
    }

    /// True when this exact firing was already answered: the same event for
    /// the same call, or a call-less event repeated within a few seconds.
    fn seen_before(&mut self, key: String, has_call: bool, now: i64) -> bool {
        let dup = self
            .seen
            .iter()
            .any(|(k, at)| *k == key && (has_call || now - at <= SAME_FIRING_SECS));
        if !dup {
            self.seen.push_back((key, now));
            while self.seen.len() > SEEN_KEEP {
                self.seen.pop_front();
            }
        }
        dup
    }
}

/// What reflex keeps of one session across restarts, in `sessions.json`.
#[derive(Debug, Default, Serialize, Deserialize)]
struct SavedSession {
    turn: u32,
    touched: i64,
    contexts: HashMap<String, Context>,
    #[serde(default)]
    checkpoint: Option<compaction::Checkpoint>,
}

struct Global {
    dir: PathBuf,
    p: Persisted,
    sessions: HashMap<String, SessionState>,
    in_flight: Vec<String>,
    dirty: bool,
    last_persist: i64,
    last_heartbeat: i64,
    /// One background task per store drains other sessions' backlog.
    bg_running: bool,
}

impl Global {
    fn load(dir: PathBuf) -> Global {
        let p = std::fs::read_to_string(dir.join("state.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Persisted>(&t).ok())
            .unwrap_or_default();
        let saved: HashMap<String, SavedSession> =
            std::fs::read_to_string(dir.join("sessions.json"))
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or_default();
        let sessions = saved
            .into_iter()
            .map(|(id, s)| {
                (
                    id,
                    SessionState {
                        contexts: s.contexts,
                        turn: s.turn,
                        touched: s.touched,
                        checkpoint: s.checkpoint,
                        ..SessionState::default()
                    },
                )
            })
            .collect();
        let mut g = Global {
            dir,
            p,
            sessions,
            in_flight: Vec::new(),
            dirty: false,
            last_persist: 0,
            last_heartbeat: 0,
            bg_running: false,
        };
        if g.p.ledger_tip.is_none() {
            g.p.ledger_tip = last_ledger_digest(&g.dir.join("ledger.jsonl"));
        }
        g
    }

    fn session(&mut self, id: &str) -> &mut SessionState {
        let now = Utc::now().timestamp();
        if !self.sessions.contains_key(id)
            && self.sessions.len() >= SESSIONS_KEEP
            && let Some(oldest) = self
                .sessions
                .iter()
                .min_by_key(|(_, s)| s.touched)
                .map(|(k, _)| k.clone())
        {
            self.sessions.remove(&oldest);
        }
        let s = self.sessions.entry(id.to_string()).or_default();
        s.touched = now;
        s
    }

    fn persist(&mut self, force: bool) {
        let now = Utc::now().timestamp();
        if !(force || (self.dirty && now - self.last_persist >= PERSIST_EVERY_SECS)) {
            return;
        }
        if std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        let tmp = self.dir.join("state.json.tmp");
        let Ok(text) = serde_json::to_string(&self.p) else {
            return;
        };
        let saved: HashMap<&String, SavedSession> = self
            .sessions
            .iter()
            .filter(|(_, s)| !s.contexts.is_empty() || s.checkpoint.is_some())
            .map(|(id, s)| {
                (
                    id,
                    SavedSession {
                        turn: s.turn,
                        touched: s.touched,
                        contexts: s.contexts.clone(),
                        checkpoint: s.checkpoint.clone(),
                    },
                )
            })
            .collect();
        let sessions_tmp = self.dir.join("sessions.json.tmp");
        if let Ok(t) = serde_json::to_string(&saved)
            && std::fs::write(&sessions_tmp, t).is_ok()
        {
            let _ = std::fs::rename(&sessions_tmp, self.dir.join("sessions.json"));
        }
        if std::fs::write(&tmp, text).is_ok()
            && std::fs::rename(&tmp, self.dir.join("state.json")).is_ok()
        {
            self.dirty = false;
            self.last_persist = now;
        }
    }

    /// The Python stop hook steps aside while this file is fresh.
    fn heartbeat(&mut self) {
        let now = Utc::now().timestamp();
        if now - self.last_heartbeat < HEARTBEAT_EVERY_SECS {
            return;
        }
        if std::fs::create_dir_all(&self.dir).is_ok()
            && std::fs::write(self.dir.join("heartbeat"), now.to_string()).is_ok()
        {
            self.last_heartbeat = now;
        }
    }
}

fn registry() -> &'static std::sync::Mutex<HashMap<PathBuf, Arc<std::sync::Mutex<Global>>>> {
    static REG: OnceLock<std::sync::Mutex<HashMap<PathBuf, Arc<std::sync::Mutex<Global>>>>> =
        OnceLock::new();
    REG.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn global_for(storage: &Arc<Storage>) -> Arc<std::sync::Mutex<Global>> {
    let dir = storage.data_dir().join("reflex");
    let mut reg = registry().lock().unwrap_or_else(|e| e.into_inner());
    reg.entry(dir.clone())
        .or_insert_with(|| Arc::new(std::sync::Mutex::new(Global::load(dir))))
        .clone()
}

// ---------------------------------------------------------------------------
// Reading the gate's receipts
// ---------------------------------------------------------------------------

fn day_files(rdir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(rdir) else {
        return Vec::new();
    };
    let mut out: Vec<(String, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let stem = name.strip_suffix(".jsonl")?;
            NaiveDate::parse_from_str(stem, "%Y-%m-%d").ok()?;
            Some((name, e.path()))
        })
        .collect();
    out.sort();
    out
}

/// Read new lines from the gate's day files, check each against the digest
/// chain, and queue the rows reflex acts on. A line that breaks the chain is
/// counted and never queued.
fn scan(g: &mut Global, home: &Path) {
    let files = day_files(&home.join("receipts"));
    if files.is_empty() {
        return;
    }
    if !g.p.started {
        g.p.started = true;
        g.dirty = true;
        let first = Utc::now().date_naive() - Days::new(FIRST_RUN_DAYS);
        for (name, path) in &files {
            let day = name
                .strip_suffix(".jsonl")
                .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
            if day.is_some_and(|d| d < first) {
                let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                upsert(&mut g.p.files, name, size, 4096);
                g.p.tip = super::precedent::last_digest(path).or(g.p.tip.take());
            }
        }
    }
    for (name, path) in &files {
        let Ok(size) = std::fs::metadata(path).map(|m| m.len()) else {
            continue;
        };
        let mut start = lookup(&g.p.files, name).copied().unwrap_or(0);
        if start > size {
            start = 0;
        }
        if start == size {
            continue;
        }
        let Ok(mut f) = std::fs::File::open(path) else {
            continue;
        };
        if f.seek(SeekFrom::Start(start)).is_err() {
            continue;
        }
        let mut buf = Vec::with_capacity((size - start) as usize);
        if f.take(size - start).read_to_end(&mut buf).is_err() {
            continue;
        }
        // A line still being written waits for the next firing.
        let end = buf.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        if end == 0 {
            continue;
        }
        for line in String::from_utf8_lossy(&buf[..end]).lines() {
            if line.trim().is_empty() {
                continue;
            }
            g.p.lines_read += 1;
            let parsed = serde_json::from_str::<Value>(line.trim()).ok();
            let stored = parsed
                .as_ref()
                .and_then(|r| r.get("digest"))
                .and_then(Value::as_str)
                .map(str::to_string);
            let broke = match (&parsed, &stored) {
                (Some(rec), Some(d)) => {
                    let linked =
                        g.p.tip
                            .as_deref()
                            .is_none_or(|t| rec.get("prev").and_then(Value::as_str) == Some(t));
                    let own = super::precedent::gate_digest(rec).is_none_or(|x| x == *d);
                    !(linked && own)
                }
                _ => true,
            };
            g.p.tip = stored;
            if broke {
                g.p.chain_breaks += 1;
                continue;
            }
            if let Some(row) = parsed.as_ref().and_then(GateRow::from_receipt) {
                let d16 = row.d16();
                let known = lookup(&g.p.done, &d16).is_some()
                    || g.p.pending.iter().any(|r| r.digest == row.digest);
                if !known {
                    g.p.pending.push(row);
                }
            }
        }
        upsert(&mut g.p.files, name, start + end as u64, 4096);
        g.dirty = true;
    }
    let names: Vec<&String> = files.iter().map(|(n, _)| n).collect();
    g.p.files.retain(|(n, _)| names.contains(&n));
    if g.p.pending.len() > MAX_PENDING {
        let drop = g.p.pending.len() - MAX_PENDING;
        g.p.pending.drain(..drop);
    }
}

// ---------------------------------------------------------------------------
// Turning rows into memories
// ---------------------------------------------------------------------------

enum Job {
    Stop {
        row: GateRow,
        rule_node: Option<String>,
        manifest: Vec<String>,
        command: Option<String>,
        prior: Option<Resolution>,
    },
    Resolve {
        row: GateRow,
        stop: OpenStop,
        rule_node: Option<String>,
    },
}

enum Outcome {
    Stop {
        row: GateRow,
        node: String,
        rule_node: String,
        command: Option<String>,
        prior: Option<Resolution>,
        wrote: bool,
    },
    Resolve {
        row: GateRow,
        node: String,
        stop: OpenStop,
        wrote: bool,
    },
    Wait {
        digest: String,
    },
}

fn how_of(row: &GateRow) -> String {
    match row.kind.as_str() {
        "PERMIT_GRANTED" => format!(
            "the owner granted a one-time permit from their own terminal for action id {}",
            if row.action.is_empty() {
                "unknown"
            } else {
                &row.action
            }
        ),
        _ if row.via == "law_lift" => format!(
            "the law's own compliant path, then the call ran once: {}",
            if row.preview.is_empty() {
                "(no preview)"
            } else {
                &row.preview
            }
        ),
        _ => format!(
            "the owner's one-time permit was used for: {}",
            if row.preview.is_empty() {
                "(no preview)"
            } else {
                &row.preview
            }
        ),
    }
}

/// Pick this firing's work under the lock: the current session's rows first,
/// each group oldest first, so a stop is saved before the rows that answer it.
fn take_jobs(g: &mut Global, session: &str) -> Vec<Job> {
    let mut order: Vec<usize> = (0..g.p.pending.len()).collect();
    order.sort_by_key(|&i| {
        let r = &g.p.pending[i];
        (r.session != session, r.ts.clone())
    });
    let mut jobs = Vec::new();
    let mut give_up = Vec::new();
    for i in order {
        if jobs.len() >= MAX_WRITES {
            break;
        }
        let row = g.p.pending[i].clone();
        if g.in_flight.contains(&row.digest) {
            continue;
        }
        if row.kind == "STOP" {
            let rule_node = lookup(&g.p.rules, &row.rule).cloned();
            let prior = lookup(&g.p.resolutions, &row.rule).cloned();
            let snap = g.sessions.get(&row.session).and_then(|s| {
                s.snapshots
                    .iter()
                    .rev()
                    .find(|x| !row.call.is_empty() && x.call == row.call)
                    .cloned()
            });
            let (manifest, command) = match snap {
                Some(s) => (s.manifest, s.command),
                None => (Vec::new(), None),
            };
            g.in_flight.push(row.digest.clone());
            jobs.push(Job::Stop {
                row,
                rule_node,
                manifest,
                command,
                prior,
            });
            continue;
        }
        let target = if row.kind == "PERMIT_GRANTED" {
            let want: String = row.stop_receipt.chars().take(16).collect();
            g.p.open.iter().find(|s| s.d16 == want).cloned()
        } else {
            g.p.open
                .iter()
                .rev()
                .find(|s| {
                    s.session == row.session
                        && s.ts <= row.ts
                        && ((row.via == "law_lift" && s.rule == row.rule)
                            || (!row.action.is_empty() && s.action == row.action))
                })
                .cloned()
        };
        match target {
            Some(stop) => {
                let rule_node = lookup(&g.p.rules, &stop.rule).cloned();
                g.in_flight.push(row.digest.clone());
                jobs.push(Job::Resolve {
                    row,
                    stop,
                    rule_node,
                });
            }
            None => {
                let r = &mut g.p.pending[i];
                r.tries += 1;
                if r.tries > RESOLUTION_MAX_TRIES {
                    give_up.push(r.digest.clone());
                }
            }
        }
    }
    if !give_up.is_empty() {
        g.p.pending.retain(|r| !give_up.contains(&r.digest));
        g.dirty = true;
    }
    jobs
}

struct Ctx<'a> {
    storage: &'a Arc<Storage>,
    cognitive: &'a Arc<Mutex<CognitiveEngine>>,
    output_config: &'a OutputConfig,
}

/// The memory carrying this exact tag, or None when no memory carries it.
async fn tagged(ctx: &Ctx<'_>, tag: &str) -> Result<Option<String>, String> {
    let out = super::recall::execute(
        ctx.storage,
        ctx.cognitive,
        ctx.output_config,
        Some(json!({"handle": tag, "detail_level": "brief", "limit": 5})),
    )
    .await;
    let v = match out {
        Ok(v) => v,
        Err(e) if e.contains("handle_required") => return Ok(None),
        Err(e) => return Err(e),
    };
    if v.get("error").and_then(Value::as_str) == Some("handle_required") {
        return Ok(None);
    }
    let nodes = v
        .get("nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for node in nodes {
        let has = node
            .get("tags")
            .and_then(Value::as_array)
            .is_some_and(|t| t.iter().any(|x| x.as_str() == Some(tag)));
        if let Some(id) = node.get("id").and_then(Value::as_str)
            && has
            && is_memory_id(id)
        {
            return Ok(Some(id.to_string()));
        }
    }
    Ok(None)
}

async fn ingest(ctx: &Ctx<'_>, item: Value) -> Result<String, String> {
    let out = super::smart_ingest::execute(
        ctx.storage,
        ctx.cognitive,
        Some(json!({"items": [item], "outputProfile": "compact"})),
    )
    .await?;
    let res = out
        .get("results")
        .and_then(Value::as_array)
        .and_then(|r| r.first())
        .cloned()
        .unwrap_or(out);
    for key in ["nodeId", "echoNodeId", "duplicateOf"] {
        if let Some(id) = res.get(key).and_then(Value::as_str)
            && is_memory_id(id)
        {
            return Ok(id.to_string());
        }
    }
    Err(format!(
        "smart_ingest saved nothing: {}",
        res.to_string().chars().take(300).collect::<String>()
    ))
}

fn source_of(row: &GateRow) -> String {
    format!(
        "operator-gate receipt {}, vestige reflex {VERSION}",
        row.digest
    )
}

fn or_unknown(s: &str) -> &str {
    if s.is_empty() { "unknown" } else { s }
}

fn rule_item(row: &GateRow) -> Value {
    let rule = &row.rule;
    let name = if row.rule_name.is_empty() {
        String::new()
    } else {
        format!(" {}", row.rule_name)
    };
    json!({
        "node_type": "pattern",
        "tags": [rule, format!("op-rule-{rule}"), "operator", "operator-rule"],
        "source": source_of(row),
        "content": format!(
            "OPERATOR RULE {rule}{name}. Recorded the first time this rule stopped a call: {} UTC, host {}, \
             session {}. The gate's reason that time: {}. Every stop under this rule is a memory tagged \
             operator-stop and {rule} that is derived_from this memory, so a causal walk from any of those \
             stops reaches the rule.",
            row.ts,
            or_unknown(&row.source),
            or_unknown(&row.session.chars().take(8).collect::<String>()),
            if row.detail.is_empty() { "not given" } else { &row.detail },
        ),
    })
}

fn stop_item(row: &GateRow, rule_node: &str, manifest: &[String], command: Option<&str>) -> Value {
    let d16 = row.d16();
    let name = if row.rule_name.is_empty() {
        String::new()
    } else {
        format!(" {}", row.rule_name)
    };
    let mut content = format!(
        "OPERATOR STOP, recorded from the gate's own receipt. When: {} UTC. Host: {}, session {}. \
         Working directory: {}. Tool: {}. Rule: {}{name}. Command, as the gate's redacted preview: {}. \
         Why: {}. Receipt digest {d16}, chain prev {}, gate version {}, action id {}. Nothing was run. \
         Derived from the rule memory {rule_node}. The way forward is the compliant path the stop named, or a \
         one-time permit for that action id, which only the owner grants from their own terminal.",
        row.ts,
        or_unknown(&row.source),
        or_unknown(&row.session.chars().take(8).collect::<String>()),
        or_unknown(&row.cwd),
        or_unknown(&row.tool),
        row.rule,
        if row.preview.is_empty() {
            "none"
        } else {
            &row.preview
        },
        if row.detail.is_empty() {
            "not given"
        } else {
            &row.detail
        },
        if row.prev.is_empty() {
            "start".to_string()
        } else {
            row.prev.chars().take(16).collect()
        },
        or_unknown(&row.gate_version),
        if row.action.is_empty() {
            "none"
        } else {
            &row.action
        },
    );
    if !manifest.is_empty() {
        let shown: Vec<&str> = manifest
            .iter()
            .rev()
            .take(MANIFEST_IN_STOP)
            .map(String::as_str)
            .collect();
        content.push_str(&format!(
            " Context manifest {} over the {} memories in this agent's context at that tool call; \
             newest first, {} of them: {}.",
            manifest_digest(manifest),
            manifest.len(),
            shown.len(),
            shown.join(", ")
        ));
    }
    if let Some(c) = command {
        content.push_str(&format!(" Command fingerprint {c}."));
    }
    let mut tags = vec![
        row.rule.clone(),
        format!("op-receipt-{d16}"),
        "operator".into(),
        "operator-stop".into(),
    ];
    if !row.session.is_empty() {
        tags.push(session_tag(&row.session));
    }
    json!({
        "node_type": "event",
        "tags": tags,
        "source": source_of(row),
        "links": [{"kind": "derived_from", "to": rule_node}],
        "content": content,
    })
}

fn resolution_item(row: &GateRow, stop: &OpenStop, rule_node: Option<&str>) -> Value {
    let d16 = row.d16();
    let mut links = vec![json!({"kind": "closes", "to": stop.node})];
    if let Some(r) = rule_node {
        links.push(json!({"kind": "derived_from", "to": r}));
    }
    let mut tags = vec![
        stop.rule.clone(),
        format!("op-resolution-{d16}"),
        "operator".into(),
        "operator-resolution".into(),
    ];
    if !row.session.is_empty() {
        tags.push(session_tag(&row.session));
    }
    json!({
        "node_type": "event",
        "tags": tags,
        "source": source_of(row),
        "links": links,
        "content": format!(
            "OPERATOR RESOLUTION, recorded from the gate's own receipt. When: {} UTC. Host: {}, session {}. \
             How the stop was answered: {}. This closes the stop {} (receipt {}) under {}. \
             Receipt digest {d16}, chain prev {}. The gate decided; this records how its stop was answered.",
            row.ts,
            or_unknown(&row.source),
            or_unknown(&row.session.chars().take(8).collect::<String>()),
            how_of(row),
            stop.node,
            stop.d16,
            stop.rule,
            if row.prev.is_empty() { "start".to_string() } else { row.prev.chars().take(16).collect() },
        ),
    })
}

async fn run_job(ctx: &Ctx<'_>, job: Job) -> Result<Outcome, (String, String)> {
    match job {
        Job::Stop {
            row,
            rule_node,
            manifest,
            command,
            prior,
        } => {
            let digest = row.digest.clone();
            let fail = |e: String| (digest.clone(), e);
            if let Some(node) = tagged(ctx, &format!("op-receipt-{}", row.d16()))
                .await
                .map_err(fail)?
            {
                let rule_node = match rule_node {
                    Some(r) => r,
                    None => tagged(ctx, &format!("op-rule-{}", row.rule))
                        .await
                        .map_err(fail)?
                        .unwrap_or_default(),
                };
                return Ok(Outcome::Stop {
                    row,
                    node,
                    rule_node,
                    command,
                    prior,
                    wrote: false,
                });
            }
            let rule_node = match rule_node {
                Some(r) => r,
                None => match tagged(ctx, &format!("op-rule-{}", row.rule))
                    .await
                    .map_err(fail)?
                {
                    Some(r) => r,
                    None => ingest(ctx, rule_item(&row)).await.map_err(fail)?,
                },
            };
            let node = ingest(
                ctx,
                stop_item(&row, &rule_node, &manifest, command.as_deref()),
            )
            .await
            .map_err(fail)?;
            Ok(Outcome::Stop {
                row,
                node,
                rule_node,
                command,
                prior,
                wrote: true,
            })
        }
        Job::Resolve {
            row,
            stop,
            rule_node,
        } => {
            let digest = row.digest.clone();
            let fail = |e: String| (digest.clone(), e);
            if let Some(node) = tagged(ctx, &format!("op-resolution-{}", row.d16()))
                .await
                .map_err(fail)?
            {
                return Ok(Outcome::Resolve {
                    row,
                    node,
                    stop,
                    wrote: false,
                });
            }
            let node = ingest(ctx, resolution_item(&row, &stop, rule_node.as_deref()))
                .await
                .map_err(fail)?;
            Ok(Outcome::Resolve {
                row,
                node,
                stop,
                wrote: true,
            })
        }
    }
}

fn job_row(job: &Job) -> &GateRow {
    match job {
        Job::Stop { row, .. } | Job::Resolve { row, .. } => row,
    }
}

/// Write other sessions' rows off the hook's path. At most one task per store;
/// rows it cannot take now are released for a later firing.
fn spawn_background(
    global: &Arc<std::sync::Mutex<Global>>,
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &Arc<OutputConfig>,
    jobs: Vec<Job>,
) {
    if jobs.is_empty() {
        return;
    }
    {
        let Ok(mut g) = global.lock() else {
            return;
        };
        if g.bg_running || tokio::runtime::Handle::try_current().is_err() {
            for j in &jobs {
                let d = job_row(j).digest.clone();
                g.in_flight.retain(|x| *x != d);
            }
            return;
        }
        g.bg_running = true;
    }
    let global = Arc::clone(global);
    let storage = Arc::clone(storage);
    let cognitive = Arc::clone(cognitive);
    let output_config = Arc::clone(output_config);
    tokio::spawn(async move {
        let started = Instant::now();
        let ctx = Ctx {
            storage: &storage,
            cognitive: &cognitive,
            output_config: &output_config,
        };
        let mut outcomes = Vec::new();
        for job in jobs {
            match run_job(&ctx, job).await {
                Ok(o) => outcomes.push(o),
                Err((digest, e)) => {
                    tracing::warn!(error = %e, "reflex background write waits for a later firing");
                    outcomes.push(Outcome::Wait { digest });
                }
            }
        }
        let Ok(mut g) = global.lock() else {
            return;
        };
        let mut wrote = Vec::new();
        for o in outcomes {
            if let Some(entry) = apply(&mut g, o) {
                wrote.push(entry);
            }
        }
        if !wrote.is_empty() {
            let inp = Input {
                event: "background".into(),
                host: "reflex".into(),
                ..Input::default()
            };
            append_ledger(
                &mut g,
                &inp,
                0,
                &[],
                &wrote,
                started.elapsed().as_millis(),
                json!({}),
            );
            g.persist(true);
        }
        g.bg_running = false;
    });
}

/// Fold one finished job back into the shared state; returns the ledger entry.
fn apply(g: &mut Global, outcome: Outcome) -> Option<Value> {
    match outcome {
        Outcome::Wait { digest } => {
            g.in_flight.retain(|d| *d != digest);
            None
        }
        Outcome::Stop {
            row,
            node,
            rule_node,
            command,
            prior,
            wrote,
        } => {
            let d16 = row.d16();
            g.in_flight.retain(|d| *d != row.digest);
            g.p.pending.retain(|r| r.digest != row.digest);
            upsert(&mut g.p.done, &d16, node.clone(), MAX_DONE);
            if !rule_node.is_empty() {
                upsert(&mut g.p.rules, &row.rule, rule_node.clone(), 1000);
            }
            if let Some(c) = &command {
                upsert(&mut g.p.commands, c, d16.clone(), MAX_COMMANDS);
            }
            g.p.open.retain(|s| s.d16 != d16);
            g.p.open.push(OpenStop {
                d16: d16.clone(),
                node: node.clone(),
                session: row.session.clone(),
                rule: row.rule.clone(),
                action: row.action.clone(),
                ts: row.ts.clone(),
            });
            if g.p.open.len() > MAX_OPEN {
                let drop = g.p.open.len() - MAX_OPEN;
                g.p.open.drain(..drop);
            }
            g.p.recent.retain(|r| r.d16 != d16);
            g.p.recent.push(Recent {
                d16: d16.clone(),
                node: node.clone(),
                rule: row.rule.clone(),
                ts: row.ts.clone(),
                preview: row.preview.chars().take(120).collect(),
            });
            if g.p.recent.len() > RECENT_KEEP {
                let drop = g.p.recent.len() - RECENT_KEEP;
                g.p.recent.drain(..drop);
            }
            if !row.session.is_empty() {
                let note = StopNote {
                    d16: d16.clone(),
                    node: node.clone(),
                    rule: row.rule.clone(),
                    rule_node: rule_node.clone(),
                    preview: row.preview.chars().take(160).collect(),
                    detail: row.detail.chars().take(200).collect(),
                    ts: row.ts.clone(),
                    resolution: prior,
                };
                let s = g.session(&row.session);
                if !s.stops.iter().any(|x| x.d16 == d16) {
                    s.stops.push(note);
                }
            }
            g.dirty = true;
            Some(
                json!({"kind": "stop", "receipt": d16, "memory": node, "rule": row.rule, "new": wrote}),
            )
        }
        Outcome::Resolve {
            row,
            node,
            stop,
            wrote,
        } => {
            g.in_flight.retain(|d| *d != row.digest);
            g.p.pending.retain(|r| r.digest != row.digest);
            g.p.open.retain(|s| s.d16 != stop.d16);
            upsert(
                &mut g.p.resolutions,
                &stop.rule,
                Resolution {
                    node: node.clone(),
                    how: how_of(&row),
                    ts: row.ts.clone(),
                    stop_d16: stop.d16.clone(),
                },
                1000,
            );
            g.dirty = true;
            Some(
                json!({"kind": "resolution", "receipt": row.d16(), "memory": node, "closes": stop.node, "new": wrote}),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// What the agent is told
// ---------------------------------------------------------------------------

fn command_hash(command: &str) -> String {
    let normalized = command.split_whitespace().collect::<Vec<_>>().join(" ");
    sha2::Sha256::digest(normalized.as_bytes())
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn hhmm(ts: &str) -> String {
    ts.get(11..16)
        .map(|t| format!("{t}Z"))
        .unwrap_or_else(|| ts.to_string())
}

fn stop_line(n: &StopNote) -> String {
    let mut line = format!(
        "Operator stopped `{}` under {} at {} (receipt {}). Saved as {}{}.",
        if n.preview.is_empty() {
            "(no preview)"
        } else {
            &n.preview
        },
        n.rule,
        hhmm(&n.ts),
        n.d16,
        n.node,
        if n.rule_node.is_empty() {
            String::new()
        } else {
            format!(", derived from its rule {}", n.rule_node)
        },
    );
    if !n.detail.is_empty() {
        line.push_str(&format!(" Why: {}.", n.detail.trim_end_matches('.')));
    }
    if let Some(r) = &n.resolution {
        line.push_str(&format!(
            " Last time {} stopped a call it was answered by {} ({}).",
            n.rule,
            r.how.trim_end_matches('.'),
            r.node
        ));
    }
    line.push_str(" Take the safe path the stop named, or ask the owner for a one-time permit.");
    line
}

/// Marks a background note in the told list, apart from stop receipts.
const NOTE_MARK: &str = "note\u{0}";
/// Marks a retired belief in the told list.
const STALE_MARK: &str = "stale\u{0}";

/// Compose this firing's context within the event and turn budgets, and mark
/// what it tells. `lead` goes first: what this very firing found. Returns the
/// text and the receipts told.
fn compose(g: &mut Global, inp: &Input, lead: &[String]) -> (String, Vec<String>) {
    let budget = event_budget(&inp.event);
    if budget == 0 || inp.session.is_empty() {
        return (String::new(), Vec::new());
    }
    let recent = g.p.recent.clone();
    let retired = g.p.retired.clone();
    let seq_now = g.p.retire_seq;
    let commands = if inp.event == "PreToolUse" && !inp.command.is_empty() {
        let h = command_hash(&inp.command);
        lookup(&g.p.commands, &h).and_then(|d16| {
            let node = lookup(&g.p.done, d16)?.clone();
            let r = recent.iter().find(|r| r.d16 == *d16).cloned();
            Some((d16.clone(), node, r))
        })
    } else {
        None
    };
    let s = g.session(&inp.session);
    let cap = budget.min(TURN_BUDGET.saturating_sub(s.turn_chars));
    if cap == 0 {
        return (String::new(), Vec::new());
    }
    let mut lines: Vec<(String, Option<String>)> = lead.iter().map(|l| (l.clone(), None)).collect();
    match inp.event.as_str() {
        "SessionStart" if inp.source == "compact" => {
            for n in s.stops.iter().filter(|n| s.told.contains(&n.d16)) {
                lines.push((format!("Before compaction: {}", stop_line(n)), None));
            }
        }
        "SessionStart" => {
            let now = Utc::now().timestamp();
            let fresh: Vec<&Recent> = recent
                .iter()
                .rev()
                .filter(|r| {
                    DateTime::parse_from_rfc3339(&r.ts)
                        .map(|t| now - t.timestamp() <= RECENT_DAYS * 86_400)
                        .unwrap_or(false)
                })
                .take(RECENT_SHOWN)
                .collect();
            if !fresh.is_empty() {
                lines.push((
                    format!(
                        "Operator stopped {} action{} on this machine in the last {RECENT_DAYS} days. Each stop is a Vestige memory; recall its handle to see why, and take the safe path it names instead of trying the same thing again:",
                        fresh.len(),
                        if fresh.len() == 1 { "" } else { "s" }
                    ),
                    None,
                ));
                for r in fresh {
                    lines.push((
                        format!(
                            "- {} {}: {} -> recall(handle=\"op-receipt-{}\")",
                            r.rule,
                            r.ts,
                            if r.preview.is_empty() {
                                "(no preview)"
                            } else {
                                &r.preview
                            },
                            r.d16
                        ),
                        None,
                    ));
                }
            }
        }
        "SubagentStart" => {
            let told: Vec<&StopNote> = s.stops.iter().filter(|n| s.told.contains(&n.d16)).collect();
            if !told.is_empty() {
                lines.push((
                    "Operator stops in this session so far, which hold for you too:".to_string(),
                    None,
                ));
                for n in told.iter().rev().take(3) {
                    lines.push((format!("- {}", stop_line(n)), None));
                }
            }
        }
        "PreToolUse" => {
            // A stop the agent has not heard about yet goes first: this is the
            // last moment before its next tool call runs.
            for n in s.stops.iter().filter(|n| !s.told.contains(&n.d16)) {
                lines.push((stop_line(n), Some(n.d16.clone())));
            }
            if let Some((d16, node, r)) = commands {
                let (rule, ts) = r.map(|r| (r.rule, r.ts)).unwrap_or_default();
                lines.push((
                    format!(
                        "Operator already stopped this exact command{} (receipt {d16}, memory {node}). It will be stopped again. Take the safe path it named, or ask the owner for a one-time permit.",
                        if rule.is_empty() { String::new() } else { format!(" under {rule} at {}", hhmm(&ts)) }
                    ),
                    None,
                ));
            }
        }
        _ => {
            for n in s.stops.iter().filter(|n| !s.told.contains(&n.d16)) {
                lines.push((stop_line(n), Some(n.d16.clone())));
            }
        }
    }
    // A belief another session retired goes before background notes.
    if let Some(ctx) = s.contexts.get(&inp.agent) {
        for (line, old) in stale::lines(&retired, &inp.session, ctx) {
            lines.push((line, Some(format!("{STALE_MARK}{old}"))));
        }
    }
    for note in &s.notes {
        lines.push((note.clone(), Some(format!("{NOTE_MARK}{note}"))));
    }
    let mut text = String::new();
    let mut told = Vec::new();
    for (line, d16) in lines {
        let add = if text.is_empty() {
            line.len()
        } else {
            line.len() + 1
        };
        if text.len() + add > cap {
            break;
        }
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&line);
        if let Some(d) = d16 {
            told.push(d);
        }
    }
    s.turn_chars += text.len();
    let (stale_told, told): (Vec<String>, Vec<String>) =
        told.into_iter().partition(|d| d.starts_with(STALE_MARK));
    if let Some(ctx) = s.contexts.get_mut(&inp.agent) {
        for old in &stale_told {
            let old = &old[STALE_MARK.len()..];
            ctx.held.retain(|(k, _)| k != old);
        }
    }
    let (notes, told): (Vec<String>, Vec<String>) =
        told.into_iter().partition(|d| d.starts_with(NOTE_MARK));
    for n in notes {
        let n = &n[NOTE_MARK.len()..];
        s.notes.retain(|x| x != n);
    }
    for d in &told {
        if !s.told.contains(d) {
            s.told.push(d.clone());
        }
    }
    let told_nodes: Vec<String> = s
        .stops
        .iter()
        .filter(|n| told.contains(&n.d16))
        .map(|n| n.node.clone())
        .collect();
    let turn = s.turn;
    let ctx = s.context(&inp.agent);
    ctx.seq_now = seq_now;
    for id in told_nodes {
        ctx.hold(&id, turn, "reflex", None);
    }
    (text, told)
}

// ---------------------------------------------------------------------------
// The Hook Ledger
// ---------------------------------------------------------------------------

fn ledger_digest(row: &Value) -> String {
    let mut body = row.clone();
    if let Some(o) = body.as_object_mut() {
        o.remove("digest");
    }
    sha2::Sha256::digest(body.to_string().as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn last_ledger_digest(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?;
    serde_json::from_str::<Value>(line)
        .ok()?
        .get("digest")?
        .as_str()
        .map(str::to_string)
}

fn append_ledger(
    g: &mut Global,
    inp: &Input,
    chars: usize,
    told: &[String],
    wrote: &[Value],
    ms: u128,
    extra: Value,
) {
    if std::fs::create_dir_all(&g.dir).is_err() {
        return;
    }
    let path = g.dir.join("ledger.jsonl");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > LEDGER_ROTATE_BYTES) {
        let _ = std::fs::rename(&path, g.dir.join("ledger.jsonl.1"));
    }
    let mut row = json!({
        "v": 1,
        "ts": Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "event": inp.event,
        "session": inp.session,
        "host": inp.host,
        "tool": inp.tool,
        "call": inp.call,
        "chars": chars,
        "told": told,
        "wrote": wrote,
        "pending": g.p.pending.len(),
        "chainBreaks": g.p.chain_breaks,
        "ms": ms as u64,
        "prev": g.p.ledger_tip.clone().unwrap_or_else(|| "0".repeat(64)),
    });
    if !inp.agent.is_empty() {
        row["agent"] = json!(inp.agent);
    }
    if let (Some(o), Value::Object(more)) = (row.as_object_mut(), extra) {
        for (k, v) in more {
            o.insert(k, v);
        }
    }
    let digest = ledger_digest(&row);
    row["digest"] = json!(digest);
    let line = format!("{row}\n");
    let ok = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| f.write_all(line.as_bytes()))
        .is_ok();
    if ok {
        g.p.ledger_tip = Some(digest);
    }
}

fn parse_since(s: &str) -> Option<i64> {
    let t = s.trim();
    if t.is_empty() {
        return Some(86_400);
    }
    let unit = t.chars().last()?;
    let num = &t[..t.len() - unit.len_utf8()];
    let mult = match unit {
        'm' => 60,
        'h' => 3600,
        'd' => 86_400,
        'w' => 604_800,
        _ => return None,
    };
    num.trim().parse::<i64>().ok().map(|n| n * mult)
}

fn ledger_view(storage: &Arc<Storage>, inp: &Input) -> Value {
    ledger_at(&storage.data_dir().join("reflex"), &inp.since)
}

/// What reflex has recorded for the store at `data_dir`: the Hook Ledger for a
/// window, checked link by link, and the receipt reader's own counters. Reads
/// only `<data_dir>/reflex/`, so it works while a server owns the store.
pub fn report(data_dir: &Path, since: &str) -> Value {
    let dir = data_dir.join("reflex");
    let mut view = ledger_at(&dir, since);
    let state = std::fs::read_to_string(dir.join("state.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Persisted>(&t).ok());
    let heartbeat = std::fs::read_to_string(dir.join("heartbeat"))
        .ok()
        .and_then(|t| t.trim().parse::<i64>().ok());
    if let Some(o) = view.as_object_mut() {
        o.insert(
            "receipts".into(),
            match state {
                Some(p) => json!({
                    "linesRead": p.lines_read,
                    "chainBreaks": p.chain_breaks,
                    "filesTracked": p.files.len(),
                    "stopsRecorded": p.done.len(),
                    "rules": p.rules.len(),
                    "openStops": p.open.len(),
                    "pending": p.pending.len(),
                }),
                None => json!({"note": "reflex has not run against this store yet"}),
            },
        );
        o.insert(
            "lastFiringSecondsAgo".into(),
            json!(heartbeat.map(|h| Utc::now().timestamp() - h)),
        );
    }
    view
}

fn ledger_at(dir: &Path, since: &str) -> Value {
    let path = dir.join("ledger.jsonl");
    let since_label = if since.is_empty() { "24h" } else { since };
    let Some(window) = parse_since(since) else {
        return json!({"error": "invalid since: use forms like 30m, 24h, 7d"});
    };
    let cutoff = Utc::now().timestamp() - window;
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let mut prev: Option<String> = None;
    let mut checked = 0usize;
    let mut breaks = 0usize;
    let mut firings = 0usize;
    let mut chars = 0u64;
    let mut by_event: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    let mut stops = 0u64;
    let mut resolutions = 0u64;
    let mut told = 0u64;
    let mut folds = 0u64;
    let mut saved = 0u64;
    let mut saved_by_tool: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut injected = 0u64;
    let mut skipped = 0u64;
    let mut dropped = 0u64;
    let mut duplicates = 0u64;
    let mut sessions: Vec<String> = Vec::new();
    let mut last: VecDeque<Value> = VecDeque::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(row) = serde_json::from_str::<Value>(line) else {
            breaks += 1;
            prev = None;
            continue;
        };
        checked += 1;
        let stored = row
            .get("digest")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let linked = prev
            .as_deref()
            .is_none_or(|p| row.get("prev").and_then(Value::as_str) == Some(p));
        if !linked || ledger_digest(&row) != stored {
            breaks += 1;
        }
        prev = Some(stored);
        let ts = row
            .get("ts")
            .and_then(Value::as_str)
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|t| t.timestamp())
            .unwrap_or(0);
        if ts < cutoff {
            continue;
        }
        firings += 1;
        chars += row.get("chars").and_then(Value::as_u64).unwrap_or(0);
        *by_event
            .entry(
                row.get("event")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            )
            .or_default() += 1;
        told += row
            .get("told")
            .and_then(Value::as_array)
            .map_or(0, |a| a.len() as u64);
        for w in row
            .get("wrote")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            if w.get("new").and_then(Value::as_bool) == Some(true) {
                match w.get("kind").and_then(Value::as_str) {
                    Some("stop") => stops += 1,
                    Some("resolution") => resolutions += 1,
                    _ => {}
                }
            }
        }
        if let Some(f) = row.get("fold") {
            folds += 1;
            let before = f.get("originalChars").and_then(Value::as_u64).unwrap_or(0);
            let after = row
                .get("sentChars")
                .and_then(Value::as_u64)
                .unwrap_or(before);
            let cut = before.saturating_sub(after);
            saved += cut;
            *saved_by_tool
                .entry(
                    row.get("tool")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                )
                .or_default() += cut;
        }
        let count = |k: &str| {
            row.get(k)
                .and_then(Value::as_array)
                .map_or(0, |a| a.len() as u64)
        };
        injected += count("injected");
        skipped += count("skipped");
        dropped += count("dropped");
        if row.get("duplicate").and_then(Value::as_bool) == Some(true) {
            duplicates += 1;
        }
        if let Some(s) = row.get("session").and_then(Value::as_str)
            && !s.is_empty()
            && !sessions.iter().any(|x| x == s)
        {
            sessions.push(s.to_string());
        }
        last.push_back(row);
        if last.len() > 10 {
            last.pop_front();
        }
    }
    json!({
        "ledger": path.display().to_string(),
        "since": since_label,
        "firings": firings,
        "byEvent": by_event,
        "charsTold": chars,
        "stopsTold": told,
        "stopsRecorded": stops,
        "resolutionsRecorded": resolutions,
        "context": {
            "folds": folds,
            "charsSaved": saved,
            "charsSavedByTool": saved_by_tool,
            "injected": injected,
            "skippedAlreadyHeld": skipped,
            "droppedForTurnBudget": dropped,
            "duplicateFiringsDropped": duplicates,
        },
        "sessions": sessions.len(),
        "chain": {
            "status": if checked == 0 { "empty" } else if breaks == 0 { "ok" } else { "broken" },
            "linesChecked": checked,
            "breaks": breaks,
            "integrity": "reference_digest_not_signature",
        },
        "recent": last,
    })
}

fn status_view(g: &Global, home: &Path) -> Value {
    json!({
        "gateHome": home.display().to_string(),
        "receiptFilesTracked": g.p.files.len(),
        "receiptLinesRead": g.p.lines_read,
        "receiptChainBreaks": g.p.chain_breaks,
        "pending": g.p.pending.len(),
        "stopsRecorded": g.p.done.len(),
        "rules": g.p.rules.len(),
        "openStops": g.p.open.len(),
        "resolutions": g.p.resolutions.len(),
        "sessions": g.sessions.len(),
        "version": VERSION,
    })
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

fn shape(inp: &Input, text: &str) -> Value {
    if text.is_empty() {
        return json!({});
    }
    if inp.format == "plain" {
        return json!({"context": text, "event": inp.event});
    }
    json!({"hookSpecificOutput": {"hookEventName": inp.event, "additionalContext": text}})
}

/// One hook firing, against the gate home at `$OPERATOR_HOME` or `~/.operator`.
pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &Arc<OutputConfig>,
    conn: &ConnectionState,
    args: Option<Value>,
) -> Result<Value, String> {
    let home = super::precedent::operator_home();
    execute_with(storage, cognitive, output_config, conn, args, &home).await
}

/// One hook firing against an explicit gate home. Never fails the hook: an
/// internal error is logged and answered with an empty object.
pub async fn execute_with(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    output_config: &Arc<OutputConfig>,
    conn: &ConnectionState,
    args: Option<Value>,
    home: &Path,
) -> Result<Value, String> {
    let inp = Input::parse(args.as_ref());
    if inp.action == "ledger" {
        return Ok(ledger_view(storage, &inp));
    }
    let global = global_for(storage);
    if inp.action == "status" {
        let g = global.lock().map_err(|e| e.to_string())?;
        return Ok(status_view(&g, home));
    }
    let started = Instant::now();
    if !inp.session.is_empty()
        && let Ok(mut c) = conn.lock()
    {
        c.session = Some(inp.session.clone());
    }
    let jobs = {
        let mut g = global.lock().map_err(|e| e.to_string())?;
        if !inp.session.is_empty() {
            let now = Utc::now().timestamp();
            let s = g.session(&inp.session);
            // One firing per event: a host that reads the same settings twice
            // reports it twice, and the second copy is only ledgered.
            let key = format!("{}|{}|{}|{}", inp.event, inp.call, inp.agent, inp.source);
            let once_per_turn = matches!(
                inp.event.as_str(),
                "UserPromptSubmit"
                    | "SessionStart"
                    | "Stop"
                    | "SubagentStart"
                    | "SubagentStop"
                    | "PreCompact"
                    | "PostCompact"
            );
            if (!inp.call.is_empty() || once_per_turn)
                && s.seen_before(key, !inp.call.is_empty(), now)
            {
                append_ledger(
                    &mut g,
                    &inp,
                    0,
                    &[],
                    &[],
                    started.elapsed().as_millis(),
                    json!({"duplicate": true}),
                );
                return Ok(json!({}));
            }
            match inp.event.as_str() {
                "UserPromptSubmit" => s.start_turn(now),
                "SessionStart" if matches!(inp.source.as_str(), "clear" | "compact") => {
                    s.forget_all()
                }
                // A subagent starts blank and its context ends with it.
                "SubagentStart" if !inp.agent.is_empty() => {
                    s.contexts.remove(&inp.agent);
                }
                "SubagentStop" if !inp.agent.is_empty() => {
                    s.contexts.remove(&inp.agent);
                }
                "PreToolUse" if !inp.call.is_empty() => {
                    let manifest = s.context(&inp.agent).ids();
                    let snap = Snapshot {
                        call: inp.call.clone(),
                        agent: inp.agent.clone(),
                        command: (!inp.command.is_empty()).then(|| command_hash(&inp.command)),
                        manifest,
                    };
                    s.snapshots.push_back(snap);
                    while s.snapshots.len() > SNAPSHOT_KEEP {
                        s.snapshots.pop_front();
                    }
                }
                _ => {}
            }
        }
        scan(&mut g, home);
        take_jobs(&mut g, &inp.session)
    };
    // Only the calling session's rows are written before this hook answers.
    // Everything else drains in one background task per store, so a backlog
    // never slows a tool call.
    let (urgent, background): (Vec<Job>, Vec<Job>) = jobs
        .into_iter()
        .partition(|j| !inp.session.is_empty() && job_row(j).session == inp.session);
    let ctx = Ctx {
        storage,
        cognitive,
        output_config,
    };
    let mut outcomes = Vec::new();
    for job in urgent {
        if started.elapsed().as_millis() > WRITE_BUDGET_MS {
            outcomes.push(Outcome::Wait {
                digest: job_row(&job).digest.clone(),
            });
            continue;
        }
        match run_job(&ctx, job).await {
            Ok(o) => outcomes.push(o),
            Err((digest, e)) => {
                tracing::warn!(error = %e, "reflex could not record a gate receipt; it waits for the next firing");
                outcomes.push(Outcome::Wait { digest });
            }
        }
    }
    spawn_background(&global, storage, cognitive, output_config, background);
    let paired = pairing::on_event(&global, &ctx, &inp).await;
    let witnessed = witness::on_event(&global, &ctx, &inp).await;
    let (compacted, restored) = compaction::on_event(&global, &ctx, &inp).await;
    let mut g = global.lock().map_err(|e| e.to_string())?;
    let mut wrote = Vec::new();
    for o in outcomes {
        if let Some(entry) = apply(&mut g, o) {
            wrote.push(entry);
        }
    }
    wrote.extend(paired.wrote);
    wrote.extend(witnessed.wrote);
    wrote.extend(compacted);
    let lead: Vec<String> = restored
        .into_iter()
        .chain(paired.tell)
        .chain(witnessed.tell)
        .collect();
    let (text, told) = compose(&mut g, &inp, &lead);
    let extra = if inp.session.is_empty() {
        json!({})
    } else {
        let s = g.session(&inp.session);
        let turn = s.turn;
        let ctx = s.context(&inp.agent);
        json!({"turn": turn, "manifest": ctx.digest(), "held": ctx.held.len()})
    };
    append_ledger(
        &mut g,
        &inp,
        text.len(),
        &told,
        &wrote,
        started.elapsed().as_millis(),
        extra,
    );
    // A turn's end and every compaction step are written down at once: the
    // next firing may come from a new server process.
    let force = wrote
        .iter()
        .any(|w| w.get("new").and_then(Value::as_bool) == Some(true))
        || matches!(
            inp.event.as_str(),
            "Stop" | "SubagentStop" | "PreCompact" | "PostCompact" | "SessionStart" | "SessionEnd"
        );
    g.dirty = g.dirty || !wrote.is_empty();
    g.persist(force);
    g.heartbeat();
    Ok(shape(&inp, &text))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    struct Fixture {
        storage: Arc<Storage>,
        home: PathBuf,
        _store_dir: tempfile::TempDir,
        _home_dir: tempfile::TempDir,
        tip: String,
        conn: ConnectionState,
        cog: Arc<Mutex<CognitiveEngine>>,
        oc: Arc<OutputConfig>,
    }

    impl Fixture {
        fn new() -> Fixture {
            let store_dir = tempfile::TempDir::new().unwrap();
            let home_dir = tempfile::TempDir::new().unwrap();
            let storage =
                vestige_core::open_storage(Some(store_dir.path().join("test.db"))).unwrap();
            std::fs::create_dir_all(home_dir.path().join("receipts")).unwrap();
            Fixture {
                storage,
                home: home_dir.path().to_path_buf(),
                _store_dir: store_dir,
                _home_dir: home_dir,
                tip: "0".repeat(64),
                conn: new_connection(),
                cog: cognitive(),
                oc: Arc::new(OutputConfig::default()),
            }
        }

        fn day(&self) -> PathBuf {
            self.home
                .join("receipts")
                .join(format!("{}.jsonl", Utc::now().format("%Y-%m-%d")))
        }

        /// Append one receipt, chained and digested the way the gate does it.
        fn receipt(&mut self, mut rec: Value) -> String {
            rec["prev"] = json!(self.tip);
            rec["ts"] = json!(Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
            let digest = super::super::precedent::gate_digest(&rec).unwrap();
            rec["digest"] = json!(digest);
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.day())
                .unwrap();
            writeln!(f, "{rec}").unwrap();
            self.tip = digest.clone();
            digest
        }

        fn forged(&mut self, mut rec: Value) {
            rec["prev"] = json!(self.tip);
            rec["ts"] = json!(Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
            rec["digest"] = json!("f".repeat(64));
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.day())
                .unwrap();
            writeln!(f, "{rec}").unwrap();
            self.tip = "f".repeat(64);
        }

        async fn fire(&self, args: Value) -> Value {
            execute_with(
                &self.storage,
                &self.cog,
                &self.oc,
                &self.conn,
                Some(args),
                &self.home,
            )
            .await
            .unwrap()
        }

        /// Wait for the background drain to finish.
        async fn settle(&self) {
            let global = global_for(&self.storage);
            for _ in 0..1000 {
                if !global.lock().unwrap().bg_running {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            panic!("background drain did not finish");
        }

        /// One Vestige read the way Claude Code makes it: the PreToolUse firing
        /// with the call's id and agent, then the call itself carrying that id.
        async fn read(
            &self,
            session: &str,
            call: &str,
            agent: &str,
            tool: &str,
            args: Value,
            mut result: Value,
        ) -> Value {
            self.fire(
                json!({"event": "PreToolUse", "session": session, "call": call,
                             "agent": agent, "tool": format!("mcp__vestige__{tool}")}),
            )
            .await;
            fold(
                &self.conn,
                &self.storage,
                tool,
                Some(&args),
                Some(call),
                &mut result,
            );
            result
        }

        async fn tag(&self, tag: &str) -> Option<String> {
            let ctx = Ctx {
                storage: &self.storage,
                cognitive: &self.cog,
                output_config: &self.oc,
            };
            tagged(&ctx, tag).await.unwrap()
        }
    }

    fn stop(session: &str, rule: &str, call: &str, preview: &str) -> Value {
        json!({
            "v": "1", "decision": "STOP", "commitment": rule, "rule": "no-blind-recursive-delete",
            "session": session, "source": "claude", "tool": "Bash", "call": call, "cwd": "/repo",
            "action_preview": preview, "detail": "recursive delete outside the project",
            "action_digest": format!("act-{call}"), "gate_version": "test"
        })
    }

    fn context(v: &Value) -> String {
        v.pointer("/hookSpecificOutput/additionalContext")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }

    #[tokio::test]
    async fn a_stop_becomes_a_memory_linked_to_its_rule_and_is_told_once() {
        let mut fx = Fixture::new();
        let digest = fx.receipt(stop("s1", "OP-003", "toolu_1", "rm -rf ~/work"));
        let out = fx
            .fire(json!({"event": "PostToolUse", "session": "s1", "tool": "Bash"}))
            .await;
        let text = context(&out);
        assert!(text.contains("rm -rf ~/work"), "{out}");
        assert!(text.contains(&digest[..16]), "{out}");
        assert_eq!(out["hookSpecificOutput"]["hookEventName"], "PostToolUse");
        let stop_node = fx.tag(&format!("op-receipt-{}", &digest[..16])).await;
        let rule_node = fx.tag("op-rule-OP-003").await;
        assert!(stop_node.is_some() && rule_node.is_some());
        assert!(text.contains(stop_node.as_deref().unwrap()));
        let again = fx
            .fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        assert_eq!(again, json!({}), "a stop is told once");
    }

    #[tokio::test]
    async fn another_session_is_not_told_but_the_stop_is_still_recorded() {
        let mut fx = Fixture::new();
        let digest = fx.receipt(stop("s1", "OP-003", "toolu_1", "rm -rf ~/work"));
        let out = fx
            .fire(json!({"event": "PostToolUse", "session": "other"}))
            .await;
        assert_eq!(out, json!({}));
        fx.settle().await;
        assert!(
            fx.tag(&format!("op-receipt-{}", &digest[..16]))
                .await
                .is_some()
        );
    }

    #[tokio::test]
    async fn a_line_that_breaks_the_chain_never_becomes_a_memory() {
        let mut fx = Fixture::new();
        fx.forged(stop("s1", "OP-003", "toolu_9", "rm -rf ~/forged"));
        let out = fx
            .fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        assert_eq!(out, json!({}));
        let status = fx.fire(json!({"action": "status"})).await;
        assert_eq!(status["receiptChainBreaks"], 1);
        assert_eq!(status["stopsRecorded"], 0);
    }

    #[tokio::test]
    async fn an_existing_stop_memory_is_reused_not_written_twice() {
        let mut fx = Fixture::new();
        let digest = fx.receipt(stop("s1", "OP-003", "toolu_1", "rm -rf ~/work"));
        let ctx = Ctx {
            storage: &fx.storage,
            cognitive: &fx.cog,
            output_config: &fx.oc,
        };
        let existing = ingest(
            &ctx,
            json!({"content": "stop written by the python hook", "node_type": "event",
                   "tags": ["OP-003", format!("op-receipt-{}", &digest[..16]), "operator", "operator-stop"]}),
        )
        .await
        .unwrap();
        let out = fx
            .fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        assert!(context(&out).contains(&existing), "{out}");
        let ledger = fx.fire(json!({"action": "ledger"})).await;
        assert_eq!(ledger["stopsRecorded"], 0, "{ledger}");
    }

    #[tokio::test]
    async fn a_permit_closes_its_stop_and_the_next_stop_names_that_answer() {
        let mut fx = Fixture::new();
        let first = fx.receipt(stop("s1", "OP-011", "toolu_1", "git reset --hard"));
        fx.fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        let permit = fx.receipt(json!({
            "v": "1", "decision": "PERMIT_GRANTED", "source": "owner-tty", "session": "s1",
            "action_digest": "act-toolu_1", "stop_receipt": first, "single_use": "True"
        }));
        fx.fire(json!({"event": "UserPromptSubmit", "session": "s1"}))
            .await;
        assert!(
            fx.tag(&format!("op-resolution-{}", &permit[..16]))
                .await
                .is_some()
        );
        fx.receipt(stop("s1", "OP-011", "toolu_2", "git reset --hard HEAD~3"));
        let out = fx
            .fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        assert!(
            context(&out).contains("one-time permit from their own terminal"),
            "{out}"
        );
    }

    #[tokio::test]
    async fn a_law_lift_allow_resolves_the_newest_stop_of_that_rule() {
        let mut fx = Fixture::new();
        fx.receipt(stop(
            "s1",
            "recall-before-grep",
            "toolu_1",
            "grep -rn foo .",
        ));
        fx.fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        let allow = fx.receipt(json!({
            "v": "1", "decision": "ALLOW", "commitment": "recall-before-grep", "via": "law_lift",
            "session": "s1", "source": "claude", "tool": "Bash", "action_preview": "grep -rn foo src"
        }));
        fx.fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        assert!(
            fx.tag(&format!("op-resolution-{}", &allow[..16]))
                .await
                .is_some()
        );
        let status = fx.fire(json!({"action": "status"})).await;
        assert_eq!(status["openStops"], 0, "{status}");
    }

    #[tokio::test]
    async fn the_exact_command_stopped_before_is_flagged_on_its_next_attempt() {
        let mut fx = Fixture::new();
        fx.fire(json!({"event": "PreToolUse", "session": "s1", "call": "toolu_1", "command": "rm -rf  ~/work"})).await;
        fx.receipt(stop("s1", "OP-003", "toolu_1", "rm -rf ~/work"));
        fx.fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        let warn = fx
            .fire(json!({"event": "PreToolUse", "session": "s1", "call": "toolu_2", "command": "rm -rf ~/work"}))
            .await;
        assert!(
            context(&warn).contains("already stopped this exact command"),
            "{warn}"
        );
        let other = fx
            .fire(json!({"event": "PreToolUse", "session": "s1", "call": "toolu_3", "command": "ls ~/work"}))
            .await;
        assert_eq!(other, json!({}));
    }

    #[tokio::test]
    async fn a_stop_records_the_memories_the_agent_held_at_that_call() {
        let mut fx = Fixture::new();
        fx.fire(json!({"event": "SessionStart", "session": "s1", "source": "startup"}))
            .await;
        fx.read(
            "s1",
            "toolu_r",
            "",
            "recall",
            json!({"handle": "some-tag"}),
            json!({"nodes": [{"id": "mem-00000000000000aa"}]}),
        )
        .await;
        fx.fire(json!({"event": "PreToolUse", "session": "s1", "call": "toolu_1", "command": "rm -rf ~/work"})).await;
        let digest = fx.receipt(stop("s1", "OP-003", "toolu_1", "rm -rf ~/work"));
        fx.fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        let id = fx
            .tag(&format!("op-receipt-{}", &digest[..16]))
            .await
            .unwrap();
        let ctx = Ctx {
            storage: &fx.storage,
            cognitive: &fx.cog,
            output_config: &fx.oc,
        };
        let got = super::super::recall::execute(
            ctx.storage,
            ctx.cognitive,
            ctx.output_config,
            Some(json!({"handle": id, "detail_level": "full"})),
        )
        .await
        .unwrap();
        assert!(got.to_string().contains("mem-00000000000000aa"), "{got}");
        let manifest = manifest_digest(&["mem-00000000000000aa".to_string()]);
        assert!(
            got.to_string()
                .contains(&format!("Context manifest {manifest}")),
            "{got}"
        );
    }

    const A: &str = "mem-00000000000000a1";
    const B: &str = "mem-00000000000000b2";

    fn body(tag: &str) -> String {
        format!(
            "DECISION {tag}: {}",
            "the long body of a memory. ".repeat(30)
        )
    }

    fn recall_result(id: &str, content: &str) -> Value {
        json!({
            "nodes": [{"id": id, "content": content, "tags": ["topic"], "type": "decision"}],
            "edges": [{"from": id, "to": B, "link_type": "derived_from", "receiptId": "eff-0000000000000001"}],
        })
    }

    #[tokio::test]
    async fn a_body_the_agent_already_holds_is_folded_and_unfolds_whole() {
        let fx = Fixture::new();
        let full = body("a");
        let first = fx
            .read(
                "s1",
                "toolu_1",
                "",
                "recall",
                json!({"handle": "topic"}),
                recall_result(A, &full),
            )
            .await;
        assert!(first.get("reflexFold").is_none(), "{first}");
        assert_eq!(first["nodes"][0]["content"], full.as_str());
        let original = recall_result(A, &full);
        let second = fx
            .read(
                "s1",
                "toolu_2",
                "",
                "recall",
                json!({"handle": "topic"}),
                original.clone(),
            )
            .await;
        let stub = second["nodes"][0]["content"].as_str().unwrap();
        assert!(
            stub.contains(&format!("memory(action='get', id='{A}')")),
            "{stub}"
        );
        assert!(stub.len() < full.len());
        assert_eq!(second["nodes"][0]["tags"], json!(["topic"]));
        assert_eq!(second["nodes"][0]["type"], "decision");
        assert_eq!(
            second["edges"], original["edges"],
            "edge paths and receipts stay whole"
        );
        let block = &second["reflexFold"];
        assert_eq!(block["folded"], json!([A]));
        assert_eq!(block["originalSha256"], sha256_hex(&original.to_string()));
        assert_eq!(block["originalChars"], original.to_string().len());
        assert!(block["foldedChars"].as_u64().unwrap() < block["originalChars"].as_u64().unwrap());
        // Asking for the memory by id returns it byte for byte.
        let unfolded = fx
            .read(
                "s1",
                "toolu_3",
                "",
                "memory",
                json!({"action": "get", "id": A}),
                json!({"id": A, "content": full}),
            )
            .await;
        assert_eq!(unfolded["content"], full.as_str());
        assert!(unfolded.get("reflexFold").is_none());
        let view = fx.fire(json!({"action": "ledger"})).await;
        assert_eq!(view["context"]["folds"], 1, "{view}");
        assert!(
            view["context"]["charsSaved"].as_u64().unwrap() > 0,
            "{view}"
        );
        assert!(
            view["context"]["charsSavedByTool"]["recall"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(view["chain"]["status"], "ok", "{view}");
    }

    #[tokio::test]
    async fn a_subagent_is_never_handed_a_stub_for_what_only_its_parent_read() {
        let fx = Fixture::new();
        let full = body("a");
        fx.read(
            "s1",
            "toolu_1",
            "",
            "recall",
            json!({"handle": "topic"}),
            recall_result(A, &full),
        )
        .await;
        fx.fire(json!({"event": "SubagentStart", "session": "s1", "agent": "ag1"}))
            .await;
        let sub = fx
            .read(
                "s1",
                "toolu_2",
                "ag1",
                "recall",
                json!({"handle": "topic"}),
                recall_result(A, &full),
            )
            .await;
        assert_eq!(sub["nodes"][0]["content"], full.as_str(), "{sub}");
        let again = fx
            .read(
                "s1",
                "toolu_3",
                "ag1",
                "recall",
                json!({"handle": "topic"}),
                recall_result(A, &full),
            )
            .await;
        assert_eq!(
            again["reflexFold"]["folded"],
            json!([A]),
            "the subagent now holds it"
        );
    }

    #[tokio::test]
    async fn compaction_and_clear_empty_what_the_agent_holds() {
        let fx = Fixture::new();
        let full = body("a");
        for (i, source) in ["compact", "clear"].iter().enumerate() {
            fx.read(
                "s1",
                &format!("toolu_{i}a"),
                "",
                "recall",
                json!({"handle": "topic"}),
                recall_result(A, &full),
            )
            .await;
            fx.fire(json!({"event": "SessionStart", "session": "s1", "source": source}))
                .await;
            let after = fx
                .read(
                    "s1",
                    &format!("toolu_{i}b"),
                    "",
                    "recall",
                    json!({"handle": "topic"}),
                    recall_result(A, &full),
                )
                .await;
            assert_eq!(
                after["nodes"][0]["content"],
                full.as_str(),
                "after {source}: {after}"
            );
        }
    }

    #[tokio::test]
    async fn a_read_reflex_cannot_tie_to_an_agent_is_left_as_it_was() {
        let fx = Fixture::new();
        let full = body("a");
        fx.read(
            "s1",
            "toolu_1",
            "",
            "recall",
            json!({}),
            recall_result(A, &full),
        )
        .await;
        for call in [Some("toolu_unseen"), None] {
            let mut result = recall_result(A, &full);
            fold(&fx.conn, &fx.storage, "recall", None, call, &mut result);
            assert_eq!(result, recall_result(A, &full), "{call:?}");
        }
    }

    #[tokio::test]
    async fn a_changed_body_or_a_named_memory_is_shown_in_full() {
        let fx = Fixture::new();
        fx.read(
            "s1",
            "toolu_1",
            "",
            "recall",
            json!({}),
            recall_result(A, &body("a")),
        )
        .await;
        let changed = fx
            .read(
                "s1",
                "toolu_2",
                "",
                "recall",
                json!({}),
                recall_result(A, &body("b")),
            )
            .await;
        assert_eq!(changed["nodes"][0]["content"], body("b").as_str());
        let named = fx
            .read(
                "s1",
                "toolu_3",
                "",
                "recall",
                json!({"handle": A}),
                recall_result(A, &body("b")),
            )
            .await;
        assert_eq!(named["nodes"][0]["content"], body("b").as_str());
    }

    fn packet(first: u64, n: u64, chars: usize) -> Value {
        json!({"memories": (0..n).map(|i| json!({
            "id": format!("mem-{:016x}", first + i), "content": "m".repeat(chars), "tags": ["x"]
        })).collect::<Vec<_>>()})
    }

    #[tokio::test]
    async fn the_memory_hook_never_injects_a_memory_twice_and_shares_the_turn_budget() {
        let fx = Fixture::new();
        let hook = json!({"session": "s1", "event": "prompt", "host": "claude"});
        let mut p1 = packet(0xa0, 3, 700);
        admit_injection(&fx.storage, &hook, &mut p1);
        assert_eq!(p1["memories"].as_array().unwrap().len(), 3, "{p1}");
        let mut p2 = packet(0xa0, 3, 700);
        admit_injection(&fx.storage, &hook, &mut p2);
        assert_eq!(p2["memories"], json!([]), "{p2}");
        assert_eq!(
            p2["reflex"]["skippedAlreadyHeld"].as_array().unwrap().len(),
            3
        );
        let mut p3 = packet(0xb0, 12, 700);
        admit_injection(&fx.storage, &hook, &mut p3);
        let kept = p3["memories"].as_array().unwrap().len();
        let dropped = p3["reflex"]["droppedForTurnBudget"]
            .as_array()
            .unwrap()
            .len();
        assert!(kept > 0 && dropped > 0 && kept + dropped == 12, "{p3}");
        let view = fx.fire(json!({"action": "ledger"})).await;
        assert_eq!(view["context"]["injected"], 3 + kept as u64, "{view}");
        assert_eq!(view["context"]["skippedAlreadyHeld"], 3, "{view}");
        let s = global_for(&fx.storage);
        let g = s.lock().unwrap();
        assert!(g.sessions["s1"].turn_chars <= TURN_BUDGET - NOTICE_RESERVE);
    }

    #[tokio::test]
    async fn a_second_copy_of_the_same_firing_is_only_ledgered() {
        let fx = Fixture::new();
        let pre =
            json!({"event": "PreToolUse", "session": "s1", "call": "toolu_1", "tool": "Bash"});
        fx.fire(pre.clone()).await;
        assert_eq!(fx.fire(pre).await, json!({}));
        fx.fire(json!({"event": "UserPromptSubmit", "session": "s1"}))
            .await;
        fx.fire(json!({"event": "UserPromptSubmit", "session": "s1"}))
            .await;
        let view = fx.fire(json!({"action": "ledger"})).await;
        assert_eq!(view["context"]["duplicateFiringsDropped"], 2, "{view}");
        assert_eq!(view["chain"]["status"], "ok", "{view}");
    }

    const CARGO_ERR: &str = "Exit code 101\nerror[E0499]: cannot borrow `lines` at src/lib.rs:3107";

    fn failure(session: &str, call: &str) -> Value {
        json!({"event": "PostToolUseFailure", "session": session, "tool": "Bash", "call": call,
               "command": "cd /repo && cargo test -p vestige-mcp", "error": CARGO_ERR, "cwd": "/repo"})
    }

    #[tokio::test]
    async fn a_first_failure_is_saved_and_tells_the_agent_nothing() {
        let fx = Fixture::new();
        let out = fx.fire(failure("s1", "toolu_f1")).await;
        assert_eq!(
            out,
            json!({}),
            "a first-time failure puts nothing in front of the model"
        );
        let sig = pairing::classify("Bash", "cargo test -p vestige-mcp", CARGO_ERR, false)
            .unwrap()
            .sig;
        assert!(fx.tag(&format!("fail-{sig}")).await.is_some());
    }

    #[tokio::test]
    async fn the_next_success_records_the_fix_and_a_repeat_is_told_it_at_once() {
        let fx = Fixture::new();
        let sig = pairing::classify("Bash", "cargo test -p vestige-mcp", CARGO_ERR, false)
            .unwrap()
            .sig;
        fx.fire(failure("s1", "toolu_f1")).await;
        fx.fire(
            json!({"event": "PostToolUse", "session": "s1", "tool": "Edit", "call": "toolu_e1",
                       "file": "/repo/src/lib.rs"}),
        )
        .await;
        fx.fire(
            json!({"event": "PostToolUse", "session": "s1", "tool": "Bash", "call": "toolu_ok",
                       "command": "cargo test -p vestige-mcp"}),
        )
        .await;
        let fix = fx.tag(&format!("fix-{sig}")).await;
        assert!(
            fix.is_some(),
            "the success after the failure is recorded as its fix"
        );
        let failure_node = fx.tag(&format!("fail-{sig}")).await.unwrap();
        let ctx = Ctx {
            storage: &fx.storage,
            cognitive: &fx.cog,
            output_config: &fx.oc,
        };
        let view = super::super::recall::execute(
            ctx.storage,
            ctx.cognitive,
            ctx.output_config,
            Some(json!({"handle": failure_node, "detail_level": "full_neighbors"})),
        )
        .await
        .unwrap();
        assert!(view.to_string().contains("closed_by"), "{view}");
        assert!(
            view.to_string().contains("/repo/src/lib.rs"),
            "the fix names the file edited"
        );
        // Another session hits the same failure: it hears the fix before guessing.
        let out = fx.fire(failure("s2", "toolu_f2")).await;
        let text = context(&out);
        assert!(text.contains("happened before"), "{out}");
        assert!(text.contains(fix.as_deref().unwrap()), "{out}");
        assert!(text.contains("cargo test -p vestige-mcp"), "{out}");
        assert_eq!(
            out["hookSpecificOutput"]["hookEventName"],
            "PostToolUseFailure"
        );
    }

    #[tokio::test]
    async fn a_probe_that_fails_by_design_writes_nothing() {
        let fx = Fixture::new();
        for (call, command, error) in [
            (
                "t1",
                "ls ~/not-yet",
                "Exit code 1\nls: ~/not-yet: No such file or directory",
            ),
            ("t2", "grep -rn needle src", "Exit code 1"),
            ("t3", "git diff --quiet", "Exit code 1"),
        ] {
            fx.fire(
                json!({"event": "PostToolUseFailure", "session": "s1", "tool": "Bash",
                           "call": call, "command": command, "error": error}),
            )
            .await;
        }
        fx.fire(json!({"event": "PostToolUseFailure", "session": "s1", "tool": "Bash",
                       "call": "t4", "command": "cargo test", "error": CARGO_ERR, "interrupt": "true"}))
            .await;
        let view = fx.fire(json!({"action": "ledger"})).await;
        assert!(!view.to_string().contains("\"kind\":\"failure\""), "{view}");
    }

    // -- Witness receipts ----------------------------------------------------

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn git_repo() -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        for args in [
            vec!["init", "-q", "-b", "main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "Test"],
            vec!["config", "commit.gpgsign", "false"],
        ] {
            git(dir.path(), &args);
        }
        std::fs::write(dir.path().join("lib.rs"), "fn add() {}\n").unwrap();
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-q", "-m", "first"]);
        dir
    }

    fn allow(fx: &Fixture, commands: &[&str]) {
        let list: Vec<String> = commands.iter().map(|c| format!("{c:?}")).collect();
        std::fs::write(
            fx.storage.data_dir().join("vestige.toml"),
            format!("[witness]\ncommands = [{}]\n", list.join(", ")),
        )
        .unwrap();
    }

    fn run_args(event: &str, call: &str, command: &str, repo: &Path) -> Value {
        json!({"event": event, "session": "s-witness", "tool": "Bash", "call": call,
               "command": command, "cwd": repo.to_string_lossy()})
    }

    /// The witness Reflex recorded for this tree, opened against the store's key.
    async fn witness_at(
        fx: &Fixture,
        tree: &str,
    ) -> Option<(String, vestige_proof::witness::WitnessRecord)> {
        let node = fx.tag(&format!("tree-{tree}")).await?;
        let content = fx.storage.get_node(&node).unwrap().unwrap().content;
        let line = content.lines().last().unwrap();
        let env: vestige_proof::dsse::Envelope = serde_json::from_str(line).unwrap();
        let actor = witness::actor_for(&fx.storage).unwrap();
        let key = vestige_proof::didkey::TrustedKey::parse(actor.did()).unwrap();
        let (record, did) = vestige_proof::witness::WitnessRecord::open(&env, &[key]).unwrap();
        assert_eq!(did, actor.did());
        Some((node, record))
    }

    #[tokio::test]
    async fn an_allowlisted_test_run_becomes_a_signed_witness_bound_to_its_tree() {
        let fx = Fixture::new();
        allow(&fx, &["cargo test --workspace"]);
        let repo = git_repo();
        std::fs::write(repo.path().join("new.rs"), "fn sub() {}\n").unwrap();
        let index = repo.path().join(".git").join("index");
        let index_before = std::fs::read(&index).unwrap();
        fx.fire(run_args(
            "PreToolUse",
            "toolu_w1",
            "cargo test --workspace",
            repo.path(),
        ))
        .await;
        let mut post = run_args(
            "PostToolUse",
            "toolu_w1",
            "cargo test --workspace",
            repo.path(),
        );
        post["stdout"] = json!("test result: ok. 3 passed\n");
        post["stderr"] = json!("   Compiling x\n");
        let out = fx.fire(post).await;
        let tree = vestige_proof::git::working_tree(repo.path()).unwrap();
        let (node, r) = witness_at(&fx, &tree)
            .await
            .expect("a witness tagged with the tree");
        assert_eq!(r.command, "cargo test --workspace");
        assert_eq!(r.exit, Some(0));
        assert_eq!(r.tree, tree);
        assert!(r.bound_to(&tree), "{r:?}");
        assert_eq!(r.tests, None, "plain text output carries no count");
        assert_eq!(
            r.output_sha256,
            vestige_proof::witness::output_digest(
                "test result: ok. 3 passed\n",
                "   Compiling x\n",
                None
            )
            .0
        );
        assert_eq!(
            (r.session.as_str(), r.call.as_str(), r.dir.as_str()),
            ("s-witness", "toolu_w1", "")
        );
        assert_eq!(
            r.head.as_deref(),
            Some(git(repo.path(), &["rev-parse", "HEAD"]).as_str())
        );
        assert!(fx.tag(&witness::session_tag("s-witness")).await.is_some());
        let told = context(&out);
        assert!(
            told.contains("witnessed `cargo test --workspace` green") && told.contains(&node),
            "{out}"
        );
        assert_eq!(
            std::fs::read(&index).unwrap(),
            index_before,
            "the user's index is never written"
        );
        assert_eq!(
            git(repo.path(), &["status", "--porcelain"]),
            "?? new.rs",
            "nothing was staged"
        );
    }

    #[tokio::test]
    async fn a_failed_run_is_witnessed_red_with_its_exit_code() {
        let fx = Fixture::new();
        allow(&fx, &["cargo test --workspace"]);
        let repo = git_repo();
        fx.fire(run_args(
            "PreToolUse",
            "toolu_w2",
            "cargo test --workspace",
            repo.path(),
        ))
        .await;
        let mut post = run_args(
            "PostToolUseFailure",
            "toolu_w2",
            "cargo test --workspace",
            repo.path(),
        );
        post["error"] = json!("Exit code 101\ntest add ... FAILED\n");
        let out = fx.fire(post).await;
        let tree = vestige_proof::git::working_tree(repo.path()).unwrap();
        let (_, r) = witness_at(&fx, &tree).await.unwrap();
        assert_eq!(r.exit, Some(101));
        assert_eq!(
            r.output_sha256,
            vestige_proof::witness::output_digest(
                "",
                "",
                Some("Exit code 101\ntest add ... FAILED\n")
            )
            .0
        );
        assert!(context(&out).contains("red"), "{out}");
    }

    #[tokio::test]
    async fn a_json_report_that_ran_zero_tests_is_seen() {
        let fx = Fixture::new();
        allow(&fx, &["npx jest --json"]);
        let repo = git_repo();
        fx.fire(run_args(
            "PreToolUse",
            "toolu_w3",
            "npx jest --json",
            repo.path(),
        ))
        .await;
        let mut post = run_args("PostToolUse", "toolu_w3", "npx jest --json", repo.path());
        post["stdout"] = json!(r#"{"numTotalTests":0,"numPassedTests":0,"numFailedTests":0}"#);
        let out = fx.fire(post).await;
        let tree = vestige_proof::git::working_tree(repo.path()).unwrap();
        let (_, r) = witness_at(&fx, &tree).await.unwrap();
        let t = r.tests.expect("the JSON report is counted");
        assert_eq!((t.collected, t.source.as_str()), (0, "jest-json"));
        assert!(t.ran_none());
        assert!(context(&out).contains("zero tests"), "{out}");
    }

    #[tokio::test]
    async fn a_junit_report_written_during_the_run_is_counted() {
        let fx = Fixture::new();
        let repo = git_repo();
        std::fs::create_dir_all(repo.path().join(".vestige")).unwrap();
        std::fs::write(repo.path().join(".gitignore"), "reports/\n").unwrap();
        std::fs::write(
            repo.path().join(".vestige/witness.toml"),
            "[witness]\ncommands = [\"make check\"]\n\n[witness.junit]\n\"make check\" = \"reports/junit.xml\"\n",
        )
        .unwrap();
        fx.fire(run_args(
            "PreToolUse",
            "toolu_w4",
            "make check",
            repo.path(),
        ))
        .await;
        std::fs::create_dir_all(repo.path().join("reports")).unwrap();
        std::fs::write(
            repo.path().join("reports/junit.xml"),
            r#"<testsuites tests="7" failures="0"><testsuite tests="7"/></testsuites>"#,
        )
        .unwrap();
        fx.fire(run_args(
            "PostToolUse",
            "toolu_w4",
            "make check",
            repo.path(),
        ))
        .await;
        let tree = vestige_proof::git::working_tree(repo.path()).unwrap();
        let (_, r) = witness_at(&fx, &tree)
            .await
            .expect("the repository's own allowlist counts");
        assert_eq!(
            r.tests.map(|t| (t.collected, t.source)),
            Some((7, "junit".to_string()))
        );
    }

    #[tokio::test]
    async fn a_run_that_changed_the_files_is_recorded_but_binds_to_no_tree() {
        let fx = Fixture::new();
        allow(&fx, &["cargo test --workspace"]);
        let repo = git_repo();
        fx.fire(run_args(
            "PreToolUse",
            "toolu_w5",
            "cargo test --workspace",
            repo.path(),
        ))
        .await;
        std::fs::write(repo.path().join("lib.rs"), "fn add() { changed() }\n").unwrap();
        fx.fire(run_args(
            "PostToolUse",
            "toolu_w5",
            "cargo test --workspace",
            repo.path(),
        ))
        .await;
        let tree = vestige_proof::git::working_tree(repo.path()).unwrap();
        let (_, r) = witness_at(&fx, &tree).await.unwrap();
        assert!(r.tree_before.is_some() && r.tree_before.as_deref() != Some(tree.as_str()));
        assert!(!r.bound_to(&tree));
    }

    #[tokio::test]
    async fn only_the_exact_allowlisted_command_in_the_foreground_is_witnessed() {
        let fx = Fixture::new();
        allow(&fx, &["cargo test --workspace"]);
        let repo = git_repo();
        let tree = vestige_proof::git::working_tree(repo.path()).unwrap();
        let mut piped = run_args(
            "PostToolUse",
            "toolu_w6",
            "cargo test --workspace | tail -5",
            repo.path(),
        );
        piped["stdout"] = json!("ok");
        fx.fire(piped).await;
        let mut bg = run_args(
            "PostToolUse",
            "toolu_w7",
            "cargo test --workspace",
            repo.path(),
        );
        bg["background"] = json!("true");
        fx.fire(bg).await;
        let mut cut = run_args(
            "PostToolUse",
            "toolu_w8",
            "cargo test --workspace",
            repo.path(),
        );
        cut["interrupted"] = json!("true");
        fx.fire(cut).await;
        let mut other = run_args(
            "PostToolUse",
            "toolu_w9",
            "cargo test --workspace",
            repo.path(),
        );
        other["tool"] = json!("Read");
        fx.fire(other).await;
        assert!(fx.tag(&format!("tree-{tree}")).await.is_none());
        let view = fx.fire(json!({"action": "ledger"})).await;
        assert!(!view.to_string().contains("\"kind\":\"witness\""), "{view}");
    }

    // -- Proof bundles -------------------------------------------------------

    fn trusted(fx: &Fixture) -> vestige_proof::didkey::TrustedKey {
        let actor = witness::actor_for(&fx.storage).unwrap();
        vestige_proof::didkey::TrustedKey::parse(actor.did()).unwrap()
    }

    /// A red run, a fix, the same command green, one Operator stop, then a
    /// commit of exactly the tested files.
    async fn worked_session(fx: &mut Fixture, repo: &Path) {
        let cmd = "cargo test --workspace";
        fx.fire(run_args("PreToolUse", "toolu_p1", cmd, repo)).await;
        let mut red = run_args("PostToolUseFailure", "toolu_p1", cmd, repo);
        red["error"] = json!("Exit code 101\nerror[E0425]: cannot find function `sub`");
        fx.fire(red).await;
        std::fs::write(repo.join("lib.rs"), "fn add() {}\nfn sub() {}\n").unwrap();
        fx.fire(
            json!({"event": "PostToolUse", "session": "s-witness", "tool": "Edit",
                       "call": "toolu_e", "file": repo.join("lib.rs").to_string_lossy()}),
        )
        .await;
        fx.receipt(stop("s-witness", "OP-003", "toolu_rm", "rm -rf build"));
        fx.fire(run_args("PreToolUse", "toolu_p2", cmd, repo)).await;
        let mut green = run_args("PostToolUse", "toolu_p2", cmd, repo);
        green["stdout"] = json!("test result: ok. 2 passed\n");
        fx.fire(green).await;
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-q", "-m", "add sub"]);
    }

    #[tokio::test]
    async fn prove_bundles_the_witnesses_of_a_commit_with_its_pairings_and_stops() {
        let mut fx = Fixture::new();
        allow(&fx, &["cargo test --workspace", "pnpm test"]);
        let repo = git_repo();
        worked_session(&mut fx, repo.path()).await;
        let head = git(repo.path(), &["rev-parse", "HEAD"]);
        let tree = git(repo.path(), &["rev-parse", "HEAD^{tree}"]);
        let out = super::super::prove::prove(
            &fx.storage,
            repo.path(),
            "HEAD",
            super::super::prove::Out::Both,
        )
        .unwrap();
        assert_eq!(
            (out["commit"].as_str(), out["tree"].as_str()),
            (Some(head.as_str()), Some(tree.as_str()))
        );
        let summary = &out["summary"];
        assert_eq!(summary["green"], 1, "{out}");
        assert_eq!(summary["unwitnessed"], 1, "pnpm test was never run: {out}");
        assert_eq!(
            summary["witnesses"], 1,
            "the red run was at another tree: {out}"
        );
        assert_eq!(summary["pairings"], 1, "{out}");
        assert_eq!(summary["stops"], 1, "{out}");

        let file = repo
            .path()
            .join(vestige_proof::PROOF_DIR)
            .join(vestige_proof::bundle_file_name(&tree));
        let bytes = std::fs::read(&file).unwrap();
        assert_eq!(out["bundleSha256"], vestige_proof::sha256_hex(&bytes));
        let note = vestige_proof::git::read_note(repo.path(), &head).unwrap();
        assert_eq!(
            format!("{note}\n").into_bytes(),
            bytes,
            "the note holds the same bundle"
        );

        let env = vestige_proof::statement::parse_bundle(&bytes).unwrap();
        let key = trusted(&fx);
        let (st, did) =
            vestige_proof::statement::Statement::open(&env, std::slice::from_ref(&key)).unwrap();
        assert_eq!(did, key.did);
        let p = &st.predicate;
        assert_eq!(st.subject[0].digest["gitTree"], tree);
        assert_eq!(p.commit, head);
        let green = p
            .claims
            .iter()
            .find(|c| c.command == "cargo test --workspace")
            .unwrap();
        assert_eq!(green.status, vestige_proof::claims::Status::Green);
        let w = &p.witnesses[0];
        assert_eq!(green.witness.as_deref(), Some(w.memory.as_str()));
        assert_eq!(w.strata_digest.as_ref().map(String::len), Some(64), "{w:?}");
        let (record, _) =
            vestige_proof::witness::WitnessRecord::open(&w.envelope, std::slice::from_ref(&key))
                .unwrap();
        assert!(record.bound_to(&tree));
        let pairing = &p.pairings[0];
        assert!(
            pairing
                .signature
                .as_deref()
                .is_some_and(|s| s.starts_with("fail-")),
            "{pairing:?}"
        );
        assert_eq!(pairing.session, "s-witness");
        assert!(
            fx.storage
                .get_node(&pairing.fix)
                .unwrap()
                .unwrap()
                .tags
                .contains(&"reflex-fix".to_string())
        );
        let stop = &p.stops[0];
        assert_eq!(stop.rule, "OP-003");
        assert!(stop.receipt.starts_with("op-receipt-"), "{stop:?}");
        assert_eq!(p.sessions, vec!["s-witness".to_string()]);
        assert_eq!(p.claim_boundary, vestige_proof::statement::CLAIM_BOUNDARY);

        // The offline verifier accepts exactly this bundle for exactly this tree.
        use vestige_proof::verify::{Policy, verify_bundle};
        let trusted = std::slice::from_ref(&key);
        let report = verify_bundle(&bytes, trusted, Some(&tree), &Policy::default());
        assert!(report.ok, "{:?}", report.failures);
        let strict = Policy {
            require: vec![],
            strict: true,
        };
        let report = verify_bundle(&bytes, trusted, Some(&tree), &strict);
        assert!(
            report
                .failures
                .iter()
                .any(|f| f.contains("`pnpm test` is unwitnessed")),
            "{:?}",
            report.failures
        );
    }

    #[tokio::test]
    async fn a_committed_bundle_proves_the_same_tree_and_an_untested_commit_proves_nothing_green() {
        let mut fx = Fixture::new();
        allow(&fx, &["cargo test --workspace"]);
        let repo = git_repo();
        worked_session(&mut fx, repo.path()).await;
        let first = super::super::prove::prove(
            &fx.storage,
            repo.path(),
            "HEAD",
            super::super::prove::Out::File,
        )
        .unwrap();
        git(repo.path(), &["add", "-A"]);
        git(repo.path(), &["commit", "-q", "-m", "proof"]);
        let again = super::super::prove::prove(
            &fx.storage,
            repo.path(),
            "HEAD",
            super::super::prove::Out::None,
        )
        .unwrap();
        assert_eq!(
            again["tree"], first["tree"],
            "the bundle never changes the tree it proves"
        );
        assert_eq!(again["claims"], first["claims"]);
        std::fs::write(
            repo.path().join("lib.rs"),
            "fn add() {}\nfn sub() {}\nfn mul() {}\n",
        )
        .unwrap();
        git(repo.path(), &["commit", "-qam", "untested change"]);
        let untested = super::super::prove::prove(
            &fx.storage,
            repo.path(),
            "HEAD",
            super::super::prove::Out::None,
        )
        .unwrap();
        assert_eq!(untested["summary"]["green"], 0, "{untested}");
        assert_eq!(untested["claims"][0]["status"], "unwitnessed", "{untested}");
        assert_eq!(untested["summary"]["pairings"], 0);
    }

    #[tokio::test]
    async fn the_receipt_tool_runs_prove_and_refuses_mixed_arguments() {
        let mut fx = Fixture::new();
        allow(&fx, &["cargo test --workspace"]);
        let repo = git_repo();
        worked_session(&mut fx, repo.path()).await;
        let path = repo.path().to_string_lossy().to_string();
        let out = super::super::receipt::execute(
            &fx.storage,
            Some(json!({"action": "prove", "repoPath": path, "commit": "HEAD", "out": "none"})),
        )
        .await
        .unwrap();
        assert_eq!(out["summary"]["green"], 1, "{out}");
        for bad in [
            json!({"action": "prove"}),
            json!({"action": "prove", "repoPath": "relative/path"}),
            json!({"action": "prove", "repoPath": path, "out": "gist"}),
            json!({"action": "prove", "repoPath": path, "receipt_id": "mem-1"}),
            json!({"action": "get", "receipt_id": "mem-1", "commit": "HEAD"}),
        ] {
            assert!(
                super::super::receipt::execute(&fx.storage, Some(bad.clone()))
                    .await
                    .is_err(),
                "{bad}"
            );
        }
    }

    #[tokio::test]
    async fn a_background_finding_is_told_once_at_the_next_firing() {
        let fx = Fixture::new();
        fx.fire(json!({"event": "UserPromptSubmit", "session": "s1"}))
            .await;
        {
            let global = global_for(&fx.storage);
            let mut g = global.lock().unwrap();
            g.session("s1")
                .notes
                .push("Reflex: a lesson has faded: mem-0000000000000001.".into());
        }
        let out = fx
            .fire(json!({"event": "PreToolUse", "session": "s1", "call": "t1", "tool": "Bash"}))
            .await;
        assert!(context(&out).contains("lesson has faded"), "{out}");
        let again = fx
            .fire(json!({"event": "PreToolUse", "session": "s1", "call": "t2", "tool": "Bash"}))
            .await;
        assert_eq!(again, json!({}));
    }

    fn other_connection(session: &str) -> ConnectionState {
        let conn = new_connection();
        conn.lock().unwrap().session = Some(session.to_string());
        conn
    }

    #[tokio::test]
    async fn a_retired_belief_reaches_every_other_session_holding_it_once() {
        let fx = Fixture::new();
        fx.read(
            "s1",
            "toolu_r1",
            "",
            "recall",
            json!({"handle": "topic"}),
            recall_result(A, &body("a")),
        )
        .await;
        fx.read(
            "s2",
            "toolu_r2",
            "",
            "recall",
            json!({"handle": "topic"}),
            recall_result(A, &body("a")),
        )
        .await;
        // Session s2 corrects the memory; s1 still holds the old version.
        stale::record(
            &other_connection("s2"),
            &fx.storage,
            "memory",
            &json!({"action": "edit", "nodeId": B, "supersedes": A,
                    "newContentPreview": "The service listens on port 4000. The old note said 3000.",
                    "receiptId": "eff-0000000000000009"}),
        );
        let out = fx
            .fire(
                json!({"event": "PreToolUse", "session": "s1", "call": "toolu_n1", "tool": "Bash"}),
            )
            .await;
        let text = context(&out);
        assert!(text.contains(A) && text.contains(B), "{out}");
        assert!(text.contains("eff-0000000000000009"), "{out}");
        assert!(text.contains("port 4000"), "{out}");
        let again = fx
            .fire(
                json!({"event": "PreToolUse", "session": "s1", "call": "toolu_n2", "tool": "Bash"}),
            )
            .await;
        assert_eq!(again, json!({}), "told once");
        let editor = fx
            .fire(
                json!({"event": "PreToolUse", "session": "s2", "call": "toolu_n3", "tool": "Bash"}),
            )
            .await;
        assert_eq!(
            editor,
            json!({}),
            "the session that made the change is not told"
        );
        let stranger = fx
            .fire(
                json!({"event": "PreToolUse", "session": "s3", "call": "toolu_n4", "tool": "Bash"}),
            )
            .await;
        assert_eq!(
            stranger,
            json!({}),
            "a session that never held it is not told"
        );
    }

    #[tokio::test]
    async fn a_memory_read_after_its_retirement_is_not_news() {
        let fx = Fixture::new();
        stale::record(
            &other_connection("s2"),
            &fx.storage,
            "memory",
            &json!({"action": "demoted", "nodeId": A}),
        );
        fx.read(
            "s1",
            "toolu_r1",
            "",
            "recall",
            json!({"handle": "topic"}),
            recall_result(A, &body("a")),
        )
        .await;
        let out = fx
            .fire(
                json!({"event": "PreToolUse", "session": "s1", "call": "toolu_n1", "tool": "Bash"}),
            )
            .await;
        assert_eq!(out, json!({}), "{out}");
    }

    /// Two real memories, a decision and a fact, read by session s1.
    async fn held_pair(fx: &Fixture) -> (String, String) {
        let ctx = Ctx {
            storage: &fx.storage,
            cognitive: &fx.cog,
            output_config: &fx.oc,
        };
        let d = ingest(
            &ctx,
            json!({"content": "DECISION: deploys go out only from main, never from a feature branch.",
                   "node_type": "decision", "tags": ["deploys"]}),
        )
        .await
        .unwrap();
        let f = ingest(
            &ctx,
            json!({"content": "FACT: the staging API listens on port 4000.", "node_type": "fact", "tags": ["deploys"]}),
        )
        .await
        .unwrap();
        fx.read(
            "s1",
            "toolu_r1",
            "",
            "recall",
            json!({"handle": "deploys"}),
            json!({"nodes": [{"id": f, "content": "FACT: the staging API listens on port 4000."},
                             {"id": d, "content": "DECISION: deploys go out only from main."}]}),
        )
        .await;
        (d, f)
    }

    #[tokio::test]
    async fn compaction_gives_back_exactly_what_the_summary_dropped() {
        let fx = Fixture::new();
        let (d, f) = held_pair(&fx).await;
        // Claude Code's order: PreCompact, SessionStart(compact), PostCompact.
        fx.fire(json!({"event": "PreCompact", "session": "s1", "trigger": "auto"}))
            .await;
        assert!(fx.tag("compaction-checkpoint").await.is_some());
        let start = fx
            .fire(json!({"event": "SessionStart", "session": "s1", "source": "compact"}))
            .await;
        assert_eq!(start, json!({}), "the restore waits for the audit");
        fx.fire(
            json!({"event": "PostCompact", "session": "s1", "trigger": "auto",
                       "summary": format!("We tuned staging. The port fact {f} still holds.")}),
        )
        .await;
        assert!(fx.tag("compaction-audit").await.is_some());
        let out = fx
            .fire(json!({"event": "UserPromptSubmit", "session": "s1"}))
            .await;
        let text = context(&out);
        assert!(text.contains(&d) && text.contains("[decision]"), "{out}");
        assert!(
            !text.contains(&format!("- {f}")),
            "the kept memory is not repeated: {out}"
        );
        assert!(text.contains("audit mem-"), "{out}");
    }

    #[tokio::test]
    async fn without_an_audit_every_held_memory_comes_back_decisions_first() {
        let fx = Fixture::new();
        let (d, f) = held_pair(&fx).await;
        fx.fire(json!({"event": "PreCompact", "session": "s1", "trigger": "manual"}))
            .await;
        fx.fire(json!({"event": "SessionStart", "session": "s1", "source": "compact"}))
            .await;
        let out = fx
            .fire(json!({"event": "PreToolUse", "session": "s1", "call": "toolu_after", "tool": "Bash"}))
            .await;
        let text = context(&out);
        let (di, fi) = (text.find(&d).unwrap(), text.find(&f).unwrap());
        assert!(di < fi, "the decision comes first: {out}");
        assert!(text.contains("may have dropped"), "{out}");
    }

    #[tokio::test]
    async fn an_edited_memory_comes_back_as_its_current_version() {
        let fx = Fixture::new();
        let (d, _f) = held_pair(&fx).await;
        let ctx = Ctx {
            storage: &fx.storage,
            cognitive: &fx.cog,
            output_config: &fx.oc,
        };
        let d2 = ingest(
            &ctx,
            json!({"content": "DECISION: deploys go out from main or a release branch.", "node_type": "decision", "tags": ["deploys"]}),
        )
        .await
        .unwrap();
        fx.fire(json!({"event": "PreCompact", "session": "s1"}))
            .await;
        stale::record(
            &other_connection("s2"),
            &fx.storage,
            "memory",
            &json!({"action": "edit", "nodeId": d2, "supersedes": d}),
        );
        fx.fire(json!({"event": "SessionStart", "session": "s1", "source": "compact"}))
            .await;
        let out = fx
            .fire(json!({"event": "UserPromptSubmit", "session": "s1"}))
            .await;
        let text = context(&out);
        assert!(
            text.contains(&format!("{d2} [decision] (current version of {d})")),
            "{out}"
        );
    }

    #[test]
    fn a_replay_folds_recorded_repeats_and_restarts_after_compaction() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("session.jsonl");
        let text = serde_json::to_string(&recall_result(A, &body("a"))).unwrap();
        fn call(lines: &mut Vec<Value>, id: &str, text: &str) {
            lines.push(json!({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": id, "name": "mcp__vestige__recall", "input": {"handle": "topic"}}]}}));
            lines.push(json!({"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": id, "content": [{"type": "text", "text": text}]}]}}));
        }
        let mut lines = vec![json!({"type": "user", "message": {"content": "go"}})];
        call(&mut lines, "t1", &text);
        call(&mut lines, "t2", &text);
        lines.push(json!({"type": "system", "subtype": "compact_boundary"}));
        call(&mut lines, "t3", &text);
        let out: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(&path, out).unwrap();
        let report = replay(&[path]);
        let recall = &report["byTool"]["recall"];
        assert_eq!(recall["calls"], 3, "{report}");
        assert_eq!(recall["foldedCalls"], 1, "{report}");
        assert!(recall["charsSaved"].as_u64().unwrap() > 0, "{report}");
        assert_eq!(report["compactions"], 1);
    }

    #[tokio::test]
    async fn context_stays_inside_the_event_budget() {
        let mut fx = Fixture::new();
        for i in 0..8 {
            fx.receipt(stop(
                "s1",
                "OP-003",
                &format!("toolu_{i}"),
                &"x".repeat(150),
            ));
        }
        let out = fx
            .fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        assert!(
            context(&out).len() <= event_budget("PostToolUse"),
            "{}",
            context(&out).len()
        );
        let next = fx
            .fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        assert!(
            !context(&next).is_empty(),
            "the rest are told on later firings"
        );
    }

    #[tokio::test]
    async fn the_hook_ledger_chains_every_firing_and_detects_an_edit() {
        let mut fx = Fixture::new();
        fx.receipt(stop("s1", "OP-003", "toolu_1", "rm -rf ~/work"));
        fx.fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        fx.fire(json!({"event": "Stop", "session": "s1"})).await;
        let view = fx.fire(json!({"action": "ledger"})).await;
        assert_eq!(view["chain"]["status"], "ok", "{view}");
        assert_eq!(view["firings"], 2);
        assert_eq!(view["stopsRecorded"], 1);
        let path = fx.storage.data_dir().join("reflex").join("ledger.jsonl");
        let edited = std::fs::read_to_string(&path)
            .unwrap()
            .replacen("\"Stop\"", "\"Stip\"", 1);
        std::fs::write(&path, edited).unwrap();
        let view = fx.fire(json!({"action": "ledger"})).await;
        assert_eq!(view["chain"]["status"], "broken", "{view}");
    }

    #[tokio::test]
    async fn events_without_context_and_unfilled_substitutions_answer_empty() {
        let fx = Fixture::new();
        assert_eq!(
            fx.fire(json!({"event": "Stop", "session": "s1"})).await,
            json!({})
        );
        assert_eq!(
            fx.fire(json!({"event": "PostToolUse", "session": "${session_id}"}))
                .await,
            json!({})
        );
        assert_eq!(fx.fire(json!({})).await, json!({}));
    }

    #[tokio::test]
    async fn plain_format_answers_for_the_command_shim() {
        let mut fx = Fixture::new();
        fx.receipt(stop("s1", "OP-003", "toolu_1", "rm -rf ~/work"));
        let out = fx
            .fire(json!({"event": "PostToolUse", "session": "s1", "format": "plain"}))
            .await;
        assert!(
            out["context"].as_str().unwrap().contains("rm -rf ~/work"),
            "{out}"
        );
        assert!(out.get("hookSpecificOutput").is_none());
    }

    #[tokio::test]
    async fn pre_tool_use_tells_an_untold_stop_before_the_next_call() {
        let mut fx = Fixture::new();
        let digest = fx.receipt(stop("s1", "OP-003", "toolu_1", "rm -rf ~/work"));
        let out = fx
            .fire(json!({"event": "PreToolUse", "session": "s1", "call": "toolu_2", "command": "echo next"}))
            .await;
        assert!(context(&out).contains(&digest[..16]), "{out}");
        assert_eq!(out["hookSpecificOutput"]["hookEventName"], "PreToolUse");
    }

    #[tokio::test]
    async fn other_sessions_backlog_drains_in_the_background_and_is_ledgered() {
        let mut fx = Fixture::new();
        let a = fx.receipt(stop("other-1", "OP-003", "toolu_1", "rm -rf ~/a"));
        let b = fx.receipt(stop(
            "other-2",
            "OP-004",
            "toolu_2",
            "git reflog expire --all",
        ));
        let out = fx
            .fire(json!({"event": "PostToolUse", "session": "s1"}))
            .await;
        assert_eq!(out, json!({}), "nothing to tell this session");
        fx.settle().await;
        assert!(fx.tag(&format!("op-receipt-{}", &a[..16])).await.is_some());
        assert!(fx.tag(&format!("op-receipt-{}", &b[..16])).await.is_some());
        let view = fx.fire(json!({"action": "ledger"})).await;
        assert_eq!(view["stopsRecorded"], 2, "{view}");
        assert_eq!(view["byEvent"]["background"], 1, "{view}");
        assert_eq!(view["chain"]["status"], "ok", "{view}");
    }

    #[test]
    fn memory_ids_and_rules_are_recognized_exactly() {
        assert!(is_memory_id("mem-00000000000acd2b"));
        assert!(is_memory_id("68540393-203d-428b-9a1a-854da288e655"));
        assert!(!is_memory_id("mem-123"));
        assert!(!is_memory_id("op-receipt-b2c87261ed002a83"));
        assert!(is_rule("OP-003") && is_rule("recall-before-grep"));
        assert!(!is_rule("-x") && !is_rule("bad rule"));
    }

    #[test]
    fn rows_that_answer_no_stop_are_not_queued() {
        let d = "a".repeat(64);
        assert!(GateRow::from_receipt(&json!({"decision": "PASS", "digest": d})).is_none());
        assert!(GateRow::from_receipt(&json!({"decision": "ALLOW", "digest": d})).is_none());
        assert!(
            GateRow::from_receipt(&json!({"decision": "ALLOW", "digest": d, "via": "law_lift"}))
                .is_some()
        );
        assert!(
            GateRow::from_receipt(&json!({"decision": "PERMIT_GRANTED", "digest": d})).is_none()
        );
        assert!(
            GateRow::from_receipt(
                &json!({"decision": "STOP", "digest": d, "commitment": "bad rule"})
            )
            .is_none()
        );
    }
}
