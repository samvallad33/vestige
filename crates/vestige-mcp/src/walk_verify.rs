//! # `vestige prove`: the walk proposes, the test decides
//!
//! A causal walk returns leads: commits the failure memory reaches over
//! recorded edges. A lead is not a cause. `prove` takes those leads and runs
//! the user's own test on them, so the answer it prints is a tested one:
//!
//! 1. the causal walk from the failure memory (`--logged-write`), in process;
//! 2. a time gate: a lead committed after the failure was reported, or
//!    outside `good..bad`, is dropped;
//! 3. the protocol is frozen before any test runs: the test's sha256, the
//!    two ends and the leads are hashed and saved as a memory, so none of
//!    them can be chosen after seeing a result. Then the test must pass on
//!    `--good` and fail on `--bad`;
//! 4. a bisect over the leads only, then the parent of the earliest failing
//!    lead. "fails, parent passes" is a tested boundary;
//! 5. stock `git bisect run` over the whole range as confirmation, reusing
//!    every verdict already recorded, and a replay that counts how many runs
//!    plain bisect needs;
//! 6. why: the smallest set of the commit's changes that still fails on its
//!    parent (Zeller's ddmin), the rest of the commit without that set (must
//!    pass), and an undo on the bad ref (must pass).
//!
//! The result ends in a verdict card of five rungs (LEAD, BOUNDARY,
//! CONFIRMED, ISOLATED, REVERSED), each with whether it holds and the run
//! numbers that back it.
//!
//! Every test run is written twice: appended to a probe log in which each
//! entry carries the sha256 of the one before it, and saved in the store as
//! an `event` memory through the calls `vestige ingest` makes. The report
//! file holds the whole chain and `vestige prove --check` re-verifies it.
//!
//! Two kinds of line are printed and they are labelled:
//! `[recorded link]` is a lead from the walk, `[tested]` is a test run.
//!
//! The test follows `git bisect run`: exit 0 is good, 125 is "cannot test
//! this commit", anything else in 1..=127 is bad.
//!
//! ## Byte compatibility
//!
//! This is a port of the `walk-verify.py` reference tool and its reports are
//! interchangeable with that tool's. An entry's hash is
//! `sha256(prev_hash + body)`, where `body` is the entry without its `hash`
//! field in the form Python's
//! `json.dumps(sort_keys=True, separators=(",", ":"))` writes: keys sorted,
//! no spaces, every character outside printable ASCII escaped as `\uXXXX`
//! (UTF-16 surrogate pairs above the BMP). [`canonical_json`] is that form.
//!
//! ## One writer
//!
//! A CLI command that opens a Strata log is its only writer until it exits,
//! so the processes `git bisect run` starts cannot write to the store. They
//! run the test, hand the raw result back through a file, and this process
//! saves the memory and extends the chain as each result arrives.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use chrono::{DateTime, Utc};
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use vestige_core::{IngestInput, SecretPolicy, Storage};

/// One probe-log entry, as it is hashed and as the report stores it.
pub type Entry = Map<String, Value>;

/// The `prev` of the first entry of a chain.
pub const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Name of the hidden subcommand `git bisect run` calls back into.
pub const CHILD_COMMAND: &str = "_prove";

/// Report format written here: the fields `walk-verify.py` version 5 writes
/// for a run without its `--flaky` mode, which is not ported.
const TOOL: &str = "vestige prove (walk-verify 5)";

/// The exit-code rule, as the frozen protocol states it.
const PROTOCOL_RULE: &str = "exit 0 good, 125 cannot test, any other code bad";

const READING: &str = "walk candidates are recorded links (leads). probes and first_bad_commit are tested results of this repro script. why.minimal_failing_changes is tested too: those changes alone, applied to the parent, fail it.";

/// Arguments of `vestige prove`.
#[derive(Debug, Clone, clap::Args)]
pub struct ProveArgs {
    /// The failure memory the causal walk starts from
    #[arg(long, value_name = "MEMORY_ID", required_unless_present = "check")]
    pub logged_write: Option<String>,
    /// Git repository the failure is in. Tests run in a temporary worktree
    /// of it, never in this checkout
    #[arg(long, value_name = "DIR", required_unless_present = "check")]
    pub repo: Option<PathBuf>,
    /// A ref where the test passes
    #[arg(long, value_name = "REF", required_unless_present = "check")]
    pub good: Option<String>,
    /// A ref where the test fails
    #[arg(long, value_name = "REF", required_unless_present = "check")]
    pub bad: Option<String>,
    /// One shell command to use as the test: exit 0 good, 125 cannot test,
    /// any other 1..127 bad
    #[arg(long, value_name = "COMMAND", conflicts_with = "oracle")]
    pub test: Option<String>,
    /// An executable test script, same exit codes as --test
    #[arg(long, value_name = "SCRIPT")]
    pub oracle: Option<PathBuf>,
    /// When the failure was reported (RFC 3339). A commit made after this
    /// cannot be its cause
    #[arg(long, value_name = "RFC3339", required_unless_present = "check")]
    pub reported_at: Option<String>,
    /// Where to write the JSON report (refuses to overwrite)
    #[arg(long, value_name = "FILE", required_unless_present = "check")]
    pub report: Option<PathBuf>,
    /// Tag carried by every memory this run writes
    #[arg(long, default_value = "walk-verify")]
    pub slug: String,
    /// Bisect over at most this many of the newest leads
    #[arg(long, default_value_t = 12)]
    pub max_candidates: usize,
    /// At most this many test runs for the search inside the commit
    #[arg(long, default_value_t = 24)]
    pub max_line_runs: i64,
    /// Stop at the first bad commit; skip the search inside it
    #[arg(long)]
    pub no_why: bool,
    /// How many leads to list
    #[arg(long, default_value_t = 6)]
    pub show: usize,
    /// Re-verify a report offline: the hash chain is intact, the first bad
    /// commit tested bad and its parent tested good
    #[arg(
        long,
        value_name = "REPORT",
        conflicts_with_all = ["logged_write", "repo", "good", "bad", "test", "oracle", "reported_at", "report"]
    )]
    pub check: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Palette {
    b: &'static str,
    d: &'static str,
    c: &'static str,
    g: &'static str,
    r: &'static str,
    y: &'static str,
    o: &'static str,
}

const COLOR: Palette = Palette {
    b: "\x1b[1m",
    d: "\x1b[2m",
    c: "\x1b[1;36m",
    g: "\x1b[1;32m",
    r: "\x1b[1;31m",
    y: "\x1b[1;33m",
    o: "\x1b[0m",
};

const PLAIN: Palette = Palette {
    b: "",
    d: "",
    c: "",
    g: "",
    r: "",
    y: "",
    o: "",
};

impl Palette {
    /// Color on a terminal or under `FORCE_COLOR`; never under `NO_COLOR`.
    fn detect() -> Self {
        let set = |name: &str| std::env::var_os(name).is_some_and(|value| !value.is_empty());
        if set("FORCE_COLOR") {
            COLOR
        } else if set("NO_COLOR") || !std::io::stdout().is_terminal() {
            PLAIN
        } else {
            COLOR
        }
    }

    fn verdict(&self, verdict: &str) -> &'static str {
        match verdict {
            "good" => self.g,
            "bad" => self.r,
            _ => self.y,
        }
    }
}

/// A refusal with its own exit code. The message is the whole output.
#[derive(Debug)]
struct Stop {
    code: i32,
    message: String,
}

/// An error that ends the run with `code` after printing `message`.
fn stop(code: i32, message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Stop {
        code,
        message: message.into(),
    })
}

impl fmt::Display for Stop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Stop {}

// ---------------------------------------------------------------------------
// Text helpers that keep Python's meaning of "line", "strip" and "[:n]"
// ---------------------------------------------------------------------------

/// The first `n` characters (not bytes) of `s`.
fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((index, _)) => &s[..index],
        None => s,
    }
}

fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `str.splitlines()`: every line boundary Python knows, `\r\n` as one, and
/// no empty last line.
fn split_lines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((index, c)) = chars.next() {
        if !is_line_break(c) {
            continue;
        }
        lines.push(&s[start..index]);
        start = index + c.len_utf8();
        if c == '\r'
            && let Some(&(next, '\n')) = chars.peek()
        {
            chars.next();
            start = next + 1;
        }
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// `str.strip()`.
fn strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

/// The last line a test printed (stdout, then stderr), cut to 160 characters.
fn last_line(stdout: &[u8], stderr: &[u8]) -> String {
    let mut text = String::from_utf8_lossy(stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(stderr));
    split_lines(strip(&text))
        .last()
        .map(|line| head(line, 160).to_string())
        .unwrap_or_default()
}

/// How Python prints a JSON value with `%s`: `None` for null or missing.
fn py_display(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => "False".to_string(),
        Some(other) => canonical_json(other),
    }
}

fn opt_display(value: Option<&str>) -> &str {
    value.unwrap_or("None")
}

// ---------------------------------------------------------------------------
// Canonical JSON and the hash chain
// ---------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `json.dumps(value, sort_keys=True, separators=(",", ":"))`, byte for byte.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(number, out),
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => write_object(map, None, out),
    }
}

fn sorted_keys<'a>(map: &'a Map<String, Value>, skip: Option<&str>) -> Vec<&'a String> {
    let mut keys: Vec<&String> = map
        .keys()
        .filter(|key| Some(key.as_str()) != skip)
        .collect();
    // Byte order of UTF-8 is code point order, which is how Python sorts.
    keys.sort();
    keys
}

fn write_object(map: &Map<String, Value>, skip: Option<&str>, out: &mut String) {
    out.push('{');
    for (index, key) in sorted_keys(map, skip).into_iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        write_string(key, out);
        out.push(':');
        write_canonical(&map[key], out);
    }
    out.push('}');
}

/// Python's `ensure_ascii` string form: printable ASCII as is, the short
/// escapes, and `\uXXXX` (lowercase hex, surrogate pairs) for the rest.
fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn write_number(number: &Number, out: &mut String) {
    if let Some(value) = number.as_i64() {
        out.push_str(&value.to_string());
    } else if let Some(value) = number.as_u64() {
        out.push_str(&value.to_string());
    } else if let Some(value) = number.as_f64() {
        out.push_str(&float_repr(value));
    } else {
        out.push_str(&number.to_string());
    }
}

/// `repr(float)`: the shortest digits that round-trip, as a plain decimal
/// with at least one fractional digit, or with a signed two-digit exponent
/// below 1e-4 and from 1e16 up.
fn float_repr(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_string();
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if !(-4..16).contains(&exponent) {
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{sign}{first}{fraction}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        )
    } else if exponent < 0 {
        let zeros = "0".repeat(exponent.unsigned_abs() as usize - 1);
        format!("{sign}0.{zeros}{digits}")
    } else {
        let whole = exponent as usize + 1;
        if digits.len() <= whole {
            let zeros = "0".repeat(whole - digits.len());
            format!("{sign}{digits}{zeros}.0")
        } else {
            let (integer, fraction) = digits.split_at(whole);
            format!("{sign}{integer}.{fraction}")
        }
    }
}

/// `json.dumps(value, indent=1, sort_keys=True)`: the layout of the report.
fn pretty_json(value: &Value) -> String {
    let mut out = String::new();
    write_pretty(value, 0, &mut out);
    out
}

fn write_pretty(value: &Value, depth: usize, out: &mut String) {
    let line = |depth: usize, out: &mut String| {
        out.push('\n');
        out.push_str(&" ".repeat(depth));
    };
    match value {
        Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                line(depth + 1, out);
                write_pretty(item, depth + 1, out);
            }
            line(depth, out);
            out.push(']');
        }
        Value::Object(map) if !map.is_empty() => {
            out.push('{');
            for (index, key) in sorted_keys(map, None).into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                line(depth + 1, out);
                write_string(key, out);
                out.push_str(": ");
                write_pretty(&map[key], depth + 1, out);
            }
            line(depth, out);
            out.push('}');
        }
        other => write_canonical(other, out),
    }
}

/// The hash of one entry: `sha256(prev + canonical JSON of the entry without
/// its "hash" field)`, hex.
pub fn chain_hash(prev: &str, entry: &Entry) -> String {
    let mut body = String::from(prev);
    write_object(entry, Some("hash"), &mut body);
    sha256_hex(body.as_bytes())
}

/// The hash of a frozen protocol: sha256 over the canonical JSON of its
/// fields, without the `sha256` and `memory` the report adds afterwards.
pub fn protocol_hash(protocol: &Map<String, Value>) -> String {
    let body: Map<String, Value> = protocol
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "sha256" | "memory"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    sha256_hex(canonical_json(&Value::Object(body)).as_bytes())
}

/// What a test's exit code means, as `git bisect run` reads it.
pub fn verdict_of(exit: i64) -> &'static str {
    match exit {
        0 => "good",
        125 => "skip",
        _ => "bad",
    }
}

// ---------------------------------------------------------------------------
// The probe log
// ---------------------------------------------------------------------------

/// The probe log of one run: a JSONL file the bisect child processes read,
/// and the same entries in memory. Only the parent process appends.
struct Session {
    path: PathBuf,
    entries: Vec<Entry>,
}

impl Session {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            entries: Vec::new(),
        }
    }

    fn read(path: &Path) -> Vec<Entry> {
        let Ok(text) = fs::read_to_string(path) else {
            return Vec::new();
        };
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str::<Entry>(line).ok())
            .collect()
    }

    fn cached(&self, commit: &str) -> Option<&Entry> {
        self.entries
            .iter()
            .find(|entry| entry.get("commit").and_then(Value::as_str) == Some(commit))
    }

    /// Number the entry, link it to the one before, hash it, write it.
    fn append(&mut self, mut entry: Entry) -> anyhow::Result<Entry> {
        let prev = self
            .entries
            .last()
            .and_then(|last| last.get("hash"))
            .and_then(Value::as_str)
            .unwrap_or(ZERO_HASH)
            .to_string();
        entry.insert("n".to_string(), json!(self.entries.len() + 1));
        entry.insert("prev".to_string(), json!(prev));
        entry.remove("hash");
        let hash = chain_hash(&prev, &entry);
        entry.insert("hash".to_string(), json!(hash));
        append_line(&self.path, &Value::Object(entry.clone()))
            .with_context(|| format!("cannot write the probe log {}", self.path.display()))?;
        self.entries.push(entry.clone());
        Ok(entry)
    }
}

fn append_line(path: &Path, value: &Value) -> std::io::Result<()> {
    let mut line = serde_json::to_string(value).map_err(std::io::Error::other)?;
    line.push('\n');
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(line.as_bytes())
}

fn text_of<'a>(entry: &'a Entry, key: &str) -> &'a str {
    entry.get(key).and_then(Value::as_str).unwrap_or("")
}

// ---------------------------------------------------------------------------
// git and the test
// ---------------------------------------------------------------------------

fn git_command(dir: &Path) -> Command {
    let mut command = Command::new("git");
    command.arg("-C").arg(dir).stdin(Stdio::null());
    command
}

/// Run git; success and its trimmed stdout.
fn git(dir: &Path, args: &[&str]) -> (bool, String) {
    match git_command(dir).args(args).output() {
        Ok(output) => (
            output.status.success(),
            strip(&String::from_utf8_lossy(&output.stdout)).to_string(),
        ),
        Err(_) => (false, String::new()),
    }
}

fn git_ok(dir: &Path, args: &[&str]) -> bool {
    git(dir, args).0
}

#[cfg(unix)]
fn signal_of(status: &ExitStatus) -> i64 {
    use std::os::unix::process::ExitStatusExt;
    i64::from(status.signal().unwrap_or(1))
}

#[cfg(not(unix))]
fn signal_of(_status: &ExitStatus) -> i64 {
    1
}

/// The exit code, or minus the signal that ended the process.
fn exit_code(status: &ExitStatus) -> i64 {
    status.code().map_or_else(|| -signal_of(status), i64::from)
}

struct TestRun {
    exit: i64,
    said: String,
    at: String,
}

fn now_stamp() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
}

/// Run the test in the worktree as it stands.
fn run_test(oracle: &Path, worktree: &Path) -> anyhow::Result<TestRun> {
    let at = now_stamp();
    #[cfg(unix)]
    let mut command = Command::new(oracle);
    #[cfg(not(unix))]
    let mut command = {
        let mut command = Command::new("sh");
        command.arg(oracle);
        command
    };
    let output = command
        .current_dir(worktree)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("cannot run the test {}", oracle.display()))?;
    interrupted()?;
    Ok(TestRun {
        exit: exit_code(&output.status),
        said: last_line(&output.stdout, &output.stderr),
        at,
    })
}

/// A probe entry before it is saved and chained.
fn new_entry(
    commit: &str,
    subject: &str,
    run: &TestRun,
    oracle_sha256: &str,
    phase: &str,
) -> Entry {
    let mut entry = Entry::new();
    entry.insert("commit".to_string(), json!(commit));
    entry.insert("subject".to_string(), json!(head(subject, 120)));
    entry.insert("verdict".to_string(), json!(verdict_of(run.exit)));
    entry.insert("exit".to_string(), json!(run.exit));
    entry.insert("oracle_said".to_string(), json!(run.said));
    entry.insert("oracle_sha256".to_string(), json!(oracle_sha256));
    entry.insert("phase".to_string(), json!(phase));
    entry.insert("at".to_string(), json!(run.at));
    entry
}

/// The memory text of a probe. `on_commit` is a run on one checked-out
/// commit; otherwise the entry's `commit` is a label for a worktree state.
fn probe_text(entry: &Entry, on_commit: bool) -> String {
    let what = if on_commit {
        format!("Commit {}", head(text_of(entry, "commit"), 10))
    } else {
        text_of(entry, "commit").to_string()
    };
    format!(
        "Oracle probe: {what} ({}) tested {}. Repro script exit code {}, result: {}. Script sha256 {}, run at {}, phase {}. A tested result for this one state, not a claim about any other.",
        head(text_of(entry, "subject"), 100),
        text_of(entry, "verdict").to_uppercase(),
        entry.get("exit").and_then(Value::as_i64).unwrap_or(0),
        text_of(entry, "oracle_said"),
        head(text_of(entry, "oracle_sha256"), 16),
        text_of(entry, "at"),
        text_of(entry, "phase"),
    )
}

/// Save one `event` memory the way `vestige ingest` does: the default-scope
/// ingest with the secret gate on, then auto-connect on exact identities.
/// `tags` is the comma-separated list `ingest --tags` takes. `None` when the
/// gate refused the write; the probe log still records the run.
fn remember(storage: &Storage, text: &str, tags: &str) -> Option<String> {
    let input = IngestInput {
        content: text.to_string(),
        node_type: "event".to_string(),
        source: Some("walk-verify".to_string()),
        sentiment_score: 0.0,
        sentiment_magnitude: 0.0,
        tags: tags
            .split(',')
            .map(|tag| tag.trim().to_string())
            .filter(|tag| !tag.is_empty())
            .collect(),
        valid_from: None,
        valid_until: None,
        validity_inferred: false,
        source_envelope: None,
    };
    let node = match storage.ingest_with_secret_policy(input, SecretPolicy::Reject) {
        Ok(node) => node,
        Err(err) => {
            eprintln!("  (this run was not saved as a memory: {err})");
            return None;
        }
    };
    if let Err(err) = crate::auto_connect::auto_connect_new_memory(
        storage,
        &node.id,
        vestige_core::DEFAULT_MEMORY_SCOPE,
        &node.content,
        &node.tags,
    ) {
        eprintln!("  (auto-connect skipped for {}: {err})", node.id);
    }
    Some(node.id)
}

// ---------------------------------------------------------------------------
// Cleanup: the worktree and the scratch directory always go away
// ---------------------------------------------------------------------------

struct Scratch {
    repo: PathBuf,
    dir: PathBuf,
    worktree: Option<PathBuf>,
}

static SCRATCH: Mutex<Option<Scratch>> = Mutex::new(None);
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

fn remove_worktree() {
    let taken = SCRATCH.lock().ok().and_then(|mut guard| {
        let scratch = guard.as_mut()?;
        Some((scratch.repo.clone(), scratch.worktree.take()?))
    });
    let Some((repo, worktree)) = taken else {
        return;
    };
    let Some(path) = worktree.to_str() else {
        return;
    };
    let removed = git_ok(&repo, &["worktree", "remove", "--force", path]);
    if !removed || worktree.exists() {
        // The directory goes first; prune then drops the stale registration.
        let _ = fs::remove_dir_all(&worktree);
        git_ok(&repo, &["worktree", "prune"]);
    }
}

fn remove_scratch() {
    remove_worktree();
    if let Some(scratch) = SCRATCH.lock().ok().and_then(|mut guard| guard.take()) {
        let _ = fs::remove_dir_all(&scratch.dir);
    }
}

/// Removes the worktree and the scratch directory when dropped, on every
/// path out of [`prove`]. A panic hook and a SIGINT/SIGTERM flag cover the
/// exits a destructor does not see.
struct ScratchGuard;

impl ScratchGuard {
    fn arm(repo: PathBuf, dir: PathBuf) -> Self {
        if let Ok(mut guard) = SCRATCH.lock() {
            *guard = Some(Scratch {
                repo,
                dir,
                worktree: None,
            });
        }
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            remove_scratch();
            previous(info);
        }));
        watch_signals();
        ScratchGuard
    }

    fn worktree_added(&self, worktree: &Path) {
        if let Ok(mut guard) = SCRATCH.lock()
            && let Some(scratch) = guard.as_mut()
        {
            scratch.worktree = Some(worktree.to_path_buf());
        }
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        remove_scratch();
    }
}

#[cfg(unix)]
extern "C" fn on_signal(_signal: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// Ctrl-C reaches the test and git too; they die, the run notices the flag
/// after the child returns, and the worktree is removed on the way out.
#[cfg(unix)]
fn watch_signals() {
    let handler = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
    // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGTERM, handler);
    }
}

#[cfg(not(unix))]
fn watch_signals() {}

fn interrupted() -> anyhow::Result<()> {
    if INTERRUPTED.load(Ordering::SeqCst) {
        return Err(stop(
            130,
            "interrupted; the temporary worktree was removed.",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The commit's changes, one unit per hunk
// ---------------------------------------------------------------------------

/// One independently applicable change of a commit: a hunk, or the whole
/// file diff for a file that is added, deleted, or has no hunks (binary,
/// mode change, pure rename).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    pub file: String,
    /// The file's diff header, repeated before the hunks of one file. Empty
    /// for a whole-file unit, whose `body` carries its own header.
    pub header: Vec<u8>,
    pub body: Vec<u8>,
    /// First line of the hunk in the new file; 0 for a whole-file unit.
    pub start: u64,
    /// The lines the hunk adds, for display.
    pub added: Vec<String>,
}

/// Offsets of the lines of `text` that start with `prefix`.
fn line_starts_with(text: &[u8], prefix: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut start = 0;
    while start < text.len() {
        if text[start..].starts_with(prefix) {
            offsets.push(start);
        }
        match text[start..].iter().position(|byte| *byte == b'\n') {
            Some(newline) => start += newline + 1,
            None => break,
        }
    }
    offsets
}

fn line_at(text: &[u8], offset: usize) -> &[u8] {
    let end = text[offset..]
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(text.len(), |newline| offset + newline);
    &text[offset..end]
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// The path a file diff names: `+++ b/<path>`, else `--- a/<path>`, else its
/// first line.
fn diff_path(file_diff: &[u8]) -> String {
    for prefix in [&b"+++ b/"[..], &b"--- a/"[..]] {
        for offset in line_starts_with(file_diff, prefix) {
            let rest = &line_at(file_diff, offset)[prefix.len()..];
            if !rest.is_empty() {
                return String::from_utf8_lossy(rest).into_owned();
            }
        }
    }
    String::from_utf8_lossy(line_at(file_diff, 0)).into_owned()
}

/// `N` of a hunk header `@@ -a[,b] +N[,d] @@`; 0 when it does not parse.
fn hunk_start(hunk: &[u8]) -> u64 {
    let line = String::from_utf8_lossy(line_at(hunk, 0)).into_owned();
    let parse = || -> Option<u64> {
        let rest = line.strip_prefix("@@ -")?;
        let rest = rest.trim_start_matches(|c: char| c.is_ascii_digit());
        let rest = match rest.strip_prefix(',') {
            Some(count) => count.trim_start_matches(|c: char| c.is_ascii_digit()),
            None => rest,
        };
        let rest = rest.strip_prefix(" +")?;
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    };
    parse().unwrap_or(0)
}

/// Split a unified diff into units: one per hunk, or the whole file diff for
/// new, deleted and hunkless files.
pub fn split_hunks(diff: &[u8]) -> Vec<Unit> {
    let mut units = Vec::new();
    let files = line_starts_with(diff, b"diff --git ");
    for (index, &begin) in files.iter().enumerate() {
        let end = files.get(index + 1).copied().unwrap_or(diff.len());
        let file_diff = &diff[begin..end];
        let path = diff_path(file_diff);
        let hunks = line_starts_with(file_diff, b"@@ ");
        let header = &file_diff[..hunks.first().copied().unwrap_or(file_diff.len())];
        let whole = hunks.is_empty()
            || contains(header, b"new file mode")
            || contains(header, b"deleted file mode");
        if whole {
            units.push(Unit {
                file: path,
                header: Vec::new(),
                body: file_diff.to_vec(),
                start: 0,
                added: Vec::new(),
            });
            continue;
        }
        for (position, &hunk_begin) in hunks.iter().enumerate() {
            let hunk_end = hunks.get(position + 1).copied().unwrap_or(file_diff.len());
            let hunk = &file_diff[hunk_begin..hunk_end];
            let text = String::from_utf8_lossy(hunk);
            let added = split_lines(&text)
                .into_iter()
                .skip(1)
                .filter_map(|line| line.strip_prefix('+'))
                .map(str::to_string)
                .collect();
            units.push(Unit {
                file: path.clone(),
                header: header.to_vec(),
                body: hunk.to_vec(),
                start: hunk_start(hunk),
                added,
            });
        }
    }
    units
}

/// The patch that applies exactly these units, in diff order: each file's
/// header once, then its hunks.
pub fn patch_of(units: &[&Unit]) -> Vec<u8> {
    let mut patch = Vec::new();
    let mut last: Option<&[u8]> = None;
    for unit in units {
        if unit.header.is_empty() || last != Some(unit.header.as_slice()) {
            patch.extend_from_slice(&unit.header);
            last = Some(unit.header.as_slice());
        }
        patch.extend_from_slice(&unit.body);
    }
    patch
}

fn names_of(units: &[&Unit]) -> String {
    units
        .iter()
        .map(|unit| format!("{}:{}", unit.file, unit.start))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Zeller's ddmin: shrink `items` to a 1-minimal subset for which `fails`
/// still holds. `fails` spends the budget; the search stops when it is used
/// up, and the subset returned then still fails but may shrink further.
pub fn ddmin<T, F>(items: Vec<T>, budget: &Cell<i64>, mut fails: F) -> Vec<T>
where
    T: Clone + PartialEq,
    F: FnMut(&[T]) -> bool,
{
    let mut items = items;
    let mut n = 2usize;
    while items.len() >= 2 && budget.get() > 0 {
        let size = std::cmp::max(1, items.len() / n);
        let subsets: Vec<Vec<T>> = items.chunks(size).map(<[T]>::to_vec).collect();
        let mut moved = false;
        for subset in &subsets {
            if budget.get() <= 0 {
                break;
            }
            if fails(subset) {
                items = subset.clone();
                n = 2;
                moved = true;
                break;
            }
        }
        if !moved {
            for subset in &subsets {
                if budget.get() <= 0 || subsets.len() <= 2 {
                    break;
                }
                let complement: Vec<T> = items
                    .iter()
                    .filter(|item| !subset.contains(item))
                    .cloned()
                    .collect();
                if fails(&complement) {
                    items = complement;
                    n = std::cmp::max(n - 1, 2);
                    moved = true;
                    break;
                }
            }
        }
        if !moved {
            if n >= items.len() {
                break;
            }
            n = std::cmp::min(items.len(), n * 2);
        }
    }
    items
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// One lead: a memory the walk reached whose text names a commit.
#[derive(Debug, Clone)]
struct Lead {
    rank: usize,
    memory: String,
    depth: u64,
    /// The sha as the memory wrote it.
    short: String,
    commit: Option<String>,
}

/// The sha a memory names when its text starts `Commit <sha>:` or
/// `Commit <sha> `: 7 to 40 lowercase hex characters.
fn commit_named(content: &str) -> Option<String> {
    let rest = content.trim_start().strip_prefix("Commit ")?;
    let sha: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || ('a'..='f').contains(c))
        .collect();
    if !(7..=40).contains(&sha.len()) {
        return None;
    }
    match rest[sha.len()..].chars().next() {
        Some(':' | ' ' | '\n') => Some(sha),
        _ => None,
    }
}

/// The leads of a recorded walk: its causes, in rank order, that name a commit.
fn leads_of(walk: &Value) -> Vec<Lead> {
    walk["causes"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, cause)| {
            Some(Lead {
                rank: index + 1,
                memory: cause["id"].as_str()?.to_string(),
                depth: cause["depth"].as_u64().unwrap_or(0),
                short: commit_named(cause["content"].as_str()?)?,
                commit: None,
            })
        })
        .collect()
}

/// What the search inside the first bad commit found.
struct Why {
    /// Verdict of the undo on the bad ref, or `rewritten` when nothing applied.
    revert: Option<String>,
    undo_how: Option<&'static str>,
    /// Verdict of the commit without the minimal set, or `does not apply`.
    without: Option<String>,
    units: usize,
    minimal: Vec<Unit>,
    complete: bool,
    undo_patch: Option<Vec<u8>>,
}

struct Prover<'a> {
    storage: &'a Storage,
    pal: Palette,
    repo: PathBuf,
    worktree: PathBuf,
    oracle: PathBuf,
    oracle_sha256: String,
    slug: String,
    session: Session,
}

impl Prover<'_> {
    /// Save the run as a memory, then chain it.
    fn admit(&mut self, mut entry: Entry, on_commit: bool) -> anyhow::Result<Entry> {
        let text = probe_text(&entry, on_commit);
        let tags = format!(
            "oracle-probe,probe-{},{}",
            text_of(&entry, "verdict"),
            self.slug
        );
        let memory = remember(self.storage, &text, &tags);
        entry.insert(
            "memory".to_string(),
            memory.map_or(Value::Null, Value::String),
        );
        self.session.append(entry)
    }

    fn print_tested(&self, entry: &Entry) {
        let Palette { b, d, o, .. } = self.pal;
        let verdict = text_of(entry, "verdict");
        println!(
            "  {b}[tested]{o} {} {}{:<4}{o} {}  {d}-> {}{o}",
            head(text_of(entry, "commit"), 10),
            self.pal.verdict(verdict),
            verdict.to_uppercase(),
            text_of(entry, "oracle_said"),
            py_display(entry.get("memory")),
        );
    }

    fn print_reused(&self, entry: &Entry) {
        let Palette { d, o, .. } = self.pal;
        let verdict = text_of(entry, "verdict");
        println!(
            "  {d}[tested]{o} {} {}{:<4}{o} {d}reused probe {}{o}",
            head(text_of(entry, "commit"), 10),
            self.pal.verdict(verdict),
            verdict.to_uppercase(),
            py_display(entry.get("n")),
        );
    }

    fn show(&self, what: &str, entry: &Entry) {
        let Palette { b, d, o, .. } = self.pal;
        let verdict = text_of(entry, "verdict");
        println!(
            "  {b}[tested]{o} {what}: {}{:<4}{o} {}  {d}-> {}{o}",
            self.pal.verdict(verdict),
            verdict.to_uppercase(),
            text_of(entry, "oracle_said"),
            py_display(entry.get("memory")),
        );
    }

    /// Test one commit, or reuse its recorded verdict.
    fn probe(&mut self, sha: &str, phase: &str) -> anyhow::Result<Entry> {
        if let Some(hit) = self.session.cached(sha).cloned() {
            self.print_reused(&hit);
            return Ok(hit);
        }
        let checkout = git_command(&self.worktree)
            .args(["checkout", "-q", "--detach", sha])
            .output()
            .context("cannot run git")?;
        if !checkout.status.success() {
            return Err(stop(
                2,
                format!(
                    "checkout of {} failed: {}",
                    head(sha, 10),
                    head(strip(&String::from_utf8_lossy(&checkout.stderr)), 200)
                ),
            ));
        }
        let run = run_test(&self.oracle, &self.worktree)?;
        let (_, subject) = git(&self.repo, &["show", "-s", "--format=%s", sha]);
        let entry = new_entry(sha, &subject, &run, &self.oracle_sha256, phase);
        let entry = self.admit(entry, true)?;
        self.print_tested(&entry);
        Ok(entry)
    }

    /// Test whatever is in the worktree now and record it under `label`.
    fn probe_state(&mut self, label: &str, subject: &str, phase: &str) -> anyhow::Result<Entry> {
        let run = run_test(&self.oracle, &self.worktree)?;
        let entry = new_entry(label, subject, &run, &self.oracle_sha256, phase);
        self.admit(entry, false)
    }

    /// Take the results the bisect children have handed back since the last
    /// call: save and chain each new run, print each line.
    fn take_handed_back(&mut self, pending: &Path, taken: &mut usize) -> anyhow::Result<()> {
        let Ok(text) = fs::read_to_string(pending) else {
            return Ok(());
        };
        // A line still being written has no newline yet; it waits.
        let complete = text.rfind('\n').map_or("", |end| &text[..=end]);
        for line in complete.lines().skip(*taken) {
            *taken += 1;
            let Ok(record) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if let Some(n) = record.get("reused").and_then(Value::as_u64) {
                let hit = usize::try_from(n)
                    .ok()
                    .and_then(|n| n.checked_sub(1))
                    .and_then(|index| self.session.entries.get(index))
                    .cloned();
                if let Some(hit) = hit {
                    self.print_reused(&hit);
                }
            } else if let Some(entry) = record.get("entry").and_then(Value::as_object) {
                let entry = self.admit(entry.clone(), true)?;
                self.print_tested(&entry);
            }
        }
        Ok(())
    }

    /// Stock `git bisect run` over `good..bad`, with this binary's hidden
    /// subcommand as the script. Returns the first bad commit git names and
    /// git's own output.
    fn bisect(
        &mut self,
        child: &ChildFiles,
        good: &str,
        bad: &str,
    ) -> anyhow::Result<(Option<String>, String)> {
        let exe = std::env::current_exe().context("cannot find this executable")?;
        git_ok(&self.worktree, &["bisect", "start", bad, good]);
        let errors = fs::File::create(&child.errors).context("cannot create the bisect log")?;
        let mut bisect = git_command(&self.worktree)
            .args(["bisect", "run"])
            .arg(&exe)
            .args([CHILD_COMMAND, "probe"])
            .arg(&child.cfg)
            .stdout(Stdio::piped())
            .stderr(Stdio::from(errors))
            .spawn()
            .context("cannot run git bisect")?;
        let mut transcript = String::new();
        let mut first_bad = None;
        let mut taken = 0usize;
        let mut read = || -> anyhow::Result<()> {
            let Some(stdout) = bisect.stdout.take() else {
                return Ok(());
            };
            // git prints after each run, so each line is a cue to pick up
            // what the child just handed back.
            for line in BufReader::new(stdout).split(b'\n') {
                let line = String::from_utf8_lossy(&line?).into_owned();
                self.take_handed_back(&child.pending, &mut taken)?;
                if let Some(sha) = first_bad_named(&line) {
                    first_bad = Some(sha);
                }
                transcript.push_str(&line);
                transcript.push('\n');
            }
            Ok(())
        };
        let outcome = read();
        if outcome.is_err() {
            let _ = bisect.kill();
        }
        let _ = bisect.wait();
        let outcome = outcome.and_then(|()| self.take_handed_back(&child.pending, &mut taken));
        git_ok(&self.worktree, &["bisect", "reset"]);
        outcome?;
        interrupted()?;
        if first_bad.is_none() {
            let errors = fs::read_to_string(&child.errors).unwrap_or_default();
            transcript.push_str(strip(&errors));
        }
        Ok((first_bad, transcript))
    }

    /// How many runs plain `git bisect` needs on this range: replay it
    /// against the tested answer, without running the test.
    fn replay_plain_bisect(
        &mut self,
        child: &ChildFiles,
        good: &str,
        bad: &str,
    ) -> anyhow::Result<Option<u64>> {
        let exe = std::env::current_exe().context("cannot find this executable")?;
        git_ok(&self.worktree, &["bisect", "start", bad, good]);
        let _ = git_command(&self.worktree)
            .args(["bisect", "run"])
            .arg(&exe)
            .args([CHILD_COMMAND, "sim"])
            .arg(&child.cfg)
            .output();
        git_ok(&self.worktree, &["bisect", "reset"]);
        interrupted()?;
        Ok(fs::read(&child.counter)
            .ok()
            .map(|bytes| bytes.len() as u64))
    }

    /// Put the worktree on `commit`, clean.
    fn reset_to(&self, commit: &str) {
        git_ok(&self.worktree, &["revert", "--abort"]);
        git_ok(
            &self.worktree,
            &["checkout", "-q", "--detach", "-f", commit],
        );
        git_ok(&self.worktree, &["reset", "-q", "--hard"]);
        git_ok(&self.worktree, &["clean", "-fdq"]);
    }

    /// Apply a patch to the worktree, plainly or else three-way.
    fn apply(&self, patch: &[u8], reverse: bool) -> bool {
        for three_way in [false, true] {
            let mut command = git_command(&self.worktree);
            command.args(["apply", "--whitespace=nowarn"]);
            if reverse {
                command.arg("-R");
            }
            if three_way {
                command.arg("--3way");
            }
            let applied = command
                .arg("-")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .and_then(|mut child| {
                    if let Some(mut stdin) = child.stdin.take() {
                        // git may stop reading a patch it rejects; its exit
                        // status is the answer either way.
                        let _ = stdin.write_all(patch);
                    }
                    child.wait()
                })
                .is_ok_and(|status| status.success());
            if applied {
                return true;
            }
            git_ok(&self.worktree, &["reset", "-q", "--hard"]);
        }
        false
    }

    fn worktree_diff(&self) -> Vec<u8> {
        git_command(&self.worktree)
            .args(["diff", "HEAD"])
            .output()
            .map(|output| output.stdout)
            .unwrap_or_default()
    }

    /// Three tests on the first bad commit. `lines`: ddmin over its changes
    /// applied to its parent, the smallest set that still fails. `without`:
    /// the commit's other changes, without that set, which must pass.
    /// `undo`: on the bad ref, undo the whole commit or, when later commits
    /// are in the way, only the lines found, which must pass.
    fn explain(
        &mut self,
        bad: &str,
        first_bad: &str,
        max_line_runs: i64,
        bad_name: &str,
    ) -> anyhow::Result<Why> {
        let Palette { d, o, .. } = self.pal;
        let short = head(first_bad, 10).to_string();
        let (_, parent) = git(&self.repo, &["rev-parse", &format!("{first_bad}^")]);
        let diff = git_command(&self.repo)
            .args(["diff", &parent, first_bad])
            .output()
            .context("cannot run git diff")?
            .stdout;
        let units = split_hunks(&diff);
        let total = units.len();
        let pick =
            |indexes: &[usize]| -> Vec<&Unit> { indexes.iter().map(|&i| &units[i]).collect() };

        let budget = Cell::new(max_line_runs);
        let mut cache: HashMap<Vec<usize>, bool> = HashMap::new();
        let mut failure: Option<anyhow::Error> = None;
        let all: Vec<usize> = (0..total).collect();
        let minimal = if total >= 2 {
            ddmin(all.clone(), &budget, |subset| {
                let mut key = subset.to_vec();
                key.sort_unstable();
                if let Some(&known) = cache.get(&key) {
                    return known;
                }
                let ordered = pick(&key);
                self.reset_to(&parent);
                if !self.apply(&patch_of(&ordered), false) {
                    cache.insert(key, false);
                    return false;
                }
                budget.set(budget.get() - 1);
                let label = format!(
                    "{} + {} of {total} changes",
                    head(&parent, 10),
                    ordered.len()
                );
                match self.probe_state(&label, &names_of(&ordered), "lines") {
                    Ok(entry) => {
                        self.show(
                            &format!("parent + {:2} of {total} changes", ordered.len()),
                            &entry,
                        );
                        let fails = text_of(&entry, "verdict") == "bad";
                        cache.insert(key, fails);
                        fails
                    }
                    Err(err) => {
                        // Nothing more can be tested; end the search.
                        failure = Some(err);
                        budget.set(0);
                        false
                    }
                }
            })
        } else {
            all.clone()
        };
        if let Some(err) = failure {
            return Err(err);
        }
        let mut why = Why {
            revert: None,
            undo_how: None,
            without: None,
            units: total,
            minimal: minimal.iter().map(|&i| units[i].clone()).collect(),
            complete: budget.get() > 0,
            undo_patch: None,
        };
        let found = pick(&minimal);

        let rest: Vec<usize> = all
            .iter()
            .copied()
            .filter(|i| !minimal.contains(i))
            .collect();
        if !minimal.is_empty() && !rest.is_empty() {
            self.reset_to(&parent);
            if self.apply(&patch_of(&pick(&rest)), false) {
                let entry = self.probe_state(
                    &format!("{} + the other {} changes", head(&parent, 10), rest.len()),
                    &format!("the commit without: {}", names_of(&found)),
                    "without",
                )?;
                why.without = Some(text_of(&entry, "verdict").to_string());
                self.show(
                    &format!(
                        "parent + the other {} changes, without the {} found",
                        rest.len(),
                        minimal.len()
                    ),
                    &entry,
                );
            } else {
                why.without = Some("does not apply".to_string());
            }
        }

        self.reset_to(bad);
        if git_ok(
            &self.worktree,
            &["revert", "--no-commit", "--no-edit", first_bad],
        ) {
            why.undo_patch = Some(self.worktree_diff());
            let entry = self.probe_state(
                &format!("{} with {short} undone", head(bad, 10)),
                &format!("undo of the whole first bad commit on {bad_name}"),
                "undo",
            )?;
            why.revert = Some(text_of(&entry, "verdict").to_string());
            why.undo_how = Some("whole commit");
            self.show(&format!("{bad_name} with the whole commit undone"), &entry);
        } else {
            self.reset_to(bad);
            if !minimal.is_empty() && self.apply(&patch_of(&found), true) {
                why.undo_patch = Some(self.worktree_diff());
                let entry = self.probe_state(
                    &format!(
                        "{} with {} found change(s) of {short} undone",
                        head(bad, 10),
                        minimal.len()
                    ),
                    &format!("undo of: {}", names_of(&found)),
                    "undo",
                )?;
                why.revert = Some(text_of(&entry, "verdict").to_string());
                why.undo_how = Some("found lines only");
                println!(
                    "  {d}the whole commit no longer undoes cleanly on {bad_name} (later commits touched other parts of it), so only the lines found are undone{o}"
                );
                self.show(&format!("{bad_name} with only those lines undone"), &entry);
            } else {
                why.revert = Some("rewritten".to_string());
                println!(
                    "  {d}later commits rewrote these exact lines, so they cannot be undone mechanically on {bad_name}{o}"
                );
            }
        }
        self.reset_to(bad);
        Ok(why)
    }
}

/// The files the parent shares with the processes `git bisect run` starts.
struct ChildFiles {
    cfg: PathBuf,
    pending: PathBuf,
    errors: PathBuf,
    counter: PathBuf,
}

/// The commit a `git bisect` output line names as first bad.
fn first_bad_named(line: &str) -> Option<String> {
    let sha = line.strip_suffix(" is the first bad commit")?;
    let hex = sha
        .chars()
        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c));
    (hex && matches!(sha.len(), 40 | 64)).then(|| sha.to_string())
}

fn expand_user(path: &Path) -> PathBuf {
    if let Ok(rest) = path.strip_prefix("~")
        && let Some(dirs) = directories::BaseDirs::new()
    {
        return dirs.home_dir().join(rest);
    }
    path.to_path_buf()
}

/// `os.path.abspath`: absolute and normalized, symlinks left alone.
fn abspath(path: &Path) -> anyhow::Result<PathBuf> {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("cannot read the current directory")?
            .join(path)
    };
    let mut out = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    Ok(out)
}

fn utf8(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .with_context(|| format!("path is not UTF-8: {}", path.display()))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// What steps 3 to 6 established, all of it in the worktree.
struct Tested {
    boundary: Option<Lead>,
    runs_before_bisect: usize,
    first_bad: Option<String>,
    bisect_output: String,
    plain_runs: Option<u64>,
    why: Option<Why>,
}

/// Run `vestige prove`. Returns the process exit code: 0 when the report was
/// written, 1 when the run was refused or could not decide.
pub fn prove(storage: &Arc<Storage>, data_dir: &Path, args: &ProveArgs) -> anyhow::Result<i32> {
    match prove_inner(storage, data_dir, args) {
        Ok(code) => Ok(code),
        Err(err) => match err.downcast::<Stop>() {
            Ok(stop) => {
                println!("{}", stop.message);
                Ok(stop.code)
            }
            Err(err) => Err(err),
        },
    }
}

fn prove_inner(storage: &Arc<Storage>, data_dir: &Path, args: &ProveArgs) -> anyhow::Result<i32> {
    let pal = Palette::detect();
    let Palette {
        b,
        d,
        c,
        g,
        r,
        y,
        o,
    } = pal;
    let (Some(failure), Some(repo), Some(good_ref), Some(bad_ref), Some(reported_at), Some(report)) = (
        args.logged_write.as_deref(),
        args.repo.as_deref(),
        args.good.as_deref(),
        args.bad.as_deref(),
        args.reported_at.as_deref(),
        args.report.as_deref(),
    ) else {
        anyhow::bail!(
            "prove needs --logged-write, --repo, --good, --bad, --reported-at and --report, or --check <report>"
        );
    };
    let repo = abspath(&expand_user(repo))?;
    let store = abspath(data_dir)?;
    let out = abspath(&expand_user(report))?;
    if out.exists() {
        return Err(stop(1, format!("refusing to overwrite {}", out.display())));
    }
    if !out.parent().is_some_and(Path::is_dir) {
        return Err(stop(
            1,
            format!(
                "the directory for --report does not exist: {}",
                out.display()
            ),
        ));
    }
    let Ok(reported) = DateTime::parse_from_rfc3339(reported_at) else {
        return Err(stop(
            1,
            format!("--reported-at wants RFC 3339, like 2026-04-06T19:21:50Z, not {reported_at}"),
        ));
    };

    let tmp = tempfile::Builder::new()
        .prefix("walk-verify-")
        .tempdir()
        .context("cannot create a temporary directory")?
        .keep();
    let scratch = ScratchGuard::arm(repo.clone(), tmp.clone());

    let oracle = if let Some(command) = &args.test {
        let script = tmp.join("test-command.sh");
        fs::write(&script, format!("#!/bin/sh\n{command}\n"))
            .context("cannot write the test script")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&script, fs::Permissions::from_mode(0o755))
                .context("cannot make the test script executable")?;
        }
        script
    } else if let Some(script) = &args.oracle {
        abspath(&expand_user(script))?
    } else {
        return Err(stop(1, "give --test 'one command' or --oracle script"));
    };
    let oracle_sha256 = sha256_hex(
        &fs::read(&oracle)
            .with_context(|| format!("cannot read the test script {}", oracle.display()))?,
    );
    let worktree = tmp.join("checkout");
    let child = ChildFiles {
        cfg: tmp.join("cfg.json"),
        pending: tmp.join("handed-back.jsonl"),
        errors: tmp.join("bisect.stderr"),
        counter: tmp.join("sim.count"),
    };
    let session_path = tmp.join("probes.jsonl");
    let mut cfg = json!({
        "repo": utf8(&repo)?,
        "store": utf8(&store)?,
        "oracle": utf8(&oracle)?,
        "oracle_sha256": oracle_sha256,
        "worktree": utf8(&worktree)?,
        "session": utf8(&session_path)?,
        "pending": utf8(&child.pending)?,
        "slug": args.slug,
    });
    fs::write(&child.cfg, cfg.to_string()).context("cannot write the run configuration")?;

    let (good_ok, good) = git(&repo, &["rev-parse", &format!("{good_ref}^{{commit}}")]);
    let (bad_ok, bad) = git(&repo, &["rev-parse", &format!("{bad_ref}^{{commit}}")]);
    if !good_ok || !bad_ok {
        return Err(stop(1, "cannot resolve --good/--bad"));
    }
    let range = format!("{good}..{bad}");
    let (_, n_window) = git(&repo, &["rev-list", "--count", &range]);

    println!(
        "\n{b}Step 1. The walk proposes{o}  {d}$ vestige causal-walk --logged-write {failure}{o}"
    );
    let walk = tokio::runtime::Runtime::new()
        .context("cannot start the runtime for the walk")?
        .block_on(crate::tools::causal_walk::execute(
            storage,
            Some(json!({
                "scope": vestige_core::DEFAULT_MEMORY_SCOPE,
                "start_points": [{"kind": "logged_write", "node_id": failure}],
            })),
        ));
    let mut leads = match &walk {
        Ok(walk) => {
            if let Some(refusal) = walk["needs_report"].as_object() {
                let detail = refusal
                    .get("detail")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| canonical_json(&Value::Object(refusal.clone())));
                println!("  {y}The walk refused to start from {failure}: {detail}{o}");
            }
            leads_of(walk)
        }
        Err(err) => {
            println!("  {y}The walk could not run from {failure}: {err}{o}");
            Vec::new()
        }
    };
    println!(
        "  {n_window} commits between {good_ref} and {bad_ref}. The walk reaches {} of them over recorded links.",
        leads.len()
    );

    let mut kept: Vec<Lead> = Vec::new();
    let mut dropped: Vec<(Lead, String)> = Vec::new();
    for lead in &mut leads {
        let (found, full) = git(&repo, &["rev-parse", &format!("{}^{{commit}}", lead.short)]);
        if !found {
            dropped.push((lead.clone(), "not a commit in this repo".to_string()));
            continue;
        }
        lead.commit = Some(full.clone());
        let (_, when) = git(&repo, &["show", "-s", "--format=%cI", &full]);
        if DateTime::parse_from_rfc3339(&when).is_ok_and(|committed| committed > reported) {
            dropped.push((
                lead.clone(),
                format!("committed {}, after the report", head(&when, 10)),
            ));
            continue;
        }
        let in_bad = git_ok(&repo, &["merge-base", "--is-ancestor", &full, &bad]);
        let in_good = git_ok(&repo, &["merge-base", "--is-ancestor", &full, &good]);
        if !in_bad || in_good {
            dropped.push((lead.clone(), format!("outside {good_ref}..{bad_ref}")));
            continue;
        }
        kept.push(lead.clone());
    }
    println!("\n{b}Step 2. Time gate{o}  a cause cannot come after its effect");
    for (lead, reason) in &dropped {
        println!("  {d}dropped {}: {reason}{o}", lead.short);
    }
    println!(
        "  {} candidate(s) kept, {} dropped.",
        kept.len(),
        dropped.len()
    );
    for lead in kept.iter().take(args.show) {
        let commit = lead.commit.as_deref().unwrap_or(&lead.short);
        let (_, subject) = git(&repo, &["show", "-s", "--format=%s", commit]);
        println!(
            "  {d}[recorded link]{o} #{} depth {}  {c}{} {}{o}",
            lead.rank,
            lead.depth,
            lead.short,
            head(&subject, 86)
        );
    }
    if kept.len() > args.show {
        println!("  {d}... and {} more{o}", kept.len() - args.show);
    }

    // Frozen before the first test: what is tested, with what, on which leads.
    let kept_commits: Vec<&str> = kept
        .iter()
        .filter_map(|lead| lead.commit.as_deref())
        .collect();
    let mut protocol = json!({
        "slug": args.slug,
        "failure_memory": failure,
        "good": good,
        "bad": bad,
        "oracle_sha256": oracle_sha256,
        "rule": PROTOCOL_RULE,
        "candidates": kept_commits,
        "max_candidates": args.max_candidates,
        "max_line_runs": args.max_line_runs,
        "flaky": null,
        "frozen_at": now_stamp(),
    });
    let protocol_sha256 = protocol.as_object().map(protocol_hash).unwrap_or_default();
    let protocol_text = format!(
        "Protocol ({}), frozen before any test run: failure {failure}, good {}, bad {}, repro script sha256 {}, rule: {PROTOCOL_RULE}. {} candidates in walk order: {}. Protocol sha256 {protocol_sha256}. Nothing in this protocol may change after the first test.",
        args.slug,
        head(&good, 10),
        head(&bad, 10),
        head(&oracle_sha256, 16),
        kept.len(),
        kept_commits
            .iter()
            .map(|commit| head(commit, 10))
            .collect::<Vec<_>>()
            .join(" "),
    );
    let protocol_memory = remember(
        storage.as_ref(),
        &protocol_text,
        &format!("oracle-protocol,{}", args.slug),
    );
    println!(
        "\n{b}Protocol frozen before any test{o}  the script, the two ends and the {} candidates are on record  {d}-> {}  sha256 {}{o}",
        kept.len(),
        opt_display(protocol_memory.as_deref()),
        head(&protocol_sha256, 16)
    );
    protocol["sha256"] = json!(protocol_sha256);
    protocol["memory"] = json!(protocol_memory);

    // Oldest to newest. Step 4 lines the leads up in this order, under git
    // bisect's own assumption: one first bad commit, everything after it bad.
    let (_, order) = git(&repo, &["rev-list", "--topo-order", "--reverse", &range]);
    let position: HashMap<&str, usize> = order
        .split_whitespace()
        .enumerate()
        .map(|(index, sha)| (sha, index))
        .collect();

    let added = git_command(&repo)
        .args(["worktree", "add", "-q", "--detach", utf8(&worktree)?, &bad])
        .output()
        .context("cannot run git")?;
    if !added.status.success() {
        return Err(stop(
            1,
            format!(
                "worktree failed: {}",
                strip(&String::from_utf8_lossy(&added.stderr))
            ),
        ));
    }
    scratch.worktree_added(&worktree);

    let mut prover = Prover {
        storage: storage.as_ref(),
        pal,
        repo: repo.clone(),
        worktree: worktree.clone(),
        oracle: oracle.clone(),
        oracle_sha256: oracle_sha256.clone(),
        slug: args.slug.clone(),
        session: Session::new(session_path),
    };

    // Steps 3 to 6 need the worktree; it is removed before the result is
    // written, whatever they return.
    let mut steps = || -> anyhow::Result<Option<Tested>> {
        println!(
            "\n{b}Step 3. The repro script must tell the two ends apart{o}  {d}{}{o}",
            file_name(&oracle)
        );
        let on_good = prover.probe(&good, "baseline")?;
        let on_bad = prover.probe(&bad, "baseline")?;
        if text_of(&on_good, "verdict") != "good" || text_of(&on_bad, "verdict") != "bad" {
            println!(
                "{r}The script does not pass on {good_ref} and fail on {bad_ref}, so nothing can be decided with it.{o}"
            );
            return Ok(None);
        }

        println!("\n{b}Step 4. Bisect over the candidates only, then test the parent{o}");
        let mut line_up: Vec<(usize, Lead)> = kept
            .iter()
            .filter_map(|lead| {
                let at = position.get(lead.commit.as_deref()?)?;
                Some((*at, lead.clone()))
            })
            .collect();
        line_up.sort_by_key(|(at, _)| *at);
        if args.max_candidates > 0 && line_up.len() > args.max_candidates {
            line_up.drain(..line_up.len() - args.max_candidates);
        }
        let mut line_up: Vec<Lead> = line_up.into_iter().map(|(_, lead)| lead).collect();
        let mut low = 0isize;
        let mut high = line_up.len() as isize - 1;
        let mut earliest_bad: Option<Lead> = None;
        while low <= high {
            let middle = ((low + high) / 2) as usize;
            let lead = line_up[middle].clone();
            let commit = lead.commit.clone().unwrap_or_default();
            let entry = prover.probe(&commit, "candidate")?;
            match text_of(&entry, "verdict") {
                "bad" => {
                    earliest_bad = Some(lead);
                    high = middle as isize - 1;
                }
                "good" => low = middle as isize + 1,
                _ => {
                    line_up.remove(middle);
                    high -= 1;
                }
            }
        }
        let mut boundary = None;
        if let Some(lead) = earliest_bad {
            let commit = lead.commit.clone().unwrap_or_default();
            let (has_parent, parent) = git(&repo, &["rev-parse", &format!("{commit}^")]);
            if !has_parent {
                return Err(stop(2, format!("cannot find the parent of {}", lead.short)));
            }
            let on_parent = prover.probe(&parent, "parent")?;
            if text_of(&on_parent, "verdict") == "good" {
                println!("  {g}{} fails and its parent passes.{o}", lead.short);
                boundary = Some(lead);
            } else {
                println!(
                    "  {y}The earliest failing candidate's parent also fails, so the first bad commit is not among the candidates.{o}"
                );
            }
        } else {
            println!(
                "  {y}No candidate fails, so the first bad commit is not among the candidates.{o}"
            );
        }
        let runs_before_bisect = prover.session.entries.len();

        println!(
            "\n{b}Step 5. Confirm with stock git bisect over all {n_window} commits{o}  {d}$ git bisect run{o}"
        );
        let (first_bad, bisect_output) = prover.bisect(&child, &good, &bad)?;
        let mut plain_runs = None;
        let mut why = None;
        if let Some(first_bad) = &first_bad {
            cfg["sim"] = json!({"first_bad": first_bad, "counter": utf8(&child.counter)?});
            fs::write(&child.cfg, cfg.to_string()).context("cannot write the run configuration")?;
            plain_runs = prover.replay_plain_bisect(&child, &good, &bad)?;
            if !args.no_why {
                println!(
                    "\n{b}Step 6. Why: find the lines, test the commit without them, undo them on the broken version{o}"
                );
                why = Some(prover.explain(&bad, first_bad, args.max_line_runs, bad_ref)?);
            }
        }
        Ok(Some(Tested {
            boundary,
            runs_before_bisect,
            first_bad,
            bisect_output,
            plain_runs,
            why,
        }))
    };
    let tested = steps();
    remove_worktree();
    let Some(tested) = tested? else {
        return Ok(1);
    };
    let entries = prover.session.entries.clone();
    let Tested {
        boundary,
        runs_before_bisect,
        first_bad,
        bisect_output,
        plain_runs,
        why,
    } = tested;

    println!("\n{b}Result{o}");
    let mut subject = String::new();
    if let Some(first_bad) = &first_bad {
        subject = git(&repo, &["show", "-s", "--format=%s", first_bad]).1;
        let (_, when) = git(&repo, &["show", "-s", "--format=%cs", first_bad]);
        println!(
            "  {b}[tested]{o} git bisect: {g}{}{o} is the first bad commit",
            head(first_bad, 10)
        );
        println!("           {b}{}{o}  {d}({when}){o}", head(&subject, 100));
    } else {
        let tail: String = {
            let count = bisect_output.chars().count();
            bisect_output
                .chars()
                .skip(count.saturating_sub(600))
                .collect()
        };
        println!("  git bisect did not name a first bad commit. Its output:\n{tail}");
    }
    let agree = boundary
        .as_ref()
        .is_some_and(|lead| first_bad.is_some() && lead.commit == first_bad);
    let reached = first_bad.as_ref().and_then(|first_bad| {
        leads
            .iter()
            .find(|lead| lead.commit.as_ref() == Some(first_bad))
    });
    if let Some(lead) = boundary.as_ref().filter(|_| agree) {
        println!(
            "  The walk had this commit as lead #{} of {}. git bisect, run on its own, names the same commit.",
            lead.rank,
            leads.len()
        );
    } else if let Some(lead) = reached {
        println!(
            "  The walk reached this commit as lead #{} of {} (depth {}).",
            lead.rank,
            leads.len(),
            lead.depth
        );
    } else if first_bad.is_some() {
        println!(
            "  {y}The walk did not reach this commit. The tested result stands; the walk missed it.{o}"
        );
    }
    let found_in = runs_before_bisect as i64 - 2;
    if agree && let Some(plain) = plain_runs.filter(|plain| *plain > 0) {
        println!(
            "  Found in {b}{found_in} test runs{o} on the walk's {} leads. Plain git bisect needs {plain} on the same {n_window} commits {d}(replayed against the tested answer){o}.",
            kept.len()
        );
    }
    if let Some(why) = &why {
        let count = why.minimal.len();
        let that = if count == 1 {
            "that change"
        } else {
            "those changes"
        };
        if count > 0 && why.units > 1 {
            println!(
                "  {b}[tested]{o} Of its {} changes, {b}{count} {} enough to cause it{o}{}:",
                why.units,
                if count == 1 { "is" } else { "are" },
                if why.complete {
                    ""
                } else {
                    " (run budget reached, may shrink further)"
                }
            );
            for unit in &why.minimal {
                println!("    {c}{} line {}{o}", unit.file, unit.start);
                for line in unit.added.iter().take(8) {
                    println!("      {g}+ {}{o}", head(line, 110));
                }
            }
        }
        match why.without.as_deref() {
            Some("good") => {
                println!("  {b}[tested]{o} The rest of the commit, without {that}, passes.")
            }
            Some("bad") => println!(
                "  {b}[tested]{o} The rest of the commit still fails without {that}, so more than one part of it carries the bug."
            ),
            _ => {}
        }
        match (why.revert.as_deref(), why.undo_how) {
            (Some("good"), Some("whole commit")) => {
                println!("  {b}[tested]{o} Undo this one commit on {bad_ref} and the bug is gone.");
            }
            (Some("good"), _) => {
                println!("  {b}[tested]{o} Undo just {that} on {bad_ref} and the bug is gone.");
            }
            (Some("bad"), _) => println!(
                "  {b}[tested]{o} Undoing it on {bad_ref} does not fix it; something later also carries the bug."
            ),
            _ => {}
        }
    }
    println!(
        "  {d}{} runs recorded in total, including the two ends, the git bisect confirmation and the line search.{o}",
        entries.len()
    );

    // Five rungs, each with whether it holds and the runs that back it.
    let mut card: Vec<(&str, bool, String, String)> = Vec::new();
    if let Some(first_bad) = &first_bad {
        let runs = |backing: &[&Entry]| {
            if backing.is_empty() {
                return "no run".to_string();
            }
            let numbers: Vec<String> = backing
                .iter()
                .map(|entry| py_display(entry.get("n")))
                .collect();
            format!("runs {}", numbers.join(","))
        };
        let in_phase = |phase: &str| -> Vec<&Entry> {
            entries
                .iter()
                .filter(|entry| text_of(entry, "phase") == phase)
                .collect()
        };
        let first_on = |commit: &str| -> Vec<&Entry> {
            entries
                .iter()
                .filter(|entry| text_of(entry, "commit") == commit)
                .take(1)
                .collect()
        };
        let on_first_bad = first_on(first_bad);
        let (_, parent) = git(&repo, &["rev-parse", &format!("{first_bad}^")]);
        let on_parent = first_on(&parent);
        let newest_first = position.len() - position.get(first_bad.as_str()).copied().unwrap_or(0);
        card.push((
            "LEAD",
            reached.is_some(),
            match reached {
                Some(lead) => format!(
                    "the walk reached it over recorded links (lead {} of {})",
                    lead.rank,
                    leads.len()
                ),
                None => "the walk reached it over recorded links".to_string(),
            },
            match reached {
                Some(lead) => format!("recorded link {}", lead.memory),
                None => "the walk did not reach it".to_string(),
            },
        ));
        let boundary_holds = on_first_bad
            .first()
            .is_some_and(|entry| text_of(entry, "verdict") == "bad")
            && on_parent
                .first()
                .is_some_and(|entry| text_of(entry, "verdict") == "good");
        let both: Vec<&Entry> = on_first_bad.iter().chain(&on_parent).copied().collect();
        card.push((
            "BOUNDARY",
            boundary_holds,
            "it fails the test and the commit before it passes".to_string(),
            runs(&both),
        ));
        let bisected = in_phase("bisect");
        card.push((
            "CONFIRMED",
            true,
            format!("stock git bisect over all {n_window} commits names it"),
            if bisected.is_empty() {
                "reused runs".to_string()
            } else {
                runs(&bisected)
            },
        ));
        if let Some(why) = &why {
            let count = why.minimal.len();
            let searched: Vec<&Entry> = in_phase("lines")
                .into_iter()
                .chain(in_phase("without"))
                .collect();
            card.push((
                "ISOLATED",
                why.units > 1 && count < why.units && why.without.as_deref() == Some("good"),
                if why.units > 1 {
                    format!(
                        "{count} of its {} changes alone causes it, and the rest passes without {}",
                        why.units,
                        if count == 1 { "it" } else { "them" }
                    )
                } else {
                    "the commit is a single change".to_string()
                },
                runs(&searched),
            ));
            card.push((
                "REVERSED",
                why.revert.as_deref() == Some("good"),
                format!(
                    "undoing {} on {bad_ref} makes the test pass again",
                    if why.undo_how == Some("whole commit") {
                        "the whole commit"
                    } else {
                        "just those lines"
                    }
                ),
                runs(&in_phase("undo")),
            ));
        }
        let held = card.iter().filter(|rung| rung.1).count();
        println!(
            "\n{b}Verdict{o}  commit {g}{}{o}  {b}{held} of {} rungs hold{o}",
            head(first_bad, 10),
            card.len()
        );
        for (name, holds, statement, proof) in &card {
            println!(
                "  {}{name:<9}{o} {}  {statement}  {d}{proof}{o}",
                if *holds { g } else { y },
                if *holds { "yes" } else { "no " }
            );
        }
        println!(
            "  {d}In plain git log order this is commit {newest_first} of {n_window}. Protocol {} was frozen before the first test.{o}",
            opt_display(protocol_memory.as_deref())
        );
        println!(
            "  {d}Not claimed: why the authors made the change, whether other inputs fail too, or what the right fix is.{o}"
        );
    }

    let mut result_memory = None;
    if let Some(first_bad) = &first_bad {
        let walk_said = match reached {
            Some(lead) => format!("had reached this commit as lead #{}", lead.rank),
            None => "had not reached this commit".to_string(),
        };
        let lines_said = match &why {
            Some(why) => {
                let changes = why
                    .minimal
                    .iter()
                    .map(|unit| format!("{} line {}", unit.file, unit.start))
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(
                    "Undo on the bad ref ({}) tested {}; the commit without the minimal changes tested {}. Minimal failing changes: {}.",
                    opt_display(why.undo_how),
                    opt_display(why.revert.as_deref()).to_uppercase(),
                    opt_display(why.without.as_deref()).to_uppercase(),
                    if changes.is_empty() {
                        "not narrowed"
                    } else {
                        &changes
                    }
                )
            }
            None => "Lines not searched.".to_string(),
        };
        let text = format!(
            "Tested result ({}): Commit {} ({}) is the first bad commit between {good_ref} and {bad_ref} according to git bisect run with repro script sha256 {}. {} probes, each recorded as its own memory tagged oracle-probe. The causal walk from {failure} {walk_said}. This says which commit first makes the repro script fail. {lines_said}",
            args.slug,
            head(first_bad, 10),
            head(&subject, 100),
            head(&oracle_sha256, 16),
            entries.len(),
        );
        result_memory = remember(
            storage.as_ref(),
            &text,
            &format!("oracle-result,{}", args.slug),
        );
    }

    let mut why_report = Value::Null;
    let mut undo_patch: Option<(PathBuf, &[u8])> = None;
    if let Some(why) = &why {
        why_report = json!({
            "undo_on_bad": why.revert,
            "undo_how": why.undo_how,
            "commit_without_minimal": why.without,
            "changes_in_commit": why.units,
            "search_complete": why.complete,
            "minimal_failing_changes": why.minimal.iter().map(|unit| json!({
                "file": unit.file,
                "line": unit.start,
                "added": unit.added,
            })).collect::<Vec<_>>(),
        });
        if let Some(patch) = why.undo_patch.as_deref().filter(|patch| !patch.is_empty())
            && why.revert.as_deref() == Some("good")
        {
            let out_text = utf8(&out)?;
            let path = PathBuf::from(match out_text.strip_suffix(".json") {
                Some(stem) => format!("{stem}.undo.patch"),
                None => format!("{out_text}.undo.patch"),
            });
            why_report["undo_patch"] = json!({
                "file": file_name(&path),
                "applies_to": bad_ref,
                "sha256": sha256_hex(patch),
            });
            undo_patch = Some((path, patch));
        }
    }
    let report = json!({
        "tool": TOOL,
        "slug": args.slug,
        "repo": utf8(&repo)?,
        "good": {"ref": good_ref, "commit": good},
        "bad": {"ref": bad_ref, "commit": bad},
        "window_commits": n_window.parse::<u64>().unwrap_or(0),
        "failure_memory": failure,
        "reported_at": reported_at,
        "store": utf8(&store)?,
        "oracle": {"file": file_name(&oracle), "sha256": oracle_sha256},
        "walk": {
            "reached": leads.len(),
            "kept": kept.len(),
            "dropped": dropped.iter().map(|(lead, reason)| json!({"commit": lead.short, "why": reason})).collect::<Vec<_>>(),
            "candidates": kept.iter().map(|lead| json!({
                "rank": lead.rank,
                "depth": lead.depth,
                "commit": lead.commit,
                "memory": lead.memory,
            })).collect::<Vec<_>>(),
        },
        "protocol": protocol,
        "flaky": null,
        "verdict_card": card.iter().map(|(rung, holds, statement, proof)| json!({
            "rung": rung,
            "holds": holds,
            "statement": statement,
            "proof": proof,
        })).collect::<Vec<_>>(),
        "probes": entries,
        "first_bad_commit": first_bad,
        "walk_lead_rank": reached.map(|lead| lead.rank),
        "plain_bisect_runs_replayed": plain_runs,
        "found_in_runs": found_in,
        "why": why_report,
        "result_memory": result_memory,
        "chain_head": entries.last().map(|entry| text_of(entry, "hash")),
        "reading": READING,
    });
    if let Some((path, patch)) = &undo_patch {
        fs::write(path, patch).with_context(|| format!("cannot write {}", path.display()))?;
    }
    fs::write(&out, pretty_json(&report))
        .with_context(|| format!("cannot write {}", out.display()))?;
    if let Some((path, _)) = &undo_patch {
        println!(
            "  The undo that was tested, as a patch on {bad_ref}: {b}{}{o}",
            path.display()
        );
    }
    println!(
        "\n  Every probe is a memory in the store{}. Report: {b}{}{o}",
        result_memory
            .as_ref()
            .map(|id| format!(", result {id}"))
            .unwrap_or_default(),
        out.display()
    );
    println!(
        "  {d}Anyone can re-check the report offline: vestige prove --check {}{o}\n",
        file_name(&out)
    );
    drop(scratch);
    Ok(0)
}

// ---------------------------------------------------------------------------
// The processes `git bisect run` starts
// ---------------------------------------------------------------------------

/// Entry point of the hidden subcommand. `probe` tests the commit git has
/// checked out (or reuses its recorded verdict) and exits with the test's
/// code; `sim` answers from the known first bad commit and counts the call.
/// Neither opens the store. An exit of 255 makes `git bisect run` stop.
pub fn child(mode: &str, cfg: &Path) -> i32 {
    let outcome = fs::read_to_string(cfg)
        .context("cannot read the run configuration")
        .and_then(|text| serde_json::from_str::<Value>(&text).context("bad run configuration"))
        .and_then(|cfg| match mode {
            "probe" => child_probe(&cfg),
            "sim" => child_sim(&cfg),
            other => anyhow::bail!("unknown mode {other}"),
        });
    match outcome {
        Ok(code) => code,
        Err(err) => {
            eprintln!("vestige {CHILD_COMMAND} {mode}: {err:#}");
            255
        }
    }
}

fn cfg_path(cfg: &Value, key: &str) -> anyhow::Result<PathBuf> {
    cfg[key]
        .as_str()
        .map(PathBuf::from)
        .with_context(|| format!("the run configuration has no {key}"))
}

fn child_probe(cfg: &Value) -> anyhow::Result<i32> {
    let worktree = cfg_path(cfg, "worktree")?;
    let pending = cfg_path(cfg, "pending")?;
    let (found, sha) = git(&worktree, &["rev-parse", "HEAD"]);
    anyhow::ensure!(found, "cannot read HEAD of {}", worktree.display());
    let exit_of = |entry: &Entry| entry.get("exit").and_then(Value::as_i64).unwrap_or(255) as i32;

    let recorded = Session::read(&cfg_path(cfg, "session")?);
    if let Some(hit) = recorded
        .iter()
        .find(|entry| text_of(entry, "commit") == sha)
    {
        append_line(&pending, &json!({"reused": hit.get("n"), "commit": sha}))?;
        return Ok(exit_of(hit));
    }
    // A run handed back earlier in this bisect and not chained yet.
    let handed_back = fs::read_to_string(&pending).unwrap_or_default();
    for line in handed_back.lines() {
        if let Ok(record) = serde_json::from_str::<Value>(line)
            && let Some(entry) = record.get("entry").and_then(Value::as_object)
            && text_of(entry, "commit") == sha
        {
            return Ok(exit_of(entry));
        }
    }

    let run = run_test(&cfg_path(cfg, "oracle")?, &worktree)?;
    let (_, subject) = git(
        &cfg_path(cfg, "repo")?,
        &["show", "-s", "--format=%s", &sha],
    );
    let oracle_sha256 = cfg["oracle_sha256"].as_str().unwrap_or_default();
    let entry = new_entry(&sha, &subject, &run, oracle_sha256, "bisect");
    append_line(&pending, &json!({"entry": entry}))?;
    Ok(run.exit as i32)
}

fn child_sim(cfg: &Value) -> anyhow::Result<i32> {
    let first_bad = cfg["sim"]["first_bad"]
        .as_str()
        .context("the run configuration has no sim.first_bad")?;
    let counter = cfg["sim"]["counter"]
        .as_str()
        .context("the run configuration has no sim.counter")?;
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(counter)?
        .write_all(b"x")?;
    let worktree = cfg_path(cfg, "worktree")?;
    let bad = git_ok(
        &worktree,
        &["merge-base", "--is-ancestor", first_bad, "HEAD"],
    );
    Ok(i32::from(bad))
}

// ---------------------------------------------------------------------------
// --check
// ---------------------------------------------------------------------------

/// Re-verify a report offline. Returns the process exit code: 0 when the
/// hash chain is intact, the first bad commit's recorded verdict is bad,
/// each of its parents is recorded good (or is an ancestor of a commit that
/// is), the frozen protocol still has the hash it was given and predates
/// the first test, and the undo patch beside the report is the one hashed;
/// 1 otherwise. The parents are looked up in the report's repo when it is
/// on this machine.
pub fn check(report: &Path) -> anyhow::Result<i32> {
    let Palette { r, o, .. } = Palette::detect();
    let text =
        fs::read_to_string(report).with_context(|| format!("cannot read {}", report.display()))?;
    let rep: Value =
        serde_json::from_str(&text).with_context(|| format!("{} is not JSON", report.display()))?;
    let probes = rep["probes"]
        .as_array()
        .with_context(|| format!("{} has no probes", report.display()))?;

    let mut prev = ZERO_HASH.to_string();
    for probe in probes {
        let entry = probe.as_object();
        let want = entry.map(|entry| chain_hash(&prev, entry));
        let linked = entry
            .and_then(|entry| entry.get("prev"))
            .and_then(Value::as_str)
            == Some(prev.as_str());
        let hashed = entry
            .and_then(|entry| entry.get("hash"))
            .and_then(Value::as_str)
            == want.as_deref();
        let Some(want) = want.filter(|_| linked && hashed) else {
            println!(
                "{r}probe {}: hash does not match. The report was changed after it was written.{o}",
                py_display(probe.get("n"))
            );
            return Ok(1);
        };
        prev = want;
    }
    if rep["chain_head"].as_str() != Some(prev.as_str()) {
        println!("{r}chain head does not match the last probe.{o}");
        return Ok(1);
    }
    let mut verdicts: HashMap<&str, &str> = HashMap::new();
    for probe in probes {
        if let (Some(commit), Some(verdict)) = (probe["commit"].as_str(), probe["verdict"].as_str())
        {
            verdicts.insert(commit, verdict);
        }
    }
    println!(
        "{} probes, hash chain intact, head {}",
        probes.len(),
        head(&prev, 16)
    );

    let mut sound = true;
    if let Some(first_bad) = rep["first_bad_commit"].as_str() {
        let verdict = verdicts.get(first_bad).copied().unwrap_or("MISSING");
        println!(
            "first bad commit {}: recorded verdict {verdict}",
            head(first_bad, 10)
        );
        if verdict != "bad" {
            println!("{r}the first bad commit is not recorded as bad by a probe.{o}");
            sound = false;
        }
        let repo = rep["repo"]
            .as_str()
            .map(Path::new)
            .filter(|repo| repo.is_dir());
        let parents = repo.and_then(|repo| {
            let (found, parents) = git(repo, &["show", "-s", "--format=%P", first_bad]);
            found.then_some((repo, parents))
        });
        match parents {
            Some((repo, parents)) => {
                for parent in parents.split_whitespace() {
                    let verdict = verdicts.get(parent).copied();
                    println!(
                        "  its parent {}: recorded verdict {}",
                        head(parent, 10),
                        verdict.unwrap_or("not probed")
                    );
                    match verdict {
                        Some("good") => {}
                        Some(_) => {
                            println!(
                                "{r}a parent of the first bad commit is not recorded as good.{o}"
                            );
                            sound = false;
                        }
                        None => {
                            // git bisect never tests a commit a good one
                            // already rules out: its ancestors.
                            let implied = verdicts.iter().find(|(commit, verdict)| {
                                **verdict == "good"
                                    && git_ok(
                                        repo,
                                        &["merge-base", "--is-ancestor", parent, commit],
                                    )
                            });
                            match implied {
                                Some((commit, _)) => println!(
                                    "    it is an ancestor of {}, which is recorded good",
                                    head(commit, 10)
                                ),
                                None => {
                                    println!(
                                        "{r}a parent of the first bad commit has no good verdict behind it.{o}"
                                    );
                                    sound = false;
                                }
                            }
                        }
                    }
                }
            }
            None => println!(
                "  its parent: not looked up, the repo {} with this commit is not on this machine",
                py_display(rep.get("repo"))
            ),
        }
    } else {
        println!("this report names no first bad commit");
    }
    if let Some(protocol) = rep["protocol"].as_object() {
        let matches = protocol.get("sha256").and_then(Value::as_str)
            == Some(protocol_hash(protocol).as_str());
        let frozen_at = protocol
            .get("frozen_at")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let first_at = probes
            .first()
            .map(|probe| probe["at"].as_str().unwrap_or_default())
            .unwrap_or_default();
        println!(
            "protocol {}: {}, frozen {frozen_at}, first test {first_at}",
            py_display(protocol.get("memory")),
            if matches {
                "hash matches"
            } else {
                "HASH DOES NOT MATCH"
            }
        );
        // The protocol must be the one hashed, and on record before any test.
        if !matches || (!first_at.is_empty() && first_at < frozen_at) {
            return Ok(1);
        }
    }
    for rung in rep["verdict_card"].as_array().into_iter().flatten() {
        println!(
            "  {:<9} {}  {}",
            py_display(rung.get("rung")),
            if rung["holds"] == true { "yes" } else { "no " },
            py_display(rung.get("statement"))
        );
    }
    if let Some(why) = rep["why"].as_object() {
        for change in why
            .get("minimal_failing_changes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            println!(
                "minimal failing change: {} line {}",
                py_display(change.get("file")),
                py_display(change.get("line"))
            );
        }
        println!(
            "the commit without it: recorded verdict {}",
            py_display(why.get("commit_without_minimal"))
        );
        println!(
            "undo on {} ({}): recorded verdict {}",
            py_display(rep["bad"].get("ref")),
            py_display(why.get("undo_how")),
            py_display(why.get("undo_on_bad"))
        );
        if let Some(patch) = why.get("undo_patch").and_then(Value::as_object) {
            let name = patch
                .get("file")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let beside = abspath(report)?
                .parent()
                .map(|dir| dir.join(name))
                .unwrap_or_else(|| PathBuf::from(name));
            if let Ok(bytes) = fs::read(&beside) {
                let same = Some(sha256_hex(&bytes).as_str())
                    == patch.get("sha256").and_then(Value::as_str);
                println!(
                    "undo patch {name}: {}",
                    if same {
                        "sha256 matches the report"
                    } else {
                        "DOES NOT MATCH the report"
                    }
                );
                if !same {
                    return Ok(1);
                }
            }
        }
    }
    println!(
        "script sha256 {} ({})",
        head(rep["oracle"]["sha256"].as_str().unwrap_or_default(), 16),
        py_display(rep["oracle"].get("file"))
    );
    Ok(if sound { 0 } else { 1 })
}

/// `vestige prove`: `--check` needs no store, everything else does.
/// `open_store` is the CLI's own way of opening the store it chose.
pub fn run<F>(args: &ProveArgs, open_store: F) -> anyhow::Result<i32>
where
    F: FnOnce() -> anyhow::Result<(Arc<Storage>, PathBuf)>,
{
    if let Some(report) = &args.check {
        return check(&expand_user(report));
    }
    let (storage, data_dir) = open_store()?;
    prove(&storage, &data_dir, args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(value: Value) -> Entry {
        value.as_object().cloned().expect("an object")
    }

    // Written by Python 3 with
    // json.dumps(entry, sort_keys=True, separators=(",", ":")) and
    // hashlib.sha256((prev + body).encode()).hexdigest().
    const PY_BODY_1: &str = r#"{"at":"2026-10-05T23:45:31+00:00","commit":"4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270","exit":1,"memory":null,"n":1,"oracle_said":"gave up after 6.88s (ConnectionError) \u007f\u0001 \u2028 end/","oracle_sha256":"725c8d23f3905746a09db5450859e50bde9a7569ff40e328f380ff1e917a533b","phase":"candidate","prev":"0000000000000000000000000000000000000000000000000000000000000000","subject":"Caf\u00e9 \u2014 \"quoted\" back\\slash \ud83d\ude00 tab\there","verdict":"bad"}"#;
    const PY_HASH_1: &str = "308763e1c81affd3566e016dd013789219fe38c904984b32ac59739d892bd285";
    const PY_BODY_2: &str = r#"{"at":"2026-10-05T23:45:56+00:00","commit":"742b13bdce + 2 of 4 changes","exit":-9,"memory":"mem-0000000000000729","n":2,"oracle_said":"","oracle_sha256":"725c8d23f3905746a09db5450859e50bde9a7569ff40e328f380ff1e917a533b","phase":"lines","prev":"308763e1c81affd3566e016dd013789219fe38c904984b32ac59739d892bd285","subject":"redis/asyncio/connection.py:296, redis/connection.py:843","verdict":"skip"}"#;
    const PY_HASH_2: &str = "d2ea4ac4658c38b0f5ef1f9ce06e102aa976c194f1f53deb9ca51db3853a01c3";

    fn first_entry() -> Entry {
        entry(json!({
            "commit": "4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270",
            "subject": "Caf\u{e9} \u{2014} \"quoted\" back\\slash \u{1F600} tab\there",
            "verdict": "bad",
            "exit": 1,
            "oracle_said": "gave up after 6.88s (ConnectionError) \u{7f}\u{1} \u{2028} end/",
            "oracle_sha256": "725c8d23f3905746a09db5450859e50bde9a7569ff40e328f380ff1e917a533b",
            "phase": "candidate",
            "at": "2026-10-05T23:45:31+00:00",
            "memory": null,
            "n": 1,
            "prev": ZERO_HASH,
        }))
    }

    fn second_entry() -> Entry {
        entry(json!({
            "commit": "742b13bdce + 2 of 4 changes",
            "subject": "redis/asyncio/connection.py:296, redis/connection.py:843",
            "verdict": "skip",
            "exit": -9,
            "oracle_said": "",
            "oracle_sha256": "725c8d23f3905746a09db5450859e50bde9a7569ff40e328f380ff1e917a533b",
            "phase": "lines",
            "at": "2026-10-05T23:45:56+00:00",
            "memory": "mem-0000000000000729",
            "n": 2,
            "prev": PY_HASH_1,
        }))
    }

    #[test]
    fn canonical_json_is_pythons_sorted_compact_ascii_form() {
        assert_eq!(canonical_json(&Value::Object(first_entry())), PY_BODY_1);
        assert_eq!(canonical_json(&Value::Object(second_entry())), PY_BODY_2);
        let mixed = json!({
            "z": [1, 2.5, 1e16, 1e-05, 0.1, 100.0, true, false, null, {"b": 1, "a": "\u{8}\u{c}\n\r"}],
            "a": {},
            "m": [],
            "big": 12345678901234567890u64,
            "neg": -0.0,
        });
        assert_eq!(
            canonical_json(&mixed),
            r#"{"a":{},"big":12345678901234567890,"m":[],"neg":-0.0,"z":[1,2.5,1e+16,1e-05,0.1,100.0,true,false,null,{"a":"\b\f\n\r","b":1}]}"#
        );
    }

    #[test]
    fn chain_hash_matches_the_python_vector_and_ignores_the_hash_field() {
        assert_eq!(chain_hash(ZERO_HASH, &first_entry()), PY_HASH_1);
        assert_eq!(chain_hash(PY_HASH_1, &second_entry()), PY_HASH_2);
        let mut hashed = first_entry();
        hashed.insert("hash".to_string(), json!(PY_HASH_1));
        assert_eq!(chain_hash(ZERO_HASH, &hashed), PY_HASH_1);
        let mut changed = first_entry();
        changed.insert("verdict".to_string(), json!("good"));
        assert_ne!(chain_hash(ZERO_HASH, &changed), PY_HASH_1);
    }

    #[test]
    fn floats_print_as_python_repr() {
        for (value, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (123456.789, "123456.789"),
            (1.5e300, "1.5e+300"),
            (-2.5, "-2.5"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (0.30000000000000004, "0.30000000000000004"),
        ] {
            assert_eq!(float_repr(value), want);
        }
    }

    #[test]
    fn the_report_layout_is_pythons_indent_one() {
        let value = json!({"b": [1, {"x": [], "a": {}}], "a": "\u{e9}", "c": {"k": null}});
        assert_eq!(
            pretty_json(&value),
            "{\n \"a\": \"\\u00e9\",\n \"b\": [\n  1,\n  {\n   \"a\": {},\n   \"x\": []\n  }\n ],\n \"c\": {\n  \"k\": null\n }\n}"
        );
    }

    #[test]
    fn the_session_numbers_links_and_hashes_each_entry() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::new(dir.path().join("probes.jsonl"));
        let mut one = first_entry();
        one.remove("n");
        one.remove("prev");
        let one = session.append(one).unwrap();
        assert_eq!(one["n"], 1);
        assert_eq!(one["prev"], ZERO_HASH);
        assert_eq!(one["hash"], PY_HASH_1);
        let mut two = second_entry();
        two.remove("n");
        two.remove("prev");
        let two = session.append(two).unwrap();
        assert_eq!(two["n"], 2);
        assert_eq!(two["prev"], PY_HASH_1);
        assert_eq!(two["hash"], PY_HASH_2);
        // What the bisect child reads back is what was appended.
        let read = Session::read(&session.path);
        assert_eq!(read, session.entries);
        assert!(
            session
                .cached("4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270")
                .is_some()
        );
        assert!(session.cached("4a6c2c0ff8").is_none());
    }

    #[test]
    fn verdicts_follow_git_bisect() {
        assert_eq!(verdict_of(0), "good");
        assert_eq!(verdict_of(125), "skip");
        for exit in [1, 2, 124, 126, 127] {
            assert_eq!(verdict_of(exit), "bad", "exit {exit}");
        }
        // Outside 0..=127 the reference tool records bad too; git bisect
        // itself stops on such a code during the confirmation.
        assert_eq!(verdict_of(137), "bad");
        assert_eq!(verdict_of(-9), "bad");
    }

    #[test]
    fn lines_and_cuts_follow_python() {
        assert_eq!(
            split_lines("a\nb\r\nc\rd\u{2028}e\n"),
            ["a", "b", "c", "d", "e"]
        );
        assert_eq!(split_lines("a\n\nb"), ["a", "", "b"]);
        assert!(split_lines("").is_empty());
        assert_eq!(head("h\u{e9}llo", 2), "h\u{e9}");
        assert_eq!(head("hi", 5), "hi");
        assert_eq!(last_line(b"one\ntwo\n", b""), "two");
        assert_eq!(last_line(b"one\n", b"warn\nlast  \n\n"), "last");
        assert_eq!(last_line(b"progress 1\rprogress 2", b""), "progress 2");
        assert_eq!(last_line(b"", b""), "");
        assert_eq!(last_line("x".repeat(200).as_bytes(), b"").len(), 160);
    }

    #[test]
    fn a_lead_is_a_memory_that_starts_with_a_commit_sha() {
        assert_eq!(
            commit_named("Commit 4a6c2c0ff8: Adding retries").as_deref(),
            Some("4a6c2c0ff8")
        );
        assert_eq!(
            commit_named("  Commit abcdef1 fix").as_deref(),
            Some("abcdef1")
        );
        assert_eq!(
            commit_named("Commit abcdef1\nbody").as_deref(),
            Some("abcdef1")
        );
        let full = "a".repeat(40);
        assert_eq!(commit_named(&format!("Commit {full}: x")), Some(full));
        assert_eq!(commit_named("Commit abcdef: too short"), None);
        assert_eq!(
            commit_named(&format!("Commit {}: too long", "a".repeat(41))),
            None
        );
        assert_eq!(commit_named("Commit abcdef1"), None);
        assert_eq!(commit_named("Commit ABCDEF1: upper"), None);
        assert_eq!(commit_named("Issue 4026: Commit abcdef1: x"), None);

        let walk = json!({"causes": [
            {"id": "mem-01", "depth": 1, "content": "Commit 1111111: one"},
            {"id": "mem-02", "depth": 1, "content": "Issue report, not a commit"},
            {"id": "mem-03", "depth": 2, "content": "Commit 3333333: three"},
        ]});
        let leads = leads_of(&walk);
        assert_eq!(leads.len(), 2);
        assert_eq!((leads[0].rank, leads[0].short.as_str()), (1, "1111111"));
        assert_eq!(
            (leads[1].rank, leads[1].depth, leads[1].memory.as_str()),
            (3, 2, "mem-03")
        );
    }

    const DIFF: &str = r"diff --git a/src/calc.sh b/src/calc.sh
index 1111111..2222222 100644
--- a/src/calc.sh
+++ b/src/calc.sh
@@ -1,3 +1,4 @@
 #!/bin/sh
+# a comment
 a=1
 b=2
@@ -10,3 +11,3 @@ footer
 x
-echo $((a + b))
+echo $((a - b))
 y
\ No newline at end of file
diff --git a/new.txt b/new.txt
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+hello
diff --git a/old.txt b/old.txt
deleted file mode 100644
index 4444444..0000000
--- a/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-bye
diff --git a/logo.png b/logo.png
index 5555555..6666666 100644
Binary files a/logo.png and b/logo.png differ
";

    #[test]
    fn a_diff_splits_into_one_unit_per_hunk_and_whole_files() {
        let units = split_hunks(DIFF.as_bytes());
        let shape: Vec<(&str, u64, usize)> = units
            .iter()
            .map(|unit| (unit.file.as_str(), unit.start, unit.added.len()))
            .collect();
        assert_eq!(
            shape,
            [
                ("src/calc.sh", 1, 1),
                ("src/calc.sh", 11, 1),
                ("new.txt", 0, 0),
                ("old.txt", 0, 0),
                ("diff --git a/logo.png b/logo.png", 0, 0),
            ]
        );
        assert_eq!(units[0].added, ["# a comment"]);
        assert_eq!(units[1].added, ["echo $((a - b))"]);
        assert!(units[0].header.starts_with(b"diff --git a/src/calc.sh"));
        assert!(units[0].header.ends_with(b"+++ b/src/calc.sh\n"));
        assert_eq!(units[0].header, units[1].header);
        assert!(units[1].body.ends_with(b"\\ No newline at end of file\n"));
        // Whole-file units carry their own header.
        assert!(units[2].header.is_empty());
        assert!(units[2].body.starts_with(b"diff --git a/new.txt"));
        assert!(units[4].body.ends_with(b"differ\n"));
        assert!(split_hunks(b"").is_empty());
        assert!(split_hunks(b"not a diff\n").is_empty());
    }

    #[test]
    fn a_patch_of_units_repeats_each_file_header_once() {
        let units = split_hunks(DIFF.as_bytes());
        let all: Vec<&Unit> = units.iter().collect();
        assert_eq!(patch_of(&all), DIFF.as_bytes());
        // The second hunk alone still gets its file header.
        let second = patch_of(&[&units[1]]);
        let text = String::from_utf8(second).unwrap();
        assert!(text.starts_with("diff --git a/src/calc.sh b/src/calc.sh\n"));
        assert!(text.contains("@@ -10,3 +11,3 @@ footer\n"));
        assert!(!text.contains("# a comment"));
        assert_eq!(text.matches("+++ b/src/calc.sh").count(), 1);
        let both = String::from_utf8(patch_of(&[&units[0], &units[1]])).unwrap();
        assert_eq!(both.matches("+++ b/src/calc.sh").count(), 1);
        assert_eq!(
            names_of(&[&units[0], &units[1]]),
            "src/calc.sh:1, src/calc.sh:11"
        );
        assert_eq!(hunk_start(b"@@ -843 +843,8 @@ def connect"), 843);
        assert_eq!(hunk_start(b"not a hunk"), 0);
    }

    /// `fails` for a failure that needs every item of `cause`; counts runs
    /// and spends the budget like the real test does.
    fn needs<'a>(
        cause: &'a [u32],
        budget: &'a Cell<i64>,
        runs: &'a Cell<u32>,
    ) -> impl FnMut(&[u32]) -> bool + 'a {
        move |subset| {
            budget.set(budget.get() - 1);
            runs.set(runs.get() + 1);
            cause.iter().all(|item| subset.contains(item))
        }
    }

    #[test]
    fn ddmin_finds_the_one_change_that_fails() {
        let (budget, runs) = (Cell::new(100), Cell::new(0));
        let found = ddmin((0..8).collect(), &budget, needs(&[5], &budget, &runs));
        assert_eq!(found, [5]);
        assert!(runs.get() <= 8, "{} runs", runs.get());
    }

    #[test]
    fn ddmin_keeps_changes_that_only_fail_together() {
        let (budget, runs) = (Cell::new(100), Cell::new(0));
        let found = ddmin((0..10).collect(), &budget, needs(&[2, 7], &budget, &runs));
        assert_eq!(found, [2, 7]);
        // 1-minimal: dropping either one no longer fails.
        let mut check = needs(&[2, 7], &budget, &runs);
        assert!(check(&found));
        assert!(!check(&[2]));
        assert!(!check(&[7]));
    }

    #[test]
    fn ddmin_leaves_short_inputs_alone() {
        let (budget, runs) = (Cell::new(100), Cell::new(0));
        assert_eq!(ddmin(vec![4], &budget, needs(&[4], &budget, &runs)), [4]);
        assert_eq!(
            ddmin(Vec::new(), &budget, needs(&[], &budget, &runs)),
            [] as [u32; 0]
        );
        assert_eq!(runs.get(), 0);
    }

    #[test]
    fn ddmin_stops_when_the_budget_is_spent() {
        let (budget, runs) = (Cell::new(3), Cell::new(0));
        let cause = [2, 7, 11];
        let found = ddmin((0..16).collect(), &budget, needs(&cause, &budget, &runs));
        assert_eq!(runs.get(), 3, "one run per unit of budget");
        assert!(budget.get() <= 0);
        // Not minimal yet, but what is returned still fails.
        assert!(found.len() > cause.len());
        assert!(cause.iter().all(|item| found.contains(item)));

        let (budget, runs) = (Cell::new(0), Cell::new(0));
        let untouched = ddmin((0..4).collect(), &budget, needs(&[1], &budget, &runs));
        assert_eq!(untouched, [0, 1, 2, 3]);
        assert_eq!(runs.get(), 0);
    }

    #[test]
    fn git_names_the_first_bad_commit_on_its_own_line() {
        let sha = "4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270";
        assert_eq!(
            first_bad_named(&format!("{sha} is the first bad commit")).as_deref(),
            Some(sha)
        );
        assert_eq!(
            first_bad_named("Bisecting: 3 revisions left to test after this"),
            None
        );
        assert_eq!(first_bad_named("4a6c2c0 is the first bad commit"), None);
    }

    #[test]
    fn paths_are_made_absolute_without_touching_symlinks() {
        assert_eq!(
            abspath(Path::new("/a/./b/../c")).unwrap(),
            Path::new("/a/c")
        );
        let relative = abspath(Path::new("x/y.json")).unwrap();
        assert!(relative.is_absolute());
        assert!(relative.ends_with("x/y.json"));
    }

    /// A report whose chain is built here, with no repo behind it.
    fn write_report(dir: &Path, tamper: impl FnOnce(&mut Value)) -> PathBuf {
        let mut session = Session::new(dir.join("probes.jsonl"));
        for (commit, verdict, exit) in [("1".repeat(40), "good", 0), ("2".repeat(40), "bad", 1)] {
            let run = TestRun {
                exit,
                said: format!("said {verdict}"),
                at: "2026-10-05T23:45:31+00:00".to_string(),
            };
            let mut entry = new_entry(&commit, "subject", &run, "feed", "baseline");
            entry.insert("memory".to_string(), Value::Null);
            session.append(entry).unwrap();
        }
        let mut report = json!({
            "repo": dir.join("no-such-repo"),
            "bad": {"ref": "v2"},
            "oracle": {"file": "test-command.sh", "sha256": "feed"},
            "probes": session.entries,
            "first_bad_commit": "2".repeat(40),
            "chain_head": text_of(session.entries.last().unwrap(), "hash"),
            "why": null,
        });
        tamper(&mut report);
        let path = dir.join("report.json");
        fs::write(&path, pretty_json(&report)).unwrap();
        path
    }

    #[test]
    fn check_passes_an_untouched_report_and_fails_a_changed_one() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(check(&write_report(dir.path(), |_| {})).unwrap(), 0);
        // One verdict changed after the fact.
        let changed = write_report(dir.path(), |report| {
            report["probes"][1]["verdict"] = json!("good");
        });
        assert_eq!(check(&changed).unwrap(), 1);
        // A probe dropped from the end, head left as it was.
        let dropped = write_report(dir.path(), |report| {
            report["probes"].as_array_mut().unwrap().pop();
        });
        assert_eq!(check(&dropped).unwrap(), 1);
        // An intact chain that names a commit no probe found bad.
        let renamed = write_report(dir.path(), |report| {
            report["first_bad_commit"] = json!("1".repeat(40));
        });
        assert_eq!(check(&renamed).unwrap(), 1);
        // An undo patch that is not the one the report hashed.
        fs::write(dir.path().join("report.undo.patch"), "other bytes").unwrap();
        let patched = write_report(dir.path(), |report| {
            report["why"] = json!({"undo_patch": {"file": "report.undo.patch", "sha256": "00"}});
        });
        assert_eq!(check(&patched).unwrap(), 1);
    }

    fn protocol_fields(frozen_at: &str) -> Entry {
        entry(json!({
            "slug": "tinyproj",
            "failure_memory": "mem-0000000000000021",
            "good": "1".repeat(40),
            "bad": "2".repeat(40),
            "oracle_sha256": "feed".repeat(16),
            "rule": PROTOCOL_RULE,
            "candidates": ["3".repeat(40), "4".repeat(40)],
            "max_candidates": 12,
            "max_line_runs": 24,
            "frozen_at": frozen_at,
        }))
    }

    #[test]
    fn the_protocol_hash_matches_the_python_vector_and_skips_what_the_report_adds() {
        // hashlib.sha256(json.dumps(proto, sort_keys=True,
        // separators=(",", ":")).encode()).hexdigest() of the same fields.
        let python = "78b455558c77ced14264fd17125da27df66c64fe52842444fd15d2fdbc7db7a9";
        let mut protocol = protocol_fields("2026-10-06T00:05:06+00:00");
        assert_eq!(protocol_hash(&protocol), python);
        protocol.insert("sha256".to_string(), json!(python));
        protocol.insert("memory".to_string(), json!("mem-0000000000000755"));
        assert_eq!(protocol_hash(&protocol), python);
        protocol.insert("candidates".to_string(), json!(["3".repeat(40)]));
        assert_ne!(protocol_hash(&protocol), python);
    }

    #[test]
    fn check_holds_the_protocol_to_its_hash_and_to_the_first_test() {
        let dir = tempfile::tempdir().unwrap();
        // The probes of `write_report` ran at 23:45:31.
        let with_protocol = |frozen_at: &str, edit: fn(&mut Value)| {
            let mut protocol = Value::Object(protocol_fields(frozen_at));
            protocol["sha256"] = json!(protocol_hash(protocol.as_object().unwrap()));
            protocol["memory"] = json!("mem-01");
            edit(&mut protocol);
            write_report(dir.path(), |report| {
                report["protocol"] = protocol;
                report["verdict_card"] = json!([
                    {"rung": "LEAD", "holds": true, "statement": "s", "proof": "p"},
                    {"rung": "REVERSED", "holds": false, "statement": "s", "proof": "no run"},
                ]);
            })
        };
        let frozen_first = with_protocol("2026-10-05T23:45:30+00:00", |_| {});
        assert_eq!(check(&frozen_first).unwrap(), 0);
        let same_second = with_protocol("2026-10-05T23:45:31+00:00", |_| {});
        assert_eq!(check(&same_second).unwrap(), 0);
        // A lead added to the protocol after it was hashed.
        let widened = with_protocol("2026-10-05T23:45:30+00:00", |protocol| {
            protocol["candidates"] = json!(["5".repeat(40)]);
        });
        assert_eq!(check(&widened).unwrap(), 1);
        // A protocol written down after the first test had already run.
        let frozen_late = with_protocol("2026-10-05T23:45:32+00:00", |_| {});
        assert_eq!(check(&frozen_late).unwrap(), 1);
    }
}
