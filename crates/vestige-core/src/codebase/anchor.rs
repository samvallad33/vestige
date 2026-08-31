//! Self-verifying source anchors for code memory.
//!
//! # The defect this exists to fix
//!
//! A code memory used to anchor to source with nothing but a file path. The
//! `codebase` MCP tool printed the caller's `files` array into the memory's
//! markdown body (`crates/vestige-mcp/src/tools/codebase_unified.rs`) and the
//! in-process [`crate::codebase::CodeEntity`] type carried
//! `file_path: Option<PathBuf>` plus `line_number: Option<u32>`
//! (`crates/vestige-core/src/codebase/types.rs`). Neither shape can answer the
//! only question that matters at retrieval time:
//!
//! > Does the code this memory describes still exist, and does it still say
//! > the same thing?
//!
//! Because nothing could answer it, a memory that had rotted was served with
//! exactly the same confidence as one that was still true. That is the actual
//! failure: not that a memory goes wrong, but that going wrong is invisible.
//!
//! # Why symbols alone are not enough
//!
//! Anchoring to a symbol instead of a line is strictly better - `state.py:552`
//! rots within a week, `state.py::load_config` survives an insertion above it.
//! But symbols get renamed, moved between files, and - worst of all - *kept*
//! while their body is rewritten. A symbol-only anchor still cannot detect the
//! dangerous case where the name resolves but the behavior the memory
//! described is gone.
//!
//! So an anchor stores three layers, weakest to strongest:
//!
//! 1. `file_path` - where to look.
//! 2. `symbol` - what to look for. Survives line drift.
//! 3. `content_hash` - a blake3 hash of the *normalized anchored span*.
//!    Survives renames and relocation, and is the only layer that can tell
//!    "same name, different code" from "still true".
//!
//! Line numbers are still recorded, but only as a reporting hint. They are
//! never the identity: a span that moved but hashes identically is
//! [`AnchorStatus::Moved`], which is fresh, not stale.
//!
//! # Preservation, not deletion
//!
//! Verification never edits or deletes a memory. A stale anchor is *reported*
//! as stale so the reader can see it; the memory itself is left exactly as the
//! user wrote it. Deleting would hide the problem instead of fixing it.
//!
//! # Migration safety
//!
//! Every code memory written before this module existed has no anchor row at
//! all, and anchors captured against an unreadable file have a `NULL`
//! `content_hash`. Both degrade to [`AnchorStatus::Unverifiable`] - explicitly
//! "we cannot check this", never "this is wrong". Telling someone their
//! correct memory is stale would be a worse bug than the one being fixed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::storage::{Result as StorageResult, SqliteMemoryStore};

/// Domain separator so an anchor hash can never collide with another blake3
/// use in the codebase (embeddings, connector content hashes, security
/// fingerprints).
const ANCHOR_HASH_DOMAIN: &str = "vestige:code-anchor:v1\n";

/// Hex characters kept from the blake3 digest. 32 hex chars = 128 bits, far
/// beyond what a collision between two source spans would ever need, and short
/// enough to print in a tool response.
const ANCHOR_HASH_LEN: usize = 32;

/// Files larger than this are not hashed. A 4 MB minified bundle is not a
/// meaningful anchor target, and reading one on every `get_context` would make
/// retrieval slow enough that users disable verification - which is the same
/// as not having it.
pub const MAX_ANCHORED_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Upper bound on how many lines one anchor may cover. A memory about "this
/// whole 3000-line file" is not an anchor, it is a file reference, and hashing
/// it would mark the memory stale on every unrelated edit.
pub const MAX_SPAN_LINES: usize = 400;

/// Cap on the relocation scan. Beyond this the file is large enough that a
/// full window scan is not worth the retrieval latency; the anchor degrades to
/// an honest [`AnchorStatus::Unverifiable`] rather than a guessed verdict.
const MAX_RELOCATION_WINDOWS: usize = 50_000;

/// How many lines after a definition line to keep looking for the opening
/// brace of its block (a wrapped signature).
const BRACE_LOOKAHEAD: usize = 6;

/// Tokens that make a line containing `symbol` look like a *definition* rather
/// than a call site. Deliberately cross-language: this module must work on a
/// repository it has never seen without a parser for every language in it.
const DEFINITION_KEYWORDS: &[&str] = &[
    "fn",
    "func",
    "function",
    "def",
    "class",
    "struct",
    "enum",
    "trait",
    "impl",
    "interface",
    "type",
    "const",
    "static",
    "let",
    "var",
    "val",
    "public",
    "private",
    "protected",
    "export",
    "module",
    "package",
    "namespace",
    "sub",
    "procedure",
];

// ============================================================================
// STATUS
// ============================================================================

/// The verdict of checking one anchor against the source tree as it is *now*.
///
/// The split that matters is `is_stale()`: only [`Drifted`](Self::Drifted) and
/// [`Missing`](Self::Missing) accuse a memory of being wrong, and both require
/// positive evidence (a stored hash that no longer matches). Everything else
/// is either fresh or an honest admission that we cannot tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorStatus {
    /// The anchored span still hashes identically, at the recorded position.
    Verified,
    /// The anchored span still hashes identically but lives at a different
    /// line now. The memory is *fresh*; only the coordinate moved. This is the
    /// case a line-number anchor would have reported as a false alarm.
    Moved,
    /// The anchor target is still findable (the file exists, and the symbol
    /// name is still present) but the anchored content no longer matches. The
    /// memory may describe code that has since changed underneath it. This is
    /// the dangerous case the user found by hand.
    Drifted,
    /// The anchored content is gone and the symbol cannot be found - or the
    /// file itself no longer exists.
    Missing,
    /// We cannot check this anchor: it predates content hashing, no span could
    /// be captured, or the file is unreadable/too large today. Never an
    /// accusation - an unverifiable memory may be perfectly correct.
    Unverifiable,
}

impl AnchorStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Moved => "moved",
            Self::Drifted => "drifted",
            Self::Missing => "missing",
            Self::Unverifiable => "unverifiable",
        }
    }

    /// Parse the persisted string form. Deliberately not `FromStr`: this is
    /// a storage detail, not a user-facing parse.
    pub fn parse_status(raw: &str) -> Option<Self> {
        match raw {
            "verified" => Some(Self::Verified),
            "moved" => Some(Self::Moved),
            "drifted" => Some(Self::Drifted),
            "missing" => Some(Self::Missing),
            "unverifiable" => Some(Self::Unverifiable),
            _ => None,
        }
    }

    /// Does this verdict mean the memory should be shown with a staleness
    /// warning? Only positive evidence of divergence counts. "We could not
    /// check" is never staleness.
    pub fn is_stale(&self) -> bool {
        matches!(self, Self::Drifted | Self::Missing)
    }

    /// Does this verdict confirm the memory still matches the code?
    pub fn is_fresh(&self) -> bool {
        matches!(self, Self::Verified | Self::Moved)
    }
}

// ============================================================================
// ANCHOR
// ============================================================================

/// What the caller asked to anchor to, before it is resolved against the tree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnchorDraft {
    /// Repository-relative (or absolute) path to the anchored file.
    pub file_path: String,
    /// Symbol the memory is about, if any.
    pub symbol: Option<String>,
    /// Optional caller-supplied kind ("fn", "class", ...), for display only.
    pub symbol_kind: Option<String>,
    /// Explicit 1-based start line, when the caller knows the exact span.
    pub start_line: Option<u32>,
    /// Explicit 1-based inclusive end line.
    pub end_line: Option<u32>,
}

impl AnchorDraft {
    pub fn new(file_path: impl Into<String>) -> Self {
        Self {
            file_path: file_path.into(),
            ..Default::default()
        }
    }

    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    pub fn with_lines(mut self, start: u32, end: u32) -> Self {
        self.start_line = Some(start);
        self.end_line = Some(end);
        self
    }

    /// Parse the compact `path#symbol` / `path:start-end` forms accepted by the
    /// `codebase` tool's `files` array, so existing callers get real anchoring
    /// without a schema change on their side. A plain path stays a plain path.
    pub fn parse(raw: &str) -> Self {
        let raw = raw.trim();
        if let Some((path, symbol)) = raw.split_once('#')
            && !path.is_empty()
            && !symbol.is_empty()
        {
            return Self::new(path).with_symbol(symbol);
        }
        // `path:12-40` and `path:12`. Guarded so a Windows drive letter
        // (`C:\src\x.rs`) and a plain path with a colon are left alone.
        if let Some((path, tail)) = raw.rsplit_once(':')
            && path.len() > 1
            && !tail.is_empty()
        {
            let (start_raw, end_raw) = match tail.split_once('-') {
                Some((a, b)) => (a, b),
                None => (tail, tail),
            };
            if let (Ok(start), Ok(end)) = (start_raw.parse::<u32>(), end_raw.parse::<u32>())
                && start >= 1
                && end >= start
            {
                return Self::new(path).with_lines(start, end);
            }
        }
        Self::new(raw)
    }
}

/// A persisted, self-verifying anchor from one memory to one place in source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeAnchor {
    pub id: String,
    pub node_id: String,
    pub file_path: String,
    pub symbol: Option<String>,
    pub symbol_kind: Option<String>,
    /// Capture-time first line of the anchored span (1-based). Reporting hint
    /// only - a shifted line never makes a memory stale on its own.
    pub start_line: Option<u32>,
    /// Capture-time last line of the anchored span (1-based, inclusive).
    pub end_line: Option<u32>,
    /// Number of normalized lines the hash covers.
    pub span_lines: Option<u32>,
    /// blake3 of the normalized span. `None` means unverifiable, not stale.
    pub content_hash: Option<String>,
    pub captured_at: DateTime<Utc>,
    pub last_verified_at: Option<DateTime<Utc>>,
    pub last_status: Option<AnchorStatus>,
}

impl CodeAnchor {
    /// Is there enough stored information to actually re-check this anchor?
    pub fn is_verifiable(&self) -> bool {
        self.content_hash.is_some() && self.span_lines.is_some_and(|n| n > 0)
    }
}

/// The result of checking one anchor against the tree as it is now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorVerification {
    pub anchor_id: String,
    pub node_id: String,
    pub file_path: String,
    pub symbol: Option<String>,
    pub status: AnchorStatus,
    /// Human-readable reason, written to be read by whoever is about to trust
    /// (or not trust) the memory.
    pub detail: String,
    pub recorded_line: Option<u32>,
    /// Where the anchored content or symbol lives now, when we found it.
    pub current_line: Option<u32>,
    pub checked_at: DateTime<Utc>,
}

impl AnchorVerification {
    pub fn is_stale(&self) -> bool {
        self.status.is_stale()
    }
}

// ============================================================================
// NORMALIZATION AND HASHING
// ============================================================================

/// Normalize source for hashing: drop blank lines, trim each remaining line,
/// and keep its original 1-based line number.
///
/// Whitespace insensitivity is deliberate. Re-indenting a function (moving it
/// into a new module, changing a nesting level, a formatter pass) does not
/// change what the code says, and marking a correct memory stale for it would
/// train users to ignore the warning - which would destroy the feature's only
/// value. Any token change still changes the hash.
fn normalize(source: &str) -> Vec<(u32, &str)> {
    source
        .lines()
        .enumerate()
        .filter_map(|(idx, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(((idx + 1) as u32, trimmed))
            }
        })
        .collect()
}

/// Hash an already-normalized run of lines.
fn hash_normalized(lines: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ANCHOR_HASH_DOMAIN.as_bytes());
    for line in lines {
        hasher.update(line.as_bytes());
        hasher.update(b"\n");
    }
    hasher.finalize().to_hex()[..ANCHOR_HASH_LEN].to_string()
}

/// Hash the normalized form of an arbitrary source span. Public so a caller can
/// hash a snippet it already holds without going through the filesystem.
pub fn hash_span(source: &str) -> Option<(String, u32, u32, u32)> {
    let normalized = normalize(source);
    if normalized.is_empty() {
        return None;
    }
    let texts: Vec<&str> = normalized.iter().map(|(_, t)| *t).collect();
    let first = normalized.first().expect("non-empty").0;
    let last = normalized.last().expect("non-empty").0;
    Some((
        hash_normalized(&texts),
        first,
        last,
        normalized.len() as u32,
    ))
}

// ============================================================================
// SYMBOL LOCATION
// ============================================================================

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

/// Whole-word (identifier-boundary) search for `needle` in `haystack`.
fn contains_symbol(haystack: &str, needle: &str) -> bool {
    symbol_offsets(haystack, needle).next().is_some()
}

fn symbol_offsets<'a>(haystack: &'a str, needle: &'a str) -> impl Iterator<Item = usize> + 'a {
    haystack.match_indices(needle).filter_map(move |(at, _)| {
        let before_ok = haystack[..at]
            .chars()
            .next_back()
            .is_none_or(|c| !is_ident_char(c));
        let after_ok = haystack[at + needle.len()..]
            .chars()
            .next()
            .is_none_or(|c| !is_ident_char(c));
        if before_ok && after_ok {
            Some(at)
        } else {
            None
        }
    })
}

/// Statement keywords that make `<word> symbol(` a *call*, not a definition.
/// Without these, `return load_config(path)` reads as a C-style signature.
const CALL_LEAD_KEYWORDS: &[&str] = &[
    "return", "await", "yield", "if", "while", "for", "new", "throw", "raise", "assert", "print",
    "else", "elif", "match", "case", "in", "and", "or", "not",
];

/// Does this line look like it *defines* `symbol`, as opposed to calling it?
///
/// Language-agnostic heuristic, in preference order:
///
/// 1. A definition keyword appears as a word before the symbol
///    (`pub fn load_config`, `def load_config`, `export class Foo`). This one
///    rule covers Rust, Python, TS/JS, Go, Java, C#, Ruby and PHP.
/// 2. Binding shapes: `symbol =`, `symbol:`, `symbol<`, `symbol {`, as long as
///    the symbol is not being read off something else (`obj.symbol`).
/// 3. `symbol(` only when the prefix looks like a return type or modifier
///    (`int load_config(`), never a bare call statement (`load_config(x);`)
///    and never a call in expression position (`x = load_config(`,
///    `return load_config(`).
fn line_defines_symbol(line: &str, symbol: &str) -> bool {
    let Some(at) = symbol_offsets(line, symbol).next() else {
        return false;
    };
    let prefix = &line[..at];
    // A definition keyword only defines *this* symbol when no binding operator
    // sits between them: in `let c = load_config(path)` the `let` defines `c`,
    // and `load_config` is a call on the right-hand side.
    if !prefix.contains('=')
        && prefix
            .split(|c: char| !is_ident_char(c))
            .any(|word| DEFINITION_KEYWORDS.contains(&word))
    {
        return true;
    }

    let prefix_trimmed = prefix.trim();
    // `obj.symbol(...)`, `(symbol)`, `[symbol]` are never definitions.
    if prefix_trimmed.starts_with(['.', '(', '[']) || prefix_trimmed.ends_with('.') {
        return false;
    }

    let rest = line[at + symbol.len()..].trim_start();
    if rest.starts_with(['=', ':', '<', '{']) {
        return true;
    }
    if !rest.starts_with('(') {
        return false;
    }

    // C-style `int load_config(...)`: the prefix must look like a type or
    // modifier, i.e. end in an identifier and not be a statement keyword.
    let last_word = prefix_trimmed
        .rsplit(|c: char| !is_ident_char(c))
        .next()
        .unwrap_or("");
    !last_word.is_empty()
        && prefix_trimmed.ends_with(is_ident_char)
        && !CALL_LEAD_KEYWORDS.contains(&last_word)
}

/// First line (1-based) that looks like the definition of `symbol`.
pub fn find_symbol_definition(source: &str, symbol: &str) -> Option<u32> {
    source
        .lines()
        .enumerate()
        .find(|(_, line)| line_defines_symbol(line, symbol))
        .map(|(idx, _)| (idx + 1) as u32)
}

/// First line (1-based) mentioning `symbol` at all. Used only to distinguish
/// [`AnchorStatus::Drifted`] ("the name is still here, the code changed") from
/// [`AnchorStatus::Missing`] ("it is gone").
pub fn find_symbol_anywhere(source: &str, symbol: &str) -> Option<u32> {
    source
        .lines()
        .enumerate()
        .find(|(_, line)| contains_symbol(line, symbol))
        .map(|(idx, _)| (idx + 1) as u32)
}

fn indent_width(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Find the last line (1-based, inclusive) of the block that starts at
/// `def_line`. Braces first (Rust/TS/Go/Java/C/...), indentation second
/// (Python/Ruby/YAML), and the definition line alone as the floor.
fn block_end(lines: &[&str], def_line: u32) -> u32 {
    let def_idx = (def_line as usize).saturating_sub(1);
    if def_idx >= lines.len() {
        return def_line;
    }
    let hard_cap = (def_idx + MAX_SPAN_LINES).min(lines.len());

    // Brace matching. Start counting from the first line at/after the
    // definition that contains an opening brace, to tolerate a wrapped
    // signature spilling the `{` onto a later line.
    let brace_start =
        (def_idx..(def_idx + BRACE_LOOKAHEAD).min(lines.len())).find(|i| lines[*i].contains('{'));
    if let Some(start) = brace_start {
        let mut depth: i32 = 0;
        for (offset, line) in lines[start..hard_cap].iter().enumerate() {
            depth += line.matches('{').count() as i32;
            depth -= line.matches('}').count() as i32;
            if depth <= 0 && line.contains('}') {
                return (start + offset + 1) as u32;
            }
        }
        // Unbalanced within the cap: fall through to the indentation rule
        // rather than swallowing the rest of the file.
    }

    // Indentation rule.
    let base = indent_width(lines[def_idx]);
    let mut end = def_idx;
    for (offset, line) in lines[def_idx + 1..hard_cap].iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if indent_width(line) > base {
            end = def_idx + 1 + offset;
        } else {
            break;
        }
    }
    (end + 1) as u32
}

/// Resolve a draft to a concrete 1-based inclusive line span in `source`.
/// `None` means "path-only anchor": nothing to hash, so it will degrade to
/// [`AnchorStatus::Unverifiable`] rather than pretend.
fn resolve_span(source: &str, draft: &AnchorDraft) -> Option<(u32, u32)> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return None;
    }
    let total = lines.len() as u32;

    if let Some(start) = draft.start_line {
        if start < 1 || start > total {
            return None;
        }
        let end = draft
            .end_line
            .unwrap_or(start)
            .clamp(start, total)
            .min(start + MAX_SPAN_LINES as u32 - 1);
        return Some((start, end));
    }

    let symbol = draft.symbol.as_deref()?;
    let def_line = find_symbol_definition(source, symbol)?;
    Some((def_line, block_end(&lines, def_line)))
}

// ============================================================================
// CAPTURE
// ============================================================================

/// Resolve an anchor path against the repository root. Absolute paths are used
/// as given (a user may legitimately record one).
fn resolve_path(repo_root: &Path, file_path: &str) -> PathBuf {
    let candidate = Path::new(file_path);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        repo_root.join(candidate)
    }
}

/// Read a file for anchoring, refusing anything too large or not valid UTF-8.
fn read_anchor_file(path: &Path) -> Result<String, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("cannot stat file: {e}"))?;
    if !meta.is_file() {
        return Err("path is not a regular file".to_string());
    }
    if meta.len() > MAX_ANCHORED_FILE_BYTES {
        return Err(format!(
            "file is {} bytes, above the {} byte anchoring limit",
            meta.len(),
            MAX_ANCHORED_FILE_BYTES
        ));
    }
    std::fs::read_to_string(path).map_err(|e| format!("cannot read file as UTF-8 text: {e}"))
}

/// Turn a draft into a stored anchor by hashing what it points at *today*.
///
/// This never fails: an unreadable file, a missing symbol, or a path-only
/// draft all produce an anchor with `content_hash = None`. Recording an
/// unverifiable anchor is strictly better than recording nothing, because the
/// path and symbol are still preserved and the retrieval path can say
/// truthfully that it could not check them.
pub fn capture_anchor(node_id: &str, repo_root: &Path, draft: &AnchorDraft) -> CodeAnchor {
    let now = Utc::now();
    let mut anchor = CodeAnchor {
        id: format!("anchor-{}", Uuid::new_v4()),
        node_id: node_id.to_string(),
        file_path: draft.file_path.clone(),
        symbol: draft.symbol.clone(),
        symbol_kind: draft.symbol_kind.clone(),
        start_line: draft.start_line,
        end_line: draft.end_line,
        span_lines: None,
        content_hash: None,
        captured_at: now,
        last_verified_at: None,
        last_status: None,
    };

    let path = resolve_path(repo_root, &draft.file_path);
    let Ok(source) = read_anchor_file(&path) else {
        return anchor;
    };
    let Some((start, end)) = resolve_span(&source, draft) else {
        return anchor;
    };

    let lines: Vec<&str> = source.lines().collect();
    let slice = lines[(start as usize - 1)..(end as usize).min(lines.len())].join("\n");
    if let Some((hash, first_rel, last_rel, count)) = hash_span(&slice) {
        // `hash_span` numbers lines within the slice; translate back to the
        // file's own coordinates so the recorded position is meaningful.
        anchor.start_line = Some(start + first_rel - 1);
        anchor.end_line = Some(start + last_rel - 1);
        anchor.span_lines = Some(count);
        anchor.content_hash = Some(hash);
    }
    anchor
}

// ============================================================================
// VERIFY
// ============================================================================

fn verdict(
    anchor: &CodeAnchor,
    status: AnchorStatus,
    detail: impl Into<String>,
    current_line: Option<u32>,
) -> AnchorVerification {
    AnchorVerification {
        anchor_id: anchor.id.clone(),
        node_id: anchor.node_id.clone(),
        file_path: anchor.file_path.clone(),
        symbol: anchor.symbol.clone(),
        status,
        detail: detail.into(),
        recorded_line: anchor.start_line,
        current_line,
        checked_at: Utc::now(),
    }
}

/// Check one anchor against the working tree as it is now.
///
/// The ordering of the checks is the whole contract:
///
/// * A missing file is [`Missing`](AnchorStatus::Missing) even for an
///   unverifiable anchor - "the file you named is gone" is a fact we *can*
///   establish without a hash.
/// * A present file with no stored hash is
///   [`Unverifiable`](AnchorStatus::Unverifiable). This is the legacy path and
///   it must never say "stale".
/// * A stored hash that still matches somewhere is fresh, wherever it moved to.
/// * Only a stored hash that no longer matches anywhere can accuse the memory.
pub fn verify_anchor(anchor: &CodeAnchor, repo_root: &Path) -> AnchorVerification {
    let path = resolve_path(repo_root, &anchor.file_path);

    if !path.exists() {
        return verdict(
            anchor,
            AnchorStatus::Missing,
            format!(
                "`{}` no longer exists under {}. Whatever this memory says about it cannot be checked against anything.",
                anchor.file_path,
                repo_root.display()
            ),
            None,
        );
    }

    let source = match read_anchor_file(&path) {
        Ok(text) => text,
        Err(reason) => {
            return verdict(
                anchor,
                AnchorStatus::Unverifiable,
                format!("`{}` exists but {reason}.", anchor.file_path),
                None,
            );
        }
    };

    if !anchor.is_verifiable() {
        let symbol_note = match anchor.symbol.as_deref() {
            Some(symbol) => match find_symbol_definition(&source, symbol)
                .or_else(|| find_symbol_anywhere(&source, symbol))
            {
                Some(line) => format!(" `{symbol}` is currently at line {line}."),
                None => format!(" `{symbol}` is not present in the file today."),
            },
            None => String::new(),
        };
        return verdict(
            anchor,
            AnchorStatus::Unverifiable,
            format!(
                "No content hash was recorded for this anchor, so it cannot be verified - it may well be correct.{symbol_note} Re-save this memory with an anchored symbol or line span to make it self-checking."
            ),
            None,
        );
    }

    let expected = anchor.content_hash.as_deref().expect("is_verifiable");
    let span_len = anchor.span_lines.expect("is_verifiable") as usize;
    let normalized = normalize(&source);

    if normalized.len() >= span_len {
        let texts: Vec<&str> = normalized.iter().map(|(_, t)| *t).collect();

        // Fast path: check the recorded position first. Unchanged files, which
        // are the overwhelming majority on any single retrieval, cost one hash.
        if let Some(recorded) = anchor.start_line
            && let Some(idx) = normalized.iter().position(|(line, _)| *line == recorded)
            && idx + span_len <= texts.len()
            && hash_normalized(&texts[idx..idx + span_len]) == expected
        {
            return verdict(
                anchor,
                AnchorStatus::Verified,
                format!(
                    "Anchored content at `{}`{} still matches byte for byte.",
                    anchor.file_path,
                    anchor
                        .symbol
                        .as_deref()
                        .map(|s| format!(" ({s})"))
                        .unwrap_or_default()
                ),
                Some(recorded),
            );
        }

        // Relocation scan: the same content may simply have moved.
        let windows = texts.len() - span_len + 1;
        if windows <= MAX_RELOCATION_WINDOWS {
            for start in 0..windows {
                if hash_normalized(&texts[start..start + span_len]) == expected {
                    let now_at = normalized[start].0;
                    return verdict(
                        anchor,
                        AnchorStatus::Moved,
                        format!(
                            "Anchored content is unchanged but moved from line {} to line {now_at} in `{}`. The memory is still accurate; only the coordinate shifted.",
                            anchor
                                .start_line
                                .map(|l| l.to_string())
                                .unwrap_or_else(|| "?".into()),
                            anchor.file_path
                        ),
                        Some(now_at),
                    );
                }
            }
        } else {
            return verdict(
                anchor,
                AnchorStatus::Unverifiable,
                format!(
                    "`{}` has {} candidate positions, above the {MAX_RELOCATION_WINDOWS} scan limit, so relocation could not be ruled out. Not treating this as stale.",
                    anchor.file_path, windows
                ),
                None,
            );
        }
    }

    // The recorded content is not in this file any more. Is the symbol?
    if let Some(symbol) = anchor.symbol.as_deref()
        && let Some(line) = find_symbol_definition(&source, symbol)
            .or_else(|| find_symbol_anywhere(&source, symbol))
    {
        return verdict(
            anchor,
            AnchorStatus::Drifted,
            format!(
                "`{symbol}` still exists in `{}` (now around line {line}) but its code has changed since this memory was written. Re-read the source before trusting this memory.",
                anchor.file_path
            ),
            Some(line),
        );
    }

    verdict(
        anchor,
        AnchorStatus::Missing,
        format!(
            "The code this memory was anchored to is no longer present in `{}`{}. Treat the memory as historical until it is re-checked.",
            anchor.file_path,
            anchor
                .symbol
                .as_deref()
                .map(|s| format!(", and `{s}` is gone too"))
                .unwrap_or_default()
        ),
        None,
    )
}

// ============================================================================
// STORAGE
// ============================================================================

const ANCHOR_COLUMNS: &str = "id, node_id, file_path, symbol, symbol_kind, start_line, end_line, \
     span_lines, content_hash, captured_at, last_verified_at, last_status";

fn row_to_anchor(row: &rusqlite::Row<'_>) -> rusqlite::Result<CodeAnchor> {
    let captured_at: String = row.get(9)?;
    let last_verified_at: Option<String> = row.get(10)?;
    let last_status: Option<String> = row.get(11)?;
    Ok(CodeAnchor {
        id: row.get(0)?,
        node_id: row.get(1)?,
        file_path: row.get(2)?,
        symbol: row.get(3)?,
        symbol_kind: row.get(4)?,
        start_line: row.get::<_, Option<i64>>(5)?.map(|v| v as u32),
        end_line: row.get::<_, Option<i64>>(6)?.map(|v| v as u32),
        span_lines: row.get::<_, Option<i64>>(7)?.map(|v| v as u32),
        content_hash: row.get(8)?,
        captured_at: DateTime::parse_from_rfc3339(&captured_at)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        last_verified_at: last_verified_at
            .and_then(|raw| DateTime::parse_from_rfc3339(&raw).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        last_status: last_status.as_deref().and_then(AnchorStatus::parse_status),
    })
}

impl SqliteMemoryStore {
    /// Persist source anchors for a memory.
    ///
    /// The `node_id` foreign key cascades on delete, so purging a memory takes
    /// its anchored paths and hashes with it - no orphaned record of what file
    /// a purged memory pointed at.
    pub fn record_code_anchors(&self, anchors: &[CodeAnchor]) -> StorageResult<usize> {
        if anchors.is_empty() {
            return Ok(0);
        }
        let conn = self.writer.lock().unwrap();
        let mut written = 0;
        for anchor in anchors {
            conn.execute(
                &format!(
                    "INSERT OR REPLACE INTO code_memory_anchors ({ANCHOR_COLUMNS}) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"
                ),
                params![
                    anchor.id,
                    anchor.node_id,
                    anchor.file_path,
                    anchor.symbol,
                    anchor.symbol_kind,
                    anchor.start_line.map(|v| v as i64),
                    anchor.end_line.map(|v| v as i64),
                    anchor.span_lines.map(|v| v as i64),
                    anchor.content_hash,
                    anchor.captured_at.to_rfc3339(),
                    anchor.last_verified_at.map(|dt| dt.to_rfc3339()),
                    anchor.last_status.map(|s| s.as_str().to_string()),
                ],
            )?;
            written += 1;
        }
        Ok(written)
    }

    /// Load every anchor recorded for one memory.
    pub fn code_anchors_for_node(&self, node_id: &str) -> StorageResult<Vec<CodeAnchor>> {
        let conn = self.reader.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {ANCHOR_COLUMNS} FROM code_memory_anchors WHERE node_id = ?1 \
             ORDER BY file_path, start_line"
        ))?;
        let rows = stmt.query_map(params![node_id], row_to_anchor)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Load anchors for a batch of memories, keyed by node id. Memories with no
    /// anchors are simply absent from the map - that absence is what the
    /// retrieval path reports as "unverifiable", never as "stale".
    pub fn code_anchors_for_nodes(
        &self,
        node_ids: &[String],
    ) -> StorageResult<HashMap<String, Vec<CodeAnchor>>> {
        let mut out: HashMap<String, Vec<CodeAnchor>> = HashMap::new();
        if node_ids.is_empty() {
            return Ok(out);
        }
        let conn = self.reader.lock().unwrap();
        let placeholders = vec!["?"; node_ids.len()].join(", ");
        let mut stmt = conn.prepare(&format!(
            "SELECT {ANCHOR_COLUMNS} FROM code_memory_anchors \
             WHERE node_id IN ({placeholders}) ORDER BY file_path, start_line"
        ))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(node_ids.iter()), row_to_anchor)?;
        for anchor in rows {
            let anchor = anchor?;
            out.entry(anchor.node_id.clone()).or_default().push(anchor);
        }
        Ok(out)
    }

    /// Cache the latest verdict for an anchor. Purely informational: the
    /// retrieval path always re-verifies against the live tree, because a
    /// cached "verified" is exactly the kind of confident lie this feature
    /// exists to remove.
    pub fn record_anchor_verification(
        &self,
        anchor_id: &str,
        status: AnchorStatus,
        checked_at: DateTime<Utc>,
    ) -> StorageResult<()> {
        let conn = self.writer.lock().unwrap();
        conn.execute(
            "UPDATE code_memory_anchors SET last_verified_at = ?1, last_status = ?2 WHERE id = ?3",
            params![checked_at.to_rfc3339(), status.as_str(), anchor_id],
        )?;
        Ok(())
    }

    /// Remove every anchor for a memory. Not needed for `purge_node` (the
    /// foreign key cascades) but useful when a memory is re-anchored.
    pub fn delete_code_anchors_for_node(&self, node_id: &str) -> StorageResult<usize> {
        let conn = self.writer.lock().unwrap();
        Ok(conn.execute(
            "DELETE FROM code_memory_anchors WHERE node_id = ?1",
            params![node_id],
        )?)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const RUST_SOURCE: &str = "\
use std::fs;

pub fn load_config(path: &str) -> Config {
    let raw = fs::read_to_string(path).unwrap();
    parse(&raw)
}

pub fn other() -> u8 {
    7
}
";

    fn write_file(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    fn draft_symbol(file: &str, symbol: &str) -> AnchorDraft {
        AnchorDraft::new(file).with_symbol(symbol)
    }

    // --- normalization + hashing ------------------------------------------

    #[test]
    fn reindenting_and_blank_lines_do_not_change_the_hash() {
        let a = "fn f() {\n    let x = 1;\n}\n";
        let b = "\n        fn f() {\n\n                let x = 1;\n        }\n\n";
        assert_eq!(hash_span(a).unwrap().0, hash_span(b).unwrap().0);
    }

    #[test]
    fn changing_a_single_token_changes_the_hash() {
        let a = "fn f() {\n    let x = 1;\n}\n";
        let b = "fn f() {\n    let x = 2;\n}\n";
        assert_ne!(hash_span(a).unwrap().0, hash_span(b).unwrap().0);
    }

    // --- symbol location ---------------------------------------------------

    #[test]
    fn definition_line_is_preferred_over_a_call_site() {
        let src = "load_config(\"a\");\npub fn load_config(p: &str) {}\n";
        assert_eq!(find_symbol_definition(src, "load_config"), Some(2));
    }

    #[test]
    fn call_sites_in_every_common_shape_are_not_definitions() {
        for call in [
            "load_config(\"a\");",
            "    let c = load_config(path);",
            "    return load_config(path);",
            "    self.load_config(path)",
            "    obj.load_config()",
            "    await load_config(path)",
        ] {
            assert!(
                !line_defines_symbol(call, "load_config"),
                "must not read as a definition: {call}"
            );
        }
    }

    #[test]
    fn definitions_in_several_languages_are_recognized() {
        for def in [
            "pub fn load_config(p: &str) {",       // Rust
            "def load_config(path):",              // Python
            "export function load_config(p) {",    // TS/JS
            "func load_config(p string) error {",  // Go
            "  public void load_config(String p)", // Java
            "int load_config(char *p) {",          // C
            "const load_config = (p) => {",        // JS binding
            "  load_config: (p) => {",             // object literal member
        ] {
            assert!(
                line_defines_symbol(def, "load_config"),
                "must read as a definition: {def}"
            );
        }
    }

    #[test]
    fn symbol_search_respects_identifier_boundaries() {
        let src = "fn load_config_v2() {}\n";
        assert_eq!(find_symbol_anywhere(src, "load_config"), None);
        assert_eq!(find_symbol_anywhere(src, "load_config_v2"), Some(1));
    }

    #[test]
    fn python_style_blocks_end_by_indentation() {
        let src = "def a():\n    x = 1\n    y = 2\n\ndef b():\n    pass\n";
        let lines: Vec<&str> = src.lines().collect();
        let def = find_symbol_definition(src, "a").unwrap();
        assert_eq!(def, 1);
        assert_eq!(block_end(&lines, def), 3);
    }

    #[test]
    fn brace_blocks_end_at_the_matching_close() {
        let lines: Vec<&str> = RUST_SOURCE.lines().collect();
        let def = find_symbol_definition(RUST_SOURCE, "load_config").unwrap();
        assert_eq!(def, 3);
        assert_eq!(block_end(&lines, def), 6);
    }

    // --- draft parsing -----------------------------------------------------

    #[test]
    fn compact_anchor_forms_parse() {
        assert_eq!(
            AnchorDraft::parse("src/state.py#load_config"),
            AnchorDraft::new("src/state.py").with_symbol("load_config")
        );
        assert_eq!(
            AnchorDraft::parse("src/state.py:552-560"),
            AnchorDraft::new("src/state.py").with_lines(552, 560)
        );
        assert_eq!(
            AnchorDraft::parse("src/state.py:552"),
            AnchorDraft::new("src/state.py").with_lines(552, 552)
        );
        // A plain path stays a plain path: existing callers are unchanged.
        assert_eq!(
            AnchorDraft::parse("src/state.py"),
            AnchorDraft::new("src/state.py")
        );
    }

    // --- the four verdicts -------------------------------------------------

    #[test]
    fn unchanged_code_verifies() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", RUST_SOURCE);
        let anchor = capture_anchor("n1", dir.path(), &draft_symbol("src/lib.rs", "load_config"));
        assert!(anchor.is_verifiable(), "symbol anchor must capture a hash");

        let v = verify_anchor(&anchor, dir.path());
        assert_eq!(v.status, AnchorStatus::Verified, "{}", v.detail);
        assert!(!v.is_stale());
    }

    /// The exact false alarm a line-number anchor produces: code shifted down,
    /// nothing about it changed. This must NOT be reported as stale.
    #[test]
    fn shifted_but_identical_code_is_moved_not_stale() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", RUST_SOURCE);
        let anchor = capture_anchor("n1", dir.path(), &draft_symbol("src/lib.rs", "load_config"));
        assert_eq!(anchor.start_line, Some(3));

        let shifted = format!("// a new header\n// and another\n\n{RUST_SOURCE}");
        write_file(dir.path(), "src/lib.rs", &shifted);

        let v = verify_anchor(&anchor, dir.path());
        assert_eq!(v.status, AnchorStatus::Moved, "{}", v.detail);
        assert!(!v.is_stale(), "a pure relocation must never read as stale");
        assert_eq!(v.current_line, Some(6));
    }

    /// The dangerous case the user found by hand: the symbol still resolves,
    /// so every symbol-only scheme says "fine", but the body it described is
    /// gone.
    #[test]
    fn same_symbol_with_a_rewritten_body_is_drifted() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", RUST_SOURCE);
        let anchor = capture_anchor("n1", dir.path(), &draft_symbol("src/lib.rs", "load_config"));

        let rewritten = RUST_SOURCE.replace(
            "    let raw = fs::read_to_string(path).unwrap();\n    parse(&raw)",
            "    Config::from_env()",
        );
        assert_ne!(rewritten, RUST_SOURCE);
        write_file(dir.path(), "src/lib.rs", &rewritten);

        let v = verify_anchor(&anchor, dir.path());
        assert_eq!(v.status, AnchorStatus::Drifted, "{}", v.detail);
        assert!(v.is_stale(), "a rewritten body must be visibly stale");
        assert!(v.detail.contains("load_config"));
    }

    #[test]
    fn a_deleted_symbol_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", RUST_SOURCE);
        let anchor = capture_anchor("n1", dir.path(), &draft_symbol("src/lib.rs", "load_config"));

        write_file(
            dir.path(),
            "src/lib.rs",
            "pub fn other() -> u8 {\n    7\n}\n",
        );
        let v = verify_anchor(&anchor, dir.path());
        assert_eq!(v.status, AnchorStatus::Missing, "{}", v.detail);
        assert!(v.is_stale());
    }

    #[test]
    fn a_deleted_file_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(dir.path(), "src/lib.rs", RUST_SOURCE);
        let anchor = capture_anchor("n1", dir.path(), &draft_symbol("src/lib.rs", "load_config"));
        std::fs::remove_file(&path).unwrap();

        let v = verify_anchor(&anchor, dir.path());
        assert_eq!(v.status, AnchorStatus::Missing);
        assert!(v.detail.contains("no longer exists"));
    }

    // --- migration safety: legacy anchors must NEVER read as stale ---------

    /// The single most important guarantee in this module. A memory saved
    /// before content hashing existed has no hash. Even when the file it names
    /// has been rewritten beyond recognition, it must degrade to
    /// "unverifiable" - telling a user their correct memory is wrong would be
    /// a worse bug than the one being fixed.
    #[test]
    fn a_legacy_anchor_without_a_hash_is_unverifiable_never_stale() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", RUST_SOURCE);

        let legacy = CodeAnchor {
            id: "anchor-legacy".to_string(),
            node_id: "n1".to_string(),
            file_path: "src/lib.rs".to_string(),
            symbol: Some("load_config".to_string()),
            symbol_kind: None,
            start_line: Some(552), // a rotted line number from the old shape
            end_line: None,
            span_lines: None,
            content_hash: None, // the legacy shape: nothing to check against
            captured_at: Utc::now(),
            last_verified_at: None,
            last_status: None,
        };

        let v = verify_anchor(&legacy, dir.path());
        assert_eq!(v.status, AnchorStatus::Unverifiable);
        assert!(
            !v.is_stale(),
            "a hashless anchor must never accuse a memory"
        );

        // ...and it stays unverifiable even after a total rewrite.
        write_file(
            dir.path(),
            "src/lib.rs",
            "totally different\ncontent here\n",
        );
        let v = verify_anchor(&legacy, dir.path());
        assert_eq!(v.status, AnchorStatus::Unverifiable);
        assert!(!v.is_stale());
    }

    /// A path-only anchor (no symbol, no line span) captures no hash, so it is
    /// unverifiable while the file exists - but a deleted file is still a fact
    /// we can state.
    #[test]
    fn a_path_only_anchor_is_unverifiable_until_the_file_disappears() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_file(dir.path(), "src/lib.rs", RUST_SOURCE);
        let anchor = capture_anchor("n1", dir.path(), &AnchorDraft::new("src/lib.rs"));
        assert!(!anchor.is_verifiable());
        assert_eq!(
            verify_anchor(&anchor, dir.path()).status,
            AnchorStatus::Unverifiable
        );

        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            verify_anchor(&anchor, dir.path()).status,
            AnchorStatus::Missing
        );
    }

    #[test]
    fn an_unresolvable_symbol_captures_no_hash_and_stays_unverifiable() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", RUST_SOURCE);
        let anchor = capture_anchor("n1", dir.path(), &draft_symbol("src/lib.rs", "nope"));
        assert!(anchor.content_hash.is_none());
        let v = verify_anchor(&anchor, dir.path());
        assert_eq!(v.status, AnchorStatus::Unverifiable);
        assert!(!v.is_stale());
    }

    #[test]
    fn explicit_line_spans_are_hashed_and_survive_relocation() {
        let dir = tempfile::tempdir().unwrap();
        write_file(dir.path(), "src/lib.rs", RUST_SOURCE);
        let anchor = capture_anchor(
            "n1",
            dir.path(),
            &AnchorDraft::new("src/lib.rs").with_lines(3, 6),
        );
        assert!(anchor.is_verifiable());
        assert_eq!(
            verify_anchor(&anchor, dir.path()).status,
            AnchorStatus::Verified
        );

        write_file(
            dir.path(),
            "src/lib.rs",
            &format!("// header\n{RUST_SOURCE}"),
        );
        assert_eq!(
            verify_anchor(&anchor, dir.path()).status,
            AnchorStatus::Moved
        );
    }

    // --- persistence -------------------------------------------------------

    fn store() -> (SqliteMemoryStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteMemoryStore::new(Some(dir.path().join("t.db"))).unwrap();
        (store, dir)
    }

    fn ingest(store: &SqliteMemoryStore, content: &str) -> String {
        store
            .ingest(crate::IngestInput {
                content: content.to_string(),
                node_type: "pattern".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: vec!["codebase".to_string()],
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap()
            .id
    }

    #[test]
    fn anchors_round_trip_through_storage() {
        let (store, _dir) = store();
        let repo = tempfile::tempdir().unwrap();
        write_file(repo.path(), "src/lib.rs", RUST_SOURCE);
        let node_id = ingest(&store, "load_config reads the file eagerly");

        let anchor = capture_anchor(
            &node_id,
            repo.path(),
            &draft_symbol("src/lib.rs", "load_config"),
        );
        assert_eq!(
            store
                .record_code_anchors(std::slice::from_ref(&anchor))
                .unwrap(),
            1
        );

        let loaded = store.code_anchors_for_node(&node_id).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], anchor);

        let batch = store
            .code_anchors_for_nodes(std::slice::from_ref(&node_id))
            .unwrap();
        assert_eq!(batch.get(&node_id).map(|v| v.len()), Some(1));
    }

    #[test]
    fn a_memory_with_no_anchor_row_is_simply_absent() {
        let (store, _dir) = store();
        let node_id = ingest(&store, "a legacy code memory with no anchors");
        assert!(store.code_anchors_for_node(&node_id).unwrap().is_empty());
        assert!(store.code_anchors_for_nodes(&[node_id]).unwrap().is_empty());
    }

    #[test]
    fn verification_status_is_cached_but_not_trusted() {
        let (store, _dir) = store();
        let repo = tempfile::tempdir().unwrap();
        write_file(repo.path(), "src/lib.rs", RUST_SOURCE);
        let node_id = ingest(&store, "anchored memory");
        let anchor = capture_anchor(
            &node_id,
            repo.path(),
            &draft_symbol("src/lib.rs", "load_config"),
        );
        store
            .record_code_anchors(std::slice::from_ref(&anchor))
            .unwrap();

        store
            .record_anchor_verification(&anchor.id, AnchorStatus::Drifted, Utc::now())
            .unwrap();
        let loaded = store.code_anchors_for_node(&node_id).unwrap();
        assert_eq!(loaded[0].last_status, Some(AnchorStatus::Drifted));
        assert!(loaded[0].last_verified_at.is_some());

        // The live check still wins over the cached verdict.
        assert_eq!(
            verify_anchor(&loaded[0], repo.path()).status,
            AnchorStatus::Verified
        );
    }

    #[test]
    fn purging_a_memory_takes_its_anchors_with_it() {
        let (store, _dir) = store();
        let repo = tempfile::tempdir().unwrap();
        write_file(repo.path(), "src/lib.rs", RUST_SOURCE);
        let node_id = ingest(&store, "anchored memory that will be purged");
        let anchor = capture_anchor(
            &node_id,
            repo.path(),
            &draft_symbol("src/lib.rs", "load_config"),
        );
        store.record_code_anchors(&[anchor]).unwrap();
        assert_eq!(store.code_anchors_for_node(&node_id).unwrap().len(), 1);

        store.purge_node(&node_id, Some("test")).unwrap();
        assert!(
            store.code_anchors_for_node(&node_id).unwrap().is_empty(),
            "purge must not leave the anchored file path behind"
        );
    }

    #[test]
    fn deleting_anchors_for_a_node_is_explicit_and_scoped() {
        let (store, _dir) = store();
        let repo = tempfile::tempdir().unwrap();
        write_file(repo.path(), "src/lib.rs", RUST_SOURCE);
        let keep = ingest(&store, "keep me");
        let drop = ingest(&store, "re-anchor me");
        store
            .record_code_anchors(&[
                capture_anchor(
                    &keep,
                    repo.path(),
                    &draft_symbol("src/lib.rs", "load_config"),
                ),
                capture_anchor(&drop, repo.path(), &draft_symbol("src/lib.rs", "other")),
            ])
            .unwrap();

        assert_eq!(store.delete_code_anchors_for_node(&drop).unwrap(), 1);
        assert!(store.code_anchors_for_node(&drop).unwrap().is_empty());
        assert_eq!(store.code_anchors_for_node(&keep).unwrap().len(), 1);
    }
}
