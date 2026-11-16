//! `vestige hook session-start`: rules that survive the compact.
//!
//! A Claude Code `SessionStart` command hook runs on `startup`, `resume`,
//! `clear` and `compact`. This module turns its stdin payload into the hook's
//! stdout: the decision and correction memories recorded under a few exact
//! tags, each line carrying its memory id, as `additionalContext`.
//!
//! The tags are fixed by rule, never guessed from text:
//!
//! - `pinned`;
//! - each line of an optional `.vestige-rules` file in `cwd` (or the
//!   repository root), one exact tag per line, `#` for comments;
//! - only when asked for (`--repo-tag`): the git repository name of the
//!   payload's `cwd` (the `origin` remote's name, else the main checkout's
//!   folder name, else the folder name), plus its lowercase spelling when
//!   that differs. It is off by default because a store that tags every
//!   memory with its project holds far more than rules under that tag.
//!
//! Each tag is resolved as an exact handle (`recall` `handle`): a memory is
//! listed only when it carries that tag byte for byte. Nothing is searched,
//! ranked by similarity or matched by words. The caller (the CLI) does the
//! store access, through the running owner or the log directly; this module
//! is pure apart from reading the repository's `.git` and `.vestige-rules`
//! files.
//!
//! Fail open: a payload this module cannot use gives `None`, and the hook
//! prints nothing and exits 0, so a broken store never blocks a session.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

/// The fixed tag every session gets, whatever the repository.
pub const PINNED_TAG: &str = "pinned";

/// The per-repository rules file: one exact tag per line.
pub const RULES_FILE: &str = ".vestige-rules";

/// Node types a session gets reminded of.
pub const RULE_TYPES: [&str; 2] = ["decision", "correction"];

/// Most tags read from a rules file.
const MAX_FILE_TAGS: usize = 32;

/// Longest tag accepted from a rules file, in bytes.
const MAX_TAG_BYTES: usize = 200;

/// Largest rules file read, in bytes.
const MAX_RULES_FILE_BYTES: u64 = 16 * 1024;

/// Longest memory content shown on one line, in characters.
const MAX_LINE_CHARS: usize = 600;

/// Defaults for the rendered context.
pub const DEFAULT_MAX_ITEMS: usize = 8;
pub const DEFAULT_MAX_BYTES: usize = 4096;

/// What a `SessionStart` hook receives on stdin. Unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStartInput {
    pub session_id: Option<String>,
    pub cwd: Option<PathBuf>,
    /// `startup`, `resume`, `clear`, `compact` (or `fork`). `startup` when
    /// the payload names none.
    pub source: String,
    pub hook_event_name: Option<String>,
}

/// Parse the hook payload. `None` when it is not a JSON object, or when it
/// names an event other than `SessionStart` (the hook's output would be
/// wrong for that event).
pub fn parse_input(raw: &str) -> Option<SessionStartInput> {
    let value: Value = serde_json::from_str(raw.trim()).ok()?;
    let object = value.as_object()?;
    let text = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
    };
    let hook_event_name = text("hook_event_name");
    if hook_event_name
        .as_deref()
        .is_some_and(|name| name != "SessionStart")
    {
        return None;
    }
    Some(SessionStartInput {
        session_id: text("session_id"),
        cwd: text("cwd").map(PathBuf::from),
        source: text("source").unwrap_or_else(|| "startup".to_string()),
        hook_event_name,
    })
}

/// The exact tags a session in `cwd` is given, in order, without duplicates:
/// the repository name and its lowercase spelling (only when `repo_tag` is
/// set), `pinned`, then the rules file's tags.
pub fn rule_tags(cwd: &Path, repo_tag: bool) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    let mut push = |tag: String| {
        if !tag.is_empty() && !tags.contains(&tag) {
            tags.push(tag);
        }
    };
    let checkout = find_checkout(cwd);
    if repo_tag && let Some(name) = repo_name(cwd, checkout.as_ref()) {
        let lower = name.to_ascii_lowercase();
        push(name);
        push(lower);
    }
    push(PINNED_TAG.to_string());
    let mut files = vec![cwd.join(RULES_FILE)];
    if let Some(checkout) = &checkout
        && checkout.toplevel != cwd
    {
        files.push(checkout.toplevel.join(RULES_FILE));
    }
    for file in files {
        for tag in read_rules_file(&file) {
            push(tag);
        }
    }
    tags
}

/// Tags listed in a rules file: one per line, trimmed, `#` starts a comment
/// line, blank lines skipped. A missing or unreadable file lists none.
pub fn read_rules_file(path: &Path) -> Vec<String> {
    let Ok(meta) = std::fs::metadata(path) else {
        return Vec::new();
    };
    if !meta.is_file() || meta.len() > MAX_RULES_FILE_BYTES {
        return Vec::new();
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    parse_rules_text(&text)
}

/// [`read_rules_file`] on text already read.
pub fn parse_rules_text(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter(|line| line.len() <= MAX_TAG_BYTES)
        .map(str::to_string)
        .take(MAX_FILE_TAGS)
        .collect()
}

/// A git checkout found above a directory.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Checkout {
    /// The working tree's root (the directory holding `.git`).
    toplevel: PathBuf,
    /// The repository's common git directory (for a linked worktree, the
    /// main checkout's `.git`).
    common_dir: Option<PathBuf>,
}

fn find_checkout(cwd: &Path) -> Option<Checkout> {
    for dir in cwd.ancestors() {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return Some(Checkout {
                toplevel: dir.to_path_buf(),
                common_dir: Some(dot_git),
            });
        }
        if dot_git.is_file() {
            return Some(Checkout {
                toplevel: dir.to_path_buf(),
                common_dir: linked_common_dir(dir, &dot_git),
            });
        }
    }
    None
}

/// The common git directory of a linked worktree or submodule, from its
/// `.git` file (`gitdir: <path>`) and that directory's `commondir` file.
fn linked_common_dir(toplevel: &Path, dot_git_file: &Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(dot_git_file).ok()?;
    let gitdir = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("gitdir:"))?
        .trim();
    let gitdir = toplevel.join(gitdir);
    match std::fs::read_to_string(gitdir.join("commondir")) {
        Ok(common) => Some(gitdir.join(common.trim())),
        Err(_) => Some(gitdir),
    }
}

/// The repository's name: the `origin` remote's last path segment (without
/// `.git`), else the main checkout's folder name, else the working tree's
/// folder name; outside git, the folder name of `cwd`.
fn repo_name(cwd: &Path, checkout: Option<&Checkout>) -> Option<String> {
    let Some(checkout) = checkout else {
        return folder_name(cwd);
    };
    if let Some(common) = &checkout.common_dir {
        if let Some(name) = origin_name(&common.join("config")) {
            return Some(name);
        }
        let common = normalize(common);
        if common.file_name().is_some_and(|name| name == ".git")
            && let Some(name) = common.parent().and_then(folder_name)
        {
            return Some(name);
        }
    }
    folder_name(&checkout.toplevel)
}

/// `a/b/../c` as `a/c`, without touching the filesystem.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

fn folder_name(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

/// The name in `[remote "origin"] url = ...` of a git config file.
fn origin_name(config: &Path) -> Option<String> {
    let text = std::fs::read_to_string(config).ok()?;
    let mut in_origin = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_origin = line == "[remote \"origin\"]";
            continue;
        }
        if !in_origin {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != "url" {
            continue;
        }
        let url = value.trim().trim_end_matches('/');
        let last = url.rsplit(['/', ':', '\\']).next()?;
        let name = last.strip_suffix(".git").unwrap_or(last).trim();
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }
    None
}

/// One memory a tag resolved to, as the `recall` handle payload gives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleNode {
    pub id: String,
    pub node_type: String,
    pub content: String,
    /// RFC 3339, when the recall payload carries it.
    pub created_at: Option<String>,
}

/// The memories an exact-tag `recall` returned. Only a `kind: "tag"` answer
/// counts: an id or id-prefix match, an ambiguity or a miss gives nothing,
/// because the hook asked for a tag.
pub fn nodes_from_recall(value: &Value) -> Vec<RuleNode> {
    if value.get("kind").and_then(Value::as_str) != Some("tag") {
        return Vec::new();
    }
    value
        .get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|node| {
            Some(RuleNode {
                id: node.get("id")?.as_str()?.to_string(),
                node_type: node
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                content: node
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                created_at: node
                    .get("createdAt")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect()
}

/// Limits on the rendered context.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_items: usize,
    pub max_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_items: DEFAULT_MAX_ITEMS,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// The context text for the resolved tags, or `None` when no decision or
/// correction memory carries any of them.
///
/// `resolved` pairs each tag with the memories it resolved to. A memory
/// under several tags is listed once, with each tag. Newest first: by
/// `createdAt` when the payload carries it, then by id (Strata ids are
/// `mem-<log sequence>`, so a later id is a later write).
pub fn render_context(
    resolved: &[(String, Vec<RuleNode>)],
    source: &str,
    limits: Limits,
) -> Option<String> {
    let mut by_id: BTreeMap<String, (RuleNode, Vec<String>)> = BTreeMap::new();
    for (tag, nodes) in resolved {
        for node in nodes {
            if !RULE_TYPES.contains(&node.node_type.as_str()) {
                continue;
            }
            let entry = by_id
                .entry(node.id.clone())
                .or_insert_with(|| (node.clone(), Vec::new()));
            if !entry.1.contains(tag) {
                entry.1.push(tag.clone());
            }
        }
    }
    if by_id.is_empty() {
        return None;
    }
    let mut rules: Vec<(RuleNode, Vec<String>)> = by_id.into_values().collect();
    rules.sort_by(|(a, _), (b, _)| {
        let when = |node: &RuleNode| {
            node.created_at
                .as_deref()
                .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
        };
        when(b).cmp(&when(a)).then_with(|| b.id.cmp(&a.id))
    });

    let tags: Vec<&str> = resolved.iter().map(|(tag, _)| tag.as_str()).collect();
    let total = rules.len();
    let header = format!(
        "Vestige rules for this session (source: {source}). These are the decision and \
         correction memories recorded under the exact tags [{}], newest first. They were \
         resolved by exact tag through recall(handle=...), not by search or similarity. \
         Each line starts with its memory id; recall(handle='<id>') returns the full memory \
         and receipt(action='get', receipt_id='<id>') its receipt on the hash-chained log.{}",
        tags.join(", "),
        if source == "compact" {
            " The conversation was just compacted: these rules are restated from the log, \
             whatever the summary kept."
        } else {
            ""
        }
    );
    // Room kept for the closing line that names what was left out.
    let reserve = 240;
    let mut text = header;
    let mut shown = 0usize;
    for (node, matched) in &rules {
        if shown >= limits.max_items {
            break;
        }
        let line = rule_line(node, matched);
        if text.len() + 1 + line.len() + reserve > limits.max_bytes {
            if shown == 0 {
                // Always show the newest rule, cut to what fits.
                let room = limits.max_bytes.saturating_sub(text.len() + 1 + reserve);
                let cut = cut_to_bytes(&line, room);
                if !cut.is_empty() {
                    text.push('\n');
                    text.push_str(&cut);
                    shown = 1;
                }
            }
            break;
        }
        text.push('\n');
        text.push_str(&line);
        shown += 1;
    }
    if shown == 0 {
        return None;
    }
    if shown < total {
        text.push_str(&format!(
            "\n{} more under these tags left out (limit {} entries, {} bytes); \
             recall(handle='<tag>') lists every one.",
            total - shown,
            limits.max_items,
            limits.max_bytes
        ));
    }
    Some(text)
}

fn rule_line(node: &RuleNode, matched: &[String]) -> String {
    let content: String = node
        .content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let content = if content.chars().count() > MAX_LINE_CHARS {
        let mut cut: String = content.chars().take(MAX_LINE_CHARS).collect();
        cut.push_str(" [...]");
        cut
    } else {
        content
    };
    let date = node
        .created_at
        .as_deref()
        .and_then(|at| at.get(..10))
        .map(|day| format!(", {day}"))
        .unwrap_or_default();
    format!(
        "- {} [{}, tag {}{}] {}",
        node.id,
        node.node_type,
        matched.join(" + "),
        date,
        content
    )
}

/// The longest prefix of `text` within `max` bytes, on a char boundary.
fn cut_to_bytes(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// The hook's stdout for `context`.
pub fn hook_output(context: &str) -> Value {
    json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    })
}

/// Rules files one repository may carry. The project tool writes the same
/// fenced block into each of them.
pub const RULES_FILENAMES: [&str; 4] = ["CLAUDE.md", "AGENTS.md", "GEMINI.md", ".cursorrules"];

/// Flags for a missing path or a command whose recorded runs failed.
/// `failed` pairs a command line with the memory id that recorded the failure.
/// A path that exists, and a command with no failure record, stay quiet.
pub fn rules_file_flags(cwd: &Path, failed: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for name in RULES_FILENAMES {
        let path = cwd.join(name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        out.extend(flags_in(name, cwd, &text, failed));
    }
    out.sort();
    out.dedup();
    out
}

fn flags_in(name: &str, cwd: &Path, text: &str, failed: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    let mut fenced = false;
    for (number, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((_, receipt)) = failed.iter().find(|(command, _)| command == trimmed) {
                out.push(format!(
                    "{name}:{} command `{trimmed}` failed, receipt `{receipt}`",
                    number + 1
                ));
            }
            continue;
        }
        for token in backtick_tokens(line) {
            if !path_token(&token) {
                continue;
            }
            let candidate = if Path::new(&token).is_absolute() {
                PathBuf::from(&token)
            } else {
                cwd.join(&token)
            };
            if !candidate.exists() {
                out.push(format!(
                    "{name}:{} path `{token}` is not in the tree",
                    number + 1
                ));
            }
        }
    }
    out
}

fn backtick_tokens(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('`') else {
            break;
        };
        out.push(after[..end].to_string());
        rest = &after[end + 1..];
    }
    out
}

fn path_token(token: &str) -> bool {
    if token.is_empty()
        || token.len() > 240
        || token.contains([' ', '\t', ':', '*', '?', '`'])
        || token.starts_with("http")
        || token.starts_with("vst")
        || token.starts_with('$')
    {
        return false;
    }
    token.contains('/') || token.contains('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, node_type: &str, content: &str, created_at: Option<&str>) -> RuleNode {
        RuleNode {
            id: id.to_string(),
            node_type: node_type.to_string(),
            content: content.to_string(),
            created_at: created_at.map(str::to_string),
        }
    }

    #[test]
    fn payload_parses_every_documented_source_and_refuses_other_events() {
        for source in ["startup", "resume", "clear", "compact", "fork"] {
            let raw = format!(
                r#"{{"session_id":"abc","transcript_path":"/t.jsonl","cwd":"/w/tidetable","hook_event_name":"SessionStart","source":"{source}","model":"m"}}"#
            );
            let input = parse_input(&raw).expect("payload");
            assert_eq!(input.source, source);
            assert_eq!(input.cwd.as_deref(), Some(Path::new("/w/tidetable")));
            assert_eq!(input.session_id.as_deref(), Some("abc"));
        }
        assert_eq!(parse_input("{}").unwrap().source, "startup");
        assert!(parse_input("not json").is_none());
        assert!(parse_input("[1,2]").is_none());
        assert!(parse_input(r#"{"hook_event_name":"Stop","cwd":"/w"}"#).is_none());
    }

    #[test]
    fn rules_file_is_one_exact_tag_per_line() {
        let tags = parse_rules_text("# comment\n\n  deploy  \r\ncodebase:tidetable\nTwo Words\n");
        assert_eq!(tags, vec!["deploy", "codebase:tidetable", "Two Words"]);
        let long = "x".repeat(MAX_TAG_BYTES + 1);
        assert!(parse_rules_text(&long).is_empty());
        let many: String = (0..100).map(|i| format!("t{i}\n")).collect();
        assert_eq!(parse_rules_text(&many).len(), MAX_FILE_TAGS);
    }

    #[test]
    fn repo_name_comes_from_origin_then_the_main_checkout_then_the_folder() {
        let root = tempfile::tempdir().unwrap();
        // No git: the folder name. (Called directly: the temp root itself
        // may sit inside some checkout.)
        let plain = root.path().join("Plain-Dir");
        std::fs::create_dir_all(&plain).unwrap();
        assert_eq!(repo_name(&plain, None).as_deref(), Some("Plain-Dir"));

        // A checkout without a remote: its folder name, from a subfolder too.
        let repo = root.path().join("tidetable");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        assert_eq!(
            rule_tags(&repo.join("src"), true),
            vec!["tidetable", "pinned"]
        );
        // A capitalised name is tried as written and in lowercase: two exact
        // handles, never a case-folded match.
        let upper = root.path().join("TideTable");
        std::fs::create_dir_all(upper.join(".git")).unwrap();
        assert_eq!(
            rule_tags(&upper, true),
            vec!["TideTable", "tidetable", "pinned"]
        );

        // The origin remote's name wins over the folder name.
        std::fs::write(
            repo.join(".git/config"),
            "[core]\n\tbare = false\n[remote \"upstream\"]\n\turl = https://x/y/other.git\n[remote \"origin\"]\n\turl = git@github.com:sam/vestige.git\n",
        )
        .unwrap();
        assert_eq!(rule_tags(&repo, true), vec!["vestige", "pinned"]);

        // A linked worktree in another folder resolves to the same name.
        let main = root.path().join("main-checkout");
        let wt_git = main.join(".git/worktrees/wt");
        std::fs::create_dir_all(&wt_git).unwrap();
        std::fs::write(wt_git.join("commondir"), "../..\n").unwrap();
        let wt = root.path().join("vestige-wt-feature");
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", wt_git.display())).unwrap();
        assert_eq!(rule_tags(&wt, true), vec!["main-checkout", "pinned"]);
        std::fs::write(
            main.join(".git/config"),
            "[remote \"origin\"]\n\turl = https://github.com/sam/tidetable\n",
        )
        .unwrap();
        assert_eq!(rule_tags(&wt, true), vec!["tidetable", "pinned"]);

        // The rules file adds its tags, in order, without duplicates.
        std::fs::write(wt.join(RULES_FILE), "release\npinned\ntidetable\nci\n").unwrap();
        assert_eq!(
            rule_tags(&wt, true),
            vec!["tidetable", "pinned", "release", "ci"]
        );
    }

    #[test]
    fn the_repo_name_tag_is_opt_in() {
        let root = tempfile::tempdir().unwrap();
        let repo = root.path().join("tidetable");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(
            repo.join(".git/config"),
            "[remote \"origin\"]\n\turl = git@github.com:sam/vestige.git\n",
        )
        .unwrap();
        // By default a session gets `pinned` only, inside a checkout or not.
        assert_eq!(rule_tags(&repo, false), vec!["pinned"]);
        assert_eq!(rule_tags(&repo.join("src"), false), vec!["pinned"]);
        let plain = root.path().join("Plain-Dir");
        std::fs::create_dir_all(&plain).unwrap();
        assert_eq!(rule_tags(&plain, false), vec!["pinned"]);
        // The rules file at the repository root still counts from a subfolder,
        // and naming the repository there opts its tag in by hand.
        std::fs::write(repo.join(RULES_FILE), "release\nvestige\n").unwrap();
        assert_eq!(
            rule_tags(&repo.join("src"), false),
            vec!["pinned", "release", "vestige"]
        );
    }

    #[test]
    fn only_an_exact_tag_answer_yields_nodes() {
        let tag = json!({"kind": "tag", "nodes": [
            {"id": "mem-1", "type": "decision", "content": "a", "tags": ["pinned"]}
        ]});
        assert_eq!(nodes_from_recall(&tag).len(), 1);
        let by_id = json!({"kind": "memory", "nodes": [
            {"id": "mem-1", "type": "decision", "content": "a", "tags": []}
        ]});
        assert!(nodes_from_recall(&by_id).is_empty());
        assert!(
            nodes_from_recall(&json!({"error": "handle_required", "candidates": []})).is_empty()
        );
    }

    #[test]
    fn context_lists_rules_newest_first_once_each_with_ids() {
        let resolved = vec![
            (
                "tidetable".to_string(),
                vec![
                    node("mem-0000000000000003", "decision", "use metric units", None),
                    node("mem-0000000000000004", "fact", "a fact is not a rule", None),
                ],
            ),
            (
                "pinned".to_string(),
                vec![
                    node("mem-0000000000000003", "decision", "use metric units", None),
                    node(
                        "mem-0000000000000009",
                        "correction",
                        "never  push\nto main",
                        None,
                    ),
                ],
            ),
        ];
        let text = render_context(&resolved, "compact", Limits::default()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].contains("source: compact"), "{text}");
        assert!(lines[0].contains("just compacted"), "{text}");
        assert!(lines[0].contains("[tidetable, pinned]"), "{text}");
        assert!(lines[0].contains("not by search or similarity"), "{text}");
        assert_eq!(
            &lines[1..],
            [
                "- mem-0000000000000009 [correction, tag pinned] never push to main",
                "- mem-0000000000000003 [decision, tag tidetable + pinned] use metric units",
            ]
        );
        assert!(!text.contains("a fact is not a rule"));
    }

    #[test]
    fn created_at_orders_before_ids_and_shows_the_day() {
        let resolved = vec![(
            "pinned".to_string(),
            vec![
                node(
                    "mem-2",
                    "decision",
                    "older",
                    Some("2026-01-02T00:00:00+00:00"),
                ),
                node(
                    "mem-1",
                    "decision",
                    "newer",
                    Some("2026-03-04T00:00:00+00:00"),
                ),
            ],
        )];
        let text = render_context(&resolved, "startup", Limits::default()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[1], "- mem-1 [decision, tag pinned, 2026-03-04] newer");
        assert_eq!(lines[2], "- mem-2 [decision, tag pinned, 2026-01-02] older");
        assert!(!lines[0].contains("just compacted"), "{text}");
    }

    #[test]
    fn context_is_capped_by_entries_and_bytes() {
        let nodes: Vec<RuleNode> = (0..20)
            .map(|i| {
                node(
                    &format!("mem-{i:016x}"),
                    "decision",
                    &"rule ".repeat(40),
                    None,
                )
            })
            .collect();
        let resolved = vec![("pinned".to_string(), nodes)];
        let text = render_context(&resolved, "startup", Limits::default()).unwrap();
        assert!(text.len() <= DEFAULT_MAX_BYTES, "{}", text.len());
        let shown = text
            .lines()
            .filter(|line| line.starts_with("- mem-"))
            .count();
        assert_eq!(shown, DEFAULT_MAX_ITEMS);
        assert!(text.contains("12 more under these tags left out"), "{text}");

        let tight = Limits {
            max_items: 8,
            max_bytes: 1200,
        };
        let text = render_context(&resolved, "startup", tight).unwrap();
        assert!(text.len() <= 1200, "{}", text.len());
        assert!(
            text.lines()
                .any(|line| line.starts_with("- mem-0000000000000013"))
        );

        let huge = vec![(
            "pinned".to_string(),
            vec![node("mem-1", "decision", &"é".repeat(5000), None)],
        )];
        let text = render_context(&huge, "startup", tight).unwrap();
        assert!(text.len() <= 1200, "{}", text.len());
        assert!(text.contains("- mem-1"));
    }

    #[test]
    fn nothing_to_say_is_none() {
        assert!(render_context(&[], "startup", Limits::default()).is_none());
        let facts = vec![(
            "pinned".to_string(),
            vec![node("mem-1", "fact", "not a rule", None)],
        )];
        assert!(render_context(&facts, "startup", Limits::default()).is_none());
    }

    #[test]
    fn output_is_the_documented_session_start_shape() {
        let out = hook_output("ctx");
        assert_eq!(out["hookSpecificOutput"]["hookEventName"], "SessionStart");
        assert_eq!(out["hookSpecificOutput"]["additionalContext"], "ctx");
        assert_eq!(out.as_object().unwrap().len(), 1);
    }

    #[test]
    fn a_missing_path_and_a_failed_command_are_flagged_and_clean_lines_are_quiet() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "ok\n").unwrap();
        std::fs::write(
            dir.path().join("CLAUDE.md"),
            "See `README.md` and `src/gone.rs`.\n\n```\ncargo test -p missing\ncargo test -p vestige-mcp\n```\n",
        )
        .unwrap();
        let flags = rules_file_flags(
            dir.path(),
            &[(
                "cargo test -p missing".into(),
                "mem-00000000000000aa".into(),
            )],
        );
        let text = flags.join("\n");
        assert!(text.contains("src/gone.rs"), "{text}");
        assert!(text.contains("cargo test -p missing"), "{text}");
        assert!(text.contains("mem-00000000000000aa"), "{text}");
        assert!(!text.contains("README.md"), "{text}");
        assert!(!text.contains("vestige-mcp"), "{text}");
    }
}
