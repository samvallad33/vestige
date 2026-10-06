//! # Commit records
//!
//! Turn `git log -p` output into memory records so Backfill can join a failure
//! to the change that caused it, not only to another description of it. Each
//! commit becomes one record tagged `git-commit`; its files, modules, hunk
//! spans (new-side line ranges, the anchors blame needs) and import edges ride
//! in the content alongside the hunk-header symbols, so the query-time entity
//! extractor picks them up as join keys with no schema change.

use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, BTreeSet};

/// Tag marking a commit record. Deliberately not identifier-shaped, so it
/// never becomes a causal join key itself.
pub const COMMIT_TAG: &str = "git-commit";
/// `source_system` key for the idempotent source upsert.
pub const SOURCE_SYSTEM: &str = "git";

/// One `@@ -a[,b] +c[,d] @@` hunk span on the new side — where the changed
/// lines live in the post-commit file, so blame can anchor onto it.
#[derive(Debug, Clone, PartialEq)]
pub struct HunkSpan {
    /// File the hunk belongs to (the b/ side name, like `files`).
    pub file: String,
    /// New-side start line (`+c`).
    pub start: u32,
    /// New-side line count (`,d`); an omitted count means a single line.
    pub len: u32,
    /// Symbol from the hunk-header trailing context, when one parses.
    pub symbol: Option<String>,
}

/// One parsed commit.
#[derive(Debug, Clone, PartialEq)]
pub struct GitCommit {
    pub sha: String,
    pub time: DateTime<Utc>,
    pub subject: String,
    pub files: Vec<String>,
    /// Files the diff contained beyond [`MAX_FILES`] (recorded as a count, not
    /// names, so the content can say "+N more" instead of truncating silently).
    pub extra_files: usize,
    /// Symbols from diff hunk headers, path-qualified (`<file>/<symbol>`) so
    /// they pass the entity shape test; a bare lowercase `fn_name` would not.
    pub symbols: Vec<String>,
    /// Hunk spans (new-side line ranges) per file, bounded by [`MAX_HUNKS`].
    pub hunks: Vec<HunkSpan>,
    /// Hunks beyond [`MAX_HUNKS`] (recorded as a count, not spans, so the
    /// content can say "+N more" instead of truncating silently).
    pub extra_hunks: usize,
    /// Import edges harvested from changed lines, sorted by (file, target):
    /// (file in this commit, target path, resolved). A resolved target is the
    /// repo-relative path that exactly matched this commit's files or their
    /// module dirs; an unresolved target keeps the module path as written —
    /// never a guess.
    pub imports: Vec<(String, String, bool)>,
    /// Parent SHAs, in `git log %P` order. Empty when the log format did not
    /// carry parents.
    pub parents: Vec<String>,
    /// The 40-hex SHA from a `This reverts commit <sha>.` line git revert
    /// writes. Anything else in the message is not a revert.
    pub reverts: Option<String>,
    /// The 40-hex SHA from a `(cherry picked from commit <sha>)` line.
    /// Anything else in the message is not a cherry-pick.
    pub cherry_picked_from: Option<String>,
    /// `Fixes: <sha>` trailers, lowercase, exactly 40 hex digits, as written.
    /// A short prefix is not a trailer and is never expanded.
    pub fixes: Vec<String>,
    /// Lockfile version changes in this commit's diff. A package is included
    /// only when it loses exactly one version and gains exactly one other.
    pub lock_bumps: Vec<LockBump>,
}

/// One package whose lockfile entry moved from `old_version` to `new_version`
/// in a single commit. Parsed from the lockfile diff, not from the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockBump {
    /// `cargo`, `uv`, `poetry`, `npm`, or `go`.
    pub ecosystem: String,
    pub package: String,
    pub old_version: String,
    pub new_version: String,
}

/// Dangling anchor for one resolved package version.
/// `pkg:<ecosystem>:<name>@<version>`. The name may itself contain `@`
/// (`@scope/pkg`); the version is the span after the last `@`.
pub fn package_anchor_id(ecosystem: &str, name: &str, version: &str) -> String {
    format!("pkg:{ecosystem}:{name}@{version}")
}

/// `ecosystem`, package name, version from a registry or vendor path.
///
/// `.../.cargo/registry/.../tokio-1.39.0/src/lib.rs` is
/// `(cargo, tokio, 1.39.0)`. A source path with no registry, vendor,
/// `node_modules`, or `pkg/mod` marker is not a package path.
pub fn registry_package(path: &str) -> Option<(String, String, String)> {
    let path = path.trim().replace('\\', "/");
    let parts: Vec<&str> = path.split('/').filter(|part| !part.is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }
    let ecosystem = registry_ecosystem(&parts)?;
    let dirs = &parts[..parts.len() - 1];
    if ecosystem == "go" {
        return go_module_from_path(dirs).map(|(name, version)| ("go".into(), name, version));
    }
    for component in dirs.iter().rev() {
        if let Some((name, version)) = split_name_version(component) {
            return Some((ecosystem.into(), name, version));
        }
    }
    None
}

fn registry_ecosystem(parts: &[&str]) -> Option<&'static str> {
    if parts.windows(2).any(|pair| pair == ["pkg", "mod"]) {
        return Some("go");
    }
    if parts.contains(&"node_modules") {
        return Some("npm");
    }
    let joined = parts.join("/");
    if parts.contains(&"vendor")
        || joined.contains(".cargo/")
        || joined.contains("/registry/src/")
        || joined.contains(".cargo/registry")
    {
        return Some("cargo");
    }
    None
}

/// `github.com/stretchr/testify@v1.8.0` under `pkg/mod`.
fn go_module_from_path(dirs: &[&str]) -> Option<(String, String)> {
    let mod_at = dirs.iter().position(|part| *part == "mod")?;
    let rest = &dirs[mod_at + 1..];
    let at = rest.iter().position(|part| part.contains("@v"))?;
    let (leaf, version) = split_go_version(rest[at])?;
    let mut name: Vec<&str> = rest[..=at].to_vec();
    name[at] = leaf;
    let name = name.join("/");
    if name.is_empty() {
        return None;
    }
    Some((name, version))
}

fn split_go_version(component: &str) -> Option<(&str, String)> {
    let (name, version) = component.rsplit_once("@v")?;
    if name.is_empty() || !is_semver(version) {
        return None;
    }
    Some((name, version.to_string()))
}

/// `{name}-{semver}`, with `name` allowed to contain hyphens.
/// `tokio-util-0.7.10` → (`tokio-util`, `0.7.10`).
/// `tokio-1.39.0-alpha.1` → (`tokio`, `1.39.0-alpha.1`).
fn split_name_version(component: &str) -> Option<(String, String)> {
    let mut start = component.len();
    while let Some(rel) = component[..start].rfind('-') {
        let version = &component[rel + 1..];
        if rel > 0 && is_semver(version) {
            return Some((component[..rel].to_string(), version.to_string()));
        }
        if rel == 0 {
            break;
        }
        start = rel;
    }
    None
}

fn is_semver(version: &str) -> bool {
    let core = if let Some((core, pre)) = version.split_once('-') {
        if pre.is_empty()
            || !pre
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
        {
            return false;
        }
        core
    } else {
        version
    };
    let mut parts = core.split('.');
    let Some(major) = parts.next() else {
        return false;
    };
    let Some(minor) = parts.next() else {
        return false;
    };
    let Some(patch) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    numeric_id(major) && numeric_id(minor) && numeric_id(patch)
}

fn numeric_id(part: &str) -> bool {
    !part.is_empty()
        && part.chars().all(|c| c.is_ascii_digit())
        && (part == "0" || !part.starts_with('0'))
}

fn lock_kind(path: &str) -> Option<&'static str> {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    match name {
        "Cargo.lock" => Some("cargo"),
        "uv.lock" => Some("uv"),
        "poetry.lock" => Some("poetry"),
        "package-lock.json" => Some("npm"),
        "go.sum" => Some("go"),
        _ => None,
    }
}

/// Version moves in a `git log -p` diff. A package counts only when the diff
/// removes exactly one of its versions and adds exactly one different version.
pub fn lock_bumps_from_diff(diff: &str) -> Vec<LockBump> {
    let mut bumps = Vec::new();
    let mut ecosystem: Option<&str> = None;
    let mut removed: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut added: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut toml_name: Option<String> = None;
    let mut npm_name: Option<String> = None;

    let flush = |ecosystem: Option<&str>,
                 removed: &mut BTreeMap<String, BTreeSet<String>>,
                 added: &mut BTreeMap<String, BTreeSet<String>>,
                 bumps: &mut Vec<LockBump>| {
        let Some(ecosystem) = ecosystem else {
            removed.clear();
            added.clear();
            return;
        };
        let mut names: BTreeSet<String> = BTreeSet::new();
        names.extend(removed.keys().cloned());
        names.extend(added.keys().cloned());
        for name in names {
            let old = removed.get(&name).filter(|versions| versions.len() == 1);
            let new = added.get(&name).filter(|versions| versions.len() == 1);
            if let (Some(old), Some(new)) = (old, new) {
                let old_version = old.iter().next().map(String::as_str).unwrap_or("");
                let new_version = new.iter().next().map(String::as_str).unwrap_or("");
                if !old_version.is_empty()
                    && old_version != new_version
                    && acceptable_package(&name)
                    && is_semver(old_version)
                    && is_semver(new_version)
                {
                    bumps.push(LockBump {
                        ecosystem: ecosystem.to_string(),
                        package: name.to_string(),
                        old_version: old_version.to_string(),
                        new_version: new_version.to_string(),
                    });
                }
            }
        }
        removed.clear();
        added.clear();
    };

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("diff --git a/") {
            flush(ecosystem, &mut removed, &mut added, &mut bumps);
            toml_name = None;
            npm_name = None;
            ecosystem = rest
                .split_once(" b/")
                .map(|(_, path)| path.trim().trim_matches('"'))
                .and_then(lock_kind);
            continue;
        }
        let Some(ecosystem_now) = ecosystem else {
            continue;
        };
        let Some((marker, payload)) = diff_payload(line) else {
            continue;
        };
        if ecosystem_now == "go" {
            if marker == ' ' {
                continue;
            }
            if let Some((module, version)) = go_sum_line(payload) {
                let slot = if marker == '+' {
                    &mut added
                } else {
                    &mut removed
                };
                slot.entry(module).or_default().insert(version);
            }
            continue;
        }
        if ecosystem_now == "npm" {
            if let Some(name) = npm_package_key(payload) {
                npm_name = Some(name);
            }
            if marker != ' '
                && let Some(version) = json_string_field(payload, "version")
                && let Some(name) = npm_name.clone()
            {
                let slot = if marker == '+' {
                    &mut added
                } else {
                    &mut removed
                };
                slot.entry(name).or_default().insert(version);
            }
            continue;
        }
        if let Some(name) = toml_string_field(payload, "name") {
            toml_name = Some(name);
        }
        if marker != ' '
            && let Some(version) = toml_string_field(payload, "version")
            && let Some(name) = toml_name.clone()
        {
            let slot = if marker == '+' {
                &mut added
            } else {
                &mut removed
            };
            slot.entry(name).or_default().insert(version);
        }
    }
    flush(ecosystem, &mut removed, &mut added, &mut bumps);
    bumps
}

fn acceptable_package(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 200
        && !name.chars().any(|c| c.is_whitespace() || c.is_control())
        && !name.contains("..")
}

fn diff_payload(line: &str) -> Option<(char, &str)> {
    if line.is_empty()
        || line.starts_with("diff ")
        || line.starts_with("index ")
        || line.starts_with("@@")
        || line.starts_with("+++")
        || line.starts_with("---")
        || line.starts_with("new file")
        || line.starts_with("deleted file")
        || line.starts_with("similarity ")
        || line.starts_with("rename ")
        || line.starts_with("old mode")
        || line.starts_with("new mode")
        || line.starts_with("Binary ")
    {
        return None;
    }
    let marker = line.as_bytes()[0] as char;
    if marker == '+' || marker == '-' || marker == ' ' {
        Some((marker, &line[1..]))
    } else {
        None
    }
}

fn toml_string_field(line: &str, key: &str) -> Option<String> {
    let line = line.trim();
    let rest = line.strip_prefix(key)?.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    quoted(rest)
}

fn json_string_field(line: &str, key: &str) -> Option<String> {
    let line = line.trim().trim_end_matches(',').trim();
    let needle = format!("\"{key}\"");
    let rest = line.strip_prefix(&needle)?.trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    quoted(rest)
}

fn quoted(text: &str) -> Option<String> {
    let rest = text.trim_start().strip_prefix('"')?;
    let (value, _) = rest.split_once('"')?;
    if value.is_empty() || value.chars().any(|c| c.is_control() || c == '\\') {
        return None;
    }
    Some(value.to_string())
}

fn npm_package_key(line: &str) -> Option<String> {
    let line = line
        .trim()
        .trim_end_matches('{')
        .trim()
        .trim_end_matches(',')
        .trim();
    let line = line.strip_suffix(':')?.trim();
    let key = quoted(line)?;
    if !key.starts_with("node_modules/") && !key.contains("/node_modules/") {
        return None;
    }
    let name = key.rsplit("/node_modules/").next().unwrap_or(key.as_str());
    let name = name.trim_start_matches("node_modules/");
    acceptable_package(name).then(|| name.to_string())
}

fn go_sum_line(payload: &str) -> Option<(String, String)> {
    let mut parts = payload.split_whitespace();
    let module = parts.next()?;
    let raw = parts.next()?;
    let raw = raw.strip_suffix("/go.mod").unwrap_or(raw);
    let version = raw.strip_prefix('v')?;
    if !acceptable_package(module) || !is_semver(version) {
        return None;
    }
    Some((module.to_string(), version.to_string()))
}

/// `git log --pretty=format:` that carries parents and the body (where the
/// revert trailer lives) and ends the header before the diff.
pub const GIT_LOG_PRETTY: &str = "%x1e%H%x1f%aI%x1f%P%x1f%s%x1f%b%x1d";

/// The exact trailer line `git revert` writes. The SHA is 40 hex digits and
/// the line ends with a period. No other wording counts.
pub fn revert_target(body: &str) -> Option<String> {
    for line in body.lines() {
        let line = line.trim_end_matches('\r').trim();
        let Some(rest) = line.strip_prefix("This reverts commit ") else {
            continue;
        };
        let Some(sha) = rest.strip_suffix('.') else {
            continue;
        };
        if is_full_sha(sha) {
            return Some(sha.to_ascii_lowercase());
        }
    }
    None
}

/// The exact trailer line `git cherry-pick -x` writes. The SHA is 40 hex
/// digits and the parentheses are part of the line. No other wording counts.
pub fn cherry_picked_from(body: &str) -> Option<String> {
    for line in body.lines() {
        let line = line.trim_end_matches('\r').trim();
        let Some(rest) = line.strip_prefix("(cherry picked from commit ") else {
            continue;
        };
        let Some(sha) = rest.strip_suffix(')') else {
            continue;
        };
        if is_full_sha(sha) {
            return Some(sha.to_ascii_lowercase());
        }
    }
    None
}

/// `Fixes: <sha>` when the token is a full 40-hex commit id and the only
/// thing on the line after the prefix. A 7–39 hex prefix, `fixes:`, and a
/// SHA inside a sentence are not trailers. Nothing here asks git to expand
/// a prefix.
pub fn fixes_targets(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in body.lines() {
        let line = line.trim_end_matches('\r').trim();
        let Some(rest) = line.strip_prefix("Fixes:") else {
            continue;
        };
        let rest = rest.trim();
        let mut tokens = rest.split_whitespace();
        let Some(sha) = tokens.next() else {
            continue;
        };
        if tokens.next().is_some() {
            continue;
        }
        if is_full_sha(sha) {
            let sha = sha.to_ascii_lowercase();
            if !out.contains(&sha) {
                out.push(sha);
            }
        }
    }
    out
}

fn is_full_sha(s: &str) -> bool {
    s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())
}

fn parent_shas(field: &str) -> Vec<String> {
    field
        .split_whitespace()
        .filter(|sha| is_full_sha(sha))
        .map(|sha| sha.to_ascii_lowercase())
        .collect()
}

/// `(header, diff, new_format)`. New format ends the header at `\x1d` so the
/// body can contain newlines. The older format is one header line, then the diff.
fn split_log_chunk(chunk: &str) -> (&str, &str, bool) {
    if let Some((header, diff)) = chunk.split_once('\u{1d}') {
        (header, diff, true)
    } else if let Some((head, diff)) = chunk.split_once('\n') {
        (head, diff, false)
    } else {
        (chunk, "", false)
    }
}

const RECORD_SEP: char = '\u{1e}';
const UNIT_SEP: char = '\u{1f}';
const MAX_FILES: usize = 50;
const MAX_SYMBOLS: usize = 40;
/// Hunk spans kept per commit; overflow is counted in `extra_hunks`.
const MAX_HUNKS: usize = 200;
/// Upper bound for the comma-joined span list on the `hunks:` content line.
const MAX_HUNK_LINE: usize = 400;
/// Import edges kept per commit.
const MAX_IMPORTS: usize = 40;

/// Syntax family of a captured import statement — decides how the written
/// target maps onto candidate repo-relative paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ImportKind {
    /// `use a::b::Item;` — `::`-separated; a `crate` root maps to `src/`.
    Rust,
    /// `import x.y.z` / `from x.y import z` — `.`-separated module path.
    Dotted,
    /// `#include "p"` / `#include <p>` — path used verbatim.
    Include,
}

/// Parse `git log -p` output produced with [`GIT_LOG_PRETTY`].
///
/// Also accepts the older three-field header (`sha`, author time, subject)
/// so fixtures written before parents and the body were recorded still parse.
/// Files come from `diff --git a/X b/Y` lines (the b/ side wins, so renames
/// land under the new name); symbols from hunk-header trailing context.
pub fn parse_git_log(raw: &str) -> Vec<GitCommit> {
    let mut out: Vec<GitCommit> = Vec::new();
    for chunk in raw.split(RECORD_SEP) {
        if chunk.is_empty() {
            continue;
        }
        let (head, diff, new_format) = split_log_chunk(chunk);
        // Body is the remainder, so a unit separator inside a message cannot
        // shift the subject onto the next field.
        let mut fields = head.splitn(5, UNIT_SEP);
        let sha = fields.next().unwrap_or("").trim().to_string();
        // A full 40-hex sha or nothing: subjects are not sanitized for \x1e/\x1f,
        // so a control char in a message can fabricate a phantom record header.
        if !is_full_sha(&sha) {
            continue;
        }
        let time = fields
            .next()
            .unwrap_or("")
            .trim()
            .parse::<DateTime<Utc>>()
            .unwrap_or_default();
        let (parents, subject, reverts, cherry_picked_from, fixes) = if new_format {
            let parents = parent_shas(fields.next().unwrap_or(""));
            let subject = fields.next().unwrap_or("").trim().to_string();
            let message = fields.next().unwrap_or("");
            (
                parents,
                subject,
                revert_target(message),
                cherry_picked_from(message),
                fixes_targets(message),
            )
        } else {
            let subject = fields.next().unwrap_or("").trim().to_string();
            (Vec::new(), subject, None, None, Vec::new())
        };
        let body = diff;

        let mut files: Vec<String> = Vec::new();
        let mut extra_files = 0usize;
        // once the cap is hit, further hunks belong to files we never recorded;
        // attaching their symbols to files.last() would fabricate join keys
        let mut files_capped = false;
        let mut symbols: BTreeSet<String> = BTreeSet::new();
        let mut hunks: Vec<HunkSpan> = Vec::new();
        let mut extra_hunks = 0usize;
        // (file, written target, syntax) — resolution is deferred until the
        // whole commit is scanned and the file list is complete
        let mut raw_imports: BTreeSet<(String, String, ImportKind)> = BTreeSet::new();
        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("diff --git a/") {
                match rest.split_once(" b/") {
                    // git C-quotes exotic paths; the +++ line below re-captures them
                    Some((_, b)) if !b.contains('"') => {
                        push_file(&mut files, b.trim(), &mut extra_files, &mut files_capped)
                    }
                    _ => {}
                }
            } else if let Some(rest) = line.strip_prefix("+++ b/") {
                let b = rest.trim().trim_matches('"');
                if !b.is_empty() && b != "/dev/null" {
                    push_file(&mut files, b, &mut extra_files, &mut files_capped);
                }
            } else if line.starts_with("@@") {
                let mut parts = line.split("@@");
                parts.next();
                let ranges = parts.next().unwrap_or("");
                let ctx = parts.next().unwrap_or("");
                let sym = leading_identifier(ctx);
                if let Some(file) = files.last().filter(|_| !files_capped) {
                    if let Some(sym) = sym.as_ref()
                        && symbols.len() < MAX_SYMBOLS
                    {
                        symbols.insert(format!("{file}/{sym}"));
                    }
                    // blame anchors: the +c[,d] side is where the new lines live
                    if let Some((start, len)) = parse_new_side(ranges) {
                        if hunks.len() < MAX_HUNKS {
                            hunks.push(HunkSpan {
                                file: file.clone(),
                                start,
                                len,
                                symbol: sym.clone(),
                            });
                        } else {
                            extra_hunks += 1;
                        }
                    }
                }
            } else if line.starts_with('+') || line.starts_with('-') {
                let text = line.trim_start_matches(['+', '-']);
                // import edges ride the changed lines. Identifier-shaped
                // names in the diff body are not copied onto the record:
                // a `mentions:` line would become an entity-name join key.
                if raw_imports.len() < MAX_IMPORTS
                    && let Some(file) = files.last().filter(|_| !files_capped)
                    && let Some((target, kind)) = import_target(text)
                {
                    raw_imports.insert((file.clone(), target, kind));
                }
            }
        }

        // resolve import edges against this commit's own file list — exact
        // module-segment matching only; anything else stays unresolved
        let mut imports: Vec<(String, String, bool)> = Vec::new();
        for (file, target, kind) in raw_imports {
            if imports.len() >= MAX_IMPORTS {
                break;
            }
            let resolved = resolve_import(&target, kind, &files);
            imports.push(match resolved {
                Some(path) => (file, path, true),
                None => (file, target, false),
            });
        }
        let lock_bumps = lock_bumps_from_diff(body);
        out.push(GitCommit {
            sha,
            time,
            subject,
            files,
            extra_files,
            symbols: symbols.into_iter().collect(),
            hunks,
            extra_hunks,
            imports,
            parents,
            reverts,
            cherry_picked_from,
            fixes,
            lock_bumps,
        });
    }
    out
}

fn push_file(
    files: &mut Vec<String>,
    path: &str,
    extra_files: &mut usize,
    files_capped: &mut bool,
) {
    if path.is_empty() || files.iter().any(|x| x == path) {
        return;
    }
    if files.len() < MAX_FILES {
        files.push(path.to_string());
    } else {
        *files_capped = true;
        *extra_files += 1;
    }
}

/// New side of an `@@` header's range list: the `+c[,d]` token, parsed as
/// (start, len). An omitted `,d` means a single-line hunk (len 1). A missing
/// or malformed `+` token yields `None` and the hunk is skipped.
fn parse_new_side(ranges: &str) -> Option<(u32, u32)> {
    let plus = ranges.split_whitespace().find(|t| t.starts_with('+'))?;
    let mut it = plus.strip_prefix('+')?.splitn(2, ',');
    let start = it.next()?.parse().ok()?;
    let len = match it.next() {
        Some(l) => l.parse().ok()?,
        None => 1,
    };
    Some((start, len))
}

/// If `text` (a changed diff line with its +/- markers stripped) is an
/// import/use statement, return its written target and syntax family.
/// Exact prefixes only: `use <path>::...`, `import x.y.z` (multi-segment, so
/// stdlib noise like `import os` is not an edge), `from x.y import z`, and
/// `#include "p"` / `#include <p>`.
fn import_target(text: &str) -> Option<(String, ImportKind)> {
    let t = text.trim();
    if let Some(rest) = t.strip_prefix("use ") {
        // `use a::b::{c, d};` / `use a::b as c;` — keep the path before the
        // group brace, the alias, or the semicolon
        let path = rest
            .split(['{', ';'])
            .next()
            .unwrap_or_default()
            .split(" as ")
            .next()
            .unwrap_or_default()
            .trim()
            .trim_end_matches(':')
            .trim();
        return rust_like(path).then(|| (path.to_string(), ImportKind::Rust));
    }
    if let Some(rest) = t.strip_prefix("from ") {
        let (module, _) = rest.split_once(" import ")?;
        let module = module.trim();
        return dotted_like(module).then(|| (module.to_string(), ImportKind::Dotted));
    }
    if let Some(rest) = t.strip_prefix("import ") {
        let path = rest
            .split(';')
            .next()
            .unwrap_or_default()
            .split(" as ")
            .next()
            .unwrap_or_default()
            .trim();
        let stripped = path.trim_start_matches('.');
        return (dotted_like(path) && stripped.contains('.'))
            .then(|| (path.to_string(), ImportKind::Dotted));
    }
    if let Some(rest) = t.strip_prefix("#include") {
        let r = rest.trim();
        let inner = r
            .strip_prefix('<')
            .and_then(|x| x.strip_suffix('>'))
            .or_else(|| r.strip_prefix('"').and_then(|x| x.strip_suffix('"')))?;
        return (!inner.is_empty()).then(|| (inner.to_string(), ImportKind::Include));
    }
    None
}

/// A `::`-separated path of identifier segments (`crate::store::Store`).
fn rust_like(path: &str) -> bool {
    let p = path.trim();
    !p.is_empty()
        && p.split("::")
            .all(|seg| !seg.is_empty() && seg.chars().all(|c| c.is_alphanumeric() || c == '_'))
}

/// A `.`-separated module path (`x.y.z`, `.relative.mod` allowed).
fn dotted_like(path: &str) -> bool {
    let stripped = path.trim().trim_start_matches('.');
    !stripped.is_empty()
        && stripped
            .split('.')
            .all(|seg| !seg.is_empty() && seg.chars().all(|c| c.is_alphanumeric() || c == '_'))
}

/// Resolve a written import target to a repo-relative path using ONLY the
/// same commit's file list: exact string/module-segment matching, no fuzzy
/// matching, no guessing. File candidates win over module-dir candidates and
/// the longest (most specific) module path is tried first.
fn resolve_import(target: &str, kind: ImportKind, files: &[String]) -> Option<String> {
    let mut file_cands: Vec<String> = Vec::new();
    let mut dir_cands: Vec<String> = Vec::new();
    match kind {
        ImportKind::Rust => {
            let mut segs: Vec<&str> = target.split("::").collect();
            // `crate` is the crate root and roots at src/; `self`/`super` and
            // external crates have no deterministic repo mapping and stay
            // unresolved unless their segments happen to match exactly
            if segs.first() == Some(&"crate") {
                segs[0] = "src";
            }
            let joined = segs.join("/");
            if segs.len() >= 2 {
                // `a::b::Item` — Item is an item in module a::b (the common
                // reading of a CamelCase tail) or a module itself
                let module = segs[..segs.len() - 1].join("/");
                if module == "src" {
                    // item re-exported from the crate root
                    file_cands.push("src/lib.rs".into());
                    file_cands.push("src/main.rs".into());
                } else {
                    file_cands.push(format!("{module}.rs"));
                    file_cands.push(format!("{module}/mod.rs"));
                    dir_cands.push(module);
                }
                file_cands.push(format!("{joined}.rs"));
                file_cands.push(format!("{joined}/mod.rs"));
                dir_cands.push(joined);
            } else {
                file_cands.push(format!("{joined}.rs"));
            }
        }
        ImportKind::Dotted => {
            // python-style: x/y/z.py before x/y.py before x.py, each with an
            // __init__.py variant; bare prefixes act as module dirs
            let segs: Vec<&str> = target.trim_start_matches('.').split('.').collect();
            for n in (1..=segs.len()).rev() {
                let p = segs[..n].join("/");
                file_cands.push(format!("{p}.py"));
                file_cands.push(format!("{p}/__init__.py"));
                dir_cands.push(p);
            }
        }
        ImportKind::Include => file_cands.push(target.to_string()),
    }
    // pass 1: exact file hit
    for cand in &file_cands {
        if let Some(f) = files.iter().find(|f| *f == cand) {
            return Some(f.clone());
        }
    }
    // pass 2: a module dir of a committed file
    for cand in &dir_cands {
        if files
            .iter()
            .any(|f| f.rsplit_once('/').is_some_and(|(d, _)| d == cand))
        {
            return Some(cand.clone());
        }
    }
    None
}

/// Leading identifier of a hunk-header context, skipping language keywords:
/// `@@ -1,2 +3,4 @@ fn write_file(x: u8)` -> `write_file`,
/// `@@ ... @@ def save(self)` -> `save`.
fn leading_identifier(ctx: &str) -> Option<String> {
    const KEYWORDS: &[&str] = &[
        "fn",
        "def",
        "function",
        "func",
        "method",
        "class",
        "struct",
        "impl",
        "public",
        "private",
        "protected",
        "static",
        "async",
        "const",
        "let",
        "var",
        "extern",
        "unsafe",
        "pub",
        "export",
        "return",
        "type",
        "interface",
        "enum",
        "trait",
        "virtual",
        "template",
        "override",
        "final",
    ];
    let mut ident = String::new();
    for raw in ctx.split_whitespace() {
        let word = raw.trim_start_matches('&');
        if word.is_empty() {
            continue;
        }
        if ident.is_empty() && KEYWORDS.contains(&word) {
            continue;
        }
        // cut at the first non-identifier char: "write_event(self," -> "write_event"
        ident = word
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        break;
    }
    // value-taking keywords ("return None;", "return true") hand us a VALUE,
    // not a declaration name; language literals are never hunk symbols
    const JUNK: &[&str] = &[
        "none",
        "some",
        "ok",
        "err",
        "true",
        "false",
        "self",
        "super",
        "null",
        "nil",
        "undefined",
        "return",
    ];
    let lower = ident.to_lowercase();
    (!ident.is_empty()
        && ident.chars().any(|c| !c.is_ascii_digit())
        && !JUNK.contains(&lower.as_str()))
    .then_some(ident)
}

/// Content for a commit record. Files, modules, symbols, hunks and import
/// edges are the handles the diff structure recorded. Changed-line
/// identifiers are not copied onto a `mentions:` line.
pub fn record_content(c: &GitCommit) -> String {
    let mut s = format!("commit {} {}", c.sha, c.subject);
    if !c.files.is_empty() {
        s.push_str("\nfiles: ");
        s.push_str(&c.files.join(", "));
        if c.extra_files > 0 {
            s.push_str(&format!(" (+{} more)", c.extra_files));
        }
    }
    // Only multi-segment module paths pass the shape test; single-segment dirs
    // ("src") are already covered by the file entities beneath them.
    let modules: BTreeSet<String> = c
        .files
        .iter()
        .filter_map(|f| f.rsplit_once('/').map(|(d, _)| d.to_string()))
        .filter(|d| d.contains('/'))
        .collect();
    if !modules.is_empty() {
        s.push_str("\nmodules: ");
        s.push_str(&modules.into_iter().collect::<Vec<_>>().join(", "));
    }
    if !c.symbols.is_empty() {
        s.push_str("\nsymbols: ");
        s.push_str(&c.symbols.join(", "));
    }
    if !c.hunks.is_empty() {
        s.push_str("\nhunks: ");
        let items: Vec<String> = c
            .hunks
            .iter()
            .map(|h| format!("{}:{}+{}", h.file, h.start, h.len))
            .collect();
        // keep the joined list bounded (~400 chars); dropped spans are counted
        // together with extra_hunks so truncation stays visible
        let mut shown = items.len();
        while shown > 1 && items[..shown].join(", ").len() > MAX_HUNK_LINE {
            shown -= 1;
        }
        s.push_str(&items[..shown].join(", "));
        let hidden = c.hunks.len() - shown + c.extra_hunks;
        if hidden > 0 {
            s.push_str(&format!(" (+{hidden} more)"));
        }
    }
    if !c.imports.is_empty() {
        s.push_str("\nimports: ");
        let pairs: Vec<String> = c
            .imports
            .iter()
            .map(|(file, target, resolved)| {
                if *resolved {
                    format!("{file}->{target}")
                } else {
                    // unresolved stays unresolved AND visible
                    format!("{file}->?{target}")
                }
            })
            .collect();
        s.push_str(&pairs.join(", "));
    }
    s
}

/// Extract X.Y / X.Y.Z version tokens from prose ("worked in 1.41.0, broke in 1.42.1").
/// Groups are capped at 3 digits so calendar strings ("2026.09") don't match.
pub fn extract_versions(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut tok = String::new();
    let flush = |tok: &mut String, out: &mut Vec<String>| {
        let groups: Vec<&str> = tok.split('.').collect();
        let ok = groups.len() >= 2
            && groups.len() <= 3
            && groups
                .iter()
                .all(|g| !g.is_empty() && g.len() <= 3 && g.chars().all(|c| c.is_ascii_digit()));
        if ok && !out.iter().any(|v| v == tok) {
            out.push(tok.clone());
        }
        tok.clear();
    };
    for c in text.chars() {
        if c.is_ascii_digit() || c == '.' {
            tok.push(c);
        } else {
            flush(&mut tok, &mut out);
        }
    }
    flush(&mut tok, &mut out);
    out
}

fn parse_version(v: &str) -> Vec<u64> {
    v.split('.').map(|g| g.parse().unwrap_or(0)).collect()
}

fn cmp_version(a: &str, b: &str) -> std::cmp::Ordering {
    let (a, b) = (parse_version(a), parse_version(b));
    let n = a.len().max(b.len());
    for i in 0..n {
        let (x, y) = (
            a.get(i).copied().unwrap_or(0),
            b.get(i).copied().unwrap_or(0),
        );
        if x != y {
            return x.cmp(&y);
        }
    }
    std::cmp::Ordering::Equal
}

/// Tags embedding one of `versions` ("v1.42.1", "release-1.42.1"). Returns the
/// matching tag names, sorted ascending by version.
pub fn match_version_tags(tags: &[&str], versions: &[String]) -> Vec<String> {
    let mut matched: Vec<String> = tags
        .iter()
        .filter(|t| {
            versions.iter().any(|v| {
                t.rsplit(['v', '-', '_'])
                    .any(|seg| extract_versions(seg).iter().any(|ev| ev == v))
            })
        })
        .map(|t| t.to_string())
        .collect();
    matched.sort_by(|a, b| cmp_version(a, b));
    matched
}

/// (worked_in = lowest, broke_in = highest) when at least two distinct
/// versions matched — the "broke after upgrading" window.
pub fn version_range(matched_tags: &[String]) -> Option<(String, String)> {
    let mut versions: Vec<String> = matched_tags
        .iter()
        .filter_map(|t| extract_versions(t).into_iter().next_back())
        .collect();
    versions.sort_by(|a, b| cmp_version(a, b));
    versions.dedup();
    if versions.len() < 2 {
        return None;
    }
    Some((versions[0].clone(), versions[versions.len() - 1].clone()))
}

/// SHAs from `git rev-list A..B` output, lowercased.
pub fn parse_rev_list(raw: &str) -> std::collections::HashSet<String> {
    raw.lines()
        .map(|l| l.trim().to_ascii_lowercase())
        .filter(|l| !l.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced::retroactive_backfill::extract_entities;

    const FIXTURE: &str = "\u{1e}abc123def4567890abc123def4567890abc12345\u{1f}2026-09-01T12:00:00+00:00\u{1f}fix: bound the heartbeat write
diff --git a/events/local.py b/events/local.py
index 111..222 100644
--- a/events/local.py
+++ b/events/local.py
@@ -10,7 +10,8 @@ def write_event(self, payload):
+from events import local
+import ghost.module.nothere
     return out
diff --git a/src/store.rs b/src/store.rs
@@ -40,6 +40,7 @@ impl LocalFileStore {
     ok
diff --git a/src/app.rs b/src/app.rs
index 333..444 100644
--- a/src/app.rs
+++ b/src/app.rs
@@ -1,3 +1,6 @@ struct App
+use crate::store::Store;
+use serde_magic::Wizard;
     fn main()
diff --git a/src/wrap.c b/src/wrap.c
@@ -2,1 +2,3 @@
+#include \"src/legacy.h\"
+#include <stdint.h>
diff --git a/src/legacy.h b/src/legacy.h
@@ -0,0 +1,1 @@
+int legacy(void);
\u{1e}fff000111222333444555666777888999aaaabbb\u{1f}2026-09-02T08:30:00+00:00\u{1f}docs: readme
diff --git a/README.md b/README.md
@@ -1,3 +1,4 @@ 
     text
";

    #[test]
    fn parse_extracts_files_symbols_and_time() {
        let commits = parse_git_log(FIXTURE);
        assert_eq!(commits.len(), 2);
        let c = &commits[0];
        assert_eq!(c.sha, "abc123def4567890abc123def4567890abc12345");
        assert_eq!(
            c.files,
            vec![
                "events/local.py",
                "src/store.rs",
                "src/app.rs",
                "src/wrap.c",
                "src/legacy.h"
            ]
        );
        assert!(
            c.symbols
                .contains(&"events/local.py/write_event".to_string())
        );
        assert!(
            c.symbols
                .contains(&"src/store.rs/LocalFileStore".to_string())
        );
        assert_eq!(c.time.to_rfc3339(), "2026-09-01T12:00:00+00:00");
        // the docs commit touches files but has no parseable symbol context
        assert_eq!(commits[1].files, vec!["README.md"]);
        assert!(commits[1].symbols.is_empty());
    }

    #[test]
    fn hunk_spans_carry_new_side_numbers_and_camel_symbols() {
        let commits = parse_git_log(FIXTURE);
        let c = &commits[0];
        // one span per @@ header, attributed to the file being diffed
        assert_eq!(c.hunks.len(), 5);
        assert_eq!(
            c.hunks[0],
            HunkSpan {
                file: "events/local.py".into(),
                start: 10,
                len: 8,
                symbol: Some("write_event".into())
            }
        );
        assert_eq!(
            c.hunks[1],
            HunkSpan {
                file: "src/store.rs".into(),
                start: 40,
                len: 7,
                symbol: Some("LocalFileStore".into()) // camel-case header symbol
            }
        );
        // new-file hunk: -0,0 +1,1, no context symbol
        assert_eq!(
            c.hunks[4],
            HunkSpan {
                file: "src/legacy.h".into(),
                start: 1,
                len: 1,
                symbol: None
            }
        );
        // second commit keeps its own span
        assert_eq!(
            commits[1].hunks,
            vec![HunkSpan {
                file: "README.md".into(),
                start: 1,
                len: 4,
                symbol: None
            }]
        );
    }

    #[test]
    fn hunk_len_omitted_means_one_line() {
        let raw = "\u{1e}5555555555555555555555555555555555555555\u{1f}2026-09-01T12:00:00+00:00\u{1f}solo\n"
            .to_string()
            + "diff --git a/src/solo.rs b/src/solo.rs\n"
            + "@@ -5 +9 @@ fn solo()\n"
            + "+only line";
        let commits = parse_git_log(&raw);
        assert_eq!(
            commits[0].hunks,
            vec![HunkSpan {
                file: "src/solo.rs".into(),
                start: 9,
                len: 1,
                symbol: Some("solo".into())
            }]
        );
    }

    #[test]
    fn record_content_tokens_are_live_join_keys() {
        let commits = parse_git_log(FIXTURE);
        let content = record_content(&commits[0]);
        let ents = extract_entities(&content, &[]);
        for want in [
            "events/local.py",
            "src/store.rs",
            "events/local.py/write_event",
        ] {
            assert!(ents.iter().any(|e| e == want), "missing {want} in {ents:?}");
        }
        // hunk spans and import edges are part of the record content
        assert!(
            content.contains("\nhunks: events/local.py:10+8, src/store.rs:40+7, src/app.rs:1+6, src/wrap.c:2+3, src/legacy.h:1+1"),
            "hunks line wrong: {content}"
        );
        assert!(
            content.contains("\nimports: events/local.py->events, events/local.py->?ghost.module.nothere, src/app.rs->src/store.rs, src/app.rs->?serde_magic::Wizard, src/wrap.c->src/legacy.h, src/wrap.c->?stdint.h"),
            "imports line wrong: {content}"
        );
    }

    #[test]
    fn import_edges_resolve_against_the_same_commit() {
        let commits = parse_git_log(FIXTURE);
        let imports = &commits[0].imports;
        // `use crate::store::Store;` maps onto src/store.rs in the same commit
        assert!(
            imports
                .iter()
                .any(|(f, t, r)| f == "src/app.rs" && t == "src/store.rs" && *r),
            "crate::store::Store must resolve to src/store.rs: {imports:?}"
        );
        // quoted #include resolves verbatim; <> include has no repo mapping
        assert!(
            imports
                .iter()
                .any(|(f, t, r)| f == "src/wrap.c" && t == "src/legacy.h" && *r)
        );
        assert!(
            imports
                .iter()
                .any(|(f, t, r)| f == "src/wrap.c" && t == "stdint.h" && !*r)
        );
        // `from events import local` maps onto the module dir of events/local.py
        assert!(
            imports
                .iter()
                .any(|(f, t, r)| f == "events/local.py" && t == "events" && *r)
        );
        // an external crate and a missing module stay unresolved and visible
        assert!(
            imports
                .iter()
                .any(|(f, t, r)| f == "src/app.rs" && t == "serde_magic::Wizard" && !*r),
            "unresolvable rust import missing: {imports:?}"
        );
        assert!(
            imports
                .iter()
                .any(|(f, t, r)| f == "events/local.py" && t == "ghost.module.nothere" && !*r),
            "unresolvable python import missing: {imports:?}"
        );
        // the second commit has no import-bearing changed lines
        assert!(commits[1].imports.is_empty());
    }

    #[test]
    fn import_edges_are_deduped_and_capped() {
        let mut raw = String::from(
            "\u{1e}6666666666666666666666666666666666666666\u{1f}2026-09-01T12:00:00+00:00\u{1f}wire up\n",
        );
        raw.push_str("diff --git a/src/app.rs b/src/app.rs\n@@ -1,2 +1,3 @@\n");
        for i in 0..45 {
            raw.push_str(&format!("+use ext_crate_{i}::Thing;\n"));
        }
        // the same statement twice is one edge
        raw.push_str("+use ext_crate_0::Thing;\n");
        let commits = parse_git_log(&raw);
        assert_eq!(commits[0].imports.len(), MAX_IMPORTS, "cap at 40");
        assert!(commits[0].imports.iter().all(|(_, _, r)| !*r));
        assert_eq!(
            commits[0]
                .imports
                .iter()
                .filter(|(f, _, _)| f == "src/app.rs")
                .count(),
            MAX_IMPORTS
        );
    }

    #[test]
    fn hunk_caps_overflow_and_bound_the_content_line() {
        let mut raw = String::from(
            "\u{1e}7777777777777777777777777777777777777777\u{1f}2026-09-01T12:00:00+00:00\u{1f}spans\n",
        );
        raw.push_str("diff --git a/src/big.rs b/src/big.rs\n");
        for i in 0..250 {
            raw.push_str(&format!("@@ -{i},1 +{i},2 @@ fn site_{i}()\n"));
        }
        let commits = parse_git_log(&raw);
        assert_eq!(commits[0].hunks.len(), MAX_HUNKS);
        assert_eq!(commits[0].extra_hunks, 50);
        let content = record_content(&commits[0]);
        let hunk_line = content
            .lines()
            .find(|l| l.starts_with("hunks: "))
            .expect("hunks line present");
        assert!(
            hunk_line.len() <= MAX_HUNK_LINE + 24,
            "line must stay ~bounded: {} chars",
            hunk_line.len()
        );
        // every span is accounted for: shown + hidden == 250
        let shown = hunk_line.matches(".rs:").count();
        let hidden: usize = hunk_line
            .split("(+")
            .nth(1)
            .and_then(|rest| rest.split(' ').next())
            .and_then(|n| n.parse().ok())
            .expect("visible (+N more) counter");
        assert_eq!(
            shown + hidden,
            250,
            "shown {shown} + hidden {hidden} != 250"
        );
    }

    #[test]
    fn phantom_records_from_control_chars_are_skipped() {
        // a subject containing \x1e fabricates a second record header whose
        // "sha" is prose: only full 40-hex shas become records
        let raw = "\u{1e}1111111111111111111111111111111111111111\u{1f}2026-09-01T12:00:00+00:00\u{1f}subject with\u{1e}embedded split\u{1f}2026-09-01T00:00:00+00:00\u{1f}junk";
        let commits = parse_git_log(raw);
        assert_eq!(commits.len(), 1, "phantom header must not become a record");
        assert_eq!(commits[0].sha, "1111111111111111111111111111111111111111");
    }

    #[test]
    fn file_cap_counts_extras_and_stops_symbol_attribution() {
        let mut raw = String::from(
            "\u{1e}2222222222222222222222222222222222222222\u{1f}2026-09-01T12:00:00+00:00\u{1f}big move\n",
        );
        for i in 0..52 {
            raw.push_str(&format!("diff --git a/src/f{i}.rs b/src/f{i}.rs\n"));
            raw.push_str(&format!("@@ -1,2 +1,3 @@ fn handler_{i}(x: u8)\n"));
        }
        let commits = parse_git_log(&raw);
        assert_eq!(commits[0].files.len(), MAX_FILES);
        assert_eq!(commits[0].extra_files, 2);
        // hunks past the cap must NOT attach their symbols to file #50
        assert!(
            !commits[0].symbols.iter().any(|s| s.contains("handler_50")),
            "symbol past the file cap must not be attributed to the last recorded file"
        );
        assert!(commits[0].symbols.iter().any(|s| s.contains("handler_0")));
        // hunks past the file cap must not be attributed to file #50 either:
        // 52 file blocks, but only the 50 recorded files keep their spans
        assert_eq!(commits[0].hunks.len(), MAX_FILES);
        assert!(
            commits[0]
                .hunks
                .iter()
                .all(|h| h.file != "src/f50.rs" && h.file != "src/f51.rs")
        );
        assert_eq!(
            commits[0].extra_hunks, 0,
            "dropped-for-attribution is not span overflow"
        );
        let content = record_content(&commits[0]);
        assert!(
            content.contains("(+2 more)"),
            "truncation must be visible: {content}"
        );
    }

    #[test]
    fn quoted_paths_fall_back_to_plusplus_line() {
        // git C-quotes exotic paths in the diff --git header; the +++ line
        // re-captures the file (quotes stripped)
        let raw = "\u{1e}3333333333333333333333333333333333333333\u{1f}2026-09-01T12:00:00+00:00\u{1f}odd path\n"
            .to_string()
            + "diff --git \"a/spa ce\" \"b/spa ce\"\n"
            + "index 111..222 100644\n"
            + "--- a/\"spa ce\"\n"
            + "+++ b/\"spa ce\"\n"
            + "@@ -1,2 +1,3 @@ fn main()\n";
        let commits = parse_git_log(&raw);
        assert_eq!(
            commits[0].files,
            vec!["spa ce"],
            "files: {:?}",
            commits[0].files
        );
        assert!(commits[0].symbols.contains(&"spa ce/main".to_string()));
    }

    #[test]
    fn modifier_keywords_are_not_symbols() {
        assert_eq!(
            leading_identifier("pub fn write_event(self, x: u8)"),
            Some("write_event".into())
        );
        assert_eq!(
            leading_identifier("pub struct Config {"),
            Some("Config".into())
        );
        assert_eq!(
            leading_identifier("export function save()"),
            Some("save".into())
        );
        assert_eq!(leading_identifier("return None;"), None);
        assert_eq!(leading_identifier("trait Store {"), Some("Store".into()));
    }

    #[test]
    fn versions_ranges_and_rev_list() {
        let text = "Worked in 1.41.0, broke after upgrading to 1.42.1";
        let versions = extract_versions(text);
        assert_eq!(versions, vec!["1.41.0", "1.42.1"]);
        // calendar strings and years must not match
        assert!(extract_versions("since 2026.09 it failed on 2026-09-28").is_empty());

        let tags = ["v1.41.0", "v1.42.1", "whitepaper-tr-2026-01"];
        let matched = match_version_tags(&tags, &versions);
        assert_eq!(matched, vec!["v1.41.0", "v1.42.1"]);
        assert_eq!(
            version_range(&matched),
            Some(("1.41.0".into(), "1.42.1".into()))
        );
        assert_eq!(version_range(&["v1.42.1".into()]), None);

        let shas = parse_rev_list("ABC111\n\n def222 ");
        assert!(shas.contains("abc111") && shas.contains("def222"));
    }

    #[test]
    fn parents_and_the_git_revert_trailer_parse_and_nothing_else_does() {
        let parent = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let reverted = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let sha = "cccccccccccccccccccccccccccccccccccccccc";
        let raw = format!(
            "\u{1e}{sha}\u{1f}2024-07-23T16:00:00+00:00\u{1f}{parent}\u{1f}Revert \"time wheel\"\u{1f}Revert \"time wheel\"\n\nThis reverts commit {reverted}.\r\n\nA later sentence mentions commit deadbeef but is not the trailer.\u{1d}\ndiff --git a/src/a.rs b/src/a.rs\n@@ -1 +1 @@ fn load()\n-a\n+b\n"
        );
        let commits = parse_git_log(&raw);
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].parents, vec![parent]);
        assert_eq!(commits[0].reverts.as_deref(), Some(reverted));
        assert_eq!(commits[0].files, vec!["src/a.rs"]);
        assert!(revert_target("this reverts commit {reverted}.").is_none());
        assert!(revert_target("This reverts commit abc.").is_none());
        assert!(revert_target("This reverts commit {reverted}").is_none());
    }

    #[test]
    fn cherry_pick_and_fixes_trailers_are_exact_shas() {
        let original = "dddddddddddddddddddddddddddddddddddddddd";
        let fixed = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
        let short = "abcdef123456";
        let sha = "ffffffffffffffffffffffffffffffffffffffff";
        let parent = "1111111111111111111111111111111111111111";
        let raw = format!(
            "\u{1e}{sha}\u{1f}2024-07-23T16:00:00+00:00\u{1f}{parent}\u{1f}backport\u{1f}backport\n\n(cherry picked from commit {original})\r\nFixes: {fixed}\nFixes: {short}\nfixes: {fixed}\nFixes: {fixed} and more\nFixes: abc\n\u{1d}\n"
        );
        let commits = parse_git_log(&raw);
        assert_eq!(commits[0].cherry_picked_from.as_deref(), Some(original));
        assert_eq!(commits[0].fixes, vec![fixed.to_string()]);
        assert!(cherry_picked_from("(cherry picked from commit abc)").is_none());
        assert!(fixes_targets(&format!("Fixes: {short}")).is_empty());
        assert!(fixes_targets(&format!("fixes: {fixed}")).is_empty());
        assert!(fixes_targets(&format!("Fixes: {fixed} extra")).is_empty());
        assert_eq!(
            fixes_targets(&format!("Fixes: {fixed}")),
            vec![fixed.to_string()]
        );
    }

    #[test]
    fn diff_body_identifiers_are_not_written_as_mentions() {
        let sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let raw = format!(
            "\u{1e}{sha}\u{1f}2024-07-23T16:00:00+00:00\u{1f}\u{1f}set the pool\u{1f}set the pool\u{1d}\ndiff --git a/src/a.rs b/src/a.rs\n@@ -1 +1 @@ fn load()\n-old\n+connection_pool = 1\n"
        );
        let commits = parse_git_log(&raw);
        assert_eq!(commits.len(), 1);
        let content = record_content(&commits[0]);
        assert!(!content.contains("mentions:"), "{content}");
        assert!(
            !content.contains("connection_pool"),
            "a diff-body identifier is not a join key: {content}"
        );
        assert!(content.contains("files: src/a.rs"), "{content}");
    }

    #[test]
    fn lockfile_diffs_record_exact_version_moves_and_ignore_prose() {
        let diff = "\
diff --git a/Cargo.lock b/Cargo.lock
index 111..222 100644
--- a/Cargo.lock
+++ b/Cargo.lock
@@ -1,6 +1,6 @@
 [[package]]
 name = \"reqwest\"
-version = \"0.12.8\"
+version = \"0.12.9\"
 source = \"registry+https://github.com/rust-lang/crates.io-index\"
@@ -20,6 +20,9 @@ checksum = \"aa\"
+[[package]]
+name = \"added-only\"
+version = \"1.0.0\"
 [[package]]
 name = \"libc\"
-version = \"0.2.1\"
-version = \"0.2.2\"
+version = \"0.2.3\"
+version = \"0.2.4\"
diff --git a/uv.lock b/uv.lock
@@ -1,3 +1,3 @@
 name = \"httpx\"
-version = \"0.27.0\"
+version = \"0.28.1\"
diff --git a/poetry.lock b/poetry.lock
@@ -1,3 +1,3 @@
 name = \"requests\"
-version = \"2.31.0\"
+version = \"2.32.3\"
diff --git a/package-lock.json b/package-lock.json
@@ -10,3 +10,3 @@
     \"node_modules/@scope/left-pad\": {
-      \"version\": \"1.0.0\",
+      \"version\": \"1.0.1\",
     \"node_modules/a/node_modules/bar\": {
-      \"version\": \"2.0.0\",
+      \"version\": \"2.0.1\",
diff --git a/go.sum b/go.sum
@@ -1,2 +1,2 @@
-github.com/foo/bar v1.2.3 h1:aaa
-github.com/foo/bar v1.2.3/go.mod h1:bbb
+github.com/foo/bar v1.2.4 h1:ccc
+github.com/foo/bar v1.2.4/go.mod h1:ddd
diff --git a/README.md b/README.md
@@ -1 +1 @@
-bumped serde from 1.0.0 to 9.9.9
+The message says reqwest 0.12.8 -> 0.12.9 but this file is not a lockfile.
";
        let bumps = lock_bumps_from_diff(diff);
        let got: Vec<String> = bumps
            .iter()
            .map(|bump| {
                format!(
                    "{} {} {} {}",
                    bump.ecosystem, bump.package, bump.old_version, bump.new_version
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                "cargo reqwest 0.12.8 0.12.9",
                "uv httpx 0.27.0 0.28.1",
                "poetry requests 2.31.0 2.32.3",
                "npm @scope/left-pad 1.0.0 1.0.1",
                "npm bar 2.0.0 2.0.1",
                "go github.com/foo/bar 1.2.3 1.2.4",
            ]
        );
        assert!(
            !got.iter().any(|row| row.contains("serde")
                || row.contains("added-only")
                || row.contains("libc")),
            "prose, adds, and ambiguous multi-version edits are not bumps: {got:?}"
        );
    }

    #[test]
    fn registry_paths_name_the_crate_and_version_exactly() {
        let path = "/home/alex/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.39.0/src/util/linked_list.rs";
        assert_eq!(
            registry_package(path),
            Some(("cargo".into(), "tokio".into(), "1.39.0".into()))
        );
        assert_eq!(
            registry_package("vendor/tokio-util-0.7.10/src/lib.rs"),
            Some(("cargo".into(), "tokio-util".into(), "0.7.10".into()))
        );
        assert_eq!(
            registry_package("vendor/tokio-1.39.0-alpha.1/src/lib.rs"),
            Some(("cargo".into(), "tokio".into(), "1.39.0-alpha.1".into()))
        );
        assert_eq!(
            registry_package("/go/pkg/mod/github.com/stretchr/testify@v1.8.0/assert/assertions.go"),
            Some((
                "go".into(),
                "github.com/stretchr/testify".into(),
                "1.8.0".into()
            ))
        );
        assert_eq!(
            package_anchor_id("cargo", "tokio", "1.39.0"),
            "pkg:cargo:tokio@1.39.0"
        );
        assert_eq!(
            package_anchor_id("npm", "@scope/left-pad", "1.0.1"),
            "pkg:npm:@scope/left-pad@1.0.1"
        );
        assert_eq!(registry_package("tokio/src/util/linked_list.rs"), None);
        assert_eq!(registry_package("pkg/kubelet/server.go"), None);
        assert_eq!(registry_package("server.go"), None);
    }
}
