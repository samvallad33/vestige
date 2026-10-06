//! Resolve a `stack_frame` or `failing_test` to the commits git records as
//! having changed that file before the failing revision.
//!
//! The path is an exact identity (`path:<file>` touched edges). The line, when
//! the frame has one, is `git blame` at the failing revision. Suspects are
//! commits that touched that path and are ancestors of the revision (parent
//! edges, or `git merge-base --is-ancestor` when a merge was only bridged).
//! Nothing is ranked by the text of a message: later-reverted, blame, hunk
//! overlap, then parent distance, then sha.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use vestige_core::Storage;
use vestige_core::advanced::git_records;

use super::repo_ingest::run_git;

pub(crate) struct FrameHit {
    pub id: String,
    pub sha: String,
    pub hops: u32,
    pub blame: bool,
    pub later_reverted: bool,
    pub revert_id: Option<String>,
    pub touched_hunk: bool,
    pub path_anchor: String,
}

impl FrameHit {
    /// Smaller sorts first. Reverted commits, then the blame commit, then a
    /// recorded hunk that covers the line, then fewer parent hops, then sha.
    pub(crate) fn rank_key(&self) -> (bool, bool, bool, u32, &str) {
        (
            !self.later_reverted,
            !self.blame,
            !self.touched_hunk,
            self.hops,
            self.sha.as_str(),
        )
    }
}

pub(crate) struct FrameResolution {
    pub hits: Vec<FrameHit>,
    pub detail: String,
}

struct Revision {
    sha: String,
    node_id: String,
    checkout: PathBuf,
}

struct Blamed {
    sha: String,
    /// Line number in the blamed commit. Equal to the frame line when blame
    /// was of a whole file.
    line: u32,
}

/// `Some` when this start is a path we could blame against a recorded
/// checkout. `None` leaves the caller on the plain recorded-edge walk.
pub(crate) fn resolve_start(
    storage: &Arc<Storage>,
    scope: &str,
    kind: &str,
    locator: &str,
    node_id: Option<&str>,
    node_cap: usize,
) -> Result<Option<FrameResolution>, String> {
    let parsed = match kind {
        "stack_frame" => split_frame(locator),
        "failing_test" => test_file(locator),
        _ => None,
    };
    let Some((path, line)) = parsed else {
        return Ok(None);
    };
    let Some(revision) = failing_revision(storage, scope, node_id, node_cap)? else {
        return Ok(None);
    };
    if !revision.checkout.is_dir() {
        return Ok(None);
    }
    let blamed = blame_at(&revision.checkout, &revision.sha, &path, line)?;
    let anchor = git_records::path_anchor(&path);
    let hops = parent_hops(storage, &revision.node_id, node_cap.max(64))?;
    let mut hits = suspects(
        storage,
        scope,
        &revision,
        &FrameQuery {
            path: &path,
            anchor: &anchor,
            line,
        },
        blamed.as_ref(),
        &hops,
    )?;
    // A path git cannot blame and no ingested commit touched is not a
    // resolution: leave the recorded-edge walk in place.
    if hits.is_empty() {
        return Ok(None);
    }
    hits.sort_by(|a, b| a.rank_key().cmp(&b.rank_key()));
    hits.truncate(node_cap.max(1));
    let blame_sha = blamed
        .as_ref()
        .map(|blamed| blamed.sha.as_str())
        .unwrap_or("none");
    let detail = match line {
        Some(line) => format!(
            "resolved {path}:{line} at {} by git blame {blame_sha}; {} commit(s) touched that path before the failing revision",
            revision.sha,
            hits.len()
        ),
        None => format!(
            "resolved {path} at {} by the last commit git records for that path ({blame_sha}); {} commit(s) touched it before the failing revision",
            revision.sha,
            hits.len()
        ),
    };
    Ok(Some(FrameResolution { hits, detail }))
}

fn full_sha(sha: &str) -> bool {
    sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit())
}

/// `file:line`, `file:line:column`, or a bare relative path. Absolute paths,
/// `..`, and option-shaped names are refused before they can reach git.
fn split_frame(frame: &str) -> Option<(String, Option<u32>)> {
    let frame = frame.trim().replace('\\', "/");
    if frame.is_empty() {
        return None;
    }
    let (mut path, mut line) = (frame.as_str(), None);
    // A trailing `:digits` is a line, and a second one is a column.
    for _ in 0..2 {
        let Some((head, tail)) = path.rsplit_once(':') else {
            break;
        };
        if tail.is_empty() || !tail.bytes().all(|b| b.is_ascii_digit()) {
            break;
        }
        let parsed: u32 = tail.parse().ok()?;
        if parsed == 0 {
            return None;
        }
        if line.is_none() {
            line = Some(parsed);
        }
        path = head;
    }
    if !path_ok(path) {
        return None;
    }
    Some((path.to_string(), line))
}

fn test_file(name: &str) -> Option<(String, Option<u32>)> {
    let file = name.trim().split("::").next().unwrap_or("").trim();
    let (path, line) = split_frame(file)?;
    if path.contains('/') || path.contains('.') {
        Some((path, line))
    } else {
        None
    }
}

fn path_ok(path: &str) -> bool {
    if path.is_empty() || path.starts_with('-') || path.starts_with('/') {
        return false;
    }
    Path::new(path)
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
}

fn checkout_of(content: &str) -> Option<PathBuf> {
    content.lines().find_map(|line| {
        let path = line.strip_prefix("checkout ")?.trim();
        if path.is_empty() {
            None
        } else {
            Some(PathBuf::from(path))
        }
    })
}

fn sha_of(node: &vestige_core::KnowledgeNode) -> Option<String> {
    if let Some(sha) = node.tags.iter().find_map(|tag| tag.strip_prefix("commit:"))
        && full_sha(sha)
    {
        return Some(sha.to_ascii_lowercase());
    }
    let rest = node.content.lines().next()?.strip_prefix("commit ")?;
    let sha = rest.split_whitespace().next()?;
    full_sha(sha).then(|| sha.to_ascii_lowercase())
}

fn revision_from_node(storage: &Arc<Storage>, id: &str) -> Result<Option<Revision>, String> {
    let Some(node) = storage.get_node(id).map_err(|err| err.to_string())? else {
        return Ok(None);
    };
    let Some(sha) = sha_of(&node) else {
        return Ok(None);
    };
    let Some(checkout) = checkout_of(&node.content) else {
        return Ok(None);
    };
    Ok(Some(Revision {
        sha,
        node_id: node.id,
        checkout,
    }))
}

fn failing_revision(
    storage: &Arc<Storage>,
    scope: &str,
    node_id: Option<&str>,
    node_cap: usize,
) -> Result<Option<Revision>, String> {
    if let Some(id) = node_id {
        if let Some(revision) = revision_from_node(storage, id)? {
            return Ok(Some(revision));
        }
        let edges = storage
            .get_connections_for_memory(id)
            .map_err(|err| err.to_string())?;
        let mut found: Vec<Revision> = Vec::new();
        for edge in edges {
            if edge.link_type == "derived_from"
                && edge.source_id == id
                && let Some(revision) = revision_from_node(storage, &edge.target_id)?
            {
                found.push(revision);
            }
        }
        found.sort_by(|a, b| a.sha.cmp(&b.sha));
        if let Some(revision) = found.pop() {
            return Ok(Some(revision));
        }
    }
    tip_revision(storage, scope, node_cap)
}

/// Newest ingested commit in the scope. Used when the start point names a
/// path and no failure memory. It is the tip of what `ingest_repo` recorded,
/// not a guess about which line failed.
fn tip_revision(
    storage: &Arc<Storage>,
    scope: &str,
    node_cap: usize,
) -> Result<Option<Revision>, String> {
    let cap = i32::try_from(node_cap.clamp(1, 5000)).unwrap_or(500);
    let nodes = storage
        .current_code_context_nodes("event", Some(git_records::COMMIT_TAG), scope, cap)
        .map_err(|err| err.to_string())?;
    let mut best: Option<(i64, String, Revision)> = None;
    for node in nodes {
        let Some(sha) = sha_of(&node) else {
            continue;
        };
        let Some(checkout) = checkout_of(&node.content) else {
            continue;
        };
        let when = node.valid_from.map(|t| t.timestamp_millis()).unwrap_or(0);
        let revision = Revision {
            sha,
            node_id: node.id.clone(),
            checkout,
        };
        match &best {
            Some((at, id, _)) if (*at, id.as_str()) >= (when, revision.node_id.as_str()) => {}
            _ => best = Some((when, revision.node_id.clone(), revision)),
        }
    }
    Ok(best.map(|(_, _, revision)| revision))
}

fn blame_at(
    root: &Path,
    rev: &str,
    path: &str,
    line: Option<u32>,
) -> Result<Option<Blamed>, String> {
    if !full_sha(rev) || !path_ok(path) {
        return Ok(None);
    }
    if let Some(line) = line {
        let run = run_git(
            root,
            &[
                "blame".into(),
                "--porcelain".into(),
                "-L".into(),
                format!("{line},{line}"),
                rev.into(),
                "--".into(),
                path.into(),
            ],
        )?;
        if run.failure.is_some() {
            return Ok(None);
        }
        let text = String::from_utf8_lossy(&run.stdout);
        let mut parts = text.lines().next().unwrap_or("").split_whitespace();
        let Some(sha) = parts.next() else {
            return Ok(None);
        };
        if !full_sha(sha) {
            return Ok(None);
        }
        let orig = parts.next().and_then(|n| n.parse().ok()).unwrap_or(line);
        return Ok(Some(Blamed {
            sha: sha.to_ascii_lowercase(),
            line: orig,
        }));
    }
    let run = run_git(
        root,
        &[
            "log".into(),
            "-n".into(),
            "1".into(),
            "--format=%H".into(),
            rev.into(),
            "--".into(),
            path.into(),
        ],
    )?;
    if run.failure.is_some() {
        return Ok(None);
    }
    let sha = String::from_utf8_lossy(&run.stdout).trim().to_string();
    if !full_sha(&sha) {
        return Ok(None);
    }
    Ok(Some(Blamed {
        sha: sha.to_ascii_lowercase(),
        line: 0,
    }))
}

fn parent_hops(
    storage: &Arc<Storage>,
    start: &str,
    cap: usize,
) -> Result<HashMap<String, u32>, String> {
    let mut hops = HashMap::from([(start.to_string(), 0u32)]);
    let mut queue = VecDeque::from([start.to_string()]);
    while let Some(current) = queue.pop_front() {
        let depth = hops[&current];
        if hops.len() >= cap || depth >= 64 {
            continue;
        }
        let edges = storage
            .get_connections_for_memory(&current)
            .map_err(|err| err.to_string())?;
        let mut nexts: Vec<String> = edges
            .into_iter()
            .filter(|edge| edge.link_type == "derived_from" && edge.source_id == current)
            .map(|edge| edge.target_id)
            .collect();
        nexts.sort();
        nexts.dedup();
        for next in nexts {
            if hops.contains_key(&next) {
                continue;
            }
            hops.insert(next.clone(), depth + 1);
            queue.push_back(next);
        }
    }
    Ok(hops)
}

fn recorded_ancestor(root: &Path, ancestor: &str, rev: &str) -> bool {
    if ancestor == rev {
        return true;
    }
    if !full_sha(ancestor) || !full_sha(rev) {
        return false;
    }
    run_git(
        root,
        &[
            "merge-base".into(),
            "--is-ancestor".into(),
            ancestor.into(),
            rev.into(),
        ],
    )
    .ok()
    .is_some_and(|run| run.failure.is_none())
}

fn node_by_sha(storage: &Arc<Storage>, scope: &str, sha: &str) -> Result<Option<String>, String> {
    let resolution = storage.resolve_handle(&format!("commit:{sha}"));
    for id in resolution.ids {
        if storage
            .node_is_in_scope(&id, scope)
            .map_err(|err| err.to_string())?
        {
            return Ok(Some(id));
        }
    }
    Ok(None)
}

fn revert_of(storage: &Arc<Storage>, id: &str) -> Result<Option<String>, String> {
    let edges = storage
        .get_connections_for_memory(id)
        .map_err(|err| err.to_string())?;
    Ok(edges.into_iter().find_map(|edge| {
        (edge.link_type == "corrects" && edge.target_id == id && edge.source_id != id)
            .then_some(edge.source_id)
    }))
}

fn hunk_covers(content: &str, path: &str, line: u32) -> bool {
    if line == 0 {
        return false;
    }
    let Some(text) = content.lines().find(|line| line.starts_with("hunks: ")) else {
        return false;
    };
    for item in text.trim_start_matches("hunks: ").split(", ") {
        let item = item.split(" (+").next().unwrap_or(item).trim();
        let Some((file, span)) = item.rsplit_once(':') else {
            continue;
        };
        if file != path {
            continue;
        }
        let Some((start, len)) = span.split_once('+') else {
            continue;
        };
        let (Ok(start), Ok(len)) = (start.parse::<u32>(), len.parse::<u32>()) else {
            continue;
        };
        if len > 0 && line >= start && line < start.saturating_add(len) {
            return true;
        }
    }
    false
}

struct FrameQuery<'a> {
    path: &'a str,
    anchor: &'a str,
    line: Option<u32>,
}

fn suspects(
    storage: &Arc<Storage>,
    scope: &str,
    revision: &Revision,
    frame: &FrameQuery<'_>,
    blamed: Option<&Blamed>,
    hops: &HashMap<String, u32>,
) -> Result<Vec<FrameHit>, String> {
    let edges = storage
        .get_connections_for_memory(frame.anchor)
        .map_err(|err| err.to_string())?;
    let mut ids: Vec<String> = edges
        .into_iter()
        .filter(|edge| edge.link_type == "touched" && edge.target_id == frame.anchor)
        .map(|edge| edge.source_id)
        .collect();
    if let Some(blamed) = blamed
        && let Some(id) = node_by_sha(storage, scope, &blamed.sha)?
    {
        ids.push(id);
    }
    ids.sort();
    ids.dedup();

    let mut hits = Vec::new();
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id.clone()) {
            continue;
        }
        if !storage
            .node_is_in_scope(&id, scope)
            .map_err(|err| err.to_string())?
        {
            continue;
        }
        let Some(node) = storage.get_node(&id).map_err(|err| err.to_string())? else {
            continue;
        };
        let Some(sha) = sha_of(&node) else {
            continue;
        };
        let hop = hops.get(&id).copied();
        let ancestor = hop.is_some() || recorded_ancestor(&revision.checkout, &sha, &revision.sha);
        if !ancestor {
            continue;
        }
        let blame = blamed.is_some_and(|blamed| blamed.sha == sha);
        let line = if blame {
            blamed.map(|blamed| blamed.line).filter(|line| *line > 0)
        } else {
            frame.line
        };
        let touched_hunk =
            blame || line.is_some_and(|line| hunk_covers(&node.content, frame.path, line));
        let revert_id = revert_of(storage, &id)?;
        hits.push(FrameHit {
            id,
            sha,
            hops: hop.unwrap_or(u32::MAX),
            blame,
            later_reverted: revert_id.is_some(),
            revert_id,
            touched_hunk,
            path_anchor: frame.anchor.to_string(),
        });
    }
    Ok(hits)
}
