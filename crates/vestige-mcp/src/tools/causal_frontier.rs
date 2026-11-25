//! # God-Tier `causal_walk` Frontier Engines
//!
//! Deterministic, zero-vector, zero-LLM causal verification engines that turn
//! Vestige's Strata log + Git history + Reflex witnesses into a complete causal
//! proof engine:
//!
//! 1. **Forward Chesterton's Fence (`action="fence"`)** — `git blame` on target
//!    lines -> forward Strata BFS across `closes`, `corrects`, `evidence_of`,
//!    `pinned`, and `supersedes` edges + signed actor provenance (`human-git` vs
//!    `agent-mcp`) BEFORE an edit lands.
//! 2. **Zombie-Patch `git patch-id --stable` Tombstone Detector** — hashes a
//!    proposed diff or working tree and matches it against previously reverted
//!    (`corrects` target) or `rejected` commits to block whack-a-mole loops.
//! 3. **Counterfactual Halpern–Pearl Hunk Ablator (`action="ablate"`)** — runs
//!    inside an isolated temporary `git worktree` (never touches the user's
//!    checkout), performs a **Red-Before / Green-After Tautology Check** to catch
//!    self-confirming tests, splits the commit diff into atomic hunks, and tests
//!    leave-one-hunk-out counterfactuals + joint minimality to separate
//!    Halpern–Pearl `NECESSARY_CAUSE` hunks from `SUPERSTITIOUS_NOISE`.
//! 4. **Algebraic "Why-Not" Provenance Minimal-Cut Debugger (`action="why_not"`)**
//!    — proves mathematically why `expected_id` did not appear in a walk
//!    (`SCOPE_MISMATCH`, `SUPERSEDED_NODE`, `TOUCHED_EDGE_CAP_OBSTRUCTION`,
//!    `DEPTH_BOUND_OBSTRUCTION`, `DIRECTION_REVERSED_OBSTRUCTION`, or
//!    `DISCONNECTED_COMPONENT` with shared exact anchors).
//! 5. **Walk Enrichments (`enrich_walk_response`)** — additive, deterministic
//!    frontier analytics attached to every `action="walk"` result:
//!    - `diverse_causes` (maximally anchor-disjoint why-provenance slice)
//!    - `interaction_pairs` (`TraceVIC` + `TrAC` multi-commit joint-cause pairs)
//!    - `zombie_oscillations` (`RETRACE` revert/supersede oscillation cycles)
//!    - `trust_audit` (compaction-proof cryptographic trust-floor audit)
//!    - `env_drift` (`OOM-RL II` non-secret `env:*` / `cfg:*` drift detector)

use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use vestige_core::{ConnectionRecord, KnowledgeNode, Storage};

const WALK_MAX_DEPTH: u32 = 8;
const FENCE_MAX_DEPTH: u32 = 6;
const WHY_NOT_DIAG_DEPTH: u32 = 16;
const MAX_ABLATE_HUNKS_CAP: usize = 32;

const LINEAGE_LINKS: &[&str] = &["derived_from", "evidence_of", "closed_by", "closes", "corrects"];

// ---------------------------------------------------------------------------
// Shared Git Plumbing Helpers (Local-Only, Safe Config Overrides)
// ---------------------------------------------------------------------------

fn git_cmd(repo: &Path) -> Command {
    let mut cmd = Command::new("git");
    cmd.arg("-C")
        .arg(repo)
        .args([
            "-c",
            "core.fsmonitor=false",
            "-c",
            "color.ui=false",
            "-c",
            "diff.external=",
        ])
        .stdin(Stdio::null());
    cmd
}

fn git_stdout(repo: &Path, args: &[&str]) -> Result<String, String> {
    let out = git_cmd(repo)
        .args(args)
        .output()
        .map_err(|e| format!("failed to run git {args:?}: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!("git {args:?} failed: {stderr}"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn git_patch_id_stable(repo: &Path, diff_text: &str) -> Option<String> {
    if diff_text.trim().is_empty() {
        return None;
    }
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["patch-id", "--stable"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(diff_text.as_bytes());
    }
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .split_whitespace()
        .next()
        .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(|s| s.to_ascii_lowercase())
}

fn validate_repo_dir(args: &Value) -> Result<PathBuf, String> {
    let raw = args
        .get("repoPath")
        .or_else(|| args.get("repo_path"))
        .or_else(|| args.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "repoPath is required and must be an absolute git checkout path".to_string())?;
    let path = PathBuf::from(raw);
    if !path.is_absolute() {
        return Err(format!("repoPath '{raw}' must be an absolute path"));
    }
    if !path.is_dir() {
        return Err(format!("repoPath '{raw}' does not exist or is not a directory"));
    }
    Ok(path)
}

fn actor_provenance(node: &KnowledgeNode) -> &'static str {
    if node.tags.iter().any(|t| t == "actor:human-git") {
        return "human-git";
    }
    if node.tags.iter().any(|t| t == "actor:agent-mcp") {
        return "agent-mcp";
    }
    if node
        .tags
        .iter()
        .any(|t| t == "operator-stop" || t == "operator-rule")
    {
        return "operator-gate";
    }
    if node
        .source
        .as_deref()
        .is_some_and(|s| s.starts_with("vestige hook post-commit"))
    {
        return "git-hook";
    }
    "agent-or-user"
}

fn trust_tier(node: &KnowledgeNode) -> &'static str {
    if node.tags.iter().any(|t| t == "pinned" || t == "owner-directive") {
        return "owner_pinned";
    }
    if node
        .tags
        .iter()
        .any(|t| t.starts_with("tree-") || t == "reflex-witness" || t == "operator-stop")
    {
        return "reflex_witness";
    }
    if node
        .tags
        .iter()
        .any(|t| t == crate::commit_receipt::RECEIPT_TAG || t == "git-commit")
    {
        return "commit_receipt";
    }
    if node.tags.iter().any(|t| {
        matches!(
            t.as_str(),
            "untrusted" | "external" | "web" | "scraped" | "third-party"
        )
    }) || node.source.as_deref().is_some_and(|s| {
        let lower = s.to_ascii_lowercase();
        lower.starts_with("http://") || lower.starts_with("https://") || lower.contains("untrusted")
    }) {
        return "untrusted_external";
    }
    "agent_derived"
}

// ---------------------------------------------------------------------------
// 1. Forward "Chesterton's Fence" Pre-Edit Shield + Zombie Patch Tombstones
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FenceTarget {
    path: String,
    start_line: Option<u32>,
    end_line: Option<u32>,
}

fn parse_fence_target(raw: &str) -> Option<FenceTarget> {
    let trimmed = raw.trim().replace('\\', "/");
    let clean = trimmed.strip_prefix("./").unwrap_or(&trimmed);
    if clean.is_empty() {
        return None;
    }
    if let Some((path, range)) = clean.rsplit_once(':') {
        if let Some((a, b)) = range.split_once('-')
            && let (Ok(s), Ok(e)) = (a.parse::<u32>(), b.parse::<u32>())
            && s > 0
            && e >= s
        {
            return Some(FenceTarget {
                path: path.to_string(),
                start_line: Some(s),
                end_line: Some(e),
            });
        }
        if let Ok(line) = range.parse::<u32>()
            && line > 0
        {
            return Some(FenceTarget {
                path: path.to_string(),
                start_line: Some(line),
                end_line: Some(line),
            });
        }
    }
    Some(FenceTarget {
        path: clean.to_string(),
        start_line: None,
        end_line: None,
    })
}

#[derive(Debug, Clone)]
struct BlamedCommit {
    sha: String,
    author: String,
    lines_blamed: usize,
}

fn blame_target(repo: &Path, target: &FenceTarget) -> Result<Vec<BlamedCommit>, String> {
    let mut args: Vec<String> = vec!["blame".into(), "--porcelain".into()];
    if let (Some(s), Some(e)) = (target.start_line, target.end_line) {
        args.push("-L".into());
        args.push(format!("{s},{e}"));
    }
    args.push("--".into());
    args.push(target.path.clone());
    let str_args: Vec<&str> = args.iter().map(String::as_str).collect();
    let stdout = git_stdout(repo, &str_args)?;

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut authors: HashMap<String, String> = HashMap::new();
    let mut current_sha: Option<String> = None;

    for line in stdout.lines() {
        if let Some(first) = line.split_whitespace().next()
            && first.len() == 40
            && first.bytes().all(|b| b.is_ascii_hexdigit())
            && !first.bytes().all(|b| b == b'0')
        {
            let sha = first.to_ascii_lowercase();
            *counts.entry(sha.clone()).or_insert(0) += 1;
            current_sha = Some(sha);
            continue;
        }
        if let Some(author) = line.strip_prefix("author ")
            && let Some(ref sha) = current_sha
        {
            authors
                .entry(sha.clone())
                .or_insert_with(|| author.trim().to_string());
        }
    }

    let mut out: Vec<BlamedCommit> = counts
        .into_iter()
        .map(|(sha, lines_blamed)| {
            let author = authors
                .get(&sha)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            BlamedCommit {
                sha,
                author,
                lines_blamed,
            }
        })
        .collect();
    out.sort_by(|a, b| b.lines_blamed.cmp(&a.lines_blamed).then(a.sha.cmp(&b.sha)));
    Ok(out)
}

/// Forward Strata BFS from a set of seed commit/file nodes to find EVERY
/// decision, closed failure, revert, pinned directive, and witness guarding them.
fn forward_fence_walk(
    storage: &Storage,
    scope: &str,
    seeds: &[String],
    all_edges: &[ConnectionRecord],
) -> Vec<(KnowledgeNode, String, u32)> {
    let mut adj: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
    for e in all_edges {
        // Forward direction:
        // - If X `derived_from` Seed, X is downstream of Seed -> follow target -> source.
        // - If X `evidence_of` Seed, X witnesses Seed -> follow target -> source.
        // - If X `corrects` Seed, X reverted/corrected Seed -> follow target -> source.
        // - If Seed `closes` Failure, Failure is the exact bug Seed fixed -> follow source -> target.
        // - If Seed `closed_by` Fix, Fix guards Seed -> follow source -> target.
        // - If X `supersedes` Seed, X replaced Seed -> follow target -> source.
        match e.link_type.as_str() {
            "derived_from" | "evidence_of" | "corrects" | "supersedes" => {
                adj.entry(e.target_id.as_str())
                    .or_default()
                    .push((e.source_id.as_str(), e.link_type.as_str()));
            }
            "closes" | "closed_by" => {
                adj.entry(e.source_id.as_str())
                    .or_default()
                    .push((e.target_id.as_str(), e.link_type.as_str()));
                adj.entry(e.target_id.as_str())
                    .or_default()
                    .push((e.source_id.as_str(), e.link_type.as_str()));
            }
            _ => {}
        }
    }

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, String, u32)> = VecDeque::new();
    for s in seeds {
        if visited.insert(s.clone()) {
            queue.push_back((s.clone(), "seed".to_string(), 0));
        }
    }

    let mut out = Vec::new();
    while let Some((curr, via, depth)) = queue.pop_front() {
        if let Ok(Some(node)) = storage.get_node(&curr)
            && storage.node_is_in_scope(&curr, scope).unwrap_or(false)
        {
            out.push((node, via.clone(), depth));
        }
        if depth >= FENCE_MAX_DEPTH {
            continue;
        }
        if let Some(neighbors) = adj.get(curr.as_str()) {
            for &(next, kind) in neighbors {
                if visited.insert(next.to_string()) {
                    queue.push_back((next.to_string(), kind.to_string(), depth + 1));
                }
            }
        }
    }
    out
}

fn extract_commit_sha_from_node(node: &KnowledgeNode) -> Option<String> {
    for tag in &node.tags {
        if let Some(sha) = tag.strip_prefix("commit:")
            && sha.len() >= 7
            && sha.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Some(sha.to_ascii_lowercase());
        }
    }
    None
}

fn detect_zombie_patch(
    storage: &Storage,
    repo: &Path,
    all_edges: &[ConnectionRecord],
    proposed_diff: &str,
) -> (Option<String>, Vec<Value>) {
    let Some(proposed_pid) = git_patch_id_stable(repo, proposed_diff) else {
        return (None, Vec::new());
    };

    // Collect every commit SHA that was previously reverted (`corrects` target)
    // or explicitly tagged `rejected` / `zombie-tombstone`.
    let mut tombstones: BTreeMap<String, (String, String)> = BTreeMap::new();
    for edge in all_edges {
        if edge.link_type == "corrects"
            && let Ok(Some(target)) = storage.get_node(&edge.target_id)
            && let Some(sha) = extract_commit_sha_from_node(&target)
        {
            tombstones.insert(
                sha,
                (
                    edge.source_id.clone(),
                    format!("reverted/corrected by {}", edge.source_id),
                ),
            );
        }
    }
    for tag in ["rejected", "zombie-tombstone"] {
        for id in storage.resolve_handle(tag).ids {
            if let Ok(Some(node)) = storage.get_node(&id)
                && let Some(sha) = extract_commit_sha_from_node(&node)
            {
                tombstones.entry(sha).or_insert((
                    id,
                    format!("marked `{tag}` in Strata"),
                ));
            }
        }
    }

    let mut matches = Vec::new();
    for (sha, (reverter_id, reason)) in tombstones {
        if let Ok(commit_diff) = git_stdout(repo, &["show", "--format=", &sha])
            && let Some(tomb_pid) = git_patch_id_stable(repo, &commit_diff)
            && tomb_pid == proposed_pid
        {
            matches.push(json!({
                "code": "ZOMBIE_REGRESSION_DETECTED",
                "patch_id": proposed_pid,
                "reverted_commit": sha,
                "tombstone_memory_id": reverter_id,
                "reason": reason,
                "action": "STOP — your proposed diff has the exact canonical git patch-id of a previously reverted or rejected commit. Review the tombstone memory before re-introducing it.",
            }));
        }
    }
    (Some(proposed_pid), matches)
}

/// `causal_walk(action="fence", repoPath=..., targets=[...])`
pub async fn execute_fence(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let raw = args.unwrap_or_else(|| json!({}));
    let repo = validate_repo_dir(&raw)?;
    let scope = raw
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("user")
        .to_string();
    let target_list: Vec<FenceTarget> = raw
        .get("targets")
        .or_else(|| raw.get("files"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .filter_map(parse_fence_target)
                .collect()
        })
        .unwrap_or_default();
    if target_list.is_empty() {
        return Err(
            "causal_walk action='fence' requires non-empty `targets` (e.g. [\"src/auth.rs:10-40\"]) or `files`"
                .to_string(),
        );
    }
    let proposed_diff = raw
        .get("proposed_diff")
        .or_else(|| raw.get("proposedDiff"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let check_working_tree = raw
        .get("check_working_tree")
        .or_else(|| raw.get("checkWorkingTree"))
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let storage = Arc::clone(storage);
    tokio::task::spawn_blocking(move || {
        let all_edges = storage
            .get_all_connections()
            .map_err(|e| format!("failed to read Strata connections: {e}"))?;

        let mut fence_reports = Vec::new();
        let mut highest_severity = "OPEN_GROUND";

        for target in &target_list {
            let blamed = blame_target(&repo, target).unwrap_or_default();
            let mut seed_ids: Vec<String> = Vec::new();
            let mut seen_seed: HashSet<String> = HashSet::new();

            // Exact commit handles from git blame + exact file anchor handle.
            for b in &blamed {
                let short = &b.sha[..b.sha.len().min(7)];
                for handle in [format!("commit:{}", b.sha), format!("commit:{short}")] {
                    for id in storage.resolve_handle(&handle).ids {
                        if seen_seed.insert(id.clone()) {
                            seed_ids.push(id);
                        }
                    }
                }
            }
            for id in storage.resolve_handle(&format!("file:{}", target.path)).ids {
                if seen_seed.insert(id.clone()) {
                    seed_ids.push(id);
                }
            }

            let guarded = forward_fence_walk(storage.as_ref(), &scope, &seed_ids, &all_edges);
            let mut has_critical = false;
            let mut has_guarded = false;
            let mut guards_json = Vec::new();

            for (node, via, depth) in &guarded {
                let is_pinned = node.tags.iter().any(|t| t == "pinned");
                let is_fix_or_revert = matches!(via.as_str(), "closes" | "closed_by" | "corrects");
                let actor = actor_provenance(node);
                let is_human_guard = actor == "human-git";
                if is_pinned || is_fix_or_revert || is_human_guard {
                    has_critical = true;
                } else {
                    has_guarded = true;
                }
                guards_json.push(json!({
                    "id": node.id,
                    "node_type": node.node_type,
                    "via": via,
                    "depth": depth,
                    "actor": actor,
                    "trust_tier": trust_tier(node),
                    "pinned": is_pinned,
                    "tags": node.tags,
                    "summary": node.content.chars().take(240).collect::<String>(),
                }));
            }

            let severity = if has_critical {
                highest_severity = "CRITICAL_FENCE";
                "CRITICAL_FENCE"
            } else if has_guarded {
                if highest_severity == "OPEN_GROUND" {
                    highest_severity = "GUARDED_FENCE";
                }
                "GUARDED_FENCE"
            } else {
                "OPEN_GROUND"
            };

            fence_reports.push(json!({
                "path": target.path,
                "start_line": target.start_line,
                "end_line": target.end_line,
                "severity": severity,
                "blamed_commits": blamed.iter().map(|b| json!({
                    "sha": b.sha,
                    "author": b.author,
                    "lines_blamed": b.lines_blamed,
                })).collect::<Vec<_>>(),
                "guarding_records": guards_json,
            }));
        }

        let effective_diff = match proposed_diff {
            Some(d) => d,
            None if check_working_tree => git_stdout(&repo, &["diff", "HEAD"]).unwrap_or_default(),
            None => String::new(),
        };
        let (proposed_patch_id, mut zombie_matches) =
            detect_zombie_patch(storage.as_ref(), &repo, &all_edges, &effective_diff);
        let (_, git_zombies) = git_native_zombie_matches(&repo, &effective_diff);
        for gz in git_zombies {
            let rev = gz["reverted_commit"].as_str().unwrap_or("");
            if !zombie_matches
                .iter()
                .any(|z| z["reverted_commit"].as_str() == Some(rev))
            {
                zombie_matches.push(gz);
            }
        }
        let blocked_by_zombie = !zombie_matches.is_empty();
        if blocked_by_zombie {
            highest_severity = "ZOMBIE_REGRESSION_BLOCKED";
        }

        Ok(json!({
            "tool": "causal_walk",
            "action": "fence",
            "direction": "forward",
            "scope": scope,
            "severity": highest_severity,
            "blocked_by_zombie": blocked_by_zombie,
            "proposed_patch_id": proposed_patch_id,
            "zombie_matches": zombie_matches,
            "fences": fence_reports,
            "note": "Forward Chesterton's Fence shield: blames target lines via local git, walks forward across recorded Strata lineage/closes/corrects/pinned edges, and hashes the proposed diff against reverted/rejected commit patch-ids.",
        }))
    })
    .await
    .map_err(|e| format!("fence task failed: {e}"))?
}

// ---------------------------------------------------------------------------
// 2. Algebraic "Why-Not" Provenance Semiring Minimal-Cut Debugger
// ---------------------------------------------------------------------------

fn resolve_single_handle(storage: &Storage, handle: &str) -> Option<String> {
    if storage.get_node(handle).ok().flatten().is_some() {
        return Some(handle.to_string());
    }
    storage.resolve_handle(handle).ids.into_iter().next()
}

/// Run constrained backward walk BFS matching `causal_walk.rs` exact law
/// (`MAX_DEPTH = 8`, `<= 1` `touched` edge, same `scope`).
fn constrained_backward_path(
    storage: &Storage,
    scope: &str,
    start: &str,
    expected: &str,
    edges: &[ConnectionRecord],
) -> Option<Vec<Value>> {
    let mut adj: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
    for e in edges {
        match e.link_type.as_str() {
            "derived_from" | "corrects" => {
                adj.entry(e.source_id.as_str())
                    .or_default()
                    .push((e.target_id.as_str(), e.link_type.as_str()));
            }
            "evidence_of" | "closed_by" | "closes" => {
                adj.entry(e.target_id.as_str())
                    .or_default()
                    .push((e.source_id.as_str(), e.link_type.as_str()));
            }
            "touched" => {
                adj.entry(e.source_id.as_str())
                    .or_default()
                    .push((e.target_id.as_str(), "touched"));
                adj.entry(e.target_id.as_str())
                    .or_default()
                    .push((e.source_id.as_str(), "touched"));
            }
            _ => {}
        }
    }

    // State: (node_id, touched_used, depth, path)
    let mut visited: HashSet<(String, bool)> = HashSet::new();
    let mut queue: VecDeque<(String, bool, u32, Vec<Value>)> = VecDeque::new();
    visited.insert((start.to_string(), false));
    queue.push_back((start.to_string(), false, 0, Vec::new()));

    while let Some((curr, touched_used, depth, path)) = queue.pop_front() {
        if curr == expected {
            return Some(path);
        }
        if depth >= WALK_MAX_DEPTH {
            continue;
        }
        if let Some(neighbors) = adj.get(curr.as_str()) {
            for &(next, kind) in neighbors {
                let is_touched = kind == "touched";
                if is_touched && touched_used {
                    continue;
                }
                if storage.get_node(next).ok().flatten().is_some()
                    && !storage.node_is_in_scope(next, scope).unwrap_or(false)
                {
                    continue;
                }
                let next_touched = touched_used || is_touched;
                if visited.insert((next.to_string(), next_touched)) {
                    let mut next_path = path.clone();
                    next_path.push(json!({"from": curr, "to": next, "edge": kind}));
                    queue.push_back((next.to_string(), next_touched, depth + 1, next_path));
                }
            }
        }
    }
    None
}

type DiagHop = (String, String, String, bool);

/// Unconstrained diagnostic BFS (upstream causal + unlimited `touched`, or undirected)
/// to isolate the minimal algebraic obstruction when the constrained walk fails.
fn unconstrained_diagnostic_path(
    start: &str,
    expected: &str,
    edges: &[ConnectionRecord],
    allow_reverse: bool,
) -> Option<Vec<DiagHop>> {
    // Each hop: (from, to, link_type, was_reversed)
    let mut adj: HashMap<&str, Vec<(&str, &str, bool)>> = HashMap::new();
    for e in edges {
        match e.link_type.as_str() {
            "derived_from" | "corrects" => {
                adj.entry(e.source_id.as_str())
                    .or_default()
                    .push((e.target_id.as_str(), e.link_type.as_str(), false));
                if allow_reverse {
                    adj.entry(e.target_id.as_str())
                        .or_default()
                        .push((e.source_id.as_str(), e.link_type.as_str(), true));
                }
            }
            "evidence_of" | "closed_by" | "closes" => {
                adj.entry(e.target_id.as_str())
                    .or_default()
                    .push((e.source_id.as_str(), e.link_type.as_str(), false));
                if allow_reverse {
                    adj.entry(e.source_id.as_str())
                        .or_default()
                        .push((e.target_id.as_str(), e.link_type.as_str(), true));
                }
            }
            "touched" | "supersedes" => {
                adj.entry(e.source_id.as_str())
                    .or_default()
                    .push((e.target_id.as_str(), e.link_type.as_str(), false));
                adj.entry(e.target_id.as_str())
                    .or_default()
                    .push((e.source_id.as_str(), e.link_type.as_str(), allow_reverse));
            }
            _ => {}
        }
    }

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, Vec<DiagHop>)> = VecDeque::new();
    visited.insert(start.to_string());
    queue.push_back((start.to_string(), Vec::new()));

    while let Some((curr, path)) = queue.pop_front() {
        if curr == expected {
            return Some(path);
        }
        if path.len() as u32 >= WHY_NOT_DIAG_DEPTH {
            continue;
        }
        if let Some(neighbors) = adj.get(curr.as_str()) {
            for &(next, kind, rev) in neighbors {
                if visited.insert(next.to_string()) {
                    let mut next_path = path.clone();
                    next_path.push((curr.clone(), next.to_string(), kind.to_string(), rev));
                    queue.push_back((next.to_string(), next_path));
                }
            }
        }
    }
    None
}

/// `causal_walk(action="why_not", from_handle=..., expected_id=...)`
pub async fn execute_why_not(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let raw = args.unwrap_or_else(|| json!({}));
    let scope = raw.get("scope").and_then(Value::as_str).unwrap_or("user");
    let from_raw = raw
        .get("from_handle")
        .or_else(|| raw.get("fromHandle"))
        .or_else(|| raw.get("node_id"))
        .or_else(|| raw.get("nodeId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "causal_walk action='why_not' requires `from_handle` (or `node_id`) and `expected_id`"
                .to_string()
        })?;
    let expected_raw = raw
        .get("expected_id")
        .or_else(|| raw.get("expectedId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "causal_walk action='why_not' requires `expected_id`".to_string())?;

    let Some(start_id) = resolve_single_handle(storage.as_ref(), from_raw) else {
        return Ok(json!({
            "tool": "causal_walk",
            "action": "why_not",
            "obstruction_code": "MISSING_START_NODE",
            "from_handle": from_raw,
            "expected_id": expected_raw,
            "detail": format!("Start handle `{from_raw}` does not resolve to any recorded Strata node."),
        }));
    };
    let Some(expected_id) = resolve_single_handle(storage.as_ref(), expected_raw) else {
        return Ok(json!({
            "tool": "causal_walk",
            "action": "why_not",
            "obstruction_code": "MISSING_EXPECTED_NODE",
            "from_id": start_id,
            "expected_id": expected_raw,
            "detail": format!("Expected handle `{expected_raw}` does not exist in Strata (never ingested or wrong handle)."),
            "remedy": "Ingest the commit/memory via remember(action='save') or session(action='codebase', params={'action':'ingest_repo', ...}).",
        }));
    };

    let start_node = storage
        .get_node(&start_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("start node `{start_id}` disappeared"))?;
    let expected_node = storage
        .get_node(&expected_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("expected node `{expected_id}` disappeared"))?;

    let in_walk_scope = storage
        .node_is_in_scope(&expected_id, scope)
        .map_err(|e| e.to_string())?;
    if !in_walk_scope {
        return Ok(json!({
            "tool": "causal_walk",
            "action": "why_not",
            "obstruction_code": "SCOPE_MISMATCH",
            "from_id": start_id,
            "expected_id": expected_id,
            "walk_scope": scope,
            "detail": format!("Expected node `{expected_id}` is outside walk scope `{scope}` (cross-scope traversal is strictly disabled)."),
            "remedy": "Re-run causal_walk with the matching scope where expected_id was recorded.",
        }));
    }

    let edges = storage
        .get_all_connections()
        .map_err(|e| format!("failed to load edges: {e}"))?;

    // Check if already reached under exact constrained causal_walk rules!
    if let Some(path) = constrained_backward_path(storage.as_ref(), scope, &start_id, &expected_id, &edges) {
        return Ok(json!({
            "tool": "causal_walk",
            "action": "why_not",
            "obstruction_code": "ALREADY_REACHED",
            "from_id": start_id,
            "expected_id": expected_id,
            "hops": path.len(),
            "path": path,
            "detail": "No obstruction — expected_id IS reachable under standard causal_walk constraints.",
        }));
    }

    // Check if superseded by a newer record.
    let superseder = edges
        .iter()
        .find(|e| e.link_type == "supersedes" && e.target_id == expected_id)
        .map(|e| e.source_id.clone());

    // Try upstream-only unconstrained (no touched cap, depth up to 16).
    if let Some(diag_path) = unconstrained_diagnostic_path(&start_id, &expected_id, &edges, false) {
        let touched_hops: Vec<&(String, String, String, bool)> = diag_path
            .iter()
            .filter(|(_, _, kind, _)| kind == "touched")
            .collect();
        let path_json: Vec<Value> = diag_path
            .iter()
            .map(|(u, v, k, _)| json!({"from": u, "to": v, "edge": k}))
            .collect();

        if touched_hops.len() >= 2 {
            let cut = touched_hops[1];
            return Ok(json!({
                "tool": "causal_walk",
                "action": "why_not",
                "obstruction_code": "TOUCHED_EDGE_CAP_OBSTRUCTION",
                "from_id": start_id,
                "expected_id": expected_id,
                "superseded_by": superseder,
                "touched_edges_on_path": touched_hops.len(),
                "blocking_cut_edge": { "from": cut.0, "to": cut.1, "edge": cut.2 },
                "unconstrained_path": path_json,
                "detail": format!(
                    "Path from `{start_id}` to `{expected_id}` crosses {} `touched` edges; causal_walk allows at most 1 `touched` edge per path (blocked at `{} --touched--> {}`).",
                    touched_hops.len(), cut.0, cut.1
                ),
                "remedy": format!(
                    "Promote the blocking co-touch into lineage: curate(action='link', source_id='{}', target_id='{}', link_type='derived_from')",
                    cut.0, cut.1
                ),
            }));
        }
        if diag_path.len() as u32 > WALK_MAX_DEPTH {
            return Ok(json!({
                "tool": "causal_walk",
                "action": "why_not",
                "obstruction_code": "DEPTH_BOUND_OBSTRUCTION",
                "from_id": start_id,
                "expected_id": expected_id,
                "required_depth": diag_path.len(),
                "max_depth": WALK_MAX_DEPTH,
                "unconstrained_path": path_json,
                "detail": format!(
                    "Valid lineage path exists at depth {}, exceeding MAX_DEPTH ({WALK_MAX_DEPTH}).",
                    diag_path.len()
                ),
            }));
        }
    }

    // Try reverse / undirected path -> catches forward-vs-backward direction inversions!
    if let Some(rev_path) = unconstrained_diagnostic_path(&start_id, &expected_id, &edges, true) {
        let path_json: Vec<Value> = rev_path
            .iter()
            .map(|(u, v, k, rev)| json!({"from": u, "to": v, "edge": k, "reversed": rev}))
            .collect();
        return Ok(json!({
            "tool": "causal_walk",
            "action": "why_not",
            "obstruction_code": "DIRECTION_REVERSED_OBSTRUCTION",
            "from_id": start_id,
            "expected_id": expected_id,
            "superseded_by": superseder,
            "undirected_path": path_json,
            "detail": format!(
                "`{expected_id}` is connected to `{start_id}` only by traversing forward/downstream edges (it is downstream or sibling, not an upstream causal ancestor)."
            ),
            "remedy": "Use causal_walk(action='fence') for forward traversal, or verify edge direction (`derived_from` points from child -> parent).",
        }));
    }

    if let Some(sup_id) = superseder {
        return Ok(json!({
            "tool": "causal_walk",
            "action": "why_not",
            "obstruction_code": "SUPERSEDED_NODE",
            "from_id": start_id,
            "expected_id": expected_id,
            "superseded_by": sup_id,
            "detail": format!("`{expected_id}` has no path from `{start_id}` and was superseded by `{sup_id}`."),
        }));
    }

    // Completely disconnected component -> check shared exact anchors (file:/commit:/pkg:).
    let start_tags: HashSet<&str> = start_node.tags.iter().map(String::as_str).collect();
    let shared_anchors: Vec<&str> = expected_node
        .tags
        .iter()
        .map(String::as_str)
        .filter(|t| start_tags.contains(t))
        .collect();

    Ok(json!({
        "tool": "causal_walk",
        "action": "why_not",
        "obstruction_code": "DISCONNECTED_COMPONENT",
        "from_id": start_id,
        "expected_id": expected_id,
        "shared_exact_tags": shared_anchors,
        "detail": format!(
            "No graph path (directed or undirected, within depth {WHY_NOT_DIAG_DEPTH}) connects `{start_id}` and `{expected_id}` in Strata."
        ),
        "remedy": format!(
            "Record the exact causal edge with curate(action='link', source_id='{start_id}', target_id='{expected_id}', link_type='derived_from')."
        ),
    }))
}

// ---------------------------------------------------------------------------
// 3. Counterfactual Halpern–Pearl Hunk Ablator + Tautology Detector
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AblationHunk {
    pub index: usize,
    pub file: String,
    pub header: String,
    pub patch: String,
}

/// Split a `git diff --unified=3` output into atomic single-hunk patches that
/// `git apply -R` can revert individually for Halpern–Pearl counterfactual tests.
pub fn split_unified_diff_into_hunks(diff: &str) -> Vec<AblationHunk> {
    let mut out = Vec::new();
    let mut file_header = String::new();
    let mut current_file = String::new();
    let mut current_hunk_header = String::new();
    let mut current_hunk_body = String::new();

    let flush = |out: &mut Vec<AblationHunk>,
                 file_header: &str,
                 current_file: &str,
                 hunk_header: &mut String,
                 hunk_body: &mut String| {
        if !hunk_header.is_empty() && !file_header.is_empty() {
            let mut patch = String::with_capacity(file_header.len() + hunk_body.len());
            patch.push_str(file_header);
            patch.push_str(hunk_body);
            let idx = out.len();
            out.push(AblationHunk {
                index: idx,
                file: current_file.to_string(),
                header: hunk_header.clone(),
                patch,
            });
            hunk_header.clear();
            hunk_body.clear();
        }
    };

    for line in diff.lines() {
        if line.starts_with("diff --git ") {
            flush(
                &mut out,
                &file_header,
                &current_file,
                &mut current_hunk_header,
                &mut current_hunk_body,
            );
            file_header.clear();
            current_file.clear();
            file_header.push_str(line);
            file_header.push('\n');
            continue;
        }
        if line.starts_with("@@ ") {
            flush(
                &mut out,
                &file_header,
                &current_file,
                &mut current_hunk_header,
                &mut current_hunk_body,
            );
            current_hunk_header = line.to_string();
            current_hunk_body.push_str(line);
            current_hunk_body.push('\n');
            continue;
        }
        if current_hunk_header.is_empty() {
            if let Some(rest) = line.strip_prefix("+++ b/") {
                current_file = rest.to_string();
            } else if current_file.is_empty()
                && let Some(rest) = line.strip_prefix("--- a/")
            {
                current_file = rest.to_string();
            }
            file_header.push_str(line);
            file_header.push('\n');
        } else {
            current_hunk_body.push_str(line);
            current_hunk_body.push('\n');
        }
    }
    flush(
        &mut out,
        &file_header,
        &current_file,
        &mut current_hunk_header,
        &mut current_hunk_body,
    );
    out
}

struct WorktreeGuard {
    repo: PathBuf,
    worktree: PathBuf,
    _tempdir: tempfile::TempDir,
}

impl WorktreeGuard {
    fn create(repo: &Path, rev: &str) -> Result<Self, String> {
        let tempdir = tempfile::tempdir().map_err(|e| format!("tempdir failed: {e}"))?;
        let worktree = tempdir.path().join("wt");
        let wt_str = worktree
            .to_str()
            .ok_or_else(|| "non-UTF-8 worktree path".to_string())?;
        git_stdout(repo, &["worktree", "add", "--detach", "--quiet", wt_str, rev])?;
        Ok(Self {
            repo: repo.to_path_buf(),
            worktree,
            _tempdir: tempdir,
        })
    }
}

impl Drop for WorktreeGuard {
    fn drop(&mut self) {
        if let Some(wt) = self.worktree.to_str() {
            let _ = git_cmd(&self.repo)
                .args(["worktree", "remove", "--force", wt])
                .output();
        }
    }
}

fn run_oracle(worktree: &Path, oracle_cmd: &str) -> Result<bool, String> {
    #[cfg(windows)]
    let mut cmd = {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd");
        // Pass `/C <oracle_cmd>` verbatim so embedded quotes inside oracle_cmd
        // are not backslash-escaped by Rust's Windows Command argument quoting.
        c.raw_arg(format!("/C {oracle_cmd}"));
        c
    };
    #[cfg(not(windows))]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.args(["-c", oracle_cmd]);
        c
    };
    cmd.current_dir(worktree)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let status = cmd
        .status()
        .map_err(|e| format!("failed to execute oracle_cmd '{oracle_cmd}': {e}"))?;
    Ok(status.success())
}

fn apply_reverse_patch(worktree: &Path, patch: &str) -> bool {
    let Ok(mut child) = Command::new("git")
        .arg("-C")
        .arg(worktree)
        .args(["apply", "-R", "--whitespace=nowarn", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(patch.as_bytes());
    }
    child.wait().is_ok_and(|s| s.success())
}

/// `causal_walk(action="ablate", repoPath=..., commit="HEAD", oracle_cmd=...)`
pub async fn execute_ablate(args: Option<Value>) -> Result<Value, String> {
    let raw = args.unwrap_or_else(|| json!({}));
    let repo = validate_repo_dir(&raw)?;
    let commit_ref = raw
        .get("commit")
        .and_then(Value::as_str)
        .unwrap_or("HEAD")
        .trim()
        .to_string();
    let oracle_cmd = raw
        .get("oracle_cmd")
        .or_else(|| raw.get("oracleCmd"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "causal_walk action='ablate' requires `oracle_cmd` (the deterministic test command that must fail on commit^ and pass on commit)"
                .to_string()
        })?
        .to_string();
    let max_hunks = raw
        .get("max_hunks")
        .or_else(|| raw.get("maxHunks"))
        .and_then(Value::as_u64)
        .map(|n| (n as usize).clamp(1, MAX_ABLATE_HUNKS_CAP))
        .unwrap_or(16);

    tokio::task::spawn_blocking(move || {
        let target_sha = git_stdout(&repo, &["rev-parse", &commit_ref])?
            .trim()
            .to_ascii_lowercase();
        let parent_sha = git_stdout(&repo, &["rev-parse", &format!("{target_sha}^")])?
            .trim()
            .to_ascii_lowercase();

        let guard = WorktreeGuard::create(&repo, &parent_sha)?;

        // Step 1: Tautology / Self-Confirming Test Proof (oracle MUST FAIL on parent commit!).
        let parent_passes = run_oracle(&guard.worktree, &oracle_cmd)?;
        if parent_passes {
            return Ok(json!({
                "tool": "causal_walk",
                "action": "ablate",
                "status": "rejected_tautology",
                "code": "TAUTOLOGICAL_SELF_CONFIRMING_TEST",
                "commit": target_sha,
                "parent": parent_sha,
                "tautology_check": "failed",
                "detail": "oracle_cmd ALREADY PASSED on parent commit (before the fix). This is a self-confirming test and proves zero causal effect for the commit.",
            }));
        }

        // Step 2: Target commit MUST PASS oracle_cmd.
        git_stdout(&guard.worktree, &["checkout", "--quiet", "--force", &target_sha])?;
        let target_passes = run_oracle(&guard.worktree, &oracle_cmd)?;
        if !target_passes {
            return Ok(json!({
                "tool": "causal_walk",
                "action": "ablate",
                "status": "target_failed",
                "code": "TARGET_NOT_GREEN",
                "commit": target_sha,
                "parent": parent_sha,
                "tautology_check": "passed",
                "detail": "oracle_cmd failed on target commit; cannot ablate a non-passing fix.",
            }));
        }

        // Step 3: Split commit diff into atomic single-hunk patches.
        let diff = git_stdout(
            &repo,
            &["diff", "--no-ext-diff", "--no-color", "--unified=3", &parent_sha, &target_sha],
        )?;
        let hunks = split_unified_diff_into_hunks(&diff);
        let truncated_hunks = hunks.len() > max_hunks;
        let tested_hunks: Vec<&AblationHunk> = hunks.iter().take(max_hunks).collect();

        let mut necessary = Vec::new();
        let mut superstitious = Vec::new();
        let mut coupled = Vec::new();

        for hunk in &tested_hunks {
            let _ = git_stdout(&guard.worktree, &["checkout", "--quiet", "--force", &target_sha]);
            let _ = git_stdout(&guard.worktree, &["clean", "-fdq"]);
            if !apply_reverse_patch(&guard.worktree, &hunk.patch) {
                coupled.push(hunk);
                continue;
            }
            let still_passes = run_oracle(&guard.worktree, &oracle_cmd)?;
            if still_passes {
                superstitious.push(*hunk);
            } else {
                necessary.push(*hunk);
            }
        }

        // Step 4: Joint Minimality Verification — strip ALL superstitious hunks together!
        let (joint_minimality_verified, interacting_hunk_warning) = if superstitious.is_empty() {
            (true, false)
        } else {
            let _ = git_stdout(&guard.worktree, &["checkout", "--quiet", "--force", &target_sha]);
            let _ = git_stdout(&guard.worktree, &["clean", "-fdq"]);
            let all_applied = superstitious
                .iter()
                .all(|h| apply_reverse_patch(&guard.worktree, &h.patch));
            if !all_applied {
                (false, true)
            } else {
                let joint_passes = run_oracle(&guard.worktree, &oracle_cmd)?;
                (joint_passes, !joint_passes)
            }
        };

        let digest_input = format!(
            "ablate:{parent_sha}->{target_sha}:nec={}:sup={}:coup={}:joint={joint_minimality_verified}",
            necessary.len(),
            superstitious.len(),
            coupled.len()
        );
        let proof_digest = blake3::hash(digest_input.as_bytes()).to_hex().to_string();

        let hunk_json = |h: &&AblationHunk, verdict: &str| {
            json!({
                "index": h.index,
                "file": h.file,
                "header": h.header,
                "verdict": verdict,
            })
        };

        Ok(json!({
            "tool": "causal_walk",
            "action": "ablate",
            "status": "completed",
            "commit": target_sha,
            "parent": parent_sha,
            "tautology_check": "passed",
            "total_hunks": hunks.len(),
            "tested_hunks": tested_hunks.len(),
            "truncated": truncated_hunks,
            "necessary_hunks": necessary.iter().map(|h| hunk_json(h, "NECESSARY_CAUSE")).collect::<Vec<_>>(),
            "superstitious_hunks": superstitious.iter().map(|h| hunk_json(h, "SUPERSTITIOUS_NOISE")).collect::<Vec<_>>(),
            "coupled_hunks": coupled.iter().map(|h| hunk_json(h, "COUPLED_HUNK")).collect::<Vec<_>>(),
            "joint_minimality_verified": joint_minimality_verified,
            "interacting_hunk_warning": interacting_hunk_warning,
            "proof_digest_blake3": proof_digest,
            "note": "Halpern-Pearl counterfactual hunk ablation in an isolated temporary git worktree: verified oracle RED on parent, GREEN on target, and leave-one-hunk-out necessity per hunk.",
        }))
    })
    .await
    .map_err(|e| format!("ablate task failed: {e}"))?
}

// ---------------------------------------------------------------------------
// 4. Additive Walk Enrichments (Diverse Why-Provenance, Interaction Pairs,
//    Zombie Oscillations, Cryptographic Trust Floor, Env/Toolchain Drift)
// ---------------------------------------------------------------------------

/// Enrich a completed `causal_walk(action="walk")` JSON payload with additive,
/// strictly deterministic frontier analytics without modifying `causes` order.
pub fn enrich_walk_response(storage: &Storage, response: &mut Value) {
    let Some(causes) = response.get("causes").and_then(Value::as_array) else {
        return;
    };
    if causes.is_empty() {
        return;
    }

    let mut nodes_by_id: BTreeMap<String, KnowledgeNode> = BTreeMap::new();
    let mut ordered_cause_ids: Vec<String> = Vec::new();
    for row in causes {
        if let Some(id) = row.get("id").and_then(Value::as_str) {
            ordered_cause_ids.push(id.to_string());
            if let Ok(Some(node)) = storage.get_node(id) {
                nodes_by_id.insert(id.to_string(), node);
            }
        }
    }
    if let Some(starts) = response.get("starts").and_then(Value::as_array) {
        for s in starts {
            if let Some(id) = s.as_str()
                && let Ok(Some(node)) = storage.get_node(id)
            {
                nodes_by_id.insert(id.to_string(), node);
            }
        }
    }

    // 4A. Diverse Why-Provenance: greedy maximally anchor-disjoint top-5 witness slice.
    let diverse_causes = select_diverse_causes(causes, &nodes_by_id, 5);

    // 4B. Multi-Commit Joint-Cause / Interaction Pairs (`TraceVIC` + `TrAC`):
    // distinct commits among reached causes that mutate the same exact `file:` anchor.
    let interaction_pairs = detect_interaction_pairs(&ordered_cause_ids, &nodes_by_id);

    // 4C. Zombie-Bug Oscillation Cycles (`RETRACE`):
    // reached nodes/commits involved in >= 2 `corrects` or `supersedes` flips.
    let all_edges = storage.get_all_connections().unwrap_or_default();
    let zombie_oscillations = detect_zombie_oscillations(&nodes_by_id, &all_edges);

    // 4D. Compaction-Proof Cryptographic Trust Floor Audit:
    // grade every reached cause by deterministic trust tier and flag laundering risk.
    let trust_audit = audit_trust_floor(&ordered_cause_ids, &nodes_by_id, &all_edges);

    // 4E. Environment & Toolchain Drift Detector (`OOM-RL II`):
    // compare exact `env:<key>:<val>` and `cfg:<key>:<val>` tags across reached nodes.
    let env_drift = detect_env_drift(&nodes_by_id);

    if let Some(obj) = response.as_object_mut() {
        obj.insert("diverse_causes".to_string(), json!(diverse_causes));
        obj.insert("interaction_pairs".to_string(), json!(interaction_pairs));
        obj.insert("zombie_oscillations".to_string(), json!(zombie_oscillations));
        obj.insert("trust_audit".to_string(), trust_audit);
        obj.insert("env_drift".to_string(), json!(env_drift));
    }
}

fn anchor_set(node: &KnowledgeNode) -> BTreeSet<String> {
    node.tags
        .iter()
        .filter(|t| {
            t.starts_with("file:")
                || t.starts_with("pkg:")
                || t.starts_with("env:")
                || t.starts_with("cfg:")
                || t.starts_with("commit:")
        })
        .cloned()
        .collect()
}

fn select_diverse_causes(
    causes: &[Value],
    nodes_by_id: &BTreeMap<String, KnowledgeNode>,
    k: usize,
) -> Vec<Value> {
    let mut covered_anchors: HashSet<String> = HashSet::new();
    let mut chosen_indices: Vec<usize> = Vec::new();
    let mut used = vec![false; causes.len()];

    for _ in 0..k.min(causes.len()) {
        let mut best_idx: Option<usize> = None;
        let mut best_gain: usize = 0;

        for (idx, row) in causes.iter().enumerate() {
            if used[idx] {
                continue;
            }
            let id = row.get("id").and_then(Value::as_str).unwrap_or("");
            let gain = nodes_by_id
                .get(id)
                .map(|n| {
                    anchor_set(n)
                        .into_iter()
                        .filter(|a| !covered_anchors.contains(a))
                        .count()
                })
                .unwrap_or(0);
            if best_idx.is_none() || gain > best_gain {
                best_idx = Some(idx);
                best_gain = gain;
            }
        }
        let Some(pick) = best_idx else {
            break;
        };
        if !chosen_indices.is_empty() && best_gain == 0 {
            break;
        }
        used[pick] = true;
        chosen_indices.push(pick);
        if let Some(id) = causes[pick].get("id").and_then(Value::as_str)
            && let Some(node) = nodes_by_id.get(id)
        {
            for a in anchor_set(node) {
                covered_anchors.insert(a);
            }
        }
    }

    chosen_indices
        .into_iter()
        .filter_map(|idx| {
            let row = &causes[idx];
            let id = row.get("id").and_then(Value::as_str)?;
            let anchors: Vec<String> = nodes_by_id
                .get(id)
                .map(|n| anchor_set(n).into_iter().collect())
                .unwrap_or_default();
            Some(json!({
                "id": id,
                "rank_index": idx,
                "distinct_anchors": anchors,
            }))
        })
        .collect()
}

fn detect_interaction_pairs(
    cause_ids: &[String],
    nodes_by_id: &BTreeMap<String, KnowledgeNode>,
) -> Vec<Value> {
    let mut file_to_commits: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for id in cause_ids {
        let Some(node) = nodes_by_id.get(id) else {
            continue;
        };
        let Some(sha) = extract_commit_sha_from_node(node) else {
            continue;
        };
        for tag in &node.tags {
            if let Some(file) = tag.strip_prefix("file:") {
                file_to_commits
                    .entry(file.to_string())
                    .or_default()
                    .push((id.clone(), sha.clone()));
            }
        }
    }

    let mut pairs: BTreeMap<(String, String), (String, String, BTreeSet<String>)> = BTreeMap::new();
    for (file, commits) in file_to_commits {
        for i in 0..commits.len() {
            for j in (i + 1)..commits.len() {
                let (id_a, sha_a) = &commits[i];
                let (id_b, sha_b) = &commits[j];
                if sha_a == sha_b {
                    continue;
                }
                let key = if id_a <= id_b {
                    (id_a.clone(), id_b.clone())
                } else {
                    (id_b.clone(), id_a.clone())
                };
                let (sa, sb) = if id_a <= id_b {
                    (sha_a.clone(), sha_b.clone())
                } else {
                    (sha_b.clone(), sha_a.clone())
                };
                pairs
                    .entry(key)
                    .or_insert_with(|| (sa, sb, BTreeSet::new()))
                    .2
                    .insert(file.clone());
            }
        }
    }

    pairs
        .into_iter()
        .take(10)
        .map(|((id_a, id_b), (sha_a, sha_b, files))| {
            json!({
                "commit_a": sha_a,
                "commit_b": sha_b,
                "memory_a": id_a,
                "memory_b": id_b,
                "shared_files": files.into_iter().collect::<Vec<_>>(),
                "hypothesis": "Multi-commit interaction candidate (TraceVIC/TrAC): both commits mutated overlapping file anchor(s); test joint contingency if single-commit bisect is inconclusive.",
            })
        })
        .collect()
}

fn detect_zombie_oscillations(
    nodes_by_id: &BTreeMap<String, KnowledgeNode>,
    edges: &[ConnectionRecord],
) -> Vec<Value> {
    let mut file_flips: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for edge in edges {
        if !matches!(edge.link_type.as_str(), "corrects" | "supersedes") {
            continue;
        }
        let Some(src) = nodes_by_id.get(&edge.source_id) else {
            continue;
        };
        let Some(dst) = nodes_by_id.get(&edge.target_id) else {
            continue;
        };
        let dst_files: HashSet<&str> = dst
            .tags
            .iter()
            .filter_map(|t| t.strip_prefix("file:"))
            .collect();
        for f in src.tags.iter().filter_map(|t| t.strip_prefix("file:")) {
            if dst_files.contains(f) {
                file_flips.entry(f.to_string()).or_default().push(json!({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "edge": edge.link_type,
                }));
            }
        }
    }

    file_flips
        .into_iter()
        .filter(|(_, flips)| flips.len() >= 2)
        .map(|(file, flips)| {
            json!({
                "file": file,
                "flip_count": flips.len(),
                "flips": flips,
                "warning": "RETRACE oscillation cycle detected: >= 2 revert/supersede flips on the same file anchor.",
            })
        })
        .collect()
}

fn audit_trust_floor(
    cause_ids: &[String],
    nodes_by_id: &BTreeMap<String, KnowledgeNode>,
    edges: &[ConnectionRecord],
) -> Value {
    let mut tiers: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut untrusted_ids: HashSet<&str> = HashSet::new();
    for id in cause_ids {
        if let Some(node) = nodes_by_id.get(id) {
            let tier = trust_tier(node);
            *tiers.entry(tier).or_insert(0) += 1;
            if tier == "untrusted_external" {
                untrusted_ids.insert(id.as_str());
            }
        }
    }

    let mut laundering_alerts = Vec::new();
    if !untrusted_ids.is_empty() {
        for edge in edges {
            if LINEAGE_LINKS.contains(&edge.link_type.as_str())
                && untrusted_ids.contains(edge.target_id.as_str())
                && let Some(downstream) = nodes_by_id.get(&edge.source_id)
                && matches!(
                    trust_tier(downstream),
                    "owner_pinned" | "agent_derived"
                )
            {
                laundering_alerts.push(json!({
                    "code": "UNTRUSTED_UPSTREAM_LAUNDERING_RISK",
                    "untrusted_source_id": edge.target_id,
                    "downstream_id": edge.source_id,
                    "downstream_tier": trust_tier(downstream),
                    "edge": edge.link_type,
                }));
            }
        }
    }

    json!({
        "tier_counts": tiers,
        "laundering_risk": !laundering_alerts.is_empty(),
        "laundering_alerts": laundering_alerts,
    })
}

fn detect_env_drift(nodes_by_id: &BTreeMap<String, KnowledgeNode>) -> Vec<Value> {
    // Tags of form `env:<key>:<value>` or `cfg:<key>:<value>`.
    let mut key_values: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    for (id, node) in nodes_by_id {
        for tag in &node.tags {
            let Some((prefix, rest)) = tag.split_once(':') else {
                continue;
            };
            if !matches!(prefix, "env" | "cfg") {
                continue;
            }
            let Some((key, val)) = rest.split_once(':') else {
                continue;
            };
            if key.is_empty() || val.is_empty() {
                continue;
            }
            key_values
                .entry(format!("{prefix}:{key}"))
                .or_default()
                .entry(val.to_string())
                .or_default()
                .push(id.clone());
        }
    }

    key_values
        .into_iter()
        .filter(|(_, vals)| vals.len() >= 2)
        .map(|(key, vals)| {
            let distinct: Vec<Value> = vals
                .into_iter()
                .map(|(v, ids)| json!({"value": v, "node_ids": ids}))
                .collect();
            json!({
                "key": key,
                "distinct_values": distinct,
                "warning": "OOM-RL II environment/config drift detected across causal chain.",
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 5. Git-Native Zombie-Patch Guard, Reflex PreToolUse Fence Hook & PR Audit Bot
// ---------------------------------------------------------------------------

/// Extract every 40-hex commit SHA named by git's standard `This reverts commit <sha>`
/// trailer across the most recent 250 revert commits in `repo`.
fn git_reverted_commits(repo: &Path) -> Vec<(String, String)> {
    const SEP: &str = "---VESTIGE-COMMIT-END---";
    let Ok(log) = git_stdout(
        repo,
        &[
            "log",
            "-n",
            "250",
            "--grep=This reverts commit ",
            &format!("--format=%H%n%B%n{SEP}"),
        ],
    ) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for block in log.split(SEP) {
        let mut lines = block.trim().lines();
        let Some(revert_sha) = lines
            .next()
            .map(str::trim)
            .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        else {
            continue;
        };
        let revert_lower = revert_sha.to_ascii_lowercase();
        for line in lines {
            if let Some(rest) = line.trim().strip_prefix("This reverts commit ") {
                let target = rest.trim_end_matches('.').trim();
                if target.len() == 40
                    && target.bytes().all(|b| b.is_ascii_hexdigit())
                    && seen.insert(target.to_ascii_lowercase())
                {
                    out.push((target.to_ascii_lowercase(), revert_lower.clone()));
                }
            }
        }
    }
    out
}

/// Match `proposed_diff` against every previously reverted commit in `repo` using
/// canonical `git patch-id --stable`. Works offline with zero Strata store required
/// (powers both `vestige hook pre-commit` and the GitHub Causal Gate PR Bot).
pub fn git_native_zombie_matches(repo: &Path, proposed_diff: &str) -> (Option<String>, Vec<Value>) {
    let Some(proposed_pid) = git_patch_id_stable(repo, proposed_diff) else {
        return (None, Vec::new());
    };
    let mut matches = Vec::new();
    for (reverted_sha, revert_by_sha) in git_reverted_commits(repo) {
        if let Ok(commit_diff) = git_stdout(repo, &["show", "--format=", &reverted_sha])
            && let Some(tomb_pid) = git_patch_id_stable(repo, &commit_diff)
            && tomb_pid == proposed_pid
        {
            matches.push(json!({
                "code": "ZOMBIE_REGRESSION_DETECTED",
                "patch_id": proposed_pid,
                "reverted_commit": reverted_sha,
                "reverted_by_commit": revert_by_sha,
                "reason": format!("matches canonical git patch-id of commit {} previously reverted by {}", &reverted_sha[..8], &revert_by_sha[..8]),
                "action": "STOP — this diff re-introduces a previously reverted commit. Inspect the revert commit before proceeding.",
            }));
        }
    }
    (Some(proposed_pid), matches)
}

/// Synchronous, low-latency Chesterton's Fence check invoked inside Reflex `PreToolUse`
/// right before an agent calls `Edit` / `Write` / `edit_file` / `write_file`.
pub fn pre_edit_fence_summary(
    storage: &Storage,
    file: &str,
    cwd: Option<&Path>,
) -> Option<String> {
    let base = cwd.or_else(|| Path::new(file).parent())?;
    if !base.exists() {
        return None;
    }
    let repo_root_str = git_stdout(base, &["rev-parse", "--show-toplevel"]).ok()?;
    let repo_root = PathBuf::from(repo_root_str.trim());
    let file_path = Path::new(file);
    let rel = if file_path.is_absolute() {
        file_path
            .strip_prefix(&repo_root)
            .ok()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|| file.replace('\\', "/"))
    } else {
        file.trim_start_matches("./").replace('\\', "/")
    };

    let target = parse_fence_target(&rel)?;
    let blamed = blame_target(&repo_root, &target).unwrap_or_default();
    let mut seed_ids = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for b in blamed.iter().take(8) {
        let short = &b.sha[..b.sha.len().min(7)];
        for handle in [format!("commit:{}", b.sha), format!("commit:{short}")] {
            for id in storage.resolve_handle(&handle).ids {
                if seen.insert(id.clone()) {
                    seed_ids.push(id);
                }
            }
        }
    }
    for id in storage.resolve_handle(&format!("file:{}", target.path)).ids {
        if seen.insert(id.clone()) {
            seed_ids.push(id);
        }
    }
    if seed_ids.is_empty() {
        return None;
    }
    let all_edges = storage.get_all_connections().ok()?;
    let guarded = forward_fence_walk(storage, "user", &seed_ids, &all_edges);
    let mut critical_ids = Vec::new();
    for (node, via, _) in &guarded {
        let is_pinned = node.tags.iter().any(|t| t == "pinned");
        let is_fix = matches!(via.as_str(), "closes" | "closed_by" | "corrects");
        let is_human = actor_provenance(node) == "human-git";
        if (is_pinned || is_fix || is_human) && !critical_ids.contains(&node.id) {
            critical_ids.push(node.id.clone());
        }
    }
    if critical_ids.is_empty() {
        return None;
    }
    let preview: Vec<&str> = critical_ids.iter().take(3).map(String::as_str).collect();
    Some(format!(
        "Chesterton's Fence [CRITICAL_FENCE]: `{}` is guarded by {} pinned/fix/human Strata invariant(s) [{}]. Run causal_walk(action='fence') before altering guarded lines.",
        target.path,
        critical_ids.len(),
        preview.join(", ")
    ))
}

/// Configuration for `vestige pr-audit` (the engine behind the Vestige Causal Gate GitHub Bot).
#[derive(Debug, Clone)]
pub struct PrAuditConfig {
    pub repo: PathBuf,
    pub base: String,
    pub head: String,
    pub oracle_cmd: Option<String>,
    pub max_hunks: usize,
}

/// Deterministic Pull Request Causal Audit Report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PrAuditReport {
    pub base_sha: String,
    pub head_sha: String,
    pub commits_audited: usize,
    pub files_touched: usize,
    pub proof_notes_present: usize,
    pub blocked_by_zombie: bool,
    pub zombie_matches: Vec<Value>,
    pub chesterton_fences: Vec<Value>,
    pub ablation: Option<Value>,
    pub proof_digest_blake3: String,
}

/// Parse old-side modified/deleted line ranges from `git diff -U0 base..head`
/// so we can blame the exact lines at `base` that this PR is overwriting or deleting.
fn parse_old_side_ranges(diff: &str) -> Vec<(String, Vec<(u32, u32)>)> {
    let mut files: Vec<(String, Vec<(u32, u32)>)> = Vec::new();
    let mut current_path = String::new();
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("--- ") {
            current_path = rest.strip_prefix("a/").unwrap_or("").to_string();
            if !current_path.is_empty() && !files.iter().any(|(p, _)| p == &current_path) {
                files.push((current_path.clone(), Vec::new()));
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("@@ ")
            && !current_path.is_empty()
            && let Some(minus) = rest.split_whitespace().find(|w| w.starts_with('-'))
        {
            let spec = &minus[1..];
            let (start, count) = match spec.split_once(',') {
                Some((s, c)) => (s.parse::<u32>().ok(), c.parse::<u32>().ok()),
                None => (spec.parse::<u32>().ok(), Some(1)),
            };
            if let (Some(s), Some(c)) = (start, count)
                && s > 0
                && c > 0
                && let Some((_, ranges)) = files.last_mut()
            {
                ranges.push((s, s.saturating_add(c.saturating_sub(1))));
            }
        }
    }
    files
}

fn commit_subject_is_guard(subject: &str) -> bool {
    // hardcode-scan: protocol (conventional-commit & git trailer guard markers)
    const GUARD_PREFIXES: &[&str] = &[
        "fix", "revert", "hotfix", "security", "sec(", "bugfix", "fix(",
    ];
    let lower = subject.trim().to_ascii_lowercase();
    GUARD_PREFIXES.iter().any(|p| lower.starts_with(p))
        || lower.contains("fixes #")
        || lower.contains("closes #")
        || lower.contains("regression")
}

/// Run the full deterministic Pull Request Causal Audit (`base..head`).
/// Works offline with zero LLM calls, zero vectors, and zero external APIs.
pub fn run_pr_audit(
    storage: Option<&Storage>,
    cfg: &PrAuditConfig,
) -> Result<PrAuditReport, String> {
    if !cfg.repo.is_dir() {
        return Err(format!("repo '{}' is not a directory", cfg.repo.display()));
    }
    let base_sha = git_stdout(&cfg.repo, &["rev-parse", &cfg.base])?
        .trim()
        .to_ascii_lowercase();
    let head_sha = git_stdout(&cfg.repo, &["rev-parse", &cfg.head])?
        .trim()
        .to_ascii_lowercase();
    let range = format!("{base_sha}..{head_sha}");

    let commit_shas: Vec<String> = git_stdout(&cfg.repo, &["rev-list", "--no-merges", &range])
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|s| s.len() == 40)
        .map(|s| s.to_ascii_lowercase())
        .collect();

    // 1. Zombie-Patch Scan: check every commit in PR + the net squashed diff!
    let mut zombie_matches = Vec::new();
    let mut seen_zombie: HashSet<(String, String)> = HashSet::new();
    let all_edges = storage
        .and_then(|st| st.get_all_connections().ok())
        .unwrap_or_default();

    let mut check_diff = |label: &str, diff_text: &str| {
        let (_, mut found) = git_native_zombie_matches(&cfg.repo, diff_text);
        if let Some(st) = storage {
            let (_, strata_found) = detect_zombie_patch(st, &cfg.repo, &all_edges, diff_text);
            found.extend(strata_found);
        }
        for mut z in found {
            let rev = z["reverted_commit"].as_str().unwrap_or("").to_string();
            if seen_zombie.insert((label.to_string(), rev)) {
                if let Some(obj) = z.as_object_mut() {
                    obj.insert("offending_source".to_string(), json!(label));
                }
                zombie_matches.push(z);
            }
        }
    };

    for sha in &commit_shas {
        if let Ok(cdiff) = git_stdout(&cfg.repo, &["show", "--format=", sha]) {
            check_diff(&sha[..8], &cdiff);
        }
    }
    if let Ok(net_diff) = git_stdout(
        &cfg.repo,
        &["diff", "--no-ext-diff", "--no-color", &base_sha, &head_sha],
    ) {
        check_diff("pr-net-diff", &net_diff);
    }

    // 2. Forward Chesterton's Fence Audit on old-side overwritten/deleted lines at `base_sha`.
    let diff_u0 = git_stdout(
        &cfg.repo,
        &["diff", "-U0", "--no-ext-diff", "--no-color", &base_sha, &head_sha],
    )
    .unwrap_or_default();
    let touched_old = parse_old_side_ranges(&diff_u0);
    let files_touched = touched_old.len();
    let mut chesterton_fences = Vec::new();
    let mut seen_fence: HashSet<(String, String)> = HashSet::new();

    for (file, ranges) in touched_old.iter().take(24) {
        for &(start, end) in ranges.iter().take(4) {
            let mut blame_args: Vec<String> = vec![
                "blame".into(),
                "--porcelain".into(),
                base_sha.clone(),
                "-L".into(),
                format!("{start},{end}"),
                "--".into(),
                file.clone(),
            ];
            let refs: Vec<&str> = blame_args.iter().map(String::as_str).collect();
            let Ok(blame_out) = git_stdout(&cfg.repo, &refs) else {
                blame_args.clear();
                continue;
            };
            let mut blamed_shas: BTreeSet<String> = BTreeSet::new();
            for line in blame_out.lines() {
                if let Some(tok) = line.split_whitespace().next()
                    && tok.len() == 40
                    && tok.bytes().all(|b| b.is_ascii_hexdigit())
                    && !tok.bytes().all(|b| b == b'0')
                {
                    blamed_shas.insert(tok.to_ascii_lowercase());
                }
            }
            for sha in blamed_shas {
                if !seen_fence.insert((file.clone(), sha.clone())) {
                    continue;
                }
                let meta = git_stdout(&cfg.repo, &["show", "-s", "--format=%an\t%s", &sha])
                    .unwrap_or_default();
                let (author, subject) = meta
                    .trim()
                    .split_once('\t')
                    .map(|(a, s)| (a.to_string(), s.to_string()))
                    .unwrap_or_else(|| ("unknown".to_string(), String::new()));

                let mut strata_guards: Vec<String> = Vec::new();
                if let Some(st) = storage {
                    let short = &sha[..7];
                    for handle in [format!("commit:{sha}"), format!("commit:{short}")] {
                        for id in st.resolve_handle(&handle).ids {
                            if !strata_guards.contains(&id) {
                                strata_guards.push(id);
                            }
                        }
                    }
                }

                if commit_subject_is_guard(&subject) || !strata_guards.is_empty() {
                    chesterton_fences.push(json!({
                        "file": file,
                        "lines_at_base": format!("{start}-{end}"),
                        "blamed_commit": sha,
                        "author": author,
                        "subject": subject,
                        "strata_records": strata_guards,
                        "severity": if !strata_guards.is_empty() || subject.to_ascii_lowercase().starts_with("revert") {
                            "CRITICAL_FENCE"
                        } else {
                            "GUARDED_FENCE"
                        },
                    }));
                }
            }
        }
    }

    // 3. Signed `refs/notes/vestige-proof` presence check across PR commits.
    let mut proof_notes_present = 0usize;
    for sha in &commit_shas {
        if git_stdout(&cfg.repo, &["notes", "--ref=refs/notes/vestige-proof", "show", sha]).is_ok() {
            proof_notes_present += 1;
        }
    }

    // 4. Optional Halpern-Pearl Counterfactual Hunk Ablation across HEAD commit.
    let ablation = if let Some(ref oracle) = cfg.oracle_cmd {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("failed to build runtime for ablation: {e}"))?;
        let val = rt.block_on(execute_ablate(Some(json!({
            "repoPath": cfg.repo.display().to_string(),
            "commit": head_sha,
            "oracle_cmd": oracle,
            "max_hunks": cfg.max_hunks,
        }))))?;
        Some(val)
    } else {
        None
    };

    let blocked_by_zombie = !zombie_matches.is_empty();
    let digest_input = format!(
        "pr-audit:{base_sha}..{head_sha}:commits={}:files={files_touched}:zombies={}:fences={}:notes={proof_notes_present}",
        commit_shas.len(),
        zombie_matches.len(),
        chesterton_fences.len(),
    );
    let proof_digest_blake3 = blake3::hash(digest_input.as_bytes()).to_hex().to_string();

    Ok(PrAuditReport {
        base_sha,
        head_sha,
        commits_audited: commit_shas.len(),
        files_touched,
        proof_notes_present,
        blocked_by_zombie,
        zombie_matches,
        chesterton_fences,
        ablation,
        proof_digest_blake3,
    })
}

/// Render a deterministic, proof-carrying Markdown PR comment for the Vestige Causal Gate Bot.
pub fn render_pr_audit_markdown(report: &PrAuditReport) -> String {
    let status_badge = if report.blocked_by_zombie {
        "🔴 **BLOCKED — ZOMBIE REGRESSION DETECTED**"
    } else if !report.chesterton_fences.is_empty() {
        "🟡 **PASSED WITH CHESTERTON'S FENCE WARNINGS**"
    } else {
        "🟢 **VERIFIED CLEAN**"
    };

    let mut md = String::new();
    md.push_str("<!-- vestige-causal-gate -->\n");
    md.push_str("## ⚡ Vestige Causal Gate — Deterministic Proof Card\n\n");
    md.push_str(&format!("**Status:** {status_badge}\n\n"));
    md.push_str("| Metric | Value |\n| :--- | :--- |\n");
    md.push_str(&format!(
        "| **Range** | `{}..{}` ({} commits, {} files) |\n",
        &report.base_sha[..report.base_sha.len().min(8)],
        &report.head_sha[..report.head_sha.len().min(8)],
        report.commits_audited,
        report.files_touched
    ));
    md.push_str(&format!(
        "| **Zombie Regressions (`patch-id --stable`)** | **{}** |\n",
        report.zombie_matches.len()
    ));
    md.push_str(&format!(
        "| **Chesterton's Fence Guards Touched** | **{}** |\n",
        report.chesterton_fences.len()
    ));
    md.push_str(&format!(
        "| **Signed `refs/notes/vestige-proof` Commits** | **{}/{}** |\n",
        report.proof_notes_present, report.commits_audited
    ));
    md.push_str(&format!(
        "| **BLAKE3 Audit Digest** | `{}` |\n\n",
        &report.proof_digest_blake3[..24]
    ));

    if !report.zombie_matches.is_empty() {
        md.push_str("### 🚨 Zombie-Regression Tombstone Hits (`git patch-id --stable`)\n\n");
        md.push_str("| PR Source | Re-Introduced Commit | Originally Reverted By | Canonical Patch-ID |\n| :--- | :--- | :--- | :--- |\n");
        for z in &report.zombie_matches {
            let src = z["offending_source"].as_str().unwrap_or("?");
            let rev = z["reverted_commit"].as_str().unwrap_or("?");
            let by = z["reverted_by_commit"]
                .as_str()
                .or_else(|| z["tombstone_memory_id"].as_str())
                .unwrap_or("?");
            let pid = z["patch_id"].as_str().unwrap_or("?");
            md.push_str(&format!(
                "| `{src}` | `{}` | `{}` | `{}` |\n",
                &rev[..rev.len().min(8)],
                &by[..by.len().min(12)],
                &pid[..pid.len().min(12)]
            ));
        }
        md.push('\n');
    }

    if !report.chesterton_fences.is_empty() {
        md.push_str("### 🛡️ Forward Chesterton's Fence Audit (Prior Fixes/Reverts Modified)\n\n");
        md.push_str("| Severity | File & Lines (`base`) | Blamed Commit | Author | Original Guard Subject |\n| :--- | :--- | :--- | :--- | :--- |\n");
        for f in report.chesterton_fences.iter().take(20) {
            let sev = f["severity"].as_str().unwrap_or("GUARDED_FENCE");
            let file = f["file"].as_str().unwrap_or("?");
            let lines = f["lines_at_base"].as_str().unwrap_or("?");
            let sha = f["blamed_commit"].as_str().unwrap_or("?");
            let author = f["author"].as_str().unwrap_or("?");
            let subj = f["subject"].as_str().unwrap_or("").replace('|', "\\|");
            md.push_str(&format!(
                "| `{sev}` | `{file}:{lines}` | `{}` | {author} | {subj} |\n",
                &sha[..sha.len().min(8)]
            ));
        }
        md.push('\n');
    }

    if let Some(ref abl) = report.ablation {
        md.push_str("### 🔬 Halpern–Pearl Counterfactual Hunk Ablation\n\n");
        let status = abl["status"].as_str().unwrap_or("unknown");
        let taut = abl["tautology_check"].as_str().unwrap_or("unknown");
        let nec = abl["necessary_hunks"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0);
        let sup = abl["superstitious_hunks"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0);
        md.push_str(&format!(
            "- **Status:** `{status}` · **Tautology Check:** `{taut}` · **Necessary Hunks:** `{nec}` · **Superstitious Noise Hunks:** `{sup}`\n\n"
        ));
    }

    md.push_str("---\n*Computed 100% locally by **Vestige** — zero LLMs, zero vectors, zero cloud, \\$0.00.*\n");
    md
}

/// Render a SARIF 2.1.0 report for native GitHub Code Scanning inline PR annotations & Security tab.
pub fn render_pr_audit_sarif(report: &PrAuditReport) -> Value {
    let mut results: Vec<Value> = Vec::new();
    for z in &report.zombie_matches {
        let src = z["offending_source"].as_str().unwrap_or("PR");
        let rev = z["reverted_commit"].as_str().unwrap_or("?");
        let pid = z["patch_id"].as_str().unwrap_or("?");
        let uri = z["file"].as_str().unwrap_or("README.md");
        results.push(json!({
            "ruleId": "VST-ZOMBIE-001",
            "level": "error",
            "message": {
                "text": format!(
                    "Zombie regression detected: `{src}` matches canonical `git patch-id --stable` ({}) of previously reverted commit `{}`.",
                    &pid[..pid.len().min(12)],
                    &rev[..rev.len().min(8)]
                )
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": { "uri": uri },
                    "region": { "startLine": 1 }
                }
            }]
        }));
    }
    for f in &report.chesterton_fences {
        let file = f["file"].as_str().unwrap_or("unknown");
        let lines = f["lines_at_base"].as_str().unwrap_or("1");
        let start_line = lines
            .split('-')
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1)
            .max(1);
        let sev = f["severity"].as_str().unwrap_or("GUARDED_FENCE");
        let sha = f["blamed_commit"].as_str().unwrap_or("?");
        let subj = f["subject"].as_str().unwrap_or("");
        results.push(json!({
            "ruleId": "VST-FENCE-001",
            "level": if sev == "CRITICAL_FENCE" { "error" } else { "warning" },
            "message": {
                "text": format!(
                    "Chesterton's Fence ({sev}): modifies lines originally guarded by `{}` ({subj}).",
                    &sha[..sha.len().min(8)]
                )
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": { "uri": file },
                    "region": { "startLine": start_line }
                }
            }]
        }));
    }
    json!({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "Vestige Causal Gate",
                    "version": env!("CARGO_PKG_VERSION"),
                    "informationUri": "https://github.com/samvallad33/vestige",
                    "rules": [
                        {
                            "id": "VST-ZOMBIE-001",
                            "name": "ZombieRegressionPatchId",
                            "shortDescription": { "text": "Commit re-introduces a previously reverted patch-id" },
                            "defaultConfiguration": { "level": "error" }
                        },
                        {
                            "id": "VST-FENCE-001",
                            "name": "ChestertonsFenceGuardTouched",
                            "shortDescription": { "text": "PR modifies code lines blamed to a prior fix, revert, or pinned Strata guard" },
                            "defaultConfiguration": { "level": "warning" }
                        }
                    ]
                }
            },
            "properties": {
                "baseSha": report.base_sha,
                "headSha": report.head_sha,
                "blockedByZombie": report.blocked_by_zombie,
                "proofDigestBlake3": report.proof_digest_blake3
            },
            "results": results
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::{ConnectionRecord, IngestInput};

    fn open_store() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    fn save_node(storage: &Arc<Storage>, scope: &str, content: &str, node_type: &str, tags: &[&str]) -> String {
        storage
            .ingest_in_scope(
                IngestInput {
                    content: content.to_string(),
                    node_type: node_type.to_string(),
                    tags: tags.iter().map(|s| (*s).to_string()).collect(),
                    source: Some("test".to_string()),
                    ..Default::default()
                },
                scope,
            )
            .unwrap()
            .id
    }

    fn add_edge(storage: &Arc<Storage>, src: &str, dst: &str, kind: &str) {
        let now = chrono::Utc::now();
        storage
            .save_connection(&ConnectionRecord {
                source_id: src.to_string(),
                target_id: dst.to_string(),
                link_type: kind.to_string(),
                strength: 1.0,
                created_at: now,
                last_activated: now,
                activation_count: 0,
            })
            .unwrap();
    }

    fn git_run(repo: &Path, args: &[&str]) {
        let st = Command::new("git")
            .current_dir(repo)
            .args(args)
            .status()
            .unwrap();
        assert!(st.success(), "git {args:?} failed");
    }

    #[tokio::test]
    async fn fence_warns_on_blamed_pinned_guard_and_blocks_zombie_revert_patch() {
        let (storage, _sdir) = open_store();
        let repo = tempfile::tempdir().unwrap();
        git_run(repo.path(), &["init", "-q"]);
        git_run(repo.path(), &["config", "user.email", "sam@example.com"]);
        git_run(repo.path(), &["config", "user.name", "Sam"]);
        std::fs::create_dir_all(repo.path().join("src")).unwrap();
        std::fs::write(repo.path().join("src/guard.rs"), "pub fn guard() -> bool { true }\n").unwrap();
        git_run(repo.path(), &["add", "src/guard.rs"]);
        git_run(repo.path(), &["commit", "-q", "-m", "add critical guard"]);
        let guard_sha = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();

        // Bad commit that weakens guard, then gets reverted!
        std::fs::write(repo.path().join("src/guard.rs"), "pub fn guard() -> bool { false }\n").unwrap();
        git_run(repo.path(), &["add", "src/guard.rs"]);
        git_run(repo.path(), &["commit", "-q", "-m", "bad weaken guard"]);
        let bad_sha = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();
        let bad_diff = git_stdout(repo.path(), &["show", "--format=", &bad_sha]).unwrap();
        git_run(repo.path(), &["revert", "--no-edit", &bad_sha]);

        // Record Strata memories: guard commit -> pinned decision closing failure; bad commit corrected by revert.
        let guard_commit_id = save_node(
            &storage,
            "user",
            "Commit adding guard",
            "event",
            &[&format!("commit:{guard_sha}"), "file:src/guard.rs", "actor:human-git"],
        );
        let pinned_rule_id = save_node(
            &storage,
            "user",
            "OWNER DIRECTIVE: never return false from guard()",
            "decision",
            &["pinned", "file:src/guard.rs"],
        );
        add_edge(&storage, &pinned_rule_id, &guard_commit_id, "derived_from");

        let bad_mem_id = save_node(
            &storage,
            "user",
            "Bad commit weakening guard",
            "event",
            &[&format!("commit:{bad_sha}"), "file:src/guard.rs"],
        );
        let revert_mem_id = save_node(&storage, "user", "Reverted bad weaken guard", "event", &["revert"]);
        add_edge(&storage, &revert_mem_id, &bad_mem_id, "corrects");

        // Now ask `fence` BEFORE editing src/guard.rs:1 AND pass the zombie bad_diff!
        let out = execute_fence(
            &storage,
            Some(json!({
                "repoPath": repo.path().display().to_string(),
                "targets": ["src/guard.rs:1-1"],
                "proposed_diff": bad_diff,
            })),
        )
        .await
        .expect("fence");

        assert_eq!(out["severity"], json!("ZOMBIE_REGRESSION_BLOCKED"), "{out}");
        assert_eq!(out["blocked_by_zombie"], json!(true), "{out}");
        assert_eq!(
            out["zombie_matches"][0]["code"],
            json!("ZOMBIE_REGRESSION_DETECTED"),
            "{out}"
        );
        assert_eq!(out["fences"][0]["severity"], json!("CRITICAL_FENCE"), "{out}");
        assert!(
            out["fences"][0]["guarding_records"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["id"] == pinned_rule_id && r["pinned"] == json!(true)),
            "Chesterton's Fence must surface the pinned owner directive guarding src/guard.rs:1: {out}"
        );
    }

    #[tokio::test]
    async fn why_not_proves_touched_cap_scope_mismatch_and_reversed_direction() {
        let (storage, _sdir) = open_store();
        let root = save_node(&storage, "user", "root cause", "event", &["file:src/a.rs"]);
        let mid = save_node(&storage, "user", "mid record", "event", &["file:src/a.rs"]);
        let leaf = save_node(&storage, "user", "leaf symptom", "event", &["file:src/a.rs"]);
        add_edge(&storage, &leaf, &mid, "touched");
        add_edge(&storage, &mid, &root, "touched");

        // 1. Two-hop touched chain -> proves TOUCHED_EDGE_CAP_OBSTRUCTION naming exact cut edge!
        let cap = execute_why_not(
            &storage,
            Some(json!({"from_handle": leaf, "expected_id": root, "scope": "user"})),
        )
        .await
        .unwrap();
        assert_eq!(cap["obstruction_code"], json!("TOUCHED_EDGE_CAP_OBSTRUCTION"), "{cap}");
        assert_eq!(cap["blocking_cut_edge"]["from"], json!(mid), "{cap}");
        assert_eq!(cap["blocking_cut_edge"]["to"], json!(root), "{cap}");

        // 2. Scope mismatch -> proves SCOPE_MISMATCH!
        let other = save_node(&storage, "other-project", "foreign cause", "event", &[]);
        let sc = execute_why_not(
            &storage,
            Some(json!({"from_handle": leaf, "expected_id": other, "scope": "user"})),
        )
        .await
        .unwrap();
        assert_eq!(sc["obstruction_code"], json!("SCOPE_MISMATCH"), "{sc}");

        // 3. Reversed direction (asking why downstream child isn't an upstream cause of parent)
        let parent = save_node(&storage, "user", "parent commit", "event", &[]);
        let child = save_node(&storage, "user", "child fix", "decision", &[]);
        add_edge(&storage, &child, &parent, "derived_from");
        let rev = execute_why_not(
            &storage,
            Some(json!({"from_handle": parent, "expected_id": child, "scope": "user"})),
        )
        .await
        .unwrap();
        assert_eq!(rev["obstruction_code"], json!("DIRECTION_REVERSED_OBSTRUCTION"), "{rev}");
    }

    #[tokio::test]
    async fn ablate_separates_necessary_hunk_from_superstitious_noise_and_catches_tautology() {
        let repo = tempfile::tempdir().unwrap();
        git_run(repo.path(), &["init", "-q"]);
        git_run(repo.path(), &["config", "user.email", "dev@example.com"]);
        git_run(repo.path(), &["config", "user.name", "Dev"]);

        // Parent commit: real_value is BROKEN (0), noise_comment is untouched.
        std::fs::write(repo.path().join("real.txt"), "BROKEN\n").unwrap();
        std::fs::write(repo.path().join("noise.txt"), "hello\n").unwrap();
        git_run(repo.path(), &["add", "."]);
        git_run(repo.path(), &["commit", "-q", "-m", "parent broken"]);

        // Fix commit: 1 real necessary hunk (real.txt -> FIXED) + 1 superstitious noise hunk (noise.txt)!
        std::fs::write(repo.path().join("real.txt"), "FIXED\n").unwrap();
        std::fs::write(repo.path().join("noise.txt"), "hello_superstitious_sleep\n").unwrap();
        git_run(repo.path(), &["add", "."]);
        git_run(repo.path(), &["commit", "-q", "-m", "fix with mountain of garbage"]);

        // Oracle passes iff real.txt contains FIXED (cross-platform python one-liner).
        let real_oracle = "python -c \"import sys; sys.exit(0 if 'FIXED' in open('real.txt').read() else 1)\"";
        let ablated = execute_ablate(Some(json!({
            "repoPath": repo.path().display().to_string(),
            "commit": "HEAD",
            "oracle_cmd": real_oracle,
        })))
        .await
        .expect("ablate");

        assert_eq!(ablated["status"], json!("completed"), "{ablated}");
        assert_eq!(ablated["tautology_check"], json!("passed"), "{ablated}");
        assert_eq!(ablated["necessary_hunks"].as_array().unwrap().len(), 1, "{ablated}");
        assert_eq!(ablated["necessary_hunks"][0]["file"], json!("real.txt"), "{ablated}");
        assert_eq!(ablated["superstitious_hunks"].as_array().unwrap().len(), 1, "{ablated}");
        assert_eq!(ablated["superstitious_hunks"][0]["file"], json!("noise.txt"), "{ablated}");
        assert_eq!(ablated["joint_minimality_verified"], json!(true), "{ablated}");

        // Tautology test: oracle that ALWAYS exits 0 must be REJECTED!
        let fake_oracle = "python -c \"import sys; sys.exit(0)\"";
        let taut = execute_ablate(Some(json!({
            "repoPath": repo.path().display().to_string(),
            "commit": "HEAD",
            "oracle_cmd": fake_oracle,
        })))
        .await
        .expect("tautology check");
        assert_eq!(taut["status"], json!("rejected_tautology"), "{taut}");
        assert_eq!(taut["code"], json!("TAUTOLOGICAL_SELF_CONFIRMING_TEST"), "{taut}");
    }

    #[test]
    fn enrich_walk_detects_interaction_pairs_oscillations_trust_laundering_and_env_drift() {
        let (storage, _sdir) = open_store();
        let c1 = save_node(
            &storage,
            "user",
            "Commit A changes parser",
            "event",
            &["commit:1111111", "file:src/parser.rs", "env:rustc:1.85"],
        );
        let c2 = save_node(
            &storage,
            "user",
            "Commit B changes parser caller",
            "event",
            &["commit:2222222", "file:src/parser.rs", "env:rustc:1.86"],
        );
        let untrusted = save_node(
            &storage,
            "user",
            "Scraped external snippet",
            "note",
            &["untrusted", "file:src/other.rs"],
        );
        let pinned = save_node(
            &storage,
            "user",
            "Pinned rule derived from untrusted",
            "decision",
            &["pinned"],
        );
        add_edge(&storage, &pinned, &untrusted, "derived_from");
        // Oscillation: two flips on src/parser.rs!
        add_edge(&storage, &c2, &c1, "corrects");
        add_edge(&storage, &c1, &c2, "supersedes");

        let mut resp = json!({
            "starts": [pinned],
            "causes": [
                {"id": c1},
                {"id": c2},
                {"id": untrusted},
            ]
        });
        enrich_walk_response(storage.as_ref(), &mut resp);

        assert!(
            !resp["interaction_pairs"].as_array().unwrap().is_empty(),
            "must detect multi-commit interaction pair on src/parser.rs: {resp}"
        );
        assert!(
            !resp["zombie_oscillations"].as_array().unwrap().is_empty(),
            "must detect RETRACE oscillation cycle on src/parser.rs: {resp}"
        );
        assert_eq!(
            resp["trust_audit"]["laundering_risk"],
            json!(true),
            "must catch untrusted -> pinned laundering: {resp}"
        );
        assert!(
            resp["env_drift"]
                .as_array()
                .unwrap()
                .iter()
                .any(|d| d["key"] == "env:rustc"),
            "must detect env:rustc 1.85 -> 1.86 drift: {resp}"
        );
        assert!(
            resp["diverse_causes"].as_array().unwrap().len() >= 2,
            "must select maximally anchor-disjoint diverse causes: {resp}"
        );
    }

    #[test]
    fn pr_audit_catches_chestertons_fence_and_zombie_revert_and_renders_markdown() {
        let repo = tempfile::tempdir().unwrap();
        git_run(repo.path(), &["init", "-q"]);
        git_run(repo.path(), &["config", "user.email", "sam@example.com"]);
        git_run(repo.path(), &["config", "user.name", "Sam"]);

        // 1. Initial commit + a security/bug fix commit establishing a Chesterton's Fence.
        std::fs::create_dir_all(repo.path().join("src")).unwrap();
        std::fs::write(repo.path().join("src/guard.rs"), "pub fn check() -> bool { false }\n").unwrap();
        std::fs::write(repo.path().join("src/zombie.rs"), "pub fn mode() -> u8 { 1 }\n").unwrap();
        git_run(repo.path(), &["add", "."]);
        git_run(repo.path(), &["commit", "-q", "-m", "init repo"]);

        std::fs::write(repo.path().join("src/guard.rs"), "pub fn check() -> bool { true }\n").unwrap();
        git_run(repo.path(), &["add", "src/guard.rs"]);
        git_run(repo.path(), &["commit", "-q", "-m", "fix(auth): prevent bypass regression"]);

        // 2. Bad commit on zombie.rs that gets reverted on main.
        std::fs::write(repo.path().join("src/zombie.rs"), "pub fn mode() -> u8 { 99 }\n").unwrap();
        git_run(repo.path(), &["add", "src/zombie.rs"]);
        git_run(repo.path(), &["commit", "-q", "-m", "bad mode 99"]);
        let bad_sha = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();
        git_run(repo.path(), &["revert", "--no-edit", &bad_sha]);
        let base_sha = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();

        // 3. PR commit modifies the Chesterton's Fence line in guard.rs AND re-applies the reverted zombie change!
        std::fs::write(repo.path().join("src/guard.rs"), "pub fn check() -> bool { false }\n").unwrap();
        git_run(repo.path(), &["add", "src/guard.rs"]);
        git_run(repo.path(), &["commit", "-q", "-m", "refactor guard"]);

        std::fs::write(repo.path().join("src/zombie.rs"), "pub fn mode() -> u8 { 99 }\n").unwrap();
        git_run(repo.path(), &["add", "src/zombie.rs"]);
        git_run(repo.path(), &["commit", "-q", "-m", "reintroduce mode 99"]);
        let head_sha = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();

        let report = run_pr_audit(
            None,
            &PrAuditConfig {
                repo: repo.path().to_path_buf(),
                base: base_sha,
                head: head_sha,
                oracle_cmd: None,
                max_hunks: 16,
            },
        )
        .expect("run_pr_audit");

        assert!(report.blocked_by_zombie, "must catch zombie regression: {report:?}");
        assert!(!report.zombie_matches.is_empty(), "{report:?}");
        assert!(
            report
                .chesterton_fences
                .iter()
                .any(|f| f["file"] == "src/guard.rs" && f["subject"].as_str().unwrap_or("").contains("prevent bypass")),
            "must surface fix(auth) Chesterton's Fence on src/guard.rs: {report:?}"
        );

        let md = render_pr_audit_markdown(&report);
        assert!(md.contains("<!-- vestige-causal-gate -->"), "{md}");
        assert!(md.contains("ZOMBIE REGRESSION DETECTED"), "{md}");
        assert!(md.contains("src/guard.rs"), "{md}");

        let sarif = render_pr_audit_sarif(&report);
        assert_eq!(sarif["version"], "2.1.0");
        let results = sarif["runs"][0]["results"].as_array().unwrap();
        assert!(results.iter().any(|r| r["ruleId"] == "VST-ZOMBIE-001"));
        assert!(results.iter().any(|r| r["ruleId"] == "VST-FENCE-001"));
    }
}
