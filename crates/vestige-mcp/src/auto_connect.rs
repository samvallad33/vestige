//! # Auto-Connect (the ingest-time share of `vestige connect`)
//!
//! `vestige connect` bridges ingest isolation after the fact: two memories
//! that plainly talk about the same `euler` or `path.py` sit unconnected
//! until someone runs the command. Auto-connect moves the same bridge onto
//! the write path — every memory the CLI or the `smart_ingest` tool just
//! saved is immediately joined to the memories it shares entities with, so
//! `causal-walk --logged-write` works on freshly ingested memories without a
//! second command.
//!
//! Where `vestige connect` is a full scan (extract entities from every node
//! in the scope, intersect every pair), auto-connect runs on ONE memory and
//! must stay cheap. It links on exact identities only:
//!
//! * a tag, exactly as recorded on the memory
//! * a repo-relative path (`src/lib.rs`, `x/mlxrunner/mlx/random.go`)
//! * a commit sha
//! * an issue ref (`GH-42`, `#123`, `owner/repo#123`)
//! * a url
//!
//! Free-text words and path directory segments are not edge keys. Segments
//! may still appear in [`extract_entities`] for the connect command; they
//! do not create a `touched` edge by themselves.
//!
//! 1. Collect those identity keys once ([`edge_keys`]). No ML, no
//!    similarity, no keyword overlap.
//! 2. Resolve each recorded tag through [`Storage::resolve_handle`] — the
//!    exact path `vestige recall --handle <tag>` takes. A `Tag` resolution
//!    names the live nodes carrying that tag.
//! 3. When the memory carries a path, sha, issue ref, or url, page the
//!    scope and keep nodes whose identity keys intersect. Content words are
//!    never resolved as handles.
//! 4. For each candidate (minus the memory itself, minus pairs already
//!    joined by a recorded edge either direction), confirm the shared
//!    identity keys by set intersection and write one `touched` edge through
//!    [`Storage::save_connection`], exactly as the connect command writes
//!    its edges.
//!
//! Two invariants are inherited from the connect command: the older memory
//! of a pair is the edge's source (the walk's rule for `touched`), and edges
//! stay within one scope. Idempotency rides the store's per-memory edge
//! index ([`Storage::get_connections_for_memory`]): a pair joined by any
//! recorded edge — including one written by an earlier auto-connect or by
//! `vestige connect` — is left alone, so re-ingesting or re-running never
//! stacks parallel edges.
//!
//! A commit sha, path, issue ref, or url in the content is enough: it does
//! not have to be stored as a tag. `vestige connect` remains the catch-up
//! scan for anything auto-connect did not join.

use std::collections::HashSet;

use chrono::Utc;
use vestige_core::storage::{HandleKind, Storage};
use vestige_core::{ConnectionRecord, KnowledgeNode};

/// Entities resolved as tag handles per ingest. A normal memory carries far
/// fewer; the cap keeps a pathological (near-1MB) memory from turning the
/// write path into thousands of handle resolutions.
const MAX_TAG_LOOKUPS: usize = 50;

/// Edges one auto-connect pass will write. The same safety cap the connect
/// command ships as its `--max-edges` default.
const MAX_AUTO_EDGES: usize = 100;

/// What one auto-connect pass did: how many `touched` edges it wrote, and
/// the union of the entities those edges joined on (sorted, deduplicated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoConnectReport {
    pub edges: usize,
    pub shared_entities: Vec<String>,
}

/// Join the memory `memory_id` (content `content`, tags `tags`, written in
/// `scope`) to the existing memories it shares entities with, writing at
/// most [`MAX_AUTO_EDGES`] `touched` edges. The memory must already be
/// saved; failures are returned as `Err` for the caller to report without
/// undoing the ingest.
///
/// `content` and `tags` are the saved memory's own values (the caller holds
/// the node the ingest returned); the store is consulted for the node's
/// creation time, which decides edge direction and must be read back rather
/// than trusted from the caller (a backdated write rewrites it).
pub fn auto_connect_new_memory(
    storage: &Storage,
    memory_id: &str,
    scope: &str,
    content: &str,
    tags: &[String],
) -> Result<AutoConnectReport, String> {
    let new_node = storage
        .get_node(memory_id)
        .map_err(|err| format!("auto-connect could not read {memory_id} back: {err}"))?
        .ok_or_else(|| format!("auto-connect could not read {memory_id} back: not found"))?;

    // Sorted and deduplicated by edge_keys, so the same memory always
    // looks up the same identities.
    let keys = edge_keys(content, tags);
    if keys.is_empty() {
        return Ok(AutoConnectReport {
            edges: 0,
            shared_entities: Vec::new(),
        });
    }
    let new_keys: HashSet<String> = keys.iter().cloned().collect();
    let content_keys = content_identity_keys(content);

    // Pairs the new memory already shares a recorded edge with (either
    // direction), normalized so id order cannot hide a duplicate. This is
    // the per-memory slice of the connect command's joined-pair set, read
    // from the store's edge index instead of the whole edge list.
    let mut joined: HashSet<(String, String)> = storage
        .get_connections_for_memory(memory_id)
        .map_err(|err| format!("auto-connect could not read {memory_id}'s edges: {err}"))?
        .into_iter()
        .filter_map(|edge| pair_key(&edge.source_id, &edge.target_id))
        .collect();

    // Exact tags only. A content word is never resolved as a handle, so
    // "code" / "lead" / "only" cannot pull in every memory tagged with a
    // generic word.
    let mut candidates: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for tag in tags.iter().take(MAX_TAG_LOOKUPS) {
        let tag = tag.trim();
        if tag.is_empty() {
            continue;
        }
        let resolution = storage.resolve_handle(tag);
        if resolution.kind != HandleKind::Tag {
            continue;
        }
        for id in resolution.ids {
            if id != memory_id && seen.insert(id.clone()) {
                candidates.push(id);
            }
        }
    }
    // A path, sha, issue ref, or url is not a tag. Page the scope and keep
    // memories whose exact identity keys intersect. Strata's page read is a
    // full scope load, so one large page covers a normal store.
    if !content_keys.is_empty() {
        const PAGE: i32 = 4_096;
        let mut offset = 0i32;
        loop {
            let page = storage
                .get_all_nodes_in_scope(scope, PAGE, offset)
                .map_err(|err| format!("auto-connect could not read scope {scope}: {err}"))?;
            if page.is_empty() {
                break;
            }
            let full = page.len() == PAGE as usize;
            for node in page {
                if node.id == memory_id || seen.contains(&node.id) {
                    continue;
                }
                let shares = edge_keys(&node.content, &node.tags)
                    .into_iter()
                    .any(|key| new_keys.contains(&key));
                if shares && seen.insert(node.id.clone()) {
                    candidates.push(node.id);
                }
            }
            if !full {
                break;
            }
            let next = offset.saturating_add(PAGE);
            if next == offset {
                break;
            }
            offset = next;
        }
    }

    let now = Utc::now();
    let mut report = AutoConnectReport {
        edges: 0,
        shared_entities: Vec::new(),
    };
    let mut shared_seen: HashSet<String> = HashSet::new();
    for candidate in candidates {
        if report.edges >= MAX_AUTO_EDGES {
            break;
        }
        let Some(node) = storage
            .get_node(&candidate)
            .map_err(|err| format!("auto-connect could not read {candidate} back: {err}"))?
        else {
            continue;
        };
        // Edges stay within one scope, the invariant the connect command
        // and declared links both keep.
        let in_scope = storage
            .node_is_in_scope(&candidate, scope)
            .map_err(|err| format!("auto-connect could not place {candidate} in scope: {err}"))?;
        if !in_scope {
            continue;
        }
        let key = match pair_key(memory_id, &candidate) {
            Some(key) => key,
            None => continue,
        };
        if joined.contains(&key) {
            continue;
        }
        let shared = shared_entities(&new_keys, &node);
        if shared.is_empty() {
            continue;
        }
        // Direction follows the walk's rule for `touched`: the source is
        // the earlier record, so from the target the walk goes to the
        // source. Id order breaks creation-time ties, as in the connect
        // command's sort.
        let (source_id, target_id) = earlier_first(&new_node, &node, memory_id, &candidate);
        // strength 0.5: a co-touch is a moderate link, the same weight the
        // connect command writes.
        let edge = ConnectionRecord {
            source_id,
            target_id,
            strength: 0.5,
            link_type: "touched".to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        };
        storage
            .save_connection(&edge)
            .map_err(|err| format!("auto-connect edge {candidate} was not admitted: {err}"))?;
        joined.insert(key);
        report.edges += 1;
        shared_seen.extend(shared);
    }

    report.shared_entities = {
        let mut entities: Vec<String> = shared_seen.into_iter().collect();
        entities.sort();
        entities
    };
    Ok(report)
}

/// A normalized, order-independent pair key, `None` for a self-pair.
fn pair_key(left: &str, right: &str) -> Option<(String, String)> {
    (left != right).then(|| {
        if left < right {
            (left.to_string(), right.to_string())
        } else {
            (right.to_string(), left.to_string())
        }
    })
}

/// The exact identities two memories share: the intersection of the new
/// memory's edge keys and the candidate's, sorted. Free-text words and path
/// directory segments are absent from both sides.
fn shared_entities(new_keys: &HashSet<String>, candidate: &KnowledgeNode) -> Vec<String> {
    let candidate_keys: HashSet<String> = edge_keys(&candidate.content, &candidate.tags)
        .into_iter()
        .collect();
    let mut shared: Vec<String> = new_keys.intersection(&candidate_keys).cloned().collect();
    shared.sort();
    shared
}

/// Exact identities that may justify a `touched` edge: recorded tags, plus
/// repo-relative paths, commit shas, issue refs, and urls from the content.
/// Commit shas are lowercased so `A1B2C3D` and `a1b2c3d` are one identity.
/// Directory segments and free-text words are not included.
fn edge_keys(content: &str, tags: &[String]) -> Vec<String> {
    let mut keys = content_identity_keys(content);
    for tag in tags {
        let tag = tag.trim();
        if !tag.is_empty() {
            keys.push(tag.to_string());
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

/// Path, commit sha, issue ref, and url. Email and version spans are not
/// identities this bridge links on. Paths are the cleaned repo-relative
/// token (`x/mlx/random.go`, `path.py`); directory segments are not keys,
/// and a trailing comma or parenthesis is not part of the path.
fn content_identity_keys(content: &str) -> Vec<String> {
    use crate::intake::entities::{EntityKind, extract_typed_spans};

    let mut keys = repo_relative_paths(content);
    for span in extract_typed_spans(content) {
        match span.kind {
            EntityKind::CommitSha => keys.push(span.surface.to_ascii_lowercase()),
            EntityKind::Url | EntityKind::IssueRef => keys.push(span.surface),
            EntityKind::FilePath | EntityKind::Email | EntityKind::Version => {}
        }
    }
    keys
}

/// Repo-relative file tokens: a slash or a bare filename with an extension.
/// `src/lib.rs` and `path.py` qualify. `mlx` (a directory segment) and
/// `v0.24.0` (a version) do not.
fn repo_relative_paths(content: &str) -> Vec<String> {
    let mut keys = Vec::new();
    for word in content.split_whitespace() {
        let cleaned = word.trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '.' && c != '_' && c != '-' && c != '/' && c != '~'
        });
        if is_repo_relative_path(cleaned) {
            keys.push(cleaned.to_string());
        }
    }
    keys
}

fn is_repo_relative_path(token: &str) -> bool {
    if token.len() <= 3 || token.contains("://") || is_pure_number(token) || is_version_token(token)
    {
        return false;
    }
    let final_segment = token.rsplit('/').next().unwrap_or(token);
    let Some(dot) = final_segment.rfind('.') else {
        return false;
    };
    dot > 0
        && dot + 1 < final_segment.len()
        && final_segment[dot + 1..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric())
}

fn is_version_token(token: &str) -> bool {
    let body = token.strip_prefix('v').unwrap_or(token);
    let mut parts = body.split('.');
    let Some(first) = parts.next() else {
        return false;
    };
    if first.is_empty() || !first.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let mut groups = 1usize;
    for part in parts {
        if part.is_empty() || !part.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        groups += 1;
    }
    groups >= 2
}

/// The pair ordered (source, target) with the earlier record first:
/// creation time, then id, the same order the connect command sorts by.
fn earlier_first(
    new_node: &KnowledgeNode,
    candidate: &KnowledgeNode,
    memory_id: &str,
    candidate_id: &str,
) -> (String, String) {
    let new_first = new_node
        .created_at
        .cmp(&candidate.created_at)
        .then_with(|| memory_id.cmp(candidate_id))
        == std::cmp::Ordering::Less;
    if new_first {
        (memory_id.to_string(), candidate_id.to_string())
    } else {
        (candidate_id.to_string(), memory_id.to_string())
    }
}

/// Deterministic entities for `vestige connect` and auto-connect: dotted
/// file-path tokens, snake_case / camelCase identifier words, and the
/// memory's own tags. No ML, no similarity — the same text always yields
/// the same entities.
///
/// Identifiers are lowercased so `Euler` and `euler` join; file paths and
/// tags keep their case (a path is its exact bytes). Sharing for `vestige
/// connect` is decided by set intersection downstream. Auto-connect does
/// not use these free-text identifiers or path directory segments as edge
/// keys; see [`edge_keys`].
pub fn extract_entities(content: &str, tags: &[String]) -> Vec<String> {
    let mut entities = Vec::new();

    // File paths (anything with a dot and an extension): split on
    // whitespace, strip surrounding punctuation, keep dotted tokens
    // (`path.py`, `db.config.yaml`). Pure numbers like `1.2.3` are not
    // entities.
    for word in content.split_whitespace() {
        let cleaned =
            word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_' && c != '-');
        if cleaned.contains('.') && cleaned.len() > 3 && !is_pure_number(cleaned) {
            entities.push(cleaned.to_string());
            // A path's directory segments are domain tokens: `x/mlxrunner/mlx/random.go`
            // says the memory is about `mlxrunner` and `mlx`, so a failure tagged
            // `mlx` joins the commits that touched that tree without anyone
            // naming a file in the failure report.
            if cleaned.contains('/') {
                for segment in cleaned.split('/') {
                    if segment.len() >= 3 && !is_pure_number(segment) && !is_stopword(segment) {
                        entities.push(segment.to_lowercase());
                    }
                }
            }
        }
    }

    // Identifiers (camelCase, snake_case, >= 4 chars): split on everything
    // that is not alphanumeric or `_`, keep tokens long enough to mean
    // something, minus stopwords and pure numbers.
    for word in content.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.len() >= 4 && !is_pure_number(word) && !is_stopword(word) {
            entities.push(word.to_lowercase());
        }
    }

    // Tags are entities exactly as recorded.
    for tag in tags {
        entities.push(tag.clone());
    }

    entities.sort();
    entities.dedup();
    entities
}

/// `true` when the token is digits only (plus the dots of a dotted token):
/// line numbers, ids and versions are not entities.
fn is_pure_number(word: &str) -> bool {
    word.chars().all(|c| c.is_ascii_digit() || c == '.')
}

/// Small embedded stop-word list: common English words that would otherwise
/// join any two memories written in English.
fn is_stopword(word: &str) -> bool {
    const STOPWORDS: &[&str] = &[
        "that", "this", "with", "from", "have", "been", "were", "will", "would", "could", "should",
        "there", "their", "about", "which", "when", "what", "while", "these", "those", "then",
        "than", "they", "them", "into", "over", "after", "before", "under", "above", "below",
        "between", "because", "since", "until", "against", "without", "within", "across", "the",
        "and", "but", "for", "not", "you", "all", "can", "her", "was", "one", "our", "out", "day",
        "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who",
        "boy", "did", "its", "let", "put", "say", "she", "too", "use", "dad", "mom", "try", "ask",
    ];
    STOPWORDS.contains(&word.to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entities_cover_paths_identifiers_and_tags() {
        let entities = extract_entities(
            "PR 4337 removed npoints floor from euler() in path.py",
            &["euler".to_string(), "path.py".to_string()],
        );
        assert!(entities.contains(&"euler".to_string()));
        assert!(entities.contains(&"npoints".to_string()));
        assert!(entities.contains(&"path.py".to_string()));
        // Stopwords, short tokens and pure numbers are not entities.
        assert!(!entities.contains(&"from".to_string()));
        assert!(!entities.contains(&"4337".to_string()));
        assert!(!entities.iter().any(|e| e == "pr"));
    }

    #[test]
    fn identifiers_join_case_insensitively_and_paths_do_not() {
        let upper = extract_entities("Euler bends deform", &[]);
        let lower = extract_entities("euler bends deform", &[]);
        assert_eq!(upper, lower);

        let path = extract_entities("edited path.py", &[]);
        assert!(path.contains(&"path.py".to_string()));
        assert!(!path.contains(&"PATH.PY".to_string()));
    }

    #[test]
    fn pair_key_is_order_independent_and_rejects_self_pairs() {
        assert_eq!(
            pair_key("b-node", "a-node"),
            Some(("a-node".to_string(), "b-node".to_string()))
        );
        assert_eq!(pair_key("a-node", "b-node"), pair_key("b-node", "a-node"));
        assert_eq!(pair_key("same", "same"), None);
    }

    #[test]
    fn module_prefix_entities_from_paths() {
        let commit = extract_entities(
            "commit 4860130f839a mlx: rework the MLX sampler (#16122)\nfiles: x/mlxrunner/mlx/random.go, x/mlxrunner/sample/sampler.go",
            &[],
        );
        let failure = extract_entities(
            "Severe inter-prompt delay regression on Apple Silicon MLX in v0.24.0",
            &["mlx".to_string()],
        );
        let shared: HashSet<String> = commit.iter().cloned().collect();
        let hit: Vec<&String> = failure.iter().filter(|e| shared.contains(*e)).collect();
        assert!(
            !hit.is_empty(),
            "failure tagged mlx must share an entity with the mlxrunner commit; commit entities: {commit:?}, failure entities: {failure:?}"
        );
        // The directory segment stays an entity. It is not an edge key.
        let path_keys = edge_keys(
            "commit 4860130f839a mlx: rework the MLX sampler (#16122)\nfiles: x/mlxrunner/mlx/random.go, x/mlxrunner/sample/sampler.go",
            &[],
        );
        assert!(
            path_keys
                .iter()
                .any(|key| key == "x/mlxrunner/mlx/random.go")
        );
        assert!(
            !path_keys
                .iter()
                .any(|key| key == "mlx" || key == "mlxrunner")
        );
    }

    /// Two memories that share only common words (and a directory segment)
    /// get no `touched` edge. Two that share a commit sha get one.
    #[test]
    fn common_words_create_no_edges_and_a_shared_commit_sha_creates_one() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/smart-ingest-exact-identities")
            .join(format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir_all(&dir).unwrap();
        let _cleanup = RmDir(dir.clone());

        let storage = crate::strata_memory::open(&dir).unwrap();
        let scope = vestige_core::DEFAULT_MEMORY_SCOPE;
        let ingest = |content: &str, tags: &[&str]| {
            storage
                .ingest(vestige_core::IngestInput {
                    content: content.to_string(),
                    tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
                    ..vestige_core::IngestInput::default()
                })
                .unwrap()
        };

        // "only" / "lead" / "code" are the generic words real stores were
        // joining on. The first memory is tagged `code` and `mlx` so the
        // old extractor would resolve both as tag handles.
        let prose = ingest("the only lead is the code", &["code", "mlx"]);
        let path = "x/mlxrunner/mlx/random.go";
        assert!(
            extract_entities(&format!("touched {path}"), &[])
                .iter()
                .any(|entity| entity == "mlx"),
            "a path directory segment stays an entity"
        );
        let overlap = ingest(&format!("only the lead code lives under {path}"), &[]);
        let words = auto_connect_new_memory(
            storage.as_ref(),
            &overlap.id,
            scope,
            &overlap.content,
            &overlap.tags,
        )
        .unwrap();
        assert_eq!(
            words.edges, 0,
            "common words and a directory segment must not create edges: {words:?}"
        );
        assert!(
            storage
                .get_connections_for_memory(&prose.id)
                .unwrap()
                .is_empty(),
            "the prose memory stays unlinked"
        );

        let sha = "a1b2c3d4e5f67890";
        let landed = ingest(&format!("landed commit {sha} on main"), &[]);
        let reverted = ingest(&format!("revert {sha} after the outage"), &[]);
        let linked = auto_connect_new_memory(
            storage.as_ref(),
            &reverted.id,
            scope,
            &reverted.content,
            &reverted.tags,
        )
        .unwrap();
        assert_eq!(linked.edges, 1, "{linked:?}");
        assert_eq!(linked.shared_entities, vec![sha.to_string()]);
        let edges = storage.get_connections_for_memory(&reverted.id).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].link_type, "touched");
        assert!(
            (edges[0].source_id == landed.id && edges[0].target_id == reverted.id)
                || (edges[0].source_id == reverted.id && edges[0].target_id == landed.id)
        );
    }

    struct RmDir(std::path::PathBuf);
    impl Drop for RmDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
