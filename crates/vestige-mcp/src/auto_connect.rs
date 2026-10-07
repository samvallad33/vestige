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
//! must stay cheap:
//!
//! 1. Extract the new memory's entities once ([`extract_entities`], the same
//!    deterministic extractor the connect command uses — file paths,
//!    identifiers, tags; no ML, no similarity).
//! 2. Resolve each entity as a tag handle through
//!    [`Storage::resolve_handle`] — the exact path `vestige recall --handle
//!    <tag>` takes. A `Tag` resolution names the live nodes carrying that
//!    tag, so candidate generation is the store's own tag lookup, never a
//!    pairwise scan of the log.
//! 3. For each candidate (minus the memory itself, minus pairs already
//!    joined by a recorded edge either direction), confirm the shared
//!    entities by set intersection and write one `touched` edge through
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
//! Candidate generation only sees entities that exist as TAGS on other
//! memories; two memories sharing a bare content identifier with no matching
//! tag are still the full scan's job (`vestige connect` remains the
//! catch-up command).

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

    // Sorted and deduplicated by extract_entities, so the lookup cap below
    // always drops the same entities for the same memory.
    let entities = extract_entities(content, tags);
    if entities.is_empty() {
        return Ok(AutoConnectReport {
            edges: 0,
            shared_entities: Vec::new(),
        });
    }
    let new_entities: HashSet<String> = entities.iter().cloned().collect();

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

    // Candidate generation: each entity resolved as a tag handle names the
    // live nodes carrying that tag. Only Tag resolutions are consumed — a
    // handle that resolved as a memory id (or an id prefix) matched a node's
    // id, not a shared entity, and the pair would not survive the
    // intersection check anyway.
    let mut candidates: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for entity in entities.iter().take(MAX_TAG_LOOKUPS) {
        let resolution = storage.resolve_handle(entity);
        if resolution.kind != HandleKind::Tag {
            continue;
        }
        for id in resolution.ids {
            if id != memory_id && seen.insert(id.clone()) {
                candidates.push(id);
            }
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
        let shared = shared_entities(&new_entities, &node);
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

/// The entities two memories share: the intersection of the new memory's
/// entity set and the candidate's, sorted. A tag-handle candidate always
/// shares at least the tag it was found by, but the intersection is still
/// computed — it is what the caller reports, and it keeps any non-tag
/// resolution honest.
fn shared_entities(new_entities: &HashSet<String>, candidate: &KnowledgeNode) -> Vec<String> {
    let candidate_entities: HashSet<String> = extract_entities(&candidate.content, &candidate.tags)
        .into_iter()
        .collect();
    let mut shared: Vec<String> = new_entities
        .intersection(&candidate_entities)
        .cloned()
        .collect();
    shared.sort();
    shared
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
/// tags keep their case (a path is its exact bytes). Sharing is decided by
/// set intersection downstream, which is also where the "identifier must
/// appear in >= 2 memories" rule lives: an identifier shared by a pair
/// appears in 2 memories by definition, so no separate document-frequency
/// pass is needed.
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
    }
}
