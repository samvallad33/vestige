//! # Auto-Connect (the ingest-time share of `vestige connect`)
//!
//! `vestige connect` bridges ingest isolation after the fact: two memories
//! that record the same `src/path.py` or carry the same tag sit unconnected
//! until someone runs the command. Auto-connect moves the same bridge onto
//! the write path, so `causal-walk --logged-write` works on freshly ingested
//! memories without a second command.
//!
//! ## What joins two memories: exact identities only
//!
//! Vestige 4.x finds, ranks, pairs and explains nothing by embeddings,
//! keyword or shared-word overlap, or free-text search. A `touched` edge
//! written here therefore rests on an **exact identity** both memories
//! record, never on a word they happen to share. [`extract_identities`]
//! defines the whole set:
//!
//! | kind | exact definition |
//! |---|---|
//! | `tag` | a tag of the memory, byte for byte (case-sensitive) |
//! | `path` | a whitespace-delimited token that is file-path shaped ([`is_file_path`]), byte for byte |
//! | `commit` | a token that is entirely a git sha: 40 hex, or 7-39 hex with at least one digit and one letter |
//! | `issue` | a token of the form `owner/repo#123`, or a GitHub issue / pull URL (normalized to that form) |
//! | `url` | a token starting `http://` or `https://`, byte for byte |
//!
//! Prose words are not identities: `euler bends deform` and `euler refactor`
//! share a word and are NOT joined. A tag is also classified as a token, so
//! a tag `path.py` is the same `path` identity as `path.py` in another
//! memory's text.
//!
//! ## The too-common-tag guard
//!
//! A tag joins every pair that carries it, so a tag on every memory of a
//! batch (a campaign tag) would join everything to everything and say
//! nothing. A tag carried by more than [`MAX_TAG_CARRIERS`] memories of the
//! scope is too common to be evidence and is skipped (and named in the
//! report). The guard applies to tags only: a shared file path is the strong
//! signal and always joins.
//!
//! At ingest the guard counts the carriers present at the time of the write.
//! The log is append-only, so the edges the first carriers of a tag received
//! before it grew common stay recorded; `vestige connect` over the finished
//! scope applies the guard to the whole scope at once.
//!
//! ## Explainable edges
//!
//! Every edge is reported with the pair it joined and the exact identities
//! that joined it ([`JoinedPair`]), by `vestige ingest`, `vestige connect`
//! and the `smart_ingest` tool alike.
//!
//! ## How the ingest-time pass stays cheap
//!
//! Where `vestige connect` is a full scan (extract identities from every
//! node in the scope, intersect every pair), auto-connect runs on ONE memory:
//!
//! 1. Extract the new memory's identities once.
//! 2. Resolve each identity's value as a handle through
//!    [`Storage::resolve_handle`], the exact path `vestige recall --handle`
//!    takes. A `Tag` resolution names the live nodes carrying that exact
//!    tag; a `File` resolution (legacy stores) names the nodes recording
//!    that exact path token. Candidate generation is the store's own exact
//!    lookup, never a pairwise scan of the log.
//! 3. For each candidate (minus the memory itself, minus pairs already
//!    joined by a recorded edge either direction), confirm the shared
//!    identities by set intersection and write one `touched` edge through
//!    [`Storage::save_connection`], exactly as the connect command writes
//!    its edges.
//!
//! Two invariants are inherited from the connect command: the older memory
//! of a pair is the edge's source (the walk's rule for `touched`), and edges
//! stay within one scope. Idempotency rides the store's per-memory edge
//! index ([`Storage::get_connections_for_memory`]): a pair joined by any
//! recorded edge is left alone, so re-ingesting or re-running never stacks
//! parallel edges.
//!
//! On a Strata log candidate generation only sees identities that exist as
//! TAGS on other memories; two memories sharing a path only in their text,
//! with no matching tag, are the full scan's job (`vestige connect` remains
//! the catch-up command).
//!
//! An edge written here records that two memories name the same exact
//! thing. It is not a cause: a causal walk over these edges returns
//! hypotheses, never proven causes.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use chrono::Utc;
use vestige_core::storage::{HandleKind, Storage};
use vestige_core::{ConnectionRecord, KnowledgeNode};

/// Identities resolved as handles per ingest. A normal memory carries far
/// fewer; the cap keeps a pathological (near-1MB) memory from turning the
/// write path into thousands of handle resolutions. Tags are looked up
/// first; a tag past the cap is never judged and therefore never joins.
const MAX_HANDLE_LOOKUPS: usize = 50;

/// Edges one auto-connect pass will write. The same safety cap the connect
/// command ships as its `--max-edges` default.
const MAX_AUTO_EDGES: usize = 100;

/// The most memories of one scope that may carry a tag for it to still join
/// them. A tag on `k` memories joins `k * (k - 1) / 2` pairs: 14 carriers
/// are 91 pairs, 15 are 105. Past 14 a single tag would by itself exceed the
/// 100-edge budget of one pass ([`MAX_AUTO_EDGES`], `connect --max-edges`),
/// which is the point where it stops describing a pair and starts describing
/// the batch. Such a tag is skipped and named in the report.
pub const MAX_TAG_CARRIERS: usize = 14;

/// The kind of an exact identity. Order matters: tags sort first, so the
/// ingest-time lookup cap judges every tag before spending lookups on
/// content tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IdentityKind {
    Tag,
    Path,
    Commit,
    Issue,
    Url,
}

impl IdentityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            IdentityKind::Tag => "tag",
            IdentityKind::Path => "path",
            IdentityKind::Commit => "commit",
            IdentityKind::Issue => "issue",
            IdentityKind::Url => "url",
        }
    }
}

/// One exact identity a memory records. Two memories share it only when
/// kind and value are both equal. Rendered `kind:value`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Identity {
    pub kind: IdentityKind,
    pub value: String,
}

impl Identity {
    fn new(kind: IdentityKind, value: impl Into<String>) -> Self {
        Self {
            kind,
            value: value.into(),
        }
    }
}

impl fmt::Display for Identity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind.as_str(), self.value)
    }
}

/// One `touched` edge and the exact identities that justify it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinedPair {
    pub source_id: String,
    pub target_id: String,
    /// `kind:value`, sorted.
    pub identities: Vec<String>,
}

/// What one auto-connect pass did: the edges it wrote, each with the
/// identities that joined it; the union of those identities (sorted,
/// deduplicated); and the tags it refused to join on because more than
/// [`MAX_TAG_CARRIERS`] memories of the scope carry them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutoConnectReport {
    pub edges: usize,
    pub pairs: Vec<JoinedPair>,
    pub shared_identities: Vec<String>,
    pub skipped_common_tags: Vec<String>,
}

/// Join the memory `memory_id` (content `content`, tags `tags`, written in
/// `scope`) to the existing memories it shares an exact identity with,
/// writing at most [`MAX_AUTO_EDGES`] `touched` edges. The memory must
/// already be saved; failures are returned as `Err` for the caller to report
/// without undoing the ingest.
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

    // Sorted (tags first) and deduplicated by extract_identities, so the
    // lookup cap below always drops the same identities for the same memory.
    let identities = extract_identities(content, tags);
    let mut report = AutoConnectReport::default();
    if identities.is_empty() {
        return Ok(report);
    }

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

    // In-scope candidate nodes, read once each. `None` records a candidate
    // that is retired, missing, or in another scope.
    let mut nodes: HashMap<String, Option<KnowledgeNode>> = HashMap::new();
    let mut load = |id: &str| -> Result<Option<KnowledgeNode>, String> {
        if let Some(cached) = nodes.get(id) {
            return Ok(cached.clone());
        }
        // Edges stay within one scope, the invariant the connect command
        // and declared links both keep.
        let in_scope = storage
            .node_is_in_scope(id, scope)
            .map_err(|err| format!("auto-connect could not place {id} in scope: {err}"))?;
        let node = if in_scope {
            storage
                .get_node(id)
                .map_err(|err| format!("auto-connect could not read {id} back: {err}"))?
        } else {
            None
        };
        nodes.insert(id.to_string(), node.clone());
        Ok(node)
    };

    // Candidate generation: each identity's value resolved as a handle.
    // Only exact Tag / File resolutions are consumed; every candidate is
    // still confirmed by intersection below. The evidence set is what the
    // new memory may join on: every non-tag identity, and each tag that was
    // looked up and found on at most MAX_TAG_CARRIERS memories of the scope.
    let mut evidence: BTreeSet<Identity> = identities
        .iter()
        .filter(|identity| identity.kind != IdentityKind::Tag)
        .cloned()
        .collect();
    let mut candidates: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for identity in identities.iter().take(MAX_HANDLE_LOOKUPS) {
        let resolution = storage.resolve_handle(&identity.value);
        if !matches!(resolution.kind, HandleKind::Tag | HandleKind::File) {
            // A lone tag has no peer to join; it is still fair evidence.
            if identity.kind == IdentityKind::Tag {
                evidence.insert(identity.clone());
            }
            continue;
        }
        if identity.kind == IdentityKind::Tag {
            // The guard: count the in-scope memories carrying this exact
            // tag, the new memory included, and stop as soon as the tag is
            // known to be too common.
            let mut carriers: Vec<String> = Vec::new();
            let mut too_common = false;
            for id in &resolution.ids {
                if id == memory_id {
                    continue;
                }
                let Some(node) = load(id)? else { continue };
                if node.tags.iter().any(|tag| tag == &identity.value) {
                    carriers.push(id.clone());
                    if carriers.len() + 1 > MAX_TAG_CARRIERS {
                        too_common = true;
                        break;
                    }
                }
            }
            if too_common {
                report.skipped_common_tags.push(identity.value.clone());
                continue;
            }
            evidence.insert(identity.clone());
            for id in carriers {
                if seen.insert(id.clone()) {
                    candidates.push(id);
                }
            }
        } else {
            for id in resolution.ids {
                if id != memory_id && seen.insert(id.clone()) {
                    candidates.push(id);
                }
            }
        }
    }

    let now = Utc::now();
    let mut shared_seen: BTreeSet<String> = BTreeSet::new();
    for candidate in candidates {
        if report.edges >= MAX_AUTO_EDGES {
            break;
        }
        let Some(node) = load(&candidate)? else {
            continue;
        };
        let key = match pair_key(memory_id, &candidate) {
            Some(key) => key,
            None => continue,
        };
        if joined.contains(&key) {
            continue;
        }
        let candidate_identities: BTreeSet<Identity> =
            extract_identities(&node.content, &node.tags)
                .into_iter()
                .collect();
        let shared: Vec<String> = evidence
            .intersection(&candidate_identities)
            .map(Identity::to_string)
            .collect();
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
            source_id: source_id.clone(),
            target_id: target_id.clone(),
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
        shared_seen.extend(shared.iter().cloned());
        report.pairs.push(JoinedPair {
            source_id,
            target_id,
            identities: shared,
        });
    }

    report.shared_identities = shared_seen.into_iter().collect();
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

/// The tags too common to join on: every tag carried by more than
/// [`MAX_TAG_CARRIERS`] of the given memories, with its carrier count. Each
/// item is one memory's tag list; a tag repeated on one memory counts once.
pub fn too_common_tags<'a>(
    tag_lists: impl IntoIterator<Item = &'a [String]>,
) -> BTreeMap<String, usize> {
    let mut carriers: BTreeMap<String, usize> = BTreeMap::new();
    for tags in tag_lists {
        let distinct: BTreeSet<&String> = tags.iter().collect();
        for tag in distinct {
            *carriers.entry(tag.clone()).or_insert(0) += 1;
        }
    }
    carriers.retain(|_, count| *count > MAX_TAG_CARRIERS);
    carriers
}

/// The identities that join two memories: the exact intersection of their
/// identity sets, minus tags named in `common_tags`. Sorted.
pub fn joining_identities(
    left: &BTreeSet<Identity>,
    right: &BTreeSet<Identity>,
    common_tags: &BTreeMap<String, usize>,
) -> Vec<Identity> {
    left.intersection(right)
        .filter(|identity| {
            identity.kind != IdentityKind::Tag || !common_tags.contains_key(&identity.value)
        })
        .cloned()
        .collect()
}

/// How many distinct values a set of joining identities names. A tag that is
/// itself a path is one shared thing recorded under two kinds; `--min-shared`
/// counts it once.
pub fn distinct_values(identities: &[Identity]) -> usize {
    identities
        .iter()
        .map(|identity| identity.value.as_str())
        .collect::<BTreeSet<_>>()
        .len()
}

/// The exact identities of one memory, for `vestige connect` and
/// auto-connect: its tags, and the file paths, commit shas, issue references
/// and URLs that appear as whole tokens in its text or as tags. Sorted (tags
/// first) and deduplicated. No ML, no similarity, no words: the same memory
/// always yields the same identities, and a prose word never is one.
pub fn extract_identities(content: &str, tags: &[String]) -> Vec<Identity> {
    let mut identities: BTreeSet<Identity> = BTreeSet::new();
    for tag in tags {
        if tag.is_empty() {
            continue;
        }
        // A tag is an identity exactly as recorded, and also whatever exact
        // identity its bytes spell (a tag `path.py` is the path `path.py`).
        identities.insert(Identity::new(IdentityKind::Tag, tag.clone()));
        identities.extend(classify_token(tag));
    }
    for token in content.split_whitespace() {
        identities.extend(classify_token(token));
    }
    identities.into_iter().collect()
}

/// Classify one whitespace-delimited token. Surrounding quotes, brackets and
/// sentence punctuation are not part of the token. Returns nothing for a
/// prose word.
fn classify_token(raw: &str) -> Vec<Identity> {
    let token = raw
        .trim_start_matches(['"', '\'', '`', '(', '[', '{', '<'])
        .trim_end_matches([
            '"', '\'', '`', ')', ']', '}', '>', ',', ';', '!', '?', '.', ':',
        ]);
    if token.is_empty() {
        return Vec::new();
    }

    // URL: scheme plus a non-empty remainder, byte for byte. A GitHub issue
    // or pull URL also names the issue it points at.
    for scheme in ["https://", "http://"] {
        if let Some(rest) = token.strip_prefix(scheme) {
            if rest.is_empty() {
                return Vec::new();
            }
            let mut found = vec![Identity::new(IdentityKind::Url, token)];
            if let Some(issue) = github_issue_from_url(rest) {
                found.push(Identity::new(IdentityKind::Issue, issue));
            }
            return found;
        }
    }

    if is_issue_ref(token) {
        return vec![Identity::new(IdentityKind::Issue, token)];
    }
    if is_commit_sha(token) {
        return vec![Identity::new(
            IdentityKind::Commit,
            token.to_ascii_lowercase(),
        )];
    }

    // File path: a `:line` or `:line:col` suffix and a leading `./` are not
    // part of the path.
    let mut path = token;
    for _ in 0..2 {
        if let Some((head, tail)) = path.rsplit_once(':')
            && !tail.is_empty()
            && tail.bytes().all(|b| b.is_ascii_digit())
        {
            path = head;
        }
    }
    while let Some(rest) = path.strip_prefix("./") {
        path = rest;
    }
    if is_file_path(path) {
        return vec![Identity::new(IdentityKind::Path, path)];
    }
    Vec::new()
}

/// `owner/repo#123`: a GitHub owner (alphanumerics and `-`), one `/`, a
/// repository name (alphanumerics, `.`, `_`, `-`), `#`, and a number. A bare
/// `#123` names no repository and is not an identity.
fn is_issue_ref(token: &str) -> bool {
    let Some((repo_path, number)) = token.split_once('#') else {
        return false;
    };
    let Some((owner, repo)) = repo_path.split_once('/') else {
        return false;
    };
    is_github_owner(owner)
        && is_github_repo(repo)
        && !number.is_empty()
        && number.bytes().all(|b| b.is_ascii_digit())
}

fn is_github_owner(owner: &str) -> bool {
    !owner.is_empty()
        && owner
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

fn is_github_repo(repo: &str) -> bool {
    !repo.is_empty()
        && repo
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// `github.com/<owner>/<repo>/issues/<n>` or `.../pull/<n>` (the URL with
/// its scheme already removed), optionally followed by `/...`, `#...` or
/// `?...`, as `owner/repo#n`.
fn github_issue_from_url(rest: &str) -> Option<String> {
    let rest = rest.strip_prefix("github.com/")?;
    let mut parts = rest.splitn(4, '/');
    let owner = parts.next()?;
    let repo = parts.next()?;
    let kind = parts.next()?;
    let tail = parts.next()?;
    if !is_github_owner(owner) || !is_github_repo(repo) || !matches!(kind, "issues" | "pull") {
        return None;
    }
    let digits = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    let (number, after) = tail.split_at(digits);
    if number.is_empty() || !(after.is_empty() || after.starts_with(['/', '#', '?'])) {
        return None;
    }
    Some(format!("{owner}/{repo}#{number}"))
}

/// A token that is entirely a git sha: 40 hex characters, or an abbreviated
/// 7-39 hex characters with at least one digit and one letter (so neither a
/// number like `1234567` nor a word like `defaced` is one). Abbreviated and
/// full forms are different tokens and do not join each other.
fn is_commit_sha(token: &str) -> bool {
    if !token.bytes().all(|b| b.is_ascii_hexdigit()) {
        return false;
    }
    match token.len() {
        40 => true,
        7..=39 => {
            token.bytes().any(|b| b.is_ascii_digit())
                && token.bytes().any(|b| b.is_ascii_alphabetic())
        }
        _ => false,
    }
}

/// A file-path-shaped token, decided by shape alone:
///
/// - only ASCII letters, digits and `. _ / + @ ~ -`, with no empty segment;
/// - its last segment is `stem.ext`, where `ext` is 1-10 alphanumerics with
///   at least one letter (so `1.2.3` and `v1.4.1` are not paths);
/// - with a `/` (`src/a.c`, `migrations/001.sql`) that is enough;
/// - without one (`path.py`), the stem must be 2+ characters, contain a
///   letter and not be version-shaped (`v1.4.x`), and a one-letter extension
///   must be lowercase, which keeps `e.g`, `i.e`, `U.S.A` and `Ph.D` out.
///
/// Extensionless names (`Makefile`) and bare directories are not matched:
/// by shape they are indistinguishable from words.
pub fn is_file_path(token: &str) -> bool {
    if token.is_empty()
        || !token.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'/' | b'+' | b'@' | b'~' | b'-')
        })
    {
        return false;
    }
    let has_separator = token.contains('/');
    let mut segments = token.split('/').peekable();
    // A leading `/` (absolute path) gives one empty first segment; any other
    // empty segment (`a//b`, trailing `/`) is not a file path.
    if token.starts_with('/') {
        segments.next();
    }
    let mut last = "";
    for segment in segments {
        if segment.is_empty() {
            return false;
        }
        last = segment;
    }
    let Some((stem, ext)) = last.rsplit_once('.') else {
        return false;
    };
    if stem.is_empty()
        || ext.is_empty()
        || ext.len() > 10
        || !ext.bytes().all(|b| b.is_ascii_alphanumeric())
        || !ext.bytes().any(|b| b.is_ascii_alphabetic())
    {
        return false;
    }
    if has_separator {
        return true;
    }
    stem.len() >= 2
        && stem.bytes().any(|b| b.is_ascii_alphabetic())
        && !is_version_shaped(stem)
        && (ext.len() >= 2 || ext.bytes().all(|b| b.is_ascii_lowercase()))
}

/// `1.4`, `v1`, `V2.10`: an optional `v` and dot-separated numbers.
fn is_version_shaped(stem: &str) -> bool {
    let digits = stem.strip_prefix(['v', 'V']).unwrap_or(stem);
    !digits.is_empty()
        && digits
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vestige_core::IngestInput;

    fn rendered(content: &str, tags: &[&str]) -> Vec<String> {
        let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
        extract_identities(content, &tags)
            .iter()
            .map(Identity::to_string)
            .collect()
    }

    fn shared(left: (&str, &[&str]), right: (&str, &[&str])) -> Vec<String> {
        let set = |(content, tags): (&str, &[&str])| -> BTreeSet<Identity> {
            let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
            extract_identities(content, &tags).into_iter().collect()
        };
        joining_identities(&set(left), &set(right), &BTreeMap::new())
            .iter()
            .map(Identity::to_string)
            .collect()
    }

    #[test]
    fn identities_are_tags_and_exact_tokens_never_words() {
        let identities = rendered(
            "PR 4337 removed npoints floor from euler() in path.py",
            &["euler", "path.py"],
        );
        assert_eq!(
            identities,
            vec!["tag:euler", "tag:path.py", "path:path.py"],
            "words (npoints, euler(), removed, floor) and numbers are not identities"
        );
    }

    #[test]
    fn shared_words_do_not_join() {
        // The pair the old extractor joined on the lowercased word `euler`.
        assert!(
            shared(
                (
                    "euler refactor touched connection_pool in the database",
                    &[]
                ),
                (
                    "Euler bends deform under load, connection_pool exhausted",
                    &[]
                ),
            )
            .is_empty()
        );
        // Every word in common, no identity in common.
        assert!(
            shared(
                ("redis timeout errors dropping connections", &["bug"]),
                ("redis timeout errors dropping connections", &["incident"]),
            )
            .is_empty()
        );
        // A word that equals another memory's tag is still a word.
        assert!(
            shared(
                ("euler bends deform", &[]),
                ("fixed the spiral", &["euler"])
            )
            .is_empty()
        );
    }

    #[test]
    fn exact_tag_joins_and_is_case_sensitive() {
        assert_eq!(
            shared(
                ("first", &["euler", "geometry"]),
                ("second", &["bug", "euler"])
            ),
            vec!["tag:euler"]
        );
        assert!(shared(("first", &["Euler"]), ("second", &["euler"])).is_empty());
        assert!(shared(("first", &["euler"]), ("second", &["euler-bends"])).is_empty());
    }

    #[test]
    fn exact_file_path_joins_and_is_byte_exact() {
        assert_eq!(
            shared(
                (
                    "Commit abc: fix. Touched: src/execution/index/art/art.cpp test/a.test",
                    &[]
                ),
                (
                    "crash at `src/execution/index/art/art.cpp:214:9`, see log",
                    &[]
                ),
            ),
            vec!["path:src/execution/index/art/art.cpp"]
        );
        // Same basename, different path: not the same file.
        assert!(
            shared(
                ("Touched: src/a/config.go", &[]),
                ("Touched: src/b/config.go", &[]),
            )
            .is_empty()
        );
        // A basename alone is not the full path.
        assert!(shared(("Touched: src/a/config.go", &[]), ("edit config.go", &[])).is_empty());
        // Case is part of a path.
        assert!(shared(("edited path.py", &[]), ("edited PATH.PY", &[])).is_empty());
        // A leading ./ is not.
        assert_eq!(
            shared(("see ./src/lib.rs", &[]), ("(src/lib.rs)", &[])),
            vec!["path:src/lib.rs"]
        );
        // A tag that is a path is that path.
        assert_eq!(
            shared(("commit", &["worktree.go"]), ("panic in worktree.go.", &[])),
            vec!["path:worktree.go"]
        );
    }

    #[test]
    fn file_path_shape_excludes_versions_abbreviations_and_words() {
        for path in [
            "path.py",
            "db.config.yaml",
            "src/a.c",
            "/usr/lib/x.so",
            "migrations/001.sql",
            ".github/workflows/ci.yml",
            "Node.js",
        ] {
            assert!(is_file_path(path), "{path} is a file path");
        }
        for not_path in [
            "1.2.3",
            "v1.4.1",
            "v1.4.x",
            "10.5kb",
            "e.g",
            "i.e",
            "U.S.A",
            "Ph.D",
            "and/or",
            "src/",
            "a//b.c",
            "Makefile",
            ".gitignore",
            "timeout",
            "foo.bar()",
        ] {
            assert!(!is_file_path(not_path), "{not_path} is not a file path");
        }
        // Sentence punctuation is not part of a token.
        assert_eq!(
            rendered("It timed out. See e.g. the log.", &[]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn commit_sha_must_be_the_whole_token() {
        let full = "6c231e84aa0f6c1d5e0d3b7a9c1f2e3d4b5a6978";
        assert_eq!(
            shared(
                (&format!("reverts {full}."), &[]),
                (&format!("Commit {}: fix", full.to_uppercase()), &[]),
            ),
            vec![format!("commit:{full}")]
        );
        assert_eq!(
            shared(
                ("Commit 6c231e84: validate", &[]),
                ("bisected to (6c231e84)", &[])
            ),
            vec!["commit:6c231e84"]
        );
        // An abbreviation is a different token from the full sha.
        assert!(shared((&format!("see {full}"), &[]), ("see 6c231e84", &[])).is_empty());
        // Numbers, hex-spelled words, short tokens and substrings are not shas.
        assert_eq!(
            rendered("1234567 defaced abc123 x6c231e84 6c231e84-dirty", &[]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn issue_reference_needs_owner_and_repo() {
        assert_eq!(
            shared(
                ("fixes go-git/go-git#2322.", &[]),
                (
                    "see https://github.com/go-git/go-git/issues/2322#issuecomment-1",
                    &[]
                ),
            ),
            vec!["issue:go-git/go-git#2322"]
        );
        assert_eq!(
            rendered("https://github.com/duckdb/duckdb/pull/19248", &[]),
            vec![
                "issue:duckdb/duckdb#19248",
                "url:https://github.com/duckdb/duckdb/pull/19248"
            ]
        );
        // A bare number names no repository; a different number is a
        // different issue.
        assert_eq!(
            rendered("Merge pull request #2137 (#2137)", &[]),
            Vec::<String>::new()
        );
        assert!(shared(("a/b#1", &[]), ("a/b#12", &[])).is_empty());
        assert_eq!(
            rendered("https://github.com/duckdb/duckdb/issues/", &[]),
            vec!["url:https://github.com/duckdb/duckdb/issues/"]
        );
    }

    #[test]
    fn url_is_byte_exact() {
        assert_eq!(
            shared(
                ("docs: <https://example.com/a/b?x=1>", &[]),
                ("see https://example.com/a/b?x=1.", &[]),
            ),
            vec!["url:https://example.com/a/b?x=1"]
        );
        assert!(
            shared(
                ("https://example.com/a", &[]),
                ("https://example.com/b", &[])
            )
            .is_empty()
        );
        assert_eq!(rendered("https://", &[]), Vec::<String>::new());
    }

    #[test]
    fn too_common_tags_are_named_and_do_not_join() {
        let lists: Vec<Vec<String>> = (0..=MAX_TAG_CARRIERS)
            .map(|i| {
                let mut tags = vec!["campaign".to_string(), "campaign".to_string()];
                if i < MAX_TAG_CARRIERS {
                    tags.push("at-the-limit".to_string());
                }
                tags
            })
            .collect();
        let common = too_common_tags(lists.iter().map(Vec::as_slice));
        // MAX_TAG_CARRIERS + 1 carriers is too common; exactly
        // MAX_TAG_CARRIERS is not.
        assert_eq!(
            common,
            BTreeMap::from([("campaign".to_string(), MAX_TAG_CARRIERS + 1)])
        );

        let set = |tags: &[&str], content: &str| -> BTreeSet<Identity> {
            let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
            extract_identities(content, &tags).into_iter().collect()
        };
        let left = set(&["campaign", "at-the-limit"], "Touched: src/a.rs");
        let right = set(&["campaign", "at-the-limit"], "Touched: src/a.rs");
        let joined: Vec<String> = joining_identities(&left, &right, &common)
            .iter()
            .map(Identity::to_string)
            .collect();
        assert_eq!(joined, vec!["tag:at-the-limit", "path:src/a.rs"]);
    }

    #[test]
    fn a_tag_that_is_a_path_counts_once() {
        let tags = vec!["path.py".to_string()];
        let set: BTreeSet<Identity> = extract_identities("", &tags).into_iter().collect();
        let joined = joining_identities(&set, &set, &BTreeMap::new());
        assert_eq!(joined.len(), 2, "{joined:?}");
        assert_eq!(distinct_values(&joined), 1);
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

    // ---- against a real Strata log ----

    fn store() -> (tempfile::TempDir, Arc<Storage>) {
        let dir = tempfile::tempdir().expect("temp dir");
        let storage = crate::strata_memory::open(dir.path()).expect("open strata log");
        (dir, storage)
    }

    /// Save one memory and run the ingest-time pass on it.
    fn save(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> (String, AutoConnectReport) {
        let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
        let node = storage
            .ingest_in_scope(
                IngestInput {
                    content: content.to_string(),
                    tags: tags.clone(),
                    ..Default::default()
                },
                "user",
            )
            .expect("ingest");
        let report = auto_connect_new_memory(storage.as_ref(), &node.id, "user", content, &tags)
            .expect("auto-connect");
        (node.id, report)
    }

    fn edges(storage: &Arc<Storage>) -> usize {
        storage.get_all_connections().expect("edges").len()
    }

    #[test]
    fn ingest_joins_on_exact_tag_and_reports_the_pair() {
        let (_dir, storage) = store();
        let (first, quiet) = save(&storage, "removed the npoints floor", &["euler"]);
        assert_eq!(quiet, AutoConnectReport::default());

        let (second, report) = save(&storage, "bends deform", &["bug", "euler"]);
        assert_eq!(report.edges, 1);
        assert_eq!(report.shared_identities, vec!["tag:euler"]);
        assert_eq!(
            report.pairs,
            vec![JoinedPair {
                source_id: first,
                target_id: second,
                identities: vec!["tag:euler".to_string()],
            }]
        );
        assert_eq!(edges(&storage), 1);
    }

    #[test]
    fn ingest_does_not_join_on_shared_words() {
        let (_dir, storage) = store();
        save(&storage, "Changed redis timeout to 5s in config", &["ops"]);
        // Shares the words redis and timeout with the first memory, and its
        // text names the first memory's tag as a word. No identity is shared.
        let (_, report) = save(&storage, "redis timeout errors, ops paged", &["bug"]);
        assert_eq!(report, AutoConnectReport::default());
        assert_eq!(edges(&storage), 0);
    }

    #[test]
    fn ingest_joins_a_path_in_the_text_to_the_same_path_recorded_as_a_tag() {
        let (_dir, storage) = store();
        let (commit, _) = save(
            &storage,
            "Commit 6c231e84: validate dot components. Touched: worktree.go",
            &["worktree.go"],
        );
        let (failure, report) = save(&storage, "checkout fails in worktree.go:412", &["failure"]);
        assert_eq!(report.edges, 1);
        assert_eq!(
            report.pairs,
            vec![JoinedPair {
                source_id: commit,
                target_id: failure,
                identities: vec!["path:worktree.go".to_string()],
            }]
        );
    }

    #[test]
    fn ingest_skips_a_tag_carried_by_more_than_the_limit() {
        let (_dir, storage) = store();
        // The first MAX_TAG_CARRIERS carriers are within the limit and join.
        for i in 0..MAX_TAG_CARRIERS {
            let (_, report) = save(&storage, &format!("batch memory {i}"), &["campaign"]);
            assert_eq!(report.edges, i, "carrier {i} joins the {i} before it");
            assert!(report.skipped_common_tags.is_empty());
        }
        let before = edges(&storage);
        assert_eq!(before, MAX_TAG_CARRIERS * (MAX_TAG_CARRIERS - 1) / 2);

        // One more makes the tag too common: it is named and joins nothing.
        let (_, report) = save(&storage, "one more batch memory", &["campaign"]);
        assert_eq!(report.edges, 0);
        assert_eq!(report.skipped_common_tags, vec!["campaign"]);
        assert_eq!(edges(&storage), before);

        // A path still joins a memory that also carries the common tag, and
        // the common tag is not listed as a reason.
        save(&storage, "Touched: src/gate.rs", &["gate.rs"]);
        let (_, report) = save(&storage, "panic at src/gate.rs:9 in gate.rs", &["campaign"]);
        assert_eq!(report.edges, 1);
        assert_eq!(
            report.shared_identities,
            vec!["path:gate.rs", "path:src/gate.rs"]
        );
        assert_eq!(report.skipped_common_tags, vec!["campaign"]);
    }

    #[test]
    fn ingest_keeps_edges_inside_the_scope() {
        let (_dir, storage) = store();
        storage
            .ingest_in_scope(
                IngestInput {
                    content: "other project".to_string(),
                    tags: vec!["euler".to_string()],
                    ..Default::default()
                },
                "elsewhere",
            )
            .expect("ingest");
        let (_, report) = save(&storage, "bends deform", &["euler"]);
        assert_eq!(report, AutoConnectReport::default());
        assert_eq!(edges(&storage), 0);
    }
}
