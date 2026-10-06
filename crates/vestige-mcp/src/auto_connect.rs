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
//! share a word and are NOT joined, and neither are two texts that both say
//! `connection_pool`. A tag is also classified as a token, so a tag `path.py`
//! is the same `path` identity as `path.py` in another memory's text.
//!
//! ## Which tags join: the hub rule and the edge budget
//!
//! A path, sha, issue reference or URL names one object, so two memories
//! recording it always join. A tag is a label, and a label can be on
//! everything. Whether a tag still says something about a pair is decided
//! from two counts the log holds: `N`, the memories of the scope carrying
//! the tag, and `M`, the memories of the scope. Nothing about a repository,
//! a tag's spelling or a particular commit enters either rule.
//!
//! 1. **The hub rule.** A tag carried by more than half the scope
//!    (`2 * N > M`) describes the scope rather than a pair: not carrying it
//!    is the rarer fact. It is skipped ([`is_hub_tag`]). "Half" needs a
//!    scope large enough to mean something, so a tag on at most
//!    [`SMALL_TAG_GROUP`] memories is never a hub; that number is not
//!    chosen, it is the largest group whose every pair fits one pass's edge
//!    budget (14 carriers are 91 pairs, 15 are 105, the budget is
//!    [`MAX_AUTO_EDGES`] = 100).
//! 2. **The edge budget.** A tag joins whole or not at all: there is no
//!    principled way to pick some of a tag's carriers, so a tag is never
//!    joined to a subset of them.
//!    - At ingest the new memory may receive at most [`MAX_AUTO_EDGES`]
//!      edges. Its exact references are counted first; then its tags are
//!      taken **fewest carriers first**, and each joins only if every edge
//!      it adds still fits what is left of the budget.
//!    - In the full scan (`vestige connect`) a tag on `N` memories joins
//!      `N * (N - 1) / 2` pairs, which is the quadratic blow-up the scan
//!      guards against: a tag on more than [`SMALL_TAG_GROUP`] memories
//!      joins only if those pairs fit the pass's `--max-edges`
//!      ([`scan_skipped_tags`]). With the default budget of 100 that is the
//!      same line as the hub floor.
//!
//! Every tag a pass did not join on is reported with `N`, `M` and the reason
//! ([`SkippedTag`]: `skipped tag X: carried by N of M (...)`). Nothing is
//! skipped silently.
//!
//! This replaces a flat cap of 14 carriers, which hid real joins in any
//! scope larger than a few dozen memories: a component tag on 18 of 79
//! records is specific (under a quarter of the scope) and was dropped. No
//! tag the flat cap let through is a hub under these rules.
//!
//! At ingest both rules read the scope as it is at the time of the write.
//! The log is append-only, so edges written before a tag grew into a hub
//! stay recorded; `vestige connect` applies the rules to the whole scope at
//! once.
//!
//! ## Strongest first, before the budget
//!
//! When the budget cannot hold every candidate (only exact references can
//! overflow it, since tags join whole), candidates are ranked before the cut
//! ([`rank_candidates`]): most distinct shared identities first, then the
//! rarer identities first (fewer carriers in the scope), then id. The same
//! order ranks the candidates of a causal walk, so the edges an ingest
//! writes are the ones the walk would list first. What the cut leaves
//! unlinked is counted in the report, and `vestige connect` writes it.
//!
//! ## Explainable edges
//!
//! Every edge is reported with the pair it joined and the exact identities
//! that joined it ([`JoinedPair`]), by `vestige ingest`, `vestige connect`
//! and the `smart_ingest` tool alike.
//!
//! ## How the ingest-time pass works
//!
//! Where `vestige connect` intersects every pair of the scope, auto-connect
//! runs on ONE memory:
//!
//! 1. Extract the new memory's identities once.
//! 2. Read the scope once ([`scan_scope`]) and, for each live memory, keep
//!    the identities it records that the new memory records too. The same
//!    extraction runs on both sides, so a path that appears only in the two
//!    texts joins exactly as one recorded as a tag does. The pass is linear
//!    in the scope and yields `N` for every identity and `M` for the scope.
//! 3. Apply the hub rule and the budget, rank, and write one `touched` edge
//!    per kept candidate through [`Storage::save_connection`], exactly as
//!    the connect command writes its edges. Pairs already joined by a
//!    recorded edge (either direction) are left alone.
//!
//! Two invariants are inherited from the connect command: the older memory
//! of a pair is the edge's source (the walk's rule for `touched`), and edges
//! stay within one scope. Idempotency rides the store's per-memory edge
//! index ([`Storage::get_connections_for_memory`]): a pair joined by any
//! recorded edge is left alone, so re-ingesting or re-running never stacks
//! parallel edges.
//!
//! An edge written here records that two memories name the same exact
//! thing. It is not a cause: a causal walk over these edges returns
//! hypotheses, never proven causes.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

use chrono::{DateTime, Utc};
use vestige_core::ConnectionRecord;
use vestige_core::storage::Storage;

/// The edge budget of one pass. One ingest-time auto-connect writes at most
/// this many `touched` edges, and `vestige connect` ships the same number as
/// its `--max-edges` default.
pub const MAX_AUTO_EDGES: usize = 100;

/// The largest tag group whose every pair fits one pass's edge budget:
/// `k * (k - 1) / 2 <= MAX_AUTO_EDGES` holds up to `k = 14` (91 pairs; 15
/// carriers are 105). A tag on at most this many memories is a small group:
/// it is never a hub, and the full scan always joins it.
pub const SMALL_TAG_GROUP: usize = largest_group_within(MAX_AUTO_EDGES);

/// The name [`SMALL_TAG_GROUP`] had when it was a flat cap on every tag. It
/// no longer caps anything by itself (see the module docs); it is kept so
/// code written against that name still builds.
pub const MAX_TAG_CARRIERS: usize = SMALL_TAG_GROUP;

/// The pairs among `carriers` memories, each joined to every other.
pub const fn pairs_among(carriers: usize) -> usize {
    carriers.saturating_mul(carriers.saturating_sub(1)) / 2
}

/// The largest group whose pairs fit `edges`.
const fn largest_group_within(edges: usize) -> usize {
    let mut carriers = 1;
    while pairs_among(carriers + 1) <= edges {
        carriers += 1;
    }
    carriers
}

/// The hub rule: is a tag carried by `carriers` of the `scope_size` memories
/// of a scope a hub? It is when more than half the scope carries it and it
/// is past the small group ([`SMALL_TAG_GROUP`]), below which "half the
/// scope" is a handful of memories and says nothing.
pub fn is_hub_tag(carriers: usize, scope_size: usize) -> bool {
    carriers > SMALL_TAG_GROUP && carriers.saturating_mul(2) > scope_size
}

/// The kind of an exact identity. Order matters: tags sort first, so an
/// identity list always reads tags, then references.
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
    /// `kind:value`, tags first, then sorted.
    pub identities: Vec<String>,
}

/// Why a pass did not join on a tag. Rendered as the reason in
/// `skipped tag X: carried by N of M (reason)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The hub rule: more than half the scope carries the tag.
    Hub,
    /// At ingest: joining the tag whole would add `needed` edges to the new
    /// memory, and only `left` of the write's `budget` remain.
    OverWriteBudget {
        needed: usize,
        left: usize,
        budget: usize,
    },
    /// In the full scan: the tag's `pairs` alone exceed the pass's
    /// `max_edges`.
    OverScanBudget { pairs: usize, max_edges: usize },
}

impl fmt::Display for SkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkipReason::Hub => write!(f, "more than half the scope carries it"),
            SkipReason::OverWriteBudget {
                needed,
                left,
                budget,
            } => write!(
                f,
                "joining it whole needs {needed} more edge(s), {left} left of the {budget}-edge budget of one write"
            ),
            SkipReason::OverScanBudget { pairs, max_edges } => write!(
                f,
                "its {pairs} pairs exceed the {max_edges}-edge budget of this pass"
            ),
        }
    }
}

/// A tag a pass did not join on: how many memories of the scope carry it
/// (the memory being written included), how many memories the scope holds,
/// and why. Rendered `skipped tag X: carried by N of M (reason)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedTag {
    pub tag: String,
    pub carriers: usize,
    pub scope_size: usize,
    pub reason: SkipReason,
}

impl fmt::Display for SkippedTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "skipped tag {}: carried by {} of {} ({})",
            self.tag, self.carriers, self.scope_size, self.reason
        )
    }
}

/// What one auto-connect pass did: the edges it wrote, strongest first, each
/// with the identities that joined it; the union of those identities
/// (sorted, deduplicated); and every tag it did not join on, with the reason.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutoConnectReport {
    pub edges: usize,
    pub pairs: Vec<JoinedPair>,
    pub shared_identities: Vec<String>,
    /// The names of the tags in `skipped`, sorted.
    pub skipped_common_tags: Vec<String>,
    /// Every tag the pass did not join on, sorted by tag: its carriers, the
    /// scope size and the reason (hub, or over the write's edge budget).
    pub skipped: Vec<SkippedTag>,
    /// Memories that share at least one counted identity with the new memory
    /// and were not already joined to it: the ranked set the budget applied
    /// to.
    pub candidates: usize,
    /// Ranked candidates the [`MAX_AUTO_EDGES`] budget left unlinked (always
    /// the weakest). `vestige connect` writes them.
    pub not_linked: usize,
}

/// One candidate for a `touched` edge or a walk rank: the memory, and the
/// exact identities it shares with the memory being explained, each with the
/// number of memories of the scope that record it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedCandidate {
    pub id: String,
    pub shared: Vec<(Identity, usize)>,
}

impl RankedCandidate {
    /// The carriers of each distinct shared value, fewest first. A tag that
    /// is itself a path is one shared thing recorded under two kinds; it
    /// counts once, at the smaller of its two carrier counts.
    pub fn rarity(&self) -> Vec<usize> {
        let mut by_value: BTreeMap<&str, usize> = BTreeMap::new();
        for (identity, carriers) in &self.shared {
            by_value
                .entry(identity.value.as_str())
                .and_modify(|least| *least = (*least).min(*carriers))
                .or_insert(*carriers);
        }
        let mut rarity: Vec<usize> = by_value.into_values().collect();
        rarity.sort_unstable();
        rarity
    }

    /// The part of the order the identities decide: more distinct shared
    /// values first, then the rarer values first (the fewest-carrier value
    /// of each candidate is compared first, then the next).
    pub fn strength(&self) -> (Reverse<usize>, Vec<usize>) {
        let rarity = self.rarity();
        (Reverse(rarity.len()), rarity)
    }
}

/// Order candidates strongest first: most distinct shared identity values,
/// then rarer identities (fewer carriers), then id. Nothing but the recorded
/// identities, their carrier counts and the id decides the order, so the
/// same store always ranks the same way.
pub fn rank_candidates(candidates: &mut [RankedCandidate]) {
    candidates.sort_by_cached_key(|candidate| (candidate.strength(), candidate.id.clone()));
}

/// One memory of a scope and the wanted identities it records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    pub id: String,
    pub created_at: DateTime<Utc>,
    /// The wanted identities this memory records; tags first, then sorted.
    pub identities: Vec<Identity>,
}

/// One pass over the live memories of a scope: how many there are, and which
/// of them record which of the wanted identities.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScopeScan {
    /// `M`: the live memories of the scope.
    pub scope_size: usize,
    /// The memories recording at least one wanted identity, by id.
    pub holders: Vec<Holder>,
}

impl ScopeScan {
    /// `N` for each wanted identity some memory records: how many memories
    /// of the scope record it.
    pub fn carriers(&self) -> BTreeMap<Identity, usize> {
        let mut carriers: BTreeMap<Identity, usize> = BTreeMap::new();
        for holder in &self.holders {
            for identity in &holder.identities {
                *carriers.entry(identity.clone()).or_insert(0) += 1;
            }
        }
        carriers
    }
}

/// Read a scope once and keep, for every live memory, the identities it
/// records that are in `wanted`. The same [`extract_identities`] runs on
/// every memory, so a match is an exact identity both sides record, whether
/// it came from a tag or from the text. Linear in the scope.
pub fn scan_scope(
    storage: &Storage,
    scope: &str,
    wanted: &BTreeSet<Identity>,
) -> Result<ScopeScan, String> {
    // One call: the store sorts and pages the whole scope per call, so
    // paging here would redo that work for every page.
    let nodes = storage
        .get_all_nodes_in_scope(scope, i32::MAX, 0)
        .map_err(|err| format!("could not read scope '{scope}': {err}"))?;
    let mut scan = ScopeScan {
        scope_size: nodes.len(),
        holders: Vec::new(),
    };
    for node in nodes {
        let identities: Vec<Identity> = extract_identities(&node.content, &node.tags)
            .into_iter()
            .filter(|identity| wanted.contains(identity))
            .collect();
        if !identities.is_empty() {
            scan.holders.push(Holder {
                id: node.id,
                created_at: node.created_at,
                identities,
            });
        }
    }
    scan.holders.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(scan)
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

    let mine: BTreeSet<Identity> = extract_identities(content, tags).into_iter().collect();
    let mut report = AutoConnectReport::default();
    if mine.is_empty() {
        return Ok(report);
    }

    // Pairs the new memory already shares a recorded edge with (either
    // direction), normalized so id order cannot hide a duplicate. This is
    // the per-memory slice of the connect command's joined-pair set, read
    // from the store's edge index instead of the whole edge list.
    let joined: HashSet<(String, String)> = storage
        .get_connections_for_memory(memory_id)
        .map_err(|err| format!("auto-connect could not read {memory_id}'s edges: {err}"))?
        .into_iter()
        .filter_map(|edge| pair_key(&edge.source_id, &edge.target_id))
        .collect();

    // One pass over the scope. Edges stay within one scope, the invariant
    // the connect command and declared links both keep, so memories of
    // other scopes are neither candidates nor counted.
    let scan = scan_scope(storage, scope, &mine)
        .map_err(|err| format!("auto-connect {err}"))?;
    // `N` and `M` count the new memory too: it is one of the scope's
    // memories and records every one of its own identities, whether or not
    // the scan listed it.
    let listed = scan.holders.iter().any(|holder| holder.id == memory_id);
    let scope_size = scan.scope_size + usize::from(!listed);
    let others: Vec<&Holder> = scan
        .holders
        .iter()
        .filter(|holder| holder.id != memory_id)
        .collect();
    let mut carriers: BTreeMap<Identity, usize> =
        mine.iter().map(|identity| (identity.clone(), 1)).collect();
    for other in &others {
        for identity in &other.identities {
            if let Some(count) = carriers.get_mut(identity) {
                *count += 1;
            }
        }
    }

    // The memories an edge could still be written to.
    let open: Vec<&Holder> = others
        .iter()
        .copied()
        .filter(|holder| pair_key(memory_id, &holder.id).is_some_and(|key| !joined.contains(&key)))
        .collect();

    // What the new memory may join on. Exact references (path, commit,
    // issue, URL) always count; every memory sharing one is a candidate.
    let mut evidence: BTreeSet<Identity> = mine
        .iter()
        .filter(|identity| identity.kind != IdentityKind::Tag)
        .cloned()
        .collect();
    let mut selected: BTreeSet<&str> = open
        .iter()
        .filter(|holder| holder.identities.iter().any(|i| evidence.contains(i)))
        .map(|holder| holder.id.as_str())
        .collect();

    // Tags: fewest carriers first, each whole or not at all. A hub is
    // skipped; so is a tag whose every new edge no longer fits the budget.
    let mut own_tags: Vec<&Identity> = mine
        .iter()
        .filter(|identity| identity.kind == IdentityKind::Tag)
        .collect();
    own_tags.sort_by(|a, b| {
        carriers[*a]
            .cmp(&carriers[*b])
            .then_with(|| a.value.cmp(&b.value))
    });
    for tag in own_tags {
        let carried_by = carriers[tag];
        if is_hub_tag(carried_by, scope_size) {
            report.skipped.push(SkippedTag {
                tag: tag.value.clone(),
                carriers: carried_by,
                scope_size,
                reason: SkipReason::Hub,
            });
            continue;
        }
        let fresh: Vec<&str> = open
            .iter()
            .filter(|holder| {
                !selected.contains(holder.id.as_str()) && holder.identities.contains(tag)
            })
            .map(|holder| holder.id.as_str())
            .collect();
        let left = MAX_AUTO_EDGES.saturating_sub(selected.len());
        if fresh.len() > left {
            report.skipped.push(SkippedTag {
                tag: tag.value.clone(),
                carriers: carried_by,
                scope_size,
                reason: SkipReason::OverWriteBudget {
                    needed: fresh.len(),
                    left,
                    budget: MAX_AUTO_EDGES,
                },
            });
            continue;
        }
        evidence.insert(tag.clone());
        selected.extend(fresh);
    }
    report.skipped.sort_by(|a, b| a.tag.cmp(&b.tag));
    report.skipped_common_tags = report
        .skipped
        .iter()
        .map(|skipped| skipped.tag.clone())
        .collect();

    // Rank before the budget: each candidate with the counted identities it
    // shares, strongest first.
    let mut ranked: Vec<RankedCandidate> = open
        .iter()
        .filter(|holder| selected.contains(holder.id.as_str()))
        .map(|holder| RankedCandidate {
            id: holder.id.clone(),
            shared: holder
                .identities
                .iter()
                .filter(|identity| evidence.contains(*identity))
                .map(|identity| (identity.clone(), carriers[identity]))
                .collect(),
        })
        .collect();
    rank_candidates(&mut ranked);
    report.candidates = ranked.len();
    report.not_linked = ranked.len().saturating_sub(MAX_AUTO_EDGES);

    let created_at: BTreeMap<&str, DateTime<Utc>> = open
        .iter()
        .map(|holder| (holder.id.as_str(), holder.created_at))
        .collect();
    let now = Utc::now();
    let mut shared_seen: BTreeSet<String> = BTreeSet::new();
    for candidate in ranked.into_iter().take(MAX_AUTO_EDGES) {
        let Some(candidate_created) = created_at.get(candidate.id.as_str()) else {
            continue;
        };
        let shared: Vec<String> = candidate
            .shared
            .iter()
            .map(|(identity, _)| identity.to_string())
            .collect();
        // Direction follows the walk's rule for `touched`: the source is
        // the earlier record, so from the target the walk goes to the
        // source. Id order breaks creation-time ties, as in the connect
        // command's sort.
        let (source_id, target_id) = earlier_first(
            (memory_id, new_node.created_at),
            (candidate.id.as_str(), *candidate_created),
        );
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
        storage.save_connection(&edge).map_err(|err| {
            format!(
                "auto-connect edge {} was not admitted: {err}",
                candidate.id
            )
        })?;
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

/// A pair of `(id, creation time)` ordered (source, target) with the earlier
/// record first: creation time, then id, the same order the connect command
/// sorts by.
fn earlier_first(left: (&str, DateTime<Utc>), right: (&str, DateTime<Utc>)) -> (String, String) {
    let left_first = left.1.cmp(&right.1).then_with(|| left.0.cmp(right.0))
        == std::cmp::Ordering::Less;
    if left_first {
        (left.0.to_string(), right.0.to_string())
    } else {
        (right.0.to_string(), left.0.to_string())
    }
}

/// The tags a full scan of a scope (`vestige connect`) does not join on,
/// sorted by tag. Each item of `tag_lists` is one memory's tag list; a tag
/// repeated on one memory counts once.
///
/// A tag on `N` of the scope's `M` memories joins `N * (N - 1) / 2` pairs. A
/// small group (`N <= SMALL_TAG_GROUP`) always joins. Past that the tag is
/// skipped when it is a hub ([`is_hub_tag`]), or when its pairs alone exceed
/// `max_edges`, the edge budget of the pass: the guard against a quadratic
/// blow-up, stated in the caller's own budget.
pub fn scan_skipped_tags<'a>(
    tag_lists: impl IntoIterator<Item = &'a [String]>,
    max_edges: usize,
) -> Vec<SkippedTag> {
    let mut scope_size = 0usize;
    let mut carriers: BTreeMap<&'a str, usize> = BTreeMap::new();
    for tags in tag_lists {
        scope_size += 1;
        let distinct: BTreeSet<&'a str> = tags.iter().map(String::as_str).collect();
        for tag in distinct {
            *carriers.entry(tag).or_insert(0) += 1;
        }
    }
    carriers
        .into_iter()
        .filter(|(_, count)| *count > SMALL_TAG_GROUP)
        .filter_map(|(tag, count)| {
            let reason = if is_hub_tag(count, scope_size) {
                SkipReason::Hub
            } else if pairs_among(count) > max_edges {
                SkipReason::OverScanBudget {
                    pairs: pairs_among(count),
                    max_edges,
                }
            } else {
                return None;
            };
            Some(SkippedTag {
                tag: tag.to_string(),
                carriers: count,
                scope_size,
                reason,
            })
        })
        .collect()
}

/// The tags a full scan with the default edge budget ([`MAX_AUTO_EDGES`])
/// does not join on, with their carrier counts: [`scan_skipped_tags`]
/// reduced to the map [`joining_identities`] takes. With that budget every
/// tag on more than [`SMALL_TAG_GROUP`] memories is skipped.
pub fn too_common_tags<'a>(
    tag_lists: impl IntoIterator<Item = &'a [String]>,
) -> BTreeMap<String, usize> {
    scan_skipped_tags(tag_lists, MAX_AUTO_EDGES)
        .into_iter()
        .map(|skipped| (skipped.tag, skipped.carriers))
        .collect()
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

/// Could this raw token be an identity at all? A cheap necessary condition
/// that lets [`classify_token`] drop a prose word without classifying it; it
/// decides nothing by itself. Every identity kind leaves a mark in the raw
/// bytes: a URL has the `:` of its scheme, an issue reference its `#`, a
/// file path the `.` before its extension, and a sha at least seven hex
/// digits.
fn may_be_identity(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    bytes.iter().any(|b| matches!(b, b'.' | b'#' | b':'))
        || bytes.iter().filter(|b| b.is_ascii_hexdigit()).count() >= 7
}

/// Classify one whitespace-delimited token. Surrounding quotes, brackets and
/// sentence punctuation are not part of the token. Returns nothing for a
/// prose word.
fn classify_token(raw: &str) -> Vec<Identity> {
    if !may_be_identity(raw) {
        return Vec::new();
    }
    classify_marked_token(raw)
}

/// [`classify_token`] without the cheap first check: the whole definition of
/// what a token is.
fn classify_marked_token(raw: &str) -> Vec<Identity> {
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
        // A code-shaped word both texts use (`connection_pool`) is still a
        // word: it joins nothing either.
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
    fn the_cheap_first_check_never_hides_an_identity() {
        // Every token the full definition classifies passes the first check,
        // and a token the check drops classifies to nothing: the check only
        // saves work.
        for token in [
            "path.py",
            "`src/execution/index/art/art.cpp:214:9`,",
            "./src/lib.rs",
            "(src/lib.rs)",
            "6c231e84",
            "(6C231E84)",
            "6c231e84aa0f6c1d5e0d3b7a9c1f2e3d4b5a6978.",
            "go-git/go-git#2322.",
            "<https://example.com/a/b?x=1>",
            "http://localhost/x",
            "https://github.com/duckdb/duckdb/pull/19248",
            ".github/workflows/ci.yml",
        ] {
            assert!(
                !classify_marked_token(token).is_empty(),
                "{token} is an identity"
            );
            assert!(may_be_identity(token), "{token} must pass the first check");
            assert_eq!(classify_token(token), classify_marked_token(token));
        }
        for token in [
            "connection",
            "retry",
            "socket_connect_timeout",
            "ConnectionPool",
            "euler()",
            "and/or",
            "Makefile",
            "abc123",
            "4337",
            "defaced",
            "(#2137)",
            "note:",
            "1.2.3",
            "",
        ] {
            assert!(
                classify_marked_token(token).is_empty(),
                "{token} is not an identity"
            );
            assert!(classify_token(token).is_empty());
        }
    }

    #[test]
    fn the_small_group_is_the_largest_whose_pairs_fit_one_pass() {
        assert_eq!(MAX_AUTO_EDGES, 100);
        assert_eq!(SMALL_TAG_GROUP, 14);
        assert_eq!(MAX_TAG_CARRIERS, SMALL_TAG_GROUP);
        assert_eq!(pairs_among(SMALL_TAG_GROUP), 91);
        assert_eq!(pairs_among(SMALL_TAG_GROUP + 1), 105);
        assert_eq!(pairs_among(0), 0);
        assert_eq!(pairs_among(1), 0);
        assert_eq!(pairs_among(2), 1);
    }

    #[test]
    fn a_hub_is_a_tag_more_than_half_the_scope_carries() {
        // A campaign tag on 78 of 79 memories, a directory tag on 60 of 79.
        assert!(is_hub_tag(78, 79));
        assert!(is_hub_tag(60, 79));
        // 40 of 79 is more than half; 39 of 79 is not.
        assert!(is_hub_tag(40, 79));
        assert!(!is_hub_tag(39, 79));
        // The case the flat cap of 14 dropped: 18 commits plus the report
        // carry the tag, 19 of 79, under a quarter of the scope.
        assert!(!is_hub_tag(19, 79));
        // Exactly half is not more than half.
        assert!(!is_hub_tag(15, 30));
        assert!(is_hub_tag(16, 30));
        // A small group is never a hub, even when it is the whole scope.
        assert!(!is_hub_tag(2, 2));
        assert!(!is_hub_tag(14, 14));
        assert!(is_hub_tag(15, 15));
    }

    fn tag_lists(scope: usize, tags: &[(&str, usize)]) -> Vec<Vec<String>> {
        (0..scope)
            .map(|i| {
                tags.iter()
                    .filter(|(_, carriers)| i < *carriers)
                    .map(|(tag, _)| tag.to_string())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_full_scan_skips_hubs_and_tags_whose_pairs_exceed_its_budget() {
        let lists = tag_lists(
            79,
            &[
                ("campaign", 78),
                ("directory", 60),
                ("component", 19),
                ("group", 14),
            ],
        );
        let skipped =
            |max_edges| scan_skipped_tags(lists.iter().map(Vec::as_slice), max_edges);

        // Default budget: 19 carriers are 171 pairs, more than 100.
        let rendered: Vec<String> = skipped(MAX_AUTO_EDGES)
            .iter()
            .map(SkippedTag::to_string)
            .collect();
        assert_eq!(
            rendered,
            vec![
                "skipped tag campaign: carried by 78 of 79 (more than half the scope carries it)",
                "skipped tag component: carried by 19 of 79 (its 171 pairs exceed the 100-edge budget of this pass)",
                "skipped tag directory: carried by 60 of 79 (more than half the scope carries it)",
            ]
        );
        // A budget that holds the 171 pairs joins the component tag; a hub
        // stays skipped whatever the budget.
        let names: Vec<String> = skipped(171).into_iter().map(|s| s.tag).collect();
        assert_eq!(names, vec!["campaign", "directory"]);
        let names: Vec<String> = skipped(170).into_iter().map(|s| s.tag).collect();
        assert_eq!(names, vec!["campaign", "component", "directory"]);
        // A small group joins even under a budget smaller than its pairs:
        // the scan's own --max-edges cut then applies, and is reported.
        assert!(skipped(1).iter().all(|s| s.tag != "group"));
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

    fn candidate(id: &str, shared: &[(IdentityKind, &str, usize)]) -> RankedCandidate {
        RankedCandidate {
            id: id.to_string(),
            shared: shared
                .iter()
                .map(|(kind, value, carriers)| (Identity::new(*kind, *value), *carriers))
                .collect(),
        }
    }

    #[test]
    fn candidates_rank_by_shared_identities_then_rarity_then_id() {
        use IdentityKind::{Path, Tag};
        let mut ranked = vec![
            // One shared tag, 19 carriers.
            candidate("mem-01", &[(Tag, "connection", 19)]),
            candidate("mem-02", &[(Tag, "connection", 19)]),
            // One shared tag, 2 carriers: rarer, so ahead of the two above.
            candidate("mem-03", &[(Tag, "retry", 2)]),
            // Two shared identities: ahead of every single one.
            candidate("mem-04", &[(Tag, "connection", 19), (Tag, "retry", 2)]),
            // A tag that is itself a path is one shared thing, not two.
            candidate("mem-05", &[(Tag, "path.py", 3), (Path, "path.py", 5)]),
            // Two shared, the rarest of them rarer than mem-04's rarest.
            candidate(
                "mem-06",
                &[(Path, "redis/connection.py", 1), (Tag, "connection", 19)],
            ),
            // Two shared, same rarest as mem-04, the second one rarer.
            candidate("mem-07", &[(Tag, "retry", 2), (Tag, "backoff", 4)]),
            // Nothing shared (reached by a declared link): last.
            candidate("mem-00", &[]),
        ];
        rank_candidates(&mut ranked);
        let order: Vec<String> = ranked.iter().map(|c| c.id.clone()).collect();
        assert_eq!(
            order,
            vec![
                "mem-06", "mem-07", "mem-04", "mem-03", "mem-05", "mem-01", "mem-02", "mem-00"
            ]
        );
        assert_eq!(ranked[4].rarity(), vec![3], "path.py counts once");
        // The same input in any order ranks the same way.
        ranked.reverse();
        rank_candidates(&mut ranked);
        let again: Vec<String> = ranked.iter().map(|c| c.id.clone()).collect();
        assert_eq!(again, order);
    }

    #[test]
    #[ignore]
    fn tmp_probe_scan_cost() {
        // TEMPORARY measurement, removed before commit.
        let words = [
            "the", "connection", "retry", "was", "added", "when", "server", "timeout",
            "regression", "since", "version", "tests", "stopped", "passing", "handler",
            "smart_ingest", "causal_walk", "ConnectionPool", "note:", "(see", "below).",
        ];
        let mut memories: Vec<(String, Vec<String>)> = Vec::new();
        for i in 0..10_000usize {
            let mut text = String::with_capacity(2200);
            let mut n = i;
            while text.len() < 2000 {
                n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let pick = (n >> 33) as usize;
                if pick % 40 == 0 {
                    text.push_str(&format!("crates/mod{}/src/file{}.rs:{} ", pick % 50, pick % 300, pick % 900));
                } else if pick % 97 == 0 {
                    text.push_str(&format!("https://example.com/a/{} ", pick % 1000));
                } else {
                    text.push_str(words[pick % words.len()]);
                    text.push(' ');
                }
            }
            memories.push((text, vec!["vestige".to_string(), format!("topic-{}", i % 400)]));
        }
        let bytes: usize = memories.iter().map(|(t, _)| t.len()).sum();
        let wanted: BTreeSet<Identity> = extract_identities(&memories[17].0, &memories[17].1)
            .into_iter()
            .collect();
        for round in 0..3 {
            let started = std::time::Instant::now();
            let mut holders = 0usize;
            let mut total = 0usize;
            for (text, tags) in &memories {
                let found = extract_identities(text, tags);
                total += found.len();
                if found.iter().any(|identity| wanted.contains(identity)) {
                    holders += 1;
                }
            }
            eprintln!(
                "round {round}: {} memories, {} MB, {} identities, {} holders, {:?}",
                memories.len(),
                bytes / 1_000_000,
                total,
                holders,
                started.elapsed()
            );
        }
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

    /// Save one memory without connecting it: a memory that was already in
    /// the scope before the write under test.
    fn put(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> String {
        storage
            .ingest_in_scope(
                IngestInput {
                    content: content.to_string(),
                    tags: tags.iter().map(|t| t.to_string()).collect(),
                    ..Default::default()
                },
                "user",
            )
            .expect("ingest")
            .id
    }

    /// Save one memory and run the ingest-time pass on it.
    fn save(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> (String, AutoConnectReport) {
        let id = put(storage, content, tags);
        let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
        let report = auto_connect_new_memory(storage.as_ref(), &id, "user", content, &tags)
            .expect("auto-connect");
        (id, report)
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
        assert!(report.skipped.is_empty());
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

    /// The path is a tag of neither memory: it appears only in the two
    /// texts. The ingest itself writes the join; no `vestige connect` runs.
    #[test]
    fn ingest_joins_a_path_that_appears_only_in_the_two_texts() {
        let (_dir, storage) = store();
        // Tagged the way a commit importer tags: folder segments, file name
        // and stem, never the full path.
        let (commit, _) = save(
            &storage,
            "Commit 1a2b3c4: fix script reload. Touched: modules/mono/csharp_script.cpp modules/mono/csharp_script.h",
            &["chal-commit", "modules", "mono", "csharp_script.cpp", "csharp_script"],
        );
        let (same_name_elsewhere, _) = save(
            &storage,
            "Commit 5d6e7f8: port. Touched: modules/gdscript/csharp_script.cpp",
            &["chal-commit", "modules", "gdscript"],
        );
        let (other, _) = save(
            &storage,
            "Commit 9a8b7c6: docs. Touched: doc/classes/Node.xml",
            &["chal-commit", "doc", "classes"],
        );
        let before = edges(&storage);

        let (failure, report) = save(
            &storage,
            "Crash on reload, trace ends at modules/mono/csharp_script.cpp:2345 in CSharpScript::reload",
            &["challenge", "failure"],
        );
        assert_eq!(report.edges, 1, "{report:?}");
        assert_eq!(
            report.pairs,
            vec![JoinedPair {
                source_id: commit,
                target_id: failure.clone(),
                identities: vec!["path:modules/mono/csharp_script.cpp".to_string()],
            }]
        );
        assert_eq!(
            report.shared_identities,
            vec!["path:modules/mono/csharp_script.cpp"]
        );
        assert_eq!(edges(&storage), before + 1);
        // Same file name under another folder is another file; a commit
        // that names neither is not joined.
        let touching: Vec<String> = storage
            .get_connections_for_memory(&failure)
            .expect("edges")
            .into_iter()
            .map(|edge| edge.source_id)
            .collect();
        assert!(!touching.contains(&same_name_elsewhere));
        assert!(!touching.contains(&other));

        // Nothing is left for a second pass over the same memory.
        let tags: Vec<String> = vec!["challenge".to_string(), "failure".to_string()];
        let again = auto_connect_new_memory(
            storage.as_ref(),
            &failure,
            "user",
            "Crash on reload, trace ends at modules/mono/csharp_script.cpp:2345 in CSharpScript::reload",
            &tags,
        )
        .expect("auto-connect");
        assert_eq!(again.edges, 0);
        assert_eq!(edges(&storage), before + 1);
    }

    /// The shape the flat cap of 14 carriers broke: a failure report tagged
    /// with a component that 18 of 78 commit records carry. 19 of 79 is not
    /// a hub, and 18 edges fit one write, so the report is joined to every
    /// one of them; the campaign tag on 78 of 79 is a hub and is named.
    #[test]
    fn ingest_joins_a_component_tag_on_18_of_79_and_skips_the_hub() {
        let (_dir, storage) = store();
        let mut commits = Vec::new();
        for i in 0..78 {
            let mut tags = vec!["chal-commit".to_string(), format!("only-{i}")];
            if i % 4 == 0 && commits.len() < 18 {
                tags.push("connection".to_string());
            }
            if i == 8 {
                tags.push("retry".to_string());
            }
            let tags: Vec<&str> = tags.iter().map(String::as_str).collect();
            let id = put(&storage, &format!("commit record {i}"), &tags);
            if tags.contains(&"connection") {
                commits.push(id);
            }
        }
        assert_eq!(commits.len(), 18);

        let (failure, report) = save(
            &storage,
            "connecting takes very long since the upgrade",
            &["challenge", "failure", "connection", "retry", "backoff"],
        );
        assert_eq!(report.edges, 18, "{report:?}");
        assert_eq!(report.candidates, 18);
        assert_eq!(report.not_linked, 0);
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
        assert_eq!(
            report.shared_identities,
            vec!["tag:connection", "tag:retry"]
        );
        // Strongest first: the one commit sharing two tags, then the rest
        // in id order, each with the identity that joined it.
        assert_eq!(report.pairs[0].source_id, commits[2]);
        assert_eq!(
            report.pairs[0].identities,
            vec!["tag:connection".to_string(), "tag:retry".to_string()]
        );
        let rest: Vec<&str> = report.pairs[1..]
            .iter()
            .map(|pair| pair.source_id.as_str())
            .collect();
        let expected: Vec<&str> = commits
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != 2)
            .map(|(_, id)| id.as_str())
            .collect();
        assert_eq!(rest, expected);
        for pair in &report.pairs {
            assert_eq!(pair.target_id, failure);
        }

        // One more commit record: the campaign tag is on 79 of 80 and is
        // skipped by name with both counts; the component tag joins it to
        // its 19 other carriers.
        let (_, report) = save(
            &storage,
            "commit record 78",
            &["chal-commit", "connection"],
        );
        assert_eq!(report.edges, 19, "{report:?}");
        assert_eq!(report.shared_identities, vec!["tag:connection"]);
        assert_eq!(report.skipped_common_tags, vec!["chal-commit"]);
        assert_eq!(
            report.skipped,
            vec![SkippedTag {
                tag: "chal-commit".to_string(),
                carriers: 79,
                scope_size: 80,
                reason: SkipReason::Hub,
            }]
        );
        assert_eq!(
            report.skipped[0].to_string(),
            "skipped tag chal-commit: carried by 79 of 80 (more than half the scope carries it)"
        );
    }

    /// A small group always joins, and one carrier more than half of a
    /// small scope makes the tag a hub: the behavior of the flat cap, kept
    /// where it was right.
    #[test]
    fn ingest_joins_a_small_group_and_skips_the_tag_once_it_is_a_hub() {
        let (_dir, storage) = store();
        for i in 0..SMALL_TAG_GROUP {
            let (_, report) = save(&storage, &format!("batch memory {i}"), &["campaign"]);
            assert_eq!(report.edges, i, "carrier {i} joins the {i} before it");
            assert!(report.skipped.is_empty());
        }
        let before = edges(&storage);
        assert_eq!(before, pairs_among(SMALL_TAG_GROUP));

        // The fifteenth carrier: 15 of 15 memories. Named, joins nothing.
        let (_, report) = save(&storage, "one more batch memory", &["campaign"]);
        assert_eq!(report.edges, 0);
        assert_eq!(report.skipped_common_tags, vec!["campaign"]);
        assert_eq!(
            report.skipped[0].to_string(),
            "skipped tag campaign: carried by 15 of 15 (more than half the scope carries it)"
        );
        assert_eq!(edges(&storage), before);

        // A path still joins a memory that also carries the hub tag, and
        // the hub tag is not listed as a reason.
        save(&storage, "Touched: src/gate.rs", &["gate.rs"]);
        let (_, report) = save(&storage, "panic at src/gate.rs:9 in gate.rs", &["campaign"]);
        assert_eq!(report.edges, 1);
        assert_eq!(
            report.shared_identities,
            vec!["path:gate.rs", "path:src/gate.rs"]
        );
        assert_eq!(report.skipped_common_tags, vec!["campaign"]);
    }

    /// The budget of one write is spent on the tags with the fewest carriers
    /// first, and a tag joins whole or not at all: the widest tag no longer
    /// fits and is named with what it needed and what was left. Taken in
    /// name order instead, the widest tag would have used the budget and the
    /// middle one would have been dropped.
    #[test]
    fn ingest_takes_tags_with_the_fewest_carriers_first_under_the_edge_budget() {
        let (_dir, storage) = store();
        for i in 0..60 {
            put(&storage, &format!("wide {i}"), &["a-wide"]);
        }
        for i in 0..50 {
            put(&storage, &format!("mid {i}"), &["b-mid"]);
        }
        for i in 0..2 {
            put(&storage, &format!("rare {i}"), &["c-rare"]);
        }
        for i in 0..20 {
            put(&storage, &format!("filler {i}"), &[]);
        }

        let (_, report) = save(&storage, "the new memory", &["a-wide", "b-mid", "c-rare"]);
        assert_eq!(report.edges, 52, "{:?}", report.skipped);
        assert_eq!(report.not_linked, 0);
        assert_eq!(report.shared_identities, vec!["tag:b-mid", "tag:c-rare"]);
        assert_eq!(
            report.skipped,
            vec![SkippedTag {
                tag: "a-wide".to_string(),
                carriers: 61,
                scope_size: 133,
                reason: SkipReason::OverWriteBudget {
                    needed: 60,
                    left: 48,
                    budget: MAX_AUTO_EDGES,
                },
            }]
        );
        assert_eq!(
            report.skipped[0].to_string(),
            "skipped tag a-wide: carried by 61 of 133 (joining it whole needs 60 more edge(s), 48 left of the 100-edge budget of one write)"
        );
        // The rarest tag's carriers are linked first.
        assert_eq!(report.pairs[0].identities, vec!["tag:c-rare".to_string()]);
        assert_eq!(report.pairs[1].identities, vec!["tag:c-rare".to_string()]);
        assert_eq!(report.pairs[2].identities, vec!["tag:b-mid".to_string()]);
        assert_eq!(edges(&storage), 52);
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

    /// Exact references always count, so they alone can overflow the budget.
    /// The cut keeps the strongest candidates and counts the rest.
    #[test]
    fn the_edge_budget_keeps_the_strongest_candidates_and_counts_the_rest() {
        let (_dir, storage) = store();
        // More single-path candidates than the budget, written first...
        let mut weak = Vec::new();
        for i in 0..MAX_AUTO_EDGES + 5 {
            weak.push(put(
                &storage,
                &format!("commit {i}: noise. Touched: src/hot.rs"),
                &[],
            ));
        }
        // ...and one that shares two paths, written last (the highest id).
        let strong = put(
            &storage,
            "commit strong. Touched: src/hot.rs src/rare.rs",
            &[],
        );
        let (_, report) = save(&storage, "fails in src/hot.rs and src/rare.rs", &[]);
        assert_eq!(report.candidates, MAX_AUTO_EDGES + 6);
        assert_eq!(report.edges, MAX_AUTO_EDGES);
        assert_eq!(report.not_linked, 6);
        assert!(report.skipped.is_empty());
        // The strongest candidate is linked first, with both identities;
        // the rest follow in id order until the budget is spent.
        assert_eq!(report.pairs[0].source_id, strong);
        assert_eq!(
            report.pairs[0].identities,
            vec![
                "path:src/hot.rs".to_string(),
                "path:src/rare.rs".to_string()
            ]
        );
        assert_eq!(report.pairs[1].source_id, weak[0]);
        assert_eq!(
            report.pairs[MAX_AUTO_EDGES - 1].source_id,
            weak[MAX_AUTO_EDGES - 2]
        );
        assert_eq!(edges(&storage), MAX_AUTO_EDGES);
    }
}
