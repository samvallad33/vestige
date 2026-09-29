//! Typed edge vocabulary and traversal on `memory_connections`.
//!
//! `memory_connections` (V3) is a generic activation-network edge table whose
//! `link_type` has always been a free-form string (`semantic`, `temporal`,
//! ...). That freedom is fine for spreading activation but useless for
//! provenance walks: nothing can rely on what an edge *means*.
//!
//! This module layers the owner-approved 8-type vocabulary on top of the same
//! table (V39 adds the `edge_meta` JSON payload, the `created_by_run`
//! provenance column, and the `(link_type, target_id)` index):
//!
//! | type           | intent                                        | gating        |
//! |----------------|-----------------------------------------------|---------------|
//! | `touched`      | commit → file/symbol                          | declared      |
//! | `anchored_to`  | record ↔ file/symbol/span                     | declared      |
//! | `derived_from` | record → record                               | declared only |
//! | `supersedes`   | record → record                               | review-gated  |
//! | `corrects`     | record → record                               | review-gated  |
//! | `closed_by`    | issue → PR → commit → check                   | declared      |
//! | `projected_to` | projection trail (pre-existing concept)       | declared      |
//! | `evidence_of`  | walk-trail edge; replaces `backfill_candidate`| declared      |
//!
//! Vocabulary membership is validated in Rust at write time
//! ([`SqliteMemoryStore::save_typed_edge`]), not by a SQL CHECK, so legacy
//! rows keep validating and the vocabulary can grow without a schema
//! rewrite. `supersedes` intentionally coexists with the
//! `knowledge_nodes.superseded_by` bitemporal column from V14: the column is
//! the merge/supersede *state*, the edge type is the review-gated *trail*.
//! The column is NOT migrated into edges.
//!
//! Purge tombstones chose the table design (`purge_tombstones`, V39) over the
//! synthetic `tombstone_of` marker-node edge: a marker node would need a fake
//! `knowledge_nodes` row (content is NOT NULL) and would fight the ON DELETE
//! CASCADE foreign keys that edges live under.

use chrono::{DateTime, Utc};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet, VecDeque};

use super::sqlite::SqliteMemoryStore;
use super::{Result, StorageError};

/// The owner-approved typed-edge vocabulary, in canonical order.
pub const TYPED_EDGE_VOCABULARY: &[&str] = &[
    "touched",
    "anchored_to",
    "derived_from",
    "supersedes",
    "corrects",
    "closed_by",
    "projected_to",
    "evidence_of",
];

/// A typed edge kind from the owner-approved vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// commit → file/symbol
    Touched,
    /// record ↔ file/symbol/span
    AnchoredTo,
    /// record → record, declared only (never inferred)
    DerivedFrom,
    /// record → record, review-gated invalidation trail
    Supersedes,
    /// record → record, review-gated correction trail
    Corrects,
    /// issue → PR → commit → check
    ClosedBy,
    /// projection trail (pre-existing concept, now first-class)
    ProjectedTo,
    /// walk-trail evidence edge; replaces `backfill_candidate` for new writes
    EvidenceOf,
}

impl EdgeKind {
    /// Every vocabulary member, in [`TYPED_EDGE_VOCABULARY`] order.
    pub const ALL: [EdgeKind; 8] = [
        EdgeKind::Touched,
        EdgeKind::AnchoredTo,
        EdgeKind::DerivedFrom,
        EdgeKind::Supersedes,
        EdgeKind::Corrects,
        EdgeKind::ClosedBy,
        EdgeKind::ProjectedTo,
        EdgeKind::EvidenceOf,
    ];

    /// The canonical `link_type` string stored in `memory_connections`.
    pub fn as_str(self) -> &'static str {
        match self {
            EdgeKind::Touched => "touched",
            EdgeKind::AnchoredTo => "anchored_to",
            EdgeKind::DerivedFrom => "derived_from",
            EdgeKind::Supersedes => "supersedes",
            EdgeKind::Corrects => "corrects",
            EdgeKind::ClosedBy => "closed_by",
            EdgeKind::ProjectedTo => "projected_to",
            EdgeKind::EvidenceOf => "evidence_of",
        }
    }

    /// Parse a stored/incoming `link_type` against the vocabulary.
    ///
    /// Returns `None` for anything outside
    /// [`TYPED_EDGE_VOCABULARY`][TYPED_EDGE_VOCABULARY], including legacy
    /// activation-network types like `semantic` — the typed surface and the
    /// legacy surface are deliberately distinct.
    pub fn parse(link_type: &str) -> Option<EdgeKind> {
        TYPED_EDGE_VOCABULARY
            .iter()
            .position(|candidate| *candidate == link_type)
            .map(|index| EdgeKind::ALL[index])
    }
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which side of a node to list edges for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// Edges where the node is the `source_id`.
    Outgoing,
    /// Edges where the node is the `target_id`.
    Incoming,
    /// Both orientations.
    Both,
}

/// The `edge_meta` JSON payload (V39): commit/anchor evidence for an edge.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeMeta {
    /// Commit (or other artifact) sha the edge was declared against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha: Option<String>,
    /// Source span the edge anchors to (`file.rs:120-140`, symbol, path).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<String>,
    /// The run that observed/declared the evidence, when it differs from
    /// `created_by_run` (e.g. a walk trail re-recorded by a later run).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

impl EdgeMeta {
    /// An empty payload; serializes to `{}`.
    pub fn empty() -> Self {
        Self::default()
    }
}

/// A typed edge as written to / read from `memory_connections`.
///
/// `link_type` is carried as the raw string so the MCP surface can pass
/// caller input straight through; [`SqliteMemoryStore::save_typed_edge`]
/// rejects anything outside the vocabulary at write time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedEdge {
    pub source_id: String,
    pub target_id: String,
    /// Must be a [`TYPED_EDGE_VOCABULARY`] member; validated on save.
    pub link_type: String,
    /// Activation strength; typed edges default to 1.0 (they are declared
    /// facts, not Hebbian co-activation scores).
    pub strength: f64,
    /// V39 `edge_meta` JSON payload: `{sha, span, run_id}`.
    pub meta: EdgeMeta,
    /// V39 run provenance: the agent run that declared this edge.
    pub created_by_run: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_activated: DateTime<Utc>,
    pub activation_count: i32,
}

impl TypedEdge {
    /// A typed edge with default bookkeeping (strength 1.0, timestamps now,
    /// empty payload). Set `meta` / `created_by_run` on the returned value.
    pub fn new(source_id: impl Into<String>, target_id: impl Into<String>, link_type: &str) -> Self {
        let now = Utc::now();
        Self {
            source_id: source_id.into(),
            target_id: target_id.into(),
            link_type: link_type.to_string(),
            strength: 1.0,
            meta: EdgeMeta::default(),
            created_by_run: None,
            created_at: now,
            last_activated: now,
            activation_count: 0,
        }
    }
}

/// A purge tombstone row (`purge_tombstones`, V39).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurgeTombstone {
    pub purged_id: String,
    pub purged_at: DateTime<Utc>,
    pub reason: Option<String>,
    /// SHA-256 hex of the purged node's content, captured before erasure —
    /// content-free but verifiable evidence that *something* occupied the id.
    pub prior_content_hash: Option<String>,
}

/// Build a SQL `IN (...)` list placeholder fragment for the given kinds.
/// The values are vocabulary constants (never user input), so inlining the
/// quoted literals is safe and keeps the statement shape fixed.
fn in_list(kinds: &[EdgeKind]) -> String {
    kinds
        .iter()
        .map(|kind| format!("'{}'", kind.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

impl SqliteMemoryStore {
    /// Persist a typed edge, validating the vocabulary first.
    ///
    /// Unknown `link_type` values — including legacy activation types such as
    /// `semantic` — are rejected with [`StorageError::InvalidEdge`]. The row
    /// lands in the shared `memory_connections` table with the V39 `edge_meta`
    /// payload and `created_by_run` provenance stamped. `INSERT OR REPLACE`
    /// matches [`SqliteMemoryStore::save_connection`]: re-declaring the same
    /// (source, target) pair is idempotent and overwrites in full.
    pub fn save_typed_edge(&self, edge: &TypedEdge) -> Result<()> {
        if EdgeKind::parse(&edge.link_type).is_none() {
            return Err(StorageError::InvalidEdge(format!(
                "unknown link_type '{}'; typed edges must be one of: {}",
                edge.link_type,
                TYPED_EDGE_VOCABULARY.join(", ")
            )));
        }
        let edge_meta_json = serde_json::to_string(&edge.meta)
            .map_err(|error| StorageError::Init(format!("edge_meta serialization failed: {error}")))?;
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        writer.execute(
            "INSERT OR REPLACE INTO memory_connections (
                source_id, target_id, strength, link_type, created_at, last_activated,
                activation_count, edge_meta, created_by_run
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                edge.source_id,
                edge.target_id,
                edge.strength,
                edge.link_type,
                edge.created_at.to_rfc3339(),
                edge.last_activated.to_rfc3339(),
                edge.activation_count,
                edge_meta_json,
                edge.created_by_run,
            ],
        )?;
        Ok(())
    }

    /// List typed edges incident to `node_id`, filtered by direction.
    ///
    /// Only rows whose `link_type` is in the vocabulary are returned — legacy
    /// activation-network connections stay invisible to the typed surface
    /// (use `get_connections_for_memory` for those).
    pub fn edges_for(&self, node_id: &str, dir: EdgeDirection) -> Result<Vec<TypedEdge>> {
        let orientation = match dir {
            EdgeDirection::Outgoing => "source_id = ?1",
            EdgeDirection::Incoming => "target_id = ?1",
            EdgeDirection::Both => "(source_id = ?1 OR target_id = ?1)",
        };
        let sql = format!(
            "SELECT source_id, target_id, strength, link_type, created_at, last_activated,
                    activation_count, edge_meta, created_by_run
             FROM memory_connections
             WHERE {orientation} AND link_type IN ({})
             ORDER BY created_at DESC, source_id, target_id",
            in_list(&EdgeKind::ALL)
        );
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(&sql)?;
        let rows = stmt.query_map(params![node_id], row_to_typed_edge)?;
        let mut edges = Vec::new();
        for row in rows {
            edges.push(row?);
        }
        Ok(edges)
    }

    /// Exact forward reachability from `start_id` over the given edge kinds.
    ///
    /// Iterative breadth-first search over outgoing typed edges: each node is
    /// visited at most once (cycle-safe), each visited node is expanded at
    /// most once (exact, no repeated work), and only nodes within `max_depth`
    /// hops from the start are returned. The start node itself is never part
    /// of the result — reachability answers "where can I get to from here".
    /// An empty `link_types` slice closes every type and returns an empty
    /// vector. Results are sorted lexicographically for determinism.
    ///
    /// Direction is forward (source → target) because typed trails flow that
    /// way: a derivation points at what it was derived from, a closure chain
    /// points at what closed it.
    pub fn edge_reachability(
        &self,
        start_id: &str,
        link_types: &[EdgeKind],
        max_depth: usize,
    ) -> Result<Vec<String>> {
        if link_types.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "SELECT target_id FROM memory_connections
             WHERE source_id = ?1 AND link_type IN ({})",
            in_list(link_types)
        );
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let mut stmt = reader.prepare(&sql)?;

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::from([(start_id.to_string(), 0)]);
        while let Some((node, depth)) = queue.pop_front() {
            if !visited.insert(node.clone()) {
                continue;
            }
            if depth >= max_depth {
                continue;
            }
            let targets = stmt.query_map(params![node], |row| row.get::<_, String>(0))?;
            for row in targets {
                let target = row?;
                if !visited.contains(&target) {
                    queue.push_back((target, depth + 1));
                }
            }
        }
        visited.remove(start_id);
        let mut reached: Vec<String> = visited.into_iter().collect();
        reached.sort();
        Ok(reached)
    }

    /// Compute the derived subgraph rooted at `root_id` for gate review.
    ///
    /// Returns `root_id` plus every node reachable from it via `derived_from`
    /// edges (forward, cycle-safe, unbounded depth), sorted. **Deletes
    /// nothing and writes nothing** — the returned set is the input a review
    /// gate examines before any retire decision is allowed to mutate; how a
    /// decision is persisted belongs to the review flow, not this listing.
    /// Nodes the subgraph merely touches via other edge kinds (anchors,
    /// evidence, closures) are deliberately absent: only derivation lineage
    /// retires with the root.
    pub fn retire_subgraph(&self, root_id: &str) -> Result<Vec<String>> {
        let mut ids: BTreeSet<String> = self
            .edge_reachability(root_id, &[EdgeKind::DerivedFrom], usize::MAX)?
            .into_iter()
            .collect();
        ids.insert(root_id.to_string());
        Ok(ids.into_iter().collect())
    }

    /// Record a purge tombstone for `purged_id`.
    ///
    /// Called by the purge flow (real or simulated) so the store keeps a
    /// content-free, verifiable record: when it happened, why, and the
    /// SHA-256 of the content that occupied the id (computed from
    /// `knowledge_nodes` if the row still exists — a simulated purge — and
    /// NULL once the content is already gone). Re-recording the same id
    /// replaces the row, matching the at-most-one-tombstone-per-id contract
    /// of V13's `deletion_tombstones`.
    pub fn record_tombstone(&self, purged_id: &str, reason: &str) -> Result<()> {
        let writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        // Read the content on the writer connection so the hash and the
        // insert observe the same committed snapshot without a second lock.
        let content: Option<String> = writer
            .query_row(
                "SELECT content FROM knowledge_nodes WHERE id = ?1",
                params![purged_id],
                |row| row.get(0),
            )
            .optional()?;
        let prior_content_hash = content.map(|content| crate::actor::revision_digest(&content));
        writer.execute(
            "INSERT OR REPLACE INTO purge_tombstones
                (purged_id, purged_at, reason, prior_content_hash)
             VALUES (?1, ?2, ?3, ?4)",
            params![purged_id, Utc::now().to_rfc3339(), reason, prior_content_hash],
        )?;
        Ok(())
    }

    /// Fetch the purge tombstone for `purged_id`, if one was recorded.
    pub fn get_purge_tombstone(&self, purged_id: &str) -> Result<Option<PurgeTombstone>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        reader
            .query_row(
                "SELECT purged_id, purged_at, reason, prior_content_hash
                 FROM purge_tombstones WHERE purged_id = ?1",
                params![purged_id],
                |row| {
                    Ok(PurgeTombstone {
                        purged_id: row.get(0)?,
                        purged_at: parse_timestamp(&row.get::<_, String>(1)?),
                        reason: row.get(2)?,
                        prior_content_hash: row.get(3)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }
}

/// Map a `memory_connections` row onto a [`TypedEdge`]. Callers have already
/// filtered `link_type` to the vocabulary. A present-but-unparseable
/// `edge_meta` blob degrades to an empty payload rather than failing the
/// read (the edge itself remains valid evidence).
fn row_to_typed_edge(row: &rusqlite::Row<'_>) -> rusqlite::Result<TypedEdge> {
    let edge_meta_json: Option<String> = row.get(7)?;
    let meta = edge_meta_json
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default();
    Ok(TypedEdge {
        source_id: row.get(0)?,
        target_id: row.get(1)?,
        strength: row.get(2)?,
        link_type: row.get(3)?,
        created_at: parse_timestamp(&row.get::<_, String>(4)?),
        last_activated: parse_timestamp(&row.get::<_, String>(5)?),
        activation_count: row.get(6).unwrap_or(0),
        meta,
        created_by_run: row.get(8)?,
    })
}

/// Parse an RFC 3339 timestamp, falling back to now on malformed input —
/// the same tolerance `row_to_connection` applies to legacy rows.
fn parse_timestamp(raw: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;
    use tempfile::tempdir;

    /// A store on a temp directory, with fresh migrations applied.
    fn test_store() -> SqliteMemoryStore {
        let dir = tempdir().expect("tempdir");
        SqliteMemoryStore::new(Some(dir.path().join("edges-test.db"))).expect("store")
    }

    /// Insert a minimal knowledge node so typed edges satisfy the
    /// `memory_connections` foreign keys.
    fn seed_node(store: &SqliteMemoryStore, id: &str, content: &str) {
        let writer = store.writer.lock().expect("writer");
        let now = Utc::now().to_rfc3339();
        writer
            .execute(
                "INSERT INTO knowledge_nodes
                    (id, content, node_type, created_at, updated_at, last_accessed, tags, scope)
                 VALUES (?1, ?2, 'fact', ?3, ?3, ?3, '[]', 'user')",
                params![id, content, now],
            )
            .expect("seed node");
    }

    fn node_exists(store: &SqliteMemoryStore, id: &str) -> bool {
        let reader = store.reader.lock().expect("reader");
        reader
            .query_row(
                "SELECT 1 FROM knowledge_nodes WHERE id = ?1",
                params![id],
                |_| Ok(()),
            )
            .is_ok()
    }

    #[test]
    fn typed_save_accepts_the_vocabulary_and_rejects_unknown_types() {
        let store = test_store();
        seed_node(&store, "n-src", "source record");
        seed_node(&store, "n-dst", "target record");

        // Every vocabulary member saves and round-trips.
        for (index, kind) in TYPED_EDGE_VOCABULARY.iter().enumerate() {
            let mut edge = TypedEdge::new("n-src", "n-dst", kind);
            // Reuse one (source, target) PK per iteration: give each kind its
            // own pair so earlier rows are not replaced away.
            edge.source_id = format!("n-src-{index}");
            edge.target_id = format!("n-dst-{index}");
            seed_node(&store, &edge.source_id, "src");
            seed_node(&store, &edge.target_id, "dst");
            edge.meta = EdgeMeta {
                sha: Some("abc123".into()),
                span: Some("src/lib.rs:10-20".into()),
                run_id: None,
            };
            edge.created_by_run = Some("run-42".to_string());
            store.save_typed_edge(&edge).expect("vocabulary member saves");

            let outgoing = store
                .edges_for(&edge.source_id, EdgeDirection::Outgoing)
                .expect("outgoing edges");
            assert_eq!(outgoing.len(), 1, "one typed edge for {kind}");
            assert_eq!(outgoing[0].link_type, *kind);
            assert_eq!(outgoing[0].target_id, edge.target_id);
            assert_eq!(outgoing[0].meta.sha.as_deref(), Some("abc123"));
            assert_eq!(outgoing[0].meta.span.as_deref(), Some("src/lib.rs:10-20"));
            assert_eq!(outgoing[0].created_by_run.as_deref(), Some("run-42"));

            let incoming = store
                .edges_for(&edge.target_id, EdgeDirection::Incoming)
                .expect("incoming edges");
            assert_eq!(incoming.len(), 1, "incoming visible for {kind}");
            assert_eq!(incoming[0].source_id, edge.source_id);
        }

        // Unknown types are rejected — including a legacy activation type.
        for unknown in ["wibble", "semantic", "backfill_candidate", ""] {
            let edge = TypedEdge::new("n-src", "n-dst", unknown);
            let error = store
                .save_typed_edge(&edge)
                .expect_err("unknown link_type must be rejected");
            assert!(
                error.to_string().contains("unknown link_type"),
                "error must name the rejection: {error}"
            );
        }
        // And nothing from the rejected batch was written.
        let both = store
            .edges_for("n-src", EdgeDirection::Both)
            .expect("both directions");
        assert!(both.is_empty(), "rejected writes must not persist");
    }

    #[test]
    fn reachability_is_exact_over_a_diamond_and_survives_cycles() {
        let store = test_store();
        for id in ["a", "b", "c", "d", "x"] {
            seed_node(&store, id, &format!("node {id}"));
        }
        // Diamond: a -> b -> d, a -> c -> d, plus a cycle back d -> a.
        for (src, dst) in [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "a")] {
            store
                .save_typed_edge(&TypedEdge::new(src, dst, "derived_from"))
                .expect("diamond edge");
        }
        // A different-kind edge that must stay outside a derived_from walk.
        store
            .save_typed_edge(&TypedEdge::new("a", "x", "touched"))
            .expect("touched edge");

        let derived = [EdgeKind::DerivedFrom];

        // Depth-bounded: exactly the nodes reachable within the budget,
        // each once (the diamond joins at d without duplicating it).
        assert_eq!(
            store
                .edge_reachability("a", &derived, 1)
                .expect("depth 1 reachability"),
            vec!["b".to_string(), "c".to_string()]
        );
        assert_eq!(
            store
                .edge_reachability("a", &derived, 2)
                .expect("depth 2 reachability"),
            vec!["b".to_string(), "c".to_string(), "d".to_string()]
        );

        // Unbounded: the d -> a cycle terminates (a is not re-expanded and
        // never appears in its own result set).
        assert_eq!(
            store
                .edge_reachability("a", &derived, usize::MAX)
                .expect("cycle-safe reachability"),
            vec!["b".to_string(), "c".to_string(), "d".to_string()]
        );

        // Type filter: including `touched` pulls x in; an empty filter
        // closes the walk entirely.
        assert_eq!(
            store
                .edge_reachability("a", &[EdgeKind::DerivedFrom, EdgeKind::Touched], usize::MAX)
                .expect("multi-type reachability"),
            vec!["b".to_string(), "c".to_string(), "d".to_string(), "x".to_string()]
        );
        assert!(store
            .edge_reachability("a", &[], usize::MAX)
            .expect("empty type set")
            .is_empty());
    }

    #[test]
    fn retire_subgraph_lists_derived_lineage_without_deleting() {
        let store = test_store();
        // r derives m1 derives m2; m2 merely touches an unrelated node.
        for id in ["r", "m1", "m2", "bystander"] {
            seed_node(&store, id, &format!("node {id}"));
        }
        for (src, dst, kind) in [
            ("r", "m1", "derived_from"),
            ("m1", "m2", "derived_from"),
            ("m2", "bystander", "touched"),
        ] {
            store
                .save_typed_edge(&TypedEdge::new(src, dst, kind))
                .expect("chain edge");
        }

        let listed = store.retire_subgraph("r").expect("retire listing");
        assert_eq!(
            listed,
            vec!["m1".to_string(), "m2".to_string(), "r".to_string()],
            "root plus derived_from lineage; touched bystanders excluded"
        );

        // The listing deleted nothing: every node (bystander included) and
        // every edge is still on disk.
        for id in ["r", "m1", "m2", "bystander"] {
            assert!(node_exists(&store, id), "{id} must survive the listing");
        }
        let reader = store.reader.lock().expect("reader");
        let edge_count: i64 = reader
            .query_row("SELECT COUNT(*) FROM memory_connections", [], |row| row.get(0))
            .expect("count edges");
        drop(reader);
        assert_eq!(edge_count, 3, "no edge may be removed by retire_subgraph");

        // A derived_from cycle through the root must not hang the listing.
        store
            .save_typed_edge(&TypedEdge::new("m2", "r", "derived_from"))
            .expect("cycle edge");
        assert_eq!(
            store.retire_subgraph("r").expect("cycle-safe listing"),
            vec!["m1".to_string(), "m2".to_string(), "r".to_string()]
        );
    }

    #[test]
    fn tombstone_is_recorded_on_a_simulated_purge() {
        let store = test_store();
        seed_node(&store, "victim", "content that will be purged");
        assert!(
            store.get_purge_tombstone("victim").expect("lookup")
                .is_none(),
            "no tombstone before the purge"
        );

        // Simulated purge: content still present when the tombstone is taken.
        store
            .record_tombstone("victim", "gdpr erasure request")
            .expect("record tombstone");

        let tombstone = store
            .get_purge_tombstone("victim")
            .expect("lookup after record")
            .expect("tombstone row exists");
        assert_eq!(tombstone.purged_id, "victim");
        assert_eq!(tombstone.reason.as_deref(), Some("gdpr erasure request"));
        assert_eq!(
            tombstone.prior_content_hash.as_deref(),
            Some(crate::actor::revision_digest("content that will be purged").as_str()),
            "hash is taken from the still-present content"
        );
        assert!(node_exists(&store, "victim"), "simulated purge deletes nothing");

        // After the real delete, re-recording keeps exactly one row and the
        // original hash is gone (content no longer readable).
        let writer = store.writer.lock().expect("writer");
        writer
            .execute("DELETE FROM knowledge_nodes WHERE id = 'victim'", [])
            .expect("real purge");
        drop(writer);
        store
            .record_tombstone("victim", "post-delete re-record")
            .expect("re-record tombstone");
        let re_recorded = store
            .get_purge_tombstone("victim")
            .expect("lookup after re-record")
            .expect("exactly one tombstone row");
        assert_eq!(re_recorded.reason.as_deref(), Some("post-delete re-record"));
        assert_eq!(
            re_recorded.prior_content_hash, None,
            "no content to hash after the real purge"
        );
    }
}
