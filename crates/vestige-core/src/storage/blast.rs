//! Blast Radius — exact downstream reach of a cause/source record.
//!
//! Given a root memory (the cause/source side of a recorded causal edge),
//! `blast_radius` walks `memory_connections` edges breadth-first and reports
//! every record that descends from it, plus every record sharing its commit
//! sha. This is an EXACT traversal of stored edges: no scoring, no ranking,
//! no causal claim — the edges themselves are hypotheses recorded by
//! `backfill` / lifecycle promotion, and this module only measures how far
//! they reach.
//!
//! Edge direction matches how causes are persisted: `backfill`'s promote
//! block and the lifecycle backfill sweep both save
//! `source_id = cause.memory_id, target_id = failure_node.id`, so the root is
//! the SOURCE side and traversal follows SOURCE -> TARGET only.
//!
//! `retire_affected` is deliberately NOT a delete: it routes each affected
//! record through the existing suppression mechanism (`suppress_memory`),
//! which journals `suppression_operations`, stays Rac1-cascade eligible, and
//! remains reversible inside the 24h labile window. Destructive operations
//! belong behind the Memory-PR review gate at the tool layer (see
//! `vestige-mcp`'s `gate_pending_memory_mutation`, which wraps this same
//! storage-level suppress path).

use chrono::Utc;

use crate::memory::KnowledgeNode;

use super::sqlite::SqliteMemoryStore;
use super::{Result, StorageError};

/// Causal lineage edge types followed by the default blast traversal.
pub const BLAST_LINK_TYPES: [&str; 3] = ["derived_from", "backfill_candidate", "evidence_of"];

/// BFS depth cap. A->B->... chains deeper than this are not reported; the
/// edge chain that deep is already well past hypothesis strength.
pub const BLAST_MAX_DEPTH: u32 = 5;

/// Minimum hex length accepted for a `commit <sha>` line and for sha-prefix
/// root resolution (guards against prose false positives).
const MIN_SHA_CHARS: usize = 6;

// `BlastAffected`, `BlastReport`, and `RetireOutcome` are defined in (and
// re-exported from) `crate::storage::types`.
pub use crate::storage::types::{BlastAffected, BlastReport, RetireOutcome};

/// Extract the sha from a record's `commit <sha> ...` line, if any.
///
/// Commit records are persisted with a first line of
/// `commit <sha> <subject>` (see `advanced::git_records::record_content`).
/// The hex + minimum-length guard keeps prose mentions of the word "commit"
/// from matching.
pub fn commit_sha_of(content: &str) -> Option<String> {
    for line in content.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix("commit ") else {
            continue;
        };
        let Some(token) = rest.split_whitespace().next() else {
            continue;
        };
        if token.len() >= MIN_SHA_CHARS && token.chars().all(|c| c.is_ascii_hexdigit()) {
            return Some(token.to_ascii_lowercase());
        }
    }
    None
}

/// valid_until IS NULL or > now.
fn is_open(node: &KnowledgeNode, now: chrono::DateTime<Utc>) -> bool {
    node.valid_until.is_none_or(|until| until > now)
}

impl SqliteMemoryStore {
    /// Exact blast radius over the default causal lineage edge set
    /// ([`BLAST_LINK_TYPES`]). See [`Self::blast_radius_with_link_types`].
    pub fn blast_radius(&self, root_id: &str, open_only: bool) -> Result<BlastReport> {
        self.blast_radius_with_link_types(root_id, open_only, &BLAST_LINK_TYPES)
    }

    /// Breadth-first, cycle-safe, depth-capped traversal over
    /// `memory_connections` edges with `link_type` in `link_types`,
    /// following SOURCE -> TARGET direction only (the root is the cause/
    /// source side, matching how `backfill` persists candidate edges).
    ///
    /// The report includes:
    /// - the root itself at depth 0 (via "root") — the anchor of the query,
    ///   reported even when `open_only` filters its descendants;
    /// - every record sharing the root's commit sha (parsed from a
    ///   `commit <sha>` line) as depth-0 siblings (via "shared_commit:<sha>");
    /// - every SOURCE -> TARGET descendant, via the edge's link_type.
    ///
    /// When `open_only` is true, non-root entries with
    /// `valid_until <= now` are filtered out. Traversal is exact: no
    /// scoring, no eligibility heuristics beyond `open_only`.
    pub fn blast_radius_with_link_types(
        &self,
        root_id: &str,
        open_only: bool,
        link_types: &[&str],
    ) -> Result<BlastReport> {
        let root = self
            .get_node(root_id)?
            .ok_or_else(|| StorageError::NotFound(root_id.to_string()))?;
        let now = Utc::now();

        let mut affected = vec![BlastAffected {
            id: root.id.clone(),
            via: "root".to_string(),
            depth: 0,
        }];
        let mut visited = std::collections::HashSet::from([root.id.clone()]);

        // Depth-0 siblings: records carrying the same `commit <sha>` line.
        // The root's sha is only known after loading it, so this is a
        // bounded paged scan (same cost class as `get_all_connections`).
        if let Some(sha) = commit_sha_of(&root.content) {
            for sibling in self.page_all_nodes()? {
                if sibling.id == root.id
                    || visited.contains(&sibling.id)
                    || commit_sha_of(&sibling.content).as_deref() != Some(sha.as_str())
                {
                    continue;
                }
                if open_only && !is_open(&sibling, now) {
                    continue;
                }
                visited.insert(sibling.id.clone());
                affected.push(BlastAffected {
                    id: sibling.id,
                    via: format!("shared_commit:{sha}"),
                    depth: 0,
                });
            }
        }

        // BFS over SOURCE -> TARGET edges. `get_connections_for_memory`
        // returns edges at both endpoints; only source == current edges are
        // followed so direction is preserved.
        let mut queue = std::collections::VecDeque::from([(root.id.clone(), 0u32)]);
        while let Some((current, depth)) = queue.pop_front() {
            if depth >= BLAST_MAX_DEPTH {
                continue;
            }
            for edge in self.get_connections_for_memory(&current)? {
                if edge.source_id != current
                    || !link_types.contains(&edge.link_type.as_str())
                    || edge.target_id == current
                {
                    continue;
                }
                let target = edge.target_id.clone();
                if !visited.insert(target.clone()) {
                    continue; // cycle / diamond: first (shallowest) visit wins
                }
                let Some(node) = self.get_node(&target)? else {
                    continue; // dangling edge: report nothing, do not crash
                };
                if open_only && !is_open(&node, now) {
                    continue;
                }
                affected.push(BlastAffected {
                    id: target.clone(),
                    via: edge.link_type.clone(),
                    depth: depth + 1,
                });
                queue.push_back((target, depth + 1));
            }
        }

        // Root stays first (it is already at depth 0); order the rest by
        // (depth, id) so repeated runs on an unchanged store are identical.
        affected[1..].sort_by(|a, b| (a.depth, &a.id).cmp(&(b.depth, &b.id)));
        let total = affected.len();
        Ok(BlastReport {
            root_id: root.id,
            affected,
            total,
        })
    }

    /// Resolve a commit-sha prefix to the newest record whose `commit <sha>`
    /// line starts with it. Returns `Ok(None)` when nothing matches.
    /// Prefixes shorter than [`MIN_SHA_CHARS`] are rejected to avoid
    /// ambiguous matches.
    pub fn resolve_commit_sha_root(&self, sha_prefix: &str) -> Result<Option<String>> {
        let prefix = sha_prefix.trim().to_ascii_lowercase();
        if prefix.len() < MIN_SHA_CHARS || !prefix.chars().all(|c| c.is_ascii_hexdigit()) {
            return Ok(None);
        }
        // get_all_nodes orders by created_at DESC, so the first match is the
        // newest record carrying that sha.
        for node in self.page_all_nodes()? {
            if let Some(sha) = commit_sha_of(&node.content)
                && sha.starts_with(&prefix)
            {
                return Ok(Some(node.id));
            }
        }
        Ok(None)
    }

    /// Retire every affected record WITHOUT deleting anything: each id is
    /// flipped through the existing suppression mechanism
    /// ([`Self::suppress_memory`]), so the row survives, the operation is
    /// journaled in `suppression_operations`, the Rac1 cascade stays
    /// eligible, and the 24h labile reversal window applies. Returns one
    /// outcome per input id, in input order.
    ///
    /// Tool layers must route each retire through the Memory-PR review gate
    /// before calling this (the same storage path `gate_pending_memory_mutation`
    /// wraps for `suppress`/`purge`).
    pub fn retire_affected(&self, ids: &[&str], reason: &str) -> Vec<RetireOutcome> {
        ids.iter()
            .map(|id| match self.suppress_memory(id) {
                Ok(node) => {
                    tracing::info!(
                        id = %id,
                        count = node.suppression_count,
                        reason = %reason,
                        "blast retire: suppression applied (no deletion)"
                    );
                    RetireOutcome {
                        id: (*id).to_string(),
                        suppressed: true,
                        suppression_count: node.suppression_count,
                        error: None,
                    }
                }
                Err(error) => {
                    tracing::warn!(id = %id, reason = %reason, %error, "blast retire failed");
                    RetireOutcome {
                        id: (*id).to_string(),
                        suppressed: false,
                        suppression_count: 0,
                        error: Some(error.to_string()),
                    }
                }
            })
            .collect()
    }

    /// Page through every knowledge node (all scopes, deterministic
    /// created_at DESC order). Bounded by [`BLAST_SCAN_NODE_CAP`] so a
    /// pathological store cannot turn a diagnostic into an unbounded read.
    fn page_all_nodes(&self) -> Result<Vec<KnowledgeNode>> {
        const PAGE: i32 = 1000;
        let mut all = Vec::new();
        let mut offset = 0;
        loop {
            let page = self.get_all_nodes(PAGE, offset)?;
            let got = page.len();
            all.extend(page);
            if got < PAGE as usize || all.len() >= BLAST_SCAN_NODE_CAP {
                break;
            }
            offset += PAGE;
        }
        Ok(all)
    }
}

/// Upper bound on nodes scanned for shared-sha siblings / sha-prefix root
/// resolution. 20k covers operational stores; larger stores should shard
/// scopes.
pub const BLAST_SCAN_NODE_CAP: usize = 20_000;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IngestInput;
    use chrono::Duration;

    fn store() -> SqliteMemoryStore {
        let dir = tempfile::TempDir::new().unwrap();
        SqliteMemoryStore::new(Some(dir.path().join("blast.db"))).unwrap()
    }

    fn ingest(s: &SqliteMemoryStore, content: &str) -> String {
        s.ingest(IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap()
        .id
    }

    fn ingest_with_valid_until(
        s: &SqliteMemoryStore,
        content: &str,
        valid_until: Option<chrono::DateTime<Utc>>,
    ) -> String {
        s.ingest(IngestInput {
            content: content.to_string(),
            node_type: "fact".to_string(),
            valid_until,
            ..Default::default()
        })
        .unwrap()
        .id
    }

    fn edge(s: &SqliteMemoryStore, source: &str, target: &str, link_type: &str) {
        s.save_connection(&crate::storage::ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 0.8,
            link_type: link_type.to_string(),
            created_at: Utc::now(),
            last_activated: Utc::now(),
            activation_count: 0,
        })
        .unwrap();
    }

    #[test]
    fn blast_follows_source_to_target_direction() {
        let s = store();
        let cause = ingest(&s, "deploy used stale config cache");
        let failure = ingest(&s, "prod outage: config not reloaded");
        // cause is the SOURCE side, matching backfill's promote block
        edge(&s, &cause, &failure, "backfill_candidate");
        // an edge pointing INTO the cause must not pull the upstream in
        let upstream = ingest(&s, "stale config cache introduced in 1.2.0");
        edge(&s, &upstream, &cause, "backfill_candidate");

        let report = s.blast_radius(&cause, false).unwrap();
        assert_eq!(report.root_id, cause);
        assert_eq!(report.total, 2);
        assert_eq!(report.affected[0].id, cause);
        assert_eq!(report.affected[0].via, "root");
        assert_eq!(report.affected[0].depth, 0);
        assert_eq!(report.affected[1].id, failure);
        assert_eq!(report.affected[1].via, "backfill_candidate");
        assert_eq!(report.affected[1].depth, 1);
    }

    #[test]
    fn blast_reports_shared_sha_siblings_at_depth_zero() {
        let s = store();
        let sha = "a1b2c3d4e5f6";
        let root = ingest(&s, &format!("commit {sha} fix: reload config"));
        let sibling = ingest(&s, &format!("commit {sha} (cherry-pick) hotfix"));
        let other = ingest(&s, "commit 000000000000 unrelated commit record");

        let report = s.blast_radius(&root, false).unwrap();
        let ids: Vec<&str> = report.affected.iter().map(|a| a.id.as_str()).collect();
        assert!(ids.contains(&sibling.as_str()));
        assert!(!ids.contains(&other.as_str()));
        let sib = report
            .affected
            .iter()
            .find(|a| a.id == sibling)
            .unwrap();
        assert_eq!(sib.depth, 0);
        assert_eq!(sib.via, format!("shared_commit:{sha}"));
    }

    #[test]
    fn blast_diamond_reports_converging_node_once() {
        let s = store();
        let root = ingest(&s, "cause");
        let a = ingest(&s, "A");
        let b = ingest(&s, "B");
        let c = ingest(&s, "C");
        edge(&s, &root, &a, "derived_from");
        edge(&s, &root, &b, "derived_from");
        edge(&s, &a, &c, "derived_from");
        edge(&s, &b, &c, "derived_from");

        let report = s.blast_radius(&root, false).unwrap();
        assert_eq!(report.total, 4, "C must appear once, not twice");
        assert_eq!(
            report
                .affected
                .iter()
                .filter(|x| x.id == c)
                .count(),
            1
        );
        let c_entry = report.affected.iter().find(|x| x.id == c).unwrap();
        assert_eq!(c_entry.depth, 2);
    }

    #[test]
    fn blast_cycle_terminates() {
        let s = store();
        let a = ingest(&s, "A");
        let b = ingest(&s, "B");
        let c = ingest(&s, "C");
        edge(&s, &a, &b, "evidence_of");
        edge(&s, &b, &c, "evidence_of");
        edge(&s, &c, &a, "evidence_of"); // cycle back to root
        let self_loop = ingest(&s, "S");
        edge(&s, &self_loop, &self_loop, "evidence_of");

        let report = s.blast_radius(&a, false).unwrap();
        assert_eq!(report.total, 3);
        let solo = s.blast_radius(&self_loop, false).unwrap();
        assert_eq!(solo.total, 1, "self-loop must not duplicate the root");
    }

    #[test]
    fn blast_depth_cap_five() {
        let s = store();
        let root = ingest(&s, "n0");
        let mut prev = root.clone();
        for i in 1..=7 {
            let next = ingest(&s, &format!("n{i}"));
            edge(&s, &prev, &next, "derived_from");
            prev = next;
        }
        let report = s.blast_radius(&root, false).unwrap();
        // root + n1..n5 = 6; n6, n7 are past the cap
        assert_eq!(report.total, 6);
        assert!(report.affected.iter().all(|a| a.depth <= BLAST_MAX_DEPTH));
    }

    #[test]
    fn open_only_filters_expired_valid_until() {
        let s = store();
        let root = ingest(&s, "cause");
        let open = ingest(&s, "still open failure");
        let expired = ingest_with_valid_until(
            &s,
            "historical failure",
            Some(Utc::now() - Duration::try_hours(1).unwrap()),
        );
        edge(&s, &root, &open, "backfill_candidate");
        edge(&s, &root, &expired, "backfill_candidate");

        let open_report = s.blast_radius(&root, true).unwrap();
        let ids: Vec<&str> = open_report.affected.iter().map(|a| a.id.as_str()).collect();
        assert!(ids.contains(&open.as_str()));
        assert!(!ids.contains(&expired.as_str()));
        assert_eq!(open_report.total, 2, "root + open failure");

        let all_report = s.blast_radius(&root, false).unwrap();
        assert_eq!(all_report.total, 3);
    }

    #[test]
    fn commit_sha_parser_shape() {
        assert_eq!(
            commit_sha_of("commit a1b2c3d4e5f6 fix: thing"),
            Some("a1b2c3d4e5f6".to_string())
        );
        assert_eq!(commit_sha_of("we commit changes daily"), None);
        assert_eq!(commit_sha_of("commit short nope"), None);
        assert_eq!(commit_sha_of("first line\ncommit a1b2c3d4e5f6 on line two"), Some("a1b2c3d4e5f6".to_string()));
    }

    #[test]
    fn resolve_commit_sha_prefix_finds_newest() {
        let s = store();
        let old = ingest(&s, "commit feed0000abcd first landed");
        let newest = ingest(&s, "commit feed0000ef01 relanded as hotfix");
        let got = s.resolve_commit_sha_root("feed0000").unwrap();
        assert_eq!(got.as_deref(), Some(newest.as_str()));
        assert_ne!(got.as_deref(), Some(old.as_str()));
        assert_eq!(s.resolve_commit_sha_root("zz9").unwrap(), None);
    }

    #[test]
    fn blast_missing_root_is_not_found() {
        let s = store();
        let err = s.blast_radius("00000000-0000-0000-0000-000000000000", false);
        assert!(matches!(err, Err(StorageError::NotFound(_))));
    }

    #[test]
    fn retire_affected_suppresses_without_deleting() {
        let s = store();
        let cause = ingest(&s, "cause");
        let failure = ingest(&s, "failure");
        edge(&s, &cause, &failure, "backfill_candidate");
        let report = s.blast_radius(&cause, false).unwrap();
        let ids: Vec<&str> = report.affected.iter().map(|a| a.id.as_str()).collect();

        let outcomes = s.retire_affected(&ids, "blast retire test");
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.iter().all(|o| o.suppressed));

        for id in ids {
            let node = s.get_node(id).unwrap().unwrap();
            assert_eq!(node.suppression_count, 1, "suppressed, not deleted");
            assert!(node.suppressed_at.is_some());
        }
    }

    #[test]
    fn retire_affected_reports_per_id_errors() {
        let s = store();
        let live = ingest(&s, "live");
        let missing = "ffffffff-0000-0000-0000-000000000000";
        let outcomes = s.retire_affected(&[live.as_str(), missing], "mixed");
        assert_eq!(outcomes.len(), 2);
        assert!(outcomes[0].suppressed);
        assert_eq!(outcomes[0].suppression_count, 1);
        assert!(!outcomes[1].suppressed);
        assert!(outcomes[1].error.as_deref().unwrap().contains(missing));
    }
}
