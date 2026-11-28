//! Code anchor registry: a derived index rebuilt from the anchor ops
//! (`RecordAnchors`, `ReplaceAnchors`, `RecordAnchorVerdict`) on replay.
//!
//! The index keeps every admitted row, including rows of a node that was
//! later retired. Readers filter on node liveness (see
//! [`crate::StrataStore::anchors_for`]) so a retired memory's anchors are never
//! returned, while replay stays a pure fold of the log.

use std::collections::BTreeMap;

use crate::types::AnchorRecord;

/// node id -> anchor id -> anchor, plus anchor id -> node id.
#[derive(Debug, Clone, Default)]
pub(crate) struct AnchorIndex {
    by_node: BTreeMap<String, BTreeMap<String, AnchorRecord>>,
    node_of: BTreeMap<String, String>,
}

impl AnchorIndex {
    /// Insert or replace rows by anchor id. A row whose id already exists
    /// under another node moves to the node it now names.
    pub(crate) fn record(&mut self, anchors: &[AnchorRecord]) {
        for anchor in anchors {
            if let Some(previous) = self.node_of.get(&anchor.id).cloned()
                && previous != anchor.node_id
            {
                self.remove_row(&previous, &anchor.id);
            }
            self.node_of
                .insert(anchor.id.clone(), anchor.node_id.clone());
            self.by_node
                .entry(anchor.node_id.clone())
                .or_default()
                .insert(anchor.id.clone(), anchor.clone());
        }
    }

    /// Drop every row of `node_id`, then insert `anchors`.
    pub(crate) fn replace(&mut self, node_id: &str, anchors: &[AnchorRecord]) {
        if let Some(rows) = self.by_node.remove(node_id) {
            for id in rows.keys() {
                self.node_of.remove(id);
            }
        }
        self.record(anchors);
    }

    /// Cache a verdict on an existing row. An unknown id changes nothing.
    pub(crate) fn verdict(&mut self, anchor_id: &str, status: &str, checked_at_ms: i64) {
        let Some(node_id) = self.node_of.get(anchor_id) else {
            return;
        };
        if let Some(row) = self
            .by_node
            .get_mut(node_id)
            .and_then(|rows| rows.get_mut(anchor_id))
        {
            row.last_status = Some(status.to_string());
            row.last_verified_at_ms = Some(checked_at_ms);
        }
    }

    /// Rows of one node, in anchor-id order. Liveness is the caller's check.
    pub(crate) fn rows_of(&self, node_id: &str) -> Vec<AnchorRecord> {
        self.by_node
            .get(node_id)
            .map(|rows| rows.values().cloned().collect())
            .unwrap_or_default()
    }

    /// One row by anchor id.
    pub(crate) fn get(&self, anchor_id: &str) -> Option<&AnchorRecord> {
        let node_id = self.node_of.get(anchor_id)?;
        self.by_node.get(node_id)?.get(anchor_id)
    }

    /// Every row in (node id, anchor id) order. Used by the state digest.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &AnchorRecord> {
        self.by_node.values().flat_map(|rows| rows.values())
    }

    fn remove_row(&mut self, node_id: &str, anchor_id: &str) {
        let now_empty = match self.by_node.get_mut(node_id) {
            Some(rows) => {
                rows.remove(anchor_id);
                rows.is_empty()
            }
            None => false,
        };
        if now_empty {
            self.by_node.remove(node_id);
        }
    }
}
