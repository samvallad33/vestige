use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::common::{EngramProjection, MemoryMap, RealityPolicy, content_hash, hash_json};
use crate::leg3_reality_index::IndexedRecallResult;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct WorkingSetEntry {
    pub memory_id: String,
    pub revision: String,
    pub content_hash: String,
    pub content: String,
    pub score: f64,
    pub source: String,
}

impl WorkingSetEntry {
    fn leaf_projection(&self) -> (&str, &str, &str) {
        (&self.memory_id, &self.revision, &self.content_hash)
    }
}

pub fn merkle_root(entries: &[WorkingSetEntry]) -> String {
    if entries.is_empty() {
        return hash_json("merkle_", &Vec::<String>::new());
    }

    let mut sorted = entries.to_vec();
    sorted.sort_by(|left, right| left.memory_id.cmp(&right.memory_id));
    let mut level: Vec<String> = sorted
        .iter()
        .map(|entry| hash_json("leaf_", &entry.leaf_projection()))
        .collect();

    while level.len() > 1 {
        if level.len() % 2 == 1 {
            let last = level.last().expect("non-empty Merkle level").clone();
            level.push(last);
        }
        level = level
            .chunks(2)
            .map(|pair| hash_json("node_", &(pair[0].as_str(), pair[1].as_str())))
            .collect();
    }

    format!(
        "merkle_{}",
        level[0]
            .split_once('_')
            .map(|(_, value)| value)
            .unwrap_or(&level[0])
    )
}

#[derive(Clone, Debug)]
pub struct RealitySnapshot {
    pub snapshot_id: String,
    pub reality_fingerprint: String,
    pub reality_checkpoint: String,
    pub root: String,
    pub rank_seal: String,
    pub entries: Vec<WorkingSetEntry>,
}

impl RealitySnapshot {
    pub fn by_id(&self) -> BTreeMap<String, WorkingSetEntry> {
        self.entries
            .iter()
            .cloned()
            .map(|entry| (entry.memory_id.clone(), entry))
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RankUpdate {
    pub memory_id: String,
    pub score: f64,
    pub source: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealityDelta {
    pub from_snapshot: Option<String>,
    pub to_snapshot: String,
    pub reality_fingerprint: String,
    pub reality_checkpoint: String,
    pub previous_root: Option<String>,
    pub new_root: String,
    pub additions: Vec<WorkingSetEntry>,
    pub updates: Vec<WorkingSetEntry>,
    pub removals: Vec<String>,
    pub rank_updates: Vec<RankUpdate>,
    pub seal: String,
}

impl RealityDelta {
    fn seal_value(&self) -> serde_json::Value {
        serde_json::json!({
            "from": self.from_snapshot,
            "to": self.to_snapshot,
            "reality": self.reality_fingerprint,
            "realityCheckpoint": self.reality_checkpoint,
            "previousRoot": self.previous_root,
            "newRoot": self.new_root,
            "additions": self.additions,
            "updates": self.updates,
            "removals": self.removals,
            "rankUpdates": self.rank_updates,
        })
    }

    pub fn verify(&self) -> bool {
        self.seal == hash_json("delta_", &self.seal_value())
    }
}

pub struct RealityWorkingSet {
    pub max_entries: usize,
    snapshots: BTreeMap<String, RealitySnapshot>,
}

impl RealityWorkingSet {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            snapshots: BTreeMap::new(),
        }
    }

    pub fn build_from_recall(
        &mut self,
        recall: &IndexedRecallResult,
        policy: &RealityPolicy,
        memories: &MemoryMap,
        prefetch_ids: &[String],
        eligible_ids: &BTreeSet<String>,
    ) -> RealitySnapshot {
        let mut selected = BTreeMap::new();
        for hit in &recall.hits {
            if selected.len() >= self.max_entries {
                break;
            }
            if let Some(memory) = memories.get(&hit.memory_id) {
                selected.insert(hit.memory_id.clone(), entry(memory, hit.score, "recall"));
            }
        }

        // Predictions are suggestions, never authority. Every prefetched ID is
        // independently checked against the compiled eligible universe.
        for memory_id in prefetch_ids {
            if selected.len() >= self.max_entries {
                break;
            }
            if selected.contains_key(memory_id) || !eligible_ids.contains(memory_id) {
                continue;
            }
            if let Some(memory) = memories.get(memory_id) {
                selected.insert(memory_id.clone(), entry(memory, 0.0, "prediction"));
            }
        }

        let entries: Vec<WorkingSetEntry> = selected.into_values().collect();
        let root = merkle_root(&entries);
        let rank_projection: Vec<(&str, f64, &str)> = entries
            .iter()
            .map(|entry| (entry.memory_id.as_str(), entry.score, entry.source.as_str()))
            .collect();
        let rank_seal = hash_json("rank_", &rank_projection);
        let snapshot_id = hash_json(
            "snapshot_",
            &(
                policy.fingerprint.as_str(),
                recall.reality_checkpoint.as_str(),
                root.as_str(),
                rank_seal.as_str(),
            ),
        );
        let snapshot = RealitySnapshot {
            snapshot_id: snapshot_id.clone(),
            reality_fingerprint: policy.fingerprint.clone(),
            reality_checkpoint: recall.reality_checkpoint.clone(),
            root,
            rank_seal,
            entries,
        };
        self.snapshots.insert(snapshot_id, snapshot.clone());
        snapshot
    }

    pub fn delta(
        &self,
        previous_id: Option<&str>,
        current: &RealitySnapshot,
    ) -> Result<RealityDelta, String> {
        let previous = previous_id.and_then(|id| self.snapshots.get(id));
        if previous.is_some_and(|old| old.reality_fingerprint != current.reality_fingerprint) {
            return Err("Reality fingerprint changed; full snapshot required".into());
        }

        let old = previous.map(RealitySnapshot::by_id).unwrap_or_default();
        let new = current.by_id();
        let old_ids: BTreeSet<String> = old.keys().cloned().collect();
        let new_ids: BTreeSet<String> = new.keys().cloned().collect();

        let additions = new_ids
            .difference(&old_ids)
            .filter_map(|id| new.get(id).cloned())
            .collect();
        let removals = old_ids.difference(&new_ids).cloned().collect();
        let mut updates = Vec::new();
        let mut rank_updates = Vec::new();

        for id in old_ids.intersection(&new_ids) {
            let old_entry = &old[id];
            let new_entry = &new[id];
            if old_entry.revision != new_entry.revision {
                updates.push(new_entry.clone());
            } else if old_entry.score != new_entry.score || old_entry.source != new_entry.source {
                rank_updates.push(RankUpdate {
                    memory_id: id.clone(),
                    score: new_entry.score,
                    source: new_entry.source.clone(),
                });
            }
        }

        let mut delta = RealityDelta {
            from_snapshot: previous.map(|snapshot| snapshot.snapshot_id.clone()),
            to_snapshot: current.snapshot_id.clone(),
            reality_fingerprint: current.reality_fingerprint.clone(),
            reality_checkpoint: current.reality_checkpoint.clone(),
            previous_root: previous.map(|snapshot| snapshot.root.clone()),
            new_root: current.root.clone(),
            additions,
            updates,
            removals,
            rank_updates,
            seal: String::new(),
        };
        delta.seal = hash_json("delta_", &delta.seal_value());
        Ok(delta)
    }
}

fn entry(memory: &EngramProjection, score: f64, source: &str) -> WorkingSetEntry {
    WorkingSetEntry {
        memory_id: memory.memory_id.clone(),
        revision: memory.revision(),
        content_hash: content_hash(&memory.content),
        content: memory.content.clone(),
        score,
        source: source.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{EngramProjection, RealityPolicy};
    use crate::leg3_reality_index::IndexedRecallEngine;

    #[test]
    fn unchanged_content_is_not_resent_when_rank_changes() {
        let mut memories = MemoryMap::new();
        memories.insert(
            "a".into(),
            EngramProjection::context("a", "backup s3 deploy", "A"),
        );
        let policy = RealityPolicy::new(Some("A".into()), ["A".into()].into_iter().collect(), 1);
        let mut engine = IndexedRecallEngine::new(10, 100);
        engine.rebuild(&memories);
        let eligible = engine.index.compile(&memories, policy.clone());
        let first_recall = engine.recall("backup", policy.clone(), &memories, 5, None);
        let second_recall = engine.recall("deploy", policy.clone(), &memories, 5, None);
        let mut working_set = RealityWorkingSet::new(8);
        let first = working_set.build_from_recall(
            &first_recall,
            &policy,
            &memories,
            &[],
            &eligible.memory_ids,
        );
        let second = working_set.build_from_recall(
            &second_recall,
            &policy,
            &memories,
            &[],
            &eligible.memory_ids,
        );
        let delta = working_set
            .delta(Some(&first.snapshot_id), &second)
            .expect("same-Reality delta");
        assert!(delta.updates.is_empty());
        assert!(delta.verify());
    }

    #[test]
    fn content_mutation_is_one_update() {
        let mut memories = MemoryMap::new();
        memories.insert("a".into(), EngramProjection::context("a", "backup s3", "A"));
        let policy = RealityPolicy::new(Some("A".into()), ["A".into()].into_iter().collect(), 1);
        let mut engine = IndexedRecallEngine::new(10, 100);
        engine.rebuild(&memories);
        let first_eligible = engine.index.compile(&memories, policy.clone());
        let first_recall = engine.recall("backup", policy.clone(), &memories, 5, None);
        let mut working_set = RealityWorkingSet::new(8);
        let first = working_set.build_from_recall(
            &first_recall,
            &policy,
            &memories,
            &[],
            &first_eligible.memory_ids,
        );

        memories.get_mut("a").expect("test memory").content = "backup encrypted s3".into();
        engine.rebuild(&memories);
        let second_eligible = engine.index.compile(&memories, policy.clone());
        let second_recall = engine.recall("backup", policy.clone(), &memories, 5, None);
        let second = working_set.build_from_recall(
            &second_recall,
            &policy,
            &memories,
            &[],
            &second_eligible.memory_ids,
        );
        let delta = working_set
            .delta(Some(&first.snapshot_id), &second)
            .expect("same-Reality delta");

        assert_eq!(delta.updates.len(), 1);
        assert_ne!(first.root, second.root);
        assert!(delta.verify());
    }
    #[test]
    fn snapshot_and_delta_preserve_reality_checkpoint() {
        let mut memories = MemoryMap::new();
        memories.insert("a".into(), EngramProjection::context("a", "backup s3", "A"));
        let policy = RealityPolicy::new(Some("A".into()), ["A".into()].into_iter().collect(), 1);
        let mut engine = IndexedRecallEngine::new(10, 100);
        engine.rebuild(&memories);
        let eligible = engine.index.compile(&memories, policy.clone());
        let recall = engine.recall("backup", policy.clone(), &memories, 5, None);
        let mut working_set = RealityWorkingSet::new(8);
        let snapshot =
            working_set.build_from_recall(&recall, &policy, &memories, &[], &eligible.memory_ids);
        let delta = working_set.delta(None, &snapshot).expect("initial delta");
        assert_eq!(snapshot.reality_checkpoint, recall.reality_checkpoint);
        assert_eq!(delta.reality_checkpoint, recall.reality_checkpoint);
        assert!(delta.verify());
    }
}
