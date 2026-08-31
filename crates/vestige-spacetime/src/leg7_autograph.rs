use std::collections::{BTreeSet, HashMap};

use serde::Serialize;

use crate::common::{
    ContextHierarchy, EngramProjection, MemoryMap, RealityPolicy, ScopeKind, hash_json,
};
use crate::leg3_reality_index::{IndexedRecallEngine, RealityIndex};

#[derive(Clone, Debug)]
pub struct EdgePassport {
    pub scope: ScopeKind,
    pub effective_context_id: Option<String>,
    pub valid_from_ms: Option<i64>,
    pub valid_until_ms: Option<i64>,
    pub origin: String,
    pub authority: u8,
    pub taints: BTreeSet<String>,
    pub allowed_purposes: Option<BTreeSet<String>>,
    pub allowed_actors: Option<BTreeSet<String>>,
}

impl EdgePassport {
    fn valid_at(&self, at_ms: i64) -> bool {
        self.valid_from_ms.is_none_or(|value| at_ms >= value)
            && self.valid_until_ms.is_none_or(|value| at_ms < value)
    }
}

#[derive(Clone, Debug)]
pub struct PassportedEdge {
    pub edge_id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    pub passport: EdgePassport,
}

#[derive(Clone, Debug)]
pub struct CrossRealityHypothesis {
    pub hypothesis_id: String,
    pub parent_ids: Vec<String>,
    /// Exact parent revisions at proposal time. Review must fail closed when a
    /// parent has changed, rather than approving a stale synthesis proposal.
    pub parent_revisions: Vec<(String, String)>,
    pub source_context_ids: Vec<String>,
    pub proposed_context_id: String,
    pub proposed_content: String,
    pub relation: String,
    pub active_traversal: bool,
    pub requires_review: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexTaskStatus {
    Pending,
    Indexed,
}

#[derive(Clone, Debug)]
pub struct IndexTask {
    pub memory_id: String,
    pub context_id: String,
    pub revision: String,
    pub status: IndexTaskStatus,
}

#[derive(Clone, Debug)]
pub struct DerivationOutcome {
    pub memory: Option<EngramProjection>,
    pub edges: Vec<PassportedEdge>,
    pub hypothesis: Option<CrossRealityHypothesis>,
    pub index_task: Option<IndexTask>,
}

/// All explicit inputs to a new derived memory transaction.
pub struct DerivationRequest<'a> {
    pub memory_id: &'a str,
    pub content: &'a str,
    pub parent_ids: &'a [String],
    pub relation: &'a str,
    pub fault_after_memory: bool,
}

#[derive(Clone, Debug)]
pub struct HybridEvidenceBundle {
    pub semantic_ids: Vec<String>,
    pub adjacent_ids: Vec<String>,
    pub graph_edges: Vec<String>,
    pub eligible_universe_seal: String,
    pub reality_checkpoint: String,
}

#[derive(Clone, Debug, Default)]
pub struct PassportedGraph {
    pub edges: HashMap<String, PassportedEdge>,
    adjacency: HashMap<String, BTreeSet<String>>,
}

impl PassportedGraph {
    pub fn add(&mut self, edge: PassportedEdge) {
        if self.edges.contains_key(&edge.edge_id) {
            self.remove(&edge.edge_id);
        }
        let edge_id = edge.edge_id.clone();
        self.adjacency
            .entry(edge.source_id.clone())
            .or_default()
            .insert(edge_id.clone());
        self.adjacency
            .entry(edge.target_id.clone())
            .or_default()
            .insert(edge_id.clone());
        self.edges.insert(edge_id, edge);
    }

    pub fn remove(&mut self, edge_id: &str) -> Option<PassportedEdge> {
        let edge = self.edges.remove(edge_id)?;
        for node_id in [&edge.source_id, &edge.target_id] {
            let remove_bucket = if let Some(edge_ids) = self.adjacency.get_mut(node_id) {
                edge_ids.remove(edge_id);
                edge_ids.is_empty()
            } else {
                false
            };
            if remove_bucket {
                self.adjacency.remove(node_id);
            }
        }
        Some(edge)
    }

    pub fn edges_for(&self, memory_id: &str) -> Vec<&PassportedEdge> {
        self.adjacency
            .get(memory_id)
            .into_iter()
            .flat_map(|edge_ids| edge_ids.iter())
            .filter_map(|edge_id| self.edges.get(edge_id))
            .collect()
    }
}

pub struct AutoGraphEngine {
    pub hierarchy: ContextHierarchy,
    pub graph: PassportedGraph,
    pub hypotheses: HashMap<String, CrossRealityHypothesis>,
    pub index_tasks: HashMap<String, IndexTask>,
}

impl AutoGraphEngine {
    pub fn new(hierarchy: ContextHierarchy) -> Self {
        Self {
            hierarchy,
            graph: PassportedGraph::default(),
            hypotheses: HashMap::new(),
            index_tasks: HashMap::new(),
        }
    }

    fn context(memory: &EngramProjection) -> Option<String> {
        match memory.scope {
            ScopeKind::Context => memory
                .effective_context_id
                .clone()
                .or(memory.birth_context_id.clone()),
            ScopeKind::Global | ScopeKind::Unlocated => None,
        }
    }

    fn target_context(&self, parents: &[EngramProjection]) -> Option<String> {
        let contexts: Vec<String> = parents.iter().filter_map(Self::context).collect();
        if contexts.is_empty() {
            None
        } else {
            self.hierarchy.narrowest_containing(&contexts)
        }
    }

    fn common_ancestor(&self, parents: &[EngramProjection]) -> Option<String> {
        let contexts: Vec<String> = parents.iter().filter_map(Self::context).collect();
        self.hierarchy.lca(&contexts)
    }

    fn edge_id(source: &str, target: &str, relation: &str, context: Option<&str>) -> String {
        hash_json("edge_", &(source, target, relation, context))
    }

    fn intersect_optional(
        sets: impl Iterator<Item = Option<BTreeSet<String>>>,
    ) -> Option<BTreeSet<String>> {
        let mut restricted: Vec<BTreeSet<String>> = sets.flatten().collect();
        if restricted.is_empty() {
            return None;
        }
        let mut result = restricted.remove(0);
        for set in restricted {
            result = result.intersection(&set).cloned().collect();
        }
        Some(result)
    }

    fn derived_memory(
        memory_id: &str,
        content: &str,
        parents: &[EngramProjection],
        target_context: Option<&str>,
    ) -> Result<EngramProjection, String> {
        let authority = parents
            .iter()
            .map(|parent| parent.authority)
            .min()
            .ok_or("parents required")?;
        let taints = parents
            .iter()
            .flat_map(|parent| parent.taints.iter().cloned())
            .collect();
        let valid_from_ms = parents
            .iter()
            .filter_map(|parent| parent.valid_from_ms)
            .max();
        let valid_until_ms = parents
            .iter()
            .filter_map(|parent| parent.valid_until_ms)
            .min();
        if valid_from_ms
            .zip(valid_until_ms)
            .is_some_and(|(start, end)| end <= start)
        {
            return Err("parent worldlines do not overlap".into());
        }

        Ok(EngramProjection {
            memory_id: memory_id.into(),
            content: content.into(),
            tags: Vec::new(),
            node_type: "fact".into(),
            scope: if target_context.is_some() {
                ScopeKind::Context
            } else {
                ScopeKind::Global
            },
            birth_context_id: target_context.map(String::from),
            effective_context_id: target_context.map(String::from),
            origin: "derived".into(),
            authority,
            valid_from_ms,
            valid_until_ms,
            taints,
            allowed_purposes: Self::intersect_optional(
                parents.iter().map(|parent| parent.allowed_purposes.clone()),
            ),
            allowed_actors: Self::intersect_optional(
                parents.iter().map(|parent| parent.allowed_actors.clone()),
            ),
            lineage: parents
                .iter()
                .map(|parent| parent.memory_id.clone())
                .collect(),
            claim_key: None,
        })
    }

    fn edge(
        parent: &EngramProjection,
        child: &EngramProjection,
        relation: &str,
        target_context: Option<&str>,
    ) -> PassportedEdge {
        PassportedEdge {
            edge_id: Self::edge_id(
                &parent.memory_id,
                &child.memory_id,
                relation,
                target_context,
            ),
            source_id: parent.memory_id.clone(),
            target_id: child.memory_id.clone(),
            relation: relation.into(),
            passport: EdgePassport {
                scope: if target_context.is_some() {
                    ScopeKind::Context
                } else {
                    ScopeKind::Global
                },
                effective_context_id: target_context.map(String::from),
                valid_from_ms: child.valid_from_ms,
                valid_until_ms: child.valid_until_ms,
                origin: child.origin.clone(),
                authority: child.authority,
                taints: child.taints.clone(),
                allowed_purposes: child.allowed_purposes.clone(),
                allowed_actors: child.allowed_actors.clone(),
            },
        }
    }

    fn load_parents(
        memories: &MemoryMap,
        parent_ids: &[String],
    ) -> Result<Vec<EngramProjection>, String> {
        parent_ids
            .iter()
            .map(|memory_id| {
                memories
                    .get(memory_id)
                    .cloned()
                    .ok_or_else(|| format!("missing parent {memory_id}"))
            })
            .collect()
    }

    pub fn derive(
        &mut self,
        index: &mut RealityIndex,
        memories: &mut MemoryMap,
        request: DerivationRequest<'_>,
    ) -> Result<DerivationOutcome, String> {
        let DerivationRequest {
            memory_id,
            content,
            parent_ids,
            relation,
            fault_after_memory,
        } = request;
        let parents = Self::load_parents(memories, parent_ids)?;
        let target_context = self.target_context(&parents);

        if target_context.is_none()
            && parents
                .iter()
                .any(|parent| parent.scope == ScopeKind::Context)
        {
            let proposed_context_id = self
                .common_ancestor(&parents)
                .ok_or("cross-Reality derivation has no common ancestor")?;
            #[derive(Serialize)]
            struct HypothesisProjection<'a> {
                parents: &'a [String],
                target: &'a str,
                content_hash: String,
                relation: &'a str,
            }
            let source_context_ids: BTreeSet<String> =
                parents.iter().filter_map(Self::context).collect();
            let parent_revisions = parents
                .iter()
                .map(|parent| (parent.memory_id.clone(), parent.revision()))
                .collect();
            let hypothesis = CrossRealityHypothesis {
                hypothesis_id: hash_json(
                    "hyp_",
                    &HypothesisProjection {
                        parents: parent_ids,
                        target: &proposed_context_id,
                        content_hash: blake3::hash(content.as_bytes()).to_hex().to_string(),
                        relation,
                    },
                ),
                parent_ids: parent_ids.to_vec(),
                parent_revisions,
                source_context_ids: source_context_ids.into_iter().collect(),
                proposed_context_id,
                proposed_content: content.into(),
                relation: relation.into(),
                active_traversal: false,
                requires_review: true,
            };
            self.hypotheses
                .insert(hypothesis.hypothesis_id.clone(), hypothesis.clone());
            return Ok(DerivationOutcome {
                memory: None,
                edges: Vec::new(),
                hypothesis: Some(hypothesis),
                index_task: None,
            });
        }

        self.commit_derivation(
            index,
            memories,
            memory_id,
            content,
            &parents,
            parent_ids,
            relation,
            target_context.as_deref(),
            None,
            fault_after_memory,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_derivation(
        &mut self,
        index: &mut RealityIndex,
        memories: &mut MemoryMap,
        memory_id: &str,
        content: &str,
        parents: &[EngramProjection],
        _parent_ids: &[String],
        relation: &str,
        target_context: Option<&str>,
        hypothesis: Option<CrossRealityHypothesis>,
        fault_after_memory: bool,
    ) -> Result<DerivationOutcome, String> {
        if memories.contains_key(memory_id) {
            return Err("derived memory ID already exists".into());
        }
        let memories_before = memories.clone();
        let graph_before = self.graph.clone();
        let tasks_before = self.index_tasks.clone();

        let result = (|| {
            let child = Self::derived_memory(memory_id, content, parents, target_context)?;
            memories.insert(memory_id.into(), child.clone());
            if fault_after_memory {
                return Err("fault injection after Engram write".into());
            }

            let edges: Vec<PassportedEdge> = parents
                .iter()
                .map(|parent| Self::edge(parent, &child, relation, target_context))
                .collect();
            for edge in &edges {
                self.graph.add(edge.clone());
            }
            let index_task = IndexTask {
                memory_id: memory_id.into(),
                context_id: target_context.unwrap_or("__global__").into(),
                revision: child.revision(),
                status: IndexTaskStatus::Pending,
            };
            self.index_tasks
                .insert(memory_id.into(), index_task.clone());
            index.rebuild(memories);
            Ok(DerivationOutcome {
                memory: Some(child),
                edges,
                hypothesis,
                index_task: Some(index_task),
            })
        })();

        if result.is_err() {
            *memories = memories_before;
            self.graph = graph_before;
            self.index_tasks = tasks_before;
            index.rebuild(memories);
        }
        result
    }

    pub fn approve(
        &mut self,
        index: &mut RealityIndex,
        memories: &mut MemoryMap,
        hypothesis_id: &str,
        confirm: bool,
        memory_id: &str,
    ) -> Result<DerivationOutcome, String> {
        if !confirm {
            return Err("cross-Reality synthesis requires review".into());
        }
        let hypothesis = self
            .hypotheses
            .get(hypothesis_id)
            .cloned()
            .ok_or("hypothesis missing")?;
        let parent_ids = hypothesis.parent_ids.clone();
        let proposed_content = hypothesis.proposed_content.clone();
        let relation = hypothesis.relation.clone();
        let proposed_context_id = hypothesis.proposed_context_id.clone();
        let parents = Self::load_parents(memories, &parent_ids)?;
        let current_revisions: Vec<(String, String)> = parents
            .iter()
            .map(|parent| (parent.memory_id.clone(), parent.revision()))
            .collect();
        if current_revisions != hypothesis.parent_revisions {
            return Err("hypothesis parent revision changed".into());
        }
        let current_contexts: Vec<String> = parents
            .iter()
            .filter_map(Self::context)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if current_contexts != hypothesis.source_context_ids
            || self.target_context(&parents).is_some()
            || self.common_ancestor(&parents).as_deref() != Some(proposed_context_id.as_str())
        {
            return Err("hypothesis Context topology changed".into());
        }
        let outcome = self.commit_derivation(
            index,
            memories,
            memory_id,
            &proposed_content,
            &parents,
            &parent_ids,
            &relation,
            Some(&proposed_context_id),
            Some(hypothesis),
            false,
        )?;
        self.hypotheses.remove(hypothesis_id);
        Ok(outcome)
    }

    pub fn complete_index(
        &mut self,
        memories: &MemoryMap,
        memory_id: &str,
        indexed_context: &str,
    ) -> Result<(), String> {
        let task = self.index_tasks.get_mut(memory_id).ok_or("task missing")?;
        if task.context_id != indexed_context {
            return Err("index Context mismatch".into());
        }
        if task.revision != memories.get(memory_id).ok_or("memory missing")?.revision() {
            return Err("memory revision changed".into());
        }
        task.status = IndexTaskStatus::Indexed;
        Ok(())
    }

    fn edge_allowed(
        edge: &PassportedEdge,
        policy: &RealityPolicy,
        eligible_memory_ids: &BTreeSet<String>,
    ) -> bool {
        if !eligible_memory_ids.contains(&edge.source_id)
            || !eligible_memory_ids.contains(&edge.target_id)
        {
            return false;
        }
        match edge.passport.scope {
            ScopeKind::Global => {
                if !policy.include_global {
                    return false;
                }
            }
            ScopeKind::Context => {
                let Some(context) = &edge.passport.effective_context_id else {
                    return false;
                };
                if !policy.cross_context && !policy.context_cone.contains(context) {
                    return false;
                }
            }
            ScopeKind::Unlocated => return false,
        }
        if !edge.passport.valid_at(policy.valid_at_ms)
            || !policy.allowed_origins.contains(&edge.passport.origin)
            || edge.passport.authority < policy.minimum_authority
            || edge
                .passport
                .taints
                .iter()
                .any(|taint| policy.denied_taints.contains(taint))
            || edge
                .passport
                .allowed_purposes
                .as_ref()
                .is_some_and(|purposes| !purposes.contains(&policy.purpose))
            || edge
                .passport
                .allowed_actors
                .as_ref()
                .is_some_and(|actors| !actors.contains(&policy.actor))
        {
            return false;
        }
        true
    }

    pub fn neighbors(
        &self,
        index: &RealityIndex,
        memories: &MemoryMap,
        seed: &str,
        policy: RealityPolicy,
    ) -> Result<Vec<(String, String)>, String> {
        let compiled = index.compile(memories, policy.clone());
        if !compiled.memory_ids.contains(seed) {
            return Err("seed outside active Reality".into());
        }
        let mut result = Vec::new();
        let mut edges = self.graph.edges_for(seed);
        edges.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
        for edge in edges {
            if seed != edge.source_id && seed != edge.target_id {
                continue;
            }
            if !Self::edge_allowed(edge, &policy, &compiled.memory_ids) {
                continue;
            }
            let other = if seed == edge.source_id {
                edge.target_id.clone()
            } else {
                edge.source_id.clone()
            };
            result.push((other, edge.edge_id.clone()));
        }
        Ok(result)
    }

    pub fn hybrid_bundle(
        &self,
        indexed: &IndexedRecallEngine,
        memories: &MemoryMap,
        query: &str,
        policy: RealityPolicy,
        limit: usize,
    ) -> Result<HybridEvidenceBundle, String> {
        let recalled = indexed.recall(query, policy.clone(), memories, limit, None);
        let semantic_ids: Vec<String> = recalled
            .hits
            .iter()
            .map(|hit| hit.memory_id.clone())
            .collect();
        let mut adjacent_ids = Vec::new();
        let mut graph_edges = Vec::new();
        for memory_id in &semantic_ids {
            for (neighbor, edge_id) in
                self.neighbors(&indexed.index, memories, memory_id, policy.clone())?
            {
                if !semantic_ids.contains(&neighbor) && !adjacent_ids.contains(&neighbor) {
                    adjacent_ids.push(neighbor);
                }
                if !graph_edges.contains(&edge_id) {
                    graph_edges.push(edge_id);
                }
            }
        }
        adjacent_ids.truncate(limit);
        Ok(HybridEvidenceBundle {
            semantic_ids,
            adjacent_ids,
            graph_edges,
            eligible_universe_seal: recalled.eligible_universe_seal,
            reality_checkpoint: recalled.reality_checkpoint,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{EngramProjection, MemoryMap, RealityPolicy};

    fn hierarchy() -> ContextHierarchy {
        let mut hierarchy = ContextHierarchy::default();
        hierarchy.add("eng", None);
        hierarchy.add("A", Some("eng"));
        hierarchy.add("B", Some("eng"));
        hierarchy
    }

    fn policy(context: &str) -> RealityPolicy {
        RealityPolicy::new(
            Some(context.into()),
            [context.into(), "eng".into()].into_iter().collect(),
            100,
        )
    }

    #[test]
    fn siblings_create_hypothesis_not_bridge() {
        let mut memories = MemoryMap::new();
        memories.insert("a".into(), EngramProjection::context("a", "A rule", "A"));
        memories.insert("b".into(), EngramProjection::context("b", "B rule", "B"));
        let mut index = RealityIndex::new(10);
        index.rebuild(&memories);
        let mut graph = AutoGraphEngine::new(hierarchy());
        let outcome = graph
            .derive(
                &mut index,
                &mut memories,
                DerivationRequest {
                    memory_id: "x",
                    content: "shared",
                    parent_ids: &["a".into(), "b".into()],
                    relation: "derived_from",
                    fault_after_memory: false,
                },
            )
            .expect("create hypothesis");
        assert!(outcome.memory.is_none());
        assert!(outcome.hypothesis.is_some());
        assert!(graph.graph.edges.is_empty());
    }

    #[test]
    fn approved_synthesis_does_not_bridge_siblings() {
        let mut memories = MemoryMap::new();
        memories.insert("a".into(), EngramProjection::context("a", "A rule", "A"));
        memories.insert("b".into(), EngramProjection::context("b", "B rule", "B"));
        let mut index = RealityIndex::new(10);
        index.rebuild(&memories);
        let mut graph = AutoGraphEngine::new(hierarchy());
        let hypothesis = graph
            .derive(
                &mut index,
                &mut memories,
                DerivationRequest {
                    memory_id: "unused",
                    content: "shared",
                    parent_ids: &["a".into(), "b".into()],
                    relation: "derived_from",
                    fault_after_memory: false,
                },
            )
            .expect("create hypothesis")
            .hypothesis
            .expect("hypothesis");
        graph
            .approve(
                &mut index,
                &mut memories,
                &hypothesis.hypothesis_id,
                true,
                "shared",
            )
            .expect("approve hypothesis");

        let from_a = graph
            .neighbors(&index, &memories, "a", policy("A"))
            .expect("A neighbors");
        assert!(from_a.iter().any(|(neighbor, _)| neighbor == "shared"));
        let from_shared = graph
            .neighbors(&index, &memories, "shared", policy("A"))
            .expect("shared neighbors");
        assert!(!from_shared.iter().any(|(neighbor, _)| neighbor == "b"));
    }

    #[test]
    fn fault_rolls_back() {
        let mut memories = MemoryMap::new();
        memories.insert("a".into(), EngramProjection::context("a", "A1", "A"));
        memories.insert("a2".into(), EngramProjection::context("a2", "A2", "A"));
        let mut index = RealityIndex::new(10);
        index.rebuild(&memories);
        let mut graph = AutoGraphEngine::new(hierarchy());
        assert!(
            graph
                .derive(
                    &mut index,
                    &mut memories,
                    DerivationRequest {
                        memory_id: "d",
                        content: "derived",
                        parent_ids: &["a".into(), "a2".into()],
                        relation: "derived_from",
                        fault_after_memory: true,
                    },
                )
                .is_err()
        );
        assert!(!memories.contains_key("d"));
        assert!(graph.graph.edges.is_empty());
    }

    #[test]
    fn derivation_refuses_to_overwrite_an_existing_memory() {
        let mut memories = MemoryMap::new();
        memories.insert("a".into(), EngramProjection::context("a", "A1", "A"));
        memories.insert("a2".into(), EngramProjection::context("a2", "A2", "A"));
        memories.insert(
            "d".into(),
            EngramProjection::context("d", "must remain intact", "A"),
        );
        let mut index = RealityIndex::new(10);
        index.rebuild(&memories);
        let mut graph = AutoGraphEngine::new(hierarchy());
        let error = graph
            .derive(
                &mut index,
                &mut memories,
                DerivationRequest {
                    memory_id: "d",
                    content: "replacement attempt",
                    parent_ids: &["a".into(), "a2".into()],
                    relation: "derived_from",
                    fault_after_memory: false,
                },
            )
            .expect_err("existing memory must never be overwritten");
        assert_eq!(error, "derived memory ID already exists");
        assert_eq!(memories["d"].content, "must remain intact");
        assert!(graph.graph.edges.is_empty());
    }

    #[test]
    fn hypothesis_is_revision_bound_and_single_use() {
        let mut memories = MemoryMap::new();
        memories.insert("a".into(), EngramProjection::context("a", "A rule", "A"));
        memories.insert("b".into(), EngramProjection::context("b", "B rule", "B"));
        let mut index = RealityIndex::new(10);
        index.rebuild(&memories);
        let mut graph = AutoGraphEngine::new(hierarchy());
        let hypothesis = graph
            .derive(
                &mut index,
                &mut memories,
                DerivationRequest {
                    memory_id: "unused",
                    content: "shared",
                    parent_ids: &["a".into(), "b".into()],
                    relation: "derived_from",
                    fault_after_memory: false,
                },
            )
            .expect("create hypothesis")
            .hypothesis
            .expect("hypothesis");

        memories.get_mut("a").expect("parent").content = "changed".into();
        index.rebuild(&memories);
        assert_eq!(
            graph
                .approve(
                    &mut index,
                    &mut memories,
                    &hypothesis.hypothesis_id,
                    true,
                    "shared",
                )
                .expect_err("stale proposal must fail"),
            "hypothesis parent revision changed"
        );
        assert!(!memories.contains_key("shared"));

        let fresh = graph
            .derive(
                &mut index,
                &mut memories,
                DerivationRequest {
                    memory_id: "unused",
                    content: "shared after review",
                    parent_ids: &["a".into(), "b".into()],
                    relation: "derived_from",
                    fault_after_memory: false,
                },
            )
            .expect("create fresh hypothesis")
            .hypothesis
            .expect("fresh hypothesis");
        graph
            .approve(
                &mut index,
                &mut memories,
                &fresh.hypothesis_id,
                true,
                "shared",
            )
            .expect("approve fresh hypothesis");
        assert!(!graph.hypotheses.contains_key(&fresh.hypothesis_id));
        assert_eq!(
            graph
                .approve(
                    &mut index,
                    &mut memories,
                    &fresh.hypothesis_id,
                    true,
                    "second-shared",
                )
                .expect_err("approved hypothesis is single use"),
            "hypothesis missing"
        );
    }
    #[test]
    fn hybrid_bundle_shares_reality_checkpoint_and_blocks_manual_bridge() {
        let mut memories = MemoryMap::new();
        memories.insert(
            "a".into(),
            EngramProjection::context("a", "A migration", "A"),
        );
        memories.insert(
            "b".into(),
            EngramProjection::context("b", "B migration", "B"),
        );
        let mut indexed = IndexedRecallEngine::new(10, 100);
        indexed.rebuild(&memories);
        let mut graph = AutoGraphEngine::new(hierarchy());
        graph.graph.add(PassportedEdge {
            edge_id: "wrong-bridge".into(),
            source_id: "a".into(),
            target_id: "b".into(),
            relation: "semantic".into(),
            passport: EdgePassport {
                scope: ScopeKind::Context,
                effective_context_id: Some("A".into()),
                valid_from_ms: None,
                valid_until_ms: None,
                origin: "derived".into(),
                authority: 3,
                taints: BTreeSet::new(),
                allowed_purposes: None,
                allowed_actors: None,
            },
        });
        let p = policy("A");
        let direct = indexed.recall("migration", p.clone(), &memories, 5, None);
        let bundle = graph
            .hybrid_bundle(&indexed, &memories, "migration", p, 5)
            .expect("hybrid bundle");
        assert_eq!(bundle.reality_checkpoint, direct.reality_checkpoint);
        assert!(!bundle.adjacent_ids.iter().any(|id| id == "b"));
        assert!(!bundle.graph_edges.iter().any(|id| id == "wrong-bridge"));
    }
}
