use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

pub fn hash_json<T: Serialize>(prefix: &str, value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("canonicalizable spec value");
    format!("{}{}", prefix, blake3::hash(&bytes).to_hex())
}

pub fn content_hash(content: &str) -> String {
    blake3::hash(content.as_bytes()).to_hex().to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScopeKind {
    Context,
    Global,
    Unlocated,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngramProjection {
    pub memory_id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub node_type: String,
    pub scope: ScopeKind,
    pub birth_context_id: Option<String>,
    pub effective_context_id: Option<String>,
    pub origin: String,
    pub authority: u8,
    pub valid_from_ms: Option<i64>,
    pub valid_until_ms: Option<i64>,
    pub taints: BTreeSet<String>,
    pub allowed_purposes: Option<BTreeSet<String>>,
    pub allowed_actors: Option<BTreeSet<String>>,
    pub lineage: Vec<String>,
    pub claim_key: Option<String>,
}

impl EngramProjection {
    pub fn revision(&self) -> String {
        hash_json("rev_", self)
    }

    pub fn context(memory_id: &str, content: &str, context_id: &str) -> Self {
        Self {
            memory_id: memory_id.into(),
            content: content.into(),
            tags: Vec::new(),
            node_type: "fact".into(),
            scope: ScopeKind::Context,
            birth_context_id: Some(context_id.into()),
            effective_context_id: Some(context_id.into()),
            origin: "user".into(),
            authority: 3,
            valid_from_ms: None,
            valid_until_ms: None,
            taints: BTreeSet::new(),
            allowed_purposes: None,
            allowed_actors: None,
            lineage: Vec::new(),
            claim_key: None,
        }
    }

    pub fn global(memory_id: &str, content: &str) -> Self {
        let mut memory = Self::context(memory_id, content, "");
        memory.scope = ScopeKind::Global;
        memory.birth_context_id = None;
        memory.effective_context_id = None;
        memory
    }

    pub fn unlocated(memory_id: &str, content: &str) -> Self {
        let mut memory = Self::global(memory_id, content);
        memory.scope = ScopeKind::Unlocated;
        memory
    }

    pub fn valid_at(&self, at_ms: i64) -> bool {
        if self.valid_from_ms.is_some_and(|value| at_ms < value) {
            return false;
        }
        if self.valid_until_ms.is_some_and(|value| at_ms >= value) {
            return false;
        }
        true
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RealityPolicy {
    pub fingerprint: String,
    pub active_context_id: Option<String>,
    pub context_cone: BTreeSet<String>,
    pub include_global: bool,
    pub include_unlocated: bool,
    pub cross_context: bool,
    pub allowed_origins: BTreeSet<String>,
    pub minimum_authority: u8,
    pub denied_taints: BTreeSet<String>,
    pub purpose: String,
    pub actor: String,
    pub valid_at_ms: i64,
}

impl RealityPolicy {
    pub fn new(
        active_context_id: Option<String>,
        context_cone: BTreeSet<String>,
        valid_at_ms: i64,
    ) -> Self {
        let mut policy = Self {
            fingerprint: String::new(),
            active_context_id,
            context_cone,
            include_global: true,
            include_unlocated: false,
            cross_context: false,
            allowed_origins: [
                "user",
                "agent",
                "repo",
                "connector",
                "tool",
                "derived",
                "imported",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            minimum_authority: 1,
            denied_taints: ["poisoned", "quarantined"]
                .into_iter()
                .map(String::from)
                .collect(),
            purpose: "general".into(),
            actor: "agent".into(),
            valid_at_ms,
        };
        policy.refresh_fingerprint();
        policy
    }

    pub fn refresh_fingerprint(&mut self) {
        #[derive(Serialize)]
        struct Projection<'a> {
            active: &'a Option<String>,
            cone: &'a BTreeSet<String>,
            global: bool,
            unlocated: bool,
            cross: bool,
            origins: &'a BTreeSet<String>,
            authority: u8,
            taints: &'a BTreeSet<String>,
            purpose: &'a str,
            actor: &'a str,
            at: i64,
        }

        self.fingerprint = hash_json(
            "reality_",
            &Projection {
                active: &self.active_context_id,
                cone: &self.context_cone,
                global: self.include_global,
                unlocated: self.include_unlocated,
                cross: self.cross_context,
                origins: &self.allowed_origins,
                authority: self.minimum_authority,
                taints: &self.denied_taints,
                purpose: &self.purpose,
                actor: &self.actor,
                at: self.valid_at_ms,
            },
        );
    }
}

#[derive(Clone, Debug, Default)]
pub struct ContextHierarchy {
    parent: HashMap<String, Option<String>>,
}

impl ContextHierarchy {
    pub fn add(&mut self, id: &str, parent: Option<&str>) {
        self.parent.insert(id.into(), parent.map(String::from));
    }

    pub fn ancestors(&self, id: &str) -> Result<BTreeSet<String>, String> {
        let mut result = BTreeSet::new();
        let mut current = Some(id.to_string());
        while let Some(context_id) = current {
            if !result.insert(context_id.clone()) {
                return Err("context parent cycle".into());
            }
            current = self.parent.get(&context_id).cloned().flatten();
        }
        Ok(result)
    }

    pub fn depth(&self, id: &str) -> Option<usize> {
        self.ancestors(id).ok().map(|set| set.len())
    }

    pub fn lca(&self, ids: &[String]) -> Option<String> {
        let first = ids.first()?;
        let first_ancestors = self.ancestors(first).ok()?;
        let all: Vec<BTreeSet<String>> = ids
            .iter()
            .map(|id| self.ancestors(id))
            .collect::<Result<_, _>>()
            .ok()?;

        first_ancestors
            .into_iter()
            .filter(|candidate| all.iter().all(|set| set.contains(candidate)))
            .max_by_key(|candidate| self.depth(candidate).unwrap_or_default())
    }

    pub fn narrowest_containing(&self, ids: &[String]) -> Option<String> {
        let mut best: Option<(usize, String)> = None;
        for candidate in ids {
            let Ok(ancestors) = self.ancestors(candidate) else {
                continue;
            };
            if !ids.iter().all(|id| ancestors.contains(id)) {
                continue;
            }
            let depth = ancestors.len();
            if best.as_ref().is_none_or(|(current, _)| depth > *current) {
                best = Some((depth, candidate.clone()));
            }
        }
        best.map(|(_, context_id)| context_id)
    }
}

pub type MemoryMap = BTreeMap<String, EngramProjection>;
