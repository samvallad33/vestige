use std::collections::{BTreeMap, BTreeSet, HashMap};

use roaring::RoaringBitmap;
use serde::Serialize;

use crate::common::{EngramProjection, MemoryMap, RealityPolicy, ScopeKind, hash_json};

pub const GLOBAL_PAGE: &str = "__global__";
pub const UNLOCATED_PAGE: &str = "__unlocated__";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EligibilityReason {
    Eligible,
    WrongContext,
    GlobalDisabled,
    Unlocated,
    Worldline,
    OriginDenied,
    AuthorityTooLow,
    Tainted,
    PurposeDenied,
    ActorDenied,
    ContextMissing,
}

impl EligibilityReason {
    pub const fn key(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::WrongContext => "wrong_context",
            Self::GlobalDisabled => "global_disabled",
            Self::Unlocated => "unlocated",
            Self::Worldline => "outside_worldline",
            Self::OriginDenied => "origin_denied",
            Self::AuthorityTooLow => "authority_too_low",
            Self::Tainted => "lineage_tainted",
            Self::PurposeDenied => "purpose_denied",
            Self::ActorDenied => "actor_denied",
            Self::ContextMissing => "context_missing",
        }
    }
}

#[derive(Clone, Debug)]
pub struct EligibilityDecision {
    pub memory_id: String,
    pub eligible: bool,
    pub reason: EligibilityReason,
}

#[derive(Clone, Debug)]
pub struct RealityPage {
    pub page_id: String,
    pub scope_key: String,
    pub ordinals: Vec<u32>,
    pub manifest_seal: String,
}

#[derive(Clone, Debug, Default)]
pub struct OrdinalDirectory {
    next: u32,
    id_to_ord: HashMap<String, u32>,
    ord_to_id: HashMap<u32, String>,
}

impl OrdinalDirectory {
    pub fn ensure(&mut self, memory_id: &str) -> u32 {
        if let Some(value) = self.id_to_ord.get(memory_id) {
            return *value;
        }
        let value = self.next;
        self.next = self
            .next
            .checked_add(1)
            .expect("u32 ordinal space exhausted");
        self.id_to_ord.insert(memory_id.into(), value);
        self.ord_to_id.insert(value, memory_id.into());
        value
    }

    pub fn memory_id(&self, ordinal: u32) -> Option<&str> {
        self.ord_to_id.get(&ordinal).map(String::as_str)
    }
}

#[derive(Clone, Debug)]
pub struct CompiledEligibleSet {
    pub policy: RealityPolicy,
    pub ordinals: RoaringBitmap,
    pub memory_ids: BTreeSet<String>,
    pub decisions: Vec<EligibilityDecision>,
    pub blocked_counts: BTreeMap<String, u64>,
    pub fine_candidate_count: u64,
    pub global_store_count: usize,
    pub page_ids: Vec<String>,
    pub page_manifest_seals: Vec<String>,
    pub routing_plan_seal: String,
    pub universe_seal: String,
    pub reality_checkpoint: String,
}

#[derive(Clone, Debug)]
pub struct RealityIndex {
    pub page_size: usize,
    pub ordinals: OrdinalDirectory,
    pub context_members: HashMap<String, RoaringBitmap>,
    pub all_context_members_bitmap: RoaringBitmap,
    pub global_members: RoaringBitmap,
    pub unlocated_members: RoaringBitmap,
    pub context_missing_members: RoaringBitmap,
    pub origin_members: HashMap<String, RoaringBitmap>,
    pub authority_members: HashMap<u8, RoaringBitmap>,
    pub pages: BTreeMap<String, RealityPage>,
    pub pages_by_scope: HashMap<String, Vec<String>>,
    pub generation: u64,
}

impl RealityIndex {
    pub fn new(page_size: usize) -> Self {
        Self {
            page_size: page_size.max(1),
            ordinals: OrdinalDirectory::default(),
            context_members: HashMap::new(),
            all_context_members_bitmap: RoaringBitmap::new(),
            global_members: RoaringBitmap::new(),
            unlocated_members: RoaringBitmap::new(),
            context_missing_members: RoaringBitmap::new(),
            origin_members: HashMap::new(),
            authority_members: HashMap::new(),
            pages: BTreeMap::new(),
            pages_by_scope: HashMap::new(),
            generation: 0,
        }
    }

    pub fn rebuild(&mut self, memories: &MemoryMap) {
        self.context_members.clear();
        self.all_context_members_bitmap.clear();
        self.global_members.clear();
        self.unlocated_members.clear();
        self.context_missing_members.clear();
        self.origin_members.clear();
        self.authority_members.clear();

        for memory in memories.values() {
            let ordinal = self.ordinals.ensure(&memory.memory_id);
            match memory.scope {
                ScopeKind::Global => {
                    self.global_members.insert(ordinal);
                }
                ScopeKind::Unlocated => {
                    self.unlocated_members.insert(ordinal);
                }
                ScopeKind::Context => {
                    if let Some(context) = memory
                        .effective_context_id
                        .as_ref()
                        .or(memory.birth_context_id.as_ref())
                    {
                        self.context_members
                            .entry(context.clone())
                            .or_default()
                            .insert(ordinal);
                        self.all_context_members_bitmap.insert(ordinal);
                    } else {
                        self.context_missing_members.insert(ordinal);
                    }
                }
            }
            self.origin_members
                .entry(memory.origin.clone())
                .or_default()
                .insert(ordinal);
            self.authority_members
                .entry(memory.authority)
                .or_default()
                .insert(ordinal);
        }

        self.rebuild_pages();
        self.generation = self.generation.saturating_add(1);
    }

    fn rebuild_pages(&mut self) {
        self.pages.clear();
        self.pages_by_scope.clear();
        let mut scopes: Vec<(String, RoaringBitmap)> = self
            .context_members
            .iter()
            .map(|(scope, bitmap)| (scope.clone(), bitmap.clone()))
            .collect();
        scopes.push((GLOBAL_PAGE.into(), self.global_members.clone()));
        scopes.push((UNLOCATED_PAGE.into(), self.unlocated_members.clone()));
        scopes.sort_by(|left, right| left.0.cmp(&right.0));

        for (scope, bitmap) in scopes {
            let ordinals: Vec<u32> = bitmap.iter().collect();
            for (page_number, chunk) in ordinals.chunks(self.page_size).enumerate() {
                let page_id = format!("rp:{scope}:{page_number}");
                #[derive(Serialize)]
                struct Manifest<'a> {
                    page: &'a str,
                    scope: &'a str,
                    ordinals: &'a [u32],
                }
                let manifest_seal = hash_json(
                    "page_",
                    &Manifest {
                        page: &page_id,
                        scope: &scope,
                        ordinals: chunk,
                    },
                );
                self.pages_by_scope
                    .entry(scope.clone())
                    .or_default()
                    .push(page_id.clone());
                self.pages.insert(
                    page_id.clone(),
                    RealityPage {
                        page_id,
                        scope_key: scope.clone(),
                        ordinals: chunk.to_vec(),
                        manifest_seal,
                    },
                );
            }
        }
    }

    pub fn route_pages(&self, policy: &RealityPolicy) -> Vec<String> {
        let mut scopes = policy.context_cone.clone();
        if policy.cross_context {
            scopes.extend(self.context_members.keys().cloned());
        }
        if policy.include_global {
            scopes.insert(GLOBAL_PAGE.into());
        }
        if policy.include_unlocated {
            scopes.insert(UNLOCATED_PAGE.into());
        }
        let mut routed = Vec::new();
        for scope in scopes {
            if let Some(page_ids) = self.pages_by_scope.get(&scope) {
                routed.extend(page_ids.iter().cloned());
            }
        }
        routed
    }

    fn all_context_members(&self) -> RoaringBitmap {
        self.all_context_members_bitmap.clone()
    }

    fn allowed_context_members(&self, policy: &RealityPolicy) -> RoaringBitmap {
        if policy.cross_context {
            return self.all_context_members();
        }
        let mut result = RoaringBitmap::new();
        for context in &policy.context_cone {
            if let Some(bitmap) = self.context_members.get(context) {
                result |= bitmap.clone();
            }
        }
        result
    }

    fn coarse_candidates(&self, policy: &RealityPolicy) -> RoaringBitmap {
        let mut result = self.allowed_context_members(policy);
        if policy.include_global {
            result |= self.global_members.clone();
        }
        if policy.include_unlocated {
            result |= self.unlocated_members.clone();
        }
        result
    }

    fn fine_decide(memory: &EngramProjection, policy: &RealityPolicy) -> EligibilityDecision {
        let deny = |reason| EligibilityDecision {
            memory_id: memory.memory_id.clone(),
            eligible: false,
            reason,
        };

        match memory.scope {
            ScopeKind::Global if !policy.include_global => {
                return deny(EligibilityReason::GlobalDisabled);
            }
            ScopeKind::Unlocated if !policy.include_unlocated => {
                return deny(EligibilityReason::Unlocated);
            }
            ScopeKind::Context => {
                let Some(context) = memory
                    .effective_context_id
                    .as_ref()
                    .or(memory.birth_context_id.as_ref())
                else {
                    return deny(EligibilityReason::ContextMissing);
                };
                if !policy.cross_context && !policy.context_cone.contains(context) {
                    return deny(EligibilityReason::WrongContext);
                }
            }
            _ => {}
        }

        if !memory.valid_at(policy.valid_at_ms) {
            return deny(EligibilityReason::Worldline);
        }
        if !policy.allowed_origins.contains(&memory.origin) {
            return deny(EligibilityReason::OriginDenied);
        }
        if memory.authority < policy.minimum_authority {
            return deny(EligibilityReason::AuthorityTooLow);
        }
        if memory
            .taints
            .iter()
            .any(|taint| policy.denied_taints.contains(taint))
        {
            return deny(EligibilityReason::Tainted);
        }
        if memory
            .allowed_purposes
            .as_ref()
            .is_some_and(|purposes| !purposes.contains(&policy.purpose))
        {
            return deny(EligibilityReason::PurposeDenied);
        }
        if memory
            .allowed_actors
            .as_ref()
            .is_some_and(|actors| !actors.contains(&policy.actor))
        {
            return deny(EligibilityReason::ActorDenied);
        }

        EligibilityDecision {
            memory_id: memory.memory_id.clone(),
            eligible: true,
            reason: EligibilityReason::Eligible,
        }
    }

    fn add_blocked(blocked: &mut BTreeMap<String, u64>, reason: EligibilityReason, count: u64) {
        if count > 0 {
            *blocked.entry(reason.key().into()).or_default() += count;
        }
    }

    pub fn compile(&self, memories: &MemoryMap, policy: RealityPolicy) -> CompiledEligibleSet {
        let mut blocked = BTreeMap::new();
        let all_context = self.all_context_members();
        let allowed_context = self.allowed_context_members(&policy);

        if !policy.cross_context {
            Self::add_blocked(
                &mut blocked,
                EligibilityReason::WrongContext,
                all_context.difference_len(&allowed_context),
            );
        }
        if !policy.include_global {
            Self::add_blocked(
                &mut blocked,
                EligibilityReason::GlobalDisabled,
                self.global_members.len(),
            );
        }
        if !policy.include_unlocated {
            Self::add_blocked(
                &mut blocked,
                EligibilityReason::Unlocated,
                self.unlocated_members.len(),
            );
        }
        Self::add_blocked(
            &mut blocked,
            EligibilityReason::ContextMissing,
            self.context_missing_members.len(),
        );

        let coarse = self.coarse_candidates(&policy);

        let mut origin_allowed = RoaringBitmap::new();
        for origin in &policy.allowed_origins {
            if let Some(bitmap) = self.origin_members.get(origin) {
                origin_allowed |= bitmap.clone();
            }
        }
        Self::add_blocked(
            &mut blocked,
            EligibilityReason::OriginDenied,
            coarse.difference_len(&origin_allowed),
        );
        let mut after_origin = coarse;
        after_origin &= origin_allowed;

        let mut authority_allowed = RoaringBitmap::new();
        for (authority, bitmap) in &self.authority_members {
            if *authority >= policy.minimum_authority {
                authority_allowed |= bitmap.clone();
            }
        }
        Self::add_blocked(
            &mut blocked,
            EligibilityReason::AuthorityTooLow,
            after_origin.difference_len(&authority_allowed),
        );
        let mut candidates = after_origin;
        candidates &= authority_allowed;
        let fine_candidate_count = candidates.len();

        let mut eligible = RoaringBitmap::new();
        let mut decisions = Vec::new();
        for ordinal in candidates.iter() {
            let Some(memory_id) = self.ordinals.memory_id(ordinal) else {
                continue;
            };
            let Some(memory) = memories.get(memory_id) else {
                continue;
            };
            let decision = Self::fine_decide(memory, &policy);
            if decision.eligible {
                eligible.insert(ordinal);
            } else {
                Self::add_blocked(&mut blocked, decision.reason, 1);
            }
            decisions.push(decision);
        }

        let memory_ids: BTreeSet<String> = eligible
            .iter()
            .filter_map(|ordinal| self.ordinals.memory_id(ordinal).map(String::from))
            .collect();
        let projection: Vec<(&str, String)> = memory_ids
            .iter()
            .filter_map(|memory_id| {
                memories
                    .get(memory_id)
                    .map(|memory| (memory_id.as_str(), memory.revision()))
            })
            .collect();
        // The eligible-universe seal intentionally excludes physical page layout.
        // It is the stable identity shared by capability and delta operations.
        let universe_seal = hash_json("eligible_", &projection);

        let page_ids = self.route_pages(&policy);
        let page_manifest_seals: Vec<String> = page_ids
            .iter()
            .filter_map(|page_id| self.pages.get(page_id))
            .map(|page| page.manifest_seal.clone())
            .collect();
        let routing_plan_seal = hash_json("routing_", &page_manifest_seals);
        // Physical page layout is an execution concern, not an authority input.
        // Re-paging or adding a row that fails fine eligibility may change the
        // routing seal, but must not revoke capabilities or perturb deltas.
        let reality_checkpoint = hash_json(
            "checkpoint_",
            &(policy.fingerprint.as_str(), universe_seal.as_str()),
        );

        CompiledEligibleSet {
            page_ids,
            page_manifest_seals,
            routing_plan_seal,
            universe_seal,
            reality_checkpoint,
            policy,
            ordinals: eligible,
            memory_ids,
            decisions,
            blocked_counts: blocked,
            fine_candidate_count,
            global_store_count: memories.len(),
        }
    }

    pub fn verify_page_purity(&self, memories: &MemoryMap) -> Vec<String> {
        let mut invalid = Vec::new();
        for page in self.pages.values() {
            for ordinal in &page.ordinals {
                let Some(memory_id) = self.ordinals.memory_id(*ordinal) else {
                    invalid.push(page.page_id.clone());
                    break;
                };
                let Some(memory) = memories.get(memory_id) else {
                    invalid.push(page.page_id.clone());
                    break;
                };
                let actual_scope = match memory.scope {
                    ScopeKind::Global => Some(GLOBAL_PAGE),
                    ScopeKind::Unlocated => Some(UNLOCATED_PAGE),
                    ScopeKind::Context => memory
                        .effective_context_id
                        .as_deref()
                        .or(memory.birth_context_id.as_deref()),
                };
                if actual_scope != Some(page.scope_key.as_str()) {
                    invalid.push(page.page_id.clone());
                    break;
                }
            }
        }
        invalid.sort();
        invalid.dedup();
        invalid
    }
}

fn terms(text: &str) -> Vec<String> {
    text.split(|character: char| {
        !(character.is_ascii_alphanumeric() || "_./:-".contains(character))
    })
    .filter(|term| !term.is_empty())
    .map(str::to_ascii_lowercase)
    .collect()
}

#[derive(Clone, Debug)]
pub struct RealityLexicalIndex {
    pub k1: f64,
    pub b: f64,
}

impl Default for RealityLexicalIndex {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl RealityLexicalIndex {
    pub fn score(
        &self,
        query: &str,
        memory_ids: &BTreeSet<String>,
        memories: &MemoryMap,
    ) -> BTreeMap<String, f64> {
        if memory_ids.is_empty() {
            return BTreeMap::new();
        }

        let documents: BTreeMap<String, Vec<String>> = memory_ids
            .iter()
            .filter_map(|memory_id| {
                memories.get(memory_id).map(|memory| {
                    let mut text = memory.content.clone();
                    if !memory.tags.is_empty() {
                        text.push(' ');
                        text.push_str(&memory.tags.join(" "));
                    }
                    (memory_id.clone(), terms(&text))
                })
            })
            .collect();
        let document_count = documents.len() as f64;
        let average_length =
            documents.values().map(Vec::len).sum::<usize>() as f64 / document_count.max(1.0);

        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        for document in documents.values() {
            let unique: BTreeSet<&String> = document.iter().collect();
            for term in unique {
                *document_frequency.entry(term.clone()).or_default() += 1;
            }
        }

        let query_terms = terms(query);
        let mut scores = BTreeMap::new();
        for (memory_id, document) in documents {
            let mut term_frequency: HashMap<String, usize> = HashMap::new();
            for token in &document {
                *term_frequency.entry(token.clone()).or_default() += 1;
            }
            let document_length = document.len() as f64;
            let mut score = 0.0;
            for term in &query_terms {
                let frequency = *term_frequency.get(term).unwrap_or(&0) as f64;
                if frequency == 0.0 {
                    continue;
                }
                let frequency_documents = *document_frequency.get(term).unwrap_or(&0) as f64;
                let inverse_document_frequency = (1.0
                    + (document_count - frequency_documents + 0.5) / (frequency_documents + 0.5))
                    .ln();
                let denominator = frequency
                    + self.k1
                        * (1.0 - self.b + self.b * document_length / average_length.max(1e-12));
                score += inverse_document_frequency * (frequency * (self.k1 + 1.0)) / denominator;
            }
            scores.insert(memory_id, score);
        }
        scores
    }
}

fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    left.iter().zip(right).map(|(x, y)| x * y).sum()
}

fn hashed_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let mut result = vec![0.0; dimensions];
    for token in terms(text) {
        let hash = blake3::hash(token.as_bytes());
        let bytes = hash.as_bytes();
        let bucket =
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize % dimensions;
        result[bucket] += if bytes[4] & 1 == 1 { 1.0 } else { -1.0 };
    }
    unit(result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorMode {
    Exact,
    PagedAnn,
}

#[derive(Clone, Debug)]
pub struct RealityVectorIndex {
    pub exact_threshold: usize,
    pub dimensions: usize,
    vectors: HashMap<String, Vec<f32>>,
}

impl RealityVectorIndex {
    pub fn new(exact_threshold: usize, dimensions: usize) -> Self {
        Self {
            exact_threshold: exact_threshold.max(1),
            dimensions: dimensions.max(1),
            vectors: HashMap::new(),
        }
    }

    pub fn register(&mut self, memory_id: &str, vector: Vec<f32>) {
        self.vectors.insert(memory_id.into(), unit(vector));
    }

    fn vector_for(&self, memory_id: &str, memories: &MemoryMap) -> Vec<f32> {
        self.vectors
            .get(memory_id)
            .cloned()
            .unwrap_or_else(|| hashed_embedding(&memories[memory_id].content, self.dimensions))
    }

    pub fn score(
        &self,
        query: &str,
        eligible: &CompiledEligibleSet,
        memories: &MemoryMap,
        explicit_query: Option<&[f32]>,
    ) -> (BTreeMap<String, f32>, VectorMode) {
        let query_vector = explicit_query
            .map(|vector| unit(vector.to_vec()))
            .unwrap_or_else(|| hashed_embedding(query, self.dimensions));
        let mode = if eligible.memory_ids.len() <= self.exact_threshold {
            VectorMode::Exact
        } else {
            // This standalone crate models page routing deterministically. The
            // Vestige adapter maps each page to a Context-pure USearch index.
            VectorMode::PagedAnn
        };
        let scores = eligible
            .memory_ids
            .iter()
            .map(|memory_id| {
                (
                    memory_id.clone(),
                    cosine(&query_vector, &self.vector_for(memory_id, memories)),
                )
            })
            .collect();
        (scores, mode)
    }
}

#[derive(Clone, Debug)]
pub struct RecallHit {
    pub memory_id: String,
    pub score: f64,
}

#[derive(Clone, Debug)]
pub struct IndexedRecallResult {
    pub hits: Vec<RecallHit>,
    pub eligible_universe_seal: String,
    pub blocked_counts: BTreeMap<String, u64>,
    pub fine_candidate_count: u64,
    pub global_store_count: usize,
    pub vector_mode: VectorMode,
    pub routed_pages: Vec<String>,
    pub reality_checkpoint: String,
}

pub struct IndexedRecallEngine {
    pub index: RealityIndex,
    pub lexical: RealityLexicalIndex,
    pub vector: RealityVectorIndex,
}

impl IndexedRecallEngine {
    pub fn new(page_size: usize, exact_threshold: usize) -> Self {
        Self {
            index: RealityIndex::new(page_size),
            lexical: RealityLexicalIndex::default(),
            vector: RealityVectorIndex::new(exact_threshold, 64),
        }
    }

    pub fn rebuild(&mut self, memories: &MemoryMap) {
        self.index.rebuild(memories);
    }

    pub fn recall(
        &self,
        query: &str,
        policy: RealityPolicy,
        memories: &MemoryMap,
        limit: usize,
        explicit_query: Option<&[f32]>,
    ) -> IndexedRecallResult {
        let eligible = self.index.compile(memories, policy);
        let lexical_scores = self.lexical.score(query, &eligible.memory_ids, memories);
        let (vector_scores, vector_mode) =
            self.vector
                .score(query, &eligible, memories, explicit_query);

        let lexical_ranks: HashMap<String, usize> = {
            let mut ranked: Vec<(String, f64)> = eligible
                .memory_ids
                .iter()
                .map(|memory_id| {
                    (
                        memory_id.clone(),
                        *lexical_scores.get(memory_id).unwrap_or(&0.0),
                    )
                })
                .collect();
            ranked.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            ranked
                .into_iter()
                .enumerate()
                .map(|(index, (memory_id, _))| (memory_id, index + 1))
                .collect()
        };
        let vector_ranks: HashMap<String, usize> = {
            let mut ranked: Vec<(String, f32)> = eligible
                .memory_ids
                .iter()
                .map(|memory_id| {
                    (
                        memory_id.clone(),
                        *vector_scores.get(memory_id).unwrap_or(&0.0),
                    )
                })
                .collect();
            ranked.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            ranked
                .into_iter()
                .enumerate()
                .map(|(index, (memory_id, _))| (memory_id, index + 1))
                .collect()
        };

        let mut hits: Vec<RecallHit> = eligible
            .memory_ids
            .iter()
            .map(|memory_id| {
                let lexical_rank = *lexical_ranks
                    .get(memory_id)
                    .expect("eligible memory has lexical rank")
                    as f64;
                let vector_rank = *vector_ranks
                    .get(memory_id)
                    .expect("eligible memory has vector rank")
                    as f64;
                RecallHit {
                    memory_id: memory_id.clone(),
                    score: 0.3 / (60.0 + lexical_rank) + 0.7 / (60.0 + vector_rank),
                }
            })
            .collect();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.memory_id.cmp(&right.memory_id))
        });
        hits.truncate(limit.max(1));

        IndexedRecallResult {
            hits,
            eligible_universe_seal: eligible.universe_seal,
            blocked_counts: eligible.blocked_counts,
            fine_candidate_count: eligible.fine_candidate_count,
            global_store_count: eligible.global_store_count,
            vector_mode,
            routed_pages: eligible.page_ids,
            reality_checkpoint: eligible.reality_checkpoint,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::EngramProjection;

    fn policy(context: &str) -> RealityPolicy {
        RealityPolicy::new(
            Some(context.into()),
            [context.to_string()].into_iter().collect(),
            100,
        )
    }

    #[test]
    fn sibling_population_never_enters_rank_math() {
        let mut memories = MemoryMap::new();
        memories.insert(
            "a".into(),
            EngramProjection::context("a", "Backups S3", "A"),
        );
        for index in 0..10_000 {
            let id = format!("b{index}");
            memories.insert(
                id.clone(),
                EngramProjection::context(&id, "Backups Backblaze backup backup", "B"),
            );
        }
        let mut engine = IndexedRecallEngine::new(128, 50_000);
        engine.rebuild(&memories);
        let result = engine.recall("backups", policy("A"), &memories, 5, None);
        assert_eq!(result.global_store_count, 10_001);
        assert_eq!(result.fine_candidate_count, 1);
        assert_eq!(result.hits[0].memory_id, "a");
    }

    #[test]
    fn local_bm25_ignores_sibling_corpus_statistics() {
        let mut memories = MemoryMap::new();
        memories.insert("a".into(), EngramProjection::context("a", "backup s3", "A"));
        let mut engine = IndexedRecallEngine::new(64, 50_000);
        engine.rebuild(&memories);
        let before = engine
            .recall("backup s3", policy("A"), &memories, 5, None)
            .hits[0]
            .score;
        for index in 0..3_000 {
            let id = format!("b{index}");
            memories.insert(id.clone(), EngramProjection::context(&id, "backup s3", "B"));
        }
        engine.rebuild(&memories);
        let after = engine
            .recall("backup s3", policy("A"), &memories, 5, None)
            .hits[0]
            .score;
        assert_eq!(before, after);
    }

    #[test]
    fn page_manifests_are_pure() {
        let mut memories = MemoryMap::new();
        for index in 0..40 {
            let id = format!("a{index}");
            memories.insert(id.clone(), EngramProjection::context(&id, "x", "A"));
        }
        for index in 0..40 {
            let id = format!("b{index}");
            memories.insert(id.clone(), EngramProjection::context(&id, "x", "B"));
        }
        let mut index = RealityIndex::new(7);
        index.rebuild(&memories);
        assert!(index.verify_page_purity(&memories).is_empty());
        assert!(
            index
                .pages
                .values()
                .all(|page| page.manifest_seal.starts_with("page_"))
        );
    }

    #[test]
    fn negative_space_accounts_for_the_complete_store() {
        let mut memories = MemoryMap::new();
        memories.insert(
            "eligible".into(),
            EngramProjection::context("eligible", "x", "A"),
        );
        memories.insert(
            "sibling".into(),
            EngramProjection::context("sibling", "x", "B"),
        );
        memories.insert("global".into(), EngramProjection::global("global", "x"));
        memories.insert(
            "unlocated".into(),
            EngramProjection::unlocated("unlocated", "x"),
        );

        let mut denied_origin = EngramProjection::context("origin", "x", "A");
        denied_origin.origin = "web".into();
        memories.insert("origin".into(), denied_origin);

        let mut weak = EngramProjection::context("weak", "x", "A");
        weak.authority = 0;
        memories.insert("weak".into(), weak);

        let mut expired = EngramProjection::context("expired", "x", "A");
        expired.valid_until_ms = Some(100);
        memories.insert("expired".into(), expired);

        let mut index = RealityIndex::new(8);
        index.rebuild(&memories);
        let mut p = policy("A");
        p.include_global = false;
        p.allowed_origins = ["user".into()].into_iter().collect();
        p.minimum_authority = 3;
        p.refresh_fingerprint();
        let compiled = index.compile(&memories, p);
        let blocked: u64 = compiled.blocked_counts.values().sum();
        assert_eq!(blocked as usize + compiled.memory_ids.len(), memories.len());
        assert_eq!(
            compiled.memory_ids,
            ["eligible".into()].into_iter().collect()
        );
    }
    #[test]
    fn reality_checkpoint_ignores_siblings_but_tracks_active_reality() {
        let mut memories = MemoryMap::new();
        memories.insert(
            "a".into(),
            EngramProjection::context("a", "A database", "A"),
        );
        memories.insert(
            "b".into(),
            EngramProjection::context("b", "B database", "B"),
        );
        let mut engine = IndexedRecallEngine::new(8, 100);
        engine.rebuild(&memories);
        let before = engine.recall("database", policy("A"), &memories, 5, None);

        memories.get_mut("b").expect("sibling memory").content =
            "B database changed radically".into();
        engine.rebuild(&memories);
        let sibling_change = engine.recall("database", policy("A"), &memories, 5, None);
        assert_eq!(before.reality_checkpoint, sibling_change.reality_checkpoint);

        memories.get_mut("a").expect("active memory").content = "A database changed".into();
        engine.rebuild(&memories);
        let active_change = engine.recall("database", policy("A"), &memories, 5, None);
        assert_ne!(before.reality_checkpoint, active_change.reality_checkpoint);
    }

    #[test]
    fn ineligible_active_context_row_changes_routing_not_authority() {
        let mut memories = MemoryMap::new();
        memories.insert(
            "a".into(),
            EngramProjection::context("a", "A database", "A"),
        );
        let mut engine = IndexedRecallEngine::new(1, 100);
        engine.rebuild(&memories);
        let before = engine.index.compile(&memories, policy("A"));

        let mut denied =
            EngramProjection::context("denied", "Untrusted in-context instruction", "A");
        denied.origin = "web".into();
        memories.insert("denied".into(), denied);
        engine.rebuild(&memories);
        let after = engine.index.compile(&memories, policy("A"));

        assert_eq!(before.universe_seal, after.universe_seal);
        assert_eq!(before.reality_checkpoint, after.reality_checkpoint);
        assert_ne!(before.routing_plan_seal, after.routing_plan_seal);
    }
}
