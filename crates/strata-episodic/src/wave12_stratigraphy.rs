//! Wave 12 (#661–#670): Historical, diplomatic & stratigraphic operators
//! (terminus bounds, stratigraphy, palimpsests, criterion of embarrassment,
//! dendrochronology cross-dating, prosopography, die-links, damnatio,
//! lectio difficilior, Homeric entity capsules).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// #661: Interval dating from exact constraints (*terminus post quem* / *ante quem*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminusInterval {
    pub earliest_ms: u64,
    pub latest_ms: u64,
}

impl TerminusInterval {
    pub fn intersect(self, other: Self) -> Option<Self> {
        let earliest_ms = self.earliest_ms.max(other.earliest_ms);
        let latest_ms = self.latest_ms.min(other.latest_ms);
        (earliest_ms <= latest_ms).then_some(Self {
            earliest_ms,
            latest_ms,
        })
    }
}

/// #662: Import order as uncertain metadata (`Stratigraphy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChronologyCertainty {
    ExactEventClock,
    StratigraphicBatchDepositOnly,
}

/// #663: Preserved historical undertext (`Palimpsest`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PalimpsestNode {
    pub node_id: String,
    pub overtext: String,
    pub undertext_layers: Vec<String>,
}

impl PalimpsestNode {
    pub fn new(node_id: impl Into<String>, initial: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            overtext: initial.into(),
            undertext_layers: Vec::new(),
        }
    }

    pub fn revise(&mut self, next_text: impl Into<String>) {
        let old = std::mem::replace(&mut self.overtext, next_text.into());
        self.undertext_layers.push(old);
    }
}

/// #664: Evidence quality without self-interest heuristics (`EmbarrassmentCriterion`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationPosture {
    StandardClaim,
    VerifiedFailureAdmission,
}

/// #665: Cross-dating against exact life-event anchors (`Dendrochronology` tree rings).
pub fn cross_date_from_anchors(
    after_anchor_ms: Option<u64>,
    before_anchor_ms: Option<u64>,
) -> Option<TerminusInterval> {
    let earliest_ms = after_anchor_ms.unwrap_or(0);
    let latest_ms = before_anchor_ms.unwrap_or(u64::MAX);
    (earliest_ms <= latest_ms).then_some(TerminusInterval {
        earliest_ms,
        latest_ms,
    })
}

/// #666: Explainable shared-context network views (`Prosopography`).
pub fn prosopography_co_participants(
    event_participants: &BTreeMap<String, BTreeSet<String>>, // event_id -> entities
    focal_entity: &str,
) -> BTreeMap<String, Vec<String>> {
    let mut shared: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (ev_id, members) in event_participants {
        if members.contains(focal_entity) {
            for peer in members {
                if peer != focal_entity {
                    shared.entry(peer.clone()).or_default().push(ev_id.clone());
                }
            }
        }
    }
    shared
}

/// #667: Witnessed recurring-problem candidates (`DieLink` exact signature grouping).
pub fn group_die_links(
    incidents: &[(String, String, String)], // (incident_id, error_code, file_anchor)
) -> BTreeMap<(String, String), Vec<String>> {
    let mut map: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for (inc_id, err, file) in incidents {
        map.entry((err.clone(), file.clone()))
            .or_default()
            .push(inc_id.clone());
    }
    map.into_iter().filter(|(_, ids)| ids.len() >= 2).collect()
}

/// #668: Owner-controlled surfacing suppression with structural integrity (`Damnatio`).
#[derive(Debug, Default)]
pub struct DamnatioRegistry {
    muted_from_surfacing: BTreeSet<String>,
}

impl DamnatioRegistry {
    pub fn mute_proactive_surfacing(&mut self, node_id: impl Into<String>) {
        self.muted_from_surfacing.insert(node_id.into());
    }

    pub fn allow_proactive(&self, node_id: &str) -> bool {
        !self.muted_from_surfacing.contains(node_id)
    }

    /// Structural walks ALWAYS traverse muted nodes so causal integrity never breaks!
    pub fn allow_structural_walk(&self, _node_id: &str) -> bool {
        true
    }
}

/// #669: Verbatim critical-term preservation (`LectioDifficilior`).
pub fn lectio_difficilior_preserved(summary_text: &str, locked_verbatim_terms: &[&str]) -> Result<(), Vec<String>> {
    let missing: Vec<String> = locked_verbatim_terms
        .iter()
        .filter(|term| !summary_text.contains(*term))
        .map(|s| (*s).to_string())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing)
    }
}

/// #670: Exact compact entity capsules (`HomericEpithet`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HomericEntityCapsule {
    pub entity: String,
    pub epithet_facts: Vec<String>,
    pub backing_memory_ids: Vec<String>,
}

pub fn compile_homeric_capsule(
    entity: &str,
    facts_with_ids: &[(String, String)], // (mem_id, concise_fact)
    max_facts: usize,
) -> HomericEntityCapsule {
    let mut epithet_facts = Vec::new();
    let mut backing_memory_ids = Vec::new();
    for (id, fact) in facts_with_ids.iter().take(max_facts) {
        backing_memory_ids.push(id.clone());
        epithet_facts.push(fact.clone());
    }
    HomericEntityCapsule {
        entity: entity.to_string(),
        epithet_facts,
        backing_memory_ids,
    }
}
