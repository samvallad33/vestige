//! Wave 10 (#641–#650): Musical / structural interval, counterpoint & recurrence operators.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// #641: Explicit foundational life/project arcs (`Ursatz`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrsatzLifeArc {
    pub arc_id: String,
    pub title: String,
    pub child_milestones: Vec<String>,
}

/// #642: Persistent scoped context intervals (`Prolongation`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProlongationInterval {
    pub context_id: String,
    pub start_seq: u64,
    pub end_seq: Option<u64>,
}

impl ProlongationInterval {
    pub fn governs(&self, seq: u64) -> bool {
        seq >= self.start_seq && self.end_seq.is_none_or(|end| seq < end)
    }
}

/// #643: Entity timeline isolation (`Counterpoint`).
#[derive(Debug, Default)]
pub struct CounterpointTimelines {
    streams: BTreeMap<String, Vec<(u64, String)>>, // entity -> [(seq, event_id)]
}

impl CounterpointTimelines {
    pub fn record(&mut self, entity: &str, seq: u64, event_id: &str) {
        self.streams
            .entry(entity.to_string())
            .or_default()
            .push((seq, event_id.to_string()));
    }

    pub fn entity_stream(&self, entity: &str) -> &[(u64, String)] {
        self.streams.get(entity).map_or(&[], Vec::as_slice)
    }

    pub fn joint_consonances(&self, entity_a: &str, entity_b: &str) -> Vec<String> {
        let a_events: BTreeSet<&str> = self
            .entity_stream(entity_a)
            .iter()
            .map(|(_, ev)| ev.as_str())
            .collect();
        self.entity_stream(entity_b)
            .iter()
            .filter(|(_, ev)| a_events.contains(ev.as_str()))
            .map(|(_, ev)| ev.clone())
            .collect()
    }
}

/// #644: Bounded temporary context anchors (`PedalPoint`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PedalPointAnchor {
    pub handle: String,
    pub pinned_at_turn: u32,
    pub max_turns: u32,
}

impl PedalPointAnchor {
    pub fn is_live(&self, current_turn: u32) -> bool {
        current_turn >= self.pinned_at_turn
            && current_turn.saturating_sub(self.pinned_at_turn) < self.max_turns
    }
}

/// #645: Expected versus actual outcome views (`DeceptiveCadence`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeceptiveCadenceView {
    pub expectation_id: String,
    pub actual_outcome_id: String,
    pub diverged: bool,
}

pub fn compare_expected_actual(exp_id: &str, exp_val: &str, act_id: &str, act_val: &str) -> DeceptiveCadenceView {
    DeceptiveCadenceView {
        expectation_id: exp_id.to_string(),
        actual_outcome_id: act_id.to_string(),
        diverged: exp_val != act_val,
    }
}

/// #646: Recorded-topology motif recurrence (`Leitmotif` exact edge-kind sequence).
pub fn count_leitmotif_occurrences(edge_kind_trace: &[String], motif: &[String]) -> usize {
    if motif.is_empty() || edge_kind_trace.len() < motif.len() {
        return 0;
    }
    edge_kind_trace
        .windows(motif.len())
        .filter(|w| *w == motif)
        .count()
}

/// #647: Explicit open-loop lifecycle (`Preparation -> Suspension -> Resolution`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenLoopState {
    Prepared,
    Suspended { blocker_id: String },
    Resolved { receipt_id: String },
}

/// #648: Exact recurrence collision calendar (`Polyrhythm` LCM collision).
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

pub fn polyrhythm_next_collision(period_a: u64, period_b: u64) -> u64 {
    if period_a == 0 || period_b == 0 {
        return 0;
    }
    (period_a / gcd(period_a, period_b)).saturating_mul(period_b)
}

/// #649: Bounded exact-edge activation neighborhoods (`Overtones` 1/hop decay).
pub fn overtone_exact_neighborhood(
    seed: &str,
    edges: &[(String, String)],
    max_hops: usize,
) -> BTreeMap<String, u64> {
    let mut distances: BTreeMap<String, usize> = BTreeMap::new();
    distances.insert(seed.to_string(), 0);
    let mut frontier = vec![seed.to_string()];
    for hop in 1..=max_hops {
        let mut next = Vec::new();
        for node in &frontier {
            for (u, v) in edges {
                if u == node && !distances.contains_key(v) {
                    distances.insert(v.clone(), hop);
                    next.push(v.clone());
                }
            }
        }
        frontier = next;
    }
    distances
        .into_iter()
        .filter(|(_, d)| *d > 0)
        .map(|(k, d)| (k, 1000 / (d as u64)))
        .collect()
}

/// #650: Opt-in absence-duration signals (`Rests`).
pub fn opt_in_absence_rest_ms(opted_in: bool, last_event_ms: u64, now_ms: u64) -> Option<u64> {
    opted_in.then(|| now_ms.saturating_sub(last_event_ms))
}
