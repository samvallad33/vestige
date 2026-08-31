//! Active Forgetting — Top-Down Inhibitory Control of Memory (v2.0.5)
//!
//! Implements user-initiated memory suppression, distinct from passive FSRS
//! decay and from bottom-up retrieval-induced forgetting (Anderson 1994,
//! `memory_states.rs`). This module models the right-lateral-prefrontal-cortex
//! gated inhibitory pathway, where top-down cognitive control compounds with
//! each stopping attempt (Suppression-Induced Forgetting) and spreads via a
//! Rac1-GTPase-like cascade to co-activated synaptic neighbors.
//!
//! ## References
//!
//! - Anderson, M. C., Crespo-Garcia, M., & Subbulakshmi, S. (2025). Brain
//!   mechanisms underlying the inhibitory control of thought. *Nature Reviews
//!   Neuroscience* 26(7):415-437. DOI: 10.1038/s41583-025-00929-y. Establishes
//!   rDLPFC as the domain-general inhibitory controller; SIF scales with
//!   stopping attempts; incentive-resistant.
//! - Cervantes-Sandoval, I., Davis, R. L., & Berry, J. A. (2020). Rac1 Impairs
//!   Forgetting-Induced Cellular Plasticity in Mushroom Body Output Neurons.
//!   *Front Cell Neurosci* 14:258. DOI 10.3389/fncel.2020.00258. Implicates
//!   Rac1 in active, biologically driven forgetting rather than passive decay.
//!
//! Honesty note: the compounding-with-repeated-attempts property is a real
//! operationalisation of the SIF finding. The neighbour cascade is our own
//! design, named after Rac1 by analogy only — the cited paper does not
//! demonstrate forgetting spreading to co-activated neighbours. Every constant
//! below is our tuning, not a published value.
//!
//! ## Contrast with existing modules
//!
//! - `memory_states.rs` (Anderson 1994, RIF): BOTTOM-UP, passive consequence
//!   of retrieval competition. When memory A wins a query, its competitors
//!   automatically lose retrievability.
//! - `active_forgetting.rs` (Anderson 2025, SIF + Davis Rac1): TOP-DOWN,
//!   user-initiated via the `suppress` MCP tool. Compounds with each call.
//!   Spreads to neighbors. Reversible within a 24h labile window.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Default SIF penalty coefficient per suppression increment.
pub const DEFAULT_SIF_K: f64 = 0.15;

/// Maximum cumulative penalty from compounding suppression. Our tuning; the
/// saturating shape follows the SIF literature, the value does not come from it.
pub const DEFAULT_MAX_PENALTY: f64 = 0.8;

/// Cascade attenuation factor for Rac1 spreading to co-activated neighbors.
pub const DEFAULT_CASCADE_DECAY: f64 = 0.3;

/// Labile window in hours during which a suppression may be reversed.
/// Parallels Nader's 5-minute reconsolidation window on a 24-hour axis.
pub const DEFAULT_LABILE_HOURS: i64 = 24;

/// Maximum per-neighbor retrieval-strength decrement during cascade.
pub const DEFAULT_CASCADE_RETRIEVAL_DECREMENT_CAP: f64 = 0.15;

/// Top-down inhibitory control over memory retrieval.
///
/// Stateless — all persistent state lives on the `knowledge_nodes` table
/// (columns `suppression_count`, `suppressed_at`). This struct exposes pure
/// helper functions consumed by `Storage::suppress_memory`,
/// `Storage::reverse_suppression`, `Storage::apply_rac1_cascade`, and the
/// `search_unified` score adjustment stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveForgettingSystem {
    /// Penalty coefficient per suppression increment (SIF).
    pub k: f64,
    /// Maximum cumulative penalty cap.
    pub max_penalty: f64,
    /// Cascade attenuation factor for Rac1 spreading.
    pub cascade_decay: f64,
    /// Reversal window in hours.
    pub labile_hours: i64,
}

impl Default for ActiveForgettingSystem {
    fn default() -> Self {
        Self {
            k: DEFAULT_SIF_K,
            max_penalty: DEFAULT_MAX_PENALTY,
            cascade_decay: DEFAULT_CASCADE_DECAY,
            labile_hours: DEFAULT_LABILE_HOURS,
        }
    }
}

impl ActiveForgettingSystem {
    /// Create a new system with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the retrieval-score penalty for a memory with the given
    /// suppression count. Penalty grows linearly then saturates at
    /// `max_penalty` (Anderson's empirical SIF ceiling).
    ///
    /// Applied in `search_unified` as `score *= (1.0 - penalty)`.
    pub fn retrieval_penalty(&self, suppression_count: i32) -> f64 {
        if suppression_count <= 0 {
            return 0.0;
        }
        (self.k * suppression_count as f64).min(self.max_penalty)
    }

    /// Return `true` if a suppression is within the labile window and
    /// therefore reversible. Matches reconsolidation semantics on a 24h axis.
    pub fn is_reversible(&self, suppressed_at: DateTime<Utc>) -> bool {
        Utc::now() - suppressed_at < Duration::hours(self.labile_hours)
    }

    /// Stability multiplier to apply to a neighbor of a suppressed memory
    /// during the Rac1 cascade. Stronger co-activation edges propagate more
    /// decay. A 1.0 edge yields `(1 - cascade_decay)` = 0.7 by default
    /// (30% stability loss per cascade hop), clamped never below 0.1.
    pub fn cascade_stability_factor(&self, edge_strength: f64) -> f64 {
        (1.0 - self.cascade_decay * edge_strength.clamp(0.0, 1.0)).max(0.1)
    }

    /// Retrieval-strength decrement for a cascade neighbor, proportional to
    /// co-activation edge strength and capped at
    /// `DEFAULT_CASCADE_RETRIEVAL_DECREMENT_CAP`.
    pub fn cascade_retrieval_decrement(&self, edge_strength: f64) -> f64 {
        (0.05 * edge_strength.clamp(0.0, 1.0)).min(DEFAULT_CASCADE_RETRIEVAL_DECREMENT_CAP)
    }

    /// Time remaining in the labile window, or `None` if expired.
    pub fn remaining_labile_time(&self, suppressed_at: DateTime<Utc>) -> Option<Duration> {
        let window = Duration::hours(self.labile_hours);
        let elapsed = Utc::now() - suppressed_at;
        if elapsed >= window {
            None
        } else {
            Some(window - elapsed)
        }
    }

    /// Deadline timestamp after which reversal will fail.
    pub fn reversible_until(&self, suppressed_at: DateTime<Utc>) -> DateTime<Utc> {
        suppressed_at + Duration::hours(self.labile_hours)
    }
}

/// Aggregate statistics about active-forgetting state across all memories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SuppressionStats {
    /// Total memories with suppression_count > 0.
    pub total_suppressed: usize,
    /// Memories suppressed within the last `labile_hours` (still reversible).
    pub recently_reversible: usize,
    /// Mean suppression_count across all suppressed memories.
    pub avg_suppression_count: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sif_penalty_compounds() {
        let sys = ActiveForgettingSystem::new();
        assert_eq!(sys.retrieval_penalty(0), 0.0);
        assert!((sys.retrieval_penalty(1) - 0.15).abs() < 1e-9);
        assert!((sys.retrieval_penalty(2) - 0.30).abs() < 1e-9);
        assert!((sys.retrieval_penalty(5) - 0.75).abs() < 1e-9);
        // Saturates at max_penalty
        assert!((sys.retrieval_penalty(6) - 0.80).abs() < 1e-9);
        assert!((sys.retrieval_penalty(100) - 0.80).abs() < 1e-9);
    }

    #[test]
    fn test_labile_window_reversible() {
        let sys = ActiveForgettingSystem::new();
        let recent = Utc::now() - Duration::hours(23);
        assert!(sys.is_reversible(recent));
        let expired = Utc::now() - Duration::hours(25);
        assert!(!sys.is_reversible(expired));
        assert!(sys.is_reversible(Utc::now()));
    }

    #[test]
    fn test_cascade_attenuation() {
        let sys = ActiveForgettingSystem::new();
        let strong = sys.cascade_stability_factor(0.9);
        let weak = sys.cascade_stability_factor(0.1);
        assert!(strong < weak, "strong edges should propagate more decay");
        // Zero edge → no decay
        assert!((sys.cascade_stability_factor(0.0) - 1.0).abs() < 1e-9);
        // Factor never zeroes out
        assert!(sys.cascade_stability_factor(1.0) >= 0.1);
    }

    #[test]
    fn test_default_params_reasonable() {
        let sys = ActiveForgettingSystem::new();
        assert!(sys.k > 0.0 && sys.k <= 0.25, "k should be in (0, 0.25]");
        assert!(
            sys.max_penalty >= 0.5 && sys.max_penalty <= 0.95,
            "max_penalty should be in [0.5, 0.95]"
        );
        assert!(sys.labile_hours >= 12 && sys.labile_hours <= 72);
        assert!(sys.cascade_decay > 0.0 && sys.cascade_decay < 1.0);
    }

    #[test]
    fn test_reversible_until_deadline() {
        let sys = ActiveForgettingSystem::new();
        let now = Utc::now();
        let deadline = sys.reversible_until(now);
        let expected = now + Duration::hours(24);
        assert!((deadline - expected).num_milliseconds().abs() < 100);
    }

    #[test]
    fn test_remaining_labile_time_expired_returns_none() {
        let sys = ActiveForgettingSystem::new();
        let past = Utc::now() - Duration::hours(30);
        assert!(sys.remaining_labile_time(past).is_none());
        let recent = Utc::now() - Duration::hours(10);
        let remaining = sys.remaining_labile_time(recent);
        assert!(remaining.is_some());
        // Should have ~14 hours left (24h window - 10h elapsed)
        let hours_left = remaining.unwrap().num_hours();
        assert!((13..=14).contains(&hours_left));
    }

    #[test]
    fn test_cascade_retrieval_decrement_capped() {
        let sys = ActiveForgettingSystem::new();
        assert!((sys.cascade_retrieval_decrement(0.0) - 0.0).abs() < 1e-9);
        assert!(sys.cascade_retrieval_decrement(0.5) <= DEFAULT_CASCADE_RETRIEVAL_DECREMENT_CAP);
        assert!(sys.cascade_retrieval_decrement(1.0) <= DEFAULT_CASCADE_RETRIEVAL_DECREMENT_CAP);
    }
}
