//! # Agent Black Box, Receipts & Memory PRs — the cognitive flight recorder
//!
//! This module holds the **pure** data model and classification logic for three
//! tightly-related capabilities that together make Vestige *the black box,
//! immune system, and cinematic debugger for agent memory*:
//!
//! 1. **Agent Black Box** — a replayable trace of everything an agent run did to
//!    memory: prompt → retrieved → suppressed → activated edges → tool calls →
//!    writes → contradictions → vetoes → dream consolidation → final answer.
//!    The event model is [`MemoryTraceEvent`].
//!
//! 2. **Memory Receipts** — every important retrieval returns a structured
//!    [`Receipt`]: what was retrieved, what was suppressed and why, the
//!    activation path that surfaced it, the trust floor, the decay risk, and any
//!    mutations. A receipt is the "nutrition label" for a piece of agent memory.
//!
//! 3. **Memory PRs** — changes to an agent's *brain* are reviewed like changes
//!    to code. Ordinary context auto-commits (and always leaves a receipt), but
//!    risky writes — contradictions against high-trust memory, supersede / forget
//!    / merge / protect, identity / preference / workflow / positioning facts,
//!    permission / auth / security / money / legal facts, dream consolidation
//!    proposals, decay-below-threshold resurrection, low-confidence batch
//!    imports, and weak-provenance connector writes — open a reviewable
//!    [`MemoryPr`]. The gating decision is [`classify_write`].
//!
//! ## Design north star (shared with [`crate::advanced::merge_supersede`])
//!
//! - **append-only** — trace events are never mutated, only appended, so a run
//!   replays exactly as the agent experienced it.
//! - **self-explaining** — every gated write carries the [`RiskSignal`]s that
//!   explain *why* it needs review, in plain language.
//! - **opt-in friction** — the default [`ReviewMode::RiskGated`] keeps ordinary
//!   memory frictionless and only opens a PR when the agent tries to rewrite its
//!   own brain. [`ReviewMode::Fast`] never gates; [`ReviewMode::Paranoid`] gates
//!   every write.
//! - **DB-free** — this module is pure logic so it is unit-testable without a
//!   database. Persistence (the `agent_traces`, `memory_receipts`, and
//!   `memory_prs` tables) lives in [`crate::storage`].
//!
//! The killer line, made literal by [`classify_write`]:
//!
//! > Vestige auto-remembers ordinary context, but opens a Memory PR when the
//! > agent tries to rewrite its own brain.

use serde::{Deserialize, Serialize};

mod receipt;
mod review;

pub use receipt::{DecayRisk, Receipt, ReceiptMutation, SuppressedReceiptEntry};
pub use review::{
    classify_write, MemoryPr, MemoryPrAction, MemoryPrKind, MemoryPrStatus, ReviewMode, RiskClass,
    RiskSignal, WriteContext, HIGH_TRUST_FLOOR, LOW_CONFIDENCE_FLOOR,
};

// ============================================================================
// TRACE EVENTS — the black-box flight recorder
// ============================================================================

/// One append-only event in an agent run's black-box trace.
///
/// Mirrors the TypeScript `MemoryTraceEvent` union exactly (tagged on `type`,
/// camelCase fields) so the dashboard, the `vestige://trace/{runId}` MCP
/// resource, and the exported `.vestige-trace.json` all speak one schema.
///
/// ```ts
/// type MemoryTraceEvent =
///   | { type: "mcp.call"; runId; tool; argsHash; at }
///   | { type: "memory.retrieve"; runId; ids; activation; at }
///   | { type: "memory.suppress"; runId; id; reason }
///   | { type: "memory.write"; runId; id; diff; source }
///   | { type: "sanhedrin.veto"; runId; claim; evidenceIds; confidence }
///   | { type: "dream.patch"; runId; proposalIds; at };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum MemoryTraceEvent {
    /// An MCP tool was invoked. The args are stored as a hash (not the raw
    /// payload) so traces never leak prompt contents or secrets.
    #[serde(rename = "mcp.call")]
    McpCall {
        #[serde(rename = "runId")]
        run_id: String,
        tool: String,
        #[serde(rename = "argsHash")]
        args_hash: String,
        at: i64,
    },

    /// Memories were retrieved, with per-id spreading-activation strength so the
    /// graph replay can pulse exactly the nodes the agent saw, at their weight.
    #[serde(rename = "memory.retrieve")]
    MemoryRetrieve {
        #[serde(rename = "runId")]
        run_id: String,
        ids: Vec<String>,
        activation: std::collections::BTreeMap<String, f64>,
        at: i64,
    },

    /// A memory that *would* have surfaced was suppressed, with the reason —
    /// this is the "what the agent chose NOT to use" channel.
    #[serde(rename = "memory.suppress")]
    MemorySuppress {
        #[serde(rename = "runId")]
        run_id: String,
        id: String,
        reason: SuppressReason,
        #[serde(default)]
        at: i64,
    },

    /// A memory was written / strengthened. `diff` is an opaque JSON description
    /// of the change; `source` records who caused it.
    #[serde(rename = "memory.write")]
    MemoryWrite {
        #[serde(rename = "runId")]
        run_id: String,
        id: String,
        diff: serde_json::Value,
        source: WriteSource,
        #[serde(default)]
        at: i64,
    },

    /// A contradiction was detected between memories during a run — its own
    /// first-class event (not folded into `memory.suppress`), so the Black Box
    /// can show the exact contradiction decision the agent faced.
    #[serde(rename = "contradiction.detected")]
    ContradictionDetected {
        #[serde(rename = "runId")]
        run_id: String,
        /// The two (or more) memory ids in tension.
        ids: Vec<String>,
        /// The id the agent trusted (kept), if it resolved the tension.
        #[serde(rename = "winnerId", skip_serializing_if = "Option::is_none")]
        winner_id: Option<String>,
        /// Plain-language description of the contradiction.
        detail: String,
        #[serde(default)]
        at: i64,
    },

    /// The Sanhedrin verifier vetoed a claim the agent was about to assert,
    /// citing the evidence it weighed and its confidence.
    #[serde(rename = "sanhedrin.veto")]
    SanhedrinVeto {
        #[serde(rename = "runId")]
        run_id: String,
        claim: String,
        #[serde(rename = "evidenceIds")]
        evidence_ids: Vec<String>,
        confidence: f64,
        #[serde(default)]
        at: i64,
    },

    /// Dream consolidation proposed a patch to memory (merge / insight / prune).
    #[serde(rename = "dream.patch")]
    DreamPatch {
        #[serde(rename = "runId")]
        run_id: String,
        #[serde(rename = "proposalIds")]
        proposal_ids: Vec<String>,
        at: i64,
    },
}

impl MemoryTraceEvent {
    /// The run this event belongs to.
    pub fn run_id(&self) -> &str {
        match self {
            MemoryTraceEvent::McpCall { run_id, .. }
            | MemoryTraceEvent::MemoryRetrieve { run_id, .. }
            | MemoryTraceEvent::MemorySuppress { run_id, .. }
            | MemoryTraceEvent::MemoryWrite { run_id, .. }
            | MemoryTraceEvent::ContradictionDetected { run_id, .. }
            | MemoryTraceEvent::SanhedrinVeto { run_id, .. }
            | MemoryTraceEvent::DreamPatch { run_id, .. } => run_id,
        }
    }

    /// The wall-clock millisecond timestamp the event was recorded at.
    pub fn at(&self) -> i64 {
        match self {
            MemoryTraceEvent::McpCall { at, .. }
            | MemoryTraceEvent::MemoryRetrieve { at, .. }
            | MemoryTraceEvent::MemorySuppress { at, .. }
            | MemoryTraceEvent::MemoryWrite { at, .. }
            | MemoryTraceEvent::ContradictionDetected { at, .. }
            | MemoryTraceEvent::SanhedrinVeto { at, .. }
            | MemoryTraceEvent::DreamPatch { at, .. } => *at,
        }
    }

    /// Short stable kind label used for filtering / the `event_type` column.
    pub fn kind(&self) -> &'static str {
        match self {
            MemoryTraceEvent::McpCall { .. } => "mcp.call",
            MemoryTraceEvent::MemoryRetrieve { .. } => "memory.retrieve",
            MemoryTraceEvent::MemorySuppress { .. } => "memory.suppress",
            MemoryTraceEvent::MemoryWrite { .. } => "memory.write",
            MemoryTraceEvent::ContradictionDetected { .. } => "contradiction.detected",
            MemoryTraceEvent::SanhedrinVeto { .. } => "sanhedrin.veto",
            MemoryTraceEvent::DreamPatch { .. } => "dream.patch",
        }
    }

    /// Stamp `at` on events that left it defaulted (the recorder fills this so
    /// callers don't have to thread a clock through every emit site).
    pub fn with_at(mut self, now_ms: i64) -> Self {
        match &mut self {
            MemoryTraceEvent::McpCall { at, .. }
            | MemoryTraceEvent::MemoryRetrieve { at, .. }
            | MemoryTraceEvent::MemorySuppress { at, .. }
            | MemoryTraceEvent::MemoryWrite { at, .. }
            | MemoryTraceEvent::ContradictionDetected { at, .. }
            | MemoryTraceEvent::SanhedrinVeto { at, .. }
            | MemoryTraceEvent::DreamPatch { at, .. } => {
                if *at == 0 {
                    *at = now_ms;
                }
            }
        }
        self
    }
}

/// Why a memory was suppressed during a run. Mirrors the TS union member
/// `"low_trust" | "decayed" | "contradicted" | "privacy"`, plus `competition`
/// for the existing spreading-activation competition suppression.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SuppressReason {
    /// Below the trust floor for this retrieval.
    LowTrust,
    /// FSRS retrievability decayed below the usable threshold.
    Decayed,
    /// Contradicted by a higher-trust memory.
    Contradicted,
    /// Withheld for privacy / sensitivity reasons.
    Privacy,
    /// Lost spreading-activation competition to a stronger memory.
    Competition,
}

impl SuppressReason {
    /// Stable string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            SuppressReason::LowTrust => "low_trust",
            SuppressReason::Decayed => "decayed",
            SuppressReason::Contradicted => "contradicted",
            SuppressReason::Privacy => "privacy",
            SuppressReason::Competition => "competition",
        }
    }
}

/// Who caused a `memory.write`. Mirrors the TS `"agent" | "user" | "dream"`,
/// plus `connector` for external-source sync writes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WriteSource {
    /// The agent wrote it autonomously.
    Agent,
    /// The user explicitly asked for it.
    User,
    /// Produced by dream consolidation.
    Dream,
    /// Ingested by an external connector (GitHub, Redmine, …).
    Connector,
}

impl WriteSource {
    /// Stable string label.
    pub fn as_str(&self) -> &'static str {
        match self {
            WriteSource::Agent => "agent",
            WriteSource::User => "user",
            WriteSource::Dream => "dream",
            WriteSource::Connector => "connector",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_event_roundtrips_with_ts_shape() {
        let ev = MemoryTraceEvent::McpCall {
            run_id: "run_123".into(),
            tool: "deep_reference".into(),
            args_hash: "abc".into(),
            at: 42,
        };
        let json = serde_json::to_value(&ev).unwrap();
        // Tagged on `type`, camelCase runId/argsHash — exactly the TS contract.
        assert_eq!(json["type"], "mcp.call");
        assert_eq!(json["runId"], "run_123");
        assert_eq!(json["argsHash"], "abc");
        assert_eq!(json["at"], 42);

        let back: MemoryTraceEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, ev);
    }

    #[test]
    fn retrieve_event_carries_activation_map() {
        let mut activation = std::collections::BTreeMap::new();
        activation.insert("mem_1".to_string(), 0.91);
        activation.insert("mem_7".to_string(), 0.42);
        let ev = MemoryTraceEvent::MemoryRetrieve {
            run_id: "r".into(),
            ids: vec!["mem_1".into(), "mem_7".into()],
            activation,
            at: 1,
        };
        let json = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["type"], "memory.retrieve");
        assert_eq!(json["activation"]["mem_1"], 0.91);
    }

    #[test]
    fn with_at_fills_only_when_unset() {
        let ev = MemoryTraceEvent::MemorySuppress {
            run_id: "r".into(),
            id: "m".into(),
            reason: SuppressReason::Contradicted,
            at: 0,
        }
        .with_at(999);
        assert_eq!(ev.at(), 999);

        let ev2 = MemoryTraceEvent::DreamPatch {
            run_id: "r".into(),
            proposal_ids: vec!["p".into()],
            at: 7,
        }
        .with_at(999);
        assert_eq!(ev2.at(), 7, "explicit timestamp must not be overwritten");
    }

    #[test]
    fn suppress_reason_labels_match_ts() {
        assert_eq!(SuppressReason::LowTrust.as_str(), "low_trust");
        assert_eq!(SuppressReason::Contradicted.as_str(), "contradicted");
        // Serde uses the same snake_case form on the wire.
        assert_eq!(
            serde_json::to_value(SuppressReason::Privacy).unwrap(),
            serde_json::json!("privacy")
        );
    }
}
