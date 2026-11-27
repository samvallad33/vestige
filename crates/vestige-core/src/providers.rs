//! Pluggable public provider boundaries for external guard & rerank extensions.
//!
//! The open-core AGPL-3.0 engine ships with strictly deterministic, zero-vector,
//! zero-LLM defaults:
//! - [`DeterministicRuleGuardProvider`] for [`SeedGuardProvider`]
//! - [`IdentityRerankProvider`] for [`RerankProvider`]
//!
//! External or commercial binaries may register custom in-process implementations
//! at startup via [`install_seed_guard_provider`] and [`install_rerank_provider`]
//! without modifying the deterministic retrieval or Strata kernel.

use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

/// Fixed-point unit `1.0` in `Q16.16` confidence representation.
pub const CONFIDENCE_Q16_ONE: u32 = 65_536;

/// Six-tier authority classification for tool-call & prompt governance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GuardAuthority {
    /// Interactive human owner at a local terminal.
    Owner,
    /// Explicit local operator policy / permit authority.
    Operator,
    /// Scoped repository / workspace automation.
    Workspace,
    /// Autonomous primary coding agent.
    #[default]
    Agent,
    /// Delegated child subagent with attenuated scope.
    Subagent,
    /// Untrusted external content (web, third-party tool output, webhook).
    UntrustedExternal,
}

impl GuardAuthority {
    /// Canonical lowercase snake_case wire string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Owner => "owner",
            Self::Operator => "operator",
            Self::Workspace => "workspace",
            Self::Agent => "agent",
            Self::Subagent => "subagent",
            Self::UntrustedExternal => "untrusted_external",
        }
    }
}

/// Action verdict returned by a [`SeedGuardProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardDecision {
    /// Action is permitted to proceed.
    Allow,
    /// Action requires explicit owner confirmation or secondary escalation.
    Ask,
    /// Action is blocked (fail-closed).
    Deny,
}

impl GuardDecision {
    /// Canonical lowercase wire string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Ask => "ask",
            Self::Deny => "deny",
        }
    }
}

/// Structured evaluation input passed to a [`SeedGuardProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GuardEvalRequest {
    /// Hook lifecycle event name (e.g., `"PreToolUse"`, `"UserPromptSubmit"`, `"Stop"`).
    pub event: String,
    /// Host runtime identifier (e.g., `"claude-code"`, `"antigravity"`, `"cursor"`, `"codex"`).
    pub runtime: String,
    /// Normalized tool name (e.g., `"Bash"`, `"Edit"`, `"Write"`, `"Read"`, `"mcp__vestige__remember"`).
    pub tool_name: String,
    /// Shell command string if this is a command execution tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Target filesystem path if this is a file read/write/edit tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    /// Canonical JSON payload of tool arguments or prompt metadata.
    #[serde(default)]
    pub payload_json: String,
    /// Current working directory of the agent session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Caller authority tier.
    #[serde(default)]
    pub authority: GuardAuthority,
    /// Active session or conversation identifier.
    #[serde(default)]
    pub session_id: String,
    /// Deterministic rule ID already matched by the fast static gate, if any (e.g., `"OP-011"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deterministic_rule_hit: Option<String>,
}

/// Structured verdict returned by a [`SeedGuardProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardEvalVerdict {
    /// Final governance decision (`allow`, `ask`, or `deny`).
    pub decision: GuardDecision,
    /// Rule identifier responsible for the verdict, if any (e.g., `"OP-011"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_id: Option<String>,
    /// Optional OWASP Agentic Security taxonomy code (`"ASI01"`..`"ASI10"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owasp_asi: Option<String>,
    /// Fixed-point `Q16.16` confidence in `0..=65536`.
    pub confidence_q16: u32,
    /// When `true`, signals a fast Tier-1 evaluator requesting Tier-2 deep reasoning escalation.
    pub escalate_to_reasoner: bool,
    /// Human-readable explanation suitable for host deny/ask messages and audit logs.
    pub reason: String,
    /// Identifier of the provider that produced this verdict.
    pub provider_id: String,
    /// Evaluation wall-clock duration in microseconds.
    pub latency_us: u64,
}

impl GuardEvalVerdict {
    /// Construct a deterministic allow verdict.
    pub fn allow(provider_id: impl Into<String>, latency_us: u64) -> Self {
        Self {
            decision: GuardDecision::Allow,
            rule_id: None,
            owasp_asi: None,
            confidence_q16: CONFIDENCE_Q16_ONE,
            escalate_to_reasoner: false,
            reason: String::new(),
            provider_id: provider_id.into(),
            latency_us,
        }
    }

    /// Construct a deterministic deny verdict with rule and reason.
    pub fn deny(
        provider_id: impl Into<String>,
        rule_id: impl Into<String>,
        owasp_asi: Option<String>,
        reason: impl Into<String>,
        latency_us: u64,
    ) -> Self {
        Self {
            decision: GuardDecision::Deny,
            rule_id: Some(rule_id.into()),
            owasp_asi,
            confidence_q16: CONFIDENCE_Q16_ONE,
            escalate_to_reasoner: false,
            reason: reason.into(),
            provider_id: provider_id.into(),
            latency_us,
        }
    }
}

/// Pluggable security guard evaluation trait for tool calls, prompts, and stop gates.
///
/// The open-core default is [`DeterministicRuleGuardProvider`]. Downstream builds
/// may install a custom in-process evaluator via [`install_seed_guard_provider`].
pub trait SeedGuardProvider: Send + Sync {
    /// Stable identifier of this guard provider (recorded in receipts/diagnostics).
    fn provider_id(&self) -> &str;

    /// Evaluate a hook event and return a deterministic or model-assisted verdict.
    fn evaluate(&self, req: &GuardEvalRequest) -> GuardEvalVerdict;

    /// Returns `true` if this provider includes a secondary deep-reasoning escalation stage.
    fn supports_deep_reasoning(&self) -> bool {
        false
    }
}

/// Default open-core guard provider: purely deterministic, zero LLM, zero network.
///
/// Respects any pre-matched `deterministic_rule_hit` by failing closed; otherwise
/// allows the request so the caller's deterministic reference monitor remains authoritative.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeterministicRuleGuardProvider;

impl SeedGuardProvider for DeterministicRuleGuardProvider {
    fn provider_id(&self) -> &str {
        "vestige-deterministic-rules"
    }

    fn evaluate(&self, req: &GuardEvalRequest) -> GuardEvalVerdict {
        if let Some(rule) = req.deterministic_rule_hit.as_deref() {
            return GuardEvalVerdict::deny(
                self.provider_id(),
                rule,
                None,
                format!("Blocked by deterministic rule {rule}."),
                0,
            );
        }
        GuardEvalVerdict::allow(self.provider_id(), 0)
    }
}

/// Candidate item supplied to a [`RerankProvider`] after first-stage deterministic retrieval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankCandidate {
    /// Canonical memory or turn identifier.
    pub id: String,
    /// Exact stored text (never mutated).
    pub text: String,
    /// Optional session identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional ISO-8601 timestamp or date string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Deterministic base score from `vestige_core::retrieval`.
    pub base_score: f64,
    /// Zero-based rank assigned by `vestige_core::retrieval`.
    pub base_rank: usize,
}

/// Structured reranking request passed to a [`RerankProvider`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankRequest {
    /// Raw query string.
    pub query: String,
    /// Optional reference date for temporal queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_date: Option<String>,
    /// Ranked candidate list from the deterministic retrieval engine.
    pub candidates: Vec<RerankCandidate>,
    /// Maximum number of ranked results requested.
    pub top_k: usize,
    /// When `true`, Slot 0 (`base_rank == 0`) MUST remain at index 0 in the output.
    pub lock_slot0: bool,
}

/// Scored candidate returned by a [`RerankProvider`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankScore {
    /// Canonical memory or turn identifier.
    pub id: String,
    /// Fixed-point `Q16.16` score for deterministic tie-breaking.
    pub score_q16: i64,
    /// Calibrated floating-point score for display/diagnostics.
    pub calibrated_score: f64,
    /// Optional short rationale when reasoning mode was invoked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

/// Reranked slate returned by a [`RerankProvider`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Ordered candidates, highest priority first, truncated to `top_k`.
    pub ranked: Vec<RerankScore>,
    /// Identifier of the provider that produced this ranking.
    pub provider_id: String,
    /// Reranking wall-clock duration in microseconds.
    pub latency_us: u64,
    /// Whether an optional reasoning pass was used for this query.
    pub used_reasoning: bool,
}

/// Pluggable post-retrieval reranking trait.
///
/// The open-core default is [`IdentityRerankProvider`], which preserves the
/// frozen zero-vector `vestige_core::retrieval` ordering bit-for-bit at `0 µs`.
pub trait RerankProvider: Send + Sync {
    /// Stable identifier of this rerank provider.
    fn provider_id(&self) -> &str;

    /// Rerank `req.candidates`, respecting `req.lock_slot0` and `req.top_k`.
    fn rerank(&self, req: &RerankRequest) -> RerankResponse;
}

/// Default open-core rerank provider: strict identity pass-through (`0 µs`, zero vectors, zero LLMs).
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityRerankProvider;

impl RerankProvider for IdentityRerankProvider {
    fn provider_id(&self) -> &str {
        "vestige-total-recall-identity"
    }

    fn rerank(&self, req: &RerankRequest) -> RerankResponse {
        let limit = if req.top_k == 0 {
            req.candidates.len()
        } else {
            req.top_k.min(req.candidates.len())
        };
        let ranked = req
            .candidates
            .iter()
            .take(limit)
            .map(|c| RerankScore {
                id: c.id.clone(),
                score_q16: (c.base_score * 65_536.0).round() as i64,
                calibrated_score: c.base_score,
                rationale: None,
            })
            .collect();
        RerankResponse {
            ranked,
            provider_id: self.provider_id().to_string(),
            latency_us: 0,
            used_reasoning: false,
        }
    }
}

/// Enforce `req.lock_slot0`: guarantees that when `lock_slot0` is set and a
/// candidate with `base_rank == 0` exists in `req.candidates`, it remains at
/// index `0` of `resp.ranked` without duplicating or dropping entries.
pub fn enforce_slot0_lock(req: &RerankRequest, resp: &mut RerankResponse) {
    if !req.lock_slot0 {
        return;
    }
    let Some(slot0) = req.candidates.iter().find(|c| c.base_rank == 0) else {
        return;
    };
    if let Some(pos) = resp.ranked.iter().position(|r| r.id == slot0.id) {
        if pos > 0 {
            let item = resp.ranked.remove(pos);
            resp.ranked.insert(0, item);
        }
    } else {
        let item = RerankScore {
            id: slot0.id.clone(),
            score_q16: (slot0.base_score * 65_536.0).round() as i64,
            calibrated_score: slot0.base_score,
            rationale: None,
        };
        resp.ranked.insert(0, item);
        if req.top_k > 0 && resp.ranked.len() > req.top_k {
            resp.ranked.truncate(req.top_k);
        }
    }
}

static SEED_GUARD_PROVIDER: OnceLock<Arc<dyn SeedGuardProvider>> = OnceLock::new();
static RERANK_PROVIDER: OnceLock<Arc<dyn RerankProvider>> = OnceLock::new();

/// Register a process-wide [`SeedGuardProvider`] once at startup.
///
/// Returns `Err(provider)` if a provider was already installed.
pub fn install_seed_guard_provider(
    provider: Arc<dyn SeedGuardProvider>,
) -> Result<(), Arc<dyn SeedGuardProvider>> {
    SEED_GUARD_PROVIDER.set(provider)
}

/// Return the active [`SeedGuardProvider`], defaulting to [`DeterministicRuleGuardProvider`].
pub fn active_seed_guard_provider() -> Arc<dyn SeedGuardProvider> {
    SEED_GUARD_PROVIDER
        .get()
        .cloned()
        .unwrap_or_else(|| Arc::new(DeterministicRuleGuardProvider))
}

/// Register a process-wide [`RerankProvider`] once at startup.
///
/// Returns `Err(provider)` if a provider was already installed.
pub fn install_rerank_provider(
    provider: Arc<dyn RerankProvider>,
) -> Result<(), Arc<dyn RerankProvider>> {
    RERANK_PROVIDER.set(provider)
}

/// Return the active [`RerankProvider`], defaulting to [`IdentityRerankProvider`].
pub fn active_rerank_provider() -> Arc<dyn RerankProvider> {
    RERANK_PROVIDER
        .get()
        .cloned()
        .unwrap_or_else(|| Arc::new(IdentityRerankProvider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_guard_allows_clean_and_denies_rule_hit() {
        let guard = DeterministicRuleGuardProvider;
        let clean = GuardEvalRequest {
            event: "PreToolUse".into(),
            runtime: "claude-code".into(),
            tool_name: "Bash".into(),
            command: Some("cargo test -p vestige-core".into()),
            authority: GuardAuthority::Agent,
            ..Default::default()
        };
        let v_ok = guard.evaluate(&clean);
        assert_eq!(v_ok.decision, GuardDecision::Allow);
        assert_eq!(v_ok.provider_id, "vestige-deterministic-rules");
        assert!(!guard.supports_deep_reasoning());

        let blocked = GuardEvalRequest {
            deterministic_rule_hit: Some("OP-011".into()),
            ..clean
        };
        let v_deny = guard.evaluate(&blocked);
        assert_eq!(v_deny.decision, GuardDecision::Deny);
        assert_eq!(v_deny.rule_id.as_deref(), Some("OP-011"));
        assert!(v_deny.reason.contains("OP-011"));
    }

    #[test]
    fn identity_reranker_preserves_exact_order_and_slot0_lock_holds() {
        let req = RerankRequest {
            query: "what database did we choose?".into(),
            query_date: None,
            candidates: vec![
                RerankCandidate {
                    id: "t0".into(),
                    text: "We switched to Strata append-only log.".into(),
                    session_id: Some("s1".into()),
                    timestamp: None,
                    base_score: 12.5,
                    base_rank: 0,
                },
                RerankCandidate {
                    id: "t1".into(),
                    text: "Earlier we discussed Postgres.".into(),
                    session_id: Some("s0".into()),
                    timestamp: None,
                    base_score: 9.0,
                    base_rank: 1,
                },
            ],
            top_k: 2,
            lock_slot0: true,
        };
        let resp = IdentityRerankProvider.rerank(&req);
        assert_eq!(resp.provider_id, "vestige-total-recall-identity");
        assert_eq!(resp.ranked[0].id, "t0");
        assert_eq!(resp.ranked[1].id, "t1");
        assert!(!resp.used_reasoning);

        // Simulate a custom reranker attempting to demote Slot 0 while lock_slot0 == true.
        let mut flipped = RerankResponse {
            ranked: vec![
                RerankScore {
                    id: "t1".into(),
                    score_q16: 65_536,
                    calibrated_score: 1.0,
                    rationale: None,
                },
                RerankScore {
                    id: "t0".into(),
                    score_q16: 32_768,
                    calibrated_score: 0.5,
                    rationale: None,
                },
            ],
            provider_id: "custom-test".into(),
            latency_us: 100,
            used_reasoning: true,
        };
        enforce_slot0_lock(&req, &mut flipped);
        assert_eq!(flipped.ranked[0].id, "t0");
        assert_eq!(flipped.ranked[1].id, "t1");
    }

    #[test]
    fn authority_and_decision_wire_strings_match_serde() {
        assert_eq!(GuardAuthority::UntrustedExternal.as_str(), "untrusted_external");
        assert_eq!(GuardDecision::Deny.as_str(), "deny");
        let json = serde_json::to_string(&GuardAuthority::Subagent).unwrap();
        assert_eq!(json, "\"subagent\"");
    }
}
