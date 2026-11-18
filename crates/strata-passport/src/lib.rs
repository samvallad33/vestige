//! # strata-passport — Tranche 1 Operating System Kernel (#252, #370, #382,
//! #396, #420, #495, #523, #524, #525, #559, #563, #566, #569, #570, #762)
//!
//! Pure, deterministic, zero-vector enforcement of:
//! - Offline-attenuable HMAC-BLAKE3 Subagent Passports & P2P namespace scopes (#523, #559, #762)
//! - Speaker & Actor Provenance, independent reinforcement, and verified supersession (#252, #525)
//! - Forged-reasoning execution-receipt verification (#566)
//! - Anchor Watch file-change staleness detector (#524) & `Calls` vs `Touched` symbol edges (#382)
//! - Fixed-epoch FSRS Brain Print (#495) & Schema Fast-Track initial stability (#563)
//! - Wave 0 explicit reason codes (#370), `SessionStart` lifecycle capsule (#396),
//!   `vestige-guard` claim verifier (#420), and July MCP stateless routing & Tasks (#569).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

// ============================================================================
// 1. Offline-Attenuable Subagent Passports & P2P Capability Scopes (#523, #559, #762)
// ============================================================================

/// First-party caveat attenuating a Subagent Passport offline without issuer contact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Caveat {
    /// Restrict subagent to an allowlist of MCP/shell tool names.
    AllowTools(BTreeSet<String>),
    /// Restrict filesystem reads/writes to exact directory prefixes.
    AllowPathPrefixes(BTreeSet<String>),
    /// Restrict P2P namespace replication to a capability-scoped prefix (#762).
    AllowNamespacePrefix(String),
    /// Hard expiration timestamp in milliseconds.
    ExpiresAtMs(u64),
    /// Read-only mode (forbids any state mutation).
    ReadOnly,
}

impl Caveat {
    fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("caveat serializes deterministically")
    }
}

/// Execution context checked against a [`SubagentPassport`].
#[derive(Debug, Clone)]
pub struct CapabilityContext {
    pub tool_name: String,
    pub target_path: Option<String>,
    pub p2p_namespace: Option<String>,
    pub now_ms: u64,
    pub is_mutation: bool,
}

/// Offline-attenuable Macaroon/Biscuit-style Subagent Passport (#523, #559, #762).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentPassport {
    pub passport_id: String,
    pub parent_session_id: String,
    pub child_agent_id: String,
    /// Exact memory handles injected into the subagent at spawn (#523).
    pub injected_handles: Vec<String>,
    /// Ordered caveats appended offline; removing or reordering breaks `signature_tag`.
    pub caveats: Vec<Caveat>,
    pub signature_tag: [u8; 32],
}

fn keyed_chain_step(key: &[u8; 32], msg: &[u8]) -> [u8; 32] {
    *blake3::keyed_hash(key, msg).as_bytes()
}

impl SubagentPassport {
    /// Mint a root Subagent Passport carrying exact memory handles (#523).
    pub fn issue(
        root_secret: &[u8; 32],
        passport_id: impl Into<String>,
        parent_session_id: impl Into<String>,
        child_agent_id: impl Into<String>,
        injected_handles: Vec<String>,
    ) -> Self {
        let passport_id = passport_id.into();
        let parent_session_id = parent_session_id.into();
        let child_agent_id = child_agent_id.into();
        let mut header = Vec::new();
        header.extend_from_slice(passport_id.as_bytes());
        header.push(b'|');
        header.extend_from_slice(parent_session_id.as_bytes());
        header.push(b'|');
        header.extend_from_slice(child_agent_id.as_bytes());
        for h in &injected_handles {
            header.push(b'|');
            header.extend_from_slice(h.as_bytes());
        }
        let signature_tag = keyed_chain_step(root_secret, &header);
        Self {
            passport_id,
            parent_session_id,
            child_agent_id,
            injected_handles,
            caveats: Vec::new(),
            signature_tag,
        }
    }

    /// Attenuate this passport offline by chaining a new caveat (#559, #762).
    /// Requires ONLY the current `signature_tag`, never `root_secret`.
    #[must_use]
    pub fn attenuate(&self, caveat: Caveat) -> Self {
        let mut next = self.clone();
        next.signature_tag = keyed_chain_step(&self.signature_tag, &caveat.canonical_bytes());
        next.caveats.push(caveat);
        next
    }

    /// Cryptographically verify the HMAC chain and evaluate all caveats.
    pub fn verify(
        &self,
        root_secret: &[u8; 32],
        ctx: &CapabilityContext,
    ) -> Result<(), &'static str> {
        let mut header = Vec::new();
        header.extend_from_slice(self.passport_id.as_bytes());
        header.push(b'|');
        header.extend_from_slice(self.parent_session_id.as_bytes());
        header.push(b'|');
        header.extend_from_slice(self.child_agent_id.as_bytes());
        for h in &self.injected_handles {
            header.push(b'|');
            header.extend_from_slice(h.as_bytes());
        }
        let mut tag = keyed_chain_step(root_secret, &header);
        for cav in &self.caveats {
            tag = keyed_chain_step(&tag, &cav.canonical_bytes());
        }
        if tag != self.signature_tag {
            return Err("PassportSignatureMismatch");
        }

        for cav in &self.caveats {
            match cav {
                Caveat::AllowTools(allowed) => {
                    if !allowed.contains(&ctx.tool_name) {
                        return Err("ToolDeniedByCaveat");
                    }
                }
                Caveat::AllowPathPrefixes(prefixes) => {
                    if let Some(path) = &ctx.target_path {
                        let norm = path.replace('\\', "/");
                        if !prefixes.iter().any(|p| norm.starts_with(p)) {
                            return Err("PathDeniedByCaveat");
                        }
                    }
                }
                Caveat::AllowNamespacePrefix(ns_prefix) => {
                    if let Some(ns) = &ctx.p2p_namespace
                        && !ns.starts_with(ns_prefix)
                    {
                        return Err("NamespaceDeniedByCaveat");
                    }
                }
                Caveat::ExpiresAtMs(exp) => {
                    if ctx.now_ms > *exp {
                        return Err("PassportExpired");
                    }
                }
                Caveat::ReadOnly => {
                    if ctx.is_mutation {
                        return Err("MutationDeniedByReadOnlyCaveat");
                    }
                }
            }
        }
        Ok(())
    }
}

/// Signed edge returned when a subagent completes and folds its trail back (#523).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildTrailEdge {
    pub passport_id: String,
    pub from_injected_handle: String,
    pub produced_handle: String,
    pub edge_hash: String,
}

pub fn fold_child_trail(
    passport: &SubagentPassport,
    produced_handles: &[String],
) -> Vec<ChildTrailEdge> {
    let mut edges = Vec::new();
    for src in &passport.injected_handles {
        for dst in produced_handles {
            let mut h = blake3::Hasher::new();
            h.update(passport.passport_id.as_bytes());
            h.update(b"->");
            h.update(src.as_bytes());
            h.update(b"->");
            h.update(dst.as_bytes());
            edges.push(ChildTrailEdge {
                passport_id: passport.passport_id.clone(),
                from_injected_handle: src.clone(),
                produced_handle: dst.clone(),
                edge_hash: h.finalize().to_hex().to_string(),
            });
        }
    }
    edges
}

// ============================================================================
// 2. Speaker & Actor Provenance + Forged Reasoning Defense (#252, #525, #566)
// ============================================================================

/// Who spoke the memory into existence (#525 & #252).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VoiceRole {
    Owner,
    Agent,
    TaintedSession,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub memory_id: String,
    pub actor_id: String,
    pub voice: VoiceRole,
    pub corroborating_actors: BTreeSet<String>,
}

impl ProvenanceRecord {
    pub fn new(memory_id: impl Into<String>, actor_id: impl Into<String>, voice: VoiceRole) -> Self {
        let aid = actor_id.into();
        let mut corroborating_actors = BTreeSet::new();
        if voice != VoiceRole::TaintedSession {
            corroborating_actors.insert(aid.clone());
        }
        Self {
            memory_id: memory_id.into(),
            actor_id: aid,
            voice,
            corroborating_actors,
        }
    }

    /// Independent reinforcement (#252): same actor or tainted session never inflates count.
    pub fn reinforce(&mut self, reinforcing_actor: &str, voice: VoiceRole) -> bool {
        if voice == VoiceRole::TaintedSession || reinforcing_actor == self.actor_id {
            return false;
        }
        self.corroborating_actors.insert(reinforcing_actor.to_string())
    }

    pub fn independent_count(&self) -> usize {
        self.corroborating_actors.len()
    }
}

/// Verified supersession (#252): non-owners cannot supersede an Owner memory without owner approval.
pub fn verify_supersession(
    existing: &ProvenanceRecord,
    challenger_voice: VoiceRole,
    owner_approved: bool,
) -> Result<(), &'static str> {
    if challenger_voice == VoiceRole::TaintedSession {
        return Err("TaintedSessionCannotSupersede");
    }
    if existing.voice == VoiceRole::Owner && challenger_voice != VoiceRole::Owner && !owner_approved {
        return Err("AgentCannotSupersedeOwnerWithoutApproval");
    }
    Ok(())
}

/// Forged reasoning defense (#566): completion claims require an execution receipt hash.
pub fn verify_completion_claim(
    claim_text: &str,
    execution_receipt_hash: Option<&[u8; 32]>,
) -> Result<bool, &'static str> {
    let low = claim_text.to_ascii_lowercase();
    let claims_done = ["completed", "fixed", "deployed", "merged", "executed", "resolved"]
        .iter()
        .any(|kw| low.contains(kw));
    if claims_done && execution_receipt_hash.is_none() {
        return Err("ForgedReasoningMissingExecutionReceipt");
    }
    Ok(claims_done)
}

// ============================================================================
// 3. Anchor Watch (#524) & Code Call Relation Edges (#382)
// ============================================================================

/// Explicit separation of code invocation (`Calls`) vs modification (`Touched`) (#382).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeRelationEdge {
    Calls {
        caller_symbol: String,
        callee_symbol: String,
    },
    Touched {
        commit_or_turn: String,
        file_path: String,
        symbol: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleAnchorAlert {
    pub decision_id: String,
    pub file_path: String,
    pub pinned_blake3: [u8; 32],
    pub observed_blake3: [u8; 32],
}

/// Anchor Watch (#524): triggers immediate re-verification when anchored files change.
#[derive(Debug, Default, Clone)]
pub struct AnchorWatchRegistry {
    watches: BTreeMap<String, Vec<(String, [u8; 32])>>,
}

impl AnchorWatchRegistry {
    pub fn watch(&mut self, decision_id: impl Into<String>, file_path: impl Into<String>, content: &[u8]) {
        let norm = file_path.into().replace('\\', "/");
        let hash = *blake3::hash(content).as_bytes();
        self.watches
            .entry(norm)
            .or_default()
            .push((decision_id.into(), hash));
    }

    pub fn on_file_changed(&self, file_path: &str, new_content: &[u8]) -> Vec<StaleAnchorAlert> {
        let norm = file_path.replace('\\', "/");
        let observed = *blake3::hash(new_content).as_bytes();
        let mut alerts = Vec::new();
        if let Some(entries) = self.watches.get(&norm) {
            for (decision_id, pinned) in entries {
                if *pinned != observed {
                    alerts.push(StaleAnchorAlert {
                        decision_id: decision_id.clone(),
                        file_path: norm.clone(),
                        pinned_blake3: *pinned,
                        observed_blake3: observed,
                    });
                }
            }
        }
        alerts
    }
}

// ============================================================================
// 4. Fixed-Epoch Brain Print (#495) & Schema Fast-Track (#563)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardSnapshot {
    pub memory_id: String,
    pub stability_milli: u64,
    pub difficulty_milli: u64,
    pub last_review_ms: u64,
}

/// Deterministic Brain Print (#495): evaluates purely against `store_watermark_ms`
/// (NEVER wall-clock `now()`), so an unchanged store never re-keys over time.
pub fn compute_brain_print(cards: &[CardSnapshot], store_watermark_ms: u64) -> [u8; 32] {
    let mut sorted = cards.to_vec();
    sorted.sort_by(|a, b| a.memory_id.cmp(&b.memory_id));
    let mut h = blake3::Hasher::new();
    h.update(b"strata-brain-print-v1:");
    h.update(&store_watermark_ms.to_le_bytes());
    for c in sorted {
        let elapsed_ms = store_watermark_ms.saturating_sub(c.last_review_ms);
        h.update(c.memory_id.as_bytes());
        h.update(&c.stability_milli.to_le_bytes());
        h.update(&c.difficulty_milli.to_le_bytes());
        h.update(&elapsed_ms.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

/// Schema Fast-Track (#563): new memories matching proven high-stability schemas
/// start with a deterministic stability boost (capped at 2.0x = 2000 permille).
pub fn schema_fast_track_stability(base_stability_milli: u64, proven_schema_matches: usize) -> u64 {
    let bonus_permille = (proven_schema_matches as u64 * 250).min(1000);
    base_stability_milli.saturating_mul(1000 + bonus_permille) / 1000
}

// ============================================================================
// 5. Wave 0 Reason Codes (#370), SessionStart Hooks (#396), Guard (#420) & July MCP (#569)
// ============================================================================

/// Explicit Wave 0 reason codes (#370) — never silent zeros.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrataReasonCode {
    UnavailableIn4_0,
    SimilarityDisabled,
    PendingStrata,
}

impl StrataReasonCode {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::UnavailableIn4_0 => "unavailable_in_4_0",
            Self::SimilarityDisabled => "similarity_disabled",
            Self::PendingStrata => "pending_strata",
        }
    }
}

/// `SessionStart` lifecycle events (#396).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStartKind {
    Startup,
    Resume,
    Clear,
    Compact,
    Fork,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionStartCapsule {
    pub kind: SessionStartKind,
    pub pinned_handles: Vec<String>,
    pub inherited_taint: bool,
}

pub fn build_session_start_capsule(
    kind: SessionStartKind,
    pinned_handles: Vec<String>,
    prior_taint: bool,
) -> SessionStartCapsule {
    let inherited_taint = match kind {
        SessionStartKind::Startup | SessionStartKind::Clear => false,
        SessionStartKind::Resume | SessionStartKind::Compact | SessionStartKind::Fork => prior_taint,
    };
    SessionStartCapsule {
        kind,
        pinned_handles,
        inherited_taint,
    }
}

/// Signed verdict returned by `vestige-guard` (#420).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardVerdictKind {
    Allow,
    Contradicted,
    Unproven,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardVerdict {
    pub verdict: GuardVerdictKind,
    pub cited_memories: Vec<String>,
    pub receipt_blake3: String,
}

pub fn check_claim_against_memories(
    claim: &str,
    active_facts: &[(String, String)],
    forbid_phrases: &[(String, String)],
) -> GuardVerdict {
    let low = claim.to_ascii_lowercase();
    let mut contradicted_by = Vec::new();
    for (mem_id, forbidden) in forbid_phrases {
        if low.contains(&forbidden.to_ascii_lowercase()) {
            contradicted_by.push(mem_id.clone());
        }
    }
    let (verdict, cited) = if !contradicted_by.is_empty() {
        (GuardVerdictKind::Contradicted, contradicted_by)
    } else {
        let mut supporting = Vec::new();
        for (mem_id, fact) in active_facts {
            if low.contains(&fact.to_ascii_lowercase()) {
                supporting.push(mem_id.clone());
            }
        }
        if supporting.is_empty() {
            (GuardVerdictKind::Unproven, Vec::new())
        } else {
            (GuardVerdictKind::Allow, supporting)
        }
    };
    let mut h = blake3::Hasher::new();
    h.update(claim.as_bytes());
    h.update(format!("{verdict:?}").as_bytes());
    for c in &cited {
        h.update(c.as_bytes());
    }
    GuardVerdict {
        verdict,
        cited_memories: cited,
        receipt_blake3: h.finalize().to_hex().to_string(),
    }
}

/// July MCP Specification stateless routing & async Tasks state machine (#569).
pub const MCP_JULY_PROTOCOL_VERSION: &str = "2025-07-01";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum McpTaskState {
    Working,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpTaskRecord {
    pub task_id: String,
    pub session_id: Option<String>,
    pub state: McpTaskState,
    pub result_json: Option<String>,
}

#[derive(Debug, Default)]
pub struct McpStatelessTaskRouter {
    tasks: BTreeMap<String, McpTaskRecord>,
}

impl McpStatelessTaskRouter {
    pub fn validate_headers(protocol_version: Option<&str>) -> Result<&'static str, &'static str> {
        match protocol_version {
            Some("2025-07-01") | Some("2025-03-26") | Some("2024-11-05") | None => {
                Ok(MCP_JULY_PROTOCOL_VERSION)
            }
            Some(_) => Err("UnsupportedMcpProtocolVersion"),
        }
    }

    pub fn create_task(&mut self, task_id: impl Into<String>, session_id: Option<String>) -> McpTaskRecord {
        let id = task_id.into();
        let rec = McpTaskRecord {
            task_id: id.clone(),
            session_id,
            state: McpTaskState::Working,
            result_json: None,
        };
        self.tasks.insert(id, rec.clone());
        rec
    }

    pub fn complete_task(&mut self, task_id: &str, result_json: impl Into<String>) -> Option<McpTaskRecord> {
        let rec = self.tasks.get_mut(task_id)?;
        if rec.state == McpTaskState::Working {
            rec.state = McpTaskState::Completed;
            rec.result_json = Some(result_json.into());
        }
        Some(rec.clone())
    }

    pub fn cancel_task(&mut self, task_id: &str) -> Option<McpTaskRecord> {
        let rec = self.tasks.get_mut(task_id)?;
        if rec.state == McpTaskState::Working {
            rec.state = McpTaskState::Cancelled;
        }
        Some(rec.clone())
    }

    pub fn get_task(&self, task_id: &str) -> Option<&McpTaskRecord> {
        self.tasks.get(task_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_523_559_762_offline_attenuable_subagent_passport_and_trail() {
        let secret = [0x42; 32];
        let root = SubagentPassport::issue(
            &secret,
            "pass-01",
            "sess-parent",
            "subagent-research",
            vec!["mem-rule-1".to_string()],
        );

        // Offline attenuation without root_secret!
        let attenuated = root
            .attenuate(Caveat::AllowTools(BTreeSet::from(["recall".to_string()])))
            .attenuate(Caveat::AllowNamespacePrefix("team/sec/".to_string()))
            .attenuate(Caveat::ExpiresAtMs(5_000))
            .attenuate(Caveat::ReadOnly);

        let ok_ctx = CapabilityContext {
            tool_name: "recall".to_string(),
            target_path: None,
            p2p_namespace: Some("team/sec/logs".to_string()),
            now_ms: 4_000,
            is_mutation: false,
        };
        assert_eq!(attenuated.verify(&secret, &ok_ctx), Ok(()));

        // Mutation blocked by ReadOnly caveat
        let mut bad_ctx = ok_ctx.clone();
        bad_ctx.is_mutation = true;
        assert_eq!(
            attenuated.verify(&secret, &bad_ctx),
            Err("MutationDeniedByReadOnlyCaveat")
        );

        // Tampering by stripping last caveat breaks HMAC signature chain
        let mut stripped = attenuated.clone();
        stripped.caveats.pop();
        assert_eq!(
            stripped.verify(&secret, &ok_ctx),
            Err("PassportSignatureMismatch")
        );

        // Trail folds back deterministically (#523)
        let edges = fold_child_trail(&attenuated, &["mem-finding-9".to_string()]);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from_injected_handle, "mem-rule-1");
    }

    #[test]
    fn issue_252_525_566_provenance_reinforcement_supersession_and_forged_reasoning() {
        let mut owner_rec = ProvenanceRecord::new("mem-1", "sam", VoiceRole::Owner);
        assert!(!owner_rec.reinforce("sam", VoiceRole::Owner));
        assert!(!owner_rec.reinforce("bot-tainted", VoiceRole::TaintedSession));
        assert!(owner_rec.reinforce("verifier-agent", VoiceRole::Agent));
        assert_eq!(owner_rec.independent_count(), 2);

        assert_eq!(
            verify_supersession(&owner_rec, VoiceRole::Agent, false),
            Err("AgentCannotSupersedeOwnerWithoutApproval")
        );
        assert_eq!(verify_supersession(&owner_rec, VoiceRole::Agent, true), Ok(()));

        // #566 forged reasoning check
        assert_eq!(
            verify_completion_claim("Auth bug fixed and deployed", None),
            Err("ForgedReasoningMissingExecutionReceipt")
        );
        let exec_hash = [0x11; 32];
        assert_eq!(
            verify_completion_claim("Auth bug fixed and deployed", Some(&exec_hash)),
            Ok(true)
        );
    }

    #[test]
    fn issue_382_524_anchor_watch_and_call_edges() {
        let call = CodeRelationEdge::Calls {
            caller_symbol: "auth::login".to_string(),
            callee_symbol: "token::verify".to_string(),
        };
        let touch = CodeRelationEdge::Touched {
            commit_or_turn: "87a1e6d7".to_string(),
            file_path: "src/auth.rs".to_string(),
            symbol: Some("auth::login".to_string()),
        };
        assert_ne!(call, touch);

        let mut reg = AnchorWatchRegistry::default();
        reg.watch("mem-dec-7", "src\\auth.rs", b"fn verify() -> bool { true }");
        assert!(
            reg.on_file_changed("src/auth.rs", b"fn verify() -> bool { true }")
                .is_empty()
        );
        let alerts = reg.on_file_changed("src/auth.rs", b"fn verify() -> bool { false }");
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].decision_id, "mem-dec-7");
    }

    #[test]
    fn issue_495_563_brain_print_time_invariance_and_schema_fast_track() {
        let cards = vec![CardSnapshot {
            memory_id: "mem-a".to_string(),
            stability_milli: 10_000,
            difficulty_milli: 5_000,
            last_review_ms: 1_700_000_000_000,
        }];
        let p1 = compute_brain_print(&cards, 1_700_086_400_000);
        // Evaluating hours later with an unchanged store watermark produces identical bytes!
        let p2 = compute_brain_print(&cards, 1_700_086_400_000);
        assert_eq!(p1, p2);

        // #563 schema fast-track
        assert_eq!(schema_fast_track_stability(1_000, 0), 1_000);
        assert_eq!(schema_fast_track_stability(1_000, 2), 1_500);
        assert_eq!(schema_fast_track_stability(1_000, 99), 2_000);
    }

    #[test]
    fn issue_370_396_420_569_lifecycle_guard_and_july_mcp_tasks() {
        assert_eq!(StrataReasonCode::UnavailableIn4_0.as_str(), "unavailable_in_4_0");

        let cap_compact = build_session_start_capsule(
            SessionStartKind::Compact,
            vec!["mem-pin".to_string()],
            true,
        );
        assert!(cap_compact.inherited_taint);
        let cap_clear = build_session_start_capsule(
            SessionStartKind::Clear,
            vec!["mem-pin".to_string()],
            true,
        );
        assert!(!cap_clear.inherited_taint);

        let verdict = check_claim_against_memories(
            "We erased the audit segment",
            &[("mem-1".to_string(), "append-only".to_string())],
            &[("mem-rule-append".to_string(), "erased the audit".to_string())],
        );
        assert_eq!(verdict.verdict, GuardVerdictKind::Contradicted);
        assert_eq!(verdict.cited_memories, vec!["mem-rule-append"]);

        let mut router = McpStatelessTaskRouter::default();
        assert_eq!(
            McpStatelessTaskRouter::validate_headers(Some("2025-07-01")),
            Ok("2025-07-01")
        );
        router.create_task("t-1", Some("sess-1".to_string()));
        let done = router.complete_task("t-1", r#"{"ok":true}"#).unwrap();
        assert_eq!(done.state, McpTaskState::Completed);
    }
}
