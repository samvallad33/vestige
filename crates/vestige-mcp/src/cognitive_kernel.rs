//! Closed-Loop Deterministic Cognitive Kernel for Vestige.
//!
//! Wires the 5 previously-isolated `strata-*` kernel crates (`strata-passport`,
//! `strata-runtime`, `strata-episodic`, `strata-causal-math`, `strata-simd`)
//! directly into live `vestige-mcp` execution and Reflex hooks:
//!
//! 1. **Strata Intake Compiler** (`compile_ingest_write`):
//!    - Chiral role & exclusive state-slot extraction (`ExclusiveSlotRegister`,
//!      `ChiralRoleBinding`), verified supersession (`VoiceRole`), automatic
//!      signed `supersedes` linking + conflict quarantine (`destructive_interference_admit`).
//!    - File/symbol BLAKE3 registration into `AnchorWatchRegistry`.
//!    - Native Strata 9-hour Synaptic Tagging & Capture (`STC`).
//! 2. **Autonomous Anchor Watch** (`sweep_anchor_watch`, `on_file_mutated`):
//!    - Detects silent codebase drift via BLAKE3 comparison, transitions
//!      affected memories into `Labile / Drifted` (`ReconsolidationGate`), and
//!      shunts their conductance inside `circuit_index`.
//! 3. **Multi-Agent Swarm OS** (`issue_subagent_passport`, `verify_subagent_tool`,
//!    `stage_subagent_write`, `fold_subagent_stop`, `compute_correction_ripple`):
//!    - Offline-attenuable HMAC-BLAKE3 `SubagentPassport` enforcement on `PreToolUse`.
//!    - Speculative `CognitiveBranchDag` + `CausalVectorClock` memory branches
//!      with proof-gated `guarded_merge` on `SubagentStop`.
//!    - Live cross-agent & same-session `correction_ripple_targets`.
//! 4. **Prefrontal Blast-Radius Simulator** (`simulate_edit_blast_radius`):
//!    - Runs before `edit_file`/`write_file`/`git commit` in `PreToolUse` and
//!      `predict`, surfacing anchored invariants, `Labile` drifted beliefs,
//!      Ochiai historical co-failure spectrum, and required test witnesses.
//! 5. **Output Operator** (`audit_stop_output`):
//!    - Enforces `verify_post_generation` (`OverruledCitationRejected` +
//!      `FalseAbstentionRejected`) and `#566` forged-completion / unwitnessed
//!      verification blocks on `Stop` and `SubagentStop`.
//! 6. **Sleep Law Inducer & Prospective Push** (`induce_standing_laws`,
//!    `push_due_intentions`, `drain_due_intentions`):
//!    - Galois Formal Concept Analysis + Ochiai failure spectrum + Titration
//!      promotion (`k >= 3`) turning repeated mistakes into provable candidate
//!      Operator/Rulebook laws overnight.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use strata_causal_math::wave15_ceiling::{ReflexPostVerdict, verify_post_generation};
use strata_causal_math::wave16_algebra::galois_attribute_closure;
use strata_episodic::wave8_physics::{
    ExclusiveSlotRegister, PacketAdmission, destructive_interference_admit,
};
use strata_episodic::wave9_chemistry::{ChiralRoleBinding, evaluate_titration_threshold};
use strata_passport::{
    AnchorWatchRegistry, CapabilityContext, Caveat, ChildTrailEdge, ProvenanceRecord,
    StaleAnchorAlert, SubagentPassport, VoiceRole, check_claim_against_memories, fold_child_trail,
    verify_completion_claim, verify_supersession,
};
use strata_runtime::analytics_w20_22::{ReconsolidationGate, ochiai_failure_spectrum};
use strata_runtime::distributed_w23::{
    CausalVectorClock, CognitiveBranchDag, correction_ripple_targets,
};
use vestige_core::{ConnectionRecord, SecretPolicy, Storage};

fn make_edge(source_id: &str, target_id: &str, link_type: &str, strength: f64) -> ConnectionRecord {
    let now = Utc::now();
    ConnectionRecord {
        source_id: source_id.to_string(),
        target_id: target_id.to_string(),
        link_type: link_type.to_string(),
        strength,
        activation_count: 1,
        created_at: now,
        last_activated: now,
    }
}

const SYNAPTIC_WINDOW_SECS: i64 = 9 * 3600;
const SYNAPTIC_MAX_CAPTURES: usize = 6;

/// Compiled write-time report returned in `remember(action="save")` responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntakeCompileReport {
    pub extracted_slots: Vec<String>,
    pub chiral_bindings: Vec<String>,
    pub auto_superseded_ids: Vec<String>,
    pub conflict_holds: Vec<String>,
    pub anchored_files: Vec<String>,
    pub synaptic_captured_handles: Vec<String>,
}

/// Pre-execution counterfactual simulation report surfaced before edits/commits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlastRadiusReport {
    pub target: String,
    pub anchored_memories: Vec<String>,
    pub labile_drifted_memories: Vec<String>,
    pub ochiai_risk_score: f64,
    pub historical_failures: Vec<String>,
    pub required_witnesses: Vec<String>,
    pub summary_line: Option<String>,
}

/// Provable candidate standing law synthesized overnight by `dream_compile` / `consolidate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InducedLawProposal {
    pub law_key: String,
    pub title: String,
    pub trigger_attributes: Vec<String>,
    pub witness_ids: Vec<String>,
    pub ochiai_score: f64,
    pub promoted_by_titration: bool,
    pub proof_blake3: String,
    pub saved_proposal_id: Option<String>,
}

/// Subagent passport + branch fold summary produced on `SubagentStop`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentFoldReport {
    pub agent: String,
    pub passport_id: String,
    pub trail_edges: Vec<ChildTrailEdge>,
    pub branch_merged: bool,
    pub merged_count: usize,
    pub merge_error: Option<String>,
    pub vector_clock: CausalVectorClock,
}

#[derive(Clone)]
struct SynapticTagEntry {
    id: String,
    scope: String,
    ts: DateTime<Utc>,
}

#[derive(Clone)]
struct LabileRecord {
    file_path: String,
    observed_at_seq: u64,
    alerted: bool,
}

struct KernelState {
    seq: u64,
    slots: ExclusiveSlotRegister,
    slot_provenance: HashMap<String, ProvenanceRecord>,
    anchor_watch: AnchorWatchRegistry,
    watched_paths: HashMap<String, PathBuf>,
    labile_anchors: HashMap<String, LabileRecord>,
    synaptic_tags: Vec<SynapticTagEntry>,
    passports: HashMap<String, SubagentPassport>,
    subagent_produced: HashMap<String, Vec<String>>,
    branch_dag: CognitiveBranchDag,
    vector_clock: CausalVectorClock,
    due_intentions: Vec<(String, String)>,
    announced_due: HashSet<String>,
}

impl Default for KernelState {
    fn default() -> Self {
        let mut branch_dag = CognitiveBranchDag::default();
        branch_dag.init_main(Vec::new());
        Self {
            seq: 0,
            slots: ExclusiveSlotRegister::default(),
            slot_provenance: HashMap::new(),
            anchor_watch: AnchorWatchRegistry::default(),
            watched_paths: HashMap::new(),
            labile_anchors: HashMap::new(),
            synaptic_tags: Vec::new(),
            passports: HashMap::new(),
            subagent_produced: HashMap::new(),
            branch_dag,
            vector_clock: CausalVectorClock::default(),
            due_intentions: Vec::new(),
            announced_due: HashSet::new(),
        }
    }
}

fn kernel_registry() -> &'static Mutex<HashMap<PathBuf, KernelState>> {
    static REG: OnceLock<Mutex<HashMap<PathBuf, KernelState>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn store_root_secret(storage: &Storage) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"vestige-subagent-passport-root-v1:");
    h.update(storage.data_dir().to_string_lossy().as_bytes());
    let key_path = vestige_core::actor::actor_key_path_for_data_dir(storage.data_dir());
    if let Ok(bytes) = std::fs::read(key_path) {
        h.update(&bytes);
    }
    *h.finalize().as_bytes()
}

/// Snapshot of currently drifted/labile memory IDs so `circuit_index` shunts their
/// sink conductance immediately.
pub fn labile_memory_ids(storage: &Storage) -> HashSet<String> {
    let dir = storage.data_dir().to_path_buf();
    let reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    reg.get(&dir)
        .map(|k| k.labile_anchors.keys().cloned().collect())
        .unwrap_or_default()
}

/// Extract canonical `(subject, slot_key, value)` + `ChiralRoleBinding` from a memory text
/// deterministically (zero LLM, pure structural patterns).
pub fn extract_exclusive_slots(content: &str) -> Vec<(String, String, String, ChiralRoleBinding)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.len() > 280 {
            continue;
        }
        // Pattern 1: `<subject>.<attr> = <value>` or `<KEY> = <value>`
        if let Some((lhs, rhs)) = line.split_once('=').or_else(|| line.split_once(" is now ")) {
            let lhs_clean = lhs
                .trim()
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_' && c != '-')
                .to_ascii_lowercase();
            let rhs_clean = rhs
                .trim()
                .trim_end_matches(['.', ';', ','])
                .to_string();
            if !lhs_clean.is_empty()
                && lhs_clean.len() <= 64
                && lhs_clean.split_whitespace().count() <= 3
                && !rhs_clean.is_empty()
                && rhs_clean.len() <= 120
            {
                let (subj, attr) = match lhs_clean.split_once('.') {
                    Some((s, a)) => (s.to_string(), a.to_string()),
                    None => {
                        let parts: Vec<&str> = lhs_clean.split_whitespace().collect();
                        if parts.len() >= 2 {
                            (parts[0].to_string(), parts[1..].join("_"))
                        } else {
                            ("config".to_string(), lhs_clean.clone())
                        }
                    }
                };
                let key = format!("{subj}:{attr}");
                if seen.insert(key) {
                    out.push((
                        subj.clone(),
                        attr.clone(),
                        rhs_clean.clone(),
                        ChiralRoleBinding {
                            subject: subj,
                            predicate: format!("sets:{attr}"),
                            object: rhs_clean,
                        },
                    ));
                }
            }
        }
        // Pattern 2: `<subject> changed/moved/switched from <old> to <new>`
        let low = line.to_ascii_lowercase();
        for verb in [" moved from ", " changed from ", " switched from "] {
            if let Some(pos) = low.find(verb) {
                let subj_raw = line[..pos].trim();
                let tail = &line[pos + verb.len()..];
                if let Some(to_pos) = tail.to_ascii_lowercase().find(" to ") {
                    let new_val = tail[to_pos + 4..]
                        .trim()
                        .trim_end_matches(['.', ';', ','])
                        .to_string();
                    let words: Vec<&str> = subj_raw
                        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
                        .filter(|w| w.len() >= 2)
                        .collect();
                    if !words.is_empty() && !new_val.is_empty() {
                        let subj = words[0].to_ascii_lowercase();
                        let attr = if words.len() >= 2 {
                            words[1..].join("_").to_ascii_lowercase()
                        } else {
                            "value".to_string()
                        };
                        let key = format!("{subj}:{attr}");
                        if seen.insert(key) {
                            out.push((
                                subj.clone(),
                                attr.clone(),
                                new_val.clone(),
                                ChiralRoleBinding {
                                    subject: subj,
                                    predicate: format!("transitioned:{attr}"),
                                    object: new_val,
                                },
                            ));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Structured write-time request passed to [`compile_ingest_write`].
pub struct IntakeWriteRequest<'a> {
    pub node_id: &'a str,
    pub content: &'a str,
    pub node_type: &'a str,
    pub tags: &'a [String],
    pub scope: &'a str,
    pub repo_path: Option<&'a str>,
    pub files: &'a [String],
}

/// Step 3A: Write-Time Deterministic Strata Intake Compiler.
/// Runs automatically inside `remember(action="save")` so every write gets:
/// - Exclusive state-slot extraction + verified auto-supersession or conflict quarantine
/// - File/symbol BLAKE3 Anchor Watch registration
/// - Native Strata 9-hour Synaptic Tagging & Capture (`STC`)
/// - Incremental `ZSet` delta notification for `circuit_index`
pub fn compile_ingest_write(storage: &Storage, req: &IntakeWriteRequest<'_>) -> IntakeCompileReport {
    let IntakeWriteRequest {
        node_id,
        content,
        node_type,
        tags,
        scope,
        repo_path,
        files,
    } = *req;
    let dir = storage.data_dir().to_path_buf();
    let mut reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let k = reg.entry(dir).or_default();
    k.seq += 1;

    let is_owner = tags
        .iter()
        .any(|t| t == "pinned" || t == "owner-directive" || t == "owner");
    let voice = if is_owner {
        VoiceRole::Owner
    } else {
        VoiceRole::Agent
    };

    let mut extracted_slots = Vec::new();
    let mut chiral_bindings = Vec::new();
    let mut auto_superseded_ids = Vec::new();
    let mut conflict_holds = Vec::new();

    // 1. Exclusive state slots + Chiral role bindings + verified supersession.
    for (subj, attr, val, binding) in extract_exclusive_slots(content) {
        let slot_label = format!("slot:{subj}:{attr}");
        extracted_slots.push(slot_label.clone());
        chiral_bindings.push(format!(
            "{} --[{}]--> {}",
            binding.subject, binding.predicate, binding.object
        ));

        let existing_id = k.slots.current(&subj, &attr).map(str::to_string);
        if let Some(prev_id) = existing_id
            && prev_id != node_id
        {
            let prev_prov = k
                .slot_provenance
                .get(&prev_id)
                .cloned()
                .unwrap_or_else(|| ProvenanceRecord::new(prev_id.clone(), "store", VoiceRole::Agent));
            match verify_supersession(&prev_prov, voice, is_owner) {
                Ok(()) => {
                    k.slots.assign(&subj, &attr, node_id);
                    k.slot_provenance.insert(
                        node_id.to_string(),
                        ProvenanceRecord::new(node_id, scope, voice),
                    );
                    let edge = make_edge(node_id, &prev_id, "supersedes", 1.0);
                    if storage.save_connection(&edge).is_ok() {
                        auto_superseded_ids.push(prev_id);
                    }
                }
                Err(reason) => {
                    let admit = destructive_interference_admit(&[
                        (prev_id.clone(), true),
                        (node_id.to_string(), false),
                    ]);
                    if matches!(admit, Some(PacketAdmission::HoldConflict { .. })) {
                        conflict_holds.push(format!("{slot_label}:{prev_id}({reason})"));
                        let edge = make_edge(node_id, &prev_id, "corrects", 0.9);
                        let _ = storage.save_connection(&edge);
                    }
                }
            }
        } else {
            k.slots.assign(&subj, &attr, node_id);
            k.slot_provenance.insert(
                node_id.to_string(),
                ProvenanceRecord::new(node_id, scope, voice),
            );
        }
        let _ = val;
    }

    // 2. File/symbol BLAKE3 Anchor Watch registration (`#524`).
    let mut candidate_files: Vec<String> = files.to_vec();
    for tag in tags {
        if let Some(f) = tag.strip_prefix("file:") {
            candidate_files.push(f.to_string());
        }
    }
    let mut anchored_files = Vec::new();
    let mut seen_anchors = HashSet::new();
    let base_dir = repo_path
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok());
    for raw_file in candidate_files {
        let clean = raw_file.split('#').next().unwrap_or(&raw_file).trim();
        if clean.is_empty() {
            continue;
        }
        let norm = clean.replace('\\', "/");
        if !seen_anchors.insert(norm.clone()) {
            continue;
        }
        let full_path = if Path::new(clean).is_absolute() {
            PathBuf::from(clean)
        } else if let Some(ref base) = base_dir {
            base.join(clean)
        } else {
            PathBuf::from(clean)
        };
        if let Ok(bytes) = std::fs::read(&full_path) {
            k.anchor_watch.watch(node_id, norm.clone(), &bytes);
            k.watched_paths.insert(norm.clone(), full_path);
            // Fresh write re-attests this anchor: clear any prior Labile state on this node.
            k.labile_anchors.remove(node_id);
            anchored_files.push(norm);
        }
    }

    // 3. Native Strata 9-Hour Synaptic Tagging & Capture (`STC`).
    let now = Utc::now();
    k.synaptic_tags
        .retain(|entry| (now - entry.ts).num_seconds() <= SYNAPTIC_WINDOW_SECS);
    let is_prp_shock = is_owner
        || node_type.eq_ignore_ascii_case("decision")
        || tags.iter().any(|t| {
            matches!(
                t.as_str(),
                "decision" | "directive" | "proven" | "closes" | "root-cause" | "breakthrough"
            )
        });
    let mut synaptic_captured_handles = Vec::new();
    if is_prp_shock {
        let targets: Vec<String> = k
            .synaptic_tags
            .iter()
            .rev()
            .filter(|e| e.scope == scope && e.id != node_id)
            .map(|e| e.id.clone())
            .take(SYNAPTIC_MAX_CAPTURES)
            .collect();
        for target_id in targets {
            let edge = make_edge(node_id, &target_id, "derived_from", 0.72);
            if storage.save_connection(&edge).is_ok() {
                synaptic_captured_handles.push(target_id);
            }
        }
    }
    k.synaptic_tags.push(SynapticTagEntry {
        id: node_id.to_string(),
        scope: scope.to_string(),
        ts: now,
    });
    if k.synaptic_tags.len() > 256 {
        let drop = k.synaptic_tags.len() - 256;
        k.synaptic_tags.drain(..drop);
    }

    // Also update main branch DAG + causal clock.
    k.branch_dag.append("main", node_id.to_string());
    k.vector_clock.tick("main");

    drop(reg);
    crate::circuit_index::mark_zset_delta(storage, node_id, 1);

    IntakeCompileReport {
        extracted_slots,
        chiral_bindings,
        auto_superseded_ids,
        conflict_holds,
        anchored_files,
        synaptic_captured_handles,
    }
}

/// Step 3B: Autonomous Anchor Watch (`#524`) + Reconsolidation Gate (`#558`).
/// Scans watched files (or one mutated file) on disk against pinned BLAKE3 hashes.
/// Drifted memories immediately enter `Labile` state (shunting their circuit conductance)
/// and emit a crisp Reflex ledger line once.
pub fn sweep_anchor_watch(storage: &Storage, touched_file: Option<&str>) -> Vec<StaleAnchorAlert> {
    let dir = storage.data_dir().to_path_buf();
    let mut reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let k = reg.entry(dir).or_default();
    k.seq += 1;
    let seq_now = k.seq;

    let targets: Vec<(String, PathBuf)> = match touched_file {
        Some(f) if !f.trim().is_empty() => {
            let norm = f.replace('\\', "/");
            k.watched_paths
                .iter()
                .filter(|(k_path, full)| {
                    *k_path == &norm
                        || norm.ends_with(k_path.as_str())
                        || full.to_string_lossy().replace('\\', "/").ends_with(&norm)
                })
                .map(|(k_path, full)| (k_path.clone(), full.clone()))
                .collect()
        }
        _ => k
            .watched_paths
            .iter()
            .map(|(k_path, full)| (k_path.clone(), full.clone()))
            .collect(),
    };

    let mut fresh_alerts = Vec::new();
    for (norm, full_path) in targets {
        let Ok(bytes) = std::fs::read(&full_path) else {
            continue;
        };
        let alerts = k.anchor_watch.on_file_changed(&norm, &bytes);
        for alert in alerts {
            let entry = k
                .labile_anchors
                .entry(alert.decision_id.clone())
                .or_insert_with(|| LabileRecord {
                    file_path: alert.file_path.clone(),
                    observed_at_seq: seq_now,
                    alerted: false,
                });
            let recon_gate = ReconsolidationGate {
                recalled_at_seq: Some(entry.observed_at_seq),
                failure_after_recall: true,
                window_seqs: 64,
            };
            if !entry.alerted && recon_gate.may_reconsolidate(seq_now) {
                entry.alerted = true;
                fresh_alerts.push(alert);
            }
        }
    }

    let drifted_ids: Vec<String> = fresh_alerts.iter().map(|a| a.decision_id.clone()).collect();
    drop(reg);
    for id in drifted_ids {
        crate::circuit_index::mark_zset_delta(storage, &id, -1);
    }
    fresh_alerts
}

/// Format fresh Anchor Watch drift alerts as concise Reflex ledger notices.
pub fn anchor_watch_lines(storage: &Storage, touched_file: Option<&str>) -> Vec<String> {
    let alerts = sweep_anchor_watch(storage, touched_file);
    alerts
        .into_iter()
        .take(4)
        .map(|a| {
            format!(
                "Anchor Watch: `{}` changed on disk without a fresh receipt; anchored memory {} is now Labile (shunted in GhostCurrent). Re-verify or supersede with curate/remember.",
                a.file_path, a.decision_id
            )
        })
        .collect()
}

/// Step 4A: Mint an offline-attenuable HMAC-BLAKE3 `SubagentPassport` + fork an isolated
/// speculative branch `branch:<agent>` in `CognitiveBranchDag` on `SubagentStart`.
pub fn issue_subagent_passport(
    storage: &Storage,
    parent_session: &str,
    child_agent: &str,
    injected_handles: Vec<String>,
    tool_input: &Value,
) -> (SubagentPassport, String) {
    let secret = store_root_secret(storage);
    let passport_id = format!(
        "pass-{}-{}",
        parent_session.chars().take(8).collect::<String>(),
        child_agent.chars().take(12).collect::<String>()
    );
    let mut passport = SubagentPassport::issue(
        &secret,
        passport_id.clone(),
        parent_session,
        child_agent,
        injected_handles,
    );

    // Apply requested or role-derived offline caveats.
    if let Some(arr) = tool_input.get("allow_tools").and_then(Value::as_array) {
        let set: BTreeSet<String> = arr
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        if !set.is_empty() {
            passport = passport.attenuate(Caveat::AllowTools(set));
        }
    }
    if let Some(arr) = tool_input.get("allow_path_prefixes").and_then(Value::as_array) {
        let set: BTreeSet<String> = arr
            .iter()
            .filter_map(Value::as_str)
            .map(|s| s.replace('\\', "/"))
            .collect();
        if !set.is_empty() {
            passport = passport.attenuate(Caveat::AllowPathPrefixes(set));
        }
    }
    if let Some(ns) = tool_input.get("allow_namespace").and_then(Value::as_str)
        && !ns.trim().is_empty()
    {
        passport = passport.attenuate(Caveat::AllowNamespacePrefix(ns.trim().to_string()));
    }
    if matches!(tool_input.get("read_only"), Some(Value::Bool(true)))
        || tool_input
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|r| {
                let low = r.to_ascii_lowercase();
                low.contains("research") || low.contains("read-only") || low.contains("readonly")
            })
    {
        passport = passport.attenuate(Caveat::ReadOnly);
    }
    if let Some(ttl_secs) = tool_input.get("ttl_secs").and_then(Value::as_u64) {
        let exp = (Utc::now().timestamp_millis().max(0) as u64)
            .saturating_add(ttl_secs.saturating_mul(1000));
        passport = passport.attenuate(Caveat::ExpiresAtMs(exp));
    }

    let branch_name = format!("branch:{child_agent}");
    let dir = storage.data_dir().to_path_buf();
    let mut reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let k = reg.entry(dir).or_default();
    k.passports.insert(child_agent.to_string(), passport.clone());
    k.subagent_produced.entry(child_agent.to_string()).or_default();
    k.branch_dag.fork("main", &branch_name);
    k.vector_clock.tick(child_agent);

    (passport, branch_name)
}

/// Step 4B: Cryptographically enforce `SubagentPassport` on every `PreToolUse` from a subagent.
pub fn verify_subagent_tool(
    storage: &Storage,
    child_agent: &str,
    tool_name: &str,
    target_path: Option<&str>,
    namespace: Option<&str>,
    is_mutation: bool,
) -> Result<(), String> {
    if child_agent.is_empty() {
        return Ok(());
    }
    let dir = storage.data_dir().to_path_buf();
    let reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let Some(passport) = reg.get(&dir).and_then(|k| k.passports.get(child_agent)) else {
        return Ok(());
    };
    let secret = store_root_secret(storage);
    let ctx = CapabilityContext {
        tool_name: tool_name.to_string(),
        target_path: target_path.map(|s| s.replace('\\', "/")),
        p2p_namespace: namespace.map(str::to_string),
        now_ms: Utc::now().timestamp_millis().max(0) as u64,
        is_mutation,
    };
    passport.verify(&secret, &ctx).map_err(|code| {
        format!(
            "SubagentPassport `{}` denied `{tool_name}` for agent `{child_agent}`: {code}",
            passport.passport_id
        )
    })
}

/// Record a speculative memory ID produced by a subagent onto its `CognitiveBranchDag` branch.
pub fn stage_subagent_write(storage: &Storage, child_agent: &str, mem_id: &str) {
    if child_agent.is_empty() || mem_id.is_empty() {
        return;
    }
    let dir = storage.data_dir().to_path_buf();
    let mut reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let k = reg.entry(dir).or_default();
    let branch_name = format!("branch:{child_agent}");
    k.branch_dag.append(&branch_name, mem_id.to_string());
    k.subagent_produced
        .entry(child_agent.to_string())
        .or_default()
        .push(mem_id.to_string());
    k.vector_clock.tick(child_agent);
}

/// Step 4C: Fold a finishing subagent's signed child trail and execute `guarded_merge`
/// from `branch:<agent>` into `main` on `SubagentStop`.
pub fn fold_subagent_stop(storage: &Storage, child_agent: &str) -> Option<SubagentFoldReport> {
    if child_agent.is_empty() {
        return None;
    }
    let dir = storage.data_dir().to_path_buf();
    let mut reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let k = reg.entry(dir).or_default();
    let passport = k.passports.remove(child_agent)?;
    let produced = k.subagent_produced.remove(child_agent).unwrap_or_default();
    let trail_edges = fold_child_trail(&passport, &produced);
    let branch_name = format!("branch:{child_agent}");
    let (branch_merged, merged_count, merge_error) =
        match k.branch_dag.guarded_merge(&branch_name, "main", "UNVERIFIED:") {
            Ok(n) => (true, n, None),
            Err(e) => (false, 0, Some(e.to_string())),
        };
    k.vector_clock.tick("main");
    Some(SubagentFoldReport {
        agent: child_agent.to_string(),
        passport_id: passport.passport_id,
        trail_edges,
        branch_merged,
        merged_count,
        merge_error,
        vector_clock: k.vector_clock.clone(),
    })
}

/// Step 4D: Compute exact active agent/session targets holding a superseded memory ID
/// via `strata_runtime::distributed_w23::correction_ripple_targets`.
pub fn compute_correction_ripple(
    retired_id: &str,
    manifests: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<String> {
    correction_ripple_targets(retired_id, manifests)
}

/// Step 5: Prefrontal Counterfactual Blast-Radius Simulator.
/// Runs in `< 3 ms` inside `PreToolUse` (before file edits / `git commit`) AND in `predict`.
pub fn simulate_edit_blast_radius(
    storage: &Storage,
    target_file: &str,
    cwd: Option<&Path>,
) -> BlastRadiusReport {
    let norm = target_file.replace('\\', "/");
    let leaf = norm.rsplit('/').next().unwrap_or(&norm);
    let nodes = storage.get_all_nodes(i32::MAX, 0).unwrap_or_default();
    let retired = storage.superseded_node_ids().unwrap_or_default();
    let labile = labile_memory_ids(storage);

    let mut anchored_memories = Vec::new();
    let mut labile_drifted_memories = Vec::new();
    let mut historical_failures = Vec::new();

    // Ochiai co-failure counts: (failed_with_file, passed_with_file, failed_without_file)
    let (mut ef, mut ep, mut nf) = (0usize, 0usize, 0usize);

    for node in &nodes {
        if retired.contains(&node.id) {
            continue;
        }
        let touches_file = node.tags.iter().any(|t| {
            let t_norm = t.replace('\\', "/");
            t_norm == format!("file:{norm}")
                || (!leaf.is_empty() && t_norm.ends_with(leaf))
        }) || (!leaf.is_empty() && leaf.len() >= 5 && node.content.contains(leaf));

        let is_failure = node.tags.iter().any(|t| {
            t == "operator-stop" || t == "failure" || t == "bug" || t == "regression"
        }) || node.node_type == "event";

        if touches_file {
            if is_failure {
                ef += 1;
                historical_failures.push(node.id.clone());
            } else {
                ep += 1;
                anchored_memories.push(node.id.clone());
            }
            if labile.contains(&node.id) {
                labile_drifted_memories.push(node.id.clone());
            }
        } else if is_failure {
            nf += 1;
        }
    }

    let ochiai_risk_score = if ef == 0 {
        0.0
    } else {
        ochiai_failure_spectrum(ef as u64, nf.min(64) as u64, ep as u64)
    };

    let required_witnesses: Vec<String> = {
        let cfg = crate::tools::reflex::config_for(storage.data_dir(), cwd);
        cfg.commands.into_iter().take(4).collect()
    };

    anchored_memories.truncate(6);
    historical_failures.truncate(4);
    labile_drifted_memories.truncate(4);

    let summary_line = if anchored_memories.is_empty()
        && historical_failures.is_empty()
        && labile_drifted_memories.is_empty()
    {
        None
    } else {
        let mut parts = Vec::new();
        if !anchored_memories.is_empty() {
            parts.push(format!("anchored=[{}]", anchored_memories.join(", ")));
        }
        if !labile_drifted_memories.is_empty() {
            parts.push(format!("labile_drifted=[{}]", labile_drifted_memories.join(", ")));
        }
        if !historical_failures.is_empty() {
            parts.push(format!(
                "prior_failures=[{}] (ochiai={:.2})",
                historical_failures.join(", "),
                ochiai_risk_score
            ));
        }
        if !required_witnesses.is_empty() {
            parts.push(format!("witnesses=[{}]", required_witnesses.join(" | ")));
        }
        Some(format!(
            "Prefrontal Blast-Radius (`{norm}`): {}.",
            parts.join("; ")
        ))
    };

    BlastRadiusReport {
        target: norm,
        anchored_memories,
        labile_drifted_memories,
        ochiai_risk_score,
        historical_failures,
        required_witnesses,
        summary_line,
    }
}

/// Step 6: Output Operator (`Stop` / `SubagentStop` Post-Generation Verification Gate).
/// Enforces:
/// 1. Overruled / superseded citation rejection (`verify_post_generation`)
/// 2. False abstention rejection when grounded memories are already held (`verify_post_generation`)
/// 3. Forged verification / unwitnessed test-pass claims after code edits (`verify_completion_claim`)
/// 4. Contradiction against pinned `forbid` laws (`check_claim_against_memories`)
pub fn audit_stop_output(
    storage: &Storage,
    draft_answer: &str,
    held_ids: &HashSet<String>,
    edited_files: &[String],
    passed_gates: &[String],
    last_prompt: &str,
) -> Vec<String> {
    if draft_answer.trim().is_empty() {
        return Vec::new();
    }
    let mut reasons = Vec::new();

    let superseded: BTreeSet<String> = storage
        .superseded_node_ids()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let labile: BTreeSet<String> = labile_memory_ids(storage).into_iter().collect();
    let mut overruled = superseded;
    overruled.extend(labile);

    // Check if the prompt's terms are already covered by the agent's held manifest.
    let prompt_terms = crate::circuit_index::tokenize_query_terms(last_prompt);
    let premises_covered = !held_ids.is_empty()
        && !prompt_terms.is_empty()
        && held_ids.iter().any(|id| {
            storage.get_node(id).ok().flatten().is_some_and(|n| {
                let low = n.content.to_ascii_lowercase();
                prompt_terms.iter().any(|t| low.contains(t))
            })
        });

    match verify_post_generation(draft_answer, premises_covered, &overruled) {
        ReflexPostVerdict::Valid => {}
        ReflexPostVerdict::OverruledCitationRejected(id) => {
            reasons.push(format!(
                "Output Operator blocked: cited overruled/labile memory {id} as active truth."
            ));
        }
        ReflexPostVerdict::FalseAbstentionRejected => {
            let sample: Vec<String> = held_ids.iter().take(4).cloned().collect();
            reasons.push(format!(
                "Output Operator blocked: false abstention while grounded memories [{}] are already held in context.",
                sample.join(", ")
            ));
        }
    }

    // Forged verification / unwitnessed completion defense (#566):
    // If files were edited this turn and assistant claims tests/build passed with zero witness gates.
    let low = draft_answer.to_ascii_lowercase();
    let claims_verified_pass = [
        "all tests pass",
        "tests passed",
        "cargo test passed",
        "clippy clean",
        "0 warnings",
        "verified passing",
    ]
    .iter()
    .any(|p| low.contains(p));
    if !edited_files.is_empty() && claims_verified_pass && passed_gates.is_empty() {
        let receipt_hash: Option<[u8; 32]> = None;
        if let Err(code) = verify_completion_claim("tests completed and fixed", receipt_hash.as_ref()) {
            reasons.push(format!(
                "Output Operator blocked ({code}): claimed passing tests after editing [{}] with zero passing witness gates this turn.",
                edited_files.join(", ")
            ));
        }
    }

    // Guard check against pinned explicit forbid rules (`forbid:<phrase>` tags).
    let mut forbid_phrases: Vec<(String, String)> = Vec::new();
    for id in held_ids.iter().take(24) {
        if let Ok(Some(node)) = storage.get_node(id) {
            for tag in &node.tags {
                if let Some(phrase) = tag.strip_prefix("forbid:")
                    && !phrase.trim().is_empty()
                {
                    forbid_phrases.push((node.id.clone(), phrase.trim().to_string()));
                }
            }
        }
    }
    if !forbid_phrases.is_empty() {
        let verdict = check_claim_against_memories(draft_answer, &[], &forbid_phrases);
        if verdict.verdict == strata_passport::GuardVerdictKind::Contradicted {
            reasons.push(format!(
                "Output Operator blocked: output violated pinned forbid rule(s) [{}] (receipt {}).",
                verdict.cited_memories.join(", "),
                &verdict.receipt_blake3[..16]
            ));
        }
    }

    reasons
}

/// Step 7A: Prospective Memory Live Push Queue (bridging Autopilot timer -> Reflex hooks).
pub fn push_due_intentions(storage: &Storage, due_items: &[(String, String)]) {
    if due_items.is_empty() {
        return;
    }
    let dir = storage.data_dir().to_path_buf();
    let mut reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let k = reg.entry(dir).or_default();
    for (id, summary) in due_items {
        if k.announced_due.insert(id.clone()) {
            k.due_intentions.push((id.clone(), summary.clone()));
        }
    }
}

/// Drain pending due intentions on the very next `UserPromptSubmit` or `PreToolUse` hook firing.
pub fn drain_due_intention_lines(storage: &Storage) -> Vec<String> {
    let dir = storage.data_dir().to_path_buf();
    let mut reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let Some(k) = reg.get_mut(&dir) else {
        return Vec::new();
    };
    let items = std::mem::take(&mut k.due_intentions);
    items
        .into_iter()
        .take(4)
        .map(|(id, summary)| format!("Prospective Trigger due ({id}): {summary}"))
        .collect()
}

/// Step 7B: True Sleep Compiler — Galois Lattice FCA + Ochiai Failure Spectrum + Titration
/// (`k >= 3`) turning repeated episodic stops/failures into provable Candidate Standing Laws!
pub fn induce_standing_laws(
    storage: &Storage,
    scope: &str,
    persist_proposals: bool,
) -> Result<Vec<InducedLawProposal>, String> {
    let nodes = storage
        .get_all_nodes_in_scope(scope, i32::MAX, 0)
        .map_err(|e| e.to_string())?;
    let retired = storage.superseded_node_ids().map_err(|e| e.to_string())?;

    // Build formal context: failure/stop episodes -> structural attributes (rules, file tags, tool tags).
    let mut failure_objects: Vec<(String, BTreeSet<String>)> = Vec::new();
    let mut all_attr_pass_counts: HashMap<String, usize> = HashMap::new();
    let mut attr_witnesses: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for node in &nodes {
        if retired.contains(&node.id) {
            continue;
        }
        let is_failure = node.tags.iter().any(|t| {
            t == "operator-stop" || t == "failure" || t == "bug" || t == "regression"
        });
        let attrs: BTreeSet<String> = node
            .tags
            .iter()
            .filter(|t| {
                t.starts_with("rule:")
                    || t.starts_with("file:")
                    || t.starts_with("tool:")
                    || t.starts_with("slot:")
                    || t.starts_with("op-")
            })
            .cloned()
            .collect();

        if is_failure && !attrs.is_empty() {
            for a in &attrs {
                attr_witnesses.entry(a.clone()).or_default().push(node.id.clone());
            }
            failure_objects.push((node.id.clone(), attrs));
        } else {
            for a in &attrs {
                *all_attr_pass_counts.entry(a.clone()).or_default() += 1;
            }
        }
    }

    let total_failures = failure_objects.len();
    let mut proposals = Vec::new();
    let mut seen_closures: HashSet<Vec<String>> = HashSet::new();

    for (seed_attr, witnesses) in &attr_witnesses {
        let titration = evaluate_titration_threshold(seed_attr, witnesses, 3);
        if !titration.promoted {
            continue;
        }
        let closure: Vec<String> = galois_attribute_closure(
            &failure_objects,
            &BTreeSet::from([seed_attr.clone()]),
        )
        .into_iter()
        .collect();
        if closure.is_empty() || !seen_closures.insert(closure.clone()) {
            continue;
        }
        let ef = witnesses.len();
        let ep = all_attr_pass_counts.get(seed_attr).copied().unwrap_or(0);
        let nf = total_failures.saturating_sub(ef);
        let ochiai = ochiai_failure_spectrum(ef as u64, nf as u64, ep as u64);

        let mut h = blake3::Hasher::new();
        h.update(b"strata-sleep-law-v1:");
        for a in &closure {
            h.update(a.as_bytes());
            h.update(b"|");
        }
        for w in witnesses {
            h.update(w.as_bytes());
            h.update(b",");
        }
        let proof_blake3 = h.finalize().to_hex().to_string();
        let law_key = format!("law-proposal:{}", &proof_blake3[..12]);
        let title = format!(
            "Candidate Standing Law [{}]: {} recurring failure episodes share invariant signature [{}] (Ochiai {:.2})",
            law_key,
            witnesses.len(),
            closure.join(" & "),
            ochiai
        );

        let mut saved_proposal_id = None;
        if persist_proposals {
            let already_exists = nodes.iter().any(|n| n.tags.iter().any(|t| t == &law_key));
            if !already_exists {
                let mut prop_tags = vec![
                    "law-proposal".to_string(),
                    "dream-compiled".to_string(),
                    law_key.clone(),
                ];
                prop_tags.extend(closure.iter().take(4).cloned());
                let input = vestige_core::IngestInput {
                    content: format!(
                        "{title}. Derived from witnesses: [{}]. Promote with `curate` or tag `pinned` to activate as standing law.",
                        witnesses.join(", ")
                    ),
                    node_type: "pattern".to_string(),
                    tags: prop_tags,
                    source: Some("dream_compile:law_inducer".to_string()),
                    sentiment_score: 0.0,
                    sentiment_magnitude: 0.0,
                    valid_from: Some(Utc::now()),
                    valid_until: None,
                    ..Default::default()
                };
                if let Ok(created) = storage.ingest_in_scope_with_secret_policy(
                    input,
                    scope,
                    SecretPolicy::default(),
                ) {
                    for w in witnesses.iter().take(8) {
                        let edge = make_edge(&created.id, w, "derived_from", 0.9);
                        let _ = storage.save_connection(&edge);
                    }
                    saved_proposal_id = Some(created.id);
                }
            }
        }

        proposals.push(InducedLawProposal {
            law_key,
            title,
            trigger_attributes: closure,
            witness_ids: witnesses.clone(),
            ochiai_score: ochiai,
            promoted_by_titration: titration.promoted,
            proof_blake3,
            saved_proposal_id,
        });
    }

    Ok(proposals)
}

/// Inspect live kernel telemetry (for health/status/tests).
pub fn kernel_telemetry(storage: &Storage) -> Value {
    let dir = storage.data_dir().to_path_buf();
    let reg = kernel_registry().lock().unwrap_or_else(|e| e.into_inner());
    let Some(k) = reg.get(&dir) else {
        return json!({
            "active": true,
            "watchedFiles": 0,
            "labileAnchors": 0,
            "activePassports": 0,
            "synapticWindowEntries": 0,
            "pendingDueIntentions": 0,
        });
    };
    json!({
        "active": true,
        "seq": k.seq,
        "watchedFiles": k.watched_paths.len(),
        "labileAnchors": k.labile_anchors.len(),
        "labileFiles": k
            .labile_anchors
            .values()
            .map(|r| r.file_path.as_str())
            .collect::<Vec<_>>(),
        "activePassports": k.passports.len(),
        "synapticWindowEntries": k.synaptic_tags.len(),
        "pendingDueIntentions": k.due_intentions.len(),
        "vectorClock": k.vector_clock.entries,
    })
}
