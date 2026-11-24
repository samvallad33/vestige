//! End-to-end verification of the Closed-Loop Deterministic Cognitive Kernel:
//! 1. Unified GhostCurrent Circuit Index (`circuit_index`) + SIMD + AC conflict + Back-EMF
//! 2. Write-Time Strata Intake Compiler (exclusive slots, owner conflict hold, 9h Synaptic Capture)
//! 3. Autonomous BLAKE3 Anchor Watch (`#524`) + GhostCurrent labile shunting
//! 4. Multi-Agent Swarm OS (`SubagentPassport`, `CognitiveBranchDag`, `correction_ripple`)
//! 5. Prefrontal Blast-Radius Simulator, Output Operator (`Stop` gate), Prospective Push,
//!    and Galois/Ochiai/Titration Sleep Law Inducer.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;

use chrono::Utc;
use serde_json::json;
use tempfile::tempdir;
use vestige_core::{ConnectionRecord, IngestInput, SecretPolicy, Storage};
use vestige_mcp::circuit_index::{CircuitQuery, query_circuit};
use vestige_mcp::cognitive_kernel::{
    IntakeWriteRequest, anchor_watch_lines, audit_stop_output, compile_ingest_write,
    compute_correction_ripple, drain_due_intention_lines, fold_subagent_stop, induce_standing_laws,
    issue_subagent_passport, labile_memory_ids, push_due_intentions, simulate_edit_blast_radius,
    stage_subagent_write, verify_subagent_tool,
};

fn open_test_store(dir: &std::path::Path) -> Arc<Storage> {
    vestige_mcp::strata_memory::open(dir).expect("open real Strata log")
}

fn save_raw(storage: &Storage, content: &str, node_type: &str, tags: &[&str]) -> String {
    let node = storage
        .ingest_in_scope_with_secret_policy(
            IngestInput {
                content: content.to_string(),
                node_type: node_type.to_string(),
                tags: tags.iter().map(|s| (*s).to_string()).collect(),
                source: Some("test".to_string()),
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                valid_from: Some(Utc::now()),
                valid_until: None,
                ..Default::default()
            },
            "user",
            SecretPolicy::default(),
        )
        .expect("ingest node");
    node.id
}

#[test]
fn unified_ghostcurrent_circuit_returns_proofs_conflicts_and_back_emf() {
    let dir = tempdir().expect("tempdir");
    let storage_arc = open_test_store(dir.path());
    let storage = storage_arc.as_ref();

    let m1 = save_raw(
        storage,
        "Gateway auth token TTL is configured to 60 minutes in auth.rs",
        "decision",
        &["auth", "file:src/auth.rs"],
    );
    let m2 = save_raw(
        storage,
        "Legacy gateway auth token TTL was 15 minutes in auth.rs",
        "fact",
        &["auth", "file:src/auth.rs"],
    );
    let now = Utc::now();
    storage
        .save_connection(&ConnectionRecord {
            source_id: m1.clone(),
            target_id: m2.clone(),
            link_type: "corrects".to_string(),
            strength: 0.95,
            activation_count: 1,
            created_at: now,
            last_activated: now,
        })
        .expect("save corrects edge");

    let labile = HashSet::new();
    let resp = query_circuit(
        storage,
        &CircuitQuery {
            query: "gateway auth token ttl nonexistent_quantum_token",
            scope: Some("user"),
            include_superseded: false,
            valid_at: None,
            top_k: 5,
            labile_ids: &labile,
        },
    );

    assert!(!resp.hits.is_empty(), "circuit must retrieve grounded auth hits");
    assert!(resp.hits[0].proof.current_q16 > 0);
    assert!(!resp.hits[0].proof.path_witness.is_empty());
    assert!(
        !resp.conflicts.is_empty(),
        "AC Belnap interference must surface explicit contradicts pair"
    );
    assert!(
        resp.epistemic
            .missing_premises
            .contains(&"nonexistent_quantum_token".to_string()),
        "Open-Circuit Back-EMF must report uncovered premise"
    );
    assert!(resp.epistemic.back_emf_q16 > 0);
}

#[test]
fn strata_intake_compiles_exclusive_slots_owner_holds_and_9h_synaptic_capture() {
    let dir = tempdir().expect("tempdir");
    let storage_arc = open_test_store(dir.path());
    let storage = storage_arc.as_ref();

    // 1. Two working notes enter the 9-hour synaptic tag window.
    let n1 = save_raw(storage, "Investigated port binding failure on staging", "note", &["deploy"]);
    let _ = compile_ingest_write(
        storage,
        &IntakeWriteRequest {
            node_id: &n1,
            content: "Investigated port binding failure on staging",
            node_type: "note",
            tags: &["deploy".to_string()],
            scope: "user",
            repo_path: None,
            files: &[],
        },
    );
    let n2 = save_raw(storage, "deploy.port = 8080", "fact", &["deploy"]);
    let r2 = compile_ingest_write(
        storage,
        &IntakeWriteRequest {
            node_id: &n2,
            content: "deploy.port = 8080",
            node_type: "fact",
            tags: &["deploy".to_string()],
            scope: "user",
            repo_path: None,
            files: &[],
        },
    );
    assert!(r2.extracted_slots.contains(&"slot:deploy:port".to_string()));
    assert!(r2.auto_superseded_ids.is_empty());

    // 2. Owner-pinned decision updates the same exclusive slot -> auto-supersedes n2 AND
    //    triggers 9-hour Synaptic Tagging & Capture linking prior window notes!
    let n3 = save_raw(storage, "deploy.port = 9090", "decision", &["pinned", "deploy"]);
    let r3 = compile_ingest_write(
        storage,
        &IntakeWriteRequest {
            node_id: &n3,
            content: "deploy.port = 9090",
            node_type: "decision",
            tags: &["pinned".to_string(), "deploy".to_string()],
            scope: "user",
            repo_path: None,
            files: &[],
        },
    );
    assert!(
        r3.auto_superseded_ids.contains(&n2),
        "new slot value must auto-supersede prior slot owner"
    );
    assert!(
        r3.synaptic_captured_handles.contains(&n1),
        "PRP shock decision must retroactively capture 9h synaptic window note"
    );

    // 3. Unapproved Agent tries to overwrite Owner-pinned slot -> HoldConflict!
    let n4 = save_raw(storage, "deploy.port = 7777", "fact", &["deploy"]);
    let r4 = compile_ingest_write(
        storage,
        &IntakeWriteRequest {
            node_id: &n4,
            content: "deploy.port = 7777",
            node_type: "fact",
            tags: &["deploy".to_string()],
            scope: "user",
            repo_path: None,
            files: &[],
        },
    );
    assert!(
        !r4.conflict_holds.is_empty(),
        "non-owner must not silently overwrite Owner-pinned exclusive slot"
    );
}

#[test]
fn anchor_watch_detects_code_drift_and_shunts_ghostcurrent() {
    let dir = tempdir().expect("tempdir");
    let storage_arc = open_test_store(dir.path());
    let storage = storage_arc.as_ref();
    let src_file = dir.path().join("auth.rs");
    std::fs::write(&src_file, b"pub fn verify() -> bool { true }\n").expect("write v1");

    let mem_id = save_raw(
        storage,
        "verify() in auth.rs always checks Ed25519 signature",
        "decision",
        &["file:auth.rs"],
    );
    let rep = compile_ingest_write(
        storage,
        &IntakeWriteRequest {
            node_id: &mem_id,
            content: "verify() in auth.rs always checks Ed25519 signature",
            node_type: "decision",
            tags: &["file:auth.rs".to_string()],
            scope: "user",
            repo_path: Some(dir.path().to_str().unwrap()),
            files: &["auth.rs".to_string()],
        },
    );
    assert_eq!(rep.anchored_files, vec!["auth.rs".to_string()]);

    // Mutate file on disk without curate -> Anchor Watch marks memory Labile!
    std::fs::write(&src_file, b"pub fn verify() -> bool { false }\n").expect("write v2");
    let alerts = anchor_watch_lines(storage, Some("auth.rs"));
    assert_eq!(alerts.len(), 1);
    assert!(alerts[0].contains(&mem_id));

    let labile = labile_memory_ids(storage);
    assert!(labile.contains(&mem_id));

    // Unified GhostCurrent circuit must shunt and label the drifted anchor!
    let resp = query_circuit(
        storage,
        &CircuitQuery {
            query: "verify auth.rs Ed25519",
            scope: Some("user"),
            include_superseded: false,
            valid_at: None,
            top_k: 5,
            labile_ids: &labile,
        },
    );
    let hit = resp.hits.iter().find(|h| h.id == mem_id).expect("hit present");
    assert!(hit.proof.labile_drifted);
    assert_eq!(hit.proof.polarity_phase, "labile_drifted_shunt");
}

#[test]
fn subagent_passport_branch_dag_and_correction_ripple_enforce_swarm_isolation() {
    let dir = tempdir().expect("tempdir");
    let storage_arc = open_test_store(dir.path());
    let storage = storage_arc.as_ref();

    let (passport, branch) = issue_subagent_passport(
        storage,
        "sess-parent-1",
        "worker-readonly",
        vec!["mem-00000000000000a1".to_string()],
        &json!({
            "read_only": true,
            "allow_path_prefixes": ["crates/vestige-core/"],
            "allow_tools": ["read_file", "recall", "edit_file"]
        }),
    );
    assert_eq!(branch, "branch:worker-readonly");
    assert_eq!(passport.caveats.len(), 3);

    // Allowed read inside permitted prefix passes.
    assert!(
        verify_subagent_tool(
            storage,
            "worker-readonly",
            "read_file",
            Some("crates/vestige-core/src/lib.rs"),
            None,
            false
        )
        .is_ok()
    );
    // Mutation blocked by ReadOnly caveat.
    assert!(
        verify_subagent_tool(
            storage,
            "worker-readonly",
            "edit_file",
            Some("crates/vestige-core/src/lib.rs"),
            None,
            true
        )
        .unwrap_err()
        .contains("MutationDeniedByReadOnlyCaveat")
    );
    // Out-of-scope path blocked by AllowPathPrefixes caveat.
    assert!(
        verify_subagent_tool(
            storage,
            "worker-readonly",
            "read_file",
            Some("crates/vestige-mcp/src/main.rs"),
            None,
            false
        )
        .unwrap_err()
        .contains("PathDeniedByCaveat")
    );

    // Speculative branch write + clean fold on SubagentStop.
    stage_subagent_write(storage, "worker-readonly", "mem-00000000000000b2");
    let fold = fold_subagent_stop(storage, "worker-readonly").expect("fold report");
    assert!(fold.branch_merged);
    assert_eq!(fold.merged_count, 1);
    assert_eq!(fold.trail_edges.len(), 1);

    // Poisoned speculative branch write is quarantined by guarded_merge!
    let _ = issue_subagent_passport(storage, "sess-parent-1", "worker-bad", vec![], &json!({}));
    stage_subagent_write(storage, "worker-bad", "UNVERIFIED:hallucinated-claim");
    let bad_fold = fold_subagent_stop(storage, "worker-bad").expect("bad fold");
    assert!(!bad_fold.branch_merged);
    assert!(bad_fold.merge_error.is_some());

    // Correction ripple locates sibling agents holding a superseded handle.
    let mut manifests = BTreeMap::new();
    manifests.insert(
        "sess-1:worker-a".to_string(),
        BTreeSet::from(["mem-00000000000000a1".to_string()]),
    );
    manifests.insert(
        "sess-2:main".to_string(),
        BTreeSet::from(["mem-00000000000000ff".to_string()]),
    );
    assert_eq!(
        compute_correction_ripple("mem-00000000000000a1", &manifests),
        vec!["sess-1:worker-a".to_string()]
    );
}

#[test]
fn blast_radius_output_operator_prospective_push_and_sleep_law_inducer_work_end_to_end() {
    let dir = tempdir().expect("tempdir");
    let storage_arc = open_test_store(dir.path());
    let storage = storage_arc.as_ref();

    // Seed 3 recurring failure episodes sharing rule:OP-001 + file:scripts/deploy.sh.
    let f1 = save_raw(
        storage,
        "OP-001 stopped deploy.sh subshell command",
        "event",
        &["operator-stop", "rule:OP-001", "file:scripts/deploy.sh"],
    );
    let f2 = save_raw(
        storage,
        "OP-001 stopped second deploy.sh subshell call",
        "event",
        &["operator-stop", "rule:OP-001", "file:scripts/deploy.sh"],
    );
    let f3 = save_raw(
        storage,
        "OP-001 stopped third deploy.sh subshell call",
        "event",
        &["operator-stop", "rule:OP-001", "file:scripts/deploy.sh"],
    );
    let rule_mem = save_raw(
        storage,
        "Never use raw shell substitution in deploy.sh",
        "decision",
        &["pinned", "file:scripts/deploy.sh", "forbid:rm -rf /tmp/prod"],
    );

    // 1. Prefrontal Blast-Radius Simulator computes Ochiai risk before editing deploy.sh!
    let blast = simulate_edit_blast_radius(storage, "scripts/deploy.sh", None);
    assert!(blast.ochiai_risk_score > 0.5);
    assert_eq!(blast.historical_failures.len(), 3);
    assert!(blast.anchored_memories.contains(&rule_mem));
    assert!(blast.summary_line.is_some());

    // 2. Output Operator blocks forged test-pass claims, false abstention, and forbid violations.
    let held: HashSet<String> = [rule_mem.clone(), f1, f2, f3].into_iter().collect();
    let forged = audit_stop_output(
        storage,
        "Done! All tests pass and clippy clean.",
        &held,
        &["scripts/deploy.sh".to_string()],
        &[],
        "update deploy.sh",
    );
    assert!(
        forged
            .iter()
            .any(|r| r.contains("ForgedReasoningMissingExecutionReceipt")),
        "unwitnessed test-pass claim after edit must block: {forged:?}"
    );

    let abstained = audit_stop_output(
        storage,
        "I don't know how deploy.sh should avoid shell substitution.",
        &held,
        &[],
        &[],
        "How should deploy.sh avoid shell substitution?",
    );
    assert!(
        abstained.iter().any(|r| r.contains("false abstention")),
        "false abstention while grounded memory is held must block: {abstained:?}"
    );

    let forbidden = audit_stop_output(
        storage,
        "Run rm -rf /tmp/prod to reset state.",
        &held,
        &[],
        &[],
        "reset state",
    );
    assert!(
        forbidden.iter().any(|r| r.contains("pinned forbid rule")),
        "forbid rule violation must block: {forbidden:?}"
    );

    // 3. Prospective push bridges Autopilot due triggers -> Reflex live injection.
    push_due_intentions(
        storage,
        &[("int-1".to_string(), "Verify production backup checksum".to_string())],
    );
    let drained = drain_due_intention_lines(storage);
    assert_eq!(drained.len(), 1);
    assert!(drained[0].contains("int-1"));
    assert!(drain_due_intention_lines(storage).is_empty());

    // 4. Overnight Sleep Law Inducer compiles the 3 recurring stops into 1 provable Candidate Law!
    let laws = induce_standing_laws(storage, "user", true).expect("induce laws");
    assert!(
        !laws.is_empty(),
        "3 recurring failure episodes must cross titration threshold and induce a candidate law"
    );
    assert!(laws[0].promoted_by_titration);
    assert!(laws[0].trigger_attributes.contains(&"rule:OP-001".to_string()));
    assert!(laws[0].trigger_attributes.contains(&"file:scripts/deploy.sh".to_string()));
    assert!(laws[0].saved_proposal_id.is_some());
}
