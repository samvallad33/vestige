#![forbid(unsafe_code)]

use ed25519_dalek::{Signer, SigningKey};
use vestige_core::chain_fold::{
    GENESIS_PREV_HASH, SignedWriterFrame, WriterId, fold_writer_chains,
};
use vestige_core::provenance::{
    CausalEdge, PolicyVerdict, ProvenanceEventKind, StrataProvenanceReceipt, TouchedFileHunk,
    project_provenance_bodies, prove_causal_chain, quarantine_body, release_body,
};

#[test]
fn test_seven_field_receipt_proves_symptom_back_to_prompt_and_commit() {
    let sk = SigningKey::from_bytes(&[42u8; 32]);
    let prompt_hash = StrataProvenanceReceipt::compute_prompt_hash("fix auth race condition");

    let edit_receipt = StrataProvenanceReceipt {
        writer_id: WriterId::from_bytes(sk.verifying_key().to_bytes()),
        passport_digest: [7u8; 32],
        session_id: "sess-1".into(),
        turn_seq: 1,
        prompt_hash,
        git_head_sha: "abc123def".into(),
        files: vec![TouchedFileHunk {
            path: "src/auth.rs".into(),
            start_line: Some(40),
            end_line: Some(58),
        }],
        tool_call_digest: StrataProvenanceReceipt::compute_tool_call_digest("Edit", "src/auth.rs"),
        exit_code: Some(0),
        error_sig: None,
        parent_frame_digest: [0u8; 32],
        causal_edge: CausalEdge::Root,
        policy_verdict: PolicyVerdict::Allowed {
            rule_id: "auto_capture_edit".into(),
        },
        kind: ProvenanceEventKind::EditAnchor,
        summary: "Edited src/auth.rs".into(),
    };
    let f0 = edit_receipt
        .into_signed_frame(&sk, 0, 1, 1_000, GENESIS_PREV_HASH)
        .expect("sign f0");

    let fail_receipt = StrataProvenanceReceipt {
        kind: ProvenanceEventKind::OpenFailure,
        exit_code: Some(101),
        error_sig: Some("exit:101:panicked at token expired".into()),
        parent_frame_digest: f0.frame_digest,
        causal_edge: CausalEdge::EvidenceOf,
        summary: "cargo test failed".into(),
        ..edit_receipt.clone()
    };
    let f1 = fail_receipt
        .into_signed_frame(&sk, 1, 2, 1_100, f0.frame_digest)
        .expect("sign f1");

    let save_receipt = StrataProvenanceReceipt {
        kind: ProvenanceEventKind::RememberSave,
        exit_code: Some(0),
        error_sig: Some("exit:101:panicked at token expired".into()),
        parent_frame_digest: f1.frame_digest,
        causal_edge: CausalEdge::Closes,
        summary: "Fixed clock skew check in src/auth.rs".into(),
        ..edit_receipt
    };
    let f2 = save_receipt
        .into_signed_frame(&sk, 2, 3, 1_200, f1.frame_digest)
        .expect("sign f2");

    let folded = fold_writer_chains(&[f2.clone(), f0.clone(), f1.clone()]);
    assert_eq!(folded.admitted_frames.len(), 3);

    let proof = prove_causal_chain(&folded.admitted_frames, f2.frame_digest);
    assert_eq!(proof.len(), 3);
    assert_eq!(proof[0].0, f2.frame_digest);
    assert_eq!(proof[1].0, f1.frame_digest);
    assert_eq!(proof[2].0, f0.frame_digest);
    assert_eq!(proof[2].1.git_head_sha, "abc123def");
    assert_eq!(proof[2].1.prompt_hash, prompt_hash);
}

#[test]
fn test_orphan_receipt_without_anchors_is_rejected_with_missing_provenance_receipt() {
    let sk = SigningKey::from_bytes(&[42u8; 32]);
    let orphan = StrataProvenanceReceipt {
        writer_id: WriterId::from_bytes(sk.verifying_key().to_bytes()),
        passport_digest: [0u8; 32],
        session_id: String::new(),
        turn_seq: 0,
        prompt_hash: [0u8; 32],
        git_head_sha: String::new(),
        files: Vec::new(),
        tool_call_digest: [0u8; 32],
        exit_code: None,
        error_sig: None,
        parent_frame_digest: [0u8; 32],
        causal_edge: CausalEdge::Root,
        policy_verdict: PolicyVerdict::Allowed {
            rule_id: "remember.autobind.v1".into(),
        },
        kind: ProvenanceEventKind::RememberSave,
        summary: "orphan save without any anchor".into(),
    };

    let receipt = orphan
        .require_anchor()
        .expect_err("orphan receipt must be rejected");
    assert_ne!(receipt.receipt_digest, [0u8; 32]);
    assert!(
        receipt
            .reason
            .contains("no session turn, tool, file, or causal link anchor")
    );
}

#[test]
fn test_flipped_signature_byte_causes_prove_causal_chain_to_fail() {
    let sk = SigningKey::from_bytes(&[42u8; 32]);
    let prompt_hash = StrataProvenanceReceipt::compute_prompt_hash("fix auth race condition");

    let edit_receipt = StrataProvenanceReceipt {
        writer_id: WriterId::from_bytes(sk.verifying_key().to_bytes()),
        passport_digest: [7u8; 32],
        session_id: "sess-tamper".into(),
        turn_seq: 1,
        prompt_hash,
        git_head_sha: "abc123def".into(),
        files: vec![TouchedFileHunk {
            path: "src/auth.rs".into(),
            start_line: Some(1),
            end_line: Some(10),
        }],
        tool_call_digest: StrataProvenanceReceipt::compute_tool_call_digest("Edit", "src/auth.rs"),
        exit_code: Some(0),
        error_sig: None,
        parent_frame_digest: [0u8; 32],
        causal_edge: CausalEdge::Root,
        policy_verdict: PolicyVerdict::Allowed {
            rule_id: "auto_capture_edit".into(),
        },
        kind: ProvenanceEventKind::EditAnchor,
        summary: "Edited src/auth.rs".into(),
    };
    let f0 = edit_receipt
        .into_signed_frame(&sk, 0, 1, 1_000, GENESIS_PREV_HASH)
        .expect("sign f0");

    let mut f1 = StrataProvenanceReceipt {
        kind: ProvenanceEventKind::OpenFailure,
        exit_code: Some(101),
        error_sig: Some("exit:101:test failed".into()),
        parent_frame_digest: f0.frame_digest,
        causal_edge: CausalEdge::EvidenceOf,
        summary: "cargo test failed".into(),
        ..edit_receipt.clone()
    }
    .into_signed_frame(&sk, 1, 2, 1_100, f0.frame_digest)
    .expect("sign f1");

    let f2 = StrataProvenanceReceipt {
        kind: ProvenanceEventKind::RememberSave,
        exit_code: Some(0),
        error_sig: Some("exit:101:test failed".into()),
        parent_frame_digest: f1.frame_digest,
        causal_edge: CausalEdge::Closes,
        summary: "Fixed bug".into(),
        ..edit_receipt
    }
    .into_signed_frame(&sk, 2, 3, 1_200, f1.frame_digest)
    .expect("sign f2");

    // Flip a single bit in f1's Ed25519 signature.
    f1.signature[0] ^= 0x01;

    // 1. Direct prove_causal_chain drops f1 on verify_cryptography(), breaking the chain before f0.
    let direct_proof = prove_causal_chain(&[f0.clone(), f1.clone(), f2.clone()], f2.frame_digest);
    assert_ne!(direct_proof.len(), 3);
    assert!(!direct_proof.iter().any(|(d, _)| *d == f0.frame_digest));

    // 2. Folding also rejects f1 and quarantines downstream f2, leaving prove_causal_chain empty.
    let folded = fold_writer_chains(&[f0, f1, f2.clone()]);
    let folded_proof = prove_causal_chain(&folded.admitted_frames, f2.frame_digest);
    assert!(folded_proof.is_empty());
}

fn sign_payload(sk: &SigningKey, payload: &[u8]) -> SignedWriterFrame {
    let writer_id = WriterId::from_bytes(sk.verifying_key().to_bytes());
    let payload_digest = SignedWriterFrame::compute_payload_digest(payload);
    let frame_digest = SignedWriterFrame::compute_frame_digest(
        &writer_id,
        0,
        1,
        0,
        &GENESIS_PREV_HASH,
        &payload_digest,
    );
    SignedWriterFrame {
        writer_id,
        seq: 0,
        lamport: 1,
        timestamp_ms: 0,
        prev_hash: GENESIS_PREV_HASH,
        payload_digest,
        frame_digest,
        signature: sk.sign(&frame_digest).to_bytes().to_vec(),
        payload: payload.to_vec(),
    }
}

#[test]
fn orphan_memory_is_quarantined_under_missing_provenance_receipt() {
    let (body, receipt) = quarantine_body("", "orphan memory with no anchor").expect("body");
    let again = quarantine_body("", "orphan memory with no anchor").expect("body");
    assert_eq!(body, again.0);
    assert_eq!(receipt, again.1);
    assert_ne!(receipt.receipt_digest, [0u8; 32]);

    let view = project_provenance_bodies(&[body]);
    assert_eq!(view.quarantined.len(), 1);
    assert_eq!(view.quarantined[0].receipt, receipt);
    assert!(view.released.is_empty());
    assert_eq!(view.retained.len(), 1);
}

#[test]
fn appended_release_frame_frees_the_quarantine_and_keeps_the_record() {
    let (body, receipt) = quarantine_body("", "orphan memory with no anchor").expect("body");
    let anchor = [11u8; 32];
    let release = release_body(receipt.receipt_digest, anchor);
    let view = project_provenance_bodies(&[body.clone(), release]);
    assert!(view.quarantined.is_empty());
    assert_eq!(view.retained.len(), 1);
    assert_eq!(view.retained[0].summary, "orphan memory with no anchor");
    assert_eq!(view.released.len(), 1);
    assert_eq!(view.released[0].receipt_digest, receipt.receipt_digest);
    assert_eq!(view.released[0].anchor_digest, anchor);
    assert!(view.retained[0].summary.contains("orphan"));
}

#[test]
fn rerunning_quarantine_and_release_yields_identical_frames() {
    let sk = SigningKey::from_bytes(&[9u8; 32]);
    let (body, receipt) = quarantine_body("", "orphan memory with no anchor").expect("body");
    let (body_again, receipt_again) =
        quarantine_body("", "orphan memory with no anchor").expect("body");
    let anchor = [11u8; 32];
    let release = release_body(receipt.receipt_digest, anchor);
    let release_again = release_body(receipt_again.receipt_digest, anchor);

    let first = sign_payload(&sk, body.as_bytes());
    let second = sign_payload(&sk, body_again.as_bytes());
    assert_eq!(first, second);
    assert!(first.verify_cryptography().is_ok());
    assert_eq!(
        sign_payload(&sk, release.as_bytes()),
        sign_payload(&sk, release_again.as_bytes())
    );
}
