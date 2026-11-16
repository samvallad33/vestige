#![forbid(unsafe_code)]

//! Wave 14M (#774) — Auto-Bind in `vestige-mcp` `remember(action="save")`
//!
//! Ensures RSB and full 7-field provenance fire even when the agent omits `files` or `links`,
//! and rejects unanchored orphan saves with a deterministic `MissingProvenanceReceipt`.

use std::fmt;

use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use vestige_core::chain_fold::{Hash32, SignedWriterFrame, WriterId};
use vestige_core::provenance::{
    CausalEdge, MissingProvenanceReceipt, ORPHAN_QUARANTINE_REASON, PolicyVerdict,
    ProvenanceEventKind, StrataProvenanceReceipt, TouchedFileHunk,
};

use super::pairing::ReflexSessionState;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RememberSaveInput {
    pub content: String,
    pub files: Option<Vec<TouchedFileHunk>>,
    pub links: Option<Vec<Hash32>>,
    pub causal_edge: Option<CausalEdge>,
    pub passport_digest: Option<Hash32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoBoundRememberResult {
    pub signed_frame: SignedWriterFrame,
    pub receipt: StrataProvenanceReceipt,
    /// All failure/fix/edit digests queued for Retroactive Salience Backfill (RSB).
    pub rsb_backfill_targets: Vec<Hash32>,
}

#[derive(Debug)]
pub enum RememberAutoBindError {
    MissingProvenance(MissingProvenanceReceipt),
    Serialization(serde_json::Error),
}

impl fmt::Display for RememberAutoBindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingProvenance(receipt) => {
                write!(f, "missing provenance anchor: {}", receipt.reason)
            }
            Self::Serialization(err) => write!(f, "provenance serialization error: {err}"),
        }
    }
}

impl std::error::Error for RememberAutoBindError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(err) => Some(err),
            Self::MissingProvenance(_) => None,
        }
    }
}

impl From<serde_json::Error> for RememberAutoBindError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err)
    }
}

/// Auto-binds omitted `files`, `links`, `(session_id, turn_seq, prompt_hash)`, and `git_head_sha`
/// from `ReflexSessionState`, rejects saves lacking any session/tool/file/link anchor, emits a
/// signed 7-field `SignedWriterFrame`, and marks the session anchored so the Save-Gate unlocks.
pub fn execute_remember_save_with_autobind(
    reflex: &mut ReflexSessionState,
    input: RememberSaveInput,
    signing_key: &SigningKey,
    seq: u64,
    lamport: u64,
    timestamp_ms: u64,
    prev_hash: Hash32,
) -> Result<AutoBoundRememberResult, RememberAutoBindError> {
    let auto_ctx = reflex.build_auto_bind_context();

    let has_session_turn = !reflex.session_id.trim().is_empty()
        && (reflex.turn_seq > 0 || reflex.prompt_hash != [0u8; 32]);
    let has_tool_anchor = !reflex.open_failures.is_empty()
        || !reflex.resolved_pairs.is_empty()
        || !reflex.anchor_events.is_empty();
    let has_file_anchor = input.files.as_ref().is_some_and(|files| !files.is_empty())
        || !auto_ctx.bound_files.is_empty();
    let has_link_anchor = input.links.as_ref().is_some_and(|links| !links.is_empty())
        || auto_ctx.parent_frame_digest != [0u8; 32];

    if !has_session_turn && !has_tool_anchor && !has_file_anchor && !has_link_anchor {
        return Err(RememberAutoBindError::MissingProvenance(
            MissingProvenanceReceipt::new(
                &reflex.session_id,
                &input.content,
                ORPHAN_QUARANTINE_REASON,
            ),
        ));
    }

    let bound_files = match input.files {
        Some(explicit) if !explicit.is_empty() => explicit,
        _ => auto_ctx.bound_files,
    };

    let (parent_frame_digest, causal_edge) = match input.links {
        Some(explicit_links) if !explicit_links.is_empty() => (
            explicit_links[0],
            input.causal_edge.unwrap_or(auto_ctx.causal_edge),
        ),
        _ => (
            auto_ctx.parent_frame_digest,
            input.causal_edge.unwrap_or(auto_ctx.causal_edge),
        ),
    };

    let writer_id = WriterId::from_bytes(signing_key.verifying_key().to_bytes());
    let tool_call_digest =
        StrataProvenanceReceipt::compute_tool_call_digest("remember", &input.content);

    let receipt = StrataProvenanceReceipt {
        writer_id,
        passport_digest: input.passport_digest.unwrap_or([0u8; 32]),
        session_id: auto_ctx.session_id,
        turn_seq: auto_ctx.turn_seq,
        prompt_hash: auto_ctx.prompt_hash,
        git_head_sha: auto_ctx.git_head_sha,
        files: bound_files.clone(),
        tool_call_digest,
        exit_code: Some(0),
        error_sig: auto_ctx.resolved_error_sigs.last().cloned(),
        parent_frame_digest,
        causal_edge,
        policy_verdict: PolicyVerdict::Allowed {
            rule_id: "remember.autobind.v1".to_owned(),
        },
        kind: ProvenanceEventKind::RememberSave,
        summary: input.content,
    };

    let signed_frame =
        receipt.into_signed_frame(signing_key, seq, lamport, timestamp_ms, prev_hash)?;

    let covered_paths: Vec<String> = bound_files.iter().map(|h| h.path.clone()).collect();
    reflex.mark_remember_saved(&covered_paths, signed_frame.frame_digest);

    Ok(AutoBoundRememberResult {
        signed_frame,
        receipt,
        rsb_backfill_targets: auto_ctx.rsb_target_digests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::reflex::pairing::save_gate::SaveGateVerdict;
    use vestige_core::chain_fold::GENESIS_PREV_HASH;

    #[test]
    fn test_autobind_attaches_unsaved_files_and_open_failure_and_unlocks_gate() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let mut reflex = ReflexSessionState::new("sess-autobind");
        reflex.on_user_prompt([1u8; 32], "abc123def");

        let edit_digest = [2u8; 32];
        reflex.record_file_anchor(
            "edit",
            "abc123def",
            vec![TouchedFileHunk {
                path: "src/app.rs".to_owned(),
                start_line: Some(10),
                end_line: Some(15),
            }],
            edit_digest,
            1_000,
        );

        let fail_digest = [3u8; 32];
        reflex.record_command_outcome(
            "cargo test",
            101,
            Some("exit:101:skew check failed".to_owned()),
            [4u8; 32],
            "abc123def",
            fail_digest,
            1_100,
        );

        let fix_digest = [5u8; 32];
        reflex.record_command_outcome(
            "cargo test",
            0,
            None,
            [6u8; 32],
            "abc123def",
            fix_digest,
            1_200,
        );

        assert!(matches!(
            reflex.check_save_gate(),
            SaveGateVerdict::Block { .. }
        ));

        let res = execute_remember_save_with_autobind(
            &mut reflex,
            RememberSaveInput {
                content: "Fixed clock skew threshold".to_owned(),
                files: None,
                links: None,
                causal_edge: None,
                passport_digest: None,
            },
            &sk,
            0,
            1,
            1_300,
            GENESIS_PREV_HASH,
        )
        .expect("autobind should succeed");

        assert_eq!(res.receipt.files.len(), 1);
        assert_eq!(res.receipt.files[0].path, "src/app.rs");
        assert_eq!(res.receipt.parent_frame_digest, fail_digest);
        assert_eq!(res.receipt.causal_edge, CausalEdge::Closes);
        assert_eq!(res.rsb_backfill_targets, vec![fail_digest, fix_digest]);
        assert_eq!(reflex.check_save_gate(), SaveGateVerdict::Allow);
    }

    #[test]
    fn test_orphan_save_with_no_anchors_returns_missing_provenance_receipt() {
        let sk = SigningKey::from_bytes(&[9u8; 32]);
        let mut empty_reflex = ReflexSessionState::new("");

        let err = execute_remember_save_with_autobind(
            &mut empty_reflex,
            RememberSaveInput {
                content: "orphan decision".to_owned(),
                files: None,
                links: None,
                causal_edge: None,
                passport_digest: None,
            },
            &sk,
            0,
            1,
            1_000,
            GENESIS_PREV_HASH,
        )
        .expect_err("orphan save must be rejected");

        match err {
            RememberAutoBindError::MissingProvenance(receipt) => {
                assert_ne!(receipt.receipt_digest, [0u8; 32]);
            }
            other => panic!("expected MissingProvenance, got {other:?}"),
        }
        assert_eq!(empty_reflex, ReflexSessionState::new(""));
    }

    #[test]
    fn session_receipt_autobinds_so_rsb_targets_the_failure_and_rerun_is_identical() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let mut reflex = ReflexSessionState::new("sess-autobind");
        reflex.on_user_prompt([1u8; 32], "abc123def");
        reflex.record_file_anchor(
            "edit",
            "abc123def",
            vec![TouchedFileHunk {
                path: "src/app.rs".to_owned(),
                start_line: Some(10),
                end_line: Some(15),
            }],
            [2u8; 32],
            1_000,
        );
        let fail_digest = [3u8; 32];
        reflex.record_command_outcome(
            "cargo test",
            101,
            Some("exit:101:skew check failed".to_owned()),
            [4u8; 32],
            "abc123def",
            fail_digest,
            1_100,
        );
        reflex.record_command_outcome(
            "cargo test",
            0,
            None,
            [6u8; 32],
            "abc123def",
            [5u8; 32],
            1_200,
        );
        let input = RememberSaveInput {
            content: "Fixed clock skew threshold".to_owned(),
            files: None,
            links: None,
            causal_edge: None,
            passport_digest: None,
        };
        let first = execute_remember_save_with_autobind(
            &mut reflex.clone(),
            input.clone(),
            &sk,
            0,
            1,
            1_300,
            GENESIS_PREV_HASH,
        )
        .expect("autobind");
        let second = execute_remember_save_with_autobind(
            &mut reflex,
            input,
            &sk,
            0,
            1,
            1_300,
            GENESIS_PREV_HASH,
        )
        .expect("rerun");

        assert_eq!(first.signed_frame, second.signed_frame);
        assert_eq!(first.rsb_backfill_targets, second.rsb_backfill_targets);
        assert!(first.rsb_backfill_targets.contains(&fail_digest));
        assert_eq!(first.receipt.parent_frame_digest, fail_digest);
        assert_eq!(first.receipt.causal_edge, CausalEdge::Closes);
    }
}
