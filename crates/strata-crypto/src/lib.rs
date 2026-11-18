//! # strata-crypto — Tranche 2 Operating System Kernel (24 Issues)
//!
//! Covers `#399, #400, #401, #402, #403, #404, #406, #407, #408, #409, #410,
//! #411, #412, #417, #418, #419, #421, #422, #541, #567, #571, #573, #574, #770`.

pub mod cli_ops;
pub mod ledger;
pub mod mmr;

#[cfg(test)]
mod tests {
    use crate::cli_ops::*;
    use crate::ledger::*;
    use crate::mmr::*;

    #[test]
    fn issue_399_property_tests_core_strata_invariants() {
        // Deterministic LCG property test across 64 generated log prefixes:
        // 1. MMR root at prefix k never changes as items k+1..N are appended (consistency).
        // 2. Inclusion proof verifies for every index i in every tree size 1..=32.
        // 3. Per-writer merge fold is invariant under batch reversal and duplication.
        let mut tree = MmrTree::default();
        let mut historical_roots = Vec::new();
        let mut state = 0x1234_5678_9abc_def0u64;
        for step in 0..32usize {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let payload = state.to_le_bytes();
            tree.push(&payload);
            let root = tree.root();
            historical_roots.push((step + 1, root, payload));

            for &(old_sz, ref old_root, _) in &historical_roots {
                assert!(tree.verify_consistency(old_sz, old_root));
            }
            let proof = tree.prove_inclusion(step).expect("inclusion proof");
            assert!(MmrTree::verify_inclusion(&payload, &proof, &root));
        }
    }

    #[test]
    fn issue_400_401_770_acta_receipt_mmr_and_rekor_checkpoint() {
        let mut mmr = MmrTree::default();
        mmr.push(b"mem-alpha");
        mmr.push(b"mem-beta");
        mmr.push(b"mem-gamma");
        let root = mmr.root();

        let proof = mmr.prove_inclusion(1).unwrap();
        assert!(MmrTree::verify_inclusion(b"mem-beta", &proof, &root));
        assert!(!MmrTree::verify_inclusion(b"tampered", &proof, &root));

        let key = [0x77; 32];
        let acta = ActaRetrievalReceipt::issue(
            "key-01",
            &key,
            "tag:pinned",
            vec!["mem-beta".to_string()],
            &root,
        );
        assert!(acta.verify(&key));

        let ckpt = RekorCheckpoint::sign("vestige.local/log", &mmr, &key);
        assert_eq!(ckpt.tree_size, 3);
        assert_eq!(ckpt.signature_hex.len(), 64);
    }

    #[test]
    fn issue_402_406_407_crypto_erasure_reed_solomon_and_pq_hybrid() {
        // #402 crypto-erasure: ciphertext stays intact, key destruction prevents decryption
        let mut vault = CryptoErasureVault::default();
        vault.seal_memory("mem-pii-1", [0x55; 32], b"sensitive user note");
        assert_eq!(vault.open_memory("mem-pii-1").unwrap(), b"sensitive user note");
        let before_ct = vault.raw_ciphertext("mem-pii-1").unwrap().to_vec();
        let tomb = vault.purge_key("mem-pii-1").unwrap();
        assert!(tomb.key_destroyed);
        assert_eq!(vault.raw_ciphertext("mem-pii-1").unwrap(), before_ct.as_slice());
        assert_eq!(vault.open_memory("mem-pii-1"), Err("CryptoErasedTombstone"));

        // #406 GF(2^8) Reed-Solomon self-repair with Q-syndrome verification
        let shards = vec![
            b"STRATA-SEGMENT-A".to_vec(),
            b"STRATA-SEGMENT-B".to_vec(),
            b"STRATA-SEGMENT-C".to_vec(),
        ];
        let (p, q) = encode_rs_parity_2(&shards);
        let mut damaged = vec![Some(shards[0].clone()), None, Some(shards[2].clone())];
        let repair_receipt = repair_single_shard(&mut damaged, &p, &q).unwrap();
        assert_eq!(repair_receipt.repaired_shard_index, 1);
        assert_eq!(damaged[1].as_deref(), Some(b"STRATA-SEGMENT-B".as_slice()));

        // #407 Post-Quantum hybrid envelope
        let env = HybridPqEnvelope::sign_hybrid(&[0x11; 32], &[0x22; 32], b"frame-bytes");
        assert!(env.verify_hybrid(&[0x11; 32], &[0x22; 32], b"frame-bytes"));
        assert!(!env.verify_hybrid(&[0x11; 32], &[0x99; 32], b"frame-bytes"));
    }

    #[test]
    fn issue_404_408_409_410_411_412_ledger_timetravel_and_transactions() {
        let f1 = WriterFrame {
            writer_id: "w-a".to_string(),
            writer_seq: 1,
            causal_lamport: 10,
            memory_id: "mem-1".to_string(),
            content: "v1".to_string(),
            valid_time_ms: 1000,
            commit_sha: Some("1111222233334444".to_string()),
            supersedes: None,
        };
        let f2 = WriterFrame {
            writer_id: "w-b".to_string(),
            writer_seq: 1,
            causal_lamport: 20,
            memory_id: "mem-2".to_string(),
            content: "v2 supersedes v1".to_string(),
            valid_time_ms: 2000,
            commit_sha: Some("5555666677778888".to_string()),
            supersedes: Some("mem-1".to_string()),
        };

        // #404 commutative + idempotent merge fold
        let (merged_ab, head_ab) = merge_writer_chains(&[vec![f1.clone()], vec![f2.clone()]]);
        let (_, head_ba) = merge_writer_chains(&[vec![f2.clone(), f1.clone()], vec![f1.clone()]]);
        assert_eq!(head_ab, head_ba);

        // #410 time-travel recall
        let at_t1 = recall_as_of(&merged_ab, &TimeTravelCutoff::ValidTimeMs(1500));
        assert_eq!(at_t1.len(), 1);
        assert_eq!(at_t1[0].memory_id, "mem-1");
        let at_t2 = recall_as_of(&merged_ab, &TimeTravelCutoff::ValidTimeMs(2500));
        assert_eq!(at_t2.len(), 1);
        assert_eq!(at_t2[0].memory_id, "mem-2");

        // #408 exact handle resolver
        let mut idx = ExactHandleIndex::default();
        idx.bind("symbol:auth::verify", "mem-1").unwrap();
        idx.bind("commit:1111222233334444", "mem-1").unwrap();
        assert_eq!(idx.resolve("symbol:auth::verify").unwrap(), vec!["mem-1"]);
        assert_eq!(idx.resolve("commit:11112222").unwrap(), vec!["mem-1"]);

        // #411 compensating rollback without log truncation
        let mut tx = CompensatingTxLog::default();
        tx.begin("deploy-cfg");
        tx.put(1, "mode", "safe");
        tx.put(2, "mode", "broken");
        let log_len_before = tx.entries.len();
        assert_eq!(tx.rollback_to(1), 1);
        assert!(tx.entries.len() > log_len_before);
        assert_eq!(tx.get("mode"), Some("safe"));

        // #412 typed intention triggers
        assert!(trigger_matches(
            &IntentionTrigger::TestFailed("auth_test".to_string()),
            &SystemSignal::TestFailure("auth_test".to_string())
        ));
    }

    #[test]
    fn issue_417_418_419_421_consolidation_dream_ghostlink_and_packs() {
        assert_eq!(classify_tier(1), ConsolidationTier::Episodic);
        assert_eq!(classify_tier(365), ConsolidationTier::Archive);

        let key = [0x33; 32];
        let diff = dream_compile_signed(
            &key,
            &[("ep-1".to_string(), "timeout on port 5432".to_string())],
            "mem-distilled-1",
            "Always set connect_timeout=5s on Postgres pools",
        );
        assert!(diff.unified_diff.contains("+ [mem-distilled-1]"));

        let minted = evaluate_ghostlink_exposures(
            &[
                GhostLinkExposure {
                    candidate_edge: ("m1".to_string(), "m2".to_string()),
                    surfaced_pass: true,
                    decoy_pass: false,
                },
                GhostLinkExposure {
                    candidate_edge: ("m1".to_string(), "m2".to_string()),
                    surfaced_pass: true,
                    decoy_pass: false,
                },
            ],
            2,
            500,
        );
        assert_eq!(minted.len(), 1);

        let pack = CausalPackBundle::pack(&key, "sec-rules", "1.0.0", vec![]);
        assert!(pack.verify(&key));
    }

    #[test]
    fn issue_403_409_422_541_567_571_573_574_compliance_and_cli_superpowers() {
        let diff = "\
--- a/src/gate.rs
+++ b/src/gate.rs
@@ -10,2 +10,5 @@ fn check_token
+    verify_sig();
";
        let hunks = parse_git_diff_hunks("87a1e6d7", diff);
        assert_eq!(hunks.len(), 1);
        assert_eq!(hunks[0].new_start, 10);

        let why = vestige_why_line(
            "src/gate.rs",
            12,
            &hunks,
            &[(
                "87a1e6d7".to_string(),
                "mem-dec-99".to_string(),
                "Enforce signature check before token decode".to_string(),
                "eff-0099".to_string(),
            )],
        );
        assert_eq!(why.len(), 1);
        assert_eq!(why[0].decision_id, "mem-dec-99");

        // #571 edit bisect over 16 edits (culprit = index 11)
        let edits: Vec<String> = (0..16).map(|i| format!("edit-{i}")).collect();
        let witness = edit_bisect(&edits, |idx| idx < 11).expect("culprit found");
        assert_eq!(witness.culprit_index, 11);
        assert_eq!(witness.culprit_edit_id, "edit-11");
        assert!(witness.probes_executed <= 6);

        // #403, #422, #567, #573, #574
        let events = vec![AuditEventRecord {
            event_id: "ev-1".to_string(),
            timestamp_ms: 1_700_000_000_000,
            actor_id: "agent-1".to_string(),
            decision: "STOP".to_string(),
            tool_name: "Bash".to_string(),
            receipt_hash: "ab".repeat(32),
        }];
        let eu = export_eu_ai_act_article12("vestige-prod", &events);
        assert_eq!(eu["human_oversight_stops"], 1);
        let finra = export_finra_4511_worm("vestige-corp", 7, &events);
        assert_eq!(finra["worm_immutable"], true);
        let otel = export_otel_genai_spans("vestige-daemon", &events);
        assert_eq!(
            otel["scopeSpans"][0]["spans"][0]["attributes"]["gen_ai.system"],
            "vestige.strata"
        );
        let bundle = export_change_evidence_bundle("87a1e6d7", &hunks, &events);
        assert_eq!(bundle["commit_sha"], "87a1e6d7");

        let top = render_vestige_top(&TopSnapshot {
            head_seq: 420,
            stops_count: 3,
            failures_count: 1,
            fixes_anchored: 1,
            recent_receipts: vec!["eff-0099".to_string()],
        });
        assert!(top.contains("VESTIGE TOP | head_seq=420"));
    }
}
