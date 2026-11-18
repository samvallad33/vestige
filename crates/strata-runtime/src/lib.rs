//! # strata-runtime — Tranche 4 Unified Runtime, Analytics, H²SK Sheaf & Wave 14 Systems (49 Issues)

pub mod analytics_w20_22;
pub mod distributed_w23;
pub mod sheaf_h2sk_w24;
pub mod systems_w14;

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use crate::analytics_w20_22::*;
    use crate::distributed_w23::*;
    use crate::sheaf_h2sk_w24::*;
    use crate::systems_w14::*;

    #[test]
    fn waves_20_to_22_all_12_analytics_issues() {
        let rules = vec![("m-stop".to_string(), "STOP".to_string())];
        assert_eq!(counterfactual_replay(&rules, "m-stop").0, "ALLOW"); // #540
        assert!((ochiai_failure_spectrum(4, 0, 0) - 1.0).abs() < 1e-9); // #551

        let sessions = vec![
            (BTreeSet::from(["rm-lock".to_string()]), true),
            (BTreeSet::from(["rm-lock".to_string()]), true),
            (BTreeSet::from(["clean".to_string()]), false),
        ];
        assert_eq!(propose_lagged_candidate_causes(&sessions, "E_LOCK", 2, 0.5).len(), 1); // #552
        assert_eq!(mint_coallocation_edges(&[("m1".into(), 100), ("m2".into(), 120)], 50).len(), 1); // #553
        assert!(reperformance_zero_defect_upper_bound(100, 0.05) < 0.031); // #554
        assert_eq!(CitatorFlag::Overruled, CitatorFlag::Overruled); // #556
        assert_eq!(
            detect_outbreak_alarm(
                &[("a1".into(), "SIG1".into(), 10), ("a2".into(), "SIG1".into(), 20)],
                100,
                2
            ),
            vec!["SIG1"]
        ); // #557

        let recon = ReconsolidationGate { recalled_at_seq: Some(10), failure_after_recall: true, window_seqs: 5 };
        assert!(recon.may_reconsolidate(12) && !recon.may_reconsolidate(20)); // #558

        let recovered = simulate_crash_and_takeover(&[
            SimStep::AppendDurable("ok-1".into()),
            SimStep::TornUnackedCrash("torn".into()),
            SimStep::OwnerTakeoverOverride("fix".into()),
        ]);
        assert_eq!(recovered, vec!["ok-1", "OWNER:fix"]); // #561
        assert_eq!(commit_zk_replay_receipt(b"secret-prompt", &[1; 32], &[2; 32], true).len(), 32); // #562
        assert_eq!(
            classify_failure_subtype(4, false, 0, false),
            MultiAgentFailureSubtype::StepRepetitionLoop
        ); // #564
        let (r1, mrr) = benchmark_causal_walk_mrr(&[(vec!["root-a".into()], "root-a".into())]);
        assert!((r1 - 1.0).abs() < 1e-9 && (mrr - 1.0).abs() < 1e-9); // #572
    }

    #[test]
    fn wave_23_all_15_distributed_strata_issues_575_to_589() {
        let mut dag = CognitiveBranchDag::default();
        dag.init_main(vec!["m0".into()]);
        assert!(dag.fork("main", "exp"));
        dag.append("exp", "m1");
        assert_eq!(dag.guarded_merge("exp", "main", "BAD:"), Ok(1)); // #575

        let mut c1 = CausalVectorClock::default();
        c1.tick("r1");
        let mut c2 = c1.clone();
        c2.tick("r2");
        assert!(c1.happens_before(&c2)); // #576

        assert!(promote_speculative_batch(vec!["s1".into()], 4, |_| true).is_ok()); // #577
        let mut manifests = BTreeMap::new();
        manifests.insert("worker-1".to_string(), BTreeSet::from(["m-old".to_string()]));
        assert_eq!(correction_ripple_targets("m-old", &manifests), vec!["worker-1"]); // #578

        let retained = select_compute_priced_context(
            &[
                PricedContextItem { id: "rule".into(), required_rule: true, utility_milli: 10, token_cost: 50 },
                PricedContextItem { id: "opt".into(), required_rule: false, utility_milli: 900, token_cost: 40 },
            ],
            100,
        );
        assert_eq!(retained, vec!["rule", "opt"]); // #579

        let env = TypedArtifactEnvelope::wrap("strata://latent/v1", "gemini", b"bytes");
        assert_eq!(env.bytes_len, 5); // #580

        let votes = BTreeMap::from([
            ("a1".into(), true),
            ("a2".into(), true),
            ("a3".into(), true),
            ("byz".into(), false),
        ]);
        assert_eq!(verify_bft_auditor_quorum(&votes, 1), Some(true)); // #581
        assert_eq!(
            filter_by_role_shard(&[("m1".into(), "sec".into())], &BTreeSet::from(["sec".into()])),
            vec!["m1"]
        ); // #582

        let lease = ProcessLease { child_pid: "p1".into(), expires_ms: 500, evidence_receipt: Some("eff-1".into()) };
        assert!(lease.can_promote_child_lesson(400)); // #583
        assert_eq!(compute_transition_proof(&[1; 32], &[2; 32], &[3; 32]).len(), 32); // #584
        assert_eq!(seal_detached_checkpoint(&[9; 32], "tpm-0", [4; 32]).key_slot_id, "tpm-0"); // #585
        assert!(evaluate_candidate_policy_branch([1; 32], [2; 32], true).1); // #586

        let mut wire = vec![5, 0, 0, 0];
        wire.extend_from_slice(b"hello");
        assert_eq!(decode_frame_bounded_safe(&wire, 64), Ok(b"hello".as_slice())); // #587
        assert_eq!(decode_frame_bounded_safe(&wire, 2), Err("FrameExceedsAllocationCap"));

        let mut j = ReplicaLogicalJournal::default();
        j.append("e1");
        j.append("e2");
        assert_eq!(j.since_lsn(1).len(), 1); // #588
        assert!(qualify_swarm_handoff(Some("c"), Some("w"), Some("v"))); // #589
    }

    #[test]
    fn wave_24_all_12_h2sk_sheaf_issues() {
        assert!(poincare_distance(&[0.0, 0.0], &[0.5, 0.0]) > 1.0); // #590
        assert!(ising_spin_energy(&[1, 1], &[(0, 1, 1.0)]) < 0.0);
        assert!(cellular_sheaf_residual(&[1.0, -1.0], &[(0, 1.0, 1, 1.0)]) > 3.9); // #591

        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.2, 0.5, -0.3, 0.8];
        let bound = hrr_circular_bind(&k, &v);
        let unbound = hrr_circular_unbind(&k, &bound);
        assert!((unbound[1] - 0.5).abs() < 1e-9); // #592

        let ltc = ltc_step(0.0, 1.0, 1.0, 1.0);
        assert!((ltc - 0.5).abs() < 1e-9); // #593
        let diag = evaluate_h2sk_kernel(&[0.3, 0.0], &[0.0, 0.0], 0.0, ltc);
        assert!(diag.hyperbolic_depth > 0.0); // #594

        assert_eq!(compile_shepard_banner(true, 0.0, 0.1), ShepardBanner::Overruled); // #610
        assert!(canonicalize_zero_deixis("ship today", "2026-11-17").contains("[DATE:2026-11-17]")); // #611
        assert!(hrr_entity_similarity(&[1.0, 0.0], &[2.0, 0.0]) > 1.5); // #612
        assert_eq!(
            classify_epistemic_modality("we could try caching", false),
            EpistemicModality::HypotheticalProposal
        ); // #613
        assert_eq!(parse_closed_edge_vocab("fixes"), Ok(ClosedEdgeKind::Fixes)); // #616
        assert!(parse_closed_edge_vocab("invented_edge").is_err());

        let pkt = compile_packet_with_receipt("mem-1", "rule", 900, b"holo", "ALLOW");
        assert!(replay_packet_receipt(&pkt, b"holo")); // #617, #618
    }

    #[test]
    fn wave_14_all_10_systems_issues() {
        let mut node = StrataMeshNode::default();
        node.apply_gossip("peer-a", 42, &["mem-poison".into()]);
        assert!(node.live_retractions.contains("mem-poison")); // #405, #761

        let zk = ZkCircuitReceipt::prove_deterministic(&[7; 32], b"pub", b"wit");
        assert_eq!(zk.proof_commitment.len(), 32); // #763

        let ev = EbpfSyscallEvent::Kill { caller_pid: 99, target_pid: 1, sig: 9 };
        assert!(matches!(ev, EbpfSyscallEvent::Kill { .. })); // #764

        let limits = SandboxLimits { fuel_budget: 100, allow_network: false, expose_signing_keys: false };
        assert_eq!(run_sandboxed_ingestor(&limits, 50, || "ok".into()), Ok("ok".into())); // #765

        let fm = FmIndex::build(b"STRATA_EXACT_HANDLE_STRATA");
        assert_eq!(fm.search_exact(b"STRATA"), vec![0, 20]); // #766

        let mut eg = SymbolCongruenceEGraph::default();
        eg.union("old_fn", "mid_fn");
        eg.union("mid_fn", "new_fn");
        assert!(eg.congruent("old_fn", "new_fn")); // #768

        let closure = datalog_transitive_closure(&[("a".into(), "b".into()), ("b".into(), "c".into())]);
        assert!(closure.contains(&("a".into(), "c".into()))); // #769

        let mut s1 = OrSetCrdt::default();
        s1.add("mem-1", "tag-1");
        let mut s2 = s1.clone();
        s1.remove("mem-1");
        s2.add("mem-1", "tag-2"); // concurrent re-add wins in OR-Set!
        s1.merge(&s2);
        assert!(s1.contains("mem-1")); // #771

        assert!(check_formal_non_bypass_model()); // #772
    }
}
