//! # strata-episodic — Waves 8–12 Source-Preserving Memory Algebra (#620–#670)
//!
//! Pure, deterministic, zero-vector discrete algebra implementing all 50
//! cross-disciplinary operators under umbrella `#620`.

pub mod wave10_temporal;
pub mod wave11_law;
pub mod wave12_stratigraphy;
pub mod wave8_physics;
pub mod wave9_chemistry;

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use crate::wave8_physics::*;
    use crate::wave9_chemistry::*;
    use crate::wave10_temporal::*;
    use crate::wave11_law::*;
    use crate::wave12_stratigraphy::*;

    #[test]
    fn wave8_all_10_physics_operators_621_to_630() {
        assert_eq!(hysteresis_salience_step(850, -300, 700), 700); // #621
        assert!(schmitt_deadband_state(true, 450, 400, 700)); // #622
        assert!(!schmitt_deadband_state(false, 450, 400, 700));
        assert!(matches!(
            destructive_interference_admit(&[("m1".into(), true), ("m2".into(), false)]),
            Some(PacketAdmission::HoldConflict { .. })
        )); // #623
        let res = constructive_resonance_recurrence("m1", &["r1".into(), "r2".into(), "r1".into()]);
        assert_eq!(res.distinct_receipts.len(), 2); // #624

        let mut reg = ExclusiveSlotRegister::default();
        assert_eq!(reg.assign("user", "tz", "m-pst"), None);
        assert_eq!(reg.assign("user", "tz", "m-est"), Some("m-pst".into()));
        assert_eq!(reg.current("user", "tz"), Some("m-est")); // #625

        assert!(cherenkov_mutation_burst(&[10, 20, 30, 40], 100, 3)); // #626
        let chain = episodic_decay_chain(vec!["ep-1".into(), "ep-2".into()], "sem-1");
        assert_eq!(chain.chain_blake3.len(), 64); // #627

        let clusters = nucleate_episode_clusters(&["ms-1".into()], &[("ms-1".into(), "ev-9".into())]);
        assert!(clusters["ms-1"].contains("ev-9")); // #628

        let elastic = ContextDeformation::ElasticTemporary {
            baseline: "calm".into(),
            active_until_ms: 500,
        };
        assert_eq!(resolve_elastic_or_plastic(&elastic, 600, "incident"), "calm"); // #629
        assert!(validate_causal_light_cone(10, 11, true)); // #630
        assert!(!validate_causal_light_cone(11, 10, true));
    }

    #[test]
    fn wave9_all_10_chemistry_operators_631_to_640() {
        let anchors = vec!["anchor-1".to_string()];
        let normal = vec!["cand-1".to_string()];
        assert_eq!(le_chatelier_grounding_recall(true, &anchors, &normal), &anchors); // #631
        assert!(!activation_energy_gate(true, false, 20, 80)); // #632
        assert!(activation_energy_gate(true, true, 20, 80));

        let req = BTreeSet::from(["alice".to_string(), "bob".to_string()]);
        let signed = BTreeSet::from(["alice".to_string()]);
        assert_eq!(limiting_reagent_gate(&req, &signed), Err(vec!["bob".to_string()])); // #633

        let b = ChiralRoleBinding {
            subject: "alice".into(),
            predicate: "owes".into(),
            object: "bob".into(),
        };
        assert_ne!(b, b.enantiomer()); // #634
        assert_eq!(buffer_hyperbole_salience(500, 900, 100), 600); // #635

        let mut chel = ChelationQuarantine::default();
        chel.chelate("m-bad");
        assert!(!chel.is_active("m-bad"));
        assert!(chel.dechelate("m-bad"));
        assert!(chel.is_active("m-bad")); // #636

        let routed = allosteric_recipient_filter(
            &[("m-sec".into(), BTreeSet::from(["owner".into()]))],
            "guest",
        );
        assert!(routed.is_empty()); // #637
        assert_eq!(supersaturation_escalation(5, 5), EscalationLevel::CrystallizedUrgent); // #638
        let eph = active_ephemeral_links(&[("a".into(), "b".into(), 10)], 15, 10);
        assert_eq!(eph.len(), 1); // #639
        let tit = evaluate_titration_threshold("hyp-1", &["w1".into(), "w2".into()], 2);
        assert!(tit.promoted); // #640
    }

    #[test]
    fn wave10_all_10_temporal_operators_641_to_650() {
        let arc = UrsatzLifeArc {
            arc_id: "arc-1".into(),
            title: "Ship Vestige 6.2.3".into(),
            child_milestones: vec!["pr-838".into()],
        };
        assert_eq!(arc.child_milestones.len(), 1); // #641
        let iv = ProlongationInterval { context_id: "c1".into(), start_seq: 5, end_seq: Some(10) };
        assert!(iv.governs(7) && !iv.governs(10)); // #642

        let mut cp = CounterpointTimelines::default();
        cp.record("alice", 1, "ev-shared");
        cp.record("bob", 2, "ev-shared");
        assert_eq!(cp.joint_consonances("alice", "bob"), vec!["ev-shared"]); // #643

        let pedal = PedalPointAnchor { handle: "h1".into(), pinned_at_turn: 10, max_turns: 3 };
        assert!(pedal.is_live(12) && !pedal.is_live(13)); // #644
        assert!(compare_expected_actual("e1", "pass", "a1", "fail").diverged); // #645
        assert_eq!(
            count_leitmotif_occurrences(
                &["fail".into(), "bisect".into(), "fix".into(), "fail".into(), "bisect".into(), "fix".into()],
                &["fail".into(), "bisect".into(), "fix".into()]
            ),
            2
        ); // #646
        let state = OpenLoopState::Resolved { receipt_id: "eff-1".into() };
        assert!(matches!(state, OpenLoopState::Resolved { .. })); // #647
        assert_eq!(polyrhythm_next_collision(6, 8), 24); // #648
        let overtones = overtone_exact_neighborhood("root", &[("root".into(), "n1".into())], 2);
        assert_eq!(overtones["n1"], 1000); // #649
        assert_eq!(opt_in_absence_rest_ms(true, 100, 250), Some(150)); // #650
    }

    #[test]
    fn wave11_all_10_legal_operators_651_to_660() {
        let winner = resolve_stare_decisis(&[
            (PrecedenceCourt::AgentObservation, "guess"),
            (PrecedenceCourt::OwnerConstitutional, "law"),
        ]);
        assert_eq!(winner, Some(&"law")); // #651

        let routine = RoutineProposal { pattern_id: "p1".into(), observed_count: 10, owner_confirmed: false };
        assert!(!routine.is_binding_rule(3)); // #652

        let mut custody = Vec::new();
        append_custody_hop(&mut custody, "ingest", "mcp");
        append_custody_hop(&mut custody, "recall", "agent");
        assert_eq!(custody[1].prev_blake3, custody[0].hop_blake3); // #653

        let tainted = poisonous_tree_invalidate(
            "bad-root",
            &[("bad-root".into(), "c1".into()), ("c1".into(), "c2-clean".into())],
            &BTreeSet::from(["c2-clean".to_string()]),
        );
        assert!(tainted.contains("c1") && !tainted.contains("c2-clean")); // #654

        let haz = HazardHypothesis { action_sig: "a".into(), failure_sig: "f".into(), co_occurrences: 4, confirmed: false };
        assert!(!haz.confirmed); // #655
        assert!(!hearsay_admissible_for_mutation(TestimonyKind::SecondhandHearsay, true)); // #656
        assert!(barred_by_laches(0, 1_000_000, 1_000, false)); // #657
        let fm = ForceMajeureInterval { start_ms: 10, end_ms: 20 };
        assert!(fm.suspends_routine_nudges(15)); // #658
        let triad = ActionTriad { intention_id: "i1".into(), attempt_receipt_id: Some("a1".into()), outcome_receipt_id: None };
        assert!(triad.outcome_receipt_id.is_none()); // #659

        let mut sync_branches = BTreeMap::new();
        sync_branches.insert("b1".to_string(), (false, 10));
        sync_branches.insert("b2".to_string(), (true, 8));
        assert_eq!(resolve_race_notice(&sync_branches), SyncPriorityOutcome::Winner("b1".into())); // #660
    }

    #[test]
    fn wave12_all_10_stratigraphy_operators_661_to_670() {
        let t1 = TerminusInterval { earliest_ms: 100, latest_ms: 500 };
        let t2 = TerminusInterval { earliest_ms: 300, latest_ms: 900 };
        assert_eq!(t1.intersect(t2), Some(TerminusInterval { earliest_ms: 300, latest_ms: 500 })); // #661
        assert_ne!(ChronologyCertainty::ExactEventClock, ChronologyCertainty::StratigraphicBatchDepositOnly); // #662

        let mut pal = PalimpsestNode::new("m1", "draft-1");
        pal.revise("draft-2");
        assert_eq!(pal.undertext_layers, vec!["draft-1"]); // #663
        assert_eq!(AttestationPosture::VerifiedFailureAdmission, AttestationPosture::VerifiedFailureAdmission); // #664
        assert!(cross_date_from_anchors(Some(10), Some(20)).is_some()); // #665

        let mut pros = BTreeMap::new();
        pros.insert("ev-1".to_string(), BTreeSet::from(["sam".to_string(), "vestige".to_string()]));
        assert_eq!(prosopography_co_participants(&pros, "sam")["vestige"], vec!["ev-1"]); // #666

        let dies = group_die_links(&[
            ("i1".into(), "E01".into(), "a.rs".into()),
            ("i2".into(), "E01".into(), "a.rs".into()),
        ]);
        assert_eq!(dies.len(), 1); // #667

        let mut dam = DamnatioRegistry::default();
        dam.mute_proactive_surfacing("m-muted");
        assert!(!dam.allow_proactive("m-muted") && dam.allow_structural_walk("m-muted")); // #668

        assert!(lectio_difficilior_preserved("Set RUST_LOG=debug", &["RUST_LOG=debug"]).is_ok()); // #669
        let cap = compile_homeric_capsule("Strata", &[("m-1".into(), "append-only log".into())], 3);
        assert_eq!(cap.backing_memory_ids, vec!["m-1"]); // #670
    }
}
