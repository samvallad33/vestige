//! # strata-causal-math — Tranche 6 Waves 15–19 Retrieval Ceiling, Causal Posets,
//! Semirings, Hodge Topology, Physics & Historical Math (27 Issues)

pub mod wave15_ceiling;
pub mod wave16_algebra;
pub mod wave17_causal_poset;
pub mod wave19_physics_history;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::wave15_ceiling::*;
    use crate::wave16_algebra::*;
    use crate::wave17_causal_poset::*;
    use crate::wave19_physics_history::*;

    #[test]
    fn wave15_all_7_ceiling_and_verifier_issues() {
        let proof = prove_premise_coverage(
            &["auth.ttl".into(), "db.host".into()],
            &BTreeSet::from(["auth.ttl".to_string()]),
        );
        assert!(!proof.all_covered && proof.missing_premises == vec!["db.host"]); // #725

        let rep = evaluate_five_piece_ceiling(
            &["p1".into()],
            &[("p1".into(), true)], // superseded -> uncovered -> clean abstention!
        );
        assert!(rep.abstained_cleanly); // #726

        let (dig, win) = v12_preserve_user_window(&[(-1, "skip".into()), (0, "keep".into())]);
        assert_eq!(win, "keep");
        assert_eq!(dig.len(), 64); // #614

        assert_eq!(
            verify_post_generation("I don't know", true, &BTreeSet::new()),
            ReflexPostVerdict::FalseAbstentionRejected
        ); // #615
        assert_eq!(
            verify_post_generation("See mem-old", true, &BTreeSet::from(["mem-old".into()])),
            ReflexPostVerdict::OverruledCitationRejected("mem-old".into())
        );

        let scored = score_unified_benchmark(
            &[(vec!["m1".into(), "m2".into()], BTreeSet::from(["m1".into(), "m2".into()]))],
            &[true, true],
        );
        assert!((scored.all_at_5 - 1.0).abs() < 1e-9); // #595, #596, #619
    }

    #[test]
    fn wave16_all_8_algebra_and_optimization_issues_701_to_708() {
        let slots = submodular_greedy_slots(
            &[
                ("d1".into(), BTreeSet::from(["f1".into(), "f2".into()])),
                ("d2".into(), BTreeSet::from(["f3".into()])),
            ],
            2,
        );
        assert_eq!(slots, vec!["d1", "d2"]);
        let ds = sinkhorn_doubly_stochastic(&[vec![2.0, 1.0], vec![1.0, 3.0]], 25);
        assert!((ds[0][0] + ds[0][1] - 1.0).abs() < 1e-6); // #701

        // Diamond 0 -> {1, 2} -> 3: node 0 must dominate node 3!
        let dom = compute_dominators(0, 4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert!(dom[3].contains(&0) && !dom[3].contains(&1)); // #702

        let dist = tropical_all_pairs_shortest(3, &[(0, 1, 4), (1, 2, 5), (0, 2, 20)]);
        assert_eq!(dist[0][2], 9); // #703

        // Cyclic 3-loop (0->1: +1, 1->2: +1, 2->0: +1) has non-zero curl residual!
        let (_, curl_sq) = hodgerank_preference_decomposition_helper();
        assert!(curl_sq > 0.5); // #704

        let closed = galois_attribute_closure(
            &[("o1".into(), BTreeSet::from(["a".into(), "b".into()]))],
            &BTreeSet::from(["a".into()]),
        );
        assert!(closed.contains("b"));
        assert!(mdl_setup_turn_bits(20, 5, 10) < 0); // #705

        let mut eg = CongruenceEGraph::default();
        eg.add_app("hash", "tok_v1", "res_1");
        eg.add_app("hash", "tok_v2", "res_2");
        eg.merge_aliases("tok_v1", "tok_v2");
        assert!(eg.equiv("res_1", "res_2")); // #706

        let mut zs = ZSet::default();
        zs.apply_delta("edge-1", 1);
        zs.apply_delta("edge-1", -1);
        assert!(zs.weights.is_empty()); // #707

        let objs = BTreeSet::from(["A".to_string(), "B".to_string()]);
        let morphs = BTreeSet::from([
            ("A".into(), "A".into()),
            ("B".into(), "B".into()),
            ("A".into(), "B".into()),
        ]);
        assert!(verify_category_schema(&objs, &morphs)); // #708
    }

    fn hodgerank_preference_decomposition_helper() -> (Vec<f64>, f64) {
        hodgerank_potential_and_curl(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)])
    }

    #[test]
    fn wave17_all_5_causal_poset_issues() {
        // Collider `0 -> 1 <- 2`: d-separated when `{}` conditioned, ACTIVE when `{1}` conditioned!
        let collider = vec![(0, 1), (2, 1)];
        assert!(bayes_ball_d_separated(3, &collider, 0, 2, &BTreeSet::new()));
        assert!(!bayes_ball_d_separated(3, &collider, 0, 2, &BTreeSet::from([1]))); // #727

        let (new_n, swig_edges) = swig_node_split(3, &[(0, 1), (1, 2)], 1);
        assert_eq!(new_n, 4);
        assert_eq!(swig_edges, vec![(0, 1), (3, 2)]); // #730

        let mb = markov_blanket(4, &[(0, 1), (1, 2), (3, 2)], 1);
        assert_eq!(mb, BTreeSet::from([0, 2, 3])); // #731

        // Chain poset `0 <= 1 <= 2`: mu(0,1) = -1, mu(0,2) = 0!
        let leq = vec![
            vec![true, true, true],
            vec![false, true, true],
            vec![false, false, true],
        ];
        let mu = rota_moebius_matrix(&leq);
        assert_eq!(mu[0][1], -1);
        assert_eq!(mu[0][2], 0); // #732

        assert_eq!(alexandrov_interval(&leq, 0, 2), vec![0, 1, 2]);
        assert_eq!(longest_causal_chain_len(3, &[(0, 1), (1, 2)]), 3); // #734
    }

    #[test]
    fn wave19_all_7_physics_chemistry_topology_and_history_issues() {
        // Destructive interference: phase 0 (+1) + phase 2 (-1) = (0, 0)
        assert_eq!(z4_path_integral_sum(&[0, 2]), (0, 0)); // #739

        let focus = dort_dominant_cause_focus(&[vec![3.0, 0.0], vec![0.0, 1.0]], 20);
        assert!(focus[0].abs() > 0.99); // #740

        // Reversible pair `A <-> B`: n=2 complexes, l=1 linkage class, s=1 -> deficiency = 0!
        let delta = feinberg_deficiency(
            2,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &[(0, 1), (1, 0)],
        );
        assert_eq!(delta, 0); // #741

        let lap1 = hodge_1_laplacian_apply(
            3,
            &[(0, 1), (1, 2), (0, 2)],
            &[(0, 1, 2)],
            &[1.0, 1.0, 1.0],
        );
        assert_eq!(lap1.len(), 3); // #742

        let (coarse, cap) = renormalize_haystack_pyramid(&[0.2, 0.8, 0.9], &BTreeSet::from([2]), 0.7);
        assert_eq!(coarse.len(), 2);
        assert_eq!(cap, vec![2]); // #743

        let mut tp = TorrensPacioliRegister::default();
        tp.transfer_title("slot:db_port", "mem-deed-6432");
        assert_eq!(tp.current_deed("slot:db_port"), Some("mem-deed-6432"));
        assert!(tp.trial_balance_ok()); // #746

        let khipu = encode_khipu_rollup(vec![12, 305, 8]);
        assert_eq!(khipu.top_cord_rollup, 325);
        assert!(verify_khipu_audit(&khipu)); // #751
    }
}
