//! Mirrored unit tests: `graph_tests` (10 tests).

use super::*;


#[test]
fn knowledge_update_prefers_newest_state() {
    let req = RankRequest {
        sessions: vec![
            Session { id: "old".into(), date: "2024/01/10 (Wed) 09:15".into(), timestamp: 0, turns: vec![turn("user", "My brother just moved to Seattle for a new job.")] },
            Session { id: "noise".into(), date: "2024/02/01 (Thu) 09:15".into(), timestamp: 0, turns: vec![turn("user", "I baked sourdough bread this weekend.")] },
            Session { id: "new".into(), date: "2024/03/22 (Fri) 18:40".into(), timestamp: 0, turns: vec![turn("user", "My brother hated the rain in Seattle so he relocated to Austin, Texas.")] },
        ],
        queries: vec![Query { text: "Where does my brother currently live?".into(), date: String::new() }],
        top_k: 3,
        params: Params::default(),
    };
    let r = rank(&req);
    assert!(! r.results[0].intent.current_state);
    // Without a current-state phrase, the earlier Seattle mention outranks the move.
    assert_eq!(r.results[0].sessions[0].id, "old");
}

#[test]
fn answer_turn_inherits_question_entities() {
    let req = RankRequest {
        sessions: vec![Session {
            id: "D1".into(),
            date: String::new(),
            timestamp: 0,
            turns: vec![
                Turn { id: "D1:1".into(), role: "Caroline".into(), content: "I went hiking with my dog.".into(), ..Turn::default() },
                Turn { id: "D1:2".into(), role: "Melanie".into(), content: "Are you still painting that sunrise landscape?".into(), ..Turn::default() },
                Turn { id: "D1:3".into(), role: "Caroline".into(), content: "Yeah, I switched to acrylics for it yesterday.".into(), ..Turn::default() },
                Turn { id: "D1:4".into(), role: "Melanie".into(), content: "My kids love the beach.".into(), ..Turn::default() },
            ],
        }],
        queries: vec![Query { text: "What medium did Caroline use for her sunrise landscape?".into(), date: String::new() }],
        top_k: 5,
        params: Params::default(),
    };
    let r = rank(&req);
    let top2: Vec<&str> = r.results[0].turns.iter().take(2).map(|t| t.id.as_str()).collect();
    assert!(top2.contains(&"D1:3"), "{top2:?}");
}

#[test]
fn ppr_turn_scores_remain_bounded_and_length_fair() {
    let filler_essay = "flask app code ".repeat(120);
    let req = RankRequest {
        sessions: vec![
            Session { id: "s1".into(), date: String::new(), timestamp: 0, turns: vec![
                Turn { id: "gold".into(), role: "user".into(), content: "My first sprint ends on March 29 for login features.".into(), ..Turn::default() },
                Turn { id: "reply".into(), role: "assistant".into(), content: format!("Here is a sprint plan ending March 29: {filler_essay}"), ..Turn::default() },
            ]},
            Session { id: "s2".into(), date: String::new(), timestamp: 0, turns: vec![
                Turn { id: "noise".into(), role: "assistant".into(), content: format!("Another unrelated deployment guide: {filler_essay}"), ..Turn::default() },
            ]},
            Session { id: "s3".into(), date: String::new(), timestamp: 0, turns: vec![
                Turn { id: "noise2".into(), role: "user".into(), content: "Unrelated question about SQLite migrations.".into(), ..Turn::default() },
                Turn { id: "noise3".into(), role: "assistant".into(), content: "Sure, here are SQLite migration commands.".into(), ..Turn::default() },
                Turn { id: "noise4".into(), role: "user".into(), content: "Thanks!".into(), ..Turn::default() },
            ]},
        ],
        queries: vec![Query { text: "When does my first sprint end?".into(), date: String::new() }],
        top_k: 5,
        params: Params::default(),
    };
    let top2: Vec<String> = rank(&req).results[0].turns.iter().take(2).map(|t| t.id.clone()).collect();
    assert!(top2.contains(&"gold".to_string()), "{top2:?}");
    assert_ne!(rank(&req).results[0].turns[0].id, "noise");
}

#[test]
fn session_multihop_bridges_second_hop() {
    let req = RankRequest {
        sessions: vec![
            Session { id: "hop1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Kiss and Tell is a 1945 film starring Shirley Temple as Corliss Archer.")] },
            Session { id: "dup1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "A Kiss for Corliss is the sequel comedy film featuring Corliss Archer.")] },
            Session { id: "dup2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Meet Corliss Archer is a radio sitcom centered on Corliss Archer.")] },
            Session { id: "dup3".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Janet Waldo voiced Corliss Archer in the radio broadcast.")] },
            Session { id: "dup4".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "F. Hugh Herbert created Corliss Archer as a character.")] },
            Session { id: "hop2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Shirley Temple served as United States Chief of Protocol and ambassador to Ghana.")] },
            Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Baking sourdough bread at high altitude requires extra hydration.")] },
        ],
        queries: vec![Query { text: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?".into(), date: String::new() }],
        top_k: 5,
        params: Params::default(),
    };
    let top5: Vec<String> = rank(&req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
    assert_eq!(top5[0], "hop1");
    assert!(top5.contains(&"hop2".to_string()), "Expected hop2 bridged into top5: {top5:?}");
}

#[test]
fn previous_state_matches_and_zero_hit_bridge_skips_preference() {
    let req = RankRequest {
        sessions: vec![
            Session { id: "gold".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "I worked as a marketing specialist in my previous role at a startup.")] },
            Session { id: "d1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Here is a general list of common occupation titles across industries.")] },
            Session { id: "d2".into(), date: String::new(), timestamp: 0, turns: vec![turn("assistant", "Waiters and teachers are common occupation categories.")] },
        ],
        queries: vec![Query { text: "What was my previous occupation?".into(), date: String::new() }],
        top_k: 3,
        params: Params::default(),
    };
    assert_eq!(rank(&req).results[0].sessions[0].id, "gold");

    let pref_req = RankRequest {
        sessions: vec![
            Session { id: "s1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Can you recommend a great show to watch on Paramount Plus tonight?")] },
            Session { id: "zero".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Paramount Plus runs on Apple TV and Roku devices.")] },
            Session { id: "pref".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "I love comedy specials and standup shows.")] },
        ],
        queries: vec![Query { text: "Can you recommend a show or movie for me to watch tonight?".into(), date: String::new() }],
        top_k: 2,
        params: Params::default(),
    };
    let top2: Vec<String> = rank(&pref_req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
    assert!(!top2.contains(&"zero".to_string()), "zero-overlap bridge should not enter preference top2: {top2:?}");
}

#[test]
fn short_reply_inherits_the_neighbor_turn() {
    let req = RankRequest {
        sessions: vec![Session {
            id: "chat".into(),
            date: String::new(),
            timestamp: 1,
            turns: vec![
                turn("alex", "The walnut cabinet needs another coat of oil."),
                turn("blair", "Done."),
                turn("alex", "Unrelated note about train tickets and a station platform."),
            ],
        }],
        queries: vec![Query { text: "Who finished the walnut cabinet oil?".into(), date: String::new() }],
        top_k: 2,
        params: Params::default(),
    };
    let ranked = rank(&req);
    let ids: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.id.as_str()).collect();
    assert!(ids.iter().any(|id| id.ends_with("#1")), "short reply missed the neighbor words: {ids:?}");
}

#[test]
fn single_token_entity_and_header_title_bridge_second_hop() {
    let req = RankRequest {
        sessions: vec![
            Session { id: "hop1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The Harbor Glassworks exhibition was curated by Mateo in Lisbon.")] },
            Session { id: "d1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Harbor Glassworks opened a small gift counter near the ferry.")] },
            Session { id: "d2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Visitors praised the Harbor Glassworks lighting display.")] },
            Session { id: "d3".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Tickets for Harbor Glassworks sold out by noon.")] },
            Session { id: "d4".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Catalogs from Harbor Glassworks listed twelve bowls.")] },
            Session { id: "hop2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Mateo (ceramist)\nMateo trained at the Porto Kiln Guild before curating glass.")] },
            Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Sourdough starter hydration depends on flour protein.")] },
        ],
        queries: vec![Query { text: "Where did the curator of the Harbor Glassworks exhibition train?".into(), date: String::new() }],
        top_k: 5,
        params: Params::default(),
    };
    let top5: Vec<String> = rank(&req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
    assert!(top5.contains(&"hop2".to_string()), "single-token proper bridge missed: {top5:?}");
}

#[test]
fn exact_edge_replaces_a_weaker_slot_without_shared_words() {
    let req = RankRequest {
        sessions: vec![
            Session {
                id: "now".into(),
                date: String::new(),
                timestamp: 2,
                turns: vec![Turn {
                    id: "trigger".into(),
                    role: "user".into(),
                    content: "The ledger key is cedar.".into(),
                    links: vec![TurnLink { kind: "backfill".into(), to: "earlier".into(), degree: 2 }],
                    stability: 4.0,
                }],
            },
            Session {
                id: "then".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![Turn {
                    id: "earlier".into(),
                    role: "user".into(),
                    content: "quartz violin under the stairs".into(),
                    stability: 9.0,
                    ..Turn::default()
                }],
            },
            Session { id: "n1".into(), date: String::new(), timestamp: 3, turns: vec![turn("user", "marble compass on the table")] },
            Session { id: "n2".into(), date: String::new(), timestamp: 4, turns: vec![turn("user", "linen harbor in the fog")] },
            Session { id: "n3".into(), date: String::new(), timestamp: 5, turns: vec![turn("user", "pewter lantern by the door")] },
            Session { id: "n4".into(), date: String::new(), timestamp: 6, turns: vec![turn("user", "silk orchard past the gate")] },
        ],
        queries: vec![Query { text: "Where is the ledger key?".into(), date: String::new() }],
        top_k: 5,
        params: Params::default(),
    };
    let ranked = rank(&req);
    let ids: Vec<String> = ranked.results[0].turns.iter().map(|t| t.id.clone()).collect();
    assert!(ids.iter().any(|id| id == "trigger"), "{ids:?}");
    assert!(ids.iter().any(|id| id == "earlier"), "exact edge missed the zero-overlap turn: {ids:?}");
}

#[test]
fn a_backward_edge_lands_inside_the_first_five_not_the_tail() {
    let mut sessions = vec![
        Session {
            id: "now".into(),
            date: String::new(),
            timestamp: 20,
            turns: vec![Turn {
                id: "trigger".into(),
                role: "user".into(),
                content: "The ledger key is cedar.".into(),
                links: vec![TurnLink { kind: "backfill".into(), to: "earlier".into(), degree: 2 }],
                ..Turn::default()
            }],
        },
        Session {
            id: "then".into(),
            date: String::new(),
            timestamp: 1,
            turns: vec![Turn {
                id: "earlier".into(),
                role: "user".into(),
                content: "quartz violin under the stairs".into(),
                ..Turn::default()
            }],
        },
    ];
    for i in 0..8 {
        sessions.push(Session {
            id: format!("f{i}"),
            date: String::new(),
            timestamp: 2 + i,
            turns: vec![turn("user", &format!("ledger {}", "padding ".repeat(40)))],
        });
    }
    let req = RankRequest {
        sessions,
        queries: vec![Query { text: "Where is the ledger key?".into(), date: String::new() }],
        top_k: 10,
        params: Params::default(),
    };
    let ranked = rank(&req);
    let rows: Vec<String> = ranked.results[0].turns.iter().map(|t| format!("{}:{:.3}", t.id, t.score)).collect();
    let pos = rows.iter().position(|row| row.starts_with("earlier:"));
    assert!(pos.is_some_and(|p| p < 5), "backward edge parked in the tail: {rows:?}");
}

#[test]
fn wordnet_hypernym_bridges_a_query_word_the_corpus_never_used() {
    let req = RankRequest {
        sessions: vec![
            Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "We repainted the fence and planted tomatoes.")] },
            Session { id: "gold".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "We brought home a puppy from the shelter.")] },
        ],
        queries: vec![Query { text: "Where did we get our dog?".into(), date: String::new() }],
        top_k: 2,
        params: Params::default(),
    };
    let ranked = rank(&req);
    assert_eq!(ranked.results[0].sessions[0].id, "gold", "{:?}", ranked.results[0].sessions);
    assert!(ranked.results[0].sessions[0].score > 0.0);

    // Control (1): any lexical overlap closes the gate, so "dog" is not expanded
    // and the turn that shares "fence" wins.
    let overlap = RankRequest {
        queries: vec![Query { text: "Did the dog jump the fence?".into(), date: String::new() }],
        ..req
    };
    assert_eq!(rank(&overlap).results[0].sessions[0].id, "noise");
}

#[test]
fn issue_854_outcome_conductance_redirects_current_with_receipts_and_bit_exact_replay() {
    use crate::retrieval::graph::ghostcurrent_mdl_q16::{GhostCurrentInput, run_ghostcurrent_mdl_q16};

    let make_corpus = |links_on_seed: Vec<TurnLink>| -> Vec<Session> {
        vec![
            Session {
                id: "s0".into(),
                date: String::new(),
                timestamp: 100,
                turns: vec![Turn {
                    id: "t0".into(),
                    role: "user".into(),
                    content: "CobaltRelay reported a timeout failure.".into(),
                    links: links_on_seed,
                    stability: 0.0,
                }],
            },
            Session {
                id: "s1".into(),
                date: String::new(),
                timestamp: 200,
                turns: vec![Turn {
                    id: "t1".into(),
                    role: "user".into(),
                    content: "CobaltRelay path checked AlphaBreaker module.".into(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
            Session {
                id: "s2".into(),
                date: String::new(),
                timestamp: 300,
                turns: vec![Turn {
                    id: "t2".into(),
                    role: "user".into(),
                    content: "CobaltRelay path checked BetaBreaker module.".into(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
        ]
    };

    let run_gc = |sessions: &[Session]| {
        let p = Params::default();
        let ix = build_index(sessions, &p);
        let q = Query {
            text: "Which CobaltRelay breaker resolved the timeout?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let q_cnt = plan.groups.len().min(64);
        let q_mask = if q_cnt == 0 { 0 } else { (1u64 << q_cnt) - 1 };
        let t_own = vec![10.0f32, 0.0, 0.0];
        let t_exact_mask = vec![1u64, 0, 0];
        let final_turns = vec![0usize];
        let inp = GhostCurrentInput {
            ix: &ix,
            plan: &plan,
            q: &q,
            eff_it: &plan.intent,
            t_own: &t_own,
            t_exact_mask: &t_exact_mask,
            final_turn_indices: &final_turns,
            q_group_mask: q_mask,
            q_group_cnt: q_cnt,
            deficit_a: false,
            deficit_b: true,
            is_encyclopedic_multi_doc: false,
        };
        run_ghostcurrent_mdl_q16(&inp)
    };

    // 1. Baseline (zero outcomes -> 0.00% drift, equal symmetric branches, t1 wins tie-break)
    let base = run_gc(&make_corpus(Vec::new()));
    assert!(base.deficit_fired);
    assert_eq!(base.ranked_targets.first().map(|&(ti, _)| ti), Some(1));
    assert!(base.witness_receipt_ids.is_empty());
    let base_g1 = base.target_scores.get(&1).copied().unwrap_or(0);
    let base_g2 = base.target_scores.get(&2).copied().unwrap_or(0);
    assert_eq!(base_g1, base_g2, "symmetric branches must tie with zero outcomes");

    // 2. Plasticized with recorded Strata outcomes: reinforce (t0 -> t2), damp (t0 -> t1)
    let fwd_links = vec![
        TurnLink { kind: "outcome:dead_end:rcpt-dead-1".into(), to: "t1".into(), degree: 2 },
        TurnLink { kind: "outcome:closed_by_false_assumption:rcpt-false-2".into(), to: "t1".into(), degree: 2 },
        TurnLink { kind: "outcome:helpful:rcpt-help-3".into(), to: "t2".into(), degree: 2 },
        TurnLink { kind: "outcome:accepted:rcpt-acc-4".into(), to: "t2".into(), degree: 2 },
        TurnLink { kind: "outcome:causal_walk_proved:rcpt-prove-5".into(), to: "t2".into(), degree: 2 },
    ];
    let plast = run_gc(&make_corpus(fwd_links.clone()));
    assert_eq!(
        plast.ranked_targets.first().map(|&(ti, _)| ti),
        Some(2),
        "reinforced helpful/accepted/proved branch must flip current from t1 to t2"
    );
    let plast_g1 = plast.target_scores.get(&1).copied().unwrap_or(0);
    let plast_g2 = plast.target_scores.get(&2).copied().unwrap_or(0);
    assert!(plast_g2 > base_g2, "reinforced branch conductance must increase ({plast_g2} > {base_g2})");
    assert!(plast_g1 < base_g1 && plast_g1 > 0, "damped dead_end branch must stay finite and drop ({plast_g1} < {base_g1})");
    assert!(plast.witness_receipt_ids.iter().any(|r| r == "rcpt-help-3"));
    assert!(plast.witness_receipt_ids.iter().any(|r| r == "rcpt-acc-4"));
    assert!(plast.witness_receipt_ids.iter().any(|r| r == "rcpt-prove-5"));

    // 3. Bit-exact order-independent replay determinism
    let mut rev_links = fwd_links;
    rev_links.reverse();
    let replay = run_gc(&make_corpus(rev_links));
    assert_eq!(plast.ranked_targets, replay.ranked_targets);
    assert_eq!(plast.target_scores, replay.target_scores);
    assert_eq!(plast.witness_receipt_ids, replay.witness_receipt_ids);
}

#[test]
fn issue_853_ac_constructive_interference_vs_analepsis_flashback_and_negation() {
    use crate::retrieval::graph::ghostcurrent_mdl_q16::{GhostCurrentInput, run_ghostcurrent_mdl_q16};

    let sessions = vec![
        Session {
            id: "s0".into(),
            date: String::new(),
            timestamp: 100,
            turns: vec![Turn {
                id: "t0".into(),
                role: "user".into(),
                content: "Next month VortexEngine will ship with CobaltMesh and QuartzCache.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
        Session {
            id: "s1".into(),
            date: String::new(),
            timestamp: 200,
            turns: vec![Turn {
                id: "t1_flashback".into(),
                role: "user".into(),
                content: "Years earlier VortexEngine had used LegacyTape storage.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
        Session {
            id: "s2".into(),
            date: String::new(),
            timestamp: 300,
            turns: vec![Turn {
                id: "t2_negated".into(),
                role: "user".into(),
                content: "VortexEngine does not use CobaltMesh in production.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
        Session {
            id: "s3".into(),
            date: String::new(),
            timestamp: 400,
            turns: vec![Turn {
                id: "t3_live".into(),
                role: "user".into(),
                content: "VortexEngine runs CobaltMesh and QuartzCache on HelsinkiCluster.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
        Session {
            id: "s4".into(),
            date: String::new(),
            timestamp: 500,
            turns: vec![Turn {
                id: "t4_decoy".into(),
                role: "user".into(),
                content: "VortexEngine documentation mentions OsloNode.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
    ];
    let p = Params::default();
    let ix = build_index(&sessions, &p);
    let q = Query {
        text: "What cluster does VortexEngine currently run on?".into(),
        date: String::new(),
    };
    let mut plan = plan_query(&ix, &q, &p);
    plan.intent.current_state = true;
    let q_cnt = plan.groups.len().min(64);
    let q_mask = if q_cnt == 0 { 0 } else { (1u64 << q_cnt) - 1 };
    let t_own = vec![10.0f32, 0.0, 0.0, 0.0, 0.0];
    let t_exact_mask = vec![1u64, 0, 0, 0, 0];
    let final_turns = vec![0usize];
    let inp = GhostCurrentInput {
        ix: &ix,
        plan: &plan,
        q: &q,
        eff_it: &plan.intent,
        t_own: &t_own,
        t_exact_mask: &t_exact_mask,
        final_turn_indices: &final_turns,
        q_group_mask: q_mask,
        q_group_cnt: q_cnt,
        deficit_a: false,
        deficit_b: true,
        is_encyclopedic_multi_doc: false,
    };
    let res = run_ghostcurrent_mdl_q16(&inp);
    assert!(res.deficit_fired);
    assert_eq!(
        res.ranked_targets.first().map(|&(ti, _)| ti),
        Some(3),
        "3 agreeing carriers onto t3_live must beat single-carrier decoy, flashback, and negated paths: {:?}",
        res.ranked_targets
    );
    let (live_gt, live_gf) = res.target_admittance.get(&3).copied().unwrap_or((0, 0));
    let (decoy_gt, _) = res.target_admittance.get(&4).copied().unwrap_or((0, 0));
    let (flash_gt, flash_gf) = res.target_admittance.get(&1).copied().unwrap_or((0, 0));
    let (neg_gt, neg_gf) = res.target_admittance.get(&2).copied().unwrap_or((0, 0));
    assert!(
        live_gt > decoy_gt && live_gt > live_gf,
        "3 agreeing parallel carriers must constructively amplify G_t ({live_gt}) > decoy ({decoy_gt}) and G_f ({live_gf})"
    );
    assert!(
        flash_gf > 0 && flash_gt <= flash_gf,
        "prospective->flashback analepsis clash must deposit into G_f (gt={flash_gt}, gf={flash_gf})"
    );
    assert!(
        neg_gf > 0 && neg_gt <= neg_gf,
        "negated turn under current_state must deposit into G_f (gt={neg_gt}, gf={neg_gf})"
    );
}

#[test]
fn issue_853_belnap_both_dual_witness_coseats_thesis_and_antithesis_slots_0_and_1() {
    use crate::retrieval::graph::ghostcurrent_mdl_q16::{
        GhostCurrentInput, GhostCurrentSlot0Ctx, apply_ghostcurrent_slot0_promotion,
        run_ghostcurrent_mdl_q16,
    };

    let sessions = vec![
        Session {
            id: "s0".into(),
            date: String::new(),
            timestamp: 100,
            turns: vec![Turn {
                id: "t0_seed".into(),
                role: "user".into(),
                content: "Security audit opened for OrionCore encryption.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
        Session {
            id: "s1".into(),
            date: String::new(),
            timestamp: 200,
            turns: vec![Turn {
                id: "t1_thesis".into(),
                role: "user".into(),
                content: "OrionCore uses ArgonCipher encryption in production.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
        Session {
            id: "s2".into(),
            date: String::new(),
            timestamp: 300,
            turns: vec![Turn {
                id: "t2_antithesis".into(),
                role: "user".into(),
                content: "OrionCore does not use ArgonCipher encryption in production.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
        Session {
            id: "s3".into(),
            date: String::new(),
            timestamp: 400,
            turns: vec![Turn {
                id: "t3_noise".into(),
                role: "user".into(),
                content: "We baked sourdough loaves on Sunday afternoon.".into(),
                links: Vec::new(),
                stability: 0.0,
            }],
        },
    ];
    let p = Params::default();
    let ix = build_index(&sessions, &p);
    let q = Query {
        text: "Is there a contradiction about whether OrionCore uses ArgonCipher?".into(),
        date: String::new(),
    };
    let plan = plan_query(&ix, &q, &p);
    let q_cnt = plan.groups.len().min(64);
    let q_mask = if q_cnt == 0 { 0 } else { (1u64 << q_cnt) - 1 };
    let t_own = vec![10.0f32, 4.0, 4.0, 0.0];
    let t_exact_mask = vec![1u64, 1, 1, 0];
    let mut final_turns = vec![0usize, 3, 1, 2];
    let mut final_sessions = vec![0usize, 3, 1, 2];
    let inp = GhostCurrentInput {
        ix: &ix,
        plan: &plan,
        q: &q,
        eff_it: &plan.intent,
        t_own: &t_own,
        t_exact_mask: &t_exact_mask,
        final_turn_indices: &final_turns,
        q_group_mask: q_mask,
        q_group_cnt: q_cnt,
        deficit_a: false,
        deficit_b: false,
        is_encyclopedic_multi_doc: false,
    };
    let res = run_ghostcurrent_mdl_q16(&inp);
    assert!(res.deficit_fired && res.can_override_slot0);
    assert_eq!(
        res.belnap_dual_pair,
        Some((1, 2)),
        "contradiction query must pair thesis t1 and negated antithesis t2: adm={:?}",
        res.target_admittance
    );

    let mut s0_ctx = GhostCurrentSlot0Ctx {
        ix: &ix,
        plan: &plan,
        q: &q,
        eff_it: &plan.intent,
        pref_on: false,
        deficit_a: false,
        res: &res,
        t_exact_mask: &t_exact_mask,
        q_group_mask: q_mask,
        k: 4,
        final_turn_indices: &mut final_turns,
        final_session_indices: &mut final_sessions,
    };
    apply_ghostcurrent_slot0_promotion(&mut s0_ctx);
    assert_eq!(
        &final_turns[..2],
        &[1usize, 2usize],
        "Belnap Both must co-seat [v_thesis, v_antithesis] into Slots 0 & 1"
    );
    assert_eq!(
        &final_sessions[..2],
        &[1usize, 2usize],
        "Belnap Both must sync thesis and antithesis sessions into Session Slots 0 & 1"
    );
}
