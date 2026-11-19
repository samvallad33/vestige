//! Mirrored unit tests: `fusion_tests` (5 tests).

use super::*;


#[test]
fn kemeny_puts_the_condorcet_winner_first() {
    let net = vec![
        vec![0, 1, 1],
        vec![-1, 0, 1],
        vec![-1, -1, 0],
    ];
    assert_eq!(kemeny_young(&net, &[0, 1, 2]), vec![0, 1, 2]);
}

#[test]
fn kemeny_branch_and_bound_matches_brute_force() {
    let net = vec![
        vec![0, 2, -1, 1],
        vec![-2, 0, 3, -1],
        vec![1, -3, 0, 2],
        vec![-1, 1, -2, 0],
    ];
    let rsb = [0, 1, 2, 3];
    let solved = kemeny_branch_and_bound(&[0, 1, 2, 3], &net, &rsb);
    let mut best = i64::MIN;
    let mut best_order = Vec::new();
    let mut perm = vec![0usize, 1, 2, 3];
    fn walk(perm: &mut [usize], i: usize, net: &[Vec<i64>], best: &mut i64, best_order: &mut Vec<usize>) {
        if i == perm.len() {
            let score = kemeny_pair_score(perm, net);
            if score > *best {
                *best = score;
                *best_order = perm.to_vec();
            }
            return;
        }
        for j in i..perm.len() {
            perm.swap(i, j);
            walk(perm, i + 1, net, best, best_order);
            perm.swap(i, j);
        }
    }
    walk(&mut perm, 0, &net, &mut best, &mut best_order);
    assert_eq!(kemeny_pair_score(&solved, &net), best);
    let solved_key: Vec<usize> = solved.iter().copied().collect();
    let mut lex_best = solved_key.clone();
    let mut perm = vec![0usize, 1, 2, 3];
    fn walk_lex(perm: &mut [usize], i: usize, net: &[Vec<i64>], best: i64, lex_best: &mut Vec<usize>) {
        if i == perm.len() {
            if kemeny_pair_score(perm, net) == best && perm < lex_best {
                *lex_best = perm.to_vec();
            }
            return;
        }
        for j in i..perm.len() {
            perm.swap(i, j);
            walk_lex(perm, i + 1, net, best, lex_best);
            perm.swap(i, j);
        }
    }
    walk_lex(&mut perm, 0, &net, best, &mut lex_best);
    assert_eq!(solved_key, lex_best);
}

#[test]
fn kemeny_tie_keeps_the_rsb_order() {
    let net = vec![vec![0i64; 4]; 4];
    assert_eq!(kemeny_young(&net, &[0, 1, 2, 3]), vec![0, 1, 2, 3]);
}

#[test]
fn one_chat_cannot_fill_the_top_five_when_another_fact_exists() {
    let walnut = "The walnut cabinet oil was mixed, brushed, and left to cure overnight.";
    let req = RankRequest {
        sessions: vec![
            Session {
                id: "walnut".into(),
                date: "2024/01/02 (Tue) 09:00".into(),
                timestamp: 0,
                turns: vec![
                    turn("user", walnut),
                    turn("user", "More walnut cabinet oil notes about the second coat and the brush."),
                    turn("user", "Still the walnut cabinet oil, now describing the rag and the cure."),
                    turn("user", "Again the walnut cabinet oil, with the tin and the cloth."),
                    turn("user", "Yet more walnut cabinet oil, including the shelf and the lid."),
                ],
            },
            Session {
                id: "maple".into(),
                date: "2024/03/04 (Mon) 09:00".into(),
                timestamp: 0,
                turns: vec![turn("user", "The maple drawer rails were cut, sanded, and fitted today.")],
            },
        ],
        queries: vec![Query {
            text: "What connects the walnut cabinet oil and the maple drawer rails?".into(),
            date: String::new(),
        }],
        top_k: 5,
        params: Params::default(),
    };
    let ranked = rank(&req);
    assert!(!ranked.results[0].intent.multi_hop);
    let sessions: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.session_id.as_str()).collect();
    assert!(sessions.contains(&"maple"), "second fact was buried in one chat: {sessions:?}");
    let walnut_n = sessions.iter().filter(|s| **s == "walnut").count();
    assert_eq!(walnut_n, 3, "without a multi-hop phrase the walnut chat keeps three seats: {sessions:?}");
}

#[test]
fn step10_florilegium_relaxes_floors_for_multi_session() {
    assert_eq!(step10_florilegium_floors(2), (0.38, 0.20, false));
    assert_eq!(step10_florilegium_floors(1), (0.45, 0.25, true));
}
