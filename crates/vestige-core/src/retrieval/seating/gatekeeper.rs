//! Seating pass: `gatekeeper`.

use crate::retrieval::*;


/// Rank-1 gatekeeper. Runs after RSB. Reorders the RSB window. It does not
/// grow the window. The pool is that window, so its size is the caller's k.
pub(in crate::retrieval) fn rank1_gatekeeper(ix: &Index, query: &str, list: &mut Vec<usize>, t_scene: &[f32]) {
    let n = list.len();
    if n < 2 {
        return;
    }
    let pool = list.clone();
    let slots = wh_slots(query);
    let degree = pool_structural_degree(ix, &pool);
    let mut net = vec![vec![0i64; n]; n];
    for a in 0..n {
        for b in (a + 1)..n {
            let (ta, tb) = (pool[a], pool[b]);
            let mut margin = 0i64;
            let scene_a = t_scene.get(ta).copied().unwrap_or(0.0);
            let scene_b = t_scene.get(tb).copied().unwrap_or(0.0);
            match scene_a.partial_cmp(&scene_b).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            match degree[a].cmp(&degree[b]) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            let stab_a = ix.turns.get(ta).map(|t| t.stability).unwrap_or(0.0);
            let stab_b = ix.turns.get(tb).map(|t| t.stability).unwrap_or(0.0);
            match stab_a.partial_cmp(&stab_b).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Greater => margin += 1,
                std::cmp::Ordering::Less => margin -= 1,
                std::cmp::Ordering::Equal => {}
            }
            if slots != 0 {
                let ok_a = wh_satisfied(ix, ta, slots, degree[a] > 0);
                let ok_b = wh_satisfied(ix, tb, slots, degree[b] > 0);
                if ok_a && !ok_b {
                    margin += 1;
                } else if ok_b && !ok_a {
                    margin -= 1;
                }
            }
            net[a][b] = margin;
            net[b][a] = -margin;
        }
    }
    let rsb: Vec<usize> = (0..n).collect();
    let mut order = kemeny_young(&net, &rsb);
    if order.len() != n {
        order = rsb.clone();
    }
    let mut turned: Vec<usize> = order.iter().map(|&i| pool[i]).collect();
    if turn_is_superseded(ix, turned[0]) {
        if let Some(pos) = turned.iter().position(|&ti| !turn_is_superseded(ix, ti)) {
            let keeper = turned.remove(pos);
            turned.insert(0, keeper);
        }
    }
    *list = turned;
}

/// Slot 1 is the RSB v4 head. The gatekeeper reorders only the rest.
/// ATMS, the four ballots, and the Copeland cap run on that tail exactly as
/// they do on a full window. The head is not a candidate for demotion.
pub(in crate::retrieval) fn rank1_gatekeeper_keep_rsb_head(ix: &Index, query: &str, list: &mut Vec<usize>, t_scene: &[f32]) {
    if list.len() < 2 {
        return;
    }
    let head = list[0];
    let mut tail = list[1..].to_vec();
    rank1_gatekeeper(ix, query, &mut tail, t_scene);
    list.clear();
    list.push(head);
    list.extend(tail);
}
