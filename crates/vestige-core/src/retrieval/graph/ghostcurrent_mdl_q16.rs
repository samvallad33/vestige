//! GhostCurrent — Arbitrary-Depth Absorbing Kirchhoff–MDL Multi-Hop Engine (Q16.16)
//!
//! Replaces depth-capped 2-hop heuristics with a single principled integer electrical-network
//! and Rissanen MDL solver:
//!   1. **Epistemic Deficit Gate**: seals Slot 0 on solved single-hop queries (zero regression),
//!      opens strictly on coverage/type/supersession/multi-hop epistemic deficits.
//!   2. **4-Tier Typed Carrier Graph + Logarithmic Hub Conductance**:
//!      `g(c) = tier_w * Q16^2 / (isqrt(df * Q16) * (1 + floor(log2(df))) * 128)`.
//!   3. **Dual Current Injection**: lexical seed turns + virtual query node `Q` via KITE/WordNet/GeoNames.
//!   4. **Strata Live-Leaf Law**: superseded turns (`g_sink = 0`) forward current along
//!      arbitrary-depth `2..=16` revision chains until reaching an un-superseded live leaf.
//!   5. **Hashimoto Non-Backtracking (`c_in != c_out`) + Series/Parallel Kirchhoff Summation**:
//!      multiple independent clue paths converging on one turn ADD their conductances.
//!   6. **Rissanen MDL Branch-and-Bound Dijkstra**: bounded `<= 128` settled states, zero floats.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16;
use crate::retrieval::graph::ghostlink_dgs::{Q16, integer_sqrt, ochiai_divergence_q16};
use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::seating::at1_scorer_q16::{
    QueryAt1Spec, build_query_at1_spec, eval_li_roth_turn,
};
use crate::retrieval::{Index, Query, QueryIntent, QueryPlan};

const TIER_ENTITY: u64 = 128;
const TIER_SUPERSEDE: u64 = 112;
const TIER_CONTENT: u64 = 72;
const TIER_BRIDGE: u64 = 56;

const MAX_SETTLED: usize = 128;
const MAX_HOPS: u8 = 6;
const MAX_SUP_HOPS: u8 = 16;
const MAX_SEED_TURNS: usize = 8;
const MAX_NEIGHBORS_PER_CARRIER: usize = 10;
const HOP_MDL_COST_Q16: u64 = 2_048; // 0.03125 Q16 per hop beyond hop 1
const MIN_CURRENT_Q16: u64 = 1_024; // 0.015625 Q16 pruning floor
const SUP_FORWARD_DECAY_Q16: u64 = 62_259; // 0.95 Q16 per supersession link
const NO_CARRIER: u32 = u32::MAX;
const VIRT_Q_CARRIER: u32 = u32::MAX - 1;
const SUPERSEDE_CARRIER: u32 = u32::MAX - 2;

/// Compact output of GhostCurrent stored on `RankCtx` across Stages 11 and 12.
#[derive(Default, Clone)]
pub(in crate::retrieval) struct GhostCurrentResult {
    pub(in crate::retrieval) deficit_fired: bool,
    pub(in crate::retrieval) can_override_slot0: bool,
    pub(in crate::retrieval) sup_live_leaf_of_h0: Option<usize>,
    pub(in crate::retrieval) ranked_targets: Vec<(usize, u64)>,
    pub(in crate::retrieval) target_scores: BTreeMap<usize, u64>,
    pub(in crate::retrieval) witness_spine: Vec<usize>,
    /// Proof-carrying Strata outcome receipt IDs traversed along `witness_spine` (`#854`).
    pub(in crate::retrieval) witness_receipt_ids: Vec<String>,
}

/// Read-only inputs to `run_ghostcurrent_mdl_q16`.
pub(in crate::retrieval) struct GhostCurrentInput<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) plan: &'a QueryPlan,
    pub(in crate::retrieval) q: &'a Query,
    pub(in crate::retrieval) eff_it: &'a QueryIntent,
    pub(in crate::retrieval) t_own: &'a [f32],
    pub(in crate::retrieval) t_exact_mask: &'a [u64],
    pub(in crate::retrieval) final_turn_indices: &'a [usize],
    pub(in crate::retrieval) q_group_mask: u64,
    pub(in crate::retrieval) q_group_cnt: usize,
    pub(in crate::retrieval) deficit_a: bool,
    pub(in crate::retrieval) deficit_b: bool,
    pub(in crate::retrieval) is_encyclopedic_multi_doc: bool,
}

/// Stage-11 direct Top-5 injection context.
pub(in crate::retrieval) struct GhostCurrentStage11Ctx<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) eff_it: &'a QueryIntent,
    pub(in crate::retrieval) pref_on: bool,
    pub(in crate::retrieval) res: &'a GhostCurrentResult,
    pub(in crate::retrieval) t_exact_mask: &'a [u64],
    pub(in crate::retrieval) q_group_mask: u64,
    pub(in crate::retrieval) deficit_a: bool,
    pub(in crate::retrieval) smax: f32,
    pub(in crate::retrieval) k: usize,
    pub(in crate::retrieval) final_turn_indices: &'a mut Vec<usize>,
    pub(in crate::retrieval) t_scene: &'a mut [f32],
    pub(in crate::retrieval) bridge_boost: &'a mut [f32],
}

/// Stage-12 post-AT1 Slot-0 & witness-spine co-seating context.
pub(in crate::retrieval) struct GhostCurrentSlot0Ctx<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) plan: &'a QueryPlan,
    pub(in crate::retrieval) q: &'a Query,
    pub(in crate::retrieval) eff_it: &'a QueryIntent,
    pub(in crate::retrieval) pref_on: bool,
    pub(in crate::retrieval) deficit_a: bool,
    pub(in crate::retrieval) res: &'a GhostCurrentResult,
    pub(in crate::retrieval) t_exact_mask: &'a [u64],
    pub(in crate::retrieval) q_group_mask: u64,
    pub(in crate::retrieval) k: usize,
    pub(in crate::retrieval) final_turn_indices: &'a mut Vec<usize>,
    pub(in crate::retrieval) final_session_indices: &'a mut Vec<usize>,
}

/// Pure-integer logarithmic hub conductance in Q16.16.
#[inline]
pub(in crate::retrieval) fn hub_conductance_q16(tier_w: u64, df: u32) -> u64 {
    let d = (df as u64).max(1);
    let log2_p1 = (64 - d.leading_zeros()) as u64;
    let sq = integer_sqrt(d.saturating_mul(Q16)).max(1);
    let num = (tier_w as u128)
        .saturating_mul(Q16 as u128)
        .saturating_mul(Q16 as u128);
    let den = (sq as u128)
        .saturating_mul(log2_p1 as u128)
        .saturating_mul(128u128)
        .max(1);
    (num / den).min(Q16 as u128) as u64
}

/// Series conductance `(g1 * g2) / (g1 + g2)` in Q16.16.
#[inline]
fn series_conductance_q16(g1: u64, g2: u64) -> u64 {
    let sum = (g1 as u128).saturating_add(g2 as u128);
    if sum == 0 {
        return 0;
    }
    ((g1 as u128).saturating_mul(g2 as u128) / sum) as u64
}

/// Epistemic Deficit Gate: returns `(deficit_fired, can_override_slot0)`.
pub(in crate::retrieval) fn evaluate_epistemic_deficit_gate(
    inp: &GhostCurrentInput<'_>,
    spec: &QueryAt1Spec,
    t_own_q16: &[u64],
) -> (bool, bool) {
    let Some(&h0) = inp.final_turn_indices.first() else {
        return (false, false);
    };
    let head0_cov = (inp.t_exact_mask.get(h0).copied().unwrap_or(0) & inp.q_group_mask)
        .count_ones() as usize;
    let head0_full = inp.q_group_cnt > 0 && head0_cov >= inp.q_group_cnt;
    let head0_sup = turn_is_superseded(inp.ix, h0);
    let head0_meta_echo = crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
        inp.ix,
        h0,
        &inp.q.text,
    );
    let head0_only_excl = !inp.plan.excluded_ids.is_empty()
        && inp.ix.turns[h0]
            .content
            .iter()
            .any(|id| inp.plan.excluded_ids.contains(id))
        && head0_cov <= 1;
    let head0_q_or_bc = inp.ix.turns[h0].is_pure_info_question
        || inp.ix.turns[h0].is_backchannel
        || head0_meta_echo
        || head0_only_excl;
    let (lr_h0, dur_h0) = eval_li_roth_turn(inp.ix, spec, h0);
    let head0_type_miss = spec.has_wh_answer_class() && (lr_h0 < 0 || dur_h0 < 0);
    let head0_margin_strong = inp.final_turn_indices.get(1).is_none_or(|&h1| {
        t_own_q16
            .get(h0)
            .copied()
            .unwrap_or(0)
            .saturating_mul(20)
            >= t_own_q16
                .get(h1)
                .copied()
                .unwrap_or(0)
                .saturating_mul(23)
    });

    let sealed = head0_full
        && !head0_sup
        && !head0_q_or_bc
        && !head0_type_miss
        && !inp.eff_it.multi_hop
        && !inp.eff_it.temporal
        && !inp.eff_it.current_state
        && head0_margin_strong;

    if sealed {
        return (false, false);
    }

    let deficit_fired = inp.deficit_a
        || inp.deficit_b
        || head0_sup
        || head0_q_or_bc
        || head0_type_miss
        || inp.eff_it.multi_hop
        || inp.eff_it.current_state
        || (inp.q_group_cnt >= 2 && head0_cov < inp.q_group_cnt);

    let can_override_slot0 = deficit_fired
        && (head0_sup
            || head0_q_or_bc
            || head0_type_miss
            || head0_cov == 0
            || inp.deficit_a
            || (inp.q_group_cnt >= 2 && head0_cov < inp.q_group_cnt));

    (deficit_fired, can_override_slot0)
}

#[derive(Clone, Eq, PartialEq)]
struct SearchState {
    conductance_q16: u64,
    depth: u8,
    sup_hops: u8,
    turn: usize,
    last_carrier: u32,
    origin_sess: usize,
    path_idx: usize,
}

impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.conductance_q16
            .cmp(&other.conductance_q16)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.turn.cmp(&self.turn))
    }
}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct PathNode {
    turn: usize,
    parent: Option<usize>,
    receipt_ids: Vec<String>,
}

fn reconstruct_path(paths: &[PathNode], idx: usize) -> Vec<usize> {
    let mut rev = Vec::new();
    let mut cur = Some(idx);
    while let Some(i) = cur {
        rev.push(paths[i].turn);
        cur = paths[i].parent;
        if rev.len() > (MAX_HOPS as usize + MAX_SUP_HOPS as usize + 2) {
            break;
        }
    }
    rev.reverse();
    rev
}

fn reconstruct_receipt_ids(paths: &[PathNode], idx: usize) -> Vec<String> {
    let mut chain: Vec<usize> = Vec::new();
    let mut cur = Some(idx);
    while let Some(i) = cur {
        chain.push(i);
        cur = paths[i].parent;
        if chain.len() > (MAX_HOPS as usize + MAX_SUP_HOPS as usize + 2) {
            break;
        }
    }
    chain.reverse();
    let mut receipts: Vec<String> = Vec::new();
    for i in chain {
        for r in &paths[i].receipt_ids {
            if !receipts.contains(r) {
                receipts.push(r.clone());
            }
        }
    }
    receipts
}

/// Self-Plasticizing Strata Conductance (`#854`): deterministic `Q16.16` outcome multiplier
/// `M_outcome(u, v)` and exact proof receipt IDs for hop `(u, v)`.
/// Returns `(CONDUCTANCE_Q16_ONE, [])` in O(1) whenever no outcome receipts exist (0.00% drift).
fn edge_outcome_plasticity_q16(ix: &Index, u: usize, v: usize) -> (u64, Vec<String>) {
    let edges = &ix.turns[u].outcome_edges;
    if edges.is_empty() {
        return (crate::composition::CONDUCTANCE_Q16_ONE, Vec::new());
    }
    let mut outcomes: Vec<&str> = Vec::new();
    let mut receipts: Vec<String> = Vec::new();
    for (target, otype, rcpt) in edges {
        if *target == v {
            outcomes.push(otype.as_str());
            if !rcpt.is_empty() && !receipts.contains(rcpt) {
                receipts.push(rcpt.clone());
            }
        }
    }
    if outcomes.is_empty() {
        return (crate::composition::CONDUCTANCE_Q16_ONE, Vec::new());
    }
    let mult_q16 = crate::composition::outcome_conductance_multiplier_q16(&outcomes);
    (mult_q16, receipts)
}

fn path_contains(paths: &[PathNode], idx: usize, target: usize) -> bool {
    let mut cur = Some(idx);
    let mut guard = 0usize;
    while let Some(i) = cur {
        if paths[i].turn == target {
            return true;
        }
        cur = paths[i].parent;
        guard += 1;
        if guard > 32 {
            break;
        }
    }
    false
}

/// Build forward supersession adjacency `t_old -> Vec<t_new>` from `ix.turns[*].supersedes_turns`.
fn build_forward_supersession(ix: &Index) -> BTreeMap<usize, Vec<usize>> {
    let mut fwd: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (new_ti, tr) in ix.turns.iter().enumerate() {
        for &old_ti in &tr.supersedes_turns {
            if old_ti < ix.turns.len() && old_ti != new_ti {
                fwd.entry(old_ti).or_default().push(new_ti);
            }
        }
    }
    fwd
}

/// Run the full GhostCurrent absorbing Kirchhoff–MDL solver.
pub(in crate::retrieval) fn run_ghostcurrent_mdl_q16(
    inp: &GhostCurrentInput<'_>,
) -> GhostCurrentResult {
    let nt = inp.ix.turns.len();
    if nt < 2 || inp.final_turn_indices.is_empty() {
        return GhostCurrentResult::default();
    }
    if inp.is_encyclopedic_multi_doc && !inp.deficit_a {
        return GhostCurrentResult::default();
    }

    let t_own_q16: Vec<u64> = inp
        .t_own
        .iter()
        .map(|&v| ieee754_bits_to_q16(v.to_bits()))
        .collect();
    let spec = build_query_at1_spec(inp.ix, inp.plan, inp.q);
    let (deficit_fired, can_override_slot0) = evaluate_epistemic_deficit_gate(inp, &spec, &t_own_q16);
    if !deficit_fired {
        return GhostCurrentResult::default();
    }

    let fwd_sup = build_forward_supersession(inp.ix);
    // Walk exact forward revision chain strictly from Slot-0 `h0` to its terminal live leaf.
    let sup_live_leaf_of_h0 = inp.final_turn_indices.first().copied().and_then(|h0| {
        if !turn_is_superseded(inp.ix, h0) {
            return None;
        }
        let mut cur = h0;
        let mut seen: Vec<usize> = vec![h0];
        for _ in 0..MAX_SUP_HOPS {
            let Some(nexts) = fwd_sup.get(&cur) else {
                break;
            };
            let Some(&nxt) = nexts.iter().rev().find(|&&n| !seen.contains(&n)) else {
                break;
            };
            seen.push(nxt);
            cur = nxt;
            if !turn_is_superseded(inp.ix, cur) {
                return Some(cur);
            }
        }
        None
    });
    let max_df = ((nt as u64 * 12) / 100).max(8).min(64) as u32;
    let max_own = inp
        .final_turn_indices
        .iter()
        .take(MAX_SEED_TURNS)
        .map(|&ti| t_own_q16.get(ti).copied().unwrap_or(0))
        .max()
        .unwrap_or(1)
        .max(1);

    let mut heap: BinaryHeap<SearchState> = BinaryHeap::new();
    let mut paths: Vec<PathNode> = Vec::with_capacity(256);
    let mut seed_mask = 0u64;

    // Dual Source A: Top lexical seed turns
    for &seed_ti in inp.final_turn_indices.iter().take(MAX_SEED_TURNS) {
        if seed_ti >= nt || inp.ix.turns[seed_ti].is_backchannel {
            continue;
        }
        seed_mask |= inp.t_exact_mask.get(seed_ti).copied().unwrap_or(0) & inp.q_group_mask;
        let own = t_own_q16.get(seed_ti).copied().unwrap_or(0);
        let init_g = (Q16 >> 1)
            .saturating_add(((own as u128 * (Q16 >> 1) as u128) / max_own as u128) as u64);
        let p_idx = paths.len();
        paths.push(PathNode {
            turn: seed_ti,
            parent: None,
            receipt_ids: Vec::new(),
        });
        heap.push(SearchState {
            conductance_q16: init_g,
            depth: 0,
            sup_hops: 0,
            turn: seed_ti,
            last_carrier: NO_CARRIER,
            origin_sess: inp.ix.turns[seed_ti].session,
            path_idx: p_idx,
        });
    }

    // Dual Source B: Virtual Query Node `Q` injecting via KITE/WordNet/GeoNames/Wikidata bridge IDs
    let mut virt_ids = spec.virtual_query_vocab_ids();
    if inp.deficit_a {
        let sp_single = crate::retrieval::lexical::tier_bridge_q16::detect_single_subject_speaker(
            inp.ix,
            inp.plan,
            &inp.q.text,
        );
        let push_lex = |virt_ids: &mut Vec<u32>, lemma: &str| {
            for part in lemma.split_ascii_whitespace() {
                if part.len() < 3 || virt_ids.len() >= 48 {
                    continue;
                }
                let st = crate::retrieval::stem(part);
                for cid in [
                    inp.ix.vocab.get(part),
                    inp.ix.vocab.get(&crate::retrieval::stem_term(&st)),
                ]
                .into_iter()
                .flatten()
                {
                    if !inp.plan.excluded_ids.contains(&cid) && !virt_ids.contains(&cid) {
                        virt_ids.push(cid);
                    }
                }
            }
        };
        for g_word in &inp.plan.groups {
            if g_word.starts_with('\u{1}')
                || g_word.len() < 3
                || !crate::retrieval::is_content(g_word)
                || sp_single
                    .as_deref()
                    .is_some_and(|sp| g_word.eq_ignore_ascii_case(sp))
            {
                continue;
            }
            for g_inst in crate::geonames::geo_class_instances(g_word).into_iter().take(8) {
                push_lex(&mut virt_ids, &g_inst);
            }
            for (g_nb, _) in crate::geonames::geo_neighbors(g_word).into_iter().take(6) {
                push_lex(&mut virt_ids, g_nb);
            }
            for w_inst in crate::wordnet::class_to_instances(g_word, 2).into_iter().take(8) {
                push_lex(&mut virt_ids, w_inst);
            }
            for (w_nb, _) in crate::wordnet::full_lexicon_neighbors(g_word).into_iter().take(8) {
                push_lex(&mut virt_ids, w_nb);
            }
            for wd_inst in crate::wikidata::direct_instances_of(g_word, 12)
                .into_iter()
                .chain(crate::wikidata::upward_containers(g_word, 2))
            {
                push_lex(&mut virt_ids, &wd_inst);
            }
            if virt_ids.len() >= 48 {
                break;
            }
        }
    }
    let mut virt_seeded = 0usize;
    for &vid in virt_ids.iter().take(32) {
        if inp.plan.excluded_ids.contains(&vid) {
            continue;
        }
        if let Some(post) = inp.ix.t_postings.get(vid as usize) {
            let df = post.len() as u32;
            if df == 0 || df > max_df {
                continue;
            }
            let g_virt = hub_conductance_q16(TIER_BRIDGE, df);
            if g_virt < MIN_CURRENT_Q16 {
                continue;
            }
            for &raw_ti in post.iter().take(6) {
                let ti = raw_ti as usize;
                if ti >= nt || inp.ix.turns[ti].is_backchannel {
                    continue;
                }
                let p_idx = paths.len();
                paths.push(PathNode {
                    turn: ti,
                    parent: None,
                    receipt_ids: Vec::new(),
                });
                heap.push(SearchState {
                    conductance_q16: g_virt,
                    depth: 1,
                    sup_hops: 0,
                    turn: ti,
                    last_carrier: VIRT_Q_CARRIER,
                    origin_sess: usize::MAX,
                    path_idx: p_idx,
                });
                virt_seeded += 1;
                if virt_seeded >= 32 {
                    break;
                }
            }
        }
        if virt_seeded >= 32 {
            break;
        }
    }

    // Parallel Kirchhoff accumulation: per-target per-carrier max conductance so distinct
    // clue paths ADD linearly while duplicate paths on the same carrier cannot double-count.
    let mut per_target_carrier: BTreeMap<(usize, u32), u64> = BTreeMap::new();
    let mut best_path_for_target: BTreeMap<usize, (u64, usize)> = BTreeMap::new();
    let mut target_receipt_ids: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut settled = 0usize;
    let mut visited_state: BTreeMap<(usize, u32), u64> = BTreeMap::new();

    while let Some(st) = heap.pop() {
        if settled >= MAX_SETTLED {
            break;
        }
        let key = (st.turn, st.last_carrier);
        if visited_state
            .get(&key)
            .is_some_and(|&prev_g| prev_g >= st.conductance_q16)
        {
            continue;
        }
        visited_state.insert(key, st.conductance_q16);
        settled += 1;

        let tr = &inp.ix.turns[st.turn];
        let is_sup = turn_is_superseded(inp.ix, st.turn);

        // Strata Live-Leaf Law: superseded turns have g_sink = 0 and forward current along
        // arbitrary-depth revision chains up to MAX_SUP_HOPS.
        if is_sup {
            if st.sup_hops < MAX_SUP_HOPS
                && let Some(nexts) = fwd_sup.get(&st.turn)
            {
                let g_fwd_base = ((st.conductance_q16 as u128 * SUP_FORWARD_DECAY_Q16 as u128) >> 16)
                    as u64;
                if g_fwd_base >= MIN_CURRENT_Q16 {
                    for &nxt in nexts.iter().take(4) {
                        if path_contains(&paths, st.path_idx, nxt) {
                            continue;
                        }
                        let (m_q16, rcpts) = edge_outcome_plasticity_q16(inp.ix, st.turn, nxt);
                        let g_fwd = crate::composition::apply_outcome_conductance_q16(
                            g_fwd_base, m_q16,
                        );
                        if g_fwd < MIN_CURRENT_Q16 {
                            continue;
                        }
                        let p_idx = paths.len();
                        paths.push(PathNode {
                            turn: nxt,
                            parent: Some(st.path_idx),
                            receipt_ids: rcpts,
                        });
                        heap.push(SearchState {
                            conductance_q16: g_fwd,
                            depth: st.depth.max(1),
                            sup_hops: st.sup_hops + 1,
                            turn: nxt,
                            last_carrier: SUPERSEDE_CARRIER,
                            origin_sess: st.origin_sess,
                            path_idx: p_idx,
                        });
                    }
                }
            }
            continue;
        }

        // Absorbing sink deposit on live declarative turns reached via >= 1 hop or revision chain
        if st.depth >= 1 && !tr.is_pure_info_question && !tr.is_backchannel {
            let mut sink_g = st.conductance_q16;
            if spec.has_wh_answer_class() {
                let (lr, dur) = eval_li_roth_turn(inp.ix, &spec, st.turn);
                if lr > 0 || dur > 0 {
                    sink_g = sink_g.saturating_add(sink_g / 3);
                } else if lr < 0 || dur < 0 {
                    sink_g >>= 1;
                }
            }
            let new_cov = ((inp.t_exact_mask.get(st.turn).copied().unwrap_or(0)
                & inp.q_group_mask)
                & !seed_mask)
                .count_ones() as u64;
            if new_cov > 0 {
                sink_g = sink_g.saturating_add((sink_g * new_cov) >> 2);
            }
            if st.origin_sess != usize::MAX && tr.session != st.origin_sess {
                sink_g = sink_g.saturating_add(sink_g >> 3);
            }
            if st.sup_hops > 0 {
                // Live leaf terminating a supersession chain receives priority weight
                sink_g = sink_g.saturating_add(sink_g >> 1);
            }
            per_target_carrier
                .entry((st.turn, st.last_carrier))
                .and_modify(|v| {
                    if sink_g > *v {
                        *v = sink_g;
                    }
                })
                .or_insert(sink_g);
            best_path_for_target
                .entry(st.turn)
                .and_modify(|(bg, bp)| {
                    if sink_g > *bg {
                        *bg = sink_g;
                        *bp = st.path_idx;
                    }
                })
                .or_insert((sink_g, st.path_idx));
            let path_rcpts = reconstruct_receipt_ids(&paths, st.path_idx);
            if !path_rcpts.is_empty() {
                let entry = target_receipt_ids.entry(st.turn).or_default();
                for r in path_rcpts {
                    if !entry.contains(&r) {
                        entry.push(r);
                    }
                }
            }
        }

        if st.depth >= MAX_HOPS {
            continue;
        }

        // Expand outgoing typed carriers with strict Hashimoto non-backtracking (c != last_carrier)
        let mut carriers: Vec<(u32, u64)> = Vec::new();
        for &eid in &tr.entities {
            if eid != st.last_carrier && !inp.plan.excluded_ids.contains(&eid) {
                carriers.push((eid, TIER_ENTITY));
            }
        }
        for &cid in &tr.coref_entities {
            if cid != st.last_carrier
                && !inp.plan.excluded_ids.contains(&cid)
                && !carriers.iter().any(|&(c, _)| c == cid)
            {
                carriers.push((cid, TIER_ENTITY));
            }
        }
        for &tid in &tr.content {
            if tid != st.last_carrier
                && !inp.plan.excluded_ids.contains(&tid)
                && !carriers.iter().any(|&(c, _)| c == tid)
            {
                carriers.push((tid, TIER_CONTENT));
            }
        }

        // Rank carriers by highest hub conductance and expand top 10
        let mut scored_carriers: Vec<(u64, u32)> = Vec::with_capacity(carriers.len());
        for (cid, tw) in carriers {
            let Some(post) = inp.ix.t_postings.get(cid as usize) else {
                continue;
            };
            let df = post.len() as u32;
            if df < 2 || df > max_df {
                continue;
            }
            let g_edge = hub_conductance_q16(tw, df);
            if g_edge >= MIN_CURRENT_Q16 {
                scored_carriers.push((g_edge, cid));
            }
        }
        scored_carriers.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        for (g_edge, cid) in scored_carriers.into_iter().take(10) {
            let Some(post) = inp.ix.t_postings.get(cid as usize) else {
                continue;
            };
            let mdl_cost = if st.depth >= 1 { HOP_MDL_COST_Q16 } else { 0 };
            for &raw_v in post.iter().take(MAX_NEIGHBORS_PER_CARRIER) {
                let v = raw_v as usize;
                if v == st.turn || v >= nt || inp.ix.turns[v].is_backchannel {
                    continue;
                }
                if path_contains(&paths, st.path_idx, v) {
                    continue;
                }
                // #854 Self-Plasticizing Strata Conductance: scale wire conductance by recorded
                // outcome multiplier BEFORE Ohm/Kirchhoff series combination (`r = Q16^2 / kappa_eff`).
                let (m_q16, rcpts) = edge_outcome_plasticity_q16(inp.ix, st.turn, v);
                let g_edge_eff = crate::composition::apply_outcome_conductance_q16(g_edge, m_q16);
                let g_series = series_conductance_q16(st.conductance_q16, g_edge_eff);
                let g_next = g_series.saturating_sub(mdl_cost);
                if g_next < MIN_CURRENT_Q16 {
                    continue;
                }
                let u_sess = inp.ix.turns[st.turn].session;
                let v_sess = inp.ix.turns[v].session;
                let g_hop = if u_sess != v_sess
                    && u_sess < inp.ix.session_rare_nt.len()
                    && v_sess < inp.ix.session_rare_nt.len()
                {
                    let div = ochiai_divergence_q16(
                        &inp.ix.session_rare_nt[u_sess],
                        &inp.ix.session_rare_nt[v_sess],
                    );
                    // Faloutsos divergent conductance lift: 1.0x..1.25x for cross-session complement hops
                    g_next.saturating_add(((g_next as u128 * div as u128) >> 18) as u64)
                } else {
                    g_next
                };
                let p_idx = paths.len();
                paths.push(PathNode {
                    turn: v,
                    parent: Some(st.path_idx),
                    receipt_ids: rcpts,
                });
                heap.push(SearchState {
                    conductance_q16: g_hop,
                    depth: st.depth + 1,
                    sup_hops: st.sup_hops,
                    turn: v,
                    last_carrier: cid,
                    origin_sess: st.origin_sess,
                    path_idx: p_idx,
                });
            }
        }

        // Also follow explicit RSB backward exact_edges (Tier 2)
        for &(deg, target_ti) in tr.exact_edges.iter().take(4) {
            if target_ti >= nt
                || target_ti == st.turn
                || inp.ix.turns[target_ti].is_backchannel
                || path_contains(&paths, st.path_idx, target_ti)
            {
                continue;
            }
            let (m_q16, rcpts) = edge_outcome_plasticity_q16(inp.ix, st.turn, target_ti);
            let g_edge_base = hub_conductance_q16(TIER_SUPERSEDE, deg.max(2));
            let g_edge_eff = crate::composition::apply_outcome_conductance_q16(g_edge_base, m_q16);
            let g_next = series_conductance_q16(st.conductance_q16, g_edge_eff)
                .saturating_sub(if st.depth >= 1 { HOP_MDL_COST_Q16 } else { 0 });
            if g_next < MIN_CURRENT_Q16 {
                continue;
            }
            let p_idx = paths.len();
            paths.push(PathNode {
                turn: target_ti,
                parent: Some(st.path_idx),
                receipt_ids: rcpts,
            });
            heap.push(SearchState {
                conductance_q16: g_next,
                depth: st.depth + 1,
                sup_hops: st.sup_hops,
                turn: target_ti,
                last_carrier: SUPERSEDE_CARRIER,
                origin_sess: st.origin_sess,
                path_idx: p_idx,
            });
        }
    }

    // Parallel Kirchhoff summation across independent carriers for each target turn
    let mut totals: BTreeMap<usize, u64> = BTreeMap::new();
    for ((ti, _carrier), g) in per_target_carrier {
        totals
            .entry(ti)
            .and_modify(|sum| *sum = sum.saturating_add(g))
            .or_insert(g);
    }

    let mut ranked: Vec<(usize, u64)> = totals.iter().map(|(&ti, &g)| (ti, g)).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(12);

    let witness_spine = ranked
        .first()
        .and_then(|&(ti, _)| best_path_for_target.get(&ti).copied())
        .map(|(_, p_idx)| reconstruct_path(&paths, p_idx))
        .unwrap_or_default();
    let mut witness_receipt_ids = ranked
        .first()
        .and_then(|&(ti, _)| target_receipt_ids.remove(&ti))
        .unwrap_or_default();
    witness_receipt_ids.sort();
    witness_receipt_ids.dedup();

    GhostCurrentResult {
        deficit_fired,
        can_override_slot0,
        sup_live_leaf_of_h0,
        ranked_targets: ranked,
        target_scores: totals,
        witness_spine,
        witness_receipt_ids,
    }
}

/// Stage-11 direct Top-5 seating + bounded scene/session-bridge lift (named AND anonymous corpora).
pub(in crate::retrieval) fn apply_ghostcurrent_stage11_injection(
    ctx: &mut GhostCurrentStage11Ctx<'_>,
) {
    if !ctx.res.deficit_fired || ctx.res.ranked_targets.is_empty() {
        return;
    }
    let max_g = ctx.res.ranked_targets[0].1.max(1);
    let t_scene_max = ctx.t_scene.iter().copied().fold(0.0f32, f32::max);
    let head_t_opt = ctx.final_turn_indices.first().copied();
    let head_s_opt = head_t_opt.map(|h0| ctx.ix.turns[h0].session);
    let head_mask = head_t_opt.map_or(0u64, |h0| {
        ctx.t_exact_mask.get(h0).copied().unwrap_or(0) & ctx.q_group_mask
    });
    let head_sup = head_t_opt.is_some_and(|h0| turn_is_superseded(ctx.ix, h0));
    let pref_guard = ctx.eff_it.preference || ctx.pref_on;

    let mut sess_best: BTreeMap<usize, (u64, bool, bool)> = BTreeMap::new();
    for &(ti, g) in ctx.res.ranked_targets.iter().take(6) {
        if ti >= ctx.t_scene.len() {
            continue;
        }
        let ti_mask = ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & ctx.q_group_mask;
        let has_exact = ti_mask != 0;
        let adds_new = (ti_mask & !head_mask) != 0;
        if pref_guard && !ctx.deficit_a && !has_exact {
            continue;
        }
        let ratio = (g as f32) / (max_g as f32);
        if t_scene_max > 0.0 && (ctx.deficit_a || adds_new) {
            ctx.t_scene[ti] += t_scene_max * 0.14 * ratio;
        }
        let si = ctx.ix.turns[ti].session;
        sess_best
            .entry(si)
            .and_modify(|(b, ex, an)| {
                if g > *b {
                    *b = g;
                }
                *ex |= has_exact;
                *an |= adds_new;
            })
            .or_insert((g, has_exact, adds_new));
    }
    if ctx.smax > 0.0 {
        let bb_cap = if ctx.deficit_a { 0.20 } else { 0.10 };
        for (si, (g, has_exact, adds_new)) in sess_best {
            if Some(si) == head_s_opt || si >= ctx.bridge_boost.len() {
                continue;
            }
            if pref_guard && !ctx.deficit_a && !has_exact {
                continue;
            }
            if !ctx.deficit_a && head_mask != 0 && !adds_new {
                continue;
            }
            let ratio = (g as f32) / (max_g as f32);
            ctx.bridge_boost[si] += ctx.smax * bb_cap * ratio;
        }
    }

    // Directly seat top 2 unseated GhostCurrent targets into Slots 1..5 (respecting per-session cap)
    let mut seated = 0usize;
    let mut head_union = 0u64;
    for &p in ctx.final_turn_indices.iter().take(3) {
        head_union |= ctx.t_exact_mask.get(p).copied().unwrap_or(0) & ctx.q_group_mask;
    }
    for &(cand_ti, _) in ctx.res.ranked_targets.iter().take(4) {
        if seated >= 2 || ctx.final_turn_indices.contains(&cand_ti) {
            continue;
        }
        if turn_is_superseded(ctx.ix, cand_ti)
            || ctx.ix.turns[cand_ti].is_pure_info_question
            || ctx.ix.turns[cand_ti].is_backchannel
        {
            continue;
        }
        let cand_mask = ctx.t_exact_mask.get(cand_ti).copied().unwrap_or(0) & ctx.q_group_mask;
        let adds_new_group = (cand_mask & !head_union) != 0;
        if !ctx.deficit_a && !head_sup && !ctx.eff_it.multi_hop && !adds_new_group {
            continue;
        }
        if pref_guard && !ctx.deficit_a && cand_mask == 0 {
            continue;
        }
        let top5_len = ctx.final_turn_indices.len().min(5);
        let cand_sess = ctx.ix.turns[cand_ti].session;
        let same_sess_top5 = ctx.final_turn_indices[..top5_len]
            .iter()
            .filter(|&&p| ctx.ix.turns[p].session == cand_sess)
            .count();
        if same_sess_top5 >= 2 {
            continue;
        }
        if top5_len < 2 {
            if ctx.deficit_a || adds_new_group {
                ctx.final_turn_indices.push(cand_ti);
                seated += 1;
            }
            continue;
        }
        let min_victim_slot = if (ctx.deficit_a || ctx.eff_it.multi_hop) && !pref_guard {
            1usize
        } else {
            2usize
        };
        let victim = (min_victim_slot..top5_len).rev().find(|&s| {
            let inc_ti = ctx.final_turn_indices[s];
            let inc_sess = ctx.ix.turns[inc_ti].session;
            let dup_sess = ctx.final_turn_indices[..s]
                .iter()
                .any(|&p| ctx.ix.turns[p].session == inc_sess);
            let mut prev_m = 0u64;
            for &p in &ctx.final_turn_indices[..s] {
                prev_m |= ctx.t_exact_mask.get(p).copied().unwrap_or(0) & ctx.q_group_mask;
            }
            let zero_marginal = ((ctx.t_exact_mask.get(inc_ti).copied().unwrap_or(0)
                & ctx.q_group_mask)
                & !prev_m)
                == 0;
            (turn_is_superseded(ctx.ix, inc_ti) && !ctx.eff_it.temporal)
                || (dup_sess && zero_marginal)
        });
        if let Some(v_slot) = victim {
            ctx.final_turn_indices.insert(v_slot, cand_ti);
            if ctx.final_turn_indices.len() > ctx.k {
                ctx.final_turn_indices.truncate(ctx.k);
            }
            seated += 1;
        }
    }
}

/// Stage-12 post-AT1 Slot-0 & witness-spine co-seating (strict zero-regression Epistemic Deficit law).
///
/// Only overrides Slot 0 when EITHER:
///   1. `h0` is directly superseded AND GhostCurrent proved `sup_live_leaf_of_h0` along `h0`'s own
///      forward revision chain (never a random unrelated hop!), OR
///   2. `h0` exhibits a verified structural epistemic deficit (`h0_non_sp_cov == 0` under `deficit_a`,
///      meta-echo/sign-off closure, or excluded-only match) AND `best_ti` is a declarative,
///      speaker-aligned witness beating `h0` by `>= 1.25x` integer Kirchhoff current.
pub(in crate::retrieval) fn apply_ghostcurrent_slot0_promotion(
    ctx: &mut GhostCurrentSlot0Ctx<'_>,
) {
    if !ctx.res.can_override_slot0 {
        return;
    }
    let Some(&h0) = ctx.final_turn_indices.first() else {
        return;
    };
    let h0_sup = turn_is_superseded(ctx.ix, h0);
    let sp_single = crate::retrieval::lexical::tier_bridge_q16::detect_single_subject_speaker(
        ctx.ix,
        ctx.plan,
        &ctx.q.text,
    );
    let mut non_sp_group_mask = 0u64;
    for (gi, gw) in ctx.plan.groups.iter().enumerate().take(64) {
        if sp_single
            .as_deref()
            .is_some_and(|sp| gw.eq_ignore_ascii_case(sp))
        {
            continue;
        }
        non_sp_group_mask |= 1u64 << gi;
    }
    let h0_non_sp_cov =
        (ctx.t_exact_mask.get(h0).copied().unwrap_or(0) & ctx.q_group_mask & non_sp_group_mask)
            .count_ones();
    let h0_is_meta_echo =
        crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
            ctx.ix,
            h0,
            &ctx.q.text,
        );
    let h0_only_excluded = !ctx.plan.excluded_ids.is_empty()
        && ctx.ix.turns[h0]
            .content
            .iter()
            .any(|id| ctx.plan.excluded_ids.contains(id))
        && h0_non_sp_cov == 0;
    let h0_deficit = (ctx.deficit_a && h0_non_sp_cov == 0) || h0_is_meta_echo || h0_only_excluded;
    let h0_protected = ctx.ix.turns[h0].is_standing_policy
        || ctx.ix.turns[h0].is_personal_constraint
        || ctx.ix.turns[h0].has_tool_call
        || ctx.eff_it.is_event_sequence;
    let pref_guard = ctx.eff_it.preference || ctx.pref_on;
    let ql_gc = ctx.q.text.trim().to_ascii_lowercase();
    let is_factual_past_gc = ql_gc.contains("did i ")
        || ql_gc.contains("did we ")
        || ql_gc.contains("have i ")
        || ql_gc.contains("what is my ")
        || ql_gc.contains("when did ")
        || ql_gc.starts_with("how many ")
        || ql_gc.starts_with("how long ");
    let allow_standing_policy_gc = !is_factual_past_gc
        && !pref_guard
        && (ctx.eff_it.task_under_rules
            || ql_gc.starts_with("how do i ")
            || ql_gc.starts_with("how should i ")
            || ql_gc.starts_with("what are the different ways")
            || (ql_gc.contains("which ") && ql_gc.contains(" should i")));

    // Case 1: Proven live-leaf terminating h0's own forward supersession chain (strictly on current-state queries
    // where the live leaf shares query topic coverage so unrelated revision links never hijack IF/IE/MSR/KU heads)
    let proven_sup_leaf = if h0_sup && ctx.eff_it.current_state && !ctx.eff_it.temporal && !h0_protected {
        ctx.res.sup_live_leaf_of_h0.filter(|&leaf| {
            let leaf_cov =
                (ctx.t_exact_mask.get(leaf).copied().unwrap_or(0) & ctx.q_group_mask).count_ones();
            leaf != h0
                && leaf_cov >= 1
                && !turn_is_superseded(ctx.ix, leaf)
                && !ctx.ix.turns[leaf].is_pure_info_question
                && !ctx.ix.turns[leaf].is_backchannel
                && (allow_standing_policy_gc || !ctx.ix.turns[leaf].is_standing_policy)
        })
    } else {
        None
    };

    // Case 2: Un-handcuffed epistemic deficit rescue (`h0_non_sp_cov == 0`, meta-echo, or excluded-only head)
    // gated by speaker alignment and strict `>= 1.25x` integer Kirchhoff current margin over `h0`.
    let zero_cov_rescue = if proven_sup_leaf.is_none()
        && h0_deficit
        && !h0_protected
        && !pref_guard
    {
        let h0_g = ctx.res.target_scores.get(&h0).copied().unwrap_or(0);
        let spec = build_query_at1_spec(ctx.ix, ctx.plan, ctx.q);
        ctx.res.ranked_targets.iter().copied().find_map(|(best_ti, best_g)| {
            if best_ti == h0
                || turn_is_superseded(ctx.ix, best_ti)
                || ctx.ix.turns[best_ti].is_pure_info_question
                || ctx.ix.turns[best_ti].is_backchannel
                || (!allow_standing_policy_gc && ctx.ix.turns[best_ti].is_standing_policy)
                || crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo(
                    ctx.ix,
                    best_ti,
                    &ctx.q.text,
                )
            {
                return None;
            }
            if !ctx.plan.excluded_ids.is_empty()
                && ctx.ix.turns[best_ti]
                    .content
                    .iter()
                    .any(|id| ctx.plan.excluded_ids.contains(id))
            {
                let b_ns = (ctx.t_exact_mask.get(best_ti).copied().unwrap_or(0)
                    & ctx.q_group_mask
                    & non_sp_group_mask)
                    .count_ones();
                if b_ns == 0 {
                    return None;
                }
            }
            if let Some(sp) = sp_single.as_deref() {
                let sp_ok = ctx.ix.turns[best_ti].speaker.eq_ignore_ascii_case(sp)
                    || ctx.ix.turns[best_ti]
                        .body_for_headers
                        .to_ascii_lowercase()
                        .contains(sp);
                if !sp_ok {
                    return None;
                }
            }
            if best_g.saturating_mul(4) < h0_g.saturating_mul(5) {
                return None;
            }
            let best_cov = (ctx.t_exact_mask.get(best_ti).copied().unwrap_or(0)
                & ctx.q_group_mask
                & non_sp_group_mask)
                .count_ones();
            let (lr_b, dur_b) = eval_li_roth_turn(ctx.ix, &spec, best_ti);
            let best_type_hit = spec.has_wh_answer_class() && (lr_b > 0 || dur_b > 0);
            let has_proven_receipt = !ctx.res.witness_receipt_ids.is_empty()
                && ctx.res.ranked_targets.first().is_some_and(|&(t0, _)| t0 == best_ti);
            let strong_enough = best_cov >= 1
                || (best_type_hit && best_g >= (Q16 * 3) / 4)
                || (has_proven_receipt && best_g >= (Q16 >> 1))
                || ((h0_is_meta_echo || h0_only_excluded) && best_g >= (Q16 >> 1));
            if strong_enough {
                Some(best_ti)
            } else {
                None
            }
        })
    } else {
        None
    };

    let Some(best_ti) = proven_sup_leaf.or(zero_cov_rescue) else {
        return;
    };

    if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == best_ti) {
        ctx.final_turn_indices.remove(pos);
    }
    ctx.final_turn_indices.insert(0, best_ti);

    // Co-seat at most 1 multi-hop witness turn into Slot 1 only on zero-coverage rescue
    if zero_cov_rescue.is_some() {
        for &w_ti in ctx.res.witness_spine.iter().rev() {
            if w_ti == best_ti
                || w_ti == h0
                || ctx.ix.turns[w_ti].is_backchannel
                || ctx.final_turn_indices.iter().take(3).any(|&x| x == w_ti)
            {
                continue;
            }
            if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == w_ti) {
                ctx.final_turn_indices.remove(pos);
            }
            let at = 1usize.min(ctx.final_turn_indices.len());
            ctx.final_turn_indices.insert(at, w_ti);
            break;
        }
    }
    if ctx.final_turn_indices.len() > ctx.k {
        ctx.final_turn_indices.truncate(ctx.k);
    }

    // Sync owning session into Session Slot 0 only on proven supersession or true zero-coverage rescue
    let best_sess = ctx.ix.turns[best_ti].session;
    if !ctx.final_session_indices.is_empty() && ctx.final_session_indices[0] != best_sess {
        if let Some(spos) = ctx
            .final_session_indices
            .iter()
            .position(|&s| s == best_sess)
        {
            ctx.final_session_indices.remove(spos);
        }
        ctx.final_session_indices.insert(0, best_sess);
        if ctx.final_session_indices.len() > ctx.k {
            ctx.final_session_indices.truncate(ctx.k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index, plan_query};

    fn mk_turn(id: &str, role: &str, text: &str) -> Turn {
        Turn {
            id: id.to_string(),
            role: role.to_string(),
            content: text.to_string(),
            links: Vec::new(),
            stability: 0.0,
        }
    }

    fn mk_sess(id: &str, turns: Vec<Turn>) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
            timestamp: 0,
            turns,
        }
    }

    fn mk_q(text: &str) -> Query {
        Query {
            text: text.to_string(),
            date: String::new(),
        }
    }

    #[test]
    fn single_hop_solved_head_seals_slot0() {
        let p = Params::default();
        let corpus = vec![
            mk_sess("s0", vec![mk_turn("t0", "Alice", "I visited Kyoto in Japan last summer.")]),
            mk_sess("s1", vec![mk_turn("t1", "Bob", "Kyoto has great temples.")]),
        ];
        let ix = build_index(&corpus, &p);
        let q = mk_q("Where did Alice visit in Japan?");
        let plan = plan_query(&ix, &q, &p);
        let q_cnt = plan.groups.len().min(64);
        let q_mask = if q_cnt == 0 { 0 } else { (1u64 << q_cnt) - 1 };
        let t_own = vec![20.0f32, 5.0f32];
        let t_exact_mask = vec![q_mask, 1u64];
        let final_turns = vec![0usize, 1usize];
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
        assert!(!res.deficit_fired, "solved single-hop head must seal GhostCurrent");
        assert!(!res.can_override_slot0);
    }

    #[test]
    fn four_hop_supersession_chain_forwards_to_live_leaf() {
        let p = Params::default();
        let corpus = vec![
            mk_sess("s0", vec![mk_turn("t0", "user", "My favorite database is Postgres.")]),
            mk_sess("s1", vec![mk_turn("t1", "user", "Actually I switched from Postgres to SQLite.")]),
            mk_sess("s2", vec![mk_turn("t2", "user", "No longer SQLite, I moved to DuckDB.")]),
            mk_sess("s3", vec![mk_turn("t3", "user", "Updated again: now I exclusively use ClickHouse.")]),
        ];
        let mut ix = build_index(&corpus, &p);
        ix.turns[1].supersedes_turns = vec![0];
        ix.turns[2].supersedes_turns = vec![1];
        ix.turns[3].supersedes_turns = vec![2];
        ix.turn_superseded = vec![true, true, true, false];

        let q = mk_q("What is my current favorite database?");
        let plan = plan_query(&ix, &q, &p);
        let q_cnt = plan.groups.len().min(64);
        let q_mask = if q_cnt == 0 { 0 } else { (1u64 << q_cnt) - 1 };
        let t_own = vec![12.0f32, 3.0, 2.0, 1.0];
        let t_exact_mask = vec![1u64, 0, 0, 0];
        let final_turns = vec![0usize, 1, 2, 3];
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
        assert!(res.can_override_slot0);
        assert_eq!(
            res.ranked_targets.first().map(|&(ti, _)| ti),
            Some(3),
            "4-hop revision chain must terminate at live leaf t3"
        );
    }

    #[test]
    fn parallel_three_clue_convergence_beats_single_decoy() {
        let p = Params::default();
        let corpus = vec![
            mk_sess("s0", vec![mk_turn("t0", "user", "AlphaBeacon joined CobaltGuild near ZephyrPeak.")]),
            mk_sess("s1", vec![mk_turn("t1", "user", "AlphaBeacon wrote a note about DecoyBridge.")]),
            mk_sess("s2", vec![mk_turn("t2", "user", "AlphaBeacon and CobaltGuild met at ZephyrPeak in Oslo.")]),
        ];
        let ix = build_index(&corpus, &p);
        let q = mk_q("Where did AlphaBeacon and CobaltGuild meet near ZephyrPeak?");
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
        let res = run_ghostcurrent_mdl_q16(&inp);
        assert!(res.deficit_fired);
        assert_eq!(
            res.ranked_targets.first().map(|&(ti, _)| ti),
            Some(2),
            "3 converging carriers must beat 1-carrier decoy"
        );
    }

    #[test]
    fn hashimoto_non_backtracking_blocks_same_carrier_bounce() {
        let p = Params::default();
        let corpus = vec![
            mk_sess("s0", vec![mk_turn("t0", "user", "QuasarToken started the experiment.")]),
            mk_sess("s1", vec![mk_turn("t1", "user", "QuasarToken finished in Zurich.")]),
        ];
        let ix = build_index(&corpus, &p);
        let q = mk_q("Where did QuasarToken finish?");
        let plan = plan_query(&ix, &q, &p);
        let q_cnt = plan.groups.len().min(64);
        let q_mask = if q_cnt == 0 { 0 } else { (1u64 << q_cnt) - 1 };
        let t_own = vec![8.0f32, 0.0];
        let t_exact_mask = vec![1u64, 0];
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
        assert_eq!(res.ranked_targets.first().map(|&(ti, _)| ti), Some(1));
        assert!(
            !res.witness_spine.windows(3).any(|w| w[0] == w[2]),
            "non-backtracking path must never bounce A -> B -> A"
        );
    }
}
