//! Shared pipeline state (`RankCtx`) and integer-quantized bag helper across the 13 ranking stages.

use crate::retrieval::*;

pub(in crate::retrieval) fn bag_to_q8_into(
    bag: &Bag,
    plan: &QueryPlan,
    st: &Stats,
    terms: &mut Vec<crate::retrieval::lexical::qld::QldTermInput>,
) -> crate::retrieval::lexical::qld::QldBagInput {
    let mut dl_q8 = [0u64; 3];
    let mut avdl_q8 = [0u64; 3];
    for f in 0..3 {
        dl_q8[f] = (bag.len[f] * 256.0).round().max(0.0) as u64;
        avdl_q8[f] = (st.avg[f] * 256.0).round().max(256.0) as u64;
    }
    terms.clear();
    for qt in &plan.terms {
        let df = st.df.get(&qt.id).copied().unwrap_or(0);
        let w_q8 = (qt.weight * 256.0).round().max(0.0) as u64;
        if df == 0 || w_q8 == 0 {
            continue;
        }
        let mut tf_q8 = [0u64; 3];
        let mut any = false;
        for f in 0..3 {
            let tf = bag.tf[f].get(&qt.id).copied().unwrap_or(0.0);
            if tf > 0.0 {
                tf_q8[f] = (tf * 256.0).round().max(1.0) as u64;
                any = true;
            }
        }
        if any {
            terms.push(crate::retrieval::lexical::qld::QldTermInput {
                tf_q8,
                w_q8,
                df_doc: df as u64,
            });
        }
    }
    crate::retrieval::lexical::qld::QldBagInput {
        dl_q8,
        avdl_q8,
        n_docs: (st.n as u64).max(1),
    }
}



/// Mutable pipeline state passed straight through Stages 01..13 of `rank_one`.
pub(in crate::retrieval) struct RankCtx<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) q: &'a Query,
    pub(in crate::retrieval) p: &'a Params,
    pub(in crate::retrieval) k: usize,
    pub(in crate::retrieval) ns: usize,
    pub(in crate::retrieval) nt: usize,
    pub(in crate::retrieval) plan: QueryPlan,
    pub(in crate::retrieval) pref_on: bool,
    pub(in crate::retrieval) persona_infer_on: bool,
    pub(in crate::retrieval) episodic_user_quote: bool,
    pub(in crate::retrieval) user_role: bool,
    pub(in crate::retrieval) fl: Fields,

    // Stage 01
    pub(in crate::retrieval) s_mask: Vec<u64>,
    pub(in crate::retrieval) s_exact_mask: Vec<u64>,
    pub(in crate::retrieval) s_score: Vec<f32>,
    pub(in crate::retrieval) smax: f32,

    // Stage 02
    pub(in crate::retrieval) t_own: Vec<f32>,
    pub(in crate::retrieval) t_raw: Vec<f32>,
    pub(in crate::retrieval) t_turn_raw: Vec<f32>,
    pub(in crate::retrieval) t_mask: Vec<u64>,
    pub(in crate::retrieval) t_exact_mask: Vec<u64>,

    // Stage 03
    pub(in crate::retrieval) t_scene: Vec<f32>,
    pub(in crate::retrieval) t_scene_sess: Vec<f32>,
    pub(in crate::retrieval) q_toks_eff: Vec<String>,
    pub(in crate::retrieval) past_cue_eff: bool,
    pub(in crate::retrieval) seq_mode_eff: bool,
    pub(in crate::retrieval) has_named_speakers_eff: bool,
    pub(in crate::retrieval) is_monologue_corpus: bool,
    pub(in crate::retrieval) has_compact_rule_register: bool,
    pub(in crate::retrieval) cond_rule_turns: Vec<usize>,
    pub(in crate::retrieval) eff_it: QueryIntent,
    pub(in crate::retrieval) t_bridge: Vec<f32>,
    pub(in crate::retrieval) t_bridge_hits: Vec<u8>,

    // Stage 04
    pub(in crate::retrieval) lexical_smax: f32,
    pub(in crate::retrieval) lexical_tmax_sess: f32,
    pub(in crate::retrieval) lexical_tmax: f32,
    pub(in crate::retrieval) standing_rules: Vec<RankedItem>,
    pub(in crate::retrieval) s_policy_floored: Vec<bool>,
    pub(in crate::retrieval) t_policy_floored: Vec<bool>,
    pub(in crate::retrieval) s_pref_floored: Vec<bool>,
    pub(in crate::retrieval) t_pref_floored: Vec<bool>,
    pub(in crate::retrieval) initiative_content: Vec<DetHashSet<u32>>,
    pub(in crate::retrieval) bridge_boost: Vec<f32>,
    pub(in crate::retrieval) s_ord_base: Vec<usize>,

    // Stage 05
    pub(in crate::retrieval) total_q_groups: usize,
    pub(in crate::retrieval) mask_limit: u64,
    pub(in crate::retrieval) final_session_indices: Vec<usize>,
    pub(in crate::retrieval) strong_top_sessions: usize,

    // Stage 06
    pub(in crate::retrieval) max_per_session_top5: usize,
    pub(in crate::retrieval) turn_ts: Vec<i64>,
    pub(in crate::retrieval) final_turn_indices: Vec<usize>,

    // Stage 10
    pub(in crate::retrieval) base_q16: Vec<u64>,

    // Stage 11
    pub(in crate::retrieval) q_group_cnt: usize,
    pub(in crate::retrieval) q_group_mask: u64,
    pub(in crate::retrieval) grounded_group_mask: u64,
    pub(in crate::retrieval) top3_union_exact: u64,
    pub(in crate::retrieval) has_ungrounded_rare_oov: bool,
    pub(in crate::retrieval) deficit_a: bool,
    pub(in crate::retrieval) deficit_b: bool,
    pub(in crate::retrieval) torrens_locked_head: bool,
}
