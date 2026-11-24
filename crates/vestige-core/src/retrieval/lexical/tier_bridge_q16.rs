//! Stage-01 Speaker-Sparse 5-Lexicon Tier-Bridge Virtual Carriers & Meta-Echo / Rare-Entity Guards.
//!
//! Pure `Q16.16` integer carrier scoring across Vestige's 5 SHA-256-pinned offline lexicons:
//! - GeoNames (`0.75` = `49_152` Q16)
//! - WordNet 3.1 (`0.72` = `47_186` Q16)
//! - Wikidata CC0 (`0.68` = `44_564` Q16)
//! - ConceptNet 5.7 2-hop (`0.55` = `36_045` Q16)
//! - Roget 1911 (`0.45` = `29_491` Q16)

use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::{is_content, raw_tokens, stem, stem_term, DetHashSet, Index, Query, QueryPlan};

const W_GEONAMES_Q16: u64 = 49_152;
const W_WORDNET_Q16: u64 = 47_186;
const W_WIKIDATA_Q16: u64 = 44_564;
const W_CONCEPTNET_Q16: u64 = 36_045;
const W_ROGET_Q16: u64 = 29_491;

pub(in crate::retrieval) fn speaker_scope_ok(
    ix: &Index,
    ti: usize,
    subject_speaker: Option<&str>,
) -> bool {
    let tr = &ix.turns[ti];
    if tr.is_question || tr.is_backchannel {
        return false;
    }
    match subject_speaker {
        Some(sp) => tr.speaker.eq_ignore_ascii_case(sp),
        None => true,
    }
}

pub(in crate::retrieval) fn detect_single_subject_speaker(
    ix: &Index,
    plan: &QueryPlan,
    q_text: &str,
) -> Option<String> {
    let mut all_speakers: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for t in ix.turns.iter().take(128) {
        let sp = t.speaker.trim();
        if !sp.is_empty() && !sp.eq_ignore_ascii_case("user") && !sp.eq_ignore_ascii_case("assistant") {
            all_speakers.insert(sp.to_ascii_lowercase());
        }
    }
    if all_speakers.is_empty() {
        return None;
    }
    let q_toks = raw_tokens(q_text);
    let mut matched: Vec<String> = Vec::new();
    for sp in &all_speakers {
        let in_groups = plan.groups.iter().any(|g| g.eq_ignore_ascii_case(sp));
        let in_raw = q_toks.iter().any(|w| w.eq_ignore_ascii_case(sp));
        if (in_groups || in_raw) && !matched.contains(sp) {
            matched.push(sp.clone());
        }
    }
    if matched.len() == 1 {
        matched.into_iter().next()
    } else {
        None
    }
}

pub(in crate::retrieval) fn apply_stage01_tier_bridge_q16(
    ix: &Index,
    q: &Query,
    plan: &QueryPlan,
    s_exact_mask: &[u64],
    s_score: &mut [f32],
) {
    let ns = s_score.len();
    if ns < 4 {
        return;
    }

    // 1. Dampen sessions whose ONLY query-matching user turns are meta-echoes / sign-offs:
    let q_term_ids: DetHashSet<u32> = plan.terms.iter().map(|qt| qt.id).collect();
    for (si, sc_slot) in s_score.iter_mut().enumerate() {
        if *sc_slot <= 0.0 {
            continue;
        }
        let mut matching_user = 0usize;
        let mut echo_user = 0usize;
        let mut substantive_asst = false;
        for &ti in &ix.session_turns[si] {
            let tr = &ix.turns[ti];
            let hits = tr.content.iter().any(|id| q_term_ids.contains(id))
                || tr.feats.iter().any(|f| q_term_ids.contains(&f.id));
            if !hits {
                continue;
            }
            if tr.field == 0 {
                matching_user += 1;
                if is_discourse_closure_or_meta_echo(ix, ti, &q.text) {
                    echo_user += 1;
                }
            } else if !tr.is_backchannel && tr.content.iter().any(|id| plan.focal_ids.contains(id)) {
                substantive_asst = true;
            }
        }
        if matching_user > 0 && matching_user == echo_user && !substantive_asst {
            *sc_slot *= 0.25;
        }
    }

    let subject_sp = detect_single_subject_speaker(ix, plan, &q.text);
    let has_named_dialogue_tb = ix
        .turns
        .iter()
        .take(32)
        .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");

    // 2. Stage-01 Rare-Entity Compound Lift (`1 <= s_df <= 2`, e.g. `Flask-Login` / `\u{1}ent:flasklogin`)
    //    scoped strictly to anonymous user/assistant chats so named-dialogue Temporal sessions stay untouched:
    if !has_named_dialogue_tb && !plan.rare_entity_ids.is_empty() && ns >= 10 {
        let pre_max = s_score.iter().copied().fold(0.0f32, f32::max);
        if pre_max > 0.0 {
            for (si, sc_slot) in s_score.iter_mut().enumerate() {
                let has_rare_non_echo = ix.session_turns[si].iter().any(|&ti| {
                    !is_discourse_closure_or_meta_echo(ix, ti, &q.text)
                        && plan.rare_entity_ids.iter().any(|rid| {
                            ix.turns[ti].content.contains(rid)
                                || ix.turns[ti].entities.contains(rid)
                                || ix.turns[ti].feats.iter().any(|f| f.id == *rid)
                        })
                });
                if has_rare_non_echo {
                    *sc_slot = (*sc_slot * 1.48).max(pre_max * 0.92);
                }
            }
        }
    }

    // 3. Speaker-Sparse 5-Lexicon Tier-Bridge Virtual Carriers:
    let sp_ref = subject_sp.as_deref();
    let mut nonsp_group_mask: u64 = 0;
    for qt in &plan.terms {
        if qt.exact_hit
            && let Some(g) = qt.group
            && g < 64
        {
            let gnm = plan.groups.get(g).map(String::as_str).unwrap_or("");
            if !sp_ref.is_some_and(|sp| gnm.eq_ignore_ascii_case(sp)) {
                nonsp_group_mask |= 1u64 << g;
            }
        }
    }

    // Compute max exact non-speaker coverage among speaker-scoped turns:
    let mut non_speaker_cov: u32 = 0;
    for (si, &em) in s_exact_mask.iter().enumerate() {
        let sess_ns_cov = (em & nonsp_group_mask).count_ones();
        if sess_ns_cov == 0 {
            continue;
        }
        if let Some(sp) = sp_ref {
            let mut best_sp_turn = 0u32;
            for &ti in &ix.session_turns[si] {
                if !speaker_scope_ok(ix, ti, Some(sp)) {
                    continue;
                }
                let mut t_cov = 0u32;
                for qt in &plan.terms {
                    if qt.exact_hit
                        && let Some(g) = qt.group
                        && g < 64
                        && (nonsp_group_mask & (1u64 << g)) != 0
                        && (ix.turns[ti].content.contains(&qt.id)
                            || ix.turns[ti].feats.iter().any(|f| f.id == qt.id))
                    {
                        t_cov += 1;
                    }
                }
                best_sp_turn = best_sp_turn.max(t_cov);
            }
            non_speaker_cov = non_speaker_cov.max(best_sp_turn);
        } else {
            non_speaker_cov = non_speaker_cov.max(sess_ns_cov);
        }
    }

    let raw_q = raw_tokens(&q.text);
    // hardcode-scan: closed-class (WH-determiner head extraction)
    let mut wh_head_words: Vec<String> = Vec::new();
    for (idx, tok) in raw_q.iter().enumerate() {
        if matches!(tok.as_str(), "what" | "which") {
            for nxt in raw_q.iter().skip(idx + 1).take(3) {
                if matches!(
                    nxt.as_str(),
                    "did" | "does" | "do" | "is" | "are" | "was" | "were" | "would" | "could"
                        | "might" | "may" | "should" | "has" | "have" | "had" | "kind" | "type"
                        | "sort" | "of" | "a" | "an" | "the"
                ) {
                    continue;
                }
                if is_content(nxt) && !sp_ref.is_some_and(|sp| nxt.eq_ignore_ascii_case(sp)) {
                    wh_head_words.push(nxt.clone());
                }
            }
        }
    }
    let has_wh_head = !wh_head_words.is_empty();

    // Protect anonymous multi-keyword benchmarks and temporal queries when exact overlap already exists:
    if sp_ref.is_none() && non_speaker_cov >= 1 {
        return;
    }
    if plan.intent.temporal && non_speaker_cov >= 1 {
        return;
    }
    if sp_ref.is_some() && non_speaker_cov >= 2 && !has_wh_head {
        return;
    }

    // Collect open-class non-speaker, non-excluded query words (including Porphyry-discounted WH heads):
    let mut seed_words: Vec<String> = Vec::new();
    for w in &raw_q {
        if !is_content(w) || w.len() < 3 {
            continue;
        }
        if sp_ref.is_some_and(|sp| w.eq_ignore_ascii_case(sp)) {
            continue;
        }
        if ix.vocab.get(w).is_some_and(|id| plan.excluded_ids.contains(&id)) {
            continue;
        }
        if !seed_words.contains(w) {
            seed_words.push(w.clone());
        }
    }
    if seed_words.is_empty() {
        return;
    }

    let ns_u32 = ns as u32;
    let df_cap = ns_u32.div_ceil(50).max(3);
    let mut best_carrier: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();

    for qw in &seed_words {
        let q_id = ix.vocab.get(qw);
        let q_st_id = ix.vocab.get(&stem_term(&stem(qw)));
        let df_speaker: u32 = if let Some(sp) = sp_ref {
            ix.turns
                .iter()
                .enumerate()
                .filter(|&(ti, _)| speaker_scope_ok(ix, ti, Some(sp)))
                .filter(|&(_, tr)| {
                    q_id.is_some_and(|id| tr.content.contains(&id))
                        || q_st_id.is_some_and(|id| tr.feats.iter().any(|f| f.id == id))
                })
                .count() as u32
        } else {
            q_id.and_then(|id| ix.s_stats.df.get(&id).copied()).unwrap_or(0)
        };

        let is_wh_noun = wh_head_words.iter().any(|w| w == qw);
        let fire_word = if sp_ref.is_some() {
            df_speaker == 0 || non_speaker_cov == 0 || is_wh_noun
        } else {
            df_speaker == 0 && non_speaker_cov == 0
        };
        if !fire_word {
            continue;
        }

        let mut admit = |lemma: &str, w_rel_q16: u64| {
            for part in lemma.split_ascii_whitespace() {
                if part.len() < 3 || part.eq_ignore_ascii_case(qw) {
                    continue;
                }
                let st = stem(part);
                for cid in [ix.vocab.get(part), ix.vocab.get(&stem_term(&st))]
                    .into_iter()
                    .flatten()
                {
                    if plan.excluded_ids.contains(&cid) {
                        continue;
                    }
                    let s_df = ix.s_stats.df.get(&cid).copied().unwrap_or(0);
                    if !(1..=df_cap).contains(&s_df) {
                        continue;
                    }
                    let ic_q16 = aspect_ic_q16(s_df, ns_u32);
                    let s_bridge = (((ic_q16 as u128) * (w_rel_q16 as u128)) >> 16) as u64;
                    let ent = best_carrier.entry(cid).or_insert(0);
                    if s_bridge > *ent {
                        *ent = s_bridge;
                    }
                }
            }
        };

        for g_inst in crate::geonames::geo_class_instances(qw) {
            admit(&g_inst, W_GEONAMES_Q16);
        }
        for (g_nb, _) in crate::geonames::geo_neighbors(qw) {
            admit(g_nb, W_GEONAMES_Q16);
        }
        for w_inst in crate::wordnet::class_to_instances(qw, 2) {
            admit(w_inst, W_WORDNET_Q16);
        }
        for (w_nb, _) in crate::wordnet::full_lexicon_neighbors(qw) {
            admit(w_nb, W_WORDNET_Q16);
        }
        for wd_inst in crate::wikidata::direct_instances_of(qw, 16)
            .into_iter()
            .chain(crate::wikidata::upward_containers(qw, 2))
        {
            admit(&wd_inst, W_WIKIDATA_Q16);
        }
        for (cn_nb, _) in crate::conceptnet::two_hop_neighbors(qw, 12) {
            admit(cn_nb, W_CONCEPTNET_Q16);
        }
        for (rg_nb, _) in crate::roget::conceptual_neighbors(qw, 48) {
            admit(rg_nb, W_ROGET_Q16);
        }
    }

    if best_carrier.is_empty() {
        return;
    }

    let mut top_carriers: Vec<(u32, u64)> = best_carrier.into_iter().collect();
    top_carriers.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top_carriers.truncate(16);

    let max_single_bridge = top_carriers.first().map(|x| x.1).unwrap_or(1).max(1);
    let mut sess_bridge_q16 = vec![0u64; ns];
    let mut peak_sess_bridge = 0u64;
    for (si, sb_slot) in sess_bridge_q16.iter_mut().enumerate() {
        let mut best_turn_sc = 0u64;
        for &ti in &ix.session_turns[si] {
            if !speaker_scope_ok(ix, ti, sp_ref) {
                continue;
            }
            let tr = &ix.turns[ti];
            let mut tsc = 0u64;
            for &(cid, w_b) in &top_carriers {
                if tr.content.contains(&cid) || tr.feats.iter().any(|f| f.id == cid) {
                    tsc = tsc.saturating_add(w_b);
                }
            }
            best_turn_sc = best_turn_sc.max(tsc);
        }
        *sb_slot = best_turn_sc;
        peak_sess_bridge = peak_sess_bridge.max(best_turn_sc);
    }

    if peak_sess_bridge == 0 {
        return;
    }

    let cur_smax = s_score.iter().copied().fold(0.0f32, f32::max).max(0.01);
    let denom = peak_sess_bridge.max(max_single_bridge);
    let boost_scale = if non_speaker_cov == 0 { 0.62f32 } else { 0.24f32 };
    for (si, &sb) in sess_bridge_q16.iter().enumerate() {
        if sb == 0 {
            continue;
        }
        let ratio = ((sb as f32) / (denom as f32)).clamp(0.0, 1.0);
        s_score[si] += cur_smax * boost_scale * ratio;
    }
}
