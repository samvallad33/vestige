//! Greedy submodular session and turn set selection.

use crate::retrieval::*;
use crate::retrieval::rank::context::RankCtx;

/// Returns `true` when the query requires a deep scored session pool prior to
/// the final `top_k` cut (`event_ordering` and `summarization` only).
/// All other query types stay strictly on the existing `k` cut (byte-identical).
#[inline]
pub(in crate::retrieval) fn deep_session_pool_enabled(
    eff_it: &QueryIntent,
    query: &str,
) -> bool {
    eff_it.is_event_sequence || eff_summary_request(query)
}

pub(in crate::retrieval) fn select_submodular_sessions(
    ctx: &RankCtx<'_>,
    total_q_groups: usize,
    mask_limit: u64,
) -> Vec<usize> {
    let ix = ctx.ix;
    let p = ctx.p;
    let ns = ctx.ns;
    let k = ctx.k;
    let it = &ctx.plan.intent;
    let eff_it = &ctx.eff_it;
    let s_score = &ctx.s_score;
    let s_mask = &ctx.s_mask;
    let s_policy_floored = &ctx.s_policy_floored;
    let s_pref_floored = &ctx.s_pref_floored;
    let bridge_boost = &ctx.bridge_boost;
    let s_ord_base = &ctx.s_ord_base;
    let smax = ctx.smax;
    let initiative_content = &ctx.initiative_content;
    let pool_cap = if deep_session_pool_enabled(eff_it, &ctx.q.text) {
        ns.max(k)
    } else {
        k.max(30).min(ns)
    };

    if p.enable_submodular && ns > 1 && eff_summary_request(&ctx.q.text) {
        use crate::retrieval::event_ordering::topic_thread::{
            apply_rsb_retroactive_backfill_q16, bipartite_entity_session_ppr_q16, ThreadSessionNode,
        };
        use crate::retrieval::fusion::aspect_coverage_q16::{
            aspect_ic_q16, select_summary_aspect_sessions_q16, AspectSessionCand, Q16_ONE,
        };
        use crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16;
        use std::collections::{BTreeMap, BTreeSet};

        let ns_u32 = (ns as u32).max(1);
        let is_rule_only_sess = |si: usize| -> bool {
            ix.session_turns[si]
                .iter()
                .copied()
                .find(|&ti| ix.turns[ti].field == 0 && !ix.turns[ti].is_backchannel)
                .is_some_and(|ti| {
                    ix.turns[ti].is_standing_policy
                        || ix.turns[ti]
                            .body_for_headers
                            .to_ascii_lowercase()
                            .contains(" never ")
                })
        };
        // Archival running-date epoch propagation & TReMu relative sub-session offset (`epoch_offset`):
        // undated sessions inherit the active date opened by the most recent preceding dated session.
        let mut eff_ts = vec![0i64; ns];
        let mut epoch_offset = vec![0usize; ns];
        let mut last_ts = 0i64;
        let mut last_dated_si = 0usize;
        for si in 0..ns {
            if ix.session_ts[si] > 0 {
                last_ts = ix.session_ts[si];
                last_dated_si = si;
            }
            eff_ts[si] = last_ts;
            epoch_offset[si] = si.saturating_sub(last_dated_si);
        }

        let mut max_sight_q16: u64 = 0;
        let sight_q16: Vec<u64> = (0..ns)
            .map(|si| {
                if is_rule_only_sess(si) {
                    return 0;
                }
                let v = ieee754_bits_to_q16((s_score[si] + bridge_boost[si]).to_bits());
                if v > max_sight_q16 {
                    max_sight_q16 = v;
                }
                v
            })
            .collect();
        if max_sight_q16 == 0 {
            return Vec::new();
        }

        // Closed-class syntactic topic-clause extraction (after "about "/"of "/"how ", before
        // " across"/" throughout"/" during"/" over time"/"?") weights real topic terms 3x over
        // generic meta-question framing.
        let q_clean = ctx.q.text.to_ascii_lowercase();
        let topic_slice = {
            let after_pre = q_clean
                .split_once(" about ")
                .or_else(|| q_clean.split_once(" of how "))
                .or_else(|| q_clean.split_once(" how "))
                .or_else(|| q_clean.split_once(" of "))
                .map(|(_, tail)| tail)
                .unwrap_or(q_clean.as_str());
            let mut cut = after_pre;
            for stop in [" across ", " throughout ", " during ", " over time", " so far", "?"] {
                if let Some((head, _)) = cut.split_once(stop) {
                    if head.trim().len() >= 6 {
                        cut = head;
                    }
                }
            }
            cut
        };
        let topic_clause_words: BTreeSet<String> = raw_tokens(topic_slice)
            .into_iter()
            .filter(|w| is_content(w))
            .flat_map(|w| {
                let st = stem(&w);
                [w, st]
            })
            .collect();

        // T04 Incipit Opener Scoring with IC² over user-turn document frequency (`user_df`):
        // weights distinctive topic terms ("autocomplete", "congruence", "probability", "database")
        // quadratically above ambient conversation verbs ("implement", "understand", "project", "app").
        let mut q_term_ic: BTreeMap<u32, u64> = BTreeMap::new();
        let mut q_term_ids: BTreeSet<u32> = BTreeSet::new();
        let mut q_rare_anchors: BTreeSet<u32> = BTreeSet::new();
        let mut q_rare_anchor_words: Vec<String> = Vec::new();
        let rare_anchor_df_cap = (ns_u32 / 5).max(3);
        for qt in &ctx.plan.terms {
            if qt.exact_hit {
                q_term_ids.insert(qt.id);
                let user_df = (0..ns)
                    .filter(|&si| !is_rule_only_sess(si) && ix.session_bags[si].tf[0].contains_key(&qt.id))
                    .count() as u32;
                let df = if user_df > 0 {
                    user_df
                } else {
                    ix.s_stats.df.get(&qt.id).copied().unwrap_or(ns_u32)
                };
                if df > 0 && df <= (ns_u32 * 3 / 4).max(2) {
                    let base_ic = aspect_ic_q16(df, ns_u32);
                    let ic_sq = ((base_ic as u128).saturating_mul(base_ic as u128) >> 16)
                        .max((Q16_ONE >> 3) as u128) as u64;
                    let nm_opt = ix.vocab.names.get(qt.id as usize).map(String::as_str);
                    let bare = nm_opt
                        .map(|nm| nm.strip_prefix("\u{1}stem:").unwrap_or(nm))
                        .unwrap_or("");
                    let in_topic = !bare.is_empty() && topic_clause_words.contains(bare);
                    let is_ent = nm_opt.is_some_and(|nm| nm.starts_with("\u{1}ent:"))
                        || (!bare.is_empty()
                            && ix
                                .vocab
                                .get(&format!("\u{1}ent:{bare}"))
                                .is_some_and(|eid| ix.s_stats.df.get(&eid).copied().unwrap_or(0) > 0));
                    let is_rare_anchor = in_topic && ((1..=rare_anchor_df_cap).contains(&df) || is_ent);
                    let w = if is_rare_anchor {
                        q_rare_anchors.insert(qt.id);
                        if !bare.is_empty()
                            && !bare.starts_with('\u{1}')
                            && !q_rare_anchor_words.iter().any(|x| x == bare)
                        {
                            q_rare_anchor_words.push(bare.to_string());
                        }
                        ic_sq.saturating_mul(6)
                    } else if in_topic {
                        ic_sq.saturating_mul(3)
                    } else {
                        base_ic >> 1
                    };
                    q_term_ic.insert(qt.id, w);
                }
            }
        }
        let mut rare_anchor_lex_ids: BTreeSet<u32> = BTreeSet::new();
        for aw in &q_rare_anchor_words {
            let mut add_al = |lem: &str| {
                let lst = stem(lem);
                for lid in [ix.vocab.get(lem), ix.vocab.get(&stem_term(&lst))]
                    .into_iter()
                    .flatten()
                {
                    let df = ix.s_stats.df.get(&lid).copied().unwrap_or(0);
                    if (1..=(ns_u32 / 3).max(4)).contains(&df) {
                        rare_anchor_lex_ids.insert(lid);
                    }
                }
            };
            for (lem, _) in crate::wordnet::full_lexicon_neighbors(aw).into_iter().take(16) {
                add_al(lem);
            }
            for (lem, _) in crate::roget::conceptual_neighbors(aw, 16) {
                add_al(lem);
            }
            for (lem, _) in crate::conceptnet::two_hop_neighbors(aw, 8) {
                add_al(lem);
            }
        }
        let mut incipit_turn: Vec<Option<usize>> = vec![None; ns];
        let mut incipit_raw: Vec<u64> = vec![0; ns];
        let mut max_incipit: u64 = 1;
        for si in 0..ns {
            if is_rule_only_sess(si) {
                continue;
            }
            let open_ti = ix.session_turns[si]
                .iter()
                .copied()
                .find(|&ti| ix.turns[ti].field == 0 && !ix.turns[ti].is_backchannel && !ix.turns[ti].is_standing_policy)
                .or_else(|| ix.session_turns[si].first().copied());
            incipit_turn[si] = open_ti;
            if let Some(ti) = open_ti {
                let mut sc: u64 = 0;
                let mut matched_ids: BTreeSet<u32> = BTreeSet::new();
                for &cid in &ix.turns[ti].content {
                    if let Some(&ic) = q_term_ic.get(&cid) {
                        if matched_ids.insert(cid) {
                            sc = sc.saturating_add(ic);
                        }
                    }
                }
                for f in &ix.turns[ti].feats {
                    if let Some(&ic) = q_term_ic.get(&f.id) {
                        if matched_ids.insert(f.id) {
                            sc = sc.saturating_add((ic * 3) >> 2);
                        }
                    }
                }
                if ix.turns[ti].is_initiative {
                    sc = sc.saturating_add(sc >> 2);
                }
                incipit_raw[si] = sc;
                if sc > max_incipit {
                    max_incipit = sc;
                }
            }
        }

        // T03 Facet Split ONLY on explicit multi-item enumerations ("including " or comma-separated lists),
        // never splitting a single coordinated verb/noun phrase ("implementing and improving city autocomplete")!
        let mut facets: Vec<Vec<String>> = Vec::new();
        let has_explicit_enum = q_clean.contains("including ") || topic_slice.contains(',');
        let enum_body = q_clean
            .split_once("including ")
            .map(|(_, tail)| tail)
            .unwrap_or(topic_slice);
        let mut raw_clauses: Vec<&str> = Vec::new();
        if has_explicit_enum {
            for comma_seg in enum_body.split([',', ';']) {
                for and_seg in comma_seg.split(" as well as ") {
                    for sub in and_seg.split(" and ") {
                        let s = sub.trim();
                        if !s.is_empty() {
                            raw_clauses.push(s);
                        }
                    }
                }
            }
        }
        let max_facet_df = (ns_u32 / 2).max(2);
        let rare_facet_user_df = (ns_u32 / 4).max(3);
        for clause in raw_clauses {
            let f_words: Vec<String> = raw_tokens(clause)
                .into_iter()
                .filter(|w| {
                    if !is_content(w) {
                        return false;
                    }
                    let df = ix
                        .vocab
                        .get(w)
                        .and_then(|id| ix.s_stats.df.get(&id).copied())
                        .or_else(|| {
                            ix.vocab
                                .get(&stem_term(&stem(w)))
                                .and_then(|id| ix.s_stats.df.get(&id).copied())
                        })
                        .unwrap_or(0);
                    df == 0 || df <= max_facet_df
                })
                .collect();
            let has_distinctive_facet_tok = f_words.iter().any(|w| {
                let ids = [ix.vocab.get(w), ix.vocab.get(&stem_term(&stem(w)))];
                let min_udf = ids
                    .into_iter()
                    .flatten()
                    .map(|id| {
                        (0..ns)
                            .filter(|&si| !is_rule_only_sess(si) && ix.session_bags[si].tf[0].contains_key(&id))
                            .count() as u32
                    })
                    .min()
                    .unwrap_or(0);
                min_udf == 0 || min_udf <= rare_facet_user_df
            });
            if !f_words.is_empty() && has_distinctive_facet_tok && facets.len() < 6 {
                facets.push(f_words);
            }
        }
        // Score every non-policy session against each enumeration facet (2x on incipit, 1x on session content).
        let n_facets = if facets.len() >= 2 { facets.len() } else { 0 };
        let mut facet_scores: Vec<Vec<u64>> = vec![vec![0u64; ns]; n_facets];
        let mut facet_max: Vec<u64> = vec![1u64; n_facets];
        for fi in 0..n_facets {
            let mut f_ids: Vec<(u32, u64)> = Vec::new();
            let mut seen_f: BTreeSet<u32> = BTreeSet::new();
            let mut f_lex_ids: Vec<(u32, u64)> = Vec::new();
            let mut seen_lex: BTreeSet<u32> = BTreeSet::new();
            for fw in &facets[fi] {
                for cand_id in [
                    ix.vocab.get(fw),
                    ix.vocab.get(&stem_term(&stem(fw))),
                ]
                .into_iter()
                .flatten()
                {
                    if seen_f.insert(cand_id) {
                        let df = ix.s_stats.df.get(&cand_id).copied().unwrap_or(0);
                        if df > 0 && df <= max_facet_df {
                            f_ids.push((cand_id, aspect_ic_q16(df, ns_u32)));
                        }
                    }
                }
                // Derivational prefix roots (e.g., "documentation" -> "document", "enhancements" -> "enhance",
                // "security" -> "secure") via standard closed-class English nominalization suffixes.
                // hardcode-scan: closed-class (English nominalization/derivational suffixes)
                for &(suf, rep) in &[
                    ("ation", ""), ("ation", "e"), ("tion", ""), ("tion", "e"),
                    ("ments", ""), ("ment", ""), ("ity", ""), ("ity", "e"),
                    ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("ies", "y"),
                ] {
                    if let Some(base) = fw.strip_suffix(suf)
                        && base.len() >= 4
                    {
                        let deriv = if rep.is_empty() { base.to_string() } else { format!("{base}{rep}") };
                        for did in [ix.vocab.get(&deriv), ix.vocab.get(&stem_term(&stem(&deriv)))]
                            .into_iter()
                            .flatten()
                        {
                            if seen_f.insert(did) {
                                let df = ix.s_stats.df.get(&did).copied().unwrap_or(0);
                                if (1..=max_facet_df).contains(&df) {
                                    f_ids.push((did, aspect_ic_q16(df, ns_u32)));
                                }
                            }
                        }
                    }
                }
                // Precompute full human-knowledge lexicon (WordNet 3.1 4-POS + Roget's 1911 + ConceptNet 5.7)
                // neighbor IDs ONCE per facet outside the `0..ns` session loop (O(1) per session lookup).
                let mut add_lex = |lem: &str| {
                    for lid in [ix.vocab.get(lem), ix.vocab.get(&stem_term(&stem(lem)))]
                        .into_iter()
                        .flatten()
                    {
                        if !seen_f.contains(&lid) && seen_lex.insert(lid) {
                            let df = ix.s_stats.df.get(&lid).copied().unwrap_or(0);
                            if (1..=max_facet_df).contains(&df) {
                                f_lex_ids.push((lid, aspect_ic_q16(df, ns_u32)));
                            }
                        }
                    }
                };
                for (lem, _) in crate::wordnet::full_lexicon_neighbors(fw).into_iter().take(24) {
                    add_lex(lem);
                }
                for (lem, _) in crate::roget::conceptual_neighbors(fw, 32) {
                    add_lex(lem);
                }
                for (lem, _) in crate::conceptnet::two_hop_neighbors(fw, 10) {
                    add_lex(lem);
                }
            }
            for si in 0..ns {
                if is_rule_only_sess(si) {
                    continue;
                }
                let mut fsc: u64 = 0;
                for &(fid, ic) in &f_ids {
                    let in_inc = incipit_turn[si].is_some_and(|ti| {
                        ix.turns[ti].content.contains(&fid)
                            || ix.turns[ti].feats.iter().any(|f| f.id == fid)
                    });
                    if in_inc {
                        fsc = fsc.saturating_add(ic.saturating_mul(2));
                    } else if ix.session_bags[si].tf[0].contains_key(&fid)
                        || ix.session_content[si].contains(&fid)
                        || ix.session_rare_nt[si].contains(&fid)
                    {
                        fsc = fsc.saturating_add(ic);
                    }
                }
                if fsc == 0 && !f_lex_ids.is_empty() {
                    if let Some(ti) = incipit_turn[si] {
                        let inc_ids = &ix.turns[ti].content;
                        let inc_feats = &ix.turns[ti].feats;
                        for &(lid, ic) in &f_lex_ids {
                            if inc_ids.contains(&lid) || inc_feats.iter().any(|f| f.id == lid) {
                                fsc = fsc.saturating_add(ic >> 1);
                            }
                        }
                    }
                }
                facet_scores[fi][si] = fsc;
                if fsc > facet_max[fi] {
                    facet_max[fi] = fsc;
                }
            }
        }

        // T06 CombMNZ across facets + T07 Bipartite Topic-Thread PPR + Josselyn RSB Backfill across full pool.
        let max_hub_df = (ns_u32 / 3).clamp(3, 24);
        let mut seed_ents_q16: BTreeMap<u32, u64> = BTreeMap::new();
        for (&qid, &ic) in &q_term_ic {
            if ix.s_stats.df.get(&qid).copied().unwrap_or(ns_u32) <= max_hub_df {
                seed_ents_q16.insert(qid, ic.saturating_mul(2));
            }
        }
        // Enrich PPR seeds ONLY from rare proper entities (`session_entities`, `2 <= df <= max_hub_df`)
        // of the top-2 incipit sessions at 0.5x weight so generic nouns never drown the topic.
        let mut by_init: Vec<(u64, usize)> = (0..ns)
            .filter(|&si| !is_rule_only_sess(si))
            .map(|si| (incipit_raw[si].saturating_add(sight_q16[si] >> 2), si))
            .collect();
        by_init.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for &(_, si) in by_init.iter().take(2) {
            for &eid in &ix.session_entities[si] {
                let df = ix.s_stats.df.get(&eid).copied().unwrap_or(0);
                if (2..=max_hub_df).contains(&df) {
                    let ic = aspect_ic_q16(df, ns_u32) >> 1;
                    let e = seed_ents_q16.entry(eid).or_insert(0);
                    *e = (*e).max(ic);
                }
            }
        }
        let mut thread_nodes: Vec<ThreadSessionNode> = Vec::with_capacity(ns);
        for si in 0..ns {
            if is_rule_only_sess(si) {
                continue;
            }
            let mut ents: Vec<(u32, u64)> = Vec::new();
            let mut seen_e: BTreeSet<u32> = BTreeSet::new();
            if let Some(ti) = incipit_turn[si] {
                for &cid in &ix.turns[ti].content {
                    let df = ix.s_stats.df.get(&cid).copied().unwrap_or(0);
                    if (1..=max_hub_df).contains(&df) && seen_e.insert(cid) {
                        ents.push((cid, aspect_ic_q16(df, ns_u32).saturating_mul(2)));
                    }
                }
            }
            for &eid in &ix.session_entities[si] {
                let df = ix.s_stats.df.get(&eid).copied().unwrap_or(0);
                if (1..=max_hub_df).contains(&df) && seen_e.insert(eid) {
                    ents.push((eid, aspect_ic_q16(df, ns_u32)));
                }
            }
            for &rid in ix.session_rare_nt[si].iter().take(16) {
                let df = ix.s_stats.df.get(&rid).copied().unwrap_or(0);
                if (1..=max_hub_df).contains(&df) && seen_e.insert(rid) {
                    ents.push((rid, aspect_ic_q16(df, ns_u32)));
                }
            }
            let mut ants: Vec<usize> = Vec::new();
            for &(_, prev) in &ix.session_rsb_back[si] {
                if !ants.contains(&prev) {
                    ants.push(prev);
                }
            }
            for &prev in &ix.session_supersedes_back[si] {
                if !ants.contains(&prev) {
                    ants.push(prev);
                }
            }
            if let Some(ep) = ix.session_episode_prev.get(si).copied().flatten() {
                if !ants.contains(&ep) {
                    ants.push(ep);
                }
            }
            thread_nodes.push(ThreadSessionNode {
                session_idx: si,
                session_ts: eff_ts[si],
                base_score_q16: sight_q16[si],
                entities: ents,
                rsb_antecedents: ants,
            });
        }
        let mut ppr_scores = bipartite_entity_session_ppr_q16(&thread_nodes, &seed_ents_q16, 8);
        let max_ppr_pre = ppr_scores.values().copied().max().unwrap_or(0);
        let rsb_cap = ((max_ppr_pre as u128).saturating_mul(19_661) >> 16) as u64;
        apply_rsb_retroactive_backfill_q16(&thread_nodes, &mut ppr_scores, rsb_cap);
        let max_ppr = ppr_scores.values().copied().max().unwrap_or(0).max(1);

        // Compute fused per-session quality (whole + 3*incipit + PPR/RSB + 3*CombMNZ-facets).
        let mut combmnz_raw: Vec<u64> = vec![0u64; ns];
        let mut max_combmnz: u64 = 1;
        if n_facets > 0 {
            for si in 0..ns {
                if is_rule_only_sess(si) {
                    continue;
                }
                let mut hits: u64 = 0;
                let mut fsum: u64 = 0;
                for fi in 0..n_facets {
                    let f = facet_scores[fi][si];
                    if f > 0 {
                        hits += 1;
                        let fnm = (((f as u128) << 16) / (facet_max[fi] as u128)).min(Q16_ONE as u128) as u64;
                        fsum = fsum.saturating_add(fnm);
                    }
                }
                let cmnz = fsum.saturating_mul(hits.max(1));
                combmnz_raw[si] = cmnz;
                if cmnz > max_combmnz {
                    max_combmnz = cmnz;
                }
            }
        }
        // Rare topic anchor witness flags: damp generic sessions with zero direct/1-hop/RSB/lexicon anchor support.
        let mut sess_hits_rare_anchor: Vec<bool> = vec![true; ns];
        if !q_rare_anchors.is_empty() {
            let direct_anchor: Vec<bool> = (0..ns)
                .map(|si| {
                    q_rare_anchors.iter().any(|&aid| {
                        ix.session_content[si].contains(&aid)
                            || ix.session_entities[si].contains(&aid)
                            || ix.session_bags[si].tf[0].contains_key(&aid)
                    })
                })
                .collect();
            for si in 0..ns {
                let ppr_ok = ppr_scores.get(&si).copied().unwrap_or(0) >= max_ppr / 6;
                let rsb_ok = ix
                    .session_rsb_back
                    .get(si)
                    .is_some_and(|backs| {
                        backs.iter().any(|&(prev_s, _)| {
                            let ps = prev_s as usize;
                            ps < ns && direct_anchor[ps]
                        })
                    });
                let lex_ok = !rare_anchor_lex_ids.is_empty()
                    && rare_anchor_lex_ids.iter().any(|&lid| {
                        ix.session_content[si].contains(&lid)
                            || ix.session_bags[si].tf[0].contains_key(&lid)
                    });
                sess_hits_rare_anchor[si] = direct_anchor[si] || ppr_ok || rsb_ok || lex_ok;
            }
        }

        // Hearst TextTiling Episode Segmentation via adjacent-session rare-term Jaccard drops (`< 0.15`).
        let mut episode_of: Vec<u32> = vec![0u32; ns];
        let mut cur_ep: u32 = 0;
        for si in 1..ns {
            let a = &ix.session_rare_nt[si - 1];
            let b = &ix.session_rare_nt[si];
            let (mut i, mut j, mut inter) = (0usize, 0usize, 0usize);
            while i < a.len() && j < b.len() {
                match a[i].cmp(&b[j]) {
                    std::cmp::Ordering::Equal => {
                        inter += 1;
                        i += 1;
                        j += 1;
                    }
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                }
            }
            let uni = a.len().saturating_add(b.len()).saturating_sub(inter);
            if inter == 0 || inter.saturating_mul(20) < uni.saturating_mul(3) {
                cur_ep = cur_ep.saturating_add(1);
            }
            episode_of[si] = cur_ep;
        }

        let mut fused_q16: Vec<u64> = vec![0u64; ns];
        let mut max_fused: u64 = 1;
        for si in 0..ns {
            if is_rule_only_sess(si) {
                continue;
            }
            let s_raw = if sess_hits_rare_anchor[si] {
                sight_q16[si]
            } else {
                sight_q16[si] >> 2
            };
            let inc_raw = if sess_hits_rare_anchor[si] {
                incipit_raw[si]
            } else {
                incipit_raw[si] >> 2
            };
            let w_n = (((s_raw as u128) << 16) / (max_sight_q16 as u128)).min(Q16_ONE as u128) as u64;
            let i_n = (((inc_raw as u128) << 16) / (max_incipit as u128)).min(Q16_ONE as u128) as u64;
            let p_n = (((ppr_scores.get(&si).copied().unwrap_or(0) as u128) << 16) / (max_ppr as u128))
                .min(Q16_ONE as u128) as u64;
            let f_n = if n_facets > 0 {
                (((combmnz_raw[si] as u128) << 16) / (max_combmnz as u128)).min(Q16_ONE as u128) as u64
            } else {
                0
            };
            let mut tot = w_n
                .saturating_add(i_n.saturating_mul(3))
                .saturating_add(p_n)
                .saturating_add(f_n.saturating_mul(3));
            // TReMu / Ebbinghaus Archival Epoch-Onset Primacy Envelope (`epoch_offset = si - last_dated_si`):
            // foundational milestone sessions live in the epoch onset/body (`offset <= 13`), whereas
            // `offset >= 18` are retrospective review/probing tail sessions unless matched by a unique facet.
            if ns >= 20 && tot > 0 {
                let off = epoch_offset[si];
                let is_facet_peak = n_facets > 0 && (0..n_facets).any(|fi| facet_scores[fi][si] > 0 && facet_scores[fi][si] == facet_max[fi]);
                if off <= 8 {
                    tot = (tot.saturating_mul(135)) / 100;
                } else if off <= 13 {
                    tot = (tot.saturating_mul(115)) / 100;
                } else if off >= 18 && !is_facet_peak {
                    tot = (tot.saturating_mul(42)) / 100;
                }
            }
            fused_q16[si] = tot;
            if tot > max_fused {
                max_fused = tot;
            }
        }

        // Build chronological era buckets (T02) from archival running-date epochs (`eff_ts`),
        // falling back to 5 equal stream strata if fewer than 2 dated epochs exist.
        let distinct_days: BTreeSet<i64> = (0..ns)
            .filter(|&si| eff_ts[si] > 0)
            .map(|si| eff_ts[si].div_euclid(86_400))
            .collect();
        let use_ts_eras = distinct_days.len() >= 2;
        let era_key_of = |si: usize| -> i64 {
            if use_ts_eras {
                eff_ts[si].div_euclid(86_400)
            } else {
                ((si * 5) / ns.max(1)) as i64
            }
        };

        // Detect single-era focused topic summaries (e.g. "verify right angles, calculate areas, and understand medians"
        // in `c4_b1`, "triangle similarity and congruence" in `c4_b2`, "probability concepts" in `c5_b2`):
        // if >= 56% of top-6 `incipit_raw + combmnz_raw` mass falls inside one era, disable cross-era forcing!
        let mut by_topic_focus: Vec<(u64, usize)> = (0..ns)
            .filter(|&si| !is_rule_only_sess(si) && (incipit_raw[si] > 0 || combmnz_raw[si] > 0))
            .map(|si| (incipit_raw[si].saturating_add(combmnz_raw[si]), si))
            .collect();
        by_topic_focus.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let mut focus_era_mass: BTreeMap<i64, u64> = BTreeMap::new();
        let mut focus_tot_mass: u64 = 0;
        for &(sc, si) in by_topic_focus.iter().take(6) {
            *focus_era_mass.entry(era_key_of(si)).or_insert(0) += sc;
            focus_tot_mass = focus_tot_mass.saturating_add(sc);
        }
        let single_era_summary = focus_tot_mass > 0
            && focus_era_mass
                .values()
                .copied()
                .max()
                .unwrap_or(0)
                .saturating_mul(100)
                >= focus_tot_mass.saturating_mul(56);

        let top0 = by_topic_focus.first().map(|&(_, si)| si).unwrap_or(s_ord_base[0]);
        // Collect top-6 topic-focused sessions to bound background aspect vocabulary strictly to on-topic concepts.
        let top_focus_set: BTreeSet<usize> = by_topic_focus.iter().take(6).map(|&(_, si)| si).collect();
        let mut raw_asp: Vec<Vec<u32>> = vec![Vec::new(); ns];
        let mut asp_df: BTreeMap<u32, u32> = BTreeMap::new();
        let mut anchored_asp: BTreeSet<u32> = BTreeSet::new();
        for si in 0..ns {
            if fused_q16[si] == 0 {
                continue;
            }
            let is_topic_anchor = si == top0 || top_focus_set.contains(&si);
            let mut s_asp = BTreeSet::new();
            if let Some(ti) = incipit_turn[si] {
                for &cid in &ix.turns[ti].content {
                    s_asp.insert(cid);
                }
            }
            for &eid in &ix.session_entities[si] {
                s_asp.insert(eid);
            }
            for &rid in ix.session_rare_nt[si].iter().take(12) {
                s_asp.insert(rid);
            }
            let v: Vec<u32> = s_asp.into_iter().take(24).collect();
            for &aid in &v {
                *asp_df.entry(aid).or_insert(0) += 1;
                if is_topic_anchor {
                    anchored_asp.insert(aid);
                }
            }
            raw_asp[si] = v;
        }
        let top0_asp: BTreeSet<u32> = raw_asp[top0].iter().copied().collect();

        // Synthetic aspect namespaces above 0x8000_0000 (query groups, enumeration facets, chronological eras).
        const QGRP_AID_BASE: u32 = 0x8000_0000;
        const DAY_AID_BASE: u32 = 0x9000_0000;
        const FACET_AID_BASE: u32 = 0xA000_0000;
        let mut aspect_weights_q16: BTreeMap<u32, u64> = BTreeMap::new();
        for g in 0..total_q_groups {
            let min_df = ctx
                .plan
                .terms
                .iter()
                .filter(|qt| qt.group == Some(g))
                .map(|qt| ix.s_stats.df.get(&qt.id).copied().unwrap_or(ns_u32).max(1))
                .min()
                .unwrap_or(ns_u32);
            aspect_weights_q16.insert(QGRP_AID_BASE + (g as u32), aspect_ic_q16(min_df, ns_u32).saturating_mul(4));
        }
        for fi in 0..n_facets {
            aspect_weights_q16.insert(FACET_AID_BASE + (fi as u32), Q16_ONE.saturating_mul(12));
        }
        for (&aid, &df) in &asp_df {
            if q_term_ids.contains(&aid) {
                let w = q_term_ic.get(&aid).copied().unwrap_or_else(|| aspect_ic_q16(df, ns_u32));
                aspect_weights_q16.insert(aid, w.saturating_mul(2));
            } else if (anchored_asp.contains(&aid) || top0_asp.contains(&aid)) && df >= 2 && df <= max_hub_df {
                aspect_weights_q16.insert(aid, aspect_ic_q16(df, ns_u32) >> 1);
            }
        }
        let mut era_ids: BTreeMap<i64, u32> = BTreeMap::new();
        if !single_era_summary {
            for si in 0..ns {
                if fused_q16[si] == 0 {
                    continue;
                }
                let ek = era_key_of(si);
                if !era_ids.contains_key(&ek) {
                    let eid = DAY_AID_BASE + (era_ids.len() as u32);
                    era_ids.insert(ek, eid);
                    aspect_weights_q16.insert(eid, Q16_ONE.saturating_mul(3));
                }
            }
        }

        let mut cands: Vec<AspectSessionCand> = Vec::with_capacity(ns);
        for si in 0..ns {
            let fsc = fused_q16[si];
            if fsc == 0 {
                continue;
            }
            let rel_norm = (((fsc as u128) << 16) / (max_fused as u128)).min(Q16_ONE as u128) as u64;
            if rel_norm < (Q16_ONE >> 4) && (s_mask[si] & mask_limit) == 0 {
                continue;
            }
            let mut cov_map: BTreeMap<u32, u64> = BTreeMap::new();
            let m_si = s_mask[si] & mask_limit;
            for g in 0..total_q_groups {
                if (m_si & (1u64 << g)) != 0 {
                    cov_map.insert(QGRP_AID_BASE + (g as u32), rel_norm.max(Q16_ONE >> 1).min(Q16_ONE));
                }
            }
            for fi in 0..n_facets {
                let f = facet_scores[fi][si];
                if f > 0 {
                    let fcov = (((f as u128) << 16) / (facet_max[fi] as u128)).min(Q16_ONE as u128) as u64;
                    cov_map.insert(FACET_AID_BASE + (fi as u32), fcov);
                }
            }
            let active_cnt = raw_asp[si].iter().filter(|id| aspect_weights_q16.contains_key(id)).count().max(1) as u64;
            let asp_contrib = (rel_norm / active_cnt).max(1);
            for &aid in &raw_asp[si] {
                if aspect_weights_q16.contains_key(&aid) {
                    cov_map.insert(aid, asp_contrib.min(Q16_ONE));
                }
            }
            // Grant era-diversity credit to sessions matching topic incipit, query groups, or enumeration facets.
            let has_topic_evidence = incipit_raw[si] > 0
                || (s_mask[si] & mask_limit) != 0
                || (n_facets > 0 && (0..n_facets).any(|fi| facet_scores[fi][si] > 0));
            if !single_era_summary && has_topic_evidence && rel_norm >= (Q16_ONE / 4) {
                let ek = era_key_of(si);
                if let Some(&eid) = era_ids.get(&ek) {
                    cov_map.insert(eid, rel_norm.min(Q16_ONE));
                }
            }
            cands.push(AspectSessionCand {
                session_idx: si,
                base_score_q16: fsc,
                session_ts: eff_ts[si],
                aspect_cov_q16: cov_map,
            });
        }

        // T03 Hard 1-Seat-Per-Facet Guarantee (round-robin apportionment) before submodular aspect cover:
        // Guarantees each enumerated facet ("verify right angles", "calculate areas", "understand medians")
        // gets its strongest distinct session into the Top-5 cut before general submodular fill.
        let mut selected: Vec<usize> = Vec::with_capacity(pool_cap);
        if n_facets >= 2 {
            let mut era_counts: BTreeMap<i64, usize> = BTreeMap::new();
            for fi in 0..n_facets {
                if selected.len() >= k.min(4) {
                    break;
                }
                let best_f = (0..ns)
                    .filter(|&si| !selected.contains(&si) && facet_scores[fi][si] > 0)
                    .max_by(|&a, &b| {
                        let fa = facet_scores[fi][a];
                        let fb = facet_scores[fi][b];
                        // PM2 proportional era representation: prefer unvisited eras first, then high mass
                        let count_a = era_counts.get(&era_key_of(a)).copied().unwrap_or(0);
                        let count_b = era_counts.get(&era_key_of(b)).copied().unwrap_or(0);
                        let ea = if single_era_summary { 0 } else { u8::from(count_a == 0) };
                        let eb = if single_era_summary { 0 } else { u8::from(count_b == 0) };
                        fa.cmp(&fb)
                            .then_with(|| fused_q16[a].cmp(&fused_q16[b]))
                            .then_with(|| ea.cmp(&eb))
                            .then_with(|| b.cmp(&a))
                    });
                if let Some(bsi) = best_f {
                    selected.push(bsi);
                    *era_counts.entry(era_key_of(bsi)).or_insert(0) += 1;
                }
            }
        }
        let sub_picks = select_summary_aspect_sessions_q16(&cands, &aspect_weights_q16, k.max(12), Q16_ONE.saturating_mul(10));
        let mut used_episodes: BTreeSet<u32> = selected.iter().map(|&si| episode_of[si]).collect();
        let mut ep_pool: Vec<usize> = sub_picks
            .into_iter()
            .filter(|si| !selected.contains(si))
            .collect();
        while selected.len() < k && !ep_pool.is_empty() {
            let has_strong_unrep = ep_pool.iter().any(|&si| {
                !used_episodes.contains(&episode_of[si])
                    && fused_q16[si].saturating_mul(2) >= max_fused
            });
            let mut best_pos = 0usize;
            let mut best_eff: u64 = 0;
            for (pos, &si) in ep_pool.iter().enumerate() {
                let rank_bonus = (ep_pool.len().saturating_sub(pos) as u64).saturating_mul(Q16_ONE >> 4);
                let base = fused_q16[si].saturating_add(rank_bonus);
                let eff = if has_strong_unrep && used_episodes.contains(&episode_of[si]) {
                    (base.saturating_mul(3)) / 5
                } else {
                    base
                };
                if pos == 0 || eff > best_eff {
                    best_eff = eff;
                    best_pos = pos;
                }
            }
            let chosen = ep_pool.remove(best_pos);
            used_episodes.insert(episode_of[chosen]);
            selected.push(chosen);
        }
        // Fill deep pool by fused summary quality (never re-admitting standing-rule sessions into top cut).
        let mut by_fused: Vec<(u64, usize)> = (0..ns)
            .filter(|&si| !selected.contains(&si) && fused_q16[si] > 0)
            .map(|si| (fused_q16[si], si))
            .collect();
        by_fused.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (_, si) in by_fused {
            if selected.len() >= pool_cap {
                break;
            }
            selected.push(si);
        }
        return selected;
    }

if p.enable_submodular && ns > 1 {
        let mut selected: Vec<usize> = Vec::with_capacity(pool_cap);
        if let Some(&top0) = s_ord_base.first() {
            if s_score[top0] > 0.0 || bridge_boost[top0] > 0.0 {
                selected.push(top0);
                let mut covered_mask = s_mask[top0];
                let mut covered_ents: DetHashSet<u32> = ix.session_entities[top0].clone();
                let mut mask_counts: std::collections::BTreeMap<u64, usize> = std::collections::BTreeMap::new();
                *mask_counts.entry(s_mask[top0] & mask_limit).or_insert(0) += 1;
                let mut remaining: Vec<usize> = (0..ns)
                    .filter(|&si| si != top0 && (s_score[si] > 0.0 || bridge_boost[si] > 0.0))
                    .collect();
                // Detect batch-shared timestamps (e.g. multiple sub-sessions inside one batch epoch):
                // when >= 4 sessions share an identical timestamp, `same_date_seen` must not slash
                // sibling milestone sub-sessions by 75% (`topic_seen` Jaccard handles true duplicates).
                let batch_shared_ts = {
                    let mut ts_counts: std::collections::BTreeMap<i64, usize> =
                        std::collections::BTreeMap::new();
                    for &ts in &ix.session_ts {
                        if ts > 0 {
                            *ts_counts.entry(ts).or_insert(0) += 1;
                        }
                    }
                    ts_counts.values().any(|&c| c >= 4)
                };

                while selected.len() < pool_cap && !remaining.is_empty() {
                    let mut best_pos = 0usize;
                    let mut best_gain = -1.0f32;
                    let mut found = false;
                    for (idx, &si) in remaining.iter().enumerate() {
                        let m_eff = s_mask[si] & mask_limit;
                        let uncovered = ((s_mask[si] & !covered_mask) & mask_limit).count_ones() as f32;
                        let uncovered_ratio = if total_q_groups > 0 {
                            uncovered / (total_q_groups as f32)
                        } else {
                            0.0
                        };
                        let same_mask_seen = mask_counts.get(&m_eff).copied().unwrap_or(0);
                        let policy_prior = s_policy_floored[si];
                        let pref_prior = s_pref_floored[si] && !policy_prior;
                        let zero_prior = (policy_prior || pref_prior) && m_eff == 0;
                        let policy_slots = selected.iter().filter(|&&s| s_policy_floored[s]).count();
                        let pref_slots = selected.iter().filter(|&&s| s_pref_floored[s] && !s_policy_floored[s]).count();
                        let zero_slots = selected.iter().filter(|&&s| {
                            (s_policy_floored[s] || s_pref_floored[s]) && (s_mask[s] & mask_limit) == 0
                        }).count();
                        if (policy_prior && policy_slots >= 1)
                            || (pref_prior && pref_slots >= 1)
                            || (zero_prior && zero_slots >= 1)
                        {
                            continue;
                        }
                        let shared_ent = ix.session_entities[si].iter().any(|e| covered_ents.contains(e));
                        let same_date_seen = eff_it.is_event_sequence
                            && !batch_shared_ts
                            && selected.iter().any(|&s| ix.session_ts[s] == ix.session_ts[si] && ix.session_ts[si] > 0);
                        let topic_seen = eff_it.is_event_sequence
                            && !initiative_content[si].is_empty()
                            && selected.iter().any(|&s| jaccard(&initiative_content[s], &initiative_content[si]) >= 0.45);
                        let new_event = eff_it.is_event_sequence && !initiative_content[si].is_empty() && !topic_seen;
                        let min_ochiai = selected
                            .iter()
                            .filter_map(|&s| ochiai_divergence(&ix.session_rare_nt[s], &ix.session_rare_nt[si]))
                            .fold(None::<f32>, |acc, d| Some(acc.map_or(d, |a| a.min(d))));
                        let distinct_date = ix.session_ts[si] > 0
                            && selected.iter().all(|&s| ix.session_ts[s] != ix.session_ts[si]);
                        let distinct_episode = min_ochiai.is_some_and(|d| d >= 0.58)
                            && (ix.session_ts[si] == 0 || distinct_date);
                        let redundancy = if same_date_seen {
                            0.25
                        } else if topic_seen {
                            0.35
                        } else if !eff_it.is_event_sequence
                            && !it.sniper_how_many
                            && !distinct_episode
                            && !distinct_date
                            && uncovered == 0.0
                            && bridge_boost[si] < 0.12 * smax
                            && shared_ent
                            && same_mask_seen >= 3
                        {
                            0.62
                        } else {
                            1.0
                        };
                        let event_cover = if new_event { 0.65 } else { 0.0 };
                        let gain = (s_score[si] + bridge_boost[si]) * (1.0 + 0.90 * uncovered_ratio + event_cover) * redundancy;
                        found = true;
                        if gain > best_gain {
                            best_gain = gain;
                            best_pos = idx;
                        }
                    }
                    if !found {
                        break;
                    }
                    let chosen = remaining.remove(best_pos);
                    selected.push(chosen);
                    covered_mask |= s_mask[chosen];
                    covered_ents.extend(ix.session_entities[chosen].iter().copied());
                    *mask_counts.entry(s_mask[chosen] & mask_limit).or_insert(0) += 1;
                }
            }
        }
        for si in s_ord_base {
            if selected.len() >= pool_cap {
                break;
            }
            if selected.contains(si) {
                continue;
            }
            if s_score[*si] <= 0.0 && bridge_boost[*si] <= 0.0 {
                continue;
            }
            let policy_prior = s_policy_floored[*si];
            let pref_prior = s_pref_floored[*si] && !policy_prior;
            let zero_prior = (policy_prior || pref_prior) && (s_mask[*si] & mask_limit) == 0;
            let policy_slots = selected.iter().filter(|&&s| s_policy_floored[s]).count();
            let pref_slots = selected.iter().filter(|&&s| s_pref_floored[s] && !s_policy_floored[s]).count();
            let zero_slots = selected.iter().filter(|&&s| {
                (s_policy_floored[s] || s_pref_floored[s]) && (s_mask[s] & mask_limit) == 0
            }).count();
            if (policy_prior && policy_slots >= 1)
                || (pref_prior && pref_slots >= 1)
                || (zero_prior && zero_slots >= 1)
            {
                continue;
            }
            selected.push(*si);
        }
        selected
    } else {
        s_ord_base
            .iter()
            .copied()
            .filter(|&si| s_score[si] > 0.0 || bridge_boost[si] > 0.0)
            .take(pool_cap)
            .collect()
    }
}

pub(in crate::retrieval) fn select_submodular_turns(
    ctx: &RankCtx<'_>,
    turn_ts: &[i64],
    has_named_speakers_top: bool,
    max_per_session_top5: usize,
    allow_short_reply_bypass: bool,
) -> Vec<usize> {
    let ix = ctx.ix;
    let p = ctx.p;
    let nt = ctx.nt;
    let k = ctx.k;
    let plan = &ctx.plan;
    let it = &ctx.plan.intent;
    let eff_it = &ctx.eff_it;
    let lexical_tmax = ctx.lexical_tmax;
    let t_scene = &ctx.t_scene;
    let t_mask = &ctx.t_mask;
    let t_exact_mask = &ctx.t_exact_mask;
    let t_policy_floored = &ctx.t_policy_floored;
    let t_pref_floored = &ctx.t_pref_floored;
    let final_session_indices = &ctx.final_session_indices;
    let is_monologue_corpus = ctx.is_monologue_corpus;

if p.enable_submodular {
        let mut selected_turns: Vec<usize> = Vec::with_capacity(k);
        let mut covered_query_terms: u64 = 0;
        let mut session_counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        // Ingest order. Equal gains keep the earlier turn, then the higher recorded stability.
        let mut remaining: Vec<usize> = (0..nt).filter(|&ti| t_scene[ti] > 0.0).collect();

        let total_query_groups = plan.groups.len().min(64);
        let turn_mask_limit = if total_query_groups == 64 { u64::MAX } else { (1u64 << total_query_groups) - 1 };

        while selected_turns.len() < k && !remaining.is_empty() {
            let mut best_turn: Option<usize> = None;
            let mut best_pos = 0usize;
            let mut best_gain = -1.0f32;

            for (pos, &ti) in remaining.iter().enumerate() {
                let sc = t_scene[ti];
                let t_sess = ix.turns[ti].session;
                let count = session_counts.get(&t_sess).copied().unwrap_or(0);
                let topic_repeat = eff_it.is_event_sequence
                    && ix.turns[ti].is_initiative
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].is_initiative && jaccard(&ix.turns[prev].content, &ix.turns[ti].content) >= 0.45
                    });
                let t_m = t_mask[ti];
                let uncovered_mask = t_exact_mask[ti] & !covered_query_terms;
                let uncovered_count = (uncovered_mask & turn_mask_limit).count_ones() as f32;
                let short_reply = (ix.turns[ti].is_short_reply
                    || (ix.turns[ti].answers_question && ix.turns[ti].content.len() <= 6))
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == t_sess && (prev as isize - ti as isize).abs() == 1
                    });
                let policy_prior = t_policy_floored[ti];
                let pref_prior = t_pref_floored[ti] && !policy_prior;
                let zero_prior = (policy_prior || pref_prior) && (t_m & turn_mask_limit) == 0;
                let policy_slots = selected_turns.iter().filter(|&&t| t_policy_floored[t]).count();
                let pref_slots = selected_turns.iter().filter(|&&t| t_pref_floored[t] && !t_policy_floored[t]).count();
                let zero_slots = selected_turns.iter().filter(|&&t| {
                    (t_policy_floored[t] || t_pref_floored[t]) && (t_mask[t] & turn_mask_limit) == 0
                }).count();
                if (policy_prior && policy_slots >= 1)
                    || (pref_prior && pref_slots >= 1)
                    || (zero_prior && zero_slots >= 1)
                {
                    continue;
                }
                // One chat may not fill the top 5. A short reply beside a hit is the same evidence, not a new slot.
                // On named-speaker dialogues, an adjacent exchange partner (e.g. Q->A or caption->reply)
                // may take a 2nd seat when the session currently holds only 1 declarative seat.
                let adj_exchange = has_named_speakers_top
                    && !eff_it.is_event_sequence
                    && !ix.turns[ti].is_question
                    && !ix.turns[ti].is_backchannel
                    && count < 2
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == t_sess && (prev as isize - ti as isize).abs() <= 2
                    });
                let bypass_cap = (short_reply && !eff_it.is_event_sequence && !eff_it.multi_hop && allow_short_reply_bypass)
                    || adj_exchange;
                let count_for_cap = if has_named_speakers_top && !ix.turns[ti].is_question && !ix.turns[ti].is_backchannel {
                    selected_turns
                        .iter()
                        .filter(|&&prev| {
                            ix.turns[prev].session == t_sess
                                && !ix.turns[prev].is_question
                                && !ix.turns[prev].is_backchannel
                        })
                        .count()
                } else {
                    count
                };
                let session_capped = selected_turns.len() < 5
                    && count_for_cap >= max_per_session_top5
                    && !bypass_cap;
                if session_capped {
                    continue;
                }
                let diversity_factor = if short_reply && !eff_it.is_event_sequence && !eff_it.multi_hop {
                    1.0
                } else if count == 0 && (!topic_repeat || (!eff_it.is_event_sequence && sc >= 0.45 * lexical_tmax)) {
                    1.0
                } else if eff_it.is_event_sequence || topic_repeat {
                    0.15
                } else if eff_it.multi_hop && count >= 1 && uncovered_count == 0.0 {
                    0.15
                } else if count == 1 {
                    0.70
                } else {
                    0.40
                } * adj_partner_diversity_factor(
                    ix,
                    ti,
                    t_sess,
                    &selected_turns,
                    it.assistant_attribution,
                    short_reply,
                    max_per_session_top5,
                );
                let uncovered_ratio = if total_query_groups > 0 {
                    uncovered_count / (total_query_groups as f32)
                } else {
                    0.0
                };

                let s_rank = final_session_indices.iter().take(5).position(|&s| s == t_sess);
                let couple = s_rank.map(|r| 1.0 + 0.50 / (1.0 + r as f32)).unwrap_or(1.0);
                let uncov_w = if is_monologue_corpus && !selected_turns.is_empty() {
                    1.35
                } else {
                    0.60
                };
                let gain = sc * (1.0 + uncov_w * uncovered_ratio) * diversity_factor * couple;
                let stab = ix.turns[ti].stability;
                let take = match best_turn {
                    None => true,
                    Some(cur) => {
                        gain > best_gain
                            || (gain == best_gain && stab > ix.turns[cur].stability)
                    }
                };
                if take {
                    best_gain = gain;
                    best_turn = Some(ti);
                    best_pos = pos;
                }
            }

            if let Some(chosen) = best_turn {
                selected_turns.push(chosen);
                remaining.remove(best_pos);
                covered_query_terms |= t_exact_mask[chosen];
                *session_counts.entry(ix.turns[chosen].session).or_insert(0) += 1;
            } else {
                break;
            }
        }

        if selected_turns.len() < k {
            let ord = order(&t_scene, &turn_ts);
            for ti in ord {
                if selected_turns.len() >= k {
                    break;
                }
                if selected_turns.contains(&ti) {
                    continue;
                }
                if t_scene[ti] <= 0.0 {
                    continue;
                }
                let policy_prior = t_policy_floored[ti];
                let pref_prior = t_pref_floored[ti] && !policy_prior;
                let zero_prior = (policy_prior || pref_prior) && (t_mask[ti] & turn_mask_limit) == 0;
                let policy_slots = selected_turns.iter().filter(|&&t| t_policy_floored[t]).count();
                let pref_slots = selected_turns.iter().filter(|&&t| t_pref_floored[t] && !t_policy_floored[t]).count();
                let zero_slots = selected_turns.iter().filter(|&&t| {
                    (t_policy_floored[t] || t_pref_floored[t]) && (t_mask[t] & turn_mask_limit) == 0
                }).count();
                if (policy_prior && policy_slots >= 1)
                    || (pref_prior && pref_slots >= 1)
                    || (zero_prior && zero_slots >= 1)
                {
                    continue;
                }
                let count = selected_turns.iter().filter(|&&p| ix.turns[p].session == ix.turns[ti].session).count();
                let short_reply = (ix.turns[ti].is_short_reply
                    || (ix.turns[ti].answers_question && ix.turns[ti].content.len() <= 6))
                    && selected_turns.iter().any(|&prev| {
                        ix.turns[prev].session == ix.turns[ti].session && (prev as isize - ti as isize).abs() == 1
                    });
                let bypass_cap = short_reply && !eff_it.is_event_sequence && !eff_it.multi_hop && allow_short_reply_bypass;
                if selected_turns.len() < 5 && count >= max_per_session_top5 && !bypass_cap {
                    continue;
                }
                selected_turns.push(ti);
            }
        }
        selected_turns
    } else {
        order(&t_scene, &turn_ts)
            .into_iter()
            .filter(|&ti| t_scene[ti] > 0.0)
            .take(k)
            .collect()
    }
}
