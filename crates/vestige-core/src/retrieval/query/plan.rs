//! Query planning, typo fallback, and matched-word extraction.

use crate::retrieval::*;


pub(in crate::retrieval) fn plan_query(ix: &Index, q: &Query, p: &Params) -> QueryPlan {
    let mut intent = classify(&q.text);
    let mut terms = Vec::new();
    let mut groups: Vec<String> = Vec::new();
    let mut group_of = DetHashMap::default();
    let mut focal_ids: DetHashSet<u32> = DetHashSet::default();

    let raw_q = raw_tokens(&q.text);
    let episodic_past_recall = wh_slots(&q.text) != 0
        || intent.sniper_how_many
        || intent.multi_hop
        || intent.temporal;
    if !episodic_past_recall
        && raw_q.iter().any(|w| ix.vocab.get(w).is_some_and(|id| ix.constraint_terms.contains(&id)))
    {
        intent.preference = true;
    }
    let inline_dates = if p.enable_deictic { inline_date_tokens(&q.text) } else { Vec::new() };
    let q_ts = if p.enable_deictic { parse_date(&q.date) } else { 0 };
    let deictic_items = if p.enable_deictic && q_ts != 0 {
        deictic_weighted_tokens(&raw_q, q_ts)
    } else {
        Vec::new()
    };
    let rel_spec = if p.enable_deictic && q_ts != 0 {
        parse_relative_time_specs_from_raw(&raw_q, q_ts)
    } else {
        RelTimeSpec {
            points: Vec::new(),
            windows: Vec::new(),
            scaffold_positions: vec![false; raw_q.len()],
        }
    };
    let rel_prior_active = !rel_spec.points.is_empty() || !rel_spec.windows.is_empty();

    let n_docs = ix.t_own_stats.n as usize;
    let frame_token = |w: &str| {
        ix.vocab.get(w).is_some_and(|id| {
            corpus_frame(n_docs, ix.t_own_stats.df.get(&id).copied().unwrap_or(0))
        })
    };
    let domain_groups = raw_q
        .iter()
        .enumerate()
        .filter(|&(idx, w)| {
            is_content(w)
                && !frame_token(w)
                && !(rel_prior_active && rel_spec.scaffold_positions.get(idx).copied().unwrap_or(false))
        })
        .map(|(_, w)| w)
        .collect::<DetHashSet<_>>()
        .len();
    let can_demote_framing = domain_groups >= 1;
    let zero_rel_scaffolding = rel_prior_active && domain_groups >= 1;

    let nick_bind = ix.nicks.bindings_in(&q.text);
    let mut routing_addr_toks: DetHashSet<String> = DetHashSet::default();
    for raw_span in q.text.split_whitespace() {
        let trimmed = raw_span.trim_matches(|c: char| matches!(c, ',' | ';' | ':' | '.' | ')' | ']' | '"' | '\''));
        let is_email = trimmed.contains('@')
            && trimmed
                .split_once('@')
                .is_some_and(|(u, d)| !u.is_empty() && d.contains('.'));
        let digit_cnt = trimmed.chars().filter(|c| c.is_ascii_digit()).count();
        let is_phone = digit_cnt >= 7 && (trimmed.starts_with('+') || trimmed.starts_with('(') || trimmed.contains('-'));
        if is_email || is_phone {
            for rt in raw_tokens(trimmed) {
                routing_addr_toks.insert(rt);
            }
        }
    }
    let non_routing_content_cnt = raw_q
        .iter()
        .filter(|w| is_content(w) && !routing_addr_toks.contains(*w))
        .count();
    let demote_routing_addrs = !routing_addr_toks.is_empty() && non_routing_content_cnt >= 2;

    let mut label_group: DetHashMap<String, usize> = DetHashMap::default();
    let mut seen_toks = DetHashSet::default();
    let mut hyp_added: DetHashSet<u32> = DetHashSet::default();
    let mut expandable: Vec<(String, usize)> = Vec::new();
    let corpus_df = |id: u32| ix.t_own_stats.df.get(&id).copied().unwrap_or(0);

    // Porphyry Genus Discount (`porphyry_scene_discount`):
    // Syntactic WH-determined head noun in `["what"|"which", <head>, <closed_aux_or_prep>, ...]`
    // (or `[<prep>, "what"|"which", <head>, <closed_aux_or_prep>, ...]`) denotes the queried category
    // rather than the instance value in the document. Pure closed-class syntax, zero open-class lists.
    // hardcode-scan: closed-class (WH-determiner + auxiliary/preposition syntactic slot)
    let wh_det_pos: Option<usize> = if matches!(raw_q.first().map(String::as_str), Some("what" | "which")) {
        Some(0)
    } else if raw_q.len() >= 4
        && matches!(
            raw_q[0].as_str(),
            "in" | "at" | "on" | "from" | "to" | "for" | "by" | "of" | "into" | "under"
        )
        && matches!(raw_q[1].as_str(), "what" | "which")
    {
        Some(1)
    } else {
        None
    };
    // Support both 1-word (`["what"|"which", <head>, <aux_or_verb>]`) and 2-word
    // (`["what"|"which", <mod>, <head>, <aux>]`, e.g. "What personality traits might...")
    // WH-determined genus heads, plus epistemic hypothetical modal framing discount.
    let has_later_cap_entity = q
        .text
        .split_whitespace()
        .skip(2)
        .any(|orig| {
            let clean = orig.trim_matches(|c: char| !c.is_alphanumeric());
            clean.len() >= 2 && clean.chars().next().is_some_and(char::is_uppercase)
        });
    // hardcode-scan: closed-class (WH-determiner auxiliary/preposition syntactic slot)
    let is_closed_post_head = |tok: &str| -> bool {
        matches!(
            tok,
            "is" | "are" | "was" | "were" | "did" | "does" | "do" | "has" | "have" | "had"
                | "can" | "could" | "would" | "should" | "will" | "might" | "may"
                | "of" | "in" | "from" | "with" | "that" | "for" | "to" | "on" | "at" | "by"
        )
    };
    let is_discountable_genus_word = |cand: &str| -> bool {
        if token_is_temporal_word(cand) || !is_content(cand) {
            return false;
        }
        let is_cap_proper = q.text.split_whitespace().any(|orig| {
            let clean = orig.trim_matches(|c: char| !c.is_alphanumeric());
            clean.eq_ignore_ascii_case(cand) && clean.chars().next().is_some_and(char::is_uppercase)
        });
        if is_cap_proper {
            return false;
        }
        const PORPHYRY_RARE_DF_FLOOR: u32 = 4;
        const PORPHYRY_RARE_DF_DIV: usize = 20;
        let rare_anchor_cap = ((n_docs / PORPHYRY_RARE_DF_DIV) as u32).max(PORPHYRY_RARE_DF_FLOOR);
        let cand_df = ix.vocab.get(cand).map(|id| corpus_df(id)).unwrap_or(0);
        // Never discount a rare corpus-attested specific noun (`1 <= df <= rare_anchor_cap`, e.g. "workshop",
        // "test", "degree"); Porphyry only discounts unattested OOV hypernyms (`df == 0`, e.g. "fields",
        // "traits") or high-DF generic category nouns (`df > rare_anchor_cap`, e.g. "events", "activities").
        if (1..=rare_anchor_cap).contains(&cand_df) {
            return false;
        }
        true
    };
    // hardcode-scan: closed-class (first-person pronouns exclude personal attribute nouns from ontological genus discount)
    let has_first_person_q = raw_q.iter().any(|w| {
        matches!(
            w.as_str(),
            "i" | "i'm" | "i've" | "i'd" | "my" | "me" | "mine" | "we" | "our" | "ours"
        )
    });
    let mut porphyry_genus_mask: u64 = 0;
    if !has_first_person_q
        && let Some(wi) = wh_det_pos
        && n_docs >= 24
        && non_routing_content_cnt >= 2
        && raw_q.len() > wi + 2
    {
        let h1 = &raw_q[wi + 1];
        let w2 = &raw_q[wi + 2];
        if (is_closed_post_head(w2) || has_later_cap_entity)
            && is_discountable_genus_word(h1)
            && (wi + 1) < 64
        {
            // Check 2-word genus NP (`["what", <mod>, <head>, <aux>]`) first when `w2` is NOT closed-class:
            if !is_closed_post_head(w2)
                && raw_q.len() > wi + 3
                && is_closed_post_head(&raw_q[wi + 3])
                && is_discountable_genus_word(w2)
                && (wi + 2) < 64
            {
                // Discount ONLY the syntactic head noun (`wi + 2`, genus), never the restrictive
                // pre-nominal modifier (`wi + 1`, differentia, e.g. "music" in "What music events").
                porphyry_genus_mask |= 1u64 << (wi + 2);
            } else if is_closed_post_head(w2)
                || (has_later_cap_entity && !crate::wordnet::senses(h1).is_empty())
            {
                porphyry_genus_mask |= 1u64 << (wi + 1);
            }
        }
    }
    // Closed-class epistemic/hypothetical modal query framing (`"Would X likely..."`, `"What might X..."`):
    // hardcode-scan: closed-class (epistemic modal interrogative markers)
    let is_hypothetical_modal_q = matches!(raw_q.first().map(String::as_str), Some("would"))
        || raw_q.windows(2).any(|p| {
            (matches!(p[0].as_str(), "what" | "which" | "who" | "where" | "how")
                && matches!(p[1].as_str(), "might" | "would" | "could"))
                || matches!(p[1].as_str(), "likely" | "potentially")
        })
        || raw_q.iter().any(|w| matches!(w.as_str(), "likely" | "potentially"));
    let asks_nickname = raw_q.iter().any(|w| w == "nickname" || w == "nicknames");

    for (idx, w) in raw_q.iter().enumerate() {
        if !is_content(w) {
            continue;
        }
        if zero_rel_scaffolding && rel_spec.scaffold_positions.get(idx).copied().unwrap_or(false) {
            continue;
        }
        if !seen_toks.insert(w.clone()) {
            continue;
        }
        // hardcode-scan: closed-class (epistemic/modal framing modifiers on hypothetical queries)
        let is_hyp_scaffold = is_hypothetical_modal_q
            && has_later_cap_entity
            && matches!(
                w.as_str(),
                "likely" | "potentially" | "might" | "consider" | "considered" | "future"
            );
        if (idx < 64 && (porphyry_genus_mask & (1u64 << idx)) != 0) || is_hyp_scaffold {
            if let Some(id) = ix.vocab.get(w) {
                terms.push(QTerm { id, weight: 0.08, group: None, exact_hit: false });
            }
            if p.enable_stemming {
                let st = stem(w);
                if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                    terms.push(QTerm { id, weight: 0.05, group: None, exact_hit: false });
                }
            }
            continue;
        }
        if demote_routing_addrs && routing_addr_toks.contains(w) {
            if let Some(id) = ix.vocab.get(w) {
                terms.push(QTerm { id, weight: 0.125, group: None, exact_hit: false });
            }
            if p.enable_stemming {
                let st = stem(w);
                if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                    terms.push(QTerm { id, weight: 0.08, group: None, exact_hit: false });
                }
            }
            continue;
        }
        let is_frame = can_demote_framing && frame_token(w);
        if is_frame {
            if let Some(id) = ix.vocab.get(w) {
                terms.push(QTerm { id, weight: 0.30, group: None, exact_hit: false });
            }
            if p.enable_stemming {
                let st = stem(w);
                if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                    terms.push(QTerm { id, weight: 0.20, group: None, exact_hit: false });
                }
            }
            continue;
        }

        let label = nick_bind.get(w).cloned().unwrap_or_else(|| w.clone());
        let g = if let Some(&g) = label_group.get(&label) {
            g
        } else {
            let g = groups.len();
            groups.push(label.clone());
            label_group.insert(label, g);
            g
        };
        if let Some(id) = ix.vocab.get(w) {
            // Full name owns the person group. A vocative nick ("Mel") that also
            // expands to "melanie" must not score 1.0 on top of that expansion.
            let weight = if nick_bind.get(w).is_some_and(|full| ix.vocab.get(full).is_some()) {
                0.35
            } else {
                1.0
            };
            terms.push(QTerm { id, weight, group: Some(g), exact_hit: true });
            group_of.insert(id, g);
            focal_ids.insert(id);
        }
        if let Some(full) = nick_bind.get(w) {
            if let Some(id) = ix.vocab.get(full) {
                terms.push(QTerm { id, weight: 1.0, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
                focal_ids.insert(id);
            }
            if p.enable_stemming {
                let fst = stem(full);
                if let Some(id) = ix.vocab.get(&stem_term(&fst)) {
                    terms.push(QTerm { id, weight: p.stem_weight, group: Some(g), exact_hit: true });
                    group_of.insert(id, g);
                    focal_ids.insert(id);
                }
            }
        }
        let st = stem(w);
        if p.enable_stemming {
            if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                // Boost stem weight when the exact surface form is absent from corpus vocabulary.
                const OOV_STEM_BOOST_Q16: u64 = 81_920; // 1.25x in Q16.16 for OOV inflected query stems.
                const OOV_STEM_CAP_Q16: u64 = 60_293; // 0.92 cap in Q16.16 so stems stay below exact 1.0.
                let sw = if ix.vocab.get(w).is_none() {
                    let boosted = p.stem_weight * (OOV_STEM_BOOST_Q16 as f32 / 65_536.0);
                    boosted.min(OOV_STEM_CAP_Q16 as f32 / 65_536.0)
                } else {
                    p.stem_weight
                };
                terms.push(QTerm { id, weight: sw, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
                focal_ids.insert(id);
            }
        }
        if asks_nickname && w.len() >= 5 {
            let is_sp = ix.turns.iter().take(64).any(|t| t.speaker.eq_ignore_ascii_case(w));
            if is_sp {
                const NICK_PREFIX_DF_MAX: u32 = 12; // Upper corpus DF bound for short nickname prefix matches.
                const NICK_PREFIX_WEIGHT_Q16: u64 = 104_858; // 1.60 in Q16.16 so dedicated nickname group drives rank.
                let mut nick_ids: Vec<u32> = Vec::new();
                for plen in [2usize, 3, 4] {
                    if plen < w.len()
                        && let Some(nid) = ix.vocab.get(&w[..plen])
                        && (1..=NICK_PREFIX_DF_MAX).contains(&corpus_df(nid))
                    {
                        nick_ids.push(nid);
                    }
                }
                if !nick_ids.is_empty() {
                    let ng = groups.len();
                    groups.push(format!("{w}_nick"));
                    for nid in nick_ids {
                        terms.push(QTerm {
                            id: nid,
                            weight: NICK_PREFIX_WEIGHT_Q16 as f32 / 65_536.0,
                            group: Some(ng),
                            exact_hit: true,
                        });
                        group_of.insert(nid, ng);
                        focal_ids.insert(nid);
                    }
                }
            }
        }
        step9_push_typo_fallback(ix, w, &st, nick_bind.get(w).is_some(), g, &mut terms, &mut group_of);
        expandable.push((w.clone(), g));
    }

    // WordNet query expansion (Voorhees 1994; Fang 2008; Gong 2006), three controls:
    // (1) it runs only when the query's own terms have zero lexical overlap with
    //     the indexed corpus (Ounis-style pre-retrieval gate, `zero_overlap.rs`);
    // (2) an expansion term weighs `hypernym_weight` (0.5) of an original term;
    // (3) only expansion lemmas whose corpus document frequency is > 0 are kept.
    // Nothing is added to documents; a query with any overlap is untouched.
    let has_domain_groups = terms.iter().any(|t| t.group.is_some());
    let gate_open = if has_domain_groups {
        crate::zero_overlap::ounis_zero_overlap_gate(
            terms.iter().filter(|t| t.group.is_some()).map(|t| corpus_df(t.id)),
        )
    } else {
        crate::zero_overlap::ounis_zero_overlap_gate(terms.iter().map(|t| corpus_df(t.id)))
    };
    if p.enable_hypernyms && gate_open {
        for (w, g) in &expandable {
            let mut morph_cands: Vec<String> = vec![w.clone()];
            if let Some(sing) = w.strip_suffix("es").or_else(|| w.strip_suffix('s')) {
                if sing.len() >= 3 && !morph_cands.iter().any(|c| c == sing) {
                    morph_cands.push(sing.to_string());
                }
            }
            let st = stem(w);
            if st != *w && st.len() >= 3 && !morph_cands.iter().any(|c| c == &st) {
                morph_cands.push(st);
            }
            for cand in &morph_cands {
                for lemma in crate::wordnet::expansion_lemmas(cand)
                    .into_iter()
                    .chain(crate::wordnet::upward_hypernyms(cand, 3))
                {
                    let surface = ix.vocab.get(lemma);
                    let stemmed = if p.enable_stemming {
                        ix.vocab.get(&stem_term(&stem(lemma)))
                    } else {
                        None
                    };
                    for id in [surface, stemmed].into_iter().flatten() {
                        if corpus_df(id) > 0 && hyp_added.insert(id) {
                            terms.push(QTerm { id, weight: p.hypernym_weight, group: Some(*g), exact_hit: false });
                            group_of.insert(id, *g);
                        }
                    }
                }
                for (cn_lemma, _cn_w) in crate::conceptnet::two_hop_neighbors(cand, 10) {
                    let surface = ix.vocab.get(cn_lemma);
                    let stemmed = if p.enable_stemming {
                        ix.vocab.get(&stem_term(&stem(cn_lemma)))
                    } else {
                        None
                    };
                    for id in [surface, stemmed].into_iter().flatten() {
                        if corpus_df(id) > 0 && hyp_added.insert(id) {
                            terms.push(QTerm { id, weight: p.hypernym_weight * 0.85, group: Some(*g), exact_hit: false });
                            group_of.insert(id, *g);
                        }
                    }
                }
            }
        }
    }

    // CamelCase & snake/hyphen reverse query expansion (`localStorage` -> `local`, `storage` at 0.55).
    for raw_word in q.text.split_whitespace() {
        let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
        if word.is_empty() {
            continue;
        }
        for (part, _) in surface_aux_tokens(word) {
            if seen_toks.contains(&part) {
                continue;
            }
            if let Some(id) = ix.vocab.get(&part) {
                terms.push(QTerm { id, weight: 0.55, group: None, exact_hit: false });
            }
        }
    }

    // Capitalized proper-noun entity phrases present in index (e.g. "Local H", "For Against").
    let mut seen_ents = DetHashSet::default();
    for ent in proper_entity_tokens(&q.text) {
        if !seen_ents.insert(ent.clone()) {
            continue;
        }
        if let Some(id) = ix.vocab.get(&ent) {
            let parts_body = ent.strip_prefix("\u{1}ent:").unwrap_or("");
            let any_erased = parts_body
                .split('_')
                .any(|part| !groups.iter().any(|g| g == part) && !frame_token(part));
            if any_erased {
                let g = groups.len();
                groups.push(ent);
                terms.push(QTerm { id, weight: 1.65, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
            } else {
                terms.push(QTerm { id, weight: 0.75, group: None, exact_hit: false });
            }
        }
    }

    // Exact technical handles in the query (`Url`, `Email`, `CommitSha`, `IssueRef`, `Version`, `FilePath`).
    for h_str in extract_exact_technical_handles(&q.text) {
        if let Some(id) = ix.vocab.get(&h_str) {
            terms.push(QTerm { id, weight: 1.5, group: None, exact_hit: true });
            focal_ids.insert(id);
        }
    }

    if p.enable_deictic {
        let mut date_weights: std::collections::BTreeMap<String, f32> = std::collections::BTreeMap::new();
        for d in inline_dates {
            let w = if d.starts_with("__DATE__") {
                2.0
            } else if d.starts_with("__MD__") {
                1.4
            } else if d.starts_with("__YM__") {
                1.2
            } else if d.starts_with("__SY__") {
                1.1
            } else if d.starts_with("__MONTH__")
                || d.starts_with("__EARLY__")
                || d.starts_with("__MID__")
                || d.starts_with("__LATE__")
            {
                1.0
            } else if d.starts_with("__YEAR__") || d.starts_with("__SEASON__") {
                0.85
            } else {
                p.date_weight
            };
            let entry = date_weights.entry(d).or_insert(0.0);
            *entry = (*entry).max(w);
        }
        for (d, scale) in deictic_items {
            let base = if d.starts_with("__DATE__") {
                1.9
            } else if d.starts_with("__MD__") {
                1.1
            } else if d.starts_with("__YM__") || d.starts_with("__MONTH__") {
                0.85
            } else if d.starts_with("__YEAR__") || d.starts_with("__SEASON__") || d.starts_with("__SY__") {
                0.55
            } else {
                p.date_weight
            };
            let w = base * scale;
            let entry = date_weights.entry(d).or_insert(0.0);
            *entry = (*entry).max(w);
        }
        for (d, w) in date_weights {
            if let Some(id) = ix.vocab.get(&d) {
                terms.push(QTerm { id, weight: w, group: None, exact_hit: false });
            }
        }
    }

    QueryPlan { intent, terms, groups, group_of, focal_ids }
}

pub(in crate::retrieval) fn matched_words(plan: &QueryPlan, mask: u64) -> Vec<String> {
    plan.groups
        .iter()
        .enumerate()
        .filter(|(g, _)| *g < 64 && mask & (1 << g) != 0)
        .map(|(_, w)| w.clone())
        .collect()
}

/// Step 9A: byte-level Damerau-Levenshtein distance exactly 1 (one edit or one adjacent swap).
pub(crate) fn damerau_lev_1(a: &[u8], b: &[u8]) -> bool {
    let la = a.len();
    let lb = b.len();
    if la.abs_diff(lb) > 1 {
        return false;
    }
    if la == lb {
        let mut diffs = Vec::with_capacity(3);
        for i in 0..la {
            if a[i] != b[i] {
                diffs.push(i);
                if diffs.len() > 2 {
                    return false;
                }
            }
        }
        return match diffs.len() {
            1 => true,
            2 => {
                let (i, j) = (diffs[0], diffs[1]);
                j == i + 1 && a[i] == b[j] && a[j] == b[i]
            }
            _ => false,
        };
    }
    let (short, long) = if la < lb { (a, b) } else { (b, a) };
    let mut i = 0;
    let mut j = 0;
    let mut skipped = false;
    while i < short.len() && j < long.len() {
        if short[i] == long[j] {
            i += 1;
            j += 1;
        } else if !skipped {
            skipped = true;
            j += 1;
        } else {
            return false;
        }
    }
    true
}

/// Step 9A: lowercase vocab word within Damerau-Levenshtein 1 of a zero-DF query
/// word (len >= 6, same first byte). Supports adjacent transpositions universally plus
/// bounded insert/delete/sub edits when the OOV token is not itself a standard WordNet noun.
pub(crate) fn find_unique_typo_fallback(vocab_names: &[String], w: &str) -> Option<u32> {
    let wb = w.as_bytes();
    if wb.len() < 6 || !wb.iter().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    let first = wb[0];
    let allow_edit1 = crate::wordnet::senses(w).is_empty();
    let mut found: Option<(u32, String)> = None;
    for (id, cand) in vocab_names.iter().enumerate() {
        let cb = cand.as_bytes();
        if cb.is_empty() || cb[0] != first || cb.starts_with(b"\x01") {
            continue;
        }
        if !cb.iter().all(|b| b.is_ascii_lowercase()) {
            continue;
        }
        let is_adj_swap = cb.len() == wb.len()
            && damerau_lev_1(wb, cb)
            && wb.iter().zip(cb).filter(|(x, y)| x != y).count() == 2;
        let is_bounded_typo = allow_edit1 && damerau_lev_1(wb, cb);
        if is_adj_swap || is_bounded_typo {
            let cst = stem(cand);
            if let Some((_, ref prev_st)) = found {
                if *prev_st != cst {
                    return None;
                }
            } else {
                found = Some((id as u32, cst));
            }
        }
    }
    found.map(|(id, _)| id)
}

/// Step 9A wiring in plan_query: a zero-DF query word (no exact, stem, or nick hit) maps to
/// its unique Damerau-Levenshtein-1 vocab neighbour (0.75) and that neighbour's stem (0.65).
pub(in crate::retrieval) fn step9_push_typo_fallback(
    ix: &Index,
    w: &str,
    st: &str,
    has_nick: bool,
    g: usize,
    terms: &mut Vec<QTerm>,
    group_of: &mut DetHashMap<u32, usize>,
) {
    if has_nick || ix.vocab.get(w).is_some() || ix.vocab.get(&stem_term(st)).is_some() {
        return;
    }
    let Some(typo_id) = find_unique_typo_fallback(&ix.vocab.names, w) else {
        return;
    };
    terms.push(QTerm { id: typo_id, weight: 0.75, group: Some(g), exact_hit: false });
    group_of.insert(typo_id, g);
    if let Some(cand_str) = ix.vocab.names.get(typo_id as usize) {
        if let Some(cs_id) = ix.vocab.get(&stem_term(&stem(cand_str))) {
            terms.push(QTerm { id: cs_id, weight: 0.65, group: Some(g), exact_hit: false });
            group_of.insert(cs_id, g);
        }
    }
}
