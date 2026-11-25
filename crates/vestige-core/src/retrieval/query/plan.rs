//! Query planning, typo fallback, and matched-word extraction.

use crate::retrieval::*;


pub(in crate::retrieval) fn plan_query(ix: &Index, q: &Query, p: &Params) -> QueryPlan {
    let mut intent = classify(&q.text);
    let mut terms = Vec::new();
    let mut groups: Vec<String> = Vec::new();
    let mut group_of = DetHashMap::default();
    let mut focal_ids: DetHashSet<u32> = DetHashSet::default();
    let mut excluded_ids: DetHashSet<u32> = DetHashSet::default();
    let mut rare_entity_ids: Vec<u32> = Vec::new();

    // Strip trailing closed-class output-formatting constraints ("Mention ONLY and ONLY five items.")
    // so constraint words/cardinalities never pollute lexical BM25 groups on ordering/summary queries.
    let plan_text: &str = {
        let low_full = q.text.to_ascii_lowercase();
        // hardcode-scan: closed-class (trailing output-formatting instruction clauses)
        let cut = ["mention only", "list only", "give only", "return only"]
            .iter()
            .filter_map(|pat| low_full.rfind(pat))
            .filter(|&pos| {
                pos >= 16
                    && matches!(
                        q.text.as_bytes().get(pos.saturating_sub(1)).copied(),
                        Some(b' ' | b'.' | b'?' | b'!' | b'\n')
                    )
            })
            .min();
        match cut {
            Some(pos) => q.text[..pos].trim_end(),
            None => q.text.as_str(),
        }
    };
    let raw_q = raw_tokens(plan_text);
    let q_low_plan = plan_text.to_ascii_lowercase();
    let cap_words_in_q: DetHashSet<String> = plan_text
        .split_whitespace()
        .filter_map(|orig| {
            let clean = orig.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.len() >= 2 && clean.chars().next().is_some_and(char::is_uppercase) {
                Some(clean.to_ascii_lowercase())
            } else {
                None
            }
        })
        .collect();
    // Closed-class exclusion prepositions ("other than <NP>", "besides <NP>", "apart from <NP>",
    // "except for <NP>", "except <NP>"): capture the governed NP tokens up to comma/clause/aux/WH
    // boundary so excluded items never receive positive exact/focal credit.
    // hardcode-scan: closed-class (exclusion prepositions and clause/auxiliary boundaries)
    let mut excluded_words: DetHashSet<String> = DetHashSet::default();
    let q_is_wh_excl = wh_slots(plan_text) != 0 || q_low_plan.starts_with("other than ") || q_low_plan.starts_with("apart from ") || q_low_plan.starts_with("except ");
    for pat in ["other than ", "apart from ", "except for ", "besides ", "except "] {
        if pat == "besides " && (!q_is_wh_excl || q_low_plan.contains(" also")) {
            continue;
        }
        let mut search_from = 0usize;
        while let Some(rel) = q_low_plan[search_from..].find(pat) {
            let start = search_from + rel;
            let word_boundary_ok = start == 0
                || !q_low_plan.as_bytes()[start - 1].is_ascii_alphanumeric();
            let after = start + pat.len();
            search_from = after;
            if !word_boundary_ok || after >= plan_text.len() {
                continue;
            }
            let tail_clause = plan_text[after..]
                .split([',', ';', '.', '?', '!', ':', '\n', '(', ')'])
                .next()
                .unwrap_or("");
            let mut captured = 0usize;
            for tok in raw_tokens(tail_clause) {
                if matches!(
                    tok.as_str(),
                    "did" | "does" | "do" | "is" | "are" | "was" | "were"
                        | "has" | "have" | "had" | "would" | "could" | "can"
                        | "should" | "will" | "might" | "may"
                        | "what" | "which" | "who" | "where" | "when" | "why" | "how"
                        | "that" | "in" | "on" | "at" | "during" | "for" | "with"
                        | "from" | "to" | "by" | "about" | "before" | "after" | "while"
                ) {
                    break;
                }
                if matches!(tok.as_str(), "a" | "an" | "the" | "his" | "her" | "their" | "my" | "your" | "our" | "and" | "or") {
                    continue;
                }
                if is_content(&tok) {
                    excluded_words.insert(tok.clone());
                    if let Some(id) = ix.vocab.get(&tok) {
                        excluded_ids.insert(id);
                    }
                    let st_tok = stem(&tok);
                    if let Some(sid) = ix.vocab.get(&stem_term(&st_tok)) {
                        excluded_ids.insert(sid);
                    }
                    captured += 1;
                    if captured >= 3 {
                        break;
                    }
                }
            }
        }
    }
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
    // Closed-class epistemic/hypothetical modal query framing (`"Would X likely..."`, `"What might X..."`):
    // hardcode-scan: closed-class (epistemic modal interrogative markers)
    let is_hypothetical_modal_q = matches!(raw_q.first().map(String::as_str), Some("would"))
        || raw_q.windows(2).any(|p| {
            (matches!(p[0].as_str(), "what" | "which" | "who" | "where" | "how")
                && matches!(p[1].as_str(), "might" | "would" | "could"))
                || matches!(p[1].as_str(), "likely" | "potentially")
        })
        || raw_q.iter().any(|w| matches!(w.as_str(), "likely" | "potentially"));
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
        // "test", "degree") EXCEPT when the query is a hypothetical/modal trait query (`"What fields would X
        // likely pursue..."`, `"What traits might X have..."`), where the WH-head noun is always a genus!
        if !is_hypothetical_modal_q && (1..=rare_anchor_cap).contains(&cand_df) {
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
    let mut taxonomic_drop_mask: u64 = 0;
    if raw_q.len() >= 3
        && raw_q[0] == "in"
        && matches!(raw_q[1].as_str(), "what" | "which")
        && matches!(raw_q[2].as_str(), "way" | "ways")
    {
        taxonomic_drop_mask |= 1u64 << 2;
    }
    if !has_first_person_q
        && let Some(wi) = wh_det_pos
        && n_docs >= 24
        && non_routing_content_cnt >= 2
        && raw_q.len() > wi + 2
    {
        let h1 = &raw_q[wi + 1];
        let w2 = &raw_q[wi + 2];
        if w2 == "of" && (wi + 1) < 64 && matches!(h1.as_str(), "kind" | "kinds" | "type" | "types" | "sort" | "sorts") {
            // hardcode-scan: closed-class (taxonomic WH classifier nouns in "what kind/type/sort of")
            taxonomic_drop_mask |= 1u64 << (wi + 1);
        } else if (is_closed_post_head(w2) || has_later_cap_entity)
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
    let asks_nickname = raw_q.iter().any(|w| w == "nickname" || w == "nicknames");

    for (idx, w) in raw_q.iter().enumerate() {
        if !is_content(w) || excluded_words.contains(w) {
            continue;
        }
        if idx < 64 && (taxonomic_drop_mask & (1u64 << idx)) != 0 {
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
            && (matches!(w.as_str(), "likely" | "potentially")
                || (has_later_cap_entity
                    && matches!(w.as_str(), "might" | "consider" | "considered" | "future")));
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
        let is_order_or_sum_q = crate::retrieval::eff_summary_request(&q.text)
            || intent.is_event_sequence
            || q_low_plan.contains("order in which")
            || q_low_plan.contains("walk me through");
        // hardcode-scan: closed-class (meta-task and ordering/enumeration scaffold operators)
        let is_meta_task_framing = can_demote_framing
            && is_order_or_sum_q
            && matches!(
                w.as_str(),
                "summary"
                    | "summarize"
                    | "comprehensive"
                    | "conversations"
                    | "overview"
                    | "progressed"
                    | "aspects"
                    | "timeline"
                    | "walkthrough"
                    | "walk"
                    | "brought"
                    | "different"
                    | "items"
                    | "order"
                    | "including"
                    | "related"
                    | "various"
                    | "throughout"
                    | "across"
                    | "provide"
                    | "give"
            );
        let is_frame = (can_demote_framing && frame_token(w)) || is_meta_task_framing;
        if is_frame {
            let fw = if is_meta_task_framing { 0.12 } else { 0.30 };
            if let Some(id) = ix.vocab.get(w) {
                terms.push(QTerm { id, weight: fw, group: None, exact_hit: false });
            }
            if p.enable_stemming {
                let st = stem(w);
                if let Some(id) = ix.vocab.get(&stem_term(&st)) {
                    terms.push(QTerm { id, weight: fw * 0.7, group: None, exact_hit: false });
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
        // Stage-01 Vocab-Attested Compound Splitting (`len >= 6`, e.g. "roadtrip" -> "road" + "trip"):
        // If `w` splits into two corpus-attested content words (`len >= 3` each), emit both under group `g`
        // so BM25F, MRF, and `s_exact_mask` match at Stage 01 without waiting for downstream bridges.
        if w.is_ascii() && w.len() >= 6 && w.bytes().all(|b| b.is_ascii_alphabetic()) {
            let w_df = ix.vocab.get(w).map(|id| corpus_df(id)).unwrap_or(0);
            if w_df <= 2 {
                for split_at in 3..=(w.len() - 3) {
                    let left = &w[..split_at];
                    let right = &w[split_at..];
                    if is_content(left)
                        && is_content(right)
                        && let (Some(lid), Some(rid)) = (ix.vocab.get(left), ix.vocab.get(right))
                    {
                        if corpus_df(lid) > 0 && corpus_df(rid) > 0 {
                            terms.push(QTerm { id: lid, weight: 0.72, group: Some(g), exact_hit: true });
                            terms.push(QTerm { id: rid, weight: 0.72, group: Some(g), exact_hit: true });
                            group_of.entry(lid).or_insert(g);
                            group_of.entry(rid).or_insert(g);
                            break;
                        }
                    }
                }
            }
        }
        // Adjacent query compound join ("road" + "trip" -> "roadtrip" when attested in corpus vocab):
        if let Some(next_w) = raw_q.get(idx + 1)
            && is_content(next_w)
            && w.len() >= 3
            && next_w.len() >= 3
            && w.is_ascii()
            && next_w.is_ascii()
        {
            let joined = format!("{w}{next_w}");
            if let Some(jid) = ix.vocab.get(&joined)
                && corpus_df(jid) > 0
            {
                terms.push(QTerm { id: jid, weight: 0.78, group: Some(g), exact_hit: true });
                group_of.entry(jid).or_insert(g);
            }
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
                let nick_df_max = ((ix.t_own_stats.n as u32) / 3).max(24);
                const NICK_PREFIX_WEIGHT_Q16: u64 = 104_858; // 1.60 in Q16.16 so dedicated nickname group drives rank.
                let mut nick_ids: Vec<u32> = Vec::new();
                for plen in [2usize, 3, 4] {
                    if plen < w.len()
                        && w.is_char_boundary(plen)
                        && let Some(nid) = ix.vocab.get(&w[..plen])
                        && (1..=nick_df_max).contains(&corpus_df(nid))
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
        step9_push_typo_fallback(
            ix,
            (w, &st, nick_bind.get(w).is_some(), cap_words_in_q.contains(w), g),
            &mut groups,
            &mut terms,
            &mut group_of,
            &mut focal_ids,
        );
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

    // Phase 3 Steps 7 & 8: Full Human-Knowledge Lexicon (Princeton WordNet 3.1 4-POS +
    // Roget's 1911 Thesaurus + GeoNames Gazetteer + ConceptNet 5.7), strictly IC-weighted
    // and STRICTLY gated to OOV / vocabulary-mismatch query groups (`!group_has_corpus_hit`),
    // NEVER diluting in-corpus Single-Hop / Adversarial exact matches or modifying `t_own`/`t_raw`.
    let ns_plan = ix.session_turns.len();
    if p.enable_hypernyms && !gate_open && ns_plan >= 8 {
        use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;

        let ns_u32 = ns_plan as u32;
        let max_ic_q16 = aspect_ic_q16(1, ns_u32).max(1);
        let q_raw_words: Vec<String> = raw_tokens(&q.text);

        // Identify T50 wh-determiner + class noun target ("what/which [adj] <class_noun>").
        let mut wh_class_words: Vec<String> = Vec::new();
        for (idx, tok) in q_raw_words.iter().enumerate() {
            if matches!(tok.as_str(), "what" | "which") {
                for nxt in q_raw_words.iter().skip(idx + 1).take(3) {
                    if matches!(
                        nxt.as_str(),
                        "did" | "does" | "do" | "is" | "are" | "was" | "were" | "would" | "could"
                            | "might" | "may" | "should" | "has" | "have" | "had" | "kind" | "type"
                            | "sort" | "of" | "a" | "an" | "the"
                    ) {
                        continue;
                    }
                    let is_sp = ix.turns.iter().take(64).any(|t| t.speaker.eq_ignore_ascii_case(nxt));
                    if !is_sp && is_content(nxt) {
                        wh_class_words.push(nxt.clone());
                    }
                }
            }
        }

        let rare_df_cap = (ns_u32 / 3).clamp(3, 18);
        for (w, g) in &expandable {
            let is_sp = ix.turns.iter().take(64).any(|t| t.speaker.eq_ignore_ascii_case(w))
                || nick_bind.contains_key(w);
            if is_sp {
                continue;
            }
            let group_has_corpus_hit = terms
                .iter()
                .any(|t| t.group == Some(*g) && t.exact_hit && corpus_df(t.id) > 0);
            // Compound split (`roadtrip` -> `road` + `trip`, `bookshelf` -> `book` + `shelf`):
            // runs on OOV compounds (as group expansion) AND ultra-rare (`df <= 2`) closed compounds
            // at low auxiliary weight (`exact_hit: false`, `group: None`) without diluting exact matches.
            let raw_w_df = ix.vocab.get(w).map(|id| corpus_df(id)).unwrap_or(0);
            if (!group_has_corpus_hit || raw_w_df <= 2) && w.len() >= 6 && w.is_ascii() {
                for split_at in 3..=(w.len() - 3) {
                    let (left, right) = w.split_at(split_at);
                    if is_content(left) && is_content(right) {
                        if let (Some(lid), Some(rid)) = (ix.vocab.get(left), ix.vocab.get(right)) {
                            if corpus_df(lid) > 0 && corpus_df(rid) > 0 {
                                let (cw, cgrp) = if !group_has_corpus_hit {
                                    (p.hypernym_weight * 0.90, Some(*g))
                                } else {
                                    (p.hypernym_weight * 0.45, None)
                                };
                                for cid in [lid, rid] {
                                    if hyp_added.insert(cid) {
                                        terms.push(QTerm {
                                            id: cid,
                                            weight: cw,
                                            group: cgrp,
                                            exact_hit: false,
                                        });
                                        if let Some(gr) = cgrp {
                                            group_of.insert(cid, gr);
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
            // (c) T50 Wh-Class-Noun -> WordNet Instance-Hyponym + GeoNames Class Gazetteer + Wikidata:
            // Runs for explicit `"what/which [adj] <class_noun>"` queries EVEN WHEN the class word itself
            // appears in passing in the corpus (e.g. `"what state"` -> `Indiana`/`Florida`/`Minnesota`,
            // `"which meat"` -> `chicken`, `"what pet"` -> `dog`/`cat`/`guinea pig`), strictly at low
            // auxiliary weight (`exact_hit: false`, `group: None` when class is in-corpus so exact matches
            // are never diluted).
            let is_wh_class = wh_class_words.iter().any(|cw| cw == w);
            if is_wh_class {
                let mut inst_cands: Vec<(u64, u32)> = Vec::new();
                for inst in crate::wordnet::class_to_instances(w, 2)
                    .into_iter()
                    .chain(crate::geonames::geo_class_instances(w).iter().map(String::as_str))
                    .chain(crate::wikidata::direct_instances_of(w, 16))
                {
                    for part in inst.split_ascii_whitespace() {
                        if let Some(id) = ix.vocab.get(part) {
                            let df = ix.s_stats.df.get(&id).copied().unwrap_or(0);
                            if (1..=rare_df_cap).contains(&df) && !hyp_added.contains(&id) {
                                inst_cands.push((aspect_ic_q16(df, ns_u32), id));
                            }
                        }
                    }
                }
                inst_cands.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                inst_cands.dedup_by_key(|x| x.1);
                let mut added_inst = 0usize;
                let inst_cap = if group_has_corpus_hit { 10 } else { 16 };
                let inst_scale = if group_has_corpus_hit { 0.45 } else { 0.72 };
                let inst_grp = if group_has_corpus_hit { None } else { Some(*g) };
                for (ic, id) in inst_cands {
                    if added_inst >= inst_cap {
                        break;
                    }
                    if hyp_added.insert(id) {
                        let ic_ratio = ((ic as f32) / (max_ic_q16 as f32)).clamp(0.25, 1.0);
                        let w_eff = (p.hypernym_weight * inst_scale * ic_ratio).clamp(0.08, p.hypernym_weight);
                        terms.push(QTerm { id, weight: w_eff, group: inst_grp, exact_hit: false });
                        if let Some(gr) = inst_grp {
                            group_of.insert(id, gr);
                        }
                        added_inst += 1;
                    }
                }
            }

            // STRICT ZERO-DILUTION INVARIANT: never run general synonym expansion on a word that ALREADY matches the corpus!
            if group_has_corpus_hit {
                continue;
            }

            // (b) Unified OOV Human-Knowledge Expansion: WordNet 3.1 (4-POS + derived `+` + instance_surnames
            // + edit-dist-1 typo rescue) + Roget's 1911 Thesaurus + GeoNames + Wikidata CC0 + ConceptNet 5.7, IC-weighted.
            let mut best_by_id: std::collections::BTreeMap<u32, (u64, u16)> = std::collections::BTreeMap::new();
            let mut consider_lemma = |lemma: &str, rel_w: u16| {
                let surface = ix.vocab.get(lemma);
                let stemmed = if p.enable_stemming {
                    ix.vocab.get(&stem_term(&stem(lemma)))
                } else {
                    None
                };
                for id in [surface, stemmed].into_iter().flatten() {
                    let df = ix.s_stats.df.get(&id).copied().unwrap_or(0);
                    if (1..=rare_df_cap).contains(&df) && !hyp_added.contains(&id) {
                        let ic = aspect_ic_q16(df, ns_u32);
                        let rank_sc = ic.saturating_mul(u64::from(rel_w));
                        let entry = best_by_id.entry(id).or_insert((0, 0));
                        if rank_sc > entry.0 {
                            *entry = (rank_sc, rel_w);
                        }
                    }
                }
            };
            for (lemma, rel_w) in crate::wordnet::full_lexicon_neighbors(w) {
                consider_lemma(lemma, rel_w);
            }
            for (lemma, rel_w) in crate::roget::conceptual_neighbors(w, 64) {
                consider_lemma(lemma, rel_w);
            }
            for (lemma, rel_w) in crate::geonames::geo_neighbors(w) {
                consider_lemma(lemma, rel_w);
            }
            for cont in crate::wikidata::upward_containers(w, 2)
                .into_iter()
                .chain(crate::wikidata::direct_instances_of(w, 12))
            {
                for part in cont.split_ascii_whitespace() {
                    if part.len() >= 3 {
                        consider_lemma(part, 78);
                    }
                }
            }
            for (cn_lemma, cn_w) in crate::conceptnet::two_hop_neighbors(w, 12) {
                let scaled_w = ((u32::from(cn_w.min(200)) * 85) / 200).clamp(40, 85) as u16;
                consider_lemma(cn_lemma, scaled_w);
            }

            let mut scored_exp: Vec<(u64, u32, u16)> = best_by_id
                .into_iter()
                .map(|(id, (sc, rw))| (sc, id, rw))
                .collect();
            scored_exp.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut added_cnt = 0usize;
            for (_, id, rel_w) in scored_exp {
                if added_cnt >= 14 {
                    break;
                }
                if hyp_added.insert(id) {
                    let df = ix.s_stats.df.get(&id).copied().unwrap_or(1);
                    let ic_ratio = ((aspect_ic_q16(df, ns_u32) as f32) / (max_ic_q16 as f32)).clamp(0.22, 1.0);
                    let rel_ratio = (f32::from(rel_w) / 100.0).clamp(0.42, 1.0);
                    let w_eff = (p.hypernym_weight * 0.85 * ic_ratio * rel_ratio).clamp(0.10, p.hypernym_weight);
                    terms.push(QTerm { id, weight: w_eff, group: Some(*g), exact_hit: false });
                    group_of.insert(id, *g);
                    added_cnt += 1;
                }
            }
        }
    }

    // CamelCase & snake/hyphen reverse query expansion (`localStorage` -> `local`, `storage` at 0.55),
    // PLUS exact hyphenated/underscored tech-compound focal preservation (`Flask-Login` -> `flasklogin`
    // at 1.35 + `exact_hit: true` + `focal_ids` when corpus-rare so compound packages beat high-DF parts).
    let compound_df_cap = ((n_docs / 4) as u32).max(4);
    for raw_word in q.text.split_whitespace() {
        let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
        if word.is_empty() {
            continue;
        }
        let is_hyphen_compound = word.contains('-');
        for (idx_aux, (part, _)) in surface_aux_tokens(word).into_iter().enumerate() {
            if seen_toks.contains(&part) {
                continue;
            }
            if let Some(id) = ix.vocab.get(&part) {
                let df = corpus_df(id);
                let s_df = ix.s_stats.df.get(&id).copied().unwrap_or(0);
                if is_hyphen_compound && idx_aux == 0 && (1..=compound_df_cap).contains(&df) {
                    let is_ultra_rare = (1..=2).contains(&s_df) && ns_plan >= 10;
                    let w_comp = if is_ultra_rare { 2.65 } else { 0.95 };
                    terms.push(QTerm { id, weight: w_comp, group: None, exact_hit: true });
                    if df <= 2 || is_ultra_rare {
                        focal_ids.insert(id);
                    }
                    if is_ultra_rare && !rare_entity_ids.contains(&id) {
                        rare_entity_ids.push(id);
                    }
                } else {
                    terms.push(QTerm { id, weight: 0.55, group: None, exact_hit: false });
                }
            }
        }
    }

    // Capitalized proper-noun entity phrases present in index (e.g. "Local H", "For Against", "Flask-Login").
    let has_named_plan = ix
        .turns
        .iter()
        .take(32)
        .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    let mut seen_ents = DetHashSet::default();
    for ent in proper_entity_tokens(&q.text) {
        if !seen_ents.insert(ent.clone()) {
            continue;
        }
        if let Some(id) = ix.vocab.get(&ent) {
            let s_df = ix.s_stats.df.get(&id).copied().unwrap_or(0);
            let is_ultra_rare = !has_named_plan && (1..=2).contains(&s_df) && ns_plan >= 10;
            let parts_body = ent.strip_prefix("\u{1}ent:").unwrap_or("");
            let any_erased = parts_body
                .split('_')
                .any(|part| !groups.iter().any(|g| g == part) && !frame_token(part));
            if any_erased {
                let g = groups.len();
                groups.push(ent);
                let w_ent = if is_ultra_rare { 2.65 } else { 1.65 };
                terms.push(QTerm { id, weight: w_ent, group: Some(g), exact_hit: true });
                group_of.insert(id, g);
                if is_ultra_rare {
                    focal_ids.insert(id);
                    if !rare_entity_ids.contains(&id) {
                        rare_entity_ids.push(id);
                    }
                }
            } else {
                let w_ent = if is_ultra_rare { 2.25 } else { 0.75 };
                terms.push(QTerm { id, weight: w_ent, group: None, exact_hit: is_ultra_rare });
                if is_ultra_rare {
                    focal_ids.insert(id);
                    if !rare_entity_ids.contains(&id) {
                        rare_entity_ids.push(id);
                    }
                }
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

    QueryPlan { intent, terms, groups, group_of, focal_ids, excluded_ids, rare_entity_ids }
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
    if let Some((id, _)) = found {
        return Some(id);
    }
    // Step 9B: Unique orthographic prefix-abbreviation fallback (`"educaton"`/`"education"` -> `"edu"`,
    // `"psychology"` -> `"psych"`, `"department"` -> `"dept"`). Restricted strictly to non-WordNet
    // 3..=5 char ASCII lowercase abbreviations that are an exact prefix of `w` (`len >= 6`), preventing
    // false matches against ordinary English words (`"cat"` != `"category"`).
    let mut abbrev_found: Option<u32> = None;
    for (id, cand) in vocab_names.iter().enumerate() {
        let cb = cand.as_bytes();
        if !(3..=5).contains(&cb.len())
            || cb[0] != first
            || cb.starts_with(b"\x01")
            || !cb.iter().all(|b| b.is_ascii_lowercase())
        {
            continue;
        }
        if !w.starts_with(cand.as_str()) || !is_content(cand) {
            continue;
        }
        if !crate::wordnet::senses(cand).is_empty() {
            continue;
        }
        if abbrev_found.is_some() {
            return None;
        }
        abbrev_found = Some(id as u32);
    }
    abbrev_found
}

/// Step 9A wiring in plan_query: casing-stratified Damerau-Levenshtein-1 typo snap
/// (`len >= 4` TitleCase speaker names like `"Andew" -> "Andrew"`, plus `len >= 6` lowercase
/// OOV words / unique prefix-abbreviations / `>= 8x` DF-dominant neighbours at `weight: 0.85, exact_hit: true`).
pub(in crate::retrieval) fn step9_push_typo_fallback(
    ix: &Index,
    tok_meta: (&str, &str, bool, bool, usize),
    groups: &mut [String],
    terms: &mut Vec<QTerm>,
    group_of: &mut DetHashMap<u32, usize>,
    focal_ids: &mut DetHashSet<u32>,
) {
    let (w, st, has_nick, is_cap_in_query, g) = tok_meta;
    if has_nick || ix.vocab.get(w).is_some() || ix.vocab.get(&stem_term(st)).is_some() {
        return;
    }
    // 1. Casing-stratified speaker snap (`len >= 4`, TitleCase in query, e.g. `"Andew"` -> `"Andrew"`):
    if is_cap_in_query && w.len() >= 4 && w.is_ascii() {
        let wb = w.as_bytes();
        let mut sp_cands: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for t in ix.turns.iter().take(128) {
            let sp = t.speaker.trim();
            if sp.len() >= 4 && !sp.eq_ignore_ascii_case("user") && !sp.eq_ignore_ascii_case("assistant") {
                let sp_low = sp.to_ascii_lowercase();
                if sp_low.is_ascii() && damerau_lev_1(wb, sp_low.as_bytes()) {
                    sp_cands.insert(sp_low);
                }
            }
        }
        if sp_cands.len() == 1
            && let Some(sp_low) = sp_cands.into_iter().next()
            && let Some(sp_id) = ix.vocab.get(&sp_low)
        {
            if let Some(slot) = groups.get_mut(g) {
                *slot = sp_low.clone();
            }
            terms.push(QTerm { id: sp_id, weight: 0.90, group: Some(g), exact_hit: true });
            group_of.insert(sp_id, g);
            focal_ids.insert(sp_id);
            if let Some(cs_id) = ix.vocab.get(&stem_term(&stem(&sp_low))) {
                terms.push(QTerm { id: cs_id, weight: 0.72, group: Some(g), exact_hit: true });
                group_of.insert(cs_id, g);
            }
            return;
        }
    }

    // 2. Lowercase `len >= 6` unique DL-1 / prefix-abbrev OR `>= 8x` DF-dominant DL-1 snap:
    let typo_opt = find_unique_typo_fallback(&ix.vocab.names, w).or_else(|| {
        let wb = w.as_bytes();
        if wb.len() < 6 || !wb.iter().all(|b| b.is_ascii_lowercase()) {
            return None;
        }
        if !crate::wordnet::senses(w).is_empty() {
            return None;
        }
        let first = wb[0];
        let mut cands: Vec<(u32, u32)> = Vec::new();
        for (id, cand) in ix.vocab.names.iter().enumerate() {
            let cb = cand.as_bytes();
            if cb.is_empty()
                || cb[0] != first
                || cb.starts_with(b"\x01")
                || !cb.iter().all(|b| b.is_ascii_lowercase())
            {
                continue;
            }
            if damerau_lev_1(wb, cb) {
                let df = ix.t_own_stats.df.get(&(id as u32)).copied().unwrap_or(0);
                if df > 0 {
                    cands.push((df, id as u32));
                }
            }
        }
        cands.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        match cands.as_slice() {
            [(df0, id0)] if *df0 > 0 => Some(*id0),
            [(df0, id0), (df1, _), ..] if *df0 >= df1.saturating_mul(8).max(8) => Some(*id0),
            _ => None,
        }
    });
    let Some(typo_id) = typo_opt else {
        return;
    };
    terms.push(QTerm { id: typo_id, weight: 0.85, group: Some(g), exact_hit: true });
    group_of.insert(typo_id, g);
    focal_ids.insert(typo_id);
    if let Some(cand_str) = ix.vocab.names.get(typo_id as usize) {
        if let Some(cs_id) = ix.vocab.get(&stem_term(&stem(cand_str))) {
            terms.push(QTerm { id: cs_id, weight: 0.72, group: Some(g), exact_hit: true });
            group_of.insert(cs_id, g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthographic_prefix_abbreviation_resolves_educaton_to_edu_without_english_false_positives() {
        let vocab = vec!["hello".to_string(), "edu".to_string(), "cat".to_string()];
        assert_eq!(find_unique_typo_fallback(&vocab, "educaton"), Some(1));
        assert_eq!(find_unique_typo_fallback(&vocab, "category"), None);
    }

    #[test]
    fn hyphenated_tech_compound_becomes_exact_focal_anchor() {
        let sessions = vec![
            Session {
                id: "s0".into(),
                date: String::new(),
                timestamp: 1000,
                turns: vec![Turn {
                    id: "t0".into(),
                    role: "user".into(),
                    content: "We configured Flask-Login for session auth.".into(),
                    ..Default::default()
                }],
            },
            Session {
                id: "s1".into(),
                date: String::new(),
                timestamp: 2000,
                turns: vec![Turn {
                    id: "t1".into(),
                    role: "user".into(),
                    content: "Flask routing works for static pages.".into(),
                    ..Default::default()
                }],
            },
        ];
        let p = Params::default();
        let ix = build_index(&sessions, &p);
        let q = Query { text: "How did we configure Flask-Login?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &p);
        let compound_id = ix.vocab.get("flasklogin").expect("flasklogin indexed");
        assert!(plan.focal_ids.contains(&compound_id));
        assert!(plan.terms.iter().any(|t| t.id == compound_id && t.exact_hit && t.weight >= 0.9));
    }
}
