//! KITE Upward-Cone Typed Facet Index + Formal Concept Analysis (FCA) Closed-Intent Meet.
//!
//! Inspired by KITE (`2610.41521`, Oct 2026) and Retrieval-Grounded FCA (`2607.01773`, Jul 2026).
//! Instead of expanding an abstract query word downward into hundreds of noisy hyponyms,
//! indexes every declarative turn's content words UPWARD 1..=2 hops through the pinned
//! SHA-256 WordNet 3.1 hypernym DAG and Wikidata CC0 container graph into compact typed
//! synset/concept postings. At query time, abstract Open-Domain/Multi-Hop facets intersect
//! concrete turns in O(1) with participant-facet gating and FCA specificity weighting.

use crate::retrieval::{DetHashMap, DetHashSet, Index, QueryPlan, TurnIx, is_content, raw_tokens};
use std::collections::BTreeSet;

const Q16_ONE: u64 = 65_536;

/// Upward-cone typed facet postings compiled at ingest time (zero vectors, 100% deterministic).
#[derive(Default)]
pub(in crate::retrieval) struct KiteTypedIndex {
    /// WordNet 3.1 specific synset ID -> `(turn_idx, specificity_w)` postings.
    pub(in crate::retrieval) synset_postings: DetHashMap<u32, Vec<(usize, u16)>>,
    /// Upward WordNet/Wikidata hypernym/container token -> `(turn_idx, specificity_w)` postings.
    pub(in crate::retrieval) concept_postings: DetHashMap<String, Vec<(usize, u16)>>,
}

/// Build the KITE upward-cone typed facet index over declarative conversation turns.
pub(in crate::retrieval) fn build_kite_typed_index(turns: &[TurnIx]) -> KiteTypedIndex {
    let mut synset_postings: DetHashMap<u32, Vec<(usize, u16)>> = DetHashMap::default();
    let mut concept_postings: DetHashMap<String, Vec<(usize, u16)>> = DetHashMap::default();

    // Cap per-conversation indexing budget for very large multi-thousand-turn corpora while
    // keeping full coverage on dialogue corpora (LoCoMo ~400-700 turns, BEAM ~300-600 turns/conv).
    let max_words_per_turn = if turns.len() > 8_000 {
        22usize
    } else if turns.len() > 1_500 {
        28usize
    } else {
        36usize
    };

    type KiteWordEntry = (Vec<(u32, u16)>, Vec<(String, u16)>);
    thread_local! {
        static KITE_WORD_CACHE: std::cell::RefCell<DetHashMap<String, KiteWordEntry>> =
            std::cell::RefCell::new(DetHashMap::default());
    }

    KITE_WORD_CACHE.with(|cell| {
        let mut wcache = cell.borrow_mut();
        if wcache.len() >= 49_152 {
            wcache.clear();
        }
        for (ti, tr) in turns.iter().enumerate() {
            if tr.is_backchannel || tr.is_question || tr.is_standing_policy {
                continue;
            }
            // Only index substantive declarative prose (`field == 0` user or named speaker turns).
            if tr.field != 0 {
                continue;
            }
            let mut seen_words: DetHashSet<String> = DetHashSet::default();
            let mut seen_syns: DetHashSet<u32> = DetHashSet::default();
            let mut seen_concepts: DetHashSet<String> = DetHashSet::default();

            let toks = raw_tokens(&tr.body_for_headers);
            let mut indexed = 0usize;
            for tok in toks {
                if tok.len() < 4 || !is_content(&tok) || !seen_words.insert(tok.clone()) {
                    continue;
                }
                indexed += 1;
                if indexed > max_words_per_turn {
                    break;
                }
                if !wcache.contains_key(&tok) {
                    let mut w_syns: Vec<(u32, u16)> = Vec::new();
                    let mut w_cons: Vec<(String, u16)> = Vec::new();
                    let mut w_seen_s: DetHashSet<u32> = DetHashSet::default();
                    let mut w_seen_c: DetHashSet<String> = DetHashSet::default();
                    // 1. Direct WordNet senses (`weight = 96`) + specific upward ancestor synsets (`desc <= 160`)
                    let senses = crate::wordnet::senses(&tok);
                    if !senses.is_empty() && senses.len() <= 8 {
                        for &syn in senses.iter().take(3) {
                            if w_seen_s.insert(syn) {
                                w_syns.push((syn, 96));
                            }
                        }
                        for (anc_syn, desc) in crate::wordnet::specific_ancestor_synsets(&tok, 2, 160)
                            .into_iter()
                            .take(6)
                        {
                            if w_seen_s.insert(anc_syn) {
                                let w: u16 = if desc <= 20 {
                                    88
                                } else if desc <= 65 {
                                    76
                                } else {
                                    62
                                };
                                w_syns.push((anc_syn, w));
                            }
                        }
                    }
                    // 2. Upward WordNet hypernym lemmas (1..=2 hops)
                    for hyp in crate::wordnet::upward_hypernyms(&tok, 2).into_iter().take(6) {
                        if hyp.len() >= 4 && hyp != tok && w_seen_c.insert(hyp.to_string()) {
                            w_cons.push((hyp.to_string(), 74));
                        }
                    }
                    // 3. Wikidata CC0 upward container tokens (e.g., `bach -> composer, music`, `vivaldi -> composer`)
                    for cont in crate::wikidata::upward_containers(&tok, 2).into_iter().take(4) {
                        for part in cont.split_ascii_whitespace() {
                            if part.len() >= 4 && part != tok && w_seen_c.insert(part.to_string()) {
                                w_cons.push((part.to_string(), 80));
                            }
                        }
                    }
                    wcache.insert(tok.clone(), (w_syns, w_cons));
                }
                if let Some((w_syns, w_cons)) = wcache.get(&tok) {
                    for &(syn, w) in w_syns {
                        if seen_syns.insert(syn) {
                            synset_postings.entry(syn).or_default().push((ti, w));
                        }
                    }
                    for (con, w) in w_cons {
                        if seen_concepts.insert(con.clone()) {
                            concept_postings.entry(con.clone()).or_default().push((ti, *w));
                        }
                    }
                }
            }
        }
    });

    KiteTypedIndex {
        synset_postings,
        concept_postings,
    }
}

/// Score turns and sessions via KITE upward-cone facet intersection + FCA closed-intent specificity.
/// Returns `(session_lifts_q16, turn_lifts_q16)` bounded by `(sess_cap_q16, turn_cap_q16)`.
pub(in crate::retrieval) fn score_kite_typed_facet_meet_q16(
    ix: &Index,
    plan: &QueryPlan,
    q_text: &str,
    queried_speakers: &BTreeSet<String>,
    sess_cap_q16: u64,
    turn_cap_q16: u64,
) -> (Vec<u64>, Vec<u64>) {
    let ns = ix.session_turns.len();
    let nt = ix.turns.len();
    let sess_out = vec![0u64; ns];
    let mut turn_out = vec![0u64; nt];
    if ns == 0 || nt == 0 || (sess_cap_q16 == 0 && turn_cap_q16 == 0) {
        return (sess_out, turn_out);
    }

    let _ = plan;
    let q_toks: Vec<String> = raw_tokens(q_text)
        .into_iter()
        .filter(|w| w.len() >= 4 && is_content(w) && !queried_speakers.contains(w))
        .collect();
    if q_toks.is_empty() {
        return (sess_out, turn_out);
    }

    // Track per-turn raw FCA specificity score and distinct query facets matched.
    let mut turn_raw: DetHashMap<usize, (u64, u32)> = DetHashMap::default();
    let nt_u32 = (nt as u32).max(4);
    let max_df = (nt / 5).max(12);
    let kite = ix.kite_index();

    let mut add_hit = |ti: usize, w_u16: u16, df: usize, facet_bit: u32| {
        if df == 0 || df > max_df {
            return;
        }
        let tr = &ix.turns[ti];
        if !queried_speakers.is_empty()
            && !queried_speakers.contains(&tr.speaker.to_ascii_lowercase())
        {
            return;
        }
        // FCA closed-intent specificity: rarer synset/concept across the conversation = sharper meet.
        let ratio = (nt_u32 as u64) / (df as u64).max(1);
        let log_ic = (64 - ratio.leading_zeros()) as u64; // floor(log2(nt/df)) + 1
        let contrib = ((u64::from(w_u16) * Q16_ONE) / 100).saturating_mul(log_ic.max(1));
        let e = turn_raw.entry(ti).or_insert((0, 0));
        e.0 = e.0.saturating_add(contrib);
        e.1 |= facet_bit;
    };

    for (fi, qw) in q_toks.iter().take(16).enumerate() {
        let fbit = 1u32 << fi;
        // 1. Direct query synset lookup in upward turn cone (abstract query hits concrete hyponym turn!)
        let q_senses = crate::wordnet::senses(qw);
        if !q_senses.is_empty() && q_senses.len() <= 8 {
            for &syn in q_senses.iter().take(3) {
                if let Some(post) = kite.synset_postings.get(&syn) {
                    let df = post.len();
                    for &(ti, w) in post {
                        add_hit(ti, w, df, fbit);
                    }
                }
            }
            // 2. Shared specific ancestor synset meet (`Vivaldi` <-> `Bach/Mozart` via `composer` synset!)
            for (anc_syn, desc) in crate::wordnet::specific_ancestor_synsets(qw, 2, 120)
                .into_iter()
                .take(4)
            {
                if let Some(post) = kite.synset_postings.get(&anc_syn) {
                    let df = post.len();
                    let scale: u16 = if desc <= 25 { 82 } else { 64 };
                    for &(ti, w) in post {
                        let eff_w = ((u32::from(w) * u32::from(scale)) / 100) as u16;
                        add_hit(ti, eff_w, df, fbit);
                    }
                }
            }
        }
        // 3. Direct query word against upward hypernym/Wikidata concept postings (`identity`, `political`, `composer`)
        if let Some(post) = kite.concept_postings.get(qw.as_str()) {
            let df = post.len();
            for &(ti, w) in post {
                add_hit(ti, w, df, fbit);
            }
        }
        // 4. Controlled 1-hop relaxation: query upward Wikidata containers (`vivaldi -> composer, music`)
        for cont in crate::wikidata::upward_containers(qw, 1).into_iter().take(3) {
            for part in cont.split_ascii_whitespace() {
                if part.len() >= 4
                    && part != qw
                    && let Some(post) = kite.concept_postings.get(part)
                {
                    let df = post.len();
                    for &(ti, w) in post {
                        add_hit(ti, (w * 3) / 4, df, fbit);
                    }
                }
            }
        }
        // 5. Query gloss tokens -> concept postings (catches `personality -> trait/character`)
        for gt in crate::wordnet::gloss_tokens_of(qw).into_iter().take(6) {
            if gt.len() >= 4
                && gt != qw
                && crate::wordnet::senses(gt).len() <= 4
                && let Some(post) = kite.concept_postings.get(gt)
            {
                let df = post.len();
                for &(ti, w) in post {
                    add_hit(ti, (w * 2) / 3, df, fbit);
                }
            }
        }
    }

    if turn_raw.is_empty() {
        return (sess_out, turn_out);
    }

    // Apply multi-facet conjunction bonus (FCA Galois closed-intent intersection) + habitual boost.
    let mut sess_raw = vec![0u64; ns];
    let mut max_turn = 0u64;
    let mut max_sess = 0u64;
    for (ti, (raw_sc, fmask)) in &turn_raw {
        let facets = fmask.count_ones().max(1) as u64;
        // Multi-facet intersection multiplier: 1 facet = 1.0x, 2 facets = 1.45x, 3+ = 1.85x
        let mult = match facets {
            1 => 100,
            2 => 145,
            _ => 185,
        };
        let mut sc = raw_sc.saturating_mul(mult) / 100;
        if ix.turns[*ti].is_habitual {
            sc = sc.saturating_add(sc / 4);
        }
        turn_out[*ti] = sc;
        if sc > max_turn {
            max_turn = sc;
        }
        let si = ix.turns[*ti].session;
        if si < ns {
            // Session score = max turn + 1/4 second-order evidence accumulation
            if sc > sess_raw[si] {
                sess_raw[si] = sc.saturating_add(sess_raw[si] >> 2);
            } else {
                sess_raw[si] = sess_raw[si].saturating_add(sc >> 2);
            }
            if sess_raw[si] > max_sess {
                max_sess = sess_raw[si];
            }
        }
    }

    let turn_out = if max_turn > 0 && turn_cap_q16 > 0 {
        crate::retrieval::fusion::pit_commensuration_q16::fuse_channels_pit_nqc_q16(
            turn_out.len(),
            &[(&turn_out, crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE)],
            10,
            turn_cap_q16,
        )
    } else {
        vec![0u64; turn_out.len()]
    };
    let sess_out = if max_sess > 0 && sess_cap_q16 > 0 {
        crate::retrieval::fusion::pit_commensuration_q16::fuse_channels_pit_nqc_q16(
            ns,
            &[(&sess_raw, crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE)],
            10,
            sess_cap_q16,
        )
    } else {
        sess_out
    };
    (sess_out, turn_out)
}
