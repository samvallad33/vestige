//! Reverse-Dictionary, Local Push-PPR, Harmonic Laplacian Voltage, Intrinsic-IC ESA,
//! 4-Hop Upward/Derivational Taxonomic Closure, and Sentence-Anchored Bridge Expansion (`Q16.16`).
//!
//! Pure deterministic fixed-point integer arithmetic (`ONE = 65_536`), zero vectors,
//! zero embeddings, zero LLMs, zero open-class word lists. Built strictly over Vestige's
//! SHA-256-pinned offline lexicons (WordNet 3.1, ConceptNet 5.7, Roget 1911, Wikidata CC0)
//! and closed-class English morphology.
//!
//! Grounded in:
//! - Thorat & Choudhari (COLING 2016, C16-1263): Hub-pruned bidirectional distance-decay
//!   reverse dictionary with `> 256` back-link hub cutoff and clause-conjunction filtering.
//! - Agirre, Lopez de Lacalle & Soroa (CL 2014, J14-1003): Degree-normalized local
//!   Push-PPR (`UKB`) with symmetric `1 / sqrt(deg(u) * deg(v))` edge scaling and
//!   `deg > 2048` hub-sink containment.
//! - Rao, Yarowsky & Callison-Burch (COLING 2008, W08-2006): Local Graph Laplacian
//!   Dirichlet harmonic voltage relaxation over `<= 160` nodes (`16` Gauss-Seidel rounds).
//! - Seco et al. (2004), Sanchez et al. (2011), Lin (1998), Patwardhan & Pedersen (2006,
//!   W06-2501), Gabrilovich & Markovitch (2007): Intrinsic IC, hub-rejecting Lin LCS,
//!   and sparse top-24 second-order gloss/ESA intersection.
//! - Wu (2026, `arXiv:2610.22588`, `2611.33124`): Deterministic 4-signal sentence-anchored
//!   bridge expansion (`SAG2`) with top-5 exact-match protection.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::retrieval::context::RankCtx;
use crate::retrieval::fusion::kemeny_mrf::ieee754_bits_to_q16;
use crate::retrieval::graph::supersession::turn_is_superseded;
use crate::retrieval::index::normalize::derivational_base_stems;
use crate::retrieval::lexical::speaker_translation_q16::is_open_domain_inferential_shape;
use crate::retrieval::{DetHashMap, DetHashSet, Index, is_content, raw_tokens, stem, stopwords};

/// Fixed-point unit `1.0` in `Q16.16`.
const ONE: i64 = 65_536;
/// Fixed-point `0.5` in `Q16.16`.
const HALF: i64 = 32_768;
/// `ln(2)` in `Q16.16` (`0.693147 * 65536`).
const LN2_Q16: i64 = 45_426;

/// Reverse-dictionary hub back-link cutoff (`Thorat & Choudhari 2016`).
const RD_HUB_BACKLINK_CAP: usize = 256;
/// Reverse-dictionary hop distance decay `alpha = 0.50` in `Q16.16`.
const RD_ALPHA_Q16: i64 = 32_768;
/// Reverse-dictionary conjunction hit multiplier `lambda = 0.50` in `Q16.16`.
const RD_LAMBDA_Q16: i64 = 32_768;
/// Maximum canonical headwords emitted by reverse-dictionary resolution.
const RD_TOP_K: usize = 8;

/// Push-PPR damping factor `c = 0.85` in `Q16.16`.
const PPR_C_Q16: i64 = 55_706;
/// Push-PPR teleportation weight `1 - c = 0.15` in `Q16.16`.
const PPR_ONE_MINUS_C_Q16: i64 = 9_830;
/// Push-PPR residual push threshold `eps = 6` (`~9.15e-5` in `Q16.16`).
const PPR_EPS_Q16: i64 = 6;
/// Maximum sparse residual pushes per query (`< 5 ms` CPU budget).
const PPR_MAX_PUSHES: usize = 256;
/// Hard degree hub-sink cap: nodes with `deg > 2048` absorb mass and never emit.
const PPR_HUB_SINK_DEG: u32 = 2_048;
/// Maximum nodes in the query-local Push-PPR concept graph.
const PPR_MAX_NODES: usize = 200;

/// Maximum nodes in the local Graph Laplacian harmonic subgraph.
const HARMONIC_MAX_NODES: usize = 160;
/// Number of integer Gauss-Seidel relaxation sweeps.
const HARMONIC_ITERS: usize = 16;
/// Ground-leakage conductance `gamma = 0.05` in `Q16.16`.
const HARMONIC_GAMMA_Q16: i64 = 3_277;

/// Total Princeton WordNet 3.1 noun synset nodes (`data.noun`).
const WN_MAX_NODES: u64 = 82_115;
/// `ln(82_115)` in `Q16.16` (`11.31587 * 65536`).
const LN_WN_MAX_NODES_Q16: i64 = 741_595;
/// Approximate leaf synsets in WordNet 3.1 noun DAG (`Sanchez et al. 2011`).
const WN_MAX_LEAVES: u64 = 64_992;
/// `ln(64_993)` in `Q16.16` (`11.08203 * 65536`).
const LN_WN_MAX_LEAVES_Q16: i64 = 726_277;
/// Minimum intrinsic IC for an LCS to be considered non-hub (`0.15` in `Q16.16`).
const HUB_IC_FLOOR_Q16: i64 = 9_830;
/// Sparse top-K high-IC gloss/ESA entries retained per concept.
const ESA_TOP_K: usize = 24;

/// Maximum seed turns inspected by deterministic `SAG2` bridge extraction.
const SAG2_MAX_SEEDS: usize = 10;
/// Maximum sentence-anchored bridge terms emitted by `SAG2`.
const SAG2_TOP_L: usize = 16;

/// 4-hop upward taxonomic decay `gamma^h` (`gamma = 0.75`) in `Q16.16`.
const UPWARD_HOP_DECAY_Q16: [i64; 5] = [65_536, 49_152, 36_864, 27_648, 20_736];

/// Bounded candidate budgets for `< 5 ms/query` execution.
const MAX_CANDIDATE_SESSIONS: usize = 24;
const MAX_SCORED_TURNS: usize = 64;
const MAX_HARMONIC_TURNS: usize = 16;

/// Symbolic proof thresholds and Owner-Directive additive caps (`Q16.16`).
const BRIDGE_MIN_PROOF_Q16: i64 = 16_384;
const STRONG_SLOT0_PROOF_Q16: i64 = 32_768;
/// Strict `0.35 * max_bm25` additive ceiling whenever `star_bit` is covered.
const STAR_COVERED_CAP_Q16: i64 = 22_938;
/// Full tie-break ceiling (`0.65 * max_bm25`) when exact lexical overlap is zero.
const ZERO_OVERLAP_CAP_Q16: i64 = 42_598;

// hardcode-scan: closed-class (English derivational prefixes verified strictly against WordNet senses)
const DERIV_PREFIXES: &[&str] = &[
    "un", "re", "non", "pre", "post", "over", "under", "sub", "inter", "anti", "mis", "dis", "co",
    "auto", "semi", "out",
];

// =========================================================================
// DETERMINISTIC FIXED-POINT Q16.16 INTEGER ARITHMETIC
// =========================================================================

/// Exact deterministic integer square root (`floor(sqrt(n))`).
#[inline]
fn isqrt_u64(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let shift = (64 - n.leading_zeros()).div_ceil(2);
    let mut x = 1u64 << shift;
    loop {
        let next = (x + n / x) >> 1;
        if next >= x {
            return x;
        }
        x = next;
    }
}

/// Deterministic `Q16.16` multiply using `i128` intermediate precision.
#[inline]
fn q16_mul(a: i64, b: i64) -> i64 {
    (((a as i128) * (b as i128)) >> 16) as i64
}

/// Deterministic `Q16.16` divide using `i128` intermediate precision.
#[inline]
fn q16_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    (((a as i128) << 16) / (b as i128)) as i64
}

/// Deterministic `log2(x)` in `Q16.16` for `x >= 1` using bit-length + 4-bit mantissa LUT.
#[inline]
fn log2_q16(x: u64) -> i64 {
    if x <= 1 {
        return 0;
    }
    let exp = 63 - x.leading_zeros();
    let frac_idx = if exp >= 4 {
        ((x >> (exp - 4)) & 0xF) as usize
    } else {
        ((x << (4 - exp)) & 0xF) as usize
    };
    const MANT_Q16: [i64; 16] = [
        0, 5_731, 11_136, 16_248, 21_098, 25_709, 30_105, 34_305, 38_326, 42_182, 45_887, 49_452,
        52_887, 56_201, 59_403, 62_500,
    ];
    ((exp as i64) << 16) + MANT_Q16[frac_idx]
}

/// Deterministic `ln(x)` in `Q16.16` for `x >= 1`.
#[inline]
fn ln_q16(x: u64) -> i64 {
    q16_mul(log2_q16(x), LN2_Q16)
}

/// Deterministic `ln(num / den)` in `Q16.16` for `num >= den > 0` with 8-bit sub-integer shift.
#[inline]
fn ln_ratio_q16(num: u64, den: u64) -> i64 {
    if den == 0 || num <= den {
        return 0;
    }
    let diff_log2 = if num < (1u64 << 55) && den < (1u64 << 55) {
        log2_q16(num << 8) - log2_q16(den << 8)
    } else {
        log2_q16(num) - log2_q16(den)
    };
    q16_mul(diff_log2.max(0), LN2_Q16)
}

/// Robertson-Sparck Jones BM25 IDF `ln(1 + (N - df + 0.5) / (df + 0.5))` in `Q16.16`.
#[inline]
fn idf_q16(n_docs: u64, df: u32) -> i64 {
    if df == 0 {
        return 0;
    }
    let n = n_docs.max(1);
    let d = df as u64;
    ln_ratio_q16(2 * n + 2, 2 * d + 1).max(0)
}

/// Normalize a query/document surface word into clean lowercase without ever slicing mid-UTF-8.
fn clean_word_lower(w: &str) -> String {
    let trimmed = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
    trimmed.to_lowercase()
}

/// Normalize and filter a slice of surface words to unique content tokens (order-preserving).
fn collect_content_words(words: &[&str], cap: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(words.len().min(cap));
    let mut seen: DetHashSet<String> = DetHashSet::default();
    let stop = stopwords();
    for &raw in words {
        let low = clean_word_lower(raw);
        if low.chars().count() < 3 || stop.contains(low.as_str()) || !is_content(&low) {
            continue;
        }
        if seen.insert(low.clone()) {
            out.push(low);
            if out.len() >= cap {
                break;
            }
        }
    }
    out
}

// =========================================================================
// 1. HUB-PRUNED BIDIRECTIONAL DISTANCE-DECAY REVERSE DICTIONARY (C16-1263)
// =========================================================================

/// Estimated reverse-definition back-link fanout for a WordNet token.
/// High-polysemy words whose reverse gloss map was already pruned (`> 140` in `wordnet.rs`)
/// and whose sense fanout is broad are treated as `> RD_HUB_BACKLINK_CAP`.
fn wordnet_rev_backlink_count(tok: &str) -> usize {
    let syn_hits = crate::wordnet::synsets_with_gloss(tok).len();
    let sense_cnt = crate::wordnet::senses(tok).len();
    if syn_hits == 0 && sense_cnt >= 10 {
        return RD_HUB_BACKLINK_CAP + sense_cnt * 16;
    }
    let comp_hits = crate::wordnet::gloss_and_compound_concepts(tok).len();
    let full_hits = crate::wordnet::full_lexicon_neighbors(tok).len();
    syn_hits.saturating_add(comp_hits).saturating_add(full_hits)
}

/// Definition-side Information Content `IC_def(tok)` in `Q16.16` derived from back-link fanout
/// and optional corpus IDF.
fn def_ic_q16(ix: &Index, tok: &str, backlinks: usize) -> i64 {
    let lex_ic = (ONE - q16_div(ln_q16((backlinks as u64).saturating_add(1)), LN_WN_MAX_NODES_Q16))
        .clamp(ONE >> 3, ONE);
    if let Some(vid) = ix.vocab.get(tok) {
        let df = ix.t_own_stats.df.get(&vid).copied().unwrap_or(0);
        if df > 0 {
            let n_docs = (ix.t_own_stats.n as u64).max(ix.turns.len() as u64).max(1);
            let corpus_ic = idf_q16(n_docs, df).clamp(ONE >> 3, ONE * 3);
            return ((lex_ic + corpus_ic) >> 1).clamp(ONE >> 3, ONE * 2);
        }
    }
    lex_ic
}

/// Configuration for [`reverse_dict_expand_core`].
struct RevDictExpandCfg {
    min_hits: u32,
    top_k: usize,
}

/// Core Thorat & Choudhari (COLING 2016) hub-pruned bidirectional reverse-dictionary scorer
/// operating on an explicit reverse-gloss map (`query_tok -> defined_headword_ids`),
/// 1-hop forward neighbors (`query_tok -> neighbor_toks`), and per-token back-link counts.
///
/// Guarantees:
/// - Any query token with `backlinks > RD_HUB_BACKLINK_CAP` (`256`) CANNOT seed candidates;
///   it only applies a bounded verifier bonus to headwords already reached by `>= min_hits`
///   low-degree seeds.
/// - Clause-conjunction filter requires `hits >= cfg.min_hits`.
/// - Scores by `base * (1 + 0.5 * min(hits - 1, 4))` in `Q16.16`, deterministic tie-break.
/// Returns `(target_id, sim_rd_q16, hit_mask)`.
fn reverse_dict_expand_core(
    q_tokens: &[u32],
    backlink_counts: &BTreeMap<u32, usize>,
    rev_gloss: &BTreeMap<u32, Vec<u32>>,
    fwd_hop1: &BTreeMap<u32, Vec<u32>>,
    ic_map: &BTreeMap<u32, i64>,
    cfg: RevDictExpandCfg,
) -> Vec<(u32, i64, u32)> {
    let mut acc: BTreeMap<u32, (i64, u32)> = BTreeMap::new();
    let mut hub_tokens: Vec<(usize, u32, i64)> = Vec::new();
    let mut rare_seed_count = 0usize;

    for (qi, &tok) in q_tokens.iter().take(16).enumerate() {
        let bl = backlink_counts
            .get(&tok)
            .copied()
            .unwrap_or_else(|| rev_gloss.get(&tok).map_or(0, Vec::len));
        let ic = ic_map.get(&tok).copied().unwrap_or(ONE).max(ONE >> 3);
        if bl > RD_HUB_BACKLINK_CAP {
            hub_tokens.push((qi, tok, ic));
            continue;
        }
        rare_seed_count += 1;
        let bit = 1u32 << qi;
        if let Some(direct) = rev_gloss.get(&tok) {
            for &hw in direct {
                let e = acc.entry(hw).or_insert((0, 0));
                e.0 = e.0.saturating_add(ic);
                e.1 |= bit;
            }
        }
        if let Some(nbrs) = fwd_hop1.get(&tok) {
            let decay_ic = q16_mul(ic, RD_ALPHA_Q16);
            for &nbr in nbrs {
                let nbl = backlink_counts
                    .get(&nbr)
                    .copied()
                    .unwrap_or_else(|| rev_gloss.get(&nbr).map_or(0, Vec::len));
                if nbl > RD_HUB_BACKLINK_CAP {
                    continue;
                }
                if let Some(rev_n) = rev_gloss.get(&nbr) {
                    for &hw in rev_n {
                        let e = acc.entry(hw).or_insert((0, 0));
                        e.0 = e.0.saturating_add(decay_ic);
                        e.1 |= bit;
                    }
                }
            }
        }
    }

    let eff_min_hits: u32 = cfg
        .min_hits
        .min(if rare_seed_count >= 2 { 2 } else { 1 })
        .max(1);

    // Dual-tier hub verifier bonus: >256-backlink hubs never seed candidates,
    // they only verify candidates already conjunctively reached by >= eff_min_hits rare seeds.
    for &(qi, hub_tok, ic_hub) in &hub_tokens {
        let Some(rev_h) = rev_gloss.get(&hub_tok) else {
            continue;
        };
        let bonus = (ic_hub >> 2).max(1);
        let bit = 1u32 << qi;
        for &hw in rev_h {
            if let Some(entry) = acc.get_mut(&hw)
                && entry.1.count_ones() >= eff_min_hits
            {
                entry.0 = entry.0.saturating_add(bonus);
                entry.1 |= bit;
            }
        }
    }

    let mut out: Vec<(u32, i64, u32)> = acc
        .into_iter()
        .filter_map(|(hw, (sc, mask))| {
            let hits = mask.count_ones();
            if hits < eff_min_hits {
                return None;
            }
            let mult_q16 = ONE + RD_LAMBDA_Q16 * i64::from((hits - 1).min(4));
            let sim_rd = q16_mul(sc, mult_q16);
            (sim_rd > 0).then_some((hw, sim_rd, mask))
        })
        .collect();
    out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.truncate(cfg.top_k);
    out
}

/// Resolve top-8 canonical corpus-attested headwords `(vocab_id, Sim_RD_q16)` for a
/// definitional or circumlocution query via hub-pruned bidirectional reverse-dictionary
/// expansion over WordNet 3.1 (`synsets_with_gloss`, `gloss_tokens_of`, `upward_hypernyms`,
/// `full_lexicon_neighbors`, `gloss_and_compound_concepts`).
pub(in crate::retrieval) fn reverse_dict_headwords(
    ix: &Index,
    q_words: &[&str],
) -> Vec<(u32, i64)> {
    let clean_q = collect_content_words(q_words, 16);
    if clean_q.is_empty() || ix.turns.is_empty() {
        return Vec::new();
    }
    let q_set: DetHashSet<&str> = clean_q.iter().map(String::as_str).collect();
    let max_head_df = ((ix.turns.len() * 25) / 100).max(8) as u32;

    let mut vid_acc: BTreeMap<u32, (i64, u32)> = BTreeMap::new();
    let mut hub_words: Vec<(usize, &str, i64)> = Vec::new();
    let mut rare_seed_count = 0usize;

    // Build explicit query-local reverse-gloss tables for `reverse_dict_expand_core`.
    let mut tok_ids: BTreeMap<String, u32> = BTreeMap::new();
    let mut q_tok_ids: Vec<u32> = Vec::with_capacity(clean_q.len());
    let mut backlink_counts: BTreeMap<u32, usize> = BTreeMap::new();
    let mut rev_gloss: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut fwd_hop1: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut ic_map: BTreeMap<u32, i64> = BTreeMap::new();

    let intern_rd_tok = |word: &str, map: &mut BTreeMap<String, u32>| -> u32 {
        if let Some(&id) = map.get(word) {
            return id;
        }
        let id = map.len() as u32;
        map.insert(word.to_string(), id);
        id
    };

    let add_vid = |word: &str, weight_q16: i64, bit: u32, acc: &mut BTreeMap<u32, (i64, u32)>| {
        if weight_q16 <= 0 || word.len() < 3 || q_set.contains(word) || !is_content(word) {
            return;
        }
        if let Some(vid) = ix.vocab.get(word) {
            let df = ix.t_own_stats.df.get(&vid).copied().unwrap_or(0);
            if (1..=max_head_df).contains(&df) {
                let e = acc.entry(vid).or_insert((0, 0));
                e.0 = e.0.saturating_add(weight_q16);
                e.1 |= bit;
            }
        }
    };

    for (qi, qw) in clean_q.iter().enumerate() {
        let tid = intern_rd_tok(qw, &mut tok_ids);
        q_tok_ids.push(tid);
        let bl = wordnet_rev_backlink_count(qw);
        let ic = def_ic_q16(ix, qw, bl);
        backlink_counts.insert(tid, bl);
        ic_map.insert(tid, ic);
        let direct_syns = crate::wordnet::synsets_with_gloss(qw);
        if !direct_syns.is_empty() {
            rev_gloss.insert(tid, direct_syns.to_vec());
        }

        if bl > RD_HUB_BACKLINK_CAP {
            hub_words.push((qi, qw.as_str(), ic));
            continue;
        }
        rare_seed_count += 1;
        let bit = 1u32 << qi;

        // 1. Direct reverse-gloss mono-headwords & compound concepts, verified by `gloss_tokens_of`.
        for cand in crate::wordnet::gloss_and_compound_concepts(qw)
            .into_iter()
            .take(40)
        {
            if q_set.contains(cand) || cand.len() < 3 {
                continue;
            }
            let direct_def = crate::wordnet::gloss_tokens_of(cand)
                .iter()
                .any(|&g| g == qw.as_str());
            let w = if direct_def {
                ic
            } else {
                q16_mul(ic, RD_ALPHA_Q16)
            };
            add_vid(cand, w, bit, &mut vid_acc);
        }

        // 2. Bipartite gloss-hypernym & LCS sibling bridge (1-hop decayed).
        let hop1_w = q16_mul(ic, RD_ALPHA_Q16);
        for (cand, _) in crate::wordnet::dual_gloss_hypernym_bridge(qw, 10) {
            add_vid(cand, hop1_w, bit, &mut vid_acc);
        }
        for (cand, _) in crate::wordnet::lcs_sibling_resonance(qw, 10) {
            add_vid(cand, hop1_w, bit, &mut vid_acc);
        }

        // 3. 1-hop forward neighbors (`upward_hypernyms` + `full_lexicon_neighbors`) -> reverse gloss.
        let mut hop1_nbrs: Vec<&'static str> = Vec::with_capacity(14);
        for hyp in crate::wordnet::upward_hypernyms(qw, 1).into_iter().take(6) {
            if !hop1_nbrs.contains(&hyp) {
                hop1_nbrs.push(hyp);
            }
        }
        for (nbr, rel_w) in crate::wordnet::full_lexicon_neighbors(qw)
            .into_iter()
            .take(10)
        {
            if rel_w >= 72 && !hop1_nbrs.contains(&nbr) {
                hop1_nbrs.push(nbr);
            }
        }
        let mut nbr_ids: Vec<u32> = Vec::with_capacity(hop1_nbrs.len());
        for nbr in hop1_nbrs {
            add_vid(nbr, hop1_w, bit, &mut vid_acc);
            let nid = intern_rd_tok(nbr, &mut tok_ids);
            nbr_ids.push(nid);
            backlink_counts
                .entry(nid)
                .or_insert_with(|| wordnet_rev_backlink_count(nbr));
            rev_gloss
                .entry(nid)
                .or_insert_with(|| crate::wordnet::synsets_with_gloss(nbr).iter().take(24).copied().collect());
        }
        fwd_hop1.insert(tid, nbr_ids);
    }

    // Run hub-pruned bidirectional synset expansion and resolve synsets to corpus headwords via KITE.
    let ranked_syns = reverse_dict_expand_core(
        &q_tok_ids,
        &backlink_counts,
        &rev_gloss,
        &fwd_hop1,
        &ic_map,
        RevDictExpandCfg {
            min_hits: 1,
            top_k: 28,
        },
    );
    let min_hits: u32 = if rare_seed_count >= 2 { 2 } else { 1 };

    if !ranked_syns.is_empty() {
        let kite = ix.kite_index();
        for &(w_syn, syn_sc, syn_mask) in &ranked_syns {
            let Some(postings) = kite.synset_postings.get(&w_syn) else {
                continue;
            };
            let mut resolved_for_syn: DetHashSet<u32> = DetHashSet::default();
            for &(ti, _) in postings.iter().take(10) {
                let tr = &ix.turns[ti];
                for &cid in &tr.content {
                    if !resolved_for_syn.insert(cid) {
                        continue;
                    }
                    let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
                    if !(1..=max_head_df).contains(&df) {
                        continue;
                    }
                    let Some(name) = ix.vocab.names.get(cid as usize) else {
                        continue;
                    };
                    if name.starts_with('\u{1}')
                        || name.len() < 3
                        || q_set.contains(name.as_str())
                    {
                        continue;
                    }
                    if crate::wordnet::senses(name).contains(&w_syn) {
                        let e = vid_acc.entry(cid).or_insert((0, 0));
                        e.0 = e.0.saturating_add(syn_sc);
                        e.1 |= syn_mask;
                    }
                }
            }
        }
    }

    // Dual-tier hub verifier bonus (Thorat & Choudhari 2016):
    // Hubs (>256 backlinks) NEVER create candidates, only verify conjunctively reached ones.
    if !hub_words.is_empty() {
        for (&vid, entry) in &mut vid_acc {
            if entry.1.count_ones() < min_hits {
                continue;
            }
            let Some(cand_word) = ix.vocab.names.get(vid as usize) else {
                continue;
            };
            let cand_gloss = crate::wordnet::gloss_tokens_of(cand_word);
            let cand_senses = crate::wordnet::senses(cand_word);
            for &(qi, h_tok, ic_h) in &hub_words {
                let verified = cand_gloss.contains(&h_tok)
                    || crate::wordnet::synsets_with_gloss(h_tok)
                        .iter()
                        .any(|s| cand_senses.contains(s));
                if verified {
                    entry.0 = entry.0.saturating_add((ic_h >> 2).max(1));
                    entry.1 |= 1u32 << qi;
                }
            }
        }
    }

    // Clause-conjunction filter & `Sim_RD(Q, w)` multiplier.
    let mut out: Vec<(u32, i64)> = vid_acc
        .into_iter()
        .filter_map(|(vid, (sc, mask))| {
            let hits = mask.count_ones();
            if hits < min_hits {
                return None;
            }
            let mult_q16 = ONE + RD_LAMBDA_Q16 * i64::from((hits - 1).min(4));
            let sim_rd = q16_mul(sc, mult_q16);
            (sim_rd > 0).then_some((vid, sim_rd))
        })
        .collect();
    out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.truncate(RD_TOP_K);
    out
}

// =========================================================================
// 2. DEGREE-NORMALIZED LOCAL PUSH-PPR (`UKB`, Agirre & Soroa CL 2014)
// =========================================================================

/// Compact deterministic sparse graph for local Push-PPR (`<= 200` nodes).
struct LocalPprGraph {
    /// Node index -> concept string.
    labels: Vec<String>,
    /// Concept string -> node index.
    index: BTreeMap<String, u32>,
    /// Global/effective node degree (used for `1 / sqrt(deg_u * deg_v)` and `> 2048` hub-sink).
    global_deg: Vec<u32>,
    /// Pre-scaled symmetric normalized outgoing transitions `(neighbor_id, w_q16)`.
    adj: Vec<Vec<(u32, i64)>>,
}

impl LocalPprGraph {
    fn new(cap: usize) -> Self {
        Self {
            labels: Vec::with_capacity(cap),
            index: BTreeMap::new(),
            global_deg: Vec::with_capacity(cap),
            adj: Vec::with_capacity(cap),
        }
    }

    fn intern(&mut self, label: &str, est_deg: u32, max_nodes: usize) -> Option<u32> {
        if let Some(&id) = self.index.get(label) {
            if est_deg > self.global_deg[id as usize] {
                self.global_deg[id as usize] = est_deg;
            }
            return Some(id);
        }
        if self.labels.len() >= max_nodes {
            return None;
        }
        let id = self.labels.len() as u32;
        self.labels.push(label.to_string());
        self.index.insert(label.to_string(), id);
        self.global_deg.push(est_deg.max(1));
        self.adj.push(Vec::new());
        Some(id)
    }

    #[inline]
    fn degree(&self, u: u32) -> u32 {
        self.global_deg
            .get(u as usize)
            .copied()
            .unwrap_or(1)
            .max(self.adj.get(u as usize).map_or(1, |v| v.len() as u32))
    }

    /// Normalize raw undirected/directed edge weights by `w_tilde = w / sqrt(deg(u) * deg(v))`
    /// and row-normalize each node's outgoing distribution (leaking deficit mass when a node
    /// connects only into high-degree hubs).
    fn finalize_symmetric_transitions(&mut self, raw_edges: &BTreeMap<(u32, u32), i64>) {
        let n = self.labels.len();
        let mut temp: Vec<Vec<(u32, i64)>> = vec![Vec::new(); n];
        for (&(u, v), &w_raw) in raw_edges {
            if u == v || w_raw <= 0 || (u as usize) >= n || (v as usize) >= n {
                continue;
            }
            let du = self.degree(u) as u64;
            let dv = self.degree(v) as u64;
            let geom = isqrt_u64(du.saturating_mul(dv)).max(1) as i64;
            let w_scaled = ((w_raw as i128 * 8) / (geom as i128)) as i64;
            if w_scaled > 0 {
                temp[u as usize].push((v, w_scaled));
            }
        }
        for (u, edges) in temp.into_iter().enumerate() {
            let row_sum: i64 = edges.iter().map(|&(_, w)| w).fold(0i64, i64::saturating_add);
            if row_sum <= 0 {
                continue;
            }
            // Floor denominator at `ONE` so lone transitions into mega-hubs attenuate cleanly.
            let denom = row_sum.max(ONE);
            let mut norm: Vec<(u32, i64)> = edges
                .into_iter()
                .filter_map(|(v, w)| {
                    let wn = q16_div(w, denom);
                    (wn > 0).then_some((v, wn))
                })
                .collect();
            norm.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            self.adj[u] = norm;
        }
    }
}

/// Estimated lexicon degree for a word across WordNet, ConceptNet, Roget, and Wikidata.
fn estimate_lexicon_degree(word: &str) -> u32 {
    let wn_senses = crate::wordnet::senses(word).len() as u32;
    let wn_nbrs = crate::wordnet::full_lexicon_neighbors(word).len() as u32;
    let syn_back = crate::wordnet::synsets_with_gloss(word).len() as u32;
    if wn_senses >= 14 {
        return PPR_HUB_SINK_DEG + wn_senses * 32;
    }
    (wn_senses * 8)
        .saturating_add(wn_nbrs)
        .saturating_add(syn_back * 2)
        .max(2)
}

/// Anderson-Chung-Lang sparse residual Push-PPR (`Q16.16`) with `deg > 2048` hub-sink cap.
fn sparse_push_ppr_q16(
    graph: &LocalPprGraph,
    seeds: &[(u32, i64)],
    eps_q16: i64,
    max_pushes: usize,
) -> Vec<(u32, i64)> {
    let mut p: BTreeMap<u32, i64> = BTreeMap::new();
    let mut r: BTreeMap<u32, i64> = BTreeMap::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut in_queue: DetHashSet<u32> = DetHashSet::default();

    for &(u, mass) in seeds {
        if mass <= 0 || (u as usize) >= graph.labels.len() {
            continue;
        }
        *r.entry(u).or_insert(0) += mass;
        if in_queue.insert(u) {
            queue.push_back(u);
        }
    }

    let mut pushes = 0usize;
    while let Some(u) = queue.pop_front() {
        in_queue.remove(&u);
        if pushes >= max_pushes {
            break;
        }
        let ru = r.get(&u).copied().unwrap_or(0);
        if ru < eps_q16 {
            continue;
        }
        pushes += 1;
        *p.entry(u).or_insert(0) += q16_mul(PPR_ONE_MINUS_C_Q16, ru);
        r.insert(u, 0);

        // Hard hub-sink cap: mega-hubs absorb `(1 - c) * ru` and NEVER emit forward.
        if graph.degree(u) > PPR_HUB_SINK_DEG {
            continue;
        }
        let push_mass = q16_mul(PPR_C_Q16, ru);
        if push_mass <= 0 {
            continue;
        }
        for &(v, w_q16) in &graph.adj[u as usize] {
            let delta = q16_mul(push_mass, w_q16);
            if delta <= 0 {
                continue;
            }
            let rv = r.entry(v).or_insert(0);
            *rv += delta;
            if *rv >= eps_q16 && in_queue.insert(v) {
                queue.push_back(v);
            }
        }
    }

    let mut out: Vec<(u32, i64)> = p.into_iter().filter(|&(_, v)| v > 0).collect();
    out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// Build a query-local multi-relational concept graph (WordNet 3.1 + ConceptNet 5.7 +
/// Roget 1911 + Wikidata CC0, `<= 200` nodes) and run degree-normalized Push-PPR.
fn query_local_push_ppr(ix: &Index, q_words: &[String]) -> Vec<(String, i64)> {
    if q_words.is_empty() {
        return Vec::new();
    }
    let mut graph = LocalPprGraph::new(PPR_MAX_NODES);
    let mut raw_edges: BTreeMap<(u32, u32), i64> = BTreeMap::new();
    let mut raw_seeds: Vec<(u32, i64)> = Vec::new();
    let n_docs = (ix.t_own_stats.n as u64).max(ix.turns.len() as u64).max(1);

    let add_undirected =
        |g: &mut LocalPprGraph, edges: &mut BTreeMap<(u32, u32), i64>, u: u32, nbr: &str, w: i64| {
            let clean = clean_word_lower(nbr);
            if clean.chars().count() < 3 || !is_content(&clean) {
                return;
            }
            let deg = estimate_lexicon_degree(&clean);
            if let Some(v) = g.intern(&clean, deg, PPR_MAX_NODES)
                && v != u
            {
                let e_uv = edges.entry((u, v)).or_insert(0);
                *e_uv = (*e_uv).max(w);
                let e_vu = edges.entry((v, u)).or_insert(0);
                *e_vu = (*e_vu).max(w);
            }
        };

    // Seed nodes with `v_i = IDF(q_k) / max(|senses(q_k)|, 1)`.
    for qw in q_words.iter().take(10) {
        let deg = estimate_lexicon_degree(qw);
        let Some(uid) = graph.intern(qw, deg, PPR_MAX_NODES) else {
            break;
        };
        let idf = ix
            .vocab
            .get(qw)
            .and_then(|vid| ix.t_own_stats.df.get(&vid).copied())
            .map_or(ONE, |df| idf_q16(n_docs, df).max(ONE >> 1));
        let sense_div = (crate::wordnet::senses(qw).len() as i64).clamp(1, 8);
        raw_seeds.push((uid, (idf / sense_div).max(ONE >> 3)));
    }

    // Expand 1-hop multi-relational neighborhood around seeds, then light 2-hop cross-links.
    let seed_ids: Vec<u32> = raw_seeds.iter().map(|&(u, _)| u).collect();
    for &uid in &seed_ids {
        let w = graph.labels[uid as usize].clone();
        if graph.degree(uid) > PPR_HUB_SINK_DEG {
            continue;
        }
        // WordNet 3.1: full 4-POS lexicon neighbors + upward hypernyms + gloss tokens.
        for (nbr, rel_w) in crate::wordnet::full_lexicon_neighbors(&w)
            .into_iter()
            .take(10)
        {
            let w_q16 = (i64::from(rel_w) * ONE) / 100;
            add_undirected(&mut graph, &mut raw_edges, uid, nbr, w_q16);
        }
        for hyp in crate::wordnet::upward_hypernyms(&w, 2).into_iter().take(6) {
            add_undirected(&mut graph, &mut raw_edges, uid, hyp, 49_152);
        }
        if (1..=5).contains(&crate::wordnet::senses(&w).len()) {
            for gt in crate::wordnet::gloss_tokens_of(&w).into_iter().take(6) {
                add_undirected(&mut graph, &mut raw_edges, uid, gt, 40_960);
            }
        }
        // ConceptNet 5.7: reciprocal 2-hop + direct 1-hop assertions.
        for (nbr, sc) in crate::conceptnet::reciprocal_two_hop_neighbors(&w, 8) {
            let w_q16 = ((i64::from(sc) * ONE) / 200).clamp(16_384, ONE);
            add_undirected(&mut graph, &mut raw_edges, uid, nbr, w_q16);
        }
        // Roget 1911: conceptual head/semicolon neighbors.
        for (nbr, sc) in crate::roget::conceptual_neighbors(&w, 6) {
            let w_q16 = ((i64::from(sc) * ONE) / 120).clamp(16_384, 52_428);
            add_undirected(&mut graph, &mut raw_edges, uid, nbr, w_q16);
        }
        // Wikidata CC0: upward containers + direct typed edges.
        for cont in crate::wikidata::upward_containers(&w, 2).into_iter().take(5) {
            for part in cont.split_ascii_whitespace() {
                add_undirected(&mut graph, &mut raw_edges, uid, part, 52_428);
            }
        }
        for (dst, _) in crate::wikidata::direct_typed_edges(&w).into_iter().take(5) {
            for part in dst.split_ascii_whitespace() {
                add_undirected(&mut graph, &mut raw_edges, uid, part, 55_706);
            }
        }
    }

    // Light 2-hop bridge closure among existing non-hub frontier nodes (no new node explosion).
    let hop1_end = graph.labels.len().min(64);
    for uid_usize in seed_ids.len()..hop1_end {
        let uid = uid_usize as u32;
        if graph.degree(uid) > PPR_HUB_SINK_DEG {
            continue;
        }
        let w = graph.labels[uid_usize].clone();
        for (nbr, rel_w) in crate::wordnet::full_lexicon_neighbors(&w)
            .into_iter()
            .take(5)
        {
            if let Some(&vid) = graph.index.get(nbr)
                && vid != uid
            {
                let w_q16 = (i64::from(rel_w) * HALF) / 100;
                let e_uv = raw_edges.entry((uid, vid)).or_insert(0);
                *e_uv = (*e_uv).max(w_q16);
                let e_vu = raw_edges.entry((vid, uid)).or_insert(0);
                *e_vu = (*e_vu).max(w_q16);
            }
        }
    }

    graph.finalize_symmetric_transitions(&raw_edges);

    // Normalize seed mass to sum to `ONE`.
    let seed_sum: i64 = raw_seeds
        .iter()
        .map(|&(_, m)| m)
        .fold(0i64, i64::saturating_add)
        .max(1);
    let norm_seeds: Vec<(u32, i64)> = raw_seeds
        .into_iter()
        .map(|(u, m)| (u, q16_div(m, seed_sum).max(1)))
        .collect();

    let ranked = sparse_push_ppr_q16(&graph, &norm_seeds, PPR_EPS_Q16, PPR_MAX_PUSHES);
    let seed_set: DetHashSet<u32> = seed_ids.into_iter().collect();
    ranked
        .into_iter()
        .filter(|&(u, _)| !seed_set.contains(&u))
        .take(20)
        .map(|(u, sc)| (graph.labels[u as usize].clone(), sc))
        .collect()
}

// =========================================================================
// 3. LOCAL GRAPH LAPLACIAN HARMONIC VOLTAGE RELAXATION (COLING 2008 W08-2006)
// =========================================================================

/// Undirected weighted local subgraph (`<= 160` nodes) for Dirichlet harmonic relaxation.
struct HarmonicSubgraph {
    adj: Vec<Vec<(usize, i64)>>,
}

impl HarmonicSubgraph {
    fn new(num_nodes: usize) -> Self {
        Self {
            adj: vec![Vec::new(); num_nodes],
        }
    }

    fn add_undirected_edge(&mut self, u: usize, v: usize, conductance_q16: i64) {
        if u == v || conductance_q16 <= 0 || u >= self.adj.len() || v >= self.adj.len() {
            return;
        }
        if let Some(existing) = self.adj[u].iter_mut().find(|(nbr, _)| *nbr == v) {
            existing.1 = existing.1.max(conductance_q16);
        } else {
            self.adj[u].push((v, conductance_q16));
        }
        if let Some(existing) = self.adj[v].iter_mut().find(|(nbr, _)| *nbr == u) {
            existing.1 = existing.1.max(conductance_q16);
        } else {
            self.adj[v].push((u, conductance_q16));
        }
    }
}

/// Run `HARMONIC_ITERS` (`16`) integer Gauss-Seidel Dirichlet relaxation updates:
/// `u(v) = (sum_{z in N(v)} w_{vz} * u(z)) / (gamma + sum_{z in N(v)} w_{vz})`
/// with pinned boundary sources `q_sources` (`u(q) = 65_536`) and ground leakage `gamma_q16`.
/// Returns mean equilibrium voltage across `d_targets` in `Q16.16`.
fn harmonic_affinity_q16(
    sub: &HarmonicSubgraph,
    q_sources: &[(usize, i64)],
    d_targets: &[usize],
    gamma_q16: i64,
) -> i64 {
    let n = sub.adj.len();
    if n == 0 || q_sources.is_empty() || d_targets.is_empty() {
        return 0;
    }
    let mut u = vec![0i64; n];
    let mut pinned = vec![false; n];
    for &(q, val) in q_sources {
        if q < n {
            u[q] = val.clamp(0, ONE);
            pinned[q] = true;
        }
    }
    let gamma = gamma_q16.max(1) as i128;
    for _ in 0..HARMONIC_ITERS {
        for v in 0..n {
            if pinned[v] {
                continue;
            }
            let mut num: i128 = 0;
            let mut den: i128 = gamma;
            for &(nbr, w_q16) in &sub.adj[v] {
                num += (w_q16 as i128) * (u[nbr] as i128);
                den += w_q16 as i128;
            }
            u[v] = if den > 0 {
                ((num / den) as i64).clamp(0, ONE)
            } else {
                0
            };
        }
    }
    let mut sum: i128 = 0;
    let mut cnt: i128 = 0;
    for &d in d_targets {
        if d < n {
            sum += u[d] as i128;
            cnt += 1;
        }
    }
    if cnt == 0 {
        0
    } else {
        ((sum / cnt) as i64).clamp(0, ONE)
    }
}

// =========================================================================
// 4 & 6. SECO/SANCHEZ INTRINSIC IC, LIN LCS, SPARSE TOP-24 ESA,
//        4-HOP UPWARD WORDNET/WIKIDATA CLOSURE & DERIVATIONAL MORPHOLOGY
// =========================================================================

/// Seco, Veale & Hayes (2004) Intrinsic IC in `Q16.16`:
/// `IC_Seco(c) = 1 - ln(desc + 1) / ln(max_nodes)`.
#[inline]
fn seco_ic_q16(desc_count: u32) -> i64 {
    if (desc_count as u64) >= WN_MAX_NODES {
        return 0;
    }
    let num = ln_q16((desc_count as u64).saturating_add(1));
    (ONE - q16_div(num, LN_WN_MAX_NODES_Q16)).clamp(0, ONE)
}

/// Sanchez, Batet & Isern (2011) DAG-safe Intrinsic IC in `Q16.16` normalized to `[0, ONE]`:
/// `IC_Sanchez(c) = -ln(((leaves / subsumers) + 1) / (max_leaves + 1)) / ln(max_leaves + 1)`.
#[inline]
fn sanchez_ic_q16(leaves: u32, subsumers: u32) -> i64 {
    let eff_leaves = (leaves as u64) / (subsumers.max(1) as u64);
    let num = eff_leaves.saturating_add(1);
    let den = WN_MAX_LEAVES + 1;
    if num >= den {
        return 0;
    }
    let neg_ln = ln_ratio_q16(den, num);
    q16_div(neg_ln, LN_WN_MAX_LEAVES_Q16).clamp(0, ONE)
}

/// Combined Seco/Sanchez intrinsic IC for a word given its WordNet `specific_ancestor_synsets`.
fn word_intrinsic_ic_q16(ancestors: &[(u32, u32)]) -> i64 {
    if ancestors.is_empty() {
        return HALF;
    }
    let min_desc = ancestors[0].1;
    let seco = seco_ic_q16(min_desc);
    let subsumers = ancestors.len() as u32;
    let sanchez = sanchez_ic_q16(min_desc, subsumers);
    ((seco + sanchez) >> 1).clamp(ONE >> 3, ONE)
}

/// Lin (1998) taxonomic similarity in `Q16.16` over precomputed specific ancestor synsets:
/// `sim_Lin(c1, c2) = 2 * IC(LCS) / (IC(c1) + IC(c2))`, automatically `0` at root hubs.
fn lin_similarity_q16(
    anc1: &[(u32, u32)],
    ic1: i64,
    anc2: &[(u32, u32)],
    ic2: i64,
) -> i64 {
    if anc1.is_empty() || anc2.is_empty() || ic1 + ic2 <= 0 {
        return 0;
    }
    let mut best_ic_lcs: i64 = 0;
    for &(s1, d1) in anc1 {
        if anc2.iter().any(|&(s2, _)| s2 == s1) {
            let ic_s = seco_ic_q16(d1);
            if ic_s > best_ic_lcs {
                best_ic_lcs = ic_s;
            }
        }
    }
    if best_ic_lcs <= HUB_IC_FLOOR_Q16 {
        return 0;
    }
    q16_div(2 * best_ic_lcs, ic1 + ic2).clamp(0, ONE)
}

/// Deterministic derivational morphology variants (closed-class suffixes via
/// `derivational_base_stems` + closed-class prefixes verified against `wordnet::senses` +
/// WordNet `+` derivationally related forms). 100% multibyte UTF-8 safe.
fn derivational_variants(word: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut push_uniq = |cand: String| {
        if cand.chars().count() >= 3 && cand != word && !out.contains(&cand) {
            out.push(cand);
        }
    };

    // 1. Suffix-based derivational base stems (WordNet-verified).
    if word.is_ascii() {
        for base in derivational_base_stems(word) {
            push_uniq(base);
        }
        // 2. Closed-class prefix stripping strictly verified against WordNet noun senses.
        for &pref in DERIV_PREFIXES {
            if let Some(rest) = word.strip_prefix(pref)
                && rest.len() >= 4
                && !crate::wordnet::senses(rest).is_empty()
            {
                push_uniq(rest.to_string());
            }
        }
    }

    // 3. Snowball/Porter deterministic stem.
    let st = stem(word);
    if st.chars().count() >= 4 {
        push_uniq(st);
    }

    // 4. Cross-POS WordNet `+` derivationally related forms / pertainyms (`rel_w >= 95`).
    for (nbr, rel_w) in crate::wordnet::full_lexicon_neighbors(word)
        .into_iter()
        .take(8)
    {
        if rel_w >= 95 {
            push_uniq(nbr.to_string());
        }
    }

    out
}

/// Precomputed symbolic profile for a single content word (reused across turns of one query).
struct WordSymbolicProfile {
    word: String,
    ancestors: Vec<(u32, u32)>,
    ic_q16: i64,
    supersense: u64,
    /// Sparse top-24 sorted high-IC 2nd-order gloss/ESA vector (`L1`-normalized to `ONE`).
    sparse_esa: Vec<(String, i64)>,
    /// 4-hop upward WordNet/Wikidata closure concepts -> `0.75^hop` weight in `Q16.16`.
    upward_4hop: BTreeMap<String, i64>,
    /// Derivational base/prefix/cross-POS variants.
    derivs: Vec<String>,
}

/// Build a cached [`WordSymbolicProfile`] for `word` strictly from offline lexicons.
fn build_word_profile(word: &str) -> WordSymbolicProfile {
    let ancestors = crate::wordnet::specific_ancestor_synsets(word, 4, 1_200);
    let ic_q16 = word_intrinsic_ic_q16(&ancestors);
    let supersense = crate::wordnet::supersense_mask(word);
    let derivs = derivational_variants(word);

    // Build 4-hop upward WordNet + Wikidata closure with `gamma = 0.75` decay per hop.
    let mut upward_4hop: BTreeMap<String, i64> = BTreeMap::new();
    upward_4hop.insert(word.to_string(), UPWARD_HOP_DECAY_Q16[0]);
    for d in &derivs {
        upward_4hop
            .entry(d.clone())
            .and_modify(|cur| *cur = (*cur).max(UPWARD_HOP_DECAY_Q16[1]))
            .or_insert(UPWARD_HOP_DECAY_Q16[1]);
    }
    for hop in 1..=4usize {
        let decay = UPWARD_HOP_DECAY_Q16[hop];
        for hyp in crate::wordnet::upward_hypernyms(word, hop)
            .into_iter()
            .take(12)
        {
            if hyp.len() >= 3 {
                upward_4hop
                    .entry(hyp.to_string())
                    .and_modify(|cur| *cur = (*cur).max(decay))
                    .or_insert(decay);
            }
        }
        if hop <= 3 {
            for hol in crate::wordnet::holonyms_and_upward(word, hop)
                .into_iter()
                .take(10)
            {
                if hol.len() >= 3 {
                    upward_4hop
                        .entry(hol.to_string())
                        .and_modify(|cur| *cur = (*cur).max(decay))
                        .or_insert(decay);
                }
            }
            for cont in crate::wikidata::upward_containers(word, hop)
                .into_iter()
                .take(8)
            {
                for part in cont.split_ascii_whitespace() {
                    if part.len() >= 3 && is_content(part) {
                        upward_4hop
                            .entry(part.to_string())
                            .and_modify(|cur| *cur = (*cur).max(decay))
                            .or_insert(decay);
                    }
                }
            }
        }
    }
    for ph in crate::wordnet::phrase_and_lemma_closure(word, 2)
        .into_iter()
        .take(8)
    {
        for part in ph.split_ascii_whitespace() {
            if part.len() >= 3 && is_content(part) {
                upward_4hop
                    .entry(part.to_string())
                    .and_modify(|cur| *cur = (*cur).max(UPWARD_HOP_DECAY_Q16[2]))
                    .or_insert(UPWARD_HOP_DECAY_Q16[2]);
            }
        }
    }

    // Build sparse top-24 high-IC 2nd-order gloss / ESA vector (`Patwardhan & Pedersen 2006`).
    let mut raw_esa: BTreeMap<String, i64> = BTreeMap::new();
    let mut add_esa = |tok: &str, scale_q16: i64| {
        if tok.len() < 3 || tok == word || !is_content(tok) {
            return;
        }
        let scnt = crate::wordnet::senses(tok).len();
        if scnt > 6 {
            return;
        }
        let tok_ic = (ONE / (scnt.max(1) as i64)).max(ONE >> 3);
        let w = q16_mul(tok_ic, scale_q16);
        if w > 0 {
            raw_esa
                .entry(tok.to_string())
                .and_modify(|cur| *cur = (*cur).max(w))
                .or_insert(w);
        }
    };

    // Direct WordNet gloss tokens (`scale = 1.0`).
    for gt in crate::wordnet::gloss_tokens_of(word).into_iter().take(16) {
        add_esa(gt, ONE);
    }
    // 2nd-order gloss tokens from immediate hypernyms (`scale = 0.75`, capped at 2 parents).
    for hyp in crate::wordnet::upward_hypernyms(word, 1).into_iter().take(2) {
        add_esa(hyp, UPWARD_HOP_DECAY_Q16[1]);
        for hgt in crate::wordnet::gloss_tokens_of(hyp).into_iter().take(8) {
            add_esa(hgt, UPWARD_HOP_DECAY_Q16[1]);
        }
    }
    // ConceptNet reciprocal 2-hop explicit concepts (`scale = 0.75`).
    for (cn, sc) in crate::conceptnet::reciprocal_two_hop_neighbors(word, 8) {
        let scale = ((i64::from(sc) * UPWARD_HOP_DECAY_Q16[1]) / 150).clamp(16_384, ONE);
        add_esa(cn, scale);
    }
    // Wikidata CC0 upward container concepts (`scale = 0.75`).
    for cont in crate::wikidata::upward_containers(word, 2).into_iter().take(6) {
        for part in cont.split_ascii_whitespace() {
            add_esa(part, UPWARD_HOP_DECAY_Q16[1]);
        }
    }

    // Keep top `ESA_TOP_K` (`24`) highest-weight terms, L1-normalize to `ONE`, sort by key.
    let mut top_esa: Vec<(String, i64)> = raw_esa.into_iter().collect();
    top_esa.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top_esa.truncate(ESA_TOP_K);
    let l1_sum: i64 = top_esa
        .iter()
        .map(|(_, w)| *w)
        .fold(0i64, i64::saturating_add);
    if l1_sum > 0 {
        for (_, w) in &mut top_esa {
            *w = q16_div(*w, l1_sum).max(1);
        }
    }
    top_esa.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    WordSymbolicProfile {
        word: word.to_string(),
        ancestors,
        ic_q16,
        supersense,
        sparse_esa: top_esa,
        upward_4hop,
        derivs,
    }
}

/// `O(K)` sorted merge-join dot-product / overlap over sparse top-24 L1-normalized ESA lists.
/// Scaled by `min(ONE, 2 * sum min(a_i, b_i))` + quadratic emphasis on multi-term overlap.
fn sparse_esa_overlap_q16(a: &[(String, i64)], b: &[(String, i64)]) -> i64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (mut i, mut j) = (0usize, 0usize);
    let mut shared_mass: i64 = 0;
    let mut shared_cnt: i64 = 0;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                shared_mass = shared_mass.saturating_add(a[i].1.min(b[j].1));
                shared_cnt += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    if shared_cnt == 0 {
        return 0;
    }
    // Histogram-intersection similarity in `[0, ONE]` with multi-facet reinforcement.
    let mult = if shared_cnt >= 2 { ONE + (ONE >> 2) } else { ONE };
    q16_mul(shared_mass.saturating_mul(2), mult).clamp(0, ONE)
}

/// Pairwise taxonomic + derivational + 4-hop upward + `supersense_mask` compatibility score.
fn pair_symbolic_similarity_q16(
    q_prof: &WordSymbolicProfile,
    d_prof: &WordSymbolicProfile,
) -> i64 {
    if q_prof.word == d_prof.word {
        return ONE;
    }

    // 1. Lin taxonomic similarity via Seco/Sanchez intrinsic IC (automatically 0 at root hubs).
    let lin = lin_similarity_q16(
        &q_prof.ancestors,
        q_prof.ic_q16,
        &d_prof.ancestors,
        d_prof.ic_q16,
    );

    // 2. Sparse top-24 2nd-order gloss/ESA intersection.
    let esa = sparse_esa_overlap_q16(&q_prof.sparse_esa, &d_prof.sparse_esa);

    // 3. Asymmetric 4-hop upward closure (`turn_word => query_word` concrete->abstract!)
    //    + Heyting implication + derivational morphology bridge.
    let mut upward_sim: i64 = 0;
    if let Some(&w_up) = d_prof.upward_4hop.get(&q_prof.word) {
        upward_sim = upward_sim.max(w_up);
    }
    if let Some(&w_down) = q_prof.upward_4hop.get(&d_prof.word) {
        // Inverse direction (query more specific than turn) is attenuated by 0.625.
        upward_sim = upward_sim.max(q16_mul(w_down, 40_960));
    }
    // Check shared specific upward concept (`Vivaldi -> composer <- Bach/Mozart`).
    if upward_sim < UPWARD_HOP_DECAY_Q16[2] {
        for (c, &wq) in &q_prof.upward_4hop {
            if let Some(&wd) = d_prof.upward_4hop.get(c)
                && crate::wordnet::senses(c).len() <= 5
            {
                let meet = q16_mul(wq.min(wd), 49_152);
                upward_sim = upward_sim.max(meet);
            }
        }
    }
    let heyt = crate::wordnet::heyting_implication_ic_q16(&d_prof.word, &q_prof.word);
    if heyt >= 68 {
        let heyt_q16 = (i64::from(heyt) * ONE) / 100;
        upward_sim = upward_sim.max(heyt_q16);
    }
    // Derivational morphology match (`bake <-> bakery`, `educate <-> educational`, `unhappy <-> happy`).
    let deriv_match = q_prof.derivs.iter().any(|dq| {
        dq == &d_prof.word || d_prof.derivs.iter().any(|dd| dd == dq)
    }) || d_prof.derivs.iter().any(|dd| dd == &q_prof.word);
    if deriv_match {
        upward_sim = upward_sim.max(55_706); // 0.85
    }

    let mut best = lin.max(esa).max(upward_sim);
    if best == 0 {
        return 0;
    }

    // 4. WordNet `supersense_mask` selectional compatibility check:
    //    Boost compatible semantic classes (+12.5%); dampen disjoint narrow-sense noise (*0.75).
    if q_prof.supersense != 0 && d_prof.supersense != 0 {
        if (q_prof.supersense & d_prof.supersense) != 0 {
            best = best.saturating_add(best >> 3).min(ONE);
        } else if !deriv_match
            && heyt < 68
            && crate::wordnet::senses(&q_prof.word).len() <= 2
            && crate::wordnet::senses(&d_prof.word).len() <= 2
        {
            best = (best * 3) >> 2;
        }
    }

    best.clamp(0, ONE)
}

/// Build the local 2-hop Harmonic Laplacian subgraph (`<= 160` nodes) bridging query
/// profiles `S_Q` and turn profiles `S_D`, then return equilibrium mean voltage in `Q16.16`.
fn local_harmonic_bridge_q16(
    q_profs: &[&WordSymbolicProfile],
    d_profs: &[&WordSymbolicProfile],
) -> i64 {
    if q_profs.is_empty() || d_profs.is_empty() {
        return 0;
    }
    let mut node_idx: BTreeMap<String, usize> = BTreeMap::new();
    let mut node_deg: Vec<u32> = Vec::with_capacity(HARMONIC_MAX_NODES);

    let intern = |w: &str,
                  est_deg: u32,
                  idx_map: &mut BTreeMap<String, usize>,
                  deg_vec: &mut Vec<u32>|
     -> Option<usize> {
        if est_deg > PPR_HUB_SINK_DEG {
            return None;
        }
        if let Some(&id) = idx_map.get(w) {
            if est_deg > deg_vec[id] {
                deg_vec[id] = est_deg;
            }
            return Some(id);
        }
        if deg_vec.len() >= HARMONIC_MAX_NODES {
            return None;
        }
        let id = deg_vec.len();
        idx_map.insert(w.to_string(), id);
        deg_vec.push(est_deg.max(1));
        Some(id)
    };

    let mut q_sources: Vec<(usize, i64)> = Vec::new();
    let mut d_targets: Vec<usize> = Vec::new();

    for qp in q_profs.iter().take(10) {
        let deg = estimate_lexicon_degree(&qp.word).min(PPR_HUB_SINK_DEG);
        if let Some(id) = intern(&qp.word, deg, &mut node_idx, &mut node_deg)
            && !q_sources.iter().any(|&(x, _)| x == id)
        {
            q_sources.push((id, ONE));
        }
    }
    for dp in d_profs.iter().take(12) {
        let deg = estimate_lexicon_degree(&dp.word).min(PPR_HUB_SINK_DEG);
        if let Some(id) = intern(&dp.word, deg, &mut node_idx, &mut node_deg)
            && !d_targets.contains(&id)
        {
            d_targets.push(id);
        }
    }
    if q_sources.is_empty() || d_targets.is_empty() {
        return 0;
    }

    // Collect intermediate 1-hop bridge concepts shared or adjacent to `S_Q` and `S_D`.
    let mut raw_edges: Vec<(usize, usize, i64)> = Vec::new();

    // Direct pairwise symbolic conductances between `S_Q` and `S_D`.
    for qp in q_profs.iter().take(10) {
        let Some(&qid) = node_idx.get(&qp.word) else {
            continue;
        };
        for dp in d_profs.iter().take(12) {
            let Some(&did) = node_idx.get(&dp.word) else {
                continue;
            };
            if qid == did {
                continue;
            }
            let sim = pair_symbolic_similarity_q16(qp, dp);
            if sim >= 12_288 {
                let ic_scale = ((qp.ic_q16 + dp.ic_q16) >> 1).max(ONE >> 2);
                raw_edges.push((qid, did, q16_mul(sim, ic_scale)));
            }
        }
    }

    // 2-hop intermediate bridge nodes from `upward_4hop` and `sparse_esa` of `S_Q` & `S_D`.
    for qp in q_profs.iter().take(8) {
        let Some(&qid) = node_idx.get(&qp.word) else {
            continue;
        };
        for (c, &w_up) in qp.upward_4hop.iter().take(8) {
            if c == &qp.word {
                continue;
            }
            let deg = estimate_lexicon_degree(c);
            if let Some(cid) = intern(c, deg, &mut node_idx, &mut node_deg) {
                raw_edges.push((qid, cid, q16_mul(w_up, qp.ic_q16)));
            }
        }
        for (g, w_g) in qp.sparse_esa.iter().take(6) {
            let deg = estimate_lexicon_degree(g);
            if let Some(gid) = intern(g, deg, &mut node_idx, &mut node_deg) {
                raw_edges.push((qid, gid, q16_mul((*w_g).saturating_mul(4).min(ONE), qp.ic_q16)));
            }
        }
    }
    for dp in d_profs.iter().take(10) {
        let Some(&did) = node_idx.get(&dp.word) else {
            continue;
        };
        for (c, &w_up) in dp.upward_4hop.iter().take(8) {
            if c == &dp.word {
                continue;
            }
            if let Some(&cid) = node_idx.get(c) {
                raw_edges.push((did, cid, q16_mul(w_up, dp.ic_q16)));
            }
        }
        for (g, w_g) in dp.sparse_esa.iter().take(6) {
            if let Some(&gid) = node_idx.get(g) {
                raw_edges.push((did, gid, q16_mul((*w_g).saturating_mul(4).min(ONE), dp.ic_q16)));
            }
        }
    }

    let mut sub = HarmonicSubgraph::new(node_deg.len());
    for (u, v, w_raw) in raw_edges {
        if u == v || w_raw <= 0 {
            continue;
        }
        let geom = isqrt_u64((node_deg[u] as u64).saturating_mul(node_deg[v] as u64)).max(1) as i64;
        let scaled = ((w_raw as i128 * 4) / (geom as i128)) as i64;
        if scaled > 0 {
            sub.add_undirected_edge(u, v, scaled.min(ONE));
        }
    }

    harmonic_affinity_q16(&sub, &q_sources, &d_targets, HARMONIC_GAMMA_Q16)
}

/// Compute BMA (Seco/Sanchez IC + Lin + Sparse Top-24 ESA + 4-hop Upward + Derivational)
/// fused with optional Harmonic Laplacian Voltage Relaxation in pure `Q16.16`.
fn score_profiles_bma_q16(
    q_profs: &[&WordSymbolicProfile],
    d_profs: &[&WordSymbolicProfile],
) -> i64 {
    if q_profs.is_empty() || d_profs.is_empty() {
        return 0;
    }
    let mut bma_num: i128 = 0;
    let mut bma_den: i128 = 0;
    let mut matched_q = 0i64;
    for qp in q_profs {
        let mut best_pair = 0i64;
        for dp in d_profs {
            let sim = pair_symbolic_similarity_q16(qp, dp);
            if sim > best_pair {
                best_pair = sim;
            }
        }
        if best_pair >= 12_288 {
            matched_q += 1;
        }
        bma_num += (qp.ic_q16 as i128) * (best_pair as i128);
        bma_den += qp.ic_q16 as i128;
    }
    if bma_den <= 0 {
        return 0;
    }
    let mut bma = (bma_num / bma_den) as i64;
    if matched_q >= 2 && bma > 0 {
        let bonus = ((matched_q - 1).min(3) * bma) >> 3;
        bma = bma.saturating_add(bonus).min(ONE);
    }
    bma.clamp(0, ONE)
}

/// Score semantic/taxonomic affinity between query words and candidate turn words in pure
/// `Q16.16`, fusing Seco/Sanchez Intrinsic IC + Lin LCS + Sparse Top-24 2nd-Order Gloss/ESA
/// + 4-Hop Upward WordNet/Wikidata + Derivational Morphology + Local Graph Laplacian Harmonic
/// Voltage Relaxation (`16` Gauss-Seidel sweeps).
#[cfg(test)]
pub(in crate::retrieval) fn score_turn_harmonic_esa_q16(
    q_words: &[&str],
    turn_words: &[&str],
) -> i64 {
    let q_clean = collect_content_words(q_words, 10);
    let d_clean = collect_content_words(turn_words, 16);
    if q_clean.is_empty() || d_clean.is_empty() {
        return 0;
    }
    let q_owned: Vec<WordSymbolicProfile> = q_clean.iter().map(|w| build_word_profile(w)).collect();
    let d_owned: Vec<WordSymbolicProfile> = d_clean.iter().map(|w| build_word_profile(w)).collect();
    let q_refs: Vec<&WordSymbolicProfile> = q_owned.iter().collect();
    let d_refs: Vec<&WordSymbolicProfile> = d_owned.iter().collect();

    let bma_q16 = score_profiles_bma_q16(&q_refs, &d_refs);
    let harm_q16 = local_harmonic_bridge_q16(&q_refs, &d_refs);
    // 0.60 * BMA + 0.40 * Harmonic Voltage, preserving `[0, ONE]`.
    (q16_mul(bma_q16, 39_322) + q16_mul(harm_q16, 26_214)).clamp(0, ONE)
}

// =========================================================================
// 5. DETERMINISTIC SAG2 4-SIGNAL SENTENCE-ANCHORED BRIDGE EXPANSION
// =========================================================================

/// Extract up to `top_l` (`<= 16`) sentence-anchored bridge terms `(vocab_id, a_r_q16)`
/// from seed turns `seeds = [(turn_idx, score_q16)]` using the exact 4-signal `SAG2` formula:
/// `a_r(t) = max_{d in S_r} [ (1 + ln(tf(t, d))) * I(t) * (1 + g_q(t, d)) * w_r(d) ]`
/// in pure deterministic `Q16.16` integer arithmetic.
fn sag2_extract_bridge_terms(
    ix: &Index,
    seeds: &[(usize, i64)],
    q_token_ids: &DetHashSet<u32>,
    top_l: usize,
) -> Vec<(u32, i64)> {
    if seeds.is_empty() || top_l == 0 || ix.turns.is_empty() {
        return Vec::new();
    }
    let n_docs = (ix.t_own_stats.n as u64).max(ix.turns.len() as u64).max(1);
    let max_bridge_df = ((ix.turns.len() * 25) / 100).max(6) as u32;
    let max_s = seeds
        .iter()
        .map(|&(_, s)| s)
        .max()
        .unwrap_or(0)
        .max(1);

    let mut best_a: BTreeMap<u32, i64> = BTreeMap::new();

    for &(ti, s_d) in seeds.iter().take(SAG2_MAX_SEEDS) {
        let Some(tr) = ix.turns.get(ti) else {
            continue;
        };
        if tr.is_question || tr.is_backchannel {
            continue;
        }
        // `w_r(d) = 0.5 + 0.5 * s_r(d) / max_{d'} s_r(d')` in `Q16.16`.
        let w_d = HALF + (((HALF as i128) * (s_d.max(0) as i128)) / (max_s as i128)) as i64;

        // Compute integer turn-level term frequencies `tf(t, d)` from turn prose.
        let mut turn_tf: DetHashMap<u32, u64> = DetHashMap::default();
        for tok in raw_tokens(&tr.body_for_headers) {
            if is_content(&tok)
                && let Some(vid) = ix.vocab.get(&tok)
            {
                *turn_tf.entry(vid).or_insert(0) += 1;
            }
        }

        // Split turn into sentences (UTF-8 safe char delimiters) and compute sentence-anchored `g_q`.
        for sent in tr.body_for_headers.split(['.', '!', '?', '\n', ';']) {
            let trimmed = sent.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut sent_ids: Vec<u32> = Vec::new();
            for tok in raw_tokens(trimmed) {
                if tok.chars().count() < 3 || !is_content(&tok) {
                    continue;
                }
                if let Some(vid) = ix.vocab.get(&tok)
                    && !sent_ids.contains(&vid)
                {
                    sent_ids.push(vid);
                }
            }
            if sent_ids.is_empty() {
                continue;
            }

            // `g_q(u) = sum_{v in T(u) cap T(q)} I(v)` in `Q16.16`.
            let mut g_q: i64 = 0;
            for &vid in &sent_ids {
                if q_token_ids.contains(&vid) {
                    let df = ix.t_own_stats.df.get(&vid).copied().unwrap_or(0);
                    g_q = g_q.saturating_add(idf_q16(n_docs, df));
                }
            }
            let one_plus_gq = ONE.saturating_add(g_q);

            for &tid in &sent_ids {
                if q_token_ids.contains(&tid) {
                    continue;
                }
                let df = ix.t_own_stats.df.get(&tid).copied().unwrap_or(0);
                if df == 0 || df > max_bridge_df {
                    continue;
                }
                let tf_d = turn_tf.get(&tid).copied().unwrap_or(1).max(1);
                let tf_w = ONE.saturating_add(ln_q16(tf_d));
                let idf = idf_q16(n_docs, df);
                let step1 = q16_mul(tf_w, idf);
                let step2 = q16_mul(step1, one_plus_gq);
                let a_val = q16_mul(step2, w_d);
                if a_val > 0 {
                    let entry = best_a.entry(tid).or_insert(0);
                    if a_val > *entry {
                        *entry = a_val;
                    }
                }
            }
        }
    }

    let mut ranked: Vec<(u32, i64)> = best_a.into_iter().collect();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(top_l);
    ranked
}

// =========================================================================
// 7. PIPELINE ENTRY POINT: DEFICIT-GATED BRIDGE & OWNER-DIRECTIVE SEATING
// =========================================================================

/// Closed-class biographical / taxonomic hypernym recall shape detector:
/// matches WH-interrogatives (`what`/`which`/`who`/`whose`) paired with an abstract
/// WordNet hypernym noun (`desc_count >= 12`), zero open-class word lists.
fn is_biographical_hypernym_shape(tokens: &[&str]) -> bool {
    if tokens.len() < 3 {
        return false;
    }
    let first = clean_word_lower(tokens[0]);
    let has_wh = matches!(first.as_str(), "what" | "which" | "who" | "whose")
        || tokens.iter().take(3).any(|w| {
            let low = clean_word_lower(w);
            matches!(low.as_str(), "what" | "which" | "who" | "whose")
        });
    if !has_wh {
        return false;
    }
    tokens.iter().any(|&w| {
        let low = clean_word_lower(w);
        low.chars().count() >= 4
            && is_content(&low)
            && crate::wordnet::specific_ancestor_synsets(&low, 0, 2_000)
                .first()
                .is_some_and(|&(_, dc)| dc >= 12)
    })
}

/// Extract the lowest-DF exact non-stopword query group bit (`star_bit`, Owner Directive).
/// Prefers non-speaker content terms when named speakers are present.
fn lowest_df_star_bit(ctx: &RankCtx<'_>, speaker_lowers: &[String]) -> u64 {
    let stop = stopwords();
    let mut best_non_sp: Option<(u32, usize)> = None;
    let mut best_any: Option<(u32, usize)> = None;
    for qt in &ctx.plan.terms {
        let Some(gi) = qt.group else {
            continue;
        };
        if !qt.exact_hit || gi >= 64 {
            continue;
        }
        let df = ctx.ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
        if df == 0 {
            continue;
        }
        let Some(name) = ctx.ix.vocab.names.get(qt.id as usize) else {
            continue;
        };
        if name.starts_with('\u{1}') || name.chars().count() < 3 || stop.contains(name.as_str()) {
            continue;
        }
        let is_sp = speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(name));
        if best_any.is_none_or(|(bdf, bg)| df < bdf || (df == bdf && gi < bg)) {
            best_any = Some((df, gi));
        }
        if !is_sp && best_non_sp.is_none_or(|(bdf, bg)| df < bdf || (df == bdf && gi < bg)) {
            best_non_sp = Some((df, gi));
        }
    }
    best_non_sp
        .or(best_any)
        .map_or(0, |(_, gi)| 1u64 << gi)
}

/// Extract up to `cap` cleanest content words of turn `ti`, prioritized by lowest corpus DF.
fn turn_rare_content_words(ix: &Index, ti: usize, cap: usize) -> Vec<String> {
    let tr = &ix.turns[ti];
    let mut scored: Vec<(u32, String)> = Vec::new();
    let stop = stopwords();
    for &cid in &tr.content {
        let Some(name) = ix.vocab.names.get(cid as usize) else {
            continue;
        };
        if name.starts_with('\u{1}')
            || name.chars().count() < 3
            || stop.contains(name.as_str())
            || !is_content(name)
        {
            continue;
        }
        let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(u32::MAX);
        scored.push((df, name.clone()));
    }
    scored.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.truncate(cap);
    scored.into_iter().map(|(_, w)| w).collect()
}

/// Deficit-gated Reverse-Dictionary + Push-PPR + Harmonic Laplacian + Intrinsic-IC ESA +
/// `SAG2` bridge pass over `RankCtx`.
///
/// Activates ONLY when the query exhibits lexical deficit (incomplete exact group coverage,
/// zero-overlap candidates, ungrounded rare OOV, inferential/open-domain shape, or biographical
/// hypernym recall). Strictly enforces:
/// - Additive bridge updates bounded to `<= 0.35 * max_bm25` whenever `star_bit` is covered.
/// - Zero promotion of question or backchannel turns.
/// - Never displaces a declarative turn containing `star_bit` with one missing `star_bit`.
/// - Promotes proven symbolic bridges into Slots `2..=4` (or Slot `0` ONLY when Top-5 has
///   zero exact `star_bit` coverage and zero exact lexical overlap).
pub(in crate::retrieval) fn apply_revdict_harmonic_bridge(ctx: &mut RankCtx<'_>) {
    if ctx.nt == 0
        || ctx.ns == 0
        || ctx.final_turn_indices.is_empty()
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.sniper_how_many
    {
        return;
    }

    let ix = ctx.ix;
    let q_raw = raw_tokens(&ctx.q.text);
    let q_refs: Vec<&str> = q_raw.iter().map(String::as_str).collect();

    // Discover named speakers in the conversation so queries targeting a speaker stay aligned.
    let mut speaker_lowers: Vec<String> = Vec::new();
    for tr in ix.turns.iter().take(64) {
        if !tr.speaker.is_empty() && tr.speaker != "user" && tr.speaker != "assistant" {
            let sl = tr.speaker.to_ascii_lowercase();
            if !speaker_lowers.contains(&sl) {
                speaker_lowers.push(sl);
            }
        }
    }
    let queried_speakers: BTreeSet<String> = q_raw
        .iter()
        .filter(|w| speaker_lowers.iter().any(|sp| sp.eq_ignore_ascii_case(w)))
        .map(|w| w.to_ascii_lowercase())
        .collect();

    let q_group_cnt = ctx.plan.groups.len().min(64);
    let q_mask = if q_group_cnt == 64 {
        u64::MAX
    } else if q_group_cnt > 0 {
        (1u64 << q_group_cnt) - 1
    } else {
        0
    };
    let star_bit = lowest_df_star_bit(ctx, &speaker_lowers);

    let top5_has_star_bit = star_bit != 0
        && ctx.final_turn_indices.iter().take(5).any(|&ti| {
            let tr = &ix.turns[ti];
            !tr.is_question
                && !tr.is_backchannel
                && (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & star_bit) != 0
        });
    let top5_has_any_exact = q_mask != 0
        && ctx.final_turn_indices.iter().take(5).any(|&ti| {
            (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & q_mask) != 0
        });
    let top0_full_cov = q_mask != 0
        && ctx.final_turn_indices.first().is_some_and(|&h0| {
            let tr = &ix.turns[h0];
            !tr.is_question
                && !tr.is_backchannel
                && (ctx.t_exact_mask.get(h0).copied().unwrap_or(0) & q_mask) == q_mask
        });

    let is_inferential = is_open_domain_inferential_shape(&q_refs);
    let is_bio_hypernym = is_biographical_hypernym_shape(&q_refs);
    let has_deficit = ctx.deficit_a
        || ctx.deficit_b
        || ctx.has_ungrounded_rare_oov
        || !top5_has_star_bit
        || !top0_full_cov
        || is_inferential
        || is_bio_hypernym;
    if !has_deficit {
        return;
    }

    // Build filtered non-speaker query content words (`<= 10`).
    let q_words_owned: Vec<String> = collect_content_words(&q_refs, 12)
        .into_iter()
        .filter(|w| !queried_speakers.contains(w))
        .take(10)
        .collect();
    if q_words_owned.is_empty() {
        return;
    }
    let q_word_refs: Vec<&str> = q_words_owned.iter().map(String::as_str).collect();

    // 1. Reverse-Dictionary canonical headwords (`<= 8`).
    let rd_heads = reverse_dict_headwords(ix, &q_word_refs);
    let rd_map: DetHashMap<u32, i64> = rd_heads.iter().copied().collect();

    // 2. Local Push-PPR concepts mapped to corpus vocabulary IDs (`<= 20`).
    let ppr_terms = query_local_push_ppr(ix, &q_words_owned);
    let mut ppr_vid_map: DetHashMap<u32, i64> = DetHashMap::default();
    for (concept, mass_q16) in &ppr_terms {
        if let Some(vid) = ix.vocab.get(concept) {
            let e = ppr_vid_map.entry(vid).or_insert(0);
            *e = (*e).max(*mass_q16);
        }
    }

    // 3. Deterministic `SAG2` 4-signal sentence-anchored bridge terms from Top-10 seeds.
    let mut q_and_rd_ids: DetHashSet<u32> = DetHashSet::default();
    for qt in &ctx.plan.terms {
        if qt.exact_hit {
            q_and_rd_ids.insert(qt.id);
        }
    }
    for &(vid, _) in &rd_heads {
        q_and_rd_ids.insert(vid);
    }
    let mut sag2_seeds: Vec<(usize, i64)> = Vec::with_capacity(SAG2_MAX_SEEDS);
    for &ti in ctx.final_turn_indices.iter().take(SAG2_MAX_SEEDS) {
        let sc_q16 = ieee754_bits_to_q16(ctx.t_scene.get(ti).copied().unwrap_or(0.0).to_bits())
            .min(i64::MAX as u64) as i64;
        sag2_seeds.push((ti, sc_q16.max(ONE >> 4)));
    }
    let sag2_terms = sag2_extract_bridge_terms(ix, &sag2_seeds, &q_and_rd_ids, SAG2_TOP_L);

    // 4. Assemble bounded candidate pool: Top-24 sessions + SAG2/RD unseen declarative turns.
    let mut cand_sessions: Vec<usize> = Vec::with_capacity(MAX_CANDIDATE_SESSIONS);
    for &si in ctx.final_session_indices.iter().take(MAX_CANDIDATE_SESSIONS) {
        if !cand_sessions.contains(&si) {
            cand_sessions.push(si);
        }
    }
    for &(vid, _) in rd_heads.iter().take(6) {
        if let Some(post) = ix.s_postings.get(vid as usize) {
            for &si_u32 in post.iter().take(4) {
                let si = si_u32 as usize;
                if si < ctx.ns
                    && !cand_sessions.contains(&si)
                    && cand_sessions.len() < MAX_CANDIDATE_SESSIONS
                {
                    cand_sessions.push(si);
                }
            }
        }
    }

    let mut cand_turns: Vec<usize> = Vec::with_capacity(MAX_SCORED_TURNS);
    let mut cand_seen: DetHashSet<usize> = DetHashSet::default();
    let mut sag2_turn_bonus: DetHashMap<usize, i64> = DetHashMap::default();

    let speaker_ok = |ti: usize| -> bool {
        if queried_speakers.is_empty() {
            return true;
        }
        let tr = &ix.turns[ti];
        let sp = tr.speaker.to_ascii_lowercase();
        if queried_speakers.contains(&sp) {
            return true;
        }
        queried_speakers.iter().any(|qs| {
            ix.vocab
                .get(qs)
                .is_some_and(|vid| tr.content.contains(&vid) || tr.entities.contains(&vid))
        })
    };

    let nt_cap = ctx.nt;
    let push_turn = |ti: usize, list: &mut Vec<usize>, seen: &mut DetHashSet<usize>| {
        if ti >= nt_cap || list.len() >= MAX_SCORED_TURNS || !seen.insert(ti) {
            return;
        }
        let tr = &ix.turns[ti];
        if tr.is_question
            || tr.is_backchannel
            || turn_is_superseded(ix, ti)
            || !speaker_ok(ti)
        {
            return;
        }
        list.push(ti);
    };

    // Priority A: Unseen turns hitting Reverse-Dictionary canonical headwords.
    for &(vid, _) in &rd_heads {
        if let Some(post) = ix.t_postings.get(vid as usize) {
            for &ti_u32 in post.iter().take(4) {
                push_turn(ti_u32 as usize, &mut cand_turns, &mut cand_seen);
            }
        }
    }
    // Priority B: Unseen turns pulled by `SAG2` bridge terms (with reciprocal rank `1 / (60 + r)`).
    for (rank_idx, &(vid, a_r)) in sag2_terms.iter().enumerate() {
        let rr_q16 = ONE / (60 + rank_idx as i64);
        let term_sc = (a_r >> 2).saturating_add(rr_q16).min(ONE);
        if let Some(post) = ix.t_postings.get(vid as usize) {
            for &ti_u32 in post.iter().take(4) {
                let ti = ti_u32 as usize;
                if ti < ctx.nt {
                    let e = sag2_turn_bonus.entry(ti).or_insert(0);
                    *e = e.saturating_add(term_sc).min(ONE * 2);
                    push_turn(ti, &mut cand_turns, &mut cand_seen);
                }
            }
        }
    }
    // Priority C: Top declarative turns from the Top-24 candidate sessions.
    for &si in &cand_sessions {
        let mut sess_decl: Vec<(usize, u64)> = ix.session_turns[si]
            .iter()
            .copied()
            .filter(|&ti| {
                let tr = &ix.turns[ti];
                !tr.is_question && !tr.is_backchannel && !turn_is_superseded(ix, ti) && speaker_ok(ti)
            })
            .map(|ti| {
                let sc = ieee754_bits_to_q16(ctx.t_scene.get(ti).copied().unwrap_or(0.0).to_bits());
                (ti, sc)
            })
            .collect();
        sess_decl.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for &(ti, _) in sess_decl.iter().take(3) {
            push_turn(ti, &mut cand_turns, &mut cand_seen);
        }
    }

    if cand_turns.is_empty() {
        return;
    }

    // 5. Precompute query word profiles ONCE and cache candidate word profiles across turns.
    let q_profiles: Vec<WordSymbolicProfile> = q_words_owned
        .iter()
        .map(|w| build_word_profile(w))
        .collect();
    let q_prof_refs: Vec<&WordSymbolicProfile> = q_profiles.iter().collect();
    let mut word_cache: DetHashMap<String, WordSymbolicProfile> = DetHashMap::default();

    struct TurnBridgeEval {
        ti: usize,
        si: usize,
        pre_q16: i64,
        total_q16: i64,
        turn_words: Vec<String>,
        has_star: bool,
    }

    let mut evals: Vec<TurnBridgeEval> = Vec::with_capacity(cand_turns.len());
    for &ti in &cand_turns {
        let tr = &ix.turns[ti];
        let mut rd_sc: i64 = 0;
        let mut ppr_sc: i64 = 0;
        for &cid in &tr.content {
            if let Some(&v) = rd_map.get(&cid) {
                rd_sc = rd_sc.saturating_add(v);
            }
            if let Some(&v) = ppr_vid_map.get(&cid) {
                ppr_sc = ppr_sc.saturating_add(v);
            }
        }
        rd_sc = rd_sc.min(ONE * 2);
        ppr_sc = ppr_sc.min(ONE * 2);
        let sag2_sc = sag2_turn_bonus.get(&ti).copied().unwrap_or(0).min(ONE * 2);

        let turn_words = turn_rare_content_words(ix, ti, 12);
        for w in &turn_words {
            if !word_cache.contains_key(w) {
                word_cache.insert(w.clone(), build_word_profile(w));
            }
        }
        let d_refs: Vec<&WordSymbolicProfile> = turn_words
            .iter()
            .filter_map(|w| word_cache.get(w))
            .collect();
        let bma_q16 = score_profiles_bma_q16(&q_prof_refs, &d_refs);

        let pre_q16 = (rd_sc >> 1)
            .saturating_add(ppr_sc >> 2)
            .saturating_add(sag2_sc >> 2)
            .saturating_add(bma_q16);
        let has_star =
            star_bit != 0 && (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & star_bit) != 0;

        if pre_q16 > 0 {
            evals.push(TurnBridgeEval {
                ti,
                si: tr.session,
                pre_q16,
                total_q16: pre_q16,
                turn_words,
                has_star,
            });
        }
    }

    if evals.is_empty() {
        return;
    }

    // Refine top-16 candidates with full 16-sweep Harmonic Laplacian Voltage Relaxation.
    evals.sort_unstable_by(|a, b| b.pre_q16.cmp(&a.pre_q16).then_with(|| a.ti.cmp(&b.ti)));
    for ev in evals.iter_mut().take(MAX_HARMONIC_TURNS) {
        let d_refs: Vec<&WordSymbolicProfile> = ev
            .turn_words
            .iter()
            .filter_map(|w| word_cache.get(w))
            .collect();
        let harm_q16 = local_harmonic_bridge_q16(&q_prof_refs, &d_refs);
        ev.total_q16 = ev.pre_q16.saturating_add(q16_mul(harm_q16, 26_214));
    }

    // Final deterministic ranking of proven symbolic bridge candidates.
    evals.sort_unstable_by(|a, b| {
        b.total_q16
            .cmp(&a.total_q16)
            .then_with(|| a.ti.cmp(&b.ti))
    });

    // 6. Owner-Directive Bounded Score Updates (`<= 0.35 * max_bm25` when `star_bit` is covered).
    let max_t_bm25_q16 = ieee754_bits_to_q16(ctx.lexical_tmax.max(1.0).to_bits())
        .min(i64::MAX as u64) as i64;
    let max_s_bm25_q16 = ieee754_bits_to_q16(ctx.smax.max(1.0).to_bits())
        .min(i64::MAX as u64) as i64;
    let cap_ratio_q16 = if top5_has_star_bit || top5_has_any_exact {
        STAR_COVERED_CAP_Q16
    } else {
        ZERO_OVERLAP_CAP_Q16
    };
    let turn_add_cap_q16 = q16_mul(max_t_bm25_q16, cap_ratio_q16);
    let sess_add_cap_q16 = q16_mul(max_s_bm25_q16, cap_ratio_q16);
    let best_total = evals[0].total_q16.max(1);

    let mut sess_best_q16: BTreeMap<usize, i64> = BTreeMap::new();
    for ev in &evals {
        if ev.total_q16 < BRIDGE_MIN_PROOF_Q16 {
            continue;
        }
        let norm_q16 = q16_div(ev.total_q16, best_total).clamp(0, ONE);
        let add_t_q16 = q16_mul(turn_add_cap_q16, norm_q16).min(turn_add_cap_q16);
        if ev.ti < ctx.t_bridge.len() && add_t_q16 > 0 {
            ctx.t_bridge[ev.ti] += (add_t_q16 as f32) / (ONE as f32);
        }
        let e = sess_best_q16.entry(ev.si).or_insert(0);
        *e = (*e).max(norm_q16);
    }
    for (si, norm_q16) in sess_best_q16 {
        let add_s_q16 = q16_mul(sess_add_cap_q16, norm_q16).min(sess_add_cap_q16);
        if si < ctx.bridge_boost.len() && add_s_q16 > 0 {
            ctx.bridge_boost[si] += (add_s_q16 as f32) / (ONE as f32);
        }
    }

    // 7. Safe Slot Seating respecting `star_bit` Owner Directive & Torrens/Implicit Locks.
    let slot0_protected = ctx.torrens_locked_head
        || ctx.implicit_disclosure_locked
        || top5_has_star_bit
        || top5_has_any_exact;

    // Case A: Top-5 has ZERO exact `star_bit` coverage AND zero exact lexical overlap ->
    // allow the strongest proven symbolic bridge into Slot 0 (with session sync).
    if !slot0_protected
        && let Some(top_ev) = evals.first()
        && top_ev.total_q16 >= STRONG_SLOT0_PROOF_Q16
    {
        if let Some(pos) = ctx
            .final_turn_indices
            .iter()
            .position(|&x| x == top_ev.ti)
        {
            ctx.final_turn_indices.remove(pos);
        }
        ctx.final_turn_indices.insert(0, top_ev.ti);
        if ctx.final_turn_indices.len() > ctx.k {
            ctx.final_turn_indices.truncate(ctx.k);
        }
        if let Some(spos) = ctx
            .final_session_indices
            .iter()
            .position(|&s| s == top_ev.si)
        {
            ctx.final_session_indices.remove(spos);
        }
        ctx.final_session_indices.insert(0, top_ev.si);
    }

    // Case B: Promote up to 2 unseated proven symbolic bridges into Slots 2..=4 without ever
    // touching Slots 0..=1 or evicting a declarative `star_bit` turn with a non-`star_bit` turn.
    let top_window = ctx.k.min(5);
    let mut seated = 0usize;
    if top_window >= 3 {
        for ev in &evals {
            if seated >= 2 || ev.total_q16 < BRIDGE_MIN_PROOF_Q16 {
                break;
            }
            let top5_t_end = ctx.final_turn_indices.len().min(top_window);
            if ctx.final_turn_indices[..top5_t_end].contains(&ev.ti) {
                continue;
            }
            if ctx.final_turn_indices.len() < top_window {
                ctx.final_turn_indices.push(ev.ti);
            } else {
                // Pick the lowest-priority evictable slot in 2..top_window (never evict a declarative
                // `star_bit` turn unless `ev` also carries `star_bit`).
                let victim_slot = (2..top_window).rev().find(|&slot| {
                    let cur_ti = ctx.final_turn_indices[slot];
                    let cur_tr = &ix.turns[cur_ti];
                    let cur_decl_star = star_bit != 0
                        && !cur_tr.is_question
                        && !cur_tr.is_backchannel
                        && (ctx.t_exact_mask.get(cur_ti).copied().unwrap_or(0) & star_bit) != 0;
                    !cur_decl_star || ev.has_star
                });
                let Some(slot) = victim_slot else {
                    continue;
                };
                if let Some(pos) = ctx.final_turn_indices.iter().position(|&x| x == ev.ti) {
                    ctx.final_turn_indices.remove(pos);
                }
                let safe_slot = slot.min(ctx.final_turn_indices.len());
                ctx.final_turn_indices.insert(safe_slot, ev.ti);
            }
            if ctx.final_turn_indices.len() > ctx.k {
                ctx.final_turn_indices.truncate(ctx.k);
            }

            // Synchronize session into Slots 2..top_window without evicting `star_bit` sessions.
            let top5_s_end = ctx.final_session_indices.len().min(top_window);
            if !ctx.final_session_indices[..top5_s_end].contains(&ev.si) {
                if ctx.final_session_indices.len() < top_window {
                    ctx.final_session_indices.push(ev.si);
                } else if let Some(s_slot) = (2..top_window).rev().find(|&slot| {
                    let cur_si = ctx.final_session_indices[slot];
                    let cur_has_star = star_bit != 0
                        && (ctx.s_exact_mask.get(cur_si).copied().unwrap_or(0) & star_bit) != 0;
                    !cur_has_star || ev.has_star
                }) {
                    if let Some(pos) = ctx.final_session_indices.iter().position(|&s| s == ev.si) {
                        ctx.final_session_indices.remove(pos);
                    }
                    let safe_slot = s_slot.min(ctx.final_session_indices.len());
                    ctx.final_session_indices.insert(safe_slot, ev.si);
                }
            }
            seated += 1;
        }
    }

    if ctx.final_session_indices.len() > ctx.k.max(5) {
        ctx.final_session_indices.truncate(ctx.k.max(5));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::graph::ghostcurrent_mdl_q16::GhostCurrentResult;
    use crate::retrieval::lexical::bm25f::Fields;
    use crate::retrieval::{Params, Query, QueryIntent, Session, Turn, build_index, plan_query};

    fn make_session(id: &str, turns: &[(&str, &str, &str)]) -> Session {
        Session {
            id: id.to_string(),
            date: "2024/03/10 (Sun) 10:00".to_string(),
            timestamp: 1_710_064_800,
            turns: turns
                .iter()
                .map(|&(tid, role, content)| Turn {
                    id: tid.to_string(),
                    role: role.to_string(),
                    content: content.to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn revdict_prunes_hubs_over_256_backlinks_and_enforces_conjunction() {
        // Synthetic Thorat & Choudhari test:
        // Token 1 (hub: 300 backlinks) points to headwords 100..400.
        // Tokens 2 & 3 (rare: 2 backlinks each) intersect on headword 105.
        let hub_targets: Vec<u32> = (100..400).collect();
        let mut backlink_counts: BTreeMap<u32, usize> = BTreeMap::new();
        backlink_counts.insert(1, 300);
        backlink_counts.insert(2, 2);
        backlink_counts.insert(3, 2);

        let mut rev_gloss: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        rev_gloss.insert(1, hub_targets);
        rev_gloss.insert(2, vec![105, 901]);
        rev_gloss.insert(3, vec![105, 902]);

        let fwd_hop1: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        let mut ic_map: BTreeMap<u32, i64> = BTreeMap::new();
        ic_map.insert(1, ONE >> 2);
        ic_map.insert(2, ONE);
        ic_map.insert(3, ONE);

        // Hub-only query must emit zero candidates (cannot seed > 256 backlink hub).
        let hub_only = reverse_dict_expand_core(
            &[1],
            &backlink_counts,
            &rev_gloss,
            &fwd_hop1,
            &ic_map,
            RevDictExpandCfg {
                min_hits: 2,
                top_k: RD_TOP_K,
            },
        );
        assert!(
            hub_only.is_empty(),
            "token with >256 backlinks must never seed candidates: {hub_only:?}"
        );

        // Conjunction of two rare seeds + hub verifier: ONLY headword 105 passes (`hits >= 2`),
        // 901 & 902 are rejected by clause conjunction, and hub-only 100..399 never leak in.
        let res = reverse_dict_expand_core(
            &[1, 2, 3],
            &backlink_counts,
            &rev_gloss,
            &fwd_hop1,
            &ic_map,
            RevDictExpandCfg {
                min_hits: 2,
                top_k: RD_TOP_K,
            },
        );
        assert_eq!(res.len(), 1, "expected single conjunctive winner: {res:?}");
        assert_eq!(res[0].0, 105);
        assert!(res[0].1 > ONE, "conjunction multiplier should lift Sim_RD");

        // Verify real WordNet reverse_dict_headwords is deterministic on a corpus index.
        let sessions = vec![make_session(
            "s0",
            &[
                (
                    "t0",
                    "user",
                    "My husband and I celebrated our anniversary with a quiet dinner.",
                ),
                (
                    "t1",
                    "user",
                    "I visited a local bakery for fresh sourdough bread.",
                ),
            ],
        )];
        let ix = build_index(&sessions, &Params::default());
        let r1 = reverse_dict_headwords(&ix, &["married", "partner", "spouse"]);
        let r2 = reverse_dict_headwords(&ix, &["married", "partner", "spouse"]);
        assert_eq!(r1, r2, "reverse_dict_headwords must be 100% deterministic");
    }

    #[test]
    fn push_ppr_degree_normalized_convergence_and_hub_sink_containment() {
        let mut g = LocalPprGraph::new(16);
        let q = g.intern("query_seed", 4, 16).unwrap();
        let specific_mid = g.intern("specific_mid", 4, 16).unwrap();
        let specific_dst = g.intern("specific_dst", 4, 16).unwrap();
        let mega_hub = g.intern("mega_hub", 4_096, 16).unwrap();
        let hub_dst = g.intern("hub_dst", 4, 16).unwrap();

        let mut raw_edges: BTreeMap<(u32, u32), i64> = BTreeMap::new();
        raw_edges.insert((q, specific_mid), ONE);
        raw_edges.insert((specific_mid, specific_dst), ONE);
        raw_edges.insert((q, mega_hub), ONE);
        raw_edges.insert((mega_hub, hub_dst), ONE);
        g.finalize_symmetric_transitions(&raw_edges);

        let ppr = sparse_push_ppr_q16(&g, &[(q, ONE)], PPR_EPS_Q16, PPR_MAX_PUSHES);
        let mass_of = |id: u32| ppr.iter().find(|&&(u, _)| u == id).map_or(0, |&(_, m)| m);

        let m_spec = mass_of(specific_dst);
        let m_hub_dst = mass_of(hub_dst);
        assert!(
            m_spec > 0,
            "specific 2-hop target must receive positive Push-PPR mass"
        );
        assert_eq!(
            m_hub_dst, 0,
            "hub-sink (deg > 2048) must absorb mass and never emit to hub_dst"
        );
        assert!(
            mass_of(specific_mid) > mass_of(mega_hub),
            "1/sqrt(deg_u * deg_v) scaling must favor low-degree bridge over mega-hub"
        );
    }

    #[test]
    fn harmonic_laplacian_sums_parallel_clues_and_attenuates_hubs() {
        // Single clue vs two parallel clues feeding the same target node.
        let mut g1 = HarmonicSubgraph::new(3);
        g1.add_undirected_edge(0, 2, HALF);
        let v_single = harmonic_affinity_q16(&g1, &[(0, ONE)], &[2], HARMONIC_GAMMA_Q16);

        let mut g2 = HarmonicSubgraph::new(3);
        g2.add_undirected_edge(0, 2, HALF);
        g2.add_undirected_edge(1, 2, HALF);
        let v_parallel =
            harmonic_affinity_q16(&g2, &[(0, ONE), (1, ONE)], &[2], HARMONIC_GAMMA_Q16);

        assert!(
            v_parallel > v_single,
            "parallel voltage sources ({v_parallel}) must strictly beat single clue ({v_single})"
        );

        // Kirchhoff hub-shunting test: a hub connected to 32 grounded/unpinned branches loses voltage.
        let mut g_hub = HarmonicSubgraph::new(35);
        g_hub.add_undirected_edge(0, 1, HALF); // query -> hub
        g_hub.add_undirected_edge(1, 2, HALF); // hub -> target
        for branch in 3..35 {
            g_hub.add_undirected_edge(1, branch, HALF);
        }
        let v_via_hub = harmonic_affinity_q16(&g_hub, &[(0, ONE)], &[2], HARMONIC_GAMMA_Q16);
        assert!(
            v_single > v_via_hub,
            "direct specific edge ({v_single}) must beat high-degree hub shunt ({v_via_hub})"
        );
    }

    #[test]
    fn sag2_four_signal_prioritizes_sentence_grounded_bridge_terms() {
        let sessions = vec![
            make_session(
                "s0",
                &[(
                    "t0",
                    "user",
                    "We planned a camping trip to Yosemite last summer. The wallpaper in the hallway peeled.",
                )],
            ),
            make_session(
                "s1",
                &[(
                    "t1",
                    "user",
                    "My friend loves hiking trails near Yosemite valley.",
                )],
            ),
            make_session(
                "s2",
                &[(
                    "t2",
                    "user",
                    "I bought groceries and cooked pasta for dinner.",
                )],
            ),
        ];
        let ix = build_index(&sessions, &Params::default());
        let mut q_ids: DetHashSet<u32> = DetHashSet::default();
        for w in ["camping", "trip"] {
            if let Some(vid) = ix.vocab.get(w) {
                q_ids.insert(vid);
            }
        }
        let terms = sag2_extract_bridge_terms(&ix, &[(0, ONE)], &q_ids, SAG2_TOP_L);
        let yosemite_id = ix.vocab.get("yosemite").expect("yosemite indexed");
        let wallpaper_id = ix.vocab.get("wallpaper").expect("wallpaper indexed");

        let yosemite_sc = terms
            .iter()
            .find(|&&(id, _)| id == yosemite_id)
            .map_or(0, |&(_, s)| s);
        let wallpaper_sc = terms
            .iter()
            .find(|&&(id, _)| id == wallpaper_id)
            .map_or(0, |&(_, s)| s);
        assert!(
            yosemite_sc > wallpaper_sc,
            "sentence-grounded bridge 'yosemite' ({yosemite_sc}) must outrank ungrounded 'wallpaper' ({wallpaper_sc})"
        );
    }

    #[test]
    fn utf8_safety_and_star_bit_slot0_protection() {
        // 1. Multibyte UTF-8 safety across all public functions (zero panics).
        let utf8_q = ["café", "résumé", "naïve", "東京", "über", "baking"];
        let utf8_d = ["bakery", "café", "pâtisserie", "東京", "sourdough"];
        let sc = score_turn_harmonic_esa_q16(&utf8_q, &utf8_d);
        assert!(sc > 0, "expected positive harmonic-ESA score on UTF-8 inputs");

        // 2. Owner Directive `star_bit` Slot-0 protection in `apply_revdict_harmonic_bridge`:
        // Slot 0 holds a declarative turn containing the lowest-DF exact query noun ("quokka").
        // A bridged turn missing "quokka" MUST NEVER displace Slot 0!
        let sessions = vec![
            make_session(
                "s_exact",
                &[(
                    "t_exact",
                    "user",
                    "I saw a tiny quokka on Rottnest Island during my bakery tour.",
                )],
            ),
            make_session(
                "s_bridge",
                &[(
                    "t_bridge",
                    "user",
                    "I visited a wonderful café and bakery that makes fresh pastries and bread.",
                )],
            ),
            make_session(
                "s_noise1",
                &[(
                    "t_noise1",
                    "user",
                    "Did we schedule the meeting for Tuesday afternoon?",
                )],
            ),
            make_session(
                "s_noise2",
                &[("t_noise2", "user", "Okay sounds good, thanks!")],
            ),
            make_session(
                "s_noise3",
                &[("t_noise3", "user", "Let me check the calendar later.")],
            ),
        ];
        let params = Params::default();
        let ix = build_index(&sessions, &params);
        let _ = reverse_dict_headwords(&ix, &["café", "résumé", "東京", "baking"]);
        let q = Query {
            text: "Would I enjoy baking near a quokka?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &params);
        let quokka_group = plan
            .terms
            .iter()
            .find(|qt| ix.vocab.names.get(qt.id as usize).map(String::as_str) == Some("quokka"))
            .and_then(|qt| qt.group)
            .expect("quokka exact group");
        let quokka_bit = 1u64 << quokka_group;
        let ns = ix.session_turns.len();
        let nt = ix.turns.len();
        let mut t_exact_mask = vec![0u64; nt];
        let mut s_exact_mask = vec![0u64; ns];
        t_exact_mask[0] = quokka_bit;
        s_exact_mask[0] = quokka_bit;

        let mut ctx = RankCtx {
            ix: &ix,
            q: &q,
            p: &params,
            k: 5,
            ns,
            nt,
            plan,
            pref_on: false,
            persona_infer_on: false,
            episodic_user_quote: false,
            user_role: false,
            fl: Fields {
                w: [1.0, 1.0, 0.75],
                b: [0.75, 0.75, 0.0],
            },
            s_mask: s_exact_mask.clone(),
            s_exact_mask,
            s_score: vec![10.0, 1.0, 0.5, 0.2, 0.1],
            smax: 10.0,
            t_own: vec![10.0, 1.0, 0.5, 0.2, 0.1],
            t_raw: vec![10.0, 1.0, 0.5, 0.2, 0.1],
            t_turn_raw: vec![10.0, 1.0, 0.5, 0.2, 0.1],
            t_mask: t_exact_mask.clone(),
            t_exact_mask,
            t_scene: vec![10.0, 1.0, 0.5, 0.2, 0.1],
            t_scene_sess: vec![10.0, 1.0, 0.5, 0.2, 0.1],
            q_toks_eff: raw_tokens(&q.text),
            past_cue_eff: false,
            seq_mode_eff: false,
            has_named_speakers_eff: false,
            is_monologue_corpus: true,
            has_compact_rule_register: false,
            cond_rule_turns: Vec::new(),
            eff_it: QueryIntent::default(),
            t_bridge: vec![0.0; nt],
            t_bridge_hits: vec![0; nt],
            lexical_smax: 10.0,
            lexical_tmax_sess: 10.0,
            lexical_tmax: 10.0,
            standing_rules: Vec::new(),
            s_policy_floored: vec![false; ns],
            t_policy_floored: vec![false; nt],
            s_pref_floored: vec![false; ns],
            t_pref_floored: vec![false; nt],
            initiative_content: Vec::new(),
            bridge_boost: vec![0.0; ns],
            s_ord_base: vec![0, 1, 2, 3, 4],
            total_q_groups: 2,
            mask_limit: 0b11,
            final_session_indices: vec![0, 2, 3, 4, 1],
            strong_top_sessions: 1,
            max_per_session_top5: 2,
            turn_ts: vec![0; nt],
            final_turn_indices: vec![0, 2, 3, 4, 1],
            base_q16: vec![65_536; nt],
            q_group_cnt: 2,
            q_group_mask: 0b11,
            grounded_group_mask: quokka_bit,
            top3_union_exact: quokka_bit,
            has_ungrounded_rare_oov: true,
            deficit_a: false,
            deficit_b: true,
            torrens_locked_head: false,
            implicit_disclosure_locked: false,
            ghostcurrent: GhostCurrentResult::default(),
            wf_g1_s10: Vec::new(),
            wf_g3_t10: Vec::new(),
            wf_g4_s5: Vec::new(),
            wf_g4_flags: Vec::new(),
        };

        apply_revdict_harmonic_bridge(&mut ctx);
        assert_eq!(
            ctx.final_turn_indices[0], 0,
            "Slot 0 declarative star_bit winner MUST NEVER be displaced by bridged turn"
        );
        assert!(
            ctx.t_bridge[1] <= 3.51,
            "additive bridge boost ({}) must stay <= 0.35 * max_bm25 (3.5)",
            ctx.t_bridge[1]
        );
    }

    #[test]
    fn intrinsic_ic_lin_rejects_root_hubs_and_bridges_concrete_to_abstract() {
        let low_desc = seco_ic_q16(2);
        let root_hub = seco_ic_q16(80_000);
        assert!(
            low_desc > root_hub && root_hub <= HUB_IC_FLOOR_Q16,
            "Seco IC must assign near-zero IC to root hubs ({root_hub}) vs leaf ({low_desc})"
        );
        let concrete_vs_abstract =
            score_turn_harmonic_esa_q16(&["musical", "instrument"], &["violin", "concert"]);
        let unrelated =
            score_turn_harmonic_esa_q16(&["musical", "instrument"], &["refrigerator", "plumbing"]);
        assert!(
            concrete_vs_abstract > unrelated,
            "4-hop upward + Lin/ESA bridge ({concrete_vs_abstract}) must beat unrelated turn ({unrelated})"
        );
    }
}
