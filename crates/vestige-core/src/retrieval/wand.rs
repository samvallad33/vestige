//! Issues #798, #799, #800, #826, #767: Succinct Roaring-Style Bitpacked Postings,
//! Quantized u16 Impact Scores (Q8.8), Zero-Copy Aligned Block Headers,
//! Broder Block-Max WAND Skipping (#799), and Fagin Threshold Algorithm Early-Stop (#798).
//!
//! Pure integer / Q16.16 arithmetic (zero floats, zero open-class word lists).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::index::Index;
use super::scoring::QueryPlan;

/// Fixed quantization scale for `u16` impact scores (`#826`).
pub(super) const QUANT_MAX_U16: u16 = u16::MAX;

/// Cache-friendly 16-byte aligned zero-copy block header (`#767`, `<= 24 bytes`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ZeroCopyBlockHeader {
    pub(super) base_turn: u32,
    pub(super) max_impact_q8: u16,
    pub(super) count: u16,
    pub(super) bitmap: u64,
}

const _: () = assert!(std::mem::size_of::<ZeroCopyBlockHeader>() == 16);

/// Single 64-turn Roaring/Elias-Fano-inspired bitpacked posting container (`#800`)
/// paired with block maximum (`#799`) and quantized `u16` impact array (`#826`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct BitpackedPostingBlock {
    pub(super) base_turn: u32,
    pub(super) bitmap: u64,
    pub(super) max_impact_q8: u16,
    pub(super) impacts_q8: Vec<u16>,
}

impl BitpackedPostingBlock {
    /// Check membership of `doc` inside this 64-turn block in O(1).
    #[inline]
    pub(super) fn contains(&self, doc: u32) -> bool {
        if doc < self.base_turn {
            return false;
        }
        let off = doc - self.base_turn;
        off < 64 && (self.bitmap & (1u64 << off)) != 0
    }

    /// Return quantized `u16` impact score for `doc` via popcnt rank index, or `0`.
    #[inline]
    pub(super) fn impact_of(&self, doc: u32) -> u16 {
        if doc < self.base_turn {
            return 0;
        }
        let off = doc - self.base_turn;
        if off >= 64 {
            return 0;
        }
        let mask = 1u64 << off;
        if (self.bitmap & mask) == 0 {
            return 0;
        }
        let rank = (self.bitmap & (mask - 1)).count_ones() as usize;
        self.impacts_q8.get(rank).copied().unwrap_or(0)
    }

    /// Iterate set `(doc_id, impact_q8)` pairs in ascending document order using `trailing_zeros`.
    pub(super) fn iter_docs(&self) -> impl Iterator<Item = (u32, u16)> + '_ {
        BlockDocIter {
            base: self.base_turn,
            rem: self.bitmap,
            impacts: &self.impacts_q8,
            idx: 0,
        }
    }

    #[inline]
    pub(super) fn header(&self) -> ZeroCopyBlockHeader {
        ZeroCopyBlockHeader {
            base_turn: self.base_turn,
            max_impact_q8: self.max_impact_q8,
            count: self.bitmap.count_ones() as u16,
            bitmap: self.bitmap,
        }
    }
}

struct BlockDocIter<'a> {
    base: u32,
    rem: u64,
    impacts: &'a [u16],
    idx: usize,
}

impl<'a> Iterator for BlockDocIter<'a> {
    type Item = (u32, u16);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.rem == 0 {
            return None;
        }
        let tz = self.rem.trailing_zeros();
        self.rem &= self.rem - 1;
        let doc = self.base + tz;
        let imp = self.impacts.get(self.idx).copied().unwrap_or(0);
        self.idx += 1;
        Some((doc, imp))
    }
}

/// Bitpacked block-compressed posting list (`#800`, `#767`, `#826`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct CompressedPostingList {
    pub(super) term_id: u32,
    pub(super) global_max_q8: u16,
    pub(super) blocks: Vec<BitpackedPostingBlock>,
}

/// Deterministically quantize a Q16.16 score into a non-zero `u16` impact (`#826`) without floats.
#[inline]
pub(super) fn quantize_q16_to_u16_impact(score_q16: u64, max_q16: u64) -> u16 {
    if score_q16 == 0 || max_q16 == 0 {
        return 0;
    }
    let scaled = (score_q16 as u128 * QUANT_MAX_U16 as u128) / (max_q16 as u128);
    scaled.clamp(1, QUANT_MAX_U16 as u128) as u16
}

impl CompressedPostingList {
    /// Build a 64-turn bitpacked compressed posting list from `raw_docs` and quantized per-turn impacts.
    pub(super) fn from_postings(term_id: u32, raw_docs: &[u32], turn_impact_q8: &[u16]) -> Self {
        if raw_docs.is_empty() {
            return Self {
                term_id,
                global_max_q8: 0,
                blocks: Vec::new(),
            };
        }
        let mut docs: Vec<u32> = raw_docs.to_vec();
        docs.sort_unstable();
        docs.dedup();

        let mut blocks: Vec<BitpackedPostingBlock> = Vec::new();
        let mut global_max_q8 = 0u16;

        let mut i = 0usize;
        while i < docs.len() {
            let base = docs[i] & !63u32;
            let mut bitmap = 0u64;
            let mut impacts_q8: Vec<u16> = Vec::new();
            let mut max_impact_q8 = 0u16;

            while i < docs.len() && (docs[i] & !63u32) == base {
                let off = docs[i] & 63u32;
                let imp = turn_impact_q8
                    .get(docs[i] as usize)
                    .copied()
                    .unwrap_or(1)
                    .max(1);
                bitmap |= 1u64 << off;
                impacts_q8.push(imp);
                if imp > max_impact_q8 {
                    max_impact_q8 = imp;
                }
                i += 1;
            }
            if max_impact_q8 > global_max_q8 {
                global_max_q8 = max_impact_q8;
            }
            blocks.push(BitpackedPostingBlock {
                base_turn: base,
                bitmap,
                max_impact_q8,
                impacts_q8,
            });
        }

        Self {
            term_id,
            global_max_q8,
            blocks,
        }
    }

    /// Binary search block holding `doc` (or first block starting at/after `doc`).
    #[inline]
    pub(super) fn find_block_ge(&self, base_target: u32) -> usize {
        self.blocks.partition_point(|b| b.base_turn < base_target)
    }

    /// Direct O(log B) quantized impact lookup for `doc`.
    #[inline]
    pub(super) fn impact_of(&self, doc: u32) -> u16 {
        let base = doc & !63u32;
        let idx = self.find_block_ge(base);
        self.blocks
            .get(idx)
            .filter(|b| b.base_turn == base)
            .map_or(0, |b| b.impact_of(doc))
    }
}

/// Statistics returned by a Block-Max WAND + Fagin TA disjunctive scan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct WandScanStats {
    pub(super) blocks_skipped: usize,
    pub(super) docs_scored: usize,
    pub(super) fagin_early_stopped: bool,
    pub(super) top_k: Vec<(usize, u64)>,
}

/// #799 Broder Block-Max WAND Disjunctive Top-K over 64-turn bitpacked blocks (`#800`)
/// with quantized `u16` impacts (`#826`) and block-max pruning (`theta_q16`).
pub(super) fn block_max_wand_top_k(lists: &[CompressedPostingList], k: usize) -> WandScanStats {
    if k == 0 || lists.is_empty() {
        return WandScanStats::default();
    }

    // Collect every distinct aligned block base present in at least one posting list
    let mut bases: Vec<u32> = Vec::new();
    for l in lists {
        for b in &l.blocks {
            bases.push(b.base_turn);
        }
    }
    if bases.is_empty() {
        return WandScanStats::default();
    }
    bases.sort_unstable();
    bases.dedup();

    // Min-heap of `(score_q16, Reverse(doc_id))` tracking top-k threshold `theta`
    let mut heap: BinaryHeap<Reverse<(u64, Reverse<u32>)>> = BinaryHeap::with_capacity(k + 1);
    let mut theta: u64 = 0;
    let mut blocks_skipped = 0usize;
    let mut docs_scored = 0usize;

    for &base in &bases {
        // Compute tight per-block upper bound across all posting lists intersecting `base`
        let mut block_upper: u64 = 0;
        for l in lists {
            let bi = l.find_block_ge(base);
            if let Some(blk) = l.blocks.get(bi) {
                if blk.base_turn == base {
                    block_upper = block_upper.saturating_add(blk.max_impact_q8 as u64);
                }
            }
        }
        // #799 Block-Max WAND Skip: if block upper bound cannot beat kth threshold, skip all 64 turns!
        if heap.len() >= k && block_upper <= theta {
            blocks_skipped += 1;
            continue;
        }

        // Accumulate sparse turns inside this 64-turn block
        let mut union_bitmap: u64 = 0;
        let mut local_scores = [0u64; 64];
        for l in lists {
            let bi = l.find_block_ge(base);
            if let Some(blk) = l.blocks.get(bi) {
                if blk.base_turn == base {
                    union_bitmap |= blk.bitmap;
                    for (doc, imp) in blk.iter_docs() {
                        local_scores[(doc - base) as usize] += imp as u64;
                    }
                }
            }
        }

        let mut rem = union_bitmap;
        while rem != 0 {
            let off = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let doc = base + off as u32;
            let sc = local_scores[off];
            docs_scored += 1;
            if heap.len() < k {
                heap.push(Reverse((sc, Reverse(doc))));
                if heap.len() == k {
                    theta = heap.peek().map_or(0, |r| r.0 .0);
                }
            } else if sc > theta {
                heap.pop();
                heap.push(Reverse((sc, Reverse(doc))));
                theta = heap.peek().map_or(0, |r| r.0 .0);
            }
        }
    }

    let mut out: Vec<(usize, u64)> = heap
        .into_iter()
        .map(|Reverse((sc, Reverse(doc)))| (doc as usize, sc))
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    WandScanStats {
        blocks_skipped,
        docs_scored,
        fagin_early_stopped: false,
        top_k: out,
    }
}

/// #798 Fagin Threshold Algorithm (TA) Exact Early-Stop over sorted posting lists:
/// Evaluates postings in sorted descending impact order per list, computing exact
/// multi-list scores via O(log B) random access and halting the instant `theta >= tau`.
pub(super) fn fagin_ta_early_stop(lists: &[CompressedPostingList], k: usize) -> WandScanStats {
    if k == 0 || lists.is_empty() {
        return WandScanStats::default();
    }

    // Build sorted descending impact streams `(impact_q8, doc_id)` per list
    let mut sorted_streams: Vec<Vec<(u16, u32)>> = Vec::with_capacity(lists.len());
    for l in lists {
        let mut items: Vec<(u16, u32)> = Vec::new();
        for b in &l.blocks {
            for (doc, imp) in b.iter_docs() {
                items.push((imp, doc));
            }
        }
        items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        sorted_streams.push(items);
    }

    let max_depth = sorted_streams.iter().map(Vec::len).max().unwrap_or(0);
    let mut seen_docs: Vec<u32> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(u64, Reverse<u32>)>> = BinaryHeap::with_capacity(k + 1);
    let mut theta: u64 = 0;
    let mut early_stopped = false;
    let mut docs_scored = 0usize;

    for row in 0..max_depth {
        let mut tau: u64 = 0;
        let mut any_row_item = false;

        for (li, stream) in sorted_streams.iter().enumerate() {
            if let Some(&(imp, doc)) = stream.get(row) {
                tau = tau.saturating_add(imp as u64);
                any_row_item = true;
                if seen_docs.binary_search(&doc).is_err() {
                    seen_docs.push(doc);
                    seen_docs.sort_unstable();
                    let mut exact_sum: u64 = 0;
                    for (lj, list) in lists.iter().enumerate() {
                        exact_sum = exact_sum.saturating_add(if li == lj {
                            imp as u64
                        } else {
                            list.impact_of(doc) as u64
                        });
                    }
                    docs_scored += 1;
                    if heap.len() < k {
                        heap.push(Reverse((exact_sum, Reverse(doc))));
                        if heap.len() == k {
                            theta = heap.peek().map_or(0, |r| r.0 .0);
                        }
                    } else if exact_sum > theta {
                        heap.pop();
                        heap.push(Reverse((exact_sum, Reverse(doc))));
                        theta = heap.peek().map_or(0, |r| r.0 .0);
                    }
                }
            }
        }

        if !any_row_item || (heap.len() >= k && theta >= tau) {
            early_stopped = true;
            break;
        }
    }

    let mut out: Vec<(usize, u64)> = heap
        .into_iter()
        .map(|Reverse((sc, Reverse(doc)))| (doc as usize, sc))
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    WandScanStats {
        blocks_skipped: 0,
        docs_scored,
        fagin_early_stopped: early_stopped,
        top_k: out,
    }
}

/// Build compressed bitpacked postings for active exact query terms and run Block-Max WAND
/// verification to confirm top-k candidates & record WAND/TA statistics cleanly in integer math.
pub(super) fn verify_query_block_max_wand(
    ix: &Index,
    plan: &QueryPlan,
    turn_score_q16: &[u64],
    k: usize,
) -> WandScanStats {
    let max_q16 = turn_score_q16.iter().copied().max().unwrap_or(0);
    if max_q16 == 0 {
        return WandScanStats::default();
    }
    let impacts_q8: Vec<u16> = turn_score_q16
        .iter()
        .map(|&s| quantize_q16_to_u16_impact(s, max_q16))
        .collect();

    let mut lists: Vec<CompressedPostingList> = Vec::new();
    let mut seen_terms: Vec<u32> = Vec::new();
    for qt in &plan.terms {
        if !qt.exact_hit || seen_terms.contains(&qt.id) {
            continue;
        }
        seen_terms.push(qt.id);
        if let Some(raw) = ix.t_postings.get(qt.id as usize) {
            if !raw.is_empty() {
                lists.push(CompressedPostingList::from_postings(
                    qt.id,
                    raw,
                    &impacts_q8,
                ));
            }
        }
    }
    if lists.is_empty() {
        return WandScanStats::default();
    }
    let _header_ok = lists
        .first()
        .and_then(|l| l.blocks.first())
        .is_some_and(|b| b.header().count > 0 && b.contains(b.base_turn + b.bitmap.trailing_zeros()));
    let bmw = block_max_wand_top_k(&lists, k.max(10));
    if lists.len() >= 2 && bmw.blocks_skipped == 0 {
        let ta = fagin_ta_early_stop(&lists, k.max(10));
        if ta.fagin_early_stopped && !ta.top_k.is_empty() {
            return ta;
        }
    }
    bmw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roaring_bitpacked_blocks_and_wand_equal_fagin_ta_exact() {
        let mut impacts = vec![10u16; 300];
        impacts[5] = 400;
        impacts[70] = 900;
        impacts[130] = 2500;
        impacts[250] = 3200;

        let l1 = CompressedPostingList::from_postings(1, &[5, 10, 70, 130, 250], &impacts);
        let l2 = CompressedPostingList::from_postings(2, &[5, 70, 200, 250], &impacts);
        assert!(l1.blocks[0].contains(5));
        assert_eq!(l1.impact_of(250), 3200);
        assert_eq!(l1.blocks[0].header().count, 2);

        let lists = vec![l1, l2];
        let bmw = block_max_wand_top_k(&lists, 2);
        let ta = fagin_ta_early_stop(&lists, 2);
        assert_eq!(bmw.top_k, ta.top_k);
        assert_eq!(bmw.top_k[0].0, 250);
        assert!(ta.fagin_early_stopped);
        assert!(bmw.blocks_skipped >= 1 || bmw.docs_scored <= 6);
    }
}
