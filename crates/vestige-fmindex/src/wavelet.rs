#![allow(unsafe_code)]

//! Compact `u64`-word BitVector (`rank1`, `rank0`, `select1`, `select0`) with a
//! `#[cfg(target_feature = "popcnt")]` fast path + identical safe scalar SWAR fallback,
//! and a 9-bit / configurable-bit Wavelet Matrix over `u16` symbols.

use serde::{Deserialize, Serialize};

use crate::error::FmError;

const SUPERBLOCK_BITS: usize = 512;
const WORDS_PER_SUPERBLOCK: usize = SUPERBLOCK_BITS / 64;
const SELECT_SAMPLE_STEP: usize = 256;

/// Safe scalar SWAR 64-bit popcount with identical semantics to `_popcnt64`.
#[inline]
pub const fn popcount_scalar(mut x: u64) -> u32 {
    x = x - ((x >> 1) & 0x5555_5555_5555_5555);
    x = (x & 0x3333_3333_3333_3333) + ((x >> 2) & 0x3333_3333_3333_3333);
    x = (x + (x >> 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    ((x.wrapping_mul(0x0101_0101_0101_0101)) >> 56) as u32
}

/// Hardware `popcnt` fast path when `target_feature = "popcnt"` is enabled on x86_64,
/// falling back cleanly to the identical safe SWAR scalar implementation elsewhere.
#[inline]
pub fn popcount_u64(word: u64) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "popcnt"))]
    {
        // SAFETY: Guarded by `#[cfg(all(target_arch = "x86_64", target_feature = "popcnt"))]`.
        unsafe { core::arch::x86_64::_popcnt64(word as i64) as u32 }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "popcnt")))]
    {
        popcount_scalar(word)
    }
}

/// Dense `u64`-word bitvector with $O(1)$ `rank1`/`rank0` and sampled `select1`/`select0`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitVector {
    len: usize,
    total_ones: usize,
    words: Vec<u64>,
    superblocks: Vec<u32>,
    select1_samples: Vec<u32>,
    select0_samples: Vec<u32>,
}

impl BitVector {
    pub fn from_bools(bits: &[bool]) -> Self {
        let len = bits.len();
        let word_count = len.div_ceil(64);
        let mut words = vec![0u64; word_count];

        for (i, &b) in bits.iter().enumerate() {
            if b {
                words[i >> 6] |= 1u64 << (i & 63);
            }
        }

        Self::from_words(words, len)
    }

    pub fn from_words(words: Vec<u64>, len: usize) -> Self {
        let sb_count = len.div_ceil(SUPERBLOCK_BITS) + 1;
        let mut superblocks = Vec::with_capacity(sb_count);
        let mut total_ones = 0usize;

        for (idx, &w) in words.iter().enumerate() {
            if idx % WORDS_PER_SUPERBLOCK == 0 {
                superblocks.push(total_ones as u32);
            }
            total_ones += popcount_u64(w) as usize;
        }
        while superblocks.len() < sb_count {
            superblocks.push(total_ones as u32);
        }

        let mut select1_samples = Vec::new();
        let mut select0_samples = Vec::new();
        let mut ones_seen = 0usize;
        let mut zeros_seen = 0usize;

        for (w_idx, &w) in words.iter().enumerate() {
            let valid_bits = (len.saturating_sub(w_idx * 64)).min(64);
            for b in 0..valid_bits {
                let bit_pos = w_idx * 64 + b;
                if ((w >> b) & 1) == 1 {
                    if ones_seen.is_multiple_of(SELECT_SAMPLE_STEP) {
                        select1_samples.push(bit_pos as u32);
                    }
                    ones_seen += 1;
                } else {
                    if zeros_seen.is_multiple_of(SELECT_SAMPLE_STEP) {
                        select0_samples.push(bit_pos as u32);
                    }
                    zeros_seen += 1;
                }
            }
        }

        Self {
            len,
            total_ones,
            words,
            superblocks,
            select1_samples,
            select0_samples,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn total_ones(&self) -> usize {
        self.total_ones
    }

    #[inline]
    pub fn total_zeros(&self) -> usize {
        self.len - self.total_ones
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn get(&self, idx: usize) -> bool {
        if idx >= self.len {
            return false;
        }
        ((self.words[idx >> 6] >> (idx & 63)) & 1) == 1
    }

    /// Number of `1` bits strictly before `idx` (`bits[0..idx]`).
    #[inline]
    pub fn rank1(&self, idx: usize) -> usize {
        if idx == 0 {
            return 0;
        }
        if idx >= self.len {
            return self.total_ones;
        }
        let sb_idx = idx / SUPERBLOCK_BITS;
        let mut count = self.superblocks[sb_idx] as usize;
        let sb_word_start = sb_idx * WORDS_PER_SUPERBLOCK;
        let target_word = idx >> 6;

        for &w in &self.words[sb_word_start..target_word] {
            count += popcount_u64(w) as usize;
        }

        let rem_bits = idx & 63;
        if rem_bits > 0 {
            let mask = (1u64 << rem_bits) - 1;
            count += popcount_u64(self.words[target_word] & mask) as usize;
        }
        count
    }

    /// Number of `0` bits strictly before `idx` (`bits[0..idx]`).
    #[inline]
    pub fn rank0(&self, idx: usize) -> usize {
        let clamped = idx.min(self.len);
        clamped - self.rank1(clamped)
    }

    /// Index of the `k`-th `1` bit (0-indexed), or `None` if `k >= total_ones`.
    pub fn select1(&self, k: usize) -> Option<usize> {
        if k >= self.total_ones {
            return None;
        }
        let sample_idx = k / SELECT_SAMPLE_STEP;
        let start_bit = self.select1_samples[sample_idx] as usize;
        let mut word_idx = start_bit >> 6;
        let start_offset = start_bit & 63;

        let mut rem = k - sample_idx * SELECT_SAMPLE_STEP;
        let first_word = self.words[word_idx] & (!0u64 << start_offset);
        let first_ones = popcount_u64(first_word) as usize;

        if rem < first_ones {
            return Some((word_idx << 6) + nth_set_bit(first_word, rem));
        }
        rem -= first_ones;
        word_idx += 1;

        while word_idx < self.words.len() {
            let w = self.words[word_idx];
            let ones = popcount_u64(w) as usize;
            if rem < ones {
                return Some((word_idx << 6) + nth_set_bit(w, rem));
            }
            rem -= ones;
            word_idx += 1;
        }
        None
    }

    /// Index of the `k`-th `0` bit (0-indexed), or `None` if `k >= total_zeros`.
    pub fn select0(&self, k: usize) -> Option<usize> {
        if k >= self.total_zeros() {
            return None;
        }
        let sample_idx = k / SELECT_SAMPLE_STEP;
        let start_bit = self.select0_samples[sample_idx] as usize;
        let mut word_idx = start_bit >> 6;
        let start_offset = start_bit & 63;

        let mut rem = k - sample_idx * SELECT_SAMPLE_STEP;
        let valid_first = (self.len - word_idx * 64).min(64);
        let valid_mask = if valid_first == 64 {
            !0u64
        } else {
            (1u64 << valid_first) - 1
        };
        let first_zeros_word = (!self.words[word_idx]) & valid_mask & (!0u64 << start_offset);
        let first_zeros = popcount_u64(first_zeros_word) as usize;

        if rem < first_zeros {
            return Some((word_idx << 6) + nth_set_bit(first_zeros_word, rem));
        }
        rem -= first_zeros;
        word_idx += 1;

        while word_idx < self.words.len() {
            let valid = (self.len - word_idx * 64).min(64);
            let mask = if valid == 64 {
                !0u64
            } else {
                (1u64 << valid) - 1
            };
            let zw = (!self.words[word_idx]) & mask;
            let zeros = popcount_u64(zw) as usize;
            if rem < zeros {
                return Some((word_idx << 6) + nth_set_bit(zw, rem));
            }
            rem -= zeros;
            word_idx += 1;
        }
        None
    }

    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&(self.words.len() as u64).to_le_bytes());
        for &w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    pub fn deserialize_from(cursor: &mut &[u8]) -> Result<Self, FmError> {
        let len = read_u64(cursor)? as usize;
        let word_count = read_u64(cursor)? as usize;
        let expected_words = len.div_ceil(64);
        if word_count != expected_words {
            return Err(FmError::CorruptIndex(format!(
                "BitVector word count mismatch: {word_count} vs {expected_words}"
            )));
        }
        let mut words = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            words.push(read_u64(cursor)?);
        }
        Ok(Self::from_words(words, len))
    }
}

#[inline]
fn nth_set_bit(mut word: u64, n: usize) -> usize {
    for _ in 0..n {
        word &= word - 1;
    }
    word.trailing_zeros() as usize
}

/// Wavelet Matrix over `u16` symbols (`bits_per_symbol` <= 16; `9` for 258-symbol FM-index).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaveletMatrix {
    len: usize,
    bits_per_symbol: u8,
    levels: Vec<(BitVector, usize)>,
}

impl WaveletMatrix {
    /// Builds a Wavelet Matrix over `symbols` where every symbol `< (1 << bits_per_symbol)`.
    pub fn build(symbols: &[u16], bits_per_symbol: u8) -> Self {
        let len = symbols.len();
        let mut current = symbols.to_vec();
        let mut levels = Vec::with_capacity(bits_per_symbol as usize);

        for level in 0..bits_per_symbol {
            let shift = bits_per_symbol - 1 - level;
            let mut bits = Vec::with_capacity(len);
            let mut zeros = Vec::with_capacity(len);
            let mut ones = Vec::with_capacity(len);

            for &sym in &current {
                let b = ((sym >> shift) & 1) == 1;
                bits.push(b);
                if b {
                    ones.push(sym);
                } else {
                    zeros.push(sym);
                }
            }

            let zero_count = zeros.len();
            zeros.extend_from_slice(&ones);
            current = zeros;
            levels.push((BitVector::from_bools(&bits), zero_count));
        }

        Self {
            len,
            bits_per_symbol,
            levels,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Accesses the symbol at position `idx`, or `None` if `idx >= len`.
    pub fn access(&self, idx: usize) -> Option<u16> {
        self.access_and_rank(idx).map(|(sym, _)| sym)
    }

    /// Returns `(symbol_at_idx, rank(symbol_at_idx, idx))` in a single downward pass.
    pub fn access_and_rank(&self, mut idx: usize) -> Option<(u16, usize)> {
        if idx >= self.len {
            return None;
        }
        let mut sym = 0u16;
        let mut left = 0usize;

        for (bv, zero_count) in &self.levels {
            let bit = bv.get(idx);
            sym <<= 1;
            if bit {
                sym |= 1;
                left = *zero_count + bv.rank1(left);
                idx = *zero_count + bv.rank1(idx);
            } else {
                left = bv.rank0(left);
                idx = bv.rank0(idx);
            }
        }

        Some((sym, idx - left))
    }

    /// Counts occurrences of `symbol` strictly before `idx` (`symbols[0..idx]`).
    pub fn rank(&self, symbol: u16, mut idx: usize) -> usize {
        if idx == 0 || self.len == 0 {
            return 0;
        }
        if idx > self.len {
            idx = self.len;
        }

        let mut left = 0usize;
        let mut right = idx;

        for (level_idx, (bv, zero_count)) in self.levels.iter().enumerate() {
            let shift = self.bits_per_symbol - 1 - level_idx as u8;
            let bit = ((symbol >> shift) & 1) == 1;
            if bit {
                left = *zero_count + bv.rank1(left);
                right = *zero_count + bv.rank1(right);
            } else {
                left = bv.rank0(left);
                right = bv.rank0(right);
            }
            if left == right {
                return 0;
            }
        }

        right - left
    }

    /// Returns the index of the `k`-th occurrence (0-indexed) of `symbol`, or `None`.
    pub fn select(&self, symbol: u16, k: usize) -> Option<usize> {
        if self.len == 0 {
            return None;
        }

        let mut left = 0usize;
        let mut right = self.len;
        let mut path = Vec::with_capacity(self.levels.len());

        for (level_idx, (bv, zero_count)) in self.levels.iter().enumerate() {
            let shift = self.bits_per_symbol - 1 - level_idx as u8;
            let bit = ((symbol >> shift) & 1) == 1;
            path.push((bv, *zero_count, bit, left));
            if bit {
                left = *zero_count + bv.rank1(left);
                right = *zero_count + bv.rank1(right);
            } else {
                left = bv.rank0(left);
                right = bv.rank0(right);
            }
            if left >= right {
                return None;
            }
        }

        if k >= right - left {
            return None;
        }

        let mut pos = left + k;
        for &(bv, zero_count, bit, prev_left) in path.iter().rev() {
            if bit {
                let ones_before_prev = bv.rank1(prev_left);
                let target_one = ones_before_prev + (pos - zero_count - ones_before_prev);
                pos = bv.select1(target_one)?;
            } else {
                let target_zero = pos;
                pos = bv.select0(target_zero)?;
            }
        }

        Some(pos)
    }

    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.push(self.bits_per_symbol);
        for (bv, zero_count) in &self.levels {
            out.extend_from_slice(&(*zero_count as u64).to_le_bytes());
            bv.serialize_into(out);
        }
    }

    pub fn deserialize_from(cursor: &mut &[u8]) -> Result<Self, FmError> {
        let len = read_u64(cursor)? as usize;
        let bits_per_symbol = read_u8(cursor)?;
        if bits_per_symbol == 0 || bits_per_symbol > 16 {
            return Err(FmError::CorruptIndex(format!(
                "invalid bits_per_symbol: {bits_per_symbol}"
            )));
        }
        let mut levels = Vec::with_capacity(bits_per_symbol as usize);
        for _ in 0..bits_per_symbol {
            let zero_count = read_u64(cursor)? as usize;
            let bv = BitVector::deserialize_from(cursor)?;
            if bv.len() != len {
                return Err(FmError::CorruptIndex(
                    "WaveletMatrix level length mismatch".to_owned(),
                ));
            }
            levels.push((bv, zero_count));
        }
        Ok(Self {
            len,
            bits_per_symbol,
            levels,
        })
    }
}

pub(crate) fn read_u8(cursor: &mut &[u8]) -> Result<u8, FmError> {
    let (&first, rest) = cursor
        .split_first()
        .ok_or_else(|| FmError::CorruptIndex("unexpected EOF reading u8".to_owned()))?;
    *cursor = rest;
    Ok(first)
}

pub(crate) fn read_u64(cursor: &mut &[u8]) -> Result<u64, FmError> {
    if cursor.len() < 8 {
        return Err(FmError::CorruptIndex(
            "unexpected EOF reading u64".to_owned(),
        ));
    }
    let (bytes, rest) = cursor.split_at(8);
    *cursor = rest;
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| FmError::CorruptIndex("u64 slice error".to_owned()))?;
    Ok(u64::from_le_bytes(arr))
}
