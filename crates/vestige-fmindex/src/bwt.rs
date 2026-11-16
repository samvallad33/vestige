#![forbid(unsafe_code)]

//! Run-Length Burrows-Wheeler Transform (RLBWT) + Elias-Fano sparse bitvectors
//! for compressed $O(\log \sigma)$ rank and LF-mapping over 10M+ token haystacks.

use serde::{Deserialize, Serialize};

use crate::error::FmError;
use crate::suffix::ALPHABET_SIZE;
use crate::wavelet::{read_u64, read_u8, BitVector, WaveletMatrix};

const SYMBOL_BITS: u8 = 9;

/// Compressed monotone integer sequence (`ones` in `[0..universe)`) using Elias-Fano encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EliasFano {
    universe: usize,
    n_ones: usize,
    low_bits: u8,
    low_words: Vec<u64>,
    high_bv: BitVector,
}

impl EliasFano {
    pub fn build(universe: usize, ones: &[usize]) -> Self {
        let n_ones = ones.len();
        if n_ones == 0 {
            return Self {
                universe,
                n_ones: 0,
                low_bits: 0,
                low_words: Vec::new(),
                high_bv: BitVector::from_bools(&[]),
            };
        }

        let ratio = (universe / n_ones).max(1);
        let low_bits = (usize::BITS - ratio.leading_zeros()).saturating_sub(1) as u8;
        let low_mask = if low_bits == 0 {
            0usize
        } else {
            (1usize << low_bits) - 1
        };

        let total_low_bits = n_ones * low_bits as usize;
        let mut low_words = vec![0u64; total_low_bits.div_ceil(64)];

        let max_high = (universe >> low_bits) + n_ones + 1;
        let mut high_words = vec![0u64; max_high.div_ceil(64)];

        for (i, &x) in ones.iter().enumerate() {
            if low_bits > 0 {
                let low_val = (x & low_mask) as u64;
                let bit_offset = i * low_bits as usize;
                let w_idx = bit_offset >> 6;
                let b_off = bit_offset & 63;
                low_words[w_idx] |= low_val << b_off;
                if b_off + low_bits as usize > 64 {
                    low_words[w_idx + 1] |= low_val >> (64 - b_off);
                }
            }
            let high_pos = (x >> low_bits) + i;
            high_words[high_pos >> 6] |= 1u64 << (high_pos & 63);
        }

        let high_bv = BitVector::from_words(high_words, max_high);

        Self {
            universe,
            n_ones,
            low_bits,
            low_words,
            high_bv,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n_ones
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_ones == 0
    }

    #[inline]
    fn get_low(&self, idx: usize) -> usize {
        if self.low_bits == 0 {
            return 0;
        }
        let bit_offset = idx * self.low_bits as usize;
        let w_idx = bit_offset >> 6;
        let b_off = bit_offset & 63;
        let mask = (1u64 << self.low_bits) - 1;
        let mut val = self.low_words[w_idx] >> b_off;
        if b_off + self.low_bits as usize > 64 {
            val |= self.low_words[w_idx + 1] << (64 - b_off);
        }
        (val & mask) as usize
    }

    /// Returns the `k`-th stored integer (0-indexed), or `universe` if `k >= n_ones`.
    #[inline]
    pub fn select1(&self, k: usize) -> usize {
        if k >= self.n_ones {
            return self.universe;
        }
        let high_pos = self.high_bv.select1(k).unwrap_or(self.universe);
        let high_val = high_pos - k;
        (high_val << self.low_bits) | self.get_low(k)
    }

    /// Counts stored integers strictly smaller than `bound`.
    pub fn rank1(&self, bound: usize) -> usize {
        if self.n_ones == 0 || bound == 0 {
            return 0;
        }
        if bound >= self.universe {
            return self.n_ones;
        }

        let h_target = bound >> self.low_bits;
        let (mut pos, mut k) = if h_target == 0 {
            (0usize, 0usize)
        } else {
            let Some(zero_pos) = self.high_bv.select0(h_target - 1) else {
                return self.n_ones;
            };
            let pos = zero_pos + 1;
            (pos, pos - h_target)
        };

        let low_mask = if self.low_bits == 0 {
            0usize
        } else {
            (1usize << self.low_bits) - 1
        };
        let bound_low = bound & low_mask;

        while k < self.n_ones && self.high_bv.get(pos) {
            if self.get_low(k) >= bound_low {
                break;
            }
            k += 1;
            pos += 1;
        }
        k
    }

    /// Returns `Some(k)` if `target` is the `k`-th stored integer, otherwise `None`.
    pub fn index_of(&self, target: usize) -> Option<usize> {
        let k = self.rank1(target);
        if k < self.n_ones && self.select1(k) == target {
            Some(k)
        } else {
            None
        }
    }

    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.universe as u64).to_le_bytes());
        out.extend_from_slice(&(self.n_ones as u64).to_le_bytes());
        out.push(self.low_bits);
        out.extend_from_slice(&(self.low_words.len() as u64).to_le_bytes());
        for &w in &self.low_words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        self.high_bv.serialize_into(out);
    }

    pub fn deserialize_from(cursor: &mut &[u8]) -> Result<Self, FmError> {
        let universe = read_u64(cursor)? as usize;
        let n_ones = read_u64(cursor)? as usize;
        let low_bits = read_u8(cursor)?;
        let low_words_len = read_u64(cursor)? as usize;
        let mut low_words = Vec::with_capacity(low_words_len);
        for _ in 0..low_words_len {
            low_words.push(read_u64(cursor)?);
        }
        let high_bv = BitVector::deserialize_from(cursor)?;
        Ok(Self {
            universe,
            n_ones,
            low_bits,
            low_words,
            high_bv,
        })
    }
}

/// Run-Length BWT supporting exact backward search (`rank`, `lf`, `bwt_len`, `num_runs`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlBwt {
    text_len: usize,
    num_runs: usize,
    run_heads: WaveletMatrix,
    b_runs: EliasFano,
    bp_runs: EliasFano,
    c_bytes: Vec<usize>,
    c_runs: Vec<usize>,
}

impl RlBwt {
    /// Constructs the RLBWT directly from `(syms, suffix_array)`.
    pub fn build(syms: &[u16], sa: &[u32]) -> Self {
        let text_len = syms.len();
        let mut runs: Vec<(u16, usize)> = Vec::new();
        let mut b_ones: Vec<usize> = Vec::new();

        for (i, &sa_pos) in sa.iter().enumerate() {
            let pos = sa_pos as usize;
            let bwt_sym = if pos == 0 {
                syms[text_len - 1]
            } else {
                syms[pos - 1]
            };

            if let Some((last_sym, last_len)) = runs.last_mut() {
                if *last_sym == bwt_sym {
                    *last_len += 1;
                    continue;
                }
            }
            runs.push((bwt_sym, 1));
            b_ones.push(i);
        }

        let num_runs = runs.len();
        let mut heads = Vec::with_capacity(num_runs);
        let mut byte_counts = vec![0usize; ALPHABET_SIZE];
        let mut run_counts = vec![0usize; ALPHABET_SIZE];
        let mut runs_by_sym: Vec<Vec<usize>> = vec![Vec::new(); ALPHABET_SIZE];

        for &(sym, run_len) in &runs {
            heads.push(sym);
            let s_idx = sym as usize;
            byte_counts[s_idx] += run_len;
            run_counts[s_idx] += 1;
            runs_by_sym[s_idx].push(run_len);
        }

        let mut c_bytes = vec![0usize; ALPHABET_SIZE];
        let mut c_runs = vec![0usize; ALPHABET_SIZE];
        let mut acc_b = 0usize;
        let mut acc_r = 0usize;
        for c in 0..ALPHABET_SIZE {
            c_bytes[c] = acc_b;
            c_runs[c] = acc_r;
            acc_b += byte_counts[c];
            acc_r += run_counts[c];
        }

        let mut bp_ones = Vec::with_capacity(num_runs);
        let mut f_pos = 0usize;
        for sym_runs in &runs_by_sym {
            for &run_len in sym_runs {
                bp_ones.push(f_pos);
                f_pos += run_len;
            }
        }

        Self {
            text_len,
            num_runs,
            run_heads: WaveletMatrix::build(&heads, SYMBOL_BITS),
            b_runs: EliasFano::build(text_len, &b_ones),
            bp_runs: EliasFano::build(text_len, &bp_ones),
            c_bytes,
            c_runs,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.text_len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.text_len == 0
    }

    #[inline]
    pub fn num_runs(&self) -> usize {
        self.num_runs
    }

    #[inline]
    pub fn c_byte(&self, sym: u16) -> usize {
        self.c_bytes.get(sym as usize).copied().unwrap_or(self.text_len)
    }

    /// Number of occurrences of `sym` in `L[0..idx]`.
    pub fn rank(&self, sym: u16, idx: usize) -> usize {
        if idx == 0 || self.text_len == 0 {
            return 0;
        }
        let clamped = idx.min(self.text_len);
        let k = self.b_runs.rank1(clamped);
        let j = k - 1;
        let s_idx = sym as usize;
        let m = self.run_heads.rank(sym, j);
        let mut total = self.bp_runs.select1(self.c_runs[s_idx] + m) - self.c_bytes[s_idx];
        if self.run_heads.access(j) == Some(sym) {
            total += clamped - self.b_runs.select1(j);
        }
        total
    }

    /// Computes `(L[idx], LF(idx))` in a single run lookup + Wavelet Matrix pass.
    pub fn symbol_and_lf(&self, idx: usize) -> Option<(u16, usize)> {
        if idx >= self.text_len {
            return None;
        }
        let k = self.b_runs.rank1(idx + 1);
        let j = k - 1;
        let (sym, m) = self.run_heads.access_and_rank(j)?;
        let run_start = self.b_runs.select1(j);
        let lf_pos = self.bp_runs.select1(self.c_runs[sym as usize] + m) + (idx - run_start);
        Some((sym, lf_pos))
    }

    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.text_len as u64).to_le_bytes());
        out.extend_from_slice(&(self.num_runs as u64).to_le_bytes());
        self.run_heads.serialize_into(out);
        self.b_runs.serialize_into(out);
        self.bp_runs.serialize_into(out);
        for &v in &self.c_bytes {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        for &v in &self.c_runs {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
    }

    pub fn deserialize_from(cursor: &mut &[u8]) -> Result<Self, FmError> {
        let text_len = read_u64(cursor)? as usize;
        let num_runs = read_u64(cursor)? as usize;
        let run_heads = WaveletMatrix::deserialize_from(cursor)?;
        let b_runs = EliasFano::deserialize_from(cursor)?;
        let bp_runs = EliasFano::deserialize_from(cursor)?;
        let mut c_bytes = Vec::with_capacity(ALPHABET_SIZE);
        for _ in 0..ALPHABET_SIZE {
            c_bytes.push(read_u64(cursor)? as usize);
        }
        let mut c_runs = Vec::with_capacity(ALPHABET_SIZE);
        for _ in 0..ALPHABET_SIZE {
            c_runs.push(read_u64(cursor)? as usize);
        }
        Ok(Self {
            text_len,
            num_runs,
            run_heads,
            b_runs,
            bp_runs,
            c_bytes,
            c_runs,
        })
    }
}
