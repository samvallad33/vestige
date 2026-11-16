#![forbid(unsafe_code)]

//! Sampled Suffix Array (for `locate`) and Sampled Inverse Suffix Array (for `extract`)
//! compressed via `EliasFano` sparse bitvectors and configurable `sample_rate`.

use serde::{Deserialize, Serialize};

use crate::bwt::EliasFano;
use crate::error::FmError;
use crate::wavelet::read_u64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampledSa {
    text_len: usize,
    sample_rate: u32,
    sampled_bwt_indices: EliasFano,
    sampled_sa_quotients: Vec<u32>,
    isa_samples: Vec<u32>,
    end_bwt_idx: u32,
}

impl SampledSa {
    /// Builds sampled SA and ISA structures for a given `sample_rate` (must be power of 2 >= 1).
    pub fn build(sa: &[u32], sample_rate: u32) -> Result<Self, FmError> {
        if sample_rate == 0 || !sample_rate.is_power_of_two() {
            return Err(FmError::InvalidSampleRate(sample_rate));
        }

        let text_len = sa.len();
        let mask = sample_rate - 1;
        let mut sampled_bwt_indices = Vec::with_capacity(text_len.div_ceil(sample_rate as usize));
        let mut sampled_sa_quotients = Vec::with_capacity(sampled_bwt_indices.capacity());
        let mut isa_samples = vec![0u32; text_len.div_ceil(sample_rate as usize)];
        let mut end_bwt_idx = 0u32;

        for (bwt_idx, &sa_pos) in sa.iter().enumerate() {
            if (sa_pos & mask) == 0 {
                let slot = (sa_pos / sample_rate) as usize;
                sampled_bwt_indices.push(bwt_idx);
                sampled_sa_quotients.push(sa_pos / sample_rate);
                isa_samples[slot] = bwt_idx as u32;
            }
            if sa_pos as usize == text_len - 1 {
                end_bwt_idx = bwt_idx as u32;
            }
        }

        Ok(Self {
            text_len,
            sample_rate,
            sampled_bwt_indices: EliasFano::build(text_len, &sampled_bwt_indices),
            sampled_sa_quotients,
            isa_samples,
            end_bwt_idx,
        })
    }

    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Resolves `bwt_idx` to exact text position `SA[bwt_idx]` using `lf_fn`.
    pub fn resolve_sa<F>(&self, mut bwt_idx: usize, mut lf_fn: F) -> Option<usize>
    where
        F: FnMut(usize) -> Option<usize>,
    {
        let mut steps = 0usize;
        let sr = self.sample_rate as usize;
        loop {
            if let Some(k) = self.sampled_bwt_indices.index_of(bwt_idx) {
                let base_pos = self.sampled_sa_quotients[k] as usize * sr;
                return Some((base_pos + steps) % self.text_len);
            }
            bwt_idx = lf_fn(bwt_idx)?;
            steps += 1;
            if steps > self.text_len {
                return None;
            }
        }
    }

    /// Returns `(nearest_sampled_pos >= target_end, its_bwt_idx)` for backward `extract`.
    pub fn nearest_isa_anchor(&self, target_end: usize) -> (usize, usize) {
        let sr = self.sample_rate as usize;
        let next_slot = target_end.div_ceil(sr);
        if next_slot < self.isa_samples.len() {
            (next_slot * sr, self.isa_samples[next_slot] as usize)
        } else {
            (self.text_len - 1, self.end_bwt_idx as usize)
        }
    }

    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.text_len as u64).to_le_bytes());
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.end_bwt_idx.to_le_bytes());
        self.sampled_bwt_indices.serialize_into(out);
        out.extend_from_slice(&(self.sampled_sa_quotients.len() as u64).to_le_bytes());
        for &v in &self.sampled_sa_quotients {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.isa_samples.len() as u64).to_le_bytes());
        for &v in &self.isa_samples {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    pub fn deserialize_from(cursor: &mut &[u8]) -> Result<Self, FmError> {
        let text_len = read_u64(cursor)? as usize;
        let sample_rate = read_u32(cursor)?;
        if sample_rate == 0 || !sample_rate.is_power_of_two() {
            return Err(FmError::InvalidSampleRate(sample_rate));
        }
        let end_bwt_idx = read_u32(cursor)?;
        let sampled_bwt_indices = EliasFano::deserialize_from(cursor)?;
        let q_len = read_u64(cursor)? as usize;
        let mut sampled_sa_quotients = Vec::with_capacity(q_len);
        for _ in 0..q_len {
            sampled_sa_quotients.push(read_u32(cursor)?);
        }
        let isa_len = read_u64(cursor)? as usize;
        let mut isa_samples = Vec::with_capacity(isa_len);
        for _ in 0..isa_len {
            isa_samples.push(read_u32(cursor)?);
        }
        Ok(Self {
            text_len,
            sample_rate,
            sampled_bwt_indices,
            sampled_sa_quotients,
            isa_samples,
            end_bwt_idx,
        })
    }
}

fn read_u32(cursor: &mut &[u8]) -> Result<u32, FmError> {
    if cursor.len() < 4 {
        return Err(FmError::CorruptIndex(
            "unexpected EOF reading u32".to_owned(),
        ));
    }
    let (bytes, rest) = cursor.split_at(4);
    *cursor = rest;
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| FmError::CorruptIndex("u32 slice error".to_owned()))?;
    Ok(u32::from_le_bytes(arr))
}
