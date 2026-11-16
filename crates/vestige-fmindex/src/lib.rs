#![deny(unsafe_code)]

//! # `vestige-fmindex` (`#766`)
//!
//! Deterministic Run-Length FM-Index backed by SA-IS, RLBWT, Elias-Fano sparse
//! bitvectors, and a `u64`-word Wavelet Matrix.
//!
//! - Zero embeddings, zero floats, zero fuzzy matching.
//! - Multi-document separator (`SYM_DOC_SEP`) guarantees no match crosses a doc boundary
//!   while supporting all 256 raw byte values (`0x00..=0xFF`).
//! - Canonical serialization (`to_bytes` / `from_bytes`) and deterministic BLAKE3 `digest()`.

pub mod bwt;
pub mod error;
pub mod sample;
pub mod suffix;
pub mod wavelet;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bwt::RlBwt;
pub use crate::error::FmError;
use crate::sample::SampledSa;
use crate::suffix::{build_suffix_array, encode_documents, BYTE_OFFSET};

pub type DocId = u32;

const MAGIC: &[u8; 8] = b"VFMIDX01";
const FORMAT_VERSION: u16 = 1;
pub const DEFAULT_SAMPLE_RATE: u32 = 64;

/// A single exact substring match within a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Hit {
    pub byte_offset: u32,
    pub doc_id: DocId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DocSpan {
    doc_id: DocId,
    start: u32,
    len: u32,
}

/// Deterministic Run-Length FM-Index over multi-document byte slices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FmIndex {
    rlbwt: RlBwt,
    samples: SampledSa,
    docs: Vec<DocSpan>,
    doc_lookup: BTreeMap<DocId, usize>,
    digest: [u8; 32],
}

impl FmIndex {
    /// Builds an `FmIndex` using [`DEFAULT_SAMPLE_RATE`] (`64`).
    pub fn build(docs: &[(DocId, &[u8])]) -> Result<Self, FmError> {
        Self::build_with_sample_rate(docs, DEFAULT_SAMPLE_RATE)
    }

    /// Builds an `FmIndex` with an explicit power-of-two suffix array `sample_rate`.
    pub fn build_with_sample_rate(
        docs: &[(DocId, &[u8])],
        sample_rate: u32,
    ) -> Result<Self, FmError> {
        if docs.is_empty() {
            return Err(FmError::EmptyCorpus);
        }
        if sample_rate == 0 || !sample_rate.is_power_of_two() {
            return Err(FmError::InvalidSampleRate(sample_rate));
        }

        let total_raw: usize = docs.iter().map(|(_, b)| b.len()).sum();
        let total_syms = total_raw.saturating_add(docs.len()).saturating_add(1);
        if total_syms > u32::MAX as usize {
            return Err(FmError::CorpusTooLarge(total_syms));
        }

        // Canonicalize doc order deterministically by `doc_id` if caller passed unsorted IDs?
        // Preserve caller slice order unless duplicate doc_id appears, and reject duplicate doc_ids.
        let mut doc_lookup = BTreeMap::new();
        for (idx, &(doc_id, _)) in docs.iter().enumerate() {
            if doc_lookup.insert(doc_id, idx).is_some() {
                return Err(FmError::CorruptIndex(format!("duplicate doc_id {doc_id}")));
            }
        }

        let (syms, raw_spans) = encode_documents(docs);
        let sa = build_suffix_array(&syms);
        let rlbwt = RlBwt::build(&syms, &sa);
        let samples = SampledSa::build(&sa, sample_rate)?;

        let doc_spans: Vec<DocSpan> = raw_spans
            .into_iter()
            .map(|(doc_id, start, len)| DocSpan { doc_id, start, len })
            .collect();

        let mut index = Self {
            rlbwt,
            samples,
            docs: doc_spans,
            doc_lookup,
            digest: [0u8; 32],
        };
        index.digest = compute_digest(&index.to_bytes_without_digest());
        Ok(index)
    }

    /// Counts exact occurrences of `pattern` across all indexed documents.
    /// Empty patterns return `0`. Matches never cross document boundaries.
    pub fn count(&self, pattern: &[u8]) -> u64 {
        match self.backward_search(pattern) {
            Some((lo, hi)) => (hi - lo) as u64,
            None => 0,
        }
    }

    /// Locates up to `limit` exact matches of `pattern`, ordered deterministically by
    /// `(byte_offset, doc_id)`. Empty patterns return an empty `Vec`.
    pub fn locate(&self, pattern: &[u8], limit: usize) -> Vec<Hit> {
        if limit == 0 {
            return Vec::new();
        }
        let Some((lo, hi)) = self.backward_search(pattern) else {
            return Vec::new();
        };

        let range_len = hi - lo;
        // When the total match count is small or unbounded, resolve all hits and sort by
        // `(byte_offset, doc_id)`. On huge adversarial ranges with small `limit`, cap work.
        let resolve_cap = if limit == usize::MAX {
            range_len
        } else {
            range_len.min(limit.max(65_536))
        };

        let mut hits = Vec::with_capacity(resolve_cap);
        for bwt_idx in lo..(lo + resolve_cap) {
            if let Some(abs_pos) = self
                .samples
                .resolve_sa(bwt_idx, |cur| self.rlbwt.symbol_and_lf(cur).map(|(_, next)| next))
            {
                if let Some(hit) = self.map_abs_pos(abs_pos as u32) {
                    hits.push(hit);
                }
            }
        }

        hits.sort();
        hits.truncate(limit);
        hits
    }

    /// Reconstructs `bytes[start .. start + len]` from `doc_id` without storing a raw copy.
    pub fn extract(&self, doc_id: DocId, start: usize, len: usize) -> Result<Vec<u8>, FmError> {
        let &doc_idx = self
            .doc_lookup
            .get(&doc_id)
            .ok_or(FmError::InvalidDocId(doc_id))?;
        let span = &self.docs[doc_idx];
        let doc_len = span.len as usize;
        if start.saturating_add(len) > doc_len {
            return Err(FmError::OutOfBounds {
                doc_id,
                start,
                len,
                doc_len,
            });
        }
        if len == 0 {
            return Ok(Vec::new());
        }

        let abs_end = span.start as usize + start + len;
        let (anchor_pos, mut cur_bwt_idx) = self.samples.nearest_isa_anchor(abs_end);

        // Walk backward from `anchor_pos` to `abs_end`.
        let skip_steps = anchor_pos - abs_end;
        for _ in 0..skip_steps {
            let (_, next_idx) = self
                .rlbwt
                .symbol_and_lf(cur_bwt_idx)
                .ok_or_else(|| FmError::CorruptIndex("LF walk failed during extract".to_owned()))?;
            cur_bwt_idx = next_idx;
        }

        // Collect `len` bytes backward and reverse.
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let (sym, next_idx) = self
                .rlbwt
                .symbol_and_lf(cur_bwt_idx)
                .ok_or_else(|| FmError::CorruptIndex("LF walk failed during extract".to_owned()))?;
            if sym < BYTE_OFFSET {
                return Err(FmError::CorruptIndex(
                    "encountered sentinel byte inside document span".to_owned(),
                ));
            }
            out.push((sym - BYTE_OFFSET) as u8);
            cur_bwt_idx = next_idx;
        }
        out.reverse();
        Ok(out)
    }

    /// Canonical BLAKE3 digest of this index.
    #[inline]
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Number of BWT runs in the underlying RLBWT.
    #[inline]
    pub fn num_runs(&self) -> usize {
        self.rlbwt.num_runs()
    }

    /// Serializes the complete index into deterministic bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.to_bytes_without_digest();
        out.extend_from_slice(&self.digest);
        out
    }

    /// Deserializes and verifies an `FmIndex` from bytes produced by `to_bytes()`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FmError> {
        if bytes.len() < MAGIC.len() + 2 + 32 {
            return Err(FmError::CorruptIndex("buffer too short".to_owned()));
        }
        let (body, digest_slice) = bytes.split_at(bytes.len() - 32);
        let expected_digest: [u8; 32] = digest_slice
            .try_into()
            .map_err(|_| FmError::CorruptIndex("invalid trailing digest".to_owned()))?;
        let actual_digest = compute_digest(body);
        if actual_digest != expected_digest {
            return Err(FmError::CorruptIndex(
                "BLAKE3 index digest mismatch".to_owned(),
            ));
        }

        let mut cursor = body;
        if &cursor[..8] != MAGIC {
            return Err(FmError::CorruptIndex("invalid magic header".to_owned()));
        }
        cursor = &cursor[8..];

        let version = read_u16(&mut cursor)?;
        if version != FORMAT_VERSION {
            return Err(FmError::UnsupportedVersion(version));
        }

        let doc_count = wavelet::read_u64(&mut cursor)? as usize;
        let mut docs = Vec::with_capacity(doc_count);
        let mut doc_lookup = BTreeMap::new();
        for idx in 0..doc_count {
            let doc_id = read_u32(&mut cursor)?;
            let start = read_u32(&mut cursor)?;
            let len = read_u32(&mut cursor)?;
            doc_lookup.insert(doc_id, idx);
            docs.push(DocSpan { doc_id, start, len });
        }

        let rlbwt = RlBwt::deserialize_from(&mut cursor)?;
        let samples = SampledSa::deserialize_from(&mut cursor)?;
        if !cursor.is_empty() {
            return Err(FmError::CorruptIndex(
                "unexpected trailing bytes in FmIndex body".to_owned(),
            ));
        }

        Ok(Self {
            rlbwt,
            samples,
            docs,
            doc_lookup,
            digest: expected_digest,
        })
    }

    fn backward_search(&self, pattern: &[u8]) -> Option<(usize, usize)> {
        if pattern.is_empty() || pattern.len() > self.rlbwt.len() {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = self.rlbwt.len();
        for &b in pattern.iter().rev() {
            let sym = u16::from(b) + BYTE_OFFSET;
            let c_start = self.rlbwt.c_byte(sym);
            lo = c_start + self.rlbwt.rank(sym, lo);
            hi = c_start + self.rlbwt.rank(sym, hi);
            if lo >= hi {
                return None;
            }
        }
        Some((lo, hi))
    }

    fn map_abs_pos(&self, abs_pos: u32) -> Option<Hit> {
        let idx = self
            .docs
            .partition_point(|span| span.start <= abs_pos)
            .checked_sub(1)?;
        let span = &self.docs[idx];
        let rel = abs_pos.checked_sub(span.start)?;
        if rel < span.len {
            Some(Hit {
                byte_offset: rel,
                doc_id: span.doc_id,
            })
        } else {
            None
        }
    }

    fn to_bytes_without_digest(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.docs.len() as u64).to_le_bytes());
        for span in &self.docs {
            out.extend_from_slice(&span.doc_id.to_le_bytes());
            out.extend_from_slice(&span.start.to_le_bytes());
            out.extend_from_slice(&span.len.to_le_bytes());
        }
        self.rlbwt.serialize_into(&mut out);
        self.samples.serialize_into(&mut out);
        out
    }
}

fn compute_digest(body: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vestige.fmindex.v1\0");
    hasher.update(body);
    *hasher.finalize().as_bytes()
}

fn read_u16(cursor: &mut &[u8]) -> Result<u16, FmError> {
    if cursor.len() < 2 {
        return Err(FmError::CorruptIndex(
            "unexpected EOF reading u16".to_owned(),
        ));
    }
    let (bytes, rest) = cursor.split_at(2);
    *cursor = rest;
    let arr: [u8; 2] = bytes
        .try_into()
        .map_err(|_| FmError::CorruptIndex("u16 slice error".to_owned()))?;
    Ok(u16::from_le_bytes(arr))
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
