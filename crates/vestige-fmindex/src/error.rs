#![forbid(unsafe_code)]

use std::fmt;

use crate::DocId;

/// Errors returned by `FmIndex` construction, extraction, or deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FmError {
    EmptyCorpus,
    InvalidSampleRate(u32),
    CorpusTooLarge(usize),
    InvalidDocId(DocId),
    OutOfBounds {
        doc_id: DocId,
        start: usize,
        len: usize,
        doc_len: usize,
    },
    CorruptIndex(String),
    UnsupportedVersion(u16),
}

impl fmt::Display for FmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCorpus => write!(f, "cannot build FmIndex from an empty corpus"),
            Self::InvalidSampleRate(r) => {
                write!(f, "sample_rate must be a power of two >= 1, got {r}")
            }
            Self::CorpusTooLarge(n) => {
                write!(f, "corpus symbol count {n} exceeds u32::MAX limit")
            }
            Self::InvalidDocId(id) => write!(f, "unknown doc_id {id}"),
            Self::OutOfBounds {
                doc_id,
                start,
                len,
                doc_len,
            } => write!(
                f,
                "extract range [{start}..{}) out of bounds for doc_id {doc_id} (len={doc_len})",
                start.saturating_add(*len)
            ),
            Self::CorruptIndex(msg) => write!(f, "corrupt FmIndex encoding: {msg}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported FmIndex format version: {v}"),
        }
    }
}

impl std::error::Error for FmError {}
