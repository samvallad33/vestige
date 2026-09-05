//! Portable archive types for exact Vestige-to-Vestige transfer.
//!
//! This format preserves SQLite row data instead of re-ingesting memories. It is
//! intentionally storage-level: import can keep IDs, FSRS state, graph edges,
//! suppression state, embeddings, and audit/history rows intact.

use chrono::{DateTime, Utc};
use rusqlite::types::Value;
use serde::{Deserialize, Serialize};

/// Current portable archive format identifier.
pub const PORTABLE_ARCHIVE_FORMAT: &str = "vestige.portable.v1";

/// Full exact portable archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableArchive {
    /// Stable format marker used for compatibility checks.
    pub archive_format: String,
    /// Vestige version that produced the archive.
    pub vestige_version: String,
    /// SQLite schema version of the source database.
    pub schema_version: u32,
    /// Archive creation timestamp.
    pub exported_at: DateTime<Utc>,
    /// Export mode. v1 only writes "exact".
    pub mode: String,
    /// Dumped storage tables in deterministic import order.
    pub tables: Vec<PortableTable>,
}

impl PortableArchive {
    /// Count all rows across all tables.
    pub fn total_rows(&self) -> usize {
        self.tables.iter().map(|table| table.rows.len()).sum()
    }
}

/// One table in a portable archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableTable {
    /// SQLite table name.
    pub name: String,
    /// Column names in row value order.
    pub columns: Vec<String>,
    /// Raw rows. Each row has the same order as `columns`.
    pub rows: Vec<Vec<PortableValue>>,
}

/// SQLite value encoded in JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "camelCase")]
pub enum PortableValue {
    /// SQL NULL.
    Null,
    /// SQL INTEGER.
    Integer(i64),
    /// SQL REAL.
    Real(f64),
    /// SQL TEXT.
    Text(String),
    /// SQL BLOB, hex encoded.
    Blob(String),
}

impl PortableValue {
    /// Convert this portable value back into a rusqlite owned value.
    pub(crate) fn to_sql_value(&self) -> Result<Value, String> {
        match self {
            Self::Null => Ok(Value::Null),
            Self::Integer(value) => Ok(Value::Integer(*value)),
            Self::Real(value) => Ok(Value::Real(*value)),
            Self::Text(value) => Ok(Value::Text(value.clone())),
            Self::Blob(value) => decode_hex(value).map(Value::Blob),
        }
    }
}

/// Import behavior for duplicate primary keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortableImportMode {
    /// Reject import if user data already exists, then insert rows exactly.
    EmptyOnly,
    /// Merge archive rows into an existing database.
    ///
    /// This mode is intended for file-backed sync between devices. It applies
    /// tombstones, upserts row-keyed state, and appends audit/history rows.
    Merge,
}

/// Summary of an exact portable import.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortableImportReport {
    /// Number of imported tables.
    pub tables_imported: usize,
    /// Number of imported rows.
    pub rows_imported: usize,
    /// Number of archive tables skipped because the target schema lacks them.
    pub tables_skipped: usize,
    /// Whether FTS was rebuilt after import.
    pub fts_rebuilt: bool,
    /// Number of rows inserted.
    #[serde(default)]
    pub rows_inserted: usize,
    /// Number of existing rows updated/replaced.
    #[serde(default)]
    pub rows_updated: usize,
    /// Number of rows skipped because local state was newer or unsupported.
    #[serde(default)]
    pub rows_skipped: usize,
    /// Number of local rows deleted by imported tombstones.
    #[serde(default)]
    pub rows_deleted: usize,
    /// Number of merge conflicts resolved by keeping local state.
    #[serde(default)]
    pub conflicts_kept_local: usize,
}

pub(crate) fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn decode_hex(input: &str) -> Result<Vec<u8>, String> {
    if !input.len().is_multiple_of(2) {
        return Err("hex blob has odd length".to_string());
    }

    let mut out = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();
    for &[high, low] in bytes.as_chunks::<2>().0 {
        let high = hex_value(high)?;
        let low = hex_value(low)?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_value(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(format!("invalid hex byte: {}", byte as char)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip() {
        let bytes = vec![0, 1, 2, 15, 16, 127, 128, 255];
        let encoded = encode_hex(&bytes);
        assert_eq!(decode_hex(&encoded).unwrap(), bytes);
    }

    #[test]
    fn rejects_invalid_hex() {
        assert!(decode_hex("f").is_err());
        assert!(decode_hex("zz").is_err());
    }
}
