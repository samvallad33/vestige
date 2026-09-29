//! Portable archive types for exact Vestige-to-Vestige transfer.
//!
//! This format preserves SQLite row data instead of re-ingesting memories. It is
//! intentionally storage-level: import can keep IDs, FSRS state, graph edges,
//! suppression state, embeddings, and audit/history rows intact.

use rusqlite::types::Value;

/// Current portable archive format identifier.
pub const PORTABLE_ARCHIVE_FORMAT: &str = "vestige.portable.v1";

// `PortableArchive`, `PortableTable`, `PortableValue`, `PortableImportMode`,
// and `PortableImportReport` are defined in (and re-exported from)
// `crate::storage::types`.
pub use crate::storage::types::{
    PortableArchive, PortableImportMode, PortableImportReport, PortableTable, PortableValue,
};

impl PortableArchive {
    /// Count all rows across all tables.
    pub fn total_rows(&self) -> usize {
        self.tables.iter().map(|table| table.rows.len()).sum()
    }
}

// `PortableTable` and `PortableValue` definitions live in
// `crate::storage::types` (see the re-export above).

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

// `PortableImportMode` and `PortableImportReport` definitions live in
// `crate::storage::types` (see the re-export above).

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
