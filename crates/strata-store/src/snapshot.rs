//! Advisory LZ4-compressed, BLAKE3-verified cold snapshot cache for [`StrataStore`].
//!
//! The signed append-only [`strata::StrataLog`] remains the SOLE source of truth.
//! A snapshot file (`snapshot.lz4`) is strictly an advisory read-acceleration cache:
//! it is accepted ONLY when its magic, format version, BLAKE3 body digest, `head_seq`,
//! and `head_hash` all match the live log head byte-for-byte. Any mismatch, corruption,
//! or appended frame transparently falls back to full deterministic log replay.

use std::fs;
use std::path::Path;

use borsh::{BorshDeserialize, BorshSerialize};

use crate::StoreError;
use crate::types::{ConnectionRecord, NodeRecord};

/// Magic bytes identifying a Vestige Strata LZ4 cold snapshot v1.
pub const SNAPSHOT_MAGIC: [u8; 8] = *b"STRSNAP1";
/// Snapshot filename placed beside `store.meta` inside the store directory.
pub const SNAPSHOT_FILE: &str = "snapshot.lz4";

/// Header + payload committed inside `snapshot.lz4`.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
pub struct StoreSnapshotV1 {
    /// Fixed format magic (`STRSNAP1`).
    pub magic: [u8; 8],
    /// Log sequence number of the last acknowledged frame covered by this snapshot.
    pub head_seq: u64,
    /// Chain hash of the frame at `head_seq`.
    pub head_hash: [u8; 32],
    /// Canonical `StrataStore::state_digest()` at `head_seq`.
    pub state_digest: [u8; 32],
    /// Cached maximum clock in milliseconds.
    pub head_clock_ms: i64,
    /// All node records in canonical id order.
    pub nodes: Vec<NodeRecord>,
    /// All typed edges in landing order.
    pub edges: Vec<ConnectionRecord>,
}

/// Outer integrity envelope: magic + BLAKE3 digest over the compressed LZ4 body.
#[derive(Debug, Clone, PartialEq, Eq, BorshSerialize, BorshDeserialize)]
struct SnapshotEnvelope {
    magic: [u8; 8],
    payload_blake3: [u8; 32],
    lz4_payload: Vec<u8>,
}

/// Atomically write an LZ4-compressed, BLAKE3-verified snapshot to `<dir>/snapshot.lz4`.
pub fn write_snapshot(dir: &Path, snapshot: &StoreSnapshotV1) -> Result<usize, StoreError> {
    let raw = borsh::to_vec(snapshot).map_err(|e| StoreError::Encode(e.to_string()))?;
    let compressed = lz4_flex::compress_prepend_size(&raw);
    let payload_blake3 = *blake3::hash(&compressed).as_bytes();
    let envelope = SnapshotEnvelope {
        magic: SNAPSHOT_MAGIC,
        payload_blake3,
        lz4_payload: compressed,
    };
    let bytes = borsh::to_vec(&envelope).map_err(|e| StoreError::Encode(e.to_string()))?;
    let tmp = dir.join(format!("{SNAPSHOT_FILE}.{}.tmp", std::process::id()));
    fs::write(&tmp, &bytes)?;
    fs::rename(&tmp, dir.join(SNAPSHOT_FILE))?;
    Ok(bytes.len())
}

/// Read and strictly verify `<dir>/snapshot.lz4` against `(expected_head_seq, expected_head_hash)`.
///
/// Returns `Ok(None)` if the file is absent, stale, truncated, or fails BLAKE3 verification —
/// callers transparently fall back to full deterministic log replay.
pub fn load_verified_snapshot(
    dir: &Path,
    expected_head_seq: u64,
    expected_head_hash: [u8; 32],
) -> Option<StoreSnapshotV1> {
    let bytes = fs::read(dir.join(SNAPSHOT_FILE)).ok()?;
    let env = SnapshotEnvelope::try_from_slice(&bytes).ok()?;
    if env.magic != SNAPSHOT_MAGIC {
        return None;
    }
    if *blake3::hash(&env.lz4_payload).as_bytes() != env.payload_blake3 {
        return None;
    }
    let raw = lz4_flex::decompress_size_prepended(&env.lz4_payload).ok()?;
    let snap = StoreSnapshotV1::try_from_slice(&raw).ok()?;
    if snap.magic != SNAPSHOT_MAGIC
        || snap.head_seq != expected_head_seq
        || snap.head_hash != expected_head_hash
    {
        return None;
    }
    Some(snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VALID_FOREVER_MS;

    #[test]
    fn lz4_snapshot_roundtrips_and_rejects_stale_or_tampered_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "strata-lz4-snap-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let snap = StoreSnapshotV1 {
            magic: SNAPSHOT_MAGIC,
            head_seq: 42,
            head_hash: [0xabu8; 32],
            state_digest: [0xcdu8; 32],
            head_clock_ms: 1_795_800_000_000,
            nodes: vec![NodeRecord {
                id: "mem-0000000000000001".into(),
                kernel_id: 2,
                scope: "user".into(),
                content: "Zero-startup LZ4 snapshot verification invariant.".repeat(16),
                node_type: "decision".into(),
                tags: vec!["pinned".into()],
                created_at_ms: 1_795_800_000_000,
                valid_from_ms: 1_795_800_000_000,
                valid_until_ms: VALID_FOREVER_MS,
                superseded_by: None,
                source: None,
                source_updated_at_ms: None,
            }],
            edges: Vec::new(),
        };

        let written = write_snapshot(&dir, &snap).expect("write lz4 snapshot");
        assert!(written > 0 && written < 512, "expected high LZ4 compression, got {written}B");

        // 1. Exact head_seq + head_hash loads cleanly.
        let loaded = load_verified_snapshot(&dir, 42, [0xabu8; 32]).expect("verified snapshot");
        assert_eq!(loaded, snap);

        // 2. Stale head_seq or different head_hash invalidates automatically.
        assert!(load_verified_snapshot(&dir, 43, [0xabu8; 32]).is_none());
        assert!(load_verified_snapshot(&dir, 42, [0xefu8; 32]).is_none());

        // 3. Flipping 1 byte inside snapshot.lz4 fails BLAKE3 check and returns None.
        let path = dir.join(SNAPSHOT_FILE);
        let mut raw = fs::read(&path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xff;
        fs::write(&path, &raw).unwrap();
        assert!(load_verified_snapshot(&dir, 42, [0xabu8; 32]).is_none());

        let _ = fs::remove_dir_all(&dir);
    }
}
