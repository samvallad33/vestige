//! Hello, segment bytes, ack, admit.
//!
//! The bytes on the wire are an export directory: segment files, `writer.pub`,
//! and `chain.manifest`. `strata.key` is refused before it is written. The
//! receiver replicates each segment, acks the durable watermark, then admits
//! the range through the store fold.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use strata_store::{
    mesh_grant_context, AdmissionContext, ChainManifest, ImportReport, StoreError, StrataStore,
};
use thiserror::Error;

use crate::transport::MeshTransport;

const KEY_FILE: &str = "strata.key";

/// Why a mesh exchange stopped.
#[derive(Debug, Error)]
pub enum MeshError {
    /// The store refused the bytes.
    #[error("store: {0}")]
    Store(#[from] StoreError),
    /// The transport failed.
    #[error("transport: {0}")]
    Transport(String),
    /// A file in the export is not a segment, a manifest, or writer.pub.
    #[error("export: {0}")]
    Export(String),
    /// The peer sent `strata.key`.
    #[error("peer export contained strata.key; the replica did not read it")]
    SigningSeed,
}

/// What one side admitted.
#[derive(Debug)]
pub struct ImportSide {
    /// The store's import report.
    pub report: ImportReport,
    /// Durable `(seq, frame hash)` acks written before the merge.
    pub acks: Vec<(u64, [u8; 32])>,
}

/// Both sides after one exchange.
#[derive(Debug)]
pub struct SyncReport {
    /// The side that opened the exchange.
    pub local: ImportSide,
    /// The side that answered.
    pub remote: ImportSide,
}

/// Exchange the two stores' chains directly. Tests use this. A socket does not.
pub fn exchange(local: &mut StrataStore, remote: &mut StrataStore) -> Result<SyncReport, MeshError> {
    let local_bytes = pack_export(local)?;
    let remote_bytes = pack_export(remote)?;
    let local_side = apply_export(local, &remote_bytes, None, true)?;
    let remote_side = apply_export(remote, &local_bytes, None, true)?;
    Ok(SyncReport {
        local: local_side,
        remote: remote_side,
    })
}

/// Send this store's export and admit the peer's export.
///
/// `ack_log`, when set, receives one line per durable replica watermark:
/// `seq hex-frame-hash`. No key material.
pub fn sync_over<T: MeshTransport>(
    store: &mut StrataStore,
    transport: &mut T,
    ack_log: Option<&Path>,
) -> Result<ImportSide, MeshError> {
    let request = pack_export(store)?;
    let response = transport
        .roundtrip(&request)
        .map_err(MeshError::Transport)?;
    // A sync does not grant. Pairing already admitted the writer.
    apply_export(store, &response, ack_log, false)
}

/// Serve one request: admit the caller's export, answer with ours.
pub fn serve(
    store: &mut StrataStore,
    request: &[u8],
    ack_log: Option<&Path>,
) -> Result<Vec<u8>, MeshError> {
    let _ = apply_export(store, request, ack_log, false)?;
    pack_export(store)
}

/// Bytes of this store's chain export. `strata.key` is not included.
pub fn pack_export(store: &StrataStore) -> Result<Vec<u8>, MeshError> {
    let dir = tempfile_dir("pack")?;
    store.export_chain(&dir)?;
    if dir.join(KEY_FILE).exists() {
        let _ = fs::remove_dir_all(&dir);
        return Err(MeshError::SigningSeed);
    }
    let packed = pack_dir(&dir)?;
    let _ = fs::remove_dir_all(&dir);
    Ok(packed)
}

/// Replicate `bytes` one durable frame at a time, then admit the range.
///
/// `grant` is the human-confirmed path. A sync leaves it false, so a writer
/// without a grant is not admitted.
pub fn apply_export(
    store: &mut StrataStore,
    bytes: &[u8],
    ack_log: Option<&Path>,
    grant: bool,
) -> Result<ImportSide, MeshError> {
    let dir = tempfile_dir("apply")?;
    unpack_dir(bytes, &dir)?;
    if dir.join(KEY_FILE).exists() {
        let _ = fs::remove_dir_all(&dir);
        return Err(MeshError::SigningSeed);
    }
    let manifest = read_manifest(&dir)?;
    let ctx = if grant {
        mesh_grant_context()
    } else {
        AdmissionContext::default()
    };
    if manifest.policy_hash != store.policy().policy_hash() {
        let report = store.import_chain(&dir, &ctx)?;
        let _ = fs::remove_dir_all(&dir);
        return Ok(ImportSide {
            report,
            acks: Vec::new(),
        });
    }
    let mut acks = Vec::new();
    while let Some((seq, frame_hash)) = store.replicate_frame(manifest.verifying_key, &dir)? {
        acks.push((seq, frame_hash));
        if let Some(path) = ack_log {
            append_ack(path, seq, &frame_hash)?;
        }
        pace();
    }
    let report = store.import_chain(&dir, &ctx)?;
    let _ = fs::remove_dir_all(&dir);
    Ok(ImportSide { report, acks })
}

fn append_ack(path: &Path, seq: u64, frame_hash: &[u8; 32]) -> Result<(), MeshError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{seq} {}", hex32(frame_hash))?;
    file.sync_all()?;
    Ok(())
}

fn pace() {
    let Ok(ms) = std::env::var("VESTIGE_MESH_PACE_MS") else {
        return;
    };
    let Ok(ms) = ms.parse::<u64>() else {
        return;
    };
    if ms > 0 {
        std::thread::sleep(Duration::from_millis(ms));
    }
}

fn hex32(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

fn read_manifest(dir: &Path) -> Result<ChainManifest, MeshError> {
    let bytes = fs::read(dir.join("chain.manifest"))?;
    borsh::from_slice(&bytes).map_err(|err| MeshError::Export(err.to_string()))
}

fn pack_dir(dir: &Path) -> Result<Vec<u8>, MeshError> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        if name == KEY_FILE {
            return Err(MeshError::SigningSeed);
        }
        if name.ends_with(".seg") || name == "chain.manifest" || name == "writer.pub" || name == "head.state" {
            files.push((name, fs::read(&path)?));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    borsh::to_vec(&files).map_err(|err| MeshError::Export(err.to_string()))
}

fn unpack_dir(bytes: &[u8], dir: &Path) -> Result<(), MeshError> {
    fs::create_dir_all(dir)?;
    let files: Vec<(String, Vec<u8>)> =
        borsh::from_slice(bytes).map_err(|err| MeshError::Export(err.to_string()))?;
    for (name, bytes) in files {
        if name == KEY_FILE || name.contains("..") || name.contains('/') || name.contains('\\') {
            return Err(MeshError::SigningSeed);
        }
        let mut file = fs::File::create(dir.join(&name))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    Ok(())
}

fn tempfile_dir(label: &str) -> Result<PathBuf, MeshError> {
    let dir = std::env::temp_dir().join(format!(
        "strata-mesh-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

impl From<std::io::Error> for MeshError {
    fn from(err: std::io::Error) -> Self {
        MeshError::Export(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use strata_store::{IngestInput, StrataStore};

    use super::*;
    use crate::transport::MemoryPipe;

    fn dir(name: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("strata-mesh-proto-{n}-{name}"));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn input(content: &str) -> IngestInput {
        IngestInput {
            content: content.to_string(),
            source: None,
            source_updated_at_ms: None,
            node_type: String::new(),
            tags: Vec::new(),
            created_at_ms: Some(1_700_000_000_000),
            valid_from_ms: None,
            valid_until_ms: None,
        }
    }

    #[test]
    fn memory_pipe_sync_reaches_one_digest_and_stops_ping_pong() {
        let mut left = StrataStore::open(dir("left")).unwrap();
        let mut right = StrataStore::open(dir("right")).unwrap();
        left.ingest(input("left fact")).unwrap();
        right.ingest(input("right fact")).unwrap();
        let first = exchange(&mut left, &mut right).unwrap();
        assert!(!first.local.acks.is_empty());
        assert!(first.local.report.receipt_seq.is_some());
        assert_eq!(left.mesh_digest(), right.mesh_digest());

        let mut pipe = MemoryPipe::new(|request| {
            serve(&mut right, request, None).map_err(|err| err.to_string())
        });
        let again = sync_over(&mut left, &mut pipe, None).unwrap();
        assert!(again.report.receipt_only);
        assert!(again.report.receipt_seq.is_none());
        assert_eq!(left.mesh_digest(), right.mesh_digest());
    }

    #[test]
    fn an_acked_frame_is_still_there_after_the_temp_export_is_gone() {
        let mut left = StrataStore::open(dir("ack-left")).unwrap();
        let mut right = StrataStore::open(dir("ack-right")).unwrap();
        left.ingest(input("durable")).unwrap();
        right.ingest(input("other")).unwrap();
        let ack_path = dir("acks").join("acks.log");
        let packed = pack_export(&left).unwrap();
        let side = apply_export(&mut right, &packed, Some(&ack_path), true).unwrap();
        assert!(!side.acks.is_empty());
        let text = fs::read_to_string(&ack_path).unwrap();
        let (seq, _hash) = side.acks[0];
        assert!(text.contains(&format!("{seq} ")));
        let (last_seq, last_hash) = side.acks.last().copied().unwrap();
        assert!(text.contains(&format!("{last_seq} ")));
        let head = right.replica_head(left.verifying_key_bytes()).unwrap();
        assert_eq!(head, Some((last_seq, last_hash)));
    }
}
