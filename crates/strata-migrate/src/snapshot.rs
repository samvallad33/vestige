//! Reading a migrated STRATA log back into typed records.
//!
//! This is the log-level reopen path: consumers (and the migration tests)
//! assert against the log itself via `strata::StrataLog::read_frames`,
//! decoded here into the migration record types.

use strata::StrataLog;
use strata_kernel::checkpoint::Checkpoint;
use strata_kernel::event::ReviewEvent;

use crate::records::*;
use crate::MigrationError;

/// Everything a migration wrote, decoded from log frames.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    pub genesis: Option<GenesisRecord>,
    pub params: Option<ParamsRecord>,
    pub nodes: Vec<NodeRecord>,
    pub edges: Vec<EdgeRecord>,
    /// Association rows (`semantic` / `similarity`). Not causal edges.
    pub legacy_links: Vec<LegacyLinkRecord>,
    pub reviews: Vec<ReviewEvent>,
    pub tombstones: Vec<TombstoneRecord>,
    pub supersessions: Vec<SupersessionRecord>,
    pub checkpoints: Vec<Checkpoint>,
    /// The final signed MIGRATION_RECEIPT, when the log carries one.
    pub receipt: Option<MigrationReceipt>,
    /// Count of frames whose kind is not part of the migration family
    /// (e.g. gate records sharing the log).
    pub other_frames: usize,
}

/// Read and decode a full migration snapshot from `log` (from seq 1).
pub fn read_snapshot(log: &StrataLog) -> Result<Snapshot, MigrationError> {
    let frames = log
        .read_frames(1)
        .map_err(|e| MigrationError::Strata(e.to_string()))?;

    let mut snapshot = Snapshot::default();
    for frame in frames {
        let decode = |e: borsh::io::Error| {
            MigrationError::Corrupt(format!(
                "frame seq {} kind {:#04x}: {e}",
                frame.seq, frame.kind
            ))
        };
        match frame.kind {
            KIND_GENESIS => {
                snapshot.genesis = Some(decode_genesis(&frame.payload).map_err(decode)?)
            }
            KIND_PARAMS => snapshot.params = Some(decode_params(&frame.payload).map_err(decode)?),
            KIND_NODE => snapshot
                .nodes
                .push(decode_node(&frame.payload).map_err(decode)?),
            KIND_EDGE => snapshot
                .edges
                .push(decode_edge(&frame.payload).map_err(decode)?),
            KIND_LEGACY_LINK => snapshot
                .legacy_links
                .push(decode_legacy_link(&frame.payload).map_err(decode)?),
            KIND_FSRS_REVIEW => snapshot
                .reviews
                .push(decode_review(&frame.payload).map_err(decode)?),
            KIND_TOMBSTONE => snapshot
                .tombstones
                .push(decode_tombstone(&frame.payload).map_err(decode)?),
            KIND_SUPERSESSION => snapshot
                .supersessions
                .push(decode_supersession(&frame.payload).map_err(decode)?),
            KIND_CHECKPOINT => snapshot
                .checkpoints
                .push(decode_checkpoint(&frame.payload).map_err(decode)?),
            KIND_MIGRATION_RECEIPT => {
                snapshot.receipt = Some(decode_receipt(&frame.payload).map_err(decode)?)
            }
            _ => snapshot.other_frames += 1,
        }
    }
    Ok(snapshot)
}
