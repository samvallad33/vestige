//! Strata `selftest`. Reads the user's log. The gate round-trip opens a
//! scratch store outside that directory.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use borsh::BorshDeserialize;
use ed25519_dalek::Verifier;
use serde_json::{Value, json};
use strata_gate::{
    EffectRecord, EventLog, GateRecord, MemLog, Propose, RecordKind, Verdict, policy_hash,
    rederive_verdicts, sweep,
};
use vestige_core::Storage;

struct ScannedFrame {
    kind: u8,
    payload: Vec<u8>,
}

struct LogScan {
    segments: u32,
    frames: Vec<ScannedFrame>,
}

struct ReceiptReport {
    effects: u64,
    writes: u64,
    orphans: u64,
    unmatched_effects: u64,
    gates: u64,
    gaps: u64,
    verdicts: Vec<(u64, Verdict)>,
}

struct RoundTrip {
    read_back: bool,
    node_id: String,
    effects: u64,
    gaps: u64,
    tail_frames: u64,
    checkpoint_chain_ok: bool,
    outside: bool,
    deleted: bool,
}

pub(super) fn run(storage: &Arc<Storage>) -> Result<Value, String> {
    let user_dir = storage.data_dir();
    let before = directory_blake3(user_dir)?;
    let nodes_before = node_count(storage)?;

    let scan = scan_log(&user_dir.join("log"))?;
    let receipts = check_receipts(&scan.frames)?;
    let trip = gate_round_trip(user_dir)?;

    let nodes_after = node_count(storage)?;
    let after = directory_blake3(user_dir)?;
    let touched = before != after;
    let registry_same = nodes_before == nodes_after;
    let log_ok = scan.segments >= 1;
    let receipts_ok =
        receipts.orphans == 0 && receipts.unmatched_effects == 0 && receipts.gaps == 0;
    let round_ok = trip.read_back
        && trip.gaps == 0
        && trip.checkpoint_chain_ok
        && trip.tail_frames > 0
        && trip.effects > 0
        && trip.outside
        && trip.deleted;
    let ok = log_ok && receipts_ok && round_ok && !touched && registry_same;
    if !ok {
        return Err(format!(
            "selftest failed: log={log_ok} receipts={receipts_ok} round_trip={round_ok} live_store_touched={touched} registry_same={registry_same}"
        ));
    }

    Ok(json!({
        "tool": "selftest",
        "kind": "strata_selftest",
        "ok": ok,
        "live_store_touched": touched,
        "user_store_blake3_before": hex32(&before),
        "user_store_blake3_after": hex32(&after),
        "user_nodes_before": nodes_before,
        "user_nodes_after": nodes_after,
        "scratch_outside_user_store": trip.outside,
        "scratch_store_deleted": trip.deleted,
        "checks": {
            "log_integrity": {
                "ok": log_ok,
                "segments": scan.segments,
                "frames": scan.frames.len() as u64,
            },
            "receipts": {
                "ok": receipts_ok,
                "effects": receipts.effects,
                "writes": receipts.writes,
                "orphans": receipts.orphans,
                "unmatched_effects": receipts.unmatched_effects,
                "gates": receipts.gates,
                "gaps": receipts.gaps,
            },
            "gate_round_trip": {
                "ok": round_ok,
                "read_back": trip.read_back,
                "node_id": trip.node_id,
                "effects": trip.effects,
                "gaps": trip.gaps,
                "tail_frames": trip.tail_frames,
                "checkpoint_chain_ok": trip.checkpoint_chain_ok,
            },
        },
    }))
}

fn node_count(storage: &Arc<Storage>) -> Result<usize, String> {
    storage
        .get_all_nodes(i32::MAX, 0)
        .map(|nodes| nodes.len())
        .map_err(|e| format!("selftest: read user registry: {e}"))
}

pub(super) fn directory_blake3(root: &Path) -> Result<[u8; 32], String> {
    let files = collect_files(root)?;
    let mut hasher = blake3::Hasher::new();
    for (name, bytes) in &files {
        hasher.update(name.as_bytes());
        hasher.update(&[0xff]);
        hasher.update(bytes);
        hasher.update(&[0xff]);
    }
    Ok(*hasher.finalize().as_bytes())
}

fn collect_files(root: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries =
            std::fs::read_dir(&dir).map_err(|e| format!("selftest: read directory: {e}"))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("selftest: read dirent: {e}"))?;
            let path = entry.path();
            let ft = entry
                .file_type()
                .map_err(|e| format!("selftest: file type: {e}"))?;
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                let rel = path
                    .strip_prefix(root)
                    .map_err(|e| format!("selftest: relative path: {e}"))?;
                let name = rel.to_string_lossy().replace('\\', "/");
                let bytes =
                    std::fs::read(&path).map_err(|e| format!("selftest: read {name}: {e}"))?;
                files.push((name, bytes));
            } else {
                return Err("selftest: unsupported file in the user store".into());
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
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

fn scan_log(log_dir: &Path) -> Result<LogScan, String> {
    if !log_dir.is_dir() {
        return Err("log integrity: log directory is missing".into());
    }
    let segs = list_segments(log_dir)?;
    if segs.is_empty() {
        return Err("log integrity: no segment files".into());
    }
    let key = read_verifying_key(log_dir)?;
    let mut prev = strata::GENESIS_PREV_SEGMENT_HASH;
    let mut frames = Vec::new();
    let last = segs.len() - 1;
    for (idx, (no, path)) in segs.iter().enumerate() {
        let bytes = std::fs::read(path).map_err(|e| format!("log integrity: segment {no}: {e}"))?;
        let header = parse_header(&bytes)
            .ok_or_else(|| format!("log integrity: segment {no}: header unreadable"))?;
        if header.prev_segment_hash != prev {
            return Err(format!(
                "log integrity: segment {no}: prev_segment_hash does not match the previous segment"
            ));
        }
        let body = scan_segment(&bytes, &header, *no)?;
        let is_last = idx == last;
        if let Some(trailer) = body.trailer {
            let key = key.as_ref().ok_or_else(|| {
                format!("log integrity: segment {no}: sealed trailer but strata.key is missing")
            })?;
            validate_trailer(&trailer, &body.leaves, body.frame_count, &header, key)
                .map_err(|reason| format!("log integrity: segment {no}: {reason}"))?;
            prev = *blake3::hash(&bytes).as_bytes();
        } else if !is_last {
            return Err(format!(
                "log integrity: segment {no}: sealed segment is missing its trailer"
            ));
        }
        frames.extend(body.frames);
    }
    Ok(LogScan {
        segments: u32::try_from(segs.len()).unwrap_or(u32::MAX),
        frames,
    })
}

struct SegmentBody {
    frames: Vec<ScannedFrame>,
    leaves: Vec<[u8; 32]>,
    frame_count: u64,
    trailer: Option<strata::SegmentTrailer>,
}

fn scan_segment(
    bytes: &[u8],
    header: &strata::SegmentHeader,
    segment_no: u32,
) -> Result<SegmentBody, String> {
    let mut frames = Vec::new();
    let mut leaves = Vec::new();
    let mut frame_count = 0u64;
    let mut last_frame_hash = strata::header_hash(header);
    let mut off = strata::HEADER_WIRE_SIZE;
    loop {
        let rem = bytes.len().saturating_sub(off);
        if rem == 0 {
            return Ok(SegmentBody {
                frames,
                leaves,
                frame_count,
                trailer: None,
            });
        }
        if rem == strata::TRAILER_WIRE_SIZE {
            let trailer =
                borsh::from_slice::<strata::SegmentTrailer>(&bytes[off..]).map_err(|_| {
                    format!(
                        "log integrity: segment {segment_no}: trailer-sized tail failed to parse"
                    )
                })?;
            return Ok(SegmentBody {
                frames,
                leaves,
                frame_count,
                trailer: Some(trailer),
            });
        }
        if rem < strata::FRAME_FIXED_WIRE_SIZE {
            return Err(format!(
                "log integrity: segment {segment_no}: short frame at offset {off}"
            ));
        }
        let (frame, used) = strata::parse_frame(&bytes[off..]).map_err(|e| {
            format!("log integrity: segment {segment_no}: frame parse at offset {off}: {e}")
        })?;
        if frame.payload_blake3 != strata::payload_blake3(frame.kind, &frame.payload) {
            return Err(format!(
                "log integrity: segment {segment_no}: payload blake3 mismatch at offset {off}"
            ));
        }
        if frame.prev_frame_hash != last_frame_hash {
            return Err(format!(
                "log integrity: segment {segment_no}: frame chain link mismatch at offset {off}"
            ));
        }
        last_frame_hash = strata::frame_hash(&frame);
        leaves.push(frame.payload_blake3);
        frames.push(ScannedFrame {
            kind: frame.kind,
            payload: frame.payload,
        });
        frame_count += 1;
        off += used;
    }
}

fn validate_trailer(
    trailer: &strata::SegmentTrailer,
    leaves: &[[u8; 32]],
    frame_count: u64,
    header: &strata::SegmentHeader,
    key: &ed25519_dalek::VerifyingKey,
) -> Result<(), String> {
    if trailer.frame_count != frame_count {
        return Err(format!(
            "trailer frame_count {} != scanned {frame_count}",
            trailer.frame_count
        ));
    }
    let root = strata::merkle_root(leaves);
    if trailer.merkle_root != root {
        return Err("trailer merkle root mismatch".into());
    }
    let msg = strata::signature_message(
        &header.segment_id,
        &header.prev_segment_hash,
        &trailer.merkle_root,
    );
    let sig = ed25519_dalek::Signature::from_bytes(&trailer.signature);
    key.verify(&msg, &sig)
        .map_err(|_| "trailer signature invalid".to_string())
}

fn parse_header(bytes: &[u8]) -> Option<strata::SegmentHeader> {
    if bytes.len() < strata::HEADER_WIRE_SIZE {
        return None;
    }
    let header: strata::SegmentHeader =
        borsh::from_slice(&bytes[..strata::HEADER_WIRE_SIZE]).ok()?;
    (header.magic == strata::SEGMENT_MAGIC && header.version == strata::SEGMENT_VERSION)
        .then_some(header)
}

fn read_verifying_key(log_dir: &Path) -> Result<Option<ed25519_dalek::VerifyingKey>, String> {
    let path = log_dir.join("strata.key");
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("log integrity: read strata.key: {e}"))?;
    let seed: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| "log integrity: strata.key is not 32 bytes".to_string())?;
    Ok(Some(
        ed25519_dalek::SigningKey::from_bytes(&seed).verifying_key(),
    ))
}

fn list_segments(dir: &Path) -> Result<Vec<(u32, PathBuf)>, String> {
    let mut out = Vec::new();
    for entry in
        std::fs::read_dir(dir).map_err(|e| format!("log integrity: read log directory: {e}"))?
    {
        let entry = entry.map_err(|e| format!("log integrity: read dirent: {e}"))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(stem) = name.strip_suffix(".seg") else {
            continue;
        };
        let (num, hex) = stem
            .split_once('-')
            .ok_or_else(|| format!("log integrity: bad segment name: {name}"))?;
        if num.len() != 8
            || !num.bytes().all(|b| b.is_ascii_digit())
            || hex.len() != 32
            || !hex.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(format!("log integrity: bad segment name: {name}"));
        }
        let no: u32 = num
            .parse()
            .map_err(|_| format!("log integrity: segment number out of range: {name}"))?;
        out.push((no, entry.path()));
    }
    out.sort_by_key(|(no, _)| *no);
    for (i, (no, _)) in out.iter().enumerate() {
        let expected = u32::try_from(i).unwrap_or(u32::MAX);
        if *no != expected {
            return Err(format!(
                "log integrity: segment numbering not dense: expected {expected:08}, found {no:08}"
            ));
        }
    }
    Ok(out)
}

fn check_receipts(frames: &[ScannedFrame]) -> Result<ReceiptReport, String> {
    let mut mem = MemLog::new();
    let mut propose_at: HashMap<u64, Propose> = HashMap::new();
    let mut gates_for: HashMap<u64, Vec<(u64, GateRecord)>> = HashMap::new();
    let mut pending: HashMap<[u8; 32], VecDeque<u64>> = HashMap::new();
    let mut stored: Vec<(u64, Verdict)> = Vec::new();
    let expected_policy = policy_hash(&strata_store::default_policy());
    let mut effects = 0u64;
    let mut writes = 0u64;
    let mut orphans = 0u64;

    for frame in frames {
        if let Some(kind) = RecordKind::from_u8(frame.kind) {
            let gseq = mem.append(kind, frame.payload.clone()).seq;
            match kind {
                RecordKind::Propose => {
                    let propose = Propose::try_from_slice(&frame.payload)
                        .map_err(|e| format!("receipts: propose at gate seq {gseq}: {e}"))?;
                    propose_at.insert(gseq, propose);
                }
                RecordKind::Gate => {
                    let gate = GateRecord::try_from_slice(&frame.payload)
                        .map_err(|e| format!("receipts: gate at gate seq {gseq}: {e}"))?;
                    if gate.policy_hash != expected_policy {
                        return Err(format!(
                            "receipts: gate at seq {gseq} policy hash does not match the pinned policy"
                        ));
                    }
                    stored.push((gseq, gate.verdict));
                    gates_for
                        .entry(gate.propose_seq)
                        .or_default()
                        .push((gseq, gate));
                }
                RecordKind::Effect => {
                    let effect = EffectRecord::try_from_slice(&frame.payload)
                        .map_err(|e| format!("receipts: effect at gate seq {gseq}: {e}"))?;
                    effects += 1;
                    let covering = propose_at
                        .get(&effect.propose_seq)
                        .is_some_and(|propose| propose.action_hash == effect.action_hash);
                    let admitting = gates_for.get(&effect.propose_seq).is_some_and(|gates| {
                        gates.iter().any(|(seq, gate)| {
                            *seq == effect.gate_seq && gate.verdict == Verdict::Allow && *seq < gseq
                        })
                    });
                    if !covering || !admitting {
                        return Err(format!(
                            "receipts: effect at gate seq {gseq} is not admitted by an Allow gate"
                        ));
                    }
                    pending
                        .entry(effect.payload_digest)
                        .or_default()
                        .push_back(gseq);
                }
                RecordKind::Gap
                | RecordKind::LessonAlarm
                | RecordKind::Canary
                | RecordKind::Alert => {}
            }
        } else if frame.kind == strata_store::KIND_STORE_WRITE {
            writes += 1;
            let digest = *blake3::hash(&frame.payload).as_bytes();
            let admitted = pending.get_mut(&digest).and_then(|queue| queue.pop_front());
            let decoded = strata_store::StoreOp::try_from_slice(&frame.payload).is_ok();
            if admitted.is_none() || !decoded {
                orphans += 1;
            }
        }
    }

    let unmatched_effects = pending.values().map(|queue| queue.len() as u64).sum();
    let rederived = rederive_verdicts(&mem, &strata_store::default_policy())
        .map_err(|e| format!("receipts: rederive: {e}"))?;
    let gaps = sweep(&mem);
    let verdicts_match = stored == rederived;
    if orphans > 0 || unmatched_effects > 0 || !verdicts_match || !gaps.is_empty() {
        return Err(format!(
            "receipts: orphans={orphans} unmatched={unmatched_effects} verdicts_match={verdicts_match} gaps={}",
            gaps.len()
        ));
    }
    Ok(ReceiptReport {
        effects,
        writes,
        orphans,
        unmatched_effects,
        gates: stored.len() as u64,
        gaps: gaps.len() as u64,
        verdicts: stored,
    })
}

fn gate_round_trip(user_dir: &Path) -> Result<RoundTrip, String> {
    let scratch = tempfile::TempDir::new().map_err(|e| format!("gate round-trip: tempdir: {e}"))?;
    let outside = scratch_is_outside(user_dir, scratch.path())?;
    if !outside {
        return Err("gate round-trip: scratch store is inside the user store".into());
    }

    let content = "selftest gate round-trip";
    let mut store = strata_store::StrataStore::open(scratch.path())
        .map_err(|e| format!("gate round-trip: open: {e}"))?;
    let node_id = store
        .ingest(strata_store::IngestInput {
            content: content.to_string(),
            node_type: "fact".into(),
            tags: Vec::new(),
            created_at_ms: Some(0),
            valid_from_ms: None,
            valid_until_ms: None,
        })
        .map_err(|e| format!("gate round-trip: ingest: {e}"))?;
    let read_back = store
        .get_node(&node_id)
        .is_some_and(|node| node.content == content);
    if !read_back {
        return Err("gate round-trip: read-back content does not match the ingest".into());
    }

    store
        .seal_checkpoint()
        .map_err(|e| format!("gate round-trip: seal checkpoint: {e}"))?;
    let checkpoint_chain_ok = match store.verify_checkpoint_chain() {
        Ok(()) => true,
        Err(e) => return Err(format!("gate round-trip: checkpoint chain: {e}")),
    };
    let gaps = store.sweep();
    if !gaps.is_empty() {
        return Err(format!(
            "gate round-trip: sweep found {} gap(s)",
            gaps.len()
        ));
    }
    let rederived = store
        .rederive_verdicts()
        .map_err(|e| format!("gate round-trip: rederive: {e}"))?;
    let tail = store
        .log()
        .verify_tail()
        .map_err(|e| format!("gate round-trip: verify_tail: {e}"))?;
    let durable = store
        .log()
        .read_frames(1)
        .map_err(|e| format!("gate round-trip: read frames: {e}"))?;
    let scanned = strata_store_frames(&durable);
    let receipts = check_receipts(&scanned).map_err(|e| format!("gate round-trip: {e}"))?;
    if receipts.verdicts != rederived {
        return Err(
            "gate round-trip: stored gate verdicts differ from the store re-derivation".into(),
        );
    }

    let effects = receipts.effects;
    let tail_frames = tail.frames_verified;
    drop(store);
    let deleted = scratch.close().is_ok();
    if !deleted {
        return Err("gate round-trip: scratch store was not deleted".into());
    }
    Ok(RoundTrip {
        read_back,
        node_id,
        effects,
        gaps: gaps.len() as u64,
        tail_frames,
        checkpoint_chain_ok,
        outside,
        deleted,
    })
}

fn scratch_is_outside(user_dir: &Path, scratch: &Path) -> Result<bool, String> {
    let user = std::fs::canonicalize(user_dir)
        .map_err(|e| format!("gate round-trip: user store path: {e}"))?;
    let scratch = std::fs::canonicalize(scratch)
        .map_err(|e| format!("gate round-trip: scratch path: {e}"))?;
    Ok(!scratch.starts_with(&user))
}

fn strata_store_frames(frames: &[strata::FrameRecord]) -> Vec<ScannedFrame> {
    frames
        .iter()
        .map(|frame| ScannedFrame {
            kind: frame.kind,
            payload: frame.payload.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::Arc;

    use serde_json::json;
    use vestige_core::{IngestInput, Storage};

    use super::hex32;

    fn open_store() -> (Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    /// Independent of [`collect_files`]: a raw byte map of the real data dir.
    fn snapshot(dir: &Path) -> BTreeMap<String, Vec<u8>> {
        let mut out = BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if entry.file_type().unwrap().is_dir() {
                    stack.push(path);
                } else {
                    let rel = path
                        .strip_prefix(dir)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.insert(rel, std::fs::read(&path).unwrap());
                }
            }
        }
        out
    }

    fn blake3_snapshot(files: &BTreeMap<String, Vec<u8>>) -> String {
        let mut hasher = blake3::Hasher::new();
        for (name, bytes) in files {
            hasher.update(name.as_bytes());
            hasher.update(&[0xff]);
            hasher.update(bytes);
            hasher.update(&[0xff]);
        }
        let digest: [u8; 32] = hasher.finalize().into();
        hex32(&digest)
    }

    fn assert_no_sqlite(dir: &Path) {
        let mut stack = vec![dir.to_path_buf()];
        while let Some(path) = stack.pop() {
            for entry in std::fs::read_dir(&path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                assert!(
                    !name.ends_with(".sqlite")
                        && !name.ends_with(".sqlite3")
                        && !name.ends_with(".db")
                        && !name.ends_with(".db-wal")
                        && !name.ends_with(".db-shm"),
                    "sqlite file created: {name}"
                );
                if path.is_dir() {
                    stack.push(path);
                }
            }
        }
    }

    fn flip_last_segment_byte(dir: &Path) {
        let mut segs: Vec<_> = std::fs::read_dir(dir.join("log"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("seg"))
            .collect();
        segs.sort();
        let path = &segs[0];
        let mut bytes = std::fs::read(path).unwrap();
        assert!(bytes.len() > strata::HEADER_WIRE_SIZE);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(path, bytes).unwrap();
    }

    #[tokio::test]
    async fn seeded_store_selftest_runs_every_check_and_leaves_the_bytes_unchanged() {
        let (storage, dir) = open_store();
        storage
            .ingest(IngestInput {
                content: "user-store fixture for selftest".into(),
                ..Default::default()
            })
            .unwrap();
        let before = snapshot(dir.path());
        let before_hash = blake3_snapshot(&before);

        let first = super::super::execute(&storage, None)
            .await
            .expect("selftest");
        let second = super::super::execute(&storage, None)
            .await
            .expect("selftest is deterministic");
        assert_eq!(first, second);

        let after = snapshot(dir.path());
        assert_eq!(before, after, "selftest wrote the user data dir");
        assert_eq!(before_hash, blake3_snapshot(&after));
        assert_eq!(first["user_store_blake3_before"], before_hash);
        assert_eq!(first["user_store_blake3_after"], before_hash);
        assert_eq!(first["live_store_touched"], json!(false));
        assert_eq!(first["ok"], json!(true));
        assert_eq!(first["scratch_outside_user_store"], json!(true));
        assert_eq!(first["scratch_store_deleted"], json!(true));
        assert_eq!(first["user_nodes_before"], json!(1));
        assert_eq!(first["user_nodes_after"], json!(1));
        assert_eq!(first["checks"]["log_integrity"]["ok"], json!(true));
        assert_eq!(first["checks"]["log_integrity"]["segments"], json!(1));
        assert_eq!(first["checks"]["log_integrity"]["frames"], json!(4));
        assert_eq!(first["checks"]["receipts"]["ok"], json!(true));
        assert_eq!(first["checks"]["receipts"]["effects"], json!(1));
        assert_eq!(first["checks"]["receipts"]["writes"], json!(1));
        assert_eq!(first["checks"]["receipts"]["orphans"], json!(0));
        assert_eq!(first["checks"]["receipts"]["unmatched_effects"], json!(0));
        assert_eq!(first["checks"]["receipts"]["gates"], json!(1));
        assert_eq!(first["checks"]["receipts"]["gaps"], json!(0));
        assert_eq!(first["checks"]["gate_round_trip"]["ok"], json!(true));
        assert_eq!(first["checks"]["gate_round_trip"]["read_back"], json!(true));
        assert_eq!(first["checks"]["gate_round_trip"]["effects"], json!(1));
        assert_eq!(first["checks"]["gate_round_trip"]["gaps"], json!(0));
        assert_eq!(first["checks"]["gate_round_trip"]["tail_frames"], json!(5));
        assert_eq!(
            first["checks"]["gate_round_trip"]["checkpoint_chain_ok"],
            json!(true)
        );
        let node_id = first["checks"]["gate_round_trip"]["node_id"]
            .as_str()
            .unwrap();
        assert!(
            node_id.starts_with("mem-") && node_id.len() == 20,
            "{node_id}"
        );
        assert_eq!(storage.get_all_nodes(10, 0).unwrap().len(), 1);
        assert_no_sqlite(dir.path());
    }

    #[tokio::test]
    async fn empty_store_still_round_trips_the_gate_without_writing() {
        let (storage, dir) = open_store();
        let before = snapshot(dir.path());
        let out = super::super::execute(&storage, None)
            .await
            .expect("selftest");
        assert_eq!(snapshot(dir.path()), before);
        assert_eq!(out["user_store_blake3_before"], blake3_snapshot(&before));
        assert_eq!(out["checks"]["log_integrity"]["frames"], json!(0));
        assert_eq!(out["checks"]["receipts"]["effects"], json!(0));
        assert_eq!(out["checks"]["receipts"]["gates"], json!(0));
        assert_eq!(out["checks"]["gate_round_trip"]["read_back"], json!(true));
        assert_eq!(out["checks"]["gate_round_trip"]["effects"], json!(1));
        assert_eq!(out["user_nodes_before"], json!(0));
        assert_eq!(out["user_nodes_after"], json!(0));
        assert_no_sqlite(dir.path());
    }

    #[tokio::test]
    async fn tampered_segment_makes_selftest_fail_without_writing() {
        let (storage, dir) = open_store();
        storage
            .ingest(IngestInput {
                content: "user-store fixture for selftest".into(),
                ..Default::default()
            })
            .unwrap();
        flip_last_segment_byte(dir.path());
        let before = snapshot(dir.path());
        let err = super::super::execute(&storage, None)
            .await
            .expect_err("tampered log must fail selftest");
        assert!(
            err.contains("log integrity"),
            "expected a log integrity failure, got {err}"
        );
        assert!(!err.contains("pending_strata"), "{err}");
        assert_eq!(
            snapshot(dir.path()),
            before,
            "failure path wrote the user store"
        );
        assert_eq!(storage.get_all_nodes(10, 0).unwrap().len(), 1);
        assert_no_sqlite(dir.path());
    }
}
