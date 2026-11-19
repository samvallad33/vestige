//! Shared deterministic import pipeline, git exact-match binder, secret scanner,
//! resumable checkpoints, and `ImportReceipt` (#775 & #774).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::connector::{ConnectorFamily, ConnectorReadError, RawSourceRecord, SourceConnector};

/// Reason code for a quarantined / held record during import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HoldReason {
    /// Unanchored record held under Wave 14M (#774).
    MissingProvenanceReceipt,
    /// Index-only record (FAISS/hnswlib/Annoy) without sidecar text payload.
    NoTextPayload,
    /// Duplicate `source_id` encountered in source stream.
    DuplicateSourceId,
    /// Oversized payload exceeding deterministic frame safety bound.
    OversizedPayload,
}

/// Stripped vector audit summary (vectors NEVER enter Strata ranking).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorAuditStub {
    pub dims: usize,
    pub blake3_hex: String,
}

/// Canonical frame produced by the importer for Strata admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedFrame {
    pub mem_id: String,
    pub source_id: String,
    pub writer_id: String,
    pub text: String,
    pub metadata_json: String,
    pub vector_audit: Option<VectorAuditStub>,
    pub commit_anchor: Option<String>,
    pub verified_files: Vec<String>,
    pub cited_mem_ids: Vec<String>,
    pub quarantined: Option<HoldReason>,
    pub frame_hash: String,
}

/// Secret finding summary recorded in the `ImportReceipt`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecretFinding {
    pub source_id: String,
    pub pattern: String,
}

/// Complete cryptographic `ImportReceipt` emitted by every import run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportReceipt {
    pub connector_slug: String,
    pub connector_family: String,
    pub connector_version: String,
    pub collection: String,
    pub snapshot_digest: String,
    pub canonical_head: String,
    pub imported_count: usize,
    pub linked_count: usize,
    pub held_count: usize,
    pub held_reasons: BTreeMap<String, HoldReason>,
    pub id_map: BTreeMap<String, String>,
    pub unresolved_refs: Vec<String>,
    pub ambiguous_refs: Vec<String>,
    pub reason_codes: Vec<String>,
    pub read_errors: Vec<ConnectorReadError>,
    pub secret_findings: Vec<SecretFinding>,
}

/// Resumable signed cursor checkpoint so interrupted imports finish identically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportCheckpoint {
    pub collection: String,
    pub next_cursor: Option<String>,
    pub staged_records: Vec<RawSourceRecord>,
    pub checkpoint_blake3: String,
}

impl ImportCheckpoint {
    pub fn seal(
        collection: String,
        next_cursor: Option<String>,
        staged_records: Vec<RawSourceRecord>,
    ) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(collection.as_bytes());
        if let Some(c) = &next_cursor {
            h.update(c.as_bytes());
        }
        h.update(&(staged_records.len() as u64).to_le_bytes());
        let checkpoint_blake3 = h.finalize().to_hex().to_string();
        Self {
            collection,
            next_cursor,
            staged_records,
            checkpoint_blake3,
        }
    }

    pub fn verify(&self) -> bool {
        let expected = Self::seal(
            self.collection.clone(),
            self.next_cursor.clone(),
            self.staged_records.clone(),
        );
        expected.checkpoint_blake3 == self.checkpoint_blake3
    }

    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        fs::write(path, bytes)
    }

    pub fn load_from_file(path: &Path) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        let cp: Self = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
        if !cp.verify() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "ImportCheckpoint integrity check failed",
            ));
        }
        Ok(cp)
    }
}

/// Exact-match Git commit & tree path index (never links by time proximity).
#[derive(Debug, Clone, Default)]
pub struct GitExactIndex {
    /// Map of full 40-hex commit SHA -> set of tracked relative file paths at that commit.
    pub commits: BTreeMap<String, BTreeSet<String>>,
}

impl GitExactIndex {
    pub fn resolve_commit(&self, prefix: &str) -> Result<Option<String>, &'static str> {
        let p = prefix.trim().to_ascii_lowercase();
        if p.len() < 8 || p.len() > 40 || !p.chars().all(|c| c.is_ascii_hexdigit()) {
            return Ok(None);
        }
        let matches: Vec<&String> = self
            .commits
            .keys()
            .filter(|sha| sha.starts_with(&p))
            .collect();
        match matches.len() {
            0 => Ok(None),
            1 => Ok(Some(matches[0].clone())),
            _ => Err("AmbiguousRef"),
        }
    }

    pub fn verify_file_at_commit(&self, sha: &str, rel_path: &str) -> bool {
        let norm = rel_path.replace('\\', "/");
        if norm.starts_with('/') || norm.split('/').any(|seg| seg == "..") {
            return false;
        }
        self.commits
            .get(sha)
            .is_some_and(|tree| tree.contains(&norm))
    }
}

const MAX_TEXT_BYTES: usize = 2 * 1024 * 1024; // 2 MiB safety bound

fn scan_text_secrets(source_id: &str, text: &str) -> Vec<SecretFinding> {
    let mut findings = Vec::new();
    let patterns = [
        ("openai_api_key", "sk-proj-"),
        ("openai_legacy_key", "sk-"),
        ("github_pat", "ghp_"),
        ("aws_access_key", "AKIA"),
        ("private_key_pem", "-----BEGIN PRIVATE KEY-----"),
        ("rsa_private_key_pem", "-----BEGIN RSA PRIVATE KEY-----"),
    ];
    for (name, needle) in patterns {
        if text.contains(needle) {
            findings.push(SecretFinding {
                source_id: source_id.to_string(),
                pattern: name.to_string(),
            });
            break;
        }
    }
    findings
}

fn strip_vectors_from_metadata(meta: &Value) -> (Value, Option<VectorAuditStub>) {
    let mut cleaned = meta.clone();
    let mut stub = None;
    if let Some(obj) = cleaned.as_object_mut() {
        for key in ["embedding", "vector", "values", "embeddings"] {
            if let Some(val) = obj.remove(key)
                && let Some(arr) = val.as_array()
            {
                let mut h = blake3::Hasher::new();
                for v in arr {
                    let f = v.as_f64().unwrap_or(0.0) as f32;
                    h.update(&f.to_le_bytes());
                }
                stub = Some(VectorAuditStub {
                    dims: arr.len(),
                    blake3_hex: h.finalize().to_hex().to_string(),
                });
            }
        }
    }
    (cleaned, stub)
}

fn compute_vector_stub(vec: &[f32]) -> VectorAuditStub {
    let mut h = blake3::Hasher::new();
    for f in vec {
        h.update(&f.to_le_bytes());
    }
    VectorAuditStub {
        dims: vec.len(),
        blake3_hex: h.finalize().to_hex().to_string(),
    }
}

fn deterministic_mem_id(slug: &str, collection: &str, source_id: &str) -> String {
    let mut h = blake3::Hasher::new();
    h.update(b"strata-import-v1:");
    h.update(slug.as_bytes());
    h.update(b":");
    h.update(collection.as_bytes());
    h.update(b":");
    h.update(source_id.as_bytes());
    let hex = h.finalize().to_hex();
    format!("mem-{}", &hex.as_str()[..16])
}

/// Options controlling a single import execution.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    pub git_index: Option<GitExactIndex>,
    pub resume_checkpoint: Option<ImportCheckpoint>,
    pub stop_after_pages: Option<usize>,
}

/// Output of running the universal import pipeline.
#[derive(Debug, Clone)]
pub struct ImportRunOutput {
    pub frames: Vec<ImportedFrame>,
    pub receipt: ImportReceipt,
    pub pending_checkpoint: Option<ImportCheckpoint>,
}

/// Execute the Wave 14N universal import pipeline over any [`SourceConnector`].
pub fn run_import(
    connector: &dyn SourceConnector,
    collection: &str,
    opts: ImportOptions,
) -> ImportRunOutput {
    let mut read_errors = Vec::new();
    let mut reason_codes = BTreeSet::new();

    let snapshot_digest = match connector.snapshot_digest(collection) {
        Ok(d) => d,
        Err(e) => {
            reason_codes.insert(e.code.clone());
            read_errors.push(e);
            "unavailable".to_string()
        }
    };

    let reported_count = match connector.count(collection) {
        Ok(c) => c,
        Err(e) => {
            reason_codes.insert(e.code.clone());
            read_errors.push(e);
            None
        }
    };

    let (mut cursor, mut raw_records) = match opts.resume_checkpoint {
        Some(cp) if cp.collection == collection && cp.verify() => {
            (cp.next_cursor, cp.staged_records)
        }
        _ => (None, Vec::new()),
    };

    let mut pages_pulled = 0usize;
    loop {
        if let Some(limit) = opts.stop_after_pages
            && pages_pulled >= limit
        {
            let cp = ImportCheckpoint::seal(collection.to_string(), cursor.clone(), raw_records);
            return ImportRunOutput {
                frames: Vec::new(),
                receipt: ImportReceipt {
                    connector_slug: connector.slug().to_string(),
                    connector_family: connector.family().code().to_string(),
                    connector_version: connector.version().to_string(),
                    collection: collection.to_string(),
                    snapshot_digest,
                    canonical_head: "paused-at-checkpoint".to_string(),
                    imported_count: 0,
                    linked_count: 0,
                    held_count: 0,
                    held_reasons: BTreeMap::new(),
                    id_map: BTreeMap::new(),
                    unresolved_refs: Vec::new(),
                    ambiguous_refs: Vec::new(),
                    reason_codes: vec!["PausedAtCheckpoint".to_string()],
                    read_errors,
                    secret_findings: Vec::new(),
                },
                pending_checkpoint: Some(cp),
            };
        }

        match connector.pull_page(collection, cursor.as_deref()) {
            Ok(page) => {
                raw_records.extend(page.records);
                pages_pulled += 1;
                match page.next_cursor {
                    Some(next) => cursor = Some(next),
                    None => break,
                }
            }
            Err(e) => {
                reason_codes.insert(e.code.clone());
                read_errors.push(e);
                break;
            }
        }
    }

    if let Some(expected) = reported_count
        && (raw_records.len() as u64) != expected
    {
        reason_codes.insert("IncompleteExport".to_string());
    }

    // Deterministic per-writer canonical sort by source_id (#404 / #775).
    // Preserve duplicate detection before deduplicating.
    let mut seen_ids = BTreeSet::new();
    let mut duplicate_ids = BTreeSet::new();
    for rec in &raw_records {
        if !seen_ids.insert(rec.source_id.clone()) {
            duplicate_ids.insert(rec.source_id.clone());
        }
    }
    raw_records.sort_by(|a, b| a.source_id.cmp(&b.source_id));

    let mut id_map = BTreeMap::new();
    for rec in &raw_records {
        id_map
            .entry(rec.source_id.clone())
            .or_insert_with(|| deterministic_mem_id(connector.slug(), collection, &rec.source_id));
    }

    // Also track reverse citations so cited records count as causally linked.
    let mut cited_targets = BTreeSet::new();
    for rec in &raw_records {
        for c in &rec.cites {
            if id_map.contains_key(c) {
                cited_targets.insert(c.clone());
            }
        }
    }

    let writer_id = format!("importer:{}", connector.slug());
    let mut frames = Vec::new();
    let mut held_reasons = BTreeMap::new();
    let mut unresolved_refs = Vec::new();
    let mut ambiguous_refs = Vec::new();
    let mut secret_findings = Vec::new();
    let mut processed_ids = BTreeSet::new();
    let mut chain_head = *blake3::hash(b"strata-import-genesis").as_bytes();
    let mut linked_count = 0usize;

    for rec in raw_records {
        if !processed_ids.insert(rec.source_id.clone()) {
            held_reasons.insert(
                format!("{}#dup", rec.source_id),
                HoldReason::DuplicateSourceId,
            );
            reason_codes.insert("DuplicateSourceId".to_string());
            continue;
        }

        let mem_id = id_map[&rec.source_id].clone();
        let (clean_meta, meta_vec_stub) = strip_vectors_from_metadata(&rec.metadata);
        let vec_stub = rec
            .embedding
            .as_deref()
            .map(compute_vector_stub)
            .or(meta_vec_stub);

        let text = match &rec.text {
            Some(t) if !t.trim().is_empty() => t.clone(),
            _ => {
                held_reasons.insert(rec.source_id.clone(), HoldReason::NoTextPayload);
                reason_codes.insert("NoTextPayload".to_string());
                continue;
            }
        };

        if text.len() > MAX_TEXT_BYTES {
            held_reasons.insert(rec.source_id.clone(), HoldReason::OversizedPayload);
            reason_codes.insert("OversizedPayload".to_string());
            continue;
        }

        secret_findings.extend(scan_text_secrets(&rec.source_id, &text));

        // Exact-match commit & file binding
        let mut commit_anchor = None;
        let mut verified_files = Vec::new();
        if let Some(cref) = &rec.commit_ref {
            if let Some(git) = &opts.git_index {
                match git.resolve_commit(cref) {
                    Ok(Some(full_sha)) => {
                        for fp in &rec.file_paths {
                            if git.verify_file_at_commit(&full_sha, fp) {
                                verified_files.push(fp.replace('\\', "/"));
                            } else {
                                unresolved_refs.push(format!("{}:{fp}", rec.source_id));
                            }
                        }
                        commit_anchor = Some(full_sha);
                    }
                    Ok(None) => {
                        unresolved_refs.push(format!("{}:{cref}", rec.source_id));
                    }
                    Err("AmbiguousRef") => {
                        ambiguous_refs.push(format!("{}:{cref}", rec.source_id));
                        reason_codes.insert("AmbiguousRef".to_string());
                    }
                    Err(_) => {}
                }
            } else {
                unresolved_refs.push(format!("{}:{cref}", rec.source_id));
            }
        } else {
            for fp in &rec.file_paths {
                if fp.contains("..") || fp.starts_with('/') {
                    unresolved_refs.push(format!("{}:{fp}", rec.source_id));
                }
            }
        }

        let mut cited_mem_ids = Vec::new();
        for c in &rec.cites {
            if let Some(target_mem) = id_map.get(c) {
                cited_mem_ids.push(target_mem.clone());
            } else {
                unresolved_refs.push(format!("{}->{}", rec.source_id, c));
            }
        }

        let has_anchor = commit_anchor.is_some()
            || !verified_files.is_empty()
            || !cited_mem_ids.is_empty()
            || cited_targets.contains(&rec.source_id);

        let quarantined = if has_anchor {
            linked_count += 1;
            None
        } else {
            held_reasons.insert(rec.source_id.clone(), HoldReason::MissingProvenanceReceipt);
            reason_codes.insert("MissingProvenanceReceipt".to_string());
            Some(HoldReason::MissingProvenanceReceipt)
        };

        let metadata_json = serde_json::to_string(&clean_meta).unwrap_or_else(|_| "{}".to_string());

        let mut fh = blake3::Hasher::new();
        fh.update(&chain_head);
        fh.update(mem_id.as_bytes());
        fh.update(rec.source_id.as_bytes());
        fh.update(text.as_bytes());
        fh.update(metadata_json.as_bytes());
        if let Some(ca) = &commit_anchor {
            fh.update(ca.as_bytes());
        }
        for vf in &verified_files {
            fh.update(vf.as_bytes());
        }
        for cm in &cited_mem_ids {
            fh.update(cm.as_bytes());
        }
        if let Some(q) = &quarantined {
            fh.update(format!("{q:?}").as_bytes());
        }
        chain_head = *fh.finalize().as_bytes();
        let frame_hash = blake3::Hash::from_bytes(chain_head).to_hex().to_string();

        frames.push(ImportedFrame {
            mem_id,
            source_id: rec.source_id,
            writer_id: writer_id.clone(),
            text,
            metadata_json,
            vector_audit: vec_stub,
            commit_anchor,
            verified_files,
            cited_mem_ids,
            quarantined,
            frame_hash,
        });
    }

    let canonical_head = blake3::Hash::from_bytes(chain_head).to_hex().to_string();
    let held_count = held_reasons.len();
    let imported_count = frames.len();

    ImportRunOutput {
        frames,
        receipt: ImportReceipt {
            connector_slug: connector.slug().to_string(),
            connector_family: connector.family().code().to_string(),
            connector_version: connector.version().to_string(),
            collection: collection.to_string(),
            snapshot_digest,
            canonical_head,
            imported_count,
            linked_count,
            held_count,
            held_reasons,
            id_map,
            unresolved_refs,
            ambiguous_refs,
            reason_codes: reason_codes.into_iter().collect(),
            read_errors,
            secret_findings,
        },
        pending_checkpoint: None,
    }
}

/// Appends a reversible release frame freeing a previously quarantined record
/// once an exact causal anchor arrives (#775 section 4).
pub fn release_quarantined_frame(
    frame: &ImportedFrame,
    verified_commit_sha: String,
) -> ImportedFrame {
    let mut released = frame.clone();
    released.commit_anchor = Some(verified_commit_sha.clone());
    released.quarantined = None;
    let mut fh = blake3::Hasher::new();
    fh.update(frame.frame_hash.as_bytes());
    fh.update(b":release:");
    fh.update(verified_commit_sha.as_bytes());
    released.frame_hash = fh.finalize().to_hex().to_string();
    released
}

/// Family H / Universal JSONL & NDJSON Snapshot Connector (works for direct exports
/// and recorded fixtures across all 8 protocol families).
pub struct JsonlSnapshotConnector {
    pub store_slug: String,
    pub store_family: ConnectorFamily,
    pub path: std::path::PathBuf,
    pub page_size: usize,
    pub reported_total_override: Option<u64>,
}

impl JsonlSnapshotConnector {
    pub fn new(
        store_slug: impl Into<String>,
        store_family: ConnectorFamily,
        path: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            store_slug: store_slug.into(),
            store_family,
            path: path.into(),
            page_size: 250,
            reported_total_override: None,
        }
    }

    fn read_lines(&self) -> Result<Vec<String>, ConnectorReadError> {
        let bytes = fs::read(&self.path).map_err(|e| ConnectorReadError {
            code: "SourceReadIoError".to_string(),
            detail: e.to_string(),
        })?;
        let text = String::from_utf8(bytes).map_err(|e| ConnectorReadError {
            code: "InvalidUtf8Payload".to_string(),
            detail: e.to_string(),
        })?;
        Ok(text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect())
    }
}

impl SourceConnector for JsonlSnapshotConnector {
    fn slug(&self) -> &str {
        &self.store_slug
    }

    fn family(&self) -> ConnectorFamily {
        self.store_family
    }

    fn list_collections(&self) -> Result<Vec<String>, ConnectorReadError> {
        Ok(vec!["default".to_string()])
    }

    fn count(&self, _collection: &str) -> Result<Option<u64>, ConnectorReadError> {
        if let Some(ov) = self.reported_total_override {
            return Ok(Some(ov));
        }
        let lines = self.read_lines()?;
        Ok(Some(lines.len() as u64))
    }

    fn pull_page(
        &self,
        _collection: &str,
        cursor: Option<&str>,
    ) -> Result<crate::connector::PullPage, ConnectorReadError> {
        let lines = self.read_lines()?;
        let start: usize = cursor
            .map(|c| {
                c.parse::<usize>().map_err(|e| ConnectorReadError {
                    code: "InvalidCursor".to_string(),
                    detail: e.to_string(),
                })
            })
            .transpose()?
            .unwrap_or(0);

        let end = (start + self.page_size).min(lines.len());
        let mut records = Vec::with_capacity(end.saturating_sub(start));
        for line in &lines[start..end] {
            let rec: RawSourceRecord =
                serde_json::from_str(line).map_err(|e| ConnectorReadError {
                    code: "MalformedRecordJson".to_string(),
                    detail: e.to_string(),
                })?;
            records.push(rec);
        }
        let next_cursor = if end < lines.len() {
            Some(end.to_string())
        } else {
            None
        };
        Ok(crate::connector::PullPage {
            records,
            next_cursor,
        })
    }

    fn snapshot_digest(&self, _collection: &str) -> Result<String, ConnectorReadError> {
        let bytes = fs::read(&self.path).map_err(|e| ConnectorReadError {
            code: "SourceReadIoError".to_string(),
            detail: e.to_string(),
        })?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}
