//! # strata-import — Wave 14N Universal Importer (#775)
//!
//! Converts every vector store, database, search engine, and agent memory layer
//! into an anchored, vector-free Strata causal graph with signed receipts.

pub mod connector;
pub mod pipeline;

pub use connector::{
    ConnectorFamily, ConnectorReadError, PullPage, RawSourceRecord, STORE_CATALOG, SourceConnector,
    StoreCatalogEntry, lookup_catalog,
};
pub use pipeline::{
    GitExactIndex, HoldReason, ImportCheckpoint, ImportOptions, ImportReceipt, ImportRunOutput,
    ImportedFrame, JsonlSnapshotConnector, SecretFinding, VectorAuditStub,
    release_quarantined_frame, run_import,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn sample_git_index() -> GitExactIndex {
        let mut commits = BTreeMap::new();
        commits.insert(
            "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678".to_string(),
            BTreeSet::from(["src/auth.rs".to_string(), "README.md".to_string()]),
        );
        commits.insert(
            "a1b2c3d4ffff0000111122223333444455556666".to_string(),
            BTreeSet::from(["src/lib.rs".to_string()]),
        );
        GitExactIndex { commits }
    }

    #[test]
    fn catalog_covers_all_eight_families_with_zero_unsupported() {
        let mut families = BTreeSet::new();
        for entry in STORE_CATALOG {
            assert!(!entry.slug.is_empty());
            assert!(!entry.driver.is_empty());
            families.insert(entry.family);
        }
        assert_eq!(families.len(), 8, "all 8 protocol families must be present");
        assert!(lookup_catalog("pgvector").is_some());
        assert!(lookup_catalog("pinecone").is_some());
        assert!(lookup_catalog("qdrant").is_some());
        assert!(lookup_catalog("chroma").is_some());
        assert!(lookup_catalog("mem0").is_some());
        assert!(lookup_catalog("letta").is_some());
        assert!(lookup_catalog("faiss").is_some());
        assert!(lookup_catalog("jsonl").is_some());
    }

    #[test]
    fn deterministic_reimport_strips_vectors_quarantines_orphans_and_resumes() {
        let dir = std::env::temp_dir().join(format!("strata-import-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("export.jsonl");

        // Adversarial + mixed fixture:
        // - rec-2 (placed before rec-1 to verify canonical source_id ordering)
        // - rec-1 anchored to exact unique commit SHA prefix + valid file + vector stripped
        // - rec-3 index-only with no text -> NoTextPayload
        // - rec-4 ambiguous 8-char SHA prefix + secret -> AmbiguousRef + MissingProvenanceReceipt + secret finding
        // - rec-5 path traversal attempt `../etc/passwd` -> rejected
        let lines = [
            serde_json::json!({
                "source_id": "rec-2",
                "text": "Follow-up citing rec-1",
                "cites": ["rec-1"],
                "embedding": [0.1, 0.2, 0.3, 0.4]
            }),
            serde_json::json!({
                "source_id": "rec-1",
                "text": "Auth token verification invariant",
                "commit_ref": "a1b2c3d4e5",
                "file_paths": ["src/auth.rs", "../etc/passwd"],
                "metadata": {"embedding": [0.9, -0.1], "env": "prod"}
            }),
            serde_json::json!({
                "source_id": "rec-3",
                "text": null,
                "embedding": [0.5, 0.5]
            }),
            serde_json::json!({
                "source_id": "rec-4",
                "text": "Orphan note with leaked key sk-proj-1234567890abcdef",
                "commit_ref": "a1b2c3d4"
            }),
            serde_json::json!({
                "source_id": "rec-1",
                "text": "Duplicate id attempt"
            }),
        ];
        let body = lines
            .iter()
            .map(|v| serde_json::to_string(v).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, body).expect("write jsonl");

        let mut conn = JsonlSnapshotConnector::new("qdrant", ConnectorFamily::NativeVectorDb, &path);
        conn.page_size = 2;

        let full_run = run_import(
            &conn,
            "memories",
            ImportOptions {
                git_index: Some(sample_git_index()),
                resume_checkpoint: None,
                stop_after_pages: None,
            },
        );

        // 1. Vectors stripped completely; only BLAKE3 audit stub remains
        for f in &full_run.frames {
            assert!(!f.metadata_json.contains("0.9"));
            assert!(!f.metadata_json.contains("embedding"));
        }
        assert_eq!(full_run.frames[0].source_id, "rec-1");
        assert_eq!(
            full_run.frames[0].verified_files,
            vec!["src/auth.rs".to_string()]
        );
        assert!(full_run.frames[0].quarantined.is_none());

        // 2. Ambiguous SHA + unanchored orphan quarantined + secret detected
        assert_eq!(
            full_run.receipt.held_reasons.get("rec-3"),
            Some(&HoldReason::NoTextPayload)
        );
        assert_eq!(
            full_run.receipt.held_reasons.get("rec-4"),
            Some(&HoldReason::MissingProvenanceReceipt)
        );
        assert_eq!(
            full_run.receipt.held_reasons.get("rec-1#dup"),
            Some(&HoldReason::DuplicateSourceId)
        );
        assert!(!full_run.receipt.ambiguous_refs.is_empty());
        assert_eq!(full_run.receipt.secret_findings.len(), 1);

        // 3. Mid-run kill + checkpoint resume produces IDENTICAL canonical_head & receipt!
        let paused = run_import(
            &conn,
            "memories",
            ImportOptions {
                git_index: Some(sample_git_index()),
                resume_checkpoint: None,
                stop_after_pages: Some(1),
            },
        );
        let cp = paused.pending_checkpoint.expect("checkpoint emitted");
        let cp_file = dir.join("import.ckpt.json");
        cp.save_to_file(&cp_file).expect("save checkpoint");
        let loaded_cp = ImportCheckpoint::load_from_file(&cp_file).expect("load checkpoint");

        let resumed = run_import(
            &conn,
            "memories",
            ImportOptions {
                git_index: Some(sample_git_index()),
                resume_checkpoint: Some(loaded_cp),
                stop_after_pages: None,
            },
        );
        assert_eq!(
            full_run.receipt.canonical_head,
            resumed.receipt.canonical_head,
            "resumed import must match full import canonical_head byte-for-byte"
        );
        assert_eq!(full_run.frames, resumed.frames);

        // 4. Reversible quarantine release when anchor arrives later
        let orphan_frame = full_run
            .frames
            .iter()
            .find(|f| f.source_id == "rec-4")
            .expect("rec-4 frame");
        let released = release_quarantined_frame(
            orphan_frame,
            "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678".to_string(),
        );
        assert!(released.quarantined.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
