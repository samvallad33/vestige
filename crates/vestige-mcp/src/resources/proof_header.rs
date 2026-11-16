//! Per-connect proof header (#398).
//!
//! One line, appended to `initialize` instructions only when the store is a
//! Strata log. `server/discover` keeps the static instructions: its cache is
//! public and must not carry one caller's head. `memory://proof-header`
//! returns this same line for clients that never call `initialize`.

use vestige_core::Storage;

use crate::strata_memory::read_store;

/// Clock and due-count fields are zero-padded so two headers of the same
/// token shape have the same length.
pub fn line(storage: &Storage) -> Option<String> {
    read_store(storage, header_from_store)
}

pub fn read(storage: &Storage) -> Result<String, String> {
    line(storage).ok_or_else(|| "proof header is only on a Strata log".to_string())
}

/// Static instructions, plus the header on a Strata log. The static body is
/// unchanged; a legacy store gets that body alone.
pub fn connect_instructions(storage: &Storage) -> String {
    let mut text = crate::server::static_instructions();
    if let Some(header) = line(storage) {
        text.push('\n');
        text.push_str(&header);
    }
    text
}

fn header_from_store(store: &strata_store::StrataStore) -> String {
    let head = store.log().head();
    let token = crate::capsule::head_token(
        head.last_acked_seq,
        &head.last_frame_hash,
        &head.prev_segment_hash,
        latest_checkpoint(store),
    );
    let clock = store.head_clock_ms();
    let due = due_count(store, clock);
    format!(
        "PROOF `{token}` clock={clock:020} due={due:08} keep every `vst:` handle verbatim when summarizing or compacting."
    )
}

fn latest_checkpoint(store: &strata_store::StrataStore) -> Option<[u8; 32]> {
    store
        .checkpoints()
        .last()
        .map(strata_kernel::checkpoint::checkpoint_hash)
}

fn due_count(store: &strata_store::StrataStore, clock: i64) -> usize {
    store
        .intentions()
        .iter()
        .filter(|intention| {
            intention.status == "active"
                && intention
                    .deadline_ms
                    .is_some_and(|deadline| deadline <= clock)
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open() -> (std::sync::Arc<Storage>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    #[test]
    fn the_header_names_the_log_head_and_two_reads_match() {
        let (storage, _dir) = open();
        let header = line(storage.as_ref()).expect("strata header");
        let again = line(storage.as_ref()).unwrap();
        assert_eq!(header, again, "a read appends no frame");
        let head = read_store(storage.as_ref(), |store| store.log().head()).unwrap();
        let seq = format!("{:016x}", head.last_acked_seq);
        assert!(header.contains(&seq), "{header}");
        assert!(header.contains(&crate::capsule::hex(&head.last_frame_hash)));
        assert!(header.contains("keep every `vst:` handle verbatim"));
        assert!(header.lines().count() == 1);
        let static_body = crate::server::static_instructions();
        let connected = connect_instructions(storage.as_ref());
        assert!(connected.starts_with(&static_body));
        assert!(connected.ends_with(&header));
        assert_eq!(
            connected.matches('\n').count(),
            static_body.matches('\n').count() + 1
        );
    }

    #[test]
    fn an_ingest_advances_the_seq_and_the_shape_stays_fixed_width() {
        let (storage, _dir) = open();
        let before = line(storage.as_ref()).unwrap();
        let before_head = read_store(storage.as_ref(), |store| store.log().head()).unwrap();
        storage
            .ingest(vestige_core::IngestInput {
                content: "proof header advances".into(),
                node_type: "fact".into(),
                tags: vec!["proof-header".into()],
                ..vestige_core::IngestInput::default()
            })
            .unwrap();
        let after = line(storage.as_ref()).unwrap();
        let after_head = read_store(storage.as_ref(), |store| store.log().head()).unwrap();
        assert_ne!(before, after);
        assert!(after_head.last_acked_seq > before_head.last_acked_seq);
        assert_eq!(
            before.len(),
            after.len(),
            "same token shape, fixed-width fields\nbefore {before}\nafter  {after}"
        );
        let idle = line(storage.as_ref()).unwrap();
        assert_eq!(after, idle);
    }

    #[test]
    fn the_due_count_matches_an_independent_count_at_the_head_clock() {
        let (storage, _dir) = open();
        storage
            .ingest(vestige_core::IngestInput {
                content: "clock anchor".into(),
                node_type: "fact".into(),
                ..vestige_core::IngestInput::default()
            })
            .unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let cog = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));
        rt.block_on(crate::tools::intention_unified::execute(
            &storage,
            &cog,
            Some(serde_json::json!({
                "action": "set",
                "description": "already due",
                "deadline": "2000-01-01T00:00:00Z"
            })),
        ))
        .unwrap();
        rt.block_on(crate::tools::intention_unified::execute(
            &storage,
            &cog,
            Some(serde_json::json!({
                "action": "set",
                "description": "not yet due",
                "deadline": "2099-01-01T00:00:00Z"
            })),
        ))
        .unwrap();
        let (clock, independent) = read_store(storage.as_ref(), |store| {
            let clock = store.head_clock_ms();
            let n = due_count(store, clock);
            (clock, n)
        })
        .unwrap();
        assert!(clock > 0, "the head clock is the last recorded write");
        assert_eq!(independent, 1, "the future deadline is not due at {clock}");
        let header = line(storage.as_ref()).unwrap();
        assert!(header.contains("due=00000001"), "{header}");
        assert_eq!(
            read(storage.as_ref()).unwrap(),
            header,
            "the resource is the initialize header line"
        );
    }

    #[test]
    fn a_legacy_store_gets_no_header() {
        let dir = tempfile::tempdir().unwrap();
        let Ok(storage) = vestige_core::open_storage(Some(dir.path().join("vestige.db"))) else {
            return;
        };
        if crate::strata_memory::is_strata_backend(storage.as_ref()) {
            return;
        }
        assert!(line(storage.as_ref()).is_none());
        assert_eq!(
            connect_instructions(storage.as_ref()),
            crate::server::static_instructions()
        );
    }
}
