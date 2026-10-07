//! The two preregistered corpora.
//!
//! Both are built in-process through the store's recorded operations and
//! then opened as a copy. Content strings are unread labels. No production
//! database is copied.

use std::collections::BTreeMap;

use strata_store::{AnchorRecord, ConnectionRecord, IngestInput, StoreError, StrataStore};
use tempfile::TempDir;

use crate::events::Rec;
use crate::mechanism::admitted_events;
use crate::open::{CopiedStore, open_copy};
use crate::protocol::{
    CLOCK_ORIGIN_MS, CLOCK_STEP_MS, RECORDED_CORPUS, RECORDED_SCOPE, SYNTH_CORPUS, SYNTH_SCOPE,
};
use crate::rng::log_seed;

/// A seeded log, opened only through a copy.
pub struct LoadedCorpus {
    /// Corpus id.
    pub id: &'static str,
    /// Copy the harness reads. The writer has already been dropped.
    pub copy: CopiedStore,
    /// Events of the full log, in frame order.
    pub events: Vec<Rec>,
    _origin: TempDir,
}

/// Build `synth-track-v1` or `recorded-ops-v1`.
pub fn load_corpus(id: &str) -> Result<LoadedCorpus, StoreError> {
    match id {
        SYNTH_CORPUS => load(SYNTH_CORPUS, build_synth),
        RECORDED_CORPUS => load(RECORDED_CORPUS, build_recorded),
        other => Err(StoreError::InvalidInput(format!("unknown corpus {other}"))),
    }
}

fn load(
    id: &'static str,
    build: fn(&mut StrataStore) -> Result<(), StoreError>,
) -> Result<LoadedCorpus, StoreError> {
    let origin = tempfile::tempdir().map_err(StoreError::Io)?;
    {
        let mut store = StrataStore::open_seeded(origin.path(), log_seed(id))?;
        build(&mut store)?;
    }
    let copy = open_copy(origin.path())?;
    let events = admitted_events(&copy.store)?;
    Ok(LoadedCorpus {
        id,
        copy,
        events,
        _origin: origin,
    })
}

fn build_synth(store: &mut StrataStore) -> Result<(), StoreError> {
    let mut w = Script::new(store, SYNTH_SCOPE);
    w.ingest("dead", "fact", &[])?;
    for i in 0..8 {
        let d = format!("d{i}");
        let p = format!("p{i}");
        w.ingest(&d, "fact", &[])?;
        w.ingest(&p, "fact", &[])?;
        w.edge(&d, &p, "derived_from")?;
        w.edge(&d, "dead", "derived_from")?;
        w.review(&p, 4)?;
    }
    for label in ["hub", "a1", "a2", "a3", "a4", "a5"] {
        w.ingest(label, "fact", &[])?;
    }
    let chain = [
        ("hub", "a1"),
        ("a1", "a2"),
        ("a2", "a3"),
        ("a3", "a4"),
        ("a4", "a5"),
        ("a5", "hub"),
    ];
    for (from, to) in chain {
        w.edge(from, to, "derived_from")?;
    }
    for label in ["hub", "a1", "a2", "a3", "a4", "a5"] {
        w.edge(label, "dead", "derived_from")?;
    }
    w.review("a5", 4)?;
    for _ in 0..3 {
        episode(&mut w, &["hub", "a1", "a2", "a3", "a4", "a5"], 3)?;
    }
    episode(&mut w, &["hub", "a1", "a2", "a3", "a4", "a5"], 3)?;
    Ok(())
}

fn build_recorded(store: &mut StrataStore) -> Result<(), StoreError> {
    let mut w = Script::new(store, RECORDED_SCOPE);
    w.ingest("dead", "fact", &[])?;
    for i in 0..6 {
        let d = format!("d{i}");
        let p = format!("p{i}");
        w.ingest(&d, "fact", &[])?;
        w.ingest(&p, "fact", &[])?;
        w.edge(&d, &p, "derived_from")?;
        w.edge(&d, "dead", "derived_from")?;
        w.review(&p, 4)?;
    }
    for label in ["failure", "lesson", "fix"] {
        w.ingest(label, "fact", &[])?;
    }
    for (from, to) in [
        ("failure", "lesson"),
        ("failure", "dead"),
        ("lesson", "fix"),
        ("fix", "failure"),
        ("lesson", "dead"),
        ("fix", "dead"),
    ] {
        w.edge(from, to, "derived_from")?;
    }
    w.review("fix", 4)?;
    w.ingest("w1", "fact", &[])?;
    w.ingest("w2", "fact", &[])?;
    w.ingest(
        "weave",
        "composition",
        &["ghostlink", "ghostlink-weave", "outcome:dead_end"],
    )?;
    w.edge("weave", "w1", "derived_from")?;
    w.edge("weave", "w2", "derived_from")?;
    w.ingest("pen", "fact", &[])?;
    w.ingest("scrap", "fact", &[])?;
    w.edge("pen", "scrap", "corrects")?;
    w.anchors(
        "fix",
        &[
            ("anc-attach", "src/attach.rs"),
            ("anc-receipt", "src/receipt.rs"),
        ],
    )?;
    for _ in 0..3 {
        episode(&mut w, &["failure", "lesson", "fix"], 3)?;
    }
    episode(&mut w, &["failure", "lesson", "fix"], 3)?;
    w.anchors("fix", &[("anc-receipt-suffix", "src/receipt.rs")])?;
    Ok(())
}

fn episode(w: &mut Script<'_>, labels: &[&str], rating: u8) -> Result<(), StoreError> {
    for label in labels {
        w.review(label, rating)?;
    }
    Ok(())
}

struct Script<'a> {
    store: &'a mut StrataStore,
    scope: &'static str,
    tick: i64,
    ids: BTreeMap<String, String>,
}

impl<'a> Script<'a> {
    fn new(store: &'a mut StrataStore, scope: &'static str) -> Self {
        Self {
            store,
            scope,
            tick: 0,
            ids: BTreeMap::new(),
        }
    }

    fn clock(&mut self) -> i64 {
        let ms = CLOCK_ORIGIN_MS + self.tick * CLOCK_STEP_MS;
        self.tick += 1;
        ms
    }

    fn id(&self, label: &str) -> Result<String, StoreError> {
        self.ids.get(label).cloned().ok_or_else(|| {
            StoreError::InvalidInput(format!("corpus label {label} was not ingested"))
        })
    }

    fn ingest(&mut self, label: &str, node_type: &str, tags: &[&str]) -> Result<(), StoreError> {
        let created_at_ms = self.clock();
        let id = self.store.ingest_in_scope(
            IngestInput {
                content: label.to_string(),
                node_type: node_type.to_string(),
                tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
                created_at_ms: Some(created_at_ms),
                ..IngestInput::default()
            },
            self.scope,
        )?;
        self.ids.insert(label.to_string(), id);
        Ok(())
    }

    fn edge(&mut self, from: &str, to: &str, link: &str) -> Result<(), StoreError> {
        let created_at_ms = self.clock();
        let source_id = self.id(from)?;
        let target_id = self.id(to)?;
        self.store.save_connection(&ConnectionRecord {
            source_id,
            target_id,
            strength_milli: 1000,
            link_type: link.to_string(),
            meta_sha: None,
            created_at_ms,
            activation_count: 0,
        })?;
        Ok(())
    }

    fn review(&mut self, label: &str, rating: u8) -> Result<(), StoreError> {
        let reviewed_at_ms = self.clock();
        let id = self.id(label)?;
        self.store.review_at(&id, rating, Some(reviewed_at_ms))?;
        Ok(())
    }

    fn anchors(&mut self, label: &str, rows: &[(&str, &str)]) -> Result<(), StoreError> {
        let captured_at_ms = self.clock();
        let node_id = self.id(label)?;
        let anchors = rows
            .iter()
            .map(|(id, file_path)| AnchorRecord {
                id: (*id).to_string(),
                node_id: node_id.clone(),
                file_path: (*file_path).to_string(),
                symbol: None,
                symbol_kind: None,
                start_line: None,
                end_line: None,
                span_lines: None,
                content_hash: None,
                captured_at_ms,
                last_verified_at_ms: None,
                last_status: None,
            })
            .collect();
        self.store.record_anchors(anchors)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Body;

    fn counts(events: &[Rec]) -> (usize, usize, usize, usize) {
        let mut upserts = 0;
        let mut edges = 0;
        let mut reviews = 0;
        let mut anchors = 0;
        for rec in events {
            match rec.body {
                Body::Upsert { .. } => upserts += 1,
                Body::Edge { .. } => edges += 1,
                Body::Review { .. } => reviews += 1,
                Body::Anchor { .. } => anchors += 1,
            }
        }
        (upserts, edges, reviews, anchors)
    }

    #[test]
    fn synth_script_matches_section_8_1() {
        let loaded = load_corpus(SYNTH_CORPUS).unwrap();
        assert_eq!(counts(&loaded.events), (23, 28, 33, 0));
        assert!(loaded.events.iter().all(|rec| !matches!(
            &rec.body,
            Body::Edge { link, .. } if link == "closed_by" || link == "corrects"
        )));
    }

    #[test]
    fn recorded_script_matches_section_8_2() {
        let loaded = load_corpus(RECORDED_CORPUS).unwrap();
        assert_eq!(counts(&loaded.events), (21, 21, 19, 3));
        let corrects = loaded
            .events
            .iter()
            .filter(|rec| matches!(&rec.body, Body::Edge { link, .. } if link == "corrects"))
            .count();
        assert_eq!(corrects, 1);
        let paths: Vec<&str> = loaded
            .events
            .iter()
            .filter_map(|rec| match &rec.body {
                Body::Anchor { path, .. } => Some(path.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            paths,
            vec!["src/attach.rs", "src/receipt.rs", "src/receipt.rs"]
        );
    }
}
