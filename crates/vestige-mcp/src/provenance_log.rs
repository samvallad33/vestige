//! #774 quarantine records carried by the same admitted frames mesh sync folds.
//!
//! An orphan is an `UpsertNode` whose content is the canonical quarantine body.
//! A release is a later `UpsertNode`. The first node stays in the log.
//! `fold_chains` already hashes both, so a loopback sync lands both devices
//! on one digest.

#![forbid(unsafe_code)]

use serde_json::{Value, json};
use strata_store::{IngestInput, StrataStore};
use vestige_core::Storage;
use vestige_core::provenance::{
    decode_hex_32, encode_hex, project_provenance_bodies, quarantine_body, release_body,
};

use crate::strata_memory::with_store;

/// Append the orphan body when this store does not already hold it.
pub fn quarantine(storage: &Storage, summary: &str) -> Result<Value, String> {
    with_store(storage, |store| quarantine_store(store, summary))?
}

/// Append the anchor frame when this store does not already hold it.
pub fn release(storage: &Storage, receipt_hex: &str, anchor_hex: &str) -> Result<Value, String> {
    with_store(storage, |store| {
        release_store(store, receipt_hex, anchor_hex)
    })?
}

/// Quarantine projection plus the mesh digest. Reads the log and opens no socket.
pub fn status(storage: &Storage) -> Result<Value, String> {
    with_store(storage, |store| status_store(store))
}

/// [`quarantine`] against an already-open store.
pub fn quarantine_store(store: &mut StrataStore, summary: &str) -> Result<Value, String> {
    let (body, receipt) = quarantine_body("", summary)?;
    let appended = ingest_if_absent(store, &body, &["provenance:quarantine".to_string()])?;
    Ok(outcome(
        "quarantined",
        appended,
        &receipt.receipt_digest,
        store,
    ))
}

/// [`release`] against an already-open store.
pub fn release_store(
    store: &mut StrataStore,
    receipt_hex: &str,
    anchor_hex: &str,
) -> Result<Value, String> {
    let receipt_digest = decode_hex_32(receipt_hex).ok_or("receipt must be 64 hex characters")?;
    let anchor_digest = decode_hex_32(anchor_hex).ok_or("anchor must be 64 hex characters")?;
    let known = project_provenance_bodies(&live_bodies(store));
    if !known
        .retained
        .iter()
        .any(|record| record.receipt.receipt_digest == receipt_digest)
    {
        return Err("no quarantined memory has that receipt".to_string());
    }
    let marker = release_body(receipt_digest, anchor_digest);
    let appended = ingest_if_absent(store, &marker, &["provenance:release".to_string()])?;
    if appended {
        let summary = known
            .retained
            .iter()
            .find(|record| record.receipt.receipt_digest == receipt_digest)
            .map(|record| record.summary.clone())
            .ok_or("quarantine record disappeared before the release frame")?;
        ingest_if_absent(
            store,
            &summary,
            &[
                format!("anchor:{}", encode_hex(&anchor_digest)),
                "provenance:released".to_string(),
            ],
        )?;
    }
    Ok(outcome("released", appended, &receipt_digest, store))
}

/// [`status`] against an already-open store.
pub fn status_store(store: &StrataStore) -> Value {
    let view = project_provenance_bodies(&live_bodies(store));
    let quarantined: Vec<Value> = view
        .quarantined
        .iter()
        .map(|record| {
            json!({
                "receipt_digest": encode_hex(&record.receipt.receipt_digest),
                "summary_digest": encode_hex(&record.receipt.summary_digest),
                "summary": record.summary,
                "reason": record.receipt.reason,
            })
        })
        .collect();
    let released: Vec<Value> = view
        .released
        .iter()
        .map(|record| {
            json!({
                "receipt_digest": encode_hex(&record.receipt_digest),
                "anchor_digest": encode_hex(&record.anchor_digest),
                "summary": record.summary,
            })
        })
        .collect();
    json!({
        "mesh_digest": encode_hex(&store.mesh_digest()),
        "quarantined": quarantined,
        "released": released,
        "retained_quarantine_bodies": view.retained.len(),
    })
}

fn outcome(state: &str, appended: bool, receipt: &[u8; 32], store: &StrataStore) -> Value {
    json!({
        "state": state,
        "appended": appended,
        "receipt_digest": encode_hex(receipt),
        "mesh_digest": encode_hex(&store.mesh_digest()),
    })
}

fn live_bodies(store: &StrataStore) -> Vec<String> {
    store.mesh_upsert_contents()
}

fn ingest_if_absent(
    store: &mut StrataStore,
    content: &str,
    tags: &[String],
) -> Result<bool, String> {
    if store
        .nodes()
        .iter()
        .any(|node| node.is_live() && node.content == content)
    {
        return Ok(false);
    }
    store
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                node_type: "provenance".to_string(),
                tags: tags.to_vec(),
                created_at_ms: Some(0),
                ..IngestInput::default()
            },
            "",
        )
        .map_err(|err| err.to_string())?;
    Ok(true)
}
