//! Strata journal for `intention` `action=graph`.
//!
//! Each committed command and the scope snapshot are intention rows admitted
//! by [`strata_store::StrataStore::upsert_intentions`]
//! (`StoreOp::UpsertIntentions`). One check of the journal is one admitted
//! batch. Replay applies those recorded commands in order. It does not invent
//! a link from keywords, entity names, embeddings, or cosine similarity.
//! A memory commitment is an exact id lookup.

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use strata_store::VALID_FOREVER_MS;
use vestige_core::intention_graph::{Command, IntentionGraph};

/// `source_type` on journal and snapshot rows. Prospective readers skip it.
pub(crate) const SOURCE: &str = "intention_graph";

const JOURNAL_KIND: &str = "graph_journal";
const STATE_KIND: &str = "graph_state";
const RECORDED: &str = "recorded";
const MAX_GRAPH_BYTES: usize = 16 * 1024 * 1024;
const MAX_COMMAND_BYTES: usize = 128 * 1024;
const MAX_JOURNAL_ENTRIES: i64 = 20_000;

#[derive(Debug, Deserialize)]
struct JournalMeta {
    seq: i64,
    output_digest: String,
    state_digest: String,
    evaluated_at: String,
}

fn storage_error(error: impl std::fmt::Display) -> String {
    format!("Intention graph storage: {error}")
}

fn digest(value: &str) -> String {
    Sha256::digest(value.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_scope(scope: &str) -> Result<(), String> {
    if scope.is_empty()
        || scope.len() > 128
        || !scope
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"._-/".contains(&c))
    {
        return Err(
            "scope must contain 1..128 ASCII letters, digits, '.', '_', '-', or '/'".into(),
        );
    }
    Ok(())
}

fn require_memory_id(memory_id: &str) -> Result<(), String> {
    let mem = memory_id
        .strip_prefix("mem-")
        .is_some_and(|rest| rest.len() == 16 && rest.bytes().all(|b| b.is_ascii_hexdigit()));
    if mem || uuid::Uuid::parse_str(memory_id).is_ok() {
        return Ok(());
    }
    Err("memory_id must be a mem- id or UUID".into())
}

fn effective_scope(scope: &str) -> &str {
    let trimmed = scope.trim();
    if trimmed.is_empty() { "user" } else { trimmed }
}

fn journal_id(scope: &str, seq: i64) -> String {
    format!("igj|{scope}|{seq:020}")
}

fn snapshot_id(scope: &str) -> String {
    format!("igs|{scope}")
}

fn blank_intention(id: String, scope: &str, now: DateTime<Utc>) -> strata_store::IntentionRecord {
    strata_store::IntentionRecord {
        id,
        content: String::new(),
        trigger_type: String::new(),
        trigger_data: String::new(),
        priority: 0,
        status: RECORDED.to_string(),
        created_at_ms: now.timestamp_millis(),
        deadline_ms: None,
        fulfilled_at_ms: None,
        reminder_count: 0,
        last_reminded_at_ms: None,
        notes: None,
        tags: Vec::new(),
        related_memories: Vec::new(),
        snoozed_until_ms: None,
        source_type: SOURCE.to_string(),
        source_data: None,
        scope: Some(scope.to_string()),
    }
}

fn load_journal(
    store: &strata_store::StrataStore,
    scope: &str,
) -> Result<Vec<strata_store::IntentionRecord>, String> {
    let prefix = format!("igj|{scope}|");
    let mut rows: Vec<_> = store
        .intentions()
        .into_iter()
        .filter(|row| row.id.starts_with(&prefix))
        .collect();
    rows.sort_by(|left, right| left.id.cmp(&right.id));
    for (index, row) in rows.iter().enumerate() {
        let seq = i64::try_from(index).map_err(storage_error)? + 1;
        if row.id != journal_id(scope, seq)
            || row.source_type != SOURCE
            || row.trigger_type != JOURNAL_KIND
        {
            return Err("intention graph journal sequence is invalid".into());
        }
    }
    if i64::try_from(rows.len()).unwrap_or(i64::MAX) > MAX_JOURNAL_ENTRIES {
        return Err("intention graph journal sequence is invalid".into());
    }
    Ok(rows)
}

fn load_graph(store: &strata_store::StrataStore, scope: &str) -> Result<IntentionGraph, String> {
    let Some(row) = store.get_intention(&snapshot_id(scope)) else {
        return Ok(IntentionGraph::default());
    };
    if row.source_type != SOURCE || row.trigger_type != STATE_KIND {
        return Err("intention graph snapshot is not a recorded graph state".into());
    }
    serde_json::from_str(&row.content).map_err(storage_error)
}

fn journal_limit() -> String {
    "intention graph journal limit reached; preserve this scope and create a new scope with reviewed plan definitions".into()
}

/// Apply one graph command. A state change appends the command and replaces
/// the snapshot in a single `UpsertIntentions` batch.
pub(crate) fn apply(
    store: &mut strata_store::StrataStore,
    scope: &str,
    command: Command,
    now: DateTime<Utc>,
) -> Result<Value, String> {
    validate_scope(scope)?;
    let command_json = serde_json::to_string(&command).map_err(storage_error)?;
    if command_json.len() > MAX_COMMAND_BYTES {
        return Err("intention graph command exceeds 128 KiB".into());
    }
    let prior = load_journal(store, scope)?;
    let mut graph = load_graph(store, scope)?;
    let before = serde_json::to_string(&graph).map_err(storage_error)?;
    let output = graph.apply(command, now)?;
    let after = serde_json::to_string(&graph).map_err(storage_error)?;
    if after.len() > MAX_GRAPH_BYTES {
        return Err("intention graph exceeds the 16 MiB local scope limit".into());
    }
    let mut committed_seq = None;
    if before != after {
        let seq = i64::try_from(prior.len()).map_err(storage_error)? + 1;
        if seq > MAX_JOURNAL_ENTRIES {
            return Err(journal_limit());
        }
        let output_json = serde_json::to_string(&output).map_err(storage_error)?;
        let meta = json!({
            "seq": seq,
            "output_digest": digest(&output_json),
            "state_digest": digest(&after),
            "evaluated_at": now.to_rfc3339(),
        });
        let mut journal = blank_intention(journal_id(scope, seq), scope, now);
        journal.content = command_json;
        journal.trigger_type = JOURNAL_KIND.to_string();
        journal.trigger_data = meta.to_string();
        let mut snapshot = blank_intention(snapshot_id(scope), scope, now);
        snapshot.content = after;
        snapshot.trigger_type = STATE_KIND.to_string();
        snapshot.trigger_data = json!({"updated_at": now.to_rfc3339()}).to_string();
        store
            .upsert_intentions(vec![journal, snapshot])
            .map_err(storage_error)?;
        committed_seq = Some(seq);
    }
    Ok(json!({
        "scope": scope,
        "journal_seq": committed_seq,
        "result": output,
        "boundary": "Local deterministic evaluation of supplied evidence; no external action or independent fact verification.",
    }))
}

/// Rebuild one scope from the recorded command journal and compare digests.
pub(crate) fn replay(store: &strata_store::StrataStore, scope: &str) -> Result<Value, String> {
    validate_scope(scope)?;
    let rows = load_journal(store, scope)?;
    let mut graph = IntentionGraph::default();
    let mut count = 0_i64;
    for row in &rows {
        count += 1;
        let meta: JournalMeta = serde_json::from_str(&row.trigger_data).map_err(storage_error)?;
        if meta.seq != count || meta.seq > MAX_JOURNAL_ENTRIES {
            return Err("intention graph journal sequence is invalid".into());
        }
        let command: Command = serde_json::from_str(&row.content).map_err(storage_error)?;
        let now = DateTime::parse_from_rfc3339(&meta.evaluated_at)
            .map_err(storage_error)?
            .with_timezone(&Utc);
        let output = graph.apply(command, now)?;
        let actual_output = digest(&serde_json::to_string(&output).map_err(storage_error)?);
        let actual_state = digest(&serde_json::to_string(&graph).map_err(storage_error)?);
        if actual_output != meta.output_digest || actual_state != meta.state_digest {
            return Err(format!(
                "intention graph replay diverged at sequence {count}"
            ));
        }
    }
    let replayed = serde_json::to_string(&graph).map_err(storage_error)?;
    let stored = store.get_intention(&snapshot_id(scope));
    if let Some(row) = &stored {
        if row.source_type != SOURCE || row.trigger_type != STATE_KIND || row.content != replayed {
            return Err("intention graph replay differs from the current snapshot".into());
        }
    } else if count != 0 {
        return Err("intention graph replay differs from the current snapshot".into());
    }
    Ok(json!({
        "scope": scope,
        "matched": true,
        "commands": count,
        "state_digest": digest(&replayed),
        "boundary": "Deterministic local replay, not proof of external facts, delivery, or cryptographic authenticity.",
    }))
}

/// Exact-id content commitment. The memory text is not copied into the result.
pub(crate) fn memory_snapshot(
    store: &strata_store::StrataStore,
    scope: &str,
    memory_id: &str,
    now: DateTime<Utc>,
) -> Result<Value, String> {
    validate_scope(scope)?;
    require_memory_id(memory_id)?;
    let now_ms = now.timestamp_millis();
    let value = store.get_node(memory_id).and_then(|record| {
        let in_scope = effective_scope(&record.scope) == scope;
        let live = record.superseded_by.is_none();
        let started = record.valid_from_ms <= now_ms;
        let open = record.valid_until_ms == VALID_FOREVER_MS || record.valid_until_ms > now_ms;
        (in_scope && live && started && open).then(|| digest(&record.content))
    });
    Ok(json!({
        "source_key": format!("memory:{memory_id}"),
        "memory_id": memory_id,
        "value": value,
        "observed_at": now,
        "boundary": "Local scoped content commitment; not external fact verification.",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use strata_store::IngestInput;

    fn at() -> DateTime<Utc> {
        "2026-10-01T09:00:00Z".parse().unwrap()
    }

    fn plan(id: &str) -> Command {
        serde_json::from_value(json!({
            "action": "plan",
            "id": id,
            "description": "Synthetic graph plan",
            "requirements": [],
            "conflict_keys": []
        }))
        .unwrap()
    }

    #[test]
    fn graph_journal_replays_across_reopen_and_rejects_a_bad_scope() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
            assert!(apply(&mut store, "", plan("p1"), at()).is_err());
            assert!(store.intentions().is_empty());
            let written = apply(&mut store, "user", plan("p1"), at()).unwrap();
            assert!(written["journal_seq"].as_i64().unwrap() >= 1);
            let again = apply(&mut store, "user", plan("p1"), at()).unwrap_err();
            assert!(again.contains("already exists"), "{again}");
            assert_eq!(replay(&store, "user").unwrap()["commands"], 1);
            assert_eq!(replay(&store, "other").unwrap()["commands"], 0);
        }
        let store = strata_store::StrataStore::open(dir.path()).unwrap();
        let replayed = replay(&store, "user").unwrap();
        assert_eq!(replayed["matched"], true);
        assert_eq!(replayed["commands"], 1);
        assert_eq!(replayed["state_digest"].as_str().unwrap().len(), 64);
    }

    #[test]
    fn graph_replay_detects_a_replaced_digest() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
        apply(&mut store, "user", plan("p1"), at()).unwrap();
        let id = journal_id("user", 1);
        let mut row = store.get_intention(&id).unwrap();
        let mut meta: Value = serde_json::from_str(&row.trigger_data).unwrap();
        meta["output_digest"] = json!("changed");
        row.trigger_data = meta.to_string();
        store.upsert_intentions(vec![row]).unwrap();
        let err = replay(&store, "user").unwrap_err();
        assert!(err.contains("diverged"), "{err}");
    }

    #[test]
    fn memory_snapshot_is_an_exact_id_commitment() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
        let id = store
            .ingest_in_scope(
                IngestInput {
                    content: "Synthetic projector requirement alpha".into(),
                    ..IngestInput::default()
                },
                "project-a",
            )
            .unwrap();
        let first = memory_snapshot(&store, "project-a", &id, at()).unwrap();
        assert_eq!(first["value"].as_str().unwrap().len(), 64);
        assert!(!first.to_string().contains("Synthetic"));
        assert!(memory_snapshot(&store, "project-b", &id, at()).unwrap()["value"].is_null());
        assert!(memory_snapshot(&store, "project-a", "not-an-id", at()).is_err());
    }
}
