use crate::arms;
use crate::git::git_head;
use crate::types::{Arm, FailuresFile, LOOKBACK_DAYS, RunOutput, dataset_id_for};
use anyhow::{Context, Result};
use chrono::Utc;
use std::fs;
use std::path::{Path, PathBuf};
use vestige_core::Storage;

pub fn run(store: &Path, failures: &Path, arm: Arm, out: Option<&Path>) -> Result<RunOutput> {
    if !store.exists() {
        anyhow::bail!("store does not exist: {store:?}");
    }
    let failures_doc: FailuresFile = serde_json::from_slice(
        &fs::read(failures).with_context(|| format!("read failures {failures:?}"))?,
    )?;
    let dataset_id = if failures_doc.dataset_id.is_empty() {
        dataset_id_for(&failures_doc.failure_ids)
    } else {
        failures_doc.dataset_id.clone()
    };
    let scratch = scratch_copy(store, arm)?;
    let storage =
        Storage::new(Some(scratch.clone())).with_context(|| format!("open scratch {scratch:?}"))?;
    let started_at = Utc::now();
    let executed = arms::run_arm(arm, &storage, &failures_doc.failure_ids)?;
    let finished_at = Utc::now();
    let lookback = match arm {
        Arm::Lexical | Arm::LexicalOr => None,
        Arm::Backfill | Arm::CausalGraph => Some(LOOKBACK_DAYS),
    };
    let output = RunOutput {
        arm,
        store: store.display().to_string(),
        store_id: failures_doc.store_id,
        commit: git_head(),
        search_mode: executed.search_mode,
        lookback_days: lookback,
        top_k: crate::types::TOP_K,
        started_at,
        finished_at,
        queries: executed.queries,
        accumulation_ms: executed.accumulation_ms,
        dataset_id,
        scratch_store: Some(scratch.display().to_string()),
        embedding_ready: storage.is_embedding_ready(),
    };
    let encoded = serde_json::to_vec_pretty(&output)?;
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &encoded)?;
    } else {
        println!("{}", String::from_utf8_lossy(&encoded));
    }
    Ok(output)
}

fn scratch_copy(store: &Path, arm: Arm) -> Result<PathBuf> {
    let parent = store.parent().unwrap_or(Path::new("."));
    let stem = store
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("vestige");
    let dest = parent.join(format!("{stem}.{}.scratch.db", arm.as_str()));
    fs::copy(store, &dest).with_context(|| format!("copy {store:?} → {dest:?}"))?;
    for suffix in ["-wal", "-shm"] {
        let src = PathBuf::from(format!("{}{suffix}", store.display()));
        if src.exists() {
            let dst = PathBuf::from(format!("{}{suffix}", dest.display()));
            fs::copy(&src, &dst)?;
        }
    }
    Ok(dest)
}
