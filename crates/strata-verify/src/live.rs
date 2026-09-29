//! Verification of a live strata-store directory.
//!
//! Layout: `<dir>/log/*.seg` plus `<dir>/store.meta`. There is no `kernel.log`.

use std::path::Path;

use strata_gate::EventLog;
use strata_gate::record::Verdict;
use strata_store::{StrataEventLog, StrataStore};

/// Outcome of verifying a live store folder.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct LiveStoreReport {
    /// Frames in the store log.
    pub frames_total: u64,
    /// `sweep` returned no structural gaps.
    pub sweep_clear: bool,
    /// Re-derived GATE verdicts match the verdicts stored in the log.
    pub verdicts_match: bool,
    /// Human-readable detail for every failure.
    pub failures: Vec<String>,
    /// True when no check failed.
    pub ok: bool,
}

/// Verify `dir` as a live strata-store. Open replays the log and checks the
/// checkpoint chain against `store.meta` when that file is present.
pub fn verify_live_store(dir: &Path) -> Result<LiveStoreReport, String> {
    let store = StrataStore::open(dir).map_err(|e| format!("store open: {e}"))?;
    let mut failures = Vec::new();
    let frames_total = store.log().head().frames_total;
    if !store.checkpoints().is_empty() && !dir.join("store.meta").is_file() {
        failures.push("store.meta missing".into());
    }
    let gaps = store.sweep();
    let sweep_clear = gaps.is_empty();
    if !sweep_clear {
        failures.push(format!("gate sweep found {} structural gap(s)", gaps.len()));
    }
    let derived = store
        .rederive_verdicts()
        .map_err(|e| format!("rederive: {e}"))?;
    let gate_log =
        StrataEventLog::new(store.log().clone()).map_err(|e| format!("gate log: {e}"))?;
    let stored: Vec<(u64, Verdict)> = gate_log
        .events_before(u64::MAX)
        .into_iter()
        .filter_map(|ev| ev.gate().map(|g| (ev.seq, g.verdict)))
        .collect();
    let verdicts_match = derived == stored;
    if !verdicts_match {
        failures.push(format!(
            "rederived GATE verdicts differ from the log ({} rederived, {} stored)",
            derived.len(),
            stored.len()
        ));
    }
    let ok = failures.is_empty();
    Ok(LiveStoreReport {
        frames_total,
        sweep_clear,
        verdicts_match,
        failures,
        ok,
    })
}
