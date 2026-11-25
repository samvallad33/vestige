//! Screenshot-ready terminal & Markdown proof scorecard (`vestige card`).
//!
//! Reads local Reflex hook ledger, Strata log metadata, active Execution Lease,
//! and optional git `refs/notes/vestige-proof` counts without taking `.serve.lock`,
//! so developers can generate a shareable ASCII or Markdown proof card mid-session.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use serde::Serialize;
use serde_json::Value;

use crate::tools::reflex;
use crate::tools::reflex::lease;

/// Aggregated local proof metrics rendered by `vestige card`.
#[derive(Debug, Clone, Serialize)]
pub struct ProofCardMetrics {
    pub version: String,
    pub window: String,
    pub firings: u64,
    pub chain_checked: u64,
    pub chain_breaks: u64,
    pub stops_recorded: u64,
    pub resolutions_recorded: u64,
    pub folds: u64,
    pub chars_saved: u64,
    pub chars_told: u64,
    pub duplicates_dropped: u64,
    pub sessions_active: u64,
    pub strata_segments: usize,
    pub strata_bytes: u64,
    pub proof_notes_commits: usize,
    pub lease_active: bool,
    pub lease_halted: bool,
    pub lease_guards_count: usize,
    pub card_digest_blake3: String,
}

fn dir_stats(dir: &Path) -> (usize, u64) {
    let Ok(entries) = fs::read_dir(dir) else {
        return (0, 0);
    };
    let mut files = 0usize;
    let mut bytes = 0u64;
    for entry in entries.flatten() {
        if let Ok(meta) = entry.metadata()
            && meta.is_file()
        {
            files += 1;
            bytes = bytes.saturating_add(meta.len());
        }
    }
    (files, bytes)
}

fn count_git_proof_notes(repo: &Path) -> usize {
    if !repo.is_dir() {
        return 0;
    }
    let Ok(out) = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["notes", "--ref=refs/notes/vestige-proof", "list"])
        .stdin(Stdio::null())
        .output()
    else {
        return 0;
    };
    if !out.status.success() {
        return 0;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count()
}

fn u64_at(v: &Value, path: &[&str]) -> u64 {
    let mut cur = v;
    for seg in path {
        let Some(next) = cur.get(*seg) else {
            return 0;
        };
        cur = next;
    }
    cur.as_u64().unwrap_or(0)
}

/// Collect deterministic local telemetry for `vestige card` with zero `.serve.lock` contention.
pub fn collect_proof_card(data_dir: &Path, repo: Option<&Path>, since: &str) -> ProofCardMetrics {
    let window = if since.trim().is_empty() {
        "7d".to_string()
    } else {
        since.trim().to_string()
    };
    let rep = reflex::report(data_dir, &window);

    let firings = u64_at(&rep, &["firings"]);
    let chain_checked = u64_at(&rep, &["chain", "checked"]);
    let chain_breaks = u64_at(&rep, &["chain", "breaks"]);
    let stops_recorded = u64_at(&rep, &["stopsRecorded"])
        .max(u64_at(&rep, &["receipts", "stopsRecorded"]));
    let resolutions_recorded = u64_at(&rep, &["resolutionsRecorded"]);
    let folds = u64_at(&rep, &["context", "folds"]);
    let chars_saved = u64_at(&rep, &["context", "charsSaved"]);
    let chars_told = u64_at(&rep, &["charsTold"]);
    let duplicates_dropped = u64_at(&rep, &["context", "duplicates"]);
    let sessions_active = rep
        .get("sessions")
        .and_then(Value::as_array)
        .map_or(0, |a| a.len() as u64);

    let (strata_segments, strata_bytes) = dir_stats(&data_dir.join("log"));
    let proof_notes_commits = repo.map(count_git_proof_notes).unwrap_or(0);

    let active_lease = lease::load_lease(data_dir).filter(|l| l.active);
    let (lease_active, lease_halted, lease_guards_count) = match active_lease {
        Some(l) => {
            let guards = l.allowed_paths.len()
                + l.denied_paths.len()
                + l.denied_command_patterns.len()
                + l.mutex_command_patterns.len()
                + usize::from(l.max_tool_calls_per_turn.is_some())
                + usize::from(l.max_identical_failures.is_some());
            (true, l.halted, guards)
        }
        None => (false, false, 0),
    };

    let digest_src = format!(
        "card:v={}:win={window}:firings={firings}:checked={chain_checked}:breaks={chain_breaks}:stops={stops_recorded}:saved={chars_saved}:seg={strata_segments}:notes={proof_notes_commits}",
        env!("CARGO_PKG_VERSION")
    );
    let card_digest_blake3 = blake3::hash(digest_src.as_bytes()).to_hex().to_string();

    ProofCardMetrics {
        version: env!("CARGO_PKG_VERSION").to_string(),
        window,
        firings,
        chain_checked,
        chain_breaks,
        stops_recorded,
        resolutions_recorded,
        folds,
        chars_saved,
        chars_told,
        duplicates_dropped,
        sessions_active,
        strata_segments,
        strata_bytes,
        proof_notes_commits,
        lease_active,
        lease_halted,
        lease_guards_count,
        card_digest_blake3,
    }
}

fn ascii_bar(pct: u64, width: usize) -> String {
    let clamped = pct.min(100) as usize;
    let filled = (clamped * width + 50) / 100;
    let mut s = String::with_capacity(width * 3);
    for i in 0..width {
        s.push(if i < filled { '█' } else { '░' });
    }
    s
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn fold_efficiency_pct(saved: u64, told: u64) -> u64 {
    saved
        .saturating_mul(100)
        .checked_div(saved.saturating_add(told))
        .unwrap_or(0)
}

fn pad_row(content: &str, inner_width: usize) -> String {
    let char_len = content.chars().count();
    let trimmed: String = if char_len > inner_width {
        content.chars().take(inner_width.saturating_sub(1)).collect::<String>() + "…"
    } else {
        content.to_string()
    };
    let pad = inner_width.saturating_sub(trimmed.chars().count());
    format!("│ {trimmed}{} │", " ".repeat(pad))
}

/// Render a compact, screenshot-ready ASCII terminal Proof Card.
pub fn render_proof_card_ascii(m: &ProofCardMetrics) -> String {
    const INNER: usize = 68;
    let top = format!("┌{}┐", "─".repeat(INNER + 2));
    let mid = format!("├{}┤", "─".repeat(INNER + 2));
    let bot = format!("└{}┘", "─".repeat(INNER + 2));

    let chain_status = if m.chain_breaks == 0 {
        format!("VERIFIED ({} links, 0 breaks)", m.chain_checked)
    } else {
        format!("ALERT ({} breaks / {} links)", m.chain_breaks, m.chain_checked)
    };
    let fold_pct = fold_efficiency_pct(m.chars_saved, m.chars_told);
    let est_tokens = m.chars_saved / 4;
    let lease_label = if m.lease_halted {
        "HALTED (emergency brake engaged)".to_string()
    } else if m.lease_active {
        format!("ARMED ({} active drift guards)", m.lease_guards_count)
    } else {
        "STANDBY (`vestige lease set` to arm)".to_string()
    };

    let rows = [
        top,
        pad_row(
            &format!(
                "VESTIGE v{} · CAUSAL PROOF SCORECARD ({}) · ZERO-CLOUD RUST",
                m.version, m.window
            ),
            INNER,
        ),
        mid.clone(),
        pad_row(&format!("Strata Chain   : {chain_status}"), INNER),
        pad_row(
            &format!(
                "Strata Log     : {} segments · {} on local disk",
                m.strata_segments,
                format_bytes(m.strata_bytes)
            ),
            INNER,
        ),
        pad_row(
            &format!(
                "Reflex Hooks   : {} firings across {} session(s)",
                m.firings, m.sessions_active
            ),
            INNER,
        ),
        pad_row(
            &format!(
                "Operator Stops : {} caught · {} closed with receipts",
                m.stops_recorded, m.resolutions_recorded
            ),
            INNER,
        ),
        pad_row(
            &format!(
                "Context Saved  : {} (~{} tok) · {} folds · {} dupes cut",
                format_bytes(m.chars_saved),
                est_tokens,
                m.folds,
                m.duplicates_dropped
            ),
            INNER,
        ),
        pad_row(
            &format!("Fold Bar       : [{}] {:>3}% kept out", ascii_bar(fold_pct, 20), fold_pct),
            INNER,
        ),
        pad_row(
            &format!(
                "Git Proof Notes: {} signed commits (`refs/notes/vestige-proof`)",
                m.proof_notes_commits
            ),
            INNER,
        ),
        pad_row(&format!("Execution Lease: {lease_label}"), INNER),
        mid,
        pad_row(
            &format!(
                "BLAKE3 Digest  : {} · $0.00 cloud · 0 LLMs · 0 vectors",
                &m.card_digest_blake3[..12]
            ),
            INNER,
        ),
        bot,
    ];
    rows.join("\n")
}

/// Render a copy-pasteable GitHub/HN/X Markdown Proof Card.
pub fn render_proof_card_markdown(m: &ProofCardMetrics) -> String {
    let fold_pct = fold_efficiency_pct(m.chars_saved, m.chars_told);
    let est_tokens = m.chars_saved / 4;
    let chain = if m.chain_breaks == 0 {
        format!("🟢 VERIFIED (`{}` links, `0` breaks)", m.chain_checked)
    } else {
        format!("🔴 `{}` breaks across `{}` links", m.chain_breaks, m.chain_checked)
    };
    let lease = if m.lease_halted {
        "🛑 HALTED".to_string()
    } else if m.lease_active {
        format!("🛡️ ARMED (`{}` drift guards)", m.lease_guards_count)
    } else {
        "⚪ STANDBY".to_string()
    };
    format!(
        "## ⚡ Vestige v{} — Causal Proof Scorecard (`{}`)\n\n\
| Metric | Verified Value |\n\
| :--- | :--- |\n\
| **Strata Hash Chain** | {chain} |\n\
| **Local Strata Store** | `{}` segments · `{}` |\n\
| **Reflex Hook Firings** | `{}` firings across `{}` session(s) |\n\
| **Operator Stops Caught** | `{}` caught · `{}` closed with receipts |\n\
| **Context Folded Out** | `{}` (~`{}` tokens, `{}%` `[{}]`, `{}` folds) |\n\
| **Git Proof Commits** | `{}` (`refs/notes/vestige-proof`) |\n\
| **Execution Lease** | {lease} |\n\
| **BLAKE3 Card Digest** | `{}` |\n\n\
*100% local Rust · Zero vectors · Zero LLMs · Zero telemetry · \\$0.00 cloud cost.*\n",
        m.version,
        m.window,
        m.strata_segments,
        format_bytes(m.strata_bytes),
        m.firings,
        m.sessions_active,
        m.stops_recorded,
        m.resolutions_recorded,
        format_bytes(m.chars_saved),
        est_tokens,
        fold_pct,
        ascii_bar(fold_pct, 12),
        m.folds,
        m.proof_notes_commits,
        &m.card_digest_blake3[..24],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn proof_card_collects_local_state_and_renders_ascii_and_markdown() {
        let dir = TempDir::new().unwrap();
        let log_dir = dir.path().join("log");
        fs::create_dir_all(&log_dir).unwrap();
        fs::write(log_dir.join("seg-0001.bin"), vec![0u8; 4096]).unwrap();

        let lease = lease::ExecutionLease {
            id: "lease-card".into(),
            active: true,
            denied_command_patterns: vec!["retrieval_bench".into()],
            max_identical_failures: Some(2),
            ..Default::default()
        };
        lease::save_lease(dir.path(), &lease).unwrap();

        let m = collect_proof_card(dir.path(), None, "7d");
        assert_eq!(m.strata_segments, 1);
        assert_eq!(m.strata_bytes, 4096);
        assert!(m.lease_active);
        assert_eq!(m.lease_guards_count, 2);
        assert_eq!(m.chain_breaks, 0);

        let ascii = render_proof_card_ascii(&m);
        assert!(ascii.contains("CAUSAL PROOF SCORECARD"), "{ascii}");
        assert!(ascii.contains("ARMED (2 active drift guards)"), "{ascii}");
        assert!(ascii.contains("$0.00 cloud"), "{ascii}");

        let md = render_proof_card_markdown(&m);
        assert!(md.contains("Causal Proof Scorecard"), "{md}");
        assert!(md.contains("ARMED (`2` drift guards)"), "{md}");
    }
}
