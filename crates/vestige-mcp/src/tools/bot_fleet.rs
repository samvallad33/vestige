//! # Vestige Deterministic GitHub Bot Fleet Engines
//!
//! Pure-Rust, zero-LLM, zero-vector, zero-cloud engines powering Vestige's
//! GitHub Bot fleet (Bots #2–#5 alongside `causal_frontier::run_pr_audit` for Bot #1):
//!
//! - **Bot #4 — Concealment & Biba Integrity Shield (`run_shield_audit`)**:
//!   Detects invisible Unicode TAG-block smuggling (`U+E0000..=U+E007F`), Trojan
//!   Source Bidi overrides (`U+202A..=U+202E`, `U+2066..=U+2069`), stealth
//!   zero-width injection, secret leaks (`vestige_core::scan_secrets`), and Biba
//!   high-integrity control-plane touches (`.github/workflows/`, `AGENTS.md`,
//!   `CLAUDE.md`, `.git/hooks/`, `scripts/hardcode_scan.sh`).
//! - **Bot #3 — BISCEPTER Red-Main Bisect Planner (`run_bisect_plan`)**:
//!   Scores every commit in `good..bad` with deterministic `Q16.16` `BISCEPTER`
//!   probability mass (power-law recency × exact anchor overlap × revert adjacency),
//!   computes the optimal weighted-median first pivot, and reports exact probe
//!   savings over naive 50/50 `git bisect`.
//! - **Bot #5 — Memory Hygiene & RETRACE Oscillation Auditor (`run_hygiene_audit`)**:
//!   Ranks files with repeated revert/fix oscillation cycles (`RETRACE`), audits
//!   `AGENTS.md` / `CLAUDE.md` / `.cursorrules` for >250-line context-rot bloat,
//!   and verifies every cited repo-relative source file still exists on disk.

use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::walk_verify::biscepter::{
    CommitRiskSignals, score_commit_prior_q16, simulate_uniform_probes, simulate_weighted_probes,
    weighted_median_pivot,
};

fn git_stdout(repo: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "-c",
            "core.fsmonitor=false",
            "-c",
            "color.ui=false",
            "-c",
            "diff.external=",
        ])
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("failed to run git {args:?}: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!("git {args:?} failed: {stderr}"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

// ---------------------------------------------------------------------------
// Bot #4: Concealment & Biba Integrity Shield (+ remember_unified Firewall)
// ---------------------------------------------------------------------------

/// A single byte-exact invisible-Unicode concealment finding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConcealmentHit {
    pub kind: &'static str,
    pub codepoint: String,
    pub char_offset: usize,
}

/// Scan `text` for invisible Unicode concealment / Trojan Source attacks:
/// - Unicode TAG-block smuggling (`U+E0000..=U+E007F`) — hidden MCP prompt injection
/// - Trojan Source Bidi overrides (`U+202A..=U+202E`, `U+2066..=U+2069`, `U+200E`, `U+200F`)
/// - Stealth zero-width characters (`U+200B`, `U+200C`, `U+200D`, `U+2060`, mid-text `U+FEFF`)
pub fn scan_concealed_unicode(text: &str) -> Vec<ConcealmentHit> {
    let mut hits = Vec::new();
    for (idx, ch) in text.chars().enumerate() {
        let cp = ch as u32;
        let kind = match cp {
            0xE0000..=0xE007F => Some("UNICODE_TAG_SMUGGLING"),
            0x202A..=0x202E | 0x2066..=0x2069 | 0x200E | 0x200F => Some("BIDI_OVERRIDE_TROJAN"),
            0x200B | 0x200C | 0x200D | 0x2060 => Some("ZERO_WIDTH_STEALTH"),
            0xFEFF if idx > 0 => Some("ZERO_WIDTH_STEALTH"),
            _ => None,
        };
        if let Some(k) = kind {
            hits.push(ConcealmentHit {
                kind: k,
                codepoint: format!("U+{cp:04X}"),
                char_offset: idx,
            });
            if hits.len() >= 32 {
                break;
            }
        }
    }
    hits
}

/// Returns true for high-integrity control-plane files under a Biba integrity lattice.
fn is_biba_high_integrity_path(path: &str) -> bool {
    // hardcode-scan: protocol (control-plane file paths for Biba integrity boundary)
    const EXACT_OR_PREFIX: &[&str] = &[
        ".github/workflows/",
        ".git/hooks/",
        "AGENTS.md",
        "CLAUDE.md",
        ".cursorrules",
        ".clinerules",
        "scripts/hardcode_scan.sh",
        "scripts/hardcode_scan_selftest.sh",
    ];
    let norm = path.replace('\\', "/");
    EXACT_OR_PREFIX
        .iter()
        .any(|p| norm == *p || norm.starts_with(p) || norm.ends_with(&format!("/{p}")))
}

/// Scan an added Rust source line for raw UTF-8 byte-index string slicing (`&ident[..N]` or
/// `&ident[..ident.len() - k]`) that panics on multibyte UTF-8 characters like `'—'` or `'é'`.
/// Skips comments, safe `floor_char_boundary`/`is_char_boundary`/`as_bytes` lines, and hex/digest/byte slices.
pub fn scan_unsafe_rust_utf8_slices(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    if trimmed.is_empty()
        || trimmed.starts_with("//")
        || trimmed.starts_with("/*")
        || trimmed.starts_with('*')
        || trimmed.contains("floor_char_boundary")
        || trimmed.contains("ceil_char_boundary")
        || trimmed.contains("is_char_boundary")
        || trimmed.contains(".as_bytes()")
        || trimmed.contains(".into_bytes()")
        || trimmed.contains(".bytes()")
    {
        return Vec::new();
    }

    let bytes = trimmed.as_bytes();
    let mut findings = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'&' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' || bytes[j] == b'.')
            {
                j += 1;
            }
            if j > start && j < bytes.len() && bytes[j] == b'[' {
                let expr = &trimmed[start..j];
                let ident_leaf = expr.rsplit('.').next().unwrap_or(expr).to_ascii_lowercase();
                if let Some(close_rel) = trimmed[j + 1..].find(']') {
                    let inner = trimmed[j + 1..j + 1 + close_rel].trim();
                    let is_safe_non_str = ident_leaf.contains("byte")
                        || ident_leaf.contains("buf")
                        || ident_leaf.contains("digest")
                        || ident_leaf.contains("hex")
                        || ident_leaf.contains("hash")
                        || ident_leaf.contains("sha")
                        || ident_leaf.contains("vec")
                        || ident_leaf.contains("arr")
                        || ident_leaf.contains("slice")
                        || ident_leaf.contains("turns")
                        || ident_leaf.contains("sess")
                        || ident_leaf.contains("items")
                        || ident_leaf.contains("args")
                        || ident_leaf.contains("ids");

                    if !is_safe_non_str && inner.contains("..") {
                        let is_len_sub = inner.contains(".len()")
                            && (inner.contains('-') || inner.contains("saturating_sub"));
                        let has_byte_bound = inner
                            .split("..")
                            .filter(|part| !part.is_empty())
                            .any(|part| {
                                let p = part.trim().trim_start_matches('=');
                                p.parse::<usize>().is_ok_and(|n| n >= 4)
                                    || p.contains(".min(")
                            });
                        if is_len_sub || has_byte_bound {
                            findings.push(format!("&{expr}[{inner}]"));
                        }
                    }
                    i = j + 1 + close_rel + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    findings
}

/// Scan an added Python line for mutable default arguments (`def f(x=[])` or `def f(x={})`).
pub fn scan_python_mutable_defaults(line: &str) -> Option<&'static str> {
    let t = line.trim();
    if !t.starts_with("def ") && !t.starts_with("async def ") {
        return None;
    }
    if t.contains("=[]") || t.contains("= []") || t.contains("={}") || t.contains("= {}") {
        Some("Python mutable default argument (`=[]` / `={}`) shares state across calls; default to `None`.")
    } else {
        None
    }
}

/// Scan an added TypeScript line for type-safety bypasses (`@ts-ignore`, `@ts-nocheck`, `as any`).
pub fn scan_typescript_unsafe_bypasses(line: &str) -> Option<&'static str> {
    let t = line.trim();
    if t.contains("@ts-ignore") || t.contains("@ts-nocheck") {
        Some("TypeScript compiler suppression (`@ts-ignore`/`@ts-nocheck`) hides type errors; use explicit types or `@ts-expect-error` with rationale.")
    } else if !t.starts_with("//") && t.contains(" as any") {
        Some("Unsafe TypeScript cast (`as any`) disables static type checking; narrow with `unknown` or a type guard.")
    } else {
        None
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ShieldReport {
    pub base_sha: String,
    pub head_sha: String,
    pub files_scanned: usize,
    pub added_lines_scanned: usize,
    pub blocked: bool,
    pub unicode_findings: Vec<Value>,
    pub secret_findings: Vec<Value>,
    pub utf8_slice_alerts: Vec<Value>,
    pub language_rule_alerts: Vec<Value>,
    pub biba_touches: Vec<String>,
    pub proof_digest_blake3: String,
}

/// Run the deterministic Concealment, Secret, UTF-8 Slice Safety, and Biba Integrity audit across `base..head`.
pub fn run_shield_audit(repo: &Path, base: &str, head: &str) -> Result<ShieldReport, String> {
    if !repo.is_dir() {
        return Err(format!("repo '{}' is not a directory", repo.display()));
    }
    let base_sha = git_stdout(repo, &["rev-parse", base])?
        .trim()
        .to_ascii_lowercase();
    let head_sha = git_stdout(repo, &["rev-parse", head])?
        .trim()
        .to_ascii_lowercase();
    let diff = git_stdout(
        repo,
        &["diff", "-U0", "--no-ext-diff", "--no-color", &base_sha, &head_sha],
    )?;

    let mut current_file = String::new();
    let mut current_line: u32 = 0;
    let mut files_seen: HashSet<String> = HashSet::new();
    let mut added_lines_scanned: usize = 0;
    let mut unicode_findings: Vec<Value> = Vec::new();
    let mut secret_findings: Vec<Value> = Vec::new();
    let mut utf8_slice_alerts: Vec<Value> = Vec::new();
    let mut language_rule_alerts: Vec<Value> = Vec::new();
    let mut biba_touches: Vec<String> = Vec::new();

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            current_file = rest.to_string();
            if files_seen.insert(current_file.clone())
                && is_biba_high_integrity_path(&current_file)
            {
                biba_touches.push(current_file.clone());
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("@@ ") {
            if let Some(plus) = rest.split_whitespace().find(|w| w.starts_with('+')) {
                let start_str = plus[1..].split(',').next().unwrap_or("0");
                current_line = start_str.parse::<u32>().unwrap_or(0);
            }
            continue;
        }
        if let Some(added) = line.strip_prefix('+')
            && !line.starts_with("+++")
            && !current_file.is_empty()
        {
            added_lines_scanned += 1;
            let u_hits = scan_concealed_unicode(added);
            for h in u_hits {
                unicode_findings.push(json!({
                    "file": current_file,
                    "line": current_line,
                    "kind": h.kind,
                    "codepoint": h.codepoint,
                    "char_offset": h.char_offset,
                }));
            }
            for f in vestige_core::scan_secrets(added) {
                if f.blocks_ingestion() {
                    // Never echo secret bytes — report pattern kind and file:line only.
                    secret_findings.push(json!({
                        "file": current_file,
                        "line": current_line,
                        "kind": f.kind.as_str(),
                    }));
                }
            }
            if current_file.ends_with(".rs") {
                for expr in scan_unsafe_rust_utf8_slices(added) {
                    utf8_slice_alerts.push(json!({
                        "file": current_file,
                        "line": current_line,
                        "pattern": expr,
                        "fix": "Use floor_char_boundary(...) or chars().take(...)",
                    }));
                }
            } else if current_file.ends_with(".py")
                && let Some(msg) = scan_python_mutable_defaults(added)
            {
                language_rule_alerts.push(json!({
                    "file": current_file,
                    "line": current_line,
                    "rule_id": "VST-PY-MUTDEF-001",
                    "message": msg,
                }));
            } else if (current_file.ends_with(".ts") || current_file.ends_with(".tsx"))
                && let Some(msg) = scan_typescript_unsafe_bypasses(added)
            {
                language_rule_alerts.push(json!({
                    "file": current_file,
                    "line": current_line,
                    "rule_id": "VST-TS-UNSAFE-001",
                    "message": msg,
                }));
            }
            current_line = current_line.saturating_add(1);
        }
    }

    biba_touches.sort();
    let blocked = !unicode_findings.is_empty() || !secret_findings.is_empty();
    let digest_input = format!(
        "shield:{base_sha}..{head_sha}:files={}:lines={added_lines_scanned}:unicode={}:secrets={}:utf8_slices={}:lang_rules={}:biba={}",
        files_seen.len(),
        unicode_findings.len(),
        secret_findings.len(),
        utf8_slice_alerts.len(),
        language_rule_alerts.len(),
        biba_touches.len()
    );
    let proof_digest_blake3 = blake3::hash(digest_input.as_bytes()).to_hex().to_string();

    Ok(ShieldReport {
        base_sha,
        head_sha,
        files_scanned: files_seen.len(),
        added_lines_scanned,
        blocked,
        unicode_findings,
        secret_findings,
        utf8_slice_alerts,
        language_rule_alerts,
        biba_touches,
        proof_digest_blake3,
    })
}

pub fn render_shield_markdown(report: &ShieldReport) -> String {
    let badge = if report.blocked {
        "🔴 **BLOCKED — CONCEALMENT OR SECRET VIOLATION**"
    } else if !report.utf8_slice_alerts.is_empty() {
        "🟡 **CLEAN WITH RUST UTF-8 SLICE PANIC WARNINGS**"
    } else if !report.biba_touches.is_empty() {
        "🟡 **CLEAN (BIBA HIGH-INTEGRITY FILES MODIFIED — OWNER REVIEW RECOMMENDED)**"
    } else {
        "🟢 **VERIFIED CLEAN**"
    };
    let mut md = String::new();
    md.push_str("<!-- vestige-shield -->\n");
    md.push_str("## 🛡️ Vestige Concealment & Biba Integrity Shield\n\n");
    md.push_str(&format!("**Status:** {badge}\n\n"));
    md.push_str("| Check | Count |\n| :--- | :--- |\n");
    md.push_str(&format!(
        "| **Added Lines Scanned** | {} across {} files |\n",
        report.added_lines_scanned, report.files_scanned
    ));
    md.push_str(&format!(
        "| **Invisible Unicode / Trojan Bidi Smuggling** | **{}** |\n",
        report.unicode_findings.len()
    ));
    md.push_str(&format!(
        "| **Hard-Block Secret Patterns** | **{}** |\n",
        report.secret_findings.len()
    ));
    md.push_str(&format!(
        "| **Rust Raw UTF-8 Byte-Slice Panic Alerts** | **{}** |\n",
        report.utf8_slice_alerts.len()
    ));
    md.push_str(&format!(
        "| **Biba High-Integrity Control-Plane Touches** | **{}** |\n",
        report.biba_touches.len()
    ));
    md.push_str(&format!(
        "| **BLAKE3 Shield Digest** | `{}` |\n\n",
        &report.proof_digest_blake3[..24]
    ));

    if !report.unicode_findings.is_empty() {
        md.push_str("### 🚨 Concealed Unicode / Trojan Source Findings\n\n");
        md.push_str("| File:Line | Attack Class | Codepoint |\n| :--- | :--- | :--- |\n");
        for h in report.unicode_findings.iter().take(25) {
            md.push_str(&format!(
                "| `{}:{}` | `{}` | `{}` |\n",
                h["file"].as_str().unwrap_or("?"),
                h["line"].as_u64().unwrap_or(0),
                h["kind"].as_str().unwrap_or("?"),
                h["codepoint"].as_str().unwrap_or("?"),
            ));
        }
        md.push('\n');
    }

    if !report.secret_findings.is_empty() {
        md.push_str("### 🔐 Secret-Pattern Findings *(values redacted)*\n\n");
        md.push_str("| File:Line | Pattern Kind |\n| :--- | :--- |\n");
        for s in report.secret_findings.iter().take(25) {
            md.push_str(&format!(
                "| `{}:{}` | `{}` |\n",
                s["file"].as_str().unwrap_or("?"),
                s["line"].as_u64().unwrap_or(0),
                s["kind"].as_str().unwrap_or("?"),
            ));
        }
        md.push('\n');
    }

    if !report.utf8_slice_alerts.is_empty() {
        md.push_str("### 🦀 Rust UTF-8 Byte-Slice Panic Alerts\n\n");
        md.push_str("| File:Line | Unsafe Slice Pattern | Deterministic Fix |\n| :--- | :--- | :--- |\n");
        for a in report.utf8_slice_alerts.iter().take(25) {
            md.push_str(&format!(
                "| `{}:{}` | `{}` | {} |\n",
                a["file"].as_str().unwrap_or("?"),
                a["line"].as_u64().unwrap_or(0),
                a["pattern"].as_str().unwrap_or("?"),
                a["fix"].as_str().unwrap_or("Use floor_char_boundary"),
            ));
        }
        md.push('\n');
    }

    if !report.biba_touches.is_empty() {
        md.push_str("### 🔒 Biba High-Integrity Control-Plane Files Modified\n\n");
        for f in &report.biba_touches {
            md.push_str(&format!("- `{f}`\n"));
        }
        md.push('\n');
    }

    md.push_str("---\n*Verified 100% deterministically by **Vestige Shield** — zero LLMs, zero cloud, \\$0.00.*\n");
    md
}

/// Render a SARIF 2.1.0 report for native GitHub Code Scanning inline PR annotations & Security tab.
pub fn render_shield_sarif(report: &ShieldReport) -> Value {
    let mut results: Vec<Value> = Vec::new();
    for h in &report.unicode_findings {
        let file = h["file"].as_str().unwrap_or("unknown");
        let line = h["line"].as_u64().unwrap_or(1).max(1);
        let kind = h["kind"].as_str().unwrap_or("UNICODE_CONCEALMENT");
        let cp = h["codepoint"].as_str().unwrap_or("?");
        results.push(json!({
            "ruleId": "VST-UNICODE-001",
            "level": "error",
            "message": {
                "text": format!("Concealed Unicode / Trojan Source character detected ({kind}, {cp}).")
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": { "uri": file },
                    "region": { "startLine": line }
                }
            }]
        }));
    }
    for s in &report.secret_findings {
        let file = s["file"].as_str().unwrap_or("unknown");
        let line = s["line"].as_u64().unwrap_or(1).max(1);
        let kind = s["kind"].as_str().unwrap_or("secret");
        results.push(json!({
            "ruleId": "VST-SECRET-001",
            "level": "error",
            "message": {
                "text": format!("Hard-block secret pattern detected ({kind}; value redacted).")
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": { "uri": file },
                    "region": { "startLine": line }
                }
            }]
        }));
    }
    for a in &report.utf8_slice_alerts {
        let file = a["file"].as_str().unwrap_or("unknown");
        let line = a["line"].as_u64().unwrap_or(1).max(1);
        let pat = a["pattern"].as_str().unwrap_or("?");
        let fix = a["fix"].as_str().unwrap_or("Use floor_char_boundary(...)");
        results.push(json!({
            "ruleId": "VST-UTF8-SLICE-001",
            "level": "warning",
            "message": {
                "text": format!("Raw UTF-8 byte-index slice `{pat}` may panic on multibyte characters. {fix}")
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": { "uri": file },
                    "region": { "startLine": line }
                }
            }]
        }));
    }
    for a in &report.language_rule_alerts {
        let file = a["file"].as_str().unwrap_or("unknown");
        let line = a["line"].as_u64().unwrap_or(1).max(1);
        let rule_id = a["rule_id"].as_str().unwrap_or("VST-LANG-001");
        let msg = a["message"].as_str().unwrap_or("Language rule alert");
        results.push(json!({
            "ruleId": rule_id,
            "level": "warning",
            "message": { "text": msg },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": { "uri": file },
                    "region": { "startLine": line }
                }
            }]
        }));
    }
    for path in &report.biba_touches {
        results.push(json!({
            "ruleId": "VST-BIBA-001",
            "level": "note",
            "message": {
                "text": format!("Biba high-integrity control-plane file modified: `{path}`. Owner review recommended.")
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": { "uri": path },
                    "region": { "startLine": 1 }
                }
            }]
        }));
    }
    json!({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "Vestige Shield",
                    "version": env!("CARGO_PKG_VERSION"),
                    "informationUri": "https://github.com/samvallad33/vestige",
                    "rules": [
                        {
                            "id": "VST-UNICODE-001",
                            "name": "ConcealedUnicodeSmuggling",
                            "shortDescription": { "text": "Invisible Unicode TAG, Trojan Bidi override, or zero-width stealth character" },
                            "defaultConfiguration": { "level": "error" }
                        },
                        {
                            "id": "VST-SECRET-001",
                            "name": "HardBlockSecretPattern",
                            "shortDescription": { "text": "Credential or private key pattern detected in added diff lines" },
                            "defaultConfiguration": { "level": "error" }
                        },
                        {
                            "id": "VST-UTF8-SLICE-001",
                            "name": "UnsafeRustUtf8ByteSlice",
                            "shortDescription": { "text": "Raw UTF-8 byte-index string slice in Rust that panics on multibyte characters" },
                            "defaultConfiguration": { "level": "warning" }
                        },
                        {
                            "id": "VST-PY-MUTDEF-001",
                            "name": "PythonMutableDefaultArgument",
                            "shortDescription": { "text": "Mutable list or dict literal used as a Python function default argument" },
                            "defaultConfiguration": { "level": "warning" }
                        },
                        {
                            "id": "VST-TS-UNSAFE-001",
                            "name": "TypeScriptSafetyBypass",
                            "shortDescription": { "text": "TypeScript @ts-ignore, @ts-nocheck, or unsafe `as any` type cast" },
                            "defaultConfiguration": { "level": "warning" }
                        },
                        {
                            "id": "VST-BIBA-001",
                            "name": "BibaControlPlaneModification",
                            "shortDescription": { "text": "High-integrity CI workflow, hook, or agent rule file modified" },
                            "defaultConfiguration": { "level": "note" }
                        }
                    ]
                }
            },
            "properties": {
                "baseSha": report.base_sha,
                "headSha": report.head_sha,
                "blocked": report.blocked,
                "proofDigestBlake3": report.proof_digest_blake3
            },
            "results": results
        }]
    })
}

// ---------------------------------------------------------------------------
// Bot #3: BISCEPTER Q16.16 Red-Main Bisect Planner
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct BisectPlanCandidate {
    pub index_oldest_first: usize,
    pub sha: String,
    pub author: String,
    pub subject: String,
    pub prior_weight_q16: u64,
    pub touches_failing_anchor: bool,
    pub oscillation_neighbor: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BisectPlanReport {
    pub good_sha: String,
    pub bad_sha: String,
    pub total_commits: usize,
    pub recommended_pivot: Option<BisectPlanCandidate>,
    pub top_suspects: Vec<BisectPlanCandidate>,
    pub expected_biscepter_probes_top1: usize,
    pub uniform_bisect_probes_top1: usize,
    pub proof_digest_blake3: String,
}

/// Compute a deterministic `BISCEPTER` Q16.16 weighted-median bisect plan across `good..bad`.
pub fn run_bisect_plan(
    repo: &Path,
    good: &str,
    bad: &str,
    failing_files: &[String],
) -> Result<BisectPlanReport, String> {
    if !repo.is_dir() {
        return Err(format!("repo '{}' is not a directory", repo.display()));
    }
    let good_sha = git_stdout(repo, &["rev-parse", good])?
        .trim()
        .to_ascii_lowercase();
    let bad_sha = git_stdout(repo, &["rev-parse", bad])?
        .trim()
        .to_ascii_lowercase();
    let range = format!("{good_sha}..{bad_sha}");

    let shas: Vec<String> = git_stdout(repo, &["rev-list", "--reverse", "--no-merges", &range])?
        .lines()
        .map(str::trim)
        .filter(|s| s.len() == 40)
        .map(|s| s.to_ascii_lowercase())
        .collect();

    let norm_failing: Vec<String> = failing_files
        .iter()
        .map(|f| f.trim().replace('\\', "/").trim_start_matches("./").to_string())
        .filter(|f| !f.is_empty())
        .collect();

    let total = shas.len();
    let mut candidates = Vec::with_capacity(total);
    let mut weights = Vec::with_capacity(total);

    for (idx, sha) in shas.iter().enumerate() {
        let meta = git_stdout(repo, &["show", "-s", "--format=%an\t%s", sha]).unwrap_or_default();
        let (author, subject) = meta
            .trim()
            .split_once('\t')
            .map(|(a, s)| (a.to_string(), s.to_string()))
            .unwrap_or_else(|| ("unknown".to_string(), String::new()));

        let touched_files: Vec<String> =
            git_stdout(repo, &["diff-tree", "--no-commit-id", "--name-only", "-r", sha])
                .unwrap_or_default()
                .lines()
                .map(|l| l.trim().replace('\\', "/"))
                .filter(|l| !l.is_empty())
                .collect();

        let touches_failing_anchor = !norm_failing.is_empty()
            && touched_files.iter().any(|tf| {
                norm_failing
                    .iter()
                    .any(|ff| tf == ff || tf.ends_with(&format!("/{ff}")))
            });
        let lower_subj = subject.to_ascii_lowercase();
        let oscillation_neighbor = lower_subj.starts_with("revert ")
            || lower_subj.contains("fix(")
            || lower_subj.starts_with("fix:");

        let w = score_commit_prior_q16(&CommitRiskSignals {
            age_from_bad: total.saturating_sub(1).saturating_sub(idx),
            touches_failing_anchor,
            unwitnessed: git_stdout(repo, &["notes", "--ref=refs/notes/vestige-proof", "show", sha])
                .is_err(),
            oscillation_neighbor,
            bad_outcomes: 0,
            good_outcomes: 0,
        });
        weights.push(w);
        candidates.push(BisectPlanCandidate {
            index_oldest_first: idx,
            sha: sha.clone(),
            author,
            subject,
            prior_weight_q16: w,
            touches_failing_anchor,
            oscillation_neighbor,
        });
    }

    let pivot_idx = weighted_median_pivot(&weights);
    let recommended_pivot = pivot_idx.and_then(|i| candidates.get(i).cloned());

    let mut ranked = candidates.clone();
    ranked.sort_by(|a, b| {
        b.prior_weight_q16
            .cmp(&a.prior_weight_q16)
            .then(b.index_oldest_first.cmp(&a.index_oldest_first))
    });
    let top_suspects: Vec<BisectPlanCandidate> = ranked.into_iter().take(10).collect();

    let top1_idx = top_suspects
        .first()
        .map(|c| c.index_oldest_first)
        .unwrap_or(0);
    let expected_biscepter_probes_top1 = simulate_weighted_probes(&weights, top1_idx);
    let uniform_bisect_probes_top1 = simulate_uniform_probes(total, top1_idx);

    let digest_input = format!(
        "bisect-plan:{good_sha}..{bad_sha}:n={total}:pivot={}:probes={expected_biscepter_probes_top1}/{uniform_bisect_probes_top1}",
        pivot_idx.unwrap_or(0)
    );
    let proof_digest_blake3 = blake3::hash(digest_input.as_bytes()).to_hex().to_string();

    Ok(BisectPlanReport {
        good_sha,
        bad_sha,
        total_commits: total,
        recommended_pivot,
        top_suspects,
        expected_biscepter_probes_top1,
        uniform_bisect_probes_top1,
        proof_digest_blake3,
    })
}

pub fn render_bisect_plan_markdown(report: &BisectPlanReport) -> String {
    let mut md = String::new();
    md.push_str("<!-- vestige-bisector -->\n");
    md.push_str("## 🎯 Vestige Bisector — BISCEPTER Q16.16 Red-Main Triage\n\n");
    md.push_str("| Metric | Value |\n| :--- | :--- |\n");
    md.push_str(&format!(
        "| **Range (`good..bad`)** | `{}..{}` ({} commits) |\n",
        &report.good_sha[..report.good_sha.len().min(8)],
        &report.bad_sha[..report.bad_sha.len().min(8)],
        report.total_commits
    ));
    if let Some(ref p) = report.recommended_pivot {
        md.push_str(&format!(
            "| **Recommended BISCEPTER First Pivot** | `{}` (#{}/{}) — {} |\n",
            &p.sha[..p.sha.len().min(8)],
            p.index_oldest_first + 1,
            report.total_commits,
            p.subject.replace('|', "\\|")
        ));
    }
    md.push_str(&format!(
        "| **Probes to Top-1 Suspect (BISCEPTER vs 50/50)** | **{}** vs **{}** |\n",
        report.expected_biscepter_probes_top1, report.uniform_bisect_probes_top1
    ));
    md.push_str(&format!(
        "| **BLAKE3 Plan Digest** | `{}` |\n\n",
        &report.proof_digest_blake3[..24]
    ));

    if !report.top_suspects.is_empty() {
        md.push_str("### 🔬 Top Ranked Suspects (Deterministic Q16.16 Mass)\n\n");
        md.push_str("| Rank | Commit | Q16 Mass | Anchor Hit | Author | Subject |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n");
        for (i, c) in report.top_suspects.iter().enumerate() {
            md.push_str(&format!(
                "| #{} | `{}` | `{}` | {} | {} | {} |\n",
                i + 1,
                &c.sha[..c.sha.len().min(8)],
                c.prior_weight_q16,
                if c.touches_failing_anchor { "YES" } else { "-" },
                c.author,
                c.subject.replace('|', "\\|")
            ));
        }
        md.push('\n');
    }

    md.push_str("### ▶️ Run Deterministic Proof Locally\n\n```sh\n");
    md.push_str(&format!(
        "vestige prove --repo . --good {} --bad {} --test <repro-script>\n",
        &report.good_sha[..report.good_sha.len().min(10)],
        &report.bad_sha[..report.bad_sha.len().min(10)]
    ));
    md.push_str("```\n\n---\n*Computed offline by **Vestige Bisector (BISCEPTER Q16.16)** — zero LLMs, zero cloud, \\$0.00.*\n");
    md
}

// ---------------------------------------------------------------------------
// Bot #5: Memory Hygiene & RETRACE Oscillation Hotspot Auditor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct HygieneReport {
    pub commits_scanned: usize,
    pub hygiene_score: u32,
    pub oscillation_hotspots: Vec<Value>,
    pub rule_file_audits: Vec<Value>,
    pub proof_digest_blake3: String,
}

fn strip_vestige_projection_fence(content: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in content.lines() {
        if line.contains("vestige:projection:begin") {
            inside = true;
            continue;
        }
        if line.contains("vestige:projection:end") {
            inside = false;
            continue;
        }
        if !inside {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn extract_cited_repo_paths(text: &str) -> Vec<String> {
    // hardcode-scan: protocol (source/config file extensions for dead-reference auditing)
    const EXTS: &[&str] = &[
        ".rs", ".ts", ".tsx", ".js", ".svelte", ".py", ".sh", ".toml", ".yml", ".yaml", ".md",
    ];
    let mut out = Vec::new();
    for raw in text.split_whitespace() {
        let cleaned = raw
            .trim_end_matches('.')
            .trim_matches(|c: char| {
                matches!(
                    c,
                    '`' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '<' | '>'
                )
            });
        if cleaned.contains("://") || !cleaned.contains('/') || cleaned.starts_with('/') {
            continue;
        }
        let path_part = cleaned
            .rsplit_once(':')
            .map(|(l, _)| l)
            .unwrap_or(cleaned)
            .split('#')
            .next()
            .unwrap_or(cleaned)
            .strip_prefix("./")
            .unwrap_or(cleaned);
        if path_part.contains('*') || path_part.contains('<') || path_part.contains('{') {
            continue;
        }
        let lower = path_part.to_ascii_lowercase();
        if EXTS.iter().any(|e| lower.ends_with(e))
            && !out.iter().any(|existing: &String| existing == path_part)
        {
            out.push(path_part.to_string());
        }
    }
    out
}

/// Run weekly deterministic Memory Hygiene & RETRACE Oscillation Hotspot audit on `repo`.
pub fn run_hygiene_audit(repo: &Path, max_commits: usize) -> Result<HygieneReport, String> {
    if !repo.is_dir() {
        return Err(format!("repo '{}' is not a directory", repo.display()));
    }
    let limit = max_commits.clamp(10, 2000);
    const SEP: &str = "---VESTIGE-HYGIENE-SEP---";
    let log = git_stdout(
        repo,
        &[
            "log",
            "-n",
            &limit.to_string(),
            "--no-merges",
            "--name-only",
            &format!("--format={SEP}%H%x09%s"),
        ],
    )?;

    let mut commits_scanned = 0usize;
    let mut file_reverts: BTreeMap<String, usize> = BTreeMap::new();
    let mut file_fixes: BTreeMap<String, usize> = BTreeMap::new();
    let mut file_touches: BTreeMap<String, usize> = BTreeMap::new();

    for block in log.split(SEP).skip(1) {
        let mut lines = block.lines();
        let Some(header) = lines.next() else {
            continue;
        };
        commits_scanned += 1;
        let subject = header.split_once('\t').map(|(_, s)| s).unwrap_or("");
        let lower = subject.trim().to_ascii_lowercase();
        let is_revert = lower.starts_with("revert ") || lower.contains("revert \"");
        let is_fix = lower.starts_with("fix:") || lower.starts_with("fix(");

        for file in lines.map(str::trim).filter(|l| !l.is_empty()) {
            *file_touches.entry(file.to_string()).or_insert(0) += 1;
            if is_revert {
                *file_reverts.entry(file.to_string()).or_insert(0) += 1;
            }
            if is_fix {
                *file_fixes.entry(file.to_string()).or_insert(0) += 1;
            }
        }
    }

    let mut hotspots: Vec<(usize, usize, usize, String)> = file_touches
        .into_iter()
        .filter_map(|(file, touches)| {
            let rev = *file_reverts.get(&file).unwrap_or(&0);
            let fix = *file_fixes.get(&file).unwrap_or(&0);
            if rev >= 1 || fix >= 4 {
                Some((rev * 10 + fix, rev, touches, file))
            } else {
                None
            }
        })
        .collect();
    hotspots.sort_by(|a, b| b.0.cmp(&a.0).then(a.3.cmp(&b.3)));

    let oscillation_hotspots: Vec<Value> = hotspots
        .into_iter()
        .take(12)
        .map(|(risk, reverts, total_touches, file)| {
            let fixes = *file_fixes.get(&file).unwrap_or(&0);
            json!({
                "file": file,
                "reverts": reverts,
                "fixes": fixes,
                "total_touches": total_touches,
                "oscillation_risk": risk,
            })
        })
        .collect();

    // Audit rule files (AGENTS.md, CLAUDE.md, .cursorrules) for >250-line bloat & dead file refs!
    // hardcode-scan: protocol (standard agent instruction rule filenames)
    const RULE_FILES: &[&str] = &["AGENTS.md", "CLAUDE.md", ".cursorrules"];
    let mut rule_file_audits = Vec::new();
    let mut penalty: u32 = 0;

    for name in RULE_FILES {
        let full: PathBuf = repo.join(name);
        if !full.is_file() {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&full) else {
            continue;
        };
        let authored = strip_vestige_projection_fence(&raw);
        let authored_lines = authored.lines().count();
        let bloated = authored_lines > 250;
        if bloated {
            penalty = penalty.saturating_add(15);
        }
        let cited = extract_cited_repo_paths(&authored);
        let dead_refs: Vec<String> = cited
            .into_iter()
            .filter(|p| !repo.join(p).exists())
            .collect();
        penalty = penalty.saturating_add((dead_refs.len() as u32).saturating_mul(5));

        rule_file_audits.push(json!({
            "file": name,
            "authored_lines": authored_lines,
            "bloated": bloated,
            "dead_file_refs": dead_refs,
            "status": if !dead_refs.is_empty() {
                "STALE_DEAD_FILE_REFS"
            } else if bloated {
                "CONTEXT_ROT_BLOATED"
            } else {
                "HEALTHY"
            },
        }));
    }

    let revert_hotspot_penalty = (oscillation_hotspots
        .iter()
        .filter(|h| h["reverts"].as_u64().unwrap_or(0) >= 1)
        .count() as u32)
        .saturating_mul(5);
    let hygiene_score = 100u32.saturating_sub(penalty.saturating_add(revert_hotspot_penalty));

    let digest_input = format!(
        "hygiene:commits={commits_scanned}:score={hygiene_score}:hotspots={}:rules={}",
        oscillation_hotspots.len(),
        rule_file_audits.len()
    );
    let proof_digest_blake3 = blake3::hash(digest_input.as_bytes()).to_hex().to_string();

    Ok(HygieneReport {
        commits_scanned,
        hygiene_score,
        oscillation_hotspots,
        rule_file_audits,
        proof_digest_blake3,
    })
}

pub fn render_hygiene_markdown(report: &HygieneReport) -> String {
    let badge = if report.hygiene_score >= 85 {
        "🟢 **HEALTHY**"
    } else if report.hygiene_score >= 60 {
        "🟡 **DRIFT DETECTED**"
    } else {
        "🔴 **HYGIENE ATTENTION NEEDED**"
    };
    let mut md = String::new();
    md.push_str("<!-- vestige-hygiene -->\n");
    md.push_str("## 🧹 Vestige Weekly Memory Hygiene & RETRACE Oscillation Report\n\n");
    md.push_str(&format!(
        "**Score:** **{}/100** ({badge}) · Scanned **{}** commits · Digest `{}`\n\n",
        report.hygiene_score,
        report.commits_scanned,
        &report.proof_digest_blake3[..24]
    ));

    if !report.rule_file_audits.is_empty() {
        md.push_str("### 📄 Agent Rule-File Rot & Context-Bloat Audit\n\n");
        md.push_str("| Rule File | Authored Lines | Status | Dead File References |\n| :--- | :--- | :--- | :--- |\n");
        for r in &report.rule_file_audits {
            let dead: Vec<&str> = r["dead_file_refs"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let dead_str = if dead.is_empty() {
                "none".to_string()
            } else {
                dead.iter().map(|d| format!("`{d}`")).collect::<Vec<_>>().join(", ")
            };
            md.push_str(&format!(
                "| `{}` | {} | `{}` | {} |\n",
                r["file"].as_str().unwrap_or("?"),
                r["authored_lines"].as_u64().unwrap_or(0),
                r["status"].as_str().unwrap_or("?"),
                dead_str
            ));
        }
        md.push('\n');
    }

    if !report.oscillation_hotspots.is_empty() {
        md.push_str("### 🔁 RETRACE Oscillation & Churn Hotspots\n\n");
        md.push_str("| File | Reverts | Fix Commits | Total Touches | Risk |\n| :--- | :--- | :--- | :--- | :--- |\n");
        for h in &report.oscillation_hotspots {
            md.push_str(&format!(
                "| `{}` | **{}** | {} | {} | `{}` |\n",
                h["file"].as_str().unwrap_or("?"),
                h["reverts"].as_u64().unwrap_or(0),
                h["fixes"].as_u64().unwrap_or(0),
                h["total_touches"].as_u64().unwrap_or(0),
                h["oscillation_risk"].as_u64().unwrap_or(0),
            ));
        }
        md.push('\n');
    }

    md.push_str("---\n*Generated offline by **Vestige Memory Hygiene Bot** — zero LLMs, zero vectors, \\$0.00.*\n");
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git_run(repo: &Path, args: &[&str]) {
        let st = Command::new("git")
            .current_dir(repo)
            .args(args)
            .status()
            .unwrap();
        assert!(st.success(), "git {args:?} failed");
    }

    #[test]
    fn shield_catches_unicode_tag_smuggling_bidi_trojan_secrets_and_biba_touches() {
        let repo = tempfile::tempdir().unwrap();
        git_run(repo.path(), &["init", "-q"]);
        git_run(repo.path(), &["config", "user.email", "sec@example.com"]);
        git_run(repo.path(), &["config", "user.name", "Sec"]);
        std::fs::write(repo.path().join("AGENTS.md"), "# Clean rules\n").unwrap();
        std::fs::write(repo.path().join("notes.txt"), "hello\n").unwrap();
        git_run(repo.path(), &["add", "."]);
        git_run(repo.path(), &["commit", "-q", "-m", "base"]);
        let base = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();

        // Plant invisible Unicode TAG char (U+E0041) + Trojan Bidi RLO (U+202E) + Biba rule touch!
        let smuggled = "safe text \u{E0041} and bidi \u{202E} spoof\n";
        std::fs::write(repo.path().join("notes.txt"), smuggled).unwrap();
        std::fs::write(repo.path().join("AGENTS.md"), "# Modified rule file\n").unwrap();
        git_run(repo.path(), &["add", "."]);
        git_run(repo.path(), &["commit", "-q", "-m", "stealth pr"]);
        let head = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();

        let report = run_shield_audit(repo.path(), &base, &head).expect("shield audit");
        assert!(report.blocked, "shield must block concealed unicode: {report:?}");
        assert!(
            report
                .unicode_findings
                .iter()
                .any(|h| h["kind"] == "UNICODE_TAG_SMUGGLING"),
            "{report:?}"
        );
        assert!(
            report
                .unicode_findings
                .iter()
                .any(|h| h["kind"] == "BIDI_OVERRIDE_TROJAN"),
            "{report:?}"
        );
        assert!(
            report.biba_touches.iter().any(|f| f == "AGENTS.md"),
            "must flag Biba control-plane touch on AGENTS.md: {report:?}"
        );
        let md = render_shield_markdown(&report);
        assert!(md.contains("<!-- vestige-shield -->"), "{md}");
        assert!(md.contains("UNICODE_TAG_SMUGGLING"), "{md}");

        let sarif = render_shield_sarif(&report);
        assert_eq!(sarif["version"], "2.1.0");
        let results = sarif["runs"][0]["results"].as_array().unwrap();
        assert!(results.iter().any(|r| r["ruleId"] == "VST-UNICODE-001"));
        assert!(results.iter().any(|r| r["ruleId"] == "VST-BIBA-001"));
    }

    #[test]
    fn bisect_plan_ranks_failing_file_anchor_and_picks_biscepter_pivot() {
        let repo = tempfile::tempdir().unwrap();
        git_run(repo.path(), &["init", "-q"]);
        git_run(repo.path(), &["config", "user.email", "dev@example.com"]);
        git_run(repo.path(), &["config", "user.name", "Dev"]);
        std::fs::create_dir_all(repo.path().join("src")).unwrap();
        std::fs::write(repo.path().join("src/a.rs"), "pub fn a() {}\n").unwrap();
        std::fs::write(repo.path().join("src/target.rs"), "pub fn t() -> i32 { 1 }\n").unwrap();
        git_run(repo.path(), &["add", "."]);
        git_run(repo.path(), &["commit", "-q", "-m", "good base"]);
        let good = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();

        for i in 0..5 {
            std::fs::write(repo.path().join("src/a.rs"), format!("pub fn a() {{ let _ = {i}; }}\n")).unwrap();
            git_run(repo.path(), &["add", "src/a.rs"]);
            git_run(repo.path(), &["commit", "-q", "-m", &format!("unrelated {i}")]);
        }
        std::fs::write(repo.path().join("src/target.rs"), "pub fn t() -> i32 { 2 }\n").unwrap();
        git_run(repo.path(), &["add", "src/target.rs"]);
        git_run(repo.path(), &["commit", "-q", "-m", "fix(target): touch failing file"]);
        let bad = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();

        let report = run_bisect_plan(
            repo.path(),
            &good,
            &bad,
            &["src/target.rs".to_string()],
        )
        .expect("bisect plan");
        assert_eq!(report.total_commits, 6, "{report:?}");
        assert!(report.recommended_pivot.is_some(), "{report:?}");
        assert!(
            report.top_suspects[0].touches_failing_anchor,
            "top suspect must be the commit touching src/target.rs: {report:?}"
        );
        let md = render_bisect_plan_markdown(&report);
        assert!(md.contains("<!-- vestige-bisector -->"), "{md}");
    }

    #[test]
    fn hygiene_audit_catches_retrace_oscillation_hotspot_and_stale_rule_refs() {
        let repo = tempfile::tempdir().unwrap();
        git_run(repo.path(), &["init", "-q"]);
        git_run(repo.path(), &["config", "user.email", "dev@example.com"]);
        git_run(repo.path(), &["config", "user.name", "Dev"]);
        std::fs::create_dir_all(repo.path().join("src")).unwrap();
        std::fs::write(repo.path().join("src/flaky.rs"), "pub fn f() -> u8 { 1 }\n").unwrap();
        std::fs::write(
            repo.path().join("AGENTS.md"),
            "Always check `src/flaky.rs` and `src/deleted_ghost.rs`.\n",
        )
        .unwrap();
        git_run(repo.path(), &["add", "."]);
        git_run(repo.path(), &["commit", "-q", "-m", "init"]);

        std::fs::write(repo.path().join("src/flaky.rs"), "pub fn f() -> u8 { 2 }\n").unwrap();
        git_run(repo.path(), &["add", "src/flaky.rs"]);
        git_run(repo.path(), &["commit", "-q", "-m", "bad flaky edit"]);
        let bad = git_stdout(repo.path(), &["rev-parse", "HEAD"]).unwrap().trim().to_string();
        git_run(repo.path(), &["revert", "--no-edit", &bad]);

        let report = run_hygiene_audit(repo.path(), 50).expect("hygiene audit");
        assert!(
            report
                .oscillation_hotspots
                .iter()
                .any(|h| h["file"] == "src/flaky.rs" && h["reverts"].as_u64().unwrap_or(0) >= 1),
            "must detect RETRACE revert hotspot on src/flaky.rs: {report:?}"
        );
        assert!(
            report.rule_file_audits.iter().any(|r| {
                r["file"] == "AGENTS.md"
                    && r["dead_file_refs"]
                        .as_array()
                        .map(|a| a.iter().any(|v| v == "src/deleted_ghost.rs"))
                        .unwrap_or(false)
            }),
            "must detect dead file ref src/deleted_ghost.rs in AGENTS.md: {report:?}"
        );
        let md = render_hygiene_markdown(&report);
        assert!(md.contains("<!-- vestige-hygiene -->"), "{md}");
        assert!(md.contains("src/deleted_ghost.rs"), "{md}");
    }

    #[test]
    fn shield_catches_unsafe_rust_utf8_byte_slices_without_false_positives() {
        // Must catch raw string byte slices like habitus_energy_q16.rs:242 and wikidata.rs:159.
        assert_eq!(
            scan_unsafe_rust_utf8_slices("let head = &lower[..64];"),
            vec!["&lower[..64]"]
        );
        assert_eq!(
            scan_unsafe_rust_utf8_slices("let apoc = format!(\"{w0}{}\", &w1[..w1.len() - 1]);"),
            vec!["&w1[..w1.len() - 1]"]
        );
        // Must NOT false-positive on safe char-boundary guards, comments, bytes, or hex digests.
        assert!(scan_unsafe_rust_utf8_slices("let head = &lower[..lower.floor_char_boundary(64)];").is_empty());
        assert!(scan_unsafe_rust_utf8_slices("// comment &lower[..64]").is_empty());
        assert!(scan_unsafe_rust_utf8_slices("let d = &report.proof_digest_blake3[..24];").is_empty());
        assert!(scan_unsafe_rust_utf8_slices("let b = &raw_bytes[..16];").is_empty());
    }
}
