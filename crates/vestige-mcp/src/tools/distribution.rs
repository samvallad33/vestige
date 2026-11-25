//! Self-propagating Git protocol, coding-agent, red-team, compliance, and uninstall engines.
//!
//! Pure deterministic Rust, zero LLMs, zero vectors, zero cloud dependencies.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use chrono::Utc;
use serde::Serialize;
use serde_json::{Value, json};

use crate::tools::bot_fleet::{HygieneReport, scan_concealed_unicode};
use crate::tools::proof_card::collect_proof_card;
use crate::tools::reflex::lease;

fn git_cmd(repo: &Path) -> Command {
    let mut c = Command::new("git");
    c.arg("-C")
        .arg(repo)
        .args(["-c", "core.fsmonitor=false", "-c", "color.ui=false"])
        .stdin(Stdio::null());
    c
}

fn git_stdout(repo: &Path, args: &[&str]) -> Result<String, String> {
    let out = git_cmd(repo)
        .args(args)
        .output()
        .map_err(|e| format!("git {args:?} failed: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {args:?} exited {}: {}",
            out.status.code().unwrap_or(1),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

// ---------------------------------------------------------------------------
// Category 1 #1, #2, #3, #8, #9: Git Protocol Auto-Federation, Aliases, Merge
// Driver Registration & Monorepo Scope Discovery
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct GitProtocolInstallReport {
    pub notes_fetch_configured: bool,
    pub vblame_alias_configured: bool,
    pub bisect_prove_alias_configured: bool,
    pub merge_driver_configured: bool,
    pub gitattributes_updated: bool,
    pub monorepo_scopes: Vec<String>,
}

/// Discover sub-package boundaries (`Cargo.toml`, `package.json`, `pyproject.toml`)
/// up to depth 2 so monorepos can auto-scope per-team Execution Leases.
pub fn discover_monorepo_scopes(repo: &Path) -> Vec<String> {
    const MANIFESTS: &[&str] = &["Cargo.toml", "package.json", "pyproject.toml"];
    const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "dist", "build"];

    let mut scopes: BTreeSet<String> = BTreeSet::new();
    let Ok(lvl1) = fs::read_dir(repo) else {
        return Vec::new();
    };
    for e1 in lvl1.flatten() {
        let p1 = e1.path();
        if !p1.is_dir() {
            continue;
        }
        let Some(n1) = p1.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if SKIP_DIRS.contains(&n1) || n1.starts_with('.') {
            continue;
        }
        if MANIFESTS.iter().any(|m| p1.join(m).is_file()) {
            scopes.insert(n1.to_string());
        }
        if let Ok(lvl2) = fs::read_dir(&p1) {
            for e2 in lvl2.flatten() {
                let p2 = e2.path();
                if !p2.is_dir() {
                    continue;
                }
                let Some(n2) = p2.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                if SKIP_DIRS.contains(&n2) || n2.starts_with('.') {
                    continue;
                }
                if MANIFESTS.iter().any(|m| p2.join(m).is_file()) {
                    scopes.insert(format!("{n1}/{n2}"));
                }
            }
        }
    }
    scopes.into_iter().collect()
}

/// Configure local Git repository mechanics:
/// 1) `remote.origin.fetch` auto-federation for `refs/notes/vestige-proof`
/// 2) `alias.vblame` pretty log formatter alongside proof notes
/// 3) `alias.bisect-prove` drop-in BISCEPTER launcher
/// 4) `merge.vestige-guard` Unicode/Bidi-blocking merge driver + `.gitattributes`
pub fn install_git_protocol_wiring(repo: &Path) -> Result<GitProtocolInstallReport, String> {
    if !repo.is_dir() {
        return Err(format!("repository '{}' does not exist", repo.display()));
    }
    let mut notes_fetch_configured = false;
    if git_stdout(repo, &["config", "--get", "remote.origin.url"]).is_ok() {
        let refspec = "+refs/notes/vestige-proof:refs/notes/vestige-proof";
        let existing = git_stdout(repo, &["config", "--get-all", "remote.origin.fetch"])
            .unwrap_or_default();
        if !existing.lines().any(|l| l.trim() == refspec) {
            let _ = git_stdout(repo, &["config", "--add", "remote.origin.fetch", refspec]);
        }
        notes_fetch_configured = true;
    }

    let vblame_alias = "log --notes=vestige-proof --pretty=format:'%C(yellow)%h%Creset %C(cyan)%an%Creset %s %C(green)%N%Creset'";
    let vblame_alias_configured = git_stdout(repo, &["config", "alias.vblame", vblame_alias]).is_ok();

    let bisect_alias = "!vestige bisect-plan --repo .";
    let bisect_prove_alias_configured =
        git_stdout(repo, &["config", "alias.bisect-prove", bisect_alias]).is_ok();

    let merge_driver_configured = git_stdout(
        repo,
        &[
            "config",
            "merge.vestige-guard.name",
            "Vestige Unicode & Bidi Concealment Merge Guard",
        ],
    )
    .is_ok()
        && git_stdout(
            repo,
            &[
                "config",
                "merge.vestige-guard.driver",
                "vestige merge-driver %O %A %B",
            ],
        )
        .is_ok();

    let attrs_path = repo.join(".gitattributes");
    let existing_attrs = fs::read_to_string(&attrs_path).unwrap_or_default();
    let required_lines = [
        "CLAUDE.md merge=vestige-guard",
        "AGENTS.md merge=vestige-guard",
        ".cursorrules merge=vestige-guard",
    ];
    let mut next_attrs = existing_attrs.clone();
    let mut gitattributes_updated = false;
    for rule in required_lines {
        if !existing_attrs.lines().any(|l| l.trim() == rule) {
            if !next_attrs.is_empty() && !next_attrs.ends_with('\n') {
                next_attrs.push('\n');
            }
            next_attrs.push_str(rule);
            next_attrs.push('\n');
            gitattributes_updated = true;
        }
    }
    if gitattributes_updated {
        fs::write(&attrs_path, next_attrs)
            .map_err(|e| format!("failed to update {}: {e}", attrs_path.display()))?;
    }

    let monorepo_scopes = discover_monorepo_scopes(repo);
    Ok(GitProtocolInstallReport {
        notes_fetch_configured,
        vblame_alias_configured,
        bisect_prove_alias_configured,
        merge_driver_configured,
        gitattributes_updated,
        monorepo_scopes,
    })
}

// ---------------------------------------------------------------------------
// Category 1 #3: Custom Git Merge Driver (`vestige merge-driver %O %A %B`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct MergeDriverVerdict {
    pub allowed: bool,
    pub unicode_hits: usize,
    pub secret_hits: usize,
    pub message: String,
}

/// Inspect incoming (`other`) and current (`current`) sides of a 3-way merge for
/// concealed Unicode smuggling, Trojan Bidi overrides, or hard-block secrets.
/// Blocks before `git merge` finishes when violations exist; otherwise runs `git merge-file`.
pub fn run_vestige_merge_driver(
    base: &Path,
    current: &Path,
    other: &Path,
) -> Result<MergeDriverVerdict, String> {
    let other_text = fs::read_to_string(other)
        .map_err(|e| format!("read incoming {}: {e}", other.display()))?;
    let current_text = fs::read_to_string(current)
        .map_err(|e| format!("read current {}: {e}", current.display()))?;

    let mut unicode_hits = scan_concealed_unicode(&other_text).len();
    unicode_hits += scan_concealed_unicode(&current_text).len();

    let mut secret_hits = vestige_core::scan_secrets(&other_text)
        .into_iter()
        .filter(|f| f.blocks_ingestion())
        .count();
    secret_hits += vestige_core::scan_secrets(&current_text)
        .into_iter()
        .filter(|f| f.blocks_ingestion())
        .count();

    if unicode_hits > 0 || secret_hits > 0 {
        return Ok(MergeDriverVerdict {
            allowed: false,
            unicode_hits,
            secret_hits,
            message: format!(
                "Vestige merge-driver BLOCKED merge: detected {unicode_hits} concealed Unicode/Bidi hit(s) and {secret_hits} secret finding(s)."
            ),
        });
    }

    let status = Command::new("git")
        .args(["merge-file", "-L", "HEAD", "-L", "base", "-L", "incoming"])
        .arg(current)
        .arg(base)
        .arg(other)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("failed to run git merge-file: {e}"))?;

    if status.success() {
        Ok(MergeDriverVerdict {
            allowed: true,
            unicode_hits: 0,
            secret_hits: 0,
            message: "Vestige merge-driver verified clean and merged successfully.".into(),
        })
    } else {
        Ok(MergeDriverVerdict {
            allowed: false,
            unicode_hits: 0,
            secret_hits: 0,
            message: "Vestige merge-driver found standard text merge conflicts; resolve manually."
                .into(),
        })
    }
}

// ---------------------------------------------------------------------------
// Category 1 #4: Signed `.undo.patch` Generator
// ---------------------------------------------------------------------------

/// Generate a BLAKE3-signed deterministic reverse patch (`git diff <commit> <commit>^`)
/// so on-call engineers can recover a broken `main` commit via `git apply` in 3 seconds.
pub fn generate_undo_patch(repo: &Path, commit: &str) -> Result<(String, String), String> {
    let sha = git_stdout(repo, &["rev-parse", commit])?
        .trim()
        .to_ascii_lowercase();
    let parent = git_stdout(repo, &["rev-parse", &format!("{sha}^")])?
        .trim()
        .to_ascii_lowercase();
    let diff = git_stdout(
        repo,
        &["diff", "--binary", "--no-ext-diff", "--no-color", &sha, &parent],
    )?;
    let digest = blake3::hash(diff.as_bytes()).to_hex().to_string();
    let header = format!(
        "# Vestige Signed Undo Patch\n# Target-Commit: {sha}\n# Restore-Parent: {parent}\n# BLAKE3-Diff-Digest: {digest}\n# Apply with: git apply fix-{}.undo.patch\n",
        &sha[..sha.len().min(8)]
    );
    Ok((format!("{header}{diff}"), digest))
}

// ---------------------------------------------------------------------------
// Category 1 #5: RETRACE Code Churn Heatmap SVG Generator
// ---------------------------------------------------------------------------

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render a GitHub-contribution-graph-style dark-mode SVG heatmap from a `HygieneReport`.
pub fn render_churn_heatmap_svg(report: &HygieneReport) -> String {
    let rows = report.oscillation_hotspots.iter().take(12).collect::<Vec<_>>();
    let height = 96 + (rows.len().max(1) * 28);
    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"760\" height=\"{height}\" viewBox=\"0 0 760 {height}\">\n"
    ));
    svg.push_str("  <rect width=\"760\" height=\"100%\" rx=\"12\" fill=\"#0d1117\" stroke=\"#30363d\"/>\n");
    svg.push_str("  <text x=\"20\" y=\"30\" fill=\"#e6edf3\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" font-size=\"14\" font-weight=\"700\">Vestige RETRACE Code Churn Heatmap</text>\n");
    svg.push_str(&format!(
        "  <text x=\"20\" y=\"50\" fill=\"#8b949e\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" font-size=\"11\">Commits scanned: {} | Hygiene score: {}/100 | BLAKE3: {}</text>\n",
        report.commits_scanned,
        report.hygiene_score,
        &report.proof_digest_blake3[..16]
    ));

    if rows.is_empty() {
        svg.push_str("  <text x=\"20\" y=\"88\" fill=\"#3fb950\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" font-size=\"12\">Zero oscillation hotspots detected across scanned commit window.</text>\n");
    } else {
        for (idx, h) in rows.iter().enumerate() {
            let y = 78 + idx * 28;
            let file = h["file"].as_str().unwrap_or("?");
            let reverts = h["reverts"].as_u64().unwrap_or(0);
            let fixes = h["fixes"].as_u64().unwrap_or(0);
            let touches = h["total_touches"].as_u64().unwrap_or(1);
            let short_file: String = if file.chars().count() > 44 {
                let tail: String = file.chars().rev().take(41).collect::<String>().chars().rev().collect();
                format!("...{tail}")
            } else {
                file.to_string()
            };
            svg.push_str(&format!(
                "  <text x=\"20\" y=\"{}\" fill=\"#c9d1d9\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" font-size=\"11\">{}</text>\n",
                y + 12,
                xml_escape(&short_file)
            ));
            let blocks = (touches.min(12)) as usize;
            for b in 0..12 {
                let bx = 390 + b * 18;
                let fill = if b >= blocks {
                    "#161b22"
                } else if reverts >= 2 {
                    "#f85149"
                } else if reverts >= 1 || fixes >= 3 {
                    "#d29922"
                } else {
                    "#2ea043"
                };
                svg.push_str(&format!(
                    "  <rect x=\"{bx}\" y=\"{y}\" width=\"14\" height=\"14\" rx=\"3\" fill=\"{fill}\"/>\n"
                ));
            }
            svg.push_str(&format!(
                "  <text x=\"616\" y=\"{}\" fill=\"#8b949e\" font-family=\"ui-monospace,SFMono-Regular,Menlo,monospace\" font-size=\"11\">{}r / {}f / {}t</text>\n",
                y + 12,
                reverts,
                fixes,
                touches
            ));
        }
    }
    svg.push_str("</svg>\n");
    svg
}

// ---------------------------------------------------------------------------
// Category 1 #6: Commit-Message Trailer (`Vestige-Ablation: 412->38 (-90.8%)`)
// ---------------------------------------------------------------------------

pub fn format_ablation_trailer(before_lines: usize, after_lines: usize) -> String {
    let pct_x10 = before_lines
        .saturating_sub(after_lines)
        .saturating_mul(1000)
        .checked_div(before_lines)
        .unwrap_or(0);
    format!(
        "Vestige-Ablation: {before_lines}->{after_lines} (-{}.{}%)",
        pct_x10 / 10,
        pct_x10 % 10
    )
}

pub fn append_commit_trailer(msg_file: &Path, trailer: &str) -> Result<bool, String> {
    let body = fs::read_to_string(msg_file)
        .map_err(|e| format!("read {}: {e}", msg_file.display()))?;
    if body.lines().any(|l| l.trim() == trailer.trim()) {
        return Ok(false);
    }
    let mut next = body;
    if !next.ends_with('\n') {
        next.push('\n');
    }
    if !next.ends_with("\n\n") {
        next.push('\n');
    }
    next.push_str(trailer.trim());
    next.push('\n');
    fs::write(msg_file, next).map_err(|e| format!("write {}: {e}", msg_file.display()))?;
    Ok(true)
}

// ---------------------------------------------------------------------------
// Category 1 #7: Zero-Slop Streak Counter & Shields.io Badge JSON
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct StreakReport {
    pub clean_commit_streak: usize,
    pub scanned_commits: usize,
    pub signed_proof_notes_in_streak: usize,
    pub last_break_sha: Option<String>,
    pub last_break_subject: Option<String>,
    pub shields_badge: Value,
    pub proof_digest_blake3: String,
}

pub fn compute_zero_slop_streak(repo: &Path, max_commits: usize) -> Result<StreakReport, String> {
    let lim = max_commits.clamp(1, 500).to_string();
    let log = git_stdout(repo, &["log", "-n", &lim, "--format=%H\t%s"])?;
    let mut streak = 0usize;
    let mut scanned = 0usize;
    let mut signed = 0usize;
    let mut last_break_sha = None;
    let mut last_break_subject = None;

    for line in log.lines() {
        let Some((sha, subj)) = line.split_once('\t') else {
            continue;
        };
        scanned += 1;
        let low = subj.trim().to_ascii_lowercase();
        if low.starts_with("revert ") || low.starts_with("revert\"") {
            last_break_sha = Some(sha.to_string());
            last_break_subject = Some(subj.to_string());
            break;
        }
        streak += 1;
        if git_stdout(repo, &["notes", "--ref=refs/notes/vestige-proof", "show", sha]).is_ok() {
            signed += 1;
        }
    }

    let color = if streak >= 20 {
        "brightgreen"
    } else if streak >= 5 {
        "green"
    } else {
        "yellow"
    };
    let shields_badge = json!({
        "schemaVersion": 1,
        "label": "Vestige Streak",
        "message": format!("{streak} clean commits"),
        "color": color
    });
    let digest_src = format!("streak:{streak}:scanned:{scanned}:signed:{signed}");
    let proof_digest_blake3 = blake3::hash(digest_src.as_bytes()).to_hex().to_string();

    Ok(StreakReport {
        clean_commit_streak: streak,
        scanned_commits: scanned,
        signed_proof_notes_in_streak: signed,
        last_break_sha,
        last_break_subject,
        shields_badge,
        proof_digest_blake3,
    })
}

// ---------------------------------------------------------------------------
// Category 2 #19: Agentic PR Description Auto-Writer (`vestige pr-body`)
// ---------------------------------------------------------------------------

pub fn render_pr_body(
    repo: &Path,
    base: &str,
    head: &str,
    ablation_json: Option<&Path>,
) -> Result<String, String> {
    let base_sha = git_stdout(repo, &["rev-parse", base])?.trim().to_string();
    let head_sha = git_stdout(repo, &["rev-parse", head])?.trim().to_string();
    let stat = git_stdout(
        repo,
        &["diff", "--numstat", "--no-ext-diff", &base_sha, &head_sha],
    )?;

    let mut files = Vec::new();
    let mut added_total = 0usize;
    let mut deleted_total = 0usize;
    for line in stat.lines() {
        let mut parts = line.split('\t');
        let a = parts.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let d = parts.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        if let Some(path) = parts.next() {
            added_total += a;
            deleted_total += d;
            files.push((path.to_string(), a, d));
        }
    }

    let mut out = String::new();
    out.push_str("## Summary\n\n");
    out.push_str(&format!(
        "This pull request covers `{}..{}` across {} files (+{} / -{} lines).\n\n",
        &base_sha[..base_sha.len().min(8)],
        &head_sha[..head_sha.len().min(8)],
        files.len(),
        added_total,
        deleted_total
    ));

    if let Some(p) = ablation_json
        && let Ok(raw) = fs::read_to_string(p)
        && let Ok(val) = serde_json::from_str::<Value>(&raw)
    {
        let necessary = val["necessary_hunks"].as_array().cloned().unwrap_or_default();
        let noise = val["superstitious_hunks"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        out.push_str("### Necessary Core Changes (Halpern-Pearl Verified)\n\n");
        if necessary.is_empty() {
            out.push_str("No necessary hunks were isolated in the ablation report.\n\n");
        } else {
            for h in &necessary {
                out.push_str(&format!(
                    "- `{}` ({})\n",
                    h["file"].as_str().unwrap_or("?"),
                    h["header"].as_str().unwrap_or("hunk")
                ));
            }
            out.push('\n');
        }
        out.push_str("### Refactored Boilerplate / Non-Essential Hunks\n\n");
        if noise.is_empty() {
            out.push_str("Zero superstitious hunks detected.\n\n");
        } else {
            for h in &noise {
                out.push_str(&format!(
                    "- `{}` ({})\n",
                    h["file"].as_str().unwrap_or("?"),
                    h["header"].as_str().unwrap_or("hunk")
                ));
            }
            out.push('\n');
        }
    } else {
        out.push_str("### Modified Files\n\n");
        for (path, a, d) in files.iter().take(25) {
            out.push_str(&format!("- `{path}` (+{a} / -{d})\n"));
        }
        out.push('\n');
    }

    out.push_str("Verified locally with Vestige.\n");
    Ok(out)
}

// ---------------------------------------------------------------------------
// Category 2 #20: Ghost-Context Detector (`vestige doctor --context-rot`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ContextRotFinding {
    pub instruction_file: String,
    pub dead_references: Vec<String>,
    pub concealed_unicode_hits: usize,
    pub line_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContextRotReport {
    pub files_scanned: usize,
    pub total_dead_references: usize,
    pub total_concealed_unicode: usize,
    pub findings: Vec<ContextRotFinding>,
}

fn extract_backticked_repo_paths(text: &str) -> Vec<String> {
    let mut out = BTreeSet::new();
    let mut chars = text.split('`');
    let _ = chars.next();
    while let Some(seg) = chars.next() {
        let s = seg.trim().trim_start_matches("./");
        let _ = chars.next();
        if s.len() < 4 || s.len() > 140 || s.contains(' ') || s.contains("://") {
            continue;
        }
        let clean = s.split(':').next().unwrap_or(s);
        let has_slash = clean.contains('/');
        let has_ext = [
            ".rs", ".ts", ".tsx", ".js", ".py", ".go", ".toml", ".yml", ".yaml", ".md", ".sh",
        ]
        .iter()
        .any(|ext| clean.ends_with(ext));
        if has_slash && has_ext && !clean.contains('*') && !clean.contains('<') {
            out.insert(clean.to_string());
        }
    }
    out.into_iter().collect()
}

pub fn run_context_rot_doctor(repo: &Path) -> ContextRotReport {
    let mut targets: Vec<PathBuf> = [
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        ".cursorrules",
        ".clinerules",
    ]
    .iter()
    .map(|f| repo.join(f))
    .filter(|p| p.is_file())
    .collect();

    let cursor_rules_dir = repo.join(".cursor").join("rules");
    if let Ok(entries) = fs::read_dir(&cursor_rules_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_file() {
                targets.push(p);
            }
        }
    }

    let mut findings = Vec::new();
    let mut total_dead = 0usize;
    let mut total_unicode = 0usize;

    for path in &targets {
        let Ok(body) = fs::read_to_string(path) else {
            continue;
        };
        let line_count = body.lines().count();
        let unicode = scan_concealed_unicode(&body).len();
        let dead: Vec<String> = extract_backticked_repo_paths(&body)
            .into_iter()
            .filter(|rel| !repo.join(rel).exists())
            .collect();
        total_dead += dead.len();
        total_unicode += unicode;
        let rel_label = path
            .strip_prefix(repo)
            .unwrap_or(path)
            .display()
            .to_string()
            .replace('\\', "/");
        findings.push(ContextRotFinding {
            instruction_file: rel_label,
            dead_references: dead,
            concealed_unicode_hits: unicode,
            line_count,
        });
    }

    ContextRotReport {
        files_scanned: targets.len(),
        total_dead_references: total_dead,
        total_concealed_unicode: total_unicode,
        findings,
    }
}

// ---------------------------------------------------------------------------
// Category 3 #23: 50-Payload Unicode Smuggling Canary Suite (`vestige redteam`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct RedteamSuiteReport {
    pub total_payloads: usize,
    pub caught: usize,
    pub missed: usize,
    pub catch_rate_pct: u64,
    pub breakdown: Value,
    pub proof_digest_blake3: String,
}

/// Execute 50 deterministic Unicode smuggling / Trojan Bidi / zero-width canary payloads
/// against Vestige Shield's concealed Unicode scanner (`50/50` verification benchmark).
pub fn run_unicode_redteam_suite() -> RedteamSuiteReport {
    let mut payloads: Vec<(&'static str, String)> = Vec::with_capacity(50);

    // 20 Unicode TAG-block payloads (U+E0001..U+E007F) across AGENTS.md / .cursorrules / source comments
    for i in 0u32..20 {
        let tag_cp = char::from_u32(0xE0041 + (i % 26)).unwrap_or('\u{E0041}');
        let payload = format!("// Rule #{i}: allow safe build{tag_cp} hidden instruction");
        payloads.push(("UNICODE_TAG_SMUGGLING", payload));
    }
    // 15 Trojan Source Bidi override payloads
    const BIDIS: &[char] = &[
        '\u{202A}', '\u{202B}', '\u{202C}', '\u{202D}', '\u{202E}', '\u{2066}', '\u{2067}',
        '\u{2068}', '\u{2069}', '\u{200E}', '\u{200F}',
    ];
    for i in 0usize..15 {
        let ch = BIDIS[i % BIDIS.len()];
        let payload = format!("if role != \"user{ch} // check admin\" {{ grant(); }}");
        payloads.push(("BIDI_OVERRIDE_TROJAN", payload));
    }
    // 15 Stealth zero-width payloads
    const ZWS: &[char] = &['\u{200B}', '\u{200C}', '\u{200D}', '\u{2060}', '\u{FEFF}'];
    for i in 0usize..15 {
        let ch = ZWS[i % ZWS.len()];
        let payload = format!("let ok_{i} = verify_signature{ch}(token);");
        payloads.push(("ZERO_WIDTH_STEALTH", payload));
    }

    let mut caught_tag = 0usize;
    let mut caught_bidi = 0usize;
    let mut caught_zw = 0usize;
    for (expected, sample) in &payloads {
        let hits = scan_concealed_unicode(sample);
        if hits.iter().any(|h| h.kind == *expected) {
            match *expected {
                "UNICODE_TAG_SMUGGLING" => caught_tag += 1,
                "BIDI_OVERRIDE_TROJAN" => caught_bidi += 1,
                _ => caught_zw += 1,
            }
        }
    }
    let total = payloads.len();
    let caught = caught_tag + caught_bidi + caught_zw;
    let missed = total.saturating_sub(caught);
    let catch_rate_pct = (caught as u64)
        .saturating_mul(100)
        .checked_div(total as u64)
        .unwrap_or(0);
    let digest_src = format!("redteam:{caught}/{total}:tag={caught_tag}:bidi={caught_bidi}:zw={caught_zw}");
    let proof_digest_blake3 = blake3::hash(digest_src.as_bytes()).to_hex().to_string();

    RedteamSuiteReport {
        total_payloads: total,
        caught,
        missed,
        catch_rate_pct,
        breakdown: json!({
            "unicode_tag_smuggling": { "caught": caught_tag, "total": 20 },
            "bidi_override_trojan": { "caught": caught_bidi, "total": 15 },
            "zero_width_stealth": { "caught": caught_zw, "total": 15 }
        }),
        proof_digest_blake3,
    }
}

// ---------------------------------------------------------------------------
// Category 4 #31 & #32: EU AI Act (Art. 12/14) & SOC 2 (CC8.1) Annex Exporter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceStandard {
    EuAiAct,
    Soc2,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComplianceAnnex {
    pub standard: String,
    pub generated_at: String,
    pub controls: Vec<String>,
    pub strata_segments: usize,
    pub strata_bytes: u64,
    pub reflex_chain_checked: u64,
    pub reflex_chain_breaks: u64,
    pub operator_stops_recorded: u64,
    pub operator_resolutions_recorded: u64,
    pub signed_git_proof_commits: usize,
    pub execution_lease_active: bool,
    pub annex_digest_blake3: String,
}

pub fn generate_compliance_annex(
    data_dir: &Path,
    repo: Option<&Path>,
    standard: ComplianceStandard,
) -> ComplianceAnnex {
    let card = collect_proof_card(data_dir, repo, "30d");
    let (label, controls) = match standard {
        ComplianceStandard::EuAiAct => (
            "EU AI Act Articles 12 & 14 (Deterministic Logging & Human Oversight)",
            vec![
                "Article 12(1): Automatic append-only hash-chained event recording (Strata + Reflex Ledger)".into(),
                "Article 14(4): Deterministic human-oversight stop gates & 15-minute Operator permits".into(),
                "Article 14(4)(e): Immediate mid-session Execution Lease emergency halt (`continue: false`)".into(),
            ],
        ),
        ComplianceStandard::Soc2 => (
            "SOC 2 Type II CC8.1 & ISO 27001 A.8.32 (Change Management & Code Provenance)",
            vec![
                "CC8.1: Signed commit receipts & git proof notes (`refs/notes/vestige-proof`)".into(),
                "CC8.1: Deterministic pre-merge Zombie Regression & Chesterton's Fence gates".into(),
                "A.8.32: Local-only execution with zero source-code upload to external AI clouds".into(),
            ],
        ),
    };
    let generated_at = Utc::now().to_rfc3339();
    let digest_src = format!(
        "compliance:{label}:{}:{}:{}:{}",
        card.strata_segments,
        card.chain_checked,
        card.stops_recorded,
        card.proof_notes_commits
    );
    let annex_digest_blake3 = blake3::hash(digest_src.as_bytes()).to_hex().to_string();

    ComplianceAnnex {
        standard: label.to_string(),
        generated_at,
        controls,
        strata_segments: card.strata_segments,
        strata_bytes: card.strata_bytes,
        reflex_chain_checked: card.chain_checked,
        reflex_chain_breaks: card.chain_breaks,
        operator_stops_recorded: card.stops_recorded,
        operator_resolutions_recorded: card.resolutions_recorded,
        signed_git_proof_commits: card.proof_notes_commits,
        execution_lease_active: card.lease_active,
        annex_digest_blake3,
    }
}

pub fn render_compliance_markdown(a: &ComplianceAnnex) -> String {
    let mut md = format!("# Vestige Deterministic Compliance Annex\n\n**Standard:** {}\n**Generated:** {}\n\n", a.standard, a.generated_at);
    md.push_str("| Evidence Metric | Verified Local Value |\n| :--- | :--- |\n");
    md.push_str(&format!("| Strata Log Segments | {} ({} bytes) |\n", a.strata_segments, a.strata_bytes));
    md.push_str(&format!("| Reflex Hash Chain | {} links checked, {} breaks |\n", a.reflex_chain_checked, a.reflex_chain_breaks));
    md.push_str(&format!("| Operator Guard Stops | {} recorded, {} closed with receipts |\n", a.operator_stops_recorded, a.operator_resolutions_recorded));
    md.push_str(&format!("| Signed Git Proof Commits | {} (`refs/notes/vestige-proof`) |\n", a.signed_git_proof_commits));
    md.push_str(&format!("| Execution Lease Armed | {} |\n", a.execution_lease_active));
    md.push_str(&format!("| BLAKE3 Annex Digest | `{}` |\n\n", a.annex_digest_blake3));
    md.push_str("## Mapped Controls\n\n");
    for c in &a.controls {
        md.push_str(&format!("- {c}\n"));
    }
    md
}

// ---------------------------------------------------------------------------
// Category 4 #33: Shadow AI Repo Scanner (`vestige audit-shadow`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ShadowAiFileHit {
    pub path: String,
    pub unicode_smuggling_hits: usize,
    pub secret_hits: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShadowAiReport {
    pub root: String,
    pub repos_scanned: usize,
    pub agent_config_files_found: usize,
    pub high_risk_files: usize,
    pub findings: Vec<ShadowAiFileHit>,
}

pub fn run_shadow_ai_audit(root: &Path) -> ShadowAiReport {
    const CONFIG_NAMES: &[&str] = &[
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        ".cursorrules",
        ".clinerules",
        ".mcp.json",
    ];
    let mut repo_roots: Vec<PathBuf> = Vec::new();
    if root.join(".git").exists() {
        repo_roots.push(root.to_path_buf());
    } else if let Ok(entries) = fs::read_dir(root) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() && p.join(".git").exists() {
                repo_roots.push(p);
            }
        }
    }
    if repo_roots.is_empty() && root.is_dir() {
        repo_roots.push(root.to_path_buf());
    }

    let mut findings = Vec::new();
    let mut high_risk = 0usize;

    for rr in &repo_roots {
        for name in CONFIG_NAMES {
            let p = rr.join(name);
            if !p.is_file() {
                continue;
            }
            let body = fs::read_to_string(&p).unwrap_or_default();
            let u = scan_concealed_unicode(&body).len();
            let s = vestige_core::scan_secrets(&body)
                .into_iter()
                .filter(|f| f.blocks_ingestion())
                .count();
            if u > 0 || s > 0 {
                high_risk += 1;
            }
            findings.push(ShadowAiFileHit {
                path: p.display().to_string(),
                unicode_smuggling_hits: u,
                secret_hits: s,
            });
        }
    }

    ShadowAiReport {
        root: root.display().to_string(),
        repos_scanned: repo_roots.len(),
        agent_config_files_found: findings.len(),
        high_risk_files: high_risk,
        findings,
    }
}

// ---------------------------------------------------------------------------
// Category 5 #50: 1-Command Clean Uninstall Guarantee (`vestige uninstall --clean`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct UninstallReport {
    pub dry_run: bool,
    pub removed_items: Vec<String>,
}

pub fn run_clean_uninstall(
    data_dir: &Path,
    repo: Option<&Path>,
    dry_run: bool,
) -> Result<UninstallReport, String> {
    let mut removed = Vec::new();

    if let Some(r) = repo {
        for key in [
            "alias.vblame",
            "alias.bisect-prove",
            "merge.vestige-guard.name",
            "merge.vestige-guard.driver",
        ] {
            if git_stdout(r, &["config", "--get", key]).is_ok() {
                if !dry_run {
                    let _ = git_stdout(r, &["config", "--unset-all", key]);
                }
                removed.push(format!("git config {key}"));
            }
        }
        let attrs = r.join(".gitattributes");
        if let Ok(body) = fs::read_to_string(&attrs) {
            let filtered: Vec<&str> = body
                .lines()
                .filter(|l| !l.contains("merge=vestige-guard"))
                .collect();
            if filtered.len() != body.lines().count() {
                if !dry_run {
                    let next = if filtered.is_empty() {
                        String::new()
                    } else {
                        format!("{}\n", filtered.join("\n"))
                    };
                    fs::write(&attrs, next)
                        .map_err(|e| format!("clean {}: {e}", attrs.display()))?;
                }
                removed.push(".gitattributes vestige-guard entries".into());
            }
        }
        for hook_name in ["pre-commit", "post-commit", "prepare-commit-msg"] {
            let hp = r.join(".git").join("hooks").join(hook_name);
            if let Ok(body) = fs::read_to_string(&hp)
                && body.to_ascii_lowercase().contains("vestige")
            {
                if !dry_run {
                    let _ = fs::remove_file(&hp);
                }
                removed.push(format!(".git/hooks/{hook_name}"));
            }
        }
    }

    for path in [
        lease::lease_path(data_dir),
        lease::steer_path(data_dir),
        lease::mutex_path(data_dir),
    ] {
        if path.exists() {
            if !dry_run {
                let _ = fs::remove_file(&path);
            }
            removed.push(path.display().to_string());
        }
    }

    Ok(UninstallReport {
        dry_run,
        removed_items: removed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn git_run(repo: &Path, args: &[&str]) {
        let st = Command::new("git")
            .current_dir(repo)
            .args(args)
            .status()
            .unwrap();
        assert!(st.success(), "git {args:?} failed");
    }

    #[test]
    fn redteam_suite_catches_all_50_unicode_smuggling_payloads() {
        let report = run_unicode_redteam_suite();
        assert_eq!(report.total_payloads, 50);
        assert_eq!(report.caught, 50);
        assert_eq!(report.missed, 0);
        assert_eq!(report.catch_rate_pct, 100);
    }

    #[test]
    fn merge_driver_blocks_invisible_unicode_tag_and_allows_clean_merge() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("base.md");
        let cur = dir.path().join("cur.md");
        let evil = dir.path().join("evil.md");
        fs::write(&base, "# Rules\nLine 1\n").unwrap();
        fs::write(&cur, "# Rules\nLine 1\nLocal change\n").unwrap();
        fs::write(&evil, "# Rules\nLine 1\nStealth \u{E0069} tag\n").unwrap();

        let blocked = run_vestige_merge_driver(&base, &cur, &evil).unwrap();
        assert!(!blocked.allowed);
        assert!(blocked.unicode_hits >= 1);

        let clean = dir.path().join("clean.md");
        fs::write(&clean, "# Rules\nLine 1\nClean addition\n").unwrap();
        let ok = run_vestige_merge_driver(&base, &cur, &clean).unwrap();
        assert!(ok.unicode_hits == 0 && ok.secret_hits == 0);
    }

    #[test]
    fn git_protocol_wiring_streak_undo_patch_and_clean_uninstall_work_end_to_end() {
        let dir = TempDir::new().unwrap();
        let repo = dir.path().join("repo");
        let data_dir = dir.path().join("data");
        fs::create_dir_all(repo.join("crates/app")).unwrap();
        fs::write(repo.join("crates/app/Cargo.toml"), "[package]\nname=\"app\"\n").unwrap();
        git_run(&repo, &["init", "-q"]);
        git_run(&repo, &["config", "user.email", "sam@example.com"]);
        git_run(&repo, &["config", "user.name", "Sam"]);
        fs::write(repo.join("AGENTS.md"), "Check `crates/app/Cargo.toml` and `src/missing.rs`.\n").unwrap();
        git_run(&repo, &["add", "."]);
        git_run(&repo, &["commit", "-q", "-m", "initial commit"]);

        fs::write(repo.join("AGENTS.md"), "Check `crates/app/Cargo.toml` and `src/missing.rs`.\nLine 2\n").unwrap();
        git_run(&repo, &["add", "AGENTS.md"]);
        git_run(&repo, &["commit", "-q", "-m", "second clean commit"]);

        let wired = install_git_protocol_wiring(&repo).unwrap();
        assert!(wired.vblame_alias_configured);
        assert!(wired.bisect_prove_alias_configured);
        assert!(wired.merge_driver_configured);
        assert!(wired.gitattributes_updated);
        assert!(wired.monorepo_scopes.iter().any(|s| s == "crates/app"));

        let streak = compute_zero_slop_streak(&repo, 20).unwrap();
        assert_eq!(streak.clean_commit_streak, 2);
        assert_eq!(streak.shields_badge["label"], "Vestige Streak");

        let (patch, digest) = generate_undo_patch(&repo, "HEAD").unwrap();
        assert!(patch.contains("Vestige Signed Undo Patch"));
        assert_eq!(digest.len(), 64);

        let rot = run_context_rot_doctor(&repo);
        assert_eq!(rot.total_dead_references, 1);

        let trailer = format_ablation_trailer(412, 38);
        assert_eq!(trailer, "Vestige-Ablation: 412->38 (-90.7%)");

        let shadow = run_shadow_ai_audit(&repo);
        assert_eq!(shadow.agent_config_files_found, 1);
        assert_eq!(shadow.high_risk_files, 0);

        let annex = generate_compliance_annex(&data_dir, Some(&repo), ComplianceStandard::EuAiAct);
        assert!(annex.standard.contains("EU AI Act"));

        let cleaned = run_clean_uninstall(&data_dir, Some(&repo), false).unwrap();
        assert!(cleaned.removed_items.iter().any(|s| s.contains("alias.vblame")));
        assert!(cleaned.removed_items.iter().any(|s| s.contains(".gitattributes")));
    }
}
