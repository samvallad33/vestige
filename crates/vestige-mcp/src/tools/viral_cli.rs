//! Viral CLI commands for Vestige (`bench-proof`, `demo`, `diff-mind`, `vaccine`, `replay`).
//!
//! 100% local, zero-vector, zero-LLM, `$0.00` deterministic developer tools backed directly by
//! `vestige-core` retrieval, `strata-kernel` (`ALGO_V1`..=`ALGO_V4` FSRS-6/7), and local Strata.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use chrono::{Duration, Utc};
use serde::Serialize;
use strata_kernel::canonical::from_q32_32;
use strata_kernel::checkpoint::{Checkpoint, checkpoint_hash, state_root};
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::{
    ALGO_V1, ALGO_V2, ALGO_V3, ALGO_V4, FsrsFold, is_known_algo_version,
};
use strata_kernel::kernel::Kernel;
use strata_kernel::state::State;
use vestige_core::fsrs::{
    Rating, initial_state_fsrs7, next_interval_fsrs7, next_state_fsrs7, retrievability_fsrs7,
};
use vestige_core::retrieval::{Params, Query, RankRequest, Session, Turn, rank};
use vestige_core::{IngestInput, Storage};

// ============================================================================
// 1. `vestige bench-proof`
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct BenchProofCase {
    pub id: String,
    pub category: String,
    pub passed: bool,
    pub top1_id: String,
    pub expected_id: String,
    pub latency_us: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchProofReport {
    pub version: String,
    pub passed: usize,
    pub total: usize,
    pub total_latency_ms: f64,
    pub cost_usd: f64,
    pub vectors: usize,
    pub receipt_blake3: String,
    pub cases: Vec<BenchProofCase>,
}

fn mk_turn(id: &str, role: &str, content: &str) -> Turn {
    Turn {
        id: id.to_string(),
        role: role.to_string(),
        content: content.to_string(),
        links: Vec::new(),
        stability: 0.0,
    }
}

/// Run the live deterministic zero-vector retrieval + FSRS-7 proof suite on local CPU.
pub fn run_bench_proof() -> BenchProofReport {
    let t_start = Instant::now();
    let mut cases = Vec::new();

    // Case 1: Bi-temporal state supersession (updated deadline beats stale initial deadline)
    {
        let c_start = Instant::now();
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "s0".into(),
                    date: "2026-05-01".into(),
                    timestamp: 1,
                    turns: vec![mk_turn(
                        "t_old",
                        "user",
                        "I initially scheduled the database migration for Tuesday afternoon at 2 PM.",
                    )],
                },
                Session {
                    id: "s1".into(),
                    date: "2026-05-03".into(),
                    timestamp: 2,
                    turns: vec![mk_turn(
                        "t_new",
                        "user",
                        "Ops called to reschedule the database migration, and now it is moved to Wednesday at 9 AM.",
                    )],
                },
            ],
            queries: vec![Query {
                text: "What time is the database migration scheduled?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        let top1 = res.results[0]
            .turns
            .first()
            .map(|t| t.id.clone())
            .unwrap_or_default();
        cases.push(BenchProofCase {
            id: "supersession_update".into(),
            category: "Bi-Temporal Supersession".into(),
            passed: top1 == "t_new",
            top1_id: top1,
            expected_id: "t_new".into(),
            latency_us: c_start.elapsed().as_micros(),
        });
    }

    // Case 2: Exact entity / zero-entity steal guard
    {
        let c_start = Instant::now();
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "s0".into(),
                    date: "2026-06-01".into(),
                    timestamp: 1,
                    turns: vec![
                        mk_turn("t_noise", "user", "Let us check the general settings before launch."),
                        mk_turn(
                            "t_gold",
                            "user",
                            "I configured the Starlight renewal discount cap at 5% after reviewing margins.",
                        ),
                    ],
                },
                Session {
                    id: "s1".into(),
                    date: "2026-06-02".into(),
                    timestamp: 2,
                    turns: vec![mk_turn(
                        "t_decoy",
                        "user",
                        "All assets for the quarterly newsletter were approved ahead of schedule.",
                    )],
                },
            ],
            queries: vec![Query {
                text: "For the Starlight renewal, what is the approved discount percentage?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        let top1 = res.results[0]
            .turns
            .first()
            .map(|t| t.id.clone())
            .unwrap_or_default();
        cases.push(BenchProofCase {
            id: "exact_entity_anchor".into(),
            category: "Exact Entity Anchor".into(),
            passed: top1 == "t_gold",
            top1_id: top1,
            expected_id: "t_gold".into(),
            latency_us: c_start.elapsed().as_micros(),
        });
    }

    // Case 3: Causal Source-Attribution Snap (substantive user origin beats verbose assistant echo)
    {
        let c_start = Instant::now();
        let req = RankRequest {
            sessions: vec![Session {
                id: "s0".into(),
                date: "2026-03-15".into(),
                timestamp: 1,
                turns: vec![
                    mk_turn(
                        "u_origin",
                        "user",
                        "I'm working on two-week sprints and sprint one finishes on March 29 before our final deployment deadline on April 19 for transaction management features, can you help me plan?",
                    ),
                    mk_turn(
                        "a_echo",
                        "assistant",
                        "Certainly! Let us build a comprehensive sprint plan covering March 29, April 19, transaction management features, and the final deployment deadline with thorough weekly milestones.",
                    ),
                ],
            }],
            queries: vec![Query {
                text: "How many weeks are between finishing the transaction management features and the final deployment deadline?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        let top1 = res.results[0]
            .turns
            .first()
            .map(|t| t.id.clone())
            .unwrap_or_default();
        cases.push(BenchProofCase {
            id: "causal_source_snap".into(),
            category: "Source-Attribution Snap".into(),
            passed: top1 == "u_origin",
            top1_id: top1,
            expected_id: "u_origin".into(),
            latency_us: c_start.elapsed().as_micros(),
        });
    }

    // Case 4: Standing Policy / Constraint Priority
    {
        let c_start = Instant::now();
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "s0".into(),
                    date: "2026-04-01".into(),
                    timestamp: 1,
                    turns: vec![mk_turn(
                        "t_rule",
                        "user",
                        "Always include rollback verification steps whenever I ask you to draft a production deployment checklist.",
                    )],
                },
                Session {
                    id: "s1".into(),
                    date: "2026-04-10".into(),
                    timestamp: 2,
                    turns: vec![mk_turn(
                        "t_chatter",
                        "user",
                        "The staging deployment finished quickly today.",
                    )],
                },
            ],
            queries: vec![Query {
                text: "Draft a production deployment checklist for tonight's release.".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let res = rank(&req);
        let top1 = res.results[0]
            .turns
            .first()
            .map(|t| t.id.clone())
            .unwrap_or_default();
        cases.push(BenchProofCase {
            id: "standing_policy_rule".into(),
            category: "Standing Policy Recall".into(),
            passed: top1 == "t_rule",
            top1_id: top1,
            expected_id: "t_rule".into(),
            latency_us: c_start.elapsed().as_micros(),
        });
    }

    // Case 5: FSRS-7 (`ALGO_V4`) bit-exact dual-stability replay & monotonicity
    {
        let c_start = Instant::now();
        let k = Kernel::<ReviewEvent>::for_version(ALGO_V4).expect("ALGO_V4 registered");
        let events = [
            ReviewEvent { card_id: 1, rating: 3, event_seq: 1 },
            ReviewEvent { card_id: 1, rating: 4, event_seq: 5 },
            ReviewEvent { card_id: 1, rating: 1, event_seq: 20 },
            ReviewEvent { card_id: 1, rating: 3, event_seq: 23 },
        ];
        let mut s1 = State::default();
        let mut s2 = State::default();
        for e in &events {
            k.apply(&mut s1, e);
            k.apply(&mut s2, e);
        }
        let r1 = state_root(&s1);
        let r2 = state_root(&s2);
        let c = &s1.cards[&1];
        let ret_now = FsrsFold::retrievability_v4(c, s1.short_stability_q.get(&1).copied(), 23);
        let ret_30d = FsrsFold::retrievability_v4(c, s1.short_stability_q.get(&1).copied(), 53);
        let ok = r1 == r2 && ret_now == 1.0 && ret_30d < ret_now && ret_30d > 0.0;
        cases.push(BenchProofCase {
            id: "fsrs7_strata_replay".into(),
            category: "FSRS-7 Bit-Exact Replay".into(),
            passed: ok,
            top1_id: hex_prefix(&r1, 16),
            expected_id: hex_prefix(&r2, 16),
            latency_us: c_start.elapsed().as_micros(),
        });
    }

    let passed = cases.iter().filter(|c| c.passed).count();
    let total = cases.len();
    let total_latency_ms = (t_start.elapsed().as_micros() as f64) / 1000.0;

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vestige-bench-proof-v1:");
    for c in &cases {
        hasher.update(c.id.as_bytes());
        hasher.update(if c.passed { b":PASS:" } else { b":FAIL:" });
        hasher.update(c.top1_id.as_bytes());
    }
    let receipt_blake3 = hasher.finalize().to_hex().to_string();

    BenchProofReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        passed,
        total,
        total_latency_ms,
        cost_usd: 0.0,
        vectors: 0,
        receipt_blake3,
        cases,
    }
}

pub fn render_bench_proof_ascii(r: &BenchProofReport) -> String {
    let mut out = String::new();
    out.push_str("+------------------------------------------------------------------+\n");
    out.push_str(&format!(
        "| VESTIGE v{:<8} DETERMINISTIC BENCH-PROOF ({}/{} PASSED)          |\n",
        r.version, r.passed, r.total
    ));
    out.push_str("+------------------------------------------------------------------+\n");
    for c in &r.cases {
        let status = if c.passed { "PASS" } else { "FAIL" };
        out.push_str(&format!(
            "| [{}] {:<26} top1={:<16} {:>6} us |\n",
            status, c.category, c.top1_id, c.latency_us
        ));
    }
    out.push_str("+------------------------------------------------------------------+\n");
    out.push_str(&format!(
        "| Latency: {:>6.2} ms | Vectors: {} | LLM Calls: 0 | Cost: $0.00   |\n",
        r.total_latency_ms, r.vectors
    ));
    out.push_str(&format!(
        "| BLAKE3 Receipt: {:<48} |\n",
        &r.receipt_blake3[..48.min(r.receipt_blake3.len())]
    ));
    out.push_str("+------------------------------------------------------------------+\n");
    out
}

pub fn render_bench_proof_markdown(r: &BenchProofReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "## Vestige v{} Bench-Proof ({}/{} Passed)\n\n",
        r.version, r.passed, r.total
    ));
    out.push_str("| Check | Category | Result | Top-1 | Latency |\n");
    out.push_str("| :--- | :--- | :---: | :--- | ---: |\n");
    for c in &r.cases {
        out.push_str(&format!(
            "| `{}` | {} | {} | `{}` | {} us |\n",
            c.id,
            c.category,
            if c.passed { "PASS" } else { "FAIL" },
            c.top1_id,
            c.latency_us
        ));
    }
    out.push_str(&format!(
        "\n- Total latency: `{:.2} ms` | Vectors: `0` | LLM calls: `0` | API cost: `$0.00`\n- BLAKE3 receipt: `{}`\n",
        r.total_latency_ms, r.receipt_blake3
    ));
    out
}

// ============================================================================
// 2. `vestige demo`
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct DemoReport {
    pub winner_turn_id: String,
    pub winner_excerpt: String,
    pub retrieval_latency_us: u128,
    pub fsrs7_s_long_days: f64,
    pub fsrs7_s_short_days: f64,
    pub fsrs7_r_day1: f64,
    pub fsrs7_r_day7: f64,
    pub fsrs7_r_day30: f64,
    pub fsrs7_next_review_days: f64,
    pub strata_state_root: String,
    pub checkpoint_hash: String,
}

/// Run an ephemeral zero-pollution end-to-end demo of Strata + FSRS-7 + zero-vector retrieval.
pub fn run_demo() -> DemoReport {
    let t0 = Instant::now();
    let req = RankRequest {
        sessions: vec![
            Session {
                id: "day1_incident".into(),
                date: "2026-07-01".into(),
                timestamp: 1,
                turns: vec![mk_turn(
                    "m_initial",
                    "user",
                    "We initially set the Redis session TTL to 60 seconds, which caused checkout timeouts under spike load.",
                )],
            },
            Session {
                id: "day3_fix".into(),
                date: "2026-07-03".into(),
                timestamp: 2,
                turns: vec![mk_turn(
                    "m_verified_fix",
                    "user",
                    "Root cause confirmed: switched the Redis session TTL to 900 seconds with jittered refresh; checkout timeouts dropped to zero.",
                )],
            },
        ],
        queries: vec![Query {
            text: "What is the configured Redis session TTL?".into(),
            date: String::new(),
        }],
        top_k: 5,
        params: Params::default(),
    };
    let ranked = rank(&req);
    let retrieval_latency_us = t0.elapsed().as_micros();
    let winner_id = ranked.results[0]
        .turns
        .first()
        .map(|t| t.id.clone())
        .unwrap_or_default();
    let winner_excerpt = if winner_id == "m_verified_fix" {
        "Root cause confirmed: switched the Redis session TTL to 900 seconds with jittered refresh; checkout timeouts dropped to zero.".to_string()
    } else {
        winner_id.clone()
    };

    // Real FSRS-7 (`ALGO_V4`) dual-stability evolution
    let fs7_0 = initial_state_fsrs7(Rating::Good);
    let fs7_1 = next_state_fsrs7(fs7_0, 3.0, Rating::Easy);
    let r1 = retrievability_fsrs7(fs7_1, 1.0);
    let r7 = retrievability_fsrs7(fs7_1, 7.0);
    let r30 = retrievability_fsrs7(fs7_1, 30.0);
    let next_ivl = next_interval_fsrs7(fs7_1, 0.90);

    // Real Strata ALGO_V4 checkpoint seal
    let k = Kernel::<ReviewEvent>::for_version(ALGO_V4).expect("ALGO_V4");
    let mut state = State::default();
    k.apply(&mut state, &ReviewEvent { card_id: 101, rating: 3, event_seq: 1 });
    k.apply(&mut state, &ReviewEvent { card_id: 101, rating: 4, event_seq: 4 });
    let genesis = Checkpoint::genesis(ALGO_V4);
    let sealed = Checkpoint::seal(ALGO_V4, 4, checkpoint_hash(&genesis), &state);

    DemoReport {
        winner_turn_id: winner_id,
        winner_excerpt,
        retrieval_latency_us,
        fsrs7_s_long_days: fs7_1.s_long,
        fsrs7_s_short_days: fs7_1.s_short,
        fsrs7_r_day1: r1,
        fsrs7_r_day7: r7,
        fsrs7_r_day30: r30,
        fsrs7_next_review_days: next_ivl,
        strata_state_root: blake3::Hash::from_bytes(sealed.state_root).to_hex().to_string(),
        checkpoint_hash: blake3::Hash::from_bytes(sealed.hash()).to_hex().to_string(),
    }
}

pub fn render_demo_ascii(d: &DemoReport) -> String {
    format!(
        "=== VESTIGE LIVE DEMO (0 vectors, 0 LLM calls, $0.00 cost) ===\n\
         \n\
         1. Bi-Temporal Supersession Retrieval ({} us):\n\
            Query:  \"What is the configured Redis session TTL?\"\n\
            Rank 1: [{}] {}\n\
         \n\
         2. FSRS-7 (ALGO_V4) Dual-Stability Memory Curve:\n\
            S_long = {:.2}d | S_short = {:.2}d | Next 90% review = {:.1}d\n\
            Retention: Day 1 = {:.1}% | Day 7 = {:.1}% | Day 30 = {:.1}%\n\
         \n\
         3. Strata Cryptographic Proof (BLAKE3):\n\
            State Root:      {}\n\
            Checkpoint Hash: {}\n",
        d.retrieval_latency_us,
        d.winner_turn_id,
        d.winner_excerpt,
        d.fsrs7_s_long_days,
        d.fsrs7_s_short_days,
        d.fsrs7_next_review_days,
        d.fsrs7_r_day1 * 100.0,
        d.fsrs7_r_day7 * 100.0,
        d.fsrs7_r_day30 * 100.0,
        d.strata_state_root,
        d.checkpoint_hash,
    )
}

// ============================================================================
// 3. `vestige diff-mind`
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct MindDiffItem {
    pub id: String,
    pub node_type: String,
    pub retention_pct: f64,
    pub stability_days: f64,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiffMindReport {
    pub since_days: i64,
    pub total_memories: usize,
    pub added_count: usize,
    pub fading_count: usize,
    pub strongest_count: usize,
    pub added: Vec<MindDiffItem>,
    pub fading: Vec<MindDiffItem>,
    pub strongest: Vec<MindDiffItem>,
}

/// Compute a git-diff-style cognitive diff over the local Strata memory store.
pub fn run_diff_mind(storage: &Storage, since_days: i64) -> Result<DiffMindReport, String> {
    let nodes = storage
        .get_all_nodes(2000, 0)
        .map_err(|e| format!("failed to read memories: {e}"))?;
    let now = Utc::now();
    let cutoff = now - Duration::days(since_days.max(1));

    let mut added = Vec::new();
    let mut fading = Vec::new();
    let mut strongest = Vec::new();

    for n in &nodes {
        let excerpt: String = n
            .content
            .replace('\n', " ")
            .chars()
            .take(88)
            .collect();
        let item = MindDiffItem {
            id: n.id.clone(),
            node_type: n.node_type.clone(),
            retention_pct: (n.retention_strength * 100.0).clamp(0.0, 100.0),
            stability_days: n.stability,
            excerpt,
        };
        if n.created_at >= cutoff {
            added.push(item.clone());
        }
        if n.retention_strength < 0.60 {
            fading.push(item.clone());
        } else {
            strongest.push(item);
        }
    }

    added.sort_by(|a, b| b.stability_days.partial_cmp(&a.stability_days).unwrap_or(std::cmp::Ordering::Equal));
    fading.sort_by(|a, b| a.retention_pct.partial_cmp(&b.retention_pct).unwrap_or(std::cmp::Ordering::Equal));
    strongest.sort_by(|a, b| b.stability_days.partial_cmp(&a.stability_days).unwrap_or(std::cmp::Ordering::Equal));

    let added_count = added.len();
    let fading_count = fading.len();
    let strongest_count = strongest.len();
    added.truncate(12);
    fading.truncate(10);
    strongest.truncate(10);

    Ok(DiffMindReport {
        since_days,
        total_memories: nodes.len(),
        added_count,
        fading_count,
        strongest_count,
        added,
        fading,
        strongest,
    })
}

pub fn render_diff_mind_text(r: &DiffMindReport, markdown: bool) -> String {
    let mut out = String::new();
    if markdown {
        out.push_str(&format!(
            "## Vestige Cognitive Diff (last {}d | {} total memories)\n\n",
            r.since_days, r.total_memories
        ));
    } else {
        out.push_str(&format!(
            "=== VESTIGE DIFF-MIND (last {}d | {} total memories) ===\n",
            r.since_days, r.total_memories
        ));
    }
    out.push_str(&format!(
        "+ Newly Anchored: {} | Strong Core: {} | Fading (<60%): {}\n\n",
        r.added_count, r.strongest_count, r.fading_count
    ));
    if !r.added.is_empty() {
        out.push_str("--- Newly Anchored (+) ---\n");
        for it in &r.added {
            out.push_str(&format!(
                "  + [{}] ({}, {:.0}% ret) {}\n",
                it.id, it.node_type, it.retention_pct, it.excerpt
            ));
        }
        out.push('\n');
    }
    if !r.fading.is_empty() {
        out.push_str("--- Fading Memories Due For Review (-) ---\n");
        for it in &r.fading {
            out.push_str(&format!(
                "  - [{}] ({:.0}% ret, S={:.1}d) {}\n",
                it.id, it.retention_pct, it.stability_days, it.excerpt
            ));
        }
        out.push('\n');
    }
    if !r.strongest.is_empty() {
        out.push_str("--- Strongest Load-Bearing Invariants (^) ---\n");
        for it in &r.strongest {
            out.push_str(&format!(
                "  ^ [{}] (S={:.1}d, {:.0}% ret) {}\n",
                it.id, it.stability_days, it.retention_pct, it.excerpt
            ));
        }
    }
    out
}

// ============================================================================
// 4. `vestige vaccine`
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct VaccineAntibody {
    pub file: String,
    pub bug_commits: usize,
    pub revert_commits: usize,
    pub rule: String,
    pub sample_subject: String,
    pub anchored_memory_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VaccineReport {
    pub repo: String,
    pub commits_scanned: usize,
    pub injected: bool,
    pub antibodies: Vec<VaccineAntibody>,
}

/// Scan git history for recurring bug/revert hotspots and synthesize deterministic antibodies.
pub fn run_vaccine(
    repo: &Path,
    max_commits: usize,
    inject: bool,
    storage: Option<&Storage>,
) -> Result<VaccineReport, String> {
    let limit = max_commits.clamp(10, 2000).to_string();
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args([
            "log",
            "-n",
            &limit,
            "--name-only",
            "--pretty=format:===COMMIT===%H|%s",
        ])
        .output()
        .map_err(|e| format!("failed to run git log: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git log failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut commits_scanned = 0usize;
    let mut cur_is_bug = false;
    let mut cur_is_revert = false;
    let mut cur_subject = String::new();
    // file -> (bug_count, revert_count, latest_subject)
    let mut hotspots: BTreeMap<String, (usize, usize, String)> = BTreeMap::new();

    for raw in stdout.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("===COMMIT===") {
            commits_scanned += 1;
            let subj = rest.split_once('|').map(|(_, s)| s).unwrap_or("");
            let low = subj.to_ascii_lowercase();
            cur_is_revert = low.starts_with("revert ") || low.contains("revert \"");
            cur_is_bug = cur_is_revert
                || low.starts_with("fix")
                || low.contains("hotfix")
                || low.contains("regression")
                || low.contains("bugfix")
                || low.contains("panic");
            cur_subject = subj.to_string();
        } else if !line.is_empty() && cur_is_bug {
            let entry = hotspots
                .entry(line.to_string())
                .or_insert_with(|| (0, 0, cur_subject.clone()));
            entry.0 += 1;
            if cur_is_revert {
                entry.1 += 1;
            }
        }
    }

    let mut ranked: Vec<(String, usize, usize, String)> = hotspots
        .into_iter()
        .filter(|(_, (bugs, revs, _))| *bugs >= 2 || *revs >= 1)
        .map(|(f, (b, r, s))| (f, b, r, s))
        .collect();
    ranked.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(10);

    let mut antibodies = Vec::new();
    for (file, bugs, revs, sample) in ranked {
        let rule = format!(
            "VACCINE GUARD for `{file}` ({bugs} fix commits, {revs} reverts; latest: \"{sample}\"): run targeted tests + `vestige fence --target {file}` before modifying."
        );
        let mut anchored_id = None;
        if inject && let Some(st) = storage {
            let input = IngestInput {
                content: rule.clone(),
                node_type: "correction".to_string(),
                source: Some(format!("vestige-vaccine:{file}")),
                tags: vec!["vaccine".to_string(), "guard".to_string()],
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            };
            if let Ok(node) = st.ingest(input) {
                anchored_id = Some(node.id);
            }
        }
        antibodies.push(VaccineAntibody {
            file,
            bug_commits: bugs,
            revert_commits: revs,
            rule,
            sample_subject: sample,
            anchored_memory_id: anchored_id,
        });
    }

    Ok(VaccineReport {
        repo: repo.display().to_string(),
        commits_scanned,
        injected: inject && storage.is_some(),
        antibodies,
    })
}

pub fn render_vaccine_text(r: &VaccineReport, markdown: bool) -> String {
    let mut out = String::new();
    if markdown {
        out.push_str(&format!(
            "## Vestige Regression Vaccine ({} commits scanned, {} antibodies)\n\n",
            r.commits_scanned,
            r.antibodies.len()
        ));
    } else {
        out.push_str(&format!(
            "=== VESTIGE VACCINE ({} commits scanned | {} regression antibodies | injected={}) ===\n",
            r.commits_scanned,
            r.antibodies.len(),
            r.injected
        ));
    }
    if r.antibodies.is_empty() {
        out.push_str("No repeat regression hotspots found in the scanned commit window.\n");
        return out;
    }
    for a in &r.antibodies {
        let id_tag = a
            .anchored_memory_id
            .as_deref()
            .map(|id| format!(" [anchored: {id}]"))
            .unwrap_or_default();
        out.push_str(&format!(
            "- `{}` ({} fixes, {} reverts){}\n  {}\n",
            a.file, a.bug_commits, a.revert_commits, id_tag, a.rule
        ));
    }
    out
}

// ============================================================================
// 5. `vestige replay`
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ReplayStep {
    pub step: usize,
    pub event_seq: u64,
    pub rating: u8,
    pub s_long_days: f64,
    pub s_short_days: Option<f64>,
    pub difficulty: f64,
    pub state_root: String,
    pub checkpoint_hash: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayComparison {
    pub elapsed_days: u64,
    pub fsrs6_v3_r: f64,
    pub fsrs7_v4_r: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayReport {
    pub algo_version: u32,
    pub algo_label: String,
    pub deterministic_verified: bool,
    pub final_state_root: String,
    pub steps: Vec<ReplayStep>,
    pub fsrs6_vs_fsrs7: Vec<ReplayComparison>,
}

fn algo_name(v: u32) -> &'static str {
    match v {
        ALGO_V1 => "ALGO_V1 (Strata FSRS-v1)",
        ALGO_V2 => "ALGO_V2 (Strata FSRS-v2)",
        ALGO_V3 => "ALGO_V3 (Official FSRS-6 21-param)",
        ALGO_V4 => "ALGO_V4 (Official FSRS-7 34-param Dual-Stability)",
        _ => "UNKNOWN",
    }
}

/// Deterministic time-travel replay of a Strata review trace under `algo_version` (`1..=4`).
pub fn run_replay(algo_version: u32, compare: bool) -> Result<ReplayReport, String> {
    if !is_known_algo_version(algo_version) {
        return Err(format!("unknown algorithm version {algo_version} (supported: 1, 2, 3, 4)"));
    }
    let trace = [
        ReviewEvent { card_id: 1, rating: 3, event_seq: 1 },
        ReviewEvent { card_id: 1, rating: 3, event_seq: 5 },
        ReviewEvent { card_id: 1, rating: 4, event_seq: 16 },
        ReviewEvent { card_id: 1, rating: 1, event_seq: 40 },
        ReviewEvent { card_id: 1, rating: 3, event_seq: 42 },
        ReviewEvent { card_id: 1, rating: 4, event_seq: 56 },
    ];
    let kernel = Kernel::<ReviewEvent>::for_version(algo_version)
        .map_err(|e| format!("kernel error: {e}"))?;

    let mut state = State::default();
    let mut prev_cp = Checkpoint::genesis(algo_version);
    let mut steps = Vec::with_capacity(trace.len());
    for (idx, ev) in trace.iter().enumerate() {
        kernel.apply(&mut state, ev);
        let cp = Checkpoint::seal(algo_version, ev.event_seq, checkpoint_hash(&prev_cp), &state);
        let card = &state.cards[&1];
        steps.push(ReplayStep {
            step: idx + 1,
            event_seq: ev.event_seq,
            rating: ev.rating,
            s_long_days: from_q32_32(card.stability_q),
            s_short_days: state.short_stability_q.get(&1).map(|&q| from_q32_32(q)),
            difficulty: from_q32_32(card.difficulty_q),
            state_root: hex_prefix(&cp.state_root, 24),
            checkpoint_hash: hex_prefix(&cp.hash(), 24),
        });
        prev_cp = cp;
    }

    // Second independent replay to prove bit-exact determinism.
    let mut state_check = State::default();
    for ev in &trace {
        kernel.apply(&mut state_check, ev);
    }
    let root1 = state_root(&state);
    let root2 = state_root(&state_check);

    let mut fsrs6_vs_fsrs7 = Vec::new();
    if compare {
        let k3 = Kernel::<ReviewEvent>::for_version(ALGO_V3).expect("v3");
        let k4 = Kernel::<ReviewEvent>::for_version(ALGO_V4).expect("v4");
        let mut s3 = State::default();
        let mut s4 = State::default();
        for ev in &trace {
            k3.apply(&mut s3, ev);
            k4.apply(&mut s4, ev);
        }
        let c3 = &s3.cards[&1];
        let c4 = &s4.cards[&1];
        let ss4 = s4.short_stability_q.get(&1).copied();
        for elapsed in [1u64, 3, 7, 14, 30, 90] {
            let r3 = FsrsFold::retrievability(c3, c3.last_seq + elapsed, ALGO_V3).unwrap_or(0.0);
            let r4 = FsrsFold::retrievability_v4(c4, ss4, c4.last_seq + elapsed);
            fsrs6_vs_fsrs7.push(ReplayComparison {
                elapsed_days: elapsed,
                fsrs6_v3_r: r3,
                fsrs7_v4_r: r4,
            });
        }
    }

    Ok(ReplayReport {
        algo_version,
        algo_label: algo_name(algo_version).to_string(),
        deterministic_verified: root1 == root2,
        final_state_root: blake3::Hash::from_bytes(root1).to_hex().to_string(),
        steps,
        fsrs6_vs_fsrs7,
    })
}

pub fn render_replay_text(r: &ReplayReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== VESTIGE STRATA REPLAY | {} | bit-exact={} ===\n",
        r.algo_label, r.deterministic_verified
    ));
    for s in &r.steps {
        let ss = s
            .s_short_days
            .map(|v| format!(" S_short={v:>6.2}d"))
            .unwrap_or_default();
        out.push_str(&format!(
            "  step {:>2} | seq={:<3} G={} | S={:>6.2}d{} D={:.2} | root={} cp={}\n",
            s.step, s.event_seq, s.rating, s.s_long_days, ss, s.difficulty, s.state_root, s.checkpoint_hash
        ));
    }
    out.push_str(&format!("Final BLAKE3 state_root: {}\n", r.final_state_root));
    if !r.fsrs6_vs_fsrs7.is_empty() {
        out.push_str("\n--- FSRS-6 (ALGO_V3) vs FSRS-7 (ALGO_V4) Retention Curve ---\n");
        for cmp in &r.fsrs6_vs_fsrs7 {
            out.push_str(&format!(
                "  +{:>2}d | FSRS-6 R = {:>5.1}% | FSRS-7 R = {:>5.1}%\n",
                cmp.elapsed_days,
                cmp.fsrs6_v3_r * 100.0,
                cmp.fsrs7_v4_r * 100.0
            ));
        }
    }
    out
}

fn hex_prefix(bytes: &[u8; 32], chars: usize) -> String {
    let full = blake3::Hash::from_bytes(*bytes).to_hex().to_string();
    full.chars().take(chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_proof_passes_all_five_deterministic_checks() {
        let r = run_bench_proof();
        assert_eq!(r.passed, r.total, "expected 5/5 bench-proof checks to pass: {:#?}", r.cases);
        assert_eq!(r.vectors, 0);
        assert_eq!(r.cost_usd, 0.0);
        assert_eq!(r.receipt_blake3.len(), 64);
    }

    #[test]
    fn demo_surfaces_verified_fix_and_seals_fsrs7_checkpoint() {
        let d = run_demo();
        assert_eq!(d.winner_turn_id, "m_verified_fix");
        assert!(d.fsrs7_s_long_days > 0.0);
        assert!(d.fsrs7_s_short_days <= 0.8 * d.fsrs7_s_long_days + 1e-6);
        assert!(d.fsrs7_r_day1 > d.fsrs7_r_day7 && d.fsrs7_r_day7 > d.fsrs7_r_day30);
        assert_eq!(d.strata_state_root.len(), 64);
    }

    #[test]
    fn replay_verifies_bit_exact_determinism_across_v1_to_v4() {
        for v in [ALGO_V1, ALGO_V2, ALGO_V3, ALGO_V4] {
            let r = run_replay(v, v == ALGO_V4).expect("valid algo");
            assert!(r.deterministic_verified);
            assert_eq!(r.steps.len(), 6);
            if v == ALGO_V4 {
                assert!(r.steps.iter().all(|s| s.s_short_days.is_some()));
                assert_eq!(r.fsrs6_vs_fsrs7.len(), 6);
            }
        }
        assert!(run_replay(99, false).is_err());
    }
}
