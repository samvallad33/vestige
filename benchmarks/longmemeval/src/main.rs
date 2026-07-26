//! LongMemEval retrieval harness for Vestige.
//!
//! Measures **retrieval only**: given a question and its haystack of chat
//! sessions, how often does Vestige surface the session that actually contains
//! the answer? No generation stage, no LLM judge, no API key, no network.
//!
//! The numbers here are **recall@k**, not end-to-end QA accuracy. They must
//! never be published as a bare "LongMemEval score". See README.md.
//!
//! Benchmark: <https://github.com/xiaowu0162/LongMemEval> (ICLR 2025,
//! arXiv:2410.10813). Data: `xiaowu0162/longmemeval-cleaned` on HuggingFace, MIT.
//!
//! Harness-validity decisions (v2, after adversarial audit):
//! - The 30 abstention instances (`question_id` ending `_abs`) are EXCLUDED
//!   from retrieval scoring, matching upstream's retrieval evaluation.
//! - Session ids never enter the store. Gold session ids in this dataset all
//!   contain the literal token "answer", and tags are FTS-indexed, so storing
//!   them would place a gold-correlated token inside the searchable index.
//!   Nodes are tagged with opaque `s<idx>` markers instead; the id mapping and
//!   the gold set live only in the harness.
//! - `--mode chunked` splits long sessions into <=CHUNK_BYTES pieces (never
//!   mid-turn) because the embedder truncates at 8192 BYTES: 63.5% of all
//!   sessions and 92.8% of gold sessions exceed that, so in whole-session mode
//!   semantic search is blind to their tails (FTS still indexes full text).
//!   Chunked mode removes that blindness while scoring stays session-level.
//! - After ingest, the harness asserts every node got an embedding; instances
//!   with missing embeddings are skipped and counted, never silently scored.
//! - Both any-gold recall@k and all-gold recall_all@k are reported: any-hit is
//!   the lenient variant, and multi-session questions need >1 session.
//! - Per-question results stream to a JSONL dump plus a manifest, so the run
//!   is auditable row by row rather than aggregate-only.
//!
//! Usage:
//!   lme-retrieval <dataset.json> [--mode whole|chunked|turn] [--limit N]
//!                 [--out results/run.jsonl]

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::PathBuf;

use serde::Deserialize;
use vestige_core::{IngestInput, Storage};

#[derive(Debug, Deserialize)]
struct Instance {
    question_id: String,
    question_type: String,
    question: String,
    #[allow(dead_code)]
    answer: serde_json::Value,
    #[allow(dead_code)]
    question_date: String,
    haystack_session_ids: Vec<String>,
    #[allow(dead_code)]
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
    answer_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct Turn {
    role: String,
    content: String,
    #[serde(default)]
    #[allow(dead_code)]
    has_answer: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// One node per session (production-faithful; embedder truncates at 8192 bytes).
    Whole,
    /// Long sessions split into <=CHUNK_BYTES nodes; scoring stays session-level.
    Chunked,
    /// One node per turn; scored turn-level via the dataset's has_answer labels.
    Turn,
}

const KS: [usize; 4] = [1, 3, 5, 10];
/// Candidates fetched per query; scored at every k in KS.
const RETRIEVE_N: i32 = 10;
/// Chunk budget: below the embedder's 8192-byte truncation with headroom.
const CHUNK_BYTES: usize = 7500;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path: PathBuf = match args.next() {
        Some(p) => p.into(),
        None => {
            eprintln!(
                "usage: lme-retrieval <dataset.json> [--mode whole|chunked|turn] [--limit N] [--out run.jsonl]"
            );
            std::process::exit(2);
        }
    };

    let mut mode = Mode::Whole;
    let mut limit: Option<usize> = None;
    let mut out: Option<PathBuf> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--mode" => {
                mode = match args.next().as_deref() {
                    Some("chunked") => Mode::Chunked,
                    Some("turn") => Mode::Turn,
                    Some("whole") | None => Mode::Whole,
                    Some(other) => {
                        eprintln!("unknown mode: {other}");
                        std::process::exit(2);
                    }
                }
            }
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()),
            "--out" => out = args.next().map(PathBuf::from),
            other => {
                eprintln!("unknown flag: {other}");
                std::process::exit(2);
            }
        }
    }

    let raw = std::fs::read_to_string(&path)?;
    let mut instances: Vec<Instance> = serde_json::from_str(&raw)?;
    let total_loaded = instances.len();

    // Upstream's retrieval evaluation skips abstention questions: there is no
    // valid answer session to retrieve. Scoring them would make these numbers
    // non-comparable, so they are excluded up front and the count is reported.
    let abstention: usize = instances
        .iter()
        .filter(|i| i.question_id.ends_with("_abs"))
        .count();
    instances.retain(|i| !i.question_id.ends_with("_abs"));
    if let Some(n) = limit {
        instances.truncate(n);
    }

    let mode_name = match mode {
        Mode::Whole => "whole-session",
        Mode::Chunked => "chunked-session",
        Mode::Turn => "turn",
    };
    eprintln!(
        "LongMemEval retrieval harness v2\n  dataset:    {}\n  loaded:     {total_loaded} ({abstention} _abs excluded from scoring)\n  scoring:    {} instances\n  mode:       {mode_name}\n  retrieval:  hybrid FTS5/BM25 + HNSW cosine, RRF k=60 fusion (shipped default; legacy weight params are no-ops)\n",
        path.display(),
        instances.len(),
    );

    let mut dump = match &out {
        Some(p) => {
            if let Some(dir) = p.parent() {
                std::fs::create_dir_all(dir)?;
            }
            Some(std::io::BufWriter::new(std::fs::File::create(p)?))
        }
        None => None,
    };

    let mut hits = [0usize; KS.len()];
    let mut hits_all = [0usize; KS.len()];
    let mut per_type: std::collections::BTreeMap<String, ([usize; KS.len()], [usize; KS.len()], usize)> =
        Default::default();
    let mut scored = 0usize;
    let mut skipped: Vec<(String, String)> = Vec::new();

    for (i, inst) in instances.iter().enumerate() {
        match run_instance(inst, mode) {
            Ok(outcome) => {
                scored += 1;
                let entry = per_type
                    .entry(inst.question_type.clone())
                    .or_insert(([0; KS.len()], [0; KS.len()], 0));
                entry.2 += 1;
                for (ki, k) in KS.iter().enumerate() {
                    if outcome.gold_ranks.iter().any(|r| *r < *k) {
                        hits[ki] += 1;
                        entry.0[ki] += 1;
                    }
                    // all-gold: every distinct gold unit found within top k.
                    if outcome.n_gold > 0
                        && outcome.distinct_gold_within(*k) == outcome.n_gold
                    {
                        hits_all[ki] += 1;
                        entry.1[ki] += 1;
                    }
                }
                if let Some(w) = dump.as_mut() {
                    let row = serde_json::json!({
                        "question_id": inst.question_id,
                        "question_type": inst.question_type,
                        "mode": mode_name,
                        "n_gold": outcome.n_gold,
                        "gold_ranks": outcome.gold_ranks,
                        "gold_units_ranked": outcome.gold_units,
                        "top": outcome.top,
                    });
                    writeln!(w, "{row}")?;
                }
            }
            Err(e) => {
                eprintln!("  ! {} skipped: {e}", inst.question_id);
                skipped.push((inst.question_id.clone(), e));
            }
        }
        if (i + 1) % 25 == 0 {
            eprintln!("  .. {}/{} scored", i + 1, instances.len());
        }
    }

    println!("\n=== LongMemEval-S retrieval-only results (v2 harness) ===");
    println!(
        "mode={mode_name} scored={scored} skipped={} abstention_excluded={abstention}",
        skipped.len()
    );
    if !skipped.is_empty() {
        println!("skipped ids (excluded from denominator — inspect before publishing):");
        for (id, e) in &skipped {
            println!("  {id}: {e}");
        }
    }
    println!("\noverall (any gold unit in top k / ALL gold units in top k):");
    for (ki, k) in KS.iter().enumerate() {
        println!(
            "  recall@{k:<3} {:>6.1}%   recall_all@{k:<3} {:>6.1}%   ({}/{} any)",
            pct(hits[ki], scored),
            pct(hits_all[ki], scored),
            hits[ki],
            scored
        );
    }
    println!("\nby question type (any-hit):");
    for (qtype, (h, _ha, n)) in &per_type {
        let cells: Vec<String> = KS
            .iter()
            .enumerate()
            .map(|(ki, k)| format!("@{k}={:.0}%", pct(h[ki], *n)))
            .collect();
        println!("  {qtype:<26} n={n:<4} {}", cells.join("  "));
    }
    println!(
        "\nNOTE: retrieval recall, NOT end-to-end QA accuracy. Report as\n\
         \"LongMemEval-S, session-level recall@k, retrieval-only ({mode_name})\"."
    );
    Ok(())
}

fn pct(num: usize, den: usize) -> f64 {
    if den == 0 { 0.0 } else { num as f64 * 100.0 / den as f64 }
}

struct Outcome {
    /// 0-based ranks (position in the returned list) holding any gold unit.
    gold_ranks: Vec<usize>,
    /// (gold_unit_index, rank) pairs for distinct-gold accounting.
    gold_units: Vec<(usize, usize)>,
    /// Number of distinct gold units (sessions, or gold turns in turn mode).
    n_gold: usize,
    /// Top of the ranked list as (unit_label, score) for the JSONL dump.
    top: Vec<(String, f32)>,
}

impl Outcome {
    fn distinct_gold_within(&self, k: usize) -> usize {
        let mut seen = HashSet::new();
        for (unit, rank) in &self.gold_units {
            if *rank < k {
                seen.insert(*unit);
            }
        }
        seen.len()
    }
}

fn run_instance(inst: &Instance, mode: Mode) -> Result<Outcome, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let storage = Storage::new(Some(dir.path().join("lme.db"))).map_err(|e| e.to_string())?;
    // Without this, embeddings are never generated and hybrid search silently
    // degrades to keyword-only — quietly understating the result.
    storage
        .init_embeddings()
        .map_err(|e| format!("init_embeddings failed: {e}"))?;

    let gold_sessions: HashSet<usize> = inst
        .haystack_session_ids
        .iter()
        .enumerate()
        .filter(|(_, sid)| inst.answer_session_ids.contains(sid))
        .map(|(idx, _)| idx)
        .collect();

    // gold unit index per ingested node tag; session ids stay OUT of the store.
    let mut node_gold: HashMap<String, usize> = HashMap::new(); // tag -> gold unit
    let mut tag_unit: HashMap<String, usize> = HashMap::new(); // tag -> unit (for dump labels)
    let mut n_nodes = 0usize;
    let mut gold_unit_count = 0usize;
    let mut gold_unit_ids: HashMap<usize, usize> = HashMap::new(); // session idx -> gold unit no.

    let ingest = |content: String, tag: String| -> Result<(), String> {
        storage
            .ingest(IngestInput {
                content,
                node_type: "event".to_string(),
                tags: vec![tag],
                ..Default::default()
            })
            .map(|_| ())
            .map_err(|e| e.to_string())
    };

    match mode {
        Mode::Whole | Mode::Chunked => {
            for (sidx, session) in inst.haystack_sessions.iter().enumerate() {
                let is_gold = gold_sessions.contains(&sidx);
                if is_gold {
                    gold_unit_ids.entry(sidx).or_insert_with(|| {
                        let u = gold_unit_count;
                        gold_unit_count += 1;
                        u
                    });
                }
                let pieces: Vec<String> = match mode {
                    Mode::Whole => {
                        vec![
                            session
                                .iter()
                                .map(|t| format!("{}: {}", t.role, t.content))
                                .collect::<Vec<_>>()
                                .join("\n"),
                        ]
                    }
                    _ => chunk_session(session, CHUNK_BYTES),
                };
                for (cidx, piece) in pieces.into_iter().enumerate() {
                    let tag = format!("s{sidx}.c{cidx}");
                    if is_gold {
                        node_gold.insert(tag.clone(), gold_unit_ids[&sidx]);
                    }
                    tag_unit.insert(tag.clone(), sidx);
                    ingest(piece, tag)?;
                    n_nodes += 1;
                }
            }
        }
        Mode::Turn => {
            for (sidx, session) in inst.haystack_sessions.iter().enumerate() {
                for (tidx, turn) in session.iter().enumerate() {
                    let tag = format!("s{sidx}.t{tidx}");
                    if turn.has_answer {
                        let u = gold_unit_count;
                        gold_unit_count += 1;
                        node_gold.insert(tag.clone(), u);
                    }
                    tag_unit.insert(tag.clone(), sidx);
                    ingest(format!("{}: {}", turn.role, turn.content), tag)?;
                    n_nodes += 1;
                }
            }
        }
    }

    // Embedding-integrity gate: if any node failed to embed, this instance is
    // not comparable (semantic side silently missing) — skip it, loudly.
    let stats = storage.get_stats().map_err(|e| e.to_string())?;
    if (stats.nodes_with_embeddings as usize) < n_nodes {
        return Err(format!(
            "embedding integrity: {}/{n_nodes} nodes embedded",
            stats.nodes_with_embeddings
        ));
    }

    let results = storage
        // Weights are no-ops in the shipped code (RRF k=60 is what runs); passed
        // for signature compatibility only.
        .hybrid_search(&inst.question, RETRIEVE_N, 0.3, 0.7)
        .map_err(|e| e.to_string())?;

    // In chunked mode, multiple chunks of one session can occupy several ranks;
    // collapse to session-level ranks (first appearance wins).
    let mut gold_ranks = Vec::new();
    let mut gold_units = Vec::new();
    let mut top = Vec::new();
    let mut seen_units: HashSet<usize> = HashSet::new();
    let mut session_rank = 0usize;
    for r in results.iter() {
        let tag = match r.node.tags.first() {
            Some(t) => t.clone(),
            None => continue,
        };
        let unit = match tag_unit.get(&tag) {
            Some(u) => *u,
            None => continue,
        };
        let collapse = matches!(mode, Mode::Whole | Mode::Chunked);
        if collapse && !seen_units.insert(unit) {
            continue; // later chunk of an already-ranked session
        }
        let rank = if collapse { session_rank } else { top.len() };
        top.push((format!("s{unit}"), r.combined_score));
        if let Some(g) = node_gold.get(&tag) {
            gold_ranks.push(rank);
            gold_units.push((*g, rank));
        }
        if collapse {
            session_rank += 1;
        }
    }

    let n_gold = match mode {
        Mode::Whole | Mode::Chunked => gold_unit_count,
        Mode::Turn => gold_unit_count,
    };
    Ok(Outcome {
        gold_ranks,
        gold_units,
        n_gold,
        top,
    })
}

/// Split a session into chunks of at most `budget` bytes, never mid-turn.
/// A single turn larger than the budget becomes its own (truncatable) chunk.
fn chunk_session(session: &[Turn], budget: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for t in session {
        let line = format!("{}: {}\n", t.role, t.content);
        if !cur.is_empty() && cur.len() + line.len() > budget {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push_str(&line);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}
