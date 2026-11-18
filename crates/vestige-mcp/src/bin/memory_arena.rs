//! #835 Reproducible MemoryArena & Public Benchmark Suite CLI.
//!
//! Usage:
//!   cargo run --release --bin memory_arena -- --suite all
//!   cargo run --release --bin memory_arena -- --suite scorecard --json
//!   cargo run --release --bin memory_arena -- --suite stage1 --live

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Parser, ValueEnum};
use serde_json::{Value, json};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Suite {
    All,
    Scorecard,
    Stage1,
}

#[derive(Parser, Debug)]
#[command(
    name = "memory_arena",
    version,
    about = "Vestige Public MemoryArena Leaderboard & Reproducible Benchmark Runner"
)]
struct Args {
    /// Suite to render or reproduce
    #[arg(long, value_enum, default_value_t = Suite::All)]
    suite: Suite,
    /// Optional filter passed to benchmarks/run_stage1_retrieval.mjs (--only=<name>)
    #[arg(long)]
    only: Option<String>,
    /// Directory holding verified release JSON receipts
    #[arg(long, default_value = "benchmarks/results/v6.2.2")]
    results_dir: PathBuf,
    /// Re-execute live retrieval benchmarks via node benchmarks/run_stage1_retrieval.mjs
    #[arg(long)]
    live: bool,
    /// Emit machine-readable JSON
    #[arg(long)]
    json: bool,
}

fn find_results_dir(requested: &Path) -> PathBuf {
    if requested.is_dir() {
        return requested.to_path_buf();
    }
    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(requested);
    if fallback.is_dir() {
        return fallback;
    }
    requested.to_path_buf()
}

fn collect_receipts(dir: &Path) -> Vec<(String, Value)> {
    let mut rows = Vec::new();
    let Ok(rd) = fs::read_dir(dir) else {
        return rows;
    };
    let mut files: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    files.sort();
    for path in files {
        if let Ok(raw) = fs::read_to_string(&path)
            && let Ok(val) = serde_json::from_str::<Value>(&raw)
        {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            rows.push((name, val));
        }
    }
    rows
}

fn run_live_suite(only: Option<&str>) -> anyhow::Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script = repo_root.join("benchmarks/run_stage1_retrieval.mjs");
    let mut cmd = Command::new("node");
    cmd.current_dir(&repo_root).arg(&script);
    if let Some(filter) = only.filter(|s| !s.is_empty()) {
        cmd.arg(format!("--only={filter}"));
    }
    let status = cmd.status()?;
    if !status.success() {
        anyhow::bail!("live benchmark runner exited with {status}");
    }
    Ok(())
}

fn build_architecture_profiles() -> Vec<Value> {
    let mut list = Vec::new();
    list.push(json!({
        "system": "Vestige v6.2.2 (Local Rust)",
        "embeddings": "None (0)",
        "llm_calls": 0,
        "cost_per_1k_usd": 0.0,
        "rss_mb": 350,
        "proof": "BLAKE3 + Ed25519 + Strata"
    }));
    list.push(json!({
        "system": "Mem0 / Mem0-Graph",
        "embeddings": "External vector DB + embedding API",
        "llm_calls": "1-3 per write/query",
        "cost_per_1k_usd": "variable API",
        "proof": "None"
    }));
    list.push(json!({
        "system": "Letta / MemGPT",
        "embeddings": "Archival vector index",
        "llm_calls": "multi-turn tool loop",
        "cost_per_1k_usd": "variable API",
        "proof": "None"
    }));
    list.push(json!({
        "system": "Zep / Graphiti",
        "embeddings": "Embeddings + LLM extraction",
        "llm_calls": "per episode",
        "cost_per_1k_usd": "cloud/API",
        "proof": "None"
    }));
    list.push(json!({
        "system": "Microsoft GraphRAG",
        "embeddings": "Embeddings + LLM summaries",
        "llm_calls": "O(chunks)",
        "cost_per_1k_usd": "high ingest cost",
        "proof": "None"
    }));
    list
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.live || args.suite == Suite::Stage1 {
        run_live_suite(args.only.as_deref())?;
    }
    let dir = find_results_dir(&args.results_dir);
    let receipts = collect_receipts(&dir);
    let profiles = build_architecture_profiles();

    if args.json {
        let receipt_map: serde_json::Map<String, Value> = receipts.into_iter().collect();
        let payload = json!({
            "version": env!("CARGO_PKG_VERSION"),
            "results_dir": dir,
            "receipt_count": receipt_map.len(),
            "receipts": receipt_map,
            "architecture_comparison": profiles,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!(
        "=== Vestige v{} Public MemoryArena & Benchmark Suite ===",
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "Results dir: {} ({} JSON receipts)",
        dir.display(),
        receipts.len()
    );
    println!(
        "Engine: Pure Deterministic Integer Q16.16 Rust | 0 Embeddings | 0 LLM Calls | $0.00/1k queries"
    );
    println!();
    for (name, val) in &receipts {
        let keys = val.as_object().map(|o| o.len()).unwrap_or(0);
        println!("  [OK] receipt={name}.json ({keys} top-level fields)");
    }
    if receipts.is_empty() {
        println!(
            "  [INFO] No JSON receipts in {} yet; pass --live to execute benchmarks/run_stage1_retrieval.mjs.",
            dir.display()
        );
    }
    println!();
    println!("Use `--json` for full structured metrics or `--live` to run `node benchmarks/run_stage1_retrieval.mjs`.");
    Ok(())
}
