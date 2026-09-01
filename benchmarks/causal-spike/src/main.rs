//! Preregistered v3.0 causal-graph spike harness.
//! Protocol: docs/v3/CAUSAL-SPIKE-PROTOCOL.md (immutable after first `run`).

mod arms;
mod git;
mod probe;
mod rng;
mod run;
mod score;
mod seed;
mod types;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use types::{Arm, DEFAULT_SEED};

#[derive(Parser)]
#[command(name = "causal-spike")]
#[command(about = "Preregistered v3.0 causal-graph spike harness")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the three blinded stores + manifest (never overwrites a store).
    Seed {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = DEFAULT_SEED)]
        seed: u64,
    },
    /// Rank candidates for one arm. Never reads the manifest.
    Run {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        failures: PathBuf,
        #[arg(long)]
        arm: Arm,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Score run JSON against the ground-truth manifest. Writes results/.
    Score {
        #[arg(long)]
        runs: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Rank B for one id on a caller-supplied store copy. Never used on live.
    Probe {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        failure_id: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Seed { out, seed } => {
            let manifest = seed::seed(&out, seed)?;
            eprintln!("seeded {out:?} → {}", manifest.display());
        }
        Cmd::Run {
            store,
            failures,
            arm,
            out,
        } => {
            run::run(&store, &failures, arm, out.as_deref())?;
        }
        Cmd::Score { runs, manifest } => {
            let path = score::score(&runs, &manifest)?;
            eprintln!("scored → {}", path.display());
        }
        Cmd::Probe {
            store,
            failure_id,
            out,
        } => {
            let result = probe::probe(&store, &failure_id)?;
            let encoded = serde_json::to_vec_pretty(&result)?;
            if let Some(path) = out {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(path, &encoded)?;
            } else {
                println!("{}", String::from_utf8_lossy(&encoded));
            }
        }
    }
    Ok(())
}
