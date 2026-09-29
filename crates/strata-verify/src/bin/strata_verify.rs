//! Thin CLI for the standalone build: `strata-verify <dir>`. The vestige
//! binary exposes the same report as `vestige strata-verify <dir>`; this bin
//! exists so the crate is runnable without the vestige workspace.

use std::path::PathBuf;

fn main() {
    let Some(dir) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: strata-verify <store-dir>");
        std::process::exit(2);
    };
    // A migrated STRATA directory (MIGRATION_RECEIPT inside) gets the
    // migration verification: chain, receipt checksum + signature, and a
    // frame-count replay against the receipt.
    let has_migration_receipt = strata::StrataLog::open(&dir)
        .ok()
        .and_then(|log| log.read_frames(1).ok())
        .map(|frames| frames.iter().any(|f| f.kind == 46))
        .unwrap_or(false);

    if has_migration_receipt {
        match strata_verify::migration::verify_migrated_log(&dir) {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("report serializes")
                );
                if report.failures.is_empty() {
                    println!("OK");
                } else {
                    println!("FAILED");
                    for failure in &report.failures {
                        eprintln!("  {failure}");
                    }
                    std::process::exit(1);
                }
            }
            Err(err) => {
                eprintln!("FAILED: {err}");
                std::process::exit(1);
            }
        }
    }

    let report = strata_verify::verify_store(&dir);
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("report serializes")
    );
    if report.ok() {
        println!("OK");
    } else {
        println!("FAILED");
        for failure in &report.failures {
            eprintln!("  {failure}");
        }
        std::process::exit(1);
    }
}
