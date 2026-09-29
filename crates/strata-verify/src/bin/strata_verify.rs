//! `strata-verify <dir>`
//!
//! `<dir>` is a migrated STRATA log, a live strata-store folder (`log/` and
//! `store.meta`), or the legacy `kernel.log` layout.

use std::path::{Path, PathBuf};

fn dir_has_segments(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .any(|entry| entry.path().extension().is_some_and(|ext| ext == "seg"))
}

fn main() {
    let Some(dir) = std::env::args().nth(1).map(PathBuf::from) else {
        eprintln!("usage: strata-verify <store-dir>");
        std::process::exit(2);
    };

    if dir_has_segments(&dir) {
        match strata_verify::migration::verify_migrated_log(&dir) {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("report serializes")
                );
                if report.ok && report.failures.is_empty() {
                    println!("OK");
                    std::process::exit(0);
                }
                println!("FAILED");
                for failure in &report.failures {
                    eprintln!("  {failure}");
                }
                std::process::exit(1);
            }
            Err(err) => {
                eprintln!("FAILED: {err}");
                std::process::exit(1);
            }
        }
    }

    if dir.join("log").is_dir() || dir.join("store.meta").is_file() {
        match strata_verify::live::verify_live_store(&dir) {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("report serializes")
                );
                if report.ok {
                    println!("OK");
                    std::process::exit(0);
                }
                println!("FAILED");
                for failure in &report.failures {
                    eprintln!("  {failure}");
                }
                std::process::exit(1);
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
