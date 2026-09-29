//! `vestige-upgrade` — the only 4.0 binary that links the v3 SQLite reader.
//!
//! ```text
//! vestige-upgrade --db <vestige.db>
//! vestige-upgrade migrate --from <src> [--to <dst>] [--dry-run] [--accept-wal-snapshot]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use strata_migrate::MigrateOptions;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("--db") => match args.next() {
            Some(path) if args.next().is_none() => upgrade_db(PathBuf::from(path)),
            _ => usage(),
        },
        Some("migrate") => migrate(args.collect()),
        Some("-h") | Some("--help") => {
            print_usage();
            ExitCode::SUCCESS
        }
        _ => usage(),
    }
}

fn upgrade_db(db: PathBuf) -> ExitCode {
    match vestige_upgrade::upgrade_if_needed(&db) {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            let _ = std::io::Write::flush(&mut std::io::stderr());
            ExitCode::from(1)
        }
    }
}

fn migrate(args: Vec<String>) -> ExitCode {
    let mut from = None;
    let mut to = None;
    let mut dry_run = false;
    let mut accept_wal_snapshot = false;
    let mut rest = args.into_iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--from" => from = rest.next().map(PathBuf::from),
            "--to" => to = rest.next().map(PathBuf::from),
            "--dry-run" => dry_run = true,
            "--accept-wal-snapshot" => accept_wal_snapshot = true,
            other => {
                eprintln!("unknown argument {other}");
                return usage();
            }
        }
    }
    let Some(from) = from else {
        eprintln!("migrate requires --from <src>");
        return usage();
    };
    let destination = match to {
        Some(dir) => dir,
        None => from
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("strata"),
    };
    let options = MigrateOptions {
        dry_run,
        accept_wal_snapshot,
        ..MigrateOptions::default()
    };
    let report = match strata_migrate::migrate_with_options(&from, &destination, options) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("import failed: {err}");
            return ExitCode::from(1);
        }
    };
    println!("=== Vestige migrate-to-strata ===");
    println!(
        "Source (never modified): {} (keep this file; it is your pre-migration record)",
        from.display()
    );
    if !report.source_blake3.is_empty() {
        println!("Source BLAKE3: {}", report.source_blake3);
    }
    println!("Destination: {}", destination.display());
    println!(
        "Migrated: {} nodes, {} edges, {} fsrs events",
        report.nodes, report.edges, report.fsrs_events
    );
    if dry_run {
        println!("Dry run: nothing was written.");
        return ExitCode::SUCCESS;
    }
    if !report.verify_passed || !report.receipt_verified {
        eprintln!("migration log failed verification");
        return ExitCode::from(1);
    }
    println!(
        "MIGRATION_RECEIPT: {} (signature verified: {})",
        report.receipt_digest.clone().unwrap_or_default(),
        report.receipt_verified
    );
    println!("Replay verification: {}", report.verify_passed);
    ExitCode::SUCCESS
}

fn print_usage() {
    eprintln!(
        "\
usage:
  vestige-upgrade --db <vestige.db>
  vestige-upgrade migrate --from <src> [--to <dst>] [--dry-run] [--accept-wal-snapshot]"
    );
}

fn usage() -> ExitCode {
    print_usage();
    ExitCode::from(2)
}
