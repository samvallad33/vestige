use std::io::{self, BufRead, Write};
use vestige_core::retrieval::{self, RankRequest};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: retrieval-bench [--worker]");
        eprintln!(
            "Reads line-delimited JSON RankRequest from stdin and writes line-delimited JSON RankResponse to stdout."
        );
        return;
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<RankRequest>(trimmed) {
            Ok(req) => {
                let resp = retrieval::rank(&req);
                let json = serde_json::to_string(&resp).expect("Failed to serialize RankResponse");
                let _ = writeln!(stdout, "{}", json);
                let _ = stdout.flush();
                write_wh_pool_file();
            }
            Err(e) => {
                let err_msg = serde_json::to_string(&e.to_string()).unwrap();
                let _ = writeln!(stdout, "{{\"error\":{}}}", err_msg);
                let _ = stdout.flush();
            }
        }
    }
    let line = wh_pool_line();
    println!("{line}");
    eprintln!("{line}");
    write_wh_pool_file();
}

fn wh_pool_line() -> String {
    let (queries, swaps, below_k, vote_nz, against, lifts) =
        vestige_core::retrieval::wh_pool_counts();
    format!(
        "vestige_wh_pool queries={queries} swaps={swaps} below_k={below_k} wh_vote_nz={vote_nz} against_order={against} lifts_into_top5={lifts}"
    )
}

/// Overwrite the counter file after every batch so a harness kill still
/// leaves the latest totals. The rename keeps a half-written update from
/// replacing a good line.
fn write_wh_pool_file() {
    let Ok(path) = std::env::var("VESTIGE_WH_POOL_FILE") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let tmp = format!("{path}.tmp");
    if std::fs::write(&tmp, wh_pool_line() + "\n").is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}
