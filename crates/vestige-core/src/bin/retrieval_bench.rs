use std::io::{self, BufRead, Write};
use vestige_core::retrieval::{self, RankRequest};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: retrieval-bench [--worker]");
        eprintln!("Reads line-delimited JSON RankRequest from stdin and writes line-delimited JSON RankResponse to stdout.");
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
            }
            Err(e) => {
                let err_msg = serde_json::to_string(&e.to_string()).unwrap();
                let _ = writeln!(stdout, "{{\"error\":{}}}", err_msg);
                let _ = stdout.flush();
            }
        }
    }
}
