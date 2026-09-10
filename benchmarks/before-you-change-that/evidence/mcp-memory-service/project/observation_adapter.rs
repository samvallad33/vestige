//! Generic primitive adapter. It contains no evaluator cases or expected data.

#[path = "src/ledger/reconcile.rs"]
mod reconcile;

use reconcile::{ApplyState, EventKind, LedgerEvent, ReconcileOutcome, Reconciler};
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

fn safe(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_alphanumeric()
        || byte == b'-' || byte == b'_')
}

fn state_name(value: &ApplyState) -> &'static str {
    match value {
        ApplyState::Applied => "Applied",
        ApplyState::Rejected => "Rejected",
        ApplyState::Duplicate => "Duplicate",
    }
}

fn emit(case: &str, result: Result<ReconcileOutcome, String>) {
    match result {
        Ok(value) => println!(
            "REC|{}|OK|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            case, state_name(&value.state), value.selected_policy,
            value.policy_generation, value.reversal_horizon,
            value.journal_entries, value.balance_cents, value.checkpoint,
            value.pending_reversals, if value.recovered_marker { 1 } else { 0 },
        ),
        Err(_) => println!("REC|{}|ERR|||||||||", case),
    }
}

fn number(value: &str) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| "invalid number".to_string())
}

fn ledger(root: &Path, name: &str) -> Result<PathBuf, String> {
    if !safe(name) {
        return Err("unsafe ledger identifier".to_string());
    }
    Ok(root.join(name))
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() != 3 {
        println!("BOOT|ERR");
        std::process::exit(2);
    }
    let service = match fs::read_to_string(&arguments[1]).ok()
        .and_then(|text| Reconciler::from_text(&text).ok()) {
        Some(value) => value,
        None => {
            println!("BOOT|ERR");
            std::process::exit(2);
        }
    };
    let root = PathBuf::from(&arguments[2]);
    if fs::create_dir_all(&root).is_err() {
        println!("BOOT|ERR");
        std::process::exit(2);
    }
    println!("BOOT|OK");

    for raw in io::stdin().lock().lines() {
        let line = match raw { Ok(value) => value, Err(_) => break };
        let fields: Vec<&str> = line.split('|').collect();
        if fields.is_empty() || fields.len() < 3 || !safe(fields[1]) || !safe(fields[2]) {
            println!("INPUT|ERR");
            continue;
        }
        let case = fields[1];
        let state_dir = match ledger(&root, fields[2]) {
            Ok(value) => value,
            Err(_) => {
                println!("INPUT|ERR");
                continue;
            }
        };
        match fields[0] {
            "C" if fields.len() == 8 => {
                let result = match (number(fields[4]), number(fields[5])) {
                    (Ok(sequence), Ok(amount_cents)) => service.reconcile_one(&state_dir, LedgerEvent {
                        id: fields[3], sequence, schema: fields[6], wire_format: fields[7],
                        kind: EventKind::Charge { amount_cents },
                    }),
                    _ => Err("invalid charge".to_string()),
                };
                emit(case, result);
            }
            "R" if fields.len() == 9 => {
                let result = match (number(fields[4]), number(fields[6])) {
                    (Ok(sequence), Ok(target_sequence)) => service.reconcile_one(&state_dir, LedgerEvent {
                        id: fields[3], sequence, schema: fields[7], wire_format: fields[8],
                        kind: EventKind::Reversal { target_id: fields[5], target_sequence },
                    }),
                    _ => Err("invalid reversal".to_string()),
                };
                emit(case, result);
            }
            "E" if fields.len() == 3 => {
                let output = root.join(format!("{}.state", case));
                match service.export_snapshot(&state_dir, &output) {
                    Ok(value) => println!("EXP|{}|OK|{}|{}|{}|{}|{}|{}", case,
                        value.selected_policy, value.policy_generation, value.journal_entries,
                        value.balance_cents, value.checkpoint, value.pending_reversals),
                    Err(_) => println!("EXP|{}|ERR||||||", case),
                }
            }
            _ => println!("INPUT|ERR"),
        }
    }
}
