#[path = "src/ledger/reconcile.rs"]
mod reconcile;

use reconcile::{ApplyState, EventKind, LedgerEvent, Reconciler};
use std::fs;
use std::path::PathBuf;

fn main() {
    let service = Reconciler::from_text(include_str!("runtime.conf")).expect("configuration must parse");
    let root = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        std::env::temp_dir().join(format!("intention-public-{}", std::process::id()))
    });
    let ledger = root.join("ledger");
    fs::create_dir_all(&ledger).expect("ledger directory must be writable");

    let pending = service.reconcile_one(&ledger, LedgerEvent {
        id: "rev-late",
        sequence: 104,
        schema: "LEDGER_SCHEMA_V3",
        wire_format: "REVERSAL_LINK_V2",
        kind: EventKind::Reversal { target_id: "charge-late", target_sequence: 100 },
    }).expect("pending reversal must execute");
    let charge = service.reconcile_one(&ledger, LedgerEvent {
        id: "charge-late",
        sequence: 100,
        schema: "LEDGER_SCHEMA_V3",
        wire_format: "REVERSAL_LINK_V2",
        kind: EventKind::Charge { amount_cents: 2500 },
    }).expect("delayed charge must execute");
    let reconcile_ok = pending.state == ApplyState::Applied
        && charge.state == ApplyState::Applied
        && charge.balance_cents == 0
        && charge.pending_reversals == 0
        && charge.journal_entries == 3;
    println!("{}: delayed reversal reload policy={} checkpoint={} entries={} balance={} pending={}",
             if reconcile_ok { "PASS" } else { "FAIL" }, charge.selected_policy,
             charge.checkpoint, charge.journal_entries, charge.balance_cents,
             charge.pending_reversals);

    let export_path = root.join("export.state");
    let export = service.export_snapshot(&ledger, &export_path).expect("export must execute");
    let export_ok = export.selected_policy == "LARCH"
        && fs::read(&export_path).ok().as_deref()
            == Some(format!("ledger-export|{}|{}|{}|{}\n", export.journal_entries,
                            export.balance_cents, export.checkpoint,
                            export.pending_reversals).as_bytes());
    println!("{}: unrelated export policy={} generation={} entries={}",
             if export_ok { "PASS" } else { "FAIL" }, export.selected_policy,
             export.policy_generation, export.journal_entries);

    let passed = usize::from(reconcile_ok) + usize::from(export_ok);
    println!("Application checks: {}/2 passed", passed);
    if passed != 2 {
        std::process::exit(1);
    }
}
