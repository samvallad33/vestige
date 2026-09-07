//! Append-only synthetic ledger reconciliation with deterministic local files.
//!
//! There is no network, database, or real payment effect. Each call reloads the
//! journal from disk, appends new facts, and atomically refreshes derived state.

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

#[derive(Clone, Debug)]
struct Binding {
    policy: String,
    generation: u64,
}

#[derive(Clone, Debug)]
struct Policy {
    scope: String,
    generation: u64,
    schema: String,
    wire_format: String,
    reversal_horizon: u64,
    checkpoint_mode: String,
}

#[derive(Default)]
struct PolicyBuilder {
    scope: Option<String>,
    generation: Option<u64>,
    schema: Option<String>,
    wire_format: Option<String>,
    reversal_horizon: Option<u64>,
    checkpoint_mode: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Reconciler {
    routes: HashMap<String, Binding>,
    policies: HashMap<String, Policy>,
}

#[derive(Clone, Debug)]
pub enum EventKind<'a> {
    Charge { amount_cents: u64 },
    Reversal { target_id: &'a str, target_sequence: u64 },
}

#[derive(Clone, Debug)]
pub struct LedgerEvent<'a> {
    pub id: &'a str,
    pub sequence: u64,
    pub schema: &'a str,
    pub wire_format: &'a str,
    pub kind: EventKind<'a>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyState {
    Applied,
    Rejected,
    Duplicate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReconcileOutcome {
    pub state: ApplyState,
    pub selected_policy: String,
    pub policy_generation: u64,
    pub reversal_horizon: u64,
    pub journal_entries: usize,
    pub balance_cents: i64,
    pub checkpoint: u64,
    pub pending_reversals: usize,
    pub recovered_marker: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportOutcome {
    pub selected_policy: String,
    pub policy_generation: u64,
    pub journal_entries: usize,
    pub balance_cents: i64,
    pub checkpoint: u64,
    pub pending_reversals: usize,
}

#[derive(Clone, Debug)]
struct Pending {
    reversal_id: String,
    target_id: String,
    target_sequence: u64,
}

#[derive(Default)]
struct Loaded {
    event_ids: HashSet<String>,
    charges: HashMap<String, (u64, u64)>,
    reversed_targets: HashSet<String>,
    pending: Vec<Pending>,
    journal_entries: usize,
    max_sequence: u64,
    balance_cents: i64,
    checkpoint: u64,
}

impl Reconciler {
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut routes = HashMap::new();
        let mut builders: HashMap<String, PolicyBuilder> = HashMap::new();
        for (index, raw_line) in text.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("invalid config line {}", index + 1))?;
            let key = key.trim();
            let value = value.trim();
            if let Some(route) = key.strip_prefix("route.") {
                let (policy, generation_text) = value
                    .split_once('@')
                    .ok_or_else(|| format!("route {} lacks a generation", route))?;
                let generation = generation_text
                    .parse::<u64>()
                    .map_err(|_| format!("route {} has invalid generation", route))?;
                routes.insert(route.to_string(), Binding { policy: policy.to_string(), generation });
                continue;
            }
            if let Some(rest) = key.strip_prefix("policy.") {
                let (name, field) = rest
                    .rsplit_once('.')
                    .ok_or_else(|| format!("invalid policy key {}", key))?;
                let builder = builders.entry(name.to_string()).or_default();
                match field {
                    "scope" => builder.scope = Some(value.to_string()),
                    "generation" => builder.generation = Some(parse_number(name, field, value)?),
                    "schema" => builder.schema = Some(value.to_string()),
                    "wire_format" => builder.wire_format = Some(value.to_string()),
                    "REVERSAL_HORIZON" => builder.reversal_horizon = Some(parse_number(name, field, value)?),
                    "checkpoint_mode" => builder.checkpoint_mode = Some(value.to_string()),
                    _ => return Err(format!("unknown policy field {}", field)),
                }
                continue;
            }
            if key.starts_with("diagnostic.") {
                continue;
            }
            return Err(format!("unknown config key {}", key));
        }

        let mut policies = HashMap::new();
        for (name, builder) in builders {
            let error_name = name.clone();
            let missing = |field: &str| format!("policy {} lacks {}", error_name, field);
            let policy = Policy {
                scope: builder.scope.ok_or_else(|| missing("scope"))?,
                generation: builder.generation.ok_or_else(|| missing("generation"))?,
                schema: builder.schema.ok_or_else(|| missing("schema"))?,
                wire_format: builder.wire_format.ok_or_else(|| missing("wire_format"))?,
                reversal_horizon: builder.reversal_horizon.ok_or_else(|| missing("REVERSAL_HORIZON"))?,
                checkpoint_mode: builder.checkpoint_mode.ok_or_else(|| missing("checkpoint_mode"))?,
            };
            policies.insert(name, policy);
        }
        Ok(Self { routes, policies })
    }

    fn selected(&self, route: &str, scope: &str) -> Result<(&Binding, &Policy), String> {
        let binding = self.routes.get(route).ok_or_else(|| format!("unknown route {}", route))?;
        let policy = self.policies.get(&binding.policy)
            .ok_or_else(|| format!("unknown policy {}", binding.policy))?;
        if policy.generation != binding.generation {
            return Err("policy generation mismatch".to_string());
        }
        if policy.scope != scope {
            return Err("policy scope mismatch".to_string());
        }
        Ok((binding, policy))
    }

    pub fn reconcile_one(&self, state_dir: &Path, event: LedgerEvent<'_>) -> Result<ReconcileOutcome, String> {
        let (binding, policy) = self.selected("RECONCILE", "reconciliation")?;
        fs::create_dir_all(state_dir).map_err(|error| error.to_string())?;
        let marker = state_dir.join("reconcile.marker");
        let recovered_marker = marker.exists();
        fs::write(&marker, b"in-progress\n").map_err(|error| error.to_string())?;
        let result = self.reconcile_inner(state_dir, event, binding, policy, recovered_marker);
        let _ = fs::remove_file(&marker);
        result
    }

    fn reconcile_inner(
        &self,
        state_dir: &Path,
        event: LedgerEvent<'_>,
        binding: &Binding,
        policy: &Policy,
        recovered_marker: bool,
    ) -> Result<ReconcileOutcome, String> {
        let mut loaded = load_state(state_dir)?;
        if loaded.event_ids.contains(event.id) {
            return Ok(outcome(ApplyState::Duplicate, binding, policy, &loaded, recovered_marker));
        }
        if event.schema != policy.schema || event.wire_format != policy.wire_format {
            return Ok(outcome(ApplyState::Rejected, binding, policy, &loaded, recovered_marker));
        }
        if event.sequence <= loaded.checkpoint {
            return Ok(outcome(ApplyState::Rejected, binding, policy, &loaded, recovered_marker));
        }

        let mut lines = Vec::new();
        match event.kind {
            EventKind::Charge { amount_cents } => {
                if amount_cents > i64::MAX as u64 {
                    return Ok(outcome(ApplyState::Rejected, binding, policy, &loaded, recovered_marker));
                }
                lines.push(format!("C|{}|{}|{}", event.id, event.sequence, amount_cents));
                loaded.event_ids.insert(event.id.to_string());
                loaded.charges.insert(event.id.to_string(), (event.sequence, amount_cents));
                loaded.balance_cents += amount_cents as i64;
                loaded.max_sequence = loaded.max_sequence.max(event.sequence);

                let pending: Vec<Pending> = loaded.pending.iter()
                    .filter(|item| item.target_id == event.id && item.target_sequence == event.sequence)
                    .cloned()
                    .collect();
                for item in pending {
                    if !loaded.reversed_targets.contains(event.id) {
                        lines.push(format!("T|{}|{}|{}", item.reversal_id, event.id, amount_cents));
                        loaded.reversed_targets.insert(event.id.to_string());
                        loaded.balance_cents -= amount_cents as i64;
                    }
                    loaded.pending.retain(|candidate| candidate.reversal_id != item.reversal_id);
                }
            }
            EventKind::Reversal { target_id, target_sequence } => {
                if event.sequence < target_sequence
                    || event.sequence - target_sequence > policy.reversal_horizon
                    || loaded.reversed_targets.contains(target_id)
                {
                    return Ok(outcome(ApplyState::Rejected, binding, policy, &loaded, recovered_marker));
                }
                loaded.event_ids.insert(event.id.to_string());
                loaded.max_sequence = loaded.max_sequence.max(event.sequence);
                if let Some((charge_sequence, amount_cents)) = loaded.charges.get(target_id).copied() {
                    if charge_sequence != target_sequence {
                        return Ok(outcome(ApplyState::Rejected, binding, policy, &loaded, recovered_marker));
                    }
                    lines.push(format!("R|{}|{}|{}|{}|{}", event.id, event.sequence,
                                       target_id, target_sequence, amount_cents));
                    loaded.reversed_targets.insert(target_id.to_string());
                    loaded.balance_cents -= amount_cents as i64;
                } else {
                    lines.push(format!("P|{}|{}|{}|{}", event.id, event.sequence,
                                       target_id, target_sequence));
                    loaded.pending.push(Pending {
                        reversal_id: event.id.to_string(),
                        target_id: target_id.to_string(),
                        target_sequence,
                    });
                }
            }
        }

        append_lines(&state_dir.join("ledger.log"), &lines)?;
        loaded.journal_entries += lines.len();
        loaded.checkpoint = checkpoint_for(policy, &loaded)?;
        atomic_write(&state_dir.join("checkpoint.state"), format!("{}\n", loaded.checkpoint).as_bytes())?;
        atomic_write(&state_dir.join("balance.state"), format!("{}\n", loaded.balance_cents).as_bytes())?;
        Ok(outcome(ApplyState::Applied, binding, policy, &loaded, recovered_marker))
    }

    pub fn export_snapshot(&self, state_dir: &Path, output_path: &Path) -> Result<ExportOutcome, String> {
        let (binding, policy) = self.selected("EXPORT_SNAPSHOT", "export")?;
        let loaded = load_state(state_dir)?;
        let bytes = format!("ledger-export|{}|{}|{}|{}\n", loaded.journal_entries,
                            loaded.balance_cents, loaded.checkpoint, loaded.pending.len());
        atomic_write(output_path, bytes.as_bytes())?;
        Ok(ExportOutcome {
            selected_policy: binding.policy.clone(),
            policy_generation: policy.generation,
            journal_entries: loaded.journal_entries,
            balance_cents: loaded.balance_cents,
            checkpoint: loaded.checkpoint,
            pending_reversals: loaded.pending.len(),
        })
    }
}

fn parse_number(policy: &str, field: &str, value: &str) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| format!("policy {} has invalid {}", policy, field))
}

fn load_state(state_dir: &Path) -> Result<Loaded, String> {
    let mut loaded = Loaded::default();
    let journal = state_dir.join("ledger.log");
    if journal.is_file() {
        let text = fs::read_to_string(&journal).map_err(|error| error.to_string())?;
        for line in text.lines() {
            let fields: Vec<&str> = line.split('|').collect();
            loaded.journal_entries += 1;
            match fields.as_slice() {
                ["C", id, sequence, amount] => {
                    let sequence = sequence.parse::<u64>().map_err(|_| "invalid charge sequence".to_string())?;
                    let amount = amount.parse::<u64>().map_err(|_| "invalid charge amount".to_string())?;
                    loaded.event_ids.insert((*id).to_string());
                    loaded.charges.insert((*id).to_string(), (sequence, amount));
                    loaded.balance_cents += amount as i64;
                    loaded.max_sequence = loaded.max_sequence.max(sequence);
                }
                ["P", id, sequence, target, target_sequence] => {
                    let sequence = sequence.parse::<u64>().map_err(|_| "invalid reversal sequence".to_string())?;
                    let target_sequence = target_sequence.parse::<u64>().map_err(|_| "invalid target sequence".to_string())?;
                    loaded.event_ids.insert((*id).to_string());
                    loaded.pending.push(Pending { reversal_id: (*id).to_string(),
                                                  target_id: (*target).to_string(), target_sequence });
                    loaded.max_sequence = loaded.max_sequence.max(sequence);
                }
                ["R", id, sequence, target, target_sequence, amount] => {
                    let sequence = sequence.parse::<u64>().map_err(|_| "invalid reversal sequence".to_string())?;
                    let _ = target_sequence.parse::<u64>().map_err(|_| "invalid target sequence".to_string())?;
                    let amount = amount.parse::<u64>().map_err(|_| "invalid reversal amount".to_string())?;
                    loaded.event_ids.insert((*id).to_string());
                    loaded.reversed_targets.insert((*target).to_string());
                    loaded.balance_cents -= amount as i64;
                    loaded.max_sequence = loaded.max_sequence.max(sequence);
                }
                ["T", reversal_id, target, amount] => {
                    let amount = amount.parse::<u64>().map_err(|_| "invalid tombstone amount".to_string())?;
                    loaded.reversed_targets.insert((*target).to_string());
                    loaded.pending.retain(|item| item.reversal_id != *reversal_id);
                    loaded.balance_cents -= amount as i64;
                }
                _ => return Err("invalid append-only journal entry".to_string()),
            }
        }
    }
    let checkpoint = state_dir.join("checkpoint.state");
    if checkpoint.is_file() {
        loaded.checkpoint = fs::read_to_string(checkpoint)
            .map_err(|error| error.to_string())?
            .trim()
            .parse::<u64>()
            .map_err(|_| "invalid checkpoint".to_string())?;
    }
    Ok(loaded)
}

fn checkpoint_for(policy: &Policy, loaded: &Loaded) -> Result<u64, String> {
    match policy.checkpoint_mode.as_str() {
        "MAX_SEEN" => Ok(loaded.max_sequence),
        "PIN_PENDING" => {
            let pin = loaded.pending.iter().map(|item| item.target_sequence.saturating_sub(1)).min();
            Ok(pin.map(|value| value.min(loaded.max_sequence)).unwrap_or(loaded.max_sequence))
        }
        "READ_ONLY" => Ok(loaded.checkpoint),
        _ => Err("unknown checkpoint mode".to_string()),
    }
}

fn append_lines(path: &Path, lines: &[String]) -> Result<(), String> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)
        .map_err(|error| error.to_string())?;
    for line in lines {
        file.write_all(line.as_bytes()).map_err(|error| error.to_string())?;
        file.write_all(b"\n").map_err(|error| error.to_string())?;
    }
    file.sync_all().map_err(|error| error.to_string())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let name = path.file_name().and_then(|value| value.to_str())
        .ok_or_else(|| "state path must have a UTF-8 file name".to_string())?;
    let temporary = path.with_file_name(format!(".{}.{}.part", name, std::process::id()));
    let result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)
            .map_err(|error| error.to_string())?;
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        fs::rename(&temporary, path).map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn outcome(state: ApplyState, binding: &Binding, policy: &Policy, loaded: &Loaded,
           recovered_marker: bool) -> ReconcileOutcome {
    ReconcileOutcome {
        state,
        selected_policy: binding.policy.clone(),
        policy_generation: policy.generation,
        reversal_horizon: policy.reversal_horizon,
        journal_entries: loaded.journal_entries,
        balance_cents: loaded.balance_cents,
        checkpoint: loaded.checkpoint,
        pending_reversals: loaded.pending.len(),
        recovered_marker,
    }
}
