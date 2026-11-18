//! Per-writer deterministic merge fold, typed exact handle resolver, hunk-level
//! `ingest-git`, time-travel recall, compensating transactions, typed intention
//! triggers, consolidation tiers, signed `dream_compile`, GhostLink exposure
//! backtest, and `vestige pack` segment bundles (#404, #408–#412, #417–#419, #421).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Frame emitted by an independent writer chain (#404).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriterFrame {
    pub writer_id: String,
    pub writer_seq: u64,
    pub causal_lamport: u64,
    pub memory_id: String,
    pub content: String,
    pub valid_time_ms: u64,
    pub commit_sha: Option<String>,
    pub supersedes: Option<String>,
}

/// Deterministic, order-independent merge fold across per-writer chains (#404).
/// Commutative, associative, and idempotent: any permutation/duplication of input
/// writer batches folds to the exact same ordered sequence and `canonical_head`.
pub fn merge_writer_chains(batches: &[Vec<WriterFrame>]) -> (Vec<WriterFrame>, [u8; 32]) {
    let mut dedup: BTreeMap<(u64, String, u64), WriterFrame> = BTreeMap::new();
    for batch in batches {
        for f in batch {
            dedup.insert((f.causal_lamport, f.writer_id.clone(), f.writer_seq), f.clone());
        }
    }
    let ordered: Vec<WriterFrame> = dedup.into_values().collect();
    let mut h = blake3::Hasher::new();
    h.update(b"strata-merge-fold-v1:");
    for f in &ordered {
        h.update(&f.causal_lamport.to_le_bytes());
        h.update(f.writer_id.as_bytes());
        h.update(&f.writer_seq.to_le_bytes());
        h.update(f.memory_id.as_bytes());
        h.update(f.content.as_bytes());
    }
    (ordered, *h.finalize().as_bytes())
}

/// Exact handle resolver for `symbol:`, `file:`, `test:`, `run:`, and `commit:` (#408).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactNamespaceHandle {
    Symbol(String),
    File(String),
    Test(String),
    Run(String),
    Commit(String),
}

#[derive(Debug, Default)]
pub struct ExactHandleIndex {
    entries: BTreeMap<(String, String), BTreeSet<String>>,
}

impl ExactHandleIndex {
    pub fn parse_handle(raw: &str) -> Result<ExactNamespaceHandle, &'static str> {
        let (ns, val) = raw.split_once(':').ok_or("MissingHandlePrefix")?;
        if val.is_empty() {
            return Err("EmptyHandleTarget");
        }
        match ns {
            "symbol" => Ok(ExactNamespaceHandle::Symbol(val.to_string())),
            "file" => Ok(ExactNamespaceHandle::File(val.replace('\\', "/"))),
            "test" => Ok(ExactNamespaceHandle::Test(val.to_string())),
            "run" => Ok(ExactNamespaceHandle::Run(val.to_string())),
            "commit" => {
                if val.len() < 8 || !val.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err("CommitPrefixTooShort");
                }
                Ok(ExactNamespaceHandle::Commit(val.to_ascii_lowercase()))
            }
            _ => Err("UnknownHandleNamespace"),
        }
    }

    pub fn bind(&mut self, handle: &str, memory_id: impl Into<String>) -> Result<(), &'static str> {
        let parsed = Self::parse_handle(handle)?;
        let (ns, key) = match parsed {
            ExactNamespaceHandle::Symbol(s) => ("symbol".to_string(), s),
            ExactNamespaceHandle::File(f) => ("file".to_string(), f),
            ExactNamespaceHandle::Test(t) => ("test".to_string(), t),
            ExactNamespaceHandle::Run(r) => ("run".to_string(), r),
            ExactNamespaceHandle::Commit(c) => ("commit".to_string(), c),
        };
        self.entries.entry((ns, key)).or_default().insert(memory_id.into());
        Ok(())
    }

    pub fn resolve(&self, handle: &str) -> Result<Vec<String>, &'static str> {
        let parsed = Self::parse_handle(handle)?;
        match parsed {
            ExactNamespaceHandle::Commit(prefix) => {
                let matching_commits: Vec<_> = self
                    .entries
                    .iter()
                    .filter(|((ns, sha), _)| ns == "commit" && sha.starts_with(&prefix))
                    .collect();
                if matching_commits.len() > 1 {
                    return Err("AmbiguousCommitPrefix");
                }
                Ok(matching_commits
                    .first()
                    .map(|(_, ids)| ids.iter().cloned().collect())
                    .unwrap_or_default())
            }
            other => {
                let (ns, key) = match other {
                    ExactNamespaceHandle::Symbol(s) => ("symbol".to_string(), s),
                    ExactNamespaceHandle::File(f) => ("file".to_string(), f),
                    ExactNamespaceHandle::Test(t) => ("test".to_string(), t),
                    ExactNamespaceHandle::Run(r) => ("run".to_string(), r),
                    ExactNamespaceHandle::Commit(_) => unreachable!(),
                };
                Ok(self
                    .entries
                    .get(&(ns, key))
                    .map(|s| s.iter().cloned().collect())
                    .unwrap_or_default())
            }
        }
    }
}

/// Hunk-level `Touched` edges extracted from unified git diffs (#409).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HunkTouchedEdge {
    pub commit_sha: String,
    pub file_path: String,
    pub new_start: u32,
    pub new_count: u32,
    pub context_symbol: Option<String>,
}

pub fn parse_git_diff_hunks(commit_sha: &str, diff_text: &str) -> Vec<HunkTouchedEdge> {
    let mut current_file = String::new();
    let mut out = Vec::new();
    for line in diff_text.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            current_file = rest.trim().to_string();
        } else if line.starts_with("@@ ") && !current_file.is_empty() {
            // Format: `@@ -old_s,old_c +new_s,new_c @@ optional_fn_ctx`
            if let Some(after_at) = line.strip_prefix("@@ ")
                && let Some((ranges, tail)) = after_at.split_once(" @@")
                && let Some(plus_part) = ranges.split_whitespace().find(|p| p.starts_with('+'))
            {
                let nums = &plus_part[1..];
                let (start_s, count_s) = nums.split_once(',').unwrap_or((nums, "1"));
                let new_start = start_s.parse::<u32>().unwrap_or(1);
                let new_count = count_s.parse::<u32>().unwrap_or(1);
                let ctx = tail.trim();
                out.push(HunkTouchedEdge {
                    commit_sha: commit_sha.to_string(),
                    file_path: current_file.clone(),
                    new_start,
                    new_count,
                    context_symbol: (!ctx.is_empty()).then(|| ctx.to_string()),
                });
            }
        }
    }
    out
}

/// Time-travel recall at a log `seq`, `valid_time_ms`, or `commit_sha` (#410).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeTravelCutoff {
    LogSeq(usize),
    ValidTimeMs(u64),
    CommitSha(String),
}

pub fn recall_as_of<'a>(frames: &'a [WriterFrame], cutoff: &TimeTravelCutoff) -> Vec<&'a WriterFrame> {
    let slice: &[WriterFrame] = match cutoff {
        TimeTravelCutoff::LogSeq(seq) => &frames[..(*seq).min(frames.len())],
        TimeTravelCutoff::ValidTimeMs(_) => frames,
        TimeTravelCutoff::CommitSha(sha) => {
            if let Some(pos) = frames
                .iter()
                .rposition(|f| f.commit_sha.as_deref() == Some(sha.as_str()))
            {
                &frames[..=pos]
            } else {
                &[]
            }
        }
    };

    let filtered: Vec<&WriterFrame> = slice
        .iter()
        .filter(|f| match cutoff {
            TimeTravelCutoff::ValidTimeMs(t) => f.valid_time_ms <= *t,
            _ => true,
        })
        .collect();

    let superseded: BTreeSet<&str> = filtered
        .iter()
        .filter_map(|f| f.supersedes.as_deref())
        .collect();

    filtered
        .into_iter()
        .filter(|f| !superseded.contains(f.memory_id.as_str()))
        .collect()
}

/// Named transactions with `commit` and `rollback_to(step)` via compensating records (#411).
/// Never truncates the append-only log!
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxLogEntry {
    Begin { tx_name: String },
    Write { step: u32, key: String, value: String },
    Compensate { target_step: u32, key: String, restored_value: Option<String> },
    Commit { tx_name: String },
}

#[derive(Debug, Default)]
pub struct CompensatingTxLog {
    pub entries: Vec<TxLogEntry>,
    state: BTreeMap<String, String>,
    snapshots: Vec<(u32, String, Option<String>)>,
}

impl CompensatingTxLog {
    pub fn begin(&mut self, tx_name: impl Into<String>) {
        self.entries.push(TxLogEntry::Begin { tx_name: tx_name.into() });
        self.snapshots.clear();
    }

    pub fn put(&mut self, step: u32, key: impl Into<String>, value: impl Into<String>) {
        let k = key.into();
        let v = value.into();
        let prev = self.state.insert(k.clone(), v.clone());
        self.snapshots.push((step, k.clone(), prev));
        self.entries.push(TxLogEntry::Write { step, key: k, value: v });
    }

    /// Roll back to `step` by appending compensating records in reverse order (#411).
    pub fn rollback_to(&mut self, step: u32) -> usize {
        let mut compensated = 0usize;
        while let Some((s, _, _)) = self.snapshots.last() {
            if *s <= step {
                break;
            }
            let (s, k, prev) = self.snapshots.pop().unwrap();
            match &prev {
                Some(old) => {
                    self.state.insert(k.clone(), old.clone());
                }
                None => {
                    self.state.remove(&k);
                }
            }
            self.entries.push(TxLogEntry::Compensate {
                target_step: s,
                key: k,
                restored_value: prev,
            });
            compensated += 1;
        }
        compensated
    }

    pub fn commit(&mut self, tx_name: impl Into<String>) {
        self.snapshots.clear();
        self.entries.push(TxLogEntry::Commit { tx_name: tx_name.into() });
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.state.get(key).map(String::as_str)
    }
}

/// Typed intention triggers (#412).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentionTrigger {
    FileChanged(String),
    TestFailed(String),
    CommitTouchedPathPrefix(String),
    CiStatusFailed { workflow: String },
    AtOrAfterMs(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemSignal {
    FileEdited(String),
    TestFailure(String),
    CommitLanded { touched_paths: Vec<String> },
    CiCompleted { workflow: String, passed: bool },
    ClockTickMs(u64),
}

pub fn trigger_matches(trigger: &IntentionTrigger, signal: &SystemSignal) -> bool {
    match (trigger, signal) {
        (IntentionTrigger::FileChanged(p), SystemSignal::FileEdited(f)) => {
            p.replace('\\', "/") == f.replace('\\', "/")
        }
        (IntentionTrigger::TestFailed(t), SystemSignal::TestFailure(f)) => t == f,
        (
            IntentionTrigger::CommitTouchedPathPrefix(pref),
            SystemSignal::CommitLanded { touched_paths },
        ) => touched_paths
            .iter()
            .any(|tp| tp.replace('\\', "/").starts_with(pref)),
        (
            IntentionTrigger::CiStatusFailed { workflow },
            SystemSignal::CiCompleted { workflow: wf, passed },
        ) => workflow == wf && !*passed,
        (IntentionTrigger::AtOrAfterMs(deadline), SystemSignal::ClockTickMs(now)) => now >= deadline,
        _ => false,
    }
}

/// Staged consolidation tiers by stability + archive compaction receipt (#417).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationTier {
    Episodic,
    Working,
    Consolidated,
    Archive,
}

pub fn classify_tier(stability_days: u32) -> ConsolidationTier {
    match stability_days {
        0..=2 => ConsolidationTier::Episodic,
        3..=20 => ConsolidationTier::Working,
        21..=179 => ConsolidationTier::Consolidated,
        _ => ConsolidationTier::Archive,
    }
}

/// Signed, diffable `dream_compile` (#418).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DreamCompileDiff {
    pub source_episode_ids: Vec<String>,
    pub distilled_memory_id: String,
    pub unified_diff: String,
    pub signature_hex: String,
}

pub fn dream_compile_signed(
    signing_key: &[u8; 32],
    episodes: &[(String, String)],
    distilled_id: impl Into<String>,
    distilled_summary: &str,
) -> DreamCompileDiff {
    let distilled_memory_id = distilled_id.into();
    let mut diff_lines = Vec::new();
    let mut ids = Vec::new();
    for (id, body) in episodes {
        ids.push(id.clone());
        diff_lines.push(format!("- [{id}] {body}"));
    }
    diff_lines.push(format!("+ [{distilled_memory_id}] {distilled_summary}"));
    let unified_diff = diff_lines.join("\n");
    let signature_hex = blake3::keyed_hash(signing_key, unified_diff.as_bytes())
        .to_hex()
        .to_string();
    DreamCompileDiff {
        source_episode_ids: ids,
        distilled_memory_id,
        unified_diff,
        signature_hex,
    }
}

/// GhostLink exposure ledger & decoy-controlled edge minting (#419).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GhostLinkExposure {
    pub candidate_edge: (String, String),
    pub surfaced_pass: bool,
    pub decoy_pass: bool,
}

pub fn evaluate_ghostlink_exposures(
    exposures: &[GhostLinkExposure],
    min_trials: usize,
    min_lift_milli: i64,
) -> Vec<(String, String)> {
    let mut stats: BTreeMap<(String, String), (usize, usize, usize)> = BTreeMap::new();
    for ex in exposures {
        let entry = stats.entry(ex.candidate_edge.clone()).or_default();
        entry.0 += 1;
        if ex.surfaced_pass {
            entry.1 += 1;
        }
        if ex.decoy_pass {
            entry.2 += 1;
        }
    }
    let mut minted = Vec::new();
    for (edge, (trials, surf_wins, decoy_wins)) in stats {
        if trials >= min_trials {
            let surf_rate = (surf_wins as i64 * 1000) / (trials as i64);
            let decoy_rate = (decoy_wins as i64 * 1000) / (trials as i64);
            if surf_rate - decoy_rate >= min_lift_milli {
                minted.push(edge);
            }
        }
    }
    minted
}

/// Signed segment packs (`vestige pack`, `install`, `verify`) (#421).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalPackBundle {
    pub pack_name: String,
    pub version: String,
    pub frames: Vec<WriterFrame>,
    pub bundle_blake3: String,
    pub signature_hex: String,
}

impl CausalPackBundle {
    pub fn pack(
        signing_key: &[u8; 32],
        pack_name: impl Into<String>,
        version: impl Into<String>,
        frames: Vec<WriterFrame>,
    ) -> Self {
        let pack_name = pack_name.into();
        let version = version.into();
        let serialized = serde_json::to_vec(&frames).expect("serialize pack frames");
        let bundle_blake3 = blake3::hash(&serialized).to_hex().to_string();
        let msg = format!("{pack_name}|{version}|{bundle_blake3}");
        let signature_hex = blake3::keyed_hash(signing_key, msg.as_bytes())
            .to_hex()
            .to_string();
        Self {
            pack_name,
            version,
            frames,
            bundle_blake3,
            signature_hex,
        }
    }

    pub fn verify(&self, signing_key: &[u8; 32]) -> bool {
        let serialized = match serde_json::to_vec(&self.frames) {
            Ok(b) => b,
            Err(_) => return false,
        };
        if blake3::hash(&serialized).to_hex().as_str() != self.bundle_blake3 {
            return false;
        }
        let msg = format!("{}|{}|{}", self.pack_name, self.version, self.bundle_blake3);
        blake3::keyed_hash(signing_key, msg.as_bytes())
            .to_hex()
            .as_str()
            == self.signature_hex
    }
}
