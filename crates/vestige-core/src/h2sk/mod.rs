//! Exact, source-carrying H²SK packet compiler. Continuous views are derived
//! diagnostics; only recorded slots, values and closed typed edges set currency.
pub mod admit;
pub mod compile;
pub mod gate;
pub mod pragmatics;
pub mod spectral;
pub use compile::{compile_log_prefix, verify_receipt};
pub use gate::{GateInput, GateVerdict, verify_draft};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    Touched,
    Supersedes,
    Corrects,
    Contradicts,
    Entails,
    Replaces,
    Contains,
    Next,
}
impl EdgeKind {
    pub fn currency(self) -> bool {
        matches!(self, Self::Supersedes | Self::Corrects | Self::Replaces)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEdge {
    pub from: String,
    pub to: String,
    pub kind: EdgeKind,
    pub seq: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEvent {
    pub id: String,
    pub seq: u64,
    pub text: String,
    pub tags: Vec<String>,
    pub source_date: Option<String>,
    pub stability_q: Option<i64>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogPrefix {
    pub offset: u64,
    pub source_root: String,
    pub events: Vec<LogEvent>,
    pub edges: Vec<LogEdge>,
    #[serde(default)]
    pub previous_receipt_hash: Option<String>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompileOptions {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub symbols: Vec<String>,
    #[serde(default)]
    pub today: Option<String>,
    #[serde(default)]
    pub bindings: BTreeMap<String, String>,
    #[serde(default)]
    pub view_ids: Option<Vec<String>>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    Executed,
    ActivePreference,
    Hypothetical,
    Rejected,
    Unspecified,
}
impl Modality {
    pub fn banner(self) -> &'static str {
        match self {
            Self::Executed => "EXECUTED FACT",
            Self::ActivePreference => "ACTIVE PREFERENCE / GEAR",
            Self::Hypothetical => "HYPOTHETICAL",
            Self::Rejected => "REJECTED OPTION",
            Self::Unspecified => "UNSPECIFIED STATEMENT",
        }
    }
    pub fn authoritative(self) -> bool {
        matches!(self, Self::Executed | Self::ActivePreference)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrencyStatus {
    UniqueApex,
    NoUniqueApex,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrencyRow {
    pub slot: String,
    pub status: CurrencyStatus,
    pub apex: Option<String>,
    pub contenders: Vec<String>,
    pub horosphere: Vec<String>,
    pub off_spine: Vec<String>,
    pub values: BTreeMap<String, Vec<String>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    pub id: String,
    pub quote: String,
    pub canonical: String,
    pub source_date: Option<String>,
    pub modality: Modality,
    pub banner: Option<String>,
    pub deltas: Vec<pragmatics::Delta>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impulse {
    pub id: String,
    pub basis: String,
    pub tau_micros_days: u64,
    pub activation_ppm: u32,
    pub previous_receipt_hash: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    pub version: u32,
    pub offset: u64,
    pub source_root: String,
    pub query_hash: String,
    pub previous_receipt_hash: Option<String>,
    pub tally: BTreeMap<String, usize>,
    pub currency: Vec<CurrencyRow>,
    pub section: Vec<Section>,
    pub frustration: Vec<String>,
    pub inclusion: Vec<LogEdge>,
    pub impulse: Vec<Impulse>,
    pub hologram: Option<spectral::SpectrumReceipt>,
    pub gate: GateVerdict,
    pub hash: String,
}
pub(crate) fn values(event: &LogEvent) -> Vec<String> {
    let bound: std::collections::BTreeSet<_> = event
        .tags
        .iter()
        .filter_map(|tag| {
            tag.strip_prefix("slot-value:")
                .and_then(|binding| binding.split_once('=').map(|(_, value)| value.to_string()))
        })
        .collect();
    if !bound.is_empty() {
        return bound.into_iter().collect();
    }
    event
        .tags
        .iter()
        .filter(|tag| {
            ["num:", "date:", "lit:", "id:"]
                .iter()
                .any(|p| tag.starts_with(p))
        })
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn slot_key(event: &LogEvent, tag: &str) -> Option<String> {
    let name = tag.strip_prefix("slot:")?;
    if name.contains(':') {
        return Some(tag.to_string());
    }
    let people: std::collections::BTreeSet<_> = event
        .tags
        .iter()
        .filter(|t| t.starts_with("person:"))
        .collect();
    if people.len() == 1 {
        return Some(format!("{tag}@{}", people.first().unwrap()));
    }
    if !people.is_empty() {
        return None;
    }
    let symbols: std::collections::BTreeSet<_> = event
        .tags
        .iter()
        .filter(|t| t.starts_with("symbol:"))
        .collect();
    (symbols.len() == 1).then(|| format!("{tag}@{}", symbols.first().unwrap()))
}
