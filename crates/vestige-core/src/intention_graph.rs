//! Pure, deterministic intention evidence lifecycle evaluation.
//!
//! This module deliberately has no storage, network, model, or side-effect
//! dependencies. Callers provide evidence snapshots and persist the serialized
//! graph plus their own immutable command journal. Evidence is caller-asserted:
//! an evaluation label records what this state machine concluded from those
//! assertions, but is not cryptographic proof, truth verification, or authority
//! to perform an external effect.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const INTENTION_GRAPH_SCHEMA_VERSION: u32 = 1;
pub const INTENTION_GRAPH_ALGORITHM_VERSION: &str = "intention-evidence-v1";

const MAX_PLANS: usize = 1_024;
const MAX_REQUIREMENTS_PER_PLAN: usize = 128;
const MAX_VERSIONS_PER_PLAN: usize = 1_024;
const MAX_SOURCES: usize = 4_096;
const MAX_EVENT_RECEIPTS: usize = 16_384;
const MAX_EVALUATIONS: usize = 16_384;
const MAX_ATTENTION_EVENTS: usize = 16_384;
const MAX_EVENTS_PER_OBSERVATION: usize = 256;
const MAX_CONFLICT_KEYS: usize = 64;
const MAX_ID_BYTES: usize = 160;
const MAX_SOURCE_KEY_BYTES: usize = 512;
const MAX_DESCRIPTION_BYTES: usize = 8_192;
const MAX_TEXT_BYTES: usize = 4_096;
const MAX_ATTENTION_INTERVAL_SECONDS: u64 = 365 * 24 * 60 * 60;
const MAX_RECEIPT_RESPONSE_BYTES: usize = 256 * 1_024;

fn default_attention_interval_seconds() -> u64 {
    60 * 60
}

fn default_schema_version() -> u32 {
    INTENTION_GRAPH_SCHEMA_VERSION
}

fn default_algorithm_version() -> String {
    INTENTION_GRAPH_ALGORITHM_VERSION.to_owned()
}

/// A bounded scalar suitable for exact deterministic evidence comparisons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EvidenceValue {
    Bool(bool),
    Integer(i64),
    Text(String),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Low,
    #[default]
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventExpectation {
    Occurred,
    Absent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum EvidenceCondition {
    Exists,
    Equals { value: EvidenceValue },
    NotEquals { value: EvidenceValue },
}

/// A requirement is exact and declarative. No semantic inference is performed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Requirement {
    Evidence {
        id: String,
        source_key: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        memory_id: Option<String>,
        condition: EvidenceCondition,
    },
    Event {
        id: String,
        source_key: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        memory_id: Option<String>,
        event: String,
        expectation: EventExpectation,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_start: Option<DateTime<Utc>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_end: Option<DateTime<Utc>>,
        #[serde(default)]
        grace_seconds: u64,
    },
    PlanFulfilled {
        id: String,
        plan_id: String,
    },
}

impl Requirement {
    pub fn id(&self) -> &str {
        match self {
            Self::Evidence { id, .. } | Self::Event { id, .. } | Self::PlanFulfilled { id, .. } => {
                id
            }
        }
    }

    pub fn source_key(&self) -> Option<&str> {
        match self {
            Self::Evidence { source_key, .. } | Self::Event { source_key, .. } => Some(source_key),
            Self::PlanFulfilled { .. } => None,
        }
    }

    pub fn prerequisite_plan_id(&self) -> Option<&str> {
        match self {
            Self::PlanFulfilled { plan_id, .. } => Some(plan_id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompletionBasis {
    UserReport { report: String },
    Evidence { requirement_ids: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedCommand {
    pub command: Command,
    pub now: DateTime<Utc>,
}

/// Public command contract. It is internally tagged so MCP callers send a
/// single object with an `action` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Command {
    Plan {
        id: String,
        description: String,
        #[serde(default)]
        priority: Priority,
        #[serde(default)]
        requirements: Vec<Requirement>,
        #[serde(default = "default_attention_interval_seconds")]
        min_attention_interval_seconds: u64,
        #[serde(default)]
        conflict_keys: Vec<String>,
    },
    Revise {
        id: String,
        expected_version: u32,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        priority: Option<Priority>,
        #[serde(default)]
        requirements: Option<Vec<Requirement>>,
        #[serde(default)]
        min_attention_interval_seconds: Option<u64>,
        #[serde(default)]
        conflict_keys: Option<Vec<String>>,
    },
    Observe {
        event_id: String,
        source_key: String,
        source_revision: u64,
        #[serde(default)]
        value: Option<EvidenceValue>,
        #[serde(default)]
        observed_events: Vec<String>,
        #[serde(default)]
        covered_events: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        coverage_start: Option<DateTime<Utc>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        coverage_end: Option<DateTime<Utc>>,
    },
    Evaluate {
        #[serde(default)]
        id: Option<String>,
    },
    Explain {
        id: String,
    },
    Portfolio,
    Complete {
        id: String,
        expected_version: u32,
        basis: CompletionBasis,
    },
    Cancel {
        id: String,
        expected_version: u32,
        reason: String,
    },
    Acknowledge {
        queue_id: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanLifecycle {
    Active,
    Fulfilled,
    Disputed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Readiness {
    Ready,
    Unknown,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequirementState {
    Satisfied,
    Unmet,
    Missing,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanVersion {
    pub version: u32,
    pub revised_at: DateTime<Utc>,
    pub description: String,
    pub priority: Priority,
    pub requirements: Vec<Requirement>,
    pub min_attention_interval_seconds: u64,
    pub conflict_keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRecord {
    pub basis: CompletionBasis,
    pub plan_version: u32,
    pub completed_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disputed_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispute_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancellationRecord {
    pub reason: String,
    pub plan_version: u32,
    pub cancelled_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub original_description: String,
    pub versions: Vec<PlanVersion>,
    pub lifecycle: PlanLifecycle,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion: Option<CompletionRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancellation: Option<CancellationRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_attention_acknowledged_at: Option<DateTime<Utc>>,
}

impl Plan {
    pub fn current(&self) -> &PlanVersion {
        self.versions
            .last()
            .expect("a validated plan always has one version")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSource {
    pub source_key: String,
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<EvidenceValue>,
    pub observed_events: BTreeSet<String>,
    pub covered_events: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage_start: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage_end: Option<DateTime<Utc>>,
    pub observed_at: DateTime<Utc>,
    /// Fixed provenance label; callers cannot mark supplied evidence verified.
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceProvenance {
    CallerAsserted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationReceipt {
    pub event_id: String,
    pub command_digest: String,
    pub source_key: String,
    pub source_revision: u64,
    pub response: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequirementEvaluation {
    pub requirement_id: String,
    pub state: RequirementState,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_revision: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSnapshot {
    pub plan_id: String,
    pub plan_version: u32,
    pub lifecycle: PlanLifecycle,
    pub readiness: Readiness,
    pub requirements: Vec<RequirementEvaluation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRecord {
    pub label: String,
    pub evaluated_at: DateTime<Utc>,
    pub snapshot: EvaluationSnapshot,
    pub record_kind: EvaluationRecordKind,
    pub cryptographic_proof: bool,
    pub grants_authority: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationRecordKind {
    DeterministicEvaluation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionEvent {
    pub queue_id: String,
    pub fingerprint: String,
    pub plan_id: String,
    pub priority: Priority,
    pub created_at: DateTime<Utc>,
    pub deliver_after: DateTime<Utc>,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acknowledged_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_at: Option<DateTime<Utc>>,
}

/// Serializable state. BTreeMap/BTreeSet are used throughout so snapshots and
/// responses are stable across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentionGraph {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default = "default_algorithm_version")]
    pub algorithm_version: String,
    pub plans: BTreeMap<String, Plan>,
    pub sources: BTreeMap<String, EvidenceSource>,
    pub observation_receipts: BTreeMap<String, ObservationReceipt>,
    pub evaluations: BTreeMap<String, EvaluationRecord>,
    pub attention_queue: BTreeMap<String, AttentionEvent>,
    #[serde(default)]
    pub attention_sequence_by_plan: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_mutation_at: Option<DateTime<Utc>>,
}

impl Default for IntentionGraph {
    fn default() -> Self {
        Self {
            schema_version: INTENTION_GRAPH_SCHEMA_VERSION,
            algorithm_version: INTENTION_GRAPH_ALGORITHM_VERSION.to_owned(),
            plans: BTreeMap::new(),
            sources: BTreeMap::new(),
            observation_receipts: BTreeMap::new(),
            evaluations: BTreeMap::new(),
            attention_queue: BTreeMap::new(),
            attention_sequence_by_plan: BTreeMap::new(),
            last_mutation_at: None,
        }
    }
}

impl IntentionGraph {
    /// Apply one command atomically. Read-only commands borrow the current
    /// graph; mutating commands commit only after every validation succeeds.
    pub fn apply(&mut self, command: Command, now: DateTime<Utc>) -> Result<Value, String> {
        self.validate_state()?;
        match &command {
            Command::Explain { id } => {
                self.reject_historical_read(now)?;
                self.explain(id, now)
            }
            Command::Portfolio => {
                self.reject_historical_read(now)?;
                self.portfolio(now)
            }
            _ => {
                let is_noop_reapplication = match &command {
                    Command::Observe { event_id, .. } => {
                        self.observation_receipts.contains_key(event_id)
                    }
                    Command::Acknowledge { queue_id } => self
                        .attention_queue
                        .get(queue_id)
                        .is_some_and(|event| event.acknowledged_at.is_some()),
                    Command::Complete {
                        id,
                        expected_version,
                        basis,
                    } => self.plans.get(id).is_some_and(|plan| {
                        plan.lifecycle == PlanLifecycle::Fulfilled
                            && plan.current().version == *expected_version
                            && plan.completion.as_ref().is_some_and(|completion| {
                                completion.plan_version == *expected_version
                                    && completion.basis == *basis
                            })
                    }),
                    Command::Cancel {
                        id,
                        expected_version,
                        reason,
                    } => self.plans.get(id).is_some_and(|plan| {
                        plan.lifecycle == PlanLifecycle::Cancelled
                            && plan.current().version == *expected_version
                            && plan.cancellation.as_ref().is_some_and(|cancellation| {
                                cancellation.plan_version == *expected_version
                                    && cancellation.reason == *reason
                            })
                    }),
                    _ => false,
                };
                if !is_noop_reapplication
                    && let Some(last_mutation_at) = self.last_mutation_at
                    && now < last_mutation_at
                {
                    return Err(format!(
                        "out-of-order mutation timestamp {now}; last mutation was {last_mutation_at}"
                    ));
                }
                let mut next = self.clone();
                let response = next.apply_mutating(command, now)?;
                if !is_noop_reapplication {
                    next.last_mutation_at = Some(now);
                }
                // A successful command must never persist a state that the
                // next command would reject. Validate output-derived limits
                // as well as the caller's input before committing the clone.
                next.validate_state()?;
                *self = next;
                Ok(response)
            }
        }
    }

    fn validate_engine_version(&self) -> Result<(), String> {
        if self.schema_version != INTENTION_GRAPH_SCHEMA_VERSION {
            return Err(format!(
                "unsupported intention graph schema version {}; expected {}",
                self.schema_version, INTENTION_GRAPH_SCHEMA_VERSION
            ));
        }
        if self.algorithm_version != INTENTION_GRAPH_ALGORITHM_VERSION {
            return Err(format!(
                "unsupported intention graph algorithm version '{}'; expected '{}'",
                self.algorithm_version, INTENTION_GRAPH_ALGORITHM_VERSION
            ));
        }
        Ok(())
    }

    fn reject_historical_read(&self, now: DateTime<Utc>) -> Result<(), String> {
        if self
            .last_mutation_at
            .is_some_and(|last_mutation_at| now < last_mutation_at)
        {
            return Err(format!(
                "historical evaluation is unsupported: requested {now}, current state is from {}",
                self.last_mutation_at.expect("checked as some")
            ));
        }
        Ok(())
    }

    /// Validate a deserialized graph before any evaluator path can rely on its
    /// internal invariants.
    pub fn validate_state(&self) -> Result<(), String> {
        self.validate_engine_version()?;
        if self.plans.len() > MAX_PLANS {
            return Err(format!(
                "serialized graph exceeds plan limit of {MAX_PLANS}"
            ));
        }
        if self.sources.len() > MAX_SOURCES {
            return Err(format!(
                "serialized graph exceeds source limit of {MAX_SOURCES}"
            ));
        }
        if self.observation_receipts.len() > MAX_EVENT_RECEIPTS {
            return Err(format!(
                "serialized graph exceeds observation receipt limit of {MAX_EVENT_RECEIPTS}"
            ));
        }
        if self.evaluations.len() > MAX_EVALUATIONS {
            return Err(format!(
                "serialized graph exceeds evaluation limit of {MAX_EVALUATIONS}"
            ));
        }
        if self.attention_queue.len() > MAX_ATTENTION_EVENTS {
            return Err(format!(
                "serialized graph exceeds attention limit of {MAX_ATTENTION_EVENTS}"
            ));
        }

        let mut latest_stored_at: Option<DateTime<Utc>> = None;
        for (plan_id, plan) in &self.plans {
            validate_id("serialized plan map key", plan_id)?;
            if plan.id != *plan_id {
                return Err(format!(
                    "serialized plan map key '{plan_id}' does not match embedded id '{}'",
                    plan.id
                ));
            }
            validate_description(&plan.original_description)?;
            if plan.versions.is_empty() {
                return Err(format!("serialized plan '{plan_id}' has no versions"));
            }
            if plan.versions.len() > MAX_VERSIONS_PER_PLAN {
                return Err(format!(
                    "serialized plan '{plan_id}' exceeds version limit of {MAX_VERSIONS_PER_PLAN}"
                ));
            }
            let mut prior_revised_at = None;
            for (index, version) in plan.versions.iter().enumerate() {
                let expected_version = u32::try_from(index + 1)
                    .map_err(|_| format!("serialized plan '{plan_id}' version overflow"))?;
                if version.version != expected_version {
                    return Err(format!(
                        "serialized plan '{plan_id}' has non-contiguous version {}; expected {expected_version}",
                        version.version
                    ));
                }
                if prior_revised_at.is_some_and(|prior| version.revised_at < prior) {
                    return Err(format!(
                        "serialized plan '{plan_id}' has out-of-order revision timestamps"
                    ));
                }
                validate_description(&version.description)?;
                validate_plan_fields(
                    plan_id,
                    &version.requirements,
                    version.min_attention_interval_seconds,
                    &version.conflict_keys,
                )?;
                prior_revised_at = Some(version.revised_at);
                update_latest(&mut latest_stored_at, version.revised_at);
            }
            match plan.lifecycle {
                PlanLifecycle::Fulfilled | PlanLifecycle::Disputed if plan.completion.is_none() => {
                    return Err(format!(
                        "serialized {:?} plan '{plan_id}' has no completion record",
                        plan.lifecycle
                    ));
                }
                PlanLifecycle::Cancelled if plan.cancellation.is_none() => {
                    return Err(format!(
                        "serialized cancelled plan '{plan_id}' has no cancellation record"
                    ));
                }
                _ => {}
            }
            if let Some(completion) = &plan.completion {
                validate_completion_basis(&completion.basis)?;
                if completion.plan_version == 0 || completion.plan_version > plan.current().version
                {
                    return Err(format!(
                        "serialized plan '{plan_id}' has invalid completion plan_version {}",
                        completion.plan_version
                    ));
                }
                if completion.disputed_at.is_some() != completion.dispute_reason.is_some() {
                    return Err(format!(
                        "serialized plan '{plan_id}' has incomplete dispute metadata"
                    ));
                }
                update_latest(&mut latest_stored_at, completion.completed_at);
                if let Some(disputed_at) = completion.disputed_at {
                    if disputed_at < completion.completed_at {
                        return Err(format!(
                            "serialized plan '{plan_id}' dispute predates completion"
                        ));
                    }
                    update_latest(&mut latest_stored_at, disputed_at);
                }
            }
            if let Some(cancellation) = &plan.cancellation {
                validate_bounded_nonempty(
                    "serialized cancellation reason",
                    &cancellation.reason,
                    MAX_TEXT_BYTES,
                )?;
                if cancellation.plan_version == 0
                    || cancellation.plan_version > plan.current().version
                {
                    return Err(format!(
                        "serialized plan '{plan_id}' has invalid cancellation plan_version {}",
                        cancellation.plan_version
                    ));
                }
                update_latest(&mut latest_stored_at, cancellation.cancelled_at);
            }
            if let Some(acknowledged_at) = plan.last_attention_acknowledged_at {
                update_latest(&mut latest_stored_at, acknowledged_at);
            }
        }
        for (plan_id, plan) in &self.plans {
            self.validate_dependency_graph(plan_id, &plan.current().requirements)?;
        }

        for (source_key, source) in &self.sources {
            validate_source_key(source_key)?;
            if source.source_key != *source_key {
                return Err(format!(
                    "serialized source map key '{source_key}' does not match embedded key '{}'",
                    source.source_key
                ));
            }
            if source.revision == 0 {
                return Err(format!(
                    "serialized source '{source_key}' has revision zero"
                ));
            }
            validate_evidence_value(source.value.as_ref())?;
            validate_event_list(
                "serialized observed_events",
                &source.observed_events.iter().cloned().collect::<Vec<_>>(),
            )?;
            validate_event_list(
                "serialized covered_events",
                &source.covered_events.iter().cloned().collect::<Vec<_>>(),
            )?;
            validate_window(
                source.coverage_start,
                source.coverage_end,
                "serialized observation coverage",
            )?;
            if source
                .coverage_end
                .is_some_and(|coverage_end| coverage_end > source.observed_at)
            {
                return Err(format!(
                    "serialized source '{source_key}' promises coverage after it was observed"
                ));
            }
            update_latest(&mut latest_stored_at, source.observed_at);
        }

        for (event_id, receipt) in &self.observation_receipts {
            validate_id("serialized observation receipt key", event_id)?;
            if receipt.event_id != *event_id {
                return Err(format!(
                    "serialized observation receipt key '{event_id}' does not match embedded id '{}'",
                    receipt.event_id
                ));
            }
            validate_source_key(&receipt.source_key)?;
            let source = self.sources.get(&receipt.source_key).ok_or_else(|| {
                format!(
                    "serialized observation receipt '{event_id}' references missing source '{}'",
                    receipt.source_key
                )
            })?;
            if receipt.source_revision == 0 || receipt.source_revision > source.revision {
                return Err(format!(
                    "serialized observation receipt '{event_id}' has invalid source revision {}",
                    receipt.source_revision
                ));
            }
            if receipt.command_digest.len() != 64
                || !receipt
                    .command_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(format!(
                    "serialized observation receipt '{event_id}' has invalid command digest"
                ));
            }
            let response_size = serde_json::to_vec(&receipt.response)
                .map_err(|error| format!("cannot serialize receipt '{event_id}': {error}"))?
                .len();
            if response_size > MAX_RECEIPT_RESPONSE_BYTES {
                return Err(format!(
                    "serialized observation receipt '{event_id}' response exceeds {MAX_RECEIPT_RESPONSE_BYTES} bytes"
                ));
            }
        }

        for (label, evaluation) in &self.evaluations {
            validate_id("serialized evaluation label", label)?;
            if evaluation.label != *label {
                return Err(format!(
                    "serialized evaluation key '{label}' does not match embedded label '{}'",
                    evaluation.label
                ));
            }
            let plan = self
                .plans
                .get(&evaluation.snapshot.plan_id)
                .ok_or_else(|| {
                    format!(
                        "serialized evaluation '{label}' references missing plan '{}'",
                        evaluation.snapshot.plan_id
                    )
                })?;
            if evaluation.snapshot.plan_version == 0
                || evaluation.snapshot.plan_version > plan.current().version
            {
                return Err(format!(
                    "serialized evaluation '{label}' has invalid plan version {}",
                    evaluation.snapshot.plan_version
                ));
            }
            if evaluation.snapshot.requirements.len() > MAX_REQUIREMENTS_PER_PLAN {
                return Err(format!(
                    "serialized evaluation '{label}' has too many requirement results"
                ));
            }
            if evaluation.cryptographic_proof || evaluation.grants_authority {
                return Err(format!(
                    "serialized evaluation '{label}' illegally claims proof or authority"
                ));
            }
            update_latest(&mut latest_stored_at, evaluation.evaluated_at);
        }

        for (queue_id, event) in &self.attention_queue {
            validate_id("serialized attention key", queue_id)?;
            if event.queue_id != *queue_id {
                return Err(format!(
                    "serialized attention key '{queue_id}' does not match embedded id '{}'",
                    event.queue_id
                ));
            }
            if !self.plans.contains_key(&event.plan_id) {
                return Err(format!(
                    "serialized attention '{queue_id}' references missing plan '{}'",
                    event.plan_id
                ));
            }
            if event.deliver_after < event.created_at {
                return Err(format!(
                    "serialized attention '{queue_id}' is deliverable before creation"
                ));
            }
            if event
                .acknowledged_at
                .is_some_and(|acknowledged_at| acknowledged_at < event.created_at)
                || event
                    .superseded_at
                    .is_some_and(|superseded_at| superseded_at < event.created_at)
            {
                return Err(format!(
                    "serialized attention '{queue_id}' has an invalid lifecycle timestamp"
                ));
            }
            update_latest(&mut latest_stored_at, event.created_at);
            if let Some(acknowledged_at) = event.acknowledged_at {
                update_latest(&mut latest_stored_at, acknowledged_at);
            }
            if let Some(superseded_at) = event.superseded_at {
                update_latest(&mut latest_stored_at, superseded_at);
            }
        }
        for plan_id in self.attention_sequence_by_plan.keys() {
            if !self.plans.contains_key(plan_id) {
                return Err(format!(
                    "serialized attention sequence references missing plan '{plan_id}'"
                ));
            }
        }
        if self.attention_sequence_by_plan.len() > self.plans.len() {
            return Err("serialized graph has too many attention sequence entries".to_owned());
        }
        match (latest_stored_at, self.last_mutation_at) {
            (Some(_), None) => {
                return Err("serialized non-empty graph has no last_mutation_at".to_owned());
            }
            (Some(latest), Some(last)) if latest > last => {
                return Err(format!(
                    "serialized state timestamp {latest} is later than last_mutation_at {last}"
                ));
            }
            _ => {}
        }
        Ok(())
    }

    fn apply_mutating(&mut self, command: Command, now: DateTime<Utc>) -> Result<Value, String> {
        match command {
            Command::Plan {
                id,
                description,
                priority,
                requirements,
                min_attention_interval_seconds,
                conflict_keys,
            } => self.plan(
                id,
                description,
                priority,
                requirements,
                min_attention_interval_seconds,
                conflict_keys,
                now,
            ),
            Command::Revise {
                id,
                expected_version,
                description,
                priority,
                requirements,
                min_attention_interval_seconds,
                conflict_keys,
            } => self.revise(
                id,
                expected_version,
                description,
                priority,
                requirements,
                min_attention_interval_seconds,
                conflict_keys,
                now,
            ),
            Command::Observe {
                event_id,
                source_key,
                source_revision,
                value,
                observed_events,
                covered_events,
                coverage_start,
                coverage_end,
            } => self.observe(
                event_id,
                source_key,
                source_revision,
                value,
                observed_events,
                covered_events,
                coverage_start,
                coverage_end,
                now,
            ),
            Command::Evaluate { id } => self.evaluate(id.as_deref(), now),
            Command::Complete {
                id,
                expected_version,
                basis,
            } => self.complete(id, expected_version, basis, now),
            Command::Cancel {
                id,
                expected_version,
                reason,
            } => self.cancel(id, expected_version, reason, now),
            Command::Acknowledge { queue_id } => self.acknowledge(queue_id, now),
            Command::Explain { .. } | Command::Portfolio => {
                Err("read-only command reached mutating dispatcher".to_owned())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn plan(
        &mut self,
        id: String,
        description: String,
        priority: Priority,
        requirements: Vec<Requirement>,
        min_attention_interval_seconds: u64,
        conflict_keys: Vec<String>,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        validate_id("plan id", &id)?;
        validate_description(&description)?;
        if self.plans.contains_key(&id) {
            return Err(format!("plan '{id}' already exists"));
        }
        if self.plans.len() >= MAX_PLANS {
            return Err(format!("plan limit of {MAX_PLANS} reached"));
        }
        validate_plan_fields(
            &id,
            &requirements,
            min_attention_interval_seconds,
            &conflict_keys,
        )?;
        self.validate_dependency_graph(&id, &requirements)?;

        let version = PlanVersion {
            version: 1,
            revised_at: now,
            description: description.clone(),
            priority,
            requirements,
            min_attention_interval_seconds,
            conflict_keys: normalized_unique(conflict_keys),
        };
        self.plans.insert(
            id.clone(),
            Plan {
                id: id.clone(),
                original_description: description,
                versions: vec![version],
                lifecycle: PlanLifecycle::Active,
                completion: None,
                cancellation: None,
                last_attention_acknowledged_at: None,
            },
        );

        let evaluations = self.evaluate_plan_set([id.clone()].into_iter().collect(), now)?;
        Ok(json!({
            "plan": self.plans.get(&id),
            "evaluations": evaluations,
            "attention": self.queue_partition(now),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn revise(
        &mut self,
        id: String,
        expected_version: u32,
        description: Option<String>,
        priority: Option<Priority>,
        requirements: Option<Vec<Requirement>>,
        min_attention_interval_seconds: Option<u64>,
        conflict_keys: Option<Vec<String>>,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        validate_id("plan id", &id)?;
        let plan = self
            .plans
            .get(&id)
            .ok_or_else(|| format!("unknown plan '{id}'"))?;
        if plan.lifecycle == PlanLifecycle::Cancelled {
            return Err(format!("cancelled plan '{id}' cannot be revised"));
        }
        let current = plan.current().clone();
        if current.version != expected_version {
            return Err(format!(
                "stale plan version for '{id}': expected {expected_version}, current is {}",
                current.version
            ));
        }

        let next_description = description.unwrap_or(current.description);
        validate_description(&next_description)?;
        let next_requirements = requirements.unwrap_or(current.requirements);
        let next_interval =
            min_attention_interval_seconds.unwrap_or(current.min_attention_interval_seconds);
        let next_conflicts = conflict_keys.unwrap_or(current.conflict_keys);
        validate_plan_fields(&id, &next_requirements, next_interval, &next_conflicts)?;
        self.validate_dependency_graph(&id, &next_requirements)?;

        let next_version = current
            .version
            .checked_add(1)
            .ok_or_else(|| format!("version overflow for plan '{id}'"))?;
        if plan.versions.len() >= MAX_VERSIONS_PER_PLAN {
            return Err(format!(
                "plan '{id}' reached version limit of {MAX_VERSIONS_PER_PLAN}; start a new scoped plan and retain this graph for audit"
            ));
        }
        let plan = self.plans.get_mut(&id).expect("plan existence checked");
        plan.versions.push(PlanVersion {
            version: next_version,
            revised_at: now,
            description: next_description,
            priority: priority.unwrap_or(current.priority),
            requirements: next_requirements,
            min_attention_interval_seconds: next_interval,
            conflict_keys: normalized_unique(next_conflicts),
        });
        if matches!(
            plan.lifecycle,
            PlanLifecycle::Fulfilled | PlanLifecycle::Disputed
        ) {
            plan.lifecycle = PlanLifecycle::Active;
            if let Some(completion) = &mut plan.completion {
                completion.disputed_at = Some(now);
                completion.dispute_reason = Some(format!(
                    "plan revised from version {} to version {next_version}; prior completion requires reconciliation",
                    completion.plan_version
                ));
            }
        }

        let affected = self.dependent_closure([id.clone()].into_iter().collect());
        let evaluations = self.evaluate_plan_set(affected, now)?;
        Ok(json!({
            "plan": self.plans.get(&id),
            "evaluations": evaluations,
            "attention": self.queue_partition(now),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        event_id: String,
        source_key: String,
        source_revision: u64,
        value: Option<EvidenceValue>,
        observed_events: Vec<String>,
        covered_events: Vec<String>,
        coverage_start: Option<DateTime<Utc>>,
        coverage_end: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        validate_id("event id", &event_id)?;
        validate_source_key(&source_key)?;
        validate_evidence_value(value.as_ref())?;
        validate_event_list("observed_events", &observed_events)?;
        validate_event_list("covered_events", &covered_events)?;
        validate_window(coverage_start, coverage_end, "observation coverage")?;
        if source_revision == 0 {
            return Err("source_revision must be greater than zero".to_owned());
        }

        let digest_material = Command::Observe {
            event_id: event_id.clone(),
            source_key: source_key.clone(),
            source_revision,
            value: value.clone(),
            observed_events: observed_events.clone(),
            covered_events: covered_events.clone(),
            coverage_start,
            coverage_end,
        };
        let command_digest = stable_digest(&digest_material)?;
        if let Some(prior) = self.observation_receipts.get(&event_id) {
            if prior.command_digest != command_digest {
                return Err(format!(
                    "event id '{event_id}' was already used for a different observation"
                ));
            }
            let mut response = prior.response.clone();
            if let Some(object) = response.as_object_mut() {
                object.insert("idempotent".to_owned(), Value::Bool(true));
            }
            return Ok(response);
        }
        if coverage_end.is_some_and(|coverage_end| coverage_end > now) {
            return Err("coverage_end cannot be later than the observation time".to_owned());
        }

        if self.observation_receipts.len() >= MAX_EVENT_RECEIPTS {
            return Err(format!(
                "observation receipt limit of {MAX_EVENT_RECEIPTS} reached"
            ));
        }
        if !self.sources.contains_key(&source_key) && self.sources.len() >= MAX_SOURCES {
            return Err(format!("evidence source limit of {MAX_SOURCES} reached"));
        }
        if let Some(current) = self.sources.get(&source_key)
            && source_revision <= current.revision
        {
            return Err(format!(
                "stale source revision for '{source_key}': got {source_revision}, current is {}",
                current.revision
            ));
        }

        self.sources.insert(
            source_key.clone(),
            EvidenceSource {
                source_key: source_key.clone(),
                revision: source_revision,
                value,
                observed_events: observed_events.into_iter().collect(),
                covered_events: covered_events.into_iter().collect(),
                coverage_start,
                coverage_end,
                observed_at: now,
                provenance: EvidenceProvenance::CallerAsserted,
            },
        );

        let direct: BTreeSet<String> = self
            .plans
            .iter()
            .filter(|(_, plan)| {
                plan.current()
                    .requirements
                    .iter()
                    .any(|requirement| requirement.source_key() == Some(source_key.as_str()))
            })
            .map(|(plan_id, _)| plan_id.clone())
            .collect();
        let affected = self.dependent_closure(direct);
        let affected_plan_ids: Vec<String> = affected.iter().cloned().collect();
        let evaluations = self.evaluate_plan_set(affected, now)?;

        let response = json!({
            "event_id": event_id,
            "idempotent": false,
            "source_key": source_key,
            "source_revision": source_revision,
            "evidence_provenance": "caller_asserted",
            "affected_plan_ids": affected_plan_ids,
            "evaluations": evaluations,
            "attention": self.queue_partition(now),
        });
        self.observation_receipts.insert(
            event_id.clone(),
            ObservationReceipt {
                event_id,
                command_digest,
                source_key,
                source_revision,
                response: response.clone(),
            },
        );
        Ok(response)
    }

    fn evaluate(&mut self, id: Option<&str>, now: DateTime<Utc>) -> Result<Value, String> {
        let roots = if let Some(id) = id {
            validate_id("plan id", id)?;
            if !self.plans.contains_key(id) {
                return Err(format!("unknown plan '{id}'"));
            }
            self.dependent_closure([id.to_owned()].into_iter().collect())
        } else {
            self.plans.keys().cloned().collect()
        };
        let evaluations = self.evaluate_plan_set(roots, now)?;
        Ok(json!({
            "evaluations": evaluations,
            "attention": self.queue_partition(now),
        }))
    }

    fn complete(
        &mut self,
        id: String,
        expected_version: u32,
        basis: CompletionBasis,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        validate_id("plan id", &id)?;
        let plan = self
            .plans
            .get(&id)
            .ok_or_else(|| format!("unknown plan '{id}'"))?;
        if plan.lifecycle == PlanLifecycle::Cancelled {
            return Err(format!("cancelled plan '{id}' cannot be completed"));
        }
        if plan.current().version != expected_version {
            return Err(format!(
                "stale plan version for '{id}': expected {expected_version}, current is {}",
                plan.current().version
            ));
        }
        if let Some(completion) = &plan.completion {
            if plan.lifecycle == PlanLifecycle::Fulfilled && completion.basis == basis {
                return Ok(json!({
                    "plan": plan,
                    "idempotent": true,
                    "attention": self.queue_partition(now),
                }));
            }
            if plan.lifecycle == PlanLifecycle::Fulfilled {
                return Err(format!(
                    "plan '{id}' is already fulfilled with a different completion basis"
                ));
            }
            if plan.lifecycle == PlanLifecycle::Disputed
                && completion.plan_version == expected_version
            {
                return Err(format!(
                    "disputed plan '{id}' must be explicitly revised before recompletion"
                ));
            }
        }
        validate_completion_basis(&basis)?;

        if let CompletionBasis::Evidence { requirement_ids } = &basis {
            let snapshot = self.evaluate_plan_read_only(&id, now)?;
            let by_id: BTreeMap<&str, RequirementState> = snapshot
                .requirements
                .iter()
                .map(|evaluation| (evaluation.requirement_id.as_str(), evaluation.state))
                .collect();
            if let Some(unsatisfied) = snapshot
                .requirements
                .iter()
                .find(|evaluation| evaluation.state != RequirementState::Satisfied)
            {
                return Err(format!(
                    "evidence completion rejected: all current requirements must be satisfied; '{}' is {:?}",
                    unsatisfied.requirement_id, unsatisfied.state
                ));
            }
            for requirement_id in requirement_ids {
                match by_id.get(requirement_id.as_str()) {
                    Some(RequirementState::Satisfied) => {}
                    Some(state) => {
                        return Err(format!(
                            "evidence completion rejected: requirement '{requirement_id}' is {state:?}"
                        ));
                    }
                    None => {
                        return Err(format!(
                            "evidence completion rejected: unknown requirement '{requirement_id}'"
                        ));
                    }
                }
            }
        }

        let plan = self.plans.get_mut(&id).expect("plan existence checked");
        plan.lifecycle = PlanLifecycle::Fulfilled;
        plan.completion = Some(CompletionRecord {
            basis,
            plan_version: expected_version,
            completed_at: now,
            disputed_at: None,
            dispute_reason: None,
        });
        plan.cancellation = None;

        let affected = self.dependent_closure([id.clone()].into_iter().collect());
        let evaluations = self.evaluate_plan_set(affected, now)?;
        Ok(json!({
            "plan": self.plans.get(&id),
            "evaluations": evaluations,
            "attention": self.queue_partition(now),
        }))
    }

    fn cancel(
        &mut self,
        id: String,
        expected_version: u32,
        reason: String,
        now: DateTime<Utc>,
    ) -> Result<Value, String> {
        validate_id("plan id", &id)?;
        validate_bounded_nonempty("cancellation reason", &reason, MAX_TEXT_BYTES)?;
        let plan = self
            .plans
            .get(&id)
            .ok_or_else(|| format!("unknown plan '{id}'"))?;
        if plan.current().version != expected_version {
            return Err(format!(
                "stale plan version for '{id}': expected {expected_version}, current is {}",
                plan.current().version
            ));
        }
        if let Some(cancellation) = &plan.cancellation {
            if cancellation.reason == reason {
                return Ok(json!({
                    "plan": plan,
                    "idempotent": true,
                    "attention": self.queue_partition(now),
                }));
            }
            return Err(format!(
                "plan '{id}' is already cancelled with a different reason"
            ));
        }
        let plan = self.plans.get_mut(&id).expect("plan existence checked");
        plan.lifecycle = PlanLifecycle::Cancelled;
        plan.cancellation = Some(CancellationRecord {
            reason,
            plan_version: expected_version,
            cancelled_at: now,
        });

        let affected = self.dependent_closure([id.clone()].into_iter().collect());
        let evaluations = self.evaluate_plan_set(affected, now)?;
        Ok(json!({
            "plan": self.plans.get(&id),
            "evaluations": evaluations,
            "attention": self.queue_partition(now),
        }))
    }

    fn acknowledge(&mut self, queue_id: String, now: DateTime<Utc>) -> Result<Value, String> {
        validate_id("queue id", &queue_id)?;
        let event = self
            .attention_queue
            .get_mut(&queue_id)
            .ok_or_else(|| format!("unknown attention queue id '{queue_id}'"))?;
        if event.superseded_at.is_some() {
            return Err(format!(
                "attention queue id '{queue_id}' has been superseded"
            ));
        }
        if event.acknowledged_at.is_none() {
            event.acknowledged_at = Some(now);
            if let Some(plan) = self.plans.get_mut(&event.plan_id) {
                plan.last_attention_acknowledged_at = Some(now);
            }
        }
        Ok(json!({
            "attention_event": self.attention_queue.get(&queue_id),
            "attention": self.queue_partition(now),
        }))
    }

    fn explain(&self, id: &str, now: DateTime<Utc>) -> Result<Value, String> {
        validate_id("plan id", id)?;
        let plan = self
            .plans
            .get(id)
            .ok_or_else(|| format!("unknown plan '{id}'"))?;
        let evaluation = self.evaluate_plan_read_only(id, now)?;
        let source_keys: BTreeSet<String> = plan
            .current()
            .requirements
            .iter()
            .filter_map(|requirement| requirement.source_key().map(str::to_owned))
            .collect();
        let sources: Vec<&EvidenceSource> = source_keys
            .iter()
            .filter_map(|source_key| self.sources.get(source_key))
            .collect();
        let pending_attention: Vec<&AttentionEvent> = self
            .attention_queue
            .values()
            .filter(|event| {
                event.plan_id == id
                    && event.acknowledged_at.is_none()
                    && event.superseded_at.is_none()
            })
            .collect();

        Ok(json!({
            "plan": plan,
            "current": plan.current(),
            "evaluation": evaluation,
            "sources": sources,
            "pending_attention": pending_attention,
            "evidence_boundary": {
                "provenance": "caller_asserted",
                "cryptographic_proof": false,
                "grants_authority": false,
                "note": "Evaluation records are deterministic labels over caller-supplied evidence, not proof, truth verification, or authorization for an effect."
            }
        }))
    }

    fn portfolio(&self, now: DateTime<Utc>) -> Result<Value, String> {
        let mut summaries = Vec::with_capacity(self.plans.len());
        for plan in self.plans.values() {
            let evaluation = self.evaluate_plan_read_only(&plan.id, now)?;
            summaries.push(json!({
                "id": plan.id,
                "version": plan.current().version,
                "description": plan.current().description,
                "priority": plan.current().priority,
                "lifecycle": plan.lifecycle,
                "readiness": evaluation.readiness,
            }));
        }
        summaries.sort_by(|a, b| {
            let a_id = a.get("id").and_then(Value::as_str).unwrap_or_default();
            let b_id = b.get("id").and_then(Value::as_str).unwrap_or_default();
            let a_priority = self
                .plans
                .get(a_id)
                .map(|plan| plan.current().priority)
                .unwrap_or(Priority::Low);
            let b_priority = self
                .plans
                .get(b_id)
                .map(|plan| plan.current().priority)
                .unwrap_or(Priority::Low);
            b_priority.cmp(&a_priority).then_with(|| a_id.cmp(b_id))
        });

        let mut shared_by_signature: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        let mut conflicts_by_key: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut unmet_prerequisites = Vec::new();
        for plan in self
            .plans
            .values()
            .filter(|plan| !matches!(plan.lifecycle, PlanLifecycle::Cancelled))
        {
            for requirement in &plan.current().requirements {
                let signature = requirement_signature(requirement)?;
                shared_by_signature
                    .entry(signature)
                    .or_default()
                    .push(json!({
                        "plan_id": plan.id,
                        "requirement_id": requirement.id(),
                    }));
                if let Requirement::PlanFulfilled {
                    id: requirement_id,
                    plan_id: prerequisite_plan_id,
                } = requirement
                {
                    let prerequisite_status = self
                        .plans
                        .get(prerequisite_plan_id)
                        .map(|prerequisite| prerequisite.lifecycle);
                    if prerequisite_status != Some(PlanLifecycle::Fulfilled) {
                        unmet_prerequisites.push(json!({
                            "plan_id": plan.id,
                            "requirement_id": requirement_id,
                            "prerequisite_plan_id": prerequisite_plan_id,
                            "prerequisite_status": prerequisite_status,
                        }));
                    }
                }
            }
            if matches!(
                plan.lifecycle,
                PlanLifecycle::Active | PlanLifecycle::Disputed
            ) {
                for conflict_key in &plan.current().conflict_keys {
                    conflicts_by_key
                        .entry(conflict_key.clone())
                        .or_default()
                        .push(plan.id.clone());
                }
            }
        }

        let shared_requirements: Vec<Value> = shared_by_signature
            .into_iter()
            .filter_map(|(signature, uses)| {
                (uses.len() > 1).then(|| json!({ "signature": signature, "uses": uses }))
            })
            .collect();
        let conflicts: Vec<Value> = conflicts_by_key
            .into_iter()
            .filter_map(|(conflict_key, plan_ids)| {
                (plan_ids.len() > 1)
                    .then(|| json!({ "conflict_key": conflict_key, "plan_ids": plan_ids }))
            })
            .collect();

        Ok(json!({
            "plans": summaries,
            "shared_requirements": shared_requirements,
            "conflicts": conflicts,
            "unmet_prerequisites": unmet_prerequisites,
            "attention": self.queue_partition(now),
        }))
    }

    fn evaluate_plan_set(
        &mut self,
        plan_ids: BTreeSet<String>,
        now: DateTime<Utc>,
    ) -> Result<Vec<EvaluationRecord>, String> {
        let ordered = self.topological_plan_ids(&plan_ids);
        let mut records = Vec::with_capacity(ordered.len());
        for plan_id in ordered {
            records.push(self.evaluate_and_record(&plan_id, now)?);
        }
        Ok(records)
    }

    fn evaluate_and_record(
        &mut self,
        plan_id: &str,
        now: DateTime<Utc>,
    ) -> Result<EvaluationRecord, String> {
        let initial = self.evaluate_plan_read_only(plan_id, now)?;
        let dispute_requirement = {
            let plan = self
                .plans
                .get(plan_id)
                .ok_or_else(|| format!("unknown plan '{plan_id}'"))?;
            if plan.lifecycle != PlanLifecycle::Fulfilled {
                None
            } else {
                match plan.completion.as_ref().map(|record| &record.basis) {
                    Some(CompletionBasis::Evidence { .. }) => initial
                        .requirements
                        .iter()
                        .find(|evaluation| evaluation.state != RequirementState::Satisfied)
                        .map(|evaluation| evaluation.requirement_id.clone()),
                    _ => None,
                }
            }
        };
        if let Some(requirement_id) = dispute_requirement {
            let plan = self.plans.get_mut(plan_id).expect("plan existence checked");
            plan.lifecycle = PlanLifecycle::Disputed;
            if let Some(completion) = &mut plan.completion {
                completion.disputed_at = Some(now);
                completion.dispute_reason = Some(format!(
                    "completion evidence requirement '{requirement_id}' is no longer satisfied"
                ));
            }
        }

        let snapshot = self.evaluate_plan_read_only(plan_id, now)?;
        let label_material = json!({
            "plan_id": snapshot.plan_id,
            "plan_version": snapshot.plan_version,
            "lifecycle": snapshot.lifecycle,
            "readiness": snapshot.readiness,
            "requirements": snapshot.requirements,
            "evaluated_at": now,
        });
        let label = format!("eval_{}", stable_digest(&label_material)?);
        let record = EvaluationRecord {
            label: label.clone(),
            evaluated_at: now,
            snapshot,
            record_kind: EvaluationRecordKind::DeterministicEvaluation,
            cryptographic_proof: false,
            grants_authority: false,
        };
        if !self.evaluations.contains_key(&label) && self.evaluations.len() >= MAX_EVALUATIONS {
            return Err(format!(
                "evaluation record limit of {MAX_EVALUATIONS} reached"
            ));
        }
        self.evaluations.insert(label, record.clone());
        self.maybe_enqueue_attention(&record, now)?;
        Ok(record)
    }

    fn evaluate_plan_read_only(
        &self,
        plan_id: &str,
        now: DateTime<Utc>,
    ) -> Result<EvaluationSnapshot, String> {
        let plan = self
            .plans
            .get(plan_id)
            .ok_or_else(|| format!("unknown plan '{plan_id}'"))?;
        let current = plan.current();
        let requirements: Vec<RequirementEvaluation> = current
            .requirements
            .iter()
            .map(|requirement| self.evaluate_requirement(requirement, now))
            .collect();
        let readiness = if requirements.iter().any(|evaluation| {
            matches!(
                evaluation.state,
                RequirementState::Unmet | RequirementState::Missing
            )
        }) {
            Readiness::Blocked
        } else if requirements
            .iter()
            .any(|evaluation| evaluation.state == RequirementState::Unknown)
        {
            Readiness::Unknown
        } else {
            Readiness::Ready
        };
        Ok(EvaluationSnapshot {
            plan_id: plan_id.to_owned(),
            plan_version: current.version,
            lifecycle: plan.lifecycle,
            readiness,
            requirements,
        })
    }

    fn evaluate_requirement(
        &self,
        requirement: &Requirement,
        now: DateTime<Utc>,
    ) -> RequirementEvaluation {
        match requirement {
            Requirement::Evidence {
                id,
                source_key,
                condition,
                ..
            } => {
                let Some(source) = self.sources.get(source_key) else {
                    return requirement_evaluation(
                        id,
                        RequirementState::Unknown,
                        "source has not been observed",
                        Some(source_key),
                        None,
                    );
                };
                let (state, detail) = match (condition, source.value.as_ref()) {
                    (EvidenceCondition::Exists, Some(_)) => {
                        (RequirementState::Satisfied, "source has a value")
                    }
                    (EvidenceCondition::Exists, None) => (
                        RequirementState::Missing,
                        "source was observed without a value",
                    ),
                    (EvidenceCondition::Equals { .. }, None)
                    | (EvidenceCondition::NotEquals { .. }, None) => (
                        RequirementState::Missing,
                        "source was observed without a comparable value",
                    ),
                    (EvidenceCondition::Equals { value }, Some(actual)) if actual == value => (
                        RequirementState::Satisfied,
                        "source value equals expected value",
                    ),
                    (EvidenceCondition::Equals { .. }, Some(_)) => (
                        RequirementState::Unmet,
                        "source value does not equal expected value",
                    ),
                    (EvidenceCondition::NotEquals { value }, Some(actual)) if actual != value => (
                        RequirementState::Satisfied,
                        "source value differs from excluded value",
                    ),
                    (EvidenceCondition::NotEquals { .. }, Some(_)) => (
                        RequirementState::Unmet,
                        "source value equals excluded value",
                    ),
                };
                requirement_evaluation(id, state, detail, Some(source_key), Some(source.revision))
            }
            Requirement::Event {
                id,
                source_key,
                event,
                expectation,
                window_start,
                window_end,
                grace_seconds,
                ..
            } => {
                let Some(source) = self.sources.get(source_key) else {
                    return requirement_evaluation(
                        id,
                        RequirementState::Unknown,
                        "event source has not been observed",
                        Some(source_key),
                        None,
                    );
                };
                let occurred = source.observed_events.contains(event);
                let occurrence_applies =
                    occurred && event_occurrence_applies(source, *window_start, *window_end);
                let coverage_complete = event_coverage_complete(
                    source,
                    event,
                    *window_start,
                    *window_end,
                    *grace_seconds,
                    now,
                );
                let (state, detail) = match (
                    expectation,
                    occurred,
                    occurrence_applies,
                    coverage_complete,
                ) {
                    (EventExpectation::Occurred, true, true, _) => (
                        RequirementState::Satisfied,
                        "required event is present in the supplied snapshot",
                    ),
                    (EventExpectation::Occurred, true, false, _) => (
                        RequirementState::Unknown,
                        "event is present but its observation interval is not inside the required window",
                    ),
                    (EventExpectation::Occurred, false, _, true) => (
                        RequirementState::Missing,
                        "required event is absent from complete observation coverage",
                    ),
                    (EventExpectation::Occurred, false, _, false) => (
                        RequirementState::Unknown,
                        "required event is absent but observation coverage is incomplete",
                    ),
                    (EventExpectation::Absent, true, true, _) => (
                        RequirementState::Unmet,
                        "excluded event is present in the supplied snapshot",
                    ),
                    (EventExpectation::Absent, true, false, _) => (
                        RequirementState::Unknown,
                        "excluded event is present but is not placed inside the required window",
                    ),
                    (EventExpectation::Absent, false, _, true) => (
                        RequirementState::Satisfied,
                        "excluded event is absent after complete window coverage and grace",
                    ),
                    (EventExpectation::Absent, false, _, false) => (
                        RequirementState::Unknown,
                        "event absence is not established by complete elapsed window coverage",
                    ),
                };
                requirement_evaluation(id, state, detail, Some(source_key), Some(source.revision))
            }
            Requirement::PlanFulfilled { id, plan_id } => match self.plans.get(plan_id) {
                Some(plan) if plan.lifecycle == PlanLifecycle::Fulfilled => requirement_evaluation(
                    id,
                    RequirementState::Satisfied,
                    "prerequisite plan is fulfilled",
                    None,
                    None,
                ),
                Some(plan) => requirement_evaluation(
                    id,
                    RequirementState::Unmet,
                    &format!("prerequisite plan is {:?}", plan.lifecycle).to_lowercase(),
                    None,
                    None,
                ),
                None => requirement_evaluation(
                    id,
                    RequirementState::Missing,
                    "prerequisite plan does not exist",
                    None,
                    None,
                ),
            },
        }
    }

    fn maybe_enqueue_attention(
        &mut self,
        record: &EvaluationRecord,
        now: DateTime<Utc>,
    ) -> Result<(), String> {
        let plan = self
            .plans
            .get(&record.snapshot.plan_id)
            .ok_or_else(|| format!("unknown plan '{}'", record.snapshot.plan_id))?;
        let requirement_states: Vec<Value> = record
            .snapshot
            .requirements
            .iter()
            .map(|requirement| {
                json!({
                    "requirement_id": requirement.requirement_id,
                    "state": requirement.state,
                })
            })
            .collect();
        let fingerprint_material = json!({
            "plan_id": record.snapshot.plan_id,
            "plan_version": record.snapshot.plan_version,
            "lifecycle": record.snapshot.lifecycle,
            "readiness": record.snapshot.readiness,
            "requirement_states": requirement_states,
        });
        let fingerprint = stable_digest(&fingerprint_material)?;
        if self.attention_queue.values().any(|event| {
            event.plan_id == record.snapshot.plan_id
                && event.superseded_at.is_none()
                && event.fingerprint == fingerprint
        }) {
            return Ok(());
        }
        for event in self.attention_queue.values_mut().filter(|event| {
            event.plan_id == record.snapshot.plan_id
                && event.superseded_at.is_none()
                && event.fingerprint != fingerprint
        }) {
            event.superseded_at = Some(now);
        }
        let needs_attention = record.snapshot.lifecycle == PlanLifecycle::Disputed
            || record.snapshot.lifecycle == PlanLifecycle::Active;
        if !needs_attention {
            return Ok(());
        }
        if self.attention_queue.len() >= MAX_ATTENTION_EVENTS {
            return Err(format!(
                "attention event limit of {MAX_ATTENTION_EVENTS} reached"
            ));
        }
        let sequence = self
            .attention_sequence_by_plan
            .get(&record.snapshot.plan_id)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| {
                format!(
                    "attention sequence overflow for plan '{}'",
                    record.snapshot.plan_id
                )
            })?;
        let queue_id = format!(
            "attn_{}",
            stable_digest(&json!({
                "plan_id": record.snapshot.plan_id,
                "fingerprint": fingerprint,
                "sequence": sequence,
            }))?
        );
        let interval = Duration::seconds(plan.current().min_attention_interval_seconds as i64);
        let deliver_after = match plan.last_attention_acknowledged_at {
            Some(last) => last
                .checked_add_signed(interval)
                .ok_or_else(|| "attention interval exceeds timestamp range".to_owned())?
                .max(now),
            None => now,
        };
        let reason = if record.snapshot.lifecycle == PlanLifecycle::Disputed {
            "previously fulfilled completion evidence is disputed".to_owned()
        } else if record.snapshot.readiness == Readiness::Ready {
            "plan requirements are ready".to_owned()
        } else {
            format!("plan readiness is {:?}", record.snapshot.readiness).to_lowercase()
        };
        self.attention_queue.insert(
            queue_id.clone(),
            AttentionEvent {
                queue_id,
                fingerprint,
                plan_id: record.snapshot.plan_id.clone(),
                priority: plan.current().priority,
                created_at: now,
                deliver_after,
                reason,
                acknowledged_at: None,
                superseded_at: None,
            },
        );
        self.attention_sequence_by_plan
            .insert(record.snapshot.plan_id.clone(), sequence);
        Ok(())
    }

    fn dependent_closure(&self, mut affected: BTreeSet<String>) -> BTreeSet<String> {
        loop {
            let mut changed = false;
            for (plan_id, plan) in &self.plans {
                if affected.contains(plan_id) {
                    continue;
                }
                if plan.current().requirements.iter().any(|requirement| {
                    requirement
                        .prerequisite_plan_id()
                        .is_some_and(|prerequisite| affected.contains(prerequisite))
                }) {
                    affected.insert(plan_id.clone());
                    changed = true;
                }
            }
            if !changed {
                return affected;
            }
        }
    }

    fn topological_plan_ids(&self, plan_ids: &BTreeSet<String>) -> Vec<String> {
        fn visit(
            graph: &IntentionGraph,
            plan_ids: &BTreeSet<String>,
            plan_id: &str,
            visited: &mut BTreeSet<String>,
            ordered: &mut Vec<String>,
        ) {
            if !visited.insert(plan_id.to_owned()) {
                return;
            }
            if let Some(plan) = graph.plans.get(plan_id) {
                for prerequisite in plan
                    .current()
                    .requirements
                    .iter()
                    .filter_map(Requirement::prerequisite_plan_id)
                    .filter(|prerequisite| plan_ids.contains(*prerequisite))
                {
                    visit(graph, plan_ids, prerequisite, visited, ordered);
                }
            }
            ordered.push(plan_id.to_owned());
        }

        let mut visited = BTreeSet::new();
        let mut ordered = Vec::with_capacity(plan_ids.len());
        for plan_id in plan_ids {
            visit(self, plan_ids, plan_id, &mut visited, &mut ordered);
        }
        ordered
    }

    fn validate_dependency_graph(
        &self,
        candidate_id: &str,
        candidate_requirements: &[Requirement],
    ) -> Result<(), String> {
        for prerequisite_id in candidate_requirements
            .iter()
            .filter_map(Requirement::prerequisite_plan_id)
        {
            if !self.plans.contains_key(prerequisite_id) {
                return Err(format!(
                    "plan '{candidate_id}' references unknown prerequisite plan '{prerequisite_id}'"
                ));
            }
        }

        fn visit(
            graph: &IntentionGraph,
            candidate_id: &str,
            candidate_requirements: &[Requirement],
            current: &str,
            visiting: &mut BTreeSet<String>,
            visited: &mut BTreeSet<String>,
        ) -> bool {
            if !visiting.insert(current.to_owned()) {
                return true;
            }
            if visited.contains(current) {
                visiting.remove(current);
                return false;
            }
            let requirements = if current == candidate_id {
                candidate_requirements
            } else if let Some(plan) = graph.plans.get(current) {
                &plan.current().requirements
            } else {
                visiting.remove(current);
                return false;
            };
            for next in requirements
                .iter()
                .filter_map(Requirement::prerequisite_plan_id)
            {
                if visit(
                    graph,
                    candidate_id,
                    candidate_requirements,
                    next,
                    visiting,
                    visited,
                ) {
                    return true;
                }
            }
            visiting.remove(current);
            visited.insert(current.to_owned());
            false
        }

        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        if visit(
            self,
            candidate_id,
            candidate_requirements,
            candidate_id,
            &mut visiting,
            &mut visited,
        ) {
            return Err(format!(
                "plan '{candidate_id}' would introduce a prerequisite cycle"
            ));
        }
        Ok(())
    }

    fn queue_partition(&self, now: DateTime<Utc>) -> Value {
        let mut pending: Vec<&AttentionEvent> = self
            .attention_queue
            .values()
            .filter(|event| event.acknowledged_at.is_none() && event.superseded_at.is_none())
            .collect();
        pending.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.deliver_after.cmp(&b.deliver_after))
                .then_with(|| a.queue_id.cmp(&b.queue_id))
        });
        let (deliverable, withheld): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .partition(|event| event.deliver_after <= now);
        json!({
            "deliverable_queue_ids": deliverable
                .iter()
                .map(|event| event.queue_id.as_str())
                .collect::<Vec<_>>(),
            "withheld_queue_ids": withheld
                .iter()
                .map(|event| event.queue_id.as_str())
                .collect::<Vec<_>>(),
            "pending_count": deliverable.len() + withheld.len(),
        })
    }
}

/// Deterministically reconstruct a graph from a caller-owned journal.
pub fn replay(entries: &[TimedCommand]) -> Result<IntentionGraph, String> {
    let mut graph = IntentionGraph::default();
    for (index, entry) in entries.iter().enumerate() {
        graph
            .apply(entry.command.clone(), entry.now)
            .map_err(|error| format!("replay entry {index} failed: {error}"))?;
    }
    Ok(graph)
}

/// Public schema for the nested command payload. It intentionally documents
/// semantic boundaries in addition to field types because the engine cannot
/// verify a source or authorize an external action.
pub fn schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Vestige Intention Evidence Lifecycle",
        "schema_version": INTENTION_GRAPH_SCHEMA_VERSION,
        "algorithm_version": INTENTION_GRAPH_ALGORITHM_VERSION,
        "type": "object",
        "oneOf": [
                {
                    "title": "plan",
                    "type": "object",
                    "required": ["action", "id", "description"],
                    "properties": {
                        "action": { "const": "plan" },
                        "id": bounded_string_schema(MAX_ID_BYTES),
                        "description": bounded_string_schema(MAX_DESCRIPTION_BYTES),
                        "priority": priority_schema(),
                        "requirements": requirements_schema(),
                        "min_attention_interval_seconds": { "type": "integer", "minimum": 0, "maximum": MAX_ATTENTION_INTERVAL_SECONDS },
                        "conflict_keys": bounded_string_array_schema(MAX_CONFLICT_KEYS, MAX_ID_BYTES),
                    },
                    "additionalProperties": false,
                },
                {
                    "title": "revise",
                    "type": "object",
                    "required": ["action", "id", "expected_version"],
                    "properties": {
                        "action": { "const": "revise" },
                        "id": bounded_string_schema(MAX_ID_BYTES),
                        "expected_version": { "type": "integer", "minimum": 1 },
                        "description": bounded_string_schema(MAX_DESCRIPTION_BYTES),
                        "priority": priority_schema(),
                        "requirements": requirements_schema(),
                        "min_attention_interval_seconds": { "type": "integer", "minimum": 0, "maximum": MAX_ATTENTION_INTERVAL_SECONDS },
                        "conflict_keys": bounded_string_array_schema(MAX_CONFLICT_KEYS, MAX_ID_BYTES),
                    },
                    "additionalProperties": false,
                },
                {
                    "title": "observe",
                    "type": "object",
                    "required": ["action", "event_id", "source_key", "source_revision"],
                    "properties": {
                        "action": { "const": "observe" },
                        "event_id": bounded_string_schema(MAX_ID_BYTES),
                        "source_key": bounded_string_schema(MAX_SOURCE_KEY_BYTES),
                        "source_revision": { "type": "integer", "minimum": 1 },
                        "value": evidence_value_schema(true),
                        "observed_events": bounded_string_array_schema(MAX_EVENTS_PER_OBSERVATION, MAX_ID_BYTES),
                        "covered_events": bounded_string_array_schema(MAX_EVENTS_PER_OBSERVATION, MAX_ID_BYTES),
                        "coverage_start": { "type": "string", "format": "date-time" },
                        "coverage_end": { "type": "string", "format": "date-time" },
                    },
                    "additionalProperties": false,
                },
                {
                    "title": "evaluate",
                    "type": "object",
                    "required": ["action"],
                    "properties": { "action": { "const": "evaluate" }, "id": bounded_string_schema(MAX_ID_BYTES) },
                    "additionalProperties": false,
                },
                {
                    "title": "explain",
                    "type": "object",
                    "required": ["action", "id"],
                    "properties": { "action": { "const": "explain" }, "id": bounded_string_schema(MAX_ID_BYTES) },
                    "additionalProperties": false,
                },
                {
                    "title": "portfolio",
                    "type": "object",
                    "required": ["action"],
                    "properties": { "action": { "const": "portfolio" } },
                    "additionalProperties": false,
                },
                {
                    "title": "complete",
                    "type": "object",
                    "required": ["action", "id", "expected_version", "basis"],
                    "properties": {
                        "action": { "const": "complete" },
                        "id": bounded_string_schema(MAX_ID_BYTES),
                        "expected_version": { "type": "integer", "minimum": 1 },
                        "basis": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "required": ["type", "report"],
                                    "properties": { "type": { "const": "user_report" }, "report": bounded_string_schema(MAX_TEXT_BYTES) },
                                    "additionalProperties": false,
                                },
                                {
                                    "type": "object",
                                    "required": ["type", "requirement_ids"],
                                    "properties": { "type": { "const": "evidence" }, "requirement_ids": bounded_string_array_schema(MAX_REQUIREMENTS_PER_PLAN, MAX_ID_BYTES) },
                                    "additionalProperties": false,
                                }
                            ]
                        }
                    },
                    "additionalProperties": false,
                },
                {
                    "title": "cancel",
                    "type": "object",
                    "required": ["action", "id", "expected_version", "reason"],
                    "properties": { "action": { "const": "cancel" }, "id": bounded_string_schema(MAX_ID_BYTES), "expected_version": { "type": "integer", "minimum": 1 }, "reason": bounded_string_schema(MAX_TEXT_BYTES) },
                    "additionalProperties": false,
                },
                {
                    "title": "acknowledge",
                    "type": "object",
                    "required": ["action", "queue_id"],
                    "properties": { "action": { "const": "acknowledge" }, "queue_id": bounded_string_schema(MAX_ID_BYTES) },
                    "additionalProperties": false,
                }
        ],
        "semantics": {
            "evidence_provenance": "caller_asserted",
            "evaluation_records": "deterministic labels, not cryptographic proof or authorization",
            "absence": "requires an elapsed explicit requirement window plus complete source coverage of that window and named event",
            "external_effects": "never performed or authorized by this engine, including purchases",
            "journal": "caller-owned; replay accepts timestamped commands",
        },
        "limits": {
            "plans": MAX_PLANS,
            "requirements_per_plan": MAX_REQUIREMENTS_PER_PLAN,
            "sources": MAX_SOURCES,
            "event_receipts": MAX_EVENT_RECEIPTS,
        }
    })
}

fn requirements_schema() -> Value {
    json!({
        "type": "array",
        "maxItems": MAX_REQUIREMENTS_PER_PLAN,
        "items": {
            "oneOf": [
                {
                    "title": "evidence",
                    "type": "object",
                    "required": ["type", "id", "source_key", "condition"],
                    "properties": {
                        "type": { "const": "evidence" },
                        "id": bounded_string_schema(MAX_ID_BYTES),
                        "source_key": bounded_string_schema(MAX_SOURCE_KEY_BYTES),
                        "memory_id": bounded_string_schema(MAX_ID_BYTES),
                        "condition": {
                            "oneOf": [
                                { "type": "object", "required": ["op"], "properties": { "op": { "const": "exists" } }, "additionalProperties": false },
                                { "type": "object", "required": ["op", "value"], "properties": { "op": { "const": "equals" }, "value": evidence_value_schema(false) }, "additionalProperties": false },
                                { "type": "object", "required": ["op", "value"], "properties": { "op": { "const": "not_equals" }, "value": evidence_value_schema(false) }, "additionalProperties": false },
                            ]
                        }
                    },
                    "additionalProperties": false,
                },
                {
                    "title": "event",
                    "type": "object",
                    "required": ["type", "id", "source_key", "event", "expectation"],
                    "properties": {
                        "type": { "const": "event" },
                        "id": bounded_string_schema(MAX_ID_BYTES),
                        "source_key": bounded_string_schema(MAX_SOURCE_KEY_BYTES),
                        "memory_id": bounded_string_schema(MAX_ID_BYTES),
                        "event": bounded_string_schema(MAX_ID_BYTES),
                        "expectation": { "enum": ["occurred", "absent"] },
                        "window_start": { "type": "string", "format": "date-time" },
                        "window_end": { "type": "string", "format": "date-time" },
                        "grace_seconds": { "type": "integer", "minimum": 0, "maximum": MAX_ATTENTION_INTERVAL_SECONDS },
                    },
                    "additionalProperties": false,
                },
                {
                    "title": "plan_fulfilled",
                    "type": "object",
                    "required": ["type", "id", "plan_id"],
                    "properties": {
                        "type": { "const": "plan_fulfilled" },
                        "id": bounded_string_schema(MAX_ID_BYTES),
                        "plan_id": bounded_string_schema(MAX_ID_BYTES),
                    },
                    "additionalProperties": false,
                }
            ]
        }
    })
}

fn priority_schema() -> Value {
    json!({ "enum": ["low", "normal", "high", "critical"] })
}

fn evidence_value_schema(nullable: bool) -> Value {
    let mut variants = vec![
        json!({ "type": "boolean" }),
        json!({ "type": "integer", "minimum": i64::MIN, "maximum": i64::MAX }),
        bounded_string_schema(MAX_TEXT_BYTES),
    ];
    if nullable {
        variants.push(json!({ "type": "null" }));
    }
    json!({ "oneOf": variants })
}

fn bounded_string_schema(max_length: usize) -> Value {
    json!({ "type": "string", "minLength": 1, "maxLength": max_length })
}

fn bounded_string_array_schema(max_items: usize, max_length: usize) -> Value {
    json!({
        "type": "array",
        "maxItems": max_items,
        "uniqueItems": true,
        "items": bounded_string_schema(max_length),
    })
}

fn validate_plan_fields(
    plan_id: &str,
    requirements: &[Requirement],
    min_attention_interval_seconds: u64,
    conflict_keys: &[String],
) -> Result<(), String> {
    if requirements.len() > MAX_REQUIREMENTS_PER_PLAN {
        return Err(format!(
            "plan '{plan_id}' has {} requirements; maximum is {MAX_REQUIREMENTS_PER_PLAN}",
            requirements.len()
        ));
    }
    if min_attention_interval_seconds > MAX_ATTENTION_INTERVAL_SECONDS {
        return Err(format!(
            "min_attention_interval_seconds exceeds {MAX_ATTENTION_INTERVAL_SECONDS}"
        ));
    }
    if conflict_keys.len() > MAX_CONFLICT_KEYS {
        return Err(format!(
            "plan '{plan_id}' has {} conflict keys; maximum is {MAX_CONFLICT_KEYS}",
            conflict_keys.len()
        ));
    }
    let mut requirement_ids = BTreeSet::new();
    for requirement in requirements {
        validate_requirement(plan_id, requirement)?;
        if !requirement_ids.insert(requirement.id()) {
            return Err(format!(
                "duplicate requirement id '{}' in plan '{plan_id}'",
                requirement.id()
            ));
        }
    }
    validate_unique_strings("conflict_keys", conflict_keys, MAX_ID_BYTES)?;
    Ok(())
}

fn validate_requirement(plan_id: &str, requirement: &Requirement) -> Result<(), String> {
    validate_id("requirement id", requirement.id())?;
    match requirement {
        Requirement::Evidence {
            source_key,
            memory_id,
            condition,
            ..
        } => {
            validate_source_key(source_key)?;
            if let Some(memory_id) = memory_id {
                validate_id("memory id", memory_id)?;
            }
            validate_memory_binding(source_key, memory_id.as_deref())?;
            match condition {
                EvidenceCondition::Exists => {}
                EvidenceCondition::Equals { value } | EvidenceCondition::NotEquals { value } => {
                    validate_evidence_value(Some(value))?;
                }
            }
        }
        Requirement::Event {
            source_key,
            memory_id,
            event,
            expectation,
            window_start,
            window_end,
            grace_seconds,
            ..
        } => {
            validate_source_key(source_key)?;
            if let Some(memory_id) = memory_id {
                validate_id("memory id", memory_id)?;
            }
            validate_memory_binding(source_key, memory_id.as_deref())?;
            validate_bounded_nonempty("event name", event, MAX_ID_BYTES)?;
            validate_window(*window_start, *window_end, "event requirement")?;
            if *grace_seconds > MAX_ATTENTION_INTERVAL_SECONDS {
                return Err(format!(
                    "event grace_seconds exceeds {MAX_ATTENTION_INTERVAL_SECONDS}"
                ));
            }
            if *expectation == EventExpectation::Absent
                && (window_start.is_none() || window_end.is_none())
            {
                return Err(
                    "an absent event requirement needs window_start and window_end".to_owned(),
                );
            }
        }
        Requirement::PlanFulfilled {
            plan_id: prerequisite_plan_id,
            ..
        } => {
            validate_id("prerequisite plan id", prerequisite_plan_id)?;
            if prerequisite_plan_id == plan_id {
                return Err(format!("plan '{plan_id}' cannot require itself"));
            }
        }
    }
    Ok(())
}

fn validate_completion_basis(basis: &CompletionBasis) -> Result<(), String> {
    match basis {
        CompletionBasis::UserReport { report } => {
            validate_bounded_nonempty("completion report", report, MAX_TEXT_BYTES)
        }
        CompletionBasis::Evidence { requirement_ids } => {
            if requirement_ids.is_empty() {
                return Err("evidence completion needs at least one requirement id".to_owned());
            }
            if requirement_ids.len() > MAX_REQUIREMENTS_PER_PLAN {
                return Err(format!(
                    "evidence completion has too many requirement ids; maximum is {MAX_REQUIREMENTS_PER_PLAN}"
                ));
            }
            validate_unique_strings("completion requirement_ids", requirement_ids, MAX_ID_BYTES)
        }
    }
}

fn validate_evidence_value(value: Option<&EvidenceValue>) -> Result<(), String> {
    if let Some(EvidenceValue::Text(text)) = value {
        validate_bounded_nonempty("evidence text", text, MAX_TEXT_BYTES)?;
    }
    Ok(())
}

fn validate_event_list(label: &str, events: &[String]) -> Result<(), String> {
    if events.len() > MAX_EVENTS_PER_OBSERVATION {
        return Err(format!(
            "{label} has {} entries; maximum is {MAX_EVENTS_PER_OBSERVATION}",
            events.len()
        ));
    }
    validate_unique_strings(label, events, MAX_ID_BYTES)
}

fn validate_unique_strings(label: &str, values: &[String], max_bytes: usize) -> Result<(), String> {
    let mut unique = BTreeSet::new();
    for value in values {
        validate_bounded_nonempty(label, value, max_bytes)?;
        if !unique.insert(value) {
            return Err(format!("{label} contains duplicate value '{value}'"));
        }
    }
    Ok(())
}

fn validate_window(
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
    label: &str,
) -> Result<(), String> {
    match (start, end) {
        (None, None) => Ok(()),
        (Some(start), Some(end)) if start < end => Ok(()),
        (Some(_), Some(_)) => Err(format!("{label} start must be before end")),
        _ => Err(format!("{label} requires both start and end")),
    }
}

fn validate_id(label: &str, value: &str) -> Result<(), String> {
    validate_bounded_nonempty(label, value, MAX_ID_BYTES)
}

fn validate_source_key(value: &str) -> Result<(), String> {
    validate_bounded_nonempty("source key", value, MAX_SOURCE_KEY_BYTES)
}

fn validate_memory_binding(source_key: &str, memory_id: Option<&str>) -> Result<(), String> {
    if let Some(memory_id) = memory_id {
        let expected = format!("memory:{memory_id}");
        if source_key != expected {
            return Err(format!(
                "memory_id '{memory_id}' must use source_key '{expected}'"
            ));
        }
    }
    Ok(())
}

fn validate_description(value: &str) -> Result<(), String> {
    validate_bounded_nonempty("description", value, MAX_DESCRIPTION_BYTES)
}

fn validate_bounded_nonempty(label: &str, value: &str, max_bytes: usize) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    if value.len() > max_bytes {
        return Err(format!(
            "{label} is {} bytes; maximum is {max_bytes}",
            value.len()
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(format!("{label} must not contain control characters"));
    }
    Ok(())
}

fn normalized_unique(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values.dedup();
    values
}

fn stable_digest<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("failed to serialize deterministic digest input: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn requirement_signature(requirement: &Requirement) -> Result<String, String> {
    let material = match requirement {
        Requirement::Evidence {
            source_key,
            memory_id,
            condition,
            ..
        } => json!({
            "type": "evidence",
            "source_key": source_key,
            "memory_id": memory_id,
            "condition": condition,
        }),
        Requirement::Event {
            source_key,
            memory_id,
            event,
            expectation,
            window_start,
            window_end,
            grace_seconds,
            ..
        } => json!({
            "type": "event",
            "source_key": source_key,
            "memory_id": memory_id,
            "event": event,
            "expectation": expectation,
            "window_start": window_start,
            "window_end": window_end,
            "grace_seconds": grace_seconds,
        }),
        Requirement::PlanFulfilled { plan_id, .. } => json!({
            "type": "plan_fulfilled",
            "plan_id": plan_id,
        }),
    };
    Ok(format!("req_{}", stable_digest(&material)?))
}

fn requirement_evaluation(
    requirement_id: &str,
    state: RequirementState,
    detail: &str,
    source_key: Option<&String>,
    source_revision: Option<u64>,
) -> RequirementEvaluation {
    RequirementEvaluation {
        requirement_id: requirement_id.to_owned(),
        state,
        detail: detail.to_owned(),
        source_key: source_key.cloned(),
        source_revision,
    }
}

fn event_coverage_complete(
    source: &EvidenceSource,
    event: &str,
    window_start: Option<DateTime<Utc>>,
    window_end: Option<DateTime<Utc>>,
    grace_seconds: u64,
    now: DateTime<Utc>,
) -> bool {
    if !source.covered_events.contains(event) {
        return false;
    }
    match (window_start, window_end) {
        (Some(required_start), Some(required_end)) => {
            let grace_elapsed = required_end
                .checked_add_signed(Duration::seconds(grace_seconds as i64))
                .is_some_and(|deadline| now >= deadline);
            let covers_window = source.coverage_start.zip(source.coverage_end).is_some_and(
                |(actual_start, actual_end)| {
                    actual_start <= required_start && actual_end >= required_end
                },
            );
            grace_elapsed && covers_window
        }
        (None, None) => true,
        _ => false,
    }
}

fn event_occurrence_applies(
    source: &EvidenceSource,
    window_start: Option<DateTime<Utc>>,
    window_end: Option<DateTime<Utc>>,
) -> bool {
    match (window_start, window_end) {
        (Some(required_start), Some(required_end)) => source
            .coverage_start
            .zip(source.coverage_end)
            .is_some_and(|(observed_start, observed_end)| {
                observed_start >= required_start && observed_end <= required_end
            }),
        (None, None) => true,
        _ => false,
    }
}

fn update_latest(latest: &mut Option<DateTime<Utc>>, candidate: DateTime<Utc>) {
    if latest.is_none_or(|current| candidate > current) {
        *latest = Some(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn t(hour: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 8, hour, 0, 0)
            .single()
            .unwrap()
    }

    fn evidence_requirement(id: &str, source_key: &str, expected: bool) -> Requirement {
        Requirement::Evidence {
            id: id.to_owned(),
            source_key: source_key.to_owned(),
            memory_id: None,
            condition: EvidenceCondition::Equals {
                value: EvidenceValue::Bool(expected),
            },
        }
    }

    fn plan_command(id: &str, requirements: Vec<Requirement>) -> Command {
        Command::Plan {
            id: id.to_owned(),
            description: format!("Plan {id}"),
            priority: Priority::High,
            requirements,
            min_attention_interval_seconds: 3_600,
            conflict_keys: vec![],
        }
    }

    fn observe_bool(event_id: &str, source_key: &str, revision: u64, value: bool) -> Command {
        Command::Observe {
            event_id: event_id.to_owned(),
            source_key: source_key.to_owned(),
            source_revision: revision,
            value: Some(EvidenceValue::Bool(value)),
            observed_events: vec![],
            covered_events: vec![],
            coverage_start: None,
            coverage_end: None,
        }
    }

    fn lifecycle(graph: &IntentionGraph, id: &str) -> PlanLifecycle {
        graph.plans.get(id).unwrap().lifecycle
    }

    #[test]
    fn coverage_grace_at_timestamp_limit_remains_unknown() {
        let end = DateTime::<Utc>::MAX_UTC;
        let start = end - Duration::hours(1);
        let source = EvidenceSource {
            source_key: "calendar".into(),
            revision: 1,
            value: None,
            observed_events: BTreeSet::new(),
            covered_events: ["arrival".into()].into_iter().collect(),
            coverage_start: Some(start),
            coverage_end: Some(end),
            observed_at: end,
            provenance: EvidenceProvenance::CallerAsserted,
        };
        assert!(!event_coverage_complete(
            &source,
            "arrival",
            Some(start),
            Some(end),
            1,
            end
        ));
        assert!(event_coverage_complete(
            &source,
            "arrival",
            Some(start),
            Some(end),
            0,
            end
        ));
    }

    #[test]
    fn oversized_affected_response_fails_atomically_without_poisoning_graph() {
        let mut graph = IntentionGraph::default();
        for index in 0..64 {
            let requirements = (0..32)
                .map(|r| evidence_requirement(&format!("r{r}"), "shared", true))
                .collect();
            graph
                .apply(plan_command(&format!("p{index}"), requirements), t(0))
                .unwrap();
        }
        let before = serde_json::to_value(&graph).unwrap();
        let error = graph
            .apply(observe_bool("large", "shared", 1, true), t(1))
            .unwrap_err();
        assert!(error.contains("response"), "{error}");
        assert_eq!(serde_json::to_value(&graph).unwrap(), before);
        graph.validate_state().unwrap();
        assert!(graph.apply(Command::Portfolio, t(1)).is_ok());
    }

    #[test]
    fn premise_reversal_disputes_evidence_completion() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                plan_command("ship", vec![evidence_requirement("tests", "ci", true)]),
                t(0),
            )
            .unwrap();
        graph
            .apply(observe_bool("ci-1", "ci", 1, true), t(1))
            .unwrap();
        graph
            .apply(
                Command::Complete {
                    id: "ship".to_owned(),
                    expected_version: 1,
                    basis: CompletionBasis::Evidence {
                        requirement_ids: vec!["tests".to_owned()],
                    },
                },
                t(2),
            )
            .unwrap();
        assert_eq!(lifecycle(&graph, "ship"), PlanLifecycle::Fulfilled);

        graph
            .apply(observe_bool("ci-2", "ci", 2, false), t(3))
            .unwrap();
        assert_eq!(lifecycle(&graph, "ship"), PlanLifecycle::Disputed);
        assert!(
            graph
                .plans
                .get("ship")
                .unwrap()
                .completion
                .as_ref()
                .unwrap()
                .dispute_reason
                .as_deref()
                .unwrap()
                .contains("tests")
        );
    }

    #[test]
    fn unrelated_observation_does_not_reevaluate_plan() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                plan_command("ship", vec![evidence_requirement("tests", "ci", true)]),
                t(0),
            )
            .unwrap();
        let before = graph.evaluations.len();
        let response = graph
            .apply(observe_bool("weather-1", "weather", 1, true), t(1))
            .unwrap();
        assert_eq!(response["affected_plan_ids"], json!([]));
        assert_eq!(graph.evaluations.len(), before);
    }

    #[test]
    fn stale_revision_and_conflicting_event_reuse_are_rejected_atomically() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(observe_bool("ci-1", "ci", 2, true), t(0))
            .unwrap();
        let snapshot = serde_json::to_value(&graph).unwrap();

        let stale = graph.apply(observe_bool("ci-stale", "ci", 1, false), t(1));
        assert!(stale.unwrap_err().contains("stale source revision"));
        assert_eq!(serde_json::to_value(&graph).unwrap(), snapshot);

        let conflicting = graph.apply(observe_bool("ci-1", "ci", 3, false), t(2));
        assert!(conflicting.unwrap_err().contains("different observation"));
        assert_eq!(serde_json::to_value(&graph).unwrap(), snapshot);
    }

    #[test]
    fn exact_observation_replay_is_idempotent_and_attention_is_deduplicated() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                plan_command("ship", vec![evidence_requirement("tests", "ci", true)]),
                t(0),
            )
            .unwrap();
        let command = observe_bool("ci-1", "ci", 1, false);
        graph.apply(command.clone(), t(1)).unwrap();
        let receipts = graph.observation_receipts.len();
        let queue = graph.attention_queue.len();
        let response = graph.apply(command, t(4)).unwrap();
        assert_eq!(response["idempotent"], true);
        assert_eq!(graph.observation_receipts.len(), receipts);
        assert_eq!(graph.attention_queue.len(), queue);

        graph
            .apply(observe_bool("ci-2", "ci", 2, false), t(5))
            .unwrap();
        assert_eq!(
            graph.attention_queue.len(),
            queue,
            "a newer source revision with the same logical result must not flood attention"
        );
    }

    #[test]
    fn offline_event_is_unknown_and_covered_absence_is_missing() {
        let start = t(0);
        let end = t(1);
        let requirement = Requirement::Event {
            id: "deploy_seen".to_owned(),
            source_key: "audit".to_owned(),
            memory_id: None,
            event: "deployed".to_owned(),
            expectation: EventExpectation::Occurred,
            window_start: Some(start),
            window_end: Some(end),
            grace_seconds: 60,
        };
        let mut graph = IntentionGraph::default();
        graph
            .apply(plan_command("release", vec![requirement]), t(0))
            .unwrap();
        let offline = graph
            .apply(
                Command::Explain {
                    id: "release".to_owned(),
                },
                t(2),
            )
            .unwrap();
        assert_eq!(offline["evaluation"]["requirements"][0]["state"], "unknown");

        graph
            .apply(
                Command::Observe {
                    event_id: "audit-1".to_owned(),
                    source_key: "audit".to_owned(),
                    source_revision: 1,
                    value: None,
                    observed_events: vec![],
                    covered_events: vec!["deployed".to_owned()],
                    coverage_start: Some(start),
                    coverage_end: Some(end),
                },
                t(2),
            )
            .unwrap();
        let covered = graph
            .apply(
                Command::Explain {
                    id: "release".to_owned(),
                },
                t(2),
            )
            .unwrap();
        assert_eq!(covered["evaluation"]["requirements"][0]["state"], "missing");
    }

    #[test]
    fn absence_requires_elapsed_explicit_window_and_complete_coverage() {
        let requirement = Requirement::Event {
            id: "no_failure".to_owned(),
            source_key: "audit".to_owned(),
            memory_id: None,
            event: "failed".to_owned(),
            expectation: EventExpectation::Absent,
            window_start: Some(t(0)),
            window_end: Some(t(2)),
            grace_seconds: 3_600,
        };
        let mut graph = IntentionGraph::default();
        graph
            .apply(plan_command("stable", vec![requirement]), t(0))
            .unwrap();
        graph
            .apply(
                Command::Observe {
                    event_id: "audit-1".to_owned(),
                    source_key: "audit".to_owned(),
                    source_revision: 1,
                    value: None,
                    observed_events: vec![],
                    covered_events: vec!["failed".to_owned()],
                    coverage_start: Some(t(0)),
                    coverage_end: Some(t(2)),
                },
                t(2),
            )
            .unwrap();
        let before_grace = graph
            .apply(
                Command::Explain {
                    id: "stable".to_owned(),
                },
                t(2),
            )
            .unwrap();
        assert_eq!(
            before_grace["evaluation"]["requirements"][0]["state"],
            "unknown"
        );
        let after_grace = graph
            .apply(
                Command::Explain {
                    id: "stable".to_owned(),
                },
                t(3),
            )
            .unwrap();
        assert_eq!(
            after_grace["evaluation"]["requirements"][0]["state"],
            "satisfied"
        );
    }

    #[test]
    fn user_completion_and_cancellation_are_explicit_lifecycle_states() {
        let mut graph = IntentionGraph::default();
        graph.apply(plan_command("a", vec![]), t(0)).unwrap();
        graph
            .apply(
                Command::Complete {
                    id: "a".to_owned(),
                    expected_version: 1,
                    basis: CompletionBasis::UserReport {
                        report: "User confirmed the work is complete".to_owned(),
                    },
                },
                t(1),
            )
            .unwrap();
        assert_eq!(lifecycle(&graph, "a"), PlanLifecycle::Fulfilled);

        graph.apply(plan_command("b", vec![]), t(1)).unwrap();
        graph
            .apply(
                Command::Cancel {
                    id: "b".to_owned(),
                    expected_version: 1,
                    reason: "No longer wanted".to_owned(),
                },
                t(2),
            )
            .unwrap();
        assert_eq!(lifecycle(&graph, "b"), PlanLifecycle::Cancelled);
    }

    #[test]
    fn evidence_completion_requires_every_requirement_and_exact_retry_is_noop() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                plan_command(
                    "ship",
                    vec![
                        evidence_requirement("tests", "ci", true),
                        evidence_requirement("approval", "review", true),
                    ],
                ),
                t(0),
            )
            .unwrap();
        graph
            .apply(observe_bool("ci-1", "ci", 1, true), t(1))
            .unwrap();
        graph
            .apply(observe_bool("review-1", "review", 1, false), t(2))
            .unwrap();
        let rejected = graph.apply(
            Command::Complete {
                id: "ship".to_owned(),
                expected_version: 1,
                basis: CompletionBasis::Evidence {
                    requirement_ids: vec!["tests".to_owned()],
                },
            },
            t(3),
        );
        assert!(rejected.unwrap_err().contains("all current requirements"));

        graph
            .apply(observe_bool("review-2", "review", 2, true), t(3))
            .unwrap();
        let complete = Command::Complete {
            id: "ship".to_owned(),
            expected_version: 1,
            basis: CompletionBasis::Evidence {
                requirement_ids: vec!["tests".to_owned()],
            },
        };
        graph.apply(complete.clone(), t(4)).unwrap();
        let frozen = serde_json::to_value(&graph).unwrap();
        let retry = graph.apply(complete, t(5)).unwrap();
        assert_eq!(retry["idempotent"], true);
        assert_eq!(serde_json::to_value(&graph).unwrap(), frozen);

        graph
            .apply(observe_bool("review-3", "review", 3, false), t(6))
            .unwrap();
        assert_eq!(lifecycle(&graph, "ship"), PlanLifecycle::Disputed);
    }

    #[test]
    fn identical_cancel_retry_preserves_original_cancellation() {
        let mut graph = IntentionGraph::default();
        graph.apply(plan_command("stop", vec![]), t(0)).unwrap();
        let cancel = Command::Cancel {
            id: "stop".to_owned(),
            expected_version: 1,
            reason: "User cancelled".to_owned(),
        };
        graph.apply(cancel.clone(), t(1)).unwrap();
        let frozen = serde_json::to_value(&graph).unwrap();
        let retry = graph.apply(cancel, t(2)).unwrap();
        assert_eq!(retry["idempotent"], true);
        assert_eq!(serde_json::to_value(&graph).unwrap(), frozen);

        let conflicting = graph.apply(
            Command::Cancel {
                id: "stop".to_owned(),
                expected_version: 1,
                reason: "Different reason".to_owned(),
            },
            t(3),
        );
        assert!(conflicting.unwrap_err().contains("different reason"));
        assert_eq!(serde_json::to_value(&graph).unwrap(), frozen);
    }

    #[test]
    fn attention_withholding_retains_pending_event() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                plan_command("ship", vec![evidence_requirement("tests", "ci", true)]),
                t(0),
            )
            .unwrap();
        let first_id = graph.attention_queue.keys().next().cloned().unwrap();
        graph
            .apply(Command::Acknowledge { queue_id: first_id }, t(1))
            .unwrap();
        let response = graph
            .apply(observe_bool("ci-1", "ci", 1, false), t(1))
            .unwrap();
        let withheld = response["attention"]["withheld_queue_ids"]
            .as_array()
            .unwrap();
        assert_eq!(withheld.len(), 1);
        let withheld_id = withheld[0].as_str().unwrap();
        assert!(graph.attention_queue.contains_key(withheld_id));

        graph
            .apply(
                Command::Evaluate {
                    id: Some("ship".to_owned()),
                },
                t(1),
            )
            .unwrap();
        assert_eq!(
            graph
                .attention_queue
                .values()
                .filter(|event| event.acknowledged_at.is_none())
                .count(),
            1
        );
    }

    #[test]
    fn attention_requeues_a_returned_state_but_dedupes_unchanged_revisions() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                plan_command("ship", vec![evidence_requirement("tests", "ci", true)]),
                t(0),
            )
            .unwrap();
        graph
            .apply(observe_bool("ci-1", "ci", 1, false), t(1))
            .unwrap();
        let first_unmet = graph
            .attention_queue
            .values()
            .find(|event| event.superseded_at.is_none())
            .unwrap()
            .clone();
        graph
            .apply(observe_bool("ci-2", "ci", 2, true), t(2))
            .unwrap();
        graph
            .apply(observe_bool("ci-3", "ci", 3, false), t(3))
            .unwrap();
        let returned_unmet = graph
            .attention_queue
            .values()
            .find(|event| event.superseded_at.is_none())
            .unwrap()
            .clone();
        assert_eq!(returned_unmet.fingerprint, first_unmet.fingerprint);
        assert_ne!(returned_unmet.queue_id, first_unmet.queue_id);

        let count = graph.attention_queue.len();
        graph
            .apply(observe_bool("ci-4", "ci", 4, false), t(4))
            .unwrap();
        assert_eq!(graph.attention_queue.len(), count);
    }

    #[test]
    fn portfolio_reports_shared_requirements_prerequisites_and_conflicts() {
        let shared_a = evidence_requirement("shared-a", "budget", true);
        let shared_b = evidence_requirement("shared-b", "budget", true);
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                Command::Plan {
                    id: "foundation".to_owned(),
                    description: "Foundation".to_owned(),
                    priority: Priority::Critical,
                    requirements: vec![shared_a],
                    min_attention_interval_seconds: 0,
                    conflict_keys: vec!["exclusive:gpu".to_owned()],
                },
                t(0),
            )
            .unwrap();
        graph
            .apply(
                Command::Plan {
                    id: "dependent".to_owned(),
                    description: "Dependent".to_owned(),
                    priority: Priority::Normal,
                    requirements: vec![
                        shared_b,
                        Requirement::PlanFulfilled {
                            id: "foundation_done".to_owned(),
                            plan_id: "foundation".to_owned(),
                        },
                    ],
                    min_attention_interval_seconds: 0,
                    conflict_keys: vec!["exclusive:gpu".to_owned()],
                },
                t(0),
            )
            .unwrap();
        let portfolio = graph.apply(Command::Portfolio, t(1)).unwrap();
        assert_eq!(
            portfolio["shared_requirements"].as_array().unwrap().len(),
            1
        );
        assert_eq!(portfolio["conflicts"].as_array().unwrap().len(), 1);
        assert_eq!(
            portfolio["unmet_prerequisites"].as_array().unwrap().len(),
            1
        );
        assert_eq!(portfolio["plans"][0]["id"], "foundation");
    }

    #[test]
    fn replay_is_deterministic_and_preserves_original_description() {
        let entries = vec![
            TimedCommand {
                command: plan_command("ship", vec![]),
                now: t(0),
            },
            TimedCommand {
                command: Command::Revise {
                    id: "ship".to_owned(),
                    expected_version: 1,
                    description: Some("Revised ship plan".to_owned()),
                    priority: None,
                    requirements: None,
                    min_attention_interval_seconds: None,
                    conflict_keys: None,
                },
                now: t(1),
            },
        ];
        let first = replay(&entries).unwrap();
        let second = replay(&entries).unwrap();
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
        let plan = first.plans.get("ship").unwrap();
        assert_eq!(plan.original_description, "Plan ship");
        assert_eq!(plan.current().description, "Revised ship plan");
        assert_eq!(plan.versions.len(), 2);
    }

    #[test]
    fn reverse_lexical_dependency_chain_is_disputed_prerequisite_first() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                plan_command(
                    "z-source",
                    vec![evidence_requirement("source_ok", "source", true)],
                ),
                t(0),
            )
            .unwrap();
        graph
            .apply(observe_bool("source-1", "source", 1, true), t(1))
            .unwrap();
        graph
            .apply(
                Command::Complete {
                    id: "z-source".to_owned(),
                    expected_version: 1,
                    basis: CompletionBasis::Evidence {
                        requirement_ids: vec!["source_ok".to_owned()],
                    },
                },
                t(2),
            )
            .unwrap();
        graph
            .apply(
                plan_command(
                    "a-dependent",
                    vec![Requirement::PlanFulfilled {
                        id: "source_done".to_owned(),
                        plan_id: "z-source".to_owned(),
                    }],
                ),
                t(3),
            )
            .unwrap();
        graph
            .apply(
                Command::Complete {
                    id: "a-dependent".to_owned(),
                    expected_version: 1,
                    basis: CompletionBasis::Evidence {
                        requirement_ids: vec!["source_done".to_owned()],
                    },
                },
                t(4),
            )
            .unwrap();

        let response = graph
            .apply(observe_bool("source-2", "source", 2, false), t(5))
            .unwrap();
        assert_eq!(
            response["affected_plan_ids"],
            json!(["a-dependent", "z-source"])
        );
        assert_eq!(lifecycle(&graph, "z-source"), PlanLifecycle::Disputed);
        assert_eq!(lifecycle(&graph, "a-dependent"), PlanLifecycle::Disputed);
        let labels: Vec<&str> = response["evaluations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|record| record["snapshot"]["plan_id"].as_str().unwrap())
            .collect();
        assert_eq!(labels, vec!["z-source", "a-dependent"]);
    }

    #[test]
    fn prerequisite_references_and_cycles_are_rejected_atomically() {
        let mut graph = IntentionGraph::default();
        let unknown = graph.apply(
            plan_command(
                "a",
                vec![Requirement::PlanFulfilled {
                    id: "b_done".to_owned(),
                    plan_id: "b".to_owned(),
                }],
            ),
            t(0),
        );
        assert!(unknown.unwrap_err().contains("unknown prerequisite"));
        assert!(graph.plans.is_empty());

        graph.apply(plan_command("a", vec![]), t(0)).unwrap();
        graph
            .apply(
                plan_command(
                    "b",
                    vec![Requirement::PlanFulfilled {
                        id: "a_done".to_owned(),
                        plan_id: "a".to_owned(),
                    }],
                ),
                t(1),
            )
            .unwrap();
        let before = serde_json::to_value(&graph).unwrap();
        let cycle = graph.apply(
            Command::Revise {
                id: "a".to_owned(),
                expected_version: 1,
                description: None,
                priority: None,
                requirements: Some(vec![Requirement::PlanFulfilled {
                    id: "b_done".to_owned(),
                    plan_id: "b".to_owned(),
                }]),
                min_attention_interval_seconds: None,
                conflict_keys: None,
            },
            t(2),
        );
        assert!(cycle.unwrap_err().contains("prerequisite cycle"));
        assert_eq!(serde_json::to_value(&graph).unwrap(), before);
    }

    #[test]
    fn memory_id_requires_exact_memory_source_key() {
        let mut graph = IntentionGraph::default();
        let bad = graph.apply(
            plan_command(
                "remember",
                vec![Requirement::Evidence {
                    id: "memory_current".to_owned(),
                    source_key: "memory:other".to_owned(),
                    memory_id: Some("abc".to_owned()),
                    condition: EvidenceCondition::Exists,
                }],
            ),
            t(0),
        );
        assert!(
            bad.unwrap_err()
                .contains("must use source_key 'memory:abc'")
        );
        assert!(graph.plans.is_empty());
    }

    #[test]
    fn unsupported_versions_and_out_of_order_mutations_are_rejected() {
        let mut incompatible = IntentionGraph {
            algorithm_version: "future-algorithm".to_owned(),
            ..IntentionGraph::default()
        };
        let error = incompatible.apply(Command::Portfolio, t(0)).unwrap_err();
        assert!(error.contains("unsupported intention graph algorithm version"));

        let mut graph = IntentionGraph::default();
        graph.apply(plan_command("a", vec![]), t(2)).unwrap();
        let before = serde_json::to_value(&graph).unwrap();
        let error = graph
            .apply(
                Command::Evaluate {
                    id: Some("a".to_owned()),
                },
                t(1),
            )
            .unwrap_err();
        assert!(error.contains("out-of-order mutation timestamp"));
        assert_eq!(serde_json::to_value(&graph).unwrap(), before);
    }

    #[test]
    fn plan_priority_defaults_to_normal_during_deserialization() {
        let command: Command = serde_json::from_value(json!({
            "action": "plan",
            "id": "p",
            "description": "Default priority"
        }))
        .unwrap();
        let Command::Plan { priority, .. } = command else {
            panic!("expected plan command")
        };
        assert_eq!(priority, Priority::Normal);
    }

    #[test]
    fn corrupted_empty_version_state_returns_error_instead_of_panicking() {
        let mut graph = IntentionGraph::default();
        graph.apply(plan_command("p", vec![]), t(0)).unwrap();
        graph.plans.get_mut("p").unwrap().versions.clear();
        let error = graph.apply(Command::Portfolio, t(1)).unwrap_err();
        assert!(error.contains("has no versions"));
    }

    #[test]
    fn future_coverage_is_rejected_and_windowed_occurrence_must_be_inside_window() {
        let mut graph = IntentionGraph::default();
        graph
            .apply(
                plan_command(
                    "watch",
                    vec![Requirement::Event {
                        id: "event_seen".to_owned(),
                        source_key: "audit".to_owned(),
                        memory_id: None,
                        event: "done".to_owned(),
                        expectation: EventExpectation::Occurred,
                        window_start: Some(t(0)),
                        window_end: Some(t(3)),
                        grace_seconds: 0,
                    }],
                ),
                t(0),
            )
            .unwrap();
        let frozen = serde_json::to_value(&graph).unwrap();
        let future = graph.apply(
            Command::Observe {
                event_id: "future".to_owned(),
                source_key: "audit".to_owned(),
                source_revision: 1,
                value: None,
                observed_events: vec!["done".to_owned()],
                covered_events: vec!["done".to_owned()],
                coverage_start: Some(t(0)),
                coverage_end: Some(t(2)),
            },
            t(1),
        );
        assert!(future.unwrap_err().contains("coverage_end"));
        assert_eq!(serde_json::to_value(&graph).unwrap(), frozen);

        graph
            .apply(
                Command::Observe {
                    event_id: "outside".to_owned(),
                    source_key: "audit".to_owned(),
                    source_revision: 1,
                    value: None,
                    observed_events: vec!["done".to_owned()],
                    covered_events: vec!["done".to_owned()],
                    coverage_start: Some(t(3)),
                    coverage_end: Some(t(4)),
                },
                t(4),
            )
            .unwrap();
        let outside = graph
            .apply(
                Command::Explain {
                    id: "watch".to_owned(),
                },
                t(4),
            )
            .unwrap();
        assert_eq!(outside["evaluation"]["requirements"][0]["state"], "unknown");

        graph
            .apply(
                Command::Observe {
                    event_id: "inside".to_owned(),
                    source_key: "audit".to_owned(),
                    source_revision: 2,
                    value: None,
                    observed_events: vec!["done".to_owned()],
                    covered_events: vec!["done".to_owned()],
                    coverage_start: Some(t(1)),
                    coverage_end: Some(t(2)),
                },
                t(5),
            )
            .unwrap();
        let inside = graph
            .apply(
                Command::Explain {
                    id: "watch".to_owned(),
                },
                t(5),
            )
            .unwrap();
        assert_eq!(
            inside["evaluation"]["requirements"][0]["state"],
            "satisfied"
        );
    }

    #[test]
    fn revising_a_fulfilled_plan_reopens_it_and_invalidates_prior_completion() {
        let mut graph = IntentionGraph::default();
        graph.apply(plan_command("p", vec![]), t(0)).unwrap();
        graph
            .apply(
                Command::Complete {
                    id: "p".to_owned(),
                    expected_version: 1,
                    basis: CompletionBasis::UserReport {
                        report: "Done".to_owned(),
                    },
                },
                t(1),
            )
            .unwrap();
        graph
            .apply(
                Command::Revise {
                    id: "p".to_owned(),
                    expected_version: 1,
                    description: Some("Reopened plan".to_owned()),
                    priority: None,
                    requirements: None,
                    min_attention_interval_seconds: None,
                    conflict_keys: None,
                },
                t(2),
            )
            .unwrap();
        let plan = graph.plans.get("p").unwrap();
        assert_eq!(plan.lifecycle, PlanLifecycle::Active);
        assert_eq!(plan.current().version, 2);
        assert!(plan.completion.as_ref().unwrap().disputed_at.is_some());
    }

    #[test]
    fn schema_exposes_all_commands_and_evidence_boundary() {
        let schema = schema();
        let serialized = serde_json::to_string(&schema).unwrap();
        for action in [
            "plan",
            "revise",
            "observe",
            "evaluate",
            "explain",
            "portfolio",
            "complete",
            "cancel",
            "acknowledge",
        ] {
            assert!(serialized.contains(&format!("\"{action}\"")));
        }
        assert!(serialized.contains("not cryptographic proof or authorization"));
    }
}
