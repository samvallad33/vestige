use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Serialize;

use crate::common::{EngramProjection, MemoryMap, RealityPolicy, hash_json};
use crate::leg3_reality_index::CompiledEligibleSet;

#[derive(Clone, Debug, PartialEq)]
pub enum ClaimSemantics {
    Exclusive,
    Set,
    Numeric { tolerance: f64 },
    Version,
    Freeform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimIntent {
    Assertion,
    Correction,
    Transition,
    PlannedTransition,
}

impl ClaimIntent {
    const fn key(self) -> &'static str {
        match self {
            Self::Assertion => "assertion",
            Self::Correction => "correction",
            Self::Transition => "transition",
            Self::PlannedTransition => "planned_transition",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriftKind {
    None,
    Reinforcement,
    Additive,
    Contradiction,
    Successor,
    ScheduledSuccessor,
    NumericDrift,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    RepairWorldline,
    QuarantineClaim,
}

#[derive(Clone, Debug)]
pub struct ClaimAssertion {
    pub memory_id: String,
    pub subject: String,
    pub predicate: String,
    pub value: String,
    pub confidence: f64,
    pub intent: ClaimIntent,
}

impl ClaimAssertion {
    pub fn claim_key(&self) -> String {
        format!(
            "{}|{}",
            self.subject.trim().to_ascii_lowercase(),
            self.predicate.trim().to_ascii_lowercase()
        )
    }
}

#[derive(Clone, Debug)]
pub struct ClaimMemoryPr {
    pub pr_id: String,
    pub kind: ProposalKind,
    pub drift_kind: DriftKind,
    pub policy_fingerprint: String,
    pub reality_checkpoint: String,
    pub claim_key: String,
    pub old_memory_id: String,
    pub new_memory_id: String,
    pub old_revision: String,
    pub new_revision: String,
    pub proposed_valid_until_ms: Option<i64>,
    pub evidence: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct ClaimAssessment {
    pub drift_kind: DriftKind,
    pub claim_key: String,
    pub compared_memory_ids: Vec<String>,
    pub proposal: Option<ClaimMemoryPr>,
}

#[derive(Default)]
pub struct ClaimSchemaRegistry {
    schemas: HashMap<String, ClaimSemantics>,
}

impl ClaimSchemaRegistry {
    pub fn define(&mut self, predicate: &str, semantics: ClaimSemantics) {
        self.schemas
            .insert(predicate.trim().to_ascii_lowercase(), semantics);
    }

    pub fn get(&self, predicate: &str) -> ClaimSemantics {
        self.schemas
            .get(&predicate.trim().to_ascii_lowercase())
            .cloned()
            .unwrap_or(ClaimSemantics::Freeform)
    }
}

#[derive(Default)]
pub struct ClaimLattice {
    pub assertions: HashMap<String, ClaimAssertion>,
    pub claim_members: HashMap<String, BTreeSet<String>>,
    pub schemas: ClaimSchemaRegistry,
}

impl ClaimLattice {
    pub fn register(
        &mut self,
        assertion: ClaimAssertion,
        memories: &MemoryMap,
    ) -> Result<(), String> {
        let memory = memories.get(&assertion.memory_id).ok_or("memory missing")?;
        let key = assertion.claim_key();
        if memory
            .claim_key
            .as_deref()
            .is_some_and(|stored| stored != key)
        {
            return Err("Engram claim_key disagrees with assertion".into());
        }

        if let Some(prior) = self.assertions.get(&assertion.memory_id) {
            let prior_key = prior.claim_key();
            let remove_bucket = if let Some(members) = self.claim_members.get_mut(&prior_key) {
                members.remove(&assertion.memory_id);
                members.is_empty()
            } else {
                false
            };
            if remove_bucket {
                self.claim_members.remove(&prior_key);
            }
        }

        self.claim_members
            .entry(key)
            .or_default()
            .insert(assertion.memory_id.clone());
        self.assertions
            .insert(assertion.memory_id.clone(), assertion);
        Ok(())
    }

    pub fn remove_assertion(&mut self, memory_id: &str) {
        let Some(assertion) = self.assertions.remove(memory_id) else {
            return;
        };
        let key = assertion.claim_key();
        let remove_bucket = if let Some(members) = self.claim_members.get_mut(&key) {
            members.remove(memory_id);
            members.is_empty()
        } else {
            false
        };
        if remove_bucket {
            self.claim_members.remove(&key);
        }
    }

    fn parse_version(value: &str) -> Vec<u64> {
        value
            .split(|character: char| !character.is_ascii_digit())
            .filter(|part| !part.is_empty())
            .filter_map(|part| part.parse().ok())
            .collect()
    }

    fn relationship(
        old: &ClaimAssertion,
        new: &ClaimAssertion,
        semantics: &ClaimSemantics,
    ) -> DriftKind {
        if old.value.trim().eq_ignore_ascii_case(new.value.trim()) {
            return DriftKind::Reinforcement;
        }
        match semantics {
            ClaimSemantics::Set => DriftKind::Additive,
            ClaimSemantics::Numeric { tolerance } => {
                match (
                    old.value.trim().parse::<f64>(),
                    new.value.trim().parse::<f64>(),
                ) {
                    (Ok(left), Ok(right)) if (left - right).abs() <= *tolerance => {
                        DriftKind::Reinforcement
                    }
                    _ => DriftKind::NumericDrift,
                }
            }
            ClaimSemantics::Version => {
                let old_version = Self::parse_version(&old.value);
                let new_version = Self::parse_version(&new.value);
                if !old_version.is_empty() && new_version > old_version {
                    DriftKind::Successor
                } else {
                    DriftKind::Contradiction
                }
            }
            ClaimSemantics::Exclusive => DriftKind::Contradiction,
            ClaimSemantics::Freeform => DriftKind::None,
        }
    }

    fn governance_revision(memory: &EngramProjection) -> String {
        let mut normalized = memory.clone();
        normalized.taints.remove("quarantined");
        hash_json("claimrev_", &normalized)
    }

    pub fn assess(
        &self,
        new_memory: &EngramProjection,
        new_claim: &ClaimAssertion,
        policy: &RealityPolicy,
        eligible: &CompiledEligibleSet,
        memories: &MemoryMap,
    ) -> Result<ClaimAssessment, String> {
        if new_memory.memory_id != new_claim.memory_id {
            return Err("claim memory mismatch".into());
        }
        let claim_key = new_claim.claim_key();
        if new_memory
            .claim_key
            .as_deref()
            .is_some_and(|stored| stored != claim_key)
        {
            return Err("new Engram claim_key mismatch".into());
        }

        let collision_ids = self
            .claim_members
            .get(&claim_key)
            .cloned()
            .unwrap_or_default();
        let mut candidates: Vec<&ClaimAssertion> = collision_ids
            .intersection(&eligible.memory_ids)
            .filter_map(|memory_id| self.assertions.get(memory_id))
            .collect();
        if candidates.is_empty() {
            return Ok(ClaimAssessment {
                drift_kind: DriftKind::None,
                claim_key,
                compared_memory_ids: Vec::new(),
                proposal: None,
            });
        }

        candidates.sort_by(|left, right| {
            let left_memory = &memories[&left.memory_id];
            let right_memory = &memories[&right.memory_id];
            right_memory
                .authority
                .cmp(&left_memory.authority)
                .then_with(|| left.memory_id.cmp(&right.memory_id))
        });
        let old_claim = candidates[0];
        let old_memory = &memories[&old_claim.memory_id];
        let semantics = self.schemas.get(&new_claim.predicate);
        let relationship = Self::relationship(old_claim, new_claim, &semantics);
        let compared_memory_ids = candidates
            .iter()
            .map(|claim| claim.memory_id.clone())
            .collect();

        if matches!(
            relationship,
            DriftKind::None | DriftKind::Reinforcement | DriftKind::Additive
        ) {
            return Ok(ClaimAssessment {
                drift_kind: relationship,
                claim_key,
                compared_memory_ids,
                proposal: None,
            });
        }

        let new_valid_from = new_memory.valid_from_ms.unwrap_or(policy.valid_at_ms);
        if old_memory
            .valid_until_ms
            .is_some_and(|until| until <= new_valid_from)
        {
            return Ok(ClaimAssessment {
                drift_kind: DriftKind::Successor,
                claim_key,
                compared_memory_ids: vec![old_claim.memory_id.clone()],
                proposal: None,
            });
        }

        let transition_evidence = matches!(
            new_claim.intent,
            ClaimIntent::Correction | ClaimIntent::Transition | ClaimIntent::PlannedTransition
        ) || relationship == DriftKind::Successor;
        if new_claim.intent == ClaimIntent::PlannedTransition
            && new_valid_from <= policy.valid_at_ms
        {
            return Err("planned transition requires a future valid_from".into());
        }
        let scheduled = new_valid_from > policy.valid_at_ms;
        let sufficient_authority = new_memory.authority >= old_memory.authority;
        let drift_kind = if !transition_evidence || !sufficient_authority {
            DriftKind::Contradiction
        } else if scheduled {
            DriftKind::ScheduledSuccessor
        } else if relationship == DriftKind::NumericDrift {
            DriftKind::NumericDrift
        } else {
            DriftKind::Successor
        };

        let proposal_kind = if drift_kind == DriftKind::Contradiction {
            ProposalKind::QuarantineClaim
        } else {
            ProposalKind::RepairWorldline
        };
        let proposed_valid_until_ms = match proposal_kind {
            ProposalKind::RepairWorldline => Some(new_valid_from),
            ProposalKind::QuarantineClaim => None,
        };

        #[derive(Serialize)]
        struct ProposalProjection<'a> {
            claim: &'a str,
            old: &'a str,
            new: &'a str,
            old_revision: &'a str,
            new_revision: &'a str,
            until: Option<i64>,
            reality: &'a str,
            checkpoint: &'a str,
            kind: &'a str,
            intent: &'a str,
        }
        let proposal_kind_name = match proposal_kind {
            ProposalKind::RepairWorldline => "repair",
            ProposalKind::QuarantineClaim => "quarantine",
        };
        let old_revision = Self::governance_revision(old_memory);
        let new_revision = Self::governance_revision(new_memory);
        let pr_id = hash_json(
            "mpr_",
            &ProposalProjection {
                claim: &claim_key,
                old: &old_claim.memory_id,
                new: &new_claim.memory_id,
                old_revision: &old_revision,
                new_revision: &new_revision,
                until: proposed_valid_until_ms,
                reality: &policy.fingerprint,
                checkpoint: &eligible.reality_checkpoint,
                kind: proposal_kind_name,
                intent: new_claim.intent.key(),
            },
        );
        let evidence = [
            ("oldAuthority".into(), old_memory.authority.to_string()),
            ("newAuthority".into(), new_memory.authority.to_string()),
            ("semantics".into(), format!("{semantics:?}")),
            ("scheduled".into(), scheduled.to_string()),
            ("candidateCount".into(), candidates.len().to_string()),
            ("claimIntent".into(), new_claim.intent.key().into()),
        ]
        .into_iter()
        .collect();

        Ok(ClaimAssessment {
            drift_kind,
            claim_key: claim_key.clone(),
            compared_memory_ids,
            proposal: Some(ClaimMemoryPr {
                pr_id,
                kind: proposal_kind,
                drift_kind,
                policy_fingerprint: policy.fingerprint.clone(),
                reality_checkpoint: eligible.reality_checkpoint.clone(),
                claim_key,
                old_memory_id: old_claim.memory_id.clone(),
                new_memory_id: new_claim.memory_id.clone(),
                old_revision,
                new_revision,
                proposed_valid_until_ms,
                evidence,
            }),
        })
    }

    pub fn stage_write(
        &mut self,
        mut memory: EngramProjection,
        claim: ClaimAssertion,
        assessment: &ClaimAssessment,
        memories: &mut MemoryMap,
    ) -> Result<(), String> {
        if assessment.proposal.is_some() {
            memory.taints.insert("quarantined".into());
        }
        let memory_id = memory.memory_id.clone();
        let prior = memories.insert(memory_id.clone(), memory);
        if let Err(error) = self.register(claim, memories) {
            match prior {
                Some(previous) => {
                    memories.insert(memory_id, previous);
                }
                None => {
                    memories.remove(&memory_id);
                }
            }
            return Err(error);
        }
        Ok(())
    }

    pub fn apply(
        &self,
        proposal: &ClaimMemoryPr,
        confirm: bool,
        policy: &RealityPolicy,
        eligible: &CompiledEligibleSet,
        memories: &mut MemoryMap,
    ) -> Result<(), String> {
        if !confirm {
            return Err("claim governance action requires confirmation".into());
        }
        if proposal.policy_fingerprint != policy.fingerprint {
            return Err("claim PR Reality policy changed".into());
        }
        if proposal.reality_checkpoint != eligible.reality_checkpoint {
            return Err("claim PR Reality checkpoint changed".into());
        }
        let old = memories
            .get(&proposal.old_memory_id)
            .cloned()
            .ok_or("old memory missing")?;
        let new = memories
            .get(&proposal.new_memory_id)
            .cloned()
            .ok_or("new memory missing")?;
        if old.claim_key.as_deref() != Some(proposal.claim_key.as_str())
            || new.claim_key.as_deref() != Some(proposal.claim_key.as_str())
        {
            return Err("claim topology changed".into());
        }
        if Self::governance_revision(&old) != proposal.old_revision {
            return Err("old claim changed since proposal".into());
        }
        if Self::governance_revision(&new) != proposal.new_revision {
            return Err("new claim changed since proposal".into());
        }

        match proposal.kind {
            ProposalKind::QuarantineClaim => Ok(()),
            ProposalKind::RepairWorldline => {
                let until = proposal
                    .proposed_valid_until_ms
                    .ok_or("missing proposed worldline end")?;
                let mut old_updated = old;
                old_updated.valid_until_ms = Some(until);
                let mut new_updated = new;
                new_updated.taints.remove("quarantined");
                memories.insert(old_updated.memory_id.clone(), old_updated);
                memories.insert(new_updated.memory_id.clone(), new_updated);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{EngramProjection, RealityPolicy};
    use crate::leg3_reality_index::RealityIndex;

    fn policy() -> RealityPolicy {
        RealityPolicy::new(Some("A".into()), ["A".into()].into_iter().collect(), 100)
    }

    #[test]
    fn unrelated_predicate_does_not_false_alarm() {
        let mut memories = MemoryMap::new();
        let mut old = EngramProjection::context("old", "Backblaze", "A");
        old.claim_key = Some("svc|backup_destination".into());
        memories.insert("old".into(), old);

        let mut index = RealityIndex::new(10);
        index.rebuild(&memories);
        let eligible = index.compile(&memories, policy());
        let mut lattice = ClaimLattice::default();
        lattice
            .schemas
            .define("backup_destination", ClaimSemantics::Exclusive);
        lattice
            .schemas
            .define("primary_database", ClaimSemantics::Exclusive);
        lattice
            .register(
                ClaimAssertion {
                    memory_id: "old".into(),
                    subject: "svc".into(),
                    predicate: "backup_destination".into(),
                    value: "Backblaze".into(),
                    confidence: 1.0,
                    intent: ClaimIntent::Assertion,
                },
                &memories,
            )
            .expect("register old claim");

        let mut new = EngramProjection::context("new", "Postgres", "A");
        new.claim_key = Some("svc|primary_database".into());
        let claim = ClaimAssertion {
            memory_id: "new".into(),
            subject: "svc".into(),
            predicate: "primary_database".into(),
            value: "Postgres".into(),
            confidence: 1.0,
            intent: ClaimIntent::Assertion,
        };
        let assessment = lattice
            .assess(&new, &claim, &policy(), &eligible, &memories)
            .expect("assess claim");
        assert_eq!(assessment.drift_kind, DriftKind::None);
    }

    #[test]
    fn weaker_conflict_is_quarantined_not_worldline_repair() {
        let mut memories = MemoryMap::new();
        let mut old = EngramProjection::context("old", "Postgres", "A");
        old.claim_key = Some("svc|db".into());
        old.authority = 4;
        memories.insert("old".into(), old);

        let mut index = RealityIndex::new(10);
        index.rebuild(&memories);
        let eligible = index.compile(&memories, policy());
        let mut lattice = ClaimLattice::default();
        lattice.schemas.define("db", ClaimSemantics::Exclusive);
        lattice
            .register(
                ClaimAssertion {
                    memory_id: "old".into(),
                    subject: "svc".into(),
                    predicate: "db".into(),
                    value: "Postgres".into(),
                    confidence: 1.0,
                    intent: ClaimIntent::Assertion,
                },
                &memories,
            )
            .expect("register old claim");

        let mut new = EngramProjection::context("new", "SQLite", "A");
        new.claim_key = Some("svc|db".into());
        new.authority = 1;
        let claim = ClaimAssertion {
            memory_id: "new".into(),
            subject: "svc".into(),
            predicate: "db".into(),
            value: "SQLite".into(),
            confidence: 0.3,
            intent: ClaimIntent::Assertion,
        };
        let assessment = lattice
            .assess(&new, &claim, &policy(), &eligible, &memories)
            .expect("assess claim");
        let proposal = assessment.proposal.expect("quarantine proposal");
        assert_eq!(proposal.kind, ProposalKind::QuarantineClaim);
        assert!(proposal.proposed_valid_until_ms.is_none());
    }

    #[test]
    fn typed_transition_is_checkpoint_and_revision_bound() {
        let mut memories = MemoryMap::new();
        let mut old = EngramProjection::context("old", "MySQL", "A");
        old.claim_key = Some("svc|db".into());
        memories.insert("old".into(), old);

        let mut index = RealityIndex::new(1);
        index.rebuild(&memories);
        let p = policy();
        let eligible_before = index.compile(&memories, p.clone());
        let mut lattice = ClaimLattice::default();
        lattice.schemas.define("db", ClaimSemantics::Exclusive);
        lattice
            .register(
                ClaimAssertion {
                    memory_id: "old".into(),
                    subject: "svc".into(),
                    predicate: "db".into(),
                    value: "MySQL".into(),
                    confidence: 1.0,
                    intent: ClaimIntent::Assertion,
                },
                &memories,
            )
            .expect("register old claim");

        let mut new = EngramProjection::context("new", "PostgreSQL", "A");
        new.claim_key = Some("svc|db".into());
        let new_claim = ClaimAssertion {
            memory_id: "new".into(),
            subject: "svc".into(),
            predicate: "db".into(),
            value: "PostgreSQL".into(),
            confidence: 1.0,
            intent: ClaimIntent::Transition,
        };
        let assessment = lattice
            .assess(&new, &new_claim, &p, &eligible_before, &memories)
            .expect("assess transition");
        assert_eq!(assessment.drift_kind, DriftKind::Successor);
        let proposal = assessment.proposal.clone().expect("repair proposal");
        lattice
            .stage_write(new, new_claim, &assessment, &mut memories)
            .expect("stage quarantined claim");
        index.rebuild(&memories);
        let eligible_current = index.compile(&memories, p.clone());
        assert_eq!(
            proposal.reality_checkpoint,
            eligible_current.reality_checkpoint
        );
        lattice
            .apply(&proposal, true, &p, &eligible_current, &mut memories)
            .expect("apply current proposal");
        assert_eq!(memories["old"].valid_until_ms, Some(p.valid_at_ms));
        assert!(!memories["new"].taints.contains("quarantined"));
    }

    #[test]
    fn stale_quarantined_candidate_cannot_reuse_worldline_pr() {
        let mut memories = MemoryMap::new();
        let mut old = EngramProjection::context("old", "MySQL", "A");
        old.claim_key = Some("svc|db".into());
        memories.insert("old".into(), old);
        let mut index = RealityIndex::new(1);
        index.rebuild(&memories);
        let p = policy();
        let eligible_before = index.compile(&memories, p.clone());
        let mut lattice = ClaimLattice::default();
        lattice.schemas.define("db", ClaimSemantics::Exclusive);
        lattice
            .register(
                ClaimAssertion {
                    memory_id: "old".into(),
                    subject: "svc".into(),
                    predicate: "db".into(),
                    value: "MySQL".into(),
                    confidence: 1.0,
                    intent: ClaimIntent::Assertion,
                },
                &memories,
            )
            .expect("register old claim");
        let mut new = EngramProjection::context("new", "PostgreSQL", "A");
        new.claim_key = Some("svc|db".into());
        let claim = ClaimAssertion {
            memory_id: "new".into(),
            subject: "svc".into(),
            predicate: "db".into(),
            value: "PostgreSQL".into(),
            confidence: 1.0,
            intent: ClaimIntent::Transition,
        };
        let assessment = lattice
            .assess(&new, &claim, &p, &eligible_before, &memories)
            .expect("assess transition");
        let proposal = assessment.proposal.clone().expect("repair proposal");
        lattice
            .stage_write(new, claim, &assessment, &mut memories)
            .expect("stage claim");
        memories.get_mut("new").expect("staged claim").content =
            "attacker changed candidate to SQLite".into();
        index.rebuild(&memories);
        let eligible_current = index.compile(&memories, p.clone());
        let error = lattice
            .apply(&proposal, true, &p, &eligible_current, &mut memories)
            .expect_err("stale candidate must fail");
        assert_eq!(error, "new claim changed since proposal");
    }
}
