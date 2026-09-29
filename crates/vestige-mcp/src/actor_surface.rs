//! Actor provenance response surfaces (#252 Phase A).
//!
//! One place builds the wire shapes for actor identity, role resolution, and
//! endorsement events so every mutating tool reports provenance the same way.
//! The rule these surfaces must never violate: provenance records who made or
//! endorsed a claim; it never establishes that the claim is true, and a
//! claimed role never overrides process identity or self-grants authority.

use serde_json::{Value, json};
use std::sync::Arc;
use vestige_core::actor::RoleResolution;
use vestige_core::storage::{ActorMutationOutcome, EndorsementEventRecord, Storage};

/// The `actor` block from a fresh role resolution. Field-for-field identical
/// to [`actor_block`] and to the persisted receipt's `actor` provenance.
pub fn actor_block_from_resolution(did: &str, resolution: &RoleResolution) -> Value {
    json!({
        "id": did,
        "claimedRole": resolution.claimed_role,
        "effectiveRole": resolution.effective_role,
        "resolvedWeight": resolution.resolved_weight,
        "resolutionDisposition": resolution.disposition.as_str(),
        "policyVersion": resolution.policy_version,
    })
}

/// Resolve the call's claimed role and attach the `actor` block to a
/// mutating response. Silent when no process actor is bound: surfaces never
/// claim an identity that is not bound to the store.
pub fn attach_actor_block(
    response: &mut Value,
    storage: &Arc<Storage>,
    claimed_role: Option<&str>,
) {
    if let Ok((did, resolution)) = storage.resolve_actor_role(claimed_role)
        && let Some(obj) = response.as_object_mut()
    {
        obj.insert(
            "actor".to_string(),
            actor_block_from_resolution(&did, &resolution),
        );
    }
}

/// The `actor` block from a recorded endorsement event.
pub fn actor_block(record: &EndorsementEventRecord) -> Value {
    json!({
        "id": record.actor_did,
        "claimedRole": record.claimed_role,
        "effectiveRole": record.effective_role,
        "resolvedWeight": record.resolved_weight,
        "resolutionDisposition": record.resolution_disposition,
        "policyVersion": record.policy_version,
    })
}

/// The `endorsement` block: the actor-attributed evidence recorded beside the
/// mutation, with its revision binding and independent-prior contribution.
pub fn endorsement_block(outcome: &ActorMutationOutcome) -> Value {
    let record = &outcome.endorsement;
    json!({
        "eventId": record.event_id,
        "kind": record.endorsement_kind,
        "revisionDigest": record.revision_digest,
        "independentPrior": record.independent_prior,
        "alreadyRecorded": outcome.already_recorded,
        "receiptId": record.receipt_id,
        "claimBoundary": "Records who endorsed this exact content revision. It never establishes the claim is true, and self-support contributes zero independent prior.",
    })
}
