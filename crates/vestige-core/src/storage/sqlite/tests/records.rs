//! Tests for `sqlite/records.rs`: ComposedGraph persistence, dream history
//! and counters.

use super::*;

#[test]
fn test_composition_save_query_outcome_and_never_composed() {
    let storage = create_test_storage();
    let first = storage
        .ingest(IngestInput {
            content: "Oracle drift can break delayed settlement.".to_string(),
            node_type: "fact".to_string(),
            tags: vec![
                "protocolgate".to_string(),
                "boundary-oracle".to_string(),
                "settlement".to_string(),
            ],
            ..Default::default()
        })
        .unwrap();
    let second = storage
        .ingest(IngestInput {
            content: "Withdrawal queues can settle stale claims.".to_string(),
            node_type: "pattern".to_string(),
            tags: vec![
                "protocolgate".to_string(),
                "boundary-queue".to_string(),
                "settlement".to_string(),
            ],
            ..Default::default()
        })
        .unwrap();
    let third = storage
        .ingest(IngestInput {
            content: "Keeper roles can drift from local validation paths.".to_string(),
            node_type: "pattern".to_string(),
            tags: vec![
                "protocolgate".to_string(),
                "boundary-role".to_string(),
                "settlement".to_string(),
            ],
            ..Default::default()
        })
        .unwrap();

    let before = storage
        .get_never_composed_candidates(10, Some(&["protocolgate".to_string()]))
        .unwrap();
    let first_second_before = before
        .iter()
        .find(|candidate| {
            let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
            pair == Storage::pair_key(&first.id, &second.id)
        })
        .expect("uncomposed first/second pair should be ranked before any event");
    assert!(
        first_second_before.bridge_score > 0.0,
        "candidate should expose a bridge score"
    );
    assert!(
        first_second_before.novelty_score > 0.0,
        "candidate should expose a novelty score"
    );
    assert_eq!(
        first_second_before.outcome_signal, "clean",
        "new candidate should start without prior outcome context"
    );
    assert!(
        first_second_before
            .composition_question
            .contains("composed through"),
        "candidate should include a promptable composition question"
    );

    let event = CompositionEventRecord {
        id: "composition-test-1".to_string(),
        created_at: Utc::now(),
        tool: "deep_reference".to_string(),
        mode: "bounty".to_string(),
        query: Some("oracle drift delayed settlement".to_string()),
        query_hash: Some("sha256:test".to_string()),
        confidence: Some(0.87),
        status: Some("resolved".to_string()),
        output_preview: Some("Compose oracle drift with withdrawal queue.".to_string()),
        metadata: serde_json::json!({"workflow": "test"}),
    };
    let members = vec![
        CompositionMemberRecord {
            event_id: event.id.clone(),
            memory_id: first.id.clone(),
            role: "primary".to_string(),
            rank: 0,
            trust: Some(0.8),
            score: Some(0.9),
            preview: Some(preview(&first.content, 120)),
            metadata: serde_json::json!({}),
        },
        CompositionMemberRecord {
            event_id: event.id.clone(),
            memory_id: second.id.clone(),
            role: "supporting".to_string(),
            rank: 1,
            trust: Some(0.7),
            score: Some(0.75),
            preview: Some(preview(&second.content, 120)),
            metadata: serde_json::json!({}),
        },
    ];
    storage.save_composition(&event, &members, &[]).unwrap();

    let outcome = CompositionOutcomeRecord {
        id: "composition-outcome-1".to_string(),
        event_id: event.id.clone(),
        outcome_type: "submitted".to_string(),
        labeled_at: Utc::now(),
        label_source: "test".to_string(),
        confidence_delta: Some(0.1),
        notes: Some("Report submitted".to_string()),
        metadata: serde_json::json!({"severity": "high"}),
    };
    storage.record_composition_outcome(&outcome).unwrap();

    let fetched = storage.get_composition_event(&event.id).unwrap().unwrap();
    assert_eq!(fetched.mode, "bounty");
    assert_eq!(fetched.metadata["workflow"], "test");

    let fetched_members = storage.get_composition_members(&event.id).unwrap();
    assert_eq!(fetched_members.len(), 2);
    assert_eq!(fetched_members[0].role, "primary");

    let fetched_outcomes = storage.get_composition_outcomes(&event.id).unwrap();
    assert_eq!(fetched_outcomes.len(), 1);
    assert_eq!(fetched_outcomes[0].outcome_type, "submitted");

    let for_memory = storage.get_compositions_for_memory(&first.id, 5).unwrap();
    assert_eq!(for_memory.len(), 1);
    assert_eq!(for_memory[0].id, event.id);

    let neighbors = storage.get_composition_neighbors(&first.id, 5).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].memory_id, second.id);

    let after = storage
        .get_never_composed_candidates(10, Some(&["protocolgate".to_string()]))
        .unwrap();
    assert!(
        !after.iter().any(|candidate| {
            let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
            pair == Storage::pair_key(&first.id, &second.id)
        }),
        "already-composed first/second pair should be removed"
    );
    assert!(
        after.iter().any(|candidate| {
            let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
            pair == Storage::pair_key(&first.id, &third.id)
                || pair == Storage::pair_key(&second.id, &third.id)
        }),
        "other protocolgate pairs should remain candidates"
    );
}

#[test]
fn test_composition_neighbors_count_distinct_events_not_member_roles() {
    let storage = create_test_storage();
    let first = storage
        .ingest(IngestInput {
            content: "Oracle role appears once in the event.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string(), "settlement".to_string()],
            ..Default::default()
        })
        .unwrap();
    let second = storage
        .ingest(IngestInput {
            content: "Queue role appears under two evidence roles.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["protocolgate".to_string(), "settlement".to_string()],
            ..Default::default()
        })
        .unwrap();

    storage
        .save_composition(
            &CompositionEventRecord {
                id: "multi-role-neighbor-event".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "bounty".to_string(),
                query: Some("multi role neighbor".to_string()),
                query_hash: Some("fnv1a64:neighbor".to_string()),
                confidence: Some(0.7),
                status: Some("resolved".to_string()),
                output_preview: None,
                metadata: serde_json::json!({}),
            },
            &[
                CompositionMemberRecord {
                    event_id: "multi-role-neighbor-event".to_string(),
                    memory_id: first.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.8),
                    score: Some(0.9),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: "multi-role-neighbor-event".to_string(),
                    memory_id: second.id.clone(),
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.7),
                    score: Some(0.8),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: "multi-role-neighbor-event".to_string(),
                    memory_id: second.id.clone(),
                    role: "related".to_string(),
                    rank: 2,
                    trust: Some(0.7),
                    score: Some(0.6),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
            ],
            &[],
        )
        .unwrap();

    let neighbors = storage.get_composition_neighbors(&first.id, 10).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].memory_id, second.id);
    assert_eq!(
        neighbors[0].composed_count, 1,
        "one event with multiple member roles should count as one composition"
    );
}

#[test]
fn test_never_composed_tag_filter_includes_older_tagged_candidates() {
    let storage = create_test_storage();
    let first = storage
        .ingest(IngestInput {
            content: "Older Vestige composition frontier about outcome-shaped recall.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["project:vestige".to_string(), "composition".to_string()],
            ..Default::default()
        })
        .unwrap();
    let second = storage
        .ingest(IngestInput {
            content: "Older Vestige composition frontier about never-composed recall.".to_string(),
            node_type: "pattern".to_string(),
            tags: vec!["project:vestige".to_string(), "composition".to_string()],
            ..Default::default()
        })
        .unwrap();

    for idx in 0..751 {
        storage
            .ingest(IngestInput {
                content: format!("Unrelated recent memory {idx} for scan-window pressure."),
                node_type: "fact".to_string(),
                tags: vec!["unrelated".to_string()],
                ..Default::default()
            })
            .unwrap();
    }

    let candidates = storage
        .get_never_composed_candidates(10, Some(&["project".to_string()]))
        .unwrap();
    assert!(
        candidates.iter().any(|candidate| {
            let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
            pair == Storage::pair_key(&first.id, &second.id)
        }),
        "tag-filtered frontier should include older namespaced-tag memories outside the base scan window"
    );
}

#[test]
fn test_never_composed_carries_prior_outcome_signal() {
    let storage = create_test_storage();
    let first = storage
        .ingest(IngestInput {
            content: "Oracle drift lane previously looked duplicate-prone.".to_string(),
            node_type: "fact".to_string(),
            tags: vec![
                "protocolgate".to_string(),
                "boundary-oracle".to_string(),
                "settlement".to_string(),
            ],
            ..Default::default()
        })
        .unwrap();
    let second = storage
        .ingest(IngestInput {
            content: "Withdrawal queue lane had weak proof.".to_string(),
            node_type: "fact".to_string(),
            tags: vec![
                "protocolgate".to_string(),
                "boundary-queue".to_string(),
                "settlement".to_string(),
            ],
            ..Default::default()
        })
        .unwrap();
    let third = storage
        .ingest(IngestInput {
            content: "Keeper settlement lane has not been composed with oracle drift.".to_string(),
            node_type: "pattern".to_string(),
            tags: vec![
                "protocolgate".to_string(),
                "boundary-role".to_string(),
                "settlement".to_string(),
            ],
            ..Default::default()
        })
        .unwrap();

    let event = CompositionEventRecord {
        id: "prior-outcome-composition".to_string(),
        created_at: Utc::now(),
        tool: "deep_reference".to_string(),
        mode: "bounty".to_string(),
        query: Some("oracle withdrawal duplicate risk".to_string()),
        query_hash: Some("fnv1a64:prior".to_string()),
        confidence: Some(0.4),
        status: Some("closed".to_string()),
        output_preview: Some("Prior composition was labeled duplicate risk.".to_string()),
        metadata: serde_json::json!({}),
    };
    storage
        .save_composition(
            &event,
            &[
                CompositionMemberRecord {
                    event_id: event.id.clone(),
                    memory_id: first.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.7),
                    score: Some(0.8),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: event.id.clone(),
                    memory_id: second.id.clone(),
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.7),
                    score: Some(0.8),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
            ],
            &[CompositionOutcomeRecord {
                id: "prior-outcome-label".to_string(),
                event_id: event.id.clone(),
                outcome_type: "duplicate_risk".to_string(),
                labeled_at: Utc::now(),
                label_source: "test".to_string(),
                confidence_delta: Some(-0.2),
                notes: Some("Duplicate family in prior lane.".to_string()),
                metadata: serde_json::json!({}),
            }],
        )
        .unwrap();

    let candidates = storage
        .get_never_composed_candidates(10, Some(&["protocolgate".to_string()]))
        .unwrap();
    let candidate = candidates
        .iter()
        .find(|candidate| {
            let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
            pair == Storage::pair_key(&first.id, &third.id)
        })
        .expect("untried first/third pair should remain a frontier candidate");

    assert!(
        candidate
            .prior_outcomes
            .iter()
            .any(|outcome| outcome == "duplicate_risk"),
        "frontier candidate should expose prior outcome labels from either member"
    );
    assert_eq!(candidate.outcome_signal, "prior_duplicate_risk");
    assert!(
        candidate.outcome_score_adjustment < 0.0,
        "duplicate-risk history should reduce but not hide the untried lane"
    );
}

#[test]
fn test_never_composed_marks_mixed_prior_outcomes() {
    let storage = create_test_storage();
    let successful = storage
        .ingest(IngestInput {
            content: "Accepted release lane linked rollback evidence to install telemetry."
                .to_string(),
            node_type: "decision".to_string(),
            tags: vec![
                "project:vestige".to_string(),
                "release".to_string(),
                "telemetry".to_string(),
            ],
            ..Default::default()
        })
        .unwrap();
    let closed = storage
        .ingest(IngestInput {
            content: "Closed release lane linked install telemetry to out-of-scope claims."
                .to_string(),
            node_type: "incident".to_string(),
            tags: vec![
                "project:vestige".to_string(),
                "release".to_string(),
                "telemetry".to_string(),
            ],
            ..Default::default()
        })
        .unwrap();
    let success_helper = storage
        .ingest(IngestInput {
            content: "Helper memory for an accepted release composition.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["project:vestige".to_string(), "release".to_string()],
            ..Default::default()
        })
        .unwrap();
    let closed_helper = storage
        .ingest(IngestInput {
            content: "Helper memory for a closed release composition.".to_string(),
            node_type: "fact".to_string(),
            tags: vec!["project:vestige".to_string(), "release".to_string()],
            ..Default::default()
        })
        .unwrap();

    storage
        .save_composition(
            &CompositionEventRecord {
                id: "prior-success-composition".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "release".to_string(),
                query: Some("accepted release lane".to_string()),
                query_hash: Some("fnv1a64:success".to_string()),
                confidence: Some(0.9),
                status: Some("resolved".to_string()),
                output_preview: None,
                metadata: serde_json::json!({}),
            },
            &[
                CompositionMemberRecord {
                    event_id: "prior-success-composition".to_string(),
                    memory_id: successful.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.9),
                    score: Some(0.9),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: "prior-success-composition".to_string(),
                    memory_id: success_helper.id,
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.7),
                    score: Some(0.6),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
            ],
            &[CompositionOutcomeRecord {
                id: "prior-success-label".to_string(),
                event_id: "prior-success-composition".to_string(),
                outcome_type: "accepted".to_string(),
                labeled_at: Utc::now(),
                label_source: "test".to_string(),
                confidence_delta: Some(0.2),
                notes: None,
                metadata: serde_json::json!({}),
            }],
        )
        .unwrap();

    storage
        .save_composition(
            &CompositionEventRecord {
                id: "prior-closed-composition".to_string(),
                created_at: Utc::now(),
                tool: "deep_reference".to_string(),
                mode: "release".to_string(),
                query: Some("closed release lane".to_string()),
                query_hash: Some("fnv1a64:closed".to_string()),
                confidence: Some(0.3),
                status: Some("closed".to_string()),
                output_preview: None,
                metadata: serde_json::json!({}),
            },
            &[
                CompositionMemberRecord {
                    event_id: "prior-closed-composition".to_string(),
                    memory_id: closed.id.clone(),
                    role: "primary".to_string(),
                    rank: 0,
                    trust: Some(0.8),
                    score: Some(0.7),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
                CompositionMemberRecord {
                    event_id: "prior-closed-composition".to_string(),
                    memory_id: closed_helper.id,
                    role: "supporting".to_string(),
                    rank: 1,
                    trust: Some(0.7),
                    score: Some(0.6),
                    preview: None,
                    metadata: serde_json::json!({}),
                },
            ],
            &[CompositionOutcomeRecord {
                id: "prior-closed-label".to_string(),
                event_id: "prior-closed-composition".to_string(),
                outcome_type: "closed_by_scope".to_string(),
                labeled_at: Utc::now(),
                label_source: "test".to_string(),
                confidence_delta: Some(-0.3),
                notes: None,
                metadata: serde_json::json!({}),
            }],
        )
        .unwrap();

    let candidates = storage
        .get_never_composed_candidates(10, Some(&["project".to_string()]))
        .unwrap();
    let candidate = candidates
        .iter()
        .find(|candidate| {
            let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
            pair == Storage::pair_key(&successful.id, &closed.id)
        })
        .expect("untried success/closed pair should remain a frontier candidate");

    assert_eq!(candidate.outcome_signal, "mixed_prior_outcomes");
    assert!(
        candidate
            .prior_outcomes
            .iter()
            .any(|outcome| outcome == "accepted")
    );
    assert!(
        candidate
            .prior_outcomes
            .iter()
            .any(|outcome| outcome == "closed_by_scope")
    );
}

#[test]
fn test_never_composed_surfaces_weak_tie_shared_terms_without_shared_tags() {
    let storage = create_test_storage();
    let incident = storage
        .ingest(IngestInput {
            content: "OpenCode handshake stalls when embedding startup blocks stdio negotiation."
                .to_string(),
            node_type: "incident".to_string(),
            tags: vec!["opencode".to_string(), "startup".to_string()],
            ..Default::default()
        })
        .unwrap();
    let mitigation = storage
        .ingest(IngestInput {
            content: "JetBrains startup should keep embedding backfill behind the handshake."
                .to_string(),
            node_type: "mitigation".to_string(),
            tags: vec!["jetbrains".to_string(), "background-work".to_string()],
            ..Default::default()
        })
        .unwrap();

    let candidates = storage.get_never_composed_candidates(10, None).unwrap();
    let candidate = candidates
        .iter()
        .find(|candidate| {
            let pair = Storage::pair_key(&candidate.first_id, &candidate.second_id);
            pair == Storage::pair_key(&incident.id, &mitigation.id)
        })
        .expect("shared terms should surface a weak-tie candidate without shared tags");

    assert!(
        candidate.shared_tags.is_empty(),
        "test fixture intentionally has no shared tags"
    );
    assert!(
        candidate
            .shared_terms
            .iter()
            .any(|term| term == "embedding" || term == "startup" || term == "handshake"),
        "shared terms should explain the candidate"
    );
    assert!(
        candidate.bridge_score > 0.5,
        "different tags and node types should create a bridge signal"
    );
}

#[test]
fn test_dream_history_save_and_get_last() {
    let storage = create_test_storage();
    let now = Utc::now();

    let record = DreamHistoryRecord {
        dreamed_at: now,
        duration_ms: 1500,
        memories_replayed: 50,
        connections_found: 12,
        insights_generated: 3,
        memories_strengthened: 8,
        memories_compressed: 2,
        phase_nrem1_ms: None,
        phase_nrem3_ms: None,
        phase_rem_ms: None,
        phase_integration_ms: None,
        summaries_generated: None,
        emotional_memories_processed: None,
        creative_connections_found: None,
    };

    let id = storage.save_dream_history(&record).unwrap();
    assert!(id > 0);

    let last = storage.get_last_dream().unwrap();
    assert!(last.is_some());
    // Timestamps should be within 1 second (RFC3339 round-trip)
    let diff = (last.unwrap() - now).num_seconds().abs();
    assert!(diff <= 1, "Timestamp mismatch: diff={}s", diff);
}

#[test]
fn test_dream_history_empty() {
    let storage = create_test_storage();
    let last = storage.get_last_dream().unwrap();
    assert!(last.is_none());
}

#[test]
fn test_count_memories_since() {
    let storage = create_test_storage();
    let before = Utc::now() - Duration::seconds(10);

    for i in 0..5 {
        storage
            .ingest(IngestInput {
                content: format!("Count test memory {}", i),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
    }

    let count = storage.count_memories_since(before).unwrap();
    assert_eq!(count, 5);

    let future = Utc::now() + Duration::hours(1);
    let count_future = storage.count_memories_since(future).unwrap();
    assert_eq!(count_future, 0);
}
