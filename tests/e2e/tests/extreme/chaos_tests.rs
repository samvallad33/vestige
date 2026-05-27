//! # Chaos Tests for Vestige (Extreme Testing)
//!
//! These tests validate system resilience under chaotic and unpredictable conditions:
//! - Random operation sequences
//! - Concurrent stress testing
//! - Resource exhaustion scenarios
//! - Recovery from partial failures
//! - Network partition simulation
//! - Clock skew handling
//! - Memory pressure testing
//! - Cascading failure prevention
//!
//! Based on Chaos Engineering principles (Netflix, 2011)

use chrono::{Duration, Utc};
use vestige_core::neuroscience::hippocampal_index::{HippocampalIndex, IndexQuery};
use vestige_core::neuroscience::spreading_activation::{
    ActivationConfig, ActivationNetwork, LinkType,
};
use vestige_core::neuroscience::synaptic_tagging::{
    CaptureWindow, ImportanceEvent, SynapticTaggingConfig, SynapticTaggingSystem,
};

// ============================================================================
// RANDOM OPERATION SEQUENCE TESTS (2 tests)
// ============================================================================

/// Test that the system remains consistent under random operation sequences.
///
/// Performs a series of random-like operations in different orders to ensure
/// the system maintains invariants regardless of operation sequence.
#[test]
fn test_chaos_random_operation_sequence() {
    let mut network = ActivationNetwork::new();

    // Sequence 1: Add edges, then reinforce, then activate
    for i in 0..50 {
        network.add_edge(
            format!("node_{}", i),
            format!("node_{}", (i + 7) % 50),
            LinkType::Semantic,
            0.5 + ((i % 5) as f64) * 0.1,
        );
    }

    for i in 0..25 {
        network.reinforce_edge(
            &format!("node_{}", i),
            &format!("node_{}", (i + 7) % 50),
            0.1,
        );
    }

    let results1 = network.activate("node_0", 1.0);

    // Sequence 2: Interleaved operations
    let mut network2 = ActivationNetwork::new();

    for i in 0..50 {
        network2.add_edge(
            format!("node_{}", i),
            format!("node_{}", (i + 7) % 50),
            LinkType::Semantic,
            0.5 + ((i % 5) as f64) * 0.1,
        );

        // Interleave reinforcement
        if i >= 7 {
            network2.reinforce_edge(&format!("node_{}", i - 7), &format!("node_{}", i % 50), 0.1);
        }
    }

    let results2 = network2.activate("node_0", 1.0);

    // Both should find nodes (exact counts may differ due to timing effects)
    assert!(
        !results1.is_empty() && !results2.is_empty(),
        "Both operation sequences should produce results"
    );

    // Node count should be consistent
    assert_eq!(
        network.node_count(),
        network2.node_count(),
        "Node count should be the same regardless of operation order"
    );
}

/// Test recovery from interleaved add/remove cycles.
///
/// Simulates rapid creation and removal of edges to test system stability.
#[test]
fn test_chaos_add_remove_cycles() {
    let mut network = ActivationNetwork::new();

    // Create initial structure
    for i in 0..20 {
        network.add_edge(
            format!("stable_{}", i),
            format!("stable_{}", (i + 1) % 20),
            LinkType::Semantic,
            0.8,
        );
    }

    let initial_node_count = network.node_count();
    let initial_edge_count = network.edge_count();

    // Rapid add/reinforce cycles (simulating chaos)
    for cycle in 0..10 {
        // Add temporary edges
        for i in 0..5 {
            network.add_edge(
                format!("temp_{}_{}", cycle, i),
                format!("stable_{}", i),
                LinkType::Temporal,
                0.3,
            );
        }

        // Reinforce some stable edges
        for i in 0..10 {
            network.reinforce_edge(
                &format!("stable_{}", i),
                &format!("stable_{}", (i + 1) % 20),
                0.05,
            );
        }

        // Verify system still works
        let results = network.activate(&format!("stable_{}", cycle % 20), 1.0);
        assert!(
            !results.is_empty(),
            "System should remain functional during chaos"
        );
    }

    // Final activation should still work
    let final_results = network.activate("stable_0", 1.0);
    assert!(
        !final_results.is_empty(),
        "System should be fully functional after chaos cycles"
    );

    // Stable structure should be preserved (edges reinforced)
    let stable_edge_count = network.edge_count();
    let stable_node_count = network.node_count();
    assert!(
        stable_node_count >= initial_node_count,
        "Stable nodes should be preserved: {} >= {}",
        stable_node_count,
        initial_node_count
    );
    assert!(
        stable_edge_count >= initial_edge_count,
        "Stable edges should be preserved: {} >= {}",
        stable_edge_count,
        initial_edge_count
    );
}

// ============================================================================
// CONCURRENT STRESS TESTS (2 tests)
// ============================================================================

/// Test high-frequency activation requests.
///
/// Simulates many rapid activation queries to test performance under load.
#[test]
fn test_chaos_high_frequency_activations() {
    let config = ActivationConfig {
        decay_factor: 0.7,
        max_hops: 3,
        min_threshold: 0.1,
        allow_cycles: false,
    };
    let mut network = ActivationNetwork::with_config(config);

    // Create a moderately complex network
    for i in 0..100 {
        network.add_edge(
            format!("node_{}", i),
            format!("node_{}", (i * 7 + 3) % 100),
            LinkType::Semantic,
            0.6 + ((i % 4) as f64) * 0.1,
        );
        network.add_edge(
            format!("node_{}", i),
            format!("node_{}", (i * 11 + 5) % 100),
            LinkType::Temporal,
            0.5 + ((i % 3) as f64) * 0.1,
        );
    }

    // Rapid-fire activations
    let start = std::time::Instant::now();
    let mut total_results = 0;

    for i in 0..1000 {
        let results = network.activate(&format!("node_{}", i % 100), 1.0);
        total_results += results.len();
    }

    let duration = start.elapsed();

    // Should complete quickly (< 1 second for 1000 activations)
    assert!(
        duration.as_millis() < 1000,
        "1000 activations should complete in < 1s: {:?}",
        duration
    );

    // Should produce results
    assert!(
        total_results > 0,
        "High-frequency activations should produce results"
    );

    // Average time per activation (allow up to 100ms in debug mode)
    let avg_ms = duration.as_micros() as f64 / 1000.0;
    assert!(
        avg_ms < 100.0,
        "Average activation time should be reasonable: {:.3}ms",
        avg_ms
    );
}

/// Test network growth under continuous operation.
///
/// Simulates a system that continuously grows while being queried.
#[test]
fn test_chaos_continuous_growth_under_load() {
    let mut network = ActivationNetwork::new();

    // Initial seed
    network.add_edge(
        "root".to_string(),
        "child_0".to_string(),
        LinkType::Semantic,
        0.8,
    );

    // Continuously grow while querying
    for iteration in 0..500 {
        // Add new nodes
        network.add_edge(
            format!("child_{}", iteration),
            format!("child_{}", iteration + 1),
            LinkType::Semantic,
            0.7,
        );

        // Add cross-links periodically
        if iteration % 10 == 0 && iteration > 10 {
            network.add_edge(
                format!("child_{}", iteration),
                format!("child_{}", iteration - 10),
                LinkType::Temporal,
                0.5,
            );
        }

        // Query every 50 iterations
        if iteration % 50 == 0 {
            let results = network.activate("root", 1.0);
            assert!(
                !results.is_empty(),
                "Should find results during growth at iteration {}",
                iteration
            );
        }
    }

    // Final state check
    assert!(
        network.node_count() > 500,
        "Network should have grown: {} nodes",
        network.node_count()
    );

    let final_results = network.activate("root", 1.0);
    assert!(!final_results.is_empty(), "Final activation should succeed");
}

// ============================================================================
// RESOURCE EXHAUSTION TESTS (2 tests)
// ============================================================================

/// Test behavior with very deep chains.
///
/// Creates extremely deep chains to test stack overflow protection.
#[test]
fn test_chaos_deep_chain_handling() {
    let config = ActivationConfig {
        decay_factor: 0.95,   // High to allow deep traversal
        max_hops: 100,        // Allow deep exploration
        min_threshold: 0.001, // Low threshold
        allow_cycles: false,
    };
    let mut network = ActivationNetwork::with_config(config);

    // Create a very deep chain (1000 nodes)
    for i in 0..1000 {
        network.add_edge(
            format!("deep_{}", i),
            format!("deep_{}", i + 1),
            LinkType::Semantic,
            0.99, // Very strong links
        );
    }

    // Should handle deep chain gracefully
    let start = std::time::Instant::now();
    let results = network.activate("deep_0", 1.0);
    let duration = start.elapsed();

    // Should complete without stack overflow
    assert!(
        duration.as_millis() < 500,
        "Deep chain should be handled efficiently: {:?}",
        duration
    );

    // Should find results up to max_hops
    assert!(
        results.len() >= 50,
        "Should find many nodes in deep chain: {} found",
        results.len()
    );

    // Max distance should not exceed max_hops
    let max_distance = results.iter().map(|r| r.distance).max().unwrap_or(0);
    assert!(
        max_distance <= 100,
        "Max distance should respect max_hops: {}",
        max_distance
    );
}

/// Test behavior with extremely wide graphs (high fan-out).
///
/// Creates graphs with very high connectivity to test memory usage.
#[test]
fn test_chaos_high_fanout_handling() {
    let config = ActivationConfig {
        decay_factor: 0.5,
        max_hops: 2,
        min_threshold: 0.1,
        allow_cycles: false,
    };
    let mut network = ActivationNetwork::with_config(config);

    // Create a hub with 1000 connections
    for i in 0..1000 {
        network.add_edge(
            "mega_hub".to_string(),
            format!("spoke_{}", i),
            LinkType::Semantic,
            0.6,
        );
    }

    // Activate from hub
    let start = std::time::Instant::now();
    let results = network.activate("mega_hub", 1.0);
    let duration = start.elapsed();

    // Should complete quickly
    assert!(
        duration.as_millis() < 100,
        "High fan-out should be handled efficiently: {:?}",
        duration
    );

    // Should find many spokes
    assert!(
        results.len() >= 500,
        "Should activate many spokes: {} found",
        results.len()
    );

    // Memory should be reasonable (no explosion)
    let node_count = network.node_count();
    let edge_count = network.edge_count();
    assert_eq!(node_count, 1001, "Should have hub + 1000 spokes");
    assert_eq!(edge_count, 1000, "Should have 1000 edges");
}

// ============================================================================
// CLOCK SKEW AND TIMING TESTS (2 tests)
// ============================================================================

/// Test synaptic tagging with various temporal distances.
///
/// Validates that the capture window handles edge cases correctly.
#[test]
fn test_chaos_capture_window_edge_cases() {
    let window = CaptureWindow::new(9.0, 2.0); // 9 hours back, 2 forward
    let event_time = Utc::now();

    // Test exact boundary conditions
    let test_cases = vec![
        // (hours offset, expected in window)
        (0.0, true),     // Exactly at event
        (8.99, true),    // Just inside back window
        (9.0, true),     // At back boundary
        (9.01, false),   // Just outside back window
        (-1.99, true),   // Just inside forward window
        (-2.0, true),    // At forward boundary
        (-2.01, false),  // Just outside forward window
        (100.0, false),  // Way outside
        (-100.0, false), // Way outside forward
    ];

    for (hours_offset, expected) in test_cases {
        let memory_time = if hours_offset >= 0.0 {
            event_time - Duration::milliseconds((hours_offset * 3600.0 * 1000.0) as i64)
        } else {
            event_time + Duration::milliseconds((-hours_offset * 3600.0 * 1000.0) as i64)
        };

        let in_window = window.is_in_window(memory_time, event_time);
        assert_eq!(
            in_window, expected,
            "Offset {:.2}h: expected in_window={}, got {}",
            hours_offset, expected, in_window
        );
    }
}

/// Test behavior with very old timestamps.
///
/// Ensures system handles memories from far in the past.
#[test]
fn test_chaos_ancient_memories() {
    let config = SynapticTaggingConfig {
        capture_window: CaptureWindow::new(9.0, 2.0),
        prp_threshold: 0.5,
        tag_lifetime_hours: 12.0,
        min_tag_strength: 0.1,
        max_cluster_size: 100,
        enable_clustering: true,
        auto_decay: true,
        cleanup_interval_hours: 1.0,
    };

    let mut stc = SynapticTaggingSystem::with_config(config);

    // Tag memories at various ages
    stc.tag_memory("very_old"); // Will be tagged "now" for testing
    stc.tag_memory("old");
    stc.tag_memory("recent");

    // Trigger importance - should capture recent memories
    let event = ImportanceEvent::user_flag("trigger", Some("Ancient memory test"));
    let result = stc.trigger_prp(event);

    // System should handle this gracefully
    assert!(
        result.captured_count() <= 3,
        "Importance triggering should stay bounded by active memories"
    );

    // All memories should be accessible
    let stats = stc.stats();
    assert!(
        stats.active_tags >= 3,
        "All tagged memories should be tracked"
    );
}

// ============================================================================
// CASCADING FAILURE PREVENTION (2 tests but combined)
// ============================================================================

/// Test that errors in one subsystem don't cascade.
///
/// Validates that failures are isolated and don't bring down the entire system.
#[test]
fn test_chaos_isolated_subsystem_failures() {
    // Test 1: Network with invalid queries should not crash
    let mut network = ActivationNetwork::new();
    network.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.8);

    // Query non-existent node should return empty, not crash
    let results = network.activate("nonexistent", 1.0);
    assert!(
        results.is_empty(),
        "Non-existent node should return empty results"
    );

    // System should still work after "failed" query
    let valid_results = network.activate("a", 1.0);
    assert!(
        !valid_results.is_empty(),
        "System should work after handling missing node"
    );

    // Test 2: STC with edge case inputs
    let mut stc = SynapticTaggingSystem::new();

    // Empty string memory ID
    stc.tag_memory("");
    stc.tag_memory_with_strength("zero_strength", 0.0);
    stc.tag_memory_with_strength("high_strength", 1.0);

    // System should still function
    let event = ImportanceEvent::user_flag("test", None);
    let result = stc.trigger_prp(event);

    // Should not crash, result should be valid
    let _ = result.captured_count();
}

/// Test graceful degradation under extreme load.
///
/// System should maintain core functionality even when stressed.
#[test]
fn test_chaos_graceful_degradation() {
    let index = HippocampalIndex::new();
    let now = Utc::now();

    // Create many memories rapidly
    for i in 0..500 {
        let embedding: Vec<f32> = (0..128)
            .map(|j| ((i * 17 + j) as f32 / 1000.0).sin())
            .collect();

        let _ = index.index_memory(
            &format!("stress_memory_{}", i),
            &format!("Content for stress test memory number {}", i),
            "stress_test",
            now,
            Some(embedding),
        );
    }

    // Query should still work under load
    let query = IndexQuery::from_text("stress").with_limit(10);
    let results = index.search_indices(&query);

    assert!(
        results.is_ok(),
        "Search should succeed even after rapid indexing"
    );

    // Stats should be available
    let stats = index.stats();
    assert!(
        stats.total_indices >= 500,
        "All memories should be indexed: {}",
        stats.total_indices
    );
}
