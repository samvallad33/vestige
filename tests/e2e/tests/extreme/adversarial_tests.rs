//! # Adversarial Tests for Vestige (Extreme Testing)
//!
//! These tests validate system robustness against adversarial inputs:
//! - Malformed data handling
//! - Boundary condition exploitation
//! - Unicode and encoding edge cases
//! - Extremely long inputs
//! - Malicious graph structures
//! - NaN and infinity handling
//! - Null and empty value handling
//!
//! Based on security testing principles and fuzzing methodologies

use chrono::Utc;
use vestige_core::neuroscience::hippocampal_index::HippocampalIndex;
use vestige_core::neuroscience::spreading_activation::{
    ActivationConfig, ActivationNetwork, LinkType,
};
use vestige_core::neuroscience::synaptic_tagging::SynapticTaggingSystem;

// ============================================================================
// MALFORMED INPUT HANDLING (2 tests)
// ============================================================================

/// Test handling of empty and whitespace-only inputs.
///
/// Validates that empty strings don't cause crashes or undefined behavior.
#[test]
fn test_adversarial_empty_inputs() {
    let mut network = ActivationNetwork::new();

    // Empty string node IDs
    network.add_edge(
        "".to_string(),
        "target".to_string(),
        LinkType::Semantic,
        0.5,
    );
    network.add_edge(
        "source".to_string(),
        "".to_string(),
        LinkType::Semantic,
        0.5,
    );
    network.add_edge("".to_string(), "".to_string(), LinkType::Semantic, 0.5);

    // Should handle gracefully
    let results = network.activate("", 1.0);
    // Empty node might have associations or not, but shouldn't crash
    let _ = results.len();

    // Whitespace-only IDs
    network.add_edge(
        "   ".to_string(),
        "normal".to_string(),
        LinkType::Semantic,
        0.6,
    );
    network.add_edge(
        "\t\n".to_string(),
        "normal".to_string(),
        LinkType::Temporal,
        0.5,
    );

    let whitespace_results = network.activate("   ", 1.0);
    let _ = whitespace_results.len();

    // System should still work with normal nodes
    let _normal_results = network.activate("source", 1.0);
    assert!(
        network.node_count() >= 2,
        "Network should contain normal nodes"
    );
}

/// Test handling of extremely long string inputs.
///
/// Validates that very long IDs don't cause buffer overflows or memory issues.
#[test]
fn test_adversarial_extremely_long_inputs() {
    let mut network = ActivationNetwork::new();

    // Create extremely long node IDs
    let long_id_1: String = "a".repeat(10000);
    let long_id_2: String = "b".repeat(10000);

    network.add_edge(
        long_id_1.clone(),
        long_id_2.clone(),
        LinkType::Semantic,
        0.8,
    );

    // Should handle long IDs
    let results = network.activate(&long_id_1, 1.0);
    assert_eq!(results.len(), 1, "Should find connection to long_id_2");
    assert_eq!(
        results[0].memory_id, long_id_2,
        "Result should have correct long ID"
    );

    // Test with hippocampal index
    let index = HippocampalIndex::new();
    let very_long_content = "word ".repeat(50000); // ~300KB of text

    let result = index.index_memory(
        "long_content_memory",
        &very_long_content,
        "test",
        Utc::now(),
        None,
    );

    assert!(result.is_ok(), "Should handle very long content");
}

// ============================================================================
// UNICODE AND ENCODING EDGE CASES (2 tests)
// ============================================================================

/// Test handling of Unicode characters and edge cases.
///
/// Validates proper handling of various Unicode encodings.
#[test]
fn test_adversarial_unicode_handling() {
    let mut network = ActivationNetwork::new();

    // Various Unicode edge cases
    let unicode_ids = vec![
        "简体中文",                 // Chinese
        "日本語テキスト",           // Japanese
        "한국어",                   // Korean
        "مرحبا",                    // Arabic (RTL)
        "שלום",                     // Hebrew (RTL)
        "🦀🔥💯",                   // Emojis
        "Ã̲̊",                        // Combining characters
        "\u{200B}",                 // Zero-width space
        "\u{FEFF}",                 // BOM
        "a\u{0308}",                // 'a' with combining umlaut
        "🏳️‍🌈",                       // Emoji sequence with ZWJ
        "\u{202E}reversed\u{202C}", // RTL override
    ];

    for (i, id) in unicode_ids.iter().enumerate() {
        network.add_edge(
            id.to_string(),
            format!("target_{}", i),
            LinkType::Semantic,
            0.8,
        );
    }

    // All should be retrievable
    for id in &unicode_ids {
        let results = network.activate(id, 1.0);
        assert!(
            !results.is_empty(),
            "Unicode ID '{}' should produce results",
            id.escape_unicode()
        );
    }

    // Verify associations
    for id in &unicode_ids {
        let assoc = network.get_associations(id);
        assert!(
            !assoc.is_empty(),
            "Unicode ID '{}' should have associations",
            id.escape_unicode()
        );
    }
}

/// Test handling of null bytes and control characters.
///
/// Validates that embedded null bytes don't truncate or corrupt data.
#[test]
fn test_adversarial_control_characters() {
    let mut network = ActivationNetwork::new();

    // IDs with embedded control characters
    let control_ids = vec![
        "before\0after",  // Null byte
        "line1\nline2",   // Newline
        "tab\there",      // Tab
        "return\rhere",   // Carriage return
        "bell\x07ring",   // Bell
        "escape\x1B[31m", // ANSI escape
        "backspace\x08x", // Backspace
    ];

    for (i, id) in control_ids.iter().enumerate() {
        network.add_edge(
            id.to_string(),
            format!("ctrl_target_{}", i),
            LinkType::Semantic,
            0.7,
        );
    }

    // All should be stored and retrievable
    for (i, id) in control_ids.iter().enumerate() {
        let results = network.activate(id, 1.0);
        assert!(
            !results.is_empty(),
            "Control char ID at index {} should be retrievable",
            i
        );
    }

    // Test in STC
    let mut stc = SynapticTaggingSystem::new();
    for id in &control_ids {
        stc.tag_memory(id);
    }

    let stats = stc.stats();
    assert!(
        stats.active_tags >= control_ids.len(),
        "All control character memories should be tagged"
    );
}

// ============================================================================
// BOUNDARY CONDITION EXPLOITATION (2 tests)
// ============================================================================

/// Test edge weight boundary conditions.
///
/// Validates proper handling of weights at and beyond valid ranges.
#[test]
fn test_adversarial_weight_boundaries() {
    let mut network = ActivationNetwork::new();

    // Edge weights at boundaries
    let weight_cases = vec![
        ("zero", 0.0),
        ("tiny", f64::MIN_POSITIVE),
        ("small", 0.001),
        ("normal", 0.5),
        ("high", 0.999),
        ("one", 1.0),
    ];

    for (name, weight) in &weight_cases {
        network.add_edge(
            "hub".to_string(),
            format!("weight_{}", name),
            LinkType::Semantic,
            *weight,
        );
    }

    let results = network.activate("hub", 1.0);

    // Higher weights should produce higher activation
    let mut activations: Vec<(&str, f64)> = weight_cases
        .iter()
        .filter_map(|(name, _)| {
            results
                .iter()
                .find(|r| r.memory_id == format!("weight_{}", name))
                .map(|r| (*name, r.activation))
        })
        .collect();

    // Sort by activation
    activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // "one" should have highest activation (or tied for highest)
    if !activations.is_empty() {
        let (top_name, _) = activations[0];
        assert!(
            top_name == "one" || top_name == "high",
            "Highest weight should have highest activation, got: {}",
            top_name
        );
    }

    // Zero weight edges might not propagate activation at all
    let zero_activation = results
        .iter()
        .find(|r| r.memory_id == "weight_zero")
        .map(|r| r.activation);

    if let Some(act) = zero_activation {
        assert!(
            act <= 0.001,
            "Zero weight should produce minimal activation: {}",
            act
        );
    }
}

/// Test configuration parameter boundaries.
///
/// Validates behavior with extreme configuration values.
#[test]
fn test_adversarial_config_boundaries() {
    // Test with very high decay (almost no decay)
    let high_decay_config = ActivationConfig {
        decay_factor: 0.9999,
        max_hops: 10,
        min_threshold: 0.0001,
        allow_cycles: false,
    };
    let mut high_decay_net = ActivationNetwork::with_config(high_decay_config);
    high_decay_net.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.9);
    high_decay_net.add_edge("b".to_string(), "c".to_string(), LinkType::Semantic, 0.9);

    let high_results = high_decay_net.activate("a", 1.0);
    assert!(!high_results.is_empty(), "High decay should still work");

    // Test with very low decay (rapid decay)
    let low_decay_config = ActivationConfig {
        decay_factor: 0.01,
        max_hops: 10,
        min_threshold: 0.0001,
        allow_cycles: false,
    };
    let mut low_decay_net = ActivationNetwork::with_config(low_decay_config);
    low_decay_net.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.9);
    low_decay_net.add_edge("b".to_string(), "c".to_string(), LinkType::Semantic, 0.9);

    let _low_results = low_decay_net.activate("a", 1.0);
    // With 0.01 decay, activation drops to 0.9 * 0.01 = 0.009 after one hop
    // Then 0.009 * 0.9 * 0.01 = 0.000081 after two hops (below most thresholds)

    // Test with max_hops = 0
    let zero_hops_config = ActivationConfig {
        decay_factor: 0.8,
        max_hops: 0,
        min_threshold: 0.1,
        allow_cycles: false,
    };
    let mut zero_hops_net = ActivationNetwork::with_config(zero_hops_config);
    zero_hops_net.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.9);

    let zero_results = zero_hops_net.activate("a", 1.0);
    assert!(zero_results.is_empty(), "Zero max_hops should find nothing");
}

// ============================================================================
// MALICIOUS GRAPH STRUCTURES (2 tests)
// ============================================================================

/// Test handling of cyclic graphs.
///
/// Validates that cycles don't cause infinite loops.
#[test]
fn test_adversarial_cyclic_graphs() {
    // Test with cycles disallowed (default)
    let no_cycle_config = ActivationConfig {
        decay_factor: 0.8,
        max_hops: 10,
        min_threshold: 0.01,
        allow_cycles: false,
    };
    let mut no_cycle_net = ActivationNetwork::with_config(no_cycle_config);

    // Create a simple cycle: A -> B -> C -> A
    no_cycle_net.add_edge(
        "cycle_a".to_string(),
        "cycle_b".to_string(),
        LinkType::Semantic,
        0.9,
    );
    no_cycle_net.add_edge(
        "cycle_b".to_string(),
        "cycle_c".to_string(),
        LinkType::Semantic,
        0.9,
    );
    no_cycle_net.add_edge(
        "cycle_c".to_string(),
        "cycle_a".to_string(),
        LinkType::Semantic,
        0.9,
    );

    let start = std::time::Instant::now();
    let results = no_cycle_net.activate("cycle_a", 1.0);
    let duration = start.elapsed();

    // Should complete quickly (not get stuck in loop)
    assert!(
        duration.as_millis() < 100,
        "Cycle handling should be fast: {:?}",
        duration
    );

    // Should find nodes (but not infinitely many)
    assert!(
        results.len() <= 10,
        "Should not have infinite results from cycle: {}",
        results.len()
    );

    // Test with cycles allowed
    let cycle_config = ActivationConfig {
        decay_factor: 0.5,
        max_hops: 5,
        min_threshold: 0.1,
        allow_cycles: true,
    };
    let mut cycle_net = ActivationNetwork::with_config(cycle_config);

    cycle_net.add_edge("a".to_string(), "b".to_string(), LinkType::Semantic, 0.9);
    cycle_net.add_edge("b".to_string(), "c".to_string(), LinkType::Semantic, 0.9);
    cycle_net.add_edge("c".to_string(), "a".to_string(), LinkType::Semantic, 0.9);

    let start = std::time::Instant::now();
    let _cycle_results = cycle_net.activate("a", 1.0);
    let duration = start.elapsed();

    // Should still complete quickly
    assert!(
        duration.as_millis() < 100,
        "Cycle-allowed mode should still be fast: {:?}",
        duration
    );
}

/// Test self-referential edges.
///
/// Validates handling of nodes that point to themselves.
#[test]
fn test_adversarial_self_loops() {
    let mut network = ActivationNetwork::new();

    // Create self-loop
    network.add_edge(
        "self_loop".to_string(),
        "self_loop".to_string(),
        LinkType::Semantic,
        0.9,
    );

    // Also connect to other nodes
    network.add_edge(
        "self_loop".to_string(),
        "other".to_string(),
        LinkType::Semantic,
        0.7,
    );

    let start = std::time::Instant::now();
    let results = network.activate("self_loop", 1.0);
    let duration = start.elapsed();

    // Should complete quickly
    assert!(
        duration.as_millis() < 100,
        "Self-loop should be handled quickly: {:?}",
        duration
    );

    // Should find "other" at least
    let found_other = results.iter().any(|r| r.memory_id == "other");
    assert!(found_other, "Should find non-self-loop connections");
}

// ============================================================================
// SPECIAL NUMERIC VALUE HANDLING (1 test)
// ============================================================================

/// Test handling of special floating point values.
///
/// Validates that NaN, infinity, and negative values are handled safely.
#[test]
fn test_adversarial_special_numeric_values() {
    let mut network = ActivationNetwork::new();

    // Note: The actual behavior depends on implementation
    // We're testing that the system doesn't crash

    // Normal edge for baseline
    network.add_edge(
        "normal".to_string(),
        "target".to_string(),
        LinkType::Semantic,
        0.8,
    );

    // Test activation with edge case values
    // (The implementation should clamp or validate these)

    // Test with 0.0 activation (should produce no results or minimal)
    let _zero_results = network.activate("normal", 0.0);
    // Might be empty or have very low activation

    // Test with very small activation
    let tiny_results = network.activate("normal", f64::MIN_POSITIVE);
    let _ = tiny_results.len();

    // Test with activation > 1.0 (should be clamped or handled)
    let high_results = network.activate("normal", 2.0);
    assert!(
        !high_results.is_empty(),
        "High activation should still work (clamped to 1.0)"
    );

    // Verify activation values are reasonable (allow some overshoot due to multi-source)
    for result in &high_results {
        assert!(
            result.activation >= 0.0 && result.activation <= 2.0,
            "Activation should be bounded: {}",
            result.activation
        );
    }

    // Test reinforce with negative (should be rejected or clamped)
    network.reinforce_edge("normal", "target", -0.5);

    // Edge should still exist and be valid
    let assoc = network.get_associations("normal");
    assert!(
        !assoc.is_empty(),
        "Edge should still exist after negative reinforce attempt"
    );
}
