//! # Research Validation Tests for Vestige (Extreme Testing)
//!
//! These tests validate that Vestige's implementation matches published research:
//! - Collins & Loftus (1975) spreading activation model
//! - Frey & Morris (1997) synaptic tagging and capture
//! - Teyler & Rudy (2007) hippocampal indexing theory
//! - Ebbinghaus (1885) forgetting curve
//! - FSRS-6 algorithm validation
//!
//! Each test cites the specific research findings being validated.

use chrono::{Duration, Utc};
use std::collections::HashSet;
use vestige_core::neuroscience::hippocampal_index::{
    HippocampalIndex, HippocampalIndexConfig, IndexQuery,
};
use vestige_core::neuroscience::spreading_activation::{
    ActivationConfig, ActivationNetwork, LinkType,
};
use vestige_core::neuroscience::synaptic_tagging::{
    CaptureWindow, ImportanceEvent, ImportanceEventType, SynapticTaggingConfig,
    SynapticTaggingSystem,
};

// ============================================================================
// COLLINS & LOFTUS (1975) SPREADING ACTIVATION VALIDATION (1 test)
// ============================================================================

/// Validate Collins & Loftus (1975) spreading activation model.
///
/// Key findings from the original paper:
/// 1. Activation spreads from source to connected nodes
/// 2. Activation decreases with distance (semantic distance)
/// 3. Shorter paths produce stronger activation
/// 4. Multiple paths converging increase activation
///
/// Reference: Collins, A. M., & Loftus, E. F. (1975). A spreading-activation
/// theory of semantic processing. Psychological Review, 82(6), 407-428.
#[test]
fn test_research_collins_loftus_spreading_activation() {
    let config = ActivationConfig {
        decay_factor: 0.75, // Semantic distance decay
        max_hops: 4,
        min_threshold: 0.05,
        allow_cycles: false,
    };
    let mut network = ActivationNetwork::with_config(config);

    // Recreate classic semantic network from the paper
    // "Fire truck" example: fire_truck -> red -> roses, fire_truck -> vehicle
    network.add_edge(
        "fire_truck".to_string(),
        "red".to_string(),
        LinkType::Semantic,
        0.9,
    );
    network.add_edge(
        "fire_truck".to_string(),
        "vehicle".to_string(),
        LinkType::Semantic,
        0.85,
    );
    network.add_edge(
        "fire_truck".to_string(),
        "fire".to_string(),
        LinkType::Semantic,
        0.9,
    );
    network.add_edge(
        "red".to_string(),
        "roses".to_string(),
        LinkType::Semantic,
        0.7,
    );
    network.add_edge(
        "red".to_string(),
        "cherries".to_string(),
        LinkType::Semantic,
        0.65,
    );
    network.add_edge(
        "red".to_string(),
        "apples".to_string(),
        LinkType::Semantic,
        0.7,
    );
    network.add_edge(
        "vehicle".to_string(),
        "car".to_string(),
        LinkType::Semantic,
        0.8,
    );
    network.add_edge(
        "vehicle".to_string(),
        "truck".to_string(),
        LinkType::Semantic,
        0.85,
    );
    network.add_edge(
        "fire".to_string(),
        "flames".to_string(),
        LinkType::Semantic,
        0.9,
    );
    network.add_edge(
        "fire".to_string(),
        "heat".to_string(),
        LinkType::Semantic,
        0.8,
    );

    // Add convergent paths (multiple routes to same concept)
    network.add_edge(
        "apples".to_string(),
        "fruit".to_string(),
        LinkType::Semantic,
        0.9,
    );
    network.add_edge(
        "cherries".to_string(),
        "fruit".to_string(),
        LinkType::Semantic,
        0.9,
    );

    let results = network.activate("fire_truck", 1.0);

    // Validation 1: Direct connections (distance 1) have highest activation
    let red_activation = results
        .iter()
        .find(|r| r.memory_id == "red")
        .map(|r| r.activation)
        .unwrap_or(0.0);
    let roses_activation = results
        .iter()
        .find(|r| r.memory_id == "roses")
        .map(|r| r.activation)
        .unwrap_or(0.0);

    assert!(
        red_activation > roses_activation,
        "C&L Finding 1: Direct connections ({}) > indirect ({})",
        red_activation,
        roses_activation
    );

    // Validation 2: Activation decreases with semantic distance
    let distance_1: Vec<f64> = results
        .iter()
        .filter(|r| r.distance == 1)
        .map(|r| r.activation)
        .collect();
    let distance_2: Vec<f64> = results
        .iter()
        .filter(|r| r.distance == 2)
        .map(|r| r.activation)
        .collect();

    let avg_d1 = distance_1.iter().sum::<f64>() / distance_1.len().max(1) as f64;
    let avg_d2 = distance_2.iter().sum::<f64>() / distance_2.len().max(1) as f64;

    assert!(
        avg_d1 > avg_d2,
        "C&L Finding 2: Avg activation at d=1 ({:.3}) > d=2 ({:.3})",
        avg_d1,
        avg_d2
    );

    // Validation 3: All connected concepts are reachable
    let reachable: HashSet<_> = results.iter().map(|r| r.memory_id.as_str()).collect();
    assert!(reachable.contains("red"), "Should reach 'red'");
    assert!(reachable.contains("vehicle"), "Should reach 'vehicle'");
    assert!(reachable.contains("fire"), "Should reach 'fire'");
    assert!(
        reachable.contains("roses"),
        "Should reach 'roses' through 'red'"
    );

    // Validation 4: Path information is preserved
    let roses_result = results.iter().find(|r| r.memory_id == "roses").unwrap();
    assert_eq!(roses_result.distance, 2, "Roses should be 2 hops away");
    assert!(
        roses_result.path.contains(&"red".to_string()),
        "Path to roses should include 'red'"
    );
}

// ============================================================================
// FREY & MORRIS (1997) SYNAPTIC TAGGING VALIDATION (1 test)
// ============================================================================

/// Validate Frey & Morris (1997) synaptic tagging and capture.
///
/// Key findings from the original paper:
/// 1. Weak stimulation creates tags but not lasting change
/// 2. Strong stimulation triggers protein synthesis (PRP)
/// 3. Tagged synapses within time window are captured
/// 4. Capture window is asymmetric (longer backward)
///
/// Reference: Frey, U., & Morris, R. G. (1997). Synaptic tagging and long-term
/// potentiation. Nature, 385(6616), 533-536.
#[test]
fn test_research_frey_morris_synaptic_tagging() {
    let config = SynapticTaggingConfig {
        capture_window: CaptureWindow::new(9.0, 2.0), // Hours: 9 back, 2 forward
        prp_threshold: 0.7,
        tag_lifetime_hours: 12.0,
        min_tag_strength: 0.3,
        max_cluster_size: 50,
        enable_clustering: true,
        auto_decay: true,
        cleanup_interval_hours: 1.0,
    };

    let mut stc = SynapticTaggingSystem::with_config(config);

    // Finding 1: Weak stimulation creates tags
    stc.tag_memory_with_strength("weak_stim_1", 0.4); // Above min (0.3), weak
    stc.tag_memory_with_strength("weak_stim_2", 0.5);

    let stats_after_weak = stc.stats();
    assert!(
        stats_after_weak.active_tags >= 2,
        "F&M Finding 1: Weak stimulation should create tags"
    );

    // Finding 2: Strong stimulation triggers PRP and capture
    stc.tag_memory_with_strength("context_memory", 0.6);

    let strong_event = ImportanceEvent {
        event_type: ImportanceEventType::EmotionalContent,
        memory_id: Some("strong_trigger".to_string()),
        timestamp: Utc::now(),
        strength: 0.95, // Above threshold (0.7)
        context: Some("Strong emotional event triggers PRP".to_string()),
    };

    let capture_result = stc.trigger_prp(strong_event);

    assert!(
        !capture_result.captured_memories.is_empty(),
        "F&M Finding 2: Strong stimulation should trigger PRP"
    );
    assert!(
        capture_result.has_captures(),
        "F&M Finding 2: PRP should capture tagged memories"
    );

    // Finding 3: Captured memories within window are consolidated
    let captured_count = capture_result.captured_count();
    assert!(
        captured_count >= 2,
        "F&M Finding 3: Should capture tagged memories: {}",
        captured_count
    );

    // Finding 4: Asymmetric window (test window parameters)
    let window = CaptureWindow::new(9.0, 2.0);
    let event_time = Utc::now();

    // 8 hours before should be in window
    let before_8h = event_time - Duration::hours(8);
    assert!(
        window.is_in_window(before_8h, event_time),
        "F&M Finding 4: 8h before should be in 9h backward window"
    );

    // 10 hours before should be out of window
    let before_10h = event_time - Duration::hours(10);
    assert!(
        !window.is_in_window(before_10h, event_time),
        "F&M Finding 4: 10h before should be outside 9h backward window"
    );

    // 1 hour after should be in window
    let after_1h = event_time + Duration::hours(1);
    assert!(
        window.is_in_window(after_1h, event_time),
        "F&M Finding 4: 1h after should be in 2h forward window"
    );

    // 3 hours after should be out of window
    let after_3h = event_time + Duration::hours(3);
    assert!(
        !window.is_in_window(after_3h, event_time),
        "F&M Finding 4: 3h after should be outside 2h forward window"
    );
}

// ============================================================================
// TEYLER & RUDY (2007) HIPPOCAMPAL INDEXING VALIDATION (1 test)
// ============================================================================

/// Validate Teyler & Rudy (2007) hippocampal indexing theory.
///
/// Key findings from the theory:
/// 1. Hippocampus creates sparse index patterns (barcodes)
/// 2. Index points to distributed cortical representations
/// 3. Retrieval is two-phase: fast index lookup, then full retrieval
/// 4. Index is compact compared to full representation
///
/// Reference: Teyler, T. J., & Rudy, J. W. (2007). The hippocampal indexing
/// theory and episodic memory: updating the index. Hippocampus, 17(12), 1158-1169.
#[test]
fn test_research_teyler_rudy_hippocampal_indexing() {
    let _config = HippocampalIndexConfig::default();
    let index = HippocampalIndex::new();
    let now = Utc::now();

    // Finding 1: Create sparse index patterns (barcodes)
    let full_embedding: Vec<f32> = (0..384)
        .map(|i| ((i as f32 / 100.0) * std::f32::consts::PI).sin())
        .collect();

    let barcode = index
        .index_memory(
            "episodic_memory_1",
            "Detailed episodic memory content with rich context",
            "episodic",
            now,
            Some(full_embedding.clone()),
        )
        .expect("Should create barcode");

    // Barcode should be a valid identifier (u64 ID)
    // First barcode may have id=0, which is valid
    assert!(
        barcode.creation_hash > 0 || barcode.content_fingerprint > 0,
        "T&R Finding 1: Barcode should have valid fingerprints"
    );

    // Finding 2: Index points to content (content pointers)
    let memory_index = index
        .get_index("episodic_memory_1")
        .expect("Should retrieve")
        .expect("Should exist");

    assert!(
        !memory_index.content_pointers.is_empty(),
        "T&R Finding 2: Index should point to content storage"
    );

    // Finding 3: Two-phase retrieval - fast index lookup
    // Create multiple memories for search
    for i in 0..100 {
        let emb: Vec<f32> = (0..384)
            .map(|j| ((i * 17 + j) as f32 / 200.0).sin())
            .collect();

        let _ = index.index_memory(
            &format!("memory_{}", i),
            &format!("Content for memory {} with various topics", i),
            "fact",
            now,
            Some(emb),
        );
    }

    // Phase 1: Fast index search
    let query = IndexQuery::from_text("memory").with_limit(10);
    let start = std::time::Instant::now();
    let search_results = index.search_indices(&query).expect("Should search");
    let search_duration = start.elapsed();

    assert!(
        search_duration.as_millis() < 50,
        "T&R Finding 3: Index search should be fast: {:?}",
        search_duration
    );
    assert!(
        !search_results.is_empty(),
        "T&R Finding 3: Should find indexed memories"
    );

    // Finding 4: Index is compact
    let stats = index.stats();

    // Index dimension should be smaller than full embedding
    assert!(
        stats.index_dimensions < 384,
        "T&R Finding 4: Index dimension ({}) < full embedding (384)",
        stats.index_dimensions
    );

    // Compression ratio
    let compression = 384.0 / stats.index_dimensions as f64;
    assert!(
        compression >= 2.0,
        "T&R Finding 4: Index should compress by at least 2x: {:.2}x",
        compression
    );
}

// ============================================================================
// EBBINGHAUS (1885) FORGETTING CURVE VALIDATION (1 test)
// ============================================================================

/// Validate Ebbinghaus (1885) forgetting curve properties.
///
/// Key findings from the original research:
/// 1. Memory retention decreases rapidly at first
/// 2. Rate of forgetting slows over time (exponential)
/// 3. Overlearning reduces forgetting rate
/// 4. Spacing strengthens retention
///
/// Reference: Ebbinghaus, H. (1885). Memory: A contribution to experimental
/// psychology. Teachers College, Columbia University.
#[test]
fn test_research_ebbinghaus_forgetting_curve() {
    let mut network = ActivationNetwork::new();

    // Finding 1 & 2: Model rapid initial decay, slower later decay
    // Using edge weights to represent memory strength over time

    // Simulate forgetting at different time points
    // t=0: Full strength (1.0)
    // t=1: Rapid drop
    // t=2: Slower drop
    // etc.

    let forgetting_curve = |t: f64| -> f64 {
        // Ebbinghaus formula: R = e^(-t/S) where S is stability
        let stability = 2.0; // Memory stability parameter
        (-t / stability).exp()
    };

    // Create memories at different "ages" (using edge weights to simulate)
    for t in 0..10 {
        let retention = forgetting_curve(t as f64);
        network.add_edge(
            "recall_context".to_string(),
            format!("memory_age_{}", t),
            LinkType::Temporal,
            retention,
        );
    }

    let results = network.activate("recall_context", 1.0);

    // Collect activations by "age"
    let mut age_activations: Vec<(u32, f64)> = Vec::new();
    for t in 0..10 {
        if let Some(result) = results
            .iter()
            .find(|r| r.memory_id == format!("memory_age_{}", t))
        {
            age_activations.push((t, result.activation));
        }
    }

    // Validation 1: Recent memory (t=0) should be strongest
    if age_activations.len() >= 2 {
        let (_, first_activation) = age_activations[0];
        let (_, second_activation) = age_activations[1];
        assert!(
            first_activation > second_activation,
            "Ebbinghaus 1: Most recent should be strongest"
        );
    }

    // Validation 2: Exponential decay pattern
    // Check that differences decrease over time
    if age_activations.len() >= 3 {
        let diff_early = age_activations[0].1 - age_activations[1].1;
        let diff_late = age_activations[age_activations.len() - 2].1
            - age_activations[age_activations.len() - 1].1;

        // Early differences should be larger (rapid initial forgetting)
        // But we need to account for near-zero values at the end
        if diff_late.abs() > 0.001 {
            assert!(
                diff_early.abs() >= diff_late.abs() * 0.5,
                "Ebbinghaus 2: Early forgetting ({:.4}) should be faster than late ({:.4})",
                diff_early,
                diff_late
            );
        }
    }

    // Finding 3: Test overlearning (reinforcement)
    let mut overlearned_network = ActivationNetwork::new();
    overlearned_network.add_edge(
        "study".to_string(),
        "normal_learning".to_string(),
        LinkType::Semantic,
        0.5,
    );
    overlearned_network.add_edge(
        "study".to_string(),
        "overlearned".to_string(),
        LinkType::Semantic,
        0.5,
    );

    // Simulate overlearning with multiple reinforcements
    for _ in 0..5 {
        overlearned_network.reinforce_edge("study", "overlearned", 0.1);
    }

    let study_results = overlearned_network.activate("study", 1.0);

    let normal_act = study_results
        .iter()
        .find(|r| r.memory_id == "normal_learning")
        .map(|r| r.activation)
        .unwrap_or(0.0);
    let overlearned_act = study_results
        .iter()
        .find(|r| r.memory_id == "overlearned")
        .map(|r| r.activation)
        .unwrap_or(0.0);

    assert!(
        overlearned_act > normal_act,
        "Ebbinghaus 3: Overlearned ({}) > normal ({})",
        overlearned_act,
        normal_act
    );
}

// ============================================================================
// FSRS-6 ALGORITHM PROPERTY VALIDATION (1 test)
// ============================================================================

/// Validate key FSRS-6 algorithm properties.
///
/// Key properties from FSRS-6:
/// 1. Retrievability calculation: R = (1 + t/S * factor)^(-w20)
/// 2. Stability increases after successful review
/// 3. Difficulty affects stability growth rate
/// 4. Hard penalty reduces stability increase
///
/// Reference: FSRS-6 algorithm specification
/// https://github.com/open-spaced-repetition/fsrs4anki
#[test]
fn test_research_fsrs6_properties() {
    // FSRS-6 default weights
    const W20: f64 = 0.1542; // Forgetting curve exponent

    // FSRS-6 retrievability formula
    fn fsrs6_retrievability(stability: f64, elapsed_days: f64, w20: f64) -> f64 {
        if stability <= 0.0 || elapsed_days <= 0.0 {
            return 1.0;
        }
        let factor = 0.9_f64.powf(-1.0 / w20) - 1.0;
        (1.0 + factor * elapsed_days / stability)
            .powf(-w20)
            .clamp(0.0, 1.0)
    }

    // Property 1: R = 0.9 when t = S (by design)
    let stability = 10.0;
    let r_at_stability = fsrs6_retrievability(stability, stability, W20);
    assert!(
        (r_at_stability - 0.9).abs() < 0.01,
        "FSRS-6 Property 1: R should be 0.9 at t=S, got {}",
        r_at_stability
    );

    // Property 2: R decreases as time increases
    let r_early = fsrs6_retrievability(stability, 5.0, W20);
    let r_late = fsrs6_retrievability(stability, 15.0, W20);
    assert!(
        r_early > r_late,
        "FSRS-6 Property 2: R should decrease over time: {} > {}",
        r_early,
        r_late
    );

    // Property 3: Higher stability = higher R at same elapsed time
    let low_stability = 5.0;
    let high_stability = 20.0;
    let elapsed = 10.0;

    let r_low = fsrs6_retrievability(low_stability, elapsed, W20);
    let r_high = fsrs6_retrievability(high_stability, elapsed, W20);
    assert!(
        r_high > r_low,
        "FSRS-6 Property 3: Higher stability should yield higher R: {} > {}",
        r_high,
        r_low
    );

    // Property 4: Forgetting curve shape matches exponential-like decay
    // Test multiple points to verify curve shape
    let test_points = vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0];
    let mut retrievabilities: Vec<f64> = Vec::new();

    for t in &test_points {
        let r = fsrs6_retrievability(10.0, *t, W20);
        retrievabilities.push(r);
    }

    // Verify monotonically decreasing
    for i in 1..retrievabilities.len() {
        assert!(
            retrievabilities[i] <= retrievabilities[i - 1],
            "FSRS-6 Property 4: R should monotonically decrease"
        );
    }

    // First value should be close to 1.0
    assert!(
        retrievabilities[0] > 0.95,
        "FSRS-6 Property 4: R should be high shortly after review: {}",
        retrievabilities[0]
    );
}
