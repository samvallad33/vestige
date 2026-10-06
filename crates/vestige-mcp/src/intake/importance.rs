//! Importance scoring for the ingest proof path (INGEST V5, Lane C).
//!
//! Pure function of bytes: no clocks, no RNG, no network, no storage, no
//! globals. Output is version-pinned by [`WEIGHTS_VERSION`]; any change to
//! the formula or its normalizations requires a new version string.
//!
//! Published formula (every weight named):
//!
//! ```text
//! score = clamp01( W_ENTITY_DENSITY  * entity_density       // 0.25
//!                + W_ENTROPY_NORM    * entropy_norm          // 0.20
//!                + W_LENGTH_BAND     * length_band           // 0.15
//!                + W_TYPE_TOKEN_NORM * type_token_norm       // 0.15
//!                + W_TAG_SIGNAL      * tag_signal            // 0.15
//!                + W_UNIQUE_ENTITY   * unique_entity_ratio ) // 0.10
//! ```
//!
//! Normalizations:
//!
//! * `entity_density  = min(1, entity_count / 16)`
//! * `entropy_norm    = clamp01((entropy_bits_per_byte - 3.0) / 5.0)`
//!   where `entropy_bits_per_byte` is the byte-histogram Shannon entropy of
//!   the content (ASCII text lands ~3-6 bits/byte).
//! * `length_band     = 1 - min(1, |ln(max(len,1)/512)| / ln(16))`
//!   (sweet spot 512 bytes, ±4x falloff).
//! * `type_token_norm = clamp01(type_token_ratio)`, where the ratio is
//!   unique words / total words over a whitespace split, lowercased.
//! * `tag_signal      = min(1, tag_count / 8)`
//! * `unique_entity_ratio = 0` when `entity_count == 0`, else
//!   `unique_entity_count / entity_count`; unique = distinct `(kind,
//!   surface)` pairs.
//!
//! [`score`] and [`recompute_from_factors`] are one implementation with two
//! names (byte-identical outputs by construction); responses can therefore
//! recompute a score from logged factors alone.

use std::collections::BTreeSet;

use serde_json::json;

use super::entities::EntitySpan;

const WEIGHTS_VERSION: &str = "linear-v1";

// Named weights of the published linear-v1 formula (sum = 1.0).
const W_ENTITY_DENSITY: f64 = 0.25;
const W_ENTROPY_NORM: f64 = 0.20;
const W_LENGTH_BAND: f64 = 0.15;
const W_TYPE_TOKEN_NORM: f64 = 0.15;
const W_TAG_SIGNAL: f64 = 0.15;
const W_UNIQUE_ENTITY: f64 = 0.10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImportanceFactors {
    pub length_bytes: usize,
    pub entity_count: usize,
    pub unique_entity_count: usize,
    pub tag_count: usize,
    pub entropy_bits_per_byte: f64,
    pub type_token_ratio: f64,
}

/// Compute the six importance factors from raw inputs.
///
/// Deterministic and pure; `unique_entity_count` counts distinct
/// `(kind, surface)` pairs (BTreeSet, so no iteration-order dependence).
fn compute_factors(content: &str, spans: &[EntitySpan], tags: &[String]) -> ImportanceFactors {
    let mut unique: BTreeSet<(&str, &str)> = BTreeSet::new();
    for span in spans {
        unique.insert((span.kind.as_str(), span.surface.as_str()));
    }
    ImportanceFactors {
        length_bytes: content.len(),
        entity_count: spans.len(),
        unique_entity_count: unique.len(),
        tag_count: tags.len(),
        entropy_bits_per_byte: shannon_entropy_bits_per_byte(content),
        type_token_ratio: type_token_ratio(content),
    }
}

/// The published linear-v1 score, clamped to `[0, 1]`.
pub fn score(f: &ImportanceFactors) -> f64 {
    recompute_from_factors(f)
}

/// Recompute the score from logged factors alone. This is THE single
/// implementation of the published formula; [`score`] is an alias for it,
/// so the two can never drift.
fn recompute_from_factors(f: &ImportanceFactors) -> f64 {
    let entity_density = (f.entity_count as f64 / 16.0).min(1.0);
    let entropy_norm = clamp01((f.entropy_bits_per_byte - 3.0) / 5.0);
    let length_band =
        1.0 - ((f.length_bytes.max(1) as f64 / 512.0).ln().abs() / 16f64.ln()).min(1.0);
    let type_token_norm = clamp01(f.type_token_ratio);
    let tag_signal = (f.tag_count as f64 / 8.0).min(1.0);
    let unique_entity_ratio = if f.entity_count == 0 {
        0.0
    } else {
        f.unique_entity_count as f64 / f.entity_count as f64
    };
    let raw = W_ENTITY_DENSITY * entity_density
        + W_ENTROPY_NORM * entropy_norm
        + W_LENGTH_BAND * length_band
        + W_TYPE_TOKEN_NORM * type_token_norm
        + W_TAG_SIGNAL * tag_signal
        + W_UNIQUE_ENTITY * unique_entity_ratio;
    clamp01(raw)
}

/// Compute factors and render the exact response shape:
/// `{score, formula, weightsVersion, factors}` where `factors` carries all
/// six fields and every f64 is rounded to 3 decimals via
/// `(x * 1000.0).round() / 1000.0`.
pub fn compute_and_format(
    content: &str,
    spans: &[EntitySpan],
    tags: &[String],
) -> serde_json::Value {
    let f = compute_factors(content, spans, tags);
    json!({
        "score": round3(recompute_from_factors(&f)),
        "formula": WEIGHTS_VERSION,
        "weightsVersion": WEIGHTS_VERSION,
        "factors": {
            "length_bytes": f.length_bytes,
            "entity_count": f.entity_count,
            "unique_entity_count": f.unique_entity_count,
            "tag_count": f.tag_count,
            "entropy_bits_per_byte": round3(f.entropy_bits_per_byte),
            "type_token_ratio": round3(f.type_token_ratio),
        }
    })
}

/// Byte-histogram Shannon entropy in bits per byte; 0.0 for empty input.
fn shannon_entropy_bits_per_byte(content: &str) -> f64 {
    let bytes = content.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut histogram = [0u64; 256];
    for &b in bytes {
        histogram[b as usize] += 1;
    }
    let total = bytes.len() as f64;
    let mut bits = 0.0;
    for &count in &histogram {
        if count == 0 {
            continue;
        }
        let p = count as f64 / total;
        bits -= p * p.log2();
    }
    bits
}

/// Unique words / total words over a whitespace split, lowercased; 0.0 for
/// empty input.
fn type_token_ratio(content: &str) -> f64 {
    let words: Vec<String> = content
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    if words.is_empty() {
        return 0.0;
    }
    let unique: BTreeSet<&str> = words.iter().map(|w| w.as_str()).collect();
    unique.len() as f64 / words.len() as f64
}

fn clamp01(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// 3-decimal rounding, the spec-pinned method: `(x * 1000.0).round() / 1000.0`.
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intake::entities::extract_typed_spans;

    const GOLDEN_CONTENT: &str = "alpha content with src/store.py and commit a1b2c3d";
    const GOLDEN_TAGS: &[&str] = &["fix", "store"];

    fn golden_tags() -> Vec<String> {
        GOLDEN_TAGS.iter().map(|t| t.to_string()).collect()
    }

    #[test]
    fn weights_version_is_pinned() {
        assert_eq!(WEIGHTS_VERSION, "linear-v1");
    }

    #[test]
    fn score_equals_recompute_from_factors() {
        let spans = extract_typed_spans(GOLDEN_CONTENT);
        let f = compute_factors(GOLDEN_CONTENT, &spans, &golden_tags());
        assert_eq!(score(&f), recompute_from_factors(&f));
        // Bit-identical, not just approximately equal.
        assert!(score(&f).total_cmp(&recompute_from_factors(&f)) == std::cmp::Ordering::Equal);
    }

    #[test]
    fn empty_content_factors_and_zero_score() {
        let f = compute_factors("", &[], &[]);
        assert_eq!(
            f,
            ImportanceFactors {
                length_bytes: 0,
                entity_count: 0,
                unique_entity_count: 0,
                tag_count: 0,
                entropy_bits_per_byte: 0.0,
                type_token_ratio: 0.0,
            }
        );
        assert_eq!(score(&f), 0.0);
        let v = compute_and_format("", &[], &[]);
        assert_eq!(v["score"].as_f64(), Some(0.0));
    }

    #[test]
    fn factors_are_all_finite() {
        for content in [
            "",
            "ascii only text 123",
            "重要 fix in 🚒 src/store.py",
            "aaaa",
        ] {
            let f = compute_factors(content, &[], &[]);
            assert!(f.entropy_bits_per_byte.is_finite(), "{content:?}");
            assert!(f.type_token_ratio.is_finite(), "{content:?}");
            let s = score(&f);
            assert!(s.is_finite(), "{content:?}");
            assert!((0.0..=1.0).contains(&s), "{content:?}");
        }
    }

    #[test]
    fn unique_entities_are_kind_surface_pairs() {
        let content = "a1b2c3d and again a1b2c3d";
        let spans = extract_typed_spans(content);
        assert_eq!(spans.len(), 2);
        let f = compute_factors(content, &spans, &[]);
        assert_eq!(f.entity_count, 2);
        assert_eq!(f.unique_entity_count, 1);
        // Same surface under a different kind would count as distinct; here
        // the identical pair collapses, so the ratio is exactly one half.
        assert_eq!(f.unique_entity_count as f64 / f.entity_count as f64, 0.5);
    }

    #[test]
    fn golden_factors_for_reference_content() {
        let spans = extract_typed_spans(GOLDEN_CONTENT);
        assert_eq!(spans.len(), 2);
        let f = compute_factors(GOLDEN_CONTENT, &spans, &golden_tags());
        assert_eq!(f.length_bytes, 50);
        assert_eq!(f.entity_count, 2);
        assert_eq!(f.unique_entity_count, 2);
        assert_eq!(f.tag_count, 2);
        assert_eq!(f.type_token_ratio, 1.0); // 7 words, all distinct
    }

    #[test]
    fn score_lives_in_open_unit_interval_for_real_content() {
        let spans = extract_typed_spans(GOLDEN_CONTENT);
        let f = compute_factors(GOLDEN_CONTENT, &spans, &golden_tags());
        let s = score(&f);
        assert!(s > 0.0 && s <= 1.0, "score {s}");
        let v = compute_and_format(GOLDEN_CONTENT, &spans, &golden_tags());
        let rendered = v["score"].as_f64().expect("score is a number");
        assert!(rendered > 0.0 && rendered <= 1.0, "rendered {rendered}");
    }

    #[test]
    fn output_shape_and_three_decimal_rounding() {
        let spans = extract_typed_spans(GOLDEN_CONTENT);
        let v = compute_and_format(GOLDEN_CONTENT, &spans, &golden_tags());
        assert_eq!(v["formula"], "linear-v1");
        assert_eq!(v["weightsVersion"], WEIGHTS_VERSION);
        let factors = &v["factors"];
        for key in [
            "length_bytes",
            "entity_count",
            "unique_entity_count",
            "tag_count",
        ] {
            assert!(factors[key].is_u64(), "{key}");
        }
        assert_eq!(factors["length_bytes"].as_u64(), Some(50));
        assert_eq!(factors["entity_count"].as_u64(), Some(2));
        assert_eq!(factors["unique_entity_count"].as_u64(), Some(2));
        assert_eq!(factors["tag_count"].as_u64(), Some(2));
        for key in ["entropy_bits_per_byte", "type_token_ratio"] {
            let x = factors[key].as_f64().expect(key);
            assert_eq!(round3(x), x, "{key} not 3-decimal: {x}");
        }
        let s = v["score"].as_f64().unwrap();
        assert_eq!(round3(s), s, "score not 3-decimal: {s}");
        // Exactly the four documented top-level keys.
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort();
        assert_eq!(
            sorted_keys,
            vec!["factors", "formula", "score", "weightsVersion"]
        );
    }

    #[test]
    fn deterministic_double_run() {
        let spans = extract_typed_spans(GOLDEN_CONTENT);
        let a = compute_and_format(GOLDEN_CONTENT, &spans, &golden_tags());
        let b = compute_and_format(GOLDEN_CONTENT, &spans, &golden_tags());
        assert_eq!(a, b);
        let f1 = compute_factors(GOLDEN_CONTENT, &spans, &golden_tags());
        let f2 = compute_factors(GOLDEN_CONTENT, &spans, &golden_tags());
        assert_eq!(f1, f2);
        assert_eq!(score(&f1), score(&f2));
    }

    // Golden vector (print-once pinned): the exact compute_and_format
    // output for the reference content "alpha content with src/store.py and
    // commit a1b2c3d" with tags ["fix","store"] and the spans extracted from
    // it (FilePath src/store.py + CommitSha a1b2c3d).
    #[test]
    fn golden_importance_vector() {
        let spans = extract_typed_spans(GOLDEN_CONTENT);
        let v = compute_and_format(GOLDEN_CONTENT, &spans, &golden_tags());
        let expected = json!({
            "score": 0.394,
            "formula": "linear-v1",
            "weightsVersion": "linear-v1",
            "factors": {
                "length_bytes": 50,
                "entity_count": 2,
                "unique_entity_count": 2,
                "tag_count": 2,
                "entropy_bits_per_byte": 4.271,
                "type_token_ratio": 1.0
            }
        });
        assert_eq!(v, expected, "golden drift: {v}");
        // The rendered score is exactly the 3-decimal rounding of the raw one.
        let f = compute_factors(GOLDEN_CONTENT, &spans, &golden_tags());
        assert_eq!(round3(score(&f)), 0.394);
    }
}
