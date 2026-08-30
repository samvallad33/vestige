//! Shared contradiction detection for the write path and the retrieval path.
//!
//! This module exists because the two paths had *separate* implementations of
//! the same concept and they drifted. The retrieval side (`cross_reference`)
//! grew negation symmetry, antonym pairs and a divergence test; the write side
//! (`prediction_error`) kept a single directional negation scan. The result was
//! that `smart_ingest` could not see a correction that retrieval could see
//! perfectly well — and the write path is the one that decides whether the
//! dissenting memory is *stored at all*. Retrieval-side protection cannot
//! protect a memory that was destroyed one stage earlier.
//!
//! There is now one implementation. The only legitimate difference between the
//! two call sites is how each establishes that the two texts are *about the
//! same subject*, which [`SubjectIdentity`] names explicitly.

/// Explicit polarity flips: one text takes a negative stance, the other the
/// paired positive one.
///
/// Bare triggers with no paired positive term ("not ", "instead of", "rather
/// than") are deliberately absent: they match ordinary complementary phrasing
/// and made benign additive notes ("Do not forget to configure X") read as
/// corrections.
const NEGATION_PAIRS: &[(&str, &str)] = &[
    ("don't", "do"),
    ("never", "always"),
    ("avoid", "use"),
    ("wrong", "right"),
    ("incorrect", "correct"),
    ("deprecated", "recommended"),
    ("outdated", "current"),
    ("removed", "added"),
    ("disabled", "enabled"),
];

/// Opposite-direction claims with NO negation word in either text.
///
/// Requiring both sides to be present, in opposite texts, is a far stricter
/// test than asymmetric presence, and it catches the class the negation scan
/// structurally cannot: "prompt diversity HURTS accuracy" against "prompt
/// diversity IMPROVES accuracy" contains no negation at all.
const ANTONYM_PAIRS: &[(&str, &str)] = &[
    ("hurts", "improves"),
    ("hurt", "improved"),
    ("degrades", "enhances"),
    ("decreases", "increases"),
    ("reduces", "raises"),
    ("worse", "better"),
    ("slower", "faster"),
    ("fails", "succeeds"),
    ("breaks", "fixes"),
    ("unsafe", "safe"),
    ("unstable", "stable"),
    ("unsupported", "supported"),
];

/// Phrases that mark one text as explicitly revising the other.
const CORRECTION_SIGNALS: &[&str] = &[
    "actually",
    "correction",
    "update:",
    "updated:",
    "fixed",
    "was wrong",
    "changed to",
    "now uses",
    "replaced by",
    "superseded",
    "no longer",
    "instead of",
    "switched to",
    "migrated to",
];

/// How the caller has established that two texts are about the same subject.
///
/// This is the *only* thing that legitimately differs between the write path
/// and the retrieval path, so it is a parameter rather than a second copy of
/// the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubjectIdentity {
    /// Nothing is known beyond the text itself, so subject identity must be
    /// inferred from shared substantive vocabulary. Used at retrieval time,
    /// where a pair is drawn from thousands of unrelated memories and a false
    /// positive is expensive.
    FromTextOverlap,
    /// The caller has already proven the two texts are about the same subject
    /// by other means — at write time the candidate arrives from an embedding
    /// similarity gate, so requiring lexical overlap *as well* would discard
    /// exactly the corrections that are short and reworded.
    AlreadyEstablished,
}

impl SubjectIdentity {
    /// Shared substantive words required before any shape is considered.
    fn min_shared_words(self) -> usize {
        match self {
            Self::FromTextOverlap => 4,
            Self::AlreadyEstablished => 0,
        }
    }

    /// Shared substantive words required before a correction *marker* counts.
    /// Higher than the base floor because these phrases appear in plenty of
    /// memories that revise nothing.
    fn min_shared_words_for_correction_signal(self) -> usize {
        match self {
            Self::FromTextOverlap => 6,
            Self::AlreadyEstablished => 0,
        }
    }
}

fn substantive_words(lowered: &str) -> std::collections::HashSet<&str> {
    lowered.split_whitespace().filter(|w| w.len() > 3).collect()
}

/// Tokens used for the divergence test.
///
/// Unlike [`substantive_words`], short tokens containing a digit are kept. A
/// version conflict ("PostgreSQL 14" against "PostgreSQL 16") puts the entire
/// discriminating signal in a two-character token, so a length filter alone
/// makes the two texts look identical and the conflict invisible.
fn divergence_tokens(lowered: &str) -> std::collections::HashSet<String> {
    lowered
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() > 3 || w.chars().any(|c| c.is_ascii_digit()))
        .filter(|w| !w.is_empty())
        .collect()
}

/// Do these two texts appear to assert incompatible things?
///
/// Symmetric in `a` and `b`: a correction is a correction regardless of which
/// side arrived first. The write path previously tested only "is the NEW text
/// the negative one", so ingesting "Always use X" over a stored "Never use X"
/// was read as agreement and *strengthened the claim being corrected*.
pub fn appears_contradictory(a: &str, b: &str, identity: SubjectIdentity) -> bool {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();

    let a_words = substantive_words(&a_lower);
    let b_words = substantive_words(&b_lower);
    let shared_words = a_words.intersection(&b_words).count();

    if shared_words < identity.min_shared_words() {
        return false;
    }

    // Polarity flip: one side carries a negative stance the other lacks.
    for (neg, _opp) in NEGATION_PAIRS {
        if (a_lower.contains(neg) && !b_lower.contains(neg))
            || (b_lower.contains(neg) && !a_lower.contains(neg))
        {
            return true;
        }
    }

    // Opposite directions, no negation in either.
    for (neg, opp) in ANTONYM_PAIRS {
        let a_neg = a_lower.contains(neg);
        let b_neg = b_lower.contains(neg);
        let a_opp = a_lower.contains(opp);
        let b_opp = b_lower.contains(opp);
        if (a_neg && b_opp && !b_neg && !a_opp) || (b_neg && a_opp && !a_neg && !b_opp) {
            return true;
        }
    }

    // Mutually exclusive VALUES for the same attribute — the most common
    // real-world contradiction, which neither test above can see. "Priya holds
    // a Bachelor degree" against "Priya holds a Master of Science degree"
    // contains no negation and no antonym, yet both cannot be true.
    //
    // The distinguishing signal is DIVERGENCE versus ELABORATION. If the texts
    // overlap heavily but NEITHER's vocabulary is a subset of the other's, they
    // are asserting different things in the same slot. If one IS a subset, it
    // is an elaboration ("works in Leeds" versus "works in Leeds and London")
    // and must not be flagged.
    {
        let a_tokens = divergence_tokens(&a_lower);
        let b_tokens = divergence_tokens(&b_lower);
        let shared = a_tokens.intersection(&b_tokens).count();
        let union = a_tokens.union(&b_tokens).count();
        let overlap = if union == 0 {
            0.0
        } else {
            shared as f32 / union as f32
        };
        let a_only = a_tokens.difference(&b_tokens).count();
        let b_only = b_tokens.difference(&a_tokens).count();
        // Both sides distinctive => divergence, not elaboration.
        let diverges = a_only > 0 && b_only > 0;
        // Keep the divergence small relative to the shared core: two memories
        // that merely touch the same topic diverge in many terms, whereas a
        // same-attribute conflict differs in only a few.
        let small_divergence = (a_only + b_only) <= shared;
        if overlap >= 0.6 && diverges && small_divergence {
            return true;
        }
    }

    // Explicit revision marker, present in exactly one side.
    if shared_words >= identity.min_shared_words_for_correction_signal() {
        for signal in CORRECTION_SIGNALS {
            if a_lower.contains(signal) != b_lower.contains(signal) {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four shapes measured against the real binary, every one of which the
    /// old write-path detector read as agreement and reinforced. Similarities
    /// are the values observed through the live embedding service, all above
    /// the 0.92 near-identical threshold, which is why they short-circuited.
    #[test]
    fn write_path_sees_every_measured_correction_shape() {
        let established = SubjectIdentity::AlreadyEstablished;
        let cases = [
            (
                "Always use prompt diversity for AIMO submissions",
                "Never use prompt diversity for AIMO submissions",
                "negation flip, positive arriving second (0.965)",
            ),
            (
                "Prompt diversity improves accuracy at high temperature",
                "Prompt diversity hurts accuracy at high temperature",
                "antonym, no negation in either (0.962)",
            ),
            (
                "The production database runs PostgreSQL 16",
                "The production database runs PostgreSQL 14",
                "mutually exclusive version values (0.940)",
            ),
            (
                "Priya holds a Master degree in computer science from Leeds",
                "Priya holds a Bachelor degree in computer science from Leeds",
                "mutually exclusive attribute values (0.976)",
            ),
        ];
        for (new, old, shape) in cases {
            assert!(
                appears_contradictory(new, old, established),
                "must detect {shape}"
            );
            assert!(
                appears_contradictory(old, new, established),
                "must be symmetric: {shape}"
            );
        }
    }

    /// The reason the write-path detector was narrow in the first place. These
    /// must keep NOT firing, or ingestion starts demoting correct memories.
    #[test]
    fn benign_additive_notes_are_not_corrections() {
        let established = SubjectIdentity::AlreadyEstablished;
        let cases = [
            (
                "Do not forget to configure the async runtime for the worker pool",
                "Use the async runtime for the worker pool",
            ),
            (
                "You cannot skip the migration step",
                "Run the migration step before deploying",
            ),
            (
                "Use async/await for performance",
                "Use async patterns when needed",
            ),
            // Strict elaboration: one side's vocabulary is a subset of the other's.
            (
                "Vestige uses FSRS-6 with 21 parameters for scheduling",
                "Vestige uses FSRS-6 for scheduling",
            ),
        ];
        for (a, b) in cases {
            assert!(
                !appears_contradictory(a, b, established),
                "additive/elaborating note must not read as a contradiction: {a:?}"
            );
            assert!(
                !appears_contradictory(b, a, established),
                "…in either order: {b:?}"
            );
        }
    }

    /// Retrieval draws pairs from thousands of unrelated memories, so it still
    /// requires lexical evidence that the two are about the same subject. The
    /// write path has that from the embedding gate and must not re-require it.
    #[test]
    fn subject_identity_gate_differs_between_the_two_call_sites() {
        let short_correction = (
            "Actually, the correct approach is Redis",
            "The approach is Redis",
        );
        assert!(
            appears_contradictory(
                short_correction.0,
                short_correction.1,
                SubjectIdentity::AlreadyEstablished
            ),
            "write path: embedding already established subject identity"
        );
        assert!(
            !appears_contradictory(
                short_correction.0,
                short_correction.1,
                SubjectIdentity::FromTextOverlap
            ),
            "retrieval path: too little shared vocabulary to risk a false positive"
        );
    }

    /// Unrelated memories that merely share a topic word must never pair up.
    #[test]
    fn unrelated_memories_are_not_contradictions() {
        for identity in [
            SubjectIdentity::FromTextOverlap,
            SubjectIdentity::AlreadyEstablished,
        ] {
            assert!(!appears_contradictory(
                "FSRS-6 upgrade research sources for the scheduler rewrite",
                "The dashboard renders the memory graph with WebGPU particles",
                identity
            ));
        }
    }
}
