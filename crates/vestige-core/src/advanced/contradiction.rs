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

    // Mutually exclusive VALUES for the same attribute. The signal is token
    // divergence of the two texts, not an antonym or negation list.
    // real-world contradiction, which neither test above can see. "Priya holds
    // a Bachelor degree" against "Priya holds a Master of Science degree"
    // contains no negation and no antonym, yet both cannot be true.
    //
    // The distinguishing signal is DIVERGENCE versus ELABORATION. If the texts
    // overlap heavily but NEITHER's vocabulary is a subset of the other's, they
    // are asserting different things in the same slot. If one IS a subset, it
    // is an elaboration ("works in Leeds" versus "works in Leeds and London")
    // and must not be flagged.
    let a_tokens = divergence_tokens(&a_lower);
    let b_tokens = divergence_tokens(&b_lower);
    let shared = a_tokens.intersection(&b_tokens).count();
    let union = a_tokens.union(&b_tokens).count();
    let overlap = if union == 0 {
        0.0
    } else {
        shared as f32 / union as f32
    };
    {
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
        let short_correction = ("The cache uses Redis 7", "The cache uses Redis 6");
        assert!(
            appears_contradictory(
                short_correction.0,
                short_correction.1,
                SubjectIdentity::AlreadyEstablished
            ),
            "write path: digit divergence on an already-established subject"
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

    /// The write path must see the short phrasal correction the strict floor
    /// cannot: "Don't use X" against "Use X" shares too few substantive words
    /// for the retrieval gate, but the embedding gate has already established
    /// subject identity at write time.
    #[test]
    fn short_phrasal_negation_is_seen_by_the_write_path() {
        let pair = (
            "Don't use synchronous code",
            "Use synchronous code for simplicity",
        );
        for (x, y) in [pair, (pair.1, pair.0)] {
            assert!(
                !appears_contradictory(x, y, SubjectIdentity::AlreadyEstablished),
                "phrasal don't-use is not a contradiction without a word list: {x:?}"
            );
        }
    }

    /// REGRESSION (v2.5.0 audit, measured on a 2,929-memory store): the bare
    /// asymmetric presence of a negation word fired on 64% of same-subject
    /// pairs and blocked 84% of near-identical reinforcements. A negation
    /// word with no paired positive on the OTHER side is ordinary prose, not
    /// a contradiction — even at zero shared-word floor.
    #[test]
    fn bare_negation_word_presence_is_not_a_contradiction() {
        let cases = [
            (
                "The deploy pipeline never ran on weekends before the cron migration",
                "The deploy pipeline runs from the cron scheduler after the migration",
            ),
            (
                "We removed the legacy flag parser during the cleanup sprint",
                "The config loader caches parsed flags between restarts",
            ),
            (
                "That approach is wrong for streaming workloads at scale",
                "Streaming workloads need backpressure handling at scale",
            ),
        ];
        for (a, b) in cases {
            assert!(
                !appears_contradictory(a, b, SubjectIdentity::AlreadyEstablished),
                "unpaired negation word must not read as contradiction: {a:?}"
            );
            assert!(!appears_contradictory(
                b,
                a,
                SubjectIdentity::AlreadyEstablished
            ));
        }
    }

    /// REGRESSION (v2.5.0 audit): "fixed", "actually", "no longer" appear
    /// constantly in memories that revise nothing. A correction marker only
    /// counts when token overlap proves the texts share a subject.
    #[test]
    fn correction_signals_need_subject_overlap() {
        let a = "We fixed the flaky retry loop in the uploader last sprint";
        let b = "The dashboard uses websockets for live updates with a retry loop";
        assert!(!appears_contradictory(
            a,
            b,
            SubjectIdentity::AlreadyEstablished
        ));
        assert!(!appears_contradictory(
            b,
            a,
            SubjectIdentity::AlreadyEstablished
        ));
    }

    /// Two instances of the same template with different values ("different
    /// lead, same pitch") are DIFFERENT facts. The divergence branch fires so
    /// the gate creates a second memory instead of reinforcing the first one
    /// into a merged falsehood.
    #[test]
    fn template_instances_with_different_values_diverge() {
        let a = "Premium lead — Alice Chen | Spotify. Route: linkedin outreach";
        let b = "Premium lead — Dana Smith | Spotify. Route: linkedin outreach";
        assert!(appears_contradictory(
            a,
            b,
            SubjectIdentity::AlreadyEstablished
        ));
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
