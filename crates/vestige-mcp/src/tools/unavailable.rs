//! One shape for "this build cannot do that".
//!
//! A zero count says "looked and found none". It must never also mean "could
//! not look". An action, or one part of a larger report, that needs a
//! capability this build does not have says so with `status: "unavailable"`
//! and a stable `reason`, and carries no count or list a caller could misread
//! as an empty result.

use serde_json::{Value, json};

/// Needs the embedding runtime: similarity discovery, merge scoring, ...
pub const EMBEDDINGS_UNAVAILABLE: &str = "embeddings_unavailable";

/// Compares names or text, which a Strata log does not do.
pub const SIMILARITY_DISABLED: &str = "similarity_disabled";

/// Needs the synaptic tag store, which a Strata log does not record in 4.x.
pub const SYNAPTIC_CAPTURE_UNAVAILABLE: &str = "synaptic_capture_unavailable";

/// The `tagSuggestionStatus` of a save on a Strata log. Suggesting a close
/// existing tag means comparing tag names, so there is nothing to report and
/// nothing failed: tags are stored exactly as given.
pub fn tag_suggestions_on_strata(scope: &str) -> Value {
    let mut out = part(
        SIMILARITY_DISABLED,
        "Tag suggestions compare tag names, which a Strata log does not do. Tags are stored exactly as given.",
    );
    out["scope"] = Value::String(scope.to_string());
    out
}

/// The `synapticCapture` of a save on a Strata log. The write itself was
/// admitted and has its own receipt; only the tag-and-capture side effect is
/// absent. `durable` and `tagPersisted` stay, as `false`, for callers that
/// read them.
pub fn synaptic_capture_on_strata() -> Value {
    let mut out = part(
        SYNAPTIC_CAPTURE_UNAVAILABLE,
        "A Strata log records no synaptic tags or capture events in 4.x, so no capture is claimed. The write was admitted and has its own receipt.",
    );
    out["durable"] = Value::Bool(false);
    out["tagPersisted"] = Value::Bool(false);
    out
}

/// The refusal for a whole action a Strata log cannot honor in 4.0. Shared by
/// the server's withheld-action table and by the handlers it does not reach
/// (hidden aliases carry no `action` field), so every path says the same thing.
pub fn withheld_in_4_0(what: &str, why: &str) -> String {
    format!("unavailable_in_4_0: {what} is not available on Strata in Vestige 4.0: {why}.")
}

/// Why `maintain consolidate` is withheld on Strata.
pub const CONSOLIDATE_NOOP: &str = "every consolidate phase (all, lifecycle, logs) is a no-op on a Strata log, so nothing would be decayed, promoted, pruned, merged or embedded, and the zeros a no-op returns would read as a completed pass; maintain action='dream' replays recorded edges and folds one FSRS review per endpoint";

/// One part of a larger report that cannot be produced. It has no `count`,
/// `candidates` or other list, so it cannot be read as "none found".
pub fn part(reason: &str, detail: &str) -> Value {
    json!({
        "status": "unavailable",
        "reason": reason,
        "detail": detail,
    })
}

/// True when `value` is an unavailable action or part.
pub fn is_unavailable(value: &Value) -> bool {
    value["status"] == "unavailable"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_refusal_names_what_why_and_carries_the_stable_prefix() {
        let msg = withheld_in_4_0("maintain action 'consolidate'", "it is a no-op");
        assert_eq!(
            msg,
            "unavailable_in_4_0: maintain action 'consolidate' is not available on Strata in Vestige 4.0: it is a no-op."
        );
        assert!(CONSOLIDATE_NOOP.contains("maintain action='dream'"));
    }

    #[test]
    fn an_unavailable_part_carries_nothing_that_reads_as_zero() {
        let out = part(EMBEDDINGS_UNAVAILABLE, "scored by similarity");
        assert!(is_unavailable(&out));
        let object = out.as_object().unwrap();
        for forbidden in ["count", "candidates", "total", "items", "error"] {
            assert!(
                !object.contains_key(forbidden),
                "`{forbidden}` would let a caller read this as an empty result"
            );
        }
        assert!(!is_unavailable(&json!({"status": "completed"})));
    }

    #[test]
    fn the_strata_save_parts_are_typed_and_carry_no_error_field() {
        let tags = tag_suggestions_on_strata("user");
        assert!(is_unavailable(&tags));
        assert_eq!(tags["reason"], SIMILARITY_DISABLED);
        assert_eq!(tags["scope"], "user");

        let capture = synaptic_capture_on_strata();
        assert!(is_unavailable(&capture));
        assert_eq!(capture["reason"], SYNAPTIC_CAPTURE_UNAVAILABLE);
        assert_eq!(capture["durable"], false);
        assert_eq!(capture["tagPersisted"], false);

        for value in [&tags, &capture] {
            let text = value.to_string();
            assert!(
                value.get("error").is_none(),
                "a successful save must not carry an error field: {text}"
            );
            assert!(
                !text.contains("Initialization error") && !text.contains("not committed"),
                "the part must not read as a failure: {text}"
            );
        }
    }
}
