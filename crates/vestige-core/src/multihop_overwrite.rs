//! Step 2: Deterministic zero-float nominal terminal promotion.
//!
//! Pioneers used by the rule that ships:
//! - Patrick Blackburn: the speaker field is a nominal anchor, not a word list.
//! - Richard J. Wallace: promote the unique terminal hop already inside the top 5.
//! - Elizabeth Ligon Bjork: a terminal question yields its slot; it is not deleted.
//!
//! HARD INVARIANT: this module contains no floating-point values.
//!
//! The question that ends on `?` is the carrier. When the top 5 holds exactly
//! one later turn that does not end on `?`, spoken by a different nominal the
//! query names, that turn takes slot 0. Top-5 membership is unchanged. Role
//! labels `user` and `assistant` are not nominals. Two competing nominals do
//! not fire. Ranking reads only this index plus the query text.

/// Per-turn index built at ingest. `terminal_question[i]` is true when turn
/// `i`'s prose, before a fenced block, ends on `?`. `speaker[i]` is the role
/// field lowercased at ingest.
pub(crate) struct MultiHopIndex {
    pub(crate) terminal_question: Vec<bool>,
    pub(crate) speaker: Vec<String>,
}

impl MultiHopIndex {
    pub(crate) fn new(terminal_question: Vec<bool>, speaker: Vec<String>) -> Self {
        Self {
            terminal_question,
            speaker,
        }
    }
}

/// Orthographic terminal question. A `?` earlier in the prose does not count,
/// and a trailing code fence is ignored. This is stricter than `is_question`.
pub(crate) fn terminal_question(text: &str) -> bool {
    let prose = text.split("```").next().unwrap_or(text);
    prose.trim_end().ends_with('?')
}

fn is_role_label(speaker: &str) -> bool {
    speaker.eq_ignore_ascii_case("user") || speaker.eq_ignore_ascii_case("assistant")
}

fn query_names_nominal(query: &str, speaker_lower: &str) -> bool {
    if speaker_lower.is_empty() || is_role_label(speaker_lower) {
        return false;
    }
    let bytes = query.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && !is_nominal_byte(bytes[i]) {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && is_nominal_byte(bytes[i]) {
            i += 1;
        }
        if start == i {
            break;
        }
        let mut end = i;
        if end >= start + 2 && bytes[end - 2] == b'\'' && bytes[end - 1].eq_ignore_ascii_case(&b's')
        {
            end -= 2;
        } else if end > start && bytes[end - 1] == b'\'' {
            end -= 1;
        }
        if query[start..end].eq_ignore_ascii_case(speaker_lower) {
            return true;
        }
    }
    false
}

fn is_nominal_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'\''
}

/// Move the unique named declarative in slots `1..5` to slot 0 when slot 0 is
/// a terminal question spoken by someone else. Returns whether the list changed.
/// Does not change list length or top-5 membership.
pub(crate) fn seat_nominal_terminal(
    ix: &MultiHopIndex,
    query: &str,
    list: &mut Vec<usize>,
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let head = list[0];
    if !ix.terminal_question.get(head).copied().unwrap_or(false) {
        return false;
    }
    let head_speaker = ix.speaker.get(head).map(String::as_str).unwrap_or("");
    let mut chosen: Option<usize> = None;
    let window = list.len().min(5);
    for &ti in list.iter().take(window).skip(1) {
        if ix.terminal_question.get(ti).copied().unwrap_or(false) {
            continue;
        }
        let speaker = match ix.speaker.get(ti).map(String::as_str) {
            Some(s) => s,
            None => continue,
        };
        if speaker.eq_ignore_ascii_case(head_speaker) || !query_names_nominal(query, speaker) {
            continue;
        }
        if chosen.is_some() {
            return false;
        }
        chosen = Some(ti);
    }
    let Some(ti) = chosen else {
        return false;
    };
    let Some(pos) = list.iter().position(|&t| t == ti) else {
        return false;
    };
    if pos == 0 || pos >= 5 {
        return false;
    }
    list.remove(pos);
    list.insert(0, ti);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(rows: &[(&str, bool)]) -> MultiHopIndex {
        MultiHopIndex::new(
            rows.iter().map(|(_, q)| *q).collect(),
            rows.iter().map(|(s, _)| (*s).to_string()).collect(),
        )
    }

    #[test]
    fn seats_the_unique_named_declarative_over_a_terminal_question() {
        let ix = index(&[("caroline", true), ("melanie", false), ("caroline", false)]);
        let mut list = vec![0usize, 1, 2];
        assert!(seat_nominal_terminal(
            &ix,
            "What activities does Melanie partake in?",
            &mut list
        ));
        assert_eq!(list, vec![1, 0, 2]);
    }

    #[test]
    fn possessive_nominal_still_matches_and_role_labels_do_not() {
        let ix = index(&[("assistant", true), ("user", false)]);
        let mut list = vec![0usize, 1];
        assert!(!seat_nominal_terminal(
            &ix,
            "What did the user decide?",
            &mut list
        ));
        assert_eq!(list, vec![0, 1]);

        let ix = index(&[("caroline", true), ("melanie", false)]);
        let mut list = vec![0usize, 1];
        assert!(seat_nominal_terminal(
            &ix,
            "What is Melanie's favorite class?",
            &mut list
        ));
        assert_eq!(list, vec![1, 0]);
    }

    #[test]
    fn two_nominals_and_a_non_terminal_head_stay_put() {
        let ix = index(&[("caroline", true), ("melanie", false), ("jon", false)]);
        let mut list = vec![0usize, 1, 2];
        assert!(!seat_nominal_terminal(
            &ix,
            "What did Melanie and Jon plan?",
            &mut list
        ));
        assert_eq!(list, vec![0, 1, 2]);

        let ix = index(&[("caroline", false), ("melanie", false)]);
        let mut list = vec![0usize, 1];
        assert!(!seat_nominal_terminal(
            &ix,
            "What activities does Melanie partake in?",
            &mut list
        ));
        assert_eq!(list, vec![0, 1]);
    }

    #[test]
    fn terminal_question_rejects_an_internal_mark() {
        assert!(terminal_question("Do you have any other ways?"));
        assert!(terminal_question(
            "Can you initialize SQLite?\n```python\nprint(1)\n```"
        ));
        assert!(!terminal_question(
            "I passed. What's next is the interview."
        ));
        assert!(!terminal_question("No question here."));
    }
}
