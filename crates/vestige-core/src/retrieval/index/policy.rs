//! Standing-policy, durable user directive, and preference-frame classification.

use crate::retrieval::*;


/// (policy_key, turn, session, ts, has_explicit_scope, scope_words, condition_words)
pub(in crate::retrieval) type RawStandingRule = (String, usize, usize, i64, bool, Vec<String>, Vec<String>);

/// Standing-policy shape from closed-class grammar only: a declarative (not a
/// question) whose first clause opens, after an optional "please", with a
/// frequency adverb or negated imperative ("always", "never", "don't",
/// "do not") followed by a base-form verb. The verb test is morphology (no
/// `-ed` past form), not a verb list; the verb must be followed by a noun
/// phrase, and the clause must carry at least two content tokens, so a bare
/// "never mind" or "don't know where" is not a rule.
pub fn utterance_is_standing_policy(text: &str) -> bool {
    let prose = text.split("```").next().unwrap_or(text).trim();
    if prose.is_empty() || prose.ends_with('?') {
        return false;
    }
    let lower = prose.to_lowercase();
    let trimmed = lower.trim_start();
    let trimmed = trimmed.strip_prefix("please ").map(str::trim_start).unwrap_or(trimmed);
    let rest = ["always ", "never ", "don't ", "don\u{2019}t ", "do not "]
        .iter()
        .find_map(|lead| trimmed.strip_prefix(lead));
    let Some(rest) = rest else { return false };
    let clause = rest.split(['.', '!', ';', ',', '\n']).next().unwrap_or(rest);
    let words: Vec<String> = raw_tokens(clause);
    let Some(verb) = words.first() else { return false };
    if verb.len() > 4 && verb.ends_with("ed") {
        return false;
    }
    // The verb must take a direct object: the next token opens a noun phrase
    // (a content word or a determiner), not a preposition, wh-word or subject
    // pronoun ("don't worry about", "don't know where", "don't think we").
    let Some(next) = words.get(1) else { return false };
    let opens_np = is_content(next) || DETERMINERS.contains(&next.as_str());
    opens_np && words.iter().filter(|w| is_content(w)).count() >= 2
}

// hardcode-scan: closed-class (compound personal preference modal grammar)
pub(in crate::retrieval) fn has_compound_preference_frame(text: &str) -> bool {
    let prose = text.split("```").next().unwrap_or(text).trim();
    if prose.is_empty() {
        return false;
    }
    let lower = prose.to_lowercase();
    let hit = [
        "i prefer ", "i'd prefer ", "i\u{2019}d prefer ",
        "i'd like to ", "i\u{2019}d like to ", "i'd rather ", "i\u{2019}d rather ",
        "my favorite ", "i really like ", "i enjoy ",
    ]
    .iter()
    .any(|pat| {
        lower.starts_with(pat)
            || lower.contains(&format!(". {pat}"))
            || lower.contains(&format!(", {pat}"))
            || lower.contains(&format!("; {pat}"))
            || lower.contains(&format!("so {pat}"))
            || lower.contains(&format!("and {pat}"))
            || lower.contains(&format!("but {pat}"))
    });
    hit && raw_tokens(prose).iter().filter(|w| is_content(w)).count() >= 3
}

// hardcode-scan: closed-class (Halakha-ke comparative contrast prepositions)
pub(in crate::retrieval) fn has_comparative_choice_frame(text: &str) -> bool {
    let prose = text.split("```").next().unwrap_or(text).trim();
    if prose.is_empty() || prose.ends_with('?') {
        return false;
    }
    let lower = prose.to_lowercase();
    let has_first_person = lower.starts_with("i ")
        || lower.starts_with("i'")
        || lower.starts_with("i\u{2019}")
        || lower.starts_with("my ")
        || lower.contains(" i ")
        || lower.contains(" i'")
        || lower.contains(" i\u{2019}")
        || lower.contains(" my ");
    let has_contrast = lower.contains(" rather than ")
        || lower.contains(" instead of ")
        || lower.contains(" prefer ");
    has_first_person
        && has_contrast
        && raw_tokens(prose).iter().filter(|w| is_content(w)).count() >= 3
}

/// [W2-028/031] Closed-class durable user-directive detector (superset of `utterance_is_standing_policy`
/// catching first-person modal/negation constraints mid-utterance such as "I never...", "From now on...").
// hardcode-scan: closed-class (durable user directive grammar)
pub(in crate::retrieval) fn is_durable_user_directive(text: &str) -> bool {
    if utterance_is_standing_policy(text) {
        return true;
    }
    let prose = text.split("```").next().unwrap_or(text).trim();
    if prose.is_empty() {
        return false;
    }
    let lower = prose.to_lowercase();
    let has_compound_pref = has_compound_preference_frame(prose);
    if prose.ends_with('?') && !has_compound_pref {
        return false;
    }
    let has_trigger = has_compound_pref
        || [
            "i always ", "i never ", "i cannot ", "i can't ", "i can\u{2019}t ",
            "i won't ", "i won\u{2019}t ", "i must ", "i don't ", "i don\u{2019}t ",
            "we always ", "we never ", "from now on ", "please always ", "please never ",
        ]
        .iter()
        .any(|pat| {
            lower.starts_with(pat)
                || lower.contains(&format!(". {pat}"))
                || lower.contains(&format!(", {pat}"))
                || lower.contains(&format!("; {pat}"))
        });
    has_trigger && raw_tokens(prose).iter().filter(|w| is_content(w)).count() >= 3
}

/// Closed-class English determiners (articles, demonstratives, possessives, quantifiers).
pub(in crate::retrieval) const DETERMINERS: &[&str] =
    // hardcode-scan: closed-class (determiners)
    &["a", "an", "the", "this", "that", "these", "those", "my", "your", "our", "their", "its", "his", "her", "every", "each", "all", "any", "no", "some"];

/// Scope words are the turn's own content tokens. No directive-verb list.
pub(in crate::retrieval) fn extract_rule_scope(text: &str) -> (bool, Vec<String>) {
    let (prose, _) = split_blip_caption(text);
    (
        false,
        raw_tokens(prose).into_iter().filter(|w| is_content(w)).collect(),
    )
}

/// Condition clause of a standing rule ("Always X when I ask about Y" -> Y).
/// Splits the imperative into action + condition at the first closed-class
/// subordinator, then (if present) after the first closed-class preposition, so
/// only the topic the rule is conditioned on can gate rule promotion.
/// Rules with no subordinate condition return an empty list (unconditional).
pub(in crate::retrieval) fn extract_rule_condition(text: &str) -> Vec<String> {
    let (prose, _) = split_blip_caption(text);
    let first = prose.split("->").next().unwrap_or(prose);
    let first = first.split(['.', '!', ';', '\n']).next().unwrap_or(first);
    let toks = raw_tokens(first);
    // hardcode-scan: closed-class (subordinating conjunctions)
    const SUBORD: &[&str] = &["when", "whenever", "if", "while", "during", "once"];
    // hardcode-scan: closed-class (topic prepositions)
    const TOPIC_PREP: &[&str] = &["about", "regarding", "on", "for", "with", "of", "concerning", "involving"];
    let Some(sp) = toks.iter().position(|w| SUBORD.contains(&w.as_str())) else {
        return Vec::new();
    };
    let cond = &toks[sp + 1..];
    let cond = match cond.iter().position(|w| TOPIC_PREP.contains(&w.as_str())) {
        Some(pp) if pp + 1 < cond.len() => &cond[pp + 1..],
        _ => cond,
    };
    cond.iter().filter(|w| is_content(w)).cloned().collect()
}


/// Stable plan id for one normalized standing-policy utterance.
pub fn standing_policy_id(text: &str) -> String {
    let norm = text.trim().to_lowercase();
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in norm.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("sp-{hash:016x}")
}
