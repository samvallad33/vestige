//! Exact longest common token run, via a suffix automaton.
//!
//! The construction is Blumer et al. (1985): the smallest deterministic
//! automaton whose language is the set of substrings of a token string, at
//! most `2n − 1` states. Transitions are extended online with the clone split
//! from Crochemore, "Transducers and repetitions", Theoretical Computer
//! Science 45 (1986). The longest common substring is the same walk as
//! cp-algorithms (suffix automaton, "Finding the longest common substring of
//! two strings") and `oxicuda_seq::string::suffix_automaton::longest_common_substring`:
//! build the automaton on one string, stream the other, and follow suffix
//! links on a mismatch. `general-sam` is the same automaton over a generic
//! alphabet; the alphabet here is an exact token id, not a byte, so a match
//! cannot end in the middle of a token.
//!
//! Longest common substring length is symmetric, so the automaton is built on
//! the query and each stored turn is streamed through it. That is the same
//! length as building the automaton on the turn and streaming the query.

use std::collections::HashMap;

use crate::suffix_automaton_guard::SUFFIX_BELOW_WINDOW;

struct SamState {
    len: u32,
    link: i32,
    next: HashMap<u32, u32>,
}

/// Suffix automaton over exact token ids.
pub struct SuffixAutomaton {
    states: Vec<SamState>,
    last: u32,
}

impl SuffixAutomaton {
    pub fn build(tokens: &[u32]) -> Self {
        let mut sam = Self {
            states: vec![SamState { len: 0, link: -1, next: HashMap::new() }],
            last: 0,
        };
        for &tok in tokens {
            sam.extend(tok);
        }
        sam
    }

    fn extend(&mut self, c: u32) {
        let cur = self.states.len() as u32;
        let cur_len = self.states[self.last as usize].len + 1;
        self.states.push(SamState { len: cur_len, link: 0, next: HashMap::new() });
        let mut p = self.last as i32;
        while p != -1 && !self.states[p as usize].next.contains_key(&c) {
            self.states[p as usize].next.insert(c, cur);
            p = self.states[p as usize].link;
        }
        if p == -1 {
            self.states[cur as usize].link = 0;
        } else {
            let q = self.states[p as usize].next[&c];
            if self.states[p as usize].len + 1 == self.states[q as usize].len {
                self.states[cur as usize].link = q as i32;
            } else {
                let clone = self.states.len() as u32;
                let clone_state = SamState {
                    len: self.states[p as usize].len + 1,
                    link: self.states[q as usize].link,
                    next: self.states[q as usize].next.clone(),
                };
                self.states.push(clone_state);
                while p != -1 && self.states[p as usize].next.get(&c) == Some(&q) {
                    self.states[p as usize].next.insert(c, clone);
                    p = self.states[p as usize].link;
                }
                self.states[q as usize].link = clone as i32;
                self.states[cur as usize].link = clone as i32;
            }
        }
        self.last = cur;
    }

    /// Longest prefix of a suffix of `needle` that the automaton accepts.
    pub fn longest_match(&self, needle: &[u32]) -> u32 {
        let mut v: i32 = 0;
        let mut matched: u32 = 0;
        let mut best: u32 = 0;
        for &c in needle {
            while v != 0 && !self.states[v as usize].next.contains_key(&c) {
                v = self.states[v as usize].link;
                matched = self.states[v as usize].len;
            }
            if let Some(&to) = self.states[v as usize].next.get(&c) {
                v = to as i32;
                matched += 1;
                if matched > best {
                    best = matched;
                }
            }
        }
        best
    }
}

/// Longest run of identical token ids shared by `left` and `right`.
///
/// The ranker builds one automaton on the query and streams every turn, which
/// is this length. The function is the symmetric pair form the tests check.
#[cfg_attr(not(test), allow(dead_code))]
pub fn longest_common_token_run(left: &[u32], right: &[u32]) -> u32 {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    // Build on the shorter string. Length is symmetric (Crochemore).
    if left.len() <= right.len() {
        SuffixAutomaton::build(left).longest_match(right)
    } else {
        SuffixAutomaton::build(right).longest_match(left)
    }
}

/// Case-folded exact tokens. No stemming and no word list.
///
/// A token is a maximal alphanumeric run. `.`, `-`, and `/` stay inside the
/// token only when both neighbours are alphanumeric, so `5.3.0`, `zx-42`, and
/// `a/b` stay whole and a sentence period splits. A comma between digits is a
/// thousands separator (`1,200` and `1200` are the same digit string). An
/// apostrophe is dropped (`don't` → `dont`). An `@` plus the following handle
/// characters is one token.
pub fn exact_tokens(text: &str) -> Vec<String> {
    let folded = text.to_lowercase();
    let chars: Vec<char> = folded.chars().collect();
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    let flush = |buf: &mut String, out: &mut Vec<String>| {
        if !buf.is_empty() {
            out.push(std::mem::take(buf));
        }
    };
    while i < chars.len() {
        let c = chars[i];
        if c == '@' {
            flush(&mut buf, &mut out);
            let mut handle = String::from("@");
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                handle.push(chars[i]);
                i += 1;
            }
            if handle.len() > 1 {
                out.push(handle);
            }
            continue;
        }
        if c == ','
            && !buf.is_empty()
            && buf.chars().last().is_some_and(|d| d.is_ascii_digit())
            && i + 1 < chars.len()
            && chars[i + 1].is_ascii_digit()
        {
            i += 1;
            continue;
        }
        if c == '\'' || c == '’' {
            i += 1;
            continue;
        }
        if matches!(c, '.' | '-' | '/')
            && !buf.is_empty()
            && buf.chars().last().is_some_and(|d| d.is_alphanumeric())
            && i + 1 < chars.len()
            && chars[i + 1].is_alphanumeric()
        {
            buf.push(c);
            i += 1;
            continue;
        }
        if c.is_alphanumeric() {
            buf.push(c);
            i += 1;
            continue;
        }
        flush(&mut buf, &mut out);
        i += 1;
    }
    flush(&mut buf, &mut out);
    out
}

pub fn intern_exact_tokens(vocab: &mut HashMap<String, u32>, text: &str) -> Vec<u32> {
    exact_tokens(text)
        .into_iter()
        .map(|tok| {
            if let Some(&id) = vocab.get(&tok) {
                id
            } else {
                let id = vocab.len() as u32;
                vocab.insert(tok, id);
                id
            }
        })
        .collect()
}

/// Query token ids in the index vocabulary. A token that no turn stored gets a
/// fresh id, so it breaks a run instead of matching a different unknown word.
pub fn query_exact_ids(vocab: &HashMap<String, u32>, query: &str) -> Vec<u32> {
    let mut unknown: HashMap<String, u32> = HashMap::new();
    let mut next = vocab.len() as u32;
    let mut ids = Vec::new();
    for tok in exact_tokens(query) {
        if let Some(&id) = vocab.get(&tok) {
            ids.push(id);
        } else {
            let id = if let Some(&id) = unknown.get(&tok) {
                id
            } else {
                let id = next;
                next += 1;
                unknown.insert(tok, id);
                id
            };
            ids.push(id);
        }
    }
    ids
}

/// How many outside turns the tail pool may add. Re-exported for the ranker.
pub fn below_window() -> usize {
    SUFFIX_BELOW_WINDOW
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_lcs(a: &[u32], b: &[u32]) -> u32 {
        let mut best = 0u32;
        for i in 0..a.len() {
            for j in 0..b.len() {
                let mut k = 0u32;
                while i + (k as usize) < a.len()
                    && j + (k as usize) < b.len()
                    && a[i + (k as usize)] == b[j + (k as usize)]
                {
                    k += 1;
                }
                if k > best {
                    best = k;
                }
            }
        }
        best
    }

    #[test]
    fn automaton_matches_the_naive_longest_common_run_both_ways() {
        let samples: &[&[u32]] = &[
            &[],
            &[1],
            &[1, 2, 3, 1, 2],
            &[2, 3, 1, 2, 9],
            &[1, 2, 3, 2, 3],
            &[2, 3, 2, 3],
            &[7, 7, 7],
            &[4, 5, 4, 5, 6, 4, 5],
            &[9, 8, 9, 8, 7, 9, 8, 9],
        ];
        for a in samples {
            for b in samples {
                let got = longest_common_token_run(a, b);
                let naive = naive_lcs(a, b);
                let on_left = SuffixAutomaton::build(a).longest_match(b);
                let on_right = SuffixAutomaton::build(b).longest_match(a);
                assert_eq!(got, naive, "sam {a:?} vs {b:?}");
                assert_eq!(on_left, on_right, "automaton side {a:?} {b:?}");
                assert_eq!(got, on_left, "shorter-side build {a:?} {b:?}");
            }
        }
        let mut state = 0x1234_5678u32;
        let mut next_tok = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) % 6
        };
        for n in 0..40 {
            let len_a = (n % 17) + 1;
            let len_b = ((n * 3) % 19) + 1;
            let a: Vec<u32> = (0..len_a).map(|_| next_tok()).collect();
            let b: Vec<u32> = (0..len_b).map(|_| next_tok()).collect();
            assert_eq!(longest_common_token_run(&a, &b), naive_lcs(&a, &b));
        }
    }

    #[test]
    fn exact_tokens_keep_names_handles_and_digits() {
        assert_eq!(exact_tokens("Sony A7"), vec!["sony".to_string(), "a7".to_string()]);
        assert_eq!(exact_tokens("mail me at @Sam_V"), vec!["mail".to_string(), "me".to_string(), "at".to_string(), "@sam_v".to_string()]);
        assert_eq!(exact_tokens("budget 1,200"), vec!["budget".to_string(), "1200".to_string()]);
        assert_eq!(exact_tokens("code ZX-42"), vec!["code".to_string(), "zx-42".to_string()]);
        assert_eq!(exact_tokens("Don't stop."), vec!["dont".to_string(), "stop".to_string()]);
        assert_eq!(exact_tokens("Bootstrap v5.3.0 end. Next"), vec!["bootstrap".to_string(), "v5.3.0".to_string(), "end".to_string(), "next".to_string()]);
        assert_eq!(exact_tokens("path a/b/c"), vec!["path".to_string(), "a/b/c".to_string()]);
    }

    #[test]
    fn repeated_calls_return_the_same_run() {
        let turn = [3, 1, 4, 1, 5, 9, 2, 6];
        let query = [1, 5, 9, 2, 7];
        let once = longest_common_token_run(&turn, &query);
        let twice = longest_common_token_run(&turn, &query);
        assert_eq!(once, 4);
        assert_eq!(once, twice);
    }

    #[test]
    fn below_window_constant_is_the_guard() {
        assert_eq!(below_window(), 8);
        assert_eq!(below_window(), SUFFIX_BELOW_WINDOW);
    }
}
