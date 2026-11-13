//! FCA second-stage bridge. Parameters below were fixed before the first
//! benchmark run. Do not change them after seeing a card.
//!
//! The formal context is the post-gatekeeper window plus the next
//! [`FCA_BELOW`] positive-score turns that missed that window. Objects are
//! those turns. Attributes are exact identical tokens (capitalized names,
//! handles, digit runs), the session id, and one undirected stored edge
//! (`policy_backlinks`, `supersedes_turns`, `exact_edges`, or the RSB
//! antecedent) between two objects. An attribute stays only when its support
//! in this context is [`FCA_PAIR_SUPPORT`]. That is the attribute concept of
//! a pair: support 1 is not shared, and support above 2 is not a pair.
//!
//! Concepts are the closed sets of Ganter's NextClosure. A below-window turn
//! is a bridge when stored edges form a two-hop path through it: the lattice
//! contains two concepts of extent size 2, each pairing it with a different
//! in-window turn, and each of those intents contains the stored edge between
//! that pair. A shared name, handle, digit, or session is an endpoint ballot
//! and does not vote. Each pair of window turns joined by such edges is one
//! vote. The greatest vote wins. A tie keeps the earlier tail turn, which is
//! the kernel's order. The winner is inserted at the last index inside
//! [`FCA_RECALL_DEPTH`] and the tail drops off `k`. Index 0 is not written.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Tail admitted under the window. `k` on the card is 10, so the context has
/// at most 26 objects. A pair-attribute lattice on `n` objects has at most
/// `C(n, 2) + n + 2` concepts, which NextClosure finishes.
pub const FCA_BELOW: usize = 16;

/// Last rank whose membership this stage may change. Recall@5 is the set of
/// the first five ranks (Manning, Raghavan, Schütze, IIR, ch. 8). The seat is
/// `min(k, FCA_RECALL_DEPTH) - 1`, which is inside slots `1..k-1` and leaves
/// index 0 untouched. Ranks above that seat keep the gatekeeper order.
pub const FCA_RECALL_DEPTH: usize = 5;

/// Support of an attribute in the bridge context. Two is the extent of a pair
/// concept. It is not a sweep.
pub const FCA_PAIR_SUPPORT: usize = 2;

/// A name or a handle is an identical token of at least this many characters.
/// One letter is a closed class, not a name.
pub const FCA_NAME_MIN_CHARS: usize = 2;

/// A digit token is a maximal ASCII digit run of at least this length. A
/// one-digit run is one of ten symbols.
pub const FCA_DIGIT_MIN_CHARS: usize = 2;

static FIRES: AtomicU64 = AtomicU64::new(0);

pub fn fires() -> u64 {
    FIRES.load(Ordering::Relaxed)
}

pub(crate) fn note_fire() {
    let n = FIRES.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some(path) = fire_path() {
        let _ = std::fs::write(path, n.to_string());
    }
}

fn fire_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| std::env::var("VESTIGE_FCA_FIRE_PATH").ok())
        .as_deref()
}

/// One object of the formal context. `tokens` are the ids assigned at index
/// time, sorted and unique.
#[derive(Clone, Debug)]
pub(crate) struct FcaObj {
    pub turn: usize,
    pub session: usize,
    pub tokens: Vec<u32>,
}

/// Capitalized names, `@handles`, and digit runs. No stop list. Hubs die
/// later, when support is not 2. Names are the lowercase letters of a token
/// whose first character is uppercase. Handles keep the case after `@`.
pub(crate) fn surface_keys(text: &str) -> Vec<String> {
    let mut keys = Vec::new();
    for raw in text.split_whitespace() {
        let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '@');
        if trimmed.is_empty() {
            continue;
        }
        push_digits(&mut keys, trimmed);
        if let Some(rest) = trimmed.strip_prefix('@') {
            let handle: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if handle.chars().count() >= FCA_NAME_MIN_CHARS {
                keys.push(format!("h:@{handle}"));
            }
            continue;
        }
        let mut letters = String::new();
        for c in trimmed.chars() {
            if c.is_alphabetic() {
                for low in c.to_lowercase() {
                    letters.push(low);
                }
            } else {
                break;
            }
        }
        if letters.chars().count() >= FCA_NAME_MIN_CHARS
            && trimmed.chars().next().is_some_and(|c| c.is_uppercase())
        {
            keys.push(format!("n:{letters}"));
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

fn push_digits(keys: &mut Vec<String>, token: &str) {
    let mut run = String::new();
    for c in token.chars() {
        if c.is_ascii_digit() {
            run.push(c);
            continue;
        }
        flush_digits(keys, &mut run);
    }
    flush_digits(keys, &mut run);
}

fn flush_digits(keys: &mut Vec<String>, run: &mut String) {
    if run.chars().count() >= FCA_DIGIT_MIN_CHARS {
        keys.push(format!("d:{run}"));
    }
    run.clear();
}

/// Insert `bridge` into slots `1..k-1`. Returns whether the list changed.
/// Index 0 is read and never written.
pub(crate) fn place_bridge(list: &mut Vec<usize>, bridge: usize, k: usize) -> bool {
    if k < 2 || list.is_empty() || list.contains(&bridge) {
        return false;
    }
    let limit = FCA_RECALL_DEPTH.min(k);
    if limit < 2 {
        return false;
    }
    let head = list[0];
    if list.len() < limit {
        list.push(bridge);
    } else {
        let seat = limit - 1;
        if seat == 0 || seat >= k {
            return false;
        }
        list.insert(seat, bridge);
        if list.len() > k {
            list.truncate(k);
        }
    }
    debug_assert_eq!(list[0], head);
    list[0] == head && list.contains(&bridge)
}

/// The below-window turn with the most two-hop votes, or nothing.
pub(crate) fn select_bridge(
    window: &[FcaObj],
    below: &[FcaObj],
    edges: &[(usize, usize)],
    k: usize,
) -> Option<usize> {
    if k < 2 || window.len() < 2 || below.is_empty() {
        return None;
    }
    if FCA_RECALL_DEPTH.min(k) < 2 {
        return None;
    }

    let mut objs: Vec<&FcaObj> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for obj in window {
        if seen.insert(obj.turn) {
            objs.push(obj);
        }
    }
    let n_window = objs.len();
    if n_window < 2 {
        return None;
    }
    for obj in below {
        if seen.insert(obj.turn) {
            objs.push(obj);
        }
    }
    if objs.len() == n_window {
        return None;
    }

    let mut token_df: HashMap<u32, usize> = HashMap::new();
    let mut session_df: HashMap<usize, usize> = HashMap::new();
    for obj in &objs {
        for &tok in &obj.tokens {
            *token_df.entry(tok).or_insert(0) += 1;
        }
        *session_df.entry(obj.session).or_insert(0) += 1;
    }
    let mut kept_tokens: Vec<u32> = token_df
        .into_iter()
        .filter(|(_, c)| *c == FCA_PAIR_SUPPORT)
        .map(|(tok, _)| tok)
        .collect();
    kept_tokens.sort_unstable();
    let mut kept_sessions: Vec<usize> = session_df
        .into_iter()
        .filter(|(_, c)| *c == FCA_PAIR_SUPPORT)
        .map(|(session, _)| session)
        .collect();
    kept_sessions.sort_unstable();
    let mut kept_edges: Vec<(usize, usize)> = edges
        .iter()
        .filter_map(|&(a, b)| {
            if a == b || !seen.contains(&a) || !seen.contains(&b) {
                return None;
            }
            Some(if a < b { (a, b) } else { (b, a) })
        })
        .collect();
    kept_edges.sort_unstable();
    kept_edges.dedup();

    let m = kept_tokens.len() + kept_sessions.len() + kept_edges.len();
    if m == 0 {
        return None;
    }
    let mut token_at: HashMap<u32, usize> = HashMap::with_capacity(kept_tokens.len());
    for (i, tok) in kept_tokens.iter().enumerate() {
        token_at.insert(*tok, i);
    }
    let session_base = kept_tokens.len();
    let mut session_at: HashMap<usize, usize> = HashMap::with_capacity(kept_sessions.len());
    for (i, session) in kept_sessions.iter().enumerate() {
        session_at.insert(*session, session_base + i);
    }
    let edge_base = session_base + kept_sessions.len();
    let mut edge_at: HashMap<(usize, usize), usize> = HashMap::with_capacity(kept_edges.len());
    for (i, edge) in kept_edges.iter().enumerate() {
        edge_at.insert(*edge, edge_base + i);
    }

    let mut obj_bits: Vec<Bits> = objs
        .iter()
        .map(|obj| {
            let mut bits = Bits::empty(m);
            for &tok in &obj.tokens {
                if let Some(&attr) = token_at.get(&tok) {
                    bits.set(attr);
                }
            }
            if let Some(&attr) = session_at.get(&obj.session) {
                bits.set(attr);
            }
            bits
        })
        .collect();
    let local_of: HashMap<usize, usize> = objs
        .iter()
        .enumerate()
        .map(|(i, obj)| (obj.turn, i))
        .collect();
    for &(a, b) in &kept_edges {
        let Some(&attr) = edge_at.get(&(a, b)) else {
            continue;
        };
        let Some(&la) = local_of.get(&a) else {
            continue;
        };
        let Some(&lb) = local_of.get(&b) else {
            continue;
        };
        obj_bits[la].set(attr);
        obj_bits[lb].set(attr);
    }

    let n = obj_bits.len();
    let bottom = Bits::ones(m);
    let cap = n * n.saturating_sub(1) / 2 + n + 2;
    let mut concepts: HashSet<Bits> = HashSet::new();
    let mut seen_intents: HashSet<Bits> = HashSet::new();
    let mut current = close(&Bits::empty(m), &obj_bits, &bottom);
    let mut complete = true;
    loop {
        if !seen_intents.insert(current.clone()) || concepts.len() >= cap {
            complete = false;
            break;
        }
        concepts.insert(extent_of(&current, &obj_bits));
        match next_closure(&current, &obj_bits, &bottom) {
            Some(next) => current = next,
            None => break,
        }
    }
    if !complete {
        return None;
    }

    let mut best_votes = 0usize;
    let mut best_turn = None;
    for b in n_window..n {
        let mut hops: Vec<Bits> = Vec::new();
        let turn_b = objs[b].turn;
        for w in 0..n_window {
            let turn_w = objs[w].turn;
            let edge_key = if turn_w < turn_b {
                (turn_w, turn_b)
            } else {
                (turn_b, turn_w)
            };
            // The hop is the concept of this stored edge. A token or session
            // shared with the endpoint is not a ballot.
            let Some(&edge_attr) = edge_at.get(&edge_key) else {
                continue;
            };
            let shared = obj_bits[w].intersect(&obj_bits[b]);
            if !shared.has(edge_attr) {
                continue;
            }
            let intent = close(&shared, &obj_bits, &bottom);
            if !intent.has(edge_attr) {
                continue;
            }
            let ext = extent_of(&intent, &obj_bits);
            if ext.count_ones() != 2 || !ext.has(w) || !ext.has(b) || !concepts.contains(&ext) {
                continue;
            }
            if hops
                .iter()
                .any(|prev| !intent.difference_any(prev) && !prev.difference_any(&intent))
            {
                continue;
            }
            hops.push(intent);
        }
        let mut votes = 0usize;
        for i in 0..hops.len() {
            for j in (i + 1)..hops.len() {
                if hops[i].difference_any(&hops[j]) && hops[j].difference_any(&hops[i]) {
                    votes += 1;
                }
            }
        }
        if votes > best_votes {
            best_votes = votes;
            best_turn = Some(objs[b].turn);
        }
    }
    best_turn
}

#[derive(Clone, Debug, Eq)]
struct Bits {
    words: Vec<u64>,
    width: usize,
}

impl PartialEq for Bits {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width && self.words == other.words
    }
}

impl std::hash::Hash for Bits {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.width.hash(state);
        self.words.hash(state);
    }
}

impl Bits {
    fn empty(width: usize) -> Self {
        let words = if width == 0 {
            Vec::new()
        } else {
            vec![0; width.div_ceil(64)]
        };
        Self { words, width }
    }

    fn ones(width: usize) -> Self {
        let mut bits = Self::empty(width);
        bits.words.fill(u64::MAX);
        bits.mask_tail();
        bits
    }

    fn mask_tail(&mut self) {
        let rem = self.width % 64;
        if rem != 0 {
            let last = self.words.len() - 1;
            self.words[last] &= (1u64 << rem) - 1;
        }
    }

    fn set(&mut self, index: usize) {
        self.words[index / 64] |= 1u64 << (index % 64);
    }

    fn has(&self, index: usize) -> bool {
        self.words[index / 64] & (1u64 << (index % 64)) != 0
    }

    fn count_ones(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    fn superset_of(&self, seed: &Self) -> bool {
        self.words
            .iter()
            .zip(&seed.words)
            .all(|(have, need)| have & need == *need)
    }

    fn and_with(&mut self, other: &Self) {
        for (word, other_word) in self.words.iter_mut().zip(&other.words) {
            *word &= *other_word;
        }
    }

    fn intersect(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.and_with(other);
        out
    }

    fn difference_any(&self, other: &Self) -> bool {
        self.words
            .iter()
            .zip(&other.words)
            .any(|(word, other_word)| word & !other_word != 0)
    }

    fn clear_from(&mut self, index: usize) {
        if index >= self.width {
            return;
        }
        let word_index = index / 64;
        let bit = index % 64;
        self.words[word_index] &= (1u64 << bit).wrapping_sub(1);
        for word in self.words.iter_mut().skip(word_index + 1) {
            *word = 0;
        }
    }

    fn gains_below(&self, index: usize, original: &Self) -> bool {
        let full_words = index / 64;
        for word in 0..full_words {
            if self.words[word] & !original.words[word] != 0 {
                return true;
            }
        }
        let bit = index % 64;
        if bit != 0 && full_words < self.words.len() {
            let mask = (1u64 << bit) - 1;
            if self.words[full_words] & !original.words[full_words] & mask != 0 {
                return true;
            }
        }
        false
    }
}

fn close(seed: &Bits, objects: &[Bits], bottom: &Bits) -> Bits {
    if seed.width == 0 {
        return Bits::empty(0);
    }
    let mut any = false;
    let mut intent = Bits::ones(seed.width);
    for obj in objects {
        if obj.superset_of(seed) {
            any = true;
            intent.and_with(obj);
        }
    }
    if any { intent } else { bottom.clone() }
}

fn extent_of(intent: &Bits, objects: &[Bits]) -> Bits {
    let mut ext = Bits::empty(objects.len());
    for (i, obj) in objects.iter().enumerate() {
        if obj.superset_of(intent) {
            ext.set(i);
        }
    }
    ext
}

fn next_closure(intent: &Bits, objects: &[Bits], bottom: &Bits) -> Option<Bits> {
    let m = intent.width;
    for i in (0..m).rev() {
        if intent.has(i) {
            continue;
        }
        let mut candidate = intent.clone();
        candidate.clear_from(i);
        candidate.set(i);
        let closed = close(&candidate, objects, bottom);
        if !closed.gains_below(i, intent) {
            return Some(closed);
        }
    }
    None
}

#[cfg(test)]
fn enumerate_intents(objects: &[Bits]) -> Option<Vec<Bits>> {
    if objects.is_empty() {
        return Some(Vec::new());
    }
    let m = objects[0].width;
    let bottom = Bits::ones(m);
    let n = objects.len();
    let cap = n * n.saturating_sub(1) / 2 + n + 2;
    let mut intents = Vec::new();
    let mut seen = HashSet::new();
    let mut current = close(&Bits::empty(m), objects, &bottom);
    loop {
        if !seen.insert(current.clone()) || intents.len() >= cap {
            return None;
        }
        intents.push(current.clone());
        match next_closure(&current, objects, &bottom) {
            Some(next) => current = next,
            None => return Some(intents),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(turn: usize, session: usize, tokens: &[u32]) -> FcaObj {
        FcaObj {
            turn,
            session,
            tokens: tokens.to_vec(),
        }
    }

    fn bit_row(width: usize, set: &[usize]) -> Bits {
        let mut bits = Bits::empty(width);
        for &i in set {
            bits.set(i);
        }
        bits
    }

    #[test]
    fn next_closure_matches_the_powerset_on_the_three_attribute_context() {
        let objects = vec![
            bit_row(3, &[0, 1]),
            bit_row(3, &[0, 2]),
            bit_row(3, &[1, 2]),
        ];
        let enumerated = enumerate_intents(&objects).expect("pair lattice finishes");
        let bottom = Bits::ones(3);
        let mut brute = HashSet::new();
        for mask in 0..8u32 {
            let mut seed = Bits::empty(3);
            for i in 0..3 {
                if mask & (1 << i) != 0 {
                    seed.set(i);
                }
            }
            brute.insert(close(&seed, &objects, &bottom));
        }
        assert_eq!(enumerated.len(), brute.len());
        assert_eq!(enumerated.len(), 8);
        for intent in enumerated {
            assert!(brute.contains(&intent));
        }
    }

    #[test]
    fn next_closure_matches_the_powerset_on_a_wider_context() {
        let rows: [&[usize]; 5] = [&[0, 1], &[1, 2], &[2, 3], &[0, 3, 4], &[4, 5]];
        let objects: Vec<Bits> = rows.iter().map(|row| bit_row(6, row)).collect();
        let enumerated = enumerate_intents(&objects).expect("lattice finishes");
        let bottom = Bits::ones(6);
        let mut brute = HashSet::new();
        for mask in 0..(1u32 << 6) {
            let mut seed = Bits::empty(6);
            for i in 0..6 {
                if mask & (1 << i) != 0 {
                    seed.set(i);
                }
            }
            brute.insert(close(&seed, &objects, &bottom));
        }
        let got: HashSet<Bits> = enumerated.into_iter().collect();
        assert_eq!(got, brute);
    }

    #[test]
    fn repeated_enumeration_is_identical() {
        let objects = vec![bit_row(4, &[0, 1]), bit_row(4, &[1, 2]), bit_row(4, &[2, 3])];
        assert_eq!(
            enumerate_intents(&objects).unwrap(),
            enumerate_intents(&objects).unwrap()
        );
    }

    #[test]
    fn two_token_concepts_are_not_a_bridge() {
        let window = vec![obj(0, 0, &[1]), obj(1, 1, &[2])];
        let below = vec![obj(2, 2, &[1, 2])];
        assert_eq!(select_bridge(&window, &below, &[], 10), None);
    }

    #[test]
    fn two_stored_edges_select_the_below_turn() {
        let window = vec![obj(0, 0, &[1]), obj(1, 1, &[2])];
        let below = vec![obj(2, 2, &[1, 2])];
        assert_eq!(select_bridge(&window, &below, &[(0, 2), (1, 2)], 10), Some(2));
        assert_eq!(select_bridge(&window, &below, &[(0, 2), (1, 2)], 10), Some(2));
    }

    #[test]
    fn one_shared_token_is_not_a_bridge() {
        let window = vec![obj(0, 0, &[1]), obj(1, 1, &[3])];
        let below = vec![obj(2, 2, &[1])];
        assert_eq!(select_bridge(&window, &below, &[], 10), None);
    }

    #[test]
    fn a_token_shared_by_three_objects_is_not_a_pair() {
        let window = vec![obj(0, 0, &[1]), obj(1, 1, &[1])];
        let below = vec![obj(2, 2, &[1])];
        assert_eq!(select_bridge(&window, &below, &[], 10), None);
    }

    #[test]
    fn an_edge_and_a_token_are_not_a_path() {
        let window = vec![obj(0, 0, &[7]), obj(1, 1, &[])];
        let below = vec![obj(2, 2, &[7])];
        assert_eq!(select_bridge(&window, &below, &[(1, 2)], 10), None);
    }

    #[test]
    fn equal_votes_keep_the_earlier_tail_turn() {
        let window = vec![obj(0, 0, &[]), obj(1, 1, &[])];
        let below = vec![obj(10, 3, &[]), obj(11, 4, &[])];
        let edges = [(0, 10), (1, 10), (0, 11), (1, 11)];
        assert_eq!(select_bridge(&window, &below, &edges, 10), Some(10));
    }

    #[test]
    fn more_votes_beat_an_earlier_tail_turn() {
        let window = vec![obj(0, 0, &[]), obj(1, 1, &[]), obj(2, 2, &[])];
        let below = vec![obj(10, 3, &[]), obj(11, 4, &[])];
        let edges = [(0, 10), (1, 10), (0, 11), (1, 11), (2, 11)];
        assert_eq!(select_bridge(&window, &below, &edges, 10), Some(11));
    }

    #[test]
    fn place_bridge_keeps_index_zero_and_seats_inside_recall_depth() {
        let mut list: Vec<usize> = (0..10).collect();
        assert!(place_bridge(&mut list, 42, 10));
        assert_eq!(list[0], 0);
        assert_eq!(list[4], 42);
        assert_eq!(list.len(), 10);
        assert_eq!(&list[1..4], &[1, 2, 3]);
        assert_eq!(&list[5..], &[4, 5, 6, 7, 8]);
        assert!(!list.contains(&9));
    }

    #[test]
    fn place_bridge_appends_while_the_recall_depth_has_room() {
        let mut list = vec![0usize, 1];
        assert!(place_bridge(&mut list, 7, 10));
        assert_eq!(list, vec![0, 1, 7]);
    }

    #[test]
    fn place_bridge_on_a_full_pair_replaces_slot_one_and_keeps_the_head() {
        let mut list = vec![0usize, 1];
        assert!(place_bridge(&mut list, 7, 2));
        assert_eq!(list, vec![0, 7]);
    }

    #[test]
    fn place_bridge_refuses_a_window_of_one_rank() {
        let mut list = vec![0usize];
        assert!(!place_bridge(&mut list, 7, 1));
        assert_eq!(list, vec![0]);
    }

    #[test]
    fn surface_keys_keep_names_handles_and_digit_runs_without_a_stop_list() {
        let keys = surface_keys("The Helena met @Sam in 2023. A 3 was not enough.");
        assert!(keys.contains(&"n:the".to_string()));
        assert!(keys.contains(&"n:helena".to_string()));
        assert!(keys.contains(&"h:@Sam".to_string()));
        assert!(keys.contains(&"d:2023".to_string()));
        assert!(!keys.iter().any(|k| k == "d:3"));
        assert!(!keys.iter().any(|k| k == "n:a" || k == "n:met"));
    }
}
