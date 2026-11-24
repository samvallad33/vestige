//! Wave 19 (#746) Torrens Live Title Register + Functional Multi-Hop Chain Deduction
//! + Physical Conservation-of-Possession Spatial Automaton.
//!
//! Pure integer (`u64` / `Q16.16`, zero `f32`/`f64` floats) and pure closed-class
//! English syntax (zero open-class word lists).
//!
//! 1. **Torrens Functional Key Supersession (#746)**:
//!    Indexes single-sentence relational assertions (`(Subject, Predicate) -> Value`)
//!    into a live title register. When a later turn updates the same `(Subject, Predicate)`
//!    key with a new value, the prior record is marked superseded (`torrens_superseded`),
//!    eliminating outdated branches before multi-hop traversal regardless of entity frequency.
//! 2. **Multi-Hop Functional Chain Deduction (`k = 1..=5`)**:
//!    Walks live Torrens titles `E_0 -> E_1 -> ... -> E_m` in `Q16.16` fixed-point,
//!    scoring predicate alignment via stems, WordNet definitions/lemmas, and ConceptNet,
//!    returning the chain ordered `[Terminus, ..., Anchor]` so the answer turn seats first.
//! 3. **Conservation-of-Possession Spatial Automaton**:
//!    Tracks `<Agent> ... to the <Place>` movement and `<Agent> ... the <Object>` possession
//!    parity across turns to resolve 2-hop (`Where is X?`) and 3-hop (`Where was X before Y?`)
//!    state trajectories deterministically.

use std::collections::BTreeMap;

use crate::retrieval::graph::ghostlink_dgs::Q16;
use crate::retrieval::index::{Index, TurnIx};
use crate::retrieval::stem;

/// A compiled functional `(Subject, Predicate) -> Value` assertion in the Torrens register.
#[derive(Clone, Debug)]
pub(in crate::retrieval) struct TorrensRecord {
    pub(in crate::retrieval) turn: usize,
    pub(in crate::retrieval) key_hash: u64,
    pub(in crate::retrieval) subj_phrase: String,
    pub(in crate::retrieval) pred_stems: Vec<String>,
    pub(in crate::retrieval) pred_raw: Vec<String>,
    pub(in crate::retrieval) val_phrase: String,
}

/// Pre-indexed closed-class spatial movement, compass relation, ditransitive transfer,
/// or transitive object interaction event.
#[derive(Clone, Debug)]
pub(in crate::retrieval) enum SpatialEvent {
    Move {
        turn: usize,
        agent: String,
        place: String,
    },
    Interact {
        turn: usize,
        agent: String,
        obj: String,
        explicit_release: bool,
        explicit_acquire: bool,
    },
    Compass {
        turn: usize,
        lhs: String,
        dir: String,
        rhs: String,
    },
    Transfer {
        turn: usize,
        giver: String,
        obj: String,
        recipient: String,
    },
}

#[inline]
fn fnv1a_u64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Strip leading numbered-list prefix (`"146. "`, `"12) "`) from a line/turn.
pub(in crate::retrieval) fn strip_leading_ordinal(text: &str) -> &str {
    let trimmed = text.trim_start();
    let bytes = trimmed.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i > 0 && i < bytes.len() && (bytes[i] == b'.' || bytes[i] == b')') {
        let rest = &trimmed[i + 1..];
        if rest.starts_with(' ') || rest.starts_with('\t') {
            return rest.trim_start();
        }
    }
    trimmed
}

/// Collapse closing quotes right before possessive `'s` (`"\"Free as a Bird\"'s"` -> `"\"Free as a Bird's\""`)
/// so multi-word quoted titles retain Saxon genitive attachment.
fn normalize_quote_possessive(s: &str) -> String {
    if !s.contains('"') && !s.contains('”') {
        return s.to_string();
    }
    s.replace("\"'s", "'s\"")
        .replace("\"’s", "’s\"")
        .replace("”'s", "'s”")
        .replace("”’s", "’s”")
}

fn cased_word_tokens(s: &str) -> Vec<&str> {
    s.split(|c: char| !c.is_alphanumeric() && c != '-' && c != '\'' && c != '’')
        .map(|t| t.trim_matches(|c: char| c == '\'' || c == '’' || c == '-'))
        .filter(|t| !t.is_empty())
        .collect()
}

/// Closed-class derivational suffix stripping (`citizenship` -> `citizen`, `education` -> `educ`)
/// followed by inflectional `stem`.
fn deriv_root(word: &str) -> String {
    let mut base = word;
    for suf in [
        "ship", "hood", "ation", "tion", "ment", "ness", "ance", "ence", "ity",
    ] {
        if let Some(stripped) = base.strip_suffix(suf)
            && stripped.len() >= 4
        {
            base = stripped;
            break;
        }
    }
    stem(base)
}

#[inline]
fn norm_lower(tok: &str) -> String {
    tok.to_lowercase().replace('’', "'")
}

#[inline]
fn starts_upper(tok: &str) -> bool {
    tok.chars().next().is_some_and(char::is_uppercase)
}

#[inline]
fn is_rel_pivot(low: &str) -> bool {
    matches!(
        low,
        "is" | "was" | "are" | "were" | "to" | "of" | "as" | "by" | "for"
    )
}

#[inline]
fn is_first_person_pronoun(low: &str) -> bool {
    matches!(low, "i" | "i'm" | "i've" | "i'd" | "i'll" | "my" | "me" | "we" | "we're" | "us")
}

/// Attempt to parse a single-sentence functional assertion into a `TorrensRecord`.
pub(in crate::retrieval) fn parse_torrens_sentence(raw_prose: &str, turn: usize) -> Option<TorrensRecord> {
    let clean_raw = strip_leading_ordinal(raw_prose).trim();
    if clean_raw.is_empty() || clean_raw.ends_with('?') || clean_raw.contains('\n') {
        return None;
    }
    let norm_quotes = normalize_quote_possessive(clean_raw);
    let clean = norm_quotes.as_str();
    // Single-sentence guard: reject multi-sentence paragraphs
    if let Some(dot_pos) = clean.find(['.', '!', '?']) {
        if dot_pos + 1 < clean.len() && clean[dot_pos + 1..].trim().len() > 2 {
            return None;
        }
    }
    let cased = cased_word_tokens(clean);
    if !(4..=26).contains(&cased.len()) {
        return None;
    }
    let lowers: Vec<String> = cased.iter().map(|t| norm_lower(t)).collect();
    // Skip first-person conversational statements ("I am ...", "My brother ...")
    if lowers.iter().enumerate().any(|(idx, w)| {
        is_first_person_pronoun(w) || (w == "our" && (idx == 0 || !starts_upper(cased[idx])))
    }) {
        return None;
    }

    // Find rightmost relational pivot leaving 1..=8 value tokens on the right,
    // refusing to split inside capitalized proper names like `"Olga of Kiev"`.
    let n = cased.len();
    let mut split_idx: Option<usize> = None;
    let min_p = 2usize;
    let max_p = n.saturating_sub(2);
    for p in (min_p..=max_p).rev() {
        if n - 1 - p > 8 {
            break;
        }
        let lw = lowers[p].as_str();
        if !is_rel_pivot(lw) {
            continue;
        }
        if matches!(lw, "of" | "to" | "as" | "by" | "for") && starts_upper(cased[p - 1]) {
            continue;
        }
        split_idx = Some(p);
        break;
    }
    let p = split_idx?;
    let prefix_cased = &cased[..=p];
    let prefix_low = &lowers[..=p];
    let val_low_slice = &lowers[p + 1..];

    // Require at least one proper/entity token in prefix (beyond bare sentence-initial "The"/"A")
    let has_entity_in_prefix = prefix_cased.iter().enumerate().any(|(i, tok)| {
        starts_upper(tok) && !(i == 0 && matches!(prefix_low[0].as_str(), "the" | "a" | "an" | "in" | "on"))
    }) || (p >= 3 && prefix_low[0] != "the" && prefix_low[1] != "the");
    if !has_entity_in_prefix {
        return None;
    }

    // Check Saxon genitive `<Subj>'s <pred...> is/was <Val>` when `'s` is not inside `"The ... of ..."`
    let saxon_idx = if matches!(prefix_low[p].as_str(), "is" | "was" | "are" | "were") {
        (0..p.saturating_sub(1)).rfind(|&idx| {
            prefix_low[idx].ends_with("'s")
                && prefix_low[idx].len() > 2
                && (prefix_low[0] != "the" || !prefix_low[1..idx].iter().any(|w| w == "of"))
        })
    } else {
        None
    };

    // Extract `(subj_phrase, pred_tokens)`:
    // Form S: `<Subj>'s <pred...> is/was <Val>`
    // Form A: `"The <pred...> of <Subj...> is/was <Val>"` (including nested-`of` proper spans)
    // Form A-rel: `"The <pred...> where/which/that/whom <Subj...> [is/was <verb...>] is/was <Val>"`
    // Form B: `"<Subj...> is/was/verbs <pred...> to/of/in/as/by/for <Val>"`
    let (subj_tokens, pred_tokens_raw): (Vec<String>, Vec<String>) = if let Some(s_idx) = saxon_idx
    {
        let mut subj_vec = prefix_low[..=s_idx].to_vec();
        if subj_vec.first().map(String::as_str) == Some("the") && subj_vec.len() > 1 {
            subj_vec.remove(0);
        }
        if let Some(last) = subj_vec.last_mut()
            && let Some(base) = last.strip_suffix("'s")
        {
            *last = base.to_string();
        }
        let pred_vec = prefix_low[s_idx + 1..p].to_vec();
        if subj_vec.is_empty() || pred_vec.is_empty() {
            return None;
        }
        (subj_vec, pred_vec)
    } else if prefix_low.first().map(String::as_str) == Some("the")
        && matches!(prefix_low[p].as_str(), "is" | "was" | "are" | "were")
        && prefix_low[1..p].iter().any(|w| w == "of")
    {
        let of_pos = 1 + prefix_low[1..p].iter().position(|w| w == "of")?;
        let cap_first = ((of_pos + 1)..p).find(|&idx| {
            starts_upper(prefix_cased[idx])
                && !(idx == of_pos + 1 && prefix_low[idx] == "the")
        });
        let cap_last = ((of_pos + 1)..p)
            .rev()
            .find(|&idx| starts_upper(prefix_cased[idx]));
        let has_nested_frame = if let (Some(cs), Some(ce)) = (cap_first, cap_last) {
            cs <= ce
                && (cs > of_pos + 2
                    || (cs == of_pos + 2 && prefix_low[of_pos + 1] != "the")
                    || ce + 1 < p)
        } else {
            false
        };
        if has_nested_frame
            && let (Some(cs), Some(ce)) = (cap_first, cap_last)
        {
            let subj_vec = prefix_low[cs..=ce].to_vec();
            let mut pred_vec: Vec<String> = prefix_low[1..cs]
                .iter()
                .filter(|w| !matches!(w.as_str(), "of" | "in" | "the" | "for" | "to"))
                .cloned()
                .collect();
            for w in &prefix_low[ce + 1..p] {
                if !matches!(w.as_str(), "of" | "in" | "the" | "for" | "to") {
                    pred_vec.push(w.clone());
                }
            }
            if subj_vec.is_empty() || pred_vec.is_empty() {
                return None;
            }
            (subj_vec, pred_vec)
        } else {
            // Check double-`of` functional container (`"The country of citizenship of X is Y"`):
            // `subj` sits after the last lowercase `"of"` before proper entity `X`.
            let second_of = ((of_pos + 1)..p)
                .rfind(|&idx| prefix_low[idx] == "of" && !starts_upper(prefix_cased[idx - 1]));
            let split_of = second_of.unwrap_or(of_pos);
            let pred_slice = &prefix_low[1..split_of];
            let mut subj_slice = &prefix_low[split_of + 1..p];
            if subj_slice.first().map(String::as_str) == Some("the") && subj_slice.len() > 1 {
                subj_slice = &subj_slice[1..];
            }
            if subj_slice.is_empty() || pred_slice.is_empty() {
                return None;
            }
            (subj_slice.to_vec(), pred_slice.to_vec())
        }
    } else if prefix_low.first().map(String::as_str) == Some("the")
            && matches!(prefix_low[p].as_str(), "is" | "was" | "are" | "were")
            && prefix_low[1..p]
                .iter()
                .any(|w| matches!(w.as_str(), "where" | "which" | "that" | "whom"))
        {
            let rel_pos = 1 + prefix_low[1..p]
                .iter()
                .position(|w| matches!(w.as_str(), "where" | "which" | "that" | "whom"))?;
            let inner_aux = ((rel_pos + 1)..p)
                .find(|&idx| {
                    matches!(
                        prefix_low[idx].as_str(),
                        "is" | "was" | "are" | "were" | "has" | "had" | "did"
                    )
                })
                .unwrap_or(p);
            let mut subj_slice = &prefix_low[rel_pos + 1..inner_aux];
            if subj_slice.first().map(String::as_str) == Some("the") && subj_slice.len() > 1 {
                subj_slice = &subj_slice[1..];
            }
            let mut pred_vec = prefix_low[1..rel_pos].to_vec();
            if inner_aux < p {
                pred_vec.extend_from_slice(&prefix_low[inner_aux + 1..p]);
            }
            if subj_slice.is_empty() || pred_vec.is_empty() {
                return None;
            }
            (subj_slice.to_vec(), pred_vec)
        } else {
            // Find the first copula/auxiliary in Form B (`is`, `was`, `are`, `were`, `has`, `had`) first,
            // falling back to capitalized-to-lowercase verb boundary (`"Olga of Kiev died in the city of"`).
            let mut verb_pos: Option<usize> = (1..p).find(|&idx| {
                matches!(
                    prefix_low[idx].as_str(),
                    "is" | "was" | "are" | "were" | "has" | "had"
                )
            });
            if verb_pos.is_none() {
                for idx in 1..p {
                    let lw = prefix_low[idx].as_str();
                    if !starts_upper(prefix_cased[idx]) && lw != "of" && lw != "the" && lw != "and" {
                        if prefix_cased[idx..p].iter().all(|t| !starts_upper(t)) {
                            verb_pos = Some(idx);
                            break;
                        }
                    }
                }
            }
            let vp = verb_pos?;
            // Handle `"The <pred...> of <ProperSubj> is located in the city of <Val>"` in Form B:
            let nested_of = if prefix_low.first().map(String::as_str) == Some("the") && vp >= 4 {
                (2..vp.saturating_sub(1)).find(|&idx| {
                    prefix_low[idx] == "of"
                        && prefix_cased[1..idx].iter().all(|t| !starts_upper(t))
                        && ((idx + 1)..vp).any(|k| starts_upper(prefix_cased[k]))
                })
            } else {
                None
            };
            if let Some(of_b) = nested_of {
                let mut subj_slice = &prefix_low[of_b + 1..vp];
                if subj_slice.first().map(String::as_str) == Some("the") && subj_slice.len() > 1 {
                    subj_slice = &subj_slice[1..];
                }
                let mut pred_vec = prefix_low[1..of_b].to_vec();
                pred_vec.extend_from_slice(&prefix_low[vp..p]);
                if subj_slice.is_empty() || pred_vec.is_empty() {
                    return None;
                }
                (subj_slice.to_vec(), pred_vec)
            } else {
                let mut subj_slice = &prefix_low[..vp];
                if subj_slice.first().map(String::as_str) == Some("the") && subj_slice.len() > 1 {
                    subj_slice = &subj_slice[1..];
                }
                let pred_slice = &prefix_low[vp..p];
                if subj_slice.is_empty() || pred_slice.is_empty() {
                    return None;
                }
                (subj_slice.to_vec(), pred_slice.to_vec())
            }
        };

    let mut val_slice = val_low_slice;
    if val_slice.first().map(String::as_str) == Some("the") && val_slice.len() > 1 {
        val_slice = &val_slice[1..];
    }
    if val_slice.is_empty() {
        return None;
    }

    let mut pred_raw: Vec<String> = Vec::new();
    let mut pred_stems: Vec<String> = Vec::new();
    for w in &pred_tokens_raw {
        if w.len() >= 3
            && !matches!(
                w.as_str(),
                "the" | "was" | "were" | "are" | "with" | "for" | "from" | "into" | "name" | "current"
            )
        {
            if !pred_raw.contains(w) {
                pred_raw.push(w.clone());
            }
            let st = deriv_root(w);
            if st.len() >= 3 && !pred_stems.contains(&st) {
                pred_stems.push(st);
            }
        }
    }
    if pred_stems.is_empty() {
        return None;
    }

    let subj_phrase = subj_tokens.join(" ");
    let val_phrase = val_slice.join(" ");
    // Canonicalize functional key via multi-word/single-word WordNet primary synset when unambiguous
    // (`chief executive officer` == `ceo`, `citizenship` -> `citizen` via `deriv_root`), dropping
    // leading container nouns in `"country of citizenship"` so updates collide cleanly at 262k scale.
    let pred_joined = pred_raw
        .iter()
        .filter(|w| *w != "of" && *w != "in")
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let key_pred = if let Some(&syn) = crate::wordnet::senses(&pred_joined).first() {
        format!("syn:{syn}")
    } else if pred_stems.len() >= 2
        && pred_tokens_raw.contains(&"of".to_string())
        && let Some(last_st) = pred_stems.last()
    {
        if let Some(&syn) = pred_raw
            .last()
            .and_then(|lw| crate::wordnet::senses(lw).first())
        {
            format!("syn:{syn}")
        } else {
            last_st.clone()
        }
    } else if pred_raw.len() == 1
        && let Some(&syn) = crate::wordnet::senses(&pred_raw[0]).first()
    {
        format!("syn:{syn}")
    } else {
        pred_stems.join("_")
    };
    let key_str = format!("{subj_phrase}||{key_pred}");
    let key_hash = fnv1a_u64(key_str.as_bytes());

    Some(TorrensRecord {
        turn,
        key_hash,
        subj_phrase,
        pred_stems,
        pred_raw,
        val_phrase,
    })
}

/// Extract closed-class `<CapAgent> ... to the <Place>`, `<CapAgent> ... the <Obj>`,
/// `"The <X> is <cardinal_dir> of the <Y>"`, and `<Cap1> <verb> the <Obj> to <Cap2>` events.
fn extract_spatial_events(raw_prose: &str, turn: usize, out: &mut Vec<SpatialEvent>) {
    for raw_sent in raw_prose.split(['.', '!', '?', '\n']) {
        let sent = strip_leading_ordinal(raw_sent).trim();
        if sent.is_empty() {
            continue;
        }
        let cased = cased_word_tokens(sent);
        if !(4..=16).contains(&cased.len()) {
            continue;
        }
        let lowers: Vec<String> = cased.iter().map(|t| norm_lower(t)).collect();
        let n = lowers.len();
        let low0 = lowers[0].clone();

        // Pattern 0A (Compass): `"The <lhs> is <cardinal_dir> of the <rhs>"`
        if n == 7
            && low0 == "the"
            && lowers[2] == "is"
            && lowers[4] == "of"
            && lowers[5] == "the"
            && !cased[1..].iter().any(|t| starts_upper(t))
            && crate::wordnet::opposite_cardinal_direction(&lowers[3]).is_some()
        {
            out.push(SpatialEvent::Compass {
                turn,
                lhs: lowers[1].clone(),
                dir: lowers[3].clone(),
                rhs: lowers[6].clone(),
            });
            continue;
        }

        // Sentence-initial proper name `<Agent>` (not a determiner/pronoun)
        if !starts_upper(cased[0]) || cased[0].len() < 3 {
            continue;
        }
        if matches!(
            low0.as_str(),
            "the" | "this" | "that" | "these" | "those" | "there" | "here" | "when" | "where" | "what" | "who" | "why" | "how" | "she" | "they" | "his" | "her" | "our" | "but" | "and" | "then" | "after" | "before"
        ) {
            continue;
        }

        // Pattern 0B (Trivalent Transfer): `<CapAgent1> <verb> "the" <obj> "to" <CapAgent2>`
        if n == 6
            && !starts_upper(cased[1])
            && lowers[2] == "the"
            && !starts_upper(cased[3])
            && lowers[4] == "to"
            && starts_upper(cased[5])
            && cased[5].len() >= 3
        {
            out.push(SpatialEvent::Transfer {
                turn,
                giver: low0,
                obj: lowers[3].clone(),
                recipient: lowers[5].clone(),
            });
            continue;
        }

        // Reject if any inner word (except pos 0) is capitalized, so book clauses with multiple proper nouns skip
        if cased[1..].iter().any(|t| starts_upper(t)) {
            continue;
        }

        // Pattern 1: `<Agent> [1..3 words] "to" "the" <Place>`
        if n >= 5 && lowers[n - 3] == "to" && lowers[n - 2] == "the" && (2..=4).contains(&(n - 3)) {
            out.push(SpatialEvent::Move {
                turn,
                agent: low0,
                place: lowers[n - 1].clone(),
            });
            continue;
        }

        // Pattern 1A: `<Agent> [1..3 words] "to" "the" <Place> "carrying" "the" <Obj>`
        if n >= 8
            && lowers[n - 3] == "carrying"
            && lowers[n - 2] == "the"
            && lowers[n - 6] == "to"
            && lowers[n - 5] == "the"
            && (2..=4).contains(&(n - 6))
        {
            out.push(SpatialEvent::Move {
                turn,
                agent: low0.clone(),
                place: lowers[n - 4].clone(),
            });
            out.push(SpatialEvent::Interact {
                turn,
                agent: low0,
                obj: lowers[n - 1].clone(),
                explicit_release: false,
                explicit_acquire: true,
            });
            continue;
        }

        // Pattern 1B: `<Agent> [1..2 words] "the" <Obj> ("in"|"at"|"on"|"from") "the" <Place> ["before" ("leaving"|"going") ("to"|"for") "the" <NextPlace>]`
        if let Some(the_pos) = (2..=3usize).find(|&p| p + 4 < n && lowers[p] == "the") {
            let prep = lowers[the_pos + 2].as_str();
            if matches!(prep, "in" | "at" | "on" | "from")
                && lowers[the_pos + 3] == "the"
                && !matches!(lowers[the_pos - 1].as_str(), "to" | "in" | "at" | "on" | "of")
            {
                let place_pos = the_pos + 4;
                let next_place = if place_pos + 1 == n {
                    Some(None)
                } else if place_pos + 6 == n
                    && lowers[place_pos + 1] == "before"
                    && matches!(lowers[place_pos + 2].as_str(), "leaving" | "going")
                    && matches!(lowers[place_pos + 3].as_str(), "to" | "for")
                    && lowers[place_pos + 4] == "the"
                {
                    Some(Some(lowers[place_pos + 5].clone()))
                } else {
                    None
                };
                if let Some(opt_next) = next_place {
                    let has_up = lowers[1..the_pos].iter().any(|w| w == "up");
                    let has_down = lowers[1..the_pos].iter().any(|w| w == "down");
                    let explicit_acquire = prep == "from" || has_up;
                    let explicit_release = !explicit_acquire && (has_down || opt_next.is_some());
                    out.push(SpatialEvent::Move {
                        turn,
                        agent: low0.clone(),
                        place: lowers[place_pos].clone(),
                    });
                    out.push(SpatialEvent::Interact {
                        turn,
                        agent: low0.clone(),
                        obj: lowers[the_pos + 1].clone(),
                        explicit_release,
                        explicit_acquire,
                    });
                    if let Some(nxt_pl) = opt_next {
                        out.push(SpatialEvent::Move {
                            turn,
                            agent: low0,
                            place: nxt_pl,
                        });
                    }
                    continue;
                }
            }
        }

        // Pattern 2: `<Agent> [1..2 words] "the" <Obj> ["there"]` (no `"to"` before `"the"`)
        let (the_pos, obj_pos) = if n >= 4 && lowers[n - 2] == "the" {
            (n - 2, n - 1)
        } else if n >= 5 && lowers[n - 1] == "there" && lowers[n - 3] == "the" {
            (n - 3, n - 2)
        } else {
            continue;
        };
        if !(2..=3).contains(&the_pos) || lowers[the_pos - 1] == "to" || lowers[the_pos - 1] == "in" || lowers[the_pos - 1] == "at" || lowers[the_pos - 1] == "on" || lowers[the_pos - 1] == "of" {
            continue;
        }
        let explicit_release = lowers[1..the_pos].iter().any(|w| w == "down");
        let explicit_acquire = lowers[1..the_pos].iter().any(|w| w == "up");
        out.push(SpatialEvent::Interact {
            turn,
            agent: low0,
            obj: lowers[obj_pos].clone(),
            explicit_release,
            explicit_acquire,
        });
    }
}

/// Build the Torrens Title Register and Spatial Possession Event Log at ingest time.
pub(in crate::retrieval) fn build_torrens_and_spatial_index(
    turn_prose_cache: &[String],
    turns: &[TurnIx],
) -> (Vec<TorrensRecord>, Vec<bool>, Vec<SpatialEvent>) {
    let nt = turns.len();
    let mut records: Vec<TorrensRecord> = Vec::new();
    let mut torrens_superseded = vec![false; nt];
    let mut title_holder: BTreeMap<u64, (usize, String)> = BTreeMap::new();
    let mut spatial_events: Vec<SpatialEvent> = Vec::new();

    for ti in 0..nt {
        let t = &turns[ti];
        if t.field != 0 || t.is_backchannel {
            continue;
        }
        let prose = match turn_prose_cache.get(ti) {
            Some(s) => s.as_str(),
            None => continue,
        };
        extract_spatial_events(prose, ti, &mut spatial_events);

        if t.is_question || t.is_standing_policy {
            continue;
        }
        if let Some(mut rec) = parse_torrens_sentence(prose, ti) {
            // Scope functional keys per named speaker in multi-party chats so Speaker A never supersedes Speaker B:
            if !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant" {
                let mut sp_key = t.speaker.to_ascii_lowercase().into_bytes();
                sp_key.push(b':');
                sp_key.extend_from_slice(&rec.key_hash.to_le_bytes());
                rec.key_hash = fnv1a_u64(&sp_key);
            }
            if let Some((prev_ti, prev_val)) = title_holder.get(&rec.key_hash) {
                if *prev_ti < ti && turns[*prev_ti].session <= t.session && *prev_val != rec.val_phrase {
                    torrens_superseded[*prev_ti] = true;
                }
            }
            title_holder.insert(rec.key_hash, (ti, rec.val_phrase.clone()));
            records.push(rec);
        }
    }

    (records, torrens_superseded, spatial_events)
}

/// Score alignment in Q16.16 between a live Torrens record's predicate and remaining query relation cues,
/// returning `(score_q16, newly_consumed_stem_mask)`.
fn score_predicate_match_masked_q16(
    rec: &TorrensRecord,
    q_stems: &[String],
    used_mask: u64,
    q_phrase_links: &[(String, u64)],
    q_expanded: &[(String, u64)],
) -> (u64, u64) {
    let mut score: u64 = 0;
    let mut new_mask = used_mask;
    for ps in &rec.pred_stems {
        for (idx, qs) in q_stems.iter().take(64).enumerate() {
            let bit = 1u64 << idx;
            if (new_mask & bit) != 0 {
                continue;
            }
            if qs == ps {
                score = score.saturating_add(Q16 * 3);
                new_mask |= bit;
                break;
            }
            if ps.len() >= 5
                && qs.len() >= 5
                && (ps.starts_with(qs.as_str()) || qs.starts_with(ps.as_str()))
            {
                score = score.saturating_add(Q16 * 2);
                new_mask |= bit;
                break;
            }
        }
    }
    // Check multi-word synset phrase equivalence (`chief executive officer` <-> `ceo`, `"pass away"` <-> `died`)
    let rec_phrase = rec
        .pred_raw
        .iter()
        .filter(|w| *w != "of" && *w != "in")
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    if rec_phrase.contains(' ') {
        if let Some((_, bits)) = q_phrase_links
            .iter()
            .find(|(qe, b)| qe == &rec_phrase && (*b & !new_mask) != 0)
        {
            score = score.saturating_add(Q16 * 3);
            new_mask |= *bits;
        }
    }
    for pr in &rec.pred_raw {
        let pr_root = deriv_root(pr);
        for (qe, qe_bits) in q_phrase_links {
            if (*qe_bits & !new_mask) != 0 && (qe == pr || deriv_root(qe) == pr_root) {
                score = score.saturating_add(Q16 * 2);
                new_mask |= *qe_bits;
                break;
            }
        }
    }

    let mut exp_bonus: u64 = 0;
    let max_exp_bonus: u64 = (Q16 * 3) >> 2;
    let mut matched_wn_hyper_any = false;
    for pr in &rec.pred_raw {
        for up in crate::wordnet::holonyms_and_upward(pr, 8) {
            let up_st = deriv_root(up);
            if let Some((idx, _)) = q_stems
                .iter()
                .take(64)
                .enumerate()
                .find(|&(idx, qs)| (new_mask & (1u64 << idx)) == 0 && qs == &up_st)
            {
                score = score.saturating_add(Q16);
                new_mask |= 1u64 << idx;
                matched_wn_hyper_any = true;
                break;
            }
        }
        if !matched_wn_hyper_any && (new_mask == used_mask) && exp_bonus < max_exp_bonus {
            if let Some((_, qe_bits)) = q_expanded
                .iter()
                .find(|(qe, b)| (*b & !new_mask) != 0 && qe == pr)
            {
                exp_bonus = max_exp_bonus;
                new_mask |= *qe_bits;
                break;
            }
            let pst = deriv_root(pr);
            let pst_trim = pst.trim_end_matches(|c| c == 'y' || c == 'i');
            if let Some((_, qe_bits)) = q_expanded.iter().find(|(qe, b)| {
                if (*b & !new_mask) == 0 {
                    return false;
                }
                let qst = deriv_root(qe);
                let qst_trim = qst.trim_end_matches(|c| c == 'y' || c == 'i');
                qst == pst
                    || (pst_trim.len() >= 4
                        && qst_trim.len() >= 4
                        && (pst_trim.starts_with(qst_trim) || qst_trim.starts_with(pst_trim)))
            }) {
                exp_bonus = exp_bonus.saturating_add(Q16 >> 1).min(max_exp_bonus);
                new_mask |= *qe_bits;
                continue;
            }
            for mid in crate::wordnet::expansion_lemmas(pr).into_iter().chain(
                crate::conceptnet::two_hop_neighbors(pr, 24)
                    .into_iter()
                    .map(|(s, _)| s),
            ) {
                if let Some((_, qe_bits)) = q_expanded
                    .iter()
                    .find(|(qe, b)| (*b & !new_mask) != 0 && qe == mid)
                {
                    exp_bonus = max_exp_bonus;
                    new_mask |= *qe_bits;
                    break;
                }
            }
        }
    }
    if new_mask == used_mask {
        return (0, used_mask);
    }
    (score.saturating_add(exp_bonus), new_mask)
}

/// Deduce a multi-hop functional chain (`k = 2..=5` live Torrens hops) for relational WH-queries
/// via bounded beam lookahead (`<= 24` states per hop), returning `([ti_terminus, ..., ti_anchor], matched_cues)`
/// so the final answer turn seats at Slot 0 even across unlexicalized intermediate bridges.
pub(in crate::retrieval) fn resolve_torrens_chain_q16(ix: &Index, q_text: &str) -> (Vec<usize>, u32) {
    if ix.torrens_records.len() < 3 {
        return (Vec::new(), 0);
    }
    let norm_q = normalize_quote_possessive(q_text);
    let cased = cased_word_tokens(&norm_q);
    if cased.len() < 4 {
        return (Vec::new(), 0);
    }
    let lowers: Vec<String> = cased.iter().map(|t| norm_lower(t)).collect();

    // Collect live records grouped by `subj_phrase` and inverse `val_phrase`
    let mut by_subj: BTreeMap<&str, Vec<&TorrensRecord>> = BTreeMap::new();
    let mut by_val: BTreeMap<&str, Vec<&TorrensRecord>> = BTreeMap::new();
    for rec in &ix.torrens_records {
        if ix.torrens_superseded.get(rec.turn).copied().unwrap_or(false) {
            continue;
        }
        by_subj.entry(rec.subj_phrase.as_str()).or_default().push(rec);
        by_val.entry(rec.val_phrase.as_str()).or_default().push(rec);
    }
    if by_subj.is_empty() {
        return (Vec::new(), 0);
    }

    // Collect capitalized proper spans in `q_text` (excluding sentence-initial WH/preposition word)
    let mut q_proper_spans: Vec<String> = Vec::new();
    let mut cur_span: Vec<String> = Vec::new();
    for (idx, tok) in cased.iter().enumerate() {
        let is_q_lead = idx == 0
            && matches!(
                lowers[0].as_str(),
                "what" | "which" | "who" | "where" | "when" | "how" | "whose" | "whom" | "in" | "on" | "to" | "by" | "for" | "from" | "of" | "the"
            );
        if !is_q_lead && starts_upper(tok) {
            let mut clean_lw = lowers[idx].clone();
            if let Some(b) = clean_lw.strip_suffix("'s") {
                clean_lw = b.to_string();
            }
            cur_span.push(clean_lw);
        } else if !cur_span.is_empty() && matches!(lowers[idx].as_str(), "of" | "the" | "al") {
            cur_span.push(lowers[idx].clone());
        } else if !cur_span.is_empty() {
            while cur_span
                .last()
                .is_some_and(|w| matches!(w.as_str(), "of" | "the" | "al"))
            {
                cur_span.pop();
            }
            if !cur_span.is_empty() {
                q_proper_spans.push(cur_span.join(" "));
                cur_span.clear();
            }
        }
    }
    while cur_span
        .last()
        .is_some_and(|w| matches!(w.as_str(), "of" | "the" | "al"))
    {
        cur_span.pop();
    }
    if !cur_span.is_empty() {
        q_proper_spans.push(cur_span.join(" "));
    }

    let lowers_deposs: Vec<String> = lowers
        .iter()
        .map(|w| w.strip_suffix("'s").unwrap_or(w.as_str()).to_string())
        .collect();

    // O(1) query n-gram probes (`1..=7` tokens) into `by_subj` and `by_val`
    let mut seed_subjs: Vec<&str> = Vec::new();
    let qn = lowers.len();
    for start in 0..qn {
        for len in 1..=7usize {
            if start + len > qn {
                break;
            }
            let cand1 = lowers[start..start + len].join(" ");
            let cand2 = lowers_deposs[start..start + len].join(" ");
            for cand in [cand1.as_str(), cand2.as_str()] {
                if cand.len() < 3 {
                    continue;
                }
                if let Some((&k, _)) = by_subj.get_key_value(cand).or_else(|| by_val.get_key_value(cand))
                    && !seed_subjs.contains(&k)
                {
                    seed_subjs.push(k);
                }
            }
        }
    }
    if seed_subjs.is_empty() {
        return (Vec::new(), 0);
    }
    let is_proper_seed = |s: &str| -> bool {
        q_proper_spans
            .iter()
            .any(|p| p == s || p.contains(s) || s.contains(p.as_str()))
    };
    if !q_proper_spans.is_empty() {
        seed_subjs.retain(|s| is_proper_seed(s));
        if seed_subjs.is_empty() {
            return (Vec::new(), 0);
        }
    }
    seed_subjs.sort_by(|a, b| {
        is_proper_seed(b)
            .cmp(&is_proper_seed(a))
            .then_with(|| b.split_whitespace().count().cmp(&a.split_whitespace().count()))
            .then_with(|| b.len().cmp(&a.len()))
            .then_with(|| a.cmp(b))
    });

    let seed_primary = seed_subjs[0];
    let seed_toks: Vec<&str> = if is_proper_seed(seed_primary) {
        seed_primary.split_whitespace().collect()
    } else {
        Vec::new()
    };
    let mut q_stems: Vec<String> = Vec::new();
    let mut raw_stem_positions: Vec<(usize, usize)> = Vec::new();
    for (pos, w_raw) in lowers_deposs.iter().enumerate() {
        let w = w_raw.as_str();
        if w.len() < 3 || seed_toks.contains(&w) {
            continue;
        }
        if matches!(
            w,
            "what" | "which" | "who" | "where" | "when" | "the" | "was" | "were" | "are" | "did" | "does" | "for" | "with" | "from" | "that" | "associated" | "serves" | "name" | "known" | "current" | "hold" | "holds" | "pertain" | "belongs"
        ) {
            continue;
        }
        let st = deriv_root(w);
        if st.len() >= 3 {
            let s_idx = if let Some(p) = q_stems.iter().position(|x| x == &st) {
                p
            } else {
                let p = q_stems.len();
                q_stems.push(st);
                p
            };
            if s_idx < 64 {
                raw_stem_positions.push((pos, s_idx));
            }
        }
    }
    let mut q_phrase_links: Vec<(String, u64)> = Vec::new();
    let mut q_expanded: Vec<(String, u64)> = Vec::new();
    let push_pair = |dst: &mut Vec<(String, u64)>, term: &str, bits: u64| {
        if let Some(entry) = dst.iter_mut().find(|(s, _)| s == term) {
            entry.1 |= bits;
        } else {
            dst.push((term.to_string(), bits));
        }
    };
    for start in 0..lowers_deposs.len() {
        for len in 2..=3usize {
            if start + len > lowers_deposs.len() {
                break;
            }
            let ph = lowers_deposs[start..start + len].join(" ");
            if !crate::wordnet::senses(&ph).is_empty() {
                let mut ph_bits = 0u64;
                for &(p, s_idx) in &raw_stem_positions {
                    if p >= start && p < start + len {
                        ph_bits |= 1u64 << s_idx;
                    }
                }
                for cl in crate::wordnet::phrase_and_lemma_closure(&ph, 1) {
                    push_pair(&mut q_phrase_links, cl, ph_bits);
                    push_pair(&mut q_expanded, cl, ph_bits);
                }
            }
        }
    }
    for (pos, s_idx) in raw_stem_positions {
        let w = lowers_deposs[pos].as_str();
        let bit = 1u64 << s_idx;
        for cl in crate::wordnet::phrase_and_lemma_closure(w, 1) {
            push_pair(&mut q_phrase_links, cl, bit);
        }
        for hyp in crate::wordnet::expansion_lemmas(w)
            .into_iter()
            .chain(crate::wordnet::phrase_and_lemma_closure(w, 1))
            .chain(crate::wordnet::gloss_and_compound_concepts(w))
        {
            push_pair(&mut q_expanded, hyp, bit);
        }
        for (cn, _) in crate::conceptnet::two_hop_neighbors(w, 24) {
            push_pair(&mut q_expanded, cn, bit);
        }
    }

    let prep_hops = lowers
        .iter()
        .filter(|w| {
            matches!(
                w.as_str(),
                "of" | "for" | "where" | "which" | "whom" | "whose" | "by" | "to"
            ) || w.ends_with("'s")
        })
        .count();
    let max_hops = (prep_hops + 2).clamp(4, 5);

    #[derive(Clone)]
    struct BeamState {
        cur_subj: String,
        turns: Vec<usize>,
        visited: Vec<String>,
        used_mask: u64,
        score_q16: u64,
        seed_was_proper: bool,
        first_hop_matched: bool,
    }

    let mut beam: Vec<BeamState> = seed_subjs
        .iter()
        .take(3)
        .map(|&s| BeamState {
            cur_subj: s.to_string(),
            turns: Vec::new(),
            visited: vec![s.to_string()],
            used_mask: 0,
            score_q16: 0,
            seed_was_proper: is_proper_seed(s),
            first_hop_matched: false,
        })
        .collect();

    let mut best_chain: Option<(Vec<usize>, u64, u32)> = None;

    for hop in 0..max_hops {
        let mut next_beam: Vec<BeamState> = Vec::new();
        for st in &beam {
            let mut candidates: Vec<(&TorrensRecord, &str, bool)> = Vec::new();
            let stripped_subj = st
                .cur_subj
                .strip_prefix("the ")
                .unwrap_or(st.cur_subj.as_str())
                .rsplit_once(" of ")
                .map(|(_, tail)| tail);
            let mut fwd_degree = by_subj.get(st.cur_subj.as_str()).map_or(0, Vec::len);
            if let Some(edges) = by_subj.get(st.cur_subj.as_str()) {
                for &rec in edges {
                    candidates.push((rec, rec.val_phrase.as_str(), false));
                }
            }
            if let Some(tail_s) = stripped_subj
                && tail_s != st.cur_subj
                && let Some(edges) = by_subj.get(tail_s)
            {
                fwd_degree += edges.len();
                for &rec in edges {
                    candidates.push((rec, rec.val_phrase.as_str(), false));
                }
            }
            let inv_edges = by_val.get(st.cur_subj.as_str());
            let allow_inv = (hop == 0 && fwd_degree == 0)
                || (!st.turns.is_empty() && fwd_degree == 0 && inv_edges.is_some_and(|v| v.len() <= 2));
            if allow_inv && let Some(inv_list) = inv_edges {
                for &rec in inv_list {
                    candidates.push((rec, rec.subj_phrase.as_str(), true));
                }
            }
            for (rec, next_ent, is_inv) in candidates {
                if st.turns.contains(&rec.turn) || st.visited.iter().any(|v| v == next_ent) {
                    continue;
                }
                let (raw_hop_sc, next_mask) =
                    score_predicate_match_masked_q16(rec, &q_stems, st.used_mask, &q_phrase_links, &q_expanded);
                let hop_sc = if is_inv {
                    if raw_hop_sc == 0 {
                        0
                    } else {
                        raw_hop_sc.saturating_sub(Q16 >> 2).max(Q16 >> 1)
                    }
                } else if raw_hop_sc == 0
                    && hop == 1
                    && st.turns.len() == 1
                    && st.seed_was_proper
                    && st.first_hop_matched
                    && fwd_degree == 1
                {
                    // Idiomatic terminal relation (`"breathe his last"`, `"pass away"`) where Hop 0
                    // locked a proper entity + explicit cue and the intermediate entity has a unique live fact.
                    Q16 >> 1
                } else {
                    raw_hop_sc
                };
                let edge_gain = if hop_sc > 0 {
                    hop_sc
                } else if !is_inv && hop + 1 < max_hops && fwd_degree <= 24 {
                    // Unlexicalized intermediate forward bridge (`spouse -> married`, `origin -> born`)
                    Q16 >> 3
                } else {
                    0
                };
                if edge_gain == 0 {
                    continue;
                }
                let mut next_turns = st.turns.clone();
                next_turns.push(rec.turn);
                let next_score = st.score_q16.saturating_add(edge_gain);
                let cues = next_mask.count_ones();
                let prev_cues = st.used_mask.count_ones();
                let next_first_matched = if hop == 0 {
                    raw_hop_sc >= Q16
                } else {
                    st.first_hop_matched
                };

                // A multi-hop candidate (`len >= 2`) is admissible as a terminal answer ONLY when
                // its final hop matched an explicit query/ontology target (`hop_sc >= (Q16 >> 1)`)
                // and consumed a new query cue (`cues > prev_cues`, preventing overshoot past an already-solved target)
                // or is the unique 2-hop idiomatic target (`hop == 1 && fwd_degree == 1`).
                let solved_new_cue = cues > prev_cues || (hop == 1 && next_first_matched && fwd_degree == 1);
                if next_turns.len() >= 2
                    && hop_sc >= (Q16 >> 1)
                    && solved_new_cue
                    && (cues >= 2 || hop_sc >= Q16 || (hop == 1 && next_first_matched && fwd_degree == 1))
                {
                    let term_total = next_score.saturating_add((cues as u64) * Q16);
                    if best_chain
                        .as_ref()
                        .map_or(true, |(_, bs, _)| term_total > *bs)
                    {
                        best_chain = Some((next_turns.clone(), term_total, cues.max(2)));
                    }
                }

                let next_stripped = next_ent
                    .strip_prefix("the ")
                    .unwrap_or(next_ent)
                    .rsplit_once(" of ")
                    .map(|(_, tl)| tl);
                if by_subj.contains_key(next_ent)
                    || by_val.contains_key(next_ent)
                    || next_stripped.is_some_and(|tl| by_subj.contains_key(tl))
                {
                    let mut next_vis = st.visited.clone();
                    next_vis.push(next_ent.to_string());
                    next_beam.push(BeamState {
                        cur_subj: next_ent.to_string(),
                        turns: next_turns,
                        visited: next_vis,
                        used_mask: next_mask,
                        score_q16: next_score,
                        seed_was_proper: st.seed_was_proper,
                        first_hop_matched: next_first_matched,
                    });
                }
            }
        }
        if next_beam.is_empty() {
            break;
        }
        next_beam.sort_by(|a, b| {
            b.score_q16
                .cmp(&a.score_q16)
                .then_with(|| a.turns.cmp(&b.turns))
        });
        next_beam.truncate(64);
        beam = next_beam;
    }

    let Some((mut chain_turns, _, best_cues)) = best_chain else {
        return (Vec::new(), 0);
    };
    // Reverse so `[Terminus, ..., Hop1]` places the final answer turn at Slot 0!
    chain_turns.reverse();
    (chain_turns, best_cues)
}

/// Resolve 1-hop (`Where is Mary?`), 2-hop (`Where is the X?`), 3-hop (`Where was the X before the Y?`),
/// cardinal compass (`What is <dir> of the <Y>?` / `What is the <X> <dir> of?`), and trivalent transfer
/// (`Who did <A> <v> the <O> to?` / `What did <A> <v> to <B>?` / `Who <v> the <O>?`) state queries.
pub(in crate::retrieval) fn resolve_spatial_possession(
    ix: &Index,
    q_text: &str,
    sniper_where: bool,
) -> Vec<usize> {
    if ix.spatial_events.is_empty() {
        return Vec::new();
    }
    let cased = cased_word_tokens(q_text);
    if cased.len() < 3 {
        return Vec::new();
    }
    let lowers: Vec<String> = cased.iter().map(|t| norm_lower(t)).collect();
    let n = lowers.len();

    // 1. Cardinal Compass Deduction (`What is <dir> of the <Y>?` / `What is the <X> <dir> of?`)
    if n == 6 && lowers[0] == "what" && lowers[1] == "is" {
        if lowers[3] == "of"
            && lowers[4] == "the"
            && let Some(opp) = crate::wordnet::opposite_cardinal_direction(&lowers[2])
        {
            let dir = lowers[2].as_str();
            let y = lowers[5].as_str();
            for ev in ix.spatial_events.iter().rev() {
                if let SpatialEvent::Compass { turn, lhs, dir: ed, rhs } = ev
                    && ((ed == dir && rhs == y) || (lhs == y && ed == opp))
                {
                    return vec![*turn];
                }
            }
        } else if lowers[2] == "the"
            && lowers[5] == "of"
            && let Some(opp) = crate::wordnet::opposite_cardinal_direction(&lowers[4])
        {
            let x = lowers[3].as_str();
            let dir = lowers[4].as_str();
            for ev in ix.spatial_events.iter().rev() {
                if let SpatialEvent::Compass { turn, lhs, dir: ed, rhs } = ev
                    && ((lhs == x && ed == dir) || (ed == opp && rhs == x))
                {
                    return vec![*turn];
                }
            }
        }
    }

    // 2. Trivalent Transfer Deduction (`Who/What ... give/received ...`)
    if matches!(lowers[0].as_str(), "who" | "what") {
        // 2a: `"Who did <A> <v> the <O> to?"` (n == 7)
        if n == 7
            && lowers[0] == "who"
            && lowers[1] == "did"
            && starts_upper(cased[2])
            && lowers[4] == "the"
            && lowers[6] == "to"
        {
            let a = lowers[2].as_str();
            let o = lowers[5].as_str();
            for ev in ix.spatial_events.iter().rev() {
                if let SpatialEvent::Transfer { turn, giver, obj, .. } = ev
                    && giver == a
                    && obj == o
                {
                    return vec![*turn];
                }
            }
        }
        // 2b: `"What did <A> <v> to <B>?"` (n == 6)
        if n == 6
            && lowers[0] == "what"
            && lowers[1] == "did"
            && starts_upper(cased[2])
            && lowers[4] == "to"
            && starts_upper(cased[5])
        {
            let a = lowers[2].as_str();
            let b = lowers[5].as_str();
            for ev in ix.spatial_events.iter().rev() {
                if let SpatialEvent::Transfer { turn, giver, recipient, .. } = ev
                    && giver == a
                    && recipient == b
                {
                    return vec![*turn];
                }
            }
        }
        // 2c: `"Who <v> the <O> to <B>?"` (n == 6)
        if n == 6
            && lowers[0] == "who"
            && lowers[2] == "the"
            && lowers[4] == "to"
            && starts_upper(cased[5])
        {
            let o = lowers[3].as_str();
            let b = lowers[5].as_str();
            for ev in ix.spatial_events.iter().rev() {
                if let SpatialEvent::Transfer { turn, obj, recipient, .. } = ev
                    && obj == o
                    && recipient == b
                {
                    return vec![*turn];
                }
            }
        }
        // 2d: `"Who <v> the <O>?"` (n == 4)
        if n == 4 && lowers[0] == "who" && lowers[2] == "the" {
            let o = lowers[3].as_str();
            for ev in ix.spatial_events.iter().rev() {
                if let SpatialEvent::Transfer { turn, obj, .. } = ev
                    && obj == o
                {
                    return vec![*turn];
                }
            }
        }
    }

    if !sniper_where || lowers[0] != "where" {
        return Vec::new();
    }

    // 3. Direct Agent Location (`"Where is/was <CapAgent>?"`, n == 3)
    if n == 3 && matches!(lowers[1].as_str(), "is" | "was") && starts_upper(cased[2]) {
        let target_agent = lowers[2].as_str();
        for ev in ix.spatial_events.iter().rev() {
            if let SpatialEvent::Move { turn, agent, .. } = ev
                && agent == target_agent
            {
                return vec![*turn];
            }
        }
        return Vec::new();
    }

    if n < 4 || ix.spatial_events.len() < 2 {
        return Vec::new();
    }

    // 4. Parse `"Where is/was the <obj> [before [the] <before_loc>]?"`
    let before_pos = lowers.iter().position(|w| w == "before");
    let obj_end = before_pos.unwrap_or(lowers.len());
    let Some(the_pos) = lowers[..obj_end].iter().position(|w| w == "the") else {
        return Vec::new();
    };
    if the_pos + 1 >= obj_end {
        return Vec::new();
    }
    let target_obj = &lowers[the_pos + 1];
    let before_loc: Option<&str> = before_pos.and_then(|bp| {
        let rest = &lowers[bp + 1..];
        if rest.first().map(String::as_str) == Some("the") {
            rest.get(1).map(String::as_str)
        } else {
            rest.first().map(String::as_str)
        }
    });

    // Verify `target_obj` actually participates in at least one `Interact` event
    if !ix
        .spatial_events
        .iter()
        .any(|ev| matches!(ev, SpatialEvent::Interact { obj, .. } if obj == target_obj))
    {
        return Vec::new();
    }

    let mut agent_loc: BTreeMap<&str, (&str, usize)> = BTreeMap::new();
    let mut held_by: Option<&str> = None;
    let mut last_actor: Option<&str> = None;
    // History of distinct `(place, loc_turn_idx, event_turn_idx)` transitions for `target_obj`
    let mut obj_history: Vec<(&str, usize, usize)> = Vec::new();

    for ev in &ix.spatial_events {
        match ev {
            SpatialEvent::Move { turn, agent, place } => {
                agent_loc.insert(agent.as_str(), (place.as_str(), *turn));
                if held_by == Some(agent.as_str()) {
                    if let Some((last_p, last_loc_t, last_ev_t)) = obj_history.last_mut() {
                        if *last_p == place.as_str() {
                            *last_loc_t = *turn;
                            *last_ev_t = *turn;
                        } else {
                            obj_history.push((place.as_str(), *turn, *turn));
                        }
                    } else {
                        obj_history.push((place.as_str(), *turn, *turn));
                    }
                }
            }
            SpatialEvent::Interact {
                turn,
                agent,
                obj,
                explicit_release,
                explicit_acquire,
            } => {
                if obj != target_obj {
                    continue;
                }
                last_actor = Some(agent.as_str());
                if !*explicit_acquire && (*explicit_release || held_by == Some(agent.as_str())) {
                    // Agent drops/leaves `target_obj` at their current location (`loc_turn` carries room name)
                    if let Some(&(pl, loc_turn)) = agent_loc.get(agent.as_str()) {
                        if let Some((last_p, last_loc_t, last_ev_t)) = obj_history.last_mut() {
                            if *last_p == pl {
                                *last_loc_t = loc_turn;
                                *last_ev_t = *turn;
                            } else {
                                obj_history.push((pl, loc_turn, *turn));
                            }
                        } else {
                            obj_history.push((pl, loc_turn, *turn));
                        }
                    }
                    held_by = None;
                } else {
                    // Agent acquires `target_obj` (`loc_turn` carries room name, `*turn` carries pickup)
                    held_by = Some(agent.as_str());
                    if let Some(&(pl, loc_turn)) = agent_loc.get(agent.as_str()) {
                        if let Some((last_p, last_loc_t, last_ev_t)) = obj_history.last_mut() {
                            if *last_p == pl {
                                *last_loc_t = loc_turn;
                                *last_ev_t = *turn;
                            } else {
                                obj_history.push((pl, loc_turn, *turn));
                            }
                        } else {
                            obj_history.push((pl, loc_turn, *turn));
                        }
                    }
                }
            }
            SpatialEvent::Compass { .. } | SpatialEvent::Transfer { .. } => {}
        }
    }

    let push_uniq = |out: &mut Vec<usize>, cap: usize, ti: usize| {
        if out.len() < cap && !out.contains(&ti) {
            out.push(ti);
        }
    };

    if let Some(b_loc) = before_loc {
        // 1. Search backward for the latest occurrence of `b_loc` in `obj_history` with a preceding distinct location
        for idx in (1..obj_history.len()).rev() {
            if obj_history[idx].0 == b_loc && obj_history[idx - 1].0 != b_loc {
                let mut out = Vec::with_capacity(4);
                let (_, loc_t, ev_t) = obj_history[idx - 1];
                push_uniq(&mut out, 4, loc_t);
                push_uniq(&mut out, 4, ev_t);
                for k in (0..idx - 1).rev() {
                    push_uniq(&mut out, 4, obj_history[k].1);
                    push_uniq(&mut out, 4, obj_history[k].2);
                }
                return out;
            }
        }
        // 2. `agent_loc` fallback for `"dropped the <obj> in the <place> before leaving for the <b_loc>"`:
        if let Some(last_idx) = obj_history.len().checked_sub(1)
            && obj_history[last_idx].0 != b_loc
            && last_actor
                .and_then(|a| agent_loc.get(a))
                .is_some_and(|&(pl, _)| pl == b_loc)
        {
            let mut out = Vec::with_capacity(4);
            let (_, loc_t, ev_t) = obj_history[last_idx];
            push_uniq(&mut out, 4, loc_t);
            push_uniq(&mut out, 4, ev_t);
            for k in (0..last_idx).rev() {
                push_uniq(&mut out, 4, obj_history[k].1);
                push_uniq(&mut out, 4, obj_history[k].2);
            }
            return out;
        }
        Vec::new()
    } else if let Some(last_idx) = obj_history.len().checked_sub(1) {
        let mut out = Vec::with_capacity(4);
        let (_, loc_t, ev_t) = obj_history[last_idx];
        push_uniq(&mut out, 4, loc_t);
        push_uniq(&mut out, 4, ev_t);
        for k in (0..last_idx).rev() {
            push_uniq(&mut out, 4, obj_history[k].1);
            push_uniq(&mut out, 4, obj_history[k].2);
        }
        out
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_ordinal_and_parse_torrens_are_pure_integer_deterministic() {
        assert_eq!(
            strip_leading_ordinal("146. The author of Our Mutual Friend is Charles Darwin."),
            "The author of Our Mutual Friend is Charles Darwin."
        );
        let r1 = parse_torrens_sentence(
            "107. The author of Our Mutual Friend is Charles Dickens.",
            0,
        )
        .unwrap();
        let r2 = parse_torrens_sentence(
            "146. The author of Our Mutual Friend is Charles Darwin.",
            1,
        )
        .unwrap();
        assert_eq!(r1.key_hash, r2.key_hash);
        assert_eq!(r1.subj_phrase, "our mutual friend");
        assert_eq!(r2.val_phrase, "charles darwin");

        let r3 = parse_torrens_sentence("175. Igor of Kiev is married to Olga of Kiev.", 2)
            .unwrap();
        assert_eq!(r3.subj_phrase, "igor of kiev");
        assert_eq!(r3.val_phrase, "olga of kiev");

        let r4 = parse_torrens_sentence(
            "389. association football was created in the country of Italy.",
            3,
        )
        .unwrap();
        assert_eq!(r4.subj_phrase, "association football");
        assert_eq!(r4.val_phrase, "italy");

        let r5 = parse_torrens_sentence(
            "453. The univeristy where Wilhelm II was educated is Baldwin Wallace University.",
            4,
        )
        .unwrap();
        assert_eq!(r5.subj_phrase, "wilhelm ii");
        assert_eq!(r5.val_phrase, "baldwin wallace university");
    }

    #[test]
    fn multihop_beam_chain_and_premise_coverage_certificate() {
        use crate::retrieval::{rank, Params, Query, RankRequest, Session, Turn};
        let facts = [
            "107. The author of Our Mutual Friend is Charles Dickens.",
            "146. The author of Our Mutual Friend is Charles Darwin.",
            "199. Charles Darwin is married to Amala Paul.",
            "322. Amala Paul is a citizen of Belgium.",
            "401. Charles Dickens is a citizen of England.",
        ];
        let sessions: Vec<Session> = facts
            .iter()
            .enumerate()
            .map(|(i, &f)| Session {
                id: format!("f{i}"),
                date: String::new(),
                timestamp: i as i64,
                turns: vec![Turn {
                    id: format!("f{i}"),
                    role: "user".into(),
                    content: f.into(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            })
            .collect();
        let resp = rank(&RankRequest {
            sessions,
            queries: vec![
                Query {
                    text: "What is the country of citizenship of the spouse of the author of Our Mutual Friend?".into(),
                    date: String::new(),
                },
                Query {
                    text: "What telescope did Charles Darwin invent?".into(),
                    date: String::new(),
                },
                Query {
                    text: "Where is the submarine docked?".into(),
                    date: String::new(),
                },
            ],
            top_k: 5,
            params: Params::default(),
        });
        assert_eq!(resp.results[0].turns[0].id, "f3");
        assert_eq!(resp.results[1].premise_coverage.status, "PARTIAL_PREMISE");
        assert_eq!(resp.results[2].premise_coverage.status, "ABSTAIN");
    }

    #[test]
    fn mab_mh_chains_resolve_colin_irwin_laura_bush_and_free_as_a_bird() {
        use crate::retrieval::{rank, Params, Query, RankRequest, Session, Turn};
        let facts = [
            "1. The chairperson of Fatah is Mahmoud Abbas.",
            "48. Israel was founded by David Ben-Gurion.",
            "49. Israel was founded by Philippe, Duke of Orléans.",
            "55. Colin Irwin is a citizen of United Kingdom.",
            "84. Laura Bush is married to George W. Bush.",
            "94. Princess Alice, Duchess of Gloucester is affiliated with the religion of Anglicanism.",
            "98. Israel is located in the continent of Asia.",
            "109. The chief executive officer of Twitter is Jack Dorsey.",
            "125. Olga of Kiev died in the city of Kyiv.",
            "127. Israel is located in the continent of North America.",
            "129. Joe Hockey is a citizen of Australia.",
            "134. Moshe Kahlon works in the field of politician.",
            "135. Laura Bush is married to Princess Alice, Duchess of Gloucester.",
            "167. The capital of Italy is Rome.",
            "175. Igor of Kiev is married to Olga of Kiev.",
            "178. Steve Jobs was born in the city of San Francisco.",
            "179. The capital of Soviet Union is Russellville.",
            "188. The chief executive officer of Microsoft is Steve Jobs.",
            "209. The capital of Italy is Duluth.",
            "216. basketball was created in the country of United States of America.",
            "232. The chairperson of Fatah is Moshe Kahlon.",
            "236. The director of The Beatles is Brian Epstein.",
            "275. The chief executive officer of Twitter is Bernard Arnault.",
            "291. Surrey Senior Cup is associated with the sport of association football.",
            "313. Benjamin Mazar is a citizen of Israel.",
            "315. Surrey Senior Cup is associated with the sport of basketball.",
            "319. Bernard Arnault is a citizen of France.",
            "348. basketball was created in the country of Soviet Union.",
            "350. Free as a Bird was performed by The Beatles.",
            "368. Moshe Kahlon works in the field of soldier.",
            "397. San Francisco is located in the continent of North America.",
            "427. Joe Hockey is a citizen of Italy.",
            "441. Colin Irwin is a citizen of Israel.",
            "443. The director of The Beatles is Joe Hockey.",
            "450. Olga of Kiev died in the city of Rodez.",
        ];
        let sessions: Vec<Session> = facts
            .iter()
            .enumerate()
            .map(|(i, &f)| Session {
                id: format!("f{}", f.split('.').next().unwrap()),
                date: String::new(),
                timestamp: i as i64,
                turns: vec![Turn {
                    id: format!("f{}", f.split('.').next().unwrap()),
                    role: "user".into(),
                    content: f.into(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            })
            .collect();
        let resp = rank(&RankRequest {
            sessions,
            queries: vec![
                Query { text: "In which location did the spouse of Igor of Kiev pass away?".into(), date: String::new() },
                Query { text: "To which continent does the country of citizenship of Colin Irwin pertain?".into(), date: String::new() },
                Query { text: "Which faith is the partner of Laura Bush associated with?".into(), date: String::new() },
                Query { text: "Which country does the current CEO of Twitter hold citizenship of?".into(), date: String::new() },
                Query { text: "Which city is the capital of the country where the performer of \"Free as a Bird\"'s director/manager comes from?".into(), date: String::new() },
                Query { text: "What is the job title of the chairperson of the political organization Fatah?".into(), date: String::new() },
                Query { text: "From which capital city did the Surrey Senior Cup, a sports competition, originate?".into(), date: String::new() },
            ],
            top_k: 10,
            params: Params::default(),
        });
        assert_eq!(resp.results[0].turns[0].id, "f450", "Olga Rodez");
        assert_eq!(resp.results[1].turns[0].id, "f127", "Israel North America");
        assert_eq!(resp.results[2].turns[0].id, "f94", "Princess Alice Anglicanism");
        assert_eq!(resp.results[3].turns[0].id, "f319", "Bernard Arnault France");
        assert_eq!(resp.results[4].turns[0].id, "f209", "Italy Duluth");
        assert_eq!(resp.results[5].turns[0].id, "f368", "Fatah Moshe soldier");
        assert_eq!(resp.results[6].turns[0].id, "f179", "Surrey Soviet Russellville");
    }
}
