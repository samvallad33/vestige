//! Index construction: `Vocab`, `Bag`, `TurnIx`, `Index` and `build_index` (incl. ingest-time supersession edges).
//! Split out of the former single-file `retrieval.rs` with no logic change.

use super::*;

// =========================================================================
// INDEX
// =========================================================================

#[derive(Default)]
pub(super) struct Vocab {
    pub(super) map: DetHashMap<String, u32>,
    pub(super) names: Vec<String>,
}

impl Vocab {
    pub(super) fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.map.insert(s.to_string(), id);
        self.names.push(s.to_string());
        id
    }
    pub(super) fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
}

#[derive(Clone, Copy)]
pub(super) struct Feat {
    pub(super) id: u32,
    pub(super) w: f32,
    pub(super) counts_len: bool,
    pub(super) inheritable: bool,
}

#[derive(Default, Clone)]
pub(super) struct Bag {
    pub(super) tf: [DetHashMap<u32, f32>; 3],
    pub(super) len: [f32; 3],
}

impl Bag {
    pub(super) fn add(&mut self, field: usize, f: Feat, scale: f32) {
        *self.tf[field].entry(f.id).or_insert(0.0) += f.w * scale;
        if f.counts_len {
            self.len[field] += f.w * scale;
        }
    }

    pub(super) fn add_inherited(&mut self, field: usize, f: Feat, scale: f32) {
        *self.tf[field].entry(f.id).or_insert(0.0) += f.w * scale;
        if f.counts_len {
            self.len[field] += 0.50 * f.w * scale;
        }
    }
}

pub(super) struct StandingRule {
    pub(super) policy_key: String,
    pub(super) turn: usize,
    pub(super) session: usize,
    pub(super) ts: i64,
    pub(super) has_explicit_scope: bool,
    pub(super) scope: DetHashSet<u32>,
    pub(super) assistant_nouns: DetHashSet<u32>,
}

pub(super) struct TurnIx {
    pub(super) session: usize,
    pub(super) id: String,
    pub(super) field: usize,
    pub(super) feats: Vec<Feat>,
    pub(super) blip_feats: Vec<Feat>,
    /// Exact/stem term ids of content tokens in order, for span proximity.
    pub(super) seq: Vec<(u32, u32)>,
    pub(super) content: DetHashSet<u32>,
    /// Capitalized proper-entity term IDs inside this turn.
    pub(super) entities: DetHashSet<u32>,
    /// Body text (header line removed) for lazy exact title-span linking on hot anchors.
    pub(super) body_for_headers: String,
    pub(super) span_ids_cache: OnceLock<Vec<u32>>,
    pub(super) speaker: String,
    pub(super) is_question: bool,
    pub(super) is_pure_info_question: bool,
    pub(super) pre_question_ids: DetHashSet<u32>,
    pub(super) question_clause_ids: DetHashSet<u32>,
    pub(super) answers_question: bool,
    pub(super) is_short_reply: bool,
    pub(super) is_backchannel: bool,
    pub(super) anaphoric: bool,
    #[allow(dead_code)]
    pub(super) is_personal_constraint: bool,
    pub(super) has_temporal: bool,
    pub(super) has_numeric: bool,
    pub(super) has_spatial: bool,
    pub(super) has_causal: bool,
    pub(super) is_short_question: bool,
    pub(super) is_standing_policy: bool,
    /// User turn that opens a concrete event, not a rule and not a backchannel.
    pub(super) is_initiative: bool,
    /// Previous turn in this session with a different role. Position and role only.
    /// Backward only. A later turn points at the earlier one. There is no forward edge.
    pub(super) answers: Option<usize>,
    /// Caller & ingest-time RSB edges resolved to `(degree, target)` — strictly backward.
    pub(super) exact_edges: Vec<(u32, usize)>,
    /// SalienceJoin v2: earlier standing-policy turns this later turn points at.
    /// Promoted only when that policy is a standing rule that fired for the query.
    pub(super) policy_backlinks: Vec<usize>,
    /// Backward state-revision / contradiction edges (`t_new -> t_old`).
    pub(super) supersedes_turns: Vec<usize>,
    pub(super) stability: f32,
    /// ISO date or HH:MM in the turn text. Character form, not a month name.
    pub(super) form_clock: bool,
    /// An ASCII digit in the turn text. Number words do not count.
    pub(super) form_digit: bool,
    /// An @handle, a URL, or a path token.
    pub(super) form_handle: bool,
    /// A capitalized word of two or more letters that is not sentence-initial.
    pub(super) form_midcap: bool,
    /// Ingest-time count of mid-sentence capitalized tokens, digits, and quotes (Rung 1 `--no-dump` safe).
    pub(super) slot_entities: u16,
    /// Clean mid-sentence slot entities excluding first-person pronouns ("I", "I'm", "I've", "I'll", "I'd").
    pub(super) slot_entities_clean: u16,
    /// Step 9 ingest-time prose shape (`--no-dump` safe): list-marker lines and first game move.
    pub(super) list_item_count: u8,
    pub(super) first_move_num: u16,
    /// Step 14 ingest-time closed-class user preference/constraint role-object ids (`prose_body` only).
    pub(super) role_objs: Vec<u32>,
    pub(super) raw_links: Vec<TurnLink>,
    /// Dump-only. Full prose body, filled only when VESTIGE_REPLAY_DUMP is set.
    /// Never read by ranking.
    pub(super) body: String,
}

pub(super) struct Stats {
    pub(super) n: f32,
    pub(super) df: DetHashMap<u32, u32>,
    pub(super) avg: [f32; 3],
}

pub(super) fn stats_of(bags: &[Bag]) -> Stats {
    let mut df: DetHashMap<u32, u32> = DetHashMap::default();
    let mut sum = [0.0f32; 3];
    let mut cnt = [0.0f32; 3];
    for b in bags {
        let mut seen: DetHashSet<u32> = DetHashSet::default();
        for f in 0..3 {
            for (&t, &v) in &b.tf[f] {
                if v > 0.0 {
                    seen.insert(t);
                }
            }
            if b.len[f] > 0.0 {
                sum[f] += b.len[f];
                cnt[f] += 1.0;
            }
        }
        for t in seen {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    Stats {
        n: bags.len() as f32,
        df,
        avg: [
            if cnt[0] > 0.0 { sum[0] / cnt[0] } else { 1.0 },
            if cnt[1] > 0.0 { sum[1] / cnt[1] } else { 1.0 },
            if cnt[2] > 0.0 { sum[2] / cnt[2] } else { 1.0 },
        ],
    }
}

pub(super) struct HeaderKey {
    pub(super) tokens: Vec<u32>,
    pub(super) sessions: Vec<(usize, f32)>,
}

pub(super) struct Index {
    pub(super) vocab: Vocab,
    pub(super) session_ids: Vec<String>,
    pub(super) session_ts: Vec<i64>,
    pub(super) session_turns: Vec<Vec<usize>>,
    pub(super) session_entities: Vec<DetHashSet<u32>>,
    pub(super) session_content: Vec<DetHashSet<u32>>,
    /// Rarest non-hub content token IDs per session (sorted, up to 32) for GhostLink Ochiai divergence.
    pub(super) session_rare_nt: Vec<Vec<u32>>,
    /// Backward-only session RSB edges (`later_session -> (degree, earlier_session)`).
    pub(super) session_rsb_back: Vec<Vec<(u32, usize)>>,
    /// Backward-only session contradiction/supersession edges (`s_new -> s_old`).
    pub(super) session_supersedes_back: Vec<Vec<usize>>,
    /// Backward-only Temporal Context Episode neighbor (`si -> si - 1` when sharing episode/date anchor).
    pub(super) session_episode_prev: Vec<Option<usize>>,
    pub(super) turns: Vec<TurnIx>,
    pub(super) session_bags: Vec<Bag>,
    pub(super) turn_own_bags: Vec<Bag>,
    pub(super) turn_bags: Vec<Bag>,
    /// (session index, first turn position within session, last turn position inclusive)
    pub(super) windows: Vec<(usize, usize, usize)>,
    pub(super) window_bags: Vec<Bag>,
    pub(super) s_stats: Stats,
    pub(super) t_own_stats: Stats,
    pub(super) t_stats: Stats,
    pub(super) w_stats: Stats,
    /// Content-term ids that are the object of a personal constraint ("allergic to X").
    pub(super) constraint_terms: DetHashSet<u32>,
    /// Compiled SOAR/Rete standing rules for the stratified governing channel (`RankResult.standing_rules`).
    pub(super) standing_rules: Vec<StandingRule>,
    /// Exact `{Header}\\n{Body}` titles, hashed, for the session bridge.
    pub(super) header_dict: DetHashMap<u64, Vec<HeaderKey>>,
    pub(super) header_max_words: usize,
    /// Unique title-case speaker nickname bindings (`Mel` -> `melanie`).
    pub(super) nicks: SpeakerNickIndex,
    /// Recall cues: per turn, the sorted ids of the kernel tokens (`raw_tokens` + `stem`) of its
    /// full content, in a private id space (`cue_vocab`) so the ranker vocabulary is untouched.
    pub(super) turn_cues: Vec<Vec<u32>>,
    pub(super) cue_vocab: DetHashMap<String, u32>,
    /// Cue ids that occur in any turn's role / speaker field (structural metadata, not a list).
    pub(super) role_cues: DetHashSet<u32>,
    /// Sedgewick dense `u32` TokenId inverted postings (sessions, windows, turns).
    pub(super) s_postings: Vec<Vec<u32>>,
    pub(super) w_postings: Vec<Vec<u32>>,
    pub(super) t_postings: Vec<Vec<u32>>,
    /// Private vocabulary ID space for 1-hop ingest WordNet bridge lemmas so `ix.vocab`
    /// and all primary BM25F token IDs remain 100% bit-for-bit identical to `main`.
    pub(super) bridge_vocab: Vocab,
    /// 1-hop WordNet lemma IDs (in `bridge_vocab` ID space) indexed per turn.
    pub(super) turn_bridge_postings: Vec<Vec<u32>>,
    /// Corpus-learned Positive PMI associations `(partner_id, weight)` over rare content tokens.
    pub(super) ppmi_neighbors: DetHashMap<u32, Vec<(u32, f32)>>,
    /// O(1) precomputed supersession flag (`true` if any later turn supersedes `ti`).
    pub(super) turn_superseded: Vec<bool>,
    /// Burton Bloom 64-bit membership filters per session and per turn.
    #[allow(dead_code)]
    pub(super) session_bloom: Vec<u64>,
    #[allow(dead_code)]
    pub(super) turn_bloom: Vec<u64>,
    /// Cache-aligned `<= 24B` turn metadata (20,000 turns <= 480 KB, fits in Sandy Bridge L3).
    #[allow(dead_code)]
    pub(super) turn_meta: Vec<TurnMeta>,
}

/// Compact cache-aligned turn metadata (`<= 24 bytes`) for L3-resident linear scans.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct TurnMeta {
    pub(super) bloom: u64,
    pub(super) session: u32,
    pub(super) content_len: u16,
    pub(super) entity_len: u16,
    pub(super) flags: u16,
    pub(super) field: u8,
    pub(super) _pad: u8,
}

const _: () = assert!(std::mem::size_of::<TurnMeta>() <= 24);

pub(super) const TM_QUESTION: u16 = 1 << 0;
pub(super) const TM_PURE_INFO_Q: u16 = 1 << 1;
pub(super) const TM_ANSWERS_Q: u16 = 1 << 2;
pub(super) const TM_SHORT_REPLY: u16 = 1 << 3;
pub(super) const TM_BACKCHANNEL: u16 = 1 << 4;
pub(super) const TM_ANAPHORIC: u16 = 1 << 5;
pub(super) const TM_CONSTRAINT: u16 = 1 << 6;
pub(super) const TM_STANDING_POLICY: u16 = 1 << 8;

pub(super) fn build_turn_meta(turns: &[TurnIx], turn_bloom: &[u64]) -> Vec<TurnMeta> {
    turns
        .iter()
        .zip(turn_bloom.iter())
        .map(|(t, &bloom)| {
            let mut flags = 0u16;
            if t.is_question { flags |= TM_QUESTION; }
            if t.is_pure_info_question { flags |= TM_PURE_INFO_Q; }
            if t.answers_question { flags |= TM_ANSWERS_Q; }
            if t.is_short_reply { flags |= TM_SHORT_REPLY; }
            if t.is_backchannel { flags |= TM_BACKCHANNEL; }
            if t.anaphoric { flags |= TM_ANAPHORIC; }
            if t.is_personal_constraint { flags |= TM_CONSTRAINT; }
            if t.is_standing_policy { flags |= TM_STANDING_POLICY; }
            TurnMeta {
                bloom,
                session: t.session as u32,
                content_len: t.content.len().min(u16::MAX as usize) as u16,
                entity_len: t.entities.len().min(u16::MAX as usize) as u16,
                flags,
                field: t.field as u8,
                _pad: 0,
            }
        })
        .collect()
}

/// Burton Bloom (1970) 64-bit 2-hash bitmask for a dense `u32` TokenId.
#[inline]
pub(super) fn bloom_mask64(id: u32) -> u64 {
    let mut x = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let b1 = (x & 63) as u32;
    let b2 = ((x >> 6) & 63) as u32;
    (1u64 << b1) | (1u64 << b2)
}

#[allow(dead_code)]
#[inline]
pub(super) fn session_may_contain_token(ix: &Index, si: usize, id: u32) -> bool {
    let m = bloom_mask64(id);
    (ix.session_bloom[si] & m) == m
}

pub(super) fn build_postings(bags: &[Bag], vocab_len: usize) -> Vec<Vec<u32>> {
    let mut postings = vec![Vec::new(); vocab_len];
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    for (doc_idx, b) in bags.iter().enumerate() {
        seen.clear();
        for f in 0..3 {
            for (&t, &v) in &b.tf[f] {
                if v > 0.0 {
                    seen.insert(t);
                }
            }
        }
        let d = doc_idx as u32;
        for &t in &seen {
            if let Some(list) = postings.get_mut(t as usize) {
                list.push(d);
            }
        }
    }
    postings
}

pub(super) fn build_turn_postings_and_bloom(
    turn_bags: &[Bag],
    turn_own_bags: &[Bag],
    vocab_len: usize,
) -> (Vec<Vec<u32>>, Vec<u64>) {
    let mut postings = vec![Vec::new(); vocab_len];
    let mut blooms = vec![0u64; turn_bags.len()];
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    for (ti, (tb, ob)) in turn_bags.iter().zip(turn_own_bags.iter()).enumerate() {
        seen.clear();
        for f in 0..3 {
            for (&t, &v) in &tb.tf[f] {
                if v > 0.0 {
                    seen.insert(t);
                }
            }
            for (&t, &v) in &ob.tf[f] {
                if v > 0.0 {
                    seen.insert(t);
                }
            }
        }
        let d = ti as u32;
        let mut mask = 0u64;
        for &t in &seen {
            mask |= bloom_mask64(t);
            if let Some(list) = postings.get_mut(t as usize) {
                list.push(d);
            }
        }
        blooms[ti] = mask;
    }
    (postings, blooms)
}

pub(super) fn build_session_bloom(bags: &[Bag]) -> Vec<u64> {
    bags.iter()
        .map(|b| {
            let mut mask = 0u64;
            for f in 0..3 {
                for (&t, &v) in &b.tf[f] {
                    if v > 0.0 {
                        mask |= bloom_mask64(t);
                    }
                }
            }
            mask
        })
        .collect()
}

pub(super) fn collect_posting_candidates(postings: &[Vec<u32>], plan: &QueryPlan, total_docs: usize) -> Vec<usize> {
    if total_docs == 0 || plan.terms.is_empty() {
        return Vec::new();
    }
    let mut seen = vec![0u8; total_docs];
    let mut out = Vec::new();
    for qt in &plan.terms {
        if let Some(list) = postings.get(qt.id as usize) {
            for &doc in list {
                let idx = doc as usize;
                if seen[idx] == 0 {
                    seen[idx] = 1;
                    out.push(idx);
                }
            }
        }
    }
    out
}

/// Kernel tokens of `text` for recall-cue matching: the same tokenizer and stemmer as ranking.
pub(super) fn cue_tokens(text: &str) -> Vec<String> {
    raw_tokens(text).into_iter().map(|w| stem(&w)).collect()
}

pub(super) fn intern_cues(text: &str, vocab: &mut DetHashMap<String, u32>) -> Vec<u32> {
    let mut ids: Vec<u32> = cue_tokens(text)
        .into_iter()
        .map(|w| {
            let next = vocab.len() as u32;
            *vocab.entry(w).or_insert(next)
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub(super) fn stem_term(s: &str) -> String {
    format!("\u{1}stem:{s}")
}

/// Object tokens of a personal constraint. The marker is not enough; the thing
/// constrained has to be in the index or a later question cannot hit it.
/// Parse both compact `key:value` tokens and structured `Key: Value` clauses
/// (excluding URL schemes `http:`/`https:` and `[BLIP caption]`), emitting exact
/// `(slot_key, slot_value)` pairs so typed attributes affect ranking and supersession.
pub(super) fn typed_slot_tokens(text: &str) -> Vec<(String, String)> {
    // URI schemes are protocol structure, not a topical vocabulary.
    const SKIP_KEYS: &[&str] = &["http", "https", "mailto", "file"];
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed_line = line.trim();
        if trimmed_line.starts_with('[') || trimmed_line.starts_with("```") {
            continue;
        }
        // Check structured `Key: Value` at line/clause head (up to 2 words for key, 1-2 words for value).
        if let Some((lhs, rhs)) = trimmed_line.split_once(':') {
            let key_words: Vec<&str> = lhs.split_whitespace().collect();
            if (1..=2).contains(&key_words.len())
                && key_words.iter().all(|w| w.len() >= 2 && w.chars().all(|c| c.is_ascii_alphabetic() || c == '_' || c == '-'))
            {
                let norm_key = key_words.join("_").to_lowercase();
                if !SKIP_KEYS.contains(&norm_key.as_str()) {
                    for val in rhs
                        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-' && c != '$')
                        .filter(|w| !w.is_empty())
                        .take(2)
                    {
                        let clean_val = val.trim_matches('$').to_lowercase();
                        if clean_val.len() >= 2 && !stopwords().contains(clean_val.as_str()) {
                            out.push((norm_key.clone(), clean_val));
                        }
                    }
                }
            }
        }
        // Also check inline `key:value` tokens without spaces.
        for raw in trimmed_line.split_whitespace() {
            let tok = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != ':' && c != '_' && c != '-');
            let Some((key, value)) = tok.split_once(':') else { continue };
            let kl = key.to_lowercase();
            if kl.len() >= 2
                && kl.chars().all(|c| c.is_ascii_alphabetic())
                && !SKIP_KEYS.contains(&kl.as_str())
                && !value.is_empty()
                && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                out.push((kl, value.to_lowercase()));
            }
        }
    }
    out
}

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

/// Closed-class English determiners (articles, demonstratives, possessives, quantifiers).
pub(super) const DETERMINERS: &[&str] =
    // hardcode-scan: closed-class (determiners)
    &["a", "an", "the", "this", "that", "these", "those", "my", "your", "our", "their", "its", "his", "her", "every", "each", "all", "any", "no", "some"];

/// Scope words are the turn's own content tokens. No directive-verb list.
pub(super) fn extract_rule_scope(text: &str) -> (bool, Vec<String>) {
    let (prose, _) = split_blip_caption(text);
    (
        false,
        raw_tokens(prose).into_iter().filter(|w| is_content(w)).collect(),
    )
}

/// Pure byte-exact technical handle extractor for Retroactive Salience Backfill (RSB):
/// URLs, emails, commit SHAs (7..=40 hex with >=1 alpha + >=1 digit), issue refs (`PROJ-123`),
/// version tags (`v1.2.3`), and file paths (`dir/file.ext`). Zero word similarity.
pub(super) fn extract_exact_technical_handles(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = DetHashSet::default();
    for raw_tok in text.split_whitespace() {
        let tok = raw_tok.trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';' | '"' | '\'' | '`'));
        let tok_clean = tok.trim_end_matches('.');
        if tok_clean.len() < 3 {
            continue;
        }
        // 1. URL
        if tok_clean.starts_with("http://") || tok_clean.starts_with("https://") {
            let h = format!("\u{1}h:url:{}", tok_clean.to_lowercase());
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 2. Email
        if let Some((local, dom)) = tok_clean.split_once('@') {
            if !local.is_empty()
                && dom.contains('.')
                && !dom.starts_with('.')
                && !dom.ends_with('.')
                && local.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-'))
                && dom.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-'))
            {
                let h = format!("\u{1}h:email:{}", tok_clean.to_lowercase());
                if seen.insert(h.clone()) {
                    out.push(h);
                }
                continue;
            }
        }
        // 3. IssueRef (`ABC-123`)
        if let Some((proj, num)) = tok_clean.split_once('-') {
            if (2..=10).contains(&proj.len())
                && proj.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                && proj.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                && !num.is_empty()
                && num.len() <= 7
                && num.chars().all(|c| c.is_ascii_digit())
            {
                let h = format!("\u{1}h:issue:{}", tok_clean.to_lowercase());
                if seen.insert(h.clone()) {
                    out.push(h);
                }
                continue;
            }
        }
        // 4. CommitSha (7..=40 hex with >=1 hex letter and >=1 digit)
        if (7..=40).contains(&tok_clean.len())
            && tok_clean.chars().all(|c| c.is_ascii_hexdigit())
            && tok_clean.chars().any(|c| c.is_ascii_digit())
            && tok_clean.chars().any(|c| matches!(c.to_ascii_lowercase(), 'a'..='f'))
        {
            let h = format!("\u{1}h:sha:{}", &tok_clean.to_lowercase()[..7.min(tok_clean.len())]);
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 5. Version (`v1.2` or `1.2.3`)
        let ver_cand = tok_clean.strip_prefix('v').or_else(|| tok_clean.strip_prefix('V')).unwrap_or(tok_clean);
        let ver_parts: Vec<&str> = ver_cand.split('.').collect();
        if ver_parts.len() >= 2
            && (tok_clean.starts_with('v') || tok_clean.starts_with('V') || ver_parts.len() >= 3)
            && ver_parts.iter().all(|p| !p.is_empty() && p.len() <= 5 && p.chars().all(|c| c.is_ascii_digit()))
        {
            let h = format!("\u{1}h:ver:{ver_cand}");
            if seen.insert(h.clone()) {
                out.push(h);
            }
            continue;
        }
        // 6. FilePath (`dir/file.ext` or `src\file.rs`)
        let norm_path = tok_clean.replace('\\', "/");
        if norm_path.contains('/')
            && !norm_path.contains("://")
            && norm_path.chars().any(|c| c.is_ascii_alphabetic())
            && norm_path.rsplit('/').next().is_some_and(|leaf| leaf.contains('.') && leaf.len() >= 3)
        {
            let h = format!("\u{1}h:path:{}", norm_path.to_lowercase());
            if seen.insert(h.clone()) {
                out.push(h);
            }
        }
    }
    out
}

/// Technical-handle term ids (`\u{1}h:…`) are linked only by backward RSB edges.
/// Undirected PPR must not reactivate them, or a later zero-overlap session is pulled forward.
pub(super) fn is_technical_handle(vocab: &Vocab, id: u32) -> bool {
    vocab.names.get(id as usize).is_some_and(|name| name.starts_with("\u{1}h:"))
}

/// Episode prefix before `_s{digits}` when sub-sessions belong to one contiguous batch,
/// else empty string (never inspects benchmark labels).
pub(super) fn session_episode_key<'a>(session_id: &'a str, date: &'a str) -> String {
    if let Some((prefix, suffix)) = session_id.rsplit_once("_s") {
        if !prefix.is_empty() && !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return format!("ep:{prefix}");
        }
    }
    let trimmed = date.trim();
    if !trimmed.is_empty() {
        return format!("dt:{trimmed}");
    }
    String::new()
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

pub(super) fn detect_question(content: &str, raw_len: usize) -> (bool, bool) {
    let prose = content.split("```").next().unwrap_or(content).trim_end();
    if let Some(q_pos) = prose.rfind('?') {
        let tail = prose[q_pos + 1..].trim();
        let ends_with_q = tail.is_empty() || tail.chars().all(|c| !c.is_ascii_alphabetic());
        let is_q = ends_with_q || prose.len() <= 350;
        let is_short_q = ends_with_q && raw_len < 10;
        return (is_q, is_short_q);
    }
    (false, false)
}

/// Closed-class interrogatives and auxiliaries. Not a topical vocabulary.
// hardcode-scan: closed-class (question words)
pub(super) const INFO_Q_WORDS: &[&str] = &[
    "who", "whom", "whose", "what", "where", "when", "why", "which", "how",
    "did", "do", "does", "is", "are", "was", "were",
    "have", "has", "had", "can", "could", "would", "should", "will",
];

pub(super) fn take_word(text: &str) -> Option<(&str, &str)> {
    let text = text.trim_start();
    let end = text
        .char_indices()
        .find(|(_, c)| !c.is_ascii_alphanumeric())
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    if end == 0 { None } else { Some((&text[..end], &text[end..])) }
}

pub(super) fn skip_comma(text: &str) -> &str {
    let s = text.trim_start();
    s.strip_prefix(',').unwrap_or(s).trim_start()
}

pub(super) fn is_speaker_alias(tok: &str, names: &DetHashSet<String>) -> bool {
    if names.contains(tok) {
        return true;
    }
    if tok.len() < 3 || stopwords().contains(tok) {
        return false;
    }
    names.iter().any(|n| n.len() >= tok.len() + 3 && n.starts_with(tok))
}

pub(super) fn strip_one_question_lead<'a>(text: &'a str, names: &DetHashSet<String>) -> Option<&'a str> {
    let s = text.trim_start();
    let (tok, rest) = take_word(s)?;
    let low = tok.to_ascii_lowercase();
    // A corpus-attested speaker name followed by a comma is a vocative, not content.
    if is_speaker_alias(&low, names) && rest.trim_start().starts_with(',') {
        return Some(skip_comma(rest));
    }
    None
}

pub(super) fn info_question_start(clause: &str) -> bool {
    let s = clause.trim_start();
    let Some((tok, rest)) = take_word(s) else { return false };
    let low = tok.to_ascii_lowercase();
    if low == "what" {
        if let Some((next, _)) = take_word(skip_comma(rest)) {
            let n = next.to_ascii_lowercase();
            if n == "a" || n == "an" {
                return false;
            }
        }
    }
    INFO_Q_WORDS.contains(&low.as_str())
}

pub(super) fn stripped_question_clause<'a>(clause: &'a str, names: &DetHashSet<String>) -> &'a str {
    let mut s = clause.trim();
    for _ in 0..8 {
        if info_question_start(s) {
            return s;
        }
        match strip_one_question_lead(s, names) {
            Some(next) => {
                let next = next.trim_start();
                if next.is_empty() || next.len() == s.len() {
                    return s;
                }
                s = next;
            }
            None => return s,
        }
    }
    s
}

pub(super) fn last_sentence_boundary(head: &str) -> Option<usize> {
    let bytes = head.as_bytes();
    let mut end = head.len();
    while end > 0 {
        let rel = head[..end].rfind(|c: char| matches!(c, '.' | '!' | ';' | '\n'))?;
        let ok = match bytes[rel] {
            b'.' => {
                let prev_digit = rel > 0 && bytes[rel - 1].is_ascii_digit();
                let next_ok = head[rel + 1..].chars().next().is_none_or(|c| c.is_whitespace());
                !prev_digit && next_ok
            }
            _ => true,
        };
        if ok {
            return Some(rel);
        }
        end = rel;
    }
    None
}

pub(super) fn terminal_question_spans(prose: &str) -> Option<(&str, &str)> {
    let prose = prose.split("```").next().unwrap_or(prose);
    let q_pos = prose.rfind('?')?;
    let tail = prose[q_pos + 1..].trim();
    if !(tail.is_empty() || tail.chars().all(|c| !c.is_ascii_alphabetic())) {
        return None;
    }
    let head = &prose[..q_pos];
    match last_sentence_boundary(head) {
        Some(i) => Some((&head[..i], head[i + 1..].trim())),
        None => Some(("", head.trim())),
    }
}

pub(super) fn token_is_temporal_word(w: &str) -> bool {
    MONTHS.contains(&w)
        || WEEKDAYS.contains(&w)
        || [
            "yesterday", "tomorrow", "ago", "today", "tonight",
            "week", "weeks", "weekend", "month", "months", "year", "years",
            "days", "spring", "summer", "autumn", "winter",
            "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        ]
        .contains(&w)
        || (w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
}

pub(super) fn pre_has_answer_payload(pre: &str) -> bool {
    if pre.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    raw_tokens(pre).iter().any(|w| token_is_temporal_word(w))
}

pub(super) fn content_term_ids(vocab: &mut Vocab, text: &str, stemming: bool) -> DetHashSet<u32> {
    let mut ids = DetHashSet::default();
    for w in raw_tokens(text).into_iter().filter(|w| is_content(w)) {
        ids.insert(vocab.intern(&w));
        if stemming {
            ids.insert(vocab.intern(&stem_term(&stem(&w))));
        }
    }
    ids
}

pub(super) struct QuestionShape {
    pub(super) pre_ids: DetHashSet<u32>,
    pub(super) clause_ids: DetHashSet<u32>,
    pub(super) is_pure: bool,
}

pub(super) fn analyze_question(
    vocab: &mut Vocab,
    prose: &str,
    is_question: bool,
    stemming: bool,
    names: &DetHashSet<String>,
) -> QuestionShape {
    let lower = prose.to_ascii_lowercase();
    let legacy = is_question && info_question_start(lower.trim_start());
    let tag = {
        let t = lower.trim();
        t.ends_with("right?") || t.ends_with("yeah?") || t.ends_with("huh?")
    };
    let Some((pre, clause)) = terminal_question_spans(prose) else {
        return QuestionShape {
            pre_ids: DetHashSet::default(),
            clause_ids: DetHashSet::default(),
            is_pure: is_question && legacy && !tag,
        };
    };
    let stripped = stripped_question_clause(clause, names);
    let terminal_info = info_question_start(stripped);
    let payload = pre_has_answer_payload(pre);
    QuestionShape {
        pre_ids: content_term_ids(vocab, pre, stemming),
        clause_ids: content_term_ids(vocab, clause, stemming),
        is_pure: is_question && !tag && !payload && (legacy || terminal_info),
    }
}

pub(super) fn camel_case_parts(ident: &str) -> Vec<String> {
    let chars: Vec<char> = ident.chars().collect();
    let mut seen_lower = false;
    let mut gated = false;
    for &c in &chars {
        if c.is_ascii_lowercase() {
            seen_lower = true;
        } else if seen_lower && c.is_ascii_uppercase() {
            gated = true;
            break;
        }
    }
    if !gated {
        return Vec::new();
    }
    let mut raw_parts: Vec<String> = Vec::new();
    let mut start = 0usize;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let hump = (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase();
        let acronym_tail = prev.is_ascii_uppercase()
            && cur.is_ascii_uppercase()
            && chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
        if hump || acronym_tail {
            raw_parts.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    raw_parts.push(chars[start..].iter().collect());
    if raw_parts.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for part in raw_parts {
        let lower = part.to_lowercase();
        if lower.chars().count() < 3 || !is_content(&lower) || out.contains(&lower) {
            continue;
        }
        out.push(lower);
    }
    out
}

pub(super) fn surface_aux_tokens(word: &str) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    if word.contains('_') || word.contains('-') {
        let joined: String = word
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if (4..=48).contains(&joined.len()) && joined.chars().any(|c| c.is_ascii_alphabetic()) {
            out.push((joined, 1.0));
        }
    }
    for piece in word.split(|c: char| !c.is_ascii_alphanumeric()) {
        if piece.is_empty() {
            continue;
        }
        for part in camel_case_parts(piece) {
            if !out.iter().any(|(p, _)| p == &part) {
                out.push((part, 0.85));
            }
        }
    }
    out
}

pub(super) const NICK_TAIL: usize = 3;

pub(super) struct SpeakerNickIndex {
    pub(super) prefix_of: DetHashMap<String, String>,
    pub(super) witnessed: DetHashMap<String, DetHashSet<String>>,
    pub(super) lower_attested: DetHashSet<String>,
}

pub(super) fn common_speaker_tokens() -> &'static DetHashSet<String> {
    static S: OnceLock<DetHashSet<String>> = OnceLock::new();
    S.get_or_init(|| {
        let mut s = DetHashSet::default();
        for w in STOPWORDS
            .iter()
            .chain(STARTER_STOP)
            .chain(MONTHS.iter())
            .chain(WEEKDAYS.iter())
        {
            s.insert((*w).to_string());
        }
        for word in MONTHS.iter().chain(WEEKDAYS.iter()) {
            for len in 2..word.len() {
                s.insert(word[..len].to_string());
            }
        }
        s
    })
}

pub(super) fn common_speaker_token(w: &str) -> bool {
    common_speaker_tokens().contains(w)
}

pub(super) fn inflectional_tail(nick: &str, full: &str) -> bool {
    matches!(
        full.strip_prefix(nick),
        Some("s" | "es" | "ed" | "ing" | "er" | "ers" | "ly")
    )
}

pub(super) struct SurfaceTok {
    pub(super) lower: String,
    pub(super) title_case: bool,
    pub(super) sentence_initial: bool,
}

pub(super) fn surface_tokens(text: &str) -> Vec<SurfaceTok> {
    let mut out = Vec::new();
    let mut sentence_initial = true;
    let mut current = String::new();
    let flush = |current: &mut String, sentence_initial: bool, out: &mut Vec<SurfaceTok>| {
        if current.is_empty() {
            return;
        }
        let piece = std::mem::take(current);
        let owned = piece.clone();
        let lower_piece = owned.to_lowercase();
        let cleaned_lower = clean_contraction(&lower_piece);
        let cleaned_raw = if cleaned_lower.len() == lower_piece.len() {
            owned
        } else {
            let keep = cleaned_lower.chars().count();
            owned.chars().take(keep).collect()
        };
        if cleaned_raw.is_empty() {
            return;
        }
        let mut chars = cleaned_raw.chars();
        let first_upper = chars.next().is_some_and(|c| c.is_ascii_uppercase());
        let rest_lower = chars.all(|c| !c.is_ascii_uppercase());
        let lower: String = cleaned_raw.chars().flat_map(|c| c.to_lowercase()).collect();
        if lower.is_empty() {
            return;
        }
        out.push(SurfaceTok {
            lower,
            title_case: first_upper && rest_lower,
            sentence_initial,
        });
    };
    for ch in text.chars() {
        let keep = ch.is_alphanumeric() || ch == '\'' || ch == '’';
        if keep {
            current.push(ch);
            continue;
        }
        let was_empty = current.is_empty();
        flush(&mut current, sentence_initial, &mut out);
        if matches!(ch, '.' | '?' | '!' | '\n') {
            sentence_initial = true;
        } else if !was_empty {
            sentence_initial = false;
        }
    }
    flush(&mut current, sentence_initial, &mut out);
    out
}

impl SpeakerNickIndex {
    pub(super) fn build(sessions: &[Session]) -> Self {
        let mut exact_roles: DetHashSet<String> = DetHashSet::default();
        let mut targets: DetHashSet<String> = DetHashSet::default();
        let mut lower_attested: DetHashSet<String> = DetHashSet::default();
        let mut texts: Vec<&str> = Vec::new();
        for s in sessions {
            for t in &s.turns {
                texts.push(t.content.as_str());
                let role = t.role.trim();
                if role.is_empty()
                    || role.eq_ignore_ascii_case("user")
                    || role.eq_ignore_ascii_case("assistant")
                {
                    continue;
                }
                for w in raw_tokens(role) {
                    exact_roles.insert(w.clone());
                    if w.len() >= 4
                        && !STARTER_STOP.contains(&w.as_str())
                        && !stopwords().contains(w.as_str())
                    {
                        targets.insert(w);
                    }
                }
            }
        }
        for text in &texts {
            for tok in surface_tokens(text) {
                if !tok.title_case {
                    lower_attested.insert(tok.lower);
                }
            }
        }
        let mut hits: DetHashMap<String, Vec<String>> = DetHashMap::default();
        let mut ordered: Vec<&String> = targets.iter().collect();
        ordered.sort();
        for name in ordered {
            for len in 2..=5 {
                if name.len() < len + NICK_TAIL {
                    continue;
                }
                let p = &name[..len];
                if exact_roles.contains(p) || common_speaker_token(p) || inflectional_tail(p, name) {
                    continue;
                }
                hits.entry(p.to_string()).or_default().push(name.clone());
            }
        }
        let mut prefix_of: DetHashMap<String, String> = DetHashMap::default();
        for (p, mut names) in hits {
            names.sort();
            names.dedup();
            if names.len() == 1 {
                prefix_of.insert(p, names.pop().unwrap());
            }
        }
        let mut witnessed: DetHashMap<String, DetHashSet<String>> = DetHashMap::default();
        for text in &texts {
            for tok in surface_tokens(text) {
                if !tok.title_case {
                    continue;
                }
                let Some(full) = prefix_of.get(&tok.lower) else { continue };
                if tok.sentence_initial && lower_attested.contains(&tok.lower) {
                    continue;
                }
                witnessed.entry(full.clone()).or_default().insert(tok.lower);
            }
        }
        Self { prefix_of, witnessed, lower_attested }
    }

    pub(super) fn bind_token(&self, tok: &SurfaceTok, sentence_initial: bool) -> Option<&str> {
        if !tok.title_case || common_speaker_token(&tok.lower) {
            return None;
        }
        let full = self.prefix_of.get(&tok.lower)?;
        if sentence_initial && self.lower_attested.contains(&tok.lower) {
            return None;
        }
        let seen = self.witnessed.get(full);
        let witnessed_this = seen.is_some_and(|s| s.contains(&tok.lower));
        if !(3..=4).contains(&tok.lower.len()) && !witnessed_this {
            return None;
        }
        if seen.is_some_and(|s| !s.is_empty() && !s.contains(&tok.lower)) {
            return None;
        }
        Some(full.as_str())
    }

    pub(super) fn bindings_in(&self, text: &str) -> DetHashMap<String, String> {
        let mut out = DetHashMap::default();
        for tok in surface_tokens(text) {
            if let Some(full) = self.bind_token(&tok, tok.sentence_initial) {
                out.insert(tok.lower, full.to_string());
            }
        }
        out
    }
}

pub(super) fn digit_token_set(text: &str) -> DetHashSet<String> {
    raw_tokens(text)
        .into_iter()
        .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
        .collect()
}

pub(super) fn closed_locative(w: &str) -> bool {
    matches!(w, "in" | "at" | "to" | "from" | "on" | "into" | "over" | "by")
}

pub(super) fn build_index(sessions: &[Session], p: &Params) -> Index {
    let nicks = SpeakerNickIndex::build(sessions);
    let mut vocab = Vocab::default();
    let mut turns: Vec<TurnIx> = Vec::new();
    let mut session_ids = Vec::new();
    let mut session_ts = Vec::new();
    let mut session_ep_keys = Vec::with_capacity(sessions.len());
    let mut session_turns = Vec::new();
    let mut session_entities = Vec::with_capacity(sessions.len());
    let mut session_content = Vec::with_capacity(sessions.len());
    let mut raw_standing_rules: Vec<(String, usize, usize, i64, bool, Vec<String>)> = Vec::new();
    let mut turn_tech_handles: Vec<Vec<u32>> = Vec::new();
    let mut turn_prose_cache: Vec<String> = Vec::new();
    let mut cue_vocab: DetHashMap<String, u32> = DetHashMap::default();
    let mut turn_cues: Vec<Vec<u32>> = Vec::new();
    let mut role_cues: DetHashSet<u32> = DetHashSet::default();
    let mut constraint_terms: DetHashSet<u32> = DetHashSet::default();
    let mut header_dict: DetHashMap<u64, Vec<HeaderKey>> = DetHashMap::default();
    let mut header_max_words = 0usize;
    for (si, s) in sessions.iter().enumerate() {
        let Some(first) = s.turns.first() else { continue };
        let Some((full_h, core_h)) = extract_header_titles(&first.content) else { continue };
        for (span_str, mult) in [(full_h, 1.0f32), (core_h, 0.88f32)] {
            let htoks = norm_span_tokens(span_str);
            if htoks.is_empty() || htoks.len() > 10 {
                continue;
            }
            if htoks.len() == 1 && (htoks[0].len() < 3 || stopwords().contains(htoks[0].as_str())) {
                continue;
            }
            header_max_words = header_max_words.max(htoks.len());
            let hids: Vec<u32> = htoks.iter().map(|w| vocab.intern(w)).collect();
            let hh = hash_u32_slice(&hids);
            let bucket = header_dict.entry(hh).or_default();
            if let Some(existing) = bucket.iter_mut().find(|k| k.tokens == hids) {
                if let Some(pair) = existing.sessions.iter_mut().find(|(id, _)| *id == si) {
                    pair.1 = pair.1.max(mult);
                } else {
                    existing.sessions.push((si, mult));
                }
            } else {
                bucket.push(HeaderKey { tokens: hids, sessions: vec![(si, mult)] });
            }
        }
    }

    // Pre-pass: collect exact single-token proper names evidenced by speakers, mid-clause
    // capitalization, multi-word entity parts, or document header titles so sentence-initial
    // occurrences of the same byte-equal name bind to the same identity node.
    let mut known_proper: DetHashSet<String> = DetHashSet::default();
    for s in sessions {
        for t in &s.turns {
            let sp = t.role.trim().to_lowercase();
            if !sp.is_empty() && sp != "user" && sp != "assistant" {
                for w in raw_tokens(&sp) {
                    if w.len() >= 3 && !STARTER_STOP.contains(&w.as_str()) && !stopwords().contains(w.as_str()) {
                        known_proper.insert(w);
                    }
                }
            }
            for ent in proper_entity_tokens(&t.content) {
                if let Some(body) = ent.strip_prefix("\u{1}ent:") {
                    if !body.contains('_') && body.len() >= 3 {
                        known_proper.insert(body.to_string());
                    }
                }
            }
        }
    }

    for (si, s) in sessions.iter().enumerate() {
        let ts = if s.date.is_empty() { s.timestamp } else { match parse_date(&s.date) { 0 => s.timestamp, t => t } };
        session_ids.push(s.id.clone());
        session_ts.push(ts);
        session_ep_keys.push(session_episode_key(&s.id, &s.date));
        let date_toks = if p.enable_deictic && ts != 0 { calendar_tokens(ts) } else { Vec::new() };
        let mut members = Vec::new();
        let mut s_ents = DetHashSet::default();
        let mut s_cont = DetHashSet::default();
        let mut speaker_names: DetHashSet<String> = DetHashSet::default();
        for t in &s.turns {
            let sp = t.role.trim().to_lowercase();
            if !sp.is_empty() && sp != "user" && sp != "assistant" {
                for w in raw_tokens(&sp) {
                    speaker_names.insert(w);
                }
            }
        }
        // Collect named speakers in this session so vocative address ("Thanks, Caroline!")
        // is capped at speaker identity parity (0.72) and never outranks Caroline speaking.
        let mut sess_other_speakers: DetHashMap<usize, DetHashSet<String>> = DetHashMap::default();
        for (idx_a, ta) in s.turns.iter().enumerate() {
            let spa = ta.role.trim().to_lowercase();
            let mut others = DetHashSet::default();
            for (idx_b, tb) in s.turns.iter().enumerate() {
                if idx_a == idx_b {
                    continue;
                }
                let spb = tb.role.trim().to_lowercase();
                if !spb.is_empty() && spb != "user" && spb != "assistant" && spb != spa {
                    for sw in raw_tokens(&spb) {
                        if let Some(set) = nicks.witnessed.get(&sw) {
                            for nick in set {
                                others.insert(nick.clone());
                            }
                        }
                        others.insert(sw);
                    }
                }
            }
            sess_other_speakers.insert(idx_a, others);
        }

        for (ti, t) in s.turns.iter().enumerate() {
            let (prose_body, blip_opt) = split_blip_caption(&t.content);
            let raw = raw_tokens(prose_body);
            let nick_bound = nicks.bindings_in(prose_body);
            let role = t.role.trim();
            let speaker = role.to_lowercase();
            let vocative_partners = sess_other_speakers.get(&ti);
            let field = usize::from(p.enable_roles && role.eq_ignore_ascii_case("assistant"));
            let mut feats = Vec::new();
            let mut blip_feats = Vec::new();
            let mut seq = Vec::new();
            let mut content = DetHashSet::default();
            let mut entities = DetHashSet::default();
            let mut tech_handles: Vec<u32> = Vec::new();
            for w in raw.iter().filter(|w| is_content(w)) {
                let is_vocative_partner = vocative_partners.is_some_and(|vp| vp.contains(w.as_str()));
                let e = vocab.intern(w);
                let tok_w = if is_vocative_partner { 0.72 } else { 1.0 };
                feats.push(Feat { id: e, w: tok_w, counts_len: true, inheritable: !is_vocative_partner });
                content.insert(e);
                s_cont.insert(e);
                if let Some(full) = nick_bound.get(w.as_str()) {
                    if !raw.iter().any(|t_tok| t_tok == full) {
                        let voc = vocative_partners.is_some_and(|vp| vp.contains(full.as_str()) || vp.contains(w.as_str()));
                        let fid = vocab.intern(full);
                        feats.push(Feat {
                            id: fid,
                            w: if voc { 0.72 } else { 1.0 },
                            counts_len: false,
                            inheritable: false,
                        });
                    }
                }
                let st = stem(w);
                let sid = if p.enable_stemming { vocab.intern(&stem_term(&st)) } else { e };
                if p.enable_stemming && !is_vocative_partner {
                    feats.push(Feat { id: sid, w: 1.0, counts_len: false, inheritable: true });
                }
                seq.push((e, sid));
                if known_proper.contains(w) && !is_vocative_partner {
                    let eid = vocab.intern(&format!("\u{1}ent:{w}"));
                    entities.insert(eid);
                    s_ents.insert(eid);
                }
            }
            // Stream 2: Auxiliary BLIP caption field (w=0.75, b_blip=0, counts_len:false, inheritable:false)
            if let Some(cap) = blip_opt {
                for bw in raw_tokens(cap).into_iter().filter(|w| is_content(w)) {
                    let bid = vocab.intern(&bw);
                    blip_feats.push(Feat { id: bid, w: 1.0, counts_len: false, inheritable: false });
                    if p.enable_stemming {
                        let bst = stem(&bw);
                        let bsid = vocab.intern(&stem_term(&bst));
                        blip_feats.push(Feat { id: bsid, w: 0.75, counts_len: false, inheritable: false });
                    }
                }
                for bent in proper_entity_tokens(cap) {
                    let eid = vocab.intern(&bent);
                    blip_feats.push(Feat { id: eid, w: 0.85, counts_len: false, inheritable: false });
                    entities.insert(eid);
                    s_ents.insert(eid);
                }
            }
            // Pre-saturation BM25F title boost for structured `{Title}\n{Body}` documents.
            if let Some(core_title) = extract_header_title(prose_body) {
                for hw in raw_tokens(core_title).into_iter().filter(|w| is_content(w)) {
                    let hid = vocab.intern(&hw);
                    feats.push(Feat { id: hid, w: 1.5, counts_len: true, inheritable: false });
                    if p.enable_stemming {
                        let hst = stem(&hw);
                        let hsid = vocab.intern(&stem_term(&hst));
                        feats.push(Feat { id: hsid, w: 1.0, counts_len: false, inheritable: false });
                    }
                }
            }
            for ent in proper_entity_tokens(prose_body) {
                if let Some(body) = ent.strip_prefix("\u{1}ent:") {
                    if !body.contains('_') && vocative_partners.is_some_and(|vp| vp.contains(body)) {
                        continue;
                    }
                }
                let eid = vocab.intern(&ent);
                feats.push(Feat { id: eid, w: 1.0, counts_len: false, inheritable: true });
                entities.insert(eid);
                s_ents.insert(eid);
            }
            for raw_word in prose_body.split_whitespace() {
                let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
                if word.is_empty() {
                    continue;
                }
                for (part, aux_w) in surface_aux_tokens(word) {
                    if raw.iter().any(|t_tok| t_tok == &part) {
                        continue;
                    }
                    let id = vocab.intern(&part);
                    feats.push(Feat { id, w: aux_w, counts_len: false, inheritable: false });
                }
            }
            let slots = typed_slot_tokens(prose_body);
            // Preference and constraint flags follow recorded structure: a typed
            // `key: value` slot. Phrase markers no longer set them.
            let is_personal_constraint = field == 0 && !slots.is_empty();
            if p.enable_deictic {
                for (d, w) in deictic_weighted_tokens(&raw, ts) {
                    feats.push(Feat { id: vocab.intern(&d), w, counts_len: false, inheritable: false });
                }
                for d in inline_date_tokens(prose_body) {
                    feats.push(Feat { id: vocab.intern(&d), w: 0.9, counts_len: false, inheritable: false });
                }
                for d in &date_toks {
                    feats.push(Feat { id: vocab.intern(d), w: p.date_weight, counts_len: false, inheritable: false });
                }
            }
            if !speaker.is_empty() && speaker != "user" && speaker != "assistant" {
                for w in raw_tokens(&speaker) {
                    let id = vocab.intern(&w);
                    // Speaker identity edge: own speech carries full parity (1.05) over vocative address.
                    feats.push(Feat { id, w: 1.05, counts_len: false, inheritable: false });
                    let eid = vocab.intern(&format!("\u{1}ent:{w}"));
                    entities.insert(eid);
                    s_ents.insert(eid);
                }
            }
            for (slot_key, slot_val) in slots {
                let slot_ent = vocab.intern(&format!("\u{1}slot:{slot_key}"));
                let kid = vocab.intern(&slot_key);
                let vid = vocab.intern(&slot_val);
                feats.push(Feat { id: kid, w: 0.5, counts_len: false, inheritable: false });
                feats.push(Feat { id: vid, w: 1.6, counts_len: false, inheritable: true });
                feats.push(Feat { id: slot_ent, w: 0.8, counts_len: false, inheritable: false });
                content.insert(vid);
                s_cont.insert(vid);
                constraint_terms.insert(vid);
                entities.insert(slot_ent);
                s_ents.insert(slot_ent);
                tech_handles.push(slot_ent);
            }
            for h_str in extract_exact_technical_handles(prose_body) {
                let hid = vocab.intern(&h_str);
                feats.push(Feat { id: hid, w: 1.25, counts_len: false, inheritable: true });
                entities.insert(hid);
                s_ents.insert(hid);
                tech_handles.push(hid);
            }
            let (is_question, is_short_question) = detect_question(prose_body, raw.len());
            let qshape = analyze_question(&mut vocab, prose_body, is_question, p.enable_stemming, &speaker_names);
            let is_pure_info_question = qshape.is_pure;
            let anaphoric = raw.iter().take(12).any(|w| ANAPHORS.contains(&w.as_str()));
            let has_temporal = raw.iter().any(|w| {
                token_is_temporal_word(w.as_str())
                    || matches!(w.as_str(), "am" | "pm")
                    || (w.len() >= 3 && (w.ends_with("st") || w.ends_with("nd") || w.ends_with("rd") || w.ends_with("th")) && w[..w.len()-2].chars().all(|c| c.is_ascii_digit()))
            });
            let has_numeric = prose_body.chars().any(|c| c.is_ascii_digit());
            let has_spatial = !entities.is_empty() && raw.iter().any(|w| closed_locative(w));
            let has_causal = !t.links.is_empty();
            let is_standing_policy = field == 0 && utterance_is_standing_policy(prose_body);
            // A short turn with no date, digit, or proper name is a backchannel
            // by shape. No discourse-word list.
            let is_backchannel = !is_question
                && content.len() <= 2
                && !has_temporal
                && !has_numeric
                && entities.is_empty()
                && !is_standing_policy;
            let mut role_objs: Vec<u32> = Vec::new();
            if field == 0 && !is_backchannel {
                let (r_objs, _r_attrs) = step14_extract_role_frames(prose_body);
                for obj in r_objs {
                    let pid = vocab.intern(&obj);
                    if !role_objs.contains(&pid) {
                        role_objs.push(pid);
                    }
                    if p.enable_stemming {
                        let st = stem(&obj);
                        let sid = vocab.intern(&stem_term(&st));
                        if !role_objs.contains(&sid) {
                            role_objs.push(sid);
                        }
                    }
                }
            }
            let is_initiative = field == 0
                && !is_standing_policy
                && !anaphoric
                && !is_short_question
                && content.len() >= 4;
            let turn_idx = turns.len();
            if is_standing_policy {
                let (has_explicit_scope, scope_words) = extract_rule_scope(prose_body);
                raw_standing_rules.push((
                    standing_policy_id(prose_body),
                    turn_idx,
                    si,
                    ts,
                    has_explicit_scope,
                    scope_words,
                ));
            }
            let id = if t.id.is_empty() { format!("{}#{}", s.id, ti) } else { t.id.clone() };
            let body_for_headers = if header_max_words > 0 {
                let body = if extract_header_titles(prose_body).is_some() {
                    prose_body.split_once('\n').map(|(_, rest)| rest).unwrap_or(prose_body)
                } else {
                    prose_body
                };
                body.to_string()
            } else {
                String::new()
            };
            turn_tech_handles.push(tech_handles);
            turn_prose_cache.push(prose_body.to_string());
            turn_cues.push(intern_cues(&t.content, &mut cue_vocab));
            role_cues.extend(intern_cues(&t.role, &mut cue_vocab));
            members.push(turn_idx);
            turns.push(TurnIx {
                session: si,
                id,
                field,
                feats,
                blip_feats,
                seq,
                content,
                entities,
                body_for_headers,
                span_ids_cache: OnceLock::new(),
                speaker,
                is_question,
                is_pure_info_question,
                pre_question_ids: qshape.pre_ids,
                question_clause_ids: qshape.clause_ids,
                answers_question: false,
                is_short_reply: false,
                is_backchannel,
                anaphoric,
                is_personal_constraint,
                has_temporal,
                has_numeric,
                has_spatial,
                has_causal,
                is_short_question,
                is_standing_policy,
                is_initiative,
                answers: None,
                exact_edges: Vec::new(),
                policy_backlinks: Vec::new(),
                supersedes_turns: Vec::new(),
                stability: t.stability,
                form_clock: has_iso_or_hhmm(prose_body),
                form_digit: has_ascii_digit(prose_body),
                form_handle: has_handle_or_path(prose_body),
                form_midcap: has_mid_sentence_name(prose_body),
                slot_entities: count_slot_entities_mode(prose_body, false),
                slot_entities_clean: count_slot_entities_mode(prose_body, true),
                list_item_count: step9_list_item_count(prose_body),
                first_move_num: step9_first_move_num(prose_body),
                role_objs,
                raw_links: t.links.clone(),
                body: if replay_dump_path().is_some() {
                    prose_body.to_string()
                } else {
                    String::new()
                },
            });
        }
        session_turns.push(members);
        session_entities.push(s_ents);
        session_content.push(s_cont);
    }
    // Mark DAMSL adjacency pairs before bag construction: question/answer pairs AND
    // short value/commitment replies to preceding statements ("Done.", "Last Tuesday.", "Me too!").
    for members in &session_turns {
        for pos in 1..members.len() {
            let (a, b) = (members[pos - 1], members[pos]);
            if turns[a].speaker != turns[b].speaker {
                turns[b].answers = Some(a);
                if turns[a].is_question {
                    turns[b].answers_question = true;
                }
                if !turns[b].is_question
                    && !turns[b].is_backchannel
                    && turns[b].content.len() <= 5
                    && (turns[a].is_question
                        || turns[b].anaphoric
                        || turns[b].content.len() <= 2
                        || turns[b].has_temporal
                        || turns[b].has_numeric)
                {
                    turns[b].is_short_reply = true;
                }
            }
        }
    }

    let mut id_at: DetHashMap<String, usize> = DetHashMap::default();
    for (i, t) in turns.iter().enumerate() {
        if !t.id.is_empty() {
            id_at.entry(t.id.clone()).or_insert(i);
        }
    }
    for ti in 0..turns.len() {
        let pending = std::mem::take(&mut turns[ti].raw_links);
        for link in pending {
            let Some(&target) = id_at.get(&link.to) else { continue };
            if target == ti {
                continue;
            }
            turns[ti].exact_edges.push((link.degree.max(2), target));
        }
    }

    // ---- Ingest-Time Retroactive Salience Backfill (RSB), Contradiction Supersession,
    //      Synaptic Tagging 9h CaptureWindow, and Temporal Context Episode Contiguity ----
    // Strictly BACKWARD-ONLY (`later -> earlier`). Zero forward edges, zero word similarity.
    let ns_count = session_turns.len();
    let mut session_rsb_back: Vec<Vec<(u32, usize)>> = vec![Vec::new(); ns_count];
    let mut session_supersedes_back: Vec<Vec<usize>> = vec![Vec::new(); ns_count];
    let mut session_episode_prev: Vec<Option<usize>> = vec![None; ns_count];
    for si in 1..ns_count {
        let same_ep = !session_ep_keys[si].is_empty() && session_ep_keys[si] == session_ep_keys[si - 1];
        let same_nonzero_ts = session_ts[si] > 0 && session_ts[si] == session_ts[si - 1];
        if same_ep || same_nonzero_ts {
            session_episode_prev[si] = Some(si - 1);
        }
    }

    // 1. Exact Technical Handle & Typed Slot RSB (`df <= 6`): later turn/session -> earlier turn/session.
    let mut handle_postings: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (ti, hs) in turn_tech_handles.iter().enumerate() {
        for &h in hs {
            let list = handle_postings.entry(h).or_default();
            if list.last().copied() != Some(ti) {
                list.push(ti);
            }
        }
    }
    for (_, list) in handle_postings {
        let deg = list.len() as u32;
        if !(2..=6).contains(&deg) {
            continue;
        }
        for idx in 1..list.len() {
            let later = list[idx];
            let earlier = list[idx - 1];
            if !turns[later].exact_edges.iter().any(|&(_, t)| t == earlier) {
                turns[later].exact_edges.push((deg, earlier));
            }
            let s_lat = turns[later].session;
            let s_ear = turns[earlier].session;
            if s_lat > s_ear && !session_rsb_back[s_lat].iter().any(|&(_, s)| s == s_ear) {
                session_rsb_back[s_lat].push((deg, s_ear));
            }
        }
    }

    // 2. State-Update & Structural Contradiction Supersession Chain (`appears_contradictory`):
    //    When two user turns share a rare proper entity (`2 <= df <= 8`) or constraint object
    //    and either `advanced::contradiction::appears_contradictory` fires or the later turn
    //    asserts an explicit revision/constraint, record a backward `supersedes` edge (`later -> earlier`).
    let mut ent_user_postings: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (ti, t) in turns.iter().enumerate() {
        if t.field != 0 || t.is_question || t.is_backchannel {
            continue;
        }
        for &eid in &t.entities {
            ent_user_postings.entry(eid).or_default().push(ti);
        }
        for &cid in &t.content {
            if constraint_terms.contains(&cid) {
                ent_user_postings.entry(cid).or_default().push(ti);
            }
        }
    }
    for (_, mut list) in ent_user_postings {
        list.dedup();
        if !(2..=8).contains(&list.len()) {
            continue;
        }
        for idx in 1..list.len() {
            let later = list[idx];
            let prev = list[idx - 1];
            let s_lat = turns[later].session;
            let s_prev = turns[prev].session;
            if s_lat <= s_prev {
                continue;
            }
            let lat_prose = &turn_prose_cache[later];
            let prev_prose = &turn_prose_cache[prev];
            let later_digits = digit_token_set(lat_prose);
            let prev_digits = digit_token_set(prev_prose);
            let digit_shift = !later_digits.is_empty() && !prev_digits.is_empty() && later_digits != prev_digits;
            let has_rev_cue = turns[later].is_personal_constraint || digit_shift;
            let contradicts = crate::advanced::contradiction::appears_contradictory(
                prev_prose,
                lat_prose,
                crate::advanced::contradiction::SubjectIdentity::AlreadyEstablished,
            );
            if contradicts || has_rev_cue {
                if !turns[later].supersedes_turns.contains(&prev) {
                    turns[later].supersedes_turns.push(prev);
                }
                if !session_supersedes_back[s_lat].contains(&s_prev) {
                    session_supersedes_back[s_lat].push(s_prev);
                }
            }
        }
    }

    // 3. Synaptic Tagging `CaptureWindow` (Frey & Morris 1997; Redondo & Morris 2011; 9-hour window):
    //    A high-salience turn (`is_standing_policy`, `is_personal_constraint`, or `FAILURE_MARKERS`)
    //    captures the immediately preceding tagged user initiative turn within `CaptureWindow::backward_hours` (9h)
    //    or the immediate preceding same-episode sub-session (`session_episode_prev`).
    let capture_window = crate::neuroscience::CaptureWindow::default();
    let max_back_secs = (capture_window.backward_hours * 3600.0) as i64;
    for ti in 1..turns.len() {
        let is_prp = turns[ti].is_standing_policy || turns[ti].is_personal_constraint;
        if !is_prp {
            continue;
        }
        let s_cur = turns[ti].session;
        let ts_cur = session_ts[s_cur];
        // Search backward for the closest preceding user initiative turn inside the 9h capture window (or same episode).
        for back_s in (0..s_cur).rev() {
            let ts_prev = session_ts[back_s];
            let in_clock_window = ts_cur > 0 && ts_prev > 0 && ts_cur >= ts_prev && (ts_cur - ts_prev) <= max_back_secs;
            let is_ep_prev = session_episode_prev[s_cur] == Some(back_s);
            if !in_clock_window && !is_ep_prev {
                if ts_cur > 0 && ts_prev > 0 && (ts_cur - ts_prev) > max_back_secs {
                    break;
                }
                continue;
            }
            if let Some(&init_ti) = session_turns[back_s].iter().rev().find(|&&cand| turns[cand].is_initiative) {
                // Require at least 1 shared entity or schema domain so unrelated adjacent topics don't capture blindly.
                let shares_anchor = turns[ti].entities.iter().any(|e| turns[init_ti].entities.contains(e));
                if shares_anchor {
                    if !turns[ti].exact_edges.iter().any(|&(_, t)| t == init_ti) {
                        turns[ti].exact_edges.push((3, init_ti));
                    }
                    if !session_rsb_back[s_cur].iter().any(|&(_, s)| s == back_s) {
                        session_rsb_back[s_cur].push((3, back_s));
                    }
                }
                break;
            }
        }
    }

    // 4. SalienceJoin. A later turn records a backward link to an earlier
    //    standing policy in the same session, or in an earlier session that shares
    //    the episode key. The link is session or episode identity, not a word match.
    //    It is not a general exact edge and it does not write session_rsb_back.
    //    v1's global winner seat raised a rule from a different batch. v3 does not
    //    require the policy to match the query. Rank time seats it behind the
    //    retrieved turn that points at it.
    for si in 0..ns_count {
        let members = &session_turns[si];
        for (pos, &p_ti) in members.iter().enumerate() {
            if !turns[p_ti].is_standing_policy {
                continue;
            }
            for &t_ti in &members[pos + 1..] {
                if !turns[t_ti].policy_backlinks.contains(&p_ti) {
                    turns[t_ti].policy_backlinks.push(p_ti);
                }
            }
        }
    }
    for si in 0..ns_count {
        if session_ep_keys[si].is_empty() {
            continue;
        }
        for sj in 0..si {
            if session_ep_keys[sj] != session_ep_keys[si] {
                continue;
            }
            for &p_ti in &session_turns[sj] {
                if !turns[p_ti].is_standing_policy {
                    continue;
                }
                for &t_ti in &session_turns[si] {
                    if !turns[t_ti].policy_backlinks.contains(&p_ti) {
                        turns[t_ti].policy_backlinks.push(p_ti);
                    }
                }
            }
        }
    }

    // Session bags (own text + field-2 BLIP), uncarried turn bags (`turn_own_bags`),
    // and carried turn bags (`turn_bags` — never carrying into questions!).
    let mut session_bags = Vec::with_capacity(session_turns.len());
    let mut turn_own_bags: Vec<Bag> = (0..turns.len()).map(|_| Bag::default()).collect();
    let mut turn_bags: Vec<Bag> = (0..turns.len()).map(|_| Bag::default()).collect();
    for members in &session_turns {
        let mut sb = Bag::default();
        for (pos, &ti) in members.iter().enumerate() {
            let t = &turns[ti];
            for &f in &t.feats {
                sb.add(t.field, f, 1.0);
                turn_own_bags[ti].add(t.field, f, 1.0);
                turn_bags[ti].add(t.field, f, 1.0);
            }
            for &bf in &t.blip_feats {
                sb.add(2, bf, 1.0);
                turn_own_bags[ti].add(2, bf, 1.0);
                turn_bags[ti].add(2, bf, 1.0);
            }
            // In named-speaker dialogues (`LoCoMo`), a follow-up question does NOT state the
            // previous speaker's facts; in `user`/`assistant` dialogues, user follow-up questions
            // still inherit conversational context (`pos - k`).
            let is_named_speaker_q = t.is_question
                && !t.speaker.is_empty()
                && t.speaker != "user"
                && t.speaker != "assistant";
            if p.enable_carry && pos > 0 && !is_named_speaker_q {
                let answer = t.answers_question;
                let factor = if t.is_backchannel {
                    0.15
                } else if answer || t.anaphoric {
                    0.85
                } else {
                    p.carry_plain_factor
                };
                let field = t.field;
                let mut inherited: Vec<(Feat, f32)> = Vec::new();
                for k in 1..=3usize {
                    if pos < k {
                        break;
                    }
                    let src = &turns[members[pos - k]];
                    let w = if k == 1 && answer {
                        p.carry_answer
                    } else if k == 1 && t.is_short_reply {
                        0.75
                    } else {
                        p.carry[k - 1] * factor
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    // Inherit open-class content (exact + stem + hypernym + proper entities), never speaker tag or dates.
                    for &f in src.feats.iter().filter(|f| f.inheritable) {
                        inherited.push((f, w));
                    }
                }
                for (f, w) in inherited {
                    turn_bags[ti].add_inherited(field, f, w);
                }
            }
            // Third-position uptake: a concise value-bearing turn inherits light context from pos + 1.
            if p.enable_carry && !t.is_backchannel && !t.is_question && t.content.len() <= 4 && pos + 1 < members.len() {
                let nxt = &turns[members[pos + 1]];
                for &f in nxt.feats.iter().filter(|f| f.inheritable) {
                    turn_bags[ti].add_inherited(t.field, f, 0.25);
                }
            }
        }
        session_bags.push(sb);
    }

    // Max-passage windows.
    let w = p.window.max(1);
    let mut windows = Vec::new();
    let mut window_bags = Vec::new();
    for (si, members) in session_turns.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let last_start = members.len().saturating_sub(w);
        for start in 0..=last_start {
            let end = (start + w).min(members.len()) - 1;
            let mut b = Bag::default();
            for &ti in &members[start..=end] {
                for &f in &turns[ti].feats {
                    b.add(turns[ti].field, f, 1.0);
                }
                for &bf in &turns[ti].blip_feats {
                    b.add(2, bf, 1.0);
                }
            }
            windows.push((si, start, end));
            window_bags.push(b);
        }
    }

    let s_stats = stats_of(&session_bags);
    let t_own_stats = stats_of(&turn_own_bags);
    let t_stats = stats_of(&turn_bags);
    let w_stats = stats_of(&window_bags);

    // Precompute rarest non-hub tokens per session (`df >= 2 && df <= 0.20 * N`, up to 32 sorted)
    // for GhostLink Ochiai divergence (`1 - |Nt(a) & Nt(b)| / sqrt(|Nt(a)| * |Nt(b)|)`).
    let ns_total = session_bags.len();
    let max_rare_nt_df = ((s_stats.n * 0.20).max(4.0)) as u32;
    let session_rare_nt: Vec<Vec<u32>> = session_content
        .iter()
        .map(|sc| {
            let mut v: Vec<(u32, u32)> = sc
                .iter()
                .filter_map(|&id| {
                    let df = s_stats.df.get(&id).copied().unwrap_or(0);
                    (df >= 2 && df <= max_rare_nt_df).then_some((df, id))
                })
                .collect();
            v.sort_unstable();
            let mut ids: Vec<u32> = v.into_iter().take(32).map(|(_, id)| id).collect();
            ids.sort_unstable();
            ids
        })
        .collect();

    let fan_cut = if ns_total >= 16 {
        ((s_stats.n * 0.12).max(4.0)) as u32
    } else {
        u32::MAX
    };
    let mut standing_rules = Vec::with_capacity(raw_standing_rules.len());
    for (policy_key, turn_idx, sess_idx, ts, has_explicit_scope, scope_words) in raw_standing_rules {
        let mut scope = DetHashSet::default();
        for w in &scope_words {
            if let Some(id) = vocab.get(w.as_str()) {
                let df = t_own_stats.df.get(&id).copied().unwrap_or(0);
                if corpus_frame(t_own_stats.n as usize, df) {
                    continue;
                }
            }
            if let Some(id) = vocab.get(w) {
                if s_stats.df.get(&id).copied().unwrap_or(0) <= fan_cut {
                    scope.insert(id);
                }
            }
            let st = stem(w);
            if let Some(sid) = vocab.get(&stem_term(&st)) {
                if s_stats.df.get(&sid).copied().unwrap_or(0) <= fan_cut {
                    scope.insert(sid);
                }
            }
        }
        let mut assistant_nouns = DetHashSet::default();
        let members = &session_turns[sess_idx];
        if let Some(pos) = members.iter().position(|&ti| ti == turn_idx) {
            if pos + 1 < members.len() {
                let adj = &turns[members[pos + 1]];
                if adj.field == 1 {
                    for &cid in &adj.content {
                        if s_stats.df.get(&cid).copied().unwrap_or(0) <= fan_cut {
                            assistant_nouns.insert(cid);
                        }
                    }
                }
            }
        }
        standing_rules.push(StandingRule {
            policy_key,
            turn: turn_idx,
            session: sess_idx,
            ts,
            has_explicit_scope,
            scope,
            assistant_nouns,
        });
    }

    let vocab_len = vocab.names.len();
    let s_postings = build_postings(&session_bags, vocab_len);
    let w_postings = build_postings(&window_bags, vocab_len);
    let (t_postings, turn_bloom) = build_turn_postings_and_bloom(&turn_bags, &turn_own_bags, vocab_len);
    let session_bloom = build_session_bloom(&session_bags);
    let turn_meta = build_turn_meta(&turns, &turn_bloom);

    let mut turn_superseded = vec![false; turns.len()];
    for t in &turns {
        for &old in &t.supersedes_turns {
            if old < turn_superseded.len() {
                turn_superseded[old] = true;
            }
        }
    }

    // 1-hop ingest WordNet bridge postings in private `bridge_vocab` (`ix.vocab` untouched!).
    // Memoize WordNet + stem expansion once per unique rare `cid` (0-diff exact parity, ~5x faster).
    let n_turns_u32 = (t_own_stats.n as u32).max(1);
    let max_rare_df = (n_turns_u32 / 8).max(4);
    let mut bridge_vocab = Vocab::default();
    let mut turn_bridge_sets: Vec<DetHashSet<u32>> = vec![DetHashSet::default(); turns.len()];
    if p.enable_hypernyms {
        let mut cid_to_bridge: DetHashMap<u32, Vec<u32>> = DetHashMap::default();
        for (&cid, &df) in &t_own_stats.df {
            if df == 0 || df > max_rare_df {
                continue;
            }
            let word = &vocab.names[cid as usize];
            let mut bids = Vec::new();
            for lemma in wordnet_lemmas_with_morph(word) {
                let lid = bridge_vocab.intern(lemma);
                if !bids.contains(&lid) {
                    bids.push(lid);
                }
                if p.enable_stemming {
                    let sid = bridge_vocab.intern(&stem_term(&stem(lemma)));
                    if !bids.contains(&sid) {
                        bids.push(sid);
                    }
                }
            }
            // Hovy 3-hop upward-only `@` hypernyms (zero downward explosion)
            for lemma in crate::wordnet::upward_hypernyms(word, 3) {
                let lid = bridge_vocab.intern(lemma);
                if !bids.contains(&lid) {
                    bids.push(lid);
                }
                if p.enable_stemming {
                    let sid = bridge_vocab.intern(&stem_term(&stem(lemma)));
                    if !bids.contains(&sid) {
                        bids.push(sid);
                    }
                }
            }
            if !bids.is_empty() {
                cid_to_bridge.insert(cid, bids);
            }
        }
        for ti in 0..turns.len() {
            if turns[ti].is_backchannel {
                continue;
            }
            for &cid in &turns[ti].content {
                if let Some(bids) = cid_to_bridge.get(&cid) {
                    turn_bridge_sets[ti].extend(bids.iter().copied());
                }
            }
        }
    }
    let mut turn_bridge_postings = vec![Vec::new(); bridge_vocab.names.len()];
    for (ti, bset) in turn_bridge_sets.iter().enumerate() {
        for &bid in bset {
            if let Some(list) = turn_bridge_postings.get_mut(bid as usize) {
                list.push(ti as u32);
            }
        }
    }

    // Haystack Positive PMI associations over rare content tokens (`2 <= df <= max_rare_df`).
    let mut pair_counts: DetHashMap<(u32, u32), u32> = DetHashMap::default();
    for ti in 0..turns.len() {
        let mut rare_toks: Vec<u32> = turns[ti]
            .content
            .iter()
            .copied()
            .filter(|id| {
                let df = t_own_stats.df.get(id).copied().unwrap_or(0);
                df >= 2 && df <= max_rare_df
            })
            .collect();
        if let Some(ans_prev) = turns[ti].answers {
            for &id in &turns[ans_prev].content {
                let df = t_own_stats.df.get(&id).copied().unwrap_or(0);
                if df >= 2 && df <= max_rare_df {
                    rare_toks.push(id);
                }
            }
        }
        rare_toks.sort_unstable();
        rare_toks.dedup();
        if rare_toks.len() <= 24 {
            for i in 0..rare_toks.len() {
                for j in (i + 1)..rare_toks.len() {
                    *pair_counts.entry((rare_toks[i], rare_toks[j])).or_insert(0) += 1;
                }
            }
        }
    }
    let mut ppmi_neighbors: DetHashMap<u32, Vec<(u32, f32)>> = DetHashMap::default();
    let n_f = t_own_stats.n.max(1.0);
    for ((u, v), c_uv) in pair_counts {
        if c_uv < 2 {
            continue;
        }
        let df_u = t_own_stats.df.get(&u).copied().unwrap_or(1) as f32;
        let df_v = t_own_stats.df.get(&v).copied().unwrap_or(1) as f32;
        let ppmi = ((c_uv as f32 * n_f) / (df_u * df_v)).log2();
        if ppmi >= 2.5 {
            let w = (ppmi / 10.0).clamp(0.18, 0.45);
            ppmi_neighbors.entry(u).or_default().push((v, w));
            ppmi_neighbors.entry(v).or_default().push((u, w));
        }
    }
    for list in ppmi_neighbors.values_mut() {
        list.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        list.truncate(6);
    }

    Index {
        vocab,
        session_ids,
        session_ts,
        session_turns,
        session_entities,
        session_content,
        session_rare_nt,
        session_rsb_back,
        session_supersedes_back,
        session_episode_prev,
        turns,
        session_bags,
        turn_own_bags,
        turn_bags,
        windows,
        window_bags,
        s_stats,
        t_own_stats,
        t_stats,
        w_stats,
        constraint_terms,
        standing_rules,
        header_dict,
        header_max_words,
        nicks,
        turn_cues,
        cue_vocab,
        role_cues,
        s_postings,
        w_postings,
        t_postings,
        bridge_vocab,
        turn_bridge_postings,
        ppmi_neighbors,
        turn_superseded,
        session_bloom,
        turn_bloom,
        turn_meta,
    }
}
