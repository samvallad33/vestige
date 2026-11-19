//! Stage 2: Index structures (`Vocab`, `Bag`, `TurnIx`, `Index`) and `build_index`.

use crate::retrieval::*;

pub(in crate::retrieval) mod normalize;
pub(in crate::retrieval) mod policy;
pub(in crate::retrieval) mod postings;
pub(in crate::retrieval) mod rsb;

pub(in crate::retrieval) use self::normalize::*;
pub use self::policy::*;
pub(in crate::retrieval) use self::postings::*;
pub(in crate::retrieval) use self::rsb::*;

// =========================================================================
// INDEX
// =========================================================================

#[derive(Default)]
pub(in crate::retrieval) struct Vocab {
    pub(in crate::retrieval) map: DetHashMap<String, u32>,
    pub(in crate::retrieval) names: Vec<String>,
}

impl Vocab {
    pub(in crate::retrieval) fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.map.insert(s.to_string(), id);
        self.names.push(s.to_string());
        id
    }
    pub(in crate::retrieval) fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
}

#[derive(Clone, Copy)]
pub(in crate::retrieval) struct Feat {
    pub(in crate::retrieval) id: u32,
    pub(in crate::retrieval) w: f32,
    pub(in crate::retrieval) counts_len: bool,
    pub(in crate::retrieval) inheritable: bool,
}

#[derive(Default, Clone)]
pub(in crate::retrieval) struct Bag {
    pub(in crate::retrieval) tf: [DetHashMap<u32, f32>; 3],
    pub(in crate::retrieval) len: [f32; 3],
}

impl Bag {
    pub(in crate::retrieval) fn add(&mut self, field: usize, f: Feat, scale: f32) {
        *self.tf[field].entry(f.id).or_insert(0.0) += f.w * scale;
        if f.counts_len {
            self.len[field] += f.w * scale;
        }
    }

    pub(in crate::retrieval) fn add_inherited(&mut self, field: usize, f: Feat, scale: f32) {
        *self.tf[field].entry(f.id).or_insert(0.0) += f.w * scale;
        if f.counts_len {
            self.len[field] += 0.50 * f.w * scale;
        }
    }
}

pub(in crate::retrieval) struct StandingRule {
    pub(in crate::retrieval) policy_key: String,
    pub(in crate::retrieval) turn: usize,
    pub(in crate::retrieval) session: usize,
    pub(in crate::retrieval) ts: i64,
    pub(in crate::retrieval) has_explicit_scope: bool,
    pub(in crate::retrieval) scope: DetHashSet<u32>,
    pub(in crate::retrieval) assistant_nouns: DetHashSet<u32>,
    /// Exact + stem ids of the rule's own condition clause (empty = unconditional).
    pub(in crate::retrieval) condition: DetHashSet<u32>,
    /// Surface words of the condition clause (for WordNet lemma matching).
    pub(in crate::retrieval) condition_words: Vec<String>,
}

pub(in crate::retrieval) struct TurnIx {
    pub(in crate::retrieval) session: usize,
    pub(in crate::retrieval) id: String,
    pub(in crate::retrieval) field: usize,
    pub(in crate::retrieval) feats: Vec<Feat>,
    pub(in crate::retrieval) blip_feats: Vec<Feat>,
    /// Exact/stem term ids of content tokens in order, for span proximity.
    pub(in crate::retrieval) seq: Vec<(u32, u32)>,
    pub(in crate::retrieval) content: DetHashSet<u32>,
    /// Capitalized proper-entity term IDs inside this turn.
    pub(in crate::retrieval) entities: DetHashSet<u32>,
    /// Body text (header line removed) for lazy exact title-span linking on hot anchors.
    pub(in crate::retrieval) body_for_headers: String,
    pub(in crate::retrieval) span_ids_cache: OnceLock<Vec<u32>>,
    pub(in crate::retrieval) speaker: String,
    pub(in crate::retrieval) is_question: bool,
    pub(in crate::retrieval) is_pure_info_question: bool,
    pub(in crate::retrieval) pre_question_ids: DetHashSet<u32>,
    pub(in crate::retrieval) question_clause_ids: DetHashSet<u32>,
    pub(in crate::retrieval) answers_question: bool,
    pub(in crate::retrieval) is_short_reply: bool,
    pub(in crate::retrieval) is_backchannel: bool,
    pub(in crate::retrieval) anaphoric: bool,
    #[allow(dead_code)]
    pub(in crate::retrieval) is_personal_constraint: bool,
    pub(in crate::retrieval) has_temporal: bool,
    pub(in crate::retrieval) has_numeric: bool,
    pub(in crate::retrieval) has_spatial: bool,
    pub(in crate::retrieval) has_causal: bool,
    pub(in crate::retrieval) is_short_question: bool,
    pub(in crate::retrieval) is_standing_policy: bool,
    /// User turn that opens a concrete event, not a rule and not a backchannel.
    pub(in crate::retrieval) is_initiative: bool,
    /// Previous turn in this session with a different role. Position and role only.
    /// Backward only. A later turn points at the earlier one. There is no forward edge.
    pub(in crate::retrieval) answers: Option<usize>,
    /// Caller & ingest-time RSB edges resolved to `(degree, target)` — strictly backward.
    pub(in crate::retrieval) exact_edges: Vec<(u32, usize)>,
    /// SalienceJoin v2: earlier standing-policy turns this later turn points at.
    /// Promoted only when that policy is a standing rule that fired for the query.
    pub(in crate::retrieval) policy_backlinks: Vec<usize>,
    /// Backward state-revision / contradiction edges (`t_new -> t_old`).
    pub(in crate::retrieval) supersedes_turns: Vec<usize>,
    pub(in crate::retrieval) stability: f32,
    /// ISO date or HH:MM in the turn text. Character form, not a month name.
    pub(in crate::retrieval) form_clock: bool,
    /// An ASCII digit in the turn text. Number words do not count.
    pub(in crate::retrieval) form_digit: bool,
    /// An @handle, a URL, or a path token.
    pub(in crate::retrieval) form_handle: bool,
    /// A capitalized word of two or more letters that is not sentence-initial.
    pub(in crate::retrieval) form_midcap: bool,
    /// Ingest-time count of mid-sentence capitalized tokens, digits, and quotes (Rung 1 `--no-dump` safe).
    pub(in crate::retrieval) slot_entities: u16,
    /// Clean mid-sentence slot entities excluding first-person pronouns ("I", "I'm", "I've", "I'll", "I'd").
    pub(in crate::retrieval) slot_entities_clean: u16,
    /// Step 9 ingest-time prose shape (`--no-dump` safe): list-marker lines and first game move.
    pub(in crate::retrieval) list_item_count: u8,
    pub(in crate::retrieval) first_move_num: u16,
    /// Step 14 ingest-time closed-class user preference/constraint role-object ids (`prose_body` only).
    pub(in crate::retrieval) role_objs: Vec<u32>,
    pub(in crate::retrieval) raw_links: Vec<TurnLink>,
    /// Dump-only. Full prose body, filled only when VESTIGE_REPLAY_DUMP is set.
    /// Never read by ranking.
    pub(in crate::retrieval) body: String,
}

pub(in crate::retrieval) struct Stats {
    pub(in crate::retrieval) n: f32,
    pub(in crate::retrieval) df: DetHashMap<u32, u32>,
    pub(in crate::retrieval) avg: [f32; 3],
}

pub(in crate::retrieval) fn stats_of(bags: &[Bag]) -> Stats {
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

pub(in crate::retrieval) struct HeaderKey {
    pub(in crate::retrieval) tokens: Vec<u32>,
    pub(in crate::retrieval) sessions: Vec<(usize, f32)>,
}

pub(in crate::retrieval) struct Index {
    pub(in crate::retrieval) vocab: Vocab,
    pub(in crate::retrieval) session_ids: Vec<String>,
    pub(in crate::retrieval) session_ts: Vec<i64>,
    pub(in crate::retrieval) session_turns: Vec<Vec<usize>>,
    pub(in crate::retrieval) session_entities: Vec<DetHashSet<u32>>,
    pub(in crate::retrieval) session_content: Vec<DetHashSet<u32>>,
    /// Rarest non-hub content token IDs per session (sorted, up to 32) for GhostLink Ochiai divergence.
    pub(in crate::retrieval) session_rare_nt: Vec<Vec<u32>>,
    /// Backward-only session RSB edges (`later_session -> (degree, earlier_session)`).
    pub(in crate::retrieval) session_rsb_back: Vec<Vec<(u32, usize)>>,
    /// Backward-only session contradiction/supersession edges (`s_new -> s_old`).
    pub(in crate::retrieval) session_supersedes_back: Vec<Vec<usize>>,
    /// Backward-only Temporal Context Episode neighbor (`si -> si - 1` when sharing episode/date anchor).
    pub(in crate::retrieval) session_episode_prev: Vec<Option<usize>>,
    pub(in crate::retrieval) turns: Vec<TurnIx>,
    pub(in crate::retrieval) session_bags: Vec<Bag>,
    pub(in crate::retrieval) turn_own_bags: Vec<Bag>,
    pub(in crate::retrieval) turn_bags: Vec<Bag>,
    /// (session index, first turn position within session, last turn position inclusive)
    pub(in crate::retrieval) windows: Vec<(usize, usize, usize)>,
    pub(in crate::retrieval) window_bags: Vec<Bag>,
    pub(in crate::retrieval) s_stats: Stats,
    pub(in crate::retrieval) t_own_stats: Stats,
    pub(in crate::retrieval) t_stats: Stats,
    pub(in crate::retrieval) w_stats: Stats,
    /// Content-term ids that are the object of a personal constraint ("allergic to X").
    pub(in crate::retrieval) constraint_terms: DetHashSet<u32>,
    /// Compiled SOAR/Rete standing rules for the stratified governing channel (`RankResult.standing_rules`).
    pub(in crate::retrieval) standing_rules: Vec<StandingRule>,
    /// Exact `{Header}\\n{Body}` titles, hashed, for the session bridge.
    pub(in crate::retrieval) header_dict: DetHashMap<u64, Vec<HeaderKey>>,
    pub(in crate::retrieval) header_max_words: usize,
    /// Unique title-case speaker nickname bindings (`Mel` -> `melanie`).
    pub(in crate::retrieval) nicks: SpeakerNickIndex,
    /// Recall cues: per turn, the sorted ids of the kernel tokens (`raw_tokens` + `stem`) of its
    /// full content, in a private id space (`cue_vocab`) so the ranker vocabulary is untouched.
    pub(in crate::retrieval) turn_cues: Vec<Vec<u32>>,
    pub(in crate::retrieval) cue_vocab: DetHashMap<String, u32>,
    /// Cue ids that occur in any turn's role / speaker field (structural metadata, not a list).
    pub(in crate::retrieval) role_cues: DetHashSet<u32>,
    /// Sedgewick dense `u32` TokenId inverted postings (sessions, windows, turns).
    pub(in crate::retrieval) s_postings: Vec<Vec<u32>>,
    pub(in crate::retrieval) w_postings: Vec<Vec<u32>>,
    pub(in crate::retrieval) t_postings: Vec<Vec<u32>>,
    /// Private vocabulary ID space for 1-hop ingest WordNet bridge lemmas so `ix.vocab`
    /// and all primary BM25F token IDs remain 100% bit-for-bit identical to `main`.
    pub(in crate::retrieval) bridge_vocab: Vocab,
    /// 1-hop WordNet lemma IDs (in `bridge_vocab` ID space) indexed per turn.
    pub(in crate::retrieval) turn_bridge_postings: Vec<Vec<u32>>,
    /// Corpus-learned Positive PMI associations `(partner_id, weight)` over rare content tokens.
    pub(in crate::retrieval) ppmi_neighbors: DetHashMap<u32, Vec<(u32, f32)>>,
    /// O(1) precomputed supersession flag (`true` if any later turn supersedes `ti`).
    pub(in crate::retrieval) turn_superseded: Vec<bool>,
    /// Burton Bloom 64-bit membership filters per session and per turn.
    #[allow(dead_code)]
    pub(in crate::retrieval) session_bloom: Vec<u64>,
    #[allow(dead_code)]
    pub(in crate::retrieval) turn_bloom: Vec<u64>,
    /// Cache-aligned `<= 24B` turn metadata (20,000 turns <= 480 KB, fits in Sandy Bridge L3).
    #[allow(dead_code)]
    pub(in crate::retrieval) turn_meta: Vec<TurnMeta>,
    /// [W2-028/031] 64-bit bloom register over durable user-directive turns & their turn indices.
    pub(in crate::retrieval) standing_register: u64,
    pub(in crate::retrieval) standing_directive_turns: Vec<usize>,
    /// User turns explicitly asserting compound personal preferences (`"i prefer"`, `"i'd like to"`, `"i'd rather"`).
    pub(in crate::retrieval) preference_directive_turns: Vec<usize>,
    /// [#746] Torrens functional title register & conservation-of-possession spatial log.
    pub(in crate::retrieval) torrens_records: Vec<crate::retrieval::graph::torrens::TorrensRecord>,
    pub(in crate::retrieval) torrens_superseded: Vec<bool>,
    pub(in crate::retrieval) spatial_events: Vec<crate::retrieval::graph::torrens::SpatialEvent>,
}

/// Compact cache-aligned turn metadata (`<= 24 bytes`) for L3-resident linear scans.
#[repr(C)]
#[derive(Clone, Copy)]
pub(in crate::retrieval) struct TurnMeta {
    pub(in crate::retrieval) bloom: u64,
    pub(in crate::retrieval) session: u32,
    pub(in crate::retrieval) content_len: u16,
    pub(in crate::retrieval) entity_len: u16,
    pub(in crate::retrieval) flags: u16,
    pub(in crate::retrieval) field: u8,
    pub(in crate::retrieval) _pad: u8,
}

const _: () = assert!(std::mem::size_of::<TurnMeta>() <= 24);

pub(in crate::retrieval) const TM_QUESTION: u16 = 1 << 0;
pub(in crate::retrieval) const TM_PURE_INFO_Q: u16 = 1 << 1;
pub(in crate::retrieval) const TM_ANSWERS_Q: u16 = 1 << 2;
pub(in crate::retrieval) const TM_SHORT_REPLY: u16 = 1 << 3;
pub(in crate::retrieval) const TM_BACKCHANNEL: u16 = 1 << 4;
pub(in crate::retrieval) const TM_ANAPHORIC: u16 = 1 << 5;
pub(in crate::retrieval) const TM_CONSTRAINT: u16 = 1 << 6;
pub(in crate::retrieval) const TM_STANDING_POLICY: u16 = 1 << 8;

pub(in crate::retrieval) fn build_index(sessions: &[Session], p: &Params) -> Index {
    let nicks = SpeakerNickIndex::build(sessions);
    let mut vocab = Vocab::default();
    let mut turns: Vec<TurnIx> = Vec::new();
    let mut session_ids = Vec::new();
    let mut session_ts = Vec::new();
    let mut session_ep_keys = Vec::with_capacity(sessions.len());
    let mut session_turns = Vec::new();
    let mut session_entities = Vec::with_capacity(sessions.len());
    let mut session_content = Vec::with_capacity(sessions.len());
    let mut raw_standing_rules: Vec<RawStandingRule> = Vec::new();
    let mut turn_tech_handles: Vec<Vec<u32>> = Vec::new();
    let mut turn_prose_cache: Vec<String> = Vec::new();
    let mut cue_vocab: DetHashMap<String, u32> = DetHashMap::default();
    let mut turn_cues: Vec<Vec<u32>> = Vec::new();
    let mut role_cues: DetHashSet<u32> = DetHashSet::default();
    let mut constraint_terms: DetHashSet<u32> = DetHashSet::default();
    let mut standing_register: u64 = 0;
    let mut standing_directive_turns: Vec<usize> = Vec::new();
    let mut preference_directive_turns: Vec<usize> = Vec::new();
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
            // Index caption tokens into `blip_feats`, turn `content`, and `seq` (after sentinel gap)
            // for turn-level visual queries without polluting session-level `s_cont`/`s_ents` exact masks.
            if let Some(cap) = blip_opt {
                if !seq.is_empty() {
                    seq.push((u32::MAX, u32::MAX));
                }
                for bw in raw_tokens(cap).into_iter().filter(|w| is_content(w)) {
                    let bid = vocab.intern(&bw);
                    blip_feats.push(Feat { id: bid, w: 1.0, counts_len: false, inheritable: false });
                    content.insert(bid);
                    let bsid = if p.enable_stemming {
                        let bst = stem(&bw);
                        let id = vocab.intern(&stem_term(&bst));
                        blip_feats.push(Feat { id, w: 0.75, counts_len: false, inheritable: false });
                        id
                    } else {
                        bid
                    };
                    seq.push((bid, bsid));
                }
                for bent in proper_entity_tokens(cap) {
                    let eid = vocab.intern(&bent);
                    blip_feats.push(Feat { id: eid, w: 0.85, counts_len: false, inheritable: false });
                    entities.insert(eid);
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
                    if !body.contains('_')
                        && (vocative_partners.is_some_and(|vp| vp.contains(body))
                            || raw_tokens(&speaker).iter().any(|sw| sw == body))
                    {
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
                    if is_content(&part) {
                        content.insert(id);
                        s_cont.insert(id);
                    }
                }
            }
            let slots = typed_slot_tokens(prose_body);
            // Keep `is_personal_constraint` keyed strictly to recorded `key: value` slots
            // so `plan_query` preserves the public `RankResult.intent.preference` contract;
            // prose stance/choice turns enter `preference_directive_turns` and `role_objs` directly below.
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
            if field == 0
                && !is_standing_policy
                && !is_short_question
                && ((!is_pure_info_question && is_personal_constraint)
                    || has_compound_preference_frame(prose_body)
                    || has_comparative_choice_frame(prose_body))
            {
                preference_directive_turns.push(turn_idx);
            }
            if field == 0 && (is_standing_policy || is_personal_constraint || is_durable_user_directive(prose_body)) {
                for &cid in &content {
                    standing_register |= 1u64 << (cid % 64);
                }
                standing_directive_turns.push(turn_idx);
            }
            if is_standing_policy {
                let (has_explicit_scope, scope_words) = extract_rule_scope(prose_body);
                raw_standing_rules.push((
                    standing_policy_id(prose_body),
                    turn_idx,
                    si,
                    ts,
                    has_explicit_scope,
                    scope_words,
                    extract_rule_condition(prose_body),
                ));
            }
            let id = if t.id.is_empty() { format!("{}#{}", s.id, ti) } else { t.id.clone() };
            let body_for_headers = if extract_header_titles(prose_body).is_some() {
                prose_body.split_once('\n').map(|(_, rest)| rest).unwrap_or(prose_body)
            } else {
                prose_body
            }
            .to_string();
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

    let (session_rsb_back, session_supersedes_back, session_episode_prev) =
        rsb::build_rsb_and_supersession_edges(
            &session_turns,
            &session_ep_keys,
            &session_ts,
            &turn_tech_handles,
            &constraint_terms,
            &turn_prose_cache,
            &mut turns,
        );

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
    for (policy_key, turn_idx, sess_idx, ts, has_explicit_scope, scope_words, condition_words) in raw_standing_rules {
        let mut condition = DetHashSet::default();
        for w in &condition_words {
            if let Some(id) = vocab.get(w.as_str()) {
                condition.insert(id);
            }
            if let Some(sid) = vocab.get(&stem_term(&stem(w))) {
                condition.insert(sid);
            }
        }
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
            // `asbab_occasio_index`: enrich standing-rule scope with rare eliciting tokens (`df <= 8`)
            // from adjacent same-session context (`pos - 2 ..= pos + 1`) so zero-overlap directives retrieve.
            let ctx_lo = pos.saturating_sub(2);
            let ctx_hi = (pos + 1).min(members.len().saturating_sub(1));
            for &ctx_ti in &members[ctx_lo..=ctx_hi] {
                if ctx_ti == turn_idx {
                    continue;
                }
                for &cid in &turns[ctx_ti].content {
                    let t_df = t_own_stats.df.get(&cid).copied().unwrap_or(99);
                    let s_df = s_stats.df.get(&cid).copied().unwrap_or(99);
                    if t_df <= 8 && s_df <= fan_cut {
                        scope.insert(cid);
                        if let Some(w_str) = vocab.names.get(cid as usize)
                            && !w_str.starts_with('\u{1}')
                        {
                            let st = stem(w_str);
                            if let Some(sid) = vocab.get(&stem_term(&st))
                                && s_stats.df.get(&sid).copied().unwrap_or(99) <= fan_cut
                            {
                                scope.insert(sid);
                            }
                        }
                    }
                }
            }
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
            condition,
            condition_words,
        });
    }

    let vocab_len = vocab.names.len();
    let s_postings = build_postings(&session_bags, vocab_len);
    let w_postings = build_postings(&window_bags, vocab_len);
    let (t_postings, turn_bloom) = build_turn_postings_and_bloom(&turn_bags, &turn_own_bags, vocab_len);
    let session_bloom = build_session_bloom(&session_bags);
    let turn_meta = build_turn_meta(&turns, &turn_bloom);

    let is_monologue_stream = !turns.iter().any(|t| t.field == 1)
        && !turns
            .iter()
            .take(16)
            .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    let (torrens_records, torrens_superseded, spatial_events) = if is_monologue_stream {
        crate::retrieval::graph::torrens::build_torrens_and_spatial_index(&turn_prose_cache, &turns)
    } else {
        (Vec::new(), vec![false; turns.len()], Vec::new())
    };

    let mut turn_superseded = if is_monologue_stream {
        torrens_superseded.clone()
    } else {
        vec![false; turns.len()]
    };
    for t in &turns {
        for &old in &t.supersedes_turns {
            if old < turn_superseded.len() {
                turn_superseded[old] = true;
            }
        }
    }

    let (bridge_vocab, turn_bridge_postings, ppmi_neighbors) =
        postings::build_bridge_postings_and_ppmi(p, &vocab, &turns, &t_own_stats);

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
        standing_register,
        standing_directive_turns,
        preference_directive_turns,
        torrens_records,
        torrens_superseded,
        spatial_events,
    }
}
