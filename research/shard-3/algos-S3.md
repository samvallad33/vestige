# Shard 3 algorithms

Ninety designs, S3-001 through S3-090, from Jewish and Islamic legal, textual-transmission, and responsa records fetched live. Forty-five are a single mechanism. Forty-five join two mechanisms into one predicate. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`. Hooks are only `crates/vestige-core/src/retrieval.rs` at 3121-3144 (feat emission), 5028-5032 (scene-positive seed), or 6272-6273 (seat after Step 33, before the replay dump). No design reads `TurnIx.body` while ranking, calls `rsb_seat_behind`, edits `HYPERNYM_LATTICE`, or retunes an existing scalar.

Closed-class tokens used as shape tests: if, not, no, never, and, or, to, for, so, will, must, shall, you, I, all, any, every, more, less, than, who, what, which, when, where. Orthography: digits, title case, brackets, emphasis glyphs, quotation marks, newlines, punctuation, token count. No domain word list.

Primary ceilings, from funnel-S3.md: instruction following slot-1 reorder tops out at 18/40, and the current pool at 35/40, one short of 36. Preference following tops out at 14/39 and 30/39, six short of 36. The first ten builds are index and seed rules. Ordering seats come after gold can enter the pool.

## Top 10 build order

1. **S3-071** (BEAM instruction_following and preference_following, no_pool). Copy a digit onto the next turn only when that turn is a completion (digit, quote, or short assent). The verbal offer itself is not operative.
2. **S3-048** (BEAM instruction_following, no_pool). The one directive prototype passes on its digits only, along a shared non-digit token. Not Step 23's full token copy.
3. **S3-009** (BEAM instruction_following, no_pool and pool_not_k). An if that is not doubled is not a gate. The main clause matches without the one-sided condition.
4. **S3-085** (BEAM instruction_following, no_pool and t5_not1). A doubled condition resists a later general ordinance. An undoubled if is demoted instead.
5. **S3-036** (BEAM instruction_following, pool_not_k). Non-adjacent directive turns in one session that share a token pull each other into the seed.
6. **S3-002** (BEAM preference_following and instruction_following, pool_not_k). A specific list followed by any/all expands onto later turns that use the general. The general is not replaced.
7. **S3-017** (BEAM preference_following, no_pool). A short verbal offer does not hold the preference. Its one content token moves to the next declarative, and the offer cannot sit first.
8. **S3-010** (BEAM instruction_following and preference_following, no_pool). When the session has no local hit, one foreign directive turn that holds the token enters the seed. A session that already has the token imports nothing.
9. **S3-067** (BEAM instruction_following, no_pool). Only digits that sit after the last to/for/so in the question are written onto the next turn.
10. **S3-054** (BEAM instruction_following and preference_following). No local hit imports the kingdom's directive. Three or more local hits, and a query with no digit and no negation, use the session's own majority instead.

After those ten, the first ordering seats to build are S3-050 (counted majority, suspended when the query is a claim), S3-012 (one attestation acquits, two convict), and S3-083 (Eduyot's two axes, with a floor of two on a negated query).

## The ninety

### S3-001 dayo_digit_cap

Rabbinic, Sifra baraita of Rabbi Yishmael, tannaitic. Tag ORDERING. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF t5_not1 12; PF t5_not1 7. Combined: single mechanism.

Sifra on Numbers 12:14 draws a kal vachomer from seven days to fourteen, then cuts it back: it suffices that the inference parallel what it is deduced from, so the result stays seven. The query contains a closed-class comparative (more, less, than) and a digit. Candidate turns share the query's non-digit content tokens and each expose one digit. Slot 1 is the turn whose digit equals the query digit when that digit is at most the smallest digit in the shared set (the source measure). A larger digit cannot win by being 'stronger'. No weight is scaled. Not comparative partition (which splits a than-query into two sides) and not ana ±10% approximation. Dayo is a hard cap at the source digit. Kal vachomer here does not escalate.

Math: Let D be the digits of turns that cover the query's non-digit tokens. cap = min(D). Seat argmin_t |digit(t) - d_q| over {t : digit(t) <= cap}, preferring digit(t) = d_q when d_q <= cap. If d_q > cap, seat the turn holding cap.

```rust
fn s3_001_dayo(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let Some(dq) = plan_digit(plan) else { return };
    let mut cap: Option<u32> = None;
    for &i in list.iter() {
        if let Some(d) = turn_digit(&ix.turns[i]) { cap = Some(cap.map_or(d, |c| c.min(d))); }
    }
    let Some(cap) = cap else { return };
    let bound = dq.min(cap);
    if let Some(pos) = list.iter().position(|&i| turn_digit(&ix.turns[i]) == Some(bound)) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-002 prat_uklal_expansion

Rabbinic, Sifra rule 5, tannaitic. Tag RECALL/INDEX. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF no_pool 9; PF pool_not_k 14; IF pool_not_k 14. Combined: single mechanism.

Sifra rule 5, prat u-klal, on Exodus 22:9: an ass or an ox or a lamb is specific, and 'any beast' is general, and the general adds to the specific. Inside one turn, two or more content tokens separated by the closed-class or, followed by any or all plus one further content token. That last token is the general. Turns later in the same session that contain the general token inherit the specific tokens as feats with counts_len false and inheritable false. The general is not replaced. A query that names a listed specific can hit the later general turn. Turn 1 item 3 is kelal u-ferat, which replaces the general with the specific. This rule runs the other direction: the general adds. It does not copy a whole question onto an answer (Step 23).

Math: On a turn with or-list S and a following any/all token g: for each later turn u in the session with g in content(u), add Feat {id: s, w: 1, counts_len: false, inheritable: false} for every s in S not already in content(u).

```rust
fn s3_002_prat_uklal(session: &[TurnDraft], vocab: &mut Vocab) {
    // called from build_index while feats are emitted; reads tokens, never body at rank time
    for (i, t) in session.iter().enumerate() {
        let Some((specs, general)) = or_list_then_any(t) else { continue };
        for u in session.iter().skip(i + 1) {
            if u.content.contains(&general) {
                for s in &specs {
                    if !u.content.contains(s) {
                        u.feats.push(Feat { id: *s, w: 1.0, counts_len: false, inheritable: false });
                    }
                }
            }
        }
    }
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-003 binyan_av_one_prototype

Rabbinic, Sifra rule 3a, tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF no_pool 5. Combined: single mechanism.

Sifra rule 3a, binyan av from one verse: mishkav is not like moshav, but what is common to them is the rule that transfers. A directive-shaped turn is one that contains the closed-class you and no question mark. The first such turn in a session that also holds a content token is the prototype. Later turns in the session that contain that same content token and are not themselves directive inherit a non-length directive flag feat. They become retrievable by an instruction-shaped query that has the token. The prototype's other tokens are not copied. Not gezera shava (a shared rare string bridges two passages both ways). Only a directive flag moves, and only forward from one prototype. Not Step 23's token copy.

Math: Let p be the least turn index in the session with you and no '?'. Let c be its rarest content id. For every later turn u with c in content(u), add Feat {id: DIRECTIVE, w: 1, counts_len: false, inheritable: false}.

```rust
fn s3_003_binyan_one(session: &[TurnDraft]) {
    let Some(p) = session.iter().position(|t| has_you(t) && !t.has_question) else { return };
    let Some(c) = rarest_content(&session[p]) else { return };
    for u in session.iter().skip(p + 1) {
        if u.content.contains(&c) && !(has_you(u) && !u.has_question) {
            u.feats.push(Feat { id: DIRECTIVE, w: 1.0, counts_len: false, inheritable: false });
        }
    }
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-004 binyan_av_two_prototypes

Rabbinic, Sifra rule 3b, tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14. Combined: single mechanism.

Sifra rule 3b derives the rule from two verses that share an introduction ('command') and otherwise differ. One verse is not enough. The same directive flag as S3-003, but the content token must already occur in two directive-shaped turns before any later turn inherits the flag. A single directive mention does not teach the session. The flag is written only after the second directive turn that carries the token. Cuts false instruction hits from a one-off. Step 27's two-witness rule, if built, gives full credit to a token attested twice. This does not change token credit. It gates inheritance of a directive flag on two prototypes.

Math: Flag(u) = 1 iff there exist i < j < u, all in the session, each of i and j directive-shaped, and token c is in all three contents. Otherwise Flag(u) = 0.

```rust
fn s3_004_binyan_two(session: &[TurnDraft]) {
    let mut seen: DetHashMap<u32, u8> = DetHashMap::default();
    for t in session {
        if has_you(t) && !t.has_question {
            for c in t.content.iter() { *seen.entry(*c).or_insert(0) += 1; }
        } else {
            for c in t.content.iter() {
                if seen.get(c).copied().unwrap_or(0) >= 2 {
                    t.feats.push(Feat { id: DIRECTIVE, w: 1.0, counts_len: false, inheritable: false });
                    break;
                }
            }
        }
    }
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-005 quantifiable_majority

Rabbinic, Chullin 11a, amoraic, citing Exodus 23:2. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7; PF k_not5 2. Combined: single mechanism.

Chullin 11a: a piece of meat found before ten shops, nine kosher and one not, follows the majority. The majority is quantifiable and present, not an imagined one. Among pool turns that contain the query's rarest content token, count how many also contain a closed-class negation (not, no, never) and how many do not. The set must be at least three turns so the majority is counted, not guessed. If one polarity's count is strictly greater, the earliest turn of that polarity in the current list takes slot 1. Applies only on preference queries (no you, or the query has a negation or a comparative). Does not fire on a tie. Not the two-witness threshold and not WIN #2's coverage margin. This is a count of turns, and it refuses to move on a tie. S3-050 suspends it when the query is a claim.

Math: Let n1, n0 be the counts of negated and unnegated pool turns holding the rarest query token. If max(n1, n0) >= 3 and max > min, seat the least index of the winning polarity. If n1 == n0, this rule does not move the list (see S3-006).

```rust
fn s3_005_rov(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let Some(c) = rarest_query_token(plan) else { return };
    let mut neg = Vec::new(); let mut pos = Vec::new();
    for (i, &t) in list.iter().enumerate() {
        if !ix.turns[t].content.contains(&c) { continue; }
        if turn_negated(&ix.turns[t]) { neg.push(i); } else { pos.push(i); }
    }
    let (win, lose) = if neg.len() > pos.len() { (&neg, &pos) } else { (&pos, &neg) };
    if win.len() >= 3 && win.len() > lose.len() {
        let t = list.remove(win[0]); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Chullin.11a?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-006 even_split_stipulated_status

Rabbinic, Ketubot 15b, tannaitic mishnah plus Rav. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7. Combined: single mechanism.

Ketubot 15b: if half the city is gentile and half Jewish, the baby is deemed a Jew. Rav adds that a single majority is not enough for lineage. An even split is not 'follow the majority'. Same polarity counts as S3-005, and the two counts are equal and at least one. Block every majority-style swap for this query. Slot 1 stays the incumbent. The tie has a stipulated status, which in the ranker is the turn already seated, not a new vote. Not chazakah (S3-007), which applies when the incumbent itself has no polarity marker. This applies when the attesting set is split evenly.

Math: If n1 = n0 and n1 >= 1, forbid_majority_swap = 1 and the list is unchanged by S3-005 and by any later polarity-count seat in this pass.

```rust
fn s3_006_even_split(ix: &Index, plan: &QueryPlan, list: &[usize]) -> bool {
    let Some(c) = rarest_query_token(plan) else { return false };
    let mut n1 = 0u32; let mut n0 = 0u32;
    for &t in list {
        if !ix.turns[t].content.contains(&c) { continue; }
        if turn_negated(&ix.turns[t]) { n1 += 1; } else { n0 += 1; }
    }
    n1 == n0 && n1 >= 1
}
```

Record: https://www.sefaria.org/api/texts/Ketubot.15b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-007 presumptive_status

Rabbinic, Niddah 2a, tannaitic, the Rabbis against Shammai and Hillel. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7; PF k_not5 2. Combined: single mechanism.

Niddah 2a: where the status of an item is uncertain it retains its presumptive status, and the woman stays in the presumptive status of being pure. Slot 1 contains the query's rarest content token and contains neither a negation nor a comparative. A challenger in the top 5 adds a negation the query does not contain. The challenger does not take slot 1. Uncertainty is not a new status. A later turn that only adds a polarity the query did not ask for leaves the incumbent in place. Not Torrens and not the praetor standing-edict seat. This is a refusal to replace an unmarked incumbent with a polarity the query did not request. It does not prefer the latest will-clause.

Math: Swap is forbidden when polarity(list[0]) = 0 and polarity(challenger) != 0 and the query has no negation token. Otherwise this rule is silent.

```rust
fn s3_007_chazakah(ix: &Index, plan: &QueryPlan, list: &[usize], challenger: usize) -> bool {
    if query_has_negation(plan) { return false; }
    let head = &ix.turns[list[0]];
    !turn_negated(head) && turn_negated(&ix.turns[challenger])
}
```

Record: https://www.sefaria.org/api/texts/Niddah.2a?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-008 unconscious_despair_keeps_ownership

Rabbinic, Bava Metzia 21b, amoraic, Abaye. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; PF no_pool 9. Combined: single mechanism.

Bava Metzia 21b: Abaye says unconscious despair is not despair. The owner who has not learned of the loss maintains ownership. A content token whose document frequency is 1. No second mention exists anywhere in the index. The hapax is stored at weight 1 with a keep-flag. No later pass may drop it, zero it, or mark the turn non-inheritable for lack of a repetition. Silence is not a retraction. Not bitul / frequency nullification by a high-df neighbor, and not a recency deletion. The token stays because the owner never marked it gone.

Math: If df(c) = 1, feat(c).w = 1 and keep(c) = 1. A demotion rule may change w only when keep(c) = 0.

```rust
fn s3_008_yeush(df: &DetHashMap<u32, u32>, feat: &mut Feat) {
    if df.get(&feat.id).copied().unwrap_or(0) == 1 {
        feat.w = 1.0;
        // keep-bit stored in counts_len stays true; a later decay pass must read df and refuse
    }
}
```

Record: https://www.sefaria.org/api/texts/Bava_Metzia.21b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-009 doubled_condition

Rabbinic, Kiddushin 61a, tannaitic, Rabbi Meir. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF pool_not_k 14; IF t5_not1 12. Combined: single mechanism.

Kiddushin 61a, Rabbi Meir: any condition that is not doubled, stating both fulfillment and non-fulfillment, as Moses did with Gad and Reuben, is not a condition. A turn contains the closed-class if. It is doubled only when the same session also contains if and a negation (not, no, never) sharing one content token with the first if-turn. A one-sided if is not a gate. Its if-clause tokens are stored counts_len false, so an instruction query is covered by the main clause without having to match the undoubled condition. A doubled pair stays a real restriction: both branches remain required. Not the praetor's always/never seat. A one-sided if is demoted from the coverage gate; it is not promoted. Turn 1 item 37 already seats standing imperatives and is not this test.

Math: doubled(c, session) = 1 iff some turn has if and c, and some turn has if and not and c. If doubled = 0, every token in the if-clause has counts_len false. If doubled = 1, those tokens keep counts_len true.

```rust
fn s3_009_tenai(session: &[TurnDraft]) {
    let mut both: DetHashSet<u32> = DetHashSet::default();
    // a content id is doubled when it occurs in an if-clause and in an if+negation clause
    for t in session {
        if clause_has_if(t) && clause_has_negation(t) {
            for c in if_clause_tokens(t) { both.insert(c); }
        }
    }
    for t in session {
        if clause_has_if(t) && !clause_has_negation(t) {
            for f in t.feats.iter_mut() {
                if if_clause_tokens(t).contains(&f.id) && !both.contains(&f.id) {
                    f.counts_len = false;
                }
            }
        }
    }
}
```

Record: https://www.sefaria.org/api/texts/Kiddushin.61a?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-010 law_of_the_kingdom_fills_the_gap

Rabbinic, Gittin 10b, amoraic, Shmuel. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; PF no_pool 9. Combined: single mechanism.

Gittin 10b, Shmuel: the law of the kingdom is the law. A transfer the local law accepts is valid. A directive-shaped turn (you, no question mark) in another session contains the query's rarest content token, and the query's own session has no turn containing that token. That foreign directive turn is added to the PPR seed list when t_scene would otherwise be zero, so a session with no local rule can still surface the standing rule. Sessions that already contain the token do not import it. Not a khipu entity bridge (that crosses sessions on a shared entity even when the session already has material). This fires only when the session has zero hits for the token, and only for a directive-shaped turn. It does not rank by which kingdom, and it does not prefer recency.

Math: Import(u) into the seed iff directive(u) and c in content(u) and no turn in the query session contains c. Seed insertion is the existing t_scene > 0 path plus this one extra index.

```rust
fn s3_010_dina(ix: &Index, plan: &QueryPlan, seeds: &mut Vec<usize>) {
    let Some(c) = rarest_query_token(plan) else { return };
    let qs = query_session(ix, plan);
    if ix.turns.iter().any(|t| t.session == qs && t.content.contains(&c)) { return; }
    if let Some(u) = ix.turns.iter().position(|t| t.session != qs && has_you_turn(t) && !t.is_question && t.content.contains(&c)) {
        if !seeds.contains(&u) { seeds.push(u); }
    }
}
```

Record: https://www.sefaria.org/api/texts/Gittin.10b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 5028-5032. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-011 hazamah_alibi_not_story

Rabbinic, Mishnah Makkot 1:4, tannaitic. Tag ORDERING. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF t5_not1 12; PF t5_not1 7. Combined: single mechanism.

Mishnah Makkot 1:4: witnesses are not conspiring until the second set impeaches the witnesses themselves. Saying the killing was elsewhere, while contradicting the story, does not make them hazamah. 'He was with us' at a different place does. Two turns share a content token. A story-contradiction is a different second content token. An alibi is a different digit (the place or the time, both orthographic digits) while the query digit equals only one of them. A story-contradiction leaves both turns eligible. An alibi removes the turn whose digit differs from the query digit from slot-1 contention. The turn is not deleted from the pool. One contradicted token is not zeroed; the whole turn loses slot 1 only on the digit alibi. Not negation-scope zeroing of one token, and not isnad chain length. The impeachment is a digit mismatch against a turn that holds the query digit. A mere different object token does not impeach.

Math: If digit(query) = digit(A) != digit(B) and A and B share a content token, B is ineligible for slot 1. If A and B differ only in a non-digit content token, both stay eligible.

```rust
fn s3_011_hazamah(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let Some(dq) = plan_digit(plan) else { return };
    let Some(c) = rarest_query_token(plan) else { return };
    list.retain(|&t| {
        let d = turn_digit(&ix.turns[t]);
        // drop from slot-1 list only when this turn shares c and its digit conflicts
        !(ix.turns[t].content.contains(&c) && d.is_some() && d != Some(dq) && list.iter().any(|&o| o != t && ix.turns[o].content.contains(&c) && turn_digit(&ix.turns[o]) == Some(dq)))
    });
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Makkot.1.4?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-012 acquit_by_one_convict_by_two

Rabbinic, Mishnah Sanhedrin 4:1, tannaitic. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF k_not5 2; IF k_not5 3; PF t5_not1 7. Combined: single mechanism.

Mishnah Sanhedrin 4:1: a capital court acquits by a majority of one and convicts only by a majority of two. Monetary cases decide either way by a majority of one. A query that contains a closed-class negation is the severe (liability) class. A query with no negation is the lenient class. Attestation is the number of pool turns that contain the rarest query token and match the query's polarity. A lenient seat is allowed from a single attesting turn. A severe seat requires two attesting turns; a lone negated turn cannot take slot 1 over an incumbent that already covers the token. Not a uniform two-witness rule. The threshold flips with the query's polarity. Not hakirot's core-cue gate.

Math: Let a be the number of pool turns with the rare token and the query's polarity. If query has a negation, slot-1 change is legal only when a >= 2. If query has no negation, a >= 1 suffices.

```rust
fn s3_012_majority_gap(ix: &Index, plan: &QueryPlan, list: &[usize]) -> bool {
    let Some(c) = rarest_query_token(plan) else { return false };
    let a = list.iter().filter(|&&t| ix.turns[t].content.contains(&c) && polarity_matches(plan, &ix.turns[t])).count();
    if query_has_negation(plan) { a >= 2 } else { a >= 1 }
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Sanhedrin.4.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-013 undiscussed_note_is_not_a_ruling

Rabbinic, Bava Batra 130b, amoraic, Rabbi Yohanan and Rava. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12. Combined: single mechanism.

Bava Batra 130b: do not act on the ruling unless it is said to be a practical halakha. After the teacher's death, do not tear the note up, and do not learn from it either. A note is a turn with one punctuation unit and fewer than eight tokens. A ruling is a turn with two or more punctuation units covering the same query tokens. The note cannot take slot 1 while a ruling covering the same tokens is in the top 5. The note stays in the pool. It is not deleted and it is not treated as a precedent. Not the information-residual preference for a turn with extra non-query tokens. This is a clause-count bar on short undivided notes. The note stays addressable.

Math: If tokens(list[0]) < 8 and clauses(list[0]) = 1, and some i in 1..5 has clauses(list[i]) >= 2 and covers the head's query tokens, seat list[i].

```rust
fn s3_013_not_a_ruling(ix: &Index, list: &mut Vec<usize>) {
    let head = &ix.turns[list[0]];
    if head.seq.len() >= 8 || clause_count(head) >= 2 { return; }
    if let Some(pos) = (1..list.len().min(5)).find(|&i| clause_count(&ix.turns[list[i]]) >= 2 && covers_same(ix, list[0], list[i])) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Bava_Batra.130b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-014 no_mediation_after_polarity

Rabbinic, Sanhedrin 6b, tannaitic, Rabbi Eliezer son of Rabbi Yosei HaGelili. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7. Combined: single mechanism.

Sanhedrin 6b: once the verdict has been issued it is not permitted to mediate, and Rabbi Eliezer says it is prohibited to mediate a dispute. A turn contains both a negated clause and an unnegated clause over the same content token. The query contains only one of those polarities. The both-polarity turn cannot take slot 1. A turn that carries only the query's polarity, if one is in the top 5, is seated instead. The compromise turn stays in the pool. Turn 1 item 36 machloket keeps both opinions in the top of the list. This refuses the blended turn as slot 1 and does not delete either pure turn.

Math: If polarity_set(list[0]) = {0, 1} and polarity(query) is a singleton p, and some i < 5 has polarity_set = {p}, seat i.

```rust
fn s3_014_no_peshara(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    if !both_polarities(&ix.turns[list[0]]) { return; }
    let want_neg = query_has_negation(plan);
    if let Some(pos) = (1..list.len().min(5)).find(|&i| turn_negated(&ix.turns[list[i]]) == want_neg && !both_polarities(&ix.turns[list[i]])) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Sanhedrin.6b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-015 not_in_heaven

Rabbinic, Bava Metzia 59b, tannaitic, Rabbi Yirmeya on Deuteronomy 30:12. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: single mechanism.

Bava Metzia 59b: it is not in heaven. Once the Torah was given at Sinai, a Divine Voice is not regarded. Exactly one turn in the top 5 has an orthographic spectacle (an exclamation mark, or a token wrapped in emphasis glyphs). At least two other top-5 turns cover the same content tokens with no spectacle. The spectacle turn cannot take slot 1. The earliest plain turn that covers the same tokens is seated. The marked turn stays in the pool. A single sign does not outvote the plain majority. Ancient rubric boosts decorated lines. This is the opposite seat, and only when the decoration is unique against two plain covers. It does not retune the rubric weight.

Math: If |{i < 5 : spectacle(list[i])}| = 1 and |{i < 5 : !spectacle and covers(head)}| >= 2, seat the least plain index.

```rust
fn s3_015_not_in_heaven(ix: &Index, list: &mut Vec<usize>) {
    let marked: Vec<usize> = (0..list.len().min(5)).filter(|&i| spectacle(&ix.turns[list[i]])).collect();
    if marked.len() != 1 { return; }
    let plain: Vec<usize> = (0..list.len().min(5)).filter(|&i| !spectacle(&ix.turns[list[i]])).collect();
    if plain.len() >= 2 {
        let t = list.remove(plain[0]); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Bava_Metzia.59b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-016 crowns_are_not_the_letter

Rabbinic, Menachot 29b, amoraic aggadah on the tagin. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; PF no_pool 9. Combined: single mechanism.

Menachot 29b: Moses finds the crowns being tied on the letters, and they are there for a later reader. The crown is not a different letter. Parenthetical spans, and glyphs * _ # ~ stuck to a token. The letters inside are the token. The wrapper is a separate flag. The inner token is interned as the plain token, so a query of the plain form hits it. The wrapper sets a non-length annotation flag. A query that does not contain the inner token does not match via the flag. Emphasis does not mint a second vocabulary id. Ancient rubric uses decoration as emphasis. This refuses to let the decoration change the token id. Inverted-nun displacement is S3-020 and applies to bracketed spans at seat time, not at intern time.

Math: strip_edge(token) = token with leading and trailing * _ # ~ removed. id = intern(strip_edge). If strip_edge != token, also set form_annotation on the turn. Parenthetical tokens are counts_len false.

```rust
fn s3_016_crowns(raw: &str) -> (String, bool) {
    let bytes = raw.trim_matches(|c| c == '*' || c == '_' || c == '#' || c == '~');
    let marked = bytes.len() != raw.len();
    (bytes.to_string(), marked)
}
```

Record: https://www.sefaria.org/api/texts/Menachot.29b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-017 words_alone_do_not_acquire

Rabbinic, Mishneh Torah Sales 1, medieval, Maimonides, restating Kiddushin. Tag RECALL/INDEX. Targets BEAM preference_following. Miss pattern: PF no_pool 9; PF pool_not_k 14. Combined: single mechanism.

Mishneh Torah, Sales 1: an article is not acquired through a verbal agreement, even when witnesses testify that the principals agreed. A short turn has at most four tokens, or it is a question, and it holds exactly one content token. The next turn in the same session is longer and is not a question. The content token is copied onto the next turn as a non-length feat, and the short turn is marked non-operative for slot 1. Agreement speech does not itself hold the preference. The completing turn does. Step 23 copies every token of an eliciting question onto the assistant answer. This copies one content token off a short offer onto the next declarative, and it bars the short turn from slot 1. Amen-token copy is the other direction and is not repeated.

Math: If |tokens(t)| <= 4 or question(t), and |content(t)| = 1, and next(t) exists in the session and is declarative: add Feat {id: c, counts_len: false} to next(t), and set operative(t) = 0.

```rust
fn s3_017_words_no_title(session: &[TurnDraft]) {
    for i in 0..session.len().saturating_sub(1) {
        let t = &session[i];
        if t.content.len() == 1 && (t.seq.len() <= 4 || t.has_question) && !session[i + 1].has_question {
            let c = *t.content.iter().next().unwrap();
            session[i + 1].feats.push(Feat { id: c, w: 1.0, counts_len: false, inheritable: false });
            session[i].operative = false;
        }
    }
}
```

Record: https://www.sefaria.org/api/texts/Mishneh_Torah%2C_Sales.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-018 release_is_not_acquisition

Rabbinic, Mishnah Kiddushin 1:1, tannaitic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7; PF pool_not_k 14. Combined: single mechanism.

Mishnah Kiddushin 1:1: a woman is acquired by money, document, or intercourse, and acquires herself by divorce or by the husband's death. The ways in are not the ways out. A query with a closed-class negation is a release query. A turn that contains the content token only outside a negated clause is an acquisition turn. An acquisition turn cannot take slot 1 on a release query while any top-5 turn has the token inside a negated clause. The acquisition turn stays in the pool for affirmative queries. Not a global negation-scope zero. Affirmative queries still seat the acquisition turn. Only a release-shaped query refuses it.

Math: If negation(query) = 1 and exists i < 5 with token c inside a negated clause, then every j whose only hits for c are unnegated is ineligible for slot 1.

```rust
fn s3_018_release(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    if !query_has_negation(plan) { return; }
    let Some(c) = rarest_query_token(plan) else { return };
    if !list.iter().take(5).any(|&t| token_under_negation(&ix.turns[t], c)) { return; }
    if !token_under_negation(&ix.turns[list[0]], c) {
        if let Some(pos) = (1..list.len().min(5)).find(|&i| token_under_negation(&ix.turns[list[i]], c)) {
            let t = list.remove(pos); list.insert(0, t);
        }
    }
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Kiddushin.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-019 reopen_for_acquittal_only

Rabbinic, Mishnah Sanhedrin 4:1, tannaitic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF pool_not_k 14; PF k_not5 2. Combined: single mechanism.

Mishnah Sanhedrin 4:1: in capital cases the court brings the case back to acquit and does not bring it back to convict. Monetary cases reopen either way. A later turn in the top pool adds a negation the incumbent lacks, or it removes one. The query either has a negation (severe) or it does not (ordinary). On a severe query, a later turn may enter the top 5 and take slot 1 only when it adds an acquittal relative to the incumbent (it negates where the incumbent did not, matching the query). On an ordinary query, a later turn may not flip the incumbent's polarity at all. Not Torrens recency. A later turn moves only in the acquittal direction on a severe query, and it does not move on an ordinary query just because it is later.

Math: Let flip = polarity(candidate) != polarity(head). Allow the seat iff (query is severe AND polarity(candidate) matches the query) OR (query is ordinary AND flip = 0).

```rust
fn s3_019_reopen(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let severe = query_has_negation(plan);
    for i in 1..list.len().min(10) {
        let flip = turn_negated(&ix.turns[list[i]]) != turn_negated(&ix.turns[list[0]]);
        let matches = turn_negated(&ix.turns[list[i]]) == severe;
        if (severe && matches && flip) || (!severe && !flip && covers_same(ix, list[0], list[i]) && i < 5) {
            if severe && matches && flip {
                let t = list.remove(i); list.insert(0, t);
            }
            return;
        }
    }
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Sanhedrin.4.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-020 inverted_nun_displacement

Masoretic, Numbers 10:35, masoretic pointing of a biblical paragraph. Tag ORDERING. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF t5_not1 12; PF t5_not1 7. Combined: single mechanism.

Numbers 10:35 is fetched with the inverted nun in the Hebrew, the masoretic mark nun hafukha before 'when the ark set out'. The paragraph stays in the text and is marked off from it. Tokens inside square brackets, or inside an HTML comment, or adjacent to the inverted-nun character, are displaced. A turn whose only overlap with the query is displaced tokens cannot take slot 1. The turn remains in the pool. If the same turn also has a non-displaced hit, it stays eligible. Not deletion of the marked paragraph, and not Torrens. The mark keeps the span out of slot 1 while the bytes stay. S3-016 only normalizes emphasis glyphs at intern time.

Math: eligible(t) = 1 iff some query token lands in t outside a displaced span. If the head fails eligible and some i < 5 passes, seat i.

```rust
fn s3_020_nun(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let only_displaced = |t: usize| plan_tokens(plan).iter().all(|c| !ix.turns[t].content.contains(c) || displaced(&ix.turns[t], *c));
    if only_displaced(list[0]) {
        if let Some(pos) = (1..list.len().min(5)).find(|&i| !only_displaced(list[i])) {
            let t = list.remove(pos); list.insert(0, t);
        }
    }
}
```

Record: https://www.sefaria.org/api/texts/Numbers.10.35?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-021 two_widths_of_column_gap

Rabbinic scribal, Tractate Soferim 2:4, minor tractate, geonic transmission of scribal halakha. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF pool_not_k 14; PF pool_not_k 14. Combined: single mechanism.

Tractate Soferim 2:4: a Torah scroll leaves two finger-breadths between columns; scrolls of the Prophets and study copies leave one thumb-breadth. A run of two or more newline characters is a wide gap. A single newline is a narrow gap. Feat inheritance (the expansions in S3-002 and the flags in S3-003) may cross a narrow gap and may not cross a wide gap. The wide gap starts a new column. Coverage is still computed per turn; this rule only stops inheritance. Ancient petuchah splits a long turn at any blank line and scores the best section. This grades the gap: one newline is not a section boundary for inheritance, two newlines are. It does not change the petuchah coverage split.

Math: inherit_across(t, u) = 1 iff u is later in the session and the raw gap between them has fewer than two consecutive newlines. A gap of two or more newlines sets inherit_across = 0.

```rust
fn s3_021_gap(between: &str) -> bool {
    // true means inheritance may cross
    !between.contains("\n\n")
}
```

Record: https://www.sefaria.org/api/texts/Tractate_Soferim.2.4?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-022 local_custom_fills_an_unspecified_query

Rabbinic, Mishnah Bava Metzia 7:1, tannaitic. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: single mechanism.

Mishnah Bava Metzia 7:1: where laborers are not accustomed to start early or stay late, the employer cannot compel it. Where feeding is the custom, he must feed them. The query has no digit and no negation. The session of slot 1 contains at least three turns that share the query's rarest content token. Slot 1 becomes the polarity held by the strict majority of those same-session turns. A one-session custom does not import tokens from outside the session. If the session is split or has fewer than three, the rule is silent. S3-005 counts the whole pool. This counts one session, and only when the query itself states no number and no negation. It does not compel a practice the session does not already show.

Math: Inside session(list[0]), let n1, n0 be polarity counts over turns holding c. If n1 + n0 >= 3 and n1 != n0 and the query has no digit and no negation, seat the earliest turn of the majority polarity.

```rust
fn s3_022_minhag(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    if query_has_negation(plan) || plan_digit(plan).is_some() { return; }
    let Some(c) = rarest_query_token(plan) else { return };
    let sess = ix.turns[list[0]].session;
    let mates: Vec<usize> = list.iter().copied().filter(|&t| ix.turns[t].session == sess && ix.turns[t].content.contains(&c)).collect();
    if mates.len() < 3 { return; }
    let neg = mates.iter().filter(|&&t| turn_negated(&ix.turns[t])).count();
    let pos = mates.len() - neg;
    if neg == pos { return; }
    let want = neg > pos;
    if let Some(posi) = list.iter().position(|&t| mates.contains(&t) && turn_negated(&ix.turns[t]) == want) {
        let t = list.remove(posi); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Bava_Metzia.7.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-023 reconciliation_cannot_flip_the_rule

Hadith, Jami at-Tirmidhi 1352, prophetic report, transmitted in the judgments chapter. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: single mechanism.

Tirmidhi 1352: reconciliation is allowed among the Muslims, except a reconciliation that makes the lawful unlawful or the unlawful lawful. The incumbent matches the query's polarity on the rare token. The challenger matches the opposite polarity and shares the other query tokens. The challenger is not seated. A harmonizing turn may reorder only when it preserves the query's polarity. Flipping lawful to unlawful is rejected even if the challenger is later or longer. Not Torrens, which would let the later turn replace the earlier. This blocks the replacement precisely when the later turn flips polarity.

Math: If polarity(head) = polarity(query) and polarity(candidate) = 1 - polarity(query), the candidate is ineligible for slot 1.

```rust
fn s3_023_sulh(ix: &Index, plan: &QueryPlan, list: &[usize], candidate: usize) -> bool {
    let qn = query_has_negation(plan);
    turn_negated(&ix.turns[list[0]]) == qn && turn_negated(&ix.turns[candidate]) != qn
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-tirmidhi/1352.json via fawazahmed0 hadith-api CDN. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-024 proof_from_the_claimant_oath_from_the_respondent

Hadith, Jami at-Tirmidhi 1341, prophetic report, judgments chapter. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7; PF k_not5 2. Combined: single mechanism.

Tirmidhi 1341: the proof is due from the claimant, and the oath is due from the one the claim is made against. A proof turn has at least eight tokens and contains a content token the incumbent lacks. An oath turn has at most four tokens. A short turn cannot displace slot 1. A long turn can, and only by adding a content token the incumbent does not have. The short turn stays as the respondent's oath and does not become the ruling. WIN #2 coverage margin requires the same session, the same speaker, and every group the head holds plus a margin. This ignores speaker and session and uses a token-length floor: four tokens or fewer never count as proof.

Math: Seat candidate i only when |tokens(i)| >= 8 and content(i) is a proper superset of content(list[0]) intersected with the query. If |tokens(i)| <= 4, i is ineligible.

```rust
fn s3_024_bayyina(ix: &Index, list: &mut Vec<usize>) {
    let head_c = &ix.turns[list[0]].content;
    if let Some(pos) = (1..list.len().min(5)).find(|&i| ix.turns[list[i]].seq.len() >= 8 && head_c.is_subset(&ix.turns[list[i]].content) && ix.turns[list[i]].content.len() > head_c.len()) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-tirmidhi/1341.json via fawazahmed0 hadith-api CDN. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-025 do_not_mirror_the_breach

Hadith, Jami at-Tirmidhi 1264, prophetic report, book of business. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: single mechanism.

Tirmidhi 1264: fulfill the trust for the one who entrusted you, and do not cheat the one who cheated you. The challenger has the same content-token set as the incumbent and the opposite polarity. The mirror is not seated. The incumbent, which holds the trust's polarity, stays. A polarity flip that adds no new content token is retaliation, not a new fact. S3-023 blocks a flip even when the challenger has extra tokens. This one blocks only the pure mirror, where the token set is identical.

Math: If content(candidate) = content(head) and polarity(candidate) != polarity(head), candidate is ineligible for slot 1.

```rust
fn s3_025_no_mirror(ix: &Index, list: &[usize], candidate: usize) -> bool {
    ix.turns[candidate].content == ix.turns[list[0]].content
        && turn_negated(&ix.turns[candidate]) != turn_negated(&ix.turns[list[0]])
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-tirmidhi/1264.json via fawazahmed0 hadith-api CDN. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-026 tie_it_and_rely

Hadith, Jami at-Tirmidhi 2517, prophetic report. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF t5_not1 12; IF no_pool 5. Combined: single mechanism.

Tirmidhi 2517: tie it and rely. Both acts are required. Leaving it loose is not the instruction. The query contains the closed-class and between two content tokens, splitting the query into a left set and a right set. A turn covers the query only when it contains at least one content token from the left of and and one from the right. A turn with only one side does not enter the operative candidate set. Both sides stay in the index. Trial balance asks every query group to be credited somewhere. This is a conjunction inside one query, keyed only on the closed-class and, and it rejects a one-sided turn rather than summing groups.

Math: cover(t) = 1 iff L ∩ content(t) is nonempty and R ∩ content(t) is nonempty, where the query's single top-level and defines L and R. Turns with cover 0 are kept in the pool and barred from slot 1.

```rust
fn s3_026_both(plan: &QueryPlan, turn: &TurnIx) -> bool {
    let Some((left, right)) = split_on_and(plan) else { return true };
    left.iter().any(|c| turn.content.contains(c)) && right.iter().any(|c| turn.content.contains(c))
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-tirmidhi/2517.json via fawazahmed0 hadith-api CDN. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-027 reward_follows_the_intention

Hadith, Sahih al-Bukhari 1, prophetic report, opening hadith of the Revelation chapter. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12; IF k_not5 3. Combined: single mechanism.

Bukhari 1: the reward of deeds depends upon the intentions, and every person will get the reward according to what he has intended. A turn contains a closed-class purpose marker, to, for, or so. Tokens after the last such marker are the intention span. Tokens before it are the preamble. When the query's content tokens sit in the intention span of one top-5 turn and only in the preamble of the incumbent, the intention turn takes slot 1. Preamble-only overlap does not beat an intention hit. Not a scalar boost on the last token. The purpose span is whatever follows the last to, for, or so, and it wins only against a preamble hit.

Math: Let last(t) be the greatest token index in t whose form is to, for, or so. intention(t) = tokens after last(t). Seat the least i < 5 such that query content ⊆ intention(list[i]), if the head's hits are all at indexes <= last(head).

```rust
fn s3_027_niyya(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let in_tail = |t: usize| purpose_index(&ix.turns[t]).is_some_and(|p| query_ids(plan).iter().all(|c| token_pos(&ix.turns[t], *c).is_some_and(|k| k > p)));
    if in_tail(list[0]) { return; }
    if let Some(pos) = (1..list.len().min(5)).find(|&i| in_tail(list[i])) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari/1.json via fawazahmed0 hadith-api CDN. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-028 abrogation_does_not_forget_the_verse

Quran 2:106, Sahih International translation resource 20, quranic. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; PF no_pool 9. Combined: single mechanism.

Quran 2:106 in translation 20: we do not abrogate a verse or cause it to be forgotten except that we bring forth one better than it or similar to it. A later turn shares some but not all content tokens with an earlier turn. 'Better' would be a strict superset or a higher equal-unit digit. 'Similar' would be an equal token set. Anything less is neither. The earlier turn is not marked superseded and is not removed from the index. A later partial overlap writes no supersession edge. Only a later turn that is a superset or an exact copy may be preferred at seat time, and even then the earlier turn stays. Torrens lets the later text replace the earlier. This forbids the replacement unless the later text covers the earlier one, and it never drops the earlier text from the pool. Naskh of the ruling while the text remains is the retention half only.

Math: supersede(earlier) = 0 whenever content(later) is not a superset of content(earlier). The earlier turn's feats are unchanged.

```rust
fn s3_028_no_forget(earlier: &TurnIx, later: &TurnIx) -> bool {
    // returns true when a supersession edge is forbidden
    !earlier.content.is_subset(&later.content)
}
```

Record: https://api.quran.com/api/v4/verses/by_key/2:106?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-029 muhkam_over_or_chains

Quran 3:7, translation resource 20, quranic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12. Combined: single mechanism.

Quran 3:7: in the Book are verses that are precise, and they are the foundation, and others are unspecific. A precise turn has one clause and no closed-class or. An unspecific turn has two or more or tokens. On an instruction-shaped query (you in the query, or the query is imperative by having no question mark and a content token), a one-clause turn in the top 5 takes slot 1 over a multi-or turn that covers the same tokens. No later 'choice' turn is required. The sanis menu yields to a later user turn that picks one item. This seat does not look for a later choice. One clause outranks two or more ors on its own.

Math: If clauses(head) > 1 and or_count(head) >= 2, and some i < 5 has clauses = 1 and or_count = 0 and covers the head's query tokens, seat i.

```rust
fn s3_029_muhkam(ix: &Index, list: &mut Vec<usize>) {
    let h = &ix.turns[list[0]];
    if or_count(h) < 2 { return; }
    if let Some(pos) = (1..list.len().min(5)).find(|&i| or_count(&ix.turns[list[i]]) == 0 && clause_count(&ix.turns[list[i]]) == 1) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/3:7?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-030 investigate_a_lone_report

Quran 49:6, translation resource 20, quranic. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: single mechanism.

Quran 49:6: if a disobedient one comes with information, investigate, lest you harm a people by ignorance. The incumbent's distinguishing token has document frequency 1, and that token is not itself in the query. A hapax that the query did not ask for cannot be the reason for slot 1. If another top-5 turn covers the query without that hapax, it is seated. If every candidate depends on a hapax the query lacks, the incumbent stays and the rule does not invent a second witness. Not a requirement that every token have two witnesses. Only a hapax that is not in the query is barred from being the sole seat reason. S3-004's two-prototype flag is a different object.

Math: If the symmetric difference content(head) minus query tokens contains a c with df(c) = 1, and some i < 5 covers the query with no df-1 token outside the query, seat i.

```rust
fn s3_030_tabayyun(ix: &Index, plan: &QueryPlan, df: &DetHashMap<u32, u32>, list: &mut Vec<usize>) {
    let q = query_ids(plan);
    let lone = |t: usize| ix.turns[t].content.iter().any(|c| !q.contains(c) && df.get(c).copied().unwrap_or(0) == 1);
    if lone(list[0]) {
        if let Some(pos) = (1..list.len().min(5)).find(|&i| !lone(list[i])) {
            let t = list.remove(pos); list.insert(0, t);
        }
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/49:6?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-031 write_the_specified_term

Quran 2:282, translation resource 20, quranic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF pool_not_k 14. Combined: single mechanism.

Quran 2:282: when you contract a debt for a specified term, write it down, and let a scribe write. A question turn contains one or more digit tokens. The next turn in the same session is not a question. Only the digit tokens, not the rest of the question, are copied onto the next turn as non-length feats. An instruction query that names the number can find the written instrument. Chatter around the number is not copied. Step 23 indexes the answer under every token of the eliciting question. This copies digit tokens only, and only onto the next non-question turn.

Math: For a question t followed by declarative u, for each digit token d in t add Feat {id: d, w: 1, counts_len: false, inheritable: false} to u.

```rust
fn s3_031_scribe(session: &[TurnDraft]) {
    for i in 0..session.len().saturating_sub(1) {
        if session[i].has_question && !session[i + 1].has_question {
            for d in digit_ids(&session[i]) {
                session[i + 1].feats.push(Feat { id: d, w: 1.0, counts_len: false, inheritable: false });
            }
        }
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/2:282?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-032 meaningless_oath_is_not_the_bond

Quran 5:89, translation resource 20, quranic. Tag ORDERING. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF t5_not1 12; PF t5_not1 7. Combined: single mechanism.

Quran 5:89: Allah does not impose blame for what is meaningless in your oaths, and does impose blame for what you contract by oath. An oath-shaped turn contains the closed-class pair I and will. It is meaningless when it contains no content token. It is a contract when it contains at least one content token. A content-free I-will turn cannot take slot 1. If a top-5 turn has I, will, and a content token overlapping the query, that turn is seated instead. The Quaker-minute rule seats the latest will-clause. This does not look at time. It drops only the oath that has no content token. A hedged if-oath is S3-009, not this.

Math: void(t) = 1 iff t contains I and will and content(t) is empty. If void(head) = 1 and some i < 5 has I, will, and a nonempty query overlap, seat i.

```rust
fn s3_032_laghw(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let void = |t: &TurnIx| has_i_will(t) && t.content.is_empty();
    if void(&ix.turns[list[0]]) {
        if let Some(pos) = (1..list.len().min(5)).find(|&i| has_i_will(&ix.turns[list[i]]) && !ix.turns[list[i]].content.is_empty() && overlaps(plan, &ix.turns[list[i]])) {
            let t = list.remove(pos); list.insert(0, t);
        }
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/5:89?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-033 lightened_number_stays_exact

Quran 8:66, translation resource 20, quranic, paired with 8:65. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12; IF pool_not_k 14. Combined: single mechanism.

Quran 8:66 follows 8:65: Allah has lightened it, and the operative steadfast-to-enemy number changes, while 8:65 remains in the text. Two pool turns share the query's non-digit content tokens and hold different digits. The query itself holds a digit. The turn whose digit equals the query digit takes slot 1. The other digit stays in the pool and does not match by being close. No ± tolerance. Ancient ana approximation matches within ten percent. This forbids a cross-digit match whenever an equal digit is in the pool. Both verses stay; nothing is deleted.

Math: If digit(query) = d and some i has digit i = d and shares the non-digit query tokens, seat the least such i. Turns with a different digit are ineligible for slot 1 on this query.

```rust
fn s3_033_exact_digit(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let Some(d) = plan_digit(plan) else { return };
    if let Some(pos) = list.iter().position(|&t| turn_digit(&ix.turns[t]) == Some(d)) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/8:66?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-034 the_text_is_guarded

Quran 15:9, translation resource 20, quranic. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; PF no_pool 9. Combined: single mechanism.

Quran 15:9: it is we who sent down the message, and we will be its guardian. A later turn's content is a near copy of an earlier turn: the earlier content is a subset and the sizes differ by at most two tokens. No supersession edge is written. The earlier turn keeps every feat. A restatement does not guardian the text by replacing it; the original stays the guarded copy. S3-028 forbids supersession on a partial overlap. This forbids it on a near duplicate as well. Ketiv/qere recency handling of near-duplicates is the thing this explicitly refuses to do.

Math: If content(earlier) ⊆ content(later) and |content(later)| - |content(earlier)| <= 2, supersede(earlier) = 0 and feats(earlier) are unchanged.

```rust
fn s3_034_guard(earlier: &TurnIx, later: &TurnIx) -> bool {
    earlier.content.is_subset(&later.content)
        && later.content.len() - earlier.content.len() <= 2
}
```

Record: https://api.quran.com/api/v4/verses/by_key/15:9?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-035 do_not_hasten_the_prefix

Quran 75:16, translation resource 20, quranic. Tag ORDERING. Targets BEAM instruction_following; LoCoMo open_domain. Miss pattern: IF t5_not1 12; LoCoMo open_domain t5_not1 17. Combined: single mechanism.

Quran 75:16: do not move your tongue with it to hasten it. Collection and recitation are a separate duty in 75:17. A query token is a proper character-prefix of a token in turn A, and an exact equal of a token in turn B, and both turns are in the pool. The exact turn takes slot 1 over the prefix turn. The prefix turn stays in the pool for queries that really are the short form. TOT prefix matching is allowed only when the full form has document frequency zero. This rule runs when the full form is present and prefers it. Tadanta is a suffix rule, not a prefix rule.

Math: If token q is a proper prefix of some token in head and an exact token in some i < k, seat the least such i.

```rust
fn s3_035_no_haste(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let prefix_only = |t: usize| query_surface(plan).iter().any(|q| ix.turns[t].surface_has_proper_prefix(q) && !ix.turns[t].content.contains(&intern(q)));
    if prefix_only(list[0]) {
        if let Some(pos) = (1..list.len().min(10)).find(|&i| !prefix_only(list[i]) && exact_cover(plan, &ix.turns[list[i]])) {
            let t = list.remove(pos); list.insert(0, t);
        }
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/75:16?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-036 recited_at_intervals

Quran 17:106, translation resource 20, quranic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF no_pool 5. Combined: single mechanism.

Quran 17:106: a Quran we have separated by intervals, so that you might recite it to the people over a prolonged period. Two directive-shaped turns in the same session share one content token and are not adjacent. At least one turn sits between them. A hit that gives one of them t_scene > 0 pulls the other into the seed list even if its own lexical score was zero. The interval is the point: non-adjacent parts of one recitation enter the pool together. Neither turn is merged away. Bell-join requires adjacent turns and a punctuation continuation. Khipu crosses sessions on an entity. This stays inside one session and requires a gap of at least one turn.

Math: Link(i, j) = 1 iff session(i) = session(j), |i - j| >= 2, both directive-shaped, and content(i) ∩ content(j) nonempty. If i is seeded and Link(i, j), seed j.

```rust
fn s3_036_intervals(ix: &Index, seeds: &mut Vec<usize>) {
    let snap: Vec<usize> = seeds.clone();
    for &i in &snap {
        for (j, t) in ix.turns.iter().enumerate() {
            if t.session != ix.turns[i].session || j.abs_diff(i) < 2 { continue; }
            if directive(&ix.turns[i]) && directive(t) && shares_content(&ix.turns[i], t) && !seeds.contains(&j) {
                seeds.push(j);
            }
        }
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/17:106?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 5028-5032. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-037 follow_the_best_of_it

Quran 39:18, translation resource 20, quranic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12. Combined: single mechanism.

Quran 39:18: those who listen to speech and follow the best of it. Two top-5 turns cover the same query tokens. One has fewer content tokens outside the query. On an instruction-shaped query, the tighter turn takes slot 1. Extra tokens are not evidence of a better answer for an instruction. The longer turn stays in the pool. Information residual treats non-query tokens as the answer-bearing part and is left untouched for open-domain. This tighter-wins rule is instruction-shaped queries only, where 3:7's precise verse is the model. It does not retune residual's weight.

Math: Among i < 5 that cover the query, seat argmin |content(i) minus query|. Ties keep the incumbent.

```rust
fn s3_037_best(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    if !instruction_shaped(plan) { return; }
    let extra = |t: usize| ix.turns[t].content.iter().filter(|c| !query_ids(plan).contains(c)).count();
    let mut best = 0usize;
    for i in 1..list.len().min(5) {
        if covers(plan, &ix.turns[list[i]]) && extra(list[i]) < extra(list[best]) { best = i; }
    }
    if best != 0 { let t = list.remove(best); list.insert(0, t); }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/39:18?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-038 fulfill_the_contract

Quran 5:1, translation resource 20, quranic. Tag ORDERING. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF t5_not1 12; PF pool_not_k 14. Combined: single mechanism.

Quran 5:1: fulfill contracts. The query contains a digit or a capitalized token. A contract turn contains that digit or capitalized token and at least one other content token. A bare turn contains only the digit or the capitalized token. The contract turn takes slot 1 over the bare turn. A number or a name alone is not the contract; the term that accompanies it is. Not S3-024's length floor of eight. Here the bare side is exactly one content token, and the contract side is that token plus any second content token.

Math: If query has a digit or a title-case token n, seat the least i < 5 such that n is in turn i and |content(i)| >= 2, when the head has |content| = 1.

```rust
fn s3_038_aqd(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let Some(n) = digit_or_title(plan) else { return };
    if ix.turns[list[0]].content.len() != 1 { return; }
    if let Some(pos) = (1..list.len().min(5)).find(|&i| ix.turns[list[i]].content.contains(&n) && ix.turns[list[i]].content.len() >= 2) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/5:1?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-039 bequest_does_not_rewrite_the_instruction

Quran 2:180, translation resource 20, quranic. Tag RECALL/INDEX. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF no_pool 9; IF t5_not1 12. Combined: single mechanism.

Quran 2:180: prescribed for you when death approaches, if he leaves wealth, is a bequest. The last turn of a session that contains a digit is the bequest field. Earlier directive turns in the session are the operating instructions. Digit tokens of the last turn are stored in a bequest field (counts_len false). Preference queries that contain a digit may match that field. Instruction queries do not: the bequest cannot take slot 1 on a you-shaped query while an earlier directive turn covers the non-digit tokens. Not Torrens. The last turn does not replace earlier instructions. It is a separate field that preference queries can read and instruction queries cannot use to override.

Math: bequest(t) = 1 iff t is the last turn of its session and t holds a digit. Instruction slot-1 eligibility requires bequest(t) = 0 or no earlier directive cover. Preference match may use the bequest digits.

```rust
fn s3_039_bequest(session: &[TurnDraft]) {
    if let Some(last) = session.last() {
        if digit_ids(last).next().is_some() {
            for d in digit_ids(last) {
                last.bequest_feats.push(Feat { id: d, w: 1.0, counts_len: false, inheritable: false });
            }
        }
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/2:180?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-040 stipulated_shares

Quran 4:11, translation resource 20, quranic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF pool_not_k 14; PF t5_not1 7. Combined: single mechanism.

Quran 4:11: Allah instructs you concerning your children, for the male a share equal to that of two females, and the verse goes on to fix the portions. A turn contains an orthographic fraction or ratio: a digit, a slash or colon, and a digit. The query contains a digit. The ratio turn takes slot 1 over a prose turn that merely shares the query's non-digit tokens. The stipulated share is the record. The prose turn stays in the pool. Not S3-033, which picks an equal digit. This picks the turn that writes a ratio pattern when the query asks with a digit and the head has no ratio.

Math: If query has a digit and some i < 5 matches the regex digit-slash-digit or digit-colon-digit and the head does not, seat i.

```rust
fn s3_040_faraid(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    if plan_digit(plan).is_none() { return; }
    if has_ratio(&ix.turns[list[0]]) { return; }
    if let Some(pos) = (1..list.len().min(5)).find(|&i| has_ratio(&ix.turns[list[i]])) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/4:11?translations=20 via Quran.com v4. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-041 sanctified_before_nightfall

Rabbinic, Rosh Hashanah 25b, amoraic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF t5_not1 12. Combined: single mechanism.

Rosh Hashanah 25b: if the court did not manage to say 'sanctified' before nightfall, the month is not sanctified that way. The declaration has a deadline. A repeated content token occurs first in a question and later in a declarative turn of the same session. The declarative is the sanctification. The question is not. On an instruction query, the question occurrence cannot take slot 1 when the later declarative in the session repeats the token. The declarative is seated. If the only occurrence is the question, and it is the last turn of the session, it is past the deadline and stays out of slot 1. Not the praetor seat of an imperative regardless of age. The question is specifically ineligible once a declarative repetition exists, which is the court having to say the formula rather than merely ask it.

Math: If head is a question and some later same-session declarative i shares a query content token, seat i. If head is a question and is the last turn of its session, it is ineligible.

```rust
fn s3_041_sanctify(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let head = &ix.turns[list[0]];
    if !head.is_question { return; }
    let sess = head.session;
    if let Some(pos) = (1..list.len().min(10)).find(|&i| {
        let t = &ix.turns[list[i]];
        t.session == sess && !t.is_question && shares_query(plan, head, t)
    }) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Rosh_Hashanah.25b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-042 the_extra_sum_needs_witnesses

Rabbinic, Shevuot 45b, amoraic baraita. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF k_not5 2; PF t5_not1 7. Combined: single mechanism.

Shevuot 45b: if the craftsman says two coins and the employer says one, the burden of proof is on the claimant. He must bring witnesses for the additional sum. An oath is not sufficient. The challenger contains a digit the incumbent does not contain. That digit is the extra sum. The challenger takes slot 1 only when a second pool turn also contains that digit. One turn's oath, the digit appearing once, does not collect the extra sum. A uniform two-witness rule would demand two attestations for every token. This demands the second attestation only for a digit the incumbent lacks.

Math: If digit(candidate) is not in digit_set(head), seat candidate only when |{t in pool : digit(candidate) in t}| >= 2.

```rust
fn s3_042_extra_sum(ix: &Index, list: &mut Vec<usize>) {
    let head_digits = digit_set(&ix.turns[list[0]]);
    if let Some(pos) = (1..list.len().min(5)).find(|&i| {
        let extra: Vec<u32> = digit_set(&ix.turns[list[i]]).difference(&head_digits).copied().collect();
        !extra.is_empty() && extra.iter().all(|d| list.iter().filter(|&&t| digit_set(&ix.turns[t]).contains(d)).count() >= 2)
    }) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Shevuot.45b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-043 a_later_ordinance_does_not_rewrite_a_condition

Rabbinic, Gittin 75a, amoraic, Rava and Rav Pappa. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12; IF pool_not_k 14. Combined: single mechanism.

Gittin 75a: one cannot apply Hillel's ordinance to conditional bills of divorce. The later general ordinance does not rewrite the condition. The incumbent contains if. A later turn in the session has no if and overlaps the incumbent's content tokens. The later general turn cannot take slot 1 away from the conditional turn. The ordinance stays in the pool for queries that have no if. Conditional instructions are not silently generalized. Not a global ban on later turns. The bar applies only when the incumbent is conditional and the challenger is a general turn in the same session. Torrens would do the opposite.

Math: If if(head) = 1 and candidate is later in the session and if(candidate) = 0, candidate is ineligible to replace head.

```rust
fn s3_043_conditional_get(ix: &Index, list: &[usize], candidate: usize) -> bool {
    clause_has_if(&ix.turns[list[0]])
        && !clause_has_if(&ix.turns[candidate])
        && ix.turns[candidate].session == ix.turns[list[0]].session
        && ix.turns[candidate].ord > ix.turns[list[0]].ord
}
```

Record: https://www.sefaria.org/api/texts/Gittin.75a?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-044 a_vow_without_an_object_is_void

Rabbinic, Nedarim 15a, amoraic, citing a baraita. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF t5_not1 12. Combined: single mechanism.

Nedarim 15a: 'sleeping is konam for me' is asked as a valid vow and rejected, because a vow does not take hold of something that has no substance, whereas an oath does. A vow-shaped turn contains I and does not contain will, must, or shall. It is void when it has no content token. An oath-shaped turn contains one of those auxiliaries. Instruction queries do not seat a vow-shaped turn that has no content token. A turn with an auxiliary, or a you-directive, that overlaps the query is seated instead. Narrative mentions of the same tokens, with neither auxiliary nor you, stay out of slot 1. S3-032 voids an I-will turn that has no content. This voids an I-without-auxiliary turn that has no object, which is the vow side of the same page's distinction, and it also requires instruction form (auxiliary or you) for slot 1. TAP form precedence is when/where only and is not retuned.

Math: operative(t) = 1 iff content(t) is nonempty AND (has_auxiliary(t) OR (has_you(t) and not question(t))). Slot 1 must be operative.

```rust
fn s3_044_vow_form(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let ok = |t: &TurnIx| !t.content.is_empty() && (has_auxiliary(t) || (has_you_turn(t) && !t.is_question));
    if ok(&ix.turns[list[0]]) { return; }
    if let Some(pos) = (1..list.len().min(5)).find(|&i| ok(&ix.turns[list[i]]) && overlaps(plan, &ix.turns[list[i]])) {
        let t = list.remove(pos); list.insert(0, t);
    }
}
```

Record: https://www.sefaria.org/api/texts/Nedarim.15a?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-045 entire_blocks_majority_as_all

Rabbinic, Horayot 3b, amoraic, Rav Huna son of Rav Hoshaya. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: single mechanism.

Horayot 3b: throughout the Torah the majority of an entity is like all of it, and here the verse says the entire assembly, so majority does not suffice. The query contains the closed-class all, every, or entire. Otherwise it does not. Without all/every, a turn that contains a strict majority of the query's content tokens may take slot 1. With all/every, majority is not enough: the turn must contain every query content token. A partial turn cannot sit first on an 'entire' query. S3-005 is a majority of turns. This is a majority of the query's own tokens, and the words all, every, and entire turn that majority off. Not a scalar retune of group coverage.

Math: If the query has all, every, or entire: require |query ∩ content(t)| = |query|. Otherwise allow |query ∩ content(t)| * 2 > |query| and seat the greatest such coverage in the top 5.

```rust
fn s3_045_entire(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) {
    let q = query_ids(plan);
    let need_all = query_has_all(plan);
    let score = |t: usize| q.iter().filter(|c| ix.turns[t].content.contains(c)).count();
    let ok = |t: usize| if need_all { score(t) == q.len() } else { score(t) * 2 > q.len() };
    if ok(list[0]) && need_all { return; }
    if let Some(pos) = (1..list.len().min(5)).filter(|&i| ok(list[i])).max_by_key(|&i| score(list[i])) {
        if score(list[pos]) > score(list[0]) { let t = list.remove(pos); list.insert(0, t); }
    }
}
```

Record: https://www.sefaria.org/api/texts/Horayot.3b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-046 dayo_only_on_a_doubled_condition

Rabbinic, Sifra kal vachomer plus Kiddushin 61a, tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF t5_not1 12. Combined: dayo digit cap + doubled condition.

Sifra caps the a fortiori at the source measure. Kiddushin 61a says a condition that is not doubled is not a condition. A comparative query (more, less, than, plus a digit) looks at turns that share the non-digit tokens. A digit may move from a source turn to a stronger turn only when the source turn's if is doubled in the session. Undoubled if-turns do not donate their digit. Doubled ones donate it capped at the source digit. The joint rule is what stops a one-sided hypothetical from inventing a stricter number. Running S3-001 and S3-009 separately would cap every comparative and demote every one-sided if. The joint rule caps only a doubled condition's digit and ignores the undoubled one.

Math: Transfer d from source s to turn t only if doubled(s) = 1 and digit(t) <= digit(s) and the query digit <= digit(s). Otherwise the if-clause digits have counts_len false, as in S3-009.

```rust
fn s3_046_dayo_tenai(doubled: bool, source_digit: u32, turn_digit: u32, query_digit: u32) -> bool {
    doubled && turn_digit <= source_digit && query_digit <= source_digit
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 | https://www.sefaria.org/api/texts/Kiddushin.61a?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-047 expansion_stops_at_the_wide_gap

Rabbinic, Sifra rule 5 plus Soferim 2:4, tannaitic rule, minor-tractate scribal measure. Tag RECALL/INDEX. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF pool_not_k 14; IF pool_not_k 14. Combined: prat u-klal expansion + two column-gap widths.

Sifra rule 5: the general adds to the specific. Soferim 2:4: two finger-breadths between Torah columns, one thumb-breadth for a narrower copy. The or-list plus any/all pattern of S3-002, and the newline-run test of S3-021. Specific tokens are inherited by a later general turn only when the gap between them is a single newline or less. A double newline is a new column and the class does not expand across it. S3-002 expands with no gap test. Petuchah scores sections. This joint rule is the expansion predicate and the wide-gap veto as one write.

Math: Add the specific feats to u iff g is in u and wide_gap(t, u) = 0. wide_gap = 1 iff the raw span between the turns contains two consecutive newlines.

```rust
fn s3_047_expand_gap(general_in_u: bool, wide_gap: bool) -> bool {
    general_in_u && !wide_gap
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 | https://www.sefaria.org/api/texts/Tractate_Soferim.2.4?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-048 prototype_inherits_digits_only

Rabbinic Sifra 3a plus Quran 2:282, tannaitic rule and quranic scribe rule. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF pool_not_k 14. Combined: one-verse prototype + digit-only instrument copy.

Sifra 3a builds a rule from what is common to one verse. Quran 2:282 tells the scribe to write the specified term of the debt. The first directive turn in a session is the prototype. The only tokens that move off it, or off the question that elicited it, are digit tokens. Later turns that share a non-digit content token with the prototype inherit its digits as non-length feats, not its other words. An instruction query with a number finds the later step. The rest of the prototype stays put. Step 23 copies every eliciting-question token. S3-003 copies a flag, not digits. The joint rule copies digits only, and only along a shared non-digit token with the one prototype.

Math: For later u sharing a non-digit token with prototype p, add Feat {id: d, counts_len: false} for each digit d in p. No non-digit token is copied.

```rust
fn s3_048_digits_from_prototype(share_nondigit: bool, digit: u32) -> Option<Feat> {
    if share_nondigit { Some(Feat { id: digit, w: 1.0, counts_len: false, inheritable: false }) } else { None }
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 | https://api.quran.com/api/v4/verses/by_key/2:282?translations=20 via Sefaria + Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-049 two_prototypes_and_a_polarity_threshold

Sifra 3b plus Mishnah Sanhedrin 4:1, tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF k_not5 3. Combined: two-verse prototype + acquit-by-one convict-by-two.

Sifra 3b requires two verses before the common rule transfers. Sanhedrin 4:1 acquits by one and convicts by two. A directive flag is inherited only from two prototypes. At seat time a negated query still needs two attesting turns, while an affirmative query may use a single inherited flag. The flag is written only after the second prototype, and a severe query cannot sit on a turn whose flag came from fewer than two attestations. Affirmative queries can. Either rule alone is a different gate. Together, inheritance and the severe seat share one counter, and a single prototype never teaches a negated instruction.

Math: inherit(u, c) = 1 iff prototypes(c) >= 2. Seat a negated query on u only if attestations(c) >= 2. Seat an affirmative query if inherit(u, c) = 1.

```rust
fn s3_049_two_and_polarity(prototypes: u8, attestations: u8, query_negated: bool) -> bool {
    prototypes >= 2 && (!query_negated || attestations >= 2)
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 | https://www.sefaria.org/api/texts/Mishnah_Sanhedrin.4.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-050 majority_yields_to_the_burden

Chullin 11a plus Bava Kamma 46b, amoraic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7; PF pool_not_k 14. Combined: quantifiable majority + burden of proof suspends majority.

Chullin 11a follows a majority that is present and counted, nine shops against one. Bava Kamma 46b says that in monetary matters we do not follow the majority; the burden of proof rests upon the claimant. A claim-shaped query contains a digit or a closed-class interrogative (who, what, which, when, where) plus a content token. Other queries are status queries. Status queries use S3-005's counted majority. Claim queries do not: the turn that adds a content token the majority turns lack takes slot 1, and the majority does not outvote it. WIN #2 always demands a coverage margin. This switches the rule off for claim-shaped queries and on for status queries, which is the page's own split between permitted/prohibited majorities and monetary proof.

Math: If claim(query) = 0, seat the strict turn-majority polarity. If claim(query) = 1, seat the least t whose content properly exceeds the intersection of the majority turns' contents. If no such t exists, leave the incumbent.

```rust
fn s3_050_rov_or_burden(claim: bool, majority_polarity_index: Option<usize>, claimant_index: Option<usize>) -> Option<usize> {
    if claim { claimant_index } else { majority_polarity_index }
}
```

Record: https://www.sefaria.org/api/texts/Chullin.11a?context=0&commentary=0 | https://www.sefaria.org/api/texts/Bava_Kamma.46b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-051 a_tie_keeps_the_presumption

Ketubot 15b plus Niddah 2a, tannaitic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7. Combined: even split is not a majority + presumptive status.

Ketubot 15b gives the even split its own stipulated outcome rather than a majority. Niddah 2a keeps the presumptive status where the status is uncertain. Polarity counts on the rare token are equal, and the incumbent has no polarity marker. The tie does not seat either polarity. The unmarked incumbent stays. That is the stipulated status of the tie, not a third vote. S3-006 blocks a majority swap on any tie. S3-007 blocks a polarity addition onto an unmarked head. The joint rule fires only when both are true, so a tie whose head is already polar keeps S3-006's incumbent even if that incumbent is marked.

Math: If n1 = n0 >= 1 and polarity(head) = unmarked, forbid every polarity seat and return the head.

```rust
fn s3_051_tie_chazakah(n1: u32, n0: u32, head_unmarked: bool) -> bool {
    n1 == n0 && n1 >= 1 && head_unmarked
}
```

Record: https://www.sefaria.org/api/texts/Ketubot.15b?context=0&commentary=0 | https://www.sefaria.org/api/texts/Niddah.2a?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-052 silence_and_restatement_both_keep_the_original

Bava Metzia 21b plus Quran 15:9, amoraic and quranic. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; PF no_pool 9. Combined: unconscious despair is not abandonment + the text is guarded.

Abaye on Bava Metzia 21b: unconscious despair is not despair. Quran 15:9: we will be its guardian. A token with df = 1, or an earlier turn that a later turn nearly copies. Neither silence nor a near-copy writes a drop or a supersession edge. The original feat stays at weight 1. The joint rule is one retention predicate with two triggers. Each half is already a retention rule. Together they share one keep-bit, so a hapax that is also restated is not dropped by either path.

Math: keep(c) = 1 if df(c) = 1 OR a later near-copy exists. Demotion and supersession both require keep = 0.

```rust
fn s3_052_keep(df: u32, near_copy: bool) -> bool {
    df == 1 || near_copy
}
```

Record: https://www.sefaria.org/api/texts/Bava_Metzia.21b?context=0&commentary=0 | https://api.quran.com/api/v4/verses/by_key/15:9?translations=20 via Sefaria + Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-053 one_sided_and_empty_are_both_non_gates

Kiddushin 61a plus Quran 5:89, tannaitic and quranic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF t5_not1 12. Combined: doubled condition + meaningless oath.

Rabbi Meir requires the condition to be doubled. Quran 5:89 does not treat the meaningless oath as the contract. An if-clause that is not doubled, or an I-will turn with an empty content set. Both shapes are stored counts_len false and barred from slot 1. Only a doubled if that also contains a content token, or an I-will turn that contains a content token, remains a gate. One predicate, two failing shapes. S3-009 and S3-032 each demote one shape. The joint rule is the single gate predicate an instruction query is allowed to require.

Math: gate(t) = 1 iff (if(t) = 1 and doubled(t) = 1 and content nonempty) OR (i_will(t) = 1 and content nonempty and if(t) = 0). Otherwise the clause tokens have counts_len false.

```rust
fn s3_053_gate(doubled_if: bool, i_will: bool, content_empty: bool) -> bool {
    !content_empty && (doubled_if || (i_will && !doubled_if))
}
```

Record: https://www.sefaria.org/api/texts/Kiddushin.61a?context=0&commentary=0 | https://api.quran.com/api/v4/verses/by_key/5:89?translations=20 via Sefaria + Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-054 statute_first_then_local_custom

Gittin 10b plus Mishnah Bava Metzia 7:1, amoraic statement and tannaitic mishnah. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; PF no_pool 9; PF t5_not1 7. Combined: law of the kingdom fills a gap + local custom.

Shmuel: the law of the kingdom is the law. The mishnah: local custom decides the workday and the feeding where the parties did not specify. If the session contains no turn with the rare token, import one foreign directive turn (the kingdom). If the session contains three or more and the query has no digit and no negation, seat the session majority (the custom). The two mechanisms are ordered. A local cover blocks the import. An absent local cover blocks the custom count, because there is nothing local to count, and the foreign directive enters the seed instead. S3-010 always considers import. S3-022 always considers the session majority. The joint rule makes them mutually exclusive on the local hit count.

Math: If local_hits(c) = 0, seed the foreign directive. If local_hits(c) >= 3 and the query has no digit and no negation, seat the in-session majority and do not import.

```rust
fn s3_054_kingdom_then_custom(local_hits: usize, query_open: bool) -> &'static str {
    if local_hits == 0 { "import" } else if local_hits >= 3 && query_open { "custom" } else { "silent" }
}
```

Record: https://www.sefaria.org/api/texts/Gittin.10b?context=0&commentary=0 | https://www.sefaria.org/api/texts/Mishnah_Bava_Metzia.7.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 5028-5032. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-055 impeach_the_spectacle_as_a_witness

Mishnah Makkot 1:4 plus Bava Metzia 59b, tannaitic. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: whole-turn alibi impeachment + a lone sign is not in heaven.

Makkot 1:4 impeaches the witnesses themselves, not one sentence of the story. Bava Metzia 59b refuses a lone heavenly sign. A turn is a spectacle (exclamation or emphasis wrap) and its digit differs from the query digit while another turn holds the query digit and a shared content token. The whole spectacle turn leaves slot-1 contention. Zeroing its marked token is not the remedy, and the sign does not outvote the plain turn that holds the query digit. The spectacle turn stays in the pool. S3-011 impeaches on a digit alibi without caring about decoration. S3-015 demotes a unique mark against two plain covers. The joint rule requires both the mark and the digit alibi.

Math: Ineligible(t) = spectacle(t) AND digit(t) != digit(query) AND some other turn shares a content token and holds digit(query).

```rust
fn s3_055_impeach_sign(spectacle: bool, digit_conflict: bool, other_holds_query_digit: bool) -> bool {
    spectacle && digit_conflict && other_holds_query_digit
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Makkot.1.4?context=0&commentary=0 | https://www.sefaria.org/api/texts/Bava_Metzia.59b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-056 a_short_note_loses_to_one_precise_clause

Bava Batra 130b plus Quran 3:7, amoraic and quranic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12. Combined: undiscussed note is not a ruling + precise verse over or-chains.

Bava Batra 130b: do not learn the ruling from the undiscussed note. Quran 3:7: the precise verses are the foundation. The incumbent is either a short one-clause note (under eight tokens) or a multi-or turn. A top-5 turn has exactly one clause, at least eight tokens, and no or. That one precise turn takes slot 1. A short note and a chain of ors both lose to it. The note is not deleted. S3-013 only bars the short note. S3-029 only prefers one clause over ors. The joint seat requires the winner to be long enough to be a ruling and precise enough to be muhkam.

Math: Seat i if clauses(i) = 1 and or_count(i) = 0 and tokens(i) >= 8 and (tokens(head) < 8 or or_count(head) >= 2).

```rust
fn s3_056_precise(head_short_or_or: bool, candidate_one_clause_long: bool) -> bool {
    head_short_or_or && candidate_one_clause_long
}
```

Record: https://www.sefaria.org/api/texts/Bava_Batra.130b?context=0&commentary=0 | https://api.quran.com/api/v4/verses/by_key/3:7?translations=20 via Sefaria + Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-057 no_compromise_and_no_switch_to_liability

Sanhedrin 6b plus Mishnah Sanhedrin 4:1, tannaitic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7. Combined: mediation forbidden + acquittal cannot switch to liability.

Sanhedrin 6b prohibits mediating the dispute. Mishnah Sanhedrin 4:1: one who taught acquittal may not return and teach liability. A turn whose first polarity on a content token is the unnegated one and whose later clause negates it, or a turn that holds both polarities in either order. The query is negated. On a severe query that turn cannot take slot 1. Opening with the lenient side and arriving at the penalty is the forbidden switch, and holding both at once is the forbidden mediation. A turn whose only polarity matches the query is seated if one exists. S3-014 bars any both-polarity turn. The mishnah's switch rule is directional. The joint predicate treats the blend and the lenient-then-severe order as the same ineligibility, and it does not bar a severe-then-lenient turn.

Math: Ineligible(t) on a negated query if both_polarities(t) = 1 OR (first_polarity(t) = positive AND later_polarity(t) = negative).

```rust
fn s3_057_no_switch(query_negated: bool, both: bool, lenient_then_severe: bool) -> bool {
    query_negated && (both || lenient_then_severe)
}
```

Record: https://www.sefaria.org/api/texts/Sanhedrin.6b?context=0&commentary=0 | https://www.sefaria.org/api/texts/Mishnah_Sanhedrin.4.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-058 crowns_intern_brackets_displace

Menachot 29b plus Numbers 10:35, amoraic aggadah and masoretic mark. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; IF t5_not1 12. Combined: crowns are not a new letter + inverted nun stays but cannot sit first.

Menachot 29b puts crowns on the letters for a later reader. Numbers 10:35 keeps the inverted-nun paragraph in the text and marks it off. Emphasis glyphs are stripped at intern time. Square brackets and the inverted nun set a displaced bit on the same token id. The plain token id still matches, so the turn is findable. At seat time a turn whose only query hits have the displaced bit cannot take slot 1. One token record carries both facts. S3-016 only strips glyphs. S3-020 only demotes brackets. The joint write stores one id and a displaced bit, so emphasis does not fork the vocabulary and brackets do not delete the hit.

Math: id = intern(strip_edge(token)). displaced(id, t) = 1 iff the occurrence was bracketed or nun-marked. eligible(t) requires a query hit with displaced = 0.

```rust
fn s3_058_mark(stripped_id: u32, displaced: bool) -> (u32, bool) {
    (stripped_id, displaced)
}
```

Record: https://www.sefaria.org/api/texts/Menachot.29b?context=0&commentary=0 | https://www.sefaria.org/api/texts/Numbers.10.35?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-059 three_media_or_it_is_still_talk

Mishneh Torah Sales 1 plus Mishnah Kiddushin 1:1, tannaitic mishnah, medieval restatement. Tag RECALL/INDEX. Targets BEAM preference_following. Miss pattern: PF no_pool 9; PF pool_not_k 14. Combined: verbal offer does not transfer + three completion media.

Sales 1: a verbal agreement does not acquire, even with witnesses. Kiddushin 1:1: acquisition is by money, by document, or by intercourse. Release is a different pair of modes. A short or question turn holds one content token. A completion medium is the next turn and is one of: it contains a digit, it contains a quoted span (a quotation mark), or it is a declarative of at most three tokens (the assent). The content token is copied onto the next turn only when that next turn is one of the three media. Any other following turn leaves the offer non-operative and copies nothing. Talk followed by more talk does not acquire. S3-017 copies onto any following declarative. The joint rule refuses the copy unless the follower is a digit, a quote, or a short assent. Step 23's full token copy is not used.

Math: copy(c onto u) = 1 iff offer(t) and u = next(t) and (digit(u) or quote(u) or (|tokens(u)| <= 3 and not question(u))). Else operative(t) = 0 and u gains no feat.

```rust
fn s3_059_kinyan(next_has_digit: bool, next_has_quote: bool, next_is_short_assent: bool) -> bool {
    next_has_digit || next_has_quote || next_is_short_assent
}
```

Record: https://www.sefaria.org/api/texts/Mishneh_Torah%2C_Sales.1?context=0&commentary=0 | https://www.sefaria.org/api/texts/Mishnah_Kiddushin.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-060 release_may_be_reopened_acquisition_may_not

Mishnah Kiddushin 1:1 plus Mishnah Sanhedrin 4:1, tannaitic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF pool_not_k 14; PF t5_not1 7. Combined: release is not the acquisition form + reopen for acquittal only.

Kiddushin 1:1 uses different acts to acquire and to release. Sanhedrin 4:1 reopens a capital case to acquit and not to convict. A negated query is a release. A later turn that places the rare token under a negation is an acquittal relative to an affirmative incumbent. On a release query the later negated turn may take slot 1, and an affirmative-only turn may not. On an affirmative query a later turn may not flip the incumbent into a negation. Direction and form are one test. S3-018 only refuses an affirmative turn on a release query. S3-019 only gates reopening. The joint rule seats the later negated turn on a release query and is silent, not merely blocking, on affirmative queries.

Math: Allow seat of later t iff query_negated AND token_under_negation(t) AND NOT token_under_negation(head). Forbid the seat when the query is affirmative.

```rust
fn s3_060_reopen_release(query_negated: bool, later_negated: bool, head_negated: bool) -> bool {
    query_negated && later_negated && !head_negated
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Kiddushin.1.1?context=0&commentary=0 | https://www.sefaria.org/api/texts/Mishnah_Sanhedrin.4.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-061 marked_text_stays_and_does_not_sit

Numbers 10:35 plus Quran 2:106, masoretic and quranic. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; IF t5_not1 12. Combined: inverted-nun retention + abrogation does not forget the verse.

The inverted nun keeps Numbers 10:35 in the scroll. Quran 2:106 does not cause the verse to be forgotten when a better or similar one is brought. A bracketed or nun-marked span, and any earlier turn that a later turn only partially overlaps. Both kinds of turn stay in the index with their feats. The marked turn additionally cannot take slot 1 on displaced-only hits. Retention is the shared half; displacement is the mark's extra bit. Nothing is dropped because a later turn exists. Torrens deletes by recency. The joint rule's only seat effect is the displaced-only bar. The constant true in the keep predicate is the point: partial overlap is not a deletion either.

Math: keep(t) = 1 when marked(t) = 1 or a later turn overlaps t only partially. slot1_barred(t) = 1 when every query hit in t is marked. keep = 1 never implies deletion.

```rust
fn s3_061_keep(marked: bool, partial_later_overlap: bool, query_hits_all_marked: bool) -> (bool, bool) {
    let keep = marked || partial_later_overlap;
    let slot1_barred = query_hits_all_marked;
    (keep, slot1_barred)
}
```

Record: https://www.sefaria.org/api/texts/Numbers.10.35?context=0&commentary=0 | https://api.quran.com/api/v4/verses/by_key/2:106?translations=20 via Sefaria + Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-062 the_end_restricts_inside_the_column

Sifra rule 12b plus Soferim 2:4, tannaitic and scribal. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF pool_not_k 14; PF t5_not1 7. Combined: learned from its end + wide gap stops the restriction.

Sifra 12b, davar ha-lamed mi-sofo: the end of the house passage names stones, wood, and mortar, and that end is what the opening is about. Soferim 2:4 separates columns by a wide gap. The last clause of a turn, after its last comma or semicolon, holds the restricting tokens. A wide gap (two newlines) ends the column. Earlier clauses of the same turn match a query only when the query is compatible with the final clause: the query does not contradict the final clause's polarity, and it shares a token with it or has no extra content token. The restriction does not cross a wide gap into the next turn. Rejected davar me-inyano learned from neighboring context. This restriction is the same turn's final clause, and it stops at Soferim's wide gap. End-weight scalar boosts are not used.

Math: match(earlier clause) = 1 iff polarity(query) is compatible with polarity(final clause) and (query shares a final-clause token or query has no content token outside the turn). The next turn across a wide gap does not inherit the final clause.

```rust
fn s3_062_end_in_column(polarity_compatible: bool, shares_final_or_empty: bool, crosses_wide_gap: bool) -> bool {
    polarity_compatible && shares_final_or_empty && !crosses_wide_gap
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 | https://www.sefaria.org/api/texts/Tractate_Soferim.2.4?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-063 the_extra_digit_needs_two_and_a_long_turn

Shevuot 45b plus Tirmidhi 1341, amoraic baraita and prophetic report. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF k_not5 2; PF t5_not1 7. Combined: extra sum needs witnesses + a short turn is the oath not the proof.

Shevuot 45b: the craftsman needs witnesses for the extra coin; an oath is not enough. Tirmidhi 1341: the proof is on the claimant and the oath is on the one claimed against. A challenger adds a digit the incumbent lacks. It counts as proof only when that digit appears in at least two pool turns and the challenger itself has at least eight tokens. A short turn that mentions the extra digit is the oath and cannot take slot 1. A long turn can, and only with a second attestation of the digit. Both conditions are required together. S3-042 ignores length. S3-024 ignores the second attestation. The joint seat requires both, so a long unique digit and a short repeated digit both fail.

Math: Seat i only if digit(i) minus digit(head) is nonempty AND |{t : digit in t}| >= 2 AND |tokens(i)| >= 8.

```rust
fn s3_063_extra_proof(has_extra_digit: bool, attestations: usize, tokens: usize) -> bool {
    has_extra_digit && attestations >= 2 && tokens >= 8
}
```

Record: https://www.sefaria.org/api/texts/Shevuot.45b?context=0&commentary=0 | https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-tirmidhi/1341.json via Sefaria + fawazahmed0 hadith-api. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-064 proof_adds_a_token_the_oath_does_not

Tirmidhi 1341, both clauses of one report, prophetic report. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7; PF k_not5 2. Combined: claimant must add a token + respondent's short turn cannot displace.

Tirmidhi 1341 states both halves in one sentence: the proof is due from the claimant, and the oath is due from the one the claim is made against. Proof is a turn of at least eight tokens that contains a query content token the incumbent lacks. The oath is a turn of at most four tokens. Slot 1 changes only for the proof shape. The short turn is recorded as present and is skipped. A long turn that adds no new query token is not proof either. S3-024 uses a superset of all content. This joint form requires the new token to be one the query asked for, and it states the short-turn skip in the same predicate.

Math: Seat i iff |tokens(i)| >= 8 AND query ∩ (content(i) minus content(head)) is nonempty. Skip i when |tokens(i)| <= 4.

```rust
fn s3_064_proof_not_oath(tokens: usize, new_query_token: bool) -> bool {
    tokens >= 8 && new_query_token
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-tirmidhi/1341.json via fawazahmed0 hadith-api. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-065 a_mirror_does_not_change_status

Tirmidhi 1264 plus Niddah 2a, prophetic report and tannaitic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7. Combined: do not mirror the breach + presumptive status on uncertainty.

Tirmidhi 1264: fulfill the trust, and do not cheat the one who cheated you. Niddah 2a: uncertainty keeps the presumptive status. The challenger has the same content set as the head and the opposite polarity, and the query itself has no negation that would make the flip the thing asked for. The mirror does not seat. The head's status continues. If the query does contain the negation, the rule is silent and a real negated answer can still win by other rules. S3-025 blocks every pure mirror, including when the query asked for the flip. The joint rule blocks the mirror only when the query did not ask for it, which is the presumption half.

Math: Block(candidate) = content equal AND polarity flipped AND query_negated = 0.

```rust
fn s3_065_mirror_status(same_content: bool, flipped: bool, query_negated: bool) -> bool {
    same_content && flipped && !query_negated
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-tirmidhi/1264.json | https://www.sefaria.org/api/texts/Niddah.2a?context=0&commentary=0 via fawazahmed0 hadith-api + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-066 both_sides_and_the_specific

Tirmidhi 2517 plus Sifra rule 7, prophetic report and tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF pool_not_k 14. Combined: conjunction requires both sides + a general matches only with its specific.

Tirmidhi 2517 requires both acts, tie it and rely. Sifra rule 7: the general needs the specific and the specific needs the general, as with the firstborn and 'the male' and 'the opener of the womb'. The query has and, splitting into left and right. Independently, a turn whose general token (any or all) was indexed next to a specific list only matches when the query also contains one specific. Coverage requires both conjuncts, and a general token does not match alone. A turn that has the general and only one side of and fails. The specific tokens stay non-optional when the indexed turn used any or all. S3-026 is only the conjunction. Kelal u-ferat replaces the general. Rule 7, joined here, keeps the general and demands one specific beside both conjuncts.

Math: cover(t) = 1 iff L hits t AND R hits t AND (t has no any/all OR query hits t's specific list).

```rust
fn s3_066_both_and_specific(left: bool, right: bool, general_needs_specific: bool, specific_hit: bool) -> bool {
    left && right && (!general_needs_specific || specific_hit)
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-tirmidhi/2517.json | https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via fawazahmed0 hadith-api + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-067 only_purpose_span_digits_are_written

Bukhari 1 plus Quran 2:282, prophetic report and quranic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF pool_not_k 14. Combined: intention span after to/for/so + digit-only instrument copy.

Bukhari 1: the reward follows what was intended. Quran 2:282: write the specified term. A question turn has a purpose marker to, for, or so. Digits that occur after the last such marker are the term. Digits before it are preamble. Only purpose-span digits are copied onto the next declarative turn. A number in the preamble is not the debt's term and is not inherited. Instruction queries that quote the intended number find the instrument. S3-031 copies every digit in the question. S3-027 only reorders by the purpose span. The joint rule copies a digit only when it sits in the purpose span.

Math: Copy digit d from question t onto next u iff the token index of d is greater than the last index of to, for, or so in t.

```rust
fn s3_067_purpose_digits(digit_index: usize, last_purpose_index: Option<usize>) -> bool {
    last_purpose_index.is_some_and(|p| digit_index > p)
}
```

Record: https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari/1.json | https://api.quran.com/api/v4/verses/by_key/2:282?translations=20 via fawazahmed0 hadith-api + Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-068 keep_the_verse_match_the_digit

Quran 2:106 plus Quran 8:66, quranic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF t5_not1 12. Combined: later partial overlap does not delete + query digit matches only an equal digit.

Quran 2:106 does not cause the verse to be forgotten except for one better or similar. Quran 8:66 lightens the number in 8:65 and both verses remain. Two turns share non-digit tokens and differ in digit. Neither is removed. Both stay in the index. Slot 1 on a digit query is the equal digit only. A later smaller digit does not delete the earlier larger one, and it does not match a query for the larger digit. Torrens would drop the earlier verse. Ana ±10% would cross the digits. The joint rule keeps both and matches exactly.

Math: drop(either) = 0. Seat t iff digit(t) = digit(query). If the query has no digit, this rule does not reorder.

```rust
fn s3_068_keep_and_equal(digit_t: u32, digit_q: u32) -> (bool, bool) {
    (true, digit_t == digit_q)
}
```

Record: https://api.quran.com/api/v4/verses/by_key/2:106?translations=20 | https://api.quran.com/api/v4/verses/by_key/8:66?translations=20 via Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-069 one_clause_capped_at_the_source_digit

Quran 3:7 plus Sifra dayo, quranic and tannaitic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12. Combined: one clause beats or-chains + comparative digit cannot exceed the source.

Quran 3:7 makes the precise verses the foundation. Sifra caps the kal vachomer at the source measure. A one-clause turn with no or, whose digit is at most the smallest digit among turns sharing the non-digit query tokens. That turn takes slot 1 over a multi-or turn and over a one-clause turn whose digit exceeds the source. Precision and the cap are one seat. S3-029 ignores digits. S3-001 ignores clause shape. A one-clause turn with an over-cap digit does not win this seat.

Math: Eligible(i) = clauses(i) = 1 AND or_count(i) = 0 AND digit(i) <= min(D). Seat the least eligible i in the top 5.

```rust
fn s3_069_precise_capped(one_clause: bool, no_or: bool, digit: u32, cap: u32) -> bool {
    one_clause && no_or && digit <= cap
}
```

Record: https://api.quran.com/api/v4/verses/by_key/3:7?translations=20 | https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Quran.com + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-070 a_hapax_does_not_teach_and_does_not_sole_seat

Quran 49:6 plus Sifra 3b, quranic and tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF t5_not1 12; PF t5_not1 7; IF pool_not_k 14. Combined: lone report is investigated + a flag needs two prototypes.

Quran 49:6 says investigate a lone report. Sifra 3b transfers the rule only from two verses. A content token with df = 1, and a directive flag that would be inherited from prototypes of that token. The hapax does not inherit a directive flag, and it cannot be the sole reason for slot 1 unless the query contains it. Two prototypes are required before the flag exists. One rule, two consequences of a single count. S3-004 writes the flag. S3-030 only reseats. The joint rule refuses to write the flag from a hapax and refuses to seat on that hapax unless the query named it.

Math: flag(c) = 1 iff prototypes(c) >= 2. sole_seat_bar(c) = 1 iff df(c) = 1 AND c is not in the query. Both read the same count.

```rust
fn s3_070_two_or_query(prototypes: u8, df: u32, in_query: bool) -> (bool, bool) {
    (prototypes >= 2, df == 1 && !in_query)
}
```

Record: https://api.quran.com/api/v4/verses/by_key/49:6?translations=20 | https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Quran.com + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-071 digits_copy_only_onto_a_completion

Quran 2:282 plus Mishneh Torah Sales 1, quranic and medieval restatement of tannaitic kinyan. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF no_pool 5; PF no_pool 9. Combined: write the specified term + words alone do not acquire.

Quran 2:282: write the debt. Sales 1: the verbal agreement does not acquire even with witnesses. Digit tokens in a question or a short offer. The next turn is a completion only if it has a digit of its own, a quotation mark, or is a declarative of at most three tokens. Digits copy onto the next turn only when that turn is a completion medium. Otherwise they stay on the offer, and the offer is non-operative for slot 1. This is the main no_pool recovery: the number becomes findable on the turn that actually concludes, and nowhere else. S3-031 copies digits onto any next non-question. S3-059 copies the one content token of a short offer. The joint rule copies digits, and only onto a completion medium. It is not Step 23.

Math: copy(digits of t onto next) = 1 iff next is digit-bearing OR quoted OR a short declarative. Else operative(t) = 0 and next gains nothing.

```rust
fn s3_071_copy_on_completion(next_digit: bool, next_quote: bool, next_short_declarative: bool) -> bool {
    next_digit || next_quote || next_short_declarative
}
```

Record: https://api.quran.com/api/v4/verses/by_key/2:282?translations=20 | https://www.sefaria.org/api/texts/Mishneh_Torah%2C_Sales.1?context=0&commentary=0 via Quran.com + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-072 an_oath_with_if_waits_for_both_branches

Quran 5:89 plus Kiddushin 61a, quranic and tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF t5_not1 12. Combined: oath with if is not yet the contract + condition binds only when doubled.

Quran 5:89 distinguishes the meaningless oath from the contracted one. Kiddushin 61a requires both branches before a condition counts. A turn contains I, will, and if. That turn is not a commitment unless the session doubles the if on the same content token. Until then its clause tokens are counts_len false. An I-will without if and with a content token stays a contract under S3-032 and is not touched here. S3-053 treats a one-sided if and an empty oath as separate failures. This rule is only the intersection: I-will plus if, which binds only when doubled.

Math: gate(t) = 1 iff i_will(t) AND if(t) AND doubled(content, session). Otherwise counts_len of the if-clause is false.

```rust
fn s3_072_oath_if(i_will: bool, has_if: bool, doubled: bool) -> bool {
    i_will && has_if && doubled
}
```

Record: https://api.quran.com/api/v4/verses/by_key/5:89?translations=20 | https://www.sefaria.org/api/texts/Kiddushin.61a?context=0&commentary=0 via Quran.com + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-073 no_cross_digit_and_no_over_cap

Quran 8:66 plus Sifra dayo, quranic and tannaitic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12; IF pool_not_k 14. Combined: equal digit only + a fortiori result capped at the source digit.

Quran 8:66 keeps a different number beside 8:65. Sifra refuses to let the inference exceed the source. A query digit, a pool that contains more than one digit on the same non-digit tokens, and a source digit that is the minimum of those. Slot 1 must equal the query digit, and that digit must be at most the source minimum. A larger 'stronger' digit is ineligible even if it is the only other number. No percentage tolerance fills the gap. S3-033 seats any equal digit. S3-001 caps without requiring equality to the query. The joint seat requires both, and ana ±10% is not a fallback.

Math: Seat t iff digit(t) = digit(query) AND digit(query) <= min(D). If no such t, leave the list unchanged.

```rust
fn s3_073_equal_and_capped(digit_t: u32, digit_q: u32, cap: u32) -> bool {
    digit_t == digit_q && digit_q <= cap
}
```

Record: https://api.quran.com/api/v4/verses/by_key/8:66?translations=20 | https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Quran.com + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-074 the_original_stays_and_the_interval_links

Quran 15:9 plus Quran 17:106, quranic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF pool_not_k 14. Combined: near duplicate does not supersede + non-adjacent directives link.

Quran 15:9 guards the text. Quran 17:106 separates the recitation by intervals. Directive turns in one session, at least one turn apart, sharing a content token. A later near-copy of either. The link pulls the other interval turn into the seed. The near-copy writes no supersession edge on either endpoint. The pool gains the missing part and loses nothing. Bell-join is adjacency. This link requires a gap, and the guard half forbids treating a later restatement as a replacement of either endpoint.

Math: seed(j) if Link(i, j) and i is seeded. supersede(i) = 0 and supersede(j) = 0 even if a later turn nearly copies one of them.

```rust
fn s3_074_link_and_keep(same_session: bool, gap_at_least_two: bool, share: bool, directive: bool) -> bool {
    same_session && gap_at_least_two && share && directive
}
```

Record: https://api.quran.com/api/v4/verses/by_key/15:9?translations=20 | https://api.quran.com/api/v4/verses/by_key/17:106?translations=20 via Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 5028-5032. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-075 the_full_token_at_the_end_restricts

Quran 75:17 plus Sifra 12b, quranic and tannaitic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12; IF pool_not_k 14. Combined: exact token beats a proper prefix + the final clause restricts.

Quran 75:17: upon us is its collection and its recitation. Sifra 12b learns the scope from the end of the passage. The final clause of a turn contains the exact query token. Another turn contains only a proper prefix of that token. The exact final-clause turn takes slot 1. A prefix hit in an earlier clause does not. The final clause is the scope, and the collected form beats the hasty prefix. S3-035 prefers any exact token over a prefix. S3-062 restricts by polarity of the final clause. The joint seat requires the exact token to sit in the final clause.

Math: Seat i if some query token is exact in the final clause of i, and the head's only hit for that token is a proper prefix or sits before the final clause.

```rust
fn s3_075_exact_at_end(exact_in_final: bool, head_is_prefix_or_early: bool) -> bool {
    exact_in_final && head_is_prefix_or_early
}
```

Record: https://api.quran.com/api/v4/verses/by_key/75:17?translations=20 | https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Quran.com + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-076 the_tight_turn_must_also_be_operative

Quran 39:18 plus Nedarim 15a, quranic and amoraic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12. Combined: tighter equal coverage wins + a vow without an object is void.

Quran 39:18: follow the best of the speech. Nedarim 15a: a vow with no substance does not take hold. Among top-5 turns that cover the query, the tighter one has fewer extra content tokens. It also has to be operative: nonempty content, and either an auxiliary or a you-directive. The tightest operative turn takes slot 1. A tighter turn that is only narrative, or an empty vow-shape, is skipped. Instruction form and tightness are one comparison. S3-037 can seat a narrative turn that happens to be tight. S3-044 only checks form. The joint argmin skips inoperative turns.

Math: Seat argmin extra(i) over i < 5 where operative(i) = 1 and cover(i) = 1. operative = content nonempty AND (auxiliary OR you-directive).

```rust
fn s3_076_tight_operative(extra: usize, operative: bool, covers: bool) -> Option<usize> {
    if operative && covers { Some(extra) } else { None }
}
```

Record: https://api.quran.com/api/v4/verses/by_key/39:18?translations=20 | https://www.sefaria.org/api/texts/Nedarim.15a?context=0&commentary=0 via Quran.com + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-077 expand_only_toward_the_specifics_shape

Sifra rule 6 plus Sifra rule 5, tannaitic. Tag RECALL/INDEX. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF pool_not_k 14; IF pool_not_k 14. Combined: judge by what resembles the specific + the general adds to the list.

Sifra rule 6, kelal u-prat u-kelal: you judge only by what is like the specific. Rule 5: the general adds to the specific rather than replacing it. An or-list of specifics followed by any or all. A candidate token resembles the list when it shares the list's orthographic class: digit if the specifics are digits, title-case if they are title-case, or the same initial character class otherwise. The general adds only resembling tokens from later same-session turns, as non-length feats on the specific turn, and the other way around only for those tokens. Non-resembling tokens are not pulled into the class. Kelal u-ferat replaces the general with the specific and is turn 1 item 3. This keeps the general and admits only the orthographic class of the listed specifics.

Math: Add Feat s onto a later turn that has the general, and add a later token g' onto the specific turn, iff class(g') = class(S). class is digit, title-case, or other.

```rust
fn s3_077_like_the_specific(same_class: bool, has_general: bool) -> bool {
    same_class && has_general
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-078 the_exception_teaches_the_class_leniently

Sifra rules 8 and 9, tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF t5_not1 12. Combined: an item that left the class teaches the whole class + it teaches leniency only.

Sifra rule 8: what left the general category to teach, teaches the whole category. Rule 9: when it left for a requirement of the same matter, it left to be lenient, not to be strict. One turn in a session has a directive shape and a content token shared with other turns. Those other turns are the class. The directive flag propagates to every turn that shares the token. Propagation may clear a required counts_len bit (a requirement dropped, the leniency). It may not add a new required counts_len bit on the class. The exception does not invent a stricter gate. S3-003 propagates a flag from the first prototype only forward, without a leniency limit. Rule 8 plus rule 9 propagate to the whole class and forbid adding a requirement.

Math: For every u sharing c with the directive exception e: set directive(u) = 1. For feats of u that are counts_len and absent from e, set counts_len false. Do not set counts_len true on any feat of u that e lacks.

```rust
fn s3_078_teach_lenient(shared: bool, exception_directive: bool, feat_required_and_absent_from_exception: bool) -> (bool, bool) {
    let flag = shared && exception_directive;
    let drop_required = flag && feat_required_and_absent_from_exception;
    (flag, drop_required)
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-079 a_foreign_departure_returns_only_if_restored

Sifra rules 10 and 11, tannaitic. Tag RECALL/INDEX. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF pool_not_k 14; PF no_pool 9. Combined: a foreign departure may lighten or tighten + a new matter returns only by explicit restoration.

Sifra rule 10: what left the category for a foreign requirement left both to be lenient and to be strict. Rule 11: what left for a new learning does not return unless the text restores it explicitly. A specific turn's extra token is foreign when its orthographic class differs from the general turn's tokens. A new-matter turn introduces a content token the general turn lacks. A foreign-class specific may both add and drop counts_len requirements on itself only. A new-matter turn does not inherit the general's directive flag unless a third turn in the session contains both the general's token and the new token. That third turn is the explicit restoration. The rejected context rule learned a scope from neighboring turns without an explicit joint mention. Rule 11 requires the third turn to repeat both tokens. Rule 10's two-way change applies only to a different orthographic class.

Math: If class(extra) != class(general): the specific turn may flip its own counts_len bits. inherit(new_matter) = 1 iff a third turn contains both tokens. Otherwise inherit = 0.

```rust
fn s3_079_foreign_or_restored(foreign_class: bool, third_turn_has_both: bool) -> (bool, bool) {
    (foreign_class, third_turn_has_both)
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-080 the_end_is_the_specific_the_opening_must_resemble

Sifra 12b plus Sifra rule 6, tannaitic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF t5_not1 12. Combined: the end defines the scope + the class must resemble the specific.

Sifra 12b takes the scope from the end of the passage. Rule 6 judges the returned general by what resembles the specific. The final clause's content tokens are the specific. Earlier clauses of the turn are the general. An earlier clause matches the query only when the query shares the final clause's orthographic class (digit, title-case, or other) or shares an actual final-clause token. A query whose tokens are a different class from the ending does not get the opening. S3-062 is polarity compatibility of the final clause. This joint test is orthographic class, which is rule 6's 'like the specific' applied to rule 12b's end.

Math: match(opening) = 1 iff class(query) = class(final) OR query ∩ final is nonempty.

```rust
fn s3_080_end_resembles(same_class: bool, shares_final_token: bool) -> bool {
    same_class || shares_final_token
}
```

Record: https://www.sefaria.org/api/texts/Sifra%2C_Baraita_DeRabbi_Yishmael.1.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-081 do_not_open_on_the_penalty_or_return_to_it

Mishnah Sanhedrin 4:1, two clauses, tannaitic. Tag ORDERING. Targets BEAM preference_following. Miss pattern: PF t5_not1 7; PF k_not5 2. Combined: a severe case does not open on the penalty + a lenient argument cannot switch to liability.

Mishnah Sanhedrin 4:1: capital cases open toward acquittal and do not open toward liability, and one who taught acquittal may not return and teach liability. On a negated query, look at the order of polarity inside the candidate turn. A turn whose first clause is negated is ineligible for slot 1 on a severe query when a later clause is affirmative: that opens on the penalty and then leaves it, which the mishnah does not allow as the liability seat. A turn whose first clause is affirmative and whose later clause is negated is the forbidden return to liability. The eligible severe seat is a turn whose clauses, if it has two, do not do either of those, or a single-clause turn that matches the query. S3-057 bars both-polarity and lenient-then-severe only. This joint clause also bars severe-then-lenient as an opening on the penalty. A single matching clause remains eligible.

Math: Ineligible on a negated query if (first = negative AND later = positive) OR (first = positive AND later = negative). A single-clause turn that matches the query stays eligible.

```rust
fn s3_081_opening(query_negated: bool, first_neg: bool, later_neg: Option<bool>) -> bool {
    if !query_negated { return true; }
    match later_neg {
        None => true,
        Some(later) => first_neg == later,
    }
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Sanhedrin.4.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-082 liability_is_concluded_later_and_not_by_a_question

Mishnah Sanhedrin 4:1, the day clauses, tannaitic. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF pool_not_k 14; IF t5_not1 12. Combined: liability waits for a later turn + a question does not conclude.

Mishnah Sanhedrin 4:1: capital liability is not concluded on the same day, and the conclusion is by day. Acquittal may be concluded on the same day. Monetary cases may be concluded at night. A negated query is liability. The first turn in the session that contains the rare token is the accusation. A later declarative turn in the session is the conclusion. A question is not a conclusion. On a liability query, slot 1 must be a later declarative, not the first mention and not a question. On a non-negated query the first mention may sit. The first mention stays in the pool either way. Not Torrens: the later turn wins only on a negated query, and only over the first mention, not over every earlier turn in the corpus. Rosh Hashanah's deadline is a different page and is not this seat.

Math: If query is negated: seat the least i whose turn is declarative, same session as the first hit, and index greater than the first hit. If the head is that later declarative, leave it.

```rust
fn s3_082_later_declarative(query_negated: bool, is_later: bool, is_question: bool) -> bool {
    !query_negated || (is_later && !is_question)
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Sanhedrin.4.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-083 two_axes_and_two_witnesses_to_convict

Mishnah Eduyot 1:5 plus Mishnah Sanhedrin 4:1, tannaitic. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF k_not5 3. Combined: override only if greater in wisdom and in number + liability needs two attestations.

Eduyot 1:5: a court does not set the earlier decision aside unless it is greater in wisdom and in number. Either axis alone is not enough. Sanhedrin 4:1 convicts only by a majority of two. Wisdom is the number of query groups the candidate covers. Number is the number of pool turns that share its rare token and its polarity. Liability queries are negated. A challenger replaces the incumbent only when it covers more query groups and has more agreeing turns. On a negated query the agreeing-turn count must also be at least two. One axis, or a single severe witness, does not move slot 1. Not machloket, which keeps both, and not Torrens, which uses time. Eduyot's two axes are group-count and ally-count. The Sanhedrin half adds the floor of two on a negated query only.

Math: Replace iff groups(c) > groups(head) AND allies(c) > allies(head) AND (query is not negated OR allies(c) >= 2).

```rust
fn s3_083_two_axes(groups_c: usize, groups_h: usize, allies_c: usize, allies_h: usize, negated: bool) -> bool {
    groups_c > groups_h && allies_c > allies_h && (!negated || allies_c >= 2)
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Eduyot.1.5?context=0&commentary=0 | https://www.sefaria.org/api/texts/Mishnah_Sanhedrin.4.1?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-084 cite_the_other_side_first_and_keep_it

Eruvin 13b plus Mishnah Eduyot 1:5, tannaitic. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: the side that quotes the other side first is followed + the quoted side stays recorded.

Eruvin 13b: both are the words of the living God, and the halakha follows Beit Hillel because they taught the other view and put Beit Shammai's statements first. Eduyot 1:5 records the minority so a later court that is greater in wisdom and in number can rely on it. A turn contains its own content tokens and a content token that also appears in a different top-5 turn, and the shared token occurs earlier in the turn than the turn's unique token. That citing turn takes slot 1. The cited turn is not removed from the pool and is not marked superseded. Quoting the rival first is the reason for the seat, and keeping the rival is required. Machloket seats both opinions as a pair and does not pick one for putting the other first. Torrens would drop the earlier rival. This seats the citing turn and keeps the rival.

Math: Seat i if some j < 5, j != i, shares token s, and the first index of s in i is less than the first index of a token that is in i and not in j. Do not drop j.

```rust
fn s3_084_cite_first(shared_before_unique: bool) -> (bool, bool) {
    (shared_before_unique, true)
}
```

Record: https://www.sefaria.org/api/texts/Eruvin.13b?context=0&commentary=0 | https://www.sefaria.org/api/texts/Mishnah_Eduyot.1.5?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-085 a_doubled_condition_resists_the_later_ordinance

Gittin 75a plus Kiddushin 61a, amoraic discussion of a tannaitic condition. Tag RECALL/INDEX. Targets BEAM instruction_following. Miss pattern: IF no_pool 5; IF t5_not1 12; IF pool_not_k 14. Combined: a later general ordinance does not rewrite a condition + the condition counts only if doubled.

Gittin 75a: Hillel's ordinance is not applied to conditional bills of divorce. Kiddushin 61a: an undoubled condition is not a condition. If the if is doubled, the conditional turn is marked resistant: a later general turn in the session cannot inherit its tokens and cannot take its slot. If the if is not doubled, the if-clause is counts_len false and the later general turn is not blocked. One predicate with two outcomes. Real conditions stay specific against a later ordinance. Fake conditions get out of the way so the main clause and a later general statement can match. S3-043 blocks every later general turn whenever the head contains if, including a one-sided if. The joint rule blocks the later ordinance only for a doubled condition and demotes the rest.

Math: If doubled(c): resistant(turn) = 1 and later general turns do not inherit c. If not doubled: counts_len of the if-clause is false and resistant = 0.

```rust
fn s3_085_resist_or_drop(doubled: bool) -> (bool, bool) {
    // (resistant, demote_if_clause)
    (doubled, !doubled)
}
```

Record: https://www.sefaria.org/api/texts/Gittin.75a?context=0&commentary=0 | https://www.sefaria.org/api/texts/Kiddushin.61a?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-086 void_unless_it_is_an_oath_with_an_object_or_a_directive

Nedarim 15a plus Quran 5:89, amoraic and quranic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF pool_not_k 14; IF t5_not1 12. Combined: a vow without substance is void + a meaningless oath is not the contract.

Nedarim 15a: vows do not take hold of what has no substance, and oaths do. Quran 5:89: no blame for the meaningless oath, blame for the oath you contract. A candidate is operative only in two shapes: an auxiliary clause (will, must, shall) with at least one content token, or a you-directive with a content token and no question mark. An I without an auxiliary and without a content token is the void vow. An I-will with no content token is the meaningless oath. Slot 1 on an instruction query has to be one of the two operative shapes. Both void shapes lose to any operative top-5 turn that overlaps the query. The two pages supply the two void shapes and the two valid ones as a single form test. S3-032 checks only I-will. S3-044 checks only the vow side. The joint test is the disjunction of the two valid shapes, and both empty shapes fail it.

Math: operative = (auxiliary AND content nonempty) OR (you AND not question AND content nonempty). Seat the least operative i < 5 that overlaps the query, if the head is not operative.

```rust
fn s3_086_operative(auxiliary: bool, you_directive: bool, content_empty: bool) -> bool {
    !content_empty && (auxiliary || you_directive)
}
```

Record: https://www.sefaria.org/api/texts/Nedarim.15a?context=0&commentary=0 | https://api.quran.com/api/v4/verses/by_key/5:89?translations=20 via Sefaria + Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-087 the_declaration_must_be_complete_and_in_time

Rosh Hashanah 25b plus Quran 75:16, amoraic and quranic. Tag ORDERING. Targets BEAM instruction_following. Miss pattern: IF t5_not1 12; IF pool_not_k 14. Combined: sanctification is a spoken declaration before the deadline + do not hasten a prefix.

Rosh Hashanah 25b: if the court does not say 'sanctified' before nightfall, the declaration does not count. Quran 75:16: do not hasten with the tongue. The confirming turn is a later declarative in the session that contains the exact query token, not a proper prefix, and it is not the last turn if that last turn is a question. Slot 1 is the later exact declarative. A question, a prefix, or a last-turn question after the deadline does not sanctify. The earlier question stays in the pool. S3-041 accepts any later declarative repetition. S3-035 prefers exact over prefix anywhere. The joint seat requires a later exact declarative and rejects a trailing question.

Math: Seat the least i that is declarative, exact on a query token, and not a final-turn question. Ineligible: question, proper-prefix-only, or last turn of the session when that turn is a question.

```rust
fn s3_087_in_time(declarative: bool, exact: bool, last_and_question: bool) -> bool {
    declarative && exact && !last_and_question
}
```

Record: https://www.sefaria.org/api/texts/Rosh_Hashanah.25b?context=0&commentary=0 | https://api.quran.com/api/v4/verses/by_key/75:16?translations=20 via Sefaria + Quran.com. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-088 only_the_alibi_impeaches

Mishnah Makkot 1:4, both halves, tannaitic. Tag ORDERING. Targets BEAM instruction_following; BEAM preference_following. Miss pattern: IF t5_not1 12; PF t5_not1 7. Combined: a digit alibi impeaches the whole turn + a different object does not.

Mishnah Makkot 1:4: contradicting the event, the man or the place of the story, does not render the first witnesses conspiring. 'He was with us' at another place does. Two turns share a content token. They differ either in a digit or only in another content token. The query has a digit. If the digits differ and one of them equals the query digit, the other turn is ineligible for slot 1 as a whole. If the digits agree or the query has no digit, a different content token does not impeach anyone. Both turns stay in the pool in both cases. Negation scope zeros one token. This either removes the whole turn from slot 1 or removes nothing. Isnad length is not consulted. S3-011 is this pair; the combination column records both halves of the mishnah as one predicate so the story-contradiction branch is explicit.

Math: impeach(B) = 1 iff digit(query) = digit(A) != digit(B) and A, B share a content token. If the only difference is a non-digit token, impeach = 0.

```rust
fn s3_088_alibi(digits_differ: bool, one_equals_query: bool, only_object_differs: bool) -> bool {
    digits_differ && one_equals_query && !only_object_differs
}
```

Record: https://www.sefaria.org/api/texts/Mishnah_Makkot.1.4?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-089 a_lone_sign_loses_and_the_two_sided_turn_stays

Bava Metzia 59b plus Eruvin 13b, tannaitic. Tag ORDERING. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF t5_not1 7; IF t5_not1 12. Combined: a unique marked sign is not a proof + the turn that states both positions stays.

Bava Metzia 59b: after Sinai a Divine Voice is not regarded. Eruvin 13b: both these and those are the words of the living God, and the halakha follows the side that taught both. Exactly one top-5 turn is a spectacle. Another top-5 turn contains two polarities or contains a token from the spectacle turn plus a token the spectacle turn lacks. The spectacle cannot take slot 1. The two-sided turn is seated and the spectacle turn is kept in the pool. A sign that is the only marked turn does not beat the turn that states both positions. S3-015 seats any plain majority against a unique mark. This seats specifically the turn that carries both positions, and it keeps the marked turn, which is Eruvin's retention plus the heavenly-sign refusal.

Math: If spectacle count in the top 5 is 1 and some i is both-polarity or cites the spectacle's token, seat i and do not drop the spectacle.

```rust
fn s3_089_sign_loses(unique_spectacle: bool, other_is_two_sided: bool) -> bool {
    unique_spectacle && other_is_two_sided
}
```

Record: https://www.sefaria.org/api/texts/Bava_Metzia.59b?context=0&commentary=0 | https://www.sefaria.org/api/texts/Eruvin.13b?context=0&commentary=0 via Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 6272-6273. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

### S3-090 completion_only_inside_the_majlis

OpenITI Majalis Ibn al-Qasim plus Mishneh Torah Sales 1, 2nd-century AH Maliki sessions, medieval restatement. Tag RECALL/INDEX. Targets BEAM preference_following; BEAM instruction_following. Miss pattern: PF no_pool 9; IF no_pool 5. Combined: a completion counts only in the same session + a short offer is not itself operative.

The fetched OpenITI text identifies itself as Majalis Ibn al-Qasim, the sessions in which Ibn al-Qasim asked Malik, BookURI 0191.IbnQasimMisri.MajalisIbnQasim. Sales 1: words alone do not acquire. The offer is a short or question turn with one content token. The completion has to be the next turn and has to share the offer's session id. A turn in another session, however similar, is a different majlis. The content token copies onto the next turn only inside the session, and only when that next turn is a digit, a quote, or a short assent. The offer is non-operative. Cross-session assent does not complete it and does not receive the token. S3-010 imports a foreign directive when the session has no cover. This rule does the opposite for acquisitions: a completion outside the session does not count and is not imported. Step 23's cross-turn token copy is not bounded by a completion medium; this one is.

Math: copy(c, u) = 1 iff u = next(t) AND session(u) = session(t) AND medium(u). operative(t) = 0 whenever offer(t).

```rust
fn s3_090_majlis(same_session: bool, is_next: bool, medium: bool) -> bool {
    same_session && is_next && medium
}
```

Record: https://raw.githubusercontent.com/OpenITI/0200AH/master/data/0191IbnQasimMisri/0191IbnQasimMisri.MajalisIbnQasim/0191IbnQasimMisri.MajalisIbnQasim.Shamela0096602-ara1 | https://www.sefaria.org/api/texts/Mishneh_Torah%2C_Sales.1?context=0&commentary=0 via OpenITI GitHub raw + Sefaria. Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3144. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.

