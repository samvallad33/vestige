# Shard 1 — Mesopotamian dossier rules

Research and design only. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629` (Step 33 Triangulation). No engine edit ships with this shard. The target is 90% on every metric of LME, LoCoMo, and BEAM, and on every category. The per-row ceilings are in `funnel-S1.md`.

Every rule is integer or ordinal arithmetic on columns filled at ingest. Ranking reads those columns, `TurnIx.content`, `TurnIx.feats`, `TurnIx.seq`, `TurnIx.entities`, `TurnIx.form_digit`, and `TurnIx.session`. It does not read `ix.turns[ti].body`. New maps are `DetHashMap` / `DetHashSet`. No rule branches on the speaker string, edits `HYPERNYM_LATTICE`, calls `rsb_seat_behind`, or touches `rungs.rs` or `benchmarks/real_engine/`. Closed-class words and orthographic shape are the only lexical tests. No scalar already in the ranker (0.15, 0.68, `max_per_session_top5 = 1`) is retuned; a new predicate is a new branch.

LoCoMo Turn@1 cannot reach 1383 by reordering inside the current top 5 (ceiling 1127) or inside k (ceiling 1226). BEAM Turn@1, Turn@5, Turn@10, All@5, All@10, and every Session metric sit under their 90% lines even after a perfect reorder of the current lists. LME Turn@1 can, because Turn@5 is already 428. LME Turn All@5 cannot, because a slot-1 swap does not pull the other golds. The build order below is the path through those gaps. Ceilings are upper bounds.

## Top 10 build order

1. **S1-071 A name list that is a subset of a longer turn yields** — LoCoMo Single-Hop Turn@1; LME Turn@1. A short name list that is a strict subset yields slot 1. Single-Hop t5_not1 is 192, the largest piece of the LoCoMo Turn@1 gap of 578. LME Turn@1's entire gap sits inside Turn@5. `ORDERING`.

2. **S1-073 Different year-depth admits another turn from the same session** — BEAM All@5; LME Turn All@5; event_ordering. Admit another same-session turn when its year-depth differs. The cap literals stay. This is the All@5 lever on summarization, event ordering, multi-session, and instruction following. `RECALL/INDEX`.

3. **S1-056 The year on the basket reverse is the file's session key** — LoCoMo Session@1; BEAM Session@1/@5/@10. The year on a basket's reverse is the session key. Admit a session with s_score 0, then put the member session at slot 0. Multi-Hop and Open-Domain Session@5 are short of 90%. Five BEAM categories are short even at Session@10. `RECALL/INDEX`.

4. **S1-055 A deficit basket pulls deficit turns and not the tag** — LoCoMo Open-Domain; Multi-Hop no_pool. A deficit basket scores 0 and copies its tokens onto the deficit turns. Open-Domain in-pool is 78 and the need is 83, so at least five no_pool golds have to be created by ingest. `RECALL/INDEX`.

5. **S1-062 A gap does not join a condition to the wrong consequent** — Open-Domain and instruction-following no_pool. An if inherits its consequent only when no gap sits between them. Instruction-following in-pool is 35 and the need is 36. `RECALL/INDEX`.

6. **S1-049 A periodic share is the standing obligation** — BEAM preference_following; LME single-session-preference. The standing fact is the periodic share, inherited on the heir id. Preference in-pool is 30 and the need is 36. LME preference Turn@10 is 19 and the need is 27. `RECALL/INDEX`.

7. **S1-002 Deficit is capital minus expenditure** — LoCoMo Single-Hop and Multi-Hop pool. The answer integer is capital minus expenditure. Single-Hop k is 737 and the need is 757, so twenty pool_not_k golds have to enter. `RECALL/INDEX`.

8. **S1-042 The deficit is legal only across a blank** — LoCoMo Single-Hop and Multi-Hop. That subtraction is legal only across a blank. Without the cut the residual mixes the two sides. `RECALL/INDEX`.

9. **S1-060 A row's amount is its last integer and the row is one fact** — BEAM information_extraction. The row's amount is its last integer, inherited onto the row's other tokens. Extraction in-pool is 32 and the need is 36. `RECALL/INDEX`.

10. **S1-090 A later integer equal to earlier plus a stored period is the same series** — BEAM event_ordering All@5; temporal. Admit the later turn when its integer equals an earlier integer plus a period stored on a third turn. No planet table. event_ordering All@5 is 0/40. `RECALL/INDEX`.

## Why this order

LoCoMo Turn@1 needs +578 from the Step 34 floor of 805. Single-Hop is 263 of that at the bd2693e2 category split, Multi-Hop 157, Temporal 102, Open-Domain 65. S1-071 spends the 192 Single-Hop misses already in the top 5. S1-002 and S1-042 are the pool admission for the twenty Single-Hop golds that sit outside k, and for Multi-Hop's 83 pool_not_k. S1-055 and S1-062 are the ingest that Open-Domain and instruction-following require, because their in-pool counts are under the 90% line. S1-073 and S1-090 are All@5: BEAM All@5 is 118 against a need of 320, and LME Turn All@5 is 314 against a need of 423, with multi-session at 53/121. S1-056 is Session@1: LoCoMo is short 250, and the bench Session@5 of 1416 hides Multi-Hop 246/254 and Open-Domain 65/83. BEAM Session@10 is 275 against a need of 320, and five categories are short even there. S1-049 is preference, the category whose in-pool count is 30 against a need of 36. S1-060 is extraction, in-pool 32 against a need of 36.

## The ninety rules

### S1-001 Dossier holds many dated events

AAS tablet Aegyptus 08, 264 11 (CDLI 100212, Ur III Umma) is one balanced account: a capital of worker-days, a blank, then several expenditure lines each with its own giri3 conveyor, a szunigin total, and a la2-ia3 deficit. One physical file contains many events. The ranker treats a chat session as one event: when is_event_sequence is set, max_per_session_top5 is the literal 1 (retrieval.rs:5619-5625) and a later turn of that session is dropped at the session_capped continue (5676-5679). This rule adds a predicate at that continue. A candidate whose terminal integer differs from every already selected turn of the same session is not capped. The literal 1 is not edited, so two turns that share the terminal integer still collapse. No speaker test.

Math: `exempt(ti) = is_event_sequence AND terminal(ti) != 0 AND forall sj in selected, same session: terminal(sj) != terminal(ti); session_capped fires only when count >= 1 AND NOT exempt`

Tag `RECALL/INDEX` on BEAM event_ordering (pool_not_k 17; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:5676-5679`. Record: https://cdli.earth/artifacts/100212.json.

Dedupe: Petrie seriation (turn1 #25) re-ranks sessions to maximize session-timestamp span and reads session dates. This never reads session_ts. Florilegium (turn1 #26) and the shipped cap keep one turn per session; this is an exemption, not a new quota. Catch-lines (turn1 #22) link the tail of session s to the head of a later session. This stays inside one session.

### S1-002 Deficit is capital minus expenditure

AAS 013 (CDLI 100001) closes with si-i3-tum nig2-ka9-ak, the remainder of the balanced account, and names a year. Aegyptus 08, 264 11 (CDLI 100212) labels a la2-ia3 line after the expenditures. The remainder is an integer relationship between two sides, not a third commodity the query has to share tokens with. At ingest, while feats are built (3125-3272), store each already-parsed integer in the section it sits in. The deficit integer is emitted as an inheritable feat equal to the decimal string of |sum(section 0) - sum(section 1)|. A query whose only numeral is that deficit hits the turn even when the turn's prose never repeats the query's other nouns. Ranking reads the integer columns, not the body. This does not claim that the broken arithmetic on 100212 balances; the labels are what was read.

Math: `deficit = sum(integers in section 0) - sum(integers in section 1); emit feat decimal(deficit) inheritable`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (pool_not_k 83; no_pool 13). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100001.json.

Dedupe: Step 32 trial balance requires every query group to be credited in the turn or its uptake. This is integer subtraction, not group coverage. Stocktaking residual (turn1 #23) down-weights query terms already satisfied. Khipu summation (turn1 #24) maximizes distinct (noun, numeral) pairs and does not touch slot 0. Sexagesimal composition (ancient #25) parses place value; these integers are already parsed. Primary-aggregate (ancient #32) prefers a turn with fewer digits; this emits a new digit.

### S1-003 Blank line splits capital from outlay

AAS 065 (CDLI 100053) puts the silver capital, then the line $ blank space, then sza3-bi-ta, then the commodity lines. Co-occurrence that crosses that blank joins the capital to an expenditure it does not belong to, and a multi-hop query then retrieves the wrong side. At ingest a blank line increments the section index. Feats of section 0 are not inheritable onto tokens of section 1, and a content-token pair is not stored when the two tokens have different section indexes. Petuchah-style best-section coverage is a different operation and is not what this does: coverage is not replaced by the best section's coverage. The two sections stay two bags.

Math: `cooccur(a, b) is stored iff section(a) == section(b); blank line increments section`

Tag `RECALL/INDEX` on LOC Multi-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100053.json.

Dedupe: Petuchah/setumah (ancient #38) sets the turn's coverage to its best section. This refuses cross-section feats and does not take a max. Turin lacuna (ancient #11) drops df=0 query groups. Abusir header inheritance (ancient #16) copies a header onto cells.

### S1-004 Price is the next line's integer

On AAS 065 each commodity line is followed by a ku3-bi line, its silver. The price is the integer on the immediately following line, not a nearby integer anywhere in the turn. A query that carries two integers matches the turn only when those two integers occupy adjacent lines inside the same section. The pair is stored as an ordered feat (quantity, price). A turn that contains both integers on non-adjacent lines does not receive the pair feat, so it loses to the turn that does when the query has both.

Math: `pair(q, p) iff line(p) == line(q)+1 AND section(q) == section(p); query with two integers requires the pair feat`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100053.json.

Dedupe: Ana approximation (ancient #41) matches a digit within plus or minus 10 percent. This is exact adjacency of two integers. Digit-unit split (ancient #24) splits 5km into 5 and km. No unit suffix is split here.

### S1-005 Slash fraction binds to the previous integer

AAS 065 writes 1/2(disz), 1/3(disz), and 5/6(disz) on the silver lines. The fraction is orthographic, a token matching digits, a slash, and digits. It composes with the integer on the same line or the previous line by exact rational arithmetic: value = whole + num/den, reduced in integers. The composed numerator and denominator are both emitted. A query integer matches the numerator of that reduced value, not a window around it. No fraction word list.

Math: `if token matches ^[0-9]+/[0-9]+$, value = prev_int * den + num, emit (value, den) in lowest terms`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Multi-Hop (no_pool 13; pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100053.json.

Dedupe: Ana ±10% (ancient #41) is a tolerance. Sexagesimal composition (ancient #25) folds place-value words. This is an exact N/M token already written with a slash. Number-word composition is not reused.

### S1-006 Worker-days are the product of two integers

AAS 013 and AAS 022 (CDLI 100010) open with a crew integer and a day integer on one line (AAS 022: 3(u) 4(disz) gurusz u4 1(disz)-sze3, 34 workers for 1 day). The labor total a query asks for is the product, which is often not written as its own token. At ingest, two integers on the same line emit their product as an inheritable feat. A query whose numeral equals that product matches the line. The factors stay available for a query that names one of them. Ranking never multiplies raw text; it reads the stored product.

Math: `if a line holds exactly two integers a, b and a later line does not already equal a*b, emit feat a*b`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Multi-Hop (no_pool 13; pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100001.json.

Dedupe: Khipu (turn1 #24) covers distinct pairs across sessions and leaves slot 0 alone. This emits one integer, the product, onto the turn that holds the factors. Digit-unit split is not involved. Step 32 is group balance, not multiplication.

### S1-007 Year-name clause is the event key

LB 0068 (CDLI 388630) ends with mu ki-masz{ki} ba-hul, a year-name, after the barley receipt. The event's time is that closing clause, not the session timestamp and not a month-name lookup. At ingest the token-id sequence of the last non-blank line is stored as year_key. The ordering seat, a new function after apply_c2_step33_triangulation (algos_c2.rs:2593) called at retrieval.rs:6265-6272, sorts event-sequence candidates by the first-seen ordinal of year_key in the corpus, then by turn index. session_ts is not read. The seat runs for user/assistant transcripts; it does not return on is_event_sequence.

Math: `order by first_seen[year_key] ascending, then turn index; first_seen is the minimum turn index in corpus order that carries that key`

Tag `ORDERING` on LOC Temporal; LME temporal-reasoning; BEAM temporal_reasoning (t5_not1 7; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/388630.json.

Dedupe: Astronomical-diary calendar match (turn1 #32) multiplies a session score when a month-name window contains the session timestamp. This does not read month names or session_ts. Regnal anchoring (turn1 #41) retrieves an anchor event and ranks by date distance. This sorts by the ordinal of a stored suffix, one pass. Incipit (turn1 #31) weights the first user turn as a title. This uses the last line.

### S1-008 Year-after depth is a repetition count

CUSAS 39, 153 (CDLI 250483) dates wages with mu us2-sa ki-masz{ki} ba-hul mu us2-sa-bi: the year after the year-after, depth 2, and the work is u4 1-kam. AAS 103 (CDLI 100090) has a single mu us2-sa. Depth is the count of a closed-class follower (after / following) in the date clause, not a regnal table. The seat orders by (base year_key ordinal, depth). Two turns with the same year formula and depths 0 and 2 sort in that order even when their session timestamps run the other way. The constant weights in the ranker are untouched.

Math: `depth = count of closed-class {after, following} in the terminal clause; sort key (first_seen[year_key], depth, turn index)`

Tag `ORDERING` on LOC Temporal; LME temporal-reasoning; BEAM event_ordering (t5_not1 7; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/250483.json.

Dedupe: Regnal event-anchored dating (turn1 #41) is a two-pass retrieval against session dates. Harris matrix is an edge topology. This is a repetition count of one closed-class word plus the year-key ordinal from S1-007's column. Torrens latest-wins is not used: a higher depth sorts later only relative to the same year key, and a different year key uses first-seen order, which can be earlier in the corpus.

### S1-009 A strict subset copy yields

AAS 103 (CDLI 100090) rolls a second seal and labels it gaba-ri-dam, it is a copy. A later turn whose content-token set is a strict subset of an earlier turn in the same session, and which adds no integer the longer turn lacks, loses slot 1 to the longer turn. The seat is the new function at retrieval.rs:6265-6272. It does not use Jaccard, and it does not compare user turns with assistant turns. Speaker strings are not read. If the short turn carries an integer the long turn does not, it is not a copy and it stays.

Math: `yield if content(short) strict-subset content(long) AND integers(short) subset integers(long) AND same session AND index(short) > index(long)`

Tag `ORDERING` on LOC Single-Hop; LME single-session-assistant; LME multi-session; BEAM summarization (t5_not1 61). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/100090.json.

Dedupe: Palimpsest diff (turn1) uses Jaccard plus a diff token. Asteriskos (ancient Hexapla) keeps a verbatim context mark. Source-superset realloc is a user/assistant seat already shipped. This is strict subset of content ids plus integer-set inclusion, speaker-agnostic, same session only.

### S1-010 A seal binds only the line above it

AfO 40-41, 55 3 (CDLI 100298) rolls a seal (kiszib3 ib2-ra) against one consignment, after that consignment's ki-la2-bi weight, not against the whole tablet. A line that is only a seal mark, detected as a line whose tokens are a closed-class seal shape or a line that is a single mid-turn caption after a blank, attaches its scope to the previous line's integer and entities. Those entities inherit the integer. Entities on earlier lines do not. A query naming the sealed party hits the sealed line's integer and misses the other lines' integers.

Math: `if line i is a seal line, bind entities(line i-1) to int(line i-1) and to no other line's int`

Tag `RECALL/INDEX` on LOC Multi-Hop; BEAM information_extraction (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100298.json.

Dedupe: Notarial attestation (turn1 #40) multiplies turns by a named speaker. This does not read the speaker field. Household binding (Liye, ancient list) attaches a possessive to the nearest entity. This attaches one line, the previous line, and only when the current line is a seal line. Abusir red-stroke status (ancient #50) emits done/todo from checklist glyphs.

### S1-011 The median name is the controller

WO 13, 08 (CDLI 217459) names a merchant, then the controller (lugal-gigir-e maszkim), then di-til-la. The controller is neither the first title-case token nor the last. BIN 08, 015 (CDLI 10012) puts maszkim-bi on the field that was given. When a clause has exactly three title-case tokens, the middle one is stored as the controller feat and the two endpoints are stored as parties. A query whose sole title-case token equals the middle one retrieves this turn by that feat. A query that names an endpoint does not receive the controller feat. No word list of offices.

Math: `if title_case_count(clause) == 3, controller = token[1], parties = {token[0], token[2]}`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (pool_not_k 83; t5_not1 61). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/217459.json.

Dedupe: Third-text arbitration votes a rare token from a third session. This is the median orthographic name inside one clause. Two-witness corroboration (turn1 #33) requires two attestations. Muster-roll (turn1 #50) fills slots 2-5 with distinct names. Hakirot (ancient #3) compares core-cue sets between turns.

### S1-012 The name in both the first and last clause prevails

ITT 3, 06575 (CDLI 111382) is a ditilla. One brother di-ta in-tak4, withdrew from the case; the name that stands in the opening and again in the disposition is the party who receives the field. The prevailing entity is the title-case token that occurs in clause 0 and in the last clause. A token that occurs only in the opening is the party who left, and it is not given the verdict feat. The query matches the verdict feat. This is not 'the later turn wins': both clauses are inside one turn, and a name that appears only at the end does not prevail unless it also opened.

Math: `prevail = title_case(clause_0) intersect title_case(clause_last); withdraw = title_case(clause_0) - prevail`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop; LME multi-session (t5_not1 61; pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/111382.json.

Dedupe: Torrens supersession seats the latest record. This requires membership in both endpoints of one turn. Damnatio and sanis-menu are different seats already rejected or shipped. Step 33 triangulation requires two score sightlines and a term cover; this is a set intersection of two clauses.

### S1-013 Names between the oath and the date are not events

The Nippur slave sale ArOr 07, 8 1 (CDLI 101851) states the price, the sale, nu-gi4-gi4-da, the oath mu lugal-bi i3-pa3, then lu2-inim-ma-bi-me and the witness names, and only then the month and the year. The witness block is not an event sequence. Tokens that sit after the last content clause and before the terminal integer are marked witness_zone at ingest. The ordering seat ignores witness_zone tokens when it decides which turn states the event. A turn that is only a witness list, no terminal integer outside that zone, cannot take slot 1 over a turn that has a terminal integer.

Math: `witness_zone = tokens after the last clause boundary and before the terminal integer; event_key ignores that span`

Tag `ORDERING` on BEAM event_ordering; LOC Single-Hop (t5_not1 7). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/101851.json.

Dedupe: Method of loci (turn1 #8) walks the k-th bullet. This deletes a zone from the order; it does not pick a k-th item. Profession-list status is a separate rule (S1-034) for a digit-free list that has no date zone at all. Two-witness corroboration is not applied.

### S1-014 The terminal integer is the date, earlier integers are quantities

On the sale (CDLI 101851) the price integer is medial and the year stands after the witnesses. On LB 0068 (CDLI 388630) the year-name is the last line. When a turn has two or more integers, the one in the last eight tokens is the date key and every earlier integer is a quantity. The ordering seat sorts by the date key only. A quantity that happens to be numerically larger does not sort later. Turns with a single integer use that integer as the date key only when it sits in the last eight tokens; otherwise the turn has no date key and loses slot 1 to a turn that has one.

Math: `date_key = last integer whose token offset is within 8 of the end, if any earlier integer exists or this is the only integer in that window; else none`

Tag `ORDERING` on BEAM event_ordering; LOC Temporal; LME temporal-reasoning (t5_not1 7; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/388630.json.

Dedupe: Loci picks the k-th list item. Inline date tokens (retrieval.rs:1452-1619) parse ISO and month names; this seat does not call that function and does not add month names. Diary calendar match (turn1 #32) uses session timestamps. This sorts an integer already on the turn.

### S1-015 An oath clause adds no event

The sale's oath, mu lugal-bi i3-pa3, is a clause whose only content token is one title-case name. It is the swearer, not a thing that happened in the order. A clause with exactly one title-case token and no integer is marked oath_zone. The ordering seat does not let an oath_zone clause supply slot 1, and its token does not count as a distinct event inside the turn. Clauses that have an integer or two title-case tokens stay.

Math: `oath(clause) iff title_case_count == 1 AND integer_count == 0; event clauses are the complement`

Tag `ORDERING` on BEAM event_ordering; LOC Single-Hop (t5_not1 7). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/101851.json.

Dedupe: Epistolary formula-length exclusion drops a high-df first or last sentence from BM25 length. This zeroes a clause by an orthographic count (one title-case, no integer), not by document frequency, and only for the event order. Ezib (ancient #18, same tradition, different tablet family) zeroes excluded query tokens under except/besides. This is not an exclusion cue.

### S1-016 A doubled verb under negation is not a second event

The sale writes nu-gi4-gi4-da, a negated reduplicated verb: it shall not be returned. The reduplication is one legal act's refusal, not two returnings. A clause whose content tokens are a negation plus a token that is the same stem twice (orthographic doubling, or a hyphenated repeat) contributes zero events. The sale itself remains the event. The seat drops that clause before ordering.

Math: `if clause has a closed-class negation and a token equal to its own double, event_count(clause) = 0`

Tag `ORDERING` on BEAM event_ordering; BEAM knowledge_update (t5_not1 7). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/101851.json.

Dedupe: Dittography (Step 26) collapses two identical turns. This is one clause inside a turn. Compound-verb unit (ancient #10, šu ba-ti) emits a particle feat and halves the bare verb. This does not look at a particle class and does not halve a weight; the clause contributes no event.

### S1-017 The completion clause holds the price

The orchard sale YBC 05336 (CDLI 306421) states sa10 til-la-ni-sze3, the price in full, in-na-an-la2, he paid, then the no-contest clause and the witnesses. The price integer is the integer in the clause that also contains a closed-class completion word (full, complete, paid). Other integers in the turn are not the price. The completion-clause integer is the feat a 'how much' query matches. The English closed class is the completion set already appropriate to closed-class grammar (full, complete, paid), not a list of commodities.

Math: `price = integer in the unique clause that contains a token from {full, complete, paid}; if two such clauses, no price feat`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/306421.json.

Dedupe: Primary-aggregate prefers the turn with a single digit over a summary. This picks a clause inside one turn by a closed-class word. WH-slot voter is not used; there is no who/what branch. Kelal u-ferat is a general-then-specific legal reading and is not this.

### S1-018 Neighbors count only in order

The same sale identifies the orchard by da X, da Y, u3 da Z: an ordered chain of neighbors, not a bag of names. Two query entities match the turn only when they occur in that order, with at most one token between them, and a closed-class coordinator (and) if a third name sits in the chain. Unordered co-occurrence is not enough, so a turn that mentions both people in a witness list does not take the location. The chain is stored as an ordered triple of entity ids.

Math: `match iff exists i: entity[i]==q1 AND entity[i+1]==q2 AND offset(i+1)-offset(i) <= 2`

Tag `RECALL/INDEX` on LOC Multi-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/306421.json.

Dedupe: Hollerith (Step 30) indexes an unordered co-occurrence pair. Cartouche (ancient list) treats a consecutive title-case span as one name. This requires order and a gap of at most one token, and it stores a pair of already-separate entities. Khipu two-hop (turn1 #42) pulls a bridge token into a second session.

### S1-019 An every-period integer is a cycle, not a date

PBS 08/2, 116 (CDLI 262151) has four brothers give their nadītu sister a ration mu-am3, every year: barley, oil, and wool, as a periodic annuity rather than one transfer. A digit tied to a closed-class every/each plus a time-unit token is a period, not a point date. The ordering seat places that turn once per distinct clause that carries the period, in clause order, and does not also sort it by session_ts. A one-shot turn with a terminal date still sorts by its date key. The annuity does not compete with dated events as if it were a single day.

Math: `if clause contains {every, each} and an integer, period_ord = clause_index; it does not enter the date_key order`

Tag `ORDERING` on BEAM preference_following; LME single-session-preference (k_not5 8; t5_not1 7). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/262151.json.

Dedupe: Diary calendar match multiplies by a month-name window. This never reads a month name. Loci walks list position as the answer. Here the clause index only separates the annuity from the dated order; it is not returned as the answer to a which-item query.

### S1-020 One integer divided by the conjunct count

The same contract names four brothers, ibila-me-esz, who together owe the ration. A query that asks what one of them gives is the integer divided by the number of coordinated title-case names, integer division. At ingest, a clause with one integer and two or more title-case names joined by a closed-class and emits share = integer / name_count, and emits that share as a feat on each name. A query naming one brother and the share integer hits the feat. The undivided integer remains the feat for a query that does not name a single conjunct.

Math: `if names >= 2 AND one integer n AND a closed-class and, share = n / names; emit share on each name`

Tag `RECALL/INDEX` on LOC Multi-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/262151.json.

Dedupe: Coordinated-claimant distribution (ancient list) emits a relation feat for both conjuncts. This emits the quotient. Double-entry posting dedupe (turn1 #49) collapses turns that share a (numeral, noun) pair. This does not collapse turns. Khipu summation does not divide.

### S1-021 Two conjunct parties share the integer

YOS 08, 096 (CDLI 290103) is a partnership loan: silver nam-tab-ba-sze3, received, and when the merchant asks, the silver and its profit are paid. The principal is one integer in a clause that has two title-case names. Both names receive the integer as a shared feat. A query that names only one partner still matches the full integer, because the debt is joint. The profit clause is a later clause and is not given this feat unless it has its own integer. šu ba-ti is not used as a key; the compound-verb rule already owns that.

Math: `if a clause has exactly one integer and exactly two title-case names, both names get feat int; a following clause does not`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/290103.json.

Dedupe: Ancient #10 compound-verb emits pv:v:p for šu ba-ti. This rule never looks at a verb particle. Claimant distribution emits a relation, not a shared integer. Notarial scope multiplies by speaker; speaker is unread.

### S1-022 Half-interest is the principal over two

Hermitage 3, 073 (CDLI 211416) states sze masz2 szu-ri-a-bi, barley whose interest is one half. The half is exact. When a line holds an integer and a slash-fraction 1/2, or the closed-class token half, the interest integer emitted is principal/2. A query numeral equal to that interest matches. The match is exact integer division, not a percentage band. If the principal is odd, the remainder is dropped (integer division) and a second feat records the remainder, so a query for the remainder can still hit.

Math: `interest = principal / 2; remainder = principal % 2; both emitted`

Tag `RECALL/INDEX` on LOC Single-Hop (no_pool 13). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/211416.json.

Dedupe: Ana approximation matches within 10 percent. This is exact division by 2. Fraction binding (S1-005) composes a slash onto the previous integer as a quantity. This fires only when the fraction is 1/2 and stores the quotient as interest, a separate column from the composed quantity.

### S1-023 The first integer is the price and later integers are extras

The Fara field sale ArOr 39, 015 02 (CDLI 10009) writes the price of the field, then nig2-diri the surplus, then nig2-ba the gift. Three integers, three roles, fixed by position: the first is the price, the later ones are extras. For a query that contains one integer, the seat prefers the turn whose first integer equals the query integer over a turn whose only match is a later integer. The later integers remain retrievable for a query that contains two integers (see the combination S1-051). No commodity list.

Math: `role(int_0) = price; role(int_k) = extra for k>0; slot 1 prefers price-equal over extra-equal when the query has one integer`

Tag `ORDERING` on LOC Single-Hop; LOC Open-Domain (t5_not1 61). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/10009.json.

Dedupe: Primary-aggregate compares a summary shape (>=3 digits) with a single-transaction shape and prefers the single. This does not compare two turns by digit count. It assigns roles by position inside one turn. The seat fires only when the query has one integer.

### S1-024 The first integer is a count and the possessive integer is the measure

TMH 05, 151 (CDLI 20565) counts copper spears and then gives ki-la2-be6 41 ma-na, their weight. Two integers, two roles, told apart by position and by a possessive-shaped token (a token ending in a possessive clitic, or the closed-class word its/their) immediately before the second integer. The first integer is the count feat. The integer after the possessive is the measure feat. A query with one integer matches the count. A query with two integers matches only a turn that has both feats. The unit sign is not split off the number; digit-unit split already owns that.

Math: `count = first integer; measure = integer immediately after a possessive-shaped token; both required when the query has two integers`

Tag `RECALL/INDEX` on BEAM information_extraction; LOC Single-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/20565.json.

Dedupe: Ur III digit-unit split (ancient #24) rewrites 5km as 5 and km inside raw_tokens. This does not touch raw_tokens. It labels two integers that are already separate. ki-X-ta source/goal binding is a different ancient rule and is not reused.

### S1-025 The opening remainder is the prior period's input

Aleppo 001 (CDLI 100333, Puzriš-Dagan) counts the animals, then writes si-i3-tum, the remainder, then the year-name mu us2-sa an-sza-an{ki} ba-hul, then a blank. The remainder is the balance brought into this account, so it is ordered before the year's own expenditures, not as a new event at the year-name's date. The seat gives a turn's opening-remainder flag ordinal year_depth-1 relative to that turn's year key, and gives the expenditure lines ordinal year_depth. The opening block therefore sorts first.

Math: `if the first section is marked remainder, its ordinal = year_depth - 1; later sections use year_depth`

Tag `ORDERING` on LOC Temporal; BEAM event_ordering (t5_not1 7; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/100333.json.

Dedupe: Step 32 trial balance is query-group coverage. This only changes order. Torrens would put the latest section first. This puts the carried remainder first, which is the earlier period. Stocktaking residual down-weights query terms.

### S1-026 A year-name scopes every line of the account

AAS 013 names one year on the balanced account, and the worker-day lines belong to that year. A line without its own year key inherits the turn's single year key. Two turns whose place token matches and whose year keys differ are two events and are not collapsed, even when the crew integer is the same. The seat's sort key includes the year key, so the same field in two years stays two rows. Dittography's exact-duplicate collapse is not widened to cover this.

Math: `if the turn has exactly one year_key, every line inherits it; sort and identity include year_key, so equal integers with different keys stay distinct`

Tag `ORDERING` on LOC Temporal; LME temporal-reasoning (k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/100001.json.

Dedupe: Dittography (Step 26) collapses exact duplicate turns. Double-entry posting dedupe (turn1 #49) collapses a shared (numeral, head noun) inside the top 10. This refuses that collapse when year_key differs. Diary match is not used.

### S1-027 A basket tag points at a class and does not answer

AAS 132 (CDLI 100116) is an object tag, five lines: pisan-dub-ba, kiszib3 la2-ia3 gi, the sealed documents of reed deficits, located in the street, ki bi2-du11-ga-ta, from Biduga. The tag describes a class of other tablets. It is not the account. A turn with at most six lines, a short first line, and no section break is a label. Its own score for slot 1 is forced to zero by storing label=1. Each content token on the label is emitted as a class feat that other turns can inherit when they carry a deficit integer (S1-002's column) and the same entity. The gold account, which may not repeat the basket's wording, becomes retrievable because the class feat was written onto it. The label turn itself cannot take slot 1.

Math: `label iff line_count <= 6 AND no blank section AND first line has <= 3 tokens; slot score 0; class feats copied onto turns that share an entity and have a deficit column`

Tag `RECALL/INDEX` on LOC Open-Domain; LOC Multi-Hop (pool_not_k 83; no_pool 13). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100116.json.

Dedupe: Catch-lines (turn1 #22) link the last two user turns of one session to the first two of the next. This label does not look at session boundaries. Finding-aid smoothing (turn1 #48) blends turn, window, and session scores. This zeroes the label and copies a class feat. Incipit weights the first line; this refuses to seat it.

### S1-028 A hyphenated token after from is the place

Kaskal 15, 39 (CDLI 247781) issues goods e2-kiszib-ba-ta, out of the sealed storeroom, and then names the person who received them. The place is the hyphenated token that follows a closed-class from. The following title-case name is the person. A query whose rare token is the place matches the hyphenated token. A query whose rare token is a personal name does not match the hyphenated token. The hyphen is orthographic; no place-name gazetteer is consulted at rank time. Pleiades is a source for the design, not a runtime list.

Math: `place = the token containing '-' that immediately follows closed-class from; person = the next title-case token; the two feats are distinct`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Open-Domain (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/247781.json.

Dedupe: ki-X-ta source/goal (ancient list) binds a source and a goal as a relation on Sumerian case. This only classifies a hyphenated token after the English closed-class from, and it does not emit a goal. Scriptio continua join (ancient #43) concatenates adjacent tokens when the join exists in the vocab. This does not concatenate. The hyphen is already inside one token.

### S1-029 Of two names, the earlier is the beneficiary

BIN 08, 015 (CDLI 10012) gives a field (i3-na-ba) and then names maszkim-bi, the controller. With exactly two title-case names and one integer, the earlier name is the beneficiary and the later is the controller. A query that contains the earlier name and the integer seats this turn above a turn that contains the later name and the same integer. The seat is ordinal: beneficiary-match outranks controller-match. It does not use a who-word.

Math: `if caps == 2 AND ints == 1: beneficiary = caps[0], controller = caps[1]; rank beneficiary-match above controller-match`

Tag `ORDERING` on LOC Single-Hop (t5_not1 61). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/10012.json.

Dedupe: WH-slot voter types the question word. This ignores the question word and uses name position. Median-controller (S1-011) needs three names. This needs two. Notarial speaker scope is not used.

### S1-030 An explicit course ordinal orders repeated work

RA 012, 164 (CDLI 127648) records brick courses: length, width, and height a-ra2 1-kam, then a-ra2 2-kam, then the brick volume, and kiszib3-bi 2-am3, its sealed documents are two. The course number is an integer written next to a repeated stem, not the position of a bullet. The seat orders clauses by that adjacent integer. Line position is the tie break only. A list of names with no such integer is not ordered by this rule.

Math: `course(clause) = the integer adjacent to a stem that also occurs in another clause; sort by course, then clause index`

Tag `ORDERING` on BEAM event_ordering; LOC Temporal (t5_not1 7; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/127648.json.

Dedupe: Loci (turn1 #8) selects the k-th bullet of an assistant list. This reads an integer that is already next to a repeated stem, and it orders rather than picking one bullet as the answer. Step 9B list ordinals are the loci family and are not retuned.

### S1-031 A repeated integer is a check, not a second event

MCT 036 (CDLI 254518, CBS 03551) writes 7.36 and then 7.36 ib2-si8. The second copy of the same integer is the check line of a square, not a second event. The editorial note on that tablet even doubts a later product, so this rule does not require the square to equal the line. It only says: an integer that repeats the immediately previous integer, with no new entity between them, is one event. The seat drops the repeat from the event list. The Met object 322022 is the fragment of a mathematical problem text (accession 86.11.404, the same number as CDLI 412248, which has no transliteration). It is a second witness that a problem text exists as an object; the rule's arithmetic is taken from 254518, where the repeated line was actually read.

Math: `if int[i] == int[i-1] AND no entity token between them, event_count drops int[i]`

Tag `ORDERING` on LOC Single-Hop; BEAM summarization (k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/254518.json.

Dedupe: Dittography collapses two identical turns across the corpus. This is two adjacent integers inside one turn. Primary-aggregate prefers fewer digits globally. This removes only an immediate repeat. Sexagesimal parsing is not performed; 7.36 is whatever integer the existing digit parser already stored.

### S1-032 The same row across columns is one fact

CUSAS 21, 038 (CDLI 411619) is a proto-cuneiform mathematical tablet whose numbers sit in subcolumns 1.a, 1.b, 1.c, 1.d. Cells in one row are fields of one record. Cells in the next row are a different record. At ingest, a run of numeric lines that share a row index (the line number before the column letter, when the text has a column mark, or the same line index across a detected table) stores one fact id. A query integer matches only inside that fact. It does not match the sum of the row unless a later total line equals the sum, which is a different rule. No header text is copied onto the cells.

Math: `fact_id = row_index; a query integer matches a cell only if that cell's fact_id is the one the other query terms also hit`

Tag `RECALL/INDEX` on BEAM information_extraction; LOC Multi-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/411619.json.

Dedupe: Abusir header inheritance (ancient #16) copies the header line onto each cell as a field-2 feat. This does not copy headers. It binds by row index. Hollerith indexes an unordered pair. This binds a whole row.

### S1-033 The last integer on a line is the amount

MDP 06, 316 (CDLI 8105) is a proto-Elamite account: each entry is a sign cluster and a trailing number. The amount is the last integer on the line. The tokens before it are the account key. A query integer matches the amount feat, not an integer that happens to sit inside the key. When a line has a single integer, that integer is the amount. This is the line-final convention, independent of the row-binding rule, which needs more than one column.

Math: `amount(line) = last integer on the line; key = the tokens before it; query integer must equal amount`

Tag `RECALL/INDEX` on BEAM information_extraction; LOC Single-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/8105.json.

Dedupe: S1-032 binds columns of one row. This labels the last integer of a single line and does not require a second column. Digit-unit split rewrites the token. The integer here is already a token.

### S1-034 A digit-free name list is not a chronology

The Abu Salabikh profession list (CDLI 10101) runs dub-sar, umbisag, szagia, szabra, ensi2, szagina, and the rest in a fixed status order. SF 001 (CDLI 10566) is a god list in a fixed order, an, enlil, inanna, enki. Neither order is time. A turn with at least four title-case tokens and no integer is marked non_chronological. The event-ordering seat cannot give it slot 1 while any selected turn has a date key. The list's internal order is not promoted to an event order.

Math: `non_chronological iff title_case >= 4 AND integers == 0; such a turn loses slot 1 to any turn with a date_key`

Tag `ORDERING` on LOC Single-Hop; LOC Open-Domain; LME single-session-assistant (t5_not1 7). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/10101.json.

Dedupe: Witness-zone exclusion (S1-013) requires a date integer later in the same turn. This fires when there is no integer at all. Incipit and session topic heads weight a first line. This does not weight the list; it refuses to read it as time. Loci would treat the list position as the answer.

### S1-035 A gap mark cuts co-occurrence

RIME 3/2.04.06.01 (CDLI 226628) opens he-pi [...], a break, and then continues the royal inscription. Tokens on opposite sides of a gap mark must not become a co-occurrence, or a multi-hop query joins words the tablet does not join. A gap mark is orthographic: a line that is only brackets, ellipsis, or a token consisting of he-pi when the surrounding tokens are bracket-shaped. The practical test in English text is a line whose tokens are only '.' or a token that contains '[' or ']'. Feats do not cross it. The broken side is dropped from sums. The intact side remains.

Math: `if a line matches ^[\[\].…]+$ or a token contains '[' or ']', section increments and the broken side's integers are excluded from sums`

Tag `RECALL/INDEX` on LOC Open-Domain; LOC Multi-Hop (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/226628.json.

Dedupe: Turin lacuna (ancient #11) drops a query group whose tokens have df=0. This cuts co-occurrence inside one turn at an orthographic gap, whether or not the tokens are rare. Petuchah uses a blank line as a section for best-section coverage. A gap is not a blank, and this rule does not take a max coverage.

### S1-036 A repeated line-initial token makes one case per line

TBP 44 (CDLI 236901) is a Neo-Assyrian omen text. The lines that were read open DISZ, one protasis per line. When every content line of a turn begins with the same token, each line is its own case. The turn's match is the maximum of the line matches, not the sum, so extra omen lines do not inflate the turn over a turn that states the one matching case. The case's tokens are the only ones that inherit onto the answer feat.

Math: `if all content lines share line_initial token, score(turn) = max over lines of score(line), not the sum`

Tag `RECALL/INDEX` on LOC Open-Domain; BEAM instruction_following (pool_not_k 83; k_not5 28). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/236901.json.

Dedupe: Petuchah takes the best blank-delimited section's coverage. This fires on a repeated line-initial token, not on a blank, and the unit is the line. Responsa (Step 23) pairs a question turn with a later answer turn across speakers. This split is inside one turn. Speaker is unread.

### S1-037 An if-clause indexes its own consequent

The Code of Hammurabi stele (CDLI 249253, Louvre Sb 00008) writes laws as szum-ma condition, then the consequence. Law 1, which was read: if a man accuses a man and charges him with murder but does not prove it, his accuser is killed. The condition tokens and the consequent tokens are in one turn. At ingest the turn is split on the closed-class word if. Content tokens before the split inherit the content ids of the consequent clause, so a query that states only the condition retrieves the turn that states the consequence. The consequent does not inherit the condition, so a query about the penalty does not drag in unrelated conditions from the same turn. This is the intra-turn split. It is not a cross-turn question-to-answer link.

Math: `split at closed-class if; every condition token gains the consequent's content ids as inheritable feats; consequent tokens do not gain condition ids`

Tag `RECALL/INDEX` on LOC Open-Domain; BEAM instruction_following; LME knowledge-update (no_pool 13; pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/249253.json.

Dedupe: Responsa indexes a later answer turn from a question turn and uses speaker roles. This never reads speaker and never crosses turns. Kelal u-ferat and ars dictaminis are clause-role schemes already listed. WH-slot voter classifies the question word. Oracle-bone verification (ancient #1) demotes a hedged clause in favor of a later declarative turn. This copies ids inside one turn and does not demote a hedge.

### S1-038 A year-formula tablet supplies the ordinal of a shared suffix

LB 0800 (CDLI 368691) is a year-name tablet: one long formula naming the building of walls, and nothing else. Other tablets end with a shorter copy of such a formula (LB 0068 ends mu ki-masz ba-hul). The formula tablet is a dictionary. Its clauses, in order, define ordinals. A turn whose terminal token sequence is a superstring-or-equal of one of those clauses receives that ordinal as its year_ord at ingest. Event ordering then sorts by year_ord. The formula tablet itself is a label (line count small relative to a single clause) and does not take slot 1. session_ts is not the key.

Math: `year_ord(turn) = index of the unique formula-tablet clause that is a suffix of turn's terminal sequence; missing or ambiguous ordinal sorts last`

Tag `RECALL/INDEX` on LOC Temporal; LME temporal-reasoning; BEAM temporal_reasoning (pool_not_k 17; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/368691.json.

Dedupe: Incipit weights the first user turn of a session as a BM25F field. This uses a separate label turn as an ordinal dictionary for suffixes of other turns. Catch-lines require overlap between the tail of one session and the head of the next. Regnal anchoring does a second retrieval against session dates. None of those is a suffix-to-ordinal map.

### S1-039 Release is the year after a stated term

On the Hammurabi stele the clause that was read before law 118 says they serve for three years (mu 3(disz)-kam) and in the fourth year their release is established (ina re-bu-tim sza-at-tim an-du-ra-ar-szu-nu isz-sza-ak-ka-an). The release is not simultaneous with the pledge. Given a term integer N in one clause and an ordinal in the next clause, the release ordinal is N+1. The seat orders the pledge clause at offset 0 and the release clause at offset N+1. A later law on the same stele (law 118, read immediately after) says a merchant's sale of a pledged slave is not reclaimed. That clause has no term integer, so it does not receive a release offset. The rule fires only when both a term integer and a following ordinal exist.

Math: `if clause i holds integer N and clause i+1 holds an ordinal, release_offset = N+1; pledge_offset = 0`

Tag `ORDERING` on LOC Temporal; LME temporal-reasoning (t5_not1 7; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/249253.json.

Dedupe: Torrens seats the latest document. This spaces two clauses by N+1 inside one text. Year-after depth (S1-008) counts the word after. This adds one to a term integer written as a number. Damnatio is a deletion mark and is not this. Andurārum is not implemented as 'cancel every earlier debt'.

### S1-040 A dotted triple is the date tuple

BE 06/1, 088 (CDLI 258601) is catalogued as a case tablet, condition_description 'Case Tablet(broken)', two museum numbers CBS 01269 and CBS 01269a, dates_referenced Ammi-ditana.12.01.06. The transliteration is empty, so no inner clause is claimed. The date shape that was fetched is a three-integer dotted triple: year-of-reign, month, day. Pleiades 894089 is Sippar, the tablet's CDLI provenience. The new parser sits between inline_date_tokens (which ends at line 1619 and keeps parsing only ISO and month names) and parse_qty_word at 1621. It recognizes three dot-separated integers and stores them as (y, m, d). The seat sorts by the tuple lexicographically as integers. String order is not used, so month 12 sorts after month 2. Two turns that share the triple and whose content sets overlap are one event, because the catalog treats the two museum numbers as one act. The longer content set is kept.

Math: `key = (y, m, d) as i64; sort lexicographically; if key(a)==key(b) AND content overlap nonempty, keep the larger content set`

Tag `ORDERING` on LOC Temporal; BEAM temporal_reasoning; LME temporal-reasoning (t5_not1 7; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:1619-1621`. Record: https://cdli.earth/artifacts/258601.json.

Dedupe: Diary calendar match parses month names and weekday names into a window on session_ts. This parses only a dotted triple of integers and does not read session_ts. inline_date_tokens is not retuned. Regnal anchoring retrieves a second turn. This is one sort key.

### S1-041 Date-zone integers, not quantities, exempt the session cap

S1-001 exempts a same-session turn when any terminal integer differs. On a balanced account the last integer can be a quantity (the deficit) rather than a date, and two different deficits would then look like two times. S1-014 defines the date key as the integer in the last eight tokens only when an earlier integer exists. The combination uses that date key, and only that date key, as the exemption. A second expenditure of the same day, distinguished by a different medial quantity, stays capped. A second event with a different date-zone integer is admitted. Records: CDLI 100212 (many lines, one account) and CDLI 388630 (year-name in final position). The literal 1 at 5619-5625 is not changed.

Math: `exempt iff is_event_sequence AND date_key(ti) is Some AND date_key(ti) != date_key(sj) for every selected sj in the session`

Tag `RECALL/INDEX` on BEAM event_ordering (pool_not_k 17; k_not5 8). Hook `crates/vestige-core/src/retrieval.rs:5676-5679`. Record: https://cdli.earth/artifacts/100212.json.

Combination: dossier of many events (S1-001) + terminal integer is the date (S1-014).

Dedupe: S1-001 alone would exempt on any terminal integer, including a deficit. Petrie seriation still differs: it spreads sessions by session_ts. This is one predicate at the existing continue, using the date-zone column.

### S1-042 The deficit is legal only across a blank

S1-002 subtracts every later integer from every earlier integer. On AAS 065 the blank and sza3-bi-ta are what make the two sides real; integers inside the expenditure section are not a second capital. The combination emits a deficit feat only when a blank actually splits the turn. A turn with no blank emits no deficit, so a list of numbers is not mistaken for an account. The feat is |sum(section 0) - sum(section 1)|. Further sections after a second blank are not folded into the subtraction. Records: CDLI 100053 and CDLI 100212.

Math: `emit deficit iff section_count >= 2; deficit = sum(section 0) - sum(section 1); sections >= 2 are ignored by this feat`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (pool_not_k 83; no_pool 13). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100053.json.

Combination: blank section cut (S1-003) + capital-minus-expenditure (S1-002).

Dedupe: Trial balance is query-group coverage and does not subtract. Petuchah would score the best section. This emits one integer and only across the first blank. S1-002 without the blank is the single; this row is the gated form.

### S1-043 A price line may be a slash fraction of the line above

AAS 065 puts fractions on the ku3-bi lines: 1/2, 1/3, 5/6 of a shekel. The adjacent-line rule alone stores the raw integers and misses a query that names the composed value. The combination stores the pair (quantity_on_line_i, composed_value_on_line_i+1), where the composed value is the slash rule applied to the price line. Both the raw price integer and the composed numerator are pairable. A query matches if either form of the price sits on the next line. The two mechanisms are one feat, not two independent boosts.

Math: `price = compose(slash, integers on line i+1); pair = (int(line i), price); match if query contains int(line i) and the numerator of price`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Multi-Hop (no_pool 13). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100053.json.

Combination: adjacent price line (S1-004) + slash fraction (S1-005).

Dedupe: Ana ±10% would accept a near price. This is exact. S1-004 and S1-005 each fire alone; this row requires both the adjacency and the composition, and it does not emit a price for a fraction that is not on the next line.

### S1-044 A worker-day product is dated by its year-after depth

A product that is not dated attaches to every year the session mentions. CUSAS 39, 153 dates a one-day wage with a double year-after. AAS 013's worker-days belong to the year named on that account. The combination emits the product feat only together with the year key and the depth, as a triple (product, year_key, depth). A query integer matches the product only when the query's year key, if it has one, equals that key. Two years no longer share one labor total. Records: CDLI 100001 and CDLI 250483.

Math: `feat = (a*b, year_key, depth); match requires product equality and, when the query has a year_key, year_key equality`

Tag `RECALL/INDEX` on LOC Temporal; LOC Multi-Hop; LME temporal-reasoning (pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100001.json.

Combination: worker-day product (S1-006) + year-after depth (S1-008).

Dedupe: Regnal anchoring ranks by session-date distance to a retrieved anchor. This triple is stored at ingest and matched exactly. Diary match is not used. The product rule and the depth rule each exist as singles; the triple is the new object.

### S1-045 A copy yields only when the sealed integer matches

A gaba-ri copy (CDLI 100090) yields under S1-009 whenever its tokens are a subset. A copy that changes the sealed line is a different act: CDLI 100298 shows the seal binds one consignment's weight, not the tablet. The combination yields only when the short turn's sealed integer equals the long turn's sealed integer. A short turn whose sealed integer differs stays in contention, because it is a different consignment, not a copy. The subset test still has to pass.

Math: `yield iff strict content subset AND sealed_int(short) == sealed_int(long) AND same session`

Tag `ORDERING` on LOC Single-Hop; LOC Multi-Hop (t5_not1 61). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/100090.json.

Combination: strict-subset copy (S1-009) + seal binds the previous line (S1-010).

Dedupe: S1-009 yields on token subset even if a sealed number changed. Source-superset realloc is a shipped user/assistant seat. Palimpsest uses Jaccard. This requires equality of one integer, the sealed one, and a strict subset.

### S1-046 Witness zone stripped before the date key is read

The Nippur sale (CDLI 101851) puts the witness names after the oath and before the month and the year. LB 0068 (CDLI 388630) puts the year-name last. Taken together, the event key is the date-zone integer, and every title-case token between the last content clause and that integer is deleted before slot 1 is chosen. A turn that is only those names cannot sit at slot 1 while any candidate has a date key. The seat is the new function called at retrieval.rs:6265-6272, so it runs on Single-Hop and on LME user/assistant turns, not only on event-sequence phrasing. It does not read the speaker string.

Math: `event_key = date_key after deleting witness_zone; a witness-only turn loses slot 1 to any turn with a date_key`

Tag `ORDERING` on LOC Single-Hop; LME single-session-assistant; LME single-session-user; LME multi-session; BEAM summarization; BEAM information_extraction (Single-Hop t5_not1 192; LME t5_not1 78). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/101851.json.

Combination: witnesses are not events (S1-013) + terminal integer is the date (S1-014).

Dedupe: Loci picks the k-th bullet. This deletes a zone. Two-witness corroboration requires two attestations. S1-013 and S1-014 are the singles; this row is the joint predicate, and it is the one aimed at Single-Hop t5_not1.

### S1-047 Oath and doubled-negation clauses drop out together

On CDLI 101851 the oath (one title-case token, no integer) and nu-gi4-gi4-da (a negation plus a doubled verb) are both formula. Either rule alone leaves the other formula able to occupy slot 1. The combination drops a clause when it is an oath or a negated doubling, and only then reads the date key. What remains is the sale. The same shape, a one-name swear or a negated repetition, is what an instruction-following echo looks like when the echo restates the rule instead of the act.

Math: `drop clause if (caps==1 and ints==0) or (negation and doubled token); then apply date_key`

Tag `ORDERING` on LOC Single-Hop; BEAM instruction_following; LME single-session-assistant (Single-Hop t5_not1 192; instruction t5_not1 12). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/101851.json.

Combination: oath clause (S1-015) + negated reduplicated verb (S1-016).

Dedupe: Ezib zeroes excluded query tokens. This drops formula clauses by shape count, not by an except-word. Epistolary length exclusion uses document frequency. No df test here.

### S1-048 The price counts only inside a neighbor chain

YBC 05336 (CDLI 306421) identifies the orchard by an ordered da-chain and states the price in the completion clause. A turn that has the price integer but not the two query entities in order is a different field. The feat is emitted only when both the completion-clause integer and the ordered pair exist. A query with two entities and one integer matches that feat and not a turn that merely contains the integer. This is how a Single-Hop or Multi-Hop gold that is crowded out of k gets a pair the head does not have.

Math: `emit (price, e1, e2) iff price is the completion-clause integer AND (e1, e2) is an ordered neighbor pair with gap <= 1 token`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (Multi-Hop pool_not_k 83; Single-Hop pool_not_k 95). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/306421.json.

Combination: completion-clause price (S1-017) + ordered neighbor chain (S1-018).

Dedupe: Hollerith is an unordered pair index. Cartouche glues a title-case span into one name. This stores an ordered pair plus one integer. Khipu two-hop is a second-session bridge and is not used.

### S1-049 A periodic share is the standing obligation

PBS 08/2, 116 (CDLI 262151) is a yearly ration owed by several heirs. The standing fact is share = integer / name_count, tied to the closed-class every, not the one-shot mention of the same commodity. The feat is copied onto every turn in the session that names one of the heirs, which is what lets a preference question with no lexical overlap (no_pool) hit the ration turn, and what lets a later session that only names the heir enter the session list when its heir id matches. Preference object/attribute boosts are not used. The hook is ingest, plus the session admission predicate at 5489-5490 when the only signal is the heir id. Speaker is unread.

Math: `share = n / names if every-mark; inherit (heir_id, share) across the session; a session with s_score==0 is admitted when its heir_id set intersects the query heir`

Tag `RECALL/INDEX` on BEAM preference_following; LME single-session-preference; LOC Multi-Hop (preference no_pool 9; LME ssp pool_not_k 11). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/262151.json.

Combination: every-period integer (S1-019) + integer divided by conjunct count (S1-020).

Dedupe: Gemini preference object/attribute boosts are off limits. This is an integer quotient plus a closed-class every, inherited on an entity id. Notarial speaker scope is not used. Stocktaking does not down-weight terms.

### S1-050 Joint principal plus exact half-interest

YOS 08, 096 (CDLI 290103) is one principal shared by two named parties. Hermitage 3, 073 (CDLI 211416) states interest of one half. The combination emits interest = shared_principal / 2 on both names, and only when the half-mark is present. A query integer equal to the interest matches. A query equal to the principal still matches the principal feat. The two are different columns, so a question about the extra does not retrieve the capital line.

Math: `if two names share n and a 1/2 mark is in the next clause, emit interest = n/2 on both names`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Open-Domain (Single-Hop pool_not_k 95; Open-Domain pool_not_k 35). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/290103.json.

Combination: two conjuncts share the integer (S1-021) + half-interest (S1-022).

Dedupe: Ana ±10% is a tolerance. This is exact division by 2. Compound-verb šu ba-ti is not the key. S1-021 and S1-022 are the singles; the pair (shared principal, half) is the new feat.

### S1-051 Later integers are extras, and extras may be fractions

The Fara sale (CDLI 10009) writes price, then surplus, then gift. AAS 065 writes those later quantities as slash fractions. The seat treats integer 0 as the price even after slash composition, and treats every later composed value as an extra. When the query has one integer, a turn whose price equals it outranks a turn whose only match is an extra. That is a slot-1 swap inside the current top 5, which is the Single-Hop and Open-Domain t5_not1 pool.

Math: `price = compose(first integer line); extras = compose(later lines); slot 1 prefers price-equal when the query has one integer`

Tag `ORDERING` on LOC Single-Hop; LOC Open-Domain (Single-Hop t5_not1 192; Open-Domain t5_not1 17). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/10009.json.

Combination: first integer is the price (S1-023) + slash fraction (S1-005).

Dedupe: Primary-aggregate prefers a turn with fewer digits. This assigns roles by position inside one turn and does not compare digit counts across turns.

### S1-052 The seal binds the measure, not the count

CDLI 100298 rolls the seal against the line that carries ki-la2-bi, the weight. CDLI 20565 separates the count of spears from the weight that follows the possessive. The combination binds the seal to the measure integer only. The count integer stays unbound. A query that names the sealed party plus a number matches the measure, and the measure feat is inherited by that party's other turns in the session so an extraction question with no surface overlap still resolves. All@5 on extraction moves only if the component lines, not the sealed total alone, stay in the set: the sealed measure does not consume every slot.

Math: `sealed_measure = measure integer on the line before the seal; count is not sealed; inherit sealed_measure onto the party id`

Tag `RECALL/INDEX` on BEAM information_extraction; LOC Single-Hop (IE no_pool 8; IE pool_not_k 3). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100298.json.

Combination: seal binds the previous line (S1-010) + count then possessive measure (S1-024).

Dedupe: Digit-unit split rewrites 5km. This does not touch raw_tokens. Abusir red-stroke emits done/todo. This binds one integer, the measure, to the previous line's party.

### S1-053 The carried remainder is the next account's capital

Aleppo 001 (CDLI 100333) ends a count with si-i3-tum and a year-name. AAS 013 (CDLI 100001) opens the next balanced account on a remainder. The integer in the remainder section of the earlier turn is stored as the opening capital of any later turn that shares the place token and whose year depth is exactly one greater. A temporal question whose numeral is that carried integer then matches the later turn, which is the pool_not_k and k_not5 path for Temporal and for LME temporal. session_ts is not the link.

Math: `if turn B shares place(A) and depth(B)==depth(A)+1, opening(B) gets remainder_int(A) as an inheritable feat`

Tag `RECALL/INDEX` on LOC Temporal; LOC Multi-Hop; LME temporal-reasoning (Temporal pool_not_k 53; LME temporal k_not5 5). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100333.json.

Combination: opening remainder is prior input (S1-025) + deficit integer (S1-002).

Dedupe: Regnal anchoring retrieves an anchor and ranks by session-date distance. This copies one integer across a depth difference of 1. Trial balance is query-group coverage. Torrens would keep the later turn and drop the earlier; both stay.

### S1-054 Year scope and year-after depth are one sort key

AAS 013 and Aleppo 001 each have one year formula, and us2-sa stacks on it. The sort key is (first_seen year_key, depth). Two accounts of the same field in year 0 and year-after 1 stop tying, which is the temporal t5_not1 failure when the later year is sitting in the top 5 under the earlier one. The seat does not read session_ts and does not multiply a score. It runs at 6265-6272 for every query, including LME temporal and BEAM temporal_reasoning.

Math: `sort by (first_seen[year_key], depth, turn index)`

Tag `ORDERING` on LOC Temporal; LME temporal-reasoning; BEAM temporal_reasoning (Temporal t5_not1 61; LME temporal t5_not1 18). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/100001.json.

Combination: year-name scopes the account (S1-026) + year-after depth (S1-008).

Dedupe: Diary calendar match uses month names and session timestamps. Harris matrix is an edge topology. This is one ordinal pair already defined by S1-007 and S1-008.

### S1-055 A deficit basket pulls deficit turns and not the tag

AAS 132 (CDLI 100116) says the basket holds sealed documents of reed deficits, from Biduga. It does not contain the figures. AAS 013 contains a remainder. The tag scores 0. Each of its content tokens is written onto turns that have a deficit column and share an entity with the tag. Those turns, which do not repeat the basket's wording, become the hits. Open-Domain no_pool is the category that cannot clear 90% without this kind of inherited token: in-pool is 78 and the need is 83. The same inheritance admits a session whose own s_score is 0 when the query matches the tag. Instruction-following no_pool is the same shape when the gold turn is a member of a labeled file rather than the label.

Math: `if label(turn) then slot score 0; copy label tokens onto turns with a deficit column and a shared entity`

Tag `RECALL/INDEX` on LOC Open-Domain; LOC Multi-Hop; BEAM instruction_following (Open-Domain no_pool 14; Multi-Hop no_pool 13). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100116.json.

Combination: basket tag is not the answer (S1-027) + deficit integer (S1-002).

Dedupe: Catch-lines link the tail of one session to the head of the next. This copies a five-line tag onto deficit members and zeroes the tag. Finding-aid smoothing blends window scores. Incipit weights the first line. This refuses to seat it.

### S1-056 The year on the basket reverse is the file's session key

AAS 165 (CDLI 100145) is a basket tag whose reverse is only the year-name mu en {d}nanna masz-e i3-pa3. LB 0800 (CDLI 368691) is a year-formula tablet that gives those formulas an order. Pleiades 44626252 is Umma, the city of the AAS tags. The year on the label is the session key of every member turn, not a turn score. At the session admission filter (retrieval.rs:5489-5490) a session with s_score==0 and bridge_boost==0 is added to `remaining` when its year_key equals the query's year_key or the label's year_key. After the list is built (5602-5607) the member session, not the label session, is swapped to slot 0 when both are in the list. The scalars 0.62, 0.45, and 0.12 are not written. This is the session path. LoCoMo Session@5 already clears the bench 90% line and does not clear Multi-Hop (246<254) or Open-Domain (65<83). BEAM Session@10 misses event_ordering, extraction, instruction, preference, and summarization. Those five need a session that is outside the current top 10, which is this admission, not a reorder.

Math: `admit session s if s_score==0 and year_key(s)==year_key(query); then if label session is slot 0 and a member session is in the list, swap the member to slot 0`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Multi-Hop; LOC Temporal; LOC Open-Domain; BEAM event_ordering; BEAM information_extraction; BEAM instruction_following; BEAM preference_following; BEAM summarization; BEAM contradiction_resolution; BEAM knowledge_update; BEAM temporal_reasoning; BEAM multi_session_reasoning; LME single-session-preference; LME temporal-reasoning (LoCoMo Session@1 gap 250; BEAM Session@1 gap 187). Hook `crates/vestige-core/src/retrieval.rs:5489-5490`. Record: https://cdli.earth/artifacts/100145.json.

Combination: basket tag (S1-027) + year-formula ordinal (S1-038).

Dedupe: Incipit and session topic heads weight the first user turn. This uses a year ordinal from a separate formula tablet, matched exactly, and it zeroes the label. Catch-lines require token overlap between adjacent sessions. Florilegium keeps one excerpt per source; this admits a second session when the year key matches. No recency_boost edit.

### S1-057 The hyphenated place is not the beneficiary

CDLI 247781 issues goods from a hyphenated storehouse (e2-kiszib-ba-ta) to a person. CDLI 10012 separates the beneficiary from the later controller. The place feat and the beneficiary feat are different ids. A query name matches the beneficiary and not the hyphenated token. A query that is the place matches the hyphenated token and not the person. Single-Hop k_not5 is the case where both turns are in the top 10 and the place turn is sitting in the top 5 above the person turn; the feat gives the person turn the uncovered integer.

Math: `place = hyphenated token after from; beneficiary = earlier title-case name; the two feats do not intersect`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Multi-Hop (Single-Hop pool_not_k 95; k_not5 51). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/247781.json.

Combination: hyphenated from-place (S1-028) + earlier name is beneficiary (S1-029).

Dedupe: ki-X-ta source/goal is a different shipped rule. Scriptio continua joins adjacent tokens. This classifies one already-hyphenated token and does not concatenate.

### S1-058 The median controller is not the prevailing party

CDLI 217459 puts the controller in the middle of three names and closes the case. CDLI 111382 gives the field to the name that stands in both the opening and the disposition; the brother who withdrew does not. The controller feat and the prevail feat are different. A query whose sole name is the median retrieves the controller turn. A query whose name is in both end clauses retrieves the verdict turn. Contradiction questions fail when the claim turn outranks the disposition. The disposition is the prevail feat, and it takes slot 1 only against the claim, not against an unrelated turn.

Math: `controller = caps[1] when caps==3; prevail = intersection(first clause, last clause); slot 1 prefers prevail when the query name is in that intersection`

Tag `RECALL/INDEX` on LOC Single-Hop; LME multi-session; BEAM contradiction_resolution (contradiction t5_not1 8; LME multi-session t5_not1 25). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/217459.json.

Combination: median controller (S1-011) + name in both end clauses (S1-012).

Dedupe: Third-text arbitration votes a third session. Torrens takes the latest turn. This is an intersection of two clauses inside one turn, and it never prefers a name merely because it is later.

### S1-059 Course ordinals order the work and a repeated integer does not

CDLI 127648 numbers brick courses a-ra2 1-kam then a-ra2 2-kam. MCT 036 (CDLI 254518) repeats 7.36 on the next line as the check (ib2-si8). A repeated integer with no new entity is not a new course. The seat orders the remaining clauses by the explicit adjacent integer. The check line is dropped from the order. The Met fragment 322022 is only a catalog witness that problem texts exist; the repeat was read on 254518.

Math: `drop int[i] if int[i]==int[i-1] and no entity between; sort remaining clauses by adjacent course integer`

Tag `ORDERING` on BEAM event_ordering; LME temporal-reasoning; LOC Temporal (event_ordering t5_not1 7; temporal t5_not1). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/127648.json.

Combination: course ordinal (S1-030) + repeated integer is a check (S1-031).

Dedupe: Loci selects the k-th bullet as the answer. This orders clauses by an integer that is already written next to a repeated stem, and it deletes an immediate numeric repeat. Dittography collapses two identical turns, not two adjacent integers.

### S1-060 A row's amount is its last integer and the row is one fact

CDLI 411619 lays numbers in subcolumns of one row. CDLI 8105 puts the amount at the end of the line. The amount of a row is the last integer in that row, and the earlier cells are fields of the same fact id. A query integer matches the amount only inside the row whose other fields match the query's entities. Extraction no_pool is the case where the query names the field and not the digit: the field tokens inherit the amount, so the digit becomes reachable. All@5 keeps every row that matches a distinct query entity instead of collapsing the tablet to one number.

Math: `amount(row) = last integer in the row; inherit amount onto the row's earlier tokens; All@5 keeps one turn per matching row`

Tag `RECALL/INDEX` on BEAM information_extraction; LOC Single-Hop (IE no_pool 8; IE k_not5 2). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/411619.json.

Combination: same-row cross-column (S1-032) + line-final amount (S1-033).

Dedupe: Abusir header inheritance copies a header line onto cells. This does not copy a header. It binds a row index and labels the last integer. S1-032 and S1-033 are the singles; the amount is defined only because both hold.

### S1-061 A digit-free list and a witness block cannot be the timeline

The Abu Salabikh list (CDLI 10101) and the god list (CDLI 10566) are status orders. The sale's witness block (CDLI 101851) is a name list with a date after it. A turn with four or more title-case tokens and no date key, or a witness zone with no date key outside it, loses slot 1 to any turn that has a date key. Single-Hop's 192 t5_not1 misses are the largest Turn@1 pool in LoCoMo. This seat spends the ones whose slot 1 is a list. It does not invent an order from the list.

Math: `if (caps>=4 and date_key is none) or witness_only, lose slot 1 to any turn with a date_key`

Tag `ORDERING` on LOC Single-Hop; LOC Open-Domain; LME single-session-assistant; BEAM summarization (Single-Hop t5_not1 192). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/10101.json.

Combination: digit-free name list is not time (S1-034) + witness zone (S1-013).

Dedupe: Incipit weights a first line. Loci reads list position as the answer. This refuses both. S1-034 requires no integer at all; S1-013 requires a date later in the turn. The combination covers both shapes with one seat.

### S1-062 A gap does not join a condition to the wrong consequent

CDLI 226628 opens with he-pi, a break. The Hammurabi stele (CDLI 249253) splits each law on szum-ma, and the consequent is the ruling. ORACC SAA 04 225 (P238993) has the operative question in a broken beginning; that page is not used as an ezib rule. The combination inherits consequent ids only across an if that has no gap between the condition and the consequent. A condition on the far side of a bracket does not inherit the next law's penalty. Open-Domain and instruction-following cannot clear 90% from the in-pool set alone (Open-Domain in-pool 78<83, instruction in-pool 35<36), so this inherited id is the no_pool path. It is intra-turn. It does not pair a user turn with an assistant turn.

Math: `inherit consequent ids onto condition tokens iff an if-token exists and no gap mark lies between the if and the consequent`

Tag `RECALL/INDEX` on LOC Open-Domain; BEAM instruction_following; LME knowledge-update (Open-Domain no_pool 14; instruction no_pool 5). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/226628.json.

Combination: gap mark cuts co-occurrence (S1-035) + if indexes its consequent (S1-037).

Dedupe: Responsa links a later answer turn by speaker role. This never crosses turns and never reads speaker. Ezib zeroes tokens under an exclusion cue. Turin lacuna drops df=0 query groups. This cuts at an orthographic gap and copies ids the other way, from consequent back to condition.

### S1-063 The turn score is the best case, not the sum of cases

TBP 44 (CDLI 236901) opens each line with the same sign, one omen per line. The stele opens each law with szum-ma. When a turn is a sequence of cases, its coverage for slot 1 is the maximum case, not the sum. A long omen list stops outranking the one line that matches. The case boundary is a repeated line-initial token or a closed-class if. Petuchah's best-section coverage is keyed on blank lines and is not this trigger.

Math: `if cases are defined, slot score = max over cases of case hits, not the sum`

Tag `RECALL/INDEX` on LOC Single-Hop; BEAM instruction_following; LME multi-session; BEAM multi_session_reasoning (instruction t5_not1 12; LME multi-session t5_not1 25). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/236901.json.

Combination: one case per repeated line head (S1-036) + if-consequent (S1-037).

Dedupe: Petuchah takes the best blank-delimited section. This fires on a repeated line head or an if, and the unit is the case. The score is an integer hit count, not a new f32 weight.

### S1-064 A formula tablet's ordinal must be the account's year

LB 0800 is nothing but a year formula. AAS 013 names one year for the whole account. A turn whose terminal sequence matches formula i receives year_ord = i, and a turn that already has a different year key does not inherit it. Temporal k_not5 is a gold in the top 10 with the right year sitting under a turn from another year. The ordinal separates them. LME temporal Turn@5 is 111 and the category need is 115, so four golds have to enter the top 5 from k. This ordinal is that separation, not a session-timestamp window.

Math: `year_ord = index of the unique formula that is a suffix; a turn with a conflicting year_key gets no ordinal`

Tag `RECALL/INDEX` on LOC Temporal; LME temporal-reasoning; BEAM temporal_reasoning (Temporal k_not5 12; LME temporal k_not5 5). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/368691.json.

Combination: year-formula ordinal (S1-038) + year-name scopes the account (S1-026).

Dedupe: Incipit weights the first user turn. Regnal anchoring does a second retrieval against session dates. Diary match parses month names. This is a suffix-to-ordinal map from one label turn.

### S1-065 Release offset stacks on year-after depth

The stele clause before law 118, which was read, sets a three-year term and puts the release in the fourth year (an-du-ra-ar-szu-nu isz-sza-ak-ka-an). CUSAS 39, 153 stacks mu us2-sa twice. The release sort key is depth + (N+1), not depth alone and not N alone. A knowledge-update question that asks what changed after a term uses the same offset: the later clause is N+1 ahead of the pledge, so it sorts after it inside the top 5. No debt is deleted.

Math: `release_key = year_depth + term_N + 1; pledge_key = year_depth; sort by that pair`

Tag `ORDERING` on LOC Temporal; LME temporal-reasoning; BEAM temporal_reasoning; BEAM knowledge_update (temporal t5_not1). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/249253.json.

Combination: release at term+1 (S1-039) + year-after depth (S1-008).

Dedupe: Torrens keeps the latest document and drops the earlier. This spaces two clauses by N+1 and keeps both. Damnatio is not applied. Year-after depth alone does not add the term integer.

### S1-066 A dotted triple outranks a bare terminal integer

BE 06/1, 088 (CDLI 258601) is a case tablet, two museum numbers, dates_referenced Ammi-ditana.12.01.06. The transliteration is empty, so no inner clause is claimed. Pleiades 894089 is Sippar, the provenience. When a dotted triple is present it is the date key, as (y, m, d), and a bare terminal integer is the date key only if no triple exists. Month 12 then sorts after month 2. The ISO parser at 1452-1619 is not edited. The new parser is inserted after line 1619. Two turns that share the triple and overlap in content are one event; the longer content set is kept, which is the catalog fact that two object numbers are one act.

Math: `if dotted triple exists, date_key = (y, m, d) as integers; else date_key = terminal integer; collapse equal triples to the larger content set`

Tag `ORDERING` on LOC Temporal; BEAM temporal_reasoning; LME temporal-reasoning; BEAM event_ordering (temporal t5_not1; event_ordering t5_not1 7). Hook `crates/vestige-core/src/retrieval.rs:1619-1621`. Record: https://cdli.earth/artifacts/258601.json.

Combination: dotted date triple (S1-040) + terminal integer is the date (S1-014).

Dedupe: Diary match uses month-name lists and session_ts. This parses only a dotted triple. inline_date_tokens is not retuned. Regnal anchoring is a second retrieval.

### S1-067 Worker-days in the remainder are the prior balance of labor

AAS 022 (CDLI 100010) opens with 34 workers for 1 day on a named field, then a month, a blank, and the year. AAS 013 carries a remainder of a balanced labor account. The product on a line inside the remainder section is the prior labor balance, and it is emitted with ordinal depth-1. A later section's product is this year's labor. A query integer equal to the prior product matches the remainder feat and not this year's product. That split is what puts the gold line into k when the head line is the other year's total.

Math: `product in section 0, if the section is a remainder, is tagged prior; product in later sections is tagged current; match the tag the query's year depth selects`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Temporal (Single-Hop pool_not_k 95; Temporal pool_not_k 53). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100001.json.

Combination: worker-day product (S1-006) + opening remainder (S1-025).

Dedupe: Khipu summation maximizes distinct pairs and does not touch slot 0. This emits one tagged product. Double-entry collapse of (numeral, noun) is not applied.

### S1-068 A price pair cannot cross a blank

AAS 065 puts the silver capital, a blank, sza3-bi-ta, then commodity lines each followed by its silver. Pairing the capital integer with the first commodity's silver is a false rate. The pair feat is emitted only when the two adjacent lines share a section index. The capital stays in section 0 and is not a price. Multi-Hop and Single-Hop pool_not_k golds that state the rate then match the pair, and the capital turn does not.

Math: `pair(q, p) only if line(p)=line(q)+1 and section(q)=section(p); a blank increments section and blocks the pair`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (Multi-Hop pool_not_k 83; Single-Hop pool_not_k 95). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100053.json.

Combination: blank section cut (S1-003) + adjacent price line (S1-004).

Dedupe: Petuchah replaces coverage with the best section. This only gates which integer pairs exist. S1-004 without the blank would pair across the capital. This row is that gate.

### S1-069 Neighbors identify the field and the recurring name is the buyer

The orchard sale names the neighbors in order and the buyer in the operative clause. The ditilla (CDLI 111382) keeps the name that occurs in both the opening and the close. The feat is the ordered neighbor pair conjoined with the prevailing name. A query that names two neighbors and one party matches only when the party is the prevail name, not a witness. Slot 1 then prefers that turn over a witness turn already in the top 5.

Math: `match iff neighbor pair (q1, q2) exists and query party is in prevail(turn)`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop (Multi-Hop t5_not1 61; pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/306421.json.

Combination: neighbor chain (S1-018) + prevailing name (S1-012).

Dedupe: Muster-roll fills slots 2-5 with distinct names. This does not fill slots. It gates one turn on an ordered pair plus an intersection.

### S1-070 Half-interest applies to the measure, not the count

Interest of one half on CDLI 211416 is a rate on a quantity of barley, which is a measure, not a headcount. CDLI 20565 puts the count first and the possessive weight second. Half is applied to the measure integer only. interest = measure/2. The count is unchanged. An extraction question that asks the interest and names the commodity matches the measure-interest feat even when the count tokens are what the head turn shares with the query.

Math: `interest = measure / 2; count is not divided; emit interest on the measure's entity`

Tag `RECALL/INDEX` on LOC Single-Hop; BEAM information_extraction (IE no_pool 8; Single-Hop pool_not_k 95). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/211416.json.

Combination: half-interest (S1-022) + count versus measure (S1-024).

Dedupe: Ana ±10% is not exact division. Digit-unit split is not used. S1-022 divides the principal indiscriminately; this row divides only the measure column.

### S1-071 A name list that is a subset of a longer turn yields

A gaba-ri (CDLI 100090) is a copy and yields when it is a strict subset. A profession list (CDLI 10101) is four or more names and no integer. The combination demotes a short name list when its title-case set is a strict subset of a longer turn in the same session and the short turn adds no integer. Single-Hop t5_not1 (192) is the largest single stage in the LoCoMo Turn@1 gap of 578. This seat spends the misses whose slot 1 is that shorter list. It also runs on LME, where Turn@5 is already past the 90% line and Turn@1 is not, so the whole LME Turn@1 gap is inside the top 5.

Math: `yield if caps(short) strict-subset caps(long) and ints(short) is empty and same session and index(short)>index(long)`

Tag `ORDERING` on LOC Single-Hop; LME single-session-assistant; LME single-session-user; LME multi-session; BEAM summarization; BEAM preference_following (Single-Hop t5_not1 192; LME t5_not1 78). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/100090.json.

Combination: strict-subset copy (S1-009) + digit-free list is not time (S1-034).

Dedupe: Palimpsest uses Jaccard. Source-superset realloc is a user/assistant seat. This is a strict subset of title-case ids, speaker-agnostic, and only when the short turn has no integer. Dittography collapses exact duplicate turns; a proper subset is not a duplicate.

### S1-072 The inspection basket points at remainder turns

AAS 165 labels a basket of inspections (gurum2-ak) and puts only the year on the reverse. Aleppo 001 is a counted list closed by si-i3-tum. The label scores 0. Turns in the same year whose first section is a remainder inherit the label's inspection token. Summarization All@5 is 0/36 because one summary turn occupies the set. The remainder members are the parts, and each distinct remainder integer is kept as its own candidate so the summary line, whose integer equals none of them individually, does not replace them.

Math: `label score 0; inherit label tokens onto remainder-section turns of the same year_key; do not drop a remainder turn because a sum turn exists`

Tag `RECALL/INDEX` on LOC Open-Domain; LOC Multi-Hop; BEAM summarization (Open-Domain no_pool 14; summarization all5 0). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100145.json.

Combination: basket tag (S1-027) + opening remainder (S1-025).

Dedupe: Khipu summation fills slots 2-5 with distinct instances and leaves slot 0 untouched. This zeroes a label and keeps component integers from being replaced by a sum. Primary-aggregate prefers the single transaction over a summary for slot 1. This is an All@5 membership rule, not a slot-1 preference for the shorter turn.

### S1-073 Different year-depth admits another turn from the same session

CUSAS 39, 153 dates one day's wages with mu us2-sa stacked twice, depth 2. Aegyptus 08, 264 11 holds many expenditure lines and one deficit in a single account. S1-001 exempts a second same-session turn only when is_event_sequence is set and the terminal integers differ. All@5 is 0 on event_ordering and on summarization, and LME multi-session Turn All@5 is 53/121 against a need of 109, because the cap of 1 (and the multi-hop cap of 2) drops sibling golds. Knowledge-update All@5 is the same shape: the earlier statement and the later one share a session and differ in depth, and both have to stay in the set. This row's predicate does not read is_event_sequence. At the session_capped continue (5676-5679) a candidate is exempt when its year_depth differs from every already selected turn of that session. The literals 1, 2, and 3 at 5619-5625 are not edited. Two turns with the same depth stay capped. The exempt turn enters the top 5, which is what All@5 counts.

Math: `exempt iff year_depth(ti) != year_depth(sj) for every selected sj in the same session; the cap literals stay`

Tag `RECALL/INDEX` on BEAM event_ordering; BEAM summarization; BEAM multi_session_reasoning; BEAM instruction_following; BEAM knowledge_update; LME multi-session; LME temporal-reasoning; LME knowledge-update; LME single-session-preference (BEAM All@5 gap 202; LME All@5 gap 109). Hook `crates/vestige-core/src/retrieval.rs:5676-5679`. Record: https://cdli.earth/artifacts/250483.json.

Combination: dossier of many events (S1-001) + year-after depth (S1-008).

Dedupe: Florilegium and the shipped cap keep one turn per session. This is an exemption keyed on year depth, not a new quota. Petrie seriation spreads sessions by session_ts. Uncovered-entity rescue is a different rule and is not this: the test is year_depth inequality, not an uncovered query entity. S1-001 is the event-sequence, terminal-integer exemption. This one is year depth, on every query type.

### S1-074 The labor deficit is the unmatched worker-day product

CDLI 100212 labels la2-ia3 on a labor account whose capital is a worker-day figure. The deficit feat is the worker-day product of the capital section minus the sum of the products of the expenditure lines, and only when a blank separates them. A query whose integer equals that deficit matches, which is the Single-Hop gold that states how many were left and does not repeat the crew count. The broken signs on 100212 are not claimed to balance. The operation is the one the labels support.

Math: `if sections>=2, labor_deficit = product(section 0) - sum(product(line) for lines in section 1); emit that integer`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Multi-Hop (Single-Hop pool_not_k 95; Multi-Hop pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100212.json.

Combination: deficit integer (S1-002) + worker-day product (S1-006).

Dedupe: Trial balance is group coverage. Khipu sums instances across sessions. This subtracts products inside one turn, across one blank. Sexagesimal place value is not parsed.

### S1-075 The basket names the controller of the sealed deficits

AAS 132 names the source official, ki Biduga-ta, on a tag for sealed deficits. WO 13, 08 puts the responsible official in the median name, not at either endpoint. The tag's median title-case token, or its single named official when it has only one, is the controller id copied onto member turns. A session whose s_score is 0 is admitted at 5489-5490 when that controller id is the query's sole title-case token. Open-Domain Session@5 is 65 against a need of 83, so eighteen sessions have to enter the top 5 from outside it. This admission is that path. The tag itself still cannot take slot 1.

Math: `controller_id = the sole title-case on a label, or the median of three; admit a session with that id even if s_score==0; label slot score stays 0`

Tag `RECALL/INDEX` on LOC Open-Domain; BEAM instruction_following (Open-Domain no_pool 14; instruction no_pool 5). Hook `crates/vestige-core/src/retrieval.rs:5489-5490`. Record: https://cdli.earth/artifacts/100116.json.

Combination: basket tag (S1-027) + median controller (S1-011).

Dedupe: Session topic heads weight the first user turn's text. This admits on one entity id taken from a label's name position. WH-slot voter is not used. Notarial scope does not multiply by speaker.

### S1-076 A dotted triple plus a term places the release

The case tablet's date is a triple (year, month, day). The stele places release in year N+1 of a stated term. When both a triple and a term integer exist, the release key is (y+N+1, m, d) and the pledge key is (y, m, d). The seat sorts by that triple. A bare terminal integer does not override a triple. The two events stay; they are not collapsed to the later one.

Math: `pledge = (y, m, d); release = (y+N+1, m, d); sort lexicographically as i64 triples`

Tag `ORDERING` on LOC Temporal; BEAM event_ordering; LME temporal-reasoning (temporal t5_not1). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/258601.json.

Combination: dotted date triple (S1-040) + release at term+1 (S1-039).

Dedupe: Torrens would drop the pledge. Diary match uses month names. This adds an integer to the year component of a dotted triple and keeps both clauses.

### S1-077 A fraction in one cell composes with the integer in the same row

Subcolumns on CDLI 411619 are one fact per row. AAS 065 writes 1/2, 1/3, and 5/6 on silver lines. A slash token in one cell composes with the integer in another cell of the same row, and with no other row. The composed numerator is the row's value feat. An extraction question whose numeral is that numerator matches the row even when the query never says the fraction. Cells in the next row do not enter the composition.

Math: `if a row contains an integer n and a slash a/b, value = n*b+a in lowest terms; the feat is the numerator; other rows are excluded`

Tag `RECALL/INDEX` on BEAM information_extraction; LOC Multi-Hop (IE no_pool 8; Multi-Hop pool_not_k 83). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/411619.json.

Combination: same-row binding (S1-032) + slash fraction (S1-005).

Dedupe: Sexagesimal composition parses place-value words. This composes one slash already written in the row. Ana ±10% is a tolerance. Abusir headers are not copied.

### S1-078 A name line labels the description, and a gap blocks the label

RIME 4.01.04.05 (CDLI 218121) describes the wall Ishme-Dagan built and closes the column with mu-bi-im, its name is, after the royal name. The name line is the label of the preceding description. Those name tokens are inherited onto the description's tokens, so a query that uses the name retrieves the description. CDLI 226628 opens with a break. If a gap mark sits between the description and the name line, the inheritance does not fire, because the name may belong to the text on the other side of the break. Preference no_pool and extraction no_pool are the categories whose in-pool counts sit under the 90% line (preference 30<36, extraction 32<36). Inherited name tokens are the ingest path. This is not an incipit weight on the first line, and it is not a colon key/value gloss.

Math: `if a final line is a title-case sequence plus a copula and no gap precedes it, inherit those title-case ids onto the preceding clauses`

Tag `RECALL/INDEX` on LOC Open-Domain; BEAM information_extraction; BEAM preference_following (Open-Domain no_pool 14; IE no_pool 8; preference no_pool 9). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/218121.json.

Combination: mu-bi-im name line + gap cut (S1-035).

Dedupe: Incipit weights the first user turn as a field. This inherits a final name line backwards onto the description. Glossenkeil is a colon key/value. There is no colon test. Ezib is not applied to this tablet.

### S1-079 An account is in scope only when its year ordinal matches the formula

The year-formula tablet assigns ordinals. AAS 013 belongs to one year. Pleiades 44626252 confirms Umma as the place of that archive. A session is admitted at 5489-5490 when its year ordinal equals the query's, even if s_score is 0. A turn inside a session whose ordinal disagrees receives no inherited year feat, so a same-named field in another year does not answer. Multi-Hop Session@5 is 246 against a need of 254, eight sessions short of a reorder. This admission is those eight, when they share the year ordinal and not the surface tokens. BEAM temporal All@5 rises only for questions whose other gold is the other year of the same field, which this admission can place in the set.

Math: `admit session if year_ord(session)==year_ord(query); do not inherit the year feat across a mismatched ordinal`

Tag `RECALL/INDEX` on LOC Temporal; LOC Multi-Hop; BEAM temporal_reasoning (Temporal pool_not_k 53; Multi-Hop session short of s5). Hook `crates/vestige-core/src/retrieval.rs:5489-5490`. Record: https://cdli.earth/artifacts/368691.json.

Combination: year-formula ordinal (S1-038) + year scope (S1-026).

Dedupe: Regnal anchoring ranks by distance to a session timestamp. This is exact ordinal equality from a formula list. Session topic heads are not used. S1-056 admits on the raw year key of a basket; this row admits on the ordinal assigned by the formula tablet, and it refuses a mismatched year.

### S1-080 One query integer selects the price and ignores extras

Fara (CDLI 10009) puts the price first and the surplus and the gift later. The Larsa sale (CDLI 306421) puts the price in the completion clause. When both signals exist they must agree: the completion-clause integer is the price only if it is also the first integer. If they disagree, no price role is assigned and the seat does not move. When they agree and the query has one integer equal to that price, the turn takes slot 1 over a turn that matches only a later extra. Extraction t5_not1 is the same swap.

Math: `price is assigned only if completion_int == first_int; then a one-integer query equal to price takes slot 1`

Tag `ORDERING` on LOC Single-Hop; LOC Open-Domain; BEAM information_extraction (Single-Hop t5_not1 192; Open-Domain t5_not1 17). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/10009.json.

Combination: first integer is the price (S1-023) + completion-clause price (S1-017).

Dedupe: Primary-aggregate compares shape counts. This requires two independent position tests to agree, and it does nothing when they disagree. WH-slot is not consulted.

### S1-081 A yearly obligation is not collapsed across different depths

The nadītu ration is owed every year (CDLI 262151). A later tablet with mu us2-sa (CDLI 250483) is the next year, not a duplicate of this year. Dittography would collapse them if the commodity integer matches. This row forbids that collapse when year_depth differs, and it exempts the later year's turn from the session cap at 5676-5679 on that same test. Preference All@5 stays at 11/39 because one statement of the preference occupies the top 5 and the other year's statement is dropped. Keeping both is the All@5 move. The cap literals are not edited.

Math: `do not collapse two turns that share (entity, integer) when year_depth differs; exempt the second from session_capped on that test`

Tag `RECALL/INDEX` on BEAM preference_following; LME single-session-preference; LOC Temporal (preference all5 11; LME ssp all5 14). Hook `crates/vestige-core/src/retrieval.rs:5676-5679`. Record: https://cdli.earth/artifacts/262151.json.

Combination: every-period integer (S1-019) + year-after depth (S1-008).

Dedupe: Dittography collapses exact duplicates. Double-entry posting dedupe collapses a shared (numeral, noun) inside the top 10. This refuses the collapse when depth differs. S1-073 exempts on depth for All@5 generally; this row is the preference case where the integer matches and the depth is the only difference, and it also blocks the dedupe.

### S1-082 A controller does not receive a share of the joint integer

The partnership (CDLI 290103) shares one integer across two names. The controller (CDLI 217459) is the median of three and is not a party. When a clause has two conjuncts and a third median name, the integer is emitted on the two conjuncts only. The median gets a controller feat and not the integer. A contradiction question that asks who owes the money stops returning the official. All@5 on contradiction keeps the two parties' turns and does not spend a slot on the controller turn.

Math: `if caps==3 and one integer, share the integer on caps[0] and caps[2] only; caps[1] gets controller and not the integer`

Tag `RECALL/INDEX` on LOC Multi-Hop; BEAM contradiction_resolution (Multi-Hop pool_not_k 83; contradiction all5 22). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/290103.json.

Combination: joint integer (S1-021) + median controller (S1-011).

Dedupe: Coordinated-claimant distribution emits a relation for both conjuncts and does not exclude a median. This excludes the median from the integer. Notarial speaker scope is not used.

### S1-083 A course ordinal orders labor and the product is not a second event

CDLI 127648 orders courses by a-ra2 N. AAS 013's worker-day product is the quantity of the work, not a separate event from the course. The seat orders by the course integer. The product is attached to that course as its quantity and does not generate another sort key. Two courses with products 10 and 20 sort by 1 then 2, not by 10 then 20.

Math: `sort key = course integer; the product is stored on the course and is not a sort key`

Tag `ORDERING` on BEAM event_ordering; LOC Temporal (event_ordering t5_not1 7; Temporal t5_not1 61). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/127648.json.

Combination: course ordinal (S1-030) + worker-day product (S1-006).

Dedupe: Loci returns the k-th item as the answer. This sorts by an explicit integer and refuses to sort by the quantity. S1-030 does not know about products; this row is the refusal.

### S1-084 Names after the cases are not further cases

Each DISZ line of TBP 44 is a case. The sale tablet puts witness names after the operative clauses. When a turn has case structure, title-case tokens that sit after the last case and carry no integer are a witness zone, not another case. They do not enter the event order and they do not take slot 1. The last case's consequent remains the event.

Math: `after the last case, caps with no integer are witness_zone; event order uses cases only`

Tag `ORDERING` on BEAM event_ordering; LOC Single-Hop (event_ordering t5_not1 7; Single-Hop t5_not1 192). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/236901.json.

Combination: one case per line head (S1-036) + witness zone (S1-013).

Dedupe: Two-witness corroboration requires two attestations before credit. This deletes a tail from the order. It does not require a second witness.

### S1-085 Place and neighbor chain must both match

A hyphenated place (CDLI 247781) and an ordered neighbor chain (CDLI 306421) are two different ways a plot is identified. The location feat fires only when both are present and the query matches the place token and at least one ordered neighbor pair. Either signal alone is a weaker single (S1-028, S1-018) and does not emit this feat. Open-Domain pool_not_k is a gold that shares a place word and is still outside k; the pair feat is the extra credit that changes the submodular pick without touching the 0.15 branch.

Math: `emit location feat iff place token matches and an ordered neighbor pair matches; otherwise no feat from this row`

Tag `RECALL/INDEX` on LOC Multi-Hop; LOC Single-Hop; LOC Open-Domain (pool_not_k across Single-Hop, Multi-Hop, Open-Domain). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/247781.json.

Combination: hyphenated from-place (S1-028) + neighbor chain (S1-018).

Dedupe: Hollerith is unordered. This requires a hyphenated place token and an ordered pair. Finding-aid smoothing is not applied. The 0.15 multiplier is not edited; the feat changes coverage so the existing branch is not the thing being tuned.

### S1-086 A filiation name in the seal caption is not part of the copy test

AAS 103 (CDLI 100090) writes the copy mark and a seal caption, the scribe son of his father, after a blank. AAS 022 (CDLI 100010) has the same shape: a seal caption a-kal-la dub-sar, dumu, after the year. The father's name is filiation, not a party. The copy test ignores title-case tokens that sit after the last blank and after a closed-class filiation word (son, daughter, child). Two captions that differ only in the father's name still count as the same copy, and the short turn yields. Without that exclusion the father name blocks the subset test and the echo keeps slot 1.

Math: `copy-test tokens = content ids minus title-case tokens after the last blank that follow a filiation word; then the strict-subset yield of S1-009`

Tag `ORDERING` on LOC Single-Hop; LME single-session-user; BEAM summarization (Single-Hop t5_not1 192). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/100090.json.

Combination: strict-subset copy (S1-009) + seal line scope (S1-010).

Dedupe: Household binding attaches a possessive to the nearest entity. This excludes a post-blank filiation name from a subset test. It does not emit a household feat. Palimpsest uses Jaccard.

### S1-087 A deficit answers only inside the same year ordinal

A deficit integer from AAS 013 answers a how-many-were-left question for that year only. The case tablet's date (CDLI 258601) separates the year component from the month and the day. The deficit feat is stored as (deficit, year_component). A query matches only when its year component equals that year, or when the query has no year and the turn's year is the only candidate. Last year's deficit stops answering this year's question. Temporal pool_not_k is the gold of the right year sitting outside k while last year's number holds the slot.

Math: `feat = (deficit, year); match requires deficit equality and, when the query has a year, year equality`

Tag `RECALL/INDEX` on LOC Temporal; LME temporal-reasoning; BEAM temporal_reasoning (Temporal pool_not_k 53; LME temporal pool_not_k 9). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100001.json.

Combination: deficit integer (S1-002) + dotted year component (S1-040).

Dedupe: Diary match windows a session timestamp with month names. This is exact equality of an integer pair. Trial balance is not a subtraction. S1-002 emits a bare deficit; this row refuses a cross-year match.

### S1-088 A query integer matches at most one of a rate or a product

AAS 065 has silver rates. AAS 013 has worker-day products. Both can emit the same integer from different operations, and the wrong operation takes slot 1. When a turn contains both a price pair and a product, the query matches the pair only if both of the pair's integers are in the query, and it matches the product only if the product's factors are not themselves the query's two integers. A query with one integer matches neither feat and falls through to the raw integer. The ambiguity is removed by a conjunction, not by a new weight.

Math: `if both shapes exist: pair matches iff both pair integers are in the query; product matches iff the query integer equals the product and the query does not contain both factors`

Tag `RECALL/INDEX` on LOC Single-Hop; LOC Multi-Hop (Single-Hop pool_not_k 95; t5_not1 192). Hook `crates/vestige-core/src/retrieval.rs:3125-3272`. Record: https://cdli.earth/artifacts/100053.json.

Combination: adjacent price pair (S1-004) + worker-day product (S1-006).

Dedupe: Ana ±10% would accept either nearby number. This is an exclusive conjunction of two exact feats. No scalar is introduced. Primary-aggregate is a shape preference and does not look at which integers the query contains.

### S1-089 An opening remainder sorts one depth earlier than its own year

Aleppo 001 writes the count, then si-i3-tum, then the year-name. The remainder belongs to the previous depth. Its sort key is (year_ord, depth-1). The year's own lines use (year_ord, depth). Inside a top-5 tie between the balance brought forward and this year's expenditure, the balance sorts first. That is the temporal and event-ordering t5_not1 swap. It is not latest-wins.

Math: `remainder ordinal = depth - 1; other sections use depth; sort by (year_ord, ordinal, index)`

Tag `ORDERING` on LOC Temporal; BEAM event_ordering; LME temporal-reasoning (temporal t5_not1; event_ordering t5_not1 7). Hook `crates/vestige-core/src/retrieval.rs:6265-6272`. Record: https://cdli.earth/artifacts/100333.json.

Combination: opening remainder (S1-025) + year-after depth (S1-008).

Dedupe: Torrens puts the latest section first. This puts the carried remainder first. S1-025 states the flag; S1-008 states the depth; the key is the pair.

### S1-090 A later integer equal to earlier plus a stored period is the same series

CDLI 551464 is ADART 6, 96, BM 032342, genre astral, with no transliteration in the artifact payload. Crossref 10.31826/9781463232405-019 returns the volume title Astronomical Diaries and Related Texts from Babylonia, vol. VI: Goal Year Texts. No planet period is hardcoded, because none was read off the tablet. The period is an integer stored on a third turn. A candidate is exempt from the session cap when its date-triple year component, or its terminal integer, equals an already selected turn's integer plus that stored period. The later observation enters the top 5. event_ordering All@5 is 0/40 and its pool holds 39 of 40 golds, so the series partner is the missing gold. The cap literals stay. The case-tablet triple (CDLI 258601) supplies the year component the period is added to.

Math: `exempt iff exists selected s and a third turn p such that date(ti) == date(s) + period(p), with period(p) != 0; literals 1, 2, 3 unchanged`

Tag `RECALL/INDEX` on BEAM event_ordering; BEAM temporal_reasoning; LOC Temporal; LME temporal-reasoning (event_ordering pool_not_k 17; temporal pool_not_k). Hook `crates/vestige-core/src/retrieval.rs:5676-5679`. Record: https://cdli.earth/artifacts/551464.json.

Combination: dotted date triple (S1-040) + dossier admission (S1-001).

Dedupe: Diary calendar match multiplies a session score when a month-name window contains session_ts. This adds an integer that is stored on a turn. No month-name list, no planet table, no session_ts. S1-073 exempts on year_depth inequality. This exempts on an exact sum with a third turn's integer.
