# Shard 2 algorithms

Research and design only. No engine code is changed. Base `e2ecd1da037974c1835f563ad8b046f4a4ca3629` on `perf/c2-step33-triangulation`. Ninety rules, S2-001 through S2-090. Seventy combine two mechanisms. Fifty-two are RECALL/INDEX and thirty-eight are ORDERING.

The target is 90% on every metric of LME, LoCoMo, and BEAM, and on every category inside them. The floor is Step 34 (`62041ef5`) for Turn@1 and the attached bd2693e2 dump for every other cell. Slot-1 reorder clears LME Turn@1 at the bench (Turn@5 is 428, the bar is 423) and does not clear LoCoMo Turn@1 (1127 against 1383), BEAM Turn@1 (238 against 320), either All@5, or BEAM Session@1 (Session@10 is 275 against 320). The per-metric ceilings are in `funnel-S2.md`.

Open-Domain slot-1 reorder stops at 35/92 on the attached dump (19/92 on this HEAD after Step 33, top 5 unchanged). The wide pool holds 78. The category bar is 83, so at least five of the fourteen turns outside the pool have to be admitted. Summarization slot-1 reorder stops at 17/36. Official k holds 25. The wide pool holds 34. All@5 is 0. The bench LoCoMo bar is a different number: pool@50 already holds 1415 of 1536, which is above 1383, because Single-Hop, Multi-Hop, and Temporal carry the spare.

Turn ordering attaches after `apply_c2_step33_triangulation` and before `maybe_dump_replay` (`retrieval.rs` 6265–6272). Session ordering and session insertion attach where `final_session_indices` is built (`retrieval.rs` 5480–5600), because nothing after that list is created rewrites it. Index features are computed while the prose is still in hand at the `TurnIx` push (`retrieval.rs` 3334–3384) and are read back only as integers and `DetHashSet`s. Nothing reads `TurnIx.body`. Nothing writes `rsb_seat_behind`.

## S2-001 foreign_midcap_over_echo

P.Oxy. VII 1021 (TM 20321) is Nero's accession notice. The fetched text spends its first clause on the dead Caesar and the expected ruler, then names Nero, then closes with a date. The name that answers a descriptive question is the mid-sentence capital that the formula does not already contain. Among the current top 5, keep only turns that contain every query content term whose document frequency is at or below the median df of the query's content terms. If exactly one of those turns has an entity id that is not among the query terms, move it to slot 1. Ties and an empty rare-term set do nothing. This is the 4:28 shape: the gold was already in the top 5, and the query's words are the paraphrase.

Math: Let R be query term ids with df(id) <= median(df of query content ids). Let S be top5 turns whose content contains R. Seat t in S iff entities(t) - query_ids is nonempty and every other s in S has that difference empty.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.oxy/7/1021.xml

## S2-002 protocol_party_after_digits

P.Yadin 15 (TM 23494), fetched as Historical/p.babatha/15.xml, opens with Hadrian's titulature, the consuls of the year, the provincial era of Arabia, and the Macedonian month, and only then names Babatha and the guardians. The capitals inside that opening are the date, not the answer. At ingest, while the prose is still in hand, store the seq index of the first digit token and the seq indexes of entity tokens. In the top 5, if the head's entity indexes all fall before its first digit and exactly one other turn shares the query's rarest term and has an entity index after its first digit, that turn takes slot 1.

Math: party_ok(t) = exists entity index e with e > first_digit_index(t). Swap iff party_ok(head) is false, exactly one top5 challenger is party_ok, and that challenger contains argmin_df(query content ids).

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.babatha/15.xml

## S2-003 interior_copy_shorter

P.Yadin 16 (TM 23496) is a collated copy. The fetched text repeats the preface that the declaration was copied and checked against the notice posted in the basilica, then gives the census. Two turns that carry the same entity set are the two copies. The interior text of a double document is the shorter sealed copy. If two top-5 turns have equal entity sets and unequal content-set sizes, seat the smaller. A third turn with the same entities, or equal sizes, blocks the swap.

Math: If exactly one unordered pair in the top 5 has entities(a)==entities(b) and |content(a)| != |content(b)|, seat the member with the smaller content set. Size is the cardinality of TurnIx.content.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.babatha/16.xml

## S2-004 witness_tail_not_party

P.Cair.Zen. I 59003 (TM 665) sells a girl named Sphragis, aged seven, for fifty drachmas, then lists witnesses who are cleruchs of Toubias, each with a father's name and a city. The fetched text marks the witness section with μάρτυρες. The ranker cannot key on that word. At ingest, if a turn has at least four entity ids, the first two stay in the person-match set and the rest are stored in witness_ids, a DetHashSet that is not consulted for a query with no entity of its own. A query that does contain an entity still matches witness_ids, so a question that names a witness can find the contract. The sale's operative person is the min-df entity among the first two.

Math: person_ids(t) = first two entity ids in seq order when |entities|>=4, else all entities. witness_ids = the remainder. WH-query overlap uses person_ids. A query that already contains an entity uses entities.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.cair.zen/1/59003.xml

## S2-005 keeper_is_last_of_six

The same sale names a run of witnesses and the practice, visible in the six-man Greek witness protocol this text is an instance of, is that the syngraphophylax is the last of them. S2-004 already stops the tail from answering a query that names nobody. This rule is the opposite gate, and it is separate: when the turn has at least six entity ids, the last id is stored as keeper_id. A query whose rarest term equals keeper_id admits the turn even if person_ids miss. A turn with fewer than six entities gets keeper_id None and this rule does not fire.

Math: keeper(t) = last entity id if |entities|>=6 else none. Admit t into the pool when rarest(query)==keeper(t) and t_scene(t)==0, with score equal to the integer df rank of that id, capped below any turn that already contains the full rare set.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.cair.zen/1/59003.xml

## S2-006 two_denomination_larger

P.Yadin 11 (TM 23489), from En-Gedi, is Judah's acknowledgement that he has received a loan of forty, corrected in the text to sixty, Tyrian denarii, which are fifteen staters, on the security of a courtyard. Two numbers name one debt. At ingest, parse ASCII integers and the existing NUMBER_WORDS closed class into a Vec<u16> in order. The principal is the maximum. The smaller figures are conversions and do not answer a query digit. A query digit matches the turn only when it equals the maximum. The entity after the last number is stored as pledge_id and matches only together with that maximum.

Math: principal(t)=max(numbers(t)). Query digit d matches t iff d==principal(t). pledge matches only if the query also contains principal(t) or contains no digit.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.babatha/11.xml

## S2-007 guardian_second_entity

P.Yadin 14 (TM 23492) is a summons. The fetched text names a guardian acting for an orphan, and a hearing before the governor Iulius Iulianus. In a turn with exactly two entities and at least one digit, the second entity is the ward and the first is the representative. For a query with no entity, the top-5 turn whose second entity has the lowest df among such second entities takes slot 1, if that turn also contains the query's rarest term. Queries that already name an entity skip this rule.

Math: ward(t)=entities[1] if |entities|==2 and form_digit else none. Among top5 with ward defined and rarest(query) in content, seat the unique minimum of df(ward).

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.babatha/14.xml

## S2-008 request_tail_sentence

P.Enteux. 1 (TM 3279) is Patron's petition to Ptolemy. The fetched text addresses the king, narrates the wool he bought and the price he is being charged a second time, and closes by asking that the petition be sent on. A broader OpenAlex search for petition narrative versus the final request returned the paper 'Finding the Plea', which tests whether language models can label rhetorical parts of historical petitions. This rule does not label parts with a model or a word list. At ingest, split on orthographic sentence-final punctuation. The last sentence's term ids are stored as request_ids. For a query that contains a closed-class WH token, request_ids count twice in the integer overlap and earlier sentences count once. The overlap is a sum of integers, not a new weight knob on an existing feature.

Math: score(t)=|content(t) ∩ query| + |request_ids(t) ∩ query|. Used only as a top-5 tie break when the WH gate is on. No scalar multiplier.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.enteux/1.xml

## S2-009 earnest_versus_price

P.Enteux. 2 (TM 3280) is Harmeusis's petition. He bought 118 fleeces at four drachmas and five obols, paid an earnest of 76 drachmas, and the seller sheared and took the wool. The fetched text carries both numbers. The earnest is the smaller sum that changes hands before the goods. Store numbers in order. If the last number is strictly smaller than the max of the earlier numbers, the last number is earnest_n and does not satisfy a query digit. The max remains the price. This differs from S2-006, which keeps the max of a two-name conversion of one debt. Here the smaller figure is last, and it is ignored only in that position.

Math: if numbers.len()>=2 and numbers[last] < max(numbers[..last]) then earnest=numbers[last] and it is excluded from digit match. else no earnest.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.enteux/2.xml

## S2-010 two_cities_one_estate

P.Enteux. 8 (TM 3285) is Aristomachus's petition about a gymnasium dedicated at Samareia and property left both in Alexandria and in the village already named. The fetched text carries two place-like capitals and one act. A query that names neither place should follow the rarer capital. At ingest the entity ids are already known. The answer entity for a query with no entity is the entity with minimum df. The maximum-df entity is stored as file_place and is not sufficient, by itself, to put the turn in the pool. Both ids remain in content, so a query that actually contains the city still matches. The same higher-df versus lower-df test reorders final_session_indices at retrieval.rs 5480-5600. The session whose distinctive entity has the lower df takes session slot 1 when the head session overlaps the query only on the higher-df entity and the lower-df session is already inside the top 5 sessions. LoCoMo Session@5 is 1416 and the bar is 1383. A turn permutation after line 5600 cannot move Session@1.

Math: file_place(t)=argmax df(entities). answer_ent(t)=argmin df(entities). Pool admission on a zero-entity query requires overlap with content minus {file_place}.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; LoCoMo Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.enteux/8.xml

## S2-011 vineyard_task_clauses

P.Oxy. XIV 1631 (TM 21943) is a one-year lease of vineyard work. The fetched XML names Aurelius Serenus, the lessees with their mothers, the village of Tanaeis, and then the works. A summary of that contract is the list of tasks, not the filiation block. At ingest, find the first digit that is a one-year term (the number 1, or the NUMBER_WORDS value one). Clauses after that digit, split on comma or semicolon, are task_spans, each a DetHashSet of term ids. On a query with no WH token and no question mark, admit one seated candidate per task span: the turn itself once for the span with the most ids outside the query, and do not also seat the filiation span. The span count is capped at 8.

Math: n = min(8, task_spans.len()). Seat n copies are not created. The turn is admitted once, and its rank key is the max over spans of |span - query|. Filiation ids, those before the term digit, contribute 0 to that key.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.oxy/14/1631.xml

## S2-012 member_count_equals_names

P.Oxy. XIV 1626 (TM 21939) is an agreement of dekanos from Paneuei, with a guarantor, to supply one staff-bearer's post for animals sent to Babylon. The fetched text lists the partners and states that they are all the dekanos. Where a turn contains a closed-class cardinal, from the existing NUMBER_WORDS table, whose value equals the count of entity ids before that cardinal, those entities are joint parties of one act. A query that matches any one of them admits the turn. A cardinal that does not equal the entity count is not a group mark and this rule stays off. The OpenAlex query is broad and its top hits are retrieval textbooks, not this equality.

Math: Let c be the first NUMBER_WORDS value in the turn. Fire only if c == |entities before that token| and 2<=c<=12. joint_ids = those entities. Any query entity in joint_ids admits the turn.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.oxy/14/1626.xml

## S2-013 nilometer_additive_triple

P.Oxy. XVI 1830 (TM 37843) reports the blessed river to Calus the chartularius. The fetched text gives a rise of 12 fingers from 5 to 7 Mesore, then 2 cubits 20 fingers, then an addition of 16 fingers, and a total of 5 cubits. One later integer is the sum of earlier integers. At ingest, if some value equals the sum of two earlier values in the same turn, set additive=1 and store the total. A WH query with no digit of its own admits at most one additive turn from the session that already contains the query's rarest term. The admitted turn's pool score is the total, as an integer, which orders it under any turn that already matches two rare terms.

Math: additive(t)=1 iff exists i>j, i>k, j!=k, with numbers[i]=numbers[j]+numbers[k], all u16, no overflow past 10000. Admit the unique such t in the anchor session.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.oxy/16/1830.xml

## S2-014 two_village_rarer

P.Oxy. XVI 1831 (TM 35593) tells the headman of Takona that shepherds of his village fought the men of Tholthis and drove sheep onto their land. Two place entities, one event. In the top 5, if the head's only overlap with the query is its higher-df entity and a challenger overlaps on the lower-df entity of the same turn's session, seat the challenger. The rule fires only when both turns have exactly two entities. The folksonomy gazetteer paper returned by the related search is not this test: no external place list is loaded. The same pair of entities reorders final_session_indices at retrieval.rs 5480-5600. When both sessions are already in the top 5, the session that carries the lower-df entity takes session slot 1 if the head session carries only the higher-df entity. That ceiling is Session@5. It clears the LoCoMo bench bar and does not clear Multi-Hop (246, need 254) or Open-Domain (65, need 83).

Math: Let d1<d2 be df of the two entities. Seat the unique top5 turn that contains the d1 entity and the query's rarest non-entity term, if the head contains only the d2 entity from that pair.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1; LoCoMo Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.oxy/16/1831.xml

## S2-015 synkollesimos_date_fences

P.Fam.Tebt. 15 (TM 10732) is Isidorus's letter to the strategos Sarapion. The fetched text quotes the strategos's order to inspect a synkollesis, a pasting-together, of the books of the public archive. A pasted roll is many acts under one cover. At ingest, a line that contains a digit and is shorter than eight tokens starts a new section. Sections do not inherit each other's entity ids. On a non-WH query, the session may send one turn, or one section of a long turn, per section, capped at the number of sections, and slot 1 is the section with the largest count of terms below median df. Header tokens are not copied onto the body, which is the difference from a roster header.

Math: sections = runs split at lines with a digit and token_count<8. Quota = min(sections.len(), 8). Rank key of a section = count of its term ids with df <= corpus median. Slot 1 = unique max of that key.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.fam.tebt/15.xml

## S2-016 signalment_before_dowry

P.Fam.Tebt. 2 (TM 10737) opens with Domitian's year and then identifies Chaeremon, about forty-seven, scar on the left wrist, his son Orsenouphis, about twenty-one, scar at the left canthus, and Tepheros, about forty, scar on the left brow. The fetched text puts the ages and scars before the disposition. An age next to a scar-shaped token is a signalment, not the price. A scar-shaped token is orthographic: a short token in the same clause as a number under 100, with no currency-sized number (100 or more) in that clause. Numbers in a signalment clause do not match a query digit. The first number that is at least 100, or that follows the signalment clauses, remains the sum. The related OpenAlex query is polluted by the word scar and its top hits are medical; the tighter query 'marriage contract ages and scars before the dowry retrieval' returns work on jewellery and marriage, not this split.

Math: signalment_clause = a sentence containing a number n with n<100 and no number >=100. Digit match ignores numbers inside signalment clauses.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.fam.tebt/2.xml

## S2-017 dowry_village_block

P.Fam.Tebt. 1 (TM 10726) is fragmentary. The fetched text still shows a village, Talei, a yearly public charge, and a dowry in silver, in separate stretches. On a non-WH query, a top-5 turn whose content is a proper subset of another top-5 turn in the same session loses slot 1 to that longer turn when the longer turn's extra ids include a digit-bearing term. The fragment is the excerpt. The longer turn is the contract. This is not a length tie-break by itself: the subset relation and the extra digit are both required.

Math: If content(head) is a proper subset of content(t) for exactly one other top5 t in the same session, and t has a number that head lacks, seat t.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.fam.tebt/1.xml

## S2-018 grain_account_total_line

P.Cair.Zen. I 59004 (TM 666) lists artabas of wheat and flour at Jerusalem, Jericho, Abella, and other places, then a total, γίνονται ἀρτάβαι. The fetched text is an account, not a narrative. On a non-WH query, if the head is an additive turn (S2-013's mark) and another top-5 turn in the same session is not additive and contains more entity ids, seat the entity turn: the summary of an account that asks what happened wants the places. On a query that contains a digit, do the opposite and seat the additive turn. The two gates cannot both fire.

Math: If query has a digit: seat the unique additive top5 turn. If query has no digit and no WH: seat the unique non-additive same-session top5 turn with the larger entity count, provided the head is additive.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.cair.zen/1/59004.xml

## S2-019 jar_ledger_by_ship

P.Cair.Zen. I 59015 recto (TM 676) counts Milesian and Samian jars and half-jars, then says which were loaded on Theon's boat and which on Aeropus's. The fetched text is a ledger with a repeated unit and a changing carrier. At ingest, a repeated entity that occurs at least twice is the unit, and each other entity that is followed by a number starts a lot. On a non-WH query, admit every lot-entity's turn section, one seat per lot, capped at 6. Slot 1 is the lot whose number equals the query digit, or if the query has no digit, the lot with the largest number. The verso of the same papyrus is a second copy and is not merged.

Math: unit = entity with count>=2. lots = (entity, number) pairs where the entity is not the unit. Quota = min(6, lots.len()). Slot 1 = lot whose number equals the query digit, else max number.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.cair.zen/1/59015r.xml

## S2-020 citizenship_edict_status

P.Giss. I 40 (TM 19436) is the Greek copy of Caracalla's edict. The fetched text has the emperor's name and then the grant of Roman citizenship to those in the empire. A status grant has many high-df tokens and one low-df noun that is not an entity in the titulature. In the top 5, if the head's overlap with the query is only entity ids, and a challenger overlaps the query on a non-entity term whose df is below the median, seat the challenger. The rule does not fire when the query itself contains an entity, because then the name may be the question.

Math: Fire only if the query has no entity. Seat the unique top5 turn that contains a query term with df below the median, if the head's query overlap is confined to entity ids.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://papyri.info/editions/p.giss/1/40

## S2-021 village_aroura_under_header

P.Giss. I 7 (TM 19409) is a survey from the Apollonopolite Heptakomia. The fetched text names the village Poeis, then royal land at so many arouras, then the same village again, then Teruthis. A figure means nothing once it is pulled off its village. At ingest, a line with an entity and no number is a header. Following lines that contain a number and no new entity inherit that entity id as their section key, stored beside the number, not copied into the line's lexical bag. A query digit matches a number only together with its section key when the query also has an entity. A query with a digit and no entity matches the number and returns the section key as the answer entity.

Math: section_of(number) = nearest preceding entity line. Match(d, e) requires section_of(d)==e when the query has both. When the query has only d, the returned entity is section_of(d).

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.giss/1/7.xml

## S2-022 second_registration_longer

P.Giss. I 8 (TM 19410) says a price was paid and a sale made by an autograph cheirograph in the name of Harpokras, and that today a second person has been registered in the property archive with a copy of a cheirograph for the same plots. The fetched text is the warning that two writings exist. If two top-5 turns share an entity and a number, and one has a strictly larger content set, seat the larger one when the query contains that number. If the query contains no number, seat the smaller one. This is the reverse of interior-copy, which seats the shorter text when the entity sets are equal and no number is required.

Math: Pair a,b in top5 with entities intersecting and numbers intersecting. If the query contains the shared number, seat the larger content set. If not, seat the smaller. Exactly one such pair, or no swap.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.giss/1/8.xml

## S2-023 dream_header_then_body

UPZ I 77 (TM 3468) records two dreams. Each fetched entry starts with τὸ ἐνύπνιον ὃ εἶδεν, the dreamer's name, and a date, then the dream. Tages descends a street and counts nine houses. A second header names Ptolemaios. At ingest the first sentence, if it contains an entity and a number and is shorter than the rest of the turn, is the header. Header term ids and body term ids are stored apart. A query that contains a digit or an entity uses only the header ids. A query with neither uses only the body ids. The other half scores 0 for this feature. Both halves remain in content for ordinary overlap, and this feature is a top-5 ordering key only.

Math: key(t)=|header ∩ query| if query has a digit or an entity, else |body ∩ query|. Seat the unique strict maximum in the top 5. Ties do nothing.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/upz/1/77.xml

## S2-024 verso_continues_recto

UPZ I 3 (TM 3394) is Ptolemaios's petition from the Serapieion. He has been in katoche nine years. Herakleia took refuge in the sanctuary. Zoilos the staff-bearer handed her over. UPZ I 4 (TM 3395) is the continuation on the verso and repeats those names. The fetched pair shares two entities and the second turn has fewer entities. At ingest, if turn i and turn i+1 are adjacent in one session, share at least two entity ids, and i+1 has a smaller entity set, the content ids of i+1 are stored as a continuation set on i. A query that matches i only together with one continuation id admits i, and i+1 is not also seated. The link is adjacency plus a shared pair, not a quote mark and not a catch-line.

Math: continue(i)=content(i+1) if |entities(i) ∩ entities(i+1)|>=2 and |entities(i+1)|<|entities(i)| and session matches and index is adjacent. else empty. Admit i when (content(i) ∪ continue(i)) covers the rare set and content(i) alone does not.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/upz/1/3.xml

## S2-025 birthday_sender_not_address

T.Vindol. II 291 (TM 114529) is Claudia Severa's invitation to Sulpicia Lepidina for her birthday on 11 September. The fetched text names Severa first, then the date, then asks Lepidina to come, and the closing greetings name Cerialis. Three capitals, three jobs. At ingest on a turn with exactly three entities and a digit: sender is the first, host is the entity nearest after the first digit, addressee is the remaining one. A query with no entity that contains the rarest non-entity term seats, in the top 5, the turn whose host entity has the lowest df. A query that contains an entity skips the rule.

Math: host = entity with the smallest seq index greater than first_digit, if |entities|==3 and form_digit. Seat unique min df(host) in the top 5 among turns that contain rarest(query).

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/t.vindol/2/291.xml

## S2-026 strength_residual

T.Vindol. II 154 (TM 114419) is the strength of the First Cohort of Tungrians under Iulius Verecundus. The fetched text gives the pure number, the absentees at London and elsewhere, the sum of absentees, and the remainder present, including the sick. The first large number is the strength. A later number that equals the strength minus another number is the remainder. At ingest, if numbers[0] > numbers[i] and numbers[0] - numbers[i] equals some numbers[j], set residual = numbers[j] and strength = numbers[0]. On a non-WH query with no digit, the summary seat is the turn, once, and its rank key is the strength, so detail lines of the same session that lack this identity lose to it. On a query with a digit, only strength and residual match. The intermediate postings do not.

Math: Fire if exists i,j>0 with numbers[0]>numbers[i] and numbers[0]-numbers[i]==numbers[j]. Match digits only against {numbers[0], numbers[j]}. Summary rank key = numbers[0].

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/t.vindol/2/154.xml

## S2-027 undated_note_yields

T.Vindol. II 234 (TM 114481) is Flavius Cerialis to September: a few words about tomorrow's weather and a date. The fetched text is one clause. A turn with fewer than 20 content tokens and no digit is a note. On a non-WH query, a note at slot 1 yields to the unique other top-5 turn in the same session that has a digit or at least twice the content count. If several qualify, there is no swap. The note stays in the top 5.

Math: note(t)=content.len()<20 and numbers empty. Swap the head only if it is a note and exactly one other same-session top5 turn has a number or content.len()>=2*head.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/t.vindol/2/234.xml

## S2-028 century_blocks_between_dates

Rom.Mil.Rec. I 1 (TM 44832) is a Dura roster of AD 219. The fetched text repeats consul pairs and then the soldiers of a century: Aurelius Septimius Danymus, Aurelius Cocceius, and the rest. A summary of the unit is one soldier from each century, not the first thirty names of the first century. At ingest, a line that contains two entities and a digit and no other content is a date header. The following entity-only lines belong to that block. On a non-WH query, admit one entity from each block, the lowest-df entity in the block, capped at 8 blocks. Slot 1 is the block whose header digit is unique. Blocks do not inherit the consul names as lexical features.

Math: blocks split at lines with exactly 2 entities and at least 1 digit. Representative = argmin df(entity) in the block. Quota = min(8, blocks). Slot 1 = the representative of the block whose digit is unique in the turn.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/rom.mil.rec/1/1.xml

## S2-029 stipend_list_under_opinion

Rom.Mil.Rec. I 2 (TM 44833) is headed ad opinionem stipendii and then lists men by century and origin. The fetched text is a pay list: one header, many names. A query that matches only the header's high-df tokens must not retrieve every name equally, and a query that matches one name must retrieve that name's line rather than the header. At ingest the first line is header_ids. Each later entity is a row. A query entity matches only its row. A query with no entity matches the header and seats one row, the first, so the list does not flood the pool. This is a cap of one, not a copy of the header onto the rows.

Math: If the query contains an entity, match rows whose entity is in the query, cap 1 (the lowest df). If not, admit the turn once. Never admit more than one row from this turn.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/rom.mil.rec/1/2.xml

## S2-030 pridianum_summa

Rom.Mil.Rec. I 63 (TM 69875) is a pridianum. The fetched text names the cohort, the prefect Arruntius, the sum of soldiers, and the centurions and decurions. The same subtraction identity as S2-026 may hold. This rule is the ordering half, for a turn that already carries residual=1. On a non-WH query, a residual turn in the top 5 takes slot 1 over a non-residual turn when both share the query's rarest term. It does not admit anyone. S2-026 is the admission rule for the Vindolanda tablet. Using both on one turn is allowed because one writes the mark and the other consumes it. A turn without the mark is unchanged.

Math: If the head is not residual and exactly one top5 turn is residual and contains rarest(query), seat it. Requires plan.has_wh == false.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/rom.mil.rec/1/63.xml

## S2-031 found_ostracon_one_line

O.Claud. I 1 (TM 29810) says that whoever finds this ostracon will give a stater, and farewell. The fetched text is one sentence and one sum. A turn with at most 15 content tokens, exactly one number, and no entity is a finder notice. A query with no entity and no digit admits at most one such notice from the whole corpus, the one in the session that already contains the query's rarest term. If the anchor session has none, the rule does nothing. It does not admit every short turn.

Math: notice(t)=content.len()<=15 and numbers.len()==1 and entities empty. Admit the unique notice in the session of the turn that contains rarest(query).

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/o.claud/1/1.xml

## S2-032 water_letter_single_claim

O.Claud. I 2 (TM 29811) is Antistius Flaccus to Calinius: greetings from Raima, then the claim that the water has risen, then the date. The fetched text has two entities and one finite clause after them. At ingest, if a turn has exactly two entities and exactly one number, the term ids after the second entity and before the number are claim_ids. In the top 5, a WH query that contains none of the two entities seats the turn with the larger claim overlap, if it is unique. Entity overlap on this feature is 0, so the greeting cannot win it.

Math: claim_key(t)=|claim_ids ∩ query|. Seat the unique strict max in the top 5 when the query has a WH token and no entity.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/o.claud/1/2.xml

## S2-033 morning_report_partition

O.Bu.Njem 2 (TM 73152) gives the number at Golas on 15 June, 59, then the librarius, the optio, 21 quintanarii, stationarii, men at the bath, the sick, and the remainder. The fetched text is a partition of the first number. If the sum of the numbers after the first equals the first, or equals the first minus the last, the turn is a parade state. On a non-WH query the turn is admitted once, ranked by the first number, and the duty lines are not separate seats. A query digit matches the first number or the final remainder only.

Math: partition(t)=1 iff sum(numbers[1..])==numbers[0] or sum(numbers[1..len-1])==numbers[0]. Digit match uses {numbers[0], numbers[last]} only. Summary seats the turn once.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/o.bu.njem/2.xml

## S2-034 bu_njem_short_state

O.Bu.Njem 1 (TM 73151) is the short state: 9 February, number 47, and a remainder of 15. The fetched text has two numbers and almost no names. A turn with exactly two numbers, fewer than 25 content tokens, and no entity is a bare state. It is admitted to a non-WH query only when the anchor session, the session of the rarest query term, contains no partition turn of the S2-033 kind. The bare state is the fallback summary, one seat, and it loses to a partition turn in the same session.

Math: bare(t)=numbers.len()==2 and entities empty and content.len()<25. Admit iff the anchor session has no partition turn and has exactly one bare turn.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/o.bu.njem/1.xml

## S2-035 two_year_deposit

P.Ryl. II 125 (TM 12911) is Orsenouphis's report to the chief of police at Euhemeria. In year 14 of Tiberius, while a wall was being taken down, Petesouchos found what the writer's mother had put away in a casket in year 16 of Caesar. The fetched text has two year numbers and a list of gold earrings. The smaller year-like number is the earlier deposit only when both numbers are at least 2 and at most 50 and they differ. Store deposit_year = min and find_year = max. A query with no digit that overlaps the object terms uses deposit_year as the answer number. A query digit matches deposit_year first. The later year matches only if the query also contains a second digit.

Math: if two numbers in 2..=50 and they differ: deposit=min, found=max. Single query digit matches deposit only.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.ryl/2/125.xml

## S2-036 bearer_is_second_name

P.Herm. 1 (TM 25863) is Herakles to Musaios, asking him to receive Dioskoros, who will hand him the letter. The fetched text has two entities in a short turn and a date. In a turn with exactly two entities and fewer than 40 content tokens, the second entity is the bearer. A query with no entity seats, in the top 5, the unique turn whose bearer has the lowest df and which contains the query's rarest term. This is the two-entity case. S2-025 is the three-entity case with a digit between the names. They do not both apply to one turn.

Math: bearer=entities[1] if |entities|==2 and content.len()<40. Seat unique min df(bearer) among top5 turns that contain rarest(query), query entities empty.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.herm/1.xml

## S2-037 news_after_health_wish

P.Herm. 5 (TM 21124) spends its opening on joy, health, and a prayer, and the fetched text only then reaches the point of the letter. The opening sentence of a turn whose every content term has df above the corpus median is a wish. Wish ids are stored and are not used for pool admission. Admission uses the later sentences. This is not the Old Babylonian rule, which removes such a sentence from BM25 document length. Length is unchanged. The wish ids are simply ineligible as the only overlap that admits a turn. If every sentence is a wish, the turn is unchanged.

Math: wish_sentence = a sentence whose content ids all have df > median. admit_ids = content ids outside wish sentences. A turn is not admitted on wish ids alone.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.herm/5.xml

## S2-038 consular_and_regnal_clocks

P.Kellis I 1 (TM 20272) dates by the ninth year of Diocletian, the eighth of Maximian, the first of the Caesars Constantius and Galerius, and the consulate. The fetched text stacks two clocks. Numbers in the opening, before the first entity that is not inside a run of entities, are regnal. The number nearest a pair of entities at the end of that opening is consular. A query with one digit matches the consular number if both exist, and the regnal numbers only when the query has two or more digits. The prefect named after the date is a party, under S2-002, and is not a clock.

Math: if at least two numbers occur before the first non-entity content token after a run of entities: consular = last of those numbers, regnal = the others. A single query digit matches consular only.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.kellis/1/1.xml

## S2-039 temple_account_monthly

BGU I 1 (TM 11853) is a temple account: boat dues, beer, linen vestments on named days, incense, and a total of 1901 drachmas and change. The fetched text repeats a month-name shape and an amount, then γίνονται. Month names are not a list this rule may add. A line with a number and no entity, followed by another such line, is a charge line. The additive total, if S2-013's mark is present, is the closer. On a non-WH query the summary admits one charge line per distinct number, capped at 6, and the total line is slot 1 only when the query contains a digit. Otherwise slot 1 is the charge line with the lowest-df neighboring token.

Math: charges = numbers that fail the additive-total test. Quota = min(6, charges.len()). Slot 1 = the total if the query has a digit, else the charge whose neighbor term has minimum df.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/bgu/1/1.xml

## S2-040 unanswered_carrier

P.Mich. VIII 466 (TM 17240) is Iulius Apollinarius to his father Iulius Sabinus. The fetched text says he has written often by Saturninus the signifer, by Iulianus son of Longinus, and by Dius, and has had no answer. The carriers are entities in the middle of the letter, not the addressee and not the writer. At ingest, entities that are neither first nor last are carrier_ids when the turn has at least four entities. A query entity that hits a carrier admits the turn even if the first and last entities miss. A query with no entity does not use carrier_ids, so the father and the son remain the person match, as in S2-004.

Math: carriers = entities[1..len-1] if |entities|>=4. Query-entity hit on carriers admits. Zero-entity queries ignore carriers.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.mich/8/466.xml

## S2-041 expense_lines_then_sum

P.Col. IV 63 (TM 1727) is an account of bronze drachmas: papyrus, wages, a bank, plants, and a string of small sums. The fetched text is items then figures. On a non-WH query, an additive turn already in the top 5 takes slot 1 if the head is in the same session and is not additive. This is the account's total against its own items, and it applies only inside one session. S2-018 makes the opposite choice when the query has no digit and the non-additive turn has more entities. The two rules are ordered: S2-018 runs only when the non-additive turn has more entities. This rule runs when it does not, so a pure account, whose items are numbers rather than places, keeps its total at slot 1.

Math: If head is not additive, exactly one same-session top5 turn is additive, and that turn has entity count <= head's, seat the additive turn. Non-WH only.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.col/4/63.xml

## S2-042 receipt_mona_correction

SB I 1 (TM 32064) acknowledges goods received from Soterichos, Dionysios, and Paskapais: bundles, ropes, chests, with a count written and then corrected, μόνα τριάκοντα δύο against an earlier thirty-six. The fetched text states both numbers. A correction is a number immediately followed, within two tokens, by a different number, where the second is smaller. The second number replaces the first for digit match. The first remains in the turn but does not satisfy a query digit. The three suppliers stay joint parties under S2-012 if a cardinal equals their count. The correction rule itself only rewrites which integer is visible.

Math: if numbers[i+1] < numbers[i] and the tokens are adjacent within 2, visible digit is numbers[i+1] and numbers[i] is hidden from digit match.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/sb/1/1.xml

## S2-043 accession_name_after_formula

The same proclamation, P.Oxy. VII 1021, closes with the date and the name. S2-001 uses it for a query that already has rare terms. This rule is the summary side. On a non-WH query, a top-5 turn whose last sentence contains an entity and a digit takes slot 1 over a head whose last sentence contains neither, if both turns share the session. The proclamation's last sentence is the date plus the fact that the notice was given. A summary of an edict wants that close, not the relative clause about the predecessor.

Math: close(t)= last sentence has at least one entity and one digit. If head is not close and exactly one same-session top5 turn is close, seat it. Non-WH only.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.oxy/7/1021.xml

## S2-044 appeal_quotes_nomination

P.Oxy. IX 1204 (TM 21590) is Aurelius Ploutarchus's appeal. He was nominated to a dekaproteia by Aurelius Demetrianus while he was in the Small Oasis, and he appeals through his father. The fetched text names the nominator, the office, and the absence. A quoted act inside a petition is a run of entity ids that also occurs in an earlier turn of the same session. At ingest, if a later turn contains every entity of an earlier turn and at least two more content ids, the earlier turn's content is stored as quoted_ids on the later turn. A query that matches the later turn only by way of quoted_ids admits the earlier turn instead, and the later turn stays available one rank behind. The earlier turn is the nomination. The later turn is the appeal.

Math: if entities(early) subset of entities(late), same session, index(early)<index(late), and |content(late)| >= |content(early)|+2, then quoted(late)=content(early). A query whose overlap with late is confined to quoted_ids seats early first.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.oxy/9/1204.xml

## S2-045 prefect_window

P.Oxy. XVII 2105 (TM 17488) is a fragmentary edict of the prefect Mevius Honoratianus about a three-year contest. The fetched text names the prefect, the games, and a start on the seventh of the coming month. A window is two numbers in one turn with the second greater than the first by at most 30, and no third number between them. The window's start is the smaller. A query digit matches the start, not the end, unless the query has two digits. The prefect, the first entity, is the issuer and is the person match for a zero-entity query. The rule admits the turn when the query's rarest term is in the turn and the turn has a window, even if t_scene is 0, capped at one such turn.

Math: window if exists adjacent numbers a<b and b-a<=30. Match a single query digit to a only. Admit at most one window turn containing rarest(query).

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.oxy/17/2105.xml

## S2-046 granary_month_versus_year

P.Eleph. 10 (TM 5844) is Euphronios to Milon. The fetched text orders him to take what has come in from the bankers at the temples in Apollonopolis and the grain measured in, from the earliest time to the present, by month and by year, and to keep the years distinct. A yearly total is a number that equals the sum of at least three earlier numbers. A monthly line is a number that does not. On a non-WH query with no digit, admit the monthly lines, one seat per distinct number, capped at 6, and do not seat the yearly total. On a query with a digit, seat only the yearly total if the digit equals it, else the monthly line whose number equals the digit.

Math: year_total = a number equal to the sum of three or more earlier numbers. months = the others. Zero-digit summary uses months only. A query digit uses the total iff it equals the total.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://raw.githubusercontent.com/papyri/idp.data/master/Historical/p.eleph/10.xml

## S2-047 epitaph_son_sets_up

EDH HD000001, fetched live, is the Cumae epitaph of Nonia Optata daughter of Publius and C. Iulius Artemon, set up by their son C. Iulius Optatus, with the formula for freedmen and descendants. The commemorated names come first. The dedicator is the last entity. In a turn with at least three entities and a digit, a zero-entity query seats the top-5 turn whose last entity has the lowest df, if that turn contains the query's rarest term. The freedmen formula is a high-df tail and is not an entity. OpenAlex 'epitaph dedicator named after the deceased retrieval' returned 23 works. The top title is about literacy in Christian epigraphy of central Asia Minor, not this last-entity seat.

Math: dedicator = last entity if |entities|>=3 and form_digit. Seat unique min df(dedicator) in the top 5 among turns containing rarest(query), query entities empty.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000001

## S2-048 slave_altar_owner

EDH HD000004, fetched live, is an altar to Venus set up by a slave of Marcus Porcius Niger. The owner's name stands before the dedicant. In a turn with exactly two entities and one number, owner is the first entity and dedicant is the second. A query that contains the first entity admits the turn and the answer entity is the second. A query that contains the second admits it directly. A query that contains neither uses the second as the person, because the altar is the dedicant's act. The OpenAlex query returned one work, on musical instruments as votive gifts, which is not this seat.

Math: if |entities|==2 and numbers.len()==1: owner=entities[0], dedicant=entities[1]. Person match for a zero-entity query is the dedicant. A query that hits the owner still returns the turn, with dedicant as the extra entity.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000004

## S2-049 bodyguard_nation

EDH HD000011, fetched live, names Vetus, bodyguard of Nero Claudius Caesar Augustus, natione Batavus. The employer is a high-df name. The nation is the low-df name that follows. At ingest, in a turn with at least two entities, nation_id is the lowest-df entity that is not the first. A query with no entity that already matches the first entity's session through another term receives nation_id as the answer entity. The employer remains the person for a query that contains an entity equal to the first. The search's top hit is a message-understanding evaluation, not this split.

Math: nation = argmin df of entities[1..]. Answer entity for a zero-entity query is nation. The first entity matches only if the query contains it.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000011

## S2-050 joint_purchase

EDH HD000123, fetched live, says Q. Babullius Asclepius and Caecilia Syntrophia bought the tomb for themselves and their freedmen. Two entities, one verb, one act. If a turn has exactly two entities and exactly one number, both entities are joint buyers. A query that hits either admits the turn. The freedmen formula is not a third buyer because it contributes no entity id below median df. This differs from S2-012, which locks a group only when a cardinal equals the entity count.

Math: joint if |entities|==2 and numbers.len()==1. Hit if either entity is in the query. Both stay in the answer set.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000123

## S2-051 age_three_fields

EDH HD000256, fetched live, is Obsitia Leuca's stone for her daughter Felicitas, who lived four years, four months, and two days. Three small integers in a row are an age, not three events. At ingest, a run of three numbers each at most 120, with the second at most 12 and the third at most 31, is one age. Digit match against an age succeeds only if the query digit equals the first number, the years. The month and the day do not match a bare query digit. The mother is the last entity, the dedicator, and stays under S2-047. The deceased is the first entity.

Math: age if three consecutive numbers a,b,c with a<=120, b<=12, c<=31. Visible digit = a. b and c are hidden from single-digit match.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000256

## S2-052 sit_tibi_not_a_person

EDH HD000777, fetched live, reads Sabidia, 55 years, hic sita est, sit tibi terra levis. The formula is a run of one-letter or two-letter tokens after the number. At ingest, tokens of length 1 after the first number are formula marks and are not content ids. In the top 5, if the head's only query overlap is empty once formula marks are removed and a challenger still overlaps, seat the challenger. Sabidia's age remains a visible digit. The formula cannot be the reason a turn wins.

Math: formula token = length-1 token after the first number. Overlap for the seat ignores those ids. Seat the unique top5 turn whose non-formula overlap is nonempty if the head's is empty.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000777

## S2-053 brick_stamp_maker

EDH HD001234, fetched live, is a brick stamp of Ulpius Obsequens and Ulpius Valentinus, opus doliare from an imperial estate. A stamp is shorter than the building it marks. A turn with exactly two entities, at most 12 content tokens, and one number is a stamp. It is admitted when a longer turn in the same session contains both of its entities. The stamp is the maker. The long turn is the building. A query that hits either entity admits the stamp ahead of the long turn. If there is no long turn, the stamp is admitted alone, one per session.

Math: stamp(t)=entities.len()==2 and content.len()<=12 and numbers.len()==1. Linked long turn = same session, content.len()>stamp, both entities present. Query hit on either entity seats the stamp first.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD001234

## S2-054 patronym_after_name

EDH HD002468, fetched live, is Dis Manibus, Reginus son of Ciniucnatus. Two entities, no number beyond the opening. The second entity is the father. A zero-entity query in the top 5 seats the turn whose first entity has the lower df, not the second. A query that already contains the second entity still matches the turn, so a question about the father is not blocked. The rule only breaks a tie inside the top 5 when the query names nobody.

Math: if |entities|==2 and numbers empty: person=entities[0], father=entities[1]. Zero-entity top5 seat uses min df(person), not df(father).

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD002468

## S2-055 curavit_official

EDH HD004321, fetched live, names Virius Lupus Victorius, vir clarissimus, consularis of Campania, who saw to the work. The rank tokens are high-df. The name is low-df. In the top 5, if the head overlaps the query only on terms above median df and a challenger overlaps on a term at or below the median, seat the challenger. The inscription's useful token is the name, not the rank. This is the same df split as a status edict, but it fires inside the top 5 with no citizenship grant and no entity-count test.

Math: Seat the unique top5 turn that contains a query term with df<=median, if every query term in the head has df>median.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD004321

## S2-056 century_mark_heads_name

EDH HD018000, fetched live, is a century mark: centuria Atisi, and a break. A one-entity turn of at most six content tokens is a century mark. The next turn in the same session, if it has an entity, is the soldier, and the mark's entity is stored as his century_id. A query that hits the soldier also returns the century id. A query that hits only the mark admits the soldier turn, not the mark. The mark is a header. It is not copied into the soldier's lexical bag.

Math: mark(t)=entities.len()==1 and content.len()<=6. soldier = next turn in session with an entity. century_id(soldier)=mark's entity. A query hit on the mark's entity admits the soldier.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD018000

## S2-057 praetorian_filiation_tribe

EDH HD025000, fetched live, is L. Avaenius Passer, son of Lucius, of the Oufentina tribe, soldier of the first praetorian cohort, century of Q. Magonius. The fetched text stacks identity and posting. At ingest the entity ids in order are: name, father, tribe, cohort, century commander. With five entities, slots are fixed by position. A query with no entity returns the name, slot 0. A query that contains slot 2, the tribe, still returns slot 0 as the person and does not treat the tribe as a second person. A query digit, the cohort number, matches slot 3's neighboring number. OpenAlex 'praetorian filiation tribe then cohort retrieval' returned 0 works.

Math: if |entities|==5: name=0, father=1, tribe=2, cohort=3, century=4. Person answer = slot 0. A query entity in any slot admits the turn. The answer entity reported with the turn is slot 0.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD025000

## S2-058 two_face_patron_client

EDH HD030000, fetched live, names C. L. Severus as patron on one face and Modestus on the other, with a face break marked // in the transcription. At ingest a line that is only punctuation or a repeated slash starts a second face. Face 0's first entity is the patron. Face 1's first entity is the client. A query that hits the patron admits the client face's turn section as the answer, and the reverse. The two faces share one TurnIx but their entity bags are separate, so a match on one does not score the other's tokens. The broad patron-client search returned library-patron papers. The top titles are not a two-face seat.

Math: faces split on a slash-only line. patron=first entity of face 0, client=first entity of face 1. A hit on either admits the turn. Answer entity is the one the query did not contain, or the client if the query contained neither.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD030000

## S2-059 one_word_acclamation

EDH HD000048, fetched live, is the single word Vivant. A turn with exactly one content token is an acclamation. On a non-WH query it yields slot 1 to any other top-5 turn. If several exist, it yields to the one with the most content tokens, and a tie blocks the swap. The acclamation stays in the list. It is not deleted.

Math: if head.content.len()==1 and the query is non-WH, seat the unique top5 turn with maximum content.len(), ties unbroken.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000048

## S2-060 fragment_too_small

EDH HD000020, fetched live, is a fragment whose transcription is a broken line and the name Pica. A turn with one entity, no number, and fewer than five content tokens is a fragment. It yields slot 1 to the unique top-5 turn that contains that same entity and has more content tokens. If no such turn exists, the fragment stays, because it may be the only witness to the name. It never yields to a turn that lacks the entity.

Math: fragment(t)=entities.len()==1 and numbers empty and content.len()<5. Seat the unique longer top5 turn that contains fragment's entity.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000020

## S2-061 simonides_coseat_name

Cicero, De Oratore 2.353, fetched from Perseus, has the roof fall while Simonides is outside. The bodies cannot be named by their faces. The relatives identify them because Simonides remembers who sat where. A query with no entity and exactly one rare content term is a destroyed name. The turn that contains the rare term is the known seat. Admit at most one other turn in the same session, within four indexes, that has an entity the query does not contain. Prefer the one with the lowest-df entity. Do not walk a list of places. OpenAlex 'Simonides method loci information retrieval' returned the method-of-loci survey literature, including 'The Method of Loci: A Survey Review'. That literature is the ordinal walk, which this rule is not. The search 'identify bodies by seat Simonides retrieval' returned 48 works. The top title is about Beckett and trauma, not a co-seat admission.

Math: If the query has no entity and |rare terms|==1, let a be the turn containing that term. Admit t in the same session with |index(t)-index(a)|<=4, t!=a, entities(t)-query nonempty, choosing the unique minimum df of that extra entity. Cap 1.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi0474.phi037.perseus-lat1:2.353

## S2-062 peroration_subset

Aristotle, Rhetoric 3.13, fetched from Perseus, says the necessary parts are prothesis and pistis, and that the epilogue reminds. He also says the epilogue is omitted when the speech is short or the matter is easy to remember. On a non-WH query, a top-5 turn whose content ids are a subset of the union of earlier turns in its session, and which is shorter than that union, is the reminder. It takes slot 1. If the session has fewer than four earlier content ids, the matter is short and the reminder does not take slot 1, which is Aristotle's omission. OpenAlex 'Aristotle prothesis pistis epilogue reminder ranking' returned 0 works. 'only prothesis and pistis are necessary parts retrieval' also returned 0.

Math: reminder(t) iff content(t) is a nonempty subset of the union of content ids of earlier turns in the session, and |content(t)| < that union's cardinality, and the union has at least 4 ids. Non-WH: seat the unique reminder in the top 5.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:greekLit:tlg0086.tlg038.perseus-grc1:3.13

## S2-063 opponent_points_off_order

Quintilian, Institutio 11.2, fetched as the whole chapter from Perseus, says the speaker must remember what the other side said, not necessarily in the order it was said, but so as to place it at the opportune loci. Spoken order is session order. The opportune place for a summary is the turn that adds the most terms the query does not already share with the head. Among top-5 turns in the head's session, seat the one with the largest count of content ids outside the head's content, if that count is unique and positive. Do not sort the whole session. OpenAlex 'opponent points reseated off spoken order retrieval' returned 0 works. 'Quintilian opponent points opportune loci ranking' returned one work, on monastic formation, which is not this seat.

Math: key(t)=|content(t) - content(head)| for t in the top 5, same session as the head, t!=head. Seat the unique strict maximum if key>0. Non-WH only.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi1002.phi00111.perseus-lat1:2

## S2-064 prothesis_and_pistis_only

The same chapter says an epilogue is not necessary, and that Theodorus's extra narrative subdivisions are not new parts. For a WH query, the top 5 should prefer the turn that states a term and the turn that adds a second term, not a reminder. If the head is a reminder under S2-062's definition, it yields to the unique top-5 turn that is not a reminder and that contains the query's rarest term. The reminder stays in the list. This is the WH gate. S2-062 is the non-WH gate. They cannot both fire.

Math: If the query has a WH token and the head is a reminder and exactly one other top5 turn is not a reminder and contains rarest(query), seat that turn.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:greekLit:tlg0086.tlg038.perseus-grc1:3.13

## S2-065 partitio_quota

Cicero, Partitiones 1, fetched from Perseus, divides the art into the speaker's power, the speech, and the question, and says memory is the keeper of all of it. The speech has four parts. Narratio and confirmatio teach. Principium and peroratio move. A summary that keeps only one part has not kept the speech. On a non-WH query, take the best session by the existing session score, and from that session admit one turn of each shape that is present: a turn with a digit and an entity (narratio), a turn with neither (confirmatio), the earliest turn (principium), and a reminder under S2-062 if one exists (peroratio). Cap 4. Slot 1 is the narratio turn if it exists, else the confirmatio. Shapes are orthographic. No part-name list is added. OpenAlex 'partitio quota summarization retrieval' returned 0 works.

Math: From the unique best session, pick at most one turn per shape. Shapes are a partition of the session's turns, so a turn is in one shape. Slot 1 = narratio if present.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi0474.phi038.perseus-lat1:1

## S2-066 narratio_sedes

Partitiones 9, fetched from Perseus, calls narratio the explanation of the facts and the seat and foundation of belief. Inside the four seats S2-065 would admit, slot 1 must be the narratio shape, the turn with a digit and an entity, not the earliest turn. If that pair is already in the top 5, this rule seats the narratio turn without waiting for the quota to admit anything. A query with a WH token does not use it. The search returned five works on rhetorical devices in biblical commentary, not this ordering.

Math: Non-WH. If the top 5 contains a turn with form_digit and a nonempty entity set in the head's session, and the head lacks a digit, seat the unique such turn with the most entities.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi0474.phi038.perseus-lat1:9

## S2-067 judgment_before_order

De Inventione 1.19, fetched from Perseus, says the point of judgment is found before the speech is ordered, and that what is said first has to be drawn from what will be said later. The six parts are named there. The ranker does not look for their names. The iudicatio is the top-5 turn that contains the largest number of rare query terms. What is said first, the current head, yields if that turn is unique and is not the head. The later turn has already been found. The head is then the part that depends on it. The search 'six speech parts ordered after the judgment point retrieval' returned retrieval textbooks, not this rule.

Math: key(t)=|content(t) ∩ rare(query)|. Seat the unique strict max in the top 5. WH queries only. Rare terms are those at or below median df.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi0474.phi036.perseus-lat1:1.19

## S2-068 epideictic_by_parts

Rhetoric 3.16, fetched from Perseus, says that in epideictic the narrative is not continuous but by parts, and that the speech has an atechnon part, the facts, which the speaker did not cause, and an entechnon part. A continuous summary that returns one turn has collapsed the parts. On a non-WH query, if the best session has three or more turns with an entity, admit one turn per entity that is unique to that turn, capped at 6. Slot 1 is the turn whose unique entity has the lowest df. Turns that share all their entities with another turn are the entechnon restatement and are not given a quota seat. The search returned rhetorical readings of political speeches, not this quota.

Math: unique_ent(t)= entities of t that no other turn in the session has. Admit turns with a nonempty unique_ent, cap 6. Slot 1 = min df of those unique entities.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:greekLit:tlg0086.tlg038.perseus-grc1:3.16

## S2-069 half_fee_digression

De Oratore 2.352, fetched from Perseus, has Scopas refuse half of Simonides' fee because the song had wandered off into praise of Castor and Pollux, and tell him to collect that half from them. A turn that shares its entities with the head but adds content ids the head lacks is the digression. If the head's numbers include a value and the challenger has no number, the challenger is the unpaid half's subject and takes slot 1 when the query has no digit. A query with a digit keeps the head, because the fee is the asked amount. The search for Simonides' seat is a different rule. This one never admits a new turn.

Math: WH or not: if query has no digit, and exactly one top5 turn has entity overlap with the head, no number, and a content id the head lacks, seat it. If the query has a digit, do nothing.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi0474.phi037.perseus-lat1:2.352

## S2-070 loci_as_wax_not_a_walk

De Oratore 2.354, fetched from Perseus, says to take places and to set images of the things in them, so that the order of the places keeps the order of the things, and the images mark the things, places for wax and images for letters. The walk itself is turn1's method of loci and is not reused. What is reused is the index shape: a place is a session, and an image is an entity. At ingest, the pair (session id, entity id) is a key in a DetHashMap to the turn index. A query that contains the entity looks up that key in the session of the rarest non-entity term, and admits that turn if it is not already seated. There is no ordinal walk and no imagined room. Quintilian 11.2 quotes the same wax sentence and rejects Metrodorus's 360 places. The cap here is one image, not 360. The search 'images as letters places as wax index retrieval' returned imaging papers, not this map.

Math: index[(session, entity)] = earliest turn in that session containing the entity. Lookup uses the session of the rare-term anchor and the query's entity, if it has exactly one. Cap 1.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi0474.phi037.perseus-lat1:2.354

## S2-071 metrodorus_cap

In the same fetched chapter Quintilian is surprised that Metrodorus found 360 places in the twelve signs and calls it vanity. Any summary quota in this shard, including S2-065 and S2-068, is capped at 8 seats, and slot 1 is chosen before the cap discards the rest. This rule is the cap itself, applied at seat time: if a non-WH seat list has been extended past 8 by a quota, keep slot 1 and the 7 next-highest quota keys and drop the rest from the seated list back to the pool. They are not deleted from the index. Eight is the integer Quintilian's scorn implies as a ceiling far below 360, and it is a count of seats, not a tuned weight.

Math: If a quota appended more than 8 turns, keep list[0] and the 7 appended turns with the largest quota keys. Stable on ties by lower turn index.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi1002.phi00111.perseus-lat1:2

## S2-072 thesaurus_four_stores

The opening of the fetched Institutio 11.2 calls memory the treasury that keeps exempla, laws, responsa, sayings, and deeds ready. Four stores, distinguished by shape, not by those Latin words: a turn with an entity and a digit (deed), a turn with no entity and no digit (a rule), a short turn under 20 tokens answering a question-shaped previous turn (a responsum), and a turn with an entity and no digit (an exemplum). On a non-WH query, admit one turn of each store from the best session, cap 4. Slot 1 is the deed if the query has no digit, and the rule if the query's terms are all above median df. A store that is absent is skipped, not filled with another store's turn.

Math: Four disjoint shapes. Quota = number of shapes present, at most 4. Slot 1 = deed if any query term has df<=median, else the rule-shaped turn.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi1002.phi00111.perseus-lat1:2

## S2-073 chreophylakeion_file

Pleiades 45230554, fetched in the Dura-Europos search RSS, is the chreophylakeion, the archive office in the agora of Dura. An archive files documents under the office, and the office's name is on every tablet. A high-df entity that occurs in at least half the turns of its session is the office. It does not, by itself, admit a turn. Admission requires another query term. The place JSON for Oxyrhynchus, Pleiades 736983, is the same practice at the find-spot level: the city is the file, the person is the document. No gazetteer is loaded. The office is detected by its df inside the session. The search 'place name high document frequency person name low retrieval' returned gazetteer and RAG papers, including 'Identifying Origins of Place Names via Retrieval Augmented Generation'. Those load a gazetteer or a generator. This test is a count of turns inside one session. An office id is removed from the session admission bag as well as the turn bag. A session whose only overlap with the query is the office does not take session slot 1. Hook retrieval.rs 5480-5600.

Math: office(e, session)=1 iff e is an entity and the fraction of session turns containing e is >= 1/2, integer comparison 2*count >= n_turns. Office ids are removed from the admission bag. They stay in content for a query that actually contains them.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; LoCoMo Session@1. Record: https://pleiades.stoa.org/places/45230554

## S2-074 vindolanda_place_key

Pleiades 89313, fetched as JSON, is Vindolanda. The strength report and the invitation are both from that fort. A summary that matches only the fort's name would return every tablet. When the query's only entity is an office id under S2-073, the summary quota of S2-065 is computed on the session and the office id contributes 0 to every turn's overlap. The tablets then separate by their other tokens. This rule is the zeroing of that one id inside the quota key, not a new place list.

Math: quota_key(t) = |content(t) ∩ query| minus 1 if the intersection contains an office id. Non-WH only.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: RECALL/INDEX. Slice: BEAM summarization; LME Turn@1. Record: https://pleiades.stoa.org/places/89313/json

## S2-075 military_record_room

Pleiades 977420437, fetched in the Dura RSS, is Room W13, requisitioned for military record-keeping inside the temple of Artemis Azzanathkona. The Dura rosters are the contents of such a room. Turns that share an office id under S2-073 are the room's rolls. A non-WH summary admits one turn from each session that carries that office id, the turn with the most numbers, capped at 8. Slot 1 is the turn with the largest first number. This is how a record room is read: one sheet from each cohort, not every line of the first sheet. Sessions that do not carry the office id are not in the room and are not admitted by this rule. The session of each picked roll is inserted into final_session_indices when it is absent, capped at 8, at retrieval.rs 5480-5600. BEAM Session@10 is 275 and the bar is 320, so the insertions are aimed at the 45 sessions outside the top 10. The same picks are distinct turns, so All@5 can move. A slot-1 reorder leaves All@5 at 118.

Math: sessions = those where some entity is an office. From each, pick the turn with the most numbers. Cap 8. Slot 1 = max numbers[0], tie broken by lower session index.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1; LoCoMo Session@1. Record: https://pleiades.stoa.org/places/977420437

## S2-076 tebtynis_grapheion_day

Pleiades 737072, fetched as JSON, is Tebtynis. The family archive and the grapheion abstracts are filed under that village. A day line, a turn with a digit and fewer than eight content tokens, loses slot 1 to a longer turn in the same session that contains the same digit. The day line is the docket. The longer turn is the act. The search 'daybook entry digit equals following total retrieval' returned preservation surveys, not this preference. The village id is not loaded from Pleiades. Sameness of digit and session is the whole test.

Math: If the head has a digit, content.len()<8, and exactly one longer same-session top5 turn contains that digit, seat the longer turn.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://pleiades.stoa.org/places/737072/json

## S2-077 elephantine_temple_account

Pleiades 786021, fetched as JSON, is Elephantine, where P.Eleph. 10 orders the accounts kept by month and by year. When a year-total turn and a monthly turn from S2-046 are both in the top 5 and the query has no digit, the monthly turn takes slot 1. The island's name is not consulted. The two turns are told apart by the year-total mark. This is the ordering companion of S2-046's admission rule.

Math: Non-WH, no query digit. If the head is a year-total and exactly one top5 turn is not, and that turn shares the session, seat it.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://pleiades.stoa.org/places/786021/json

## S2-078 kellis_prefect_after_date

Pleiades 776189, fetched as JSON, is Kellis. P.Kellis I 1 names the prefect Rupilius Felix after the consular date. The prefect is the first entity that follows the consular number from S2-038. Store it as issuer_id. A zero-entity query that contains the rarest non-entity term admits the turn and reports issuer_id as the answer entity, not the emperors inside the titulature. The oasis id is not loaded. The position after the consular number is the whole test.

Math: issuer = first entity with seq index greater than the consular number's index. Answer entity for a zero-entity query is issuer, if the turn contains rarest(query).

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://pleiades.stoa.org/places/776189/json

## S2-079 mercury_fragment_yields

EDH HD000077, fetched live, is a fragment of a dedication to Mercury, three broken lines. A turn whose content tokens are fewer than four and which contains no complete entity of length greater than four characters, measured at ingest, yields slot 1 to a top-5 turn that contains a query term and has at least twice the content. If the fragment is the only turn containing the query's rarest term, it stays. A broken dedication is not promoted over a complete one, and it is not discarded when it is the only witness.

Math: broken(t)=content.len()<4 and max entity character length <=4. Yield only if another top5 turn contains rarest(query) and has content.len() >= 2*head.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD000077

## S2-080 memory_formula_fragment

EDH HD011111, fetched live, restores a Dis Manibus et memoriae aeternae formula and a span of 26 years. The fetched text is mostly brackets. A number of two digits that is at least 10 and at most 99, standing after a run of at least three length-1 tokens, is a span of years and not an age in S2-051's three-field sense. It is the visible digit for a query that contains no other digit. The restored words in brackets are not read, because the ranker does not have them as tokens unless ingest saw them. Only the number and the short-token run are used.

Math: span(t)= the unique number in 10..=99 that follows three length-1 tokens, if no age-triple exists. A single query digit matches span.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://edh.ub.uni-heidelberg.de/edh/inschrift/HD011111

## S2-081 wax_seal_not_a_match

In the fetched chapter Quintilian reports the theory that memory is a wax tablet holding the impressions of signet rings, and he does not adopt it. A near match of entity strings is not a match. Entity match stays exact, by term id. This rule forbids a seat that would treat two entity ids as equal because their raw forms share a prefix. If two top-5 turns are candidates only because their entity ids differ and one id's surface form starts with the other, neither is promoted. The practical effect is a guard: S2-061 and S2-070 may not add a second entity id on prefix grounds. Exact id equality is the only equality. No new near-match is introduced.

Math: entity equality is term-id equality. A rule in this shard returns no seat when its only distinction between two turns is a shared surface prefix of two different entity ids.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi1002.phi00111.perseus-lat1:2

## S2-082 deposit_while_speaking

The fetched chapter says that while one thing is being said, the next thing to say is already in view, and whatever is found is deposited with memory, which passes it from invention to elocution. The deposited point is not the sentence currently being spoken. At ingest, for a turn with at least two sentences, the second sentence's ids are deposit_ids. On a non-WH query, deposit_ids are the summary key and the first sentence's ids are not. This is the opposite of an incipit. The search 'memory deposits a found point while speech continues retrieval' returned work on embodied memory, not this split.

Math: summary_key(t)=|second_sentence ∩ query|. First-sentence ids contribute 0 to this key. Non-WH only. Seat the unique max in the top 5 if its key is positive.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi1002.phi00111.perseus-lat1:2

## S2-083 conjectural_two_sides

De Inventione 1.19 says that in a conjectural constitution the question and the judgment are the same: was it done, or not. Both sides stay. If two top-5 turns share their entity set and one has a digit the other lacks, a query with a digit seats the turn that has it, and a query with a closed-class negation seats the turn that lacks it. The negation test uses the existing closed-class set, not a new list. If the query has neither a digit nor a negation, there is no swap, because the issue is not yet typed. Both turns remain in the top 5 either way.

Math: Pair with equal entity sets in the top 5, exactly one of them form_digit. Digit query seats the digit turn. Negation query seats the other. Otherwise no swap.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi0474.phi036.perseus-lat1:1.19

## S2-084 multiple_issues_multiple_seats

The same fetched section says that as many constitutions as there are in the cause, so many questions, reasons, and judgments must be found. A session whose turns fall into more than one S2-072 store is a cause with several issues. On a non-WH query, the number of seats taken from that session equals the number of stores present, not one. Cap 4, which S2-071 then caps again at 8 if other quotas have appended. Slot 1 remains the store S2-072 would have chosen. The extra seats exist so All@5 can move. They are not extra copies of the same turn.

Math: seats = distinct store ids present in the best session, at most 4. One turn per store, the turn with the most content ids in that store.

Gap coverage: admits more than one turn, so All@5 and All@10 can move. A slot-1 reorder cannot. BEAM All@5 is 118 and All@10 is 138, both need 320. LME Turn All@5 is 314 and needs 423, with the mass in multi-session (53 of 121, need 109) and temporal-reasoning (74 of 127, need 115). The wide-pool ceiling of those All@ cells is not computable from the attachment, because the bits do not say whether every gold turn is inside the pool or whether a question has more than five golds. The same admission can raise LoCoMo Turn@1 toward the wide pool of 1492 and pool@50 of 1415, both above 1383, and BEAM Turn@1 toward the wide pool of 329, which is above 320. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; BEAM Turn@1; BEAM All@5; BEAM All@10; LME Turn All@5; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi0474.phi036.perseus-lat1:1.19

## S2-085 short_speech_drops_epilogue

Rhetoric 3.13 says the epilogue is dropped when the speech is small or the matter is easy to remember, because it only shortens what is already short. If the best session has fewer than four turns, S2-062's reminder does not take slot 1 even on a non-WH query. The fact turn, the one with a digit if there is one, keeps slot 1. The reminder remains in the top 5. Short is an integer count of turns, not a tuned length threshold.

Math: If session turn count < 4 and the head is a reminder and a non-reminder exists in the top 5 of that session, seat the non-reminder with a digit if unique, else the unique non-reminder.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: BEAM summarization; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:greekLit:tlg0086.tlg038.perseus-grc1:3.13

## S2-086 atechnon_facts_first

Rhetoric 3.16 separates the atechnon facts, which the speaker did not cause, from the entechnon showing. A turn with a digit is the fact. A turn without a digit, sharing the fact's session and one entity, is the showing. On a WH query, if the head has no digit and exactly one top-5 turn in its session has a digit and shares an entity with the head, seat the digit turn. The showing stays seated behind it.

Math: WH query. Head has no digit. Unique same-session top5 turn with a digit and entity intersection nonempty. Seat it.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:greekLit:tlg0086.tlg038.perseus-grc1:3.16

## S2-087 oxyrhynchus_file_not_answer

Pleiades 736983, fetched as JSON, is Oxyrhynchus/Pemje. P.Oxy. XIV 1631 names the city and then the parties. The city token, once it is an office id under S2-073, must not be the answer entity of a zero-entity query. The answer entity is the min-df entity that is not the office. This rule stores that choice on the turn at ingest. It does not load the Pleiades name. The JSON is the historical warrant that a city is a file heading. The search that returned gazetteer RAG is cited under S2-073. This rule is the answer-entity choice, which S2-073 does not make. The answer session is the session of the lowest-df entity that is not an office, and it takes session slot 1 when that session is already in the top 5. Hook retrieval.rs 5480-5600. This orders sessions the list already holds. It does not admit a session from outside that list.

Math: answer_ent(t)= argmin df of entities that are not office ids, if any. Zero-entity queries report answer_ent, not the office.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; LoCoMo Session@1. Record: https://pleiades.stoa.org/places/736983/json

## S2-088 responsa_short_reply_store

Quintilian lists responsa among the things memory keeps ready. A responsum in this index is the existing is_short_reply turn whose answers edge points at a previous turn. The ranker does not add a word list of answer verbs. If a WH query matches the previous turn on at least one rare term and misses the short reply, the short reply is admitted immediately after the previous turn. The reply's own overlap may be empty. The edge is the existing backward answers link. It is not a new forward edge, and it does not touch rsb_seat_behind.

Math: If plan.has_wh and turn p contains a rare query term and p.answers is none and some s has s.answers==Some(p) and s.is_short_reply, admit s at the seat after p if s is not already seated. Cap 1.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: LoCoMo Open-Domain; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi1002.phi00111.perseus-lat1:2

## S2-089 law_store_no_entity

A law in Quintilian's treasury has no party's name. A deed has a name and a number. If every query content term has df above the median and the query has a WH token, the head yields to the unique top-5 turn that has no entity and no digit, provided one exists. That turn is the rule. A query that contains any below-median term does not use this seat, because then a deed may be the answer. P.Hal. 1 was fetched and is only a stub in this snapshot, so the warrant is the Perseus chapter, not an unread code.

Math: If every query content id has df>median and the query has WH, seat the unique top5 turn with empty entities and form_digit false.

Gap coverage: permutes the current top 5, or breaks a tie inside it. LME Turn@1 floor is 353 and Turn@5 is 428, which clears the bench bar of 423. The same ceiling misses LME single-session-preference (Turn@5 19, need 27) and is 4 short on LME temporal-reasoning (111, need 115). It cannot clear LoCoMo Turn@1, whose Turn@5 ceiling is 1127 against 1383, or BEAM Turn@1, whose Turn@5 ceiling is 238 against 320. All@5, All@10, and Session@1 do not move, and the session list is already frozen at retrieval.rs 5480.

Tag: ORDERING. Slice: LoCoMo Open-Domain; LME Turn@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi1002.phi00111.perseus-lat1:2

## S2-090 deed_store_named_number

The companion of S2-089. A deed is a turn with at least one entity and at least one digit. On a non-WH query, if the best session contains a deed, that deed is slot 1 of the summary even when a no-entity rule-turn has a higher raw overlap. The deed is what Quintilian says the orator must have ready. The rule-turn remains a later seat under S2-084 if its store is present. If the session has several deeds, slot 1 is the deed with the most entities, ties broken by the lower turn index. This does not reorder WH queries. Those stay with S2-089 when the query is all high-df.

Math: Non-WH. deeds = turns in the best session with an entity and a digit. Slot 1 = max entity count, then min turn index. If no deed, the rule does not fire.

Gap coverage: can change which turn is in the pool or the seated list. LoCoMo Turn@1 floor is 805 and the bar is 1383. Turn@5 is 1127 and Turn@10 is 1226, so reordering inside official k is short by 157. The wide pool holds 1492 and pool@50 holds 1415, both above 1383. BEAM Turn@10 is 267 against 320, and the wide pool of 329 clears that bar. LME single-session-preference Turn@5 is 19 against a need of 27, and its pool is 30. Open-Domain's own pool is 78 against a bar of 83, so this class still cannot finish that category unless the rule admits one of the 14 turns outside the pool. If the admitted turn's session is absent from final_session_indices, that session is inserted, one extra session, at retrieval.rs 5480-5600. BEAM Session@10 is 275 against 320. LoCoMo Multi-Hop Session@5 is 246 against 254, and Open-Domain Session@5 is 65 against 83. Those category bars need a session the top 5 does not already hold.

Tag: RECALL/INDEX. Slice: BEAM summarization; LoCoMo Turn@1; LoCoMo Turn@5; LoCoMo Turn@10; BEAM Turn@1; BEAM Turn@10; LME preference Turn@1; BEAM Session@1. Record: https://www.perseus.tufts.edu/hopper/CTS?request=GetPassage&urn=urn:cts:latinLit:phi1002.phi00111.perseus-lat1:2

## Top 10 build order

The order is the size of the gap a slot-1 reorder cannot close. The historical mechanism is unchanged. The hook is whatever stage the ceiling says has to move.

1. **S2-061** (LoCoMo Turn@1). Admit one same-session turn within four indexes of the rare-term anchor when the query names nobody and that neighbor carries an entity the query lacks. Turn@10 is 1226 and the bar is 1383. The wide pool is 1492 and pool@50 is 1415.

2. **S2-013** (LoCoMo Open-Domain, and any measurement question outside the pool). Admit the unique turn in the anchor session whose integers contain a two-term sum. Open-Domain's pool is 78 against a bar of 83, so the bench-level pool surplus does not finish this category.

3. **S2-087** (LoCoMo Session@1). The entity in at least half the turns of its session is the file. Session slot 1 is the session of the lowest-df entity that is not that file, when that session is already in the top 5. Session@5 is 1416, which clears 1383. Hook 5480-5600.

4. **S2-075** (BEAM Session@1 and BEAM All@5). One number-rich turn from each session that shares an office id, and that session is inserted if the list does not hold it. Session@10 is 275 against 320. All@5 is 118 and a slot-1 reorder leaves it there.

5. **S2-065** (BEAM All@5 and All@10, LME Turn All@5). From the best session, one turn of each orthographic part, at most four. Summarization All@5 is 0. The bench All@5 cells are 118 and 314.

6. **S2-084** (LME Turn All@5). One seat per store present in the session, at most four. Multi-session All@5 is 53 of 121 (need 109). Temporal-reasoning All@5 is 74 of 127 (need 115).

7. **S2-001** (LME Turn@1). Inside the top 5, seat the unique turn that holds every rare query term and an entity the query does not contain. Turn@5 is 428, which clears 423. Preference stays short at 19 against 27.

8. **S2-028** (BEAM All@5, event_ordering). Split on date-header lines and admit the lowest-df name of each block, capped at 8. event_ordering All@5 and All@10 are both 0.

9. **S2-014** (LoCoMo Session@1). When two places are in the top 5 sessions, seat the session of the rarer one. This is ordering, so Multi-Hop (246 against 254) and Open-Domain (65 against 83) stay short.

10. **S2-012** (LoCoMo Open-Domain no_pool). A closed-class cardinal that equals the count of entity ids locks those entities as one act, and a query that names any of them admits the turn. This is an admission from outside the pool, which Open-Domain needs and the other LoCoMo categories do not.


Risks, stated without shrinking the target. S2-061 will also admit a neighbor that is not the gold. The cap of one and the four-turn window are what keep it from flooding the pool, and it still has to be scored under every turn that already covers two rare terms. S2-065, S2-028, S2-075, and S2-084 can fill All@5 with the wrong section's representative. The representative is the lowest-df entity, the digit-bearing turn, or the number-richest turn. That choice is falsifiable on the replay dump. The wide-pool ceiling of All@5 is not in the attachment, so these four rules are the way the cell moves and the dump cannot say in advance whether 320 and 423 are inside the pool. S2-075 fires only when an office id, a majority entity inside the session, is shared. Questions with no such entity need a different admission rule from this set, and the single-turn admissions carry the session insert for that case. None of the ten retunes a constant that already exists.
