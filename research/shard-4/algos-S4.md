# Shard 4 algorithms

Ninety rules from South Asian grammar, recitation, commentary, logic, and manuscript practice.
The floor is 90% on every metric of LME, LoCoMo, and BEAM, every category. Base SHA `e2ecd1da037974c1835f563ad8b046f4a4ca3629`.
Combined mechanisms: 50 of 90. ORDERING 41, RECALL/INDEX 49.

A top-5 permute clears LME Turn@1 (428 ≥ 423) and does not clear LoCoMo Turn@1 (1127 < 1383), BEAM Turn@1 (238 < 320), BEAM All@5 (118, and All@10 is 138 < 320), or BEAM Session@1 (Session@10 is 275 < 320). The build order starts with admission and the once-only quota.

## S4-001 Adhikāra frame stack

Pāṇini marks a heading with a svarita and lets it govern until the next heading of equal rank. The fetched sūtra svaritena adhikāraḥ (1.3.11) and the Mahābhāṣya's gloss that atha is used adhikārārthaḥ are the witnesses. Ingest stores a heading frame and posts its content ids onto later turns inside the frame, so a multi-hop query that names the heading still retrieves the turns that sit under it. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `frame(t) = max {h < t : rank(h) > 0 and no h' in (h, t] has rank(h') >= rank(h)}; inherit ids(frame(t)) at weight 1, length 0`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-002 Vipratiṣedha rightmost group

When two top-5 turns cover the same number of query groups, Pāṇini 1.4.2 gives the seat to the turn that matches the later group in the query. The fetched line is vipratiṣedhe paraṃ kāryam. Coverage ties no longer fall through to scene score. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if |cover(a)| = |cover(b)| then winner = argmax_t max {g : group g hits t}; else no swap`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-003 Pūrvatrāsiddha snapshot

The tripādī rule pūrvatra asiddham, fetched as ps_8,2.1, says a later operation is invisible to an earlier one. Shard seats therefore score the original top 5, and a disagreement throws away the earlier swap rather than composing the two mutations. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `challenger = f(list_before_s4); if earlier_swap != challenger then list = apply(list_before_s4, challenger)`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-004 Ekaśeṣa counted residue

Pāṇini's ekaśeṣa keeps one of several identical coordinated words and lets the grammatical number carry the count. The fetched sūtra is sasūpāṇām ekaśeṣa eka-vibhaktau. A summary turn that lists the same anchor many times is stored once, with the multiplicity as an integer, so repeated mentions stop eating BM25 length. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `post(id) once; count(t, id) = occurrences; score += count, length += 1`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-005 It-saṃjñā marker class

Pāṇini's it-markers classify an operation and are then deleted, which the fetched sūtras upadeśe 'j-anunāsika it and ādir ñi-ṭu-ḍavaḥ state. Markdown marks become an integer class on the next content id and drop out of the vocabulary, so a query cannot match the mark itself. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `class(mark) in {1, 2, 3}; post(next_id, class); do not post(mark)`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-006 Lopa place-gap

tasya lopaḥ deletes the marker and leaves its place, so two content words separated by one function word are still a pair at distance 2. The fetched sūtra is ps_1,3.9. Multi-hop cues that the dialogue splits with a determiner or auxiliary still join. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `dist(a, b) = 1 + closed_class_between; pair weight = 3 - dist for dist in {1, 2}, else 0`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-007 Bahuvrīhi external referent

A bahuvrīhi denotes something outside its own words. The fetched pair śeṣo bahuvrīhiḥ / anekam anyapadārthe is the rule. If a turn only repeats the query and the next turn adds an id the query lacks, that next turn enters the pool. This is the multi-hop case where the answer wording is not in the cue turn. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `if content(t) subset content(q) and content(t+1) \ content(q) nonempty then admit t+1`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-008 Dvandva both-or-zero

A dvandva treats both members as heads, which ca-arthe dvandvaḥ states in the fetched Aṣṭādhyāyī. A query joined by and therefore gives no credit to a turn that carries only one of the two groups, and slot 1 moves to a top-5 turn that carries both. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if query has and between g1, g2: credit(t, g1) = credit(t, g2) = 1 if both hit else 0`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-009 Parāṅgavat head gate

Jaimini's pradhāna/guṇa split, fetched at jaim_6,3.2, makes subsidiary credit conditional on the principal. The principal is the query's last content group. A turn whose modifiers sit in a different clause from that head loses them, and a top-5 turn that keeps them together takes slot 1. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `credit(modifier) = 1 iff same clause contains head_group, else 0; head_group = last content group`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_jaimini-mImAMsAsUtra.txt

## S4-010 Avyayībhāva left-head order

An avyayībhāva is head-initial, unlike a tatpuruṣa. The fetched sūtra avyayībhavaḥ is the witness. A preposition-noun query matches a turn only when the preposition precedes the noun in the turn's seq, and the reverse order cannot hold slot 1. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `credit(p, n) = 1 iff index(p) < index(n) in seq, else 0`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-011 Trairūpya three-count gate

Dharmakīrti's three forms of a reason are in the fetched Nyāyabindu line trairūpyaṃ punar liṅgasya. A bridge token keeps slot 1 only when it sits in the subject turn, appears with the property somewhere else, and appears nowhere with the negated property. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move. Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `accept(R) iff paksa(R)=1 and sapaksa(R)>=1 and vipaksa(R)=0`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-012 Anupalabdhi under conditions

Dharmakīrti allows absence as a reason only when the conditions of perception are met. The fetched Nyāyabindu sentence names upalabdhi-lakṣaṇa-prāpta. A session that already contains the other query ids and lacks the negated id is admitted; a session that never mentions the topic is not an absence witness. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83). Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `absent(session, T) = 1 iff hits(session, other_ids) = |other_ids| and hits(session, T) = 0`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-013 Kārya-hetu empty successor

From an effect one infers the cause, and the effect is not a paraphrase. The fetched example is vahnir atra dhūmād. The next turn in the session enters the pool when it shares no content id with the cue turn, which is how a multi-hop answer worded differently than the cue becomes reachable. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `admit(t+1) iff t in pool and content(t) ∩ content(t+1) = ∅ and content(t+1) ≠ ∅`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-014 Svabhāva subset gate

The śiṃśapā is a tree because the specific property is already the general one. The fetched line is vṛkṣo ayaṃ śiṃśapātvād. Slot 1 moves to the top-5 turn that contains both the rarer id and the more frequent id, and a turn that has only the rarer id yields. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if df(a) < df(b) and turn has a and not b, yield to a top-5 turn that has a and b`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-015 Asiddha-hetu rejection

A reason not present in the subject is asiddha. The fetched Nyāyabindu calls cākṣuṣatva unestablished for both parties when sound's impermanence is the thesis. Slot 1 loses when its bridge id is missing from its own content set and another top-5 turn has the bridge and the rarest query id. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if bridge not in content(head) and some top-5 t has bridge and rarest then seat t`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-016 Viruddha-hetu majority

A contradictory reason proves the opposite. The fetched line is dvayo rūpayor viparyaya-siddhau viruddhaḥ. When a bridge id is more common in the dissimilar pool than in the similar pool, the head that uses it leaves slot 1. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `viruddha(R) iff vipaksa(R) > sapaksa(R); then head yields`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-017 Anaikāntika width

An inconclusive reason is present on both sides. Nyāyabindu 03066 calls that anaikāntika, and the fetched daśaka names savyabhicāra. A bridge more frequent than the rarest query id cannot be the only reason slot 1 stands on. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `wide(R) iff df(R) > df(rarest); wide bridge cannot be the only extra id of the head`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-018 Pakṣābhāsa self-negation

Nyāyabindu's fourth false thesis is the one refuted by the speaker's own words, fetched as na anumānaṃ pramāṇam. A summary slot 1 that contains an id both plain and under a negator yields to a top-5 turn that states the id once, positively, and the negated witness stays available in the five. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `self_neg(t, id) iff id in content(t) and id in neg_scope(t); such a t yields slot 1`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-019 Avacchedaka clause lock

Navya-Nyāya calls the limitor the avacchedaka, and the fetched Gaṅgeśa passage requires the limitor to be immediate. Two cues support slot 1 only when they share a clause index. A head that merely has them in different clauses yields. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `limited(a, b) iff clause(a) = clause(b); else the pair scores 0`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gAGgeza-tattvacintAmaNizabdakhANDa.txt

## S4-020 Abhāva triple posting

Tarkasaṃgraha counts absence as the seventh category, in the fetched line that ends sapta padārthāḥ. A negation is stored as the triple (locus, absentee, limitor) and a negated multi-hop query hits that triple rather than every turn that happens to contain the absentee word. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `match iff locus=q_locus and absentee=q_absentee and limitor=absentee; else the triple contributes 0`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_annaMbhatta-tarkasaMgraha.txt

## S4-021 Prāgabhāva interval open

Prior absence is the state before the thing exists. Using Tarkasaṃgraha's abhāva as the category, ingest marks every earlier turn in the session with the id that has not yet appeared. A negated query can retrieve that earlier turn, which a positive posting of the later mention does not do. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `prag(t, id) = 1 iff pos(t) < min_pos(id) in the session; else 0`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_annaMbhatta-tarkasaMgraha.txt

## S4-022 Pradhvaṃsa destruction point

Destruction-absence starts at the first denial after an affirmation. Nyāyasūtra 2.2.12, fetched, already separates prior absence from other absence, and Tarkasaṃgraha lists abhāva among the categories. That destruction turn is pulled from the pool into the official tail so a multi-hop question about the change can see it inside k. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `destroy_pos(id) = min {pos : neg_scope has id and some earlier pos' has id plain}; admit that turn into the tail`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_annaMbhatta-tarkasaMgraha.txt

## S4-023 Anyonyābhāva exactly-one

Mutual absence says each thing is not the other. With Tarkasaṃgraha's abhāva as the category, a negated two-id query gives slot 1 to the top-5 turn that has exactly one of the ids, and an unnegated query keeps the turn that has both. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if query_neg and |ids|=2: prefer |content ∩ ids| = 1 over |content ∩ ids| = 2`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_annaMbhatta-tarkasaMgraha.txt

## S4-024 Saptabhaṅgī class quota

Āptamīmāṃsā, fetched, treats conflicting claims as unable to all be absolute. A summary top 5 is filled in the seven-predicate order: plain, negated, both, and the question-marked combinations. A second turn of a class already seated waits. That is what all@5 is missing: the other golds are other predicates. Ceiling class K_PULL: inserts an uncovered pool turn into the five. BEAM Turn All@10 is 138 against 320 and LME has no turn All@10 bit, so the quota finishes All@5 only after admission has put the other golds in the pool.

Math: `seat[i] = first pool turn of class i not yet seated, for i in 0..7 while seats < 5`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_samantabhadra-AptamImAMsA.txt

## S4-025 Nikṣepa mention zero

Nikṣepa's name-installing is mention, not use. A head whose only hits on the rarest id sit inside quotes yields to a top-5 turn where the same id is outside quotes. Āptamīmāṃsā sam_1, fetched, is the witness that a displayed attribute is not the thing itself. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if query has no quote mark and every hit of rarest on head is quote_bit then yield`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_samantabhadra-AptamImAMsA.txt

## S4-026 Yogavibhāga clause length

Splitting one rule so each half has its own domain is yogavibhāga. The fetched Kaumudī already segments the Aṣṭādhyāyī into prakaraṇas. A long summary turn is scored by its best clause's length, so the clauses that do not answer stop diluting it out of the pool. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `bm25(t) = max over clauses c in t of bm25(c) with length(c) = |content(c)|`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_varadarAja-laghusiddhAntakaumudI.txt

## S4-027 Gaṇapāṭha list object

Pāṇini's gaṇas are lists cited as wholes. The fetched sūtra bahu-gaṇa-vatu-ḍati saṅkhyā treats the gaṇa as a number-forming list. A bullet or comma list becomes one posting keyed by its head, and a summary query that hits any member retrieves that whole turn rather than a stray single item. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `gana(head) = members; hit any member => retrieve the one turn; prefer it to a singleton member turn`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-028 Pratyāhāra end-pair

Pāṇini's opening pratyāhāra aic in the fetched vṛddhir ād-aic names a span by its endpoints. A turn that holds both the session's first content id and its last is the abbreviation of that session, and a summary query on either end pulls the turns between them into the pool. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `if turn has id_at_first_pos and id_at_last_pos then admit every turn with pos in (first, last)`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-029 Anukramaṇī five-count card

Vedic anukramaṇīs index a hymn by a fixed card of counts. The fetched padapāṭha is the word list those cards govern. Each session gets five integers, including token_count/32, and summary pool admission prefers the session whose card is nearest the query card in L1. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36). Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `L1(s, q) = sum_i |card(s)_i - card(q)_i|; admit in increasing L1`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_RgvedasaMhitApadapATha.txt

## S4-030 Grantha extent checksum

Manuscript colophons state extent in granthas, and the fetched Derge Tanjur record states 213 volumes. A summary turn longer than one grantha per seated session is too long for the evidence it can actually cover, and it yields slot 1 to a shorter top-5 turn with the same groups. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move. Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `legal(t) iff content_len(t)/32 <= |{session of top-5 turns}|; else yield`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://ldspdi.bdrc.io/resource/MW23703.json

## S4-031 Pecha consecutive leaf

The fetched Derge instance IE23703 is a stack of volumes, each volume a stack of leaves. Two consecutive turns are admitted as one leaf, and the side with more query groups is the representative. The other side remains eligible for a later summary seat. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `leaf = (t, t+1) consecutive; admit both; seat argmax cover`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://ldspdi.bdrc.io/resource/IE23703.json

## S4-032 Explicit scope close

Pāṇini closes a naming section at a stated boundary, fetched as ā kaḍārādekā sañjñā. A turn that is only a number or a function word pops the heading frame, so the next topic in the session is not retrieved under the previous heading. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `if content(t) empty and tokens are digits or closed-class then pop frame; inherit stops`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-033 Mūla containment seat

The fetched Mahābhāṣya treats the short śāstra as the text under discussion. If a later turn contains every content id of an earlier short turn, a factoid seats the short turn and a summary seats the long one, provided both are already in the top 5. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if content(early) ⊆ content(late) and |content(early)| <= 8: factoid seats early, summary seats late`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pataJjali-vyAkaraNamahAbhASya.txt

## S4-034 Sūtra/bhāṣya length alternation

A bhāṣya sits next to a short sūtra, which is the shape of the fetched Mahābhāṣya. Sessions whose lengths alternate around their own median give the summary seat to a long turn and the factoid seat to a short turn that actually holds the rare id. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `short = len <= median(session); summary slot 1 must be long; factoid slot 1 must be short and hold rarest`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pataJjali-vyAkaraNamahAbhASya.txt

## S4-035 Vārttika anukta field

A vārttika supplies what the sūtra omitted. Against the fetched Mahābhāṣya, ingest posts the ids a turn adds over the previous turn as their own field, so a multi-hop query for the new fact does not drown in the ids the turn only repeats. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `anukta(t) = content(t) \ content(t-1); post that set as its own field`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pataJjali-vyAkaraNamahAbhASya.txt

## S4-036 Ṭīkā depth quota

Commentary stacks sūtra, bhāṣya, ṭīkā. The fetched Mahābhāṣya is the middle of that stack. Summary top-5 seats reserve one place each for answers-chain depth 0, 1, and 2, so a gold that is the comment on the comment is not pushed out by three turns at depth 0. Ceiling class K_PULL: inserts an uncovered pool turn into the five. BEAM Turn All@10 is 138 against 320 and LME has no turn All@10 bit, so the quota finishes All@5 only after admission has put the other golds in the pool.

Math: `reserve one seat each for depth 0, 1, 2 when the pool has that depth`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pataJjali-vyAkaraNamahAbhASya.txt

## S4-037 Adhikaraṇa limb quota

Nyāya's fetched book-2 labels, pūrvapakṣa and siddhānta, plus saṃśaya at 2.1.1, are the debate limbs. A summary top 5 takes one topic, one doubt, one prima facie, one reply, and one decision from the pool before repeating a limb. All@5 stays at 0 when every seat is the same limb. Ceiling class K_PULL: inserts an uncovered pool turn into the five. BEAM Turn All@10 is 138 against 320 and LME has no turn All@10 bit, so the quota finishes All@5 only after admission has put the other golds in the pool.

Math: `for class in [topic, doubt, prima_facie, reply, decision]: seat first pool turn of that class`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gautama-nyAyasUtra.txt

## S4-038 Pañcāvayava example link

The fetched Nyāyasūtra names the example as one of the five members and defines it as the case that has the property. A pool turn that shares a mid-frequency id with the head and lacks the rarest query id is pulled into the official tail, which is the multi-hop illustration the top 10 currently drops. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `example(u, head) iff |content(u) ∩ content(head)| >= 1 and rarest not in content(u) and some shared id has df >= 2`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gautama-nyAyasUtra.txt

## S4-039 Dūṣaṇa missing-rūpa

A refutation states what is missing. The fetched line is dūṣaṇā nyūnatā-ādy-uktiḥ. On a negated query, slot 1 moves to the top-5 turn that shares an id with the head, carries a negator, and adds an id the head does not have. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `dusana(t) iff query_neg and t has negator and |content(t) ∩ content(head)| >= 1 and |content(t) \ content(head)| >= 1`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-040 Natthi change-point

Absence-as-condition is the change at a boundary. With no Pali witness in hand, the fetched Nyāyabindu list of eleven anupalabdhi kinds is the record, and the rule admits a turn that shares no content id with the previous turn when the query's two rare ids never sit together. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `changepoint(t) iff content(t) ∩ content(t-1) = ∅; admit if the two rare ids have no joint turn`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-041 Adhikāra plus adhyāhāra

An open heading plus an elliptical clause is a different object from anuvṛtti. The fetched adhikāra sūtra supplies the frame, and the elliptical turn inherits the frame's rarest id rather than the previous turn's. A short answer under a heading becomes findable by the heading's cue. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `if |content(t)| <= 2 and no finite-shaped token: post rarest(frame(t)) onto t`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-042 Bahuvrīhi plus kārya-hetu

The thing a bahuvrīhi names is not in the compound, and the effect is not a paraphrase of the reason. Both fetched lines are required: the successor of a query-subset turn enters the pool only when it shares no content id. That is the multi-hop answer stated in different words. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `admit(t+1) iff content(t) ⊆ content(q) and content(t) ∩ content(t+1) = ∅ and content(t+1) ≠ ∅`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-043 Trairūpya plus anupalabdhi

Dharmakīrti's fetched list puts anupalabdhi among the reasons that must show three forms. A bridge is admitted only when it is in the subject, present in a similar case, and absent from dissimilar cases that actually contain the other cues. Empty sessions do not count as absence. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `accept iff paksa=1 and sapaksa>=1 and vipaksa_conditioned=0; admit the turn that holds the bridge`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-044 Avacchedaka plus dvandva

A copulative compound whose members are not under one limitor is not the compound. Slot 1 on an and-query requires both ids in the same clause, using the fetched dvandva sūtra and the fetched avacchedaka passage together. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `credit = 1 iff both members hit and clause(left) = clause(right), else 0`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gAGgeza-tattvacintAmaNizabdakhANDa.txt

## S4-045 Ekaśeṣa plus gaṇapāṭha

One residue stands for the list, and the number is the count. A summary that asks a digit matches the list object only when the stored count equals that digit, which is the fetched ekaśeṣa rule and the fetched number-gaṇa rule used together. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `match(gana, digit) iff digit_value = |members| + 1`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-046 Asiddha plus vipratiṣedha

The tripādī is invisible to earlier rules, and of two equal rules the later one wins. Both fetched sūtras apply: the equal-coverage comparison runs on the pre-swap top 5, and the turn matching the later query group is the only swap that survives. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `winner = argmax later_group on the snapshot, among equal-coverage turns; apply winner to the snapshot`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-047 It-marker plus lopa gap

An it is classified and then deleted, and lopa leaves the place. The two fetched sūtras together mean a markdown mark between two content words is gone from the vocabulary and still separates them by one place, so the pair matches at the distance-2 weight. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `delete(marker); dist(left, right) = 2; pair weight = 1`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-048 Yogavibhāga plus aṅga

Each clause is its own rule, and a subsidiary counts only beside the principal. The fetched Jaimini line and the Kaumudī's sectioning combine: a summary turn scores as its best clause, and that clause must contain the query's last content id or it scores 0. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `score(t) = max over c of (bm25(c) if c contains head else 0), length(c) = |c|`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_jaimini-mImAMsAsUtra.txt

## S4-049 Pratyāhāra plus nigamana

A conclusion repeats the thesis, and a pratyāhāra names both ends. The fetched nigamana sūtra and the fetched aic abbreviation together pick, as the summary slot, the top-5 turn that contains both endpoint ids of what the other seats cover. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `summary slot = top-5 turn that contains first(union) and last(union); else unchanged`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gautama-nyAyasUtra.txt

## S4-050 Anukramaṇī plus grantha

A catalog card that includes the extent is the manuscript's own index. The fetched Derge record states 213 volumes, and the padapāṭha is the word index. Summary admission requires the five-count nearest session and a grantha that matches that session's token_count/32 within the number of sessions the turn spans. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36). Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `admit iff |grantha(t) - grantha(session)| <= n_sessions(t) and L1 is the order key`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://ldspdi.bdrc.io/resource/MW23703.json

## S4-051 Alternation plus mūla containment

The fetched Mahābhāṣya is both longer than the sūtra and contained of it. A summary slot moves to the long turn only when the session alternates around its median and the long turn's ids contain an earlier short turn. Either condition by itself leaves slot 1 alone. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `summary seat iff len > median and some earlier short s with content(s) ⊆ content(t)`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pataJjali-vyAkaraNamahAbhASya.txt

## S4-052 Anukta plus ṭīkā depth

What the second commentary adds is the omission the first commentary still lacked. Against the fetched Mahābhāṣya, ids new at answers-chain depth 2 are posted as their own field, and additions at depth 0 or 1 are not. A multi-hop query for the latest correction hits that field. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `post anukta(t) as its own field iff depth(t) = 2`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pataJjali-vyAkaraNamahAbhASya.txt

## S4-053 Limb quota plus saptabhaṅgī

A summary has to cover the debate and the predicate. The fetched Nyāya labels and the fetched Āptamīmāṃsā together fill the five seats with distinct (limb, predicate) pairs, limb order first. That is aimed at all@5 = 0, where one gold is in the top 5 and the other golds are other kinds of turn. Ceiling class K_PULL: inserts an uncovered pool turn into the five. BEAM Turn All@10 is 138 against 320 and LME has no turn All@10 bit, so the quota finishes All@5 only after admission has put the other golds in the pool.

Math: `key = (limb, bhangi); seat distinct keys until 5; limb order first`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gautama-nyAyasUtra.txt

## S4-054 Udāharaṇa plus tātparya

The example shows which sense was meant. Fetched Nyāyasūtra 1.1.36 defines the example, and the fetched Dhvanyāloka puts a further meaning beyond the literal. Of two turns that share an ambiguous id, the one that also intersects the example's ids is admitted. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `admit t where t shares the ambiguous id and content(t) ∩ content(example) ≠ ∅`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_Anandavardhana-dhvanyAloka.txt

## S4-055 Nikṣepa plus it-marker

A name-installing marker is classified and deleted. The fetched it-sūtra and the fetched Āptamīmāṃsā line together put quoted ids in a mention class and drop the quote marks from the vocabulary. An ordinary multi-hop query matches the unquoted posting only. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `class 4 on enclosed ids; marker not posted; match class 4 iff query is quoted`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-056 Ekādeśa plus udātta

One substitute stands for two sounds and keeps the accent, which the fetched ekādeśa sūtra states. The heading token and the next content id in the same clause count as one coverage hit and as two places. Slot 1 prefers that turn over a turn that splits them across clauses. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `fused credit = 1 iff heading and next content id share a clause; places = 2`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-057 Pecha leaf min-cover

A leaf is useful when both sides add something. On the fetched Derge volume graph, a consecutive pair is admitted only when each side covers a query group the other misses, and the pair's rank is the minimum of the two covers. A one-sided pair stays two ordinary turns. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `leaf_cover = min(cover(t), cover(t+1)); admit iff leaf_cover >= 1`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://ldspdi.bdrc.io/resource/IE23703.json

## S4-058 Scope close plus frame pop

A heading governs until the section ends, and the fetched pair of sūtras gives both the heading and the boundary. Inheritance stops at a higher heading, at an empty digit turn, and at 12 turns, so a multi-hop cue from one topic stops pulling turns from the next topic. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `inherit(t) iff frame open and t <= frame_start + 12 and no close and no higher heading in between`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-059 Three hetvābhāsa gates

The fetched Nyāyabindu treats unestablished, contradictory, and inconclusive as separate faults. Slot 1 keeps a bridge only when all three pass: the bridge is in the head, the dissimilar count does not exceed the similar count, and the bridge is no more frequent than the rarest cue. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `keep iff present and vipaksa <= sapaksa and df(bridge) <= df(rarest)`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-060 Pakṣābhāsa plus dūṣaṇa

A thesis the text itself denies is replaced by the turn that says what is missing. Both fetched Nyāyabindu lines are in force: a self-negated summary head yields to the top-5 refutation that adds an id, and only if that refutation is absent does the plain positive turn take the seat. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if self_neg(head): seat dusana if any, else seat first positive-only top-5 turn`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-061 Abhāva triple plus anyonya

Absence of one thing in another is stored as a triple, and mutual absence is the case of exactly one. The fetched AnTs_80 example is a pot that is not a cloth. A negated two-id query hits the posting only when the turn has exactly one of the two ids. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `match iff query_neg and exactly one of {q1, q2} equals absentee and locus is set`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_annaMbhatta-tarkasaMgraha.txt

## S4-062 Prāg plus pradhvaṃsa interval

A thing's life in the session is the interval from its first plain mention to the first later denial. Prior absence and destruction, both grounded in the fetched sūtra and Tarkasaṃgraha, become the two ends. A past-shaped query matches inside the interval, and a negated query matches the destruction turn. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `past-shape matches [first_plain, destroy); negated query matches destroy`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_annaMbhatta-tarkasaMgraha.txt

## S4-063 Natthi at distance one

Absence is a condition of the next moment, not of a later page. With the fetched eleven-kind list as the witness, a change-point is stored only between consecutive turns with an empty content intersection, and only that neighbor is admitted when the two rare ids never co-occur. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `edge(t, t+1) iff intersection empty; no edge at distance > 1`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-064 Kośa bits plus anukramaṇī

Classification before counting is the kośa's opening move, fetched as dharmāṇāṃ pravicayam. A summary query admits a session only when the five orthographic bits match, and the padapāṭha card then orders those sessions by L1. Counts alone no longer pull a session of the wrong shape. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36). Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `admit iff bits(session) = bits(query) and then order by L1(card)`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_vasubandhu-abhidharmakoza.txt

## S4-065 Lakṣaṇā under blocked literal

Indication is allowed when the literal does not deliver the sentence. The fetched Dhvanyāloka places lakṣaṇā next to the literal. A multi-hop query whose content ids never sit together admits one turn that shares a single cue in-clause with another content id, at weight 1. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `if |{t : content(t) ⊇ query}| = 0 then admit a turn that shares one query id in-clause with a co-occurring id`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_Anandavardhana-dhvanyAloka.txt

## S4-066 Sakṛd plus grantha quota

Once an operation has applied, paribh_39 says it stays applied, and a manuscript states its extent. On a summary, a post-seat addition has to cover a query group nobody seated covers yet, and the grantha sum of the five has to stay within the number of sessions. That is aimed at all@5 = 0. Ceiling class K_PULL: inserts an uncovered pool turn into the five. BEAM Turn All@10 is 138 against 320 and LME has no turn All@10 bit, so the quota finishes All@5 only after admission has put the other golds in the pool.

Math: `add t only if cover(t) \ cover(seats) nonempty and (sum len(seats)+len(t))/32 <= n_sessions`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_nAgeza-paribhASenduzekhara.txt

## S4-067 Gaṇa plus dvandva

A list cited as a whole fails if a coordinated member is missing. The fetched gaṇa sūtra and the fetched dvandva sūtra together require every and-conjunct of the query to sit in the list, or the list scores 0 and cannot take the summary slot from a partial member. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `gana_credit = |members| if every coordinated query id is in the gaṇa, else 0`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-068 Bahuvrīhi plus avacchedaka

An external referent is named only when the compound's members are under one limitor. The fetched bahuvrīhi sūtra and the fetched avacchedaka line together admit the next turn only if the cue turn's ids are a subset of the query and sit in one clause. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `admit(t+1) iff content(t) ⊆ query and one clause holds all of content(t) and content(t+1) meets the query`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-069 Trairūpya plus kārya-hetu

Smoke is a reason for fire only when the three forms hold. The fetched effect example and the fetched trairūpya line together admit the empty-intersection successor only when the cue turn's bridge is present in a similar case and absent from the dissimilar cases. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `admit(t+1) iff disjoint(t, t+1) and sapaksa(bridge) >= 1 and vipaksa(bridge) = 0`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-070 Anvita within and across

A sentence is the connection of word-meanings, which Jaimini 1.1.5 states, and the two composition rules are applied without pretending the fetched kārikā names them. A cue scores only inside a clause that holds another cue, and the session sum of those clause scores admits a turn whose own text holds only one hop. That is the Multi-Hop pool hole. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `clause_score(c) = |query ids in c| if that count >= 2 else 0; admit t if sum of clause scores in its session >= 2`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_jaimini-mImAMsAsUtra.txt

## S4-071 Adhikāra plus pratyāhāra

A heading and a pratyāhāra together name a governed span by its two ends. The fetched adhikāra sūtra, the fetched aic abbreviation, and the fetched padapāṭha support one pool key: either end retrieves every turn inside the frame. Middle-only queries do not. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `key = (rare(frame), last_id(frame)); hit either => admit all positions in the frame`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-072 Vipratiṣedha plus aṅga

Equal rules yield to the later one, and the principal is what the act accomplishes. The fetched vipratiṣedha sūtra and the fetched Jaimini line together break an equal-coverage tie by who holds the last query id, and only then by who matches the later group. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `among equal cover, prefer has(head); then prefer max later-group index`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_jaimini-mImAMsAsUtra.txt

## S4-073 Ekaśeṣa plus saptabhaṅgī

One residue stands for the repetitions, and the conflicting claims still each need a seat. The fetched ekaśeṣa sūtra and Āptamīmāṃsā sam_3 together collapse repeated ids to a count and still fill the summary top 5 with distinct predicate classes. Ceiling class K_PULL: inserts an uncovered pool turn into the five. BEAM Turn All@10 is 138 against 320 and LME has no turn All@10 bit, so the quota finishes All@5 only after admission has put the other golds in the pool.

Math: `post repeated id once with count; seat distinct bhaṅgī classes until 5`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_samantabhadra-AptamImAMsA.txt

## S4-074 Yogavibhāga plus simile field

The illustration is not the thesis. Fetched Nyāyasūtra 1.1.36 defines the example, and the Kaumudī's sectioning supplies the split. Tokens after as or like are their own field with their own length, and a summary query that is not itself a simile matches only the thesis field. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `score = thesis_bm25 if query has no as/like, else example_bm25; length = that field's length`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gautama-nyAyasUtra.txt

## S4-075 Alternation plus hrasva/dīrgha

Short and long are the fetched vowel-length classes, counted here by token length. A summary slot in an alternating session goes to a long turn only when its long-token count equals the sum of long-token counts of the short turns already seated. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `summary seat iff long and dirgha(t) = sum dirgha(short seats)`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-076 Correction plus viruddha

A contradictory reason is replaced by the turn that denies it and adds what was missing. The fetched viruddha line supplies the count test, and the correction has to share an id, carry a negator, and add an id. Both are required. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `seat correction iff extra >= 1 and shared >= 1 and negator and vipaksa > sapaksa`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-077 Ṭīkā depth plus bahuvrīhi

The second commentary points at something the words do not contain. Against the fetched Mahābhāṣya and the fetched bahuvrīhi sūtra, a depth-2 turn licenses admission of the next turn only when that next turn shares no query id. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `admit neighbor iff depth(t) = 2 and content(neighbor) ∩ query = ∅ and neighbor nonempty`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pataJjali-vyAkaraNamahAbhASya.txt

## S4-078 Limb plus avayava agreement

The five debate roles and the five members of a proof are the same order in the fetched Nyāyasūtra. Summary seat i is the first pool turn whose earliest orthographic limb test equals i, so a doubt cannot occupy the decision seat. Ceiling class K_PULL: inserts an uncovered pool turn into the five. BEAM Turn All@10 is 138 against 320 and LME has no turn All@10 bit, so the quota finishes All@5 only after admission has put the other golds in the pool.

Math: `seat i gets the first pool turn whose first matching limb test is i`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gautama-nyAyasUtra.txt

## S4-079 Vyāpaka-anupalabdhi

Absence of the pervader blocks the inference. The fetched kārya-anupalabdhi line and the perception-condition line together reject a session that has the rarer cue and lacks the more frequent one, and they admit an absence only when the other cues, including the more frequent id, are actually there. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83). Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `reject session if it has lower-df and lacks higher-df; admit absence only if other ids and higher-df are present`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-080 Ekādeśa plus lopa place

A single substitute keeps one id, and lopa keeps the place. The fetched ekādeśa and lopa sūtras fuse two adjacent content ids into the first id while the stored place count stays 2. The marginal-mark search returned nothing usable and is not the mechanism. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `post(first); places = 2; second not posted`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-081 Volume boundary plus asiddha

A new volume does not see the previous volume's operations. The fetched Derge extent of 213 volumes and the fetched pūrvatra asiddham together drop inherited feats at a session boundary, while a heading inside the session still governs. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36). Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `if session(t) != session(t-1) then inherited feats from t-1 are not posted on t`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://ldspdi.bdrc.io/resource/MW23703.json

## S4-082 Mūla plus ekaśeṣa

The root keeps the words, and the commentary is what was added. The fetched Mahābhāṣya opening and the fetched ekaśeṣa sūtra together post the shared ids once, on the short turn, and post only the set difference on the long turn, so a summary query retrieves the commentary rather than a second copy of the root. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `post(shared) on the short turn only; post(long \ short) on the long turn`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pataJjali-vyAkaraNamahAbhASya.txt

## S4-083 Change-point plus trairūpya

The reason has to be the thing that appears on only one side of the boundary, and it still needs three forms. The fetched trairūpya line and the fetched eleven-kind absence list together admit that one-sided token only when the similar count is positive and the dissimilar count is zero. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `bridge iff one-sided at a distance-1 boundary and paksa and sapaksa >= 1 and vipaksa = 0`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## S4-084 Sphoṭa plus dvandva

An undivided utterance holds its words in one window, and a coordination holds both members. Vākyapadīya 1.50 and the fetched dvandva sūtra together: without and, slot 1 must put every query id in one clause; with and, both members must be somewhere in the top 5. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `no coordinator => one clause holds all ids; coordinator => both members present in the top 5`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_bhartRhari-vAkyapadIya.txt

## S4-085 Lakṣaṇā plus anupalabdhi

Indication runs when the literal is absent and the conditions for noticing that absence are met. The fetched Dhvanyāloka and the fetched anupalabdhi conditions together post a secondary id only in a session that already holds the other cues and still has no turn containing all of them. Ceiling class ADMIT: a t_scene>0 turn can enter the official list. Turn@1 ceiling is the pool (LoCoMo pool@50 1415 clears 1383; BEAM pool@50 328 clears 320; Multi-Hop pool@50 243 is short of 254; Open-Domain pool@50 63 is short of 83).

Math: `post secondary iff literal intersection empty and other query ids present in the session`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 5633-5730. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_Anandavardhana-dhvanyAloka.txt

## S4-086 Sakṛd plus vipratiṣedha

Paribh_39, fetched, already says a once-applied operation and a conflict rule in the same line. The seat that was not in the searches does the retrieval version: of two turns that would newly cover the same summary group, one is seated, and it is the one that matches the later query group. Ceiling class K_PULL: inserts an uncovered pool turn into the five. BEAM Turn All@10 is 138 against 320 and LME has no turn All@10 bit, so the quota finishes All@5 only after admission has put the other golds in the pool.

Math: `if two turns add the same new group, seat the one with the greater later-group index, once`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_nAgeza-paribhASenduzekhara.txt

## S4-087 Gaṇa plus pratyāhāra

A list is an abbreviation of the session only when it holds both ends. The fetched gaṇa sūtra and the fetched aic abbreviation together store a bullet or comma list as one object only if it contains the session's first and last content ids. Ceiling class FEAT: a new posting can move no_pool. The binding no_pool holes are LoCoMo Open-Domain (wide pool 78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).

Math: `valid_gana iff list_len >= 3 and has session-first id and has session-last id`

Tag: RECALL/INDEX. Hook: `crates/vestige-core/src/retrieval.rs` 3121-3144. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_pANini-aSTAdhyAyI.txt

## S4-088 Saṃśaya yields to siddhānta

Doubt is not the decision. The fetched saṃśaya label and the siddhānta label together move slot 1 from a question turn to the first non-question top-5 turn that covers the same groups, and they leave the question in the five. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `if head is_question and some s > 0 is not a question and cover(s) ⊇ cover(head) then seat s`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_gautama-nyAyasUtra.txt

## S4-089 Hrasva count plus grantha

Extent and vowel length are both integer checks. The fetched vowel-length sūtra and the fetched Derge extent together keep a summary head only when its long-token count equals the sum across the seated sessions and its grantha does not exceed the session count. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move. Session tool: LoCoMo Session@5 is 1416 and clears 1383; BEAM Session@10 is 275 and is short of 320; LME preference Session@5 is 26 and is short of 27 while Session@10 is 30.

Math: `legal iff dirgha(t) = sum dirgha(sessions) and content_len(t)/32 <= n_sessions`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://ldspdi.bdrc.io/resource/MW23703.json

## S4-090 Dūṣaṇa plus anaikāntika

Pointing out a fault counts when the fault is a reason that is too wide, and the reply has to drop that reason. The fetched dūṣaṇa line and the fetched anaikāntika line together move slot 1 only to a top-5 turn that lacks the wide bridge, shares something else, and carries a negator. Ceiling class SLOT: a permute of the current top 5. Turn@1 ceiling is Turn@5 (LME 428 clears 423; LoCoMo 1127 is short of 1383; BEAM 238 is short of 320). All@5 does not move.

Math: `seat challenger iff df(bridge) > df(rarest) and challenger lacks bridge and shares some other id and has negator`

Tag: ORDERING. Hook: `crates/vestige-core/src/retrieval.rs` 6265-6272. Record: https://gretil.sub.uni-goettingen.de/gretil/corpustei/transformations/plaintext/sa_dharmakIrti-nyAyabindu.txt

## Top-10 build order

Ordered by the gaps in TARGET_90_ALL: LoCoMo Turn@1, BEAM All@5/All@10, BEAM and LoCoMo Session@1, LME Turn@1 and Turn All@5.

1. **S4-070 Anvita within and across** (ADMIT) — LoCoMo Turn@1 and LME multi-session All@5. Clause-connected cues sum across turns, which is the pool_not_k mass (Single-Hop 95, Multi-Hop 83, Temporal 53) and the second gold of a multi-session question.
2. **S4-026 Yogavibhāga clause length** (FEAT) — Length. A long gold is scored by its best clause, which is how Single-Hop and BEAM golds fall out of t_scene and out of All@5.
3. **S4-066 Sakṛd plus grantha quota** (K_PULL) — BEAM All@5/All@10 and LME All@5. A post-seat addition must cover a group the five do not already cover. All@10 on BEAM is 138 against a need of 320, so the added turn has to come from outside the current ten.
4. **S4-013 Kārya-hetu empty successor** (ADMIT) — Empty-intersection successor enters the pool. LoCoMo pool@50 is already 1415 against a Turn@1 need of 1383; the differently worded next turn is what has to become one of those 1415 and then sit at rank 1.
5. **S4-042 Bahuvrīhi plus kārya-hetu** (ADMIT) — The same successor, only when the cue turn is a subset of the query. This is the Multi-Hop and Open-Domain case, whose pool@50 (243 and 63) is still under 254 and 83.
6. **S4-064 Kośa bits plus anukramaṇī** (FEAT) — Session admission by five bits then L1. BEAM Session@10 is 275 against 320. LoCoMo Session@5 is 1416 against 1383, so the bench is an ordering problem, and Open-Domain (65<83) and Multi-Hop (246<254) are not.
7. **S4-011 Trairūpya three-count gate** (SLOT) — LME Turn@1. The bench top-5 ceiling is 428 against 423. Preference is the category that still needs the pool (turn@10 is 19 against 27).
8. **S4-053 Limb quota plus saptabhaṅgī** (K_PULL) — BEAM All@5 once a second gold is in the pool. event_ordering and summarization are All@5 0 and All@10 0, so the quota has nothing to diversify until admission runs.
9. **S4-038 Pañcāvayava example link** (ADMIT) — LME multi-session All@5 (53 against 109). The example turn shares a mid-frequency id and lacks the rarest, which is the other session's gold.
10. **S4-012 Anupalabdhi under conditions** (ADMIT) — Conditioned absence. The categories whose wide pool is itself under 90% are LoCoMo Open-Domain (78<83), BEAM preference (30<36), and BEAM information_extraction (32<36).
