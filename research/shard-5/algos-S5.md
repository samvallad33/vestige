# Shard 5 algorithms

Base SHA `e2ecd1da037974c1835f563ad8b046f4a4ca3629` on `perf/c2-step33-triangulation`. Research designs only. No engine edit is implied by a sketch.

Primary slices: BEAM event_ordering and LoCoMo Open-Domain. Traditions: Chinese, Korean, Japanese, and Vietnamese bureaucratic, historiographic, divinatory, and archival practice.

47 of 90 combine two mechanisms. For each of those, Crossref and OpenAlex were queried and the top titles were read. The novelty cell records the query, the HTTP status, the API total, and those titles, and states that no prior combination was found in those searches.

## Top 10 build order

Build the pool-admission rules before any slot-1 swap. Event-ordering Turn@1 of 36/40 is reachable inside the scored pool (39/40) and is not reachable inside the current top 5 (14/40) or official k (22/40). Open-Domain Turn@1 of 83/92 is not reachable by ordering: the scored-pool ceiling is 78/92, so at least five currently unscored golds have to gain a match.

1. `S5-048` Date-keyed session cap — BEAM event_ordering. cap_key=(session, date_ord) if date_ord≠0 else (session, 0); at most one top-5 seat per key
2. `S5-049` Date-keyed topic repeat — BEAM event_ordering. topic_repeat = date_ord equal AND the existing jaccard test; if date_ord differs then topic_repeat=0
3. `S5-050` Outline line does not spend the cap — BEAM event_ordering. if has_temporal and content_len<=12 and lead_digit then cap_increment=0
4. `S5-051` Adjacent-day episode gate — BEAM event_ordering. apply the existing episode boost only if |date_ord(a)-date_ord(b)|=1
5. `S5-052` Office-title symmetric difference — BEAM event_ordering. sym=|titlecase(a) Δ titlecase(b)|; sym=0 ⇒ topic_repeat=1; sym=1 ⇒ topic_repeat=0
6. `S5-053` Chronicle date-head inheritance — LoCoMo Open-Domain. for t after head h until the next date-initial turn: date_feats(t) ∪= date_feats(h)
7. `S5-054` Second era year feat — LoCoMo Open-Domain. for each 4-digit year y on the turn, emit the existing date feat of y
8. `S5-055` Bundle-tag donation — LoCoMo Open-Domain. if |content(t)|<=4 and content(t)⊆content(next) then feats(next) ∪= content(t) and candidate(t)=0
9. `S5-056` Reign-ordinal year index — LoCoMo Open-Domain. if lead_ord>0 and has temporal 'year' and no 4-digit then year_index=min_year(session)+lead_ord-1
10. `S5-057` Raw article ordinal feat — LoCoMo Open-Domain. emit feat(lead_ord) as printed; match iff query digit = lead_ord

## Risks

These are measured limits, not a reason to aim lower. Open-Domain 90% needs recall of at least five no_pool golds plus seating of every pooled gold. The attached rows do not say which of the 14 no_pool golds failed for lack of a date, an ordinal, or a header, and which share no orthographic token with the query. A closed-class shape rule cannot invent a content match from nothing. The recall rules below only restore a token the ingest currently drops or fails to attach: a governing date line, a second year, a raw article integer the year-guard ignores, a bundle tag, a layer mark, an address triple. Event-ordering rules that clear a collapse when date ordinals differ can also split two golds that are true repeats of one event; the monograph form (shared document-frequency-1 token) is the tighter variant if the broad form over-fires. None of these rules retune 0.45, 0.38, 0.42, 0.15, 0.25, 0.52, or 0.65. They add an integer gate around those existing tests. They do not read `TurnIx.body`, do not call `rsb_seat_behind`, and do not delete `HYPERNYM_LATTICE`.

## Algorithms

### S5-001 Monograph date split

ORDERING for BEAM event_ordering (pool_not_k 17; k_not5 8). Chinese historiographic, jishi benmo, Southern Song, Yuan Shu, 1174. Yuan Shu's Tongjian jishi benmo (KR2c0001 in the fetched Kanripo 紀事本末 catalog) breaks the chronicle into event monographs and keeps every dated entry of that event. Two turns are one monograph only when they share a content id whose document frequency is 1. Inside that pair, unequal date_ord values are successive entries, not duplicates: topic_repeat is forced false and the one-per-session cap does not apply. Equal date_ord still uses the existing jaccard test. Ranking reads the integer fields written at ingest. It does not retune the 0.45 threshold. Catalog note: KR2c0001.

Combined mechanisms: jishi-benmo thread (shared df=1 token) + distinct date ordinals stay separate events.

Math: `thread(a,b)=1 iff exists t in content(a)∩content(b) with df(t)=1; topic_repeat=0 and cap_exempt=1 iff thread=1 and date_ord(a)≠date_ord(b)`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not catch-lines (no first-line/last-line overlap), not Petrie seriation (no date-spread objective), not Khipu (no second query), not annals-epitome equal strata.

### S5-002 Outline and witness seat

ORDERING for BEAM event_ordering (k_not5 8; pool_not_k 17). Vietnamese and Japanese historiographic, cương mục and kōbun, Nguyễn, Tự Đức commission; Dai Nihon Shiryō compilation. Khâm định Việt sử Thông giám cương mục puts a short dated outline line over a detail, and Dai Nihon Shiryō's 綱文 does the same with the following quotations as witnesses. A turn with has_temporal and content.len()<=12 whose first content token is a digit is the outline. It does not consume max_per_session_top5. Following turns are details until the next outline. A detail whose first byte is a quotation mark is a witness and cannot take the event seat. Exactly one non-witness detail, the one with the largest count of query content ids, may take that session's seat. The outline itself stays eligible and does not spend the cap. Catalog note: Q6770796 Khâm định Việt sử Thông giám cương mục.

Combined mechanisms: cương/mục outline line + kōbun headline whose following quotations are witnesses.

Math: `outline(t)=has_temporal(t)∧content_len(t)<=12∧digit(first(t)); witness(t)=first_byte is quote; cap_count(t)=0 if outline(t) or witness(t)`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. Not Abusir header inheritance (that copies markdown table headers into row feats). This only decides which turn spends the event cap. Not method of loci.

### S5-003 Reading-order event identity

ORDERING for BEAM event_ordering (k_not5 8). Japanese kanbun reading and Korean gugyeol, Nihon Shoki manuscripts read by kundoku; Samguk sagi read with gugyeol. Nihon Shoki is read by moving clauses according to kaeriten marks, and Korean gugyeol inserts short grammatical marks between content words. Clause identity is the bracketed integer already on the token, not surface order. Event identity is the pair (session, minimum bracket ordinal among clauses that contain a query content id). Tokens whose character length is 2 or less are removed from the set used for the jaccard test only; they still match queries. The session cap keys on that identity, so two bracket ordinals in one session are two events. Catalog note: Q813935 Nihon Shoki.

Combined mechanisms: kaeriten bracket permutation + gugyeol short-token drop from the event set.

Math: `event_id=(session, min {bracket_ord(c) | c contains a query content id}); jaccard uses {t in content | chars(t)>=3}`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not Step 9 method of loci and not first_move_num. Not ergative argument order. The 0.45 constant is untouched; only the set it compares changes.

### S5-004 Dual era keys

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14; pool_not_k 35). Korean historiographic, Joseon sillok, Joseon annals, 1413-1865. Joseon sillok entries record the reign year and, in diplomatic notices, the Ming or Qing era year. When a turn contains two 4-digit years, both are emitted as date feats through the existing inline-date path, at that path's existing weight. date_lo and date_hi are stored. A query year matches either feat, which is how an unscored turn becomes retrievable. For an event-sequence query the order key is date_lo, and the pair remains one event. No new weight is introduced. Catalog note: Q497030 Veritable Records of the Joseon Dynasty.

Combined mechanisms: index both era years + order the event by the earlier year.

Math: `Y=years(t); if |Y|=2 then emit both; date_lo=min Y; date_hi=max Y; event_key=date_lo`

Hook `crates/vestige-core/src/retrieval.rs` lines 3229-3238. Not ship's-logbook alignment and not regnal two-pass retrieval. Both years are indexed at ingest; there is no second query.

### S5-005 Parenthetical month and unfilled day

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese annalistic, Bamboo Annals and the founding diary, Bamboo Annals; Tang, Wen Daya, 617-618. The Bamboo Annals mark an intercalary month inside the year rather than as a new year, and Wen Daya's Da Tang chuangye qiju zhu (KR2b0006 in the same fetched catalog) leaves a day blank instead of borrowing the neighbor. A year or day token inside parentheses stores as paren_ord and does not advance date_ord. The episode backward boost is skipped unless the two sessions' date_ord values differ by exactly 1. The existing decay constants are not written. Catalog note: KR2b0001.

Combined mechanisms: intercalary date stays in the parent bucket + qiju diary does not bridge a missing day.

Math: `if token in parens then date_ord unchanged; boost_ok=1 iff |date_ord(a)-date_ord(b)|=1`

Hook `crates/vestige-core/src/retrieval.rs` lines 5458-5471. Not Harris matrix temporal ordering and not a ship's-log score bonus. The boost constants 0.52 and 0.65 stay. The gate is the integer day gap.

### S5-006 Table cell holds the day

ORDERING for BEAM event_ordering (k_not5 8). Chinese historiographic, Shiji tables, Western Han, Sima Qian; Qing commentary on the ten tables. The Shiji tables, and Wang Yue's Du Shiji shibiao (KR2a0005), put one short cell at each date instead of the biography. Among turns with the same date_ord, the turn with the smallest content.len() that has form_digit or has_temporal is the cell. Only that cell consumes the session cap for that date. A longer turn of the same date does not enter the top 5. A different date_ord still can. This is the opposite of preferring a long summary. Catalog note: KR2a0005.

Combined mechanisms: shortest biao cell at a date + one cell spends that date's event cap.

Math: `cell(d)=argmin {content_len(t) | date_ord(t)=d ∧ (form_digit∨has_temporal)}; cap(t)=1 iff t=cell(date_ord(t))`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. Not primary_aggregate, which yields a turn of three or more dates to a single-transaction turn. Not Linear B residual. Length is the key only inside one date_ord.

### S5-007 Text date over copy date

ORDERING for BEAM event_ordering (t5_not1 7; pool_not_k 17). Chinese archival, Dunhuang manuscripts, Dunhuang manuscript cache, 4th-11th century. Dunhuang manuscripts carry the text's date and a later colophon date on the same roll. When a turn has two or more date ordinals, event_ord is the first and copy_ord is the last. Event identity, the session cap, and slot order use event_ord. copy_ord is stored and is not a diversity key, so a shared copy date does not glue two texts into one event and does not split one text into two. Catalog note: Q276864 Dunhuang manuscripts.

Combined mechanisms: first in-text date is the event + last date is the copy date and is ignored.

Math: `if |dates(t)|>=2 then event_ord=dates[0], copy_ord=dates[last]; identity=event_ord`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not quote-credit, which moves quoted tokens to an earlier turn. Both dates stay on this turn; only the key changes. Nearest search title 'The Afterlife of Avestan Manuscripts: Colophons and Marginal notes' discusses colophons and does not state this key split.

### S5-008 Procedure ordinal seat

ORDERING for BEAM event_ordering (t5_not1 7). Korean bureaucratic, uigwe, Joseon royal protocols. Uigwe record a rite as numbered steps under a date. For an event-sequence query, among the already seated top 5, the seat key is the pair (date_ord, lead_ord). The unique minimum pair that also has has_temporal moves to slot 1. A turn without a leading integer sorts after every turn that has one. This runs only when is_event_sequence, after Step 33, and does not call rsb_seat_behind. It does not compare scene scores. Catalog note: Q1073141 Uigwe.

Combined mechanisms: uigwe step ordinal + annal date spine as a two-integer seat key.

Math: `key(t)=(date_ord(t), lead_ord(t)); slot1=argmin key over top5 where has_temporal=1 and lead_ord>0`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not method of loci and not Step 9's first_move_num walk. The key is the written ordinal, not a path through the session.

### S5-009 Class and length pair

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Chinese bibliographic, Siku and Yongle, Ming Yongle; Qing Siku, 1773-1782. Siku filing is a class decision, and the Yongle dadian lookup needs both keys, not one. Shape is an integer: 1 if the first content token is a 4-digit year, 2 if list_item_count>0, otherwise no pair is emitted. The second key is the length bucket of that first token: 0 if its character length is at most 3, 1 if at most 6, 2 otherwise. A pair feat is emitted only for shapes 1 and 2. The query receives the same pair only when its own first content token has that shape. Both integers must equal. Prose-to-prose matching is not added, so the feat cannot pull every paragraph into the pool. Catalog note: Q699477 Siku Quanshu.

Combined mechanisms: Siku shape class + Yongle two-key (script block and length bucket), both required.

Math: `emit pair (shape, bucket) only if shape in {1,2}; match iff shape_q=shape_t and bucket_q=bucket_t`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not scribal-hand signature seats and not tadanta head-final. No radical table is introduced. Content BM25 is unchanged.

### S5-010 Kaoyi digit witness

ORDERING for BEAM event_ordering (t5_not1 7). Chinese historiographic, Zizhi tongjian kaoyi, Northern Song, Sima Guang, presented 1084. Sima Guang's Zizhi tongjian kaoyi (KR2b0008) keeps a third source when two dated accounts disagree. When is_event_sequence and the first two seated turns have unequal date_ord, a third seated turn with content.len()<=12 decides. The decision is the integer count of shared digit-token ids, not rare content tokens and not scene score. The side with the strictly greater count moves to slot 1. A tie changes nothing. Catalog note: KR2b0008.

Combined mechanisms: kaoyi third-source adjudication + the witness is counted on shared digit tokens only.

Math: `if date_ord(s0)≠date_ord(s1) and exists t3 with content_len<=12: slot1=argmax_{i in {0,1}} |digits(si)∩digits(t3)|, strict`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not ancient third-text arbitration: no 3% scene band, no idf, and the witness must share digit ids. Event sequences only.

### S5-011 Pointer plus block address

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Japanese ritual code and Korean canon address, Engishiki, 927; Tripitaka Koreana block, 13th century. Engishiki cross-references a rite by pointing at another article, and the verified CBETA file K05n0016 is addressed as canon K, volume 5, number 16 rather than by a search of its prose. A turn whose content is only the closed-class word 'see' plus an integer donates its other feat ids to the same-session turn whose lead_ord equals that integer, and the pointer is not a candidate. Separately, three integers in a dotted or slash-separated token are one address id. A query with the same three integers matches that id. 'see' is the one closed-class word already in the grammar; no synonym list is added. Catalog note: Q1342448 Engishiki.

Combined mechanisms: 'see' plus an ordinal pointer + canon block triple as an exact address.

Math: `if content={see, n} then feats(target[lead_ord=n]) ∪= feats(pointer); addr=hash(d1,d2,d3) exact`

Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3145. Not Khipu two-hop and not uncovered-entity rescue. The pointer is an explicit 'see' plus an integer, not a second query of rare capitalized tokens.

### S5-012 Finer chronicle over the digest

ORDERING for BEAM event_ordering (t5_not1 7). Chinese historiographic, long draft and veritable record, Southern Song, Li Xinchuan, Jianyan yilai xinian yaolu. Jianyan yilai xinian yaolu keeps a denser year-by-year draft than the condensed veritable record. For an event-sequence query, among the seated top 5, the turn with the strictly greatest number of distinct date ordinals takes slot 1 when it contains every query content id the current head contains. If two turns tie on that count, a turn with at least one date ordinal outranks a turn with none. Undated turns are not deleted from k. Catalog note: 建炎以來繫年要錄 ctp:wb907864 books=1.

Combined mechanisms: higher date-token cardinality wins + a dated line beats an undated line with the same query ids.

Math: `slot1=argmax |date_set(t)| over top5, subject to query_ids(head)⊆content(t); tie break |date_set|>0`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Inverse of primary_aggregate, which prefers the single-date transaction over a summary of three or more dates. This fires only for is_event_sequence.

### S5-013 Draft bracket and last verdict

ORDERING for BEAM event_ordering (pool_not_k 17; t5_not1 7). Chinese bureaucratic and legal, Ming record and Han case slips, Ming shilu compilation; Zhangjiashan slips, 196-186 BCE. A rescript keeps the decision and drops the draft, and the Zhangjiashan case slips put the finding after the interrogation. A line whose first character is '[' is a draft: its tokens are excluded from event candidacy and from the jaccard set. Among the remaining turns of the session, the last turn that is not a question and has form_digit is the verdict. That verdict is exempt from the one-per-session cap. Earlier question turns are not the event seat. Catalog note: Q5931471 Zhangjiashan Han bamboo texts.

Combined mechanisms: bracketed draft lines are not the event + the last non-question digit turn is the verdict.

Math: `draft=line starts with '['; verdict=last t in session with ¬is_question∧form_digit; cap_exempt(verdict)=1`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not Quaker-minute latest-modal supersession and not model-text templates. The trigger is a bracket character plus the last non-question digit turn, not recency of the whole document.

### S5-014 Shared mark and endorsement count

ORDERING for BEAM event_ordering (pool_not_k 17). Japanese archival, Tō-ji Hyakugō documents, Tō-ji archive, 8th-18th century, Memory of the World. The Tō-ji Hyakugō monjo are documents that passed through an office and picked up endorsements. A token that contains no letter and no digit is a mark. Two turns that share one such token are one event: the later one does not take its own slot. The order key of the cluster is the count of line-final mark tokens, ascending, so the copy with fewer endorsements is the earlier stage. Slot order among seated event turns follows that integer. Catalog note: Q42412865 Toji Hyakugo Monjo.

Combined mechanisms: a shared non-alphanumeric mark clusters two turns + endorsement count is the route position.

Math: `cluster if mark(a)∩mark(b)≠∅; order_key=|line-final marks|; one slot per cluster`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not the Jin Hou Su bell join. That joins turns when terminal punctuation is missing. This joins turns that share a symbol token, and orders them by the integer count of those tokens.

### S5-015 Article number replacement

ORDERING for BEAM event_ordering (pool_not_k 17). Japanese and Korean statutory, Goseibai and Sokdaejeon, Kamakura 1232; Joseon supplement to the great code. Goseibai shikimoku and Sokdaejeon amend by article number. A later turn whose leading integer equals an earlier turn's leading integer replaces only the candidate with that integer. Every other leading integer in the session stays in the candidate set. The replaced turn is removed from event candidacy. This is not a latest-document win. Catalog note: Q63889 Goseibai Shikimoku.

Combined mechanisms: Goseibai article identity by leading integer + Sokdaejeon replaces only that article.

Math: `if lead(a)=lead(b) and b later than a then candidate(a)=0 and candidate(b)=1; other leads unchanged`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not Torrens supersession, which replaces the whole document. Only the matching leading integer moves.

### S5-016 Year then rank

ORDERING for BEAM event_ordering (t5_not1 7). Korean epigraphic, Gwanggaeto stele, Goguryeo, 414. The Gwanggaeto stele orders campaigns by reign year and then by the sequence inside the year. For an event-sequence query the seat key among the top 5 is the pair (date_ord, lead_ord). The unique strict minimum takes slot 1. A missing lead_ord sorts as a larger sentinel, so an explicit rank beats an unnumbered paragraph of the same year. No score ratio is consulted. Catalog note: Q701096 Gwanggaeto Stele.

Combined mechanisms: reign-year header + a second integer as the rank inside that year.

Math: `key=(date_ord, lead_ord or SENTINEL); slot1=unique argmin key`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not method of loci. Two written integers, not a walk along session position. Distinct from the uigwe rule, which also requires has_temporal on the winner and ignores turns with lead_ord 0 entirely rather than sentinel-sorting them.

### S5-017 Quoted date, ignored tail

ORDERING for BEAM event_ordering (t5_not1 7). Chinese manuscript, Dunhuang colophons, Dunhuang, 4th-11th century. On a Dunhuang roll the quoted source carries the event date and the note after the quotation carries the copy date. event_ord is the first date ordinal that sits inside a quoted span. A date ordinal that occurs only after the last quotation mark is copy_ord and is not used for identity or for the cap. If there is no quoted date, event_ord falls back to the first date that is not after the last quote. Turns with no quote are unchanged. Catalog note: Q276864 Dunhuang manuscripts.

Combined mechanisms: a date inside quotation marks is the event date + a date after the closing quote is the colophon.

Math: `event_ord=first date inside quotes; copy_ord=date after last quote; identity=event_ord`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not quote-credit, which reassigns the quoted tokens to an earlier turn. Dates stay on this turn. The Avestan colophon paper in the search titles does not state this split. Distinct from the first-versus-last rule, which does not look at quote marks.

### S5-018 Changed line on the query cycle

ORDERING for BEAM event_ordering (pool_not_k 17; t5_not1 7). Chinese divinatory, Zhouyi, Zhouyi, received text. A Zhouyi line that does not change is the same situation, not a new event. A turn is a new event only when the symmetric difference of its content ids with the previous same-session turn is nonempty. Independently, when the query contains two digit tokens and the second is between 2 and 12 inclusive, slot 1 is allowed only for a turn whose date_ord modulo that digit equals the query's first digit modulo that digit. The modulus comes from the query. No cycle length is hardcoded. A turn that fails the modulus may stay in k. Catalog note: 周易 ctp:book-of-changes books=499.

Combined mechanisms: a changed line is a new event + slot 1 must match the query digit modulo the query's own second digit.

Math: `new=|content(t) Δ content(prev)|≠0; if q2 in 2..=12 then slot1 requires date_ord mod q2 = q1 mod q2`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not office-title symmetric difference: this difference is the full content set. The modulus is the query's second digit, not a stored 12 or 60.

### S5-019 Later clock and witness rank

ORDERING for BEAM event_ordering (t5_not1 7). Han postal slips, Juyan, Han, Juyan, 2nd century BCE-2nd century CE. Juyan postal slips record a dispatch time and a receipt time, and the names at the end of the line are the people who handled it. When a turn has two clock-shaped tokens, event time is the later one. Among seated event turns, the order key is that later clock, then the count of line-final title-case tokens ascending. The unique minimum takes slot 1. No new time-word list is added; form_clock is the existing shape flag, and the second clock is a second token of that shape. Catalog note: Q1714726 wooden strips of the Han dynasty at Juyan.

Combined mechanisms: the later of two clock times is the event time + line-final title-case count is the witness order.

Math: `event_clock=max of the two clock ordinals; key=(event_clock, titlecase_tail)`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not ship's-logbook alignment, which adds a score bonus for explicit times. This only chooses which of two clocks is the key and then orders by an integer witness count.

### S5-020 Tag donation and governing date

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14; pool_not_k 35). Han slips, Juyan bundle tags, Han, Juyan. Juyan bundles carry a short tag whose words are a subset of the document, and a chronicle date line governs the lines under it until the next date line. A turn with content.len()<=4 whose content ids are a subset of the next turn's content is a tag: those ids are added to the next turn as inheritable feats using the source feat's existing weight, and the tag's event_candidate flag is 0. Separately, each following turn until the next date-initial turn receives the head's date feat ids. A query date can then match a detail line that never repeated the date. Carry's existing refusal to copy dates is the gap this fills, and only for these two shapes. Catalog note: Q1714726 wooden strips of the Han dynasty at Juyan.

Combined mechanisms: a short subset tag donates its tokens to the next slip + a date-initial line governs following lines.

Math: `tag if |content|<=4 and content⊆next; next.feats ∪= tag.feats; followers.date_feats = head.date_feats until next head`

Hook `crates/vestige-core/src/retrieval.rs` lines 3647-3680. Not the bell join (no punctuation test) and not Abusir header inheritance (the trigger is a date-initial line or a subset tag, not a markdown table header). Not ordinary carry, which skips dates.

### S5-021 Unclosed block numbers

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Korean woodblock canon, Tripitaka Koreana, 13th century; file dated in the TEI header 2025-03-12. The fetched TEI file is 高麗大藏經 number 16 in volume 5, addressed by that number rather than by a dense rank. The leading integer and any dotted number are stored as printed. A hole in the session's leading integers is left as a hole: later numbers are not shifted down, and the next turn does not inherit the missing number's feats. A query ordinal matches the raw integer only. Two turns that would become the same number after compaction stay different. Catalog note: K05n0016 高麗大藏經.

Combined mechanisms: the printed block number is stored raw + a missing number is not reassigned to the next block.

Math: `store lead_ord as printed; if n is absent, lead_ord n+1 stays n+1 and receives no feats from n`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not the Turin lacuna rule, which drops df=0 query groups from coverage. This keeps the printed integer and refuses to renumber.

### S5-022 Script block and length

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Chinese bibliographic, Yongle dadian, Ming, commissioned 1403. Yongle lookup fails if either key is wrong. Without a radical table, the two orthographic keys are the Unicode block of the first character and the character length of the first content token, bucketed at 3 and 6. A pair feat is emitted. It matches only when both integers equal the query's pair. Equality of the token string is still required for the content feat; the pair does not replace it. The pair adds a match for a query that shares the block and the bucket with a token the year-guard or the length filter would otherwise isolate. Catalog note: Q696077 Yongle Encyclopedia.

Combined mechanisms: first-character script block + character-length bucket, both required.

Math: `key=(unicode_block(first_char), bucket(chars)); match iff both components equal`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not the Siku class filter, which zeroes shapes. This pair never suppresses a content match. Not diacritic folding and not tadanta.

### S5-023 Era cut and reign arithmetic

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Japanese annalistic, nengō, Heian national histories, Nihon sandai jitsuroku, 901. The national histories cut the year at an era name. A capitalized token immediately followed by an integer of one to three digits increments an epoch counter at that point in the session. A turn that contains the existing temporal word 'year', has a leading ordinal, and has no 4-digit year stores year_index = session minimum 4-digit year + epoch + leading ordinal - 1. The query year matches year_index as a date feat through the existing date path. The arithmetic happens at ingest. There is no second retrieval pass. Catalog note: Q1052981 Nihon Sandai Jitsuroku.

Combined mechanisms: a capitalized token plus a short integer starts a new era + reign ordinal adds to the session's first year.

Math: `epoch increments at Capitalized+integer(1..3 digits); year_index=min_year(session)+epoch+lead_ord-1`

Hook `crates/vestige-core/src/retrieval.rs` lines 3229-3238. Not regnal-anchor two-pass retrieval. The year is computed once at ingest from integers already on the turn. No era-name list is added; the cut is Capitalized token plus a short integer.

### S5-024 Embedded chapter date

ORDERING for BEAM event_ordering (t5_not1 7; pool_not_k 17). Japanese and Vietnamese topical history, Ruijū kokushi, 892; An Nam chí lược, 1335. Ruijū kokushi files events under topics while the date sits inside the entry, and An Nam chí lược does the same inside chapters. The first line is a header when it contains no digit. Header tokens are not the event date. event_ord is the first date ordinal that occurs after that header. Slot order among the top 5 follows event_ord, not the turn index and not the header string. Two chapters with the same header and different embedded years stay in year order. Catalog note: Q1038639 Ruijū Kokushi.

Combined mechanisms: Ruijū topical header is not the date + An Nam chapter orders by the embedded year.

Math: `if first line has no digit then event_ord=first date after line 1; slot order = event_ord`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not session-topic incipit heads, which retrieve by the opening string. The header is ignored for the date key. Not gazetteer-plus-monograph, which also keeps a df=1 thread.

### S5-025 Second date splits the sheet

ORDERING for BEAM event_ordering (pool_not_k 17). Japanese archival, Shōsōin documents, Nara and later, Shōsōin documents. Shōsōin documents reuse a sheet, and a second dated entry on the back is not the same document as the first. A second date-initial line inside one turn opens event_ord_b. Both years are emitted as date feats, and the cap counts the turn as two events, so a later real event of the other year is not blocked by the first. This is the opposite of keeping the pair as one event ordered by the earlier year. Catalog note: Q11545362 Shōsōin Documents.

Combined mechanisms: a second date-initial line opens another event + both years stay indexed.

Math: `if date_heads(t)>=2 then keys={date_heads[0], date_heads[1]} and cap_slots(t)=2`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not petuchah, which splits a long turn at blank lines for coverage and keeps one turn score. This splits only at a second date-initial line and keeps both keys. Not the dual-era rule, which stores one event.

### S5-026 Prohibited day and conditional tail

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese divinatory and statutory, Shuihudi daybooks, Qin; Tang code commentary. Shuihudi daybooks mark days that are not to be used, and a Tang statute's operative clause is the condition. A date-initial turn whose raw tokens include a negator already in the engine's closed class ('not', 'no', 'never') is not an event candidate. Tokens after the last 'if', which is already a closed-class starter, are removed from the set used for event jaccard. They remain in the query-match feats. No new prohibition lexicon is added. Catalog note: Q689661 Shuihudi Qin bamboo texts.

Combined mechanisms: a negated date line is not an event + tokens after the last 'if' leave the event set.

Math: `if date_initial and negator in raw then candidate=0; jaccard_set = tokens up to last 'if'`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not a new word list. The negators and 'if' are the closed class already compiled into the engine. Not lacuna release of df=0 groups.

### S5-027 Generation depth then rank

ORDERING for BEAM event_ordering (t5_not1 7). Chinese genealogical tables and ranked offices, Shiji genealogical tables, Western Han; Tang liudian. The Shiji genealogical tables step down by generation, and the Tang liudian lists offices by rank number. Depth is the count of leading whitespace characters divided by 2. The seat key is (depth, lead_ord). Slot 1 is the unique minimum among the top 5. Date order is not the key. A turn with no indent and no leading integer does not win on this rule. Catalog note: KR2a0001.

Combined mechanisms: indent depth is the generation + the leading integer is the rank inside it.

Math: `depth=leading_spaces//2; key=(depth, lead_ord); slot1=unique argmin`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not method of loci and not the stele year-then-rank key. Depth replaces the year as the first component.

### S5-028 Position field and canon triple

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Korean statutory position and Korean canon address, Gyeongguk daejeon, 1485; Tripitaka Koreana block address. Gyeongguk daejeon articles sit in a fixed position, and the Koreana block is a number triple. The second digit token on a turn is stored as its own id, separate from a 4-digit year. A dotted triple is a second id. The pair feat matches only when the query has both the same second-digit value and the same triple. Either half alone does not emit the pair. This reaches a query that cites an article number the year-guard refused because it is not in 1900..2099. Catalog note: Q489676 Gyeongguk daejeon.

Combined mechanisms: the second digit token is its own field + a dotted canon triple must match with it.

Math: `pair=(digit_2, (a,b,c)); match iff both components equal the query pair`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not a hopae object: no identity-tablet record was verified, so the rule is not attributed to one. The second digit is positional, not a year-word. Not diacritic folding.

### S5-029 Numbered register inside a monograph

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese administrative, Tang liudian, Tang, 738. The Tang liudian is a numbered register of offices. When the leading integers of a session are strictly increasing, that series is the register. A later turn that shares a document-frequency-1 content id with a numbered turn and has no leading integer does not take an event slot. Numbered turns of different integers stay. The duplicate test is the hapax token, not jaccard. Catalog note: 唐六典 ctp:wb502694 books=3.

Combined mechanisms: strictly increasing leading integers mark the register + a shared df=1 token drops the unnumbered duplicate.

Math: `if lead sequence is strictly increasing and lead(b)=0 and exists t in content(a)∩content(b) with df(t)=1 then candidate(b)=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not the Liye household possessive binding. The corpus here is the Tang liudian. The test is a leading-integer series plus one shared hapax, not a pronoun.

### S5-030 Vermilion layer

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14; pool_not_k 35). Vietnamese bureaucratic, Nguyễn vermilion records, Nguyễn châu bản. Nguyễn châu bản keep the request and the vermilion decision on the same sheet. A line whose first character is '>' or '[' is the decision layer; other lines are the request. Decision-layer tokens are emitted under a layer id. Existing inference queries match the decision-layer id; question-shaped queries match the request-layer id. The event date is taken only from an unmarked line, so a date inside the marked line does not open a second event. The layer test is the first character, not an ALLCAPS word list. Catalog note: Q16481267 Châu bản triều Nguyễn.

Combined mechanisms: a marked line is the decision layer + the event date stays on the unmarked line.

Math: `layer=1 if first char in {'>','['} else 0; inference matches layer 1; event date from layer 0 only`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not the rubric ALLCAPS rule. The mark is a leading bracket or '>'. Inference uses the existing it.inference flag, not a new cue list.

### S5-031 Headline takes slot 1

ORDERING for BEAM event_ordering (t5_not1 7). Vietnamese historiographic, cương mục, Nguyễn, Tự Đức. In the cương mục the outline line is the event and the detail is commentary. For an event-sequence query, a seated turn with has_temporal and content.len()<=12 whose first token is a digit moves to slot 1 when it is the only such turn in the top 5. No longer turn of the same session may sit at slot 1 while that outline is seated. Details stay in k. This does not exempt the cap; the outline-and-witness rule does that. This one only fixes slot 1. Catalog note: Q6770796 Khâm định Việt sử Thông giám cương mục.

Combined mechanisms: the short date line is slot 1 + the following detail cannot be slot 1.

Math: `if unique t in top5 with has_temporal∧len<=12∧lead digit then slot1=t`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Distinct from the outline-and-witness rule, which refuses to spend the cap and lets the best detail hold the seat. Here the short line itself becomes slot 1.

### S5-032 Adjacent day, shortest cell

ORDERING for BEAM event_ordering (pool_not_k 17; t5_not1 7). Chinese diary and Shiji tables, Tang founding diary; Western Han tables. Wen Daya's diary does not chain across a missing day, and a Shiji table cell is the short entry at that date. The episode boost is skipped unless date_ord differs by exactly 1. Inside the day that is selected, slot 1 is the seated turn of that date_ord with the smallest content length. Longer turns of the day stay in k. The boost constants are not edited. Catalog note: KR2b0006.

Combined mechanisms: episode boost only across a 1-day gap + the shortest dated cell is the seat inside that day.

Math: `boost iff |Δdate_ord|=1; slot1(day)=argmin content_len over seated turns of that date_ord`

Hook `crates/vestige-core/src/retrieval.rs` lines 5458-5471. Distinct from the table-cell cap rule, which stops a long turn entering the top 5. This one only gates the neighbor boost and then picks the shortest already seated cell. Distinct from the one-per-day hard skip.

### S5-033 Appointment change, stable month

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese historiographic, Hanshu office table and Bamboo Annals, Eastern Han Hanshu; Bamboo Annals. The Hanshu table of offices changes when one office-holder changes, and the Bamboo Annals keep an intercalary month inside the year. Title-case token ids are the office names already detected as mid-sentence capitals. Symmetric difference of size 1 forces topic_repeat false even when jaccard is at least 0.45. Symmetric difference of size 0 forces topic_repeat true. A date token inside parentheses does not change date_ord, so an intercalary mark cannot by itself satisfy the new-date test. The 0.45 constant stays. Catalog note: Q846905 Book of Han.

Combined mechanisms: title-case symmetric difference of 1 is a new appointment + a parenthetical date does not count as a new date.

Math: `sym=|titlecase(a) Δ titlecase(b)|; if sym=1 then topic_repeat=0; if sym=0 then topic_repeat=1; paren date does not change date_ord`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not Domesday pair retrieval and not Torrens. The difference is a count of title-case ids. Not the full-content changing-line test.

### S5-034 Pointer replaces the article

ORDERING for BEAM event_ordering (pool_not_k 17). Japanese ritual cross-reference and Kamakura statute, Engishiki, 927; Goseibai shikimoku, 1232. Engishiki points at another article with a short cross-reference, and a Goseibai article is identified by its number. A turn whose content is the closed-class word 'see' plus an integer is removed from candidacy. Its content ids replace the content ids of the same-session turn with that leading integer, rather than being unioned. Other articles are untouched. A query that matched only the pointer now matches the article, and the pointer cannot occupy an event slot. Catalog note: Q1342448 Engishiki.

Combined mechanisms: 'see' plus ordinal moves its tokens onto that article + those tokens replace, they do not union.

Math: `if content(p)={see,n} then content(target[lead=n])=content(p) and candidate(p)=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not the address-triple pointer, which unions feats onto a block id. Not the Sokdaejeon rule, which replaces a later full turn rather than a 'see' pointer. Not Khipu.

### S5-035 Route order and the finalized copy

ORDERING for BEAM event_ordering (t5_not1 7). Japanese archival, Tō-ji Hyakugō, Tō-ji, 8th-18th century. Hyakugō documents accumulate endorsements as they move. Among seated turns with different date_ord values, order is the endorsement count ascending, which recovers the route. When two seated turns share a date_ord, the one with the strictly higher endorsement count takes slot 1, because it is the copy that finished the route. A tie leaves slot 1 alone. The count is the number of line-final symbol tokens, not a timestamp. Catalog note: Q42412865 Toji Hyakugo Monjo.

Combined mechanisms: ascending endorsement count is the route order + the higher count at one date is the copy that takes slot 1.

Math: `if date_ord equal then slot1=argmax symbol_tail, strict; else order by symbol_tail ascending`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not Torrens latest-document wins. The key is the mark count. Distinct from the cluster rule, which merges turns that share a mark into one slot.

### S5-036 Internal split, short tail dropped

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese manuscript practice, Dunhuang, Dunhuang, 4th-11th century. A Dunhuang sheet can hold two texts and a short dated note at the end. A second date-initial line that is not the last line opens a second event key. If the last line has fewer than 8 tokens and contains a date, that date is a colophon: it is not either key and it does not spend a cap slot. The Shōsōin split keeps every second date, including a trailing one. This one drops only the short tail. Catalog note: Q276864 Dunhuang manuscripts.

Combined mechanisms: a second internal date line splits the event + a final line shorter than 8 tokens is not a key.

Math: `keys={date heads except a final line with token_len<8}; cap_slots=|keys|`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Distinct from the Shōsōin two-key split, which keeps the second date even when it is the last line. Distinct from first-versus-last, which never opens a second key.

### S5-037 Ordinal lists do not share a cap

ORDERING for BEAM event_ordering (k_not5 8). Korean bureaucratic, uigwe, Joseon royal protocols. Uigwe steps are a numbered list, and each step is its own event. The session cap keys on (session, lead_ord) when lead_ord is nonzero, so two steps of one ceremony both enter the top 5. When the query itself contains a digit, a turn with list_item_count=0 is not eligible for those top-5 seats. It may still fill k after the top 5. The cap constant stays 1; the key changes. Catalog note: Q1073141 Uigwe.

Combined mechanisms: each leading integer is its own cap key + only list-shaped turns are eligible when the query has a digit.

Math: `cap_key=(session, lead_ord) if lead_ord≠0; if query has a digit then top5 requires list_item_count>0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. Distinct from the procedure-ordinal seat, which reorders slot 1 and does not change the cap key. Not method of loci.

### S5-038 Epoch year on a deictic line

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Vietnamese annalistic, Đại Việt sử ký toàn thư, Lê official history, compiled 1479 and later. Đại Việt sử ký toàn thư dates an entry by the reign year, and a following line says only 'that year' or 'the next year'. A turn with has_temporal, no 4-digit year, and a deictic token already recognized by the engine inherits the session's minimum 4-digit year as a date feat. If it also has a leading ordinal and the temporal word 'year', year_index = that epoch + ordinal - 1. Both feats use the existing date path. The nearest search titles, on deictic day-name systems, describe day-names and do not describe this addition. Catalog note: Q855899 Đại Việt sử ký toàn thư.

Combined mechanisms: a deictic line inherits the session's first absolute year + a leading 'year N' adds N-1 to that epoch.

Math: `if has_temporal and no 4-digit then inherit min_year; if lead_ord>0 then year_index=min_year+lead_ord-1`

Hook `crates/vestige-core/src/retrieval.rs` lines 3229-3238. Not regnal two-pass retrieval. Not the nengō rule, which also increments an epoch at a Capitalized token plus a short integer. This one does not detect era cuts.

### S5-039 Only the dated line covers the event

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese bibliographic classification applied to annals, Siku fourfold scheme, Qing. Siku filing admits a work to one class. For an event-sequence query, new_event and the event_cover term are granted only to shape 1, a turn whose first content token is a 4-digit year. Questions, lists, and prose do not receive event_cover. Their date_ord, when they have one, is still the identity used by the cap, so a prose turn of a new date is not rewarded as a new event and is not merged into another date. The 0.65 event_cover constant is not edited; the predicate that sets new_event changes. Catalog note: Q699477 Siku Quanshu.

Combined mechanisms: shape class filters event_cover + the event key remains the date ordinal.

Math: `new_event = is_event_sequence and shape=1 and date_ord not in selected; event_cover unchanged when new_event is false`

Hook `crates/vestige-core/src/retrieval.rs` lines 5521-5549. Not the pair-key recall rule, which emits a feat and does not touch event_cover. The 0.65 constant stays.

### S5-040 Stone face and woodblock number

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Chinese and Korean Buddhist canons, Fangshan stone sutras; Tripitaka Koreana, 13th century. The fetched Koreana file is addressed K.5.16. Fangshan stones add a face on the slab. The triple (canon class byte, volume, text) is stored raw. A parenthetical integer after the triple is the face. A query with three numbers matches the triple. A query with four numbers matches only when the face equals too. A three-number query does not require the face, so a stone face does not hide the text from a canon-level citation. Catalog note: K05n0016 高麗大藏經.

Combined mechanisms: Koreana (canon, volume, text) triple + a parenthetical face integer as a fourth key when the query has four numbers.

Math: `match3=(canon,vol,text); match4=match3 plus face; query width selects which key`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not the unclosed-number rule, which refuses to renumber holes. This one is the match key. Fangshan's catalog record is Q11075483, verified in the same wbgetentities batch; the URL cited is the file whose idno was actually read.

### S5-041 Chapter dates stay apart

ORDERING for BEAM event_ordering (pool_not_k 17). Vietnamese geographic compilation and Chinese event monographs, Đại Nam nhất thống chí, 19th century; Tongjian jishi benmo, 1174. Đại Nam nhất thống chí files places by chapter, and the dates inside a chapter are the events. event_ord is the embedded date, not the chapter title. Turns that share a document-frequency-1 token are one monograph, and inside it unequal date_ord values do not set topic_repeat. The chapter title, a first line with no digit, is not a date and does not glue the events. Same-date turns inside the chapter still use the existing jaccard test. Catalog note: Q8079953 Đại Nam nhất thống chí.

Combined mechanisms: a gazetteer chapter orders by its embedded dates + a shared df=1 token does not collapse different dates.

Math: `topic_repeat stays available only when date_ord equal or the shared-hapax test fails; event_ord=embedded date`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Distinct from the embedded-chapter-date seat, which only reorders slot 1 and does not change topic_repeat. Not incipit heads.

### S5-042 Article group and final finding

ORDERING for BEAM event_ordering (t5_not1 7). Chinese legal, Tang code and Zhangjiashan cases, Tang code with commentary; Zhangjiashan, 196-186 BCE. Tang lü shuyi is consulted by article number, and a Zhangjiashan case states the finding after the questions. Turns with the same leading integer are one group. For an event-sequence query, slot 1 is the last seated turn in the head's group that is not a question and has form_digit, if that turn contains every query content id the head contains. Other groups stay where they are. The search title on changing liability in Chinese law does not state this seat. Catalog note: 唐律疏議 ctp:wb785952 books=4.

Combined mechanisms: the leading article integer selects the group + the last non-question digit turn in that group takes slot 1.

Math: `group=lead_ord; slot1=last t in group with ¬is_question∧form_digit∧query_ids(head)⊆content(t)`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not Sokdaejeon replacement, which deletes the earlier article. Not the bracket-draft rule. Not Quaker minute. The group is the leading integer and the seat is the last non-question digit turn inside it.

### S5-043 One annal line per day, gaps left open

ORDERING for BEAM event_ordering (pool_not_k 17; k_not5 8). Korean and Japanese daily records, Joseon sillok; Azuma Kagami, late 13th century. The sillok keep one annal entry for a day, and Azuma Kagami leaves missing days missing. same_date_seen becomes a hard skip of a second event when date_ord values are equal, across sessions, not a 0.25 multiplier. The episode boost still requires the date ordinals to differ by exactly 1, so a gap is not filled by the neighbor. A second turn of the same day may enter k after the top 5. The 0.25 redundancy constant is not rewritten; the same-day case leaves the candidate loop. Catalog note: Q497030 Veritable Records of the Joseon Dynasty.

Combined mechanisms: one event slot per date ordinal + the episode boost does not cross a gap larger than one day.

Math: `if date_ord(a)=date_ord(b) then skip a as a second top-5 event; boost iff |Δdate_ord|=1`

Hook `crates/vestige-core/src/retrieval.rs` lines 5521-5549. Not the office-database extraction paper found under a different query about Seungjeongwon offices. This rule does not read office names. Not a retune of 0.25. Distinct from the parenthetical-month rule, which does not hard-skip a day.

### S5-044 Changed state or changed office

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese divinatory and historiographic, Zhouyi; Hanshu office table. A Zhouyi line that repeats the previous state is not a new event, and the Hanshu office table treats a single new office-holder as a new entry. topic_repeat is forced true when the content-id symmetric difference with the previous same-session turn is empty. It is forced false when the title-case symmetric difference has size 1, even if the content jaccard is high. Either condition is sufficient. The 0.45 test remains for every other pair. Catalog note: 周易 ctp:book-of-changes books=499.

Combined mechanisms: full content symmetric difference must be nonempty + a title-case difference of exactly 1 is enough by itself.

Math: `if |content Δ prev|=0 then topic_repeat=1; if |titlecase Δ prev|=1 then topic_repeat=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. The two differences are different sets. The content test alone, and the title-case test alone, are separate designs. This one fires if either condition holds.

### S5-045 Quoted annal over the biography

ORDERING for BEAM event_ordering (k_not5 8). Chinese historiographic, Shiji annals and biographies, Western Han, Sima Qian. The Shiji annals quote the dated notice, and the biography of the same day is the long unquoted narrative. event_ord prefers a date inside quotation marks. An unquoted turn with a greater content length and the same event_ord does not consume the session cap. The quoted turn does. A different date still gets a seat. This is not a shortest-cell test: a long quotation still holds the cap, and a short unquoted line does not take it away. Catalog note: KR2a0001.

Combined mechanisms: a date inside quotes is the event date + an unquoted longer turn of that date does not spend the cap.

Math: `event_ord=quoted date if any; cap(t)=0 if ¬quoted(t) and content_len(t)>content_len(quoted sibling of same event_ord)`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. Not the shortest-cell rule. Length matters only to refuse the cap to the unquoted sibling. Not quote-credit transfer to an earlier turn.

### S5-046 Block number is the step number

ORDERING for BEAM event_ordering (t5_not1 7; k_not5 8). Vietnamese woodblock and Korean protocol, Nguyễn woodblocks, Memory of the World; Joseon uigwe. Nguyễn woodblocks are cited by the number cut on the block, and a uigwe step keeps its number when a step is absent. lead_ord is the event order key for an event-sequence query. Slot order of the top 5 follows lead_ord ascending. A hole is not closed: the turn numbered 4 stays after the turn numbered 2 when 3 is absent, and it does not inherit 3's feats. Turns with lead_ord 0 sort last and do not take slot 1 while any positive lead_ord is seated. Catalog note: Q8032387 Woodblocks of the Nguyễn Dynasty.

Combined mechanisms: the printed woodblock ordinal is the event order + a missing ordinal is not renumbered, as in a uigwe step list.

Math: `order by lead_ord ascending; absent n is not filled; lead_ord 0 sorts after every positive lead`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not method of loci. Distinct from the unclosed-number recall rule, which only stores the raw integer for matching and does not reorder seats.

### S5-047 Class token and article integer

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Vietnamese institutional writing, Lịch triều hiến chương loại chí, 1821; Hồng Đức code, 1483. Lịch triều hiến chương loại chí files institutions by class, and the Hồng Đức code cites an article by number. The first title-case token is the class id. The leading integer is the article id. A pair feat is emitted only when both are present, and it matches only when the query has the same class token and the same integer. Content matching still works without the pair. The Hồng Đức bribery paper in the search titles discusses that code and does not describe this pair. The class is one orthographic token, not a list of ministry names. Catalog note: Q4896718 Lịch triều hiến chương loại chí.

Combined mechanisms: the first title-case token is the institutional class + the leading integer is the article, and the pair feat requires both.

Math: `pair=(first_titlecase, lead_ord) emitted only if both nonzero; match iff both equal`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not a domain word list of offices or ministries. Not the Yongle script-block pair, which uses a Unicode block rather than a title-case token.

### S5-048 Date-keyed session cap

ORDERING for BEAM event_ordering (k_not5 8). Chinese historiographic, jishi benmo, Southern Song, 1174. Tongjian jishi benmo keeps every dated entry of an event as its own item. While the top 5 is filling, the one-per-session cap keys on (session, date_ord) instead of session alone. Two dates in one session both enter the top 5. The cap constant stays 1. Turns with date_ord 0 still share the session key, so undated chatter does not multiply seats. Catalog note: KR2c0001.

Math: `cap_key=(session, date_ord) if date_ord≠0 else (session, 0); at most one top-5 seat per key`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. Not a retune of max_per_session_top5. The monograph-hapax combination also exempts a cap, and only inside a shared df=1 thread. This one keys every dated turn, with no hapax test.

### S5-049 Date-keyed topic repeat

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese historiographic, jishi benmo, Southern Song, 1174. The same monograph keeps successive dates instead of collapsing them. topic_repeat is computed only when the two turns have equal date_ord. Unequal date ordinals force topic_repeat false before the jaccard test is allowed to fire. The 0.45 constant is not changed. Same-date turns still use it. Catalog note: KR2c0001.

Math: `topic_repeat = date_ord equal AND the existing jaccard test; if date_ord differs then topic_repeat=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not Petrie seriation and not catch-lines. The hapax combination adds a df=1 guard this rule does not have. The cap rule is a different key and does not clear topic_repeat.

### S5-050 Outline line does not spend the cap

ORDERING for BEAM event_ordering (k_not5 8). Vietnamese historiographic, cương mục, Nguyễn, Tự Đức. The cương line is a short dated outline and is not itself a second event beside the detail. A turn with has_temporal, content length at most 12, and a leading digit does not increment the session cap. The following longer turn of that date can still take the top-5 seat. The outline remains eligible on its own merits. Catalog note: Q6770796 Khâm định Việt sử Thông giám cương mục.

Math: `if has_temporal and content_len<=12 and lead_digit then cap_increment=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. Not Abusir header copying. Distinct from the outline-and-witness combination, which also refuses quoted details. Distinct from the headline-takes-slot-1 combination.

### S5-051 Adjacent-day episode gate

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese diary, Da Tang chuangye qiju zhu, Tang, Wen Daya, 617-618. A court diary does not fill a blank day with the day before it. The existing episode backward boost is skipped unless the two sessions' date ordinals differ by exactly 1. The decay constants 0.52 and 0.65 are not written. Non-adjacent days stay out of the boost, so a neighbor stops occupying a pool seat that a real event day should have. Catalog note: KR2b0006.

Math: `apply the existing episode boost only if |date_ord(a)-date_ord(b)|=1`

Hook `crates/vestige-core/src/retrieval.rs` lines 5458-5471. Not Harris matrix ordering and not a logbook score bonus. The parenthetical-month combination also gates this boost and additionally freezes intercalary dates. This rule only gates the boost.

### S5-052 Office-title symmetric difference

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese historiographic, Hanshu office table, Eastern Han, Ban Gu. The Hanshu table of offices starts a new row when the holder changes and does not start one when the same offices repeat. Title-case token ids stand in for the office names already marked as mid-sentence capitals. Symmetric difference of size 0 forces topic_repeat. Symmetric difference of size 1 clears it. Every other size leaves the existing jaccard test in place. No office-name list is added. Catalog note: Q846905 Book of Han.

Math: `sym=|titlecase(a) Δ titlecase(b)|; sym=0 ⇒ topic_repeat=1; sym=1 ⇒ topic_repeat=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not Domesday pair retrieval and not Torrens. A Crossref paper builds an office database from Seungjeongwon ilgi with named-entity recognition; this rule does not use that diary and does not extract names. It counts title-case ids.

### S5-053 Chronicle date-head inheritance

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14; pool_not_k 35). Chinese annalistic, Zizhi tongjian, Northern Song, presented 1084. An annal date governs the lines under it until the next date line. Carry currently refuses to copy date feats. Each turn after a date-initial turn, and before the next one, receives that head's date feat ids as inheritable feats at the source feat's existing weight. A query year then matches a detail line that never repeated the year. Undated sessions are unchanged. This is the main Open-Domain recall move: it can score a gold that is indexed and currently unscored because its date lives on the previous line. Catalog note: KR2b0007.

Math: `for t after head h until the next date-initial turn: date_feats(t) ∪= date_feats(h)`

Hook `crates/vestige-core/src/retrieval.rs` lines 3647-3680. Not Abusir header inheritance, whose trigger is a markdown table or a colon-headed list. The trigger here is a date-initial line. Not ordinary carry, which skips dates.

### S5-054 Second era year feat

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Vietnamese historiographic, Đại Nam thực lục, Nguyễn veritable records. Đại Nam thực lục dates an entry in the reign year and can also cite another era's year in the same notice. Both 4-digit years are emitted as date feats through the existing inline-date path. This rule does not also choose an order key. A query that names the year the session date does not use can then score the turn. One year is what the engine already emits reliably; the second is the gap. Catalog note: Q911328 Đại Nam thực lục.

Math: `for each 4-digit year y on the turn, emit the existing date feat of y`

Hook `crates/vestige-core/src/retrieval.rs` lines 3229-3238. The dual-era combination also stores date_lo as the event order key. This rule only emits the feats.

### S5-055 Bundle-tag donation

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14; pool_not_k 35). Han slips, Juyan, Han, Juyan Lake slips. A Juyan bundle tag is a short slip whose words are a subset of the document it labels. A turn with at most four content ids, all of them contained in the next turn, donates those ids to the next turn as inheritable feats at the source weight. The tag's event_candidate flag is 0, so it does not take an event seat. A query token that occurs only on the tag then matches the document. Catalog note: Q1714726 wooden strips of the Han dynasty at Juyan.

Math: `if |content(t)|<=4 and content(t)⊆content(next) then feats(next) ∪= content(t) and candidate(t)=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not the bell join, which requires missing terminal punctuation. Not the combination that also copies a governing date line.

### S5-056 Reign-ordinal year index

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Vietnamese annalistic, Đại Việt sử ký toàn thư, Lê official history. The official history dates a year as an ordinal inside the reign. A turn that contains the existing temporal word 'year', has a leading ordinal, and has no 4-digit year stores year_index = the session's minimum 4-digit year + leading ordinal - 1. That integer is emitted as a date feat on the existing date path. A query that names the absolute year matches the ordinal line. The sum is computed once at ingest. Catalog note: Q855899 Đại Việt sử ký toàn thư.

Math: `if lead_ord>0 and has temporal 'year' and no 4-digit then year_index=min_year(session)+lead_ord-1`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. Not regnal two-pass retrieval. The nengō combination also increments an epoch at a capitalized token. This rule does not.

### S5-057 Raw article ordinal feat

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Japanese statutory, Goseibai shikimoku, Kamakura, 1232. Goseibai articles are small integers, not calendar years. The year-guard accepts only 1900..2099, so an article number never becomes a date feat. The leading integer is emitted as its own ordinal feat, stored raw, with no dense renumbering. A query digit equals that integer or it does not. This is how a question that cites a number can score a turn the year-guard dropped. Catalog note: Q63889 Goseibai Shikimoku.

Math: `emit feat(lead_ord) as printed; match iff query digit = lead_ord`

Hook `crates/vestige-core/src/retrieval.rs` lines 3229-3238. Not the year feat and not the canon triple. The unclosed-number combination also refuses to renumber and additionally blocks inheritance of the missing number's feats.

### S5-058 Dated turns take the initiative floor

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese annalistic, Zizhi tongjian, Northern Song, 1084. The chronicle floor is for the dated entry, not for every long user turn. The existing event-sequence floor is raised only for turns that have has_temporal. The floor values 0.42 and 0.38 are not edited. Undated turns keep the score they already had, so they stop crowding the selector ahead of a dated gold. Catalog note: 資治通鑑 ctp:wb548761 books=70.

Math: `raise the existing floors only where has_temporal=1; constants unchanged`

Hook `crates/vestige-core/src/retrieval.rs` lines 5199-5205. Not a retune of 0.42 or 0.38. The predicate that receives the floor changes. is_initiative itself is left as it is.

### S5-059 Same date is the written date

ORDERING for BEAM event_ordering (pool_not_k 17). Korean daily record, Ilseongnok, Joseon, 1760-1910. Ilseongnok is one record per written day. same_date_seen compares date_ord, not session_ts. Two sessions that share a timestamp and name different days are not the same date. Two turns that name the same day and sit in sessions with different timestamps are the same date. The 0.25 redundancy value is unchanged. Catalog note: Q626206 Ilseongnok.

Math: `same_date_seen iff date_ord(a)=date_ord(b) and date_ord≠0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5521-5549. Not a hard skip. The one-per-day combination skips the second seat. This one only changes which pairs receive the existing 0.25 factor.

### S5-060 Questions are not the event seat

ORDERING for BEAM event_ordering (t5_not1 7; pool_not_k 17). Han legal cases, Zhangjiashan, Zhangjiashan slips, 196-186 BCE. The Zhangjiashan case slips put the finding after the questions. For an event-sequence query, a turn with is_question is not selectable as an event seat. It stays out of the top 5. Non-question turns of that session compete as usual. This does not pick the last digit turn; the verdict combinations do that. Catalog note: Q5931471 Zhangjiashan Han bamboo texts.

Math: `if is_event_sequence and is_question then continue`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not Quaker-minute supersession and not the bracket-draft combination. The only test is the existing is_question flag.

### S5-061 Short marks leave the event set

ORDERING for BEAM event_ordering (pool_not_k 17). Korean reading tradition, gugyeol on Samguk sagi, Goryeo compilation of Samguk sagi, 1145. Gugyeol marks are short grammatical tokens between content words. Tokens of character length 2 or less are removed from the set the event jaccard test sees. They remain query-match feats. Two events stop looking alike because they share particles. The 0.45 constant stays. Catalog note: Q485552 Samguk sagi.

Math: `jaccard_set(t)={id in content(t) | chars(id)>=3}`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. The kaeriten combination also drops these tokens and additionally permutes bracket ordinals. This rule does not permute.

### S5-062 Bracket ordinal is its own cap key

ORDERING for BEAM event_ordering (k_not5 8). Japanese kanbun reading, Nihon Shoki, Nara, 720. Kaeriten reading treats a bracketed integer as the clause's place in the order. When a turn carries a bracket ordinal, the session cap keys on (session, bracket_ord). Two marked clauses in one session both enter the top 5. Turns without a bracket ordinal keep the plain session key. Surface order is not rewritten. Catalog note: Q813935 Nihon Shoki.

Math: `if bracket_ord>0 then cap_key=(session, bracket_ord)`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. Not method of loci and not Step 9 first_move_num. The reading-order combination also drops short tokens and uses the minimum bracket ordinal among query-hitting clauses.

### S5-063 First date is the event

ORDERING for BEAM event_ordering (t5_not1 7). Chinese manuscript, Dunhuang, Dunhuang cache, 4th-11th century. A Dunhuang roll can carry the text date and a later copy date. When there are two or more date ordinals, event_ord is the first and copy_ord is the last. Identity and the cap use event_ord only. Quote marks are not consulted. A single date is unchanged. Catalog note: Q276864 Dunhuang manuscripts.

Math: `if |dates|>=2 then event_ord=first, copy_ord=last`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. The quoted-date combination looks at quotation marks. This one uses position only. Not quote-credit transfer.

### S5-064 Shortest cell spends the cap

ORDERING for BEAM event_ordering (k_not5 8). Chinese historiographic, Shiji tables, Western Han; Qing Du Shiji shibiao. A table cell is the short entry at a date. For one date_ord, only the turn with the smallest content length that has a digit or a temporal token consumes the session cap. Longer turns of that date do not enter the top 5. Other dates are unaffected. Catalog note: KR2a0005.

Math: `cap holder(d)=argmin content_len among turns with date_ord=d and (form_digit or has_temporal)`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. Not primary_aggregate and not residual. The table-cell combination also hard-limits one cell per day across sessions. This one only picks the cap holder inside a date.

### S5-065 Unchanged line is not a new event

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese divinatory, Zhouyi, Zhouyi, received text. A line that does not change is the same situation. If the symmetric difference of content ids with the previous same-session turn is empty, topic_repeat is forced true. A nonempty difference leaves the existing jaccard test alone. Title-case ids are not consulted. Catalog note: 周易 ctp:book-of-changes books=499.

Math: `if |content(t) Δ content(prev)|=0 then topic_repeat=1`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. The changed-state combination also clears topic_repeat on a title-case difference of 1. This rule only forces the repeat when the content sets are equal.

### S5-066 Parenthetical date stays in the bucket

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese annalistic, Bamboo Annals, Bamboo Annals, traditionally Jin-tomb text. The Bamboo Annals record an intercalary month inside the year. A date token that sits inside parentheses is stored as paren_ord and does not advance date_ord. The bucket remains the previous unparenthesized date. The parenthetical value is still a date feat, so a query can match it, but it does not open a new event. Catalog note: KR2b0001.

Math: `if date token is inside parentheses then date_ord unchanged and paren_ord=that token`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. The gap combination also freezes this bucket and gates the episode boost. This rule does not touch the boost.

### S5-067 Tag turn is not a candidate

ORDERING for BEAM event_ordering (k_not5 8). Han slips, Juyan, Han, Juyan. The tag slip is the label, not the event. A turn with at most four content ids that are a subset of the next turn is skipped by the event selector. This rule does not copy feats. The donation rule does that. Skipping the tag frees the top-5 seat for the document. Catalog note: Q1714726 wooden strips of the Han dynasty at Juyan.

Math: `if |content|<=4 and content⊆next then candidate=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not the donation rule, which also unions feats onto the next turn. This one only removes the tag from the selector.

### S5-068 Later clock is the event time

ORDERING for BEAM event_ordering (t5_not1 7). Han postal slips, Juyan, Han, Juyan. A pass records when it was issued and when it was received. If two clock-shaped tokens are present, event_clock is the later ordinal. Slot order for an event-sequence query uses event_clock. The earlier clock remains a feat and is not the key. Witness counts are not used. Catalog note: Q1714726 wooden strips of the Han dynasty at Juyan.

Math: `if |clocks|=2 then event_clock=max(clocks)`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. The witness combination also orders by line-final title-case count. This rule only picks the later clock.

### S5-069 See-pointer donates to the article

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Japanese ritual code, Engishiki, Heian, 927. A short Engishiki cross-reference points at an article and is not the article. A turn whose content ids are the closed-class word 'see' plus one integer is removed from candidacy, and those other feat ids are unioned onto the same-session turn whose lead_ord equals the integer. The target keeps its own content. No synonym of 'see' is added. Catalog note: Q1342448 Engishiki.

Math: `if content={see,n} then feats(target[lead=n]) ∪= feats(pointer) and candidate(pointer)=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. The pointer-replace combination overwrites the target content. This one unions. Not Khipu.

### S5-070 Same article number replaces that article

ORDERING for BEAM event_ordering (pool_not_k 17). Korean statutory supplement, Sokdaejeon, Joseon supplement to Gyeongguk daejeon. Sokdaejeon amends an article by its number and leaves the other articles standing. A later turn with the same nonzero lead_ord replaces the earlier candidate of that integer and does not remove any other integer. The earlier turn's event_candidate flag becomes 0. Catalog note: Q16177106 Sokdaejeon.

Math: `if lead(a)=lead(b)≠0 and b is later then candidate(a)=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not Torrens whole-document replacement. The Goseibai combination pairs this with a second code's identity rule. This one is the replacement alone.

### S5-071 Dated line beats an undated line

ORDERING for BEAM event_ordering (t5_not1 7). Korean historiographic, Samguk sagi, Goryeo, 1145. Samguk sagi is the dated chronicle. For an event-sequence query, if the head has no date ordinal and another seated top-5 turn has one and contains every query content id the head contains, that dated turn takes slot 1. A Crossref title discusses Samguk sagi together with Samguk yusa; this rule does not combine the two books. It is one test: a date ordinal is present or it is not. Samguk yusa was verified separately and is not the mechanism. Catalog note: Q485552 Samguk sagi.

Math: `if date_ord(head)=0 and exists t in top5 with date_ord>0 and query_ids(head)⊆content(t) then slot1=t`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Not the cardinality combination, which also compares how many dates a turn has. This one only separates zero from nonzero. The sagi/yusa paper is not a claim that this seat was previously published.

### S5-072 Embedded date orders the seat

ORDERING for BEAM event_ordering (t5_not1 7). Japanese topical history, Ruijū kokushi, Heian, 892. Ruijū kokushi files by topic and dates the event inside the entry. For an event-sequence query, slot order of the top 5 follows the first date ordinal that occurs after a digit-free first line. The header string is not the key. Turn index is not the key. Catalog note: Q1038639 Ruijū Kokushi.

Math: `event_ord=first date after a digit-free header; order top5 by event_ord`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. The chapter-date combination also clears topic_repeat inside a df=1 thread. This rule only reorders the seated list.

### S5-073 Decision-layer feat

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Vietnamese bureaucratic, Nguyễn châu bản, Nguyễn court documents. The vermilion decision is the marked line. Tokens on a line whose first character is '>' or '[' are emitted under layer 1. Existing inference queries match layer 1. Question-shaped queries match layer 0. No date rule is attached. The mark is the character, not an ALLCAPS list. Catalog note: Q16481267 Châu bản triều Nguyễn.

Math: `layer=1 iff line starts with '>' or '['; inference matches layer 1`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. The vermilion combination also keeps the event date on the unmarked line. This rule only emits the layer id. Not rubric ALLCAPS.

### S5-074 Bracketed lines are not candidates

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese bureaucratic, Ming shilu, Ming veritable records. The veritable record keeps the decision and not the bracketed draft. A turn whose first character is '[' is skipped by the event selector. It is not deleted from the index. The last-digit verdict test is not applied. Catalog note: Q28772 Ming Shilu.

Math: `if first char='[' then candidate=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. The draft-and-verdict combination also selects the last non-question digit turn. This rule only drops the bracketed line.

### S5-075 Shared symbol is one slot

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese manuscript marks, Dunhuang, Dunhuang, 4th-11th century. Two notes that carry the same non-alphanumeric mark belong to one item. A token with no letter and no digit is a mark. If a selected turn shares one mark id with the candidate, the candidate does not take a second event slot. Endorsement counts are not consulted. Catalog note: Q276864 Dunhuang manuscripts.

Math: `if mark(a)∩mark(b)≠∅ then second slot=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. Not the bell join. The cluster combination also orders by endorsement count. This rule only merges the slot.

### S5-076 Dotted triple address

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Korean woodblock canon, Tripitaka Koreana; TEI file K05n0016. The fetched file is addressed as canon K, volume 5, text 16. Three integers in a dotted or slash-separated token are emitted as one address id. A query with the same three integers matches that id and no other triple. Face numbers and holes are not part of this rule. Catalog note: K05n0016 高麗大藏經.

Math: `addr=(d1,d2,d3); match iff query triple equals addr`

Hook `crates/vestige-core/src/retrieval.rs` lines 3121-3145. The stone-face combination adds a fourth integer when the query has four numbers. The unclosed-number combination refuses to renumber. This rule is the exact triple only.

### S5-077 One slot per written day

ORDERING for BEAM event_ordering (k_not5 8). Korean daily record, Seungjeongwon ilgi, Joseon royal secretariat diary, 1623-1910. The secretariat diary has one entry for a day. A second top-5 event with the same date_ord is skipped, including across sessions. The skip replaces the 0.25 same-date factor for this case; the constant is not rewritten because the candidate leaves the loop. A second turn of that day may still fill k after the top 5. Office titles are not read. A paper on building an office database from this diary is a different task and is not this gate. Catalog note: Q483160 Seungjeongwon ilgi.

Math: `if date_ord already seated in the top 5 then skip`

Hook `crates/vestige-core/src/retrieval.rs` lines 5521-5549. Not the office-NER paper on this diary. Not the sillok combination, which also gates the episode boost. The 0.25 factor is not edited.

### S5-078 Higher date cardinality takes slot 1

ORDERING for BEAM event_ordering (t5_not1 7). Chinese long draft, Jianyan yilai xinian yaolu, Southern Song, Li Xinchuan. The long draft keeps more dated notices than the digest. For an event-sequence query, the seated top-5 turn with the strictly greatest number of distinct date ordinals takes slot 1 when it contains every query content id the head contains. Undated turns are not removed. Catalog note: 建炎以來繫年要錄 ctp:wb907864 books=1.

Math: `slot1=argmax |date_set| over top5, strict, and query_ids(head)⊆content`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. Inverse of primary_aggregate, and only for is_event_sequence. The finer-chronicle combination also prefers any dated turn over an undated one on a tie. This rule has no tie break.

### S5-079 Deictic line inherits the session year

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Japanese diary chronicle, Azuma Kagami, late 13th century. A diary line that says only 'that day' belongs to the day's year. A turn with has_temporal and no 4-digit year inherits the session's minimum 4-digit year as a date feat on the existing date path. No ordinal is added. The deictic day-name papers in a related search describe day-names and do not describe this inheritance. Catalog note: Q1042324 Azuma Kagami.

Math: `if has_temporal and no 4-digit year then emit min_year(session)`

Hook `crates/vestige-core/src/retrieval.rs` lines 3229-3238. The epoch combination also adds lead_ord-1. This rule copies the year only.

### S5-080 Conditional tail leaves the event set

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese statutory, Tang code, Tang lü shuyi. The operative part of a statute is the condition, and the trailing consequence repeats stock language. Tokens after the last 'if' — already a closed-class starter — are removed from the event jaccard set. They stay in the query-match feats. Negated dates are not handled here. Catalog note: 唐律疏議 ctp:wb785952 books=4.

Math: `jaccard_set = content ids at or before the last 'if'`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. The prohibition combination also drops negated date lines. This rule only cuts the jaccard set.

### S5-081 Negated date line is not an event

ORDERING for BEAM event_ordering (pool_not_k 17). Qin daybooks, Shuihudi, Shuihudi Qin slips. A daybook prohibition is a dated line that says the day is not to be used. A date-initial turn whose raw tokens include 'not', 'no', or 'never' — already in the engine's closed class — is skipped by the event selector. No prohibition lexicon is added. Catalog note: Q689661 Shuihudi Qin bamboo texts.

Math: `if date_initial and a closed-class negator is in the raw tokens then candidate=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. The prohibition combination also cuts tokens after 'if'. This rule only skips the negated date line.

### S5-082 Indent depth orders the seat

ORDERING for BEAM event_ordering (t5_not1 7). Chinese genealogical tables, Shiji, Western Han. The genealogical tables step down by generation, written as indent. Depth is leading spaces divided by 2. For an event-sequence query the top 5 is ordered by depth ascending. Leading integers are not part of the key. Catalog note: KR2a0001.

Math: `depth=leading_spaces//2; order top5 by depth`

Hook `crates/vestige-core/src/retrieval.rs` lines 6261-6272. The generation combination uses the pair (depth, lead_ord). This rule uses depth only.

### S5-083 Second digit is its own feat

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Korean statutory, Gyeongguk daejeon, Joseon, 1485. A code line carries more than one number, and the second number is not the year. The second digit token is emitted as its own feat id. It is not passed through the 1900..2099 year-guard. A query digit matches it exactly. No canon triple is required. Catalog note: Q489676 Gyeongguk daejeon.

Math: `emit feat(second digit token); match iff query digit equals it`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. The position-and-address combination requires this digit and a canon triple together. This rule emits the digit alone.

### S5-084 Numbered series drops the unnumbered copy

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese administrative, Tang liudian, Tang, 738. The liudian is a numbered register. When leading integers in a session are strictly increasing, a later turn with lead_ord 0 whose content ids are a subset of a numbered turn does not take an event slot. The hapax test is not used. Different numbers all stay. Catalog note: 唐六典 ctp:wb502694 books=3.

Math: `if leads are strictly increasing and lead(b)=0 and content(b)⊆content(a) then candidate(b)=0`

Hook `crates/vestige-core/src/retrieval.rs` lines 5647-5710. The register-monograph combination also requires a shared df=1 token. This rule uses subset containment. Not the Liye possessive binding.

### S5-085 Quoted date is the event date

ORDERING for BEAM event_ordering (t5_not1 7). Chinese historiographic, Shiji, Western Han. The annal quotes the dated notice. If any date ordinal sits inside quotation marks, event_ord is that date. Dates outside quotes do not become the key when a quoted date exists. The cap is unchanged. Catalog note: KR2a0001.

Math: `if a date is inside quotes then event_ord=that date`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. The biography combination also refuses the cap to a longer unquoted sibling. Not quote-credit transfer to an earlier turn.

### S5-086 Short final line is not a key

ORDERING for BEAM event_ordering (t5_not1 7). Chinese historiographic, Dongguan Hanji, Eastern Han compilation. A compilation note at the end of an entry is not the event. If the last line has fewer than 8 tokens and contains a date, that date is not event_ord and does not spend a cap slot. Earlier dates on the turn are unchanged. No second key is opened. Catalog note: 東觀漢記 ctp:dong-guan-han-ji books=3.

Math: `if last line has token_len<8 and contains a date then that date is not event_ord`

Hook `crates/vestige-core/src/retrieval.rs` lines 3334-3384. The internal-split combination also opens a second key at an internal date. This rule only ignores the short tail.

### S5-087 Class token feat

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Vietnamese institutional compendium, Lịch triều hiến chương loại chí, 1821. The compendium files each institution under a class heading. The first title-case token is emitted as a class feat, the same way a header token is already emitted, and a query with that token matches it. No article integer is required and no ministry list is added. Catalog note: Q4896718 Lịch triều hiến chương loại chí.

Math: `emit feat(first title-case token)`

Hook `crates/vestige-core/src/retrieval.rs` lines 3181-3191. The class-and-article combination requires the leading integer as well. Not a domain list of office names.

### S5-088 Anaphoric line inherits the previous date

RECALL/INDEX for LoCoMo Open-Domain (no_pool 14). Chinese annalistic, Zizhi tongjian, Northern Song, 1084. A line that is already marked anaphoric continues the previous entry's date. Only when anaphoric is true, and the turn has no date of its own, the previous turn's date feat ids are copied at the source weight. Following non-anaphoric lines do not inherit. This is narrower than giving every line the governing date. Catalog note: 資治通鑑 ctp:wb548761 books=70.

Math: `if anaphoric and date_ord=0 then date_feats ∪= date_feats(previous)`

Hook `crates/vestige-core/src/retrieval.rs` lines 3647-3680. The chronicle date-head rule copies the date onto every following line until the next date head. This one copies only onto an already anaphoric turn. Carry's date skip is the gap either rule fills.

### S5-089 A digit query prefers numbered lines

ORDERING for BEAM event_ordering (k_not5 8). Korean bureaucratic, uigwe, Joseon royal protocols. A protocol question that contains a digit is asking for the numbered steps. When the query has a digit, a turn with list_item_count=0 does not enter the top 5. It may still fill k afterwards. The cap key stays the session. Leading integers are not rewritten. Catalog note: Q1073141 Uigwe.

Math: `if query has a digit and list_item_count=0 then not eligible for the top 5`

Hook `crates/vestige-core/src/retrieval.rs` lines 5619-5680. The ordinal-list combination also keys the cap on lead_ord. This rule only filters eligibility. Not method of loci.

### S5-090 Outline tokens are the event set

ORDERING for BEAM event_ordering (pool_not_k 17). Chinese historiographic, jishi benmo, Southern Song, 1174. The monograph's identity is the outline, not every sentence under it. For an event-sequence query, initiative_content receives content ids only from turns that are date-initial and have content length at most 12. Longer initiative turns do not enter that set, so jaccard no longer treats two long narratives as the same event. The jaccard constant stays. The cap is not changed here. Catalog note: KR2c0001.

Math: `initiative_content(session) = union of content ids of turns with date_initial and content_len<=12`

Hook `crates/vestige-core/src/retrieval.rs` lines 5221-5233. Not the date-keyed topic-repeat rule, which gates the comparison. This one changes the set that comparison sees. Not catch-lines.

