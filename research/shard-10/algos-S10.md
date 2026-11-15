# Shard 10 — ninety registrar algorithms

Research and design only. Base SHA `e2ecd1da037974c1835f563ad8b046f4a4ca3629`. No engine code is changed. Ranking sketches read new index fields. They do not read `turn.body`, do not touch `rsb_seat_behind`, and do not retune an existing coefficient. The ninety designs are unchanged. What changed with Sam's 90% floor is which gap each one is allowed to claim, and the ceiling of that claim. Both are in `funnel-S10.md` and in the `floor_ceiling` column of `algos-S10.csv`.

Fifty of the ninety combine two mechanisms. A sentence that says `no prior combination found in searches` means the Crossref or OpenAlex top title logged in `api-log-S10.csv` does not join those two mechanisms. It is not a claim that no such paper exists. Neighbors that were found are named in the rows they touch: Robertson, Lu and Macfarlane on XML element inheritance, and the Cretan Collection concordance of accession and catalog numbers (DOI 10.2307/jj.8692990.24).

## Top 10 build order

The floor is 90% on every metric, not the six slices. A seat cannot get there. LoCoMo Turn@1 reorder-caps at Turn@5 = 1127, which is 256 short of 1383. BEAM All@5 reorder-caps at All@10 = 138, which is 182 short of 320. BEAM Session@1 reorder-caps at Session@5 = 249, which is 71 short of 320, and Session@10 itself is 275. The session list is frozen at `retrieval.rs:5480`, before any seat at 6265. Caps below are structural upper bounds from `funnel-S10.md`. They are not expected gains. The attachment does not count how many golds carry the shape.

1. **S10-001 Atomic two-part accession** — LoCoMo Turn@1 and Session@1. Intern the dotted key before the splitter breaks it, and the feat enters `session_bags`. In-pool Turn@1 cap 1366 (short 17 of 1383). Cap 1402 if the key is what the 36 `no_pool` golds in Single-Hop, Multi-Hop, and Open-Domain lack. Single-Hop is 263 of the attachment gap.

2. **S10-026 Shared stem fills the pool under a quantifier** — BEAM All@5 and LME Turn All@5. all, every, or whole seats every turn of one accession stem. All@5 cap 258 if summarization, event_ordering, multi_session_reasoning, and preference_following are stem sets, still 62 short of 320. LME Turn All@5 cap 435, which clears 423. Session stays put until the same stem is also a feat.

3. **S10-004 Accession prefix is not the object year** — LoCoMo Temporal, LME temporal-reasoning, BEAM event_ordering and temporal_reasoning. The year that opens an accession is not the object year, and the feat is in the session bag. This row does not touch Single-Hop, so it does not, alone, clear LoCoMo Turn@1.

4. **S10-012 Closing suffix lends its tokens** — preference session, on both benches, and LME single-session-preference. Copy the closing leaf onto earlier leaves of the same stem at weight 1/2. All@5 cap 182, from preference_following and summarization only.

5. **S10-015 Quoted short name is an alias of the long name** — the same Turn@1 cap as S10-001 (1366 / 1402), plus LME preference, because the alias is a feat in the session bag. Recovers `no_pool` only when the gold turn already writes both names.

6. **S10-038 Year off line 0 is a mention** — the only row that names BEAM knowledge_update, and only when the updated value is a year on line 0. Combined with the other ingest rows this is what pushes the BEAM Session@1 structural cap to 344. contradiction_resolution is still absent from that sum.

7. **S10-017 A short line with no question mark is the command** — instruction_following. Pool cap 35, need 36, so the category clears only if at least one of the five `no_pool` golds has that shape. The hook is the pool, not the session bag.

8. **S10-010 Shorter fragment wins a covered-group tie** — the Single-Hop residue already inside Turn@5 (192 of the 347 Single-Hop misses). Seat only. Turn@1 cap 1005. Session and All@5 do not move.

9. **S10-088 Inheritance stops at an exact key** — the copy from S10-012 contributes 0 when the query already names the leaf. Keeps the preference inherit from stealing slot 1. Same session-bag path as S10-012, without the summarization All@5 term (cap 146).

10. **S10-050 Exact suffix first, then the quantifier fills the rest** — the fill from S10-026, with the named suffix held at rank 0 so the set does not bury the exact gold. Same All@5 cap, 258.

## S10-001 Atomic two-part accession

Cleveland 1971.16 (The Five Hundred Arhats, API id 145953) is one dotted key. raw_tokens at retrieval.rs:657-684 splits on the dot, so the turn is indexed as the separate strings 1971 and 16. extract_exact_technical_handles at 2309-2320 stores a version only for a v prefix or three or more numeric parts, so a two-part key is not a handle either. At ingest, scan the original text and intern acc:1971.16 as one feat with counts_len false. Rank by the integer tuple (exact accession match, covered query groups, negative token length). A prose hit on 1971 cannot outrank an exact key. This is how an Open-Domain or preference gold that the query names by its id gets into the pool at all: the bag destroyed the only token the query and the gold shared. The feat is on the turn, so `session_bags` copies it at retrieval.rs:3630 and it can move Session@1. The same key is the Single-Hop test. The structural LoCoMo Turn@1 cap of this row alone is 1366 inside the broad pool and 1402 if the key is on the 36 `no_pool` golds in Single-Hop, Multi-Hop, and Open-Domain.

Math: Let Acc(t) be the set of substrings matching ^[0-9]{2,4}\.[0-9]{1,4}$. exact(q,t)=1 iff Acc(q) intersects Acc(t), else 0. Order turns by (exact, groups(t), -len(t)) in lexicographic integer order. No new float is added to BM25.

Tag: RECALL/INDEX. Slices: OD+MH+PF. Record: https://openaccess-api.clevelandart.org/api/artworks/145953

## S10-002 Part letter is its own ordinal

Cleveland 1971.127.b is folio 141 of a Kalpa-sutra, and 1971.126.a is a different stem with a different letter. The trailing letter is not part of the stem and not decoration the splitter may drop. Store stem=1971.127 and part_ord of the letter (a=1, b=2, ...). A query that carries the letter matches only that part. A query that carries only the stem matches every part, but an exact letter still ranks first by the integer (letter_equal, stem_equal). Two consecutive stems never satisfy each other. Event-ordering uses part_ord as the order key when both turns share a stem, which beats session timestamp.

Math: part_ord(letter)=1+ (letter-'a'). key(q,t)=(1 if part(q)=part(t) else 0, 1 if stem(q)=stem(t) else 0, -part_ord(t) if the query has the closed-class token before). Compare tuples as integers.

Tag: RECALL/INDEX. Slices: MH+EO. Record: https://www.clevelandart.org/art/1971.127.b

## S10-003 Abbreviated sibling expansion

Met 76113 accession is the string 2010.402.1, .2 (Screens within Screens). The second part is written as a bare dot and a digit, not as a second full key. raw_tokens keeps none of that structure. At ingest, when a full dotted key is followed by a comma and a dot-digit, emit the sibling key with the same stem and that digit. Both keys are feats of the one turn, counts_len false. A query naming .2 hits this turn even though the full second key was never written out. A summary that asks for both parts sees two keys on one turn instead of one broken integer.

Math: From a match of KEY ',' '.' DIGITS, sibling = stem(KEY)+'.'+DIGITS. Cover(q,t)=|Acc(q) intersect (Acc(t) union Sib(t))|. Order by (cover, -len).

Tag: RECALL/INDEX. Slices: MH+SUM. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/76113

## S10-004 Accession prefix is not the object year

The same Arhats scroll is accession 1971.16, accession_date 1971-02-23, and creation 1591–1626. The four digits that open the accession are the year it was numbered, not the year it was made. A query whose only year is 1591 must match the creation span and must not be stolen by a turn whose only 1971 is inside an accession key. Store object_year from a 4-digit span that is not inside an accession token, and acq_year from the left side of an accession token. Event order sorts on object_year. Open-domain 'when' questions stop landing on the cataloguing year. Both year classes are feats, so they enter the session bag. This is the lever for LoCoMo Temporal, LME temporal-reasoning, and BEAM temporal_reasoning. It does not name Single-Hop, so it does not clear LoCoMo Turn@1 on its own.

Math: object_year(t)= the 4-digit span outside Acc(t). acq_year(t)= the 2-4 digit left part of an accession. A query year y matches t iff y is inside the object span. acq_year never matches y. Order EO turns by object_year ascending as an integer.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://www.clevelandart.org/art/1971.16

## S10-005 Sibling year, earlier may win

Met 44709 and 44710 are both titled Textile for a handscroll, accessions 1993.51.1 and 1993.51.2, dated March 18, 1714 and October 9, 1862. They share a stem and a title. Session time or a later catalog number must not decide. If the query contains a 4-digit year, the sibling whose object year equals that year wins, including when it is the earlier one. If the query has no year, neither sibling outranks the other by date. Both stay in the pool because the stem matched.

Math: Same stem and same title-token set => siblings. If Y(q) nonempty, score = 1 iff object_year(t) in Y(q), else 0. Order by (score, stem_match). A greater object_year is not a tie-break.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/44709

## S10-006 The other part is excluded, not capped

Met 204540 is a pair of cups, accession 65.152.2, .3. A query that names one part is about that part. The other part's tokens must leave the length and the overlap, not merely be down-weighted. petuchah keeps BM25F and caps a section. This rule deletes the other part from the counted token set before the groups are computed, so a long second part cannot bury a short exact part. Multi-hop gold that quotes one half of a pair stops losing to the half the query did not name.

Math: parts(t)={p1,p2} from the abbreviated-sibling parse. If the query names pi, counted(t)=tokens(pi) and len(t)=|counted(t)|. groups are computed on counted(t) only. Order by (groups, -len) on that reduced set.

Tag: RECALL/INDEX. Slices: MH+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/204540

## S10-007 Three plaques, numeric order

Met 471950, 471951 and 471952 are three plaques from one reliquary, accessions 69.238.1, 69.238.2 and 69.238.3, same title. The set is ordered by the last integer, not by session time and not by which plaque happens to share more prose with the query. A query with the closed-class tokens all, every or whole pulls all three into the pool, bypassing the per-session cap at retrieval.rs:5675-5679, and seats them in suffix order. A query that names one suffix seats that suffix and still keeps the other two behind it so a summary can see the set. all5 is 0 on both event_ordering and summarization; a set that never enters the pool cannot become all5.

Math: stem 69.238. Members ordered by suffix as i32. Under quantifier q in {all, every, whole}, pool must include every member with the stem. Slot order is suffix order. Under a query suffix s, order key is (suffix==s, -|suffix-s|).

Tag: RECALL/INDEX. Slices: EO+SUM. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/471950

## S10-008 Same title, two accessions stay distinct

Met 463192 (09.146) and 463958 (16.79) carry the same title, Altar Predella and Socle of Archbishop Don Dalmau de Mur y Cervelló, and the same date span ca. 1456–1458. Title equality is not identity. A query that contains 09.146 must return 463192 and must not be satisfied by 16.79. A query with the title and no accession keeps both addressable and does not collapse them as duplicates. Dittography collapses repeated text. These are two catalog numbers for one title, and the number is the identity.

Math: title_eq AND acc_a != acc_b => not the same turn. exact_acc dominates title overlap: order by (exact_acc, title_eq, -len). Never drop the lower one as a duplicate.

Tag: RECALL/INDEX. Slices: OD+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/463192

## S10-009 Title integers versus measurement decimals

Met 553673, Book of the Dead Papyrus with Chapters 100 and 129, accession 24.2.18. The title carries the integers 100 and 129. A measurement on a record like the Arhats scroll is a decimal (39.5). A how-many or which-number question is answered by a 2- or 3-digit integer on the first line that is not a decimal and not a 4-digit year. A decimal never answers it. This is the opposite of the existing WH-slot voter, which prefers any form_digit: a digit inside 39.5 must not win just because it is a digit.

Math: title_int(t)={n on line 0 | 10<=n<=999 and n is not part of a decimal and n is not inside Acc(t)}. measure(t)={tokens matching [0-9]+'.'[0-9] with one digit after the dot}. A how-many query matches title_int only. Order by (|title_int intersect q|, -|measure|).

Tag: RECALL/INDEX. Slices: OD+SUM. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/553673

## S10-010 Shorter exact fragment wins the tie

Met 247733 is a vase fragment, accession 07.232.107. A long roll that contains the same query groups plus a few hundred other tokens is the wrong unit for a question about the fragment. BM25's length norm is smooth and a high term frequency on the long roll still wins. The ordinal rule is stricter and only fires on a tie of covered groups: the shorter turn wins. Summarization, whose gold is the short unit, stops losing to the essay that quotes it. The hook is a seat at 6265. It can finish the 192 Single-Hop misses already inside Turn@5, and the bench Turn@1 cap of this row is 1005. Session and All@5 stay where they are.

Math: If groups(a)=groups(b), order by len ascending (integer). If groups differ, the greater group count wins and length is not consulted. No coefficient is applied to t_scene.

Tag: RECALL/INDEX. Slices: OD+SUM. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/247733

## S10-011 Parenthetical integer is not a year

Met 432480, Bookplate of George Washington, accession 20.90.37(27), dated 1772. The parenthetical 27 is a plate index sitting inside the accession. It must not become a year, a day, or a second accession. Strip a parenthetical run of 1-3 digits out of year parsing and store it as plate_ord. Event order uses 1772. A query digit 27 matches plate_ord and does not match a turn whose only 27 is a year-like 4-digit or a day next to a month name.

Math: plate_ord = the 1-3 digit run inside trailing parentheses. It is removed from the year set and from Acc. Query digit d of width <=3 matches plate_ord before it matches any year. A 4-digit query never matches plate_ord.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/432480

## S10-012 Closing suffix lends its tokens

Cleveland 1975.22.11 is the colophon leaf of Album of Miscellaneous Subjects; 1975.22.2 and 1975.22.3 are earlier leaves of the same stem. The closing leaf is the one with the greatest numeric suffix, found by integer order, not by a word list. Its content tokens are copied onto the lower suffixes of that stem as feats with weight ratio 1/2 and counts_len false. A preference or open-domain query that uses a rare token from the closing leaf can now hit the leaf the query actually names. Robertson, Lu and Macfarlane already inherit ancestor titles into XML elements. This inheritance is keyed by accession stem and runs from the max suffix downward, at a fixed ratio, with no learned field weight. The copied feats enter `session_bags`, which is the preference Session@1 path on BEAM and on LME single-session-preference. All@5 moves when the gold set is those leaves: the cap is 182, from preference_following and summarization.

Math: For stem S let c = argmax suffix. For each other part p, feats(p) gain {id in content(c)} at num=1, den=2, counts_len false. Rank compares cross-multiplied integers: score_num * other_den. No float knob.

Tag: RECALL/INDEX. Slices: PF+OD+SUM. Record: https://www.clevelandart.org/art/1975.22.11

## S10-013 Range width answers how many

Met 74462, Album of Copies of Chinese Paintings, accession 1974.224.1–.31. The en-dash between a full key and a bare dot-integer is a range. Its width is 31-1+1 = 31. A how-many question is answered by that width. A digit sitting in the prose of the same turn is not the answer. The WH-slot voter prefers any form_digit and would be free to pick the prose digit. This rule refuses that digit whenever a range width is present on the turn.

Math: width = hi - lo + 1, both integers parsed from KEY '–' '.' INT. how_many(t)=width if a range exists, else no answer from this rule. A bare integer in the body is not how_many.

Tag: RECALL/INDEX. Slices: SUM+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/74462

## S10-014 Year after 'after' is the model year

Met 204834 is dated '19th century, after a model of ca. 1877', accession 68.37. The closed-class token after is the cut. A 4-digit year later on that same line is model_year and is stored separately from object_year. A query year equal to the model year must not seat this turn as the object that was made in that year. Event order uses object_year only. No abbreviation list is required: the cut is the word after, which is closed class.

Math: If token i is 'after' and a 4-digit year occurs at j>i on the same line, that year is model_year. object_year is every other 4-digit span. Query year matches object_year only. model_year is a negative feature for an event match: match_bit = 0 when the only hit is model_year.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/204834

## S10-015 Quoted short name is an alias of the long name

Rijks id 200107928 is identifier SK-C-5. Its names include 'De Nachtwacht' and the long militia-company title, and two of the names say the company is known as a quoted short form ('Nachtwacht', 'The nightwatch'). Tokens inside single quotes on a name line become alias feats of that same turn, counts_len false. A query that uses the short form hits the turn whose long form is what the bag would have indexed. No synonym table: the alias has to be written on the turn, in quotes. Open-domain no_pool is exactly the case where the query's name and the gold's indexed name differ, and this recovers the ones where the gold turn itself records both. The alias is a feat in the session bag. The Turn@1 structural cap matches S10-001, 1366 inside the pool and 1402 with those `no_pool` golds, for turns that already write both names.

Math: alias(t)={tokens inside single quotes}. Hit if query tokens intersect (content(t) union alias(t)). An alias hit ranks as exact on those tokens. Quotes do not themselves count as tokens.

Tag: RECALL/INDEX. Slices: OD+MH. Record: https://id.rijksmuseum.nl/200107928

## S10-016 Tokens after 'in' are the container

Rijks 200380645 is titled Brief in envelop aan mevrouw Ament-van Beusechem. The closed-class token in splits the line. Tokens after in and before the next closed-class preposition are the container. Tokens before in are the contents. A multi-hop query about the container must be scored on the container span only, and a query about the contents must not receive credit for the container tokens. The two spans are separate feat lists. This stops a letter that merely travels inside something else from answering a question about that something else.

Math: Split the line at 'in'. cover_contents uses tokens before the cut, cover_container uses tokens after it. The query is scored against the side whose overlap is larger, and the other side contributes 0. Order by that one-sided cover.

Tag: RECALL/INDEX. Slices: MH. Record: https://id.rijksmuseum.nl/200380645

## S10-017 A short line with no question mark is the command

Rijks 200436215 is a commando-brief, one short order to Hendrick Liefringh. Rijks 200378217 is a long comma inventory of a folder's mixed contents. Instruction-following gold is the short order. The query shape that selects it is closed-class and orthographic: the query does not end in '?', and its first token is not a wh-word, pronoun, auxiliary or modal. Among turns that cover the same groups, the turn with the smaller token length wins, and a turn whose first line is at most 12 tokens is pulled into the pool ahead of a longer turn with equal groups. The long inventory stays in the pool behind it. Broad-pool ceiling on this slice is 35 against a need of 36, so the short golds sitting in no_pool have to become visible; length is the feature that distinguishes them once their tokens match. The hook is the pool at 5628. Session@1 does not move, because the session list is already frozen at 5480.

Math: command_query = last char != '?' AND first token not in the closed class {who,what,when,where,why,how,which,i,you,we,is,are,was,were,do,did,can,could,should,would}. Among equal groups, order by token_len ascending. A turn with first_line_len<=12 gets a pool-eligibility bit.

Tag: RECALL/INDEX. Slices: IF. Record: https://id.rijksmuseum.nl/200436215

## S10-018 Day, month and year on the title line are one event

Rijks 200424265 begins 'Brief van 16 juli 1813' and then continues with a long clause. The event date is the day, the month name and the year on that first line, as one triple (1813, 7, 16). Month names are closed class. A later 4-digit year in the rest of the line does not become a second event. A different letter that shares only the year, or only the month, does not match a query that states all three. Event order sorts on the triple. Multi-hop stops joining two letters because they both say 1813.

Math: event(t)=(Y,M,D) parsed from line 0 only, M from the closed month-name list. Match requires all components the query actually contains. Sort key is Y*10000+M*100+D as i32.

Tag: RECALL/INDEX. Slices: EO+MH. Record: https://id.rijksmuseum.nl/200424265

## S10-019 A comma inventory loses to a single part

Rijks 200378217 is a blue folder whose title is a comma-separated inventory of a sketchbook part, four loose drawings, an auction catalogue and a letter. The inventory turn is a container. If another turn's whole content is one of those comma-separated phrases, that turn is the part and it outranks the inventory for a query that matches only that phrase. The inventory stays at rank 2 or worse so a summary can still see it. Multi-hop gold that is the letter itself stops losing to the folder that merely lists the letter.

Math: A turn is an inventory when line 0 contains 3 or more commas. Phrases are the comma-split segments. If some other turn's token set equals one segment, that turn's order key is (1, groups) and the inventory's is (0, groups).

Tag: RECALL/INDEX. Slices: MH+SUM. Record: https://id.rijksmuseum.nl/200378217

## S10-020 Location code, acquisition year, creation span

V&A O125746, accession E.2601:2-1910, partNumbers also listing E.2601:1-1910, accessionYear 1910, production date 1558-1635, gallery location text 002. Three different integers sit on one object. A trailing hyphen plus a 4-digit year inside the accession is the acquisition year. A bare 1-3 digit token with no dot and no letters is a location code and is excluded from length and from year parsing. The production span, two 4-digit years joined by a hyphen that are not inside the accession, is the only event date. Instruction-following questions that quote a status code stop matching the object, and event-ordering questions stop dating it to 1910.

Math: acq = the 4-digit run after '-' at the end of an accession-shaped token. loc = a whitespace token matching ^[0-9]{1,3}$ that is not a day next to a month name. object span = 4-digit hyphen 4-digit outside the accession. Event sort uses the span. loc is deleted from the token vector before len is computed.

Tag: RECALL/INDEX. Slices: EO+IF. Record: https://api.vam.ac.uk/v2/object/O125746

## S10-021 A year inside brackets is not the object year

AIC artwork 79505 is A Hunt in the Mountains of Heaven, dated Late Ming / early Qing, 17th century. The artist line is 'Artist unknown [spurious signature of Zhao Mengfu (1254-1322'. The name and both years sit inside brackets. They are not the maker and not the object year. At ingest, tokens and 4-digit years between '[' and ']' are stored as bracket_name and bracket_year and are excluded from entity keys and from object_year. A query for 1254 must not seat this scroll. The public HTML URL for 79505 redirect-loops; the API record returned 200 and is the citation.

Math: Inside a '[' ']' span, suppress entity promotion and remove 4-digit years from object_year. Order OD/EO by (object_year match, groups). A bracket_year match scores 0 for the year component.

Tag: RECALL/INDEX. Slices: OD+EO. Record: https://api.artic.edu/api/v1/artworks/79505

## S10-022 Non-ascii run is an alias of the latin title

AIC 75361 is titled The Peach Blossom Spring 桃花源圖. The CJK run is on the same title line as the latin words. It is an alias feat of that turn, not a separate document and not a translation table. A query written in either script matches the turn. No dictionary is shipped. Open-domain no_pool includes questions whose wording never overlaps the indexed latin tokens; this only helps when the gold turn itself carries both scripts, which is the case this record documents. The same pattern is on Cleveland 145953, whose tombstone writes 五百羅漢圖 beside the English title.

Math: alias_script(t)={maximal runs of non-ascii letters on line 0}. latin(t)={ascii tokens on line 0}. The turn matches if the query intersects either set. Both sets intern as feats of one turn. counts_len is false for the alias run so it does not dilute the latin length.

Tag: RECALL/INDEX. Slices: OD. Record: https://www.artic.edu/artworks/75361

## S10-023 Letter prefix plus digits, no hyphen

Met 51088, Night Rain at Karasaki, accession JP1447. The token has no dot and no hyphen, so it is not a version and not an issue ref. extract_exact_technical_handles requires a hyphen between an uppercase project and an all-digit tail for issue refs, and raw_tokens will keep JP1447 as one token only because it is alphanumeric. Mark that shape (2-6 letters immediately followed by 3-6 digits, no hyphen) as an alpha accession. An exact match on it outranks a turn that contains 1447 alone or the title words alone. Open-domain questions that quote the catalog number stop missing a gold turn whose prose title is a different wording.

Math: alpha_acc matches ^[A-Za-z]{2,6}[0-9]{3,6}$. exact=1 iff the query contains the same string. Order (exact, groups, -len). A proper suffix of the digits does not match.

Tag: RECALL/INDEX. Slices: OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/51088

## S10-024 A private token beats a shared series

Met 55196, accession JP3210, is one print from the series Long Live Japan: One Hundred Victories, One Hundred Laughs. The series words are on many turns; the print's own token is not. IDF already discounts frequent tokens smoothly, and three medium-frequency series words can still outscore one rare word. The ordinal rule fires only when the query contains at least one token with document frequency 1 among the candidate set: that private hit dominates any number of shared hits. Turns are ordered by (private_hits, -shared_hits) as integers.

Math: A token is shared when its df inside the current candidate set is >= 3, private when df = 1. key = (count of private query tokens present, negative count of shared query tokens present). Compared as integers. BM25 is not rescaled.

Tag: RECALL/INDEX. Slices: OD+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/55196

## S10-025 Adjacent suffixes do not match

Met 50538 and 50539 are a pair of sutra covers, accessions 46.186.8 and 46.186.9. They share a stem and a title and their suffixes differ by one. A query that names .8 must not be answered by .9. Nearness of the integer is not a match. exact suffix or no accession credit. Multi-hop questions that cite one of two adjacent numbers are the ones this keeps apart.

Math: acc_credit(q,t)=1 iff the full key including the suffix is equal. |suffix_q - suffix_t|=1 contributes 0, not a partial. Order by (acc_credit, groups).

Tag: RECALL/INDEX. Slices: MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/50538

## S10-026 Shared stem fills the pool under a quantifier

Met 44338 is accession 1994.605.37, a zodiac tiger, and 44341 is 1994.605.40, a zodiac pig. The harvest contains two members of the stem, not twelve, so the rule does not invent the other ten. When the query contains all, every or whole and also contains the stem, every indexed turn with that stem is forced into the pool even if the session cap at retrieval.rs:5675 would have dropped it. Summarization and event_ordering both have all5=0. A set that is never co-present in the pool cannot be seated together. The quantifier is closed class. Without it, S10-025's exact-suffix rule still applies and the siblings stay out of each other's answers. The same union is the BEAM All@5 lever for summarization, event_ordering, multi_session_reasoning, and preference_following, and the LME Turn All@5 lever for multi-session and temporal-reasoning. The structural All@5 cap is 258, 62 short of 320, because the other five BEAM categories are not treated as stem sets. The pool insert does not move Session@1 unless the stem is also interned as a feat before line 5480.

Math: If q contains a token in {all, every, whole} and a stem S, pool = pool union {t | stem(t)=S}. Order inside the set by suffix ascending. The session cap is not applied to this union.

Tag: RECALL/INDEX. Slices: SUM+EO. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/44338

## S10-027 Overlap does not join two fragments

Met 252473 is a fragmentary marble inscription, accession 26.60.6. Two turns that share content words are not the same object. The submodular selector already treats Jaccard >= 0.45 as the same topic for event sequences (retrieval.rs:5647-5651) and then discounts the second turn. That must not be reused as an identity join. When at least one of the two turns carries an accession key, they are the same evidence only if the stems are equal. Shared words do not merge them, and a multi-hop chain cannot hop from one fragment to another on prose overlap alone. Turns with no accession key are left to the existing selector.

Math: join(a,b) = stems equal, and only evaluated when Acc(a) or Acc(b) is nonempty. Otherwise the existing Jaccard path is unchanged. A prose intersection of any size with unequal stems contributes 0 to join.

Tag: RECALL/INDEX. Slices: MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/252473

## S10-028 A repeated short line survives only if it holds a key

The V&A page for O125746 titles the object Handscroll and identifies it by E.2601:2-1910. A repeated first line of at most six tokens is a running header. Abusir drops headers. This rule drops the repeated line from the bag only when it contains no accession key and no version handle. If the line contains one, the line is the identity and stays at full weight, counts_len false so a repeated header does not inflate length. Summarization then sees the key rather than a generic repeated word, and open-domain matching can hit the header without also matching every turn that repeats a section label.

Math: header(t)= line 0 if token_len(line 0)<=6 and the same line 0 occurs on >=3 turns. If header contains an acc key or a version handle, weight stays 1 and counts_len is false. Otherwise the line's tokens are removed from content before scoring.

Tag: RECALL/INDEX. Slices: OD+SUM. Record: https://collections.vam.ac.uk/item/O125746/

## S10-029 One number versus many numbers

Met 325849 is a cuneiform private letter, accession 66.245.8. A letter carries one date. A ledger or a tombstone carries many. When the query contains exactly one 4-digit year or exactly one accession key, prefer the turn whose count of those shapes is 1 over a turn whose count is 3 or more, given equal group coverage. Instruction-following and event-ordering both lose gold between k and top 5; the extra numbers on a long turn are what push a short exact letter down.

Math: nshapes(t)=|object years|+|acc keys|. If |shapes(q)|=1, order by (groups, -|nshapes(t)-1|, -len). The distance |nshapes-1| is an integer. A turn with 4 shapes ranks below a turn with 1.

Tag: RECALL/INDEX. Slices: EO+IF. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/325849

## S10-030 Lot prefix fills ranks behind an exact key

Met 60695 is accession 67.43.15, an oracle bone. Met 42045, also fetched, is 67.43.14, the neighbor in the same lot. The shared prefix 67.43 pulls the neighbor into the pool at rank 2 or worse. It must not take slot 1 when slot 1 already holds the exact key the query named. This is not the oracle-bone verification hedge, which demands a second witness before trusting a reading. Nothing here is verified or doubted. The neighbor is only recalled so a follow-up that omits the last suffix can see the lot, and an exact query keeps its exact answer.

Math: If exact(q, head)=1, insert other turns with the same lot prefix at positions >=1, never at 0. If no exact head exists, the lot does not promote anyone to 0 either; they enter the pool unordered by prefix.

Tag: RECALL/INDEX. Slices: OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/60695

## S10-031 Parenthetical span loses to the bare year

AIC 68041, Bamboo-Covered Stream in Spring Rain, is dated 'Ming dynasty (1368–1644), dated 1441'. The span inside parentheses is the dynasty. The bare 4-digit year outside parentheses is the object year. A query for 1441 matches. A query for 1368 matches the dynasty span only and must not outrank a turn whose object year is 1368. Event order sorts on the bare year when one exists, and on the parenthetical span only when no bare year exists.

Math: bare = 4-digit years whose character span is not inside '(' ')'. paren = 4-digit years that are. object_year = bare if nonempty else paren. A query year inside paren and not in bare scores 0 against a turn that has a bare year.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://www.artic.edu/artworks/68041

## S10-032 Segment index beats a bag hit

The Cleveland tombstone for 1971.16 is period-separated segments: title, artist, medium, dimensions, credit and accession. A query token that falls inside one segment is a field hit. The same token scattered across segments is only a bag hit. Summarization wants the segment, not the bag: the credit line mentioning a fund must not answer a question about the title, and the title must not be diluted by the dimension segment. Feats carry a segment ordinal. Ranking prefers a hit whose query tokens share one segment over a hit of the same size spread across segments.

Math: seg(token) = the 0-based index of the period-separated segment on line 0. concentrated(q,t)=1 if every matched query token shares one seg. Order by (concentrated, groups, -len).

Tag: RECALL/INDEX. Slices: SUM+OD. Record: https://openaccess-api.clevelandart.org/api/artworks/145953

## S10-033 Name before the date and the date are one key

Rijks 200317464 is titled Brief van Mr. F. van der Valk aan Dr. W. Drees, 7 september 1950. The tokens before the date and the date triple are one joint key. A different letter that shares the date, or shares the addressee and not the date, does not match a query that states both. Multi-hop joins on a shared year are the failure this cuts. The month name is closed class. The name tokens are whatever the line already contains; no name list is added.

Math: joint(t)=(tokens on line 0 before the date triple, the triple). Match requires the query's date components and at least one non-closed-class token from the prefix. Order by (joint_equal, groups).

Tag: RECALL/INDEX. Slices: MH+EO. Record: https://id.rijksmuseum.nl/200317464

## S10-034 A letter span has a width

Met 39619, accession 13.220.62a–d, is one pair of table screens whose parts run from letter a to letter d. The width is ord(d)-ord(a)+1 = 4. A query that contains a cardinal or a digit equal to that width is about the whole span. A query that contains one letter matches that part's key only. The span is parsed from the original text because raw_tokens will split a–d into two letters and lose the dash.

Math: width = ord(hi)-ord(lo)+1. If the query contains an integer equal to width, the span turn is a set answer. If the query contains one letter L with lo<=L<=hi, credit is for that letter only and the other letters are excluded from the bag (see the exclusion rule). Width itself is an integer feat.

Tag: RECALL/INDEX. Slices: SUM+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/39619

## S10-035 Slash-adjacent years are one event

Met 716307 is dated '1258 AH/1842–3 CE', accession 2016.536. Two year-shapes sit on either side of a slash. They are one event, not two events a multi-hop or an event-ordering question may split. No era-abbreviation map is used. The signal is orthographic: a 3- or 4-digit run, a slash, and a 4-digit run. The pair is stored as one span. A query for either endpoint hits the span. A query that treats them as two ordered events does not get two turns. The accession year 2016 is still not the object year.

Math: A match of [0-9]{3,4} '/' [0-9]{4} is one span (lo, hi). It contributes 1 to the event count, not 2. object_year endpoints are lo and hi. acq year in the accession is excluded as in S10-004. Event sort key is lo.

Tag: RECALL/INDEX. Slices: EO. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/716307

## S10-036 A range loses to one exact key

Met 62953, Album of Blossoming Plum, accession 2017.327.6a–h. The letter span is a set of parts. When the query names one exact key and does not contain all, every or whole, a turn that holds only that key outranks the span turn. The span stays in the pool behind it. Summarization otherwise answers with the whole album. Open-domain questions that cite one part stop landing on the album record.

Math: If Acc(q) has one key and q has no quantifier, order by (exact_key, -is_span, groups). is_span=1 for a letter or numeric range. The span is demoted, not deleted.

Tag: RECALL/INDEX. Slices: SUM+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/62953

## S10-037 Tokens after 'of' are the owner key

The same Washington bookplate is titled Bookplate of George Washington. The closed-class token of introduces the owner. Those following tokens, up to the next preposition or the end of the line, are an owner key stored as feats with counts_len false. A preference question that uses the owner's name hits this key even when the rest of the title is generic. Preference-following's broad pool is 30 against a need of 36, so gold that shares only a name with the query has to gain a feat the bag was not weighting as identity. This does not boost a preference object class. It indexes a span the turn already has. Speaker is not read.

Math: owner(t)=tokens after 'of' on line 0 until {in, from, before, after}. Match if the query intersects owner(t). Order by (|intersection|, groups). Generic tokens before of do not count toward the intersection when the query intersects the owner span.

Tag: RECALL/INDEX. Slices: PF. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/432480

## S10-038 A year off the first line is a mention

The Arhats API record dates the scroll 1591–1626 and the creator biography says the painter moved to Nanjing by 1600. 1600 is a mention. It is not a second object year and it must not create a second event. Only 4-digit years on line 0, outside accessions and outside brackets, are object years. Years on later lines are mention_year and score 0 for event order. Event-ordering all5 is 0 partly because extra years invent extra events and the true set never sits together. When the updated value is a year on line 0, the same split is the only rule in this set that names BEAM knowledge_update. A knowledge update that is not a year is outside it.

Math: object years = 4-digit spans on line 0 outside Acc and outside brackets. mention years = 4-digit spans on later lines. Event sort uses object years only. A query whose year is only a mention does not promote the turn over one whose object year matches.

Tag: RECALL/INDEX. Slices: EO. Record: https://openaccess-api.clevelandart.org/api/artworks/145953

## S10-039 A repeated last token joins only on an equal stem

Cleveland 1971.126.a is the colophon on folio 167 recto of a Kalpa-sutra, and 1971.127.b is folio 141 verso of the same kind of book under a different accession stem. A catchword — the last content token of one turn equaling the first content token of the next — is not enough to join them. The stems differ (1971.126 versus 1971.127), so they are not one evidence unit. Multi-hop must not hop across that boundary on a repeated word. Join requires stem equality and the catchword. Turns with no accession stem are unchanged.

Math: catch(a,b)= last content token of a equals first content token of b. join = catch AND stem(a)=stem(b) AND stem is nonempty. Otherwise no edge is added.

Tag: RECALL/INDEX. Slices: MH. Record: https://www.clevelandart.org/art/1971.126.a

## S10-040 Hyphenated letter-digit-digit is its own key

The Night Watch record's identifier is SK-C-5, fetched on the Linked Art object. extract_exact_technical_handles splits on the first hyphen and then requires the tail to be all digits, so SK-C-5 fails (the tail is C-5). It is also not a version. Intern the whole string as one key when it matches letters, hyphen, a single letter, hyphen, digits. A query containing SK-C-5 exact-matches this turn and does not match a turn that merely contains the letter C or the digit 5. That is an open-domain and multi-hop no_pool pattern: the query's only discriminative token is an identifier the splitter destroyed.

Math: rijks_id matches ^[A-Za-z]{1,4}-[A-Za-z]-[0-9]{1,6}$. exact=1 on full-string equality after lowercasing. Order (exact, groups, -len). A proper part of the id scores 0.

Tag: RECALL/INDEX. Slices: OD+MH. Record: https://id.rijksmuseum.nl/200107928

## S10-041 Part letter gates closing-leaf inheritance

Inheritance from the closing leaf (greatest numeric suffix) is already a single rule. Combined with the part letter, tokens copy only onto turns whose letter is empty or equal, never onto a different letter of a different stem. Cleveland 1975.22.11 may lend tokens to 1975.22.2. It may not lend them to 1971.127.b. Preference no_pool gold is a leaf the query names; the closer's rare tokens become visible on that leaf and on no other letter. Weight stays the integer ratio 1/2.

Math: Copy if stem equal AND (part(p)=0 OR part(p)=part(c)) AND suffix(p)<suffix(c). Otherwise copy nothing. Score uses cross-multiplied num/den 1/2.

Combined: part-letter ordinal + max-suffix inheritance.

Searches: Robertson, Lu and Macfarlane, XML-structured documents: retrievable units (https://www.staff.city.ac.uk/~sbrp622/papers/RobertsonLuMacfarlane.pdf), inherit ancestor titles into element fields. That pairing was found. The residual below was not in that paper or in the Crossref top hit. Crossref GET /works?query=accession number colophon leaf inheritance retrieval ranking&rows=3 HTTP 200, total-results 846564 (relevance pool, not a boolean AND). Top title: Volume II Accession Number Index DOI 10.1016/b978-0-12-512512-3.50015-8. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: PF+OD. Record: https://www.clevelandart.org/art/1975.22.11

## S10-042 Quoted alias and script alias are one set

AIC 75361 puts 桃花源圖 on the same line as The Peach Blossom Spring. The Night Watch record puts a short name in single quotes beside a long militia title. A turn may carry either kind of alias, or both. The match set is content union quotes union non-ascii runs. A query in the other script, or a query using the quoted short name, hits the turn. Counts_len stays false for both alias kinds so neither dilutes the other. Open-domain no_pool is the slice this can move, and only for golds that already write both names.

Math: Hit = query intersects (content ∪ quoted ∪ non_ascii). |alias hits| is an integer added as the leading tuple component, ahead of groups.

Combined: quoted short name + non-ascii title alias.

Searches: Crossref GET /works?query=quoted title alias multilingual script retrieval ranking&rows=3 HTTP 200, total-results 494573 (relevance pool, not a boolean AND). Top title: Language-Preference-Based Re-ranking for Multilingual Swahili Information Retrieval DOI 10.1145/3539813.3545131. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD. Record: https://www.artic.edu/artworks/75361

## S10-043 Container side cannot win a contents query

A line with the closed-class token in is split into contents and container. A line with three or more commas is an inventory. When both shapes are present, the inventory is demoted and the container side still cannot satisfy a query whose overlap is entirely on the contents side. Rijks 200380645 (a letter in an envelope) is the container witness and 200378217 (the mixed folder) is the inventory witness. Multi-hop stops returning the folder, or the envelope, for a question about the letter.

Math: score = contents_overlap if that is strictly greater than container_overlap, else 0 when the turn is also an inventory. Order (score, -is_inventory, groups).

Combined: 'in' container split + comma-inventory demotion.

Searches: Crossref GET /works?query=container preposition inventory passage retrieval ranking&rows=3 HTTP 200, total-results 344084 (relevance pool, not a boolean AND). Top title: Question Answering Passage Retrieval and Re-ranking Using N-grams and SVM DOI 10.13053/cys-20-3-2470. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: MH+SUM. Record: https://id.rijksmuseum.nl/200380645

## S10-044 Command shape demotes the inventory

The command letter 200436215 is a short order. The folder 200378217 is a long inventory. Instruction-following queries are non-questions whose first token is outside the wh/pronoun/auxiliary/modal closed class. Under that query shape the integer order is (groups, -is_inventory, -len). A short command with the same groups outranks the inventory, and a command whose first line is at most 12 tokens is eligible for the pool. The slice's broad pool is 35 against a need of 36, so eligibility is the part that can move no_pool; the demotion is the part that moves t5_not1.

Math: If command_query: key=(groups, 1-is_inventory, -len). Else this rule does not fire. is_inventory=1 when line 0 has >=3 commas.

Combined: short non-question command + comma-inventory demotion.

Searches: Crossref GET /works?query=imperative sentence length instruction retrieval ranking&rows=3 HTTP 200, total-results 539479 (relevance pool, not a boolean AND). Top title: Top-k Document Ranking and Sentence Similarity Retrieval System for Covid-19 Research Papers DOI 10.21203/rs.3.rs-1885098/v1. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: IF. Record: https://id.rijksmuseum.nl/200436215

## S10-045 Title-line triple ignores the accession year

The Krasnoi letter carries 16 July 1813 on its first line. An accession such as 1971.16 carries 1971 as its prefix. The event triple is parsed only from line 0 and only from tokens that are not inside an accession key. The accession prefix cannot fill a missing year in the triple, and it cannot create a second event. Event order sorts on the triple. A question dated 1813 stops landing on a turn whose only 4-digit number is a catalog prefix.

Math: triple from line 0 after acc tokens are masked. acq_year is never a component of the triple. Sort key Y*10000+M*100+D. A turn with no triple sorts after every turn that has one, regardless of acq_year.

Combined: title-line day-month-year + accession-prefix year exclusion.

Searches: Crossref 10.2307/jj.8692990.24 is a printed concordance of museum accession and catalog numbers (The Cretan Collection in the University of Pennsylvania Museum, vol. III). That is a table, not this rank rule. It was found and is not claimed as unseen. Crossref GET /works?query=title date versus accession year catalog retrieval&rows=3 HTTP 200, total-results 1761518 (relevance pool, not a boolean AND). Top title: Table of Chinese Rulers, Dates of Accession, The Year Names (Nien Hao) with Date of Adoption DOI 10.4324/9780203841792-6. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO. Record: https://id.rijksmuseum.nl/200424265

## S10-046 Location code dropped, acquisition year kept out of the span

V&A O125746 has production 1558-1635, accessionYear 1910 inside E.2601:2-1910, and gallery text 002. The combination stores three integers and lets only the production span answer an event question. The location integer is removed from the token vector before length is computed, so an instruction that quotes 002 does not match the object on a numeric accident, and a question dated 1910 does not match it either. Both exclusions have to hold: dropping the location without freezing the acquisition year still leaves 1910 in the bag.

Math: event = production span only. acq matches a query year with bit 0. loc tokens are deleted, so they change len and cannot match. Order EO by (span_match, -len).

Combined: bare location integer + trailing acquisition year.

Searches: Crossref GET /works?query=museum acquisition year creation date retrieval ranking&rows=3 HTTP 200, total-results 1388621 (relevance pool, not a boolean AND). Top title: Documents Ranking Based on a Hybrid Language Model for Chinese Information Retrieval DOI 10.1109/icia.2006.306010. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+IF. Record: https://api.vam.ac.uk/v2/object/O125746

## S10-047 Bracket year and model year are both ineligible

AIC 79505 puts 1254-1322 inside brackets on a spurious signature. Met 204834 puts 1877 after the word after. Either shape makes a year ineligible as object_year. A turn can contain both shapes. The eligible set is 4-digit years that are outside brackets and not after 'after' on the same line. Event order and open-domain 'when' use the eligible set only. Using just one exclusion leaves the other false year in the bag, which is enough to steal slot 1 on a tie.

Math: eligible = four_digit \ brackets \ after_line. match(q,t)=1 iff q.year intersects eligible(t). A hit only in the excluded set scores 0 and loses to any eligible hit.

Combined: bracketed year suppression + year after 'after'.

Searches: Crossref GET /works?query=spurious signature date bracket retrieval ranking&rows=3 HTTP 200, total-results 362440 (relevance pool, not a boolean AND). Top title: A backend text retrieval machine for signature-based document ranking DOI 10.1007/3-540-54029-6_177. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://api.artic.edu/api/v1/artworks/79505

## S10-048 Range width outranks a prose digit

Met 74462's accession 1974.224.1–.31 has width 31. A how-many question is answered by the width. The WH-slot voter would accept any form_digit on the turn, including a digit in the prose. The combination is that the voter is not consulted when a range width exists: the width is the only numeric answer, and a prose digit scores 0 for the numeric component. Summarization and multi-hop how-many golds stop landing on a turn that merely mentions a different number.

Math: If range width w exists, numeric_answer=w and every other integer on the turn has numeric credit 0. Else this rule does not fire and the existing voter is unchanged. Order by (query digit == w, groups).

Combined: en-dash range width + refusal of a prose digit.

Searches: Crossref GET /works?query=numeric range cardinality digit retrieval ranking&rows=3 HTTP 200, total-results 548177 (relevance pool, not a boolean AND). Top title: The Cardinality-Based Ranking of Opportunity Sets in an Interactive Setting: A Simple Characterization DOI 10.2139/ssrn. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: SUM+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/74462

## S10-049 Query year picks the earlier sibling

Met 44710 is the 1862 textile, accession 1993.51.2. Met 44709 is the 1714 textile, 1993.51.1. They share a title and a stem. This combination is a seat, not a new parser: given two pool members with the same stem, a query year equal to the earlier object year swaps that turn to slot 1 even though its year is smaller and its session may be older. If the query year equals neither, the seat does not move. Torrens would keep the later textile. The reorder ceiling on these slices is the t5_not1 residue (EO 7, OD 17); this rule cannot touch no_pool.

Math: If stem(a)=stem(b) and q_year=object_year(a) and q_year!=object_year(b), a precedes b. Comparison is integer equality, not a distance. No swap when q_year is None.

Combined: sibling stem + object-year equality, earlier allowed to win.

Searches: Crossref GET /works?query=earlier dated version catalog record retrieval ranking&rows=3 HTTP 200, total-results 839119 (relevance pool, not a boolean AND). Top title: Transforming Catalog Displays: Record Clustering for Works of Fiction DOI 10.4324/9780203049624-4. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: ORDERING. Slices: EO+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/44710

## S10-050 Exact suffix first, quantifier fills the rest

Met 50539 is sutra cover 46.186.9, the neighbor of 46.186.8. Without a quantifier the suffixes do not match each other. With all, every or whole, both covers are forced into the pool and the one named by an exact suffix, if any, takes slot 1. The combination is what all5 needs: the set has to be co-present, and the named member has to lead it. Filling the set without the exact-first order answers a specific question with an arbitrary sibling. Exact-first without the fill leaves all5 at 0. The All@5 cap is the same 258 as S10-026. The named suffix stays at slot 1 so the fill does not answer a specific question with a sibling.

Math: Under a quantifier, pool gains every stem member. Order key (suffix==q.suffix, -suffix) so the named one leads and the others follow in suffix order. Without a quantifier, key is exact equality only.

Combined: exact accession suffix + quantifier stem fill.

Searches: Crossref GET /works?query=exact identifier set expansion quantifier retrieval&rows=3 HTTP 200, total-results 664128 (relevance pool, not a boolean AND). Top title: Mise au point d'une imagerie corrélative MET/NanoSIMS pour identifier, localiser et quantifier des nanoparticules dans les tissus biologiques DOI 10.70675/744f0f8cza460z4522zbdc7z0bd7661123d6. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: SUM+EO+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/50539

## S10-051 Alpha accession beats series words

Met 55196 carries JP3210 and a long series title. The alpha accession is a private key (the series words are shared). The combination orders by (exact alpha-acc, private_hits, -shared_hits). A turn that repeats the series title and lacks JP3210 cannot outrank the print the query named, no matter how many series tokens overlap. Open-domain and multi-hop pool misses on a named print are the target.

Math: key=(exact_alpha, private_df1_hits, -shared_df_ge3_hits). Integer lexicographic. BM25 is not rescaled.

Combined: letter-prefix accession + private-token dominance.

Searches: Crossref GET /works?query=series title rare token catalog number retrieval&rows=3 HTTP 200, total-results 3302043 (relevance pool, not a boolean AND). Top title: Rare Copy Number Variants DOI 10.1001/archgenpsychiatry.2010.25. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/55196

## S10-052 Same title stays split, overlap does not join

Met 463958 is accession 16.79, the same predella title as 09.146. Title equality plus a large token overlap is still not identity, because the accessions differ. The combination refuses both merges: dittography may not collapse them, and the Jaccard topic test may not mark them as one evidence unit. A query containing 16.79 returns this record. A query containing only the title keeps both. Multi-hop cannot hop from one accession to the other on the shared title alone.

Math: join=false when both acc sets are nonempty and disjoint, even if title tokens are equal and Jaccard is 1. exact_acc is the leading rank bit.

Combined: two accessions one title + overlap is not identity.

Searches: Crossref GET /works?query=catalog number identity title similarity join retrieval&rows=3 HTTP 200, total-results 1412728 (relevance pool, not a boolean AND). Top title: Using the triangle inequality to reduce the number of comparisons required for similarity-based retrieval DOI 10.1117/12. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/463958

## S10-053 A header survives only as a key

A repeated line of at most six tokens is dropped, unless the atomic accession parser finds a key on it. The V&A header Handscroll is generic and drops. The same object's accession line E.2601:2-1910 is a key and stays, counts_len false. The combination matters because the header test without the key parser would drop the accession line when a museum repeats it, and the key parser without the header test would let a repeated generic word compete with the title. Open-domain and summarization both need the key to be the thing that matches.

Math: drop line 0 iff repeats>=3 AND acc_keys(line 0) is empty AND no version handle. Else keep, and if a key is present set counts_len false.

Combined: running-header test + atomic accession key.

Searches: Crossref GET /works?query=running header accession number indexing retrieval&rows=3 HTTP 200, total-results 658158 (relevance pool, not a boolean AND). Top title: Fast image indexing based on JPEG2000 packet header DOI 10.1145/500946. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD+SUM. Record: https://collections.vam.ac.uk/item/O125746/

## S10-054 Plate index and object year stay different answers

Accession 20.90.37(27) dated 1772 has a plate index and an object year. A 1-3 digit query matches the plate and not the year. A 4-digit query matches the year and not the plate. The combination is the width test plus the exclusion: 27 does not enter the year set, and 1772 does not enter the plate set. Either leakage answers an event question with a plate index or a plate question with a year.

Math: If width(q_digit)<=3, match plate_ord only. If width==4, match object_year only. The other field's credit is 0. Order by that one credit, then groups.

Combined: parenthetical plate ordinal + object year.

Searches: Crossref GET /works?query=parenthetical plate number versus year catalog retrieval&rows=3 HTTP 200, total-results 2033139 (relevance pool, not a boolean AND). Top title: Willamette University Bulletin Catalog Number for the ninety-ninth year March, 1940 DOI 10.31096/bulletinscatalogs1940vol34no2. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/432480

## S10-055 Fewer shapes, then shorter length

The vase fragment 247733 is the short unit. A letter has one date shape; a tombstone has many. The seat orders a covered-group tie by shape distance from the query first, and by token length second. A short fragment with one shape beats a long roll with the same groups and several numbers. This is an ordering rule inside the top of the pool. It does not pull no_pool gold. The residue it can move is t5_not1.

Math: If groups equal: order by (|nshapes(t)-nshapes(q)| ascending, len ascending). Both components are integers. If groups differ, this seat does not fire.

Combined: shape-count distance + shorter-fragment tie break.

Searches: Crossref GET /works?query=short record long record numeric feature ranking retrieval&rows=3 HTTP 200, total-results 2753631 (relevance pool, not a boolean AND). Top title: The long and short of slipped discs DOI 10.1136/vr.f4767. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: ORDERING. Slices: OD+SUM+IF. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/247733

## S10-056 Lot mates sit behind the exact bone

67.43.15 (Met 60695) and 67.43.14 share a lot prefix. The combination fills the lot into the pool and then locks slot 1 to the exact full key when the query contains one. A prefix match alone never inserts at 0. Open-domain gold named by the full key stays put; the neighbor is recalled behind it for a query that only has the lot. This is not a verification hedge: an exact key is trusted on its own.

Math: If exact(q,t)=1, rank(t)=0 among the lot. Others with the same prefix append at >=1. A prefix-only match has exact=0 and cannot win a tie against exact=1.

Combined: lot-prefix pool fill + exact accession slot lock.

Searches: Crossref GET /works?query=lot prefix exact catalog number retrieval ranking&rows=3 HTTP 200, total-results 760300 (relevance pool, not a boolean AND). Top title: Highlighting exact matching via marking strategies for ad hoc document ranking with pretrained contextualized language models DOI 10.1007/s10791-022-09414-x. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/60695

## S10-057 Named part excluded from the pair

Accession 65.152.2, .3 expands to two keys. If the query names one of them, the other part's tokens are removed from the counted set before groups and length are computed. Expansion without exclusion lets the unasked cup inflate the bag. Exclusion without expansion cannot see the abbreviated '.3' at all. Multi-hop and open-domain questions that cite one number of a pair are the miss this addresses.

Math: Sib from the comma-dot parse. If q intersects Sib, counted = tokens of the named key only, len = |counted|. Order (groups on counted, -len).

Combined: abbreviated sibling expansion + other-part exclusion.

Searches: Crossref GET /works?query=paired object part exclusion retrieval ranking&rows=3 HTTP 200, total-results 2116875 (relevance pool, not a boolean AND). Top title: Scalable object-class retrieval with approximate and top-k ranking DOI 10.1109/iccv. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: MH+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/204540

## S10-058 Three suffixes seat in order under a quantifier

Met 471951 is plaque 69.238.2, between .1 and .3. Under all, every or whole the three plaques enter the pool together and seat in suffix order, not session order. Event_ordering and summarization have all5=0. Co-presence without order still fails an ordering question. Order without co-presence cannot fill all5. The quantifier is closed class. Without it, exact suffix wins and the others stay out.

Math: If quantifier and stem matches: members sorted by suffix ascending, all inserted. Session timestamp is not a key. If no quantifier: exact suffix only.

Combined: numeric suffix order + quantifier stem fill.

Searches: Crossref GET /works?query=ordered catalog suffix set retrieval ranking&rows=3 HTTP 200, total-results 550881 (relevance pool, not a boolean AND). Top title: Suffix Trees for Document Retrieval DOI 10.15368/theses. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+SUM. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/471951

## S10-059 Letter-span width is the cardinal

13.220.62a–d has alphabetic width 4. The numeric range rule answers how-many from an en-dash of integers. This combination accepts a letter span's width as the same kind of answer, and still refuses a prose digit. A query digit 4 matches the span. A query digit that appears only in the prose of the screens does not. Summarization how-many golds that cite a pair or a run of parts are the target.

Math: width = ord(hi)-ord(lo)+1 for a letter span, or hi-lo+1 for a numeric range. numeric_answer = that width. Other integers on the turn have numeric credit 0 while a width exists.

Combined: letter-span width + range width as how-many.

Searches: Crossref GET /works?query=alphabetic part span cardinality retrieval&rows=3 HTTP 200, total-results 1834268 (relevance pool, not a boolean AND). Top title: Lexical retrieval in alphabetic and non-alphabetic scripts: Evidence from brain imaging DOI 10.4324/9780203858462-24. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: SUM+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/39619

## S10-060 A slash span is one event and not an accession

716307's date string contains 1258/1842 and its accession is 2016.536. The slash pair is one event. The accession's 2016 is not a third endpoint and not a second event. Event count on this turn is 1. all5 stays 0 if a single dated object is split into two events and then fails to match either query endpoint cleanly. Both endpoints match the one span; 2016 matches nothing in the event set.

Math: events(t) = {slash span} if present, else {line-0 object years}. Acc years are subtracted from that set. |events|=1 for this record. Sort key = span.lo.

Combined: slash-adjacent years as one event + accession year exclusion.

Searches: Crossref GET /works?query=dual calendar slash date single event retrieval&rows=3 HTTP 200, total-results 2184638 (relevance pool, not a boolean AND). Top title: MAPseq and BARseq Dual Sectioning Event for a Single Mouse Brain v1 DOI 10.17504/protocols. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/716307

## S10-061 Owner span is an alias, quotes are an alias

The owner span after of and a quoted short name are both aliases of the same turn. On the Washington bookplate the owner span is the name after of. On the Night Watch record the short name is quoted. A preference or open-domain query hits the turn if it intersects either alias, and the generic words before of or outside the quotes do not count when the alias intersection is nonempty. Preference no_pool is 9 against a need to recover 6. An alias written on the gold turn is one of the few legal ways to add overlap without a synonym list.

Math: alias = owner_span ∪ quoted. If query intersects alias, leading bit = 1 and the tokens outside alias do not add groups. Else leading bit = 0 and ordinary groups apply.

Combined: tokens after 'of' + quoted short name.

Searches: Crossref GET /works?query=genitive owner quoted alias retrieval ranking&rows=3 HTTP 200, total-results 212621 (relevance pool, not a boolean AND). Top title: Web Retrieval and Ranking DOI 10.1007/978-3-540-77659-8_11. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: PF+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/432480

## S10-062 Sender prefix and the date triple bind

A query that contains both a content token and a full day-month-year matches a letter only when the token sits in the prefix before the triple and the triple matches. Rijks 200317464 has both on one line. A different letter in the same month fails the triple. The same sender without the date fails the prefix test. Multi-hop chains that join every 1950 letter are the miss. Event order uses the triple only after the prefix has matched, so a bare year cannot order this slice's answer.

Math: match = prefix_intersect > 0 AND triple_equal. Both are bits. Order (match, Y*10000+M*100+D). A single-bit hit does not beat a two-bit hit.

Combined: name-before-date prefix + title-line date triple.

Searches: Crossref GET /works?query=letter sender date joint retrieval ranking&rows=3 HTTP 200, total-results 1450477 (relevance pool, not a boolean AND). Top title: Beyond Retrieval: Joint Supervision and Multimodal Document Ranking for Textbook Question Answering DOI 10.2139/ssrn.6079476. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: MH+EO. Record: https://id.rijksmuseum.nl/200317464

## S10-063 A decimal cannot answer a title integer

The Arhats tombstone carries 39.5 and 2646.5 as measurements and carries Five Hundred in the title plus the span 1591–1626. A how-many query may match a 2- or 3-digit integer on the title segment. It must not match the integer part of a one-digit decimal. The combination is the class test: if the original substring has exactly one digit after a dot, the whole number is a measurement and is removed from the title-integer set. Summarization and open-domain numeric questions stop preferring the dimension.

Math: measure if the source span matches [0-9]+'.'[0-9] with |right|=1. title_int = line-0 integers of width 2 or 3 that are not measures, not years, not accession parts. Order (|q intersect title_int|, -|q intersect measure|).

Combined: one-digit decimal is a measurement + title integer is the count.

Searches: Crossref GET /works?query=decimal measurement versus integer catalog retrieval&rows=3 HTTP 200, total-results 1743673 (relevance pool, not a boolean AND). Top title: Integer representation of decimal numbers for exact computations DOI 10.6028/nist.ir.7144. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD+SUM. Record: https://www.clevelandart.org/art/1971.16

## S10-064 Catchword join still needs the stem

Folio 141 verso (1971.127.b) and folio 167 recto (1971.126.a) are the same kind of book under different stems. A repeated boundary token would catchword-join them. The combination allows the join only when the catchword holds and the stems are equal and nonempty. These two stems differ, so no multi-hop edge is created. Turns with no stem are left untouched, which keeps the rule off dialogues that have no catalog key.

Math: edge(a,b)=1 iff last(a)=first(b) AND stem(a)=stem(b)≠0. Else 0. Overlap size is not a term.

Combined: catchword position + accession-stem join.

Searches: Folio recto/verso guides (Sinai, MMFC, MediManus) describe cataloguing notation. None of those pages is a ranker. no prior combination found in searches of catchword join with an accession-stem gate. Crossref GET /works?query=catchword manuscript join accession retrieval&rows=3 HTTP 200, total-results 227422 (relevance pool, not a boolean AND). Top title: Design and Implementation of New Media Manuscript Retrieval System DOI 10.5220/0012282900003807. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: MH. Record: https://www.clevelandart.org/art/1971.127.b

## S10-065 Dynasty span loses to the bare year

AIC 68041 writes (1368–1644) and 1441. The combination keeps the parenthetical span as a secondary key and the bare year as the object year. A query for 1441 outranks a query-year hit that only lands inside the parentheses. A query for 1368 may match the secondary key and must not outrank a turn whose bare year is 1368. Event order sorts on the bare year when both exist.

Math: key=(bare_equal, paren_equal, -bare_year). bare_equal dominates. A paren-only hit has bare_equal=0.

Combined: parenthetical year span + bare object year.

Searches: Crossref GET /works?query=parenthetical dynasty date versus inscribed year retrieval&rows=3 HTTP 200, total-results 1490386 (relevance pool, not a boolean AND). Top title: Sparta. Inscribed tile, date? DOI 10.1163/1874-6772_seg_a56_462. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://www.artic.edu/artworks/68041

## S10-066 Both part numbers stay addressable

The V&A record lists partNumbers E.2601:1-1910 and E.2601:2-1910. Each part is a full three-slot key sharing prefix, body and acquisition year, and differing in the part integer. A query containing :2 matches only the second. The shared acquisition year 1910 and the shared body 2601 must not bridge the two parts. Multi-hop and open-domain questions that quote one part number are the miss. Both parts remain in the pool when the query has all, every or whole.

Math: exact=1 iff prefix, body, part and acq are all equal. A match on any proper subset scores 0. Under a quantifier the pool still gains every part of the body. Slot 1 stays the exact part.

Combined: three-slot accession + exact part inequality.

Searches: Crossref HTTP 200 on the last retry with no usable title. OpenAlex GET /works?search=museum part number exact match retrieval ranking&per-page=1 HTTP 200, meta.count 1759. Top title: Relevance and Ranking in Geographic Information Retrieval (https://doi.org/10.14236/ewic/fdia2011.1). The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: MH+OD. Record: https://api.vam.ac.uk/v2/object/O125746

## S10-067 Identifier exact beats the quoted alias

SK-C-5 and the quoted short name are both on Rijks 200107928. If the query contains the identifier, exact identifier match is the leading bit and the quoted alias cannot outrank it. If the query has no identifier, the quoted alias is the leading bit. A different painting that quotes the words night and watch without the identifier loses to SK-C-5 when the query contains the identifier. Open-domain and multi-hop no_pool golds that are named by an id the splitter destroyed are the ones this can add to the pool.

Math: key=(id_exact, alias_hit, groups). id_exact dominates alias_hit. Both are 0/1 integers.

Combined: hyphenated letter identifier + quoted alias.

Searches: Crossref GET /works?query=museum object identifier quoted title retrieval&rows=3 HTTP 200, total-results 796410 (relevance pool, not a boolean AND). Top title: Digital Object Identifier (DOI®) System DOI 10.1201/b11499-52. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD+MH. Record: https://id.rijksmuseum.nl/200107928

## S10-068 A mention year cannot split a slash event

Met 716315 is an album leaf dated to the first half of the 19th century, accession 2016.537, the neighbor of the double-dated leaf 716307. A year that occurs off line 0 is a mention. A slash pair on line 0 is one event. The combination deletes mention years from the event set and refuses to split the slash pair into two events because a mention year sits between the endpoints. Event count stays 1. all5 cannot recover while one object is exploded into several years.

Math: events = slash spans on line 0, else bare years on line 0. Subtract mention years and accession years. |events| is the integer used as the event count. Sort by the first event's lo.

Combined: slash span is one event + off-line year is a mention.

Searches: Crossref GET /works?query=inscription date body year event detection retrieval&rows=3 HTTP 200, total-results 3261387 (relevance pool, not a boolean AND). Top title: Sparta. Vase inscription, date? DOI 10.1163/1874-6772_seg_a29_389. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/716315

## S10-069 No quantifier demotes the range, a quantifier uses its width

Album 2017.327.6a–h is a span. If the query has all, every or whole, the width is the numeric answer and the members are filled. If it does not, a single exact key outranks the span. The combination is the branch. One branch without the other either always dumps the whole album on a specific question or never answers how many of the set. Summarization all5 is 0 until the quantifier branch fills the set; the other branch is what keeps ordinary questions precise.

Math: If quantifier: numeric=width and pool gains the span's keys. Else: order by (exact_key, -is_span). The two branches share the span parse and nothing else.

Combined: range demotion without a quantifier + width as how-many.

Searches: Crossref GET /works?query=set cardinality quantifier catalog range retrieval&rows=3 HTTP 200, total-results 769508 (relevance pool, not a boolean AND). Top title: Correction to the B. Chakraborty's article ``On the cardinality of a reduced unique range set'' DOI 10.3842/umzh. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: SUM+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/62953

## S10-070 Full title-line date picks the sibling

44709 is dated March 18, 1714, not merely 1714. Its sibling 44710 is October 9, 1862. A query that states day, month and year must match the whole triple. Matching the year alone is not enough to swap the earlier sibling into slot 1 when the query also states a month. Event order uses the triple. Two textiles that share a year and not a month stay tied rather than letting the later one win.

Math: If stem equal and the query has a month: match requires Y, M and D. Year-only equality does not fire the sibling swap. Sort key Y*10000+M*100+D.

Combined: sibling stem + day-month-year triple.

Searches: Crossref GET /works?query=dated sibling object day month year retrieval&rows=3 HTTP 200, total-results 1407968 (relevance pool, not a boolean AND). Top title: Month-Day-Year DOI 10.32388/lypvxp. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/44709

## S10-071 Parenthetical alias of either script

The Arhats tombstone writes The Five Hundred Arhats (五百羅漢圖). The parenthetical is CJK, so it is both a parenthetical alias and a script alias. The API also lists alternate titles in latin (Lohans, Disciples of Buddha). For a turn, a parenthetical on line 0 is an alias whether its letters are ascii or not. A query using either the CJK or the latin parenthetical hits the turn. Counts_len is false. Open-domain no_pool moves only when the gold turn itself carries the second name, which this tombstone does.

Math: alias = tokens inside line-0 parentheses, ascii or not, union non-ascii runs. Hit if query intersects alias. Leading bit 1. The parentheses are not tokens.

Combined: parenthetical alias + non-ascii script alias.

Searches: Crossref GET /works?query=parenthetical original title multilingual retrieval&rows=3 HTTP 200, total-results 503359 (relevance pool, not a boolean AND). Top title: Multilingual information retrieval system DOI 10.1117/12. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD. Record: https://openaccess-api.clevelandart.org/api/artworks/145953

## S10-072 Last segment is credit, not the event

The last period-separated segment of the Arhats tombstone holds the fund name and 1971.16. The first segment holds 1591–1626. The combination marks the last segment as credit: its years are acquisition years, and a query about the object must concentrate its hits in an earlier segment. A summary that quotes the fund stops outranking the title segment. An event question dated 1971 stops matching the scroll.

Math: credit = the highest segment index on line 0. Years in that segment are acq. object years are bare years in lower segments. concentrated prefers a shared segment that is not the credit segment when the query has a year.

Combined: segment index + accession year exclusion.

Searches: The Crossref concordance chapter 10.2307/jj.8692990.24 lists accession numbers against catalog numbers. It does not rank a credit segment against an object date. Crossref GET /works?query=credit line accession date versus object date retrieval&rows=3 HTTP 200, total-results 1881394 (relevance pool, not a boolean AND). Top title: The Accession Date of Hatshepsut DOI 10.4324/9780203038062-8. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+SUM. Record: https://www.clevelandart.org/art/1971.16

## S10-073 Short fragment joins only on the stem

The vase fragment is the short turn. A longer turn that shares its words is not the same object unless the accession stems match. The combination ranks the shorter turn first on a group tie and refuses the join when the stems differ or are empty. Multi-hop cannot walk from the fragment to the essay. Open-domain slot 1 on a tie stays with the fragment. Dialogues with no accession are unchanged, so the length tie-break does not fire on them through this rule.

Math: If either acc set is nonempty: join iff stems equal. On a group tie among unjoined turns, shorter len wins. If both acc sets are empty, neither clause fires.

Combined: shorter-fragment preference + stem-only join.

Searches: Crossref GET /works?query=fragment join short text accession retrieval&rows=3 HTTP 200, total-results 1134993 (relevance pool, not a boolean AND). Top title: Hash join algorithms used in text-based information retrieval: Guidelines for users DOI 10.1109/itsim. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: MH+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/247733

## S10-074 Command outranks inventory inside instructions

The folder inventory is the long comma list. A command query (no question mark, first token outside the closed wh/pronoun/auxiliary/modal class) orders by (groups, -inventory, -len) and admits a turn whose first line has at most 12 tokens into the pool. Instruction-following needs one more no_pool gold and every in-pool miss. The eligibility bit is the recall half. The demotion is the ordering half inside the top 5. Speaker is not consulted.

Math: command_query as in S10-017. key=(groups, 1-inventory, -len). Pool eligibility if line0_len<=12 and groups>0. inventory = commas>=3.

Combined: command-query shape + inventory demotion + length.

Searches: Crossref GET /works?query=instruction command versus list retrieval ranking&rows=3 HTTP 200, total-results 1466124 (relevance pool, not a boolean AND). Top title: Table 11: Paraphrase test sets versus CAsT baselines performance after reformulation, retrieval, and re-ranking DOI 10.7717/peerjcs. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: IF. Record: https://id.rijksmuseum.nl/200378217

## S10-075 A plate index does not pull a lot

(27) on accession 20.90.37(27) is a plate index. It is not a lot suffix and it must not pull other turns whose accession merely ends in 27. Lot fill runs on the dotted stem 20.90, and only behind an exact key. The plate number is excluded from the stem used for that fill. Open-domain otherwise gathers a pile of unrelated 27s into the pool and the true key falls out of the top 50.

Math: lot_stem = dotted key with the parenthetical removed. plate_ord is not a suffix. Fill uses lot_stem only. A query digit equal to plate_ord does not start a fill.

Combined: parenthetical plate + lot-prefix fill.

Searches: Crossref GET /works?query=plate number lot prefix catalog retrieval&rows=3 HTTP 200, total-results 840635 (relevance pool, not a boolean AND). Top title: Definitional Prefix Lifts Contextual Retrieval on Long Legal Documents DOI 10.2139/ssrn.6913480. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/432480

## S10-076 A dotted component is not the alpha key

JP1447 (Night Rain at Karasaki, Met 51088) has no dot and no hyphen. A dotted accession such as 67.43.15 contains digit runs that a splitter will isolate. The combination refuses to let a component of a dotted key satisfy an alpha key. Query token JP1447 matches only the unbroken letter-digit string. A turn whose bag contains 1447 because a dotted accession was split scores 0 on this bit. Open-domain and multi-hop questions that quote the print number stop landing on whatever object happens to contain those digits.

Math: alpha_hit=1 iff the original text contains the query's letter-digit token as one whitespace token. A digit run that is a component of a dotted or hyphenated key scores 0. Order (alpha_hit, groups, -len).

Combined: hyphen-free letter-digit key + dotted accession inequality.

Searches: Crossref GET /works?query=alphanumeric accession versus dotted catalog number retrieval&rows=3 HTTP 200, total-results 1248566 (relevance pool, not a boolean AND). Top title: Figure 2—figure supplement 4. Relationship between sequencing depth and number of TE variants discovered in each accession DOI 10.7554/elife. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/51088

## S10-077 A date on the container side does not count

Tokens after in are the container. A day-month-year that falls on the container side is not the letter's event. The triple is parsed from the contents side only. A multi-hop question about when the letter was written stops matching a date that belongs to the envelope, the folder, or the covering note. Event order uses the contents-side triple.

Math: Mask tokens after 'in' before parsing the triple. A date that exists only on the container side yields no triple. Sort key is the contents triple or none.

Combined: 'in' container split + title-line date triple.

Searches: Crossref GET /works?query=container date versus letter date retrieval&rows=3 HTTP 200, total-results 1370849 (relevance pool, not a boolean AND). Top title: Spot Date 2466 DOI 10.3998/gabii. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+MH. Record: https://id.rijksmuseum.nl/200380645

## S10-078 A location header drops, a key header stays

Gallery text 002 is a bare 1-3 digit token. A repeated header that is only a location code drops even though it is numeric, because a location code is not an accession key. A repeated header that contains E.2601:2-1910 stays. Instruction-following questions that quote a status or a bay number stop hitting every object in that bay. Open-domain matching keeps the accession header.

Math: loc tokens are deleted first. Header fate is then computed on what remains: drop if repeats>=3 and no acc key. A header whose only token was a location code therefore drops.

Combined: header fate + location-code deletion.

Searches: Crossref GET /works?query=gallery location code header indexing retrieval&rows=3 HTTP 200, total-results 577572 (relevance pool, not a boolean AND). Top title: Fast image indexing based on JPEG2000 packet header DOI 10.1145/500946.500948. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: IF+OD. Record: https://collections.vam.ac.uk/item/O125746/

## S10-079 Brackets suppress a name, the script alias remains

AIC 75361 carries a CJK title on the line and a spurious signature in the artist text. Brackets remove the signature's name and its years from entity and object-year keys. The CJK run is outside those brackets and stays an alias. A query in CJK still hits the scroll. A query for the bracketed name or the bracketed years does not. Using suppression alone would also be at risk of eating the CJK if a parser treated every non-ascii span as suspicious. The combination keeps the script alias iff it stands outside brackets.

Math: alias = non-ascii runs whose spans do not intersect a bracket span. Years inside brackets are ineligible. Order (alias_hit, object_year_match, groups).

Combined: bracket suppression + non-ascii alias.

Searches: Crossref GET /works?query=spurious signature multilingual title retrieval&rows=3 HTTP 200, total-results 506265 (relevance pool, not a boolean AND). Top title: Multilingual information retrieval system DOI 10.1117/12.257307. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: OD+EO. Record: https://www.artic.edu/artworks/75361

## S10-080 Width must equal the set that was filled

Plaque 69.238.3 is the third member. The filled set has size 3, which is the width of suffixes 1..3. A how-many query matches only if the query digit equals the number of members actually inserted into the pool. A width parsed from the text that does not equal the filled count is not the answer. Summarization and event_ordering all5 stay 0 when the system claims a set it did not retrieve. The check is an integer equality after the fill.

Math: filled = |{t in pool : stem(t)=S}|. answer if query digit == filled. A textual width that differs from filled scores 0. Order by equality, then suffix.

Combined: range or letter width + stem fill.

Searches: Crossref GET /works?query=set size catalog suffix count retrieval ranking&rows=3 HTTP 200, total-results 994768 (relevance pool, not a boolean AND). Top title: Suffix Trees for Document Retrieval DOI 10.15368/theses.2012.90. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: SUM+EO. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/471952

## S10-081 A model year cannot win the sibling test

204834's 1877 sits after the word after, so it is a model year. A sibling test that treated 1877 as the object year would swap this turn into slot 1 for a query dated 1877. The combination removes model years from the sibling comparison. Only object years participate. If the only year on the turn is a model year, the sibling swap does not fire. Event order stays on the object year, which on this record is the century phrase and not 1877.

Math: sibling_year(t)=object_year(t) excluding years after 'after' on the same line. Equality uses sibling_year only. A model-year-only turn compares as None and does not win.

Combined: year after 'after' + sibling year equality.

Searches: Crossref GET /works?query=model date versus object date duplicate retrieval&rows=3 HTTP 200, total-results 4245012 (relevance pool, not a boolean AND). Top title: Table 3: The Ka/Ks ratios and date of duplication for duplicate <i>CqBBX</i> genes DOI 10.7717/peerj.14463/table-3. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/204834

## S10-082 Equal dates do not fall back to accession order

09.146 and 16.79 share a title and the same date span. Accession numeric order would pick 16.79 as later. Date order cannot pick either. The combination forbids both fallbacks: when title tokens are equal and object-year spans are equal, the seat does not move, and both stay addressable. A query that contains one accession still exact-matches that one. This is an ordering refusal. It does not pull no_pool gold. The t5_not1 residue is where a false recency swap is currently free to fire.

Math: If title_eq and span_eq and acc sets differ: seat delta = 0. exact_acc still orders a query that contains a key. No comparison of acc integers and no comparison of session timestamps is allowed in this branch.

Combined: two accessions one title + refusal of recency.

Searches: Crossref GET /works?query=duplicate catalog number same date retrieval ranking&rows=3 HTTP 200, total-results 856890 (relevance pool, not a boolean AND). Top title: Near-Duplicate Video Retrieval DOI 10.1007/springerreference_65375. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: ORDERING. Slices: OD+EO. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/463192

## S10-083 One date triple beats a pile of years

The Krasnoi letter has one date triple on the first line. A ledger turn has many year shapes. On a tie of covered groups, the seat orders by distance of shape count from the query, and a turn that parsed one triple ranks ahead of a turn that parsed several. This moves gold that is already near the top (k_not5) and does not claim no_pool. Instruction-following and event-ordering both lose a handful of golds between k and the top 5 for this reason.

Math: If groups equal: order by (|n_triples(t) - n_dates(q)| ascending). n_triples counts line-0 triples only. A mention year does not increment the count.

Combined: title-line date triple + shape-count distance.

Searches: Crossref GET /works?query=single dated letter versus list of dates retrieval&rows=3 HTTP 200, total-results 2962250 (relevance pool, not a boolean AND). Top title: Data list DOI 10.2307/j.ctvxcr80d.4. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: ORDERING. Slices: EO+IF. Record: https://id.rijksmuseum.nl/200424265

## S10-084 Expanded sibling, then exclusion

Met 762162 is a pair of stirrups, accession 2017.313a, b. The abbreviated second part expands to a full key. A query that says both or two, which are closed class, keeps both parts in the counted set. A query that names one letter excludes the other. Summarization questions about the pair need both parts present. Multi-hop questions about one stirrup need the other gone. The quantifier-like closed class is the branch, and the expansion has to happen before the branch can see the second part.

Math: Expand comma-letter siblings first. If query contains both or two, counted = union of parts. Else if query names one letter, counted = that part. Order by groups on counted.

Combined: abbreviated sibling + part exclusion + exact key.

Searches: Crossref GET /works?query=pair accession both parts retrieval ranking&rows=3 HTTP 200, total-results 575306 (relevance pool, not a boolean AND). Top title: Automated retrieval and ranking of similar parts in agile manufacturing DOI 10.1080/07408179708966407. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: MH+SUM. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/762162

## S10-085 Named plaque leads, the run follows

69.238.1 is the first plaque. Under a quantifier the three plaques fill in suffix order and the one the query names, if any, is moved to the front of that run without being removed from it. Event and summary questions need the whole run in the top 5. A multi-hop question that names .1 needs that plaque at slot 1 rather than whichever sibling shares more prose. Fill without a lock answers the specific question with .2 or .3. A lock without a fill leaves all5 at 0.

Math: Run = stem members sorted by suffix. If q.suffix is in the run, rotate that member to index 0 and keep the others in suffix order after it. Session time is not consulted.

Combined: exact suffix lock + suffix-ordered fill.

Searches: Crossref GET /works?query=exact member ordered set retrieval ranking&rows=3 HTTP 200, total-results 644082 (relevance pool, not a boolean AND). Top title: Highlighting exact matching via marking strategies for ad hoc document ranking with pretrained contextualized language models DOI 10.1007/s10791-022-09414-x. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+SUM+MH. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/471950

## S10-086 Version pin beats a greater version number

extract_exact_technical_handles already stores dotted versions of three or more parts. The registry page shows 0.10.0 sorting above 0.1.3 as a tuple and still flagged not-latest. A query that contains a version handle must exact-match that handle. A greater version tuple must not take slot 1. Instruction-following gold is the version the user named. Open-domain gold that cites a version is the same shape. No new parser is added. The seat compares handle equality. Dialogues have no isLatest flag; the registry is the witness that a greater number is not the requested object, and the rule copies that refusal.

Math: If the query contains a version handle v: order by (turn handles contain v, groups). A turn whose greatest version tuple is larger and does not contain v has leading bit 0. No numeric compare of versions is performed.

Combined: exact version handle + refusal of numeric latest.

Searches: The fetched registry page marks cyanheads/met-museum-mcp-server 0.1.1, 0.1.2, 0.1.3 and 0.10.0 all isLatest false. Numeric order is not the registry's current bit. GitHub repo https://github.com/cyanheads/met-museum-mcp-server returned 200. Crossref GET /works?query=software version pin exact match versus latest retrieval&rows=3 HTTP 200, total-results 1752120 (relevance pool, not a boolean AND). Top title: Augmented Individual Income Tax Model Exact Match File, 1972 DOI 10.3886/icpsr07667. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: IF+OD. Record: https://registry.modelcontextprotocol.io/v0.1/servers?search=museum&limit=20

## S10-087 Dimension segment cannot supply a year

On the Arhats tombstone the decimals 39.5 and 2646.5 sit in the dimension segment and the span 1591–1626 sits in the title segment. A measurement is ineligible as a year even if a later bug promotes its integer part. The combination requires both: the numeric class must be a year or a span, and the segment must be the title segment, before an event match is counted. A summary of the size stops answering a question about the date, and a date question stops matching the width.

Math: event_hit = class is Year or Span AND seg is the title segment (the lowest segment that contains a 4-digit span). A measure in any segment scores 0 for the event bit.

Combined: segment concentration + measurement class.

Searches: Crossref HTTP 429 on the last retry with no usable title. OpenAlex GET /works?search=dimension field date field catalog retrieval&per-page=1 HTTP 200, meta.count 12567. Top title: Introduction to Formal Concept Analysis and Its Applications in Information Retrieval and Related Fields (https://doi.org/10.1007/978-3-319-25485-2_3). The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+SUM. Record: https://openaccess-api.clevelandart.org/api/artworks/145953

## S10-088 Inheritance stops at an exact key

Leaf 1975.22.2 may receive tokens from colophon 1975.22.11 at weight 1/2. If the query already contains the exact key 1975.22.2, those inherited tokens must not change the rank against a turn that actually contains them: the exact key locks slot 1, and inherited feats are ignored for the leading bit. The copy still exists so a query that does not name the key can find the leaf through the colophon's words. Preference and open-domain no_pool are the queries that do not share the leaf's own bag and do share the closer's. The lock is what keeps a named leaf from being stolen by the colophon itself. The session-bag path is the same as S10-012. This row does not name summarization, so its All@5 cap is the preference term only, 146.

Math: If exact(q, t): leading bit = exact, inherited feats contribute 0. Else inherited feats contribute at 1/2 via cross-multiplied integers. The colophon does not outrank an exact leaf.

Combined: max-suffix inheritance + exact accession lock.

Searches: Robertson, Lu and Macfarlane inherit ancestor titles into every child element. They do not stop the copy when the child already carries an exact identifier the query named. Crossref GET /works?query=metadata inheritance exact identifier retrieval ranking&rows=3 HTTP 200, total-results 333380 (relevance pool, not a boolean AND). Top title: Request for feedback on grant identifier metadata DOI 10.5555/q5nnatfrfc. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: PF+OD. Record: https://www.clevelandart.org/art/1975.22.2

## S10-089 Current registry bit is not the greatest version

When a turn carries a version handle and a boolean current-bit copied at ingest from a registry record, numeric order must not override that bit. The dialogue transfer, for turns that have no registry bit, is the pin rule: the named version wins. For turns that do carry the bit, a query with no version handle prefers current-bit true, and a query that names a version prefers the pin even when the bit is false. Instruction-following t5_not1 is a named version sitting in the top 5 under a greater version. This seat runs after Step 33 and only when a version handle is on the query or a current-bit is on the candidate. It does not read bodies.

Math: If q has version v: order by handle equality. Else if any candidate has current_bit: order by the bit. Else the seat does not fire. No comparison of version tuples in either branch.

Combined: isLatest flag + version pin.

Searches: On the registry page fetched for museum, versions 0.1.1 through 0.10.0 of io.github.cyanheads/met-museum-mcp-server are all isLatest false, including the numerically greatest. com.agentfieldtrip/museum 1.0.0 is isLatest true. Crossref GET /works?query=isLatest software version registry retrieval ranking&rows=3 HTTP 200, total-results 1046642 (relevance pool, not a boolean AND). Top title: Ranking application software retrieval results using application attributes and context DOI 10.1109/jcsse.2012.6261957. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: ORDERING. Slices: IF. Record: https://github.com/cyanheads/met-museum-mcp-server

## S10-090 A credit year never leads a sibling test

44710 is dated October 9, 1862 and numbered 1993.51.2. The left side of that accession is 1993, a credit year. The sibling test must compare 1862, not 1993. Otherwise a query about the nineteenth century loses to a query about the cataloguing decade, and the 1714 sibling can never win on its own year. The combination takes object year from the title segment or the date line and refuses the accession prefix, then runs the sibling equality on what remains.

Math: sibling_year = object year after credit-segment and accession-prefix removal. Equality on that value. 1993 is not a candidate value. Earlier year may win, as in S10-049, only on sibling_year.

Combined: sibling year equality + credit-segment acquisition year.

Searches: Crossref GET /works?query=acquisition year sibling object date retrieval ranking&rows=3 HTTP 200, total-results 1311065 (relevance pool, not a boolean AND). Top title: Scalable object-class retrieval with approximate and top-k ranking DOI 10.1109/iccv.2011.6126556. The top title does not join the two mechanisms named in combined_mechanisms. no prior combination found in searches of these results.

Tag: RECALL/INDEX. Slices: EO+OD. Record: https://collectionapi.metmuseum.org/public/collection/v1/objects/44710

