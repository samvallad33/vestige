# B4 general memory seats

Frozen before any per-question outcome was inspected. The target metrics
(LoCoMo single-hop Turn@1, LME Turn@1) only named where to look. The rules
below are not conditioned on a dataset, a conversation id, a category label,
a speaker name, or a count tuned to flip a question.

## Split (registered before scoring)

Design half: the question's ordinal inside its source conversation is even
(0-based). Confirmation half: odd ordinals.

- LoCoMo: the `qi` in `sample:qi`.
- LME_S: the item index in `longmemeval_s_cleaned.json` (one question per item).
- BEAM-100K: the ordinal of the question inside that conversation, in the
  order `tools/c2_bench.py` walks `probing_questions.json`.

A half is not a held-out set if its outcomes were already inspected. Neither
half had been inspected when these rules were written.

## Rejection policy (registered before scoring)

The three seats are independent. A seat is rejected if any gate question it
fired on loses a hit flag on any benchmark, including when another seat also
fired on that question. No threshold, witness, or constant is edited after
the first scored comparison. If one or more seats are clean, those seats are
re-benched once, alone. A new loss on that run rejects them. There is no
third pass.

A kept seat must show 0 losses on LME_S, LoCoMo, and BEAM-100K, and gains on
at least two benchmarks, or gains on one while firing on the others with no
losses.

## Seat 1 — stated preference outranks the prompt

Principle: the turn that states a preference is the memory; the question that
asked for it is the prompt.

A query content token is a preference object when the ingest index has
already stored it as a role-object or as the object of a personal constraint.
Speaker-name tokens and the `user` / `assistant` role labels are not objects.
No preference-word list is consulted.

Slot 1 is a question with positive scene score. The next same-session turn
takes slot 1 when it is the utterance that stated such an object (its
role-object frame or its personal-constraint frame, not a later mention) and
it already sits in the top 5. If the next turn does not, the previous turn
may. A question neighbor never does. In a named-speaker dialogue the head
speaker is not the person the query names and the neighbor is, and a
conjoined name does not fire. In a user/assistant dialogue the neighbor is
the user. Temporal questions, summary requests, and WH-person questions stay
put. Top-5 membership does not change. `rsb_seat_behind` is not touched.

## Seat 2 — internal evidence precedes borrowed evidence

Principle: antaranga before bahiranga. A head whose scene score is mostly
carried in from neighbors yields to a turn that stands on its own words.

The shipped question-head cell is unchanged. A declarative head uses the same
gates: external residual strictly above own score, challenger not a question,
challenger own score strictly above the head and a strict majority of its own
scene score, challenger session sightline (`s_score + bridge_boost`) strictly
higher, unique own score. Added witness, declarative heads only: the
challenger attests a query content term the head lacks, and the head's
content term ids do not strictly cover the challenger's. Top-5 membership
does not change. No new ratio. `rsb_seat_behind` is not touched.

## Seat 3 — a name list yields to the record that contains it

Principle: a short list of title-case entities is an index card. The longer
same-session record that contains those entities is the memory.

Slot 1's ingest entity-id set is a nonempty strict subset of another top-5
turn in the same session, slot 1 has no digit form, and the longer turn is
not a question and attests every query content term slot 1 attests. A
user/assistant session opener stays, because that turn is the start of the
episode rather than a list stub. Two maximal supersets are not one record,
so the seat stays. Top-5 membership does not change. `rsb_seat_behind` is
not touched.

## What changed before this file

No ranker code had been edited before these three seats. Datasets were
fetched to the hashes below and BEAM 100K was unpacked into the harness
layout. No dataset-conditioned branch existed in the seats. The seats were
measured once and reverted. They are not in this commit.

## Datasets

- `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- BEAM 100K parquet sha256 `c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace`
- assembled `benchmarks/BEAM/chats/100K` json tree sha256 `910c4da4660e9384ca98d33378249712468bfcd328c238e27e1028fee8bc4ec2`

## Floor (parent `cd65421a`, before the seats)

LME_S n=470 turn @1/@5/@10/all@5 = 360/428/441/323. Session @1/@5/@10 = 430/463/469.
Turn funnel corpus/pool@50/pre_rank@5/post_kemeny@5/@1 = 470/464/406/415/360.

LoCoMo n=1536 turn @1/@5/@10 = 816/1130/1236. Session @1/@5 = 1137/1422.
Turn funnel = 1535/1424/1039/1100/816.

BEAM n=355 turn @1/@5/@10/all@5/all@10 = 164/241/272/125/146. Session @1/@5/@10 = 133/249/275.
Turn funnel = 355/327/229/234/164.

## After (one shot, both halves, no edit)

Aggregate turn @1: LME 360 (unchanged), LoCoMo 800 (−16), BEAM 164 (unchanged).
Every other aggregate turn and session cell was unchanged. Gate diff:
2361 questions, 31 loss flags, 15 gain flags, all turn @1.

Gate fires: `pref_object` 0/0/0. `name_list` LME 7, LoCoMo 52, BEAM 2.
`step36x` LME 0, LoCoMo 19, BEAM 4.

Design half (even): 19 losses, 7 gains. Confirmation half (odd): 12 losses,
8 gains. Both halves lose. Neither half was inspected before the seats were
written, so neither is being called held out after the fact.

### Gate losses

- LME_S `66f24dbb` turn @1 single-session-user (even, `name_list`)
- LoCoMo even turn @1: `conv-26:24` cat 1, `conv-26:88` cat 4, `conv-26:144` cat 4, `conv-26:146` cat 4, `conv-41:94` cat 4, `conv-41:112` cat 4, `conv-41:118` cat 4, `conv-41:134` cat 4, `conv-41:150` cat 4, `conv-42:12` cat 3, `conv-42:114` cat 4, `conv-42:150` cat 4, `conv-43:148` cat 4, `conv-43:158` cat 4, `conv-44:92` cat 4, `conv-47:148` cat 4, `conv-48:180` cat 4, `conv-50:134` cat 4
- LoCoMo odd turn @1: `conv-26:99` cat 4, `conv-30:59` cat 4, `conv-41:67` cat 4, `conv-42:99` cat 4, `conv-42:131` cat 4, `conv-42:181` cat 4, `conv-43:99` cat 4, `conv-43:121` cat 4, `conv-44:3` cat 1, `conv-44:83` cat 4, `conv-44:93` cat 4, `conv-47:117` cat 4

`name_list` sat on 27 of those loss flags. `step36x` sat on 4 (`conv-41:150`, `conv-42:12`, `conv-42:181`, `conv-44:3`).

### Gate gains

- LME_S `618f13b2` turn @1 knowledge-update (odd)
- LoCoMo even turn @1: `conv-41:12` cat 1, `conv-41:40` cat 1, `conv-41:120` cat 4, `conv-42:76` cat 1, `conv-43:156` cat 4, `conv-48:122` cat 4, `conv-49:148` cat 4
- LoCoMo odd turn @1: `conv-30:41` cat 4, `conv-41:103` cat 4, `conv-42:65` cat 2, `conv-42:73` cat 3, `conv-43:119` cat 4, `conv-47:121` cat 4, `conv-50:9` cat 2

## Verdict

Rejected. Not committed. No threshold was changed after the scored run.
`pref_object` never fired, so there was no clean seat to re-bench.
`name_list` and `step36x` each lost gate questions on the half they were
not fitted to, and on the even half as well.

Gates on the measured tree, before revert: `cargo test -p vestige-core`
589 passed, exit 0. `cargo clippy -p vestige-core --all-targets -- -D warnings`
exit 0. `cargo check -p vestige-mcp` exit 0. `tools/c2_hashcheck.py` was not
run; the seats were not kept.

No condition on dataset name, conversation id, category, speaker name, or a
tuned count was added. `rsb_seat_behind`, `benchmarks/real_engine/`,
`HYPERNYM_LATTICE`, and `rungs.rs` were not touched. Ranking did not read
turn bodies. No new floating-point scaler was introduced.
