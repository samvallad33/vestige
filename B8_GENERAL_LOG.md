# Lane B8 general-memory receipt

Parent: `cd65421a6bf94424e5558639126b935980e01629` on `perf/step2-multihop-overwrite`.
Branch: `perf/bench-b8-lme-turn1`. No ranker change is in this commit.
Author of the measurement: Sam Valladares <samvalladares21@gmail.com>.

`uploads/GENERAL_ONLY.txt` overrides the benchmark-lane framing. A rule is a general memory mechanism or it is not committed. The three mechanisms below were frozen before their own measurements. None was edited after a loss. None is in the tree.

## What changed

Nothing in the ranker. `crates/vestige-core` matches the parent. Three mechanisms were compiled, unit-tested, and measured, then deleted:

1. Originalia / estreat. The session's first digit-run multiset takes slot 1 when the seated head is a later same-session turn with that same multiset. Skips an existing latest-value intent (`current_state`) and a text-level preference question. Top-5 membership unchanged.
2. Foreign midcap over an echo. The unique top-5 turn that holds every query content term at or below the lower central turn-df, and that carries an entity absent from the query and from the speaker roster, takes slot 1. Refuses when slot 1 already has such an entity.
3. Marginal note. After natural turn document frequency is fixed, each turn posts the two rarest content tokens (df 1 through 3) of its previous neighbour and of its next neighbour, once each, at unit count, without changing length or df, and without writing them into `content`.

The one-word "see" pointer was not built. It is a lexical cue, which the correction forbids.

No condition on dataset name, file format, question id, benchmark category, speaker name, session count, or turn count. No threshold was moved after a result. Role labels were not added as a gate.

## Datasets

Assembled the way `tools/c2_bench.py` ingests them.

- LongMemEval_S `benchmarks/data/longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- LoCoMo-10 `benchmarks/data/locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- BEAM 100K from `mohammadtavakoli78/BEAM` `chats/100K` (20 conversations, 355 scored). Tree sha256 `cf1d9f4ae5ff0e802cca354a0b9d8fd8ee83565e176847717725918b54dc2722` over 316 files.

## Unmodified floor

`retrieval_bench` release, funnel dump on, before any of the three edits. Aggregates match the Step 2 floor.

Field meanings, checked in `retrieval.rs` and `tools/c2_bench.py`: `corpus` is a gold id this harness ingested; `pool50` is `scene_pool50` immediately before submodular selection; `pre_rank5` is the turn list cloned immediately before `apply_step11_kemeny_wh_slot_voter`; `post_kemeny5` is the list cloned immediately after `rank1_gatekeeper_keep_rsb_head`; `@1` is `final_ids[0]`. Session `pre_rank5` and session `post_kemeny5` are `final_session_ids[:5]`.

LME_S n=470. Turn any-gold corpus/pool@50/pre_rank@5/post_kemeny@5/@1 = 470/464/406/415/360. Session = 470/470/463/463/430. Turn @1/@5/@10/All@5 = 360/428/441/323. Session @1/@5/@10/All@5/All@10 = 430/463/469/422/446.

LoCoMo n=1536. Turn funnel 1535/1424/1039/1100/816. Session funnel 1536/1536/1422/1422/1137. Turn @1/@5/@10 = 816/1130/1236. Session @1/@5 = 1137/1422.

BEAM n=355. Turn funnel 355/327/229/234/164. Session funnel 355/341/249/249/133. Turn @1/@5/@10/All@5/All@10 = 164/241/272/125/146. Session @1/@5/@10 = 133/249/275.

## Split

Design half = even file ordinal in harness order (0, 2, 4, ...), gate rows only in the totals. The odd half was not opened for any of the three treatments. It is not held-out: before the mechanisms were frozen, a 3-question LME smoke (ordinals 0, 1, and 2) scored 3/3 aggregate, and a gold-structure census of the whole LME file was partially read (type counts, `current_state` counts by type, and question id `71315a70`). Those facts were not used to add a witness. Every rule below was rejected on the even half alone.

## Rejected attempts

### Originalia, even half

Principle: the original statement of a digit multiset outranks a later copy in the same session, unless the question asks for the latest value or is a preference question.

- LME_S even gate n=235. Fires 3. Turn @1 184→183. Gains 1 (`2788b940` multi-session, turn @1). Losses 2 (`1b9b7252`, `e48988bc`, both single-session-assistant, turn @1). Other turn and session metrics unchanged.
- LoCoMo even gate n=765. Fires 0. Losses 0.
- BEAM even gate n=178. Fires 0. Losses 0.

Rejected. Losses, and it fired on one benchmark only. Not retuned.

### Foreign midcap, even half

Principle: the unique top-5 turn that covers the query's ordinary content terms and names an entity the query and the speaker roster do not name takes slot 1. An incumbent that already names such an entity stays.

- LME_S even. Fires 0. Losses 0. Turn @1 stays 184.
- LoCoMo even. Fires on 27 gate rows (35 rows including non-gate). Turn @1 415→404. Gains 2, both turn @1 (`conv-42:141`, `conv-50:146`). Losses 13, all turn @1: `conv-26:0`, `conv-26:24`, `conv-26:102`, `conv-41:54`, `conv-41:112`, `conv-41:134`, `conv-42:5`, `conv-43:99`, `conv-43:121`, `conv-44:71`, `conv-44:83`, `conv-49:42`, `conv-50:76`. Session metrics unchanged.
- BEAM even. Fires 0. Losses 0. Turn @1 stays 80.

Rejected. Losses, and the only fires were on one benchmark. Not retuned.

### Marginal note, even half

Principle: a turn carries a pointer to the rare words of the turns beside it, so those words can retrieve the turn without becoming its own statement.

- LME_S even. Fires on 34 gate rows. Turn @1 184→183, @5 216→214, All@5 154→152. Gain 1 (`5025383b` multi-session, All@5). Losses: `d3ab962e` All@5; `7161e7e2` @1/@5/All@5; `ac031881` @5/All@5.
- LoCoMo even. Fires on 334 gate rows. Turn @1 415→411, @5 572→567, @10 stays 623. Gains 14 hit-flags, losses 23. Session metrics unchanged.
- BEAM even. Fires on 18 gate rows. Turn All@5 57→56. Loss `10:temporal_reasoning:1` All@5 and `11:instruction_following:1` turn @1. Gain `11:multi_session_reasoning:1` turn @1. Turn @1 stays 80.

Rejected. Losses on every benchmark. The df bound was left at 1..=3. Not retuned.

## Gates

No winning tree, so the release gates were not claimed. Unit tests of each temporary module passed before that module was deleted (originalia 9, foreign midcap 5, marginal note 3). `cargo test -p vestige-core`, clippy, `cargo check -p vestige-mcp`, and `tools/c2_hashcheck.py` were not run as a ship gate because the ranker was restored to the parent.

## Still open

LME_S Turn @1 remains 360/470. LoCoMo Turn @1 remains 816/1536. BEAM Turn @1 remains 164/355. The three frozen mechanisms do not move those numbers without losses.
