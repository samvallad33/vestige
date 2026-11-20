//! Stage 08: C2 seating lane, `rsb_seat_behind`, Rank-1 gatekeeper, Steps 31–38, Pool C/B, peaks, and RST nucleus.

use crate::retrieval::*;
use crate::retrieval::context::RankCtx;

pub(in crate::retrieval) fn run(ctx: &mut RankCtx<'_>) {
    let ix = ctx.ix;
    let q = ctx.q;
    let k = ctx.k;
    let plan = &ctx.plan;
    let _it = &plan.intent;
    let eff_it = &ctx.eff_it;
    let standing_rules = &ctx.standing_rules;
    let s_score = &ctx.s_score;
    let bridge_boost = &ctx.bridge_boost;
    let t_own = &ctx.t_own;
    let t_scene = &ctx.t_scene;
    let t_exact_mask = &ctx.t_exact_mask;
    let turn_ts = &ctx.turn_ts;
    let mut final_turn_indices = std::mem::take(&mut ctx.final_turn_indices);

    // C2 algos lane: slot-1 seat rules, all before RSB (which then seats behind the head).
    if !eff_it.is_event_sequence {
        apply_c2_bullae_numeral_seat(ix, &mut final_turn_indices, &t_scene);
        apply_c2_coverage_margin_seat(ix, &plan, &mut final_turn_indices);
        apply_c2_testimony_seat(ix, &plan, &mut final_turn_indices, &t_scene);
        apply_c2_cross_witness_seat(ix, &plan, &mut final_turn_indices, &t_scene);
        apply_c2_mimamsa_seat(ix, &plan, &mut final_turn_indices, &t_scene);
        apply_c2_dichotomous_key(ix, &plan, &mut final_turn_indices, &t_scene);
        // Pool-fill reallocation: user source with a strict group superset takes a
        // same-session assistant's top-5 seat. Skipped on assistant-attribution so
        // the assistant's own words stay in place.
        if !eff_it.assistant_attribution {
            apply_c2_source_superset_realloc(ix, &plan, &mut final_turn_indices, &t_scene);
            apply_c2_unique_group_realloc(ix, &plan, &mut final_turn_indices, &t_scene);
            apply_c2_subsequent_answer_seat(ix, &plan, &mut final_turn_indices, &t_scene);
            // Next-user elaboration: mid-session declarative head yields to
            // the next same-session declarative user already in the top 5
            // when that follow-up is at least as on-scene. Skipped on
            // preference so the rule stays independent of the
            // preference-source seat.
            // Text-level preference only (PREF_QUERY_CUES), not the
            // constraint-object upgrade on plan.intent.preference, so
            // #12–#18 stay independent of #11 without treating a factual
            // question that mentions a constraint object as preference.
            if !c2_text_preference(&q.text) {
                apply_c2_elaboration_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Cross-session opener: mid-session user head yields to a
                // user session-opener already in the top 5 from another
                // session. Skipped on preference and on summary requests.
                apply_c2_cross_opener_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Same-session opener: mid-session declarative head yields
                // to the same session's user opener already in the top 5.
                // Skipped on preference and on summary requests.
                apply_c2_same_opener_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Queried-speaker reply: in a named-speaker dialogue, an
                // unmentioned speaker's question yields to the next
                // same-session turn by a speaker the query names, already
                // in the top 5. Skipped on preference.
                apply_c2_queried_speaker_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Queried-speaker previous: in a named-speaker dialogue, an
                // unmentioned speaker's declarative comment yields to the
                // previous same-session statement by a speaker the query
                // names, already in the top 5. Skipped on preference.
                apply_c2_queried_speaker_prev_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Queried-speaker question-head previous: in a
                // named-speaker dialogue, an unmentioned speaker's
                // question yields to the previous same-session turn
                // by a speaker the query names, already in the top 5.
                // Skipped on preference.
                apply_c2_queried_speaker_qhead_prev_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
                // Queried-speaker T@5 entry: a declarative turn by a
                // speaker the query names, already in official k outside
                // the top 5, swaps with the last unmentioned named
                // speaker in the top 5. Membership-preserving. Skipped
                // on preference.
                apply_c2_queried_speaker_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
            }
        }
        // Preference source: ReplySource is skipped on assistant_attribution,
        // but "can you recommend..." is tagged that way by a present-tense cue.
        // Membership-preserving; runs even when assistant_attribution is set.
        apply_c2_preference_source_seat(ix, &plan, &q.text, &mut final_turn_indices);
    }
    // RSB: the retrieved turn is the symptom. Seat its structural cause
    // immediately after it. The cause can share no words with the query.
    // Never in front of the symptom. A full window does not grow.
    if !eff_it.is_event_sequence {
        rsb_seat_behind(ix, &mut final_turn_indices, k);
    }
    // Rank-1 gatekeeper v2. Slot 1 stays the RSB v4 head. Slots 2..k are the
    // same Kemeny-Young order as PR 710, computed on that window with the head
    // removed. No new constant: the frozen slot is position 0 of this window.
    rank1_gatekeeper_keep_rsb_head(ix, &q.text, &mut final_turn_indices, &t_scene);
    // Queried-speaker T@5 entry, after the gatekeeper. The pre-RSB pass
    // (#17) is often undone when the gatekeeper Kemeny-reorders the tail
    // (slots 2..k). Re-applying the same membership-preserving swap here
    // is a post-gatekeeper ranking change, so the entry survives the
    // tail reorder by construction. Later post-gatekeeper C2 seats
    // may still run after this pass.
    // Same preference / event-sequence / assistant-attribution cuts.
    // Does not touch rsb_seat_behind and adds no WH-slot ballot.
    if !eff_it.is_event_sequence && !eff_it.assistant_attribution && !c2_text_preference(&q.text) {
        apply_c2_queried_speaker_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Same-session user T@5 entry: a user opener or later turn from
        // a session already in the top 5, already in official k, swaps
        // with the last assistant in the top 5. Post-gatekeeper so the
        // entry survives. Skipped on preference.
        apply_c2_same_session_user_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Cross-session user-opener T@5 entry: a user opener from a
        // session not yet in the top 5, already in official k, swaps
        // with the last non-unique assistant that is not the head's
        // next turn. Post-gatekeeper, after #20, so the entry survives.
        // Skipped on preference.
        apply_c2_cross_session_user_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Queried-speaker first-in-session seat: an unmentioned speaker
        // at slot 1 yields to a named speaker's first utterance in its
        // session, already in the top 5. Post-gatekeeper, after #21,
        // so it sees the official post-gatekeeper list (a pre-RSB
        // pass would reseat a tail occupant that the gatekeeper later
        // leaves outside the top 5). Later post-gatekeeper C2 seats
        // may still run after this pass. Skipped on preference.
        apply_c2_queried_speaker_first_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Queried-speaker first-in-session T@5 entry: a named speaker's
        // first utterance in its session, already in official k
        // outside the top 5, swaps with the last unmentioned named
        // speaker. Declaratives have no scene floor; questions reuse
        // C2_XWITNESS_RATIO. Post-gatekeeper, after the first #22 pass.
        // Skipped on preference.
        apply_c2_queried_speaker_first_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Queried-speaker first-in-session seat, again, after #23. A
        // first-in-session turn that #23 just admitted into the top 5
        // can now take slot 1; the earlier #22 pass could not see it
        // there yet. Later post-gatekeeper C2 seats may still run after
        // this pass. Skipped on preference.
        apply_c2_queried_speaker_first_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    }
    // Subsequent-answer seat, again, after the gatekeeper. Pre-RSB #10
    // refuses when the next user is still outside the top 5; RSB can
    // then seat that user into the window, and a post-GK pass can take
    // slot 1. Later post-gatekeeper C2 seats may still run after this
    // pass. Same cuts as #10: event-sequence and assistant-attribution
    // at the call site; the function still skips temporal and
    // named-speaker sessions. Does not skip preference (same as #10).
    // Does not run the gatekeeper again and adds no WH-slot ballot.
    if !eff_it.is_event_sequence && !eff_it.assistant_attribution {
        apply_c2_subsequent_answer_seat(ix, &plan, &mut final_turn_indices, &t_scene);
    }
    // Cross-session opener-pair seat: a user opener at slot 1 yields to
    // another session's user opener already in the top 5 whose
    // immediately following assistant reply is also already in the
    // top 5, when that opener is at least as on-scene. Post-gatekeeper,
    // after #26. Later post-gatekeeper C2 seats may still run after
    // this pass. Skipped on preference. Does not run the gatekeeper
    // again and adds no WH-slot ballot.
    if !eff_it.is_event_sequence && !eff_it.assistant_attribution && !c2_text_preference(&q.text) {
        apply_c2_cross_opener_pair_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Answered-opener declarative seat: a user question opener at
        // slot 1 whose assistant reply is already in the top 5 yields
        // to another session's declarative user opener already in the
        // top 5, when that opener is at least as on-scene. After #27.
        // Later post-gatekeeper C2 seats may still run after this pass.
        // Skipped on preference.
        apply_c2_answered_opener_decl_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Solo earlier-opener seat: a lone user opener at slot 1 (no
        // other turn from that session in the top 5) yields to an
        // earlier user opener from a different session already in the
        // top 5, when that opener is at least as on-scene. After #28.
        // Later post-gatekeeper C2 seats may still run after this pass.
        // Skipped on preference.
        apply_c2_solo_earlier_opener_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Pool opener T@10 entry: a user opener in the scene-ranked
        // pool, not already in the official k, takes the official-k
        // seat of the last non-unique tail assistant whose session
        // still has a user in the official k, when that opener is at
        // least as on-scene. After #29. Later post-gatekeeper C2 seats
        // may still run after this pass. The top 5 is unchanged.
        // Skipped on preference.
        apply_c2_pool_opener_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Same-session opener T@5 entry: a user opener already in
        // official k, from a session already represented in the top 5
        // by an assistant, swaps with the last same-session assistant
        // in the top 5. After #30. Later post-gatekeeper C2 seats may
        // still run after this pass. Skipped on preference.
        apply_c2_same_session_opener_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Pool mentioned-speaker T@10 entry: a mentioned-speaker turn
        // in the scene-ranked pool, not already in official k, takes
        // the official-k seat of the last unmentioned named speaker in
        // the tail, when that turn is at least as on-scene. After #31.
        // Later post-gatekeeper C2 seats may still run after this pass.
        // The top 5 is unchanged. Skipped on preference. Does not fire
        // on user/assistant chats.
        apply_c2_pool_mentioned_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Queried-speaker T@5 entry again after #32, so a mentioned
        // declarative that the pool T@10 entry just seated into
        // official k can still enter the top 5. After #32, last
        // ranking change. Skipped on preference.
        apply_c2_queried_speaker_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Late opener T@5 entry: a user opener already in official k,
        // from a session already in the top 5, swaps with the last
        // top-5 assistant of any session. After #33, last ranking
        // change, so it sees openers the pool T@10 entry (#30) seated.
        // Skipped on preference.
        apply_c2_late_opener_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Head-session opener T@5 entry: the opener of slot 1's session,
        // already in official k, swaps with the least on-scene top-5
        // user from another session when at least as on-scene. After
        // #34, last ranking change. Skipped on preference.
        apply_c2_head_session_opener_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Session substitution T@5 entry: the most on-scene official-k
        // user from a session outside the top 5 swaps with the weakest
        // top-5 user when that user is a declarative at most
        // C2_XWITNESS_RATIO of its scene. After #35, last ranking
        // change. Skipped on preference.
        apply_c2_session_substitution_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Pool session substitution T@10 entry: the most on-scene pool
        // user from a session with no official-k seat replaces the
        // weakest official-k tail turn when that turn is a declarative
        // at most C2_XWITNESS_RATIO of its scene. The top 5 is
        // unchanged. After #36, last ranking change. Skipped on
        // preference.
        apply_c2_pool_session_substitution_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Session substitution T@5 entry again after #37, so a pool user
        // that #37 just seated into official k can still enter the top
        // 5. After #37, last ranking change. Skipped on preference.
        apply_c2_session_substitution_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Mentioned-declarative session-mate T@5 entry: in a named-speaker
        // dialogue, a mentioned speaker's declarative from a top-5 session
        // swaps with the weakest top-5 declarative (at most
        // C2_XWITNESS_RATIO of its scene, never from an earlier session).
        // After #38, last ranking change. Skipped on preference.
        apply_c2_mentioned_session_mate_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Episode-run edge T@10 entry: the pool turn just before / after two
        // back-to-back official-k turns of one session takes the tail seat of
        // the least on-scene lone-session turn (at least as on-scene, never an
        // earlier session). After #39, last ranking change.
        apply_c2_episode_run_edge_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Episode-run edge T@5 entry: the edge of two back-to-back top-5 turns
        // takes the slot of the least on-scene lone-session, non-opener top-5
        // turn (at least as on-scene, never an earlier session). After #40,
        // last ranking change.
        apply_c2_episode_run_edge_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Neighbor-pair T@5 entry: in named dialogues, the official-k tail
        // neighbor of a top-5 turn swaps with the least on-scene lone-session,
        // non-opener top-5 turn (at least as on-scene, never an earlier
        // session). After #41, last ranking change.
        apply_c2_neighbor_pair_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    }
    // Assistant-reply T@5 entry, for assistant-attribution questions only
    // (outside the C2 skip above): the official-k assistant reply to a top-5
    // user turn swaps with the least on-scene other top-5 user turn (at most
    // as on-scene, never from an earlier session, S@5 kept). Last ranking
    // change.
    if !eff_it.is_event_sequence && eff_it.assistant_attribution {
        apply_c2_assistant_reply_t5_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Unseated-session opener T@10 entry (assistant-attribution only):
        // the most on-scene declarative opener of a session with no official-k
        // seat takes the tail seat of the least on-scene lone-session turn
        // (at least as on-scene, never an earlier session). After #43.
        apply_c2_attribution_unseated_opener_t10_entry(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
        // Exchequer split tally: for assistant-attribution questions, an
        // assistant turn whose own groups and the user turn it answers are
        // complementary halves takes slot 1 when that pair covers the
        // incumbent. After #44. Does not touch rsb_seat_behind.
        apply_c2_step31_exchequer_split(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    }
    // Step 32 trial balance. After every other C2 seat. A question at slot 1
    // that misses the rarest surface lexeme yields to a top-5 opener from a
    // session that is not later, when that opener posts the rare lexeme and
    // every surface lexeme slot 1 already posts. Otherwise an echo at slot 1
    // yields to the declarative turn it answers, when that turn posts the
    // above-median surface lexemes and the pair posts every surface lexeme
    // the echo's own pair does not. Does not touch rsb_seat_behind and does
    // not rewrite the session list.
    apply_c2_step32_trial_balance(ix, &plan, &mut final_turn_indices, &t_scene);
    // Step 33 triangulation. A top-5 turn that is strictly best on its own
    // lexical score and on its session score takes slot 1 when it also
    // strictly covers the head's query terms. Named-speaker dialogues only.
    // After the gatekeeper, Step 31 and Step 32. Does not touch rsb_seat_behind.
    apply_c2_step33_triangulation(
        ix,
        &plan,
        &mut final_turn_indices,
        &t_own,
        &s_score,
        &bridge_boost,
    );
    // Step 34 Bertillonage. After trial balance and triangulation. Slot 1 below
    // the query's kind-count yields to one admissible same-session top-5 card.
    // Does not touch rsb_seat_behind and does not rewrite the session list.
    apply_c2_step34_bertillonage(ix, &q.text, &mut final_turn_indices, &t_scene);
    // Step 35 Mill's method of agreement. After the gatekeeper and Steps 31–34.
    // The unique session common to every surface-anchor hit set lends slot 1
    // to its unique top-5 instance when the head posts no anchor, or to its
    // unique same-speaker turn that posts every anchor when the head posts
    // only some. A head already inside that session stays. Does not touch
    // rsb_seat_behind.
    apply_c2_step35_mill_agreement(ix, &plan, &mut final_turn_indices, &t_scene);
    // Step 36 Antaranga-bahiranga. After Bertillonage. A question at slot 1
    // whose scene score is mostly external yields to a top-5 declarative
    // turn held by its own lexical score. Does not touch rsb_seat_behind
    // and does not rewrite the session list.
    apply_c2_step36_antaranga(
        ix,
        &q.text,
        &mut final_turn_indices,
        &t_own,
        &t_scene,
        &s_score,
        &bridge_boost,
    );
    // Step 37 Kalevala parallelism. After the gatekeeper and Steps 31–35.
    // Adjacent same-speaker restatements are one line. The following line
    // takes slot 1 when its content query cues strictly cover the head and
    // it already sits in the top 5. Does not touch rsb_seat_behind.
    apply_c2_step37_kalevala(ix, &q.text, &mut final_turn_indices);
    // Step 38 Hermeneutic circle. After Antaranga. A question at slot 1 yields
    // to the one top-5 declarative that posts the same content anchors, whose
    // session covers the anchors that part does not, when the incumbent
    // session does not and the declarative's own score is strictly stronger.
    // Does not touch rsb_seat_behind and does not rewrite the session list.
    apply_c2_step38_hermeneutic(ix, &q.text, &mut final_turn_indices, &t_own);
    // Pool C session-key opener. After Bertillonage. A same-session assistant
    // elaboration in slots 2–5 yields to that session's user opener when the
    // top 5 already spans two sessions and the opener is outside it. Slot 1
    // stays. Does not touch rsb_seat_behind.
    apply_c2_pool_c_session_opener_seat(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    // Absent session key. After the same-session yield. A user opener whose
    // session is not in the top 5 takes a redundant assistant from a
    // non-dominant session. The dominant session stays. Slot 1 stays.
    apply_c2_pool_c_absent_session_key(ix, &plan, &q.text, &mut final_turn_indices, &t_scene);
    // Pool-B standing-rule seat. After Bertillonage, so a user question left
    // at slot 1 can still yield to the zero-lexical standing opener the
    // stratified channel already selected. Top-5 membership is unchanged.
    // Does not touch rsb_seat_behind and does not rewrite the session list.
    apply_c2_pool_b_standing_seat(
        ix,
        &plan,
        &q.text,
        &standing_rules,
        &mut final_turn_indices,
        &t_own,
    );
    // Buried lexical peak. After Bertillonage. A unique named-speaker
    // declarative whose own score beats the entire scene pool takes the
    // lowest-own question seat inside slots 1..4. Slot 0 and the session
    // list stay. Does not touch rsb_seat_behind.
    apply_c2_buried_lexical_peak(
        ix,
        &mut final_turn_indices,
        &t_scene,
        &t_own,
        &turn_ts,
        k,
    );
    // Anaphoric tail. After the buried peak. The strongest anaphoric
    // declarative in ranks 5..9 swaps with the lowest-own question in slots
    // 1..4 when its own score beats the whole top 5. Slot 0 and the session
    // list stay. Does not touch rsb_seat_behind.
    apply_c2_anaphoric_tail(ix, &mut final_turn_indices, &t_own);
    // Unseated session peak. After the anaphoric tail. The unique named-speaker
    // declarative outside the returned list whose own score beats every
    // returned turn takes the lowest-own question seat in slots 1..4, when its
    // session is absent from the top 5. Slot 0 and the session list stay.
    // Does not touch rsb_seat_behind.
    apply_c2_unseated_session_peak(ix, &mut final_turn_indices, &t_own, k);
    // #810: Stede RST Nucleus/Satellite preference at Rank 1 (0..5 membership unchanged)
    crate::retrieval::seating::apply_rst_nucleus_preference(ix, &mut final_turn_indices, &t_scene, &t_exact_mask);

    ctx.final_turn_indices = final_turn_indices;
}
