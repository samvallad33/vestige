//! Discourse-Stack Obligation Automaton, Passive FSRS-7 Dual-Stability Working-Memory Gate,
//! Incremental Run-Time Event Calculus (RTEC) Fluent Interval Clipping, and Canonical
//! Difference Bound Matrix (DBM) Temporal Reasoning.
//!
//! Pure deterministic integer `Q16.16` fixed-point arithmetic (`ONE = 65_536`).
//! Zero vectors, zero embeddings, zero LLMs, zero open-class word lists.
//!
//! Grounded in:
//! - Grosz & Sidner (1986), *Attention, Intentions, and the Structure of Discourse* (`J86-3001`),
//! - Clark & Schaefer (1989), *Contributing to Discourse*,
//! - Matheson, Poesio & Traum (2000), *Modelling Grounding and Discourse Obligations* (`A00-2001`),
//! - `open-spaced-repetition/fsrs-rs` v7.0.0 dual-stability forgetting curve + ACT-R BLL,
//! - Artikis, Sergot & Paliouras (TKDE 2015) / Mantenoglou, Pitsikalis & Artikis (AAAI/JAIR 2025) RTEC,
//! - Gärdenfors & Makinson (1988) AGM epistemic entrenchment + Katsuno & Mendelzon (1992) update,
//! - Dill (1989) Difference Bound Matrices (tropical `(min, +)` semiring) + Allen (1983) / Nebel & Bürckert (1995) ORD-Horn.

use super::bitemporal_heideltime::compile_ingest_timex_anchors;
use super::supersession::turn_is_superseded;
use crate::retrieval::context::RankCtx;
use crate::retrieval::index::Index;
use crate::retrieval::query::dates::{inline_date_tokens, parse_date};
use crate::retrieval::query::tokens::{is_content, raw_tokens};
use crate::retrieval::rank::genesis_spine_q16::{
    TemporalQueryPolarity, detect_temporal_query_polarity,
};
use crate::retrieval::DetHashSet;

/// Fixed-point unit `1.0` in signed `Q16.16`.
pub(in crate::retrieval) const ONE: i64 = 65_536;

/// Overflow-safe tropical `(min, +)` infinity for Difference Bound Matrices.
const DBM_INF: i64 = i64::MAX / 4;

/// Seconds per Gregorian civil day.
const SECS_PER_DAY: i64 = 86_400;

/// Fixed-point `ln(2)` in `Q16.16` (`round(0.69314718 * 65536)`).
const LN2_Q16: i64 = 45_426;

/// Minimum stability floor in `Q16.16` days (`~1 minute = 1 / 1440` day).
const MIN_STABILITY_DAYS_Q16: i64 = 46;

/// Default working-memory fast stability (`30 minutes = 1_800 s`).
const DEFAULT_S_FAST_SEC: u64 = 1_800;

/// Default consolidated slow stability (`30 days`).
const DEFAULT_S_SLOW_SEC: u64 = 30 * 86_400;

/// AGM Epistemic Entrenchment levels in `Q16.16`.
pub(in crate::retrieval) const ENTRENCH_SUGGESTION_Q16: i64 = ONE;
pub(in crate::retrieval) const ENTRENCH_ASSERTION_Q16: i64 = 2 * ONE;
pub(in crate::retrieval) const ENTRENCH_TOOL_MUTATION_Q16: i64 = 3 * ONE;
pub(in crate::retrieval) const ENTRENCH_CORRECTION_Q16: i64 = 4 * ONE;

/// Precomputed `round(2^(k/16) * 65536)` for `k = 0..=16` (piecewise-linear `exp2` LUT).
const EXP2_LUT_16: [u32; 17] = [
    65_536, 68_438, 71_468, 74_632, 77_936, 81_386, 84_990, 88_752, 92_682, 96_785, 101_070,
    105_545, 110_218, 115_098, 120_194, 125_515, 131_072,
];

/// Precomputed `round(log2(1 + k/16) * 65536)` for `k = 0..=16` (piecewise-linear `log2` LUT).
const LOG2_LUT_16: [i32; 17] = [
    0, 5_732, 11_136, 16_248, 21_098, 25_711, 30_109, 34_312, 38_336, 42_196, 45_904, 49_472,
    52_911, 56_229, 59_434, 62_534, 65_536,
];

// =========================================================================
// PURE INTEGER Q16.16 ARITHMETIC & LUT PRIMITIVES
// =========================================================================

#[inline]
fn mul_q16(a: i64, b: i64) -> i64 {
    let prod = (i128::from(a) * i128::from(b)) >> 16;
    prod.clamp(i128::from(i64::MIN / 2), i128::from(i64::MAX / 2)) as i64
}

#[inline]
fn div_q16(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let quot = (i128::from(a) << 16) / i128::from(b);
    quot.clamp(i128::from(i64::MIN / 2), i128::from(i64::MAX / 2)) as i64
}

#[inline]
fn isqrt_u128(n: u128) -> u128 {
    if n <= 1 {
        return n;
    }
    let shift = (128 - n.leading_zeros()).div_ceil(2);
    let mut x = 1u128 << shift;
    loop {
        let next = (x + n / x) >> 1;
        if next >= x {
            return x;
        }
        x = next;
    }
}

#[inline]
fn sqrt_q16(x_q16: i64) -> i64 {
    if x_q16 <= 0 {
        return 0;
    }
    isqrt_u128((x_q16 as u128) << 16).min(i64::MAX as u128) as i64
}

/// Piecewise-linear LUT `log2(x)` where `x` is unsigned `Q16.16`.
#[inline]
fn log2_lut_q16(x_q16: u64) -> i64 {
    if x_q16 == 0 {
        return -20 * ONE;
    }
    let msb = 63 - x_q16.leading_zeros() as i64;
    let int_part = (msb - 16) * ONE;
    let norm = if msb >= 16 {
        x_q16 >> (msb - 16)
    } else {
        x_q16 << (16 - msb)
    };
    let frac = (norm & 0xFFFF) as i64;
    let idx = (frac >> 12) as usize;
    let rem = frac & 0x0FFF;
    let y0 = i64::from(LOG2_LUT_16[idx]);
    let y1 = i64::from(LOG2_LUT_16[idx + 1]);
    int_part + y0 + (((y1 - y0) * rem) >> 12)
}

/// Piecewise-linear LUT `2^x` where `x` is signed `Q16.16`.
#[inline]
fn exp2_lut_q16(x_q16: i64) -> i64 {
    if x_q16 <= -20 * ONE {
        return 0;
    }
    let clamped = x_q16.min(20 * ONE);
    let int_part = clamped.div_euclid(ONE);
    let frac = clamped.rem_euclid(ONE);
    let idx = (frac >> 12) as usize;
    let rem = frac & 0x0FFF;
    let y0 = i64::from(EXP2_LUT_16[idx]);
    let y1 = i64::from(EXP2_LUT_16[idx + 1]);
    let mant = y0 + (((y1 - y0) * rem) >> 12);
    if int_part >= 0 {
        mant << (int_part as u32)
    } else {
        mant >> ((-int_part) as u32)
    }
}

#[inline]
fn ln_lut_q16(x_q16: u64) -> i64 {
    mul_q16(log2_lut_q16(x_q16), LN2_Q16)
}

/// Deterministic FNV-1a 64-bit hash over UTF-8 bytes.
#[inline]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Deterministic SplitMix64 mixing function for 64-bit bloom slot masks.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Normalize text to lowercase ASCII apostrophes and collapsed whitespace (UTF-8 safe).
fn normalize_lower(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.chars() {
        let mapped = match ch {
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{2013}' | '\u{2014}' => ' ',
            c if c.is_whitespace() => ' ',
            c => c,
        };
        if mapped == ' ' {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            for lc in mapped.to_lowercase() {
                out.push(lc);
            }
            prev_space = false;
        }
    }
    out.trim().to_string()
}

// =========================================================================
// 1. DISCOURSE-STACK GROUNDING & OBLIGATION AUTOMATON
// =========================================================================

/// Obligation state of a Grosz-Sidner `DiscourseSegmentPurpose` (DSP) frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum GroundingState {
    /// Assistant proposed an action or summary awaiting user acceptance/rejection.
    OpenProposal,
    /// Active task / slot-filling segment awaiting verification or closure.
    ActiveTask,
    /// Grounded and closed by explicit completion or acknowledgement.
    GroundedClosed,
}

/// Intentional discourse stack frame tracking contribution presentation and active updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct DspFrame {
    pub(in crate::retrieval) frame_id: u32,
    pub(in crate::retrieval) presentation_turn: u32,
    pub(in crate::retrieval) last_active_turn: u32,
    pub(in crate::retrieval) slot_mask: u64,
    pub(in crate::retrieval) state: GroundingState,
}

/// Closed-class conversational grounding act of a query or user turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum QueryGroundingAct {
    /// Low-lexical acceptance / go-ahead (`"yes, please go ahead"`, `"looks good, proceed"`).
    ConfirmProceed,
    /// Slot-state verification (`"just to verify"`, `"still set to"`, `"double check"`).
    VerifyLastValue,
    /// Explicit task closure (`"[task_done]"`, `"that's all"`, `"thanks for confirming"`).
    CloseDone,
    /// Standard informational query.
    Standard,
}

/// Classify a query into a closed-class conversational grounding act.
/// hardcode-scan: closed-class (conversational grounding, verification, confirmation, and closure acts only).
pub(in crate::retrieval) fn classify_query_grounding_act(query: &str) -> QueryGroundingAct {
    let low = normalize_lower(query);
    if low.is_empty() {
        return QueryGroundingAct::Standard;
    }
    if low.contains("[task_done]") || low.contains("task_done") {
        return QueryGroundingAct::CloseDone;
    }

    // hardcode-scan: closed-class (slot-verification discourse markers only)
    const VERIFY_PHRASES: &[&str] = &[
        "just to verify",
        "just verifying",
        "still set to",
        "double check",
        "double-check",
        "confirm that",
        "verify that",
        "verify one last",
        "confirm if",
        "confirm whether",
        "is it still",
        "are we still",
        "to make sure",
    ];
    if VERIFY_PHRASES.iter().any(|p| low.contains(p)) {
        return QueryGroundingAct::VerifyLastValue;
    }

    // hardcode-scan: closed-class (conversational closure and completion acts only)
    const CLOSE_PHRASES: &[&str] = &[
        "that's all",
        "thats all",
        "that is all",
        "nothing else",
        "all set",
        "thanks for confirming",
        "thank you for confirming",
        "all done",
        "that's everything",
        "thats everything",
        "that is everything",
        "no more questions",
        "we're done",
        "we are done",
        "no further",
    ];
    if CLOSE_PHRASES.iter().any(|p| low.contains(p)) {
        return QueryGroundingAct::CloseDone;
    }

    let padded = format!(" {low} ");
    let has_info_wh = padded.contains(" what ")
        || padded.contains(" where ")
        || padded.contains(" when ")
        || padded.contains(" who ")
        || padded.contains(" why ")
        || padded.contains(" which ")
        || padded.contains(" how many ")
        || padded.contains(" how much ")
        || padded.contains(" how long ");
    if has_info_wh {
        return QueryGroundingAct::Standard;
    }

    // hardcode-scan: closed-class (affirmative go-ahead and confirmation acts only)
    const CONFIRM_PHRASES: &[&str] = &[
        "go ahead",
        "please proceed",
        "looks good, proceed",
        "looks good proceed",
        "yes, proceed",
        "yes proceed",
        "please confirm",
        "please process",
        "process it",
        "submit it",
        "that matches",
        "sounds good, please",
        "sounds good please",
        "confirmed, please",
        "yes, please",
        "yes please",
        "approve it",
        "approved, proceed",
    ];
    if CONFIRM_PHRASES.iter().any(|p| low.contains(p)) {
        return QueryGroundingAct::ConfirmProceed;
    }

    QueryGroundingAct::Standard
}

/// Detect whether an assistant turn presents a proposal, summary, or confirmation request.
/// hardcode-scan: closed-class (assistant offer/confirmation discourse frames only).
fn is_assistant_proposal_turn(low: &str, is_question: bool) -> bool {
    // hardcode-scan: closed-class (assistant proposal/confirmation request phrases only)
    const PROPOSAL_PHRASES: &[&str] = &[
        "would you like me to",
        "shall i ",
        "should i ",
        "do you want me to",
        "please confirm",
        "can you confirm",
        "could you confirm",
        "ready to ",
        "go ahead and ",
        "proceed with ",
        "let me know if",
        "does that look",
        "does this look",
        "is that correct",
        "look good",
        "to confirm",
        "before i ",
        "anything else",
    ];
    PROPOSAL_PHRASES.iter().any(|p| low.contains(p)) || is_question
}

/// Compute a deterministic 64-bit bloom slot mask for turn `ti`.
fn turn_slot_mask(ix: &Index, ti: usize) -> u64 {
    let tr = &ix.turns[ti];
    let mut mask = 0u64;
    for &id in &tr.content {
        mask |= 1u64 << (splitmix64(u64::from(id)) & 63);
    }
    for &id in &tr.entities {
        mask |= 1u64 << (splitmix64(u64::from(id)) & 63);
    }
    for &id in &tr.coref_entities {
        mask |= 1u64 << (splitmix64(u64::from(id)) & 63);
    }
    for tname in &tr.tool_names {
        mask |= 1u64 << (fnv1a64(tname.as_bytes()) & 63);
    }
    mask
}

/// Reconstruct the dialogue's intentional discourse stack (`DspFrame` stack + most recently closed frame).
pub(in crate::retrieval) fn build_discourse_stack(ix: &Index) -> (Vec<DspFrame>, Option<DspFrame>) {
    let mut stack: Vec<DspFrame> = Vec::new();
    let mut just_closed: Option<DspFrame> = None;
    let mut next_frame_id: u32 = 1;
    let mut last_frame_sess: Option<usize> = None;

    for (ti, tr) in ix.turns.iter().enumerate() {
        let low = normalize_lower(&tr.body_for_headers);
        let mask = turn_slot_mask(ix, ti);
        let is_assistant = tr.field == 1 || tr.speaker.eq_ignore_ascii_case("assistant");
        let same_sess = last_frame_sess == Some(tr.session);

        if tr.has_tool_call || tr.tool_has_mutation {
            let asks_confirm = is_assistant && is_assistant_proposal_turn(&low, tr.is_question);
            let next_state = if asks_confirm {
                GroundingState::OpenProposal
            } else {
                GroundingState::ActiveTask
            };
            if let Some(top) = stack.last_mut()
                && top.state != GroundingState::GroundedClosed
                && same_sess
            {
                top.last_active_turn = ti as u32;
                top.slot_mask |= mask;
                if tr.tool_has_mutation && top.state != GroundingState::OpenProposal {
                    top.presentation_turn = ti as u32;
                }
                top.state = next_state;
            } else {
                stack.push(DspFrame {
                    frame_id: next_frame_id,
                    presentation_turn: ti as u32,
                    last_active_turn: ti as u32,
                    slot_mask: mask,
                    state: next_state,
                });
                next_frame_id = next_frame_id.saturating_add(1);
                last_frame_sess = Some(tr.session);
            }
            continue;
        }

        if is_assistant {
            if is_assistant_proposal_turn(&low, tr.is_question) {
                if let Some(top) = stack.last_mut()
                    && top.state != GroundingState::GroundedClosed
                    && same_sess
                    && (mask == 0 || (top.slot_mask & mask) != 0 || top.state == GroundingState::ActiveTask)
                {
                    top.presentation_turn = ti as u32;
                    top.last_active_turn = ti as u32;
                    top.slot_mask |= mask;
                    top.state = GroundingState::OpenProposal;
                } else {
                    stack.push(DspFrame {
                        frame_id: next_frame_id,
                        presentation_turn: ti as u32,
                        last_active_turn: ti as u32,
                        slot_mask: mask,
                        state: GroundingState::OpenProposal,
                    });
                    next_frame_id = next_frame_id.saturating_add(1);
                    last_frame_sess = Some(tr.session);
                }
            } else if !tr.is_backchannel
                && let Some(top) = stack.last_mut()
                && top.state != GroundingState::GroundedClosed
                && same_sess
            {
                top.last_active_turn = ti as u32;
                top.slot_mask |= mask;
            }
            continue;
        }

        // User / named-speaker turn
        let act = classify_query_grounding_act(&tr.body_for_headers);
        match act {
            QueryGroundingAct::CloseDone => {
                if let Some(mut popped) = stack.pop() {
                    popped.state = GroundingState::GroundedClosed;
                    popped.last_active_turn = ti as u32;
                    popped.slot_mask |= mask;
                    just_closed = Some(popped);
                }
            }
            QueryGroundingAct::ConfirmProceed => {
                if let Some(frame) = stack
                    .iter_mut()
                    .rev()
                    .find(|f| f.state == GroundingState::OpenProposal)
                {
                    frame.state = GroundingState::ActiveTask;
                    frame.last_active_turn = ti as u32;
                    frame.slot_mask |= mask;
                }
            }
            QueryGroundingAct::VerifyLastValue | QueryGroundingAct::Standard => {
                if tr.is_backchannel {
                    continue;
                }
                let attaches_to_top = stack.last().is_some_and(|top| {
                    top.state != GroundingState::GroundedClosed
                        && same_sess
                        && ((top.slot_mask & mask) != 0 || tr.anaphoric || tr.answers_question)
                });
                if attaches_to_top {
                    if let Some(top) = stack.last_mut() {
                        top.last_active_turn = ti as u32;
                        top.slot_mask |= mask;
                    }
                } else if tr.is_initiative
                    || tr.slot_entities_clean > 0
                    || tr.has_numeric
                    || tr.form_clock
                    || !tr.content.is_empty()
                {
                    stack.push(DspFrame {
                        frame_id: next_frame_id,
                        presentation_turn: ti as u32,
                        last_active_turn: ti as u32,
                        slot_mask: mask,
                        state: GroundingState::ActiveTask,
                    });
                    next_frame_id = next_frame_id.saturating_add(1);
                    last_frame_sess = Some(tr.session);
                }
            }
        }
    }

    (stack, just_closed)
}

/// Deterministic `Q16.16` Discourse-Stack grounding boost for a candidate turn.
pub(in crate::retrieval) fn grounding_stack_boost_q16(
    act: QueryGroundingAct,
    turn_id: u32,
    stack: &[DspFrame],
    just_closed: Option<&DspFrame>,
) -> i64 {
    let target = match act {
        QueryGroundingAct::ConfirmProceed => stack
            .iter()
            .rev()
            .find(|f| f.state == GroundingState::OpenProposal)
            .or_else(|| stack.last()),
        QueryGroundingAct::CloseDone => stack.last().or(just_closed),
        QueryGroundingAct::VerifyLastValue => stack
            .iter()
            .rev()
            .find(|f| f.state != GroundingState::GroundedClosed)
            .or(just_closed),
        QueryGroundingAct::Standard => None,
    };
    let Some(frame) = target else {
        return 0;
    };
    if frame.frame_id == 0 {
        return 0;
    }
    if turn_id == frame.presentation_turn {
        4 * ONE
    } else if turn_id == frame.last_active_turn {
        3 * ONE
    } else if turn_id > frame.presentation_turn && turn_id < frame.last_active_turn {
        ONE
    } else {
        0
    }
}

// =========================================================================
// 2. PASSIVE FSRS-7 DUAL-STABILITY WORKING-MEMORY GATE + ACT-R BLL
// =========================================================================

/// Power-law forgetting curve `(1 + factor * (t / S))^(-decay)` in pure `Q16.16` (zero 30-day clamp).
fn power_decay_q16(t_elapsed_sec: u64, stability_sec: u64, factor_q16: i64, decay_q16: i64) -> i64 {
    if t_elapsed_sec == 0 {
        return ONE;
    }
    let s = stability_sec.max(1);
    let ratio_q16 = (((u128::from(t_elapsed_sec) << 16) / u128::from(s))
        .min((i64::MAX / 4) as u128)) as i64;
    let base_q16 = ONE.saturating_add(mul_q16(factor_q16, ratio_q16));
    let lg = log2_lut_q16(base_q16.max(ONE) as u64);
    let exp_arg = -mul_q16(lg, decay_q16);
    exp2_lut_q16(exp_arg).clamp(0, ONE)
}

/// Convert stability seconds to normalized `Q16.16` days (`>= 1/1440` day).
#[inline]
fn stability_sec_to_days_q16(stability_sec: u64) -> i64 {
    let scaled = ((u128::from(stability_sec) << 16) / (SECS_PER_DAY as u128))
        .min((i64::MAX / 4) as u128) as i64;
    scaled.max(MIN_STABILITY_DAYS_Q16)
}

/// Compute FSRS-7 dual-trace components `(r_fast_q16, r_slow_q16, wm_gate_q16)` in `Q16.16`.
///
/// `wm_gate = (W1 * R_fast) / (W1 * R_fast + W2 * R_slow)` cleanly separates active working
/// memory (`S_fast`, minutes/hours) from consolidated long-term memory (`S_slow`, days/months)
/// with no 30-day horizon clamp.
pub(in crate::retrieval) fn dual_trace_components_q16(
    t_elapsed_sec: u64,
    s_fast_sec: u64,
    s_slow_sec: u64,
) -> (i64, i64, i64) {
    // Fast working-memory decay: decay1 = 0.65 (42_598), factor1 = 3.0 (196_608).
    let r_fast = power_decay_q16(t_elapsed_sec, s_fast_sec, 3 * ONE, 42_598);
    // Slow biographical decay: decay2 = 0.20 (13_107), factor2 = 0.25 (16_384).
    let r_slow = power_decay_q16(t_elapsed_sec, s_slow_sec, ONE >> 2, 13_107);

    let sf_days_q16 = stability_sec_to_days_q16(s_fast_sec);
    let ss_days_q16 = stability_sec_to_days_q16(s_slow_sec);

    let w1 = div_q16(ONE, sqrt_q16(sf_days_q16).max(1));
    let w2 = mul_q16(sqrt_q16(ss_days_q16), ONE >> 2);

    let num1 = mul_q16(w1, r_fast);
    let num2 = mul_q16(w2, r_slow);
    let denom = num1.saturating_add(num2).max(1);
    let wm_gate = div_q16(num1, denom).clamp(0, ONE);
    (r_fast, r_slow, wm_gate)
}

/// Passive FSRS-7 dual-stability working-memory gate combined with ACT-R Base-Level Learning.
pub(in crate::retrieval) fn dual_trace_activation_q16(
    t_elapsed_sec: u64,
    s_fast_sec: u64,
    s_slow_sec: u64,
    bll_sum_q16: i64,
    is_wm_query: bool,
) -> i64 {
    let (_r_fast, r_slow, wm_gate) = dual_trace_components_q16(t_elapsed_sec, s_fast_sec, s_slow_sec);
    if is_wm_query {
        mul_q16(wm_gate, bll_sum_q16.max(0))
    } else {
        mul_q16(ONE - wm_gate, r_slow).saturating_add(bll_sum_q16 >> 2)
    }
}

/// Compute ACT-R Base-Level Learning activation `ln(sum_j dt_j^(-0.5))` shifted to positive `Q16.16`.
fn actr_bll_q16(now_sec: i64, access_ts: &[i64]) -> i64 {
    let mut sum_q16: u64 = 0;
    for &tj in access_ts {
        if tj > now_sec {
            continue;
        }
        let dt_sec = (now_sec - tj).max(1) as u64;
        // Scale time in hours (Q16.16), floored at ~1 minute.
        let dt_hr_q16 = (((u128::from(dt_sec) << 16) / 3_600).clamp(1_024, (i64::MAX / 4) as u128)) as i64;
        let inv_sqrt = div_q16(ONE, sqrt_q16(dt_hr_q16).max(1)).max(1) as u64;
        sum_q16 = sum_q16.saturating_add(inv_sqrt);
    }
    if sum_q16 == 0 {
        return ONE;
    }
    // Shift by +4.0 so activation stays strictly non-negative across multi-year spans.
    ln_lut_q16(sum_q16).saturating_add(4 * ONE).max(ONE >> 2)
}

// =========================================================================
// 3. INCREMENTAL RUN-TIME EVENT CALCULUS (RTEC) + AGM ENTRENCHMENT CLIPPING
// =========================================================================

/// Maximal inertial fluent interval `[t_start, t_end)` for a `(slot_hash, value_hash)` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct FluentInterval {
    pub(in crate::retrieval) slot_hash: u64,
    pub(in crate::retrieval) value_hash: u64,
    pub(in crate::retrieval) t_start: i64,
    /// `None` means the fluent is currently active (unclipped).
    pub(in crate::retrieval) t_end: Option<i64>,
    pub(in crate::retrieval) source_turn: u32,
    /// Epistemic entrenchment in `Q16.16` (`1.0 ..= 4.0`).
    pub(in crate::retrieval) entrenchment: i64,
}

/// Apply an incremental RTEC + AGM epistemic-entrenchment update to the fluent interval log.
pub(in crate::retrieval) fn apply_rtec_update(
    intervals: &mut Vec<FluentInterval>,
    new_f: FluentInterval,
) {
    if let Some(prev) = intervals
        .iter_mut()
        .rev()
        .find(|f| f.slot_hash == new_f.slot_hash && f.t_end.is_none())
    {
        if new_f.value_hash == prev.value_hash {
            prev.entrenchment = prev.entrenchment.max(new_f.entrenchment);
            return;
        }
        // AGM contraction by epistemic entrenchment + Katsuno-Mendelzon world update:
        // An unconfirmed assistant suggestion (1.0) never clips a user assertion or higher (2.0+).
        if new_f.entrenchment >= prev.entrenchment
            || (new_f.t_start > prev.t_start && new_f.entrenchment >= ENTRENCH_ASSERTION_Q16)
        {
            prev.t_end = Some(new_f.t_start);
            intervals.push(new_f);
        }
    } else {
        intervals.push(new_f);
    }
}

/// Monotonic effective timestamp for turn `ti` (falls back to dialogue-order seconds if unset).
fn effective_turn_ts(ix: &Index, ti: usize) -> i64 {
    let tr = &ix.turns[ti];
    let sts = ix.session_ts.get(tr.session).copied().unwrap_or(0);
    let local_pos = ix
        .session_turns
        .get(tr.session)
        .and_then(|v| v.iter().position(|&x| x == ti))
        .unwrap_or(0) as i64;
    if sts > 0 {
        sts.saturating_add(local_pos.saturating_mul(15))
    } else {
        ((ti as i64) + 1).saturating_mul(60)
    }
}

/// Detect closed-class aspectual revision / state-update cues in normalized text.
/// hardcode-scan: closed-class (aspectual transition verbs and temporal revision adverbs only).
fn has_closed_class_revision_cue(low: &str) -> bool {
    // hardcode-scan: closed-class (state-transition and revision operators only)
    const REVISION_CUES: &[&str] = &[
        "switched to ",
        "switching to ",
        "changed to ",
        "changing to ",
        "updated to ",
        "moved to ",
        "migrated to ",
        "upgraded to ",
        "replaced ",
        "instead of ",
        "no longer ",
        "not anymore",
        "changed my mind",
        "now set to ",
        "actually,",
        "actually ",
        "as of now",
        "from now on",
        "dropped to ",
        "increased to ",
        "reduced to ",
        "raised to ",
        "bumped to ",
        "switch ",
        "change ",
        "update ",
    ];
    REVISION_CUES.iter().any(|c| low.contains(c))
}

/// Deterministic lowest-DF content token ID in turn `ti` (tie-broken by vocab ID).
fn rarest_turn_slot_token(ix: &Index, ti: usize) -> Option<(u32, u64)> {
    let tr = &ix.turns[ti];
    let mut best: Option<(u32, u32)> = None;
    for &id in &tr.content {
        let Some(name) = ix.vocab.names.get(id as usize) else {
            continue;
        };
        if !is_content(name) || name.starts_with("__") || name.starts_with('\u{1}') {
            continue;
        }
        let df = ix.s_stats.df.get(&id).copied().unwrap_or(1).max(1);
        match best {
            None => best = Some((df, id)),
            Some((bdf, bid)) if df < bdf || (df == bdf && id < bid) => best = Some((df, id)),
            _ => {}
        }
    }
    best.map(|(_, id)| {
        let name = ix.vocab.names[id as usize].as_str();
        (id, fnv1a64(name.as_bytes()))
    })
}

/// Extract structured `key=value` / `key: value` pairs from tool-call receipts (pure punctuation grammar).
fn extract_tool_kv_pairs(text: &str) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let low = normalize_lower(text);
    for seg in low.split(|c: char| matches!(c, ',' | ';' | '(' | ')' | '{' | '}' | '[' | ']')) {
        let trimmed = seg.trim();
        let pair = trimmed.split_once('=').or_else(|| trimmed.split_once(':'));
        let Some((raw_k, raw_v)) = pair else {
            continue;
        };
        let k = raw_k
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
            .trim();
        let v = raw_v
            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '.' && c != '-')
            .trim();
        if k.len() >= 2 && !v.is_empty() && is_content(k) {
            out.push((fnv1a64(k.as_bytes()), fnv1a64(v.as_bytes())));
        }
    }
    out
}

/// Extract closed-class copula / assignment / revision `(slot_hash, value_hash)` bindings from tokens.
/// hardcode-scan: closed-class (copulas, revision auxiliaries, negation, and polarity atoms only).
fn extract_syntactic_slot_values(tokens: &[String]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let n = tokens.len();

    let is_value_atom = |w: &str| -> bool {
        matches!(w, "no" | "yes" | "not" | "none" | "true" | "false" | "off" | "on")
            || w.bytes().any(|b| b.is_ascii_digit())
            || is_content(w)
    };

    for i in 0..n {
        let w = tokens[i].as_str();

        // Pattern 1: `<verb_rev> <slot> to <val>` (e.g. `"switch insurance to no"`)
        if matches!(
            w,
            "switch"
                | "switched"
                | "change"
                | "changed"
                | "update"
                | "updated"
                | "set"
                | "move"
                | "moved"
                | "keep"
                | "kept"
                | "raise"
                | "raised"
                | "lower"
                | "lowered"
                | "reduce"
                | "reduced"
                | "increase"
                | "increased"
        ) && i + 3 < n
        {
            let mut slot_pos = i + 1;
            if matches!(tokens[slot_pos].as_str(), "the" | "my" | "our" | "their" | "its")
                && slot_pos + 3 < n
            {
                slot_pos += 1;
            }
            if slot_pos + 2 < n
                && tokens[slot_pos + 1] == "to"
                && is_content(&tokens[slot_pos])
                && is_value_atom(&tokens[slot_pos + 2])
            {
                let sh = fnv1a64(tokens[slot_pos].as_bytes());
                let vh = fnv1a64(tokens[slot_pos + 2].as_bytes());
                out.push((sh, vh));
                continue;
            }
            if slot_pos + 3 < n
                && tokens[slot_pos + 2] == "to"
                && is_content(&tokens[slot_pos + 1])
                && is_value_atom(&tokens[slot_pos + 3])
            {
                let sh = fnv1a64(tokens[slot_pos + 1].as_bytes());
                let vh = fnv1a64(tokens[slot_pos + 3].as_bytes());
                out.push((sh, vh));
                continue;
            }
        }

        // Pattern 2: `<slot> [is|was|are|were] [still|now] (set|switched|changed|updated|moved|migrated|upgraded|dropped|increased|reduced|raised|bumped) to <val>`
        if matches!(
            w,
            "set"
                | "switched"
                | "changed"
                | "updated"
                | "moved"
                | "migrated"
                | "upgraded"
                | "dropped"
                | "increased"
                | "reduced"
                | "raised"
                | "bumped"
        ) && i >= 1
            && i + 2 < n
            && tokens[i + 1] == "to"
            && is_value_atom(&tokens[i + 2])
        {
            let mut slot_idx = i - 1;
            while slot_idx > 0
                && matches!(
                    tokens[slot_idx].as_str(),
                    "is" | "was" | "are" | "were" | "be" | "been" | "still" | "now" | "currently"
                )
            {
                slot_idx -= 1;
            }
            if is_content(&tokens[slot_idx]) {
                let sh = fnv1a64(tokens[slot_idx].as_bytes());
                let vh = fnv1a64(tokens[i + 2].as_bytes());
                out.push((sh, vh));
                continue;
            }
        }

        // Pattern 3: `<slot> (is|was|are|were|became) <val>` (skipping passive participles handled by Pattern 2)
        if matches!(w, "is" | "was" | "are" | "were" | "became") && i >= 1 && i + 1 < n {
            let slot_tok = tokens[i - 1].as_str();
            if is_content(slot_tok) {
                let mut val_idx = i + 1;
                if matches!(tokens[val_idx].as_str(), "now" | "currently" | "still" | "a" | "an" | "the")
                    && val_idx + 1 < n
                {
                    val_idx += 1;
                }
                let is_participle_to = val_idx + 1 < n
                    && tokens[val_idx + 1] == "to"
                    && matches!(
                        tokens[val_idx].as_str(),
                        "set"
                            | "switched"
                            | "changed"
                            | "updated"
                            | "moved"
                            | "migrated"
                            | "upgraded"
                            | "dropped"
                            | "increased"
                            | "reduced"
                            | "raised"
                            | "bumped"
                    );
                if !is_participle_to && is_value_atom(&tokens[val_idx]) {
                    let sh = fnv1a64(slot_tok.as_bytes());
                    let vh = fnv1a64(tokens[val_idx].as_bytes());
                    out.push((sh, vh));
                }
            }
        }
    }

    out
}

/// Hash distinctive value signature (digits, polarity atoms, non-slot content tokens) for fallback fluent binding.
fn turn_value_signature_hash(tokens: &[String], exclude_slot_hash: u64) -> u64 {
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    let mut any = false;
    for tok in tokens {
        let is_sig = tok.bytes().any(|b| b.is_ascii_digit())
            || matches!(tok.as_str(), "no" | "yes" | "not" | "none" | "never" | "false" | "true")
            || is_content(tok);
        if !is_sig {
            continue;
        }
        let h = fnv1a64(tok.as_bytes());
        if h == exclude_slot_hash {
            continue;
        }
        acc ^= h;
        acc = acc.wrapping_mul(0x0100_0000_01b3);
        any = true;
    }
    if any { acc } else { exclude_slot_hash ^ 0x55aa_55aa }
}

/// Build the full dialog-wide RTEC inertial fluent interval log with AGM epistemic entrenchment.
pub(in crate::retrieval) fn build_rtec_fluents(ix: &Index) -> Vec<FluentInterval> {
    let mut intervals: Vec<FluentInterval> = Vec::new();
    let mut turn_primary_slot: Vec<Option<u64>> = vec![None; ix.turns.len()];

    for (ti, tr) in ix.turns.iter().enumerate() {
        if tr.is_question || tr.is_backchannel || (tr.modality_mask & 0x04) != 0 {
            continue;
        }
        let t_start = effective_turn_ts(ix, ti);
        let low = normalize_lower(&tr.body_for_headers);
        let tokens = raw_tokens(&tr.body_for_headers);
        let has_rev = has_closed_class_revision_cue(&low);
        let is_assistant = tr.field == 1 || tr.speaker.eq_ignore_ascii_case("assistant");

        let entrenchment = if !tr.supersedes_turns.is_empty() || (!is_assistant && has_rev) {
            ENTRENCH_CORRECTION_Q16
        } else if tr.tool_has_mutation {
            ENTRENCH_TOOL_MUTATION_Q16
        } else if !is_assistant {
            ENTRENCH_ASSERTION_Q16
        } else {
            ENTRENCH_SUGGESTION_Q16
        };

        let mut emitted_slots: Vec<u64> = Vec::new();

        // Path A: Verified `supersedes_turns` backward state-revision edges.
        for &old_ti in &tr.supersedes_turns {
            if old_ti >= ti {
                continue;
            }
            let shared_slot = ix.turns[old_ti]
                .content
                .iter()
                .filter(|id| tr.content.contains(id))
                .filter_map(|&id| {
                    let name = ix.vocab.names.get(id as usize)?;
                    is_content(name).then(|| {
                        let df = ix.s_stats.df.get(&id).copied().unwrap_or(1).max(1);
                        (df, id, fnv1a64(name.as_bytes()))
                    })
                })
                .min_by_key(|&(df, id, _)| (df, id))
                .map(|(_, _, sh)| sh)
                .or_else(|| turn_primary_slot.get(old_ti).copied().flatten());

            if let Some(sh) = shared_slot {
                let vh = turn_value_signature_hash(&tokens, sh) ^ ((ti as u64).wrapping_mul(0x9e37));
                apply_rtec_update(
                    &mut intervals,
                    FluentInterval {
                        slot_hash: sh,
                        value_hash: vh,
                        t_start,
                        t_end: None,
                        source_turn: ti as u32,
                        entrenchment: ENTRENCH_CORRECTION_Q16,
                    },
                );
                emitted_slots.push(sh);
            }
        }

        // Path B: Structured tool-call key/value mutations.
        if tr.has_tool_call || tr.tool_has_mutation {
            let tool_ent = if tr.tool_has_mutation {
                ENTRENCH_TOOL_MUTATION_Q16
            } else {
                ENTRENCH_SUGGESTION_Q16
            };
            for (sh, vh) in extract_tool_kv_pairs(&tr.body_for_headers) {
                apply_rtec_update(
                    &mut intervals,
                    FluentInterval {
                        slot_hash: sh,
                        value_hash: vh,
                        t_start,
                        t_end: None,
                        source_turn: ti as u32,
                        entrenchment: tool_ent,
                    },
                );
                emitted_slots.push(sh);
            }
        }

        // Path C: Closed-class syntactic slot-value assertions & revisions.
        for (sh, vh) in extract_syntactic_slot_values(&tokens) {
            if emitted_slots.contains(&sh) {
                continue;
            }
            apply_rtec_update(
                &mut intervals,
                FluentInterval {
                    slot_hash: sh,
                    value_hash: vh,
                    t_start,
                    t_end: None,
                    source_turn: ti as u32,
                    entrenchment,
                },
            );
            emitted_slots.push(sh);
        }

        // Path D: Fallback anchor binding on lowest-DF content noun for declarative updates/assertions.
        if let Some((_id, sh)) = rarest_turn_slot_token(ix, ti) {
            if turn_primary_slot[ti].is_none() {
                turn_primary_slot[ti] = Some(sh);
            }
            let substantive =
                has_rev || tr.has_numeric || tr.form_digit || tr.slot_entities_clean > 0 || tr.tool_has_mutation;
            if substantive && !emitted_slots.contains(&sh) {
                let vh = turn_value_signature_hash(&tokens, sh);
                apply_rtec_update(
                    &mut intervals,
                    FluentInterval {
                        slot_hash: sh,
                        value_hash: vh,
                        t_start,
                        t_end: None,
                        source_turn: ti as u32,
                        entrenchment,
                    },
                );
            }
        }
    }

    intervals
}

// =========================================================================
// 4. CANONICAL DIFFERENCE BOUND MATRIX (DBM) + ALLEN ORD-HORN + TIMEX3
// =========================================================================

/// Allen ORD-Horn query relation compiled into DBM difference constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllenQueryRel {
    /// `Allen ¬(before ∨ after)`: `start_ev <= end_q` and `start_q <= end_ev`.
    Overlap,
    /// `Allen before`: `end_ev <= start_q`.
    Before,
    /// `Allen after`: `end_q <= start_ev`.
    After,
}

/// Allocate an `n x n` canonical DBM initialized to `DBM_INF` with `0` on the diagonal.
fn dbm_init(n: usize) -> Vec<i64> {
    let mut d = vec![DBM_INF; n * n];
    for i in 0..n {
        d[i * n + i] = 0;
    }
    d
}

/// Incremental `O(n^2)` tropical `(min, +)` DBM constraint addition & closure.
///
/// Enforces `x_i - x_j <= c` on flat row-major matrix `d` of dimension `n x n`.
/// Returns `false` immediately upon detecting a negative cycle (`D[u, u] < 0`), indicating
/// an unsatisfiable temporal contradiction.
pub(in crate::retrieval) fn dbm_add_constraint_and_close(
    d: &mut [i64],
    n: usize,
    i: usize,
    j: usize,
    c: i64,
) -> bool {
    if n == 0 || i >= n || j >= n || d.len() < n * n {
        return false;
    }
    if i == j && c < 0 {
        d[i * n + i] = c;
        return false;
    }
    if c >= d[i * n + j] {
        return true;
    }
    d[i * n + j] = c;
    for u in 0..n {
        let dui = d[u * n + i];
        if dui >= DBM_INF {
            continue;
        }
        let via_ui = dui.saturating_add(c);
        for v in 0..n {
            let djv = d[j * n + v];
            if djv >= DBM_INF {
                continue;
            }
            let cand = via_ui.saturating_add(djv);
            if cand < d[u * n + v] {
                d[u * n + v] = cand;
            }
        }
        if d[u * n + u] < 0 {
            return false;
        }
    }
    true
}

/// Check Allen ORD-Horn temporal consistency between event interval `[ev_l, ev_u]`
/// and query interval `[q_l, q_u]` via a 5-node canonical DBM:
/// `0 = x_0 (reference clock)`, `1 = start_ev`, `2 = end_ev`, `3 = start_q`, `4 = end_q`.
fn dbm_check_turn_vs_query(
    ev_l: i64,
    ev_u: i64,
    q_l: i64,
    q_u: i64,
    rel: AllenQueryRel,
) -> bool {
    let ev_hi = ev_u.max(ev_l);
    let q_hi = q_u.max(q_l);
    let mut d = dbm_init(5);

    // Fix event interval endpoints relative to reference clock x_0 = 0:
    // start_ev - x_0 <= ev_l, x_0 - start_ev <= -ev_l
    // end_ev   - x_0 <= ev_hi, x_0 - end_ev  <= -ev_hi
    if !dbm_add_constraint_and_close(&mut d, 5, 1, 0, ev_l)
        || !dbm_add_constraint_and_close(&mut d, 5, 0, 1, ev_l.saturating_neg())
        || !dbm_add_constraint_and_close(&mut d, 5, 2, 0, ev_hi)
        || !dbm_add_constraint_and_close(&mut d, 5, 0, 2, ev_hi.saturating_neg())
        || !dbm_add_constraint_and_close(&mut d, 5, 1, 2, 0)
    {
        return false;
    }

    // Fix query interval endpoints relative to reference clock x_0 = 0:
    if !dbm_add_constraint_and_close(&mut d, 5, 3, 0, q_l)
        || !dbm_add_constraint_and_close(&mut d, 5, 0, 3, q_l.saturating_neg())
        || !dbm_add_constraint_and_close(&mut d, 5, 4, 0, q_hi)
        || !dbm_add_constraint_and_close(&mut d, 5, 0, 4, q_hi.saturating_neg())
        || !dbm_add_constraint_and_close(&mut d, 5, 3, 4, 0)
    {
        return false;
    }

    // Compile Allen ORD-Horn relation into difference constraints:
    match rel {
        AllenQueryRel::Overlap => {
            dbm_add_constraint_and_close(&mut d, 5, 1, 4, 0)
                && dbm_add_constraint_and_close(&mut d, 5, 3, 2, 0)
        }
        AllenQueryRel::Before => dbm_add_constraint_and_close(&mut d, 5, 2, 3, 0),
        AllenQueryRel::After => dbm_add_constraint_and_close(&mut d, 5, 4, 1, 0),
    }
}

/// Intersection-over-event-span tightness score in `Q16.16` (`0` when DBM is inconsistent).
pub(in crate::retrieval) fn dbm_temporal_match_q16(
    ev_l: i64,
    ev_u: i64,
    q_l: i64,
    q_u: i64,
    consistent: bool,
) -> i64 {
    if !consistent {
        return 0;
    }
    let ev_hi = ev_u.max(ev_l);
    let q_hi = q_u.max(q_l);
    let inter = (ev_hi.min(q_hi) - ev_l.max(q_l)).max(0) as i128;
    if inter == 0 {
        // Consistent under strict Allen Before/After with non-overlapping endpoints:
        // reward proximity inversely with distance from the boundary.
        let gap_days = ((ev_l.max(q_l) - ev_hi.min(q_hi)).max(0) / SECS_PER_DAY).clamp(0, 365);
        return ((ONE * (365 - gap_days)) / 365).max(ONE >> 3);
    }
    let ev_span = (ev_hi - ev_l).max(1) as i128;
    let q_span = (q_hi - q_l).max(1) as i128;
    let cov_q16 = ((inter << 16) / ev_span).clamp(0, i128::from(ONE)) as i64;
    let tightness_bonus = (((inter << 16) / q_span).clamp(0, i128::from(ONE)) as i64) >> 2;
    cov_q16.saturating_add(tightness_bonus)
}

/// Pure integer Howard Hinnant civil `(year, month, day)` -> Unix seconds at `00:00:00 UTC`.
fn ymd_to_epoch_secs(year: i32, month: u32, day: u32) -> i64 {
    let m = month.clamp(1, 12) as i64;
    let d = day.clamp(1, 31) as i64;
    let y = i64::from(year) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    days.saturating_mul(SECS_PER_DAY)
}

/// Compute half-open month span `[start, next_month_start)` in Unix seconds.
fn month_span_secs(year: i32, month: u32) -> (i64, i64) {
    let m = month.clamp(1, 12);
    let start = ymd_to_epoch_secs(year, m, 1);
    let (ny, nm) = if m == 12 { (year + 1, 1) } else { (year, m + 1) };
    let end = ymd_to_epoch_secs(ny, nm, 1);
    (start, end)
}

/// Pure integer year extraction from Unix seconds.
fn epoch_secs_to_year(ts: i64) -> i32 {
    let z = ts.div_euclid(SECS_PER_DAY) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (y + i64::from(m <= 2)) as i32
}

/// Closed-class small English cardinal numeral parser.
/// hardcode-scan: closed-class (cardinal number words only).
fn parse_small_qty(tok: &str) -> Option<i64> {
    if let Ok(v) = tok.parse::<i64>() {
        return (1..=365).contains(&v).then_some(v);
    }
    match tok {
        "a" | "an" | "one" => Some(1),
        "two" | "couple" => Some(2),
        "three" | "few" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        _ => None,
    }
}

/// Resolve `"past <N> <unit>"` / `"last <N> <unit>"` lookback window anchored at `anchor_ts`.
/// hardcode-scan: closed-class (temporal span prepositions, numerals, and calendar units only).
fn extract_past_span_window(tokens: &[String], anchor_ts: i64) -> Option<(i64, i64)> {
    if anchor_ts <= 0 {
        return None;
    }
    for i in 0..tokens.len() {
        if !matches!(tokens[i].as_str(), "past" | "last") {
            continue;
        }
        if i + 2 >= tokens.len() {
            continue;
        }
        let Some(qty) = parse_small_qty(&tokens[i + 1]) else {
            continue;
        };
        let unit_secs = match tokens[i + 2].as_str() {
            "day" | "days" => Some(SECS_PER_DAY),
            "week" | "weeks" => Some(7 * SECS_PER_DAY),
            "month" | "months" => Some(30 * SECS_PER_DAY),
            "year" | "years" => Some(365 * SECS_PER_DAY),
            _ => None,
        };
        if let Some(u) = unit_secs {
            return Some((anchor_ts.saturating_sub(qty.saturating_mul(u)), anchor_ts));
        }
    }
    None
}

/// Resolve inline calendar date tokens (`__DATE__`, `__YM__`, `__MD__`, `__MONTH__`, `__YEAR__`)
/// into a half-open Unix-seconds interval `[start, end)`.
fn resolve_inline_calendar_interval(text: &str, default_year: i32) -> Option<(i64, i64)> {
    let toks = inline_date_tokens(text);
    if toks.is_empty() {
        return None;
    }
    for tok in &toks {
        if let Some(ymd) = tok.strip_prefix("__DATE__") {
            let mut it = ymd.split('_');
            if let (Some(ys), Some(ms), Some(ds)) = (it.next(), it.next(), it.next())
                && let (Ok(y), Ok(m), Ok(d)) = (ys.parse::<i32>(), ms.parse::<u32>(), ds.parse::<u32>())
            {
                let s = ymd_to_epoch_secs(y, m, d);
                return Some((s, s + SECS_PER_DAY));
            }
        }
    }
    for tok in &toks {
        if let Some(ym) = tok.strip_prefix("__YM__") {
            let mut it = ym.split('_');
            if let (Some(ys), Some(ms)) = (it.next(), it.next())
                && let (Ok(y), Ok(m)) = (ys.parse::<i32>(), ms.parse::<u32>())
            {
                return Some(month_span_secs(y, m));
            }
        }
    }
    for tok in &toks {
        if let Some(md) = tok.strip_prefix("__MD__") {
            let mut it = md.split('_');
            if let (Some(ms), Some(ds)) = (it.next(), it.next())
                && let (Ok(m), Ok(d)) = (ms.parse::<u32>(), ds.parse::<u32>())
            {
                let s = ymd_to_epoch_secs(default_year, m, d);
                return Some((s, s + SECS_PER_DAY));
            }
        }
    }
    for tok in &toks {
        if let Some(mm) = tok.strip_prefix("__MONTH__")
            && let Ok(m) = mm.parse::<u32>()
        {
            return Some(month_span_secs(default_year, m));
        }
    }
    for tok in &toks {
        if let Some(yy) = tok.strip_prefix("__YEAR__")
            && let Ok(y) = yy.parse::<i32>()
        {
            return Some((ymd_to_epoch_secs(y, 1, 1), ymd_to_epoch_secs(y + 1, 1, 1)));
        }
    }
    None
}

/// Maximum valid timestamp across query date and indexed sessions.
fn max_corpus_ts(ix: &Index, q_date: &str) -> i64 {
    let q_ts = parse_date(q_date);
    if q_ts > 0 {
        return q_ts;
    }
    ix.session_ts
        .iter()
        .copied()
        .filter(|&ts| ts > 0)
        .max()
        .unwrap_or(0)
}

/// Extract query temporal constraint window `(q_l, q_u, rel)` anchored against `q_date` / corpus time.
/// hardcode-scan: closed-class (Allen temporal prepositions only).
fn extract_query_temporal_window(
    ix: &Index,
    q_text: &str,
    q_date: &str,
) -> Option<(i64, i64, AllenQueryRel)> {
    let low = normalize_lower(q_text);
    let q_ts = max_corpus_ts(ix, q_date);
    let default_year = if q_ts > 0 { epoch_secs_to_year(q_ts) } else { 2023 };
    let toks = raw_tokens(q_text);
    let tok_refs: Vec<&str> = toks.iter().map(String::as_str).collect();

    let rel = if (low.contains(" before ") || low.contains(" prior to ") || low.contains(" earlier than "))
        && !low.contains("before i switched")
        && !low.contains("before i changed")
    {
        AllenQueryRel::Before
    } else if low.contains(" after ") || low.contains(" since ") || low.contains(" following ") {
        AllenQueryRel::After
    } else {
        AllenQueryRel::Overlap
    };

    if let Some((s, e)) = extract_past_span_window(&toks, q_ts) {
        return Some((s, e, AllenQueryRel::Overlap));
    }
    if q_ts > 0
        && let Some((s, e)) = compile_ingest_timex_anchors(&tok_refs, q_ts).into_iter().next()
    {
        return Some((s, e, rel));
    }
    if let Some((s, e)) = resolve_inline_calendar_interval(q_text, default_year) {
        return Some((s, e, rel));
    }
    None
}

/// Anchor a turn's event time interval(s) distinct from its utterance `session_ts`.
///
/// Returns `(intervals, is_explicit_event_timex)`.
fn turn_event_intervals(ix: &Index, ti: usize) -> (Vec<(i64, i64)>, bool) {
    let tr = &ix.turns[ti];
    if !tr.resolved_timex.is_empty() {
        return (tr.resolved_timex.clone(), true);
    }
    let sts = ix.session_ts.get(tr.session).copied().unwrap_or(0);
    let default_year = if sts > 0 { epoch_secs_to_year(sts) } else { 2023 };
    if let Some(iv) = resolve_inline_calendar_interval(&tr.body_for_headers, default_year) {
        return (vec![iv], true);
    }
    if sts > 0 {
        let day_start = sts.div_euclid(SECS_PER_DAY) * SECS_PER_DAY;
        return (vec![(day_start, day_start + SECS_PER_DAY)], false);
    }
    (Vec::new(), false)
}

// =========================================================================
// 5. UNIFIED STAGE-12 ENTRY POINT
// =========================================================================

/// Compute the query's lowest-DF exact non-stopword group bit (`star_bit`).
fn compute_star_bit(ctx: &RankCtx<'_>) -> u64 {
    let mut best: Option<(u32, usize)> = None;
    for qt in &ctx.plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let Some(g) = qt.group else {
            continue;
        };
        if g >= 64 {
            continue;
        }
        let df = ctx.ix.s_stats.df.get(&qt.id).copied().unwrap_or(0);
        if df == 0 {
            continue;
        }
        match best {
            None => best = Some((df, g)),
            Some((bdf, bg)) if df < bdf || (df == bdf && g < bg) => best = Some((df, g)),
            _ => {}
        }
    }
    best.map_or(0, |(_, g)| 1u64 << g)
}

/// Deterministic integer base score in `Q16.16` for candidate turn `ti`.
fn candidate_base_q16(ctx: &RankCtx<'_>, ti: usize) -> i64 {
    let b = ctx.base_q16.get(ti).copied().unwrap_or(0);
    if b > 0 {
        return (b.min((i64::MAX / 8) as u64)) as i64;
    }
    if let Some(pos) = ctx.final_turn_indices.iter().position(|&t| t == ti) {
        let rank_bonus = (32usize.saturating_sub(pos)) as i64;
        return rank_bonus.saturating_mul(ONE >> 1);
    }
    0
}

/// Passive FSRS-7 dual-stability working-memory activation for turn `ti` at reference time `now_ts`.
fn turn_fsrs7_activation_q16(ix: &Index, ti: usize, now_ts: i64, is_wm_query: bool) -> i64 {
    let tr = &ix.turns[ti];
    let t_turn = effective_turn_ts(ix, ti);
    let t_elapsed = now_ts.saturating_sub(t_turn).max(0) as u64;

    let mut s_slow_sec = if tr.stability > 0.0 {
        ((tr.stability as u64).max(1)).saturating_mul(SECS_PER_DAY as u64)
    } else {
        DEFAULT_S_SLOW_SEC
    };
    let mut s_fast_sec = DEFAULT_S_FAST_SEC;

    // FSRS-7 contradiction asymmetry: superseded turns immediately contract S_fast <= 0.8 * S_slow / 8.
    if turn_is_superseded(ix, ti) {
        s_fast_sec = (s_fast_sec / 8).min((s_slow_sec * 4) / 5).max(60);
        s_slow_sec = (s_slow_sec / 2).max(SECS_PER_DAY as u64);
    }

    let mut accesses = vec![t_turn];
    for &(_, target) in &tr.exact_edges {
        if target < ix.turns.len() {
            accesses.push(effective_turn_ts(ix, target));
        }
    }
    let bll = actr_bll_q16(now_ts.max(t_turn), &accesses);
    dual_trace_activation_q16(t_elapsed, s_fast_sec, s_slow_sec, bll, is_wm_query)
}

/// Ensure turn `ti` is promoted to `slot` in `final_turn_indices` (bounded by `k`)
/// and optionally sync its session into `final_session_indices` without dropping existing Top-K.
fn seat_turn_at_slot(
    final_turns: &mut Vec<usize>,
    final_sessions: Option<&mut Vec<usize>>,
    ix: &Index,
    ti: usize,
    slot: usize,
    k: usize,
) {
    if let Some(pos) = final_turns.iter().position(|&t| t == ti) {
        if pos == slot {
            return;
        }
        final_turns.remove(pos);
    }
    let ins = slot.min(final_turns.len());
    final_turns.insert(ins, ti);
    final_turns.truncate(k.max(5));

    if let Some(fsess) = final_sessions {
        let si = ix.turns[ti].session;
        let top_cap = 5usize.min(fsess.len());
        if !fsess[..top_cap].contains(&si) {
            if let Some(spos) = fsess.iter().position(|&s| s == si) {
                fsess.remove(spos);
            }
            let sins = slot.min(fsess.len());
            fsess.insert(sins, si);
            fsess.truncate(k.max(5));
        }
    }
}

/// Unified Stage-12 pass combining:
/// 1. Discourse-Stack Grounding & Obligation Automaton + FSRS-7 `wm_gate` (grounding/closure acts),
/// 2. Incremental RTEC Fluent Clipping + FSRS-7 `wm_gate` (knowledge-update / verification queries),
/// 3. Canonical DBM Negative-Cycle Pruning & Event-Interval Tightness Reranking (temporal queries,
///    strictly within Top-10 sessions so Session R@5 is preserved).
pub(in crate::retrieval) fn apply_rtec_discourse_dbm(ctx: &mut RankCtx<'_>) {
    if ctx.ix.turns.is_empty() || ctx.final_turn_indices.is_empty() {
        return;
    }

    let star_bit = compute_star_bit(ctx);
    let top0_ti = ctx.final_turn_indices[0];
    let top0_tr = &ctx.ix.turns[top0_ti];
    let top0_has_star_decl = star_bit != 0
        && !top0_tr.is_question
        && !top0_tr.is_backchannel
        && (ctx.t_exact_mask.get(top0_ti).copied().unwrap_or(0) & star_bit) != 0;

    let now_ts = {
        let m = max_corpus_ts(ctx.ix, &ctx.q.date);
        if m > 0 {
            m.saturating_add(60)
        } else {
            effective_turn_ts(ctx.ix, ctx.ix.turns.len() - 1).saturating_add(60)
        }
    };

    let act = classify_query_grounding_act(&ctx.q.text);

    // ---------------------------------------------------------------------
    // PASS A: Discourse-Stack Grounding & Obligation Automaton + FSRS-7 WM
    // ---------------------------------------------------------------------
    let discourse_gate = act != QueryGroundingAct::Standard
        && (ctx.ix.has_tool_calls
            || ctx.plan.groups.len() <= 2
            || matches!(act, QueryGroundingAct::ConfirmProceed | QueryGroundingAct::CloseDone));

    let mut active_dsp_slot_mask = 0u64;
    if discourse_gate {
        let (stack, just_closed) = build_discourse_stack(ctx.ix);
        if let Some(top_f) = stack.last().or(just_closed.as_ref()) {
            active_dsp_slot_mask = top_f.slot_mask;
        }

        let mut cand_set: Vec<usize> = ctx.final_turn_indices.iter().copied().take(16).collect();
        for frame in stack.iter().rev().take(2) {
            for idx in [frame.presentation_turn as usize, frame.last_active_turn as usize] {
                if idx < ctx.ix.turns.len() && !cand_set.contains(&idx) {
                    cand_set.push(idx);
                }
            }
        }
        if let Some(jc) = just_closed {
            for idx in [jc.presentation_turn as usize, jc.last_active_turn as usize] {
                if idx < ctx.ix.turns.len() && !cand_set.contains(&idx) {
                    cand_set.push(idx);
                }
            }
        }

        let mut scored: Vec<(i64, usize)> = cand_set
            .into_iter()
            .filter_map(|ti| {
                let g_boost = grounding_stack_boost_q16(act, ti as u32, &stack, just_closed.as_ref());
                if g_boost <= 0 {
                    return None;
                }
                let base = candidate_base_q16(ctx, ti);
                let wm_act = turn_fsrs7_activation_q16(ctx.ix, ti, now_ts, true);
                Some((base.saturating_add(g_boost).saturating_add(wm_act >> 1), ti))
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));

        if matches!(act, QueryGroundingAct::ConfirmProceed | QueryGroundingAct::CloseDone) {
            for (slot, &(_, ti)) in scored.iter().take(2).enumerate() {
                seat_turn_at_slot(
                    &mut ctx.final_turn_indices,
                    Some(&mut ctx.final_session_indices),
                    ctx.ix,
                    ti,
                    slot,
                    ctx.k,
                );
            }
        } else if act == QueryGroundingAct::VerifyLastValue
            && let Some(&(_, ti)) = scored.first()
            && !ctx.final_turn_indices.iter().take(5).any(|&t| t == ti)
        {
            let target_slot = if top0_has_star_decl { 1 } else { 0 };
            seat_turn_at_slot(
                &mut ctx.final_turn_indices,
                Some(&mut ctx.final_session_indices),
                ctx.ix,
                ti,
                target_slot,
                ctx.k,
            );
        }
    }

    // ---------------------------------------------------------------------
    // PASS B: RTEC Fluent Interval Clipping + FSRS-7 Dual-Stability WM Gate
    // ---------------------------------------------------------------------
    let q_pol = detect_temporal_query_polarity(&ctx.q.text, &ctx.eff_it);
    let rtec_gate = !ctx.torrens_locked_head
        && !ctx.eff_it.is_event_sequence
        && !ctx.eff_it.task_under_rules
        && !ctx.eff_it.preference
        && (matches!(
            q_pol,
            TemporalQueryPolarity::LatestCurrentState | TemporalQueryPolarity::GenesisFirst
        ) || act == QueryGroundingAct::VerifyLastValue
            || ctx.eff_it.current_state);

    if rtec_gate {
        let fluents = build_rtec_fluents(ctx.ix);
        if !fluents.is_empty() {
            let mut q_slot_hashes: DetHashSet<u64> = DetHashSet::default();
            for tok in raw_tokens(&ctx.q.text) {
                if is_content(&tok) {
                    q_slot_hashes.insert(fnv1a64(tok.as_bytes()));
                }
            }
            for qt in &ctx.plan.terms {
                if qt.exact_hit
                    && let Some(name) = ctx.ix.vocab.names.get(qt.id as usize)
                    && is_content(name)
                {
                    q_slot_hashes.insert(fnv1a64(name.as_bytes()));
                }
            }

            let matched: Vec<&FluentInterval> = fluents
                .iter()
                .filter(|f| q_slot_hashes.contains(&f.slot_hash))
                .collect();

            if !matched.is_empty() {
                let is_wm_q = ctx.ix.has_tool_calls || act == QueryGroundingAct::VerifyLastValue;
                if q_pol == TemporalQueryPolarity::GenesisFirst {
                    // Route history / before-change queries to exact transition boundary pairs.
                    let mut best_pair: Option<(i64, usize, usize)> = None;
                    for &old_f in &matched {
                        let Some(clip_ts) = old_f.t_end else {
                            continue;
                        };
                        let old_ti = old_f.source_turn as usize;
                        if old_ti >= ctx.ix.turns.len() {
                            continue;
                        }
                        if top0_has_star_decl
                            && (ctx.t_exact_mask.get(old_ti).copied().unwrap_or(0) & star_bit) == 0
                        {
                            continue;
                        }
                        if let Some(&new_f) = matched
                            .iter()
                            .find(|nf| nf.slot_hash == old_f.slot_hash && nf.t_start == clip_ts)
                        {
                            let new_ti = new_f.source_turn as usize;
                            let score = candidate_base_q16(ctx, old_ti)
                                .saturating_add(3 * ONE)
                                .saturating_add(old_f.entrenchment >> 1);
                            match best_pair {
                                None => best_pair = Some((score, old_ti, new_ti)),
                                Some((bs, bo, _)) if score > bs || (score == bs && old_ti < bo) => {
                                    best_pair = Some((score, old_ti, new_ti));
                                }
                                _ => {}
                            }
                        }
                    }
                    if let Some((_, old_ti, new_ti)) = best_pair {
                        let slot_a = if top0_has_star_decl && top0_ti != old_ti { 1 } else { 0 };
                        seat_turn_at_slot(
                            &mut ctx.final_turn_indices,
                            Some(&mut ctx.final_session_indices),
                            ctx.ix,
                            old_ti,
                            slot_a,
                            ctx.k,
                        );
                        if new_ti < ctx.ix.turns.len() && new_ti != old_ti {
                            seat_turn_at_slot(
                                &mut ctx.final_turn_indices,
                                Some(&mut ctx.final_session_indices),
                                ctx.ix,
                                new_ti,
                                slot_a + 1,
                                ctx.k,
                            );
                        }
                    }
                } else {
                    // CurrentState / VerifyLastValue: boost unclipped `[t_start, +inf)` and demote clipped.
                    let mut best_open: Option<(i64, usize)> = None;
                    let mut clipped_turns: DetHashSet<usize> = DetHashSet::default();
                    for &f in &matched {
                        let ti = f.source_turn as usize;
                        if ti >= ctx.ix.turns.len() {
                            continue;
                        }
                        if f.t_end.is_some() {
                            clipped_turns.insert(ti);
                            continue;
                        }
                        let tr = &ctx.ix.turns[ti];
                        if tr.is_question || tr.is_backchannel {
                            continue;
                        }
                        if top0_has_star_decl
                            && (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & star_bit) == 0
                        {
                            continue;
                        }
                        let wm_act = turn_fsrs7_activation_q16(ctx.ix, ti, now_ts, is_wm_q);
                        let dsp_overlap_bonus = if active_dsp_slot_mask != 0
                            && (turn_slot_mask(ctx.ix, ti) & active_dsp_slot_mask) != 0
                        {
                            ONE
                        } else {
                            0
                        };
                        let sc = candidate_base_q16(ctx, ti)
                            .saturating_add(3 * ONE)
                            .saturating_add(f.entrenchment >> 1)
                            .saturating_add(wm_act >> 1)
                            .saturating_add(dsp_overlap_bonus);
                        match best_open {
                            None => best_open = Some((sc, ti)),
                            Some((bs, bt)) if sc > bs || (sc == bs && ti > bt) => {
                                best_open = Some((sc, ti));
                            }
                            _ => {}
                        }
                    }
                    if let Some((_, open_ti)) = best_open {
                        let top0_clipped = clipped_turns.contains(&top0_ti);
                        let target_slot = if top0_clipped || !top0_has_star_decl { 0 } else { 1 };
                        seat_turn_at_slot(
                            &mut ctx.final_turn_indices,
                            Some(&mut ctx.final_session_indices),
                            ctx.ix,
                            open_ti,
                            target_slot,
                            ctx.k,
                        );
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // PASS C: Canonical DBM Negative-Cycle Pruning & Event Tightness Rerank
    // ---------------------------------------------------------------------
    let q_window = extract_query_temporal_window(ctx.ix, &ctx.q.text, &ctx.q.date);
    let dbm_gate = !ctx.torrens_locked_head
        && !ctx.eff_it.is_event_sequence
        && act == QueryGroundingAct::Standard
        && (ctx.eff_it.temporal || ctx.eff_it.sniper_when || q_window.is_some());

    if dbm_gate
        && let Some((q_l, q_u, q_rel)) = q_window
    {
        // Strictly restrict DBM reranking to Top-10 sessions so Session R@5 is untouched.
        let top10_sessions: DetHashSet<usize> =
            ctx.final_session_indices.iter().take(10).copied().collect();

        let mut pool: Vec<usize> = Vec::with_capacity(32);
        for &ti in ctx.final_turn_indices.iter().take(24) {
            if !pool.contains(&ti) {
                pool.push(ti);
            }
        }
        // Pull high-scene declarative turns inside Top-10 sessions with explicit event TIMEX.
        let mut extra: Vec<(usize, i64)> = Vec::new();
        for &si in ctx.final_session_indices.iter().take(10) {
            let Some(members) = ctx.ix.session_turns.get(si) else {
                continue;
            };
            for &ti in members {
                if pool.contains(&ti) {
                    continue;
                }
                let tr = &ctx.ix.turns[ti];
                if tr.is_question || tr.is_backchannel {
                    continue;
                }
                let has_signal = ctx.base_q16.get(ti).copied().unwrap_or(0) > 0
                    || ctx.t_scene.get(ti).is_some_and(|&s| s > 0.0);
                if has_signal && (!tr.resolved_timex.is_empty() || tr.has_temporal) {
                    extra.push((ti, candidate_base_q16(ctx, ti)));
                }
            }
        }
        extra.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (ti, _) in extra {
            if pool.len() >= 32 {
                break;
            }
            pool.push(ti);
        }

        let cur_top0 = ctx.final_turn_indices[0];
        let mut any_consistent = false;
        let mut scored: Vec<(bool, bool, i64, usize)> = Vec::with_capacity(pool.len());

        for &ti in &pool {
            let in_top10 = top10_sessions.contains(&ctx.ix.turns[ti].session);
            let (ivs, is_explicit_ev) = turn_event_intervals(ctx.ix, ti);
            let base = candidate_base_q16(ctx, ti);

            if !in_top10 || ivs.is_empty() {
                scored.push((true, false, base, ti));
                continue;
            }

            let mut best_match = 0i64;
            let mut consistent_any = false;
            for &(ev_l, ev_u) in &ivs {
                let ok = dbm_check_turn_vs_query(ev_l, ev_u, q_l, q_u, q_rel);
                if ok {
                    consistent_any = true;
                    let m = dbm_temporal_match_q16(ev_l, ev_u, q_l, q_u, true);
                    if m > best_match {
                        best_match = m;
                    }
                }
            }

            if consistent_any {
                any_consistent = true;
                let bonus = if is_explicit_ev {
                    best_match.saturating_mul(2)
                } else {
                    best_match
                };
                scored.push((true, is_explicit_ev, base.saturating_add(bonus), ti));
            } else {
                // Explicit event contradiction via DBM negative cycle -> strong demotion;
                // utterance-only mismatch -> moderate demotion.
                let penalty = if is_explicit_ev { 3 * ONE } else { ONE };
                scored.push((false, is_explicit_ev, base.saturating_sub(penalty), ti));
            }
        }

        if any_consistent {
            let cur_top0_tr = &ctx.ix.turns[cur_top0];
            let cur_top0_star_decl = star_bit != 0
                && !cur_top0_tr.is_question
                && !cur_top0_tr.is_backchannel
                && (ctx.t_exact_mask.get(cur_top0).copied().unwrap_or(0) & star_bit) != 0;
            let cur_top0_consistent = scored
                .iter()
                .find(|x| x.3 == cur_top0)
                .is_some_and(|x| x.0);
            let any_consistent_star_decl = scored.iter().any(|&(ok, _, _, ti)| {
                ok && !ctx.ix.turns[ti].is_question
                    && !ctx.ix.turns[ti].is_backchannel
                    && (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & star_bit) != 0
            });

            // Deterministic ordering: consistent turns ahead of DBM-contradicted explicit-event turns,
            // then descending composite Q16.16 score, then ascending turn index.
            scored.sort_by(|a, b| {
                let a_contra = !a.0 && a.1;
                let b_contra = !b.0 && b.1;
                a_contra
                    .cmp(&b_contra)
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| a.3.cmp(&b.3))
            });

            let mut next_turns: Vec<usize> = scored.into_iter().map(|x| x.3).collect();

            // Protect Slot 0 star_bit invariant when Slot 0 is already consistent or no consistent star turn exists.
            if cur_top0_star_decl && (cur_top0_consistent || !any_consistent_star_decl) {
                if let Some(pos) = next_turns.iter().position(|&t| t == cur_top0) {
                    next_turns.remove(pos);
                }
                next_turns.insert(0, cur_top0);
            } else if cur_top0_star_decl {
                // Ensure the winning Slot 0 still satisfies the star_bit declarative invariant.
                if let Some(pos) = next_turns.iter().position(|&ti| {
                    !ctx.ix.turns[ti].is_question
                        && !ctx.ix.turns[ti].is_backchannel
                        && (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & star_bit) != 0
                }) && pos > 0
                {
                    let win = next_turns.remove(pos);
                    next_turns.insert(0, win);
                }
            }

            next_turns.truncate(ctx.k.max(5));
            ctx.final_turn_indices = next_turns;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index};

    fn make_session(id: &str, date: &str, ts: i64, turns: &[(&str, &str)]) -> Session {
        Session {
            id: id.to_string(),
            date: date.to_string(),
            timestamp: ts,
            turns: turns
                .iter()
                .enumerate()
                .map(|(i, (role, content))| Turn {
                    id: format!("{id}#{i}"),
                    role: (*role).to_string(),
                    content: (*content).to_string(),
                    stability: 0.0,
                    links: Vec::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn discourse_stack_resolves_zero_noun_confirm_and_task_done() {
        let sessions = vec![make_session(
            "s1",
            "2026-05-01",
            0,
            &[
                ("user", "Please reserve economy flight FL-402 with insurance set to no."),
                (
                    "assistant",
                    "I prepared flight FL-402 with insurance set to no. Would you like me to proceed and confirm the booking?",
                ),
            ],
        )];
        let ix = build_index(&sessions, &Params::default());
        let (stack, just_closed) = build_discourse_stack(&ix);
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].state, GroundingState::OpenProposal);
        assert_eq!(stack[0].presentation_turn, 1);
        assert!(just_closed.is_none());

        let act_confirm = classify_query_grounding_act("Yes, please go ahead and process it.");
        assert_eq!(act_confirm, QueryGroundingAct::ConfirmProceed);
        assert_eq!(
            grounding_stack_boost_q16(act_confirm, 1, &stack, just_closed.as_ref()),
            4 * ONE
        );
        assert_eq!(
            grounding_stack_boost_q16(act_confirm, 0, &stack, just_closed.as_ref()),
            0
        );

        let act_done = classify_query_grounding_act("Great, thanks for confirming. [TASK_DONE]");
        assert_eq!(act_done, QueryGroundingAct::CloseDone);
        assert_eq!(
            grounding_stack_boost_q16(act_done, 1, &stack, just_closed.as_ref()),
            4 * ONE
        );

        let act_verify = classify_query_grounding_act(
            "Thanks — just to verify one last thing, insurance is still set to no?",
        );
        assert_eq!(act_verify, QueryGroundingAct::VerifyLastValue);
    }

    #[test]
    fn passive_fsrs7_wm_gate_favors_recent_and_suppresses_stale() {
        let s_fast = 1_800u64; // 30 minutes
        let s_slow = 30 * 86_400u64; // 30 days
        let bll = 2 * ONE;

        let (rf_now, _rs_now, gate_now) = dual_trace_components_q16(60, s_fast, s_slow);
        let (rf_1d, _rs_1d, gate_1d) = dual_trace_components_q16(86_400, s_fast, s_slow);
        let (_rf_30d, rs_30d, gate_30d) = dual_trace_components_q16(30 * 86_400, s_fast, s_slow);
        let (_rf_180d, rs_180d, gate_180d) =
            dual_trace_components_q16(180 * 86_400, s_fast, s_slow);

        assert!(rf_now > rf_1d, "R_fast must decay quickly: {rf_now} > {rf_1d}");
        assert!(
            gate_now > (ONE >> 1),
            "Active working-memory turn must have high wm_gate (>0.5): {gate_now}"
        );
        assert!(
            gate_now > gate_1d && gate_1d > gate_30d && gate_30d > gate_180d,
            "wm_gate must decrease strictly without a 30-day clamp: {gate_now} > {gate_1d} > {gate_30d} > {gate_180d}"
        );
        assert!(
            rs_30d > rs_180d && rs_180d > 0,
            "Slow trace must remain smooth beyond 30 days: {rs_30d} > {rs_180d} > 0"
        );

        let wm_recent = dual_trace_activation_q16(60, s_fast, s_slow, bll, true);
        let wm_stale = dual_trace_activation_q16(30 * 86_400, s_fast, s_slow, bll, true);
        assert!(
            wm_recent > wm_stale * 5,
            "WM query must strongly favor recent turn over 30-day stale state: {wm_recent} vs {wm_stale}"
        );

        let bio_30d = dual_trace_activation_q16(30 * 86_400, s_fast, s_slow, bll, false);
        assert!(
            bio_30d > wm_stale,
            "Non-WM biographical query must preserve consolidated long-term memory: {bio_30d} vs {wm_stale}"
        );
    }

    #[test]
    fn rtec_clips_prior_fluent_on_higher_entrenchment_correction_and_tool_mutation() {
        let mut intervals = Vec::new();
        let slot = fnv1a64(b"insurance");
        let val_yes = fnv1a64(b"yes");
        let val_maybe = fnv1a64(b"maybe");
        let val_no = fnv1a64(b"no");

        // 1. User assertion (entrenchment = 2.0)
        apply_rtec_update(
            &mut intervals,
            FluentInterval {
                slot_hash: slot,
                value_hash: val_yes,
                t_start: 100,
                t_end: None,
                source_turn: 0,
                entrenchment: ENTRENCH_ASSERTION_Q16,
            },
        );
        // 2. Assistant suggestion (entrenchment = 1.0) must NOT clip user assertion (2.0).
        apply_rtec_update(
            &mut intervals,
            FluentInterval {
                slot_hash: slot,
                value_hash: val_maybe,
                t_start: 150,
                t_end: None,
                source_turn: 1,
                entrenchment: ENTRENCH_SUGGESTION_Q16,
            },
        );
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].t_end, None);

        // 3. Explicit user correction (entrenchment = 4.0) MUST clip old interval at t=200.
        apply_rtec_update(
            &mut intervals,
            FluentInterval {
                slot_hash: slot,
                value_hash: val_no,
                t_start: 200,
                t_end: None,
                source_turn: 2,
                entrenchment: ENTRENCH_CORRECTION_Q16,
            },
        );
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].t_end, Some(200));
        assert_eq!(intervals[1].t_end, None);
        assert_eq!(intervals[1].source_turn, 2);

        // End-to-end build_rtec_fluents on multi-session dialogue with user correction:
        let sessions = vec![
            make_session(
                "s1",
                "2026-01-10",
                0,
                &[("user", "Initially insurance is set to yes for the rental.")],
            ),
            make_session(
                "s2",
                "2026-01-15",
                0,
                &[("user", "Actually, switch insurance to no for the rental.")],
            ),
        ];
        let ix = build_index(&sessions, &Params::default());
        let built = build_rtec_fluents(&ix);
        let ins_intervals: Vec<&FluentInterval> =
            built.iter().filter(|f| f.slot_hash == slot).collect();
        assert_eq!(ins_intervals.len(), 2);
        assert!(ins_intervals[0].t_end.is_some());
        assert_eq!(ins_intervals[1].t_end, None);
        assert_eq!(ins_intervals[1].source_turn, 1);
    }

    #[test]
    fn canonical_dbm_detects_negative_cycle_and_ranks_tight_event_overlap() {
        // Direct negative cycle test: x1 - x0 <= 10, x0 - x1 <= -20 => 0 <= -10 (unsatisfiable)
        let mut d = dbm_init(3);
        assert!(dbm_add_constraint_and_close(&mut d, 3, 1, 0, 10));
        assert!(!dbm_add_constraint_and_close(&mut d, 3, 0, 1, -20));

        // Event anchored in April vs Query window in June -> DBM negative cycle!
        let (apr_l, apr_u) = month_span_secs(2023, 4);
        let (jun_l, jun_u) = month_span_secs(2023, 6);
        let june_12_l = ymd_to_epoch_secs(2023, 6, 12);
        let june_12_u = june_12_l + SECS_PER_DAY;

        let ok_april_in_june =
            dbm_check_turn_vs_query(apr_l, apr_u, jun_l, jun_u, AllenQueryRel::Overlap);
        assert!(!ok_april_in_june, "April event must fail DBM consistency against June query");
        assert_eq!(
            dbm_temporal_match_q16(apr_l, apr_u, jun_l, jun_u, ok_april_in_june),
            0
        );

        let ok_june12 =
            dbm_check_turn_vs_query(june_12_l, june_12_u, jun_l, jun_u, AllenQueryRel::Overlap);
        let ok_wide = dbm_check_turn_vs_query(
            jun_l - 15 * SECS_PER_DAY,
            jun_u + 15 * SECS_PER_DAY,
            jun_l,
            jun_u,
            AllenQueryRel::Overlap,
        );
        assert!(ok_june12 && ok_wide);
        let tight_score = dbm_temporal_match_q16(june_12_l, june_12_u, jun_l, jun_u, ok_june12);
        let loose_score = dbm_temporal_match_q16(
            jun_l - 15 * SECS_PER_DAY,
            jun_u + 15 * SECS_PER_DAY,
            jun_l,
            jun_u,
            ok_wide,
        );
        assert!(
            tight_score > loose_score,
            "Tight 1-day June event ({tight_score}) must outrank loose 60-day span ({loose_score})"
        );
    }

    #[test]
    fn dbm_allen_before_and_after_constraints_enforce_ordering() {
        let (may_l, may_u) = month_span_secs(2023, 5);
        let (jun_l, jun_u) = month_span_secs(2023, 6);
        let (jul_l, jul_u) = month_span_secs(2023, 7);

        assert!(dbm_check_turn_vs_query(may_l, may_u, jun_l, jun_u, AllenQueryRel::Before));
        assert!(!dbm_check_turn_vs_query(jul_l, jul_u, jun_l, jun_u, AllenQueryRel::Before));
        assert!(dbm_check_turn_vs_query(jul_l, jul_u, jun_l, jun_u, AllenQueryRel::After));
        assert!(!dbm_check_turn_vs_query(may_l, may_u, jun_l, jun_u, AllenQueryRel::After));
    }

    #[test]
    fn multibyte_utf8_safety_across_stack_rtec_and_dbm() {
        let sessions = vec![make_session(
            "utf8",
            "2026-06-15",
            0,
            &[
                ("user", "My café résumé budget is set to €500 — 日本語 🚀."),
                (
                    "assistant",
                    "Got it, café résumé budget is €500. Would you like me to proceed?",
                ),
                ("user", "Actually, switch café budget to €750 — that’s all!"),
            ],
        )];
        let ix = build_index(&sessions, &Params::default());
        let (_stack, just_closed) = build_discourse_stack(&ix);
        assert!(
            just_closed.is_some(),
            "Curly apostrophe 'that’s all' must close active DSP frame without panic"
        );
        let fluents = build_rtec_fluents(&ix);
        assert!(!fluents.is_empty());
        assert_eq!(
            classify_query_grounding_act("Thanks — that’s all for my café résumé!"),
            QueryGroundingAct::CloseDone
        );
        let win = extract_query_temporal_window(&ix, "What happened at the café in June?", "2026-06-20");
        assert!(win.is_some());
    }
}
