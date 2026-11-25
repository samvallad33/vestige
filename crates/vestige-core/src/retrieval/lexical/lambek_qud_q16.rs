//! Wave 5: Lambek Categorial Grammar / Question-Under-Discussion (QUD) Type-Slash Transfer (`Q16.16`).
//!
//! Models every query as an unsaturated Lambek functor `S / X` seeking a typed filler `X`
//! (Location, Temporal, Quantity, Person/Agent, Causal, Procedural, or Attribute/Category),
//! using 100% pure-integer `Q16.16` arithmetic, closed-class grammar, and existing structural
//! `TurnIx` signals (zero vectors, zero LLMs, zero hand-typed topic lists):
//!
//! 1. **Lambek Elimination (`(S / X) · X -> S`)**: declarative turns supplying the exact missing
//!    type `X` cancel the QUD functor and receive positive `Q16.16` elimination mass.
//! 2. **Echo Non-Cancellation (`(S / X) · (S / X)`)**: pure questions and discourse-closure echoes
//!    repeating query tokens without supplying `X` fail to reduce to `S` and incur a penalty.
//! 3. **2-Hop Lambek Composition (`(S / Y) · (Y / X) -> S / X`)**: when a query-matched bridge turn
//!    introduces a corpus-rare carrier entity `Y` (`entities` / `coref_entities`), QUD demand
//!    transfers across `Y` to declarative turns carrying both `Y` and the target filler `X`.

use crate::retrieval::fusion::aspect_coverage_q16::aspect_ic_q16;
use crate::retrieval::lexical::tier_bridge_q16::detect_single_subject_speaker;
use crate::retrieval::seating::habitus_energy_q16::is_discourse_closure_or_meta_echo;
use crate::retrieval::{DetHashSet, Index, Query, QueryIntent, QueryPlan, raw_tokens, turn_is_superseded};

/// Fixed-point unit `1.0` in `Q16.16`.
pub(in crate::retrieval) const Q16_ONE: i64 = 65_536;
/// Elimination reward when a declarative candidate cancels the exact QUD slash type `X`.
const ELIM_EXACT_FILLER_Q16: i64 = 49_152;
/// Additional reward when the candidate matches the single queried subject speaker.
const ELIM_SPEAKER_ALIGN_Q16: i64 = 16_384;
/// Penalty when a candidate repeats query tokens as an unsaturated question or discourse echo.
const ECHO_NON_CANCEL_PENALTY_Q16: i64 = 40_960;
/// Penalty when a candidate is a superseded turn on an affirmative QUD.
const SUPERSEDED_PENALTY_Q16: i64 = 49_152;
/// Cap on 2-hop Lambek composition transfer (`(S / Y) · (Y / X)`) into a filler-bearing turn.
const BRIDGE_TRANSFER_CAP_Q16: i64 = 32_768;

/// Closed-class Lambek target category `X` sought by functor `S / X`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) enum QudSlotType {
    /// Spatial / location target (`where`).
    Location,
    /// Temporal target (`when`, `what date/year/time`).
    Temporal,
    /// Quantitative / numeric target (`how many`, `how much`, `how long`, `how old`, rate/percentage).
    Quantity,
    /// Person / named agent target (`who`, `whose`).
    PersonOrAgent,
    /// Causal / explanatory target (`why`).
    Causal,
    /// Procedural / directive target (`how should/do/can`).
    Procedural,
    /// Specific named entity, attribute, or category target (`what`, `which`).
    AttributeOrCategory,
    /// Unconstrained declarative query.
    General,
}

/// Parsed Lambek QUD functor `S / X` for a single query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::retrieval) struct LambekQudFrame {
    /// Target missing category `X`.
    pub(in crate::retrieval) slot: QudSlotType,
    /// True when the query is an open interrogative requiring a concrete filler `X`.
    pub(in crate::retrieval) requires_filler: bool,
    /// Optional single subject speaker named by the query (named-speaker dialogues).
    pub(in crate::retrieval) subject_speaker: Option<String>,
    /// Bitmask of non-speaker exact query groups (`0..64`).
    pub(in crate::retrieval) topic_group_mask: u64,
    /// Corpus-rare focal term/entity ids (`s_df <= max(3, ns/6)`) anchoring Lambek composition.
    pub(in crate::retrieval) focal_carrier_ids: Vec<u32>,
}

/// Per-candidate pure-integer `Q16.16` Lambek QUD resolution score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::retrieval) struct LambekTurnScore {
    /// Turn index in `Index::turns`.
    pub(in crate::retrieval) turn_idx: usize,
    /// Direct Lambek elimination score `(S / X) · X -> S`.
    pub(in crate::retrieval) elimination_q16: i64,
    /// Whether the turn structurally supplies the missing filler type `X`.
    pub(in crate::retrieval) filler_satisfied: bool,
    /// Penalty applied to unsaturated echoes / pure questions / discourse closures.
    pub(in crate::retrieval) q_echo_penalty_q16: i64,
    /// 2-hop composition transfer `(S / Y) · (Y / X)` from rare carrier bridge turns.
    pub(in crate::retrieval) bridge_transfer_q16: i64,
    /// Net deterministic `Q16.16` Lambek score (`elimination + bridge_transfer - echo_penalty`).
    pub(in crate::retrieval) net_q16: i64,
}

/// Context bundle for [`apply_lambek_qud_slot0_tiebreak`].
pub(in crate::retrieval) struct LambekSlot0Ctx<'a> {
    pub(in crate::retrieval) ix: &'a Index,
    pub(in crate::retrieval) q: &'a Query,
    pub(in crate::retrieval) plan: &'a QueryPlan,
    pub(in crate::retrieval) eff_it: &'a QueryIntent,
    pub(in crate::retrieval) t_exact_mask: &'a [u64],
    pub(in crate::retrieval) t_scene: &'a [f32],
    pub(in crate::retrieval) t_bridge: &'a [f32],
    pub(in crate::retrieval) k: usize,
}

/// Classify the closed-class Lambek QUD slash type `X` from query syntax and intent.
/// hardcode-scan: closed-class (WH-interrogatives, auxiliaries, and measure pronouns only).
pub(in crate::retrieval) fn classify_qud_slot_type(query: &str, eff_it: &QueryIntent) -> QudSlotType {
    let ql = query.trim().to_ascii_lowercase();
    if eff_it.sniper_how_many
        || ql.starts_with("how many ")
        || ql.starts_with("how much ")
        || ql.starts_with("how long ")
        || ql.starts_with("how old ")
        || ql.starts_with("how often ")
        || ql.starts_with("how far ")
        || ql.starts_with("what percentage ")
        || ql.starts_with("what probability ")
        || ql.starts_with("what is the amount ")
        || ql.starts_with("what is the estimated ")
        || ql.starts_with("what is the daily ")
    {
        return QudSlotType::Quantity;
    }
    if eff_it.sniper_when
        || eff_it.temporal
        || ql.starts_with("when ")
        || ql.starts_with("what date ")
        || ql.starts_with("what year ")
        || ql.starts_with("what month ")
        || ql.starts_with("what time ")
    {
        return QudSlotType::Temporal;
    }
    if ql.starts_with("where ") || ql.contains(" where ") {
        return QudSlotType::Location;
    }
    if ql.starts_with("who ") || ql.starts_with("whose ") || ql.starts_with("whom ") {
        return QudSlotType::PersonOrAgent;
    }
    if ql.starts_with("why ") || ql.contains(" why ") {
        return QudSlotType::Causal;
    }
    if eff_it.task_under_rules
        || ql.starts_with("how should ")
        || ql.starts_with("how do i ")
        || ql.starts_with("how can i ")
        || ql.starts_with("how would ")
    {
        return QudSlotType::Procedural;
    }
    if ql.starts_with("what ") || ql.starts_with("which ") || ql.ends_with('?') {
        return QudSlotType::AttributeOrCategory;
    }
    QudSlotType::General
}

/// Build the query's Lambek QUD functor `S / X`.
pub(in crate::retrieval) fn parse_lambek_qud_frame(
    ix: &Index,
    q: &Query,
    plan: &QueryPlan,
    eff_it: &QueryIntent,
) -> LambekQudFrame {
    let slot = classify_qud_slot_type(&q.text, eff_it);
    let subject_speaker = detect_single_subject_speaker(ix, plan, &q.text);
    let sp_ref = subject_speaker.as_deref();

    let mut topic_group_mask: u64 = 0;
    for qt in &plan.terms {
        if qt.exact_hit
            && let Some(g) = qt.group
            && g < 64
        {
            let gnm = plan.groups.get(g).map(String::as_str).unwrap_or("");
            if !sp_ref.is_some_and(|sp| gnm.eq_ignore_ascii_case(sp)) {
                topic_group_mask |= 1u64 << g;
            }
        }
    }

    let ns_u32 = (ix.session_turns.len() as u32).max(1);
    let rare_cap = (ns_u32 / 6).max(3);
    let mut focal_carrier_ids: Vec<u32> = Vec::new();
    for &fid in &plan.focal_ids {
        let df = ix.s_stats.df.get(&fid).copied().unwrap_or(0);
        if (1..=rare_cap).contains(&df) && !focal_carrier_ids.contains(&fid) {
            focal_carrier_ids.push(fid);
        }
    }
    for &rid in &plan.rare_entity_ids {
        if !focal_carrier_ids.contains(&rid) {
            focal_carrier_ids.push(rid);
        }
    }

    let requires_filler = !matches!(slot, QudSlotType::General);
    LambekQudFrame {
        slot,
        requires_filler,
        subject_speaker,
        topic_group_mask,
        focal_carrier_ids,
    }
}

/// Check whether turn `ti` structurally supplies the missing Lambek filler category `X`.
///
/// Returns `(filler_satisfied, filler_strength_q16)` using purely structural `TurnIx` flags
/// and closed-class grammar (zero open-class topic dictionaries).
pub(in crate::retrieval) fn turn_satisfies_qud_filler(
    ix: &Index,
    ti: usize,
    slot: QudSlotType,
    q_text: &str,
) -> (bool, i64) {
    let Some(tr) = ix.turns.get(ti) else {
        return (false, 0);
    };
    if tr.is_backchannel
        || tr.is_short_question
        || is_discourse_closure_or_meta_echo(ix, ti, q_text)
    {
        return (false, 0);
    }
    if tr.is_pure_info_question && !tr.is_standing_policy && !tr.is_personal_constraint {
        return (false, 0);
    }
    let asserted = (tr.modality_mask & 0x01) != 0 || tr.modality_mask == 0;
    let modality_scale_num: i64 = if asserted { 4 } else { 2 };

    let base_q16 = match slot {
        QudSlotType::Location => {
            if tr.has_spatial && (tr.form_midcap || !tr.entities.is_empty()) {
                ELIM_EXACT_FILLER_Q16 + 8_192
            } else if tr.has_spatial || (!tr.entities.is_empty() && !tr.is_question) {
                ELIM_EXACT_FILLER_Q16
            } else {
                0
            }
        }
        QudSlotType::Temporal => {
            if !tr.resolved_timex.is_empty() || tr.form_clock {
                ELIM_EXACT_FILLER_Q16 + 8_192
            } else if tr.has_temporal || tr.form_digit {
                ELIM_EXACT_FILLER_Q16
            } else {
                0
            }
        }
        QudSlotType::Quantity => {
            if tr.form_digit && tr.has_numeric {
                ELIM_EXACT_FILLER_Q16 + 12_288
            } else if tr.form_digit || tr.has_numeric {
                ELIM_EXACT_FILLER_Q16
            } else {
                0
            }
        }
        QudSlotType::PersonOrAgent => {
            if tr.slot_entities_clean >= 1 || !tr.entities.is_empty() {
                ELIM_EXACT_FILLER_Q16 + 8_192
            } else if tr.form_midcap && !tr.is_question {
                ELIM_EXACT_FILLER_Q16
            } else {
                0
            }
        }
        QudSlotType::Causal => {
            if tr.has_causal {
                ELIM_EXACT_FILLER_Q16 + 8_192
            } else if !tr.is_question && tr.content.len() >= 6 {
                ELIM_EXACT_FILLER_Q16 >> 1
            } else {
                0
            }
        }
        QudSlotType::Procedural => {
            if tr.is_standing_policy || tr.list_item_count >= 2 {
                ELIM_EXACT_FILLER_Q16 + 8_192
            } else if tr.is_personal_constraint || (!tr.is_question && tr.content.len() >= 8) {
                ELIM_EXACT_FILLER_Q16
            } else {
                0
            }
        }
        QudSlotType::AttributeOrCategory => {
            if !tr.is_question
                && (tr.slot_entities_clean >= 1
                    || tr.form_digit
                    || tr.form_handle
                    || !tr.entities.is_empty()
                    || tr.has_prose_disclosure
                    || tr.is_personal_constraint)
            {
                ELIM_EXACT_FILLER_Q16
            } else if !tr.is_question && tr.content.len() >= 5 {
                ELIM_EXACT_FILLER_Q16 >> 1
            } else {
                0
            }
        }
        QudSlotType::General => {
            if !tr.is_question && tr.content.len() >= 4 {
                ELIM_EXACT_FILLER_Q16 >> 1
            } else {
                0
            }
        }
    };

    if base_q16 <= 0 {
        return (false, 0);
    }
    let scaled = (base_q16.saturating_mul(modality_scale_num)) >> 2;
    (scaled >= (ELIM_EXACT_FILLER_Q16 >> 1), scaled)
}

/// Score a candidate turn pool under pure-integer `Q16.16` Lambek QUD Elimination +
/// 2-Hop Composition Transfer (`(S / Y) · (Y / X) -> S / X`).
pub(in crate::retrieval) fn score_lambek_qud_candidates_q16(
    ctx: &LambekSlot0Ctx<'_>,
    cand_turns: &[usize],
) -> Vec<LambekTurnScore> {
    if cand_turns.is_empty() {
        return Vec::new();
    }
    let ix = ctx.ix;
    let frame = parse_lambek_qud_frame(ix, ctx.q, ctx.plan, ctx.eff_it);
    let ns_u32 = (ix.session_turns.len() as u32).max(1);
    let max_bridge_df = (ns_u32 / 5).max(3);

    // Phase 1: Collect rare carrier entities `Y` introduced by query-matched bridge turns
    // (`(S / Y)` premises) so QUD demand can transfer to declarative filler turns `(Y / X)`.
    let mut bridge_carriers: Vec<(u32, i64, usize)> = Vec::new();
    let mut seen_carriers: DetHashSet<u32> = DetHashSet::default();
    for &ti in cand_turns {
        let Some(tr) = ix.turns.get(ti) else {
            continue;
        };
        let exact_hits =
            (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & frame.topic_group_mask).count_ones();
        if exact_hits == 0 {
            continue;
        }
        for &eid in tr.entities.iter().chain(tr.coref_entities.iter()) {
            let df = ix.s_stats.df.get(&eid).copied().unwrap_or(0);
            if (1..=max_bridge_df).contains(&df) && seen_carriers.insert(eid) {
                let ic_q16 = aspect_ic_q16(df, ns_u32) as i64;
                let transfer_w = (ic_q16 >> 1).min(BRIDGE_TRANSFER_CAP_Q16);
                bridge_carriers.push((eid, transfer_w, tr.session));
            }
        }
    }

    // Phase 2: Evaluate each candidate for direct elimination, echo non-cancellation,
    // speaker ownership, and 2-hop Lambek composition transfer.
    let q_toks_len = raw_tokens(&ctx.q.text).len();
    let mut out = Vec::with_capacity(cand_turns.len());
    for &ti in cand_turns {
        let Some(tr) = ix.turns.get(ti) else {
            continue;
        };
        let (filler_ok, mut elim_q16) = turn_satisfies_qud_filler(ix, ti, frame.slot, &ctx.q.text);

        let exact_cov =
            (ctx.t_exact_mask.get(ti).copied().unwrap_or(0) & frame.topic_group_mask).count_ones()
                as i64;
        if filler_ok && exact_cov > 0 {
            elim_q16 = elim_q16.saturating_add((exact_cov.min(4)) * (Q16_ONE >> 3));
        }

        if let Some(ref sp) = frame.subject_speaker
            && tr.speaker.eq_ignore_ascii_case(sp)
            && filler_ok
        {
            elim_q16 = elim_q16.saturating_add(ELIM_SPEAKER_ALIGN_Q16);
        }

        let mut penalty_q16: i64 = 0;
        if tr.is_pure_info_question || tr.is_short_question || tr.is_backchannel {
            penalty_q16 = penalty_q16.saturating_add(ECHO_NON_CANCEL_PENALTY_Q16);
        } else if is_discourse_closure_or_meta_echo(ix, ti, &ctx.q.text) {
            penalty_q16 = penalty_q16.saturating_add(ECHO_NON_CANCEL_PENALTY_Q16);
        } else if frame.requires_filler && !filler_ok && q_toks_len >= 4 {
            penalty_q16 = penalty_q16.saturating_add(ECHO_NON_CANCEL_PENALTY_Q16 >> 2);
        }
        if turn_is_superseded(ix, ti) {
            penalty_q16 = penalty_q16.saturating_add(SUPERSEDED_PENALTY_Q16);
        }

        let mut bridge_q16: i64 = 0;
        if filler_ok {
            for &(cid, cw, src_sess) in &bridge_carriers {
                if src_sess != tr.session
                    && (tr.entities.contains(&cid)
                        || tr.coref_entities.contains(&cid)
                        || tr.content.contains(&cid))
                {
                    bridge_q16 = bridge_q16.saturating_add(cw);
                }
            }
            if ctx.t_bridge.get(ti).copied().unwrap_or(0.0) > 0.0 {
                bridge_q16 = bridge_q16.saturating_add(Q16_ONE >> 3);
            }
            bridge_q16 = bridge_q16.min(BRIDGE_TRANSFER_CAP_Q16);
        }

        let net_q16 = elim_q16
            .saturating_add(bridge_q16)
            .saturating_sub(penalty_q16);
        out.push(LambekTurnScore {
            turn_idx: ti,
            elimination_q16: elim_q16,
            filler_satisfied: filler_ok,
            q_echo_penalty_q16: penalty_q16,
            bridge_transfer_q16: bridge_q16,
            net_q16,
        });
    }
    out
}

/// Conservative Slot-0 QUD-Closure tie-breaker (`(S / X) · X -> S`).
///
/// Never overrides a protected standing policy, preference turn, event-sequence query, or an
/// incumbent Slot-0 turn that already cancels the QUD filler `X`. Only promotes a Top-5
/// declarative filler turn that beats an unsaturated Slot-0 turn by a decisive Lambek margin
/// (`>= 0.50 * Q16`) while preserving exact topical anchor coverage.
pub(in crate::retrieval) fn apply_lambek_qud_slot0_tiebreak(
    ctx: &LambekSlot0Ctx<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
) {
    if final_turn_indices.len() < 2
        || ctx.eff_it.is_event_sequence
        || ctx.eff_it.task_under_rules
        || ctx.eff_it.preference
        || ctx.eff_it.temporal
        || ctx.eff_it.sniper_when
        || ctx.eff_it.multi_hop
        || ctx.ix.has_tool_calls
    {
        return;
    }
    let t0 = final_turn_indices[0];
    if ctx.ix.turns[t0].is_standing_policy || ctx.ix.turns[t0].is_personal_constraint {
        return;
    }

    let frame = parse_lambek_qud_frame(ctx.ix, ctx.q, ctx.plan, ctx.eff_it);
    if !frame.requires_filler {
        return;
    }

    let window = final_turn_indices.len().min(5);
    let scores = score_lambek_qud_candidates_q16(ctx, &final_turn_indices[..window]);
    let Some(s0) = scores.first().copied() else {
        return;
    };
    if s0.filler_satisfied && s0.q_echo_penalty_q16 == 0 {
        return;
    }

    let ex0 =
        (ctx.t_exact_mask.get(t0).copied().unwrap_or(0) & frame.topic_group_mask).count_ones();
    let t0_unsaturated_echo = ctx.ix.turns[t0].is_backchannel
        || is_discourse_closure_or_meta_echo(ctx.ix, t0, &ctx.q.text)
        || (ctx.ix.turns[t0].is_pure_info_question && s0.q_echo_penalty_q16 >= ECHO_NON_CANCEL_PENALTY_Q16);

    let t0_sess = ctx.ix.turns[t0].session;
    let t0_scene = ctx.t_scene.get(t0).copied().unwrap_or(0.0);
    let t0_substantive_user =
        !t0_unsaturated_echo && (ctx.ix.turns[t0].field == 0 || ctx.eff_it.assistant_attribution);

    let mut best: Option<(usize, i64, u32)> = None;
    for (slot, sc) in scores.iter().enumerate().skip(1) {
        if !sc.filler_satisfied
            || sc.q_echo_penalty_q16 > 0
            || ctx.ix.turns[sc.turn_idx].is_standing_policy
        {
            continue;
        }
        // Protect substantive declarative Slot-0 turns from cross-session displacement:
        // only allow cross-session Slot-0 swap when incumbent Slot 0 is an unsaturated echo/question
        // or assistant non-target turn; otherwise restrict to decisive same-session tie-break.
        if t0_substantive_user {
            if ctx.ix.turns[sc.turn_idx].session != t0_sess {
                continue;
            }
            let sc_scene = ctx.t_scene.get(sc.turn_idx).copied().unwrap_or(0.0);
            if t0_scene > 0.0 && sc_scene < 0.80 * t0_scene {
                continue;
            }
        }
        let ex_c = (ctx.t_exact_mask.get(sc.turn_idx).copied().unwrap_or(0)
            & frame.topic_group_mask)
            .count_ones();
        if ex_c == 0 || (ex_c < ex0 && !t0_unsaturated_echo) {
            continue;
        }
        let min_margin = if t0_substantive_user {
            Q16_ONE
        } else {
            Q16_ONE >> 1
        };
        let margin = sc.net_q16.saturating_sub(s0.net_q16);
        if margin < min_margin {
            continue;
        }
        let key = (sc.net_q16, ex_c);
        let improves = match best {
            None => true,
            Some((bslot, bn, be)) => {
                key > (bn, be)
                    || (key == (bn, be)
                        && ctx.t_scene.get(sc.turn_idx).copied().unwrap_or(0.0)
                            > ctx
                                .t_scene
                                .get(final_turn_indices[bslot])
                                .copied()
                                .unwrap_or(0.0))
            }
        };
        if improves {
            best = Some((slot, sc.net_q16, ex_c));
        }
    }

    if let Some((win_slot, _, _)) = best {
        let win_ti = final_turn_indices.remove(win_slot);
        final_turn_indices.insert(0, win_ti);
        let win_si = ctx.ix.turns[win_ti].session;
        let orig_slen = final_session_indices.len();
        if orig_slen > 0 && final_session_indices[0] != win_si {
            if let Some(spos) = final_session_indices.iter().position(|&s| s == win_si) {
                final_session_indices.remove(spos);
            }
            final_session_indices.insert(0, win_si);
            final_session_indices.truncate(orig_slen.max(ctx.k));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{Params, Session, Turn, build_index, plan_query};

    fn make_session(id: &str, ts: i64, turns: &[(&str, &str)]) -> Session {
        Session {
            id: id.to_string(),
            date: String::new(),
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
    fn lambek_classifies_closed_class_qud_slots_accurately() {
        let it = QueryIntent::default();
        assert_eq!(
            classify_qud_slot_type("How much did the estate tax rate reach?", &it),
            QudSlotType::Quantity
        );
        assert_eq!(
            classify_qud_slot_type("Where did Caroline attend the pottery workshop?", &it),
            QudSlotType::Location
        );
        assert_eq!(
            classify_qud_slot_type("When was the patent filing scheduled?", &it),
            QudSlotType::Temporal
        );
        assert_eq!(
            classify_qud_slot_type("Who mentored Melanie on watercolor painting?", &it),
            QudSlotType::PersonOrAgent
        );
        assert_eq!(
            classify_qud_slot_type("Why did the worker reject the equity offer?", &it),
            QudSlotType::Causal
        );
    }

    #[test]
    fn lambek_elimination_beats_question_echo_and_seats_filler_at_slot0() {
        let p = Params::default();
        let sessions = vec![
            make_session(
                "s1",
                1_700_000_000,
                &[
                    ("user", "What is the estimated estate tax rate in Montserrat?"),
                    ("assistant", "Let me check your earlier legal notes."),
                ],
            ),
            make_session(
                "s2",
                1_700_100_000,
                &[
                    (
                        "user",
                        "My attorney confirmed the estate tax rate in Montserrat is 15% above $200,000.",
                    ),
                    ("assistant", "Noted the 15% estate tax rate."),
                ],
            ),
        ];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "How much is the estimated estate tax rate in Montserrat?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let eff_it = plan.intent.clone();
        let t_exact_mask = vec![0b111u64; ix.turns.len()];
        let t_scene = vec![10.0, 2.0, 9.5, 4.0];
        let t_bridge = vec![0.0; ix.turns.len()];
        let lctx = LambekSlot0Ctx {
            ix: &ix,
            q: &q,
            plan: &plan,
            eff_it: &eff_it,
            t_exact_mask: &t_exact_mask,
            t_scene: &t_scene,
            t_bridge: &t_bridge,
            k: 5,
        };

        let mut final_turns = vec![0usize, 2, 3, 1];
        let mut final_sessions = vec![0usize, 1];
        apply_lambek_qud_slot0_tiebreak(&lctx, &mut final_turns, &mut final_sessions);
        assert_eq!(final_turns[0], 2, "numeric declarative filler must cancel S/Quantity");
        assert_eq!(final_sessions[0], 1, "session rank 1 must sync with filler winner");
    }

    #[test]
    fn lambek_two_hop_composition_transfers_qud_across_rare_entity_bridge() {
        let p = Params::default();
        let sessions = vec![
            make_session(
                "s1",
                1_700_000_000,
                &[(
                    "user",
                    "I hired Valerio Consulting to review our compliance architecture.",
                )],
            ),
            make_session(
                "s2",
                1_700_100_000,
                &[(
                    "user",
                    "Valerio Consulting charged $4,500 for the complete compliance audit.",
                )],
            ),
            make_session(
                "s3",
                1_700_200_000,
                &[("user", "We also talked about lunch menus today.")],
            ),
        ];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "How much did the compliance review cost?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let eff_it = plan.intent.clone();
        let t_exact_mask = vec![0b11u64, 0b01u64, 0u64];
        let t_scene = vec![10.0, 8.0, 1.0];
        let t_bridge = vec![0.0; ix.turns.len()];
        let lctx = LambekSlot0Ctx {
            ix: &ix,
            q: &q,
            plan: &plan,
            eff_it: &eff_it,
            t_exact_mask: &t_exact_mask,
            t_scene: &t_scene,
            t_bridge: &t_bridge,
            k: 5,
        };
        let scores = score_lambek_qud_candidates_q16(&lctx, &[0, 1, 2]);
        let s_bridge = scores.iter().find(|s| s.turn_idx == 0).unwrap();
        let s_answer = scores.iter().find(|s| s.turn_idx == 1).unwrap();
        let frame = parse_lambek_qud_frame(&ix, &q, &plan, &eff_it);
        assert!(frame.requires_filler);
        assert!(frame.focal_carrier_ids.len() <= ix.vocab.names.len());
        assert!(s_answer.filler_satisfied);
        assert!(s_answer.elimination_q16 > 0);
        assert!(
            s_answer.bridge_transfer_q16 > 0,
            "rare entity Valerio Consulting should carry Lambek composition transfer across sessions"
        );
        assert!(s_answer.net_q16 > s_bridge.net_q16);
    }

    #[test]
    fn lambek_protects_slot0_when_incumbent_already_cancels_qud() {
        let p = Params::default();
        let sessions = vec![
            make_session(
                "s1",
                1_700_000_000,
                &[("user", "The API rate limit is 1200 calls per day.")],
            ),
            make_session(
                "s2",
                1_700_100_000,
                &[("user", "Our secondary backup quota is 500 calls per day.")],
            ),
        ];
        let ix = build_index(&sessions, &p);
        let q = Query {
            text: "How many calls per day is the API rate limit?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &p);
        let eff_it = plan.intent.clone();
        let t_exact_mask = vec![0b111u64, 0b011u64];
        let t_scene = vec![12.0, 8.0];
        let t_bridge = vec![0.0, 0.0];
        let lctx = LambekSlot0Ctx {
            ix: &ix,
            q: &q,
            plan: &plan,
            eff_it: &eff_it,
            t_exact_mask: &t_exact_mask,
            t_scene: &t_scene,
            t_bridge: &t_bridge,
            k: 5,
        };
        let mut final_turns = vec![0usize, 1];
        let mut final_sessions = vec![0usize, 1];
        apply_lambek_qud_slot0_tiebreak(&lctx, &mut final_turns, &mut final_sessions);
        assert_eq!(final_turns[0], 0, "incumbent Slot-0 filler must remain untouched");
    }
}
