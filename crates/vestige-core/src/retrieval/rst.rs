//! Issue #810: Stede Rhetorical Structure Theory (RST) Nucleus vs. Satellite
//! classification in pure integer Q16.16 arithmetic. Zero floats, zero word lists.
//!
//! Distinguishes high-payload declarative nuclei (rare content assertions,
//! entities, numbers, temporal/spatial/causal markers, standing policies) from
//! conversational satellites (backchannels, short pleasantries/replies,
//! bare framing/echo questions without novel assertions).

use super::index::Index;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RstRole {
    Nucleus,
    Satellite,
}

/// Structurally classify turn `ti` as an RST `Nucleus` or `Satellite`.
/// Strictly list-free: reads only DF statistics and structural turn flags.
pub(super) fn classify_rst_role(ix: &Index, ti: usize) -> RstRole {
    let t = &ix.turns[ti];
    if t.has_numeric
        || t.has_temporal
        || t.has_spatial
        || t.has_causal
        || t.form_digit
        || t.form_clock
        || t.is_standing_policy
        || t.is_personal_constraint
    {
        return RstRole::Nucleus;
    }
    if t.is_backchannel || (t.is_short_reply && !t.answers_question) {
        return RstRole::Satellite;
    }

    let n_docs = (ix.t_own_stats.n as u32).max(1);
    let rare_df_cap = (n_docs / 4).max(3);
    let mut rare_payload_cnt = 0u32;
    for &cid in &t.content {
        let df = ix.t_own_stats.df.get(&cid).copied().unwrap_or(0);
        if df > 0 && df <= rare_df_cap {
            rare_payload_cnt += 1;
        }
    }

    if t.is_question && rare_payload_cnt <= 1 {
        return RstRole::Satellite;
    }
    if rare_payload_cnt == 0 {
        return RstRole::Satellite;
    }
    RstRole::Nucleus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rst_role_enum_distinct() {
        assert_ne!(RstRole::Nucleus, RstRole::Satellite);
    }
}
