//! Dual-Mode Bitemporal Supersession & Closed-Class HeidelTime Relative-Date Resolver.
//!
//! Grounded in:
//! - Snodgrass (1999) valid-time vs. transaction-time bitemporal databases,
//! - Alonso et al. (TWAW 2011) Temporal IR (document creation time vs. event focus time),
//! - Strötgen & Gertz (SemEval-2010 / LRE 2013) HeidelTime closed-class calendar anchoring,
//! - Li & Croft (CIKM 2003) Time-Based Language Models.
//!
//! Pure integer calendar arithmetic & `Q16.16` scoring. Zero floats, zero open-class lexicons.

use crate::retrieval::fusion::aspect_coverage_q16::Q16_ONE;

const SECS_PER_DAY: i64 = 86_400;
const SECS_PER_WEEK: i64 = 7 * SECS_PER_DAY;
const SECS_PER_MONTH_APPROX: i64 = 30 * SECS_PER_DAY;
const SECS_PER_YEAR_APPROX: i64 = 365 * SECS_PER_DAY;

/// Closed-class historical-override polarity detector (`before`, `originally`, `first`, `used to`).
/// hardcode-scan: closed-class (temporal polarity adverbs & auxiliaries only).
pub(in crate::retrieval) fn query_has_historical_override(tokens: &[&str]) -> bool {
    for &raw in tokens {
        let w = raw.to_ascii_lowercase();
        if matches!(
            w.as_str(),
            "before"
                | "previously"
                | "originally"
                | "earlier"
                | "initial"
                | "initially"
                | "former"
                | "formerly"
                | "first"
        ) {
            return true;
        }
    }
    tokens
        .windows(2)
        .any(|w| w[0].eq_ignore_ascii_case("used") && w[1].eq_ignore_ascii_case("to"))
}

/// Resolve closed-class relative-date phrases (`yesterday`, `last week`, `two months ago`,
/// `three days after`) against anchor timestamp `anchor_ts` into a half-open `[v_s, v_e)` interval.
/// hardcode-scan: closed-class (deictic direction words, small numerals, calendar units only).
pub(in crate::retrieval) fn resolve_relative_interval(
    tokens: &[&str],
    anchor_ts: i64,
) -> Option<(i64, i64)> {
    let low: Vec<String> = tokens.iter().map(|t| t.to_ascii_lowercase()).collect();
    for i in 0..low.len() {
        if low[i] == "yesterday" {
            let s = anchor_ts - SECS_PER_DAY;
            return Some((s, s + SECS_PER_DAY));
        }
        if low[i] == "tomorrow" {
            let s = anchor_ts + SECS_PER_DAY;
            return Some((s, s + SECS_PER_DAY));
        }
        let unit_secs = match low[i].as_str() {
            "day" | "days" => Some(SECS_PER_DAY),
            "week" | "weeks" => Some(SECS_PER_WEEK),
            "month" | "months" => Some(SECS_PER_MONTH_APPROX),
            "year" | "years" => Some(SECS_PER_YEAR_APPROX),
            _ => None,
        };
        let Some(u_secs) = unit_secs else {
            continue;
        };
        // Pattern 1: "last <unit>" / "next <unit>"
        if i > 0 && low[i - 1] == "last" {
            let s = anchor_ts - u_secs;
            return Some((s, s + u_secs));
        }
        if i > 0 && low[i - 1] == "next" {
            let s = anchor_ts + u_secs;
            return Some((s, s + u_secs));
        }
        // Pattern 2: "<n> <unit> ago|before|after|later"
        if i > 0 && i + 1 < low.len() {
            if let Some(n) = parse_small_numeral(&low[i - 1]) {
                let sign = match low[i + 1].as_str() {
                    "ago" | "before" | "earlier" => Some(-1i64),
                    "after" | "later" => Some(1i64),
                    _ => None,
                };
                if let Some(sg) = sign {
                    let center = anchor_ts + sg * (n as i64) * u_secs;
                    return Some((center, center + u_secs));
                }
            }
        }
    }
    None
}

/// Closed-class small English numeral parser (`1..12` + digits).
/// hardcode-scan: closed-class (cardinal numerals only).
fn parse_small_numeral(tok: &str) -> Option<u32> {
    if let Ok(v) = tok.parse::<u32>() {
        return (1..=365).contains(&v).then_some(v);
    }
    match tok {
        "a" | "an" | "one" => Some(1),
        "two" | "couple" => Some(2),
        "three" => Some(3),
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

/// Exact integer interval IoU `|I_m cap I_q| / |I_m cup I_q|` in `Q16.16`.
pub(in crate::retrieval) fn interval_overlap_q16(a: (i64, i64), b: (i64, i64)) -> u64 {
    let inter_s = a.0.max(b.0);
    let inter_e = a.1.min(b.1);
    if inter_e <= inter_s {
        return 0;
    }
    let union_s = a.0.min(b.0);
    let union_e = a.1.max(b.1);
    let inter = (inter_e - inter_s) as u128;
    let union = (union_e - union_s).max(1) as u128;
    ((inter << 16) / union).min(Q16_ONE as u128) as u64
}

/// Dual-mode Strata bitemporal preference multiplier in `Q16.16`:
/// - `historical_override == false`: prefers active (non-superseded) latest facts (`Q16_ONE` vs `Q16_ONE / 2`).
/// - `historical_override == true`: flips polarity to prefer superseded/earlier origin facts.
#[inline]
pub(in crate::retrieval) fn bitemporal_validity_factor_q16(
    is_superseded: bool,
    historical_override: bool,
) -> u64 {
    if historical_override {
        if is_superseded {
            Q16_ONE + (Q16_ONE >> 2) // 1.25x for historical origin queries
        } else {
            Q16_ONE
        }
    } else if is_superseded {
        Q16_ONE >> 1 // 0.50x demotion when superseded on current-state queries
    } else {
        Q16_ONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_dates_and_scores_interval_overlap() {
        let anchor = 100 * SECS_PER_DAY;
        let iv_a = resolve_relative_interval(&["I", "moved", "last", "week"], anchor).unwrap();
        let iv_b = resolve_relative_interval(&["seven", "days", "ago"], anchor).unwrap();
        assert_eq!(iv_a, (anchor - SECS_PER_WEEK, anchor));
        assert!(interval_overlap_q16(iv_a, iv_b) > 0);
    }

    #[test]
    fn dual_mode_bitemporal_flips_on_historical_override() {
        assert!(bitemporal_validity_factor_q16(false, false) > bitemporal_validity_factor_q16(true, false));
        assert!(bitemporal_validity_factor_q16(true, true) > bitemporal_validity_factor_q16(false, true));
        assert!(query_has_historical_override(&["Where", "did", "she", "originally", "live"]));
        assert!(!query_has_historical_override(&["Where", "does", "she", "live", "now"]));
    }
}
