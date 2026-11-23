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

/// Closed-class English weekday index (`0 = Sunday .. 6 = Saturday`, matching Unix epoch math).
/// hardcode-scan: closed-class (seven calendar weekday names only).
fn parse_weekday(tok: &str) -> Option<i64> {
    match tok {
        "sunday" | "sundays" => Some(0),
        "monday" | "mondays" => Some(1),
        "tuesday" | "tuesdays" => Some(2),
        "wednesday" | "wednesdays" => Some(3),
        "thursday" | "thursdays" => Some(4),
        "friday" | "fridays" => Some(5),
        "saturday" | "saturdays" => Some(6),
        _ => None,
    }
}

/// Pure integer Howard Hinnant civil-from-days conversion (`epoch_ts -> (year, month_name, day)`).
/// hardcode-scan: closed-class (twelve Gregorian calendar month names only).
pub(in crate::retrieval) fn epoch_ts_to_ymd_tokens(ts: i64) -> (i32, &'static str, u32) {
    const MONTHS: [&str; 12] = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ];
    let z = ts.div_euclid(SECS_PER_DAY) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as usize;
    let year = (y + i64::from(m <= 2)) as i32;
    (year, MONTHS[(m - 1).min(11)], d)
}

/// Compile ALL closed-class relative-date expressions (`"yesterday"`, `"the day before"`,
/// `"last Saturday"`, `"two Saturdays ago"`, `"last weekend"`, `"a few days ago"`,
/// `"two months ago"`, `"next week"`) against `anchor_ts` at ingest into half-open `[v_s, v_e)` spans.
/// hardcode-scan: closed-class (deictic direction words, numerals, weekdays, calendar units only).
pub(in crate::retrieval) fn compile_ingest_timex_anchors(
    tokens: &[&str],
    anchor_ts: i64,
) -> Vec<(i64, i64)> {
    let mut out: Vec<(i64, i64)> = Vec::new();
    if anchor_ts <= 0 || tokens.is_empty() {
        return out;
    }
    let low: Vec<String> = tokens.iter().map(|t| t.to_ascii_lowercase()).collect();
    // 1970-01-01 was Thursday (dow = 4 in 0=Sun..6=Sat).
    let anchor_day = anchor_ts.div_euclid(SECS_PER_DAY);
    let anchor_dow = (anchor_day + 4).rem_euclid(7);
    let day_start = anchor_day * SECS_PER_DAY;

    let mut push_iv = |iv: (i64, i64)| {
        if !out.contains(&iv) {
            out.push(iv);
        }
    };

    for i in 0..low.len() {
        if low[i] == "yesterday" {
            let s = day_start - SECS_PER_DAY;
            push_iv((s, s + SECS_PER_DAY));
            continue;
        }
        if low[i] == "tomorrow" {
            let s = day_start + SECS_PER_DAY;
            push_iv((s, s + SECS_PER_DAY));
            continue;
        }
        // "the day before" / "day before yesterday"
        if low[i] == "day" && i + 1 < low.len() && low[i + 1] == "before" {
            let offset = if i + 2 < low.len() && low[i + 2] == "yesterday" {
                2
            } else {
                1
            };
            let s = day_start - offset * SECS_PER_DAY;
            push_iv((s, s + SECS_PER_DAY));
            continue;
        }
        // "last weekend" / "next weekend" / "this weekend"
        if low[i] == "weekend" && i > 0 {
            if low[i - 1] == "last" {
                let days_back = if anchor_dow == 0 { 1 } else { anchor_dow + 1 };
                let sat = day_start - days_back * SECS_PER_DAY;
                push_iv((sat, sat + 2 * SECS_PER_DAY));
                continue;
            }
            if low[i - 1] == "next" || low[i - 1] == "this" {
                let days_fwd = (6 - anchor_dow).rem_euclid(7);
                let sat = day_start + days_fwd * SECS_PER_DAY;
                push_iv((sat, sat + 2 * SECS_PER_DAY));
                continue;
            }
        }
        // Weekday patterns: "last Saturday", "next Friday", "two Saturdays ago"
        if let Some(target_dow) = parse_weekday(&low[i]) {
            if i > 0 && low[i - 1] == "last" {
                let mut diff = (anchor_dow - target_dow).rem_euclid(7);
                if diff == 0 {
                    diff = 7;
                }
                let s = day_start - diff * SECS_PER_DAY;
                push_iv((s, s + SECS_PER_DAY));
                continue;
            }
            if i > 0 && low[i - 1] == "next" {
                let mut diff = (target_dow - anchor_dow).rem_euclid(7);
                if diff == 0 {
                    diff = 7;
                }
                let s = day_start + diff * SECS_PER_DAY;
                push_iv((s, s + SECS_PER_DAY));
                continue;
            }
            if i > 0 && i + 1 < low.len()
                && let Some(n) = parse_small_numeral(&low[i - 1])
                && matches!(low[i + 1].as_str(), "ago" | "before" | "earlier")
            {
                let mut diff = (anchor_dow - target_dow).rem_euclid(7);
                if diff == 0 {
                    diff = 7;
                }
                let total_days = diff + ((n.saturating_sub(1)) as i64) * 7;
                let s = day_start - total_days * SECS_PER_DAY;
                push_iv((s, s + SECS_PER_DAY));
                continue;
            }
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
            push_iv((s, s + u_secs));
            continue;
        }
        if i > 0 && low[i - 1] == "next" {
            let s = anchor_ts + u_secs;
            push_iv((s, s + u_secs));
            continue;
        }
        // Pattern 2: "<n> <unit> ago|before|after|later" or "a few/couple <unit> ago"
        if i > 0 && i + 1 < low.len() {
            let n_opt = parse_small_numeral(&low[i - 1]).or_else(|| {
                if low[i - 1] == "few" {
                    Some(3)
                } else if low[i - 1] == "of" && i >= 2 && low[i - 2] == "couple" {
                    Some(2)
                } else {
                    None
                }
            });
            if let Some(n) = n_opt {
                let sign = match low[i + 1].as_str() {
                    "ago" | "before" | "earlier" => Some(-1i64),
                    "after" | "later" => Some(1i64),
                    _ => None,
                };
                if let Some(sg) = sign {
                    let center = anchor_ts + sg * (n as i64) * u_secs;
                    push_iv((center, center + u_secs));
                }
            }
        }
    }
    out
}

/// Resolve closed-class relative-date phrases (`yesterday`, `last week`, `two Saturdays ago`,
/// `two months ago`, `three days after`) against anchor timestamp `anchor_ts` into a half-open `[v_s, v_e)` interval.
pub(in crate::retrieval) fn resolve_relative_interval(
    tokens: &[&str],
    anchor_ts: i64,
) -> Option<(i64, i64)> {
    compile_ingest_timex_anchors(tokens, anchor_ts).into_iter().next()
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

    #[test]
    fn ingest_timex_compiler_resolves_weekdays_and_civil_dates() {
        // 2023-05-08 00:00:00 UTC (Monday) = 19485 days since 1970-01-01.
        let may8_2023 = 19_485 * SECS_PER_DAY;
        assert_eq!(epoch_ts_to_ymd_tokens(may8_2023), (2023, "may", 8));
        // "last Saturday" from Monday May 8 -> Saturday May 6, 2023.
        let iv_sat = compile_ingest_timex_anchors(&["ran", "last", "Saturday"], may8_2023);
        assert_eq!(iv_sat.len(), 1);
        assert_eq!(epoch_ts_to_ymd_tokens(iv_sat[0].0), (2023, "may", 6));
        // "two Saturdays ago" from Monday May 8 -> Saturday April 29, 2023.
        let iv_2sat = compile_ingest_timex_anchors(&["two", "Saturdays", "ago"], may8_2023);
        assert_eq!(iv_2sat.len(), 1);
        assert_eq!(epoch_ts_to_ymd_tokens(iv_2sat[0].0), (2023, "april", 29));
    }
}
