//! Date parsing, calendar tokens, inline dates, and deictic resolution.

use crate::retrieval::*;


// =========================================================================
// DATE PARSING + DEICTIC TIME COMPILER
// =========================================================================

/// Parse the free-form date strings used by real conversation logs into Unix seconds (UTC).
pub fn parse_date(s: &str) -> i64 {
    let st = s.trim();
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(st) {
        return dt.timestamp();
    }
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(st, fmt) {
            return dt.and_utc().timestamp();
        }
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(st, "%Y-%m-%d") {
        return d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc().timestamp()).unwrap_or(0);
    }
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ':' { c.to_ascii_lowercase() } else { ' ' })
        .collect();
    let toks: Vec<&str> = cleaned.split_whitespace().collect();
    let (mut year, mut month, mut hour, mut minute) = (None, None, 0u32, 0u32);
    let mut nums: Vec<u32> = Vec::new();
    for (i, t) in toks.iter().enumerate() {
        if let Some((h, m)) = t.split_once(':') {
            let mut hh: u32 = h.parse().unwrap_or(0);
            minute = m.parse().unwrap_or(0);
            match toks.get(i + 1) {
                Some(&"pm") if hh < 12 => hh += 12,
                Some(&"am") if hh == 12 => hh = 0,
                _ => {}
            }
            hour = hh;
        } else if t.len() == 4 && t.chars().all(|c| c.is_ascii_digit()) {
            year = t.parse::<i32>().ok();
        } else if t.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(n) = t.parse::<u32>() {
                nums.push(n);
            }
        } else if t.len() >= 3 {
            if let Some(m) = MONTHS.iter().position(|mn| mn.starts_with(&t[..3]) && mn.starts_with(t.get(..t.len().min(mn.len())).unwrap_or(""))) {
                month = Some(m as u32 + 1);
            }
        }
    }
    let (m, d) = match month {
        Some(m) => (m, nums.first().copied().unwrap_or(1)),
        None => (nums.first().copied().unwrap_or(1), nums.get(1).copied().unwrap_or(1)),
    };
    let Some(y) = year else { return 0 };
    chrono::NaiveDate::from_ymd_opt(y, m, d)
        .and_then(|date| date.and_hms_opt(hour.min(23), minute.min(59), 0))
        .map(|dt| dt.and_utc().timestamp())
        .unwrap_or(0)
}

pub(in crate::retrieval) fn season_code(month: u32) -> &'static str {
    match month {
        3..=5 => "SP",
        6..=8 => "SU",
        9..=11 => "FA",
        _ => "WI",
    }
}

pub(in crate::retrieval) fn calendar_tokens(ts: i64) -> Vec<String> {
    use chrono::Datelike;
    let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) else { return Vec::new() };
    let d = dt.date_naive();
    let sc = season_code(d.month());
    let sy = if d.month() == 12 { d.year() + 1 } else { d.year() };
    vec![
        d.day().to_string(),
        MONTHS[d.month0() as usize].to_string(),
        d.year().to_string(),
        WEEKDAYS[d.weekday().num_days_from_monday() as usize].to_string(),
        format!("__DATE__{}_{:02}_{:02}", d.year(), d.month(), d.day()),
        format!("__YM__{}_{:02}", d.year(), d.month()),
        format!("__MONTH__{:02}", d.month()),
        format!("__MD__{:02}_{:02}", d.month(), d.day()),
        format!("__{}__{:02}", month_band(d.day()), d.month()),
        format!("__YEAR__{}", d.year()),
        format!("__SEASON__{sc}"),
        format!("__SY__{}_{sc}", d.year()),
        format!("__SY__{sy}_{sc}"),
    ]
}

pub(in crate::retrieval) fn month_band(day: u32) -> &'static str {
    if day <= 10 { "EARLY" } else if day >= 21 { "LATE" } else { "MID" }
}

pub(in crate::retrieval) fn month_index(tok: &str) -> Option<(u32, bool)> {
    match tok {
        "january" | "jan" => Some((1, false)),
        "february" | "feb" => Some((2, false)),
        "march" => Some((3, true)),
        "mar" => Some((3, false)),
        "april" | "apr" => Some((4, false)),
        "may" => Some((5, true)),
        "june" | "jun" => Some((6, false)),
        "july" | "jul" => Some((7, false)),
        "august" | "aug" => Some((8, false)),
        "september" | "sep" | "sept" => Some((9, false)),
        "october" | "oct" => Some((10, false)),
        "november" | "nov" => Some((11, false)),
        "december" | "dec" => Some((12, false)),
        _ => None,
    }
}

pub(in crate::retrieval) fn parse_day_token(tok: &str) -> Option<u32> {
    let stripped = tok
        .strip_suffix("st")
        .or_else(|| tok.strip_suffix("nd"))
        .or_else(|| tok.strip_suffix("rd"))
        .or_else(|| tok.strip_suffix("th"))
        .unwrap_or(tok);
    if let Ok(n) = stripped.parse::<u32>() {
        if (1..=31).contains(&n) {
            return Some(n);
        }
    }
    match tok {
        "first" => Some(1),
        "second" => Some(2),
        "third" => Some(3),
        "fourth" | "forth" => Some(4),
        "fifth" => Some(5),
        "sixth" => Some(6),
        "seventh" => Some(7),
        "eighth" => Some(8),
        "ninth" => Some(9),
        "tenth" => Some(10),
        "eleventh" => Some(11),
        "twelfth" => Some(12),
        "thirteenth" => Some(13),
        "fourteenth" => Some(14),
        "fifteenth" => Some(15),
        "sixteenth" => Some(16),
        "seventeenth" => Some(17),
        "eighteenth" => Some(18),
        "nineteenth" => Some(19),
        "twentieth" => Some(20),
        "thirtieth" => Some(30),
        _ => None,
    }
}

/// Extract explicit calendar dates written directly inside prose/queries
/// (e.g. "1 May, 2022", "May 1st, 2022", "the 15th of March", "in May", "summer of 2023", "2022-05-01").
pub(in crate::retrieval) fn inline_date_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lower = text.to_lowercase();
    // Match ISO YYYY-MM-DD or YYYY/MM/DD
    for w in lower.split_whitespace() {
        let trimmed = w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '/');
        if trimmed.len() == 10 {
            let b = trimmed.as_bytes();
            if (b.get(4) == Some(&b'-') && b.get(7) == Some(&b'-'))
                || (b.get(4) == Some(&b'/') && b.get(7) == Some(&b'/'))
            {
                let iso = trimmed.replace('/', "-");
                if let Ok(d) = chrono::NaiveDate::parse_from_str(&iso, "%Y-%m-%d") {
                    if let Some(dt) = d.and_hms_opt(0, 0, 0) {
                        out.extend(calendar_tokens(dt.and_utc().timestamp()));
                    }
                }
            }
        }
    }
    let orig_toks: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    let toks: Vec<String> = orig_toks.iter().map(|s| s.to_lowercase()).collect();

    // Closed-class temporal scaffold. Not a domain vocabulary: these are the
    // function words a calendar expression uses to attach a month name.
    // hardcode-scan: closed-class (temporal prepositions)
    const TEMP_PREPS: &[&str] = &[
        "in", "on", "of", "during", "since", "before", "after", "until", "by", "from", "through",
        "around", "this", "last", "next", "past", "every", "each", "late", "early", "mid", "middle",
    ];

    for i in 0..toks.len() {
        // A 4-digit token in the calendar range is a year. No domain blocklist.
        if toks[i].len() == 4 && toks[i].chars().all(|c| c.is_ascii_digit()) {
            if let Ok(y) = toks[i].parse::<i32>() {
                if (1900..=2099).contains(&y) {
                    out.push(format!("__YEAR__{y}"));
                }
            }
        }

        // Seasonal anchors ("summer of 2023", "in the winter", "last spring")
        let season = match toks[i].as_str() {
            "summer" => Some(("SU", false)),
            "winter" => Some(("WI", false)),
            "autumn" => Some(("FA", false)),
            "spring" => Some(("SP", true)),
            "fall" => Some(("FA", true)),
            _ => None,
        };
        if let Some((sc, ambig_season)) = season {
            let prev1 = i.checked_sub(1).map(|k| toks[k].as_str()).unwrap_or("");
            let prev2 = i.checked_sub(2).map(|k| toks[k].as_str()).unwrap_or("");
            let y_after = if toks.get(i + 1).map(String::as_str) == Some("of") {
                toks.get(i + 2)
            } else {
                toks.get(i + 1)
            }
            .and_then(|s| (s.len() == 4).then(|| s.parse::<i32>().ok()).flatten())
            .filter(|&y| (1900..=2099).contains(&y));

            if let Some(y) = y_after {
                out.push(format!("__SY__{y}_{sc}"));
                out.push(format!("__SEASON__{sc}"));
                out.push(format!("__YEAR__{y}"));
            } else if !ambig_season
                || matches!(prev1, "this" | "last" | "next" | "past" | "every" | "early" | "late" | "mid")
                || (prev1 == "the" && matches!(prev2, "in" | "during" | "since" | "before" | "after"))
            {
                out.push(format!("__SEASON__{sc}"));
            }
        }

        let Some((m, ambig_month)) = month_index(&toks[i]) else { continue };
        let band = i
            .checked_sub(1)
            .and_then(|k| match toks[k].as_str() {
                "early" | "earlier" => Some("EARLY"),
                "late" | "later" => Some("LATE"),
                "mid" | "middle" => Some("MID"),
                "of" if k >= 1 => match toks[k - 1].as_str() {
                    "beginning" | "start" => Some("EARLY"),
                    "middle" => Some("MID"),
                    "end" => Some("LATE"),
                    _ => None,
                },
                _ => None,
            });
        if let Some(b) = band {
            out.push(format!("__{b}__{m:02}"));
            out.push(format!("__MONTH__{m:02}"));
        }

        // "<day> [of] <month> [of] <year>" OR "<month> [the] <day> [of] <year>"
        let prev_num = i.checked_sub(1).and_then(|k| {
            if toks[k] == "of" && k >= 1 {
                parse_day_token(&toks[k - 1])
            } else {
                parse_day_token(&toks[k])
            }
        });
        let (next_num, day_end_idx) = if toks.get(i + 1).map(String::as_str) == Some("the") {
            (toks.get(i + 2).and_then(|s| parse_day_token(s)), i + 2)
        } else {
            (toks.get(i + 1).and_then(|s| parse_day_token(s)), i + 1)
        };
        let year_at = |idx: usize| -> Option<i32> {
            let cand = if toks.get(idx).map(String::as_str) == Some("of") {
                toks.get(idx + 1)
            } else {
                toks.get(idx)
            };
            cand.and_then(|s| (s.len() == 4).then(|| s.parse::<i32>().ok()).flatten())
                .filter(|&y| (1900..=2099).contains(&y))
        };
        let year_after_month = year_at(i + 1);
        let year_after_next = year_at(day_end_idx + 1);

        if let (Some(day), Some(y)) = (prev_num, year_after_month) {
            if let Some(dt) = chrono::NaiveDate::from_ymd_opt(y, m, day).and_then(|d| d.and_hms_opt(0, 0, 0)) {
                out.extend(calendar_tokens(dt.and_utc().timestamp()));
                continue;
            }
        }
        if let (Some(day), Some(y)) = (next_num, year_after_next) {
            if let Some(dt) = chrono::NaiveDate::from_ymd_opt(y, m, day).and_then(|d| d.and_hms_opt(0, 0, 0)) {
                out.extend(calendar_tokens(dt.and_utc().timestamp()));
                continue;
            }
        }
        if let Some(y) = year_after_month {
            out.push(format!("__YM__{y}_{m:02}"));
            out.push(format!("__MONTH__{m:02}"));
            out.push(format!("__YEAR__{y}"));
        } else if let Some(day) = prev_num.or(next_num) {
            out.push(format!("__MD__{m:02}_{day:02}"));
            out.push(format!("__MONTH__{m:02}"));
        } else if band.is_none() {
            // Bare month in prose/query ("in May", "during July", "last October").
            // Guard ambiguous "may"/"march" with preposition or capitalization + non-modal context.
            let prev1 = i.checked_sub(1).map(|k| toks[k].as_str()).unwrap_or("");
            let is_cap = orig_toks
                .get(i)
                .and_then(|s| s.chars().next())
                .is_some_and(|c| c.is_ascii_uppercase());
            if !ambig_month || TEMP_PREPS.contains(&prev1) || (is_cap && i > 0) {
                out.push(format!("__MONTH__{m:02}"));
            }
        }
    }
    out
}

pub(in crate::retrieval) fn parse_qty_word(q: &str) -> i64 {
    match q {
        "a" | "an" | "one" => 1,
        "two" | "couple" => 2,
        "three" | "few" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        other => other.parse().unwrap_or(1),
    }
}

/// Resolve relative time expressions against the session timestamp into (token, relative_weight).
pub(in crate::retrieval) fn deictic_weighted_tokens(raw: &[String], ts: i64) -> Vec<(String, f32)> {
    if ts == 0 {
        return Vec::new();
    }
    const DAY: i64 = 86_400;
    // (offset_seconds, weight, is_coarse_window)
    let mut specs: Vec<(i64, f32, bool)> = Vec::new();
    for (i, w) in raw.iter().enumerate() {
        let prev = if i > 0 { raw[i - 1].as_str() } else { "" };
        let next = raw.get(i + 1).map(String::as_str).unwrap_or("");
        match w.as_str() {
            "yesterday" => specs.push((-DAY, 1.0, false)),
            "night" if prev == "last" => specs.push((-DAY, 1.0, false)),
            "before" | "earlier" if prev == "day" => specs.push((-DAY, 0.9, false)),
            "before" if prev == "week" => specs.push((-7 * DAY, 0.9, true)),
            "before" if prev == "month" => specs.push((-30 * DAY, 0.9, true)),
            "before" if prev == "year" => specs.push((-365 * DAY, 0.9, true)),
            "after" | "later" if prev == "day" && (i < 2 || raw[i - 2] == "the" || raw[i - 2] == "a") => {
                specs.push((DAY, 0.9, false));
            }
            "after" | "following" if prev == "week" && (i < 2 || raw[i - 2] == "the" || raw[i - 2] == "a") => {
                specs.push((7 * DAY, 0.9, true));
            }
            "tomorrow" => specs.push((DAY, 1.0, false)),
            "today" | "tonight" => specs.push((0, 1.0, false)),
            "week" if matches!(prev, "last" | "past" | "previous") => specs.push((-7 * DAY, 1.0, true)),
            "week" if matches!(prev, "next" | "coming" | "following") => specs.push((7 * DAY, 1.0, true)),
            "week" if prev == "this" && i >= 2 && matches!(raw[i - 2].as_str(), "earlier" | "early") => {
                specs.push((-2 * DAY, 0.9, true));
            }
            "weekend" if matches!(prev, "last" | "this" | "past" | "previous") => specs.push((-2 * DAY, 1.0, true)),
            "weekend" if matches!(prev, "next" | "coming") => specs.push((5 * DAY, 0.9, true)),
            "month" if matches!(prev, "last" | "past" | "previous") => {
                specs.push((-30 * DAY, 1.0, true));
                specs.push((-15 * DAY, 0.85, false));
            }
            "month" if prev == "this" && i >= 2 && matches!(raw[i - 2].as_str(), "earlier" | "early") => {
                specs.push((-7 * DAY, 0.9, true));
            }
            "month" if matches!(prev, "next" | "coming" | "following") => specs.push((30 * DAY, 1.0, true)),
            "year" if matches!(prev, "last" | "past" | "previous") => specs.push((-365 * DAY, 1.0, true)),
            "year" if matches!(prev, "next" | "coming" | "following") => specs.push((365 * DAY, 1.0, true)),
            "ago" | "earlier" | "later" => {
                if matches!(prev, "days" | "weeks" | "months" | "years" | "day" | "week" | "month" | "year") {
                    let q = if i >= 3 && raw[i - 2] == "of" {
                        raw[i - 3].as_str()
                    } else if i >= 2 {
                        raw[i - 2].as_str()
                    } else {
                        ""
                    };
                    let n = parse_qty_word(q);
                    let (unit, coarse) = match prev.trim_end_matches('s') {
                        "day" => (1, n > 3),
                        "week" => (7, true),
                        "month" => (30, true),
                        _ => (365, true),
                    };
                    let dir = if w == "later" { 1 } else { -1 };
                    if n > 0 {
                        specs.push((dir * n * unit * DAY, 1.0, coarse));
                        if q == "few" && unit >= 7 {
                            specs.push((dir * (n - 1) * unit * DAY, 0.65, true));
                            specs.push((dir * (n + 1) * unit * DAY, 0.65, true));
                        }
                    }
                }
            }
            "days" | "weeks" | "months" | "years" if !matches!(next, "ago" | "earlier" | "later") => {
                // Handle "for 3 years now", "past 2 weeks", "last 3 months"
                let prep = if i >= 2 { raw[i - 2].as_str() } else { "" };
                if matches!(prep, "for" | "past" | "last" | "in") || next == "now" {
                    let n = parse_qty_word(prev);
                    let unit = match w.trim_end_matches('s') {
                        "day" => 1,
                        "week" => 7,
                        "month" => 30,
                        _ => 365,
                    };
                    if n > 0 && n <= 120 {
                        specs.push((-n * unit * DAY, 0.9, true));
                    }
                }
            }
            _ => {
                if matches!(prev, "last" | "this" | "next" | "past" | "previous" | "coming" | "following")
                    && WEEKDAYS.contains(&w.as_str())
                {
                    use chrono::Datelike;
                    if let Some(dt) = chrono::DateTime::from_timestamp(ts, 0) {
                        let today = dt.weekday().num_days_from_monday() as i64;
                        let target = WEEKDAYS.iter().position(|d| d == w).unwrap_or(0) as i64;
                        let is_future = matches!(prev, "next" | "coming" | "following");
                        let is_strict_past = matches!(prev, "last" | "past" | "previous");
                        let raw_back = (today - target + 7) % 7;
                        let raw_fwd = (target - today + 7) % 7;
                        let back = if is_strict_past && raw_back == 0 { 7 } else { raw_back };
                        let fwd = if is_future && raw_fwd == 0 { 7 } else { raw_fwd };
                        let delta = if is_future { fwd } else { -back };
                        specs.push((delta * DAY, 1.0, false));
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for (off, w, coarse) in specs {
        for tok in calendar_tokens(ts + off) {
            out.push((tok, w));
        }
        // Soft +-2d halo on coarse multi-week / multi-month expressions (handles Feb 28 / Mar 1 boundary).
        if coarse {
            for d_off in [-2 * DAY, -DAY, DAY, 2 * DAY] {
                for tok in calendar_tokens(ts + off + d_off) {
                    if tok.starts_with("__DATE__")
                        || tok.starts_with("__YM__")
                        || tok.starts_with("__MONTH__")
                        || tok.starts_with("__MD__")
                    {
                        out.push((tok, w * 0.45));
                    }
                }
            }
        }
    }
    out
}
