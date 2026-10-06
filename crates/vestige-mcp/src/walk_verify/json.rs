//! Python's JSON forms, byte for byte, and the two hashes built on them.
//!
//! The probe chain and the frozen protocol are hashed over the text Python's
//! `json.dumps(value, sort_keys=True, separators=(",", ":"))` writes: keys
//! sorted, no spaces, every character outside printable ASCII escaped as
//! `\uXXXX` (UTF-16 surrogate pairs above the BMP), floats as `repr(float)`.
//! [`canonical_json`] is that form, so a chain written here verifies in the
//! reference tool and the other way round.

use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

/// One probe-log entry, as it is hashed and as the report stores it.
pub(super) type Entry = Map<String, Value>;

/// The `prev` of the first entry of a chain.
pub(super) const ZERO_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `json.dumps(value, sort_keys=True, separators=(",", ":"))`, byte for byte.
pub(super) fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(number, out),
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => write_object(map, &[], out),
    }
}

fn sorted_keys<'a>(map: &'a Map<String, Value>, skip: &[&str]) -> Vec<&'a String> {
    let mut keys: Vec<&String> = map
        .keys()
        .filter(|key| !skip.contains(&key.as_str()))
        .collect();
    // Byte order of UTF-8 is code point order, which is how Python sorts.
    keys.sort();
    keys
}

fn write_object(map: &Map<String, Value>, skip: &[&str], out: &mut String) {
    out.push('{');
    for (index, key) in sorted_keys(map, skip).into_iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        write_string(key, out);
        out.push(':');
        if let Some(value) = map.get(key) {
            write_canonical(value, out);
        }
    }
    out.push('}');
}

/// Python's `ensure_ascii` string form: printable ASCII as is, the short
/// escapes, and `\uXXXX` (lowercase hex, surrogate pairs) for the rest.
fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn write_number(number: &Number, out: &mut String) {
    if let Some(value) = number.as_i64() {
        out.push_str(&value.to_string());
    } else if let Some(value) = number.as_u64() {
        out.push_str(&value.to_string());
    } else if let Some(value) = number.as_f64() {
        out.push_str(&float_repr(value));
    } else {
        out.push_str(&number.to_string());
    }
}

/// `repr(float)`: the shortest digits that round-trip, as a plain decimal
/// with at least one fractional digit, or with a signed two-digit exponent
/// below 1e-4 and from 1e16 up. A value that is not finite prints the way
/// `json.dumps` prints it; a JSON number never holds one.
pub(super) fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_string();
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return scientific;
    }
    if !(-4..16).contains(&exponent) {
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{sign}{first}{fraction}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        )
    } else if exponent < 0 {
        let zeros = "0".repeat(exponent.unsigned_abs() as usize - 1);
        format!("{sign}0.{zeros}{digits}")
    } else {
        let whole = exponent as usize + 1;
        if digits.len() <= whole {
            let zeros = "0".repeat(whole - digits.len());
            format!("{sign}{digits}{zeros}.0")
        } else {
            let (integer, fraction) = digits.split_at(whole);
            format!("{sign}{integer}.{fraction}")
        }
    }
}

/// `json.dumps(value, indent=1, sort_keys=True)`: the layout of the report.
pub(super) fn pretty_json(value: &Value) -> String {
    let mut out = String::new();
    write_pretty(value, 0, &mut out);
    out
}

fn write_pretty(value: &Value, depth: usize, out: &mut String) {
    let line = |depth: usize, out: &mut String| {
        out.push('\n');
        out.push_str(&" ".repeat(depth));
    };
    match value {
        Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                line(depth + 1, out);
                write_pretty(item, depth + 1, out);
            }
            line(depth, out);
            out.push(']');
        }
        Value::Object(map) if !map.is_empty() => {
            out.push('{');
            for (index, key) in sorted_keys(map, &[]).into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                line(depth + 1, out);
                write_string(key, out);
                out.push_str(": ");
                if let Some(value) = map.get(key) {
                    write_pretty(value, depth + 1, out);
                }
            }
            line(depth, out);
            out.push('}');
        }
        other => write_canonical(other, out),
    }
}

/// The hash of one entry: `sha256(prev + canonical JSON of the entry without
/// its "hash" field)`, hex.
pub(super) fn chain_hash(prev: &str, entry: &Entry) -> String {
    let mut body = String::from(prev);
    write_object(entry, &["hash"], &mut body);
    sha256_hex(body.as_bytes())
}

/// The hash of a frozen protocol: sha256 over the canonical JSON of its
/// fields, without the `sha256` and `memory` the report adds afterwards.
pub(super) fn protocol_hash(protocol: &Map<String, Value>) -> String {
    let mut body = String::new();
    write_object(protocol, &["sha256", "memory"], &mut body);
    sha256_hex(body.as_bytes())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use serde_json::json;

    fn entry(value: Value) -> Entry {
        value.as_object().cloned().expect("an object")
    }

    // Written by Python 3 with
    // json.dumps(entry, sort_keys=True, separators=(",", ":")) and
    // hashlib.sha256((prev + body).encode()).hexdigest().
    const PY_BODY_1: &str = r#"{"at":"2026-10-05T23:45:31+00:00","commit":"4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270","exit":1,"memory":null,"n":1,"oracle_said":"gave up after 6.88s (ConnectionError) \u007f\u0001 \u2028 end/","oracle_sha256":"725c8d23f3905746a09db5450859e50bde9a7569ff40e328f380ff1e917a533b","phase":"candidate","prev":"0000000000000000000000000000000000000000000000000000000000000000","subject":"Caf\u00e9 \u2014 \"quoted\" back\\slash \ud83d\ude00 tab\there","verdict":"bad"}"#;
    const PY_HASH_1: &str = "308763e1c81affd3566e016dd013789219fe38c904984b32ac59739d892bd285";
    const PY_BODY_2: &str = r#"{"at":"2026-10-05T23:45:56+00:00","commit":"742b13bdce + 2 of 4 changes","exit":-9,"memory":"mem-0000000000000729","n":2,"oracle_said":"","oracle_sha256":"725c8d23f3905746a09db5450859e50bde9a7569ff40e328f380ff1e917a533b","phase":"lines","prev":"308763e1c81affd3566e016dd013789219fe38c904984b32ac59739d892bd285","subject":"redis/asyncio/connection.py:296, redis/connection.py:843","verdict":"skip"}"#;
    const PY_HASH_2: &str = "d2ea4ac4658c38b0f5ef1f9ce06e102aa976c194f1f53deb9ca51db3853a01c3";
    // A `--flaky` entry: it also carries `runs` and `fails`.
    const PY_BODY_FLAKY: &str = r#"{"at":"2026-10-06T01:02:03+00:00","commit":"f7c1755d732a677e0eb05e74e46520115c087153","exit":1,"fails":7,"memory":"mem-0000000000000042","n":1,"oracle_said":"failed 7 of 20 runs, e.g. fail","oracle_sha256":"abababababababababababababababababababababababababababababababab","phase":"baseline","prev":"0000000000000000000000000000000000000000000000000000000000000000","runs":20,"subject":"base","verdict":"bad"}"#;
    const PY_HASH_FLAKY: &str = "bbf08225b5755228425ceb15fc2ad94b86fa75c5b553bec6b81f1b4f20dbd6e5";

    pub(in crate::walk_verify) fn first_entry() -> Entry {
        entry(json!({
            "commit": "4a6c2c0ff8fe5a2a6409cf18dc2bf2dd2a755270",
            "subject": "Caf\u{e9} \u{2014} \"quoted\" back\\slash \u{1F600} tab\there",
            "verdict": "bad",
            "exit": 1,
            "oracle_said": "gave up after 6.88s (ConnectionError) \u{7f}\u{1} \u{2028} end/",
            "oracle_sha256": "725c8d23f3905746a09db5450859e50bde9a7569ff40e328f380ff1e917a533b",
            "phase": "candidate",
            "at": "2026-10-05T23:45:31+00:00",
            "memory": null,
            "n": 1,
            "prev": ZERO_HASH,
        }))
    }

    pub(in crate::walk_verify) fn second_entry() -> Entry {
        entry(json!({
            "commit": "742b13bdce + 2 of 4 changes",
            "subject": "redis/asyncio/connection.py:296, redis/connection.py:843",
            "verdict": "skip",
            "exit": -9,
            "oracle_said": "",
            "oracle_sha256": "725c8d23f3905746a09db5450859e50bde9a7569ff40e328f380ff1e917a533b",
            "phase": "lines",
            "at": "2026-10-05T23:45:56+00:00",
            "memory": "mem-0000000000000729",
            "n": 2,
            "prev": PY_HASH_1,
        }))
    }

    pub(in crate::walk_verify) const FIRST_HASH: &str = PY_HASH_1;
    pub(in crate::walk_verify) const SECOND_HASH: &str = PY_HASH_2;

    fn flaky_entry() -> Entry {
        entry(json!({
            "commit": "f7c1755d732a677e0eb05e74e46520115c087153",
            "subject": "base",
            "verdict": "bad",
            "exit": 1,
            "oracle_said": "failed 7 of 20 runs, e.g. fail",
            "oracle_sha256": "ab".repeat(32),
            "phase": "baseline",
            "at": "2026-10-06T01:02:03+00:00",
            "runs": 20,
            "fails": 7,
            "memory": "mem-0000000000000042",
            "n": 1,
            "prev": ZERO_HASH,
        }))
    }

    #[test]
    fn canonical_json_is_pythons_sorted_compact_ascii_form() {
        assert_eq!(canonical_json(&Value::Object(first_entry())), PY_BODY_1);
        assert_eq!(canonical_json(&Value::Object(second_entry())), PY_BODY_2);
        assert_eq!(canonical_json(&Value::Object(flaky_entry())), PY_BODY_FLAKY);
        let mixed = json!({
            "z": [1, 2.5, 1e16, 1e-05, 0.1, 100.0, true, false, null, {"b": 1, "a": "\u{8}\u{c}\n\r"}],
            "a": {},
            "m": [],
            "big": 12345678901234567890u64,
            "neg": -0.0,
        });
        assert_eq!(
            canonical_json(&mixed),
            r#"{"a":{},"big":12345678901234567890,"m":[],"neg":-0.0,"z":[1,2.5,1e+16,1e-05,0.1,100.0,true,false,null,{"a":"\b\f\n\r","b":1}]}"#
        );
    }

    #[test]
    fn chain_hash_matches_the_python_vectors_and_ignores_the_hash_field() {
        assert_eq!(chain_hash(ZERO_HASH, &first_entry()), PY_HASH_1);
        assert_eq!(chain_hash(PY_HASH_1, &second_entry()), PY_HASH_2);
        assert_eq!(chain_hash(ZERO_HASH, &flaky_entry()), PY_HASH_FLAKY);
        let mut hashed = first_entry();
        hashed.insert("hash".to_string(), json!(PY_HASH_1));
        assert_eq!(chain_hash(ZERO_HASH, &hashed), PY_HASH_1);
        let mut changed = first_entry();
        changed.insert("verdict".to_string(), json!("good"));
        assert_ne!(chain_hash(ZERO_HASH, &changed), PY_HASH_1);
        let mut recounted = flaky_entry();
        recounted.insert("fails".to_string(), json!(0));
        assert_ne!(chain_hash(ZERO_HASH, &recounted), PY_HASH_FLAKY);
    }

    #[test]
    fn floats_print_as_python_repr() {
        for (value, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (0.01, "0.01"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (123456.789, "123456.789"),
            (1.5e300, "1.5e+300"),
            (-2.5, "-2.5"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (0.30000000000000004, "0.30000000000000004"),
            // (0 + 0.5) / (10 + 1) and the Fisher p of 15/50 against 0/50.
            (0.5 / 11.0, "0.045454545454545456"),
            (8.884673390211095e-06, "8.884673390211095e-06"),
        ] {
            assert_eq!(float_repr(value), want);
        }
        assert_eq!(float_repr(f64::NAN), "NaN");
        assert_eq!(float_repr(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn the_report_layout_is_pythons_indent_one() {
        let value = json!({"b": [1, {"x": [], "a": {}}], "a": "\u{e9}", "c": {"k": null}});
        assert_eq!(
            pretty_json(&value),
            "{\n \"a\": \"\\u00e9\",\n \"b\": [\n  1,\n  {\n   \"a\": {},\n   \"x\": []\n  }\n ],\n \"c\": {\n  \"k\": null\n }\n}"
        );
    }

    pub(in crate::walk_verify) fn protocol_fields(frozen_at: &str, flaky: Value) -> Entry {
        entry(json!({
            "slug": "selftest-flaky",
            "failure_memory": "mem-0000000000000021",
            "good": "1".repeat(40),
            "bad": "2".repeat(40),
            "oracle_sha256": "feed".repeat(16),
            "rule": "exit 0 good, 125 cannot test, any other code bad",
            "candidates": ["3".repeat(40), "4".repeat(40)],
            "max_candidates": 12,
            "max_line_runs": 24,
            "flaky": flaky,
            "frozen_at": frozen_at,
        }))
    }

    #[test]
    fn the_protocol_hash_matches_the_python_vectors_and_skips_what_the_report_adds() {
        // hashlib.sha256(json.dumps(proto, sort_keys=True,
        // separators=(",", ":")).encode()).hexdigest() of the same fields,
        // once for a plain run and once for a --flaky one.
        let plain = "0e85b543ab940e1f6ed686c5ce302d3f32fbb0cf7310266c0456af218debd7dd";
        let flaky = "415587e2ffb5861ce6f2c3805e6b54af0a8450337a7226824536c0eb3cd0e6fd";
        let mut protocol = protocol_fields("2026-10-06T00:05:06+00:00", Value::Null);
        assert_eq!(protocol_hash(&protocol), plain);
        protocol.insert("sha256".to_string(), json!(plain));
        protocol.insert("memory".to_string(), json!("mem-0000000000000755"));
        assert_eq!(protocol_hash(&protocol), plain);
        protocol.insert("candidates".to_string(), json!(["3".repeat(40)]));
        assert_ne!(protocol_hash(&protocol), plain);

        let settings = json!({
            "alpha": 0.01,
            "max_runs_per_commit": 80,
            "baseline_max": 100,
            "strength_runs": 50,
        });
        let protocol = protocol_fields("2026-10-06T00:05:06+00:00", settings);
        assert_eq!(protocol_hash(&protocol), flaky);
    }
}
