//! JUnit XML to run records.
//!
//! One `<testcase>` becomes one [`RunRecord`]. The run id is
//! `{suite}::{classname}::{name}` and the subject is `{classname}::{name}`.
//! Names are the attribute bytes after the five predefined XML entities are
//! decoded. Tags and attributes are matched exactly; nothing is case-folded.

use crate::types::{RunKind, RunRecord, RunStatus};

/// Parse a JUnit document into run records.
///
/// `commit` is copied onto every record. Clocks stay `0`: a JUnit file does
/// not carry the unix-millisecond fields the registry stores. A document with
/// no testcase is an empty batch, not an error.
pub fn parse_junit(xml: &str, commit: &str) -> Result<Vec<RunRecord>, String> {
    let bytes = xml.as_bytes();
    let mut index = 0;
    let mut suites: Vec<String> = Vec::new();
    let mut out = Vec::new();
    while index < bytes.len() {
        if bytes[index] != b'<' {
            index += 1;
            continue;
        }
        if bytes[index..].starts_with(b"<!--") {
            index = skip_until(bytes, index + 4, b"-->")
                .ok_or_else(|| "unterminated XML comment".to_string())?;
            continue;
        }
        if bytes[index..].starts_with(b"<?") {
            index = skip_until(bytes, index + 2, b"?>")
                .ok_or_else(|| "unterminated XML declaration".to_string())?;
            continue;
        }
        if bytes[index..].starts_with(b"<!") {
            index = skip_until(bytes, index + 2, b">")
                .ok_or_else(|| "unterminated XML declaration".to_string())?;
            continue;
        }
        let closing = bytes[index..].starts_with(b"</");
        let start = if closing { index + 2 } else { index + 1 };
        let (name, attrs, empty, next) = read_tag(bytes, start)?;
        index = next;
        if closing {
            if name == "testsuite" {
                suites.pop();
            }
            continue;
        }
        if name == "testsuite" {
            let suite = attr(&attrs, "name").unwrap_or_default();
            if empty {
                continue;
            }
            suites.push(suite);
            continue;
        }
        if name != "testcase" {
            continue;
        }
        let classname = attr(&attrs, "classname").unwrap_or_default();
        let test_name = attr(&attrs, "name").unwrap_or_default();
        if classname.is_empty() || test_name.is_empty() {
            return Err("a testcase needs classname and name".into());
        }
        let status = if empty {
            RunStatus::Passed
        } else {
            let close_at = find_close(bytes, index, "testcase")
                .ok_or_else(|| "unterminated testcase".to_string())?;
            let body = &bytes[index..close_at];
            let status = if contains_tag(body, "error") {
                RunStatus::Errored
            } else if contains_tag(body, "failure") {
                RunStatus::Failed
            } else if contains_tag(body, "skipped") {
                RunStatus::Skipped
            } else {
                RunStatus::Passed
            };
            index = close_at;
            status
        };
        let suite = suites.last().cloned().unwrap_or_default();
        out.push(RunRecord {
            run_id: format!("{suite}::{classname}::{test_name}"),
            kind: RunKind::Test,
            subject: format!("{classname}::{test_name}"),
            commit: commit.to_string(),
            status,
            started_ms: 0,
            finished_ms: 0,
        });
    }
    Ok(out)
}

fn skip_until(bytes: &[u8], from: usize, marker: &[u8]) -> Option<usize> {
    bytes[from..]
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|at| from + at + marker.len())
}

fn read_tag(bytes: &[u8], start: usize) -> Result<(String, String, bool, usize), String> {
    let mut index = start;
    while index < bytes.len() && is_name(bytes[index]) {
        index += 1;
    }
    if index == start {
        return Err("XML tag has no name".into());
    }
    let name = std::str::from_utf8(&bytes[start..index])
        .map_err(|_| "XML tag name is not UTF-8".to_string())?
        .to_string();
    let attr_start = index;
    while index < bytes.len() && bytes[index] != b'>' {
        index += 1;
    }
    if index >= bytes.len() {
        return Err("unterminated XML tag".into());
    }
    let mut empty = false;
    let mut attr_end = index;
    if attr_end > attr_start && bytes[attr_end - 1] == b'/' {
        empty = true;
        attr_end -= 1;
    }
    let attrs = std::str::from_utf8(&bytes[attr_start..attr_end])
        .map_err(|_| "XML attributes are not UTF-8".to_string())?
        .to_string();
    Ok((name, attrs, empty, index + 1))
}

fn is_name(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b':' | b'_' | b'-' | b'.')
}

fn attr(attrs: &str, key: &str) -> Option<String> {
    let bytes = attrs.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }
        let name_at = index;
        while index < bytes.len() && is_name(bytes[index]) {
            index += 1;
        }
        let name = std::str::from_utf8(&bytes[name_at..index]).ok()?;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != b'=' {
            continue;
        }
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() {
            return None;
        }
        let quote = bytes[index];
        if quote != b'"' && quote != b'\'' {
            return None;
        }
        index += 1;
        let value_at = index;
        while index < bytes.len() && bytes[index] != quote {
            index += 1;
        }
        if index >= bytes.len() {
            return None;
        }
        let raw = std::str::from_utf8(&bytes[value_at..index]).ok()?;
        index += 1;
        if name == key {
            return Some(decode_entities(raw));
        }
    }
    None
}

fn decode_entities(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let Some(end) = rest.find(';') else {
            out.push_str(rest);
            return out;
        };
        let token = &rest[..=end];
        let decoded = match token {
            "&amp;" => "&",
            "&lt;" => "<",
            "&gt;" => ">",
            "&quot;" => "\"",
            "&apos;" => "'",
            _ => {
                out.push_str(token);
                rest = &rest[end + 1..];
                continue;
            }
        };
        out.push_str(decoded);
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

fn find_close(bytes: &[u8], from: usize, name: &str) -> Option<usize> {
    let marker = format!("</{name}>");
    skip_until(bytes, from, marker.as_bytes())
}

fn contains_tag(body: &[u8], name: &str) -> bool {
    let open = format!("<{name}");
    let mut from = 0;
    while let Some(at) = body[from..]
        .windows(open.len())
        .position(|window| window == open.as_bytes())
    {
        let next = from + at + open.len();
        if next >= body.len() || !is_name(body[next]) {
            return true;
        }
        from = next;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn junit_emits_one_record_per_testcase() {
        let xml = r#"<?xml version="1.0"?>
        <testsuite name="crate::mod">
          <testcase classname="crate::mod" name="keeps_order"/>
          <testcase classname="crate::mod" name="fails&amp;logs"><failure message="no"/></testcase>
          <testcase classname="a::b" name="skipped"><skipped/></testcase>
          <testcase classname="a::b" name="errors"><error message="boom"/></testcase>
        </testsuite>"#;
        let runs = parse_junit(xml, "abc").unwrap();
        assert_eq!(runs.len(), 4);
        assert_eq!(runs[0].run_id, "crate::mod::crate::mod::keeps_order");
        assert_eq!(runs[0].subject, "crate::mod::keeps_order");
        assert_eq!(runs[0].status, RunStatus::Passed);
        assert_eq!(runs[0].commit, "abc");
        assert_eq!(runs[0].kind, RunKind::Test);
        assert_eq!(runs[1].subject, "crate::mod::fails&logs");
        assert_eq!(runs[1].status, RunStatus::Failed);
        assert_eq!(runs[2].status, RunStatus::Skipped);
        assert_eq!(runs[3].status, RunStatus::Errored);
        assert!(parse_junit("<testsuite name='s'></testsuite>", "")
            .unwrap()
            .is_empty());
    }
}
