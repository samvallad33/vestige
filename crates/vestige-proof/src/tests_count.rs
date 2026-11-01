//! How many tests a run collected, read only from a runner's own structured
//! output: one JSON report, JSON lines, or a junit XML file. Plain text is
//! never read, so a run whose runner gave neither has no count.
//!
//! | Format | Read |
//! |---|---|
//! | Jest and Vitest `--json` | `numTotalTests`, `numPassedTests`, `numFailedTests` |
//! | Mocha `--reporter json` | `stats.tests`, `stats.passes`, `stats.failures` |
//! | pytest-json-report | `summary.collected` (else `summary.total`), `passed`, `failed` |
//! | libtest JSON (`cargo test -- --format json`, nextest) | suite `test_count`, `passed`, `failed` |
//! | `go test -json` | distinct `(Package, Test)` that ran, passed, failed |
//! | junit XML | `tests`, `failures` and `errors` of `<testsuites>`, else summed over `<testsuite>` |

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct TestCount {
    pub collected: u64,
    pub passed: Option<u64>,
    pub failed: Option<u64>,
    /// Which format the count was read from.
    pub source: String,
}

impl TestCount {
    /// The run executed no test at all: none collected, or none passed or failed.
    pub fn ran_none(&self) -> bool {
        self.collected == 0 || (self.passed == Some(0) && self.failed.unwrap_or(0) == 0)
    }
}

fn num(v: &Value, path: &str) -> Option<u64> {
    v.pointer(path).and_then(Value::as_u64)
}

fn count(collected: u64, passed: Option<u64>, failed: Option<u64>, source: &str) -> TestCount {
    TestCount {
        collected,
        passed,
        failed,
        source: source.to_string(),
    }
}

/// Count tests from a runner's standard output.
pub fn from_output(output: &str) -> Option<TestCount> {
    let text = output.trim();
    if text.starts_with('{')
        && let Ok(v) = serde_json::from_str::<Value>(text)
    {
        if let Some(n) = num(&v, "/numTotalTests") {
            return Some(count(
                n,
                num(&v, "/numPassedTests"),
                num(&v, "/numFailedTests"),
                "jest-json",
            ));
        }
        if let Some(n) = num(&v, "/stats/tests") {
            return Some(count(
                n,
                num(&v, "/stats/passes"),
                num(&v, "/stats/failures"),
                "mocha-json",
            ));
        }
        if let Some(n) = num(&v, "/summary/collected").or_else(|| num(&v, "/summary/total")) {
            return Some(count(
                n,
                Some(num(&v, "/summary/passed").unwrap_or(0)),
                Some(num(&v, "/summary/failed").unwrap_or(0)),
                "pytest-json-report",
            ));
        }
    }
    json_lines(text)
}

fn json_lines(text: &str) -> Option<TestCount> {
    let mut libtest = false;
    let (mut started, mut lt_passed, mut lt_failed) = (0u64, 0u64, 0u64);
    let mut go = false;
    let mut ran = std::collections::BTreeSet::new();
    let mut go_passed = std::collections::BTreeSet::new();
    let mut go_failed = std::collections::BTreeSet::new();
    for line in text.lines().map(str::trim).filter(|l| l.starts_with('{')) {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if v.get("type").and_then(Value::as_str) == Some("suite") {
            libtest = true;
            match v.get("event").and_then(Value::as_str) {
                Some("started") => started += num(&v, "/test_count").unwrap_or(0),
                Some("ok" | "failed") => {
                    lt_passed += num(&v, "/passed").unwrap_or(0);
                    lt_failed += num(&v, "/failed").unwrap_or(0);
                }
                _ => {}
            }
            continue;
        }
        if let (Some(action), Some(pkg)) = (
            v.get("Action").and_then(Value::as_str),
            v.get("Package").and_then(Value::as_str),
        ) {
            go = true;
            let Some(test) = v.get("Test").and_then(Value::as_str) else {
                continue;
            };
            let key = (pkg.to_string(), test.to_string());
            match action {
                "run" => {
                    ran.insert(key);
                }
                "pass" => {
                    go_passed.insert(key);
                }
                "fail" => {
                    go_failed.insert(key);
                }
                _ => {}
            }
        }
    }
    if libtest {
        let collected = started.max(lt_passed + lt_failed);
        return Some(count(
            collected,
            Some(lt_passed),
            Some(lt_failed),
            "libtest-json",
        ));
    }
    if go {
        return Some(count(
            ran.len() as u64,
            Some(go_passed.len() as u64),
            Some(go_failed.len() as u64),
            "go-test-json",
        ));
    }
    None
}

/// The attributes of one XML start tag, `name="value"` or `name='value'`.
fn attributes(tag: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = tag;
    while let Some(eq) = rest.find('=') {
        let name = rest[..eq]
            .rsplit(|c: char| c.is_whitespace())
            .next()
            .unwrap_or("")
            .to_string();
        let after = rest[eq + 1..].trim_start();
        let Some(q) = after.chars().next().filter(|c| *c == '"' || *c == '\'') else {
            break;
        };
        let body = &after[1..];
        let Some(end) = body.find(q) else { break };
        out.push((name, body[..end].to_string()));
        rest = &body[end + 1..];
    }
    out
}

fn attr(attrs: &[(String, String)], name: &str) -> Option<u64> {
    attrs
        .iter()
        .find(|(k, _)| k == name)
        .and_then(|(_, v)| v.trim().parse().ok())
}

/// The `<testcase>` elements of a report: (cases, failed, skipped). A case
/// failed when it holds a `<failure>` or `<error>`, and was skipped when it
/// holds a `<skipped>`.
#[derive(Default)]
struct Cases {
    total: u64,
    failed: u64,
    skipped: u64,
    open: Option<(bool, bool)>,
}

impl Cases {
    fn close(&mut self) {
        if let Some((failed, skipped)) = self.open.take() {
            self.total += 1;
            self.failed += u64::from(failed);
            self.skipped += u64::from(skipped && !failed);
        }
    }
}

/// Count tests from a junit XML report. The runner's own `tests` totals win;
/// a report without them, such as the one Node's built-in reporter writes, is
/// counted by its `<testcase>` elements.
pub fn from_junit(xml: &str) -> Option<TestCount> {
    let mut suites_root: Option<Vec<(String, String)>> = None;
    let mut suites: Vec<Vec<(String, String)>> = Vec::new();
    let mut cases = Cases::default();
    let mut rest = xml;
    while let Some(lt) = rest.find('<') {
        rest = &rest[lt + 1..];
        if let Some(after) = rest.strip_prefix("!--") {
            rest = after.split_once("-->").map(|(_, r)| r).unwrap_or("");
            continue;
        }
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[..gt];
        let closing = tag.starts_with('/');
        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '/')
            .collect();
        let self_closing = tag.trim_end().ends_with('/');
        match (name.as_str(), closing) {
            ("testsuites", false) if suites_root.is_none() => {
                suites_root = Some(attributes(&tag[name.len()..]))
            }
            ("testsuite", false) => suites.push(attributes(&tag[name.len()..])),
            ("testcase", false) => {
                cases.close();
                cases.open = Some((false, false));
                if self_closing {
                    cases.close();
                }
            }
            ("testcase", true) => cases.close(),
            ("failure" | "error", false) => {
                if let Some(open) = cases.open.as_mut() {
                    open.0 = true;
                }
            }
            ("skipped", false) => {
                if let Some(open) = cases.open.as_mut() {
                    open.1 = true;
                }
            }
            _ => {}
        }
        rest = &rest[gt + 1..];
    }
    cases.close();
    // A missing failures, errors or skipped attribute counts as zero.
    let sum = |key: &str| -> u64 { suites.iter().filter_map(|a| attr(a, key)).sum() };
    let (collected, failed, skipped) = match &suites_root {
        Some(root) if attr(root, "tests").is_some() => (
            attr(root, "tests")?,
            attr(root, "failures").unwrap_or(0) + attr(root, "errors").unwrap_or(0),
            attr(root, "skipped")
                .or_else(|| attr(root, "disabled"))
                .unwrap_or(0),
        ),
        _ if suites.iter().any(|a| attr(a, "tests").is_some()) => (
            sum("tests"),
            sum("failures") + sum("errors"),
            sum("skipped"),
        ),
        _ if suites_root.is_some() || !suites.is_empty() => {
            (cases.total, cases.failed, cases.skipped)
        }
        _ => return None,
    };
    let passed = collected.saturating_sub(failed + skipped);
    Some(count(collected, Some(passed), Some(failed), "junit"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jest_mocha_and_pytest_reports_are_counted() {
        let jest = r#"{"numTotalTests": 12, "numPassedTests": 11, "numFailedTests": 1, "testResults": []}"#;
        assert_eq!(
            from_output(jest),
            Some(count(12, Some(11), Some(1), "jest-json"))
        );
        let zero = r#"{"numTotalTests": 0, "numPassedTests": 0, "numFailedTests": 0}"#;
        assert!(from_output(zero).unwrap().ran_none());
        let mocha = r#"{"stats": {"tests": 3, "passes": 3, "failures": 0}}"#;
        assert_eq!(from_output(mocha).unwrap().collected, 3);
        let pytest = r#"{"summary": {"collected": 0, "total": 0}}"#;
        assert!(from_output(pytest).unwrap().ran_none());
    }

    #[test]
    fn libtest_json_sums_every_suite_and_sees_an_empty_run() {
        let out = r#"
{ "type": "suite", "event": "started", "test_count": 2 }
{ "type": "test", "event": "ok", "name": "a" }
{ "type": "suite", "event": "ok", "passed": 2, "failed": 0, "ignored": 0, "measured": 0, "filtered_out": 0 }
{ "type": "suite", "event": "started", "test_count": 0 }
{ "type": "suite", "event": "ok", "passed": 0, "failed": 0, "ignored": 0, "measured": 0, "filtered_out": 0 }
"#;
        let c = from_output(out).unwrap();
        assert_eq!(
            (c.collected, c.passed, c.source.as_str()),
            (2, Some(2), "libtest-json")
        );
        assert!(!c.ran_none());
        let filtered = "{\"type\":\"suite\",\"event\":\"started\",\"test_count\":0}\n\
                        {\"type\":\"suite\",\"event\":\"ok\",\"passed\":0,\"failed\":0,\"filtered_out\":40}";
        assert!(from_output(filtered).unwrap().ran_none());
    }

    #[test]
    fn go_test_json_counts_distinct_tests() {
        let out = r#"{"Action":"start","Package":"x"}
{"Action":"run","Package":"x","Test":"TestA"}
{"Action":"output","Package":"x","Test":"TestA","Output":"ok"}
{"Action":"pass","Package":"x","Test":"TestA"}
{"Action":"run","Package":"x","Test":"TestB"}
{"Action":"fail","Package":"x","Test":"TestB"}
{"Action":"fail","Package":"x"}"#;
        assert_eq!(
            from_output(out),
            Some(count(2, Some(1), Some(1), "go-test-json"))
        );
        let none = r#"{"Action":"output","Package":"x","Output":"no test files"}
{"Action":"skip","Package":"x"}"#;
        assert!(from_output(none).unwrap().ran_none());
    }

    #[test]
    fn plain_text_has_no_count() {
        assert_eq!(
            from_output("running 0 tests\ntest result: ok. 0 passed"),
            None
        );
        assert_eq!(from_output(""), None);
    }

    #[test]
    fn junit_reads_the_root_or_sums_the_suites() {
        let root = r#"<?xml version="1.0"?><!-- <testsuite tests="99"> -->
<testsuites name="nextest" tests="5" failures="1" errors="0" skipped="1">
  <testsuite name="a" tests="5" failures="1"/>
</testsuites>"#;
        assert_eq!(from_junit(root), Some(count(5, Some(3), Some(1), "junit")));
        let suites = r#"<testsuite name='a' tests='2' failures='0'></testsuite><testsuite name="b" tests="1" errors="1"/>"#;
        assert_eq!(
            from_junit(suites),
            Some(count(3, Some(2), Some(1), "junit"))
        );
        assert!(
            from_junit(r#"<testsuites></testsuites>"#)
                .unwrap()
                .ran_none()
        );
        assert_eq!(from_junit("<html></html>"), None);
    }

    /// The report Node 26's built-in junit reporter wrote in the live check
    /// on 2026-10-29: test cases straight under the root, totals only in
    /// comments. It was first read as zero tests.
    #[test]
    fn a_report_without_totals_is_counted_by_its_test_cases() {
        let node = r#"<?xml version="1.0" encoding="utf-8"?>
<testsuites>
	<testcase name="adds" time="0.000389" classname="test" file="/repo/calc.test.js"/>
	<testcase name="multiplies" time="0.000070" classname="test" file="/repo/calc.test.js"/>
	<!-- tests 2 -->
	<!-- suites 0 -->
	<!-- pass 2 -->
	<!-- fail 0 -->
</testsuites>"#;
        assert_eq!(from_junit(node), Some(count(2, Some(2), Some(0), "junit")));
        let mixed = r#"<testsuites><testsuite name="s">
<testcase name="a"/>
<testcase name="b"><failure message="boom">x</failure></testcase>
<testcase name="c"><skipped/></testcase>
<testcase name="d"><error/></testcase>
</testsuite></testsuites>"#;
        assert_eq!(from_junit(mixed), Some(count(4, Some(1), Some(2), "junit")));
        let empty_suite = r#"<testsuites><testsuite name="s"></testsuite></testsuites>"#;
        assert!(from_junit(empty_suite).unwrap().ran_none());
    }
}
