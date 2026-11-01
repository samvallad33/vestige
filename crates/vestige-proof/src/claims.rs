//! Claims follow from witnesses and from nothing else.
//!
//! For one tree, each command gets exactly one claim. A command with a witness
//! bound to that tree is judged by its newest such witness. A command that is
//! on the allowlist but has no bound witness is `unwitnessed`. Prose never
//! enters: the same witnesses and allowlist always give the same claims, so a
//! verifier recomputes them and refuses a bundle whose claims differ.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::witness::WitnessRecord;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Exit code 0, and the runner did not report an empty run.
    Green,
    /// A non-zero exit code, or a failure the host reported without one.
    Red,
    /// Exit code 0, but the runner's JSON or junit said no test ran.
    ZeroTests,
    /// On the allowlist, with no witness bound to this tree.
    Unwitnessed,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Green => "green",
            Status::Red => "red",
            Status::ZeroTests => "zero_tests",
            Status::Unwitnessed => "unwitnessed",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Claim {
    pub command: String,
    pub status: Status,
    /// The memory holding the newest bound witness, the claim's receipt.
    pub witness: Option<String>,
    pub exit: Option<i64>,
    /// Tests the runner said it collected, when it said so.
    pub tests: Option<u64>,
    /// Bound witnesses of this command at this tree, newest included.
    pub runs: u32,
}

/// The status one witness supports.
pub fn status_of(r: &WitnessRecord) -> Status {
    match (r.exit, &r.tests) {
        (Some(0), Some(t)) if t.ran_none() => Status::ZeroTests,
        (Some(0), _) => Status::Green,
        _ => Status::Red,
    }
}

/// The claims `witnesses` (memory id, record) support at `tree`, sorted by
/// command. Witnesses not bound to `tree` are ignored.
pub fn derive(
    tree: &str,
    witnesses: &[(String, WitnessRecord)],
    allowlist: &[String],
) -> Vec<Claim> {
    let mut newest: BTreeMap<&str, (&String, &WitnessRecord, u32)> = BTreeMap::new();
    for (memory, r) in witnesses.iter().filter(|(_, r)| r.bound_to(tree)) {
        newest
            .entry(r.command.as_str())
            .and_modify(|(m, best, runs)| {
                *runs += 1;
                if (r.ended_at.as_str(), memory.as_str()) > (best.ended_at.as_str(), m.as_str()) {
                    *m = memory;
                    *best = r;
                }
            })
            .or_insert((memory, r, 1));
    }
    let mut claims: BTreeMap<String, Claim> = newest
        .into_iter()
        .map(|(command, (memory, r, runs))| {
            (
                command.to_string(),
                Claim {
                    command: command.to_string(),
                    status: status_of(r),
                    witness: Some(memory.clone()),
                    exit: r.exit,
                    tests: r.tests.as_ref().map(|t| t.collected),
                    runs,
                },
            )
        })
        .collect();
    for command in allowlist {
        claims.entry(command.clone()).or_insert_with(|| Claim {
            command: command.clone(),
            status: Status::Unwitnessed,
            witness: None,
            exit: None,
            tests: None,
            runs: 0,
        });
    }
    claims.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_count::TestCount;
    use crate::witness::tests::record;

    fn tree() -> String {
        "c".repeat(40)
    }

    #[test]
    fn the_newest_bound_witness_decides_and_every_run_is_counted() {
        let t = tree();
        let red = record("cargo test", Some(101), &t, "2026-10-29T07:00:00.000Z");
        let green = record("cargo test", Some(0), &t, "2026-10-29T07:05:00.000Z");
        let claims = derive(
            &t,
            &[("mem-2".into(), green), ("mem-1".into(), red)],
            &["cargo test".into()],
        );
        assert_eq!(
            claims,
            vec![Claim {
                command: "cargo test".into(),
                status: Status::Green,
                witness: Some("mem-2".into()),
                exit: Some(0),
                tests: None,
                runs: 2,
            }]
        );
    }

    #[test]
    fn an_unbound_witness_leaves_its_command_unwitnessed() {
        let t = tree();
        let mut moved = record("cargo test", Some(0), &t, "2026-10-29T07:00:00.000Z");
        moved.tree_before = Some("d".repeat(40));
        let other_tree = record(
            "cargo test",
            Some(0),
            &"e".repeat(40),
            "2026-10-29T07:00:00.000Z",
        );
        let claims = derive(
            &t,
            &[("mem-1".into(), moved), ("mem-2".into(), other_tree)],
            &["cargo test".into(), "pnpm test".into()],
        );
        assert!(
            claims
                .iter()
                .all(|c| c.status == Status::Unwitnessed && c.runs == 0)
        );
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn red_zero_tests_and_off_list_commands_are_kept_as_they_are() {
        let t = tree();
        let mut empty = record("npx jest --json", Some(0), &t, "2026-10-29T07:00:00.000Z");
        empty.tests = Some(TestCount {
            collected: 0,
            passed: Some(0),
            failed: Some(0),
            source: "jest-json".into(),
        });
        let fail = record("go test ./...", None, &t, "2026-10-29T07:00:00.000Z");
        let claims = derive(
            &t,
            &[("mem-1".into(), empty), ("mem-2".into(), fail)],
            &["npx jest --json".into()],
        );
        let by: BTreeMap<_, _> = claims
            .iter()
            .map(|c| (c.command.as_str(), c.status))
            .collect();
        assert_eq!(by["npx jest --json"], Status::ZeroTests);
        assert_eq!(
            by["go test ./..."],
            Status::Red,
            "a witness off today's list still counts"
        );
    }
}
