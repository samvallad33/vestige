//! `vestige prove --check`: re-verify a report offline.
//!
//! What a report claims rests on its probes, and the probes are a hash
//! chain: each entry carries the sha256 of the one before it. The check
//! recomputes the chain, then holds everything else in the report against
//! it and against what can be recomputed without running a test:
//!
//! * every probe hash and the chain head;
//! * the first bad commit is recorded bad, and each of its parents is
//!   recorded good or is an ancestor of a commit that is (looked up in the
//!   report's repo when that is on this machine);
//! * the frozen protocol still has the hash it was given and was frozen no
//!   later than the first test;
//! * each rung of the verdict card that says it holds is backed by the
//!   recorded runs, and the verdicts the `why` section quotes are the ones
//!   the probes recorded;
//! * the undo patch beside the report is the one whose hash was recorded.
//!
//! It never runs a test and never opens the store.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::Context;
use chrono::DateTime;
use serde_json::Value;

use super::git::{git_test, parents_of};
use super::json::{ZERO_HASH, chain_hash, protocol_hash, sha256_hex};
use super::stats::{REPEATED_P, fisher_p};
use super::text::{Palette, abspath, head, py_display};

fn is_verdict(text: &str) -> bool {
    matches!(text, "good" | "bad" | "skip")
}

/// Whether the recorded runs back a rung that says it holds. `None` when
/// that cannot be told from the probes alone.
fn backed(rung: &str, rep: &Value, probes: &[Value]) -> Option<bool> {
    let last_in = |phase: &str| {
        probes
            .iter()
            .rev()
            .find(|probe| probe["phase"] == phase)
            .and_then(|probe| probe["verdict"].as_str())
    };
    match rung {
        "BOUNDARY" => {
            let first_bad = rep["first_bad_commit"].as_str()?;
            Some(
                probes
                    .iter()
                    .any(|probe| probe["commit"] == first_bad && probe["verdict"] == "bad"),
            )
        }
        // One change: the boundary is its proof, and that is checked above.
        "ISOLATED" if rep["why"]["changes_in_commit"] == 1 => None,
        "ISOLATED" => Some(last_in("without") == Some("good")),
        "REVERSED" => Some(last_in("undo") == Some("good")),
        "REPEATED" => {
            let sides: Vec<(u64, u64)> = probes
                .iter()
                .filter(|probe| probe["phase"] == "strength")
                .filter_map(|probe| Some((probe["fails"].as_u64()?, probe["runs"].as_u64()?)))
                .collect();
            match sides.as_slice() {
                [(f1, n1), (f0, n0)] if f1 <= n1 && f0 <= n0 => {
                    Some(fisher_p(*f1, *n1, *f0, *n0) < REPEATED_P)
                }
                _ => Some(false),
            }
        }
        _ => None,
    }
}

/// Whether `first` is no later than `second`. Both are the tool's own
/// timestamps; text that is not a timestamp is compared as text.
fn not_after(first: &str, second: &str) -> bool {
    match (
        DateTime::parse_from_rfc3339(first),
        DateTime::parse_from_rfc3339(second),
    ) {
        (Ok(first), Ok(second)) => first <= second,
        _ => first <= second,
    }
}

/// Re-verify a report offline. Returns the process exit code: 0 when every
/// check in the module description passes, 1 otherwise.
pub fn check(report: &Path) -> anyhow::Result<i32> {
    let Palette { r, o, .. } = Palette::detect();
    let text =
        fs::read_to_string(report).with_context(|| format!("cannot read {}", report.display()))?;
    let rep: Value =
        serde_json::from_str(&text).with_context(|| format!("{} is not JSON", report.display()))?;
    let probes = rep["probes"]
        .as_array()
        .with_context(|| format!("{} has no probes", report.display()))?;

    let mut prev = ZERO_HASH.to_string();
    for probe in probes {
        let entry = probe.as_object();
        let want = entry.map(|entry| chain_hash(&prev, entry));
        let linked = entry
            .and_then(|entry| entry.get("prev"))
            .and_then(Value::as_str)
            == Some(prev.as_str());
        let hashed = entry
            .and_then(|entry| entry.get("hash"))
            .and_then(Value::as_str)
            == want.as_deref();
        let Some(want) = want.filter(|_| linked && hashed) else {
            println!(
                "{r}probe {}: hash does not match. The report was changed after it was written.{o}",
                py_display(probe.get("n"))
            );
            return Ok(1);
        };
        prev = want;
    }
    if rep["chain_head"].as_str() != Some(prev.as_str()) {
        println!("{r}chain head does not match the last probe.{o}");
        return Ok(1);
    }
    let mut verdicts: HashMap<&str, &str> = HashMap::new();
    for probe in probes {
        if let (Some(commit), Some(verdict)) = (probe["commit"].as_str(), probe["verdict"].as_str())
        {
            verdicts.insert(commit, verdict);
        }
    }
    println!(
        "{} probes, hash chain intact, head {}",
        probes.len(),
        head(&prev, 16)
    );

    let mut sound = true;
    let mut fail = |message: &str| {
        println!("{r}{message}{o}");
        sound = false;
    };
    if let Some(first_bad) = rep["first_bad_commit"].as_str() {
        let verdict = verdicts.get(first_bad).copied().unwrap_or("MISSING");
        println!(
            "first bad commit {}: recorded verdict {verdict}",
            head(first_bad, 10)
        );
        if verdict != "bad" {
            fail("the first bad commit is not recorded as bad by a probe.");
        }
        let repo = rep["repo"]
            .as_str()
            .map(Path::new)
            .filter(|repo| repo.is_dir());
        let parents = repo.and_then(|repo| Some((repo, parents_of(repo, first_bad).ok()?)));
        match parents {
            Some((repo, parents)) => {
                for parent in &parents {
                    let verdict = verdicts.get(parent.as_str()).copied();
                    println!(
                        "  its parent {}: recorded verdict {}",
                        head(parent, 10),
                        verdict.unwrap_or("not probed")
                    );
                    match verdict {
                        Some("good") => {}
                        Some(_) => {
                            fail("a parent of the first bad commit is not recorded as good.")
                        }
                        None => {
                            // git bisect never tests a commit a good one
                            // already rules out: its ancestors.
                            let implied = verdicts.iter().find(|(commit, verdict)| {
                                **verdict == "good"
                                    && git_test(
                                        repo,
                                        &["merge-base", "--is-ancestor", parent, commit],
                                    )
                                    .unwrap_or(false)
                            });
                            match implied {
                                Some((commit, _)) => println!(
                                    "    it is an ancestor of {}, which is recorded good",
                                    head(commit, 10)
                                ),
                                None => fail(
                                    "a parent of the first bad commit has no good verdict behind it.",
                                ),
                            }
                        }
                    }
                }
            }
            None => println!(
                "  its parent: not looked up, the repo {} with this commit is not on this machine",
                py_display(rep.get("repo"))
            ),
        }
    } else {
        println!("this report names no first bad commit");
    }

    if let Some(protocol) = rep["protocol"].as_object() {
        let matches = protocol.get("sha256").and_then(Value::as_str)
            == Some(protocol_hash(protocol).as_str());
        let frozen_at = protocol
            .get("frozen_at")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let first_at = probes
            .first()
            .and_then(|probe| probe["at"].as_str())
            .unwrap_or_default();
        println!(
            "protocol {}: {}, frozen {frozen_at}, first test {first_at}",
            py_display(protocol.get("memory")),
            if matches {
                "hash matches"
            } else {
                "HASH DOES NOT MATCH"
            }
        );
        if !matches {
            fail("the protocol is not the one that was hashed.");
        }
        // On record before any test, or it could have been chosen after one.
        if !first_at.is_empty() && !not_after(frozen_at, first_at) {
            fail("the protocol was frozen after the first test had run.");
        }
        for (name, sha256) in protocol
            .get("also_hashed")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            println!(
                "  also frozen: {name} sha256 {}",
                head(sha256.as_str().unwrap_or_default(), 16)
            );
        }
    }

    for rung in rep["verdict_card"].as_array().into_iter().flatten() {
        let name = py_display(rung.get("rung"));
        let holds = rung["holds"] == true;
        println!(
            "  {name:<9} {}  {}",
            if holds { "yes" } else { "no " },
            py_display(rung.get("statement"))
        );
        if holds && backed(&name, &rep, probes) == Some(false) {
            fail(&format!(
                "the card says {name} holds, but the recorded runs do not back it."
            ));
        }
    }

    if let Some(why) = rep["why"].as_object() {
        let last_in = |phase: &str| {
            probes
                .iter()
                .rev()
                .find(|probe| probe["phase"] == phase)
                .and_then(|probe| probe["verdict"].as_str())
        };
        for change in why
            .get("minimal_failing_changes")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            println!(
                "minimal failing change: {} line {}",
                py_display(change.get("file")),
                py_display(change.get("line"))
            );
        }
        let without = why.get("commit_without_minimal").and_then(Value::as_str);
        match without {
            Some(verdict) => println!("the commit without it: recorded verdict {verdict}"),
            None if why.get("changes_in_commit") == Some(&Value::from(1)) => {
                println!("the commit without it: single change, covered by the parent run");
            }
            None => println!("the commit without it: not tested"),
        }
        if let Some(verdict) = without.filter(|verdict| is_verdict(verdict))
            && last_in("without") != Some(verdict)
        {
            fail("no recorded run gives that verdict for the commit without it.");
        }
        let bad_ref = py_display(rep["bad"].get("ref"));
        let undo = why.get("undo_on_bad").and_then(Value::as_str);
        match (undo, why.get("undo_how").and_then(Value::as_str)) {
            (Some(verdict), Some(how)) => {
                println!("undo on {bad_ref} ({how}): recorded verdict {verdict}");
            }
            (Some("rewritten"), None) => {
                println!("undo on {bad_ref}: none applies, later commits rewrote these lines");
            }
            (verdict, _) => println!(
                "undo on {bad_ref}: recorded verdict {}",
                verdict.unwrap_or("None")
            ),
        }
        if let Some(verdict) = undo.filter(|verdict| is_verdict(verdict))
            && last_in("undo") != Some(verdict)
        {
            fail("no recorded run gives that verdict for the undo.");
        }
        if let Some(patch) = why.get("undo_patch").and_then(Value::as_object) {
            // Only ever a file beside the report, whatever the report says.
            let name = patch
                .get("file")
                .and_then(Value::as_str)
                .and_then(|file| Path::new(file).file_name())
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            let beside = abspath(report)?
                .parent()
                .map(|dir| dir.join(&name))
                .unwrap_or_default();
            match fs::read(&beside) {
                Ok(bytes) => {
                    let same = Some(sha256_hex(&bytes).as_str())
                        == patch.get("sha256").and_then(Value::as_str);
                    println!(
                        "undo patch {name}: {}",
                        if same {
                            "sha256 matches the report"
                        } else {
                            "DOES NOT MATCH the report"
                        }
                    );
                    if !same {
                        fail("the undo patch beside the report is not the one that was tested.");
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    println!("undo patch {name}: not beside the report, so not checked");
                }
                Err(err) => {
                    println!("undo patch {name}: cannot be read ({err})");
                    fail("the undo patch beside the report could not be checked.");
                }
            }
        }
    }
    println!(
        "script sha256 {} ({})",
        head(rep["oracle"]["sha256"].as_str().unwrap_or_default(), 16),
        py_display(rep["oracle"].get("file"))
    );
    Ok(if sound { 0 } else { 1 })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::super::json::tests::protocol_fields;
    use super::super::json::{Entry, pretty_json};
    use super::*;

    const GOOD: &str = "1111111111111111111111111111111111111111";
    const BAD: &str = "2222222222222222222222222222222222222222";

    /// One probe: `(commit, verdict, phase)` and, for repeated runs,
    /// `(runs, fails)`.
    type Row = (&'static str, &'static str, &'static str, Option<(u64, u64)>);

    /// A chain of probes, hashed the way a run hashes them.
    fn chain(rows: &[Row]) -> Vec<Value> {
        let mut prev = ZERO_HASH.to_string();
        rows.iter()
            .enumerate()
            .map(|(index, (commit, verdict, phase, counts))| {
                let mut entry: Entry = json!({
                    "commit": commit,
                    "subject": "subject",
                    "verdict": verdict,
                    "exit": i64::from(*verdict != "good"),
                    "oracle_said": format!("said {verdict}"),
                    "oracle_sha256": "feed",
                    "phase": phase,
                    "at": "2026-10-05T23:45:31+00:00",
                    "memory": null,
                    "n": index + 1,
                    "prev": prev,
                })
                .as_object()
                .cloned()
                .unwrap();
                if let Some((runs, fails)) = counts {
                    entry.insert("runs".to_string(), json!(runs));
                    entry.insert("fails".to_string(), json!(fails));
                }
                let hash = chain_hash(&prev, &entry);
                entry.insert("hash".to_string(), json!(hash));
                prev = hash;
                Value::Object(entry)
            })
            .collect()
    }

    /// A report over `rows`, with no repo behind it, after `edit`.
    fn write_report(dir: &Path, rows: &[Row], edit: impl FnOnce(&mut Value)) -> PathBuf {
        let probes = chain(rows);
        let mut report = json!({
            "repo": dir.join("no-such-repo"),
            "bad": {"ref": "v2"},
            "oracle": {"file": "test-command.sh", "sha256": "feed"},
            "first_bad_commit": BAD,
            "chain_head": probes.last().map(|probe| probe["hash"].clone()),
            "probes": probes,
            "why": null,
        });
        edit(&mut report);
        let path = dir.join("report.json");
        fs::write(&path, pretty_json(&report)).unwrap();
        path
    }

    const TWO_ENDS: [Row; 2] = [
        (GOOD, "good", "baseline", None),
        (BAD, "bad", "baseline", None),
    ];

    #[test]
    fn check_passes_an_untouched_report_and_fails_a_changed_one() {
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        assert_eq!(check(&write_report(dir, &TWO_ENDS, |_| {})).unwrap(), 0);
        // One verdict changed after the fact.
        let changed = write_report(dir, &TWO_ENDS, |report| {
            report["probes"][1]["verdict"] = json!("good");
        });
        assert_eq!(check(&changed).unwrap(), 1);
        // A probe dropped from the end, head left as it was.
        let dropped = write_report(dir, &TWO_ENDS, |report| {
            report["probes"].as_array_mut().unwrap().pop();
        });
        assert_eq!(check(&dropped).unwrap(), 1);
        // A probe dropped from the end and the head moved back to match.
        let rehung = write_report(dir, &TWO_ENDS, |report| {
            report["probes"].as_array_mut().unwrap().pop();
            report["chain_head"] = report["probes"][0]["hash"].clone();
        });
        assert_eq!(
            check(&rehung).unwrap(),
            1,
            "the first bad commit has no run left"
        );
        // An intact chain that names a commit no probe found bad.
        let renamed = write_report(dir, &TWO_ENDS, |report| {
            report["first_bad_commit"] = json!(GOOD);
        });
        assert_eq!(check(&renamed).unwrap(), 1);
        // A probe that is not an object, and a report without probes.
        let garbled = write_report(dir, &TWO_ENDS, |report| {
            report["probes"][0] = json!("a string");
        });
        assert_eq!(check(&garbled).unwrap(), 1);
        let empty = write_report(dir, &TWO_ENDS, |report| {
            report.as_object_mut().unwrap().remove("probes");
        });
        assert!(check(&empty).is_err());
        fs::write(dir.join("report.json"), "not json").unwrap();
        assert!(check(&dir.join("report.json")).is_err());
        assert!(check(&dir.join("missing.json")).is_err());
        // A report that names no commit still has a chain to check.
        let unnamed = write_report(dir, &TWO_ENDS, |report| {
            report["first_bad_commit"] = Value::Null;
        });
        assert_eq!(check(&unnamed).unwrap(), 0);
    }

    #[test]
    fn check_holds_the_undo_patch_to_its_recorded_hash() {
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        let patch = b"diff --git a/x b/x\n";
        let with_patch = |file: &str, sha256: String| {
            let file = file.to_string();
            write_report(dir, &TWO_ENDS, move |report| {
                report["why"] = json!({"undo_patch": {"file": file, "sha256": sha256}});
            })
        };
        // Not beside the report: nothing to compare, not a failure.
        assert_eq!(
            check(&with_patch("report.undo.patch", sha256_hex(patch))).unwrap(),
            0
        );
        fs::write(dir.join("report.undo.patch"), patch).unwrap();
        assert_eq!(
            check(&with_patch("report.undo.patch", sha256_hex(patch))).unwrap(),
            0
        );
        // The file is looked for beside the report, whatever path it is given.
        assert_eq!(
            check(&with_patch(
                "../elsewhere/report.undo.patch",
                sha256_hex(patch)
            ))
            .unwrap(),
            0
        );
        fs::write(dir.join("report.undo.patch"), "other bytes").unwrap();
        assert_eq!(
            check(&with_patch("report.undo.patch", sha256_hex(patch))).unwrap(),
            1
        );
    }

    #[test]
    fn check_holds_the_protocol_to_its_hash_and_to_the_first_test() {
        let dir = tempfile::tempdir().unwrap();
        // The probes of `chain` ran at 23:45:31.
        let with_protocol = |frozen_at: &str, edit: fn(&mut Value)| {
            let mut protocol = Value::Object(protocol_fields(frozen_at, Value::Null));
            protocol["sha256"] = json!(protocol_hash(protocol.as_object().unwrap()));
            protocol["memory"] = json!("mem-01");
            edit(&mut protocol);
            write_report(dir.path(), &TWO_ENDS, |report| {
                report["protocol"] = protocol;
            })
        };
        let frozen_first = with_protocol("2026-10-05T23:45:30+00:00", |_| {});
        assert_eq!(check(&frozen_first).unwrap(), 0);
        let same_second = with_protocol("2026-10-05T23:45:31+00:00", |_| {});
        assert_eq!(check(&same_second).unwrap(), 0);
        // A lead added to the protocol after it was hashed.
        let widened = with_protocol("2026-10-05T23:45:30+00:00", |protocol| {
            protocol["candidates"] = json!(["5".repeat(40)]);
        });
        assert_eq!(check(&widened).unwrap(), 1);
        // A file slipped into the frozen set afterwards.
        let slipped = with_protocol("2026-10-05T23:45:30+00:00", |protocol| {
            protocol["also_hashed"] = json!({"fixture.json": "00"});
        });
        assert_eq!(check(&slipped).unwrap(), 1);
        // A protocol written down after the first test had already run.
        let frozen_late = with_protocol("2026-10-05T23:45:32+00:00", |_| {});
        assert_eq!(check(&frozen_late).unwrap(), 1);
        // The same instant in another zone is not "late".
        let other_zone = with_protocol("2026-10-06T01:45:30+02:00", |_| {});
        assert_eq!(check(&other_zone).unwrap(), 0);
    }

    #[test]
    fn timestamps_compare_as_instants_when_they_are_timestamps() {
        assert!(not_after(
            "2026-10-05T23:45:30+00:00",
            "2026-10-05T23:45:31+00:00"
        ));
        assert!(not_after(
            "2026-10-05T23:45:31+00:00",
            "2026-10-05T23:45:31+00:00"
        ));
        assert!(!not_after(
            "2026-10-05T23:45:32+00:00",
            "2026-10-05T23:45:31+00:00"
        ));
        assert!(not_after(
            "2026-10-06T01:45:30+02:00",
            "2026-10-05T23:45:31Z"
        ));
        // Not timestamps: the reference tool's plain text comparison.
        assert!(not_after("a", "b"));
        assert!(!not_after("b", "a"));
    }

    const FULL_RUN: [Row; 7] = [
        (GOOD, "good", "baseline", None),
        (BAD, "bad", "baseline", None),
        ("1111111111 + 1 of 2 changes", "bad", "lines", None),
        ("1111111111 + the other 1 changes", "good", "without", None),
        ("2222222222 with 2222222222 undone", "good", "undo", None),
        ("2222222222 x30", "bad", "strength", Some((30, 10))),
        ("1111111111 x30", "good", "strength", Some((30, 0))),
    ];

    fn card(rows: &[(&str, bool)]) -> Value {
        rows.iter()
            .map(|(rung, holds)| json!({"rung": rung, "holds": holds, "statement": "s", "proof": "p"}))
            .collect()
    }

    #[test]
    fn check_holds_the_card_to_the_recorded_runs() {
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        let all = [
            ("LEAD", true),
            ("BOUNDARY", true),
            ("CONFIRMED", true),
            ("ISOLATED", true),
            ("REVERSED", true),
            ("REPEATED", true),
        ];
        let why = json!({
            "changes_in_commit": 2,
            "commit_without_minimal": "good",
            "undo_how": "whole commit",
            "undo_on_bad": "good",
            "minimal_failing_changes": [{"file": "calc.sh", "line": 15}],
        });
        let report = |rows: &[Row], why: Value| {
            write_report(dir, rows, |report| {
                report["verdict_card"] = card(&all);
                report["why"] = why;
            })
        };
        assert_eq!(check(&report(&FULL_RUN, why.clone())).unwrap(), 0);
        // A rung that says no is never held against the runs.
        let modest = write_report(dir, &TWO_ENDS, |report| {
            report["verdict_card"] = card(&[("ISOLATED", false), ("REVERSED", false)]);
        });
        assert_eq!(check(&modest).unwrap(), 0);

        // Each rung flipped to yes over runs that say otherwise.
        let mut undo_failed = FULL_RUN;
        undo_failed[4].1 = "bad";
        assert_eq!(check(&report(&undo_failed, json!(null))).unwrap(), 1);
        let mut rest_failed = FULL_RUN;
        rest_failed[3].1 = "bad";
        assert_eq!(check(&report(&rest_failed, json!(null))).unwrap(), 1);
        let mut weak = FULL_RUN;
        weak[5].3 = Some((30, 2));
        assert_eq!(check(&report(&weak, json!(null))).unwrap(), 1);
        assert_eq!(
            check(&report(&FULL_RUN[..5], json!(null))).unwrap(),
            1,
            "no strength runs"
        );

        // The verdicts the why section quotes are the recorded ones.
        assert_eq!(check(&report(&undo_failed, why.clone())).unwrap(), 1);
        let mut quoted = why.clone();
        quoted["commit_without_minimal"] = json!("bad");
        let quiet = write_report(dir, &FULL_RUN, |report| report["why"] = quoted);
        assert_eq!(check(&quiet).unwrap(), 1);
        // Words that are not verdicts are not held against a run.
        let mut rewritten = why.clone();
        rewritten["undo_on_bad"] = json!("rewritten");
        rewritten["undo_how"] = Value::Null;
        rewritten["commit_without_minimal"] = json!("does not apply");
        let unheld = write_report(dir, &TWO_ENDS, |report| report["why"] = rewritten);
        assert_eq!(check(&unheld).unwrap(), 0);
    }

    #[test]
    fn a_single_change_is_isolated_by_its_boundary() {
        let dir = tempfile::tempdir().unwrap();
        // One change: no `without` run exists, and none is asked for.
        let single = write_report(dir.path(), &TWO_ENDS, |report| {
            report["verdict_card"] = card(&[("BOUNDARY", true), ("ISOLATED", true)]);
            report["why"] = json!({
                "changes_in_commit": 1,
                "commit_without_minimal": null,
                "undo_how": null,
                "undo_on_bad": "rewritten",
                "minimal_failing_changes": [{"file": "calc.sh", "line": 15}],
            });
        });
        assert_eq!(check(&single).unwrap(), 0);
        assert_eq!(
            backed("ISOLATED", &json!({"why": {"changes_in_commit": 1}}), &[]),
            None
        );
        assert_eq!(
            backed("ISOLATED", &json!({"why": {"changes_in_commit": 3}}), &[]),
            Some(false)
        );
        assert_eq!(backed("LEAD", &json!({}), &[]), None);
        assert_eq!(backed("BOUNDARY", &json!({}), &[]), None);
    }
}
