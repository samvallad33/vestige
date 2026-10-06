//! Real `vestige prove`, end to end and offline: a small git repo in which
//! one commit breaks a shell test, a fresh Strata store holding a memory for
//! some of the commits and for the failure, and the whole flow run by the
//! binary. The flow re-invokes the binary through `git bisect run`, so it
//! can only be tested against the built `vestige`.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};
use tempfile::TempDir;

struct Ran {
    ok: bool,
    stdout: String,
    stderr: String,
}

impl Ran {
    fn text(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

/// Git that reads no user or system configuration, with a fixed identity
/// and commit date, so the fixture and the run are the same everywhere.
fn hermetic_git(command: &mut Command, date: &str) {
    command
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_AUTHOR_NAME", "Walk Verify")
        .env("GIT_AUTHOR_EMAIL", "walk-verify@example.com")
        .env("GIT_COMMITTER_NAME", "Walk Verify")
        .env("GIT_COMMITTER_EMAIL", "walk-verify@example.com")
        .env("GIT_AUTHOR_DATE", date)
        .env("GIT_COMMITTER_DATE", date);
}

/// After every commit of the fixture.
const NOW: &str = "2026-02-01T12:00:00+00:00";

fn git(repo: &Path, date: &str, args: &[&str]) -> String {
    let mut command = Command::new("git");
    command.arg("-C").arg(repo).args(args);
    hermetic_git(&mut command, date);
    let output = command.output().expect("spawn git");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn vestige(data_dir: &Path, args: &[&str]) -> Ran {
    let mut command = Command::new(env!("CARGO_BIN_EXE_vestige"));
    command
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env("NO_COLOR", "1")
        .env("CLICOLOR", "0")
        .env_remove("FORCE_COLOR")
        // Belt and braces: the flag already names the store, and the
        // variable makes sure nothing can fall back to a real one.
        .env("VESTIGE_DATA_DIR", data_dir);
    hermetic_git(&mut command, NOW);
    let output = command.output().expect("spawn vestige");
    Ran {
        ok: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn path_arg(path: &Path) -> &str {
    path.to_str().expect("utf-8 temp path")
}

/// `vestige ingest`, returning the new memory's id.
fn ingest(store: &Path, content: &str, tags: &str) -> String {
    let ran = vestige(
        store,
        &["ingest", content, "--tags", tags, "--node-type", "event"],
    );
    assert!(ran.ok, "{}", ran.text());
    ran.stdout
        .lines()
        .find_map(|line| line.strip_prefix("Node ID: "))
        .unwrap_or_else(|| panic!("no node id in: {}", ran.text()))
        .trim()
        .to_string()
}

const README: &str = "# tinyproj\nA tiny calculator.\nStatus: stable\n";

/// `calc.sh` with its two edit sites far enough apart to be two hunks.
fn calc(header: &str, operator: &str, filler_six: &str) -> String {
    let mut text = format!("#!/bin/sh\n# {header}\n# usage: sh calc.sh A B\na=\"$1\"\nb=\"$2\"\n");
    for n in 1..=12 {
        if n == 6 {
            text.push_str(&format!("# filler {filler_six}\n"));
        } else {
            text.push_str(&format!("# filler {n}\n"));
        }
    }
    text.push_str(&format!("sum=$((a {operator} b))\necho \"$sum\"\n"));
    text
}

fn notes(count: usize) -> String {
    let mut text = String::from("notes\n");
    for n in 1..=count {
        text.push_str(&format!("note {n}\n"));
    }
    text
}

struct Fixture {
    _dir: TempDir,
    repo: PathBuf,
    store: PathBuf,
    /// c0 (tagged v1) to c12 (tagged v2).
    commits: Vec<String>,
    failure: String,
}

/// The commit that breaks the test.
const BREAKING: usize = 5;

/// The commits that have a memory, so the walk can reach them: the good end
/// (outside the range), one before the breaking commit, the breaking commit,
/// two after it, and the bad end (committed after the report).
const REMEMBERED: [usize; 6] = [0, 2, BREAKING, 7, 9, 12];

/// Thirteen commits, v1 = c0 and v2 = c12. c5 makes four changes (a new
/// file, a README line and two hunks of calc.sh) of which one, the second
/// calc.sh hunk, turns the sum into a difference. c12 is committed after the
/// failure was reported. With `later_commit_conflicts`, c7 rewrites the
/// README line c5 changed, so reverting all of c5 on v2 conflicts.
fn fixture(later_commit_conflicts: bool) -> Fixture {
    let dir = TempDir::new().expect("temp dir");
    let repo = dir.path().join("tinyproj");
    let store = dir.path().join("store");
    std::fs::create_dir_all(&repo).unwrap();
    let day = |n: u32| format!("2026-01-{n:02}T12:00:00+00:00");
    git(&repo, &day(1), &["init", "-q", "-b", "main"]);

    let write = |name: &str, text: &str| std::fs::write(repo.join(name), text).unwrap();
    let mut commits = Vec::new();
    let mut commit = |date: &str, subject: &str| {
        git(&repo, date, &["add", "-A"]);
        git(&repo, date, &["commit", "-q", "-m", subject]);
        commits.push(git(&repo, date, &["rev-parse", "HEAD"]));
    };

    write("calc.sh", &calc("calc.sh adds two numbers", "+", "6"));
    write("README.md", README);
    write("notes.txt", &notes(0));
    commit(&day(1), "Add the calculator");
    git(&repo, &day(1), &["tag", "v1"]);

    write("notes.txt", &notes(1));
    commit(&day(2), "Take a note");
    write("README.md", &format!("{README}More docs.\n"));
    commit(&day(3), "Document more");
    write("calc.sh", &calc("calc.sh adds two numbers", "+", "six"));
    commit(&day(4), "Spell out a filler line");
    write("notes.txt", &notes(2));
    commit(&day(5), "Take a second note");

    write("calc.sh", &calc("calc.sh adds two integers", "-", "six"));
    write(
        "README.md",
        "# tinyproj\nA tiny calculator.\nStatus: reworked\nMore docs.\n",
    );
    write("CHANGES.txt", "reworked the calculator\n");
    commit(&day(6), "Rework the calculator");

    write("notes.txt", &notes(3));
    commit(&day(7), "Take a third note");
    if later_commit_conflicts {
        write(
            "README.md",
            "# tinyproj\nA tiny calculator.\nStatus: reworked twice\nMore docs.\n",
        );
        commit(&day(8), "Update the status line");
    } else {
        write("notes.txt", &notes(4));
        commit(&day(8), "Take a fourth note");
    }
    write("extra.txt", "extra\n");
    commit(&day(9), "Add an extra file");
    write("extra.txt", "extra\nmore\n");
    commit(&day(10), "Extend the extra file");
    write("notes.txt", &notes(5));
    commit(&day(11), "Take a fifth note");
    write("notes.txt", &notes(6));
    commit(&day(12), "Take a sixth note");
    write("extra.txt", "extra\nmore\nand more\n");
    commit(&day(20), "Extend the extra file again");
    git(&repo, &day(20), &["tag", "v2"]);
    assert_eq!(commits.len(), 13);

    // One memory per remembered commit, one for a sha this repo never had,
    // then the failure. The shared tag is what joins them: ingest
    // auto-connects each new memory to the earlier ones that carry it.
    for index in REMEMBERED {
        let sha = &commits[index];
        let subject = git(&repo, NOW, &["show", "-s", "--format=%s", sha]);
        ingest(
            &store,
            &format!("Commit {}: {subject}", &sha[..10]),
            "tinyproj-calc",
        );
    }
    ingest(
        &store,
        "Commit deadbeef99: a commit from some other repository",
        "tinyproj-calc",
    );
    let failure = ingest(
        &store,
        "Failure report: sh calc.sh 2 3 prints -1 on v2, it printed 5 on v1",
        "tinyproj-calc",
    );
    Fixture {
        _dir: dir,
        repo,
        store,
        commits,
        failure,
    }
}

/// Passes while calc.sh adds, fails once it subtracts.
const TEST: &str = r#"out=$(sh calc.sh 2 3); echo "2+3=$out"; test "$out" = 5"#;

/// The failure was reported between c11 and c12.
const REPORTED_AT: &str = "2026-01-15T00:00:00Z";

fn prove(fixture: &Fixture, test: &str, report: &Path) -> Ran {
    vestige(
        &fixture.store,
        &[
            "prove",
            "--logged-write",
            &fixture.failure,
            "--repo",
            path_arg(&fixture.repo),
            "--good",
            "v1",
            "--bad",
            "v2",
            "--test",
            test,
            "--reported-at",
            REPORTED_AT,
            "--slug",
            "tinyproj",
            "--report",
            path_arg(report),
        ],
    )
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read report")).expect("report json")
}

/// The whole flow on the fixture, and everything the report must then say.
fn proves_the_breaking_commit(later_commit_conflicts: bool, undo_how: &str) {
    let fixture = fixture(later_commit_conflicts);
    let breaking = &fixture.commits[BREAKING];
    let report_path = fixture._dir.path().join("report.json");
    let head_before = git(&fixture.repo, NOW, &["rev-parse", "HEAD"]);

    let ran = prove(&fixture, TEST, &report_path);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);

    // The walk's leads and the time gate.
    assert_eq!(report["walk"]["reached"], 7, "{}", ran.text());
    assert_eq!(report["walk"]["kept"], 4);
    let mut dropped: Vec<&str> = report["walk"]["dropped"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["why"].as_str().unwrap())
        .collect();
    dropped.sort_unstable();
    assert_eq!(
        dropped,
        [
            "committed 2026-01-20, after the report",
            "not a commit in this repo",
            "outside v1..v2",
        ]
    );
    assert!(ran.stdout.contains("[recorded link] #"), "{}", ran.stdout);
    assert!(
        ran.stdout
            .contains("dropped deadbeef99: not a commit in this repo"),
        "{}",
        ran.stdout
    );

    // The tested answer.
    assert_eq!(
        report["first_bad_commit"],
        breaking.as_str(),
        "{}",
        ran.text()
    );
    assert_eq!(report["window_commits"], 12);
    assert!(
        report["walk_lead_rank"]
            .as_u64()
            .is_some_and(|rank| rank >= 1)
    );
    // Four leads: the breaking commit, the lead before it, then its parent.
    assert_eq!(report["found_in_runs"], 3);
    let plain = report["plain_bisect_runs_replayed"]
        .as_u64()
        .expect("a replayed count");
    assert!(
        (3..=4).contains(&plain),
        "12 commits bisect in 3 or 4 runs, not {plain}"
    );
    assert!(
        ran.stdout.contains(&format!(
            "[tested] git bisect: {} is the first bad commit",
            &breaking[..10]
        )),
        "{}",
        ran.stdout
    );
    assert!(
        ran.stdout.contains(&format!(
            "Found in 3 test runs on the walk's 4 leads. Plain git bisect needs {plain} on the same 12 commits"
        )),
        "{}",
        ran.stdout
    );
    assert!(ran.stdout.contains("reused probe"), "{}", ran.stdout);

    // Why: one of the four changes, the rest passes without it, undoing it
    // on the bad ref passes.
    let why = &report["why"];
    assert_eq!(why["changes_in_commit"], 4);
    assert_eq!(why["search_complete"], true);
    assert_eq!(
        why["minimal_failing_changes"],
        json!([{"file": "calc.sh", "line": 15, "added": ["sum=$((a - b))"]}])
    );
    assert_eq!(why["commit_without_minimal"], "good");
    assert_eq!(why["undo_on_bad"], "good");
    assert_eq!(why["undo_how"], undo_how);
    let patch_path = fixture._dir.path().join("report.undo.patch");
    assert_eq!(why["undo_patch"]["file"], "report.undo.patch");
    assert_eq!(why["undo_patch"]["applies_to"], "v2");
    let patch = std::fs::read_to_string(&patch_path).expect("the undo patch beside the report");
    assert!(patch.contains("-sum=$((a - b))"), "{patch}");
    assert!(patch.contains("+sum=$((a + b))"), "{patch}");
    assert_eq!(
        patch.contains("CHANGES.txt"),
        undo_how == "whole commit",
        "{patch}"
    );

    // The protocol was frozen before the first test.
    let protocol = &report["protocol"];
    assert_eq!(protocol["candidates"].as_array().unwrap().len(), 4);
    assert_eq!(
        protocol["rule"],
        "exit 0 good, 125 cannot test, any other code bad"
    );
    assert!(protocol["frozen_at"].as_str().unwrap() <= report["probes"][0]["at"].as_str().unwrap());
    assert!(
        ran.stdout.contains("Protocol frozen before any test"),
        "{}",
        ran.stdout
    );

    // The verdict card: five rungs, all of them hold.
    let rungs: Vec<(&str, bool)> = report["verdict_card"]
        .as_array()
        .unwrap()
        .iter()
        .map(|rung| (rung["rung"].as_str().unwrap(), rung["holds"] == true))
        .collect();
    assert_eq!(
        rungs,
        [
            ("LEAD", true),
            ("BOUNDARY", true),
            ("CONFIRMED", true),
            ("ISOLATED", true),
            ("REVERSED", true),
        ]
    );
    assert!(ran.stdout.contains("5 of 5 rungs hold"), "{}", ran.stdout);

    // The probe log: numbered, chained, one memory per run. The bisect
    // phase holds the runs `git bisect run` made through the hidden
    // subcommand, on commits no earlier step had tested.
    let probes = report["probes"].as_array().unwrap();
    let phases: Vec<&str> = probes
        .iter()
        .map(|probe| probe["phase"].as_str().unwrap())
        .collect();
    for phase in [
        "baseline",
        "candidate",
        "parent",
        "bisect",
        "lines",
        "without",
        "undo",
    ] {
        assert!(phases.contains(&phase), "no {phase} run in {phases:?}");
    }
    assert_eq!(probes[0]["oracle_said"], "2+3=5");
    assert_eq!(probes[1]["oracle_said"], "2+3=-1");
    assert_eq!(report["chain_head"], probes.last().unwrap()["hash"]);
    assert_eq!(report["oracle"]["file"], "test-command.sh");
    let storage = vestige_mcp::strata_memory::open(&fixture.store).expect("open the store");
    for (index, probe) in probes.iter().enumerate() {
        assert_eq!(probe["n"], index + 1);
        let id = probe["memory"].as_str().expect("every run is a memory");
        let node = storage.get_node(id).unwrap().expect("the probe memory");
        assert_eq!(node.node_type, "event");
        assert_eq!(node.source.as_deref(), Some("walk-verify"));
        let verdict = probe["verdict"].as_str().unwrap();
        assert_eq!(
            node.tags,
            ["oracle-probe", &format!("probe-{verdict}"), "tinyproj"]
        );
        assert!(
            node.content.starts_with("Oracle probe: "),
            "{}",
            node.content
        );
    }
    for (id, tag) in [
        (report["result_memory"].as_str(), "oracle-result"),
        (report["protocol"]["memory"].as_str(), "oracle-protocol"),
    ] {
        let id = id.unwrap_or_else(|| panic!("no {tag} memory"));
        let node = storage.get_node(id).unwrap().expect("the memory");
        assert_eq!(node.tags, [tag, "tinyproj"]);
    }
    drop(storage);

    // The user's checkout was never touched and the worktree is gone.
    assert_eq!(git(&fixture.repo, NOW, &["rev-parse", "HEAD"]), head_before);
    assert_eq!(git(&fixture.repo, NOW, &["status", "--porcelain"]), "");
    assert_eq!(
        git(&fixture.repo, NOW, &["worktree", "list"])
            .lines()
            .count(),
        1
    );
    assert_eq!(
        git(&fixture.repo, NOW, &["branch", "--show-current"]),
        "main"
    );

    // The report re-verifies, a second run will not overwrite it, and a
    // copy with one verdict changed does not pass.
    let checked = vestige(
        &fixture.store,
        &["prove", "--check", path_arg(&report_path)],
    );
    assert!(checked.ok, "{}", checked.text());
    for line in [
        "hash chain intact",
        ": recorded verdict bad",
        ": recorded verdict good",
        ": hash matches",
        "REVERSED  yes",
        "undo patch report.undo.patch: sha256 matches the report",
    ] {
        assert!(checked.stdout.contains(line), "{line}: {}", checked.stdout);
    }
    let again = prove(&fixture, TEST, &report_path);
    assert!(!again.ok, "{}", again.text());
    assert!(
        again.stdout.contains("refusing to overwrite"),
        "{}",
        again.text()
    );

    let mut tampered = report.clone();
    let changed = tampered["probes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|probe| probe["commit"] == breaking.as_str())
        .expect("a run on the breaking commit");
    changed["verdict"] = json!("good");
    let tampered_path = fixture._dir.path().join("tampered.json");
    std::fs::write(
        &tampered_path,
        serde_json::to_string_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let rejected = vestige(
        &fixture.store,
        &["prove", "--check", path_arg(&tampered_path)],
    );
    assert!(!rejected.ok, "{}", rejected.text());
    assert!(
        rejected.stdout.contains("hash does not match"),
        "{}",
        rejected.text()
    );
}

#[test]
fn walk_verify_prove_finds_the_breaking_commit_and_undoes_only_its_bad_lines() {
    proves_the_breaking_commit(true, "found lines only");
}

#[test]
fn walk_verify_prove_undoes_the_whole_commit_when_it_still_reverts_cleanly() {
    proves_the_breaking_commit(false, "whole commit");
}

#[test]
fn walk_verify_prove_stops_when_the_test_cannot_tell_the_ends_apart() {
    let fixture = fixture(false);
    let report_path = fixture._dir.path().join("report.json");
    let ran = prove(&fixture, "true", &report_path);
    assert!(!ran.ok, "{}", ran.text());
    assert!(
        ran.stdout.contains(
            "The script does not pass on v1 and fail on v2, so nothing can be decided with it."
        ),
        "{}",
        ran.text()
    );
    assert!(!report_path.exists(), "no report for an undecided run");
    // The worktree is removed on this way out too.
    assert_eq!(
        git(&fixture.repo, NOW, &["worktree", "list"])
            .lines()
            .count(),
        1
    );
}
