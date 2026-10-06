//! Real `vestige prove`, end to end and offline, from the walk to the check.
//!
//! Every test builds a small git repository in which one commit breaks a
//! shell test, and a fresh Strata store that is filled only the way a user
//! fills one: `vestige ingest`, once per commit and once for the failure
//! report, with the files each one names as its tags. The edges the walk
//! follows are the ones ingest writes by itself when two memories record
//! the same file; no test writes an edge. Then the built binary runs
//! `vestige prove` and `vestige prove --check`. The flow re-invokes the
//! binary through `git bisect run`, so it can only be tested against the
//! real `vestige`.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tempfile::TempDir;

struct Ran {
    ok: bool,
    code: Option<i32>,
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
        .env("GIT_COMMITTER_DATE", date)
        .env("LC_ALL", "C");
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

/// The `vestige` under test, pointed at `data_dir` and nothing else.
fn vestige_command(binary: &Path, data_dir: &Path) -> Command {
    let mut command = Command::new(binary);
    command
        .arg("--data-dir")
        .arg(data_dir)
        .env("NO_COLOR", "1")
        .env("CLICOLOR", "0")
        .env_remove("FORCE_COLOR")
        // Belt and braces: the flag already names the store, and the
        // variable makes sure nothing can fall back to a real one.
        .env("VESTIGE_DATA_DIR", data_dir);
    hermetic_git(&mut command, NOW);
    command
}

fn ran(mut command: Command) -> Ran {
    let output = command.output().expect("spawn vestige");
    Ran {
        ok: output.status.success(),
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn vestige(data_dir: &Path, args: &[&str]) -> Ran {
    let mut command = vestige_command(Path::new(env!("CARGO_BIN_EXE_vestige")), data_dir);
    command.args(args);
    ran(command)
}

fn path_arg(path: &Path) -> &str {
    path.to_str().expect("utf-8 temp path")
}

/// `vestige ingest`, returning the new memory's id.
fn ingest(store: &Path, content: &str, tags: &str) -> String {
    let ran = vestige(
        store,
        &[
            "ingest",
            content,
            "--tags",
            tags,
            "--node-type",
            "event",
            "--source",
            "git-log",
        ],
    );
    assert!(ran.ok, "{}", ran.text());
    ran.stdout
        .lines()
        .find_map(|line| line.strip_prefix("Node ID: "))
        .unwrap_or_else(|| panic!("no node id in: {}", ran.text()))
        .trim()
        .to_string()
}

/// The tags of a memory about these files: each file's name and its stem,
/// the exact identities ingest joins two memories on.
fn file_tags(files: &[&str]) -> String {
    let mut tags: Vec<&str> = Vec::new();
    for file in files {
        let stem = file.rsplit_once('.').map_or(*file, |(stem, _)| stem);
        for tag in [*file, stem] {
            if !tags.contains(&tag) {
                tags.push(tag);
            }
        }
    }
    tags.join(",")
}

const README: &str = "# tinyproj\nA tiny calculator.\nStatus: stable\n";

/// `calc.sh`: a header, twelve filler lines, the sum. Its edit sites are
/// far enough apart to be separate hunks.
fn calc(header: &str, operator: &str, spelled: &[u32]) -> String {
    const WORDS: [&str; 13] = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve",
    ];
    let mut text = format!("#!/bin/sh\n# {header}\n# usage: sh calc.sh A B\na=\"$1\"\nb=\"$2\"\n");
    for n in 1..=12u32 {
        if spelled.contains(&n) {
            text.push_str(&format!("# filler {}\n", WORDS[n as usize]));
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
    dir: TempDir,
    repo: PathBuf,
    store: PathBuf,
    /// c0 (tagged v1) to c12 (tagged v2).
    commits: Vec<String>,
    failure: String,
}

/// The commit that breaks the test.
const BREAKING: usize = 5;

struct Plan {
    /// c7 rewrites the README line c5 changed, so reverting all of c5 on
    /// v2 conflicts.
    later_commit_conflicts: bool,
    /// The commits that get a memory.
    remembered: Vec<usize>,
    /// The file the failure report names.
    failure_names: &'static str,
    /// Prefix of the temp directory everything lives in.
    dir_prefix: &'static str,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            later_commit_conflicts: false,
            remembered: (0..=12).collect(),
            failure_names: "calc.sh",
            dir_prefix: "walk-verify-",
        }
    }
}

/// Thirteen commits on `main`, v1 = c0 and v2 = c12, plus a tag `side` on a
/// branch that leaves `main` at c2.
///
/// c5 makes four changes (a new file, a README line and two hunks of
/// calc.sh) of which one, the second calc.sh hunk, turns the sum into a
/// difference. c3, c10 and c12 make harmless edits to calc.sh; c12 is
/// committed after the failure was reported.
///
/// The store gets one memory per remembered commit ("Commit <sha>: subject.
/// Touched: files", tagged with those files), one for a commit of some
/// other repository that also touched calc.sh, and last the failure report,
/// tagged with the file it names. A failure that names calc.sh is therefore
/// one recorded link from c0, c3, c5, c10 and c12, and two links from the
/// older commits that share a file with one of those.
fn fixture(plan: &Plan) -> Fixture {
    let dir = tempfile::Builder::new()
        .prefix(plan.dir_prefix)
        .tempdir()
        .expect("temp dir");
    let repo = dir.path().join("tinyproj");
    let store = dir.path().join("store");
    std::fs::create_dir_all(&repo).unwrap();
    let day = |n: u32| format!("2026-01-{n:02}T12:00:00+00:00");
    git(&repo, &day(1), &["init", "-q", "-b", "main"]);

    let write = |name: &str, text: &str| std::fs::write(repo.join(name), text).unwrap();
    let mut commits: Vec<String> = Vec::new();
    let mut touched: Vec<Vec<&str>> = Vec::new();
    let mut commit = |date: &str, subject: &str, files: &[&'static str]| {
        git(&repo, date, &["add", "-A"]);
        git(&repo, date, &["commit", "-q", "-m", subject]);
        commits.push(git(&repo, date, &["rev-parse", "HEAD"]));
        touched.push(files.to_vec());
    };
    let adds = "calc.sh adds two numbers";

    write("calc.sh", &calc(adds, "+", &[]));
    write("README.md", README);
    write("notes.txt", &notes(0));
    commit(
        &day(1),
        "Add the calculator",
        &["calc.sh", "README.md", "notes.txt"],
    );
    git(&repo, &day(1), &["tag", "v1"]);

    write("notes.txt", &notes(1));
    commit(&day(2), "Take a note", &["notes.txt"]);
    write("README.md", &format!("{README}More docs.\n"));
    commit(&day(3), "Document more", &["README.md"]);
    write("calc.sh", &calc(adds, "+", &[6]));
    commit(&day(4), "Spell out a filler line", &["calc.sh"]);
    write("notes.txt", &notes(2));
    commit(&day(5), "Take a second note", &["notes.txt"]);

    write("calc.sh", &calc("calc.sh adds two integers", "-", &[6]));
    write(
        "README.md",
        "# tinyproj\nA tiny calculator.\nStatus: reworked\nMore docs.\n",
    );
    write("CHANGES.txt", "reworked the calculator\n");
    commit(
        &day(6),
        "Rework the calculator",
        &["calc.sh", "README.md", "CHANGES.txt"],
    );

    write("notes.txt", &notes(3));
    commit(&day(7), "Take a third note", &["notes.txt"]);
    if plan.later_commit_conflicts {
        write(
            "README.md",
            "# tinyproj\nA tiny calculator.\nStatus: reworked twice\nMore docs.\n",
        );
        commit(&day(8), "Update the status line", &["README.md"]);
    } else {
        write("notes.txt", &notes(4));
        commit(&day(8), "Take a fourth note", &["notes.txt"]);
    }
    write("extra.txt", "extra\n");
    commit(&day(9), "Add an extra file", &["extra.txt"]);
    write("extra.txt", "extra\nmore\n");
    commit(&day(10), "Extend the extra file", &["extra.txt"]);
    write("notes.txt", &notes(5));
    write("calc.sh", &calc("calc.sh adds two integers", "-", &[6, 9]));
    commit(
        &day(11),
        "Take a fifth note, spell out another filler",
        &["notes.txt", "calc.sh"],
    );
    write("notes.txt", &notes(6));
    commit(&day(12), "Take a sixth note", &["notes.txt"]);
    write("extra.txt", "extra\nmore\nand more\n");
    write(
        "calc.sh",
        &calc("calc.sh adds two integers", "-", &[3, 6, 9]),
    );
    commit(
        &day(20),
        "Extend the extra file again, spell out a third filler",
        &["extra.txt", "calc.sh"],
    );
    git(&repo, &day(20), &["tag", "v2"]);
    assert_eq!(commits.len(), 13);

    // A branch that leaves main at c2: its tip passes the test and is not
    // an ancestor of v2.
    git(&repo, &day(4), &["checkout", "-q", "-b", "side-branch", &commits[2]]);
    std::fs::write(repo.join("side.txt"), "side\n").unwrap();
    git(&repo, &day(4), &["add", "-A"]);
    git(&repo, &day(4), &["commit", "-q", "-m", "A side note"]);
    git(&repo, &day(4), &["tag", "side"]);
    git(&repo, &day(4), &["checkout", "-q", "main"]);

    for &index in &plan.remembered {
        let sha = &commits[index];
        let subject = git(&repo, NOW, &["show", "-s", "--format=%s", sha]);
        ingest(
            &store,
            &format!(
                "Commit {}: {subject}. Touched: {}",
                &sha[..10],
                touched[index].join(" ")
            ),
            &file_tags(&touched[index]),
        );
    }
    ingest(
        &store,
        "Commit deadbeef99: a commit from some other repository. Touched: calc.sh",
        &file_tags(&["calc.sh"]),
    );
    let failure = ingest(
        &store,
        &format!(
            "Failure report: sh {} 2 3 prints -1 on v2, it printed 5 on v1",
            plan.failure_names
        ),
        &format!("failure,{}", file_tags(&[plan.failure_names])),
    );
    Fixture {
        dir,
        repo,
        store,
        commits,
        failure,
    }
}

impl Fixture {
    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// The arguments of a `prove` on v1..v2 that writes `report`.
    fn prove_args(&self, report: &Path) -> Vec<String> {
        [
            "prove",
            "--logged-write",
            &self.failure,
            "--repo",
            path_arg(&self.repo),
            "--good",
            "v1",
            "--bad",
            "v2",
            "--reported-at",
            REPORTED_AT,
            "--slug",
            "tinyproj",
            "--report",
            path_arg(report),
        ]
        .map(str::to_string)
        .to_vec()
    }

    /// `prove` on v1..v2 with the one-line test and `extra` arguments.
    fn prove(&self, report: &Path, extra: &[&str]) -> Ran {
        let mut args = self.prove_args(report);
        args.extend(["--test", TEST].map(str::to_string));
        args.extend(extra.iter().map(|arg| (*arg).to_string()));
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        vestige(&self.store, &args)
    }

    fn check(&self, report: &Path) -> Ran {
        vestige(&self.store, &["prove", "--check", path_arg(report)])
    }

    fn short(&self, index: usize) -> &str {
        &self.commits[index][..10]
    }

    /// One worktree: the user's own. Nothing of a run is left registered.
    fn assert_no_worktree_left(&self) {
        assert_eq!(
            git(&self.repo, NOW, &["worktree", "list"]).lines().count(),
            1
        );
        assert!(!self.repo.join(".git/BISECT_LOG").exists());
        assert!(!self.repo.join(".git/BISECT_START").exists());
    }
}

/// Passes while calc.sh adds, fails once it subtracts.
const TEST: &str = r#"out=$(sh calc.sh 2 3); echo "2+3=$out"; test "$out" = 5"#;

/// The failure was reported between c11 and c12.
const REPORTED_AT: &str = "2026-01-15T00:00:00Z";

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read report")).expect("report json")
}

/// `(rung, holds)` of each row of a report's verdict card.
fn rungs(report: &Value) -> Vec<(String, bool)> {
    report["verdict_card"]
        .as_array()
        .expect("a verdict card")
        .iter()
        .map(|rung| (rung["rung"].as_str().unwrap().to_string(), rung["holds"] == true))
        .collect()
}

fn rung<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["verdict_card"]
        .as_array()
        .expect("a verdict card")
        .iter()
        .find(|rung| rung["rung"] == name)
        .unwrap_or_else(|| panic!("no {name} rung"))
}

/// The probe on one commit.
fn probe_on<'a>(report: &'a Value, commit: &str) -> &'a Value {
    report["probes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|probe| probe["commit"] == commit)
        .unwrap_or_else(|| panic!("no run on {commit}"))
}

fn phases(report: &Value) -> Vec<&str> {
    report["probes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|probe| probe["phase"].as_str().unwrap())
        .collect()
}

/// One cause of `vestige causal-walk`: its rank, memory, depth, and the
/// commit its content line names.
#[derive(Debug, PartialEq)]
struct Cause {
    rank: u64,
    memory: String,
    depth: u64,
    sha: Option<String>,
}

/// The causes plain `vestige causal-walk --logged-write` prints, read from
/// its text the way a person reads it: `#3 mem-... depth 1`, then the line
/// with the memory's content.
fn walk_printed(store: &Path, failure: &str) -> Vec<Cause> {
    let ran = vestige(store, &["causal-walk", "--logged-write", failure]);
    assert!(ran.ok, "{}", ran.text());
    let mut causes = Vec::new();
    let mut lines = ran.stdout.lines();
    while let Some(line) = lines.next() {
        let Some(rest) = line.strip_prefix('#') else {
            continue;
        };
        let fields: Vec<&str> = rest.split_whitespace().collect();
        let [rank, memory, "depth", depth] = fields.as_slice() else {
            panic!("unexpected walk line: {line}");
        };
        let content = lines.next().unwrap_or_default().trim();
        let sha = content.strip_prefix("Commit ").map(|rest| {
            rest.chars()
                .take_while(char::is_ascii_hexdigit)
                .collect::<String>()
        });
        causes.push(Cause {
            rank: rank.parse().unwrap(),
            memory: (*memory).to_string(),
            depth: depth.parse().unwrap(),
            sha,
        });
    }
    assert!(!causes.is_empty(), "the walk printed no cause: {}", ran.text());
    causes
}

/// Everything that identifies the state of the user's checkout.
fn checkout_state(repo: &Path) -> Vec<String> {
    vec![
        git(repo, NOW, &["rev-parse", "HEAD"]),
        git(repo, NOW, &["branch", "--show-current"]),
        git(repo, NOW, &["status", "--porcelain"]),
        git(repo, NOW, &["for-each-ref"]),
        git(repo, NOW, &["stash", "list"]),
        git(repo, NOW, &["diff"]),
        git(repo, NOW, &["diff", "--cached"]),
    ]
}

#[test]
fn walk_verify_prove_goes_from_the_walk_to_a_checked_report() {
    let fixture = fixture(&Plan::default());
    let breaking = &fixture.commits[BREAKING];
    let report_path = fixture.path("report.json");

    // The user is in the middle of something: a modified file, a staged
    // file and an untracked one.
    std::fs::write(fixture.repo.join("notes.txt"), "notes\nmine, unsaved\n").unwrap();
    std::fs::write(fixture.repo.join("staged.txt"), "staged\n").unwrap();
    git(&fixture.repo, NOW, &["add", "staged.txt"]);
    std::fs::write(fixture.repo.join("untracked.txt"), "untracked\n").unwrap();
    let before = checkout_state(&fixture.repo);

    // The walk on its own, before anything is proved.
    let walk = walk_printed(&fixture.store, &fixture.failure);

    let ran = fixture.prove(&report_path, &[]);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);

    // The leads `prove` used are the walk's causes: the same memories, at
    // the same ranks and depths, in the same order, each naming its commit.
    let named: Vec<&Cause> = walk.iter().filter(|cause| cause.sha.is_some()).collect();
    assert_eq!(report["walk"]["reached"], named.len(), "{}", ran.text());
    let candidates = report["walk"]["candidates"].as_array().unwrap();
    assert_eq!(report["walk"]["kept"], candidates.len());
    let mut last_rank = 0;
    for candidate in candidates {
        let rank = candidate["rank"].as_u64().unwrap();
        assert!(rank > last_rank, "leads out of walk order: {candidates:?}");
        last_rank = rank;
        let cause = &walk[rank as usize - 1];
        assert_eq!(cause.rank, rank);
        assert_eq!(candidate["memory"], cause.memory.as_str());
        assert_eq!(candidate["depth"], cause.depth);
        let sha = cause.sha.as_deref().expect("a lead names a commit");
        assert!(
            candidate["commit"].as_str().unwrap().starts_with(sha),
            "{candidate} is not {sha}"
        );
    }
    // Every dropped lead is one of the walk's causes too.
    for row in report["walk"]["dropped"].as_array().unwrap() {
        let sha = row["commit"].as_str().unwrap();
        assert!(
            named.iter().any(|cause| cause.sha.as_deref() == Some(sha)),
            "{sha} was dropped but the walk never printed it"
        );
    }
    // The protocol froze exactly those leads, in that order.
    let frozen: Vec<&Value> = candidates.iter().map(|lead| &lead["commit"]).collect();
    let protocol = &report["protocol"];
    assert_eq!(
        protocol["candidates"].as_array().unwrap().iter().collect::<Vec<_>>(),
        frozen
    );

    // What the walk reaches here: one link away the commits that touched
    // calc.sh, two links away the older commits that share a file with one
    // of those.
    let depth_of = |index: usize| {
        candidates
            .iter()
            .find(|lead| lead["commit"] == fixture.commits[index].as_str())
            .map(|lead| lead["depth"].as_u64().unwrap())
    };
    for near in [3, BREAKING, 10] {
        assert_eq!(depth_of(near), Some(1), "c{near}");
    }
    for far in [1, 2, 4, 6, 7, 8, 9] {
        assert_eq!(depth_of(far), Some(2), "c{far}");
    }
    assert_eq!(candidates.len(), 10);
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

    // Closest links first: the three leads one link away hold the
    // boundary, so the seven two links away are never looked at. Three
    // runs: the breaking commit, the lead before it, its parent.
    assert!(
        ran.stdout
            .contains("leads within 1 link of the report: 3"),
        "{}",
        ran.stdout
    );
    assert!(!ran.stdout.contains("leads within 2 links"), "{}", ran.stdout);
    assert!(
        ran.stdout.contains(&format!(
            "{} fails and its parent passes.",
            fixture.short(BREAKING)
        )),
        "{}",
        ran.stdout
    );
    assert_eq!(report["found_in_runs"], 3);
    let step_four: Vec<(&str, &str)> = report["probes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|probe| probe["phase"] == "candidate" || probe["phase"] == "parent")
        .map(|probe| {
            (
                probe["commit"].as_str().unwrap(),
                probe["verdict"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        step_four,
        [
            (breaking.as_str(), "bad"),
            (fixture.commits[3].as_str(), "good"),
            (fixture.commits[4].as_str(), "good"),
        ]
    );

    // The tested answer.
    assert_eq!(
        report["first_bad_commit"],
        breaking.as_str(),
        "{}",
        ran.text()
    );
    assert_eq!(report["tool"], "vestige prove (walk-verify 7)");
    assert_eq!(report["window_commits"], 12);
    assert_eq!(
        report["walk_lead_rank"],
        candidates
            .iter()
            .find(|lead| lead["commit"] == breaking.as_str())
            .unwrap()["rank"]
    );
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
            "Found in 3 test runs on the walk's 10 leads. Plain git bisect needs {plain} on the same 12 commits"
        )),
        "{}",
        ran.stdout
    );
    assert!(ran.stdout.contains("reused probe"), "{}", ran.stdout);

    // Why: one of the four changes, the rest passes without it, undoing
    // the commit on the bad ref passes.
    let why = &report["why"];
    assert_eq!(why["changes_in_commit"], 4);
    assert_eq!(why["search_complete"], true);
    assert_eq!(
        why["minimal_failing_changes"],
        json!([{"file": "calc.sh", "line": 15, "added": ["sum=$((a - b))"]}])
    );
    assert_eq!(why["commit_without_minimal"], "good");
    assert_eq!(why["undo_on_bad"], "good");
    assert_eq!(why["undo_how"], "whole commit");
    let patch_path = fixture.path("report.undo.patch");
    assert_eq!(why["undo_patch"]["file"], "report.undo.patch");
    assert_eq!(why["undo_patch"]["applies_to"], "v2");
    let patch = std::fs::read_to_string(&patch_path).expect("the undo patch beside the report");
    assert!(patch.contains("-sum=$((a - b))"), "{patch}");
    assert!(patch.contains("+sum=$((a + b))"), "{patch}");
    // The whole commit: the file it added is removed again.
    assert!(patch.contains("deleted file mode"), "{patch}");
    assert!(patch.contains("CHANGES.txt"), "{patch}");
    // The patch is the undo that was tested: it applies to v2 as it is.
    let scratch = fixture.path("apply-check");
    git(
        &fixture.repo,
        NOW,
        &["worktree", "add", "-q", "--detach", path_arg(&scratch), "v2"],
    );
    git(&scratch, NOW, &["apply", "--check", path_arg(&patch_path)]);
    git(
        &fixture.repo,
        NOW,
        &["worktree", "remove", "--force", path_arg(&scratch)],
    );

    // The protocol was frozen before the first test.
    assert_eq!(
        protocol["rule"],
        "exit 0 good, 125 cannot test, any other code bad"
    );
    assert_eq!(protocol["good"], fixture.commits[0].as_str());
    assert_eq!(protocol["bad"], fixture.commits[12].as_str());
    assert_eq!(protocol["oracle_sha256"], report["oracle"]["sha256"]);
    assert_eq!(protocol["also_hashed"], json!({}));
    assert_eq!(protocol["flaky"], Value::Null);
    assert_eq!(report["flaky"], Value::Null);
    assert!(protocol["frozen_at"].as_str().unwrap() <= report["probes"][0]["at"].as_str().unwrap());
    assert!(
        ran.stdout.contains("Protocol frozen before any test"),
        "{}",
        ran.stdout
    );

    // The verdict card: five rungs, all of them hold.
    assert_eq!(
        rungs(&report),
        [
            ("LEAD", true),
            ("BOUNDARY", true),
            ("CONFIRMED", true),
            ("ISOLATED", true),
            ("REVERSED", true),
        ]
        .map(|(name, holds)| (name.to_string(), holds))
    );
    assert_eq!(
        rung(&report, "ISOLATED")["statement"],
        "1 of its 4 changes alone causes it, and the rest passes without it"
    );
    assert_eq!(
        rung(&report, "REVERSED")["statement"],
        "undoing the whole commit on v2 makes the test pass again"
    );
    assert_eq!(rung(&report, "BOUNDARY")["proof"], "runs 3,5");
    assert!(ran.stdout.contains("5 of 5 rungs hold"), "{}", ran.stdout);
    assert!(
        ran.stdout
            .contains("In plain git log order this is commit 8 of 12."),
        "{}",
        ran.stdout
    );
    assert!(ran.stdout.contains("Not claimed: why the authors"), "{}", ran.stdout);

    // The probe log: numbered, chained, one memory per run. The bisect
    // phase holds the runs `git bisect run` made through the hidden
    // subcommand, on commits no earlier step had tested.
    let probes = report["probes"].as_array().unwrap();
    let phases = phases(&report);
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
    assert!(
        ran.stdout.contains("Every run is a memory in the store"),
        "{}",
        ran.stdout
    );
    let storage = vestige_mcp::strata_memory::open(&fixture.store).expect("open the store");
    for (index, probe) in probes.iter().enumerate() {
        assert_eq!(probe["n"], index + 1);
        assert!(probe.get("runs").is_none(), "one run, one verdict");
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

    // The user's checkout is exactly as it was, down to the unsaved edit,
    // and the worktree is gone.
    assert_eq!(checkout_state(&fixture.repo), before);
    assert_eq!(
        std::fs::read_to_string(fixture.repo.join("notes.txt")).unwrap(),
        "notes\nmine, unsaved\n"
    );
    assert!(fixture.repo.join("untracked.txt").exists());
    fixture.assert_no_worktree_left();

    // A second walk says what the first one said: proving added nothing
    // upstream of the failure.
    assert_eq!(walk_printed(&fixture.store, &fixture.failure), walk);

    // The report re-verifies, a second run will not overwrite it or its
    // patch, and a copy with one verdict changed does not pass.
    let checked = fixture.check(&report_path);
    assert!(checked.ok, "{}", checked.text());
    for line in [
        "hash chain intact",
        ": recorded verdict bad",
        ": recorded verdict good",
        ": hash matches",
        "LEAD      yes  the walk reached it over recorded links (lead ",
        "REVERSED  yes  undoing the whole commit on v2 makes the test pass again",
        "minimal failing change: calc.sh line 15",
        "the commit without it: recorded verdict good",
        "undo on v2 (whole commit): recorded verdict good",
        "undo patch report.undo.patch: sha256 matches the report",
    ] {
        assert!(checked.stdout.contains(line), "{line}: {}", checked.stdout);
    }
    let again = fixture.prove(&report_path, &[]);
    assert_eq!(again.code, Some(1), "{}", again.text());
    assert!(
        again.stdout.contains("refusing to overwrite"),
        "{}",
        again.text()
    );
    let beside_patch = fixture.prove(&fixture.path("report"), &[]);
    assert!(!beside_patch.ok, "{}", beside_patch.text());
    assert!(
        beside_patch.stdout.contains("report.undo.patch"),
        "an existing undo patch is not overwritten either: {}",
        beside_patch.text()
    );

    let mut tampered = report.clone();
    let changed = tampered["probes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|probe| probe["commit"] == breaking.as_str())
        .expect("a run on the breaking commit");
    changed["verdict"] = json!("good");
    let tampered_path = fixture.path("tampered.json");
    std::fs::write(
        &tampered_path,
        serde_json::to_string_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let rejected = fixture.check(&tampered_path);
    assert_eq!(rejected.code, Some(1), "{}", rejected.text());
    assert!(
        rejected.stdout.contains("hash does not match"),
        "{}",
        rejected.text()
    );
    // A card rung flipped to yes, outside the hash chain, is caught by the
    // runs it claims.
    let mut flattered = report.clone();
    flattered["why"]["undo_on_bad"] = json!("bad");
    std::fs::write(
        &tampered_path,
        serde_json::to_string_pretty(&flattered).unwrap(),
    )
    .unwrap();
    assert_eq!(fixture.check(&tampered_path).code, Some(1));
    // The undo patch edited after the fact.
    std::fs::write(&patch_path, format!("{patch}\n# and one more line\n")).unwrap();
    let repatched = fixture.check(&report_path);
    assert_eq!(repatched.code, Some(1), "{}", repatched.text());
    assert!(
        repatched.stdout.contains("DOES NOT MATCH the report"),
        "{}",
        repatched.text()
    );
}

#[test]
fn walk_verify_prove_undoes_only_the_found_lines_when_the_whole_commit_conflicts() {
    let fixture = fixture(&Plan {
        later_commit_conflicts: true,
        ..Plan::default()
    });
    let report_path = fixture.path("report.json");
    let ran = fixture.prove(&report_path, &[]);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);
    assert_eq!(
        report["first_bad_commit"],
        fixture.commits[BREAKING].as_str()
    );
    assert!(
        ran.stdout.contains(
            "the whole commit no longer undoes cleanly on v2 (later commits touched other parts of it), so only the lines found are undone"
        ),
        "{}",
        ran.stdout
    );
    let why = &report["why"];
    assert_eq!(why["undo_how"], "found lines only");
    assert_eq!(why["undo_on_bad"], "good");
    assert_eq!(
        why["minimal_failing_changes"],
        json!([{"file": "calc.sh", "line": 15, "added": ["sum=$((a - b))"]}])
    );
    // The patch undoes that one hunk and nothing else of the commit.
    let patch = std::fs::read_to_string(fixture.path("report.undo.patch")).unwrap();
    assert!(patch.contains("+sum=$((a + b))"), "{patch}");
    assert!(!patch.contains("CHANGES.txt"), "{patch}");
    assert!(!patch.contains("README.md"), "{patch}");
    assert!(!patch.contains("adds two"), "the header hunk stays: {patch}");
    assert_eq!(
        rung(&report, "REVERSED")["statement"],
        "undoing just those lines on v2 makes the test pass again"
    );
    assert!(ran.stdout.contains("5 of 5 rungs hold"), "{}", ran.stdout);
    assert!(
        ran.stdout
            .contains("Undo just that change on v2 and the bug is gone."),
        "{}",
        ran.stdout
    );
    let checked = fixture.check(&report_path);
    assert!(checked.ok, "{}", checked.text());
    assert!(
        checked
            .stdout
            .contains("undo on v2 (found lines only): recorded verdict good"),
        "{}",
        checked.stdout
    );
    fixture.assert_no_worktree_left();
}

#[test]
fn walk_verify_prove_lets_bisect_decide_when_the_walk_missed_the_commit() {
    // Nobody ever recorded the breaking commit: it has no memory.
    let fixture = fixture(&Plan {
        remembered: (0..=12).filter(|index| *index != BREAKING).collect(),
        ..Plan::default()
    });
    let breaking = &fixture.commits[BREAKING];
    let report_path = fixture.path("report.json");
    let ran = fixture.prove(&report_path, &["--no-why"]);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);

    let candidates = report["walk"]["candidates"].as_array().unwrap();
    assert!(
        candidates
            .iter()
            .all(|lead| lead["commit"] != breaking.as_str()),
        "{candidates:?}"
    );
    // One link away the earliest failing lead's parent fails too, so the
    // leads two links away are added; they do not hold the boundary either.
    assert!(
        ran.stdout.contains("leads within 1 link of the report: 2"),
        "{}",
        ran.stdout
    );
    assert!(
        ran.stdout.contains("leads within 2 links of the report: 8"),
        "{}",
        ran.stdout
    );
    assert!(
        ran.stdout
            .contains("The first bad commit is not among the leads; the full git bisect takes over."),
        "{}",
        ran.stdout
    );
    assert!(!ran.stdout.contains("fails and its parent passes"), "{}", ran.stdout);

    // git bisect still names it; the card says the walk did not.
    assert_eq!(report["first_bad_commit"], breaking.as_str());
    assert_eq!(report["walk_lead_rank"], Value::Null);
    assert!(
        ran.stdout.contains(
            "The walk did not reach this commit. The tested result stands; the walk missed it."
        ),
        "{}",
        ran.stdout
    );
    assert!(!ran.stdout.contains("Found in "), "{}", ran.stdout);
    assert_eq!(
        rungs(&report),
        [("LEAD", false), ("BOUNDARY", true), ("CONFIRMED", true)]
            .map(|(name, holds)| (name.to_string(), holds))
    );
    assert_eq!(rung(&report, "LEAD")["proof"], "the walk did not reach it");
    assert!(ran.stdout.contains("2 of 3 rungs hold"), "{}", ran.stdout);
    assert_eq!(probe_on(&report, breaking)["verdict"], "bad");
    assert_eq!(
        probe_on(&report, &fixture.commits[BREAKING - 1])["verdict"],
        "good"
    );
    assert_eq!(report["why"], Value::Null);
    let result = report["result_memory"].as_str().expect("a result memory");
    let storage = vestige_mcp::strata_memory::open(&fixture.store).expect("open the store");
    let node = storage.get_node(result).unwrap().expect("the result");
    assert!(
        node.content.contains("had not reached this commit"),
        "{}",
        node.content
    );
    assert!(node.content.ends_with("Lines not searched."), "{}", node.content);
    drop(storage);

    let checked = fixture.check(&report_path);
    assert!(checked.ok, "{}", checked.text());
    assert!(checked.stdout.contains("LEAD      no "), "{}", checked.stdout);
    fixture.assert_no_worktree_left();
}

#[test]
fn walk_verify_prove_widens_to_farther_leads_until_one_holds_the_boundary() {
    // The report names notes.txt. The commits that touched notes are one
    // link away; the breaking commit, which never touched it, is two.
    let fixture = fixture(&Plan {
        failure_names: "notes.txt",
        ..Plan::default()
    });
    let breaking = &fixture.commits[BREAKING];
    let report_path = fixture.path("report.json");
    let ran = fixture.prove(&report_path, &["--no-why"]);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);

    let lead = report["walk"]["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|lead| lead["commit"] == breaking.as_str())
        .expect("the breaking commit is a lead");
    assert_eq!(lead["depth"], 2);
    let tier = |line: &str| {
        ran.stdout
            .find(line)
            .unwrap_or_else(|| panic!("no line {line:?} in {}", ran.stdout))
    };
    let first = tier("leads within 1 link of the report: ");
    let second = tier("leads within 2 links of the report: ");
    let boundary = tier(&format!(
        "{} fails and its parent passes.",
        fixture.short(BREAKING)
    ));
    assert!(first < second && second < boundary, "{}", ran.stdout);

    assert_eq!(report["first_bad_commit"], breaking.as_str());
    assert_eq!(report["walk_lead_rank"], lead["rank"]);
    assert_eq!(
        rungs(&report),
        [("LEAD", true), ("BOUNDARY", true), ("CONFIRMED", true)]
            .map(|(name, holds)| (name.to_string(), holds))
    );
    // Runs made for the first tier are reused in the second, not repeated.
    let probes = report["probes"].as_array().unwrap();
    let mut seen: Vec<&str> = probes
        .iter()
        .map(|probe| probe["commit"].as_str().unwrap())
        .collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), probes.len(), "a commit was tested twice");
    assert_eq!(
        report["found_in_runs"].as_u64().unwrap() as usize,
        probes
            .iter()
            .filter(|probe| probe["phase"] == "candidate" || probe["phase"] == "parent")
            .count()
    );
    assert!(fixture.check(&report_path).ok);
}

#[test]
fn walk_verify_prove_caps_the_runs_on_leads_not_the_leads() {
    let fixture = fixture(&Plan::default());
    let breaking = &fixture.commits[BREAKING];

    // Three runs are enough for the three closest leads and the parent,
    // though seven more leads are on record: none of them is cut off, and
    // the breaking commit is not one of the newest three.
    let enough = fixture.path("enough.json");
    let ran = fixture.prove(&enough, &["--no-why", "--max-candidates", "3"]);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&enough);
    assert_eq!(report["protocol"]["max_candidates"], 3);
    assert_eq!(report["walk"]["kept"], 10);
    assert_eq!(report["found_in_runs"], 3);
    assert!(
        ran.stdout.contains("fails and its parent passes."),
        "{}",
        ran.stdout
    );
    assert!(!ran.stdout.contains("the cap of"), "{}", ran.stdout);
    assert_eq!(report["first_bad_commit"], breaking.as_str());

    // With two, the cap is reached before the parent is tested. It says
    // so, and git bisect names the commit all the same.
    let capped = fixture.path("capped.json");
    let ran = fixture.prove(&capped, &["--no-why", "--max-candidates", "2"]);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&capped);
    assert!(
        ran.stdout
            .contains("the cap of 2 test runs on leads is reached; the full git bisect takes over"),
        "{}",
        ran.stdout
    );
    assert!(!ran.stdout.contains("fails and its parent passes"), "{}", ran.stdout);
    assert!(!ran.stdout.contains("Found in "), "{}", ran.stdout);
    assert_eq!(report["found_in_runs"], 2);
    let phases = phases(&report);
    assert_eq!(
        phases
            .iter()
            .filter(|phase| **phase == "candidate")
            .count(),
        2
    );
    assert!(!phases.contains(&"parent"), "{phases:?}");
    assert_eq!(report["first_bad_commit"], breaking.as_str());
    assert_eq!(rungs(&report)[1], ("BOUNDARY".to_string(), true));

    // None at all: every lead is left to git bisect.
    let none = fixture.path("none.json");
    let ran = fixture.prove(&none, &["--no-why", "--max-candidates", "0"]);
    assert!(ran.ok, "{}", ran.text());
    assert!(
        ran.stdout
            .contains("the cap of 0 test runs on leads is reached"),
        "{}",
        ran.stdout
    );
    assert_eq!(read_json(&none)["first_bad_commit"], breaking.as_str());
    for report in [&enough, &capped, &none] {
        assert!(fixture.check(report).ok);
    }
    fixture.assert_no_worktree_left();
}

/// The variables `git bisect run` exports to its command.
const GIT_LOCATION_VARS: [&str; 8] = [
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
];

fn write_script(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn walk_verify_prove_keeps_git_variables_and_spaces_out_of_the_way() {
    // Every path has a space in it: the repository, the store, the report,
    // the test script, the scratch directory and the binary itself.
    let fixture = fixture(&Plan {
        dir_prefix: "walk verify ",
        ..Plan::default()
    });
    let breaking = &fixture.commits[BREAKING];
    let tmp = fixture.path("tmp dir");
    std::fs::create_dir(&tmp).unwrap();
    let bin_dir = fixture.path("bin dir");
    std::fs::create_dir(&bin_dir).unwrap();
    let binary = bin_dir.join("vestige");
    std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_vestige"), &binary).unwrap();
    let report_path = fixture.path("the report.json");

    // A test that itself uses git, and refuses to give a verdict when any
    // variable that points git somewhere is set. Under a plain
    // `git bisect run` they are.
    let script = fixture.path("the test.sh");
    let seen = fixture.path("phases seen.txt");
    write_script(
        &script,
        &format!(
            r#"for name in {vars}; do
  eval "value=\${{$name-}}"
  if [ -n "$value" ]; then echo "$name leaked into the test"; exit 125; fi
done
git rev-parse --git-dir > /dev/null || {{ echo "git does not work here"; exit 125; }}
git rev-parse HEAD >> '{seen}'
out=$(sh calc.sh 2 3); echo "2+3=$out"; test "$out" = 5"#,
            vars = GIT_LOCATION_VARS.join(" "),
            seen = seen.display(),
        ),
    );

    // A repository the environment points at. `git bisect run` sets these
    // for its command; a hook or an alias sets them for whatever it runs.
    let decoy = fixture.path("decoy repo");
    std::fs::create_dir(&decoy).unwrap();
    git(&decoy, NOW, &["init", "-q", "-b", "main"]);
    std::fs::write(decoy.join("precious.txt"), "committed\n").unwrap();
    git(&decoy, NOW, &["add", "-A"]);
    git(&decoy, NOW, &["commit", "-q", "-m", "Precious"]);
    std::fs::write(decoy.join("precious.txt"), "committed\nand unsaved work\n").unwrap();
    std::fs::write(decoy.join("untracked.txt"), "untracked\n").unwrap();
    let decoy_before = checkout_state(&decoy);
    let decoy_git = decoy.join(".git");

    let mut args = fixture.prove_args(&report_path);
    args.extend(["--oracle", path_arg(&script)].map(str::to_string));
    let mut command = vestige_command(&binary, &fixture.store);
    command
        .args(&args)
        .env("TMPDIR", &tmp)
        .env("GIT_DIR", &decoy_git)
        .env("GIT_WORK_TREE", &decoy)
        .env("GIT_INDEX_FILE", decoy_git.join("index"))
        .env("GIT_COMMON_DIR", &decoy_git)
        .env("GIT_PREFIX", "sub/")
        .env("GIT_OBJECT_DIRECTORY", decoy_git.join("objects"))
        .env("GIT_ALTERNATE_OBJECT_DIRECTORIES", decoy_git.join("objects"))
        .env("GIT_NAMESPACE", "decoy");
    let ran = ran(command);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);

    assert_eq!(report["first_bad_commit"], breaking.as_str(), "{}", ran.text());
    assert_eq!(report["repo"], path_arg(&fixture.repo));
    assert_eq!(report["oracle"]["file"], "the test.sh");
    assert!(ran.stdout.contains("5 of 5 rungs hold"), "{}", ran.stdout);
    // No run in any phase saw one of the variables: none is "cannot test".
    let probes = report["probes"].as_array().unwrap();
    for probe in probes {
        assert_ne!(probe["verdict"], "skip", "{probe}");
    }
    let phases = phases(&report);
    for phase in ["baseline", "candidate", "parent", "bisect", "lines", "undo"] {
        assert!(phases.contains(&phase), "no {phase} run in {phases:?}");
    }
    // The test's own git acted on the worktree under test, in every phase:
    // it saw the commits that were tested, never the decoy's.
    let seen = std::fs::read_to_string(&seen).unwrap();
    let decoy_head = git(&decoy, NOW, &["rev-parse", "HEAD"]);
    assert_eq!(seen.lines().count(), probes.len());
    assert!(!seen.contains(&decoy_head), "{seen}");
    assert!(seen.contains(breaking.as_str()), "{seen}");

    // The repository the environment pointed at was never acted on.
    assert_eq!(checkout_state(&decoy), decoy_before);
    assert_eq!(
        std::fs::read_to_string(decoy.join("precious.txt")).unwrap(),
        "committed\nand unsaved work\n"
    );
    assert_eq!(git(&decoy, NOW, &["worktree", "list"]).lines().count(), 1);
    assert!(!decoy_git.join("BISECT_LOG").exists());

    // The scratch directory, spaces and all, is gone.
    let left: Vec<_> = std::fs::read_dir(&tmp)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert!(left.is_empty(), "left in the scratch directory: {left:?}");
    fixture.assert_no_worktree_left();
    assert!(fixture.path("the report.undo.patch").exists());
    let mut check = vestige_command(&binary, &fixture.store);
    check.args(["prove", "--check", path_arg(&report_path)]);
    let checked = ran_ok(check);
    assert!(
        checked
            .stdout
            .contains("undo patch the report.undo.patch: sha256 matches the report"),
        "{}",
        checked.stdout
    );
}

fn ran_ok(command: Command) -> Ran {
    let ran = ran(command);
    assert!(ran.ok, "{}", ran.text());
    ran
}

#[test]
fn walk_verify_prove_reports_a_bisect_that_aborts_instead_of_trusting_it() {
    // The hidden subcommand, failing on its own, exits where git bisect
    // aborts (128 and above), never with the code of a bad commit.
    let dir = TempDir::new().unwrap();
    for mode in ["probe", "sim", "no-such-mode"] {
        let ran = vestige(
            dir.path(),
            &["_prove", mode, path_arg(&dir.path().join("no cfg.json"))],
        );
        assert_eq!(ran.code, Some(255), "{mode}: {}", ran.text());
        assert!(
            ran.stderr.contains(&format!("vestige _prove {mode} failed: ")),
            "{}",
            ran.text()
        );
    }
    let garbled = dir.path().join("cfg.json");
    std::fs::write(&garbled, "{\"repo\": ").unwrap();
    let ran = vestige(dir.path(), &["_prove", "probe", path_arg(&garbled)]);
    assert_eq!(ran.code, Some(255), "{}", ran.text());

    // The whole flow, with the subcommand dying in the middle of a bisect:
    // the test kills the process that runs it when that process is the
    // hidden subcommand.
    let fixture = fixture(&Plan::default());
    let script = fixture.path("test.sh");
    write_script(
        &script,
        r#"case "$(ps -o args= -p $PPID)" in
  *"_prove probe"*) kill -9 $PPID; sleep 1 ;;
esac
out=$(sh calc.sh 2 3); echo "2+3=$out"; test "$out" = 5"#,
    );
    let report_path = fixture.path("report.json");
    let mut args = fixture.prove_args(&report_path);
    args.extend(["--oracle", path_arg(&script)].map(str::to_string));
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let ran = vestige(&fixture.store, &args);

    // No first bad commit is named, the run says that git bisect did not
    // name one and shows what git said, and it does not exit 0.
    assert_eq!(ran.code, Some(1), "{}", ran.text());
    assert!(
        ran.stdout
            .contains("git bisect did not name a first bad commit. Its output:"),
        "{}",
        ran.stdout
    );
    assert!(
        ran.stdout.contains("bisect run failed"),
        "git's own reason is shown: {}",
        ran.stdout
    );
    assert!(!ran.stdout.contains("is the first bad commit"), "{}", ran.stdout);
    assert!(!ran.stdout.contains("Verdict"), "{}", ran.stdout);
    // What was tested before the abort is still on record and checks out;
    // nothing is claimed beyond it.
    let report = read_json(&report_path);
    assert_eq!(report["first_bad_commit"], Value::Null);
    assert_eq!(report["verdict_card"], json!([]));
    assert_eq!(report["why"], Value::Null);
    assert_eq!(report["result_memory"], Value::Null);
    assert_eq!(report["plain_bisect_runs_replayed"], Value::Null);
    assert!(!phases(&report).contains(&"bisect"));
    let checked = fixture.check(&report_path);
    assert!(checked.ok, "{}", checked.text());
    assert!(
        checked
            .stdout
            .contains("this report names no first bad commit"),
        "{}",
        checked.stdout
    );
    assert!(!fixture.path("report.undo.patch").exists());
    fixture.assert_no_worktree_left();
}

#[test]
fn walk_verify_prove_counts_a_hanging_test_as_cannot_test() {
    let fixture = fixture(&Plan::default());
    let breaking = &fixture.commits[BREAKING];
    let hanging = &fixture.commits[3];
    let child_pid = fixture.path("child.pid");
    // On one of the leads the test starts a child of its own and hangs.
    let script = fixture.path("test.sh");
    write_script(
        &script,
        &format!(
            r#"if [ "$(git rev-parse HEAD)" = "{hanging}" ]; then
  sleep 300 &
  echo $! > '{pid}'
  echo "about to hang"
  sleep 300
fi
out=$(sh calc.sh 2 3); echo "2+3=$out"; test "$out" = 5"#,
            pid = child_pid.display(),
        ),
    );
    let report_path = fixture.path("report.json");
    let mut args = fixture.prove_args(&report_path);
    args.extend(["--oracle", path_arg(&script), "--timeout", "2", "--no-why"].map(str::to_string));
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let started = Instant::now();
    let ran = vestige(&fixture.store, &args);
    assert!(ran.ok, "{}", ran.text());
    assert!(
        started.elapsed() < Duration::from_secs(120),
        "the hang was not cut short: {:?}",
        started.elapsed()
    );
    let report = read_json(&report_path);

    // The run on that commit was killed and gave no verdict; its result
    // line says why.
    let hung = probe_on(&report, hanging);
    assert_eq!(hung["verdict"], "skip");
    assert_eq!(hung["exit"], 125);
    assert_eq!(
        hung["oracle_said"],
        "timed out after 2 seconds, counted as cannot test"
    );
    assert!(
        ran.stdout.contains(&format!(
            "[tested] {} SKIP timed out after 2 seconds, counted as cannot test",
            fixture.short(3)
        )),
        "{}",
        ran.stdout
    );
    // The process it had started is gone with it.
    let pid = std::fs::read_to_string(&child_pid).unwrap();
    let alive = Command::new("kill")
        .args(["-0", pid.trim()])
        .output()
        .expect("spawn kill");
    assert!(!alive.status.success(), "the test's child {pid} is still running");
    // It left the line-up, and the commit was found without it.
    assert_eq!(report["first_bad_commit"], breaking.as_str());
    assert!(
        ran.stdout.contains("fails and its parent passes."),
        "{}",
        ran.stdout
    );
    assert_eq!(
        rungs(&report),
        [("LEAD", true), ("BOUNDARY", true), ("CONFIRMED", true)]
            .map(|(name, holds)| (name.to_string(), holds))
    );
    assert!(fixture.check(&report_path).ok);
    fixture.assert_no_worktree_left();
}

#[test]
fn walk_verify_prove_flaky_names_a_commit_that_fails_one_run_in_three() {
    let fixture = fixture(&Plan::default());
    let breaking = &fixture.commits[BREAKING];
    // Deterministic flakiness: once calc.sh subtracts, the test fails on
    // every third run, counted in a file outside the repository.
    let counter = fixture.path("count");
    let script = fixture.path("flaky.sh");
    write_script(
        &script,
        &format!(
            r#"out=$(sh calc.sh 2 3)
if [ "$out" = 5 ]; then echo "2+3=$out"; exit 0; fi
n=$(cat '{counter}' 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > '{counter}'
if [ $((n % 3)) -eq 0 ]; then echo "2+3=$out on bad run $n"; exit 1; fi
echo "2+3=$out, but run $n got lucky""#,
            counter = counter.display(),
        ),
    );
    let fixture_file = fixture.path("inputs.txt");
    std::fs::write(&fixture_file, "2 3\n").unwrap();
    let report_path = fixture.path("report.json");
    let mut args = fixture.prove_args(&report_path);
    args.extend(
        [
            "--oracle",
            path_arg(&script),
            "--flaky",
            "--strength-runs",
            "30",
            "--also-hash",
            path_arg(&fixture_file),
        ]
        .map(str::to_string),
    );
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let ran = vestige(&fixture.store, &args);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);

    // The two ends, in batches of ten until they differ beyond chance:
    // 3 of 10 and 6 of 20 failures are not enough, 10 of 30 is.
    let probes = report["probes"].as_array().unwrap();
    assert_eq!(probes[0]["commit"], fixture.commits[0].as_str());
    assert_eq!(probes[0]["phase"], "baseline");
    assert_eq!(probes[0]["verdict"], "good");
    assert_eq!(probes[0]["oracle_said"], "failed 0 of 30 runs");
    assert_eq!((&probes[0]["runs"], &probes[0]["fails"]), (&json!(30), &json!(0)));
    assert_eq!(probes[1]["commit"], fixture.commits[12].as_str());
    assert_eq!(probes[1]["verdict"], "bad");
    assert_eq!(probes[1]["oracle_said"], "failed 10 of 30 runs");
    assert_eq!((&probes[1]["runs"], &probes[1]["fails"]), (&json!(30), &json!(10)));
    assert_eq!(probes[0]["at"], probes[1]["at"]);
    assert!(
        ran.stdout
            .contains("The two ends differ beyond chance (p = 0.0004). Each commit is now tested repeatedly until the evidence is decisive at 0.01."),
        "{}",
        ran.stdout
    );
    let stats = &report["flaky"]["stats"];
    assert_eq!(stats["p0"], json!(0.5 / 31.0));
    assert_eq!(stats["p1"], json!(10.5 / 31.0));
    assert_eq!((&stats["alpha"], &stats["beta"]), (&json!(0.01), &json!(0.01)));
    assert_eq!(stats["max_runs"], 80);

    // Every later verdict is a sequential test: a commit that never fails
    // is good after 12 runs, one that fails every third run is bad within 9.
    for probe in &probes[2..] {
        let (runs, fails) = (probe["runs"].as_u64().unwrap(), probe["fails"].as_u64().unwrap());
        match (probe["phase"].as_str().unwrap(), probe["verdict"].as_str().unwrap()) {
            ("strength", _) => assert_eq!(runs, 30, "{probe}"),
            (_, "good") => assert_eq!((runs, fails), (12, 0), "{probe}"),
            (_, "bad") => assert!((2..=9).contains(&runs) && fails >= 2, "{probe}"),
            (_, other) => panic!("an undecided verdict {other}: {probe}"),
        }
        assert!(
            probe["oracle_said"]
                .as_str()
                .unwrap()
                .starts_with(&format!("failed {fails} of {runs} runs")),
            "{probe}"
        );
    }

    // The same commit, the same change, and one more rung.
    assert_eq!(report["first_bad_commit"], breaking.as_str(), "{}", ran.text());
    assert_eq!(report["found_in_runs"], 3);
    assert!(
        ran.stdout
            .contains("Found in 3 commits tested on the walk's 10 leads."),
        "{}",
        ran.stdout
    );
    assert_eq!(
        report["why"]["minimal_failing_changes"],
        json!([{"file": "calc.sh", "line": 15, "added": ["sum=$((a - b))"]}])
    );
    assert_eq!(
        rungs(&report),
        [
            ("LEAD", true),
            ("BOUNDARY", true),
            ("CONFIRMED", true),
            ("ISOLATED", true),
            ("REVERSED", true),
            ("REPEATED", true),
        ]
        .map(|(name, holds)| (name.to_string(), holds))
    );
    assert!(ran.stdout.contains("6 of 6 rungs hold"), "{}", ran.stdout);

    // How strong: thirty runs with the commit, thirty without.
    let strength = &report["flaky"]["strength"];
    let (with, without) = (&strength["with"], &strength["without"]);
    assert_eq!((&with["fails"], &with["runs"]), (&json!(10), &json!(30)));
    assert_eq!((&without["fails"], &without["runs"]), (&json!(0), &json!(30)));
    let p = strength["fisher_p"].as_f64().unwrap();
    assert!((p - 0.0003985065657050695).abs() < 1e-15, "{p}");
    let times = strength["at_least_times"].as_f64().unwrap();
    assert!((times - 1.4941164966027056).abs() < 1e-9, "{times}");
    let repeated = rung(&report, "REPEATED");
    assert_eq!(
        repeated["statement"],
        "it fails 10 of 30 times with the commit and 0 of 30 without (p = 0.0004)"
    );
    assert_eq!(
        repeated["proof"],
        format!("runs {},{}", with["probe"], without["probe"])
    );
    assert!(
        ran.stdout
            .contains("With this commit the test fails 10 of 30 times. Without it, 0 of 30."),
        "{}",
        ran.stdout
    );
    assert!(
        ran.stdout
            .contains("Chance of that split if the commit made no difference: 0.0004. Failure is at least 1.5 times more likely with it"),
        "{}",
        ran.stdout
    );
    let total: u64 = probes
        .iter()
        .map(|probe| probe["runs"].as_u64().unwrap())
        .sum();
    assert_eq!(report["flaky"]["test_runs_total"], total);
    assert!(
        ran.stdout.contains(&format!(
            "{} results recorded from {total} test runs",
            probes.len()
        )),
        "{}",
        ran.stdout
    );

    // The settings and the extra file are part of the frozen protocol.
    let protocol = &report["protocol"];
    assert_eq!(
        protocol["flaky"],
        json!({"alpha": 0.01, "max_runs_per_commit": 80, "baseline_max": 100, "strength_runs": 30})
    );
    let hashed = protocol["also_hashed"].as_object().unwrap();
    assert_eq!(hashed.len(), 1);
    assert_eq!(
        hashed["inputs.txt"],
        // sha256 of "2 3\n"
        "0e5b1c7a4a2cd5b3d1ea8b79b5a1e6a2e06b4d0c0c0f6a8d0a5b0f7f4d1a3c11"
    );

    let checked = fixture.check(&report_path);
    assert!(checked.ok, "{}", checked.text());
    for line in [
        "REPEATED  yes  it fails 10 of 30 times with the commit and 0 of 30 without (p = 0.0004)",
        "also frozen: inputs.txt sha256 ",
    ] {
        assert!(checked.stdout.contains(line), "{line}: {}", checked.stdout);
    }
    // The strength claim is held against the recorded counts.
    let mut tampered = report.clone();
    tampered["verdict_card"]
        .as_array_mut()
        .unwrap()
        .push(json!({"rung": "REPEATED", "holds": true, "statement": "s", "proof": "p"}));
    tampered["probes"].as_array_mut().unwrap().truncate(probes.len() - 2);
    tampered["chain_head"] = tampered["probes"][probes.len() - 3]["hash"].clone();
    let tampered_path = fixture.path("tampered.json");
    std::fs::write(
        &tampered_path,
        serde_json::to_string_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let rejected = fixture.check(&tampered_path);
    assert_eq!(rejected.code, Some(1), "{}", rejected.text());
    assert!(
        rejected
            .stdout
            .contains("the card says REPEATED holds, but the recorded runs do not back it."),
        "{}",
        rejected.stdout
    );
    fixture.assert_no_worktree_left();
}

#[test]
fn walk_verify_prove_flaky_stops_when_the_ends_do_not_differ() {
    let fixture = fixture(&Plan::default());
    let report_path = fixture.path("report.json");
    // The test passes on both ends, every time.
    let mut args = fixture.prove_args(&report_path);
    args.extend(["--test", "true", "--flaky", "--baseline-max", "20"].map(str::to_string));
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let ran = vestige(&fixture.store, &args);
    assert_eq!(ran.code, Some(1), "{}", ran.text());
    assert!(
        ran.stdout.contains(
            "After 20 runs on each end the failure rates are not clearly different (p = 1), so nothing can be decided with this test."
        ),
        "{}",
        ran.stdout
    );
    assert!(!report_path.exists());
    fixture.assert_no_worktree_left();

    // A test that can never be run does not loop for ever.
    let mut args = fixture.prove_args(&report_path);
    args.extend(["--test", "exit 125", "--flaky"].map(str::to_string));
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let ran = vestige(&fixture.store, &args);
    assert_eq!(ran.code, Some(1), "{}", ran.text());
    assert!(
        ran.stdout.contains("The test could not be run on v2"),
        "{}",
        ran.stdout
    );
    assert!(!report_path.exists());
    fixture.assert_no_worktree_left();
}

#[test]
fn walk_verify_prove_refuses_what_it_cannot_decide_and_leaves_nothing_behind() {
    let fixture = fixture(&Plan::default());
    let report_path = fixture.path("report.json");
    let memories_before = {
        let storage = vestige_mcp::strata_memory::open(&fixture.store).expect("open the store");
        storage.get_stats().expect("stats").total_nodes
    };
    let refused = |extra: &[&str], replace: &[(&str, &str)], message: &str| {
        let mut args = fixture.prove_args(&report_path);
        for (flag, value) in replace {
            let at = args
                .iter()
                .position(|arg| arg == flag)
                .unwrap_or_else(|| panic!("no {flag}"));
            args[at + 1] = (*value).to_string();
        }
        args.extend(extra.iter().map(|arg| (*arg).to_string()));
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let ran = vestige(&fixture.store, &args);
        assert_eq!(ran.code, Some(1), "{message}: {}", ran.text());
        assert!(ran.stdout.contains(message), "{message}: {}", ran.text());
        assert!(!report_path.exists(), "{message}: a report was written");
        fixture.assert_no_worktree_left();
    };
    let test = ["--test", TEST];

    // The two ends the wrong way round, the same, or not there.
    refused(
        &test,
        &[("--good", "v2"), ("--bad", "v1")],
        "there is no commit in v2..v1",
    );
    refused(&test, &[("--bad", "v1")], "there is no commit in v1..v1");
    refused(&test, &[("--good", "v0")], "cannot resolve --good/--bad");
    refused(&test, &[("--bad", "--all")], "cannot resolve --good/--bad");
    refused(
        &test,
        &[("--repo", path_arg(fixture.dir.path()))],
        "--repo is not a git repository",
    );
    refused(
        &test,
        &[("--repo", path_arg(&fixture.path("nowhere")))],
        "--repo is not a directory",
    );
    // No test, or one that cannot be run.
    refused(&[], &[], "give --test 'one command' or --oracle script");
    refused(&["--test", "  "], &[], "--test is empty");
    refused(
        &["--oracle", path_arg(&fixture.path("no-such-test.sh"))],
        &[],
        "there is no test script at",
    );
    let not_executable = fixture.path("plain.sh");
    std::fs::write(&not_executable, "#!/bin/sh\nexit 0\n").unwrap();
    refused(
        &["--oracle", path_arg(&not_executable)],
        &[],
        "the test script is not executable",
    );
    // Settings that mean nothing.
    refused(
        &test,
        &[("--reported-at", "yesterday")],
        "--reported-at wants RFC 3339",
    );
    refused(&test, &[("--slug", "a,b")], "--slug becomes one tag");
    refused(
        &["--test", TEST, "--flaky", "--alpha", "0.7"],
        &[],
        "--alpha is the accepted chance of a wrong verdict",
    );
    refused(
        &["--test", TEST, "--flaky", "--strength-runs", "0"],
        &[],
        "--strength-runs must be from 1 to 500",
    );
    refused(
        &["--test", TEST, "--flaky", "--baseline-max", "501"],
        &[],
        "--baseline-max must be from 1 to 500",
    );
    refused(
        &["--test", TEST, "--also-hash", path_arg(&fixture.path("gone.txt"))],
        &[],
        "cannot read the --also-hash file",
    );
    // A failure memory that is not in the store.
    refused(
        &test,
        &[("--logged-write", "mem-00000000000fffff")],
        "The walk could not start from mem-00000000000fffff",
    );
    // A report in a directory that is not there.
    let mut args = fixture.prove_args(&fixture.path("no/such/dir/report.json"));
    args.extend(test.map(str::to_string));
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let ran = vestige(&fixture.store, &args);
    assert_eq!(ran.code, Some(1), "{}", ran.text());
    assert!(
        ran.stdout.contains("the directory for --report does not exist"),
        "{}",
        ran.text()
    );

    // None of the refusals wrote a memory.
    let storage = vestige_mcp::strata_memory::open(&fixture.store).expect("open the store");
    assert_eq!(
        storage.get_stats().expect("stats").total_nodes,
        memories_before
    );
    drop(storage);

    // A test that passes on both ends decides nothing: the two runs are
    // on record, there is no report, the worktree is gone.
    let ran = fixture.prove_with(&report_path, "true", &[]);
    assert_eq!(ran.code, Some(1), "{}", ran.text());
    assert!(
        ran.stdout.contains(
            "The script does not pass on v1 and fail on v2, so nothing can be decided with it."
        ),
        "{}",
        ran.text()
    );
    assert!(!report_path.exists(), "no report for an undecided run");
    fixture.assert_no_worktree_left();
    // So does one that cannot run at all, or dies instead of failing.
    for test in ["no-such-command-anywhere", "kill -9 $$"] {
        let ran = fixture.prove_with(&report_path, test, &[]);
        assert_eq!(ran.code, Some(1), "{}", ran.text());
        assert!(
            ran.stdout.contains("so nothing can be decided with it."),
            "{}",
            ran.text()
        );
        assert!(!report_path.exists());
        fixture.assert_no_worktree_left();
    }
}

impl Fixture {
    /// `prove` on v1..v2 with `test` as the one-line test.
    fn prove_with(&self, report: &Path, test: &str, extra: &[&str]) -> Ran {
        let mut args = self.prove_args(report);
        args.extend(["--test", test].map(str::to_string));
        args.extend(extra.iter().map(|arg| (*arg).to_string()));
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        vestige(&self.store, &args)
    }
}

#[test]
fn walk_verify_prove_takes_a_good_ref_that_is_not_an_ancestor() {
    let fixture = fixture(&Plan::default());
    let breaking = &fixture.commits[BREAKING];
    let report_path = fixture.path("report.json");
    let mut args = fixture.prove_args(&report_path);
    let good = args.iter().position(|arg| arg == "--good").unwrap();
    args[good + 1] = "side".to_string();
    args.extend(["--test", TEST, "--no-why"].map(str::to_string));
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let ran = vestige(&fixture.store, &args);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);

    // It says what git bisect will do about it, and git bisect does it:
    // the merge base, c2, is tested before anything is bisected.
    assert!(
        ran.stdout.contains(
            "side is not an ancestor of v2: git bisect tests their merge base first"
        ),
        "{}",
        ran.stdout
    );
    assert_eq!(report["window_commits"], 10);
    let merge_base = probe_on(&report, &fixture.commits[2]);
    assert_eq!(merge_base["phase"], "bisect");
    assert_eq!(merge_base["verdict"], "good");
    // Leads that side already contains are outside the range.
    let outside = report["walk"]["dropped"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["why"] == "outside side..v2")
        .count();
    assert_eq!(outside, 3, "c0, c1 and c2: {}", report["walk"]["dropped"]);
    assert_eq!(report["first_bad_commit"], breaking.as_str());
    assert_eq!(
        rungs(&report),
        [("LEAD", true), ("BOUNDARY", true), ("CONFIRMED", true)]
            .map(|(name, holds)| (name.to_string(), holds))
    );
    assert!(fixture.check(&report_path).ok);
    fixture.assert_no_worktree_left();
}

/// A repository and a store whose only memory is a failure report that
/// shares nothing with any other memory: the walk has no lead to give.
struct Bare {
    dir: TempDir,
    repo: PathBuf,
    store: PathBuf,
    failure: String,
}

impl Bare {
    fn new() -> Self {
        let dir = TempDir::new().expect("temp dir");
        let repo = dir.path().join("repo");
        let store = dir.path().join("store");
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, NOW, &["init", "-q", "-b", "main"]);
        let failure = ingest(
            &store,
            "Failure report: the build went red and nobody knows since when",
            "failure,red-build",
        );
        Self {
            dir,
            repo,
            store,
            failure,
        }
    }

    fn commit(&self, subject: &str) -> String {
        git(&self.repo, NOW, &["add", "-A"]);
        git(&self.repo, NOW, &["commit", "-q", "-m", subject]);
        git(&self.repo, NOW, &["rev-parse", "HEAD"])
    }

    fn prove(&self, test: &str) -> (Ran, PathBuf) {
        let report = self.dir.path().join("report.json");
        let ran = vestige(
            &self.store,
            &[
                "prove",
                "--logged-write",
                &self.failure,
                "--repo",
                path_arg(&self.repo),
                "--good",
                "good",
                "--bad",
                "bad",
                "--test",
                test,
                "--reported-at",
                "2026-03-01T00:00:00Z",
                "--report",
                path_arg(&report),
            ],
        );
        (ran, report)
    }
}

#[test]
fn walk_verify_prove_handles_a_merge_as_the_first_bad_commit_with_no_leads() {
    // Two branches that each pass; together they fail. The merge is the
    // first bad commit.
    let bare = Bare::new();
    let write = |name: &str, text: &str| std::fs::write(bare.repo.join(name), text).unwrap();
    write("a.txt", "a\n");
    write("b.txt", "b\n");
    bare.commit("Base");
    git(&bare.repo, NOW, &["tag", "good"]);
    git(&bare.repo, NOW, &["checkout", "-q", "-b", "right"]);
    write("b.txt", "y\n");
    let right = bare.commit("Right: b becomes y");
    git(&bare.repo, NOW, &["checkout", "-q", "main"]);
    write("a.txt", "x\n");
    let left = bare.commit("Left: a becomes x");
    git(
        &bare.repo,
        NOW,
        &["merge", "-q", "--no-ff", "-m", "Merge right", "right"],
    );
    let merge = git(&bare.repo, NOW, &["rev-parse", "HEAD"]);
    git(&bare.repo, NOW, &["tag", "bad"]);

    let test = r#"if [ "$(cat a.txt)" = x ] && [ "$(cat b.txt)" = y ]; then echo "x meets y"; exit 1; fi; echo fine"#;
    let (ran, report_path) = bare.prove(test);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);

    // No lead: the walk says so and git bisect over the whole range decides.
    assert_eq!(report["walk"]["reached"], 0);
    assert_eq!(report["protocol"]["candidates"], json!([]));
    assert!(
        ran.stdout
            .contains("The walk reaches 0 of them over recorded links."),
        "{}",
        ran.stdout
    );
    assert!(
        ran.stdout
            .contains("The first bad commit is not among the leads; the full git bisect takes over."),
        "{}",
        ran.stdout
    );
    assert_eq!(report["found_in_runs"], 0);
    assert_eq!(report["first_bad_commit"], merge.as_str(), "{}", ran.text());
    assert_eq!(probe_on(&report, &left)["verdict"], "good");
    assert_eq!(probe_on(&report, &right)["verdict"], "good");

    // Its changes are read against its first parent: one change, b.txt.
    assert!(
        ran.stdout.contains(&format!(
            "{} is a merge: its changes are read against its first parent {}",
            &merge[..10],
            &left[..10]
        )),
        "{}",
        ran.stdout
    );
    let why = &report["why"];
    assert_eq!(why["changes_in_commit"], 1);
    assert_eq!(why["minimal_failing_changes"][0]["file"], "b.txt");
    assert_eq!(why["commit_without_minimal"], Value::Null);
    // Undone the same way: back to the first parent's side.
    assert_eq!(why["undo_how"], "whole commit");
    assert_eq!(why["undo_on_bad"], "good");
    let patch = std::fs::read_to_string(bare.dir.path().join("report.undo.patch")).unwrap();
    assert!(patch.contains("-y\n+b\n"), "{patch}");
    assert!(!patch.contains("a.txt"), "{patch}");

    assert_eq!(
        rungs(&report),
        [
            ("LEAD", false),
            ("BOUNDARY", true),
            ("CONFIRMED", true),
            ("ISOLATED", true),
            ("REVERSED", true),
        ]
        .map(|(name, holds)| (name.to_string(), holds))
    );
    assert_eq!(
        rung(&report, "ISOLATED")["statement"],
        "the commit is one change, and the commit before it passes"
    );
    assert_eq!(
        rung(&report, "ISOLATED")["proof"],
        rung(&report, "BOUNDARY")["proof"]
    );

    // The check wants both parents of a merge recorded good.
    let checked = vestige(&bare.store, &["prove", "--check", path_arg(&report_path)]);
    assert!(checked.ok, "{}", checked.text());
    assert_eq!(
        checked
            .stdout
            .matches(": recorded verdict good")
            .count(),
        3,
        "two parents and the undo: {}",
        checked.stdout
    );
    assert!(
        checked
            .stdout
            .contains("the commit is a single change, covered by the parent run"),
        "{}",
        checked.stdout
    );
    assert_eq!(git(&bare.repo, NOW, &["worktree", "list"]).lines().count(), 1);
}

#[test]
fn walk_verify_prove_handles_a_root_commit_as_the_first_bad_commit() {
    // A second history with its own root is merged in, and that root
    // brings the file the test trips over.
    let bare = Bare::new();
    std::fs::write(bare.repo.join("ok.txt"), "ok\n").unwrap();
    bare.commit("Base");
    git(&bare.repo, NOW, &["tag", "good"]);
    git(&bare.repo, NOW, &["checkout", "-q", "--orphan", "other"]);
    git(&bare.repo, NOW, &["rm", "-q", "-rf", "."]);
    std::fs::write(bare.repo.join("poison.txt"), "poison\n").unwrap();
    let root = bare.commit("Other history: add poison");
    git(&bare.repo, NOW, &["checkout", "-q", "main"]);
    git(
        &bare.repo,
        NOW,
        &[
            "merge",
            "-q",
            "--no-ff",
            "--allow-unrelated-histories",
            "-m",
            "Merge other",
            "other",
        ],
    );
    git(&bare.repo, NOW, &["tag", "bad"]);

    let (ran, report_path) = bare.prove(r#"if [ -e poison.txt ]; then echo poisoned; exit 1; fi; echo clean"#);
    assert!(ran.ok, "{}", ran.text());
    let report = read_json(&report_path);
    assert_eq!(report["first_bad_commit"], root.as_str(), "{}", ran.text());
    assert_eq!(report["window_commits"], 2);
    assert!(
        ran.stdout.contains(&format!(
            "{} is a root commit: it has no parent to apply its changes to, so the search inside it is skipped",
            &root[..10]
        )),
        "{}",
        ran.stdout
    );
    let why = &report["why"];
    assert_eq!(why["changes_in_commit"], 0);
    assert_eq!(why["minimal_failing_changes"], json!([]));
    // It can still be undone on the bad ref, and that is tested.
    assert_eq!(why["undo_how"], "whole commit");
    assert_eq!(why["undo_on_bad"], "good");
    assert_eq!(
        rungs(&report),
        [
            ("LEAD", false),
            ("BOUNDARY", false),
            ("CONFIRMED", true),
            ("ISOLATED", false),
            ("REVERSED", true),
        ]
        .map(|(name, holds)| (name.to_string(), holds))
    );
    assert_eq!(
        rung(&report, "ISOLATED")["statement"],
        "a root commit has no parent to apply its changes to, so they were not searched"
    );
    let checked = vestige(&bare.store, &["prove", "--check", path_arg(&report_path)]);
    assert!(checked.ok, "{}", checked.text());
    assert!(
        checked
            .stdout
            .contains("the commit's changes were not searched"),
        "{}",
        checked.stdout
    );
    assert_eq!(git(&bare.repo, NOW, &["worktree", "list"]).lines().count(), 1);
}
