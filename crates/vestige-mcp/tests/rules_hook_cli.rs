//! `vestige hook session-start` on the real binary: the Claude Code
//! SessionStart payloads recorded from Claude Code 2.1.296 (one per `source`,
//! user paths replaced) give valid hook JSON naming exactly the decision and
//! correction memories under the session's exact tags: `pinned` and the rules
//! file by default, plus the repository name with `--repo-tag`. No store, an
//! empty store, a bad payload or free text give empty stdout and exit 0. The
//! hook reads through a running owner and writes nothing to the log.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use tempfile::TempDir;
use vestige_core::{IngestInput, Storage};

const SOURCES: [&str; 5] = ["startup", "resume", "clear", "compact", "fork"];

struct Hook {
    code: Option<i32>,
    stdout: String,
    elapsed: Duration,
}

fn fixture(source: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/session_start")
        .join(format!("{source}.json"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A recorded payload with its `cwd` pointed at `cwd`.
fn payload(source: &str, cwd: &Path) -> String {
    let mut value: Value = serde_json::from_str(&fixture(source)).expect("fixture is JSON");
    value["cwd"] = Value::String(cwd.to_string_lossy().into_owned());
    value.to_string()
}

fn hook(data_dir: &Path, stdin: &str) -> Hook {
    hook_with(data_dir, stdin, &[])
}

fn hook_with(data_dir: &Path, stdin: &str, extra: &[&str]) -> Hook {
    let started = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .args(["hook", "session-start"])
        .args(extra)
        .arg("--data-dir")
        .arg(data_dir)
        .env("NO_COLOR", "1")
        .env_remove("VESTIGE_DATA_DIR")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn vestige");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(stdin.as_bytes())
        .expect("write payload");
    let output = child.wait_with_output().expect("wait");
    Hook {
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        elapsed: started.elapsed(),
    }
}

/// The hook's `additionalContext`, after checking the documented shape.
fn context(out: &Hook) -> String {
    assert_eq!(out.code, Some(0), "stdout: {}", out.stdout);
    assert_eq!(
        out.stdout.lines().count(),
        1,
        "one JSON line: {}",
        out.stdout
    );
    let value: Value = serde_json::from_str(out.stdout.trim()).expect("stdout is JSON");
    let object = value.as_object().expect("an object");
    assert_eq!(object.len(), 1, "{value}");
    let specific = &value["hookSpecificOutput"];
    assert_eq!(specific["hookEventName"], "SessionStart", "{value}");
    specific["additionalContext"]
        .as_str()
        .expect("additionalContext is a string")
        .to_string()
}

/// First line of the pinned-rule block. The default rulebook comes before
/// it, and its bullets are not memory ids.
const RULES_HEADER: &str = "Vestige rules for this session";

/// The memory ids listed in the pinned-rule block, in order.
fn listed_ids(context: &str) -> Vec<String> {
    context
        .lines()
        .skip_while(|line| !line.starts_with(RULES_HEADER))
        .filter_map(|line| line.strip_prefix("- "))
        .filter_map(|rest| rest.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

fn put(storage: &Arc<Storage>, content: &str, node_type: &str, tags: &[&str]) -> String {
    storage
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                node_type: node_type.to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                ..Default::default()
            },
            "user",
        )
        .expect("ingest")
        .id
}

/// A git checkout named `tidetable` (no remote, so its folder name is the
/// repository name).
fn repo(root: &Path) -> PathBuf {
    let repo = root.join("tidetable");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    repo
}

struct Seeded {
    data: TempDir,
    work: TempDir,
    repo_rule: String,
    pinned_rule: String,
    rules_file_rule: String,
}

/// Two rules under the session's tags, one under a tag the rules file names,
/// and decoys that must never show: a near-miss tag, another case, a word in
/// the content, a pinned fact, and free text.
fn seed() -> Seeded {
    let data = TempDir::new().unwrap();
    let work = TempDir::new().unwrap();
    let storage = vestige_mcp::strata_memory::open(data.path()).expect("open strata log");
    let repo_rule = put(
        &storage,
        "Tide heights are stored in metres, never feet.",
        "decision",
        &["tidetable"],
    );
    let pinned_rule = put(
        &storage,
        "Never push to main without the tide fixture tests passing.",
        "correction",
        &["pinned", "ci"],
    );
    let rules_file_rule = put(
        &storage,
        "Release builds are cut from the release branch only.",
        "decision",
        &["release-policy"],
    );
    put(
        &storage,
        "DECOY near-miss tag",
        "decision",
        &["tidetable-old"],
    );
    put(
        &storage,
        "DECOY other case",
        "decision",
        &["Pinned", "TIDETABLE"],
    );
    put(
        &storage,
        "DECOY mentions tidetable and pinned in its words only",
        "decision",
        &["notes"],
    );
    put(
        &storage,
        "DECOY a pinned fact is not a rule",
        "fact",
        &["pinned"],
    );
    put(&storage, "DECOY deploy rule", "decision", &["deploy"]);
    drop(storage);
    Seeded {
        data,
        work,
        repo_rule,
        pinned_rule,
        rules_file_rule,
    }
}

/// blake3 over the log directory's files, in path order.
fn log_digest(data_dir: &Path) -> String {
    let log = data_dir.join("log");
    let mut files = Vec::new();
    let mut stack = vec![log.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|name| name != "strata.lock") {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut hasher = blake3::Hasher::new();
    for path in files {
        hasher.update(
            path.strip_prefix(&log)
                .unwrap()
                .to_string_lossy()
                .as_bytes(),
        );
        hasher.update(&std::fs::read(&path).unwrap());
    }
    hasher.finalize().to_hex().to_string()
}

#[test]
fn every_recorded_source_gives_valid_hook_json_with_exactly_the_pinned_rules() {
    let seeded = seed();
    let repo = repo(seeded.work.path());
    for source in SOURCES {
        // By default: `pinned` only, even inside the `tidetable` checkout.
        let out = hook(seeded.data.path(), &payload(source, &repo));
        let text = context(&out);
        assert!(text.contains(&format!("(source: {source})")), "{text}");
        assert!(text.contains("exact tags [pinned]"), "{text}");
        assert!(text.contains("not by search or similarity"), "{text}");
        assert_eq!(
            listed_ids(&text),
            vec![seeded.pinned_rule.clone()],
            "exactly the pinned rule: {text}"
        );
        assert!(!text.contains("DECOY"), "{text}");
        assert!(
            out.elapsed < Duration::from_secs(10),
            "{source} took {:?}",
            out.elapsed
        );

        // With --repo-tag the repository name is an exact tag too.
        let out = hook_with(seeded.data.path(), &payload(source, &repo), &["--repo-tag"]);
        let text = context(&out);
        assert!(text.contains("exact tags [tidetable, pinned]"), "{text}");
        let mut ids = listed_ids(&text);
        ids.sort();
        let mut want = vec![seeded.repo_rule.clone(), seeded.pinned_rule.clone()];
        want.sort();
        assert_eq!(ids, want, "exactly the two tagged rules: {text}");
        assert!(!text.contains("DECOY"), "{text}");
    }
}

#[test]
fn the_rules_file_adds_exact_tags_and_free_text_never_matches() {
    let seeded = seed();
    let repo = repo(seeded.work.path());
    // A sentence that contains a stored tag is not that tag.
    std::fs::write(
        repo.join(".vestige-rules"),
        "# one exact tag per line\nthe deploy rule for release-policy\nrelease-policy\n",
    )
    .unwrap();
    let text = context(&hook(seeded.data.path(), &payload("compact", &repo)));
    assert!(
        text.contains("[pinned, the deploy rule for release-policy, release-policy]"),
        "{text}"
    );
    let ids = listed_ids(&text);
    assert_eq!(ids.len(), 2, "{text}");
    assert!(ids.contains(&seeded.rules_file_rule), "{text}");
    assert!(!ids.contains(&seeded.repo_rule), "{text}");
    assert!(!text.contains("DECOY deploy rule"), "{text}");

    // Naming the repository in the rules file opts its tag in without the flag.
    std::fs::write(repo.join(".vestige-rules"), "release-policy\ntidetable\n").unwrap();
    let text = context(&hook(seeded.data.path(), &payload("compact", &repo)));
    assert!(
        text.contains("[pinned, release-policy, tidetable]"),
        "{text}"
    );
    let ids = listed_ids(&text);
    assert_eq!(ids.len(), 3, "{text}");
    assert!(ids.contains(&seeded.repo_rule), "{text}");

    // Only free text: nothing resolves, nothing is printed.
    std::fs::remove_file(repo.join(".vestige-rules")).unwrap();
    let elsewhere = seeded.work.path().join("no-such-project");
    std::fs::create_dir_all(elsewhere.join(".git")).unwrap();
    let data = TempDir::new().unwrap();
    let storage = vestige_mcp::strata_memory::open(data.path()).unwrap();
    put(
        &storage,
        "DECOY words: no-such-project pinned rules",
        "decision",
        &["x"],
    );
    drop(storage);
    // The default rulebook still loads; no rule block is printed.
    let text = context(&hook(data.path(), &payload("startup", &elsewhere)));
    assert_rulebook_only(&text);
    assert!(!text.contains("DECOY"), "free text matched: {text}");
}

/// The default rulebook and nothing resolved by tag: no rule block, no ids.
fn assert_rulebook_only(text: &str) {
    assert!(
        text.starts_with("VESTIGE DEFAULT RULEBOOK"),
        "the default rulebook loads: {text}"
    );
    assert!(
        text.contains(&format!("`{}`", vestige_mcp::rulebook::HANDLE)),
        "{text}"
    );
    assert!(!text.contains(RULES_HEADER), "no rule block: {text}");
    assert!(listed_ids(text).is_empty(), "{text}");
}

#[test]
fn no_store_or_a_bad_payload_prints_nothing_and_an_empty_store_only_the_rulebook() {
    let work = TempDir::new().unwrap();
    let repo = repo(work.path());

    // No store: nothing printed, and no store is created.
    let missing = TempDir::new().unwrap();
    let dir = missing.path().join("never-created");
    let out = hook(&dir, &payload("startup", &repo));
    assert_eq!((out.code, out.stdout.as_str()), (Some(0), ""));
    assert!(!dir.join("log").exists(), "the hook created a store");

    // An empty store: the default rulebook, and no rule block.
    let empty = TempDir::new().unwrap();
    drop(vestige_mcp::strata_memory::open(empty.path()).unwrap());
    for source in SOURCES {
        let text = context(&hook(empty.path(), &payload(source, &repo)));
        assert_rulebook_only(&text);
    }

    // Payloads it cannot use.
    let seeded = seed();
    for bad in [
        "",
        "not json",
        "[1]",
        r#"{"hook_event_name":"Stop","cwd":"/"}"#,
    ] {
        let out = hook(seeded.data.path(), bad);
        assert_eq!((out.code, out.stdout.as_str()), (Some(0), ""), "{bad:?}");
    }
}

#[test]
fn a_holder_that_never_answers_runs_out_the_time_limit_silently() {
    let seeded = seed();
    let repo = repo(seeded.work.path());
    // This process holds the serve lock and publishes no endpoint: the hook
    // can neither open the log nor attach, and must give up on time.
    let _lock = vestige_mcp::attach::try_serve_lock(seeded.data.path())
        .unwrap()
        .expect("the lock is free");
    let started = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .args(["hook", "session-start", "--timeout-ms", "700", "--data-dir"])
        .arg(seeded.data.path())
        .env_remove("VESTIGE_DATA_DIR")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn vestige");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload("compact", &repo).as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let elapsed = started.elapsed();
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.is_empty(), "{:?}", output.stdout);
    assert!(elapsed >= Duration::from_millis(600), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
}

#[test]
fn reading_the_log_directly_writes_nothing() {
    let seeded = seed();
    let repo = repo(seeded.work.path());
    let before = log_digest(seeded.data.path());
    let text = context(&hook_with(
        seeded.data.path(),
        &payload("compact", &repo),
        &["--repo-tag"],
    ));
    assert_eq!(listed_ids(&text).len(), 2, "{text}");
    assert_eq!(
        before,
        log_digest(seeded.data.path()),
        "the hook wrote the log"
    );
}

/// A `vestige-mcp` serving `data_dir`, as an MCP client starts it.
struct Owner(Child);

impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start_owner(data_dir: &Path, home: &Path) -> Owner {
    let child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
        .env("VESTIGE_DATA_DIR", data_dir)
        .env("HOME", home)
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "0")
        .env("VESTIGE_AUTOPILOT_ENABLED", "0")
        .env("VESTIGE_TRACE", "0")
        .env_remove("RUST_LOG")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn vestige-mcp");
    let owner = Owner(child);
    let deadline = Instant::now() + Duration::from_secs(60);
    while vestige_mcp::attach::probe_owner_blocking(data_dir).is_none() {
        assert!(Instant::now() < deadline, "vestige-mcp never took attaches");
        std::thread::sleep(Duration::from_millis(100));
    }
    owner
}

#[test]
fn a_running_owner_answers_the_hook() {
    let seeded = seed();
    let repo = repo(seeded.work.path());
    let home = TempDir::new().unwrap();
    let _owner = start_owner(seeded.data.path(), home.path());
    // The default, through the owner: the pinned rule only.
    let text = context(&hook(seeded.data.path(), &payload("compact", &repo)));
    assert_eq!(
        listed_ids(&text),
        vec![seeded.pinned_rule.clone()],
        "{text}"
    );
    let out = hook_with(
        seeded.data.path(),
        &payload("compact", &repo),
        &["--repo-tag"],
    );
    let text = context(&out);
    let mut ids = listed_ids(&text);
    ids.sort();
    let mut want = vec![seeded.repo_rule.clone(), seeded.pinned_rule.clone()];
    want.sort();
    assert_eq!(ids, want, "{text}");
    // Through the owner the lines carry the day each rule was written.
    assert!(
        text.lines()
            .filter(|line| line.starts_with("- mem-"))
            .all(|line| line.contains(", 20")),
        "{text}"
    );
    assert!(out.elapsed < Duration::from_secs(10), "{:?}", out.elapsed);
}
