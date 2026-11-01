//! The standalone verifier binary against a real git repository, with no
//! Vestige store anywhere: HOME and VESTIGE_DATA_DIR point at an empty folder
//! that must stay empty.

use std::path::Path;
use std::process::{Command, Output};

use ed25519_dalek::Signer as _;
use vestige_proof::claims;
use vestige_proof::statement::{CLAIM_BOUNDARY, Predicate, Statement, WitnessEntry, bundle_text};
use vestige_proof::witness::{WITNESS_TYPE, WitnessRecord, output_digest};
use vestige_proof::{PROOF_DIR, bundle_file_name, didkey, git};

fn sh(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {out:?}");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

struct Fx {
    repo: tempfile::TempDir,
    home: tempfile::TempDir,
    sk: ed25519_dalek::SigningKey,
    did: String,
}

impl Fx {
    fn new() -> Fx {
        let repo = tempfile::TempDir::new().unwrap();
        for args in [
            vec!["init", "-q", "-b", "main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "Test"],
            vec!["config", "commit.gpgsign", "false"],
        ] {
            sh(repo.path(), &args);
        }
        std::fs::write(repo.path().join("lib.rs"), "fn add() {}\n").unwrap();
        sh(repo.path(), &["add", "-A"]);
        sh(repo.path(), &["commit", "-q", "-m", "first"]);
        let sk = ed25519_dalek::SigningKey::from_bytes(&[21; 32]);
        let did = didkey::from_public_key(&sk.verifying_key().to_bytes());
        Fx {
            repo,
            home: tempfile::TempDir::new().unwrap(),
            sk,
            did,
        }
    }

    fn head(&self) -> (String, String) {
        let c = git::commit_id(self.repo.path(), "HEAD").unwrap();
        let t = git::commit_tree(self.repo.path(), &c).unwrap();
        (c, t)
    }

    /// A bundle with one green witness of `cargo test` at HEAD's tree.
    fn bundle(&self) -> String {
        self.bundle_signed(&self.sk, &self.did)
    }

    fn bundle_signed(&self, sk: &ed25519_dalek::SigningKey, did: &str) -> String {
        let (commit, tree) = self.head();
        let (sha, n) = output_digest("ok", "", None);
        let record = WitnessRecord {
            kind: WITNESS_TYPE.into(),
            command: "cargo test".into(),
            exit: Some(0),
            output_sha256: sha,
            output_bytes: n,
            tree: tree.clone(),
            tree_before: Some(tree.clone()),
            head: Some(commit.clone()),
            dir: String::new(),
            tests: None,
            host: "claude".into(),
            session: "s1".into(),
            call: "toolu_1".into(),
            agent: String::new(),
            started_at: None,
            ended_at: "2026-10-29T07:00:00.000Z".into(),
            producer: "test".into(),
        };
        let sign = |m: &[u8]| sk.sign(m).to_bytes();
        let memory = "mem-00000000000000a1".to_string();
        let allowlist = vec!["cargo test".to_string(), "pnpm test".to_string()];
        let p = Predicate {
            commit,
            tree: tree.clone(),
            claims: claims::derive(&tree, &[(memory.clone(), record.clone())], &allowlist),
            allowlist,
            witnesses: vec![WitnessEntry {
                memory,
                strata_digest: None,
                envelope: record.seal(did, sign),
            }],
            pairings: vec![],
            stops: vec![],
            sessions: vec!["s1".into()],
            producer: "test".into(),
            generated_at: "2026-10-29T07:00:01.000Z".into(),
            claim_boundary: CLAIM_BOUNDARY.into(),
        };
        bundle_text(&Statement::new(p).seal(did, sign))
    }

    fn write_bundle(&self) -> std::path::PathBuf {
        self.write_bundle_text(&self.bundle())
    }

    fn write_bundle_text(&self, text: &str) -> std::path::PathBuf {
        let (_, tree) = self.head();
        let dir = self.repo.path().join(PROOF_DIR);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(bundle_file_name(&tree));
        std::fs::write(&path, text).unwrap();
        path
    }

    fn verify(&self, extra: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_vestige-verify-proof"))
            .arg("--repo")
            .arg(self.repo.path())
            .args(extra)
            .env("HOME", self.home.path())
            .env("VESTIGE_DATA_DIR", self.home.path())
            .output()
            .unwrap()
    }
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn a_bundle_beside_the_code_verifies_before_and_after_it_is_committed() {
    let fx = Fx::new();
    fx.write_bundle();
    let ok = fx.verify(&["--trust", &fx.did]);
    assert_eq!(code(&ok), 0, "{ok:?}");
    assert!(
        stdout(&ok).contains("PASS") && stdout(&ok).contains("green        cargo test"),
        "{}",
        stdout(&ok)
    );
    sh(fx.repo.path(), &["add", "-A"]);
    sh(fx.repo.path(), &["commit", "-q", "-m", "proof"]);
    let after = fx.verify(&["--trust", &fx.did]);
    assert_eq!(
        code(&after),
        0,
        "committing the bundle keeps its tree: {after:?}"
    );
    assert_eq!(
        std::fs::read_dir(fx.home.path()).unwrap().count(),
        0,
        "nothing touched a store"
    );
}

#[test]
fn one_flipped_byte_in_the_file_fails_with_exit_one() {
    let fx = Fx::new();
    let path = fx.write_bundle();
    let good = std::fs::read(&path).unwrap();
    for i in [
        0,
        good.len() / 3,
        good.len() / 2,
        good.len() - 2,
        good.len() - 1,
    ] {
        let mut bad = good.clone();
        bad[i] ^= 0x01;
        std::fs::write(&path, &bad).unwrap();
        let out = fx.verify(&["--trust", &fx.did]);
        assert_eq!(code(&out), 1, "byte {i}: {out:?}");
        assert!(stdout(&out).contains("FAIL"));
    }
}

#[test]
fn a_new_commit_without_a_bundle_is_unwitnessed_and_fails() {
    let fx = Fx::new();
    fx.write_bundle();
    std::fs::write(fx.repo.path().join("lib.rs"), "fn add() { untested() }\n").unwrap();
    sh(fx.repo.path(), &["commit", "-qam", "untested"]);
    let out = fx.verify(&["--trust", &fx.did]);
    assert_eq!(code(&out), 1, "{out:?}");
    assert!(
        stdout(&out).contains("no proof bundle for tree"),
        "{}",
        stdout(&out)
    );
}

#[test]
fn the_git_note_the_policy_formats_and_usage_errors() {
    let fx = Fx::new();
    let (commit, _) = fx.head();
    git::write_note(fx.repo.path(), &commit, fx.bundle().trim_end()).unwrap();
    let from_note = fx.verify(&["--notes", "--trust", &fx.did]);
    assert_eq!(code(&from_note), 0, "{from_note:?}");

    let keys = fx.home.path().join("keys.txt");
    std::fs::write(&keys, format!("# owner\n{}\n", fx.did)).unwrap();
    let require = fx.home.path().join("witness.toml");
    std::fs::write(&require, "[witness]\ncommands = [\"pnpm test\"]\n").unwrap();
    let required = fx.verify(&[
        "--trust-file",
        keys.to_str().unwrap(),
        "--require-file",
        require.to_str().unwrap(),
    ]);
    assert_eq!(code(&required), 1);
    assert!(
        stdout(&required).contains("required `pnpm test` is unwitnessed"),
        "{}",
        stdout(&required)
    );
    assert_eq!(code(&fx.verify(&["--trust", &fx.did, "--strict"])), 1);

    let summary = fx.repo.path().join("summary.md");
    let json = fx.verify(&[
        "--trust",
        &fx.did,
        "--format",
        "json",
        "--summary",
        summary.to_str().unwrap(),
    ]);
    let report: serde_json::Value = serde_json::from_str(&stdout(&json)).unwrap();
    assert_eq!(report["ok"], true, "{report}");
    assert_eq!(report["claims"][0]["status"], "green");
    assert!(
        std::fs::read_to_string(&summary)
            .unwrap()
            .contains("| green | `cargo test` |")
    );

    // --base reads the keys and the required commands from another commit.
    std::fs::create_dir_all(fx.repo.path().join(".vestige")).unwrap();
    std::fs::write(
        fx.repo.path().join(".vestige/trusted-keys"),
        format!("{}\n", fx.did),
    )
    .unwrap();
    std::fs::write(
        fx.repo.path().join(".vestige/witness.toml"),
        "[witness]\ncommands = [\"cargo test\"]\n",
    )
    .unwrap();
    sh(fx.repo.path(), &["add", "-A"]);
    sh(fx.repo.path(), &["commit", "-q", "-m", "pin key"]);
    sh(fx.repo.path(), &["branch", "base"]);
    std::fs::write(
        fx.repo.path().join(".vestige/witness.toml"),
        "[witness]\ncommands = [\"cargo test\", \"pnpm test\"]\n",
    )
    .unwrap();
    sh(
        fx.repo.path(),
        &["commit", "-qam", "the base asks for more"],
    );
    sh(
        fx.repo.path(),
        &["checkout", "-q", "-b", "feature", "HEAD~1"],
    );
    fx.write_bundle();
    assert_eq!(code(&fx.verify(&["--base", "base"])), 0);
    let stricter = fx.verify(&["--base", "main"]);
    assert_eq!(
        code(&stricter),
        1,
        "the base branch's list is the one enforced"
    );
    assert!(stdout(&stricter).contains("required `pnpm test` is unwitnessed"));

    assert_eq!(code(&fx.verify(&[])), 2, "no trusted key is a usage error");
    assert_eq!(code(&fx.verify(&["--trust", "did:web:example.com"])), 2);
    assert_eq!(
        code(&fx.verify(&["--format", "xml", "--trust", &fx.did])),
        2
    );
    let other = didkey::from_public_key(
        &ed25519_dalek::SigningKey::from_bytes(&[22; 32])
            .verifying_key()
            .to_bytes(),
    );
    assert_eq!(
        code(&fx.verify(&["--notes", "--trust", &other])),
        1,
        "a key nobody pinned fails"
    );
}

/// The composite Action's own script, run the way GitHub runs it: a bare
/// origin, a fresh clone as the workspace, and the pull request's base branch
/// holding the pinned key and the required commands.
#[test]
fn the_action_script_passes_a_proven_head_and_fails_an_untested_or_self_signed_one() {
    let fx = Fx::new();
    let repo = fx.repo.path();
    std::fs::create_dir_all(repo.join(".vestige")).unwrap();
    std::fs::write(repo.join(".vestige/trusted-keys"), format!("{}\n", fx.did)).unwrap();
    std::fs::write(
        repo.join(".vestige/witness.toml"),
        "[witness]\ncommands = [\"cargo test\"]\n",
    )
    .unwrap();
    sh(repo, &["add", "-A"]);
    sh(repo, &["commit", "-q", "-m", "pin the key"]);
    let origin = tempfile::TempDir::new().unwrap();
    sh(origin.path(), &["init", "-q", "--bare", "-b", "main"]);
    sh(
        repo,
        &["remote", "add", "origin", origin.path().to_str().unwrap()],
    );
    sh(repo, &["push", "-q", "origin", "main"]);

    sh(repo, &["checkout", "-q", "-b", "feature"]);
    std::fs::write(repo.join("lib.rs"), "fn add() {}\nfn sub() {}\n").unwrap();
    sh(repo, &["commit", "-qam", "add sub"]);
    fx.write_bundle();
    sh(repo, &["add", "-A"]);
    sh(repo, &["commit", "-q", "-m", "proof"]);
    sh(repo, &["push", "-q", "origin", "feature"]);
    let proven = sh(repo, &["rev-parse", "HEAD"]);

    let ws = tempfile::TempDir::new().unwrap();
    let ws_path = ws.path().join("checkout");
    let cloned = Command::new("git")
        .args([
            "clone",
            "-q",
            origin.path().to_str().unwrap(),
            ws_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(cloned.status.success(), "{cloned:?}");
    let action = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../actions/proof-carrying-pr");
    let run = |head: &str| -> (i32, String, String) {
        sh(&ws_path, &["fetch", "-q", "origin", "feature"]);
        sh(&ws_path, &["checkout", "-q", "--detach", head]);
        let output = ws.path().join("github_output");
        let summary = ws.path().join("step_summary.md");
        let _ = std::fs::remove_file(&output);
        let _ = std::fs::remove_file(&summary);
        let o = Command::new("bash")
            .arg(action.join("run.sh"))
            .env("GITHUB_WORKSPACE", &ws_path)
            .env("GITHUB_ACTION_PATH", &action)
            .env(
                "VESTIGE_VERIFY_PROOF_BIN",
                env!("CARGO_BIN_EXE_vestige-verify-proof"),
            )
            .env("PR_HEAD_SHA", head)
            .env("PR_BASE_REF", "main")
            .env("GITHUB_OUTPUT", &output)
            .env("GITHUB_STEP_SUMMARY", &summary)
            .env("HOME", fx.home.path())
            .output()
            .unwrap();
        (
            o.status.code().unwrap(),
            std::fs::read_to_string(&output).unwrap_or_default(),
            std::fs::read_to_string(&summary).unwrap_or_default(),
        )
    };

    let (code, outputs, summary) = run(&proven);
    assert_eq!(code, 0, "{outputs}\n{summary}");
    assert!(outputs.contains("verified=true"), "{outputs}");
    let (_, tree) = {
        sh(repo, &["checkout", "-q", &proven]);
        fx.head()
    };
    assert!(outputs.contains(&format!("tree={tree}")), "{outputs}");
    assert_eq!(
        summary.matches("### Proof-carrying PR: verified").count(),
        1,
        "{summary}"
    );

    sh(repo, &["checkout", "-q", "feature"]);
    std::fs::write(
        repo.join("lib.rs"),
        "fn add() {}\nfn sub() {}\nfn mul() {}\n",
    )
    .unwrap();
    sh(repo, &["commit", "-qam", "untested change"]);
    sh(repo, &["push", "-q", "origin", "feature"]);
    let untested = sh(repo, &["rev-parse", "HEAD"]);
    let (code, outputs, _) = run(&untested);
    assert_eq!(code, 1);
    assert!(outputs.contains("verified=false"), "{outputs}");

    // The pull request pins its own key and signs its own bundle. The key
    // list comes from the base branch, so that changes nothing.
    let rogue = ed25519_dalek::SigningKey::from_bytes(&[23; 32]);
    let rogue_did = didkey::from_public_key(&rogue.verifying_key().to_bytes());
    std::fs::write(repo.join(".vestige/trusted-keys"), format!("{rogue_did}\n")).unwrap();
    sh(repo, &["commit", "-qam", "trust my own key"]);
    fx.write_bundle_text(&fx.bundle_signed(&rogue, &rogue_did));
    sh(repo, &["add", "-A"]);
    sh(repo, &["commit", "-q", "-m", "self-signed proof"]);
    sh(repo, &["push", "-q", "origin", "feature"]);
    let self_signed = sh(repo, &["rev-parse", "HEAD"]);
    let (code, outputs, summary) = run(&self_signed);
    assert_eq!(code, 1, "{summary}");
    assert!(outputs.contains("verified=false"), "{outputs}");
}
