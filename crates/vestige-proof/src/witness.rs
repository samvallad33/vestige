//! One witness: the signed record of one allowlisted test command.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::didkey::TrustedKey;
use crate::dsse::{self, Envelope, WITNESS_PAYLOAD_TYPE};
use crate::tests_count::TestCount;
use crate::{ProofError, err, is_object_id, sha256_hex};

/// `_type` of a witness record.
pub const WITNESS_TYPE: &str = "https://vestige.dev/witness/v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct WitnessRecord {
    #[serde(rename = "_type")]
    pub kind: String,
    /// The exact command, equal to an allowlist entry.
    pub command: String,
    /// The process exit code; `None` when the host reported a failure without one.
    pub exit: Option<i64>,
    /// SHA-256 of the output the host reported, see [`output_digest`].
    pub output_sha256: String,
    pub output_bytes: u64,
    /// The working tree when the command finished, from `git write-tree` over a
    /// copy of the index with every change staged and `.vestige/proof` left out.
    pub tree: String,
    /// The same tree taken just before the command started. A run is bound to
    /// a tree only when both are equal: nothing changed the files while it ran.
    pub tree_before: Option<String>,
    /// The commit checked out, for reading only; the binding is the tree.
    pub head: Option<String>,
    /// The working directory relative to the repository root, `""` at the root.
    pub dir: String,
    /// What the runner said it collected, when it said so in JSON or junit.
    pub tests: Option<TestCount>,
    pub host: String,
    pub session: String,
    pub call: String,
    /// The subagent id, `""` for the main agent.
    pub agent: String,
    pub started_at: Option<String>,
    pub ended_at: String,
    pub producer: String,
}

/// The output a host reported, as canonical JSON: `{"stderr":…,"stdout":…}`
/// for a call that succeeded, `{"error":…}` for one that failed. Returns its
/// SHA-256 and length, so anyone holding the same output can check the digest.
pub fn output_digest(stdout: &str, stderr: &str, error: Option<&str>) -> (String, u64) {
    let body = match error {
        Some(e) => json!({ "error": e }),
        None => json!({ "stderr": stderr, "stdout": stdout }),
    };
    let bytes = body.to_string().into_bytes();
    (sha256_hex(&bytes), bytes.len() as u64)
}

/// The exit code in a failed shell call's error text: its `Exit code N` line.
pub fn exit_code_in(error: &str) -> Option<i64> {
    error
        .lines()
        .map(str::trim)
        .find_map(|l| l.strip_prefix("Exit code "))
        .and_then(|n| n.trim().parse().ok())
}

impl WitnessRecord {
    pub fn validate(&self) -> Result<(), ProofError> {
        if self.kind != WITNESS_TYPE {
            return err(format!("witness _type is {:?}", self.kind));
        }
        if self.command.trim().is_empty() || self.command.trim() != self.command {
            return err("witness command is empty or carries surrounding whitespace");
        }
        if !is_object_id(&self.tree) {
            return err(format!(
                "witness tree {:?} is not a git object id",
                self.tree
            ));
        }
        if let Some(b) = &self.tree_before
            && !is_object_id(b)
        {
            return err(format!("witness treeBefore {b:?} is not a git object id"));
        }
        if self.output_sha256.len() != 64
            || !self.output_sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return err("witness outputSha256 is not a SHA-256");
        }
        Ok(())
    }

    /// Bound to `tree`: taken there both before and after the run.
    pub fn bound_to(&self, tree: &str) -> bool {
        self.tree == tree && self.tree_before.as_deref() == Some(tree)
    }

    pub fn seal(&self, keyid: &str, sign: impl FnOnce(&[u8]) -> [u8; 64]) -> Envelope {
        let body = serde_json::to_vec(self).expect("a witness record always serializes");
        dsse::seal(WITNESS_PAYLOAD_TYPE, &body, keyid, sign)
    }

    pub fn open(
        env: &Envelope,
        trusted: &[TrustedKey],
    ) -> Result<(WitnessRecord, String), ProofError> {
        let (body, did) = dsse::open(env, trusted, WITNESS_PAYLOAD_TYPE)?;
        let record: WitnessRecord = serde_json::from_slice(&body)
            .map_err(|e| ProofError(format!("witness record does not parse: {e}")))?;
        record.validate()?;
        Ok((record, did))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn record(
        command: &str,
        exit: Option<i64>,
        tree: &str,
        ended: &str,
    ) -> WitnessRecord {
        let (sha, n) = output_digest("ok", "", None);
        WitnessRecord {
            kind: WITNESS_TYPE.into(),
            command: command.into(),
            exit,
            output_sha256: sha,
            output_bytes: n,
            tree: tree.into(),
            tree_before: Some(tree.into()),
            head: None,
            dir: String::new(),
            tests: None,
            host: "claude".into(),
            session: "s1".into(),
            call: "toolu_1".into(),
            agent: String::new(),
            started_at: None,
            ended_at: ended.into(),
            producer: "vestige test".into(),
        }
    }

    #[test]
    fn the_output_digest_separates_streams_and_failures() {
        let (a, _) = output_digest("ab", "", None);
        let (b, _) = output_digest("a", "b", None);
        let (c, _) = output_digest("", "", Some("ab"));
        assert!(a != b && b != c && a != c);
        assert_eq!(
            output_digest("x", "y", None).0,
            sha256_hex(br#"{"stderr":"y","stdout":"x"}"#)
        );
    }

    #[test]
    fn the_exit_code_comes_from_its_own_line() {
        assert_eq!(exit_code_in("Exit code 101\nerror: x"), Some(101));
        assert_eq!(exit_code_in("Command timed out"), None);
    }

    #[test]
    fn a_sealed_record_opens_and_a_run_that_changed_the_tree_is_not_bound() {
        let (sk, tk) = crate::dsse::tests::key(5);
        let tree = "a".repeat(40);
        let mut r = record("cargo test", Some(0), &tree, "2026-10-29T07:00:00.000Z");
        let env = r.seal(&tk.did, |m| {
            use ed25519_dalek::Signer as _;
            sk.sign(m).to_bytes()
        });
        let (back, _) = WitnessRecord::open(&env, &[tk]).unwrap();
        assert_eq!(back, r);
        assert!(back.bound_to(&tree));
        r.tree_before = Some("b".repeat(40));
        assert!(!r.bound_to(&tree));
        r.tree_before = None;
        assert!(!r.bound_to(&tree));
    }
}
