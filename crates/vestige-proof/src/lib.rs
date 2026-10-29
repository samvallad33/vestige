//! Proof-carrying pull requests.
//!
//! A witness is the signed record of one allowlisted test command an agent
//! ran: the exact command, its exit code, the SHA-256 of the output the host
//! reported, the git tree of the working directory before and after the run,
//! and the number of tests the runner collected when it said so in JSON or
//! junit. Reflex records each witness as a Strata memory tagged
//! `tree-<tree hash>`.
//!
//! `prove` gathers the witnesses bound to one commit's tree, with the Failure
//! Pairing `closes` edges and Operator stops of the same sessions, into an
//! in-toto Statement v1 inside a DSSE envelope. Every claim in it is derived
//! from a signed witness by [`claims::derive`], never written as prose, and a
//! command with no witness is listed as `unwitnessed`.
//!
//! [`verify::verify_bundle`] checks a bundle offline against pinned keys. It
//! never opens a Vestige store: it reads the bundle bytes and, for a commit,
//! asks local git for that commit's tree.

pub mod claims;
pub mod cli;
pub mod config;
pub mod didkey;
pub mod dsse;
pub mod git;
pub mod statement;
pub mod tests_count;
pub mod verify;
pub mod witness;

use sha2::Digest as _;

/// The folder inside a repository that holds exported bundles. It is left out
/// of every tree hash, so committing a bundle never changes the tree it proves.
pub const PROOF_DIR: &str = ".vestige/proof";

/// The repository's own witness config, read beside the owner's `vestige.toml`.
pub const REPO_CONFIG: &str = ".vestige/witness.toml";

/// Trusted signer keys, one `did:key` per line, read from the base branch.
pub const TRUSTED_KEYS: &str = ".vestige/trusted-keys";

/// The git notes ref a bundle can be stored under instead of a file.
pub const NOTES_REF: &str = "refs/notes/vestige-proof";

/// Lowercase hex SHA-256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A git object id: 40 lowercase hex (SHA-1) or 64 (SHA-256).
pub fn is_object_id(s: &str) -> bool {
    matches!(s.len(), 40 | 64) && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The file name a bundle for this tree is exported under.
pub fn bundle_file_name(tree: &str) -> String {
    format!("{tree}.intoto.jsonl")
}

/// Everything that can make a proof fail to build or verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofError(pub String);

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProofError {}

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T, ProofError> {
    Err(ProofError(msg.into()))
}
