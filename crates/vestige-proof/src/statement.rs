//! The proof bundle: one in-toto Statement v1 in one DSSE envelope, written as
//! a single JSON line (`<tree>.intoto.jsonl`).
//!
//! The subject is the git tree the claims hold for, with `.vestige/proof` left
//! out. The predicate carries every witness envelope verbatim, the claims
//! those witnesses support, the Failure Pairing `closes` edges and the Operator
//! stops of the witnesses' sessions, each named by the memory that is its
//! Strata receipt.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::claims::Claim;
use crate::didkey::TrustedKey;
use crate::dsse::{self, Envelope, INTOTO_PAYLOAD_TYPE};
use crate::{ProofError, err};

pub const STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";
pub const PREDICATE_TYPE: &str = "https://vestige.dev/attestations/proof-carrying-pr/v1";
/// The subject's name; its digest is `{"gitTree": <tree>}`.
pub const SUBJECT_NAME: &str = "git-tree";

/// What a bundle can and cannot show, carried inside every bundle.
pub const CLAIM_BOUNDARY: &str = concat!(
    "The signature shows these exact bytes were signed by the key named in keyid. ",
    "Each claim follows from a witness that key signed when the command finished, bound to this ",
    "tree because the tree was the same before and after the run. It does not show that the key ",
    "was out of an agent's reach, that the tests are meaningful, or that no other bundle was ",
    "signed for the same tree."
);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statement {
    #[serde(rename = "_type")]
    pub kind: String,
    pub subject: Vec<Subject>,
    #[serde(rename = "predicateType")]
    pub predicate_type: String,
    pub predicate: Predicate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    pub name: String,
    pub digest: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Predicate {
    /// The commit `prove` was asked about. The binding is the tree: a later
    /// commit that only adds this bundle has the same tree.
    pub commit: String,
    pub tree: String,
    pub claims: Vec<Claim>,
    /// The allowlist in force at that commit.
    pub allowlist: Vec<String>,
    pub witnesses: Vec<WitnessEntry>,
    pub pairings: Vec<Pairing>,
    pub stops: Vec<Stop>,
    pub sessions: Vec<String>,
    pub producer: String,
    pub generated_at: String,
    pub claim_boundary: String,
}

/// One witness as stored: the envelope its memory holds, verbatim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct WitnessEntry {
    pub memory: String,
    /// The Strata payload digest of that memory's write.
    pub strata_digest: Option<String>,
    pub envelope: Envelope,
}

/// A Failure Pairing `closes` edge recorded in the log: `fix` closed `failure`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Pairing {
    pub failure: String,
    pub fix: String,
    /// The failure's signature tag, `fail-<16 hex>`.
    pub signature: Option<String>,
    pub session: String,
    pub failure_digest: Option<String>,
    pub fix_digest: Option<String>,
}

/// An Operator stop recorded in the log, and the resolution that closed it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Stop {
    pub memory: String,
    /// The gate receipt tag, `op-receipt-<16 hex>`.
    pub receipt: String,
    pub rule: String,
    pub session: String,
    pub closed_by: Option<String>,
    pub strata_digest: Option<String>,
}

impl Statement {
    pub fn new(predicate: Predicate) -> Self {
        Statement {
            kind: STATEMENT_TYPE.to_string(),
            subject: vec![Subject {
                name: SUBJECT_NAME.to_string(),
                digest: BTreeMap::from([("gitTree".to_string(), predicate.tree.clone())]),
            }],
            predicate_type: PREDICATE_TYPE.to_string(),
            predicate,
        }
    }

    pub fn seal(&self, keyid: &str, sign: impl FnOnce(&[u8]) -> [u8; 64]) -> Envelope {
        let body = serde_json::to_vec(self).expect("a statement always serializes");
        dsse::seal(INTOTO_PAYLOAD_TYPE, &body, keyid, sign)
    }

    /// The statement inside a bundle, after its signature checks out.
    pub fn open(env: &Envelope, trusted: &[TrustedKey]) -> Result<(Statement, String), ProofError> {
        let (body, did) = dsse::open(env, trusted, INTOTO_PAYLOAD_TYPE)?;
        let s: Statement = serde_json::from_slice(&body)
            .map_err(|e| ProofError(format!("the statement does not parse: {e}")))?;
        if s.kind != STATEMENT_TYPE {
            return err(format!(
                "_type is {:?}, expected {STATEMENT_TYPE:?}",
                s.kind
            ));
        }
        if s.predicate_type != PREDICATE_TYPE {
            return err(format!(
                "predicateType is {:?}, expected {PREDICATE_TYPE:?}",
                s.predicate_type
            ));
        }
        let [subject] = s.subject.as_slice() else {
            return err("the statement must name exactly one subject");
        };
        if subject.name != SUBJECT_NAME
            || subject.digest.len() != 1
            || subject.digest.get("gitTree") != Some(&s.predicate.tree)
        {
            return err("the subject is not this predicate's git tree");
        }
        if !crate::is_object_id(&s.predicate.tree) {
            return err("the subject tree is not a git object id");
        }
        Ok((s, did))
    }
}

/// A bundle file's text: the envelope as one JSON line.
pub fn bundle_text(env: &Envelope) -> String {
    let mut line = serde_json::to_string(env).expect("an envelope always serializes");
    line.push('\n');
    line
}

/// Parse a bundle file strictly: one JSON object, optionally followed by a
/// single newline, and nothing else.
pub fn parse_bundle(bytes: &[u8]) -> Result<Envelope, ProofError> {
    let body = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    if body.is_empty() || body.contains(&b'\n') {
        return err("a bundle is exactly one JSON line");
    }
    let env: Envelope = serde_json::from_slice(body)
        .map_err(|e| ProofError(format!("the bundle is not a DSSE envelope: {e}")))?;
    if bundle_text(&env).as_bytes() != [body, b"\n"].concat() {
        return err("the bundle is not in its canonical form");
    }
    Ok(env)
}
