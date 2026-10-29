//! The offline verifier. It reads a bundle's bytes and nothing else: no
//! store, no network. Every check must pass for a bundle to verify:
//!
//! 1. The file is one DSSE envelope in canonical form, signed by exactly one
//!    pinned key, whose keyid is that key's `did:key`.
//! 2. It holds an in-toto Statement v1 with the proof-carrying-pr predicate
//!    and exactly one subject, the git tree, which must be the tree asked for.
//! 3. Every witness envelope inside opens against a pinned key and is bound
//!    to that tree.
//! 4. The claims equal what the witnesses and the allowlist give through
//!    `claims::derive`. A signer that writes a claim no witness supports fails
//!    here.
//! 5. The policy holds: at least one green claim, none red and none with zero
//!    tests, every `--require`d command green, and with `--strict` every
//!    allowlisted command green.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::claims::{self, Claim, Status};
use crate::didkey::TrustedKey;
use crate::statement::{Statement, parse_bundle};
use crate::witness::WitnessRecord;

/// What the bundle must show beyond verifying.
#[derive(Clone, Debug, Default)]
pub struct Policy {
    /// Commands that must each have a green claim.
    pub require: Vec<String>,
    /// Every allowlisted command must be green.
    pub strict: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Report {
    pub ok: bool,
    pub tree: Option<String>,
    pub commit: Option<String>,
    pub signer: Option<String>,
    pub claims: Vec<Claim>,
    pub witnesses: usize,
    pub pairings: usize,
    pub stops: usize,
    pub sessions: usize,
    /// Every reason the bundle failed, in the order found.
    pub failures: Vec<String>,
    pub claim_boundary: Option<String>,
}

impl Report {
    fn fail(mut self, why: impl Into<String>) -> Report {
        self.failures.push(why.into());
        self.ok = false;
        self
    }
}

/// Verify one bundle's bytes against the pinned keys, for `expect_tree`.
pub fn verify_bundle(
    bytes: &[u8],
    trusted: &[TrustedKey],
    expect_tree: Option<&str>,
    policy: &Policy,
) -> Report {
    let report = Report::default();
    if trusted.is_empty() {
        return report.fail("no trusted key was given");
    }
    let env = match parse_bundle(bytes) {
        Ok(e) => e,
        Err(e) => return report.fail(e.0),
    };
    let (statement, signer) = match Statement::open(&env, trusted) {
        Ok(x) => x,
        Err(e) => return report.fail(e.0),
    };
    let p = statement.predicate;
    let mut report = Report {
        ok: true,
        tree: Some(p.tree.clone()),
        commit: Some(p.commit.clone()),
        signer: Some(signer),
        claims: p.claims.clone(),
        witnesses: p.witnesses.len(),
        pairings: p.pairings.len(),
        stops: p.stops.len(),
        sessions: p.sessions.len(),
        failures: Vec::new(),
        claim_boundary: Some(p.claim_boundary.clone()),
    };
    if let Some(want) = expect_tree
        && want != p.tree
    {
        report = report.fail(format!(
            "the bundle proves tree {}, not the tree asked for, {want}",
            p.tree
        ));
    }

    let mut records: Vec<(String, WitnessRecord)> = Vec::new();
    let mut seen = BTreeSet::new();
    for w in &p.witnesses {
        if !seen.insert(w.memory.clone()) {
            report = report.fail(format!("witness {} appears twice", w.memory));
            continue;
        }
        match WitnessRecord::open(&w.envelope, trusted) {
            Ok((r, _)) if r.bound_to(&p.tree) => records.push((w.memory.clone(), r)),
            Ok(_) => {
                report = report.fail(format!(
                    "witness {} is not bound to tree {}",
                    w.memory, p.tree
                ))
            }
            Err(e) => report = report.fail(format!("witness {}: {e}", w.memory)),
        }
    }
    let derived = claims::derive(&p.tree, &records, &p.allowlist);
    if derived != p.claims {
        report = report.fail("the claims do not follow from the witnesses in the bundle");
        report.claims = derived;
    }

    let claims = report.claims.clone();
    if !claims.iter().any(|c| c.status == Status::Green) {
        report = report.fail(format!("no green witness at tree {}", p.tree));
    }
    for c in &claims {
        if matches!(c.status, Status::Red | Status::ZeroTests) {
            report = report.fail(format!(
                "`{}` is {} at this tree",
                c.command,
                c.status.as_str()
            ));
        }
    }
    for cmd in &policy.require {
        let status = claims
            .iter()
            .find(|c| c.command == *cmd)
            .map_or(Status::Unwitnessed, |c| c.status);
        if status != Status::Green {
            report = report.fail(format!(
                "required `{cmd}` is {} at this tree",
                status.as_str()
            ));
        }
    }
    if policy.strict {
        for c in claims.iter().filter(|c| c.status == Status::Unwitnessed) {
            report = report.fail(format!("`{}` is unwitnessed at this tree", c.command));
        }
    }
    report
}

/// The report as plain lines.
pub fn render_text(r: &Report) -> String {
    let mut out = String::new();
    let short = |s: &Option<String>| s.as_deref().map(|t| t.chars().take(12).collect::<String>());
    out.push_str(&format!(
        "vestige-verify-proof: {}  tree {}  commit {}  signer {}\n",
        if r.ok { "PASS" } else { "FAIL" },
        short(&r.tree).unwrap_or_else(|| "-".into()),
        short(&r.commit).unwrap_or_else(|| "-".into()),
        r.signer.as_deref().unwrap_or("-"),
    ));
    for c in &r.claims {
        out.push_str(&format!(
            "  {:<12} {}{}{}\n",
            c.status.as_str(),
            c.command,
            c.witness
                .as_deref()
                .map(|w| format!("  witness {w}"))
                .unwrap_or_default(),
            c.tests.map(|t| format!("  {t} tests")).unwrap_or_default(),
        ));
    }
    out.push_str(&format!(
        "  {} witnesses, {} Failure Pairing closes edges, {} Operator stops, {} sessions\n",
        r.witnesses, r.pairings, r.stops, r.sessions
    ));
    for f in &r.failures {
        out.push_str(&format!("  why: {f}\n"));
    }
    out
}

/// The report as a Markdown job summary.
pub fn render_markdown(r: &Report) -> String {
    let mut out = format!(
        "### Proof-carrying PR: {}\n\nTree `{}`, commit `{}`, signed by `{}`.\n\n| Status | Command | Witness | Tests |\n|---|---|---|---|\n",
        if r.ok { "verified" } else { "not verified" },
        r.tree.as_deref().unwrap_or("-"),
        r.commit.as_deref().unwrap_or("-"),
        r.signer.as_deref().unwrap_or("-"),
    );
    for c in &r.claims {
        out.push_str(&format!(
            "| {} | `{}` | {} | {} |\n",
            c.status.as_str(),
            c.command.replace('|', "\\|"),
            c.witness.as_deref().unwrap_or("none"),
            c.tests
                .map_or_else(|| "not reported".to_string(), |t| t.to_string()),
        ));
    }
    out.push_str(&format!(
        "\n{} witnesses, {} Failure Pairing closes edges, {} Operator stops.\n",
        r.witnesses, r.pairings, r.stops
    ));
    if !r.failures.is_empty() {
        out.push('\n');
        for f in &r.failures {
            out.push_str(&format!("- {f}\n"));
        }
    }
    if let Some(b) = &r.claim_boundary {
        out.push_str(&format!("\n{b}\n"));
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::dsse::tests::key;
    use crate::statement::{CLAIM_BOUNDARY, Pairing, Predicate, Stop, WitnessEntry, bundle_text};
    use crate::witness::tests::record;
    use ed25519_dalek::Signer as _;

    fn tree() -> String {
        "a1".repeat(20)
    }

    /// A bundle with a red then a green run of one command, one more
    /// allowlisted command never run, one pairing and one stop.
    pub(crate) fn bundle(sk: &ed25519_dalek::SigningKey, did: &str) -> (Vec<u8>, Predicate) {
        let t = tree();
        let sign = |m: &[u8]| sk.sign(m).to_bytes();
        let red = record("cargo test", Some(101), &t, "2026-10-29T07:00:00.000Z");
        let green = record("cargo test", Some(0), &t, "2026-10-29T07:05:00.000Z");
        let records = vec![
            ("mem-00000000000000a1".to_string(), red),
            ("mem-00000000000000a2".to_string(), green),
        ];
        let allowlist = vec!["cargo test".to_string(), "pnpm test".to_string()];
        let predicate = Predicate {
            commit: "b2".repeat(20),
            tree: t.clone(),
            claims: claims::derive(&t, &records, &allowlist),
            allowlist,
            witnesses: records
                .iter()
                .map(|(m, r)| WitnessEntry {
                    memory: m.clone(),
                    strata_digest: Some("d".repeat(64)),
                    envelope: r.seal(did, sign),
                })
                .collect(),
            pairings: vec![Pairing {
                failure: "mem-00000000000000b1".into(),
                fix: "mem-00000000000000b2".into(),
                signature: Some("fail-0123456789abcdef".into()),
                session: "s1".into(),
                failure_digest: None,
                fix_digest: None,
            }],
            stops: vec![Stop {
                memory: "mem-00000000000000c1".into(),
                receipt: "op-receipt-0123456789abcdef".into(),
                rule: "OP-003".into(),
                session: "s1".into(),
                closed_by: None,
                strata_digest: None,
            }],
            sessions: vec!["s1".into()],
            producer: "vestige test".into(),
            generated_at: "2026-10-29T07:06:00.000Z".into(),
            claim_boundary: CLAIM_BOUNDARY.into(),
        };
        let env = Statement::new(predicate.clone()).seal(did, sign);
        (bundle_text(&env).into_bytes(), predicate)
    }

    fn resign(sk: &ed25519_dalek::SigningKey, did: &str, p: Predicate) -> Vec<u8> {
        let env = Statement::new(p).seal(did, |m| sk.sign(m).to_bytes());
        bundle_text(&env).into_bytes()
    }

    #[test]
    fn a_good_bundle_verifies_and_reports_its_claims() {
        let (sk, tk) = key(11);
        let (bytes, _) = bundle(&sk, &tk.did);
        let r = verify_bundle(
            &bytes,
            std::slice::from_ref(&tk),
            Some(&tree()),
            &Policy::default(),
        );
        assert!(r.ok, "{:?}", r.failures);
        assert_eq!(r.claims.len(), 2);
        assert_eq!(r.claims[0].status, Status::Green);
        assert_eq!(r.claims[0].runs, 2);
        assert_eq!(r.claims[1].status, Status::Unwitnessed);
        assert_eq!((r.witnesses, r.pairings, r.stops), (2, 1, 1));
        assert!(render_text(&r).contains("PASS"));
        assert!(
            render_markdown(&r)
                .contains("| green | `cargo test` | mem-00000000000000a2 | not reported |")
        );
    }

    #[test]
    fn flipping_any_single_byte_fails_the_bundle() {
        let (sk, tk) = key(12);
        let (bytes, _) = bundle(&sk, &tk.did);
        let trusted = std::slice::from_ref(&tk);
        let policy = Policy::default();
        assert!(verify_bundle(&bytes, trusted, Some(&tree()), &policy).ok);
        for i in 0..bytes.len() {
            for mask in [0x01u8, 0x20, 0x80] {
                let mut bad = bytes.clone();
                bad[i] ^= mask;
                let r = verify_bundle(&bad, trusted, Some(&tree()), &policy);
                assert!(!r.ok, "flipping byte {i} with {mask:#04x} still verified");
            }
        }
    }

    #[test]
    fn a_trusted_signer_cannot_claim_what_no_witness_shows() {
        let (sk, tk) = key(13);
        let (_, p) = bundle(&sk, &tk.did);
        let trusted = std::slice::from_ref(&tk);
        let mut lie = p.clone();
        lie.claims[1].status = Status::Green;
        let r = verify_bundle(
            &resign(&sk, &tk.did, lie),
            trusted,
            None,
            &Policy::default(),
        );
        assert!(!r.ok);
        assert!(
            r.failures.iter().any(|f| f.contains("do not follow")),
            "{:?}",
            r.failures
        );
        let mut dropped = p.clone();
        dropped.witnesses.pop();
        let r = verify_bundle(
            &resign(&sk, &tk.did, dropped),
            trusted,
            None,
            &Policy::default(),
        );
        assert!(
            !r.ok,
            "dropping the newest witness changes the claim it supports"
        );
        let mut moved = p;
        moved.tree = "c3".repeat(20);
        let r = verify_bundle(
            &resign(&sk, &tk.did, moved),
            trusted,
            None,
            &Policy::default(),
        );
        assert!(
            r.failures.iter().any(|f| f.contains("not bound")),
            "{:?}",
            r.failures
        );
    }

    #[test]
    fn an_untrusted_key_a_wrong_tree_and_the_policy_each_fail() {
        let (sk, tk) = key(14);
        let (_, other) = key(15);
        let (bytes, p) = bundle(&sk, &tk.did);
        let trusted = std::slice::from_ref(&tk);
        assert!(!verify_bundle(&bytes, &[other], None, &Policy::default()).ok);
        assert!(!verify_bundle(&bytes, &[], None, &Policy::default()).ok);
        assert!(!verify_bundle(&bytes, trusted, Some(&"e4".repeat(20)), &Policy::default()).ok);
        let require = Policy {
            require: vec!["pnpm test".into()],
            strict: false,
        };
        let r = verify_bundle(&bytes, trusted, None, &require);
        assert!(
            r.failures
                .iter()
                .any(|f| f.contains("required `pnpm test` is unwitnessed")),
            "{:?}",
            r.failures
        );
        let strict = Policy {
            require: vec![],
            strict: true,
        };
        assert!(!verify_bundle(&bytes, trusted, None, &strict).ok);
        let mut red_last = p;
        red_last.witnesses.reverse();
        let newest_red = record("cargo test", Some(1), &tree(), "2026-10-29T08:00:00.000Z");
        red_last.witnesses.push(WitnessEntry {
            memory: "mem-00000000000000a3".into(),
            strata_digest: None,
            envelope: newest_red.seal(&tk.did, |m| sk.sign(m).to_bytes()),
        });
        let mut records: Vec<(String, WitnessRecord)> = red_last
            .witnesses
            .iter()
            .map(|w| {
                (
                    w.memory.clone(),
                    WitnessRecord::open(&w.envelope, trusted).unwrap().0,
                )
            })
            .collect();
        records.sort_by(|a, b| a.0.cmp(&b.0));
        red_last.claims = claims::derive(&tree(), &records, &red_last.allowlist);
        let r = verify_bundle(
            &resign(&sk, &tk.did, red_last),
            trusted,
            None,
            &Policy::default(),
        );
        assert!(
            r.failures.iter().any(|f| f.contains("`cargo test` is red")),
            "{:?}",
            r.failures
        );
    }
}
