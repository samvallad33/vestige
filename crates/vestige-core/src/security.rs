//! Local secret-detection primitives for memory ingestion.
//!
//! Vestige is often handed transcripts, incident notes, and configuration
//! snippets.  Those are useful memories, but they are also a common route for
//! credentials to enter a long-lived local store.  This module deliberately
//! keeps detection local, dependency-free, and non-retentive: callers receive
//! a credential class and a short BLAKE3 fingerprint, never the matched value.

use std::fmt;

/// Controls whether a caller deliberately permits a detected credential to be
/// stored. The default is always [`SecretPolicy::Reject`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SecretPolicy {
    /// Refuse provider-specific credential shapes before any write side effect.
    #[default]
    Reject,
    /// Allow storage only when the immediate caller made an explicit choice.
    AllowExplicitly,
}

/// The confidence attached to a possible credential finding.
///
/// Only [`SecretConfidence::Blocking`] findings reject a write by default.
/// Entropy-only findings are returned by audits for human review, but are not
/// safe enough to reject automatically (UUIDs, hashes, and model identifiers
/// often look random too).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecretConfidence {
    /// A provider-specific credential shape with a low false-positive rate.
    Blocking,
    /// A suspicious value that needs review but must not block ordinary data.
    Review,
}

impl fmt::Display for SecretConfidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Blocking => write!(f, "blocking"),
            Self::Review => write!(f, "review"),
        }
    }
}

/// A provider-specific or audit-only secret classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecretKind {
    GoogleApiKey,
    GitHubToken,
    AwsAccessKeyId,
    SlackToken,
    PemPrivateKey,
    AzureClientSecret,
    HighEntropyCandidate,
}

impl SecretKind {
    /// Stable, human-readable name suitable for safe user-facing errors.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GoogleApiKey => "Google API key",
            Self::GitHubToken => "GitHub token",
            Self::AwsAccessKeyId => "AWS access key ID",
            Self::SlackToken => "Slack token",
            Self::PemPrivateKey => "PEM private key",
            Self::AzureClientSecret => "Azure client secret",
            Self::HighEntropyCandidate => "high-entropy credential candidate",
        }
    }
}

impl fmt::Display for SecretKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A safe description of a detected credential.
///
/// The matched value is intentionally not stored here, so propagating or
/// logging a finding cannot leak the credential it describes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SecretFinding {
    pub kind: SecretKind,
    pub confidence: SecretConfidence,
    pub fingerprint: String,
}

impl SecretFinding {
    /// Whether this finding rejects a normal write.
    pub fn blocks_ingestion(&self) -> bool {
        self.confidence == SecretConfidence::Blocking
    }
}

impl fmt::Display for SecretFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.kind, self.fingerprint)
    }
}

/// Scan content for known credential shapes and audit-only suspicious values.
///
/// This intentionally avoids returning matched strings.  Provider-specific
/// matches are blocking; the high-entropy fallback is review-only.
pub fn scan_secrets(content: &str) -> Vec<SecretFinding> {
    let mut findings = Vec::new();

    // Google API keys are `AIza` followed by 35 URL-safe base64 characters.
    for value in exact_prefixed_runs(content, "AIza", 35, is_urlsafe_key_char) {
        push_finding(
            &mut findings,
            SecretKind::GoogleApiKey,
            SecretConfidence::Blocking,
            value,
        );
    }

    // GitHub publishes stable token formats: classic tokens have 36 characters
    // after their prefix and fine-grained tokens have 82. Exact shapes plus
    // delimiters avoid treating documentation or identifier-like prose as a
    // credential.
    for prefix in ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"] {
        for value in exact_prefixed_runs(content, prefix, 36, is_alphanumeric) {
            push_finding(
                &mut findings,
                SecretKind::GitHubToken,
                SecretConfidence::Blocking,
                value,
            );
        }
    }
    for value in exact_prefixed_runs(content, "github_pat_", 82, is_alphanumeric_or_underscore) {
        push_finding(
            &mut findings,
            SecretKind::GitHubToken,
            SecretConfidence::Blocking,
            value,
        );
    }

    // AKIA is the long-lived IAM key prefix; ASIA is the temporary STS form.
    for prefix in ["AKIA", "ASIA", "ABIA", "ACCA", "A3T"] {
        for value in exact_prefixed_runs(content, prefix, 20 - prefix.len(), is_aws_key_char) {
            push_finding(
                &mut findings,
                SecretKind::AwsAccessKeyId,
                SecretConfidence::Blocking,
                value,
            );
        }
    }

    for value in slack_token_runs(content) {
        push_finding(
            &mut findings,
            SecretKind::SlackToken,
            SecretConfidence::Blocking,
            value,
        );
    }

    for value in pem_private_key_runs(content) {
        push_finding(
            &mut findings,
            SecretKind::PemPrivateKey,
            SecretConfidence::Blocking,
            value,
        );
    }

    // Azure AD client secrets use the distinctive `...<digit>Q~...` shape.
    // The documented/observed form has three URL-safe characters, a digit,
    // and then 31–34 more characters after `Q~`. This catches the reported
    // secret even when it was copied into an unlabelled transcript.
    for value in azure_client_secret_runs(content) {
        push_finding(
            &mut findings,
            SecretKind::AzureClientSecret,
            SecretConfidence::Blocking,
            value,
        );
    }

    scan_labelled_secret_values(content, &mut findings);
    findings
}

fn push_finding(
    findings: &mut Vec<SecretFinding>,
    kind: SecretKind,
    confidence: SecretConfidence,
    value: &str,
) {
    let digest = blake3::hash(value.as_bytes()).to_hex().to_string();
    let finding = SecretFinding {
        kind,
        confidence,
        // 128 bits is enough to correlate local audit hits without making
        // ordinary report output a feasible secret-recovery oracle.
        fingerprint: format!("blake3:{}", &digest[..32]),
    };
    if !findings.contains(&finding) {
        findings.push(finding);
    }
}

fn exact_prefixed_runs<'a>(
    content: &'a str,
    prefix: &str,
    suffix_len: usize,
    allowed: fn(u8) -> bool,
) -> Vec<&'a str> {
    let bytes = content.as_bytes();
    content
        .match_indices(prefix)
        .filter_map(|(start, _)| {
            let end = start + prefix.len() + suffix_len;
            (end <= bytes.len()
                && (start == 0 || !allowed(bytes[start - 1]))
                && bytes[start + prefix.len()..end]
                    .iter()
                    .all(|byte| allowed(*byte))
                && (end == bytes.len() || !allowed(bytes[end])))
            .then_some(&content[start..end])
        })
        .collect()
}

fn is_alphanumeric(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
}

fn is_alphanumeric_or_underscore(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn is_urlsafe_key_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')
}

fn is_aws_key_char(byte: u8) -> bool {
    byte.is_ascii_uppercase() || byte.is_ascii_digit()
}

fn is_slack_token_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'-'
}

fn slack_token_runs(content: &str) -> Vec<&str> {
    let bytes = content.as_bytes();
    let mut values = Vec::new();
    for prefix in ["xoxb-", "xoxp-", "xoxa-", "xoxr-", "xoxs-"] {
        for (start, _) in content.match_indices(prefix) {
            if start > 0 && is_slack_token_char(bytes[start - 1]) {
                continue;
            }
            let mut end = start + prefix.len();
            while end < bytes.len() && is_slack_token_char(bytes[end]) {
                end += 1;
            }
            let value = &content[start..end];
            let segments: Vec<&str> = value.split('-').collect();
            let structured = segments.len() >= 4
                && segments[1].len() >= 6
                && segments[1].bytes().all(|byte| byte.is_ascii_digit())
                && segments[2].len() >= 6
                && segments[2].bytes().all(|byte| byte.is_ascii_digit())
                && segments[3..]
                    .iter()
                    .any(|segment| segment.len() >= 16 && segment.bytes().all(is_alphanumeric));
            if structured && (end == bytes.len() || !is_slack_token_char(bytes[end])) {
                values.push(value);
            }
        }
    }
    values
}

fn pem_private_key_runs(content: &str) -> Vec<&str> {
    let mut values = Vec::new();
    for (header, footer) in [
        ("-----BEGIN PRIVATE KEY-----", "-----END PRIVATE KEY-----"),
        (
            "-----BEGIN RSA PRIVATE KEY-----",
            "-----END RSA PRIVATE KEY-----",
        ),
        (
            "-----BEGIN EC PRIVATE KEY-----",
            "-----END EC PRIVATE KEY-----",
        ),
        (
            "-----BEGIN OPENSSH PRIVATE KEY-----",
            "-----END OPENSSH PRIVATE KEY-----",
        ),
        (
            "-----BEGIN ENCRYPTED PRIVATE KEY-----",
            "-----END ENCRYPTED PRIVATE KEY-----",
        ),
    ] {
        for (start, _) in content.match_indices(header) {
            let tail = &content[start + header.len()..];
            let end = tail
                .find(footer)
                .map(|offset| start + header.len() + offset + footer.len())
                .unwrap_or(start + header.len());
            values.push(&content[start..end]);
        }
    }
    values
}

fn is_secret_value_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'~' | b'.' | b'_' | b'-')
}

fn azure_client_secret_runs(content: &str) -> Vec<&str> {
    let bytes = content.as_bytes();
    let mut values = Vec::new();

    for (marker_start, _) in content.match_indices("Q~") {
        // `abc8Q~...`: three allowed characters and a digit precede `Q~`.
        let Some(start) = marker_start.checked_sub(4) else {
            continue;
        };
        if !content.is_char_boundary(start)
            || !bytes[start..marker_start]
                .iter()
                .take(3)
                .all(|byte| is_secret_value_char(*byte))
            || !bytes[marker_start - 1].is_ascii_digit()
        {
            continue;
        }

        let mut end = marker_start + 2;
        while end < bytes.len() && is_secret_value_char(bytes[end]) {
            end += 1;
        }

        let value = &content[start..end];
        if (37..=40).contains(&value.len())
            && (start == 0 || !is_secret_value_char(bytes[start - 1]))
            && (end == bytes.len() || !is_secret_value_char(bytes[end]))
        {
            values.push(value);
        }
    }

    values
}

fn scan_labelled_secret_values(content: &str, findings: &mut Vec<SecretFinding>) {
    let lowercase = content.to_ascii_lowercase();
    for label in [
        "client_secret",
        "clientsecret",
        "client secret",
        "api_key",
        "apikey",
        "api key",
        "access_token",
        "accesstoken",
        "access token",
        "password",
    ] {
        let mut offset = 0;
        while let Some(found) = lowercase[offset..].find(label) {
            let label_end = offset + found + label.len();
            let bytes = content.as_bytes();
            let mut start = label_end;
            while start < bytes.len() && !is_secret_value_char(bytes[start]) {
                start += 1;
            }
            let mut end = start;
            while end < bytes.len() && is_secret_value_char(bytes[end]) {
                end += 1;
            }

            // A high-entropy value beside a credential label needs a human
            // audit, but does not block storage by itself.
            if let Some(value) = content.get(start..end)
                && value.len() >= 32
                && has_mixed_classes(value)
                && shannon_entropy(value) >= 3.5
            {
                push_finding(
                    findings,
                    SecretKind::HighEntropyCandidate,
                    SecretConfidence::Review,
                    value,
                );
            }

            offset = label_end;
        }
    }
}

fn has_mixed_classes(value: &str) -> bool {
    let mut lower = false;
    let mut upper = false;
    let mut digit = false;
    for byte in value.bytes() {
        lower |= byte.is_ascii_lowercase();
        upper |= byte.is_ascii_uppercase();
        digit |= byte.is_ascii_digit();
    }
    lower && upper && digit
}

fn shannon_entropy(value: &str) -> f64 {
    let mut counts = [0usize; 256];
    for byte in value.bytes() {
        counts[byte as usize] += 1;
    }
    let len = value.len() as f64;
    counts
        .into_iter()
        .filter(|count| *count > 0)
        .map(|count| {
            let probability = count as f64 / len;
            -probability * probability.log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_known_provider_credentials_without_retaining_them() {
        let github = format!("ghp_{}", "A".repeat(36));
        let google = format!("AIza{}", "A".repeat(35));
        let content = format!("github={github}\ngoogle={google}");

        let findings = scan_secrets(&content);
        assert_eq!(findings.len(), 2);
        assert!(findings.iter().all(SecretFinding::blocks_ingestion));
        assert!(
            findings
                .iter()
                .all(|finding| !format!("{finding:?}").contains(&github))
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.kind == SecretKind::GitHubToken)
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.kind == SecretKind::GoogleApiKey)
        );
    }

    #[test]
    fn classifies_labelled_high_entropy_values_for_review_only() {
        let value = "abCDeFgHiJkLmNoPqRsTuVwXyZ0123456789";
        let findings = scan_secrets(&format!("client_secret: {value}"));

        assert!(
            findings
                .iter()
                .any(|finding| finding.kind == SecretKind::HighEntropyCandidate)
        );
        assert!(findings.iter().all(|finding| !finding.blocks_ingestion()));
    }

    #[test]
    fn finds_unlabelled_azure_client_secret_shape() {
        let azure = format!("abc8Q~{}", "a".repeat(31));
        let findings = scan_secrets(&format!("copied credential: {azure}"));

        assert!(findings.iter().any(|finding| {
            finding.kind == SecretKind::AzureClientSecret && finding.blocks_ingestion()
        }));
        assert!(
            findings
                .iter()
                .all(|finding| !format!("{finding:?}").contains(&azure))
        );
    }

    #[test]
    fn requires_exact_structured_provider_shapes() {
        let github_lookalike = format!("github_pat_{}", "a".repeat(50));
        let slack_lookalike = format!("xoxb-{}", "a".repeat(24));
        let github = format!("ghp_{}", "A".repeat(36));
        let slack = format!("xoxb-123456-654321-{}", "a".repeat(24));
        let findings = scan_secrets(&format!(
            "examples: {github_lookalike} {slack_lookalike}; active: {github} {slack}"
        ));

        assert_eq!(
            findings
                .iter()
                .filter(|finding| finding.kind == SecretKind::GitHubToken)
                .count(),
            1
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.kind == SecretKind::SlackToken)
        );
    }

    #[test]
    fn generic_labelled_tilde_text_is_review_only() {
        let findings = scan_secrets("password rotation note: only~a~placeholder~value");
        assert!(findings.iter().all(|finding| !finding.blocks_ingestion()));
    }

    #[test]
    fn pem_fingerprints_cover_key_material_not_only_the_header() {
        let first = scan_secrets("-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----");
        let second = scan_secrets("-----BEGIN PRIVATE KEY-----\nBBBB\n-----END PRIVATE KEY-----");
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_ne!(first[0].fingerprint, second[0].fingerprint);
        assert_eq!(first[0].fingerprint.len(), "blake3:".len() + 32);
    }

    #[test]
    fn ordinary_configuration_text_is_not_flagged() {
        let findings = scan_secrets(
            "Set OPENAI_API_KEY in the shell and rotate credentials outside the repository.",
        );
        assert!(findings.is_empty());
    }
}
