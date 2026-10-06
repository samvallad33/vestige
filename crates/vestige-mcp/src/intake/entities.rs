//! Entity extraction for the ingest proof path (INGEST V5, Lane C).
//!
//! Pure function of bytes: no clocks, no RNG, no network, no storage, no
//! globals. Output is version-pinned by [`EXTRACTOR_VERSION`]; any change to
//! scanning behavior requires a new version string.
//!
//! [`extract_typed_spans`] runs six hand-rolled scanners over char
//! boundaries (byte offsets come from `char_indices`, so a span NEVER splits
//! a multibyte character):
//!
//! * **Url**: `https?://` then a run of non-whitespace, with trailing
//!   punctuation `.,);]}'">` trimmed. A Url suppresses every candidate span
//!   that falls inside it (kind wins over FilePath inside a Url).
//! * **Email**: RFC-lite `local@domain` — local = 1+ chars of
//!   `[A-Za-z0-9._%+-]`, domain = a run of `[A-Za-z0-9.-]` (a domain cannot
//!   end with `.`, trailing dots are trimmed before validation) that ends in
//!   `.` + a TLD of 2+ ASCII letters.
//! * **CommitSha**: a maximal hex run of 7..=40 chars, boundary-delimited
//!   (the chars on each side of a maximal run are not hex by construction),
//!   containing at least one a-f/A-F letter (pure numbers are rejected), and
//!   not preceded by `#`.
//! * **IssueRef**: `[A-Z][A-Z0-9]{1,}-\d+` (e.g. `GH-42`), or `#\d+` where
//!   the `#` is not preceded by another `#` or an ASCII alnum, or
//!   `[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+` (`owner/repo#123`; leading `/`
//!   separators of the maximal run are trimmed, the run must contain a `/`
//!   and must not end with `/`).
//! * **Version**: `v?\d+\.\d+(\.\d+)*` with ASCII word boundaries on both
//!   ends (a word char is `[A-Za-z0-9_]`). Bare form only — no leading `@`.
//! * **FilePath**: a whitespace-delimited token with byte length >= 3, no
//!   whitespace, containing `/` AND a `.` in the final (post-last-`/`)
//!   segment, whose first char is an ASCII alnum or one of `.`, `~`, `/`;
//!   tokens that look like a Url (contain `://`) or like a whole-token
//!   Version are rejected (checker order Url > Email > CommitSha > IssueRef
//!   > Version > FilePath). Trailing punctuation is NOT trimmed from paths —
//!   > only the Url scanner trims.
//!
//! Selection: all candidates are merged and sorted by
//! `(byte_start asc, length desc, kind precedence asc)`, then swept
//! greedily so the kept spans are non-overlapping — leftmost wins, and at
//! the same start the longest span wins; kind precedence
//! `Url > Email > CommitSha > IssueRef > Version > FilePath` only breaks
//! exact same-start-and-length ties. The result is sorted by `byte_start`
//! and capped at 64 spans.

pub const EXTRACTOR_VERSION: &str = "hand-scanners-v1";

/// Maximum number of spans [`extract_typed_spans`] may return.
const MAX_SPANS: usize = 64;

/// Trailing punctuation trimmed from Url spans (spec set `.,);]}'">`).
const URL_TRAILING_PUNCT: [char; 9] = ['.', ',', ')', ';', ']', '}', '\'', '"', '>'];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityKind {
    CommitSha,
    Url,
    FilePath,
    IssueRef,
    Email,
    Version,
}

impl EntityKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityKind::CommitSha => "CommitSha",
            EntityKind::Url => "Url",
            EntityKind::FilePath => "FilePath",
            EntityKind::IssueRef => "IssueRef",
            EntityKind::Email => "Email",
            EntityKind::Version => "Version",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntitySpan {
    pub kind: EntityKind,
    pub surface: String,
    pub byte_start: usize,
    pub byte_end: usize,
}

/// Extract typed entity spans from `content`.
///
/// Deterministic, pure, sorted by `byte_start`, non-overlapping
/// (longest-leftmost wins), capped at 64 spans. All byte offsets sit on
/// char boundaries.
pub fn extract_typed_spans(content: &str) -> Vec<EntitySpan> {
    let chars: Vec<(usize, char)> = content.char_indices().collect();

    let mut candidates: Vec<EntitySpan> = Vec::new();
    scan_urls(content, &chars, &mut candidates);
    scan_emails(content, &chars, &mut candidates);
    scan_commit_shas(content, &chars, &mut candidates);
    scan_issue_refs(content, &chars, &mut candidates);
    scan_versions(content, &chars, &mut candidates);
    scan_file_paths(content, &chars, &mut candidates);

    // Longest-leftmost merge: leftmost start first; at the same start the
    // longest candidate first; exact ties broken by kind precedence
    // (Url > Email > CommitSha > IssueRef > Version > FilePath).
    candidates.sort_by(|a, b| {
        a.byte_start
            .cmp(&b.byte_start)
            .then((b.byte_end - b.byte_start).cmp(&(a.byte_end - a.byte_start)))
            .then(kind_rank(&a.kind).cmp(&kind_rank(&b.kind)))
    });

    // Greedy sweep: because candidates are ordered by start, a candidate
    // only ever conflicts with the last kept span (starts are monotone and
    // kept spans are pairwise non-overlapping).
    let mut selected: Vec<EntitySpan> = Vec::new();
    for span in candidates {
        if let Some(last) = selected.last()
            && span.byte_start < last.byte_end
        {
            continue; // overlaps an already-selected leftmost-longest span
        }
        selected.push(span);
        if selected.len() >= MAX_SPANS {
            break;
        }
    }
    // `selected` inherits the (byte_start asc) order of the sorted sweep.
    selected
}

/// Precedence rank, lower = stronger. Only breaks exact ties (same start,
/// same length): Url > Email > CommitSha > IssueRef > Version > FilePath.
fn kind_rank(k: &EntityKind) -> u8 {
    match k {
        EntityKind::Url => 0,
        EntityKind::Email => 1,
        EntityKind::CommitSha => 2,
        EntityKind::IssueRef => 3,
        EntityKind::Version => 4,
        EntityKind::FilePath => 5,
    }
}

/// Byte offset just past the char at index `idx - 1` (i.e. `idx` chars in).
fn char_end(chars: &[(usize, char)], idx: usize) -> usize {
    let (offset, c) = chars[idx - 1];
    offset + c.len_utf8()
}

fn push_span(
    out: &mut Vec<EntitySpan>,
    kind: EntityKind,
    content: &str,
    byte_start: usize,
    byte_end: usize,
) {
    out.push(EntitySpan {
        kind,
        surface: content[byte_start..byte_end].to_string(),
        byte_start,
        byte_end,
    });
}

fn is_ascii_hex(c: char) -> bool {
    c.is_ascii_hexdigit()
}

/// Word char for Version boundary checks: `[A-Za-z0-9_]` (ASCII, regex \w).
fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Local-part class for the RFC-lite Email scanner.
fn is_local_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-')
}

/// Domain class for the RFC-lite Email scanner.
fn is_domain_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '.' || c == '-'
}

/// Key class for IssueRef form 1: `[A-Z0-9]`.
fn is_issue_key_char(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit()
}

/// Segment class for IssueRef form 3: `[A-Za-z0-9_.-]` plus the bridging
/// `/` separator itself, so the maximal run can span `owner/repo`.
fn is_slash_class_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '/')
}

/// Url scanner: `https?://` + non-whitespace run, trailing punctuation
/// (spec set) trimmed. The scan resumes after the untrimmed run, so nothing
/// inside a Url can start another Url.
fn scan_urls(content: &str, chars: &[(usize, char)], out: &mut Vec<EntitySpan>) {
    let mut i = 0;
    while i < chars.len() {
        let (start, c) = chars[i];
        if c == 'h' {
            let rest = &content[start..];
            let scheme_chars = if rest.starts_with("https://") {
                8
            } else if rest.starts_with("http://") {
                7
            } else {
                i += 1;
                continue;
            };
            // Non-whitespace run (over char boundaries).
            let mut j = i + scheme_chars;
            while j < chars.len() && !chars[j].1.is_whitespace() {
                j += 1;
            }
            // Trim trailing punctuation, never below the scheme itself.
            let mut end_idx = j;
            while end_idx > i + scheme_chars && URL_TRAILING_PUNCT.contains(&chars[end_idx - 1].1) {
                end_idx -= 1;
            }
            let byte_end = char_end(chars, end_idx);
            push_span(out, EntityKind::Url, content, start, byte_end);
            i = j;
        } else {
            i += 1;
        }
    }
}

/// Email scanner, anchored on each `@`: maximal local run to the left,
/// maximal domain run to the right, then RFC-lite validation of the domain.
fn scan_emails(content: &str, chars: &[(usize, char)], out: &mut Vec<EntitySpan>) {
    for i in 0..chars.len() {
        if chars[i].1 != '@' {
            continue;
        }
        // Local part: maximal run of local-class chars to the left.
        let mut s = i;
        while s > 0 && is_local_char(chars[s - 1].1) {
            s -= 1;
        }
        if s == i {
            continue; // empty local part
        }
        // Domain: maximal run of domain-class chars to the right.
        let mut e = i + 1;
        while e < chars.len() && is_domain_char(chars[e].1) {
            e += 1;
        }
        // A domain cannot end with '.'; trim trailing dots before validating.
        while e > i + 1 && chars[e - 1].1 == '.' {
            e -= 1;
        }
        if e == i + 1 {
            continue; // empty domain
        }
        let dom_start = chars[i + 1].0;
        let dom_end = char_end(chars, e);
        let domain = &content[dom_start..dom_end];
        let valid_tld = match domain.rfind('.') {
            Some(dot) => {
                let tld = &domain[dot + 1..];
                tld.chars().count() >= 2 && tld.chars().all(|c| c.is_ascii_alphabetic())
            }
            None => false,
        };
        if !valid_tld {
            continue;
        }
        push_span(out, EntityKind::Email, content, chars[s].0, dom_end);
    }
}

/// CommitSha scanner: maximal hex runs, 7..=40 long, containing at least one
/// a-f/A-F letter, not immediately preceded by `#`.
fn scan_commit_shas(content: &str, chars: &[(usize, char)], out: &mut Vec<EntitySpan>) {
    let mut i = 0;
    while i < chars.len() {
        if !is_ascii_hex(chars[i].1) {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < chars.len() && is_ascii_hex(chars[j].1) {
            j += 1;
        }
        // Maximal hex run occupies char indices [i, j).
        let len = j - i;
        let prev_is_hash = i > 0 && chars[i - 1].1 == '#';
        let has_hex_letter = chars[i..j].iter().any(|&(_, c)| c.is_ascii_alphabetic());
        if (7..=40).contains(&len) && has_hex_letter && !prev_is_hash {
            push_span(
                out,
                EntityKind::CommitSha,
                content,
                chars[i].0,
                char_end(chars, j),
            );
        }
        i = j;
    }
}

/// IssueRef scanner: form 1 (`[A-Z][A-Z0-9]{1,}-\d+`, anchored on `-`) and
/// forms 2+3 (`#\d+` and `owner/repo#\d+`, anchored on `#`).
fn scan_issue_refs(content: &str, chars: &[(usize, char)], out: &mut Vec<EntitySpan>) {
    // Form 1: uppercase-key + '-' + digits.
    for i in 0..chars.len() {
        if chars[i].1 != '-' {
            continue;
        }
        // Maximal [A-Z0-9] run ending just before the '-'.
        let mut s = i;
        while s > 0 && is_issue_key_char(chars[s - 1].1) {
            s -= 1;
        }
        if s == i {
            continue;
        }
        // The pattern starts at the first uppercase letter of the run (a
        // leading digit run like "12PROJ" still yields "PROJ-...").
        let mut k = s;
        while k < i && !chars[k].1.is_ascii_uppercase() {
            k += 1;
        }
        if k == i || i - k < 2 {
            continue; // no leading [A-Z], or key shorter than [A-Z][A-Z0-9]{1,}
        }
        let mut d = i + 1;
        while d < chars.len() && chars[d].1.is_ascii_digit() {
            d += 1;
        }
        if d == i + 1 {
            continue; // no digits
        }
        push_span(
            out,
            EntityKind::IssueRef,
            content,
            chars[k].0,
            char_end(chars, d),
        );
    }

    // Forms 2 and 3, anchored on '#'.
    for i in 0..chars.len() {
        if chars[i].1 != '#' {
            continue;
        }
        let mut d = i + 1;
        while d < chars.len() && chars[d].1.is_ascii_digit() {
            d += 1;
        }
        if d == i + 1 {
            continue; // '#' not followed by digits
        }
        // Maximal [A-Za-z0-9_.-] run immediately before the '#'.
        let mut s = i;
        while s > 0 && is_slash_class_char(chars[s - 1].1) {
            s -= 1;
        }
        let left = &content[chars[s].0..chars[i].0];
        let trimmed = left.trim_start_matches('/');
        if trimmed.contains('/') && !trimmed.ends_with('/') {
            // Form 3: owner/repo#123 — span starts after leading '/' chars.
            let byte_start = chars[i].0 - trimmed.len();
            push_span(
                out,
                EntityKind::IssueRef,
                content,
                byte_start,
                char_end(chars, d),
            );
        } else {
            // Form 2: bare #123 — '#' must not be preceded by '#' or alnum.
            let prev_ok = match if i > 0 { Some(chars[i - 1].1) } else { None } {
                Some(c) => c != '#' && !c.is_ascii_alphanumeric(),
                None => true,
            };
            if prev_ok {
                push_span(
                    out,
                    EntityKind::IssueRef,
                    content,
                    chars[i].0,
                    char_end(chars, d),
                );
            }
        }
    }
}

/// Version scanner: `v?\d+\.\d+(\.\d+)*` with ASCII word boundaries on both
/// ends. Bare only (no leading `@` form).
fn scan_versions(content: &str, chars: &[(usize, char)], out: &mut Vec<EntitySpan>) {
    for i in 0..chars.len() {
        let c = chars[i].1;
        let v_start = c == 'v';
        if !(v_start || c.is_ascii_digit()) {
            continue;
        }
        if v_start && (i + 1 >= chars.len() || !chars[i + 1].1.is_ascii_digit()) {
            continue; // a leading 'v' must be followed by a digit
        }
        if i > 0 && is_word_char(chars[i - 1].1) {
            continue; // leading word boundary
        }
        let mut j = i;
        if chars[j].1 == 'v' {
            j += 1;
        }
        while j < chars.len() && chars[j].1.is_ascii_digit() {
            j += 1;
        }
        if j >= chars.len() || chars[j].1 != '.' {
            continue;
        }
        j += 1;
        let second = j;
        while j < chars.len() && chars[j].1.is_ascii_digit() {
            j += 1;
        }
        if j == second {
            continue; // '\d+\.' needs digits after the first dot
        }
        // Optional further '.' digits groups; a trailing '.' without digits
        // is left outside the span and still satisfies the boundary.
        loop {
            if j < chars.len() && chars[j].1 == '.' {
                let mut k = j + 1;
                while k < chars.len() && chars[k].1.is_ascii_digit() {
                    k += 1;
                }
                if k > j + 1 {
                    j = k;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        if j < chars.len() && is_word_char(chars[j].1) {
            continue; // trailing word boundary
        }
        push_span(
            out,
            EntityKind::Version,
            content,
            chars[i].0,
            char_end(chars, j),
        );
    }
}

/// FilePath scanner: whitespace-delimited tokens passing the path rules.
fn scan_file_paths(content: &str, chars: &[(usize, char)], out: &mut Vec<EntitySpan>) {
    let mut i = 0;
    while i < chars.len() {
        if chars[i].1.is_whitespace() {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && !chars[j].1.is_whitespace() {
            j += 1;
        }
        let byte_start = chars[i].0;
        let byte_end = char_end(chars, j);
        let token = &content[byte_start..byte_end];
        if is_file_path_token(token) {
            push_span(out, EntityKind::FilePath, content, byte_start, byte_end);
        }
        i = j;
    }
}

/// FilePath token rules (checker order Url > Email > CommitSha > IssueRef >
/// Version > FilePath is applied per token: a token that looks like a Url or
/// a whole-token Version is rejected here; the other kinds are separated by
/// the longest-leftmost merge in [`extract_typed_spans`]).
fn is_file_path_token(token: &str) -> bool {
    // Byte length >= 3 (Rust `str::len` semantics; ASCII paths unaffected).
    if token.len() < 3 {
        return false;
    }
    let first = match token.chars().next() {
        Some(c) => c,
        None => return false,
    };
    if !(first.is_ascii_alphanumeric() || first == '.' || first == '~' || first == '/') {
        return false;
    }
    // Reject tokens that look like a Url.
    if token.contains("://") {
        return false;
    }
    // Reject tokens that are exactly a Version.
    if is_version_token(token) {
        return false;
    }
    let Some(last_slash) = token.rfind('/') else {
        return false; // must contain '/'
    };
    // '.' must appear in the final segment.
    token[last_slash + 1..].contains('.')
}

/// Whole-token check against the Version shape `v?\d+\.\d+(\.\d+)*`.
fn is_version_token(token: &str) -> bool {
    let body = token.strip_prefix('v').unwrap_or(token);
    let mut parts = body.split('.');
    let first = parts.next().unwrap_or("");
    if first.is_empty() || !first.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let mut count = 1usize;
    for part in parts {
        if part.is_empty() || !part.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        count += 1;
    }
    count >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(input: &str) -> Vec<EntitySpan> {
        extract_typed_spans(input)
    }

    #[test]
    fn extractor_version_is_pinned() {
        assert_eq!(EXTRACTOR_VERSION, "hand-scanners-v1");
    }

    #[test]
    fn kind_strings_are_pinned() {
        assert_eq!(EntityKind::CommitSha.as_str(), "CommitSha");
        assert_eq!(EntityKind::Url.as_str(), "Url");
        assert_eq!(EntityKind::FilePath.as_str(), "FilePath");
        assert_eq!(EntityKind::IssueRef.as_str(), "IssueRef");
        assert_eq!(EntityKind::Email.as_str(), "Email");
        assert_eq!(EntityKind::Version.as_str(), "Version");
    }

    #[test]
    fn empty_input_yields_no_spans() {
        for input in ["", "   ", "\n\t\r", "\u{3000}\u{3000}"] {
            assert!(spans(input).is_empty(), "input {input:?}");
        }
    }

    // Table-driven: at least one positive case per EntityKind, with byte
    // offsets pinned to the surface's true position in the input.
    #[test]
    fn every_kind_is_extracted() {
        let cases: Vec<(&str, &str, EntityKind)> = vec![
            (
                "see https://example.com/x.py now",
                "https://example.com/x.py",
                EntityKind::Url,
            ),
            (
                "mail bob.smith+filter@sub.example.io today",
                "bob.smith+filter@sub.example.io",
                EntityKind::Email,
            ),
            ("commit a1b2c3d landed", "a1b2c3d", EntityKind::CommitSha),
            ("fixed GH-42 today", "GH-42", EntityKind::IssueRef),
            ("see #123 please", "#123", EntityKind::IssueRef),
            (
                "owner/repo#123 done",
                "owner/repo#123",
                EntityKind::IssueRef,
            ),
            ("bump v2.1.0 now", "v2.1.0", EntityKind::Version),
            ("rel 1.2.3 out", "1.2.3", EntityKind::Version),
            (
                "edit src/store.py now",
                "src/store.py",
                EntityKind::FilePath,
            ),
            (
                "edit /abs/path.txt now",
                "/abs/path.txt",
                EntityKind::FilePath,
            ),
            ("edit ./rel.js now", "./rel.js", EntityKind::FilePath),
            ("edit ~/cfg/x.yml now", "~/cfg/x.yml", EntityKind::FilePath),
        ];
        for (input, surface, kind) in cases {
            let got = spans(input);
            assert_eq!(got.len(), 1, "input {input:?} got {got:?}");
            assert_eq!(got[0].kind, kind, "input {input:?}");
            assert_eq!(got[0].surface, surface, "input {input:?}");
            assert_eq!(got[0].kind.as_str(), kind.as_str());
            let at = input.find(surface).expect("surface present in input");
            assert_eq!(got[0].byte_start, at, "input {input:?}");
            assert_eq!(got[0].byte_end, at + surface.len(), "input {input:?}");
            assert_eq!(&input[got[0].byte_start..got[0].byte_end], surface);
            assert!(input.is_char_boundary(got[0].byte_start));
            assert!(input.is_char_boundary(got[0].byte_end));
        }
    }

    #[test]
    fn url_trailing_punctuation_is_trimmed() {
        let got = spans("see https://example.com/a.py), and http://x.io.");
        assert_eq!(got.len(), 2, "got {got:?}");
        assert_eq!(got[0].surface, "https://example.com/a.py");
        assert_eq!(got[1].surface, "http://x.io");
    }

    #[test]
    fn commit_sha_rules() {
        // Pure numbers are rejected (must contain an a-f letter).
        assert!(spans("num 1234567 end").is_empty());
        // Length bounds: 6 too short, 7 ok, 40 ok, 41 too long.
        assert!(spans("hex abc123 end").is_empty());
        assert_eq!(spans("hex abc1234 end").len(), 1);
        let sha40 = "a".repeat(39) + "0"; // 40 hex chars, has letters
        assert_eq!(spans(&format!("x {sha40} y")).len(), 1);
        let sha41 = "a".repeat(40) + "1"; // 41 hex chars
        assert!(spans(&format!("x {sha41} y")).is_empty());
        // Not preceded by '#'.
        assert!(spans("tag #abcdef1 end").is_empty());
        // Boundary-delimited: one maximal hex run is one span, not two.
        let got = spans("run a1b2c3d9e0f1 end");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "a1b2c3d9e0f1");
        assert_eq!(got[0].kind, EntityKind::CommitSha);
        // Uppercase hex letters count as letters too.
        let got = spans("sha ABCDEF12 end");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "ABCDEF12");
        // A non-hex, non-'#' prefix is fine.
        assert_eq!(spans("ref xa1b2c3d end").len(), 1);
    }

    #[test]
    fn issue_ref_rules() {
        // Form 1 key must be [A-Z][A-Z0-9]{1,}: "A-1" too short.
        assert!(spans("ref A-1 end").is_empty());
        // Form 1 accepts digits inside the key.
        let got = spans("ref PR0J-12 end");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "PR0J-12");
        // Form 2: '#' not preceded by another '#' or alnum.
        assert!(spans("ref ##12 end").is_empty());
        assert!(spans("ref abc#12 end").is_empty());
        assert_eq!(spans("ref (#12) end").len(), 1);
        // Form 2 vs form 3: a '/' in the left run upgrades to owner/repo#N.
        let got = spans("owner/repo#123 done");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "owner/repo#123");
        // A trailing '/' in the left run is not form 3; falls back to #12
        // ('/' before '#' is allowed for the bare form).
        let got = spans("ref a/#12 end");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "#12");
        // Leading '/' separators of the run are trimmed for form 3.
        let got = spans("ref /a/b#12 end");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "a/b#12");
    }

    #[test]
    fn version_rules() {
        // Trailing word char breaks the boundary.
        assert!(spans("no 1.2x here").is_empty());
        // Leading word char breaks the boundary.
        assert!(spans("no dev1.2 here").is_empty());
        assert!(spans("no a_1.2 here").is_empty());
        // Deep chains consume maximally.
        let got = spans("yes 1.2.3.4.5 out");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "1.2.3.4.5");
        assert_eq!(got[0].kind, EntityKind::Version);
        // Bare only — no '@' form; '@' simply is not part of the span.
        let got = spans("dep @9.1 here");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "9.1");
    }

    #[test]
    fn file_path_rules() {
        // No '/'.
        assert!(spans("no filename.txt here").is_empty());
        // No '.' in the final segment.
        assert!(spans("no src/ here").is_empty());
        // '.' only before the slash.
        assert!(spans("no a.b/c here").is_empty());
        // Byte length < 3.
        assert!(spans("no /x here").is_empty());
        // First char must be alnum or `.`, `~`, `/`.
        assert!(spans("no -src/x.y here").is_empty());
        // Spec-literal: trailing punctuation is NOT trimmed from paths.
        let got = spans("see src/store.py, ok");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].surface, "src/store.py,");
        // Whole-token version shapes are rejected as paths.
        assert!(is_version_token("1.2.3"));
        assert!(is_version_token("v1.2.3"));
        assert!(!is_version_token("1.2/3"));
        assert!(!is_version_token("1x.2"));
        assert!(!is_version_token("12"));
    }

    #[test]
    fn overlap_precedence_url_wins() {
        // A URL containing a path and a sha-looking hex: only the Url.
        let input = "see https://github.com/vestige/core/commit/deadbeef12345 end";
        let got = spans(input);
        assert_eq!(got.len(), 1, "got {got:?}");
        assert_eq!(got[0].kind, EntityKind::Url);
        assert_eq!(
            got[0].surface,
            "https://github.com/vestige/core/commit/deadbeef12345"
        );
    }

    #[test]
    fn overlap_leftmost_path_beats_inner_sha() {
        // Longest-leftmost: the path token starts first; the hex run inside
        // it overlaps and is dropped (checker order: FilePath token is not a
        // Url/Email/sha/issue/version as a whole).
        let got = spans("path src/a1b2c3d9.py end");
        assert_eq!(got.len(), 1, "got {got:?}");
        assert_eq!(got[0].kind, EntityKind::FilePath);
        assert_eq!(got[0].surface, "src/a1b2c3d9.py");
    }

    #[test]
    fn multibyte_offsets_stay_on_char_boundaries() {
        let input = "重要 fix in 🚒 src/store.py";
        let got = spans(input);
        assert_eq!(got.len(), 1, "got {got:?}");
        let s = &got[0];
        assert_eq!(s.kind, EntityKind::FilePath);
        assert_eq!(s.surface, "src/store.py");
        let at = input.find("src/store.py").unwrap();
        assert_eq!(s.byte_start, at);
        assert_eq!(s.byte_end, at + "src/store.py".len());
        assert!(input.is_char_boundary(s.byte_start));
        assert!(input.is_char_boundary(s.byte_end));
        assert_eq!(&input[s.byte_start..s.byte_end], "src/store.py");
    }

    #[test]
    fn multibyte_neighbors_do_not_break_offsets() {
        let input = "日本語GH-12です";
        let got = spans(input);
        assert_eq!(got.len(), 1, "got {got:?}");
        assert_eq!(got[0].surface, "GH-12");
        let at = input.find("GH-12").unwrap();
        assert_eq!(got[0].byte_start, at);
        assert_eq!(got[0].byte_end, at + "GH-12".len());
        assert!(input.is_char_boundary(got[0].byte_start));
        assert!(input.is_char_boundary(got[0].byte_end));
        assert_eq!(&input[got[0].byte_start..got[0].byte_end], "GH-12");
    }

    #[test]
    fn deterministic_double_run() {
        let input = "mixed GH-7 https://x.io/a.py bob@e.io v3.2 #9 src/x.rs q";
        let a = spans(input);
        let b = spans(input);
        assert_eq!(a, b);
        assert_eq!(a.len(), 6, "got {a:?}");
    }

    #[test]
    fn cap_at_sixty_four_spans() {
        let mut input = String::new();
        for i in 0..80u32 {
            input.push_str(&format!("a00000{i:02x} "));
        }
        let got = spans(&input);
        assert_eq!(got.len(), 64, "cap not applied: {}", got.len());
        assert!(got.iter().all(|s| s.kind == EntityKind::CommitSha));
        assert_eq!(got[0].surface, "a0000000");
        assert_eq!(got[63].surface, "a000003f");
        // Sorted by byte_start and pairwise non-overlapping.
        assert!(got.windows(2).all(|w| w[0].byte_end <= w[1].byte_start));
    }

    // Golden vector: pins EXTRACTOR_VERSION's exact output for one mixed
    // paragraph (kinds, surfaces and byte offsets).
    #[test]
    fn golden_mixed_paragraph() {
        assert_eq!(EXTRACTOR_VERSION, "hand-scanners-v1");
        let input = "see GH-42 and https://github.com/vestige/core/commit/deadbeef12345 in src/store.py plus #7, v2.1.0, a1b2c3d and bob@example.io";
        let got = spans(input);
        let expected_kinds = [
            EntityKind::IssueRef,
            EntityKind::Url,
            EntityKind::FilePath,
            EntityKind::IssueRef,
            EntityKind::Version,
            EntityKind::CommitSha,
            EntityKind::Email,
        ];
        let expected: Vec<EntitySpan> = vec![
            EntitySpan {
                kind: EntityKind::IssueRef,
                surface: "GH-42".into(),
                byte_start: 4,
                byte_end: 9,
            },
            EntitySpan {
                kind: EntityKind::Url,
                surface: "https://github.com/vestige/core/commit/deadbeef12345".into(),
                byte_start: 14,
                byte_end: 66,
            },
            EntitySpan {
                kind: EntityKind::FilePath,
                surface: "src/store.py".into(),
                byte_start: 70,
                byte_end: 82,
            },
            EntitySpan {
                kind: EntityKind::IssueRef,
                surface: "#7".into(),
                byte_start: 88,
                byte_end: 90,
            },
            EntitySpan {
                kind: EntityKind::Version,
                surface: "v2.1.0".into(),
                byte_start: 92,
                byte_end: 98,
            },
            EntitySpan {
                kind: EntityKind::CommitSha,
                surface: "a1b2c3d".into(),
                byte_start: 100,
                byte_end: 107,
            },
            EntitySpan {
                kind: EntityKind::Email,
                surface: "bob@example.io".into(),
                byte_start: 112,
                byte_end: 126,
            },
        ];
        assert_eq!(got.len(), expected_kinds.len(), "got {got:?}");
        assert_eq!(got, expected);
        // Cross-check every offset against the surface's true position.
        for s in &got {
            let at = input.find(&s.surface).unwrap();
            assert_eq!(s.byte_start, at);
            assert_eq!(s.byte_end, at + s.surface.len());
            assert_eq!(&input[s.byte_start..s.byte_end], s.surface);
        }
    }
}
