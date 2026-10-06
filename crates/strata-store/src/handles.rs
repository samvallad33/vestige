//! Canonical handles for recorded structure.
//!
//! A handle names a row the log already admitted: an anchor path, a
//! repository-qualified file, or a hunk span. Encoding is byte-exact. Nothing
//! here case-folds, splits identifiers, or reads node content.

use crate::types::SourceKey;

/// RFC 3986 unreserved bytes: ALPHA / DIGIT / "-" / "." / "_" / "~".
fn unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

/// Percent-encode every byte that is not RFC 3986 unreserved.
///
/// The hex alphabet is uppercase, so the same input always encodes the same
/// way. `/`, `:`, and `#` are encoded, which is what lets a repository
/// identity that contains them round-trip through one path separator.
pub fn percent_encode(raw: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(raw.len());
    for byte in raw.bytes() {
        if unreserved(byte) {
            out.push(byte as char);
        } else {
            out.push('%');
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    out
}

fn hex_val(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Inverse of [`percent_encode`]. `None` when a `%` escape is truncated,
/// not hex, or the decoded bytes are not UTF-8.
pub fn percent_decode(raw: &str) -> Option<String> {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return None;
            }
            let hi = hex_val(bytes[index + 1])?;
            let lo = hex_val(bytes[index + 2])?;
            out.push((hi << 4) | lo);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Repository-qualified file handle: `file://` + encoded repo + `/` + encoded path.
///
/// The only raw `/` is the separator between the repository identity and the
/// repository-relative path. Two repositories that share a relative path do
/// not share this handle.
pub fn qualified_file_handle(repo: &str, path: &str) -> String {
    format!("file://{}/{}", percent_encode(repo), percent_encode(path))
}

/// Split a handle from [`qualified_file_handle`]. `None` for any other text,
/// including a bare `file:<path>`.
pub fn parse_qualified_file_handle(handle: &str) -> Option<(String, String)> {
    let rest = handle.strip_prefix("file://")?;
    let (repo, path) = rest.split_once('/')?;
    Some((percent_decode(repo)?, percent_decode(path)?))
}

/// Stable id of one hunk anchor.
///
/// Derived from the commit source key, the repository-relative file, and the
/// new-side span. A re-run of the same commit produces the same id, so the
/// anchor row is replaced in place instead of duplicated. The id does not
/// embed the path with a delimiter, so a `:` or `#` in the path cannot shift
/// the span.
pub fn hunk_anchor_id(source: &SourceKey, file: &str, start: u32, len: u32) -> String {
    let mut material = Vec::new();
    for part in [
        source.system.as_str(),
        source.project.as_str(),
        source.id.as_str(),
        file,
    ] {
        let bytes = part.as_bytes();
        material.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        material.extend_from_slice(bytes);
    }
    material.extend_from_slice(&start.to_le_bytes());
    material.extend_from_slice(&len.to_le_bytes());
    let digest = blake3::derive_key("vestige-hunk-anchor-v1", &material);
    format!("hunk-{}", hex_encode(&digest))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_file_handle_round_trips_slash_colon_and_hash() {
        let repos = [
            "github.com/a/b",
            "git@host:port/a",
            "weird#repo",
            "a/b:c#d",
            "scheme:extra/name",
        ];
        let paths = [
            "src/a.rs",
            "dir/file#name.rs",
            "a b.rs",
            "C:/odd",
            "src/A.rs",
        ];
        for repo in repos {
            for path in paths {
                let handle = qualified_file_handle(repo, path);
                let (got_repo, got_path) = parse_qualified_file_handle(&handle)
                    .unwrap_or_else(|| panic!("parse {handle}"));
                assert_eq!(got_repo, repo, "{handle}");
                assert_eq!(got_path, path, "{handle}");
                let rest = handle.strip_prefix("file://").unwrap();
                let (encoded_repo, encoded_path) = rest.split_once('/').unwrap();
                assert!(!encoded_repo.contains('/'), "{handle}");
                assert!(!encoded_path.contains('/'), "{handle}");
            }
        }
        assert_ne!(
            qualified_file_handle("github.com/a/b", "src/a.rs"),
            qualified_file_handle("github.com/a/b", "src/A.rs"),
            "paths are exact bytes"
        );
        assert!(parse_qualified_file_handle("file:src/a.rs").is_none());
        assert!(parse_qualified_file_handle("src/a.rs").is_none());
    }

    #[test]
    fn hunk_anchor_id_depends_on_source_file_and_span_only() {
        let source = SourceKey {
            system: "git".into(),
            project: "github.com/a/b".into(),
            id: "github.com/a/b#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
        };
        let id = hunk_anchor_id(&source, "src/a.rs", 10, 3);
        assert_eq!(id, hunk_anchor_id(&source, "src/a.rs", 10, 3));
        assert_ne!(id, hunk_anchor_id(&source, "src/a.rs", 11, 3));
        assert_ne!(id, hunk_anchor_id(&source, "src/a:b.rs", 10, 3));
        assert!(id.starts_with("hunk-"));
        assert!(!id.contains("src/a.rs"));
    }
}
