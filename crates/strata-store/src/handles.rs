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

/// Cap on ambiguous candidates. Matches `vestige_core::storage::MAX_CANDIDATES`.
const MAX_CANDIDATES: usize = 20;

/// Which recorded table a structural handle hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureKind {
    /// Anchor `file_path` or a repository-qualified touched target.
    File,
    /// `path#symbol` on an anchor.
    Symbol,
    /// Git commit record.
    Commit,
    /// Run subject (`classname::name`).
    Test,
    /// Run id.
    Run,
    /// More than one table hit, so nothing was picked.
    Unknown,
}

/// One recorded row behind a structural hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructureProof {
    /// Anchor row and the effect that admitted it.
    Anchor {
        /// Memory the anchor belongs to.
        id: String,
        /// Anchor id.
        anchor_id: String,
        /// `eff-` receipt.
        receipt: String,
    },
    /// Commit record: creating frame plus its effect receipt.
    Commit {
        /// Commit memory id.
        id: String,
        /// Log seq of the creating frame.
        frame_seq: u64,
        /// Chain hash of that frame, lowercase hex.
        frame_hash: String,
        /// `eff-` receipt of the creating effect.
        receipt: String,
    },
    /// Run record and the effect that admitted it.
    Run {
        /// Run id.
        id: String,
        /// `eff-` receipt.
        receipt: String,
    },
    /// Typed edge and the data frame that admitted it.
    Edge {
        /// Source memory id.
        id: String,
        /// Edge source.
        source: String,
        /// Edge target.
        target: String,
        /// Vocabulary `link_type`.
        link_type: String,
        /// Commit sha recorded on the edge, when present.
        meta_sha: Option<String>,
        /// Log seq of the `SaveEdge` frame.
        frame_seq: u64,
    },
}

/// Resolution against anchors, touched edges, commit records, and runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureResolution {
    /// Table the hit came from.
    pub kind: StructureKind,
    /// Resolved ids. Empty when the caller must disambiguate.
    pub ids: Vec<String>,
    /// `false` for a unique commit-sha prefix.
    pub exact: bool,
    /// Kind-labeled alternatives. Capped at 20.
    pub candidates: Vec<(String, StructureKind)>,
    /// Recorded rows behind `ids`.
    pub proofs: Vec<StructureProof>,
    /// Set for a typed miss or a too-short commit prefix.
    pub handle_required: Option<String>,
}

#[derive(Clone)]
struct AnchorRow {
    id: String,
    node_id: String,
}

#[derive(Clone)]
struct QualEdge {
    handle: String,
    source: String,
    target: String,
    meta_sha: Option<String>,
}

struct Tables {
    files: std::collections::HashMap<String, Vec<AnchorRow>>,
    symbols: std::collections::HashMap<String, Vec<AnchorRow>>,
    symbol_paths: std::collections::HashMap<String, std::collections::BTreeSet<String>>,
    commits: std::collections::BTreeMap<String, Vec<String>>,
    qualified: std::collections::HashMap<String, Vec<QualEdge>>,
    qualified_by_path: std::collections::HashMap<String, Vec<QualEdge>>,
    tests: std::collections::HashMap<String, Vec<String>>,
    runs: std::collections::BTreeSet<String>,
}

struct Hit {
    kind: StructureKind,
    ids: Vec<String>,
    exact: bool,
    candidates: Vec<(String, StructureKind)>,
    proofs: Vec<StructureProof>,
    handle_required: Option<String>,
}

/// Resolve `query` from recorded structure only.
///
/// Node content is never read. A bare path resolves anchor rows for that
/// exact path and lists qualified handles as candidates; it never picks a
/// repository. A qualified `file://` handle resolves only its own edges.
pub fn resolve_structure(store: &crate::StrataStore, query: &str) -> Option<StructureResolution> {
    let query = query.trim();
    if query.is_empty() {
        return None;
    }
    let tables = tables_of(store);
    if let Some(resolution) = typed(store, &tables, query) {
        return Some(resolution);
    }
    let mut hits = Vec::new();
    if let Some(hit) = file_hit(store, &tables, query, FileMode::Bare) {
        hits.push(hit);
    }
    if let Some(hit) = symbol_hit(store, &tables, query) {
        hits.push(hit);
    }
    if let Some(hit) = commit_hit(store, &tables, query, false) {
        hits.push(hit);
    }
    if let Some(hit) = test_hit(store, &tables, query) {
        hits.push(hit);
    }
    if let Some(hit) = run_hit(store, &tables, query) {
        hits.push(hit);
    }
    finish(hits)
}

fn typed(store: &crate::StrataStore, tables: &Tables, query: &str) -> Option<StructureResolution> {
    if let Some(handle) = query.strip_prefix("file://") {
        if handle.contains('/') {
            return Some(one(file_hit(store, tables, query, FileMode::Qualified)));
        }
    }
    if let Some(rest) = query.strip_prefix("file:") {
        if rest.starts_with("//") {
            return None;
        }
        return Some(one(file_hit(store, tables, rest, FileMode::AnchorsOnly)));
    }
    if let Some(rest) = query.strip_prefix("sym:") {
        return Some(one(symbol_hit(store, tables, rest)));
    }
    if let Some(rest) = query.strip_prefix("commit:") {
        return Some(one(commit_hit(store, tables, rest, true)));
    }
    if let Some(rest) = query.strip_prefix("test:") {
        return Some(one(test_hit(store, tables, rest)));
    }
    if let Some(rest) = query.strip_prefix("run:") {
        return Some(one(run_hit(store, tables, rest)));
    }
    None
}

fn one(hit: Option<Hit>) -> StructureResolution {
    match hit {
        Some(hit) => StructureResolution {
            kind: hit.kind,
            ids: hit.ids,
            exact: hit.exact,
            candidates: cap(hit.candidates),
            proofs: hit.proofs,
            handle_required: hit.handle_required,
        },
        None => StructureResolution {
            kind: StructureKind::Unknown,
            ids: Vec::new(),
            exact: false,
            candidates: Vec::new(),
            proofs: Vec::new(),
            handle_required: None,
        },
    }
}

fn finish(hits: Vec<Hit>) -> Option<StructureResolution> {
    if hits.is_empty() {
        return None;
    }
    if hits.len() == 1 {
        return Some(one(hits.into_iter().next()));
    }
    let mut candidates = Vec::new();
    for hit in hits {
        if hit.ids.is_empty() {
            candidates.extend(hit.candidates);
        } else {
            for id in hit.ids {
                candidates.push((id, hit.kind));
            }
        }
    }
    Some(StructureResolution {
        kind: StructureKind::Unknown,
        ids: Vec::new(),
        exact: false,
        candidates: cap(candidates),
        proofs: Vec::new(),
        handle_required: None,
    })
}

fn cap(mut candidates: Vec<(String, StructureKind)>) -> Vec<(String, StructureKind)> {
    candidates.truncate(MAX_CANDIDATES);
    candidates
}

enum FileMode {
    Bare,
    AnchorsOnly,
    Qualified,
}

fn file_hit(
    store: &crate::StrataStore,
    tables: &Tables,
    query: &str,
    mode: FileMode,
) -> Option<Hit> {
    match mode {
        FileMode::Qualified => {
            let edges = tables.qualified.get(query)?;
            return Some(edge_hit(store, edges));
        }
        FileMode::AnchorsOnly => {
            let rows = tables.files.get(query)?;
            return Some(anchor_hit(
                store,
                StructureKind::File,
                rows,
                true,
                Vec::new(),
            ));
        }
        FileMode::Bare => {}
    }
    let anchors = tables.files.get(query);
    let qualified = tables.qualified_by_path.get(query);
    if anchors.is_none() && qualified.is_none() {
        return None;
    }
    let mut side = Vec::new();
    if let Some(edges) = qualified {
        for edge in edges {
            let candidate = (edge.handle.clone(), StructureKind::File);
            if !side.contains(&candidate) {
                side.push(candidate);
            }
        }
    }
    if let Some(rows) = anchors {
        return Some(anchor_hit(store, StructureKind::File, rows, true, side));
    }
    Some(Hit {
        kind: StructureKind::File,
        ids: Vec::new(),
        exact: false,
        candidates: side,
        proofs: Vec::new(),
        handle_required: None,
    })
}

fn symbol_hit(store: &crate::StrataStore, tables: &Tables, query: &str) -> Option<Hit> {
    if query.is_empty() {
        return None;
    }
    if query.contains('#') {
        let rows = tables.symbols.get(query)?;
        return Some(anchor_hit(
            store,
            StructureKind::Symbol,
            rows,
            true,
            Vec::new(),
        ));
    }
    let paths = tables.symbol_paths.get(query)?;
    if paths.len() == 1 {
        let path = paths.iter().next()?;
        let key = format!("{path}#{query}");
        let rows = tables.symbols.get(&key)?;
        return Some(anchor_hit(
            store,
            StructureKind::Symbol,
            rows,
            true,
            Vec::new(),
        ));
    }
    let candidates = paths
        .iter()
        .map(|path| (format!("{path}#{query}"), StructureKind::Symbol))
        .collect();
    Some(Hit {
        kind: StructureKind::Symbol,
        ids: Vec::new(),
        exact: false,
        candidates,
        proofs: Vec::new(),
        handle_required: None,
    })
}

fn commit_hit(
    store: &crate::StrataStore,
    tables: &Tables,
    query: &str,
    typed: bool,
) -> Option<Hit> {
    match sha_shape(query) {
        Some(ShaShape::Full) => {
            let ids = tables.commits.get(query)?.clone();
            Some(commit_ids(store, ids, true))
        }
        Some(ShaShape::Prefix) => {
            let matched: Vec<&String> = tables
                .commits
                .keys()
                .filter(|sha| sha.starts_with(query))
                .collect();
            if matched.is_empty() {
                return None;
            }
            if matched.len() == 1 {
                let ids = tables.commits.get(matched[0])?.clone();
                return Some(commit_ids(store, ids, false));
            }
            let mut candidates = Vec::new();
            for sha in matched {
                for id in tables.commits.get(sha)? {
                    candidates.push((id.clone(), StructureKind::Commit));
                }
            }
            Some(Hit {
                kind: StructureKind::Commit,
                ids: Vec::new(),
                exact: false,
                candidates,
                proofs: Vec::new(),
                handle_required: None,
            })
        }
        Some(ShaShape::TooShort) if typed => Some(Hit {
            kind: StructureKind::Commit,
            ids: Vec::new(),
            exact: false,
            candidates: Vec::new(),
            proofs: Vec::new(),
            handle_required: Some(format!(
                "commit sha prefix must be at least 7 hex characters; '{query}' is too short to resolve unambiguously"
            )),
        }),
        _ => None,
    }
}

fn test_hit(store: &crate::StrataStore, tables: &Tables, query: &str) -> Option<Hit> {
    if query.is_empty() {
        return None;
    }
    let ids = tables.tests.get(query)?.clone();
    Some(run_ids(store, StructureKind::Test, ids))
}

fn run_hit(store: &crate::StrataStore, tables: &Tables, query: &str) -> Option<Hit> {
    if query.is_empty() || !tables.runs.contains(query) {
        return None;
    }
    Some(run_ids(store, StructureKind::Run, vec![query.to_string()]))
}

fn anchor_hit(
    store: &crate::StrataStore,
    kind: StructureKind,
    rows: &[AnchorRow],
    exact: bool,
    candidates: Vec<(String, StructureKind)>,
) -> Hit {
    let mut ids = Vec::new();
    let mut proofs = Vec::new();
    for row in rows {
        if !ids.contains(&row.node_id) {
            ids.push(row.node_id.clone());
        }
        if let Some(proof) = store.latest_effect(&row.id).ok().flatten() {
            proofs.push(StructureProof::Anchor {
                id: row.node_id.clone(),
                anchor_id: row.id.clone(),
                receipt: crate::effect_receipt_id(proof.effect_seq),
            });
        }
    }
    Hit {
        kind,
        ids,
        exact,
        candidates,
        proofs,
        handle_required: None,
    }
}

fn edge_hit(store: &crate::StrataStore, edges: &[QualEdge]) -> Hit {
    let mut ids = Vec::new();
    let mut proofs = Vec::new();
    for edge in edges {
        if !ids.contains(&edge.source) {
            ids.push(edge.source.clone());
        }
        let frame_seq = store
            .edge_proofs(&edge.source, &edge.target, "touched")
            .ok()
            .and_then(|proofs| proofs.last().map(|proof| proof.data_seq));
        if let Some(frame_seq) = frame_seq {
            proofs.push(StructureProof::Edge {
                id: edge.source.clone(),
                source: edge.source.clone(),
                target: edge.target.clone(),
                link_type: "touched".to_string(),
                meta_sha: edge.meta_sha.clone(),
                frame_seq,
            });
        }
    }
    Hit {
        kind: StructureKind::File,
        ids,
        exact: true,
        candidates: Vec::new(),
        proofs,
        handle_required: None,
    }
}

fn commit_ids(store: &crate::StrataStore, ids: Vec<String>, exact: bool) -> Hit {
    let mut proofs = Vec::new();
    for id in &ids {
        let Some(origin) = store.recorded_origin(id).ok().flatten() else {
            continue;
        };
        let Some(effect) = store.origin_seq(id) else {
            continue;
        };
        proofs.push(StructureProof::Commit {
            id: id.clone(),
            frame_seq: origin.frame_seq,
            frame_hash: hex_encode(&origin.frame_hash),
            receipt: crate::effect_receipt_id(effect),
        });
    }
    Hit {
        kind: StructureKind::Commit,
        ids,
        exact,
        candidates: Vec::new(),
        proofs,
        handle_required: None,
    }
}

fn run_ids(store: &crate::StrataStore, kind: StructureKind, ids: Vec<String>) -> Hit {
    let mut proofs = Vec::new();
    for id in &ids {
        if let Some(proof) = store.latest_effect(id).ok().flatten() {
            proofs.push(StructureProof::Run {
                id: id.clone(),
                receipt: crate::effect_receipt_id(proof.effect_seq),
            });
        }
    }
    Hit {
        kind,
        ids,
        exact: true,
        candidates: Vec::new(),
        proofs,
        handle_required: None,
    }
}

fn tables_of(store: &crate::StrataStore) -> Tables {
    let anchors: Vec<_> = store.each_anchor().cloned().collect();
    let mut tables = Tables {
        files: std::collections::HashMap::new(),
        symbols: std::collections::HashMap::new(),
        symbol_paths: std::collections::HashMap::new(),
        commits: std::collections::BTreeMap::new(),
        qualified: std::collections::HashMap::new(),
        qualified_by_path: std::collections::HashMap::new(),
        tests: std::collections::HashMap::new(),
        runs: std::collections::BTreeSet::new(),
    };
    for anchor in &anchors {
        let Some(node) = store.node_map().get(&anchor.node_id) else {
            continue;
        };
        if !node.is_live() {
            continue;
        }
        let row = AnchorRow {
            id: anchor.id.clone(),
            node_id: anchor.node_id.clone(),
        };
        tables
            .files
            .entry(anchor.file_path.clone())
            .or_default()
            .push(row.clone());
        if let Some(symbol) = anchor.symbol.as_deref() {
            if symbol.is_empty() {
                continue;
            }
            let key = format!("{}#{symbol}", anchor.file_path);
            tables.symbols.entry(key).or_default().push(row);
            tables
                .symbol_paths
                .entry(symbol.to_string())
                .or_default()
                .insert(anchor.file_path.clone());
        }
    }
    for node in store.node_map().values() {
        if !node.is_live() {
            continue;
        }
        if let Some(sha) = node.source.as_ref().and_then(commit_sha) {
            push_unique(tables.commits.entry(sha.to_string()).or_default(), &node.id);
        }
    }
    for edge in store.edge_list() {
        if edge.link_type != "touched" {
            continue;
        }
        let Some(source) = store.node_map().get(&edge.source_id) else {
            continue;
        };
        if !source.is_live() {
            continue;
        }
        if let Some(sha) = edge.meta_sha.as_deref().filter(|sha| is_full_sha(sha)) {
            push_unique(
                tables.commits.entry(sha.to_string()).or_default(),
                &edge.source_id,
            );
        }
        let Some((repo, path)) = parse_qualified_file_handle(&edge.target_id) else {
            continue;
        };
        if repo.is_empty() {
            continue;
        }
        let qual = QualEdge {
            handle: edge.target_id.clone(),
            source: edge.source_id.clone(),
            target: edge.target_id.clone(),
            meta_sha: edge.meta_sha.clone(),
        };
        push_edge(
            tables.qualified.entry(edge.target_id.clone()).or_default(),
            &qual,
        );
        push_edge(tables.qualified_by_path.entry(path).or_default(), &qual);
    }
    for (id, run) in store.run_map() {
        tables.runs.insert(id.clone());
        if !run.subject.is_empty() {
            tables
                .tests
                .entry(run.subject.clone())
                .or_default()
                .push(id.clone());
        }
    }
    tables
}

fn push_unique(ids: &mut Vec<String>, id: &str) {
    if !ids.iter().any(|kept| kept == id) {
        ids.push(id.to_string());
    }
}

fn push_edge(edges: &mut Vec<QualEdge>, edge: &QualEdge) {
    if !edges
        .iter()
        .any(|kept| kept.source == edge.source && kept.target == edge.target)
    {
        edges.push(edge.clone());
    }
}

fn commit_sha(source: &crate::types::SourceKey) -> Option<&str> {
    if source.system != "git" {
        return None;
    }
    if is_full_sha(&source.id) {
        return Some(source.id.as_str());
    }
    let (_, suffix) = source.id.rsplit_once('#')?;
    is_full_sha(suffix).then_some(suffix)
}

fn is_full_sha(value: &str) -> bool {
    value.len() == 40 && is_hex(value)
}

enum ShaShape {
    Full,
    Prefix,
    TooShort,
}

fn sha_shape(query: &str) -> Option<ShaShape> {
    if !is_hex(query) || query.len() > 40 {
        return None;
    }
    if query.len() == 40 {
        Some(ShaShape::Full)
    } else if query.len() >= 7 {
        Some(ShaShape::Prefix)
    } else if query.len() >= 4 && query.bytes().any(|byte| byte.is_ascii_digit()) {
        Some(ShaShape::TooShort)
    } else {
        None
    }
}

fn is_hex(query: &str) -> bool {
    !query.is_empty() && query.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Canonical listing of the derived handle tables, for the reopen check.
///
/// The bytes are a pure function of the registries replay rebuilds. They are
/// not stored on the log and they do not enter [`crate::StrataStore::state_digest`].
pub fn structure_snapshot(store: &crate::StrataStore) -> String {
    let tables = tables_of(store);
    let mut lines = Vec::new();
    let mut files: Vec<_> = tables.files.iter().collect();
    files.sort_by(|a, b| a.0.cmp(b.0));
    for (path, rows) in files {
        for row in rows {
            lines.push(format!("file\t{path}\t{}\t{}", row.id, row.node_id));
        }
    }
    let mut symbols: Vec<_> = tables.symbols.iter().collect();
    symbols.sort_by(|a, b| a.0.cmp(b.0));
    for (key, rows) in symbols {
        for row in rows {
            lines.push(format!("symbol\t{key}\t{}", row.id));
        }
    }
    for (sha, ids) in &tables.commits {
        for id in ids {
            lines.push(format!("commit\t{sha}\t{id}"));
        }
    }
    let mut qualified: Vec<_> = tables.qualified.keys().collect();
    qualified.sort();
    for handle in qualified {
        for edge in &tables.qualified[handle] {
            lines.push(format!("qualified\t{handle}\t{}", edge.source));
        }
    }
    let mut tests: Vec<_> = tables.tests.iter().collect();
    tests.sort_by(|left, right| left.0.cmp(right.0));
    for (subject, ids) in tests {
        for id in ids {
            lines.push(format!("test\t{subject}\t{id}"));
        }
    }
    for id in &tables.runs {
        lines.push(format!("run\t{id}"));
    }
    lines.join("\n")
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

    fn temp_store(name: &str) -> (crate::StrataStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "strata-handles-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (crate::StrataStore::open(&dir).unwrap(), dir)
    }

    fn put(store: &mut crate::StrataStore, content: &str, source: Option<SourceKey>) -> String {
        store
            .ingest(crate::IngestInput {
                content: content.into(),
                source,
                tags: vec!["git-commit".into()],
                ..Default::default()
            })
            .unwrap()
    }

    fn anchor(
        store: &mut crate::StrataStore,
        id: &str,
        node: &str,
        path: &str,
        symbol: Option<&str>,
    ) {
        store
            .record_anchors(vec![crate::AnchorRecord {
                id: id.into(),
                node_id: node.into(),
                file_path: path.into(),
                symbol: symbol.map(str::to_string),
                symbol_kind: symbol.map(|_| "fn".into()),
                start_line: Some(1),
                end_line: Some(1),
                span_lines: None,
                content_hash: None,
                captured_at_ms: 1,
                last_verified_at_ms: None,
                last_status: None,
            }])
            .unwrap();
    }

    #[test]
    fn handles_resolve_from_recorded_rows_and_reopen_matches() {
        const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        const SHA_B: &str = "aaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        const SHA_C: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let (mut store, dir) = temp_store("resolve");
        let bare = put(
            &mut store,
            "commit record whose content names src/a.rs and must not be how it resolves",
            Some(SourceKey {
                system: "git".into(),
                project: "code".into(),
                id: SHA_A.into(),
            }),
        );
        let qualified_commit = put(
            &mut store,
            "other commit",
            Some(SourceKey {
                system: "git".into(),
                project: "github.com/a/b".into(),
                id: format!("github.com/a/b#{SHA_C}"),
            }),
        );
        let _shared_prefix = put(
            &mut store,
            "commit that shares a 7-hex prefix",
            Some(SourceKey {
                system: "git".into(),
                project: "code".into(),
                id: SHA_B.into(),
            }),
        );
        let pattern = put(&mut store, "pattern body does not matter", None);
        let decoy = put(&mut store, "see src/a.rs in the text only", None);
        anchor(&mut store, "anc-parse", &pattern, "src/a.rs", Some("parse"));
        anchor(&mut store, "anc-upper", &pattern, "src/A.rs", Some("Parse"));
        anchor(
            &mut store,
            "anc-bak",
            &pattern,
            "src/a.rs.bak",
            Some("parse"),
        );
        let other = put(&mut store, "second symbol path", None);
        anchor(&mut store, "anc-other", &other, "src/b.rs", Some("parse"));

        let file = resolve_structure(&store, "src/a.rs").unwrap();
        assert_eq!(file.kind, StructureKind::File);
        assert_eq!(file.ids, vec![pattern.clone()]);
        assert!(file.exact);
        assert!(!file.ids.contains(&decoy));
        match &file.proofs[0] {
            StructureProof::Anchor {
                anchor_id, receipt, ..
            } => {
                assert_eq!(anchor_id, "anc-parse");
                assert!(receipt.starts_with("eff-"), "{receipt}");
            }
            other => panic!("{other:?}"),
        }
        assert!(resolve_structure(&store, "src/A.rs").unwrap().ids == vec![pattern.clone()]);
        assert_eq!(
            resolve_structure(&store, "src/a.rs.bak").unwrap().ids,
            vec![pattern.clone()]
        );
        let symbol = resolve_structure(&store, "src/a.rs#parse").unwrap();
        assert_eq!(symbol.kind, StructureKind::Symbol);
        assert_eq!(symbol.ids, vec![pattern.clone()]);
        assert!(resolve_structure(&store, "src/a.rs#Parse").is_none());
        let bare_symbol = resolve_structure(&store, "parse").unwrap();
        assert!(bare_symbol.ids.is_empty(), "{bare_symbol:?}");
        assert!(bare_symbol.candidates.len() >= 2, "{bare_symbol:?}");

        let full = resolve_structure(&store, SHA_A).unwrap();
        assert_eq!(full.kind, StructureKind::Commit);
        assert_eq!(full.ids, vec![bare.clone()]);
        assert!(full.exact);
        match &full.proofs[0] {
            StructureProof::Commit {
                frame_seq,
                frame_hash,
                receipt,
                ..
            } => {
                assert!(*frame_seq > 0);
                assert_eq!(frame_hash.len(), 64);
                assert!(receipt.starts_with("eff-"));
            }
            other => panic!("{other:?}"),
        }
        let prefix = resolve_structure(&store, &SHA_A[..8]).unwrap();
        assert_eq!(prefix.ids, vec![bare.clone()]);
        assert!(!prefix.exact);
        let ambiguous = resolve_structure(&store, &SHA_A[..7]).unwrap();
        assert!(ambiguous.ids.is_empty());
        assert!(ambiguous.candidates.len() >= 2);
        assert_eq!(
            resolve_structure(&store, SHA_C).unwrap().ids,
            vec![qualified_commit.clone()]
        );
        let too_short = resolve_structure(&store, "commit:aaaa1").unwrap();
        assert!(too_short.ids.is_empty());
        assert!(too_short
            .handle_required
            .as_deref()
            .unwrap()
            .contains("too short"));
        assert!(resolve_structure(&store, "aaaa1").is_none());

        let left = put(&mut store, "repo left", None);
        let right = put(&mut store, "repo right", None);
        let left_handle = qualified_file_handle("github.com/left/repo", "src/a.rs");
        let right_handle = qualified_file_handle("github.com/right/repo", "src/a.rs");
        for (source, target, sha) in [(&left, &left_handle, SHA_A), (&right, &right_handle, SHA_B)]
        {
            store
                .save_connection(&crate::ConnectionRecord {
                    source_id: source.clone(),
                    target_id: target.clone(),
                    strength_milli: 1000,
                    link_type: "touched".into(),
                    meta_sha: Some(sha.into()),
                    created_at_ms: 1,
                    activation_count: 0,
                })
                .unwrap();
        }
        let shared = resolve_structure(&store, "src/a.rs").unwrap();
        assert_eq!(
            shared.ids,
            vec![pattern.clone()],
            "anchors stay the resolution"
        );
        let listed: Vec<_> = shared.candidates.iter().map(|(id, _)| id.clone()).collect();
        assert!(listed.contains(&left_handle), "{listed:?}");
        assert!(listed.contains(&right_handle), "{listed:?}");
        let only_left = resolve_structure(&store, &left_handle).unwrap();
        assert_eq!(only_left.ids, vec![left.clone()]);
        assert!(matches!(only_left.proofs[0], StructureProof::Edge { .. }));
        assert!(!only_left.ids.contains(&right));

        let before_runs = store.state_digest();
        store
            .retire(
                &pattern,
                &other,
                &crate::AdmissionContext {
                    rule_id: Some(crate::RULE_SUPPRESS.to_string()),
                    confirm: false,
                },
            )
            .unwrap();
        assert!(
            resolve_structure(&store, "src/A.rs").is_none(),
            "a retired node's anchor must not resolve"
        );

        let empty = store.record_runs(Vec::new()).unwrap();
        assert!(empty.written.is_empty());
        let run = crate::RunRecord {
            run_id: "suite::crate::mod::keeps_order".into(),
            kind: crate::RunKind::Test,
            subject: "crate::mod::keeps_order".into(),
            commit: SHA_C.into(),
            status: crate::RunStatus::Failed,
            started_ms: 1,
            finished_ms: 2,
        };
        let first = store.record_runs(vec![run.clone()]).unwrap();
        assert_eq!(first.written.len(), 1);
        let again = store.record_runs(vec![run.clone()]).unwrap();
        assert!(again.written.is_empty());
        assert_eq!(again.unchanged.len(), 1);
        let mut changed = run.clone();
        changed.status = crate::RunStatus::Passed;
        let rewrite = store.record_runs(vec![changed]).unwrap();
        assert_eq!(rewrite.written.len(), 1);
        assert_eq!(store.runs().len(), 1);
        assert_eq!(
            store.run(&run.run_id).unwrap().status,
            crate::RunStatus::Passed
        );
        let resolved_run = resolve_structure(&store, "run:suite::crate::mod::keeps_order").unwrap();
        assert_eq!(resolved_run.kind, StructureKind::Run);
        assert!(matches!(resolved_run.proofs[0], StructureProof::Run { .. }));
        let resolved_test = resolve_structure(&store, "test:crate::mod::keeps_order").unwrap();
        assert_eq!(resolved_test.kind, StructureKind::Test);
        assert_eq!(resolved_test.ids, vec![run.run_id.clone()]);
        let imported = crate::parse_junit(
            r#"<testsuite name="s"><testcase classname="c" name="n"/></testsuite>"#,
            SHA_C,
        )
        .unwrap();
        let imported_first = store.record_runs(imported.clone()).unwrap();
        assert_eq!(imported_first.written, vec!["s::c::n".to_string()]);
        let imported_again = store.record_runs(imported).unwrap();
        assert!(imported_again.written.is_empty());
        assert_eq!(imported_again.unchanged.len(), 1);

        let digest = store.state_digest();
        assert_ne!(digest, before_runs);
        let snapshot = structure_snapshot(&store);
        drop(store);
        let reopened = crate::StrataStore::open(&dir).unwrap();
        assert_eq!(reopened.state_digest(), digest);
        assert_eq!(structure_snapshot(&reopened), snapshot);
        assert_eq!(
            resolve_structure(&reopened, "test:crate::mod::keeps_order")
                .unwrap()
                .ids,
            vec![run.run_id]
        );
        let quiet = temp_store("digest");
        let quiet_digest = quiet.0.state_digest();
        drop(quiet.0);
        assert_eq!(
            crate::StrataStore::open(&quiet.1).unwrap().state_digest(),
            quiet_digest
        );
    }
}
