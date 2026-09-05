//! Markdown projection: the durable subset of a scope's memory rendered into
//! the rule files other agent clients already read (CLAUDE.md, MEMORY.md).
//!
//! Vestige stays the source of truth. The projection is a fenced region the
//! store owns; everything outside the fence belongs to the human and is
//! never touched. Each projected line ends with the id of the memory it came
//! from, so a reader can trace a rule back to its evidence, and a later
//! re-import can tell an edited line from a generated one.
//!
//! Selection is deliberately narrow: decisions, patterns, and facts or notes
//! tagged `rule`, `preference` or `convention`, currently valid, at or above
//! a retention floor, in the requested scope. A projection that carried every
//! memory would be the "prompt sludge" the roadmap warns about.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::Storage;
use crate::storage::Result;

/// Which client file shape to render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProjectionFormat {
    /// A `## Vestige memory` section grouped by kind, for CLAUDE.md or AGENTS.md.
    ClaudeMd,
    /// A flat index of one line per memory, for a MEMORY.md style file.
    MemoryMd,
}

impl ProjectionFormat {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "claude-md" | "claude_md" | "claude" | "agents-md" => Some(Self::ClaudeMd),
            "memory-md" | "memory_md" | "memory" | "index" => Some(Self::MemoryMd),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ClaudeMd => "claude-md",
            Self::MemoryMd => "memory-md",
        }
    }
}

/// What to project and how much.
#[derive(Debug, Clone)]
pub struct ProjectionOptions {
    pub scope: String,
    pub format: ProjectionFormat,
    /// Memories below this retention strength are left out (default 0.3).
    pub min_retention: f64,
    /// Upper bound on projected memories (default 60).
    pub max_items: usize,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            scope: "user".to_string(),
            format: ProjectionFormat::ClaudeMd,
            min_retention: 0.3,
            max_items: 60,
        }
    }
}

/// One projected memory, kept alongside the rendered text so callers can
/// report provenance without re-parsing the Markdown.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectedItem {
    pub id: String,
    pub node_type: String,
    pub tags: Vec<String>,
    pub content: String,
    pub updated_at: DateTime<Utc>,
    pub retention: f64,
}

/// The rendered region plus the items behind it.
#[derive(Debug, Clone, Serialize)]
pub struct Projection {
    pub format: ProjectionFormat,
    pub scope: String,
    pub items: Vec<ProjectedItem>,
    /// The fenced region, begin marker through end marker, newline-terminated.
    pub region: String,
}

pub const BEGIN_MARKER: &str = "<!-- vestige:projection:begin";
pub const END_MARKER: &str = "<!-- vestige:projection:end -->";

/// Node types projected regardless of tags, in output order.
const DURABLE_TYPES: [&str; 2] = ["decision", "pattern"];
/// Tags that make a fact or note durable enough to project.
const DURABLE_TAGS: [&str; 3] = ["rule", "preference", "convention"];
/// How many candidates to pull per query before filtering.
const CANDIDATE_LIMIT: i32 = 500;
/// Longest single projected line before it is cut.
const MAX_LINE_CHARS: usize = 400;

/// Pick the durable subset of `opts.scope`.
pub fn select_durable(storage: &Storage, opts: &ProjectionOptions) -> Result<Vec<ProjectedItem>> {
    let now = Utc::now();
    let mut candidates = Vec::new();
    for node_type in DURABLE_TYPES {
        candidates.extend(storage.get_nodes_by_type_and_tag(node_type, None, CANDIDATE_LIMIT)?);
    }
    for node_type in ["fact", "note"] {
        for tag in DURABLE_TAGS {
            candidates.extend(storage.get_nodes_by_type_and_tag(node_type, Some(tag), CANDIDATE_LIMIT)?);
        }
    }

    let mut seen = std::collections::HashSet::new();
    let mut items = Vec::new();
    for node in candidates {
        if !seen.insert(node.id.clone()) {
            continue;
        }
        if node.valid_until.is_some_and(|until| until <= now) {
            continue;
        }
        if node.retention_strength < opts.min_retention {
            continue;
        }
        // The tag query is a substring match on the JSON array, so confirm
        // the tag is really present before trusting it.
        let tagged = node
            .tags
            .iter()
            .any(|tag| DURABLE_TAGS.contains(&tag.to_ascii_lowercase().as_str()));
        if !DURABLE_TYPES.contains(&node.node_type.as_str()) && !tagged {
            continue;
        }
        if !storage.node_is_in_scope(&node.id, &opts.scope)? {
            continue;
        }
        items.push(ProjectedItem {
            id: node.id,
            node_type: node.node_type,
            tags: node.tags,
            content: node.content,
            updated_at: node.updated_at,
            retention: node.retention_strength,
        });
    }

    items.sort_by(|a, b| {
        kind_rank(&a.node_type, &a.tags)
            .cmp(&kind_rank(&b.node_type, &b.tags))
            .then_with(|| b.updated_at.cmp(&a.updated_at))
            .then_with(|| a.id.cmp(&b.id))
    });
    items.truncate(opts.max_items);
    Ok(items)
}

fn kind_rank(node_type: &str, _tags: &[String]) -> u8 {
    match node_type {
        "decision" => 0,
        "pattern" => 1,
        _ => 2,
    }
}

fn kind_heading(rank: u8) -> &'static str {
    match rank {
        0 => "Decisions",
        1 => "Patterns",
        _ => "Rules and preferences",
    }
}

/// One line of Markdown for a memory: whitespace collapsed, cut at
/// `MAX_LINE_CHARS`, provenance comment at the end.
fn render_line(item: &ProjectedItem) -> String {
    let collapsed: String = item.content.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut text: String = collapsed.chars().take(MAX_LINE_CHARS).collect();
    if collapsed.chars().count() > MAX_LINE_CHARS {
        text.push('…');
    }
    format!("- {text} <!-- vestige:{} -->", item.id)
}

/// Render the fenced region for `items`. Deterministic for the same input:
/// no timestamps inside the fence, so an unchanged store projects to an
/// unchanged file.
pub fn render(format: ProjectionFormat, scope: &str, items: &[ProjectedItem]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{BEGIN_MARKER} scope={scope} format={} -->\n",
        format.label()
    ));
    match format {
        ProjectionFormat::ClaudeMd => {
            out.push_str("## Vestige memory (projected)\n\n");
            out.push_str(&format!(
                "{} durable memories from scope `{scope}`. Change them in Vestige, not here: the next projection replaces this block. The comment at the end of each line is the memory id.\n",
                items.len()
            ));
            let mut current: Option<u8> = None;
            for item in items {
                let rank = kind_rank(&item.node_type, &item.tags);
                if current != Some(rank) {
                    out.push_str(&format!("\n### {}\n", kind_heading(rank)));
                    current = Some(rank);
                }
                out.push_str(&render_line(item));
                out.push('\n');
            }
        }
        ProjectionFormat::MemoryMd => {
            out.push_str("# Memory index (projected by Vestige)\n\n");
            out.push_str(&format!(
                "{} durable memories from scope `{scope}`; one line each, memory id at the end.\n\n",
                items.len()
            ));
            for item in items {
                let line = render_line(item);
                out.push_str(&line.replacen("- ", &format!("- [{}] ", item.node_type), 1));
                out.push('\n');
            }
        }
    }
    out.push_str(END_MARKER);
    out.push('\n');
    out
}

/// Build the full projection for a scope.
pub fn project(storage: &Storage, opts: &ProjectionOptions) -> Result<Projection> {
    let items = select_durable(storage, opts)?;
    let region = render(opts.format, &opts.scope, &items);
    Ok(Projection {
        format: opts.format,
        scope: opts.scope.clone(),
        items,
        region,
    })
}

/// Replace the fenced region inside `existing` with `region`, or append it
/// when the file has none. Text outside the fence is returned byte for byte.
/// Applying the same region twice yields the same file.
pub fn splice(existing: &str, region: &str) -> String {
    let region = region.strip_suffix('\n').unwrap_or(region);
    if let Some(begin) = existing.find(BEGIN_MARKER) {
        let line_start = existing[..begin].rfind('\n').map_or(0, |i| i + 1);
        if let Some(end_rel) = existing[begin..].find(END_MARKER) {
            let end = begin + end_rel + END_MARKER.len();
            // Swallow the newline that closed the old end marker so the new
            // region's own terminator does not double it.
            let end = if existing[end..].starts_with('\n') { end + 1 } else { end };
            let mut out = String::with_capacity(existing.len() + region.len());
            out.push_str(&existing[..line_start]);
            out.push_str(region);
            out.push('\n');
            out.push_str(&existing[end..]);
            return out;
        }
    }
    let mut out = existing.to_string();
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    if !out.is_empty() && !out.ends_with("\n\n") {
        out.push('\n');
    }
    out.push_str(region);
    out.push('\n');
    out
}

/// One line of a preview diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiffLine {
    /// `'+'`, `'-'` or `' '`.
    pub kind: char,
    pub text: String,
}

/// Longest-common-subsequence line diff, enough for a rule file. Inputs
/// larger than `LCS_MAX_LINES` lines fall back to remove-all/add-all so the
/// quadratic table stays bounded.
pub fn line_diff(old: &str, new: &str) -> Vec<DiffLine> {
    const LCS_MAX_LINES: usize = 4_000;
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    if a.len() > LCS_MAX_LINES || b.len() > LCS_MAX_LINES {
        let mut out: Vec<DiffLine> = a
            .iter()
            .map(|l| DiffLine { kind: '-', text: l.to_string() })
            .collect();
        out.extend(b.iter().map(|l| DiffLine { kind: '+', text: l.to_string() }));
        return out;
    }
    let (n, m) = (a.len(), b.len());
    let mut table = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            table[i][j] = if a[i] == b[j] {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < n && j < m {
        if a[i] == b[j] {
            out.push(DiffLine { kind: ' ', text: a[i].to_string() });
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            out.push(DiffLine { kind: '-', text: a[i].to_string() });
            i += 1;
        } else {
            out.push(DiffLine { kind: '+', text: b[j].to_string() });
            j += 1;
        }
    }
    out.extend(a[i..].iter().map(|l| DiffLine { kind: '-', text: l.to_string() }));
    out.extend(b[j..].iter().map(|l| DiffLine { kind: '+', text: l.to_string() }));
    out
}

/// Counts of added and removed lines in a diff.
pub fn diff_summary(diff: &[DiffLine]) -> (usize, usize) {
    let added = diff.iter().filter(|l| l.kind == '+').count();
    let removed = diff.iter().filter(|l| l.kind == '-').count();
    (added, removed)
}

/// The changed lines with one line of context each side, as `+`/`-`/` `
/// prefixed text; unchanged runs are elided with `...`.
pub fn unified(diff: &[DiffLine], max_lines: usize) -> String {
    let mut keep = vec![false; diff.len()];
    for (i, line) in diff.iter().enumerate() {
        if line.kind != ' ' {
            let lo = i.saturating_sub(1);
            let hi = (i + 1).min(diff.len().saturating_sub(1));
            for flag in &mut keep[lo..=hi] {
                *flag = true;
            }
        }
    }
    let mut out = String::new();
    let mut emitted = 0usize;
    let mut eliding = false;
    for (i, line) in diff.iter().enumerate() {
        if !keep[i] {
            if !eliding {
                out.push_str("...\n");
                eliding = true;
            }
            continue;
        }
        eliding = false;
        if emitted >= max_lines {
            out.push_str("... (diff truncated)\n");
            break;
        }
        out.push(line.kind);
        out.push_str(&line.text);
        out.push('\n');
        emitted += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IngestInput;

    fn item(id: &str, node_type: &str, content: &str) -> ProjectedItem {
        ProjectedItem {
            id: id.to_string(),
            node_type: node_type.to_string(),
            tags: vec![],
            content: content.to_string(),
            updated_at: Utc::now(),
            retention: 0.9,
        }
    }

    #[test]
    fn render_is_deterministic_and_carries_provenance() {
        let items = vec![
            item("d1", "decision", "Use content-hashed cache keys\nfor build artifacts"),
            item("p1", "pattern", "Wrap every writer in begin_write_transaction"),
        ];
        let a = render(ProjectionFormat::ClaudeMd, "user", &items);
        let b = render(ProjectionFormat::ClaudeMd, "user", &items);
        assert_eq!(a, b);
        assert!(a.starts_with(BEGIN_MARKER));
        assert!(a.ends_with(&format!("{END_MARKER}\n")));
        assert!(a.contains("### Decisions\n- Use content-hashed cache keys for build artifacts <!-- vestige:d1 -->"));
        assert!(a.contains("### Patterns\n- Wrap every writer in begin_write_transaction <!-- vestige:p1 -->"));
        let index = render(ProjectionFormat::MemoryMd, "user", &items);
        assert!(index.contains("- [decision] Use content-hashed cache keys"));
    }

    #[test]
    fn long_content_is_cut_on_one_line() {
        let long = "x".repeat(1_000);
        let line = render_line(&item("id", "decision", &long));
        assert!(line.chars().count() < 450, "{}", line.chars().count());
        assert!(line.ends_with("<!-- vestige:id -->"));
        assert!(line.contains('…'));
    }

    #[test]
    fn splice_replaces_only_the_fence_and_is_idempotent() {
        let region = render(ProjectionFormat::ClaudeMd, "user", &[item("d1", "decision", "one")]);
        let hand_written = "# My project\n\nKeep this paragraph.\n";
        let first = splice(hand_written, &region);
        assert!(first.starts_with(hand_written), "text before the fence must survive:\n{first}");
        assert!(first.contains("<!-- vestige:d1 -->"));
        let again = splice(&first, &region);
        assert_eq!(first, again, "re-projecting an unchanged store must not change the file");

        let with_tail = format!("{first}\n## After the fence\nStill mine.\n");
        let updated = render(ProjectionFormat::ClaudeMd, "user", &[item("d2", "decision", "two")]);
        let spliced = splice(&with_tail, &updated);
        assert!(spliced.starts_with(hand_written));
        assert!(spliced.ends_with("## After the fence\nStill mine.\n"), "{spliced}");
        assert!(spliced.contains("vestige:d2") && !spliced.contains("vestige:d1"));
        assert_eq!(spliced.matches(BEGIN_MARKER).count(), 1);
    }

    #[test]
    fn line_diff_reports_the_changed_lines() {
        let old = "a\nb\nc\n";
        let new = "a\nB\nc\nd\n";
        let diff = line_diff(old, new);
        assert_eq!(diff_summary(&diff), (2, 1));
        let text = unified(&diff, 100);
        assert!(text.contains("-b\n") && text.contains("+B\n") && text.contains("+d\n"), "{text}");
        assert!(line_diff("same\n", "same\n").iter().all(|l| l.kind == ' '));
    }

    #[test]
    fn select_durable_keeps_decisions_patterns_and_rule_tagged_facts_only() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        let ingest = |content: &str, node_type: &str, tags: &[&str], valid_until| {
            storage
                .ingest(IngestInput {
                    content: content.to_string(),
                    node_type: node_type.to_string(),
                    tags: tags.iter().map(|t| t.to_string()).collect(),
                    valid_until,
                    ..Default::default()
                })
                .unwrap()
                .id
        };
        let decision = ingest("Ship releases from an integration branch", "decision", &[], None);
        let pattern = ingest("Touch files after scripted edits", "pattern", &[], None);
        let rule = ingest("Prefer tabs in Svelte files", "fact", &["preference"], None);
        let plain = ingest("The office moved in spring", "fact", &[], None);
        let expired = ingest(
            "Old decision that was superseded",
            "decision",
            &[],
            Some(Utc::now() - chrono::Duration::days(1)),
        );

        let items = select_durable(&storage, &ProjectionOptions::default()).unwrap();
        let ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&decision.as_str()), "{ids:?}");
        assert!(ids.contains(&pattern.as_str()), "{ids:?}");
        assert!(ids.contains(&rule.as_str()), "{ids:?}");
        assert!(!ids.contains(&plain.as_str()), "an untagged fact is not durable: {ids:?}");
        assert!(!ids.contains(&expired.as_str()), "an invalidated memory is out: {ids:?}");
        assert_eq!(items[0].node_type, "decision", "decisions render first");

        let strict = select_durable(
            &storage,
            &ProjectionOptions {
                min_retention: 1.1,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(strict.is_empty(), "the retention floor applies");

        let other_scope = select_durable(
            &storage,
            &ProjectionOptions {
                scope: "some-other-project".to_string(),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(other_scope.is_empty(), "scope isolation holds");
    }
}
