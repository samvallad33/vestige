//! Forbidden-mechanism guard (PR-0c ratchet).
//!
//! Scans the live sources for mechanisms the 4.0 hard rules removed
//! (FTS/BM25, embeddings/cosine, fuzzy matching, lexicons, LLM judges,
//! wall-clock/floats/HashMap in scoped decision paths, …) and fails on:
//!
//! 1. any hit that is not allowlisted as `path:Gnn`, and
//! 2. any allowlist entry that no longer matches — so the list only shrinks.
//!
//! Seed or re-verify locally with `GUARD_SEED=1 cargo test -p vestige-mcp
//! --test forbidden_mechanisms` (seed mode rewrites the allowlist; review
//! the diff before committing). Permanent, justified hits go under group 99
//! with a one-line reason and need PR-body approval.
//!
//! Excluded from scanning (v3-schema names are the migration contract, not
//! live mechanisms): `crates/vestige-core/src/storage/migrations.rs`, the
//! strata-migrate crate (read-only migrator), `docs/adr/**`,
//! `CHANGELOG.md`, and this file + its allowlist.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use regex::{Regex, RegexBuilder};

const ALLOWLIST_FILE: &str = "forbidden_allowlist.txt";

/// `(group, pattern, case_sensitive, scoped)` — scoped patterns apply only
/// to the strata crates and vestige-core's decision-path modules.
const PATTERNS: &[(&str, &str, bool, bool)] = &[
    (
        "G01",
        r"knowledge_fts|\bfts5\b|\bMATCH\s*\?|ORDER BY rank|\bbm25\b|sanitize_fts5",
        false,
        false,
    ),
    (
        "G02",
        r"\bembedding(s)?\b|node_embeddings|hashed_embedding|cosine|Vec<f32>|vector_search|fastembed|usearch|hnsw",
        false,
        false,
    ),
    (
        "G03",
        r"levenshtein|strsim|edit_distance|similar_tag_suggestions",
        false,
        false,
    ),
    (
        "G04",
        r"\bjaccard\b|\bdice\b|\bidf\b|stop_?words|TOPIC_STOPWORDS|token_set|topic_overlap",
        false,
        false,
    ),
    (
        "G05",
        r"FAILURE_MARKERS|FIX_MARKERS|LESSON_TAGS|NEGATION_PAIRS|SENSITIVE_TOPICS|default_positive_words|default_intensity_keywords|lexicon",
        true,
        false,
    ),
    (
        "G06",
        r"extract_entities|identifier_tier|normalized_tier|is_identifier_shaped|shared_tags|shared_content_terms|\bmentions\b",
        false,
        false,
    ),
    (
        "G07",
        r"LIKE\s+'%|LIKE \?\s*\|\||content LIKE|boundary_match_ids",
        false,
        false,
    ),
    (
        "G08",
        r"hybrid_search|search_terms|rewrite_queries|prose_form|content_similarity|similarity_to_query|score_pair|is_literal_query",
        false,
        false,
    ),
    (
        "G09",
        r"apply_model_verdict|chat/completions|VESTIGE_SANHEDRIN_MODEL|mlx_lm|ollama|vllm|openai|anthropic",
        false,
        false,
    ),
    (
        "G10",
        r"spreading_activation|\bactivate\(|semantic_cosine|cosine_unit|strengthen_on_access|record_batch_retrieval|hippocampal",
        false,
        false,
    ),
    (
        "G11",
        r#"LinkType::(Semantic|Temporal|Complementary|SharedConcepts)|"semantic"|"shared_concepts"|backfill_candidate"#,
        false,
        false,
    ),
    (
        "G12",
        r"Utc::now\(\)|SystemTime::now|Instant::now",
        false,
        true,
    ),
    ("G13", r"Uuid::new_v4", false, false),
    ("G14", r"partial_cmp|as f64|: f64|f32", false, true),
    ("G15", r"HashMap<|HashSet<", false, true),
    ("G16", r"try_lock\(", false, false),
    (
        "G17",
        r"looks_like_failure|is_failure_memory|contains_marker_word|get_never_composed",
        false,
        false,
    ),
    (
        "G18",
        r#"cfg\(feature\s*=\s*"(embeddings|vector-search|ort-download|ort-dynamic|qwen3-embeddings|metal|cuda|cudnn)"#,
        false,
        false,
    ),
    (
        "G19",
        r"apply_decay|labile|share tags|topics|BLOCKING PHRASE|auto_dedup_consolidation|DELETE FROM knowledge_nodes",
        false,
        false,
    ),
];

/// Scanned roots: crate sources and tests, hooks, and the dashboard surface.
const SCAN_ROOTS: &[&str] = &["crates", "hooks"];

/// Files whose v3-schema names are the migration contract, not live
/// mechanisms (same rationale the spec applies to storage/migrations.rs).
fn excluded(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.ends_with("crates/vestige-core/src/storage/migrations.rs")
        || text.contains("crates/strata-migrate/")
        || text.ends_with("tests/forbidden_mechanisms.rs")
        || text.ends_with(ALLOWLIST_FILE)
        || text.contains("docs/adr/")
        || text.ends_with("CHANGELOG.md")
}

/// Scoped paths for G12/G14/G15: the strata crates and vestige-core's
/// decision-path modules (receipt-producing code).
fn scoped(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.contains("crates/strata")
        || text.contains("crates/vestige-core/src/storage/")
        || text.contains("crates/vestige-core/src/fsrs/")
        || text.contains("crates/vestige-core/src/advanced/")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name == "target" || name == "fixtures" || name.starts_with('.') {
                continue;
            }
            walk(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn source_files() -> Vec<PathBuf> {
    // Tests run with cwd = the crate dir; scan from the repo root.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root")
        .to_path_buf();
    let mut files = Vec::new();
    for scan_root in SCAN_ROOTS {
        walk(&root.join(scan_root), &mut files);
    }
    files
        .into_iter()
        .filter(|p| {
            let text = p.to_string_lossy().to_string();
            let ext_ok = text.ends_with(".rs")
                || text.ends_with(".py")
                || text.ends_with(".sh")
                || text.ends_with(".ts")
                || text.ends_with(".svelte");
            ext_ok && !excluded(p)
        })
        .collect()
}

/// Every current hit: `path:Gnn` (path relative to the repo root).
fn current_hits() -> BTreeMap<(String, String), u64> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root");
    let files = source_files();
    let mut hits: BTreeMap<(String, String), u64> = BTreeMap::new();
    for (group, pattern, case_sensitive, is_scoped) in PATTERNS {
        let regex = RegexBuilder::new(pattern)
            .case_insensitive(!*case_sensitive)
            .build()
            .expect("guard pattern compiles");
        for path in &files {
            if *is_scoped && !scoped(path) {
                continue;
            }
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            let text = String::from_utf8_lossy(&bytes);
            let count = regex.find_iter(&text).count() as u64;
            if count > 0 {
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();
                *hits.entry((rel, (*group).to_string())).or_default() += count;
            }
        }
    }
    hits
}

fn allowlist_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(ALLOWLIST_FILE)
}

fn parse_allowlist() -> Vec<(String, String)> {
    let text = std::fs::read_to_string(allowlist_path()).unwrap_or_default();
    text.lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
        .map(|line| {
            let entry = line.split('#').next().unwrap_or(line).trim();
            let (path, group) = entry.rsplit_once(":G").unwrap_or((entry, ""));
            (path.to_string(), format!("G{group}"))
        })
        .filter(|(path, group)| !path.is_empty() && group.starts_with('G'))
        .collect()
}

fn write_allowlist(hits: &BTreeMap<(String, String), u64>) {
    let mut by_group: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for ((path, group), _) in hits {
        by_group
            .entry(group.clone())
            .or_default()
            .push(path.clone());
    }
    let mut out = String::from(
        "# Forbidden-mechanism allowlist (PR-0c ratchet). Format: <path>:G<nn>.\n\
         # Entries are removed as their group is fixed; the guard fails on\n\
         # stale entries, so this list only shrinks. Group 99 lines carry a\n\
         # permanent justification and need PR-body approval.\n",
    );
    for (group, mut paths) in by_group {
        paths.sort();
        out.push_str(&format!("\n# {group}\n"));
        for path in paths {
            out.push_str(&format!("{path}:{group}\n"));
        }
    }
    std::fs::write(allowlist_path(), out).expect("write allowlist");
}

#[test]
fn forbidden_mechanisms_guard() {
    let hits = current_hits();
    let allow: BTreeSet<(String, String)> = parse_allowlist().into_iter().collect();

    if std::env::var("GUARD_SEED").is_ok() {
        write_allowlist(&hits);
        println!("seeded {} entries", hits.len());
        return;
    }

    let mut unlisted: Vec<String> = Vec::new();
    for key in hits.keys() {
        if !allow.contains(key) {
            unlisted.push(format!("{}:{}", key.0, key.1));
        }
    }
    assert!(
        unlisted.is_empty(),
        "forbidden mechanisms found (add to the allowlist only while the \
         owning PR is open, never to make CI pass):\n{}",
        unlisted.join("\n")
    );
}

#[test]
fn forbidden_allowlist_has_no_stale_entries() {
    let hits = current_hits();
    let allow = parse_allowlist();
    let mut stale: Vec<String> = Vec::new();
    for entry in allow {
        if !hits.contains_key(&entry) {
            stale.push(format!("{}:{}", entry.0, entry.1));
        }
    }
    assert!(
        stale.is_empty(),
        "stale allowlist entries (the mechanism is gone; delete the lines):\n{}",
        stale.join("\n")
    );
}

/// The dependency lock must stay free of search-engine crates (H1/H2).
#[test]
fn no_search_deps() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"))
        .expect("root Cargo.lock");
    let exact = [
        "fastembed",
        "ort",
        "usearch",
        "tantivy",
        "tokenizers",
        "bm25",
        "probly-search",
        "sqlite-vec",
    ];
    let prefixes = ["hnsw", "candle-"];
    let mut hits = Vec::new();
    for line in lock.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix("name = ") {
            let name = name.trim_matches('"');
            if exact.contains(&name) || prefixes.iter().any(|p| name.starts_with(p)) {
                hits.push(name.to_string());
            }
        }
    }
    assert!(
        hits.is_empty(),
        "search dependencies present in Cargo.lock: {hits:?}"
    );
}

/// The frame-kind registry is append-only: every kind number pinned in
/// kinds.rs stays exactly where it was put (renumbering invalidates every
/// recorded receipt).
#[test]
fn registry_append_only() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../strata-store/src/kinds.rs"
    ))
    .expect("kinds.rs");
    let pins = [
        ("KIND_RECALL_RECEIPT", 34),
        ("KIND_WALK_RECEIPT", 35),
        ("KIND_ADMISSION_RECEIPT", 36),
        ("KIND_CLAIM_VERDICT", 37),
        ("KIND_TOOL_CALL", 38),
        ("KIND_TOOL_RESULT", 39),
        ("KIND_DERIVE", 40),
        ("KIND_COUNTERFACTUAL_RECEIPT", 41),
        ("KIND_SELFTEST_RECEIPT", 42),
        ("KIND_IMPACT_RECEIPT", 43),
        ("KIND_CONFLICT_RECEIPT", 44),
        ("KIND_GHOSTLINK_RECEIPT", 45),
        ("KIND_MIGRATION_RECEIPT", 46),
        ("KIND_PARAMS", 47),
        ("KIND_CLAIM", 48),
        ("KIND_SESSION_MARK", 49),
        ("KIND_REVIEW", 50),
        ("KIND_INTENTION_FIRED", 51),
        ("KIND_JOB", 52),
    ];
    for (name, number) in pins {
        let needle = format!("pub const {name}: u8 = {number};");
        assert!(
            text.contains(&needle),
            "frame kind {name} must stay pinned at {number} (append-only registry)"
        );
    }
}

/// Layering: the four proof-stack crates never depend on the legacy engine,
/// an HTTP stack, or SQLite. (`strata-migrate` is the one sanctioned bridge:
/// it may depend on vestige-core types and rusqlite to read v3 sources.)
#[test]
fn strata_layering() {
    let crates_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let forbidden = [
        "vestige-core",
        "vestige-mcp",
        "vestige-spacetime",
        "reqwest",
        "hyper",
        "rusqlite",
    ];
    for name in ["strata", "strata-kernel", "strata-gate", "strata-store"] {
        let manifest = std::fs::read_to_string(format!("{crates_dir}/{name}/Cargo.toml"))
            .unwrap_or_else(|_| panic!("{name}/Cargo.toml"));
        for dep in forbidden {
            assert!(
                !manifest.contains(&format!("{dep}")),
                "{name} must not depend on {dep} (strata layering)"
            );
        }
    }
}

/// Every entry in this allowlist that stays past PR 11 must carry a group-99
/// justification (helper used by the report; asserted via the count test).
#[test]
fn guard_allowlist_is_grouped() {
    let text = std::fs::read_to_string(allowlist_path()).unwrap_or_default();
    for line in text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
    {
        assert!(
            line.contains(":G"),
            "allowlist line must be <path>:G<nn>: {line}"
        );
    }
}

/// Locate the repo root for CI use (helper for the workflow wiring docs).
#[allow(dead_code)]
fn repo_root() -> Option<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    Some(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim(),
    ))
}
