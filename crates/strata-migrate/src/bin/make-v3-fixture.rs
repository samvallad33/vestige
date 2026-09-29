//! Build schema 31 / 36 / 38 fixtures from vestige-core's migration SQL.
//!
//! The statements are the `MIGRATION_V*_UP` constants (and the ALTER arrays
//! the runner applies beside them) in
//! `crates/vestige-core/src/storage/migrations.rs`. This binary executes
//! that SQL through the requested version, then inserts a small synthetic
//! dataset. It does not call vestige-core's private `apply_migrations`.
//!
//!     cargo run --manifest-path crates/strata-migrate/Cargo.toml \
//!         --bin make-v3-fixture -- <out.sqlite> <31|36|38>

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use rusqlite::types::Value;
use rusqlite::Connection;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .map(PathBuf::from)
        .context("usage: make-v3-fixture <out.sqlite> <31|36|38>")?;
    let through: u32 = args
        .next()
        .context("schema version 31, 36, or 38")?
        .parse()?;
    if !matches!(through, 31 | 36 | 38) {
        bail!("schema version must be 31, 36, or 38");
    }
    if out.exists() {
        bail!("refusing to overwrite {}", out.display());
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    build(&out, through)?;
    let conn = Connection::open(&out)?;
    let version: u32 = conn.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
        row.get(0)
    })?;
    let tables: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
        [],
        |row| row.get(0),
    )?;
    let walk: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'walk_receipts'",
        [],
        |row| row.get(0),
    )?;
    println!("schema={version} tables={tables} walk_receipts={walk}");
    if version != through {
        bail!("schema_version {version} != requested {through}");
    }
    if through == 38 && tables != 67 {
        bail!("schema 38 has {tables} tables, expected 67");
    }
    if walk != 0 {
        bail!("walk_receipts must not exist before V40");
    }
    Ok(())
}

fn migrations_source() -> anyhow::Result<String> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../vestige-core/src/storage/migrations.rs");
    std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))
}

fn build(path: &Path, through: u32) -> anyhow::Result<()> {
    let source = migrations_source()?;
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "foreign_keys", "OFF")?;
    for version in 1..=through {
        apply_version(&conn, &source, version)?;
    }
    seed(&conn)?;
    conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")?;
    conn.pragma_update(None, "journal_mode", "DELETE")?;
    conn.execute_batch("VACUUM;")?;
    Ok(())
}

fn apply_version(conn: &Connection, source: &str, version: u32) -> anyhow::Result<()> {
    let up = if version == 25 {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../vestige-core/src/storage/unlearning_store.rs");
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        extract_str_const(&text, "V25_UNLEARNING_STORAGE_SCHEMA_EXPECTATION")
            .context("missing V25 schema SQL")?
    } else {
        extract_str_const(source, &format!("MIGRATION_V{version}_UP"))
            .with_context(|| format!("missing MIGRATION_V{version}_UP"))?
    };
    let alters = match version {
        2 | 16 | 17 | 22 | 26 => {
            extract_str_array(source, &format!("MIGRATION_V{version}_ALTER_COLUMNS"))
                .with_context(|| format!("missing V{version} ALTER array"))?
        }
        14 => vec![
            "ALTER TABLE knowledge_nodes ADD COLUMN protected INTEGER NOT NULL DEFAULT 0"
                .to_string(),
            "ALTER TABLE knowledge_nodes ADD COLUMN superseded_by TEXT".to_string(),
        ],
        _ => Vec::new(),
    };
    let tx = conn.unchecked_transaction()?;
    for stmt in &alters {
        add_column_if_missing(&tx, stmt).with_context(|| format!("V{version} alter: {stmt}"))?;
    }
    let (add_columns, remaining) = split_add_column_statements(&up);
    for stmt in &add_columns {
        add_column_if_missing(&tx, stmt)
            .with_context(|| format!("V{version} add column: {stmt}"))?;
    }
    tx.execute_batch(&remaining)
        .with_context(|| format!("V{version} batch"))?;
    if version == 25 {
        tx.execute(
            "UPDATE schema_version SET version = 25, applied_at = datetime('now')",
            [],
        )?;
    }
    tx.commit()?;
    if version == 7 {
        conn.pragma_update(None, "journal_mode", "DELETE")?;
        conn.pragma_update(None, "page_size", 8192)?;
        conn.execute_batch("VACUUM;")?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
    }
    Ok(())
}

fn add_column_if_missing(conn: &Connection, sql: &str) -> rusqlite::Result<()> {
    match conn.execute(sql, []) {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(_, Some(msg)))
            if msg.contains("duplicate column name") =>
        {
            Ok(())
        }
        Err(e) => Err(e),
    }
}

fn split_add_column_statements(up: &str) -> (Vec<String>, String) {
    let mut statements = Vec::new();
    let mut remaining = String::new();
    let mut pending: Option<(String, Vec<String>)> = None;
    for line in up.lines() {
        if let Some((statement, lines)) = pending.as_mut() {
            statement.push(' ');
            statement.push_str(line.trim());
            lines.push(line.to_string());
            if line.trim_end().ends_with(';') {
                let (statement, lines) = pending.take().expect("pending");
                finish_alter(statement, lines, &mut statements, &mut remaining);
            }
            continue;
        }
        let trimmed = line.trim();
        if trimmed.to_ascii_uppercase().starts_with("ALTER TABLE") {
            if trimmed.ends_with(';') {
                finish_alter(
                    trimmed.to_string(),
                    vec![line.to_string()],
                    &mut statements,
                    &mut remaining,
                );
            } else {
                pending = Some((trimmed.to_string(), vec![line.to_string()]));
            }
            continue;
        }
        remaining.push_str(line);
        remaining.push('\n');
    }
    if let Some((_, lines)) = pending {
        for line in lines {
            remaining.push_str(&line);
            remaining.push('\n');
        }
    }
    (statements, remaining)
}

fn finish_alter(
    statement: String,
    lines: Vec<String>,
    statements: &mut Vec<String>,
    remaining: &mut String,
) {
    if statement.to_ascii_uppercase().contains(" ADD COLUMN ") {
        statements.push(statement.trim_end_matches(';').trim().to_string());
    } else {
        for line in lines {
            remaining.push_str(&line);
            remaining.push('\n');
        }
    }
}

fn extract_str_const(src: &str, name: &str) -> Option<String> {
    let marker = format!("const {name}:");
    let at = src.find(&marker)?;
    let rest = &src[at..];
    let eq = rest.find('=')?;
    let after = &rest[eq + 1..];
    let mut offset = 0usize;
    let start = loop {
        let rel = after[offset..].find('r')?;
        let abs = offset + rel;
        let next = after.as_bytes().get(abs + 1).copied();
        if next == Some(b'#') || next == Some(b'"') {
            break abs;
        }
        offset = abs + 1;
    };
    let (body, _) = raw_string(&after[start..])?;
    Some(body)
}

fn extract_str_array(src: &str, name: &str) -> Option<Vec<String>> {
    let marker = format!("const {name}:");
    let at = src.find(&marker)?;
    let rest = &src[at..];
    let open = rest.find("= &[")?;
    let close = rest[open..].find("];")?;
    let body = &rest[open..open + close];
    let mut out = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        i += 1;
        let mut s = String::new();
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => {
                    i += 1;
                    if i < bytes.len() {
                        s.push(bytes[i] as char);
                        i += 1;
                    }
                }
                b'"' => {
                    i += 1;
                    break;
                }
                c => {
                    s.push(c as char);
                    i += 1;
                }
            }
        }
        if !s.is_empty() {
            out.push(s);
        }
    }
    Some(out)
}

fn raw_string(src: &str) -> Option<(String, usize)> {
    if !src.starts_with('r') {
        return None;
    }
    let hashes = src[1..].chars().take_while(|c| *c == '#').count();
    let quote = 1 + hashes;
    if src.as_bytes().get(quote) != Some(&b'"') {
        return None;
    }
    let body_start = quote + 1;
    let closer = format!("\"{}", "#".repeat(hashes));
    let end = src[body_start..].find(&closer)?;
    Some((
        src[body_start..body_start + end].to_string(),
        body_start + end + closer.len(),
    ))
}

struct Col {
    name: String,
    notnull: bool,
    defaulted: bool,
    ty: String,
}

fn table_info(conn: &Connection, table: &str) -> anyhow::Result<Vec<Col>> {
    let escaped = table.replace('\'', "''");
    let mut stmt = conn.prepare(&format!(
        "SELECT name, type, \"notnull\", dflt_value FROM pragma_table_info('{escaped}')"
    ))?;
    let rows = stmt.query_map([], |row| {
        let default: Option<String> = row.get(3)?;
        Ok(Col {
            name: row.get(0)?,
            ty: row.get::<_, String>(1)?.to_ascii_uppercase(),
            notnull: row.get::<_, i64>(2)? != 0,
            defaulted: default.is_some(),
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

fn insert_row(conn: &Connection, table: &str, provided: &[(&str, Value)]) -> anyhow::Result<()> {
    let info = table_info(conn, table)?;
    if info.is_empty() {
        bail!("{table} has no columns");
    }
    let given: HashMap<&str, &Value> = provided.iter().map(|(k, v)| (*k, v)).collect();
    let mut names = Vec::new();
    let mut values = Vec::new();
    for col in &info {
        if let Some(value) = given.get(col.name.as_str()) {
            names.push(col.name.clone());
            values.push((*value).clone());
            continue;
        }
        if col.notnull && !col.defaulted {
            names.push(col.name.clone());
            values.push(filler(&col.name, &col.ty));
        }
    }
    let placeholders = (1..=names.len())
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let quoted = names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("INSERT INTO \"{table}\" ({quoted}) VALUES ({placeholders})");
    conn.execute(&sql, rusqlite::params_from_iter(values.iter()))
        .with_context(|| format!("{sql}"))?;
    Ok(())
}

fn filler(name: &str, ty: &str) -> Value {
    if ty.contains("INT") {
        Value::Integer(0)
    } else if ty.contains("REAL") || ty.contains("FLOA") || ty.contains("DOUB") {
        Value::Real(0.0)
    } else if ty.contains("BLOB") {
        Value::Blob(vec![0; 16])
    } else if name.ends_with("_at") || name.contains("date") || name.contains("time") {
        Value::Text("2026-01-15T10:00:00+00:00".into())
    } else {
        Value::Text("fixture".into())
    }
}

fn text(value: &str) -> Value {
    Value::Text(value.to_string())
}

fn seed(conn: &Connection) -> anyhow::Result<()> {
    let nodes = [
        (
            "11111111-1111-4111-8111-111111111111",
            "Synthetic fact: migration fixtures are never real user data",
            "fact",
            None,
        ),
        (
            "22222222-2222-4222-8222-222222222222",
            "Synthetic fact: the v3 file stays byte-identical after migration",
            "fact",
            None,
        ),
        (
            "33333333-3333-4333-8333-333333333333",
            "Synthetic procedure: run migrate-to-strata once per store",
            "procedure",
            None,
        ),
        (
            "44444444-4444-4444-8444-444444444444",
            "Synthetic superseded note kept for lineage provenance",
            "note",
            Some("22222222-2222-4222-8222-222222222222"),
        ),
    ];
    for (id, content, node_type, superseded) in nodes {
        let mut row = vec![
            ("id", text(id)),
            ("content", text(content)),
            ("node_type", text(node_type)),
            ("created_at", text("2026-01-15T10:00:00+00:00")),
            ("updated_at", text("2026-02-20T11:30:00+00:00")),
            ("last_accessed", text("2026-03-01T09:15:00+00:00")),
            (
                "tags",
                text(if id.starts_with('1') {
                    r#"["fixture","synthetic"]"#
                } else {
                    "[]"
                }),
            ),
        ];
        if superseded.is_some() {
            row.push(("superseded_by", text(superseded.unwrap())));
        }
        insert_row(conn, "knowledge_nodes", &row)?;
    }

    let edges = [
        (
            "11111111-1111-4111-8111-111111111111",
            "22222222-2222-4222-8222-222222222222",
            0.9,
            "causal",
        ),
        (
            "22222222-2222-4222-8222-222222222222",
            "33333333-3333-4333-8333-333333333333",
            0.5,
            "semantic",
        ),
        (
            "11111111-1111-4111-8111-111111111111",
            "33333333-3333-4333-8333-333333333333",
            0.7,
            "touched",
        ),
    ];
    for (source, target, strength, link_type) in edges {
        insert_row(
            conn,
            "memory_connections",
            &[
                ("source_id", text(source)),
                ("target_id", text(target)),
                ("strength", Value::Real(strength)),
                ("link_type", text(link_type)),
                ("created_at", text("2026-01-16T08:00:00+00:00")),
                ("last_activated", text("2026-03-02T08:00:00+00:00")),
                ("activation_count", Value::Integer(2)),
            ],
        )?;
    }

    insert_row(
        conn,
        "fsrs_cards",
        &[
            ("memory_id", text("11111111-1111-4111-8111-111111111111")),
            ("difficulty", Value::Real(4.5)),
            ("stability", Value::Real(12.25)),
            ("state", text("review")),
            ("reps", Value::Integer(5)),
            ("lapses", Value::Integer(2)),
        ],
    )?;
    insert_row(
        conn,
        "sync_tombstones",
        &[
            ("table_name", text("knowledge_nodes")),
            ("row_id", text("99999999-9999-4999-8999-999999999999")),
            ("deleted_at", text("2026-02-01T00:00:00+00:00")),
            ("reason", text("fixture deletion")),
        ],
    )?;
    insert_row(
        conn,
        "deletion_tombstones",
        &[
            ("memory_id", text("88888888-8888-4888-8888-888888888888")),
            ("deleted_at", text("2026-02-02T00:00:00+00:00")),
            ("reason", text("fixture purge")),
            ("node_type", text("note")),
            ("tags", text("[]")),
        ],
    )?;
    for id in [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    ] {
        insert_row(
            conn,
            "node_embeddings",
            &[
                ("node_id", text(id)),
                ("embedding", Value::Blob(vec![0u8; 16])),
                ("dimensions", Value::Integer(4)),
                ("model", text("fixture-model")),
                ("created_at", text("2026-01-15T10:01:00+00:00")),
            ],
        )?;
    }
    seed_envelopes(conn)?;
    Ok(())
}

fn seed_envelopes(conn: &Connection) -> anyhow::Result<()> {
    use base64::Engine as _;
    use vestige_core::storage::receipt_attestation::{entry_digest, payload_digest};

    let envelopes: [(&str, i64, &[u8]); 2] = [
        (
            "aaaaaaa1-0000-4000-8000-000000000001",
            0,
            b"synthetic receipt payload zero",
        ),
        (
            "aaaaaaa1-0000-4000-8000-000000000002",
            1,
            b"synthetic receipt payload one",
        ),
    ];
    let mut prev_entry = String::new();
    for (receipt_id, sequence, payload) in envelopes {
        let payload_type = "https://vestige.dev/receipt/v1";
        let signature = [0xA5u8; 64];
        let key_id = "fixture-signing-key";
        let pd = payload_digest(payload);
        let ed = entry_digest(payload_type, payload, key_id, &signature);
        let previous = if sequence == 0 {
            Value::Null
        } else {
            text(&prev_entry)
        };
        let envelope = serde_json::json!({
            "payloadType": payload_type,
            "payload": base64::engine::general_purpose::STANDARD.encode(payload),
            "signatures": [{
                "keyid": key_id,
                "sig": base64::engine::general_purpose::STANDARD.encode(signature)
            }]
        });
        let fingerprint = "f".repeat(64);
        insert_row(
            conn,
            "receipt_envelopes",
            &[
                ("receipt_id", text(receipt_id)),
                ("chain_id", text("fixture-chain")),
                ("sequence", Value::Integer(sequence)),
                ("previous_entry_digest", previous),
                ("payload_type", text(payload_type)),
                ("envelope_json", text(&envelope.to_string())),
                ("payload_digest", text(&pd)),
                ("entry_digest", text(&ed)),
                ("signing_key_id", text(key_id)),
                ("signer_key_fingerprint", text(&fingerprint)),
                ("issued_at", text("2026-03-05T12:00:00+00:00")),
                ("stored_at", text("2026-03-05T12:00:00+00:00")),
            ],
        )?;
        prev_entry = ed;
    }
    Ok(())
}
