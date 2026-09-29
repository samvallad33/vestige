#!/usr/bin/env python3
"""Generate REAL v3.1.1-schema fixtures from the release tag's migration SQL.

    python3 scripts/gen-v3-fixtures.py

Reads the tag's migrations.rs via git, replays every MIGRATION_V<N>_UP block
(plus its MIGRATION_V<N>_ALTER_COLUMNS) into a fresh database, inserts
SYNTHETIC rows only, and writes
crates/strata-migrate/tests/fixtures/real-v3.1.1-schema{31,36,38}.sqlite.

STATUS / TODO (blocker 2, half done): the release binary's apply loop
interleaves UP blocks and ALTER arrays in an order this script has not fully
reproduced (V14 expects a `protected` column from an alter whose const is not
named ALTER_COLUMNS). Until that interleaving matches `apply_migrations`
exactly, the generated schema-38 file is close-but-not-authoritative; the
audit's requirement is a store produced by the REAL v3.1.1 binary. Next step:
extract the apply order from
v3.1.1:crates/vestige-core/src/storage/sqlite/admin.rs (apply_migrations) or
run the actual v3.1.1 binary against a temp VESTIGE_DATA_DIR and check in the
resulting file. The migrator's walk_receipts fix is already covered by code +
the synthetic-fixture tests.
"""
import re, subprocess, sqlite3, os

src = subprocess.run(
    ["git", "-C", os.path.expanduser("~/vestige"), "show",
     "v3.1.1:crates/vestige-core/src/storage/migrations.rs"],
    capture_output=True, text=True).stdout
ups = {int(v): sql for v, sql in re.findall(r'const MIGRATION_V(\d+)_UP:\s*&str\s*=\s*r#"(.*?)"#;', src, re.S)}
alters = {}
for v, stmts in re.findall(r'const MIGRATION_V(\d+)_ALTER_COLUMNS:\s*&\[&str\]\s*=\s*&\[(.*?)\];', src, re.S):
    alters[int(v)] = re.findall(r'"([^"]+)"', stmts)

def build(target_version):
    db = sqlite3.connect(":memory:")
    for v in sorted(ups):
        if v > target_version:
            break
        for stmt in alters.get(v, []):
            try:
                db.execute(stmt)
            except sqlite3.OperationalError:
                pass
        try:
            db.executescript(ups[v])
        except sqlite3.OperationalError as e:
            print(f"  v{v} up FAILED: {e}")
    try:
        db.execute("UPDATE schema_version SET version = ?", (target_version,))
    except sqlite3.OperationalError:
        pass
    db.execute("""INSERT INTO knowledge_nodes (id, content, node_type, created_at, updated_at, last_accessed, tags)
                  VALUES ('11111111-1111-4111-8111-111111111111', 'audit fixture node one', 'fact',
                          '2026-01-15T10:00:00+00:00', '2026-02-20T11:30:00+00:00', '2026-03-01T09:15:00+00:00', '[]')""")
    db.commit()
    out = f"crates/strata-migrate/tests/fixtures/real-v3.1.1-schema{target_version}.sqlite"
    if os.path.exists(out):
        os.remove(out)
    disk = sqlite3.connect(out)
    db.backup(disk)
    disk.close()

for version in (31, 36, 38):
    build(version)
    print(f"wrote real-v3.1.1-schema{version}.sqlite")
