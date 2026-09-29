#!/usr/bin/env bash
# Live verdict runner for commit e5d04a3. Does not modify the source tree
# except writing verdict/logs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/verdict/logs"
mkdir -p "$LOG"
WORK=/tmp/verdict-work
rm -rf "$WORK"
mkdir -p "$WORK"
START_EPOCH=$(date +%s)
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG/00-meta.txt"
echo "head=$(git -C "$ROOT" rev-parse HEAD)" | tee -a "$LOG/00-meta.txt"
rustc --version | tee -a "$LOG/00-meta.txt"

VESTIGE="${VESTIGE:-/tmp/cargo-target/debug/vestige}"
MCP="${MCP:-/tmp/cargo-target/debug/vestige-mcp}"
VERIFY="${VERIFY:-/tmp/verify-target/debug/strata-verify}"
HARNESS="${HARNESS:-/tmp/harness-target/release/verdict-harness}"
V311="${V311:-/tmp/v311-bin/vestige}"
FIX="${FIX:-$ROOT/crates/strata-migrate/tests/fixtures/v3.1.1-sample.sqlite}"

for b in "$VESTIGE" "$MCP" "$VERIFY" "$HARNESS"; do
  if [[ ! -x "$b" ]]; then
    echo "MISSING binary $b" | tee "$LOG/MISSING.txt"
    exit 2
  fi
done

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

list_db_files() {
  local dir="$1"
  find "$dir" -type f \( -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' -o -name '*-wal' -o -name '*-shm' -o -name '*.db-journal' \) -printf '%p\n' 2>/dev/null | sort
}

# ---------------------------------------------------------------------------
# Prepare stores
# ---------------------------------------------------------------------------
python3 - <<'PY' "$WORK" "$FIX" /tmp/v311-work/data/vestige.db
import os, shutil, sqlite3, sys
work, fixture, v311 = sys.argv[1:]
os.makedirs(f"{work}/stores", exist_ok=True)

def checkpoint_copy(src, dst):
    shutil.copy2(src, dst)
    con = sqlite3.connect(dst)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.commit()
    con.close()
    for side in (dst+"-wal", dst+"-shm"):
        if os.path.exists(side):
            os.remove(side)

# 1. synthetic fixture (has walk_receipts, schema 38, thin columns)
shutil.copy2(fixture, f"{work}/stores/fixture.db")

# 2. fixture with walk_receipts removed
shutil.copy2(fixture, f"{work}/stores/fixture-no-walk.db")
con = sqlite3.connect(f"{work}/stores/fixture-no-walk.db")
con.execute("DROP TABLE IF EXISTS walk_receipts")
con.commit(); con.close()

# 3. real v3.1.1 binary store: 3 ingested facts, semantic edge, fsrs row
if os.path.exists(v311):
    dst = f"{work}/stores/v311-real.db"
    checkpoint_copy(v311, dst)
    con = sqlite3.connect(dst)
    ids = [r[0] for r in con.execute("SELECT id FROM knowledge_nodes ORDER BY created_at")]
    if len(ids) >= 2:
        con.execute(
            "INSERT INTO memory_connections (source_id, target_id, strength, link_type, created_at, last_activated, activation_count) VALUES (?,?,?,?,?,?,?)",
            (ids[0], ids[1], 0.91, "semantic", "2026-09-28T00:00:00Z", "2026-09-28T00:00:00Z", 1),
        )
        con.execute(
            "INSERT INTO memory_connections (source_id, target_id, strength, link_type, created_at, last_activated, activation_count) VALUES (?,?,?,?,?,?,?)",
            (ids[1], ids[0], 0.4, "similarity", "2026-09-28T00:00:01Z", "2026-09-28T00:00:01Z", 0),
        )
        con.execute(
            "UPDATE knowledge_nodes SET stability=?, difficulty=?, reps=?, lapses=?, learning_state=?, scope=?, source=? WHERE id=?",
            (18.5, 6.25, 7, 2, "review", "verdict-scope", "verdict-harness", ids[0]),
        )
        con.execute(
            "INSERT INTO fsrs_cards (memory_id, difficulty, stability, state, reps, lapses, last_review, due_date, elapsed_days, scheduled_days) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ids[0], 6.25, 18.5, "review", 7, 2, "2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z", 4, 21),
        )
    con.commit(); con.close()

    # 4. WAL sibling: extra row lives only in the WAL.
    # os._exit in a child skips sqlite3's connection destructor, which
    # would otherwise checkpoint the WAL on close.
    wal = f"{work}/stores/v311-wal.db"
    shutil.copy2(dst, wal)
    pid = os.fork()
    if pid == 0:
        con = sqlite3.connect(wal)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            "INSERT INTO knowledge_nodes (id, content, node_type, created_at, updated_at, last_accessed) VALUES (?,?,?,?,?,?)",
            ("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", "WAL_ONLY_ROW_NOT_IN_MAIN", "fact",
             "2026-09-29T00:00:00Z", "2026-09-29T00:00:00Z", "2026-09-29T00:00:00Z"),
        )
        con.commit()
        os._exit(0)
    os.waitpid(pid, 0)
else:
    open(f"{work}/stores/NO_V311", "w").write("v3.1.1 store missing\n")
print("prepared", os.listdir(f"{work}/stores"))
PY
echo "prepared stores" | tee "$LOG/00-stores.txt"
find "$WORK/stores" -type f -printf '%s %p\n' | tee -a "$LOG/00-stores.txt"
(cd "$WORK/stores" && sha256sum * | tee -a "$LOG/00-stores.txt")

# ---------------------------------------------------------------------------
# CHECK 1 — empty data dir, start twice
# ---------------------------------------------------------------------------
c1() {
  local name="$1" bin="$2"
  local dir="$WORK/c1-$name"
  mkdir -p "$dir/data"
  {
    echo "CMD: env -u VESTIGE_DATA_DIR HOME=$WORK/home XDG_DATA_HOME=$WORK/home/.local/share timeout 8 $bin --data-dir $dir/data"
    mkdir -p "$WORK/home"
    set +e
    env -u VESTIGE_DATA_DIR HOME="$WORK/home" XDG_DATA_HOME="$WORK/home/.local/share" \
      timeout 8 "$bin" --data-dir "$dir/data" >"$dir/out1.txt" 2>"$dir/err1.txt"
    echo "start1_exit=$?"
    echo "--- stderr start1 ---"
    cat "$dir/err1.txt"
    echo "--- stdout start1 ---"
    cat "$dir/out1.txt"
    echo "files after start1:"
    list_db_files "$dir" || true
    list_db_files "$WORK/home" || true
    env -u VESTIGE_DATA_DIR HOME="$WORK/home" XDG_DATA_HOME="$WORK/home/.local/share" \
      timeout 8 "$bin" --data-dir "$dir/data" >"$dir/out2.txt" 2>"$dir/err2.txt"
    echo "start2_exit=$?"
    echo "--- stderr start2 ---"
    cat "$dir/err2.txt"
    echo "files after start2:"
    list_db_files "$dir" || true
    find "$dir" -type f -printf '%p\n' | sort
    set -e
  } >"$LOG/01-$name.txt" 2>&1
}
# vestige-mcp with no subcommand blocks on stdio only AFTER storage init.
# vestige CLI requires a subcommand; use stats, which opens storage.
{
  echo "CMD mcp (no subcommand)"
} 
c1 mcp "$MCP"
# CLI stats
{
  dir="$WORK/c1-cli"
  mkdir -p "$dir/data" "$WORK/home"
  {
    echo "CMD: timeout 8 $VESTIGE --data-dir $dir/data stats"
    set +e
    env -u VESTIGE_DATA_DIR HOME="$WORK/home" XDG_DATA_HOME="$WORK/home/.local/share" \
      timeout 8 "$VESTIGE" --data-dir "$dir/data" stats >"$dir/out1.txt" 2>"$dir/err1.txt"
    echo "start1_exit=$?"
    echo "--- stderr ---"; cat "$dir/err1.txt"
    echo "--- stdout ---"; cat "$dir/out1.txt"
    echo "files after start1:"; list_db_files "$dir" || true
    env -u VESTIGE_DATA_DIR HOME="$WORK/home" XDG_DATA_HOME="$WORK/home/.local/share" \
      timeout 8 "$VESTIGE" --data-dir "$dir/data" stats >"$dir/out2.txt" 2>"$dir/err2.txt"
    echo "start2_exit=$?"
    echo "--- stderr2 ---"; cat "$dir/err2.txt"
    echo "files after start2:"; find "$dir" -type f -printf '%p\n' | sort
    set -e
  } >"$LOG/01-cli-stats.txt" 2>&1
}

# ---------------------------------------------------------------------------
# CHECK 2 — existing v3 store
# ---------------------------------------------------------------------------
c2_one() {
  local label="$1" src="$2"
  local dir="$WORK/c2-$label"
  mkdir -p "$dir/data"
  cp -a "$src" "$dir/data/vestige.db"
  local before after
  before=$(sha256_file "$dir/data/vestige.db")
  local mode_before
  mode_before=$(stat -c '%a' "$dir/data/vestige.db")
  {
    echo "label=$label"
    echo "sha256_before=$before"
    echo "mode_before=$mode_before"
    set +e
    env -u VESTIGE_DATA_DIR timeout 10 "$MCP" --data-dir "$dir/data" >"$dir/mcp.out" 2>"$dir/mcp.err"
    echo "mcp_exit=$?"
    echo "--- mcp stderr ---"; cat "$dir/mcp.err"
    env -u VESTIGE_DATA_DIR timeout 10 "$VESTIGE" --data-dir "$dir/data" stats >"$dir/cli.out" 2>"$dir/cli.err"
    echo "cli_exit=$?"
    echo "--- cli stderr ---"; cat "$dir/cli.err"
    echo "--- cli stdout ---"; cat "$dir/cli.out"
    set -e
    after=$(sha256_file "$dir/data/vestige.db")
    echo "sha256_after=$after"
    echo "mode_after=$(stat -c '%a' "$dir/data/vestige.db")"
    echo "sidecars:"; find "$dir/data" -maxdepth 1 -type f -printf '%f\n' | sort
  } >"$LOG/02-$label.txt" 2>&1
}
c2_one fixture "$WORK/stores/fixture.db"
if [[ -f "$WORK/stores/v311-real.db" ]]; then
  c2_one v311 "$WORK/stores/v311-real.db"
fi

# ---------------------------------------------------------------------------
# CHECK 3 — import
# ---------------------------------------------------------------------------
import_one() {
  local label="$1" src="$2" walflag="${3:-}"
  local dir="$WORK/c3-$label"
  mkdir -p "$dir"
  cp -a "$src" "$dir/source.db"
  # copy sidecars if the source is a wal db
  if [[ -f "${src}-wal" ]]; then cp -a "${src}-wal" "$dir/source.db-wal"; fi
  if [[ -f "${src}-shm" ]]; then cp -a "${src}-shm" "$dir/source.db-shm"; fi
  local before
  before=$(sha256_file "$dir/source.db")
  local wal_before="" shm_before=""
  [[ -f "$dir/source.db-wal" ]] && wal_before=$(sha256_file "$dir/source.db-wal")
  [[ -f "$dir/source.db-shm" ]] && shm_before=$(sha256_file "$dir/source.db-shm")
  {
    echo "label=$label walflag=${walflag:-none}"
    echo "sha256_before=$before"
    echo "wal_sha_before=$wal_before"
    echo "shm_sha_before=$shm_before"
    echo "sqlite_counts:"
    python3 - "$dir/source.db" <<'PY'
import sqlite3, sys
p = sys.argv[1]
con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
def count(table):
    row = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if not row or row[0] == 0:
        print(f"  {table}: ABSENT")
        return
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {n}")
for t in ("knowledge_nodes","memory_connections","fsrs_cards","walk_receipts","sync_tombstones","deletion_tombstones","node_embeddings"):
    count(t)
print("  schema", con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0])
print("  link_types:")
if con.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='memory_connections'").fetchone()[0]:
    for r in con.execute("SELECT link_type, COUNT(*) FROM memory_connections GROUP BY 1"):
        print(f"    {r[0]} {r[1]}")
PY
    set +e
    "$VESTIGE" migrate-to-strata --from "$dir/source.db" --to "$dir/strata" $walflag >"$dir/mig.out" 2>"$dir/mig.err"
    echo "migrate_exit=$?"
    echo "--- migrate stdout ---"; cat "$dir/mig.out"
    echo "--- migrate stderr ---"; cat "$dir/mig.err"
    set -e
    echo "sha256_after=$(sha256_file "$dir/source.db")"
    if [[ -f "$dir/source.db-wal" ]]; then echo "wal_sha_after=$(sha256_file "$dir/source.db-wal")"; fi
    if [[ -f "$dir/source.db-shm" ]]; then echo "shm_sha_after=$(sha256_file "$dir/source.db-shm")"; fi
    echo "dest files:"; find "$dir" -type f -printf '%p %s\n' | sort
    if [[ -d "$dir/strata" ]]; then
      "$HARNESS" dump "$dir/strata" >"$dir/dump.json" 2>"$dir/dump.err" || echo "dump_exit=$?"
      echo "--- dump stderr ---"; cat "$dir/dump.err"
      echo "--- dump ---"; cat "$dir/dump.json"
    fi
  } >"$LOG/03-$label.txt" 2>&1
}

import_one fixture "$WORK/stores/fixture.db"
import_one fixture-no-walk "$WORK/stores/fixture-no-walk.db"
if [[ -f "$WORK/stores/v311-real.db" ]]; then
  import_one v311-real "$WORK/stores/v311-real.db"
  import_one v311-wal-nofail "$WORK/stores/v311-wal.db"
  import_one v311-wal-snap "$WORK/stores/v311-wal.db" "--accept-wal-snapshot"
fi

# attachment corpus
if [[ -d "$ROOT/uploads" ]] || [[ -f "$ROOT/uploads/v3-real-stores.tar.gz" ]]; then
  echo "uploads present" >"$LOG/03-uploads.txt"
else
  echo "BLOCKED: uploads/v3-real-stores.tar.gz is not on this machine. Searched /workspace/uploads and the filesystem." >"$LOG/03-uploads.txt"
fi

# ---------------------------------------------------------------------------
# CHECK 4 — strata-verify on imported log and live store
# ---------------------------------------------------------------------------
{
  echo "CMD: $VERIFY $WORK/c3-fixture/strata"
  set +e
  "$VERIFY" "$WORK/c3-fixture/strata" >"$WORK/c4-import.out" 2>"$WORK/c4-import.err"
  echo "import_verify_exit=$?"
  echo "--- stdout ---"; cat "$WORK/c4-import.out"
  echo "--- stderr ---"; cat "$WORK/c4-import.err"
  if [[ -d "$WORK/c3-v311-real/strata" ]]; then
    echo "CMD: $VERIFY v311-real"
    "$VERIFY" "$WORK/c3-v311-real/strata" >"$WORK/c4-v311.out" 2>"$WORK/c4-v311.err"
    echo "v311_verify_exit=$?"
    echo "--- stdout ---"; cat "$WORK/c4-v311.out"
    echo "--- stderr ---"; cat "$WORK/c4-v311.err"
  fi
  set -e
} >"$LOG/04-imported-log.txt" 2>&1

{
  echo "CMD: harness live + strata-verify"
  set +e
  "$HARNESS" live "$WORK/c4-live" >"$WORK/c4-live.json" 2>"$WORK/c4-live.err"
  echo "live_exit=$?"
  echo "--- live ---"; cat "$WORK/c4-live.json"; echo; cat "$WORK/c4-live.err"
  echo "entries before verify:"; find "$WORK/c4-live" -printf '%p\n' | sort
  "$VERIFY" "$WORK/c4-live" >"$WORK/c4-live-verify.out" 2>"$WORK/c4-live-verify.err"
  echo "live_verify_exit=$?"
  echo "--- verify stdout ---"; cat "$WORK/c4-live-verify.out"
  echo "--- verify stderr ---"; cat "$WORK/c4-live-verify.err"
  echo "entries after verify:"; find "$WORK/c4-live" -printf '%p\n' | sort
  if [[ -d "$WORK/c4-live/log" ]]; then
    "$VERIFY" "$WORK/c4-live/log" >"$WORK/c4-livelog.out" 2>"$WORK/c4-livelog.err"
    echo "livelog_verify_exit=$?"
    echo "--- logdir stdout ---"; cat "$WORK/c4-livelog.out"
    echo "--- logdir stderr ---"; cat "$WORK/c4-livelog.err"
  fi
  set -e
} >"$LOG/04-live-store.txt" 2>&1

# ---------------------------------------------------------------------------
# CHECK 5 — tamper
# ---------------------------------------------------------------------------
{
  src="$WORK/c3-fixture/strata"
  echo "sealed-byte flip"
  rm -rf "$WORK/c5-flip"
  cp -a "$src" "$WORK/c5-flip"
  python3 - <<'PY'
import os, pathlib
d = pathlib.Path("/tmp/verdict-work/c5-flip")
segs = sorted(d.glob("*.seg"), key=lambda p: p.stat().st_size)
# Flip inside the largest (sealed, framed) segment, past the header.
target = segs[-1]
data = bytearray(target.read_bytes())
idx = 90 if len(data) > 100 else len(data)//2
data[idx] ^= 0xFF
target.write_bytes(data)
print(f"flipped {target.name} at {idx} size {len(data)}")
PY
  set +e
  "$VERIFY" "$WORK/c5-flip" >"$WORK/c5-flip.out" 2>"$WORK/c5-flip.err"
  echo "flip_verify_exit=$?"
  echo "--- stdout ---"; cat "$WORK/c5-flip.out"
  echo "--- stderr ---"; cat "$WORK/c5-flip.err"
  set -e

  echo "replaced / recomputed key"
  rm -rf "$WORK/c5-key"
  cp -a "$src" "$WORK/c5-key"
  "$HARNESS" forge-replaced-key "$WORK/c5-key" >"$WORK/c5-key.json" 2>"$WORK/c5-key.err" || echo "forge_key_harness_exit=$?"
  echo "--- forge key ---"; cat "$WORK/c5-key.json"; echo; cat "$WORK/c5-key.err"
  # Leave a replaced key in place and run strata-verify (harness restores the key).
  python3 - <<'PY'
from pathlib import Path
p = Path("/tmp/verdict-work/c5-key/strata.key")
p.write_bytes(b"\x11"*32)
print("wrote replacement strata.key")
PY
  set +e
  "$VERIFY" "$WORK/c5-key" >"$WORK/c5-key-verify.out" 2>"$WORK/c5-key-verify.err"
  echo "replaced_key_verify_exit=$?"
  echo "--- stdout ---"; cat "$WORK/c5-key-verify.out"
  echo "--- stderr ---"; cat "$WORK/c5-key-verify.err"
  set -e

  echo "resigned receipt with a new key"
  rm -rf "$WORK/c5-resign"
  cp -a "$src" "$WORK/c5-resign"
  "$HARNESS" forge-resigned-receipt "$WORK/c5-resign" >"$WORK/c5-resign.json" 2>"$WORK/c5-resign.err" || echo "resign_harness_exit=$?"
  echo "--- resign ---"; cat "$WORK/c5-resign.json"; echo; cat "$WORK/c5-resign.err"
  set +e
  "$VERIFY" "$WORK/c5-resign" >"$WORK/c5-resign-verify.out" 2>"$WORK/c5-resign-verify.err"
  echo "resign_verify_exit=$?"
  echo "--- stdout ---"; cat "$WORK/c5-resign-verify.out"
  echo "--- stderr ---"; cat "$WORK/c5-resign-verify.err"
  set -e
} >"$LOG/05-tamper.txt" 2>&1

# ---------------------------------------------------------------------------
# CHECK 6 — SIGKILL mid-import, then rerun; non-empty --to
# ---------------------------------------------------------------------------
gcc -shared -fPIC -o "$WORK/slowwrite.so" "$ROOT/verdict/slowwrite.c" -ldl
{
  src="$WORK/stores/fixture.db"
  rm -rf "$WORK/c6"
  mkdir -p "$WORK/c6"
  cp -a "$src" "$WORK/c6/source.db"
  echo "slow import then SIGKILL"
  set +e
  LD_PRELOAD="$WORK/slowwrite.so" "$VESTIGE" migrate-to-strata --from "$WORK/c6/source.db" --to "$WORK/c6/partial" >"$WORK/c6/partial.out" 2>"$WORK/c6/partial.err" &
  pid=$!
  echo "pid=$pid"
  for i in $(seq 1 80); do
    if find "$WORK/c6/partial" -name '*.seg' -size +400c 2>/dev/null | grep -q .; then
      echo "seg appeared at iter $i"
      kill -9 "$pid" 2>/dev/null || true
      echo "sent SIGKILL"
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "process exited before kill"
      break
    fi
    sleep 0.1
  done
  wait "$pid"
  echo "killed_exit=$?"
  echo "partial tree:"; find "$WORK/c6/partial" -type f -printf '%s %p\n' | sort
  echo "--- partial stdout ---"; cat "$WORK/c6/partial.out"
  echo "--- partial stderr ---"; cat "$WORK/c6/partial.err"
  echo "RERUN into same --to"
  "$VESTIGE" migrate-to-strata --from "$WORK/c6/source.db" --to "$WORK/c6/partial" >"$WORK/c6/rerun.out" 2>"$WORK/c6/rerun.err"
  echo "rerun_exit=$?"
  echo "--- rerun stdout ---"; cat "$WORK/c6/rerun.out"
  echo "--- rerun stderr ---"; cat "$WORK/c6/rerun.err"
  if [[ -d "$WORK/c6/partial" ]]; then
    "$HARNESS" dump "$WORK/c6/partial" >"$WORK/c6/dump.json" 2>"$WORK/c6/dump.err" || echo "dump_exit=$?"
    echo "--- dump ---"; cat "$WORK/c6/dump.json"; echo; cat "$WORK/c6/dump.err"
  fi
  echo "NON-EMPTY --to"
  mkdir -p "$WORK/c6/nonempty"
  echo unrelated >"$WORK/c6/nonempty/keep.txt"
  "$VESTIGE" migrate-to-strata --from "$WORK/c6/source.db" --to "$WORK/c6/nonempty" >"$WORK/c6/nonempty.out" 2>"$WORK/c6/nonempty.err"
  echo "nonempty_exit=$?"
  echo "--- nonempty stdout ---"; cat "$WORK/c6/nonempty.out"
  echo "--- nonempty stderr ---"; cat "$WORK/c6/nonempty.err"
  echo "nonempty files:"; find "$WORK/c6/nonempty" -type f -printf '%p\n' | sort
  echo "IDEMPOTENT rerun of a finished import"
  "$VESTIGE" migrate-to-strata --from "$WORK/c3-fixture/source.db" --to "$WORK/c3-fixture/strata" >"$WORK/c6/idem.out" 2>"$WORK/c6/idem.err"
  echo "idempotent_exit=$?"
  echo "--- idem stdout ---"; cat "$WORK/c6/idem.out"
  echo "--- idem stderr ---"; cat "$WORK/c6/idem.err"
  set -e
} >"$LOG/06-kill.txt" 2>&1

# ---------------------------------------------------------------------------
# CHECK 7 — lane E (expected fail, does not block)
# ---------------------------------------------------------------------------
{
  "$HARNESS" check7-truncate "$WORK/c7-truncate" 
  echo
  "$HARNESS" check7-key "$WORK/c7-key"
  echo
  "$HARNESS" check7-meta "$WORK/c7-meta"
} >"$LOG/07-lane-e.txt" 2>&1

# ---------------------------------------------------------------------------
# SQLite linkage of the shipped default-features binary
# ---------------------------------------------------------------------------
{
  echo "binary=$VESTIGE"
  file "$VESTIGE" || true
  echo "==== ldd vestige ===="
  ldd "$VESTIGE" || true
  echo "==== ldd vestige-mcp ===="
  ldd "$MCP" || true
  echo "==== nm sqlite symbols vestige (defined+undef) ===="
  nm -a "$VESTIGE" 2>/dev/null | rg -i "sqlite" | head -40 || true
  echo "==== nm -D dynamic sqlite ===="
  nm -D "$VESTIGE" 2>/dev/null | rg -i "sqlite" | head -20 || true
  echo "==== cargo tree rusqlite inverted (vestige-mcp) ===="
  cd "$ROOT"
  CARGO_TARGET_DIR=/tmp/cargo-target cargo tree -p vestige-mcp -i rusqlite --edges normal --prefix none 2>&1 | head -80
} >"$LOG/09-sqlite-linkage.txt" 2>&1

END_EPOCH=$(date +%s)
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/00-meta.txt"
echo "elapsed_sec=$((END_EPOCH-START_EPOCH))" | tee -a "$LOG/00-meta.txt"
echo "DONE" | tee -a "$LOG/00-meta.txt"
