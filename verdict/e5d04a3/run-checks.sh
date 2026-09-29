#!/usr/bin/env bash
# Live verdict checks 1-7 against the e5d04a3 binaries.
# Does not modify /workspace. Artifacts land in /tmp/verdict-run.
set -u
ROOT=/tmp/verdict-run
LOG=$ROOT/logs
WORK=$ROOT/work
DBS=/tmp/verdict-inputs/dbs
VESTIGE=${VESTIGE:-/tmp/verdict-target/release/vestige}
MCP=${MCP:-/tmp/verdict-target/release/vestige-mcp}
VERIFY=${VERIFY:-/tmp/verdict-target/release/strata-verify}
HARNESS=${HARNESS:-$ROOT/harness/target/release/verdict-harness}
mkdir -p "$LOG" "$WORK"
cd "$ROOT"

stamp() { date -Is; }

run() {
  local name="$1"; shift
  echo "===== $name $(stamp) =====" | tee -a "$LOG/commands.txt"
  printf 'CMD:' | tee -a "$LOG/commands.txt"
  printf ' %q' "$@" | tee -a "$LOG/commands.txt"
  echo | tee -a "$LOG/commands.txt"
  set +e
  "$@" >"$LOG/$name.out" 2>"$LOG/$name.err"
  local ec=$?
  set -e
  echo "EXIT:$ec" | tee -a "$LOG/$name.out"
  echo "$ec" >"$LOG/$name.exit"
  echo "--- stdout ---" >>"$LOG/commands.txt"
  cat "$LOG/$name.out" >>"$LOG/commands.txt"
  echo "--- stderr ---" >>"$LOG/commands.txt"
  cat "$LOG/$name.err" >>"$LOG/commands.txt"
  echo "EXIT:$ec" >>"$LOG/commands.txt"
  return 0
}

hashdir() {
  local dir="$1" out="$2"
  : >"$out"
  if [[ -d "$dir" ]]; then
    find "$dir" -type f -print0 | sort -z | while IFS= read -r -d '' f; do
      sha256sum "$f"
    done >"$out"
  fi
}

sqlite_files() {
  find "$1" -type f \( -name '*.sqlite' -o -name '*.sqlite-*' -o -name '*.db' -o -name '*.db-*' -o -name 'vestige.db*' \) -print
}

isolate_env() {
  local home="$1"
  mkdir -p "$home"
  env -i \
    PATH="$PATH" \
    HOME="$home" \
    USER="${USER:-ubuntu}" \
    LANG="${LANG:-C.UTF-8}" \
    TMPDIR="${TMPDIR:-/tmp}" \
    RUST_LOG=warn \
    "${@:2}"
}

echo "CHECK SCRIPT START $(stamp)" | tee "$LOG/run-checks.start"

# Baseline hashes of the extracted stores. Never open these in place.
hashdir "$DBS" "$LOG/stores-baseline.sha256"
cp "$LOG/stores-baseline.sha256" "$LOG/stores-before-any-check.sha256"

# ---------------------------------------------------------------------------
# Check 1 — empty data dir, twice, shipped binary
# ---------------------------------------------------------------------------
C1=$WORK/check1
rm -rf "$C1"
mkdir -p "$C1/empty-cli" "$C1/empty-mcp" "$C1/home-cli" "$C1/home-mcp"
echo "CHECK1 $(stamp)" | tee "$LOG/check1.txt"

isolate_env "$C1/home-cli" "$VESTIGE" --data-dir "$C1/empty-cli" stats \
  >"$LOG/check1-cli-run1.out" 2>"$LOG/check1-cli-run1.err"
echo EXIT:$? >>"$LOG/check1-cli-run1.out"
echo "cli run1 exit $(tail -1 "$LOG/check1-cli-run1.out")" | tee -a "$LOG/check1.txt"
echo "--- stderr ---" >>"$LOG/check1.txt"
cat "$LOG/check1-cli-run1.err" >>"$LOG/check1.txt"

isolate_env "$C1/home-cli" "$VESTIGE" --data-dir "$C1/empty-cli" stats \
  >"$LOG/check1-cli-run2.out" 2>"$LOG/check1-cli-run2.err"
echo EXIT:$? >>"$LOG/check1-cli-run2.out"
echo "cli run2 exit $(tail -1 "$LOG/check1-cli-run2.out")" | tee -a "$LOG/check1.txt"
cat "$LOG/check1-cli-run2.err" >>"$LOG/check1.txt"

# MCP: one initialize line, then EOF. timeout so a hung server is visible.
isolate_env "$C1/home-mcp" timeout 20 "$MCP" --data-dir "$C1/empty-mcp" \
  <<<"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"verdict\",\"version\":\"1\"}}}" \
  >"$LOG/check1-mcp-run1.out" 2>"$LOG/check1-mcp-run1.err"
echo EXIT:$? >>"$LOG/check1-mcp-run1.out"
echo "mcp run1 exit $(tail -1 "$LOG/check1-mcp-run1.out")" | tee -a "$LOG/check1.txt"
cat "$LOG/check1-mcp-run1.err" >>"$LOG/check1.txt"

isolate_env "$C1/home-mcp" timeout 20 "$MCP" --data-dir "$C1/empty-mcp" \
  <<<"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"verdict\",\"version\":\"1\"}}}" \
  >"$LOG/check1-mcp-run2.out" 2>"$LOG/check1-mcp-run2.err"
echo EXIT:$? >>"$LOG/check1-mcp-run2.out"
echo "mcp run2 exit $(tail -1 "$LOG/check1-mcp-run2.out")" | tee -a "$LOG/check1.txt"
cat "$LOG/check1-mcp-run2.err" >>"$LOG/check1.txt"

echo "FILES under check1:" | tee -a "$LOG/check1.txt"
find "$C1" -print | tee -a "$LOG/check1.txt"
echo "SQLITE-LIKE:" | tee -a "$LOG/check1.txt"
sqlite_files "$C1" | tee -a "$LOG/check1.txt" || true

# ---------------------------------------------------------------------------
# Check 2 — existing v3 stores are refused or upgraded; bytes unchanged
# ---------------------------------------------------------------------------
echo "CHECK2 $(stamp)" | tee "$LOG/check2.txt"
C2=$WORK/check2
rm -rf "$C2"
mkdir -p "$C2"
for name in backfill-v31 demo-v38 fresh-v38-ckpt fresh-v38-wal probe-v38-ckpt probe-v38-wal vnc-v36; do
  cp -a "$DBS/$name" "$C2/$name"
  hashdir "$C2/$name" "$LOG/check2-$name.before.sha256"
  set +e
  isolate_env "$C2/home-$name" "$VESTIGE" --data-dir "$C2/$name" stats \
    >"$LOG/check2-$name-cli.out" 2>"$LOG/check2-$name-cli.err"
  echo EXIT:$? >>"$LOG/check2-$name-cli.out"
  isolate_env "$C2/home-$name" timeout 15 "$MCP" --data-dir "$C2/$name" \
    <<<"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"verdict\",\"version\":\"1\"}}}" \
    >"$LOG/check2-$name-mcp.out" 2>"$LOG/check2-$name-mcp.err"
  echo EXIT:$? >>"$LOG/check2-$name-mcp.out"
  hashdir "$C2/$name" "$LOG/check2-$name.after.sha256"
  if cmp -s "$LOG/check2-$name.before.sha256" "$LOG/check2-$name.after.sha256"; then
    same=UNCHANGED
  else
    same=CHANGED
  fi
  {
    echo "==== $name hash:$same cli:$(grep '^EXIT:' "$LOG/check2-$name-cli.out") mcp:$(grep '^EXIT:' "$LOG/check2-$name-mcp.out")"
    echo "-- cli stderr --"
    cat "$LOG/check2-$name-cli.err"
    echo "-- mcp stderr --"
    cat "$LOG/check2-$name-mcp.err"
    if [[ "$same" != UNCHANGED ]]; then
      diff -u "$LOG/check2-$name.before.sha256" "$LOG/check2-$name.after.sha256" || true
    fi
  } | tee -a "$LOG/check2.txt"
done

# ---------------------------------------------------------------------------
# Check 3 — import every real store
# ---------------------------------------------------------------------------
echo "CHECK3 $(stamp)" | tee "$LOG/check3.txt"
C3=$WORK/check3
rm -rf "$C3"
mkdir -p "$C3"
for name in $(ls -1 "$DBS"); do
  [[ -d "$DBS/$name" ]] || continue
  cp -a "$DBS/$name" "$C3/src-$name"
  hashdir "$C3/src-$name" "$LOG/check3-$name.before.sha256"
  wal_flag=()
  if [[ -f "$C3/src-$name/vestige.db-wal" ]] && [[ -s "$C3/src-$name/vestige.db-wal" ]]; then
    wal_flag=(--accept-wal-snapshot)
  fi
  mkdir -p "$C3/home-$name"
  set +e
  isolate_env "$C3/home-$name" "$VESTIGE" --data-dir "$C3/home-$name" \
    migrate-to-strata --from "$C3/src-$name" --to "$C3/to-$name" "${wal_flag[@]}" \
    >"$LOG/check3-$name-mig.out" 2>"$LOG/check3-$name-mig.err"
  echo EXIT:$? >>"$LOG/check3-$name-mig.out"
  if [[ -f "$C3/receipt-signing.key" ]]; then
    mkdir -p "$C3/keys"
    cp -a "$C3/receipt-signing.key" "$C3/keys/$name.receipt-signing.key"
    stat -c '%a %n' "$C3/keys/$name.receipt-signing.key" >>"$LOG/check3.txt"
  fi
  hashdir "$C3/src-$name" "$LOG/check3-$name.after.sha256"
  if cmp -s "$LOG/check3-$name.before.sha256" "$LOG/check3-$name.after.sha256"; then
    same=UNCHANGED
  else
    same=CHANGED
  fi
  echo "==== $name hash:$same exit:$(grep '^EXIT:' "$LOG/check3-$name-mig.out") wal:${wal_flag[*]:-no}" | tee -a "$LOG/check3.txt"
  cat "$LOG/check3-$name-mig.out" | tee -a "$LOG/check3.txt"
  echo "-- stderr --" | tee -a "$LOG/check3.txt"
  cat "$LOG/check3-$name-mig.err" | tee -a "$LOG/check3.txt"
  # skipped table names, one per line
  python3 - "$LOG/check3-$name-mig.out" "$LOG/check3-$name.skipped" <<'PY'
import sys
text=open(sys.argv[1]).read()
out=[]
for line in text.splitlines():
    if "Skipped tables" in line:
        part=line.split(":",1)[-1]
        out=[s.strip() for s in part.split(",") if s.strip()]
open(sys.argv[2],"w").write("\n".join(out)+"\n")
print("skipped", len(out))
PY
  if [[ -d "$C3/to-$name" ]]; then
    rm -rf "$C3/dump-$name"
    mkdir -p "$C3/dump-$name"
    # dump opens the log; do it on a copy so the sealed import stays pristine
    cp -a "$C3/to-$name" "$C3/dump-$name/log"
    # receipt key lives in the parent of --to
    if [[ -f "$C3/receipt-signing.key" ]]; then
      cp -a "$C3/receipt-signing.key" "$C3/dump-$name/receipt-signing.key" || true
    fi
    # each migration writes receipt-signing.key into the parent of --to.
    # --to is $C3/to-$name, parent is $C3, so the key is shared and overwritten.
    # Capture the key that existed immediately after this migration: it was
    # written to $C3/receipt-signing.key. Copy it next to the dump log.
    "$HARNESS" dump "$C3/dump-$name/log" >"$LOG/check3-$name.dump.json" 2>"$LOG/check3-$name.dump.err" || true
    python3 "$ROOT/compare.py" "$C3/src-$name" "$LOG/check3-$name.dump.json" "$LOG/check3-$name.skipped" \
      >"$LOG/check3-$name.compare.json" 2>"$LOG/check3-$name.compare.err"
    echo "compare_exit:$?" | tee -a "$LOG/check3.txt"
  else
    echo "NO_DEST compare_exit:skipped" | tee -a "$LOG/check3.txt"
  fi
done
hashdir "$DBS" "$LOG/stores-after-check3.sha256"
if cmp -s "$LOG/stores-before-any-check.sha256" "$LOG/stores-after-check3.sha256"; then
  echo "ORIGINAL_STORES_UNCHANGED" | tee -a "$LOG/check3.txt"
else
  echo "ORIGINAL_STORES_CHANGED" | tee -a "$LOG/check3.txt"
  diff -u "$LOG/stores-before-any-check.sha256" "$LOG/stores-after-check3.sha256" | head -40 | tee -a "$LOG/check3.txt" || true
fi

echo "CHECK SCRIPT FUNCTIONAL DONE $(stamp)" | tee -a "$LOG/run-checks.start"
