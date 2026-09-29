#!/usr/bin/env bash
# Checks 4-7 plus SQLite linkage. Expects check 3 artifacts when present.
set -u
ROOT=/tmp/verdict-run
LOG=$ROOT/logs
WORK=$ROOT/work
DBS=/tmp/verdict-inputs/dbs
VESTIGE=${VESTIGE:-/tmp/verdict-target/release/vestige}
VERIFY=${VERIFY:-/tmp/verdict-target/release/strata-verify}
HARNESS=${HARNESS:-$ROOT/harness/target/release/verdict-harness}
mkdir -p "$LOG" "$WORK"

isolate_env() {
  local home="$1"; shift
  mkdir -p "$home"
  env -i PATH="$PATH" HOME="$home" USER="${USER:-ubuntu}" LANG="${LANG:-C.UTF-8}" \
    TMPDIR="${TMPDIR:-/tmp}" RUST_LOG=warn "$@"
}

hashdir() {
  local dir="$1" out="$2"
  find "$dir" -type f -print0 | sort -z | while IFS= read -r -d '' f; do
    sha256sum "$f"
  done >"$out"
}

echo "CHECK4-7 START $(date -Is)" | tee "$LOG/check4-7.txt"

# ---------------------------------------------------------------------------
# Check 6 first (uses vestige only) so it is not blocked on strata-verify
# ---------------------------------------------------------------------------
C6=$WORK/check6
rm -rf "$C6"
mkdir -p "$C6/src" "$C6/home"
cp -a "$DBS/fresh-v38-ckpt/." "$C6/src/"
hashdir "$C6/src" "$LOG/check6-src.before.sha256"

echo "==== full migrate timing ====" | tee -a "$LOG/check4-7.txt"
start=$(date +%s%3N)
set +e
isolate_env "$C6/home" "$VESTIGE" --data-dir "$C6/home" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-full" \
  >"$LOG/check6-full.out" 2>"$LOG/check6-full.err"
echo EXIT:$? >>"$LOG/check6-full.out"
# set -e disabled: collect every result
end=$(date +%s%3N)
echo "full_migrate_ms:$((end-start)) exit:$(grep '^EXIT:' "$LOG/check6-full.out")" | tee -a "$LOG/check4-7.txt"
cat "$LOG/check6-full.out" | tee -a "$LOG/check4-7.txt"

echo "==== SIGKILL mid-import ====" | tee -a "$LOG/check4-7.txt"
rm -rf "$C6/to-kill" "$C6/home-kill"
mkdir -p "$C6/home-kill"
python3 - "$VESTIGE" "$C6" <<'PY' | tee "$LOG/check6-kill.txt"
import os, signal, subprocess, time, sys
vestige, c6 = sys.argv[1], sys.argv[2]
dest = os.path.join(c6, "to-kill")
home = os.path.join(c6, "home-kill")
env = {k: os.environ[k] for k in ("PATH", "LANG", "TMPDIR") if k in os.environ}
env.update(HOME=home, RUST_LOG="warn", USER=os.environ.get("USER", "ubuntu"))
cmd = [vestige, "--data-dir", home, "migrate-to-strata", "--from", os.path.join(c6, "src"), "--to", dest]

def preload():
    so = os.path.join(c6, "slowwrite.so")
    if os.path.exists(so):
        return so
    src = os.path.join(c6, "slowwrite.c")
    open(src, "w").write(r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>
static void pause_us(void) { usleep(3000); }
ssize_t write(int fd, const void *buf, size_t n) {
  static ssize_t (*real_write)(int, const void *, size_t);
  if (!real_write) real_write = dlsym(RTLD_NEXT, "write");
  pause_us();
  return real_write(fd, buf, n);
}
ssize_t pwrite64(int fd, const void *buf, size_t n, off_t off) {
  static ssize_t (*real_pwrite)(int, const void *, size_t, off_t);
  if (!real_pwrite) real_pwrite = dlsym(RTLD_NEXT, "pwrite64");
  pause_us();
  return real_pwrite(fd, buf, n, off);
}
''')
    r = subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", so, src, "-ldl"],
                       capture_output=True, text=True)
    print("preload_gcc", r.returncode, r.stderr[-400:])
    return so if r.returncode == 0 else None

def attempt(use_preload):
    if os.path.exists(dest):
        subprocess.check_call(["rm", "-rf", dest])
    run = list(cmd)
    child_env = dict(env)
    if use_preload:
        so = preload()
        if so:
            child_env["LD_PRELOAD"] = so
            print("using", so)
    # Own session so SIGKILL of the group cannot hit this driver.
    p = subprocess.Popen(run, env=child_env, start_new_session=True,
                         stdout=open(os.path.join(c6, "kill.out"), "w"),
                         stderr=open(os.path.join(c6, "kill.err"), "w"))
    killed = False
    t0 = time.time()
    while time.time() - t0 < 120:
        hit = None
        if os.path.isdir(dest):
            for root, dirs, files in os.walk(dest):
                for f in files:
                    if f.endswith(".seg"):
                        path = os.path.join(root, f)
                        sz = os.path.getsize(path)
                        if sz > 800:
                            hit = (sz, path)
                            break
                if hit:
                    break
        if hit:
            time.sleep(0.05)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            killed = True
            break
        if p.poll() is not None:
            break
        time.sleep(0.001)
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    return killed, p.returncode

killed, code = attempt(False)
print(f"attempt1 killed={killed} code={code}")
if not killed:
    killed, code = attempt(True)
    print(f"attempt2_preload killed={killed} code={code}")
print("dest_listing:")
if os.path.isdir(dest):
    for root, dirs, files in os.walk(dest):
        for f in files:
            path = os.path.join(root, f)
            print(f"  {os.path.getsize(path):8d} {path}")
else:
    print("  (no dest)")
open(os.path.join(c6, "killed.flag"), "w").write("1\n" if killed else "0\n")
PY

echo "==== rerun into killed dest ====" | tee -a "$LOG/check4-7.txt"
set +e
isolate_env "$C6/home-rerun" "$VESTIGE" --data-dir "$C6/home-rerun" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-kill" \
  >"$LOG/check6-rerun.out" 2>"$LOG/check6-rerun.err"
echo EXIT:$? >>"$LOG/check6-rerun.out"
# set -e disabled: collect every result
echo "rerun exit:$(grep '^EXIT:' "$LOG/check6-rerun.out")" | tee -a "$LOG/check4-7.txt"
cat "$LOG/check6-rerun.out" | tee -a "$LOG/check4-7.txt"
echo "-- stderr --" | tee -a "$LOG/check4-7.txt"
cat "$LOG/check6-rerun.err" | tee -a "$LOG/check4-7.txt"
hashdir "$C6/src" "$LOG/check6-src.after.sha256"
if cmp -s "$LOG/check6-src.before.sha256" "$LOG/check6-src.after.sha256"; then
  echo "KILL_SOURCE_UNCHANGED" | tee -a "$LOG/check4-7.txt"
else
  echo "KILL_SOURCE_CHANGED" | tee -a "$LOG/check4-7.txt"
fi

echo "==== non-empty --to ====" | tee -a "$LOG/check4-7.txt"
# foreign file
mkdir -p "$C6/to-foreign"
echo junk >"$C6/to-foreign/note.txt"
set +e
isolate_env "$C6/home-foreign" "$VESTIGE" --data-dir "$C6/home-foreign" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-foreign" \
  >"$LOG/check6-foreign.out" 2>"$LOG/check6-foreign.err"
echo EXIT:$? >>"$LOG/check6-foreign.out"
# second store into a completed dest
cp -a "$DBS/vnc-v36" "$C6/src-vnc"
isolate_env "$C6/home-second" "$VESTIGE" --data-dir "$C6/home-second" \
  migrate-to-strata --from "$C6/src-vnc" --to "$C6/to-full" \
  >"$LOG/check6-second.out" 2>"$LOG/check6-second.err"
echo EXIT:$? >>"$LOG/check6-second.out"
# same source into the completed dest
isolate_env "$C6/home-same" "$VESTIGE" --data-dir "$C6/home-same" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-full" \
  >"$LOG/check6-same.out" 2>"$LOG/check6-same.err"
echo EXIT:$? >>"$LOG/check6-same.out"
# set -e disabled: collect every result
{
  echo "foreign exit:$(grep '^EXIT:' "$LOG/check6-foreign.out")"
  cat "$LOG/check6-foreign.err"
  echo "second-store exit:$(grep '^EXIT:' "$LOG/check6-second.out")"
  cat "$LOG/check6-second.err"
  echo "same-source exit:$(grep '^EXIT:' "$LOG/check6-same.out")"
  cat "$LOG/check6-same.out"
  cat "$LOG/check6-same.err"
} | tee -a "$LOG/check4-7.txt"

# frame counts if the killed dest can be opened
if [[ -d "$C6/to-kill" ]]; then
  "$HARNESS" dump "$C6/to-kill" >"$LOG/check6-kill.dump.json" 2>"$LOG/check6-kill.dump.err" || true
  python3 - <<'PY' | tee -a "$LOG/check4-7.txt"
import json
p="/tmp/verdict-run/logs/check6-kill.dump.json"
try:
    d=json.load(open(p))
except Exception as e:
    print("kill dump unreadable", e)
else:
    print("kill_dest kinds", d.get("kind_counts"), "frames", d.get("frames_total"), "nodes", len(d.get("nodes") or []))
PY
fi
if [[ -d "$C6/to-full" ]]; then
  rm -rf "$C6/dump-full"
  mkdir -p "$C6/dump-full"
  cp -a "$C6/to-full" "$C6/dump-full/log"
  [[ -f "$C6/receipt-signing.key" ]] && cp -a "$C6/receipt-signing.key" "$C6/dump-full/receipt-signing.key"
  "$HARNESS" dump "$C6/dump-full/log" >"$LOG/check6-full.dump.json" 2>"$LOG/check6-full.dump.err" || true
fi

# ---------------------------------------------------------------------------
# Check 7 — lane E, expected to fail, does not block #297
# ---------------------------------------------------------------------------
echo "==== CHECK7 harden ====" | tee -a "$LOG/check4-7.txt"
"$HARNESS" harden "$WORK/check7" >"$LOG/check7.json" 2>"$LOG/check7.err"
echo "harden_exit:$?" | tee -a "$LOG/check4-7.txt"
python3 - <<'PY' | tee -a "$LOG/check4-7.txt"
import json
d=json.load(open("/tmp/verdict-run/logs/check7.json"))
for k,v in d.items():
    print(k, "refused="+str(v.get("refused")), "truncated="+str(v.get("truncated")),
          "regen="+str(v.get("silently_regenerated")), "err="+str(v.get("open_err"))[:240])
PY

# ---------------------------------------------------------------------------
# Check 4 and 5 need strata-verify
# ---------------------------------------------------------------------------
if [[ ! -x "$VERIFY" ]]; then
  echo "STRATA_VERIFY_MISSING $VERIFY" | tee -a "$LOG/check4-7.txt"
  exit 0
fi

echo "==== CHECK4 verify ====" | tee -a "$LOG/check4-7.txt"
# untouched imported log: prefer check3 fresh-v38-ckpt, else check6 to-full
SRCLOG=""
if [[ -d "$WORK/check3/to-fresh-v38-ckpt" ]]; then
  SRCLOG="$WORK/check3/to-fresh-v38-ckpt"
elif [[ -d "$C6/to-full" ]]; then
  SRCLOG="$C6/to-full"
fi
if [[ -n "$SRCLOG" ]]; then
  rm -rf "$WORK/check4-import"
  cp -a "$SRCLOG" "$WORK/check4-import"
  hashdir "$WORK/check4-import" "$LOG/check4-import.before.sha256"
  set +e
  "$VERIFY" "$WORK/check4-import" >"$LOG/check4-import.out" 2>"$LOG/check4-import.err"
  echo EXIT:$? >>"$LOG/check4-import.out"
  set -e
  hashdir "$WORK/check4-import" "$LOG/check4-import.after.sha256"
  if cmp -s "$LOG/check4-import.before.sha256" "$LOG/check4-import.after.sha256"; then
    mut=UNCHANGED
  else
    mut=MUTATED
  fi
  echo "import-log verify exit:$(grep '^EXIT:' "$LOG/check4-import.out") files:$mut" | tee -a "$LOG/check4-7.txt"
  cat "$LOG/check4-import.out" | tee -a "$LOG/check4-7.txt"
  echo "-- stderr --" | tee -a "$LOG/check4-7.txt"
  cat "$LOG/check4-import.err" | tee -a "$LOG/check4-7.txt"
  if [[ "$mut" != UNCHANGED ]]; then
    diff -u "$LOG/check4-import.before.sha256" "$LOG/check4-import.after.sha256" | head -40 | tee -a "$LOG/check4-7.txt" || true
  fi
else
  echo "NO_IMPORTED_LOG" | tee -a "$LOG/check4-7.txt"
fi

echo "==== CHECK4 live strata-store ====" | tee -a "$LOG/check4-7.txt"
rm -rf "$WORK/live-store"
"$HARNESS" live-store "$WORK/live-store" >"$LOG/check4-live-setup.out" 2>"$LOG/check4-live-setup.err"
echo "setup_exit:$?" | tee -a "$LOG/check4-7.txt"
rm -rf "$WORK/live-store-copy" "$WORK/live-log-copy"
cp -a "$WORK/live-store" "$WORK/live-store-copy"
mkdir -p "$WORK/live-log-copy"
cp -a "$WORK/live-store/log" "$WORK/live-log-copy/log"
set +e
"$VERIFY" "$WORK/live-store-copy" >"$LOG/check4-live-root.out" 2>"$LOG/check4-live-root.err"
echo EXIT:$? >>"$LOG/check4-live-root.out"
"$VERIFY" "$WORK/live-log-copy/log" >"$LOG/check4-live-log.out" 2>"$LOG/check4-live-log.err"
echo EXIT:$? >>"$LOG/check4-live-log.out"
# set -e disabled: collect every result
{
  echo "live-root exit:$(grep '^EXIT:' "$LOG/check4-live-root.out")"
  cat "$LOG/check4-live-root.out"
  echo "-- stderr --"
  cat "$LOG/check4-live-root.err"
  echo "live-log exit:$(grep '^EXIT:' "$LOG/check4-live-log.out")"
  cat "$LOG/check4-live-log.out"
  echo "-- stderr --"
  cat "$LOG/check4-live-log.err"
  echo "files created in live-store-copy:"
  find "$WORK/live-store-copy" -type f -printf '%s %p\n'
} | tee -a "$LOG/check4-7.txt"

echo "==== CHECK5 tamper ====" | tee -a "$LOG/check4-7.txt"
if [[ -n "${SRCLOG:-}" ]]; then
  for op in flip replace-derived replace-random; do
    rm -rf "$WORK/tamper-$op"
    set +e
    "$HARNESS" mutate "$op" "$SRCLOG" "$WORK/tamper-$op" >"$LOG/check5-$op-harness.out" 2>"$LOG/check5-$op-harness.err"
    echo HEXIT:$? >>"$LOG/check5-$op-harness.out"
    "$VERIFY" "$WORK/tamper-$op" >"$LOG/check5-$op-verify.out" 2>"$LOG/check5-$op-verify.err"
    echo EXIT:$? >>"$LOG/check5-$op-verify.out"
    set -e
    {
      echo "---- $op harness:$(grep '^HEXIT:' "$LOG/check5-$op-harness.out") verify:$(grep '^EXIT:' "$LOG/check5-$op-verify.out")"
      cat "$LOG/check5-$op-harness.out"
      cat "$LOG/check5-$op-harness.err"
      cat "$LOG/check5-$op-verify.out"
      cat "$LOG/check5-$op-verify.err"
    } | tee -a "$LOG/check4-7.txt"
  done
fi

echo "CHECK4-7 END $(date -Is)" | tee -a "$LOG/check4-7.txt"
