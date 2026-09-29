#!/usr/bin/env bash
# Redo check 6 (real SIGKILL + non-empty --to) and checks 4-5 (strata-verify).
set -u
ROOT=/tmp/verdict-run
LOG=$ROOT/logs
WORK=$ROOT/work
DBS=/tmp/verdict-inputs/dbs
VESTIGE=/tmp/verdict-target/release/vestige
VERIFY=/tmp/verdict-target/release/strata-verify
HARNESS=$ROOT/harness/target/release/verdict-harness
C6=$WORK/check6b
mkdir -p "$LOG" "$C6"

isolate_env() {
  local home="$1"; shift
  mkdir -p "$home"
  env -i PATH="$PATH" HOME="$home" USER="${USER:-ubuntu}" LANG="${LANG:-C.UTF-8}" \
    TMPDIR="${TMPDIR:-/tmp}" RUST_LOG=warn "$@"
}
hashdir() {
  local dir="$1" out="$2"
  find "$dir" -type f -print0 | sort -z | while IFS= read -r -d '' f; do sha256sum "$f"; done >"$out"
}

echo "CHECK3B $(date -Is)" | tee "$LOG/check6b.txt"
rm -rf "$C6"
mkdir -p "$C6/src" "$C6/home-full"
cp -a "$DBS/fresh-v38-ckpt/." "$C6/src/"
hashdir "$C6/src" "$LOG/check6b-src.before.sha256"

isolate_env "$C6/home-full" "$VESTIGE" --data-dir "$C6/home-full" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-full" \
  >"$LOG/check6b-full.out" 2>"$LOG/check6b-full.err"
echo EXIT:$? | tee -a "$LOG/check6b-full.out" | tee -a "$LOG/check6b.txt"
cat "$LOG/check6b-full.out" | tee -a "$LOG/check6b.txt"
# second fresh migration of the same source: keys must differ, data frames comparable
isolate_env "$C6/home-full2" "$VESTIGE" --data-dir "$C6/home-full2" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-full2" \
  >"$LOG/check6b-full2.out" 2>"$LOG/check6b-full2.err"
echo EXIT:$? >>"$LOG/check6b-full2.out"
echo "key1 $(sha256sum "$C6/to-full/strata.key")" | tee -a "$LOG/check6b.txt"
echo "key2 $(sha256sum "$C6/to-full2/strata.key")" | tee -a "$LOG/check6b.txt"
echo "receipt-key mode $(stat -c '%a' "$C6/receipt-signing.key") path-parent-of-to" | tee -a "$LOG/check6b.txt"
# strata.key is inside --to
stat -c 'strata.key mode %a %n' "$C6/to-full/strata.key" | tee -a "$LOG/check6b.txt"

python3 - "$VESTIGE" "$C6" <<'PY' | tee -a "$LOG/check6b.txt"
import os, signal, subprocess, time, sys
vestige, c6 = sys.argv[1], sys.argv[2]
dest = os.path.join(c6, "to-kill")
home = os.path.join(c6, "home-kill")
os.makedirs(home, exist_ok=True)
env = {k: os.environ[k] for k in ("PATH","LANG","TMPDIR") if k in os.environ}
env.update(HOME=home, RUST_LOG="warn", USER=os.environ.get("USER","ubuntu"))
cmd = [vestige, "--data-dir", home, "migrate-to-strata", "--from", os.path.join(c6,"src"), "--to", dest]

src = os.path.join(c6, "slowwrite.c")
so = os.path.join(c6, "slowwrite.so")
open(src,"w").write(r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <unistd.h>
static void pause_us(void) { usleep(4000); }
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
r = subprocess.run(["gcc","-shared","-fPIC","-O2","-o",so,src,"-ldl"], capture_output=True, text=True)
print("preload", r.returncode, r.stderr[-200:])
env["LD_PRELOAD"] = so

if os.path.exists(dest):
    subprocess.check_call(["rm","-rf",dest])
p = subprocess.Popen(cmd, env=env, start_new_session=True,
                     stdout=open(os.path.join(c6,"kill.out"),"w"),
                     stderr=open(os.path.join(c6,"kill.err"),"w"))
killed = False
seen = 0
t0 = time.time()
while time.time()-t0 < 120:
    if p.poll() is not None:
        break
    hit = 0
    if os.path.isdir(dest):
        for root, dirs, files in os.walk(dest):
            for f in files:
                if f.endswith(".seg"):
                    hit = max(hit, os.path.getsize(os.path.join(root,f)))
    if hit > 2500 and p.poll() is None:
        os.killpg(p.pid, signal.SIGKILL)
        killed = True
        seen = hit
        break
    time.sleep(0.002)
code = p.wait()
print(f"killed={killed} code={code} size_at_kill={seen} elapsed={time.time()-t0:.2f}s")
print("dest:")
if os.path.isdir(dest):
    for root, dirs, files in os.walk(dest):
        for f in files:
            path=os.path.join(root,f)
            print(f"  {os.path.getsize(path):8d} {os.path.relpath(path, dest)}")
else:
    print("  (none)")
open(os.path.join(c6,"killed.flag"),"w").write("1\n" if killed and code != 0 else "0\n")
PY

echo "==== rerun ====" | tee -a "$LOG/check6b.txt"
isolate_env "$C6/home-rerun" "$VESTIGE" --data-dir "$C6/home-rerun" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-kill" \
  >"$LOG/check6b-rerun.out" 2>"$LOG/check6b-rerun.err"
echo RERUN_EXIT:$? | tee -a "$LOG/check6b-rerun.out" | tee -a "$LOG/check6b.txt"
cat "$LOG/check6b-rerun.out" | tee -a "$LOG/check6b.txt"
echo "-- stderr --" | tee -a "$LOG/check6b.txt"
cat "$LOG/check6b-rerun.err" | tee -a "$LOG/check6b.txt"
hashdir "$C6/src" "$LOG/check6b-src.after.sha256"
if cmp -s "$LOG/check6b-src.before.sha256" "$LOG/check6b-src.after.sha256"; then
  echo SOURCE_UNCHANGED | tee -a "$LOG/check6b.txt"
else
  echo SOURCE_CHANGED | tee -a "$LOG/check6b.txt"
fi

echo "==== non-empty ====" | tee -a "$LOG/check6b.txt"
mkdir -p "$C6/to-foreign"; echo junk >"$C6/to-foreign/note.txt"
isolate_env "$C6/home-foreign" "$VESTIGE" --data-dir "$C6/home-foreign" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-foreign" \
  >"$LOG/check6b-foreign.out" 2>"$LOG/check6b-foreign.err"
echo FOREIGN_EXIT:$? | tee -a "$LOG/check6b-foreign.out" | tee -a "$LOG/check6b.txt"
cat "$LOG/check6b-foreign.err" | tee -a "$LOG/check6b.txt"
cp -a "$DBS/vnc-v36" "$C6/src-vnc"
isolate_env "$C6/home-second" "$VESTIGE" --data-dir "$C6/home-second" \
  migrate-to-strata --from "$C6/src-vnc" --to "$C6/to-full" \
  >"$LOG/check6b-second.out" 2>"$LOG/check6b-second.err"
echo SECOND_EXIT:$? | tee -a "$LOG/check6b-second.out" | tee -a "$LOG/check6b.txt"
cat "$LOG/check6b-second.err" | tee -a "$LOG/check6b.txt"
isolate_env "$C6/home-same" "$VESTIGE" --data-dir "$C6/home-same" \
  migrate-to-strata --from "$C6/src" --to "$C6/to-full" \
  >"$LOG/check6b-same.out" 2>"$LOG/check6b-same.err"
echo SAME_EXIT:$? | tee -a "$LOG/check6b-same.out" | tee -a "$LOG/check6b.txt"
cat "$LOG/check6b-same.out" | tee -a "$LOG/check6b.txt"
cat "$LOG/check6b-same.err" | tee -a "$LOG/check6b.txt"

# dumps
"$HARNESS" dump "$C6/to-kill" >"$LOG/check6b-kill.dump.json" 2>"$LOG/check6b-kill.dump.err" || true
python3 - <<'PY' | tee -a "$LOG/check6b.txt"
import json
p="/tmp/verdict-run/logs/check6b-kill.dump.json"
try:
    d=json.load(open(p))
except Exception as e:
    print("kill dump", e)
    print(open("/tmp/verdict-run/logs/check6b-kill.dump.err").read()[:500])
else:
    print("after_rerun_or_partial kinds", d.get("kind_counts"), "frames", d.get("frames_total"), "nodes", len(d.get("nodes") or []))
PY

echo "==== CHECK4/5 ====" | tee -a "$LOG/check6b.txt"
SRCLOG=$WORK/check3/to-fresh-v38-ckpt
rm -rf "$WORK/check4-import"
cp -a "$SRCLOG" "$WORK/check4-import"
hashdir "$WORK/check4-import" "$LOG/check4-import.before.sha256"
"$VERIFY" "$WORK/check4-import" >"$LOG/check4-import.out" 2>"$LOG/check4-import.err"
echo VERIFY_IMPORT_EXIT:$? | tee -a "$LOG/check4-import.out" | tee -a "$LOG/check6b.txt"
hashdir "$WORK/check4-import" "$LOG/check4-import.after.sha256"
if cmp -s "$LOG/check4-import.before.sha256" "$LOG/check4-import.after.sha256"; then
  echo IMPORT_VERIFY_FILES_UNCHANGED | tee -a "$LOG/check6b.txt"
else
  echo IMPORT_VERIFY_FILES_MUTATED | tee -a "$LOG/check6b.txt"
  diff -u "$LOG/check4-import.before.sha256" "$LOG/check4-import.after.sha256" | head -30 | tee -a "$LOG/check6b.txt" || true
fi
echo "-- verify stdout --" | tee -a "$LOG/check6b.txt"
cat "$LOG/check4-import.out" | tee -a "$LOG/check6b.txt"
echo "-- verify stderr --" | tee -a "$LOG/check6b.txt"
cat "$LOG/check4-import.err" | tee -a "$LOG/check6b.txt"

rm -rf "$WORK/live-store"
"$HARNESS" live-store "$WORK/live-store"
rm -rf "$WORK/live-store-copy" "$WORK/live-log-copy"
cp -a "$WORK/live-store" "$WORK/live-store-copy"
mkdir -p "$WORK/live-log-copy"
cp -a "$WORK/live-store/log/." "$WORK/live-log-copy/"
"$VERIFY" "$WORK/live-store-copy" >"$LOG/check4-live-root.out" 2>"$LOG/check4-live-root.err"
echo LIVE_ROOT_EXIT:$? | tee -a "$LOG/check4-live-root.out" | tee -a "$LOG/check6b.txt"
"$VERIFY" "$WORK/live-log-copy" >"$LOG/check4-live-log.out" 2>"$LOG/check4-live-log.err"
echo LIVE_LOG_EXIT:$? | tee -a "$LOG/check4-live-log.out" | tee -a "$LOG/check6b.txt"
echo "-- live root stderr --" | tee -a "$LOG/check6b.txt"
cat "$LOG/check4-live-root.err" | tee -a "$LOG/check6b.txt"
echo "-- live log stderr --" | tee -a "$LOG/check6b.txt"
cat "$LOG/check4-live-log.err" | tee -a "$LOG/check6b.txt"
echo "-- live root stdout --" | tee -a "$LOG/check6b.txt"
head -40 "$LOG/check4-live-root.out" | tee -a "$LOG/check6b.txt"

for op in flip replace-derived replace-random; do
  rm -rf "$WORK/tamper-$op"
  "$HARNESS" mutate "$op" "$SRCLOG" "$WORK/tamper-$op" >"$LOG/check5-$op-harness.out" 2>"$LOG/check5-$op-harness.err"
  echo "HARNESS_$op:$?" | tee -a "$LOG/check5-$op-harness.out" | tee -a "$LOG/check6b.txt"
  "$VERIFY" "$WORK/tamper-$op" >"$LOG/check5-$op-verify.out" 2>"$LOG/check5-$op-verify.err"
  echo "VERIFY_$op:$?" | tee -a "$LOG/check5-$op-verify.out" | tee -a "$LOG/check6b.txt"
  echo "-- $op stderr --" | tee -a "$LOG/check6b.txt"
  cat "$LOG/check5-$op-harness.err" | tee -a "$LOG/check6b.txt"
  cat "$LOG/check5-$op-verify.err" | tee -a "$LOG/check6b.txt"
  cat "$LOG/check5-$op-harness.out" | tee -a "$LOG/check6b.txt"
done

echo "==== nometa probe ====" | tee -a "$LOG/check6b.txt"
"$HARNESS" probe-nometa "$WORK/check7-nometa" | tee -a "$LOG/check6b.txt"

# vestige strata-verify subcommand?
"$VESTIGE" strata-verify "$WORK/check4-import" >"$LOG/check4-subcommand.out" 2>"$LOG/check4-subcommand.err" || true
echo SUBCOMMAND_EXIT:$? | tee -a "$LOG/check4-subcommand.out" | tee -a "$LOG/check6b.txt"
head -5 "$LOG/check4-subcommand.err" | tee -a "$LOG/check6b.txt"
echo DONE $(date -Is) | tee -a "$LOG/check6b.txt"
