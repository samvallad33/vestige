#!/bin/sh
# Operator Lite -- OpenHands port end-to-end test. POSIX sh, stdlib-only.
#
# Hermetic: builds a throwaway HOME and project, runs the real install.sh into
# them, then drives the REAL hooks.json command (sh -c 'python3 ~/.operator/...')
# with payloads in the EXACT stdin shape OpenHands documents:
#   {"event_type","tool_name","tool_input":{"command"},"session_id","working_dir"}
#
# Nothing is ever executed: the payloads are only piped into the gate as JSON.
# The gate is analyzer-only, and ~/demo-factory does not exist (asserted).
set -u

PORT_DIR=$(cd "$(dirname "$0")" && pwd)

fail=0
ok()  { printf 'PASS: %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

# ---- hermetic environment --------------------------------------------------- #
unset OPERATOR_AGENT_SESSION OPERATOR_GATE_MODE OPERATOR_HOME

# Throwaway tree, rooted under the real HOME as a dot-directory. Deliberately
# NOT under /tmp or macOS /var/folders: the gate classifies system temp as
# scratch, so a scratch-rooted test tree could never exercise OP-003. The EXIT
# trap removes every trace; the gate itself never executes payloads.
TEST_BASE="$HOME/.opgate-openhands-test.$$"
rm -rf "$TEST_BASE"
TEST_HOME="$TEST_BASE/home"; TEST_PROJ="$TEST_BASE/project"
TMPROOT="$TEST_BASE/logs"
mkdir -p "$TEST_HOME" "$TEST_PROJ" "$TMPROOT" || exit 1
TMPDIR="$TEST_BASE/tmp"; export TMPDIR
HOME="$TEST_HOME"; export HOME
trap 'rm -rf "$TEST_BASE"' EXIT

# ---- 1. install.sh into the throwaway HOME ---------------------------------- #
if sh "$PORT_DIR/install.sh" "$TEST_PROJ" > "$TMPROOT/install.log" 2>&1; then
    ok "install.sh exited 0"
else
    bad "install.sh failed"; cat "$TMPROOT/install.log"; exit 1
fi
cat "$TMPROOT/install.log"

# ---- 2. install shape -------------------------------------------------------- #
[ -f "$HOME/.operator/gate/operator-gate.py" ] && ok "gate installed" || bad "gate missing"
[ -f "$HOME/.operator/gate/openhands-wrapper.py" ] && ok "wrapper installed" || bad "wrapper missing"
[ "$(cat "$HOME/.operator/mode" 2>/dev/null)" = "shadow" ] && ok "mode file = shadow" \
    || bad "mode file not shadow"
[ -x "$HOME/.operator/gate/operator-gate.py" ] && ok "gate is executable" || bad "gate not executable"

python3 - "$TEST_PROJ/.openhands/hooks.json" <<'PYEOF' && ok "hooks.json valid and wired" || bad "hooks.json wrong"
import json, sys
d = json.load(open(sys.argv[1]))
e = d["pre_tool_use"][0]
assert e["matcher"] == "*", e
h = e["hooks"][0]
assert h["type"] == "command" and h["command"].endswith("openhands-wrapper.py"), h
assert h["timeout"] == 10, h
PYEOF

# ---- 3. payloads, exact OpenHands stdin shape -------------------------------- #
P1='rm -rf ~/demo-factory/canary'            # plain destructive delete
P2="r''m -rf ~/demo-factory/canary"          # quote-reassembly obfuscation
P3='ls -la'                                  # benign
P4='rm -rf vestige'                          # workspace-root delete, relative path:
                                             # only judged OP-001 if the wrapper's
                                             # working_dir->cwd reshape works

WD1="$TEST_PROJ"
WD4="$TEST_HOME/Developer"                   # a protected root under the throwaway HOME

run_hook() {                                  # run_hook <cmd> <working_dir> <tag>
    _cmd="$1"; _wd="$2"; _tag="$3"
    CMD="$_cmd" WD="$_wd" python3 - > "$TMPROOT/$_tag.payload" <<'PYEOF'
import json, os
print(json.dumps({
    "event_type": "PreToolUse",
    "tool_name": "terminal",
    "tool_input": {"command": os.environ["CMD"]},
    "session_id": "oh-port-test",
    "working_dir": os.environ["WD"],
}))
PYEOF
    sh -c 'python3 ~/.operator/gate/openhands-wrapper.py' \
        < "$TMPROOT/$_tag.payload" > "$TMPROOT/$_tag.stdout" 2> "$TMPROOT/$_tag.stderr"
}

say() { printf '%s\n' "$*"; }
say ""
say "=== shadow phase (mode file = shadow; gate must analyze but always allow) ==="

run_hook "$P1" "$WD1" p1; rc=$?
say "p1 (rm -rf ~/demo-factory/canary)        exit=$rc"
[ "$rc" -eq 0 ] || bad "p1 should be allowed in shadow (got $rc)"

run_hook "$P2" "$WD1" p2; rc=$?
say "p2 (r''m -rf ... obfuscated)            exit=$rc"
[ "$rc" -eq 0 ] || bad "p2 should be allowed in shadow (got $rc)"

run_hook "$P3" "$WD1" p3; rc=$?
say "p3 (ls -la, benign)                     exit=$rc"
[ "$rc" -eq 0 ] || bad "p3 should be allowed (got $rc)"

run_hook "$P4" "$WD4" p4; rc=$?
say "p4 (rm -rf vestige, cwd=~/Developer)    exit=$rc"
[ "$rc" -eq 0 ] || bad "p4 should be allowed in shadow (got $rc)"

# ---- 4. receipt assertions (proof the gate actually analyzed) ----------------- #
say ""
say "=== receipts (hash-chained, under the throwaway HOME) ==="
C1="$P1" C2="$P2" C3="$P3" C4="$P4" WD4="$WD4" python3 - <<'PYEOF'
import glob, json, os, sys

recs = []
rdir = os.path.join(os.environ["HOME"], ".operator", "receipts")
for fn in sorted(glob.glob(os.path.join(rdir, "*.jsonl"))):
    for line in open(fn):
        line = line.strip()
        if line:
            recs.append(json.loads(line))

fails = []
def need(cond, msg):
    if cond:
        print("PASS: %s" % msg)
    else:
        fails.append(msg)
        print("FAIL: %s" % msg)

need(not [r for r in recs if r.get("decision") == "GATE_ERROR"],
     "no GATE_ERROR receipts")

def shadow_recs(cmd):
    return [r for r in recs
            if r.get("decision") == "SHADOW_STOP" and r.get("action_preview") == cmd]

for cmd, rule, label in [
    (os.environ["C1"], "OP-003", "plain rm -rf judged (OP-003)"),
    (os.environ["C2"], "OP-003", "obfuscated r''m judged through quote reassembly (OP-003)"),
    (os.environ["C4"], "OP-001", "workspace-root delete judged (OP-001) -- proves working_dir->cwd reshape"),
]:
    rs = shadow_recs(cmd)
    need(bool(rs), "shadow receipt present: %s" % label)
    if rs:
        need(rule in (rs[-1].get("commitments") or []),
             "commitments contain %s: %s" % (rule, label))
    if cmd == os.environ["C4"] and rs:
        need(rs[-1].get("cwd") == os.environ["WD4"],
             "receipt cwd == OpenHands working_dir (%s)" % os.environ["WD4"])

# a call no rule matches is receipted as PASS (gate 0.3.6+; SHADOW_STOP before)
rs3 = [r for r in recs if r.get("decision") in ("PASS", "SHADOW_STOP")
       and r.get("action_preview") == os.environ["C3"]]
need(bool(rs3) and not (rs3[-1].get("commitments") or []),
     "benign ls -la: receipted with no rule hits")

if fails:
    sys.exit(1)
PYEOF
[ $? -eq 0 ] || fail=1

# ---- 5. enforce phase: the documented flip, then a real deny ------------------- #
say ""
say "=== enforce phase (echo enforce > mode; gate must DENY exit 2) ==="
printf 'enforce\n' > "$HOME/.operator/mode"

run_hook "$P1" "$WD1" e1; rc=$?
say "p1 (rm -rf ~/demo-factory/canary)        exit=$rc"
[ "$rc" -eq 2 ] && ok "enforce: plain rm -rf denied (exit 2)" || bad "enforce: p1 expected exit 2, got $rc"
grep -q "OPERATOR: STOPPED" "$TMPROOT/e1.stderr" && ok "stderr carries the gate's reason" \
    || bad "stderr missing OPERATOR: STOPPED (got: $(head -c 200 "$TMPROOT/e1.stderr" 2>/dev/null))"
grep -q "OP-003" "$TMPROOT/e1.stderr" && ok "stderr names the rule (OP-003)" || bad "stderr missing rule id"

run_hook "$P2" "$WD1" e2; rc=$?
say "p2 (r''m -rf ... obfuscated)            exit=$rc"
[ "$rc" -eq 2 ] && ok "enforce: obfuscated rm denied (exit 2)" || bad "enforce: p2 expected exit 2, got $rc"

run_hook "$P3" "$WD1" e3; rc=$?
say "p3 (ls -la, benign)                     exit=$rc"
[ "$rc" -eq 0 ] && ok "enforce: benign still allowed" || bad "enforce: p3 expected exit 0, got $rc"

# STOP receipt for the denial
python3 - <<'PYEOF' && ok "STOP receipt recorded for the denial" || fail=1
import glob, json, os, sys
recs = []
for fn in sorted(glob.glob(os.path.join(os.environ["HOME"], ".operator", "receipts", "*.jsonl"))):
    for line in open(fn):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
sys.exit(0 if any(r.get("decision") == "STOP" for r in recs) else 1)
PYEOF

# chain integrity after all of it
python3 "$HOME/.operator/gate/operator-gate.py" verify > "$TMPROOT/verify.out" 2>&1 \
    && ok "receipt chain verifies ($(head -n 1 "$TMPROOT/verify.out"))" \
    || bad "receipt chain broken: $(cat "$TMPROOT/verify.out")"

# ---- 6. nothing was ever executed ---------------------------------------------- #
[ ! -e "$TEST_HOME/demo-factory" ] && ok "no ~/demo-factory directory was created" \
    || bad "demo-factory exists -- payloads must never execute"

# ---- summary -------------------------------------------------------------------- #
say ""
if [ "$fail" -eq 0 ]; then
    say "ALL CHECKS PASSED"
else
    say "FAILURES PRESENT"
fi
exit "$fail"
