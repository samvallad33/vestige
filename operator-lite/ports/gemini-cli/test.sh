#!/bin/sh
# Operator Lite -- Gemini CLI port end-to-end test. POSIX sh, stdlib-only.
#
# Hermetic: builds a throwaway HOME, runs the real install.sh into it, then
# drives the REAL hook command (sh -c 'python3 ~/.operator/gate/gemini-wrapper.py',
# through the shell exactly as gemini-cli spawns hook commands) with payloads
# in the EXACT stdin shape gemini-cli documents and sends:
#   {"session_id","transcript_path","cwd","hook_event_name","timestamp",
#    "tool_name","tool_input":{...}}
#
# Nothing is ever executed: the payloads are only piped into the gate as JSON.
# The gate is analyzer-only, and ~/demo-factory does not exist (asserted).
set -u

PORT_DIR=$(cd "$(dirname "$0")" && pwd)

fail=0
ok()  { printf 'PASS: %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }
say() { printf '%s\n' "$*"; }

# ---- hermetic environment --------------------------------------------------- #
unset OPERATOR_AGENT_SESSION OPERATOR_GATE_MODE OPERATOR_HOME

# Throwaway tree, rooted under the real HOME as a dot-directory. Deliberately
# NOT under /tmp or macOS /var/folders: the gate classifies system temp as
# scratch, so a scratch-rooted test tree could never exercise OP-003. The EXIT
# trap removes every trace; the gate itself never executes payloads.
TEST_BASE="$HOME/.opgate-gemini-test.$$"
rm -rf "$TEST_BASE"
TEST_HOME="$TEST_BASE/home"; TEST_PROJ="$TEST_BASE/project"
TMPROOT="$TEST_BASE/logs"
mkdir -p "$TEST_HOME/.gemini" "$TEST_PROJ" "$TMPROOT" || exit 1
TMPDIR="$TEST_BASE/tmp"; export TMPDIR
mkdir -p "$TMPDIR"

# Pre-existing settings: the install must merge, never clobber.
printf '{\n  "theme": "auto",\n  "hooks": {\n    "AfterTool": [\n      {"matcher": "web_fetch", "hooks": [{"type": "command", "command": "echo unrelated"}]}\n    ]\n  }\n}\n' \
    > "$TEST_HOME/.gemini/settings.json"

HOME="$TEST_HOME"; export HOME
trap 'rm -rf "$TEST_BASE"' EXIT

# ---- 1. install.sh into the throwaway HOME ---------------------------------- #
if sh "$PORT_DIR/install.sh" > "$TMPROOT/install.log" 2>&1; then
    ok "install.sh exited 0"
else
    bad "install.sh failed"; cat "$TMPROOT/install.log"; exit 1
fi
cat "$TMPROOT/install.log"

# ---- 2. install shape -------------------------------------------------------- #
[ -f "$HOME/.operator/gate/operator-gate.py" ] && ok "gate installed" || bad "gate missing"
[ -f "$HOME/.operator/gate/gemini-wrapper.py" ] && ok "wrapper installed" || bad "wrapper missing"
[ -x "$HOME/.operator/gate/operator-gate.py" ] && ok "gate is executable" || bad "gate not executable"
[ "$(cat "$HOME/.operator/mode" 2>/dev/null)" = "shadow" ] && ok "mode file = shadow" \
    || bad "mode file not shadow"
ls "$HOME/.gemini"/settings.json.backup-operator-* >/dev/null 2>&1 \
    && ok "pre-existing settings backed up" || bad "no settings backup written"

python3 - "$HOME/.gemini/settings.json" <<'PYEOF' && ok "settings.json merge preserved existing content and wired the hook" \
    || bad "settings.json merge wrong"
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("theme") == "auto", d                       # pre-existing key kept
after = d["hooks"]["AfterTool"]                          # pre-existing event kept
assert after[0]["hooks"][0]["command"] == "echo unrelated", after
entries = d["hooks"]["BeforeTool"]                       # our event appended
assert len(entries) == 1, entries
e = entries[0]
assert e["matcher"] == "*", e
h = e["hooks"][0]
assert h["type"] == "command", h
assert h["name"] == "operator-lite", h
assert h["command"] == "python3 ~/.operator/gate/gemini-wrapper.py", h
assert h["timeout"] == 10000, h
PYEOF

# ---- 3. payloads, exact gemini-cli BeforeTool stdin shape --------------------- #
P1='rm -rf ~/demo-factory/canary'            # plain destructive delete
P2="r''m -rf ~/demo-factory/canary"          # quote-reassembly obfuscation
P3='ls -la'                                  # benign
WD="$TEST_PROJ"                              # what gemini-cli sends as cwd

run_hook() {                                  # run_hook <tool> <tool_input_json> <tag>
    _tool="$1"; _ti="$2"; _tag="$3"
    TOOL="$_tool" TI="$_ti" WD="$WD" python3 - > "$TMPROOT/$_tag.payload" <<'PYEOF'
import json, os
print(json.dumps({
    "session_id": "gemini-port-test",
    "transcript_path": "/tmp/gemini-port-test.jsonl",
    "cwd": os.environ["WD"],
    "hook_event_name": "BeforeTool",
    "timestamp": "2026-09-29T12:00:00Z",
    "tool_name": os.environ["TOOL"],
    "tool_input": json.loads(os.environ["TI"]),
}))
PYEOF
    sh -c 'python3 ~/.operator/gate/gemini-wrapper.py' \
        < "$TMPROOT/$_tag.payload" > "$TMPROOT/$_tag.stdout" 2> "$TMPROOT/$_tag.stderr"
}

say ""
say "=== shadow phase (mode file = shadow; gate must analyze but always allow) ==="

run_hook run_shell_command "{\"command\": \"$P1\"}" p1; rc=$?
say "p1 (run_shell_command: rm -rf ~/demo-factory/canary)  exit=$rc"
[ "$rc" -eq 0 ] || bad "p1 should be allowed in shadow (got $rc)"

run_hook run_shell_command "{\"command\": \"$P2\"}" p2; rc=$?
say "p2 (r''m -rf ... obfuscated)                        exit=$rc"
[ "$rc" -eq 0 ] || bad "p2 should be allowed in shadow (got $rc)"

run_hook run_shell_command '{"command": "ls -la"}' p3; rc=$?
say "p3 (ls -la, benign)                                 exit=$rc"
[ "$rc" -eq 0 ] || bad "p3 should be allowed (got $rc)"

run_hook replace "{\"file_path\": \"$TEST_HOME/.zshrc\", \"old_string\": \"a\", \"new_string\": \"b\"}" p4; rc=$?
say "p4 (replace tool editing ~/.zshrc)                  exit=$rc"
[ "$rc" -eq 0 ] || bad "p4 should be allowed in shadow (got $rc)"

# ---- 4. receipt assertions (proof the gate actually analyzed) ----------------- #
say ""
say "=== receipts (hash-chained, under the throwaway HOME) ==="
C1="$P1" C2="$P2" C3='ls -la' WD="$WD" TH="$TEST_HOME" python3 - <<'PYEOF'
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
]:
    rs = shadow_recs(cmd)
    need(bool(rs), "shadow receipt present: %s" % label)
    if rs:
        need(rule in (rs[-1].get("commitments") or []),
             "commitments contain %s: %s" % (rule, label))
        need(rs[-1].get("cwd") == os.environ["WD"],
             "receipt cwd == gemini payload cwd: %s" % label)

# a call no rule matches is receipted as PASS (gate 0.3.6+; SHADOW_STOP before)
rs3 = [r for r in recs if r.get("decision") in ("PASS", "SHADOW_STOP")
       and r.get("action_preview") == os.environ["C3"]]
need(bool(rs3) and not (rs3[-1].get("commitments") or []),
     "benign ls -la: receipted with no rule hits")

# the reshape proof: a gemini `replace` call must reach the gate renamed to
# `edit` and be judged as a shell-init write (OP-008), not passed unjudged
edits = [r for r in recs if r.get("decision") == "SHADOW_STOP"
         and r.get("action_preview", "").startswith("edit ")]
need(bool(edits), "replace call arrives at the gate as `edit` (preview starts with 'edit')")
if edits:
    need("OP-008" in (edits[-1].get("commitments") or []),
         "replace->edit of ~/.zshrc judged as shell-init write (OP-008)")

if fails:
    sys.exit(1)
PYEOF
[ $? -eq 0 ] || fail=1

# ---- 5. enforce phase: the documented flip, then real denials ------------------ #
say ""
say "=== enforce phase (echo enforce > mode; gate must DENY exit 2) ==="
printf 'enforce\n' > "$HOME/.operator/mode"

run_hook run_shell_command "{\"command\": \"$P1\"}" e1; rc=$?
say "p1 (rm -rf ~/demo-factory/canary)                   exit=$rc"
[ "$rc" -eq 2 ] && ok "enforce: plain rm -rf denied (exit 2)" || bad "enforce: p1 expected exit 2, got $rc"
grep -q "OPERATOR: STOPPED" "$TMPROOT/e1.stderr" && ok "stderr carries the gate's reason (becomes gemini's deny reason)" \
    || bad "stderr missing OPERATOR: STOPPED (got: $(head -c 200 "$TMPROOT/e1.stderr" 2>/dev/null))"
grep -q "OP-003" "$TMPROOT/e1.stderr" && ok "stderr names the rule (OP-003)" || bad "stderr missing rule id"

run_hook run_shell_command "{\"command\": \"$P2\"}" e2; rc=$?
say "p2 (r''m -rf ... obfuscated)                        exit=$rc"
[ "$rc" -eq 2 ] && ok "enforce: obfuscated rm denied (exit 2)" || bad "enforce: p2 expected exit 2, got $rc"

run_hook run_shell_command '{"command": "ls -la"}' e3; rc=$?
say "p3 (ls -la, benign)                                 exit=$rc"
[ "$rc" -eq 0 ] && ok "enforce: benign still allowed" || bad "enforce: p3 expected exit 0, got $rc"

run_hook replace "{\"file_path\": \"$TEST_HOME/.zshrc\", \"old_string\": \"a\", \"new_string\": \"b\"}" e4; rc=$?
say "p4 (replace tool editing ~/.zshrc)                  exit=$rc"
[ "$rc" -eq 2 ] && ok "enforce: replace->edit of shell-init denied (exit 2)" \
    || bad "enforce: p4 expected exit 2, got $rc"
grep -q "OP-008" "$TMPROOT/e4.stderr" && ok "denial names OP-008 (the reshape covers write paths)" \
    || bad "e4 stderr missing OP-008"

# STOP receipts for the denials
python3 - <<'PYEOF' && ok "STOP receipts recorded for the denials" || fail=1
import glob, json, os, sys
recs = []
for fn in sorted(glob.glob(os.path.join(os.environ["HOME"], ".operator", "receipts", "*.jsonl"))):
    for line in open(fn):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
stops = [r for r in recs if r.get("decision") == "STOP"]
sys.exit(0 if len(stops) >= 3 and any(r.get("commitment") == "OP-008" for r in stops) else 1)
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
