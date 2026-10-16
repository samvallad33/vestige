#!/bin/sh
# Operator Lite -- Amazon Q CLI port end-to-end test. POSIX sh, stdlib-only.
#
# Hermetic: builds a throwaway HOME, runs the real install.sh into it twice
# (fresh create + merge-over-existing), then drives the REAL hook command
# (bash -c 'python3 "$HOME/.operator/gate/q-wrapper.py"', through bash exactly
# as Q CLI spawns hook commands -- verified in hooks.rs) with payloads in the
# EXACT stdin shape Q sends:
#   {"hook_event_name": "preToolUse", "cwd", "tool_name", "tool_input": {...}}
# (no session_id: Q's payload construction has no such field.)
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
TEST_BASE="$HOME/.opgate-amazonq-test.$$"
rm -rf "$TEST_BASE"
TEST_HOME="$TEST_BASE/home"; TEST_PROJ="$TEST_BASE/project"
TMPROOT="$TEST_BASE/logs"
mkdir -p "$TEST_HOME" "$TEST_PROJ" "$TMPROOT" || exit 1
TMPDIR="$TEST_BASE/tmp"; export TMPDIR
mkdir -p "$TMPDIR"

HOME="$TEST_HOME"; export HOME
trap 'rm -rf "$TEST_BASE"' EXIT

# ---- 1. install.sh into a fresh throwaway HOME (create path) ----------------- #
if sh "$PORT_DIR/install.sh" > "$TMPROOT/install1.log" 2>&1; then
    ok "install.sh exited 0 (fresh HOME)"
else
    bad "install.sh failed"; cat "$TMPROOT/install1.log"; exit 1
fi
cat "$TMPROOT/install1.log"

# ---- 2. install shape -------------------------------------------------------- #
[ -f "$HOME/.operator/gate/operator-gate.py" ] && ok "gate installed" || bad "gate missing"
[ -f "$HOME/.operator/gate/q-wrapper.py" ] && ok "wrapper installed" || bad "wrapper missing"
[ -x "$HOME/.operator/gate/operator-gate.py" ] && ok "gate is executable" || bad "gate not executable"
[ "$(cat "$HOME/.operator/mode" 2>/dev/null)" = "shadow" ] && ok "mode file = shadow" \
    || bad "mode file not shadow"

AGENT_FILE="$HOME/.aws/amazonq/cli-agents/q_cli_default.json"
python3 - "$AGENT_FILE" <<'PYEOF' && ok "created q_cli_default.json mirrors Q's built-in default + the hook" \
    || bad "agent file wrong"
import json, sys
d = json.load(open(sys.argv[1]))
# every key must be a real Agent field (the schema is deny_unknown_fields)
assert set(d) <= {"$schema", "name", "description", "tools", "resources",
                  "hooks", "useLegacyMcpJson"}, sorted(d)
assert d["$schema"].endswith("/schemas/agent-v1.json"), d.get("$schema")
assert d["name"] == "q_cli_default", d.get("name")
assert d["tools"] == ["*"], d.get("tools")
assert "file://.amazonq/rules/**/*.md" in d["resources"], d.get("resources")
assert d["useLegacyMcpJson"] is True, d.get("useLegacyMcpJson")
pre = d["hooks"]["preToolUse"]
assert len(pre) == 1, pre
h = pre[0]
# Hook struct keys are snake_case and none others exist
assert set(h) <= {"command", "timeout_ms", "max_output_size",
                  "cache_ttl_seconds", "matcher"}, sorted(h)
assert h["command"] == 'python3 "$HOME/.operator/gate/q-wrapper.py"', h
assert h["timeout_ms"] == 10000, h
assert "matcher" not in h          # no matcher = runs for every tool
PYEOF

# ---- 3. re-install over a pre-existing agent file (merge path) --------------- #
mkdir -p "$HOME/.aws/amazonq/cli-agents"
rm -f "$AGENT_FILE"
cat > "$AGENT_FILE" <<'JSON'
{
  "name": "q_cli_default",
  "description": "my tuned default",
  "prompt": "Be terse.",
  "tools": ["fs_read", "fs_write", "execute_bash"],
  "hooks": {
    "agentSpawn": [ { "command": "echo hello" } ]
  }
}
JSON
if sh "$PORT_DIR/install.sh" > "$TMPROOT/install2.log" 2>&1; then
    ok "install.sh exited 0 (existing agent file)"
else
    bad "second install.sh failed"; cat "$TMPROOT/install2.log"; exit 1
fi
ls "$AGENT_FILE".backup-operator-* >/dev/null 2>&1 \
    && ok "pre-existing agent file backed up" || bad "no agent-file backup written"

python3 - "$AGENT_FILE" <<'PYEOF' && ok "merge preserved existing fields and hooks, appended ours" \
    || bad "merge clobbered the agent file"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["description"] == "my tuned default", d.get("description")
assert d["prompt"] == "Be terse.", d.get("prompt")
assert d["tools"] == ["fs_read", "fs_write", "execute_bash"], d.get("tools")
assert d["hooks"]["agentSpawn"][0]["command"] == "echo hello"
pre = d["hooks"]["preToolUse"]
assert len(pre) == 1 and "q-wrapper" in pre[0]["command"], pre
PYEOF

# ---- 4. payloads, exact Q preToolUse stdin shape ------------------------------ #
P1='rm -rf ~/demo-factory/canary'            # plain destructive delete
P2="r''m -rf ~/demo-factory/canary"          # quote-reassembly obfuscation
P3='ls -la'                                  # benign
WD="$TEST_PROJ"                              # what Q sends as cwd

run_hook() {                                  # run_hook <tool> <tool_input_json> <tag>
    _tool="$1"; _ti="$2"; _tag="$3"
    TOOL="$_tool" TI="$_ti" WD="$WD" python3 - > "$TMPROOT/$_tag.payload" <<'PYEOF'
import json, os
print(json.dumps({
    "hook_event_name": "preToolUse",
    "cwd": os.environ["WD"],
    "tool_name": os.environ["TOOL"],
    "tool_input": json.loads(os.environ["TI"]),
}))
PYEOF
    bash -c 'python3 "$HOME/.operator/gate/q-wrapper.py"' \
        < "$TMPROOT/$_tag.payload" > "$TMPROOT/$_tag.stdout" 2> "$TMPROOT/$_tag.stderr"
}

say ""
say "=== shadow phase (mode file = shadow; gate must analyze but always allow) ==="

run_hook execute_bash "{\"command\": \"$P1\", \"summary\": \"cleanup\"}" p1; rc=$?
say "p1 (execute_bash: rm -rf ~/demo-factory/canary)    exit=$rc"
[ "$rc" -eq 0 ] || bad "p1 should be allowed in shadow (got $rc)"

run_hook execute_bash "{\"command\": \"$P2\"}" p2; rc=$?
say "p2 (r''m -rf ... obfuscated)                        exit=$rc"
[ "$rc" -eq 0 ] || bad "p2 should be allowed in shadow (got $rc)"

run_hook execute_bash '{"command": "ls -la"}' p3; rc=$?
say "p3 (ls -la, benign)                                 exit=$rc"
[ "$rc" -eq 0 ] || bad "p3 should be allowed (got $rc)"

run_hook fs_write "{\"command\": \"create\", \"path\": \"$TEST_HOME/.zshrc\", \"file_text\": \"export FOO=1\\n\"}" p4; rc=$?
say "p4 (fs_write create of ~/.zshrc)                    exit=$rc"
[ "$rc" -eq 0 ] || bad "p4 should be allowed in shadow (got $rc)"

run_hook fs_write "{\"command\": \"create\", \"path\": \"$TEST_HOME/settings.ini\", \"file_text\": \"token=sk_live_ABCDEF0123456789\"}" p5; rc=$?
say "p5 (fs_write carrying a credential-shaped body)     exit=$rc"
[ "$rc" -eq 0 ] || bad "p5 should be allowed in shadow (got $rc)"

# ---- 5. receipt assertions (proof the gate actually analyzed) ----------------- #
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
need(all(r.get("session") == "" for r in recs if r.get("source") == "amazon-q"),
     "receipts record an empty session (Q sends no session_id)")

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
             "receipt cwd == Q payload cwd: %s" % label)

# a call no rule matches is receipted as PASS (gate 0.3.6+; SHADOW_STOP before)
rs3 = [r for r in recs if r.get("decision") in ("PASS", "SHADOW_STOP")
       and r.get("action_preview") == os.environ["C3"]]
need(bool(rs3) and not (rs3[-1].get("commitments") or []),
     "benign ls -la: receipted with no rule hits")

# reshape proof 1: an fs_write call must reach the gate renamed to `write` and
# be judged as a shell-init write (OP-008) via its `path`, not passed unjudged
edits = [r for r in recs if r.get("decision") == "SHADOW_STOP"
         and r.get("action_preview", "").startswith("write ")
         and r.get("action_preview", "").endswith(".zshrc")]
need(bool(edits), "fs_write call arrives at the gate as `write` (preview starts with 'write')")
if edits:
    need("OP-008" in (edits[-1].get("commitments") or []),
         "fs_write of ~/.zshrc judged as shell-init write (OP-008)")

# reshape proof 2: the fs_write body must arrive under the name the gate's
# OP-S04 check reads, so a credential-shaped file_text is caught
creds = [r for r in recs if r.get("decision") == "SHADOW_STOP"
         and "OP-S04" in (r.get("commitments") or [])]
need(bool(creds), "credential-shaped fs_write body caught as OP-S04 (body remap works)")

if fails:
    sys.exit(1)
PYEOF
[ $? -eq 0 ] || fail=1

# ---- 6. GuardFall corpus through the adapter ---------------------------------- #
say ""
say "=== GuardFall corpus (every case in corpora/guardfall.json) through the adapter ==="
CORPUS="$PORT_DIR/../../corpora/guardfall.json"
if [ -f "$CORPUS" ]; then
    HOME="$HOME" CORPUS="$CORPUS" python3 - <<'PYEOF' && ok "GuardFall corpus through adapter: every case gave its expected verdict" \
        || bad "corpus through adapter had failures"
import glob
import importlib.util
import json
import os
import sys

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

home = os.environ["HOME"]
gate = load("opgate", os.path.join(home, ".operator", "gate", "operator-gate.py"))
wr = load("qwrap", os.path.join(home, ".operator", "gate", "q-wrapper.py"))
cfg = gate.load_config()
cases = json.load(open(os.environ["CORPUS"]))["cases"]
passed, failed = 0, []
import shutil, tempfile
for case in cases:
    cwd = "/tmp"
    tmpd = None
    if case.get("fixtures"):   # script files the case runs, never executed (as cmd_corpus does)
        tmpd = tempfile.mkdtemp(prefix="opgate-fixture-")
        for fx in case["fixtures"]:
            fp = os.path.join(tmpd, fx["name"])
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                f.write(fx["body"])
        cwd = tmpd
    payload = wr.reshape({"hook_event_name": "preToolUse", "cwd": cwd,
                          "tool_name": "execute_bash",
                          "tool_input": {"command": case["cmd"]}})
    assert payload["tool_name"] == "bash", payload["tool_name"]  # reshape engaged
    _, _, hits, _, _, _ = gate.classify(payload, cfg)
    if tmpd:
        shutil.rmtree(tmpd, True)
    rids = set(h[0] for h in hits)
    exp = case["expect"]
    if exp == "ALLOW":
        good = not [r for r in rids if gate.RULES.get(r, ("", "enforce", ""))[1] == "STOP"]
    elif exp == "SHADOW":
        good = bool(rids)
    else:
        good = exp in rids
    if good:
        passed += 1
    else:
        failed.append("%s expected %s got %s" % (case["id"], exp, ",".join(sorted(rids)) or "ALLOW"))
print("corpus-via-adapter: cases=%d passed=%d failed=%d" % (len(cases), passed, len(failed)))
for line in failed:
    print("  FAIL " + line)
sys.exit(0 if not failed else 1)
PYEOF
    [ $? -eq 0 ] || fail=1
else
    say "SKIP: corpus not found at $CORPUS (installed-from-tarball checkout); gate contract still covered above"
fi

# ---- 7. enforce phase: the documented flip, then real denials ------------------ #
say ""
say "=== enforce phase (echo enforce > mode; gate must DENY exit 2) ==="
printf 'enforce\n' > "$HOME/.operator/mode"

run_hook execute_bash "{\"command\": \"$P1\"}" e1; rc=$?
say "p1 (rm -rf ~/demo-factory/canary)                   exit=$rc"
[ "$rc" -eq 2 ] && ok "enforce: plain rm -rf denied (exit 2)" || bad "enforce: p1 expected exit 2, got $rc"
grep -q "OPERATOR: STOPPED" "$TMPROOT/e1.stderr" \
    && ok "stderr carries the gate's reason (becomes Q's 'PreToolHook blocked' text)" \
    || bad "stderr missing OPERATOR: STOPPED (got: $(head -c 200 "$TMPROOT/e1.stderr" 2>/dev/null))"
grep -q "OP-003" "$TMPROOT/e1.stderr" && ok "stderr names the rule (OP-003)" || bad "stderr missing rule id"

run_hook execute_bash "{\"command\": \"$P2\"}" e2; rc=$?
say "p2 (r''m -rf ... obfuscated)                        exit=$rc"
[ "$rc" -eq 2 ] && ok "enforce: obfuscated rm denied (exit 2)" || bad "enforce: p2 expected exit 2, got $rc"

run_hook execute_bash '{"command": "ls -la"}' e3; rc=$?
say "p3 (ls -la, benign)                                 exit=$rc"
[ "$rc" -eq 0 ] && ok "enforce: benign still allowed" || bad "enforce: p3 expected exit 0, got $rc"

run_hook fs_write "{\"command\": \"create\", \"path\": \"$TEST_HOME/.zshrc\", \"file_text\": \"export FOO=1\\n\"}" e4; rc=$?
say "p4 (fs_write create of ~/.zshrc)                    exit=$rc"
[ "$rc" -eq 2 ] && ok "enforce: fs_write -> write of shell-init denied (exit 2)" \
    || bad "enforce: p4 expected exit 2, got $rc"
grep -q "OP-008" "$TMPROOT/e4.stderr" && ok "denial names OP-008 (the reshape covers write paths)" \
    || bad "e4 stderr missing OP-008"

run_hook fs_write "{\"command\": \"create\", \"path\": \"$TEST_HOME/settings.ini\", \"file_text\": \"token=sk_live_ABCDEF0123456789\"}" e5; rc=$?
say "p5 (credential-shaped fs_write; OP-S04 is shadow-only) exit=$rc"
[ "$rc" -eq 0 ] && ok "enforce: OP-S04 stays log-only by design (exit 0, receipted)" \
    || bad "enforce: p5 expected exit 0 (OP-S04 is a shadow rule), got $rc"

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

# ---- 8. nothing was ever executed ---------------------------------------------- #
[ ! -e "$TEST_HOME/demo-factory" ] && ok "no ~/demo-factory directory was created" \
    || bad "demo-factory exists -- payloads must never execute"
[ ! -e "$TEST_HOME/.zshrc" ] && ok "no ~/.zshrc was written (fs_write payloads never ran)" \
    || bad ".zshrc exists -- payloads must never execute"

# ---- summary -------------------------------------------------------------------- #
say ""
if [ "$fail" -eq 0 ]; then
    say "ALL CHECKS PASSED"
else
    say "FAILURES PRESENT"
fi
exit "$fail"
