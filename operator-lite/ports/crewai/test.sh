#!/bin/sh
# Operator Lite -- CrewAI port end-to-end test. POSIX sh, stdlib-only.
#
# Hermetic: builds a throwaway HOME (NOT under /tmp or $TMPDIR -- the gate
# classifies system temp as scratch, which would weaken OP-003), runs the real
# install.sh into it, then drives the adapter's EXACT hook decision path
# (`operator_hooks.py hookcheck '<command>'`, the same _hook body crewai calls)
# plus, when a python with crewai importable is available (OPERATOR_CREWAI_PYTHON
# or a probe of python3/python3.13/...), the REAL crewai hook dispatch:
#   run_before_tool_call_hooks(ToolCallHookContext(...)) through the installed module.
#
# Nothing is ever executed: commands are only classified by the gate, which is
# analyzer-only. The EXIT trap removes every trace.
set -u

PORT_DIR=$(cd "$(dirname "$0")" && pwd)

fail=0
ok()  { printf 'PASS: %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }
section() { printf '\n== %s ==\n' "$*"; }

# ---- hermetic environment --------------------------------------------------- #
unset OPERATOR_AGENT_SESSION OPERATOR_GATE OPERATOR_HOME OPERATOR_GATE_MODE \
      OPERATOR_CREWAI_AUTOINSTALL
CREWAI_PY="${OPERATOR_CREWAI_PYTHON:-}"

TEST_BASE="$HOME/.opgate-crewai-test.$$"
rm -rf "$TEST_BASE"
TEST_HOME="$TEST_BASE/home"          # gets the real install.sh treatment
TEST_BARE="$TEST_BASE/bare"          # no install: gate is unreachable from here
mkdir -p "$TEST_HOME" "$TEST_BARE" || exit 1
TMPDIR="$TEST_BASE/tmp"; mkdir -p "$TMPDIR"; export TMPDIR
# The adapter passes os.getcwd() as the gate cwd. Run the checks from a project
# folder inside the throwaway tree, so the result does not depend on where the
# caller stood (from $HOME or above, ~/demo-factory would sit inside the cwd).
mkdir -p "$TEST_BASE/project" && cd "$TEST_BASE/project" || exit 1
ORIG_PWD=$(pwd)
HOME="$TEST_HOME"; export HOME
trap 'rm -rf "$TEST_BASE"' EXIT

RECEIPTS="$TEST_HOME/.operator/receipts/$(date -u +%Y-%m-%d).jsonl"
GATE_DST="$TEST_HOME/.operator/gate/operator-gate.py"
MODULE_DST="$TEST_HOME/.operator/ports/crewai/operator_hooks.py"

hookcheck() { # hookcheck <cmd-string> <tool> <tag>: the adapter's real decision path
  HOME="$TEST_HOME" python3 "$MODULE_DST" hookcheck "$1" "$2" \
    > "$LOGS/$3.stdout" 2> "$LOGS/$3.stderr"
  return $?
}

receipt_decision()    { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["decision"])' "$RECEIPTS" 2>/dev/null; }
receipt_commitments() { python3 -c 'import json,sys; print(",".join(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("commitments") or []))' "$RECEIPTS" 2>/dev/null; }
receipt_source()      { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("source",""))' "$RECEIPTS" 2>/dev/null; }
receipt_tool()        { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("tool",""))' "$RECEIPTS" 2>/dev/null; }
receipt_cwd()         { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("cwd",""))' "$RECEIPTS" 2>/dev/null; }

# ---- 1. install.sh into the throwaway HOME ----------------------------------- #
section "install (hermetic HOME)"
LOGS="$TEST_BASE/logs"; mkdir -p "$LOGS"
if sh "$PORT_DIR/install.sh" > "$LOGS/install.log" 2>&1; then
  ok "install.sh exited 0"
else
  bad "install.sh failed"; cat "$LOGS/install.log"; exit 1
fi
cat "$LOGS/install.log"

[ -f "$GATE_DST" ] && ok "gate installed" || bad "gate missing"
[ -x "$GATE_DST" ] && ok "gate is executable" || bad "gate not executable"
[ -f "$MODULE_DST" ] && ok "adapter installed to ~/.operator/ports/crewai/" || bad "adapter missing"
[ "$(cat "$TEST_HOME/.operator/mode" 2>/dev/null)" = "shadow" ] && ok "mode file = shadow" || bad "mode file not shadow"
grep -q "operator-lite: dialect=" "$LOGS/install.log" && \
  ok "install smoke: adapter status line printed (dialect per machine, gate resolved)" || \
  bad "no adapter status line in install.log"
grep -q "$GATE_DST" "$LOGS/install.log" && ok "status line resolves the installed gate" || bad "status line gate path wrong"

# ---- 2. shadow mode (default): receipts written, nothing blocked --------------- #
section "shadow mode (default)"

hookcheck 'ls -la' bash s-ls; rc=$?
[ "$rc" = "0" ] && ok "benign ls -> exit 0 (allowed)" || bad "benign ls exited $rc, want 0"
case "$(receipt_decision)" in PASS|SHADOW_STOP) ok "benign call receipted (PASS: no rule matched)" ;; *) bad "no receipt for benign call" ;; esac
[ "$(receipt_commitments)" = "" ] && ok "benign receipt carries no rule hits" || bad "commitments: $(receipt_commitments)"
[ "$(receipt_source)" = "crewai" ] && ok "receipt source = crewai (--source plumbed)" || bad "receipt source: $(receipt_source)"
[ "$(receipt_cwd)" = "$ORIG_PWD" ] && ok "receipt cwd = adapter process cwd (os.getcwd())" || bad "receipt cwd: $(receipt_cwd)"

hookcheck 'rm -rf ~/demo-factory/canary' bash s-rm; rc=$?
[ "$rc" = "0" ] && ok "rm -rf in shadow -> exit 0 (logged, not blocked)" || bad "rm -rf in shadow exited $rc, want 0"
[ "$(receipt_decision)" = "SHADOW_STOP" ] && ok "destructive call receipted as SHADOW_STOP" || bad "receipt decision: $(receipt_decision)"
[ "$(receipt_commitments)" = "OP-003" ] && ok "shadow receipt carries OP-003" || bad "commitments: $(receipt_commitments)"
[ "$(receipt_tool)" = "bash" ] && ok "receipt tool = bash (the crew's tool name)" || bad "receipt tool: $(receipt_tool)"

hookcheck "r''m -rf ~/demo-factory/canary" bash s-obs; rc=$?
[ "$rc" = "0" ] && ok "obfuscated r''m in shadow -> exit 0 (analyzed, logged)" || bad "obfuscated shadow exited $rc"
[ "$(receipt_commitments)" = "OP-003" ] && ok "obfuscated receipt still carries OP-003 (gate sees through quote reassembly)" || bad "obfuscated commitments: $(receipt_commitments)"

hookcheck 'echo x | tee ~/.zshrc' bash s-tee; rc=$?
[ "$rc" = "0" ] && ok "shell-init write in shadow -> exit 0 (OP-008 logged)" || bad "tee shadow exited $rc"
[ "$(receipt_commitments)" = "OP-008" ] && ok "shell-init receipt carries OP-008" || bad "tee commitments: $(receipt_commitments)"

# ---- 3. enforce mode: the documented flip, then real blocks --------------------- #
section "enforce mode"
echo enforce > "$TEST_HOME/.operator/mode"

hookcheck 'ls -la' bash e-ls; rc=$?
[ "$rc" = "0" ] && ok "benign ls still allowed in enforce" || bad "benign ls exited $rc, want 0"
[ ! -s "$LOGS/e-ls.stderr" ] && ok "allow writes no stderr" || bad "stderr noise on allow: $(cat "$LOGS/e-ls.stderr")"

hookcheck 'rm -rf ~/demo-factory/canary' bash e-rm; rc=$?
[ "$rc" = "2" ] && ok "plain rm -rf -> exit 2 (BLOCKED, HookAborted path)" || bad "rm -rf exited $rc, want 2"
grep -q "^Operator Lite blocked:" "$LOGS/e-rm.stderr" && \
  ok "reason carries the 'Operator Lite blocked:' HookAborted prefix" || bad "no Operator prefix: $(head -1 "$LOGS/e-rm.stderr")"
grep -q "OPERATOR: STOPPED" "$LOGS/e-rm.stderr" && ok "gate verdict embedded in the reason" || bad "no gate verdict in reason"
grep -q "OP-003" "$LOGS/e-rm.stderr" && ok "reason names the rule (OP-003)" || bad "no rule id in reason"

hookcheck "r''m -rf ~/demo-factory/canary" bash e-obs; rc=$?
[ "$rc" = "2" ] && ok "obfuscated r''m -rf -> exit 2 (gate sees through quote reassembly)" || bad "obfuscated enforce exited $rc"

hookcheck 'echo OPERATOR_CANARY_STOP_7f3a' bash e-canary; rc=$?
[ "$rc" = "2" ] && ok "canary -> exit 2 (proves the whole chain is wired)" || bad "canary exited $rc"

hookcheck 'rm -rf ./src' bash e-src; rc=$?
[ "$rc" = "0" ] && ok "recursive delete INSIDE the crew cwd -> allowed (precise, not trigger-happy)" || bad "in-cwd rm -rf exited $rc, want 0"

hookcheck 'echo x | tee ~/.zshrc' bash e-tee; rc=$?
[ "$rc" = "2" ] && ok "shell-init write -> exit 2 in enforce (OP-008)" || bad "tee enforce exited $rc"

grep -q '"decision": "STOP"' "$RECEIPTS" 2>/dev/null && ok "STOP receipt recorded for the denial" || bad "no STOP receipt"

# ---- 4. gate unreachable: fail closed on destructive / rc writes, open otherwise -- #
section "gate unreachable (fail-closed policy)"
UNREACH_ENV="HOME=$TEST_BARE"
env -u OPERATOR_HOME -u OPERATOR_GATE_MODE HOME="$TEST_BARE" \
    OPERATOR_GATE="$TEST_BARE/no-such-gate.py" \
    python3 "$PORT_DIR/operator_hooks.py" hookcheck 'echo x | tee ~/.zshrc' \
    > "$LOGS/u-tee.stdout" 2> "$LOGS/u-tee.stderr"
[ $? = "2" ] && ok "tee -> ~/.zshrc with gate unreachable -> BLOCKED (fails closed)" \
  || bad "unreachable tee did not fail closed"
grep -q "gate unreachable" "$LOGS/u-tee.stderr" && ok "reason says the gate was unreachable" || bad "no unreachable note"
[ ! -e "$TEST_BARE/.zshrc" ] && ok "no ~/.zshrc was created (payloads never execute)" || bad ".zshrc created"

env -u OPERATOR_HOME HOME="$TEST_BARE" OPERATOR_GATE="$TEST_BARE/no-such-gate.py" \
    python3 "$PORT_DIR/operator_hooks.py" hookcheck 'rm -rf ~/demo-factory/canary' \
    > "$LOGS/u-rm.stdout" 2> "$LOGS/u-rm.stderr"
[ $? = "2" ] && ok "rm -rf with gate unreachable -> BLOCKED (destructive-like fails closed)" \
  || bad "unreachable rm -rf did not fail closed"

env -u OPERATOR_HOME HOME="$TEST_BARE" OPERATOR_GATE="$TEST_BARE/no-such-gate.py" \
    python3 "$PORT_DIR/operator_hooks.py" hookcheck 'ls -la' \
    > "$LOGS/u-ls.stdout" 2> "$LOGS/u-ls.stderr"
[ $? = "0" ] && ok "benign ls with gate unreachable -> allowed (fails open, stderr note)" \
  || bad "unreachable benign ls did not fail open"
grep -q "failing open" "$LOGS/u-ls.stderr" && ok "fail-open is loud on stderr" || bad "silent fail-open"

printf 'raise SystemExit(7)\n' > "$TEST_BARE/stub-gate.py"
env -u OPERATOR_HOME HOME="$TEST_BARE" OPERATOR_GATE="$TEST_BARE/stub-gate.py" \
    python3 "$PORT_DIR/operator_hooks.py" hookcheck 'ls -la' \
    > "$LOGS/u-stub.stdout" 2> "$LOGS/u-stub.stderr"
[ $? = "0" ] && ok "a gate that exits 7 (not 0/2) is treated as unreachable -> fail-open" \
  || bad "weird-exit gate broke the benign path"

# ---- 5. the real crewai hook, when crewai is importable ------------------------- #
section "real crewai hook dispatch"
echo shadow > "$TEST_HOME/.operator/mode"   # section 3 left enforce; restart shadow
ok "mode reset to shadow for the dispatch tests"
if [ "$CREWAI_PY" = "" ]; then
  for cand in python3 python3.13 python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import crewai" >/dev/null 2>&1; then
      CREWAI_PY="$cand"; break
    fi
  done
fi
if [ "$CREWAI_PY" != "" ] && "$CREWAI_PY" -c "import crewai" >/dev/null 2>&1; then
  HOME="$TEST_HOME" OPERATOR_CREWAI_AUTOINSTALL=0 "$CREWAI_PY" - "$MODULE_DST" \
    > "$LOGS/crewai-test.log" 2>&1 <<'PYEOF'
import importlib.util, os, sys
import crewai
from crewai.hooks import (InterceptionPoint, HookAborted, get_hooks,
                          clear_all_hooks, dispatch)
print("crewai %s" % crewai.__version__)

orig_home = os.environ["HOME"]

module_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("operator_hooks", module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.install()
assert mod.DIALECT == "dispatcher", "expected the dispatcher dialect, got %r" % mod.DIALECT

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

# wired: our hook sits on the dispatcher's PRE_TOOL_CALL queue
check("hook registered on PRE_TOOL_CALL",
      mod._hook in get_hooks(InterceptionPoint.PRE_TOOL_CALL))

def blocked_via_public_api(tool, tool_input):
    """Exactly what a crew experiences: the legacy-compatible reducer path."""
    from crewai.hooks.tool_hooks import run_before_tool_call_hooks, ToolCallHookContext
    return run_before_tool_call_hooks(
        ToolCallHookContext(tool_name=tool, tool_input=tool_input, tool=None))

def abort_reason(tool, tool_input):
    """The (reason, source) a HookAborted carries -- telemetry, not the LLM view."""
    from crewai.hooks.tool_hooks import ToolCallHookContext
    try:
        dispatch(InterceptionPoint.PRE_TOOL_CALL,
                 ToolCallHookContext(tool_name=tool, tool_input=tool_input, tool=None))
        return None
    except HookAborted as aborted:
        return (aborted.reason, aborted.source)

# shadow mode: nothing blocks, calls are receipted
check("shadow: ls allowed", blocked_via_public_api("bash", {"command": "ls -la"}) is False)
check("shadow: rm -rf allowed (logged)",
      blocked_via_public_api("bash", {"command": "rm -rf ~/demo-factory/canary"}) is False)

# flip to enforce the documented way and retry the SAME live process
open(os.path.expanduser("~/.operator/mode"), "w").write("enforce\n")
check("enforce: rm -rf blocked",
      blocked_via_public_api("bash", {"command": "rm -rf ~/demo-factory/canary"}) is True)
check("enforce: obfuscated r''m blocked",
      blocked_via_public_api("bash", {"command": "r''m -rf ~/demo-factory/canary"}) is True)
check("enforce: ls still allowed", blocked_via_public_api("bash", {"command": "ls -la"}) is False)

reason, source = abort_reason("bash", {"command": "rm -rf ~/demo-factory/canary"})
check("HookAborted.reason starts with 'Operator Lite blocked:'",
      isinstance(reason, str) and reason.startswith("Operator Lite blocked:"))
check("HookAborted.source is operator-lite", source == "operator-lite")

# a tool with no command-shaped payload is not ours to judge
check("non-command tool_input not gated",
      blocked_via_public_api("web_search", {"query": "rust async"}) is False)

# unreachable gate from the crewai path: shell-rc write fails closed, benign opens
os.environ["HOME"] = "/nonexistent"
os.environ["OPERATOR_HOME"] = "/nonexistent"
os.environ["OPERATOR_GATE"] = "/nonexistent/gate.py"
check("unreachable + tee ~/.zshrc -> blocked (fail closed)",
      blocked_via_public_api("bash", {"command": "echo x | tee ~/.zshrc"}) is True)
check("unreachable + benign -> allowed (fail open)",
      blocked_via_public_api("bash", {"command": "ls -la"}) is False)
for var in ("OPERATOR_HOME", "OPERATOR_GATE"):
    os.environ.pop(var, None)
os.environ["HOME"] = orig_home

clear_all_hooks()
for name, passed in checks:
    print(("PASS" if passed else "FAIL") + ": " + name)
sys.exit(0 if all(p for _, p in checks) else 1)
PYEOF
  if [ $? = "0" ]; then
    ok "real crewai $("$CREWAI_PY" -c 'import crewai; print(crewai.__version__)') dispatch: all checks passed"
    sed 's/^/  /' "$LOGS/crewai-test.log"
  else
    bad "real crewai hook test failed"; sed 's/^/  /' "$LOGS/crewai-test.log"
  fi
else
  printf 'UNTESTED: the real crewai hook dispatch (no python with crewai importable on this machine).\n'
  printf 'Provide one via OPERATOR_CREWAI_PYTHON=/path/to/python and rerun to cover it.\n'
fi

# ---- 6. receipt chain integrity -------------------------------------------------- #
section "receipt chain"
chain=$(HOME="$TEST_HOME" python3 "$GATE_DST" verify 2>/dev/null | head -1)
case "$chain" in
  *"chain=OK"*) ok "receipt hash chain verifies ($chain)" ;;
  *) bad "receipt chain: $chain" ;;
esac

# ---- 7. nothing was ever executed ------------------------------------------------- #
[ ! -e "$TEST_HOME/demo-factory" ] && ok "no ~/demo-factory directory was created" \
  || bad "demo-factory exists -- payloads must never execute"
[ ! -e "$TEST_HOME/src" ] && ok "no ./src directory was created" \
  || bad "./src exists -- payloads must never execute"
[ ! -e "$TEST_HOME/.zshrc" ] && ok "~/.zshrc was never written" \
  || bad ".zshrc exists -- payloads must never execute"

# ---- summary ----------------------------------------------------------------------- #
printf '\n'
if [ "$fail" = 0 ]; then
  echo "ALL CHECKS PASSED (hermetic sandbox removed; nothing outside it touched)."
  if [ "$CREWAI_PY" = "" ]; then
    echo "UNTESTED: a live crew.kickoff() end-to-end and the real crewai hook dispatch --"
    echo "no python with crewai importable was found. The gate-call path (what the hook"
    echo "executes per call) is fully tested above; rerun with"
    echo "OPERATOR_CREWAI_PYTHON=/path/to/python-with-crewai to cover the dispatch too."
  fi
else
  echo "SOME TESTS FAILED."
fi
exit "$fail"
