#!/bin/sh
# Operator Lite -- Windsurf (Cascade) port end-to-end test. POSIX sh, stdlib-only.
#
# Hermetic: builds a throwaway HOME, workspace and projects (NOT under /tmp or
# $TMPDIR -- the gate classifies system temp as scratch, which would weaken
# OP-003), runs the real install.sh into them, then drives the EXACT hook
# command read back out of the installed hooks.json with payloads in the EXACT
# stdin shape Windsurf sends to Cascade hooks (verified against
# https://docs.devin.ai/desktop/cascade/hooks):
#   {"agent_action_name", "trajectory_id", "execution_id", "timestamp",
#    "model_name", "tool_info": {...}}
#
# Nothing is ever executed: payloads are only piped into the hook command as
# JSON, and the gate is analyzer-only. The EXIT trap removes every trace.
set -u

PORT_DIR=$(cd "$(dirname "$0")" && pwd)

fail=0
ok()  { printf 'PASS: %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }
section() { printf '\n== %s ==\n' "$*"; }

# ---- hermetic environment --------------------------------------------------- #
unset OPERATOR_AGENT_SESSION OPERATOR_GATE OPERATOR_HOME OPERATOR_GATE_MODE OPERATOR_PYTHON

TEST_BASE="$HOME/.opgate-windsurf-test.$$"
rm -rf "$TEST_BASE"
TEST_HOME="$TEST_BASE/home"
TEST_PROJ="$TEST_BASE/project"
TEST_WS_PLAIN="$TEST_BASE/ws-plain"     # no .devin, no .windsurf -> legacy path written
TEST_WS_DEVIN="$TEST_BASE/ws-devin"     # pre-existing .devin/hooks.json -> merged there
LOGS="$TEST_BASE/logs"
mkdir -p "$TEST_HOME" "$TEST_PROJ" "$TEST_WS_PLAIN" "$TEST_WS_DEVIN" "$LOGS" || exit 1
TMPDIR="$TEST_BASE/tmp"; export TMPDIR
HOME="$TEST_HOME"; export HOME
trap 'rm -rf "$TEST_BASE"' EXIT

HOOKS_JSON="$TEST_HOME/.codeium/windsurf/hooks.json"
TODAY=$(date -u +%Y-%m-%d)
RECEIPTS="$TEST_HOME/.operator/receipts/$TODAY.jsonl"

hook_cmd_of() { # extract the command install.sh wrote into hooks.json
  python3 - "$1" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1]))["hooks"]["pre_run_command"][0]["command"])
PYEOF
}

run_hook() { # run_hook <hook-cmd> <agent_action_name> <tool-info-json> <trajectory> <tag>
  _hcmd="$1"; _action="$2"; _pti="$3"; _tid="$4"; _tag="$5"
  PACTION="$_action" PTI="$_pti" PTID="$_tid" python3 - > "$LOGS/$_tag.payload" <<'PYEOF'
import json, os
print(json.dumps({
    "agent_action_name": os.environ["PACTION"],
    "trajectory_id": os.environ["PTID"],
    "execution_id": os.environ["PTID"] + "-exec-1",
    "timestamp": "2026-09-29T12:00:00Z",
    "model_name": "test-model",
    "tool_info": json.loads(os.environ["PTI"]),
}))
PYEOF
  # Windsurf runs hook commands via `bash -c` on macOS/Linux (docs: Configuration
  # Options) -- so "$HOME" in the wired command expands, exactly as it will live.
  if command -v bash >/dev/null 2>&1; then
    bash -c "$_hcmd" < "$LOGS/$_tag.payload" > "$LOGS/$_tag.stdout" 2> "$LOGS/$_tag.stderr"
  else
    sh -c "$_hcmd" < "$LOGS/$_tag.payload" > "$LOGS/$_tag.stdout" 2> "$LOGS/$_tag.stderr"
  fi
}

run_cmd()   { run_hook "$1" pre_run_command   "$2" "$3" "$4"; }
run_write() { run_hook "$1" pre_write_code    "$2" "$3" "$4"; }
run_mcp()   { run_hook "$1" pre_mcp_tool_use  "$2" "$3" "$4"; }

receipt_decision()    { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])["decision"])' "$RECEIPTS" 2>/dev/null; }
receipt_commitments() { python3 -c 'import json,sys; print(",".join(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("commitments") or []))' "$RECEIPTS" 2>/dev/null; }
receipt_source()      { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("source",""))' "$RECEIPTS" 2>/dev/null; }
receipt_tool()        { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("tool",""))' "$RECEIPTS" 2>/dev/null; }
receipt_cwd()         { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("cwd",""))' "$RECEIPTS" 2>/dev/null; }
receipt_session()     { python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get("session",""))' "$RECEIPTS" 2>/dev/null; }
gate_entries() { # operator-gate hook entries across the three wired events
  python3 - "$1" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
n = 0
for ev in ("pre_run_command", "pre_write_code", "pre_mcp_tool_use"):
    for h in d.get("hooks", {}).get(ev, []):
        j = json.dumps(h) if isinstance(h, dict) else ""
        if "windsurf-wrapper" in j or "operator-gate" in j:
            n += 1
print(n)
PYEOF
}

# ---- 1. install.sh into the throwaway HOME (user level, the default) ---------- #
section "install (user level, fresh HOME)"
if sh "$PORT_DIR/install.sh" > "$LOGS/install.log" 2>&1; then
  ok "install.sh exited 0"
else
  bad "install.sh failed"; cat "$LOGS/install.log"; exit 1
fi
cat "$LOGS/install.log"

[ -f "$TEST_HOME/.operator/gate/operator-gate.py" ] && ok "gate installed" || bad "gate missing"
[ -x "$TEST_HOME/.operator/gate/operator-gate.py" ] && ok "gate is executable" || bad "gate not executable"
[ -f "$TEST_HOME/.operator/gate/windsurf-wrapper.py" ] && ok "adapter installed next to the gate" || bad "adapter missing"
[ "$(cat "$TEST_HOME/.operator/mode" 2>/dev/null)" = "shadow" ] && ok "mode file = shadow" || bad "mode file not shadow"

python3 - "$HOOKS_JSON" <<'PYEOF' && ok "hooks.json valid + wired: 3 pre-events, flat schema (command only, no matcher/timeout), \$HOME form" || bad "hooks.json wrong"
import json, sys
d = json.load(open(sys.argv[1]))
h = d["hooks"]
assert set(h) == {"pre_run_command", "pre_write_code", "pre_mcp_tool_use"}, h
for ev, lst in h.items():
    assert isinstance(lst, list) and len(lst) == 1, (ev, lst)
    e = lst[0]
    assert "command" in e, e                                  # flat documented schema
    for absent in ("matcher", "timeout", "hooks", "type"):
        assert absent not in e, (ev, e)
    assert e["command"].startswith('python3 "$HOME/'), e
    assert e["command"].endswith('windsurf-wrapper.py"'), e
PYEOF

# ---- 2. shadow mode: receipts written, nothing blocked ------------------------ #
section "shadow mode (default)"
HOOK_CMD=$(hook_cmd_of "$HOOKS_JSON")

rc=$(run_cmd "$HOOK_CMD" '{"command_line":"ls -la","cwd":"'"$TEST_PROJ"'"}' traj-ls s-ls; echo $?)
[ "$rc" = "0" ] && ok "benign ls -> exit 0 (allowed)" || bad "benign ls exited $rc, want 0"

rc=$(run_cmd "$HOOK_CMD" '{"command_line":"rm -rf ~/demo-factory/canary","cwd":"'"$TEST_PROJ"'"}' traj-rm s-rm; echo $?)
[ "$rc" = "0" ] && ok "rm -rf in shadow -> exit 0 (logged, not blocked)" || bad "rm -rf in shadow exited $rc, want 0"
[ -f "$RECEIPTS" ] && ok "receipts file exists" || bad "no receipts file"
[ "$(receipt_decision)" = "SHADOW_STOP" ] && ok "destructive call receipted as SHADOW_STOP" || bad "receipt decision: $(receipt_decision)"
[ "$(receipt_commitments)" = "OP-003" ] && ok "shadow receipt carries OP-003" || bad "commitments: $(receipt_commitments)"
[ "$(receipt_source)" = "windsurf" ] && ok "receipt source = windsurf (--source windsurf plumbed)" || bad "receipt source: $(receipt_source)"
[ "$(receipt_tool)" = "run_terminal_cmd" ] && ok "receipt tool = run_terminal_cmd (reshape applied)" || bad "receipt tool: $(receipt_tool)"
[ "$(receipt_cwd)" = "$TEST_PROJ" ] && ok "payload tool_info.cwd reached the gate untouched" || bad "receipt cwd: $(receipt_cwd)"
[ "$(receipt_session)" = "traj-rm" ] && ok "receipt session = trajectory_id" || bad "receipt session: $(receipt_session)"

rc=$(run_write "$HOOK_CMD" "{\"file_path\":\"$TEST_HOME/.zshrc\",\"edits\":[{\"old_string\":\"a\",\"new_string\":\"b\"}]}" traj-w s-write; echo $?)
[ "$rc" = "0" ] && ok "pre_write_code -> ~/.zshrc in shadow -> exit 0 (logged)" || bad "write in shadow exited $rc, want 0"
[ "$(receipt_commitments)" = "OP-008" ] && ok "write receipt carries OP-008 (shell-init write)" || bad "write commitments: $(receipt_commitments)"
[ "$(receipt_tool)" = "edit" ] && ok "write receipt tool = edit (reshape applied)" || bad "write receipt tool: $(receipt_tool)"

rc=$(run_mcp "$HOOK_CMD" '{"mcp_server_name":"slack","mcp_tool_name":"send_message","mcp_tool_arguments":{"channel":"general","text":"hi"}}' traj-m1 s-mcp1; echo $?)
[ "$rc" = "0" ] && ok "pre_mcp_tool_use send_message in shadow -> exit 0 (shadow-only rule)" || bad "mcp in shadow exited $rc, want 0"
[ "$(receipt_commitments)" = "OP-S02" ] && ok "mcp receipt carries OP-S02 (outbound comms, mcp__ synthesized)" || bad "mcp commitments: $(receipt_commitments)"
[ "$(receipt_tool)" = "mcp__slack__send_message" ] && ok "mcp receipt tool = mcp__slack__send_message" || bad "mcp receipt tool: $(receipt_tool)"

rc=$(run_mcp "$HOOK_CMD" '{"mcp_server_name":"fs","mcp_tool_name":"read","mcp_tool_arguments":{"path":"~/.ssh/id_rsa"}}' traj-m2 s-mcp2; echo $?)
[ "$(receipt_commitments)" = "OP-S13" ] && ok "mcp read of ~/.ssh/id_rsa receipted as OP-S13 (sensitive path)" || bad "mcp2 commitments: $(receipt_commitments)"

rc=$(run_cmd "$HOOK_CMD" '{"command_line":"rm -rf ./src","cwd":"'"$TEST_PROJ"'"}' traj-src s-src; echo $?)
[ "$rc" = "0" ] && ok "recursive delete INSIDE cwd -> exit 0 (gate is precise, not trigger-happy)" || bad "in-cwd rm -rf exited $rc, want 0"

# ---- 3. merge into an existing user hooks.json (+ backup, idempotency) --------- #
section "merge into existing hooks.json (user level)"
mkdir -p "$TEST_HOME/.codeium/windsurf"
printf '{\n  "hooks": {\n    "pre_run_command": [\n      { "command": "echo my-existing-logger" }\n    ]\n  }\n}\n' > "$HOOKS_JSON"
if sh "$PORT_DIR/install.sh" > "$LOGS/install2.log" 2>&1; then
  ok "install.sh with existing hooks.json exited 0"
else
  bad "install.sh merge failed"; cat "$LOGS/install2.log"
fi
python3 - "$HOOKS_JSON" <<'PYEOF' && ok "merge preserved the existing pre_run_command hook + appended ours" || bad "merge lost data"
import json, sys
pre = json.load(open(sys.argv[1]))["hooks"]["pre_run_command"]
assert [h["command"] for h in pre][0] == "echo my-existing-logger", pre
assert any("windsurf-wrapper" in json.dumps(h) for h in pre), pre
PYEOF
ls "$HOOKS_JSON".bak-* >/dev/null 2>&1 && ok "original backed up (hooks.json.bak-*)" || bad "no backup of pre-existing hooks.json"

sh "$PORT_DIR/install.sh" > "$LOGS/install3.log" 2>&1 && ok "rerun exited 0" || bad "rerun failed"
[ "$(gate_entries "$HOOKS_JSON")" = "3" ] && ok "rerun is idempotent (still 3 operator-gate entries)" || bad "duplicate entries: $(gate_entries "$HOOKS_JSON")"
set -- "$HOOKS_JSON".bak-*
[ "$#" -eq 1 ] && ok "no extra backup on the no-op rerun" || bad "backup count changed on rerun (got $#)"

# ---- 4. invalid hooks.json: refused, untouched, manual instructions ------------- #
section "invalid hooks.json refused"
printf '{ this is not json' > "$HOOKS_JSON"
cp "$HOOKS_JSON" "$LOGS/broken-before"
BAKS_BEFORE=$(ls "$HOOKS_JSON".bak-* 2>/dev/null | wc -l | tr -d ' ')
if sh "$PORT_DIR/install.sh" > "$LOGS/install4.log" 2>&1; then
  bad "install.sh should fail on invalid hooks.json"
else
  ok "install.sh refused invalid hooks.json (nonzero exit)"
fi
cmp -s "$LOGS/broken-before" "$HOOKS_JSON" && ok "invalid file left byte-identical" || bad "invalid file was modified"
grep -q "not valid JSON" "$LOGS/install4.log" && ok "manual merge instructions printed" || bad "no guidance on refusal"
BAKS_AFTER=$(ls "$HOOKS_JSON".bak-* 2>/dev/null | wc -l | tr -d ' ')
[ "$BAKS_BEFORE" = "$BAKS_AFTER" ] && ok "no new backup written (nothing was modified)" || bad "backup count changed on refusal ($BAKS_BEFORE -> $BAKS_AFTER)"

# ---- 5. workspace-level installs (legacy path + .devin precedence) -------------- #
section "workspace-level installs"
if sh "$PORT_DIR/install.sh" "$TEST_WS_PLAIN" > "$LOGS/install5.log" 2>&1; then
  ok "workspace install exited 0"
else
  bad "workspace install failed"; cat "$LOGS/install5.log"
fi
[ -f "$TEST_WS_PLAIN/.windsurf/hooks.json" ] && ok "no existing hooks -> legacy .windsurf/hooks.json written" || bad ".windsurf/hooks.json missing"
[ "$(gate_entries "$TEST_WS_PLAIN/.windsurf/hooks.json")" = "3" ] && ok "workspace file carries all 3 pre-hooks" || bad "workspace entries: $(gate_entries "$TEST_WS_PLAIN/.windsurf/hooks.json")"

mkdir -p "$TEST_WS_DEVIN/.devin"
printf '{\n  "hooks": {\n    "pre_run_command": [\n      { "command": "echo team-logger" }\n    ]\n  }\n}\n' > "$TEST_WS_DEVIN/.devin/hooks.json"
sh "$PORT_DIR/install.sh" "$TEST_WS_DEVIN" > "$LOGS/install6.log" 2>&1 && ok "install into ws with existing .devin/hooks.json exited 0" || bad ".devin merge failed"
[ ! -e "$TEST_WS_DEVIN/.windsurf" ] && ok "no .windsurf dir shadow-created (merged into .devin instead)" || bad ".windsurf dir created despite existing .devin file"
grep -q "team-logger" "$TEST_WS_DEVIN/.devin/hooks.json" && ok "team's existing .devin hook preserved" || bad ".devin merge lost the team hook"
[ "$(gate_entries "$TEST_WS_DEVIN/.devin/hooks.json")" = "3" ] && ok "gate entries merged into .devin/hooks.json" || bad ".devin entries: $(gate_entries "$TEST_WS_DEVIN/.devin/hooks.json")"

# ---- 6. enforce mode: the documented flip, then real blocks --------------------- #
section "enforce mode"
echo enforce > "$TEST_HOME/.operator/mode"

rc=$(run_cmd "$HOOK_CMD" '{"command_line":"ls -la","cwd":"'"$TEST_PROJ"'"}' traj-ls2 e-ls; echo $?)
[ "$rc" = "0" ] && ok "benign ls -> exit 0 (allowed)" || bad "benign ls exited $rc, want 0"
[ ! -s "$LOGS/e-ls.stdout" ] && ok "benign allow keeps stdout empty (show_output would print nothing)" || bad "stdout not empty on allow"
[ ! -s "$LOGS/e-ls.stderr" ] && ok "benign allow writes no stderr" || bad "stderr noise on allow"

rc=$(run_cmd "$HOOK_CMD" '{"command_line":"rm -rf ~/demo-factory/canary","cwd":"'"$TEST_PROJ"'"}' traj-rm2 e-rm; echo $?)
[ "$rc" = "2" ] && ok "plain rm -rf -> exit 2 (BLOCKED)" || bad "rm -rf exited $rc, want 2"
grep -q "OPERATOR: STOPPED" "$LOGS/e-rm.stderr" && ok "stderr carries OPERATOR: STOPPED reason (Cascade shows it to the agent)" || bad "no OPERATOR reason on stderr"
grep -q "OP-003" "$LOGS/e-rm.stderr" && ok "stderr names the rule (OP-003)" || bad "stderr missing rule id"
[ ! -s "$LOGS/e-rm.stdout" ] && ok "block keeps stdout empty (exit 2 + stderr is the contract)" || bad "stdout not empty on block"

rc=$(run_cmd "$HOOK_CMD" '{"command_line":"r'"''"'m -rf ~/demo-factory/canary","cwd":"'"$TEST_PROJ"'"}' traj-obs e-obs; echo $?)
[ "$rc" = "2" ] && ok "obfuscated r''m -rf -> exit 2 (gate sees through quote reassembly)" || bad "obfuscated rm exited $rc, want 2"

rc=$(run_cmd "$HOOK_CMD" '{"command_line":"echo OPERATOR_CANARY_STOP_7f3a","cwd":"'"$TEST_PROJ"'"}' traj-canary e-canary; echo $?)
[ "$rc" = "2" ] && ok "canary payload -> exit 2 (proves the whole chain is wired)" || bad "canary exited $rc, want 2"

rc=$(run_write "$HOOK_CMD" "{\"file_path\":\"$TEST_HOME/.zshrc\",\"edits\":[{\"old_string\":\"a\",\"new_string\":\"b\"}]}" traj-w2 e-write; echo $?)
[ "$rc" = "2" ] && ok "pre_write_code -> ~/.zshrc in enforce -> exit 2 (OP-008)" || bad "write in enforce exited $rc, want 2"
grep -q "OP-008" "$LOGS/e-write.stderr" && ok "write stderr names OP-008" || bad "write stderr missing rule id"

rc=$(run_cmd "$HOOK_CMD" '{"command_line":"rm -rf ./src","cwd":"'"$TEST_PROJ"'"}' traj-src2 e-src; echo $?)
[ "$rc" = "0" ] && ok "in-cwd rm -rf still allowed in enforce" || bad "in-cwd rm -rf exited $rc, want 0"

grep -q '"decision": "STOP"' "$RECEIPTS" 2>/dev/null && ok "STOP receipt recorded for the denial" || bad "no STOP receipt"

# ---- 6b. guardfall corpus through the adapter (verdicts must survive reshape) ---- #
section "guardfall corpus via the windsurf adapter (enforce mode)"
CORPUS="$PORT_DIR/../../corpora/guardfall.json"
if [ -f "$CORPUS" ]; then
  # Every corpus case is replayed as a real pre_run_command payload through the
  # wrapper subprocess. Semantics mirror the gate's own `corpus guardfall`
  # runner (cmd_corpus): ALLOW = no STOP-class hit; SHADOW = any hit recorded
  # (a shadow case may legally co-fire a STOP hit, which blocks in enforce);
  # a rule id = that rule hit and the call blocked.
  GATE_CMD="$HOOK_CMD" CORPUS_FILE="$CORPUS" RECEIPTS="$RECEIPTS" python3 <<'PYEOF' \
    && ok "guardfall corpus: every case's verdict survives the windsurf reshape" \
    || bad "guardfall corpus via adapter"
import json, os, subprocess, sys

STOP_BASE = ("OP-000", "OP-001", "OP-002", "OP-003", "OP-004", "OP-005", "OP-006",
             "OP-007", "OP-008", "OP-009", "OP-010", "OP-CANARY")
receipts = os.environ["RECEIPTS"]

def last_receipt():
    try:
        with open(receipts) as f:
            lines = f.read().splitlines()
        return json.loads(lines[-1]) if lines else {}
    except Exception:
        return {}

corpus = json.load(open(os.environ["CORPUS_FILE"]))["cases"]
fails = []
import shutil, tempfile
for case in corpus:
    cwd, tmpd = "/tmp", None
    if case.get("fixtures"):   # script files the case runs, never executed (as cmd_corpus does)
        tmpd = tempfile.mkdtemp(prefix="opgate-fixture-")
        for fx in case["fixtures"]:
            fp = os.path.join(tmpd, fx["name"])
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                f.write(fx["body"])
        cwd = tmpd
    payload = json.dumps({
        "agent_action_name": "pre_run_command",
        "trajectory_id": "guardfall",
        "execution_id": "guardfall-exec",
        "timestamp": "2026-09-29T12:00:00Z",
        "model_name": "corpus",
        "tool_info": {"command_line": case["cmd"], "cwd": cwd},
    })
    r = subprocess.run(["bash", "-c", os.environ["GATE_CMD"]],
                       input=payload.encode(), capture_output=True)
    if tmpd:
        shutil.rmtree(tmpd, True)
    rec = last_receipt()
    hits = set(rec.get("commitments") or [])
    expect = str(case["expect"])
    if expect == "ALLOW":
        good = r.returncode == 0 and not (hits & set(STOP_BASE))
    elif expect == "SHADOW":
        good = bool(hits)
    else:
        good = r.returncode == 2 and expect in hits
    if not good:
        fails.append("%s expect %s got exit %d hits=%s"
                     % (case["id"], expect, r.returncode, ",".join(sorted(hits)) or "-"))
print("cases=%d passed=%d failed=%d" % (len(corpus), len(corpus) - len(fails), len(fails)))
for f in fails:
    print("  FAIL " + f)
sys.exit(1 if fails else 0)
PYEOF
else
  printf 'SKIP: corpora/guardfall.json not found\n'
fi

# ---- 7. receipt chain integrity -------------------------------------------------- #
section "receipt chain"
chain=$(python3 "$TEST_HOME/.operator/gate/operator-gate.py" verify 2>/dev/null | head -1)
case "$chain" in
  *"chain=OK"*) ok "receipt hash chain verifies ($chain)" ;;
  *) bad "receipt chain: $chain" ;;
esac

# ---- 8. nothing was ever executed ------------------------------------------------- #
[ ! -e "$TEST_HOME/demo-factory" ] && ok "no ~/demo-factory directory was created" \
  || bad "demo-factory exists -- payloads must never execute"
[ ! -e "$TEST_PROJ/src" ] && ok "no ./src directory was created" \
  || bad "./src exists -- payloads must never execute"

# ---- summary ----------------------------------------------------------------------- #
printf '\n'
if [ "$fail" = 0 ]; then
  echo "ALL CHECKS PASSED (hermetic sandbox removed; nothing outside it touched)."
  echo "UNTESTED: a live Windsurf/Cascade session (not installable on this machine) --"
  echo "hook discovery of ~/.codeium/windsurf/hooks.json (or workspace hooks.json),"
  echo "Cascade-side display of the stderr deny reason, and hook-config reload timing"
  echo "should be confirmed by running install.sh and the canary step inside Windsurf."
else
  echo "SOME TESTS FAILED."
fi
exit "$fail"
