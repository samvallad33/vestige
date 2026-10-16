#!/bin/sh
# Operator Lite -- LangChain / LangGraph port end-to-end test. POSIX sh, stdlib-only.
#
# Hermetic: builds a throwaway HOME rooted under the real HOME (NOT under /tmp
# or $TMPDIR -- the gate classifies system temp as scratch, which would weaken
# OP-003), runs the real install.sh into it, then drives the middleware's
# gate-call path directly: decide() (the exact function wrap_tool_call applies
# to every tool call) with tool args in the exact shape LangChain passes
# (request.tool_call -> {"name", "args"}).
#
# Nothing is ever executed: command strings are only piped to the gate as JSON
# payloads, and the gate is analyzer-only. The EXIT trap removes every trace.
# The real ~/.operator is never touched.
set -u

PORT_DIR=$(cd "$(dirname "$0")" && pwd)
ORIG_HOME="$HOME"

fail=0
ok()  { printf 'PASS: %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }
skip() { printf 'SKIP: %s\n' "$*"; }
section() { printf '\n== %s ==\n' "$*"; }

# ---- hermetic environment --------------------------------------------------- #
unset OPERATOR_AGENT_SESSION OPERATOR_GATE OPERATOR_HOME OPERATOR_GATE_MODE OPERATOR_PYTHON

TEST_BASE="$ORIG_HOME/.opgate-langchain-test.$$"
rm -rf "$TEST_BASE"
TEST_HOME="$TEST_BASE/home"
TEST_PROJ="$TEST_BASE/project"
LOGS="$TEST_BASE/logs"
mkdir -p "$TEST_HOME" "$TEST_PROJ" "$LOGS" || exit 1
TMPDIR="$TEST_BASE/tmp"; export TMPDIR
HOME="$TEST_HOME"; export HOME
trap 'rm -rf "$TEST_BASE"' EXIT

GATE="$TEST_HOME/.operator/gate/operator-gate.py"
MIDDLEWARE="$TEST_HOME/.operator/ports/langchain/operator_middleware.py"
TODAY=$(date -u +%Y-%m-%d)
RECEIPTS="$TEST_HOME/.operator/receipts/$TODAY.jsonl"

# decide() driver: loads the INSTALLED middleware module (the artifact the
# owner gets), applies the exact decision wrap_tool_call applies, prints the
# verdict as one JSON line, and optionally asserts the expected verdict.
decide_call() { # decide_call <tool> <args-json> <cwd-or--> <session-or--> [expected]
  MODULE="$MIDDLEWARE" PTOOL="$1" PARGS="$2" PCWD="$3" PSID="$4" PEXPECT="${5:-}" \
    python3 > "$LOGS/decide.json" <<'PYEOF'
import importlib.util, json, os, sys

spec = importlib.util.spec_from_file_location("operator_middleware", os.environ["MODULE"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cwd = os.environ["PCWD"]
sid = os.environ["PSID"]
verdict, message = mod.decide(
    os.environ["PTOOL"], json.loads(os.environ["PARGS"]),
    cwd=None if cwd == "-" else cwd,
    session_id=None if sid == "-" else sid)
expected = os.environ["PEXPECT"]
if expected and verdict != expected:
    print("DRIVER FAIL: want verdict %s got %s (%s)" % (expected, verdict, message))
    sys.exit(1)
print(json.dumps({"verdict": verdict, "message": message}))
PYEOF
  return $?
}

receipt_field() { # receipt_field <key>
  python3 -c 'import json,sys
rec = json.loads(open(sys.argv[1]).read().splitlines()[-1])
v = rec.get(sys.argv[2])
print(",".join(v) if isinstance(v, list) else ("" if v is None else v))' "$RECEIPTS" "$1" 2>/dev/null
}
receipt_lines() { wc -l < "$RECEIPTS" 2>/dev/null | tr -d ' ' ; echo; }

# ---- 1. install.sh into the throwaway HOME ----------------------------------- #
section "install (fresh HOME)"
if sh "$PORT_DIR/install.sh" > "$LOGS/install.log" 2>&1; then
  ok "install.sh exited 0"
else
  bad "install.sh failed"; cat "$LOGS/install.log"; exit 1
fi
cat "$LOGS/install.log"

[ -f "$GATE" ] && ok "gate installed" || bad "gate missing"
[ -x "$GATE" ] && ok "gate is executable" || bad "gate not executable"
[ -f "$MIDDLEWARE" ] && ok "middleware installed to ~/.operator/ports/langchain/" || bad "middleware missing"
cmp -s "$PORT_DIR/operator_middleware.py" "$MIDDLEWARE" && ok "installed middleware is byte-identical to the port" || bad "installed middleware diverges from the port"
[ "$(cat "$TEST_HOME/.operator/mode" 2>/dev/null)" = "shadow" ] && ok "mode file = shadow" || bad "mode file not shadow"
grep -q "operator_middleware" "$LOGS/install.log" && ok "usage snippet printed" || bad "no usage snippet"

# module must import with NO langchain present (stdlib-only adapter)
if python3 -c "import langchain" 2>/dev/null; then
  LC_PRESENT=1
else
  LC_PRESENT=0
fi
MODULE="$MIDDLEWARE" python3 -c '
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("operator_middleware", os.environ["MODULE"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.decide and mod.gate_verdict and mod.DESTRUCTIVE_LIKE
' && ok "middleware imports cleanly (langchain ${LC_PRESENT}: adapter is stdlib-only, lazy langchain)" \
  || bad "middleware import failed"

# ---- 2. shadow mode (default): analyzed + receipted, nothing blocked ---------- #
section "shadow mode (default)"

decide_call shell '{"command":"rm -rf ~/demo-factory/canary"}' "$TEST_PROJ" sess-rm allow \
  && ok "rm -rf in shadow -> gate allowed it (logged, not blocked)" \
  || { bad "rm -rf in shadow should be allowed"; cat "$LOGS/decide.json"; }
[ "$(receipt_field decision)" = "SHADOW_STOP" ] && ok "destructive call receipted as SHADOW_STOP" || bad "receipt decision: $(receipt_field decision)"
[ "$(receipt_field commitments)" = "OP-003" ] && ok "shadow receipt carries OP-003" || bad "commitments: $(receipt_field commitments)"
[ "$(receipt_field source)" = "langchain" ] && ok "receipt source = langchain (--source plumbed)" || bad "receipt source: $(receipt_field source)"
[ "$(receipt_field tool)" = "shell" ] && ok "receipt tool = shell (tool_name passed through)" || bad "receipt tool: $(receipt_field tool)"
[ "$(receipt_field cwd)" = "$TEST_PROJ" ] && ok "payload cwd reached the gate untouched" || bad "receipt cwd: $(receipt_field cwd)"
[ "$(receipt_field session)" = "sess-rm" ] && ok "receipt session = session_id" || bad "receipt session: $(receipt_field session)"
grep -q "rm -rf" "$RECEIPTS" 2>/dev/null && ok "receipt previews the action" || bad "no action preview in receipt"

decide_call shell '{"command":"ls -la"}' "$TEST_PROJ" sess-ls allow \
  && ok "benign ls -> allowed" || bad "benign ls should be allowed"
[ "$(receipt_field commitments)" = "" ] && ok "benign call receipted with no rule hits" || bad "benign receipt commitments: $(receipt_field commitments)"

# ---- 3. enforce mode: the documented flip, then real blocks ------------------- #
section "enforce mode"
echo enforce > "$TEST_HOME/.operator/mode"

decide_call shell '{"command":"rm -rf ~/demo-factory/canary"}' "$TEST_PROJ" sess-rm2 block \
  && ok "plain rm -rf -> BLOCKED" || { bad "rm -rf should block in enforce"; cat "$LOGS/decide.json"; }
grep -q "Operator Lite blocked:" "$LOGS/decide.json" && ok "block message prefix: Operator Lite blocked:" || bad "wrong block prefix"
grep -q "OPERATOR: STOPPED" "$LOGS/decide.json" && ok "gate stderr (the reason) carried verbatim" || bad "no gate stderr in message"
grep -q "OP-003" "$LOGS/decide.json" && ok "message names the rule (OP-003)" || bad "message missing rule id"
grep -q '"decision": "STOP"' "$RECEIPTS" 2>/dev/null && ok "STOP receipt recorded for the denial" || bad "no STOP receipt"

decide_call shell '{"command":"r'"''"'m -rf ~/demo-factory/canary"}' "$TEST_PROJ" sess-obs block \
  && ok "obfuscated r''m -rf -> BLOCKED (gate sees through quote reassembly)" || bad "obfuscated rm should block"

decide_call shell '{"command":"echo OPERATOR_CANARY_STOP_7f3a"}' "$TEST_PROJ" sess-canary block \
  && ok "canary payload -> BLOCKED (proves the whole chain is wired)" || bad "canary should block"

decide_call shell '{"command":"ls -la"}' "$TEST_PROJ" sess-ls2 allow \
  && ok "benign ls still allowed in enforce" || bad "benign ls should pass in enforce"

decide_call shell '{"command":"rm -rf ./src"}' "$TEST_PROJ" sess-src allow \
  && ok "recursive delete INSIDE cwd -> allowed (gate is precise, not trigger-happy)" || bad "in-cwd rm -rf should be allowed"

# ---- 3b. guardfall corpus through the adapter (verdicts must survive decide()) -- #
section "guardfall corpus via the adapter (enforce mode)"
CORPUS="$PORT_DIR/../../corpora/guardfall.json"
if [ -f "$CORPUS" ]; then
  MODULE="$MIDDLEWARE" CORPUS_FILE="$CORPUS" RECEIPTS="$RECEIPTS" python3 > "$LOGS/corpus.log" 2>&1 <<'PYEOF' \
    && ok "guardfall corpus: every case's verdict survives the adapter's decide()" \
    || { bad "guardfall corpus via adapter"; cat "$LOGS/corpus.log"; }
import importlib.util, json, os, sys

STOP_BASE = ("OP-000", "OP-001", "OP-002", "OP-003", "OP-004", "OP-005", "OP-006",
             "OP-007", "OP-008", "OP-009", "OP-010", "OP-CANARY")
spec = importlib.util.spec_from_file_location("operator_middleware", os.environ["MODULE"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
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
    verdict, message = mod.decide("shell", {"command": case["cmd"]}, cwd=cwd)
    if tmpd:
        shutil.rmtree(tmpd, True)
    hits = set(last_receipt().get("commitments") or [])
    expect = str(case["expect"])
    if expect == "ALLOW":
        good = verdict == "allow" and not (hits & set(STOP_BASE))
    elif expect == "SHADOW":
        # a shadow case may legally co-fire a STOP hit, which blocks in
        # enforce (same semantics as the gate's own corpus runner and the
        # windsurf port's adapter replay)
        good = bool(hits)
    else:
        good = verdict == "block" and expect in hits
    if not good:
        fails.append("%s expect %s got %s hits=%s"
                     % (case["id"], expect, verdict, ",".join(sorted(hits)) or "-"))
print("cases=%d passed=%d failed=%d" % (len(corpus), len(corpus) - len(fails), len(fails)))
for f in fails:
    print("  FAIL " + f)
sys.exit(1 if fails else 0)
PYEOF
else
  skip "corpora/guardfall.json not found"
fi

# ---- 4. gate unreachable: fail-open, fail-closed on destructive-like ---------- #
section "gate unreachable (fail-open / fail-closed policy)"
( export OPERATOR_GATE="$TEST_BASE/missing/gate.py"
  decide_call shell '{"command":"tee ~/.zshrc"}' "$TEST_PROJ" sess-rc block ) \
  && ok "rc-write with gate unreachable -> BLOCKED (fail closed, destructive-like)" \
  || { bad "unreachable + rc-write should fail closed"; cat "$LOGS/decide.json"; }
grep -q "gate unreachable" "$LOGS/decide.json" && ok "message explains the gate was unreachable" || bad "no unreachable explanation"

( export OPERATOR_GATE="$TEST_BASE/missing/gate.py"
  decide_call shell '{"command":"ls -la"}' "$TEST_PROJ" sess-open allow ) \
  && ok "benign command with gate unreachable -> allowed (fail open)" || bad "unreachable + benign should fail open"

decide_call shell '{"command":"echo OPERATOR_CANARY_STOP_7f3a"}' "$TEST_PROJ" sess-canary2 block \
  && ok "gate reachable again -> canary blocked (override cleared)" || bad "canary should block again"

# ---- 5. passthrough: non-shell-ish tools never spawn the gate ----------------- #
section "passthrough (out of scope by design)"
BEFORE=$(receipt_lines)
decide_call read_file '{"path":"notes.md"}' "$TEST_PROJ" - pass \
  && ok "tool with no command shape -> PASS (handler would run, gate not consulted)" || bad "read_file should pass through"
AFTER=$(receipt_lines)
[ "$BEFORE" = "$AFTER" ] && ok "no receipt written for passthrough (no gate spawn)" || bad "passthrough spawned the gate ($BEFORE -> $AFTER)"

decide_call shell '{"path":"notes.md"}' "$TEST_PROJ" - pass \
  && ok "shell-named tool with no command key -> PASS (nothing to judge)" || bad "shell w/o command should pass"

# ---- 6. real langchain wrap (only when langchain is importable) ---------------- #
section "real langchain middleware wrap"
if [ "$LC_PRESENT" = "1" ]; then
  MODULE="$MIDDLEWARE" python3 > "$LOGS/wrap.log" 2>&1 <<'PYEOF'
import asyncio, importlib.util, os, sys

spec = importlib.util.spec_from_file_location("operator_middleware", os.environ["MODULE"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    mw = mod.operator_middleware()
except ImportError as exc:
    print("SKIP: langchain present but agents.middleware unavailable (%s)" % exc)
    sys.exit(3)

class FakeRequest:
    def __init__(self, name, args, call_id):
        self.tool_call = {"name": name, "args": args, "id": call_id}

def handler(request):
    return "HANDLER_RAN"

blocked = mw.wrap_tool_call(
    FakeRequest("shell", {"command": "rm -rf ~/demo-factory/canary"}, "call_1"), handler)
from langchain_core.messages import ToolMessage
assert isinstance(blocked, ToolMessage), type(blocked)
assert blocked.tool_call_id == "call_1", blocked.tool_call_id
assert blocked.content.startswith("Operator Lite blocked:"), blocked.content
assert "OP-003" in blocked.content, blocked.content

allowed = mw.wrap_tool_call(FakeRequest("shell", {"command": "ls -la"}, "call_2"), handler)
assert allowed == "HANDLER_RAN", allowed

passthrough = mw.wrap_tool_call(FakeRequest("read_file", {"path": "notes.md"}, "call_3"), handler)
assert passthrough == "HANDLER_RAN", passthrough

async def async_checks():
    async def async_handler(request):     # real awrap handlers are awaitable
        return "HANDLER_RAN"
    b = await mw.awrap_tool_call(
        FakeRequest("shell", {"command": "r''m -rf ~/demo-factory/canary"}, "call_4"), async_handler)
    assert isinstance(b, ToolMessage) and "OP-003" in b.content, b
    a = await mw.awrap_tool_call(FakeRequest("shell", {"command": "ls -la"}, "call_5"), async_handler)
    assert a == "HANDLER_RAN", a

asyncio.run(async_checks())
print("wrap checks passed")
PYEOF
  wrap_rc=$?
  if [ "$wrap_rc" = "0" ]; then
    ok "wrap_tool_call: blocked call returns ToolMessage, handler never runs; allowed and passthrough calls reach the handler; async path behaves the same"
  elif [ "$wrap_rc" = "3" ]; then
    skip "langchain present but langchain.agents.middleware unavailable (pre-v1?) -- real wrap UNTESTED"
  else
    bad "real wrap test failed"; cat "$LOGS/wrap.log"
  fi
else
  skip "langchain is not importable on this machine -- real wrap_tool_call/awrap_tool_call UNTESTED here"
fi

# ---- 6b. live create_agent loop (scripted model, no API key needed) ------------- #
# The strongest local proof: a real create_agent graph with this middleware
# attached, its model replayed from a script (GenericFakeChatModel with a
# no-op bind_tools, the standard test-double pattern). A blocked tool call
# must surface to the "LLM" as a ToolMessage and the tool double must never
# run; a benign call must reach the tool.
section "live create_agent loop (scripted model)"
if [ "$LC_PRESENT" = "1" ]; then
  MODULE="$MIDDLEWARE" SENTINEL="$TEST_BASE/shell-tool-ran" \
    python3 > "$LOGS/agentloop.log" 2>&1 <<'PYEOF'
import importlib.util, os, sys

spec = importlib.util.spec_from_file_location("operator_middleware", os.environ["MODULE"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.tools import tool
    mw = mod.operator_middleware()
except ImportError as exc:
    print("SKIP: %s" % exc)
    sys.exit(3)

SENTINEL = os.environ["SENTINEL"]

class FakeToolModel(GenericFakeChatModel):
    """Scripted model; bind_tools is a no-op (replies are pre-written)."""
    def bind_tools(self, tools, **kwargs):
        return self

@tool
def run_shell_command(command: str) -> str:
    """Run a shell command (test double: never executes the command)."""
    with open(SENTINEL, "w") as f:
        f.write(command)
    return "tool ran"

def run_agent(scripted):
    agent = create_agent(model=FakeToolModel(messages=iter(scripted)),
                         tools=[run_shell_command], middleware=[mw])
    return agent.invoke({"messages": [("user", "tidy up")]})["messages"]

result = run_agent([
    AIMessage(content="", tool_calls=[{"name": "run_shell_command",
        "args": {"command": "rm -rf ~/demo-factory/canary"}, "id": "call_1"}]),
    AIMessage(content="I will not do that."),
])
tms = [m for m in result if isinstance(m, ToolMessage)]
assert tms, "no ToolMessage reached the model transcript"
assert tms[0].content.startswith("Operator Lite blocked:"), tms[0].content
assert "OP-003" in tms[0].content, tms[0].content
assert not os.path.exists(SENTINEL), "the shell tool RAN despite the block"

run_agent([
    AIMessage(content="", tool_calls=[{"name": "run_shell_command",
        "args": {"command": "ls -la"}, "id": "call_2"}]),
    AIMessage(content="done."),
])
assert os.path.exists(SENTINEL), "benign call never reached the tool"
print("AGENT-LOOP PROOF OK: blocked tool never ran; benign tool ran")
PYEOF
  loop_rc=$?
  if [ "$loop_rc" = "0" ]; then
    ok "create_agent loop: destructive tool call blocked as a ToolMessage, tool never ran; benign call reached the tool"
    grep -q "Operator Lite blocked: OPERATOR: STOPPED (OP-003" "$LOGS/agentloop.log" \
      && ok "the agent transcript carries the gate's verdict verbatim" || true
  elif [ "$loop_rc" = "3" ]; then
    skip "create_agent internals unavailable in this langchain build -- live loop UNTESTED"
  else
    bad "live create_agent loop test failed"; cat "$LOGS/agentloop.log"
  fi
else
  skip "langchain is not importable on this machine -- live create_agent loop UNTESTED here"
fi

# ---- 7. install idempotency ---------------------------------------------------- #
section "install idempotency"
sh "$PORT_DIR/install.sh" > "$LOGS/install2.log" 2>&1 && ok "rerun exited 0" || bad "rerun failed"
cmp -s "$PORT_DIR/operator_middleware.py" "$MIDDLEWARE" && ok "middleware copy still exact" || bad "middleware changed on rerun"
[ "$(cat "$TEST_HOME/.operator/mode")" = "enforce" ] && ok "existing mode kept on rerun" || bad "rerun clobbered mode"

# ---- 8. receipt chain integrity ------------------------------------------------- #
section "receipt chain"
verify_out=$(python3 "$GATE" verify 2>/dev/null)
chain=$(printf '%s\n' "$verify_out" | head -1)
case "$chain" in
  *"chain=OK"*) ok "receipt hash chain verifies ($chain)" ;;
  *) bad "receipt chain: $chain" ;;
esac
case "$verify_out" in
  *"reference_digest_not_signature"*) ok "chain states its own honesty (digests, not signatures)" ;;
  *) bad "verify output missing integrity marker" ;;
esac

# ---- 9. nothing was ever executed ------------------------------------------------ #
[ ! -e "$TEST_HOME/demo-factory" ] && ok "no demo-factory directory was created" \
  || bad "demo-factory exists -- payloads must never execute"
[ ! -e "$TEST_PROJ/src" ] && ok "no ./src directory was created" \
  || bad "./src exists -- payloads must never execute"
[ ! -e "$ORIG_HOME/demo-factory" ] && ok "the real HOME was never touched" \
  || bad "real HOME touched -- hermeticity broken"

# ---- summary ----------------------------------------------------------------------- #
printf '\n'
if [ "$fail" = 0 ]; then
  echo "ALL CHECKS PASSED (hermetic sandbox removed; nothing outside it touched)."
  if [ "$LC_PRESENT" = "1" ]; then
    echo "TESTED with langchain: wrap_tool_call/awrap_tool_call block form, handler"
    echo "pass-through, and a live create_agent loop with a scripted model (tool double"
    echo "never ran on a blocked call)."
    echo "UNTESTED: a create_agent session against a REAL hosted LLM (needs model access) --"
    echo "confirm end-to-end by asking a real agent to run: echo OPERATOR_CANARY_STOP_7f3a"
  else
    echo "UNTESTED: langchain is not installed on this machine -- the real"
    echo "wrap_tool_call/awrap_tool_call plumbing (ToolMessage construction, handler"
    echo "pass-through, async path, live create_agent loop) is written against the"
    echo "verified v1 API but never executed here."
    echo "Install langchain and rerun: pip install langchain && sh test.sh"
  fi
  echo "UNTESTED: a hand-rolled LangGraph StateGraph + ToolNode (decide() + ToolException"
  echo "path) -- ToolNode takes no middleware, so that wiring lives in user code (README)."
else
  echo "SOME TESTS FAILED."
fi
exit "$fail"
