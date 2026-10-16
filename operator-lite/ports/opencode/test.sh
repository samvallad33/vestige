#!/bin/sh
# Operator Lite — OpenCode port test harness.
#
# 1. Gate contract, end-to-end, with OpenCode-shaped payloads (tool_name "bash"):
#      plain rm -rf  -> expect exit 2 (STOP)
#      r''m obfuscation -> expect exit 2 (STOP)
#      benign ls     -> expect exit 0 (silent)
#    Test command strings are NEVER executed: they travel only as JSON on the
#    gate's stdin. Runs against a throwaway OPERATOR_HOME/HOME so no real
#    receipts are written.
# 2. Adapter contract (plugin hook driven directly, no host) when node or bun
#    exists: block / canary / benign / gate-unreachable cases.
# 3. If the `opencode` CLI exists: documented plugin-discovery validation
#    (`opencode debug config`, `opencode debug startup`). Without the CLI the
#    plugin-in-host layer is reported UNTESTED.
set -u

SRC_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PASS=0
FAIL=0

note() { printf '%s\n' "$*"; }
ok() { PASS=$((PASS + 1)); printf 'PASS  %s\n' "$*"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL  %s\n' "$*"; }

# ------------------------------------------------------------- locate gate ---
GATE="${OPERATOR_GATE:-}"
if [ -z "$GATE" ]; then
  for cand in \
    "$SRC_DIR/../../operator-gate.py" \
    "$HOME/.operator/gate/operator-gate.py" \
    "$SRC_DIR/operator-gate.py"; do
    if [ -f "$cand" ]; then GATE=$cand; break; fi
  done
fi
if [ -z "$GATE" ] || [ ! -f "$GATE" ]; then
  note "FATAL: operator-gate.py not found (set OPERATOR_GATE or run from the repo)"
  exit 1
fi
note "gate:     $GATE"
note "plugin:   $SRC_DIR/index.js"
note ""

command -v python3 >/dev/null 2>&1 || { note "FATAL: python3 not found"; exit 1; }

TD=$(mktemp -d "${TMPDIR:-/tmp}/opgate-test.XXXXXX")
TESTROOT=""
trap 'rm -rf "$TD" ${TESTROOT:+"$TESTROOT"}' EXIT INT TERM
mkdir -p "$TD"
[ -n "${HOME:-}" ] || { note "FATAL: HOME is not set; cannot build a hermetic protected root"; exit 1; }
export OPERATOR_HOME="$TD/.operator"      # isolate receipts from the real home
export OPERATOR_GATE_MODE=enforce         # force STOP decisions for this run
# Hermetic HOME must NOT live under /tmp|/var/folders|TMPDIR: the gate treats
# those as scratch (is_scratch), which would legitimately suppress OP-003/OP-004
# and turn the corpus run into a false negative.
TESTROOT="${HOME%/}/.operator-lite-port-test.$$"
GH="$TESTROOT/home"
mkdir -p "$GH/Developer"
export HOME="$GH"                         # hermetic protected roots

# ------------------------------------------- 1. gate contract (opencode shape)
note "== 1. gate contract (python3 gate hook --source opencode) =="

make_payload() {
  # $1 = output file, $2 = command string, $3 = cwd
  python3 - "$1" "$2" "$3" <<'PYEOF'
import json, sys
out, command, cwd = sys.argv[1], sys.argv[2], sys.argv[3]
with open(out, "w") as f:
    json.dump({"tool_name": "bash", "tool_input": {"command": command},
               "cwd": cwd, "session_id": "opencode-port-test"}, f)
PYEOF
}

run_case() {
  # $1 = label, $2 = expect exit (0|2), $3 = payload file, $4 = extra grep (optional)
  label=$1
  stderr_file="$TD/stderr.$4"
  python3 "$GATE" hook --source opencode < "$3" > "$TD/stdout.$4" 2> "$stderr_file"
  code=$?
  if [ "$code" -eq 0 ]; then verdict=ALLOW; else verdict="EXIT $code"; fi
  note "  [$label] gate exit=$code stderr:"
  sed 's/^/    /' "$stderr_file"
  if [ "$code" = "$2" ]; then
    ok "$label (exit $code)"
  else
    bad "$label (expected exit $2, got $code)"
  fi
}

make_payload "$TD/p1.json" 'rm -rf ~/Developer/vestige' "$HOME/Developer"
run_case "plain rm -rf of protected root -> STOP" 2 "$TD/p1.json" c1

make_payload "$TD/p2.json" "r''m -rf ~/Developer/vestige" "$HOME/Developer"
run_case "obfuscated r''m -> STOP (quote-stripped)" 2 "$TD/p2.json" c2

make_payload "$TD/p3.json" 'ls -la' "$HOME/Developer"
run_case "benign ls -> ALLOW (silent)" 0 "$TD/p3.json" c3

# bonus: the gate's own GuardFall bypass corpus, when it sits next to the gate
CORPORA_DIR=$(dirname -- "$GATE")/corpora
if [ -f "$CORPORA_DIR/guardfall.json" ]; then
  note ""
  note "== 1b. gate bypass corpus (46 cases, classification only) =="
  if python3 "$GATE" corpus guardfall; then
    ok "guardfall corpus"
  else
    bad "guardfall corpus"
  fi
fi

# --------------------------------------------- 2. adapter contract (no host) --
RUNNER=""
if command -v bun >/dev/null 2>&1; then RUNNER="bun"
elif command -v node >/dev/null 2>&1; then RUNNER="node"
fi
note ""
note "== 2. adapter contract (index.js hook driven directly, runner: ${RUNNER:-none}) =="

if [ -z "$RUNNER" ]; then
  note "SKIP  node/bun not found; adapter contract untested here"
else
  AD="$TD/adapter"
  mkdir -p "$AD"
  cp "$SRC_DIR/index.js" "$AD/index.js"
  printf '{"type":"module"}\n' > "$AD/package.json"
  cat > "$AD/driver.mjs" <<'DRIVEREOF'
import { OperatorLiteGate } from "./index.js";

const kase = process.argv[2];
const input = { tool: "bash", sessionID: "sess-opencode-port-test", callID: "call-test" };
const cases = {
  block: {
    args: { command: "rm -rf $HOME/Developer/vestige" },
    expect: "throw", contains: "Operator Lite blocked:",
  },
  canary: {
    args: { command: "echo OPERATOR_CANARY_STOP_7f3a" },
    expect: "throw", contains: "Operator Lite blocked:",
  },
  benign: {
    args: { command: "ls -la" },
    expect: "allow",
  },
  unreachable: {
    env: { OPERATOR_GATE: "/nonexistent/operator-gate.py" },
    args: { command: "rm -rf ~/Developer/vestige" },
    expect: "throw", contains: "gate unreachable",
  },
};
const c = cases[kase];
if (!c) { console.log("RESULT: unknown case"); process.exit(3); }
if (c.env) Object.assign(process.env, c.env);
process.env.OPERATOR_GATE_MODE = "enforce";

const plugin = await OperatorLiteGate({ directory: process.cwd(), client: undefined });
const hook = plugin["tool.execute.before"];
if (typeof hook !== "function") { console.log("RESULT: hook missing"); process.exit(3); }

try {
  await hook(input, { args: c.args });
  if (c.expect === "allow") { console.log("RESULT: ALLOW (no throw, as expected)"); process.exit(0); }
  console.log("RESULT: MISMATCH — expected a blocking throw, hook returned normally");
  process.exit(3);
} catch (err) {
  const msg = String(err && err.message ? err.message : err);
  if (c.expect === "throw" && msg.startsWith("Operator Lite blocked:") && msg.includes(c.contains)) {
    console.log("RESULT: BLOCKED as expected");
    console.log("  error message: " + msg.slice(0, 400));
    process.exit(0);
  }
  console.log("RESULT: MISMATCH — unexpected error: " + msg.slice(0, 400));
  process.exit(3);
}
DRIVEREOF

  export OPERATOR_GATE="$GATE"
  for kase in block canary benign unreachable; do
    out=$("$RUNNER" "$AD/driver.mjs" "$kase" 2>&1)
    rc=$?
    note "$out" | sed 's/^/  /'
    if [ "$rc" -eq 0 ]; then
      ok "adapter $kase"
    else
      bad "adapter $kase (runner exit $rc)"
    fi
  done
fi

# ------------------------------------------------- 3. plugin-in-host checks ---
note ""
note "== 3. plugin-in-host (opencode CLI) =="
if command -v opencode >/dev/null 2>&1; then
  HP="$TD/hostproj"
  mkdir -p "$HP/.opencode/plugin"
  cp "$SRC_DIR/index.js" "$HP/.opencode/plugin/operator-lite.js"
  cp "$GATE" "$HP/operator-gate.py"
  ( cd "$HP" && OPERATOR_GATE="$HP/operator-gate.py" opencode debug config > "$TD/cfg.json" 2>"$TD/cfg.err" )
  if grep -q "operator-lite.js" "$TD/cfg.json" 2>/dev/null; then
    ok "opencode debug config discovers the plugin (plugin/plugin_origins)"
    grep -A1 '"plugin_origins"' "$TD/cfg.json" | head -4 | sed 's/^/  /'
  else
    bad "opencode debug config did not list the plugin"
    sed 's/^/  /' "$TD/cfg.err" | head -5
  fi
  if ( cd "$HP" && OPERATOR_GATE="$HP/operator-gate.py" opencode debug startup --print-logs --log-level DEBUG >/dev/null 2>&1 ); then
    ok "opencode debug startup boots clean with the plugin present (smoke)"
  else
    bad "opencode debug startup failed with the plugin present"
  fi
  note ""
  note "  UNTESTED-IN-HOST: a live blocking decision inside a real OpenCode"
  note "  session (needs a configured model). Verify manually with the canary:"
  note "    echo OPERATOR_CANARY_STOP_7f3a   (shadow: logs a STOP receipt)"
else
  note "SKIP  opencode CLI not found — plugin-in-host: UNTESTED."
  note "      Gate contract and adapter contract above still fully cover the"
  note "      documented behavior; wire-up is per upstream docs."
fi

note ""
note "== summary: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
