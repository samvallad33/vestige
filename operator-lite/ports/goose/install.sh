#!/bin/sh
# Operator Lite -- Goose port installer.
#
# Owner-side script: installs the gate to ~/.operator/gate/ (downloaded from
# the Vestige repo if no bundled copy is present), wires the Goose lifecycle
# hook plugin into ~/.agents/plugins/operator-lite/ (Open Plugins spec,
# auto-discovered by goose >= v1.41.0), sets shadow mode if unset, and prints
# next steps.
#
# POSIX sh. No arguments required.
set -eu

if [ -n "${OPERATOR_AGENT_SESSION:-}" ]; then
  echo "Refusing: installs are run by the owner in their own terminal."
  exit 3
fi

REPO_RAW="https://raw.githubusercontent.com/samvallad33/vestige/main/operator-lite/operator-gate.py"

SRC_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OP_HOME="${OPERATOR_HOME:-$HOME/.operator}"
GATE_DIR="$OP_HOME/gate"
GATE="$GATE_DIR/operator-gate.py"
PLUGIN_DIR="$HOME/.agents/plugins/operator-lite"

# ---------------------------------------------------------------- gate -------
# Never overwrites an existing gate; upgrades are the owner's call.
if [ -f "$GATE" ]; then
  echo "gate:  already installed at $GATE (left untouched)"
else
  mkdir -p "$GATE_DIR"
  if [ -f "$SRC_DIR/operator-gate.py" ]; then
    cp "$SRC_DIR/operator-gate.py" "$GATE"
    echo "gate:  installed bundled copy -> $GATE"
  elif [ -f "$SRC_DIR/../../operator-gate.py" ]; then
    # a repo checkout: operator-lite/ports/goose -> operator-lite/operator-gate.py
    cp "$SRC_DIR/../../operator-gate.py" "$GATE"
    echo "gate:  installed from checkout -> $GATE"
  else
    TMP="$GATE_DIR/.operator-gate.py.$$"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "$REPO_RAW" -o "$TMP"
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "$TMP" "$REPO_RAW"
    else
      echo "error: need curl or wget (or a bundled operator-gate.py) to install the gate" >&2
      rm -f "$TMP"
      exit 1
    fi
    mv "$TMP" "$GATE"
    echo "gate:  downloaded -> $GATE"
  fi
  chmod 755 "$GATE"
  if command -v python3 >/dev/null 2>&1; then
    # syntax check; refuses to wire a broken gate
    python3 -c 'import sys; compile(open(sys.argv[1], encoding="utf-8").read(), sys.argv[1], "exec")' "$GATE"
  else
    echo "warning: python3 not found on PATH; the hook will fail-open on every tool call" >&2
  fi
fi

# ------------------------------------------------------------ mode file ------
mkdir -p "$OP_HOME"
if [ ! -f "$OP_HOME/mode" ]; then
  echo shadow > "$OP_HOME/mode"
  echo "mode:  shadow (log-only) written to $OP_HOME/mode"
else
  echo "mode:  existing $(cat "$OP_HOME/mode") left untouched ($OP_HOME/mode)"
fi

# -------------------------------------------------------- hook plugin --------
# goose auto-discovers ~/.agents/plugins/<name>/hooks/hooks.json (user scope)
# and runs each command with PLUGIN_ROOT set to the plugin directory.
mkdir -p "$PLUGIN_DIR/hooks" "$PLUGIN_DIR/scripts"
cp "$SRC_DIR/hooks.json" "$PLUGIN_DIR/hooks/hooks.json"
cp "$SRC_DIR/operator-hook.py" "$PLUGIN_DIR/scripts/operator-hook.py"
chmod 755 "$PLUGIN_DIR/scripts/operator-hook.py"
echo "hook:  installed -> $PLUGIN_DIR/ (PreToolUse -> operator-hook.py -> gate)"

cat <<EOF

Next steps:
  1. Restart goose (hooks are discovered at startup). Requires goose v1.41.0
     or newer (hooks with PreToolUse denial, released 2026-07-03).
  2. Verify the wiring from this directory:
       sh test.sh
     or end-to-end inside goose, ask it to run:
       echo OPERATOR_CANARY_STOP_7f3a
     Shadow mode logs the STOP to ~/.operator/receipts/ without blocking;
     after flipping to enforce, goose shows the gate's OPERATOR: STOPPED text
     and the tool call never runs.
  3. When the receipts look right, flip to enforce:
       echo enforce > $OP_HOME/mode
  4. Optional (strict fail-closed): if the hook itself cannot run you may
     prefer to block everything rather than allow. Edit
     $PLUGIN_DIR/hooks/hooks.json and add
       "on_failure": "block"
     inside the hook object (goose v1.48.0+).
  5. Uninstall: remove the plugin directory
       rm -rf $PLUGIN_DIR
EOF
