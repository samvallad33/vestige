#!/usr/bin/env bash
#
# verify-arms.sh -- PROVE every benchmark arm's memory layer is CALLABLE and
# returns REAL DATA before a single paid token is spent.
#
# WHY THIS EXISTS
# ---------------
# On Jul 20 2026 an adversarial audit of a completed, already-paid-for 7-arm run
# found that THREE of the six competitor memory layers had never successfully
# retrieved anything, in any recorded run:
#
#   hindsight   19/19 retrieval calls returned an ERROR string   (0 successes)
#   zep         22/22 retrieval calls returned an ERROR string   (0 successes)
#   mem0        14/15 retrieval calls returned an ERROR string   (1 success)
#
# Every one of those was scored as a substantive retrieval LOSS and aggregated
# into a published "competitor 0/5" column. They were not losses. They were
# crashes: an IPv6 port collision, a shared asyncio loop driven from three
# threads, and an ingest that exceeded its tool-call timeout. The result JSON
# reported agents_errored=0 for all of them, because the harness only ever
# tracked Vestige's own tool usage.
#
# A benchmark that cannot tell "the competitor retrieved and chose wrong" from
# "the competitor's backend never answered" is not measuring what it claims to
# measure. This script is the gate that makes that failure mode impossible to
# ship again. Run it, get a green matrix, THEN spend money.
#
# WHAT IT CHECKS, PER ARM
#   1. single-threaded  -- the tool returns real content, not an "ERROR:" string
#   2. 3 concurrent threads -- the ACTUAL fleet condition (fleet_runner.py runs
#      FLEET_SIZE agents in a ThreadPoolExecutor). Every one of the three dead
#      arms above passed condition 1 at some point and failed condition 2.
#
# USAGE
#   bash verify-arms.sh                 # all 7 arms
#   bash verify-arms.sh rag sync zep    # a subset
#
# Exit 0 = every checked arm is READY. Non-zero = do NOT start a paid run.
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v3.5}"
PY="${PYTHON:-$HERE/.venv-arms/bin/python}"
[ -x "$PY" ] || PY=python3
VESTIGE_BIN="${VESTIGE_BIN:-$HOME/vestige/target/release/vestige}"
MASTER_SEED="${MASTER_SEED:-1337}"
TRIAL="${TRIAL:-1}"

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(anarchy rag sync mem0 supermemory hindsight zep)
fi

echo "=============================================================="
echo "  ARM LIVENESS GATE -- \$0, no paid API"
echo "  arms: ${ARMS[*]}"
echo "  seed: $MASTER_SEED  trial: $TRIAL"
echo "=============================================================="

# --- 0. Dependencies the local arms need --------------------------------------
fail=0
echo
echo "-- service preflight --"
# NOTE: use a real socket connect, NOT bash's /dev/tcp. Under zsh/bash subshells
# `(exec 3<>/dev/tcp/host/port)` reports a false failure for services that are
# demonstrably up -- it reported FalkorDB DEAD while 127.0.0.1:6379 was
# answering PING with +PONG. A false DEAD here would block a run for no reason.
check_port() {  # name host port
  if "$PY" -c "
import socket,sys
try:
    socket.create_connection(('$2', $3), timeout=5).close()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    echo "   OK   $1  ($2:$3)"
  else
    echo "   DEAD $1  ($2:$3)  <-- required by an arm below"
    fail=1
  fi
}
check_port "ollama       " 127.0.0.1 11434
check_port "hindsight-api" 127.0.0.1 8899
check_port "supermemory  " 127.0.0.1 6767
check_port "falkordb(zep)" 127.0.0.1 6379

# hindsight-api binds IPv4 ONLY. If anything holds the IPv6 wildcard on its
# port, "localhost" resolves to ::1 first and every call silently hits the
# wrong server -- this actually happened (a stray `python -m http.server 8899`
# answered HTTP 501 to every hindsight request while the real service was
# healthy on IPv4). Detect the collision explicitly rather than debugging it
# again from a confusing error message.
if command -v lsof >/dev/null 2>&1; then
  listeners=$(lsof -nP -iTCP:8899 -sTCP:LISTEN -t 2>/dev/null | sort -u | wc -l | tr -d ' ')
  if [ "${listeners:-0}" -gt 1 ]; then
    echo "   WARN  more than one process is listening on :8899 -- IPv6/IPv4 collision risk"
    lsof -nP -iTCP:8899 -sTCP:LISTEN 2>/dev/null | sed 's/^/         /'
    fail=1
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "PREFLIGHT FAILED -- start the missing services before running the gate."
  exit 2
fi

# --- 1. Seed a real trial so the arms have a real corpus ----------------------
echo
echo "-- seeding trial $TRIAL (seed $MASTER_SEED) --"
SCRATCH="$(mktemp -d)"
CORRECT_KID="$("$PY" "$HERE/agent/prepare_trial.py" --repo "$REPO" --trial "$TRIAL" \
    --master-seed "$MASTER_SEED" --corpus-out "$SCRATCH/prod-corpus.json" \
    --manifest-out "$SCRATCH/manifest.json" --vestige-bin "$VESTIGE_BIN" | tail -1)"
if [ -z "${CORRECT_KID:-}" ]; then
  echo "   ERROR: prepare_trial.py produced no correct key"; exit 2
fi
echo "   correct key (memory only): $CORRECT_KID"

export VESTIGE_BIN VESTIGE_DATA_DIR="${VESTIGE_DATA_DIR:-$REPO/.vestige-demo-db}"
SEED="$REPO/.vestige-seed.sh"
if [ -f "$SEED" ]; then
  VESTIGE_BIN="$VESTIGE_BIN" VESTIGE_DATA_DIR="$VESTIGE_DATA_DIR" bash "$SEED" >/dev/null 2>&1
  echo "   vestige DB seeded -> $VESTIGE_DATA_DIR"
fi
export CORRECT_KID PROD_CORPUS="$SCRATCH/prod-corpus.json"

# --- 2. Probe every arm, single-threaded AND under fleet concurrency ----------
echo
"$PY" "$HERE/tests/arm_liveness.py" "${ARMS[@]}"
rc=$?

echo
if [ "$rc" -eq 0 ]; then
  echo "=============================================================="
  echo "  ALL CHECKED ARMS READY -- safe to start a paid run."
  echo "=============================================================="
else
  echo "=============================================================="
  echo "  ONE OR MORE ARMS ARE BROKEN -- DO NOT START A PAID RUN."
  echo "  A broken arm publishes as a competitor 0/N that is really a crash."
  echo "=============================================================="
fi
rm -rf "$SCRATCH"
exit "$rc"
