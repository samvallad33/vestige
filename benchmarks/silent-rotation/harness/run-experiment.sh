#!/usr/bin/env bash
#
# run-experiment.sh — the N-TRIAL randomized Silent Rotation benchmark.
#
# Runs N independent trials. Each trial: a fresh RANDOM correct key (name + hex),
# regenerated production corpus + memory seed, then BOTH arms (anarchy, sync)
# against that identical randomized setup. Aggregates into a statistical result
# a skeptic cannot dismiss as luck: "anarchy failed X/N, sync succeeded Y/N",
# a per-trial table, and a one-sided p-value.
#
# Because the correct key is RANDOM per trial and provably absent from the repo
# (leak-audited every trial), nobody can claim the agents were tuned to one key.
# A fixed --master-seed makes the whole experiment exactly reproducible.
#
# Required: OPENAI_API_KEY (openai path). Optional env:
#   N_TRIALS (default 5), MASTER_SEED (default 1337), FLEET_SIZE (default 3),
#   OPENAI_REASONING_EFFORT (default max), OPENAI_REASONING_SUMMARY (default detailed)
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TORTURE_REPO:-$HERE/../torture-v3.5}"
PY="${PYTHON:-python3}"
N_TRIALS="${N_TRIALS:-5}"
MASTER_SEED="${MASTER_SEED:-1337}"
export DEMO_PROFILE=keyring
export PROVIDER="${PROVIDER:-openai}"
export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-max}"
export OPENAI_REASONING_SUMMARY="${OPENAI_REASONING_SUMMARY:-detailed}"
export FLEET_SIZE="${FLEET_SIZE:-3}"

if [ "$PROVIDER" = "openai" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY not set." >&2; exit 2
fi

VESTIGE_BIN="${VESTIGE_BIN:-$HOME/vestige/target/release/vestige}"
export VESTIGE_BIN
EXPDIR="$HERE/experiment-$(printf '%s' "$MASTER_SEED")-$N_TRIALS"
mkdir -p "$EXPDIR"

echo "=============================================================="
echo "  SILENT ROTATION — N-TRIAL RANDOMIZED BENCHMARK"
echo "  trials: $N_TRIALS   master seed: $MASTER_SEED   fleet: $FLEET_SIZE"
echo "  provider: $PROVIDER   reasoning: $OPENAI_REASONING_EFFORT"
echo "  results -> $EXPDIR"
echo "=============================================================="

for T in $(seq 1 "$N_TRIALS"); do
  echo; echo "########## TRIAL $T / $N_TRIALS ##########"
  TRIALDIR="$EXPDIR/trial-$T"; mkdir -p "$TRIALDIR"
  CORPUS="$TRIALDIR/prod-corpus.json"
  MANIFEST="$TRIALDIR/manifest.json"
  # Clear ALL prior results/transcripts so nothing from a previous trial can
  # cross-contaminate this one (stale transcript / stale result JSON).
  rm -f "$HERE"/results/*.json

  # 1. Randomize this trial (fresh key/corpus/seed) + get the correct kid.
  CORRECT_KID="$("$PY" "$HERE/agent/prepare_trial.py" --repo "$REPO" --trial "$T" \
      --master-seed "$MASTER_SEED" --corpus-out "$CORPUS" --manifest-out "$MANIFEST" \
      --vestige-bin "$VESTIGE_BIN" | tail -1)"
  export CORRECT_KID PROD_CORPUS="$CORPUS"
  echo "  correct key (in memory only): $CORRECT_KID"

  # 2. prepare_trial ALREADY wrote the randomized keyring/configs into BOTH the
  #    live repo AND .repo-snapshot, so the snapshot is correct as-is. Do NOT call
  #    reset-repo.sh here: if the repo is under git (it is, for backups), reset-repo
  #    does `git checkout -- .` which REVERTS the randomization to the committed
  #    keys — silently breaking every trial. The snapshot from prepare_trial is
  #    the source of truth for agent checkouts.

  # 3. All arms against the SAME randomized trial (anarchy, then rag, then sync).
  #    run-anarchy/run-rag/run-sync honor TORTURE_REPO + DEMO_PROFILE + CORRECT_KID
  #    + PROD_CORPUS. Set ARMS="anarchy sync" to skip rag (2-arm), default is all 3.
  #    CRITICAL: clear results/ BEFORE each arm so an errored run can NEVER leave a
  #    stale previous-trial JSON for us to copy (which would silently corrupt a trial).
  #    Set VERBOSE=1 to stream each arm's live output (recommended for a single
  #    trial you want to watch); default is quiet so a multi-trial batch isn't a
  #    wall of text. Either way, a per-arm summary line prints below.
  ARMS="${ARMS:-anarchy rag sync}"
  for ARM in $ARMS; do
    echo "  -- $ARM --"
    rm -f "$HERE/results/$ARM.json"
    # EVICT every resident ollama model at the arm boundary. Ollama keeps a model
    # loaded after use, and arms use DIFFERENT local models (zep -> qwen2.5-graphiti
    # ~10GB, hindsight -> llama3.1:8b ~6.8GB). With OLLAMA_NUM_PARALLEL=1 on a single
    # GPU, arm N's residency becomes arm N+1's latency. That is exactly what killed
    # the Jul 21 Kimi run: trial-1 hindsight was healthy (ok=6 err=0, 148s) because it
    # ran BEFORE zep ever loaded qwen2.5-graphiti; by trial 2 that 10GB model was still
    # resident, hindsight's retain() contended against it, blew the 900s ingest
    # timeout, and the arm reported ok=0 err=8 memory_layer_alive=False. The bug only
    # appears from the SECOND trial onward, so it reads as random flakiness.
    "$PY" - <<'EVICT' 2>/dev/null || true
import json, urllib.request
try:
    with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=5) as r:
        models = [m["name"] for m in json.load(r).get("models", [])]
except Exception:
    models = []
for name in models:
    # keep_alive=0 tells ollama to unload this model immediately.
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": name, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30).read()
        print(f"    (evicted {name} from ollama)")
    except Exception as exc:
        print(f"    (could not evict {name}: {type(exc).__name__})")
EVICT
    if [ "${VERBOSE:-0}" = "1" ]; then
      TORTURE_REPO="$REPO" bash "$HERE/run-$ARM.sh" 2>&1 | sed 's/^/    /' || true
    else
      TORTURE_REPO="$REPO" bash "$HERE/run-$ARM.sh" >/dev/null 2>&1 || true
    fi
    if [ -f "$HERE/results/$ARM.json" ]; then
      cp "$HERE/results/$ARM.json" "$TRIALDIR/$ARM.json"
      # Always print a one-line per-arm summary (verdict + each agent's chosen key)
      # so a quiet run still shows what happened without opening the JSON.
      "$PY" -c "
import json
d=json.load(open('$TRIALDIR/$ARM.json'))
dirs=d.get('fix_directions',{})
bf=sum(1 for a in d.get('agents',[]) if a.get('used_vestige_backfill'))
rg=sum(1 for a in d.get('agents',[]) if a.get('used_rag_search'))
print('     -> %s | keys=%s | backfill=%d rag=%d | prod_replay=%s' % (
    d.get('fleet_verdict','?'), dirs, bf, rg, d.get('prod_replay_pass')))
" 2>/dev/null || true
      # SHOW_REASONING=1 -> print EVERY agent's FULL chain-of-thought for this arm,
      # per trial. This is the WHY a skeptic/investor needs to SEE: bare agents
      # rationalizing a guess vs Vestige agents citing the retrieved key. The
      # reasoning is captured per-turn in the transcript-<arm>-a*.json files; this
      # surfaces it live instead of leaving it buried on disk.
      if [ "${SHOW_REASONING:-0}" = "1" ]; then
        for TR in "$HERE"/results/transcript-"$ARM"-a*.json; do
          [ -f "$TR" ] || continue
          "$PY" -c "
import json, os, sys
f = '$TR'
try:
    d = json.load(open(f))
except Exception:
    sys.exit(0)
aid = os.path.basename(f).replace('transcript-', '').replace('.json', '')
print()
print('     ' + '=' * 72)
print('       REASONING  [%s]  status=%s  key=%s' % (
    aid, d.get('status', '?'),
    (d.get('final_contested_signatures') or {})))
print('     ' + '=' * 72)
for i, t in enumerate(d.get('turns', []), 1):
    r = (t.get('reasoning') or '').strip()
    tools = ', '.join(c.get('name', '?') for c in t.get('tool_calls', []))
    if r:
        for line in r.splitlines():
            print('       ' + line)
    if tools:
        print('       -> tools: ' + tools)
    if r or tools:
        print()
" 2>/dev/null || true
        done
      fi
    else
      echo "{\"fleet_verdict\":\"errored\",\"note\":\"run-$ARM produced no result JSON\"}" > "$TRIALDIR/$ARM.json"
      echo "     -> ERRORED (no result JSON)"
    fi
  done

  # copy transcripts for this trial (the WHY, per agent)
  cp "$HERE"/results/transcript-*.json "$TRIALDIR/" 2>/dev/null || true

  LINE="  trial $T:"
  for ARM in $ARMS; do
    V="$("$PY" -c "import json;print(json.load(open('$TRIALDIR/$ARM.json')).get('fleet_verdict','?'))")"
    LINE="$LINE  $ARM=$V"
  done
  echo "$LINE"
done

echo; echo "########## AGGREGATE ##########"
"$PY" "$HERE/aggregate-experiment.py" "$EXPDIR" "$N_TRIALS" | tee "$EXPDIR/SUMMARY.txt"
