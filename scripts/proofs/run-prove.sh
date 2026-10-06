#!/usr/bin/env bash
# vestige prove on the four real regressions. Prints one RESULT line per case.
# The known culprit is an oracle recorded below from the upstream report or
# the commit the project itself identified. It is not fed to vestige.
set -euo pipefail

VESTIGE="${VESTIGE:-./target/release/vestige}"
ROOT="${PROVE_REPOS:-/tmp/vestige-prove}"
OUT="${PROVE_OUT:-/tmp/vestige-prove-out}"
REPRO="${PROVE_REPRO:-$(cd "$(dirname "$0")" && pwd)/repros}"
ONLY="${PROVE_ONLY:-}"
mkdir -p "$OUT"

want() {
  [[ -z "$ONLY" || ",$ONLY," == *",$1,"* ]]
}

run_case() {
  local name="$1" repo="$2" good="$3" bad="$4" reported="$5" culprit="$6" test="$7"
  shift 7
  local extra=("$@")
  local data="$OUT/$name-data"
  local report="$OUT/$name.json"
  rm -rf "$data"
  mkdir -p "$data"
  local ingest
  ingest="$("$VESTIGE" --data-dir "$data" ingest "failure: $name" --tags failure --node-type event)"
  local id
  id="$(printf '%s\n' "$ingest" | awk '/Node ID/{print $NF; exit}')"
  if [[ -z "$id" ]]; then
    echo "RESULT name=$name status=ingest-failed" >&2
    printf '%s\n' "$ingest" >&2
    return 1
  fi
  local start end elapsed code=0
  start="$(date +%s)"
  set +e
  "$VESTIGE" --data-dir "$data" prove \
    --logged-write "$id" \
    --repo "$repo" \
    --good "$good" \
    --bad "$bad" \
    --test "$test" \
    --reported-at "$reported" \
    --report "$report" \
    "${extra[@]}"
  code=$?
  set -e
  end="$(date +%s)"
  elapsed="$((end - start))"
  local found="none" check="not-run"
  if [[ -f "$report" ]]; then
    found="$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r.get("first_bad_commit") or "none")' "$report")"
    if "$VESTIGE" prove --check "$report" >"$OUT/$name.check" 2>&1; then
      check="verified"
    else
      check="failed"
    fi
  fi
  local match="n/a"
  if [[ -n "$culprit" && "$culprit" != "UNSTATED" && "$found" != "none" ]]; then
    if [[ "$found" == "$culprit"* || "$culprit" == "$found"* ]]; then
      match="yes"
    else
      match="no"
    fi
  elif [[ "$culprit" == "UNSTATED" ]]; then
    match="upstream-did-not-name-a-sha"
  fi
  echo "RESULT name=$name exit=$code wall_s=$elapsed first_bad=$found known=$culprit match=$match receipt=$check report=$report memory=$id"
}

# A culprit below is a comparison oracle. It is not passed to vestige.
# UNSTATED means the upstream report did not name a commit. The go-git and
# dayjs SHAs were identified by reading the change and running the same test
# on the commit and its parent before prove.

if want redis-py-4026; then
  run_case redis-py-4026 \
    "$ROOT/redis-py" \
    "${REDIS_GOOD:-v7.1.0}" \
    "${REDIS_BAD:-v7.4.0}" \
    "2026-04-06T19:21:50Z" \
    "${REDIS_CULPRIT:-UNSTATED}" \
    "python3 $REPRO/redis_4026.py"
fi

if want go-git-2322; then
  run_case go-git-2322 \
    "$ROOT/go-git" \
    "${GOGIT_GOOD:-v5.19.0}" \
    "${GOGIT_BAD:-v5.19.1}" \
    "2026-08-17T12:25:42Z" \
    "${GOGIT_CULPRIT:-680cc5d894290bfd6fe2da935c2f3d1f6399ec6e}" \
    "sh $REPRO/go_git_2322.sh"
fi

if want dayjs-3123; then
  run_case dayjs-3123 \
    "$ROOT/dayjs" \
    "${DAYJS_GOOD:-73513ec477a969e3ef716728798903f1803d8e2d}" \
    "${DAYJS_BAD:-6609c5e54347a60bf8d327dee36190069012ba47}" \
    "2026-06-09T04:33:26Z" \
    "${DAYJS_CULPRIT:-a7f858bb70ad81f718ba35c479e84b54eace48b2}" \
    "sh $REPRO/dayjs_3123.sh"
fi

if want winston-800; then
  # The scored harness does not separate 1.0.0 from 2.1.1 (see the proof doc).
  # WINSTON_RUN=1 runs the flaky bisect anyway.
  if [[ "${WINSTON_RUN:-}" != "1" ]]; then
    echo "RESULT name=winston-800 status=not-run reason=endpoints-not-separated good=1.0.0 bad=2.1.1"
  else
    run_case winston-800 \
      "$ROOT/winston" \
      "${WINSTON_GOOD:-1.0.0}" \
      "${WINSTON_BAD:-2.1.1}" \
      "2016-02-04T18:16:07Z" \
      "${WINSTON_CULPRIT:-UNSTATED}" \
      "sh $REPRO/winston_800.sh" \
      --flaky --baseline-max 40 --max-runs 80 --strength-runs 30
  fi
fi
