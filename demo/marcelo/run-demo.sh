#!/usr/bin/env bash
# Runnable demo for the uv #10186 regression recorded in the repository tests.
# Vestige is a cognitive, deterministic memory-transaction security OS for AI agents.
set -euo pipefail

# Clone and build are the only steps that talk to the network. A surrounding
# shell may have turned lazy fetch or cargo's network off; lift that for
# these two steps, then pin both off again once the binaries exist.
unset GIT_NO_LAZY_FETCH
unset CARGO_NET_OFFLINE

PAUSE_SECONDS=2

pause() {
  if [[ "${DEMO_PAUSE:-0}" == "1" ]]; then
    sleep "$PAUSE_SECONDS"
  fi
}

say() {
  printf '\n%s\n' "$1"
  pause
}

die() {
  printf 'stopped: %s\n' "$1" >&2
  exit 1
}

ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vestige-marcelo.XXXXXX")"
VESTIGE="$ROOT/vestige"
UV="$ROOT/uv"
DATA="$ROOT/data"
STDERR_LOG="$ROOT/mcp-stderr.txt"
export ROOT
python3 -c 'import os, time; open(os.path.join(os.environ["ROOT"], "t0"), "w").write(repr(time.perf_counter()))'

printf 'work directory: %s\n' "$ROOT"

say "This step downloads the Vestige source into a temporary directory and checks out the pre-release commit."
git clone --filter=blob:none https://github.com/samvallad33/vestige.git "$VESTIGE"
git -C "$VESTIGE" fetch --filter=blob:none origin b4bcd52b81402722d6c99b96f461278a171d110d
git -C "$VESTIGE" checkout b4bcd52b81
HEAD_FULL="$(git -C "$VESTIGE" rev-parse HEAD)"
[[ "$HEAD_FULL" == "b4bcd52b81402722d6c99b96f461278a171d110d" ]] || die "checkout is $HEAD_FULL"
printf 'This is pre-release code from PR #445 (commit b4bcd52b81), not the v4.1.1 release.\n'
printf 'HEAD %s\n' "$HEAD_FULL"
git -C "$VESTIGE" remote remove origin

say "This step downloads the uv checkout at the revision the regression test reads."
git init "$UV"
git -C "$UV" remote add origin https://github.com/astral-sh/uv.git
git -C "$UV" fetch --depth=40 origin b52d48973fe9ddb2e78b663ec48a1a68f7e7802d
git -C "$UV" checkout --detach FETCH_HEAD
UV_HEAD="$(git -C "$UV" rev-parse HEAD)"
[[ "$UV_HEAD" == "b52d48973fe9ddb2e78b663ec48a1a68f7e7802d" ]] || die "uv checkout is $UV_HEAD"
git -C "$UV" cat-file -e 351d602d86c484a39bc537f1eb99866ea2c25fc1^{commit}
git -C "$UV" cat-file -e d2f58d92991fa08b24596fcc6c6472dc5015d3bc^{commit}
printf 'uv HEAD %s\n' "$UV_HEAD"
git -C "$UV" remote remove origin

say "This step compiles the default-feature binaries. After it finishes, the rest of the run stays on local files."
(
  cd "$VESTIGE"
  cargo build -p vestige-mcp
)
VESTIGE_BIN="$VESTIGE/target/debug/vestige"
MCP_BIN="$VESTIGE/target/debug/vestige-mcp"
[[ -x "$VESTIGE_BIN" && -x "$MCP_BIN" ]] || die "binaries were not produced"
export CARGO_NET_OFFLINE=true
export GIT_NO_LAZY_FETCH=1
export GIT_TERMINAL_PROMPT=0

say "This step creates a fresh empty data directory for this run."
mkdir -p "$DATA"
if [[ -n "$(find "$DATA" -mindepth 1 -print -quit)" ]]; then
  die "data directory was not empty"
fi
printf 'data directory: %s\n' "$DATA"

export DEMO_ROOT="$ROOT"
export DEMO_UV="$UV"
export DEMO_DATA="$DATA"
export DEMO_MCP="$MCP_BIN"
export DEMO_STDERR="$STDERR_LOG"

python3 - <<'PY'
import json
import os
import subprocess
import sys
import time

DATA = os.environ["DEMO_DATA"]
UV = os.environ["DEMO_UV"]
MCP = os.environ["DEMO_MCP"]
STDERR = os.environ["DEMO_STDERR"]
PAUSE = os.environ.get("DEMO_PAUSE", "0") == "1"
SCOPE = "uv-10186"
REV = "b52d48973fe9ddb2e78b663ec48a1a68f7e7802d"
FAILURE_SHA = "351d602d86c484a39bc537f1eb99866ea2c25fc1"
CAUSE_SHA = "d2f58d92991fa08b24596fcc6c6472dc5015d3bc"
SYMPTOM = (
    "failure: uv publish raises `error decoding response body` on 0.5.12. "
    "0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/"
    "12509313775/job/34898612613#step:8:12"
)
RUN_ID = (
    "https://github.com/andrew000/FTL-Extract/actions/runs/"
    "12509313775/job/34898612613#step:8:12"
)
ALLOWED = ("closed_by", "derived_from", "evidence_of", "touched", "corrects")


def say(text):
    print()
    print(text)
    if PAUSE:
        time.sleep(2)


def stop(text):
    print(f"stopped: {text}", file=sys.stderr)
    sys.exit(1)


class Server:
    def __init__(self):
        env = os.environ.copy()
        env["VESTIGE_DATA_DIR"] = DATA
        env["VESTIGE_HTTP_ENABLED"] = "0"
        env["VESTIGE_TRACE"] = "0"
        env["VESTIGE_BACKFILL_AUTOFIRE"] = "0"
        env["VESTIGE_FAILURE_FEEDBACK"] = "0"
        env["VESTIGE_DREAM_COMPILE_AUTOFIRE"] = "0"
        env.pop("RUST_LOG", None)
        err = open(STDERR, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [MCP, "--data-dir", DATA],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            env=env,
            text=True,
            bufsize=1,
        )
        self.err = err
        self.next_id = 0

    def rpc(self, method, params=None):
        self.next_id += 1
        ident = self.next_id
        message = {"jsonrpc": "2.0", "id": ident, "method": method}
        if params is not None:
            message["params"] = params
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                stop("the server closed its output")
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                stop("the server wrote a non-json line")
            if "id" not in value:
                continue
            if value.get("id") != ident:
                stop("the server answered a different request")
            if "error" in value:
                stop(f"{method} failed")
            return value["result"]

    def handshake(self):
        self.rpc(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "marcelo-demo", "version": "1"},
            },
        )
        note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self.proc.stdin.write(json.dumps(note) + "\n")
        self.proc.stdin.flush()

    def tool(self, name, arguments):
        result = self.rpc(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        if result.get("isError") is True:
            stop(f"{name} returned an error")
        body = result.get("structuredContent")
        if body is None:
            content = result.get("content") or []
            if not content or "text" not in content[0]:
                stop(f"{name} returned no body")
            body = json.loads(content[0]["text"])
        if isinstance(body, dict) and body.get("error") and "nodes" not in body:
            stop(f"{name} returned an error")
        return body

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.err.close()


def path_edges(cause):
    found = []
    for hop in cause.get("path") or []:
        kind = hop.get("link_type")
        if kind in ALLOWED and kind not in found:
            found.append(kind)
    return found


say("This step records the same repository history the regression test records.")
server = Server()
try:
    server.handshake()
    finished = False
    for round_no in range(1, 9):
        out = server.tool(
            "codebase",
            {
                "action": "ingest_repo",
                "repoPath": UV,
                "codebase": SCOPE,
                "scope": SCOPE,
                "rev": REV,
                "limit": 20,
                "dryRun": False,
            },
        )
        commits = out.get("commits") or {}
        created = commits.get("created")
        remaining = commits.get("remaining")
        stopped_budget = commits.get("stoppedByBudget")
        print(
            f"ingest round {round_no}: created={created} remaining={remaining} "
            f"stopped={str(stopped_budget).lower()}"
        )
        if out.get("error"):
            stop("ingest reported an error")
        if remaining == 0 and stopped_budget is False:
            finished = True
            break
    if not finished:
        stop("ingest did not finish")

    say("This step saves one failure record, labeled as the failure, tied to the revision where it was seen.")
    found = server.tool("recall", {"handle": f"commit:{FAILURE_SHA}"})
    nodes = found.get("nodes") or []
    if len(nodes) != 1:
        stop(f"failure revision resolved to {len(nodes)} records")
    revision_id = nodes[0]["id"]
    print(f"observed revision {FAILURE_SHA}")
    print(f"revision record {revision_id}")

    saved = server.tool(
        "smart_ingest",
        {
            "content": SYMPTOM,
            "tags": ["failure"],
            "scope": SCOPE,
            "node_type": "fact",
            "forceCreate": True,
            "links": [{"kind": "derived_from", "to": revision_id}],
        },
    )
    failure_id = saved.get("nodeId")
    links = saved.get("links") or []
    if not failure_id or saved.get("linkError") or len(links) != 1:
        stop("the failure record was not linked")
    link = links[0]
    if link.get("edge") != "derived_from" or link.get("target") != revision_id:
        stop("the failure link is not derived_from the observed revision")
    receipt_id = link.get("receiptId")
    if not receipt_id:
        stop("the failure link has no receipt")
    checked = server.tool("recall", {"handle": failure_id})
    checked_nodes = checked.get("nodes") or []
    if len(checked_nodes) != 1:
        stop("the failure record could not be read back")
    stored = checked_nodes[0]
    tags = stored.get("tags") or []
    if stored.get("content") != SYMPTOM or "failure" not in tags:
        stop("the stored failure record does not match the test")
    neighbors = [
        item
        for item in (checked.get("neighbors") or [])
        if item.get("link_type") == "derived_from"
        and item.get("from") == failure_id
        and item.get("to") == revision_id
    ]
    if len(neighbors) != 1:
        stop("the failure record does not have one derived_from link to the observed revision")
    print(f"failure record {failure_id}")
    print("label tag: failure")
    print(SYMPTOM)
    print(f"link: derived_from {FAILURE_SHA}")

    say("This step walks backward from that failure record and ranks the commits it reaches.")
    started = time.perf_counter()
    walk = server.tool(
        "causal_walk",
        {
            "scope": SCOPE,
            "scan_limit": 2000,
            "start_points": [
                {"kind": "ci_run", "run_id": RUN_ID, "node_id": failure_id}
            ],
        },
    )
    elapsed = time.perf_counter() - started
    causes = walk.get("causes") or []
    if not causes:
        stop("the walk returned no commits")
    rank = None
    for index, cause in enumerate(causes, start=1):
        sha = (cause.get("structure") or {}).get("sha") or ""
        edges = path_edges(cause)
        edge_text = ",".join(edges) if edges else "(none of the recorded edge types)"
        print(f"#{index} {sha} edges={edge_text}")
        if sha == CAUSE_SHA:
            rank = index
    print(f"causal_walk seconds: {elapsed:.6f}")
    if rank != 1:
        shown = rank if rank is not None else "absent"
        stop(f"cause {CAUSE_SHA} is rank {shown}")
    print(f"cause {CAUSE_SHA} is rank 1")

    say("This step replays the signed log and rebuilds it, then compares the two digests.")
    report = server.tool("receipt", {"action": "replay", "receipt_id": receipt_id})
    live = report.get("stateDigest") or ""
    replayed = report.get("replayedDigest") or ""
    print(f"live digest: {live}")
    print(f"replayed digest: {replayed}")
    if live and live == replayed and report.get("matched") is True:
        print("MATCH")
    else:
        print("DIGESTS DIFFER")
        sys.exit(1)
finally:
    server.close()
PY

say "This step checks the store with strata-verify and prints the signing-key fingerprint."
"$VESTIGE_BIN" strata-verify "$DATA"

python3 -c 'import os, time; t0=float(open(os.path.join(os.environ["ROOT"], "t0")).read()); print(f"demo seconds: {time.perf_counter()-t0:.6f}")'
