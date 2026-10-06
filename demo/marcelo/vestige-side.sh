#!/usr/bin/env bash
# Vestige is a cognitive, deterministic memory-transaction security OS for AI agents.
# Live half of the recording. No network. History stops at the parent of b52d489.
set -euo pipefail

die() {
  printf 'stopped: %s\n' "$1" >&2
  exit 1
}

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

DEMO_HOME="${DEMO_HOME:-$HOME/vestige-demo}"
[[ -d "$DEMO_HOME" ]] || die "run demo/marcelo/setup.sh first"
DEMO_HOME="$(cd "$DEMO_HOME" && pwd)"
[[ -f "$DEMO_HOME/.vestige-demo" ]] || die "run demo/marcelo/setup.sh first"

VESTIGE="$DEMO_HOME/vestige"
UV="$DEMO_HOME/uv"
DATA="$DEMO_HOME/data"
STDERR_LOG="$DEMO_HOME/mcp-stderr.txt"
VESTIGE_BIN="$VESTIGE/target/debug/vestige"
MCP_BIN="$VESTIGE/target/debug/vestige-mcp"
EXPECTED="b4bcd52b81402722d6c99b96f461278a171d110d"

[[ -x "$VESTIGE_BIN" && -x "$MCP_BIN" ]] || die "run demo/marcelo/setup.sh first"
[[ -d "$UV/.git" ]] || die "run demo/marcelo/setup.sh first"
HEAD_FULL="$(git -C "$VESTIGE" rev-parse HEAD)"
[[ "$HEAD_FULL" == "$EXPECTED" ]] || die "checkout is $HEAD_FULL"

export CARGO_NET_OFFLINE=true
export GIT_NO_LAZY_FETCH=1
export GIT_TERMINAL_PROMPT=0
export PYTHONUNBUFFERED=1

export DEMO_HOME
export DEMO_UV="$UV"
export DEMO_DATA="$DATA"
export DEMO_MCP="$MCP_BIN"
export DEMO_STDERR="$STDERR_LOG"
export DEMO_SCOPE="uv-10186"
export DEMO_FIX_SHA="b52d48973fe9ddb2e78b663ec48a1a68f7e7802d"
export DEMO_FAILURE_SHA="351d602d86c484a39bc537f1eb99866ea2c25fc1"
export DEMO_CAUSE_SHA="d2f58d92991fa08b24596fcc6c6472dc5015d3bc"
export DEMO_LIMIT="20"
export DEMO_SYMPTOM='failure: uv publish raises `error decoding response body` on 0.5.12. 0.5.11 works. CI https://github.com/andrew000/FTL-Extract/actions/runs/12509313775/job/34898612613#step:8:12'

DEMO_REV="$(git -C "$UV" rev-parse "${DEMO_FIX_SHA}^")"
export DEMO_REV
[[ "$DEMO_REV" != "$DEMO_FIX_SHA" ]] || die "history rev resolved to the revert commit"

printf 'This is pre-release code from PR #445 (commit b4bcd52b81), not the v4.1.1 release.\n'
printf 'HEAD %s\n' "$HEAD_FULL"

say "This step creates a fresh empty data directory for this run."
rm -rf "$DATA"
mkdir -p "$DATA"
if [[ -n "$(find "$DATA" -mindepth 1 -print)" ]]; then
  die "data directory was not empty"
fi
printf 'data directory: %s\n' "$DATA"

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
HOME = os.environ["DEMO_HOME"]
PAUSE = os.environ.get("DEMO_PAUSE", "0") == "1"
SCOPE = os.environ["DEMO_SCOPE"]
REV = os.environ["DEMO_REV"]
FIX_SHA = os.environ["DEMO_FIX_SHA"]
FAILURE_SHA = os.environ["DEMO_FAILURE_SHA"]
CAUSE_SHA = os.environ["DEMO_CAUSE_SHA"]
LIMIT = int(os.environ["DEMO_LIMIT"])
SYMPTOM = os.environ["DEMO_SYMPTOM"]
RUN_ID = SYMPTOM.split("CI ", 1)[1]
ALLOWED = ("closed_by", "derived_from", "evidence_of", "touched", "corrects")


def say(text):
    print()
    print(text)
    sys.stdout.flush()
    if PAUSE:
        time.sleep(2)


def stop(text):
    print("stopped: %s" % text, file=sys.stderr)
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
        env["GIT_NO_LAZY_FETCH"] = "1"
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
                stop("%s failed" % method)
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
            stop("%s returned an error" % name)
        body = result.get("structuredContent")
        if body is None:
            content = result.get("content") or []
            if not content or "text" not in content[0]:
                stop("%s returned no body" % name)
            body = json.loads(content[0]["text"])
        if isinstance(body, dict) and body.get("error") and "nodes" not in body:
            # An exact handle that is not in the store is an empty result.
            # Any other error still stops the run.
            if not (name == "recall" and body.get("error") == "handle_required"):
                stop("%s returned an error" % name)
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


def git_page():
    run = subprocess.run(
        [
            "git",
            "-C",
            UV,
            "log",
            "-n",
            str(LIMIT),
            "--no-merges",
            "--format=%H",
            REV,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if run.returncode != 0:
        stop("could not read the uv history page")
    page = [line.strip() for line in run.stdout.splitlines() if line.strip()]
    if len(page) != LIMIT:
        stop("history page length is %d" % len(page))
    return page


def recall_count(server, sha):
    found = server.tool("recall", {"handle": "commit:%s" % sha})
    if found.get("error") == "handle_required":
        return 0
    return len(found.get("nodes") or [])


page = git_page()
if FAILURE_SHA not in page or CAUSE_SHA not in page or FIX_SHA in page:
    stop("the history page does not match the fair cut")

say("This step records uv history through the parent of b52d489, so the later revert stays outside the record.")
server = Server()
try:
    server.handshake()
    finished = False
    seen = None
    created_total = 0
    for round_no in range(1, 9):
        out = server.tool(
            "codebase",
            {
                "action": "ingest_repo",
                "repoPath": UV,
                "codebase": SCOPE,
                "scope": SCOPE,
                "rev": REV,
                "limit": LIMIT,
                "dryRun": False,
            },
        )
        commits = out.get("commits") or {}
        created = commits.get("created")
        remaining = commits.get("remaining")
        stopped_budget = commits.get("stoppedByBudget")
        print(
            "ingest round %d: created=%s remaining=%s stopped=%s"
            % (round_no, created, remaining, str(stopped_budget).lower())
        )
        sys.stdout.flush()
        if out.get("error"):
            stop("ingest reported an error")
        if commits.get("pulledReverts") not in (0, None) or commits.get("pulledNamed") not in (0, None):
            stop("ingest pulled commits from outside the page")
        repo = out.get("repo") or {}
        if repo.get("revertMerges") not in (0, None):
            stop("ingest kept a revert merge")
        if seen is None:
            seen = commits.get("seen")
        elif commits.get("seen") != seen:
            stop("ingest changed the page size between rounds")
        if isinstance(created, int):
            created_total += created
        if remaining == 0 and stopped_budget is False:
            finished = True
            break
    if not finished:
        stop("ingest did not finish")
    if seen != len(page) or created_total != len(page):
        stop("ingested %s commits, page length %d" % (seen, len(page)))

    present = 0
    for sha in page:
        count = recall_count(server, sha)
        if count != 1:
            stop("commit %s resolved to %d records" % (sha, count))
        present += 1
    if recall_count(server, FIX_SHA) != 0:
        stop("the revert commit was ingested")
    print("history rev: %s" % REV)
    print("ingested commits: %d" % present)
    print("%s ingested" % FAILURE_SHA)
    print("%s ingested" % CAUSE_SHA)
    print("%s not ingested" % FIX_SHA)
    sys.stdout.flush()

    with open(os.path.join(HOME, "ingested-commits.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(page) + "\n")
    with open(os.path.join(HOME, "ingested-count.txt"), "w", encoding="utf-8") as handle:
        handle.write("%d\n" % present)
    with open(os.path.join(HOME, "failure.txt"), "w", encoding="utf-8") as handle:
        handle.write(SYMPTOM)
    with open(os.path.join(HOME, "cause-sha.txt"), "w", encoding="utf-8") as handle:
        handle.write(CAUSE_SHA + "\n")

    say("This step saves one failure record, labeled as the failure, tied to the revision where it was seen.")
    found = server.tool("recall", {"handle": "commit:%s" % FAILURE_SHA})
    nodes = found.get("nodes") or []
    if len(nodes) != 1:
        stop("failure revision resolved to %d records" % len(nodes))
    revision_id = nodes[0]["id"]
    print("observed revision %s" % FAILURE_SHA)
    print("revision record %s" % revision_id)

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
    print("failure record %s" % failure_id)
    print("label tag: failure")
    print(SYMPTOM)
    print("link: derived_from %s" % FAILURE_SHA)
    sys.stdout.flush()

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
        if edges:
            edge_text = ",".join(edges)
        else:
            edge_text = "(none of the recorded edge types)"
        print("#%d %s edges=%s" % (index, sha, edge_text))
        if sha == CAUSE_SHA:
            rank = index
        if sha == FIX_SHA:
            stop("the walk reached the revert commit")
    print("causal_walk seconds: %.6f" % elapsed)
    sys.stdout.flush()
    if rank != 1:
        shown = rank if rank is not None else "absent"
        stop("cause %s is rank %s" % (CAUSE_SHA, shown))
    print("cause %s is rank 1" % CAUSE_SHA)
    sys.stdout.flush()

    say("This step replays the signed log and rebuilds it, then compares the two digests.")
    report = server.tool("receipt", {"action": "replay", "receipt_id": receipt_id})
    live = report.get("stateDigest") or ""
    replayed = report.get("replayedDigest") or ""
    print("live digest: %s" % live)
    print("replayed digest: %s" % replayed)
    sys.stdout.flush()
    if live and live == replayed and report.get("matched") is True:
        print("MATCH")
        sys.stdout.flush()
    else:
        print("DIGESTS DIFFER")
        sys.exit(1)
finally:
    server.close()
PY

say "This step checks the store with strata-verify and prints the signing-key fingerprint."
"$VESTIGE_BIN" strata-verify "$DATA"
