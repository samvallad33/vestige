#!/usr/bin/env python3
"""Tests for Operator Lite `replay`. Classification only: no sample command is ever executed.
Usage: python3 operator-lite/test_replay.py [repo-root]"""
import importlib.util, json, os, shutil, subprocess, sys, tempfile

W = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(W, "operator-lite", "operator-gate.py")
fails = []
REAL_ENV = dict(os.environ)
REAL_ENV.pop("OPERATOR_AGENT_SESSION", None)


def check(name, cond, detail=""):
    print("%s  %s%s" % ("ok " if cond else "BAD", name, ("   -> " + str(detail)) if (detail and not cond) else ""))
    if not cond:
        fails.append(name)


# a fake home outside nothing real: history, gate home, all throwaway
home = tempfile.mkdtemp(prefix="oplite-replay-home-")
env = dict(os.environ, HOME=home)
env.pop("OPERATOR_HOME", None)
env.pop("OPERATOR_AGENT_SESSION", None)

# 1. classes, read through classify() exactly as replay does
os.environ["HOME"] = home
os.environ.pop("OPERATOR_HOME", None)
spec = importlib.util.spec_from_file_location("gate", GATE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
cfg = g.load_config()
REPO = os.path.join(home, "work", "app")
os.makedirs(REPO)


def classes(tool, ti):
    _, _, hits, _, _, effects = g.classify({"tool_name": tool, "tool_input": ti, "cwd": REPO}, cfg)
    return sorted(set(h[0] for h in hits)), sorted(g.own_law_classes(effects))


CASES = [
    ("git push origin main", ["push"]),
    ("git push --dry-run origin main", []),
    ("git commit --no-verify -m wip", ["no-verify"]),
    ("git commit -an -m wip", ["no-verify"]),
    ("git commit -m 'fix -n handling'", []),
    ("git status", []),
    ("npm install left-pad", ["packages"]),
    ("npm install", []),
    ("npm ci", []),
    ("pip install -r requirements.txt", []),
    ("pip install requests", ["packages"]),
    ("cargo add serde", ["packages"]),
    ("cargo install --path .", []),
    ("npx prisma migrate deploy", ["database"]),
    ("psql -c 'select 1' mydb", ["database"]),
    ("vercel deploy --prod", ["deploy"]),
    ("vercel --prod", ["deploy"]),
    ("terraform apply -auto-approve", ["deploy"]),
    ("terraform plan", []),
    ("kubectl get pods", []),
    ("kubectl apply -f k8s/", ["deploy"]),
    ("echo KEY=1 > .env", ["env-file"]),
    ("cp .env.example .env.local", ["env-file"]),
    ("echo x > .env.example", []),
    ("ls -la && cargo check", []),
    ("cd sub && git push", ["push"]),
]
for cmd, want in CASES:
    rids, got = classes("Bash", {"command": cmd})
    check("classes %-42s %s" % (cmd, want), got == want, "got %s hits %s" % (got, rids))
for path, want in [(".github/workflows/ci.yml", ["ci-config"]), ("Dockerfile", ["ci-config"]),
                   (".env", ["env-file"]), ("src/main.rs", [])]:
    rids, got = classes("Write", {"file_path": os.path.join(REPO, path), "content": "x"})
    check("classes Write %-36s %s" % (path, want), got == want, "got %s hits %s" % (got, rids))

# 2. a fake Claude Code transcript, replayed end to end through the CLI
proj = os.path.join(home, ".claude", "projects", "-work-app")
os.makedirs(proj)
OTHER = os.path.join(home, "work", "other")
os.makedirs(OTHER)


def line(tool, ti, cwd=REPO, ts="2099-01-01T00:00:00.000Z"):
    return json.dumps({"type": "assistant", "timestamp": ts, "cwd": cwd,
                       "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": tool, "input": ti}]}})


rows = [
    line("Bash", {"command": "git push origin main"}),
    line("Bash", {"command": "git push origin main"}),                      # duplicate: counted once
    line("Bash", {"command": "git push origin main"}, cwd=OTHER),           # other cwd: its own call
    line("Bash", {"command": "git push --force origin main"}),              # built-in stop
    line("Bash", {"command": "git reset --hard HEAD~1"}),                   # shadow flag
    line("Bash", {"command": "npm install left-pad"}),
    line("Bash", {"command": "ls -la"}),
    line("Write", {"file_path": os.path.join(REPO, ".github/workflows/ci.yml"), "content": "on: push"}),
    line("Read", {"file_path": os.path.join(REPO, "README.md")}),            # not a replayed tool
    line("Bash", {"command": "terraform apply -auto-approve"}, ts="2001-01-01T00:00:00.000Z"),  # older than the window
    "this line is not json but mentions \"tool_use\"",
    json.dumps(["tool_use", "a list, not a record"]),
]
with open(os.path.join(proj, "s1.jsonl"), "w") as f:
    f.write("\n".join(rows) + "\n")


def run(args, cwd=None, extra_env=None):
    e = dict(env, **(extra_env or {}))
    return subprocess.run([sys.executable, GATE] + args, capture_output=True, text=True, env=e, cwd=cwd or home)


p = run(["replay", "--json"])
check("replay --json exits 0", p.returncode == 0, p.stderr[-300:])
r = json.loads(p.stdout or "{}")
check("calls counted once per unique call (7)", r.get("calls") == 7, r.get("calls"))
check("one session", r.get("sessions") == 1, r.get("sessions"))
check("force-push would have been stopped (OP-004)", r.get("stopped") == 1 and r.get("by_stop") == {"OP-004": 1}, r.get("by_stop"))
check("reset --hard flagged in shadow (OP-S01)", r.get("flagged") == 1 and "OP-S01" in r.get("by_shadow", {}), r.get("by_shadow"))
check("undecided: 2 pushes, 1 package, 1 ci-config", r.get("by_own") == {"push": 2, "packages": 1, "ci-config": 1}, r.get("by_own"))
check("undecided calls = 4", r.get("own_calls") == 4, r.get("own_calls"))
check("record older than the window is skipped (no deploy)", "deploy" not in r.get("by_own", {}))
check("no classify errors", r.get("errors") == 0, r.get("errors"))

p = run(["replay", "--all", "--json"])
check("--all reads the old record too (deploy appears)", json.loads(p.stdout).get("by_own", {}).get("deploy") == 1, p.stdout[:200])
p = run(["replay", "--here", "--json"], cwd=OTHER)
check("--here keeps only calls made under the current directory", json.loads(p.stdout).get("calls") == 1, p.stdout[:200])

p = run(["replay"])
out = p.stdout
check("report: headline", "7 tool calls from 1 Claude Code session, last 30 days. Nothing was executed." in out, out[:200])
check("report: stop table", "Built-in rules would have stopped 1:" in out and "OP-004 no-history-destruction" in out)
check("report: undecided table", "No built-in rule decides these. They ran (4):" in out and "pushed to a remote" in out)
check("report: samples are shown relative to the working directory", "Write .github/workflows/ci.yml" in out, out[-400:])
check("piped report carries no pitch and no install line", "$149" not in out and "operator-gate upgrade" not in out and "python3" not in out, out[-300:])
check("no gate home was created by a read-only replay", not os.path.exists(os.path.join(home, ".operator")))
check("nothing in the history was executed (no left-pad, no .github written)",
      not os.path.exists(os.path.join(REPO, ".github")) and not os.path.exists(os.path.join(REPO, "node_modules")))

p = run(["replay", "--days", "x"])
check("bad --days is a usage error", p.returncode == 2 and "usage:" in p.stdout)

# 3. install runs the replay by itself and remembers the count
p = run(["install"])
check("install exits 0", p.returncode == 0, p.stderr[-300:])
check("install ends with the replay report", "operator-gate replay: 7 tool calls" in p.stdout and "mode=shadow" in p.stdout, p.stdout[-400:])
sp = os.path.join(home, ".operator", "state", "replay.json")
check("replay summary remembered for the status hint", os.path.exists(sp) and json.load(open(sp)).get("own_calls") == 4)
p = run(["status"])
check("piped status prints no hint", "operator-gate upgrade" not in p.stdout and "Your last replay" not in p.stdout, p.stdout[-200:])
p = run(["install", "--no-replay"])
check("install --no-replay skips it", "operator-gate replay:" not in p.stdout)
p = run(["install"], extra_env={"OPERATOR_AGENT_SESSION": "1"})
check("install still refuses inside an agent session", p.returncode == 3)

# 4. the hint a person sees, run under a pseudo-terminal
import pty
def tty_run(args):
    chunks = []
    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update(env)
        os.environ.pop("OPERATOR_HOME", None)
        os.environ.pop("OPERATOR_AGENT_SESSION", None)
        os.chdir(home)
        os.execv(sys.executable, [sys.executable, GATE] + args)
    while True:
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    os.waitpid(pid, 0)
    return b"".join(chunks).decode(errors="replace")
t = tty_run(["status"])
check("at a terminal, status ends with the remembered count", "Your last replay found 4 actions no built-in rule decides." in t, t[-300:])
t = tty_run(["replay"])
check("at a terminal, replay ends with the pointer and the price", "can be a law in Operator" in t and "$149 a month: operator-gate upgrade" in t, t[-400:])
check("installed gate: replay does not tell the owner to install again", "python3" not in t.split("can be a law")[-1], t[-300:])

# 5. verdicts unchanged: the hook, the corpus, the three copies
payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git push --force origin main"}, "cwd": REPO, "session_id": "t"})
open(os.path.join(home, ".operator", "mode"), "w").write("enforce\n")
p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload, capture_output=True, text=True, env=env)
lines = p.stderr.strip().splitlines()
check("hook still stops a force-push with exit 2", p.returncode == 2 and lines and lines[0].startswith("OPERATOR: STOPPED (OP-004"), p.stderr[:200])
check("stop message carries no pitch", not any(w in p.stderr for w in ("upgrade", "$149", "replay", "Operator adds")), p.stderr)
payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git push origin main"}, "cwd": REPO, "session_id": "t"})
p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload, capture_output=True, text=True, env=env)
check("hook still allows a plain push silently", p.returncode == 0 and not p.stdout and not p.stderr, (p.stdout + p.stderr)[:200])
# the corpus needs a home outside scratch (a home under the temp dir makes deletes of ~ look like scratch)
p = subprocess.run([sys.executable, GATE, "corpus", "guardfall"], capture_output=True, text=True,
                   env=dict(REAL_ENV, OPERATOR_HOME=tempfile.mkdtemp(prefix="oplite-corpus-")))
check("corpus 46/46", "cases=46 passed=46 failed=0" in p.stdout, p.stdout[-300:])
body = open(GATE, "rb").read()
for c in ("operator-lite/openclaw-plugin/gate/operator-gate.py", "operator-lite/ports/opencode/operator-gate.py"):
    check("copy identical: " + c, open(os.path.join(W, c), "rb").read() == body)

shutil.rmtree(home, ignore_errors=True)
print("\nfailures: %d" % len(fails))
sys.exit(1 if fails else 0)
