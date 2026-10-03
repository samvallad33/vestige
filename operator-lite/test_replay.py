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


CALL = [0]


def line(tool, ti, cwd=REPO, ts="2099-01-01T00:00:00.000Z", call_id=None):
    CALL[0] += 1
    return json.dumps({"type": "assistant", "timestamp": ts, "cwd": cwd,
                       "message": {"role": "assistant", "content": [
                           {"type": "tool_use", "id": call_id or "call-%d" % CALL[0], "name": tool, "input": ti}]}})


rows = [
    line("Bash", {"command": "git push origin main"}, call_id="resumed-1"),
    line("Bash", {"command": "git push origin main"}, call_id="resumed-1"),   # the same call copied by a resumed session: once
    line("Bash", {"command": "git push origin main"}),                      # the same command run again: a new call
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
check("a copied record counts once, a repeated command counts again (8 calls)", r.get("calls") == 8, r.get("calls"))
check("one session", r.get("sessions") == 1, r.get("sessions"))
check("force-push would have been stopped (OP-004)", r.get("stopped") == 1 and r.get("by_stop") == {"OP-004": 1}, r.get("by_stop"))
check("reset --hard flagged in shadow (OP-S01)", r.get("flagged") == 1 and "OP-S01" in r.get("by_shadow", {}), r.get("by_shadow"))
check("undecided: 3 pushes, 1 package, 1 ci-config", r.get("by_own") == {"push": 3, "packages": 1, "ci-config": 1}, r.get("by_own"))
check("undecided calls = 5", r.get("own_calls") == 5, r.get("own_calls"))
check("record older than the window is skipped (no deploy)", "deploy" not in r.get("by_own", {}))
check("no classify errors", r.get("errors") == 0, r.get("errors"))
check("json carries the drafted laws and the incident", [l["class"] for l in r.get("laws", [])] == ["push", "packages", "ci-config"]
      and len(r.get("incidents", [])) == 1 and r["incidents"][0]["rule"] == "OP-004" and r["incidents"][0]["project"] == "app", r.get("laws"))

p = run(["replay", "--all", "--json"])
check("--all reads the old record too (deploy appears)", json.loads(p.stdout).get("by_own", {}).get("deploy") == 1, p.stdout[:200])
p = run(["replay", "--here", "--json"], cwd=OTHER)
check("--here keeps only calls made under the current directory", json.loads(p.stdout).get("calls") == 1, p.stdout[:200])

p = run(["replay"])
out = p.stdout
check("report: headline", "operator-gate replay: the last 30 days on this machine. Nothing was executed." in out, out[:200])
check("report: scoreboard", "8  tool calls your agents made (1 Claude Code session, 1 project)" in out
      and "1  a built-in rule would have stopped" in out and "5  no built-in rule decides: only you can" in out, out[:600])
check("report: the stop is listed with its project and reason", "Would have been stopped (all of them):" in out
      and "app" in out and "OP-004 force push to a shared branch" in out and "OP-004 no-history-destruction" in out, out[:900])
check("report: undecided table", "No built-in rule decides these. They ran:" in out and "pushed to a remote" in out)
check("report: laws drafted from the history, most frequent first", "Your first laws, drafted from this history:" in out
      and out.index('"No push without my permit."') < out.index('"No new package without my review."')
      and "3 times" in out and "1 time\n" in out, out[-500:])
check("report: samples are shown relative to the working directory", "Write .github/workflows/ci.yml" in out, out[-400:])
check("piped report carries no pitch and no install line", "$149" not in out and "operator-gate upgrade" not in out and "vestige-pro-production" not in out
      and "python3" not in out and "\033[" not in out, out[-300:])
check("no gate home was created by a read-only replay", not os.path.exists(os.path.join(home, ".operator")))
check("nothing in the history was executed (no left-pad, no .github written)",
      not os.path.exists(os.path.join(REPO, ".github")) and not os.path.exists(os.path.join(REPO, "node_modules")))

p = run(["replay", "--share"])
check("--share prints counts only, nothing from the history", p.stdout.count("\n") == 3 and "made 8 tool calls" in p.stdout
      and "would have stopped 1, flagged 1, and found 5" in p.stdout and "git" not in p.stdout.replace("github.com", ""), p.stdout)
p = run(["replay", "--days", "x"])
check("bad --days is a usage error", p.returncode == 2 and "usage:" in p.stdout)

# 3. install runs the replay by itself and remembers the count
p = run(["install"])
check("install exits 0", p.returncode == 0, p.stderr[-300:])
check("install ends with the replay report", "operator-gate replay: the last 30 days" in p.stdout and "mode: shadow" in p.stdout
      and p.stdout.rstrip().endswith("operator-gate.py status"), p.stdout[-400:])
sp = os.path.join(home, ".operator", "state", "replay.json")
check("replay summary remembered for the status hint", os.path.exists(sp) and json.load(open(sp)).get("own_calls") == 5)
p = run(["status"])
check("piped status prints no hint", "operator-gate upgrade" not in p.stdout and "Your last replay" not in p.stdout, p.stdout[-200:])
p = run(["install", "--no-replay"])
p2 = run(["upgrade"])
check("upgrade opens with the laws the last replay drafted", p2.stdout.startswith("From your last replay (")
      and '"No push without my permit."' in p2.stdout and "Vestige Operator: the owner's version" in p2.stdout, p2.stdout[:300])
check("install --no-replay skips it", "operator-gate replay:" not in p.stdout)
p = run(["install"], extra_env={"OPERATOR_AGENT_SESSION": "1"})
check("install still refuses inside an agent session", p.returncode == 3)

# 3b. the README's one command: a single downloaded file, nothing beside it, in a home of its own
home2 = tempfile.mkdtemp(prefix="oplite-lone-home-")
shutil.copytree(os.path.join(home, ".claude"), os.path.join(home2, ".claude"))
os.remove(os.path.join(home2, ".claude", "settings.json"))
lone = os.path.join(tempfile.mkdtemp(prefix="oplite-download-"), "operator-gate.py")
shutil.copyfile(GATE, lone)
env2 = dict(env, HOME=home2)
p = subprocess.run([sys.executable, lone, "install"], capture_output=True, text=True, env=env2, cwd=home2)
installed = os.path.join(home2, ".operator", "gate", "operator-gate.py")
check("lone file: install exits 0 and installs a byte-identical copy",
      p.returncode == 0 and os.path.exists(installed) and open(installed, "rb").read() == open(GATE, "rb").read(), p.stderr[-300:])
try:
    wired = json.load(open(os.path.join(home2, ".claude", "settings.json")))["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
except Exception as exc:
    wired = repr(exc)
check("lone file: the Claude Code hook points at the installed copy", wired == "python3 %s hook --source claude" % installed, wired)
check("lone file: install starts in shadow and shows the replay without being asked",
      open(os.path.join(home2, ".operator", "mode")).read().strip() == "shadow"
      and "tool calls your agents made" in p.stdout and "Your first laws, drafted from this history:" in p.stdout, p.stdout[-400:])
payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git push --force origin main"}, "cwd": REPO, "session_id": "t"})
p = subprocess.run(wired.split(), input=payload, capture_output=True, text=True, env=env2)
check("lone file: the wired hook records in shadow and does not block", p.returncode == 0 and not p.stderr, p.stderr[:200])
open(os.path.join(home2, ".operator", "mode"), "w").write("enforce\n")
p = subprocess.run(wired.split(), input=payload, capture_output=True, text=True, env=env2)
check("lone file: after the flip to enforce the wired hook stops the force-push", p.returncode == 2 and "OP-004" in p.stderr, p.stderr[:200])
p = subprocess.run([sys.executable, installed, "verify"], capture_output=True, text=True, env=env2)
check("lone file: both verdicts are in the receipt chain", "receipts=2 chain=OK" in p.stdout, p.stdout[:200])
p = subprocess.run([sys.executable, installed, "corpus", "guardfall"], capture_output=True, text=True, env=env2)
check("lone file: corpus says where the corpus lives instead of failing with a traceback",
      p.returncode == 2 and "ships in the repository" in p.stdout and "Traceback" not in p.stdout + p.stderr, p.stdout[:200])
shutil.rmtree(home2, ignore_errors=True)
shutil.rmtree(os.path.dirname(lone), ignore_errors=True)

# 3c. the log: after a day of use, status says what the gate saw on this machine
home3 = tempfile.mkdtemp(prefix="oplite-log-home-")
os.makedirs(os.path.join(home3, ".claude"))
bin3 = os.path.join(home3, ".local", "bin")
os.makedirs(bin3)
env3 = dict(env, HOME=home3, PATH=bin3 + os.pathsep + env.get("PATH", ""))
p = subprocess.run([sys.executable, GATE, "install", "--no-replay"], capture_output=True, text=True, env=env3, cwd=home3)
gate3 = os.path.join(home3, ".operator", "gate", "operator-gate.py")
launcher = os.path.join(bin3, "operator-gate")
check("install places a launcher when a user bin directory is on PATH", p.returncode == 0 and os.path.exists(launcher)
      and "command: operator-gate" in p.stdout, p.stdout[-300:])
p = subprocess.run(["operator-gate", "status"], capture_output=True, text=True, env=env3, cwd=home3)
check("the launcher runs the installed gate", p.returncode == 0 and p.stdout.startswith("Operator Lite "), (p.stdout + p.stderr)[:200])
check("status before any call says so and names the hook", "Claude Code hook: registered" in p.stdout
      and "No tool calls checked in the last 7 days" in p.stdout, p.stdout[:400])
proj3 = os.path.join(home3, "work", "shop-api")
os.makedirs(proj3)


def call3(cmd):
    pl = json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": proj3, "session_id": "d"})
    return subprocess.run([sys.executable, gate3, "hook", "--source", "claude"], input=pl, capture_output=True, text=True, env=env3)


for c in ("git status", "git push --force origin main", "psql -c 'DROP TABLE users' app", "git reset --hard HEAD~1", "ls"):
    call3(c)
recs3 = []
rdir3 = os.path.join(home3, ".operator", "receipts")
for fn in sorted(x for x in os.listdir(rdir3) if x.endswith(".jsonl")):
    with open(os.path.join(rdir3, fn), encoding="utf-8") as f:
        recs3 += [json.loads(line) for line in f]
check("an allowed call is recorded as ALLOW and a rule hit as SHADOW_STOP",
      [r["decision"] for r in recs3] == ["ALLOW", "SHADOW_STOP", "SHADOW_STOP", "SHADOW_STOP", "ALLOW"], [r["decision"] for r in recs3])
p = subprocess.run(["operator-gate", "status"], capture_output=True, text=True, env=env3, cwd=home3)
st = p.stdout
check("status counts the week: 5 checked, 2 would have been stopped and ran, 1 flagged", "5  tool calls checked" in st
      and "2  would have been stopped, and ran: the gate was in shadow mode" in st and "1  flagged: recorded, not stopped" in st, st[:700])
check("status lists what ran, with the project, the rule and the command", "Ran, because the gate was in shadow mode:" in st
      and "shop-api" in st and "OP-004" in st and "git push --force origin main" in st and "OP-007" in st and "DROP TABLE users" in st, st)
check("status says how to start blocking, with a command that works", "operator-gate mode enforce" in st and "Last call checked just now." in st, st[-400:])
check("piped status carries no pitch and no colour", "upgrade" not in st and "\033[" not in st, st[-300:])
p = subprocess.run(["operator-gate", "status", "--rules"], capture_output=True, text=True, env=env3, cwd=home3)
check("status --rules is the rule table, in words a stranger can read", "OP-004" in p.stdout and "no-history-destruction" in p.stdout
      and not any(w in p.stdout for w in ("launch gates", "claim-gate", "Pro bridge", "postmark")), p.stdout[:300])
p = subprocess.run(["operator-gate", "mode", "enforce"], capture_output=True, text=True, env=env3, cwd=home3)
check("mode refuses outside an interactive terminal", p.returncode == 3 and open(os.path.join(home3, ".operator", "mode")).read().strip() == "shadow", p.stdout)
p = subprocess.run(["operator-gate", "mode"], capture_output=True, text=True, env=env3, cwd=home3)
check("mode with no argument reports the mode", p.returncode == 0 and "mode is shadow" in p.stdout, p.stdout)
open(os.path.join(home3, ".operator", "mode"), "w").write("enforce\n")
r3 = call3("git push --force origin main")
check("the stop message names a command the owner can paste", r3.returncode == 2 and "operator-gate approve " in r3.stderr, r3.stderr[-300:])
p = subprocess.run(["operator-gate", "status"], capture_output=True, text=True, env=env3, cwd=home3)
check("in enforce mode status shows the stop under Stopped", "mode: enforce (blocking)" in p.stdout and "1  stopped" in p.stdout
      and "Stopped:" in p.stdout and "operator-gate mode enforce" not in p.stdout, p.stdout[:900])
p = subprocess.run([sys.executable, gate3, "uninstall"], capture_output=True, text=True, env=env3, cwd=home3)
check("uninstall removes the hook and the launcher it placed", not os.path.exists(launcher) and "launcher removed" in p.stdout, p.stdout)
p = subprocess.run([sys.executable, gate3, "status"], capture_output=True, text=True, env=dict(env3, PATH=env.get("PATH", "")), cwd=home3)
check("without the hook status says it is not registered, and without the launcher it prints the full command",
      "Claude Code hook: NOT registered" in p.stdout and "python3 ~/.operator/gate/operator-gate.py install" in p.stdout, p.stdout[:400])
shutil.rmtree(home3, ignore_errors=True)

# 4. the hint a person sees, run under a pseudo-terminal
import pty
def tty_run(args, color=False):
    chunks = []
    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update(env)
        os.environ.pop("OPERATOR_HOME", None)
        os.environ.pop("OPERATOR_AGENT_SESSION", None)
        os.environ.pop("NO_COLOR", None)
        if not color:
            os.environ["NO_COLOR"] = "1"
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
check("at a terminal, status ends with the remembered count", "Your last replay drafted 3 laws from 5 actions no built-in rule decides." in t, t[-300:])
t = tty_run(["replay"])
check("at a terminal, replay ends with the pointer and the price", "Operator enforces those laws" in t and "$149 a month." in t
      and "https://vestige-pro-production.fly.dev/account" in t, t[-400:])
check("installed gate: replay does not tell the owner to install again", " install" not in t.split("Operator enforces")[-1], t[-300:])
t = tty_run(["mode", "enforce"])
check("at a terminal the owner can switch to enforce", "The gate now blocks what its rules stop." in t
      and open(os.path.join(home, ".operator", "mode")).read().strip() == "enforce", t[-200:])
t = tty_run(["mode", "shadow"])
check("and back to shadow", open(os.path.join(home, ".operator", "mode")).read().strip() == "shadow", t[-200:])
# upgrade --install: the archive a buyer downloaded, unpacked, and its wizard started
import io, tarfile
arch_dir = tempfile.mkdtemp(prefix="oplite-archive-")
good = os.path.join(arch_dir, "vestige-operator-test.tar.gz")
with tarfile.open(good, "w:gz") as tar:
    body = b"import sys\nprint('WIZARD STARTED', sys.argv[1:])\n"
    info = tarfile.TarInfo("vestige-operator/gate/operator-gate.py")
    info.size = len(body)
    tar.addfile(info, io.BytesIO(body))
bad = os.path.join(arch_dir, "escape.tar.gz")
with tarfile.open(bad, "w:gz") as tar:
    info = tarfile.TarInfo("vestige-operator/../outside.txt")
    info.size = 1
    tar.addfile(info, io.BytesIO(b"x"))
p = run(["upgrade", "--install", good])
check("upgrade --install refuses outside an interactive terminal", p.returncode == 3 and not os.path.exists(os.path.join(home, "vestige-operator")), p.stdout)
t = tty_run(["upgrade", "--install", bad])
check("an archive with an entry outside vestige-operator is refused", "does not belong in vestige-operator" in t
      and not os.path.exists(os.path.join(home, "outside.txt")) and not os.path.exists(os.path.join(home, "vestige-operator")), t[-300:])
t = tty_run(["upgrade", "--install", good])
check("at a terminal it prints the checksum, unpacks and starts the wizard", "SHA-256: " in t and "WIZARD STARTED ['onboard']" in t
      and os.path.exists(os.path.join(home, "vestige-operator", "gate", "operator-gate.py")), t[-400:])
payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "python3 ~/.operator/gate/operator-gate.py upgrade --install x.tar.gz"}, "cwd": REPO, "session_id": "t"})
open(os.path.join(home, ".operator", "mode"), "w").write("enforce\n")
p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload, capture_output=True, text=True, env=env)
check("an agent running upgrade --install is stopped as an owner command", p.returncode == 2 and "OP-000" in p.stderr, p.stderr[:200])
open(os.path.join(home, ".operator", "mode"), "w").write("shadow\n")
shutil.rmtree(arch_dir, ignore_errors=True)

t = tty_run(["replay"], color=True)
check("at a terminal the numbers are coloured; NO_COLOR turns it off", "\033[1;31m" in t and "\033[0m" in t, t[:300])

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
