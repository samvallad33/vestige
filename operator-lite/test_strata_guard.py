"""Operator Lite and the Vestige 4.x Strata store, plus two install-safety checks.

1. OP-002 protects the Strata data directory (VESTIGE_DATA_DIR, else the platform default): deleting,
   moving or overwriting log/, its segments, head.state, the key files, store.meta or backups/, and
   deleting or moving the data directory itself, are stopped. Reads and writes elsewhere in it pass.
2. A missing gate file never blocks the agent: the Claude Code hook command exits non-blocking when the
   gate is gone, still blocks (exit 2) when the gate says STOP, and works from a HOME with a space.
3. Install never rewrites a settings.json it cannot parse, and keeps a backup before it rewrites one.

Everything runs in a throwaway HOME made under the real HOME (the gate treats the temp directory as
scratch). No command is executed: the gate only judges them.
Run: /usr/bin/python3 test_strata_guard.py [path/to/operator-gate.py]
"""
import importlib.util, json, os, shutil, subprocess, sys, tempfile

GATE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator-gate.py"))
# The prefix must not contain "strata": earlier gates read that word anywhere in a path as the store.
ROOT = tempfile.mkdtemp(prefix="lite-store-guard-", dir=os.path.expanduser("~"))
fails = []


def check(name, cond, detail=""):
    print("%s  %s%s" % ("ok " if cond else "BAD", name, ("   -> " + str(detail)[:300]) if not cond else ""))
    if not cond:
        fails.append(name)


def env_for(home, **extra):
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "OPERATOR_AGENT_SESSION", "VESTIGE_DATA_DIR", "XDG_DATA_HOME")}
    env.update(HOME=home, OPERATOR_HOME=os.path.join(home, ".operator"), PATH="/usr/bin:/bin")
    env.update(extra)
    return env


def new_home(name, mode="enforce"):
    home = os.path.join(ROOT, name)
    os.makedirs(os.path.join(home, ".operator"))
    os.makedirs(os.path.join(home, "work", "app"))
    with open(os.path.join(home, ".operator", "mode"), "w") as f:
        f.write(mode + "\n")
    return home


def default_data_dir(home):
    """Where Vestige 4.x keeps its store on this platform (the `directories` crate's ProjectDirs)."""
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "com.vestige.core")
    if os.name == "nt":
        return os.path.join(os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming")), "vestige", "core", "data")
    return os.path.join(home, ".local", "share", "core")


def make_store(d):
    os.makedirs(os.path.join(d, "log"))
    os.makedirs(os.path.join(d, "backups"))
    os.makedirs(os.path.join(d, "exports"))
    for rel in ("log/00000000-0000000000000000.seg", "log/head.state", "log/strata.key", "actor.key",
                "receipt-signing.key", "store.meta", "upgrade.log"):
        open(os.path.join(d, rel), "w").close()


def judge(home, payload, cwd, **extra):
    payload = dict(payload, cwd=cwd, session_id="strata-guard", hook_event_name="PreToolUse")
    p = subprocess.run([sys.executable, "-I", GATE, "hook", "--source", "claude"], input=json.dumps(payload),
                       capture_output=True, text=True, env=env_for(home, **extra), cwd=cwd, timeout=60)
    return p.returncode, p.stdout + p.stderr


def bash(cmd):
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


# ---------------------------------------------------------------- 1. OP-002 and the Strata store
home = new_home("store")
D = default_data_dir(home)
make_store(D)
proj = os.path.join(home, "work", "app")
stop_cases = [
    ("rm -rf log, cwd the data dir", bash("rm -rf log"), D),
    ("rm one log segment", bash('rm "%s/log/00000000-0000000000000000.seg"' % D), proj),
    ("truncate head.state", bash(': > "%s/log/head.state"' % D), proj),
    ("rm actor.key", bash('rm "%s/actor.key"' % D), proj),
    ("rm receipt-signing.key", bash('rm "%s/receipt-signing.key"' % D), proj),
    ("rm store.meta", bash('rm "%s/store.meta"' % D), proj),
    ("rm -rf backups", bash('rm -rf "%s/backups"' % D), proj),
    ("mv log aside", bash('mv "%s/log" "%s/log.old"' % (D, D)), proj),
    ("rm -rf the data dir from its parent", bash('rm -rf "%s"' % os.path.basename(D)), os.path.dirname(D)),
    ("mv the data dir into a project", bash('mv "%s" "%s/old-store"' % (D, proj)), proj),
    ("Write tool over head.state", {"tool_name": "Write", "tool_input": {"file_path": D + "/log/head.state", "content": "x"}}, proj),
]
for name, payload, cwd in stop_cases:
    rc, out = judge(home, payload, cwd)
    check("STOP OP-002: " + name, rc == 2 and "OP-002" in out, "rc=%s %s" % (rc, out.strip()[:160]))

pass_cases = [
    ("ls the log", bash('ls -la "%s/log"' % D), proj),
    ("read upgrade.log", bash('cat "%s/upgrade.log"' % D), proj),
    ("write an export", bash('echo hi > "%s/exports/note.txt"' % D), proj),
    ("plain command", bash("git status"), proj),
]
for name, payload, cwd in pass_cases:
    rc, out = judge(home, payload, cwd)
    check("PASS: " + name, rc == 0 and "OP-002" not in out, "rc=%s %s" % (rc, out.strip()[:160]))

# VESTIGE_DATA_DIR moves the store; a folder named after Strata is not a store by its name alone.
custom = os.path.join(home, "my-strata-notes")
make_store(custom)
rc, out = judge(home, bash('rm -rf "%s/log"' % custom), proj, VESTIGE_DATA_DIR=custom)
check("STOP OP-002: rm -rf log of a VESTIGE_DATA_DIR store", rc == 2 and "OP-002" in out, "rc=%s %s" % (rc, out.strip()[:160]))
rc, out = judge(home, bash('echo hi > "%s/exports/x.json"' % custom), proj, VESTIGE_DATA_DIR=custom)
check("PASS: write an export inside a store whose path says strata", rc == 0, "rc=%s %s" % (rc, out.strip()[:160]))
rc, out = judge(home, bash("rm -rf ~/.vestige"), proj)
check("STOP: rm -rf ~/.vestige still stops", rc == 2, "rc=%s %s" % (rc, out.strip()[:160]))

# The data dirs for the other platforms, read without running there.
os.environ["HOME"] = home
spec = importlib.util.spec_from_file_location("gate_under_test", GATE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
fn = getattr(g, "vestige_data_dirs", None)
check("the gate states where Vestige keeps its store", callable(fn))
if callable(fn):
    check("macOS default", fn({}, "/Users/u", "darwin") == ["/Users/u/Library/Application Support/com.vestige.core"],
          fn({}, "/Users/u", "darwin"))
    check("Linux default", fn({}, "/home/u", "linux") == ["/home/u/.local/share/core"], fn({}, "/home/u", "linux"))
    check("Linux XDG_DATA_HOME (absolute only)",
          fn({"XDG_DATA_HOME": "/data/x"}, "/home/u", "linux") == ["/data/x/core"]
          and fn({"XDG_DATA_HOME": "rel/x"}, "/home/u", "linux") == ["/home/u/.local/share/core"],
          (fn({"XDG_DATA_HOME": "/data/x"}, "/home/u", "linux"), fn({"XDG_DATA_HOME": "rel/x"}, "/home/u", "linux")))
    win = fn({"APPDATA": "C:\\Users\\u\\AppData\\Roaming"}, "C:/Users/u", "win32")
    check("Windows default", [p.replace("\\", "/") for p in win] == ["C:/Users/u/AppData/Roaming/vestige/core/data"], win)
    check("VESTIGE_DATA_DIR comes first", fn({"VESTIGE_DATA_DIR": "/srv/v"}, "/home/u", "linux")[0] == "/srv/v",
          fn({"VESTIGE_DATA_DIR": "/srv/v"}, "/home/u", "linux"))

# ---------------------------------------------------------------- 2. a missing gate never blocks


def install(home, *args):
    src = os.path.join(home, "download", "operator-gate.py")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    shutil.copyfile(GATE, src)
    return subprocess.run([sys.executable, "-I", src, "install", "--no-replay"] + list(args), capture_output=True,
                          text=True, env=env_for(home), cwd=home, timeout=120)


def operator_commands(home):
    with open(os.path.join(home, ".claude", "settings.json")) as f:
        s = json.load(f)
    return [h["command"] for grp in s.get("hooks", {}).get("PreToolUse", []) for h in grp.get("hooks", [])
            if "operator-gate" in h.get("command", "")]


def run_hook_command(home, cmd, payload):
    payload = dict(payload, cwd=os.path.join(home, "work", "app"), session_id="s", hook_event_name="PreToolUse")
    p = subprocess.run(["/bin/sh", "-c", cmd], input=json.dumps(payload), capture_output=True, text=True,
                       env=env_for(home), cwd=os.path.join(home, "work", "app"), timeout=60)
    return p.returncode


for label in ("plain", "with space"):
    h = new_home("hook " + label if label == "with space" else "hook-plain")
    r = install(h)
    cmds = operator_commands(h)
    check("install (%s HOME) registers one hook" % label, r.returncode == 0 and len(cmds) == 1, (r.returncode, cmds, r.stdout[-200:]))
    if len(cmds) != 1:
        continue
    cmd = cmds[0]
    check("a benign call passes through the registered command (%s HOME)" % label,
          run_hook_command(h, cmd, bash("ls -la")) == 0, cmd)
    check("a STOP still blocks through it (%s HOME)" % label,
          run_hook_command(h, cmd, bash("rm -rf ~/Documents")) == 2, cmd)
    gate = os.path.join(h, ".operator", "gate", "operator-gate.py")
    os.rename(gate, gate + ".gone")
    rc = run_hook_command(h, cmd, bash("ls -la"))
    check("with the gate file gone the command does not block (%s HOME)" % label, rc not in (2,), "rc=%s %s" % (rc, cmd))
    os.rename(gate + ".gone", gate)

# The Windows form puts the existence check in Python itself. Its logic is the same on every platform, so it
# is run here through sh with this interpreter (the shell's own quoting on Windows is checked by test_windows).
fn = getattr(g, "guarded_python_command", None)
check("the gate builds a shell-independent guarded command for Windows", callable(fn))
if callable(fn):
    h = new_home("winform")
    gate = os.path.join(h, ".operator", "gate", "operator-gate.py")
    os.makedirs(os.path.dirname(gate), exist_ok=True)
    shutil.copyfile(GATE, gate)
    cmd = fn(sys.executable, gate)
    check("Windows form: a benign call passes", run_hook_command(h, cmd, bash("ls -la")) == 0, cmd)
    check("Windows form: a STOP still blocks", run_hook_command(h, cmd, bash("rm -rf ~/Documents")) == 2, cmd)
    os.rename(gate, gate + ".gone")
    rc = run_hook_command(h, cmd, bash("ls -la"))
    check("Windows form: with the gate file gone it does not block", rc == 0, "rc=%s" % rc)
    check("Windows form is recognised as this gate's registration", g.is_claude_gate_hook(cmd), cmd)

# Reinstall over the unguarded command older versions registered: one registration, the new one.
h = new_home("migrate")
os.makedirs(os.path.join(h, ".claude"))
old = "python3 %s hook --source claude" % os.path.join(h, ".operator", "gate", "operator-gate.py")
with open(os.path.join(h, ".claude", "settings.json"), "w") as f:
    json.dump({"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": old}]}]}}, f)
r = install(h)
cmds = operator_commands(h)
check("reinstall over the old command leaves exactly one Operator hook", len(cmds) == 1 and cmds[0] != old, cmds)

# ---------------------------------------------------------------- 3. install and settings.json
h = new_home("badjson")
os.makedirs(os.path.join(h, ".claude"))
bad = ('{\n  "model": "opus",\n  "permissions": {"allow": ["Bash(git status)"]},\n'
       '  "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "vestige hook session-start"}]}]},\n}\n')
sp = os.path.join(h, ".claude", "settings.json")
with open(sp, "w") as f:
    f.write(bad)
r = install(h)
with open(sp) as f:
    after = f.read()
check("an unparseable settings.json is left byte for byte", after == bad, after[:200])
check("install says it could not register and prints the hook to add", r.returncode != 0 and "PreToolUse" in r.stdout,
      (r.returncode, r.stdout[-300:]))

h = new_home("goodjson")
os.makedirs(os.path.join(h, ".claude"))
good = {"model": "opus", "hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "vestige hook session-start"}]}]}}
sp = os.path.join(h, ".claude", "settings.json")
with open(sp, "w") as f:
    json.dump(good, f)
with open(sp) as f:
    good_bytes = f.read()
r = install(h)
with open(sp) as f:
    merged = json.load(f)
check("a valid settings.json keeps its keys and hooks", merged.get("model") == "opus"
      and merged["hooks"].get("SessionStart") == good["hooks"]["SessionStart"] and len(operator_commands(h)) == 1, merged)
baks = [n for n in os.listdir(os.path.join(h, ".claude")) if n.startswith("settings.json.bak")]
check("a backup of the settings it rewrote is kept", len(baks) == 1
      and open(os.path.join(h, ".claude", baks[0])).read() == good_bytes, baks)

print("\nfailures: %d" % len(fails))
sys.exit(1 if fails else 0)
