#!/usr/bin/env python3
"""Install never downgrades: Operator Lite refuses to replace the paid Operator gate or a newer gate,
and `--force` replaces it only after keeping a copy. An agent's shell cannot install or uninstall.
The npm launcher says which gate and version it ran.

Every case runs in a throwaway HOME with a fake installed gate; nothing here touches ~/.operator.
Usage: python3 operator-lite/test_install_guard.py [repo-root]"""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile

W = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(W, "operator-lite", "operator-gate.py")
LAUNCHER = os.path.join(W, "operator-lite", "npm", "bin", "operator-lite.js")
AGENT_VARS = ("OPERATOR_AGENT_SESSION", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
with open(GATE, encoding="utf-8") as f:
    V = re.search(r'^VERSION = "([^"]+)"', f.read(), re.M).group(1)
LITE = "Operator Lite " + V
a, b, c = (int(x) for x in V.split("."))
NUMERIC_NEWER = "%d.%d.%d" % (a, b, 10 ** len(str(c)))    # newer by number, older by string: 0.3.10 vs 0.3.7
fails = []
root = tempfile.mkdtemp(prefix="oplite-guard-")

FAKE_PAID = ('#!/usr/bin/env python3\n"""A stand-in for the paid gate: its markers and nothing else."""\nimport sys\n'
             'VERSION = "0.3.13"\n\n\ndef load_law():\n    return None\n\n\ndef cmd_commitments(argv):\n    return 0\n\n\n'
             'if __name__ == "__main__":\n    print("FAKE PAID GATE ran: " + " ".join(sys.argv[1:]))\n')
LITE_SAYS = 'print("Operator Lite %s   mode: %s")\n'


def check(name, cond, detail=""):
    print("%s  %s%s" % ("ok " if cond else "BAD", name, ("   -> " + str(detail)[-600:]) if not cond else ""))
    if not cond:
        fails.append(name)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fake_home(name, installed=None, siblings=()):
    """A HOME with no agent hosts and, when given, a gate already installed at ~/.operator/gate."""
    home = os.path.join(root, name)
    os.makedirs(os.path.join(home, ".operator", "gate"))
    if installed is not None:
        with open(os.path.join(home, ".operator", "gate", "operator-gate.py"), "w") as f:
            f.write(installed)
    for s in siblings:
        with open(os.path.join(home, ".operator", "gate", s), "w") as f:
            f.write("# feature module\n")
    return home


def env_for(home, **extra):
    env = dict(os.environ, HOME=home, USERPROFILE=home, OPERATOR_HOME=os.path.join(home, ".operator"))
    for k in AGENT_VARS + ("OPERATOR_GATE_MODE", "XDG_CONFIG_HOME"):
        env.pop(k, None)
    env.update(extra)
    return env


def install(home, *args, **extra):
    p = subprocess.run([sys.executable, GATE, "install", "--no-replay"] + list(args), capture_output=True,
                       text=True, env=env_for(home, **extra), cwd=home)
    return p.returncode, p.stdout + p.stderr


def installed(home):
    return os.path.join(home, ".operator", "gate", "operator-gate.py")


# ---- must stop: the installer refuses and the installed gate is untouched -------------------------
refusals = [
    ("the paid gate (load_law, cmd_commitments)", "paid", FAKE_PAID, (), "Operator 0.3.13"),
    ("the paid gate known only by its feature modules", "paid-siblings", 'VERSION = "0.3.13"\n', ("board.py", "letter.py"),
     "Operator 0.3.13"),
    ("the paid gate at a LOWER version (edition outranks version)", "paid-old", 'VERSION = "0.3.1"\nEDITION = "operator"\n',
     (), "Operator 0.3.1"),
    ("a newer Operator Lite", "lite-newer", 'VERSION = "0.3.99"\n' + LITE_SAYS, (), "Operator Lite 0.3.99"),
    ("a newer Lite by number, older by string (%s > %s)" % (NUMERIC_NEWER, V), "lite-numeric",
     'VERSION = "%s"\n' % NUMERIC_NEWER + LITE_SAYS, (), "Operator Lite " + NUMERIC_NEWER),
    ("a gate whose version cannot be read", "no-version", "# a gate with no VERSION line\n", (), "version unreadable"),
]
for label, name, body, sib, shown in refusals:
    home = fake_home(name, body, sib)
    before = sha(installed(home))
    rc, out = install(home)
    check("refuses over %s: exit 4" % label, rc == 4, (rc, out))
    check("refuses over %s: installed gate unchanged" % label, sha(installed(home)) == before)
    check("refuses over %s: prints both versions and the path" % label,
          shown in out and LITE in out and installed(home) in out, out)
    check("refuses over %s: no hook registered" % label,
          not os.path.exists(os.path.join(home, ".claude", "settings.json")))

for var, val in (("CLAUDECODE", "1"), ("CLAUDE_CODE_ENTRYPOINT", "cli"), ("OPERATOR_AGENT_SESSION", "1")):
    home = fake_home("agent-" + var.lower())
    rc, out = install(home, **{var: val})
    check("install refuses in an agent shell (%s): exit 3, nothing installed" % var,
          rc == 3 and var in out and not os.path.exists(installed(home)), (rc, out))
    p = subprocess.run([sys.executable, GATE, "uninstall"], capture_output=True, text=True,
                       env=env_for(home, **{var: val}), cwd=home)
    check("uninstall refuses in an agent shell (%s): exit 3" % var, p.returncode == 3, p.stdout + p.stderr)

# ---- must pass: a fresh install, an upgrade, a reinstall, and an owner's --force -------------------
home = fake_home("fresh")
rc, out = install(home)
check("fresh install: exit 0", rc == 0, out)
check("fresh install: the installed gate is this file byte for byte", sha(installed(home)) == sha(GATE))
try:
    with open(os.path.join(home, ".claude", "settings.json")) as f:
        hooks = json.dumps(json.load(f))
except Exception as exc:
    hooks = str(exc)
check("fresh install: Claude Code hook registered", "operator-gate.py hook --source claude" in hooks, hooks)

for label, name, body in (("an older Operator Lite (0.3.6)", "lite-older", 'VERSION = "0.3.6"\n' + LITE_SAYS),
                          ("an older unmarked gate (0.3.4)", "unmarked-older", 'VERSION = "0.3.4"\n')):
    home = fake_home(name, body)
    rc, out = install(home)
    check("upgrades %s: exit 0" % label, rc == 0, out)
    check("upgrades %s: installed gate is now this one" % label, sha(installed(home)) == sha(GATE))
    check("upgrades %s: says what it replaced" % label, "(replaced " in out and LITE in out, out)

home = fake_home("same")
shutil.copyfile(GATE, installed(home))
rc, out = install(home)
check("reinstalling the same version: exit 0", rc == 0 and sha(installed(home)) == sha(GATE), out)

home = fake_home("forced", FAKE_PAID)
paid_sha = sha(installed(home))
rc, out = install(home, "--force")
backups = [os.path.join(home, ".operator", "install-backups", n)
           for n in sorted(os.listdir(os.path.join(home, ".operator", "install-backups")))] \
    if os.path.isdir(os.path.join(home, ".operator", "install-backups")) else []
check("--force over the paid gate: exit 0 and a WARNING", rc == 0 and "WARNING" in out, out)
check("--force over the paid gate: the paid gate is kept byte for byte", any(sha(b) == paid_sha for b in backups), backups)
check("--force over the paid gate: prints how to put it back", "to put it back: cp" in out, out)

# ---- the npm launcher: refuses through install, and names the gate it runs ---------------------------
node = shutil.which("node")
if not node:
    print("skip  launcher cases: node is not on PATH")
else:
    pkg = os.path.join(root, "pkg")
    os.makedirs(os.path.join(pkg, "bin"))
    os.makedirs(os.path.join(pkg, "gate"))
    shutil.copyfile(LAUNCHER, os.path.join(pkg, "bin", "operator-lite.js"))
    shutil.copyfile(GATE, os.path.join(pkg, "gate", "operator-gate.py"))
    launcher = os.path.join(pkg, "bin", "operator-lite.js")

    home = fake_home("npx-paid", FAKE_PAID)
    before = sha(installed(home))
    p = subprocess.run([node, launcher, "--no-replay"], capture_output=True, text=True, env=env_for(home), cwd=home)
    out = p.stdout + p.stderr
    check("npx operator-lite over the paid gate: exit 4, gate unchanged, no host wired",
          p.returncode == 4 and sha(installed(home)) == before and "Gated:" not in out, (p.returncode, out))

    p = subprocess.run([node, launcher, "status"], capture_output=True, text=True, env=env_for(home), cwd=home)
    check("npx operator-lite status names the installed paid gate on stderr",
          "running the installed gate, Operator 0.3.13 (%s)" % installed(home) in p.stderr
          and "FAKE PAID GATE ran: status" in p.stdout, p.stdout + p.stderr)

    home = fake_home("npx-none")
    p = subprocess.run([node, launcher, "status"], capture_output=True, text=True, env=env_for(home), cwd=home)
    check("npx operator-lite status before any install names the packaged Lite gate",
          "running the gate in this package, " + LITE in p.stderr, p.stdout + p.stderr)

    home = fake_home("npx-fresh")
    p = subprocess.run([node, launcher, "--no-replay"], capture_output=True, text=True, env=env_for(home), cwd=home)
    check("npx operator-lite on a fresh machine installs: exit 0", p.returncode == 0
          and sha(installed(home)) == sha(GATE) and "Gated: Claude Code" in p.stdout, p.stdout + p.stderr)

shutil.rmtree(root, ignore_errors=True)
print("\nfailures: %d" % len(fails))
sys.exit(1 if fails else 0)
