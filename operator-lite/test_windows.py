#!/usr/bin/env python3
"""Windows port tests for Operator Lite.

Part A runs everywhere: the path spelling function is pure string work.
Part B runs only on Windows: the gate as a hook, in enforce mode, against Git Bash style commands.
Classification only: no sample command is ever executed.
Usage: python operator-lite/test_windows.py [repo-root]"""
import importlib.util, json, os, shutil, subprocess, sys, tempfile

W = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(W, "operator-lite", "operator-gate.py")
fails = []


def check(name, cond, detail=""):
    print("%s  %s%s" % ("ok " if cond else "BAD", name, ("   -> " + str(detail)) if (detail != "" and not cond) else ""))
    if not cond:
        fails.append(name)


os.environ["OPERATOR_HOME"] = tempfile.mkdtemp(prefix="oplite-win-import-")
spec = importlib.util.spec_from_file_location("gate", GATE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
check("the gate imports on this platform (%s)" % os.name, True)

# Part A: one spelling for a Windows path
for given, want in [("C:\\Users\\me\\proj", "C:/Users/me/proj"), ("c:/Users/me", "C:/Users/me"),
                    ("/c/Users/me", "C:/Users/me"), ("/c", "C:/"), ("/cygdrive/d/work/x", "D:/work/x"),
                    ("/tmp/x", "/tmp/x"), ("/usr/bin/git", "/usr/bin/git"), ("src\\main.rs", "src/main.rs"),
                    ("C:\\Users\\ALPHIN~1", "C:/Users/ALPHIN~1")]:
    got = g.canon_windows(given)
    check("spelling %-24s -> %s" % (given, want), got == want, got)
# Part A, continued: the PowerShell tool, judged through classify() on any platform
cfg = g.load_config()
cwd_ps = os.path.join(os.path.expanduser("~"), "oplite-ps-project")


def ps(cmd):
    _, _, hits, _, _, effects = g.classify({"tool_name": "PowerShell", "tool_input": {"command": cmd}, "cwd": cwd_ps}, cfg)
    return sorted(set(h[0] for h in hits))


for cmd, want, why in [
        ("Remove-Item -Recurse -Force ~/Documents", "OP-003", "recursive delete of a home folder"),
        ("Remove-Item -Recurse -Force $HOME", "OP-001", "recursive delete of the home directory"),
        ("rm -r -fo $env:USERPROFILE", "OP-001", "the short aliases and parameter prefixes"),
        ("Remove-Item -Path $env:USERPROFILE/Documents, ~/Desktop -Recurse", "OP-003", "-Path with a comma list"),
        ("Get-ChildItem . ; Remove-Item -Recurse -Force ~/Documents", "OP-003", "a delete after another statement"),
        ("git push --force origin main", "OP-004", "git reads the same in PowerShell"),
        ("Remove-Item -Recurse -Force $target", "OP-S05", "a target the gate cannot resolve is recorded")]:
    got = ps(cmd)
    check("powershell %-66s -> %s (%s)" % (cmd, want, why), want in got, got)
for cmd, why in [("Remove-Item notes.txt", "a single file"), ("Remove-Item -Recurse -Force node_modules", "a build directory"),
                 ("Remove-Item -Recurse -WhatIf ~/Documents", "a dry run"), ("Get-ChildItem -Recurse ~", "a listing"),
                 ("Remove-Item -Recurse -Force $env:TEMP/oplite-scratch", "the temp directory"),
                 ("Write-Output 'Remove-Item -Recurse -Force ~'", "a quoted sentence")]:
    got = [r for r in ps(cmd) if g.RULES[r][1] == "STOP"]
    check("powershell %-66s -> allowed (%s)" % (cmd, why), not got, got)

# Part A, continued: UTF-8 in and out, on every platform. The payload goes in as bytes and the
# reply is read as bytes, the way a host sends and reads them.
u_home = tempfile.mkdtemp(prefix="oplite-utf8-")
with open(os.path.join(u_home, "mode"), "w") as f:
    f.write("enforce\n")
u_env = dict(os.environ, OPERATOR_HOME=u_home)
u_env.pop("OPERATOR_AGENT_SESSION", None)
u_env.pop("OPERATOR_GATE_MODE", None)
odd = "rm -rf ~/Documents/\u65e5\u672c\u8a9e-\u201dnotes\u201d-\u00fc"      # CJK, a curly quote cp1252 cannot decode, an umlaut
payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": odd}, "cwd": os.path.expanduser("~"),
                      "session_id": "t"}, ensure_ascii=False).encode("utf-8")
p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload, capture_output=True, env=u_env)
err = p.stderr.decode("utf-8", errors="strict") if p.stderr else ""
check("a command with non-ASCII text is still stopped, not failed open", p.returncode == 2 and "OPERATOR: STOPPED (OP-003" in err, (p.returncode, p.stderr[:200]))
if p.returncode != 2:                                # say what the gate saw, so the cause is in the log
    try:
        _, _, d_hits, _, _, d_eff = g.classify({"tool_name": "Bash", "tool_input": {"command": odd}, "cwd": os.path.expanduser("~")}, g.load_config())
        print("diag in-process hits=%s" % ascii(sorted(set(h[0] for h in d_hits))))
        for e in d_eff:
            print("diag effect prog=%s kind=%s recursive=%s unresolved=%s targets=%s" % (
                ascii(e.get("prog")), e.get("kind"), e.get("recursive"), e.get("unresolved"), ascii(e.get("targets"))))
            for t in e.get("targets") or []:
                print("diag target scratch=%s variants=%s" % (g.is_scratch(t), ascii(sorted(g.variants(t)))))
        print("diag scratch_roots=%s home=%s" % (ascii(sorted(g.scratch_roots())), ascii(g.HOME)))
    except Exception as exc:
        print("diag in-process classify raised %r" % (exc,))
    try:
        rdir = os.path.join(u_home, "receipts")
        for fn in sorted(x for x in os.listdir(rdir) if x.endswith(".jsonl")):
            with open(os.path.join(rdir, fn), encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    print("diag receipt decision=%s commitments=%s error=%s preview=%s" % (
                        r.get("decision"), r.get("commitments"), ascii(r.get("error")), ascii(r.get("action_preview"))))
    except Exception as exc:
        print("diag receipts unreadable %r" % (exc,))
check("the stop message is valid UTF-8 and names the target as written", "\u65e5\u672c\u8a9e" in err, err[:300])
p = subprocess.run([sys.executable, GATE, "verify"], capture_output=True, env=u_env)
check("the receipt with non-ASCII text is in the chain and the chain verifies", b"receipts=1 chain=OK" in p.stdout, p.stdout[:200])
with open(os.path.join(u_home, "receipts", sorted(x for x in os.listdir(os.path.join(u_home, "receipts")) if x.endswith(".jsonl"))[0]), encoding="utf-8") as f:
    check("the receipt file is UTF-8 and keeps the text", "\u65e5\u672c\u8a9e" in f.read())
shutil.rmtree(u_home, ignore_errors=True)

# install keeps non-ASCII text in the user's settings file and copies the gate byte for byte
i_home = os.path.join(os.path.expanduser("~"), "oplite-utf8-home")
shutil.rmtree(i_home, ignore_errors=True)
os.makedirs(os.path.join(i_home, ".claude"))
note = "\u65e5\u672c\u8a9e \u201dquoted\u201d \u00fc"
with open(os.path.join(i_home, ".claude", "settings.json"), "w", encoding="utf-8") as f:
    json.dump({"env": {"NOTE": note}}, f, ensure_ascii=False)
i_env = dict(os.environ, HOME=i_home, USERPROFILE=i_home)
for k in ("OPERATOR_HOME", "OPERATOR_AGENT_SESSION", "OPERATOR_GATE_MODE"):
    i_env.pop(k, None)
p = subprocess.run([sys.executable, GATE, "install", "--no-replay"], capture_output=True, env=i_env, cwd=i_home)
check("install exits 0 beside a settings file with non-ASCII text", p.returncode == 0, p.stderr[-300:])
with open(os.path.join(i_home, ".claude", "settings.json"), encoding="utf-8") as f:
    after = json.load(f)
check("the non-ASCII text in the settings file is unchanged", after.get("env", {}).get("NOTE") == note and "PreToolUse" in after.get("hooks", {}), after.get("env"))
with open(os.path.join(i_home, ".operator", "gate", "operator-gate.py"), "rb") as a, open(GATE, "rb") as b:
    check("the installed gate is byte for byte the file that was run", a.read() == b.read())
shutil.rmtree(i_home, ignore_errors=True)

if os.name != "nt":
    check("on this platform a path is left exactly as written", g.canon("/c/Users/me") == "/c/Users/me" and g.canon("a\\b") == "a\\b")
    print("\n(Part B runs on Windows only)")
    print("\nfailures: %d" % len(fails))
    sys.exit(1 if fails else 0)

# Part B: Windows only
op_home = tempfile.mkdtemp(prefix="oplite-win-")
with open(os.path.join(op_home, "mode"), "w") as f:
    f.write("enforce\n")
env = dict(os.environ, OPERATOR_HOME=op_home)
env.pop("OPERATOR_AGENT_SESSION", None)
env.pop("OPERATOR_GATE_MODE", None)
HOME = os.path.expanduser("~")
work = os.path.join(HOME, "oplite-win-work")           # a project folder outside the temp directory
os.makedirs(work, exist_ok=True)


def hook(cmd, cwd=None, tool="Bash"):
    payload = json.dumps({"tool_name": tool, "tool_input": {"command": cmd}, "cwd": cwd or work, "session_id": "t"})
    p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload,
                       capture_output=True, text=True, env=env)
    return p.returncode, (p.stderr.strip().splitlines() or [""])[0]


def stop(name, cmd, rule, tool="Bash"):
    rc, first = hook(cmd, tool=tool)
    check("stop   " + name, rc == 2 and ("(%s " % rule) in first, "exit %d %s" % (rc, first))


def allow(name, cmd, tool="Bash"):
    rc, first = hook(cmd, tool=tool)
    check("allow  " + name, rc == 0, "exit %d %s" % (rc, first))


drive, tail = HOME[0], HOME[2:].replace("\\", "/")
msys_home = "/%s%s" % (drive.lower(), tail)             # /c/Users/name
short_home = HOME
try:
    import ctypes
    buf = ctypes.create_unicode_buffer(600)
    if ctypes.windll.kernel32.GetShortPathNameW(HOME, buf, 600):
        short_home = buf.value
except Exception:
    pass
print("home=%s  git-bash=%s  short=%s  temp=%s" % (HOME, msys_home, short_home, tempfile.gettempdir()))

stop("recursive delete of the home directory, tilde", "rm -rf ~", "OP-001")
stop("recursive delete of the home directory, Git Bash path", "rm -rf %s" % msys_home, "OP-001")
stop("recursive delete of the home directory, quoted Windows path", 'rm -rf "%s"' % HOME, "OP-001")
stop("recursive delete of the home directory, forward slashes", "rm -rf %s" % HOME.replace("\\", "/"), "OP-001")
stop("recursive delete of the home directory, other letter case", "rm -rf %s" % msys_home.upper().replace("/" + drive.upper() + "/", "/" + drive.lower() + "/", 1), "OP-001")
if short_home.lower() != HOME.lower():
    stop("recursive delete of the home directory through its 8.3 short name", 'rm -rf "%s"' % short_home, "OP-001")
else:
    print("skip  this home directory has no distinct 8.3 short name")
stop("recursive delete of a folder in the home directory", "rm -rf ~/Documents", "OP-003")
stop("recursive delete through USERPROFILE", 'rm -rf "$USERPROFILE/Documents"', "OP-003")
stop("force-push", "git push --force origin main", "OP-004")
allow("recursive delete inside the temp directory, Git Bash /tmp", "rm -rf /tmp/oplite-scratch")
allow("recursive delete inside the Windows temp directory", 'rm -rf "%s"' % os.path.join(tempfile.gettempdir(), "oplite-scratch"))
allow("recursive delete of a build directory in the project", "rm -rf node_modules")
allow("plain command", "git status")

# a session whose working directory is the home directory, in the spelling Windows reports it
rc, first = hook("rm -rf ~/Documents/old-project", cwd=HOME)
check("stop   recursive delete inside the home directory when the session started there", rc == 2 and "(OP-003 " in first, "exit %d %s" % (rc, first))
rc, first = hook("rm -rf old-project", cwd=os.path.join(HOME, "Documents"))
check("stop   recursive delete of a relative folder with the session in Documents", rc == 2 and "(OP-003 " in first, "exit %d %s" % (rc, first))
rc, first = hook("Remove-Item -Recurse -Force old-project", cwd=HOME.upper(), tool="PowerShell")
check("stop   the same from PowerShell with the directory in another letter case", rc == 2 and "(OP-003 " in first, "exit %d %s" % (rc, first))

# the PowerShell tool, with Windows paths
stop("PowerShell: Remove-Item -Recurse on the home directory", "Remove-Item -Recurse -Force %s" % HOME, "OP-001", tool="PowerShell")
stop("PowerShell: the same through $env:USERPROFILE", "Remove-Item -Recurse -Force $env:USERPROFILE", "OP-001", tool="PowerShell")
if short_home.lower() != HOME.lower():
    stop("PowerShell: the same through the 8.3 short name", 'Remove-Item -Recurse -Force "%s"' % short_home, "OP-001", tool="PowerShell")
stop("PowerShell: a folder in the home directory, backslashes", "rd -r -fo $env:USERPROFILE\\Documents", "OP-003", tool="PowerShell")
allow("PowerShell: the temp directory", 'Remove-Item -Recurse -Force "$env:TEMP\\oplite-scratch"', tool="PowerShell")
allow("PowerShell: a listing", "Get-ChildItem -Recurse $env:USERPROFILE", tool="PowerShell")

# install on Windows: the hook the gate registers must be a command Windows can run
fake = os.path.join(HOME, "oplite-win-home")
shutil.rmtree(fake, ignore_errors=True)
os.makedirs(os.path.join(fake, ".claude"))
env2 = dict(env, USERPROFILE=fake, HOME=fake)
env2.pop("OPERATOR_HOME", None)
p = subprocess.run([sys.executable, GATE, "install"], capture_output=True, text=True, env=env2, cwd=fake)
check("install exits 0 in a fresh Windows home", p.returncode == 0, (p.stdout + p.stderr)[-400:])
try:
    with open(os.path.join(fake, ".claude", "settings.json")) as f:
        wired = json.load(f)["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
except Exception as exc:
    wired = repr(exc)
print("wired hook: %s" % wired)
check("the registered hook names this interpreter and the installed gate", wired.startswith('"') and "python3 " not in wired
      and "/.operator/gate/operator-gate.py" in wired and wired.endswith("hook --source claude"), wired)
payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git push --force origin main"}, "cwd": work, "session_id": "t"})
p = subprocess.run(wired, shell=True, input=payload, capture_output=True, text=True, env=env2)
check("the registered hook runs through the Windows shell and records in shadow", p.returncode == 0 and not p.stderr.strip(), p.stderr[:200])
with open(os.path.join(fake, ".operator", "mode"), "w") as f:
    f.write("enforce\n")
p = subprocess.run(wired, shell=True, input=payload, capture_output=True, text=True, env=env2)
check("after the flip to enforce the registered hook stops the force-push", p.returncode == 2 and "OP-004" in p.stderr, p.stderr[:200])
shutil.rmtree(fake, ignore_errors=True)

p = subprocess.run([sys.executable, GATE, "verify"], capture_output=True, text=True, env=env)
check("receipts were written under the Windows lock and the chain verifies", "chain=OK" in p.stdout and "receipts=0" not in p.stdout, p.stdout[:200])

shutil.rmtree(op_home, ignore_errors=True)
shutil.rmtree(work, ignore_errors=True)
print("\nfailures: %d" % len(fails))
sys.exit(1 if fails else 0)
