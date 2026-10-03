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


def hook(cmd, cwd=None):
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": cwd or work, "session_id": "t"})
    p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload,
                       capture_output=True, text=True, env=env)
    return p.returncode, (p.stderr.strip().splitlines() or [""])[0]


def stop(name, cmd, rule):
    rc, first = hook(cmd)
    check("stop   " + name, rc == 2 and ("(%s " % rule) in first, "exit %d %s" % (rc, first))


def allow(name, cmd):
    rc, first = hook(cmd)
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

p = subprocess.run([sys.executable, GATE, "verify"], capture_output=True, text=True, env=env)
check("receipts were written under the Windows lock and the chain verifies", "chain=OK" in p.stdout and "receipts=0" not in p.stdout, p.stdout[:200])

shutil.rmtree(op_home, ignore_errors=True)
shutil.rmtree(work, ignore_errors=True)
print("\nfailures: %d" % len(fails))
sys.exit(1 if fails else 0)
