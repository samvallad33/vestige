#!/usr/bin/env python3
"""Tests for the 0.3.5 parser fixes in Operator Lite: benign commands the gate used to stop, and the
writes it must still stop. Runs the gate as a hook in enforce mode with a throwaway OPERATOR_HOME.
Classification only: no sample command is ever executed.
Usage: python3 operator-lite/test_parser.py [repo-root]"""
import json, os, shutil, subprocess, sys, tempfile

W = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(W, "operator-lite", "operator-gate.py")
fails = []
op_home = tempfile.mkdtemp(prefix="oplite-parser-")
with open(os.path.join(op_home, "mode"), "w") as f:
    f.write("enforce\n")
env = dict(os.environ, OPERATOR_HOME=op_home)
env.pop("OPERATOR_AGENT_SESSION", None)
env.pop("OPERATOR_GATE_MODE", None)
work = tempfile.mkdtemp(prefix="oplite-parser-work-")


def hook(cmd, cwd=None):
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": cwd or work, "session_id": "t"})
    p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload,
                       capture_output=True, text=True, env=env)
    return p.returncode, (p.stderr.strip().splitlines() or [""])[0]


def receipts():
    out = []
    rdir = os.path.join(op_home, "receipts")
    for fn in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        if fn.endswith(".jsonl"):
            with open(os.path.join(rdir, fn)) as f:
                out += [json.loads(line) for line in f]
    return out


def allow(name, cmd, cwd=None):
    rc, first = hook(cmd, cwd)
    ok = rc == 0
    print("%s  allow  %s" % ("ok " if ok else "BAD", name) + ("" if ok else "   -> exit %d %s" % (rc, first)))
    if not ok:
        fails.append(name)


def stop(name, cmd, rule, cwd=None):
    rc, first = hook(cmd, cwd)
    ok = rc == 2 and ("(%s " % rule) in first
    print("%s  stop   %s" % ("ok " if ok else "BAD", name) + ("" if ok else "   -> exit %d %s" % (rc, first)))
    if not ok:
        fails.append(name)


def shadow(name, cmd, rule, want=True):
    before = len(receipts())
    rc, first = hook(cmd)
    new = receipts()[before:]
    flagged = any(rule in (r.get("commitments") or []) for r in new)
    ok = rc == 0 and flagged == want
    print("%s  %s %s" % ("ok " if ok else "BAD", "shadow" if want else "quiet ", name)
          + ("" if ok else "   -> exit %d flagged=%s %s" % (rc, flagged, first)))
    if not ok:
        fails.append(name)


# a script in another language is code in that language, never shell lines
script = os.path.join(work, "report.py")
with open(script, "w") as f:
    f.write("import json\nrg, _ = compute()\nrm = [x for x in rg]\nprint(json.dumps(rm))\n")
allow("python script whose variables are named like shell programs", "python3 %s --fast" % script)
allow("the same script read from stdin form", "python3 - %s" % script)

# a shell script is still walked as shell, by its own body
sh = os.path.join(work, "cleanup.sh")
with open(sh, "w") as f:
    f.write("#!/bin/sh\nrm -rf ~/Documents\n")
stop("shell script whose body deletes a home folder", "bash %s" % sh, "OP-003")

# parentheses inside quotes are data, not a subshell
allow("quoted alternation is data", 'echo "(alpha|beta)" | head -1')
allow("quoted alternation in a pipe filter", 'git status --short | grep -v -E "(build|target)" | head -20')
allow("a quoted sentence that describes a delete", 'echo "(to reset, run rm -rf ~ yourself)"')
stop("a real subshell group still counts", "(sleep 300; rm -rf ~) &", "OP-001")

# invoking the gate file is the gate program: its source is never walked as code
allow("gate verify through python3", "python3 %s verify" % GATE)
allow("gate status and replay through python3", "python3 %s status && python3 %s replay --json" % (GATE, GATE))
stop("gate install through python3 stays owner-only", "python3 %s install" % GATE, "OP-000")
stop("gate approve through python3 stays owner-only", "python3 %s approve 0123456789abcdef01234567" % GATE, "OP-000")

# inline code is judged by the path each call writes or deletes
stop("inline write to the gate's rules file, nested call",
     "python3 -c \"import os; open(os.path.expanduser('~/.operator/commitments.json'),'w').write('{}')\"", "OP-000")
stop("inline write to the gate's mode file through a variable",
     "python3 - <<'EOF'\nimport os\np = os.path.expanduser('~/.operator/mode')\nopen(p, 'w').write('off')\nEOF", "OP-000")
allow("inline write to an ordinary file", "python3 -c \"open('/tmp/opgate-out.txt','w').write('x')\"")
allow("inline note that mentions a gate path but writes elsewhere",
      "python3 - <<'EOF'\np = \"/tmp/opgate-note.md\"\ns = open(p).read()\n"
      "old = \"the off switch is the file ~/.operator/DISABLED\"\nopen(p, \"w\").write(s.replace(old, \"switched on\"))\nEOF")
shadow("inline write through a variable the gate cannot resolve is not a finding",
       "python3 - <<'EOF'\nimport sys\nopen(sys.argv[1], 'w').write('x')\nEOF", "OP-S05", want=False)
shadow("inline delete through a variable the gate cannot resolve is recorded",
       "python3 - <<'EOF'\nimport os, sys\nos.remove(sys.argv[1])\nEOF", "OP-S05", want=True)
stop("unresolved inline write in code that names a gate path fails closed",
     "python3 - <<'EOF'\nimport os, sys\nnote = '~/.operator/commitments.json'\nopen(sys.argv[1], 'w').write(note)\nEOF", "OP-000")

# a redirection written before the command does not hide the command (found 2026-10-03)
stop("redirect first, fused", ">/tmp/opgate.log rm -rf ~/Documents", "OP-003")
stop("redirect first, spaced", "> /tmp/opgate.log rm -rf ~/Documents", "OP-003")
stop("stderr redirect first", "2>/dev/null rm -rf ~/Documents", "OP-003")
stop("input redirect first", "</dev/null rm -rf ~/Documents", "OP-003")
stop("redirect, then a wrapper, then the command", ">/tmp/opgate.log sudo rm -rf ~/Documents", "OP-003")
allow("redirect first on a harmless command", ">/tmp/opgate.log echo hello")

# bash syntax that is not a program
loop = os.path.join(work, "collect.sh")
with open(loop, "w") as f:
    f.write("#!/bin/bash\nassets=()\nwhile read -r line; do\n  assets+=(\"$line\")\ndone <<<\"$listing\"\necho ${#assets[@]}\n")
shadow("array append and a here-string after done are not dynamic programs", "bash %s" % loop, "OP-S05", want=False)

stop("a command substitution in an assignment is still analyzed", "x=$(rm -rf ~/Documents)", "OP-003")

# the old true positives still hold
stop("force-push", "git push --force origin main", "OP-004")
stop("recursive delete of a home folder", "rm -rf ~/Documents", "OP-003")
allow("plain build command", "cargo check 2>&1 | tail -5")

shutil.rmtree(op_home, ignore_errors=True)
shutil.rmtree(work, ignore_errors=True)
print("\nfailures: %d" % len(fails))
sys.exit(1 if fails else 0)
