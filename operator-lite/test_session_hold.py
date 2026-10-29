#!/usr/bin/env python3
"""Operator Lite 6.0.0, a stop holds for the whole session (OP-020): once the gate stops an action on a
target, the same session reaching that target another way, a script, inline code or a different command,
is stopped as the same action and told the first receipt. Another session is judged on its own; carrying
the owner's no across sessions, hosts and other agent CLIs is the paid Operator's.

A throwaway home under this folder (never a temp folder); the gate runs as a hook in enforce mode.
Nothing is ever run.
Usage: python3 operator-lite/test_session_hold.py"""
import json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, "operator-gate.py")
BOX = tempfile.mkdtemp(prefix="op-ledger-", dir=HERE)
home, op = os.path.join(BOX, "home"), os.path.join(BOX, "op")
AGENT = ("OPERATOR_AGENT_SESSION", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
env = {k: v for k, v in os.environ.items() if k not in ("OPERATOR_HOME", "OPERATOR_GATE_MODE") + AGENT}
env.update(HOME=home, OPERATOR_HOME=op, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
wrong, ran, last = [], [], {}


def check(label, ok, detail=""):
    ran.append(label)
    print("%s %s%s" % ("ok  " if ok else "BAD ", label, ("  -- " + str(detail)[:240]) if detail and not ok else ""))
    if not ok:
        wrong.append(label)


def hook(cmd, session, cwd, source="claude"):
    payload = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": cwd,
               "session_id": session, "tool_use_id": "toolu_%s" % session}
    p = subprocess.run([sys.executable, GATE, "hook", "--source", source], input=json.dumps(payload),
                       capture_output=True, text=True, env=env, cwd=cwd, timeout=60)
    last["out"] = p.stdout
    return p.returncode, p.stderr


def ends_turn():
    """Whether the last answer asked the host to end the agent's turn (Claude Code `continue: false`)."""
    for line in (last.get("out") or "").splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if j.get("continue") is False and "receipt" in str(j.get("stopReason") or ""):
            return True
    return False


notes = os.path.join(home, "notes")
for i in range(3):
    with open(os.path.join(os.makedirs(notes, exist_ok=True) or notes, "n%d.md" % i), "w") as f:
        f.write("note %d\n" % i)
work = os.path.join(home, "work")
os.makedirs(work)
with open(os.path.join(work, "wipe.sh"), "w") as f:
    f.write('for f in "$HOME/notes"/*; do unlink "$f"; done\n')
with open(os.path.join(work, "peek.sh"), "w") as f:
    f.write('ls -la "$HOME/notes"\n')
with open(os.path.join(work, "tidy.sh"), "w") as f:
    f.write('for f in "$HOME/scratch-other"/*; do unlink "$f"; done\n')
os.makedirs(op)
with open(os.path.join(op, "mode"), "w") as f:
    f.write("enforce\n")

# before any stop, the script that empties ~/notes is judged on its own
rc, err = hook("bash wipe.sh", "fresh", work, "codex")
before = rc
check("with no stop on record, the script is judged on its own", rc in (0, 2), err[:200])

rc, err = hook("rm -rf ~/notes", "agent-a", home, "claude")
check("agent A is stopped deleting ~/notes", rc == 2 and "OP-003" in err, err[:240])
check("the first stop blocks the call and lets the turn go on", not ends_turn(), last.get("out"))

rc, err = hook("bash wipe.sh", "agent-a", work, "claude")
check("agent A, in the same session, is stopped emptying the same folder by a script (OP-020)",
      rc == 2 and "OP-020" in err, err[:400])
check("the second try at the stopped action ends the turn, naming the first receipt (#530)", ends_turn(),
      last.get("out"))

rc, err = hook("bash peek.sh", "agent-a", work, "claude")
check("a script that only names the stopped folder is stopped (OP-020 names)", rc == 2 and "OP-020" in err, err[:300])
check("and only naming it does not end the turn", not ends_turn(), last.get("out"))

rc, err = hook("rm -rf ~/notes", "agent-c", home, "codex")
rc, err = hook("bash wipe.sh", "agent-c", work, "codex")
check("on a host without turn control the second try is still stopped", rc == 2 and "OP-020" in err, err[:300])
check("and the gate sends no Claude Code turn control to that host", not ends_turn(), last.get("out"))

rc, err = hook("cat ~/notes/n0.md", "agent-a", home, "claude")
check("the same session may still read the folder", rc == 0, err[:200])
rc, err = hook("bash tidy.sh", "agent-a", work, "claude")
check("an unrelated folder stays free in the same session", rc == 0, err[:200])

rc, err = hook("bash wipe.sh", "agent-b", work, "codex")
check("agent B, in another session, is judged on its own", rc == before and "OP-020" not in err, (rc, err[:200]))
check("no machine-wide ledger is written", not os.path.isfile(os.path.join(op, "state", "chain", "_machine.jsonl")))

shutil.rmtree(BOX, ignore_errors=True)
print("\nsession hold: %d checks, wrong %d" % (len(ran), len(wrong)))
sys.exit(1 if wrong else 0)
