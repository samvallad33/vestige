"""Two ways an agent could get past Operator Lite, both closed:

1. A force push that names the protected branch only through HEAD or @ (or names no branch, from a folder
   where the branch cannot be read) is OP-004, like `git push --force origin main`.
2. The gate's own package and hook settings are the owner's: running `npx|bunx|pnpm dlx|npm exec operator-lite`
   to install, uninstall, change mode or approve is OP-000, and so is an agent writing `hooks` or
   `disableAllHooks` into a project's .claude/settings*.json (project settings outrank the user's).
Reads and ordinary work stay allowed. No command is executed: the gate only judges them.
Run: /usr/bin/python3 test_self_protect.py [path/to/operator-gate.py]
"""
import json, os, subprocess, sys, tempfile

GATE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator-gate.py"))
ROOT = tempfile.mkdtemp(prefix="lite-self-protect-", dir=os.path.expanduser("~"))
fails = []


def check(name, cond, detail=""):
    print("%s  %s%s" % ("ok " if cond else "BAD", name, ("   -> " + str(detail)[:300]) if not cond else ""))
    if not cond:
        fails.append(name)


home = os.path.join(ROOT, "home")
os.makedirs(os.path.join(home, ".operator"))
with open(os.path.join(home, ".operator", "mode"), "w") as f:
    f.write("enforce\n")
env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "OPERATOR_AGENT_SESSION")}
env.update(HOME=home, OPERATOR_HOME=os.path.join(home, ".operator"), PATH="/usr/bin:/bin:/opt/homebrew/bin")


def repo(name, branch):
    d = os.path.join(home, "work", name)
    os.makedirs(d)
    for args in (["init", "-q", "-b", branch], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init"]):
        subprocess.run(["git", "-C", d] + args, check=True, env=env, capture_output=True)
    return d


MAIN, FEAT = repo("on-main", "main"), repo("on-feature", "feature-x")
OUTSIDE = os.path.join(home, "work")


def judge(payload, cwd):
    payload = dict(payload, cwd=cwd, session_id="self-protect", hook_event_name="PreToolUse")
    p = subprocess.run([sys.executable, "-I", GATE, "hook", "--source", "claude"], input=json.dumps(payload),
                       capture_output=True, text=True, env=env, cwd=cwd, timeout=60)
    return p.returncode, p.stdout + p.stderr


def bash(cmd):
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


def stop(name, payload, cwd, rule):
    rc, out = judge(payload, cwd)
    check("STOP %s: %s" % (rule, name), rc == 2 and rule in out, "rc=%s %s" % (rc, out.strip()[:200]))


def allow(name, payload, cwd):
    rc, out = judge(payload, cwd)
    check("PASS: " + name, rc == 0, "rc=%s %s" % (rc, out.strip()[:200]))


# 1. force pushes
stop("--force origin HEAD on main", bash("git push --force origin HEAD"), MAIN, "OP-004")
stop("-f origin @ on main", bash("git push -f origin @"), MAIN, "OP-004")
stop("origin +HEAD on main", bash("git push origin +HEAD"), MAIN, "OP-004")
stop("git -C <repo on main> push --force from outside it", bash("git -C %s push --force" % MAIN), OUTSIDE, "OP-004")
stop("bare push --force where no branch can be read", bash("git push --force"), OUTSIDE, "OP-004")
stop("--force origin main (unchanged)", bash("git push --force origin main"), MAIN, "OP-004")
allow("--force origin HEAD on a feature branch", bash("git push --force origin HEAD"), FEAT)
allow("--force-with-lease origin HEAD on a feature branch", bash("git push --force-with-lease origin HEAD"), FEAT)
allow("--force origin feature-x from main", bash("git push --force origin feature-x"), MAIN)
allow("a normal push of HEAD on main", bash("git push origin HEAD"), MAIN)

# 2. the gate's package and the project's hook settings
for cmd in ("npx operator-lite uninstall", "npx --yes operator-lite", "npx operator-lite --force", "pnpm dlx operator-lite",
            "bunx operator-lite install", "npx -p operator-lite operator-lite install", "npm exec operator-lite -- uninstall",
            "operator-lite mode enforce", "operator-lite approve 0123abcd"):
    stop(cmd, bash(cmd), MAIN, "OP-000")
for cmd in ("npx operator-lite status", "npx operator-lite replay --days 7", "cat operator-lite/README.md",
            "cd operator-lite && ls", "npx prettier --check ."):
    allow(cmd, bash(cmd), MAIN)

stop("printf disableAllHooks into .claude/settings.local.json",
     bash("mkdir -p .claude && printf '{\"disableAllHooks\": true}' > .claude/settings.local.json"), MAIN, "OP-000")
stop("Write tool puts hooks in .claude/settings.json",
     {"tool_name": "Write", "tool_input": {"file_path": os.path.join(MAIN, ".claude", "settings.json"),
                                            "content": json.dumps({"hooks": {"PreToolUse": []}})}}, MAIN, "OP-000")
stop("Edit tool adds disableAllHooks to .claude/settings.local.json",
     {"tool_name": "Edit", "tool_input": {"file_path": os.path.join(MAIN, ".claude", "settings.local.json"),
                                           "old_string": "{", "new_string": "{\"disableAllHooks\": true,"}}, MAIN, "OP-000")
allow("Write tool sets only permissions in .claude/settings.json",
      {"tool_name": "Write", "tool_input": {"file_path": os.path.join(MAIN, ".claude", "settings.json"),
                                             "content": json.dumps({"permissions": {"allow": ["Bash(ls)"]}})}}, MAIN)
allow("Write tool to an ordinary project file that mentions hooks",
      {"tool_name": "Write", "tool_input": {"file_path": os.path.join(MAIN, "docs", "hooks.md"),
                                             "content": "# hooks\ndisableAllHooks is a Claude Code setting."}}, MAIN)

print("\nfailures: %d" % len(fails))
sys.exit(1 if fails else 0)
