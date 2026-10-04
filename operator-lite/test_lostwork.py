#!/usr/bin/env python3
"""Work nothing can bring back: the gate reads the state of the files and the repository.
A throwaway home holds a repository with committed, changed, untracked, ignored and fresh files.
Nothing is executed: the gate only reads. Usage: test_lostwork.py [gate.py] [-v]"""
import json, os, shutil, subprocess, sys, tempfile, time

GATE = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator-gate.py"))
VERBOSE = "-v" in sys.argv
home = tempfile.mkdtemp(prefix="loss-", dir=os.path.expanduser("~"))
repo = os.path.join(home, "work", "shop")
plain = os.path.join(home, "work", "plain")      # not a repository
clean = os.path.join(home, "work", "clean")      # a repository with nothing uncommitted
os.makedirs(os.path.join(home, ".operator"))
with open(os.path.join(home, ".operator", "mode"), "w") as f:
    f.write("enforce\n")
env = dict(os.environ, HOME=home, USERPROFILE=home, OPERATOR_HOME=os.path.join(home, ".operator"),
           GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
env.pop("OPERATOR_AGENT_SESSION", None)
env.pop("OPERATOR_GATE_MODE", None)
OLD = time.time() - 3 * 86400


def put(base, rel, body="x\n", old=True):
    p = os.path.join(base, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(body)
    if old:
        os.utime(p, (OLD, OLD))
    return p


def git(base, *args):
    subprocess.run(["git", "-C", base, "-c", "user.email=t@example.com", "-c", "user.name=t"] + list(args),
                   check=True, capture_output=True, env=env)


for base in (repo, clean):
    os.makedirs(base)
    git(base, "init", "-q", "-b", "main")
    put(base, "src/app.py", "print(1)\n")
    put(base, "lib/a.py", "a = 1\n")
    put(base, "lib/b.py", "b = 2\n")
    put(base, "README.md", "# shop\n")
    put(base, ".gitignore", ".env\nnode_modules/\n*.log\ndata/\n")
    git(base, "add", "-A")
    git(base, "commit", "-q", "-m", "first")
    os.utime(os.path.join(base, ".git"), (OLD, OLD))         # a repository that has been there for days
put(repo, "src/app.py", "print(2)  # a day of work, not committed\n")      # changed, old
put(repo, "notes.md", "ideas\n")                                             # untracked, old
put(repo, "scratch/tmp.txt", "made a minute ago\n", old=False)               # untracked, fresh
put(repo, ".env", "KEY=1\n")                                                 # ignored, cannot be made again
put(repo, "debug.log", "noise\n")                                            # ignored, can go
put(repo, "node_modules/x/index.js", "x\n")                                  # ignored build output
put(repo, "data/local.sqlite", "db\n")                                       # ignored folder holding a database
put(repo, "lib/b.py", "b = 3  # changed a minute ago, still not committed\n", old=False)
os.makedirs(os.path.join(repo, "vendor-clone"))
git(os.path.join(repo, "vendor-clone"), "init", "-q", "-b", "main")           # cloned a minute ago
put(os.path.join(repo, "vendor-clone"), "x.py", "x\n", old=False)
put(clean, "lib/a.py", "a = 9  # put away in a stash\n")
git(clean, "stash")                                                          # clean again, with one stash kept
put(plain, "report/q3.txt", "numbers\n")
put(plain, "fresh/out.txt", "just made\n", old=False)


def hook(cmd, cwd):
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": cwd, "session_id": "t"})
    p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload,
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stderr


CASES = [  # (must stop under this rule or None, where, command, what it is)
    ("OP-011", repo, "rm -rf src", "a folder with an uncommitted change"),
    ("OP-011", repo, "rm src/app.py", "one changed file"),
    ("OP-011", repo, "rm notes.md", "one untracked file that has been there for days"),
    ("OP-011", repo, "rm -rf *", "everything in the project"),
    ("OP-011", repo, "rm -rf ./*", "everything in the project, with ./"),
    ("OP-011", repo, "rm .env", "an ignored file that holds secrets"),
    ("OP-011", repo, "rm -rf data", "an ignored folder that holds a local database"),
    ("OP-004", repo, "rm -rf .", "the repository itself"),
    ("OP-004", repo, "find . -delete", "the repository through find"),
    ("OP-011", repo, "git reset --hard", "reset --hard over uncommitted changes"),
    ("OP-011", repo, "git reset --hard HEAD~0", "reset --hard to a commit"),
    ("OP-011", repo, "git checkout -- .", "checkout over every changed file"),
    ("OP-011", repo, "git checkout -- src/app.py", "checkout over one changed file"),
    ("OP-011", repo, "git checkout src/app.py", "checkout of a path without --"),
    ("OP-011", repo, "git restore src/app.py", "restore over one changed file"),
    ("OP-011", repo, "git restore .", "restore over everything"),
    ("OP-011", repo, "git checkout -f main", "a forced checkout"),
    ("OP-011", repo, "git switch --discard-changes main", "a switch that discards changes"),
    ("OP-011", repo, "git clean -fd", "clean over an untracked file"),
    ("OP-011", repo, "git clean -fdx", "clean that also takes .env"),
    ("OP-011", repo, "git -C %s reset --hard" % repo, "reset --hard through -C from elsewhere"),
    ("OP-011", plain, "rm -rf report", "a folder outside any repository"),
    ("OP-011", plain, "rm report/q3.txt", "a file outside any repository"),
    ("OP-011", repo, "git checkout -- lib/b.py", "a tracked file changed a minute ago: still work"),
    ("OP-011", repo, "rm -rf lib", "a folder holding that change"),
    (None, repo, "rm lib/a.py", "a committed, unchanged file"),
    (None, repo, "rm -rf vendor-clone", "a repository cloned a minute ago"),
    (None, clean, "rm -rf lib", "a committed, unchanged folder: git can restore it"),
    (None, repo, "rm -rf scratch", "a folder made a minute ago"),
    (None, repo, "rm debug.log", "an ignored log"),
    (None, repo, "rm -rf node_modules", "build output"),
    (None, repo, "git stash && git reset --hard", "reset --hard after stashing"),
    (None, repo, "git stash -u && git clean -fd", "clean after stashing untracked files"),
    (None, repo, "git commit -am wip && git reset --hard HEAD~1", "reset --hard after committing everything"),
    (None, repo, "git checkout main", "a plain switch"),
    (None, repo, "git restore --staged src/app.py", "unstaging only"),
    (None, repo, "git clean -n -fd", "a dry run of clean"),
    (None, repo, "git status", "git status"),
    (None, repo, "git diff", "git diff"),
    (None, clean, "git reset --hard", "reset --hard with nothing uncommitted"),
    (None, clean, "git checkout -- .", "checkout with nothing changed"),
    (None, clean, "git clean -fd", "clean with nothing untracked"),
    (None, clean, "rm -rf src", "a committed folder in a clean repository"),
    (None, plain, "rm -rf fresh", "a folder made a minute ago, outside any repository"),
    (None, repo, "trash notes.md", "moving a file to the trash"),
    ("OP-011", clean, "git stash drop", "dropping the only copy of stashed work"),
    ("OP-011", clean, "git stash clear", "clearing every stash"),
    ("OP-011", repo, "git stash && git stash drop", "stashing the changes and dropping the stash"),
    (None, repo, "git stash drop", "stash drop with no stash to lose"),
    (None, clean, "git stash list", "listing stashes"),
    ("OP-004", repo, "git push origin :main", "deleting the remote main with a colon refspec"),
    ("OP-004", repo, "git push origin --delete main", "deleting the remote main by flag"),
    ("OP-004", repo, "git push --mirror", "a mirror push"),
    (None, repo, "git push origin --delete fix-main-menu", "deleting a branch whose name only contains main"),
    (None, repo, "git push origin :old-feature", "deleting a remote feature branch"),
    ("OP-003", repo, "rsync -a --delete /tmp/empty/ ~/Documents/", "rsync --delete over a home folder"),
    (None, repo, "rsync -a --delete src/ dist/", "rsync --delete into a build folder"),
    (None, repo, "rsync -a src/ ~/Documents/copy/", "rsync without --delete"),
]
bad = 0
for rule, cwd, cmd, what in CASES:
    rc, err = hook(cmd, cwd)
    first = (err.strip().splitlines() or [""])[0]
    ok = (rc == 0) if rule is None else (rc == 2 and ("(%s " % rule) in err)
    bad += not ok
    if VERBOSE or not ok:
        why = (err.strip().splitlines() + ["", ""])[1][:150] if rc == 2 else ""
        print("%s %-6s %-52s %s %s" % ("ok " if ok else "BAD", rule or "pass", what[:52], first[:46], why))
print("lost-work cases: %d, wrong: %d" % (len(CASES), bad))
shutil.rmtree(home, ignore_errors=True)
sys.exit(1 if bad else 0)
