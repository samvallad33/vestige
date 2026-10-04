import importlib.util, os, sys, tempfile, shutil
home = tempfile.mkdtemp(prefix="pscheck-", dir=os.path.expanduser("~"))
os.environ["HOME"] = home; os.environ["OPERATOR_HOME"] = os.path.join(home, ".operator"); os.environ.pop("OPERATOR_AGENT_SESSION", None)
spec = importlib.util.spec_from_file_location("gate", (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator-gate.py"))); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
cfg = g.load_config(); work = os.path.join(home, "work"); os.makedirs(work)
def rules(cmd):
    out = []
    for e in g.analyze_powershell(cmd, work):
        out += [h[0] for h in g.classify_effect(e, cfg, work)]
    return sorted(set(out))
cases = [
 (r'& C:\msys64\usr\bin\bash.exe -lc "rm -rf /"', "OP-003|OP-001"),
 (r'& "C:\Program Files\Git\bin\bash.exe" -lc "rm -rf ~/Documents"', "OP-003"),
 (r'C:\msys64\usr\bin\bash.exe -c "git push --force origin main"', "OP-004"),
 (r'&"C:\Program Files\Git\bin\bash.exe" -c "rm -rf ~/Documents"', "OP-003"),
 (r'bash -lc "rm -rf /"', "OP-003|OP-001"),
 (r'& C:\php\php.exe artisan migrate:fresh --force', "OP-007"),
 (r'& C:\Python312\python.exe manage.py flush --noinput', "OP-007"),
 (r'& C:\msys64\usr\bin\bash.exe -lc "ls -la"', ""),
 (r'& C:\tools\node.exe --version', ""),
 (r'git status', ""),
 (r'& git status', ""),
 (r'. .\profile.ps1', ""),
 (r'Get-ChildItem C:\Users', ""),
]
bad = 0
for c, want in cases:
    r = rules(c); stop = [x for x in r if not x.startswith("OP-S")]
    ok = (not stop) if not want else any(w in stop for w in want.split("|"))
    bad += not ok
    print("%s %-72s %s" % ("ok " if ok else "BAD", c[:72], r))
print("wrong:", bad); shutil.rmtree(home, ignore_errors=True)
sys.exit(1 if bad else 0)
