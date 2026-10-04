#!/usr/bin/env python3
"""Tests for the parser in Operator Lite: benign commands the gate used to stop, and the writes,
deletes and database resets it must still stop. Runs the gate as a hook in enforce mode with a throwaway OPERATOR_HOME.
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

# a framework or admin command that empties or drops a database is the same act as DROP TABLE (found
# 2026-10-04: `php artisan migrate:fresh --seed --force` against the wrong environment passed the gate)
def hook_as(tool, cmd):
    payload = json.dumps({"tool_name": tool, "tool_input": {"command": cmd}, "cwd": work, "session_id": "t"})
    p = subprocess.run([sys.executable, GATE, "hook", "--source", "claude"], input=payload,
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stderr


def judged(name, cmd, rule, tool="Bash"):
    """rule=None: the command passes. Otherwise the gate stops it and names that rule."""
    rc, err = hook_as(tool, cmd)
    ok = (rc == 0) if rule is None else (rc == 2 and ("(%s " % rule) in err)
    print("%s  %s %s" % ("ok " if ok else "BAD", "allow " if rule is None else "stop  ", name)
          + ("" if ok else "   -> exit %d %s" % (rc, (err.strip().splitlines() or [""])[0])))
    if not ok:
        fails.append(name)


DB_RESETS = (
    "php artisan migrate:fresh --seed --force",
    "php artisan migrate:fresh --seed --force --env=production",
    "php artisan migrate:refresh",
    "php artisan migrate:reset",
    "php artisan db:wipe --force",
    "APP_ENV=production php artisan migrate:fresh",
    "./vendor/bin/sail artisan migrate:fresh --seed",
    "sail art migrate:fresh",
    "cd backend && php artisan migrate:fresh",
    "yes | php artisan migrate:fresh",
    "docker compose exec -T app php artisan migrate:fresh --force",
    "docker compose exec app sh -c 'php artisan migrate:fresh --seed'",
    'ssh deploy@prod "cd /var/www/app && php artisan migrate:fresh --force"',
    "npx prisma migrate reset --force",
    "npx prisma@6.2.1 migrate reset",
    "pnpm dlx prisma migrate reset",
    "yarn prisma migrate reset --skip-seed",
    "bunx prisma migrate reset -f",
    "pnpm exec prisma db push --force-reset",
    "npx prisma db push --accept-data-loss",
    "npx prisma --schema prisma/schema.prisma migrate reset",
    "npm run db:reset",
    "rails db:drop db:create db:migrate",
    "bin/rails db:drop",
    "bundle exec rails db:reset",
    "bundle exec rake db:drop:all",
    "RAILS_ENV=production DISABLE_DATABASE_ENVIRONMENT_CHECK=1 bin/rails db:drop",
    "rails db:migrate:reset",
    "rails db:schema:load",
    "rails db:structure:load",
    "rails db:purge",
    "rails db:truncate_all",
    "rails db:seed:replant",
    "rails db:drop:primary",
    "rails db:migrate VERSION=0",
    "heroku run rails db:drop -a myapp",
    "heroku pg:reset DATABASE_URL --confirm myapp",
    'kubectl exec deploy/web -n prod -- bash -lc "python manage.py flush --noinput"',
    'fly ssh console -C "bin/rails db:reset"',
    "python manage.py flush --noinput",
    "python3 manage.py flush",
    "./manage.py flush --no-input",
    "uv run manage.py flush --noinput",
    "python -m django flush --settings=app.settings",
    "django-admin flush",
    "python manage.py reset_db --noinput",
    "python manage.py migrate blog zero",
    "echo yes | python manage.py flush",
    "alembic downgrade base",
    "python -m alembic downgrade base",
    "flask db downgrade base",
    "dotnet ef database drop --force",
    "dotnet ef database update 0",
    "dropdb myapp_production",
    "dropdb --if-exists myapp",
    "sudo -u postgres dropdb myapp",
    "docker exec pg dropdb -U postgres myapp",
    "mysqladmin -u root -p drop myapp",
    "npx knex migrate:rollback --all",
    "mix ecto.reset",
    "mix ecto.drop",
    "MIX_ENV=prod mix ecto.rollback --all",
    "npx typeorm schema:drop -d src/data-source.ts",
    "npx mikro-orm schema:drop --run",
    "npx mikro-orm schema:fresh --run --seed",
    "npx mikro-orm migration:fresh",
    "npx sequelize-cli db:drop",
    "npx sequelize-cli db:migrate:undo:all",
    "npx drizzle-kit push --force",
    "php bin/console doctrine:database:drop --force",
    "php bin/console doctrine:schema:drop --force --full-database",
    "php bin/console d:d:d --force",
    "php bin/console doctrine:fixtures:load --no-interaction",
    "turso db destroy myapp --yes",
    "pscale database delete myapp --force",
    "neonctl branches reset main --parent",
    "neonctl projects delete abc123",
    "gcloud sql instances delete prod-db --quiet",
    "gcloud sql databases delete app --instance=prod-db",
    "aws rds delete-db-instance --db-instance-identifier prod --skip-final-snapshot",
    "aws dynamodb delete-table --table-name users",
    "firebase firestore:delete --all-collections --force",
    "firebase database:remove /users",
    "bq rm -r -f -d myproject:analytics",
    'psql "$DATABASE_URL" -c "TRUNCATE TABLE orders"',
    'mongosh mydb --eval "db.users.drop()"',
    'mongosh mydb --eval "db.users.deleteMany({})"',
    'mongo mydb --eval "db.dropDatabase()"',
    "redis-cli FLUSHALL",
    'pgcli -c "DROP DATABASE app"',
    'cockroach sql -e "DROP DATABASE app CASCADE"',
    'clickhouse-client -q "DROP TABLE events"',
    'wrangler d1 execute prod-db --command "DROP TABLE users"',
    'turso db shell myapp "DROP TABLE users"',
    'python3 -c "from app import db; db.drop_all()"',
    """python3 -c "import sqlite3; c=sqlite3.connect('app.db'); c.execute('DROP TABLE users')" """,
    """node -e "const {PrismaClient}=require('@prisma/client'); new PrismaClient().user.deleteMany()" """,
    'node -e "sequelize.sync({ force: true })"',
)
for c in DB_RESETS:
    judged("database reset: %s" % c.strip()[:70], c, "OP-007")

# the everyday database commands beside them, and the same task names as text, still pass
DB_EVERYDAY = (
    "php artisan migrate", "php artisan migrate --force", "php artisan migrate:status",
    "php artisan migrate:rollback", "php artisan db:seed", "php artisan help migrate:fresh",
    "php artisan migrate:fresh --help", "php artisan test",
    "npx prisma migrate dev --name init", "npx prisma migrate deploy", "npx prisma db push",
    "npx prisma generate", "npx prisma migrate status", "npm run dev", "npm install prisma",
    "rails db:migrate", "rails db:create", "rails db:setup", "rails db:prepare", "rails db:seed",
    "rails db:rollback STEP=1", "rails db:schema:dump", "rails db:test:prepare", "rails -T db",
    "mix ecto.migrate", "mix ecto.rollback",
    "python manage.py migrate", "python manage.py makemigrations", "python manage.py sqlflush",
    "python manage.py test", "python manage.py runserver", "python manage.py migrate blog 0003",
    "alembic upgrade head", "alembic downgrade -1", "flask db upgrade", "flask db downgrade",
    "dotnet ef database update", "dotnet ef migrations add Init",
    "dropdb --help", "createdb myapp", "mysqladmin status",
    "npx knex migrate:rollback", "npx knex migrate:latest", "npx sequelize-cli db:migrate",
    "npx sequelize-cli db:migrate:undo", "npx drizzle-kit push", "npx drizzle-kit generate",
    "npx mikro-orm schema:drop --dump", "php bin/console doctrine:database:drop",
    "php bin/console doctrine:schema:drop --dump-sql", "php bin/console doctrine:fixtures:load --append",
    "php bin/console doctrine:migrations:migrate",
    'grep -rn "migrate:fresh" .', "rg db:drop", 'git commit -m "docs: never run db:drop in production"',
    "git grep db:reset", 'echo "run php artisan migrate:fresh to reset"', "echo db:drop",
    "gcloud sql instances list", "gcloud sql instances describe prod-db", "aws rds describe-db-instances",
    "aws dynamodb list-tables", "turso db list", "neonctl branches list", "bq ls", 'bq query "SELECT 1"',
    "psql -c 'select 1' mydb", 'psql -c "DELETE FROM users WHERE id = 5" app',
    'mongosh mydb --eval "db.users.find()"', 'mongosh mydb --eval "db.users.deleteMany({age: 3})"',
    "redis-cli GET key", 'python3 -c "print(1)"', 'node -e "console.log(1)"',
    "docker compose up -d", "docker compose exec app php artisan migrate", "kubectl get pods",
    'ssh prod "cd app && bin/rails db:migrate"',
)
for c in DB_EVERYDAY:
    judged("everyday: %s" % c[:70], c, None)

# a command that names the test environment resets the test database, which is what a test run is for
for c in ("php artisan migrate:fresh --env=testing", "APP_ENV=testing php artisan migrate:fresh",
          "RAILS_ENV=test bin/rails db:drop db:create db:schema:load", "bin/rails db:reset -e test",
          "MIX_ENV=test mix ecto.reset"):
    judged("test database: %s" % c, c, None)
# ...but one quoted line that does both is not excused by its test half
judged("a remote line that resets test and production", 'ssh prod "RAILS_ENV=test rails db:reset && rails db:reset"', "OP-007")

# PowerShell: a program named by its Windows path is still that program, and what it hands to bash
# is walked (2026-09-27: `& C:\msys64\usr\bin\bash.exe -lc "... rm -rf \$X"` deleted most of C:)
judged("bash by its Windows path, behind the call operator", r'& C:\msys64\usr\bin\bash.exe -lc "rm -rf ~/Documents"',
       "OP-003", tool="PowerShell")
judged("quoted Windows path with a space", r'& "C:\Program Files\Git\bin\bash.exe" -lc "rm -rf ~/Documents"',
       "OP-003", tool="PowerShell")
judged("no call operator", r'C:\msys64\usr\bin\bash.exe -c "git push --force origin main"', "OP-004", tool="PowerShell")
judged("php by its Windows path", r"& C:\php\php.exe artisan migrate:fresh --force", "OP-007", tool="PowerShell")
judged("PowerShell fills in $X before bash runs; the backslash does not protect it",
       r'& C:\msys64\usr\bin\bash.exe -lc "export X=\$(mktemp -d); rm -rf \$X"', "OP-003", tool="PowerShell")
judged("single quotes leave $X to bash, as before", r"bash -lc 'ls $X'", None, tool="PowerShell")
judged("a harmless nested command", r'& C:\msys64\usr\bin\bash.exe -lc "ls -la"', None, tool="PowerShell")
judged("a program by its Windows path that does nothing destructive", r"& C:\tools\node.exe --version", None,
       tool="PowerShell")
judged("database reset typed into PowerShell", "php artisan migrate:fresh --seed", "OP-007", tool="PowerShell")

# the old true positives still hold
stop("force-push", "git push --force origin main", "OP-004")
stop("recursive delete of a home folder", "rm -rf ~/Documents", "OP-003")
allow("plain build command", "cargo check 2>&1 | tail -5")

shutil.rmtree(op_home, ignore_errors=True)
shutil.rmtree(work, ignore_errors=True)
print("\nfailures: %d" % len(fails))
sys.exit(1 if fails else 0)
