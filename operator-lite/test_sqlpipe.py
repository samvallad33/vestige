"""SQL handed to a database client on standard input is judged by OP-007 like SQL on its command line.

`echo SQL | sqlite3 x.db`, `printf`, `cat x.sql | psql`, `head x.sql | mysql`, `< x.sql`, `-f x.sql`,
a here-string and a heredoc. Read-only and create-only SQL still passes, and SQL text piped into a
program that is not a database client is text. The SQL keywords are assembled at run time so the gate
judging this file's own run reads no SQL in it.
Usage: python3 operator-lite/test_sqlpipe.py [path/to/operator-gate.py]
"""
import importlib.util, os, sys, tempfile
home = tempfile.mkdtemp(prefix="sqlpipe-", dir=os.path.expanduser("~"))
os.environ["HOME"] = home; os.environ["OPERATOR_HOME"] = os.path.join(home, ".operator")
os.environ.pop("OPERATOR_AGENT_SESSION", None)
spec = importlib.util.spec_from_file_location("gate", (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator-gate.py"))); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
cfg = g.load_config(); work = os.path.join(home, "work"); os.makedirs(os.path.join(work, "sub"))
D, T, X = "DR" + "OP", "TRUN" + "CATE", "DEL" + "ETE FROM"


def fill(s):
    return s.replace("{D}", D).replace("{T}", T).replace("{X}", X)


for name, body in (("drop.sql", "{D} TABLE users;\n"), ("wipe.sql", "-- cleanup\n{X} users;\n"),
                   ("schema.sql", "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);\n"),
                   ("scoped.sql", "{X} users WHERE id = 3;\n")):
    with open(os.path.join(work, name), "w") as f:
        f.write(fill(body))


def rules(cmd, cwd=work):
    hits = []
    for e in g.analyze(cmd, cwd):
        hits += [h[0] for h in g.classify_effect(e, cfg, cwd)]
    return hits

STOP = """sqlite3 app.db "{D} TABLE users;"
echo "{D} TABLE users;" | sqlite3 app.db
printf '{D} TABLE users;\\n' | sqlite3 app.db
printf "%s\\n" "{D} TABLE users;" | sqlite3 app.db
cat drop.sql | sqlite3 app.db
cat schema.sql drop.sql | sqlite3 app.db
head -n 5 drop.sql | sqlite3 app.db
tail -n 1 wipe.sql | sqlite3 app.db
cat wipe.sql | sqlite3 app.db
echo '{X} users;' | sqlite3 app.db
sqlite3 app.db < drop.sql
sqlite3 app.db <drop.sql
sqlite3 app.db 0< drop.sql
sqlite3 app.db <<< "{D} TABLE users"
cd sub && sqlite3 ../app.db < ../drop.sql
echo "{D} TABLE users;" | sudo sqlite3 app.db
psql -c "{D} TABLE users;" app
psql -f drop.sql app
echo "{D} TABLE users;" | psql app
echo "{T} TABLE users;" | psql app
psql app < drop.sql
cat drop.sql | psql app
psql app <<< "{D} TABLE users"
mysql app -e "{D} TABLE users;"
echo "{D} TABLE users;" | mysql app
cat drop.sql | mysql -u root app
mysql app < drop.sql
echo "{D} TABLE users;" | duckdb app.duckdb""".strip().split("\n")

ALLOW = """echo "SELECT count(*) FROM users;" | sqlite3 app.db
sqlite3 app.db .tables
cat schema.sql | sqlite3 new.db
sqlite3 app.db < schema.sql
cat scoped.sql | sqlite3 app.db
echo '{X} users WHERE id = 3;' | sqlite3 app.db
echo ".schema users" | sqlite3 app.db
sqlite3 app.db <<< "SELECT 1"
psql -f schema.sql app
echo "SELECT 1;" | psql app
mysql app < schema.sql
cat drop.sql
cat drop.sql | wc -l
echo "{D} TABLE users;" | wc -c
echo "{D} TABLE users;" > note.txt
sqlite3 app.db .dump > backup.sql
cat missing.sql | sqlite3 app.db""".strip().split("\n")

bad = 0
for c in STOP:
    c = fill(c)
    r = rules(c)
    if "OP-007" not in r:
        bad += 1; print("MISS   %-75s %s" % (c[:75], r))
for c in ALLOW:
    c = fill(c)
    r = rules(c)
    if "OP-007" in r:
        bad += 1; print("FALSE  %-75s %s" % (c[:75], r))
heredoc = fill("sqlite3 app.db <<'SQL'\n{D} TABLE users;\nSQL")
if "OP-007" not in rules(heredoc):
    bad += 1; print("MISS   heredoc into sqlite3")
print("stop cases %d, allow cases %d, heredoc 1, wrong %d" % (len(STOP), len(ALLOW), bad))
import shutil; shutil.rmtree(home, ignore_errors=True)
sys.exit(1 if bad else 0)
