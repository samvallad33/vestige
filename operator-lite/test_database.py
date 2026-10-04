import importlib.util, os, sys, json, tempfile
home = tempfile.mkdtemp(prefix="dbcheck-", dir=os.path.expanduser("~"))
os.environ["HOME"] = home; os.environ["OPERATOR_HOME"] = os.path.join(home, ".operator")
os.environ.pop("OPERATOR_AGENT_SESSION", None)
spec = importlib.util.spec_from_file_location("gate", (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator-gate.py"))); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
cfg = g.load_config(); work = os.path.join(home, "work"); os.makedirs(work)

def rules(cmd, tool="Bash"):
    hits = []
    effs = g.analyze_powershell(cmd, work) if tool.lower() in g.POWERSHELL_TOOLS else g.analyze(cmd, work)
    for e in effs:
        hits += [h[0] for h in g.classify_effect(e, cfg, work)]
    return hits

STOP = """php artisan migrate:fresh --seed --force
php artisan migrate:fresh --seed --force --env=production
php artisan migrate:refresh
php artisan migrate:reset
php artisan db:wipe --force
APP_ENV=production php artisan migrate:fresh
./vendor/bin/sail artisan migrate:fresh --seed
sail art migrate:fresh
cd backend && php artisan migrate:fresh
yes | php artisan migrate:fresh
docker compose exec -T app php artisan migrate:fresh --force
docker compose exec app sh -c 'php artisan migrate:fresh --seed'
ssh deploy@prod "cd /var/www/app && php artisan migrate:fresh --force"
npx prisma migrate reset --force
npx prisma@6.2.1 migrate reset
pnpm dlx prisma migrate reset
yarn prisma migrate reset --skip-seed
bunx prisma migrate reset -f
pnpm exec prisma db push --force-reset
npx prisma db push --accept-data-loss
npx prisma --schema prisma/schema.prisma migrate reset
npm run db:reset
rails db:drop db:create db:migrate
bin/rails db:drop
bundle exec rails db:reset
bundle exec rake db:drop:all
RAILS_ENV=production DISABLE_DATABASE_ENVIRONMENT_CHECK=1 bin/rails db:drop
rails db:migrate:reset
rails db:schema:load
rails db:structure:load
rails db:purge
rails db:truncate_all
rails db:seed:replant
rails db:drop:primary
rails db:migrate VERSION=0
heroku run rails db:drop -a myapp
heroku pg:reset DATABASE_URL --confirm myapp
kubectl exec deploy/web -n prod -- bash -lc "python manage.py flush --noinput"
fly ssh console -C "bin/rails db:reset"
python manage.py flush --noinput
python3 manage.py flush
./manage.py flush --no-input
uv run manage.py flush --noinput
python -m django flush --settings=app.settings
django-admin flush
python manage.py reset_db --noinput
python manage.py migrate blog zero
echo yes | python manage.py flush
alembic downgrade base
python -m alembic downgrade base
flask db downgrade base
dotnet ef database drop --force
dotnet ef database update 0
dropdb myapp_production
dropdb --if-exists myapp
sudo -u postgres dropdb myapp
docker exec pg dropdb -U postgres myapp
mysqladmin -u root -p drop myapp
mysqladmin -f drop myapp
npx knex migrate:rollback --all
mix ecto.reset
mix ecto.drop
MIX_ENV=prod mix ecto.rollback --all
npx typeorm schema:drop -d src/data-source.ts
npx mikro-orm schema:drop --run
npx mikro-orm schema:fresh --run --seed
npx mikro-orm migration:fresh
npx sequelize-cli db:drop
npx sequelize-cli db:migrate:undo:all
npx drizzle-kit push --force
php bin/console doctrine:database:drop --force
php bin/console doctrine:schema:drop --force --full-database
php bin/console d:d:d --force
php bin/console doctrine:fixtures:load --no-interaction
turso db destroy myapp --yes
pscale database delete myapp --force
neonctl branches reset main --parent
neonctl projects delete abc123
gcloud sql instances delete prod-db --quiet
gcloud sql databases delete app --instance=prod-db
aws rds delete-db-instance --db-instance-identifier prod --skip-final-snapshot
aws dynamodb delete-table --table-name users
firebase firestore:delete --all-collections --force
firebase database:remove /users
bq rm -r -f -d myproject:analytics
psql -c 'DROP TABLE users' app
psql "$DATABASE_URL" -c "TRUNCATE TABLE orders"
mongosh mydb --eval "db.users.drop()"
mongosh mydb --eval "db.users.deleteMany({})"
mongosh mydb --eval "db.dropDatabase()"
mongo mydb --eval "db.dropDatabase()"
redis-cli FLUSHALL
pgcli -c "DROP DATABASE app"
cockroach sql -e "DROP DATABASE app CASCADE"
clickhouse-client -q "DROP TABLE events"
wrangler d1 execute prod-db --command "DROP TABLE users"
turso db shell myapp "DROP TABLE users"
python3 -c "from app import db; db.drop_all()"
python3 -c "import sqlite3; c=sqlite3.connect('app.db'); c.execute('DROP TABLE users')"
node -e "const {PrismaClient}=require('@prisma/client'); new PrismaClient().user.deleteMany()"
node -e "sequelize.sync({ force: true })"
python3 -c "import sqlite3; sqlite3.connect('a.db').execute('DELETE FROM users')" """.strip().split("\n")

ALLOW = """php artisan migrate
php artisan migrate --force
php artisan migrate:status
php artisan migrate:rollback
php artisan db:seed
php artisan help migrate:fresh
php artisan migrate:fresh --help
php artisan migrate:fresh --env=testing
APP_ENV=testing php artisan migrate:fresh
php artisan test
npx prisma migrate dev --name init
npx prisma migrate deploy
npx prisma db push
npx prisma generate
npx prisma migrate status
npm run dev
npm install prisma
rails db:migrate
rails db:create
rails db:setup
rails db:prepare
rails db:seed
rails db:rollback STEP=1
rails db:schema:dump
rails db:test:prepare
RAILS_ENV=test bin/rails db:drop db:create db:schema:load
bin/rails db:reset -e test
rails -T db
MIX_ENV=test mix ecto.reset
mix ecto.migrate
mix ecto.rollback
python manage.py migrate
python manage.py makemigrations
python manage.py sqlflush
python manage.py test
python manage.py runserver
python manage.py migrate blog 0003
alembic upgrade head
alembic downgrade -1
flask db upgrade
flask db downgrade
dotnet ef database update
dotnet ef migrations add Init
dropdb --help
createdb myapp
mysqladmin status
npx knex migrate:rollback
npx knex migrate:latest
npx sequelize-cli db:migrate
npx sequelize-cli db:migrate:undo
npx drizzle-kit push
npx drizzle-kit generate
npx mikro-orm schema:drop --dump
php bin/console doctrine:database:drop
php bin/console doctrine:schema:drop --dump-sql
php bin/console doctrine:fixtures:load --append
php bin/console doctrine:migrations:migrate
grep -rn "migrate:fresh" .
rg db:drop
git commit -m "docs: never run db:drop in production"
git grep db:reset
echo "run php artisan migrate:fresh to reset"
echo db:drop
cat docs/db:reset
gcloud sql instances list
gcloud sql instances describe prod-db
aws rds describe-db-instances
aws dynamodb list-tables
turso db list
neonctl branches list
firebase deploy
bq ls
bq query "SELECT 1"
psql -c 'select 1' mydb
psql -c "DELETE FROM users WHERE id = 5" app
mongosh mydb --eval "db.users.find()"
mongosh mydb --eval "db.users.deleteMany({age: 3})"
redis-cli GET key
python3 -c "print(1)"
python3 -c "import json; print(json.dumps({'a': 1}))"
node -e "console.log(1)"
docker compose up -d
docker compose exec app php artisan migrate
kubectl get pods
ssh prod "cd app && bin/rails db:migrate"
ls -la && cargo check""".strip().split("\n")

bad = 0
for c in STOP:
    r = rules(c)
    if "OP-007" not in r:
        bad += 1; print("MISS   %-75s %s" % (c[:75], r))
for c in ALLOW:
    r = rules(c)
    if "OP-007" in r:
        bad += 1; print("FALSE  %-75s %s" % (c[:75], r))
ps = [("php artisan migrate:fresh --seed", True), ("python manage.py flush --noinput", True), ("npx prisma migrate reset --force", True),
      ("php artisan migrate", False), ("Get-ChildItem", False)]
for c, want in ps:
    r = rules(c, "PowerShell")
    if ("OP-007" in r) != want:
        bad += 1; print("PS %s %-60s %s" % ("MISS " if want else "FALSE", c, r))
print("stop cases %d, allow cases %d, powershell %d, wrong %d" % (len(STOP), len(ALLOW), len(ps), bad))
import shutil; shutil.rmtree(home, ignore_errors=True)
sys.exit(1 if bad else 0)
