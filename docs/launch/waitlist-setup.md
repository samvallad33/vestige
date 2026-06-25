# Waitlist capture — setup (10 minutes)

The public waitlist form (`/dashboard/waitlist`) is already built. It POSTs JSON
to `VITE_WAITLIST_ENDPOINT`. This guide stands up the storage backend: a
Supabase table + Edge Function that stores every signup and (optionally) emails a
confirmation.

Owned data, queryable Postgres, free tier handles launch-day volume.

## 1. Create the project

1. Create a project at <https://supabase.com> (free tier is fine for launch).
2. Grab the **Project Ref** (Settings → General) and the
   **service-role key** (Settings → API). The service-role key is a secret —
   never commit it or expose it client-side.

## 2. Deploy the schema + function

```sh
supabase link --project-ref <your-project-ref>
supabase db push                              # applies supabase/migrations/0001_waitlist.sql
supabase functions deploy waitlist-join       # deploys supabase/functions/waitlist-join
```

## 3. Configure function secrets

`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are injected automatically.
Email confirmation is optional — set these to enable it (reuses Resend):

```sh
supabase secrets set RESEND_API_KEY=<your-resend-key>
supabase secrets set WAITLIST_FROM_EMAIL="Vestige <waitlist@yourdomain>"
# Optional: lock CORS to your sites (comma-separated). Omit for "*".
supabase secrets set WAITLIST_ALLOWED_ORIGIN="https://samvallad33.github.io,https://vestige.dev"
```

## 4. Point the form at the function

The function URL looks like:
`https://<project-ref>.functions.supabase.co/waitlist-join`

Set it at build time for the dashboard:

```sh
# apps/dashboard/.env  (or your Pages build env)
VITE_WAITLIST_ENDPOINT=https://<project-ref>.functions.supabase.co/waitlist-join
```

Rebuild the dashboard (`pnpm --filter @vestige/dashboard build`) and redeploy.

## 5. Verify

```sh
curl -X POST https://<project-ref>.functions.supabase.co/waitlist-join \
  -H 'content-type: application/json' \
  -d '{"email":"you@example.com","name":"You","plan":"solo","priority":"sync"}'
# -> {"ok":true,"deduped":false}

# Re-run the same command:
# -> {"ok":true,"deduped":true}   (idempotent; no duplicate row)
```

## 6. Export the list on launch day

In the Supabase SQL editor:

```sql
select email, name, plan, priority, notes, created_at
from public.waitlist
order by created_at desc;
```

Or download a CSV from the Table Editor. That list is the asset that converts
to paid over the weeks after the spike.

## Notes

- **RLS is on, anon can do nothing.** Only the service-role key inside the
  function can insert. The public anon key (if ever used) has zero table access.
- **Honeypot + email validation** run both client-side (in the form) and
  server-side (in the function).
- **Dedup** is enforced by a unique index on the lowercased email; a re-submit
  returns `{ok:true,deduped:true}` instead of erroring or duplicating.
