# Vestige launch waitlist — go-live runbook (Supabase + referral loop)

The `/dashboard/launch` page captures signups **and turns every signup into a
sharer**: each person gets a personal referral link, sees how many friends they
brought in, and (optionally) gets a confirmation email that asks them to share.
This is the loop that actually goes viral for dev launches — a shareable artifact
and a "friends joined from your link" counter, **not** a fake queue position.

Everything below is copy-paste. Steps marked **[you]** need your Supabase login;
everything else is already wired in the code.

---

## Architecture (why it's safe)

The browser writes signups **directly** to Supabase using the **public anon key**.
The `waitlist` table has **insert-only** Row-Level Security, so the anon key can
add a row but can **never read or modify** the table. Everything the page needs
to *read* (your referral code, your referral count) goes through
`security definer` RPC functions that return only a single value — the anon key
still never gets SELECT on the table. No Edge Function is required for capture.

---

## 1. Create a Supabase project **[you]**

<https://supabase.com> → **New project**. Pick a name (e.g. `vestige-waitlist`),
a strong DB password (save it), the region closest to you. Wait ~2 min for it to
provision.

## 2. Paste this SQL **[you]** (SQL Editor → New query → **Run**)

This one block creates the table, the insert-only RLS, the referral code
generator, and the three RPCs the page calls. Safe to re-run (idempotent).

```sql
-- ── Vestige waitlist: table + referral loop ──────────────────────────────────
create table if not exists public.waitlist (
  id            bigint generated always as identity primary key,
  email         text not null,
  source        text,
  referrer      text,                       -- document.referrer (analytics only)
  referral_code text not null,              -- THIS user's own share code
  referred_by   text,                       -- the code that brought them in
  created_at    timestamptz not null default now()
);

-- one row per email (idempotent signups; duplicates resolve to the existing row)
create unique index if not exists waitlist_email_key
  on public.waitlist (lower(email));

-- fast lookups by referral code (own code + attribution counts)
create unique index if not exists waitlist_referral_code_key
  on public.waitlist (referral_code);
create index if not exists waitlist_referred_by_idx
  on public.waitlist (referred_by);

-- lock the table down, then allow ONLY anonymous inserts.
-- (Reads happen exclusively through the security-definer RPCs below, so the
--  anon role never needs — and never gets — SELECT on this table.)
alter table public.waitlist enable row level security;

drop policy if exists "anon can insert" on public.waitlist;
create policy "anon can insert"
  on public.waitlist
  for insert
  to anon
  with check (true);

-- short, unambiguous, URL-safe referral code (no 0/O/1/I/l to avoid confusion)
create or replace function public.gen_referral_code()
returns text
language plpgsql
as $$
declare
  alphabet constant text := 'abcdefghjkmnpqrstuvwxyz23456789';
  code text;
  i int;
begin
  loop
    code := '';
    for i in 1..7 loop
      code := code || substr(alphabet, 1 + floor(random() * length(alphabet))::int, 1);
    end loop;
    -- guarantee uniqueness against existing codes
    exit when not exists (select 1 from public.waitlist where referral_code = code);
  end loop;
  return code;
end;
$$;

-- The page's single entry point. Inserts (or finds existing) and returns the
-- caller's own referral code + how many people they've referred so far.
-- security definer => runs with table owner rights, so anon never reads directly.
create or replace function public.join_waitlist(
  p_email       text,
  p_referred_by text default null,
  p_referrer    text default null
)
returns table (referral_code text, referrals int, duplicate boolean)
language plpgsql
security definer
set search_path = public
as $$
declare
  v_email text := lower(trim(p_email));
  v_code  text;
  v_ref   text := nullif(trim(coalesce(p_referred_by, '')), '');
  v_dup   boolean := false;
begin
  if v_email !~ '^[^@\s]+@[^@\s]+\.[^@\s]+$' then
    raise exception 'invalid email';
  end if;

  -- already signed up? return their existing code (idempotent, no double-count).
  select w.referral_code into v_code
    from public.waitlist w
   where lower(w.email) = v_email
   limit 1;

  if v_code is null then
    v_code := public.gen_referral_code();
    -- never let someone refer themselves; ignore a self/unknown code gracefully.
    insert into public.waitlist (email, source, referrer, referral_code, referred_by)
    values (v_email, 'vestige-launch', p_referrer, v_code,
            case when v_ref = v_code then null else v_ref end);
  else
    v_dup := true;
  end if;

  return query
    select v_code,
           (select count(*)::int from public.waitlist where referred_by = v_code),
           v_dup;
end;
$$;

-- Live referral count for a given code (drives the "N friends joined" counter).
create or replace function public.referral_count(p_code text)
returns integer
language sql
security definer
set search_path = public
as $$
  select count(*)::int from public.waitlist where referred_by = p_code;
$$;

-- Total signups (drives the social-proof count once it's >= 25).
create or replace function public.waitlist_count()
returns integer
language sql
security definer
set search_path = public
as $$
  select count(*)::int from public.waitlist;
$$;

-- expose ONLY these RPCs to the anonymous browser role
grant execute on function public.join_waitlist(text, text, text) to anon;
grant execute on function public.referral_count(text)            to anon;
grant execute on function public.waitlist_count()                to anon;
```

## 3. Set the env vars **[you]**

Supabase → **Project Settings → API**. Copy **Project URL** and the **anon
public** key into `apps/dashboard/.env` (already created for you with the keys
blank — just paste):

```
VITE_SUPABASE_URL=https://<project-ref>.supabase.co
VITE_SUPABASE_ANON_KEY=<anon public key>
```

Restart the dev server (Vite only reads env at boot). For the deployed site, set
the same two vars in your build environment and rebuild
`pnpm --filter @vestige/dashboard build`.

## 4. Verify

- Open `/dashboard/launch`, enter an email, submit → success state shows your
  **personal share link** + **Share on X** / **Copy** buttons.
- Supabase → Table editor → `waitlist` → your row is there, with a `referral_code`.
- Open the page with `?ref=<that code>` in a private window, sign up a 2nd email,
  then reload the first link — the "N friends joined" counter goes up.
- The total count appears on the page once there are ≥ 25 signups.

---

## 5. (Recommended) Email each new signup a share link — Resend + Edge Function

This is the launch-day multiplier: the instant someone joins, they get an email
with **their** share link and a one-tap share button. That email is what converts
a passive signup into a sharer. ~10 minutes, free tier.

1. **Get a Resend API key** **[you]** — <https://resend.com> → API Keys → Create.
   (Free tier = 3,000 emails/mo, 100/day — plenty for a waitlist.) To send from
   your own domain, verify it under Resend → Domains; otherwise use the shared
   `onboarding@resend.dev` sender for testing.

2. **Deploy the Edge Function** (code is in
   `supabase/functions/waitlist-welcome/index.ts`). With the Supabase CLI:

   ```sh
   supabase login                              # [you] one-time
   supabase link --project-ref <project-ref>   # [you]
   supabase secrets set RESEND_API_KEY=<your key> \
     WAITLIST_FROM="Vestige <onboarding@resend.dev>" \
     WAITLIST_PUBLIC_URL="https://samvallad33.github.io/vestige/dashboard/launch"
   supabase functions deploy waitlist-welcome --no-verify-jwt
   ```

3. **Trigger it on every new signup** **[you]** — Supabase → Database →
   **Webhooks** → Create a new hook:
   - Table: `waitlist`, Events: **Insert**
   - Type: **Supabase Edge Functions** → `waitlist-welcome`
   - (HTTP headers/payload defaults are fine.)

   Now every insert fires the function, which emails the new signup their share
   link. No email = signups still captured silently; you lose only the auto-share
   nudge.

> Prefer zero email infra? Skip section 5 entirely. The in-page share link +
> referral counter still work — the email just adds a second, asynchronous
> share prompt that reaches people after they close the tab.

---

## Exporting / emailing your list on launch day

Supabase → Table editor → `waitlist` → **Export to CSV**. Or in the SQL editor:

```sql
-- everyone, newest first
select email, referral_code, referred_by, created_at
from public.waitlist order by created_at desc;

-- your top sharers (reward these people on launch day)
select referred_by as code, count(*) as brought_in
from public.waitlist
where referred_by is not null
group by referred_by order by brought_in desc limit 25;
```

That CSV is the asset you email when July 14 arrives. The "top sharers" query is
who you publicly thank / give early access first — that recognition is itself a
share trigger.

## Notes

- The page still works with **no env set** (local-only capture) for demos.
- Duplicate emails are idempotent: the same person always gets the **same**
  referral code back, so re-submitting never double-counts.
- Self-referral is blocked server-side; an unknown `?ref=` code is ignored
  gracefully (the signup still succeeds).
- A honeypot field + email-format validation block basic spam. Because inserts
  go through `join_waitlist()`, you can later add rate limiting there if needed.
