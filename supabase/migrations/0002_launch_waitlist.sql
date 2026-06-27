-- Vestige LAUNCH waitlist (/dashboard/launch) — table + referral loop.
--
-- The browser inserts directly with the public anon key. The table is
-- INSERT-ONLY to anon via RLS, and every read (own referral code, referral
-- count, total count) goes through a `security definer` RPC that returns a
-- single value, so the anon key never gets SELECT on the table.
--
-- This is the source of truth for the launch page. The older service-role
-- Edge Function schema lives in supabase/legacy/ and is retired.
--
-- Apply with: supabase db push   (or paste into the SQL editor — see
-- docs/launch/waitlist-setup.md, which mirrors this file exactly).

create table if not exists public.waitlist (
  id            bigint generated always as identity primary key,
  email         text not null,
  source        text,
  referrer      text,                       -- document.referrer (analytics only)
  referral_code text not null,              -- THIS user's own share code
  referred_by   text,                       -- the code that brought them in
  created_at    timestamptz not null default now()
);

create unique index if not exists waitlist_email_key
  on public.waitlist (lower(email));

create unique index if not exists waitlist_referral_code_key
  on public.waitlist (referral_code);
create index if not exists waitlist_referred_by_idx
  on public.waitlist (referred_by);

alter table public.waitlist enable row level security;

drop policy if exists "anon can insert" on public.waitlist;
create policy "anon can insert"
  on public.waitlist
  for insert
  to anon
  with check (true);

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
    exit when not exists (select 1 from public.waitlist where referral_code = code);
  end loop;
  return code;
end;
$$;

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

  select w.referral_code into v_code
    from public.waitlist w
   where lower(w.email) = v_email
   limit 1;

  if v_code is null then
    v_code := public.gen_referral_code();
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

create or replace function public.referral_count(p_code text)
returns integer
language sql
security definer
set search_path = public
as $$
  select count(*)::int from public.waitlist where referred_by = p_code;
$$;

create or replace function public.waitlist_count()
returns integer
language sql
security definer
set search_path = public
as $$
  select count(*)::int from public.waitlist;
$$;

grant execute on function public.join_waitlist(text, text, text) to anon;
grant execute on function public.referral_count(text)            to anon;
grant execute on function public.waitlist_count()                to anon;
