-- Vestige Pro waitlist capture.
--
-- One row per signup from the public /dashboard/waitlist form. The form POSTs
-- through the `waitlist-join` Edge Function (service-role), NOT directly to this
-- table, so Row-Level Security stays closed to anon clients while the function
-- can insert. We keep the table minimal and queryable for launch-day export.

create table if not exists public.waitlist (
    id          uuid primary key default gen_random_uuid(),
    email       text not null,
    name        text,
    plan        text,                       -- 'solo' | 'team' (free-form, not enforced)
    priority    text,                       -- 'sync' | 'team-memory' | 'postgres' | 'support'
    notes       text,
    source      text default 'vestige-pro-waitlist',
    user_agent  text,
    -- Lowercased email for case-insensitive dedup. Generated so callers can't
    -- desync it from `email`.
    email_norm  text generated always as (lower(trim(email))) stored,
    created_at  timestamptz not null default now()
);

-- One signup per email address. ON CONFLICT in the function turns a re-submit
-- into a friendly "already on the list" instead of a duplicate row or an error.
create unique index if not exists waitlist_email_norm_key
    on public.waitlist (email_norm);

-- Newest-first export / dashboards.
create index if not exists waitlist_created_at_idx
    on public.waitlist (created_at desc);

-- RLS ON, no anon policies: the anon/public key can do NOTHING here directly.
-- Only the service-role key used inside the Edge Function bypasses RLS to insert.
alter table public.waitlist enable row level security;

comment on table public.waitlist is
    'Vestige Pro launch waitlist. Inserted only via the waitlist-join Edge Function (service-role). RLS blocks all anon access.';
