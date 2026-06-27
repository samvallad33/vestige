# Legacy waitlist backend (archived)

These files backed the **old** `/dashboard/waitlist` page (Canvas2D grid +
support bot), which POSTed signups to the `waitlist-join` Edge Function using a
service-role insert.

The **live launch page** is `/dashboard/launch`. It uses a different, simpler
architecture: direct browser inserts with the public anon key + insert-only RLS,
plus `security definer` RPCs for the referral loop. See:

- Schema + RPCs: `supabase/migrations/0002_launch_waitlist.sql`
- Go-live runbook: `docs/launch/waitlist-setup.md`
- Welcome/share email: `supabase/functions/waitlist-welcome/`

Both schemas declared a table literally named `waitlist` with **incompatible**
shapes (this one is `uuid` PK with `name/plan/notes`; the launch one is `bigint`
PK with `referral_code/referred_by`). They cannot coexist in the same database
under that name. The launch schema is the source of truth as of 2026-06-26.

Kept for reference / in case the old page is ever revived (it would need its own
table name). Nothing in the app references `waitlist-join` anymore.
