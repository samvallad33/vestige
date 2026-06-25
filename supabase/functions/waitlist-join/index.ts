// Vestige Pro waitlist capture — Supabase Edge Function (Deno).
//
// The public /dashboard/waitlist form POSTs JSON here. This function:
//   1. Handles CORS preflight (the form is served from a different origin).
//   2. Validates the email and rejects honeypot/bot submissions.
//   3. Inserts one row per email into public.waitlist (dedup on re-submit).
//   4. Optionally sends a confirmation email via Resend.
//
// Secrets (set with `supabase secrets set`, NEVER committed):
//   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  — auto-injected by Supabase
//   RESEND_API_KEY                           — optional; if unset, skip email
//   WAITLIST_FROM_EMAIL                      — optional; defaults below
//   WAITLIST_ALLOWED_ORIGIN                  — optional CORS allowlist (comma-sep)

import { createClient } from "jsr:@supabase/supabase-js@2";

const DEFAULT_FROM = "Vestige <waitlist@vestige.dev>";

function corsHeaders(origin: string | null): HeadersInit {
  const allow = (Deno.env.get("WAITLIST_ALLOWED_ORIGIN") ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  // If an allowlist is configured and the origin isn't on it, fall back to the
  // first allowed origin (so browsers see a definite, non-wildcard value).
  const allowedOrigin =
    allow.length === 0 ? "*" : (origin && allow.includes(origin) ? origin : allow[0]);
  return {
    "Access-Control-Allow-Origin": allowedOrigin,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Content-Type": "application/json",
  };
}

interface WaitlistPayload {
  name?: string;
  email?: string;
  plan?: string;
  priority?: string;
  notes?: string;
  source?: string;
  companySite?: string; // honeypot — must be empty
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

async function sendConfirmation(email: string, name: string) {
  const key = Deno.env.get("RESEND_API_KEY");
  if (!key) return; // email is optional; capture still succeeds without it
  const from = Deno.env.get("WAITLIST_FROM_EMAIL") ?? DEFAULT_FROM;
  const greeting = name ? `Hi ${name},` : "Hi,";
  try {
    await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from,
        to: email,
        subject: "You're on the Vestige Pro list",
        text:
          `${greeting}\n\nYou're on the Vestige Pro early-access list. ` +
          `We'll email you before launch with your invite.\n\n— The Vestige team`,
      }),
    });
  } catch (_err) {
    // Never fail the signup because the email provider hiccuped.
  }
}

Deno.serve(async (req) => {
  const origin = req.headers.get("origin");
  const headers = corsHeaders(origin);

  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers });
  }
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ ok: false, error: "method_not_allowed" }), {
      status: 405,
      headers,
    });
  }

  let body: WaitlistPayload;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ ok: false, error: "invalid_json" }), {
      status: 400,
      headers,
    });
  }

  // Honeypot: real users never fill this hidden field. Pretend success so bots
  // get no signal, but write nothing.
  if (body.companySite && body.companySite.trim()) {
    return new Response(JSON.stringify({ ok: true, deduped: false }), { status: 200, headers });
  }

  const email = (body.email ?? "").trim();
  if (!EMAIL_RE.test(email)) {
    return new Response(JSON.stringify({ ok: false, error: "invalid_email" }), {
      status: 400,
      headers,
    });
  }

  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  );

  // Insert; on email conflict, treat as already-joined (idempotent, friendly).
  const row = {
    email,
    name: (body.name ?? "").trim() || null,
    plan: (body.plan ?? "").trim() || null,
    priority: (body.priority ?? "").trim() || null,
    notes: (body.notes ?? "").trim() || null,
    source: (body.source ?? "vestige-pro-waitlist").trim(),
    user_agent: req.headers.get("user-agent")?.slice(0, 400) ?? null,
  };

  const { error } = await supabase.from("waitlist").insert(row);

  if (error) {
    // 23505 = unique_violation on email_norm → already on the list.
    if (error.code === "23505") {
      return new Response(JSON.stringify({ ok: true, deduped: true }), { status: 200, headers });
    }
    console.error("waitlist insert failed:", error);
    return new Response(JSON.stringify({ ok: false, error: "storage_failed" }), {
      status: 500,
      headers,
    });
  }

  // Fire-and-forget confirmation; the signup is already committed.
  await sendConfirmation(email, row.name ?? "");

  return new Response(JSON.stringify({ ok: true, deduped: false }), { status: 200, headers });
});
