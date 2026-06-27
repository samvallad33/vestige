// Vestige launch waitlist — welcome + SHARE email (Supabase Edge Function, Deno).
//
// Fired by a Supabase Database Webhook on INSERT into public.waitlist. It emails
// the new signup a confirmation that contains THEIR personal referral link and a
// one-tap "Share on X" button. That email is the launch-day multiplier: it
// reaches people after they've closed the tab and nudges them to share, which is
// what actually compounds a dev-launch waitlist.
//
// Secrets (set with `supabase secrets set`, NEVER committed):
//   RESEND_API_KEY        — required to actually send (no key => no-op, 200 OK)
//   WAITLIST_FROM         — sender, e.g. "Vestige <onboarding@resend.dev>"
//                           (use a verified domain in production)
//   WAITLIST_PUBLIC_URL   — the public launch page, e.g.
//                           "https://samvallad33.github.io/vestige/dashboard/launch"
//   WAITLIST_WEBHOOK_SECRET — optional shared secret; if set, the webhook must
//                           send it as the `x-webhook-secret` header.
//
// Deploy:  supabase functions deploy waitlist-welcome --no-verify-jwt
// Wire up: Supabase → Database → Webhooks → on INSERT of public.waitlist →
//          Supabase Edge Functions → waitlist-welcome.

const DEFAULT_FROM = "Vestige <onboarding@resend.dev>";
const DEFAULT_PUBLIC_URL = "https://samvallad33.github.io/vestige/dashboard/launch";

interface WebhookPayload {
  type?: string; // 'INSERT'
  record?: {
    email?: string;
    referral_code?: string;
  };
}

function shareUrl(base: string, code: string): string {
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}ref=${encodeURIComponent(code)}`;
}

function buildEmail(link: string, xIntent: string) {
  const text =
    `You're on the Vestige waitlist. 🧠\n\n` +
    `Vestige is local-first memory for AI coding agents — rendered as a living ` +
    `WebGPU particle brain you can watch remember, forget, and reform. ` +
    `Launching July 14, 2026.\n\n` +
    `Want in earlier? Every friend who joins from your personal link moves you ` +
    `up the list:\n\n${link}\n\n` +
    `Share it on X: ${xIntent}\n\n` +
    `— Sam (building Vestige solo)`;

  const html = `<!doctype html>
<html>
  <body style="margin:0;background:#02030a;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#eaf3ff;">
    <div style="max-width:520px;margin:0 auto;padding:40px 28px;">
      <div style="font-weight:800;letter-spacing:.34em;font-size:13px;color:#8fb6ff;margin-bottom:24px;">VESTIGE</div>
      <h1 style="font-size:22px;line-height:1.3;margin:0 0 14px;">You're on the waitlist. 🧠</h1>
      <p style="font-size:15px;line-height:1.6;color:#b9cdec;margin:0 0 18px;">
        Vestige is local-first memory for AI coding agents — rendered as a living
        WebGPU particle brain you can watch remember, forget, and reform.
        <strong style="color:#8fe9ff;">Launching July 14, 2026.</strong>
      </p>
      <p style="font-size:15px;line-height:1.6;color:#b9cdec;margin:0 0 10px;">
        Want in earlier? Every friend who joins from your personal link moves you up the list:
      </p>
      <div style="background:#0a1020;border:1px solid rgba(140,190,255,.28);border-radius:12px;padding:14px 16px;margin:0 0 20px;font-size:13px;color:#bfe0ff;word-break:break-all;">
        ${link}
      </div>
      <a href="${xIntent}" style="display:inline-block;background:#f5f8ff;color:#0a0f1a;font-weight:700;font-size:15px;text-decoration:none;padding:12px 22px;border-radius:12px;">
        Share on X →
      </a>
      <p style="font-size:13px;line-height:1.6;color:#7d93b8;margin:28px 0 0;">
        — Sam, building Vestige solo. Reply any time.
      </p>
    </div>
  </body>
</html>`;

  return { text, html };
}

Deno.serve(async (req) => {
  // optional shared-secret check
  const expected = Deno.env.get("WAITLIST_WEBHOOK_SECRET");
  if (expected && req.headers.get("x-webhook-secret") !== expected) {
    return new Response(JSON.stringify({ ok: false, error: "unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  let payload: WebhookPayload;
  try {
    payload = await req.json();
  } catch {
    return new Response(JSON.stringify({ ok: false, error: "invalid_json" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const email = payload.record?.email?.trim();
  const code = payload.record?.referral_code?.trim();
  if (!email || !code) {
    // Nothing to send; ack so the webhook doesn't retry forever.
    return new Response(JSON.stringify({ ok: true, skipped: "no email/code" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  const key = Deno.env.get("RESEND_API_KEY");
  if (!key) {
    // Email is optional infra — capture already succeeded. Ack with a note.
    return new Response(JSON.stringify({ ok: true, skipped: "no RESEND_API_KEY" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  const from = Deno.env.get("WAITLIST_FROM") ?? DEFAULT_FROM;
  const base = Deno.env.get("WAITLIST_PUBLIC_URL") ?? DEFAULT_PUBLIC_URL;
  const link = shareUrl(base, code);
  const xText =
    "I just joined the Vestige waitlist — local-first memory for AI agents, " +
    "rendered as a live WebGPU particle brain. Launching July 14:";
  const xIntent =
    `https://twitter.com/intent/tweet?text=${encodeURIComponent(xText)}` +
    `&url=${encodeURIComponent(link)}`;
  const { text, html } = buildEmail(link, xIntent);

  try {
    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from,
        to: email,
        subject: "You're on the Vestige waitlist — here's your invite link",
        text,
        html,
      }),
    });
    if (!res.ok) {
      const detail = await res.text();
      console.error("resend send failed:", res.status, detail);
      // Ack anyway: the signup is committed; we don't want infinite retries.
      return new Response(JSON.stringify({ ok: false, error: "send_failed" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
  } catch (err) {
    console.error("resend send threw:", err);
    return new Response(JSON.stringify({ ok: false, error: "send_threw" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(JSON.stringify({ ok: true, sent: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
});
