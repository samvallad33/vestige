// Waitlist backend — direct Supabase calls from the browser using the public
// anon key. The `waitlist` table is INSERT-ONLY to anon (RLS), and every read
// (your own referral code, your referral count) goes through `security definer`
// RPCs that return a single value, so the public anon key never reads the table.
// See docs/launch/waitlist-setup.md for the SQL + go-live runbook.
//
// Env (set in apps/dashboard/.env, NOT committed):
//   VITE_SUPABASE_URL=https://<project-ref>.supabase.co
//   VITE_SUPABASE_ANON_KEY=<anon public key>
//
// If the env is unset, joinWaitlist returns { ok:false, reason:'unconfigured' }
// so the page can fall back gracefully (local capture) during the demo.

import { createClient, type SupabaseClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

let client: SupabaseClient | null = null;
export const waitlistConfigured = Boolean(url && anonKey);

function getClient(): SupabaseClient | null {
	if (!waitlistConfigured) return null;
	if (!client) {
		client = createClient(url as string, anonKey as string, {
			auth: { persistSession: false }
		});
	}
	return client;
}

export type JoinResult =
	| { ok: true; duplicate: boolean; referralCode: string; referrals: number }
	| { ok: false; reason: 'unconfigured' | 'invalid' | 'error'; message?: string };

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const CODE_RE = /^[a-z2-9]{4,16}$/; // matches gen_referral_code() alphabet

/** Read the `?ref=` referral code from the current URL, sanitized. */
export function getReferralCodeFromUrl(): string | undefined {
	if (typeof window === 'undefined') return undefined;
	try {
		const raw = new URLSearchParams(window.location.search).get('ref');
		const code = raw?.trim().toLowerCase();
		return code && CODE_RE.test(code) ? code : undefined;
	} catch {
		return undefined;
	}
}

/** Build the shareable link for a given referral code (absolute, deploy-aware). */
export function buildShareUrl(referralCode: string): string {
	// Prefer the public production URL so the link works when shared off-device;
	// fall back to the current origin+path for local/preview testing.
	const publicBase = (import.meta.env.VITE_PUBLIC_LAUNCH_URL as string | undefined)?.trim();
	const base =
		publicBase ||
		(typeof window !== 'undefined'
			? `${window.location.origin}${window.location.pathname}`
			: 'https://samvallad33.github.io/vestige/dashboard/launch');
	const sep = base.includes('?') ? '&' : '?';
	return `${base}${sep}ref=${encodeURIComponent(referralCode)}`;
}

/**
 * Join the waitlist. Calls the `join_waitlist` RPC which inserts (or finds an
 * existing row) and returns the caller's own referral code + how many people
 * they've referred. Idempotent on duplicate email (same code returned, no
 * double count).
 */
export async function joinWaitlist(
	email: string,
	options: { referredBy?: string; referrer?: string } = {}
): Promise<JoinResult> {
	const clean = email.trim().toLowerCase();
	if (!EMAIL_RE.test(clean)) return { ok: false, reason: 'invalid' };

	const sb = getClient();
	if (!sb) return { ok: false, reason: 'unconfigured' };

	const { data, error } = await sb.rpc('join_waitlist', {
		p_email: clean,
		p_referred_by: options.referredBy ?? null,
		p_referrer: options.referrer ?? null
	});

	if (error) {
		// The RPC raises on a malformed email; surface that as 'invalid'.
		if (/invalid email/i.test(error.message)) return { ok: false, reason: 'invalid' };
		return { ok: false, reason: 'error', message: error.message };
	}

	// RPC returns a single-row table → supabase-js gives an array of one object.
	const row = Array.isArray(data) ? data[0] : data;
	const referralCode = (row?.referral_code as string | undefined) ?? '';
	const referrals = (row?.referrals as number | undefined) ?? 0;
	const duplicate = Boolean(row?.duplicate);

	if (!referralCode) {
		// Defensive: RPC succeeded but returned nothing usable.
		return { ok: false, reason: 'error', message: 'no referral code returned' };
	}

	return { ok: true, duplicate, referralCode, referrals };
}

/**
 * Live count of people who joined from a given referral code. Powers the
 * "N friends joined from your link" counter. Returns null on error/unconfigured.
 */
export async function getReferralCount(referralCode: string): Promise<number | null> {
	const sb = getClient();
	if (!sb) return null;
	try {
		const { data, error } = await sb.rpc('referral_count', { p_code: referralCode });
		if (error || typeof data !== 'number') return null;
		return data;
	} catch {
		return null;
	}
}

/**
 * Read the live total signup count. Returns null when unconfigured or on error
 * (the page then hides the count).
 */
export async function getWaitlistCount(): Promise<number | null> {
	const sb = getClient();
	if (!sb) return null;
	try {
		const { data, error } = await sb.rpc('waitlist_count');
		if (error || typeof data !== 'number') return null;
		return data;
	} catch {
		return null;
	}
}
