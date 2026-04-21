/**
 * Pure helpers for AmbientAwarenessStrip.svelte.
 *
 * Extracted so the time-window, event-scan, and timestamp-parsing logic can
 * be unit tested in the vitest `node` environment without jsdom, Svelte
 * rendering, or fake timers bleeding into runes.
 *
 * Contracts
 * ---------
 * - `parseEventTimestamp`: handles (a) numeric ms (>1e12), (b) numeric seconds
 *   (<=1e12), (c) ISO-8601 string, (d) invalid/absent → null.
 * - `bucketizeActivity`: given ms timestamps + `now`, returns 10 counts for a
 *   5-min trailing window. Bucket 0 = oldest 30s, bucket 9 = newest 30s.
 *   Events outside [now-5m, now] are dropped (clock skew).
 * - `findRecentDream`: returns the newest-indexed (feed is newest-first)
 *   DreamCompleted whose parsed timestamp is within 24h, else null. If the
 *   timestamp is unparseable, `now` is used as the fallback (matches the
 *   component's behavior).
 * - `isDreaming`: a DreamStarted within the last 5 min NOT followed by a
 *   newer DreamCompleted. Mirrors the component's derived block exactly.
 * - `hasRecentSuppression`: any MemorySuppressed event with a parsed
 *   timestamp within `thresholdMs` of `now`. Feed is assumed newest-first —
 *   we break as soon as we pass the threshold, matching component behavior.
 *
 * All helpers are null-safe and treat unparseable timestamps consistently
 * (fall back to `now`, matching the on-screen "something just happened" feel).
 */

export interface EventLike {
	type: string;
	data?: Record<string, unknown>;
}

/**
 * Parse a VestigeEvent timestamp, checking `data.timestamp`, then `data.at`,
 * then `data.occurred_at`. Supports ms-since-epoch numbers, seconds-since-epoch
 * numbers, and ISO-8601 strings. Returns null for absent / invalid input.
 *
 * Numeric heuristic: values > 1e12 are treated as ms (2001+), values <= 1e12
 * are treated as seconds. `1e12 ms` ≈ Sept 2001, so any real ms timestamp
 * lands safely on the "ms" side.
 */
export function parseEventTimestamp(event: EventLike): number | null {
	const d = event.data;
	if (!d || typeof d !== 'object') return null;
	const raw =
		(d.timestamp as string | number | undefined) ??
		(d.at as string | number | undefined) ??
		(d.occurred_at as string | number | undefined);
	if (raw === undefined || raw === null) return null;
	if (typeof raw === 'number') {
		if (!Number.isFinite(raw)) return null;
		return raw > 1e12 ? raw : raw * 1000;
	}
	if (typeof raw !== 'string') return null;
	const ms = Date.parse(raw);
	return Number.isFinite(ms) ? ms : null;
}

export const ACTIVITY_BUCKET_COUNT = 10;
export const ACTIVITY_BUCKET_MS = 30_000;
export const ACTIVITY_WINDOW_MS = ACTIVITY_BUCKET_COUNT * ACTIVITY_BUCKET_MS;

export interface ActivityBucket {
	count: number;
	ratio: number;
}

/**
 * Bucket event timestamps into 10 × 30s buckets for a 5-min trailing window.
 * Events with `type === 'Heartbeat'` are skipped (noise). Events whose
 * timestamp is out of window (clock skew / pre-history) are dropped.
 *
 * Returned `ratio` is `count / max(1, maxBucketCount)` — so a sparkline with
 * zero events has all-zero ratios (no division by zero) and a sparkline with
 * a single spike peaks at 1.0.
 */
export function bucketizeActivity(
	events: EventLike[],
	nowMs: number,
): ActivityBucket[] {
	const start = nowMs - ACTIVITY_WINDOW_MS;
	const counts = new Array<number>(ACTIVITY_BUCKET_COUNT).fill(0);
	for (const e of events) {
		if (e.type === 'Heartbeat') continue;
		const t = parseEventTimestamp(e);
		if (t === null || t < start || t > nowMs) continue;
		const idx = Math.min(
			ACTIVITY_BUCKET_COUNT - 1,
			Math.floor((t - start) / ACTIVITY_BUCKET_MS),
		);
		counts[idx] += 1;
	}
	const max = Math.max(1, ...counts);
	return counts.map((count) => ({ count, ratio: count / max }));
}

/**
 * Find the most recent DreamCompleted within 24h of `nowMs`.
 * Feed is assumed newest-first — we return the FIRST match.
 * Unparseable timestamps fall back to `nowMs` (matches component behavior).
 */
export function findRecentDream(
	events: EventLike[],
	nowMs: number,
): EventLike | null {
	const dayAgo = nowMs - 24 * 60 * 60 * 1000;
	for (const e of events) {
		if (e.type !== 'DreamCompleted') continue;
		const t = parseEventTimestamp(e) ?? nowMs;
		if (t >= dayAgo) return e;
		return null; // newest-first: older ones definitely won't match
	}
	return null;
}

/**
 * Extract `insights_generated` / `insightsGenerated` from a DreamCompleted
 * event payload. Returns null if missing or non-numeric.
 */
export function dreamInsightsCount(event: EventLike | null): number | null {
	if (!event || !event.data) return null;
	const d = event.data;
	const raw =
		typeof d.insights_generated === 'number'
			? d.insights_generated
			: typeof d.insightsGenerated === 'number'
				? d.insightsGenerated
				: null;
	return raw !== null && Number.isFinite(raw) ? raw : null;
}

/**
 * A Dream is in flight if the newest DreamStarted is within 5 min of `nowMs`
 * AND there is no DreamCompleted with a timestamp >= that DreamStarted.
 *
 * Feed is assumed newest-first. We scan once, grabbing the first Started and
 * first Completed, then compare — matching the component's derived block.
 */
export function isDreaming(events: EventLike[], nowMs: number): boolean {
	let started: EventLike | null = null;
	let completed: EventLike | null = null;
	for (const e of events) {
		if (!started && e.type === 'DreamStarted') started = e;
		if (!completed && e.type === 'DreamCompleted') completed = e;
		if (started && completed) break;
	}
	if (!started) return false;
	const startedAt = parseEventTimestamp(started) ?? nowMs;
	const fiveMinAgo = nowMs - 5 * 60 * 1000;
	if (startedAt < fiveMinAgo) return false;
	if (!completed) return true;
	const completedAt = parseEventTimestamp(completed) ?? nowMs;
	return completedAt < startedAt;
}

/**
 * Format an "ago" duration compactly. Pure and deterministic.
 * 0-59s → "Ns ago", 60-3599s → "Nm ago", <24h → "Nh ago", else "Nd ago".
 * Negative input is clamped to 0.
 */
export function formatAgo(ms: number): string {
	const clamped = Math.max(0, ms);
	const s = Math.floor(clamped / 1000);
	if (s < 60) return `${s}s ago`;
	const m = Math.floor(s / 60);
	if (m < 60) return `${m}m ago`;
	const h = Math.floor(m / 60);
	if (h < 24) return `${h}h ago`;
	return `${Math.floor(h / 24)}d ago`;
}

/**
 * True if any MemorySuppressed event lies within `thresholdMs` of `nowMs`.
 * Feed assumed newest-first — break as soon as we encounter one OUTSIDE
 * the window (all older ones are definitely older). Unparseable timestamps
 * fall back to `nowMs` so the flash fires — matches component behavior.
 */
export function hasRecentSuppression(
	events: EventLike[],
	nowMs: number,
	thresholdMs: number = 10_000,
): boolean {
	const cutoff = nowMs - thresholdMs;
	for (const e of events) {
		if (e.type !== 'MemorySuppressed') continue;
		const t = parseEventTimestamp(e) ?? nowMs;
		if (t >= cutoff) return true;
		return false; // newest-first: older ones definitely won't match
	}
	return false;
}
