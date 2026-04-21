/**
 * Pure helpers for MemoryAuditTrail.
 *
 * Extracted for isolated unit testing in a Node (vitest) environment —
 * no DOM, no Svelte runtime, no fetch. Every function in this module is
 * deterministic given its inputs.
 */

export type AuditAction =
	| 'created'
	| 'accessed'
	| 'promoted'
	| 'demoted'
	| 'edited'
	| 'suppressed'
	| 'dreamed'
	| 'reconsolidated';

export interface AuditEvent {
	action: AuditAction;
	timestamp: string; // ISO
	old_value?: number;
	new_value?: number;
	reason?: string;
	triggered_by?: string;
}

export type MarkerKind =
	| 'dot'
	| 'arrow-up'
	| 'arrow-down'
	| 'pencil'
	| 'x'
	| 'star'
	| 'circle-arrow'
	| 'ring';

export interface Meta {
	label: string;
	color: string; // hex for dot + glow
	glyph: string; // optional inline symbol
	kind: MarkerKind;
}

/**
 * Event type → visual metadata. Each action maps to a UNIQUE marker `kind`
 * so the 8 event types are visually distinguishable without relying on the
 * colour palette alone (accessibility).
 */
export const META: Record<AuditAction, Meta> = {
	created: { label: 'Created', color: '#10b981', glyph: '', kind: 'ring' },
	accessed: { label: 'Accessed', color: '#3b82f6', glyph: '', kind: 'dot' },
	promoted: { label: 'Promoted', color: '#10b981', glyph: '', kind: 'arrow-up' },
	demoted: { label: 'Demoted', color: '#f59e0b', glyph: '', kind: 'arrow-down' },
	edited: { label: 'Edited', color: '#facc15', glyph: '', kind: 'pencil' },
	suppressed: { label: 'Suppressed', color: '#a855f7', glyph: '', kind: 'x' },
	dreamed: { label: 'Dreamed', color: '#c084fc', glyph: '', kind: 'star' },
	reconsolidated: { label: 'Reconsolidated', color: '#ec4899', glyph: '', kind: 'circle-arrow' }
};

export const VISIBLE_LIMIT = 15;

/**
 * All 8 `AuditAction` values, in the canonical order. Used both by the
 * event generator (`actionPool`) and by tests that verify uniqueness of
 * the marker mapping.
 */
export const ALL_ACTIONS: readonly AuditAction[] = [
	'created',
	'accessed',
	'promoted',
	'demoted',
	'edited',
	'suppressed',
	'dreamed',
	'reconsolidated'
] as const;

/**
 * Hash a string id into a 32-bit unsigned seed. Stable across runs.
 */
export function hashSeed(id: string): number {
	let seed = 0;
	for (let i = 0; i < id.length; i++) seed = (seed * 31 + id.charCodeAt(i)) >>> 0;
	return seed;
}

/**
 * Linear congruential PRNG bound to a mutable seed. Returns a function
 * that yields floats in `[0, 1)` — critically, NEVER 1.0, so callers
 * can safely use `Math.floor(rand() * arr.length)` without off-by-one.
 */
export function makeRand(initialSeed: number): () => number {
	let seed = initialSeed >>> 0;
	return () => {
		seed = (seed * 1664525 + 1013904223) >>> 0;
		// Divide by 2^32, not 2^32 - 1 — the latter can yield exactly 1.0
		// when seed is UINT32_MAX, breaking array-index math.
		return seed / 0x100000000;
	};
}

/**
 * Deterministic mock audit-trail generator. Same `memoryId` + `nowMs`
 * ALWAYS yields the same event sequence (critical for snapshot stability
 * and for tests). An empty `memoryId` yields no events — the audit trail
 * panel should never invent history for a non-existent memory.
 *
 * `countOverride` lets tests force a specific number of events (e.g.
 * to cross the 15-event visibility threshold, which the default range
 * 8-15 cannot do).
 */
export function generateMockAuditTrail(
	memoryId: string,
	nowMs: number = Date.now(),
	countOverride?: number
): AuditEvent[] {
	if (!memoryId) return [];

	const rand = makeRand(hashSeed(memoryId));
	const count = countOverride ?? 8 + Math.floor(rand() * 8); // default 8-15 events
	if (count <= 0) return [];

	const out: AuditEvent[] = [];

	const createdAt = nowMs - (14 + rand() * 21) * 86_400_000; // 14-35 days ago
	out.push({
		action: 'created',
		timestamp: new Date(createdAt).toISOString(),
		reason: 'smart_ingest · prediction-error gate opened',
		triggered_by: 'smart_ingest'
	});

	let t = createdAt;
	let retention = 0.5 + rand() * 0.2;
	const actionPool: AuditAction[] = [
		'accessed',
		'accessed',
		'accessed',
		'accessed',
		'promoted',
		'demoted',
		'edited',
		'dreamed',
		'reconsolidated',
		'suppressed'
	];

	for (let i = 1; i < count; i++) {
		t += rand() * 5 * 86_400_000 + 3_600_000; // 1h-5d between events
		const action = actionPool[Math.floor(rand() * actionPool.length)];
		const ev: AuditEvent = { action, timestamp: new Date(t).toISOString() };

		switch (action) {
			case 'accessed': {
				const old = retention;
				retention = Math.min(1, retention + rand() * 0.04 + 0.01);
				ev.old_value = old;
				ev.new_value = retention;
				ev.triggered_by = rand() > 0.5 ? 'search' : 'deep_reference';
				break;
			}
			case 'promoted': {
				const old = retention;
				retention = Math.min(1, retention + 0.1);
				ev.old_value = old;
				ev.new_value = retention;
				ev.reason = 'confirmed helpful by user';
				ev.triggered_by = 'memory(action=promote)';
				break;
			}
			case 'demoted': {
				const old = retention;
				retention = Math.max(0, retention - 0.15);
				ev.old_value = old;
				ev.new_value = retention;
				ev.reason = 'user flagged as outdated';
				ev.triggered_by = 'memory(action=demote)';
				break;
			}
			case 'edited': {
				ev.reason = 'content refined, FSRS state preserved';
				ev.triggered_by = 'memory(action=edit)';
				break;
			}
			case 'suppressed': {
				const old = retention;
				retention = Math.max(0, retention - 0.08);
				ev.old_value = old;
				ev.new_value = retention;
				ev.reason = 'top-down inhibition (Anderson 2025)';
				ev.triggered_by = 'suppress(dashboard)';
				break;
			}
			case 'dreamed': {
				const old = retention;
				retention = Math.min(1, retention + 0.05);
				ev.old_value = old;
				ev.new_value = retention;
				ev.reason = 'replayed during dream consolidation';
				ev.triggered_by = 'dream()';
				break;
			}
			case 'reconsolidated': {
				ev.reason = 'edited within 5-min labile window (Nader)';
				ev.triggered_by = 'reconsolidation-manager';
				break;
			}
			case 'created':
				// Created is only emitted once, as the first event. If the pool
				// ever yields it again, treat it as a no-op access marker with
				// no retention change — defensive, not expected.
				ev.triggered_by = 'smart_ingest';
				break;
		}

		out.push(ev);
	}

	// Newest first for display.
	return out.reverse();
}

/**
 * Humanised relative time. Uses supplied `nowMs` for deterministic tests;
 * defaults to `Date.now()` in production.
 *
 * Boundaries (strictly `<`, so 60s flips to "1m", 60m flips to "1h", etc.):
 *   <60s → "Ns ago"
 *   <60m → "Nm ago"
 *   <24h → "Nh ago"
 *   <30d → "Nd ago"
 *   <12mo → "Nmo ago"
 *   else → "Ny ago"
 *
 * Future timestamps (nowMs < then) clamp to "0s ago" rather than returning
 * a negative string — the audit trail is a past-only view.
 */
export function relativeTime(iso: string, nowMs: number = Date.now()): string {
	const then = new Date(iso).getTime();
	const diff = Math.max(0, nowMs - then);
	const s = Math.floor(diff / 1000);
	if (s < 60) return `${s}s ago`;
	const m = Math.floor(s / 60);
	if (m < 60) return `${m}m ago`;
	const h = Math.floor(m / 60);
	if (h < 24) return `${h}h ago`;
	const d = Math.floor(h / 24);
	if (d < 30) return `${d}d ago`;
	const mo = Math.floor(d / 30);
	if (mo < 12) return `${mo}mo ago`;
	const y = Math.floor(mo / 12);
	return `${y}y ago`;
}

/**
 * Retention delta formatter. Behaviour:
 *   (undef, undef) → null        — no retention movement on this event
 *   (undef, 0.72)  → "set 0.72"  — initial value, no prior state
 *   (0.50, undef)  → "was 0.50"  — retention cleared (rare)
 *   (0.50, 0.72)   → "0.50 → 0.72"
 *
 * The `retention ` prefix is left to the caller so tests can compare the
 * core formatted value precisely.
 */
export function formatRetentionDelta(
	oldValue: number | undefined,
	newValue: number | undefined
): string | null {
	const hasOld = typeof oldValue === 'number' && Number.isFinite(oldValue);
	const hasNew = typeof newValue === 'number' && Number.isFinite(newValue);
	if (!hasOld && !hasNew) return null;
	if (!hasOld && hasNew) return `set ${newValue!.toFixed(2)}`;
	if (hasOld && !hasNew) return `was ${oldValue!.toFixed(2)}`;
	return `${oldValue!.toFixed(2)} → ${newValue!.toFixed(2)}`;
}

/**
 * Split an event list into (visible, hiddenCount) per the 15-event cap.
 * Exactly 15 events → no toggle (hiddenCount = 0). 16+ → toggle.
 */
export function splitVisible(
	events: AuditEvent[],
	showAll: boolean
): { visible: AuditEvent[]; hiddenCount: number } {
	if (showAll || events.length <= VISIBLE_LIMIT) {
		return { visible: events, hiddenCount: Math.max(0, events.length - VISIBLE_LIMIT) };
	}
	return {
		visible: events.slice(0, VISIBLE_LIMIT),
		hiddenCount: events.length - VISIBLE_LIMIT
	};
}
