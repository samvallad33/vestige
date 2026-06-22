// ═══════════════════════════════════════════════════════════════════════════
//  AGENT BLACK BOX — presentation helpers
// ───────────────────────────────────────────────────────────────────────────
//  Pure functions that turn a raw `TraceEvent` into the label, color, glyph,
//  and one-line summary the Black Box timeline renders. Kept out of the
//  component so they are unit-testable and reused by the Proof Mode header.
// ═══════════════════════════════════════════════════════════════════════════
import type { TraceEvent } from '$lib/stores/api';

export type TraceKind = TraceEvent['type'];

/** The accent color for each trace-event kind (CSS color value). */
export function eventColor(kind: TraceKind): string {
	switch (kind) {
		case 'mcp.call':
			return 'var(--color-synapse-glow, #818cf8)';
		case 'memory.retrieve':
			return 'var(--color-recall, #10b981)';
		case 'memory.suppress':
			return '#a78bfa'; // violet — the forgetting hue
		case 'memory.write':
			return '#38bdf8'; // sky — a new write
		case 'contradiction.detected':
			return '#fb7185'; // rose — tension
		case 'sanhedrin.veto':
			return '#f43f5e'; // red — a block
		case 'dream.patch':
			return '#c084fc'; // purple — dream
		default:
			return 'var(--color-synapse, #6366f1)';
	}
}

/** A short human label for each kind. */
export function eventLabel(kind: TraceKind): string {
	switch (kind) {
		case 'mcp.call':
			return 'Tool Call';
		case 'memory.retrieve':
			return 'Retrieved';
		case 'memory.suppress':
			return 'Suppressed';
		case 'memory.write':
			return 'Wrote';
		case 'contradiction.detected':
			return 'Contradiction';
		case 'sanhedrin.veto':
			return 'Veto';
		case 'dream.patch':
			return 'Dream Patch';
		default:
			return kind;
	}
}

/** A single glyph (emoji-free SVG path is overkill here; a compact symbol). */
export function eventGlyph(kind: TraceKind): string {
	switch (kind) {
		case 'mcp.call':
			return '⟐';
		case 'memory.retrieve':
			return '◉';
		case 'memory.suppress':
			return '⊘';
		case 'memory.write':
			return '✎';
		case 'contradiction.detected':
			return '⚡';
		case 'sanhedrin.veto':
			return '⛔';
		case 'dream.patch':
			return '☾';
		default:
			return '•';
	}
}

/** A one-line summary of what an event did, for the timeline row. */
export function eventSummary(ev: TraceEvent): string {
	switch (ev.type) {
		case 'mcp.call':
			return `${ev.tool}  ·  args ${ev.argsHash.slice(0, 8)}`;
		case 'memory.retrieve':
			return `${ev.ids.length} ${ev.ids.length === 1 ? 'memory' : 'memories'} surfaced`;
		case 'memory.suppress':
			return `${ev.id.slice(0, 8)} — ${ev.reason.replace('_', ' ')}`;
		case 'memory.write':
			return `${ev.id.slice(0, 8)} — ${ev.source}`;
		case 'contradiction.detected':
			return ev.detail;
		case 'sanhedrin.veto':
			return `"${ev.claim}" (conf ${(ev.confidence * 100).toFixed(0)}%)`;
		case 'dream.patch':
			return `${ev.proposalIds.length} consolidation proposal(s)`;
		default:
			return '';
	}
}

/** The memory ids an event touched (for graph-pulse replay). */
export function eventMemoryIds(ev: TraceEvent): string[] {
	switch (ev.type) {
		case 'memory.retrieve':
			return ev.ids;
		case 'memory.suppress':
		case 'memory.write':
			return [ev.id];
		case 'contradiction.detected':
			return ev.ids;
		case 'sanhedrin.veto':
			return ev.evidenceIds;
		case 'dream.patch':
			return ev.proposalIds;
		default:
			return [];
	}
}

/** Format a millisecond timestamp as a clock time. */
export function formatAt(at: number): string {
	if (!Number.isFinite(at) || at <= 0) return '—';
	const d = new Date(at);
	return d.toLocaleTimeString(undefined, {
		hour12: false,
		hour: '2-digit',
		minute: '2-digit',
		second: '2-digit'
	});
}

/** Elapsed milliseconds of an event relative to the run's first event. */
export function relativeMs(at: number, startAt: number): number {
	return Math.max(0, at - startAt);
}
