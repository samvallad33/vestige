/**
 * Unit tests for the Agent Black Box presentation helpers.
 *
 * Scope: the pure label/color/glyph/summary/ids functions that turn a raw
 * `TraceEvent` into what the timeline renders. NeuroRuntime v0 adds two event
 * kinds — `memory.quarantine` (the Microglial Firewall) and `episode.boundary`
 * (a phase divider) — so these tests focus on those, plus the parity checks
 * that every existing kind still maps to a non-default label/glyph/color.
 */
import { describe, it, expect } from 'vitest';
import {
	eventColor,
	eventLabel,
	eventGlyph,
	eventSummary,
	eventMemoryIds,
	type TraceKind,
} from '../blackbox-helpers';
import type { TraceEvent } from '$lib/stores/api';

// Every discriminant the union currently carries — kept here so a new variant
// that forgets a helper case is caught by the parity tests below.
const ALL_KINDS: TraceKind[] = [
	'mcp.call',
	'memory.retrieve',
	'memory.suppress',
	'memory.write',
	'contradiction.detected',
	'sanhedrin.veto',
	'dream.patch',
	'memory.quarantine',
	'episode.boundary',
];

function quarantine(over: Partial<Extract<TraceEvent, { type: 'memory.quarantine' }>> = {}) {
	return {
		type: 'memory.quarantine' as const,
		runId: 'run_1',
		id: 'mem_abcdef0123',
		reason: 'prompt_injection',
		threat: 'Detected an instruction-injection payload masquerading as a memory.',
		influenceAllowed: false,
		at: 1_700_000_000_000,
		...over,
	};
}

function boundary(over: Partial<Extract<TraceEvent, { type: 'episode.boundary' }>> = {}) {
	return {
		type: 'episode.boundary' as const,
		runId: 'run_1',
		episode: 'ep_install',
		label: 'Installing',
		at: 1_700_000_000_000,
		...over,
	};
}

// ---------------------------------------------------------------------------
// Parity — every kind maps to something deliberate (not the fallthrough)
// ---------------------------------------------------------------------------

describe('helper parity across all kinds', () => {
	it('eventLabel never returns the raw discriminant for a known kind', () => {
		for (const kind of ALL_KINDS) {
			expect(eventLabel(kind)).not.toBe(kind);
		}
	});

	it('eventGlyph never returns the fallback bullet for a known kind', () => {
		for (const kind of ALL_KINDS) {
			expect(eventGlyph(kind)).not.toBe('•');
		}
	});

	it('eventColor returns a non-empty value for every known kind', () => {
		for (const kind of ALL_KINDS) {
			expect(eventColor(kind).length).toBeGreaterThan(0);
		}
	});
});

// ---------------------------------------------------------------------------
// memory.quarantine — the Microglial Firewall
// ---------------------------------------------------------------------------

describe('memory.quarantine', () => {
	it('labels it the Microglial Firewall', () => {
		expect(eventLabel('memory.quarantine')).toBe('Microglial Firewall');
	});

	it('uses a danger-red accent', () => {
		expect(eventColor('memory.quarantine')).toBe('#ef4444');
	});

	it('uses the shield glyph', () => {
		expect(eventGlyph('memory.quarantine')).toBe('🛡');
	});

	it('summary surfaces the threat prose and the humanized reason code', () => {
		const ev = quarantine();
		const summary = eventSummary(ev);
		expect(summary).toContain('instruction-injection payload');
		// reason code is rendered with underscores turned into spaces.
		expect(summary).toContain('prompt injection');
		expect(summary).not.toContain('prompt_injection');
	});

	it('humanizes ALL underscores in a multi-token reason code', () => {
		const ev = quarantine({ reason: 'contradicts_high_trust' });
		expect(eventSummary(ev)).toContain('contradicts high trust');
	});

	it('counts the quarantined memory id as touched (for graph-pulse replay)', () => {
		expect(eventMemoryIds(quarantine())).toEqual(['mem_abcdef0123']);
	});
});

// ---------------------------------------------------------------------------
// episode.boundary — a readable phase divider
// ---------------------------------------------------------------------------

describe('episode.boundary', () => {
	it('labels it Episode', () => {
		expect(eventLabel('episode.boundary')).toBe('Episode');
	});

	it('uses the flag glyph', () => {
		expect(eventGlyph('episode.boundary')).toBe('⚑');
	});

	it('summary is the human-readable label carried by the event', () => {
		expect(eventSummary(boundary({ label: 'Debugging' }))).toBe('Debugging');
	});

	it('touches no memory — it is a pure phase marker', () => {
		expect(eventMemoryIds(boundary())).toEqual([]);
	});
});
