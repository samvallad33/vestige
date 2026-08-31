/**
 * v2.0.8 Memory-state colour mode — ruthless coverage.
 *
 * Pure color/state helpers (now in $lib/memory-state). The NodeManager
 * sections were removed with the dead Three.js layer (Phase 0 subtraction),
 * suppression interaction, new-node inheritance, idempotence, and
 * round-trip fidelity. If this file is green, the feature is wired.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('three', async () => {
	const mock = await import('./three-mock');
	return { ...mock };
});

import {
	getMemoryState,
	getAhaGraphColor,
	getNodeColor,
	AHAGRAPH_COLORS,
	MEMORY_STATE_COLORS,
	MEMORY_STATE_DESCRIPTIONS,
	type MemoryState,
	type ColorMode,
} from '$lib/memory-state';
import { NODE_TYPE_COLORS } from '$types';
import { Color, Vector3, MeshStandardMaterial, SpriteMaterial } from './three-mock';
import { makeNode, resetNodeCounter } from './helpers';

// Global spy cleanup — prototype-level spies must not leak between tests.
afterEach(() => {
	vi.restoreAllMocks();
});

// ----------------------------------------------------------------------------
// getMemoryState — boundary analysis across all 4 FSRS buckets
// ----------------------------------------------------------------------------

describe('getMemoryState — bucket classification', () => {
	it.each<[number, MemoryState]>([
		[1.0, 'active'],
		[0.95, 'active'],
		[0.7, 'active'], // inclusive lower bound of active
		[0.6999999, 'dormant'], // just below active threshold
		[0.5, 'dormant'],
		[0.4, 'dormant'], // inclusive lower bound of dormant
		[0.3999999, 'silent'], // just below dormant threshold
		[0.25, 'silent'],
		[0.1, 'silent'], // inclusive lower bound of silent
		[0.0999999, 'unavailable'], // just below silent threshold
		[0.05, 'unavailable'],
		[0.0, 'unavailable'],
	])('classifies retention %f as %s', (retention, expected) => {
		expect(getMemoryState(retention)).toBe(expected);
	});

	it('handles retention > 1 as active (over-strength, shouldn\'t happen but clamp-free)', () => {
		expect(getMemoryState(1.5)).toBe('active');
		expect(getMemoryState(999)).toBe('active');
	});

	it('handles negative retention as unavailable (defensive)', () => {
		expect(getMemoryState(-0.5)).toBe('unavailable');
		expect(getMemoryState(-1000)).toBe('unavailable');
	});

	it('classifies NaN as unavailable (no predicate is true)', () => {
		expect(getMemoryState(NaN)).toBe('unavailable');
	});

	it('classifies +Infinity as active', () => {
		expect(getMemoryState(Infinity)).toBe('active');
	});

	it('classifies -Infinity as unavailable', () => {
		expect(getMemoryState(-Infinity)).toBe('unavailable');
	});

	it('is deterministic and pure — same input gives same output across 10k calls', () => {
		const samples = Array.from({ length: 10000 }, () => Math.random());
		const first = samples.map(getMemoryState);
		const second = samples.map(getMemoryState);
		expect(first).toEqual(second);
	});
});

// ----------------------------------------------------------------------------
// MEMORY_STATE_COLORS — palette integrity
// ----------------------------------------------------------------------------

describe('MEMORY_STATE_COLORS — palette integrity', () => {
	const states: MemoryState[] = ['active', 'dormant', 'silent', 'unavailable'];

	it('defines a colour for every bucket', () => {
		for (const s of states) {
			expect(MEMORY_STATE_COLORS[s]).toBeDefined();
		}
	});

	it.each(states)('%s colour is a valid 6-digit hex string', (state) => {
		const hex = MEMORY_STATE_COLORS[state];
		expect(hex).toMatch(/^#[0-9a-fA-F]{6}$/);
	});

	it('all four bucket colours are distinct', () => {
		const palette = states.map((s) => MEMORY_STATE_COLORS[s].toLowerCase());
		const unique = new Set(palette);
		expect(unique.size).toBe(4);
	});

	it('does not reuse any NODE_TYPE_COLORS value (type mode and state mode stay visually separate)', () => {
		const typeColours = new Set(
			Object.values(NODE_TYPE_COLORS).map((c) => c.toLowerCase())
		);
		for (const s of states) {
			expect(typeColours.has(MEMORY_STATE_COLORS[s].toLowerCase())).toBe(false);
		}
	});

	it('palette is a frozen record shape — all values are strings', () => {
		for (const s of states) {
			expect(typeof MEMORY_STATE_COLORS[s]).toBe('string');
		}
	});
});

// ----------------------------------------------------------------------------
// MEMORY_STATE_DESCRIPTIONS — legend text integrity
// ----------------------------------------------------------------------------

describe('MEMORY_STATE_DESCRIPTIONS — legend copy', () => {
	const states: MemoryState[] = ['active', 'dormant', 'silent', 'unavailable'];

	it('defines a description for every bucket', () => {
		for (const s of states) {
			expect(MEMORY_STATE_DESCRIPTIONS[s]).toBeDefined();
			expect(MEMORY_STATE_DESCRIPTIONS[s].length).toBeGreaterThan(5);
		}
	});

	it.each(states)('%s description contains a threshold parenthetical', (state) => {
		expect(MEMORY_STATE_DESCRIPTIONS[state]).toMatch(/\([^)]+\)/);
	});

	it('active description references the ≥ 70% threshold from getMemoryState', () => {
		expect(MEMORY_STATE_DESCRIPTIONS.active).toMatch(/70/);
	});

	it('dormant description references the 40–70% band', () => {
		expect(MEMORY_STATE_DESCRIPTIONS.dormant).toMatch(/40/);
		expect(MEMORY_STATE_DESCRIPTIONS.dormant).toMatch(/70/);
	});

	it('silent description references the 10–40% band', () => {
		expect(MEMORY_STATE_DESCRIPTIONS.silent).toMatch(/10/);
		expect(MEMORY_STATE_DESCRIPTIONS.silent).toMatch(/40/);
	});

	it('unavailable description references the < 10% threshold', () => {
		expect(MEMORY_STATE_DESCRIPTIONS.unavailable).toMatch(/10/);
	});

	it('descriptions are all distinct (no copy-paste bug)', () => {
		const lines = states.map((s) => MEMORY_STATE_DESCRIPTIONS[s]);
		expect(new Set(lines).size).toBe(4);
	});
});

// ----------------------------------------------------------------------------
// getNodeColor — dispatch correctness across modes
// ----------------------------------------------------------------------------

describe('getNodeColor — type mode', () => {
	it.each(Object.keys(NODE_TYPE_COLORS))('returns NODE_TYPE_COLORS[%s] in type mode', (t) => {
		const node = makeNode({ type: t, retention: 0.5 });
		expect(getNodeColor(node, 'type')).toBe(NODE_TYPE_COLORS[t]);
	});

	it('falls back to steel grey for an unknown type in type mode', () => {
		const node = makeNode({ type: 'totally-fake-type' as any, retention: 0.8 });
		expect(getNodeColor(node, 'type')).toBe('#8B95A5');
	});

	it('type-mode output ignores retention entirely', () => {
		const a = makeNode({ type: 'fact', retention: 0.01 });
		const b = makeNode({ type: 'fact', retention: 0.99 });
		expect(getNodeColor(a, 'type')).toBe(getNodeColor(b, 'type'));
	});
});

describe('getNodeColor — state mode', () => {
	it.each<[number, MemoryState]>([
		[0.9, 'active'],
		[0.5, 'dormant'],
		[0.2, 'silent'],
		[0.0, 'unavailable'],
	])('retention %f yields %s colour', (retention, state) => {
		const node = makeNode({ retention });
		expect(getNodeColor(node, 'state')).toBe(MEMORY_STATE_COLORS[state]);
	});

	it('state-mode output ignores node.type entirely', () => {
		const a = makeNode({ type: 'fact', retention: 0.8 });
		const b = makeNode({ type: 'decision', retention: 0.8 });
		expect(getNodeColor(a, 'state')).toBe(getNodeColor(b, 'state'));
	});

	it('state-mode tolerates unknown type (does not throw, no fallback branch used)', () => {
		const node = makeNode({ type: 'bogus' as any, retention: 0.75 });
		expect(getNodeColor(node, 'state')).toBe(MEMORY_STATE_COLORS.active);
	});
});

describe('getNodeColor — AhaGraph mode', () => {
	it.each([
		[['ahagraph', 'aha'], AHAGRAPH_COLORS.aha],
		[['ahagraph', 'confusion'], AHAGRAPH_COLORS.confusion],
		[['ahagraph', 'weak-spot'], AHAGRAPH_COLORS.confusion],
		[['ahagraph', 'failure'], AHAGRAPH_COLORS.failure],
		[['ahagraph', 'guardrail'], AHAGRAPH_COLORS.failure],
	] as Array<[string[], string]>)('maps tags %j to %s', (tags, color) => {
		const node = makeNode({ type: 'concept', tags });
		expect(getAhaGraphColor(node)).toBe(color);
		expect(getNodeColor(node, 'ahagraph')).toBe(color);
	});

	it('prioritizes aha when a note also mentions confusion tags', () => {
		const node = makeNode({ type: 'note', tags: ['ahagraph', 'aha', 'confusion'] });
		expect(getNodeColor(node, 'ahagraph')).toBe(AHAGRAPH_COLORS.aha);
	});

	it('falls back to node type when no AhaGraph learning tag is present', () => {
		const node = makeNode({ type: 'event', tags: ['ahagraph'] });
		expect(getAhaGraphColor(node)).toBeNull();
		expect(getNodeColor(node, 'ahagraph')).toBe(NODE_TYPE_COLORS.event);
	});
});

// ----------------------------------------------------------------------------
