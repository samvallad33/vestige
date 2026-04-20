// Unit tests for the Pulse toast store (v2.2).
//
// The store subscribes to `eventFeed` from `$stores/websocket` at IMPORT
// TIME, so every test re-imports the module via `vi.resetModules()` +
// dynamic import to get a fresh `lastSeen` / `nextId` / `lastConnectionAt`
// / dwell-timer registry. Without this, the module-level state leaks
// between tests (especially the 1500ms ConnectionDiscovered throttle).
//
// The eventFeed is mocked as a plain writable<VestigeEvent[]> — we push
// arrays directly, mirroring the way the real websocket store prepends
// new events at index 0.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { writable, get, type Writable } from 'svelte/store';
import type { VestigeEvent } from '$types';

// The mock `eventFeed` is hoisted so vi.mock can reference it.
const mockEventFeed: Writable<VestigeEvent[]> = writable<VestigeEvent[]>([]);

vi.mock('$stores/websocket', () => ({
	eventFeed: mockEventFeed,
}));

// Helper — make a fresh VestigeEvent with a unique object identity.
// The store uses reference equality (e === lastSeen) to detect freshness,
// so every emission must be a distinct object.
function makeEvent<T extends VestigeEvent['type']>(
	type: T,
	data: Record<string, unknown> = {},
): VestigeEvent {
	return { type, data };
}

// Prepend events onto the feed — mirrors the real websocket store, which
// does `[parsed, ...events].slice(0, 200)`. Pass a single event or an
// array (oldest-last, so `push([newest, older, oldest])` is the shape
// the real subscriber sees).
function emit(events: VestigeEvent | VestigeEvent[]) {
	const arr = Array.isArray(events) ? events : [events];
	mockEventFeed.update((prev) => [...arr, ...prev]);
}

// Reset the feed between tests. Combined with vi.resetModules() this
// guarantees each test starts with a virgin toast store.
function resetFeed() {
	mockEventFeed.set([]);
}

// Dynamically import the toast store after resetModules so we get a
// fresh subscription + fresh module-level state every test.
async function loadToastStore() {
	const mod = await import('../toast');
	return mod;
}

describe('toast store', () => {
	beforeEach(() => {
		vi.useFakeTimers();
		resetFeed();
		vi.resetModules();
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	// ---------------------------------------------------------------
	// Identity-based batch walk (silent-lobotomy fix)
	// ---------------------------------------------------------------
	describe('identity-based batch walk', () => {
		it('processes ALL events when multiple land in one tick, not just the newest', async () => {
			const { toasts } = await loadToastStore();

			const e1 = makeEvent('DreamCompleted', {
				memories_replayed: 3,
				connections_found: 1,
				insights_generated: 0,
				duration_ms: 500,
			});
			const e2 = makeEvent('ConnectionDiscovered', {
				connection_type: 'semantic',
				weight: 0.8,
			});
			const e3 = makeEvent('MemoryPromoted', { new_retention: 0.9 });

			// All three land in the same tick — emit as a single array
			// (oldest-last, matching the real store prepend order).
			emit([e3, e2, e1]);

			const list = get(toasts);
			expect(list.length).toBe(3);
			// Queue is newest-first (store prepends): [e3, e2, e1]
			expect(list[0].type).toBe('MemoryPromoted');
			expect(list[1].type).toBe('ConnectionDiscovered');
			expect(list[2].type).toBe('DreamCompleted');
		});

		it('processes events in OLDEST-first narrative order (DreamCompleted before ConnectionDiscovered)', async () => {
			const { toasts } = await loadToastStore();

			const dream = makeEvent('DreamCompleted', {
				memories_replayed: 10,
				connections_found: 2,
				insights_generated: 1,
				duration_ms: 800,
			});
			const bridge = makeEvent('ConnectionDiscovered', {
				connection_type: 'causal',
				weight: 0.75,
			});

			// dream is older, bridge is newer → emit [bridge, dream]
			emit([bridge, dream]);

			const list = get(toasts);
			// IDs are assigned sequentially as events are processed. Dream
			// gets processed first (oldest-first walk) → id=1. Bridge → id=2.
			// Store prepends, so the queue is [bridge(2), dream(1)].
			expect(list[0].id).toBeGreaterThan(list[1].id);
			expect(list[1].type).toBe('DreamCompleted');
			expect(list[0].type).toBe('ConnectionDiscovered');
		});

		it('does not duplicate toasts when the subscriber re-fires with no new events', async () => {
			const { toasts } = await loadToastStore();

			const e = makeEvent('MemoryPromoted', { new_retention: 0.85 });
			emit(e);
			expect(get(toasts).length).toBe(1);

			// Re-setting the same array (no new events) must NOT produce a
			// second toast. Also pushing an unrelated no-op update.
			mockEventFeed.update((prev) => [...prev]);
			expect(get(toasts).length).toBe(1);
		});

		it('handles empty feed updates gracefully (no toasts created)', async () => {
			const { toasts } = await loadToastStore();

			// Force the subscriber to fire with an empty array.
			mockEventFeed.set([]);
			expect(get(toasts).length).toBe(0);
		});

		it('falls back gracefully when lastSeen is evicted from the capped feed', async () => {
			const { toasts } = await loadToastStore();

			// Emit a first event that becomes lastSeen.
			const first = makeEvent('MemoryPromoted', { new_retention: 0.8 });
			emit(first);
			expect(get(toasts).length).toBe(1);

			// Now emit a burst where the old lastSeen is pushed out. Since
			// we can never match it by identity, the walk goes to the end
			// of the array and translates everything.
			const burst = [
				makeEvent('MemoryPromoted', { new_retention: 0.81 }),
				makeEvent('MemoryPromoted', { new_retention: 0.82 }),
				makeEvent('MemoryPromoted', { new_retention: 0.83 }),
			];
			// Replace the feed entirely — the old `first` event is gone.
			mockEventFeed.set(burst);

			// All three new events get translated. Plus the one we already had.
			expect(get(toasts).length).toBe(4);
		});
	});

	// ---------------------------------------------------------------
	// Event translation — one test per meaningful type
	// ---------------------------------------------------------------
	describe('event translation', () => {
		it('DreamCompleted → title + body with replayed/connections/insights/duration', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('DreamCompleted', {
					memories_replayed: 127,
					connections_found: 43,
					insights_generated: 5,
					duration_ms: 2400,
				}),
			);

			const t = get(toasts)[0];
			expect(t.title).toBe('Dream consolidated');
			expect(t.body).toContain('Replayed 127 memories');
			expect(t.body).toContain('43 new connections');
			expect(t.body).toContain('5 insights');
			expect(t.body).toContain('2.4s');
		});

		it('DreamCompleted → singular grammar when replayed === 1 and found === 1', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('DreamCompleted', {
					memories_replayed: 1,
					connections_found: 1,
					insights_generated: 1,
					duration_ms: 300,
				}),
			);

			const t = get(toasts)[0];
			expect(t.body).toContain('Replayed 1 memory');
			expect(t.body).toContain('1 new connection');
			expect(t.body).not.toContain('1 new connections');
			expect(t.body).toContain('1 insight');
		});

		it('ConsolidationCompleted → title + body with nodes/decay/embedded/duration', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('ConsolidationCompleted', {
					nodes_processed: 892,
					decay_applied: 156,
					embeddings_generated: 48,
					duration_ms: 1100,
				}),
			);

			const t = get(toasts)[0];
			expect(t.title).toBe('Consolidation swept');
			expect(t.body).toContain('892 nodes');
			expect(t.body).toContain('156 decayed');
			expect(t.body).toContain('48 embedded');
			expect(t.body).toContain('1.1s');
		});

		it('ConnectionDiscovered → title + connection type + weight', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('ConnectionDiscovered', {
					connection_type: 'semantic',
					weight: 0.87,
				}),
			);

			const t = get(toasts)[0];
			expect(t.title).toBe('Bridge discovered');
			expect(t.body).toContain('semantic');
			expect(t.body).toContain('0.87');
		});

		it('MemoryPromoted → body includes retention %', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.85 }));

			const t = get(toasts)[0];
			expect(t.title).toBe('Memory promoted');
			expect(t.body).toBe('retention 85%');
		});

		it('MemoryDemoted → body includes retention %', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryDemoted', { new_retention: 0.42 }));

			const t = get(toasts)[0];
			expect(t.title).toBe('Memory demoted');
			expect(t.body).toBe('retention 42%');
		});

		it('MemorySuppressed (cascade=0) → suppression # only', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('MemorySuppressed', {
					suppression_count: 3,
					estimated_cascade: 0,
				}),
			);

			const t = get(toasts)[0];
			expect(t.title).toBe('Forgetting');
			expect(t.body).toBe('suppression #3');
			expect(t.body).not.toContain('Rac1');
		});

		it('MemorySuppressed (cascade>0) → suppression # + Rac1 cascade mention', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('MemorySuppressed', {
					suppression_count: 2,
					estimated_cascade: 8,
				}),
			);

			const t = get(toasts)[0];
			expect(t.body).toContain('suppression #2');
			expect(t.body).toContain('Rac1 cascade');
			expect(t.body).toContain('~8 neighbors');
		});

		it('MemoryUnsuppressed (remaining>0) → remaining count', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryUnsuppressed', { remaining_count: 2 }));

			const t = get(toasts)[0];
			expect(t.title).toBe('Recovered');
			expect(t.body).toContain('2 suppressions remain');
		});

		it('MemoryUnsuppressed (remaining=0) → "fully unsuppressed"', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryUnsuppressed', { remaining_count: 0 }));

			const t = get(toasts)[0];
			expect(t.body).toBe('fully unsuppressed');
		});

		it('Rac1CascadeSwept → seeds + neighbors affected', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('Rac1CascadeSwept', {
					seeds: 3,
					neighbors_affected: 14,
				}),
			);

			const t = get(toasts)[0];
			expect(t.title).toBe('Rac1 cascade');
			expect(t.body).toContain('3 seeds');
			expect(t.body).toContain('14 dendritic spines');
			expect(t.body).toContain('pruned');
		});

		it('MemoryDeleted → body is id truncated to first 8 chars', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('MemoryDeleted', {
					id: 'deadbeefcafef00d1234567890abcdef',
				}),
			);

			const t = get(toasts)[0];
			expect(t.title).toBe('Memory deleted');
			expect(t.body).toBe('deadbeef');
		});

		it.each([
			['Heartbeat'],
			['SearchPerformed'],
			['RetentionDecayed'],
			['ActivationSpread'],
			['ImportanceScored'],
			['MemoryCreated'],
			['MemoryUpdated'],
			['DreamStarted'],
			['DreamProgress'],
			['ConsolidationStarted'],
			['Connected'],
		] as const)('noise event %s produces no toast', async ([type]) => {
			const { toasts } = await loadToastStore();

			emit(makeEvent(type as VestigeEvent['type'], {}));

			expect(get(toasts).length).toBe(0);
		});
	});

	// ---------------------------------------------------------------
	// ConnectionDiscovered throttle
	// ---------------------------------------------------------------
	describe('ConnectionDiscovered throttle', () => {
		it('two ConnectionDiscovered within 1500ms → only one toast', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('ConnectionDiscovered', {
					connection_type: 'semantic',
					weight: 0.8,
				}),
			);
			expect(get(toasts).length).toBe(1);

			// 500ms later — still inside throttle
			vi.advanceTimersByTime(500);

			emit(
				makeEvent('ConnectionDiscovered', {
					connection_type: 'causal',
					weight: 0.9,
				}),
			);

			expect(get(toasts).length).toBe(1);
		});

		it('two ConnectionDiscovered more than 1500ms apart → both toasts', async () => {
			const { toasts } = await loadToastStore();

			emit(
				makeEvent('ConnectionDiscovered', {
					connection_type: 'semantic',
					weight: 0.8,
				}),
			);
			expect(get(toasts).length).toBe(1);

			// Wait past the throttle window (1500ms).
			vi.advanceTimersByTime(1600);

			emit(
				makeEvent('ConnectionDiscovered', {
					connection_type: 'causal',
					weight: 0.9,
				}),
			);

			expect(get(toasts).length).toBe(2);
		});
	});

	// ---------------------------------------------------------------
	// Hover-panic — pauseDwell / resumeDwell
	// ---------------------------------------------------------------
	describe('hover-panic (pauseDwell / resumeDwell)', () => {
		it('auto-dismiss fires after dwellMs when not paused', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.8 }));
			const t = get(toasts)[0];
			expect(t).toBeDefined();
			expect(t.dwellMs).toBe(4500);

			// Advance just past the dwell — toast should be gone.
			vi.advanceTimersByTime(4600);
			expect(get(toasts).length).toBe(0);
		});

		it('pauseDwell stops the auto-dismiss — toast survives past natural dwellMs', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.8 }));
			const t = get(toasts)[0];

			// 1 second in, pause.
			vi.advanceTimersByTime(1000);
			toasts.pauseDwell(t.id, t.dwellMs);

			// Advance WAY past the natural dwell — still there.
			vi.advanceTimersByTime(10_000);
			expect(get(toasts).length).toBe(1);
			expect(get(toasts)[0].id).toBe(t.id);
		});

		it('resumeDwell schedules dismissal for the REMAINING time, not the full dwellMs', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.8 }));
			const t = get(toasts)[0];
			expect(t.dwellMs).toBe(4500);

			// 1 second elapsed — pause. Remaining should be ~3500ms.
			vi.advanceTimersByTime(1000);
			toasts.pauseDwell(t.id, t.dwellMs);

			// Hold paused for 10s (irrelevant to remaining calc).
			vi.advanceTimersByTime(10_000);

			// Resume — remaining is ~3500ms.
			toasts.resumeDwell(t.id);

			// At 3400ms still alive (just under remaining).
			vi.advanceTimersByTime(3400);
			expect(get(toasts).length).toBe(1);

			// At 3500ms+, dismissed.
			vi.advanceTimersByTime(200);
			expect(get(toasts).length).toBe(0);
		});

		it('double-pause is a safe no-op (second call does not corrupt remaining)', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.8 }));
			const t = get(toasts)[0];

			vi.advanceTimersByTime(500);
			toasts.pauseDwell(t.id, t.dwellMs);

			// Second pause should not throw or mutate state badly. The
			// implementation bails early because dwellTimers no longer
			// contains the id.
			expect(() => toasts.pauseDwell(t.id, t.dwellMs)).not.toThrow();

			// Still paused — advancing doesn't dismiss.
			vi.advanceTimersByTime(10_000);
			expect(get(toasts).length).toBe(1);
		});

		it('dismiss while paused clears the paused state (no zombie timer)', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.8 }));
			const t = get(toasts)[0];

			vi.advanceTimersByTime(500);
			toasts.pauseDwell(t.id, t.dwellMs);

			// Programmatic dismiss.
			toasts.dismiss(t.id);
			expect(get(toasts).length).toBe(0);

			// A later resume should be a no-op (no zombie re-schedule).
			toasts.resumeDwell(t.id);
			vi.advanceTimersByTime(10_000);
			expect(get(toasts).length).toBe(0);
		});

		it('resumeDwell on a non-paused id is a no-op', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.8 }));
			const t = get(toasts)[0];

			// Without pausing — resume is a no-op and must not schedule
			// anything new. The original timer is still ticking.
			toasts.resumeDwell(t.id);

			vi.advanceTimersByTime(4600);
			expect(get(toasts).length).toBe(0);
		});
	});

	// ---------------------------------------------------------------
	// Queue behavior
	// ---------------------------------------------------------------
	describe('queue behavior', () => {
		it('MAX_VISIBLE=4: creating a 5th toast evicts the oldest', async () => {
			const { toasts } = await loadToastStore();

			// Use MemoryPromoted (not throttled) to stack 5 toasts.
			for (let i = 0; i < 5; i++) {
				emit(makeEvent('MemoryPromoted', { new_retention: 0.5 + i * 0.01 }));
				// Small advance so each event has a distinct identity and no batch-merge.
				vi.advanceTimersByTime(10);
			}

			const list = get(toasts);
			expect(list.length).toBe(4);

			// IDs are assigned 1..5 in event-processing order. Store prepends,
			// so the queue is [id=5, id=4, id=3, id=2]; id=1 was evicted.
			const ids = list.map((t) => t.id);
			expect(ids).not.toContain(1);
			expect(ids).toContain(5);
		});

		it('clear() dismisses all toasts and cancels all timers', async () => {
			const { toasts } = await loadToastStore();

			emit([
				makeEvent('MemoryPromoted', { new_retention: 0.8 }),
				makeEvent('MemoryDemoted', { new_retention: 0.4 }),
			]);
			expect(get(toasts).length).toBe(2);

			toasts.clear();
			expect(get(toasts).length).toBe(0);

			// Advancing past the dwell must not re-fire anything (no zombie timers).
			vi.advanceTimersByTime(10_000);
			expect(get(toasts).length).toBe(0);
		});

		it('dismissing a specific id leaves the other toasts intact', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.8 }));
			vi.advanceTimersByTime(10);
			emit(makeEvent('MemoryDemoted', { new_retention: 0.4 }));

			const list = get(toasts);
			expect(list.length).toBe(2);
			const firstId = list[list.length - 1].id; // oldest

			toasts.dismiss(firstId);

			const remaining = get(toasts);
			expect(remaining.length).toBe(1);
			expect(remaining[0].id).not.toBe(firstId);
		});
	});

	// ---------------------------------------------------------------
	// Demo sequence
	// ---------------------------------------------------------------
	describe('fireDemoSequence', () => {
		it('schedules 4 toasts staggered by 800ms', async () => {
			const { toasts, fireDemoSequence } = await loadToastStore();

			fireDemoSequence();

			// t=0: nothing yet (all are setTimeout, even the first at i=0 * 800 = 0ms).
			// setTimeout(_, 0) still goes to the next tick under fake timers.
			expect(get(toasts).length).toBe(0);

			// Flush i=0.
			vi.advanceTimersByTime(1);
			expect(get(toasts).length).toBe(1);
			expect(get(toasts)[0].type).toBe('DreamCompleted');

			// i=1 at 800ms.
			vi.advanceTimersByTime(800);
			expect(get(toasts).length).toBe(2);
			expect(get(toasts)[0].type).toBe('ConnectionDiscovered');

			// i=2 at 1600ms.
			vi.advanceTimersByTime(800);
			expect(get(toasts).length).toBe(3);
			expect(get(toasts)[0].type).toBe('MemorySuppressed');

			// i=3 at 2400ms.
			vi.advanceTimersByTime(800);
			expect(get(toasts).length).toBe(4);
			expect(get(toasts)[0].type).toBe('ConsolidationCompleted');
		});
	});

	// ---------------------------------------------------------------
	// Toast shape sanity
	// ---------------------------------------------------------------
	describe('toast shape', () => {
		it('each toast has id, createdAt, color, dwellMs fields populated', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.9 }));

			const t = get(toasts)[0];
			expect(t.id).toBeTypeOf('number');
			expect(t.createdAt).toBeTypeOf('number');
			expect(t.color).toBeTypeOf('string');
			expect(t.dwellMs).toBeTypeOf('number');
			expect(t.color.length).toBeGreaterThan(0);
		});

		it('ids are strictly increasing across successive toasts', async () => {
			const { toasts } = await loadToastStore();

			emit(makeEvent('MemoryPromoted', { new_retention: 0.9 }));
			vi.advanceTimersByTime(10);
			emit(makeEvent('MemoryDemoted', { new_retention: 0.3 }));

			const list = get(toasts);
			expect(list.length).toBe(2);
			// Store prepends, so list[0] is newer = higher id.
			expect(list[0].id).toBeGreaterThan(list[1].id);
		});
	});
});
