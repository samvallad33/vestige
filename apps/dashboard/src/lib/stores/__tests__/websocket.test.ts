/**
 * Unit tests for the websocket store.
 *
 * Scope: pure-store methods and derived-store behavior that can be tested
 * without a real WebSocket connection. Connection lifecycle, reconnect
 * backoff, and live handler wiring are out of scope — those are integration
 * concerns.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { get } from 'svelte/store';

// Stub the global WebSocket BEFORE importing the store, so any accidental
// `connect()` path does not attempt a real network call and does not throw.
// The FakeWS also captures the most-recently-constructed instance so tests
// that want to drive `onmessage` (to exercise the Heartbeat branch of the
// internal update handler) can do so.
class FakeWS {
	static last: FakeWS | null = null;
	static OPEN = 1;
	readyState = 0;
	onopen: ((ev?: unknown) => void) | null = null;
	onclose: ((ev?: unknown) => void) | null = null;
	onmessage: ((ev: { data: string }) => void) | null = null;
	onerror: ((ev?: unknown) => void) | null = null;
	constructor(public url: string) {
		FakeWS.last = this;
	}
	close() {
		/* no-op */
	}
	addEventListener() {
		/* no-op */
	}
	removeEventListener() {
		/* no-op */
	}
}
vi.stubGlobal('WebSocket', FakeWS);

import {
	websocket,
	eventFeed,
	isConnected,
	memoryCount,
	avgRetention,
	suppressedCount,
	uptimeSeconds,
	heartbeat,
	formatUptime,
} from '../websocket';
import type { VestigeEvent } from '$types';

const MAX_EVENTS = 200;

function makeEvent(
	type: VestigeEvent['type'] = 'MemoryCreated',
	data: Record<string, unknown> = {}
): VestigeEvent {
	return { type, data };
}

function makeHeartbeat(data: Record<string, unknown> = {}): VestigeEvent {
	return {
		type: 'Heartbeat',
		data: {
			memory_count: 0,
			avg_retention: 0,
			suppressed_count: 0,
			uptime_secs: 0,
			...data,
		},
	};
}

/**
 * Helper: drive a heartbeat into the store via the internal onmessage path.
 * We cannot reach `update()` directly, and `injectEvent()` deliberately
 * does NOT treat heartbeats specially, so the only way to populate
 * `lastHeartbeat` is to route through the WebSocket handler.
 */
function deliverHeartbeat(hb: VestigeEvent) {
	// Disconnect to reset any prior state, then connect to install handlers
	// on a fresh FakeWS instance whose onmessage we can invoke.
	websocket.disconnect();
	websocket.connect('ws://test.invalid/ws');
	const ws = FakeWS.last;
	if (!ws || !ws.onmessage) throw new Error('FakeWS onmessage not wired');
	ws.onmessage({ data: JSON.stringify(hb) });
}

beforeEach(() => {
	// Reset the events array between tests; lastHeartbeat is explicitly left
	// alone here because `clearEvents()` preserves it (that is itself tested
	// below). For the derived-store defaults tests we call disconnect() to
	// fully reset the store.
	websocket.clearEvents();
});

// ---------------------------------------------------------------------------
// injectEvent
// ---------------------------------------------------------------------------

describe('injectEvent', () => {
	it('adds a single event at index 0', () => {
		const evt = makeEvent('MemoryCreated', { id: 'a' });
		websocket.injectEvent(evt);
		const feed = get(eventFeed);
		expect(feed.length).toBe(1);
		expect(feed[0]).toEqual(evt);
	});

	it('prepends: newest injected ends up at index 0', () => {
		const first = makeEvent('MemoryCreated', { id: 'first' });
		const second = makeEvent('MemoryUpdated', { id: 'second' });
		const third = makeEvent('MemoryDeleted', { id: 'third' });
		websocket.injectEvent(first);
		websocket.injectEvent(second);
		websocket.injectEvent(third);
		const feed = get(eventFeed);
		expect(feed.length).toBe(3);
		expect(feed[0]).toEqual(third);
		expect(feed[1]).toEqual(second);
		expect(feed[2]).toEqual(first);
	});

	it('caps the events array at MAX_EVENTS (200)', () => {
		for (let i = 0; i < MAX_EVENTS + 50; i++) {
			websocket.injectEvent(makeEvent('MemoryCreated', { seq: i }));
		}
		const feed = get(eventFeed);
		expect(feed.length).toBe(MAX_EVENTS);
	});

	it('evicts the oldest entry when at capacity (FIFO drop)', () => {
		// Fill to exactly capacity, then push one more: seq=0 should be gone.
		for (let i = 0; i < MAX_EVENTS; i++) {
			websocket.injectEvent(makeEvent('MemoryCreated', { seq: i }));
		}
		websocket.injectEvent(makeEvent('MemoryCreated', { seq: 999 }));
		const feed = get(eventFeed);
		expect(feed.length).toBe(MAX_EVENTS);
		expect(feed[0].data.seq).toBe(999);
		// Oldest (seq=0) evicted; tail is now the prior second-oldest (seq=1).
		expect(feed[feed.length - 1].data.seq).toBe(1);
		expect(feed.some((e) => e.data.seq === 0)).toBe(false);
	});

	it('triggers the eventFeed derived store to emit on each injection', () => {
		const observed: number[] = [];
		const unsub = eventFeed.subscribe((events) => {
			observed.push(events.length);
		});
		// Initial subscription fires once with current length (0 after beforeEach).
		const initialEmitCount = observed.length;
		websocket.injectEvent(makeEvent());
		websocket.injectEvent(makeEvent());
		unsub();
		// Two injections should produce two additional emits beyond the initial one.
		expect(observed.length).toBe(initialEmitCount + 2);
		expect(observed[observed.length - 1]).toBe(2);
	});

	it('does NOT treat Heartbeat-typed events specially when injected', () => {
		// Documented behavior: injectEvent is a raw prepend. Only the real
		// onmessage handler branches on type === 'Heartbeat'. If a caller
		// injects a Heartbeat, it lands in the events array, and lastHeartbeat
		// is untouched. Callers who want a heartbeat-like derived-store update
		// must route through the WebSocket handler instead.
		websocket.disconnect(); // reset lastHeartbeat to null
		const hb = makeHeartbeat({ memory_count: 42 });
		websocket.injectEvent(hb);
		const feed = get(eventFeed);
		expect(feed.length).toBe(1);
		expect(feed[0]).toEqual(hb);
		// memoryCount still 0 because lastHeartbeat was never written.
		expect(get(memoryCount)).toBe(0);
		expect(get(heartbeat)).toBeNull();
	});
});

// ---------------------------------------------------------------------------
// Derived store defaults (no heartbeat yet)
// ---------------------------------------------------------------------------

describe('derived stores — defaults with no heartbeat', () => {
	beforeEach(() => {
		// Full reset so lastHeartbeat is null.
		websocket.disconnect();
	});

	it('isConnected is false after disconnect', () => {
		expect(get(isConnected)).toBe(false);
	});

	it('heartbeat is null when no heartbeat has arrived', () => {
		expect(get(heartbeat)).toBeNull();
	});

	it('memoryCount returns 0 when no heartbeat has arrived', () => {
		expect(get(memoryCount)).toBe(0);
	});

	it('avgRetention returns 0 when no heartbeat has arrived', () => {
		expect(get(avgRetention)).toBe(0);
	});

	it('suppressedCount returns 0 when no heartbeat has arrived', () => {
		expect(get(suppressedCount)).toBe(0);
	});

	it('uptimeSeconds returns 0 when no heartbeat has arrived', () => {
		expect(get(uptimeSeconds)).toBe(0);
	});
});

// ---------------------------------------------------------------------------
// Derived stores after heartbeat
// ---------------------------------------------------------------------------

describe('derived stores — after heartbeat delivery', () => {
	it('memoryCount, avgRetention, suppressedCount, uptimeSeconds all update', () => {
		deliverHeartbeat(
			makeHeartbeat({
				memory_count: 123,
				avg_retention: 0.74,
				suppressed_count: 5,
				uptime_secs: 3661,
			})
		);
		expect(get(memoryCount)).toBe(123);
		expect(get(avgRetention)).toBeCloseTo(0.74);
		expect(get(suppressedCount)).toBe(5);
		expect(get(uptimeSeconds)).toBe(3661);
		const hb = get(heartbeat);
		expect(hb).not.toBeNull();
		expect(hb?.type).toBe('Heartbeat');
	});

	it('heartbeat events do NOT enter the events array (handled by onmessage)', () => {
		websocket.disconnect();
		deliverHeartbeat(makeHeartbeat({ memory_count: 1 }));
		expect(get(eventFeed).length).toBe(0);
	});

	it('non-heartbeat events delivered via onmessage enter the events array', () => {
		websocket.disconnect();
		websocket.connect('ws://test.invalid/ws');
		const ws = FakeWS.last!;
		ws.onmessage!({ data: JSON.stringify(makeEvent('MemoryCreated', { id: 'x' })) });
		const feed = get(eventFeed);
		expect(feed.length).toBe(1);
		expect(feed[0].type).toBe('MemoryCreated');
	});
});

// ---------------------------------------------------------------------------
// clearEvents
// ---------------------------------------------------------------------------

describe('clearEvents', () => {
	it('empties the events array', () => {
		websocket.injectEvent(makeEvent());
		websocket.injectEvent(makeEvent());
		expect(get(eventFeed).length).toBe(2);
		websocket.clearEvents();
		expect(get(eventFeed).length).toBe(0);
	});

	it('preserves lastHeartbeat (does NOT clear it)', () => {
		deliverHeartbeat(makeHeartbeat({ memory_count: 77 }));
		expect(get(memoryCount)).toBe(77);
		websocket.injectEvent(makeEvent('MemoryCreated'));
		websocket.clearEvents();
		expect(get(eventFeed).length).toBe(0);
		// lastHeartbeat untouched, so memoryCount still reflects the heartbeat.
		expect(get(memoryCount)).toBe(77);
		expect(get(heartbeat)).not.toBeNull();
	});
});

// ---------------------------------------------------------------------------
// formatUptime
// ---------------------------------------------------------------------------

describe('formatUptime', () => {
	it("returns '—' for negative input", () => {
		expect(formatUptime(-1)).toBe('—');
	});

	it("returns '—' for non-finite input (NaN, Infinity)", () => {
		expect(formatUptime(NaN)).toBe('—');
		expect(formatUptime(Infinity)).toBe('—');
		expect(formatUptime(-Infinity)).toBe('—');
	});

	it("returns '0s' for 0 (boundary: non-negative, all units zero)", () => {
		// secs=0 is NOT < 0, so it falls through to the '${s}s' branch.
		expect(formatUptime(0)).toBe('0s');
	});

	it('seconds-only branch: 47s', () => {
		expect(formatUptime(47)).toBe('47s');
	});

	it('seconds boundary: 59s → "59s"', () => {
		expect(formatUptime(59)).toBe('59s');
	});

	it('minute boundary: 60s → "1m" (no trailing 0s)', () => {
		expect(formatUptime(60)).toBe('1m');
	});

	it('minutes + seconds: 190s → "3m 10s"', () => {
		expect(formatUptime(190)).toBe('3m 10s');
	});

	it('hour boundary minus one: 3599s → "59m 59s"', () => {
		expect(formatUptime(3599)).toBe('59m 59s');
	});

	it('hour boundary: 3600s → "1h" (no trailing 0m)', () => {
		expect(formatUptime(3600)).toBe('1h');
	});

	it('hours + minutes: 11520s (3h 12m) → "3h 12m"', () => {
		expect(formatUptime(3 * 3600 + 12 * 60)).toBe('3h 12m');
	});

	it('day boundary minus one: 86399s → "23h 59m"', () => {
		// Two-most-significant-units rule: hours + minutes, seconds dropped.
		expect(formatUptime(86399)).toBe('23h 59m');
	});

	it('day boundary: 86400s → "1d" (no trailing 0h)', () => {
		expect(formatUptime(86400)).toBe('1d');
	});

	it('days + hours: 4d 2h → "4d 2h" (minutes dropped)', () => {
		expect(formatUptime(4 * 86400 + 2 * 3600 + 37 * 60)).toBe('4d 2h');
	});
});
