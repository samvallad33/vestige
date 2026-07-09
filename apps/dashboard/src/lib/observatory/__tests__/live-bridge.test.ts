/**
 * LiveBridge — the field's nervous system. These tests machine-prove the
 * behaviors that are hard to screenshot in a headless preview (the multi-second
 * dream storm, per-node decay, event dedup) by driving the bridge with a mock
 * engine + renderer and asserting the params lanes / renderer calls.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { LiveBridge } from '../live-bridge';
import { PARAM_IDX, PARAMS_FLOATS, LIVE_KIND } from '../types';
import { buildObservatoryGraph } from '../graph-upload';
import type { GraphResponse, GraphNode, VestigeEvent } from '$types';

// --- Mocks: an engine is just a params array + monotonic frame + wall clock;
// a renderer just records the buffers it was asked to upload. ---

class MockEngine {
	params = new Float32Array(PARAMS_FLOATS);
	private _frame = 0;
	// Fixed wall clock — 2026-07-01, AFTER the test nodes' lastAccessed dates so
	// elapsed is positive and the FSRS curve actually decays.
	private _now = Date.parse('2026-07-01T00:00:00Z');
	passes: unknown[] = [];
	// No GPU in the test env — the firewall renderer's upload() no-ops on a null
	// device, so arming the firewall stays a pure params-lane assertion.
	get gpuDevice() {
		return null;
	}
	get paramsBuffer() {
		return null;
	}
	addPass(p: unknown) {
		this.passes.push(p);
	}
	get totalFrames() {
		return this._frame;
	}
	get wallNowMs() {
		return this._now;
	}
	advance(frames: number) {
		this._frame += frames;
	}
	setNow(ms: number) {
		this._now = ms;
	}
}

class MockRenderer {
	graph: ReturnType<typeof buildObservatoryGraph> | null = null;
	setEdgesCalls = 0;
	lastEdgeCount = 0;
	retentionUploads: Float32Array[] = [];
	rearmCalls = 0;
	setEdges(edges: unknown[]) {
		this.setEdgesCalls++;
		this.lastEdgeCount = edges.length;
	}
	uploadLiveRetention(data: Float32Array) {
		this.retentionUploads.push(data.slice());
	}
}

function gnode(partial: Partial<GraphNode> & { id: string }): GraphNode {
	return {
		label: partial.id,
		type: 'note',
		retention: 0.8,
		tags: [],
		createdAt: '2026-06-01T00:00:00Z',
		updatedAt: '2026-06-01T00:00:00Z',
		isCenter: false,
		...partial
	};
}

function makeGraph(): ReturnType<typeof buildObservatoryGraph> {
	const nodes: GraphNode[] = [
		gnode({ id: 'a', isCenter: true, stability: 5, lastAccessed: '2026-06-20T00:00:00Z' }),
		gnode({ id: 'b', stability: 0.2, lastAccessed: '2026-06-25T00:00:00Z' }),
		gnode({ id: 'c', stability: 3, lastAccessed: '2026-06-10T00:00:00Z' }),
		gnode({ id: 'd', stability: 1, lastAccessed: '2026-06-28T00:00:00Z' })
	];
	const resp: GraphResponse = {
		nodes,
		edges: [{ source: 'a', target: 'b', weight: 0.5, type: 'semantic' }],
		center_id: 'a',
		depth: 2,
		nodeCount: nodes.length,
		edgeCount: 1
	};
	return buildObservatoryGraph(resp);
}

function ev(type: string, data: Record<string, unknown>, tsMs: number): VestigeEvent {
	return { type, data: { ...data, timestamp: new Date(tsMs).toISOString() } } as VestigeEvent;
}

function makeBridge() {
	const engine = new MockEngine();
	const renderer = new MockRenderer();
	const graph = makeGraph();
	renderer.graph = graph;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const bridge = new LiveBridge({ engine: engine as any, renderer: renderer as any, graph, seed: 'test' });
	return { engine, renderer, graph, bridge };
}

describe('LiveBridge', () => {
	let base: number;
	beforeEach(() => {
		base = 1_000_000_000_000;
	});

	it('detects live FSRS decay data and computes per-node retrievability', () => {
		const { engine, renderer, bridge } = makeBridge();
		expect(bridge.liveDecayAvailable).toBe(true);
		// Advance past the decay throttle and drain — retention should upload.
		engine.advance(10);
		bridge.drain(engine.totalFrames);
		expect(renderer.retentionUploads.length).toBeGreaterThan(0);
		const r = renderer.retentionUploads.at(-1)!;
		// Node 'b' (stability 0.2) must have decayed far more than 'a' (stability 5).
		// indices: buildObservatoryGraph sorts center first then by id → a,b,c,d
		expect(r[1]).toBeLessThan(r[0]);
		expect(r[1]).toBeGreaterThanOrEqual(0);
		expect(r[0]).toBeLessThanOrEqual(1);
	});

	it('ignores the pre-mount backlog (seeds the watermark on first ingest)', () => {
		const { engine, bridge } = makeBridge();
		// A suppress that happened BEFORE mount must NOT fire.
		bridge.ingest([ev('MemorySuppressed', { id: 'b', estimated_cascade: 3 }, base - 5000)]);
		bridge.drain(engine.totalFrames);
		expect(engine.params[PARAM_IDX.liveKind]).toBe(LIVE_KIND.none);
	});

	it('arms the contradiction firewall on a NEW MemorySuppressed for an in-field node', () => {
		const { engine, bridge } = makeBridge();
		// First ingest seeds the watermark (backlog) …
		bridge.ingest([ev('Heartbeat', {}, base)]);
		// … then a NEWER suppress fires the firewall.
		bridge.ingest([ev('MemorySuppressed', { id: 'b', estimated_cascade: 3 }, base + 1000)]);
		engine.advance(5);
		bridge.drain(engine.totalFrames);
		expect(engine.params[PARAM_IDX.liveKind]).toBe(LIVE_KIND.firewall);
		expect(engine.params[PARAM_IDX.liveEnergy]).toBeGreaterThan(0);
	});

	it('appends real ConnectionDiscovered edges and flushes ONE setEdges per frame', () => {
		const { engine, renderer, bridge } = makeBridge();
		bridge.ingest([ev('Heartbeat', {}, base)]);
		// Two new connections in the same burst → coalesced to one setEdges.
		bridge.ingest([
			ev('ConnectionDiscovered', { source_id: 'c', target_id: 'd', weight: 0.7, connection_type: 'causal' }, base + 2000),
			ev('ConnectionDiscovered', { source_id: 'b', target_id: 'c', weight: 0.6, connection_type: 'semantic' }, base + 1500)
		]);
		const before = renderer.setEdgesCalls;
		bridge.drain(engine.totalFrames);
		expect(renderer.setEdgesCalls).toBe(before + 1); // ONE flush
		// started with 1 edge (a-b), +2 new = 3
		expect(renderer.lastEdgeCount).toBe(3);
	});

	it('dedupes a re-discovered edge (no duplicate append)', () => {
		const { engine, renderer, bridge } = makeBridge();
		bridge.ingest([ev('Heartbeat', {}, base)]);
		bridge.ingest([ev('ConnectionDiscovered', { source_id: 'c', target_id: 'd' }, base + 1000)]);
		bridge.drain(engine.totalFrames);
		const afterFirst = renderer.lastEdgeCount;
		// same edge, reversed direction, later ts → must be a no-op
		bridge.ingest([ev('ConnectionDiscovered', { source_id: 'd', target_id: 'c' }, base + 2000)]);
		bridge.drain(engine.totalFrames);
		expect(renderer.lastEdgeCount).toBe(afterFirst);
	});

	it('fires the dream storm on DreamCompleted even when DreamStarted was evicted', () => {
		const { engine, bridge } = makeBridge();
		bridge.ingest([ev('Heartbeat', {}, base)]);
		// Only DreamCompleted survives the 200-event ring — the storm must STILL fire.
		bridge.ingest([ev('DreamCompleted', { connections_found: 1145, memories_replayed: 50 }, base + 3000)]);
		engine.advance(60); // ~1s in
		bridge.drain(engine.totalFrames);
		expect(engine.params[PARAM_IDX.liveKind]).toBe(LIVE_KIND.dreamStorm);
		expect(engine.params[PARAM_IDX.liveEnergy]).toBeGreaterThan(0.3);
	});

	it('the dream storm holds for seconds then settles to calm', () => {
		const { engine, bridge } = makeBridge();
		bridge.ingest([ev('Heartbeat', {}, base)]);
		bridge.ingest([ev('DreamStarted', { memory_count: 50 }, base + 1000)]);
		engine.advance(120); // 2s — mid-storm
		bridge.drain(engine.totalFrames);
		expect(engine.params[PARAM_IDX.liveKind]).toBe(LIVE_KIND.dreamStorm);
		const midEnergy = engine.params[PARAM_IDX.liveEnergy];
		expect(midEnergy).toBeGreaterThan(0);
		// Far past the window → settled.
		engine.advance(700);
		bridge.drain(engine.totalFrames);
		expect(engine.params[PARAM_IDX.liveKind]).toBe(LIVE_KIND.none);
		expect(engine.params[PARAM_IDX.liveEnergy]).toBe(0);
	});

	it('the forward-projection scrubber decays the field further', () => {
		const engine = new MockEngine();
		const renderer = new MockRenderer();
		const graph = makeGraph();
		renderer.graph = graph;
		let proj = 0;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const bridge = new LiveBridge({ engine: engine as any, renderer: renderer as any, graph, seed: 't', projectionDays: () => proj });
		engine.advance(10);
		bridge.drain(engine.totalFrames);
		const atNow = renderer.retentionUploads.at(-1)!.slice();
		proj = 180;
		bridge.refreshDecay();
		const atPlus180 = renderer.retentionUploads.at(-1)!;
		// every node with real FSRS state must be ≤ its now-value (more forgotten)
		for (let i = 0; i < atNow.length; i++) {
			expect(atPlus180[i]).toBeLessThanOrEqual(atNow[i] + 1e-6);
		}
		// and at least one strictly lower (the field visibly decayed)
		expect(atPlus180.some((v, i) => v < atNow[i] - 1e-3)).toBe(true);
	});
});
