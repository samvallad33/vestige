import {
	assertProvenance,
	type Provenance,
	type RouteEvent,
	type RouteNode,
	type RouteReceipt,
	type RouteSceneModel
} from '$lib/observatory/route-scene';
import type { Receipt, TraceDetail, TraceEvent } from '$lib/stores/api';

export type BlackboxLane = 'tool' | 'retrieve' | 'suppress' | 'write' | 'veto' | 'contradiction' | 'dream';

export interface ActivationPair {
	id: string;
	activation: number;
}

export interface BlackboxTraceImpulse {
	index: number;
	id: string;
	type: TraceEvent['type'];
	lane: BlackboxLane;
	runId: string;
	at: number;
	label: string;
	summary: string;
	memoryIds: string[];
	activationPairs: ActivationPair[];
	confidence: number;
	provenance: Provenance;
	raw: TraceEvent;
}

export interface BlackboxScene extends RouteSceneModel {
	organ: 'blackbox';
	runId: string | null;
	traceEvents: BlackboxTraceImpulse[];
	visibleEventCount: number;
	selectedIndex: number;
	startedAt: number;
	lastAt: number;
	durationMs: number;
	receiptRows: Receipt[];
}

type TraceInput = TraceDetail | (TraceDetail & { summary?: Record<string, unknown> | null }) | null | undefined;

function clamp01(v: number): number {
	return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

function num(v: unknown, fallback = 0): number {
	return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function text(v: unknown, fallback = ''): string {
	return typeof v === 'string' ? v : v == null ? fallback : String(v);
}

function traceSource(runId: string | null, index: number, eventType: string): Provenance {
	return { kind: 'trace', id: `${runId ?? 'trace:none'}:event:${index}:${eventType}` };
}

function scalarSource(name: string, value: number): Provenance {
	return { kind: 'scalar', id: `blackbox.${name}`, scalar: { name, value } };
}

function eventSource(runId: string | null, index: number, eventType: string): Provenance {
	return { kind: 'event', id: `${runId ?? 'trace:none'}:event:${index}:${eventType}` };
}

function memorySource(id: string): Provenance {
	return { kind: 'memory', id };
}

function laneFor(type: TraceEvent['type']): BlackboxLane {
	switch (type) {
		case 'mcp.call': return 'tool';
		case 'memory.retrieve': return 'retrieve';
		case 'memory.suppress': return 'suppress';
		case 'memory.write': return 'write';
		case 'sanhedrin.veto': return 'veto';
		case 'contradiction.detected': return 'contradiction';
		case 'dream.patch': return 'dream';
	}
}

function eventMemoryIds(ev: TraceEvent): string[] {
	switch (ev.type) {
		case 'memory.retrieve': return ev.ids;
		case 'memory.suppress': return [ev.id];
		case 'memory.write': return [ev.id];
		case 'contradiction.detected': return ev.ids;
		case 'sanhedrin.veto': return ev.evidenceIds;
		case 'dream.patch': return ev.proposalIds;
		case 'mcp.call': return [];
	}
}

function eventLabel(ev: TraceEvent): string {
	switch (ev.type) {
		case 'mcp.call': return ev.tool;
		case 'memory.retrieve': return `${ev.ids.length} memories retrieved`;
		case 'memory.suppress': return `suppressed ${ev.id.slice(0, 8)}`;
		case 'memory.write': return `wrote ${ev.id.slice(0, 8)}`;
		case 'contradiction.detected': return `contradiction ${ev.ids.join(' ↔ ')}`;
		case 'sanhedrin.veto': return 'Sanhedrin veto';
		case 'dream.patch': return `${ev.proposalIds.length} dream proposals`;
	}
}

function eventSummary(ev: TraceEvent): string {
	switch (ev.type) {
		case 'mcp.call': return `MCP tool ${ev.tool} called; args hash ${ev.argsHash.slice(0, 12)}`;
		case 'memory.retrieve': return `Retrieved ${ev.ids.length} memories with activation map`;
		case 'memory.suppress': return `Suppressed ${ev.id}: ${ev.reason}`;
		case 'memory.write': return `Memory write ${ev.id} from ${ev.source}`;
		case 'contradiction.detected': return ev.detail;
		case 'sanhedrin.veto': return `${Math.round(clamp01(ev.confidence) * 100)}% veto confidence: ${ev.claim}`;
		case 'dream.patch': return `Dream patch proposals: ${ev.proposalIds.join(', ')}`;
	}
}

function activationPairs(ev: TraceEvent): ActivationPair[] {
	if (ev.type !== 'memory.retrieve') return [];
	return Object.entries(ev.activation ?? {})
		.map(([id, activation]) => ({ id, activation: clamp01(activation) }))
		.sort((a, b) => b.activation - a.activation);
}

function confidenceFor(ev: TraceEvent): number {
	if (ev.type === 'sanhedrin.veto') return clamp01(ev.confidence);
	if (ev.type === 'memory.retrieve') {
		const values = Object.values(ev.activation ?? {});
		return values.length ? clamp01(values.reduce((a, b) => a + b, 0) / values.length) : 0.45;
	}
	if (ev.type === 'memory.suppress') return 0.78;
	if (ev.type === 'memory.write') return 0.72;
	if (ev.type === 'contradiction.detected') return 0.82;
	if (ev.type === 'dream.patch') return 0.52;
	return 0.62;
}

export function normalizeBlackboxScene(input: TraceInput, receipts: Receipt[] = [], selectedIndex?: number): BlackboxScene {
	const runId = input?.runId ?? null;
	const eventsRaw = input?.events ?? [];
	const summary = input?.summary ?? null;
	const startedAt = num(summary?.startedAt, eventsRaw[0]?.at ?? 0);
	const lastAt = num(summary?.lastAt, eventsRaw[eventsRaw.length - 1]?.at ?? startedAt);
	const maxVisible = eventsRaw.length ? eventsRaw.length - 1 : 0;
	const visibleTo = Math.max(0, Math.min(maxVisible, selectedIndex ?? maxVisible));
	const visibleEventCount = eventsRaw.length ? visibleTo + 1 : 0;

	const traceEvents: BlackboxTraceImpulse[] = eventsRaw.map((ev, index) => ({
		index,
		id: `${runId ?? ev.runId}:event:${index}:${ev.type}`,
		type: ev.type,
		lane: laneFor(ev.type),
		runId: ev.runId,
		at: ev.at,
		label: eventLabel(ev),
		summary: eventSummary(ev),
		memoryIds: eventMemoryIds(ev),
		activationPairs: activationPairs(ev),
		confidence: confidenceFor(ev),
		provenance: traceSource(runId, index, ev.type),
		raw: ev
	}));

	const memoryIndex = new Map<string, RouteNode>();
	for (const impulse of traceEvents) {
		for (const id of impulse.memoryIds) {
			if (!id || memoryIndex.has(id)) continue;
			const activation = impulse.activationPairs.find((p) => p.id === id)?.activation ?? 0;
			memoryIndex.set(id, {
				source: memorySource(id),
				index: memoryIndex.size,
				label: id.slice(0, 12),
				retention: Math.max(0.25, activation),
				activation,
				trust: impulse.type === 'memory.suppress' ? 0.2 : Math.max(0.35, impulse.confidence),
				suppression: impulse.type === 'memory.suppress' ? 1 : 0,
				tags: [impulse.lane, impulse.type],
				type: 'trace-memory'
			});
		}
	}
	const nodes = [...memoryIndex.values()];

	const routeEvents: RouteEvent[] = traceEvents.slice(0, visibleEventCount).map((ev) => ({
		source: eventSource(runId, ev.index, ev.type),
		type: ev.type,
		targetIndex: ev.memoryIds.length ? (memoryIndex.get(ev.memoryIds[0])?.index ?? -1) : -1,
		frame: ev.index * 34 + 18,
		energy: Math.max(0.18, ev.confidence)
	}));

	const edges = traceEvents.slice(0, visibleEventCount).flatMap((ev) => {
		const ids = ev.memoryIds.filter((id) => memoryIndex.has(id));
		if (ids.length < 2) return [];
		return ids.slice(1).map((id, i) => ({
			source: { kind: 'pair' as const, id: `${ev.id}:pair:${ids[0]}:${id}` },
			sourceIndex: memoryIndex.get(ids[0])!.index,
			targetIndex: memoryIndex.get(id)!.index,
			weight: ev.activationPairs[i + 1]?.activation ?? ev.confidence,
			kind: ev.type
		}));
	});

	const receiptRows = receipts ?? [];
	const routeReceipts: RouteReceipt[] = receiptRows.map((receipt) => {
		const ids = [...new Set([...(receipt.retrieved ?? []), ...(receipt.activation_path ?? []), ...(receipt.mutations ?? []).map((m) => m.id)])];
		return {
			source: { kind: 'receipt', id: receipt.receipt_id },
			label: `receipt ${receipt.receipt_id.slice(0, 10)}`,
			nodeIndices: ids.map((id) => memoryIndex.get(id)?.index).filter((i): i is number => typeof i === 'number')
		};
	});

	const scene: BlackboxScene = {
		organ: 'blackbox',
		nodes,
		edges,
		events: routeEvents,
		receipts: routeReceipts,
		scalars: {
			eventCount: eventsRaw.length,
			visibleEventCount,
			retrievedCount: num(summary?.retrievedCount, traceEvents.filter((e) => e.type === 'memory.retrieve').length),
			suppressedCount: num(summary?.suppressedCount, traceEvents.filter((e) => e.type === 'memory.suppress').length),
			writeCount: num(summary?.writeCount, traceEvents.filter((e) => e.type === 'memory.write').length),
			vetoCount: num(summary?.vetoCount, traceEvents.filter((e) => e.type === 'sanhedrin.veto').length),
			durationMs: Math.max(0, lastAt - startedAt),
			receiptCount: routeReceipts.length
		},
		alive: eventsRaw.length > 0,
		runId,
		traceEvents,
		visibleEventCount,
		selectedIndex: visibleTo,
		startedAt,
		lastAt,
		durationMs: Math.max(0, lastAt - startedAt),
		receiptRows
	};

	if (!scene.alive) {
		scene.receipts = receiptRows.map((receipt) => ({
			source: { kind: 'receipt', id: receipt.receipt_id },
			label: `receipt ${receipt.receipt_id.slice(0, 10)}`,
			nodeIndices: []
		}));
		scene.scalars.eventCount = 0;
		scene.scalars.visibleEventCount = 0;
		void scalarSource('empty', 0);
	}

	if (import.meta.env.DEV) assertProvenance(scene);
	return scene;
}
