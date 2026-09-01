import type { Memory } from '$types';
import type { Receipt, TraceDetail, TraceEvent } from '$lib/stores/api';
import {
	assertProvenance,
	type Provenance,
	type RouteEdge,
	type RouteEvent,
	type RouteNode,
	type RouteReceipt,
	type RouteSceneModel
} from '$lib/observatory/route-scene';

export type WitnessRole = 'retrieved' | 'path' | 'mutation' | 'suppressed';

export interface WitnessShard {
	id: string;
	label: string;
	content: string;
	role: WitnessRole;
	activation: number;
	retention: number;
	/** Exact normalized trace position at which this evidence entered the run. */
	traceTime: number;
	/** Stable receipt membership order — used for the helix, never randomized. */
	order: number;
	suppressed: boolean;
	mutated: boolean;
	provenance: Provenance;
}

export interface WitnessScene extends RouteSceneModel {
	organ: 'witness';
	runId: string | null;
	receiptId: string | null;
	shards: WitnessShard[];
	eventCount: number;
}

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function memorySource(id: string): Provenance {
	return { kind: 'memory', id };
}

function traceSource(runId: string | null, index: number, event: TraceEvent): Provenance {
	return { kind: 'trace', id: `${runId ?? 'none'}:${index}:${event.type}` };
}

function evidenceOrder(receipt: Receipt): string[] {
	return [
		...receipt.activation_path,
		...receipt.retrieved,
		...receipt.mutations.map((mutation) => mutation.id),
		...receipt.suppressed.map((suppression) => suppression.id)
	].filter((id, index, values) => Boolean(id) && values.indexOf(id) === index);
}

function roleFor(id: string, receipt: Receipt): WitnessRole {
	if (receipt.suppressed.some((suppression) => suppression.id === id)) return 'suppressed';
	if (receipt.mutations.some((mutation) => mutation.id === id)) return 'mutation';
	if (receipt.activation_path.includes(id)) return 'path';
	return 'retrieved';
}

function activationFor(id: string, events: TraceEvent[]): number {
	for (let index = events.length - 1; index >= 0; index -= 1) {
		const event = events[index];
		if (event.type === 'memory.retrieve' && typeof event.activation[id] === 'number') {
			return clamp01(event.activation[id]);
		}
	}
	return 0.48;
}

function labelFor(id: string, memory: Memory | undefined): string {
	if (!memory?.content) return `memory ${id.slice(0, 10)}`;
	const compact = memory.content.replace(/\s+/g, ' ').trim();
	return compact.length > 84 ? `${compact.slice(0, 81)}...` : compact;
}

function traceTimeFor(id: string, events: TraceEvent[]): number {
	if (!events.length) return 1;
	const eventIndex = events.findIndex((event) => {
		if (event.type === 'memory.retrieve') return event.ids.includes(id);
		if (event.type === 'memory.suppress' || event.type === 'memory.write') return event.id === id;
		if (event.type === 'contradiction.detected') return event.ids.includes(id);
		if (event.type === 'sanhedrin.veto') return event.evidenceIds.includes(id);
		if (event.type === 'dream.patch') return event.proposalIds.includes(id);
		return false;
	});
	return eventIndex < 0 ? 1 : clamp01((eventIndex + 1) / events.length);
}

/**
 * Creates the bounded, receipt-scoped model for the Witness Loom. This is
 * deliberately not a corpus graph: every rendered shard and filament names a
 * memory, trace event, or receipt that the selected agent run actually used.
 */
export function buildWitnessScene(
	detail: TraceDetail | null,
	receipt: Receipt | null,
	memoryById: ReadonlyMap<string, Memory>
): WitnessScene {
	const runId = detail?.runId ?? null;
	if (!receipt) {
		return {
			organ: 'witness',
			nodes: [],
			edges: [],
			events: [],
			receipts: [],
			scalars: { eventCount: detail?.events.length ?? 0, evidenceCount: 0 },
			alive: false,
			runId,
			receiptId: null,
			shards: [],
			eventCount: detail?.events.length ?? 0
		};
	}

	const eventList = detail?.events ?? [];
	const ids = evidenceOrder(receipt).slice(0, 64);
	const shards = ids.map((id, order): WitnessShard => {
		const memory = memoryById.get(id);
		const role = roleFor(id, receipt);
		return {
			id,
			label: labelFor(id, memory),
			content: memory?.content ?? '',
			role,
			activation: activationFor(id, eventList),
			retention: clamp01(memory?.retentionStrength ?? 0.5),
			traceTime: traceTimeFor(id, eventList),
			order,
			suppressed: role === 'suppressed',
			mutated: role === 'mutation',
			provenance: memorySource(id)
		};
	});

	const indexById = new Map(shards.map((shard, index) => [shard.id, index]));
	const nodes: RouteNode[] = shards.map((shard, index) => ({
		source: shard.provenance,
		index,
		label: shard.label,
		retention: shard.retention,
		activation: shard.activation,
		trust: receipt.trust_floor,
		suppression: shard.suppressed ? 1 : 0,
		tags: [shard.role],
		type: 'witness-shard'
	}));

	// The only persistent filaments are the ordered receipt activation path.
	// Similarity links and unrelated corpus edges do not qualify as proof.
	const edges: RouteEdge[] = receipt.activation_path.slice(1).flatMap((id, offset) => {
		const sourceId = receipt.activation_path[offset];
		const sourceIndex = indexById.get(sourceId);
		const targetIndex = indexById.get(id);
		if (sourceIndex === undefined || targetIndex === undefined) return [];
		return [{
			source: { kind: 'receipt' as const, id: `${receipt.receipt_id}:path:${offset}` },
			sourceIndex,
			targetIndex,
			weight: Math.max(nodes[sourceIndex].activation ?? 0, nodes[targetIndex].activation ?? 0),
			kind: 'receipt-path'
		}];
	});

	const events: RouteEvent[] = eventList.map((event, index) => {
		const eventIds = event.type === 'memory.retrieve'
			? event.ids
			: event.type === 'memory.suppress' || event.type === 'memory.write'
				? [event.id]
				: event.type === 'contradiction.detected'
					? event.ids
					: event.type === 'sanhedrin.veto'
						? event.evidenceIds
						: event.type === 'dream.patch'
							? event.proposalIds
							: [];
		const targetIndex = eventIds.map((id) => indexById.get(id)).find((value) => value !== undefined) ?? -1;
		return {
			source: traceSource(runId, index, event),
			type: event.type,
			targetIndex,
			frame: 18 + index * 24,
			energy: event.type === 'memory.retrieve' ? 0.86 : event.type === 'memory.suppress' ? 0.72 : 0.52
		};
	});

	const receipts: RouteReceipt[] = [{
		source: { kind: 'receipt', id: receipt.receipt_id },
		label: `receipt ${receipt.receipt_id.slice(0, 12)}`,
		nodeIndices: shards.map((_, index) => index)
	}];

	const scene: WitnessScene = {
		organ: 'witness',
		nodes,
		edges,
		events,
		receipts,
		scalars: {
			eventCount: eventList.length,
			evidenceCount: shards.length,
			pathLength: edges.length,
			trustFloor: receipt.trust_floor,
			suppressedCount: receipt.suppressed.length,
			mutationCount: receipt.mutations.length
		},
		alive: shards.length > 0,
		runId,
		receiptId: receipt.receipt_id,
		shards,
		eventCount: eventList.length
	};

	if (import.meta.env.DEV) assertProvenance(scene);
	return scene;
}
