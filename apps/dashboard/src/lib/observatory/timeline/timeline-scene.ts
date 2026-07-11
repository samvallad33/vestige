import {
	assertProvenance,
	type Provenance,
	type RouteEdge,
	type RouteEvent,
	type RouteNode,
	type RouteReceipt,
	type RouteSceneModel
} from '$lib/observatory/route-scene';
import type { Memory, MemoryAuditEvent, TimelineDay } from '$types';

export interface TimelineRing {
	id: string;
	date: string;
	index: number;
	count: number;
	radius: number;
	retention: number;
	updatedCount: number;
	suppressedCount: number;
	memoryIndices: number[];
	provenance: Provenance;
}

export interface TimelineCell {
	id: string;
	memoryId: string;
	day: string;
	dayIndex: number;
	nodeIndex: number;
	angle: number;
	radius: number;
	retention: number;
	validFrom: string;
	transactionAt: string;
	suppressed: boolean;
	rewritten: boolean;
	label: string;
	provenance: Provenance;
}

export interface TimelineAuditSlice {
	memoryId: string;
	events: MemoryAuditEvent[];
}

export interface TimelineScene extends RouteSceneModel {
	organ: 'timeline';
	rings: TimelineRing[];
	cells: TimelineCell[];
	audits: TimelineAuditSlice[];
	raw: {
		days: TimelineDay[];
		audits: Record<string, MemoryAuditEvent[]>;
	};
}

function text(v: unknown, fallback = ''): string {
	return typeof v === 'string' ? v : v == null ? fallback : String(v);
}

function num(v: unknown, fallback = 0): number {
	return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function clamp01(v: number): number {
	return Math.max(0, Math.min(1, v));
}

function source(kind: Provenance['kind'], id: string, scalar?: Provenance['scalar']): Provenance {
	return scalar ? { kind, id, scalar } : { kind, id: id || `${kind}:unknown` };
}

function scalarSource(name: string, value: number): Provenance {
	return { kind: 'scalar', id: `timeline.${name}`, scalar: { name, value } };
}

function memoryRetention(memory: Memory): number {
	return clamp01(num(memory.retentionStrength, 0));
}

function memoryTrust(memory: Memory): number {
	return clamp01(num(memory.combinedScore ?? memory.retentionStrength, memoryRetention(memory)));
}

function auditEventsFor(memory: Memory, audits: Record<string, MemoryAuditEvent[]>): MemoryAuditEvent[] {
	return audits[memory.id] ?? [];
}

function hasAction(events: MemoryAuditEvent[], action: MemoryAuditEvent['action']): boolean {
	return events.some((event) => event.action === action);
}

function eventTimestamp(memory: Memory, events: MemoryAuditEvent[], action: MemoryAuditEvent['action'], fallback: string): string {
	return text(events.find((event) => event.action === action)?.timestamp, fallback);
}

export function normalizeTimelineScene(input: {
	days?: TimelineDay[];
	totalMemories?: number;
	audits?: Record<string, MemoryAuditEvent[]>;
}): TimelineScene {
	const days = input.days ?? [];
	const audits = input.audits ?? {};
	const nodes: RouteNode[] = [];
	const cells: TimelineCell[] = [];
	const rings: TimelineRing[] = [];
	const edges: RouteEdge[] = [];
	const receipts: RouteReceipt[] = [];
	const events: RouteEvent[] = [];

	const nonEmptyDays = days.filter((day) => day.count > 0 || day.memories.length > 0);
	const ringCount = Math.max(1, nonEmptyDays.length);
	const maxCount = Math.max(1, ...nonEmptyDays.map((day) => day.count || day.memories.length));

	nonEmptyDays.forEach((day, dayIndex) => {
		const ringRadius = 0.16 + (dayIndex / Math.max(1, ringCount - 1)) * 0.70;
		const dayMemories = day.memories ?? [];
		const memoryIndices: number[] = [];
		let retentionSum = 0;
		let updatedCount = 0;
		let suppressedCount = 0;
		dayMemories.forEach((memory, memoryIndex) => {
			const audit = auditEventsFor(memory, audits);
			const retention = memoryRetention(memory);
			const rewritten = text(memory.updatedAt) !== text(memory.createdAt) || hasAction(audit, 'edited') || hasAction(audit, 'reconsolidated');
			const suppressed = hasAction(audit, 'suppressed') || num((memory as unknown as Record<string, unknown>).suppression_count, 0) > 0;
			if (rewritten) updatedCount += 1;
			if (suppressed) suppressedCount += 1;
			retentionSum += retention;

			const index = nodes.length;
			memoryIndices.push(index);
			const angle = ((memoryIndex + 0.5) / Math.max(1, dayMemories.length)) * Math.PI * 2 + dayIndex * 0.37;
			const jitter = (memoryIndex % 5 - 2) * 0.008;
			const cellRadius = ringRadius + jitter;
			const validFrom = text(memory.validFrom ?? memory.createdAt, day.date);
			const transactionAt = text(memory.updatedAt ?? memory.createdAt, validFrom);
			const label = memory.content || memory.id.slice(0, 8);
			const provenance = source('memory', memory.id);
			nodes.push({
				source: provenance,
				index,
				label,
				retention,
				trust: memoryTrust(memory),
				stability: num(memory.storageStrength, undefined as unknown as number),
				lastAccessed: memory.lastAccessedAt ?? memory.updatedAt ?? memory.createdAt,
				suppression: suppressed ? 1 : 0,
				tags: [day.date, ...(memory.tags ?? [])],
				type: memory.nodeType ?? 'memory'
			});
			cells.push({
				id: `timeline:${day.date}:${memory.id}`,
				memoryId: memory.id,
				day: day.date,
				dayIndex,
				nodeIndex: index,
				angle,
				radius: cellRadius,
				retention,
				validFrom,
				transactionAt,
				suppressed,
				rewritten,
				label,
				provenance
			});

			if (rewritten || suppressed) {
				events.push({
					source: source('event', `${memory.id}:${rewritten ? 'updated' : 'suppressed'}:${transactionAt}`),
					type: suppressed ? 'MemorySuppressed' : 'MemoryUpdated',
					targetIndex: index,
					frame: 45 + dayIndex * 10 + memoryIndex,
					energy: suppressed ? 1 : 0.65
				});
			}

			if (audit.length > 0) {
				receipts.push({
					source: source('receipt', `memory-audit:${memory.id}`),
					label: `audit ${memory.id.slice(0, 8)} · ${audit.length} events`,
					nodeIndices: [index]
				});
				for (const auditEvent of audit.slice(0, 8)) {
					events.push({
						source: source('event', `${memory.id}:${auditEvent.action}:${auditEvent.timestamp}`),
						type: `Audit:${auditEvent.action}`,
						targetIndex: index,
						frame: 70 + dayIndex * 12,
						energy: 0.4 + Math.abs(num(auditEvent.new_value, 0) - num(auditEvent.old_value, 0))
					});
				}
			}
		});

		const avgRetention = dayMemories.length ? retentionSum / dayMemories.length : 0;
		const daySource = scalarSource(`day.${day.date}.count`, day.count);
		rings.push({
			id: `timeline-day:${day.date}`,
			date: day.date,
			index: dayIndex,
			count: day.count,
			radius: ringRadius,
			retention: avgRetention,
			updatedCount,
			suppressedCount,
			memoryIndices,
			provenance: daySource
		});
		receipts.push({
			source: daySource,
			label: `${day.date} · ${day.count} memories`,
			nodeIndices: memoryIndices
		});
	});

	for (let i = 1; i < cells.length; i++) {
		edges.push({
			source: source('pair', `timeline-order:${cells[i - 1].memoryId}:${cells[i].memoryId}`),
			sourceIndex: cells[i - 1].nodeIndex,
			targetIndex: cells[i].nodeIndex,
			weight: 0.12,
			kind: 'bitemporal-order'
		});
	}

	const auditSlices = Object.entries(audits).map(([memoryId, auditEvents]) => ({ memoryId, events: auditEvents }));
	const totalMemories = num(input.totalMemories, nodes.length);
	const scene: TimelineScene = {
		organ: 'timeline',
		nodes,
		edges,
		events,
		receipts,
		scalars: {
			totalMemories,
			dayCount: nonEmptyDays.length,
			cellCount: cells.length,
			updatedCount: events.filter((event) => event.type === 'MemoryUpdated' || event.type === 'Audit:edited' || event.type === 'Audit:reconsolidated').length,
			suppressedCount: events.filter((event) => event.type === 'MemorySuppressed' || event.type === 'Audit:suppressed').length,
			maxDayCount: maxCount
		},
		alive: cells.length > 0,
		rings,
		cells,
		audits: auditSlices,
		raw: { days, audits }
	};

	if (import.meta.env.DEV) assertProvenance(scene);
	void scalarSource('totalMemories', totalMemories);
	return scene;
}

export function describeTimelinePick(scene: TimelineScene, id: string): TimelineRing | TimelineCell | null {
	return scene.cells.find((cell) => cell.id === id) ?? scene.rings.find((ring) => ring.id === id) ?? null;
}
