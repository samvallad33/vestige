/**
 * RouteEventPulse — map the live $eventFeed onto organ refresh / field
 * pulses. Start surface: feed, memories, blackbox, VerdictBar.
 */

import type { VestigeEvent } from '$types';

export const MEMORY_MUTATION_EVENTS = new Set([
	'MemoryCreated',
	'MemoryUpdated',
	'MemoryDeleted',
	'MemoryPromoted',
	'MemoryDemoted',
	'MemorySuppressed',
	'MemoryUnsuppressed',
	'ConsolidationCompleted',
	'DreamCompleted'
]);

export const VERDICT_EVENTS = new Set(['HookVerdictRecorded', 'SanhedrinVeto', 'MemorySuppressed']);

export const BACKFILL_EVENTS = new Set(['BackfillFired', 'CausalReceipt']);

export function eventMemoryIds(event: VestigeEvent): string[] {
	const d = event.data ?? {};
	const ids: string[] = [];
	for (const key of ['memory_id', 'memoryId', 'id']) {
		if (typeof d[key] === 'string') ids.push(d[key] as string);
	}
	if (Array.isArray(d.ids)) {
		for (const id of d.ids) if (typeof id === 'string') ids.push(id);
	}
	return [...new Set(ids)];
}

export function shouldRefreshMemories(event: VestigeEvent): boolean {
	return MEMORY_MUTATION_EVENTS.has(event.type);
}

export function shouldRefreshVerdicts(event: VestigeEvent): boolean {
	return VERDICT_EVENTS.has(event.type);
}

export function isBackfillPulse(event: VestigeEvent): boolean {
	return BACKFILL_EVENTS.has(event.type);
}
