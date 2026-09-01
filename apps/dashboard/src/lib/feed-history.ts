import type { ChangelogEvent } from '$stores/api';
import type { VestigeEvent, VestigeEventType } from '$types';

/** Turn a durable changelog record into the same envelope used by live WS events. */
export function changelogEventToVestigeEvent(event: ChangelogEvent): VestigeEvent {
	return {
		type: event.type as VestigeEventType,
		// The top-level changelog timestamp is canonical. Put it in the shared
		// event data too because live event consumers intentionally only read the
		// common `{ type, data }` envelope.
		data: { ...event.data, timestamp: event.timestamp }
	};
}

function eventTimestamp(event: VestigeEvent): number {
	const raw = event.data.timestamp ?? event.data.at ?? event.data.created_at;
	if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
	if (typeof raw === 'string') {
		const parsed = Date.parse(raw);
		if (Number.isFinite(parsed)) return parsed;
	}
	// Live events without a clock are genuinely "now" for display purposes.
	return Number.POSITIVE_INFINITY;
}

function eventFingerprint(event: VestigeEvent): string {
	const orderedData = Object.fromEntries(Object.entries(event.data).sort(([a], [b]) => a.localeCompare(b)));
	return `${event.type}:${JSON.stringify(orderedData)}`;
}

/**
 * Merge recorded history with the in-session WebSocket buffer without showing
 * an event twice when a reload races a live broadcast. History is read-only;
 * clearing the live buffer must never imply that it deletes audit history.
 */
export function mergeFeedEvents(history: VestigeEvent[], live: VestigeEvent[]): VestigeEvent[] {
	const seen = new Set<string>();
	return [...live, ...history]
		.filter((event) => {
			const key = eventFingerprint(event);
			if (seen.has(key)) return false;
			seen.add(key);
			return true;
		})
		.sort((a, b) => eventTimestamp(b) - eventTimestamp(a));
}
