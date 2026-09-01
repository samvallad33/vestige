import { describe, expect, it } from 'vitest';
import { changelogEventToVestigeEvent, mergeFeedEvents } from '../feed-history';
import type { VestigeEvent } from '$types';

describe('feed history hydration', () => {
	it('converts durable changelog timestamps into the shared live-event envelope', () => {
		const event = changelogEventToVestigeEvent({
			type: 'DreamCompleted',
			timestamp: '2026-07-15T10:00:00Z',
			data: { memories_replayed: 4 }
		});
		expect(event).toEqual({
			type: 'DreamCompleted',
			data: { memories_replayed: 4, timestamp: '2026-07-15T10:00:00Z' }
		});
	});

	it('deduplicates an event seen in history and over the live socket, newest first', () => {
		const historical: VestigeEvent = { type: 'DreamCompleted', data: { timestamp: '2026-07-15T10:00:00Z', memories_replayed: 4 } };
		const live: VestigeEvent = { type: 'MemoryCreated', data: { id: 'new', timestamp: '2026-07-15T11:00:00Z' } };
		expect(mergeFeedEvents([historical], [live, historical])).toEqual([live, historical]);
	});
});
