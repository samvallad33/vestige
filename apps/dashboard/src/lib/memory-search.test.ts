import { describe, expect, it } from 'vitest';
import { handleSearchView } from './memory-search';
import type { Memory, MemoryListResponse } from '$types';

function memory(id: string, tags: string[] = []): Memory {
	return {
		id,
		content: `content of ${id}`,
		nodeType: 'fact',
		tags,
		retentionStrength: 1,
		storageStrength: 1,
		retrievalStrength: 1,
		createdAt: '2026-09-30T00:00:00Z',
		updatedAt: '2026-09-30T00:00:00Z'
	};
}

function response(partial: Partial<MemoryListResponse>): MemoryListResponse {
	return { total: 0, memories: [], ...partial };
}

describe('handleSearchView', () => {
	it('shows the memory an exact id resolves to', () => {
		const view = handleSearchView(
			'mem-0000000000000001',
			response({
				total: 1,
				memories: [memory('mem-0000000000000001')],
				resolution: { handle: 'mem-0000000000000001', kind: 'memory', exact: true, ambiguous: false, handleRequired: null }
			})
		);
		expect(view.memories.map((m) => m.id)).toEqual(['mem-0000000000000001']);
		expect(view.note).toBe('Memory mem-0000000000000001.');
	});

	it('counts a tag across the whole store, not the page', () => {
		const view = handleSearchView(
			'ethena',
			response({
				total: 15,
				returned: 2,
				memories: [memory('a', ['ethena']), memory('b', ['ethena'])],
				resolution: { handle: 'ethena', kind: 'tag', exact: true, ambiguous: false, handleRequired: null }
			})
		);
		expect(view.note).toBe('15 memories tagged "ethena", showing 2.');
	});

	it('says an ambiguous prefix needs more of the id', () => {
		const view = handleSearchView(
			'mem-0000',
			response({
				total: 2,
				memories: [memory('mem-00001'), memory('mem-00002')],
				resolution: { handle: 'mem-0000', kind: 'memory', exact: false, ambiguous: true, handleRequired: null }
			})
		);
		expect(view.memories).toHaveLength(2);
		expect(view.note).toContain('more than one memory id');
	});

	it('never presents free text as an empty match', () => {
		const view = handleSearchView(
			'refund policy',
			response({
				resolution: { handle: 'refund policy', kind: 'unknown', exact: false, ambiguous: false, handleRequired: 'recall is handle-based' }
			})
		);
		expect(view.memories).toEqual([]);
		expect(view.note).toContain('No memory has the id, id prefix or tag "refund policy"');
		expect(view.note).toContain('Free-text search is not available');
	});
});
