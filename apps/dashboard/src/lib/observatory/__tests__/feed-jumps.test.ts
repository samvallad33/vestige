import { describe, it, expect } from 'vitest';
import { feedJumps } from '../feed-jumps';

describe('feedJumps', () => {
	it('emits memory + run + receipt chips from a retrieve payload', () => {
		const jumps = feedJumps({
			memory_id: 'mem-aaaa',
			run_id: 'run_abc',
			receipt_id: 'rcp_1'
		});
		expect(jumps.map((j) => j.kind)).toEqual(['memory', 'run', 'receipt']);
		expect(jumps[0]?.href).toContain('/memories?memory=mem-aaaa');
		expect(jumps[1]?.href).toContain('/blackbox?run=run_abc');
		expect(jumps[2]?.href).toContain('/observatory?receipt=rcp_1');
	});

	it('skips heartbeats so the strip is not a JSON dump of vitals', () => {
		expect(feedJumps({ memory_count: 12, id: 'nope' }, 'Heartbeat')).toEqual([]);
	});

	it('ignores empty / non-id values', () => {
		expect(feedJumps({ memory_id: '', run_id: 12 as unknown as string })).toEqual([]);
	});
});
