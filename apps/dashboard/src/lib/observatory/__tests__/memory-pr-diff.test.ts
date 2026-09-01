import { describe, it, expect } from 'vitest';
import { viewMemoryPrDiff } from '../export/memory-pr-diff';

describe('viewMemoryPrDiff', () => {
	it('lifts nested proposed/before/target without dumping JSON as the hero', () => {
		const view = viewMemoryPrDiff({
			kind: 'supersede',
			target_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
			before_content: 'old fact',
			proposed_content: 'new fact'
		});
		expect(view.targetId).toBe('aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
		expect(view.before).toBe('old fact');
		expect(view.proposed).toBe('new fact');
		expect(view.kind).toBe('supersede');
	});

	it('walks nested objects', () => {
		const view = viewMemoryPrDiff({
			change: { proposed: 'hello', before: 'bye' },
			subjectId: 'mem-1'
		});
		expect(view.proposed).toBe('hello');
		expect(view.before).toBe('bye');
		expect(view.targetId).toBe('mem-1');
	});

	it('empty diff is honest, not invented', () => {
		const view = viewMemoryPrDiff({});
		expect(view.proposed).toBeNull();
		expect(view.rest).toEqual([]);
	});
});
