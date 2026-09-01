import { describe, it, expect } from 'vitest';
import { auditDiffLines, formatRetentionArrow, isRewrittenMemory } from '../timeline-audit';

describe('timeline audit diffs', () => {
	it('renders an honest retention arrow', () => {
		expect(formatRetentionArrow(0.8, 0.4)).toBe('80% → 40% (-40)');
		expect(formatRetentionArrow(0.1, 0.25)).toBe('10% → 25% (+15)');
	});

	it('omits invented diffs when values are missing', () => {
		expect(formatRetentionArrow(undefined, 0.4)).toBeNull();
		expect(auditDiffLines({ action: 'accessed', timestamp: '2026-08-31T00:00:00Z' })).toEqual([]);
	});

	it('surfaces reason + actor with the arrow', () => {
		expect(
			auditDiffLines({
				action: 'updated',
				timestamp: '2026-08-31T00:00:00Z',
				old_value: 0.9,
				new_value: 0.2,
				reason: 'decay',
				triggered_by: 'fsrs'
			})
		).toEqual(['90% → 20% (-70)', 'decay', 'by fsrs']);
	});

	it('rewritten filter is bitemporal, not a guess', () => {
		expect(isRewrittenMemory({ createdAt: 'a', updatedAt: 'b' })).toBe(true);
		expect(isRewrittenMemory({ createdAt: 'a', updatedAt: 'a' })).toBe(false);
	});
});
