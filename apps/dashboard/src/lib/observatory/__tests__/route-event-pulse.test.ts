import { describe, it, expect } from 'vitest';
import {
	isBackfillPulse,
	shouldRefreshMemories,
	shouldRefreshVerdicts
} from '../route-event-pulse';
import type { VestigeEvent } from '$types';

function ev(type: string): VestigeEvent {
	return { type: type as VestigeEvent['type'], data: {} };
}

describe('RouteEventPulse', () => {
	it('refreshes memories on mutation events only', () => {
		expect(shouldRefreshMemories(ev('MemoryCreated'))).toBe(true);
		expect(shouldRefreshMemories(ev('Heartbeat'))).toBe(false);
	});

	it('refreshes verdicts on HookVerdictRecorded', () => {
		expect(shouldRefreshVerdicts(ev('HookVerdictRecorded'))).toBe(true);
		expect(shouldRefreshVerdicts(ev('MemoryCreated'))).toBe(false);
	});

	it('detects backfill pulses', () => {
		expect(isBackfillPulse(ev('BackfillFired'))).toBe(true);
		expect(isBackfillPulse(ev('CausalReceipt'))).toBe(true);
		expect(isBackfillPulse(ev('Heartbeat'))).toBe(false);
	});
});
