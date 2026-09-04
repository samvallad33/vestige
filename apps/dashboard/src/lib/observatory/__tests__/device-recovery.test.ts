import { describe, it, expect, vi, afterEach } from 'vitest';

// The ObservatoryEngine constructor is pure/synchronous (no GPU boot), so the
// recovery loop can be driven with zero WebGPU: `start()` is the only thing it
// calls that needs a device, and that is exactly the seam we stub.

const mockCanvas = {
	width: 512,
	height: 512,
	clientWidth: 512,
	clientHeight: 512,
	getContext: () => null,
	toDataURL: () => 'data:image/png;base64,'
} as unknown as HTMLCanvasElement;

if (typeof globalThis.document === 'undefined') {
	(globalThis as unknown as { document: unknown }).document = {
		createElement: (tag: string) => (tag === 'canvas' ? mockCanvas : {}),
		addEventListener: () => {},
		removeEventListener: () => {}
	};
}

import { ObservatoryEngine, DEVICE_RECOVERY_DELAYS_MS, type EngineStatus } from '../engine';

/** The recovery loop and its two collaborators are private; the test drives them. */
type RecoverableEngine = ObservatoryEngine & {
	recoverFromDeviceLoss(reason: string): Promise<void>;
	releaseDeviceResources(): void;
	disposed: boolean;
};

function makeEngine(): { engine: RecoverableEngine; statuses: EngineStatus[] } {
	const engine = new ObservatoryEngine({
		canvas: mockCanvas,
		demo: 'recall-path',
		seed: 'device-recovery-test'
	}) as RecoverableEngine;
	const statuses: EngineStatus[] = [];
	engine.onStatus((s) => statuses.push(s));
	// Releasing device-bound state is covered by the pass registry tests; here
	// it would only dereference a device that never existed.
	vi.spyOn(engine, 'releaseDeviceResources').mockImplementation(() => {});
	return { engine, statuses };
}

/** Attempt numbers reported to the canvas owner, in order. */
function recoveringAttempts(statuses: EngineStatus[]): number[] {
	return statuses
		.filter((s): s is Extract<EngineStatus, { state: 'recovering' }> => s.state === 'recovering')
		.map((s) => s.attempt);
}

afterEach(() => {
	vi.useRealTimers();
	vi.restoreAllMocks();
});

describe('GPU device-loss recovery schedule', () => {
	it('backs off exponentially from half a second and gives up after five attempts', () => {
		expect([...DEVICE_RECOVERY_DELAYS_MS]).toEqual([500, 1000, 2000, 4000, 8000]);
		for (let i = 1; i < DEVICE_RECOVERY_DELAYS_MS.length; i++) {
			expect(DEVICE_RECOVERY_DELAYS_MS[i]).toBe(DEVICE_RECOVERY_DELAYS_MS[i - 1] * 2);
		}
		const total = DEVICE_RECOVERY_DELAYS_MS.reduce((a, b) => a + b, 0);
		expect(total).toBeLessThanOrEqual(16_000);
	});
});

describe('GPU device-loss recovery behaviour', () => {
	it('walks the whole schedule and settles in error when every re-acquire fails', async () => {
		vi.useFakeTimers();
		const { engine, statuses } = makeEngine();
		const start = vi.spyOn(engine, 'start').mockResolvedValue(false);

		const done = engine.recoverFromDeviceLoss('gpu reset');
		await vi.runAllTimersAsync();
		await done;

		expect(start).toHaveBeenCalledTimes(DEVICE_RECOVERY_DELAYS_MS.length);
		// The canvas owner sees a numbered attempt for each one, never a silent gap.
		expect(recoveringAttempts(statuses)).toEqual([1, 2, 3, 4, 5]);
		// The ORIGINAL reason survives to the terminal status.
		expect(engine.status).toEqual({ state: 'error', reason: 'GPU device lost: gpu reset' });
	});

	it('stops at the first successful re-acquire and never reports error', async () => {
		vi.useFakeTimers();
		const { engine, statuses } = makeEngine();
		let attempts = 0;
		const start = vi.spyOn(engine, 'start').mockImplementation(async () => ++attempts === 3);

		const done = engine.recoverFromDeviceLoss('driver update');
		await vi.runAllTimersAsync();
		await done;

		expect(start).toHaveBeenCalledTimes(3);
		expect(recoveringAttempts(statuses)).toEqual([1, 2, 3]);
		expect(statuses.some((s) => s.state === 'error')).toBe(false);
	});

	it('abandons recovery once the engine is disposed', async () => {
		vi.useFakeTimers();
		const { engine, statuses } = makeEngine();
		const start = vi.spyOn(engine, 'start').mockResolvedValue(false);
		engine.disposed = true;

		const done = engine.recoverFromDeviceLoss('backgrounded');
		await vi.runAllTimersAsync();
		await done;

		// A disposed engine must not re-acquire a device, and must not paint an
		// error over a teardown the owner asked for.
		expect(start).not.toHaveBeenCalled();
		expect(statuses.some((s) => s.state === 'error')).toBe(false);
	});

	it('runs one recovery loop at a time', async () => {
		vi.useFakeTimers();
		const { engine } = makeEngine();
		const start = vi.spyOn(engine, 'start').mockResolvedValue(false);

		// device.lost can fire again while the first loop is still backing off.
		const first = engine.recoverFromDeviceLoss('gpu reset');
		const second = engine.recoverFromDeviceLoss('gpu reset again');
		await vi.runAllTimersAsync();
		await Promise.all([first, second]);

		expect(start).toHaveBeenCalledTimes(DEVICE_RECOVERY_DELAYS_MS.length);
	});
});
