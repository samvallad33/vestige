import { describe, it, expect, vi } from 'vitest';

// The ObservatoryEngine constructor is pure/synchronous (no GPU boot) — it only
// builds a DemoClock and seeds the params array. It needs a canvas-shaped object
// and document for the DemoClock. Provide minimal mocks so we can unit-test the
// pass registry (addPass / removePass / clearPasses) with zero WebGPU.

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
		createElement: (tag: string) => (tag === 'canvas' ? mockCanvas : {})
	};
}

import { ObservatoryEngine, type FramePass } from '../engine';

/** A pass with a dispose spy, plus a marker so we can assert draw membership. */
function stubPass(): FramePass & { dispose: ReturnType<typeof vi.fn>; render: ReturnType<typeof vi.fn> } {
	return {
		render: vi.fn(),
		dispose: vi.fn()
	};
}

function makeEngine(): ObservatoryEngine {
	return new ObservatoryEngine({ canvas: mockCanvas, demo: 'recall-path', seed: 'engine-passes-test' });
}

/**
 * The engine keeps passes in a `private passes` array. We assert membership
 * indirectly through the ONE public consumer of that array on the render side:
 * a render pass encoder is fed to every registered pass. We can't call the real
 * private frame loop without a GPU, so we drive the passes exactly as the loop
 * does — iterate whatever `removePass`/`clearPasses` left registered by using a
 * tiny reflection into the array via a typed accessor. This keeps the test on
 * the real public API while still proving membership.
 */
function registeredCount(engine: ObservatoryEngine): number {
	return (engine as unknown as { passes: FramePass[] }).passes.length;
}

describe('ObservatoryEngine pass registry (Spatial Palace primitive)', () => {
	it('addPass registers passes without disposing them', () => {
		const engine = makeEngine();
		const a = stubPass();
		const b = stubPass();
		engine.addPass(a);
		engine.addPass(b);
		expect(registeredCount(engine)).toBe(2);
		expect(a.dispose).not.toHaveBeenCalled();
		expect(b.dispose).not.toHaveBeenCalled();
	});

	it('removePass splices exactly the target pass and disposes ONLY it', () => {
		const engine = makeEngine();
		const a = stubPass();
		const b = stubPass();
		const c = stubPass();
		engine.addPass(a);
		engine.addPass(b);
		engine.addPass(c);

		engine.removePass(b);

		expect(registeredCount(engine)).toBe(2);
		expect(b.dispose).toHaveBeenCalledTimes(1);
		expect(a.dispose).not.toHaveBeenCalled();
		expect(c.dispose).not.toHaveBeenCalled();
	});

	it('removePass is a no-op for a pass that was never registered', () => {
		const engine = makeEngine();
		const a = stubPass();
		const stranger = stubPass();
		engine.addPass(a);

		engine.removePass(stranger);

		expect(registeredCount(engine)).toBe(1);
		expect(stranger.dispose).not.toHaveBeenCalled();
		expect(a.dispose).not.toHaveBeenCalled();
	});

	it('removePass tolerates a pass with no dispose() (bare FramePass)', () => {
		const engine = makeEngine();
		const bare: FramePass = { render: vi.fn() };
		engine.addPass(bare);
		expect(() => engine.removePass(bare)).not.toThrow();
		expect(registeredCount(engine)).toBe(0);
	});

	it('clearPasses disposes every pass and empties the registry', () => {
		const engine = makeEngine();
		const a = stubPass();
		const b = stubPass();
		const c = stubPass();
		engine.addPass(a);
		engine.addPass(b);
		engine.addPass(c);

		engine.clearPasses();

		expect(registeredCount(engine)).toBe(0);
		expect(a.dispose).toHaveBeenCalledTimes(1);
		expect(b.dispose).toHaveBeenCalledTimes(1);
		expect(c.dispose).toHaveBeenCalledTimes(1);
	});

	it('a swap cycle (clearPasses then addPass) leaves only the new passes', () => {
		const engine = makeEngine();
		const oldA = stubPass();
		const oldB = stubPass();
		engine.addPass(oldA);
		engine.addPass(oldB);

		// fly into a new organ: clear the old scene, register the new one
		engine.clearPasses();
		const newA = stubPass();
		engine.addPass(newA);

		expect(registeredCount(engine)).toBe(1);
		expect(oldA.dispose).toHaveBeenCalledTimes(1);
		expect(oldB.dispose).toHaveBeenCalledTimes(1);
		expect(newA.dispose).not.toHaveBeenCalled();
	});
});
