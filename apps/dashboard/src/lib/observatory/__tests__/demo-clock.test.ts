import { describe, it, expect } from 'vitest';
import { DemoClock, deterministicSpherePosition } from '../demo-clock';

describe('DemoClock', () => {
	it('should start at frame 0 with phase 0', () => {
		const clock = new DemoClock({ seed: 'test-seed' });
		const state = clock.state;
		expect(state.frame).toBe(0);
		expect(state.phase).toBe(0);
		expect(state.totalFrames).toBe(0);
	});

	it('should advance frame by 1 on each tick', () => {
		const clock = new DemoClock({ seed: 'test-seed' });
		clock.tick();
		expect(clock.state.frame).toBe(1);
		clock.tick();
		expect(clock.state.frame).toBe(2);
	});

	it('should wrap frame at loopFrames (default 720)', () => {
		const clock = new DemoClock({ seed: 'test-seed' });
		// Advance to loopFrames - 1
		for (let i = 0; i < 719; i++) {
			clock.tick();
		}
		expect(clock.state.frame).toBe(719);
		// One more tick should wrap to 0
		clock.tick();
		expect(clock.state.frame).toBe(0);
	});

	it('should compute correct phase (frame / loopFrames)', () => {
		const clock = new DemoClock({ seed: 'test-seed' });
		clock.tick();
		expect(clock.state.phase).toBeCloseTo(1 / 720, 6);
		clock.tick();
		expect(clock.state.phase).toBeCloseTo(2 / 720, 6);
	});

	it('should produce identical PRNG sequences for the same seed', () => {
		const clock1 = new DemoClock({ seed: 'identical-seed' });
		const clock2 = new DemoClock({ seed: 'identical-seed' });
		const values1: number[] = [];
		const values2: number[] = [];
		for (let i = 0; i < 100; i++) {
			clock1.tick();
			values1.push(clock1.state.rng());
			clock2.tick();
			values2.push(clock2.state.rng());
		}
		expect(values1).toEqual(values2);
	});

	it('should produce different PRNG sequences for different seeds', () => {
		const clock1 = new DemoClock({ seed: 'seed-a' });
		const clock2 = new DemoClock({ seed: 'seed-b' });
		clock1.tick();
		clock2.tick();
		expect(clock1.state.rng()).not.toBe(clock2.state.rng());
	});

	it('should produce different positions for different seeds', () => {
		const clock1 = new DemoClock({ seed: 'pos-seed-a' });
		const clock2 = new DemoClock({ seed: 'pos-seed-b' });
		const pos1 = deterministicSpherePosition(0, 100, 50, clock1.state.rng);
		const pos2 = deterministicSpherePosition(0, 100, 50, clock2.state.rng);
		expect(pos1).not.toEqual(pos2);
	});

	it('should produce identical positions for the same seed', () => {
		const clock1 = new DemoClock({ seed: 'same-seed' });
		const clock2 = new DemoClock({ seed: 'same-seed' });
		const pos1 = deterministicSpherePosition(0, 100, 50, clock1.state.rng);
		const pos2 = deterministicSpherePosition(0, 100, 50, clock2.state.rng);
		expect(pos1).toEqual(pos2);
	});

	it('should reset to frame 0 and re-seed PRNG', () => {
		const clock = new DemoClock({ seed: 'reset-seed' });
		// Advance 100 frames
		for (let i = 0; i < 100; i++) {
			clock.tick();
		}
		expect(clock.state.frame).toBe(100);
		expect(clock.state.totalFrames).toBe(100);

		// Reset
		clock.reset();
		expect(clock.state.frame).toBe(0);
		expect(clock.state.totalFrames).toBe(0);

		// After reset, the PRNG should produce the same values as the initial state
		const afterResetRng = clock.state.rng();
		const initialRng = new DemoClock({ seed: 'reset-seed' }).state.rng();
		expect(afterResetRng).toBe(initialRng);
	});

	it('should respect custom loopFrames', () => {
		const clock = new DemoClock({ seed: 'custom-loop', loopFrames: 360 });
		for (let i = 0; i < 360; i++) {
			clock.tick();
		}
		expect(clock.state.frame).toBe(0);
		expect(clock.state.phase).toBe(0);
	});

	it('should respect custom fps (affects loopDuration)', () => {
		const clock = new DemoClock({ seed: 'custom-fps', fps: 30, loopFrames: 300 });
		expect(clock.loopDuration).toBe(10); // 300 / 30 = 10s
	});

	it('exposes framesPerLoop for capture-mode frame normalization', () => {
		expect(new DemoClock({ seed: 'x' }).framesPerLoop).toBe(720);
		expect(new DemoClock({ seed: 'x', loopFrames: 360 }).framesPerLoop).toBe(360);
	});
});

describe('deterministicSpherePosition', () => {
	it('should place nodes on a sphere surface', () => {
		const clock = new DemoClock({ seed: 'sphere-test' });
		const pos = deterministicSpherePosition(0, 10, 50, clock.state.rng);
		const [x, y, z] = pos;
		const dist = Math.sqrt(x * x + y * y + z * z);
		// Should be approximately at the given radius (some variance from golden angle)
		expect(dist).toBeGreaterThan(40);
		expect(dist).toBeLessThan(60);
	});

	it('should produce different positions for different indices', () => {
		const clock = new DemoClock({ seed: 'sphere-test' });
		const pos0 = deterministicSpherePosition(0, 10, 50, clock.state.rng);
		const pos1 = deterministicSpherePosition(1, 10, 50, clock.state.rng);
		expect(pos0).not.toEqual(pos1);
	});
});
