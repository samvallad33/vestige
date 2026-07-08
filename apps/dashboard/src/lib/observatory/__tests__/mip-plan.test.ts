import { describe, it, expect } from 'vitest';
import { planBloomMips } from '../post/mip-plan';

describe('planBloomMips', () => {
	it('M1 Max 3024×1964: half-res base, 6 mips, halving widths', () => {
		const p = planBloomMips(3024, 1964);
		expect(p.baseW).toBe(1512);
		expect(p.baseH).toBe(982);
		expect(p.mipCount).toBe(6);
		expect(p.sizes.map(([w]) => w)).toEqual([1512, 756, 378, 189, 94, 47]);
	});

	it('64×64 → 3 mips', () => {
		const p = planBloomMips(64, 64);
		expect(p.baseW).toBe(32);
		expect(p.baseH).toBe(32);
		expect(p.mipCount).toBe(3);
		expect(p.sizes).toEqual([
			[32, 32],
			[16, 16],
			[8, 8]
		]);
	});

	it('2×2 → 1 mip (1×1 base)', () => {
		const p = planBloomMips(2, 2);
		expect(p.baseW).toBe(1);
		expect(p.baseH).toBe(1);
		expect(p.mipCount).toBe(1);
		expect(p.sizes).toEqual([[1, 1]]);
	});

	it('1×1 → 1 mip (degenerate; the up-loop runs zero times — harmless)', () => {
		const p = planBloomMips(1, 1);
		expect(p.baseW).toBe(1);
		expect(p.baseH).toBe(1);
		expect(p.mipCount).toBe(1);
	});

	it('sizes halve monotonically and never drop below 1', () => {
		for (const [w, h] of [
			[3024, 1964],
			[1920, 1080],
			[800, 600],
			[375, 812],
			[7, 3]
		] as const) {
			const p = planBloomMips(w, h);
			expect(p.sizes[0]).toEqual([p.baseW, p.baseH]);
			for (let i = 1; i < p.sizes.length; i++) {
				const [pw, ph] = p.sizes[i - 1];
				const [cw, ch] = p.sizes[i];
				expect(cw).toBe(Math.max(1, pw >> 1));
				expect(ch).toBe(Math.max(1, ph >> 1));
				expect(cw).toBeGreaterThanOrEqual(1);
				expect(ch).toBeGreaterThanOrEqual(1);
			}
		}
	});

	it('smallest mip keeps min dimension ≥ 8 when the base allows it', () => {
		for (const [w, h] of [
			[3024, 1964],
			[1920, 1080],
			[375, 812],
			[256, 128],
			[64, 64]
		] as const) {
			const p = planBloomMips(w, h);
			const [lw, lh] = p.sizes[p.mipCount - 1];
			expect(Math.min(lw, lh)).toBeGreaterThanOrEqual(8);
		}
	});

	it('mipCount is clamped to [1, 6]', () => {
		expect(planBloomMips(1, 1).mipCount).toBe(1);
		expect(planBloomMips(4, 4).mipCount).toBe(1);
		expect(planBloomMips(8192, 8192).mipCount).toBe(6);
		expect(planBloomMips(16384, 16384).mipCount).toBe(6);
	});
});
