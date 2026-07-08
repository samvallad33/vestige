/**
 * Cognitive Observatory — bloom mip-chain plan (S2).
 *
 * Pure, GPU-free math so vitest can pin the pyramid shape. Mip 0 is HALF the
 * swapchain resolution (the classic mip-bloom cost/quality point); the chain
 * stops before the smallest dimension would drop below 8px (degenerate
 * anisotropic mips smear) and is clamped to at most 6 levels.
 *
 * NOTE bloom RADIUS therefore varies with viewport size (more mips on bigger
 * canvases = wider glow). Brightness does NOT — the composite normalizes the
 * additive up-chain by textureNumLevels (post-chain.ts).
 */

export interface BloomMipPlan {
	baseW: number;
	baseH: number;
	mipCount: number;
	sizes: Array<[number, number]>;
}

export function planBloomMips(w: number, h: number): BloomMipPlan {
	const baseW = Math.max(1, w >> 1);
	const baseH = Math.max(1, h >> 1); // mip 0 = HALF res
	// min-dim ≥ 8px at the smallest mip (avoids degenerate anisotropic mips).
	const mipCount = Math.min(
		6,
		Math.max(1, 1 + Math.floor(Math.log2(Math.min(baseW, baseH) / 8)))
	);
	const sizes = Array.from({ length: mipCount }, (_, i): [number, number] => [
		Math.max(1, baseW >> i),
		Math.max(1, baseH >> i)
	]);
	return { baseW, baseH, mipCount, sizes };
}
