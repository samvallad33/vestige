/**
 * Cognitive Observatory — Khronos PBR Neutral tone mapping, CPU reference.
 *
 * Exact TS mirror of pbr_neutral() in post/shaders/post.wgsl.ts. vitest can't
 * run WGSL on a GPU, so all determinism-critical tonemap math is verified
 * against this mirror — keep the two in lockstep.
 *
 * IMPORTANT: PBR Neutral is NOT the identity below the compression knee. The
 * black-offset branch subtracts `offset` from EVERY pixel (0.04 when the min
 * channel is ≥ 0.08, else x − 6.25x²). A naive #05060a clear would therefore
 * tonemap to ≈ #010206 (crushed void). VOID_CLEAR_HDR below is the analytic
 * preimage that lands the post stack EXACTLY back on #05060a.
 */

import { BLOOM_STRENGTH } from './post-chain';

/** Khronos PBR Neutral reference (https://github.com/KhronosGroup/ToneMapping). */
export function pbrNeutralReference(
	rgb: readonly [number, number, number]
): [number, number, number] {
	const startCompression = 0.8 - 0.04;
	const desaturation = 0.15;

	let [r, g, b] = rgb;
	const x = Math.min(r, Math.min(g, b));
	const offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	r -= offset;
	g -= offset;
	b -= offset;

	const peak = Math.max(r, Math.max(g, b));
	if (peak < startCompression) return [r, g, b];

	const d = 1 - startCompression;
	const newPeak = 1 - (d * d) / (peak + d - startCompression);
	const scale = newPeak / peak;
	r *= scale;
	g *= scale;
	b *= scale;

	const gMix = 1 / (desaturation * (peak - newPeak) + 1);
	// mix(color, vec3(newPeak), 1 - g) per the Khronos spec.
	const w = 1 - gMix;
	return [r + w * (newPeak - r), g + w * (newPeak - g), b + w * (newPeak - b)];
}

// ---------------------------------------------------------------------------
// VOID_CLEAR_HDR — the HDR clear color whose post-stack output is #05060a.
//
// Derivation (verified by tone-reference.test.ts):
//  - The normalized bloom chain has flat-field gain exactly 1 (renormalized
//    Karis + exact box/tent weights + /mipCount in the composite), so a flat
//    void field enters the tonemap as (1 + BLOOM_STRENGTH) · v = 1.18 · v.
//  - Below the knee, out = in − offset with offset = x − 6.25x² (x = min
//    channel < 0.08), hence out_min = 6.25x². Solving out_min = 5/255:
//        x = sqrt((5/255)/6.25) = 0.05601120…
//        offset = x − 5/255    = 0.03640336…
//        g_in = 6/255 + offset;  b_in = 10/255 + offset
//    peak = b_in ≈ 0.0756 < 0.76 → the compression branch is NOT taken.
//  - Divide by (1 + BLOOM_STRENGTH): pbrNeutral(1.18 · VOID_CLEAR_HDR) is
//    EXACTLY (5/255, 6/255, 10/255) = #05060a.
//    Literals: r ≈ 0.0474671, g ≈ 0.0507905, b ≈ 0.0640839.
//  - f16 quantization of the clear (ulp ≈ 3e-5 near 0.05) keeps the
//    tonemapped void well inside ±0.5/255 — verified safe.
// ---------------------------------------------------------------------------

const VOID_R_IN = Math.sqrt(5 / 255 / 6.25); // 0.05601120…
const VOID_OFFSET = VOID_R_IN - 5 / 255; // 0.03640336…

export const VOID_CLEAR_HDR: GPUColorDict = {
	r: VOID_R_IN / (1 + BLOOM_STRENGTH),
	g: (6 / 255 + VOID_OFFSET) / (1 + BLOOM_STRENGTH),
	b: (10 / 255 + VOID_OFFSET) / (1 + BLOOM_STRENGTH),
	a: 1
};
