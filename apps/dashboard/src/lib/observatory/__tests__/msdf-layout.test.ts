import { describe, expect, it } from 'vitest';
import atlas from '../../../../static/msdf/jetbrains-mono.json';
import { layoutText, type MsdfAtlasJson } from '../text/layout';

describe('MSDF text layout', () => {
	it('packs the atlas V flip for the top of A loudly', () => {
		const glyphs = layoutText('A', atlas as MsdfAtlasJson);
		expect(glyphs).toHaveLength(1);
		// atlasBounds.top for A is 373.5 in the checked-in 512px atlas.
		// WebGPU texture V is top-down, so the glyph top must sample 1 - top/512.
		expect(glyphs[0].v).toBeCloseTo(1 - 373.5 / 512, 4);
		expect(glyphs[0].v).toBeCloseTo(0.2705, 4);
	});

	it('falls back to ASCII question mark for non-atlas characters', () => {
		const [fallback] = layoutText('·', atlas as MsdfAtlasJson);
		const [question] = layoutText('?', atlas as MsdfAtlasJson);
		expect(fallback).toEqual(question);
	});
});
