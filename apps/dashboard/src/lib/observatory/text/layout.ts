export interface MsdfBounds {
	left: number;
	bottom: number;
	right: number;
	top: number;
}

export interface MsdfGlyph {
	unicode: number;
	advance?: number;
	planeBounds?: MsdfBounds;
	atlasBounds?: MsdfBounds;
}

export interface MsdfAtlasJson {
	atlas: {
		width: number;
		height: number;
	};
	metrics?: {
		lineHeight?: number;
	};
	glyphs: MsdfGlyph[];
}

export interface GlyphInstance {
	/** Quad origin in em/NDC-y units, baseline-relative, +Y up. */
	x: number;
	y: number;
	/** Quad size in em/NDC-y units. */
	w: number;
	h: number;
	/** Atlas UV origin. v is already flipped from atlas yOrigin: bottom to GPU top-down V. */
	u: number;
	v: number;
	/** Atlas UV extent. vh is positive from flipped glyph top to flipped glyph bottom. */
	uw: number;
	vh: number;
}

export interface LayoutTextOptions {
	/** Maximum line width in em units. Overlong lines are truncated with ASCII dots. */
	maxWidthEm?: number;
	/** Monospace advance in em units. The checked-in atlas uses 0.6 for every ASCII glyph. */
	advance?: number;
	/** Baseline-to-baseline distance in em units. Defaults to the atlas metric, then 1.32. */
	lineHeight?: number;
}

const ASCII_MIN = 0x20;
const ASCII_MAX = 0x7e;
const FALLBACK_CODEPOINT = 0x3f; // '?'
const DEFAULT_ADVANCE = 0.6;
const DEFAULT_LINE_HEIGHT = 1.32;
const ELLIPSIS = '...';

function glyphMapFor(atlas: MsdfAtlasJson): Map<number, MsdfGlyph> {
	return new Map(atlas.glyphs.map((glyph) => [glyph.unicode, glyph]));
}

function asciiFallback(char: string): string {
	const codepoint = char.codePointAt(0) ?? FALLBACK_CODEPOINT;
	if (codepoint >= ASCII_MIN && codepoint <= ASCII_MAX) {
		return char;
	}
	return '?';
}

function truncateLine(line: string, maxChars: number | undefined): string {
	if (maxChars === undefined || maxChars < 0) {
		return line;
	}
	if (maxChars === 0) {
		return '';
	}

	const chars = Array.from(line, asciiFallback);
	if (chars.length <= maxChars) {
		return chars.join('');
	}
	if (maxChars <= ELLIPSIS.length) {
		return '.'.repeat(maxChars);
	}
	return `${chars.slice(0, maxChars - ELLIPSIS.length).join('')}${ELLIPSIS}`;
}

function preparedLines(text: string, maxChars: number | undefined): string[] {
	return text.split('\n').map((line) => truncateLine(line, maxChars));
}

/**
 * Layout ASCII-only MSDF glyph instances for the checked-in JetBrains Mono atlas.
 *
 * Coordinates are baseline-relative with +Y up. Atlas UVs are packed with the required
 * V flip: atlas yOrigin is bottom, while GPU texture V is top-down. Newlines reset penX
 * and move the baseline down by lineHeight. Advance is always 0.6em by default, even
 * for spaces and glyphs without plane bounds.
 */
export function layoutText(text: string, atlas: MsdfAtlasJson, options: LayoutTextOptions = {}): GlyphInstance[] {
	const glyphs = glyphMapFor(atlas);
	const fallback = glyphs.get(FALLBACK_CODEPOINT);
	const atlasWidth = atlas.atlas.width;
	const atlasHeight = atlas.atlas.height;
	const advance = options.advance ?? DEFAULT_ADVANCE;
	const lineHeight = options.lineHeight ?? atlas.metrics?.lineHeight ?? DEFAULT_LINE_HEIGHT;
	const maxChars = options.maxWidthEm === undefined ? undefined : Math.max(0, Math.floor(options.maxWidthEm / advance));

	const instances: GlyphInstance[] = [];
	let penY = 0;

	for (const line of preparedLines(text, maxChars)) {
		let penX = 0;
		for (const rawChar of Array.from(line)) {
			const char = asciiFallback(rawChar);
			const codepoint = char.codePointAt(0) ?? FALLBACK_CODEPOINT;
			const glyph = glyphs.get(codepoint) ?? fallback;

			if (glyph?.planeBounds && glyph.atlasBounds) {
				const plane = glyph.planeBounds;
				const atlasBounds = glyph.atlasBounds;
				const u = atlasBounds.left / atlasWidth;
				const v = 1 - atlasBounds.top / atlasHeight;
				const uw = (atlasBounds.right - atlasBounds.left) / atlasWidth;
				const vh = 1 - atlasBounds.bottom / atlasHeight - v;

				instances.push({
					x: penX + plane.left,
					y: penY + plane.bottom,
					w: plane.right - plane.left,
					h: plane.top - plane.bottom,
					u,
					v,
					uw,
					vh
				});
			}

			penX += advance;
		}
		penY -= lineHeight;
	}

	return instances;
}

export const layoutMsdfText = layoutText;
