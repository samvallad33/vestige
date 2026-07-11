// Spatial Palace — floating organ labels.
//
// The palace renders 19 organ orbs as a slowly-orbiting 3D constellation. Each
// orb needs its NAME floating beside it in crisp MSDF, tracking the orb as the
// camera orbits. This module is the CPU bridge between the 3D node field and the
// 2D TextLayerPass: given each node's already-projected screen anchor
// (`{ href, ndcX, ndcY, depth, radius }` from PalaceNodePass.getScreenPositions())
// it emits one allocation-light TextLayerItem per visible organ, offset up/right
// so the caption sits beside the glow (never on top of it), sized + brightened by
// depth (nearer = bigger + more opaque), and colored by cognitive family.
//
// No projection is done here — the pass owns the camera and the NodeState
// readback, so it already knows exactly where each orb landed. Keeping the math
// there (one place) means labels and orbs can never disagree about a node's
// position. This file is pure data-shaping: screen anchor -> TextLayerItem.

import type { TextLayerItem } from '$lib/observatory/text/text-layer';
import { rgb01 } from '$lib/observatory/cognitive-palette';
import { ORGAN_REGIONS, regionByHref, type OrganRegion } from '$lib/observatory/palace-map';

/**
 * A single organ orb's on-screen footprint, as produced each frame by
 * PalaceNodePass.getScreenPositions(). All values are in LOGICAL NDC — i.e. the
 * pre-aspect-divide NDC space TextLayerItem.x/y live in — so the label anchor can
 * be handed straight to the text pass with no further transform. (The text
 * shader applies the same square-in-both-orientations aspect divide that
 * pickAt() mirrors, so a logical-NDC anchor lands exactly over the orb.)
 */
export interface OrganScreenPos {
	/** Organ href === the palace node id. */
	href: string;
	/** Logical-NDC center of the orb, +Y up (matches TextLayerItem.x/y space). */
	ndcX: number;
	ndcY: number;
	/**
	 * 0..1 nearness: 1 = closest orb to camera, 0 = farthest. Drives label size,
	 * brightness, and the text pass's own depth/crispness channel. The pass
	 * computes this by normalizing clip-w across the frame's nodes (near = small
	 * clip-w = 1.0).
	 */
	depth: number;
	/**
	 * Projected orb radius in logical-NDC-Y units (r * f / clip_w). Used to push
	 * the label just outside the glowing disc so text never sits on the orb.
	 * Optional — a sane default is used when the pass can't supply it.
	 */
	radius?: number;
	/** False when the orb is behind the camera / off-frame; label is skipped. */
	visible?: boolean;
}

/**
 * Family -> label color. The orbs themselves are colored by the FSRS retention
 * ramp (via FAMILY_RETENTION in palace-map), so the LABELS reuse the same family
 * language in the palette's own hues: cyan/green for reasoning+memory, amber +
 * scarlet for the immune organs, indigo for the temporal organs, bright cool for
 * signal, neutral ivory for system. rgba, alpha is a per-family base that depth
 * then modulates. Kept as prebuilt tuples so the per-frame path allocates nothing.
 */
// The captions sit ON TOP of the bright additive orb bloom. The HUD title/
// subtitle prove HOT-WHITE at FULL alpha punches through the bloom (they read
// crisp in the corner AND over orbs). The old organ labels washed out not from
// color but because they were TINY (0.017-0.03, half the subtitle) and
// alpha-DIMMED (far labels 0.72, hovered-dim 0.32) — faint small grey on bloom.
// Fix (Jul 10 2026, Sam): pure hot-white at FULL alpha + much larger sizes +
// heavy stroke, so every caption reads at a glance. (A dark-ink experiment
// vanished into the depth-of-field bloom — bright is the proven path here.)
const FAMILY_LABEL_COLOR: Record<OrganRegion['family'], [number, number, number, number]> = {
	reasoning: [...rgb01('#FFFFFF'), 1],
	memory: [...rgb01('#FFFFFF'), 1],
	immune: [...rgb01('#FFFFFF'), 1],
	temporal: [...rgb01('#FFFFFF'), 1],
	signal: [...rgb01('#FFFFFF'), 1],
	system: [...rgb01('#FFFFFF'), 1]
};

/** The Observatory core caption — pure hot white, reads first. */
const CENTER_LABEL_COLOR: [number, number, number, number] = [...rgb01('#FFFFFF'), 1];

// --- Label geometry (all logical-NDC-Y units) --------------------------------
// A label floats up-and-right of its orb, just clear of the glowing disc. The
// glow's visible halo extends past the geometric radius (render-nodes halo term),
// so we clear it by a multiple of the projected radius plus a floor.
const RADIUS_CLEAR = 1.35; // push label this * projected-radius off the orb
const OFFSET_FLOOR = 0.028; // minimum clearance when a node projects tiny
const OFFSET_UP = 0.6; // vertical share of the clearance (up-right diagonal)
const OFFSET_RIGHT = 0.85; // horizontal share of the clearance

// Depth -> size/brightness. Nearer orbs get bigger, brighter captions so the
// constellation reads with real front-to-back depth instead of a flat wall.
// Sizes bumped ~1.7x (Jul 10 2026, Sam): the old 0.017-0.03 captions were half
// the subtitle size (0.032) and washed out on the bloom. Now the nearest label
// out-sizes the subtitle so the nav reads at a glance; the farthest still reads.
const SIZE_NEAR = 0.05; // NDC-Y per em for the closest orb (was 0.03)
const SIZE_FAR = 0.03; // ...for the farthest (was 0.017)
// Dark ink only punches through the bright bloom at HIGH alpha (over-blend drives
// dst→0 as alpha→1), so keep even the far labels near-opaque.
const ALPHA_NEAR = 1.0;
const ALPHA_FAR = 0.95; // was 0.72 — dark ink needs high alpha to read

// Reusable scratch item pool — reset + refilled each frame so steady-state label
// updates allocate nothing (the hot path calls this ~60x/s). Grown lazily to the
// organ count; never shrinks.
const POOL: TextLayerItem[] = [];
// Scratch rects for the per-frame label de-confliction pass (cleared each build).
const PLACED: Array<{ x0: number; x1: number; y: number; size: number }> = [];

function lerp(a: number, b: number, t: number): number {
	return a + (b - a) * t;
}
function clamp01(v: number): number {
	return v < 0 ? 0 : v > 1 ? 1 : Number.isFinite(v) ? v : 0;
}

/**
 * Build the floating organ captions for this frame.
 *
 * @param positions per-orb screen anchors from PalaceNodePass.getScreenPositions()
 * @param opts.hoveredHref  the organ under the cursor — rendered hotter/bigger
 * @param opts.dimUnhovered when a node is hovered, fade the rest so the target pops
 * @returns a stable-length TextLayerItem[] (backed by a reused pool) ready for
 *          textPass.setText(). Length varies only with how many orbs are visible.
 */
export function buildOrganLabels(
	positions: OrganScreenPos[],
	opts: { hoveredHref?: string | null; dimUnhovered?: boolean } = {}
): TextLayerItem[] {
	const hovered = opts.hoveredHref ?? null;
	const dim = opts.dimUnhovered ?? true;

	let n = 0;
	// Per-frame placed-label rects for greedy de-confliction (deterministic order
	// → capture-stable). Reused scratch to keep the hot path allocation-light.
	PLACED.length = 0;
	for (let i = 0; i < positions.length; i++) {
		const p = positions[i];
		if (p.visible === false) continue;
		const region = regionByHref(p.href);
		if (!region) continue;

		const depth = clamp01(p.depth);
		const isHover = hovered !== null && p.href === hovered;

		// Offset the label clear of the glowing disc (up-and-right diagonal).
		// getScreenPositions() doesn't expose a projected radius, so scale the
		// clearance by DEPTH: nearer orbs (depth→1) project larger and bloom
		// brighter, so their captions sit further off the halo.
		const clear = OFFSET_FLOOR + depth * RADIUS_CLEAR;
		let x = p.ndcX + clear * OFFSET_RIGHT;
		let y = p.ndcY + clear * OFFSET_UP;

		// Size + brightness ride depth; hover overrides to a hot readout.
		let size = lerp(SIZE_FAR, SIZE_NEAR, depth);
		let alpha = lerp(ALPHA_FAR, ALPHA_NEAR, depth);
		if (region.center) size *= 1.22; // the core caption reads first
		if (isHover) {
			size *= 1.28;
			alpha = 1;
		} else if (dim && hovered !== null) {
			alpha *= 0.32; // fade the field so the hovered organ pops
		}

		// ── Edge clamp: text is left-anchored and grows rightward, so a label near
		// the right screen edge clips ('TIMELINE' → 'TIMELIN'). If the estimated
		// run would cross the conservative logical bound, FLIP it to the left of
		// the orb instead. (0.62em ≈ measured average advance of the mono atlas.)
		const estWidth = region.label.length * size * 0.62;
		if (x + estWidth > 0.97) {
			x = p.ndcX - clear * OFFSET_RIGHT - estWidth;
		}
		// ── De-confliction: when two orbs project close together their captions
		// stack unreadably. Greedy pass in deterministic order: if this label's
		// rect overlaps an already-placed one, nudge it BELOW that label.
		for (let k = 0; k < PLACED.length; k++) {
			const r = PLACED[k];
			const overlapX = x < r.x1 && x + estWidth > r.x0;
			const overlapY = Math.abs(y - r.y) < (size + r.size) * 0.75;
			if (overlapX && overlapY) {
				y = r.y - (size + r.size) * 0.85; // step below the placed label
			}
		}
		PLACED.push({ x0: x, x1: x + estWidth, y, size });

		// Family color, alpha-modulated by depth/hover. Copy the base tuple so the
		// prebuilt FAMILY_LABEL_COLOR constants stay pristine.
		const base = region.center ? CENTER_LABEL_COLOR : FAMILY_LABEL_COLOR[region.family];
		const color: [number, number, number, number] = [base[0], base[1], base[2], clamp01(alpha)];

		// Reuse (or grow) the pooled item — no per-frame object churn.
		let item = POOL[n];
		if (!item) {
			item = {
				id: '',
				kind: 'palace-label',
				text: '',
				x: 0,
				y: 0,
				size: 0,
				color: [0, 0, 0, 0],
				depth: 0,
				weight: 0.75,
				revealSpan: 1,
				maxWidthEm: 24
			};
			POOL[n] = item;
		}
		item.id = 'palace-label:' + p.href;
		item.kind = 'palace-label';
		item.text = region.label; // ASCII-safe already (ORGAN_REGIONS labels are caps A-Z)
		item.x = x;
		item.y = y;
		item.size = size;
		item.color = color;
		// Readability on the bright nebula: keep EVERY caption crisp+bright (high
		// depth) and thick-stroked (high weight) so it punches through the bloom,
		// rather than dimming far labels into the glow. Depth still nudges near
		// orbs a touch crisper for subtle front-to-back feel.
		item.depth = isHover ? 1 : 0.8 + depth * 0.2;
		item.weight = 0.95;
		n++;
	}

	// Hand back exactly the filled prefix of the pool (stable identity, no slice
	// allocation on steady state — setText only reads .length worth of items).
	POOL.length = n;
	return POOL;
}

/**
 * Total organ count — handy for the pass to size its readback / positions array
 * without importing ORGAN_REGIONS itself.
 */
export const ORGAN_LABEL_COUNT = ORGAN_REGIONS.length;
