// ─────────────────────────────────────────────────────────────────────────────
// REASONING — Observable Decision Trace pass.
//
// Renders a deep_reference response as a LEFT-TO-RIGHT causal trace, entirely in
// MSDF text + light quad geometry, so the viewer watches a decision FORM from
// memory rather than reading a report:
//
//   query ─▶ [8 gate lenses along a beam] ─▶ evidence constellation (role lanes)
//         ─▶ contradiction / supersession events ─▶ recommendation nucleus ─▶ receipt
//
// Every bright object maps to a real response field. Honest empty states: no
// evidence → a calm prompt; no contradictions → the challenge lane simply stays
// empty (never faked). This pass owns its own TextLayerPass — no protected code.
//
// Signal → visual (value-suppressing: uncertainty desaturates toward neutral,
// never jitters):
//   trust      → glyph brightness + saturation (low trust = muted grey-cyan)
//   role       → horizontal lane (primary near beam, contradicting below, …)
//   confidence → recommendation nucleus size + a "LOCKED"/"OPEN" seal
// ─────────────────────────────────────────────────────────────────────────────

import type { FramePass } from '$lib/observatory/engine';
import type { ObservatoryEngine } from '$lib/observatory/engine';
import type { RoutePick } from '$lib/observatory/RouteStage.svelte';
import type { RouteSceneModel } from '$lib/observatory/route-scene';
import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
import { rgb01 } from '$lib/observatory/cognitive-palette';
import type {
	ReasoningScene,
	NormalizedEvidence
} from '$lib/observatory/reasoning/reasoning-scene';
import { createReasoningGeometryPass } from '$lib/observatory/reasoning/reasoning-geometry-pass';

// ── Palette (base hues; trust desaturates them toward neutral) ───────────────
type RGB = readonly [number, number, number];
const asRGB = (hex: string): RGB => {
	const c = rgb01(hex);
	return [c[0], c[1], c[2]];
};
const CYAN = asRGB('#00F5D4'); // forward causal signal / primary evidence
const MINT = asRGB('#9DFFEB'); // supporting / cool flow
const SCARLET = asRGB('#FF3B30'); // contradiction / veto
const AMBER = asRGB('#FFD166'); // supersession / caution
const LUCIFERIN = asRGB('#E9FFB7'); // hot recommendation
const NEUTRAL: RGB = [0.42, 0.46, 0.52]; // the grey uncertainty collapses toward
const HUD_DIM = asRGB('#6B7A88');

// ── Left-to-right gate layout (GPT-5.6-sol tuned). 8 gates as lenses along a
// causal beam, spacing ACCELERATES toward synthesis/recommendation where the
// decision forms. x spans query(left) → receipt(right).
const GATE_X = [-0.86, -0.526, -0.214, 0.074, 0.335, 0.562, 0.747, 0.86];
const BEAM_Y = 0.0;
// Short gate captions so they never collide along the beam (full labels live in
// the aria/pick payload). Ordered to match reasoning-scene STAGE order.
const GATE_SHORT = ['INTENT', 'RETRIEVE', 'ACTIVATE', 'EVIDENCE', 'CHALLENGE', 'SYNTH', 'DECIDE', 'SEAL'];

// ── Evidence role lanes (y-centers, GPT-tuned). primary hugs the beam;
// contradicting sits BELOW (oppositional); superseded lowest (fading); support above.
const LANE_Y: Record<NormalizedEvidence['role'], number> = {
	primary: 0.04,
	supporting: 0.24,
	contradicting: -0.22,
	superseded: -0.48
};
const LANE_STEP = 0.052; // per-row stacking within a lane (legible, non-overlapping)
const LANE_MAX = 6; // rows shown per lane before a supernode summary

const smoothstep = (t: number) => t * t * (3 - 2 * t);

/** Value-suppressing palette (Correll/Moritz/Heer): uncertainty collapses hue →
 *  neutral grey via smoothstep and dims alpha, never jitters, never invisible.
 *  High trust = vivid + solid. */
function trustColor(base: RGB, trust: number): [number, number, number, number] {
	const s = smoothstep(Math.max(0, Math.min(1, trust)));
	const bright = 0.72 + 0.28 * s;
	const r = (NEUTRAL[0] + (base[0] - NEUTRAL[0]) * s) * bright;
	const g = (NEUTRAL[1] + (base[1] - NEUTRAL[1]) * s) * bright;
	const b = (NEUTRAL[2] + (base[2] - NEUTRAL[2]) * s) * bright;
	const alpha = 0.4 + 0.6 * s; // floor so low-trust still reads as evidence
	return [Math.min(1, r), Math.min(1, g), Math.min(1, b), alpha];
}

function laneBase(role: NormalizedEvidence['role']): RGB {
	switch (role) {
		case 'primary':
			return CYAN;
		case 'supporting':
			return MINT;
		case 'contradicting':
			return SCARLET;
		case 'superseded':
			return AMBER;
	}
}

function ascii(v: string): string {
	return v
		.replace(/[—–]/g, '-')
		.replace(/[‘’]/g, "'")
		.replace(/[“”]/g, '"')
		.replace(/…/g, '...')
		.replace(/[^\x20-\x7E]/g, '?');
}

const REVEAL_ANCHOR = -100000;

type TracePickPayload = {
	ariaLabel: string;
	preview?: string;
};

class ReasoningTracePass implements FramePass {
	private text: TextLayerPass;
	private scene: ReasoningScene | null = null;
	private ready = false;
	private initPromise: Promise<void> | null = null;

	constructor(engine: ObservatoryEngine) {
		this.text = new TextLayerPass(engine);
	}

	uploadScene(scene: RouteSceneModel): void {
		this.scene = (scene as ReasoningScene)?.organ === 'reasoning' ? (scene as ReasoningScene) : null;
		void this.ensureReady().then(() => this.text.setText(this.build()));
	}

	render(pass: GPURenderPassEncoder): void {
		this.text.render(pass);
	}

	pickAt(ndcX: number, ndcY: number): RoutePick | null {
		return this.text.pickAt(ndcX, ndcY);
	}

	dispose(): void {
		this.text.dispose();
	}

	private async ensureReady(): Promise<void> {
		if (!this.initPromise) this.initPromise = this.text.init().then(() => void (this.ready = true));
		await this.initPromise;
	}

	// ── Compose the whole trace as MSDF items ─────────────────────────────────
	private build(): TextLayerItem[] {
		const s = this.scene;
		if (!s || !s.alive) return []; // RouteStage draws the empty label

		const items: TextLayerItem[] = [];
		const query = ascii(String(s.raw?.query ?? '')).slice(0, 60);

		// 1. QUESTION IGNITION — the query, sharp, upper-left, feeding the beam.
		items.push({
			id: 'trace:query',
			kind: 'trace-query',
			text: query ? `> ${query}` : '> (trace)',
			x: -0.94,
			y: 0.82,
			size: 0.03,
			color: [...LUCIFERIN, 1],
			weight: 0.95,
			depth: 1,
			startFrame: REVEAL_ANCHOR,
			revealSpan: 1,
			maxWidthEm: 60
		});

		// 2. EIGHT GATE LENSES along the beam. Short captions ALTERNATE above/below
		// the beam so adjacent gates never collide; the beam node sits ON the beam.
		const gates = s.stages ?? [];
		for (let i = 0; i < gates.length && i < GATE_X.length; i++) {
			const g = gates[i];
			const lit = g?.lit;
			const gx = GATE_X[i];
			const short = GATE_SHORT[i] ?? ascii((g?.kind ?? '').toUpperCase());
			const fullLabel = ascii((g?.label ?? g?.kind ?? '').toUpperCase());
			const col: [number, number, number, number] = lit ? [...CYAN, 1] : [...HUD_DIM, 0.5];
			const above = i % 2 === 0; // alternate to avoid neighbour overlap
			const labelY = above ? BEAM_Y + 0.055 : BEAM_Y - 0.075;
			// caption centered under/over its node (short * ~0.62em advance)
			const halfW = short.length * 0.014 * 0.62 * 0.5;
			items.push({
				id: `trace:gate:${g?.kind ?? i}`,
				kind: 'trace-gate',
				text: short,
				x: gx - halfW,
				y: labelY,
				size: 0.014,
				color: col,
				weight: lit ? 0.9 : 0.6,
				depth: lit ? 1 : 0.7,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 1,
				maxWidthEm: 14,
				hitPadX: 0.055,
				hitPadY: 0.08,
				...({ ariaLabel: `Gate ${fullLabel}: ${g?.count ?? 0} items, ${Math.round((g?.confidence ?? 0) * 100)}% confidence` } as Partial<TracePickPayload>)
			});
			// gate node ON the beam — @ when lit, faint tick when not
			items.push({
				id: `trace:gatenode:${i}`,
				kind: 'trace-beam',
				text: lit ? 'O' : '+',
				x: gx - 0.008,
				y: BEAM_Y - 0.012,
				size: lit ? 0.028 : 0.02,
				color: col,
				depth: lit ? 1 : 0.6,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 1,
				maxWidthEm: 3
			});
			// beam connector to the next gate (a dotted run along y=BEAM)
			if (i < GATE_X.length - 1) {
				items.push({
					id: `trace:beamseg:${i}`,
					kind: 'trace-beam',
					text: '·······',
					x: gx + 0.02,
					y: BEAM_Y - 0.006,
					size: 0.014,
					color: lit ? [...CYAN, 0.5] : [...HUD_DIM, 0.25],
					depth: 0.7,
					startFrame: REVEAL_ANCHOR,
					revealSpan: 1,
					maxWidthEm: 40
				});
			}
		}

		// 3. EVIDENCE CONSTELLATION — role lanes, trust-encoded, capped w/ supernode.
		const byRole = new Map<NormalizedEvidence['role'], NormalizedEvidence[]>();
		for (const e of s.evidence ?? []) {
			const arr = byRole.get(e.role) ?? [];
			arr.push(e);
			byRole.set(e.role, arr);
		}
		for (const [role, list] of byRole) {
			const laneY = LANE_Y[role];
			const shown = list.slice(0, LANE_MAX);
			shown.forEach((e, i) => {
				// Center the rows around the lane y (GPT-tuned) so a lane reads as a
				// tight cluster, not a one-directional list drifting off-screen.
				const y = laneY + (i - (shown.length - 1) / 2) * LANE_STEP;
				const preview = ascii(e.preview).replace(/\s+/g, ' ').trim().slice(0, 46);
				items.push({
					id: `trace:ev:${e.id}`,
					kind: 'trace-evidence',
					text: `${preview} · ${Math.round(e.trust * 100)}%`,
					// evidence sits under the evidence gate (index 3) and fans right
					x: GATE_X[3] - 0.02,
					y,
					size: 0.016,
					color: trustColor(laneBase(role), e.trust),
					weight: 0.4 + 0.5 * e.trust,
					depth: 0.6 + 0.4 * e.trust,
					startFrame: REVEAL_ANCHOR,
					revealSpan: 1,
					maxWidthEm: 52,
					hitPadX: 0.03,
					hitPadY: 0.02,
					...({ ariaLabel: `${role} evidence, trust ${Math.round(e.trust * 100)}%: ${preview}`, preview } as Partial<TracePickPayload>)
				});
			});
			// honest supernode for the remainder
			if (list.length > LANE_MAX) {
				items.push({
					id: `trace:super:${role}`,
					kind: 'trace-supernode',
					text: `+${list.length - LANE_MAX} more ${role}`,
					x: GATE_X[3] - 0.02,
					y: laneY - (LANE_MAX / 2 + 1) * LANE_STEP,
					size: 0.014,
					color: [...HUD_DIM, 0.7],
					depth: 0.6,
					startFrame: REVEAL_ANCHOR,
					revealSpan: 1,
					maxWidthEm: 30
				});
			}
			// lane label
			items.push({
				id: `trace:lanelabel:${role}`,
				kind: 'trace-hud',
				text: role.toUpperCase(),
				x: GATE_X[3] - 0.16,
				y: laneY,
				size: 0.013,
				color: [...laneBase(role), 0.55],
				depth: 0.7,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 1,
				maxWidthEm: 14
			});
		}

		// 4. RECOMMENDATION NUCLEUS — at the recommendation gate; size/label = confidence.
		if (s.recommended) {
			const conf = Math.max(0, Math.min(1, s.recommended.trust_score));
			const answer = ascii(s.recommended.answer_preview).replace(/\s+/g, ' ').trim().slice(0, 48);
			const nucleusX = GATE_X[6];
			const locked = conf >= 0.6;
			items.push({
				id: 'trace:nucleus',
				kind: 'trace-recommendation',
				text: locked ? 'O' : 'o',
				x: nucleusX,
				y: BEAM_Y,
				size: 0.05 + 0.05 * conf,
				color: trustColor(LUCIFERIN, conf),
				weight: 0.6 + 0.4 * conf,
				depth: 1,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 1,
				maxWidthEm: 4,
				hitPadX: 0.06,
				hitPadY: 0.06,
				...({ ariaLabel: `Recommendation, ${Math.round(conf * 100)}% confidence: ${answer}`, preview: answer } as Partial<TracePickPayload>)
			});
			items.push({
				id: 'trace:reclabel',
				kind: 'trace-hud',
				text: `${locked ? 'LOCKED' : 'OPEN'} · ${Math.round(conf * 100)}%`,
				x: nucleusX - 0.05,
				y: BEAM_Y - 0.11,
				size: 0.016,
				color: trustColor(LUCIFERIN, conf),
				depth: 1,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 1,
				maxWidthEm: 18
			});
			// the answer itself, wrapped below the nucleus
			items.push({
				id: 'trace:answer',
				kind: 'trace-hud',
				text: answer,
				x: nucleusX - 0.28,
				y: BEAM_Y - 0.17,
				size: 0.016,
				color: [...LUCIFERIN, 0.92],
				depth: 0.95,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 1,
				maxWidthEm: 44,
				maxLines: 2
			});
		}

		// 5. RECEIPT SEAL — at the receipt gate; a thin engraved marker, clickable.
		const recCount = (s.evidence?.length ?? 0);
		items.push({
			id: 'trace:receipt',
			kind: 'trace-receipt',
			text: `[ ${recCount} SOURCES SEALED ]`,
			x: GATE_X[7] - 0.14,
			y: BEAM_Y - 0.07,
			size: 0.015,
			color: [...MINT, 0.85],
			depth: 0.9,
			startFrame: REVEAL_ANCHOR,
			revealSpan: 1,
			maxWidthEm: 24,
			hitPadX: 0.05,
			hitPadY: 0.04,
			...({ ariaLabel: `Composition receipt: ${recCount} sources, analysed ${s.raw?.memoriesAnalyzed ?? recCount} memories` } as Partial<TracePickPayload>)
		});

		return items;
	}
}

export function createReasoningTracePasses(
	engine: ObservatoryEngine,
	_scene: RouteSceneModel
): FramePass[] {
	// Geometry (beam / ribbons / nucleus / fringe / scars) renders FIRST, then the
	// MSDF text pass on top so labels stay crisp above the glowing field.
	return [createReasoningGeometryPass(engine), new ReasoningTracePass(engine)];
}
