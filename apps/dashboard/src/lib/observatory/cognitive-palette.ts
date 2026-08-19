/**
 * Cognitive Bioluminescent Cortex — the ONE source of truth for the dashboard's
 * invented visual language (design council: Opus 4.8 × GPT-5.5, Jul 8 2026).
 *
 * The dashboard is a dark local brain in a jar. Routes are organs viewed through
 * different instruments. This is NOT purple-on-black. The base is blackwater +
 * oil-film + enzymatic light + scarlet immune heat. Every color MEANS something
 * on a Vestige-only substrate (the discipline test): retention is oxygen, trust
 * is membrane thickness, causality is a retrograde axon, suppression is scar.
 *
 * Rule of the language: magenta is reserved EXCLUSIVELY for backward causality
 * (RSB). Indigo appears ONLY as bitemporal transaction-time parallax. Neither is
 * ever a route accent. Purple, as a flat brand chrome, does not exist here.
 *
 * All values are linear-ish sRGB hex; the helpers return 0..1 rgb for WGSL
 * uniforms. Keep this file free of Svelte/DOM imports so shaders + adapters +
 * components can all share it.
 */

/** Parse '#rrggbb' → [r,g,b] in 0..1. Falls back to blackwater on bad input. */
export function rgb01(hex: string): [number, number, number] {
	const m = /^#?([0-9a-fA-F]{6})$/.exec(hex.trim());
	if (!m) return [0x02 / 255, 0x03 / 255, 0x07 / 255];
	const v = parseInt(m[1], 16);
	return [((v >> 16) & 0xff) / 255, ((v >> 8) & 0xff) / 255, (v & 0xff) / 255];
}

// ---------------------------------------------------------------------------
// Base medium — the blackwater the whole organism lives in.
// ---------------------------------------------------------------------------
export const MEDIUM = {
	blackwater: '#020307', // absolute background; NEVER tinted purple
	anaerobic: '#07100D', // nutrient medium, green-black low field
	cyanFog: '#0B171B', // deep cyan-black parallax fog
	sediment: '#11140A' // old-memory amber-black sediment
} as const;

// ---------------------------------------------------------------------------
// Living memory / FSRS retention — the oxygen gradient.
// sediment → amber debt → healthy green → luciferin white as retention rises.
// ---------------------------------------------------------------------------
export const RETENTION = {
	luciferin: '#E9FFB7', // >= 0.86 newly retrievable, hot white-green
	healthy: '#A8FF5E', // 0.65–0.86 stable
	recall: '#29F2A9', // active recall / excitation wave (activation > p90)
	bridge: '#1BD6FF', // remote association / semantic bridge
	latent: '#315CFF', // deep latent, low activation but structurally present
	debt: '#8A4B18', // 0.25–0.45 forgetting debt / dehydrated trace
	extinction: '#2A160B' // < 0.25 near-extinction sediment
} as const;

/** Ordered retention stops (low→high) for a continuous WGSL ramp. */
export const RETENTION_RAMP: readonly string[] = [
	RETENTION.extinction,
	RETENTION.debt,
	RETENTION.healthy,
	RETENTION.luciferin
];

// ---------------------------------------------------------------------------
// Trust / verifier / immune system — membranes and macrophages.
// ---------------------------------------------------------------------------
export const IMMUNE = {
	trustMembrane: '#F4F1D0', // high-trust warm ivory edge
	caution: '#FFD166', // caution verdict / yellow immune flare
	veto: '#FF3B30', // veto / contradiction injury
	suppressionScar: '#B90D2B', // permanent red-black suppression lacquer
	labile: '#FF7A1A' // reversible labile suppression window
} as const;

// ---------------------------------------------------------------------------
// Causality / RSB — the signature. Magenta lives ONLY here.
// ---------------------------------------------------------------------------
export const CAUSAL = {
	forward: '#00F5D4', // forward recall signal
	retrograde: '#FF2DF7', // backward causal backfill axon — RSB ONLY
	receiptSpark: '#FFFFFF' // one-frame causal-receipt proof spark at edge write
} as const;

// ---------------------------------------------------------------------------
// Bitemporal / audit — growth rings. Indigo lives ONLY here.
// ---------------------------------------------------------------------------
export const BITEMPORAL = {
	validRing: '#6BFFB8', // valid-time growth ring
	txShadow: '#7C6CFF', // transaction-time shadow (indigo, parallax only)
	supersession: '#FFB000' // supersession amber cut-line
} as const;

// ---------------------------------------------------------------------------
// System health / stats — vitals.
// ---------------------------------------------------------------------------
export const VITALS = {
	throughput: '#9DFFEB', // cool system flow
	backlog: '#FF4FD8' // backlog pressure — ONLY when queue growth is real
} as const;

// ---------------------------------------------------------------------------
// Retention → color (continuous). Mirrors the WGSL ramp so CPU labels and GPU
// cells agree. `r` is retrievability 0..1.
// ---------------------------------------------------------------------------
export function retentionColor(r: number): [number, number, number] {
	const t = Math.max(0, Math.min(1, r));
	// piecewise across the 4 ramp stops
	const stops = RETENTION_RAMP.map(rgb01);
	const seg = t * (stops.length - 1);
	const i = Math.min(stops.length - 2, Math.floor(seg));
	const f = seg - i;
	const a = stops[i];
	const b = stops[i + 1];
	return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ---------------------------------------------------------------------------
// SALIENCE PALETTE — "a grey mind that spends color only where it decides
// something matters." The shared ramp for the resting-cortex organs
// (observatory / explore / memories / feed). Unlike retentionColor (which is
// green even at the low end), this DESATURATES the crowd toward cold graphite
// and lets a memory EARN saturated luciferin, then gold, then white-hot as its
// salience rises. Salience here = the decision-weight of a memory (importance /
// activation / retention), 0..1 — NOT raw retention alone. The perceptual point:
// a first-time viewer sees a dim grey field where only the memories the system
// currently cares about carry color, and the very top ones blaze. Pure function,
// deterministic, no DOM — safe to call from mappers + (mirrored) from WGSL.
// ---------------------------------------------------------------------------

/** Cold graphite the unselected crowd desaturates toward (never pure black so
 *  the cell still reads as a node, not a hole). */
const GRAPHITE: [number, number, number] = [0x3a / 255, 0x44 / 255, 0x4c / 255];
/** The gold flare a high-salience memory earns before white-hot (matches the
 *  render-shader gold in the waitlist engine: energy>2 → gold → white). */
const SALIENCE_GOLD: [number, number, number] = [1.0, 0.78, 0.36];

function mix3(
	a: [number, number, number],
	b: [number, number, number],
	t: number
): [number, number, number] {
	const f = t < 0 ? 0 : t > 1 ? 1 : t;
	return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/**
 * salience → color for a resting-cortex cell. `s` is 0..1 decision-weight.
 *  - s < ~0.35  → cold graphite (the crowd; color is spent, not given)
 *  - mid        → the retention ramp fades UP out of grey (green→luciferin)
 *  - s > ~0.82  → ignites toward gold, then white-hot at the very top
 * `rescued` forces the gold→white ignition regardless of `s` (the salience
 * "vote" pulling an about-to-be-forgotten memory back into the light).
 */
export function saliencePalette(s: number, rescued = false): [number, number, number] {
	const t = Math.max(0, Math.min(1, Number.isFinite(s) ? s : 0));
	// The crowd is graphite; color earns in above ~0.30 and saturates by ~0.72.
	const earn = smooth01((t - 0.3) / 0.42);
	let col = mix3(GRAPHITE, retentionColor(t), earn);
	// Top salience flares gold, then blows to white-hot at the very peak.
	const flare = smooth01((t - 0.82) / 0.18);
	col = mix3(col, SALIENCE_GOLD, flare * 0.85);
	if (rescued) {
		// The rescued memory is the winner of the vote: force gold→white ignition.
		col = mix3(col, SALIENCE_GOLD, 0.7);
		col = mix3(col, [1, 1, 1], 0.35);
	} else {
		col = mix3(col, [1, 1, 1], smooth01((t - 0.93) / 0.07) * 0.5);
	}
	return col;
}

/** Salience → glow energy (0..1) for the same resting-cortex cells. The crowd
 *  keeps a low ember; salient cells brighten; rescued cells blaze. */
export function salienceEnergy(s: number, rescued = false): number {
	const t = Math.max(0, Math.min(1, Number.isFinite(s) ? s : 0));
	if (rescued) return 1;
	return 0.18 + 0.72 * smooth01((t - 0.15) / 0.7);
}

function smooth01(x: number): number {
	const t = x < 0 ? 0 : x > 1 ? 1 : x;
	return t * t * (3 - 2 * t);
}

/**
 * Event type → the impulse color it injects into the organism. Drives the Feed
 * bloodstream, the LiveBridge reactions, and per-organ event pulses. Only real
 * VestigeEvent variants appear here (the discipline test lives at the source).
 */
export const EVENT_IMPULSE: Record<string, string> = {
	MemoryCreated: RETENTION.healthy, // cells condense from nutrient noise
	SearchPerformed: CAUSAL.forward, // cyan chemoattractant wave
	ActivationSpread: RETENTION.recall, // green/teal excitation along real paths
	ImportanceScored: BITEMPORAL.supersession, // gold-white enzyme deposit
	RetentionDecayed: RETENTION.debt, // amber dehydration front
	ConnectionDiscovered: CAUSAL.forward, // a new axon grows
	DeepReferenceCompleted: RETENTION.luciferin, // the reasoning organ lights
	BackfillFired: CAUSAL.retrograde, // magenta retrograde axon
	CausalReceipt: CAUSAL.retrograde,
	MemorySuppressed: IMMUNE.suppressionScar, // macrophage engulfs the cell
	MemoryUnsuppressed: IMMUNE.labile,
	MemoryPromoted: RETENTION.luciferin,
	MemoryDemoted: RETENTION.debt,
	MemoryPrOpened: IMMUNE.caution, // translucent immune proposal capsule
	MemoryPrDecided: IMMUNE.trustMembrane,
	HookVerdictRecorded: IMMUNE.veto,
	TraceEvent: CAUSAL.forward,
	DreamStarted: BITEMPORAL.txShadow,
	DreamCompleted: BITEMPORAL.validRing,
	Rac1CascadeSwept: IMMUNE.suppressionScar
};

/** Impulse rgb for a VestigeEvent type (blackwater fallback for unknowns). */
export function eventImpulse01(type: string): [number, number, number] {
	return rgb01(EVENT_IMPULSE[type] ?? MEDIUM.blackwater);
}

/**
 * Trust → membrane thickness (world units, matched to the metaball edge width in
 * the field pass). complete evidence = continuous thick ring; low trust = thin
 * perforated ring. `trust` is 0..1.
 */
export function membraneWidth(trust: number): number {
	const t = Math.max(0, Math.min(1, trust));
	return 0.003 + (0.018 - 0.003) * t;
}
