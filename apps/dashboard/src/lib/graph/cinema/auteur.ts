// The Auteur — the director's brain + the typed shot-plan contract.
//
// The LLM (Tier 1) or the deterministic rule table (Tier 2) produces a
// DirectorPlan: a sequence of cinematographic Shots, one per CinemaBeat, each
// grounded in a real node and justified by a real graph metric. The camera
// runtime (director.ts) executes it. Carry-forward semantics mean a sparse or
// half-hallucinated plan ALWAYS resolves to a coherent film — the same
// robustness pattern as narrator.resolveNarration.

import type { CinemaPath, CinemaBeat } from './pathfinder';
import type { GraphSignals } from './topology';

// ── Camera grammar (string unions keep LLM output validatable) ───────────────
export type Move = 'push_in' | 'pull_back' | 'orbit' | 'crane' | 'whip_pan' | 'rack_focus' | 'hold';
export type Angle = 'eye' | 'low' | 'high'; // low = look up (power); high = look down (decay)
export type Cut = 'fly' | 'hard_cut' | 'match_cut';
export type StormMode = 'anchor' | 'connection' | 'contradiction' | 'surprise';
export type CaptionTone = 'curious' | 'tense' | 'resolved' | 'awe' | 'neutral';
export type ScoreCue = 'motif' | 'minor_drop' | 'major_resolve' | 'silence';
export type Act = 'I' | 'II' | 'III';
export type EmotionalArc = 'man_in_hole' | 'rags_to_riches' | 'icarus' | 'cinderella' | 'oedipus' | 'flat';
export type DirectorSource = 'backend-llm' | 'on-device' | 'deterministic';

/** A directed shot. Only axes that CHANGE need be set — the rest carry forward
 * from the previous resolved shot (ultimate default = today's camera constants). */
export interface Shot {
	nodeId: string; // MUST cite a real node (alignment key + grounding constraint)
	move?: Move;
	angle?: Angle;
	dutch?: number; // camera roll, radians, 0..~0.5
	standoff?: number; // world units
	flightSeconds?: number;
	dwellSeconds?: number;
	halflife?: number; // spring smoothing; 0 = jump-cut
	cut?: Cut;
	stormMode?: StormMode;
	intensity?: number; // 0..1 → scales the ignition spike
	tension?: number; // 0..1 master scalar
	act?: Act;
	tone?: CaptionTone;
	scoreCue?: ScoreCue;
	why: string; // REQUIRED: cites the real metric driving this shot
	viaEdgeKey?: string; // `${source}->${target}` for two-node framing
}

export interface DirectorPlan {
	source: DirectorSource;
	logline: string;
	arc: EmotionalArc;
	shots: Shot[];
}

/** Every axis filled after carry-forward — what the director reads each beat. */
export type ResolvedShot = Required<Omit<Shot, 'viaEdgeKey'>> & { viaEdgeKey?: string };

// Ultimate defaults — today's hardcoded camera constants, so a plan-less or
// fully-sparse run is byte-identical to the pre-Auteur camera.
export const SHOT_DEFAULTS: Omit<ResolvedShot, 'nodeId' | 'why'> = {
	move: 'hold',
	angle: 'eye',
	dutch: 0,
	standoff: 26,
	flightSeconds: 2.4,
	dwellSeconds: 3.2,
	halflife: 0.35,
	cut: 'fly',
	stormMode: 'connection',
	intensity: 0.7,
	tension: 0.3,
	act: 'I',
	tone: 'neutral',
	scoreCue: 'motif',
};

const MOVES: ReadonlySet<Move> = new Set(['push_in', 'pull_back', 'orbit', 'crane', 'whip_pan', 'rack_focus', 'hold']);
const ANGLES: ReadonlySet<Angle> = new Set(['eye', 'low', 'high']);
const CUTS: ReadonlySet<Cut> = new Set(['fly', 'hard_cut', 'match_cut']);
const STORM_MODES: ReadonlySet<StormMode> = new Set(['anchor', 'connection', 'contradiction', 'surprise']);
const TONES: ReadonlySet<CaptionTone> = new Set(['curious', 'tense', 'resolved', 'awe', 'neutral']);
const SCORE_CUES: ReadonlySet<ScoreCue> = new Set(['motif', 'minor_drop', 'major_resolve', 'silence']);
const ACTS: ReadonlySet<Act> = new Set(['I', 'II', 'III']);

function num(v: unknown, lo: number, hi: number, fallback: number): number {
	const n = typeof v === 'number' && Number.isFinite(v) ? v : NaN;
	if (Number.isNaN(n)) return fallback;
	return Math.max(lo, Math.min(hi, n));
}
function pick<T>(v: unknown, set: ReadonlySet<T>, fallback: T): T {
	return typeof v === 'string' && set.has(v as T) ? (v as T) : fallback;
}

/**
 * Resolve a DirectorPlan into one fully-specified ResolvedShot per beat.
 * Aligns by nodeId; every unspecified/garbage axis is back-filled by carry-forward
 * (previous shot → SHOT_DEFAULTS). A shot can NEVER be blank or invalid.
 */
export function resolveShots(plan: DirectorPlan | null, path: CinemaPath): ResolvedShot[] {
	const byNode = new Map<string, Shot>();
	for (const s of plan?.shots ?? []) {
		if (s && typeof s.nodeId === 'string') byNode.set(s.nodeId, s);
	}
	const resolved: ResolvedShot[] = [];
	let prev: ResolvedShot | null = null;
	for (const beat of path.beats) {
		const raw = byNode.get(beat.nodeId);
		const base = prev ?? { ...SHOT_DEFAULTS, nodeId: beat.nodeId, why: '' };
		const shot: ResolvedShot = {
			nodeId: beat.nodeId,
			move: pick(raw?.move, MOVES, base.move),
			angle: pick(raw?.angle, ANGLES, base.angle),
			dutch: num(raw?.dutch, 0, 0.6, base.dutch),
			standoff: num(raw?.standoff, 8, 90, base.standoff),
			flightSeconds: num(raw?.flightSeconds, 0.4, 6, base.flightSeconds),
			dwellSeconds: num(raw?.dwellSeconds, 0.6, 8, base.dwellSeconds),
			halflife: num(raw?.halflife, 0, 1.5, base.halflife),
			cut: pick(raw?.cut, CUTS, 'fly'), // cut never carries forward — default per beat
			stormMode: pick(raw?.stormMode, STORM_MODES, base.stormMode),
			intensity: num(raw?.intensity, 0, 1, base.intensity),
			tension: num(raw?.tension, 0, 1, base.tension),
			act: pick(raw?.act, ACTS, base.act),
			tone: pick(raw?.tone, TONES, base.tone),
			scoreCue: pick(raw?.scoreCue, SCORE_CUES, 'motif'),
			why: typeof raw?.why === 'string' && raw.why.trim() ? raw.why : base.why || 'establishing shot',
			viaEdgeKey: typeof raw?.viaEdgeKey === 'string' ? raw.viaEdgeKey : undefined,
		};
		resolved.push(shot);
		prev = shot;
	}
	return resolved;
}

// ── The deterministic auteur (Tier 2) ────────────────────────────────────────
// The graph-metric → shot-grammar rule table. This SAME table is handed to the
// LLM as its system prompt (see directorSystemPrompt), so Tier-1 output is
// directly comparable to and back-fillable against this baseline.

function actFor(progress: number): Act {
	return progress < 0.34 ? 'I' : progress < 0.72 ? 'II' : 'III';
}

/**
 * Produce a cinematic DirectorPlan from pure graph signals — no LLM. This alone
 * ships the hero film: every shot is grounded and justified by a real metric.
 */
export function planShotsDeterministic(path: CinemaPath, signals: GraphSignals): DirectorPlan {
	const n = path.beats.length;
	const shots: Shot[] = path.beats.map((beat, i) => {
		const progress = n > 1 ? i / (n - 1) : 0;
		const act = actFor(progress);
		const sig = signals.nodes.get(beat.nodeId);
		const isPeak = beat.nodeId === signals.peakBetweennessId;
		const isFinale = i === n - 1;
		const isOrigin = i === 0;

		// Default shot for a plain connection beat.
		let shot: Shot = {
			nodeId: beat.nodeId,
			move: 'push_in',
			angle: 'eye',
			cut: 'fly',
			stormMode: 'connection',
			tone: 'curious',
			scoreCue: 'motif',
			act,
			intensity: 0.6,
			tension: 0.3,
			why: 'a connected memory',
		};

		if (isOrigin) {
			shot = { ...shot, move: 'push_in', tone: 'curious', tension: 0.25, stormMode: 'anchor', why: 'opening on the focal memory' };
		}
		// High-betweenness keystone → reverent low-angle slow orbit.
		if (isPeak || (sig && sig.betweenness > 0.6)) {
			shot = { ...shot, move: 'orbit', angle: 'low', stormMode: 'anchor', intensity: 0.75, tension: 0.45, tone: 'awe', why: 'low-angle orbit — the most load-bearing memory in the graph' };
		}
		// Contradiction → Dutch push-in, hard cut, crimson chaos, minor drop.
		if (beat.kind === 'contradiction') {
			shot = { ...shot, move: 'push_in', angle: 'eye', dutch: 0.28, cut: 'hard_cut', stormMode: 'contradiction', intensity: 1, tension: 0.95, tone: 'tense', scoreCue: 'minor_drop', viaEdgeKey: beat.viaEdge ? `${beat.viaEdge.source}->${beat.viaEdge.target}` : undefined, why: 'two memories in tension — a Dutch two-shot collision' };
		}
		// Surprise edge → gold/violet convergence, rising awe.
		if (beat.kind === 'surprise') {
			shot = { ...shot, move: 'orbit', stormMode: 'surprise', intensity: 0.85, tension: 0.6, tone: 'awe', scoreCue: 'motif', why: 'a surprising, distant-but-plausible connection' };
		}
		// Fading memory → drifting high angle.
		if (sig && (sig.retention < 0.35 || sig.suppression > 0.5)) {
			shot = { ...shot, angle: 'high', move: 'pull_back', tone: 'neutral', intensity: 0.4, why: 'a fading memory — high-angle drift' };
		}
		// Recent → the "now" beat.
		if (beat.kind === 'recent') {
			shot = { ...shot, move: 'push_in', tone: 'resolved', tension: 0.4, why: 'where the memory is now' };
		}
		// Finale → crane pull-back, major resolve.
		if (isFinale) {
			shot = { ...shot, move: 'crane', cut: 'fly', stormMode: 'anchor', tone: 'awe', tension: 0.5, scoreCue: 'major_resolve', why: 'crane pull-back over the whole cluster — resolution' };
		}
		return shot;
	});

	const arc: EmotionalArc = path.beats.some((b) => b.kind === 'contradiction') ? 'man_in_hole' : 'rags_to_riches';
	const originLabel = path.beats[0]?.node.label ?? 'a memory';
	const logline = `A short film about ${originLabel} — ${n} shots through the graph${arc === 'man_in_hole' ? ', through a contradiction and out the other side' : ''}.`;

	return { source: 'deterministic', logline, arc, shots };
}

/** The rule table as an LLM system prompt — keeps Tier-1 output comparable to
 * the Tier-2 baseline (and thus back-fillable by resolveShots). */
export function directorSystemPrompt(): string {
	return [
		'You are a film director shooting a short documentary about an AI\'s own memory graph.',
		'Output a DirectorPlan: a logline, an emotional arc, and one shot per beat.',
		'Each shot MUST cite a real nodeId and a real "why" referencing a graph metric.',
		'Grammar → meaning:',
		'- high betweenness (load-bearing memory) → low-angle slow orbit, reverent',
		'- contradiction edge → Dutch angle + push_in + hard_cut + crimson storm + minor_drop score',
		'- surprising distant link → gold/violet orbit→stream convergence + awe',
		'- merge/supersede → match_cut at identical standoff+angle (same idea)',
		'- low retention / high suppression → high-angle drift (fading)',
		'- finale → crane pull_back + major_resolve',
		'Build a real emotional arc across acts I→II→III. Only specify axes that change.',
	].join('\n');
}
