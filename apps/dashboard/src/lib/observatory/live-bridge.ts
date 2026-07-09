/**
 * Live event bridge — the field's nervous system (Phase 0, v2.3).
 *
 * The Observatory field was inert: it ran the deterministic DemoClock and never
 * heard the backend. This bridge is the ONE integration that makes it live —
 * every hero (live FSRS decay, the contradiction firewall, the dream storm, the
 * causal recall wavefront) rides it.
 *
 * Contract:
 *   - ObservatoryStage pushes new VestigeEvents (from $eventFeed) via `ingest`.
 *   - The engine calls `drain(simFrame)` once per frame (its preFrameHook),
 *     AFTER p[0..11] are set, BEFORE the params buffer is written to the GPU.
 *   - `drain` applies queued events to engine params (lanes 12..15) and node
 *     buffers for the current frame. Allocation-free in the hot path: a fixed
 *     ring buffer of decoded events + preallocated typed arrays, no per-frame
 *     GC pressure.
 *
 * Determinism note: the deterministic demo loop is untouched. This bridge only
 * writes the LIVE lanes (0.0 at rest) and the live-retention buffer, so a field
 * with no backend events is pixel-identical to the pre-bridge build.
 */

import type { ObservatoryEngine } from './engine';
import type { NodeRenderer } from './node-renderer';
import type { VestigeEvent } from '$types';
import { LIVE_KIND, PARAM_IDX, type ObservatoryGraph } from './types';
import { liveRetrievability } from './fsrs';

/** A decoded live event, normalized off the wire's { type, data } envelope. */
interface DecodedEvent {
	kind: number; // LIVE_KIND.*
	startFrame: number; // monotonic sim frame it fired
	/** Primary target node id (firewall/recall focus), '' if none. */
	targetId: string;
	/** Neighbor / path node ids the event references. */
	relatedIds: string[];
	/** Contradiction / connection pairs [a,b][] the event carries. */
	pairs: [string, string][];
	/** Free-form scalar (estimated_cascade, weight, memory_count …). */
	scalar: number;
}

/** How long (sim frames @60fps) each live event's envelope plays before idle. */
const LIVE_DURATION: Record<number, number> = {
	[LIVE_KIND.firewall]: 620, // full quarantine choreography (matches demo)
	[LIVE_KIND.dreamStorm]: 0, // open-ended: held until DreamCompleted arrives
	[LIVE_KIND.causalRecall]: 260, // wavefront sweep + afterglow
	[LIVE_KIND.birth]: 180
};

export interface LiveBridgeDeps {
	engine: ObservatoryEngine;
	renderer: NodeRenderer;
	graph: ObservatoryGraph;
	/**
	 * Forward-projection scrubber (Phase 1). Days added to every node's real
	 * FSRS elapsed so decay is legible in one viewing session. A getter so the
	 * bridge always reads the live control value without re-subscribing.
	 */
	projectionDays?: () => number;
	/** Dev/verification: mirror what the bridge applied, for assertions. */
	onApply?: (info: { simFrame: number; activeKind: number; eventsSeen: number }) => void;
}

export class LiveBridge {
	private engine: ObservatoryEngine;
	private renderer: NodeRenderer;
	private graph: ObservatoryGraph;
	private projectionDays: () => number;
	private onApply?: LiveBridgeDeps['onApply'];

	/** id → stable buffer index (from the ObservatoryGraph). */
	private indexById: Map<string, number>;

	/** The single active live event (newest wins; heroes are momentary). */
	private active: DecodedEvent | null = null;
	/** True while a dream storm is open (between Started and Completed). */
	private dreamOpen = false;

	/** Preallocated per-node live-retention mirror (Phase 1). */
	private retention: Float32Array;
	/** Whether any node carries real FSRS state (else keep static retention). */
	private hasLiveDecay = false;
	/** Monotonic counter of events applied — for the dev assertion. */
	private eventsSeen = 0;
	/** Last sim frame we recomputed decay (throttle: decay drifts slowly). */
	private lastDecayFrame = -1000;

	constructor(deps: LiveBridgeDeps) {
		this.engine = deps.engine;
		this.renderer = deps.renderer;
		this.graph = deps.graph;
		this.projectionDays = deps.projectionDays ?? (() => 0);
		this.onApply = deps.onApply;
		this.indexById = deps.graph.indexById;

		const n = deps.graph.nodes.length;
		this.retention = new Float32Array(n);
		for (let i = 0; i < n; i++) {
			const node = deps.graph.nodes[i];
			this.retention[i] = node.retention;
			if (node.stability !== undefined && node.lastAccessed) this.hasLiveDecay = true;
		}

		// Seed the live lanes to a calm resting state.
		const p = this.engine.params;
		p[PARAM_IDX.liveKind] = LIVE_KIND.none;
		p[PARAM_IDX.liveStartFrame] = 0;
		p[PARAM_IDX.liveEnergy] = 0;
		p[PARAM_IDX.projectionDays] = 0;
	}

	/** Whether real per-memory FSRS decay data is present (Phase 1 gate). */
	get liveDecayAvailable(): boolean {
		return this.hasLiveDecay;
	}

	// -------------------------------------------------------------------------
	// Ingestion — called from the ObservatoryStage $eventFeed subscription.
	// $eventFeed is newest-first; we apply oldest-first so cause precedes
	// effect (mirrors Graph3D). `events` is the full store slice; we only act
	// on ones newer than the last we saw (tracked by object identity + a small
	// dedupe window on timestamp is unnecessary — the store slices FIFO).
	// -------------------------------------------------------------------------

	private lastSeenTop: VestigeEvent | null = null;

	ingest(events: VestigeEvent[]): void {
		if (events.length === 0) return;
		// Find the boundary: everything above lastSeenTop is new.
		let newCount = events.length;
		if (this.lastSeenTop) {
			const idx = events.indexOf(this.lastSeenTop);
			if (idx >= 0) newCount = idx;
		}
		if (newCount === 0) return;
		// Apply oldest→newest (reverse, since events[] is newest-first).
		for (let i = newCount - 1; i >= 0; i--) {
			this.decodeAndArm(events[i], this.engine.totalFrames);
		}
		this.lastSeenTop = events[0];
	}

	/** Decode one wire event and, if it's a hero trigger, arm it. */
	private decodeAndArm(ev: VestigeEvent, simFrame: number): void {
		const data = (ev.data ?? {}) as Record<string, unknown>;
		switch (ev.type) {
			case 'MemorySuppressed': {
				const id = str(data.id);
				if (!id || !this.indexById.has(id)) return;
				this.arm({
					kind: LIVE_KIND.firewall,
					startFrame: simFrame,
					targetId: id,
					relatedIds: this.neighborsOf(id),
					pairs: [],
					scalar: num(data.estimated_cascade)
				});
				break;
			}
			case 'DeepReferenceCompleted': {
				const pairs = decodePairs(data.contradiction_pairs);
				// Contradiction pairs both present in the field → firewall.
				const livePairs = pairs.filter(
					([a, b]) => this.indexById.has(a) && this.indexById.has(b)
				);
				if (livePairs.length > 0) {
					const target = livePairs[0][0];
					this.arm({
						kind: LIVE_KIND.firewall,
						startFrame: simFrame,
						targetId: target,
						relatedIds: livePairs.flatMap((p) => p).filter((x) => x !== target),
						pairs: livePairs,
						scalar: livePairs.length
					});
					return;
				}
				// Otherwise a recall: light the backward causal path (Phase 4).
				const primary = str(data.primary_id);
				const supporting = strArr(data.supporting_ids).filter((x) => this.indexById.has(x));
				if (primary && this.indexById.has(primary)) {
					this.arm({
						kind: LIVE_KIND.causalRecall,
						startFrame: simFrame,
						targetId: primary,
						relatedIds: supporting,
						pairs: [],
						scalar: num(data.confidence)
					});
				}
				break;
			}
			case 'BackfillFired':
			case 'CausalReceipt': {
				// Phase 4 dedicated event: carries the backward causal path.
				const path = strArr(data.path_ids ?? data.causal_path).filter((x) =>
					this.indexById.has(x)
				);
				const target = str(data.target_id ?? data.effect_id) || path[0];
				if (target && this.indexById.has(target)) {
					this.arm({
						kind: LIVE_KIND.causalRecall,
						startFrame: simFrame,
						targetId: target,
						relatedIds: path.filter((x) => x !== target),
						pairs: [],
						scalar: path.length
					});
				}
				break;
			}
			case 'DreamStarted': {
				this.dreamOpen = true;
				this.arm({
					kind: LIVE_KIND.dreamStorm,
					startFrame: simFrame,
					targetId: '',
					relatedIds: [],
					pairs: [],
					scalar: num(data.memory_count)
				});
				break;
			}
			case 'DreamCompleted': {
				this.dreamOpen = false;
				// Let the storm settle: keep the current active event but mark it
				// finite from now (drain will fade it out over ~120 frames).
				if (this.active && this.active.kind === LIVE_KIND.dreamStorm) {
					this.active.startFrame = simFrame - 500; // jump near the tail
				}
				break;
			}
			// ConnectionDiscovered is consumed by the dream-storm edge appender
			// in ObservatoryStage directly (it needs the renderer's setEdges);
			// the bridge just keeps the storm energy high while they stream.
			case 'ConnectionDiscovered': {
				if (this.dreamOpen && this.active?.kind === LIVE_KIND.dreamStorm) {
					this.active.scalar += 1; // more connections → more agitation
				}
				break;
			}
			default:
				break;
		}
	}

	private arm(ev: DecodedEvent): void {
		this.active = ev;
		this.eventsSeen++;
	}

	/** Real graph neighbors of a node id (for the firewall quarantine ring). */
	private neighborsOf(id: string): string[] {
		const idx = this.indexById.get(id);
		if (idx === undefined) return [];
		const out: string[] = [];
		for (const e of this.graph.edges) {
			if (e.sourceIndex === idx) out.push(this.graph.nodes[e.targetIndex].id);
			else if (e.targetIndex === idx) out.push(this.graph.nodes[e.sourceIndex].id);
			if (out.length >= 12) break;
		}
		return out;
	}

	// -------------------------------------------------------------------------
	// Per-frame drain — the engine's preFrameHook. Allocation-free.
	// -------------------------------------------------------------------------

	drain(simFrame: number): void {
		const p = this.engine.params;

		// --- Phase 1: live FSRS decay (recompute retrievability on the true
		// curve). Throttled to every 6 frames (10Hz) — decay drifts far slower
		// than a frame, and the scrubber jumps are applied immediately below. ---
		const proj = this.projectionDays();
		p[PARAM_IDX.projectionDays] = proj;
		if (this.hasLiveDecay && (simFrame - this.lastDecayFrame >= 6 || proj !== this.lastProj)) {
			this.recomputeDecay(proj);
			this.lastDecayFrame = simFrame;
			this.lastProj = proj;
		}

		// --- Live event envelope: lanes 12..14. ---
		if (this.active) {
			const dur = LIVE_DURATION[this.active.kind] ?? 300;
			const elapsed = simFrame - this.active.startFrame;
			const openEnded = dur === 0 && this.dreamOpen;
			if (!openEnded && dur > 0 && elapsed > dur + 140) {
				// Envelope finished (+ fade tail) → back to calm.
				this.active = null;
				p[PARAM_IDX.liveKind] = LIVE_KIND.none;
				p[PARAM_IDX.liveEnergy] = 0;
			} else {
				p[PARAM_IDX.liveKind] = this.active.kind;
				p[PARAM_IDX.liveStartFrame] = this.active.startFrame;
				p[PARAM_IDX.liveEnergy] = this.energyEnvelope(this.active, elapsed, openEnded);
			}
		} else {
			p[PARAM_IDX.liveKind] = LIVE_KIND.none;
			p[PARAM_IDX.liveEnergy] = 0;
		}

		this.onApply?.({
			simFrame,
			activeKind: p[PARAM_IDX.liveKind],
			eventsSeen: this.eventsSeen
		});
	}

	private lastProj = -1;

	/** 0..1 agitation envelope for the active event at `elapsed` frames. */
	private energyEnvelope(ev: DecodedEvent, elapsed: number, openEnded: boolean): number {
		if (elapsed < 0) return 0;
		if (openEnded) {
			// Dream storm: ramp in over 45f, hold high, agitation scales with
			// how many connections have streamed in (scalar).
			const ramp = Math.min(1, elapsed / 45);
			return ramp * Math.min(1.4, 0.6 + ev.scalar * 0.04);
		}
		const dur = LIVE_DURATION[ev.kind] ?? 300;
		const attack = Math.min(1, elapsed / 24);
		const release = 1 - Math.max(0, (elapsed - dur) / 140);
		return Math.max(0, attack * Math.min(1, release));
	}

	/** Recompute per-node live retrievability and push to the GPU (Phase 1). */
	private recomputeDecay(projectionDays: number): void {
		const nowMs = this.engine.wallNowMs;
		const nodes = this.graph.nodes;
		for (let i = 0; i < nodes.length; i++) {
			const n = nodes[i];
			this.retention[i] =
				n.stability !== undefined && n.lastAccessed
					? liveRetrievability(n.stability, n.lastAccessed, nowMs, projectionDays)
					: n.retention;
		}
		this.renderer.uploadLiveRetention(this.retention);
	}

	/** Force a decay recompute now (e.g. the scrubber moved). */
	refreshDecay(): void {
		if (this.hasLiveDecay) this.recomputeDecay(this.projectionDays());
	}
}

// --- tiny decoders (no allocation beyond the returned value) ---

function str(v: unknown): string {
	return typeof v === 'string' ? v : '';
}
function num(v: unknown): number {
	return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}
function strArr(v: unknown): string[] {
	return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}
function decodePairs(v: unknown): [string, string][] {
	if (!Array.isArray(v)) return [];
	const out: [string, string][] = [];
	for (const pair of v) {
		if (Array.isArray(pair) && pair.length >= 2 && typeof pair[0] === 'string' && typeof pair[1] === 'string') {
			out.push([pair[0], pair[1]]);
		}
	}
	return out;
}
