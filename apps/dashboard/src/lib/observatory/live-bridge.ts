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
import type { GraphResponse, VestigeEvent } from '$types';
import { LIVE_KIND, PARAM_IDX, PATH_KIND, type ObservatoryGraph, type ObservatoryEdge } from './types';
import { liveRetrievability, retrievabilityAt, MS_PER_DAY } from './fsrs';
import { FirewallRenderer } from './firewall-renderer';
import { buildLiveFirewallPlan, emptyFirewallPlan } from './firewall-plan';
import { buildRecallPath } from './path-builder';

/** A decoded live event, normalized off the wire's { type, data } envelope. */
interface DecodedEvent {
	kind: number; // LIVE_KIND.*
	startFrame: number; // monotonic sim frame it fired
	/** Primary target node id (firewall/recall focus), '' if none. */
	targetId: string;
	/** Neighbor / path node ids the event references. */
	relatedIds: string[];
	/** Exact ordered path named by a persisted receipt. */
	exactPath?: string[];
	/** Contradiction / connection pairs [a,b][] the event carries. */
	pairs: [string, string][];
	/** Free-form scalar (estimated_cascade, weight, memory_count …). */
	scalar: number;
}

/** How long (sim frames @60fps) each live event's envelope plays before idle. */
const LIVE_DURATION: Record<number, number> = {
	[LIVE_KIND.firewall]: 620, // full quarantine choreography (matches demo)
	// Dream storm: a real dream on the local brain finishes in ~150ms (far
	// faster than a frame), but the CONSOLIDATION it performed — real edges
	// appended, clusters re-settling — is a genuine physical event that takes
	// seconds to play out. So the storm holds a minimum visible window (~6s)
	// while the newly-appended springs pull the field into its new shape, then
	// settles. Honest: the edges + physics are real; only the tempo is stretched
	// to human-perceptible (like the forgetting-horizon scrubber).
	[LIVE_KIND.dreamStorm]: 360,
	[LIVE_KIND.causalRecall]: 260, // wavefront sweep + afterglow
	[LIVE_KIND.birth]: 180
};

export interface LiveBridgeDeps {
	engine: ObservatoryEngine;
	renderer: NodeRenderer;
	graph: ObservatoryGraph;
	/** The raw graph response the field was uploaded from — the causal recall
	 *  pathfinder (Phase 4) needs the full nodes+edges, not just the graph. */
	response: GraphResponse;
	/** Layout seed (matches NodeRenderer.upload) — the firewall shock delays
	 *  come from the REAL layout, so this must be the same seed the field used. */
	seed: string;
	/**
	 * Forward-projection scrubber (Phase 1). Days added to every node's real
	 * FSRS elapsed so decay is legible in one viewing session. A getter so the
	 * bridge always reads the live control value without re-subscribing.
	 */
	projectionDays?: () => number;
	/**
	 * FOSSIL LIGHT chrono scrub — SIGNED days offset from NOW (negative = the
	 * past). When non-zero, per-node retention is re-evaluated at the scrubbed
	 * instant via `retrievabilityAt` (same closed form, signed time, existence
	 * mask before each memory's createdAt). 0 = live now. A getter so the
	 * bridge always reads the live control without re-subscribing.
	 */
	chronoOffsetDays?: () => number;
	/** Dev/verification: mirror what the bridge applied, for assertions. */
	onApply?: (info: { simFrame: number; activeKind: number; eventsSeen: number }) => void;
	/** Fired when a live firewall arms — the host can surface a verdict card. */
	onFirewall?: (info: { intruderLabel: string; startFrame: number }) => void;
}

export class LiveBridge {
	private engine: ObservatoryEngine;
	private renderer: NodeRenderer;
	private graph: ObservatoryGraph;
	private response: GraphResponse;
	private seed: string;
	private projectionDays: () => number;
	private chronoOffsetDays: () => number;
	private onApply?: LiveBridgeDeps['onApply'];
	private onFirewall?: LiveBridgeDeps['onFirewall'];

	/** Live firewall pass — constructed lazily on the first firewall event so a
	 *  field that never sees a suppression pays nothing. */
	private firewall: FirewallRenderer | null = null;

	/** Live edge accumulator (Phase 3 dream storm). Starts as the uploaded
	 *  graph edges; each real ConnectionDiscovered appends one, and setEdges
	 *  regrows the GPU buffer so the new spring physically pulls its endpoints
	 *  together. Deduped by (min,max) index so a re-discovered edge is a no-op. */
	private liveEdges: ObservatoryEdge[] = [];
	private liveEdgeKeys = new Set<string>();
	/** Set when new edges arrived this frame; drain() flushes ONE setEdges per
	 *  frame so a 50-connection dream burst is one buffer regrow, not fifty. */
	private edgesDirty = false;

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
		this.response = deps.response;
		this.seed = deps.seed;
		this.projectionDays = deps.projectionDays ?? (() => 0);
		this.chronoOffsetDays = deps.chronoOffsetDays ?? (() => 0);
		this.onApply = deps.onApply;
		this.onFirewall = deps.onFirewall;
		this.indexById = deps.graph.indexById;

		const n = deps.graph.nodes.length;
		this.retention = new Float32Array(n);
		for (let i = 0; i < n; i++) {
			const node = deps.graph.nodes[i];
			this.retention[i] = node.retention;
			if (node.stability !== undefined && node.lastAccessed) this.hasLiveDecay = true;
		}

		// Seed the live edge accumulator with the uploaded graph edges so dream
		// connections APPEND to the real field, never replace it.
		this.liveEdges = deps.graph.edges.slice();
		for (const e of this.liveEdges) this.liveEdgeKeys.add(edgeKey(e.sourceIndex, e.targetIndex));

		// Only react to events that arrive AFTER the bridge exists — skip the
		// store's backlog. Anchor to the BACKEND clock (the newest event already
		// in the feed), NOT the browser wall clock: on a sandboxed backend the
		// two can skew by tens of seconds, and a browser-time floor would then
		// skip every backend event forever. seedWatermark() sets this from the
		// store the moment the bridge is wired.
		this.lastAppliedMs = 0;

		// Seed the live lanes to a calm resting state.
		const p = this.engine.params;
		p[PARAM_IDX.liveKind] = LIVE_KIND.none;
		p[PARAM_IDX.liveFrame] = 0;
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

	/**
	 * Wire timestamp (ms) of the newest event already applied — a monotonic
	 * watermark. Robust to the 200-event ring evicting our anchor during a
	 * dream's 1000+-event burst: apply every event with a strictly newer
	 * timestamp, oldest→newest, then advance the watermark. Idempotent (re-
	 * ingesting the same store is a no-op) and survives eviction (identity
	 * anchors don't). A same-ms tie can drop at most one event, whose visual
	 * impact is nil (one connection among a thousand).
	 */
	private lastAppliedMs = 0;
	private seeded = false;

	/**
	 * Anchor the watermark to the backend clock: set it to the newest event
	 * timestamp currently in the feed, so the bridge ignores the pre-mount
	 * backlog but reacts to everything after. Immune to browser↔backend clock
	 * skew. Call once, right after wiring, before the first ingest.
	 */
	seedWatermark(events: VestigeEvent[]): void {
		let maxMs = 0;
		for (const ev of events) {
			const ts = evTimestampMs(ev);
			if (ts > maxMs) maxMs = ts;
		}
		this.lastAppliedMs = maxMs;
		this.seeded = true;
	}

	/**
	 * True while a REAL live event is playing its envelope. The receipt-replay
	 * driver reads this to yield: a genuine recall/firewall/dream from the agent
	 * always preempts the ambient replay of past receipts.
	 */
	get hasActiveEvent(): boolean {
		return this.active !== null;
	}

	/**
	 * COLD-OPEN AHA — replay one of the user's REAL past recalls (from a stored
	 * receipt's activation_path) as a causalRecall, driving the exact same GCaMP
	 * wavefront a live retrieval fires. This is NOT synthetic choreography: the
	 * target + path are real memory ids the user's agent actually retrieved. A
	 * client opening their dashboard cold watches their own memory being
	 * recalled, in calcium. No-op if a real live event owns the field, if the
	 * target isn't in the current field, or if replay is globally off.
	 */
	replayRecall(targetId: string, pathIds: string[], simFrame: number): boolean {
		if (this.active !== null) return false;
		const targetIndex = this.indexById.get(targetId);
		if (targetIndex === undefined) return false;
		// Do NOT ambient-replay a recall on a memory that does not exist at the
		// current instant: during a deep chrono rewind the target may be unborn
		// (live retention 0 → the render existence mask hides it), so the recall
		// would fire into the void. Fire only on memories currently on camera.
		if ((this.retention[targetIndex] ?? 0) < 0.0005) return false;
		const related = pathIds.filter((id) => id !== targetId && this.indexById.has(id));
		this.arm({
			kind: LIVE_KIND.causalRecall,
			startFrame: simFrame,
			targetId,
			relatedIds: related,
			pairs: [],
			scalar: related.length
		});
		return true;
	}

	ingest(events: VestigeEvent[]): void {
		if (events.length === 0) return;
		// If we were never explicitly seeded, seed from this first batch (anchor
		// to the backend clock) and apply nothing — avoids replaying the backlog.
		if (!this.seeded) {
			this.seedWatermark(events);
			return;
		}
		let maxMs = this.lastAppliedMs;
		// events[] is newest-first → walk oldest→newest so cause precedes effect.
		for (let i = events.length - 1; i >= 0; i--) {
			const ev = events[i];
			const ts = evTimestampMs(ev);
			if (ts > this.lastAppliedMs) {
				this.decodeAndArm(ev, this.engine.totalFrames);
				if (ts > maxMs) maxMs = ts;
			}
		}
		this.lastAppliedMs = maxMs;
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
				// Preserve the raw receipt route. Filtering it here would turn
				// [candidate, missing, failure] into a fabricated direct edge.
				const path = strArr(data.path_ids ?? data.causal_path);
				const target = str(data.failure_id ?? data.target_id ?? data.effect_id) || path.at(-1) || path[0];
				if (target && this.indexById.has(target)) {
					this.arm({
						kind: LIVE_KIND.causalRecall,
						startFrame: simFrame,
						targetId: target,
						relatedIds: path.filter((x) => x !== target),
						exactPath: path,
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
				// A real dream completes in ~150ms and floods the 200-event WS ring
				// with ConnectionDiscovered, often EVICTING the DreamStarted we'd
				// otherwise arm on. DreamCompleted is the newest event so it always
				// survives — start (or refresh) the storm HERE too, driven by the
				// real connections_found count, so the consolidation always plays.
				// The appended edges (already applied above) need seconds to pull
				// the clusters into their new shape; the finite window plays it out.
				const found = num(data.connections_found);
				if (this.active && this.active.kind === LIVE_KIND.dreamStorm) {
					this.active.scalar = Math.max(this.active.scalar, found);
				} else {
					this.arm({
						kind: LIVE_KIND.dreamStorm,
						startFrame: simFrame,
						targetId: '',
						relatedIds: [],
						pairs: [],
						scalar: found
					});
				}
				break;
			}
			// Dream storm: append the REAL discovered edge so its spring
			// physically pulls the two memories together (clusters merge is the
			// emergent settle — no new physics, the force sim already runs).
			case 'ConnectionDiscovered': {
				const s = this.indexById.get(str(data.source_id));
				const t = this.indexById.get(str(data.target_id));
				if (s === undefined || t === undefined || s === t) break;
				const key = edgeKey(s, t);
				if (this.liveEdgeKeys.has(key)) break;
				this.liveEdgeKeys.add(key);
				this.liveEdges.push({
					sourceIndex: s,
					targetIndex: t,
					weight: num(data.weight) || 0.5,
					type: str(data.connection_type) || 'semantic'
				});
				this.edgesDirty = true; // coalesced flush in drain()
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

		// Firewall: build a live plan for the REAL intruder + its real neighbors
		// and (re)arm the quarantine pass. Lazily construct the renderer on first
		// use so a field that never sees a suppression pays nothing. Pass order is
		// safe: the bridge (and thus this renderer) is created AFTER the
		// NodeRenderer, so firewall_choreo encodes after recall_sim.
		if (ev.kind === LIVE_KIND.firewall) {
			const intruderIndex = this.indexById.get(ev.targetId);
			if (intruderIndex === undefined) return;
			const plan = buildLiveFirewallPlan(this.graph, this.seed, intruderIndex);
			if (!plan.viable) return;
			if (!this.firewall) {
				this.firewall = new FirewallRenderer({
					engine: this.engine,
					nodeRenderer: this.renderer,
					plan: emptyFirewallPlan(this.graph.nodes.length)
				});
			}
			this.firewall.rearm(plan);
			this.onFirewall?.({ intruderLabel: plan.verdict.intruderLabel, startFrame: ev.startFrame });
		}

		// Causal recall (Phase 4): a real recall (DeepReferenceCompleted /
		// BackfillFired) lights the backward CAUSE chain from the recalled memory.
		// Rebuild the PathStep buffer centered on the real target, preferring real
		// causal edges so the wavefront traces true causation, not co-occurrence.
		// The proven recall-wavefront machinery (render-path.wgsl, simulate.wgsl)
		// renders it — kind-1 (backward) hops burn into the magenta rim.
		if (ev.kind === LIVE_KIND.causalRecall && this.indexById.has(ev.targetId)) {
			if (ev.exactPath && ev.exactPath.length > 1) {
				// A persisted Backfill route is atomic. Dropping an absent waypoint
				// would forge a direct edge that the receipt never recorded.
				const ids = ev.exactPath;
				if (ids.some((id) => !this.indexById.has(id))) return;
				const data = new Uint32Array(Math.max(1, ids.length - 1) * 4);
				const steps = [];
				for (let i = 0; i < ids.length - 1; i++) {
					const sourceIndex = this.indexById.get(ids[i])!;
					const targetIndex = this.indexById.get(ids[i + 1])!;
					const beatFrame = ev.startFrame + 24 + i * 42;
					data[i * 4] = sourceIndex;
					data[i * 4 + 1] = targetIndex;
					data[i * 4 + 2] = beatFrame;
					data[i * 4 + 3] = PATH_KIND.backwardCause;
					steps.push({ sourceIndex, targetIndex, beatFrame, kind: PATH_KIND.backwardCause, beatKind: 'receipt-path', nodeId: ids[i + 1], label: 'receipt-backed candidate path' });
				}
				this.renderer.setPathSteps(data, steps);
				return;
			}
			const built = buildRecallPath(this.response, this.graph, 8, {
				preferCausal: true,
				centerId: ev.targetId
			});
			if (built.steps.length > 0) {
				// Re-anchor the beats so the wavefront fires FROM this event
				// (live_frame), not the loop clock. The render-path/simulate
				// shaders read absolute beatFrames; the demo path used 60,120,…
				// which the live loop frame also sweeps, so the existing timing
				// works — the wavefront plays over the causalRecall window.
				this.renderer.setPathSteps(built.data, built.steps);
			}
		}
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

		// Flush any edges appended this frame in ONE buffer regrow (a 50-edge
		// dream burst = one setEdges, not fifty pipeline rebuilds).
		if (this.edgesDirty) {
			this.renderer.setEdges(this.liveEdges);
			this.edgesDirty = false;
		}

		// --- Phase 1: live FSRS decay (recompute retrievability on the true
		// curve). Throttled to every 6 frames (10Hz) — decay drifts far slower
		// than a frame, and the scrubber jumps are applied immediately below. ---
		const proj = this.projectionDays();
		const chrono = this.chronoOffsetDays();
		// The params lane stays forward-only (>=0): shaders read projection_days
		// for the horizon grammar and never expect signed time. The signed
		// chrono offset lives entirely in the CPU decay eval below.
		p[PARAM_IDX.projectionDays] = Math.max(0, proj);
		// Recompute when: throttle tick (decay drifts), either time control
		// moved, or we just returned to NOW (one final pass to restore live
		// values). Chrono scrubbing works even without live FSRS state — the
		// createdAt existence mask alone still unbirths memories.
		const decayActive = this.hasLiveDecay || chrono !== 0 || this.lastChrono !== 0;
		if (
			decayActive &&
			(simFrame - this.lastDecayFrame >= 6 || proj !== this.lastProj || chrono !== this.lastChrono)
		) {
			this.recomputeDecay(proj, chrono);
			this.lastDecayFrame = simFrame;
			this.lastProj = proj;
			this.lastChrono = chrono;
		}

		// --- Live event envelope: lanes 12..14. ---
		if (this.active) {
			const dur = LIVE_DURATION[this.active.kind] ?? 300;
			const elapsed = simFrame - this.active.startFrame;
			if (elapsed > dur + 140) {
				// Envelope finished (+ fade tail) → back to calm.
				this.active = null;
				p[PARAM_IDX.liveKind] = LIVE_KIND.none;
				p[PARAM_IDX.liveEnergy] = 0;
			} else {
				p[PARAM_IDX.liveKind] = this.active.kind;
				// Event-relative frame: the live shader branches replay the
				// choreography once from here, never riding the wrapped loop.
				p[PARAM_IDX.liveFrame] = Math.max(0, elapsed);
				p[PARAM_IDX.liveEnergy] = this.energyEnvelope(this.active, elapsed, false);
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

	/** Dev/verification snapshot — current live state (not used in production). */
	debugState(): {
		activeKind: number;
		liveEnergy: number;
		liveFrame: number;
		edgeCount: number;
		eventsSeen: number;
	} {
		const p = this.engine.params;
		return {
			activeKind: p[PARAM_IDX.liveKind],
			liveEnergy: p[PARAM_IDX.liveEnergy],
			liveFrame: p[PARAM_IDX.liveFrame],
			edgeCount: this.liveEdges.length,
			eventsSeen: this.eventsSeen
		};
	}

	private lastProj = -1;
	private lastChrono = 0;

	/** 0..1+ agitation envelope for the active event at `elapsed` frames. */
	private energyEnvelope(ev: DecodedEvent, elapsed: number, _openEnded: boolean): number {
		if (elapsed < 0) return 0;
		const dur = LIVE_DURATION[ev.kind] ?? 300;
		if (ev.kind === LIVE_KIND.dreamStorm) {
			// Ramp in (45f) → sustained storm plateau (scaled by how many real
			// connections the dream found) → ease out over the last ~90f as the
			// clusters settle into their new shape.
			const ramp = Math.min(1, elapsed / 45);
			const ease = 1 - Math.max(0, (elapsed - (dur - 90)) / 90);
			const intensity = Math.min(1.4, 0.7 + ev.scalar * 0.02);
			return Math.max(0, ramp * Math.min(1, ease) * intensity);
		}
		const attack = Math.min(1, elapsed / 24);
		const release = 1 - Math.max(0, (elapsed - dur) / 140);
		return Math.max(0, attack * Math.min(1, release));
	}

	/**
	 * Recompute per-node live retrievability and push to the GPU (Phase 1 +
	 * FOSSIL LIGHT). At NOW (chrono 0) this is the original forward path; with
	 * a chrono offset the whole field is re-evaluated at the scrubbed instant
	 * on the same closed form — retention genuinely relights, and memories not
	 * yet born read exactly 0 (the render mask pops them out of existence).
	 */
	private recomputeDecay(projectionDays: number, chronoOffsetDays = 0): void {
		const nowMs = this.engine.wallNowMs;
		const nodes = this.graph.nodes;
		if (chronoOffsetDays !== 0) {
			const evalMs = nowMs + (chronoOffsetDays + Math.max(0, projectionDays)) * MS_PER_DAY;
			for (let i = 0; i < nodes.length; i++) {
				const n = nodes[i];
				this.retention[i] =
					n.stability !== undefined || n.createdAt
						? retrievabilityAt(n.stability, n.lastAccessed, n.createdAt, evalMs)
						: Math.max(0.001, n.retention);
			}
		} else {
			for (let i = 0; i < nodes.length; i++) {
				const n = nodes[i];
				this.retention[i] =
					n.stability !== undefined && n.lastAccessed
						? liveRetrievability(n.stability, n.lastAccessed, nowMs, projectionDays)
						: Math.max(0.001, n.retention);
			}
		}
		this.renderer.uploadLiveRetention(this.retention);
	}

	/** Force a decay recompute now (e.g. a time control moved). */
	refreshDecay(): void {
		const chrono = this.chronoOffsetDays();
		if (this.hasLiveDecay || chrono !== 0 || this.lastChrono !== 0) {
			this.recomputeDecay(this.projectionDays(), chrono);
			this.lastChrono = chrono;
		}
	}
}

// --- tiny decoders (no allocation beyond the returned value) ---

/** Undirected edge key (min,max) so a re-discovered edge dedupes either way. */
function edgeKey(a: number, b: number): string {
	return a < b ? `${a}-${b}` : `${b}-${a}`;
}

/**
 * Wire timestamp of an event in ms. Every VestigeEvent's data carries an RFC3339
 * `timestamp`. Falls back to a monotonic-ish 0 so a malformed event is treated
 * as old (skipped) rather than replayed forever.
 */
function evTimestampMs(ev: VestigeEvent): number {
	const raw = (ev.data as { timestamp?: unknown } | undefined)?.timestamp;
	if (typeof raw !== 'string') return 0;
	const t = Date.parse(raw);
	return Number.isFinite(t) ? t : 0;
}

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
