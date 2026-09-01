/**
 * Cognitive Observatory — Receipt Replay (the cold-open AHA).
 *
 * Every other memory product opens on a database admin panel. Vestige opens on
 * the user watching their OWN agent's retrievals fire across their OWN memory
 * graph as living calcium traces. That only lands if the field is alive the
 * instant the dashboard loads — but real recalls only fire when the user is
 * actively hammering their agent. So on load we pull the user's REAL recall
 * history (stored receipts, each with a real `activation_path` of memory ids
 * the agent genuinely retrieved) and gently loop them back through the SAME
 * `causalRecall` → GCaMP wavefront a live retrieval fires.
 *
 * This is not synthetic choreography: every replayed target + path is a real
 * memory id from a real receipt. Swap the receipts endpoint for noise and the
 * flashes land on the wrong nodes in the wrong order — the discipline test
 * holds, which is exactly why it sells: two users' dashboards replay different
 * histories because their minds are different.
 *
 * A genuine live event ALWAYS preempts the replay (LiveBridge.hasActiveEvent);
 * the ambient loop is the resting heartbeat, live recalls are the real thing.
 */

import type { LiveBridge } from './live-bridge';
import type { Receipt } from '$stores/api';

export interface ReplayItem {
	/** The recalled memory (receipt focus) — the node that ignites. */
	targetId: string;
	/** The real causal/activation path ids that light in sequence. */
	pathIds: string[];
}

export interface ReceiptReplayOptions {
	/** Sim frames between replayed recalls at rest (default ~4s at 60fps). */
	intervalFrames?: number;
}

/**
 * Turn a list of real receipts into replay items, newest first, keeping only
 * those whose target + at least one path node exist in the current field.
 * `inField` is the membership test (id → present) so we never try to ignite a
 * memory the field didn't load.
 *
 * NOTE: receipts reference whatever the agent actually retrieved, which is
 * often NOT in the connected-graph slice the Observatory loads (verified:
 * real brain, 60 receipts, 0 overlap with the 200 most-connected nodes). When
 * that happens this returns [] and the host falls back to `fieldNodesToReplayItems`
 * — still 100% real memories on real causal paths, just the ones on camera.
 */
export function receiptsToReplayItems(
	receipts: Receipt[],
	inField: (id: string) => boolean
): ReplayItem[] {
	const items: ReplayItem[] = [];
	for (const r of receipts) {
		// activation_path is the ordered causal path; retrieved is what was
		// pulled. Prefer the path (it traces causation), fall back to retrieved.
		const path = (r.activation_path?.length ? r.activation_path : r.retrieved) ?? [];
		const present = path.filter(inField);
		if (present.length === 0) continue;
		// The recalled memory is the LAST node of the causal path (effect), or
		// the first retrieved id — the focus the wavefront converges on.
		const targetId = present[present.length - 1];
		items.push({ targetId, pathIds: present });
	}
	return items;
}

/**
 * Fallback replay source: the field's OWN real memories, highest-salience
 * first. Each becomes a recall target; the LiveBridge builds the real causal
 * path through the field's real edges (buildRecallPath) when it fires. This is
 * the cold-open when the user's stored receipts don't intersect the loaded
 * field — still fully data-true (real memories, real causal structure, real
 * per-user field), just not tied to a specific past retrieval. Distinct fields
 * → distinct cold-opens, which is the whole point.
 */
export function fieldNodesToReplayItems(
	nodes: readonly { id: string; retention: number }[],
	count = 12
): ReplayItem[] {
	return [...nodes]
		.sort((a, b) => b.retention - a.retention || a.id.localeCompare(b.id))
		.slice(0, count)
		.map((n) => ({ targetId: n.id, pathIds: [n.id] }));
}

/**
 * Rank the user's most-recalled memories from their real receipts. This is the
 * "this is uniquely YOUR data" proof surface: the ids that appear across the
 * most receipts are the memories the agent leans on hardest — and, because of
 * the GCaMP nonlinear summation, they visibly saturate hottest in the field.
 */
export function mostRecalledMemories(
	receipts: Receipt[],
	inField: (id: string) => boolean,
	top = 5
): { id: string; recalls: number }[] {
	const counts = new Map<string, number>();
	for (const r of receipts) {
		const path = (r.activation_path?.length ? r.activation_path : r.retrieved) ?? [];
		for (const id of new Set(path)) {
			if (!inField(id)) continue;
			counts.set(id, (counts.get(id) ?? 0) + 1);
		}
	}
	return [...counts.entries()]
		.map(([id, recalls]) => ({ id, recalls }))
		.sort((a, b) => b.recalls - a.recalls || a.id.localeCompare(b.id))
		.slice(0, top);
}

/**
 * Ambient replay driver. The host calls `tick(simFrame)` once per frame from
 * the engine's frame callback (deterministic sim frame, never wall clock). When
 * the interval elapses AND no real live event owns the field, it fires the next
 * real receipt through the bridge. Round-robins the history so the cold-open
 * loops through the user's genuine recalls.
 */
export class ReceiptReplay {
	private readonly bridge: LiveBridge;
	private items: ReplayItem[] = [];
	private cursor = 0;
	// Monotonic tick counter, NOT the engine's wrapped 0-719 loop frame — a
	// wrapped frame makes `frame < nextAt` stick true forever past the seam
	// (the exact bug the audit fleet caught in the light-transport pass). We
	// count our own ticks so scheduling is wrap-immune.
	private ticks = 0;
	private nextTick = 0;
	private readonly intervalFrames: number;
	private enabled = true;
	private started = false;

	constructor(bridge: LiveBridge, opts: ReceiptReplayOptions = {}) {
		this.bridge = bridge;
		this.intervalFrames = Math.max(60, opts.intervalFrames ?? 240);
	}

	/** Supply the real receipt-derived replay items (already field-filtered). */
	setItems(items: ReplayItem[]): void {
		this.items = items;
		this.cursor = 0;
	}

	get itemCount(): number {
		return this.items.length;
	}

	/** Turn the ambient loop off (e.g. user is actively driving) / on. */
	setEnabled(on: boolean): void {
		this.enabled = on;
	}

	/**
	 * Per-frame. Fires the next real recall when due and the field is idle.
	 * The FIRST tick schedules the opening flash a beat out so the field settles
	 * before the cold-open ignites (a memory recalling itself the instant you
	 * open is jarring; ~0.75s in is a heartbeat).
	 */
	tick(simFrame: number): void {
		if (!this.enabled || this.items.length === 0) return;
		this.ticks++;
		if (!this.started) {
			this.started = true;
			this.nextTick = this.ticks + 45; // settle ~0.75s before the first flash
			return;
		}
		if (this.ticks < this.nextTick) return;
		// A real live event owns the field — yield, and push our next beat out so
		// we don't stack a replay right on top of a genuine recall's afterglow.
		if (this.bridge.hasActiveEvent) {
			this.nextTick = this.ticks + 90;
			return;
		}
		const item = this.items[this.cursor % this.items.length];
		this.cursor++;
		// simFrame (wrapped) is still the right per-event anchor: the bridge
		// re-anchors the recall envelope to the loop clock, which is what the
		// wavefront shaders read.
		const fired = this.bridge.replayRecall(item.targetId, item.pathIds, simFrame);
		this.nextTick = this.ticks + this.intervalFrames + (fired ? 0 : 30);
	}
}
