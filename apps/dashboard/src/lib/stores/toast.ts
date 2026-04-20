// Pulse Toast — v2.2
// Subscribes to the WebSocket event feed and surfaces meaningful cognitive
// events as ephemeral toast notifications. This is the "brain coming alive"
// moment — when the dashboard is open you SEE Vestige thinking.
//
// Design:
//  - Filter the spammy events (Heartbeat, SearchPerformed, RetentionDecayed,
//    ActivationSpread, ImportanceScored, MemoryCreated). Those fire during
//    every ingest cycle and would flood the UI.
//  - Surface the narrative events: Dreams completing, Consolidation sweeping,
//    Memories being promoted/demoted/suppressed, Rac1 cascades, new bridges.
//  - Rate-limit ConnectionDiscovered — dreams fire many of these in seconds.
//  - Auto-dismiss after 5-6s. Max 4 on screen.

import { writable, get } from 'svelte/store';
import { eventFeed } from '$stores/websocket';
import type { VestigeEvent, VestigeEventType } from '$types';
import { EVENT_TYPE_COLORS } from '$types';

export interface Toast {
	id: number;
	type: VestigeEventType;
	title: string;
	body: string;
	color: string;
	dwellMs: number;
	createdAt: number;
}

const MAX_VISIBLE = 4;
const DEFAULT_DWELL_MS = 5500;
const CONNECTION_THROTTLE_MS = 1500;

function createToastStore() {
	const { subscribe, update } = writable<Toast[]>([]);
	let nextId = 1;
	let lastConnectionAt = 0;

	// Dwell-timer registry — exposed so the component can pause on hover
	// (biological respect: don't auto-dismiss a toast the user is actively
	// reading). Paused entries store remaining ms so resume can schedule a
	// new timer for the correct duration.
	const dwellTimers = new Map<number, ReturnType<typeof setTimeout>>();
	const dwellPaused = new Map<number, { remaining: number }>();
	const dwellStart = new Map<number, number>();

	function scheduleDismiss(id: number, ms: number) {
		dwellStart.set(id, Date.now());
		const handle = setTimeout(() => {
			dwellTimers.delete(id);
			dwellStart.delete(id);
			dismiss(id);
		}, ms);
		dwellTimers.set(id, handle);
	}

	function push(toast: Omit<Toast, 'id' | 'createdAt'>) {
		const id = nextId++;
		const createdAt = Date.now();
		const entry: Toast = { id, createdAt, ...toast };
		update(list => {
			const next = [entry, ...list];
			if (next.length > MAX_VISIBLE) {
				return next.slice(0, MAX_VISIBLE);
			}
			return next;
		});
		scheduleDismiss(id, toast.dwellMs);
	}

	function dismiss(id: number) {
		const handle = dwellTimers.get(id);
		if (handle) {
			clearTimeout(handle);
			dwellTimers.delete(id);
		}
		dwellPaused.delete(id);
		dwellStart.delete(id);
		update(list => list.filter(t => t.id !== id));
	}

	function pauseDwell(id: number, toastDwellMs: number) {
		const handle = dwellTimers.get(id);
		if (!handle) return;
		clearTimeout(handle);
		dwellTimers.delete(id);
		const startedAt = dwellStart.get(id) ?? Date.now();
		const elapsed = Date.now() - startedAt;
		const remaining = Math.max(200, toastDwellMs - elapsed);
		dwellPaused.set(id, { remaining });
	}

	function resumeDwell(id: number) {
		const paused = dwellPaused.get(id);
		if (!paused) return;
		dwellPaused.delete(id);
		scheduleDismiss(id, paused.remaining);
	}

	function clear() {
		for (const handle of dwellTimers.values()) clearTimeout(handle);
		dwellTimers.clear();
		dwellPaused.clear();
		dwellStart.clear();
		update(() => []);
	}

	function translate(event: VestigeEvent): Omit<Toast, 'id' | 'createdAt'> | null {
		const color = EVENT_TYPE_COLORS[event.type] ?? '#818CF8';
		const d = event.data as Record<string, unknown>;

		switch (event.type) {
			case 'DreamCompleted': {
				const replayed = Number(d.memories_replayed ?? 0);
				const found = Number(d.connections_found ?? 0);
				const insights = Number(d.insights_generated ?? 0);
				const ms = Number(d.duration_ms ?? 0);
				const parts: string[] = [];
				parts.push(`Replayed ${replayed} ${replayed === 1 ? 'memory' : 'memories'}`);
				if (found > 0) parts.push(`${found} new connection${found === 1 ? '' : 's'}`);
				if (insights > 0) parts.push(`${insights} insight${insights === 1 ? '' : 's'}`);
				return {
					type: event.type,
					title: 'Dream consolidated',
					body: `${parts.join(' · ')} in ${(ms / 1000).toFixed(1)}s`,
					color,
					dwellMs: 7000,
				};
			}

			case 'ConsolidationCompleted': {
				const nodes = Number(d.nodes_processed ?? 0);
				const decay = Number(d.decay_applied ?? 0);
				const embeds = Number(d.embeddings_generated ?? 0);
				const ms = Number(d.duration_ms ?? 0);
				const tail: string[] = [];
				if (decay > 0) tail.push(`${decay} decayed`);
				if (embeds > 0) tail.push(`${embeds} embedded`);
				return {
					type: event.type,
					title: 'Consolidation swept',
					body: `${nodes} node${nodes === 1 ? '' : 's'}${tail.length ? ' · ' + tail.join(' · ') : ''} in ${(ms / 1000).toFixed(1)}s`,
					color,
					dwellMs: 6000,
				};
			}

			case 'ConnectionDiscovered': {
				const now = Date.now();
				if (now - lastConnectionAt < CONNECTION_THROTTLE_MS) return null;
				lastConnectionAt = now;
				const kind = String(d.connection_type ?? 'link');
				const weight = Number(d.weight ?? 0);
				return {
					type: event.type,
					title: 'Bridge discovered',
					body: `${kind} · weight ${weight.toFixed(2)}`,
					color,
					dwellMs: 4500,
				};
			}

			case 'MemoryPromoted': {
				const r = Number(d.new_retention ?? 0);
				return {
					type: event.type,
					title: 'Memory promoted',
					body: `retention ${(r * 100).toFixed(0)}%`,
					color,
					dwellMs: 4500,
				};
			}

			case 'MemoryDemoted': {
				const r = Number(d.new_retention ?? 0);
				return {
					type: event.type,
					title: 'Memory demoted',
					body: `retention ${(r * 100).toFixed(0)}%`,
					color,
					dwellMs: 4500,
				};
			}

			case 'MemorySuppressed': {
				const count = Number(d.suppression_count ?? 0);
				const cascade = Number(d.estimated_cascade ?? 0);
				return {
					type: event.type,
					title: 'Forgetting',
					body: cascade > 0
						? `suppression #${count} · Rac1 cascade ~${cascade} neighbors`
						: `suppression #${count}`,
					color,
					dwellMs: 5500,
				};
			}

			case 'MemoryUnsuppressed': {
				const remaining = Number(d.remaining_count ?? 0);
				return {
					type: event.type,
					title: 'Recovered',
					body: remaining > 0 ? `${remaining} suppression${remaining === 1 ? '' : 's'} remain` : 'fully unsuppressed',
					color,
					dwellMs: 5000,
				};
			}

			case 'Rac1CascadeSwept': {
				const seeds = Number(d.seeds ?? 0);
				const neighbors = Number(d.neighbors_affected ?? 0);
				return {
					type: event.type,
					title: 'Rac1 cascade',
					body: `${seeds} seed${seeds === 1 ? '' : 's'} · ${neighbors} dendritic spine${neighbors === 1 ? '' : 's'} pruned`,
					color,
					dwellMs: 6000,
				};
			}

			case 'MemoryDeleted': {
				return {
					type: event.type,
					title: 'Memory deleted',
					body: String(d.id ?? '').slice(0, 8),
					color,
					dwellMs: 4000,
				};
			}

			// Noise — never toast
			case 'Heartbeat':
			case 'SearchPerformed':
			case 'RetentionDecayed':
			case 'ActivationSpread':
			case 'ImportanceScored':
			case 'MemoryCreated':
			case 'MemoryUpdated':
			case 'DreamStarted':
			case 'DreamProgress':
			case 'ConsolidationStarted':
			case 'Connected':
				return null;

			default:
				return null;
		}
	}

	// Track the latest processed event by object identity. The websocket
	// store prepends new events (index 0) and caps the array, so we walk
	// from the head until we hit a previously-seen event rather than
	// comparing only events[0] — which would drop mid-burst events when
	// multiple messages arrive in the same Svelte update tick (e.g. a
	// swarm epiphany firing DreamCompleted + ConnectionDiscovered within
	// a single millisecond).
	let lastSeen: VestigeEvent | null = null;

	eventFeed.subscribe(events => {
		if (events.length === 0) return;
		const fresh: VestigeEvent[] = [];
		for (const e of events) {
			if (e === lastSeen) break;
			fresh.push(e);
		}
		if (fresh.length === 0) return;
		lastSeen = events[0];
		// Process oldest-first so narrative ordering is preserved.
		for (let i = fresh.length - 1; i >= 0; i--) {
			const translated = translate(fresh[i]);
			if (translated) push(translated);
		}
	});

	return {
		subscribe,
		dismiss,
		clear,
		pauseDwell,
		resumeDwell,
		/** Manually fire a toast (test mode / demo button). */
		push,
	};
}

export const toasts = createToastStore();

/** Fire a synthetic event sequence — used by the demo button in settings. */
export function fireDemoSequence(): void {
	const demos: Omit<Toast, 'id' | 'createdAt'>[] = [
		{
			type: 'DreamCompleted',
			title: 'Dream consolidated',
			body: 'Replayed 127 memories · 43 new connections · 5 insights in 2.4s',
			color: EVENT_TYPE_COLORS.DreamCompleted,
			dwellMs: 7000,
		},
		{
			type: 'ConnectionDiscovered',
			title: 'Bridge discovered',
			body: 'semantic · weight 0.87',
			color: EVENT_TYPE_COLORS.ConnectionDiscovered,
			dwellMs: 4500,
		},
		{
			type: 'MemorySuppressed',
			title: 'Forgetting',
			body: 'suppression #2 · Rac1 cascade ~8 neighbors',
			color: EVENT_TYPE_COLORS.MemorySuppressed,
			dwellMs: 5500,
		},
		{
			type: 'ConsolidationCompleted',
			title: 'Consolidation swept',
			body: '892 nodes · 156 decayed · 48 embedded in 1.1s',
			color: EVENT_TYPE_COLORS.ConsolidationCompleted,
			dwellMs: 6000,
		},
	];
	demos.forEach((t, i) => {
		setTimeout(() => {
			// access the store getter for type safety — push is on the factory
			(toasts as unknown as { push: typeof toasts.push }).push(t);
		}, i * 800);
	});
	// satisfy unused-import lint
	void get(toasts);
}
