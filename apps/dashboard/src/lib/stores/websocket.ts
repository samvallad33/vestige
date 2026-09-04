import { writable, derived } from 'svelte/store';
import type { VestigeEvent } from '$types';

const MAX_EVENTS = 200;

/**
 * The dashboard API and its socket are one origin from the browser's point of
 * view. Keeping the port/protocol from `location` lets Vite proxy `/ws` in
 * development and uses the deployed origin in production (including WSS on
 * HTTPS).
 */
export function defaultWebSocketUrl(location: Pick<Location, 'protocol' | 'host'>): string {
	const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
	return `${protocol}//${location.host}/ws`;
}

function createWebSocketStore() {
	const { subscribe, set, update } = writable<{
		connected: boolean;
		reconnecting: boolean;
		events: VestigeEvent[];
		lastHeartbeat: VestigeEvent | null;
		error: string | null;
	}>({
		connected: false,
		reconnecting: false,
		events: [],
		lastHeartbeat: null,
		error: null
	});

	let ws: WebSocket | null = null;
	let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	let reconnectAttempts = 0;

	function connect(url?: string) {
		// Same-host always: the release binary serves /ws itself and the vite
		// dev server proxies /ws (any port — never hardcode a dev port here:
		// a 5173-only branch silently killed the live layer on other ports).
		// window is only touched when no explicit url is given (Node tests
		// always pass one).
		const wsUrl = url || defaultWebSocketUrl(window.location);

		if (ws?.readyState === WebSocket.OPEN) return;

		try {
			ws = new WebSocket(wsUrl);

			ws.onopen = () => {
				reconnectAttempts = 0;
				update(s => ({ ...s, connected: true, reconnecting: false, error: null }));
			};

			ws.onmessage = (event) => {
				try {
					const parsed: VestigeEvent = JSON.parse(event.data);
					if (parsed.type === 'EventsDropped') {
						// A hole in the stream is a fact worth stating out loud: the
						// feed keeps the marker so the UI can show the gap and refetch.
						console.warn(
							`[vestige] live feed dropped ${String(parsed.data?.missed ?? '?')} events (slow subscriber); state may be stale until the next refresh`
						);
					}
					update(s => {
						if (parsed.type === 'Heartbeat') {
							return { ...s, lastHeartbeat: parsed };
						}
						const events = [parsed, ...s.events].slice(0, MAX_EVENTS);
						return { ...s, events };
					});
				} catch (e) {
					console.warn('[vestige] Failed to parse WebSocket message:', e);
				}
			};

			ws.onclose = () => {
				update(s => ({ ...s, connected: false }));
				scheduleReconnect(wsUrl);
			};

			ws.onerror = () => {
				update(s => ({ ...s, error: 'WebSocket connection failed' }));
			};
		} catch (e) {
			update(s => ({ ...s, error: String(e) }));
		}
	}

	function scheduleReconnect(url: string) {
		if (reconnectTimer) clearTimeout(reconnectTimer);
		update(s => ({ ...s, reconnecting: true }));
		const delay = Math.min(1000 * 2 ** reconnectAttempts, 30000);
		reconnectAttempts++;
		reconnectTimer = setTimeout(() => connect(url), delay);
	}

	function disconnect() {
		if (reconnectTimer) {
			clearTimeout(reconnectTimer);
			reconnectTimer = null;
		}
		if (ws) {
			// Detach onclose BEFORE closing: otherwise close() fires the handler,
			// which calls scheduleReconnect and resurrects the socket we just
			// asked to tear down.
			ws.onclose = null;
			ws.onerror = null;
			ws.onmessage = null;
			ws.close();
		}
		ws = null;
		// Keep lastHeartbeat: it is the last-known truth, and zeroing it made
		// every consumer flash "0 memories" on remount. Stale-but-real beats
		// fresh-but-false; the next heartbeat replaces it within seconds.
		update(s => ({ ...s, connected: false, reconnecting: false, events: [], error: null }));
	}

	function clearEvents() {
		update(s => ({ ...s, events: [] }));
	}

	/**
	 * Full teardown INCLUDING lastHeartbeat — unlike disconnect(), which keeps
	 * the last-known vitals so consumers never flash a lying zero. Use for a
	 * true fresh-start (tests, switching stores), never for route unmounts.
	 */
	function reset() {
		disconnect();
		set({ connected: false, reconnecting: false, events: [], lastHeartbeat: null, error: null });
	}

	/**
	 * Inject a synthetic event into the feed as if it had arrived over the
	 * WebSocket. Used by the dev-mode "Preview Birth Ritual" button on the
	 * Settings page to let developers trigger a demo of the v2.3 Memory Birth
	 * Ritual without ingesting a real memory. Downstream consumers —
	 * InsightToast, Graph3D — cannot distinguish synthetic from real.
	 */
	function injectEvent(event: VestigeEvent) {
		update(s => {
			const events = [event, ...s.events].slice(0, MAX_EVENTS);
			return { ...s, events };
		});
	}

	return {
		subscribe,
		connect,
		disconnect,
		reset,
		clearEvents,
		injectEvent
	};
}

export const websocket = createWebSocketStore();

// Derived stores for specific event types
export const isConnected = derived(websocket, $ws => $ws.connected);
export const isReconnecting = derived(websocket, $ws => $ws.reconnecting);
export const eventFeed = derived(websocket, $ws => $ws.events);
export const heartbeat = derived(websocket, $ws => $ws.lastHeartbeat);
// null = no heartbeat yet — render "—", never a lying literal 0. A store
// with 3,000 memories must not claim "0 memories" for the first seconds of
// every page load (or forever, when the socket can't connect).
export const memoryCount = derived(websocket, $ws =>
	($ws.lastHeartbeat?.data?.memory_count as number | undefined) ?? null
);
export const avgRetention = derived(websocket, $ws =>
	($ws.lastHeartbeat?.data?.avg_retention as number | undefined) ?? null
);
// v2.0.5: count of memories actively being forgotten (suppression_count > 0)
export const suppressedCount = derived(websocket, $ws =>
	($ws.lastHeartbeat?.data?.suppressed_count as number) ?? 0
);

// v2.0.7: uptime of the MCP server in seconds, refreshed every heartbeat.
// Exposed raw so callers can format as they like; the helper below is the
// standard compact format ("3d 4h 22m", "18m", "47s") used in the sidebar.
export const uptimeSeconds = derived(websocket, $ws =>
	($ws.lastHeartbeat?.data?.uptime_secs as number) ?? 0
);

// Agent Black Box (v2.2): the live stream of trace events, newest first. Each
// is a real `VestigeEvent::TraceEvent` backed by a persisted `agent_traces`
// row — the dashboard pulse is only ever driven by these, never by fakes.
export const traceEvents = derived(websocket, $ws =>
	$ws.events.filter((e) => e.type === 'TraceEvent')
);

// The most recent runId seen on the live feed — the "current run" indicator in
// Proof Mode / the Black Box live header.
export const liveRunId = derived(websocket, $ws => {
	const latest = $ws.events.find((e) => e.type === 'TraceEvent');
	return (latest?.data?.run_id as string) ?? null;
});

// The single most recent trace event (for the "last event" readout).
export const lastTraceEvent = derived(websocket, $ws =>
	$ws.events.find((e) => e.type === 'TraceEvent') ?? null
);

// Live Memory PR notifications (opened / decided) for the queue badge + toasts.
export const memoryPrEvents = derived(websocket, $ws =>
	$ws.events.filter((e) => e.type === 'MemoryPrOpened' || e.type === 'MemoryPrDecided')
);

export function formatUptime(secs: number): string {
	if (!Number.isFinite(secs) || secs < 0) return '—';
	const d = Math.floor(secs / 86_400);
	const h = Math.floor((secs % 86_400) / 3_600);
	const m = Math.floor((secs % 3_600) / 60);
	const s = Math.floor(secs % 60);
	// Compact representation: show the two most significant units so the
	// sidebar stays readable ("3d 4h" not "3d 4h 22m 17s", "18m 43s", etc).
	if (d > 0) return h > 0 ? `${d}d ${h}h` : `${d}d`;
	if (h > 0) return m > 0 ? `${h}h ${m}m` : `${h}h`;
	if (m > 0) return s > 0 ? `${m}m ${s}s` : `${m}m`;
	return `${s}s`;
}
