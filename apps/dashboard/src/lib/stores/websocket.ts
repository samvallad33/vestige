import { writable, derived } from 'svelte/store';
import type { VestigeEvent } from '$types';

const MAX_EVENTS = 200;

function createWebSocketStore() {
	const { subscribe, set, update } = writable<{
		connected: boolean;
		events: VestigeEvent[];
		lastHeartbeat: VestigeEvent | null;
		error: string | null;
	}>({
		connected: false,
		events: [],
		lastHeartbeat: null,
		error: null
	});

	let ws: WebSocket | null = null;
	let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	let reconnectAttempts = 0;

	function connect(url?: string) {
		const wsUrl = url || (window.location.port === '5173'
			? `ws://${window.location.hostname}:3927/ws`
			: `ws://${window.location.host}/ws`);

		if (ws?.readyState === WebSocket.OPEN) return;

		try {
			ws = new WebSocket(wsUrl);

			ws.onopen = () => {
				reconnectAttempts = 0;
				update(s => ({ ...s, connected: true, error: null }));
			};

			ws.onmessage = (event) => {
				try {
					const parsed: VestigeEvent = JSON.parse(event.data);
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
		const delay = Math.min(1000 * 2 ** reconnectAttempts, 30000);
		reconnectAttempts++;
		reconnectTimer = setTimeout(() => connect(url), delay);
	}

	function disconnect() {
		if (reconnectTimer) clearTimeout(reconnectTimer);
		ws?.close();
		ws = null;
		set({ connected: false, events: [], lastHeartbeat: null, error: null });
	}

	function clearEvents() {
		update(s => ({ ...s, events: [] }));
	}

	/**
	 * Inject a synthetic event into the feed as if it had arrived over the
	 * WebSocket. Used by the dev-mode "Preview Birth Ritual" button on the
	 * Settings page to let Sam trigger a demo of the v2.3 Memory Birth
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
		clearEvents,
		injectEvent
	};
}

export const websocket = createWebSocketStore();

// Derived stores for specific event types
export const isConnected = derived(websocket, $ws => $ws.connected);
export const eventFeed = derived(websocket, $ws => $ws.events);
export const heartbeat = derived(websocket, $ws => $ws.lastHeartbeat);
export const memoryCount = derived(websocket, $ws =>
	($ws.lastHeartbeat?.data?.memory_count as number) ?? 0
);
export const avgRetention = derived(websocket, $ws =>
	($ws.lastHeartbeat?.data?.avg_retention as number) ?? 0
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
