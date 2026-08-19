<script lang="ts">
	import { onMount } from 'svelte';
	import {
		websocket,
		eventFeed,
		isConnected,
		isReconnecting,
		heartbeat,
		uptimeSeconds,
		formatUptime
	} from '$stores/websocket';
	import { api, type ChangelogEvent } from '$stores/api';
	import { changelogEventToVestigeEvent, mergeFeedEvents } from '$lib/feed-history';
	import type { VestigeEvent } from '$types';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01 } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { eventImpulse01 } from '$lib/observatory/cognitive-palette';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import Dropdown, { type DropdownOption } from '$components/Dropdown.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';

	type FeedTextItem = TextLayerItem & { event?: VestigeEvent; eventKey?: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const AMBER = [...rgb01('#FFB000'), 0.9] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const MUTED = [...rgb01('#29F2A9'), 0.62] satisfies [number, number, number, number];
	const ROW_LIMIT = 42;

	let textPass: TextLayerPass | null = null;
	let focusedRun: string | null = null;
	// Tracked when the user picks a feed event from the canvas OR the DOM list.
	// Selection only — no API call, no clearEvents. The live feed stays intact.
	let selectedEventKey: string | null = $state(null);
	// engine handle captured in the pass factory so buildTextItems can read the live
	// viewport aspect (params[6]/[7]) for portrait-only event-line shortening.
	let engineHandle: ObservatoryEngine | null = null;
	let historyEvents = $state<VestigeEvent[]>([]);
	let historyLoading = $state(true);

	// The WebSocket only knows what happened after this page opened. Hydrate the
	// same renderer from the durable changelog so a quiet agent still has a real
	// cognitive history to inspect on first paint.
	onMount(() => {
		void api.memoryChangelog(100)
			.then((response) => {
				historyEvents = (response.events ?? []).map((event: ChangelogEvent) => changelogEventToVestigeEvent(event));
			})
			.catch(() => {
				// A live feed remains useful if the optional historical read fails.
				historyEvents = [];
			})
			.finally(() => {
				historyLoading = false;
			});
	});

	const feedEvents = $derived.by(() => mergeFeedEvents(historyEvents, $eventFeed));

	// Live viewport aspect, same source (and window fallback) TextLayerPass.portraitAdapt
	// uses. On a phone a full 138-char event line can't be size-boosted (it already
	// fills the width), so portrait renders a SHORTER line that portraitAdapt can grow
	// to a readable size. NOTHING is hardcoded per width — it scales with aspect.
	function isPortrait(): boolean {
		const vw = engineHandle?.params[6] || 0;
		const vh = engineHandle?.params[7] || 0;
		if (vw > 0 && vh > 0) return vw / vh < 0.85;
		if (typeof window !== 'undefined' && window.innerHeight > 0) return window.innerWidth / window.innerHeight < 0.85;
		return false;
	}

	let feedScene: RouteSceneModel = $derived(buildFeedScene(feedEvents));

	$effect(() => {
		textPass?.setText(buildTextItems(feedEvents, $isConnected, $isReconnecting));
	});

	// Selection state surfaces to the renderer: a chosen event is highlighted
	// through the same run-depth channel the hover handler uses, but persistently
	// (until the next pick). This is a pure SELECT, no API call.
	$effect(() => {
		if (!textPass) return;
		textPass.setRunDepth(selectedEventKey ? `feed:${selectedEventKey}` : null, 1);
	});

	function createFeedPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		engineHandle = engine;
		const field = new FeedFieldPass(engine);
		field.uploadScene(scene);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		void pass.init().then(() => pass.setText(buildTextItems(feedEvents, $isConnected, $isReconnecting)));
		return [field,
			{
				render(renderPass) {
					pass.render(renderPass);
				},
				pickAt(ndcX, ndcY) {
					const hit = pass.pickAt(ndcX, ndcY);
					if (!hit) {
						if (focusedRun) {
							focusedRun = null;
							pass.setRunDepth(null);
						}
						return null;
					}
					if (hit.id !== focusedRun) {
						focusedRun = hit.id;
						pass.setRunDepth(hit.id, 1);
					}
					return hit;
				},
				dispose() {
					pass.dispose();
					if (textPass === pass) textPass = null;
				}
			}
		];
	}

	class FeedFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
			const vw = engine.params[6] || (typeof window !== 'undefined' ? window.innerWidth : 0);
			const vh = engine.params[7] || (typeof window !== 'undefined' ? window.innerHeight : 1);
			const portrait = vw / Math.max(1, vh) < 0.85;
			// Preserve the verified portrait substrate exactly; desktop has enough room
			// for a richer field while the reading well protects every feed row.
			this.field.setIntensity(portrait ? 0.22 : 0.9);
			// Feed rows are left-anchored (x=-0.9) but stream rightward as wide lines
			// (maxWidthEm 58 ~= full width), stacked from y=+0.72 down to y=-0.78. A
			// left-column-only well would leave the right half of every row drowning,
			// so the reading well spans the whole text band the rows occupy. Desktop's
			// higher floor lets structure breathe through without competing with glyphs.
			this.field.setReadingWell({
				x: portrait ? 0 : -0.08,
				y: portrait ? -0.02 : 0.62,
				hw: portrait ? 0.94 : 0.62,
				hh: portrait ? 0.84 : 0.24,
				floor: 0.06,
				soft: 0.22
			});
		}
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node, index) => {
				const event = feedEvents[index];
				return {
					id: node.source.id,
					score: node.retention,
					hue: event ? eventImpulse01(event.type) : [CYAN[0], CYAN[1], CYAN[2]],
					energy: node.activation,
					metric2: node.trust,
					scar: node.type.includes('Deleted') || node.type.includes('Demoted') || node.type.includes('Verdict'),
					kind: 'feed-event',
					payload: event
				};
			});
			const portrait = isPortrait();
			// A desktop feed can legitimately be quiet at the exact sampling instant.
			// Anchor that real connection state as the ambient cell so the substrate
			// stays alive without changing the verified empty portrait treatment.
			if (data.length === 0 && !portrait) {
				const connected = scene.scalars.connected > 0;
				data.push({
					id: 'feed:stream-state',
					score: connected ? 0.72 : 0.48,
					hue: [CYAN[0], CYAN[1], CYAN[2]],
					energy: scene.scalars.reconnecting > 0 ? 0.82 : 0.56,
					metric2: connected ? 0.76 : 0.42,
					kind: 'feed-stream-state',
					payload: { connected, reconnecting: scene.scalars.reconnecting > 0 }
				});
			}
			// A quiet live stream may contain only a couple real events — give those a
			// broad shockwave so the field still fills. But a busy stream (up to
			// ROW_LIMIT events) needs SMALL cells or they overlap into one blob; scale
			// the cell size to the real event count instead of hardcoding the sparse case.
			const sparse = data.length < 4;
			this.field.setCells(
				layoutGalaxy(data, {
					maxRadius: sparse ? 0.78 : 0.92,
					minCellR: sparse ? (portrait ? 0.24 : 0.62) : 0.018,
					maxCellR: sparse ? (portrait ? 0.32 : 0.78) : 0.06
				}),
				portrait ? undefined : { ambient: 0.5 }
			);
		}
		compute(encoder: GPUCommandEncoder): void { this.field.compute(encoder); }
		render(pass: GPURenderPassEncoder): void { this.field.render(pass); }
		pickAt(x: number, y: number): RoutePick | null { return this.field.pickAt(x, y); }
		dispose(): void { this.field.dispose(); }
	}

	function buildFeedScene(events: VestigeEvent[]): RouteSceneModel {
		const rows = events.slice(0, ROW_LIMIT);
		return {
			organ: 'feed',
			nodes: rows.map((event, index) => ({
				source: { kind: 'event', id: eventKey(event, index) },
				index,
				label: event.type,
				retention: eventEnergy(event),
				activation: recencyDepth(rows, event, index),
				trust: eventEnergy(event),
				tags: Object.keys(event.data).slice(0, 6),
				type: event.type
			})),
			edges: [],
			events: rows.map((event, index) => ({
				source: { kind: 'event', id: eventKey(event, index) },
				type: event.type,
				targetIndex: index,
				frame: index,
				energy: eventEnergy(event)
			})),
			receipts: [],
			scalars: {
				eventCount: events.length,
				connected: $isConnected ? 1 : 0,
				reconnecting: $isReconnecting ? 1 : 0
			},
			alive: rows.length > 0
		};
	}

	function buildTextItems(events: VestigeEvent[], connected: boolean, reconnecting: boolean): FeedTextItem[] {
		const rows = events.slice(0, ROW_LIMIT);
		// Every new-in-this-fix item is gated to portrait so the 1440px desktop render
		// stays byte-identical (the desktop path below is the ORIGINAL feed layout).
		if (isPortrait()) return buildPortraitItems(rows, connected, reconnecting);

		// ── DESKTOP (unchanged): empty -> connection status; populated -> wide rows ──
		if (rows.length === 0) {
			return [statusItem(connectionPayload(connected, reconnecting), reconnecting ? AMBER : connected ? MUTED : SCARLET)];
		}
		const top = 0.72;
		const rowStep = 1.5 / Math.max(1, ROW_LIMIT - 1);
		return rows.map((event, index) => {
			const key = eventKey(event, index);
			return {
				id: `feed:${key}`,
				kind: 'feed-event',
				event,
				eventKey: key,
				text: eventLine(event, key),
				x: -0.9,
				y: top - index * rowStep,
				size: 0.024,
				color: eventColor(event),
				depth: 0.6 + 0.4 * recencyDepth(rows, event, index),
				weight: eventEnergy(event),
				// startFrame:0 + a near-instant revealSpan keeps rows fully lit
				// regardless of the demo clock, which RouteStage resets on every
				// scene upload — and feed's scene rebuilds on every WS event. A
				// longer reveal here would perpetually restart and leave rows dim.
				startFrame: 0,
				revealSpan: 1,
				maxWidthEm: 58,
				hitPadX: 0.03,
				hitPadY: 0.013
			};
		});
	}

	// ── PORTRAIT: content-first, readable feed for a phone. A first-time visitor gets
	// a real title, a plain-English status, and either a legible empty state or short
	// event rows sized to fit. All of this is portrait-only; desktop keeps its layout.
	function buildPortraitItems(rows: VestigeEvent[], connected: boolean, reconnecting: boolean): FeedTextItem[] {
		const items: FeedTextItem[] = [];
		items.push({
			id: 'feed:title',
			kind: 'feed-title',
			text: 'LIVE FEED',
			x: -0.94,
			y: 0.9,
			size: 0.04,
			color: CYAN,
			depth: 1,
			weight: 0.9,
			revealSpan: 12
		});
		items.push({
			id: 'feed:status',
			kind: 'feed-state',
			text: connectionLine(connected, reconnecting, rows.length),
			x: -0.94,
			y: 0.82,
			size: 0.022,
			color: reconnecting ? AMBER : connected ? MUTED : SCARLET,
			depth: 0.85,
			weight: 0.62,
			revealSpan: 18,
			maxWidthEm: 40
		});

		if (rows.length === 0) {
			// Legible empty state — explain what will appear here instead of a black void.
			items.push({
				id: 'feed:empty-1',
				kind: 'feed-empty',
				text: connected ? 'WAITING FOR LIVE ACTIVITY' : 'STREAM OFFLINE',
				x: -0.7,
				y: 0.12,
				size: 0.03,
				color: connected ? CYAN : SCARLET,
				depth: 0.7,
				weight: 0.7,
				revealSpan: 20,
				maxWidthEm: 40
			});
			items.push({
				id: 'feed:empty-2',
				kind: 'feed-empty',
				text: connected
					? 'Recalls, ingests and consolidations stream here as they happen.'
					: 'Reconnecting to the live event stream...',
				x: -0.7,
				y: 0.02,
				size: 0.02,
				color: MUTED,
				depth: 0.55,
				weight: 0.5,
				revealSpan: 24,
				maxWidthEm: 40
			});
			return items;
		}

		// A full 138-char line already fills the phone width, so portraitAdapt can't
		// grow it and it renders phone-tiny. Render a SHORT line (type + summary, no id),
		// fewer rows, at a larger authored size that fits and reads.
		const visible = rows.slice(0, 12);
		const top = 0.68;
		const rowStep = 0.13;
		visible.forEach((event, index) => {
			const key = eventKey(event, index);
			items.push({
				id: `feed:${key}`,
				kind: 'feed-event',
				event,
				eventKey: key,
				text: eventLineShort(event),
				x: -0.9,
				y: top - index * rowStep,
				size: 0.03,
				color: eventColor(event),
				depth: 0.6 + 0.4 * recencyDepth(visible, event, index),
				weight: eventEnergy(event),
				startFrame: 0,
				revealSpan: 1,
				maxWidthEm: 34,
				hitPadX: 0.03,
				hitPadY: 0.013
			});
		});
		return items;
	}

	// Human-readable status line — never a raw ISO timestamp (it overflowed the phone
	// edge) and never JSON. Short, self-explanatory, and count-aware.
	function connectionLine(connected: boolean, reconnecting: boolean, count: number): string {
		const state = reconnecting ? 'RECONNECTING' : connected ? 'CONNECTED' : 'OFFLINE';
		const suffix = count > 0 ? `${count} EVENT${count === 1 ? '' : 'S'}` : 'LIVE';
		return sanitizeAscii(`${state} - ${suffix}`);
	}

	function statusItem(text: string, color: [number, number, number, number]): FeedTextItem {
		return {
			id: 'feed:connection-state',
			kind: 'feed-state',
			text: sanitizeAscii(text),
			x: -0.52,
			y: 0.03,
			size: 0.038,
			color,
			depth: 0.72,
			weight: 0.62,
			revealSpan: 24,
			maxWidthEm: 48
		};
	}

	function connectionPayload(connected: boolean, reconnecting: boolean): string {
		return JSON.stringify({ connected, reconnecting, events: 0 });
	}

	function eventLine(event: VestigeEvent, key: string): string {
		const summary = payloadSummary(event.data);
		return sanitizeAscii(`${event.type} | ${key.slice(0, 24)} | ${summary}`.slice(0, 138));
	}

	// Portrait row: short enough that portraitAdapt can size-boost it to a readable
	// height on a phone. Event type + a compact payload summary, no id column.
	function eventLineShort(event: VestigeEvent): string {
		const summary = payloadSummary(event.data);
		return sanitizeAscii(`${event.type} - ${summary}`.slice(0, 46));
	}

	function payloadSummary(data: Record<string, unknown>): string {
		const pairs = Object.entries(data)
			.filter(([, value]) => value !== null && value !== undefined && typeof value !== 'object')
			.slice(0, 5)
			.map(([name, value]) => `${name}=${String(value).replace(/\s+/g, ' ').slice(0, 32)}`);
		return pairs.length ? pairs.join(' ') : JSON.stringify(data).replace(/\s+/g, ' ').slice(0, 82);
	}

	function eventKey(event: VestigeEvent, index: number): string {
		const d = event.data;
		const raw =
			d.id ??
			d.event_id ??
			d.memory_id ??
			d.trace_id ??
			d.run_id ??
			d.receipt_id ??
			d.pr_id ??
			d.timestamp ??
			d.at ??
			`${event.type}:${index}`;
		return sanitizeAscii(String(raw));
	}

	function eventEnergy(event: VestigeEvent): number {
		const d = event.data;
		const candidates = [
			d.energy,
			d.confidence,
			d.strength,
			d.weight,
			d.composite_score,
			d.new_retention,
			d.retention,
			d.result_count,
			d.connections_found,
			d.insights_generated,
			d.nodes_processed,
			d.duration_ms
		];
		const raw = candidates.map(Number).find((value) => Number.isFinite(value));
		if (raw === undefined) return 0.5;
		if (raw < 0) return 0;
		if (raw <= 1) return raw;
		return clamp01(Math.log10(raw + 1) / 3);
	}

	function recencyDepth(events: VestigeEvent[], event: VestigeEvent, index: number): number {
		const times = events.map(eventTime).filter((time): time is number => time !== null);
		const time = eventTime(event);
		if (time !== null && times.length > 1) {
			const min = Math.min(...times);
			const max = Math.max(...times);
			if (max > min) return clamp01((time - min) / (max - min));
		}
		return clamp01(1 - index / Math.max(1, ROW_LIMIT - 1));
	}

	function eventTime(event: VestigeEvent): number | null {
		const raw = event.data.timestamp ?? event.data.at ?? event.data.created_at;
		if (typeof raw === 'number' && Number.isFinite(raw)) return raw;
		if (typeof raw === 'string') {
			const parsed = Date.parse(raw);
			return Number.isFinite(parsed) ? parsed : null;
		}
		return null;
	}

	function eventColor(event: VestigeEvent): [number, number, number, number] {
		if (event.type.includes('Deleted') || event.type.includes('Demoted') || event.type.includes('Verdict')) return SCARLET;
		if (event.type.includes('Progress') || event.type.includes('Started')) return AMBER;
		return CYAN;
	}

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
	}

	function handleRoutePick(pick: RoutePick) {
		// Plain click on a feed event must SELECT/INSPECT only — never mutate.
		// Tracking the focused run here (without destroying the live feed) lets
		// the renderer highlight the chosen event. clearEvents() is reserved
		// for an explicit, separately-labeled "Clear" control, not a generic pick.
		if (pick.kind === 'feed-event') {
			const raw = String(pick.id).replace(/^feed:/, '');
			selectedEventKey = raw;
		}
	}

	// ────────────────────────────────────────────────────────────────────
	//  DOM OVERLAY — the legible, first-time-visitor layer on top of the
	//  living WebGPU field. Mirrors the /contradictions gold standard:
	//  PageHeader + real stat cards + a typed event list + selection panel.
	//  All data is READ LIVE from the websocket stores; nothing is fabricated.
	// ────────────────────────────────────────────────────────────────────

	// Human labels + accent tint for every VestigeEventType we render, so a
	// first-time visitor reads "Memory recalled" instead of raw `SearchPerformed`.
	type Tone = 'cyan' | 'amber' | 'scarlet' | 'violet' | 'green';
	const EVENT_META: Record<string, { label: string; tone: Tone }> = {
		Connected: { label: 'Stream connected', tone: 'green' },
		MemoryCreated: { label: 'Memory ingested', tone: 'green' },
		MemoryUpdated: { label: 'Memory updated', tone: 'cyan' },
		MemoryDeleted: { label: 'Memory deleted', tone: 'scarlet' },
		MemoryPromoted: { label: 'Memory promoted', tone: 'green' },
		MemoryDemoted: { label: 'Memory demoted', tone: 'amber' },
		MemorySuppressed: { label: 'Memory suppressed', tone: 'violet' },
		MemoryUnsuppressed: { label: 'Suppression reversed', tone: 'cyan' },
		Rac1CascadeSwept: { label: 'Forgetting cascade swept', tone: 'violet' },
		SearchPerformed: { label: 'Memory recalled', tone: 'cyan' },
		DreamStarted: { label: 'Dream started', tone: 'amber' },
		DreamProgress: { label: 'Dream in progress', tone: 'amber' },
		DreamCompleted: { label: 'Dream completed', tone: 'green' },
		ConsolidationStarted: { label: 'Consolidation started', tone: 'amber' },
		ConsolidationCompleted: { label: 'Consolidation completed', tone: 'green' },
		RetentionDecayed: { label: 'Retention decayed', tone: 'amber' },
		ConnectionDiscovered: { label: 'Connection discovered', tone: 'cyan' },
		ActivationSpread: { label: 'Activation spread', tone: 'cyan' },
		ImportanceScored: { label: 'Importance scored', tone: 'cyan' },
		DeepReferenceCompleted: { label: 'Deep reference completed', tone: 'green' },
		BackfillFired: { label: 'Salience backfill fired', tone: 'violet' },
		CausalReceipt: { label: 'Causal receipt written', tone: 'cyan' },
		HookVerdictRecorded: { label: 'Hook verdict recorded', tone: 'scarlet' },
		TraceEvent: { label: 'Agent trace event', tone: 'cyan' },
		MemoryPrOpened: { label: 'Memory PR opened', tone: 'amber' },
		MemoryPrDecided: { label: 'Memory PR decided', tone: 'green' },
		Heartbeat: { label: 'Heartbeat', tone: 'green' }
	};

	const TONE_DOT: Record<Tone, string> = {
		cyan: '#22C7DE',
		amber: '#FFB000',
		scarlet: '#FF3B30',
		violet: '#A78BFA',
		green: '#29F2A9'
	};

	function eventLabel(type: string): string {
		return EVENT_META[type]?.label ?? type;
	}
	function eventTone(type: string): Tone {
		return EVENT_META[type]?.tone ?? 'cyan';
	}

	// Absolute clock time for the row timestamp (falls back to "live" when the
	// event carries no time field — a real limitation, surfaced honestly).
	function eventClock(event: VestigeEvent): string {
		const t = eventTime(event);
		if (t === null) return 'live';
		try {
			return new Date(t).toLocaleTimeString(undefined, {
				hour: '2-digit',
				minute: '2-digit',
				second: '2-digit'
			});
		} catch {
			return 'live';
		}
	}

	// Filter lens — read-only view control over the SAME live list. Never mutates.
	type Lens = 'all' | 'memory' | 'cognition' | 'lifecycle';
	let lens = $state<Lens>('all');
	const lensOptions: DropdownOption[] = [
		{ value: 'all', label: 'All events', icon: 'feed' },
		{ value: 'memory', label: 'Memory changes', icon: 'memories' },
		{ value: 'cognition', label: 'Recall & reasoning', icon: 'reasoning' },
		{ value: 'lifecycle', label: 'Dreams & upkeep', icon: 'dreams' }
	];
	const MEMORY_TYPES = new Set([
		'MemoryCreated', 'MemoryUpdated', 'MemoryDeleted', 'MemoryPromoted',
		'MemoryDemoted', 'MemorySuppressed', 'MemoryUnsuppressed', 'MemoryPrOpened', 'MemoryPrDecided'
	]);
	const COGNITION_TYPES = new Set([
		'SearchPerformed', 'DeepReferenceCompleted', 'ConnectionDiscovered',
		'ActivationSpread', 'ImportanceScored', 'CausalReceipt', 'BackfillFired', 'TraceEvent'
	]);
	const LIFECYCLE_TYPES = new Set([
		'DreamStarted', 'DreamProgress', 'DreamCompleted', 'ConsolidationStarted',
		'ConsolidationCompleted', 'RetentionDecayed', 'Rac1CascadeSwept', 'HookVerdictRecorded'
	]);

	function lensMatch(type: string): boolean {
		switch (lens) {
			case 'memory': return MEMORY_TYPES.has(type);
			case 'cognition': return COGNITION_TYPES.has(type);
			case 'lifecycle': return LIFECYCLE_TYPES.has(type);
			default: return true;
		}
	}

	type FeedRow = { key: string; event: VestigeEvent; index: number };

	const rows = $derived<FeedRow[]>(
		feedEvents
			.slice(0, ROW_LIMIT)
			.map((event, index) => ({ key: eventKey(event, index), event, index }))
	);
	const visibleRows = $derived<FeedRow[]>(rows.filter((r) => lensMatch(r.event.type)));

	// ── Real stats, all read live from the websocket stores ──
	const totalEvents = $derived(feedEvents.length);
	const liveEventCount = $derived($eventFeed.length);
	const memoryEvents = $derived(feedEvents.filter((e) => MEMORY_TYPES.has(e.type)).length);
	const distinctTypes = $derived(new Set(feedEvents.map((e) => e.type)).size);
	const uptime = $derived(formatUptime($uptimeSeconds));

	// Selection — pure inspect, mirrors the canvas pick. No mutation.
	const selectedRow = $derived<FeedRow | null>(
		selectedEventKey ? visibleRows.find((r) => r.key === selectedEventKey) ?? null : null
	);
	function selectRow(key: string) {
		selectedEventKey = selectedEventKey === key ? null : key;
	}

	// Full key/value dump of a selected event — the interpretation panel. Read-only.
	function eventEntries(event: VestigeEvent): { k: string; v: string }[] {
		return Object.entries(event.data).map(([k, v]) => ({
			k,
			v: typeof v === 'object' ? JSON.stringify(v) : String(v)
		}));
	}

	// ── The ONLY caller of clearEvents in this page. Explicit, labeled, and
	//    behind a confirm step. A plain click NEVER reaches this. ──
	let confirmingClear = $state(false);
	function requestClear() {
		confirmingClear = true;
	}
	function cancelClear() {
		confirmingClear = false;
	}
	function confirmClear() {
		websocket.clearEvents();
		selectedEventKey = null;
		confirmingClear = false;
	}

	const statusLabel = $derived(
		$isReconnecting ? 'Reconnecting' : $isConnected ? 'Connected' : 'Offline'
	);
	const statusTone = $derived<Tone>(
		$isReconnecting ? 'amber' : $isConnected ? 'green' : 'scarlet'
	);
</script>

<svelte:head>
	<title>Live Feed · Vestige</title>
</svelte:head>

<RouteStage
	organ="feed"
	seed={`live-event-stream:${feedEvents.length}:${$isConnected ? 1 : 0}:${$isReconnecting ? 1 : 0}`}
	scene={feedScene}
	passes={createFeedPasses}
	loading={false}
	error={null}
	emptyLabel=""
	onpick={handleRoutePick}
/>

<div class="relative z-10 min-h-full p-6 space-y-6 pointer-events-none">
	<!-- (1) IDENTITY -->
	<div class="pointer-events-auto">
		<PageHeader
			icon="feed"
			title="Live Feed"
			subtitle="Real-time memory events as your agents read, write, and reconsider."
			accent="synapse"
		>
			<!-- Connection-status chip (real, from the websocket store) -->
			<span
				class="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium tabular-nums"
				style="border-color: {TONE_DOT[statusTone]}55; color: {TONE_DOT[statusTone]}; background: {TONE_DOT[statusTone]}12;"
			>
				<span class="ping-host inline-flex">
					<span class="h-2 w-2 rounded-full" style="background: {TONE_DOT[statusTone]}"></span>
				</span>
				{statusLabel}
			</span>
		</PageHeader>
	</div>

	<!-- (2) LIVE PROOF — real stat cards -->
	<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
		<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
			<div class="text-2xl text-bright font-bold tabular-nums">
				<AnimatedNumber value={totalEvents} />
			</div>
			<div class="text-xs text-dim mt-1">recorded + live events</div>
		</div>
		<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
			<div class="text-2xl font-bold tabular-nums" style="color: #29F2A9">
				<AnimatedNumber value={memoryEvents} />
			</div>
			<div class="text-xs text-dim mt-1">memory changes</div>
		</div>
		<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
			<div class="text-2xl text-bright font-bold tabular-nums">
				<AnimatedNumber value={distinctTypes} />
			</div>
			<div class="text-xs text-dim mt-1">distinct event kinds</div>
		</div>
		<div use:reveal={{ delay: 180, y: 12 }} class="p-4 glass rounded-xl lift">
			<div class="text-2xl text-bright font-bold tabular-nums">
				{uptime}
			</div>
			<div class="text-xs text-dim mt-1">server uptime</div>
		</div>
	</div>

	<!-- (3) PRIMARY ACTION + filter lens -->
	<div class="flex flex-wrap items-end gap-3 pointer-events-auto">
		<Dropdown
			options={lensOptions}
			value={lens}
			label="Lens"
			icon="filter"
			onChange={(v) => (lens = v as Lens)}
		/>

		{#if confirmingClear}
			<div class="ml-auto flex items-center gap-2">
				<span class="text-xs text-dim">Clear {liveEventCount} live buffered events?</span>
				<button
					type="button"
					onclick={confirmClear}
					class="rounded-lg bg-decay/20 px-3 py-2 text-xs font-medium text-decay transition hover:bg-decay/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-decay/60"
				>
					Confirm clear
				</button>
				<button
					type="button"
					onclick={cancelClear}
					class="rounded-lg border border-subtle/30 px-3 py-2 text-xs text-muted transition hover:text-text hover:border-synapse/30"
				>
					Cancel
				</button>
			</div>
		{:else}
			<button
				type="button"
				onclick={requestClear}
				disabled={liveEventCount === 0}
				title={liveEventCount === 0 ? 'No live events buffered — recorded history remains available' : 'Clear only the local live-event buffer; recorded history is not deleted'}
				class="ml-auto inline-flex items-center gap-1.5 rounded-xl border px-3.5 py-2 text-xs font-medium transition lift
					disabled:cursor-not-allowed disabled:opacity-40
					border-subtle/30 text-dim hover:enabled:text-text hover:enabled:border-synapse/30 hover:enabled:bg-white/[0.03]"
			>
				<Icon name="close" size={13} />
				Clear live buffer
			</button>
		{/if}
	</div>

	<!-- (5) STATE GUIDANCE + typed event list -->
	<div class="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-4 pointer-events-auto">
		<!-- Event list -->
		<div class="glass-panel rounded-2xl p-3 min-h-[420px]">
			{#if rows.length === 0}
				<!-- EMPTY / IDLE — designed, not a void -->
				<div class="flex flex-col items-center justify-center gap-3 py-16 text-center enter">
					<div
						class="flex h-14 w-14 items-center justify-center rounded-2xl border"
						style="border-color: {TONE_DOT[statusTone]}40; background: {TONE_DOT[statusTone]}12; color: {TONE_DOT[statusTone]};"
					>
						<Icon name="feed" size={26} draw />
					</div>
					{#if historyLoading}
						<div class="text-sm font-medium text-bright">Recovering recorded activity…</div>
						<div class="max-w-sm text-xs text-muted">Loading the durable cognitive changelog while the live stream stays connected.</div>
					{:else if $isConnected}
						<div class="text-sm font-medium text-bright">Connected — waiting for agent activity</div>
						<div class="max-w-sm text-xs text-muted">
							Every recall, ingest, promotion, and consolidation your agents perform will stream in here the moment it happens. Nothing has fired since this session opened.
						</div>
					{:else if $isReconnecting}
						<div class="text-sm font-medium text-bright">Reconnecting to the live stream…</div>
						<div class="max-w-sm text-xs text-muted">
							The event socket dropped and is retrying with backoff. Events will resume automatically once it reconnects.
						</div>
					{:else}
						<div class="text-sm font-medium text-bright">Stream offline</div>
						<div class="max-w-sm text-xs text-muted">
							The MCP server's event socket isn't reachable. Start the Vestige server, then this feed will connect on its own.
						</div>
					{/if}
				</div>
			{:else if visibleRows.length === 0}
				<div class="flex flex-col items-center justify-center gap-3 py-16 text-center">
					<div class="text-dim opacity-50 breathe">
						<Icon name="filter" size={40} strokeWidth={1.2} />
					</div>
					<p class="text-dim text-sm">No events match this lens.</p>
					<button
						type="button"
						onclick={() => (lens = 'all')}
						class="text-xs text-synapse-glow hover:underline"
					>
						Show all events
					</button>
				</div>
			{:else}
				<div class="space-y-1.5">
					{#each visibleRows as row, i (row.key + ':' + row.index)}
						{@const meta = eventTone(row.event.type)}
						{@const isSel = selectedEventKey === row.key}
						<button
							use:reveal={{ delay: Math.min(i * 24, 320), y: 8 }}
							onclick={() => selectRow(row.key)}
							class="w-full text-left rounded-xl border px-3 py-2.5 transition lift
								{isSel
									? 'bg-synapse/10 border-synapse/40 shadow-[0_0_12px_rgba(99,102,241,0.18)]'
									: 'border-subtle/20 hover:border-synapse/30 hover:bg-white/[0.02]'}"
						>
							<div class="flex items-center gap-2.5">
								<span class="h-2 w-2 shrink-0 rounded-full" style="background: {TONE_DOT[meta]}"></span>
								<span class="text-sm font-medium text-text truncate">{eventLabel(row.event.type)}</span>
								<span class="ml-auto shrink-0 font-mono text-[10px] text-muted tabular-nums">{eventClock(row.event)}</span>
							</div>
							<div class="mt-1 truncate pl-[18px] font-mono text-[11px] text-dim">
								{payloadSummary(row.event.data)}
							</div>
						</button>
					{/each}
				</div>
			{/if}
		</div>

		<!-- (6) INTERPRETATION — selection detail panel -->
		<aside class="glass rounded-2xl p-4 space-y-3 max-h-[560px] overflow-y-auto">
			{#if selectedRow}
				{@const tone = eventTone(selectedRow.event.type)}
				<div class="flex items-start justify-between gap-2 border-b border-subtle/20 pb-3">
					<div class="min-w-0">
						<div class="flex items-center gap-2">
							<span class="h-2.5 w-2.5 shrink-0 rounded-full" style="background: {TONE_DOT[tone]}"></span>
							<h2 class="text-sm font-semibold text-bright truncate">{eventLabel(selectedRow.event.type)}</h2>
						</div>
						<div class="mt-1 font-mono text-[10px] uppercase tracking-wider text-muted">{selectedRow.event.type}</div>
					</div>
					<button
						type="button"
						onclick={() => (selectedEventKey = null)}
						class="rounded-lg border border-subtle/30 px-2 py-1 text-[11px] text-muted transition hover:text-text hover:border-synapse/30"
					>
						Close
					</button>
				</div>

				<div class="flex items-center justify-between text-[11px]">
					<span class="text-muted">Fired at</span>
					<span class="font-mono text-dim tabular-nums">{eventClock(selectedRow.event)}</span>
				</div>

				<div class="space-y-1.5">
					<div class="text-[10px] uppercase tracking-wider text-muted">Payload</div>
					{#if eventEntries(selectedRow.event).length === 0}
						<div class="text-[11px] text-muted">This event carries no payload fields.</div>
					{:else}
						{#each eventEntries(selectedRow.event) as entry (entry.k)}
							<div class="flex items-start gap-2 rounded-lg bg-white/[0.03] px-2.5 py-1.5">
								<span class="shrink-0 font-mono text-[10px] text-muted">{entry.k}</span>
								<span class="ml-auto break-all text-right font-mono text-[11px] text-text">{entry.v}</span>
							</div>
						{/each}
					{/if}
				</div>
			{:else}
				<!-- Idle interpretation hint — a one-line "what you're seeing" cue -->
				<div class="flex flex-col items-center gap-2.5 py-10 text-center">
					<div class="text-dim opacity-50">
						<Icon name="sparkle" size={30} strokeWidth={1.2} />
					</div>
					<div class="text-xs font-medium text-dim">Select an event</div>
					<p class="max-w-[220px] text-[11px] text-muted">
						Click any row to inspect its full payload here. Selecting an event highlights it in the field behind — it never changes your data.
					</p>
					{#if $heartbeat}
						<div class="mt-2 w-full rounded-lg bg-white/[0.03] px-3 py-2 text-left">
							<div class="text-[10px] uppercase tracking-wider text-muted">Last heartbeat</div>
							<div class="mt-1 font-mono text-[11px] text-dim">
								{payloadSummary($heartbeat.data)}
							</div>
						</div>
					{/if}
				</div>
			{/if}
		</aside>
	</div>
</div>
