<script lang="ts">
	import { websocket, eventFeed, isConnected, isReconnecting } from '$stores/websocket';
	import type { VestigeEvent } from '$types';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01 } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { eventImpulse01 } from '$lib/observatory/cognitive-palette';

	type FeedTextItem = TextLayerItem & { event?: VestigeEvent; eventKey?: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const AMBER = [...rgb01('#FFB000'), 0.9] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const MUTED = [...rgb01('#29F2A9'), 0.62] satisfies [number, number, number, number];
	const ROW_LIMIT = 42;

	let textPass: TextLayerPass | null = null;
	let focusedRun: string | null = null;
	// engine handle captured in the pass factory so buildTextItems can read the live
	// viewport aspect (params[6]/[7]) for portrait-only event-line shortening.
	let engineHandle: ObservatoryEngine | null = null;

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

	let feedScene: RouteSceneModel = $derived(buildFeedScene($eventFeed));

	$effect(() => {
		textPass?.setText(buildTextItems($eventFeed, $isConnected, $isReconnecting));
	});

	function createFeedPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		engineHandle = engine;
		const field = new FeedFieldPass(engine);
		field.uploadScene(scene);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		void pass.init().then(() => pass.setText(buildTextItems($eventFeed, $isConnected, $isReconnecting)));
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
				const event = $eventFeed[index];
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
			.replace(/[\u2014\u2013]/g, '-')
			.replace(/[\u2018\u2019]/g, "'")
			.replace(/[\u201C\u201D]/g, '"')
			.replace(/\u2026/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
	}

	function handleRoutePick(pick: RoutePick) {
		if (pick.kind === 'feed-event') websocket.clearEvents();
	}
</script>

<svelte:head>
	<title>Feed · Vestige</title>
</svelte:head>

<RouteStage
	organ="feed"
	seed={`live-event-stream:${$eventFeed.length}:${$isConnected ? 1 : 0}:${$isReconnecting ? 1 : 0}`}
	scene={feedScene}
	passes={createFeedPasses}
	loading={false}
	error={null}
	emptyLabel=""
	onpick={handleRoutePick}
/>
