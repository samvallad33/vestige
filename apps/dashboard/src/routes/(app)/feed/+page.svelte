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

	let feedScene: RouteSceneModel = $derived(buildFeedScene($eventFeed));

	$effect(() => {
		textPass?.setText(buildTextItems($eventFeed, $isConnected, $isReconnecting));
	});

	function createFeedPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
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
		}
		uploadScene(scene: RouteSceneModel): void {
			const events = scene.nodes.map((node, index) => ({ node, event: $eventFeed[index] })).filter((item) => item.event);
			const data: FieldDatum[] = events.map(({ node, event }) => ({
				id: node.source.id,
				score: node.retention,
				hue: eventImpulse01(event.type),
				energy: node.activation,
				metric2: node.trust,
				scar: event.type.includes('Deleted') || event.type.includes('Demoted') || event.type.includes('Verdict'),
				kind: 'feed-event',
				payload: event
			}));
			// A quiet live stream may contain only the real Connected event. Give
			// sparse event impulses a broad shockwave rather than inventing motes.
			this.field.setCells(layoutGalaxy(data, { maxRadius: 0.78, minCellR: 0.24, maxCellR: 0.32 }));
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

	function eventLine(event: VestigeEvent, key: string): string {
		const summary = payloadSummary(event.data);
		return sanitizeAscii(`${event.type} | ${key.slice(0, 24)} | ${summary}`.slice(0, 138));
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

	function connectionPayload(connected: boolean, reconnecting: boolean): string {
		return JSON.stringify({ connected, reconnecting, events: 0 });
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
	emptyLabel={connectionPayload($isConnected, $isReconnecting)}
	onpick={handleRoutePick}
/>
