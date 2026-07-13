<script lang="ts">
	import { onMount } from 'svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { CAUSAL, IMMUNE, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { assertProvenance, type Provenance, type RouteNode, type RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { retentionColor } from '$lib/observatory/cognitive-palette';
	import { api } from '$stores/api';
	import type { DreamInsight, DreamResult, Memory } from '$types';

	type DreamTextItem = TextLayerItem & { insightIndex?: number };
	type DreamItemKind = 'dream-action' | 'dream-cycle' | 'dream-detail' | 'dream-insight' | 'dream-status';
	type DreamRecord = {
		id: string;
		kind: 'dream-cycle' | 'dream-insight';
		text: string;
		depth: number;
		weight: number;
		source: Provenance;
		insightIndex?: number;
	};
	type DreamScene = RouteSceneModel & {
		organ: 'dreams';
		records: DreamRecord[];
		raw: DreamResult | null;
		selectedRecordId: string | null;
		busy: boolean;
	};
	type DreamResultEnvelope = DreamResult & { message?: string };

	const CYAN = [...rgb01(CAUSAL.forward), 1] satisfies [number, number, number, number];
	const OXYGEN = [...rgb01(RETENTION.luciferin), 0.88] satisfies [number, number, number, number];
	const MUTED = [...rgb01(RETENTION.recall), 0.62] satisfies [number, number, number, number];
	const SCARLET = [...rgb01(IMMUNE.veto), 0.92] satisfies [number, number, number, number];
	const ROW_LIMIT = 36;
	const REVEAL_ANCHOR = -100000;
	const WRAP_COLUMNS = 64;

	let dreamResult: DreamResultEnvelope | null = $state(null);
	let loading = $state(false);
	let error: string | null = $state(null);
	let selectedRecordId: string | null = $state(null);
	let dormantPool: Memory[] = $state([]);
	// Real pool size (whole store, not just the sampled 80) — the dormant screen's
	// secondary stat: "N memories waiting to be replayed". Straight off /memories total.
	let dormantTotal = $state(0);

	const dreamScene = $derived.by<DreamScene>(() => normalizeDreamScene(dreamResult, selectedRecordId, loading));

	onMount(() => {
		// Seed the dormant field so the organ is ALIVE before you hit RUN — a slow,
		// dim pool of real memories waiting to be replayed. On a dream run this same
		// field storms bright with the insight cells.
		void api.memories
			.list({ limit: '80' })
			.then((res) => {
				dormantPool = res.memories;
				// /memories total is the returned page size, not the whole store — fall
				// back to it, but prefer the real store count from /stats below.
				dormantTotal = Number(res.total ?? res.memories.length) || res.memories.length;
				fieldPass?.setCells(buildFieldCells(fieldEngine));
			})
			.catch(() => {});
		// Real store size for the dormant-pool line ("N memories waiting to be replayed").
		void api
			.stats()
			.then((s) => {
				if (Number.isFinite(s.totalMemories) && s.totalMemories > 0) dormantTotal = s.totalMemories;
			})
			.catch(() => {});
	});

	async function runDream() {
		if (loading) return;
		loading = true;
		error = null;
		try {
			dreamResult = await api.dream();
			fieldPass?.setCells(buildFieldCells(fieldEngine), { ambient: 0.5 });
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	let fieldPass: LivingFieldPass | null = null;
	// Cached so setCells calls after a dream run can re-derive the live viewport aspect.
	let fieldEngine: ObservatoryEngine | null = null;

	/**
	 * Dormant → storm. At rest: the memory pool as slow dim cells (the mind
	 * asleep). After a dream: the insight nodes light up bright (consolidation
	 * storm). Both map to REAL data (memories list / dream insights).
	 */
	function buildFieldCells(engine: ObservatoryEngine | null = null) {
		const insights = dreamResult?.insights ?? [];
		let data: FieldDatum[];
		if (insights.length > 0) {
			data = insights.slice(0, 120).map((ins, i) => {
				const confidence = clamp01(Number(ins.confidence ?? 0));
				const novelty = clamp01(Number(ins.noveltyScore ?? confidence));
				return {
					id: ins.sourceMemories?.[0] ?? `dream-insight:${i}`,
					score: 0.5 + 0.5 * novelty,
					hue: novelty > 0.6 ? FIELD_HUE.retrograde : FIELD_HUE.bridge,
					energy: 0.6 + 0.4 * confidence,
					metric2: confidence,
					kind: 'dream-insight',
					payload: ins
				} satisfies FieldDatum;
			});
		} else {
			data = dormantPool.map((m) => {
				const retention = clamp01(m.retentionStrength);
				return {
					id: m.id,
					score: 0.3 + 0.3 * retention,
					hue: retentionColor(retention),
					energy: 0.12 + 0.28 * retention, // dim: the mind asleep
					metric2: retention,
					kind: 'dream-dormant',
					payload: m
				} satisfies FieldDatum;
			});
		}
		const cells = layoutGalaxy(data, { maxRadius: 0.92, minCellR: 0.014, maxCellR: 0.052 });

		// Portrait dormant screen only. The galaxy disc renders as a wide-but-short
		// band in a tall viewport, and the field is authored dim (0.26) for the
		// desktop two-column read — so the phone shows a black void around one bit of
		// centred text. Two portrait-gated (aspect < 0.85) tweaks, both derived from
		// the LIVE viewport aspect so desktop stays byte-identical:
		//   1. stretch the dormant substrate to fill the whole tall screen, and
		//   2. drop a soft focal glow-pill behind the RUN control so the bracket-text
		//      reads as a lit, tappable button. Both use only real dormant cells.
		if (dreamResult || dormantPool.length === 0) return cells;
		const aspect = viewportAspect(engine);
		if (aspect >= 0.85) return cells;
		// The cell shader maps NDC y -> clip y 1:1 (no aspect divide), so a disc of
		// radius 0.92 already spans the height; nudge the vertical spread wider so the
		// substrate reaches the top and bottom thirds instead of hugging the middle.
		for (const c of cells) {
			c.y = Math.max(-0.98, Math.min(0.98, c.y * 1.35));
		}
		// The button chrome (a framed pill) is drawn in the TEXT layer instead of the
		// field — the text pass shares the label's exact vertical mapping, so a dashed
		// over/under-rule lines up perfectly, where a cross-layer field glow never did.
		return cells;
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

	function scalarSource(name: string, value: number): Provenance {
		return { kind: 'scalar', id: `dreams.${name}`, scalar: { name, value } };
	}

	function insightId(insight: DreamInsight, index: number): string {
		const source = insight.sourceMemories?.[0] ?? `${insight.type}:${index}`;
		return sanitizeAscii(`${source}:${index}`).slice(0, 96);
	}

	function normalizeDreamScene(
		result: DreamResultEnvelope | null,
		selection: string | null,
		busy: boolean
	): DreamScene {
		if (!result) {
			return {
				organ: 'dreams',
				nodes: [],
				edges: [],
				events: [],
				receipts: [],
				scalars: {},
				alive: true,
				records: [],
				raw: null,
				selectedRecordId: selection,
				busy
			};
		}

		const insights = Array.isArray(result.insights) ? result.insights : [];
		const memoriesReplayed = Number(result.memoriesReplayed ?? 0);
		const connectionsPersisted = Number(result.connectionsPersisted ?? 0);
		const generated = Number(result.stats?.insightsGenerated ?? insights.length);
		const durationMs = Number(result.stats?.durationMs ?? 0);
		const newConnectionsFound = Number(result.stats?.newConnectionsFound ?? connectionsPersisted);
		const strengthened = Number(result.stats?.memoriesStrengthened ?? 0);
		const compressed = Number(result.stats?.memoriesCompressed ?? 0);
		const summaryParts = [
			result.status,
			String(memoriesReplayed),
			String(connectionsPersisted),
			String(generated),
			String(durationMs)
		].map((part) => sanitizeAscii(part ?? ''));
		const records: DreamRecord[] = [
			{
				id: 'dreams:cycle',
				kind: 'dream-cycle',
				text: summaryParts.join(' | '),
				depth: clamp01(memoriesReplayed / 50),
				weight: clamp01(newConnectionsFound / Math.max(1, memoriesReplayed)),
				source: scalarSource('memoriesReplayed', memoriesReplayed)
			}
		];

		if (result.message && insights.length === 0) {
			records.push({
				id: 'dreams:message',
				kind: 'dream-cycle',
				text: sanitizeAscii(result.message),
				depth: clamp01(memoriesReplayed / 50),
				weight: clamp01(generated / Math.max(1, memoriesReplayed)),
				source: scalarSource('message', memoriesReplayed)
			});
		}

		insights.forEach((insight, index) => {
			const confidence = clamp01(Number(insight.confidence ?? 0));
			const strength = clamp01(Number(insight.noveltyScore ?? confidence));
			const id = insightId(insight, index);
			const sourceMemory = insight.sourceMemories?.[0] ?? id;
			records.push({
				id: `dreams:insight:${id}`,
				kind: 'dream-insight',
				text: `C ${Math.round(confidence * 100)} · N ${Math.round(strength * 100)} | ${sanitizeAscii(insight.insight ?? '')}`,
				depth: confidence,
				weight: strength,
				source: { kind: 'memory', id: sourceMemory },
				insightIndex: index
			});
		});

		const nodes: RouteNode[] = records.map((record, index) => ({
			source: record.source,
			index,
			label: record.text,
			retention: record.weight,
			trust: record.depth,
			tags: [record.kind],
			type: record.kind
		}));
		const scene: DreamScene = {
			organ: 'dreams',
			nodes,
			edges: [],
			events: records.map((record, index) => ({
				source: record.source,
				type: record.kind,
				targetIndex: index,
				frame: 18 + index * 10,
				energy: record.weight
			})),
			receipts: [],
			scalars: {
				memoriesReplayed,
				connectionsPersisted,
				newConnectionsFound,
				insightsGenerated: generated,
				memoriesStrengthened: strengthened,
				memoriesCompressed: compressed,
				durationMs
			},
			alive: true,
			records,
			raw: result,
			selectedRecordId: selection,
			busy
		};
		if (import.meta.env.DEV) assertProvenance(scene);
		return scene;
	}

	function wrapAscii(value: string, width = WRAP_COLUMNS): string[] {
		const words = sanitizeAscii(value).split(/\s+/).filter(Boolean);
		const lines: string[] = [];
		let line = '';
		for (const word of words) {
			if (!line) {
				line = word.slice(0, width);
				continue;
			}
			if (`${line} ${word}`.length <= width) line += ` ${word}`;
			else {
				lines.push(line);
				line = word.slice(0, width);
			}
		}
		if (line) lines.push(line);
		return lines.length > 0 ? lines : [''];
	}

	function textItem(
		id: string,
		kind: DreamItemKind,
		text: string,
		y: number,
		options: Partial<DreamTextItem> = {}
	): DreamTextItem {
		return {
			id,
			kind,
			text,
			x: -0.88,
			y,
			size: 0.024,
			color: MUTED,
			depth: 0.5,
			weight: 0.5,
			startFrame: REVEAL_ANCHOR,
			revealSpan: 20,
			maxWidthEm: 64,
			...options
		};
	}

	/**
	 * Live viewport aspect, derived from the engine params (canvas px) with a
	 * window fallback — the SAME source TextLayerPass.portraitAdapt reads. Nothing
	 * is hardcoded to a phone width; the portrait branch keys off aspect < 0.85
	 * exactly like the shared adapter, so the desktop (aspect >= 0.85) render is
	 * byte-identical.
	 */
	function viewportAspect(engine: ObservatoryEngine | null): number {
		let vw = engine?.params[6] ?? 0;
		let vh = engine?.params[7] ?? 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return 1;
		return vw / vh;
	}

	function buildDreamItems(scene: RouteSceneModel, engine: ObservatoryEngine | null = null): DreamTextItem[] {
		const dream = scene as DreamScene;
		const records = dream.records.slice(0, ROW_LIMIT);
		const items: DreamTextItem[] = [];
		// Portrait: the reading column jams every row into the top ~30% and leaves
		// the lower two-thirds an empty void, and the bare "[ RUN DREAM CYCLE ]"
		// reads as a heading, not a tappable control. On a phone (aspect < 0.85)
		// pull the dormant hero into ONE centred focal point with an explicit tap
		// affordance, so a first-timer both SEES the void filled and KNOWS the
		// primary action. Landscape/desktop keep the authored top-anchored layout.
		const portrait = viewportAspect(engine) < 0.85;

		if (portrait && !dream.raw) {
			// Single centred focal group, vertically balanced in the viewport instead of
			// stacked at the top. portraitAdapt preserves authored SCREEN-y, so these are
			// on-screen positions. A soft glow-pill (built in buildFieldCells at the SAME
			// screen y) sits behind the RUN label so the bracket-text reads as a lit,
			// tappable button. Above it: a title. Below it: real pool stats + what a
			// cycle does — so the phone shows readable content, not a void.
			let hy = 0.5;
			items.push(
				textItem('dreams:title', 'dream-detail', 'DREAM ENGINE', hy, {
					x: -0.62,
					size: 0.03,
					color: MUTED,
					weight: 0.7
				})
			);
			// Button chrome — a dashed over/under-rule framing the label into a pill so
			// the bracket-text unmistakably reads as a tappable control. Drawn in the
			// TEXT layer at the SAME x/size as the label (monospace ⇒ exact width match),
			// so alignment is guaranteed on any portrait viewport. Same size as the label
			// so the dash-count spans the same width; dimmer oxygen so it frames, not
			// competes. The rules share the run's kind so a tap on the frame also fires.
			const runLabel = dream.busy ? '[  DREAMING...  ]' : '[  RUN DREAM CYCLE  ]';
			const rule = '-'.repeat(runLabel.length);
			const runY = 0.16;
			items.push(
				textItem('dreams:run-top', 'dream-action', rule, runY + 0.075, {
					x: -0.62,
					size: 0.05,
					color: dream.busy ? MUTED : OXYGEN,
					weight: 0.6
				})
			);
			hy = runY;
			items.push(
				textItem('dreams:run', 'dream-action', runLabel, hy, {
					x: -0.62,
					size: 0.05,
					color: dream.busy ? MUTED : OXYGEN,
					weight: 0.95,
					depth: 1,
					hitPadX: 0.34,
					hitPadY: 0.11
				})
			);
			items.push(
				textItem('dreams:run-bot', 'dream-action', rule, runY - 0.03, {
					x: -0.62,
					size: 0.05,
					color: dream.busy ? MUTED : OXYGEN,
					weight: 0.6
				})
			);
			hy -= 0.11;
			items.push(
				textItem('dreams:run-tap', 'dream-detail', dream.busy ? 'consolidating memory...' : 'tap the panel above to begin a cycle', hy, {
					x: -0.62,
					color: MUTED,
					size: 0.026
				})
			);
			hy -= 0.09;
			// Real secondary content: the actual dormant pool size off /memories total.
			const poolLine =
				dormantTotal > 0
					? `${dormantTotal} memories in the pool, waiting to be replayed`
					: 'loading the dormant memory pool...';
			items.push(
				textItem('dreams:pool', 'dream-detail', poolLine, hy, {
					x: -0.62,
					color: CYAN,
					size: 0.024
				})
			);
			hy -= 0.06;
			items.push(
				textItem('dreams:run-note', 'dream-detail', 'a cycle replays them, strengthens the strong,', hy, {
					x: -0.62,
					color: MUTED,
					size: 0.022
				})
			);
			hy -= 0.05;
			items.push(
				textItem('dreams:run-note2', 'dream-detail', 'and persists the new connections it finds', hy, {
					x: -0.62,
					color: MUTED,
					size: 0.022
				})
			);
			return items;
		}

		let y = 0.76;
		items.push(
			textItem('dreams:run', 'dream-action', dream.busy ? '[ DREAMING... ]' : '[ RUN DREAM CYCLE ]', y, {
				size: 0.032,
				color: dream.busy ? MUTED : OXYGEN,
				hitPadX: 0.03,
				hitPadY: 0.05
			})
		);
		y -= 0.065;
		items.push(
			textItem('dreams:run-note', 'dream-detail', 'strengthens memories and persists connections', y, {
				color: MUTED,
				size: 0.02
			})
		);
		y -= 0.09;
		if (!dream.raw) {
			items.push(
				textItem('dreams:dormant', 'dream-status', 'DREAM ENGINE DORMANT - RUN EXPLICIT CYCLE', y, {
					color: MUTED,
					size: 0.028
				})
			);
			return items;
		}

		for (const record of records) {
			items.push({
			id: record.id,
			kind: record.kind,
			text: record.text,
			x: -0.88,
			y,
			size: record.kind === 'dream-cycle' ? 0.03 : 0.024,
			color: record.id === dream.selectedRecordId ? OXYGEN : record.kind === 'dream-cycle' ? MUTED : CYAN,
			depth: record.depth,
			weight: record.weight,
			startFrame: REVEAL_ANCHOR,
			revealSpan: 20,
			maxWidthEm: 52,
			hitPadX: 0.03,
			hitPadY: 0.014,
			insightIndex: record.insightIndex
			});
			y -= 0.052;
		}

		const selected = records.find((record) => record.id === dream.selectedRecordId);
		if (!selected) return items;
		y = 0.45;
		const inspectorX = 0.08;
		items.push(
			textItem('dreams:detail-heading', 'dream-detail', 'SELECTED DREAM RECEIPT', y, {
				x: inspectorX,
				color: OXYGEN,
				maxWidthEm: 34
			})
		);
		y -= 0.052;
		if (selected.kind === 'dream-insight' && selected.insightIndex !== undefined) {
			const insight = dream.raw.insights?.[selected.insightIndex];
			if (insight) {
				wrapAscii(insight.insight, 34).forEach((line, lineIndex) => {
					items.push(
						textItem(`dreams:detail:text:${lineIndex}`, 'dream-detail', line, y, {
							x: inspectorX,
							color: CYAN,
							maxWidthEm: 34
						})
					);
					y -= 0.043;
				});
				items.push(textItem('dreams:detail:type', 'dream-detail', `TYPE: ${sanitizeAscii(insight.type)}`, y, { x: inspectorX }));
				y -= 0.043;
				items.push(textItem('dreams:detail:confidence', 'dream-detail', `CONFIDENCE: ${Math.round(clamp01(insight.confidence) * 100)}`, y, { x: inspectorX }));
				y -= 0.043;
				items.push(textItem('dreams:detail:novelty', 'dream-detail', `NOVELTY: ${Math.round(clamp01(insight.noveltyScore) * 100)}`, y, { x: inspectorX }));
				y -= 0.043;
				items.push(textItem('dreams:detail:sources', 'dream-detail', `SOURCE COUNT: ${insight.sourceMemories?.length ?? 0}`, y, { x: inspectorX }));
				y -= 0.06;
			}
		} else {
			wrapAscii(selected.text, 34).forEach((line, lineIndex) => {
				items.push(textItem(`dreams:detail:cycle:${lineIndex}`, 'dream-detail', line, y, { x: inspectorX }));
				y -= 0.043;
			});
			y -= 0.017;
		}
		items.push(
			textItem('dreams:again', 'dream-action', dream.busy ? '[ DREAMING... ]' : '[ DREAM AGAIN ]', y, {
				x: inspectorX,
				color: dream.busy ? MUTED : OXYGEN,
				size: 0.03,
				hitPadX: 0.03,
				hitPadY: 0.05
			})
		);
		return items;
	}

	class DreamTextPass implements RouteFramePass {
		private text: TextLayerPass;
		private scene: RouteSceneModel;
		private engine: ObservatoryEngine;

		constructor(engine: ObservatoryEngine, scene: RouteSceneModel) {
			this.engine = engine;
			this.scene = scene;
			this.text = new TextLayerPass(engine);
			void this.text.init().then(() => this.text.setText(buildDreamItems(this.scene, this.engine)));
		}

		uploadScene(scene: RouteSceneModel): void {
			this.scene = scene;
			this.text.setText(buildDreamItems(scene, this.engine));
		}

		render(pass: GPURenderPassEncoder): void {
			this.text.render(pass);
		}

		pickAt(ndcX: number, ndcY: number): RoutePick | null {
			return this.text.pickAt(ndcX, ndcY);
		}

		dispose(): void {
			this.text.dispose();
			void this.engine;
		}
	}

	function createDreamPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		fieldEngine = engine;
		// Portrait dormant screen ran near-black: intensity 0.26 + a wide reading well
		// left the whole phone void unlit. Gate off the LIVE viewport aspect (< 0.85)
		// so the phone gets a visibly-lit dream substrate + a tight well around only the
		// centred focal group, while desktop (aspect >= 0.85) keeps the exact authored
		// dim two-column backdrop below. Nothing hardcoded to a pixel width.
		const portrait = viewportAspect(engine) < 0.85;
		if (portrait && !dreamResult) {
			// Fill the tall void with a visibly-lit dim substrate. NO reading well here:
			// the well multiplies every cell (incl. the focal glow-pill) and would kill
			// the button chrome. The bright MSDF text (ivory/cyan) sits ON TOP and stays
			// legible over a 0.42 substrate; the pill cells are marked selected so they
			// punch a clear lit panel behind the RUN label.
			field.setIntensity(0.42);
			field.setReadingWell({ x: 0, y: 0, hw: -1, hh: 0 });
		} else {
			// Desktop can carry a richer dormant/storm field because the reading well
			// protects the complete left rows and centre inspector. Keep this branch
			// aspect-gated away from portrait, whose verified 0.42 substrate is unchanged.
			field.setIntensity(portrait ? 0.26 : 0.72);
			field.setReadingWell({ x: -0.35, y: -0.05, hw: 0.72, hh: 0.9, floor: 0.06, soft: 0.22 });
		}
		field.setCells(buildFieldCells(engine));
		const fieldWrapper: RouteFramePass = {
			compute: (encoder) => field.compute(encoder),
			render: (pass) => field.render(pass),
			pickAt: (x, y) => field.pickAt(x, y),
			dispose: () => {
				field.dispose();
				if (fieldPass === field) fieldPass = null;
			}
		};
		return [fieldWrapper, new DreamTextPass(engine, scene)];
	}

	function handleRoutePick(pick: RoutePick) {
		if (pick.kind === 'dream-action') {
			if (!loading) void runDream();
			return;
		}
		if (pick.kind !== 'dream-cycle' && pick.kind !== 'dream-insight') return;
		selectedRecordId = pick.id;
	}
</script>

<svelte:head>
	<title>Dreams · Vestige</title>
</svelte:head>

<RouteStage
	organ="dreams"
	seed={`dreams:${dreamResult?.status ?? 'pending'}:${dreamScene.records.length}:${dreamScene.scalars.durationMs ?? 0}`}
	scene={dreamScene}
	passes={createDreamPasses}
	loading={loading}
	error={error}
	onpick={handleRoutePick}
/>
