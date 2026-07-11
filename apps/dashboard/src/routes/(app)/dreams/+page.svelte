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

	const dreamScene = $derived.by<DreamScene>(() => normalizeDreamScene(dreamResult, selectedRecordId, loading));

	onMount(() => {
		// Seed the dormant field so the organ is ALIVE before you hit RUN — a slow,
		// dim pool of real memories waiting to be replayed. On a dream run this same
		// field storms bright with the insight cells.
		void api.memories
			.list({ limit: '80' })
			.then((res) => {
				dormantPool = res.memories;
				fieldPass?.setCells(buildFieldCells());
			})
			.catch(() => {});
	});

	async function runDream() {
		if (loading) return;
		loading = true;
		error = null;
		try {
			dreamResult = await api.dream();
			fieldPass?.setCells(buildFieldCells(), { ambient: 0.5 });
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	let fieldPass: LivingFieldPass | null = null;

	/**
	 * Dormant → storm. At rest: the memory pool as slow dim cells (the mind
	 * asleep). After a dream: the insight nodes light up bright (consolidation
	 * storm). Both map to REAL data (memories list / dream insights).
	 */
	function buildFieldCells() {
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
		return layoutGalaxy(data, { maxRadius: 0.92, minCellR: 0.014, maxCellR: 0.052 });
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

	function buildDreamItems(scene: RouteSceneModel): DreamTextItem[] {
		const dream = scene as DreamScene;
		const records = dream.records.slice(0, ROW_LIMIT);
		const items: DreamTextItem[] = [];
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
			void this.text.init().then(() => this.text.setText(buildDreamItems(this.scene)));
		}

		uploadScene(scene: RouteSceneModel): void {
			this.scene = scene;
			this.text.setText(buildDreamItems(scene));
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
		field.setCells(buildFieldCells());
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
