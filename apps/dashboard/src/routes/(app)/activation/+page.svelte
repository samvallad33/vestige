<script lang="ts">
	import { onMount } from 'svelte';
	import { get } from 'svelte/store';
	import { page } from '$app/stores';
	import { api } from '$stores/api';
	import type { Memory } from '$types';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01, retentionColor } from '$lib/observatory/cognitive-palette';
	import type { RouteSceneModel, RouteNode } from '$lib/observatory/route-scene';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type ActivationTextItem = TextLayerItem & { memoryId?: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const ROW_LIMIT = 36;
	const SEARCH_LIMIT = 40;

	let results = $state<Memory[]>([]);
	let total = $state(0);
	let durationMs = $state(0);
	let seedQuery = $state('');
	let loading = $state(true);
	let error: string | null = $state(null);

	onMount(() => {
		void loadActivationField();
	});

	async function resolveSearchQuery(): Promise<string> {
		const routeQuery = get(page).url.searchParams.get('q')?.trim();
		if (routeQuery) return routeQuery;
		const memories = await api.memories.list({ limit: '1' });
		const seed = memories.memories[0]?.content?.replace(/\s+/g, ' ').trim() ?? '';
		return seed.slice(0, 96);
	}

	async function loadActivationField() {
		loading = true;
		error = null;
		try {
			const q = await resolveSearchQuery();
			seedQuery = q;
			if (!q) {
				results = [];
				total = 0;
				durationMs = 0;
				return;
			}
			const res = await api.search(q, SEARCH_LIMIT);
			results = res.results;
			total = res.total;
			durationMs = res.durationMs;
		} catch (err) {
			results = [];
			total = 0;
			durationMs = 0;
			error = err instanceof Error ? err.message : 'UNKNOWN ACTIVATION FETCH ERROR';
		} finally {
			loading = false;
		}
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

	function relevance(memory: Memory): number {
		return clamp01(memory.combinedScore ?? memory.retrievalStrength ?? memory.retentionStrength ?? 0.5);
	}

	function retentionOrScore(memory: Memory): number {
		return clamp01(memory.retentionStrength ?? relevance(memory));
	}

	function activationLine(memory: Memory): string {
		const snippet = sanitizeAscii(memory.content).replace(/\s+/g, ' ').trim().slice(0, 54);
		const score = Math.round(relevance(memory) * 100);
		return sanitizeAscii(`${snippet} | ${memory.id.slice(0, 8)} | ${score}%`);
	}

	function buildTextItems(): ActivationTextItem[] {
		const rows = results.slice(0, ROW_LIMIT);
		const top = 0.72;
		const rowStep = 1.5 / Math.max(1, ROW_LIMIT - 1);
		return rows.map((memory, i) => ({
			id: `activation:${memory.id}`,
			kind: 'activation-result',
			memoryId: memory.id,
			text: activationLine(memory),
			x: -0.88,
			y: top - i * rowStep,
			size: 0.026,
			color: CYAN,
			depth: relevance(memory),
			weight: retentionOrScore(memory),
			startFrame: i * 2,
			revealSpan: 20,
			maxWidthEm: 46,
				hitPadX: 0.03,
				hitPadY: 0.015
		}));
	}

	let activationScene: RouteSceneModel = $derived({
		organ: 'activation',
		nodes: results.slice(0, ROW_LIMIT).map((memory, index) => ({
			source: { kind: 'memory', id: memory.id },
			index,
			label: sanitizeAscii(memory.content).replace(/\s+/g, ' ').trim().slice(0, 54),
			retention: retentionOrScore(memory),
			stability: memory.storageStrength,
			lastAccessed: memory.lastAccessedAt,
			activation: relevance(memory),
			trust: relevance(memory),
			tags: memory.tags ?? [],
			type: memory.nodeType
		})),
		edges: [],
		events: [],
		receipts: [],
		scalars: {
			total,
			durationMs,
			visibleResults: Math.min(results.length, ROW_LIMIT),
			queryLength: seedQuery.length
		},
		alive: results.length > 0
	});

	/**
	 * The search-result SPREAD: every returned memory becomes a living cell whose
	 * radius + glow = its search relevance, so the most-activated results are the
	 * bright motes pulled to the core while weak matches drift to the cold rim.
	 * Field renders BEHIND the MSDF result rows so the labels stay readable.
	 */
	class ActivationFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
		}
		uploadScene(scene: RouteSceneModel): void {
			const nodes = scene.nodes as RouteNode[];
			const data: FieldDatum[] = nodes.map((n) => {
				const activation = clamp01(n.activation ?? 0.5);
				const retention = clamp01(n.retention);
				return {
					id: n.source.id,
					score: activation,
					hue: retention > 0 ? retentionColor(retention) : FIELD_HUE.recall,
					energy: 0.4 + 0.6 * activation,
					metric2: retention,
					kind: 'activation-result',
					payload: n
				} satisfies FieldDatum;
			});
			this.field.setCells(layoutGalaxy(data, { maxRadius: 0.92, minCellR: 0.012, maxCellR: 0.05 }));
		}
		compute(encoder: GPUCommandEncoder): void {
			this.field.compute(encoder);
		}
		render(pass: GPURenderPassEncoder): void {
			this.field.render(pass);
		}
		pickAt(ndcX: number, ndcY: number): RoutePick | null {
			return this.field.pickAt(ndcX, ndcY);
		}
		dispose(): void {
			this.field.dispose();
		}
	}

	function createActivationPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		// Field FIRST (renders behind the MSDF labels), then the readable text rows.
		const field = new ActivationFieldPass(engine);
		field.uploadScene(scene);
		const text = new TextLayerPass(engine);
		void text.init().then(() => text.setText(buildTextItems()));
		const textPass: RouteFramePass = {
			render: (pass) => text.render(pass),
			uploadScene: () => text.setText(buildTextItems()),
			pickAt: (x, y) => text.pickAt(x, y),
			dispose: () => text.dispose()
		};
		return [field, textPass];
	}

	function pickedMemoryId(pick: RoutePick): string | null {
		// Text rows carry an ActivationTextItem (memoryId); field cells carry the
		// real RouteNode (source.id). Resolve either so a click on a bright mote OR
		// its label promotes the same memory.
		const payload = pick.payload as Partial<ActivationTextItem> & { source?: { id?: string } };
		return payload?.memoryId ?? payload?.source?.id ?? null;
	}

	async function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'activation-result') return;
		const memoryId = pickedMemoryId(pick);
		if (!memoryId) return;
		try {
			const promoted = await api.memories.promote(memoryId);
			// The promote endpoint returns a PARTIAL memory ({ id, promoted,
			// retentionStrength }) — no content/combinedScore/nodeType. Merging over
			// the existing record (not replacing it) keeps content defined so the
			// derived scene + MSDF rows never hit `undefined.replace(...)`.
			results = results.map((memory) =>
				memory.id === promoted.id ? { ...memory, ...promoted } : memory
			);
		} catch (err) {
			error = err instanceof Error ? err.message : 'UNKNOWN ACTIVATION PROMOTE ERROR';
		}
	}
</script>

<svelte:head>
	<title>Activation · Vestige</title>
</svelte:head>

<RouteStage
	organ="activation"
	seed={`activation-field:${seedQuery}:${total}:${durationMs}`}
	scene={activationScene}
	passes={createActivationPasses}
	{loading}
	{error}
	emptyLabel="NO SEARCH ACTIVATION RESULTS"
	onpick={handleRoutePick}
/>
