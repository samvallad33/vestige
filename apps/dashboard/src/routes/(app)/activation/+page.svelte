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
	const TITLE = [...rgb01('#EAF3FF'), 0.96] satisfies [number, number, number, number];
	const HEADER = [...rgb01('#7FA0B8'), 0.72] satisfies [number, number, number, number];
	const ROW_LIMIT = 36;
	// Phone shows fewer result rows so each gets real breathing room.
	const ROW_LIMIT_PORTRAIT = 16;
	const SEARCH_LIMIT = 40;
	// Pre-revealed anchor: the shared reveal ages every glyph by a GLOBAL index, so a
	// long list ages past the ~720-frame wrapped clock and all but the first rows
	// never appear. Anchor deeply negative so every row is revealed on frame 0.
	const REVEAL_ANCHOR = -100000;

	let engineRef: ObservatoryEngine | null = null;

	// Portrait / narrow viewport from the LIVE engine aspect (params[6]/[7]) with a
	// window fallback — the same signal TextLayerPass.portraitAdapt uses. Nothing is
	// hardcoded to a phone width; desktop (aspect>=0.85) is untouched.
	function isPortrait(): boolean {
		let vw = engineRef?.params[6] || 0;
		let vh = engineRef?.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return false;
		return vw / vh < 0.85;
	}

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

	function activationLine(memory: Memory, portrait: boolean): string {
		if (portrait) {
			// Phone: drop the id column and keep the primary content + the ONE headline
			// metric (relevance %). 'title  92%' reads at a glance; full detail on tap.
			const snippet = sanitizeAscii(memory.content).replace(/\s+/g, ' ').trim().slice(0, 34);
			return sanitizeAscii(`${snippet}  ${Math.round(relevance(memory) * 100)}%`);
		}
		const snippet = sanitizeAscii(memory.content).replace(/\s+/g, ' ').trim().slice(0, 54);
		const score = Math.round(relevance(memory) * 100);
		return sanitizeAscii(`${snippet} | ${memory.id.slice(0, 8)} | ${score}%`);
	}

	function buildTextItems(): ActivationTextItem[] {
		const portrait = isPortrait();
		const rowLimit = portrait ? ROW_LIMIT_PORTRAIT : ROW_LIMIT;
		const rows = results.slice(0, rowLimit);
		// Phone gets a title + column header and a shorter, generously spaced list;
		// desktop keeps the dense edge-to-edge result log exactly as authored.
		const top = portrait ? 0.6 : 0.72;
		const span = portrait ? 1.28 : 1.5;
		const rowStep = span / Math.max(1, rowLimit - 1);
		const items: ActivationTextItem[] = [];

		if (portrait) {
			items.push({
				id: 'activation:title',
				kind: 'activation-title',
				text: 'ACTIVATION',
				x: -0.62,
				y: 0.86,
				size: 0.06,
				color: TITLE,
				depth: 1,
				weight: 0.9,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 1
			});
			items.push({
				id: 'activation:header',
				kind: 'activation-header',
				text: `MEMORY / RELEVANCE  (${Math.min(results.length, rowLimit)})`,
				x: -0.62,
				y: 0.74,
				size: 0.03,
				color: HEADER,
				depth: 0.9,
				weight: 0.5,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 1
			});
		}

		rows.forEach((memory, i) => {
			items.push({
				id: `activation:${memory.id}`,
				kind: 'activation-result',
				memoryId: memory.id,
				text: activationLine(memory, portrait),
				x: portrait ? -0.62 : -0.88,
				y: top - i * rowStep,
				size: portrait ? 0.03 : 0.026,
				color: CYAN,
				depth: portrait ? Math.max(0.7, relevance(memory)) : relevance(memory),
				weight: retentionOrScore(memory),
				startFrame: portrait ? REVEAL_ANCHOR + i * 2 : i * 2,
				revealSpan: portrait ? 1 : 20,
				maxWidthEm: 46,
				hitPadX: 0.03,
				hitPadY: 0.015
			});
		});
		return items;
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
			// Landscape/desktop: the field is the hero (bright spread). Portrait: the
			// same 0.75 field becomes a blinding wall of blobs the result rows can't be
			// read over, so DIM it and open a reading well behind the label column so
			// the text sits on a calm backdrop. Both derive from the live aspect only.
			if (isPortrait()) {
				this.field.setIntensity(0.24);
				this.field.setReadingWell({ x: -0.3, y: -0.03, hw: 0.7, hh: 0.9, floor: 0.08, soft: 0.25 });
			} else {
				this.field.setIntensity(0.75);
			}
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
		engineRef = engine;
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
