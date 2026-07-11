<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type MemoryPr } from '$lib/stores/api';
	import { memoryPrEvents } from '$lib/stores/websocket';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { IMMUNE, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type MemoryPrTextItem = TextLayerItem & { prId?: string };
	type WhySignal = { code: string; detail: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const AMBER = [...rgb01(IMMUNE.caution), 0.9] satisfies [number, number, number, number];
	const MUTED = [...rgb01(RETENTION.recall), 0.62] satisfies [number, number, number, number];
	const ROW_LIMIT = 28;
	const PR_LIMIT = 100;
	// The MSDF reveal is frame-driven per glyph: the shared text layer packs
	// ageFrame = startFrame + globalGlyphIndex * 2. Across many long rows that
	// global index reaches the thousands, so late glyphs would only reveal after
	// ~7000 frames — the field renders near-black at rest. We anchor startFrame
	// far in the past so every glyph's ageFrame is already elapsed and the whole
	// queue is legible immediately (the shared layer is used by 15 organs, so the
	// fix stays here, in the item timing, not in the pass).
	const REVEAL_ANCHOR = -100000;

	let prs: MemoryPr[] = $state([]);
	let whySignals: WhySignal[] = $state([]);
	let loading = $state(true);
	let error: string | null = $state(null);

	onMount(() => {
		void loadPrs();
	});

	async function loadPrs() {
		loading = true;
		error = null;
		try {
			const res = await api.memoryPrs.list(undefined, PR_LIMIT);
			prs = res.prs;
			whySignals = [];
		} catch (err) {
			prs = [];
			whySignals = [];
			error = err instanceof Error ? err.message : 'UNKNOWN MEMORY PR FETCH ERROR';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		if ($memoryPrEvents.length) void loadPrs();
	});

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

	function numericField(value: unknown, keys: string[]): number | null {
		if (typeof value === 'number' && Number.isFinite(value)) return value;
		if (!value || typeof value !== 'object') return null;
		const record = value as Record<string, unknown>;
		for (const key of keys) {
			const candidate = record[key];
			if (typeof candidate === 'number' && Number.isFinite(candidate)) return candidate;
		}
		for (const candidate of Object.values(record)) {
			const nested = numericField(candidate, keys);
			if (nested !== null) return nested;
		}
		return null;
	}

	function confidenceDepth(pr: MemoryPr): number {
		const fromDiff = numericField(pr.diff, ['confidence', 'trust', 'contradictsTrust', 'contradicts_trust']);
		if (fromDiff !== null) return clamp01(fromDiff > 1 ? fromDiff / 100 : fromDiff);
		return 0.5;
	}

	function prLine(pr: MemoryPr): string {
		return sanitizeAscii(`${pr.title} | ${pr.id.slice(0, 8)} | ${pr.status}`)
			.replace(/\s+/g, ' ')
			.trim()
			.slice(0, 96);
	}

	function buildTextItems(): MemoryPrTextItem[] {
		const rows = prs.slice(0, ROW_LIMIT);
		const top = 0.74;
		const rowStep = 1.46 / Math.max(1, ROW_LIMIT - 1);
		const prItems = rows.map((pr, i) => ({
			id: `memory-pr:${pr.id}`,
			kind: 'memory-pr',
			prId: pr.id,
			text: prLine(pr),
			x: -0.9,
			y: top - i * rowStep,
			size: 0.025,
			color: pr.status === 'pending' ? CYAN : MUTED,
			depth: confidenceDepth(pr),
			weight: 1,
			startFrame: REVEAL_ANCHOR + i * 2,
			revealSpan: 20,
			maxWidthEm: 54,
				hitPadX: 0.03,
				hitPadY: 0.019
		})) satisfies MemoryPrTextItem[];

		const whyItems = whySignals.slice(0, 5).map((signal, i) => ({
			id: `memory-pr-why:${signal.code}:${i}`,
			kind: 'memory-pr-why',
			text: sanitizeAscii(`${signal.code}: ${signal.detail}`).replace(/\s+/g, ' ').trim().slice(0, 86),
			x: -0.82,
			y: -0.76 - i * 0.052,
			size: 0.02,
			color: AMBER,
			depth: 0.72,
			weight: 0.8,
			startFrame: REVEAL_ANCHOR + (rows.length + i) * 2,
			revealSpan: 18,
			maxWidthEm: 56
		})) satisfies MemoryPrTextItem[];

		return [...prItems, ...whyItems];
	}

	let memoryPrScene: RouteSceneModel = $derived({
		organ: 'memory-prs',
		nodes: prs.slice(0, ROW_LIMIT).map((pr, index) => ({
			source: { kind: 'pr', id: pr.id },
			index,
			label: prLine(pr),
			retention: 1,
			activation: confidenceDepth(pr),
			trust: confidenceDepth(pr),
			tags: pr.signals.map((signal) => sanitizeAscii(signal.code)),
			type: sanitizeAscii(pr.kind)
		})),
		edges: [],
		events: [],
		receipts: [],
		scalars: {
			visiblePrs: Math.min(prs.length, ROW_LIMIT),
			whySignals: whySignals.length
		},
		alive: prs.length > 0
	});

	function createMemoryPrPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		const field = new MemoryPrFieldPass(engine);
		field.uploadScene(scene);
		const text = new TextLayerPass(engine);
		void text.init().then(() => text.setText(buildTextItems()));
		return [field,
			{
				render: (pass) => text.render(pass),
				uploadScene: () => text.setText(buildTextItems()),
				pickAt: (x, y) => text.pickAt(x, y),
				dispose: () => text.dispose()
			}
		];
	}

	class MemoryPrFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) { this.field = new LivingFieldPass(engine); }
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node) => ({ id: node.source.id, score: node.activation ?? 0.5, hue: FIELD_HUE.caution, energy: node.activation, metric2: node.trust, scar: (node.tags?.length ?? 0) > 1, kind: 'memory-pr', payload: node }));
			this.field.setCells(layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.035, maxCellR: 0.09 }));
		}
		compute(encoder: GPUCommandEncoder): void { this.field.compute(encoder); }
		render(pass: GPURenderPassEncoder): void { this.field.render(pass); }
		pickAt(x: number, y: number): RoutePick | null { return this.field.pickAt(x, y); }
		dispose(): void { this.field.dispose(); }
	}

	async function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'memory-pr') return;
		// Pick can be a TEXT row (payload = MemoryPrTextItem with .prId) or a FIELD
		// cell (payload = RouteNode with .source.id == pr id). Read whichever, so
		// field cells act on the real PR, not silently no-op.
		const payload = pick.payload as Partial<MemoryPrTextItem> & { source?: { id?: string } };
		const prId = payload.prId ?? payload.source?.id;
		if (!prId) return;
		try {
			const res = (await api.memoryPrs.act(prId, 'ask_agent_why')) as { why?: WhySignal[] };
			whySignals = res.why ?? [];
		} catch (err) {
			error = err instanceof Error ? err.message : 'UNKNOWN MEMORY PR ACTION ERROR';
		}
	}
</script>

<svelte:head>
	<title>Memory PRs · Vestige</title>
</svelte:head>

<RouteStage
	organ="memory-prs"
	seed={`memory-pr-field:${prs.length}:${whySignals.length}`}
	scene={memoryPrScene}
	passes={createMemoryPrPasses}
	{loading}
	{error}
	emptyLabel="NO MEMORY PRS"
	onpick={handleRoutePick}
/>
