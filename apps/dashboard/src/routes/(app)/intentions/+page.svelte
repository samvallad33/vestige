<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { IntentionItem } from '$types';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { CAUSAL, IMMUNE, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type RichIntention = IntentionItem & {
		retention?: number;
		retentionStrength?: number;
		confidence?: number;
		reminder_count?: number;
		tags?: string[];
		related_memories?: string[];
	};
	type IntentionTextItem = TextLayerItem & { intentionId?: string };
	type PredictedIntent = { id: string; content: string; nodeType: string; predictedNeed: string; retention: number };

	const CYAN = [...rgb01(CAUSAL.forward), 1] satisfies [number, number, number, number];
	const LUCIFERIN = [...rgb01(RETENTION.luciferin), 0.88] satisfies [number, number, number, number];
	const SCARLET = [...rgb01(IMMUNE.veto), 0.92] satisfies [number, number, number, number];
	const MUTED = [...rgb01(RETENTION.recall), 0.62] satisfies [number, number, number, number];
	const AMBER = [...rgb01(IMMUNE.caution), 0.9] satisfies [number, number, number, number];
	const INTENTION_LIMIT = 36;
	const ACTIVE_FILTER = 'active';
	const ALL_FILTER = 'all';

	let intentions: RichIntention[] = $state([]);
	let predictions: PredictedIntent[] = $state([]);
	let total = $state(0);
	let filter = $state(ACTIVE_FILTER);
	let loading = $state(true);
	let error: string | null = $state(null);
	let selectedIntentionId: string | null = $state(null);
	let textPass: TextLayerPass | null = null;

	onMount(() => {
		void loadIntentions(ACTIVE_FILTER);
	});

	onDestroy(() => {
		textPass?.dispose();
		textPass = null;
	});

	async function loadIntentions(nextFilter = filter) {
		filter = nextFilter;
		loading = true;
		error = null;
		updateTextPass();
		try {
			const [res, predictionRes] = await Promise.all([api.intentions(nextFilter), api.predict()]);
			intentions = (res.intentions || []) as RichIntention[];
			predictions = (predictionRes.predictions ?? []) as PredictedIntent[];
			total = res.total ?? intentions.length;
		} catch (err) {
			intentions = [];
			predictions = [];
			total = 0;
			error = err instanceof Error ? err.message : 'UNKNOWN INTENTION FETCH ERROR';
		} finally {
			loading = false;
			updateTextPass();
		}
	}

	function createIntentionsPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		const field = new IntentionsFieldPass(engine);
		field.uploadScene(scene);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		void pass.init().then(() => updateTextPass());
		pass.setText(buildTextItems());
		return [field, pass as RouteFramePass];
	}

	class IntentionsFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) { this.field = new LivingFieldPass(engine); }
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node) => ({ id: node.source.id, score: node.activation ?? node.retention, hue: FIELD_HUE.forward, energy: node.activation, metric2: node.retention, selected: node.source.id === selectedIntentionId, kind: 'intention', payload: node }));
			// RouteStage now picks text chrome (front) before field cells (behind),
			// so the galaxy can fill without stealing the filter toggle's click.
			const sparse = data.length < 4;
			this.field.setCells(layoutGalaxy(data, { maxRadius: 0.82, minCellR: sparse ? 0.22 : 0.025, maxCellR: sparse ? 0.3 : 0.075 }));
		}
		compute(encoder: GPUCommandEncoder): void { this.field.compute(encoder); }
		render(pass: GPURenderPassEncoder): void { this.field.render(pass); }
		pickAt(x: number, y: number): RoutePick | null { return this.field.pickAt(x, y); }
		dispose(): void { this.field.dispose(); }
	}

	function updateTextPass() {
		textPass?.setText(buildTextItems());
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

	function metricFrom(value: unknown, fallback: number): number {
		return typeof value === 'number' ? clamp01(value) : fallback;
	}

	function statusWeight(intention: RichIntention): number {
		const direct = metricFrom(intention.retention ?? intention.retentionStrength, Number.NaN);
		if (Number.isFinite(direct)) return direct;
		const reminderCount = typeof intention.reminder_count === 'number' ? intention.reminder_count : 0;
		const relatedCount = Array.isArray(intention.related_memories) ? intention.related_memories.length : 0;
		return clamp01((reminderCount + relatedCount + 1) / 6);
	}

	function intentionDepth(intention: RichIntention): number {
		return metricFrom(intention.confidence, clamp01((intention.priority || 2) / 4));
	}

	function summarizeTrigger(intention: RichIntention): string {
		try {
			const data = JSON.parse(intention.trigger_data || '{}') as Record<string, unknown>;
			const candidate = data.condition ?? data.topic ?? data.at ?? data.in_minutes ?? data.inMinutes ?? data.codebase;
			return sanitizeAscii(String(candidate ?? intention.trigger_type)).replace(/\s+/g, ' ').slice(0, 26);
		} catch {
			return sanitizeAscii(intention.trigger_type).slice(0, 26);
		}
	}

	function intentionLine(intention: RichIntention): string {
		const content = sanitizeAscii(intention.content).replace(/\s+/g, ' ').trim().slice(0, 48);
		const status = sanitizeAscii(intention.status).slice(0, 12);
		const trigger = summarizeTrigger(intention);
		return sanitizeAscii(
			`${content} | ${intention.id.slice(0, 8)} | p${intention.priority} | ${status} | ${trigger}`
		);
	}

	function statusItem(text: string, color = MUTED): IntentionTextItem {
		return {
			id: 'intentions:status',
			kind: 'intention-status',
			text: sanitizeAscii(text),
			x: -0.58,
			y: 0.02,
			size: 0.044,
			color,
			depth: 0.78,
			weight: 0.62,
			revealSpan: 32,
			maxWidthEm: 50
		};
	}

	// A dedicated, explicit filter toggle (active <-> all). Selecting an intention
	// must NOT silently flip the global filter — that lives here as its own target.
	function filterToggleItem(): IntentionTextItem {
		return {
			id: 'intentions:filter',
			kind: 'intention-filter',
			text: filter === ACTIVE_FILTER ? '[ SHOWING: ACTIVE - CLICK FOR ALL ]' : '[ SHOWING: ALL - CLICK FOR ACTIVE ]',
			// Far top-left corner (radius ~1.3 from origin) — beyond the galaxy's
			// bounded reach (maxRadius 0.7 + maxCellR 0.2 ≈ 0.9), so no field cell can
			// overlap and steal this click (field is picked before text chrome).
			x: -0.94,
			y: 0.9,
			size: 0.024,
			color: AMBER,
			depth: 1,
			weight: 0.7,
			revealSpan: 12,
			maxWidthEm: 40,
			hitPadX: 0.03,
			hitPadY: 0.03
		};
	}

	function buildTextItems(): IntentionTextItem[] {
		if (loading) return [statusItem('LOADING INTENTION FIELD...', CYAN)];
		if (error) return [statusItem(`ERROR - ${error}`.slice(0, 72), SCARLET)];
		if (intentions.length === 0) return [filterToggleItem(), statusItem(`EMPTY ${filter.toUpperCase()} INTENTION FIELD`, MUTED)];

		const items: IntentionTextItem[] = [filterToggleItem()];
		const rows = intentions.slice(0, INTENTION_LIMIT);
		const top = 0.72;
		const rowStep = 1.5 / Math.max(1, INTENTION_LIMIT - 1);
		for (let i = 0; i < rows.length; i++) {
			const intention = rows[i];
			const id = `intent:${intention.id}`;
			const active = selectedIntentionId === intention.id;
			items.push({
				id,
				kind: 'intention',
				intentionId: intention.id,
				text: intentionLine(intention),
				x: -0.88,
				y: top - i * rowStep,
				size: active ? 0.031 : 0.026,
				color: active ? LUCIFERIN : CYAN,
				depth: active ? 1 : intentionDepth(intention),
				weight: statusWeight(intention),
				startFrame: i * 2,
				revealSpan: 20,
				maxWidthEm: 52,
				hitPadX: 0.03,
				hitPadY: 0.015
			});
		}
		return items;
	}

	const scene = $derived<RouteSceneModel>({
		organ: 'intentions',
		nodes: [
			...intentions.map((intention, index) => ({
			source: { kind: 'receipt' as const, id: intention.id },
			index,
			label: sanitizeAscii(intention.content).slice(0, 48),
			retention: statusWeight(intention),
			activation: intentionDepth(intention),
			trust: intentionDepth(intention),
			tags: intention.tags ?? [intention.status, intention.trigger_type].filter(Boolean),
			type: intention.trigger_type
			})),
			...predictions.map((prediction, offset) => ({
				source: { kind: 'memory' as const, id: prediction.id },
				index: intentions.length + offset,
				label: sanitizeAscii(prediction.content).slice(0, 48),
				retention: clamp01(prediction.retention),
				activation: prediction.predictedNeed === 'high' ? 1 : prediction.predictedNeed === 'medium' ? 0.65 : 0.35,
				trust: clamp01(prediction.retention),
				tags: [prediction.predictedNeed],
				type: prediction.nodeType
			}))
		],
		edges: [],
		events: [],
		receipts: intentions.map((intention, index) => ({
			source: { kind: 'receipt', id: intention.id },
			label: sanitizeAscii(intention.status),
			nodeIndices: [index]
		})),
		scalars: {
			total,
			visible: intentions.length,
			predicted: predictions.length,
			filter: filter === ALL_FILTER ? 1 : 0
		},
		alive: intentions.length > 0
	});

	function handlePick(pick: RoutePick) {
		// The explicit filter toggle switches active <-> all (its own target).
		if (pick.kind === 'intention-filter') {
			void loadIntentions(filter === ACTIVE_FILTER ? ALL_FILTER : ACTIVE_FILTER);
			return;
		}
		if (pick.kind !== 'intention') return;
		// The pick can come from the TEXT pass (payload = IntentionTextItem with
		// .intentionId) OR the FIELD pass (payload = RouteNode with .source.id).
		// Read the intention id from whichever shape it is, so field cells are
		// clickable, not just text rows.
		const payload = pick.payload as Partial<IntentionTextItem> & { source?: { id?: string } };
		selectedIntentionId = payload.intentionId ?? payload.source?.id ?? null;
		updateTextPass();
	}
</script>

<svelte:head>
	<title>Intentions · Vestige</title>
</svelte:head>

<RouteStage
	organ="intentions"
	seed={`real-intention-field:${filter}:${total}:${selectedIntentionId ?? 'none'}`}
	{scene}
	passes={createIntentionsPasses}
	loading={loading}
	error={error}
	emptyLabel={`EMPTY ${filter.toUpperCase()} INTENTION FIELD`}
	onpick={handlePick}
/>
