<script lang="ts">
	import { onMount } from 'svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { CAUSAL, IMMUNE, RETENTION, VITALS, rgb01 } from '$lib/observatory/cognitive-palette';
	import type { RouteReceipt, RouteSceneModel } from '$lib/observatory/route-scene';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { api } from '$stores/api';
	import type { ConsolidationResult, SystemStats } from '$types';

	type VitalReceipt = RouteReceipt & {
		metric: string;
		rawValue: unknown;
		magnitude: number;
	};

	const CYAN = [...rgb01(CAUSAL.forward), 1] satisfies [number, number, number, number];
	const FLOW = [...rgb01(VITALS.throughput), 0.96] satisfies [number, number, number, number];
	const LUCIFERIN = [...rgb01(RETENTION.luciferin), 0.9] satisfies [number, number, number, number];
	const AMBER = [...rgb01(IMMUNE.caution), 0.88] satisfies [number, number, number, number];
	const SCARLET = [...rgb01(IMMUNE.veto), 0.9] satisfies [number, number, number, number];

	let stats: SystemStats | null = $state(null);
	let loading = $state(true);
	let error: string | null = $state(null);
	let consolidation: ConsolidationResult | null = $state(null);

	const statsScene = $derived.by<RouteSceneModel>(() => {
		const receipts = stats ? buildReceipts(stats, consolidation) : [];
		const scalars = Object.fromEntries(receipts.map((r) => [r.metric, r.magnitude]));
		return {
			organ: 'stats',
			nodes: [],
			edges: [],
			events: [],
			receipts,
			scalars,
			alive: receipts.length > 0
		};
	});

	onMount(() => {
		void loadStats();
	});

	async function loadStats() {
		loading = true;
		error = null;
		try {
			stats = await api.stats();
		} catch (err) {
			stats = null;
			error = err instanceof Error ? err.message : String(err);
		} finally {
			loading = false;
		}
	}

	async function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'stats-vital') return;
		loading = true;
		error = null;
		try {
			consolidation = await api.consolidate();
			stats = await api.stats();
		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
		} finally {
			loading = false;
		}
	}

	function createStatsVitalsPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		// Field FIRST (renders behind), then text labels on top.
		const field = new StatsVitalsFieldPass(engine);
		field.uploadScene(scene);
		const text = new StatsVitalsTextPass(engine);
		text.uploadScene(scene);
		return [field, text];
	}

	/**
	 * Vitals as pulsing gauge-orbs: each real stat becomes a living cell whose
	 * radius + glow = its magnitude, hue by metric family. The whole set orbits
	 * as one galaxy so the vitals BREATHE instead of sitting as flat text.
	 */
	class StatsVitalsFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
			// Text-heavy organ: the field is a DIM backdrop, not the star.
			this.field.setIntensity(0.22);
			// Vitals labels run down the left column (x=-0.82, y from +0.68 to -0.68).
			// Suppress the field there so every metric row stays legible.
			this.field.setReadingWell({ x: -0.5, y: 0, hw: 0.6, hh: 0.85, floor: 0.08, soft: 0.25 });
		}
		uploadScene(scene: RouteSceneModel): void {
			const receipts = scene.receipts as VitalReceipt[];
			const data: FieldDatum[] = receipts.map((r) => ({
				id: `stats:${r.metric}`,
				score: r.magnitude,
				hue: vitalHue(r.metric, r.magnitude),
				energy: 0.4 + 0.6 * r.magnitude,
				scar: (r.metric.includes('due') || r.metric.includes('Retention')) && r.magnitude < 0.4,
				kind: 'stats-vital',
				payload: r
			}));
			// Fewer, bigger orbs than a memory galaxy — vitals are gauges, not motes.
			this.field.setCells(layoutGalaxy(data, { maxRadius: 0.86, minCellR: 0.03, maxCellR: 0.11 }));
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

	function vitalHue(metric: string, magnitude: number): [number, number, number] {
		if (metric.includes('due') || metric.includes('decay')) return magnitude > 0.4 ? FIELD_HUE.caution : FIELD_HUE.forward;
		if (metric.includes('Coverage') || metric.includes('Embeddings')) return magnitude > 0.7 ? FIELD_HUE.oxygen : FIELD_HUE.bridge;
		if (metric.includes('Retention') || metric.includes('Strength')) return magnitude < 0.4 ? FIELD_HUE.scarlet : FIELD_HUE.oxygen;
		return FIELD_HUE.recall;
	}

	class StatsVitalsTextPass implements RouteFramePass {
		private text: TextLayerPass;
		private initPromise: Promise<void> | null = null;
		private scene: RouteSceneModel | null = null;
		private focused: string | null = null;

		constructor(engine: ObservatoryEngine) {
			this.text = new TextLayerPass(engine);
		}

		uploadScene(scene: RouteSceneModel): void {
			this.scene = scene;
			void this.ensureReady().then(() => this.text.setText(this.buildItems(scene)));
		}

		render(pass: GPURenderPassEncoder): void {
			this.text.render(pass);
		}

		pickAt(ndcX: number, ndcY: number): RoutePick | null {
			const hit = this.text.pickAt(ndcX, ndcY);
			const next = hit?.id ?? null;
			if (next !== this.focused) {
				this.focused = next;
				this.text.setRunDepth(next, 1);
			}
			return hit;
		}

		dispose(): void {
			this.text.dispose();
			this.scene = null;
		}

		private async ensureReady(): Promise<void> {
			if (!this.initPromise) this.initPromise = this.text.init();
			await this.initPromise;
		}

		private buildItems(scene: RouteSceneModel): TextLayerItem[] {
			const receipts = scene.receipts as VitalReceipt[];
			if (receipts.length === 0) return [];
			const top = 0.68;
			const step = 1.36 / Math.max(1, receipts.length - 1);
			return receipts.map((receipt, i) => {
				const magnitude = clamp01(receipt.magnitude);
				return {
					id: `stats:${receipt.metric}`,
					kind: 'stats-vital',
					text: receipt.label,
					x: -0.82,
					y: top - i * step,
					size: i < 4 ? 0.036 : 0.027,
					color: vitalColor(receipt.metric, magnitude),
					// Depth drives brightness, but floor it so a low-magnitude vital
					// (e.g. an older newestMemory timestamp) never fades to unreadable —
					// every stat must be legible; magnitude still varies the glow above it.
					depth: Math.max(0.55, magnitude),
					weight: Math.max(0.18, Math.sqrt(magnitude)),
					startFrame: i * 2,
					revealSpan: 22,
					maxWidthEm: 48,
					hitPadX: 0.03,
					hitPadY: 0.03
				};
			});
		}
	}

	function buildReceipts(currentStats: SystemStats, currentConsolidation: ConsolidationResult | null): VitalReceipt[] {
		const entries = Object.entries(currentStats);
		const numericValues = entries
			.map(([, value]) => (typeof value === 'number' && Number.isFinite(value) ? Math.abs(value) : null))
			.filter((value): value is number => value !== null);
		const maxNumeric = Math.max(1, ...numericValues);
		const receipts = entries.map(([metric, rawValue], index) => {
			const magnitude = metricMagnitude(metric, rawValue, maxNumeric);
			return makeReceipt(metric, rawValue, magnitude, index);
		});
		if (currentConsolidation) {
			for (const [metric, rawValue] of Object.entries(currentConsolidation)) {
				const prefixed = `consolidate.${metric}`;
				receipts.push(makeReceipt(prefixed, rawValue, metricMagnitude(prefixed, rawValue, maxNumeric), receipts.length));
			}
		}
		return receipts;
	}

	function makeReceipt(metric: string, rawValue: unknown, magnitude: number, index: number): VitalReceipt {
		return {
			source: {
				kind: 'scalar',
				id: metric,
				scalar: { name: metric, value: magnitude }
			},
			label: sanitizeAscii(`${metric} | ${formatValue(metric, rawValue)}`),
			nodeIndices: [],
			metric,
			rawValue,
			magnitude: clamp01(magnitude)
		};
	}

	function metricMagnitude(metric: string, value: unknown, maxNumeric: number): number {
		if (typeof value === 'number' && Number.isFinite(value)) {
			if (metric.includes('Coverage')) return clamp01(value > 1 ? value / 100 : value);
			if (metric.includes('average') || metric.includes('Retention') || metric.includes('Strength')) return clamp01(value);
			return clamp01(Math.log10(Math.abs(value) + 1) / Math.log10(maxNumeric + 1));
		}
		if (typeof value === 'string') {
			const parsed = Date.parse(value);
			if (Number.isFinite(parsed)) {
				const days = Math.max(0, (Date.now() - parsed) / 86_400_000);
				return clamp01(1 / (1 + days / 30));
			}
			return clamp01(value.length / 48);
		}
		return 0.5;
	}

	function formatValue(metric: string, value: unknown): string {
		if (typeof value === 'number' && Number.isFinite(value)) {
			if (metric.includes('Coverage')) return `${value.toFixed(value > 1 ? 0 : 2)}%`;
			if (metric.includes('average')) return `${(value * 100).toFixed(1)}%`;
			if (Number.isInteger(value)) return value.toLocaleString();
			return value.toFixed(3);
		}
		if (typeof value === 'string') {
			// Compact an ISO timestamp to "YYYY-MM-DD HH:MM" so the full value fits on
			// one line (the raw ISO string with ms + timezone overruns and truncates).
			const parsed = Date.parse(value);
			if (Number.isFinite(parsed)) return value.slice(0, 16).replace('T', ' ');
			return value;
		}
		return String(value);
	}

	function vitalColor(metric: string, magnitude: number): [number, number, number, number] {
		if (metric.includes('due') || metric.includes('decay')) return magnitude > 0.4 ? AMBER : FLOW;
		if (metric.includes('Coverage') || metric.includes('Embeddings')) return magnitude > 0.7 ? LUCIFERIN : CYAN;
		if (metric.includes('Retention') || metric.includes('Strength')) return magnitude < 0.4 ? SCARLET : LUCIFERIN;
		return FLOW;
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
</script>

<RouteStage
	organ="stats"
	seed={`stats-vitals:${stats?.totalMemories ?? 0}:${stats?.dueForReview ?? 0}:${stats?.averageRetention ?? 0}`}
	scene={statsScene}
	passes={createStatsVitalsPasses}
	loading={loading}
	error={error}
	onpick={handleRoutePick}
/>
