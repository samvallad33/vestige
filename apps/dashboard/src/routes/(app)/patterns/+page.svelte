<script lang="ts">
	import { onMount } from 'svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import { api } from '$stores/api';
	import { rgb01 } from '$lib/observatory/cognitive-palette';
	import { assertProvenance, type RouteNode, type RouteSceneModel } from '$lib/observatory/route-scene';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutRings, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import type { CrossProjectCategory, CrossProjectPattern, CrossProjectPatternsResponse } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';

	type PatternLineItem = TextLayerItem & { patternKey?: string; category?: CrossProjectCategory };
	type PatternScene = RouteSceneModel & {
		organ: 'patterns';
		patterns: CrossProjectPattern[];
		projects: string[];
		maxTransferCount: number;
	};

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const AMBER = [...rgb01('#FFB020'), 0.88] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const ROW_LIMIT = 42;
	// The MSDF reveal is frame-driven per glyph: the shared text layer packs
	// ageFrame = startFrame + globalGlyphIndex * 2. Across many long rows that
	// global index reaches the thousands, so late glyphs would only reveal after
	// thousands of frames — the field renders near-black at rest. Anchor startFrame
	// far in the past so every glyph's ageFrame is already elapsed and the whole
	// queue is legible immediately (the shared layer is used by 15 organs, so the
	// fix stays here in the item timing, not in the protected pass).
	const REVEAL_ANCHOR = -100000;

	let data = $state<CrossProjectPatternsResponse>({ projects: [], patterns: [] });
	let loading = $state(true);
	let error: string | null = $state(null);
	let selectedCategory: CrossProjectCategory | null = $state(null);

	onMount(() => {
		void loadPatterns();
	});

	async function loadPatterns() {
		loading = true;
		error = null;
		try {
			data = await api.crossProjectPatterns();
		} catch (err) {
			data = { projects: [], patterns: [] };
			error = err instanceof Error ? err.message : String(err);
		} finally {
			loading = false;
		}
	}

	const visiblePatterns = $derived.by(() => {
		const patterns = selectedCategory
			? data.patterns.filter((pattern) => pattern.category === selectedCategory)
			: data.patterns;
		return [...patterns].sort((a, b) => b.transfer_count - a.transfer_count || b.confidence - a.confidence);
	});

	const patternScene = $derived.by<PatternScene>(() => normalizePatternScene(data.projects, visiblePatterns));

	function normalizePatternScene(projects: string[], patterns: CrossProjectPattern[]): PatternScene {
		const maxTransferCount = Math.max(1, ...patterns.map((pattern) => finite(pattern.transfer_count)));
		const nodes: RouteNode[] = patterns.slice(0, ROW_LIMIT).map((pattern, index) => {
			const strength = clamp01(finite(pattern.transfer_count) / maxTransferCount);
			const confidence = clamp01(pattern.confidence);
			return {
				source: { kind: 'pattern', id: patternKey(pattern) },
				index,
				label: patternLine(pattern),
				retention: confidence,
				activation: strength,
				trust: confidence,
				lastAccessed: pattern.last_used,
				tags: [pattern.category, pattern.origin_project, ...pattern.transferred_to],
				type: pattern.category
			};
		});

		const scene: PatternScene = {
			organ: 'patterns',
			nodes,
			edges: [],
			events: patterns.slice(0, ROW_LIMIT).map((pattern, index) => ({
				source: { kind: 'event', id: `patterns.${patternKey(pattern)}.${pattern.last_used}` },
				type: pattern.category,
				targetIndex: index,
				frame: 12 + index * 3,
				energy: clamp01(pattern.confidence)
			})),
			receipts: [],
			scalars: {
				projectCount: projects.length,
				patternCount: patterns.length,
				maxTransferCount,
				totalTransfers: patterns.reduce((sum, pattern) => sum + finite(pattern.transfer_count), 0)
			},
			alive: patterns.length > 0,
			patterns,
			projects,
			maxTransferCount
		};
		if (import.meta.env.DEV) assertProvenance(scene);
		return scene;
	}

	// Stable ring index per tracked category so each project-family (category) is
	// its own concentric ring. Order is fixed (not data-dependent) so the same
	// category always lands on the same ring across reloads/filters.
	const CATEGORY_RING: Record<CrossProjectCategory, number> = {
		ErrorHandling: 0,
		AsyncConcurrency: 1,
		Testing: 2,
		Architecture: 3,
		Performance: 4,
		Security: 5
	};

	function createPatternPasses(engine: ObservatoryEngine): RouteFramePass[] {
		// Field FIRST (renders behind), then MSDF text labels on top.
		const field = new PatternFieldPass(engine);
		const textPass = new TextLayerPass(engine);
		void textPass.init();
		return [
			field,
			{
				render: (pass) => textPass.render(pass),
				pickAt: (x, y) => textPass.pickAt(x, y),
				dispose: () => textPass.dispose(),
				uploadScene: (scene) => textPass.setText(buildTextItems(scene as PatternScene))
			}
		];
	}

	/**
	 * Cross-project patterns as a living field of concentric rings — one ring per
	 * category (project-family). Each cell is a REAL scene.node: radius/glow scale
	 * with transfer strength (activation), the oxygen membrane tints by retention
	 * (confidence), and high-transfer patterns burn CAUSAL.forward while the rest
	 * hold the bridge hue. The field breathes behind the readable MSDF rows.
	 */
	class PatternFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
		}
		uploadScene(scene: RouteSceneModel): void {
			const nodes = (scene as PatternScene).nodes;
			const data: FieldDatum[] = nodes.map((node) => {
				const strength = clamp01(finite(node.activation ?? 0));
				return {
					id: node.source.id,
					score: strength,
					hue: strength >= 0.5 ? FIELD_HUE.forward : FIELD_HUE.bridge,
					// Lift the glow floor so faint-transfer patterns still emit membrane
					// plasma and the whole field fills instead of pinpricking. Retention
					// (confidence) still drives the top of the range.
					energy: clamp01(0.5 + 0.5 * finite(node.retention ?? 0)),
					metric2: clamp01(finite(node.trust ?? 0)),
					kind: 'pattern',
					payload: node
				};
			});
			this.field.setCells(
				layoutRings(data, (_d, i) => ringOfNode(nodes[i]), {
					ringCount: 6,
					maxRadius: 0.92,
					minCellR: 0.055,
					maxCellR: 0.15
				})
			);
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

	function ringOfNode(node: RouteNode): number {
		const category = node.type as CrossProjectCategory;
		return CATEGORY_RING[category] ?? 0;
	}

	function buildTextItems(scene: PatternScene): PatternLineItem[] {
		const rows = scene.patterns.slice(0, ROW_LIMIT);
		const top = 0.74;
		const rowStep = 1.48 / Math.max(1, ROW_LIMIT - 1);
		return rows.map((pattern, index) => {
			const strength = clamp01(finite(pattern.transfer_count) / scene.maxTransferCount);
			const confidence = clamp01(pattern.confidence);
			// Depth drives z-layering AND the shader's idle wobble ((1-depth)*sin(time)).
			// Real cross-project data is often uniform (every transfer_count == 1 →
			// strength == 1), which would pin depth at 1.0 and freeze the field. Blend
			// strength with confidence and a gentle per-row phase so depth lives in the
			// animated 0.55..0.9 band (matching the sibling text organs) and the field
			// breathes at rest without faking any data channel.
			const rowPhase = 0.06 * Math.sin(index * 0.7);
			const depth = clamp01(0.6 + strength * 0.2 + (confidence - 0.5) * 0.4 + rowPhase);
			return {
				id: `pattern:${patternKey(pattern)}`,
				kind: 'pattern',
				patternKey: patternKey(pattern),
				category: pattern.category,
				text: patternLine(pattern),
				x: -0.88,
				y: top - index * rowStep,
				size: 0.024 + confidence * 0.005,
				color: selectedCategory && pattern.category === selectedCategory ? AMBER : CYAN,
				depth,
				weight: confidence,
				startFrame: REVEAL_ANCHOR + index * 2,
				revealSpan: 20,
				maxWidthEm: 58,
				hitPadX: 0.03,
				hitPadY: 0.013
			};
		});
	}

	function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'pattern') return;
		const item = pick.payload as PatternLineItem;
		selectedCategory = selectedCategory === item.category ? null : (item.category ?? null);
	}

	function patternLine(pattern: CrossProjectPattern): string {
		return sanitizeAscii(
			[
				pattern.name,
				pattern.origin_project,
				pattern.transferred_to.join(','),
				pattern.category,
				String(pattern.transfer_count),
				String(Math.round(clamp01(pattern.confidence) * 100)),
				pattern.last_used
			].join(' | ')
		).slice(0, 118);
	}

	function patternKey(pattern: CrossProjectPattern): string {
		return sanitizeAscii(
			[pattern.name, pattern.origin_project, pattern.category, pattern.last_used].join(':')
		).slice(0, 180);
	}

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[\u2014\u2013]/g, '-')
			.replace(/[\u2018\u2019]/g, "'")
			.replace(/[\u201C\u201D]/g, '"')
			.replace(/\u2026/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function finite(value: number): number {
		return Number.isFinite(value) ? value : 0;
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
	}
</script>

<svelte:head>
	<title>Patterns · Vestige</title>
</svelte:head>

<RouteStage
	organ="patterns"
	seed={`cross-project-patterns:${data.projects.length}:${visiblePatterns.length}:${selectedCategory ?? 'all'}`}
	scene={patternScene}
	passes={createPatternPasses}
	loading={loading}
	error={error}
	onpick={handleRoutePick}
/>
