<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import { api } from '$stores/api';
	import type { Memory } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01 } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { emptyScene, type RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutRings, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type ScheduleTextItem = TextLayerItem & { memoryId?: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const AMBER = [...rgb01('#FFB000'), 0.9] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const FETCH_LIMIT = 2000;
	const ROW_LIMIT = 40;
	// The list endpoint (/api/memories) omits FSRS review fields (nextReviewAt /
	// lastAccessedAt) — only the per-memory endpoint (/api/memories/:id) returns
	// them. We enrich a bounded window of the loaded records with their REAL
	// review timestamps so the schedule reflects genuine due-for-review data
	// instead of collapsing to an empty field. Verified: 200 parallel per-memory
	// GETs against the local brain take <1s.
	const ENRICH_LIMIT = 200;
	const ENRICH_CONCURRENCY = 16;

	let memories: Memory[] = $state([]);
	let loading = $state(true);
	let error: string | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;

	// Live viewport aspect (canvas px) — same signal TextLayerPass.portraitAdapt
	// reads (engine.params[6]/[7]), with a window fallback for the pre-frame-0
	// pass. NEVER a hardcoded phone width; desktop (aspect>=0.85) is untouched.
	function viewportAspect(): number {
		let vw = engineRef?.params[6] || 0;
		let vh = engineRef?.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return 1;
		return vw / vh;
	}

	// Trim to a cap on a word boundary so a portrait row never ends mid-token.
	function trimSnippet(text: string, cap: number): string {
		const s = sanitizeAscii(text).replace(/\s+/g, ' ').trim();
		if (s.length <= cap) return s;
		const hard = s.slice(0, cap);
		const lastSpace = hard.lastIndexOf(' ');
		return lastSpace > cap * 0.6 ? hard.slice(0, lastSpace) : hard;
	}

	onMount(() => {
		void loadSchedule();
	});

	onDestroy(() => {
		engineRef = null;
	});

	async function loadSchedule() {
		loading = true;
		error = null;
		try {
			const res = await api.memories.list({ limit: String(FETCH_LIMIT) });
			memories = await enrichReviewFields(res.memories);
		} catch (err) {
			memories = [];
			error = err instanceof Error ? err.message : 'API FETCH FAILED';
		} finally {
			loading = false;
		}
	}

	/**
	 * The list payload lacks nextReviewAt/lastAccessedAt. Backfill them from the
	 * per-memory endpoint (which does return real FSRS review timestamps) for a
	 * bounded window, in small concurrent batches. Records that already carry a
	 * nextReviewAt, or that fail to enrich, are passed through unchanged — a
	 * failed enrich just omits that row from the schedule, never fakes one.
	 */
	async function enrichReviewFields(source: Memory[]): Promise<Memory[]> {
		const enriched = source.slice();
		const targets = enriched
			.map((memory, index) => ({ memory, index }))
			.filter(({ memory }) => !memory.nextReviewAt)
			.slice(0, ENRICH_LIMIT);
		for (let i = 0; i < targets.length; i += ENRICH_CONCURRENCY) {
			const batch = targets.slice(i, i + ENRICH_CONCURRENCY);
			await Promise.all(
				batch.map(async ({ memory, index }) => {
					try {
						const full = await api.memories.get(memory.id);
						enriched[index] = { ...memory, ...full };
					} catch {
						// leave the un-enriched record; it simply won't schedule
					}
				})
			);
		}
		return enriched;
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

	function dueAt(memory: Memory): number {
		const parsed = memory.nextReviewAt ? Date.parse(memory.nextReviewAt) : Number.POSITIVE_INFINITY;
		return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
	}

	function urgency(memory: Memory, nowMs: number): number {
		const next = dueAt(memory);
		if (!Number.isFinite(next)) return 0;
		const days = (next - nowMs) / 86_400_000;
		return clamp01(1 - days / 30);
	}

	function scheduleLine(memory: Memory, nowMs: number, portrait = false): string {
		const next = dueAt(memory);
		const days = Number.isFinite(next) ? Math.ceil((next - nowMs) / 86_400_000) : 9999;
		const due = days < 0 ? `${Math.abs(days)}D OVER` : days === 0 ? 'DUE 0D' : `DUE ${days}D`;
		const pct = `${Math.round(memory.retentionStrength * 100)}%`;
		// Portrait: drop the id column and shorten the snippet so the row fits the
		// narrow width on one readable line — never edge-to-edge, never truncated
		// mid-word. Desktop keeps the full, byte-identical row.
		if (portrait) {
			return sanitizeAscii(`${trimSnippet(memory.content, 26)}  ${due} ${pct}`);
		}
		const snippet = sanitizeAscii(memory.content).replace(/\s+/g, ' ').trim().slice(0, 48);
		return sanitizeAscii(`${snippet} | ${memory.id.slice(0, 8)} | ${due} | ${pct}`);
	}

	function dueMemories(source: Memory[]): Memory[] {
		const nowMs = Date.now();
		return source
			.filter((memory) => !!memory.nextReviewAt)
			.slice()
			.sort((a, b) => {
				const dueDelta = dueAt(a) - dueAt(b);
				if (dueDelta !== 0) return dueDelta;
				return (a.retentionStrength ?? 0) - (b.retentionStrength ?? 0);
			})
			.sort((a, b) => urgency(b, nowMs) - urgency(a, nowMs));
	}

	const scheduled = $derived(dueMemories(memories));
	const scene = $derived< RouteSceneModel >(
		scheduled.length === 0
			? emptyScene('schedule')
			: {
					organ: 'schedule',
					nodes: scheduled.slice(0, ROW_LIMIT).map((memory, index) => ({
						source: { kind: 'memory', id: memory.id },
						index,
						label: scheduleLine(memory, Date.now()),
						retention: clamp01(memory.retentionStrength),
						stability: clamp01(memory.retrievalStrength),
						lastAccessed: memory.lastAccessedAt,
						activation: urgency(memory, Date.now()),
						tags: memory.tags,
						type: memory.nodeType
					})),
					edges: [],
					events: [],
					receipts: [],
					scalars: {
						scheduled: scheduled.length,
						loaded: memories.length,
						dueNow: scheduled.filter((memory) => urgency(memory, Date.now()) >= 1).length
					},
					alive: true
				}
	);

	const emptyLabel = $derived(
		memories.length === 0
			? '0 MEMORIES LOADED'
			: `${memories.length} MEMORIES / 0 REVIEW TIMESTAMPS`
	);

	function buildTextItems(routeScene: RouteSceneModel): ScheduleTextItem[] {
		const nodes = routeScene.nodes;
		const portrait = viewportAspect() < 0.85;

		if (portrait) {
			// Phone plan: ONE focal header + a short, well-spaced column with real
			// negative space instead of a 40-deep edge-to-edge wall. Row count derives
			// from the LIVE aspect (taller/narrower -> fewer rows), never a fixed phone
			// number. Rows carry a short, word-boundary-trimmed label so they never run
			// off the right edge or truncate mid-word. Desktop is untouched.
			const aspect = viewportAspect();
			const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.42));
			const rowCount = Math.max(9, Math.round(12 - 3 * portraitness));
			const rows = nodes.slice(0, rowCount);
			if (rows.length === 0) {
				return [
					{
						id: 'schedule:header',
						kind: 'schedule-header',
						text: 'REVIEW SCHEDULE',
						x: -0.6,
						y: 0.82,
						size: 0.032,
						color: CYAN,
						depth: 0.9,
						weight: 0.6,
						startFrame: -100000,
						revealSpan: 1,
						maxWidthEm: 26
					},
					{
						id: 'schedule:empty',
						kind: 'schedule-header',
						text: 'Nothing due for review',
						x: -0.62,
						y: 0.1,
						size: 0.03,
						color: AMBER,
						depth: 0.85,
						weight: 0.5,
						startFrame: -100000,
						revealSpan: 1,
						maxWidthEm: 28
					}
				];
			}
			const nowMs = Date.now();
			const memById = new Map(memories.map((memory) => [memory.id, memory]));
			const header: ScheduleTextItem = {
				id: 'schedule:header',
				kind: 'schedule-header',
				text: `REVIEW SCHEDULE  ${scheduled.length} DUE`,
				x: -0.72,
				y: 0.82,
				size: 0.03,
				color: CYAN,
				depth: 0.9,
				weight: 0.6,
				startFrame: -100000,
				revealSpan: 1,
				maxWidthEm: 30
			};
			const top = 0.6;
			const bottom = -0.72;
			const rowStep = rows.length > 1 ? (top - bottom) / (rows.length - 1) : 0;
			return [
				header,
				...rows.map((node, i) => {
					const memory = memById.get(node.source.id);
					const text = memory
						? scheduleLine(memory, nowMs, true)
						: sanitizeAscii(node.label).slice(0, 34);
					return {
						id: `schedule:${node.source.id}`,
						kind: 'schedule-memory',
						memoryId: node.source.id,
						text,
						x: -0.82,
						y: top - i * rowStep,
						size: 0.03,
						color: urgencyByNode(node.activation ?? 0, node.retention),
						depth: clamp01(0.7 + (node.activation ?? 0) * 0.3),
						weight: clamp01(node.retention),
						startFrame: -100000,
						revealSpan: 1,
						maxWidthEm: 34,
						hitPadX: 0.05,
						hitPadY: 0.03
					} satisfies ScheduleTextItem;
				})
			];
		}

		const top = 0.74;
		const rowStep = 1.52 / Math.max(1, ROW_LIMIT - 1);
		return nodes.slice(0, ROW_LIMIT).map((node, i) => ({
			id: `schedule:${node.source.id}`,
			kind: 'schedule-memory',
			memoryId: node.source.id,
			text: sanitizeAscii(node.label),
			x: -0.9,
			y: top - i * rowStep,
			size: 0.026,
			color: urgencyByNode(node.activation ?? 0, node.retention),
			// depth = trust/crispness channel (1.0 = crisp + forward + brighter).
			// It must stay high or the DOF blur + dim glow makes the small rows
			// invisible. Urgency biases it up so due-now rows read crispest;
			// retention is carried by `weight` (MSDF stroke mass), not depth.
			depth: clamp01(0.62 + (node.activation ?? 0) * 0.38),
			weight: clamp01(node.retention),
			// The MSDF reveal gate is `(params.frame - ageFrame)/revealSpan`, and
			// packGlyph bumps ageFrame by (globalGlyphIndex * 2). A 40-row schedule
			// packs ~2700 glyphs, so past the first few rows ageFrame exceeds the
			// 720-frame loop and those rows are discarded FOREVER. A large negative
			// startFrame drives every glyph's ageFrame far below 0, so reveal
			// saturates to 1 on frame 0 and EVERY due row renders immediately.
			startFrame: -100000,
			revealSpan: 1,
			maxWidthEm: 54,
				hitPadX: 0.03,
				hitPadY: 0.014
		}));
	}

	function urgencyByNode(urgencyDepth: number, retention: number): [number, number, number, number] {
		// On the real brain nearly every record is overdue (urgency saturates to 1),
		// so gating colour on urgency alone paints the whole field one flat scarlet.
		// Grade by retention instead: low-retention rows are the ones actually at
		// risk (scarlet), mid rows amber, healthy rows cyan — a readable, honest
		// heat map of what most needs review.
		if (retention < 0.4) return SCARLET;
		if (retention < 0.7) return AMBER;
		return CYAN;
	}

	class ScheduleTextPass implements RouteFramePass {
		private readonly text: TextLayerPass;
		private current: RouteSceneModel;

		constructor(engine: ObservatoryEngine, initialScene: RouteSceneModel) {
			this.current = initialScene;
			this.text = new TextLayerPass(engine);
			this.text.setText(buildTextItems(this.current));
			void this.text.init().then(() => this.text.setText(buildTextItems(this.current)));
		}

		uploadScene(nextScene: RouteSceneModel) {
			this.current = nextScene;
			this.text.setText(buildTextItems(this.current));
		}

		render(pass: GPURenderPassEncoder) {
			this.text.render(pass);
		}

		pickAt(ndcX: number, ndcY: number) {
			return this.text.pickAt(ndcX, ndcY);
		}

		dispose() {
			this.text.dispose();
		}
	}

	class ScheduleFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
			// TEXT-HEAVY organ: field is a DIM backdrop, never a blob that drowns
			// the 40-row due list. Intensity 0.22 keeps it a living whisper.
			this.field.setIntensity(0.22);
			// The schedule rows are anchored in a tall LEFT column: x from -0.9
			// extending right, y from +0.74 down to ~-0.78 (top=0.74, 40 rows @
			// rowStep). Carve a reading well over that column so the labels/values
			// stay crisp against the field.
			this.field.setReadingWell({ x: -0.35, y: -0.02, hw: 0.72, hh: 0.86, floor: 0.06, soft: 0.24 });
		}
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node) => ({
				id: node.source.id,
				score: node.activation ?? 0,
				hue: node.retention < 0.4 ? FIELD_HUE.scarlet : node.retention < 0.7 ? FIELD_HUE.caution : FIELD_HUE.oxygen,
				energy: node.activation,
				metric2: node.retention,
				scar: (node.activation ?? 0) >= 1,
				kind: 'schedule-memory',
				payload: node
			}));
			this.field.setCells(layoutRings(data, (datum) => datum.score >= 1 ? 0 : datum.score >= 0.75 ? 1 : datum.score >= 0.5 ? 2 : 3, { ringCount: 4, maxRadius: 0.88, minCellR: 0.045, maxCellR: 0.11 }));
		}
		compute(encoder: GPUCommandEncoder): void { this.field.compute(encoder); }
		render(pass: GPURenderPassEncoder): void { this.field.render(pass); }
		pickAt(x: number, y: number): RoutePick | null { return this.field.pickAt(x, y); }
		dispose(): void { this.field.dispose(); }
	}

	function createSchedulePasses(engine: ObservatoryEngine, initialScene: RouteSceneModel): RouteFramePass[] {
		engineRef = engine;
		const field = new ScheduleFieldPass(engine);
		field.uploadScene(initialScene);
		return [field, new ScheduleTextPass(engine, initialScene)];
	}

	async function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'schedule-memory') return;
		// TEXT row payload has .memoryId; FIELD cell payload is a RouteNode whose
		// source.id is the memory id. Read either so field cells promote, not no-op.
		const item = pick.payload as Partial<ScheduleTextItem> & { source?: { id?: string } };
		const memoryId = item.memoryId ?? item.source?.id;
		if (!memoryId) return;
		try {
			const promoted = await api.memories.promote(memoryId);
			// promote returns a PARTIAL {id, promoted, retentionStrength} — merge the
			// changed field into the full Memory, never replace it (a full swap drops
			// content/nodeType/... and crashes the next render — the /memories bug).
			memories = memories.map((memory) =>
				memory.id === promoted.id
					? { ...memory, retentionStrength: promoted.retentionStrength }
					: memory
			);
			error = null;
		} catch (err) {
			error = err instanceof Error ? err.message : 'PROMOTE FAILED';
		}
	}
</script>

<RouteStage
	organ="schedule"
	seed={`schedule-due-field:${scheduled.length}:${memories.length}`}
	{scene}
	passes={createSchedulePasses}
	{loading}
	{error}
	{emptyLabel}
	onpick={handleRoutePick}
/>
