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
	type PredictedIntent = {
		id: string;
		content: string;
		nodeType: string;
		predictedNeed: string;
		retention: number;
		urgency?: number;
	};

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
		void loadIntentions(ACTIVE_FILTER);
	});

	onDestroy(() => {
		textPass?.dispose();
		textPass = null;
		engineRef = null;
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
		engineRef = engine;
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
		private desktop: boolean;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
			this.desktop = viewportAspect() >= 0.85;
			// Portrait keeps its verified dim backdrop exactly. Desktop can carry a much
			// richer field because the reading well protects the intention rows.
			this.field.setIntensity(this.desktop ? 1.6 : 0.24);
			// On desktop, leave more living field around the reading column while keeping
			// the complete row span inside a soft, low-luminance well.
			this.field.setReadingWell(
				this.desktop
					? { x: -0.2, y: 0.05, hw: 0.58, hh: 0.62, floor: 0.08, soft: 0.18 }
					: { x: -0.2, y: 0.05, hw: 0.85, hh: 0.92, floor: 0.08, soft: 0.25 }
			);
		}
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node) => ({ id: node.source.id, score: node.activation ?? node.retention, hue: FIELD_HUE.forward, energy: node.activation, metric2: node.retention, selected: node.source.id === selectedIntentionId, kind: 'intention', payload: node }));
			// RouteStage now picks text chrome (front) before field cells (behind),
			// so the galaxy can fill without stealing the filter toggle's click.
			const sparse = data.length < 4;
			this.field.setCells(
				layoutGalaxy(data, {
					maxRadius: this.desktop ? 0.9 : 0.82,
					minCellR: sparse ? (this.desktop ? 0.56 : 0.22) : 0.025,
					maxCellR: sparse ? (this.desktop ? 0.7 : 0.3) : 0.075
				})
			);
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

	function intentionLine(intention: RichIntention, portrait = false): string {
		// Portrait: drop the id + trigger columns and shorten the snippet so the row
		// fits the narrow width on one readable line — never edge-to-edge, never
		// truncated mid-word. Desktop keeps the full, byte-identical row.
		if (portrait) {
			// Word-boundary trim so the row never ends mid-token. Cap sized so the
			// snippet + the trailing "  pN status" tag still fits the narrow portrait
			// width on one line (verified live at 360 and 390).
			const content = trimSnippet(intention.content, 34);
			const status = sanitizeAscii(intention.status).slice(0, 8);
			return sanitizeAscii(`${content}  p${intention.priority} ${status}`);
		}
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
	function filterToggleItem(portrait = false): IntentionTextItem {
		if (portrait) {
			// Portrait: a readable, centred heading pinned to the top band. It owns its
			// OWN mobile layout (portraitAdapt treats intention-filter as body text, so
			// author in the reclaimed-y space; the near-centre x keeps it on-screen after
			// the size boost). Drop the "SHOWING:" prefix — the shorter label leaves a
			// real right-edge margin at 360 so the closing bracket never clips.
			const portraitText =
				filter === ACTIVE_FILTER ? '[ ACTIVE - TAP FOR ALL ]' : '[ ALL - TAP FOR ACTIVE ]';
			return {
				id: 'intentions:filter',
				kind: 'intention-filter',
				text: portraitText,
				x: -0.5,
				y: 0.86,
				size: 0.03,
				color: AMBER,
				depth: 1,
				weight: 0.7,
				revealSpan: 12,
				maxWidthEm: 34,
				hitPadX: 0.06,
				hitPadY: 0.04
			};
		}
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

		const portrait = viewportAspect() < 0.85;

		if (portrait) {
			// Phone plan: ONE focal heading + a short, well-spaced column with real
			// negative space. Row count derives from the LIVE aspect (taller/narrower
			// -> fewer rows), never a fixed phone number. When the field is empty or
			// tiny, a content-first focal message tells a first-timer what they're
			// looking at instead of leaving a black void.
			// Instant reveal on portrait: the MSDF reveal gate bumps ageFrame by
			// (globalGlyphIndex*2) and discards glyphs whose ageFrame exceeds the frame
			// loop, so a per-row startFrame leaves long portrait rows half-drawn ("...is
			// li"). A large negative startFrame + revealSpan 1 saturates reveal to 1 on
			// frame 0 so EVERY row renders in full immediately. Same trick as schedule.
			const REVEAL = -100000;
			const items: IntentionTextItem[] = [filterToggleItem(true)];
			if (intentions.length === 0) {
				items.push({
					id: 'intentions:empty',
					kind: 'intention-status',
					text: 'NO ACTIVE INTENTIONS',
					x: -0.5,
					y: 0.18,
					size: 0.04,
					color: MUTED,
					depth: 0.9,
					weight: 0.6,
					startFrame: REVEAL,
					revealSpan: 1,
					maxWidthEm: 22
				});
				items.push({
					id: 'intentions:empty-sub',
					kind: 'intention-status',
					text: 'Vestige is watching for triggers. Tap the heading to see all.',
					x: -0.72,
					y: 0.02,
					size: 0.026,
					color: MUTED,
					depth: 0.78,
					weight: 0.5,
					startFrame: REVEAL,
					revealSpan: 1,
					maxWidthEm: 30
				});
				return items;
			}
			const aspect = viewportAspect();
			const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.42));
			const rowCount = Math.max(6, Math.round(10 - 4 * portraitness));
			const rows = intentions.slice(0, rowCount);
			// A count line so a single-row field reads as "1 of 1", not a lone stray.
			items.push({
				id: 'intentions:count',
				kind: 'intention-status',
				text: `${intentions.length} ACTIVE FOCUS${intentions.length === 1 ? '' : 'ES'}`,
				x: -0.62,
				y: 0.66,
				size: 0.026,
				color: MUTED,
				depth: 0.8,
				weight: 0.5,
				startFrame: REVEAL,
				revealSpan: 1,
				maxWidthEm: 28
			});
			const top = 0.5;
			const bottom = -0.7;
			const rowStep = rows.length > 1 ? (top - bottom) / (rows.length - 1) : 0;
			for (let i = 0; i < rows.length; i++) {
				const intention = rows[i];
				const active = selectedIntentionId === intention.id;
				items.push({
					id: `intent:${intention.id}`,
					kind: 'intention',
					intentionId: intention.id,
					text: intentionLine(intention, true),
					x: -0.82,
					y: top - i * rowStep,
					size: active ? 0.032 : 0.03,
					color: active ? LUCIFERIN : CYAN,
					depth: active ? 1 : Math.max(0.7, intentionDepth(intention)),
					weight: statusWeight(intention),
					startFrame: REVEAL,
					revealSpan: 1,
					maxWidthEm: 34,
					hitPadX: 0.05,
					hitPadY: 0.03
				});
			}
			return items;
		}

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
				// Real continuous urgency (FSRS decay + review schedule) drives brightness.
				// Fall back to the high/medium/low band only if an older backend omits it.
				activation:
					typeof prediction.urgency === 'number'
						? clamp01(prediction.urgency)
						: prediction.predictedNeed === 'high'
							? 1
							: prediction.predictedNeed === 'medium'
								? 0.65
								: 0.35,
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
