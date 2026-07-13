<script lang="ts">
	// ─────────────────────────────────────────────────────────────────────────
	// SETTINGS & SYSTEM — the zero-DOM WebGPU console.
	//
	// Every line on this page is breathing WebGPU: the ONLY non-WebGPU element is
	// the single <canvas> RouteStage mounts. There is no DOM control panel — the
	// title, live vitals, retention histogram, and action buttons are all MSDF
	// text runs laid into the shared cognitive field, and the buttons are PICKABLE
	// in-canvas regions that fire the real backend operations.
	//
	// Pattern mirrors stats/+page.svelte's StatsVitalsTextPass: a RouteFramePass
	// wraps a TextLayerPass, builds TextLayerItem[] from live data, and pickAt()
	// returns the run id under the cursor so onpick can switch on it and run the
	// real op (POST /api/consolidate, POST /api/dream, injectEvent, reload).
	// ─────────────────────────────────────────────────────────────────────────
	import { onMount } from 'svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { CAUSAL, IMMUNE, RETENTION, VITALS, rgb01 } from '$lib/observatory/cognitive-palette';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { api } from '$stores/api';
	import { websocket, isConnected } from '$stores/websocket';
	import type {
		ConsolidationResult,
		DreamResult,
		HealthCheck,
		RetentionDistribution,
		SystemStats
	} from '$types';

	// ── Palette (bright, full-alpha — dark text VANISHES into the MSDF field) ───
	const CYAN = [...rgb01(CAUSAL.forward), 1] satisfies [number, number, number, number];
	const FLOW = [...rgb01(VITALS.throughput), 0.98] satisfies [number, number, number, number];
	const LUCIFERIN = [...rgb01(RETENTION.luciferin), 1] satisfies [number, number, number, number];
	const RECALL = [...rgb01(RETENTION.recall), 0.98] satisfies [number, number, number, number];
	const AMBER = [...rgb01(IMMUNE.caution), 0.98] satisfies [number, number, number, number];
	const SCARLET = [...rgb01(IMMUNE.veto), 0.98] satisfies [number, number, number, number];

	// Pre-reveal EVERY glyph immediately, permanently. The MSDF reveal gate is
	// `reveal = clamp((params.frame - ageFrame) / revealSpan)` where
	// `ageFrame = startFrame + GLOBAL_glyphIndex*2` and `params.frame` is the
	// WRAPPED loop frame (0..~720). A text-heavy field (700+ glyphs) built with
	// startFrame:0 gives late glyphs ageFrame > 720, so they never clear the gate
	// and the field renders near-black. Anchoring startFrame at a large negative
	// pins ageFrame deeply negative for every glyph → reveal == 1 on frame 0 and
	// forever, so the console is fully lit at rest. (Verified live.)
	const PRE_REVEAL_FRAME = -100000;
	const REVEAL = 1;
	const READ_DEPTH = 0.95;

	// ── Live console state ──────────────────────────────────────────────────────
	let stats = $state<SystemStats | null>(null);
	// Real backend version + health, from GET /health (CARGO_PKG_VERSION). Never
	// hardcode the version — it must reflect the actual running binary.
	let health = $state<HealthCheck | null>(null);
	let retention = $state<RetentionDistribution | null>(null);
	let consolidation = $state<ConsolidationResult | null>(null);
	let dream = $state<DreamResult | null>(null);
	let busy = $state<null | 'consolidate' | 'dream' | 'refresh'>(null);
	let statusLine = $state('READY');
	let loading = $state(true);

	// The pass instance is created in the factory; keep a handle so state changes
	// re-upload the text (busy labels, fresh results) without rebuilding the pass.
	let consolePass: SettingsConsolePass | null = null;

	// A minimal alive scene — the field breathes; the console text is driven by
	// the pass directly from the live console state, not by scene primitives.
	const settingsScene = $derived.by<RouteSceneModel>(() => ({
		organ: 'settings',
		nodes: [],
		edges: [],
		events: [],
		receipts: [],
		scalars: {
			memories: stats?.totalMemories ?? 0,
			retention: stats?.averageRetention ?? 0,
			coverage: stats?.embeddingCoverage ?? 0
		},
		alive: true
	}));

	let consoleField: LivingFieldPass | null = null;

	// Re-render the console text whenever any live state changes.
	$effect(() => {
		// Touch every reactive input so the effect re-runs on each change.
		void [stats, retention, consolidation, dream, busy, $isConnected, statusLine, loading];
		consolePass?.refresh();
		consoleField?.setCells(buildConsoleCells());
	});

	// The retention distribution as a living field behind the console: each
	// histogram bucket becomes a cluster of cells (count = density), hue by
	// retention band. The console breathes on real vitals, not a black slab.
	function buildConsoleCells() {
		const dist = retention?.distribution ?? [];
		const data: FieldDatum[] = [];
		dist.forEach((bucket, i) => {
			const band = (i + 0.5) / Math.max(1, dist.length);
			// one cell per ~8 memories in the bucket (cap so a huge bucket stays sane)
			const cells = Math.min(60, Math.ceil(bucket.count / 8));
			for (let k = 0; k < cells; k++) {
				data.push({
					id: `settings:hist:${i}:${k}`,
					score: 0.25 + 0.7 * band,
					hue: band > 0.66 ? FIELD_HUE.oxygen : band > 0.33 ? FIELD_HUE.healthy : FIELD_HUE.debt,
					energy: 0.3 + 0.6 * band,
					metric2: band,
					scar: band < 0.2,
					kind: 'settings-tissue',
					payload: { band: `${i * 10}%`, count: bucket.count }
				});
			}
		});
		return layoutGalaxy(data, { maxRadius: 0.96, minCellR: 0.01, maxCellR: 0.04 });
	}

	onMount(() => {
		void loadData();
		// Rebuild the console when the viewport crosses the portrait threshold (rotate
		// / resize) so the stacked-vs-columns layout switches. rAF-debounced + bucketed
		// so a continuous drag doesn't thrash. buildConsoleItems reads the live aspect.
		let raf = 0;
		let lastPortrait = isPortrait();
		const onResize = () => {
			if (raf) return;
			raf = requestAnimationFrame(() => {
				raf = 0;
				const p = isPortrait();
				if (p !== lastPortrait) {
					lastPortrait = p;
					consolePass?.refresh();
					consoleField?.setCells(buildConsoleCells());
				}
			});
		};
		window.addEventListener('resize', onResize);
		window.addEventListener('orientationchange', onResize);
		return () => {
			if (raf) cancelAnimationFrame(raf);
			window.removeEventListener('resize', onResize);
			window.removeEventListener('orientationchange', onResize);
		};
	});

	async function loadData(): Promise<void> {
		loading = true;
		try {
			const [s, r, h] = await Promise.all([
				api.stats().catch(() => null),
				api.retentionDistribution().catch(() => null),
				api.health().catch(() => null)
			]);
			stats = s;
			retention = r;
			health = h;
		} finally {
			loading = false;
		}
	}

	async function handlePick(pick: RoutePick): Promise<void> {
		if (busy) return; // ignore picks while an op is running
		switch (pick.id) {
			case 'settings:action:consolidate':
				await runConsolidate();
				break;
			case 'settings:action:dream':
				await runDream();
				break;
			case 'settings:action:refresh':
				await runRefresh();
				break;
		}
	}

	async function runConsolidate(): Promise<void> {
		busy = 'consolidate';
		statusLine = 'CONSOLIDATING — FSRS-6 DECAY + MAINTENANCE...';
		dream = null;
		try {
			consolidation = await api.consolidate();
			await loadData();
			statusLine = 'CONSOLIDATION COMPLETE';
		} catch (err) {
			statusLine = `CONSOLIDATE FAILED - ${err instanceof Error ? err.message : String(err)}`.slice(0, 64);
		} finally {
			busy = null;
		}
	}

	async function runDream(): Promise<void> {
		busy = 'dream';
		statusLine = 'DREAMING — REPLAYING MEMORIES, FINDING CONNECTIONS...';
		consolidation = null;
		try {
			dream = await api.dream();
			await loadData();
			statusLine = 'DREAM CYCLE COMPLETE';
		} catch (err) {
			statusLine = `DREAM FAILED - ${err instanceof Error ? err.message : String(err)}`.slice(0, 64);
		} finally {
			busy = null;
		}
	}


	async function runRefresh(): Promise<void> {
		busy = 'refresh';
		statusLine = 'REFRESHING VITALS...';
		try {
			await loadData();
			statusLine = 'VITALS REFRESHED';
		} finally {
			busy = null;
		}
	}

	function createSettingsPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		const field = new LivingFieldPass(engine);
		consoleField = field;
		// Preserve the verified dim portrait treatment exactly. On desktop, enrich
		// the tissue outside a tighter reading well: the well keeps every MSDF row
		// crisp while the otherwise empty perimeter becomes a living backdrop.
		// Branch from the engine's live viewport rather than a device-width constant.
		let portraitBucket: boolean | null = null;
		const applyFieldTreatment = () => {
			const width = engine.params[6];
			const height = engine.params[7];
			if (width <= 0 || height <= 0) return;
			const portrait = width / height < 0.85;
			if (portrait === portraitBucket) return;
			portraitBucket = portrait;
			if (portrait) {
				field.setIntensity(0.18);
				field.setReadingWell({ x: -0.2, y: 0, hw: 0.85, hh: 0.92, floor: 0.06, soft: 0.22 });
			} else {
				field.setIntensity(1.0);
				field.setReadingWell({ x: -0.2, y: 0, hw: 0.62, hh: 0.72, floor: 0.12, soft: 0.14 });
			}
		};
		// Creation precedes the engine's first viewport uniform write, so start with
		// the portrait-safe values and resolve the live aspect in compute().
		field.setIntensity(0.18);
		field.setReadingWell({ x: -0.2, y: 0, hw: 0.85, hh: 0.92, floor: 0.06, soft: 0.22 });
		field.setCells(buildConsoleCells());
		const fieldWrapper: RouteFramePass = {
			compute: (encoder) => {
				applyFieldTreatment();
				field.compute(encoder);
			},
			render: (renderPass) => field.render(renderPass),
			dispose: () => {
				field.dispose();
				if (consoleField === field) consoleField = null;
			}
		};
		const pass = new SettingsConsolePass(engine);
		consolePass = pass;
		pass.uploadScene(scene);
		// Field FIRST (behind the console text).
		return [fieldWrapper, pass];
	}

	// ── The console pass: all MSDF, all pickable, zero DOM ──────────────────────
	class SettingsConsolePass implements RouteFramePass {
		private text: TextLayerPass;
		private initPromise: Promise<void> | null = null;
		private focused: string | null = null;

		constructor(engine: ObservatoryEngine) {
			this.text = new TextLayerPass(engine);
		}

		uploadScene(_scene: RouteSceneModel): void {
			void this.ensureReady().then(() => this.text.setText(buildConsoleItems()));
		}

		/** Re-lay the console text from the current live state (busy labels etc). */
		refresh(): void {
			void this.ensureReady().then(() => this.text.setText(buildConsoleItems()));
		}

		render(pass: GPURenderPassEncoder): void {
			this.text.render(pass);
		}

		pickAt(ndcX: number, ndcY: number): RoutePick | null {
			const hit = this.text.pickAt(ndcX, ndcY);
			// Only action runs are pickable targets; hover-highlight the focused one.
			const isAction = hit?.kind === 'settings-action';
			const next = isAction ? hit!.id : null;
			if (next !== this.focused) {
				this.focused = next;
				this.text.setRunDepth(next, 1);
			}
			return isAction ? hit : null;
		}

		dispose(): void {
			this.text.dispose();
			if (consolePass === this) consolePass = null;
		}

		private async ensureReady(): Promise<void> {
			if (!this.initPromise) this.initPromise = this.text.init();
			await this.initPromise;
		}
	}

	// Live viewport aspect (canvas px, window fallback) — the SAME source of truth
	// portraitAdapt uses. Portrait = narrow phone; drives a single stacked column so
	// the side-by-side desktop vitals/actions never overprint at ~390px.
	function isPortrait(): boolean {
		if (typeof window === 'undefined') return false;
		const vw = window.innerWidth;
		const vh = window.innerHeight;
		if (vw <= 0 || vh <= 0) return false;
		return vw / vh < 0.85;
	}

	// ── Layout: build the whole console as MSDF items ───────────────────────────
	function buildConsoleItems(): TextLayerItem[] {
		if (isPortrait()) return buildConsoleItemsPortrait();
		const items: TextLayerItem[] = [];

		// Masthead ----------------------------------------------------------------
		items.push(base('settings:title', 'settings-title', 'SETTINGS & SYSTEM', -0.9, 0.86, 0.052, LUCIFERIN));
		items.push(
			base(
				'settings:subtitle',
				'settings-sub',
				'TUNE THE COGNITIVE ENGINE. WATCH IT BREATHE. RUN THE RITUALS THAT KEEP MEMORY ALIVE.',
				-0.9,
				0.79,
				0.02,
				FLOW,
				64
			)
		);

		// Live vitals -------------------------------------------------------------
		const mem = stats?.totalMemories ?? 0;
		const ret = stats?.averageRetention ?? 0;
		const cov = stats?.embeddingCoverage ?? 0;
		const online = $isConnected;
		items.push(base('settings:v-mem-l', 'settings-vital', 'MEMORIES', -0.9, 0.66, 0.024, FLOW));
		items.push(base('settings:v-mem', 'settings-vital', mem.toLocaleString(), -0.9, 0.6, 0.05, CYAN));
		items.push(base('settings:v-ret-l', 'settings-vital', 'AVG RETENTION', -0.42, 0.66, 0.024, FLOW));
		items.push(
			base('settings:v-ret', 'settings-vital', `${(ret * 100).toFixed(1)}%`, -0.42, 0.6, 0.05, retentionColor(ret))
		);
		items.push(base('settings:v-ws-l', 'settings-vital', 'WEBSOCKET', 0.04, 0.66, 0.024, FLOW));
		items.push(
			base('settings:v-ws', 'settings-vital', online ? 'ONLINE' : 'OFFLINE', 0.04, 0.6, 0.05, online ? RECALL : SCARLET)
		);
		items.push(base('settings:v-ver-l', 'settings-vital', 'VESTIGE', 0.5, 0.66, 0.024, FLOW));
		items.push(base('settings:v-ver', 'settings-vital', `v${health?.version ?? '?'}`, 0.5, 0.6, 0.05, LUCIFERIN));
		items.push(
			base('settings:v-cov', 'settings-vital', `EMBEDDING COVERAGE ${cov.toFixed(0)}%`, -0.9, 0.53, 0.02, FLOW)
		);

		// Retention distribution histogram (ASCII bars) ---------------------------
		items.push(base('settings:hist-h', 'settings-hist', 'RETENTION DISTRIBUTION', -0.9, 0.44, 0.026, LUCIFERIN));
		const dist = retention?.distribution ?? [];
		const maxCount = Math.max(1, ...dist.map((b) => b.count));
		const top = 0.38;
		const step = 0.05;
		dist.forEach((bucket, i) => {
			const y = top - i * step;
			const bandLo = i * 10;
			const barLen = Math.round((bucket.count / maxCount) * 30);
			const bar = '#'.repeat(Math.max(bucket.count > 0 ? 1 : 0, barLen));
			const label = `${String(bandLo).padStart(3, ' ')}% ${bar}`;
			items.push(base(`settings:hist-${i}`, 'settings-hist', label, -0.9, y, 0.02, histColor(i), 60));
			items.push(
				base(`settings:hist-c-${i}`, 'settings-hist', bucket.count.toLocaleString(), 0.32, y, 0.02, histColor(i))
			);
		});

		// Action buttons (pickable) ----------------------------------------------
		const actionY = -0.5;
		items.push(
			actionItem(
				'settings:action:consolidate',
				busy === 'consolidate' ? '[ CONSOLIDATING... ]' : '[ CONSOLIDATE ]',
				-0.9,
				actionY,
				busy === 'consolidate' ? AMBER : CYAN
			)
		);
		items.push(
			actionItem(
				'settings:action:dream',
				busy === 'dream' ? '[ DREAMING... ]' : '[ DREAM ]',
				-0.36,
				actionY,
				busy === 'dream' ? AMBER : RECALL
			)
		);
		items.push(
			actionItem(
				'settings:action:refresh',
				busy === 'refresh' ? '[ REFRESHING... ]' : '[ REFRESH ]',
				0.2,
				actionY,
				busy === 'refresh' ? AMBER : FLOW
			)
		);

		// Status + operation results ----------------------------------------------
		items.push(base('settings:status', 'settings-status', `> ${statusLine}`, -0.9, -0.6, 0.024, LUCIFERIN, 72));

		if (consolidation) {
			const c = consolidation;
			const line = `PROCESSED ${c.nodesProcessed}   DECAYED ${c.decayApplied}   EMBEDDED ${c.embeddingsGenerated}   MERGED ${c.duplicatesMerged}   ${c.durationMs}MS`;
			items.push(base('settings:res-c', 'settings-result', line, -0.9, -0.68, 0.022, RECALL, 72));
		}
		if (dream) {
			const d = dream;
			const line = `REPLAYED ${d.memoriesReplayed}   CONNECTIONS ${d.connectionsPersisted}   INSIGHTS ${d.insights?.length ?? 0}`;
			items.push(base('settings:res-d', 'settings-result', line, -0.9, -0.68, 0.022, RECALL, 72));
			const first = d.insights?.[0];
			if (first) {
				items.push(base('settings:res-d1', 'settings-result', `~ ${first.insight}`, -0.9, -0.74, 0.02, FLOW, 76));
			}
		}

		// About / footer ----------------------------------------------------------
		items.push(
			base(
				'settings:about',
				'settings-about',
				'RUST + AXUM + SVELTEKIT 2 + SVELTE 5 + WEBGPU  |  FSRS-6  |  NOMIC EMBED v1.5 (256D)  |  USEARCH HNSW  |  LOCAL-FIRST, ZERO CLOUD',
				-0.9,
				-0.86,
				0.018,
				FLOW,
				88
			)
		);

		return items;
	}

	// ── Portrait console: ONE left-anchored vertical stack ─────────────────────
	// The desktop layout packs vitals into 4 side-by-side x-columns and the actions
	// into 3 — which overprint into garbage at phone width. In portrait we lay every
	// line on its OWN y in a single column at x=-0.9 (portraitAdapt reclaims the full
	// authored y-spread to the screen and caps size to fit width), so each vital,
	// histogram row, and action reads as its own line. Nothing here is a phone-width
	// magic number: the ONLY gate is the live aspect (isPortrait), and the y-spread is
	// the same authored ±0.86 the landscape layout uses.
	function buildConsoleItemsPortrait(): TextLayerItem[] {
		const items: TextLayerItem[] = [];
		const X = -0.9;
		let y = 0.86;
		const step = 0.066; // one line per row across the reclaimed portrait height
		const row = () => {
			const cur = y;
			y -= step;
			return cur;
		};

		// Masthead
		items.push(base('settings:title', 'settings-title', 'SETTINGS & SYSTEM', X, row(), 0.05, LUCIFERIN));
		items.push(
			base('settings:subtitle', 'settings-sub', 'TUNE THE COGNITIVE ENGINE. RUN THE RITUALS.', X, row(), 0.026, FLOW, 40)
		);
		y -= step * 0.4;

		// Vitals — each on its OWN line as "LABEL  value" so nothing overprints.
		const mem = stats?.totalMemories ?? 0;
		const ret = stats?.averageRetention ?? 0;
		const cov = stats?.embeddingCoverage ?? 0;
		const online = $isConnected;
		items.push(base('settings:v-mem', 'settings-vital', `MEMORIES  ${mem.toLocaleString()}`, X, row(), 0.03, CYAN, 40));
		items.push(
			base('settings:v-ret', 'settings-vital', `AVG RETENTION  ${(ret * 100).toFixed(1)}%`, X, row(), 0.03, retentionColor(ret), 40)
		);
		items.push(
			base('settings:v-ws', 'settings-vital', `WEBSOCKET  ${online ? 'ONLINE' : 'OFFLINE'}`, X, row(), 0.03, online ? RECALL : SCARLET, 40)
		);
		items.push(base('settings:v-ver', 'settings-vital', `VESTIGE  v${health?.version ?? '?'}`, X, row(), 0.03, LUCIFERIN, 40));
		items.push(base('settings:v-cov', 'settings-vital', `EMBEDDING COVERAGE  ${cov.toFixed(0)}%`, X, row(), 0.026, FLOW, 40));
		y -= step * 0.4;

		// Retention distribution — short bars (portrait is width-tight) with the count
		// on the SAME line so the bar can never overprint the value.
		items.push(base('settings:hist-h', 'settings-hist', 'RETENTION DISTRIBUTION', X, row(), 0.028, LUCIFERIN, 40));
		const dist = retention?.distribution ?? [];
		const maxCount = Math.max(1, ...dist.map((b) => b.count));
		dist.forEach((bucket, i) => {
			const bandLo = i * 10;
			const barLen = Math.round((bucket.count / maxCount) * 10); // short: fits phone width
			const bar = '#'.repeat(Math.max(bucket.count > 0 ? 1 : 0, barLen));
			const label = `${String(bandLo).padStart(3, ' ')}% ${bar.padEnd(10, ' ')} ${bucket.count.toLocaleString()}`;
			items.push(base(`settings:hist-${i}`, 'settings-hist', label, X, row(), 0.022, histColor(i), 40));
		});
		y -= step * 0.4;

		// Action buttons — STACKED so each is its own tappable target, not overlapped.
		items.push(
			actionItem(
				'settings:action:consolidate',
				busy === 'consolidate' ? '[ CONSOLIDATING... ]' : '[ CONSOLIDATE ]',
				X,
				row(),
				busy === 'consolidate' ? AMBER : CYAN
			)
		);
		items.push(
			actionItem('settings:action:dream', busy === 'dream' ? '[ DREAMING... ]' : '[ DREAM ]', X, row(), busy === 'dream' ? AMBER : RECALL)
		);
		items.push(
			actionItem('settings:action:refresh', busy === 'refresh' ? '[ REFRESHING... ]' : '[ REFRESH ]', X, row(), busy === 'refresh' ? AMBER : FLOW)
		);
		y -= step * 0.3;

		// Status + results
		items.push(base('settings:status', 'settings-status', `> ${statusLine}`, X, row(), 0.024, LUCIFERIN, 40));
		if (consolidation) {
			const c = consolidation;
			const line = `PROCESSED ${c.nodesProcessed}  DECAYED ${c.decayApplied}  MERGED ${c.duplicatesMerged}  ${c.durationMs}MS`;
			items.push(base('settings:res-c', 'settings-result', line, X, row(), 0.02, RECALL, 40));
		}
		if (dream) {
			const d = dream;
			const line = `REPLAYED ${d.memoriesReplayed}  CONNECTIONS ${d.connectionsPersisted}  INSIGHTS ${d.insights?.length ?? 0}`;
			items.push(base('settings:res-d', 'settings-result', line, X, row(), 0.02, RECALL, 40));
		}

		// Footer — readable size (the desktop 0.018 renders illegibly tiny on a phone).
		items.push(
			base(
				'settings:about',
				'settings-about',
				'RUST + AXUM + SVELTEKIT + WEBGPU  |  FSRS-6  |  NOMIC v1.5  |  LOCAL-FIRST',
				X,
				-0.9,
				0.02,
				FLOW,
				40
			)
		);

		return items;
	}

	function base(
		id: string,
		kind: string,
		text: string,
		x: number,
		y: number,
		size: number,
		color: [number, number, number, number],
		maxWidthEm?: number
	): TextLayerItem {
		return {
			id,
			kind,
			text: sanitizeAscii(text),
			x,
			y,
			size,
			color,
			depth: READ_DEPTH,
			weight: 0.5,
			startFrame: PRE_REVEAL_FRAME,
			revealSpan: REVEAL,
			maxWidthEm
		};
	}

	function actionItem(
		id: string,
		text: string,
		x: number,
		y: number,
		color: [number, number, number, number]
	): TextLayerItem {
		return {
			id,
			kind: 'settings-action',
			text: sanitizeAscii(text),
			x,
			y,
			size: 0.032,
			color,
			depth: 1,
			weight: 0.7,
			startFrame: PRE_REVEAL_FRAME,
			revealSpan: REVEAL,
			maxWidthEm: 40,
			// A bare glyph box (~1 line ≈ 14px) is far too thin to click reliably —
			// the whole "looks cool but can't click it" bug. Pad the hit target
			// generously (a comfortable button-sized area around the label) so a
			// normal cursor lands. Visual is unchanged; only the pick region grows.
			hitPadX: 0.03,
			hitPadY: 0.05
		};
	}

	function retentionColor(r: number): [number, number, number, number] {
		if (r > 0.7) return RECALL;
		if (r > 0.4) return AMBER;
		return SCARLET;
	}

	function histColor(bucketIndex: number): [number, number, number, number] {
		if (bucketIndex < 2) return SCARLET;
		if (bucketIndex < 4) return AMBER;
		if (bucketIndex < 7) return CYAN;
		return LUCIFERIN;
	}

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}
</script>

<RouteStage
	organ="settings"
	seed={`settings-console:${stats?.totalMemories ?? 0}:${(stats?.averageRetention ?? 0).toFixed(3)}:${$isConnected}`}
	scene={settingsScene}
	passes={createSettingsPasses}
	loading={loading}
	onpick={handlePick}
/>
