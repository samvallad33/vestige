<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { replaceState } from '$app/navigation';
	import { api } from '$stores/api';
	import type { Memory, SearchResult } from '$types';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01 } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type SearchMemory = Memory & {
		similarity?: number;
		score?: number;
		relevance?: number;
		retention?: number;
	};
	type ExploreTextItem = TextLayerItem & { memoryId?: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const MUTED = [...rgb01('#29F2A9'), 0.62] satisfies [number, number, number, number];
	const SEARCH_LIMIT = 40;
	const ROW_LIMIT = 36;
	// The shared text pass packs reveal as ageFrame = startFrame + GLOBAL glyph
	// index * 2; 36 rows × ~77 glyphs ≈ 2,845 glyphs → ages to ~5,688 frames,
	// far past the ~720-frame wrapped clock, so late rows would NEVER reveal.
	// Anchor deeply negative (same fix as /memories, /patterns, /memory-prs).
	const REVEAL_ANCHOR = -100000;

	let hostEl: HTMLDivElement | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;
	let textPass: TextLayerPass | null = null;
	let fieldPass: LivingFieldPass | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;
	let results: SearchMemory[] = $state([]);
	let searchResult: SearchResult | null = $state(null);
	let searchQuery = $state('');
	let loading = $state(true);
	let error: string | null = $state(null);
	let activeRun: string | null = null;
	// Semantic walk (Wave 3): clicking a neighbor re-centers the whole neighborhood
	// on that thought, in place. walkingFrom carries the 8-char id of the thought
	// being walked from (shown in the loading status); seededFrom marks the newest-
	// memory auto-seed used when the page is opened with no ?q=.
	let walkingFrom: string | null = $state(null);
	let seededFrom: string | null = $state(null);
	let centerId: string | null = $state(null);
	// Generation token: two rapid walks race (B finishes, then stale A overwrites
	// B's neighborhood while the URL keeps B's query). Only the newest generation
	// may apply results / clear the walking label. (GPT-5.6-sol cross-review.)
	let loadGen = 0;

	// A memory id passed from another organ (e.g. Graph's "Explore Connections")
	// seeds the walk from THAT thought instead of re-seeding from the newest one,
	// so the selected memory survives the Graph → Explore handoff.
	let seedMemoryId: string | null = $state(null);

	onMount(() => {
		const params = new URLSearchParams(window.location.search);
		searchQuery = params.get('q') ?? '';
		// `memory` is the canonical cross-organ selection contract (center = legacy).
		seedMemoryId = params.get('memory') ?? params.get('center');
		void loadNeighbors();
	});

	onDestroy(() => {
		textPass?.dispose();
		fieldPass?.dispose();
		textPass = null;
		fieldPass = null;
		engineRef = null;
	});

	// Live viewport aspect (canvas px) — same source portraitAdapt reads, never a
	// hardcoded phone width. Falls back to the window before frame 0, then 1.
	function viewportAspect(): number {
		const vw = engineRef?.params[6] || 0;
		const vh = engineRef?.params[7] || 0;
		if (vw > 0 && vh > 0) return vw / vh;
		if (typeof window !== 'undefined' && window.innerHeight > 0) {
			return window.innerWidth / window.innerHeight;
		}
		return 1;
	}

	// Field brightness for the CURRENT viewport. On desktop the neighborhood is the
	// hero (0.75). On a phone the same intensity blooms into a blinding blob behind
	// the text and destroys contrast — so portrait drops it to the documented dim-
	// backdrop range (~0.26). Gated on the LIVE aspect; desktop is unchanged.
	function fieldIntensity(): number {
		return viewportAspect() < 0.85 ? 0.24 : 0.75;
	}

	// On portrait the neighborhood column sits dead-center over the field's bright
	// selected-cell bloom, killing text contrast. Carve a reading well over the
	// text band so the field dims there (the pass auto-reflows the well for
	// portrait). Desktop gets NO well (hw<=0 disables it) → byte-identical render.
	function applyReadingWell(field: LivingFieldPass) {
		if (viewportAspect() < 0.85) {
			field.setReadingWell({ x: 0, y: 0, hw: 1.05, hh: 0.95, floor: 0.08, soft: 0.3 });
		} else {
			field.setReadingWell({ x: 0, y: 0, hw: -1, hh: 0 });
		}
	}

	async function handleReady(engine: ObservatoryEngine) {
		engineRef = engine;
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		field.setIntensity(fieldIntensity());
		applyReadingWell(field);
		field.setCells(buildFieldCells());
		engine.addPass(field);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		await pass.init();
		pass.setText(buildTextItems());
		engine.addPass(pass);
		engine.demoClock.reset();
	}

	/**
	 * Squash a memory's content into a compact semantic seed (URL + API safe).
	 * Prefers a word boundary near the cap so the seed never ends mid-word (or,
	 * worse, mid-surrogate-pair — raw slice cuts UTF-16 code units).
	 */
	function walkQuery(content: string): string {
		const squashed = content.replace(/\s+/g, ' ').trim();
		if (squashed.length <= 280) return squashed;
		const hard = squashed.slice(0, 280);
		const lastSpace = hard.lastIndexOf(' ');
		return lastSpace > 200 ? hard.slice(0, lastSpace) : hard;
	}

	/** Sync ?q= without a history entry; a failed sync is warned, never silent. */
	function syncQueryToUrl() {
		try {
			replaceState(`?q=${encodeURIComponent(searchQuery)}`, {});
		} catch (err) {
			console.warn('[explore] URL sync failed — walk state is not shareable:', err);
		}
	}

	async function loadNeighbors() {
		const gen = ++loadGen;
		error = null;
		// The search backend rejects an empty query (500). The page is zero-DOM (no
		// input field), so with no ?q= we AUTO-SEED the expedition from the newest
		// memory (verified: /api/memories returns newest-first) — landing here is
		// immediately alive, and every click walks onward from a real thought.
		if (!searchQuery.trim()) {
			try {
				// If another organ handed us a memory (?memory=<id>), seed the walk
				// from THAT thought so the selection survives the handoff. Otherwise
				// auto-seed from the newest memory so landing here is always alive.
				const seedMemory = seedMemoryId
					? await api.memories.get(seedMemoryId).catch(() => null)
					: null;
				const start = seedMemory ?? (await api.memories.list({ limit: '1' })).memories?.[0] ?? null;
				if (gen !== loadGen) return; // a newer walk superseded this load
				if (start?.content) {
					searchQuery = walkQuery(start.content);
					centerId = start.id;
					seededFrom = start.id.slice(0, 8);
					// Consume the one-shot seed so a later re-walk doesn't re-pin it.
					seedMemoryId = null;
					// The seeded expedition must be shareable/reproducible too — a
					// reload mid-ingest would otherwise seed from a different memory.
					syncQueryToUrl();
				}
			} catch {
				/* fall through to the calm empty state */
			}
		}
		if (!searchQuery.trim()) {
			loading = false;
			searchResult = null;
			results = [];
			textPass?.setText(buildTextItems());
			engineRef?.demoClock.reset();
			return;
		}
		loading = true;
		textPass?.setText(buildTextItems());
		try {
			const res = centerId
				? await api.explore(centerId, 'associations', undefined, SEARCH_LIMIT)
				: await api.search(searchQuery, SEARCH_LIMIT);
			if (gen !== loadGen) return; // stale response — the newer walk owns state
			searchResult = 'query' in res ? (res as SearchResult) : null;
			results = res.results as SearchMemory[];
		} catch (err) {
			if (gen !== loadGen) return;
			searchResult = null;
			results = [];
			error = err instanceof Error ? err.message : 'UNKNOWN EXPLORE FETCH ERROR';
		} finally {
			if (gen === loadGen) {
				loading = false;
				walkingFrom = null; // only the owning generation clears the label
				textPass?.setText(buildTextItems());
				// Re-assert intensity for the current aspect (a rotate between load and
				// now would otherwise keep the stale desktop bloom on a phone).
				fieldPass?.setIntensity(fieldIntensity());
				if (fieldPass) applyReadingWell(fieldPass);
				fieldPass?.setCells(buildFieldCells());
				engineRef?.demoClock.reset();
			}
		}
	}

	function buildFieldCells() {
		const data: FieldDatum[] = results.map((memory, index) => ({
			id: memory.id,
			score: similarity(memory),
			hue: index === 0 ? FIELD_HUE.oxygen : FIELD_HUE.bridge,
			energy: similarity(memory),
			metric2: retention(memory),
			selected: index === 0,
			kind: 'explore-neighbor',
			payload: memory
		}));
		return layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.035, maxCellR: 0.1 });
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

	function scalar(value: unknown, fallback = 0.5): number {
		return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
	}

	function similarity(memory: SearchMemory): number {
		return clamp01(
			scalar(
				memory.similarity,
				scalar(memory.score, scalar(memory.relevance, scalar(memory.combinedScore, memory.retrievalStrength)))
			)
		);
	}

	function retention(memory: SearchMemory): number {
		return clamp01(scalar(memory.retention, memory.retentionStrength));
	}

	// Trim to a cap on a word boundary near the cap so a portrait row never ends
	// mid-token; hard-slice fallback for a single unbroken token.
	function trimSnippet(text: string, cap: number): string {
		const s = sanitizeAscii(text).replace(/\s+/g, ' ').trim();
		if (s.length <= cap) return s;
		const hard = s.slice(0, cap);
		const lastSpace = hard.lastIndexOf(' ');
		return lastSpace > cap * 0.6 ? hard.slice(0, lastSpace) : hard;
	}

	function resultLine(memory: SearchMemory, portrait: boolean): string {
		const sim = `${Math.round(similarity(memory) * 100)}%`;
		if (portrait) return sanitizeAscii(`${trimSnippet(memory.content, 28)}  ${sim}`);
		return sanitizeAscii(
			`${trimSnippet(memory.content, 52)} | ${memory.id.slice(0, 8)} | ${sim} | ${Math.round(retention(memory) * 100)}%`
		);
	}

	function statusItem(text: string, color = MUTED): ExploreTextItem {
		return {
			id: 'explore:status',
			kind: 'explore-status',
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

	function buildTextItems(): ExploreTextItem[] {
		if (loading) {
			return [
				statusItem(
					walkingFrom ? `WALKING FROM ${walkingFrom}...` : 'LOADING SEARCH NEIGHBORS...',
					CYAN
				)
			];
		}
		if (error) return [statusItem(`ERROR - ${error}`.slice(0, 72), SCARLET)];
		if (results.length === 0) {
			// No query and no seedable memory → invite one; empty result → say so.
			if (!searchQuery.trim()) {
				return [statusItem('SEMANTIC EXPEDITION - EMPTY BRAIN, INGEST A MEMORY TO EXPLORE', MUTED)];
			}
			const emptyText = searchResult?.query
				? `NO NEIGHBORS FOR "${searchResult.query}" | ${searchResult.durationMs}ms`
				: `NO NEIGHBORS FOR "${searchQuery}"`;
			return [statusItem(emptyText, MUTED)];
		}

		const aspect = viewportAspect();
		const portrait = aspect < 0.85;

		if (portrait) {
			// Phone plan: ONE short focal header + a well-spaced, short-line column
			// that fits the screen band with real negative space. Row count/spacing
			// derive from the live aspect (never a fixed phone number). portraitAdapt
			// maps authored-y straight to screen-y, so we author in screen NDC.
			const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.42));
			const rowCount = Math.max(9, Math.round(12 - 3 * portraitness));
			const rows = results.slice(0, rowCount);
			const header: ExploreTextItem = {
				id: 'explore:origin',
				kind: 'explore-origin',
				text: sanitizeAscii(
					seededFrom ? `NEIGHBORHOOD OF ${seededFrom}  TAP TO WALK` : `NEIGHBORHOOD  ${rows.length} THOUGHTS  TAP TO WALK`
				),
				x: -0.82,
				y: 0.8,
				size: 0.028,
				color: MUTED,
				depth: 0.85,
				weight: 0.6,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 24,
				maxWidthEm: 30
			};
			const top = 0.62;
			const bottom = -0.72;
			const rowStep = rows.length > 1 ? (top - bottom) / (rows.length - 1) : 0;
			return [
				header,
				...rows.map((memory, i) => ({
					id: `explore:${memory.id}`,
					kind: 'explore-neighbor',
					memoryId: memory.id,
					text: resultLine(memory, true),
					x: -0.82,
					y: top - i * rowStep,
					size: 0.03,
					color: CYAN,
					depth: similarity(memory),
					weight: retention(memory),
					startFrame: REVEAL_ANCHOR + i * 2,
					revealSpan: 20,
					maxWidthEm: 34,
					hitPadX: 0.04,
					hitPadY: 0.03
				}) satisfies ExploreTextItem)
			];
		}

		const rows = results.slice(0, ROW_LIMIT);
		const top = 0.72;
		const rowStep = 1.5 / Math.max(1, ROW_LIMIT - 1);
		// Expedition origin line (display-only, not pickable — no hitPad): tells the
		// viewer where this neighborhood came from (auto-seed or an explicit query).
		const origin: ExploreTextItem = {
			id: 'explore:origin',
			kind: 'explore-origin',
			text: sanitizeAscii(
				seededFrom
					? `EXPEDITION SEEDED FROM NEWEST MEMORY ${seededFrom} - CLICK ANY THOUGHT TO WALK`
					: `SEMANTIC NEIGHBORHOOD - ${rows.length} THOUGHTS - CLICK TO WALK`
			),
			x: -0.88,
			y: 0.82,
			size: 0.022,
			color: MUTED,
			depth: 0.85,
			weight: 0.6,
			revealSpan: 24,
			maxWidthEm: 60
		};
		return [origin, ...rows.map((memory, i) => ({
			id: `explore:${memory.id}`,
			kind: 'explore-neighbor',
			memoryId: memory.id,
			text: resultLine(memory, false),
			x: -0.88,
			y: top - i * rowStep,
			size: 0.026,
			color: CYAN,
			depth: similarity(memory),
			weight: retention(memory),
			startFrame: REVEAL_ANCHOR + i * 2,
			revealSpan: 20,
			maxWidthEm: 46,
			hitPadX: 0.03,
			hitPadY: 0.015
		}))];
	}

	function pointerToNdc(e: PointerEvent | MouseEvent): { x: number; y: number } | null {
		if (!hostEl) return null;
		const rect = hostEl.getBoundingClientRect();
		if (rect.width <= 0 || rect.height <= 0) return null;
		return {
			x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
			y: -(((e.clientY - rect.top) / rect.height) * 2 - 1)
		};
	}

	function writeCursorLens(ndc: { x: number; y: number }) {
		if (!hostEl || !engineRef) return;
		const rect = hostEl.getBoundingClientRect();
		const aspect = Math.max(0.0001, rect.width / Math.max(1, rect.height));
		const raw = { x: ndc.x * Math.max(aspect, 1), y: ndc.y / Math.min(aspect, 1) };
		const prev = cursorSmoothed ?? raw;
		const next = { x: prev.x + (raw.x - prev.x) * 0.35, y: prev.y + (raw.y - prev.y) * 0.35 };
		cursorSmoothed = next;
		engineRef.setCursorPreNdc(next.x, next.y, next.x - prev.x, next.y - prev.y);
	}

	function handlePointerMove(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		writeCursorLens(ndc);
		const hit = textPass?.pickAt(ndc.x, ndc.y) ?? null;
		const nextRun = hit?.kind === 'explore-neighbor' ? hit.id : null;
		if (nextRun !== activeRun) {
			activeRun = nextRun;
			textPass?.setRunDepth(nextRun, 1);
		}
		if (hostEl) hostEl.style.cursor = nextRun ? 'crosshair' : 'default';
	}

	function handlePointerLeave() {
		cursorSmoothed = null;
		activeRun = null;
		engineRef?.setCursorPreNdc(999, 999, 0, 0);
		textPass?.setRunDepth(null);
		if (hostEl) hostEl.style.cursor = 'default';
	}

	async function handlePointerDown(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc || !textPass) return;
		const hit = textPass.pickAt(ndc.x, ndc.y);
		if (hit?.kind !== 'explore-neighbor') return;
		const item = hit.payload as ExploreTextItem;
		if (!item.memoryId) return;
		// SEMANTIC WALK (Wave 3): re-center the neighborhood on the clicked thought,
		// IN PLACE. (The old goto('/memories/{id}') 404'd — no such route exists.)
		// We search by the memory's CONTENT (semantic seed), not its id; the URL ?q=
		// is synced via replaceState so the walk is shareable/back-safe without
		// relying on onMount re-running (it doesn't on same-route navigation).
		const memory = results.find((m) => m.id === item.memoryId);
		if (!memory?.content) return;
		walkingFrom = memory.id.slice(0, 8);
		centerId = memory.id;
		seededFrom = null;
		searchQuery = walkQuery(memory.content);
		syncQueryToUrl();
		// walkingFrom is cleared by loadNeighbors' gen-guarded finally, so a rapid
		// second walk can't have its label wiped by the superseded first walk.
		await loadNeighbors();
	}
</script>

<svelte:head>
	<title>Explore · Vestige</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={hostEl} class="fixed inset-0 bg-[#020307]" onpointerdown={handlePointerDown} onpointermove={handlePointerMove} onpointerleave={handlePointerLeave}>
	<ObservatoryCanvas demo="recall-path" seed={`real-explore-neighbors:${searchQuery}:${searchResult?.total ?? 0}`} onready={handleReady} />
</div>
