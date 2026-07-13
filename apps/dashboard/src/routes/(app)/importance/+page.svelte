<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import { api } from '$stores/api';
	import type { ImportanceScore, Memory } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01 } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type ImportanceRecord = {
		memory: Memory;
		score: ImportanceScore;
	};

	type ImportanceTextItem = TextLayerItem & { memoryId?: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const MUTED = [...rgb01('#29F2A9'), 0.62] satisfies [number, number, number, number];
	const AMBER = [...rgb01('#FFB020'), 0.86] satisfies [number, number, number, number];
	const TITLE = [...rgb01('#EAF3FF'), 0.96] satisfies [number, number, number, number];
	const HEADER = [...rgb01('#7FA0B8'), 0.72] satisfies [number, number, number, number];
	const MEMORY_LIMIT = 36;
	const ROW_LIMIT = 30;
	// Phone shows fewer rows so each gets real vertical breathing room instead of a
	// 30-deep crushed wall of tiny text.
	const ROW_LIMIT_PORTRAIT = 16;
	const REVEAL_ANCHOR = -100000;
	const MIN_VISIBLE_DEPTH = 0.62;

	// Portrait / narrow viewport, derived from the LIVE engine aspect (params[6]/[7])
	// with a window fallback — same signal TextLayerPass.portraitAdapt uses. Nothing
	// is hardcoded to a phone width; desktop (aspect>=0.85) is untouched.
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

	let hostEl: HTMLDivElement | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;
	let textPass: TextLayerPass | null = null;
	let fieldPass: LivingFieldPass | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;
	let records: ImportanceRecord[] = $state([]);
	let total = $state(0);
	let loading = $state(true);
	let error: string | null = $state(null);
	let activeRun: string | null = null;

	onMount(() => {
		void loadImportanceField();
	});

	let stopReducedMotion: (() => void) | null = null;

	onDestroy(() => {
		stopReducedMotion?.();
		textPass?.dispose();
		fieldPass?.dispose();
		textPass = null;
		fieldPass = null;
		engineRef = null;
	});

	// Mounts ObservatoryCanvas directly (not RouteStage), so wire prefers-reduced-
	// motion here: a reduce-motion user gets a frozen field, not a moving one.
	function initReducedMotion(engine: ObservatoryEngine): () => void {
		if (typeof window === 'undefined') return () => {};
		const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
		engine.setPaused(mq.matches);
		const onChange = (e: MediaQueryListEvent) => engine.setPaused(e.matches);
		mq.addEventListener('change', onChange);
		return () => mq.removeEventListener('change', onChange);
	}

	async function handleReady(engine: ObservatoryEngine) {
		engineRef = engine;
		stopReducedMotion = initReducedMotion(engine);
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		// Text-heavy organ: keep the field a DIM backdrop so 30 rows of MSDF text
		// read cleanly. Rows anchor at x=-0.9 and run rightward (~54em), y from 0.72
		// down to ~-0.78, so the reading well covers the whole left+center column.
		field.setIntensity(0.22);
		field.setReadingWell({ x: -0.3, y: -0.03, hw: 0.7, hh: 0.88, floor: 0.08, soft: 0.25 });
		field.setCells(buildFieldCells());
		engine.addPass(field);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		await pass.init();
		pass.setText(buildTextItems());
		engine.addPass(pass);
		engine.demoClock.reset();
	}

	async function loadImportanceField() {
		loading = true;
		error = null;
		textPass?.setText(buildTextItems());
		try {
			const res = await api.memories.list({ limit: String(MEMORY_LIMIT) });
			total = res.total;
			const scored = await Promise.allSettled(
				res.memories.map(async (memory) => ({ memory, score: await api.importance(memory.content) }))
			);
			records = scored
				.filter((result): result is PromiseFulfilledResult<ImportanceRecord> => result.status === 'fulfilled')
				.map((result) => result.value)
				.sort((a, b) => b.score.composite - a.score.composite);
			if (res.memories.length > 0 && records.length === 0) {
				error = 'API IMPORTANCE RETURNED NO SCORES';
			}
		} catch (err) {
			records = [];
			total = 0;
			error = err instanceof Error ? err.message : 'UNKNOWN IMPORTANCE FETCH ERROR';
		} finally {
			loading = false;
			textPass?.setText(buildTextItems());
			fieldPass?.setCells(buildFieldCells());
			engineRef?.demoClock.reset();
		}
	}

	function buildFieldCells() {
		const data: FieldDatum[] = records.map((record) => ({
			id: record.memory.id,
			score: clamp01(record.score.composite),
			hue: record.score.recommendation === 'save' ? FIELD_HUE.oxygen : FIELD_HUE.caution,
			energy: Math.max(0.35, clamp01(record.score.composite)),
			metric2: clamp01(record.memory.retentionStrength),
			kind: 'importance',
			payload: record
		}));
		return layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.04, maxCellR: 0.1 });
	}

	function sanitizeAscii(value: string): string {
		if (typeof value !== 'string') return '';
		return value
			.replace(/[\u2014\u2013]/g, '-')
			.replace(/[\u2018\u2019]/g, "'")
			.replace(/[\u201C\u201D]/g, '"')
			.replace(/\u2026/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function importanceLine(record: ImportanceRecord, portrait: boolean): string {
		const { memory, score } = record;
		if (portrait) {
			// Phone: drop the id/retention/recommendation/channel columns that turn the
			// row into an unreadable wall. Keep the primary content (a longer title
			// budget) and the ONE headline metric — composite score — right-aligned by
			// the header. `title  92%` reads at a glance; the full detail lives on tap.
			const snippet = sanitizeAscii(memory.content ?? '')
				.replace(/\s+/g, ' ')
				.trim()
				.slice(0, 34);
			return sanitizeAscii(`${snippet}  ${Math.round(score.composite * 100)}%`);
		}
		const snippet = sanitizeAscii(memory.content ?? '').replace(/\s+/g, ' ').trim().slice(0, 44);
		const strongest = strongestChannel(score);
		return sanitizeAscii(
			`${snippet} | ${memory.id.slice(0, 8)} | ${Math.round(score.composite * 100)}% | ${Math.round(memory.retentionStrength * 100)}% | ${score.recommendation} | ${strongest}`
		);
	}

	function strongestChannel(score: ImportanceScore): string {
		return (Object.entries(score.channels) as [keyof ImportanceScore['channels'], number][])
			.sort((a, b) => b[1] - a[1])[0][0];
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
	}

	function statusItem(text: string, color = MUTED): ImportanceTextItem {
		return {
			id: 'importance:status',
			kind: 'importance-status',
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

	function buildTextItems(): ImportanceTextItem[] {
		if (loading) return [statusItem('LOADING IMPORTANCE FIELD...', CYAN)];
		if (error) return [statusItem(`ERROR - ${error}`.slice(0, 72), SCARLET)];
		if (records.length === 0) return [statusItem('EMPTY IMPORTANCE FIELD', MUTED)];

		const portrait = isPortrait();
		const rowLimit = portrait ? ROW_LIMIT_PORTRAIT : ROW_LIMIT;
		const rows = records.slice(0, rowLimit);
		// Phone gets a title + column header and a shorter, more generously spaced
		// list; desktop keeps the dense edge-to-edge log exactly as authored.
		const top = portrait ? 0.6 : 0.72;
		const span = portrait ? 1.28 : 1.5;
		const rowStep = span / Math.max(1, rowLimit - 1);
		const items: ImportanceTextItem[] = [];

		if (portrait) {
			items.push({
				id: 'importance:title',
				kind: 'importance-title',
				text: 'IMPORTANCE',
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
				id: 'importance:header',
				kind: 'importance-header',
				text: `MEMORY / SCORE  (${records.length})`,
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

		rows.forEach((record, i) => {
			// depth = crispness/forward channel — floor it or the DOF blur + dim glow
			// makes low-composite rows invisible (composite is low for most memories).
			const depth = Math.max(MIN_VISIBLE_DEPTH, clamp01(record.score.composite));
			const weight = clamp01(record.memory.retentionStrength);
			items.push({
				id: `importance:${record.memory.id}`,
				kind: 'importance',
				memoryId: record.memory.id,
				text: importanceLine(record, portrait),
				x: portrait ? -0.62 : -0.9,
				y: top - i * rowStep,
				size: portrait ? 0.03 : 0.025,
				color: record.score.recommendation === 'save' ? CYAN : AMBER,
				depth,
				weight,
				// The shared reveal packs ageFrame = startFrame + GLOBAL glyphIndex*2;
				// 30 long rows age far past the ~720-frame wrapped clock, so all but
				// the first row never reveal. Anchor deeply negative so every glyph is
				// pre-revealed on frame 0 (same fix as memories/schedule/patterns).
				startFrame: REVEAL_ANCHOR + i * 2,
				revealSpan: 1,
				maxWidthEm: 54,
				hitPadX: 0.03,
				hitPadY: 0.018
			});
		});
		return items;
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
		const nextRun = hit?.kind === 'importance' ? hit.id : null;
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
		// Pick the TEXT row first; if it misses, pick the FIELD cell (its payload is
		// the ImportanceRecord, id at record.memory.id). Without this, clicking a
		// glowing importance orb did nothing — only the text rows were clickable.
		let memoryId: string | undefined;
		const textHit = textPass.pickAt(ndc.x, ndc.y);
		if (textHit?.kind === 'importance') {
			memoryId = (textHit.payload as ImportanceTextItem).memoryId;
		} else {
			const fieldHit = fieldPass?.pickAt(ndc.x, ndc.y);
			if (fieldHit?.kind === 'importance') {
				const rec = fieldHit.payload as { memory?: { id?: string } };
				memoryId = rec.memory?.id;
			}
		}
		if (!memoryId) return;
		try {
			const promoted = await api.memories.promote(memoryId);
			// The promote endpoint returns a PARTIAL payload ({ id, promoted,
			// retentionStrength }) — NOT a full Memory. Merge it onto the existing
			// record so content/createdAt/etc. survive; replacing outright would drop
			// memory.content and crash importanceLine() on the next render.
			records = records.map((record) =>
				record.memory.id === promoted.id
					? {
							...record,
							memory: {
								...record.memory,
								retentionStrength: promoted.retentionStrength ?? record.memory.retentionStrength
							}
						}
					: record
			);
			textPass.setText(buildTextItems());
		} catch (err) {
			error = err instanceof Error ? err.message : 'UNKNOWN IMPORTANCE PROMOTE ERROR';
			textPass.setText(buildTextItems());
		}
	}
</script>

<svelte:head>
	<title>Importance · Vestige</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={hostEl} class="fixed inset-0 bg-[#020307]" onpointerdown={handlePointerDown} onpointermove={handlePointerMove} onpointerleave={handlePointerLeave}>
	<ObservatoryCanvas demo="recall-path" seed={`real-importance-field:${total}`} onready={handleReady} />
</div>
