<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import { api } from '$stores/api';
	import type { Memory } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01, retentionColor } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type MemoryTextItem = TextLayerItem & { memoryId?: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const MUTED = [...rgb01('#29F2A9'), 0.62] satisfies [number, number, number, number];
	const MEMORY_LIMIT = 40;
	const ROW_LIMIT = 36;
	const MIN_VISIBLE_DEPTH = 0.62;
	// The MSDF reveal is frame-driven per glyph: the shared text layer packs
	// ageFrame = startFrame + globalGlyphIndex * 2. Across many long rows that
	// global index reaches the thousands, so late glyphs would never reveal
	// before the demo clock wraps. Anchor the reveal far in the past so the
	// complete memory field is legible immediately.
	const REVEAL_ANCHOR = -100000;

	let hostEl: HTMLDivElement | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;
	let textPass: TextLayerPass | null = null;
	let fieldPass: LivingFieldPass | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;
	// Local suppressed-id set — Memory has no suppression field, so we track the
	// immune state we just applied so the field cell can scar immediately.
	let suppressedIds = $state<Set<string>>(new Set());
	let memories: Memory[] = $state([]);
	let total = $state(0);
	let loading = $state(true);
	let error: string | null = $state(null);
	let activeRun: string | null = null;
	// Tracked when the user picks a memory from the canvas. Selection only —
	// no API call. Mutations are gated to explicit modifier keys (shift/alt)
	// whose affordances are mirrored by the on-canvas hint line.
	let selectedMemoryId: string | null = $state(null);
	let search = $state('');
	let stopReducedMotion: (() => void) | null = null;
	const filteredMemories = $derived(
		memories.filter((memory) =>
			`${memory.content} ${memory.tags.join(' ')}`.toLowerCase().includes(search.trim().toLowerCase())
		)
	);
	const selectedMemory = $derived(memories.find((memory) => memory.id === selectedMemoryId) ?? null);

	onMount(() => {
		void loadMemories().then(async () => {
			// Deep-link contract: /memories?memory=<id> opens that memory's
			// inspector — Witness and the Observatory both link here. If the
			// id is not in the loaded page, fetch the real record and prepend
			// it (never a silent miss on a receipt link).
			const wanted = new URLSearchParams(window.location.search).get('memory');
			if (!wanted) return;
			if (!memories.some((m) => m.id === wanted)) {
				try {
					const record = await api.memories.get(wanted);
					memories = [record, ...memories];
				} catch {
					return; // unknown id — leave the library as-is
				}
			}
			selectedMemoryId = wanted;
		});
	});

	onDestroy(() => {
		stopReducedMotion?.();
		textPass?.dispose();
		fieldPass?.dispose();
		textPass = null;
		fieldPass = null;
		engineRef = null;
	});

	// This organ mounts ObservatoryCanvas directly (not RouteStage), so it must
	// wire prefers-reduced-motion itself: a reduce-motion user gets a frozen field
	// (the engine still renders one static frame) instead of continuous orbit.
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
		// Launch-grade ambient field: every real memory is a glowing cell. The DOM
		// overlay owns reading and actions so the scene never sacrifices usability to
		// a canvas text layer.
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		field.setIntensity(0.8);
		field.setCells(buildFieldCells());
		engine.addPass(field);
		engine.demoClock.reset();
	}

	function buildFieldCells() {
		const data: FieldDatum[] = memories.map((m) => {
			const retention = clamp01(m.retentionStrength);
			return {
				id: m.id,
				score: clamp01(m.retrievalStrength ?? retention),
				hue: retentionColor(retention),
				energy: 0.4 + 0.6 * clamp01(m.retrievalStrength ?? retention),
				metric2: retention,
				scar: suppressedIds.has(m.id),
				kind: 'memory',
				payload: m
			} satisfies FieldDatum;
		});
		return layoutGalaxy(data, { maxRadius: 0.95, minCellR: 0.014, maxCellR: 0.05 });
	}

	async function loadMemories() {
		loading = true;
		error = null;
		try {
			const res = await api.memories.list({ limit: String(MEMORY_LIMIT) });
			memories = res.memories;
			total = res.total;
		} catch (err) {
			memories = [];
			total = 0;
			error = err instanceof Error ? err.message : 'UNKNOWN MEMORY FETCH ERROR';
		} finally {
			loading = false;
			fieldPass?.setCells(buildFieldCells());
			engineRef?.demoClock.reset();
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

	// Portrait rows must READ, not dump. A phone can only fit ~10-12 rows with the
	// generous spacing the taste bar demands, and long lines would shrink to
	// illegible to fit width. So on portrait we shorten each line (drop the id
	// column, tighten the snippet) so it never runs edge-to-edge, and the caller
	// caps the row count. Everything is gated on the LIVE aspect — desktop keeps
	// the full id column and long snippet, byte-identical.
	// Trim a snippet to a cap, breaking on the last word boundary near the cap so a
	// portrait row never ends mid-token ("Jul 202"); falls back to a hard slice for
	// a single unbroken token.
	function trimSnippet(text: string, cap: number): string {
		const s = sanitizeAscii(text).replace(/\s+/g, ' ').trim();
		if (s.length <= cap) return s;
		const hard = s.slice(0, cap);
		const lastSpace = hard.lastIndexOf(' ');
		return lastSpace > cap * 0.6 ? hard.slice(0, lastSpace) : hard;
	}

	function memoryLine(memory: Memory, portrait: boolean): string {
		const pct = `${Math.round(memory.retentionStrength * 100)}%`;
		if (portrait) return sanitizeAscii(`${trimSnippet(memory.content, 28)}  ${pct}`);
		return sanitizeAscii(`${trimSnippet(memory.content, 52)} | ${memory.id.slice(0, 8)} | ${pct}`);
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
	}

	function statusItem(text: string, color = MUTED): MemoryTextItem {
		return {
			id: 'memories:status',
			kind: 'memory-status',
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

	function buildTextItems(): MemoryTextItem[] {
		if (loading) return [statusItem('LOADING MEMORY FIELD...', CYAN)];
		if (error) return [statusItem(`ERROR - ${error}`.slice(0, 72), SCARLET)];
		if (memories.length === 0) return [statusItem('EMPTY MEMORY FIELD', MUTED)];

		// Persistent on-canvas affordance: a plain click selects; shift and alt are
		// the only modifiers that mutate. Anchored low so it never collides with the
		// row band above, and shaped as a "memory-hint" kind so it is NEVER picked as
		// a memory row by the text layer.
		const hint: MemoryTextItem = {
			id: 'memories:hint',
			kind: 'memory-hint',
			text: 'click: select  ·  shift: suppress  ·  alt: unsuppress',
			x: -0.82,
			y: -0.9,
			size: 0.022,
			color: MUTED,
			depth: 0.7,
			weight: 0.5,
			startFrame: REVEAL_ANCHOR,
			revealSpan: 24,
			maxWidthEm: 50
		};

		const aspect = viewportAspect();
		const portrait = aspect < 0.85;

		if (portrait) {
			// Phone plan: ONE focal header + a short, well-spaced column that fits the
			// screen band with real negative space. Row count and spacing derive from
			// the live aspect (taller/narrower → fewer rows), never a fixed phone
			// number. portraitAdapt maps authored-y straight to screen-y (its inv
			// reclaim and the shader's aspect crush cancel), so we author in screen NDC.
			const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.42));
			// 12 rows at the wide-portrait edge, down to 9 on the tallest phones —
			// enough to be a real list, few enough to breathe and never collide.
			const rowCount = Math.max(9, Math.round(12 - 3 * portraitness));
			const rows = memories.slice(0, rowCount);
			const headerY = 0.8;
			const top = 0.62;
			const bottom = -0.72;
			const rowStep = rows.length > 1 ? (top - bottom) / (rows.length - 1) : 0;
			const header: MemoryTextItem = {
				id: 'memories:header',
				kind: 'memory-header',
				text: `MEMORY FIELD  ${total} TRACES`,
				x: -0.82,
				y: headerY,
				size: 0.03,
				color: MUTED,
				depth: 0.85,
				weight: 0.6,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 24,
				maxWidthEm: 30
			};
			return [
				hint,
				header,
				...rows.map((memory, i) => {
					const retrieval = clamp01(memory.retrievalStrength);
					const retention = clamp01(memory.retentionStrength);
					return {
						id: `mem:${memory.id}`,
						kind: 'memory',
						memoryId: memory.id,
						text: memoryLine(memory, true),
						x: -0.82,
						y: top - i * rowStep,
						size: 0.03,
						color: CYAN,
						depth: Math.max(MIN_VISIBLE_DEPTH, retrieval),
						weight: retention,
						startFrame: REVEAL_ANCHOR + i * 2,
						revealSpan: 20,
						maxWidthEm: 34,
						hitPadX: 0.04,
						hitPadY: 0.03
					} satisfies MemoryTextItem;
				})
			];
		}

		const rows = memories.slice(0, ROW_LIMIT);
		const top = 0.72;
		const rowStep = 1.5 / Math.max(1, ROW_LIMIT - 1);
		return [
			hint,
			...rows.map((memory, i) => {
				const retrieval = clamp01(memory.retrievalStrength);
				const retention = clamp01(memory.retentionStrength);
				return {
					id: `mem:${memory.id}`,
					kind: 'memory',
					memoryId: memory.id,
					text: memoryLine(memory, false),
					x: -0.88,
					y: top - i * rowStep,
					size: 0.026,
					color: CYAN,
					depth: Math.max(MIN_VISIBLE_DEPTH, retrieval),
					weight: retention,
					startFrame: REVEAL_ANCHOR + i * 2,
					revealSpan: 20,
					maxWidthEm: 46,
					hitPadX: 0.03,
					hitPadY: 0.015
				};
			})
		];
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
		if (hostEl) hostEl.style.cursor = 'default';
	}

	function handlePointerLeave() {
		cursorSmoothed = null;
		engineRef?.setCursorPreNdc(999, 999, 0, 0);
		if (hostEl) hostEl.style.cursor = 'default';
	}

	async function handlePointerDown(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		// Field cells ARE real memories — clicking one opens its inspector.
		// pickAt mirrors the animated orbit on CPU, so the click lands on the
		// cell where it is NOW, not its static home.
		const hit = fieldPass?.pickAt(ndc.x, ndc.y);
		if (hit && typeof hit.id === 'string') {
			selectedMemoryId = hit.id;
		}
	}

	async function suppressSelected() {
		if (!selectedMemory) return;
		try {
			await api.memories.suppress(selectedMemory.id, 'suppressed from memory library');
			suppressedIds = new Set(suppressedIds).add(selectedMemory.id);
			error = null;
			fieldPass?.setCells(buildFieldCells());
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unable to suppress memory';
		}
	}

	async function unsuppressSelected() {
		if (!selectedMemory) return;
		try {
			const result = await api.memories.unsuppress(selectedMemory.id);
			if (!result.stillSuppressed) {
				const next = new Set(suppressedIds);
				next.delete(selectedMemory.id);
				suppressedIds = next;
			}
			error = null;
			fieldPass?.setCells(buildFieldCells());
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unable to restore memory';
		}
	}
</script>

<svelte:head>
	<title>Memories · Vestige</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={hostEl} class="fixed inset-0 bg-[#020307]" onpointerdown={handlePointerDown} onpointermove={handlePointerMove} onpointerleave={handlePointerLeave}>
	<ObservatoryCanvas demo="recall-path" seed={`real-memory-field:${total}`} onready={handleReady} />
</div>

<main class="memory-library">
	<header class="library-head">
		<div>
			<p class="eyebrow">LOCAL MEMORY LIBRARY</p>
			<h1>Memories that can prove where an answer came from.</h1>
			<p>Each signal in the field is a real local memory. Select one to inspect or manage it.</p>
		</div>
		<div class="library-stat"><strong>{total}</strong><span>memories indexed locally</span></div>
	</header>

	<section class="library-grid">
		<div class="memory-list glass-panel">
			<div class="list-tools">
				<label for="memory-search">Search your memory</label>
				<input id="memory-search" bind:value={search} placeholder="Try refund, policy, project…" />
			</div>
			{#if loading}
				<div class="state-line">Loading your local memory…</div>
			{:else if error}
				<div class="state-line error">{error}</div>
			{:else if filteredMemories.length === 0}
				<div class="state-line">No memory matches that search.</div>
			{:else}
				<div class="memory-rows">
					{#each filteredMemories as memory (memory.id)}
						<button
							type="button"
							class:active={memory.id === selectedMemoryId}
							onclick={() => (selectedMemoryId = memory.id)}
						>
							<span class="memory-dot" style={`--strength:${Math.round(memory.retentionStrength * 100)}%`}></span>
							<span class="memory-copy"><strong>{memory.content}</strong><small>{memory.id.slice(0, 8)} · {Math.round(memory.retentionStrength * 100)}% retention</small></span>
						</button>
					{/each}
				</div>
			{/if}
		</div>

		<aside class="inspector glass-panel">
			{#if selectedMemory}
				<p class="eyebrow">SELECTED MEMORY</p>
				<h2>{selectedMemory.content}</h2>
				<div class="metric-row"><span>Retention</span><strong>{Math.round(selectedMemory.retentionStrength * 100)}%</strong></div>
				<div class="metric-row"><span>Retrieval strength</span><strong>{Math.round((selectedMemory.retrievalStrength ?? 0) * 100)}%</strong></div>
				<div class="tags">{#each selectedMemory.tags as tag}<span>{tag}</span>{/each}</div>
				<code>{selectedMemory.id}</code>
				<div class="action-row">
					{#if suppressedIds.has(selectedMemory.id)}
						<button type="button" class="secondary" onclick={unsuppressSelected}>Restore retrieval</button>
					{:else}
						<button type="button" class="danger" onclick={suppressSelected}>Suppress from retrieval</button>
					{/if}
				</div>
				<p class="inspector-note">Managing a memory is explicit. This does not rewrite its content.</p>
			{:else}
				<div class="empty-inspector"><p class="eyebrow">FIELD IS LIVE</p><h2>Select a memory to inspect its state.</h2><p>The ambient field reacts to real retention and retrieval strength. The details stay readable here.</p></div>
			{/if}
		</aside>
	</section>
</main>

<style>
	.memory-library { position: relative; z-index: 2; max-width: 1180px; margin: 0 auto; min-height: 100%; padding: 2rem clamp(1rem,3vw,2.5rem) 5rem; color:#eaf9f6; pointer-events:none; }
	.library-head,.library-grid,.glass-panel,.memory-rows,.action-row { display:flex; } .library-head { justify-content:space-between; gap:2rem; align-items:flex-end; margin-bottom:1.25rem; } .eyebrow { margin:0; color:#66e6d3; font:700 .68rem/1.2 ui-monospace,monospace; letter-spacing:.14em; } h1 { max-width:23ch; margin:.55rem 0; font-size:clamp(1.65rem,3.3vw,2.7rem); line-height:1.06; letter-spacing:-.045em; } .library-head p:not(.eyebrow),.empty-inspector p { max-width:58ch; color:#a9c4c0; line-height:1.5; } .library-stat { min-width:9rem; border-left:1px solid rgba(99,230,211,.35); padding-left:1rem; } .library-stat strong { display:block; color:#62ebd5; font-size:2rem; } .library-stat span { color:#9ab6b1; font-size:.75rem; }
	.library-grid { display:grid; grid-template-columns:minmax(0,1.28fr) minmax(300px,.72fr); gap:1rem; pointer-events:auto; } .glass-panel { flex-direction:column; border:1px solid rgba(124,198,187,.2); border-radius:1rem; background:linear-gradient(135deg,rgba(9,26,29,.9),rgba(5,14,16,.82)); backdrop-filter:blur(12px); box-shadow:0 18px 70px rgba(0,0,0,.24); }
	.memory-list { min-height:38rem; padding:1rem; } .list-tools label { display:block; color:#b8d0cc; font-size:.75rem; font-weight:700; } .list-tools input { width:100%; box-sizing:border-box; margin-top:.45rem; border:1px solid rgba(110,204,191,.28); border-radius:.65rem; background:rgba(0,0,0,.22); padding:.75rem .8rem; color:#effefb; outline:none; } .list-tools input:focus { border-color:#61e4d0; box-shadow:0 0 0 3px rgba(80,228,208,.12); }
	.memory-rows { flex-direction:column; gap:.45rem; margin-top:.9rem; overflow:auto; max-height:34rem; } .memory-rows button { display:flex; gap:.7rem; width:100%; border:1px solid transparent; border-radius:.7rem; background:rgba(255,255,255,.02); padding:.8rem; color:inherit; text-align:left; cursor:pointer; } .memory-rows button:hover,.memory-rows button.active { border-color:rgba(83,231,209,.48); background:rgba(0,223,195,.09); } .memory-dot { flex:0 0 .55rem; height:.55rem; margin-top:.25rem; border-radius:50%; background:#4ee3cd; box-shadow:0 0 calc(var(--strength) / 6) #4ee3cd; } .memory-copy { min-width:0; } .memory-copy strong { display:block; overflow:hidden; color:#ddf2ee; font-size:.86rem; line-height:1.4; text-overflow:ellipsis; white-space:nowrap; } .memory-copy small { display:block; margin-top:.25rem; color:#82a39e; font: .66rem ui-monospace,monospace; } .state-line { padding:2rem .6rem; color:#94b8b1; }.state-line.error{color:#ff9b91}
	.inspector { min-height:25rem; padding:1.35rem; align-self:start; } .inspector h2,.empty-inspector h2 { margin:.75rem 0 1rem; color:#f1fffc; font-size:1.15rem; line-height:1.45; } .metric-row { display:flex; justify-content:space-between; border-top:1px solid rgba(137,190,183,.15); padding:.7rem 0; color:#92b3ae; font-size:.8rem; } .metric-row strong { color:#77e7d6; }.tags { display:flex; flex-wrap:wrap; gap:.35rem; margin:.75rem 0; }.tags span { border:1px solid rgba(108,182,171,.24); border-radius:99px; padding:.25rem .45rem; color:#accbc6; font-size:.67rem;}.inspector code{display:block;color:#73938e;font-size:.66rem;overflow-wrap:anywhere}.action-row{margin-top:1.2rem}.action-row button{border:0;border-radius:.6rem;padding:.65rem .8rem;color:#effffd;font-weight:700;cursor:pointer}.danger{background:rgba(217,74,63,.78)}.secondary{background:rgba(0,205,177,.75)}.inspector-note{margin-top:.8rem;color:#8aa9a5;font-size:.74rem;line-height:1.45}.empty-inspector{margin:auto 0}.empty-inspector p:last-child{font-size:.83rem}
	@media(max-width:760px){.library-head,.library-grid{grid-template-columns:1fr;display:grid}.library-stat{border-left:0;border-top:1px solid rgba(99,230,211,.35);padding:1rem 0 0}.memory-list{min-height:20rem}.memory-rows{max-height:22rem}}
</style>
