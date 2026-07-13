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
	import PageHeader from '$lib/components/PageHeader.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { spotlight } from '$lib/actions/interactions';

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
	// The DOM row the user last asked "why" about — so the returned risk signals
	// render against a concrete PR, not a floating panel with no anchor.
	let whyForPrId: string | null = $state(null);

	// PORTRAIT GATE — everything below is gated on the LIVE viewport aspect so the
	// desktop (landscape, aspect>=0.85) render stays byte-identical zero-DOM: no DOM
	// overlay, full-strength in-canvas PR field. On a phone (portrait, aspect<0.85)
	// the in-canvas log wall is illegible, so we surface a readable DOM overlay AND
	// dim the field to a pure backdrop. Threshold matches TextLayerPass.portraitAdapt.
	let isPortrait = $state(false);
	onMount(() => {
		const update = () => {
			isPortrait = window.innerWidth / Math.max(1, window.innerHeight) < 0.85;
		};
		update();
		window.addEventListener('resize', update);
		return () => window.removeEventListener('resize', update);
	});

	const pendingCount = $derived(prs.filter((pr) => pr.status === 'pending').length);
	// Cap the readable DOM list to the same window the field renders so the two
	// stay in sync and the scroll stays bounded on a phone.
	const domRows = $derived(prs.slice(0, ROW_LIMIT));

	function prStatusTone(status: string): string {
		if (status === 'pending') return 'text-warning border-warning/30 bg-warning/10';
		if (status === 'approved' || status === 'promoted') return 'text-recall border-recall/25 bg-recall/10';
		if (status === 'rejected' || status === 'forgotten') return 'text-decay border-decay/25 bg-decay/10';
		return 'text-dim border-white/10 bg-white/[0.04]';
	}

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

	// In PORTRAIT the readable DOM overlay is the hero, so the in-canvas PR rows must
	// recede to a faint ambient substrate (they'd otherwise be an illegible wall of
	// log text competing with the DOM copy on top). In LANDSCAPE (desktop) the
	// in-canvas text IS the content, so it keeps full strength — desktop unchanged.
	function dim(color: [number, number, number, number]): [number, number, number, number] {
		if (!isPortrait) return color;
		return [color[0], color[1], color[2], color[3] * 0.34];
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
			color: dim(pr.status === 'pending' ? CYAN : MUTED),
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
			color: dim(AMBER),
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

	// Live handles so the portrait $effect can re-dim the field and re-push the
	// (portrait-dimmed) in-canvas text when the viewport aspect crosses the gate.
	let fieldPass: MemoryPrFieldPass | null = null;
	let textPass: TextLayerPass | null = null;

	$effect(() => {
		// Re-apply the portrait/landscape backdrop treatment whenever the gate flips.
		const portrait = isPortrait;
		fieldPass?.applyBackdrop(portrait);
		textPass?.setText(buildTextItems());
	});

	function createMemoryPrPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		const field = new MemoryPrFieldPass(engine);
		field.applyBackdrop(isPortrait);
		field.uploadScene(scene);
		fieldPass = field;
		const text = new TextLayerPass(engine);
		textPass = text;
		void text.init().then(() => text.setText(buildTextItems()));
		return [field,
			{
				render: (pass) => text.render(pass),
				uploadScene: () => text.setText(buildTextItems()),
				pickAt: (x, y) => text.pickAt(x, y),
				dispose: () => {
					if (textPass === text) textPass = null;
					text.dispose();
				}
			}
		];
	}

	class MemoryPrFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
		}
		// PORTRAIT: the readable hero is the DOM overlay, so the field recedes to a
		// faint full-frame ambient substrate behind the cards. LANDSCAPE (desktop):
		// the in-canvas PR queue IS the content, so keep the original intensity + the
		// left-column reading well that kept those rows legible — desktop unchanged.
		applyBackdrop(portrait: boolean): void {
			if (portrait) {
				this.field.setIntensity(0.14);
				this.field.setReadingWell({ x: 0, y: 0, hw: 1.0, hh: 1.0, floor: 0.05, soft: 0.35 });
			} else {
				this.field.setIntensity(0.22);
				this.field.setReadingWell({ x: -0.2, y: -0.1, hw: 0.78, hh: 0.9, floor: 0.06, soft: 0.25 });
			}
		}
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node) => ({ id: node.source.id, score: node.activation ?? 0.5, hue: FIELD_HUE.caution, energy: node.activation, metric2: node.trust, scar: (node.tags?.length ?? 0) > 1, kind: 'memory-pr', payload: node }));
			this.field.setCells(layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.035, maxCellR: 0.09 }));
		}
		compute(encoder: GPUCommandEncoder): void { this.field.compute(encoder); }
		render(pass: GPURenderPassEncoder): void { this.field.render(pass); }
		pickAt(x: number, y: number): RoutePick | null { return this.field.pickAt(x, y); }
		dispose(): void {
			if (fieldPass === this) fieldPass = null;
			this.field.dispose();
		}
	}

	async function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'memory-pr') return;
		// Pick can be a TEXT row (payload = MemoryPrTextItem with .prId) or a FIELD
		// cell (payload = RouteNode with .source.id == pr id). Read whichever, so
		// field cells act on the real PR, not silently no-op.
		const payload = pick.payload as Partial<MemoryPrTextItem> & { source?: { id?: string } };
		const prId = payload.prId ?? payload.source?.id;
		if (!prId) return;
		await askWhy(prId);
	}

	async function askWhy(prId: string) {
		whyForPrId = prId;
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

<!-- PORTRAIT-ONLY readable DOM overlay (content-first). On desktop (landscape) this
     organ stays zero-DOM: the in-canvas PR field IS the content. On a phone the field
     is an illegible log wall, so we surface THIS as the focal content and dim the field
     to a backdrop. Container is pointer-events-none so empty gaps still reach the field,
     while each card is pointer-events-auto. pb-28 clears the global MobileNav FAB. -->
{#if isPortrait}
<div
	class="relative z-10 mx-auto max-h-dvh max-w-3xl space-y-6 overflow-y-auto overscroll-contain p-6 pb-28 pointer-events-none"
>
	<!-- Opaque backing so the masthead + description read cleanly over the dim field
	     (the -mb-2 pulls the PageHeader's own bottom margin back inside the panel). -->
	<div class="glass-subtle pointer-events-auto rounded-2xl p-5 [&_header]:mb-0">
		<PageHeader
			icon="memorypr"
			title="Memory PRs: Review Queue"
			subtitle="Proposed changes to your memory (new facts, supersessions, merges, and forgets) held for review before they touch the graph. Tap a PR to ask the agent why it was proposed."
			accent="warning"
		>
			<span
				class="ping-host flex h-2 w-2 items-center justify-center text-warning"
				aria-hidden="true"
			>
				<span class="breathe h-2 w-2 rounded-full bg-warning"></span>
			</span>
			<span class="text-xs text-dim">Live</span>
		</PageHeader>
	</div>

	<!-- Status / count strip -->
	<div
		class="glass-panel pointer-events-auto flex flex-wrap items-center gap-3 rounded-2xl p-4 text-xs text-text"
		role="status"
		aria-live="polite"
	>
		{#if loading}
			<span class="breathe h-2 w-2 rounded-full bg-warning"></span>
			<span class="text-dim">Loading review queue…</span>
		{:else if error}
			<span class="h-2 w-2 rounded-full bg-decay"></span>
			<span class="text-decay">Queue unavailable</span>
		{:else}
			<span class="breathe h-2 w-2 rounded-full bg-warning"></span>
			<span class="tabular-nums">
				<AnimatedNumber value={prs.length} />
				{prs.length === 1 ? 'PR' : 'PRs'}
				· <AnimatedNumber value={pendingCount} /> pending review
			</span>
		{/if}
	</div>

	<!-- Results -->
	{#if error}
		<div
			class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center"
		>
			<div class="text-sm text-decay">Couldn't load memory PRs</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={() => void loadPrs()}
				class="mt-2 rounded-lg bg-warning/20 px-4 py-2 text-xs font-medium text-warning transition hover:bg-warning/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-warning/60"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<div class="pointer-events-auto space-y-3">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-20 rounded-2xl"></div>
			{/each}
		</div>
	{:else if domRows.length === 0}
		<div
			class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center"
		>
			<div
				class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall"
			>
				<Icon name="sparkle" size={26} draw />
			</div>
			<div class="text-sm font-medium text-bright">No pending memory PRs.</div>
			<div class="max-w-sm text-xs text-muted">
				Every proposed change has been reviewed. New facts and supersessions will queue here
				for your approval before they touch the graph.
			</div>
		</div>
	{:else}
		<div class="pointer-events-auto space-y-3">
			{#each domRows as pr, i (pr.id)}
				<button
					type="button"
					onclick={() => void askWhy(pr.id)}
					use:reveal={{ delay: Math.min(i * 35, 350), y: 12 }}
					use:spotlight
					class="spotlight-surface lift glass-panel block w-full rounded-2xl p-4 text-left transition hover:border-warning/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-warning/60"
				>
					<div class="flex items-start justify-between gap-3">
						<div class="min-w-0">
							<div class="truncate text-sm font-medium text-bright">{pr.title}</div>
							<div class="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-dim">
								<span class="font-mono">{pr.id.slice(0, 8)}</span>
								<span class="text-muted">·</span>
								<span class="uppercase tracking-wide">{pr.kind}</span>
								{#if pr.signals.length}
									<span class="text-muted">·</span>
									<span>{pr.signals.length} signal{pr.signals.length === 1 ? '' : 's'}</span>
								{/if}
							</div>
						</div>
						<span
							class="shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide {prStatusTone(pr.status)}"
						>
							{pr.status}
						</span>
					</div>

					{#if whyForPrId === pr.id && whySignals.length}
						<div class="mt-3 space-y-1.5 border-t border-white/[0.06] pt-3">
							{#each whySignals.slice(0, 5) as signal (signal.code)}
								<div class="flex gap-2 text-[11px]">
									<span class="shrink-0 font-mono text-warning">{signal.code}</span>
									<span class="text-muted">{signal.detail}</span>
								</div>
							{/each}
						</div>
					{/if}
				</button>
			{/each}
		</div>
	{/if}
</div>
{/if}
