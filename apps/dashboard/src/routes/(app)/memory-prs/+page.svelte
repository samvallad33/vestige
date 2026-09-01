<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type MemoryPr, type MemoryPrAction, type ReviewMode } from '$lib/stores/api';
	import { memoryPrEvents } from '$lib/stores/websocket';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import Icon, { type IconName } from '$lib/components/Icon.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { viewMemoryPrDiff } from '$lib/observatory/export/memory-pr-diff';
	import { base } from '$app/paths';

	type MemoryPrTextItem = TextLayerItem & { prId?: string };
	type WhySignal = { code: string; detail: string };

	const ROW_LIMIT = 28;
	const PR_LIMIT = 100;

	// --- Real state from the API ------------------------------------------------
	let prs: MemoryPr[] = $state([]);
	let total = $state(0);
	let pendingCount = $state(0);
	let mode = $state<ReviewMode>('risk_gated');
	let loading = $state(true);
	let error: string | null = $state(null);

	// The DOM row the user last selected + the "why" the agent returned for it.
	// Selection is NON-MUTATING: it only expands the row and shows detail.
	let selectedPrId: string | null = $state(null);
	let whyForPrId: string | null = $state(null);
	let whySignals: WhySignal[] = $state([]);
	let whyLoading = $state(false);
	// The action currently running against a PR (so its button shows a spinner
	// and we never double-fire a mutation). Keyed `${prId}:${action}`.
	let actingKey: string | null = $state(null);
	let actionNotice: string | null = $state(null);

	onMount(() => {
		void loadPrs();
	});

	async function loadPrs() {
		loading = true;
		error = null;
		try {
			const res = await api.memoryPrs.list(undefined, PR_LIMIT);
			prs = res.prs;
			total = res.total;
			pendingCount = res.pendingCount;
			mode = res.mode;
		} catch (err) {
			prs = [];
			total = 0;
			pendingCount = 0;
			error = err instanceof Error ? err.message : 'Failed to load memory PRs';
		} finally {
			loading = false;
		}
	}

	// Live: refresh the queue when the backend emits a memory-PR event.
	$effect(() => {
		if ($memoryPrEvents.length) void loadPrs();
	});

	// --- Derived real stats -----------------------------------------------------
	// The kinds actually present in the queue, so the "what's proposed" card is
	// real and never a constant. supersede / merge / forget / promote / new.
	const decidedCount = $derived(prs.filter((pr) => pr.status !== 'pending').length);
	const supersedeCount = $derived(prs.filter((pr) => pr.kind === 'supersede').length);
	const mergeCount = $derived(prs.filter((pr) => pr.kind === 'merge').length);
	const forgetCount = $derived(prs.filter((pr) => pr.kind === 'forget').length);
	// Total risk signals attached across every PR in the queue — the reason the
	// review gate exists. Zero when the queue is clean.
	const totalSignals = $derived(prs.reduce((sum, pr) => sum + pr.signals.length, 0));

	const modeLabel = $derived(
		mode === 'fast' ? 'Fast (auto-apply)' : mode === 'paranoid' ? 'Paranoid' : 'Risk-gated'
	);

	// --- Explicit, labeled mutations the client + backend both support. Each maps
	// 1:1 to an api.memoryPrs.act action. `ask_agent_why` is read-only; the rest
	// mutate and are only offered while a PR is still pending. Selection alone
	// never triggers any of these. ---
	// Static class strings per action so the Tailwind JIT scans them literally
	// (dynamically-built `border-${tone}` strings are never emitted). Only tokens
	// that exist in app.css are used: synapse / recall / warning / decay.
	type PrActionDef = {
		action: Exclude<MemoryPrAction, 'ask_agent_why'>;
		label: string;
		cls: string;
		icon: IconName;
	};
	const MUTATIONS: PrActionDef[] = [
		{
			action: 'promote',
			label: 'Approve',
			icon: 'sparkle',
			cls: 'border-recall/30 text-recall hover:bg-recall/15'
		},
		{
			action: 'merge',
			label: 'Merge',
			icon: 'duplicates',
			cls: 'border-synapse/30 text-synapse-glow hover:bg-synapse/15'
		},
		{
			action: 'supersede',
			label: 'Supersede',
			icon: 'timeline',
			cls: 'border-memory/30 text-memory hover:bg-memory/15'
		},
		{
			action: 'quarantine',
			label: 'Quarantine',
			icon: 'contradictions',
			cls: 'border-warning/30 text-warning hover:bg-warning/15'
		},
		{
			action: 'forget',
			label: 'Forget',
			icon: 'close',
			cls: 'border-decay/30 text-decay hover:bg-decay/15'
		}
	];

	function prStatusTone(status: string): string {
		if (status === 'pending') return 'text-warning border-warning/30 bg-warning/10';
		if (status === 'approved' || status === 'promoted')
			return 'text-recall border-recall/25 bg-recall/10';
		if (status === 'rejected' || status === 'forgotten' || status === 'quarantined')
			return 'text-decay border-decay/25 bg-decay/10';
		return 'text-dim border-white/10 bg-white/[0.04]';
	}

	// NON-MUTATING: expand/collapse a row. Never calls act().
	function selectPr(id: string) {
		selectedPrId = selectedPrId === id ? null : id;
	}

	// Read-only agent explanation for one PR.
	async function askWhy(prId: string) {
		selectedPrId = prId;
		whyForPrId = prId;
		whyLoading = true;
		whySignals = [];
		try {
			const res = (await api.memoryPrs.act(prId, 'ask_agent_why')) as { why?: WhySignal[] };
			whySignals = res.why ?? [];
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to ask the agent why';
		} finally {
			whyLoading = false;
		}
	}

	// Explicit, labeled mutation — only from a button click, never from selection.
	async function runAction(prId: string, action: PrActionDef['action'], label: string) {
		const key = `${prId}:${action}`;
		if (actingKey) return;
		actingKey = key;
		actionNotice = null;
		try {
			await api.memoryPrs.act(prId, action);
			actionNotice = `${label} applied — refreshing queue.`;
			await loadPrs();
		} catch (err) {
			error = err instanceof Error ? err.message : `Failed to ${label.toLowerCase()} this PR`;
		} finally {
			actingKey = null;
		}
	}

	// ==========================================================================
	//  WebGPU field + MSDF text layer — the ALIVE backdrop. Preserved verbatim
	//  from the original page; the DOM overlay now reads on top of it.
	// ==========================================================================
	let fieldPass: MemoryPrFieldPass | null = null;

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
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
		const fromDiff = numericField(pr.diff, [
			'confidence',
			'trust',
			'contradictsTrust',
			'contradicts_trust'
		]);
		if (fromDiff !== null) return clamp01(fromDiff > 1 ? fromDiff / 100 : fromDiff);
		return 0.5;
	}

	function prLine(pr: MemoryPr): string {
		return sanitizeAscii(`${pr.title} | ${pr.id.slice(0, 8)} | ${pr.status}`)
			.replace(/\s+/g, ' ')
			.trim()
			.slice(0, 96);
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

	function createMemoryPrPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		const field = new MemoryPrFieldPass(engine);
		field.applyBackdrop();
		field.uploadScene(scene);
		fieldPass = field;
		// The DOM overlay owns all readable content (header + stat cards + queue
		// rows). We deliberately emit NO in-canvas MSDF text so the field never
		// doubles as a raw PR-log dump bleeding through behind the glass — same
		// outcome as /explore, whose buildTextItems() returns []. The field pass
		// stays: it is the alive backdrop.
		return [field];
	}

	class MemoryPrFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
		}
		// The DOM overlay is the readable hero, so the field is a faint full-frame
		// ambient substrate behind the glass cards on every viewport.
		applyBackdrop(): void {
			this.field.setIntensity(0.16);
			this.field.setReadingWell({ x: 0, y: 0, hw: 1.0, hh: 1.0, floor: 0.05, soft: 0.35 });
		}
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node) => ({
				id: node.source.id,
				score: node.activation ?? 0.5,
				hue: FIELD_HUE.caution,
				energy: node.activation,
				metric2: node.trust,
				scar: (node.tags?.length ?? 0) > 1,
				kind: 'memory-pr',
				payload: node
			}));
			this.field.setCells(layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.035, maxCellR: 0.09 }));
		}
		compute(encoder: GPUCommandEncoder): void {
			this.field.compute(encoder);
		}
		render(pass: GPURenderPassEncoder): void {
			this.field.render(pass);
		}
		pickAt(x: number, y: number): RoutePick | null {
			return this.field.pickAt(x, y);
		}
		dispose(): void {
			if (fieldPass === this) fieldPass = null;
			this.field.dispose();
		}
	}

	// Picking a field cell or a text row is NON-MUTATING: it selects + scrolls the
	// DOM card into focus and asks the agent why (read-only). No mutation ever
	// fires from a pick — mutations are the labeled buttons only.
	function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'memory-pr') return;
		const payload = pick.payload as Partial<MemoryPrTextItem> & { source?: { id?: string } };
		const prId = payload.prId ?? payload.source?.id;
		if (!prId) return;
		void askWhy(prId);
		if (typeof document !== 'undefined') {
			document.getElementById(`pr-card-${prId}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
		}
	}
</script>

<svelte:head>
	<title>Memory Pull Requests · Vestige</title>
</svelte:head>

<RouteStage
	organ="memory-prs"
	seed={`memory-pr-field:${prs.length}:${whySignals.length}`}
	scene={memoryPrScene}
	passes={createMemoryPrPasses}
	{loading}
	{error}
	emptyLabel=""
	onpick={handleRoutePick}
/>

<!-- DOM-hybrid overlay (contradictions pattern): RouteStage renders the WebGPU
     field behind; this reads on top. Container is pointer-events-none so empty
     gaps still reach the field, every interactive child is pointer-events-auto. -->
<div class="relative z-10 min-h-full p-6 space-y-6 pointer-events-none">
	<!-- (1) IDENTITY -->
	<div class="pointer-events-auto">
		<PageHeader
			icon="memorypr"
			title="Memory Pull Requests"
			subtitle="Proposed changes to your memory (supersede / merge / forget) awaiting your review before they touch the graph."
			accent="warning"
		>
			<span
				class="ping-host flex h-2 w-2 items-center justify-center text-warning"
				aria-hidden="true"
			>
				<span class="breathe h-2 w-2 rounded-full bg-warning"></span>
			</span>
			<span class="text-dim text-sm tabular-nums inline-flex items-center gap-1.5">
				<AnimatedNumber value={pendingCount} /> pending
			</span>
		</PageHeader>
	</div>

	{#if error}
		<!-- (5) STATE GUIDANCE — error -->
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
		<!-- (5) STATE GUIDANCE — loading skeletons -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-20 rounded-xl"></div>
			{/each}
		</div>
		<div class="pointer-events-auto space-y-3">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-24 rounded-2xl"></div>
			{/each}
		</div>
	{:else if prs.length === 0}
		<!-- (5) STATE GUIDANCE — empty: icon + explanation + exact next action -->
		<div
			class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center"
		>
			<div
				class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall"
			>
				<Icon name="sparkle" size={26} draw />
			</div>
			<div class="text-sm font-medium text-bright">
				No memory PRs — nothing is proposing to change your memory.
			</div>
			<div class="max-w-md text-xs text-muted">
				When an agent wants to supersede, merge, or forget one of your memories, the change is
				held here as a pull request first. Keep working with your agent; risky brain-changes will
				queue here for your approval instead of applying silently.
			</div>
		</div>
	{:else}
		<!-- (2) LIVE PROOF — 4 oversized real stat cards -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={total} />
				</div>
				<div class="text-xs text-dim mt-1">pull requests total</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="flex items-center gap-2">
					<span class="ping-host inline-flex">
						<span class="w-2 h-2 rounded-full bg-warning"></span>
					</span>
					<div class="text-2xl font-bold tabular-nums text-warning">
						<AnimatedNumber value={pendingCount} />
					</div>
				</div>
				<div class="text-xs text-dim mt-1">awaiting your review</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={totalSignals} />
				</div>
				<div class="text-xs text-dim mt-1">risk signals flagged</div>
			</div>
			<div use:reveal={{ delay: 180, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums capitalize">{modeLabel}</div>
				<div class="text-xs text-dim mt-1">review gate mode</div>
			</div>
		</div>

		<!-- (6) INTERPRETATION — what a Memory PR is + the live kind breakdown -->
		<div
			use:reveal={{ delay: 220, y: 12 }}
			class="glass-subtle pointer-events-auto rounded-2xl p-4 text-xs text-muted"
		>
			<div class="flex flex-wrap items-center gap-x-4 gap-y-2">
				<span class="text-dim">
					A <span class="text-text font-medium">Memory PR</span> is a proposed brain-change your
					agent wants to make — supersede an outdated fact, merge duplicates, or forget something —
					held here for review instead of applied silently.
				</span>
				<span class="ml-auto flex flex-wrap items-center gap-3 tabular-nums">
					<span><span class="text-memory font-medium">{supersedeCount}</span> supersede</span>
					<span><span class="text-synapse-glow font-medium">{mergeCount}</span> merge</span>
					<span><span class="text-decay font-medium">{forgetCount}</span> forget</span>
					<span><span class="text-recall font-medium">{decidedCount}</span> decided</span>
				</span>
			</div>
		</div>

		{#if actionNotice}
			<div
				class="glass-subtle pointer-events-auto rounded-xl px-4 py-2 text-xs text-recall"
				role="status"
				aria-live="polite"
			>
				{actionNotice}
			</div>
		{/if}

		<!-- (3) PRIMARY ACTION lives per-row as labeled buttons + the review queue. -->
		<div class="pointer-events-auto space-y-3">
			{#each prs as pr, i (pr.id)}
				{@const isSelected = selectedPrId === pr.id}
				{@const isPending = pr.status === 'pending'}
				{@const diff = viewMemoryPrDiff(pr.diff)}
				<div
					id={`pr-card-${pr.id}`}
					use:reveal={{ delay: Math.min(i * 30, 300), y: 12 }}
					class="glass-panel lift rounded-2xl p-4 transition
						{isSelected ? 'border-warning/40 shadow-[0_0_18px_rgba(245,158,11,0.14)]' : ''}"
				>
					<!-- Row header — NON-MUTATING select (expand/collapse only). -->
					<button
						type="button"
						onclick={() => selectPr(pr.id)}
						class="block w-full text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-warning/50 rounded-lg"
						aria-expanded={isSelected}
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
										<span class="text-warning"
											>{pr.signals.length} signal{pr.signals.length === 1 ? '' : 's'}</span
										>
									{/if}
									<span class="text-muted">·</span>
									<span>{new Date(pr.created_at).toLocaleDateString()}</span>
								</div>
							</div>
							<span
								class="shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide {prStatusTone(
									pr.status
								)}"
							>
								{pr.status}
							</span>
						</div>
					</button>

					<!-- (6) INTERPRETATION — expanded signals + agent "why". -->
					{#if isSelected}
						<div class="mt-3 space-y-3 border-t border-white/[0.06] pt-3">
							{#if pr.signals.length}
								<div class="space-y-1.5">
									<div class="text-[10px] uppercase tracking-wider text-muted">Risk signals</div>
									{#each pr.signals as signal (signal.code)}
										<div class="flex gap-2 text-[11px]">
											<span class="shrink-0 font-mono text-warning">{signal.code}</span>
											<span class="text-muted">{signal.detail}</span>
										</div>
									{/each}
								</div>
							{:else}
								<div class="text-[11px] text-muted">No risk signals attached to this PR.</div>
							{/if}

							{#if whyForPrId === pr.id}
								<div class="space-y-1.5">
									<div class="text-[10px] uppercase tracking-wider text-muted">Agent explanation</div>
									{#if whyLoading}
										<div class="text-[11px] text-dim">Asking the agent…</div>
									{:else if whySignals.length}
										{#each whySignals.slice(0, 5) as signal (signal.code)}
											<div class="flex gap-2 text-[11px]">
												<span class="shrink-0 font-mono text-recall">{signal.code}</span>
												<span class="text-muted">{signal.detail}</span>
											</div>
										{/each}
									{:else}
										<div class="text-[11px] text-muted">The agent returned no extra reasoning.</div>
									{/if}
								</div>
							{/if}

							<div class="space-y-2 rounded-xl border border-white/[0.06] bg-black/20 p-3">
								<div class="text-[10px] uppercase tracking-wider text-muted">Proposed change</div>
								{#if diff.targetId}
									<a class="block font-mono text-[11px] text-synapse-glow hover:underline" href={`${base}/memories?memory=${encodeURIComponent(diff.targetId)}`}>target {diff.targetId}</a>
								{/if}
								{#if diff.before}
									<div class="text-[11px]"><span class="text-muted">Before:</span> {diff.before}</div>
								{/if}
								{#if diff.proposed || diff.after}
									<div class="text-[11px] text-bright"><span class="text-muted">Proposed:</span> {diff.proposed ?? diff.after}</div>
								{/if}
								{#if !diff.proposed && !diff.after && !diff.before}
									<div class="text-[11px] text-muted">No content diff was attached to this PR.</div>
								{/if}
								{#each diff.rest.slice(0, 4) as row (row.key)}
									<div class="flex gap-2 text-[10px] font-mono text-dim"><span>{row.key}</span><span class="ml-auto break-all">{row.value.slice(0, 120)}</span></div>
								{/each}
							</div>
						</div>
					{/if}

					<!-- ACTIONS — explicit labeled buttons. Read-only "Ask why" always;
					     mutations only while pending. Nothing here fires from selection. -->
					<div class="mt-3 flex flex-wrap items-center gap-2 border-t border-white/[0.06] pt-3">
						<button
							type="button"
							onclick={() => void askWhy(pr.id)}
							class="inline-flex items-center gap-1.5 rounded-lg border border-white/12 px-3 py-1.5 text-xs text-dim transition hover:text-text hover:border-warning/30 hover:bg-white/[0.03] focus:outline-none focus-visible:ring-2 focus-visible:ring-warning/50"
						>
							<Icon name="sparkle" size={13} />
							Ask agent why
						</button>

						{#if isPending}
							{#each MUTATIONS as m (m.action)}
								{@const key = `${pr.id}:${m.action}`}
								<button
									type="button"
									disabled={actingKey !== null}
									onclick={() => void runAction(pr.id, m.action, m.label)}
									class="inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition
										focus:outline-none focus-visible:ring-2 focus-visible:ring-warning/50
										disabled:opacity-40 disabled:cursor-not-allowed {m.cls}"
								>
									<Icon name={m.icon} size={13} />
									{actingKey === key ? `${m.label}…` : m.label}
								</button>
							{/each}
						{:else}
							<span class="text-[11px] text-muted" title="Only pending PRs can be acted on">
								Decided{pr.decided_at
									? ` ${new Date(pr.decided_at).toLocaleDateString()}`
									: ''} — no actions available
							</span>
						{/if}
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
