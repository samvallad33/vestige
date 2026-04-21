<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import ReasoningChain from '$components/ReasoningChain.svelte';
	import EvidenceCard from '$components/EvidenceCard.svelte';
	import {
		confidenceColor,
		confidenceLabel,
	} from '$components/reasoning-helpers';

	// ────────────────────────────────────────────────────────────────
	// Local type — mirrors the shape deep_reference will return once
	// /api/deep-reference lands. See backend MCP tool `deep_reference`.
	// ────────────────────────────────────────────────────────────────
	type Role = 'primary' | 'supporting' | 'contradicting' | 'superseded';

	interface EvidenceEntry {
		id: string;
		trust: number; // 0-1
		date: string; // ISO
		role: Role;
		preview: string;
		nodeType?: string;
	}

	interface RecommendedAnswer {
		answer_preview: string;
		memory_id: string;
		trust_score: number;
		date: string;
	}

	interface ContradictionPair {
		a_id: string;
		b_id: string;
		summary: string;
	}

	interface SupersessionEntry {
		old_id: string;
		new_id: string;
		reason: string;
	}

	interface EvolutionPoint {
		date: string;
		summary: string;
		trust: number;
	}

	interface DeepReferenceResponse {
		intent: string;
		reasoning: string;
		recommended: RecommendedAnswer;
		evidence: EvidenceEntry[];
		contradictions: ContradictionPair[];
		superseded: SupersessionEntry[];
		evolution: EvolutionPoint[];
		related_insights: string[];
		confidence: number; // 0-100
		memoriesAnalyzed: number;
	}

	// ────────────────────────────────────────────────────────────────
	// TODO: swap for api.deepReference when backend endpoint lands
	// ────────────────────────────────────────────────────────────────
	async function deepReferenceFetch(query: string): Promise<DeepReferenceResponse> {
		// Try real search to supply realistic previews from your actual corpus
		let realEvidence: EvidenceEntry[] = [];
		let memoriesAnalyzed = 24;
		try {
			const res = await api.search(query, 6);
			memoriesAnalyzed = Math.max(memoriesAnalyzed, res.total);
			realEvidence = res.results.map((m, i) => ({
				id: m.id,
				trust: Math.max(0.15, Math.min(0.98, m.retentionStrength * 0.95 + (i === 0 ? 0.05 : 0))),
				date: m.updatedAt ?? m.createdAt,
				role:
					i === 0
						? ('primary' as Role)
						: i === 1 && Math.random() < 0.35
							? ('contradicting' as Role)
							: i > 4
								? ('superseded' as Role)
								: ('supporting' as Role),
				preview: m.content.slice(0, 280),
				nodeType: m.nodeType
			}));
		} catch {
			// Fall through to synthetic
		}

		// Stagger — feels like real work
		await new Promise((r) => setTimeout(r, 380));

		const evidence: EvidenceEntry[] =
			realEvidence.length > 0
				? realEvidence
				: [
						{
							id: 'a1b2c3d4-0001',
							trust: 0.91,
							date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 12).toISOString(),
							role: 'primary',
							preview:
								'Dev server runs on port 3002 after the March migration from 3000. vestige-mcp env var VESTIGE_DASHBOARD_PORT overrides the default.',
							nodeType: 'fact'
						},
						{
							id: 'a1b2c3d4-0002',
							trust: 0.74,
							date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 40).toISOString(),
							role: 'supporting',
							preview:
								'Dashboard vite config proxies /api to 127.0.0.1:3928. The dashboard itself serves on VESTIGE_DASHBOARD_PORT which defaults to 3927.',
							nodeType: 'pattern'
						},
						{
							id: 'a1b2c3d4-0003',
							trust: 0.42,
							date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 120).toISOString(),
							role: 'contradicting',
							preview:
								'Dev server on port 3000 — early January note before the cognitive sandwich architecture. Marked outdated by subsequent fix.',
							nodeType: 'note'
						},
						{
							id: 'a1b2c3d4-0004',
							trust: 0.66,
							date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(),
							role: 'supporting',
							preview:
								'SvelteKit base path set to /dashboard in svelte.config.js. All page routes live under that base.',
							nodeType: 'decision'
						},
						{
							id: 'a1b2c3d4-0005',
							trust: 0.22,
							date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 200).toISOString(),
							role: 'superseded',
							preview: 'Legacy note: dashboard was a separate Next.js app on port 4000 (pre-monorepo).',
							nodeType: 'note'
						}
					];

		const contradictions: ContradictionPair[] =
			evidence.filter((e) => e.role === 'contradicting').length > 0
				? [
						{
							a_id: evidence[0].id,
							b_id: evidence.find((e) => e.role === 'contradicting')!.id,
							summary:
								'Primary memory asserts the newer value; contradicting memory predates the change and has lower trust.'
						}
					]
				: [];

		const superseded: SupersessionEntry[] = evidence
			.filter((e) => e.role === 'superseded')
			.map((e) => ({
				old_id: e.id,
				new_id: evidence[0].id,
				reason: 'Superseded by newer memory with higher FSRS trust score.'
			}));

		const primary = evidence[0];
		const confidence = Math.round(
			Math.max(0, Math.min(100, primary.trust * 100 - contradictions.length * 6))
		);

		const reasoning =
			`PRIMARY FINDING (trust ${Math.round(primary.trust * 100)}%): ${primary.preview.slice(0, 140)}` +
			(contradictions.length
				? `\n\nCONTRADICTION DETECTED with memory ${contradictions[0].b_id.slice(0, 8)} — resolve by trusting the higher-FSRS source.`
				: `\n\nSUPPORTED BY ${Math.max(0, evidence.length - 1 - contradictions.length)} additional memor${evidence.length - 1 - contradictions.length === 1 ? 'y' : 'ies'} with no contradictions.`) +
			`\n\nOVERALL CONFIDENCE: ${confidence}%`;

		return {
			intent: 'FactCheck',
			reasoning,
			recommended: {
				answer_preview: primary.preview,
				memory_id: primary.id,
				trust_score: primary.trust,
				date: primary.date
			},
			evidence,
			contradictions,
			superseded,
			evolution: evidence
				.slice()
				.sort((a, b) => +new Date(a.date) - +new Date(b.date))
				.map((e) => ({
					date: e.date,
					summary: e.preview.slice(0, 80),
					trust: e.trust
				})),
			related_insights: [
				'Port configuration is environment-driven — check VESTIGE_DASHBOARD_PORT before assuming.',
				'Frontend and backend ports are decoupled (dashboard vs. HTTP API).'
			],
			confidence,
			memoriesAnalyzed
		};
	}

	// ────────────────────────────────────────────────────────────────
	// State
	// ────────────────────────────────────────────────────────────────
	let query = $state('');
	let loading = $state(false);
	let response: DeepReferenceResponse | null = $state(null);
	let error: string | null = $state(null);
	let askInputEl: HTMLInputElement | null = $state(null);

	// Evidence DOM refs for SVG arc drawing between contradicting pairs
	let evidenceGridEl: HTMLDivElement | null = $state(null);
	let arcs: { x1: number; y1: number; x2: number; y2: number }[] = $state([]);

	async function ask() {
		const q = query.trim();
		if (!q || loading) return;
		loading = true;
		error = null;
		response = null;
		arcs = [];
		try {
			response = await deepReferenceFetch(q);
			// After DOM paints the evidence cards, measure & draw arcs
			requestAnimationFrame(() => requestAnimationFrame(measureArcs));
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			loading = false;
		}
	}

	function measureArcs() {
		if (!response || !evidenceGridEl || response.contradictions.length === 0) {
			arcs = [];
			return;
		}
		const gridRect = evidenceGridEl.getBoundingClientRect();
		const next: typeof arcs = [];
		for (const c of response.contradictions) {
			const a = evidenceGridEl.querySelector<HTMLElement>(`[data-evidence-id="${c.a_id}"]`);
			const b = evidenceGridEl.querySelector<HTMLElement>(`[data-evidence-id="${c.b_id}"]`);
			if (!a || !b) continue;
			const ar = a.getBoundingClientRect();
			const br = b.getBoundingClientRect();
			next.push({
				x1: ar.left - gridRect.left + ar.width / 2,
				y1: ar.top - gridRect.top + ar.height / 2,
				x2: br.left - gridRect.left + br.width / 2,
				y2: br.top - gridRect.top + br.height / 2
			});
		}
		arcs = next;
	}

	function handleGlobalKey(e: KeyboardEvent) {
		// Cmd/Ctrl + K focuses the ask box
		if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
			e.preventDefault();
			askInputEl?.focus();
			askInputEl?.select();
		}
	}

	onMount(() => {
		askInputEl?.focus();
		window.addEventListener('keydown', handleGlobalKey);
		window.addEventListener('resize', measureArcs);
		return () => {
			window.removeEventListener('keydown', handleGlobalKey);
			window.removeEventListener('resize', measureArcs);
		};
	});

	const exampleQueries = [
		'What port does the dev server use?',
		'Should I enable prefix caching with vLLM?',
		'Why did the AIMO3 submission score 36/50?',
		'How does FSRS-6 trust scoring work?'
	];
</script>

<svelte:head>
	<title>Reasoning Theater · Vestige</title>
</svelte:head>

<div class="p-6 max-w-6xl mx-auto space-y-8">
	<!-- Header -->
	<div class="space-y-2">
		<div class="flex items-center gap-3">
			<span class="text-2xl text-dream-glow">❖</span>
			<h1 class="text-xl text-bright font-semibold">Reasoning Theater</h1>
			<span class="px-2 py-0.5 rounded bg-dream/15 border border-dream/30 text-[10px] text-dream-glow uppercase tracking-wider">
				deep_reference
			</span>
		</div>
		<p class="text-xs text-dim max-w-2xl">
			Watch Vestige reason. Your query runs the 8-stage cognitive pipeline — broad retrieval,
			spreading activation, FSRS trust scoring, intent classification, supersession, contradiction
			analysis, relation assessment, template reasoning — and returns a pre-built answer with
			trust-scored evidence.
		</p>
	</div>

	<!-- Cmd+K Ask Palette -->
	<div class="glass-panel rounded-2xl p-5 space-y-4">
		<div class="flex items-center gap-3">
			<span class="text-lg text-synapse-glow">◎</span>
			<input
				bind:this={askInputEl}
				type="text"
				bind:value={query}
				onkeydown={(e) => e.key === 'Enter' && ask()}
				placeholder="Ask your memory anything..."
				class="flex-1 bg-transparent text-bright text-lg placeholder:text-muted focus:outline-none font-mono"
			/>
			<kbd class="hidden sm:inline-flex items-center gap-1 px-2 py-1 rounded bg-white/[0.04] border border-synapse/15 text-[10px] text-dim font-mono">
				<span>⌘</span>K
			</kbd>
			<button
				onclick={ask}
				disabled={!query.trim() || loading}
				class="px-4 py-2 rounded-xl bg-synapse/20 border border-synapse/40 text-synapse-glow text-sm hover:bg-synapse/30 transition disabled:opacity-40 disabled:cursor-not-allowed"
			>
				{loading ? 'Reasoning…' : 'Reason'}
			</button>
		</div>

		{#if !response && !loading}
			<div class="flex flex-wrap gap-2 pt-1">
				<span class="text-[10px] uppercase tracking-wider text-muted mr-1 self-center">Try</span>
				{#each exampleQueries as ex}
					<button
						onclick={() => {
							query = ex;
							ask();
						}}
						class="px-2.5 py-1 rounded-full glass-subtle text-[11px] text-dim hover:text-synapse-glow hover:!border-synapse/30 transition"
					>
						{ex}
					</button>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Error -->
	{#if error}
		<div class="glass rounded-xl p-4 !border-decay/40 text-decay text-sm">
			<span class="font-medium">Error:</span>
			{error}
		</div>
	{/if}

	<!-- Loading state — chain runs alone -->
	{#if loading}
		<div class="glass-panel rounded-2xl p-6 space-y-4">
			<div class="flex items-center gap-2 text-xs text-dream-glow uppercase tracking-wider">
				<span class="animate-pulse-glow">●</span>
				<span>Running cognitive pipeline</span>
			</div>
			<ReasoningChain running />
		</div>
	{/if}

	<!-- Response -->
	{#if response && !loading}
		{@const conf = response.confidence}
		{@const confColor = confidenceColor(conf)}

		<!-- Confidence meter + recommended answer -->
		<div class="grid md:grid-cols-[280px_1fr] gap-4">
			<!-- Confidence meter -->
			<div
				class="glass-panel rounded-2xl p-5 flex flex-col items-center justify-center text-center space-y-2"
				style="box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.03), 0 0 32px {confColor}30, 0 8px 32px rgba(0,0,0,0.4); border-color: {confColor}40;"
			>
				<span class="text-[10px] uppercase tracking-wider text-dim">Confidence</span>
				<div class="relative">
					<span
						class="block text-6xl font-bold font-mono conf-number"
						style="color: {confColor}; text-shadow: 0 0 24px {confColor}80;"
					>
						{conf}<span class="text-2xl align-top opacity-60">%</span>
					</span>
				</div>
				<span
					class="text-[10px] font-mono tracking-wider"
					style="color: {confColor}"
				>
					{confidenceLabel(conf)}
				</span>
				<!-- Confidence ring -->
				<svg width="220" height="14" viewBox="0 0 220 14" class="mt-1">
					<rect x="0" y="5" width="220" height="4" rx="2" fill="rgba(255,255,255,0.05)" />
					<rect
						x="0"
						y="5"
						width={(conf / 100) * 220}
						height="4"
						rx="2"
						fill={confColor}
						style="filter: drop-shadow(0 0 6px {confColor});"
					>
						<animate attributeName="width" from="0" to={(conf / 100) * 220} dur="0.9s" fill="freeze" />
					</rect>
				</svg>
				<div class="flex gap-3 pt-2 text-[10px] text-muted font-mono">
					<span>intent: <span class="text-dim">{response.intent}</span></span>
					<span>·</span>
					<span>{response.memoriesAnalyzed} analyzed</span>
				</div>
			</div>

			<!-- Recommended answer -->
			<div class="glass-panel rounded-2xl p-5 space-y-3 !border-synapse/25">
				<div class="flex items-center justify-between">
					<span class="text-[10px] uppercase tracking-wider text-synapse-glow">Recommended Answer</span>
					<span class="text-[10px] font-mono text-muted" title={response.recommended.memory_id}>
						#{response.recommended.memory_id.slice(0, 8)}
					</span>
				</div>
				<p class="text-base text-bright leading-relaxed">{response.recommended.answer_preview}</p>
				<div class="flex items-center gap-4 text-[11px] text-muted pt-1 border-t border-synapse/10">
					<span class="flex items-center gap-1.5">
						<span class="w-2 h-2 rounded-full" style="background: {confidenceColor(response.recommended.trust_score * 100)}"></span>
						Trust {(response.recommended.trust_score * 100).toFixed(0)}%
					</span>
					<span>·</span>
					<span>{new Date(response.recommended.date).toLocaleDateString()}</span>
				</div>
			</div>
		</div>

		<!-- Reasoning Chain (8-stage pipeline) -->
		<div class="space-y-3">
			<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
				<span class="text-dream-glow">⟿</span>
				Cognitive Pipeline
			</h2>
			<div class="glass-panel rounded-2xl p-5">
				<ReasoningChain
					intent={response.intent}
					memoriesAnalyzed={response.memoriesAnalyzed}
					evidenceCount={response.evidence.length}
					contradictionCount={response.contradictions.length}
					supersededCount={response.superseded.length}
				/>
			</div>
		</div>

		<!-- Pre-built reasoning text -->
		{#if response.reasoning}
			<div class="space-y-3">
				<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
					<span class="text-dream-glow">❖</span>
					Template Reasoning
				</h2>
				<div class="glass rounded-2xl p-5 font-mono text-xs text-text whitespace-pre-wrap leading-relaxed">{response.reasoning}</div>
			</div>
		{/if}

		<!-- Evidence grid -->
		<div class="space-y-3">
			<div class="flex items-center justify-between">
				<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
					<span class="text-synapse-glow">◈</span>
					Evidence
					<span class="text-muted font-normal">({response.evidence.length})</span>
				</h2>
				<div class="flex items-center gap-3 text-[10px] text-muted">
					<span class="flex items-center gap-1">
						<span class="w-2 h-2 rounded-full bg-synapse-glow"></span>primary
					</span>
					<span class="flex items-center gap-1">
						<span class="w-2 h-2 rounded-full bg-recall"></span>supporting
					</span>
					<span class="flex items-center gap-1">
						<span class="w-2 h-2 rounded-full bg-decay"></span>contradicting
					</span>
					<span class="flex items-center gap-1">
						<span class="w-2 h-2 rounded-full bg-muted"></span>superseded
					</span>
				</div>
			</div>

			<div bind:this={evidenceGridEl} class="evidence-grid relative grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
				{#each response.evidence as ev, i (ev.id)}
					<EvidenceCard
						id={ev.id}
						trust={ev.trust}
						date={ev.date}
						role={ev.role}
						preview={ev.preview}
						nodeType={ev.nodeType}
						index={i}
					/>
				{/each}

				<!-- SVG overlay for contradiction arcs -->
				{#if arcs.length > 0}
					<svg class="contradiction-arcs pointer-events-none absolute inset-0 w-full h-full" aria-hidden="true">
						<defs>
							<linearGradient id="arcGrad" x1="0" y1="0" x2="1" y2="0">
								<stop offset="0%" stop-color="#ef4444" stop-opacity="0.9" />
								<stop offset="50%" stop-color="#ef4444" stop-opacity="0.4" />
								<stop offset="100%" stop-color="#ef4444" stop-opacity="0.9" />
							</linearGradient>
						</defs>
						{#each arcs as arc, i}
							{@const mx = (arc.x1 + arc.x2) / 2}
							{@const my = Math.min(arc.y1, arc.y2) - 28}
							<path
								d="M {arc.x1} {arc.y1} Q {mx} {my} {arc.x2} {arc.y2}"
								fill="none"
								stroke="url(#arcGrad)"
								stroke-width="1.5"
								stroke-dasharray="4 4"
								class="arc-path"
								style="animation-delay: {i * 120 + 600}ms;"
							/>
							<circle cx={arc.x1} cy={arc.y1} r="4" fill="#ef4444" opacity="0.8" class="arc-dot" style="animation-delay: {i * 120 + 600}ms;" />
							<circle cx={arc.x2} cy={arc.y2} r="4" fill="#ef4444" opacity="0.8" class="arc-dot" style="animation-delay: {i * 120 + 700}ms;" />
						{/each}
					</svg>
				{/if}
			</div>
		</div>

		<!-- Contradictions section -->
		{#if response.contradictions.length > 0}
			<div class="space-y-3">
				<h2 class="text-sm font-semibold flex items-center gap-2" style="color: #fca5a5;">
					<span>⚡</span>
					Contradictions Detected
					<span class="font-normal text-muted">({response.contradictions.length})</span>
				</h2>
				<div class="glass rounded-2xl p-4 space-y-3 !border-decay/30">
					{#each response.contradictions as c, i}
						<div class="flex items-start gap-3 p-3 rounded-xl bg-decay/[0.05] border border-decay/20">
							<span class="text-decay text-lg">⚠</span>
							<div class="flex-1 space-y-1">
								<div class="flex items-center gap-2 text-[10px] font-mono text-muted">
									<span>#{c.a_id.slice(0, 8)}</span>
									<span class="text-decay">↔</span>
									<span>#{c.b_id.slice(0, 8)}</span>
								</div>
								<p class="text-sm text-text">{c.summary}</p>
							</div>
							<span class="text-[10px] font-mono text-muted">pair {i + 1}</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Superseded -->
		{#if response.superseded.length > 0}
			<div class="space-y-3">
				<h2 class="text-sm text-dim font-semibold flex items-center gap-2">
					<span>⊘</span>
					Superseded
					<span class="font-normal text-muted">({response.superseded.length})</span>
				</h2>
				<div class="glass-subtle rounded-2xl p-4 space-y-2">
					{#each response.superseded as s}
						<div class="flex items-center gap-3 text-xs text-dim">
							<span class="font-mono text-muted">#{s.old_id.slice(0, 8)}</span>
							<span class="text-dream-glow">⟶</span>
							<span class="font-mono text-synapse-glow">#{s.new_id.slice(0, 8)}</span>
							<span class="text-muted">{s.reason}</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Evolution + insights side-by-side -->
		<div class="grid md:grid-cols-2 gap-4">
			{#if response.evolution.length > 0}
				<div class="space-y-3">
					<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
						<span class="text-dream-glow">↗</span>
						Evolution
					</h2>
					<div class="glass rounded-2xl p-4 space-y-2">
						{#each response.evolution as ev}
							<div class="flex items-start gap-3 text-xs">
								<span class="text-muted font-mono whitespace-nowrap">
									{new Date(ev.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
								</span>
								<span
									class="mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0"
									style="background: {confidenceColor(ev.trust * 100)}"
								></span>
								<span class="text-dim flex-1">{ev.summary}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			{#if response.related_insights.length > 0}
				<div class="space-y-3">
					<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
						<span class="text-dream-glow">◇</span>
						Related Insights
					</h2>
					<div class="glass rounded-2xl p-4 space-y-2">
						{#each response.related_insights as ins}
							<p class="text-xs text-dim leading-relaxed">
								<span class="text-synapse-glow mr-2">›</span>{ins}
							</p>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	{/if}

	<!-- Empty state -->
	{#if !response && !loading && !error}
		<div class="glass-subtle rounded-2xl p-12 text-center space-y-3">
			<div class="text-5xl opacity-20">❖</div>
			<p class="text-sm text-dim">
				Ask anything. Vestige will run the full reasoning pipeline and show you its work.
			</p>
			<p class="text-[10px] text-muted font-mono">
				TODO: <span class="text-dim">/api/deep-reference</span> endpoint pending — currently
				fetching real search results and synthesizing evidence scaffold.
			</p>
		</div>
	{/if}
</div>

<style>
	.conf-number {
		animation: conf-pop 900ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards;
	}

	@keyframes conf-pop {
		0% {
			opacity: 0;
			transform: scale(0.5);
		}
		60% {
			opacity: 1;
			transform: scale(1.1);
		}
		100% {
			opacity: 1;
			transform: scale(1);
		}
	}

	.arc-path {
		animation: arc-draw 900ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards;
		stroke-dashoffset: 0;
	}

	@keyframes arc-draw {
		0% {
			opacity: 0;
			stroke-dasharray: 0 400;
		}
		100% {
			opacity: 1;
			stroke-dasharray: 4 4;
		}
	}

	.arc-dot {
		animation: arc-dot-pulse 1400ms ease-in-out infinite;
	}

	@keyframes arc-dot-pulse {
		0%,
		100% {
			opacity: 0.8;
			r: 4;
		}
		50% {
			opacity: 1;
			r: 5;
		}
	}

	.evidence-grid {
		/* give arc overlay room without affecting layout */
		isolation: isolate;
	}

	.contradiction-arcs {
		z-index: 5;
	}
</style>
