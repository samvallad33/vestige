<script lang="ts">
	// ═══════════════════════════════════════════════════════════════════════
	//  MEMORY PRs — approve changes to an agent's brain like code.
	// ───────────────────────────────────────────────────────────────────────
	//  Vestige auto-remembers ordinary context, but opens a Memory PR when the
	//  agent tries to rewrite its own brain. This is the cognitive immune
	//  system: a GitHub-style diff UI for cognition, with Promote / Merge /
	//  Supersede / Quarantine / Forget / Ask Agent Why, plus a one-click
	//  Fast / Risk-Gated / Paranoid mode toggle.
	// ═══════════════════════════════════════════════════════════════════════
	import { onMount } from 'svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { toasts } from '$lib/stores/toast';
	import {
		api,
		type MemoryPr,
		type MemoryPrAction,
		type ReviewMode
	} from '$lib/stores/api';
	import { memoryPrEvents } from '$lib/stores/websocket';

	let prs = $state<MemoryPr[]>([]);
	let pendingCount = $state(0);
	let mode = $state<ReviewMode>('risk_gated');
	let statusFilter = $state<string>('pending');
	let selected = $state<MemoryPr | null>(null);
	let why = $state<{ code: string; detail: string }[] | null>(null);
	let loading = $state(false);
	let busy = $state<string | null>(null);

	const modes: { id: ReviewMode; label: string; blurb: string }[] = [
		{ id: 'fast', label: 'Fast', blurb: 'Every write auto-lands. No review.' },
		{
			id: 'risk_gated',
			label: 'Risk-Gated',
			blurb: 'Ordinary writes land; risky ones open a PR.'
		},
		{ id: 'paranoid', label: 'Paranoid', blurb: 'Every write waits for approval.' }
	];

	const statuses = ['pending', 'promoted', 'merged', 'superseded', 'quarantined', 'forgotten'];

	const kindLabel: Record<string, string> = {
		new_fact: 'New fact',
		strengthened_fact: 'Strengthened',
		contradiction_detected: 'Contradiction',
		memory_superseded: 'Supersede',
		edge_added: 'New edge',
		node_decayed: 'Decayed',
		dream_consolidation: 'Dream merge'
	};

	const actions: { id: MemoryPrAction; label: string; cls: string }[] = [
		{ id: 'promote', label: 'Promote', cls: 'promote' },
		{ id: 'merge', label: 'Merge', cls: 'merge' },
		{ id: 'supersede', label: 'Supersede', cls: 'supersede' },
		{ id: 'quarantine', label: 'Quarantine', cls: 'quarantine' },
		{ id: 'forget', label: 'Forget', cls: 'forget' },
		{ id: 'ask_agent_why', label: 'Ask Agent Why', cls: 'why' }
	];

	async function load() {
		loading = true;
		try {
			const res = await api.memoryPrs.list(statusFilter || undefined, 100);
			prs = res.prs;
			pendingCount = res.pendingCount;
			mode = res.mode;
			if (selected) selected = prs.find((p) => p.id === selected!.id) ?? null;
		} finally {
			loading = false;
		}
	}

	async function setMode(m: ReviewMode) {
		const res = await api.memoryPrs.setMode(m);
		mode = res.mode;
		toasts.push({
			type: 'MemoryPromoted',
			title: 'Review mode updated',
			body: `Memory PR gating is now ${modes.find((x) => x.id === m)?.label}.`,
			color: '#6366f1',
			dwellMs: 4000
		});
	}

	async function act(pr: MemoryPr, action: MemoryPrAction) {
		busy = `${pr.id}:${action}`;
		try {
			if (action === 'ask_agent_why') {
				const res = (await api.memoryPrs.act(pr.id, action)) as {
					why: { code: string; detail: string }[];
				};
				why = res.why;
				selected = pr;
				return;
			}
			await api.memoryPrs.act(pr.id, action);
			toasts.push({
				type: 'MemoryPromoted',
				title: `PR ${action}d`,
				body: pr.title,
				color: actionColor(action),
				dwellMs: 4500
			});
			why = null;
			await load();
		} catch (e) {
			toasts.push({
				type: 'MemoryDemoted',
				title: 'Action failed',
				body: String(e),
				color: '#f43f5e',
				dwellMs: 5000
			});
		} finally {
			busy = null;
		}
	}

	function actionColor(action: MemoryPrAction): string {
		switch (action) {
			case 'promote':
				return '#10b981';
			case 'merge':
				return '#6366f1';
			case 'supersede':
				return '#38bdf8';
			case 'quarantine':
				return '#f59e0b';
			case 'forget':
				return '#f43f5e';
			default:
				return '#818cf8';
		}
	}

	function select(pr: MemoryPr) {
		selected = pr;
		why = null;
	}

	// Live: a Memory PR opened or was decided elsewhere — refresh the queue.
	$effect(() => {
		if ($memoryPrEvents.length) void load();
	});

	onMount(load);

	// Diff rendering helpers
	function diffContent(pr: MemoryPr): string {
		const node = pr.diff?.node as { content?: string } | undefined;
		return node?.content ?? '';
	}
	function diffNodeType(pr: MemoryPr): string {
		const node = pr.diff?.node as { nodeType?: string } | undefined;
		return node?.nodeType ?? '';
	}
	function diffTags(pr: MemoryPr): string[] {
		const node = pr.diff?.node as { tags?: string[] } | undefined;
		return node?.tags ?? [];
	}
</script>

<div class="mx-auto max-w-6xl px-5 py-6">
	<PageHeader
		icon="memorypr"
		title="Memory PRs"
		subtitle="Approve changes to the agent's brain like code."
		accent="synapse"
	>
		<span class="pending-badge" class:has={pendingCount > 0}>
			<AnimatedNumber value={pendingCount} /> pending
		</span>
	</PageHeader>

	<!-- ░░ THE KILLER LINE ░░ -->
	<div class="manifesto" use:reveal>
		Vestige <strong>auto-remembers ordinary context</strong>, but opens a
		<strong>Memory PR</strong> when the agent tries to <strong>rewrite its own brain</strong>.
		<span class="manifesto-note">
			Risky writes are <strong>quarantine-reviewed</strong>: recorded for audit, but held
			out of retrieval until you decide — influence suspended, history preserved.
		</span>
	</div>

	<!-- ░░ MODE TOGGLE ░░ -->
	<div class="modes glass" use:reveal>
		{#each modes as m (m.id)}
			<button class="mode" class:on={mode === m.id} onclick={() => setMode(m.id)}>
				<span class="mode-label">{m.label}</span>
				<span class="mode-blurb">{m.blurb}</span>
			</button>
		{/each}
	</div>

	<!-- ░░ STATUS FILTER ░░ -->
	<div class="filters" use:reveal>
		{#each statuses as s (s)}
			<button
				class="filter"
				class:on={statusFilter === s}
				onclick={() => {
					statusFilter = s;
					load();
				}}
			>
				{s}
			</button>
		{/each}
		<button
			class="filter"
			class:on={statusFilter === ''}
			onclick={() => {
				statusFilter = '';
				load();
			}}
		>
			all
		</button>
	</div>

	<div class="layout">
		<!-- ░░ PR LIST ░░ -->
		<aside class="pr-list glass" use:reveal>
			{#if loading}
				<p class="empty">Loading…</p>
			{:else if prs.length === 0}
				<p class="empty">
					{#if statusFilter === 'pending'}
						No pending Memory PRs. The brain is up to date — ordinary writes
						are landing automatically.
					{:else}
						No {statusFilter || ''} PRs.
					{/if}
				</p>
			{:else}
				<ul>
					{#each prs as pr (pr.id)}
						<li>
							<button
								class="pr-row"
								class:active={selected?.id === pr.id}
								onclick={() => select(pr)}
							>
								<div class="pr-row-top">
									<span class="pr-kind kind-{pr.kind}">{kindLabel[pr.kind] ?? pr.kind}</span>
									<span class="pr-status st-{pr.status}">{pr.status}</span>
								</div>
								<div class="pr-title">{pr.title}</div>
								{#if pr.signals.length}
									<div class="pr-sig-count">⚠ {pr.signals.length} risk signal{pr.signals.length > 1 ? 's' : ''}</div>
								{/if}
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</aside>

		<!-- ░░ PR DIFF DETAIL ░░ -->
		<section class="pr-detail">
			{#if !selected}
				<div class="glass center-msg">Select a Memory PR to review the diff.</div>
			{:else}
				<div class="glass diff-card" use:reveal>
					<div class="diff-head">
						<span class="pr-kind kind-{selected.kind}">{kindLabel[selected.kind] ?? selected.kind}</span>
						<span class="pr-status st-{selected.status}">{selected.status}</span>
						{#if selected.run_id}
							<a class="run-link" href={`/blackbox`} title="View the run that produced this">
								<Icon name="blackbox" size={13} /> {selected.run_id.replace('run_', '').slice(0, 8)}
							</a>
						{/if}
					</div>
					<h2 class="diff-title">{selected.title}</h2>

					<!-- The cognition diff -->
					<div class="diff-body">
						<div class="diff-meta">
							{#if diffNodeType(selected)}
								<span class="meta-pill">type: {diffNodeType(selected)}</span>
							{/if}
							{#each diffTags(selected) as t (t)}
								<span class="meta-pill tag">#{t}</span>
							{/each}
						</div>
						{#if diffContent(selected)}
							<pre class="diff-add"><span class="gutter">+</span>{diffContent(selected)}</pre>
						{/if}
					</div>

					<!-- Self-explaining risk signals -->
					{#if selected.signals.length}
						<div class="signals">
							<span class="signals-title">Why this opened</span>
							{#each selected.signals as sig (sig.code)}
								<div class="signal">
									<code class="sig-code">{sig.code}</code>
									<span class="sig-detail">{sig.detail}</span>
								</div>
							{/each}
						</div>
					{/if}

					<!-- Ask Agent Why response -->
					{#if why}
						<div class="why-box">
							<span class="why-title">Agent's reasoning</span>
							{#each why as w (w.code)}
								<div class="signal">
									<code class="sig-code">{w.code}</code>
									<span class="sig-detail">{w.detail}</span>
								</div>
							{/each}
						</div>
					{/if}

					<!-- Action buttons -->
					{#if selected.status === 'pending'}
						<div class="actions">
							{#each actions as a (a.id)}
								<button
									class="action {a.cls}"
									disabled={busy === `${selected.id}:${a.id}`}
									onclick={() => act(selected!, a.id)}
								>
									{a.label}
								</button>
							{/each}
						</div>
					{:else}
						<div class="decided">
							Decided: <strong>{selected.decision ?? selected.status}</strong>
							{#if selected.decided_at}
								<span class="text-dim">· {new Date(selected.decided_at).toLocaleString()}</span>
							{/if}
						</div>
					{/if}
				</div>
			{/if}
		</section>
	</div>
</div>

<style>
	.pending-badge {
		font-size: 0.8rem;
		font-weight: 700;
		padding: 5px 11px;
		border-radius: 8px;
		color: var(--color-text-dim, #8b8ba7);
		border: 1px solid color-mix(in oklab, white 10%, transparent);
	}
	.pending-badge.has {
		color: #f59e0b;
		border-color: color-mix(in oklab, #f59e0b 40%, transparent);
		background: color-mix(in oklab, #f59e0b 10%, transparent);
	}

	.manifesto {
		font-size: 1.05rem;
		line-height: 1.55;
		color: var(--color-text, #e2e2f0);
		padding: 16px 20px;
		border-radius: 12px;
		margin-bottom: 16px;
		background: linear-gradient(
			100deg,
			color-mix(in oklab, var(--color-synapse) 14%, transparent),
			transparent 70%
		);
		border: 1px solid color-mix(in oklab, var(--color-synapse) 20%, transparent);
	}
	.manifesto strong {
		color: var(--color-synapse-glow, #818cf8);
	}
	.manifesto-note {
		display: block;
		margin-top: 8px;
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--color-text-dim, #c0c0d8);
	}
	.manifesto-note strong {
		color: #f59e0b;
	}

	.glass {
		background: color-mix(in oklab, var(--color-void, #050510) 55%, transparent);
		border: 1px solid color-mix(in oklab, white 8%, transparent);
		backdrop-filter: blur(12px);
		border-radius: 14px;
	}

	.modes {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 8px;
		padding: 8px;
		margin-bottom: 14px;
	}
	@media (max-width: 640px) {
		.modes {
			grid-template-columns: 1fr;
		}
	}
	.mode {
		display: flex;
		flex-direction: column;
		gap: 3px;
		padding: 11px 14px;
		border-radius: 10px;
		border: 1px solid transparent;
		background: color-mix(in oklab, white 3%, transparent);
		cursor: pointer;
		text-align: left;
		transition: all 0.16s ease;
	}
	.mode:hover {
		background: color-mix(in oklab, var(--color-synapse) 10%, transparent);
	}
	.mode.on {
		border-color: color-mix(in oklab, var(--color-synapse) 50%, transparent);
		background: color-mix(in oklab, var(--color-synapse) 18%, transparent);
	}
	.mode-label {
		font-weight: 700;
		font-size: 0.9rem;
	}
	.mode-blurb {
		font-size: 0.72rem;
		color: var(--color-text-dim, #8b8ba7);
		line-height: 1.35;
	}

	.filters {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-bottom: 14px;
	}
	.filter {
		font-size: 0.74rem;
		padding: 5px 11px;
		border-radius: 7px;
		border: 1px solid color-mix(in oklab, white 8%, transparent);
		background: transparent;
		color: var(--color-text-dim, #8b8ba7);
		cursor: pointer;
		text-transform: capitalize;
		transition: all 0.16s ease;
	}
	.filter:hover {
		color: var(--color-text, #e2e2f0);
	}
	.filter.on {
		color: var(--color-synapse-glow, #818cf8);
		border-color: color-mix(in oklab, var(--color-synapse) 45%, transparent);
		background: color-mix(in oklab, var(--color-synapse) 12%, transparent);
	}

	.layout {
		display: grid;
		grid-template-columns: 300px 1fr;
		gap: 16px;
		align-items: start;
	}
	@media (max-width: 860px) {
		.layout {
			grid-template-columns: 1fr;
		}
	}

	.pr-list {
		padding: 12px;
		position: sticky;
		top: 16px;
		max-height: calc(100vh - 40px);
		overflow-y: auto;
	}
	.pr-list ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.empty {
		font-size: 0.82rem;
		color: var(--color-text-dim, #8b8ba7);
		line-height: 1.5;
		padding: 8px;
	}
	.pr-row {
		width: 100%;
		text-align: left;
		padding: 10px 12px;
		border-radius: 10px;
		border: 1px solid transparent;
		background: color-mix(in oklab, white 3%, transparent);
		cursor: pointer;
		transition: all 0.16s ease;
	}
	.pr-row:hover {
		background: color-mix(in oklab, var(--color-synapse) 10%, transparent);
	}
	.pr-row.active {
		border-color: color-mix(in oklab, var(--color-synapse) 45%, transparent);
		background: color-mix(in oklab, var(--color-synapse) 15%, transparent);
	}
	.pr-row-top {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		margin-bottom: 5px;
	}
	.pr-kind {
		font-size: 0.66rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 2px 7px;
		border-radius: 5px;
		background: color-mix(in oklab, var(--color-synapse) 16%, transparent);
		color: var(--color-synapse-glow, #818cf8);
	}
	.kind-contradiction_detected {
		background: color-mix(in oklab, #fb7185 16%, transparent);
		color: #fb7185;
	}
	.kind-memory_superseded {
		background: color-mix(in oklab, #38bdf8 16%, transparent);
		color: #38bdf8;
	}
	.kind-dream_consolidation {
		background: color-mix(in oklab, #c084fc 16%, transparent);
		color: #c084fc;
	}
	.pr-status {
		font-size: 0.64rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--color-text-dim, #8b8ba7);
		align-self: center;
	}
	.st-pending {
		color: #f59e0b;
	}
	.st-promoted {
		color: #10b981;
	}
	.st-forgotten,
	.st-quarantined {
		color: #f43f5e;
	}
	.pr-title {
		font-size: 0.84rem;
		line-height: 1.35;
		color: var(--color-text, #e2e2f0);
	}
	.pr-sig-count {
		font-size: 0.7rem;
		color: #f59e0b;
		margin-top: 5px;
	}

	.pr-detail {
		min-width: 0;
	}
	.center-msg {
		padding: 50px;
		text-align: center;
		color: var(--color-text-dim, #8b8ba7);
	}
	.diff-card {
		padding: 20px 22px;
	}
	.diff-head {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
	}
	.run-link {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		margin-left: auto;
		font-size: 0.74rem;
		color: var(--color-synapse-glow, #818cf8);
		text-decoration: none;
		padding: 3px 8px;
		border-radius: 6px;
		background: color-mix(in oklab, var(--color-synapse) 10%, transparent);
	}
	.diff-title {
		font-size: 1.15rem;
		font-weight: 700;
		margin: 12px 0 16px;
		line-height: 1.4;
	}
	.diff-body {
		margin-bottom: 16px;
	}
	.diff-meta {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-bottom: 10px;
	}
	.meta-pill {
		font-size: 0.72rem;
		padding: 2px 8px;
		border-radius: 6px;
		background: color-mix(in oklab, white 6%, transparent);
		color: var(--color-text-dim, #c0c0d8);
	}
	.meta-pill.tag {
		color: var(--color-synapse-glow, #818cf8);
	}
	.diff-add {
		margin: 0;
		padding: 12px 14px 12px 36px;
		position: relative;
		border-radius: 9px;
		background: color-mix(in oklab, #10b981 9%, transparent);
		border-left: 3px solid #10b981;
		font-size: 0.85rem;
		line-height: 1.55;
		white-space: pre-wrap;
		word-break: break-word;
		font-family: var(--font-mono, monospace);
		color: var(--color-text, #e2e2f0);
	}
	.diff-add .gutter {
		position: absolute;
		left: 12px;
		color: #10b981;
		font-weight: 700;
	}

	.signals,
	.why-box {
		margin-bottom: 16px;
		padding: 12px 14px;
		border-radius: 10px;
		background: color-mix(in oklab, #f59e0b 8%, transparent);
		border: 1px solid color-mix(in oklab, #f59e0b 22%, transparent);
	}
	.why-box {
		background: color-mix(in oklab, var(--color-synapse) 8%, transparent);
		border-color: color-mix(in oklab, var(--color-synapse) 22%, transparent);
	}
	.signals-title,
	.why-title {
		display: block;
		font-size: 0.68rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: #f59e0b;
		margin-bottom: 8px;
	}
	.why-title {
		color: var(--color-synapse-glow, #818cf8);
	}
	.signal {
		display: flex;
		gap: 8px;
		align-items: baseline;
		padding: 3px 0;
	}
	.sig-code {
		font-size: 0.7rem;
		color: #f59e0b;
		white-space: nowrap;
	}
	.sig-detail {
		font-size: 0.82rem;
		line-height: 1.4;
		color: var(--color-text, #e2e2f0);
	}

	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.action {
		font-size: 0.8rem;
		font-weight: 600;
		padding: 8px 14px;
		border-radius: 9px;
		border: 1px solid color-mix(in oklab, var(--ac, #6366f1) 40%, transparent);
		background: color-mix(in oklab, var(--ac, #6366f1) 10%, transparent);
		color: var(--ac, #818cf8);
		cursor: pointer;
		transition: all 0.16s ease;
	}
	.action:hover:not(:disabled) {
		background: color-mix(in oklab, var(--ac, #6366f1) 22%, transparent);
		transform: translateY(-1px);
	}
	.action:disabled {
		opacity: 0.5;
		cursor: wait;
	}
	.action.promote {
		--ac: #10b981;
	}
	.action.merge {
		--ac: #6366f1;
	}
	.action.supersede {
		--ac: #38bdf8;
	}
	.action.quarantine {
		--ac: #f59e0b;
	}
	.action.forget {
		--ac: #f43f5e;
	}
	.action.why {
		--ac: #818cf8;
	}
	.decided {
		font-size: 0.88rem;
		padding: 10px 14px;
		border-radius: 9px;
		background: color-mix(in oklab, white 4%, transparent);
	}
	.text-dim {
		color: var(--color-text-dim, #8b8ba7);
	}
</style>
