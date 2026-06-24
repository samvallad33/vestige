<script lang="ts">
	// ═══════════════════════════════════════════════════════════════════════
	//  MEMORY RECEIPT CARD — the nutrition label for a retrieval.
	// ───────────────────────────────────────────────────────────────────────
	//  Shows what was retrieved, what was suppressed and why, the activation
	//  path, the trust floor (the weakest link the answer rests on), and the
	//  decay risk. "Open receipt in Cinema" deep-links to the graph centered on
	//  the receipt's primary memory, starting the (protected) Cinema flythrough
	//  over the exact memory set the receipt names.
	// ═══════════════════════════════════════════════════════════════════════
	import { goto } from '$app/navigation';
	import Icon from './Icon.svelte';
	import type { Receipt } from '$lib/stores/api';

	interface Props {
		receipt: Receipt;
		compact?: boolean;
	}
	let { receipt, compact = false }: Props = $props();

	const riskColor: Record<Receipt['decay_risk'], string> = {
		low: 'var(--color-recall, #10b981)',
		medium: '#f59e0b',
		high: '#f43f5e'
	};

	function openInCinema() {
		const primary = receipt.retrieved[0];
		if (!primary) return;
		const focus = receipt.retrieved.join(',');
		goto(`/graph?center=${encodeURIComponent(primary)}&focus=${encodeURIComponent(focus)}`);
	}
</script>

<div class="receipt" class:compact style:--risk={riskColor[receipt.decay_risk]}>
	<div class="r-head">
		<code class="r-id">{receipt.receipt_id}</code>
		<span class="r-risk" style:color={riskColor[receipt.decay_risk]}>
			decay: {receipt.decay_risk}
		</span>
	</div>

	<div class="r-metrics">
		<div class="metric">
			<span class="m-val">{receipt.retrieved.length}</span>
			<span class="m-label">retrieved</span>
		</div>
		<div class="metric">
			<span class="m-val">{receipt.suppressed.length}</span>
			<span class="m-label">suppressed</span>
		</div>
		<div class="metric">
			<span class="m-val">{(receipt.trust_floor * 100).toFixed(0)}%</span>
			<span class="m-label">trust floor</span>
		</div>
	</div>

	{#if !compact}
		{#if receipt.activation_path.length}
			<div class="r-section">
				<span class="r-section-title">Activation path</span>
				{#each receipt.activation_path as path (path)}
					<div class="path">{path}</div>
				{/each}
			</div>
		{/if}

		{#if receipt.retrieved.length}
			<div class="r-section">
				<span class="r-section-title">Retrieved</span>
				<div class="chips">
					{#each receipt.retrieved as id (id)}
						<code class="chip recall">{id.slice(0, 8)}</code>
					{/each}
				</div>
			</div>
		{/if}

		{#if receipt.suppressed.length}
			<div class="r-section">
				<span class="r-section-title">Suppressed</span>
				<div class="chips">
					{#each receipt.suppressed as s (s.id)}
						<code class="chip suppress" title={s.reason}>
							{s.id.slice(0, 8)} · {s.reason.replace('_', ' ')}
						</code>
					{/each}
				</div>
			</div>
		{/if}
	{/if}

	<button class="cinema-btn" onclick={openInCinema} disabled={!receipt.retrieved.length}>
		<Icon name="sparkle" size={14} />
		Open receipt in Cinema
	</button>
</div>

<style>
	.receipt {
		border: 1px solid color-mix(in oklab, var(--risk) 30%, transparent);
		border-left: 3px solid var(--risk);
		border-radius: 12px;
		padding: 14px 16px;
		background: color-mix(in oklab, var(--color-void, #050510) 50%, transparent);
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.receipt.compact {
		gap: 10px;
		padding: 12px 14px;
	}
	.r-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 8px;
	}
	.r-id {
		font-size: 0.78rem;
		color: var(--color-synapse-glow, #818cf8);
		word-break: break-all;
	}
	.r-risk {
		font-size: 0.7rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		white-space: nowrap;
	}
	.r-metrics {
		display: flex;
		gap: 20px;
	}
	.metric {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	.m-val {
		font-size: 1.25rem;
		font-weight: 800;
		line-height: 1;
		font-variant-numeric: tabular-nums;
	}
	.m-label {
		font-size: 0.64rem;
		text-transform: uppercase;
		letter-spacing: 0.07em;
		color: var(--color-text-dim, #8b8ba7);
	}
	.r-section {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.r-section-title {
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.07em;
		color: var(--color-text-dim, #8b8ba7);
	}
	.path {
		font-size: 0.8rem;
		font-family: var(--font-mono, monospace);
		color: var(--color-text, #e2e2f0);
		padding: 4px 8px;
		border-radius: 6px;
		background: color-mix(in oklab, var(--color-synapse) 8%, transparent);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 5px;
	}
	.chip {
		font-size: 0.72rem;
		padding: 2px 8px;
		border-radius: 6px;
	}
	.chip.recall {
		color: var(--color-recall, #10b981);
		background: color-mix(in oklab, var(--color-recall) 12%, transparent);
		border: 1px solid color-mix(in oklab, var(--color-recall) 28%, transparent);
	}
	.chip.suppress {
		color: #a78bfa;
		background: color-mix(in oklab, #a78bfa 12%, transparent);
		border: 1px solid color-mix(in oklab, #a78bfa 28%, transparent);
		text-decoration: line-through;
		text-decoration-color: color-mix(in oklab, #a78bfa 50%, transparent);
	}
	.cinema-btn {
		margin-top: 2px;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 7px;
		padding: 8px 14px;
		font-size: 0.8rem;
		font-weight: 600;
		border-radius: 9px;
		border: 1px solid color-mix(in oklab, var(--color-synapse) 40%, transparent);
		background: color-mix(in oklab, var(--color-synapse) 12%, transparent);
		color: var(--color-synapse-glow, #818cf8);
		cursor: pointer;
		transition: all 0.18s ease;
	}
	.cinema-btn:hover:not(:disabled) {
		background: color-mix(in oklab, var(--color-synapse) 24%, transparent);
		transform: translateY(-1px);
	}
	.cinema-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
</style>
