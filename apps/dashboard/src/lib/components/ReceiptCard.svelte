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

	// NeuroRuntime v0 — the Microglial Firewall. `influence_allowed` is the
	// headline boolean: did anything the firewall caught reach the answer? It
	// is `false` exactly when this retrieval had a memory quarantined. Old
	// receipts omit the field, so `undefined`/`true` both read as "clean".
	const quarantined = $derived(receipt.quarantined ?? []);
	const firewallBlocked = $derived(receipt.influence_allowed === false || quarantined.length > 0);

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

	<!-- Firewall badge — the proof that nothing quarantined reached the answer. -->
	{#if firewallBlocked}
		<div class="firewall blocked" title="The Microglial Firewall quarantined a poisoned memory">
			<span class="fw-glyph">🛡</span>
			<span class="fw-text">
				FIREWALL BLOCKED {quarantined.length}
				{quarantined.length === 1 ? 'memory' : 'memories'}
				<span class="fw-flag">· influenceAllowed: false</span>
			</span>
		</div>
	{:else}
		<div class="firewall clean" title="No quarantined memory influenced this answer">
			<span class="fw-glyph">✓</span>
			<span class="fw-text">clean<span class="fw-flag"> · no quarantine</span></span>
		</div>
	{/if}

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

		{#if quarantined.length}
			<div class="r-section">
				<span class="r-section-title">Quarantined by firewall</span>
				<div class="q-list">
					{#each quarantined as q (q.id)}
						<div class="q-row" title={q.threat}>
							<code class="chip quarantine">{q.id.slice(0, 8)} · {q.reason.replace(/_/g, ' ')}</code>
							<span class="q-threat">{q.threat}</span>
						</div>
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

	/* Firewall badge — danger when blocked, a subtle green tick when clean. */
	.firewall {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 7px 11px;
		border-radius: 9px;
		font-size: 0.78rem;
		font-weight: 700;
		letter-spacing: 0.01em;
	}
	.firewall.blocked {
		color: #ef4444;
		background: color-mix(in oklab, #ef4444 14%, transparent);
		border: 1px solid color-mix(in oklab, #ef4444 40%, transparent);
		text-transform: uppercase;
	}
	.firewall.clean {
		color: var(--color-recall, #10b981);
		background: color-mix(in oklab, var(--color-recall, #10b981) 8%, transparent);
		border: 1px solid color-mix(in oklab, var(--color-recall, #10b981) 22%, transparent);
		font-weight: 600;
	}
	.fw-glyph {
		font-size: 0.95rem;
		line-height: 1;
	}
	.fw-flag {
		font-weight: 600;
		opacity: 0.75;
		font-variant-numeric: tabular-nums;
	}
	.firewall.clean .fw-flag {
		text-transform: uppercase;
		letter-spacing: 0.05em;
		font-size: 0.66rem;
	}
	.q-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.q-row {
		display: flex;
		flex-direction: column;
		gap: 3px;
	}
	.chip.quarantine {
		align-self: flex-start;
		color: #ef4444;
		background: color-mix(in oklab, #ef4444 12%, transparent);
		border: 1px solid color-mix(in oklab, #ef4444 32%, transparent);
	}
	.q-threat {
		font-size: 0.74rem;
		line-height: 1.45;
		color: var(--color-text-dim, #8b8ba7);
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
