<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import { eventFeed } from '$stores/websocket';
	import { shouldRefreshVerdicts } from '$lib/observatory/route-event-pulse';
	import type {
		SanhedrinAppealReason,
		SanhedrinClaim,
		SanhedrinReceipt,
		VerdictLevel,
	} from '$types';

	const LEVELS: VerdictLevel[] = ['PASS', 'NOTE', 'CAUTION', 'VETO', 'APPEALED'];

	let receipt = $state<SanhedrinReceipt | null>(null);
	let error = $state('');
	let expanded = $state(false);
	let appealing = $state<SanhedrinAppealReason | null>(null);
	let autoExpandedReceiptId = $state<string | null>(null);

	let verdict = $derived<VerdictLevel>(receipt?.verdictBar ?? (error ? 'CAUTION' : 'NOTE'));
	let appealClaim = $derived<SanhedrinClaim | null>(
		receipt?.claims.find((claim) => claim.decision === 'veto') ??
			receipt?.claims.find((claim) => claim.decision === 'appealed') ??
			null
	);
	let displayClaim = $derived<SanhedrinClaim | null>(
		appealClaim ??
			receipt?.claims[0] ??
			null
	);
	let visible = $derived(Boolean(receipt) || Boolean(error));

	onMount(() => {
		refresh();
		const timer = window.setInterval(refresh, 30000);
		return () => window.clearInterval(timer);
	});

	$effect(() => {
		const head = $eventFeed[0];
		if (head && shouldRefreshVerdicts(head)) void refresh();
	});

	async function refresh() {
		try {
			const next = await api.sanhedrin.latest();
			receipt = next.receipt;
			error = '';
			if (
				next.receipt?.verdictBar === 'VETO' &&
				next.receipt.id !== autoExpandedReceiptId
			) {
				expanded = true;
				autoExpandedReceiptId = next.receipt.id;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	async function appeal(reason: SanhedrinAppealReason) {
		if (!appealClaim || receipt?.verdictBar !== 'VETO') return;
		appealing = reason;
		try {
			const next = await api.sanhedrin.appeal(reason, undefined, appealClaim.id, receipt.id);
			receipt = next.receipt;
			expanded = true;
			error = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			appealing = null;
		}
	}

	function formatWhen(value?: string): string {
		if (!value) return '';
		const d = new Date(value);
		if (Number.isNaN(d.getTime())) return '';
		return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}

	function precedentText(claim: SanhedrinClaim | null | undefined): string[] {
		if (!claim?.precedent?.length) return ['No precedent attached.'];
		return claim.precedent.map((p) => p.summary ?? p.command ?? 'Precedent recorded.').slice(0, 3);
	}
</script>

{#if visible}
	<section class={`verdict-bar tone-${verdict.toLowerCase()}`} aria-live="polite">
		<button
			type="button"
			class="verdict-summary"
			aria-expanded={expanded}
			onclick={() => (expanded = !expanded)}
		>
			<span class="label">Sanhedrin</span>
			<span class="levels" aria-label="Verdict levels">
				{#each LEVELS as level}
					<span class:active={level === verdict} aria-current={level === verdict ? 'true' : undefined}>
						{level}
					</span>
				{/each}
			</span>
			<span class="sr-only">Current verdict: {verdict}</span>
			<span class="summary-text">
				{#if error}
					{error}
				{:else if receipt}
					{receipt.summary}
				{/if}
			</span>
			<span class="when">{formatWhen(receipt?.createdAt)}</span>
		</button>

		{#if expanded && receipt}
			<div class="receipt" role="region" aria-label="Sanhedrin veto receipt">
				<div class="receipt-grid">
					<div>
						<div class="field-label">Claim</div>
						<p>{displayClaim?.text ?? receipt.draftPreview}</p>
					</div>
					<div>
						<div class="field-label">Verdict</div>
						<p>{displayClaim?.decision ?? receipt.overall} · {displayClaim?.evidence_state ?? verdict}</p>
					</div>
					<div>
						<div class="field-label">Precedent</div>
						<ul>
							{#each precedentText(displayClaim) as item}
								<li>{item}</li>
							{/each}
						</ul>
					</div>
					<div>
						<div class="field-label">Fix</div>
						<p>{displayClaim?.fix || 'No change required.'}</p>
					</div>
				</div>

				<div class="appeal-row">
					<span>Appeal</span>
					{#if appealClaim && receipt.verdictBar === 'VETO'}
						<button type="button" disabled={Boolean(appealing)} onclick={() => appeal('stale')}>
							{appealing === 'stale' ? 'Saving' : 'Stale'}
						</button>
						<button type="button" disabled={Boolean(appealing)} onclick={() => appeal('wrong')}>
							{appealing === 'wrong' ? 'Saving' : 'Wrong'}
						</button>
						<button type="button" disabled={Boolean(appealing)} onclick={() => appeal('too_strict')}>
							{appealing === 'too_strict' ? 'Saving' : 'Too strict'}
						</button>
					{:else if receipt.verdictBar === 'APPEALED'}
						<p>Appeal recorded.</p>
					{:else}
						<p>No appealable veto in this receipt.</p>
					{/if}
				</div>
			</div>
		{/if}
	</section>
{/if}

<style>
	.verdict-bar {
		border-bottom: 1px solid rgba(255, 255, 255, 0.06);
		background: rgba(8, 9, 18, 0.72);
		backdrop-filter: blur(18px) saturate(160%);
		-webkit-backdrop-filter: blur(18px) saturate(160%);
		box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
	}

	.verdict-summary {
		width: 100%;
		min-height: 2.75rem;
		display: grid;
		grid-template-columns: auto auto minmax(0, 1fr) auto;
		align-items: center;
		gap: 0.75rem;
		padding: 0.55rem 1rem;
		color: var(--color-text);
		text-align: left;
		font: inherit;
	}

	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}

	.label,
	.field-label,
	.appeal-row > span {
		color: var(--color-dim);
		font-size: 0.68rem;
		text-transform: uppercase;
		letter-spacing: 0;
	}

	.levels {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}

	.levels span {
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 0.35rem;
		padding: 0.18rem 0.38rem;
		color: var(--color-muted);
		font-size: 0.64rem;
		line-height: 1;
	}

	.levels span.active {
		color: var(--color-bright);
		border-color: var(--verdict-color);
		background: color-mix(in srgb, var(--verdict-color) 18%, transparent);
		box-shadow: 0 0 14px color-mix(in srgb, var(--verdict-color) 28%, transparent);
	}

	.summary-text {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 0.78rem;
		color: var(--color-dim);
	}

	.when {
		color: var(--color-muted);
		font-size: 0.68rem;
	}

	.receipt {
		margin: 0 1rem 0.75rem;
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 0.5rem;
		background: rgba(10, 10, 26, 0.78);
		padding: 0.85rem;
	}

	.receipt-grid {
		display: grid;
		grid-template-columns: minmax(0, 1.4fr) minmax(10rem, 0.8fr) minmax(0, 1.1fr) minmax(0, 1.1fr);
		gap: 0.85rem;
	}

	.receipt p,
	.receipt li {
		margin: 0.25rem 0 0;
		color: var(--color-text);
		font-size: 0.76rem;
		line-height: 1.45;
		overflow-wrap: anywhere;
	}

	.receipt ul {
		margin: 0.25rem 0 0;
		padding-left: 1rem;
	}

	.appeal-row {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		margin-top: 0.85rem;
		flex-wrap: wrap;
	}

	.appeal-row button,
	.appeal-row p {
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 0.4rem;
		padding: 0.35rem 0.6rem;
		color: var(--color-text);
		background: rgba(255, 255, 255, 0.04);
		font-size: 0.72rem;
		margin: 0;
	}

	.appeal-row button:hover:not(:disabled),
	.verdict-summary:hover {
		background: rgba(255, 255, 255, 0.05);
	}

	.appeal-row button:disabled {
		opacity: 0.55;
		cursor: wait;
	}

	.tone-pass,
	.tone-note {
		--verdict-color: #10b981;
	}

	.tone-caution {
		--verdict-color: #f59e0b;
	}

	.tone-veto {
		--verdict-color: #ef4444;
	}

	.tone-appealed {
		--verdict-color: #22c7de;
	}

	@media (max-width: 900px) {
		.verdict-summary {
			grid-template-columns: auto minmax(0, 1fr) auto;
		}

		.levels {
			grid-column: 1 / -1;
			order: 4;
			overflow-x: auto;
			padding-bottom: 0.1rem;
		}

		.receipt-grid {
			grid-template-columns: 1fr;
		}
	}
</style>
