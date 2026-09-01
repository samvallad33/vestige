<script lang="ts">
	/**
	 * Brain print panel — structure-only signature of the live store.
	 * Print id, trait chips, permalink. Zero memory text.
	 */
	import type { BrainPrint, BrainTrait } from '$lib/observatory/brain-print';

	interface Props {
		print: BrainPrint | null;
		activePrintId: string | null;
		printing: boolean;
		error: string | null;
		copied: boolean;
		disabled: boolean;
		onprint: () => void;
		oncopy: () => void;
	}

	let {
		print,
		activePrintId,
		printing,
		error,
		copied,
		disabled,
		onprint,
		oncopy
	}: Props = $props();

	const traits: BrainTrait[] = $derived(print?.traits ?? []);
</script>

<div class="obs-print">
	<button
		class="obs-print-btn"
		type="button"
		onclick={onprint}
		disabled={disabled || printing}
		aria-busy={printing}
	>
		{printing ? 'Reading shape…' : print ? 'Recompute print' : 'Brain print'}
	</button>
	{#if error}
		<span class="obs-print-error" role="status">{error}</span>
	{/if}
	{#if activePrintId}
		<div class="obs-print-card" aria-live="polite">
			<span class="obs-print-kicker">structure only · zero memory text</span>
			<code class="obs-print-id">{activePrintId}</code>
			{#if traits.length}
				<ul class="obs-traits">
					{#each traits as trait (trait.id)}
						<li class="obs-chip">{trait.label}</li>
					{/each}
				</ul>
			{/if}
			<button class="obs-print-copy" type="button" onclick={oncopy}>
				{copied ? 'Permalink copied' : 'Copy permalink'}
			</button>
		</div>
	{/if}
</div>

<style>
	.obs-print {
		display: flex;
		flex-direction: column;
		gap: 0.55rem;
		margin-top: 0.15rem;
	}
	.obs-print-btn {
		align-self: flex-start;
		padding: 0.55rem 0.9rem;
		border-radius: 0.65rem;
		border: 1px solid rgba(34, 199, 222, 0.45);
		background: rgba(8, 28, 32, 0.72);
		color: #9ff8ec;
		font-size: 0.82rem;
		font-weight: 700;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		cursor: pointer;
		box-shadow: 0 0 18px -8px rgba(34, 199, 222, 0.7);
		transition:
			border-color 0.16s ease,
			background 0.16s ease,
			box-shadow 0.16s ease;
	}
	.obs-print-btn:hover:not(:disabled) {
		border-color: rgba(92, 240, 166, 0.7);
		background: rgba(10, 36, 40, 0.88);
		box-shadow: 0 0 22px -6px rgba(92, 240, 166, 0.55);
	}
	.obs-print-btn:focus-visible {
		outline: 2px solid rgba(34, 199, 222, 0.7);
		outline-offset: 2px;
	}
	.obs-print-btn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.obs-print-card {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		padding: 0.7rem 0.85rem;
		border-radius: 0.7rem;
		border: 1px solid rgba(140, 199, 219, 0.28);
		background: rgba(6, 16, 20, 0.62);
		-webkit-backdrop-filter: blur(8px);
		backdrop-filter: blur(8px);
	}
	.obs-print-kicker {
		font-size: 0.62rem;
		letter-spacing: 0.16em;
		text-transform: uppercase;
		color: rgba(140, 199, 180, 0.62);
	}
	.obs-print-id {
		font-size: 0.98rem;
		font-variant-numeric: tabular-nums;
		letter-spacing: 0.08em;
		color: #7ff3e6;
		text-shadow: 0 0 16px rgba(34, 199, 222, 0.45);
	}
	.obs-traits {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.obs-chip {
		padding: 0.18rem 0.5rem;
		border-radius: 999px;
		border: 1px solid rgba(92, 240, 166, 0.35);
		background: rgba(20, 48, 40, 0.45);
		color: #c8ffe8;
		font-size: 0.7rem;
		letter-spacing: 0.02em;
	}
	.obs-print-copy {
		align-self: flex-start;
		margin-top: 0.1rem;
		padding: 0;
		border: 0;
		background: transparent;
		color: #8cc7db;
		font-size: 0.76rem;
		letter-spacing: 0.02em;
		cursor: pointer;
	}
	.obs-print-copy:hover {
		text-decoration: underline;
		color: #eafffb;
	}
	.obs-print-error {
		color: #ff6a5e;
		font-size: 0.72rem;
		max-width: 22rem;
	}
</style>
