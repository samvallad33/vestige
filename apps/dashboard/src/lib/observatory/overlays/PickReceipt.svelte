<script lang="ts">
	import { base } from '$app/paths';

	/**
	 * PickReceipt — click-as-incision. One shared provenance chip that opens
	 * the named memory / receipt / run / PR. Hosts mount this instead of
	 * inventing a third inspector.
	 */
	export interface PickProvenance {
		kind: 'memory' | 'receipt' | 'run' | 'pr' | 'trace' | string;
		id: string;
		label?: string;
		detail?: string;
	}

	interface Props {
		pick: PickProvenance | null;
		onclose?: () => void;
	}

	let { pick, onclose }: Props = $props();

	const href = $derived.by(() => {
		if (!pick?.id) return null;
		if (pick.kind === 'memory' || pick.kind === 'trace') {
			return `${base}/memories?memory=${encodeURIComponent(pick.id)}`;
		}
		if (pick.kind === 'receipt') return `${base}/observatory?receipt=${encodeURIComponent(pick.id)}`;
		if (pick.kind === 'run') return `${base}/blackbox?run=${encodeURIComponent(pick.id)}`;
		if (pick.kind === 'pr') return `${base}/memory-prs`;
		return null;
	});
</script>

{#if pick}
	<aside class="pick-receipt" aria-live="polite">
		<div class="kicker">PICK RECEIPT · {pick.kind}</div>
		<code>{pick.id}</code>
		{#if pick.label}<p>{pick.label}</p>{/if}
		{#if pick.detail}<p class="detail">{pick.detail}</p>{/if}
		<div class="row">
			{#if href}
				<a href={href}>Open in organ →</a>
			{/if}
			<button type="button" onclick={() => onclose?.()}>Dismiss</button>
		</div>
	</aside>
{/if}

<style>
	.pick-receipt {
		pointer-events: auto;
		position: fixed;
		right: 1.1rem;
		bottom: 5.5rem;
		z-index: 40;
		min-width: 16rem;
		max-width: 22rem;
		padding: 0.85rem 1rem;
		border-radius: 0.85rem;
		border: 1px solid rgba(34, 199, 222, 0.35);
		background: rgba(2, 3, 7, 0.88);
		color: #eafffb;
		box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
		backdrop-filter: blur(12px);
	}
	.kicker {
		font-size: 0.58rem;
		letter-spacing: 0.16em;
		text-transform: uppercase;
		color: rgba(127, 243, 230, 0.7);
	}
	code {
		display: block;
		margin: 0.35rem 0 0.45rem;
		font-size: 0.72rem;
		color: #7ff3e6;
		overflow-wrap: anywhere;
	}
	p {
		margin: 0;
		font-size: 0.78rem;
		line-height: 1.4;
		color: #c8e8e2;
	}
	.detail {
		color: #8aa9a5;
		font-size: 0.7rem;
	}
	.row {
		display: flex;
		gap: 0.7rem;
		margin-top: 0.7rem;
		align-items: center;
	}
	a,
	button {
		background: none;
		border: 0;
		padding: 0;
		color: #8cc7db;
		font-size: 0.72rem;
		cursor: pointer;
	}
	a:hover,
	button:hover {
		color: #eafffb;
		text-decoration: underline;
	}
</style>
