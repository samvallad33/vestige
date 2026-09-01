<script lang="ts">
	import { getMemoryState, MEMORY_STATE_COLORS, type MemoryState } from '$lib/memory-state';

	interface Props {
		retention: number;
		compact?: boolean;
	}

	let { retention, compact = false }: Props = $props();
	const state: MemoryState = $derived(getMemoryState(retention));
	const color = $derived(MEMORY_STATE_COLORS[state]);
</script>

<span
	class="state-chip"
	class:compact
	style="--chip:{color}"
	title="Retention-derived accessibility (until the backend exposes the true field)"
>
	<span class="dot" aria-hidden="true"></span>
	{state}
</span>

<style>
	.state-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		border: 1px solid color-mix(in oklab, var(--chip) 45%, transparent);
		background: color-mix(in oklab, var(--chip) 14%, transparent);
		color: var(--chip);
		border-radius: 999px;
		padding: 0.12rem 0.5rem;
		font-size: 0.62rem;
		font-weight: 700;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}
	.state-chip.compact {
		padding: 0.08rem 0.38rem;
		font-size: 0.56rem;
	}
	.dot {
		width: 0.42rem;
		height: 0.42rem;
		border-radius: 50%;
		background: var(--chip);
		box-shadow: 0 0 8px var(--chip);
	}
</style>
