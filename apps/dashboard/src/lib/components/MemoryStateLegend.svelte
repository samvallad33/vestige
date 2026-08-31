<!--
	MemoryStateLegend — v2.0.8 UI gap closure

	Explains the 4 FSRS accessibility buckets users see when they toggle the
	graph's colour mode to "state". The thresholds match
	`execute_system_status` server-side so the colour bands line up exactly
	with what the Stats page reports in its stateDistribution block.

	Active (≥ 70%)    — easily retrievable, surfaces in every search
	Dormant (40–70%)  — retrievable with effort, lower rank
	Silent (10–40%)   — difficult, needs cues
	Unavailable (<10%)— needs reinforcement before it will surface again

	Ship as a small floating overlay in the top-right of the Graph page when
	colour mode is "state". Hidden when mode is "type" so the legend doesn't
	compete with the node-type palette.
-->
<script lang="ts">
	import {
		MEMORY_STATE_COLORS,
		MEMORY_STATE_DESCRIPTIONS,
		type MemoryState,
	} from '$lib/memory-state';

	// Ordered highest-accessibility first so the legend reads top-to-bottom
	// as "healthy → needs-work", matching the intuition of a status panel.
	const STATES: MemoryState[] = ['active', 'dormant', 'silent', 'unavailable'];
</script>

<div
	class="pointer-events-auto glass-subtle rounded-xl px-3 py-2.5 text-xs space-y-1.5 backdrop-blur-md border border-synapse/10"
	role="group"
	aria-label="Memory state colour legend"
>
	<div class="text-[10px] uppercase tracking-wider text-muted font-semibold mb-1.5">
		FSRS accessibility
	</div>
	{#each STATES as state (state)}
		<div class="flex items-center gap-2">
			<span
				class="w-2.5 h-2.5 rounded-full flex-shrink-0"
				style="background: {MEMORY_STATE_COLORS[state]}; box-shadow: 0 0 6px {MEMORY_STATE_COLORS[
					state
				]}55;"
			></span>
			<span class="text-text capitalize">{state}</span>
			<span class="text-muted text-[10px] ml-auto">
				{MEMORY_STATE_DESCRIPTIONS[state].match(/\(([^)]+)\)/)?.[1] ?? ''}
			</span>
		</div>
	{/each}
</div>
