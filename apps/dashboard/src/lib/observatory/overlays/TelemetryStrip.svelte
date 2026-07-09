<script lang="ts">
	/**
	 * Telemetry strip — top instrument overlay (§7.3).
	 *
	 * A floating readout, not a chrome bar: the wrapper is pointer-events-none
	 * so the memory field stays interactive everywhere; only the [url] copy
	 * button opts back in. Copies the FULL shareable demo URL including the
	 * capture frame when one is pinned.
	 */
	import { base } from '$app/paths';

	interface Props {
		demoMode?: string;
		seed?: string;
		nodeCount?: number;
		edgeCount?: number;
		centerId?: string;
		frameCount?: number;
		fpsEstimate?: number;
		freezeFrame?: number | null;
		loading?: boolean;
		error?: string;
	}

	let {
		demoMode = 'recall-path',
		seed = 'vestige-observatory-v1',
		nodeCount = 0,
		edgeCount = 0,
		centerId = '',
		frameCount = 0,
		fpsEstimate = 0,
		freezeFrame = null,
		loading = false,
		error = ''
	}: Props = $props();

	// Full shareable URL — origin + base + canonical params (+ pinned frame).
	function copyDemoUrl() {
		const q = new URLSearchParams({ demo: demoMode, seed });
		if (freezeFrame !== null) q.set('frame', String(freezeFrame));
		const url = `${window.location.origin}${base}/observatory?${q.toString()}`;
		navigator.clipboard.writeText(url).catch(() => {});
	}
</script>

<!-- Floating instrument readout — never a solid chrome bar (§7.3).
     Every span is nowrap + tabular-nums so live counters never garble or
     reflow; lower-priority readouts shed below sm/md instead of wrapping. -->
<div
	class="absolute top-0 left-0 right-0 z-20 pointer-events-none"
	style="padding-top: env(safe-area-inset-top);"
>
	<div
		class="flex items-center justify-between gap-3 px-4 py-2 bg-gradient-to-b from-[#05060a]/85 to-transparent font-mono text-xs [font-variant-numeric:tabular-nums]"
	>
		<!-- Left: demo mode + seed (seed sheds below md; mode ellipsizes,
		     never overlaps the right cluster at extreme widths) -->
		<div class="flex items-center gap-3 min-w-0 flex-1 overflow-hidden">
			<span class="text-[#5dcaa5] tracking-widest uppercase truncate">
				{demoMode}
			</span>
			<span class="hidden md:inline text-[#ffffff]/[0.5] whitespace-nowrap">
				seed={seed.slice(0, 12)}{seed.length > 12 ? '…' : ''}
			</span>
		</div>

		<!-- Center: node/edge counts (center id sheds below lg) -->
		<div class="hidden sm:flex items-center gap-4">
			<span class="text-[#ffffff]/[0.55] whitespace-nowrap">
				{nodeCount} nodes · {edgeCount} edges
			</span>
			{#if centerId}
				<span class="hidden lg:inline text-[#ffffff]/[0.5] whitespace-nowrap">
					center={centerId.slice(0, 8)}
				</span>
			{/if}
		</div>

		<!-- Right: frame, fps, controls — frame padded so the counter never jitters -->
		<div class="flex items-center gap-3">
			<span class="text-[#ffffff]/[0.55] whitespace-nowrap"
				>frame: {String(frameCount).padStart(3, ' ')}</span
			>
			{#if freezeFrame !== null}
				<span class="text-[#a6dcff] tracking-widest whitespace-nowrap">CAPTURE</span>
			{:else if fpsEstimate > 0}
				<span class="text-[#5dcaa5] whitespace-nowrap w-[6ch] text-right">
					{fpsEstimate}fps
				</span>
			{/if}
			<button
				class="text-[#ffffff]/[0.5] hover:text-[#5dcaa5] transition-colors cursor-pointer pointer-events-auto whitespace-nowrap"
				onclick={copyDemoUrl}
				title="Copy shareable demo URL"
			>
				[url]
			</button>
		</div>
	</div>
</div>

<!-- loading/error props are accepted for API parity; the route renders those states itself -->

