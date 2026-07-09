<script lang="ts">
	/**
	 * /observatory route — thin wrapper over ObservatoryStage.
	 *
	 * The full experience lives in $lib/observatory/ObservatoryStage.svelte so
	 * the MAIN dashboard graph page can host it too (its primary home). This
	 * route keeps the deep-link + recording contract byte-compatible:
	 *   ?demo=recall-path&seed=vestige-observatory-v1  — pick the moment
	 *   ?frame=N                                       — deterministic freeze
	 *   ?capture=1 (&hud=1)                            — chrome-free recording
	 */
	import { goto } from '$app/navigation';
	import { base } from '$app/paths';
	import ObservatoryStage from '$lib/observatory/ObservatoryStage.svelte';
	import { isDemoMode, type DemoMode } from '$lib/observatory/types';

	const params = new URLSearchParams(window.location.search);
	const demoParam = params.get('demo') ?? 'recall-path';
	let demo = $state<DemoMode>(isDemoMode(demoParam) ? demoParam : 'recall-path');
	const seedValue = params.get('seed') ?? 'vestige-observatory-v1';
	// Capture mode: ?frame=N freezes the sim at one loop frame (identical pixels).
	const frameParam = params.get('frame');
	const freezeFrame = frameParam !== null && frameParam !== '' ? Number(frameParam) : null;
	// Recording mode: ?capture=1 hides EVERY DOM instrument — pure canvas.
	const isCapture = params.get('capture') === '1' && params.get('hud') !== '1';

	function switchDemo(next: DemoMode) {
		if (next === demo) return;
		demo = next;
		// Keep the URL shareable — same contract as arriving via deep link.
		const url = new URL(window.location.href);
		url.searchParams.set('demo', next);
		history.replaceState(history.state, '', url);
	}
</script>

<!-- {#key} forces a full remount on demo switch: fresh engine, plans, clock —
     deterministic from frame 0, no partial-state carryover. -->
{#key demo}
	<ObservatoryStage
		{demo}
		seed={seedValue}
		{freezeFrame}
		capture={isCapture}
		showSwitcher={freezeFrame === null}
		ondemochange={switchDemo}
		onexit={() => goto(`${base}/graph`)}
	/>
{/key}
