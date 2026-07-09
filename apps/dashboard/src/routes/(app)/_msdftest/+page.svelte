<script lang="ts">
	import { onDestroy } from 'svelte';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { TextLayerPass } from '$lib/observatory/text/text-layer';

	let hostEl: HTMLDivElement | null = $state(null);
	let textPass: TextLayerPass | null = null;

	async function handleReady(engine: ObservatoryEngine) {
		const pass = new TextLayerPass(engine);
		textPass = pass;
		await pass.init();
		pass.setText({
			id: 'msdf-test-line',
			kind: 'msdf-test',
			text: 'hello | 5de3e41f | trust 51%',
			x: -0.62,
			y: 0.03,
			size: 0.105,
			color: [0.14, 0.78, 0.87, 1],
			startFrame: 0,
			revealSpan: 20,
			maxWidthEm: 18
		});
		engine.addPass(pass);
		engine.demoClock.reset();
	}

	function handlePointerDown(e: PointerEvent) {
		if (!hostEl || !textPass) return;
		const rect = hostEl.getBoundingClientRect();
		if (rect.width <= 0 || rect.height <= 0) return;
		const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
		const ndcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
		const hit = textPass.pickAt(ndcX, ndcY);
		if (hit) console.info('[msdf-test] picked', hit.id);
	}

	onDestroy(() => {
		textPass?.dispose();
		textPass = null;
	});
</script>

<!-- Transparent pointer host only; ObservatoryCanvas owns the single visible canvas. -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={hostEl} class="fixed inset-0 bg-[#020307]" onpointerdown={handlePointerDown}>
	<ObservatoryCanvas demo="recall-path" seed="msdf-test-v1" onready={handleReady} />
</div>
