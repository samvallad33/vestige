<script lang="ts">
	import { onDestroy } from 'svelte';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { TextLayerPass } from '$lib/observatory/text/text-layer';

	let hostEl: HTMLDivElement | null = $state(null);
	let textPass: TextLayerPass | null = null;
	let engineRef: ObservatoryEngine | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;

	async function handleReady(engine: ObservatoryEngine) {
		engineRef = engine;
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
			maxWidthEm: 18,
			depth: 0.51,
			weight: 0.51
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

	function pointerToNdc(e: PointerEvent): { x: number; y: number } | null {
		if (!hostEl) return null;
		const rect = hostEl.getBoundingClientRect();
		if (rect.width <= 0 || rect.height <= 0) return null;
		return {
			x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
			y: -(((e.clientY - rect.top) / rect.height) * 2 - 1)
		};
	}

	function handlePointerMove(e: PointerEvent) {
		if (!hostEl || !engineRef) return;
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		const rect = hostEl.getBoundingClientRect();
		const aspect = Math.max(0.0001, rect.width / Math.max(1, rect.height));
		const raw = {
			x: ndc.x * Math.max(aspect, 1),
			y: ndc.y / Math.min(aspect, 1)
		};
		const prev = cursorSmoothed ?? raw;
		const next = {
			x: prev.x + (raw.x - prev.x) * 0.35,
			y: prev.y + (raw.y - prev.y) * 0.35
		};
		cursorSmoothed = next;
		engineRef.setCursorPreNdc(next.x, next.y, next.x - prev.x, next.y - prev.y);
		const hit = textPass?.pickAt(ndc.x, ndc.y);
		textPass?.setRunDepth(hit?.id ?? null, hit ? 1.0 : 0.51);
	}

	function handlePointerLeave() {
		cursorSmoothed = null;
		engineRef?.setCursorPreNdc(999, 999, 0, 0);
		textPass?.setRunDepth(null, 0.51);
	}

	onDestroy(() => {
		textPass?.dispose();
		textPass = null;
		engineRef = null;
	});
</script>

<!-- Transparent pointer host only; ObservatoryCanvas owns the single visible canvas. -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={hostEl} class="fixed inset-0 bg-[#020307]" onpointerdown={handlePointerDown} onpointermove={handlePointerMove} onpointerleave={handlePointerLeave}>
	<ObservatoryCanvas demo="recall-path" seed="msdf-test-v1" onready={handleReady} />
</div>
