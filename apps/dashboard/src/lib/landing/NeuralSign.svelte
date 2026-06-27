<script lang="ts">
	// Living neural sign: VESTIGE built from REAL
	// grown dendrites (space colonization on the letterforms) with a quiet bloom.
	// Not glow-on-text.
	import { onMount } from 'svelte';
	import { growSign, type DendriteSign } from '$lib/landing/dendriteGen';

	let sign = $state<DendriteSign | null>(null);
	let reduced = $state(false);

	onMount(() => {
		reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
		// grow the dendrites once after fonts are ready (so glyph metrics are right)
		const grow = () => {
			try {
				sign = growSign();
			} catch (e) {
				console.warn('[neural-sign] grow failed:', e);
			}
		};
		if (document.fonts?.ready) {
			document.fonts.load('900 150px Inter').then(() => document.fonts.ready).then(grow).catch(grow);
		} else {
			grow();
		}
	});

</script>

<div class="neural-sign" class:reduced aria-label="Vestige">
	{#if sign}
		<svg
			viewBox={`0 0 ${sign.width} ${sign.height}`}
			class="sign-svg"
			xmlns="http://www.w3.org/2000/svg"
			aria-hidden="true"
		>
			<defs>
				<filter id="ns-bloom" x="-60%" y="-60%" width="220%" height="220%">
					<feGaussianBlur in="SourceAlpha" stdDeviation="1.6" result="b1" />
					<feColorMatrix in="b1" values="0 0 0 0 .22  0 0 0 0 1  0 0 0 0 .62  0 0 0 1 0" result="g1" />
					<feGaussianBlur in="SourceAlpha" stdDeviation="5" result="b2" />
					<feColorMatrix in="b2" values="0 0 0 0 .13  0 0 0 0 .82  0 0 0 0 1  0 0 0 .85 0" result="g2" />
					<feGaussianBlur in="SourceAlpha" stdDeviation="12" result="b3" />
					<feColorMatrix in="b3" values="0 0 0 0 .70  0 0 0 0 .42  0 0 0 0 1  0 0 0 .6 0" result="g3" />
					<feMerge>
						<feMergeNode in="g3" /><feMergeNode in="g2" /><feMergeNode in="g1" />
						<feMergeNode in="SourceGraphic" />
					</feMerge>
				</filter>
			</defs>

			<!-- the grown dendrites -->
			<g filter="url(#ns-bloom)">
				<g class="dendrites">
					{#each sign.paths as p (p.d)}
						<path d={p.d} stroke={p.col} stroke-width={p.w} stroke-linecap="round" />
					{/each}
				</g>
			</g>
		</svg>
	{/if}
</div>

<style>
	.neural-sign {
		position: fixed;
		top: clamp(0.1rem, 1vh, 0.8rem);
		left: 50%;
		transform: translateX(-50%);
		z-index: 3;
		width: min(680px, 94vw);
		pointer-events: none;
	}
	.sign-svg {
		width: 100%;
		height: auto;
		display: block;
		overflow: visible;
	}

	/* PERF: per-path stroke-dashoffset animation on 5700 paths tanks FPS to ~12
	   (SVG dashoffset is not GPU-compositable; full paint per path per frame).
	   Instead fade in the WHOLE dendrite group with ONE GPU-compositable opacity
	   transition. Static dendrites + one group fade = 120fps. */
	.dendrites {
		animation: ns-grow 1.6s ease-out both;
	}
	@keyframes ns-grow {
		from {
			opacity: 0;
			transform: scale(0.92);
			transform-origin: 50% 30%;
		}
		to {
			opacity: 1;
			transform: scale(1);
		}
	}

	.reduced .dendrites {
		animation: none;
	}
</style>
