<script lang="ts">
	// VESTIGE built from REAL grown dendrites (space colonization on the
	// letterforms via growSign) — the letters ARE organic neural branches we
	// generate, not a font. Rendered as GLOWING neon dendrites with luminous
	// synapse nodes, matching the live particle brain. The SHAPE itself is alive.
	import { onMount } from 'svelte';
	import { growSign, type DendriteSign } from '$lib/landing/dendriteGen';

	type Detail = 'desktop' | 'mobile';

	const SIGN_CACHE_PREFIX = 'vestige-launch-dendrite-sign-v3';
	const NODE_COLORS = ['#7dffb0', '#52e6ff', '#c79bff', '#9b8cff'];

	let sign = $state<DendriteSign | null>(null);
	let nodes = $state<Array<{ x: number; y: number; r: number; c: string }>>([]);
	let reduced = $state(false);
	let drawn = $state(false);
	let detail = $state<Detail>('desktop');

	function cacheKey(nextDetail: Detail) {
		return `${SIGN_CACHE_PREFIX}-${nextDetail}`;
	}

	function isDendriteSign(value: unknown): value is DendriteSign {
		const candidate = value as DendriteSign | null;
		return Boolean(
			candidate &&
				typeof candidate.width === 'number' &&
				typeof candidate.height === 'number' &&
				Array.isArray(candidate.paths) &&
				candidate.paths.length > 0
		);
	}

	function readCachedSign(key: string): DendriteSign | null {
		try {
			const raw = sessionStorage.getItem(key);
			if (!raw) return null;
			const parsed = JSON.parse(raw) as unknown;
			return isDendriteSign(parsed) ? parsed : null;
		} catch {
			return null;
		}
	}

	function writeCachedSign(key: string, value: DendriteSign) {
		try {
			sessionStorage.setItem(key, JSON.stringify(value));
		} catch {
			/* storage may be unavailable in private mode */
		}
	}

	function synapseNodes(value: DendriteSign) {
		const pts: Array<{ x: number; y: number; r: number; c: string }> = [];
		for (let i = 0; i < value.paths.length; i += 18) {
			const m = value.paths[i].d.match(/M([\d.-]+) ([\d.-]+)L/);
			if (!m) continue;
			pts.push({
				x: Number(m[1]),
				y: Number(m[2]),
				r: 1.1 + (i % 3) * 0.5,
				c: NODE_COLORS[i % NODE_COLORS.length]
			});
		}
		return pts;
	}

	function applySign(value: DendriteSign) {
		sign = value;
		nodes = synapseNodes(value);
		drawn = true;
	}

	onMount(() => {
		reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
		const coarsePointer = window.matchMedia?.('(pointer: coarse)').matches ?? false;
		const narrowViewport = window.innerWidth <= 820;
		detail = coarsePointer || narrowViewport ? 'mobile' : 'desktop';
		if (detail === 'mobile') {
			// Mobile signup must stay instantly interactive. The generated dendrite
			// SVG is beautiful, but path growth/rendering can monopolize WebKit's
			// main thread on a cold load. Keep the bridge wordmark and let the graph
			// own mobile's first seconds.
			return;
		}
		const key = cacheKey(detail);
		const cached = readCachedSign(key);
		if (cached) {
			applySign(cached);
			return;
		}

		const grow = () => {
			try {
				const s = growSign();
				writeCachedSign(key, s);
				applySign(s);
			} catch (e) {
				console.warn('[neural-wordmark] grow failed:', e);
			}
		};
		requestAnimationFrame(grow);
	});
</script>

<div class="neural-wordmark" class:mobile={detail === 'mobile'} class:reduced class:drawn aria-label="Vestige">
	{#if sign}
		<svg
			viewBox={`0 0 ${sign.width} ${sign.height}`}
			class="sign-svg"
			xmlns="http://www.w3.org/2000/svg"
			aria-hidden="true"
		>
			<defs>
				<!-- iridescent neon gradient (green -> cyan -> violet) that flows -->
				<linearGradient id="nw-grad" x1="0%" y1="0%" x2="100%" y2="0%">
					<stop offset="0%" stop-color="#5dffa6" />
					<stop offset="30%" stop-color="#36f0ff" />
					<stop offset="55%" stop-color="#6db0ff" />
					<stop offset="80%" stop-color="#b98cff" />
					<stop offset="100%" stop-color="#5dffa6" />
					<animate attributeName="x1" values="0%;100%;0%" dur="7s" repeatCount="indefinite" />
					<animate attributeName="x2" values="100%;200%;100%" dur="7s" repeatCount="indefinite" />
				</linearGradient>

				<!-- HEAVY neon bloom (matches the live particle glow) -->
				<filter id="nw-bloom" x="-40%" y="-90%" width="180%" height="280%">
					<feGaussianBlur in="SourceAlpha" stdDeviation="1.2" result="b0" />
					<feColorMatrix
						in="b0"
						values="0 0 0 0 .55  0 0 0 0 1  0 0 0 0 .75  0 0 0 1 0"
						result="g0"
					/>
					<feGaussianBlur in="SourceAlpha" stdDeviation="4" result="b1" />
					<feColorMatrix
						in="b1"
						values="0 0 0 0 .30  0 0 0 0 .92  0 0 0 0 1  0 0 0 .9 0"
						result="g1"
					/>
					<feGaussianBlur in="SourceAlpha" stdDeviation="10" result="b2" />
					<feColorMatrix
						in="b2"
						values="0 0 0 0 .55  0 0 0 0 .50  0 0 0 0 1  0 0 0 .85 0"
						result="g2"
					/>
					<feGaussianBlur in="SourceAlpha" stdDeviation="22" result="b3" />
					<feColorMatrix
						in="b3"
						values="0 0 0 0 .42  0 0 0 0 1  0 0 0 0 .68  0 0 0 .6 0"
						result="g3"
					/>
					<feMerge>
						<feMergeNode in="g3" />
						<feMergeNode in="g2" />
						<feMergeNode in="g1" />
						<feMergeNode in="g0" />
						<feMergeNode in="SourceGraphic" />
					</feMerge>
				</filter>

				<!-- subtle point-glow for the synapse nodes (small + tight) -->
				<filter id="nw-node" x="-200%" y="-200%" width="500%" height="500%">
					<feGaussianBlur in="SourceGraphic" stdDeviation="1.4" result="nb" />
					<feMerge>
						<feMergeNode in="nb" />
						<feMergeNode in="SourceGraphic" />
					</feMerge>
				</filter>
			</defs>

			<g class="bloom-wrap" filter={detail === 'mobile' ? undefined : 'url(#nw-bloom)'}>
				<g class="dendrites" stroke="url(#nw-grad)">
					{#each sign.paths as p (p.d)}
						<path d={p.d} stroke-width={Math.max(p.w, 1.4)} stroke-linecap="round" />
					{/each}
				</g>
			</g>

			<!-- luminous synapse nodes (the bright dots in the reference) -->
			<g class="synapses" filter="url(#nw-node)">
				{#each nodes as n (n.x + '_' + n.y)}
					<circle cx={n.x} cy={n.y} r={n.r} fill={n.c} />
				{/each}
			</g>
		</svg>
	{:else}
		<span class="wordmark-bridge" aria-hidden="true">VESTIGE</span>
	{/if}
</div>

<style>
		.neural-wordmark {
			width: min(760px, 94vw);
			min-height: clamp(4.2rem, 16vw, 8.5rem);
			margin: 0 auto;
			pointer-events: none;
			display: flex;
			align-items: center;
			justify-content: center;
		}
		.sign-svg {
			width: 100%;
			height: auto;
			display: block;
			overflow: visible;
		}
		.wordmark-bridge {
			display: block;
			font-size: clamp(2.45rem, 12vw, 7.8rem);
			font-weight: 900;
			line-height: 0.9;
			letter-spacing: 0;
			background: linear-gradient(92deg, #5dffa6 0%, #36f0ff 42%, #b98cff 78%, #5dffa6 100%);
			-webkit-background-clip: text;
			background-clip: text;
			color: transparent;
			filter:
				drop-shadow(0 0 8px rgba(93, 255, 166, 0.55))
				drop-shadow(0 0 26px rgba(82, 230, 255, 0.38));
		}

	/* one-shot grow-in (scale+fade), then a perpetual gentle breathing so the
	   neural SHAPE stays alive — no bouncing letters, the structure shimmers. */
	.dendrites,
	.synapses {
		opacity: 0;
		transform: scale(0.94);
		transform-origin: 50% 42%;
		transition:
			opacity 1.2s ease,
			transform 1.2s cubic-bezier(0.16, 1, 0.3, 1);
	}
	.drawn .dendrites,
	.drawn .synapses {
		opacity: 1;
		transform: scale(1);
	}
	.drawn .dendrites {
		animation: nw-breathe 5.5s ease-in-out infinite 1.1s;
	}
	/* synapse nodes twinkle independently for that living-circuit feel */
	.drawn .synapses {
		animation: nw-twinkle 2.6s ease-in-out infinite 1.1s;
	}
	@keyframes nw-breathe {
		0%,
		100% {
			transform: scale(1);
			filter: brightness(1);
		}
		50% {
			transform: scale(1.012);
			filter: brightness(1.22);
		}
	}
	@keyframes nw-twinkle {
		0%,
		100% {
			opacity: 0.85;
		}
		50% {
			opacity: 1;
		}
	}

	.reduced .dendrites,
	.reduced .synapses {
		opacity: 1;
		transform: none;
		transition: none;
		animation: none;
	}
</style>
