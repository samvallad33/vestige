<script lang="ts">
	import { onMount } from 'svelte';
	import {
		AXIS_ORDER,
		clampChannels,
		radarRadius,
		sizePreset,
	} from './importance-helpers';

	interface Props {
		novelty: number;
		arousal: number;
		reward: number;
		attention: number;
		size?: 'sm' | 'md' | 'lg';
	}

	let { novelty, arousal, reward, attention, size = 'md' }: Props = $props();

	// Size presets + padding + clamp logic live in `importance-helpers.ts` so
	// they can be tested in the Vitest node env without a jsdom harness.
	let pxSize = $derived(sizePreset(size));
	let showLabels = $derived(size !== 'sm');
	let radius = $derived(radarRadius(size));
	let cx = $derived(pxSize / 2);
	let cy = $derived(pxSize / 2);

	// Axis labels live alongside AXIS_ORDER's keys/angles so the renderer
	// never drifts from the helper geometry.
	const AXIS_LABELS: Record<(typeof AXIS_ORDER)[number]['key'], string> = {
		novelty: 'Novelty',
		arousal: 'Arousal',
		reward: 'Reward',
		attention: 'Attention'
	};

	let values = $derived(clampChannels({ novelty, arousal, reward, attention }));

	function pointAt(value: number, angle: number): [number, number] {
		const r = value * radius;
		return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r];
	}

	// Grid rings at 25/50/75/100%.
	const RINGS = [0.25, 0.5, 0.75, 1];

	function ringPath(frac: number): string {
		const pts = AXIS_ORDER.map(({ angle }) => pointAt(frac, angle));
		return (
			pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ') + ' Z'
		);
	}

	// Mount-time grow-from-center animation. We scale the values from 0 to 1
	// over ~600ms with an easeOutCubic curve so the polygon blossoms instead of
	// popping in. Reactive on prop change would also be nice but this matches
	// the brief ("animated fill-in on mount").
	let animProgress = $state(0);

	onMount(() => {
		const duration = 600;
		const start = performance.now();
		let raf = 0;
		const tick = (now: number) => {
			const t = Math.min(1, (now - start) / duration);
			// easeOutCubic
			animProgress = 1 - Math.pow(1 - t, 3);
			if (t < 1) raf = requestAnimationFrame(tick);
		};
		raf = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(raf);
	});

	let polygonPath = $derived.by(() => {
		const scale = animProgress;
		const pts = AXIS_ORDER.map(({ key, angle }) => pointAt(values[key] * scale, angle));
		return (
			pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ') + ' Z'
		);
	});

	// Label positions sit slightly outside the 100% ring.
	function labelPos(angle: number): { x: number; y: number; anchor: 'start' | 'middle' | 'end' } {
		const offset = radius + (size === 'lg' ? 18 : 12);
		const x = cx + Math.cos(angle) * offset;
		const y = cy + Math.sin(angle) * offset;
		let anchor: 'start' | 'middle' | 'end' = 'middle';
		if (Math.abs(Math.cos(angle)) > 0.5) {
			anchor = Math.cos(angle) > 0 ? 'start' : 'end';
		}
		return { x, y, anchor };
	}
</script>

<svg
	width={pxSize}
	height={pxSize}
	viewBox="0 0 {pxSize} {pxSize}"
	role="img"
	aria-label="Importance radar: novelty {(values.novelty * 100).toFixed(0)}%, arousal {(values.arousal * 100).toFixed(0)}%, reward {(values.reward * 100).toFixed(0)}%, attention {(values.attention * 100).toFixed(0)}%"
>
	<!-- Concentric rings (grid) -->
	{#each RINGS as ring}
		<path
			d={ringPath(ring)}
			fill="none"
			stroke="var(--color-muted)"
			stroke-opacity={ring === 1 ? 0.45 : 0.18}
			stroke-width={ring === 1 ? 1 : 0.75}
		/>
	{/each}

	<!-- Spokes -->
	{#each AXIS_ORDER as axis}
		{@const [x, y] = pointAt(1, axis.angle)}
		<line
			x1={cx}
			y1={cy}
			x2={x}
			y2={y}
			stroke="var(--color-muted)"
			stroke-opacity="0.2"
			stroke-width="0.75"
		/>
	{/each}

	<!-- Filled polygon -->
	<path
		d={polygonPath}
		fill="var(--color-synapse-glow)"
		fill-opacity="0.3"
		stroke="var(--color-synapse-glow)"
		stroke-width={size === 'sm' ? 1 : 1.5}
		stroke-linejoin="round"
	/>

	<!-- Vertex dots at the animated positions, only on md/lg for clarity -->
	{#if size !== 'sm'}
		{#each AXIS_ORDER as axis}
			{@const [px, py] = pointAt(values[axis.key] * animProgress, axis.angle)}
			<circle cx={px} cy={py} r={size === 'lg' ? 3 : 2.25} fill="var(--color-synapse-glow)" />
		{/each}
	{/if}

	<!-- Axis labels (name + value) -->
	{#if showLabels}
		{#each AXIS_ORDER as axis}
			{@const pos = labelPos(axis.angle)}
			<text
				x={pos.x}
				y={pos.y}
				text-anchor={pos.anchor}
				dominant-baseline="middle"
				fill="var(--color-bright)"
				font-size={size === 'lg' ? 12 : 10}
				font-family="var(--font-mono)"
				font-weight="600"
			>
				{(values[axis.key] * 100).toFixed(0)}%
			</text>
			<text
				x={pos.x}
				y={pos.y + (size === 'lg' ? 14 : 11)}
				text-anchor={pos.anchor}
				dominant-baseline="middle"
				fill="var(--color-dim)"
				font-size={size === 'lg' ? 10 : 8.5}
				font-family="var(--font-mono)"
			>
				{AXIS_LABELS[axis.key]}
			</text>
		{/each}
	{/if}
</svg>
