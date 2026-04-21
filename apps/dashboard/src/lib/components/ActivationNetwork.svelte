<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		activationColor,
		applyDecay,
		edgeStagger,
		initialActivation,
		isVisible,
		layoutNeighbours,
	} from './activation-helpers';

	/**
	 * ActivationNetwork — visualizes spreading activation (Collins & Loftus 1975)
	 * across the cognitive memory graph.
	 *
	 * Every burst places a source node at the center with activated neighbours
	 * on concentric rings. Edges draw in with a staggered delay to visualize the
	 * activation wavefront; ripple waves expand outward from the source on each
	 * burst; activation level decays every animation frame by 0.93 until it
	 * drops below 0.05, at which point the node fades out.
	 *
	 * The component supports multiple overlapping bursts — each call to
	 * `trigger(sourceId, sourceLabel, neighbours)` merges into the current
	 * activation state rather than replacing it, so live mode feels like a
	 * continuous neural storm instead of a reset.
	 */

	export interface ActivationNode {
		id: string;
		label: string;
		nodeType: string;
		// Neighbours only — omit on source
		score?: number;
	}

	interface Props {
		width?: number;
		height?: number;
		/** Current focused burst; null when idle */
		source?: ActivationNode | null;
		/** Neighbours of the current focused burst (drawn immediately on mount) */
		neighbours?: ActivationNode[];
		/** Bursts triggered via live mode — each one overlays on the graph */
		liveBurstKey?: number;
		liveBurst?: { source: ActivationNode; neighbours: ActivationNode[] } | null;
	}

	let {
		width = 900,
		height = 560,
		source = null,
		neighbours = [],
		liveBurstKey = 0,
		liveBurst = null,
	}: Props = $props();

	// Decay/geometry constants live in `./activation-helpers` so the pure-
	// function test suite can exercise them without rendering Svelte. These
	// two visual-only constants stay local because they're tied to the SVG
	// node drawing below.
	const SOURCE_RADIUS = 22;
	const NEIGHBOUR_RADIUS_BASE = 14;

	interface ActiveNode {
		id: string;
		label: string;
		nodeType: string;
		x: number;
		y: number;
		activation: number; // 0..1 — drives size and opacity
		isSource: boolean;
		sourceBurstId: number; // which burst does this node belong to
	}

	interface ActiveEdge {
		burstId: number;
		sourceNodeId: string;
		targetNodeId: string;
		// Each edge has its own draw progress so we can stagger them
		drawProgress: number; // 0..1
		staggerDelay: number; // frames to wait before drawing
		framesElapsed: number;
	}

	interface Ripple {
		burstId: number;
		x: number;
		y: number;
		radius: number;
		opacity: number;
	}

	let activeNodes = $state<ActiveNode[]>([]);
	let activeEdges = $state<ActiveEdge[]>([]);
	let ripples = $state<Ripple[]>([]);

	let burstCounter = 0;
	let animationFrame: number | null = null;
	let lastPropSource: string | null = null;
	let lastLiveKey = 0;

	function triggerBurst(
		src: ActivationNode,
		nbrs: ActivationNode[],
		centerX: number,
		centerY: number
	) {
		burstCounter += 1;
		const burstId = burstCounter;

		// If a live burst hits the same source that is already at center, offset
		// it slightly so the visual distinction is preserved without chaos.
		const jitter = liveBurstKey > 0 && activeNodes.length > 0 ? 40 : 0;
		const cx = centerX + (Math.random() - 0.5) * jitter;
		const cy = centerY + (Math.random() - 0.5) * jitter;

		// Ripple wave from this source
		ripples = [
			...ripples,
			{ burstId, x: cx, y: cy, radius: SOURCE_RADIUS, opacity: 0.75 },
			{ burstId, x: cx, y: cy, radius: SOURCE_RADIUS, opacity: 0.5 },
		];

		// Source node — full activation
		const sourceNode: ActiveNode = {
			id: `${src.id}::${burstId}`,
			label: src.label,
			nodeType: 'source',
			x: cx,
			y: cy,
			activation: 1,
			isSource: true,
			sourceBurstId: burstId,
		};

		const neighbourNodes: ActiveNode[] = [];
		const newEdges: ActiveEdge[] = [];

		// Slight rotation per burst so overlapping bursts don't fully collide
		const angleOffset = (burstCounter * 0.37) % (Math.PI * 2);
		const allPositions = layoutNeighbours(cx, cy, nbrs.length, angleOffset);

		nbrs.forEach((nbr, i) => {
			const pos = allPositions[i];
			if (!pos) return;

			neighbourNodes.push({
				id: `${nbr.id}::${burstId}`,
				label: nbr.label,
				nodeType: nbr.nodeType,
				x: pos.x,
				y: pos.y,
				activation: initialActivation(i, nbrs.length),
				isSource: false,
				sourceBurstId: burstId,
			});

			newEdges.push({
				burstId,
				sourceNodeId: sourceNode.id,
				targetNodeId: `${nbr.id}::${burstId}`,
				drawProgress: 0,
				staggerDelay: edgeStagger(i),
				framesElapsed: 0,
			});
		});

		activeNodes = [...activeNodes, sourceNode, ...neighbourNodes];
		activeEdges = [...activeEdges, ...newEdges];
	}

	function tick() {
		// Decay node activations (Collins & Loftus 1975, 0.93/frame).
		let nextNodes: ActiveNode[] = [];
		for (const n of activeNodes) {
			const nextActivation = applyDecay(n.activation);
			if (!isVisible(nextActivation)) continue;
			nextNodes.push({ ...n, activation: nextActivation });
		}
		activeNodes = nextNodes;

		// Advance edge draw progress (only for edges whose endpoints still exist)
		const liveIds = new Set(nextNodes.map((n) => n.id));
		let nextEdges: ActiveEdge[] = [];
		for (const e of activeEdges) {
			if (!liveIds.has(e.sourceNodeId) || !liveIds.has(e.targetNodeId)) continue;
			const elapsed = e.framesElapsed + 1;
			let progress = e.drawProgress;
			if (elapsed >= e.staggerDelay) {
				// 0..1 over ~15 frames (~0.25s at 60fps)
				progress = Math.min(1, progress + 1 / 15);
			}
			nextEdges.push({ ...e, framesElapsed: elapsed, drawProgress: progress });
		}
		activeEdges = nextEdges;

		// Expand ripples outward, fade opacity
		let nextRipples: Ripple[] = [];
		for (const r of ripples) {
			const nextRadius = r.radius + 6;
			const nextOpacity = r.opacity * 0.96;
			if (nextOpacity < 0.02 || nextRadius > Math.max(width, height)) continue;
			nextRipples.push({ ...r, radius: nextRadius, opacity: nextOpacity });
		}
		ripples = nextRipples;

		animationFrame = requestAnimationFrame(tick);
	}

	function clearBursts() {
		activeNodes = [];
		activeEdges = [];
		ripples = [];
	}

	// Watch for prop-driven bursts (initial search result)
	$effect(() => {
		if (!source) return;
		const sourceKey = source.id;
		if (sourceKey === lastPropSource) return;
		lastPropSource = sourceKey;
		clearBursts();
		triggerBurst(source, neighbours, width / 2, height / 2);
	});

	// Watch for live bursts — each keyed trigger overlays a new burst at a
	// random-ish location near center so they don't stack directly on top.
	$effect(() => {
		if (!liveBurst || liveBurstKey === 0) return;
		if (liveBurstKey === lastLiveKey) return;
		lastLiveKey = liveBurstKey;
		// Live bursts land near but not exactly on center so they're visually
		// distinct from the primary burst.
		const offsetX = (Math.random() - 0.5) * 120;
		const offsetY = (Math.random() - 0.5) * 120;
		triggerBurst(liveBurst.source, liveBurst.neighbours, width / 2 + offsetX, height / 2 + offsetY);
	});

	onMount(() => {
		animationFrame = requestAnimationFrame(tick);
	});

	onDestroy(() => {
		if (animationFrame !== null) cancelAnimationFrame(animationFrame);
	});

	function nodeColor(nodeType: string, isSource: boolean): string {
		return activationColor(nodeType, isSource);
	}

	function edgePoint(edge: ActiveEdge): {
		x1: number;
		y1: number;
		x2: number;
		y2: number;
	} | null {
		const src = activeNodes.find((n) => n.id === edge.sourceNodeId);
		const tgt = activeNodes.find((n) => n.id === edge.targetNodeId);
		if (!src || !tgt) return null;
		// Clip to current draw progress so the edge grows outward from source
		const x2 = src.x + (tgt.x - src.x) * edge.drawProgress;
		const y2 = src.y + (tgt.y - src.y) * edge.drawProgress;
		return { x1: src.x, y1: src.y, x2, y2 };
	}
</script>

<svg
	{width}
	{height}
	viewBox="0 0 {width} {height}"
	class="w-full h-full block"
	aria-label="Spreading activation visualization"
>
	<defs>
		<filter id="act-glow" x="-50%" y="-50%" width="200%" height="200%">
			<feGaussianBlur stdDeviation="4" result="blur" />
			<feMerge>
				<feMergeNode in="blur" />
				<feMergeNode in="SourceGraphic" />
			</feMerge>
		</filter>
		<filter id="act-glow-strong" x="-100%" y="-100%" width="300%" height="300%">
			<feGaussianBlur stdDeviation="8" result="blur" />
			<feMerge>
				<feMergeNode in="blur" />
				<feMergeNode in="SourceGraphic" />
			</feMerge>
		</filter>
		<radialGradient id="ripple-grad" cx="50%" cy="50%" r="50%">
			<stop offset="70%" stop-color="#818cf8" stop-opacity="0" />
			<stop offset="100%" stop-color="#818cf8" stop-opacity="0.7" />
		</radialGradient>
	</defs>

	<!-- Ripple wavefronts (expanding circles from source) -->
	{#each ripples as r, i (i)}
		<circle
			cx={r.x}
			cy={r.y}
			r={r.radius}
			fill="none"
			stroke="#818cf8"
			stroke-width="1.5"
			opacity={r.opacity}
		/>
	{/each}

	<!-- Edges (drawn with stagger delay so activation appears to spread) -->
	{#each activeEdges as e, i (i)}
		{@const pt = edgePoint(e)}
		{#if pt}
			<line
				x1={pt.x1}
				y1={pt.y1}
				x2={pt.x2}
				y2={pt.y2}
				stroke="#818cf8"
				stroke-width="1.2"
				stroke-linecap="round"
				opacity={0.35 * e.drawProgress}
			/>
		{/if}
	{/each}

	<!-- Nodes -->
	{#each activeNodes as n (n.id)}
		{@const color = nodeColor(n.nodeType, n.isSource)}
		{@const r = n.isSource
			? SOURCE_RADIUS * (0.7 + 0.3 * n.activation)
			: NEIGHBOUR_RADIUS_BASE * (0.5 + 0.8 * n.activation)}
		<g opacity={Math.min(1, n.activation * 1.25)}>
			<!-- Soft outer glow halo -->
			<circle
				cx={n.x}
				cy={n.y}
				r={r * 1.9}
				fill={color}
				opacity={0.18 * n.activation}
				filter="url(#act-glow-strong)"
			/>
			<!-- Core -->
			<circle
				cx={n.x}
				cy={n.y}
				r={r}
				fill={color}
				filter="url(#act-glow)"
			/>
			<!-- Inner highlight for depth -->
			<circle
				cx={n.x - r * 0.3}
				cy={n.y - r * 0.3}
				r={r * 0.35}
				fill="#ffffff"
				opacity={0.35 * n.activation}
			/>
			{#if n.isSource && n.label}
				<text
					x={n.x}
					y={n.y + r + 18}
					text-anchor="middle"
					fill="#e0e0ff"
					font-size="11"
					font-family="var(--font-mono)"
					opacity={0.9 * n.activation}
				>
					{n.label.length > 40 ? n.label.slice(0, 40) + '…' : n.label}
				</text>
			{/if}
		</g>
	{/each}
</svg>
