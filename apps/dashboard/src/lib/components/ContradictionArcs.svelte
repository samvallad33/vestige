<script lang="ts">
	/**
	 * ContradictionArcs — 2D cosmic constellation of conflicting memories.
	 *
	 * Renders each contradiction pair as two nodes connected by an arc.
	 * Arc color = similarity severity (red/amber/yellow).
	 * Arc thickness = min(trust_a, trust_b) — stake.
	 * Node size = trust score. Node hue = node_type (bioluminescent).
	 *
	 * SVG-only (no Three.js) to keep it lightweight and print-friendly.
	 */

	import {
		nodeColor,
		severityColor,
		severityLabel,
		nodeRadius,
		pairOpacity,
		truncate,
	} from './contradiction-helpers';

	export interface Contradiction {
		memory_a_id: string;
		memory_b_id: string;
		memory_a_preview: string;
		memory_b_preview: string;
		memory_a_type?: string;
		memory_b_type?: string;
		memory_a_created?: string;
		memory_b_created?: string;
		memory_a_tags?: string[];
		memory_b_tags?: string[];
		trust_a: number; // 0..1
		trust_b: number; // 0..1
		similarity: number; // 0..1
		date_diff_days: number;
		topic: string;
	}

	interface Props {
		contradictions: Contradiction[];
		focusedPairIndex?: number | null;
		onSelectPair?: (index: number | null) => void;
		width?: number;
		height?: number;
	}

	let {
		contradictions,
		focusedPairIndex = null,
		onSelectPair,
		width = 800,
		height = 600
	}: Props = $props();

	// --- Polar layout: place pairs around a circle, with the arc crossing the interior. ---
	// Each pair is given a slot on the circumference; node A and node B are placed
	// symmetrically around that slot at a small angular offset proportional to similarity
	// (more similar = farther apart visually, so the tension is readable).
	// Wrapped in $derived so Svelte 5 re-computes when $state `width`/`height` change,
	// instead of capturing their initial values once.
	const geom = $derived.by(() => {
		const cx = width / 2;
		const cy = height / 2;
		const R = Math.min(width, height) * 0.38;
		return { cx, cy, R };
	});

	interface NodePoint {
		x: number;
		y: number;
		trust: number;
		preview: string;
		type?: string;
		created?: string;
		tags?: string[];
		memoryId: string;
		pairIndex: number;
		side: 'a' | 'b';
	}

	interface ArcShape {
		pairIndex: number;
		path: string;
		color: string;
		thickness: number;
		severity: string;
		topic: string;
		similarity: number;
		dateDiff: number;
		aPoint: NodePoint;
		bPoint: NodePoint;
		// Midpoint used for particle animation origin
		midX: number;
		midY: number;
	}

	const layout = $derived.by((): { nodes: NodePoint[]; arcs: ArcShape[] } => {
		const nodes: NodePoint[] = [];
		const arcs: ArcShape[] = [];
		const n = contradictions.length || 1;

		contradictions.forEach((c, i) => {
			const slot = (i / n) * Math.PI * 2 - Math.PI / 2;
			// Angular offset between A and B within the slot — proportional to sim
			const spread = 0.18 + c.similarity * 0.22;
			const angA = slot - spread;
			const angB = slot + spread;

			// Slight radial jitter so the constellation doesn't look like a perfect ring
			const rA = geom.R + (Math.sin(i * 2.3) * 18);
			const rB = geom.R + (Math.cos(i * 1.7) * 18);

			const aPoint: NodePoint = {
				x: geom.cx + Math.cos(angA) * rA,
				y: geom.cy + Math.sin(angA) * rA,
				trust: c.trust_a,
				preview: c.memory_a_preview,
				type: c.memory_a_type,
				created: c.memory_a_created,
				tags: c.memory_a_tags,
				memoryId: c.memory_a_id,
				pairIndex: i,
				side: 'a'
			};
			const bPoint: NodePoint = {
				x: geom.cx + Math.cos(angB) * rB,
				y: geom.cy + Math.sin(angB) * rB,
				trust: c.trust_b,
				preview: c.memory_b_preview,
				type: c.memory_b_type,
				created: c.memory_b_created,
				tags: c.memory_b_tags,
				memoryId: c.memory_b_id,
				pairIndex: i,
				side: 'b'
			};

			nodes.push(aPoint, bPoint);

			// Arc bends toward the centre — cosmic bridge.
			// Control point = midpoint pulled toward centre by (1 - similarity * 0.3)
			const mx = (aPoint.x + bPoint.x) / 2;
			const my = (aPoint.y + bPoint.y) / 2;
			const pullStrength = 0.55 - c.similarity * 0.25;
			const ctrlX = mx + (geom.cx - mx) * pullStrength;
			const ctrlY = my + (geom.cy - my) * pullStrength;

			const thickness = 1 + Math.min(c.trust_a, c.trust_b) * 4;
			arcs.push({
				pairIndex: i,
				path: `M ${aPoint.x.toFixed(1)} ${aPoint.y.toFixed(1)} Q ${ctrlX.toFixed(1)} ${ctrlY.toFixed(1)} ${bPoint.x.toFixed(1)} ${bPoint.y.toFixed(1)}`,
				color: severityColor(c.similarity),
				thickness,
				severity: severityLabel(c.similarity),
				topic: c.topic,
				similarity: c.similarity,
				dateDiff: c.date_diff_days,
				aPoint,
				bPoint,
				midX: ctrlX,
				midY: ctrlY
			});
		});

		return { nodes, arcs };
	});

	// --- Hover tooltip state ---
	let hoverNode = $state<NodePoint | null>(null);
	let hoverArc = $state<ArcShape | null>(null);
	let mouseX = $state(0);
	let mouseY = $state(0);

	function onMove(e: MouseEvent) {
		const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		mouseX = e.clientX - rect.left;
		mouseY = e.clientY - rect.top;
	}

	function handleArcClick(i: number) {
		if (!onSelectPair) return;
		onSelectPair(focusedPairIndex === i ? null : i);
	}

	function handleBgClick() {
		onSelectPair?.(null);
	}
</script>

<div class="relative w-full" style="aspect-ratio: {width} / {height};">
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<svg
		{width}
		{height}
		viewBox="0 0 {width} {height}"
		class="w-full h-full"
		role="img"
		aria-label="Contradiction constellation map — click an arc to focus, click background to deselect"
		onmousemove={onMove}
		onmouseleave={() => { hoverNode = null; hoverArc = null; }}
		onclick={handleBgClick}
	>
		<defs>
			<!-- Radial glass background -->
			<radialGradient id="bgGrad" cx="50%" cy="50%" r="65%">
				<stop offset="0%" stop-color="#10102a" stop-opacity="0.9" />
				<stop offset="60%" stop-color="#0a0a1a" stop-opacity="0.7" />
				<stop offset="100%" stop-color="#050510" stop-opacity="0.4" />
			</radialGradient>

			<!-- Soft arc gradients per severity, for a glow feel -->
			<linearGradient id="arcGradRed" x1="0" y1="0" x2="1" y2="0">
				<stop offset="0%" stop-color="#ef4444" stop-opacity="0.1" />
				<stop offset="50%" stop-color="#ef4444" stop-opacity="1" />
				<stop offset="100%" stop-color="#ef4444" stop-opacity="0.1" />
			</linearGradient>
			<linearGradient id="arcGradAmber" x1="0" y1="0" x2="1" y2="0">
				<stop offset="0%" stop-color="#f59e0b" stop-opacity="0.1" />
				<stop offset="50%" stop-color="#f59e0b" stop-opacity="1" />
				<stop offset="100%" stop-color="#f59e0b" stop-opacity="0.1" />
			</linearGradient>
			<linearGradient id="arcGradYellow" x1="0" y1="0" x2="1" y2="0">
				<stop offset="0%" stop-color="#fde047" stop-opacity="0.1" />
				<stop offset="50%" stop-color="#fde047" stop-opacity="1" />
				<stop offset="100%" stop-color="#fde047" stop-opacity="0.1" />
			</linearGradient>

			<!-- Node glow filter -->
			<filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
				<feGaussianBlur stdDeviation="2.5" result="blur" />
				<feMerge>
					<feMergeNode in="blur" />
					<feMergeNode in="SourceGraphic" />
				</feMerge>
			</filter>

			<filter id="arcGlow" x="-20%" y="-20%" width="140%" height="140%">
				<feGaussianBlur stdDeviation="3" result="blur" />
				<feMerge>
					<feMergeNode in="blur" />
					<feMergeNode in="SourceGraphic" />
				</feMerge>
			</filter>
		</defs>

		<!-- Cosmic background -->
		<rect x="0" y="0" {width} {height} fill="url(#bgGrad)" rx="16" />

		<!-- Subtle reference ring so the layout reads as a constellation -->
		<circle
			cx={geom.cx}
			cy={geom.cy}
			r={geom.R}
			fill="none"
			stroke="#6366f1"
			stroke-opacity="0.06"
			stroke-dasharray="2 6"
		/>
		<circle cx={geom.cx} cy={geom.cy} r="3" fill="#6366f1" opacity="0.4" />

		<!-- Arcs (cosmic bridges) — rendered before nodes so nodes sit on top -->
		{#each layout.arcs as arc (arc.pairIndex)}
			{@const op = pairOpacity(arc.pairIndex, focusedPairIndex)}
			{@const isFocused = focusedPairIndex === arc.pairIndex}
			<!-- Outer halo -->
			<path
				d={arc.path}
				fill="none"
				stroke={arc.color}
				stroke-width={arc.thickness * 3}
				stroke-opacity={0.08 * op}
				stroke-linecap="round"
				filter="url(#arcGlow)"
				pointer-events="none"
			/>
			<!-- Primary arc -->
			<path
				d={arc.path}
				fill="none"
				stroke={arc.color}
				stroke-width={arc.thickness * (isFocused ? 1.6 : 1)}
				stroke-opacity={(isFocused ? 1 : 0.72) * op}
				stroke-linecap="round"
				class="cursor-pointer transition-all duration-200"
				onclick={(e) => { e.stopPropagation(); handleArcClick(arc.pairIndex); }}
				onmouseenter={() => (hoverArc = arc)}
				onmouseleave={() => (hoverArc = null)}
				aria-label="contradiction {arc.pairIndex + 1}: {arc.topic}"
				role="button"
				tabindex="0"
				onkeydown={(e) => { if (e.key === 'Enter') handleArcClick(arc.pairIndex); }}
			/>
			<!-- Particle: a small dashed overlay that drifts along the arc to show tension flow -->
			<path
				d={arc.path}
				fill="none"
				stroke={arc.color}
				stroke-width={Math.max(1, arc.thickness * 0.6)}
				stroke-opacity={0.85 * op}
				stroke-linecap="round"
				stroke-dasharray="2 14"
				class="arc-particle"
				style="animation-duration: {4 + (arc.pairIndex % 5)}s"
				pointer-events="none"
			/>
		{/each}

		<!-- Nodes -->
		{#each layout.nodes as node, i (node.memoryId + '-' + node.side + '-' + i)}
			{@const op = pairOpacity(node.pairIndex, focusedPairIndex)}
			{@const isFocused = focusedPairIndex === node.pairIndex}
			{@const r = nodeRadius(node.trust)}
			{@const fill = nodeColor(node.type)}
			<!-- Outer glow -->
			<circle
				cx={node.x}
				cy={node.y}
				r={r * 2.2}
				fill={fill}
				opacity={0.12 * op}
				filter="url(#nodeGlow)"
				pointer-events="none"
			/>
			<!-- Core -->
			<circle
				cx={node.x}
				cy={node.y}
				r={r}
				fill={fill}
				opacity={op}
				stroke="#ffffff"
				stroke-opacity={isFocused ? 0.85 : 0.25}
				stroke-width={isFocused ? 2 : 1}
				class="cursor-pointer transition-all duration-200"
				onmouseenter={() => (hoverNode = node)}
				onmouseleave={() => (hoverNode = null)}
				onclick={(e) => { e.stopPropagation(); handleArcClick(node.pairIndex); }}
				role="button"
				tabindex="0"
				aria-label="memory {truncate(node.preview, 40)}"
				onkeydown={(e) => { if (e.key === 'Enter') handleArcClick(node.pairIndex); }}
			/>
			<!-- Label (truncated) — only shown for focused pair to avoid clutter -->
			{#if isFocused}
				<text
					x={node.x}
					y={node.y - r - 8}
					fill="#e0e0ff"
					font-size="10"
					font-family="var(--font-mono, monospace)"
					text-anchor="middle"
					pointer-events="none"
				>{truncate(node.preview, 40)}</text>
			{/if}
		{/each}

		<!-- Legend top-left -->
		<g transform="translate(16, 16)" pointer-events="none">
			<rect x="0" y="0" width="170" height="66" rx="8"
				fill="#0a0a1a" fill-opacity="0.6" stroke="#6366f1" stroke-opacity="0.12" />
			<text x="10" y="16" fill="#7a7aaa" font-size="10" font-family="var(--font-mono, monospace)">SEVERITY</text>
			<circle cx="16" cy="30" r="4" fill="#ef4444" />
			<text x="26" y="33" fill="#e0e0ff" font-size="10" font-family="var(--font-mono, monospace)">strong (&gt;0.7)</text>
			<circle cx="16" cy="44" r="4" fill="#f59e0b" />
			<text x="26" y="47" fill="#e0e0ff" font-size="10" font-family="var(--font-mono, monospace)">moderate (0.5-0.7)</text>
			<circle cx="16" cy="58" r="4" fill="#fde047" />
			<text x="26" y="61" fill="#e0e0ff" font-size="10" font-family="var(--font-mono, monospace)">mild (0.3-0.5)</text>
		</g>
	</svg>

	<!-- Hover tooltip (absolute, HTML for readability) -->
	{#if hoverNode}
		<div
			class="pointer-events-none absolute z-10 glass-panel rounded-lg px-3 py-2 text-xs max-w-xs shadow-xl"
			style="left: {Math.max(0, Math.min(mouseX + 12, width - 240))}px; top: {Math.max(0, Math.min(mouseY - 8, height - 120))}px;"
		>
			<div class="flex items-center gap-2 mb-1">
				<div class="w-2 h-2 rounded-full" style="background: {nodeColor(hoverNode.type)}"></div>
				<span class="text-bright font-semibold">{hoverNode.type ?? 'memory'}</span>
				<span class="text-muted ml-auto">trust {(hoverNode.trust * 100).toFixed(0)}%</span>
			</div>
			<div class="text-text mb-1">{hoverNode.preview}</div>
			{#if hoverNode.created}
				<div class="text-muted text-[10px]">created {hoverNode.created}</div>
			{/if}
			{#if hoverNode.tags && hoverNode.tags.length > 0}
				<div class="text-muted text-[10px] mt-1">
					{hoverNode.tags.slice(0, 4).join(' · ')}
				</div>
			{/if}
		</div>
	{:else if hoverArc}
		<div
			class="pointer-events-none absolute z-10 glass-panel rounded-lg px-3 py-2 text-xs max-w-xs shadow-xl"
			style="left: {Math.max(0, Math.min(mouseX + 12, width - 240))}px; top: {Math.max(0, Math.min(mouseY - 8, height - 120))}px;"
		>
			<div class="flex items-center gap-2 mb-1">
				<div class="w-2 h-2 rounded-full" style="background: {hoverArc.color}"></div>
				<span class="text-bright font-semibold">{hoverArc.severity} conflict</span>
			</div>
			<div class="text-dim">topic: <span class="text-text">{hoverArc.topic}</span></div>
			<div class="text-muted text-[10px] mt-1">
				similarity {(hoverArc.similarity * 100).toFixed(0)}% · {hoverArc.dateDiff}d apart
			</div>
		</div>
	{/if}
</div>

<style>
	@keyframes arc-drift {
		0% { stroke-dashoffset: 0; }
		100% { stroke-dashoffset: -32; }
	}
	:global(.arc-particle) {
		animation-name: arc-drift;
		animation-timing-function: linear;
		animation-iteration-count: infinite;
	}
</style>
