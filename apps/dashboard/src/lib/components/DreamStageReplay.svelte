<!--
  DreamStageReplay — Visual playback of a single dream-consolidation stage.

  5 stages (per MemoryDreamer):
    1. Replay        — floating memory cards arrange themselves into a grid
    2. Cross-reference — edges connect cards (SVG lines)
    3. Strengthen    — cards pulse, brighten, gain glow
    4. Prune         — low-retention cards fade & dissolve
    5. Transfer      — cards migrate from episodic (left) → semantic (right)

  Pure CSS transforms + SVG + animations. No Three.js.
-->
<script lang="ts">
	import type { DreamResult } from '$types';
	import { clampStage } from './dream-helpers';

	interface Props {
		stage: number; // 1..5
		dreamResult: DreamResult | null;
	}

	let { stage, dreamResult }: Props = $props();

	const STAGES = [
		{
			num: 1,
			name: 'Replay',
			color: '#818cf8',
			desc: 'Hippocampal replay: tagged memories surface for consolidation.'
		},
		{
			num: 2,
			name: 'Cross-reference',
			color: '#a855f7',
			desc: 'Semantic proximity check — new edges discovered across memories.'
		},
		{
			num: 3,
			name: 'Strengthen',
			color: '#c084fc',
			desc: 'Co-activated memories strengthen; FSRS stability grows.'
		},
		{
			num: 4,
			name: 'Prune',
			color: '#ef4444',
			desc: 'Low-retention redundant memories compressed or released.'
		},
		{
			num: 5,
			name: 'Transfer',
			color: '#10b981',
			desc: 'Episodic → semantic consolidation (hippocampus → cortex).'
		}
	];

	// Lock stage to valid range (handles NaN / negatives / >5)
	let stageIdx = $derived(clampStage(stage));
	let current = $derived(STAGES[stageIdx - 1]);

	// Derive card count from dream result (clamp 6..12 for visual density)
	let cardCount = $derived.by(() => {
		if (!dreamResult) return 8;
		const n = dreamResult.memoriesReplayed ?? 8;
		return Math.max(6, Math.min(12, n));
	});

	// Connection count from dream result
	let connectionCount = $derived.by(() => {
		if (!dreamResult) return 5;
		const n = dreamResult.stats?.newConnectionsFound ?? 5;
		return Math.max(3, Math.min(cardCount, n));
	});

	let strengthenedCount = $derived.by(() => {
		if (!dreamResult) return Math.ceil(cardCount * 0.5);
		const n = dreamResult.stats?.memoriesStrengthened ?? Math.ceil(cardCount * 0.5);
		return Math.max(1, Math.min(cardCount, n));
	});

	let prunedCount = $derived.by(() => {
		if (!dreamResult) return Math.ceil(cardCount * 0.25);
		const n = dreamResult.stats?.memoriesCompressed ?? Math.ceil(cardCount * 0.25);
		return Math.max(1, Math.min(Math.floor(cardCount / 2), n));
	});

	// Deterministic pseudo-random for card positions so stage changes don't re-randomize
	function seed(i: number, salt = 0): number {
		const x = Math.sin((i + 1) * 9301 + 49297 + salt * 233) * 233280;
		return x - Math.floor(x);
	}

	// Layout: cards on a circle-ish grid in the center.
	// In stage 5, we'll override X based on transfer side.
	interface CardPos {
		id: number;
		x: number; // 0..100 percent
		y: number; // 0..100 percent
		pruned: boolean;
		strengthened: boolean;
		transferIsSemantic: boolean;
	}

	let cards = $derived.by<CardPos[]>(() => {
		const arr: CardPos[] = [];
		const cols = Math.ceil(Math.sqrt(cardCount));
		const rows = Math.ceil(cardCount / cols);
		for (let i = 0; i < cardCount; i++) {
			const col = i % cols;
			const row = Math.floor(i / cols);
			const baseX = 20 + (col / Math.max(1, cols - 1)) * 60;
			const baseY = 20 + (row / Math.max(1, rows - 1)) * 60;
			// jitter
			const jx = (seed(i, 1) - 0.5) * 8;
			const jy = (seed(i, 2) - 0.5) * 8;
			arr.push({
				id: i,
				x: baseX + jx,
				y: baseY + jy,
				pruned: i < prunedCount,
				strengthened: i < strengthenedCount,
				transferIsSemantic: i % 2 === 0
			});
		}
		return arr;
	});

	// Edges for stage 2 (and persist for 3+). Pick random pairs from available cards.
	interface Edge {
		a: number;
		b: number;
	}

	let edges = $derived.by<Edge[]>(() => {
		const e: Edge[] = [];
		const n = cards.length;
		for (let k = 0; k < connectionCount; k++) {
			const a = Math.floor(seed(k, 7) * n);
			let b = Math.floor(seed(k, 11) * n);
			if (b === a) b = (a + 1) % n;
			e.push({ a, b });
		}
		return e;
	});

	// Card displayed position depending on stage
	function cardX(card: CardPos): number {
		if (stageIdx === 5) {
			// Migrate: episodic (left 15..35%) → semantic (right 65..85%)
			const target = card.transferIsSemantic ? 75 : 25;
			return target + (seed(card.id, 5) - 0.5) * 12;
		}
		return card.x;
	}

	function cardY(card: CardPos): number {
		if (stageIdx === 5) {
			return 25 + seed(card.id, 6) * 50;
		}
		return card.y;
	}

	function cardOpacity(card: CardPos): number {
		if (stageIdx === 4 && card.pruned) return 0;
		if (stageIdx === 5 && card.pruned) return 0.15;
		return 1;
	}

	function cardScale(card: CardPos): number {
		if (stageIdx === 3 && card.strengthened) return 1.18;
		if (stageIdx === 4 && card.pruned) return 0.6;
		return 1;
	}
</script>

<div class="replay-stage glass-panel rounded-2xl overflow-hidden relative">
	<!-- Stage header -->
	<header class="flex items-center justify-between px-5 py-3 border-b border-white/5 relative z-10">
		<div class="flex items-center gap-3">
			<div
				class="stage-badge w-9 h-9 rounded-full flex items-center justify-center font-mono font-bold text-sm"
				style="
					background: color-mix(in srgb, {current.color} 20%, transparent);
					color: {current.color};
					border: 1.5px solid {current.color};
					box-shadow: 0 0 16px color-mix(in srgb, {current.color} 40%, transparent);
				"
			>
				{current.num}
			</div>
			<div>
				<div class="text-sm font-semibold text-bright tracking-wide">{current.name}</div>
				<div class="text-[11px] text-dim leading-snug max-w-md">{current.desc}</div>
			</div>
		</div>
		<div class="text-[10px] text-dim uppercase tracking-[0.15em] hidden sm:block">
			Stage {current.num} / 5
		</div>
	</header>

	<!-- Stage canvas -->
	<div
		class="stage-canvas"
		style="--stage-color: {current.color}"
		aria-label="Dream stage {current.num} — {current.name}"
	>
		<!-- Left/right labels for transfer stage -->
		{#if stageIdx === 5}
			<div class="transfer-label episodic">
				<span class="label-tag">Episodic</span>
				<span class="label-sub">hippocampus</span>
			</div>
			<div class="transfer-label semantic">
				<span class="label-tag">Semantic</span>
				<span class="label-sub">cortex</span>
			</div>
			<div class="divider-line"></div>
		{/if}

		<!-- SVG edges (visible from stage 2 onward) -->
		<svg
			class="edges-layer"
			viewBox="0 0 100 100"
			preserveAspectRatio="none"
			aria-hidden="true"
		>
			{#each edges as edge, i (edge.a + '-' + edge.b + '-' + i)}
				{@const a = cards[edge.a]}
				{@const b = cards[edge.b]}
				{#if a && b}
					{@const x1 = cardX(a)}
					{@const y1 = cardY(a)}
					{@const x2 = cardX(b)}
					{@const y2 = cardY(b)}
					<line
						x1={x1}
						y1={y1}
						x2={x2}
						y2={y2}
						stroke={current.color}
						stroke-width={stageIdx === 2 ? 0.25 : stageIdx === 3 ? 0.35 : 0.2}
						stroke-opacity={stageIdx < 2 ? 0 : stageIdx === 4 ? 0.25 : stageIdx === 5 ? 0.15 : 0.6}
						stroke-dasharray={stageIdx === 2 ? '1.2 0.8' : 'none'}
						class="edge-line"
						style="--edge-delay: {i * 80}ms"
					/>
				{/if}
			{/each}
		</svg>

		<!-- Memory cards -->
		{#each cards as card (card.id)}
			<div
				class="memory-card"
				class:is-pulsing={stageIdx === 3 && card.strengthened}
				class:is-pruning={stageIdx === 4 && card.pruned}
				class:is-transferring={stageIdx === 5}
				class:semantic-side={stageIdx === 5 && card.transferIsSemantic}
				style="
					left: {cardX(card)}%;
					top: {cardY(card)}%;
					opacity: {cardOpacity(card)};
					--card-scale: {cardScale(card)};
					--card-delay: {card.id * 40}ms;
					--card-hue: {seed(card.id, 3) * 60 - 30}deg;
				"
			>
				<div class="card-inner">
					<div class="card-dot"></div>
					<div class="card-bar"></div>
					<div class="card-bar short"></div>
				</div>
			</div>
		{/each}

		<!-- Ambient pulse for stage 1 (replay) -->
		{#if stageIdx === 1}
			<div class="replay-pulse" aria-hidden="true"></div>
		{/if}
	</div>

	<!-- Stage footer with stats -->
	<footer class="flex flex-wrap gap-x-6 gap-y-1 px-5 py-3 border-t border-white/5 text-[11px] text-dim">
		{#if stageIdx === 1}
			<span>Replaying <b class="text-bright tabular-nums">{dreamResult?.memoriesReplayed ?? cardCount}</b> memories</span>
		{:else if stageIdx === 2}
			<span>New connections found: <b class="text-bright tabular-nums">{dreamResult?.stats?.newConnectionsFound ?? connectionCount}</b></span>
		{:else if stageIdx === 3}
			<span>Strengthened: <b class="text-bright tabular-nums">{dreamResult?.stats?.memoriesStrengthened ?? strengthenedCount}</b></span>
		{:else if stageIdx === 4}
			<span>Compressed: <b class="text-bright tabular-nums">{dreamResult?.stats?.memoriesCompressed ?? prunedCount}</b></span>
		{:else if stageIdx === 5}
			<span>Connections persisted: <b class="text-bright tabular-nums">{dreamResult?.connectionsPersisted ?? 0}</b></span>
			<span>Insights: <b class="text-bright tabular-nums">{dreamResult?.stats?.insightsGenerated ?? 0}</b></span>
		{/if}
	</footer>
</div>

<style>
	.replay-stage {
		border: 1px solid rgba(168, 85, 247, 0.18);
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.03),
			0 8px 36px -8px rgba(0, 0, 0, 0.55),
			0 0 48px -16px rgba(168, 85, 247, 0.25);
	}

	.stage-canvas {
		position: relative;
		height: 360px;
		overflow: hidden;
		background:
			radial-gradient(at 50% 50%, color-mix(in srgb, var(--stage-color) 10%, transparent), transparent 60%),
			radial-gradient(at 20% 80%, rgba(99, 102, 241, 0.08), transparent 50%),
			#08081a;
	}

	.edges-layer {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
	}

	.edge-line {
		transition: stroke-opacity 520ms ease, stroke-width 520ms ease,
			x1 600ms cubic-bezier(0.34, 1.56, 0.64, 1),
			y1 600ms cubic-bezier(0.34, 1.56, 0.64, 1),
			x2 600ms cubic-bezier(0.34, 1.56, 0.64, 1),
			y2 600ms cubic-bezier(0.34, 1.56, 0.64, 1);
	}

	.memory-card {
		position: absolute;
		width: 44px;
		height: 32px;
		transform: translate(-50%, -50%) scale(var(--card-scale, 1));
		transition:
			left 600ms cubic-bezier(0.34, 1.56, 0.64, 1),
			top 600ms cubic-bezier(0.34, 1.56, 0.64, 1),
			transform 500ms cubic-bezier(0.34, 1.56, 0.64, 1),
			opacity 500ms ease;
		transition-delay: var(--card-delay, 0ms);
		animation: card-float 6s ease-in-out infinite;
		animation-delay: var(--card-delay, 0ms);
		will-change: transform;
	}

	.card-inner {
		width: 100%;
		height: 100%;
		border-radius: 6px;
		background: linear-gradient(
			135deg,
			color-mix(in srgb, var(--stage-color) 30%, transparent),
			color-mix(in srgb, var(--stage-color) 10%, transparent)
		);
		border: 1px solid color-mix(in srgb, var(--stage-color) 50%, transparent);
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.08),
			0 0 8px color-mix(in srgb, var(--stage-color) 30%, transparent);
		filter: hue-rotate(var(--card-hue, 0deg));
		padding: 5px 6px;
		display: flex;
		flex-direction: column;
		justify-content: center;
		gap: 3px;
		position: relative;
		overflow: hidden;
	}

	.card-dot {
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: color-mix(in srgb, var(--stage-color) 90%, white);
		box-shadow: 0 0 6px var(--stage-color);
		position: absolute;
		top: 4px;
		right: 4px;
	}

	.card-bar {
		height: 3px;
		border-radius: 1.5px;
		background: color-mix(in srgb, var(--stage-color) 70%, transparent);
		width: 80%;
	}

	.card-bar.short {
		width: 50%;
		opacity: 0.6;
	}

	.memory-card.is-pulsing {
		animation: card-float 6s ease-in-out infinite, card-pulse 1.4s ease-in-out infinite;
	}

	.memory-card.is-pulsing .card-inner {
		border-color: var(--color-dream-glow);
		background: linear-gradient(
			135deg,
			color-mix(in srgb, var(--color-dream-glow) 40%, transparent),
			color-mix(in srgb, var(--color-dream) 25%, transparent)
		);
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.12),
			0 0 22px color-mix(in srgb, var(--color-dream-glow) 60%, transparent),
			0 0 44px color-mix(in srgb, var(--color-dream) 35%, transparent);
	}

	.memory-card.is-pruning .card-inner {
		animation: dissolve 1.2s ease-out forwards;
	}

	.memory-card.is-transferring .card-inner {
		border-color: #10b981;
		background: linear-gradient(
			135deg,
			rgba(16, 185, 129, 0.35),
			rgba(16, 185, 129, 0.12)
		);
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.08),
			0 0 14px rgba(16, 185, 129, 0.5);
	}

	.memory-card.is-transferring.semantic-side .card-inner {
		border-color: #c084fc;
		background: linear-gradient(
			135deg,
			rgba(192, 132, 252, 0.35),
			rgba(168, 85, 247, 0.15)
		);
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.08),
			0 0 14px rgba(192, 132, 252, 0.5);
	}

	.replay-pulse {
		position: absolute;
		top: 50%;
		left: 50%;
		width: 60%;
		aspect-ratio: 1 / 1;
		transform: translate(-50%, -50%);
		border-radius: 50%;
		background: radial-gradient(circle, color-mix(in srgb, var(--stage-color) 25%, transparent), transparent 60%);
		filter: blur(30px);
		animation: pulse-in 3s ease-in-out infinite;
		pointer-events: none;
	}

	.transfer-label {
		position: absolute;
		top: 12px;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 2px;
		z-index: 2;
	}

	.transfer-label.episodic {
		left: 6%;
	}

	.transfer-label.semantic {
		right: 6%;
	}

	.label-tag {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		padding: 2px 8px;
		border-radius: 999px;
		border: 1px solid rgba(255, 255, 255, 0.15);
		background: rgba(0, 0, 0, 0.35);
		color: #e0e0ff;
	}

	.transfer-label.episodic .label-tag {
		border-color: rgba(16, 185, 129, 0.5);
		color: #10b981;
	}

	.transfer-label.semantic .label-tag {
		border-color: rgba(192, 132, 252, 0.5);
		color: #c084fc;
	}

	.label-sub {
		font-size: 9px;
		color: var(--color-dim);
		letter-spacing: 0.1em;
	}

	.divider-line {
		position: absolute;
		top: 15%;
		bottom: 15%;
		left: 50%;
		width: 1px;
		background: linear-gradient(180deg, transparent, rgba(168, 85, 247, 0.35), transparent);
		transform: translateX(-0.5px);
	}

	@keyframes card-float {
		0%, 100% { translate: 0 0; }
		25% { translate: 2px -3px; }
		50% { translate: -2px 2px; }
		75% { translate: 3px 1px; }
	}

	@keyframes card-pulse {
		0%, 100% { filter: brightness(1) hue-rotate(var(--card-hue, 0deg)); }
		50% { filter: brightness(1.3) hue-rotate(var(--card-hue, 0deg)); }
	}

	@keyframes dissolve {
		0% { opacity: 1; transform: scale(1); filter: blur(0); }
		60% { opacity: 0.3; filter: blur(2px); }
		100% { opacity: 0; transform: scale(0.5); filter: blur(6px); }
	}

	@keyframes pulse-in {
		0%, 100% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
		50% { opacity: 0.7; transform: translate(-50%, -50%) scale(1.15); }
	}

	@media (prefers-reduced-motion: reduce) {
		.memory-card { animation: none; }
		.replay-pulse { animation: none; }
		.memory-card.is-pulsing { animation: none; }
	}
</style>
