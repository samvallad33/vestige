<script lang="ts">
	interface StageResult {
		label?: string;
		value?: string | number;
	}

	interface Props {
		intent?: string;
		memoriesAnalyzed?: number;
		evidenceCount?: number;
		contradictionCount?: number;
		supersededCount?: number;
		running?: boolean; // when true, run the sequential light-up animation
		// Optional per-stage hints (one-liners) — if provided, overrides defaults
		stageHints?: Partial<Record<StageKey, string>>;
	}

	type StageKey =
		| 'broad'
		| 'spreading'
		| 'fsrs'
		| 'intent'
		| 'supersession'
		| 'contradiction'
		| 'relation'
		| 'template';

	let {
		intent = 'Synthesis',
		memoriesAnalyzed = 0,
		evidenceCount = 0,
		contradictionCount = 0,
		supersededCount = 0,
		running = false,
		stageHints = {}
	}: Props = $props();

	const STAGES: { key: StageKey; icon: string; label: string; base: string }[] = [
		{
			key: 'broad',
			icon: '◎',
			label: 'Broad Retrieval',
			base: 'Hybrid BM25 + semantic (3x overfetch) then cross-encoder rerank'
		},
		{
			key: 'spreading',
			icon: '⟿',
			label: 'Spreading Activation',
			base: 'Collins & Loftus — expand via graph edges to surface what search missed'
		},
		{
			key: 'fsrs',
			icon: '▲',
			label: 'FSRS Trust Scoring',
			base: 'retention × stability × reps ÷ lapses — which memories have earned trust'
		},
		{
			key: 'intent',
			icon: '◆',
			label: 'Intent Classification',
			base: 'FactCheck / Timeline / RootCause / Comparison / Synthesis'
		},
		{
			key: 'supersession',
			icon: '↗',
			label: 'Temporal Supersession',
			base: 'Newer high-trust memories replace older ones on the same fact'
		},
		{
			key: 'contradiction',
			icon: '⚡',
			label: 'Contradiction Analysis',
			base: 'Only flag conflicts between memories where BOTH have trust > 0.3'
		},
		{
			key: 'relation',
			icon: '⬡',
			label: 'Relation Assessment',
			base: 'Per pair: Supports / Contradicts / Supersedes / Irrelevant'
		},
		{
			key: 'template',
			icon: '❖',
			label: 'Template Reasoning',
			base: 'Build the natural-language reasoning chain you validate'
		}
	];

	// Dynamic one-liners reflecting the actual response — fall back to base
	const computed: Partial<Record<StageKey, string>> = $derived({
		broad: memoriesAnalyzed ? `Analyzed ${memoriesAnalyzed} memories · ${evidenceCount} survived ranking` : undefined,
		intent: intent ? `Classified as ${intent}` : undefined,
		supersession: supersededCount
			? `${supersededCount} outdated memor${supersededCount === 1 ? 'y' : 'ies'} superseded`
			: undefined,
		contradiction: contradictionCount
			? `${contradictionCount} real conflict${contradictionCount === 1 ? '' : 's'} between trusted memories`
			: 'No conflicts between trusted memories'
	});

	function hintFor(key: StageKey, base: string): string {
		return stageHints[key] ?? computed[key] ?? base;
	}
</script>

<div class="reasoning-chain space-y-2" class:running>
	{#each STAGES as stage, i (stage.key)}
		<div
			class="stage glass-subtle rounded-xl p-3 flex items-start gap-3 relative"
			style="animation-delay: {i * 140}ms;"
		>
			<!-- Connector line down to next stage -->
			{#if i < STAGES.length - 1}
				<div class="connector" style="animation-delay: {i * 140 + 120}ms;"></div>
			{/if}

			<!-- Stage index + icon -->
			<div class="stage-orb flex-shrink-0" style="animation-delay: {i * 140}ms;">
				<span class="text-xs text-synapse-glow">{stage.icon}</span>
			</div>

			<div class="flex-1 min-w-0">
				<div class="flex items-center gap-2 mb-0.5">
					<span class="text-[10px] font-mono text-muted">0{i + 1}</span>
					<span class="text-sm text-bright font-medium">{stage.label}</span>
				</div>
				<p class="text-xs text-dim leading-snug">{hintFor(stage.key, stage.base)}</p>
			</div>

			<span class="stage-pulse" aria-hidden="true"></span>
		</div>
	{/each}
</div>

<style>
	.stage {
		animation: stage-light 700ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards;
		position: relative;
		border-color: rgba(99, 102, 241, 0.08);
	}

	.stage-orb {
		width: 28px;
		height: 28px;
		border-radius: 50%;
		background: radial-gradient(
			circle at 30% 30%,
			rgba(129, 140, 248, 0.25),
			rgba(99, 102, 241, 0.05)
		);
		border: 1px solid rgba(99, 102, 241, 0.3);
		display: flex;
		align-items: center;
		justify-content: center;
		position: relative;
		animation: orb-glow 700ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards;
	}

	.stage-pulse {
		position: absolute;
		inset: 0;
		border-radius: 12px;
		border: 1px solid rgba(129, 140, 248, 0);
		pointer-events: none;
		animation: pulse-ring 700ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards;
	}

	.connector {
		position: absolute;
		left: 22px;
		top: 100%;
		width: 1px;
		height: 8px;
		background: linear-gradient(180deg, rgba(129, 140, 248, 0.5), rgba(168, 85, 247, 0.15));
		animation: connector-draw 500ms ease-out backwards;
	}

	.running .stage {
		animation: stage-light 700ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards,
			stage-flicker 2400ms ease-in-out infinite;
	}

	@keyframes stage-light {
		0% {
			opacity: 0;
			transform: translateX(-8px);
			border-color: rgba(99, 102, 241, 0);
		}
		60% {
			opacity: 1;
			border-color: rgba(129, 140, 248, 0.35);
		}
		100% {
			opacity: 1;
			transform: translateX(0);
			border-color: rgba(99, 102, 241, 0.08);
		}
	}

	@keyframes orb-glow {
		0% {
			transform: scale(0.6);
			opacity: 0;
			box-shadow: 0 0 0 rgba(129, 140, 248, 0);
		}
		60% {
			transform: scale(1.15);
			opacity: 1;
			box-shadow: 0 0 24px rgba(129, 140, 248, 0.8);
		}
		100% {
			transform: scale(1);
			box-shadow: 0 0 10px rgba(129, 140, 248, 0.35);
		}
	}

	@keyframes pulse-ring {
		0% {
			transform: scale(0.96);
			opacity: 0;
			border-color: rgba(129, 140, 248, 0);
		}
		70% {
			transform: scale(1);
			opacity: 1;
			border-color: rgba(129, 140, 248, 0.4);
			box-shadow: 0 0 20px rgba(129, 140, 248, 0.25);
		}
		100% {
			transform: scale(1.01);
			opacity: 0;
			border-color: rgba(129, 140, 248, 0);
			box-shadow: 0 0 0 rgba(129, 140, 248, 0);
		}
	}

	@keyframes connector-draw {
		0% {
			transform: scaleY(0);
			transform-origin: top;
			opacity: 0;
		}
		100% {
			transform: scaleY(1);
			transform-origin: top;
			opacity: 1;
		}
	}

	@keyframes stage-flicker {
		0%,
		100% {
			border-color: rgba(99, 102, 241, 0.08);
		}
		50% {
			border-color: rgba(129, 140, 248, 0.25);
		}
	}
</style>
