<script lang="ts">
	import {
		roleMetaFor,
		trustColor,
		trustPercent,
		nodeTypeColor,
		formatDate,
		shortenId,
		type EvidenceRole,
	} from './reasoning-helpers';

	interface Props {
		id: string;
		trust: number; // 0-1
		date: string; // ISO
		role: EvidenceRole;
		preview: string;
		nodeType?: string;
		index?: number; // for staggered animation
	}

	let { id, trust, date, role, preview, nodeType, index = 0 }: Props = $props();

	// Clamp for display — delegated to pure helper for testability.
	const trustPct = $derived(trustPercent(trust));
	const meta = $derived(roleMetaFor(role));
	const shortId = $derived(shortenId(id));
	const typeColor = $derived(nodeTypeColor(nodeType));
</script>

<div
	class="evidence-card glass rounded-xl p-4 space-y-3 transition relative"
	class:contradicting={role === 'contradicting'}
	class:primary={role === 'primary'}
	class:superseded={role === 'superseded'}
	style="animation-delay: {index * 80}ms;"
	data-evidence-id={id}
>
	<!-- Role banner + id -->
	<div class="flex items-center justify-between text-[10px] uppercase tracking-wider">
		<div class="flex items-center gap-2">
			<span class="role-pill px-2 py-0.5 rounded text-[10px]">
				<span class="mr-1">{meta.icon}</span>{meta.label}
			</span>
			{#if nodeType}
				<span class="px-1.5 py-0.5 rounded bg-white/[0.04]" style="color: {typeColor}">
					{nodeType}
				</span>
			{/if}
		</div>
		<span class="text-muted font-mono text-[10px]" title={id}>#{shortId}</span>
	</div>

	<!-- Preview -->
	<p class="text-sm text-text leading-relaxed line-clamp-4">{preview}</p>

	<!-- Trust bar -->
	<div class="space-y-1.5">
		<div class="flex items-center justify-between text-[10px]">
			<span class="text-dim uppercase tracking-wider">Trust</span>
			<span class="font-mono" style="color: {trustColor(trust)}">{trustPct.toFixed(0)}%</span>
		</div>
		<div class="h-1.5 bg-deep rounded-full overflow-hidden">
			<div
				class="h-full rounded-full transition-all duration-700 trust-fill"
				style="width: {trustPct}%; background: {trustColor(trust)}; box-shadow: 0 0 8px {trustColor(trust)}80;"
			></div>
		</div>
	</div>

	<!-- Date -->
	<div class="flex items-center justify-between text-[10px] text-muted pt-1">
		<span>{formatDate(date)}</span>
		<span class="font-mono opacity-60">FSRS · reps × retention</span>
	</div>
</div>

<style>
	.evidence-card {
		animation: card-rise 600ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards;
	}

	.evidence-card.primary {
		border-color: rgba(99, 102, 241, 0.35) !important;
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.04),
			0 0 32px rgba(99, 102, 241, 0.18),
			0 8px 32px rgba(0, 0, 0, 0.4);
	}

	.evidence-card.contradicting {
		border-color: rgba(239, 68, 68, 0.45) !important;
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.03),
			0 0 28px rgba(239, 68, 68, 0.2),
			0 8px 32px rgba(0, 0, 0, 0.4);
	}

	.evidence-card.superseded {
		opacity: 0.55;
	}

	.evidence-card.superseded:hover {
		opacity: 0.9;
	}

	.role-pill {
		background: rgba(99, 102, 241, 0.12);
		color: #c7cbff;
		border: 1px solid rgba(99, 102, 241, 0.25);
	}

	.evidence-card.contradicting .role-pill {
		background: rgba(239, 68, 68, 0.14);
		color: #fecaca;
		border-color: rgba(239, 68, 68, 0.4);
	}

	.evidence-card.primary .role-pill {
		background: rgba(99, 102, 241, 0.22);
		color: #a5b4ff;
		border-color: rgba(99, 102, 241, 0.5);
	}

	.trust-fill {
		animation: trust-sweep 1000ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards;
	}

	@keyframes card-rise {
		0% {
			opacity: 0;
			transform: translateY(12px) scale(0.98);
		}
		100% {
			opacity: 1;
			transform: translateY(0) scale(1);
		}
	}

	@keyframes trust-sweep {
		0% {
			width: 0% !important;
			opacity: 0.4;
		}
		100% {
			opacity: 1;
		}
	}

	.line-clamp-4 {
		display: -webkit-box;
		-webkit-line-clamp: 4;
		line-clamp: 4;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
</style>
