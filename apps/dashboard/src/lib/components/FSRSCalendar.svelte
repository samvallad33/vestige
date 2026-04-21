<script lang="ts">
	import type { Memory } from '$types';
	import { NODE_TYPE_COLORS } from '$types';
	import {
		startOfDay,
		daysBetween,
		isoDate,
		gridStartForAnchor,
		avgRetention as avgRetentionHelper,
	} from './schedule-helpers';

	type Props = {
		memories: Memory[];
		anchor?: Date;
	};

	let { memories, anchor = new Date() }: Props = $props();

	// Build a 6-row x 7-col grid starting 2 weeks before today.
	// Rows: 2 past weeks + 4 future weeks = 6 weeks = 42 cells.
	// Align so the first row starts on the Sunday that is on or before
	// (today - 14 days). This keeps today visible in week 3.
	let today = $derived(startOfDay(anchor));
	let gridStart = $derived(gridStartForAnchor(anchor));

	type DayCell = {
		date: Date;
		key: string;
		isToday: boolean;
		inWindow: boolean; // within -14 to +28 days
		memories: Memory[];
		avgRetention: number;
	};

	// Bucket memories by their nextReviewAt day (YYYY-MM-DD).
	let buckets = $derived(
		(() => {
			const map = new Map<string, Memory[]>();
			for (const m of memories) {
				if (!m.nextReviewAt) continue;
				const d = new Date(m.nextReviewAt);
				if (Number.isNaN(d.getTime())) continue;
				const key = isoDate(startOfDay(d));
				const arr = map.get(key);
				if (arr) arr.push(m);
				else map.set(key, [m]);
			}
			return map;
		})()
	);

	let cells = $derived(
		(() => {
			const out: DayCell[] = [];
			for (let i = 0; i < 42; i++) {
				const d = new Date(gridStart);
				d.setDate(d.getDate() + i);
				const key = isoDate(d);
				const ms = buckets.get(key) ?? [];
				const delta = daysBetween(d, today);
				out.push({
					date: d,
					key,
					isToday: delta === 0,
					inWindow: delta >= -14 && delta <= 28,
					memories: ms,
					avgRetention: avgRetentionHelper(ms)
				});
			}
			return out;
		})()
	);

	// Urgency coloring — emerald / amber / decay-red / synapse / muted.
	function cellColor(cell: DayCell): { bg: string; border: string; text: string } {
		if (cell.memories.length === 0) {
			return { bg: 'rgba(255,255,255,0.02)', border: 'rgba(99,102,241,0.06)', text: '#4a4a7a' };
		}
		const delta = daysBetween(cell.date, today);
		if (delta < -1) {
			// Overdue (more than 1 day in the past) — decay-red
			return {
				bg: 'rgba(239,68,68,0.16)',
				border: 'rgba(239,68,68,0.45)',
				text: '#fca5a5'
			};
		}
		if (delta >= -1 && delta <= 0) {
			// Due today (or yesterday, just at the threshold) — amber
			return {
				bg: 'rgba(245,158,11,0.18)',
				border: 'rgba(245,158,11,0.5)',
				text: '#fcd34d'
			};
		}
		if (delta > 0 && delta <= 7) {
			// Due within 7 days — synapse blue
			return {
				bg: 'rgba(99,102,241,0.16)',
				border: 'rgba(99,102,241,0.45)',
				text: '#a5b4fc'
			};
		}
		// >7 days out — muted
		return {
			bg: 'rgba(168,85,247,0.08)',
			border: 'rgba(168,85,247,0.2)',
			text: '#c084fc'
		};
	}

	let selectedKey: string | null = $state(null);
	let selectedCell = $derived(cells.find((c) => c.key === selectedKey) ?? null);

	function toggle(key: string) {
		selectedKey = selectedKey === key ? null : key;
	}

	// Retention sparkline — one point per day in the window, average retention
	// of memories due that day. Width 100% x height 48px SVG.
	const SPARK_W = 600;
	const SPARK_H = 56;

	let sparkPoints = $derived(
		(() => {
			const pts: { x: number; y: number; r: number; count: number }[] = [];
			const n = cells.length;
			for (let i = 0; i < n; i++) {
				const c = cells[i];
				const x = (i / (n - 1)) * SPARK_W;
				// Invert Y: higher retention = higher on chart = smaller y
				const r = c.avgRetention;
				const y = SPARK_H - 6 - r * (SPARK_H - 12);
				pts.push({ x, y, r, count: c.memories.length });
			}
			return pts;
		})()
	);

	let sparkPath = $derived(
		(() => {
			const valid = sparkPoints.filter((p) => p.count > 0);
			if (valid.length === 0) return '';
			return valid.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
		})()
	);

	// Today's x-position on the sparkline, so the viewer can anchor the trend.
	let todayIndex = $derived(cells.findIndex((c) => c.isToday));
	let todayX = $derived(todayIndex >= 0 ? (todayIndex / (cells.length - 1)) * SPARK_W : -1);

	const DOW_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

	function shortDate(d: Date): string {
		return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
	}

	function fullDate(d: Date): string {
		return d.toLocaleDateString(undefined, {
			weekday: 'long',
			month: 'long',
			day: 'numeric',
			year: 'numeric'
		});
	}
</script>

<div class="space-y-4">
	<!-- Retention sparkline -->
	<div class="p-4 glass-subtle rounded-xl">
		<div class="flex items-center justify-between mb-2">
			<span class="text-xs text-dim font-medium">Avg retention of memories due — last 2 weeks → next 4</span>
			<div class="flex items-center gap-3 text-[10px] text-muted">
				<span class="flex items-center gap-1"><span class="w-2 h-0.5 bg-recall"></span>retention</span>
				<span class="flex items-center gap-1"><span class="w-px h-3 bg-synapse-glow"></span>today</span>
			</div>
		</div>
		<svg viewBox="0 0 {SPARK_W} {SPARK_H}" preserveAspectRatio="none" class="w-full h-12 block" aria-hidden="true">
			<!-- Baselines at 30% / 70% -->
			<line x1="0" x2={SPARK_W} y1={SPARK_H - 6 - 0.3 * (SPARK_H - 12)} y2={SPARK_H - 6 - 0.3 * (SPARK_H - 12)}
				stroke="rgba(239,68,68,0.18)" stroke-dasharray="2 4" stroke-width="1" />
			<line x1="0" x2={SPARK_W} y1={SPARK_H - 6 - 0.7 * (SPARK_H - 12)} y2={SPARK_H - 6 - 0.7 * (SPARK_H - 12)}
				stroke="rgba(16,185,129,0.18)" stroke-dasharray="2 4" stroke-width="1" />
			{#if todayX >= 0}
				<line x1={todayX} x2={todayX} y1="0" y2={SPARK_H}
					stroke="rgba(129,140,248,0.5)" stroke-width="1" />
			{/if}
			{#if sparkPath}
				<path d={sparkPath} fill="none" stroke="var(--color-recall)" stroke-width="1.5" stroke-linejoin="round" />
			{/if}
			{#each sparkPoints as p}
				{#if p.count > 0}
					<circle cx={p.x} cy={p.y} r="1.5" fill="var(--color-recall)" />
				{/if}
			{/each}
		</svg>
	</div>

	<!-- Day-of-week header -->
	<div class="grid grid-cols-7 gap-2 px-1">
		{#each DOW_LABELS as label}
			<div class="text-[10px] text-muted font-mono uppercase tracking-wider text-center">{label}</div>
		{/each}
	</div>

	<!-- Calendar grid -->
	<div class="grid grid-cols-7 gap-2">
		{#each cells as cell (cell.key)}
			{@const colors = cellColor(cell)}
			<button
				type="button"
				onclick={() => toggle(cell.key)}
				disabled={cell.memories.length === 0}
				class="relative aspect-square rounded-lg p-2 text-left transition-all duration-200
					{cell.inWindow ? 'opacity-100' : 'opacity-35'}
					{cell.memories.length > 0 ? 'hover:scale-[1.03] cursor-pointer' : 'cursor-default'}
					{cell.isToday ? 'ring-2 ring-synapse/60 shadow-[0_0_16px_rgba(99,102,241,0.3)]' : ''}
					{selectedKey === cell.key ? 'ring-2 ring-dream/60 shadow-[0_0_16px_rgba(168,85,247,0.3)]' : ''}"
				style="background: {colors.bg}; border: 1px solid {colors.border};"
				title={`${fullDate(cell.date)} — ${cell.memories.length} due`}
			>
				<div class="flex flex-col h-full">
					<div class="flex items-start justify-between">
						<span class="text-[10px] font-mono {cell.isToday ? 'text-synapse-glow font-bold' : 'text-dim'}">
							{cell.date.getDate()}
						</span>
						{#if cell.date.getDate() === 1}
							<span class="text-[9px] text-muted">{cell.date.toLocaleDateString(undefined, { month: 'short' })}</span>
						{/if}
					</div>
					{#if cell.memories.length > 0}
						<div class="flex-1 flex flex-col items-center justify-center gap-0.5">
							<span class="text-base sm:text-lg font-bold leading-none" style="color: {colors.text}">
								{cell.memories.length}
							</span>
							{#if cell.avgRetention > 0}
								<span class="text-[9px] text-muted">{(cell.avgRetention * 100).toFixed(0)}%</span>
							{/if}
						</div>
					{/if}
				</div>
			</button>
		{/each}
	</div>

	<!-- Legend -->
	<div class="flex items-center gap-4 text-[10px] text-muted flex-wrap px-1">
		<span class="flex items-center gap-1.5">
			<span class="w-3 h-3 rounded" style="background: rgba(239,68,68,0.16); border: 1px solid rgba(239,68,68,0.45);"></span>
			Overdue
		</span>
		<span class="flex items-center gap-1.5">
			<span class="w-3 h-3 rounded" style="background: rgba(245,158,11,0.18); border: 1px solid rgba(245,158,11,0.5);"></span>
			Due today
		</span>
		<span class="flex items-center gap-1.5">
			<span class="w-3 h-3 rounded" style="background: rgba(99,102,241,0.16); border: 1px solid rgba(99,102,241,0.45);"></span>
			Within 7 days
		</span>
		<span class="flex items-center gap-1.5">
			<span class="w-3 h-3 rounded" style="background: rgba(168,85,247,0.08); border: 1px solid rgba(168,85,247,0.2);"></span>
			Future (8+ days)
		</span>
	</div>

	<!-- Expanded day panel -->
	{#if selectedCell && selectedCell.memories.length > 0}
		<div class="p-5 glass rounded-xl space-y-3 animate-panel-in">
			<div class="flex items-center justify-between">
				<div>
					<h3 class="text-sm text-bright font-semibold">{fullDate(selectedCell.date)}</h3>
					<p class="text-xs text-dim mt-0.5">
						{selectedCell.memories.length} memor{selectedCell.memories.length === 1 ? 'y' : 'ies'} due
						· avg retention {(selectedCell.avgRetention * 100).toFixed(0)}%
					</p>
				</div>
				<button
					type="button"
					onclick={() => (selectedKey = null)}
					class="text-xs text-muted hover:text-dim px-2 py-1 rounded-lg hover:bg-white/[0.03]"
					aria-label="Close"
				>
					close ×
				</button>
			</div>
			<div class="space-y-2 max-h-96 overflow-y-auto pr-1">
				{#each selectedCell.memories.slice(0, 100) as m (m.id)}
					<div class="flex items-start gap-3 p-2.5 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition">
						<span
							class="w-2 h-2 mt-1.5 rounded-full flex-shrink-0"
							style="background: {NODE_TYPE_COLORS[m.nodeType] || '#8B95A5'}"
						></span>
						<div class="flex-1 min-w-0">
							<p class="text-sm text-text leading-snug line-clamp-2">{m.content}</p>
							<div class="flex items-center gap-2 mt-1 text-[10px] text-muted">
								<span>{m.nodeType}</span>
								{#if m.reviewCount !== undefined}
									<span>· {m.reviewCount} review{m.reviewCount === 1 ? '' : 's'}</span>
								{/if}
								{#each m.tags.slice(0, 2) as tag}
									<span class="px-1 py-0.5 bg-white/[0.04] rounded text-muted">{tag}</span>
								{/each}
							</div>
						</div>
						<div class="flex flex-col items-end gap-1 flex-shrink-0">
							<div class="w-12 h-1 bg-deep rounded-full overflow-hidden">
								<div
									class="h-full rounded-full"
									style="width: {m.retentionStrength * 100}%; background: {m.retentionStrength > 0.7
										? 'var(--color-recall)'
										: m.retentionStrength > 0.4
											? 'var(--color-warning)'
											: 'var(--color-decay)'}"
								></div>
							</div>
							<span class="text-[10px] text-muted">{(m.retentionStrength * 100).toFixed(0)}%</span>
						</div>
					</div>
				{/each}
				{#if selectedCell.memories.length > 100}
					<p class="text-xs text-muted text-center pt-2">
						+{selectedCell.memories.length - 100} more
					</p>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	@keyframes panel-in {
		from {
			opacity: 0;
			transform: translateY(-4px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
	.animate-panel-in {
		animation: panel-in 0.18s ease-out;
	}
</style>
