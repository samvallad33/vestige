<!--
  PatternTransferHeatmap — CrossProjectLearner visualization.
  Symmetric N×N project grid. Cell (row=A, col=B) intensity encodes how many
  patterns were learned in project A and reused in project B. Diagonal =
  self-transfer count (project reusing its own patterns).

  Color scale: muted (no transfers) → synapse glow → dream glow for high
  transfer counts. Cells are clickable (filters sidebar to A → B pairs) and
  hoverable (tooltip with count + top 3 pattern names).

  Responsive: the grid collapses to a vertical list of "A → B : count" rows
  on small viewports so the matrix is still scannable on mobile.
-->
<script lang="ts">
	import {
		buildTransferMatrix,
		flattenNonZero,
		matrixMaxCount,
		shortProjectName,
		type PatternCategory,
		type TransferPatternLike,
	} from './patterns-helpers';

	interface Pattern extends TransferPatternLike {
		category: PatternCategory;
		last_used: string;
		confidence: number;
	}

	interface Props {
		projects: string[];
		patterns: Pattern[];
		selectedCell: { from: string; to: string } | null;
		onCellClick: (from: string, to: string) => void;
	}

	let { projects, patterns, selectedCell, onCellClick }: Props = $props();

	// Matrix build, max-count, and non-zero flattening all live in
	// `patterns-helpers.ts` so they can be unit tested in the Vitest node env.
	const matrix = $derived(buildTransferMatrix(projects, patterns));
	const maxCount = $derived(matrixMaxCount(projects, matrix) || 1);

	// Hover tooltip state
	let hoveredCell = $state<{ from: string; to: string; x: number; y: number } | null>(null);

	function cellStyle(count: number): string {
		if (count === 0) {
			return 'background: rgba(255,255,255,0.02); border-color: rgba(99,102,241,0.05);';
		}
		const intensity = count / maxCount; // 0..1
		// Two-stop gradient: synapse (indigo) for low-mid → dream (purple) for high.
		// Alpha ramps 0.10 → 0.80 so even low-count cells read clearly.
		const alpha = 0.1 + intensity * 0.7;
		if (intensity < 0.5) {
			// Synapse-dominant
			return `background: rgba(99, 102, 241, ${alpha.toFixed(3)}); border-color: rgba(129, 140, 248, ${(alpha * 0.6).toFixed(3)}); box-shadow: 0 0 ${(intensity * 14).toFixed(1)}px rgba(129, 140, 248, ${(intensity * 0.45).toFixed(3)});`;
		} else {
			// Dream-dominant for the hottest cells
			const dreamIntensity = (intensity - 0.5) * 2; // 0..1 over upper half
			const r = Math.round(99 + (168 - 99) * dreamIntensity);
			const g = Math.round(102 + (85 - 102) * dreamIntensity);
			const b = Math.round(241 + (247 - 241) * dreamIntensity);
			return `background: rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)}); border-color: rgba(192, 132, 252, ${(alpha * 0.7).toFixed(3)}); box-shadow: 0 0 ${(6 + intensity * 18).toFixed(1)}px rgba(192, 132, 252, ${(intensity * 0.55).toFixed(3)});`;
		}
	}

	function cellTextClass(count: number): string {
		if (count === 0) return 'text-muted';
		const intensity = count / maxCount;
		if (intensity >= 0.5) return 'text-bright font-semibold';
		if (intensity >= 0.2) return 'text-text';
		return 'text-dim';
	}

	function handleCellHover(ev: MouseEvent, from: string, to: string) {
		const target = ev.currentTarget as HTMLElement;
		const rect = target.getBoundingClientRect();
		hoveredCell = {
			from,
			to,
			x: rect.left + rect.width / 2,
			y: rect.top
		};
	}

	function handleCellLeave() {
		hoveredCell = null;
	}

	// Axis labels are diagonal; trim long names via the shared helper.
	const shortProject = shortProjectName;

	function isSelected(from: string, to: string): boolean {
		return selectedCell !== null && selectedCell.from === from && selectedCell.to === to;
	}

	// Flattened list for the mobile fallback: only non-zero cells, sorted desc.
	const mobileList = $derived(flattenNonZero(projects, matrix));
</script>

<div class="glass-panel relative rounded-2xl p-5">
	<!-- Desktop / tablet: grid heatmap -->
	<div class="hidden md:block">
		<div class="mb-3 flex items-center justify-between">
			<div class="text-xs text-dim">
				Rows = origin project · Columns = destination project
			</div>
			<!-- Legend gradient -->
			<div class="flex items-center gap-2">
				<span class="text-[10px] text-muted">0</span>
				<div
					class="h-2 w-32 rounded-full"
					style="background: linear-gradient(to right, rgba(255,255,255,0.05), rgba(99,102,241,0.5), rgba(168,85,247,0.85));"
				></div>
				<span class="text-[10px] text-muted">{maxCount}</span>
			</div>
		</div>

		<div class="overflow-x-auto">
			<table class="w-full border-separate" style="border-spacing: 4px;">
				<thead>
					<tr>
						<th class="w-24"></th>
						{#each projects as proj (proj)}
							<th
								class="h-20 min-w-16 max-w-20 align-bottom"
								title={proj}
							>
								<div
									class="mx-auto flex h-20 w-6 items-end justify-center"
									style="writing-mode: vertical-rl; transform: rotate(180deg);"
								>
									<span class="text-[11px] text-dim font-medium tracking-wide">
										{shortProject(proj)}
									</span>
								</div>
							</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each projects as from (from)}
						<tr>
							<td class="w-24 pr-2 text-right text-[11px] text-dim" title={from}>
								{shortProject(from)}
							</td>
							{#each projects as to (to)}
								{@const cell = matrix[from][to]}
								{@const isDiag = from === to}
								<td class="p-0">
									<button
										type="button"
										class="group relative h-10 w-full min-w-12 rounded-md border transition-all duration-200 hover:scale-110 hover:z-10 focus:outline-none focus:ring-2 focus:ring-synapse-glow"
										style="{cellStyle(cell.count)} {isSelected(from, to)
											? 'outline: 2px solid var(--color-dream-glow); outline-offset: 1px;'
											: ''} {isDiag && cell.count > 0
											? 'border-style: dashed;'
											: ''}"
										onclick={() => onCellClick(from, to)}
										onmouseenter={(e) => handleCellHover(e, from, to)}
										onmouseleave={handleCellLeave}
										aria-label="{cell.count} patterns from {from} to {to}"
									>
										<span class="text-[11px] {cellTextClass(cell.count)}">
											{cell.count || ''}
										</span>
									</button>
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<!-- Hover tooltip -->
		{#if hoveredCell}
			{@const cell = matrix[hoveredCell.from][hoveredCell.to]}
			<div
				class="glass-panel pointer-events-none fixed z-50 max-w-xs rounded-lg p-3 text-xs shadow-2xl"
				style="left: {hoveredCell.x}px; top: {hoveredCell.y - 12}px; transform: translate(-50%, -100%);"
			>
				<div class="mb-1 flex items-center gap-2">
					<span class="font-mono text-dim">{shortProject(hoveredCell.from)}</span>
					<span class="text-synapse-glow">→</span>
					<span class="font-mono text-bright">{shortProject(hoveredCell.to)}</span>
				</div>
				<div class="mb-2 text-lg font-semibold text-bright">
					{cell.count}
					<span class="text-xs font-normal text-dim">
						{cell.count === 1 ? 'pattern' : 'patterns'} transferred
					</span>
				</div>
				{#if cell.topNames.length > 0}
					<div class="space-y-1 border-t border-synapse/10 pt-2">
						<div class="text-[10px] uppercase tracking-wider text-muted">Top patterns</div>
						{#each cell.topNames as name}
							<div class="truncate text-text">· {name}</div>
						{/each}
					</div>
				{:else}
					<div class="text-muted">No transfers recorded</div>
				{/if}
			</div>
		{/if}
	</div>

	<!-- Mobile: vertical list of non-zero transfers -->
	<div class="space-y-2 md:hidden">
		<div class="mb-2 text-xs text-dim">
			{mobileList.length} transfer pair{mobileList.length === 1 ? '' : 's'} · tap to filter
		</div>
		{#if mobileList.length === 0}
			<div class="rounded-lg bg-white/[0.02] p-4 text-center text-xs text-muted">
				No cross-project transfers recorded yet.
			</div>
		{:else}
			{#each mobileList as row (row.from + '->' + row.to)}
				<button
					type="button"
					class="flex w-full items-center justify-between rounded-lg border border-synapse/10 bg-white/[0.02] p-3 transition hover:border-synapse/30 hover:bg-white/[0.04] {isSelected(
						row.from,
						row.to
					)
						? 'ring-1 ring-dream-glow'
						: ''}"
					onclick={() => onCellClick(row.from, row.to)}
				>
					<div class="flex min-w-0 flex-col items-start gap-0.5">
						<div class="flex items-center gap-1.5 text-xs">
							<span class="font-mono text-dim">{shortProject(row.from)}</span>
							<span class="text-synapse-glow">→</span>
							<span class="font-mono text-bright">{shortProject(row.to)}</span>
						</div>
						{#if row.topNames.length > 0}
							<div class="truncate text-[11px] text-muted">
								{row.topNames.join(' · ')}
							</div>
						{/if}
					</div>
					<span
						class="ml-3 flex-shrink-0 rounded-full bg-synapse/15 px-2 py-0.5 text-xs font-semibold text-synapse-glow"
					>
						{row.count}
					</span>
				</button>
			{/each}
		{/if}
	</div>
</div>
