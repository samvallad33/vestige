<script lang="ts">
	/**
	 * Timeline spine — bottom instrument (Increment 6).
	 *
	 * A 720-frame loop track: one tick per story beat (positioned at
	 * beatFrame/loopFrames), a playhead riding the live frame, and the active
	 * beat's label glowing as the wavefront lands. Pointer-events: none —
	 * pure instrument, never a control surface (§7.3).
	 */
	import type { PathStepMeta } from '$lib/observatory/path-builder';

	interface Props {
		steps?: PathStepMeta[];
		frame?: number;
		loopFrames?: number;
	}

	let { steps = [], frame = 0, loopFrames = 720 }: Props = $props();

	const pct = (f: number) => (f / loopFrames) * 100;

	// A beat is "live" from just before arrival through its afterglow.
	function beatEnergy(beatFrame: number, f: number): number {
		const d = f - beatFrame;
		if (d < -14 || d > 90) return 0;
		if (d < 0) return 1 + d / 14; // attack
		return 1 - d / 90; // decay
	}

	let activeLabel = $derived.by(() => {
		let best = '';
		let bestE = 0.15;
		for (const s of steps) {
			const e = beatEnergy(s.beatFrame, frame);
			if (e > bestE) {
				bestE = e;
				best = s.label;
			}
		}
		return best;
	});
</script>

{#if steps.length > 0}
	<div class="spine">
		{#if activeLabel}
			<div class="active-label">{activeLabel}</div>
		{/if}
		<div class="track">
			{#each steps as s (s.beatFrame)}
				<div
					class="tick"
					class:hot={beatEnergy(s.beatFrame, frame) > 0}
					class:backward={s.kind === 1}
					style="left: {pct(s.beatFrame)}%; opacity: {0.45 +
						0.55 * beatEnergy(s.beatFrame, frame)}"
					title={s.label}
				></div>
			{/each}
			<div class="playhead" style="left: {pct(frame)}%"></div>
		</div>
	</div>
{/if}

<style>
	.spine {
		position: absolute;
		left: 8%;
		right: 8%;
		/* Chromeless route: anchor to the true viewport bottom, clearing the
		   home-indicator on notched phones (safe-area) — audit blocker B3. */
		bottom: calc(2.5rem + env(safe-area-inset-bottom, 0px));
		pointer-events: none;
	}

	.active-label {
		text-align: center;
		margin-bottom: 0.6rem;
		font-family: 'SF Mono', ui-monospace, Menlo, Consolas, monospace;
		font-size: 0.72rem;
		letter-spacing: 0.08em;
		color: #cfe9ff;
		text-shadow: 0 0 24px rgba(30, 180, 255, 0.35);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.track {
		position: relative;
		height: 2px;
		background: rgba(255, 255, 255, 0.07);
		border-radius: 1px;
	}

	.tick {
		position: absolute;
		top: 50%;
		width: 3px;
		height: 10px;
		transform: translate(-50%, -50%);
		border-radius: 2px;
		background: #6ef0e6;
		transition: opacity 0.2s linear;
	}

	.tick.hot {
		box-shadow: 0 0 12px rgba(110, 240, 230, 0.75);
	}

	.tick.backward {
		background: #ff4070;
	}

	.tick.backward.hot {
		box-shadow: 0 0 12px rgba(255, 64, 112, 0.75);
	}

	.playhead {
		position: absolute;
		top: 50%;
		width: 1.5px;
		height: 16px;
		transform: translate(-50%, -50%);
		background: rgba(207, 233, 255, 0.9);
		box-shadow: 0 0 10px rgba(30, 180, 255, 0.5);
	}
</style>
