<script lang="ts">
	/**
	 * Demo verdict card — DOM instrument overlay (§7.3 grammar, ported from
	 * docs/launch/causal-brain-demo.html's .verdict).
	 *
	 * Opacity is a PURE function of the frame prop (smoothstep in TS, NO CSS
	 * transitions) so capture mode (?frame=N) renders the exact same card at
	 * the exact same alpha every time. Defaults preserve the salience-rescue
	 * card byte-for-byte (window 600 → 660, triumph tone adds no class);
	 * firewall reuses it with tone="quarantine" + fadeWindow 480/495/605/620.
	 */

	/** Structural shape — RescueVerdictCopy and FirewallPlan copy both assign. */
	interface VerdictContent {
		headline: string;
		causeLabel: string;
		receipt: string;
	}

	interface Props {
		frame?: number;
		verdict: VerdictContent;
		/** [fadeInStart, fadeInEnd, fadeOutStart, fadeOutEnd] loop frames. */
		fadeWindow?: [number, number, number, number];
		tone?: 'triumph' | 'quarantine';
	}

	let {
		frame = 0,
		verdict,
		// Hold the verdict on screen for the REST of the loop, not a ~1s flash.
		// Fade in over 600→620, hold fully readable through the field's decay to
		// rest, then fade out just before the seamless wrap (705→719, back to 0
		// at 719 so the loop seam stays clean). This is the demo money-card: the
		// viewer needs seconds to read "root cause found", not a blink.
		fadeWindow = [600, 620, 705, 719],
		tone = 'triumph'
	}: Props = $props();

	const ss = (a: number, b: number, f: number): number => {
		const t = Math.min(1, Math.max(0, (f - a) / (b - a)));
		return t * t * (3 - 2 * t);
	};

	let opacity = $derived(
		ss(fadeWindow[0], fadeWindow[1], frame) * (1 - ss(fadeWindow[2], fadeWindow[3], frame))
	);
</script>

{#if opacity > 0.001}
	<div class="verdict" class:quarantine={tone === 'quarantine'} style="opacity: {opacity}">
		<div class="k">{verdict.headline}</div>
		<div class="v">{verdict.causeLabel}</div>
		<div class="s">{verdict.receipt}</div>
	</div>
{/if}

<style>
	.verdict {
		position: fixed;
		left: 50%;
		top: 50%;
		transform: translate(-50%, -50%);
		text-align: center;
		font-family:
			-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
		padding: clamp(18px, 3vw, 40px) clamp(28px, 6vw, 80px);
		border-radius: 20px;
		background: radial-gradient(
			ellipse at center,
			rgba(5, 7, 14, 0.86) 0%,
			rgba(5, 7, 14, 0.72) 60%,
			rgba(5, 7, 14, 0) 100%
		);
		pointer-events: none;
	}

	.k {
		font-size: clamp(13px, 1.8vw, 18px);
		color: #9fd0e4;
		letter-spacing: 0.16em;
		text-transform: uppercase;
	}

	.v {
		font-size: clamp(32px, 6.4vw, 72px);
		font-weight: 600;
		margin-top: 0.12em;
		line-height: 1.05;
		background: linear-gradient(90deg, #7fe6c0, #6ef0e6, #a6dcff);
		-webkit-background-clip: text;
		background-clip: text;
		color: transparent;
		filter: drop-shadow(0 0 26px rgba(110, 240, 220, 0.45));
	}

	.s {
		font-size: clamp(11px, 1.5vw, 15px);
		color: #8fb0be;
		margin-top: 0.6em;
		font-family: 'SF Mono', ui-monospace, Menlo, Consolas, monospace;
		letter-spacing: 0.04em;
	}

	/* Quarantine tone (firewall verdict) — crimson-ember palette. Triumph adds
	   no class, so the rescue card's DOM + styles stay byte-identical. */
	.quarantine .k {
		color: #ffb0a6;
	}

	.quarantine .v {
		background: linear-gradient(90deg, #ff6a5e, #ff9d6b, #ffd2a8);
		filter: drop-shadow(0 0 26px rgba(255, 90, 70, 0.45));
	}

	.quarantine .s {
		color: #d9a49a;
	}
</style>
