<script lang="ts">
	// ═══════════════════════════════════════════════════════════════════
	//  ANIMATED NUMBER — smooth count-up / tween between values.
	// ───────────────────────────────────────────────────────────────────
	//  Tweens from the previous value to the new one with an ease-out curve.
	//  Counts up on first mount (from 0) so every figure feels like it's
	//  being tallied live, and re-tweens whenever the bound value changes
	//  (e.g. a websocket push). Respects prefers-reduced-motion: those users
	//  get the final value instantly. Pure rAF — no dependency.
	// ═══════════════════════════════════════════════════════════════════
	interface Props {
		value: number;
		/** Decimal places to render. */
		decimals?: number;
		/** Multiply before formatting (e.g. 100 for a 0-1 ratio → percent). */
		scale?: number;
		prefix?: string;
		suffix?: string;
		duration?: number;
		class?: string;
		/** Group thousands with separators. */
		group?: boolean;
	}
	let {
		value,
		decimals = 0,
		scale = 1,
		prefix = '',
		suffix = '',
		duration = 900,
		class: cls = '',
		group = true,
	}: Props = $props();

	let display = $state(0);
	let raf = 0;
	let from = 0;
	let start = 0;
	let mounted = false;

	const reduceMotion =
		typeof window !== 'undefined' &&
		window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

	function easeOutExpo(t: number): number {
		return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
	}

	function animateTo(target: number) {
		if (reduceMotion) {
			display = target;
			return;
		}
		cancelAnimationFrame(raf);
		from = display;
		start = 0;
		function tick(ts: number) {
			if (!start) start = ts;
			const t = Math.min(1, (ts - start) / duration);
			display = from + (target - from) * easeOutExpo(t);
			if (t < 1) raf = requestAnimationFrame(tick);
			else display = target;
		}
		raf = requestAnimationFrame(tick);
	}

	$effect(() => {
		// Track the incoming value; tween toward it.
		const target = value;
		if (!mounted) {
			mounted = true;
			animateTo(target);
		} else {
			animateTo(target);
		}
		return () => cancelAnimationFrame(raf);
	});

	let formatted = $derived(
		(() => {
			const v = display * scale;
			const fixed = v.toFixed(decimals);
			if (!group) return fixed;
			const [int, frac] = fixed.split('.');
			const grouped = int.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
			return frac !== undefined ? `${grouped}.${frac}` : grouped;
		})()
	);
</script>

<span class="tabular-nums {cls}">{prefix}{formatted}{suffix}</span>
