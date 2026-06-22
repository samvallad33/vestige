// ═══════════════════════════════════════════════════════════════════════
//  reveal — scroll-into-view entrance animation as a Svelte action.
// ───────────────────────────────────────────────────────────────────────
//  Usage:  <div use:reveal>            // default rise+fade
//          <div use:reveal={{ y: 24, delay: 80, once: true }}>
//
//  Adds an IntersectionObserver that flips the element to its "revealed"
//  state the first time it scrolls into view. Pairs with the .reveal /
//  .reveal-in CSS in app.css. Honors prefers-reduced-motion by revealing
//  instantly (no transform), so motion-sensitive users still see content.
//
//  Why an action and not CSS scroll-timeline alone: scroll-driven
//  animation-timeline is great for continuous scrubbing, but for a simple
//  "appear once when seen" we want a one-shot that also works as a staggered
//  list entrance with per-item delay — an observer is the robust path with
//  universal mid-2026 support.
// ═══════════════════════════════════════════════════════════════════════

export interface RevealOptions {
	/** Pixels to translate up from on entry. */
	y?: number;
	/** ms delay before the transition starts (use for stagger). */
	delay?: number;
	/** Only reveal once, then stop observing (default true). */
	once?: boolean;
	/** 0-1 visibility threshold to trigger. */
	threshold?: number;
}

const prefersReducedMotion = () =>
	typeof window !== 'undefined' &&
	window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

export function reveal(node: HTMLElement, options: RevealOptions = {}) {
	const { y = 16, delay = 0, once = true, threshold = 0.12 } = options;

	// Motion-off: show immediately, do nothing else.
	if (prefersReducedMotion()) {
		node.classList.add('reveal-in');
		return {};
	}

	node.classList.add('reveal');
	node.style.setProperty('--reveal-y', `${y}px`);
	if (delay) node.style.setProperty('--reveal-delay', `${delay}ms`);

	const io = new IntersectionObserver(
		(entries) => {
			for (const entry of entries) {
				if (entry.isIntersecting) {
					node.classList.add('reveal-in');
					if (once) io.unobserve(node);
				} else if (!once) {
					node.classList.remove('reveal-in');
				}
			}
		},
		{ threshold, rootMargin: '0px 0px -8% 0px' }
	);

	io.observe(node);

	return {
		destroy() {
			io.disconnect();
		},
	};
}
