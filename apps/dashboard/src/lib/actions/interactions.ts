// ═══════════════════════════════════════════════════════════════════════
//  Micro-interaction actions — the "alive" layer for pointer feel.
// ───────────────────────────────────────────────────────────────────────
//  All actions no-op under prefers-reduced-motion and clean up their own
//  listeners on destroy. Pure pointer events, no dependency.
// ═══════════════════════════════════════════════════════════════════════

const prefersReducedMotion = () =>
	typeof window !== 'undefined' &&
	window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

// ── magnetic: element drifts toward the cursor, snaps back on leave ──────
export interface MagneticOptions {
	/** How far the element is allowed to drift, in px. */
	strength?: number;
}
export function magnetic(node: HTMLElement, options: MagneticOptions = {}) {
	if (prefersReducedMotion()) return {};
	const strength = options.strength ?? 8;
	let raf = 0;

	function onMove(e: PointerEvent) {
		const r = node.getBoundingClientRect();
		const mx = e.clientX - (r.left + r.width / 2);
		const my = e.clientY - (r.top + r.height / 2);
		const dx = Math.max(-1, Math.min(1, mx / (r.width / 2))) * strength;
		const dy = Math.max(-1, Math.min(1, my / (r.height / 2))) * strength;
		cancelAnimationFrame(raf);
		raf = requestAnimationFrame(() => {
			node.style.transform = `translate(${dx}px, ${dy}px)`;
		});
	}
	function onLeave() {
		cancelAnimationFrame(raf);
		node.style.transform = '';
	}

	node.style.transition = 'transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)';
	node.addEventListener('pointermove', onMove);
	node.addEventListener('pointerleave', onLeave);
	return {
		destroy() {
			node.removeEventListener('pointermove', onMove);
			node.removeEventListener('pointerleave', onLeave);
			cancelAnimationFrame(raf);
		},
	};
}

// ── tilt: 3D parallax tilt toward the cursor (cards, hero panels) ────────
export interface TiltOptions {
	/** Max tilt in degrees. */
	max?: number;
	/** Lift the card toward the viewer on hover, in px. */
	lift?: number;
	/** Add a moving sheen highlight that follows the cursor. */
	glare?: boolean;
}
export function tilt(node: HTMLElement, options: TiltOptions = {}) {
	if (prefersReducedMotion()) return {};
	const max = options.max ?? 6;
	const lift = options.lift ?? 0;
	const glare = options.glare ?? false;
	let raf = 0;

	node.style.transformStyle = 'preserve-3d';
	node.style.transition = 'transform 0.3s cubic-bezier(0.23, 1, 0.32, 1)';

	if (glare) {
		node.style.setProperty('--glare-x', '50%');
		node.style.setProperty('--glare-y', '50%');
		node.style.setProperty('--glare-o', '0');
	}

	function onMove(e: PointerEvent) {
		const r = node.getBoundingClientRect();
		const px = (e.clientX - r.left) / r.width;
		const py = (e.clientY - r.top) / r.height;
		const rx = (0.5 - py) * max * 2;
		const ry = (px - 0.5) * max * 2;
		cancelAnimationFrame(raf);
		raf = requestAnimationFrame(() => {
			node.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg)${lift ? ` translateZ(${lift}px)` : ''}`;
			if (glare) {
				node.style.setProperty('--glare-x', `${px * 100}%`);
				node.style.setProperty('--glare-y', `${py * 100}%`);
				node.style.setProperty('--glare-o', '1');
			}
		});
	}
	function onLeave() {
		cancelAnimationFrame(raf);
		node.style.transform = '';
		if (glare) node.style.setProperty('--glare-o', '0');
	}

	node.addEventListener('pointermove', onMove);
	node.addEventListener('pointerleave', onLeave);
	return {
		destroy() {
			node.removeEventListener('pointermove', onMove);
			node.removeEventListener('pointerleave', onLeave);
			cancelAnimationFrame(raf);
		},
	};
}

// ── spotlight: a soft radial glow that tracks the cursor over a surface ──
//  Sets --spot-x / --spot-y custom props the element's CSS can read to
//  position a radial-gradient highlight. Makes large panels feel responsive
//  to the pointer even when nothing is "hovered".
export function spotlight(node: HTMLElement) {
	if (prefersReducedMotion()) return {};
	let raf = 0;
	function onMove(e: PointerEvent) {
		const r = node.getBoundingClientRect();
		cancelAnimationFrame(raf);
		raf = requestAnimationFrame(() => {
			node.style.setProperty('--spot-x', `${e.clientX - r.left}px`);
			node.style.setProperty('--spot-y', `${e.clientY - r.top}px`);
			node.style.setProperty('--spot-o', '1');
		});
	}
	function onLeave() {
		node.style.setProperty('--spot-o', '0');
	}
	node.addEventListener('pointermove', onMove);
	node.addEventListener('pointerleave', onLeave);
	return {
		destroy() {
			node.removeEventListener('pointermove', onMove);
			node.removeEventListener('pointerleave', onLeave);
			cancelAnimationFrame(raf);
		},
	};
}
