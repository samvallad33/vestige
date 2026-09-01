import { goto } from '$app/navigation';
import { writable } from 'svelte/store';

export type RouteBurstPhase = 'idle' | 'covering' | 'revealing';

export interface RouteBurstState {
	phase: RouteBurstPhase;
	x: number;
	y: number;
	color: string;
	reduced: boolean;
}

const IDLE: RouteBurstState = {
	phase: 'idle',
	x: 50,
	y: 50,
	color: '#E9FFB7',
	reduced: false
};

export const routeBurst = writable<RouteBurstState>(IDLE);

let navigating = false;

function wait(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function nextFrame(): Promise<void> {
	return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/**
 * Cover the old route at the swarm's flash peak, navigate behind the blast, then
 * reveal the destination. The store lives in the root layout, so the veil survives
 * destruction of the Palace route and prevents a black/void handoff.
 */
export async function burstNavigate(
	href: string,
	options: { clientX: number; clientY: number; color: string; reduced: boolean }
): Promise<boolean> {
	if (navigating) return false;
	navigating = true;

	try {
		// NO VEIL (Sam: "COMPLETELY REMOVE IT" — the white/color ball at the end).
		// The shape-specific explosion IS the transition. We just let the detonation
		// finish flinging its debris on the Palace, then navigate SILENTLY — the
		// destination page mounts underneath with its own field already lit, so there
		// is no cover, no wash, no ball. The store stays idle so nothing renders.
		await nextFrame();
		await wait(options.reduced ? 40 : 130);
		await goto(href);
		return true;
	} finally {
		navigating = false;
	}
}

