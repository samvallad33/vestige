// ─────────────────────────────────────────────────────────────────────────────
// shell.svelte.ts — global OS-shell coordination state.
//
// When a full-screen takeover owns the keyboard (Memory Cinema, an Observatory
// demo takeover), the OS command palette must NOT respond to ⌘K/Escape — otherwise
// ⌘K opens the palette *behind* the takeover and steals Escape, so the takeover
// can no longer be closed (the launch audit's most serious interaction bug).
//
// A takeover raises overlayActive; the shell's ⌘K + palette are suppressed while
// it is set. Reference-counted so nested/overlapping takeovers behave.
// ─────────────────────────────────────────────────────────────────────────────

let overlayCount = $state(0);

export const shell = {
	/** True while ANY full-screen takeover (Cinema, demo) owns the keyboard. */
	get overlayActive(): boolean {
		return overlayCount > 0;
	},
	/** A takeover calls this on open (e.g. in an $effect/onMount). */
	pushOverlay(): void {
		overlayCount += 1;
	},
	/** …and this on close/teardown. Guarded so it never goes negative. */
	popOverlay(): void {
		overlayCount = Math.max(0, overlayCount - 1);
	}
};
