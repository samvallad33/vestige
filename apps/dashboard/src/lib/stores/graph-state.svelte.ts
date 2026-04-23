// Shared graph state using Svelte 5 $state runes
// This store manages temporal playback, dream mode, and brightness.

const BRIGHTNESS_KEY = 'vestige:graph:brightness';
const BRIGHTNESS_DEFAULT = 1.0;
const BRIGHTNESS_MIN = 0.5;
const BRIGHTNESS_MAX = 2.5;

function loadBrightness(): number {
	if (typeof localStorage === 'undefined') return BRIGHTNESS_DEFAULT;
	const raw = localStorage.getItem(BRIGHTNESS_KEY);
	if (raw === null) return BRIGHTNESS_DEFAULT;
	const n = Number(raw);
	if (!Number.isFinite(n)) return BRIGHTNESS_DEFAULT;
	return Math.min(BRIGHTNESS_MAX, Math.max(BRIGHTNESS_MIN, n));
}

export const graphState = createGraphState();

function createGraphState() {
	let temporalEnabled = $state(false);
	let temporalDate = $state<Date>(new Date());
	let temporalPlaying = $state(false);
	let temporalSpeed = $state(1); // days per second: 1, 7, 30
	let dreamMode = $state(false);
	let brightness = $state(loadBrightness());

	return {
		get temporalEnabled() {
			return temporalEnabled;
		},
		set temporalEnabled(v: boolean) {
			temporalEnabled = v;
		},

		get temporalDate() {
			return temporalDate;
		},
		set temporalDate(v: Date) {
			temporalDate = v;
		},

		get temporalPlaying() {
			return temporalPlaying;
		},
		set temporalPlaying(v: boolean) {
			temporalPlaying = v;
		},

		get temporalSpeed() {
			return temporalSpeed;
		},
		set temporalSpeed(v: number) {
			temporalSpeed = v;
		},

		get dreamMode() {
			return dreamMode;
		},
		set dreamMode(v: boolean) {
			dreamMode = v;
		},

		// Global brightness multiplier for the 3D graph. Scales node emissive
		// intensity, glow opacity, and edge opacity. Persisted in localStorage
		// so it survives reloads.
		get brightness() {
			return brightness;
		},
		set brightness(v: number) {
			const clamped = Math.min(BRIGHTNESS_MAX, Math.max(BRIGHTNESS_MIN, v));
			brightness = clamped;
			if (typeof localStorage !== 'undefined') {
				try {
					localStorage.setItem(BRIGHTNESS_KEY, String(clamped));
				} catch {
					/* private browsing / quota — ignore */
				}
			}
		},
		brightnessMin: BRIGHTNESS_MIN,
		brightnessMax: BRIGHTNESS_MAX,
		brightnessDefault: BRIGHTNESS_DEFAULT,
	};
}
