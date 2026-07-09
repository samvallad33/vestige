/**
 * Deterministic demo clock for the Cognitive Observatory.
 *
 * Fixed 60fps loop, 720-frame period (12 seconds), seeded PRNG.
 * No Math.random() or performance.now() for simulation state.
 * performance.now() only schedules frames; positions/colors/path are deterministic.
 *
 * Pattern: https://gafferongames.com/post/fix_your_timestep/
 */

// ---- MurmurHash3-compatible 32-bit hash (xmur3) ----
// Used to hash a seed string into a 32-bit integer for mulberry32.
function xmur3(str: string): () => number {
	let h = 1779033703 ^ str.length;
	for (let i = 0; i < str.length; i++) {
		h = Math.imul(h ^ str.charCodeAt(i), 2654435761);
		h = (h << 13) | (h >>> 19);
	}
	return function () {
		let t = (h += 0x6d2b79f5);
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// ---- Mulberry32 seeded PRNG ----
// Fast, deterministic, 32-bit. Good enough for demo visuals.
function mulberry32(seed: number) {
	return function () {
		let t = (seed += 0x6d2b79f5);
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// ---- DemoClock ----
export interface DemoClockConfig {
	/** Frames per second (fixed) */
	fps?: number;
	/** Frames per loop (default 720 = 12s at 60fps) */
	loopFrames?: number;
	/** Seed string for deterministic PRNG */
	seed: string;
}

export interface DemoClockState {
	/** Current frame (integer, wraps at loopFrames) */
	frame: number;
	/** Loop phase: 0..1 */
	phase: number;
	/** PRNG function seeded from the provided seed */
	rng: () => number;
	/** Total frames elapsed (monotonic, does not wrap) */
	totalFrames: number;
}

export class DemoClock {
	private readonly fps: number;
	private readonly loopFrames: number;
	private readonly seedStr: string;
	private _frame: number;
	private _totalFrames: number;
	private _rng: () => number;

	constructor(config: DemoClockConfig) {
		this.fps = config.fps ?? 60;
		this.loopFrames = config.loopFrames ?? 720;
		this.seedStr = config.seed;
		this._frame = 0;
		this._totalFrames = 0;
		// Hash the seed string into a 32-bit integer, then create a mulberry32 PRNG
		const hash = xmur3(this.seedStr)();
		this._rng = mulberry32(Math.floor(hash * 2 ** 32));
	}

	/** Advance the clock by one frame. Returns the new state. */
	tick(): DemoClockState {
		this._frame = (this._frame + 1) % this.loopFrames;
		this._totalFrames++;
		return this.state;
	}

	/** Get the current clock state without advancing. */
	get state(): DemoClockState {
		return {
			frame: this._frame,
			phase: this._frame / this.loopFrames,
			rng: this._rng,
			totalFrames: this._totalFrames
		};
	}

	/** Reset the clock to frame 0. */
	reset(): void {
		this._frame = 0;
		this._totalFrames = 0;
		// Re-seed the PRNG from the original seed
		const hash = xmur3(this.seedStr)();
		this._rng = mulberry32(Math.floor(hash * 2 ** 32));
	}

	/** Get the loop duration in seconds. */
	get loopDuration(): number {
		return this.loopFrames / this.fps;
	}

	/** Frames per loop (capture mode needs this to freeze deterministically). */
	get framesPerLoop(): number {
		return this.loopFrames;
	}
}

// ---- Utility: deterministic position on a golden-angle sphere ----
// Golden-angle placement is deterministic by index. The rng provides
// a small seed-based perturbation so different seeds produce different layouts.
export function deterministicSpherePosition(
	index: number,
	total: number,
	radius: number,
	rng: () => number
): [number, number, number] {
	const goldenAngle = Math.PI * (3 - Math.sqrt(5));
	const y = 1 - (index / (total - 1 || 1)) * 2; // -1 to 1
	const radiusAtY = Math.sqrt(1 - y * y);
	const theta = goldenAngle * index;
	const x = Math.cos(theta) * radiusAtY;
	const z = Math.sin(theta) * radiusAtY;

	// Small seed-based perturbation (±5% of radius)
	const px = (rng() - 0.5) * 0.1 * radius;
	const py = (rng() - 0.5) * 0.1 * radius;
	const pz = (rng() - 0.5) * 0.1 * radius;

	return [x * radius + px, y * radius + py, z * radius + pz];
}
