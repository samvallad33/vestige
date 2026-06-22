// Memory Cinema — the camera director.
//
// Drives a smooth, cinematic camera flight through a planned CinemaPath. Pure
// choreography: it mutates a THREE.PerspectiveCamera + an OrbitControls-like
// target each frame and emits beat-arrival callbacks the narrator + sandbox
// hook into. It knows nothing about which renderer (WebGL/WebGPU) is on screen,
// so it works identically for the legacy graph and the WebGPU sandbox.
//
// Respects prefers-reduced-motion: when reduced, it JUMP-CUTS between beats
// (instant position, dwell, advance) instead of flying — captions still fire.

import * as THREE from 'three';
import type { CinemaPath, CinemaBeat } from './pathfinder';

export interface DirectorCallbacks {
	/** Fired once when the camera arrives at (or cuts to) a beat. */
	onBeat?: (beat: CinemaBeat, index: number) => void;
	/** Fired when the whole tour finishes. */
	onComplete?: () => void;
	/** Fired every frame with overall progress 0..1 (for a scrubber/progress bar). */
	onProgress?: (t: number) => void;
}

export interface DirectorOptions {
	/** Seconds of camera flight between consecutive beats. */
	flightSeconds?: number;
	/** Seconds the camera dwells on each beat before advancing. */
	dwellSeconds?: number;
	/** Stand-off distance from the focused node, in world units. */
	standoff?: number;
	/** Instant cuts instead of flights (prefers-reduced-motion). */
	reducedMotion?: boolean;
}

type Phase = 'idle' | 'flying' | 'dwelling' | 'done';

const _tmpDir = new THREE.Vector3();
const _tmpUp = new THREE.Vector3(0, 1, 0);

export class CinemaDirector {
	private camera: THREE.PerspectiveCamera;
	private target: THREE.Vector3;
	private positions: Map<string, THREE.Vector3>;
	private path: CinemaPath;
	private cb: DirectorCallbacks;
	private opts: Required<DirectorOptions>;

	private phase: Phase = 'idle';
	private beatIndex = 0;
	private phaseElapsed = 0;

	// Flight interpolation endpoints.
	private fromPos = new THREE.Vector3();
	private toPos = new THREE.Vector3();
	private fromTarget = new THREE.Vector3();
	private toTarget = new THREE.Vector3();

	constructor(
		camera: THREE.PerspectiveCamera,
		target: THREE.Vector3,
		positions: Map<string, THREE.Vector3>,
		path: CinemaPath,
		cb: DirectorCallbacks = {},
		opts: DirectorOptions = {}
	) {
		this.camera = camera;
		this.target = target;
		this.positions = positions;
		this.path = path;
		this.cb = cb;
		this.opts = {
			flightSeconds: opts.flightSeconds ?? 2.4,
			dwellSeconds: opts.dwellSeconds ?? 3.2,
			standoff: opts.standoff ?? 26,
			reducedMotion: opts.reducedMotion ?? false,
		};
	}

	get totalBeats(): number {
		return this.path.beats.length;
	}

	get isRunning(): boolean {
		return this.phase !== 'idle' && this.phase !== 'done';
	}

	/** Begin the tour from the first beat. */
	start(): void {
		if (this.path.beats.length === 0) {
			this.phase = 'done';
			this.cb.onComplete?.();
			return;
		}
		this.beatIndex = 0;
		this.beginFlightTo(0);
	}

	stop(): void {
		this.phase = 'done';
	}

	/** Compute the camera stand-off position for a beat's node. */
	private framePosition(beat: CinemaBeat, out: THREE.Vector3): THREE.Vector3 {
		const nodePos = this.positions.get(beat.nodeId);
		if (!nodePos) {
			// Node has no resolved position yet — keep current framing.
			return out.copy(this.camera.position);
		}
		// Offset back + up from the node along the current view direction so the
		// node sits centered with a cinematic slightly-above angle.
		_tmpDir.copy(this.camera.position).sub(nodePos);
		if (_tmpDir.lengthSq() < 1e-4) _tmpDir.set(0, 0.4, 1);
		_tmpDir.normalize();
		// Bias the approach vector upward a touch for a filmic tilt.
		_tmpDir.addScaledVector(_tmpUp, 0.35).normalize();
		return out.copy(nodePos).addScaledVector(_tmpDir, this.opts.standoff);
	}

	private beginFlightTo(index: number): void {
		const beat = this.path.beats[index];
		const nodePos = this.positions.get(beat.nodeId);

		this.fromPos.copy(this.camera.position);
		this.fromTarget.copy(this.target);
		this.framePosition(beat, this.toPos);
		this.toTarget.copy(nodePos ?? this.target);
		this.phaseElapsed = 0;

		if (this.opts.reducedMotion) {
			// Jump-cut: snap, fire the beat, go straight to dwelling.
			this.camera.position.copy(this.toPos);
			this.target.copy(this.toTarget);
			this.phase = 'dwelling';
			this.cb.onBeat?.(beat, index);
		} else {
			this.phase = 'flying';
		}
	}

	/** Advance the choreography. Call once per animation frame with delta seconds. */
	update(deltaSeconds: number): void {
		if (this.phase === 'idle' || this.phase === 'done') return;
		// Clamp dt so a tab-switch stall doesn't teleport the camera.
		const dt = Math.min(deltaSeconds, 0.05);
		this.phaseElapsed += dt;

		if (this.phase === 'flying') {
			const t = Math.min(1, this.phaseElapsed / this.opts.flightSeconds);
			const e = easeInOutCubic(t);
			this.camera.position.lerpVectors(this.fromPos, this.toPos, e);
			this.target.lerpVectors(this.fromTarget, this.toTarget, e);
			if (t >= 1) {
				this.phase = 'dwelling';
				this.phaseElapsed = 0;
				this.cb.onBeat?.(this.path.beats[this.beatIndex], this.beatIndex);
			}
		} else if (this.phase === 'dwelling') {
			// Gentle drift during the dwell keeps the shot alive (skipped if reduced).
			if (!this.opts.reducedMotion) {
				const nodePos = this.positions.get(this.path.beats[this.beatIndex].nodeId);
				if (nodePos) this.target.lerp(nodePos, 0.02);
			}
			if (this.phaseElapsed >= this.opts.dwellSeconds) {
				const nextIndex = this.beatIndex + 1;
				if (nextIndex >= this.path.beats.length) {
					this.phase = 'done';
					this.cb.onProgress?.(1);
					this.cb.onComplete?.();
					return;
				}
				this.beatIndex = nextIndex;
				this.beginFlightTo(nextIndex);
			}
		}

		// Overall progress across the whole tour (beat + intra-beat fraction).
		const per = 1 / this.path.beats.length;
		const intra =
			this.phase === 'flying'
				? Math.min(1, this.phaseElapsed / this.opts.flightSeconds) * 0.5
				: 0.5 + Math.min(1, this.phaseElapsed / this.opts.dwellSeconds) * 0.5;
		this.cb.onProgress?.(Math.min(1, this.beatIndex * per + intra * per));
	}
}

function easeInOutCubic(t: number): number {
	return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}
