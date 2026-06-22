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
import type { ResolvedShot } from './auteur';

export interface DirectorCallbacks {
	/** Fired once when the camera arrives at (or cuts to) a beat. The resolved
	 * shot for the beat is passed so consumers can drive storm/score/captions. */
	onBeat?: (beat: CinemaBeat, index: number, shot: ResolvedShot | null) => void;
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
	/** Optional per-beat director's plan (one ResolvedShot per beat, aligned by
	 * index). When ABSENT the camera behaves byte-identically to the pre-Auteur
	 * director — every value falls back to the constants above. When present,
	 * each shot's move/angle/dutch/standoff/flight/dwell/cut directs that beat. */
	shots?: ResolvedShot[];
	/** When true, the camera frames the WORLD ORIGIN every shot (the WebGPU storm
	 * is pinned there) instead of flying out to scattered node positions — so the
	 * subject is ALWAYS centered and can never fly off-screen. Camera variety
	 * comes purely from angle/standoff/orbit. Used by the WebGPU sandbox path. */
	centerOnOrigin?: boolean;
}

type Phase = 'idle' | 'flying' | 'dwelling' | 'done';

const _tmpDir = new THREE.Vector3();
const _tmpUp = new THREE.Vector3(0, 1, 0);
const _origin = new THREE.Vector3(0, 0, 0);

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
			shots: opts.shots ?? [],
			centerOnOrigin: opts.centerOnOrigin ?? false,
		};
	}

	/** The resolved shot directing a beat, or null when no plan was supplied
	 * (→ the camera uses the constant defaults = pre-Auteur behavior). */
	private shotAt(index: number): ResolvedShot | null {
		return this.opts.shots[index] ?? null;
	}

	/** Per-beat flight duration: the shot's value, else the global default. A
	 * hard/match cut has zero flight (handled in beginFlightTo). */
	private flightSecondsAt(index: number): number {
		return this.shotAt(index)?.flightSeconds ?? this.opts.flightSeconds;
	}
	private dwellSecondsAt(index: number): number {
		return this.shotAt(index)?.dwellSeconds ?? this.opts.dwellSeconds;
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

	/** The focal point a beat frames: the world ORIGIN in centered mode (storm is
	 * pinned there), else the node's laid-out position. */
	private focalPoint(beat: CinemaBeat): THREE.Vector3 | null {
		if (this.opts.centerOnOrigin) return _origin;
		return this.positions.get(beat.nodeId) ?? null;
	}

	/** Compute the camera stand-off position for a beat's focal point, directed by
	 * its shot (move / angle / standoff). With no shot, reproduces the original
	 * framing exactly: standoff = opts.standoff, +0.35 up-bias (filmic tilt). */
	private framePosition(beat: CinemaBeat, index: number, out: THREE.Vector3): THREE.Vector3 {
		const nodePos = this.focalPoint(beat);
		if (!nodePos) {
			// Node has no resolved position yet — keep current framing.
			return out.copy(this.camera.position);
		}
		const shot = this.shotAt(index);

		_tmpDir.copy(this.camera.position).sub(nodePos);
		if (_tmpDir.lengthSq() < 1e-4) _tmpDir.set(0, 0.4, 1);
		_tmpDir.normalize();

		// Vertical bias = the camera angle. Default +0.35 (slightly above, the
		// original filmic tilt). low = look UP at the node (power), high = look
		// DOWN (decay/fading).
		let upBias = 0.35;
		if (shot) {
			if (shot.angle === 'low') upBias = -0.45;
			else if (shot.angle === 'high') upBias = 0.7;
		}
		_tmpDir.addScaledVector(_tmpUp, upBias).normalize();

		// Stand-off = how close: push_in tightens, pull_back/crane widen.
		let standoff = shot?.standoff ?? this.opts.standoff;
		if (shot) {
			if (shot.move === 'push_in') standoff *= 0.7;
			else if (shot.move === 'pull_back') standoff *= 1.5;
			else if (shot.move === 'crane') standoff *= 1.8;
		}
		return out.copy(nodePos).addScaledVector(_tmpDir, standoff);
	}

	private beginFlightTo(index: number): void {
		const beat = this.path.beats[index];
		const nodePos = this.focalPoint(beat);
		const shot = this.shotAt(index);

		this.fromPos.copy(this.camera.position);
		this.fromTarget.copy(this.target);
		this.framePosition(beat, index, this.toPos);
		this.toTarget.copy(nodePos ?? this.target);
		this.phaseElapsed = 0;

		// A directed hard/match cut snaps instantly (like reduced-motion), so the
		// editorial "cut" reads as an edit, not a fly. reduced-motion forces this
		// for every beat regardless of shot.
		const snap = this.opts.reducedMotion || shot?.cut === 'hard_cut' || shot?.cut === 'match_cut';
		if (snap) {
			this.camera.position.copy(this.toPos);
			this.target.copy(this.toTarget);
			this.phase = 'dwelling';
			this.cb.onBeat?.(beat, index, shot);
		} else {
			this.phase = 'flying';
		}
	}

	/** Advance the choreography. Call once per animation frame with delta seconds. */
	update(deltaSeconds: number): void {
		if (this.phase === 'idle' || this.phase === 'done') return;
		// Clamp dt so a tab-switch stall doesn't teleport the camera.
		const dt = Math.max(0, Math.min(deltaSeconds, 0.05));
		this.phaseElapsed += dt;

		const flightSecs = this.flightSecondsAt(this.beatIndex);
		const dwellSecs = this.dwellSecondsAt(this.beatIndex);

		if (this.phase === 'flying') {
			const t = Math.min(1, this.phaseElapsed / flightSecs);
			const e = easeInOutCubic(t);
			this.camera.position.lerpVectors(this.fromPos, this.toPos, e);
			this.target.lerpVectors(this.fromTarget, this.toTarget, e);
			this.applyDutch(this.beatIndex, e);
			if (t >= 1) {
				this.phase = 'dwelling';
				this.phaseElapsed = 0;
				this.cb.onBeat?.(this.path.beats[this.beatIndex], this.beatIndex, this.shotAt(this.beatIndex));
			}
		} else if (this.phase === 'dwelling') {
			if (!this.opts.reducedMotion) {
				const nodePos = this.focalPoint(this.path.beats[this.beatIndex]);
				if (nodePos) {
					this.target.lerp(nodePos, 0.02); // gentle settle keeps the shot alive
					// An orbit shot slowly revolves the camera around the node
					// during the dwell — the signature "reverent" move for keystones.
					if (this.shotAt(this.beatIndex)?.move === 'orbit') {
						this.orbitAround(nodePos, dt * 0.35);
					}
				}
			}
			if (this.phaseElapsed >= dwellSecs) {
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
		// Guard against an empty path (per = 0) so progress can never be NaN.
		const per = this.path.beats.length > 0 ? 1 / this.path.beats.length : 0;
		const intra =
			this.phase === 'flying'
				? Math.min(1, this.phaseElapsed / flightSecs) * 0.5
				: 0.5 + Math.min(1, this.phaseElapsed / dwellSecs) * 0.5;
		this.cb.onProgress?.(Math.min(1, this.beatIndex * per + intra * per));
	}

	/** Revolve the camera around a node by `angle` radians (orbit shots). */
	private orbitAround(center: THREE.Vector3, angle: number): void {
		_tmpDir.copy(this.camera.position).sub(center);
		const cos = Math.cos(angle);
		const sin = Math.sin(angle);
		const x = _tmpDir.x * cos - _tmpDir.z * sin;
		const z = _tmpDir.x * sin + _tmpDir.z * cos;
		_tmpDir.x = x;
		_tmpDir.z = z;
		this.camera.position.copy(center).add(_tmpDir);
	}

	/** Roll the camera (Dutch angle) toward the shot's target roll over the
	 * flight, easing back to upright for non-Dutch shots. */
	private applyDutch(index: number, t: number): void {
		const targetRoll = this.shotAt(index)?.dutch ?? 0;
		const roll = targetRoll * t;
		// camera.up = rotate world-up around the camera's forward axis by `roll`.
		_tmpDir.set(0, 0, -1).applyQuaternion(this.camera.quaternion); // forward
		this.camera.up.set(0, 1, 0).applyAxisAngle(_tmpDir, roll);
	}
}

function easeInOutCubic(t: number): number {
	return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}
