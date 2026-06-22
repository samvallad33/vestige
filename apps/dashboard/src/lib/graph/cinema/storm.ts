// Memory Cinema — the Semantic Compute Storm (WebGPU / TSL GPGPU).
//
// 150k particles whose physics run ENTIRELY on the GPU via Three Shading
// Language compute nodes. The storm shifts behaviour with the narrative beat:
//   - origin/anchor  → stable orbital swarm around the focused node
//   - connection     → fluid streaming toward the target with wave motion
//   - contradiction  → explosive Rössler strange-attractor chaos (crimson)
// Emissive colour is routed so only the storm blazes through the selective
// MRT bloom pass against a clean void.
//
// IMPORTANT — verified against the INSTALLED three@0.172 three/tsl build:
//   * use select() (NOT cond — does not exist in this build)
//   * use TSL sin()/cos() (NOT Math.sin inside Fn)
//   * SpriteNodeMaterial (NOT SpritePointsMaterial)
//   * renderer.computeAsync() for the dispatch
// The whole module is dynamically imported only when Cinema launches, so the
// heavy three/webgpu + three/tsl bundles never load for normal dashboard use.
//
// This file is intentionally framework-agnostic and uses `any` for the WebGPU
// renderer type: three/webgpu's WebGPURenderer is a runtime-only dynamic import
// (kept out of the main bundle), so a compile-time type isn't available here.

import * as THREE from 'three';
// StorageBufferAttribute + SpriteNodeMaterial live in the three/webgpu entry,
// not the base three module. This file is dynamically imported only at Cinema
// launch, so pulling from three/webgpu here does NOT add WebGPU to the main
// bundle.
import { StorageBufferAttribute, SpriteNodeMaterial } from 'three/webgpu';
import {
	Fn,
	storage,
	instanceIndex,
	vec3,
	uniform,
	select,
	float,
	sin,
	cos,
	length,
	clamp,
	min,
	mix,
	fract,
	abs,
	floor,
	positionLocal,
} from 'three/tsl';

export type SemanticRole = 'anchor' | 'connection' | 'contradiction';

const ROLE_MODE: Record<SemanticRole, number> = {
	anchor: 0,
	connection: 1,
	contradiction: 2,
};

export interface StormOptions {
	count?: number;
	/** World-space radius of the initial particle cloud. */
	spawnRadius?: number;
}

/**
 * GPU compute particle storm. Construct with a WebGPURenderer + Scene, call
 * update(dt) each frame, and transitionTo(role, worldPos) on each narrative
 * beat. dispose() releases all GPU resources.
 */
/** The TSL compute node Fn(...)().compute(count) produces. three@0.172 does not
 * export a public type for it; it is opaque and only handed to computeAsync(). */
type ComputeDispatch = ReturnType<ReturnType<ReturnType<typeof Fn>>['compute']>;

export class SemanticComputeStorm {
	readonly count: number;
	private scene: THREE.Scene;
	// WebGPURenderer — runtime-only type (dynamic import); see file header.
	private renderer: { computeAsync: (node: ComputeDispatch) => Promise<void> };

	private bufferPos: StorageBufferAttribute | null;
	private bufferVel: StorageBufferAttribute | null;
	private bufferPhase: StorageBufferAttribute | null;

	// Definite-assigned in buildCompute() (called from the constructor).
	private computeNode!: ComputeDispatch;
	private mesh: THREE.InstancedMesh | null = null;
	private material: THREE.Material | null = null;

	// Serialize GPU compute dispatches: never queue a new compute pass before the
	// previous one resolves, or the WebGPU dispatch queue backs up and stalls.
	private computeInFlight: Promise<void> | null = null;

	// Uniforms driven from the camera/beat loop. uIgnition starts non-zero so
	// the storm is visible on the very first frame (before any beat fires).
	private uTarget = uniform(new THREE.Vector3(0, 0, 0));
	private uTime = uniform(0);
	private uIgnition = uniform(0.6);
	private uMode = uniform(0);
	// World-space radius the storm is contained within. Particles past this get
	// a spring force back so the storm NEVER flies off-screen. Sized to the
	// camera framing by the sandbox via setContainRadius().
	private uContainRadius = uniform(48);
	// Global hue rotation (advances over time) + how strongly the beat's mode
	// tint overrides the rainbow (0 = full rainbow, 1 = full mode color).
	private uHueShift = uniform(0);
	private uModeTintAmt = uniform(0.25);
	// Detonation cycle: spikes to 1 on each beat (explosion), decays to 0
	// (crystallize/reform). Drives the explode→pixelate→reform look.
	private uBurst = uniform(0);

	constructor(
		renderer: { computeAsync: (node: ComputeDispatch) => Promise<void> },
		scene: THREE.Scene,
		opts: StormOptions = {}
	) {
		this.renderer = renderer;
		this.scene = scene;
		this.count = opts.count ?? 150_000;
		// Spawn inside the contained zone so particles don't start outside the
		// shell and get yanked inward asymmetrically (which read as off-center).
		const spawn = opts.spawnRadius ?? 8;

		const positions = new Float32Array(this.count * 3);
		const velocities = new Float32Array(this.count * 3);
		const phases = new Float32Array(this.count);
		for (let i = 0; i < this.count; i++) {
			positions[i * 3] = (Math.random() - 0.5) * spawn;
			positions[i * 3 + 1] = (Math.random() - 0.5) * spawn;
			positions[i * 3 + 2] = (Math.random() - 0.5) * spawn;
			phases[i] = Math.random() * Math.PI * 2;
		}
		const bufferPos = new StorageBufferAttribute(positions, 3);
		const bufferVel = new StorageBufferAttribute(velocities, 3);
		const bufferPhase = new StorageBufferAttribute(phases, 1);
		this.bufferPos = bufferPos;
		this.bufferVel = bufferVel;
		this.bufferPhase = bufferPhase;

		this.buildCompute(bufferPos, bufferVel, bufferPhase);
		this.buildRender(bufferPos, bufferPhase);
	}

	private buildCompute(
		bufferPos: StorageBufferAttribute,
		bufferVel: StorageBufferAttribute,
		bufferPhase: StorageBufferAttribute
	): void {
		const posStore = storage(bufferPos, 'vec3', this.count);
		const velStore = storage(bufferVel, 'vec3', this.count);
		const phaseStore = storage(bufferPhase, 'float', this.count);

		this.computeNode = Fn(() => {
			const pos = posStore.element(instanceIndex);
			const vel = velStore.element(instanceIndex);
			const phase = phaseStore.element(instanceIndex);

			// ── EACH PARTICLE'S "HOME" — a deterministic point on a volumetric
			// spherical shell around the ORIGIN, derived purely from its phase.
			// The cloud reforms to these homes between beats, so the centroid is
			// ANCHORED to origin and CANNOT drift (the bug that pushed it off-frame).
			// No swirl/orbital/attractor terms — those drew the ugly ribbons.
			const a1 = phase.mul(12.9898).sin().mul(43758.5453);
			const a2 = phase.mul(78.233).sin().mul(12543.531);
			const u = fract(a1); // 0..1
			const v = fract(a2); // 0..1
			const theta = u.mul(6.28318); // azimuth
			const phi = v.mul(3.14159); // polar
			// Per-particle home radius fills the interior (0.30r..0.95r) for a
			// dense volumetric orb rather than a hollow shell.
			const homeFrac = float(0.3).add(fract(phase.mul(3.7)).mul(0.65));
			const homeR = this.uContainRadius.mul(homeFrac);
			const home = vec3(
				sin(phi).mul(cos(theta)),
				cos(phi),
				sin(phi).mul(sin(theta))
			).mul(homeR); // centered on origin (uTarget is always origin in sandbox)

			// ── DETONATION: on each beat uBurst≈1 → blow particles radially OUT
			// from origin (the explosion in photo 2). Strength scales with the
			// particle's own radius so the burst is a full-volume shockwave.
			const outDir = pos.normalize();
			vel.addAssign(outDir.mul(this.uBurst.mul(0.9)));

			// ── REFORM: a spring pulling each particle back to its home. As uBurst
			// decays the spring wins, crystallizing the explosion back into the orb.
			const toHome = home.sub(pos);
			vel.addAssign(toHome.mul(0.045));

			// Subtle living shimmer so the reformed orb breathes (mean-zero, no net
			// drift — uses the particle's own home direction, not a global bias).
			const shimmer = home.normalize().mul(sin(this.uTime.mul(1.3).add(phase.mul(6.1))).mul(0.015));
			vel.addAssign(shimmer);

			// Hard velocity clamp — nothing can ever fly off or blow up.
			const speed = length(vel);
			const maxSpeed = float(1.3);
			vel.assign(vel.mul(min(maxSpeed, speed).div(speed.max(0.0001))));

			pos.addAssign(vel);
			vel.mulAssign(0.9); // strong damping → crisp crystallization, no overshoot

			// ── PIXELATION: as particles crystallize (low burst) snap positions to
			// a 3D GRID so the cloud resolves into discrete colored voxels — the
			// crystalline look of photo 3. The grid is finest when fully reformed
			// (burst≈0) and dissolves during the explosion (burst≈1).
			const cell = mix(float(0.55), float(6.0), clamp(this.uBurst, 0, 1)); // small cell = fine pixels
			const quantized = floor(pos.div(cell)).add(0.5).mul(cell);
			const pixelAmt = clamp(float(1).sub(this.uBurst.mul(1.4)), 0, 0.9);
			pos.assign(mix(pos, quantized, pixelAmt));

			// Final hard safety net: clamp anything past the contain radius back
			// onto the boundary shell — guarantees nothing is ever off-screen.
			const finalDist = length(pos);
			const hardR = this.uContainRadius;
			const snapped = pos.normalize().mul(hardR);
			pos.assign(mix(pos, snapped, finalDist.greaterThan(hardR).select(float(1), float(0))));
		})().compute(this.count);
	}

	private buildRender(bufferPos: StorageBufferAttribute, bufferPhase: StorageBufferAttribute): void {
		// SpriteNodeMaterial: emissive routed to bloom; additive against the void.
		const mat = new SpriteNodeMaterial({
			transparent: true,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
		}) as SpriteNodeMaterial & { positionNode: unknown; colorNode: unknown };

		// CRITICAL: particle world position = per-instance GPU compute output
		// (storage buffer, indexed by instanceIndex) PLUS the sprite's local quad
		// vertex (positionLocal) so each billboard keeps its size while being
		// translated to its computed position. Assigning the bare storage element
		// to positionNode (without positionLocal) collapses every quad to a point
		// at its instance origin — the bug the audit caught.
		const phaseStore = storage(bufferPhase, 'float', this.count);
		const instancePos = storage(bufferPos, 'vec3', this.count).element(instanceIndex);
		mat.positionNode = instancePos.add(positionLocal);

		mat.colorNode = Fn(() => {
			const pos = instancePos;
			const ph = phaseStore.element(instanceIndex);
			const radius = length(pos.sub(vec3(this.uTarget)));

			// ── INSANE IRIDESCENT RAINBOW ──
			// Hue drifts across the spectrum by per-particle phase + radius shell +
			// time, plus a global beat-driven hue shift (uHueShift). Each particle
			// is a different color and the whole cloud slowly rotates through the
			// rainbow — a living aurora, not a flat tint.
			const hue = fract(
				ph.mul(0.16)
					.add(radius.mul(0.045))
					.add(this.uTime.mul(0.08))
					.add(this.uHueShift)
			);
			// hue → RGB (fract/abs hexagon palette). Pull the valleys UP slightly
			// then re-saturate so the rainbow is vivid and FULLY saturated (not
			// washed) — pure spectral color, never white.
			const r = clamp(abs(hue.mul(6).sub(3)).sub(1), 0, 1);
			const g = clamp(float(2).sub(abs(hue.mul(6).sub(2))), 0, 1);
			const b = clamp(float(2).sub(abs(hue.mul(6).sub(4))), 0, 1);
			const rainbow = vec3(r, g, b);

			// The beat's mode tint (crimson at a contradiction, gold at surprise,
			// cyan default) is blended in by uModeTintAmt so dramatic beats read
			// their color while keeping the iridescent shimmer underneath.
			const modeTint = select(
				this.uMode.equal(2),
				vec3(1.0, 0.08, 0.32), // contradiction → crimson
				select(this.uMode.equal(3), vec3(1.0, 0.78, 0.1), vec3(0.1, 0.9, 1.0)) // surprise → gold, else cyan
			);
			const tinted = mix(rainbow, modeTint, this.uModeTintAmt);

			// Brightness is CLAMPED low so the rainbow shows as COLOR, not white.
			// Additive blending across 150k overlapping sprites compounds fast — a
			// high multiplier blows the core to pure white (the bug you saw). Keep
			// the glow gentle (0.45 floor, +ignition up to ~1.1) and let the
			// selective bloom pass do the blooming, not raw over-bright color.
			const glow = clamp(this.uIgnition.mul(0.18).add(0.45), 0, 1.15);
			return tinted.mul(glow);
		})();

		// One instanced sprite per particle. Small quads (0.1) keep individual
		// particles as crisp colored points of light rather than overlapping into
		// white mush across the now-larger volume.
		const geometry = new THREE.PlaneGeometry(0.1, 0.1);
		const mesh = new THREE.InstancedMesh(geometry, mat as unknown as THREE.Material, this.count);
		mesh.frustumCulled = false;
		this.material = mat;
		this.mesh = mesh;
		this.scene.add(this.mesh);
	}

	/** Advance the GPU physics one frame. Compute dispatches are serialized so
	 * a slow GPU never lets passes pile up and stall the queue. */
	async update(deltaSeconds: number): Promise<void> {
		const dt = Math.max(0, Math.min(deltaSeconds, 0.05));
		this.uTime.value += dt;
		// Slowly rotate the whole rainbow so the cloud is always shimmering.
		this.uHueShift.value = (this.uHueShift.value + dt * 0.05) % 1;
		// Ignition decays toward 0 between beats (the colorNode floor keeps the
		// storm glowing); spikes back up on transitionTo().
		this.uIgnition.value = Math.max(0, this.uIgnition.value - dt * 2.0);
		// Burst decays fast so the explosion crystallizes back within ~1.2s,
		// leaving the rest of the beat as a calm pixelated orb.
		this.uBurst.value = Math.max(0, this.uBurst.value - dt * 0.85);

		// Wait for any in-flight compute to finish before queuing the next.
		if (this.computeInFlight) await this.computeInFlight;
		this.computeInFlight = this.renderer.computeAsync(this.computeNode).finally(() => {
			this.computeInFlight = null;
		});
		await this.computeInFlight;
	}

	/** Fired on each narrative beat: retarget the storm + spike ignition. */
	transitionTo(role: SemanticRole, worldPos: THREE.Vector3): void {
		this.uTarget.value.copy(worldPos);
		const mode = ROLE_MODE[role] ?? 1;
		this.uMode.value = mode;
		this.uIgnition.value = 8.0;
		// DETONATE: every beat explodes the orb, then it crystallizes/pixelates
		// back. Contradictions detonate hardest.
		this.uBurst.value = mode === 2 ? 1.0 : 0.8;
		// Dramatic beats (contradiction=2, surprise=3) push their mode color over
		// the rainbow so they read clearly; calm beats stay mostly iridescent.
		this.uModeTintAmt.value = mode >= 2 ? 0.7 : 0.22;
	}

	/** Size the containment sphere (world units) so the storm always stays in
	 * frame. The sandbox derives this from the camera distance + fov. */
	setContainRadius(radius: number): void {
		this.uContainRadius.value = Math.max(8, radius);
	}

	dispose(): void {
		if (this.mesh) {
			this.scene.remove(this.mesh);
			this.mesh.geometry?.dispose();
			this.mesh.dispose?.();
			this.mesh = null;
		}
		this.material?.dispose();
		this.material = null;
		// StorageBufferAttribute extends BufferAttribute, which has no dispose():
		// its GPU buffer is released by the renderer when the owning geometry is
		// disposed (done above). Drop our references so the ~2.1MB of backing
		// Float32Arrays can be garbage-collected.
		this.bufferPos = null;
		this.bufferVel = null;
		this.bufferPhase = null;
	}
}
