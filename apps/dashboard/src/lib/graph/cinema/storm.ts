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

	constructor(
		renderer: { computeAsync: (node: ComputeDispatch) => Promise<void> },
		scene: THREE.Scene,
		opts: StormOptions = {}
	) {
		this.renderer = renderer;
		this.scene = scene;
		this.count = opts.count ?? 150_000;
		const spawn = opts.spawnRadius ?? 15;

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
		this.buildRender(bufferPos);
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

			const toTarget = vec3(this.uTarget).sub(pos);

			// Mode 0 — Anchor: orbital swirl (tangential velocity around target).
			const orbital = vec3(toTarget.z, float(0), toTarget.x.negate())
				.normalize()
				.mul(0.05);

			// Mode 1 — Connection: stream toward target + per-particle wave.
			const wave = vec3(
				sin(this.uTime.add(phase)).mul(0.02),
				cos(this.uTime.add(phase)).mul(0.02),
				sin(this.uTime.mul(1.5).add(phase)).mul(0.02)
			);
			const stream = toTarget.normalize().mul(0.08).add(wave);

			// Mode 2 — Contradiction: Rössler strange-attractor chaos.
			const dt = float(0.01);
			const dx = vel.y.negate().sub(vel.z).mul(dt);
			const dy = vel.x.add(vel.y.mul(0.2)).mul(dt);
			const dz = float(0.2).add(vel.z.mul(vel.x.sub(5.7))).mul(dt);
			const chaos = vec3(dx, dy, dz).mul(2.0);

			// Runtime mode selection (select(), not cond()).
			const active = select(
				this.uMode.equal(0),
				orbital,
				select(this.uMode.equal(1), stream, chaos)
			);

			vel.addAssign(active);
			// Ignition shockwave yanks particles toward the new node on each beat.
			vel.addAssign(toTarget.normalize().mul(this.uIgnition.mul(0.02)));

			// CONTAINMENT: soft spherical boundary around the focused target so the
			// storm NEVER escapes the camera frame. Past uContainRadius a spring
			// force pulls each particle back toward the target; the force ramps in
			// smoothly (smoothstep-like) so the boundary reads as a glowing
			// membrane, not a hard wall. The chaos attractor (Rössler) is
			// unbounded by nature — this is what keeps mode 2 on-screen.
			const distFromTarget = length(toTarget.negate()); // |pos - target|
			const overflow = distFromTarget.sub(this.uContainRadius).max(0);
			const pullBack = clamp(overflow.mul(0.012), 0, 0.6);
			vel.addAssign(toTarget.normalize().mul(pullBack));

			// Hard velocity clamp so no single step can shoot a particle across
			// the frame even at peak ignition / chaos divergence.
			const speed = length(vel);
			const maxSpeed = float(1.2);
			vel.assign(vel.mul(min(maxSpeed, speed).div(speed.max(0.0001))));

			pos.addAssign(vel);
			vel.mulAssign(0.95);

			// Final safety net: if a particle still ends up beyond 1.35x the
			// radius (extreme edge case), snap it onto the boundary shell so it
			// can never be lost off-screen.
			const finalToTarget = vec3(this.uTarget).sub(pos);
			const finalDist = length(finalToTarget.negate());
			const hardR = this.uContainRadius.mul(1.35);
			const snapped = vec3(this.uTarget).sub(finalToTarget.normalize().mul(hardR));
			pos.assign(mix(pos, snapped, finalDist.greaterThan(hardR).select(float(1), float(0))));
		})().compute(this.count);
	}

	private buildRender(bufferPos: StorageBufferAttribute): void {
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
		const instancePos = storage(bufferPos, 'vec3', this.count).element(instanceIndex);
		mat.positionNode = instancePos.add(positionLocal);

		mat.colorNode = Fn(() => {
			const anchor = vec3(0.0, 1.0, 0.85); // luminescent cyan
			const link = vec3(0.2, 0.4, 1.0); // electric royal blue
			const contradiction = vec3(1.0, 0.1, 0.3); // crimson neon
			const base = select(
				this.uMode.equal(0),
				anchor,
				select(this.uMode.equal(1), link, contradiction)
			);
			// Brighten on ignition so the beat blazes through the bloom pass.
			// The +0.55 floor keeps particles visibly glowing between beats so the
			// storm never fades to black once ignition decays.
			return base.mul(this.uIgnition.mul(3.0).add(0.55));
		})();

		// One instanced sprite per particle; positions come from the GPU storage
		// buffer via positionNode, so the geometry is a single unit quad and the
		// instance count is the particle count.
		const geometry = new THREE.PlaneGeometry(0.18, 0.18);
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
		// Ignition decays toward 0 between beats (the colorNode floor keeps the
		// storm glowing); spikes back up on transitionTo().
		this.uIgnition.value = Math.max(0, this.uIgnition.value - dt * 2.0);

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
		this.uMode.value = ROLE_MODE[role] ?? 1;
		this.uIgnition.value = 8.0;
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
