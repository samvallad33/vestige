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
export class SemanticComputeStorm {
	readonly count: number;
	private scene: THREE.Scene;
	// WebGPURenderer — runtime-only type (dynamic import); see file header.
	private renderer: { computeAsync: (node: unknown) => Promise<void> };

	private bufferPos: StorageBufferAttribute;
	private bufferVel: StorageBufferAttribute;
	private bufferPhase: StorageBufferAttribute;

	private computeNode: unknown;
	private mesh!: THREE.Object3D;
	private material!: THREE.Material;

	// Uniforms driven from the camera/beat loop.
	private uTarget = uniform(new THREE.Vector3(0, 0, 0));
	private uTime = uniform(0);
	private uIgnition = uniform(0);
	private uMode = uniform(0);

	constructor(
		renderer: { computeAsync: (node: unknown) => Promise<void> },
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
		this.bufferPos = new StorageBufferAttribute(positions, 3);
		this.bufferVel = new StorageBufferAttribute(velocities, 3);
		this.bufferPhase = new StorageBufferAttribute(phases, 1);

		this.buildCompute();
		this.buildRender();
	}

	private buildCompute(): void {
		const posStore = storage(this.bufferPos, 'vec3', this.count);
		const velStore = storage(this.bufferVel, 'vec3', this.count);
		const phaseStore = storage(this.bufferPhase, 'float', this.count);

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
			pos.addAssign(vel);
			vel.mulAssign(0.95);
		})().compute(this.count);
	}

	private buildRender(): void {
		// SpriteNodeMaterial: emissive routed to bloom; additive against the void.
		const mat = new SpriteNodeMaterial({
			transparent: true,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
		}) as SpriteNodeMaterial & { positionNode: unknown; colorNode: unknown };

		mat.positionNode = storage(this.bufferPos, 'vec3', this.count).element(instanceIndex);

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
			return base.mul(this.uIgnition.mul(3.0).add(0.4));
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

	/** Advance the GPU physics one frame. */
	async update(deltaSeconds: number): Promise<void> {
		this.uTime.value += deltaSeconds;
		if (this.uIgnition.value > 0) {
			this.uIgnition.value = Math.max(0, this.uIgnition.value - deltaSeconds * 2.0);
		}
		await this.renderer.computeAsync(this.computeNode);
	}

	/** Fired on each narrative beat: retarget the storm + spike ignition. */
	transitionTo(role: SemanticRole, worldPos: THREE.Vector3): void {
		this.uTarget.value.copy(worldPos);
		this.uMode.value = ROLE_MODE[role] ?? 1;
		this.uIgnition.value = 8.0;
	}

	dispose(): void {
		this.scene.remove(this.mesh);
		(this.mesh as THREE.InstancedMesh).geometry?.dispose();
		this.material?.dispose();
	}
}
