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
// note: .max()/.div()/.sub() etc. are fluent methods on TSL nodes — no import needed.

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
	private uIgnition = uniform(0.2);
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
	// ACT DIMMER — a master brightness scalar set per beat from the narrative
	// act. Act I opens too hot (the cloud is still in its dense initial spawn and
	// the first ignition flash stacks on top), so we hold Act I dimmer and let
	// Acts II/III blaze at full. 1.0 = full brightness. Starts very low so the
	// pre-first-beat / beat-0 boot frames fade in soft instead of flashing white.
	private uActDim = uniform(0.12);
	// MORPH TARGET — which sculpted form the cloud reforms into. Advances slowly
	// over time and snaps to the next form on each beat, so the storm is forever
	// shape-shifting: sphere → torus → galaxy spiral → cube lattice → wave sheet →
	// (loops). The integer part selects the current form, the fractional part
	// cross-fades into the next, so morphs are fluid, never a hard pop.
	private uShape = uniform(0);
	private readonly shapeCount = 5;

	constructor(
		renderer: { computeAsync: (node: ComputeDispatch) => Promise<void> },
		scene: THREE.Scene,
		opts: StormOptions = {}
	) {
		this.renderer = renderer;
		this.scene = scene;
		this.count = opts.count ?? 150_000;
		// Spawn particles ALREADY SPREAD across a wide spherical SHELL (not a tiny
		// dense ball at the origin). The old ±8 cube packed all 150k into a tiny
		// volume, so the very first frame (Beat 0, before the cloud expands to its
		// rim-falloff homes) was a solid white blob — additive overlap dominates at
		// high density regardless of per-particle dimming. Booting on a broad shell
		// means the storm reads as a calm colored cloud from frame one.
		const spawn = opts.spawnRadius ?? 34;

		const positions = new Float32Array(this.count * 3);
		const velocities = new Float32Array(this.count * 3);
		const phases = new Float32Array(this.count);
		for (let i = 0; i < this.count; i++) {
			// Uniform direction on a sphere, radius biased to the outer shell so the
			// boot cloud is hollow-cored like the rim look (never a dense center).
			const u1 = Math.random();
			const u2 = Math.random();
			const theta = u1 * Math.PI * 2;
			const z = u2 * 2 - 1;
			const r = Math.sqrt(Math.max(0, 1 - z * z));
			const rad = spawn * (0.55 + Math.random() * 0.45); // shell 0.55..1.0
			positions[i * 3] = Math.cos(theta) * r * rad;
			positions[i * 3 + 1] = z * rad;
			positions[i * 3 + 2] = Math.sin(theta) * r * rad;
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

			// ── EACH PARTICLE'S "HOME" — a deterministic point on a SCULPTED FORM
			// around the ORIGIN, derived purely from its phase. The cloud reforms to
			// these homes between beats, so the centroid is ANCHORED to origin and
			// CANNOT drift. But instead of always a sphere, the home MORPHS through a
			// gallery of trippy forms (uShape) — the storm is forever shape-shifting.
			const a1 = phase.mul(12.9898).sin().mul(43758.5453);
			const a2 = phase.mul(78.233).sin().mul(12543.531);
			const a3 = phase.mul(39.346).sin().mul(24634.633);
			const u = fract(a1); // 0..1
			const v = fract(a2); // 0..1
			const w2 = fract(a3); // 0..1
			const theta = u.mul(6.28318); // azimuth 0..2π
			const phi = v.mul(3.14159); // polar 0..π
			const R = this.uContainRadius;
			// Per-particle radial fill biased to the OUTER shell (0.62..1.0) so
			// particles spread across the surface of each form and read as distinct
			// COLOR, instead of piling into a dense central core that additive-blooms
			// to white. Squaring pushes even more mass outward. This is what finally
			// kills the white center while keeping the volumetric feel.
			const shellT = fract(phase.mul(3.7));
			const homeFrac = float(0.62).add(shellT.mul(shellT).mul(0.38));

			// ── FORM 0 · SPHERE (volumetric orb) ──
			const sphere = vec3(
				sin(phi).mul(cos(theta)),
				cos(phi),
				sin(phi).mul(sin(theta))
			).mul(R.mul(homeFrac));

			// ── FORM 1 · TORUS (donut, ring radius 0.7R, tube 0.28R) ──
			const tubeR = R.mul(0.28).mul(float(0.5).add(w2.mul(0.5)));
			const ringR = R.mul(0.7);
			const torus = vec3(
				ringR.add(tubeR.mul(cos(phi.mul(2)))).mul(cos(theta)),
				tubeR.mul(sin(phi.mul(2))),
				ringR.add(tubeR.mul(cos(phi.mul(2)))).mul(sin(theta))
			);

			// ── FORM 2 · GALAXY SPIRAL (flat logarithmic spiral disc, 2 arms) ──
			const arm = u.mul(6.28318).mul(3).add(w2.mul(0.6)); // winding
			const gr = R.mul(0.2).add(R.mul(0.8).mul(w2));
			const galaxy = vec3(
				gr.mul(cos(arm)),
				R.mul(0.06).mul(sin(phase.mul(20))), // thin disc with slight z jitter
				gr.mul(sin(arm))
			);

			// ── FORM 3 · CUBE LATTICE (particles snapped onto a glowing box grid) ──
			const cube = vec3(
				u.sub(0.5).mul(2).mul(R.mul(0.85)),
				v.sub(0.5).mul(2).mul(R.mul(0.85)),
				w2.sub(0.5).mul(2).mul(R.mul(0.85))
			);

			// ── FORM 4 · WAVE SHEET (rippling plane, sinusoidal height field) ──
			const sx = u.sub(0.5).mul(2).mul(R.mul(0.95));
			const sz = v.sub(0.5).mul(2).mul(R.mul(0.95));
			const wave = vec3(
				sx,
				sin(sx.mul(0.35).add(this.uTime.mul(1.2)))
					.add(cos(sz.mul(0.35).sub(this.uTime)))
					.mul(R.mul(0.22)),
				sz
			);

			// ── MORPH BLEND ── integer part of uShape picks the current form, the
			// fractional part cross-fades into the next, so the cloud fluidly melts
			// from one sculpture to the next. select()-chain because the build has no
			// dynamic array indexing in TSL.
			const sIdx = floor(this.uShape);
			const sFrac = fract(this.uShape);
			const formA = select(
				sIdx.equal(0), sphere,
				select(sIdx.equal(1), torus,
				select(sIdx.equal(2), galaxy,
				select(sIdx.equal(3), cube, wave)))
			);
			const formB = select(
				sIdx.equal(0), torus,
				select(sIdx.equal(1), galaxy,
				select(sIdx.equal(2), cube,
				select(sIdx.equal(3), wave, sphere)))
			);
			// smoothstep-ish ease on the cross-fade for a silky morph.
			const ease = sFrac.mul(sFrac).mul(float(3).sub(sFrac.mul(2)));
			const home = mix(formA, formB, ease); // centered on origin

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
		}) as SpriteNodeMaterial & {
			positionNode: unknown;
			colorNode: unknown;
			emissiveNode: unknown;
		};

		// CRITICAL: particle world position = per-instance GPU compute output
		// (storage buffer, indexed by instanceIndex) PLUS the sprite's local quad
		// vertex (positionLocal) so each billboard keeps its size while being
		// translated to its computed position. Assigning the bare storage element
		// to positionNode (without positionLocal) collapses every quad to a point
		// at its instance origin — the bug the audit caught.
		const phaseStore = storage(bufferPhase, 'float', this.count);
		const instancePos = storage(bufferPos, 'vec3', this.count).element(instanceIndex);
		mat.positionNode = instancePos.add(positionLocal);

		// ── SHARED RAINBOW COLOR ──
		// One Fn produces the pure iridescent color for a particle; we feed it to
		// BOTH colorNode (the lit/additive surface color) AND emissiveNode (the
		// channel the selective MRT bloom reads). The original code only set
		// colorNode, so the bloom had NO color to bloom — it washed the frame to
		// white. Routing the SAME rainbow to emissive makes the bloom glow in full
		// spectral color, which is the whole point.
		const rainbowColor = Fn(() => {
			const pos = instancePos;
			const ph = phaseStore.element(instanceIndex);
			const radius = length(pos.sub(vec3(this.uTarget)));

			// Hue from many decorrelated terms so the whole spectrum is present at
			// once and forever swirling: per-particle phase, concentric radial
			// shells, a spatial XYZ band (gives morphing forms internal rainbow
			// striping), time, and a global beat hue-shift.
			const spatialBand = pos.x.mul(0.03).add(pos.y.mul(0.021)).add(pos.z.mul(0.027));
			const hue = fract(
				ph.mul(0.41)
					.add(radius.mul(0.06))
					.add(spatialBand)
					.add(this.uTime.mul(0.10))
					.add(this.uHueShift)
			);
			// hue → RGB at FULL saturation (HSV S=1,V=1) hexagon ramps. Pure jewel
			// tone per particle, never desaturated toward luma.
			const r0 = clamp(abs(hue.mul(6).sub(3)).sub(1), 0, 1);
			const g0 = clamp(float(2).sub(abs(hue.mul(6).sub(2))), 0, 1);
			const b0 = clamp(float(2).sub(abs(hue.mul(6).sub(4))), 0, 1);
			const rainbow = vec3(r0, g0, b0);

			// Beat mode tint (crimson contradiction / gold surprise / cyan default)
			// blended by uModeTintAmt so dramatic beats read their color.
			const modeTint = select(
				this.uMode.equal(2),
				vec3(1.0, 0.08, 0.32),
				select(this.uMode.equal(3), vec3(1.0, 0.78, 0.1), vec3(0.1, 0.9, 1.0))
			);
			return mix(rainbow, modeTint, this.uModeTintAmt);
		});

		// ── RIM GLOW ── THE look: bright glowing EDGES, dim center.
		// The dense middle of each form (particles near the center axis, all
		// stacking toward the camera) is what blooms to white. So we DIM the core
		// and BLAZE the rim: brightness rises with a particle's radial distance
		// from the form's center. Near center → ~0.12 (deep, calm), at the outer
		// shell → ~1.0 (full blaze). The result is the glowing-shell / hollow-eye
		// torus look — luminous silhouette, serene dark center.
		const rimFactor = Fn(() => {
			const pos = instancePos;
			// Normalized radial position 0 (center) .. 1 (contain radius).
			const rNorm = clamp(length(pos).div(this.uContainRadius.max(0.0001)), 0, 1);
			// Smooth ramp: dark core, bright rim. pow-like curve via rNorm² pushes
			// the brightness toward the outer shell so the edge reads as a crisp
			// glowing rind and the interior falls away into shadow.
			const edge = rNorm.mul(rNorm);
			return float(0.12).add(edge.mul(0.95)); // 0.12 core → ~1.07 rim
		});

		// colorNode: surface color × rim falloff × act dimmer. Moderate base so
		// additive overlap blends hues (kaleidoscope) rather than summing to white.
		mat.colorNode = Fn(() => {
			const glow = clamp(this.uIgnition.mul(0.05).add(0.5), 0, 1.0);
			return rainbowColor().mul(glow).mul(rimFactor()).mul(this.uActDim);
		})();

		// emissiveNode: what the selective bloom reads — THE glow channel. The rim
		// factor means ONLY the outer shell feeds the bloom hard, so the storm
		// haloes as a luminous spectral RING/SHELL with a calm dark center, instead
		// of a solid white blob. Modest base gain keeps overlapping hues blending
		// to new colors, never clipping past white.
		mat.emissiveNode = Fn(() => {
			const emGain = clamp(this.uIgnition.mul(0.04).add(0.6), 0, 1.1);
			return rainbowColor().mul(emGain).mul(rimFactor()).mul(this.uActDim);
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
		this.uHueShift.value = (this.uHueShift.value + dt * 0.06) % 1;
		// Drift the morph target forward continuously so the cloud is ALWAYS
		// melting toward the next sculpted form (even mid-beat). Beats snap it to
		// the next whole form for a dramatic transform; this slow drift keeps it
		// alive between beats. Wraps around the gallery.
		this.uShape.value = (this.uShape.value + dt * 0.09) % this.shapeCount;
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

	/** Fired on each narrative beat: retarget the storm + spike ignition.
	 * `act` blazes Acts II/III at full; `beatIndex` (0-based) holds the very first
	 * beats EXTRA dim — beats 0 and 1 fire while the cloud is still bunched from
	 * the initial reform and would otherwise wash to white. They ramp up to full
	 * over the opening, so the storm fades IN beautifully instead of flashing. */
	transitionTo(
		role: SemanticRole,
		worldPos: THREE.Vector3,
		act: 'I' | 'II' | 'III' = 'II',
		beatIndex = 99
	): void {
		this.uTarget.value.copy(worldPos);
		const mode = ROLE_MODE[role] ?? 1;
		this.uMode.value = mode;
		// Per-beat warm-up dim: beat 0 ≈0.12, beat 1 ≈0.22, beat 2 ≈0.45, then it
		// hands off to the act-based brightness. This specifically tames beats 0/1
		// (the only ones still washing out) without dimming the rest of Act I.
		const warmup =
			beatIndex === 0 ? 0.12 : beatIndex === 1 ? 0.2 : null;
		const actDim = act === 'I' ? 0.26 : 1.0;
		this.uActDim.value = warmup ?? actDim;
		// Ignition flash: nearly none on beats 0/1 (no punch while bunched), gentle
		// for the rest of Act I, full blaze for Acts II/III.
		this.uIgnition.value = beatIndex <= 1 ? 0.4 : act === 'I' ? 1.6 : 8.0;
		// DETONATE: every beat explodes the orb, then it crystallizes/pixelates
		// back. Contradictions detonate hardest.
		this.uBurst.value = mode === 2 ? 1.0 : 0.8;
		// MORPH: snap the cloud onward to the NEXT sculpted form on this beat, so
		// every narrative beat transforms the geometry (sphere→torus→galaxy→cube→
		// wave→…). Round up to the next whole index, then wrap.
		this.uShape.value = (Math.floor(this.uShape.value) + 1) % this.shapeCount;
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
