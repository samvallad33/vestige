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
	smoothstep,
	oneMinus,
	cross,
	sqrt,
	pow,
	mx_noise_vec3,
} from 'three/tsl';
// note: .max()/.div()/.sub()/.cos()/.sin()/.log()/.lessThanEqual() etc. are
// fluent methods on TSL nodes — no import needed.

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
	// WORLD STATE MACHINE — each narrative beat (1..7) is a UNIQUE visual world:
	//   0 nebula mist · 1 orbital anchor · 2 strange attractor · 3 detonation void
	//   4 crystal lattice · 5 fluid galaxy · 6 phyllotaxis bloom
	// Beats map 1:1 to worlds (beatIndex % 7). The compute kernel builds all 7
	// home targets + forces and select()s the live one — particles are never
	// swapped, only the forces acting on them, which IS the journey.
	private uWorld = uniform(0);
	private uPrevWorld = uniform(0);
	// Crossfade prev→current world over ~1s after each beat (eased in update()).
	// 1 = fully previous world, 0 = fully current.
	private uBlend = uniform(0);
	private readonly worldCount = 7;
	// COLOR BLAST — a LONG-LIVED chroma envelope, decoupled from the fast physics
	// burst so the detonation color OUTLIVES the shockwave (owner: "color too
	// brief"). uBlast is the 0..1 magnitude (slow ~2.8s decay); uBlastTime counts
	// seconds since the last detonation and drives the outward spectral wave.
	private uBlast = uniform(0);
	private uBlastTime = uniform(0);
	// ENDLESS DREAM MODE — after the scripted 7-beat tour, the storm keeps
	// generating crazier figures forever instead of sitting idle. uMorphSeed
	// randomizes each procedural figure (worlds 7..11); uChaos ramps 0→1 over the
	// dream so every figure is wilder than the last.
	private uMorphSeed = uniform(0);
	private uChaos = uniform(0);
	// JS-side dream state (not uniforms): which figure is live + how many fired.
	private dreamCount = 0;

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

			// ── DETERMINISTIC PER-PARTICLE BASIS (phase → stable spherical coords) ──
			const a1 = phase.mul(12.9898).sin().mul(43758.5453);
			const a2 = phase.mul(78.233).sin().mul(12543.531);
			const a3 = phase.mul(39.346).sin().mul(24634.633);
			const u = fract(a1); // 0..1
			const v = fract(a2); // 0..1
			const w2 = fract(a3); // 0..1
			const theta = u.mul(6.28318); // azimuth 0..2π
			const phi = v.mul(3.14159); // polar 0..π
			const R = this.uContainRadius;
			// Outer-shell bias (0.62..1.0) keeps the core hollow → reads as color,
			// not a white-blooming dense center. (The dialed-in anti-white-out.)
			const shellT = fract(phase.mul(3.7));
			const homeFrac = float(0.62).add(shellT.mul(shellT).mul(0.38));
			const fi = float(instanceIndex); // particle index as float (phyllotaxis)

			// ── CURL NOISE (divergence-free flow → worlds 0 nebula, 5 fluid) ──
			// Never clumps, never stops; the signature "living smoke" motion.
			const curl = Fn(([p]: [ReturnType<typeof vec3>]) => {
				const e = float(0.6);
				const dx = mx_noise_vec3(p.add(vec3(e, 0, 0))).sub(mx_noise_vec3(p.sub(vec3(e, 0, 0))));
				const dy = mx_noise_vec3(p.add(vec3(0, e, 0))).sub(mx_noise_vec3(p.sub(vec3(0, e, 0))));
				const dz = mx_noise_vec3(p.add(vec3(0, 0, e))).sub(mx_noise_vec3(p.sub(vec3(0, 0, e))));
				return vec3(dy.z.sub(dz.y), dz.x.sub(dx.z), dx.y.sub(dy.x)).normalize();
			});

			// ── 7 WORLD HOME TARGETS (all centered on origin → centroid can't drift) ──
			const sphereShell = vec3(sin(phi).mul(cos(theta)), cos(phi), sin(phi).mul(sin(theta)));
			const wNebula = sphereShell.mul(R.mul(homeFrac)); // world 0 (and 3 base)
			const wAnchor = sphereShell.mul(R.mul(float(0.5).add(shellT.mul(0.3)))); // world 1
			// world 2 attractor: home is "ahead" along the Thomas flow from current pos.
			const bT = float(0.19);
			const thomas = vec3(
				sin(pos.y).sub(pos.x.mul(bT)),
				sin(pos.z).sub(pos.y.mul(bT)),
				sin(pos.x).sub(pos.z.mul(bT))
			);
			const wAttractor = pos.add(thomas.mul(R.mul(0.12)));
			const wVoid = wNebula; // world 3 = sphere; the burst dominates this beat
			const wCrystal = vec3( // world 4 cube lattice
				u.sub(0.5).mul(2).mul(R.mul(0.8)),
				v.sub(0.5).mul(2).mul(R.mul(0.8)),
				w2.sub(0.5).mul(2).mul(R.mul(0.8))
			);
			const armAng = u.mul(6.28318).mul(3).add(w2.mul(0.6)); // world 5 galaxy spiral
			const gr = R.mul(0.2).add(R.mul(0.8).mul(w2));
			const wGalaxy = vec3(
				gr.mul(cos(armAng)),
				R.mul(0.06).mul(sin(phase.mul(20))),
				gr.mul(sin(armAng))
			);
			const golden = float(2.39996323); // world 6 phyllotaxis (Vogel sunflower)
			const pAng = fi.mul(golden);
			const pRad = sqrt(fi).mul(R.mul(0.0042)); // ~R at 150k particles
			const wPhyllo = vec3(pAng.cos().mul(pRad), R.mul(0.04).mul(sin(phase.mul(9))), pAng.sin().mul(pRad));

			// ══════════════════════════════════════════════════════════════════
			//  ENDLESS DREAM FIGURES (worlds 7..11) — the generative mode that
			//  kicks in after the scripted 7-beat tour. These are PROCEDURAL and
			//  RANDOMIZED: uMorphSeed (set per auto-beat) + uChaos (ramps up over
			//  time → each figure crazier than the last) modulate the parameters,
			//  so the same world index never looks the same twice.
			// ══════════════════════════════════════════════════════════════════
			const seed = this.uMorphSeed;
			const chaos = this.uChaos;
			// seeded per-figure scalars (deterministic hash of the seed)
			const s1 = fract(seed.mul(0.731).add(0.13));
			const s2 = fract(seed.mul(1.323).add(0.51));
			const s3 = fract(seed.mul(2.117).add(0.27));

			// world 7 · SUPERSHAPE (3D superformula — petals/stars/blobs, never same)
			const m1 = float(2).add(floor(s1.mul(14))); // symmetry 2..15
			const sfAng = theta;
			const sfR1 = pow(abs(cos(m1.mul(sfAng).div(4))), float(2).add(s2.mul(8)))
				.add(pow(abs(sin(m1.mul(sfAng).div(4))), float(2).add(s3.mul(8))))
				.add(0.0001)
				.pow(float(-0.5));
			const sfR2 = pow(abs(cos(m1.mul(phi).div(4))), float(3))
				.add(pow(abs(sin(m1.mul(phi).div(4))), float(3)))
				.add(0.0001)
				.pow(float(-0.5));
			const sfRad = R.mul(0.85).mul(clamp(sfR1.mul(sfR2).mul(0.5), 0.1, 1.4));
			const wSuper = vec3(
				sin(phi).mul(cos(theta)).mul(sfRad),
				cos(phi).mul(sfRad),
				sin(phi).mul(sin(theta)).mul(sfRad)
			);

			// world 8 · TORUS KNOT (p,q knot — randomized winding, hypnotic ribbons)
			const pKnot = float(2).add(floor(s1.mul(5))); // 2..6
			const qKnot = float(3).add(floor(s2.mul(5))); // 3..7
			const kt = fi.mul(0.0006).add(this.uTime.mul(0.1));
			const kr = cos(qKnot.mul(kt)).mul(0.4).add(1);
			const wKnot = vec3(
				kr.mul(cos(pKnot.mul(kt))),
				kr.mul(sin(pKnot.mul(kt))),
				sin(qKnot.mul(kt)).mul(0.55)
			).mul(R.mul(0.6)).add(sphereShell.mul(R.mul(0.06))); // slight fuzz

			// world 9 · WARPED LISSAJOUS LATTICE (3D sine-wave interference web)
			const fx = float(2).add(floor(s1.mul(5)));
			const fy = float(2).add(floor(s2.mul(5)));
			const fz = float(2).add(floor(s3.mul(5)));
			const lt = fi.mul(0.0007);
			const wLissa = vec3(
				sin(fx.mul(lt).add(this.uTime.mul(0.3))),
				sin(fy.mul(lt).add(1.7)),
				sin(fz.mul(lt).add(this.uTime.mul(0.2)).add(3.1))
			).mul(R.mul(0.82));

			// world 10 · HELIX STORM (twisted DNA-ish double helix that writhes)
			const hAng = fi.mul(0.0009).add(this.uTime.mul(0.4));
			const hSide = select(fract(phase.mul(2)).greaterThan(0.5), float(1), float(-1));
			const hRad = R.mul(0.55).mul(float(0.7).add(sin(hAng.mul(3)).mul(0.3).mul(chaos.add(0.3))));
			const wHelix = vec3(
				cos(hAng).mul(hRad).mul(hSide),
				fi.mul(0.00026).sub(R.mul(0.9)).mul(0.5).add(sin(this.uTime).mul(R.mul(0.1))),
				sin(hAng).mul(hRad).mul(hSide)
			);

			// world 11 · QUANTUM FOAM (curl-warped noisy blob — pure chaos, max wild)
			const foam = mx_noise_vec3(sphereShell.mul(float(1.5).add(chaos.mul(3))).add(seed)).mul(R.mul(0.5).mul(chaos.add(0.4)));
			const wFoam = sphereShell.mul(R.mul(homeFrac)).add(foam);

			// select() chain — no dynamic indexing in this TSL build.
			const homeFor = (idx: ReturnType<typeof float>) =>
				select(idx.equal(0), wNebula,
				select(idx.equal(1), wAnchor,
				select(idx.equal(2), wAttractor,
				select(idx.equal(3), wVoid,
				select(idx.equal(4), wCrystal,
				select(idx.equal(5), wGalaxy,
				select(idx.equal(6), wPhyllo,
				select(idx.equal(7), wSuper,
				select(idx.equal(8), wKnot,
				select(idx.equal(9), wLissa,
				select(idx.equal(10), wHelix, wFoam)))))))))));
			const homeCur = homeFor(float(this.uWorld));
			const homePrev = homeFor(float(this.uPrevWorld));
			// uBlend eases prev→cur (smoothstep) so the world morph is silky.
			const blendE = smoothstep(float(0), float(1), oneMinus(this.uBlend));
			const home = mix(homePrev, homeCur, blendE);

			// ── DETONATION: per-particle staggered radial blast so it blooms as a
			// shockwave, not all-at-once. uBurst spikes on each beat, decays fast.
			const outDir = pos.normalize();
			const stagger = oneMinus(fract(phase.mul(7.3)).mul(0.4));
			vel.addAssign(outDir.mul(this.uBurst.mul(0.95).mul(stagger)));

			// ── REFORM SPRING toward the (blended) world home ──
			vel.addAssign(home.sub(pos).mul(0.045));

			// ── PER-WORLD MOTION MODIFIERS (added to the spring) ──
			// worlds 0 & 5: curl turbulence (living mist / liquid arms)
			const curlV = curl(pos.mul(0.045).add(vec3(0, this.uTime.mul(0.2), 0)));
			const curlAmt = select(this.uWorld.equal(0), float(0.05),
				select(this.uWorld.equal(5), float(0.06), float(0.0)));
			vel.addAssign(curlV.mul(curlAmt));
			// world 1: orbital spin around Y (cross product → orbit, not collapse)
			vel.addAssign(cross(vec3(0, 1, 0), pos).mul(0.0009).mul(select(this.uWorld.equal(1), float(1), float(0))));
			// world 2: integrate the Thomas attractor (chaos lattice)
			vel.addAssign(thomas.mul(0.012).mul(select(this.uWorld.equal(2), float(1), float(0))));
			// world 5: tangential swirl for liquid galaxy arms
			vel.addAssign(cross(vec3(0, 1, 0), pos).mul(0.0016).mul(select(this.uWorld.equal(5), float(1), float(0))));

			// Subtle living shimmer (mean-zero, no net drift).
			const shimmer = home.normalize().mul(sin(this.uTime.mul(1.3).add(phase.mul(6.1))).mul(0.015));
			vel.addAssign(shimmer);

			// Hard velocity clamp — nothing can ever fly off or blow up.
			const speed = length(vel);
			const maxSpeed = float(1.3);
			vel.assign(vel.mul(min(maxSpeed, speed).div(speed.max(0.0001))));

			pos.addAssign(vel);
			vel.mulAssign(0.9); // strong damping → crisp crystallization, no overshoot

			// ── PIXELATION: voxel snap as particles crystallize (low burst). World 4
			// (crystal lattice) pushes it hardest for the holographic shard look.
			const crystalBoost = select(this.uWorld.equal(4), float(1.6), float(1.0));
			const cell = mix(float(0.55), float(6.0), clamp(this.uBurst, 0, 1));
			const quantized = floor(pos.div(cell)).add(0.5).mul(cell);
			const pixelAmt = clamp(oneMinus(this.uBurst.mul(1.4)), 0, 0.9).mul(crystalBoost).min(0.9);
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

		// ── IQ COSINE PALETTE ── one scalar t → smooth, vivid, loopable color.
		// color(t) = a + b·cos(2π·(c·t + d)). The workhorse for per-world palettes
		// and the spectral dispersion wave.
		const palette = Fn(
			([t, a, b, c, d]: [
				ReturnType<typeof float>,
				ReturnType<typeof vec3>,
				ReturnType<typeof vec3>,
				ReturnType<typeof vec3>,
				ReturnType<typeof vec3>,
			]) => a.add(b.mul(cos(c.mul(t).add(d).mul(6.28318))))
		);

		// ── BLACKBODY K→RGB ── real plasma-cooling color (Tanner-Helland approx).
		// Drives the detonation: blue-white core (hot) cooling to red embers as the
		// blast decays. if/else collapsed to select() for this TSL build.
		const blackbody = Fn(([kelvin]: [ReturnType<typeof float>]) => {
			const k = kelvin.div(100.0);
			const rHot = pow(k.sub(60.0).max(0.0001), float(-0.1332047592)).mul(329.698727446);
			const r = k.lessThanEqual(66.0).select(float(255.0), rHot);
			const gCool = k.max(0.0001).log().mul(99.4708025861).sub(161.1195681661);
			const gHot = pow(k.sub(60.0).max(0.0001), float(-0.0755148492)).mul(288.1221695283);
			const g = k.lessThanEqual(66.0).select(gCool, gHot);
			const bMid = k.sub(10.0).max(0.0001).log().mul(138.5177312231).sub(305.0447927307);
			const b = k.greaterThanEqual(66.0).select(
				float(255.0),
				k.lessThanEqual(19.0).select(float(0.0), bMid)
			);
			return clamp(vec3(r, g, b).div(255.0), 0, 1);
		});

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
			// tone per particle — the universal base spectrum.
			const r0 = clamp(abs(hue.mul(6).sub(3)).sub(1), 0, 1);
			const g0 = clamp(float(2).sub(abs(hue.mul(6).sub(2))), 0, 1);
			const b0 = clamp(float(2).sub(abs(hue.mul(6).sub(4))), 0, 1);
			const baseRainbow = vec3(r0, g0, b0);

			// ── PER-WORLD PALETTE ── each world gets its own cosine-palette identity
			// so the journey reads as distinct PLACES: nebula teal, anchor gold,
			// crystal foil-blue, galaxy magenta→cyan, phyllo/attractor full rainbow.
			const dWorld = select(this.uWorld.equal(0), vec3(0.55, 0.6, 0.7), // nebula teal/indigo
				select(this.uWorld.equal(1), vec3(0.05, 0.12, 0.2), // anchor gold/amber
				select(this.uWorld.equal(4), vec3(0.0, 0.25, 0.5), // crystal foil
				select(this.uWorld.equal(5), vec3(0.8, 0.0, 0.33), // galaxy magenta→cyan
				vec3(0.0, 0.33, 0.67))))); // attractor/void/phyllo → full spectrum
			const cWorld = select(this.uWorld.equal(5), vec3(2.0), vec3(1.0)); // galaxy = tighter banding
			const worldPal = palette(hue, vec3(0.5), vec3(0.5), cWorld, dWorld);
			// Blend the world palette with the pure base rainbow so it's both vivid
			// AND world-flavored (not a flat single hue).
			const rainbow = mix(baseRainbow, worldPal, 0.6);

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

		// ── THE COLOR BLAST ── the signature detonation chroma. Keyed on the LONG
		// uBlast envelope (~2.8s) so the color OUTLIVES the physics burst (owner's
		// "color too brief" fix). Two layers: a blackbody plasma core that cools as
		// the blast ages, and an outward-traveling SPECTRAL DISPERSION WAVE — rainbow
		// shockwave rings expanding through the radius over uBlastTime, like a prism
		// shattering. The unexpected color blast nobody else ships.
		const blastColor = Fn(() => {
			const pos = instancePos;
			const b = clamp(this.uBlast, 0, 1);
			const bt = this.uBlastTime;
			const rNorm = clamp(length(pos).div(this.uContainRadius.max(0.0001)), 0, 1);
			// Blackbody embers: a WARM core (capped ~5200K so it's hot-orange, NOT
			// blinding blue-white — the white-out the owner saw was a 13000K plasma
			// flash). Gentle gain so it tints, never dominates.
			const kelvin = mix(float(1600.0), float(5200.0), b);
			const gain = clamp(b.mul(1.1).add(0.4), 0, 1.3);
			const fire = blackbody(kelvin).mul(gain);
			// THE STAR OF THE BLAST — an outward SPECTRAL DISPERSION shockwave:
			// concentric rainbow rings travel out through the radius over time (red
			// lags, blue leads — real prism order). This is the color, not the fire.
			const specT = fract(rNorm.mul(1.6).sub(bt.mul(1.5)));
			const spectrum = palette(specT, vec3(0.55), vec3(0.55), vec3(3.0), vec3(0.0, 0.33, 0.67));
			// Spectrum DOMINATES (0.78); a touch of warm fire underneath for energy.
			// The owner wants a COLOR blast, so the rainbow wins over the plasma.
			return mix(fire, spectrum, float(0.78));
		});

		// colorNode: world color × rim × act dim, then the blast overrides toward
		// detonation chroma at the peak and lingers (the long uBlast tail) before
		// melting back into the next world's palette.
		mat.colorNode = Fn(() => {
			const glow = clamp(this.uIgnition.mul(0.05).add(0.5), 0, 1.0);
			const world = rainbowColor().mul(glow).mul(rimFactor()).mul(this.uActDim);
			const blastMix = smoothstep(float(0.0), float(0.85), clamp(this.uBlast, 0, 1));
			return mix(world, blastColor().mul(rimFactor()).mul(this.uActDim), blastMix);
		})();

		// emissiveNode: what the selective bloom reads — THE glow channel. Rim-gated
		// so ONLY the outer shell blooms (calm dark center, no white blob). The blast
		// gain is held below the color path (×0.85) so the bloom never clips white.
		mat.emissiveNode = Fn(() => {
			const emGain = clamp(this.uIgnition.mul(0.04).add(0.6), 0, 1.1);
			const world = rainbowColor().mul(emGain).mul(rimFactor()).mul(this.uActDim);
			const blastMix = smoothstep(float(0.0), float(0.85), clamp(this.uBlast, 0, 1));
			return mix(world, blastColor().mul(0.85).mul(rimFactor()).mul(this.uActDim), blastMix);
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
		// World crossfade: ease uBlend 1→0 over ~1s after each beat so the cloud
		// melts from the previous world's home/forces into the new one's.
		this.uBlend.value = Math.max(0, this.uBlend.value - dt * 1.0);
		// Ignition decays toward 0 between beats (spikes back up on transitionTo()).
		this.uIgnition.value = Math.max(0, this.uIgnition.value - dt * 2.0);
		// Burst decays fast so the explosion crystallizes back within ~1.2s.
		this.uBurst.value = Math.max(0, this.uBurst.value - dt * 0.85);
		// COLOR BLAST: slow decay (~2.8s) so the detonation chroma LASTS, and the
		// wave clock counts up so the spectral shockwave travels outward over time.
		this.uBlast.value = Math.max(0, this.uBlast.value - dt * 0.35);
		this.uBlastTime.value += dt;

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

		// WORLD ADVANCE: beats map 1:1 to the 7 worlds. Record the outgoing world
		// and reset the crossfade so the cloud melts prev→new over ~1s.
		this.uPrevWorld.value = this.uWorld.value;
		this.uWorld.value = beatIndex % this.worldCount;
		this.uBlend.value = 1; // 1 = fully previous; update() eases it to 0

		// Per-beat warm-up dim: beats 0/1 stay calm (dialed-in safety), then hands
		// off to the act-based brightness. Acts II/III blaze.
		const warmup = beatIndex === 0 ? 0.12 : beatIndex === 1 ? 0.2 : null;
		const actDim = act === 'I' ? 0.26 : 1.0;
		this.uActDim.value = warmup ?? actDim;
		// Ignition flash: nearly none on beats 0/1, gentle for the rest of Act I,
		// full blaze for Acts II/III.
		this.uIgnition.value = beatIndex <= 1 ? 0.4 : act === 'I' ? 1.6 : 8.0;
		// PHYSICS BURST (fast) — contradiction + the DETONATION world (3) hit hardest.
		const isDetonation = this.uWorld.value === 3;
		this.uBurst.value = mode === 2 || isDetonation ? 1.0 : 0.8;
		// COLOR BLAST (LONG) — fire the chroma envelope + reset its outward-wave
		// clock. Beats 0/1 keep it very low so the calm opener never flashes.
		this.uBlast.value = beatIndex <= 1 ? 0.25 : 1.0;
		this.uBlastTime.value = 0;
		// Dramatic beats (contradiction=2, surprise=3) push their mode color over
		// the rainbow so they read clearly; calm beats stay mostly iridescent.
		this.uModeTintAmt.value = mode >= 2 ? 0.7 : 0.22;
	}

	/**
	 * ENDLESS DREAM BEAT — fired on a timer AFTER the scripted tour ends, so the
	 * storm never sits idle. Jumps to a RANDOM procedural figure (worlds 7..11),
	 * reseeds it (so it's never the same shape twice), ramps uChaos up so each one
	 * is wilder than the last, and detonates a full color blast. This is the
	 * "random figure generator that makes even crazier beats."
	 */
	dreamBeat(): void {
		this.dreamCount += 1;
		// Pick a random wild figure (worlds 7..11 are the procedural generators).
		const world = 7 + Math.floor(Math.random() * 5);
		this.uPrevWorld.value = this.uWorld.value;
		this.uWorld.value = world;
		this.uBlend.value = 1;
		// Fresh random seed → the superformula/knot/lissajous/helix/foam params all
		// change, so the same world index never looks the same twice.
		this.uMorphSeed.value = Math.random() * 1000;
		// Chaos ramps up and saturates — figures get progressively crazier, then
		// hold at max wildness. Eases in over the first ~8 dream beats.
		this.uChaos.value = Math.min(1, 0.25 + this.dreamCount * 0.1);
		// Detonation + long color blast every dream beat — but ignition kept
		// MODERATE (not the tour's 8.0) so the random dense figures don't wash to
		// white at the blast peak. The rim-gated spectral blast carries the color.
		this.uActDim.value = 0.85;
		this.uIgnition.value = 3.0;
		this.uBurst.value = 1.0;
		this.uBlast.value = 1.0;
		this.uBlastTime.value = 0;
		// Vary the mode tint randomly too so the palette keeps surprising.
		const modes = [1, 2, 3];
		this.uMode.value = modes[Math.floor(Math.random() * modes.length)];
		this.uModeTintAmt.value = 0.3 + Math.random() * 0.5;
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
