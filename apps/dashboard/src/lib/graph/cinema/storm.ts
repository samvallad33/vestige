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
	smoothstep,
	oneMinus,
	cross,
	sqrt,
	pow,
	mx_noise_vec3,
	vec2,
	atan,
	positionView,
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
	// JARRING CLASH PAIR — which opposing inner/outer duotone is live (0..4). Set
	// per beat so every figure is a fresh ice-vs-fire / acid-vs-blood collision.
	private uClash = uniform(0);
	// NEAR-PLANE FADE — particles dissolve as they pass very close to the camera
	// (flythrough) so they never additive-pop. Distance band in world units.
	private uFadeNear = uniform(2.0);
	private uFadeBand = uniform(7.0);
	// VOLUMETRIC FOG — distant particles dim toward the void with view depth (exp
	// falloff) for atmospheric depth. Combined with near-fade in one depth read.
	private uFogDensity = uniform(0.012);
	// DEPTH OF FIELD — off-focus particles dim (read as bokeh defocus under the
	// bloom). Folded into the single depthFade depth read (no sprite-scale, which
	// is finicky + collides with the streak). Focus tracks the dive.
	private uFocus = uniform(28.0);
	private uFocusRange = uniform(20.0); // wider in-focus band → most of figure crisp
	private uDofDim = uniform(0.3); // subtle off-focus fade → depth without darkening
	// INFINITE DROSTE ZOOM — the spine. The cloud endlessly dives inward: the
	// nested inner figure grows by λ each period to become the new outer shell,
	// while a fresh inner spawns inside, looping FOREVER with no seam. λ = 1/0.52
	// (the inner scale) makes inner→outer EXACT so the snap is invisible. Pure
	// fract(uTime/T) — no camera dolly (can't clip / fight the camera clamp).
	private uZoomPeriod = uniform(9.0); // T: one promotion every 9s
	private uLambda = uniform(1.923); // 1 / 0.52 self-similar ratio
	private uZoomOn = uniform(0); // 0 = off (beats 0/1, reduced-motion), 1 = diving
	// VELOCITY-STRETCH FLYTHROUGH STREAK — when the camera plunges through the
	// shell, sprites elongate along the screen-space apparent motion vector (a
	// motion-streak look). Pure scaleNode/rotationNode (a SEPARATE output graph
	// from color/emissive) + camera-velocity uniforms → zero per-frame compute,
	// no positionView read in color/emissive. Strength is gated to 0 from JS for
	// reduced-motion, so this is a no-op until the director drives uStreak.
	private uCamVelView = uniform(new THREE.Vector3(0, 0, 0)); // view-space apparent particle velocity
	private uStreak = uniform(0); // 0..1 flythrough strength
	private uMaxStretch = uniform(7.0);
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

			// ── (u,v) MANIFOLD GRID ── THE spaghetti→skin fix. The hash-scatter
			// basis (u,v,w2) is white noise → reads as gas/strings. A deterministic
			// tensor grid over instanceIndex makes neighbors share rows/cols, so the
			// procedural forms below render as a continuous SCULPTED SKIN, not lines.
			// 387² = 149769 ≈ 150k. Pure arithmetic on fi — no buffers, no indexing.
			const GW = float(387);
			const ug = fract(fi.div(GW)); // grid u 0..1 (across a row)
			const vg = floor(fi.div(GW)).div(GW); // grid v 0..1 (down columns)

			// ── COMPLEX-MATH HELPERS (sinh/cosh/tanh are NOT in three@0.172 — expand
			// via the confirmed .exp()). Used by the Calabi–Yau + Boy's surface forms. ──
			type FNode = ReturnType<typeof float>;
			type VNode = ReturnType<typeof vec2>;
			const sinhT = (x: FNode) => x.exp().sub(x.mul(-1).exp()).mul(0.5);
			const coshT = (x: FNode) => x.exp().add(x.mul(-1).exp()).mul(0.5);
			const cMul = (a: VNode, b: VNode) =>
				vec2(a.x.mul(b.x).sub(a.y.mul(b.y)), a.x.mul(b.y).add(a.y.mul(b.x)));
			const cExp = (a: VNode) => {
				const e = a.x.exp();
				return vec2(e.mul(cos(a.y)), e.mul(sin(a.y)));
			};
			const cLog = (a: VNode) =>
				vec2(a.x.mul(a.x).add(a.y.mul(a.y)).max(1e-12).log().mul(0.5), atan(a.y, a.x));
			const cPow = (a: VNode, p: FNode) => cExp(cMul(vec2(p, float(0)), cLog(a)));
			const cCosh = (z: VNode) => vec2(coshT(z.x).mul(cos(z.y)), sinhT(z.x).mul(sin(z.y)));
			const cSinh = (z: VNode) => vec2(sinhT(z.x).mul(cos(z.y)), coshT(z.x).mul(sin(z.y)));
			const cInv = (a: VNode) => {
				const dd = a.x.mul(a.x).add(a.y.mul(a.y)).max(1e-6);
				return vec2(a.x.div(dd), a.y.mul(-1).div(dd));
			};

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

			// ══════ IMPOSSIBLE-GEOMETRY FORM PACK (worlds 8..11) ══════
			// Brand-new signature skins nobody ships as a living particle figure.

			// world 8 · CALABI–YAU quintic cross-section (6D string-theory manifold,
			// Hanson 4D→3D projection). 25 interlocking petals; α rotates it THROUGH
			// the 4th dimension so petals pass through each other. The trophy.
			const nCY = float(5);
			const patch = floor(fract(seed.mul(0.013).add(fi.mul(0.00667))).mul(25));
			const k1 = floor(patch.div(5)); // 0..4
			const k2 = patch.sub(k1.mul(5)); // 0..4
			const cyx = ug.mul(2).sub(1); // x ∈ [-1,1]
			const cyy = vg.mul(1.5708); // y ∈ [0, π/2]
			const zc = vec2(cyx, cyy);
			const e1 = cExp(vec2(float(0), k1.mul(6.28318).div(nCY)));
			const e2 = cExp(vec2(float(0), k2.mul(6.28318).div(nCY)));
			const z1 = cMul(e1, cPow(cCosh(zc), float(0.4))); // 2/n = 0.4
			const z2 = cMul(e2, cPow(cSinh(zc), float(0.4)));
			const alpha = this.uTime.mul(0.25).add(seed).add(chaos.mul(1.5));
			const wKnot = vec3(
				z1.x,
				z2.x,
				cos(alpha).mul(z1.y).add(sin(alpha).mul(z2.y))
			).mul(R.mul(0.55)); // ±1.6 → ~0.88R, centroid (0,0,0)

			// world 9 · BOY'S SURFACE (Bryant–Kusner minimal immersion of RP²) — a
			// CLOSED non-orientable surface with one triple point, no spikes. Pure
			// rational complex arithmetic over the unit disk → a perfect 2-manifold.
			const br = sqrt(ug); // sqrt → uniform area on the disk
			const bth = vg.mul(6.28318);
			const zb = vec2(br.mul(cos(bth)), br.mul(sin(bth)));
			const zb2 = cMul(zb, zb);
			const zb3 = cMul(zb2, zb);
			const zi2 = cInv(zb2);
			const zi3 = cInv(zb3);
			const denom = vec2(zb3.x.sub(zi3.x).add(2.2360679), zb3.y.sub(zi3.y)); // +√5
			const aZ = cInv(denom);
			const V0 = cMul(vec2(float(0), float(1)), vec2(zb2.x.sub(zi2.x), zb2.y.sub(zi2.y)));
			const V1 = vec2(zb2.x.add(zi2.x), zb2.y.add(zi2.y));
			const V2 = cMul(vec2(float(0), float(0.6667)), vec2(zb3.x.add(zi3.x), zb3.y.add(zi3.y)));
			const Mx = cMul(aZ, V0).x;
			const My = cMul(aZ, V1).x;
			const Mz = cMul(aZ, V2).x.add(0.5);
			const m2 = Mx.mul(Mx).add(My.mul(My)).add(Mz.mul(Mz)).max(1e-4); // sphere inversion
			const wLissa = vec3(Mx.div(m2), My.div(m2), Mz.div(m2).sub(0.86)) // sub centroid z
				.mul(R.mul(0.5));

			// world 10 · AIZAWA attractor SHELL (capped spiral torus mapped over u,v;
			// the Aizawa vector field added in the motion modifiers makes it breathe).
			const az = vg.mul(2).sub(1); // -1..1 vertical
			const ar = sqrt(float(1).sub(az.mul(az)).max(0)).mul(0.9).add(0.25); // radial profile
			const aang = ug.mul(6.28318).add(az.mul(6).mul(chaos.add(0.4))); // spiral twist
			const wHelix = vec3(
				ar.mul(cos(aang)),
				az.mul(1.4), // centered by construction
				ar.mul(sin(aang))
			).mul(R.mul(0.5));

			// world 11 · GYROID↔SCHWARZ-D Bonnet morph (triply-periodic minimal
			// surface — alien coral/bone lattice). The Bonnet angle θ continuously
			// BENDS the gyroid into Schwarz-D. A woven solid skin, never seen living.
			const period = float(2.2).add(chaos.mul(2.0));
			const gx = ug.mul(6.28318).mul(period);
			const gy = vg.mul(6.28318).mul(period);
			const gz = this.uTime.mul(0.3).add(seed.mul(6.28318));
			const gyroid = sin(gx).mul(cos(gy)).add(sin(gy).mul(cos(gz))).add(sin(gz).mul(cos(gx)));
			const schwD = cos(gx).mul(cos(gy)).mul(cos(gz)).sub(sin(gx).mul(sin(gy)).mul(sin(gz)));
			const bonnet = this.uTime.mul(0.15);
			const fTPMS = cos(bonnet).mul(gyroid).add(sin(bonnet).mul(schwD));
			const tpmsBase = vec3(
				sin(vg.mul(3.14159)).mul(cos(ug.mul(6.28318))),
				cos(vg.mul(3.14159)),
				sin(vg.mul(3.14159)).mul(sin(ug.mul(6.28318)))
			);
			const wFoam = tpmsBase.mul(R.mul(0.5).add(fTPMS.mul(R.mul(0.12)))); // skin ± displacement

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
			const outerHome = mix(homePrev, homeCur, blendE);

			// ══════════════════════════════════════════════════════════════════
			//  3D-WITHIN-3D — a NESTED INNER FIGURE at the core.
			//  ~33% of particles (a deterministic slice of the index) form a
			//  SECOND, smaller shape inside the outer shell — a different world,
			//  counter-rotating, at ~38% scale. This fills the formerly-blank-bright
			//  center with intentional structure (a figure inside a figure) and adds
			//  depth nobody ships with particles. The inner world is offset from the
			//  outer so the two layers never collapse into the same shape.
			// ══════════════════════════════════════════════════════════════════
			const isInner = fract(fi.mul(0.001).add(0.5)).greaterThan(0.66); // ~34% inner
			// Inner world = outer + 5, wrapped into 0..11 (a clearly different shape).
			// Done with select() (no .mod()) so it's valid in this TSL build.
			const innerSum = float(this.uWorld).add(5);
			const innerWorldIdx = select(innerSum.greaterThan(11), innerSum.sub(12), innerSum);
			const innerRaw = homeFor(innerWorldIdx);
			// Counter-rotate the inner figure about Y so it spins against the shell,
			// and scale it down to sit inside. cos/sin build a Y-rotation matrix.
			const ia = this.uTime.mul(0.4);
			const ic = cos(ia);
			const is = sin(ia);
			const innerRot = vec3(
				innerRaw.x.mul(ic).sub(innerRaw.z.mul(is)),
				innerRaw.y,
				innerRaw.x.mul(is).add(innerRaw.z.mul(ic))
			);
			const innerHome = innerRot.mul(0.52); // nested core at ~52% scale (spread → less white)

			// ── INFINITE ZOOM DIVE ── two layers ride offset phases of fract(uTime/T):
			// the outer grows pow(λ, phase) toward the camera then snaps back (invisible
			// because inner@1/λ == outer@1); the inner (half-period offset) grows from
			// its own base, promoted by λ so it becomes the next outer shell. uZoomOn
			// gates it → beats 0/1 + reduced-motion stay perfectly still.
			const zPhaseO = this.uTime.div(this.uZoomPeriod).fract();
			const zPhaseI = this.uTime.div(this.uZoomPeriod).add(0.5).fract();
			const zoomO = mix(float(1.0), this.uLambda.pow(zPhaseO), this.uZoomOn);
			const zoomI = mix(float(1.0), this.uLambda.pow(zPhaseI), this.uZoomOn);
			const outerDive = outerHome.mul(zoomO);
			const innerDive = innerHome.mul(zoomI.mul(this.uLambda)); // inner promoted by λ each cycle
			// Each particle is permanently outer OR inner (no flicker): pick its home.
			const home = mix(outerDive, innerDive, isInner.select(float(1), float(0)));

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
			// world 10: integrate the AIZAWA vector field so the shell breathes/spirals
			// along the real attractor flow (not a static torus).
			const azx = pos.x.div(R.mul(0.5));
			const azy = pos.y.div(R.mul(0.5));
			const azz = pos.z.div(R.mul(0.5));
			const aizawa = vec3(
				azz.sub(0.7).mul(azx).sub(azy.mul(3.5)),
				azx.mul(3.5).add(azz.sub(0.7).mul(azy)),
				float(0.6).add(azz.mul(0.95)).sub(azz.mul(azz).mul(azz).div(3)).sub(azx.mul(azx).add(azy.mul(azy)))
			);
			vel.addAssign(aizawa.mul(0.008).mul(select(this.uWorld.equal(10), float(1), float(0))));
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

		// CANONICAL SPRITE CENTER: positionNode is the sprite's CENTER only.
		// SpriteNodeMaterial.setupPositionView already builds the billboard quad
		// from positionGeometry.xy (scaled by scaleNode, rotated by rotationNode),
		// so the previous `.add(positionLocal)` double-counted the quad (harmless at
		// the 0.1 size, sub-pixel). Using the bare center is required for the
		// velocity-stretch streak (scaleNode/rotationNode now drive the quad shape).
		const phaseStore = storage(bufferPhase, 'float', this.count);
		const instancePos = storage(bufferPos, 'vec3', this.count).element(instanceIndex);
		mat.positionNode = instancePos;

		// ── VELOCITY-STRETCH STREAK (scaleNode + rotationNode) ── a SEPARATE output
		// graph from color/emissive: it reads only camera-velocity uniforms +
		// phaseStore, NEVER positionView (a 2nd positionView read in a material
		// output would stack-overflow three@0.172's node builder). Zero per-frame
		// compute — the camera velocity is one uniform pushed from the sandbox.
		const matStretch = mat as typeof mat & { scaleNode: unknown; rotationNode: unknown };
		{
			// Screen-plane apparent particle velocity (view space x/y). speed≥ε so
			// length()/atan() stay finite when the camera is still.
			const velView = vec3(this.uCamVelView);
			const vScreen = vec2(velView.x, velView.y);
			const speed = length(vScreen).max(1e-4);
			// Per-particle jitter (0.75..1.25) so streaks don't all stretch identically.
			const jit = fract(phaseStore.element(instanceIndex).mul(13.17)).mul(0.5).add(0.75);
			// Orient the quad's long axis along the motion vector. atan(y, x) is the
			// 2-arg form (full -π..π range), NOT atan2.
			matStretch.rotationNode = atan(vScreen.y, vScreen.x);
			// X-stretch: 1 (no streak) → uMaxStretch, scaled by speed × strength × jitter.
			const stretch = clamp(speed.mul(this.uStreak).mul(jit).mul(0.6).add(1.0), float(1.0), this.uMaxStretch);
			// bloat = DOF circle-of-confusion base (DOF brightness is in depthFade;
			// the sprite-scale bloat is the neutral 1.0 here). Streak elongates X only;
			// bloat stays on both axes so the DOF circle is preserved.
			const bloat = float(1.0);
			matStretch.scaleNode = vec2(bloat.mul(stretch), bloat);
		}

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

			// Recompute the inner/outer layer split (same formula as the compute kernel).
			const fiC = float(instanceIndex);
			const isInnerC = fract(fiC.mul(0.001).add(0.5)).greaterThan(0.66);

			// A flowing texture coordinate per particle — drives gradients WITHIN each
			// layer's duotone so it shimmers, but stays inside that layer's color world.
			const spatialBand = pos.x.mul(0.03).add(pos.y.mul(0.021)).add(pos.z.mul(0.027));
			const flow = fract(
				ph.mul(0.41).add(radius.mul(0.06)).add(spatialBand).add(this.uTime.mul(0.10)).add(this.uHueShift)
			);

			// ══════════════════════════════════════════════════════════════════
			//  JARRING DUOTONE CLASH — the share hook.
			//  The outer shell and the inner nested figure are painted from OPPOSING
			//  color universes (ice vs fire, acid vs blood, gold vs violet…). Not a
			//  hue shift in one rainbow — two palettes that FIGHT. uClash (set per
			//  beat) picks which clashing pair is live, so it's a fresh jarring combo
			//  every beat. Each layer is a 2-color gradient (cold→cold, hot→hot) so
			//  the layer reads as ONE color world, and the two worlds collide at the
			//  boundary. THIS is what makes someone stop scrolling and share.
			// ══════════════════════════════════════════════════════════════════
			// Five hand-picked clash pairs: [outerA, outerB, innerA, innerB].
			const cl = this.uClash; // 0..4, set per beat
			// outer gradient endpoints
			const outA = select(cl.equal(0), vec3(0.0, 0.85, 1.0), // ICE: electric cyan
				select(cl.equal(1), vec3(0.55, 1.0, 0.0), // ACID lime
				select(cl.equal(2), vec3(1.0, 0.82, 0.0), // GOLD
				select(cl.equal(3), vec3(0.0, 1.0, 0.6), // MINT/emerald
				vec3(0.1, 0.5, 1.0))))); // ELECTRIC blue
			const outB = select(cl.equal(0), vec3(0.3, 0.2, 1.0), // ICE→deep indigo
				select(cl.equal(1), vec3(0.0, 0.7, 0.5), // ACID→teal
				select(cl.equal(2), vec3(1.0, 0.4, 0.0), // GOLD→amber
				select(cl.equal(3), vec3(0.0, 0.6, 1.0), // MINT→cyan
				vec3(0.5, 0.0, 1.0))))); // ELECTRIC→violet
			// inner gradient endpoints — the OPPOSING world
			const inA = select(cl.equal(0), vec3(1.0, 0.25, 0.0), // FIRE: molten orange
				select(cl.equal(1), vec3(1.0, 0.0, 0.55), // BLOOD: hot pink
				select(cl.equal(2), vec3(0.6, 0.0, 1.0), // VIOLET
				select(cl.equal(3), vec3(1.0, 0.1, 0.3), // CRIMSON
				vec3(1.0, 0.7, 0.0))))); // GOLD
			const inB = select(cl.equal(0), vec3(1.0, 0.0, 0.3), // FIRE→crimson
				select(cl.equal(1), vec3(1.0, 0.45, 0.0), // BLOOD→orange
				select(cl.equal(2), vec3(1.0, 0.0, 0.7), // VIOLET→magenta
				select(cl.equal(3), vec3(1.0, 0.5, 0.0), // CRIMSON→amber
				vec3(1.0, 0.2, 0.4))))); // GOLD→rose

			// Each layer = a 2-stop gradient driven by `flow` (stays in its world).
			const grad = smoothstep(float(0.0), float(1.0), flow);
			const outerColor = mix(outA, outB, grad);
			const innerColor = mix(inA, inB, grad);
			// Hard pick by layer → the clash is absolute at the boundary.
			const rainbow = mix(outerColor, innerColor, isInnerC.select(float(1), float(0)));

			// Beat mode tint kept very light so it never muddies the clash.
			const modeTint = select(
				this.uMode.equal(2),
				vec3(1.0, 0.08, 0.32),
				select(this.uMode.equal(3), vec3(1.0, 0.78, 0.1), vec3(0.1, 0.9, 1.0))
			);
			return mix(rainbow, modeTint, this.uModeTintAmt.mul(0.4));
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
			// Smooth ramp: dark core, bright rim — the outer-shell glow that keeps the
			// center from blooming white (preserved white-out protection).
			const edge = rNorm.mul(rNorm);
			// FACING-RATIO FRESNEL — the "make it solid" amplifier. Approximate each
			// particle's surface normal as its outward radial direction; view ≈ +Z.
			// pow(1−|n·v|, 4) blazes the turning-away SILHOUETTE and quiets the front,
			// which flips "glowing fog" into a lit, sculpted SKIN.
			const nrm = pos.normalize();
			const fres = pow(oneMinus(abs(nrm.z)), float(4.0));
			const outerRim = float(0.12).add(edge.mul(0.6)).add(fres.mul(0.5));
			// The NESTED inner figure lives at small radius where `edge` is ~0 → it
			// would be invisible. Give inner particles their OWN brightness: a higher
			// floor + the same Fresnel silhouette so the inner figure reads as its own
			// glowing sculpted object floating inside the shell.
			const fiC = float(instanceIndex);
			const isInnerC = fract(fiC.mul(0.001).add(0.5)).greaterThan(0.66);
			// Inner glow kept VERY low so the nested figure's CLASH COLOR survives as
			// color, not white. Dense small-radius overlap blows to white fast and
			// kills the contrast — so the inner sits dim and saturated, carried by its
			// Fresnel silhouette. This is what makes the jarring inner/outer clash read.
			const innerRim = float(0.07).add(fres.mul(0.3));
			const baseRim = isInnerC.select(innerRim, outerRim);

			// ── ZOOM SEAM CROSS-FADE ── each dive layer must be fully transparent
			// exactly when it snaps (phase 0/1), so the infinite loop has NO visible
			// pop. sin(phase·π) = 0 at the snap, 1 mid-cycle. Normalize by the sum so
			// total opacity stays ≈1. uZoomOn=0 forces weight to 1 (no fade when still).
			const pO = this.uTime.div(this.uZoomPeriod).fract();
			const pI = this.uTime.div(this.uZoomPeriod).add(0.5).fract();
			const wO = sin(pO.mul(3.14159));
			const wI = sin(pI.mul(3.14159));
			const wSum = wO.add(wI).max(0.0001);
			const seamO = mix(float(1.0), wO.div(wSum).mul(2.0).min(1.0), this.uZoomOn);
			const seamI = mix(float(1.0), wI.div(wSum).mul(2.0).min(1.0), this.uZoomOn);
			const seam = isInnerC.select(seamI, seamO);
			return baseRim.mul(seam);
		});

		// ── DEPTH FADE (near dissolve × volumetric fog) ── ONE Fn, ONE positionView
		// read. Reading positionView from a SECOND Fn in the same material output
		// triggers a cyclic type-resolution stack-overflow in three@0.172's node
		// builder, so near-fade AND fog are combined here into a single depth read.
		//   near: smoothstep 0→1 across [near, near+band] (≈1 at far camera → no
		//         change until we fly inside; dissolves sprites at the camera).
		//   fog:  exp falloff dims distant particles toward the void → 3D depth.
		const depthFade = Fn(() => {
			const d = positionView.z.negate(); // +forward view distance, read ONCE
			const near = smoothstep(this.uFadeNear, this.uFadeNear.add(this.uFadeBand), d);
			const fog = clamp(this.uFogDensity.mul(d).negate().exp(), 0.18, 1.0);
			// DOF: dim particles off the focus plane (defocus → bokeh under bloom).
			// coc 0 at focus → 1 fully out of focus; brightness 1 → (1-uDofDim).
			const coc = clamp(d.sub(this.uFocus).abs().div(this.uFocusRange), 0, 1);
			const focusBright = oneMinus(coc.mul(this.uDofDim));
			return near.mul(fog).mul(focusBright);
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
			// Blast is CAPPED at 0.6 so the inner/outer CLASH duotone always shows
			// through even during a detonation — the clash is the star, the blast is
			// an accent (was fully overriding, which washed the contrast to rainbow).
			const blastMix = smoothstep(float(0.0), float(0.85), clamp(this.uBlast, 0, 1)).mul(0.6);
			return mix(world, blastColor().mul(rimFactor()).mul(this.uActDim), blastMix).mul(depthFade());
		})();

		// emissiveNode: what the selective bloom reads — THE glow channel. Rim-gated
		// so ONLY the outer shell blooms (calm dark center, no white blob). The blast
		// gain is held below the color path (×0.85) so the bloom never clips white.
		mat.emissiveNode = Fn(() => {
			const emGain = clamp(this.uIgnition.mul(0.04).add(0.6), 0, 1.1);
			const world = rainbowColor().mul(emGain).mul(rimFactor()).mul(this.uActDim);
			// Same cap as colorNode so the bloom feeds on the CLASH colors, not a
			// rainbow override.
			const blastMix = smoothstep(float(0.0), float(0.85), clamp(this.uBlast, 0, 1)).mul(0.6);
			// ×nearFade so the bloom ALSO dissolves near the camera (no near-plane flash).
			return mix(world, blastColor().mul(0.85).mul(rimFactor()).mul(this.uActDim), blastMix).mul(depthFade());
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
		// RACK FOCUS — when diving, the focus plane tracks the descent (pulls inward
		// over each zoom cycle so the new inner figure crisps up as it grows); when
		// still, a gentle breathing pull keeps the DOF alive.
		const zp = (this.uTime.value / this.uZoomPeriod.value) % 1;
		const focusTarget = this.uZoomOn.value > 0.5
			? 40 - zp * 16 // 40→24 over each dive cycle (keeps the grown figure in focus)
			: 26 + Math.sin(this.uTime.value * 0.18) * 9; // 17..35 breathing
		this.uFocus.value += (focusTarget - this.uFocus.value) * Math.min(1, dt * 3);

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
		// Cycle the jarring inner/outer clash pair so each beat collides a fresh
		// pair of opposing color worlds (ice↔fire, acid↔blood, …).
		this.uClash.value = beatIndex % 5;
		// Engage the infinite dive from Act II onward (beats 0/1 stay calm + still).
		this.uZoomOn.value = beatIndex >= 2 ? 1 : 0;

		// Per-beat warm-up dim: beats 0/1 stay calm (dialed-in safety), then hands
		// off to the act-based brightness. Acts II/III blaze.
		const warmup = beatIndex === 0 ? 0.12 : beatIndex === 1 ? 0.2 : null;
		const actDim = act === 'I' ? 0.26 : 1.0;
		this.uActDim.value = warmup ?? actDim;
		// Ignition flash: nearly none on beats 0/1, gentle for the rest of Act I,
		// strong (but no longer white-blowing) for Acts II/III — lowered from 8.0 to
		// 4.5 so the inner/outer CLASH colors survive the detonation instead of
		// washing to white.
		this.uIgnition.value = beatIndex <= 1 ? 0.4 : act === 'I' ? 1.6 : 4.5;
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
		// Random opposing clash pair each dream figure → never the same collision.
		this.uClash.value = Math.floor(Math.random() * 5);
		// Chaos ramps up and saturates — figures get progressively crazier, then
		// hold at max wildness. Eases in over the first ~8 dream beats.
		this.uChaos.value = Math.min(1, 0.25 + this.dreamCount * 0.1);
		// Dream mode dives endlessly — the infinite zoom is always on here.
		this.uZoomOn.value = 1;
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

	/** Enable/disable the infinite Droste zoom dive. Off for beats 0/1 (calm) and
	 * reduced-motion; on for Act II+ and dream mode. */
	setZoom(on: boolean): void {
		this.uZoomOn.value = on ? 1 : 0;
	}

	/** View-space apparent particle velocity (the negated, view-transformed camera
	 * velocity). Drives the streak orientation + length. */
	setCameraVel(v: THREE.Vector3): void {
		this.uCamVelView.value.copy(v);
	}

	/** Flythrough streak strength 0..1. Set to 0 for reduced-motion (no streak). */
	setStreak(s: number): void {
		this.uStreak.value = s;
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
