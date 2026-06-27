<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { growSign } from '$lib/landing/dendriteGen';

	interface Props {
		seed?: number;
		reducedMotion?: boolean;
		class?: string;
		/** element (.launch-shell) that receives the per-frame --burst / --flash
		 *  CSS vars so the DOM overlay can implode/explode in sync. */
		syncTarget?: HTMLElement;
		/** when true (email form focused/interacting) the overlay is pinned to rest
		 *  so a signup is never disrupted mid-type. */
		suppress?: boolean;
	}

	type BloomMip = { texture: any; view: any; width: number; height: number };
	type GPUHandle = {
		device: any;
		context: any;
		format: string;
		hdrFormat: string;
		particleBuffer: any;
		uniformBuffer: any;
		computePipeline: any;
		renderPipeline: any;
		tonemapPipeline: any;
		bloomDownPipeline: any;
		bloomUpPipeline: any;
		bloomBGL: any;
		fadePipeline: any;
		trailTexture: any;
		trailView: any;
		trailPrimed: boolean;
		computeBindGroup: any;
		renderBindGroup: any;
		tonemapBindGroup: any; // rebuilt on resize (depends on HDR + bloom textures)
		hdrTexture: any;
		hdrView: any;
		hdrMsaaTexture: any;
		hdrMsaaView: any;
		hdrSampler: any;
		bloomMips: BloomMip[];
		bloomBindGroups: any[]; // per-pass src bind groups, rebuilt on resize
		hdrWidth: number;
		hdrHeight: number;
		particleCount: number;
	};

	// Particle pass renders at half resolution: 4x fewer shaded pixels kills the
	// burst overdraw lag, and the softness is invisible (bloom blurs it anyway).
	const PARTICLE_SCALE = 2;
	const BLOOM_LEVELS = 5;
	// No MSAA (count 1). 4x on rgba16float spikes the resolve cost during the
	// high-variance burst; soft additive orbs barely benefit from MSAA anyway
	// (the round-orb falloff + half-res + bloom already hide aliasing). WebGPU only
	// guarantees sampleCount 1 and 4 — 2 is not portable — so we drop to 1.
	const MSAA_SAMPLES = 1;
	const USE_MSAA = false;

	// per-frame trail fade constant (set in writeUniforms, used by the fade pass).
	let trailDecay = 0.84;

	let {
		seed = 20260625,
		reducedMotion = false,
		class: className = '',
		syncTarget,
		suppress = false
	}: Props = $props();

	let gpuCanvas = $state<HTMLCanvasElement | undefined>(undefined);
	let fallbackCanvas = $state<HTMLCanvasElement | undefined>(undefined);
	let mode = $state<'booting' | 'webgpu' | 'fallback'>('booting');

	let gpu: GPUHandle | null = null;
	let frame = 0;
	let projectionFrame = 0;
	let webgpuRetryTimer: ReturnType<typeof setTimeout> | undefined;
	let viewportKickTimer: ReturnType<typeof setTimeout> | undefined;
	let webgpuBooting = false;
	let webgpuAttempt = 0;
	let webgpuBootToken = 0;
	let fallbackStarted = false;
	let lastFrame = 0;
	let startedAt = 0;
	let disposed = false;
	let fallbackDraw: ((now: number) => void) | null = null;
	let pointerX = 0;
	let pointerY = 0;
	let pointerActive = 0;
	let pointerPressure = 0;
	// uniform block: 5 * vec4f = 80 bytes (16-byte aligned). See Uniforms struct.
	const uniformData = new Float32Array(20);

	// 64 matches both proven webgpu-samples particle demos (computeBoids,
	// particles-HDR) and stays under every device's maxComputeInvocationsPerWorkgroup.
		const WORKGROUP_SIZE = 64;

		function sleep(ms: number) {
			return new Promise<void>((resolve) => setTimeout(resolve, ms));
		}

	function withTimeout<T = any>(promise: Promise<T>, ms: number, label: string): Promise<T> {
		let timer: ReturnType<typeof setTimeout>;
			const timeout = new Promise<never>((_, reject) => {
				timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
			});
		return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
	}

	async function requestBestAdapter(gpuApi: any) {
		const attempts = [
			{ powerPreference: 'high-performance' },
			{},
			{ featureLevel: 'compatibility' }
		];
		for (const descriptor of attempts) {
			try {
				const adapter = await withTimeout(
					gpuApi.requestAdapter(descriptor),
					2500,
					`WebGPU requestAdapter ${JSON.stringify(descriptor)}`
				);
				if (adapter) return adapter;
			} catch (error) {
				console.warn('[launch] WebGPU adapter attempt failed:', descriptor, error);
			}
		}
		return null;
	}

	async function requestBestDevice(adapter: any, requiredLimits: Record<string, number>) {
		try {
			return await withTimeout(
				adapter.requestDevice({ requiredLimits }),
				3500,
				'WebGPU requestDevice with limits'
			);
		} catch (error) {
			console.warn('[launch] WebGPU limited device request failed, retrying defaults:', error);
			return await withTimeout(adapter.requestDevice(), 3500, 'WebGPU requestDevice defaults');
		}
	}

		function destroyGpu() {
			cancelAnimationFrame(frame);
			frame = 0;
			gpu?.particleBuffer?.destroy?.();
			gpu?.uniformBuffer?.destroy?.();
		gpu?.hdrTexture?.destroy?.();
		gpu?.hdrMsaaTexture?.destroy?.();
		gpu?.trailTexture?.destroy?.();
		gpu?.bloomMips?.forEach((m) => m.texture?.destroy?.());
		gpu?.context?.unconfigure?.();
		gpu?.device?.destroy?.();
		gpu = null;
	}

	// ---------------------------------------------------------------------------
	//  COMPUTE SHADER
	//  One looping 4-beat cinematic (STREAM -> EXPLODE -> REFORM -> DISSOLVE)
	//  recreated from the old Three.js nodeEngine, but pure node particles.
	//  The REFORM beat cycles through memory-themed formations:
	//    glyph swarm (VESTIGE) -> brain lobes -> graph constellation ->
	//    memory lattice -> archive bead columns -> receipt streams -> burst.
	//  NOTHING here draws a line, edge, ring, grid, or stroke. Particles only.
	// ---------------------------------------------------------------------------
	const computeShader = /* wgsl */ `
struct Particle {
	pos: vec4f,   // xyz position, w = energy (for glow)
	vel: vec4f,   // xyz velocity, w = spawn delay
	aux: vec4f,   // x = id, y = lane(formation), z = shape kind, w = hue seed
	home: vec4f,  // xyz edge-shell spawn point (dissolve target), w = perimeter t
};

struct Uniforms {
	viewport: vec4f,   // x=width y=height z=dpr w=reducedMotion(0/1)
	pointer: vec4f,    // x,y in clip space, z=active, w=pressure
	time: vec4f,       // x=time(s) y=dt z=shapeA w=shapeB
	mode: vec4f,       // x=particleCount y=morphBlend(0..1) z=unused w=seed
	extra: vec4f,      // x=scrollY (reserved) ...
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

fn hash11(n: f32) -> f32 {
	return fract(sin(n * 91.3458) * 47453.5453);
}

fn rotate2(p: vec2f, a: f32) -> vec2f {
	let c = cos(a);
	let s = sin(a);
	return vec2f(p.x * c - p.y * s, p.x * s + p.y * c);
}

const TAU2: f32 = 6.28318530718;
const INV_SQRT2: f32 = 0.70710678;

// =========================================================================
//  EXOTIC FORMATIONS — strange attractors & 4D projections nobody uses as
//  glowing node clouds. Each particle integrates / parametrises to a unique
//  point so 60k orbs FILL the structure (random integration length is the key
//  trick for attractors: same curve, each orb frozen at a different point).
// =========================================================================

// AIZAWA strange attractor — a chaotic toroidal shell with an axial spike. The
// swarm traces deterministic chaos: the memory system "thinking".
fn aizawa_target(seed: f32, t: f32) -> vec3f {
	let A = 0.95; let B = 0.7; let C = 0.6; let D = 3.5; let E = 0.25; let F = 0.1;
	var p = vec3f(
		(hash11(seed + 0.1) - 0.5) * 0.2,
		(hash11(seed + 0.2) - 0.5) * 0.2,
		(hash11(seed + 0.3) - 0.5) * 0.2 + 0.1
	);
	let dt = 0.01;
	let steps = 380u + u32(hash11(seed + 7.0) * 1400.0);
	for (var i = 0u; i < steps; i = i + 1u) {
		let dx = (p.z - B) * p.x - D * p.y;
		let dy = D * p.x + (p.z - B) * p.y;
		let dz = C + A * p.z - (p.z * p.z * p.z) / 3.0
			- (p.x * p.x + p.y * p.y) * (1.0 + E * p.z)
			+ F * p.z * (p.x * p.x * p.x);
		p = p + vec3f(dx, dy, dz) * dt;
	}
	let centered = vec3f(p.x, p.y, p.z - 0.65) * 0.82;
	// stand the spike up the screen's Y axis, slow autorotate for life.
	let xz = rotate2(vec2f(centered.x, centered.y), t * 0.10);
	return vec3f(xz.x, centered.z, xz.y);
}

// THOMAS cyclically-symmetric attractor — a space-filling pretzel/lattice of
// interlocking loops, perfectly symmetric under x->y->z.
fn thomas_target(seed: f32, t: f32) -> vec3f {
	let b = 0.19;
	var p = vec3f(
		(hash11(seed + 0.1) - 0.5) * 8.0,
		(hash11(seed + 0.2) - 0.5) * 8.0,
		(hash11(seed + 0.3) - 0.5) * 8.0
	);
	let dt = 0.05;
	let steps = 260u + u32(hash11(seed + 7.0) * 1100.0);
	for (var i = 0u; i < steps; i = i + 1u) {
		let dx = sin(p.y) - b * p.x;
		let dy = sin(p.z) - b * p.y;
		let dz = sin(p.x) - b * p.z;
		p = p + vec3f(dx, dy, dz) * dt;
	}
	let s = p * (1.30 / 4.5);
	let xz = rotate2(s.xz, t * 0.08);
	return vec3f(xz.x, s.y, xz.y);
}

// GIELIS SUPERFORMULA supershape — a spiky alien crystal sea-urchin.
fn superR(m: f32, n1: f32, n2: f32, n3: f32, ang: f32) -> f32 {
	let t1 = pow(abs(cos(m * ang / 4.0)), n2);
	let t2 = pow(abs(sin(m * ang / 4.0)), n3);
	return pow(max(t1 + t2, 1e-4), -1.0 / n1);
}
fn supershape_target(seed: f32, t: f32) -> vec3f {
	let m = 7.0; let n1 = 0.20; let n2 = 1.7; let n3 = 1.7;
	let theta = hash11(seed + 0.1) * TAU2 - 3.14159265;
	let v = hash11(seed + 0.2);
	let phi = asin(clamp(2.0 * v - 1.0, -1.0, 1.0));
	let r1 = superR(m, n1, n2, n3, theta);
	let r2 = superR(m, n1, n2, n3, phi);
	var p = vec3f(
		r1 * cos(theta) * r2 * cos(phi),
		r1 * sin(theta) * r2 * cos(phi),
		r2 * sin(phi)
	) * 0.30;
	// breathe + slow spin so the crystal feels alive.
	let pulse = 1.0 + sin(t * 0.5) * 0.06;
	p = p * pulse;
	if (length(p) > 1.32) { p = normalize(p) * 1.32; }
	let xz = rotate2(p.xz, t * 0.12);
	return vec3f(xz.x, p.y, xz.y);
}

// CLIFFORD TORUS — a flat 2-torus living in 4D (S^3), stereographically
// projected to 3D. A nested-inside-itself torus no 3D parametric can make:
// "a non-trivial topological thought."
fn clifford_target(seed: f32, t: f32) -> vec3f {
	let al = hash11(seed + 0.1) * TAU2;
	let be = hash11(seed + 0.2) * TAU2;
	let w  = cos(al) * INV_SQRT2;
	let x4 = sin(al) * INV_SQRT2;
	let y4 = cos(be) * INV_SQRT2;
	let z4 = sin(be) * INV_SQRT2;
	let k = 1.0 / (1.5 - w);
	var p = vec3f(x4, y4, z4) * k * 1.4;
	if (length(p) > 1.34) { p = normalize(p) * 1.34; }
	// tumble in 3D so the self-nesting reveals itself.
	let xz = rotate2(p.xz, t * 0.14);
	let yz = rotate2(vec2f(p.y, xz.y), t * 0.09);
	return vec3f(xz.x, yz.x, yz.y);
}

// MEMORY FIELD: a full-screen living nebula. Nodes spread across the ENTIRE
// frame in slow, breathing drifts — no concentrated 2D shape, so it never
// whites out and it covers the whole viewport. This is the "ambient memory"
// beat (the swarm at rest, everything remembered at once).
fn field_target(aux: vec4f, t: f32) -> vec3f {
	let id = aux.x;
	// even spread across a wide rectangle that overfills the frame edges.
	let gx = hash11(id + 5.0) * 2.0 - 1.0;
	let gy = hash11(id + 9.0) * 2.0 - 1.0;
	let gz = hash11(id + 13.0) * 2.0 - 1.0;
	// overfill the frame edges so the nebula covers the WHOLE screen.
	let p = vec3f(gx * 2.35, gy * 1.5, gz * 0.9);
	// slow per-node breathing so the whole field undulates like a calm sea.
	let drift = vec3f(
		sin(t * 0.25 + id * 0.011),
		cos(t * 0.20 + id * 0.017),
		sin(t * 0.18 + id * 0.007)
	) * 0.10;
	return p + drift;
}

// BRAIN: two lobes, surface-weighted shell with sulci folds (from nodeEngine).
// SEEDED: uniforms.mode.w (the user's email-derived seed) perturbs the brain so
// every signup freezes on a DIFFERENT-but-bounded brain — the sulci phase rotates,
// the inter-lobe gap shifts, and the whole shell rolls a few degrees. All terms are
// bounded (sin/cos/clamped mixes) so the seed can never blow the shape out of frame
// or produce NaN.
fn brain_target(aux: vec4f, t: f32) -> vec3f {
	let id = aux.x;
	// Decorrelate the raw seed into three bounded knobs. fract(sin(...)) keeps each
	// in [0,1) for ANY finite seed (no overflow, no NaN).
	let sd = uniforms.mode.w;
	let sk = fract(sin(sd * 0.000123 + 11.0) * 43758.5453); // sulci phase knob
	let gk = fract(sin(sd * 0.000231 + 23.0) * 24634.6345); // lobe-gap knob
	let rk = fract(sin(sd * 0.000317 + 37.0) * 15731.7431); // roll knob
	let lobe = select(-1.0, 1.0, hash11(id + 7.0) > 0.5);
	let a = hash11(id + 2.0);
	let r = mix(0.78, 1.0, pow(a, 0.5));
	let theta = hash11(id + 23.0) * 6.2831853;
	let phi = acos(2.0 * hash11(id + 37.0) - 1.0);
	var x = abs(sin(phi) * cos(theta)) * lobe;
	let y = cos(phi) * 0.95;
	let z = sin(phi) * sin(theta) * 1.18;
	// SEEDED lobe gap: nominal 0.42, perturbed +/-0.06 (stays well inside frame).
	let gap = 0.42 + (gk - 0.5) * 0.12;
	x = x * 0.78 + lobe * gap;
	// SEEDED sulci: the fold phase rotates by the seed so the ridge pattern differs.
	let sulciPhase = sk * 6.2831853;
	let ridge = sin(x * 8.0 + sulciPhase + t * 0.2) * sin(y * 7.0 + sulciPhase) * 0.10;
	var p = vec3f(x, y, z) * r * (1.0 + ridge) * 1.05;
	// SEEDED roll: rotate the whole brain a few degrees about Y so the frozen brain
	// faces a per-user direction. Bounded rotation -> always in frame.
	let roll = (rk - 0.5) * 0.6;
	let xz = rotate2(p.xz, roll);
	return vec3f(xz.x, p.y, xz.y);
}

// GRAPH CONSTELLATION: fibonacci-sphere nodes only (NO edges drawn — particles
// sit on the node positions; the old version filled edges, we keep nodes only).
fn graph_target(aux: vec4f, t: f32) -> vec3f {
	let id = aux.x;
	let m = 620.0;
	let k = floor(hash11(id + 3.0) * m);
	let y = 1.0 - (k / (m - 1.0)) * 2.0;
	let rr = sqrt(max(0.0, 1.0 - y * y));
	let th = 2.39996323 * k;
	let cluster = vec3f(hash11(id + 41.0) - 0.5, hash11(id + 43.0) - 0.5, hash11(id + 47.0) - 0.5) * 0.10;
	let spin = t * 0.06;
	let p = vec3f(cos(th + spin) * rr, y, sin(th + spin) * rr) * 1.32 + cluster;
	return p;
}

// MEMORY LATTICE: hash-jittered 3D grid of cluster cells (columns + clusters,
// no box lines). Particles fill the cells, implying a lattice through density.
fn lattice_target(aux: vec4f, t: f32) -> vec3f {
	let id = aux.x;
	let g = 6.0;
	let cx = floor(hash11(id + 15.0) * g);
	let cy = floor(hash11(id + 25.0) * g);
	let cz = floor(hash11(id + 35.0) * g);
	let cell = 2.0 / g;
	var p = vec3f((cx + 0.5) * cell - 1.0, (cy + 0.5) * cell - 1.0, (cz + 0.5) * cell - 1.0) * 1.3;
	p += vec3f(hash11(id + 1.0) - 0.5, hash11(id + 2.0) - 0.5, hash11(id + 3.0) - 0.5) * 0.10;
	let sway = sin(t * 0.18 + id * 0.017) * 0.04;
	return vec3f(p.x + sway, p.y, p.z);
}

// ARCHIVE BEAD COLUMNS: vertical streams of beads on either side (no rails
// drawn — the beads themselves are the column).
fn archive_target(aux: vec4f, t: f32) -> vec3f {
	let id = aux.x;
	let side = select(-1.0, 1.0, hash11(id + 91.0) > 0.5);
	let column = floor(hash11(id + 92.0) * 11.0);
	let row = fract(id * 0.0093 + hash11(id + 93.0) * 0.31 + t * 0.05);
	// columns span the full width on BOTH sides AND fill toward center so the
	// archive reads as a full shelf, centered, not two thin edge rails.
	let x = side * (0.30 + column * 0.115 + hash11(id + 96.0) * 0.03);
	let y = row * 2.30 - 1.15;
	let z = (hash11(id + 97.0) - 0.5) * 0.55;
	return vec3f(x, y, z);
}

// RECEIPT STREAMS: bead trails that travel upward in lanes (audit trail), pure
// bead motion, no drawn line.
fn receipt_target(aux: vec4f, t: f32) -> vec3f {
	let id = aux.x;
	let lane = floor(hash11(id + 61.0) * 18.0);
	let x = lane / 17.0 * 2.9 - 1.45;
	let travel = fract(hash11(id + 62.0) + t * 0.10);
	let y = travel * 2.30 - 1.15;
	let wobble = sin(travel * 18.0 + lane) * 0.05;
	let z = (hash11(id + 64.0) - 0.5) * 0.4;
	return vec3f(x + wobble, y, z);
}

// BURST: volumetric facet cloud (the explode/reform exotic beat).
fn burst_target(aux: vec4f, t: f32) -> vec3f {
	let id = aux.x;
	let a = hash11(id + 121.0) * 6.2831853;
	let b = hash11(id + 122.0) * 6.2831853;
	let r = pow(hash11(id + 123.0), 0.30) * 1.7;
	let warp = vec3f(cos(a) * cos(b), sin(b) * 0.8, sin(a) * cos(b) * 0.9);
	let facet = vec3f(hash11(id + 124.0) - 0.5, hash11(id + 125.0) - 0.5, hash11(id + 126.0) - 0.5) * 0.34;
	return (warp + facet) * r + vec3f(sin(t * 0.17 + id * 0.003) * 0.18, 0.02, cos(t * 0.13) * 0.16);
}

// Active formation this loop — EXOTIC point-cloud structures nobody renders as
// glowing node clouds: a chaotic strange attractor, a symmetric loop-lattice, a
// spiky superformula crystal, and a 4D torus projected into 3D. (The particle
// VESTIGE wordmark was removed; the DOM wordmark already names the brand.)
fn formation_target(aux: vec4f, t: f32, shape: f32) -> vec3f {
	let seed = aux.x;
	if (shape < 0.5) { return aizawa_target(seed, t); }
	if (shape < 1.5) { return thomas_target(seed, t); }
	if (shape < 2.5) { return supershape_target(seed, t); }
	if (shape < 3.5) { return clifford_target(seed, t); }
	// shape 4 = the SEEDED BRAIN. Reachable ONLY through the signup override path
	// (shapeB=4); the ambient loop never selects it because SHAPE_COUNT stays 4.
	return brain_target(aux, t);
}

fn curl(p: vec3f, t: f32) -> vec3f {
	return vec3f(
		sin(p.y * 7.2 + t * 1.10),
		cos(p.z * 6.0 - t * 0.93),
		sin(p.x * 6.8 + t * 0.77)
	);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
	let i = gid.x;
	// guard: the last partial workgroup over-runs the array.
	if (i >= arrayLength(&particles)) {
		return;
	}

	var pcl = particles[i];
	let t = uniforms.time.x;
	let dt = min(uniforms.time.y, 0.033);
	let shapeA = uniforms.time.z;       // current formation index
	let shapeB = uniforms.time.w;       // next formation index
	let morph = uniforms.mode.y;        // 0..1 blend A -> B
	let motion = 1.0 - uniforms.viewport.w; // 0 if reduced motion

	var position = pcl.pos.xyz;
	var velocity = pcl.vel.xyz;

	// =====================================================================
	//  TRANSITION: SINGULARITY COLLAPSE -> SUPERNOVA REBIRTH.
	//  Orbs spiral INTO a central singularity (free vortex, faster near core),
	//  crush to a white-hot point, then detonate OUTWARD (forced vortex spin-out)
	//  and a strong spring snaps them onto shape B. With the trail texture this
	//  reads as a glowing accretion spiral collapsing to a star then exploding
	//  into the new form. Every term vanishes at morph 0 and 1 so orbs land
	//  EXACTLY on the formations (crisp attractors, no fuzzy haze).
	// =====================================================================
	let id = pcl.aux.x;
	let m = morph;                                  // 0..1 (already smoothstepped)
	let env = sin(m * 3.14159265);                  // 0 at ends, 1 mid -> exact landing
	let flash = exp(-pow((m - 0.5) / 0.08, 2.0));   // razor white-out spike at the core
	let inhale = smoothstep(0.0, 0.5, m);
	let outhale = smoothstep(0.5, 1.0, m);
	let punch = pow(env, 0.5);

	let targetA = formation_target(pcl.aux, t, shapeA);
	let targetB = formation_target(pcl.aux, t, shapeB);
	// per-particle staggered crossover so the swarm reorganises as a wave.
	let stagger = hash11(id + 3.0) * 0.30;
	let localMorph = smoothstep(stagger, stagger + 0.55, m);
	let shapePos = mix(targetA, targetB, localMorph);
	let toShape = shapePos - position;

	// swirl axis varies per shapeB so each collapse spins differently.
	var axis = vec3f(0.0, 1.0, 0.0);
	if (shapeB > 0.5 && shapeB < 1.5) { axis = vec3f(1.0, 0.0, 0.0); }
	else if (shapeB > 1.5 && shapeB < 2.5) { axis = normalize(vec3f(1.0, 1.0, 0.0)); }
	else if (shapeB > 2.5) { axis = normalize(vec3f(0.0, 1.0, 1.0)); }

	let rel = position;                              // collapse toward origin
	let axial = dot(rel, axis) * axis;
	let radial = rel - axial;
	let rr = max(length(radial), 0.06);             // CLAMP: free vortex K/r is NaN at 0
	let tangent = normalize(cross(axis, radial) + vec3f(1e-5, 0.0, 0.0));
	let radDir = radial / rr;

	let SWIRL_K = 0.55; let OMEGA = 1.4; let PULL = 1.15; let BURST = 1.35;
	// free vortex inward (faster near core) ramping to forced vortex outward.
	let swirl = (SWIRL_K / rr) * (1.0 - outhale) + (OMEGA * rr) * outhale;
	var impulse = tangent * swirl * env;
	impulse += -radDir * inhale * PULL * punch;     // suck in (phase A)
	impulse += radDir * outhale * BURST * punch;    // detonate out (phase C)

	// spring still converges EXACTLY onto shape B (env/outhale terms vanish at m=1).
	var kAttract = mix(0.16, 0.32, outhale);

	// chaotic curl only mid-transition (zero at hold so held orbs sit still).
	let transTurb = 0.05 * env;

	// Pointer: nodes flee the cursor and flare (alive interaction).
	let pointer = uniforms.pointer.xy;
	let pv = position.xy - pointer;
	let pd = max(length(pv), 0.05);
	let pforce = normalize(vec3f(pv, sin(t + id) * 0.15)) *
		(uniforms.pointer.z * uniforms.pointer.w * 0.028 / (pd * pd + 0.05));

	velocity += toShape * kAttract;
	velocity += impulse;
	velocity += curl(position * 0.45, t * 0.5) * transTurb;
	velocity += pforce * dt * 60.0;

	// damping: heavy at hold (crisp still orbs), lighter mid-transition so the
	// collapse + detonation carry momentum into long comet trails.
	let damping = mix(0.72, 0.94, env);
	velocity *= damping;
	let speed = length(velocity);
	let speedCap = mix(0.55, 2.6, env);
	if (speed > speedCap) { velocity *= speedCap / speed; }

	position += velocity * mix(0.35, 1.0, motion);

	// energy: the core BLAZES white-hot at the singularity (drives the bloom +
	// the tonemap flash), held orbs keep a gentle twinkle.
	let coreBlaze = (1.0 - clamp(rr, 0.0, 1.0)) * env * 4.5 + flash * 7.0;
	let twinkle = 0.5 + 0.5 * sin(t * 1.8 + id * 0.7);
	let held = 0.85 + twinkle * 0.55;
	let energy = clamp(
		coreBlaze + speed * 1.5 + held + exp(-pd * 2.2) * uniforms.pointer.z * 1.0 + 0.30,
		0.30, 7.0
	);

	pcl.pos = vec4f(position, energy);
	pcl.vel = vec4f(velocity, pcl.vel.w);
	particles[i] = pcl;
}
`;

	// ---------------------------------------------------------------------------
	//  RENDER SHADER
	//  Instanced billboard quads (draw(6, count)). Each particle is a NODE whose
	//  SHAPE is masked from quad-local coords using abs/max/step — square pixels,
	//  diamonds, soft blobs, shards, star cores, rectangular beads. NO ring or
	//  circle primitive (no length<r outline), NO velocity streak stretching.
	//  Premultiplied-alpha additive output for clean HDR glow.
	// ---------------------------------------------------------------------------
	const renderShader = /* wgsl */ `
struct Particle {
	pos: vec4f,
	vel: vec4f,
	aux: vec4f,
	home: vec4f,
};

struct Uniforms {
	viewport: vec4f,
	pointer: vec4f,
	time: vec4f,
	mode: vec4f,
	extra: vec4f,
};

struct VertexOut {
	@builtin(position) position: vec4f,
	@location(0) local: vec2f,
	@location(1) color: vec3f,
	@location(2) glow: f32,
	@location(3) kind: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

// 5-stop memory spectrum sampled at a position s in [0,1): magenta -> violet ->
// blue -> cyan -> emerald (wraps). s FLOWS over time so the whole cloud's color
// is never static.
fn spectrum(s: f32) -> vec3f {
	let magenta = vec3f(0.98, 0.24, 0.86);
	let violet  = vec3f(0.56, 0.34, 1.00);
	let blue    = vec3f(0.22, 0.46, 1.00);
	let cyan    = vec3f(0.14, 0.86, 0.98);
	let emerald = vec3f(0.22, 0.96, 0.58);
	let x = fract(s) * 5.0;
	if (x < 1.0) { return mix(magenta, violet, x); }
	if (x < 2.0) { return mix(violet, blue, x - 1.0); }
	if (x < 3.0) { return mix(blue, cyan, x - 2.0); }
	if (x < 4.0) { return mix(cyan, emerald, x - 3.0); }
	return mix(emerald, magenta, x - 4.0);
}

// LIVING color: the hue position drifts in time, traveling waves sweep across
// the cloud by world position, each node pulses, and high-energy nodes flare
// gold -> white-hot so explosions read like fireworks.
fn palette(seed: f32, energy: f32, world: vec3f, t: f32) -> vec3f {
	// base hue from the node's frozen seed, but ROTATING the whole wheel over time
	let drift = t * 0.06;
	// a slow traveling wave across space so bands of color ripple through the cloud
	let wave = sin(world.x * 0.9 - t * 0.8) * 0.10 + cos(world.y * 0.8 + t * 0.6) * 0.10;
	var col = spectrum(seed + drift + wave);

	// per-node pulse (alive shimmer) modulates brightness over time
	let pulse = 0.82 + 0.18 * sin(t * 2.2 + seed * 30.0);
	col = col * pulse;

	// energy flare ONLY for genuinely fast transition nodes (high threshold) so
	// HELD formations keep their pure saturated hue instead of washing to cream.
	let gold = vec3f(1.00, 0.78, 0.36);
	col = mix(col, gold, clamp((energy - 2.0) * 0.5, 0.0, 0.6));
	col = mix(col, vec3f(1.0), clamp((energy - 3.2) * 0.5, 0.0, 0.6));
	return col;
}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0, -1.0),
		vec2f( 1.0, -1.0),
		vec2f(-1.0,  1.0),
		vec2f(-1.0,  1.0),
		vec2f( 1.0, -1.0),
		vec2f( 1.0,  1.0)
	);

	let p = particles[instance_index];
	let t = uniforms.time.x;
	let aspect = uniforms.viewport.x / max(uniforms.viewport.y, 1.0);

	// gentle whole-cloud rotation so formations read volumetrically.
	let ca = cos(t * 0.05);
	let sa = sin(t * 0.05);
	let world = p.pos.xyz;
	let rotated = vec3f(
		world.x * ca - world.z * sa,
		world.y,
		world.x * sa + world.z * ca
	);
	// Camera pulled BACK (bigger depth) + a gentler clip scale so the ENTIRE
	// formation centered AND large enough to fill the frame edge-to-edge.
	let depth = 2.55 + rotated.z * 0.42;
	let clip = vec2f(rotated.x / aspect, rotated.y) / depth * 2.05;

	let energy = p.pos.w;
	// shape kind picks a size profile too (beads/shards slightly larger).
	// Smaller base so dense formations read as a fine TEXTURED swarm of distinct
	// nodes rather than fat blobs that merge into a solid white mass.
	let kind = p.aux.z;
	// burst overdraw fix: SUPPRESS sprite growth as energy rises so total covered
	// area stays bounded exactly when nodes are hot AND bunched (the fillrate cliff
	// that caused the gold-transition lag). Caps COVERAGE, not just per-sprite px.
	let eNorm = clamp((energy - 1.0) / 2.0, 0.0, 1.0); // 0 at hold .. 1 at peak burst
	let base = 1.6 + p.aux.w * 1.8 + energy * 1.6 + step(4.5, kind) * 1.0;
	let burstCap = mix(15.0, 8.0, eNorm);              // smaller cap when hot
	let baseClamped = min(base, burstCap);
	let pixel = 2.0 / max(min(uniforms.viewport.x, uniforms.viewport.y), 1.0);
	let local = corners[vertex_index];

	var out: VertexOut;
	out.position = vec4f(clip + local * baseClamped * pixel, 0.0, 1.0);
	out.local = local;
	// SEEDED-BRAIN HUE: only when the brain (shape 4) is involved (supernova path),
	// rotate every node's hue by a seed-derived offset so each user's brain has its
	// own colorway. Zero effect on the ambient attractors (shapeA/B are always < 3.5
	// in the ambient loop), so their palette is byte-identical.
	let brainActive = step(3.5, uniforms.time.z) + step(3.5, uniforms.time.w);
	let hueShift = select(0.0, fract(sin(uniforms.mode.w * 0.000137 + 5.0) * 33891.7), brainActive > 0.5);
	out.color = palette(p.aux.w + hueShift, energy, world, t);
	out.glow = energy;
	out.kind = kind;
	return out;
}

// SOFT node sprites from quad-local coords. Every node is a luminous orb: a
// bright gaussian CORE plus a soft HALO, with an anti-aliased silhouette. The
// shape kind only bends the distance metric (square/diamond/shard/bead/star/
// cross) so silhouettes vary, but the falloff is always SMOOTH — no crunchy
// hard edges, ZERO ring/circle outlines (filled falloffs only).
fn shape_mask(q: vec2f, kind: f32) -> f32 {
	// returns a soft silhouette weight in [0,1]; 1 at centre, smoothly -> 0 at rim.
	var dist: f32;
	if (kind < 0.5) {
		// rounded square (Chebyshev softened toward Euclidean)
		dist = mix(max(abs(q.x), abs(q.y)), length(q), 0.35);
	} else if (kind < 1.5) {
		// diamond (L1)
		dist = (abs(q.x) + abs(q.y)) * 0.82;
	} else if (kind < 2.5) {
		// round blob (L2)
		dist = length(q);
	} else if (kind < 3.5) {
		// shard: tall narrow
		dist = length(vec2f(q.x * 2.2, q.y * 0.75));
	} else if (kind < 4.5) {
		// star: min of axis-aligned and rotated diamonds -> 4-point soft star
		let a = abs(q.x) + abs(q.y);
		let b = abs(q.x - q.y) + abs(q.x + q.y);
		dist = min(a, b) * 0.7;
	} else if (kind < 5.5) {
		// wide bead
		dist = length(vec2f(q.x * 0.7, q.y * 1.5));
	} else {
		// plus / cross spark
		dist = min(max(abs(q.x) * 2.6, abs(q.y)), max(abs(q.x), abs(q.y) * 2.6)) * 0.9;
	}
	return smoothstep(1.05, 0.0, dist);
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
	let q = in.local;                  // quad-local in [-1,1]
	let d = length(q);                 // 0 centre .. ~1.41 corner
	// CLEAN ROUND ORB (Three.js Journey "light point"): the falloff is a radial
	// profile inside a HARD-ZERO window that closes at R < 1, so the quad CORNERS
	// (d up to 1.41) are EXACTLY 0 at any HDR brightness — the sprite can never
	// square off. Shape is decoupled from brightness: density + bloom build the
	// bright centre, the sprite itself stays dim and round.
	let R = 0.92;
	let rn = clamp(d / R, 0.0, 1.0);   // 0 centre .. 1 at disc edge, clamped past R
	let disc = 1.0 - smoothstep(0.0, 1.0, rn); // exactly 1 at centre, exactly 0 at rn>=1
	if (disc <= 0.001) { discard; }
	// Every node is the SAME clean round orb: tight bright core + wide soft tail,
	// both vanishing at the disc edge. No per-particle shape variety — uniform
	// round dots read premium (like the Three.js Points version); the beauty comes
	// from arrangement + motion, not sprite gimmicks.
	let core = pow(1.0 - rn, 3.0);
	let halo = (1.0 - rn) * 0.45;
	let profile = (core + halo) * disc;
	// LOW per-sprite peak — no flat pedestal, no big multiplier. Brightness scales
	// purely from the radial profile and node energy; bloom does the rest.
	let intensity = (0.18 + in.glow * 0.85);
	let rgb = in.color * intensity;
	return vec4f(rgb * profile, 1.0); // premultiplied, additive (one,one)
}
`;

	// ---------------------------------------------------------------------------
	//  TONEMAP PASS
	//  Fullscreen triangle reads the HDR accumulation texture and ACES-tonemaps it
	//  to the 8-bit swapchain. This is what keeps dense additive node clusters
	//  RICH and COLORED instead of clipping to white, and gives the whole frame a
	//  filmic, 2026-demo glow. Adds a faint hue lift in the shadows so the field
	//  never reads as flat black between nodes.
	// ---------------------------------------------------------------------------
	const tonemapShader = /* wgsl */ `
@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var hdrSampler: sampler;
@group(0) @binding(2) var bloomTex: texture_2d<f32>;

struct Uniforms {
	viewport: vec4f,
	pointer: vec4f,
	time: vec4f,
	mode: vec4f,
	extra: vec4f,   // x=burst y=flash
};
@group(0) @binding(3) var<uniform> fx: Uniforms;

struct VertexOut {
	@builtin(position) position: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertexOut {
	var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
	var out: VertexOut;
	let p = pos[vi];
	out.position = vec4f(p, 0.0, 1.0);
	out.uv = vec2f((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
	return out;
}

// Reinhard tonemap of a SCALAR (luminance). Compresses [0,inf) -> [0,1).
fn reinhard(x: f32) -> f32 {
	return x / (1.0 + x);
}

// RADIAL ZOOM BLUR toward centre — the "explosion rushes the camera" punch.
// textureSampleLevel (not textureSample) is MANDATORY inside the loop (WGSL
// forbids implicit derivatives in non-uniform control flow).
fn zoomBlur(uv: vec2f, c: vec2f, strength: f32) -> vec3f {
	var col = vec3f(0.0);
	let dir = uv - c;
	for (var j = 0; j < 12; j = j + 1) {
		let k = 1.0 - strength * (f32(j) / 12.0);
		col += textureSampleLevel(hdrTex, hdrSampler, dir * k + c, 0.0).rgb;
	}
	return col / 12.0;
}

// GOD RAYS — radial blur of the bright bloom toward centre. Each hot core throws
// volumetric light shafts at the supernova peak (GPU Gems 3 Ch.13, Mitchell).
fn godRays(uv: vec2f, c: vec2f, density: f32, decay: f32, weight: f32) -> vec3f {
	var coord = uv;
	let delta = (uv - c) * (1.0 / 16.0) * density;
	var col = vec3f(0.0);
	var illum = 1.0;
	for (var i = 0; i < 16; i = i + 1) {
		coord = coord - delta;
		let s = textureSampleLevel(bloomTex, hdrSampler, coord, 0.0).rgb;
		col += s * illum * weight;
		illum *= decay;
	}
	return col / 23.33;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
	let burst = fx.extra.x;   // sin(morph*PI): 0 hold .. 1 mid-morph
	let flash = fx.extra.y;   // razor spike at the singularity
	let c = vec2f(0.5);

	// SHOCKWAVE: an expanding ring displaces the sample coords as it sweeps past,
	// warping space like a pressure wave at the collapse.
	var uv = in.uv;
	if (burst > 0.01) {
		let toC = in.uv - c;
		let dist = length(toC);
		let waveR = burst * 0.9;                 // ring radius grows with the burst
		let ring = exp(-pow((dist - waveR) / 0.045, 2.0)); // gaussian shell
		uv = in.uv - normalize(toC + vec2f(1e-5)) * ring * 0.035 * burst;
	}

	var scene: vec3f;
	if (burst > 0.01) {
		// during the transition: zoom-blur streaks + radial chromatic split (the
		// chromatic amount also spikes on the shockwave ring edge).
		let s = 0.34 * burst;
		scene = zoomBlur(uv, c, s);
		let dir = uv - c;
		let off = dir * 0.014 * burst;
		scene.r = zoomBlur(uv + off, c, s).r;
		scene.b = zoomBlur(uv - off, c, s).b;
	} else {
		scene = textureSample(hdrTex, hdrSampler, uv).rgb; // cheap path at rest
	}
	let bloom = textureSample(bloomTex, hdrSampler, uv).rgb;
	// Combine bloom in LINEAR HDR BEFORE tonemapping; bloom strength SPIKES at the
	// singularity flash so the core blows to a white-hot supernova.
	var hdr = scene + bloom * (0.35 * (1.0 + 1.5 * flash));

	// GOD RAYS at the flash peak — light shafts streaking from the white-hot core.
	if (flash > 0.02) {
		hdr += godRays(uv, c, 0.9, 0.8836, 0.65) * flash * 1.4;
	}
	// ANAMORPHIC blue lens streak — horizontal blur of the bloom, tinted blue.
	if (burst > 0.01) {
		var streak = vec3f(0.0);
		for (var k = -5; k <= 5; k = k + 1) {
			streak += textureSampleLevel(bloomTex, hdrSampler, uv + vec2f(f32(k) * 0.020, 0.0), 0.0).rgb;
		}
		hdr += (streak / 11.0) * vec3f(0.35, 0.6, 1.0) * burst * 0.7;
	}

	// exposure spike at the collapse — rolls off filmically through Reinhard.
	hdr = hdr * (1.0 + 3.0 * flash);
	let exposed = hdr * 0.26;
	// HUE-PRESERVING tonemap: tonemap the LUMINANCE only, then scale the original
	// chroma by the tonemapped/original luminance ratio. Dense additive clusters
	// stay deeply CYAN / VIOLET / EMERALD at full density instead of whiting out.
	let lum = max(dot(exposed, vec3f(0.2126, 0.7152, 0.0722)), 1e-4);
	let toned = reinhard(lum * 1.1);
	var color = exposed * (toned / lum);
	// STRONG saturation boost so the spectrum is vivid and electric, never cream.
	let gray = dot(color, vec3f(0.299, 0.587, 0.114));
	color = clamp(gray + (color - vec3f(gray)) * 2.1, vec3f(0.0), vec3f(1.0));
	color = pow(color, vec3f(0.90));
	// VIGNETTE PULSE — tighten the frame edges during the burst so the eye locks
	// onto the supernova core (capped so the flash never gets swallowed).
	let vig = 1.0 - smoothstep(0.45, 1.25, length(in.uv - vec2f(0.5)) * (1.0 + 0.5 * burst));
	color = color * mix(1.0, vig, 0.35 * burst);
	// sub-LSB dither breaks 8-bit banding in the dark field between nodes.
	let dither = (fract(sin(dot(in.position.xy, vec2f(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
	let base = vec3f(0.010, 0.014, 0.030);
	return vec4f(base + color + dither, 1.0);
}
`;

	// ---------------------------------------------------------------------------
	//  BLOOM (dual-Kawase) — runs in linear HDR before the tonemap. A few
	//  downsample-blur passes then upsample-combine passes produce a soft colored
	//  glow around bright nodes. No threshold (blooms the whole HDR buffer) so the
	//  glow stays the node's own hue, and the tonemap combines it weakly.
	// ---------------------------------------------------------------------------
	const bloomShader = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VO {
	var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
	var out: VO;
	let v = p[vi];
	out.pos = vec4f(v, 0.0, 1.0);
	out.uv = vec2f((v.x + 1.0) * 0.5, (1.0 - v.y) * 0.5);
	return out;
}

// 5-tap downsample: center weighted, 4 corners. A pre-bloom clamp bounds how hot
// one super-bright node can drive the mip pyramid, keeping bloom cost flat during
// the bright burst (so the gold transition doesn't spike fillrate).
@fragment
fn down(in: VO) -> @location(0) vec4f {
	let texel = vec2f(1.0) / vec2f(textureDimensions(src));
	let o = texel * 1.0;
	var c = textureSample(src, samp, in.uv) * 4.0;
	c += textureSample(src, samp, in.uv + vec2f(-o.x, -o.y));
	c += textureSample(src, samp, in.uv + vec2f( o.x, -o.y));
	c += textureSample(src, samp, in.uv + vec2f(-o.x,  o.y));
	c += textureSample(src, samp, in.uv + vec2f( o.x,  o.y));
	return min(c / 8.0, vec4f(8.0));
}

// 8-tap tent upsample (widens the glow).
@fragment
fn up(in: VO) -> @location(0) vec4f {
	let texel = vec2f(1.0) / vec2f(textureDimensions(src));
	let o = texel * 1.0;
	var c = vec4f(0.0);
	c += textureSample(src, samp, in.uv + vec2f(-o.x * 2.0, 0.0));
	c += textureSample(src, samp, in.uv + vec2f( o.x * 2.0, 0.0));
	c += textureSample(src, samp, in.uv + vec2f(0.0, -o.y * 2.0));
	c += textureSample(src, samp, in.uv + vec2f(0.0,  o.y * 2.0));
	c += textureSample(src, samp, in.uv + vec2f(-o.x, o.y)) * 2.0;
	c += textureSample(src, samp, in.uv + vec2f( o.x, o.y)) * 2.0;
	c += textureSample(src, samp, in.uv + vec2f(-o.x, -o.y)) * 2.0;
	c += textureSample(src, samp, in.uv + vec2f( o.x, -o.y)) * 2.0;
	return c / 12.0;
}
`;

	// ---------------------------------------------------------------------------
	//  TRAIL FADE — the motion-trail accumulation. Each frame, before drawing the
	//  orbs, this multiplies the trail texture by a decay constant (set via
	//  setBlendConstant) so the previous frame's light dims but doesn't vanish.
	//  Orbs then draw additively ON TOP, leaving glowing comet tails when they move
	//  fast (during a transition) and tight orbs when still (at hold). NODE-ONLY:
	//  the streaks are the orbs' OWN accumulated light, not drawn line primitives.
	// ---------------------------------------------------------------------------
	const fadeShader = /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
	var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
	return vec4f(p[vi], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4f {
	// output is ignored for color; blend = dst * constant (see fade pipeline).
	return vec4f(1.0);
}
`;

	// XorShift PRNG, deterministic per seed.
	function random(seedValue: number) {
		let state = seedValue >>> 0 || 1;
		return () => {
			state ^= state << 13;
			state ^= state >>> 17;
			state ^= state << 5;
			return ((state >>> 0) % 1000000) / 1000000;
		};
	}

	// Build particles starting on an off-screen edge shell (so STREAM flies in)
	// with seed/lane/kind/hue and a dissolve home point baked in.
	function buildParticles(count: number) {
		const rand = random(seed);
		const data = new Float32Array(count * 16); // 4 vec4f per particle
		for (let i = 0; i < count; i += 1) {
			const o = i * 16;
			// edge-shell spawn (one of 4 edges, just off screen)
			const edge = Math.floor(rand() * 4);
			const tEdge = rand();
			let hx = 0;
			let hy = 0;
			const hw = 2.6;
			const hh = 1.9;
			if (edge === 0) {
				hx = -hw;
				hy = (tEdge * 2 - 1) * hh;
			} else if (edge === 1) {
				hx = hw;
				hy = (tEdge * 2 - 1) * hh;
			} else if (edge === 2) {
				hy = hh;
				hx = (tEdge * 2 - 1) * hw;
			} else {
				hy = -hh;
				hx = (tEdge * 2 - 1) * hw;
			}
			const hz = (rand() - 0.4) * 1.2;
			// bias kind toward SOFT orbs (blob/star) for a premium look; keep the
			// geometric kinds (diamond/bead/shard/square/cross) as a sparse accent.
			const rk = rand();
			const kind =
				rk < 0.4
					? 2 // soft blob
					: rk < 0.62
						? 4 // star core
						: rk < 0.76
							? 1 // diamond
							: rk < 0.86
								? 5 // bead
								: rk < 0.94
									? 3 // shard
									: rk < 0.98
										? 0 // square pixel
										: 6; // cross spark

			// pos: start on the shell
			data[o] = hx;
			data[o + 1] = hy;
			data[o + 2] = hz;
			data[o + 3] = rand() * 0.4; // energy

			// vel: zero, with staggered spawn delay in .w
			data[o + 4] = 0;
			data[o + 5] = 0;
			data[o + 6] = 0;
			data[o + 7] = 0.10 * rand() + 0.08 * ((edge + tEdge) / 4);

			// aux: id, lane, kind, hue seed
			data[o + 8] = i + rand() * 17;
			data[o + 9] = i % 7;
			data[o + 10] = kind;
			data[o + 11] = rand();

			// home: dissolve target == spawn shell, perimeter t in .w
			data[o + 12] = hx;
			data[o + 13] = hy;
			data[o + 14] = hz;
			data[o + 15] = (edge + tEdge) / 4;
		}
		return data;
	}

	function resizeCanvas(canvas: HTMLCanvasElement, maxDpr = 2) {
		const rect = canvas.getBoundingClientRect();
		const viewport = window.visualViewport;
		const cssWidth =
			rect.width > 1 ? rect.width : viewport?.width || window.innerWidth || canvas.clientWidth || 1;
		const cssHeight =
			rect.height > 1 ? rect.height : viewport?.height || window.innerHeight || canvas.clientHeight || 1;
		const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
		const width = Math.max(1, Math.floor(cssWidth * dpr));
		const height = Math.max(1, Math.floor(cssHeight * dpr));
		let changed = false;
		if (canvas.width !== width || canvas.height !== height) {
			canvas.width = width;
			canvas.height = height;
			changed = true;
		}
		return { width, height, dpr, changed };
	}

	// Number of formations in the morph cycle (VESTIGE wordmark, Aizawa, Thomas,
	// supershape, Clifford torus). Each HOLDS fully formed for HOLD_SECONDS, then
	// the dramatic singularity-collapse MORPH into the next takes MORPH_SECONDS.
	const SHAPE_COUNT = 4;
	const HOLD_SECONDS = 5.0;
	const MORPH_SECONDS = 2.0;
	const SLOT_SECONDS = HOLD_SECONDS + MORPH_SECONDS; // 7s per formation
	const LOOP = SHAPE_COUNT * SLOT_SECONDS;

	// ---- SUPERNOVA-ON-SIGNUP state machine ----------------------------------
	// On signup success the hero DECOUPLES from the ambient time cycle: it morphs
	// from whatever shape is current INTO the seeded brain (shape 4), flashes once
	// at the singularity, eases the reform, then FREEZES on the user's brain forever
	// (never returns to the ambient loop). reduced-motion takes a gentle no-flash
	// reveal instead. The brain is seeded from supernovaSeed (the email hash).
	const BRAIN_SHAPE = 4; // index in formation_target; NOT in the ambient SHAPE_COUNT loop
	const SUPERNOVA_RAMP = 1.2; // seconds 0->1 morph into the brain
	let supernovaActive = false; // true once a signup has fired
	let supernovaStart = 0; // performance.now() at trigger
	let supernovaFromShape = 0; // shapeA frozen at trigger (whatever was current)
	let supernovaSeed = 20260625; // per-user seed (email hash); defaults to hero seed
	let supernovaReduced = false; // snapshot of reducedMotion at trigger time

	// Public trigger. Called by the window 'vestige:supernova' listener (see onMount).
	// Captures the CURRENT ambient shapeA so the morph starts from what's on screen.
	function triggerSupernova(seedValue: number) {
		if (supernovaActive) {
			// already detonated — only update the seed (re-freeze on the new brain).
			if (Number.isFinite(seedValue)) supernovaSeed = seedValue >>> 0;
			return;
		}
		const tNow = (performance.now() - startedAt) / 1000;
		const cycle = tNow / SLOT_SECONDS;
		supernovaFromShape = Math.floor(cycle) % SHAPE_COUNT; // current ambient shapeA
		supernovaSeed = Number.isFinite(seedValue) ? seedValue >>> 0 : supernovaSeed;
		supernovaReduced = reducedMotion;
		supernovaStart = performance.now();
		supernovaActive = true;
	}

	function writeUniforms(now: number, delta: number) {
		if (!gpu || !gpuCanvas) return;
		const { width, height, dpr, changed } = resizeCanvas(gpuCanvas, reducedMotion ? 1.25 : 2);
		if (changed) {
			gpu.context.configure({
				device: gpu.device,
				format: gpu.format,
				alphaMode: 'opaque'
			});
		}
		const time = (now - startedAt) / 1000;

		// CONTINUOUS SHAPE-TO-SHAPE MORPH (no edge-shell stream/dissolve, no burst
		// gap). The loop walks through SHAPE_COUNT formations; each gets one slot of
		// HOLD then a MORPH straight into the next. shapeA/shapeB/morphBlend drive
		// the compute shader so a formation dissolves directly INTO the next one.
		const cycle = time / SLOT_SECONDS; // fractional formation index, rising
		let shapeA = Math.floor(cycle) % SHAPE_COUNT;
		let shapeB = (shapeA + 1) % SHAPE_COUNT;
		const elapsed = (time % SLOT_SECONDS); // seconds into this formation's slot
		// HOLD fully formed for HOLD_SECONDS, THEN the dramatic explode+reform morph
		// over MORPH_SECONDS. morphBlend stays exactly 0 for the whole 5s hold.
		let morphBlend =
			elapsed < HOLD_SECONDS ? 0 : (elapsed - HOLD_SECONDS) / MORPH_SECONDS;
		// smoothstep the morph so it eases in/out, not a linear slide.
		morphBlend = morphBlend * morphBlend * (3 - 2 * morphBlend);

		if (reducedMotion) {
			// freeze fully formed on the SEEDED BRAIN (shape 4), no morphing.
			// (was Thomas=1; the brain is the signup destination so reduced-motion
			// users still see THEIR brain, just without any motion.)
			shapeA = BRAIN_SHAPE;
			shapeB = BRAIN_SHAPE;
			morphBlend = 0;
		}

		// SUPERNOVA OVERRIDE: once a signup fires, decouple from the time cycle and
		// drive a one-shot morph INTO the seeded brain, then FREEZE there forever.
		// `supernovaReveal` (0 rest .. 1 active) is mirrored to extra.z so the tonemap
		// can suppress the flash for reduced-motion users (no god-ray spike).
		let supernovaReveal = 0;
		if (supernovaActive) {
			supernovaReveal = 1;
			const sn = (now - supernovaStart) / 1000; // seconds since trigger
			let p = Math.min(Math.max(sn / SUPERNOVA_RAMP, 0), 1); // 0..1 linear
			p = p * p * (3 - 2 * p); // smoothstep ease (matches the ambient morph feel)
			shapeA = supernovaFromShape; // start from what was on screen
			shapeB = BRAIN_SHAPE; // ... collapse into the seeded brain
			morphBlend = p; // 0 -> 1 then PINNED at 1 (sn>=RAMP -> p clamps to 1)
			// reduced-motion: morphBlend still 0->1 so the brain forms, but extra.z=1
			// and the burst/flash recompute below hard-zero the singularity FX.
		}

		// viewport
		uniformData[0] = width;
		uniformData[1] = height;
		uniformData[2] = dpr;
		uniformData[3] = reducedMotion ? 1 : 0;
		// pointer
		uniformData[4] = pointerX;
		uniformData[5] = pointerY;
		uniformData[6] = pointerActive;
		uniformData[7] = pointerPressure;
		// time
		uniformData[8] = time;
		uniformData[9] = delta;
		uniformData[10] = shapeA;
		uniformData[11] = shapeB;
		// burst envelope (0 at hold/ends, 1 mid-morph) + razor flash spike. These
		// drive BOTH the compute vortex AND the tonemap post-FX so the whole stack
		// peaks on the same frame. Declared `let` because the supernova override
		// (above) reassigns morphBlend; recompute from the FINAL value below.
		let burst = Math.sin(Math.min(Math.max(morphBlend, 0), 1) * Math.PI);
		let flash = Math.exp(-Math.pow((morphBlend - 0.5) / 0.07, 2));
		// For reduced-motion supernova, hard-zero the flash so there is no god-ray
		// photosensitivity spike (gentle seeded-brain cross-fade only).
		if (supernovaActive && supernovaReduced) {
			burst = 0;
			flash = 0;
		}
		// frame-rate-independent trail decay: short tails at hold (crisp orbs),
		// long comet tails at the morph peak. Capped < 0.95 so it never runs away.
		const dt60 = Math.min(delta, 0.033) * 60;
		trailDecay = Math.pow(0.84 + 0.10 * burst, dt60);

		// mode
		uniformData[12] = gpu.particleCount;
		uniformData[13] = morphBlend;
		uniformData[14] = burst;
		// mode.w = the seed read by brain_target. Ambient: the hero seed (attractors
		// ignore it, so this is a no-op for them). Supernova: the user's email hash,
		// so the frozen brain is THEIRS.
		uniformData[15] = supernovaActive ? supernovaSeed : seed;
		// extra: x=burst y=flash (tonemap post-FX), z=supernovaReveal (reserved flag).
		uniformData[16] = burst;
		uniformData[17] = flash;
		uniformData[18] = supernovaReveal; // was window.scrollY (read by no shader)
		uniformData[19] = 0;
		gpu.device.queue.writeBuffer(
			gpu.uniformBuffer,
			0,
			uniformData.buffer,
			uniformData.byteOffset,
			uniformData.byteLength
		);

		// Mirror the SAME burst/flash scalars to the DOM (scoped to .launch-shell)
		// so the overlay implodes/explodes in perfect sync with the particles. When
		// suppressed (form focused) or reduced-motion, pin to rest so a signup is
		// never disturbed. Frame-perfect: same call site as the GPU uniform write.
		if (syncTarget) {
			const s = syncTarget.style;
			if (suppress || reducedMotion) {
				s.setProperty('--burst', '0');
				s.setProperty('--flash', '0');
			} else {
				s.setProperty('--burst', burst.toFixed(4));
				s.setProperty('--flash', flash.toFixed(4));
			}
		}
	}

	function pickParticleCount(adapter: any): number {
		const small = window.innerWidth < 760;
		const limits = adapter?.limits ?? {};
		// device tier from fallback flag + storage/compute budget + core count.
		const maxStorage = limits.maxStorageBufferBindingSize ?? 134217728;
		const maxInvocations = limits.maxComputeInvocationsPerWorkgroup ?? 256;
		const cores = navigator.hardwareConcurrency || 8;
		const isFallback = adapter?.info?.isFallbackAdapter === true;
		const weak =
			isFallback ||
			maxInvocations < 256 ||
			(window.devicePixelRatio || 1) > 2.2 ||
			cores <= 4;
		const big = maxStorage >= 268435456 && !weak;

		// Counts tuned for additive density: enough to feel like a teeming brain,
		// not so many that thin 2D formations (the wordmark) saturate to white.
		let count: number;
		if (weak) count = small ? 20000 : 42000;
		else if (big) count = small ? 40000 : 75000;
		else count = small ? 32000 : 60000;
		if (reducedMotion) count = Math.floor(count * 0.6);
		// each particle = 64 bytes (4 vec4f). cap to the storage budget with margin.
		const maxByBudget = Math.floor((maxStorage * 0.6) / 64);
		return Math.max(8000, Math.min(count, maxByBudget));
	}

	async function bootWebGPU() {
		if (!gpuCanvas) return false;
		const nav = navigator as Navigator & { gpu?: any };
		const gpuApi = nav.gpu;
		const globals = globalThis as unknown as {
			GPUBufferUsage: Record<string, number>;
			GPUShaderStage: Record<string, number>;
		};
		if (!gpuApi || !globals.GPUBufferUsage) return false;

			const adapter: any = await requestBestAdapter(gpuApi);
			if (!adapter) return false;

		// Request the adapter's OWN storage limit back verbatim: never exceeds (so
		// requestDevice won't reject) but lifts us off the lower spec default so a
		// large particle buffer is permitted.
		const requiredLimits: Record<string, number> = {};
		if (adapter.limits?.maxStorageBufferBindingSize) {
			requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
		}
		if (adapter.limits?.maxBufferSize) {
			requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
		}
			const device: any = await requestBestDevice(adapter, requiredLimits);
		const context = gpuCanvas.getContext('webgpu');
		if (!context) return false;

		const format = gpuApi.getPreferredCanvasFormat();
		const particleCount = pickParticleCount(adapter);
		const particleData = buildParticles(particleCount);
		const usage = globals.GPUBufferUsage;

		const particleBuffer = device.createBuffer({
			label: 'vestige memory particles',
			size: particleData.byteLength,
			usage: usage.STORAGE | usage.COPY_DST
		});
		device.queue.writeBuffer(particleBuffer, 0, particleData);

		const uniformBuffer = device.createBuffer({
			label: 'vestige uniforms',
			size: uniformData.byteLength,
			usage: usage.UNIFORM | usage.COPY_DST
		});

		// rgba16float HDR accumulation target so additive node light SUMS in float
		// without clipping; the tonemap pass compresses it to the 8-bit swapchain.
		const hdrFormat = 'rgba16float';

		const computeModule = device.createShaderModule({
			label: 'vestige compute wgsl',
			code: computeShader
		});
		const renderModule = device.createShaderModule({
			label: 'vestige render wgsl',
			code: renderShader
		});
		const tonemapModule = device.createShaderModule({
			label: 'vestige tonemap wgsl',
			code: tonemapShader
		});
		const bloomModule = device.createShaderModule({
			label: 'vestige bloom wgsl',
			code: bloomShader
		});
		const fadeModule = device.createShaderModule({
			label: 'vestige fade wgsl',
			code: fadeShader
		});

			const fadePipeline: any = await withTimeout(
				device.createRenderPipelineAsync({
					label: 'vestige trail fade pipeline',
					layout: 'auto',
					vertex: { module: fadeModule, entryPoint: 'vs' },
					fragment: {
						module: fadeModule,
						entryPoint: 'fs',
						targets: [
							{
								format: hdrFormat,
								// dst' = dst * constant  (fade the trail by the blend constant)
								blend: {
									color: { srcFactor: 'zero', dstFactor: 'constant', operation: 'add' },
									alpha: { srcFactor: 'zero', dstFactor: 'constant', operation: 'add' }
								}
							}
						]
					},
					primitive: { topology: 'triangle-list' }
				}),
				3500,
				'WebGPU fade pipeline'
			);

			const computePipeline: any = await withTimeout(
				device.createComputePipelineAsync({
					label: 'vestige compute pipeline',
					layout: 'auto',
					compute: { module: computeModule, entryPoint: 'main' }
				}),
				3500,
				'WebGPU compute pipeline'
			);

			const renderPipeline: any = await withTimeout(
				device.createRenderPipelineAsync({
					label: 'vestige additive node pipeline',
					layout: 'auto',
					vertex: { module: renderModule, entryPoint: 'vs' },
					fragment: {
						module: renderModule,
						entryPoint: 'fs',
						targets: [
							{
								format: hdrFormat,
								// pure additive into HDR float: light energy accumulates unclamped.
								blend: {
									color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
									alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
								}
							}
						]
					},
					primitive: { topology: 'triangle-list' },
					multisample: { count: MSAA_SAMPLES }
				}),
				3500,
				'WebGPU particle pipeline'
			);

		// Shared explicit bind group layout for BOTH bloom passes so their bind
		// groups are interchangeable (a 'auto' layout would make them incompatible).
		const bloomBGL = device.createBindGroupLayout({
			label: 'vestige bloom bgl',
			entries: [
				{ binding: 0, visibility: globals.GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 1, visibility: globals.GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }
			]
		});
		const bloomPipelineLayout = device.createPipelineLayout({
			label: 'vestige bloom pipeline layout',
			bindGroupLayouts: [bloomBGL]
		});

			const bloomDownPipeline: any = await withTimeout(
				device.createRenderPipelineAsync({
					label: 'vestige bloom down pipeline',
					layout: bloomPipelineLayout,
					vertex: { module: bloomModule, entryPoint: 'vs' },
					fragment: { module: bloomModule, entryPoint: 'down', targets: [{ format: hdrFormat }] },
					primitive: { topology: 'triangle-list' }
				}),
				3500,
				'WebGPU bloom down pipeline'
			);

			const bloomUpPipeline: any = await withTimeout(
				device.createRenderPipelineAsync({
					label: 'vestige bloom up pipeline',
					layout: bloomPipelineLayout,
					vertex: { module: bloomModule, entryPoint: 'vs' },
					fragment: {
						module: bloomModule,
						entryPoint: 'up',
						targets: [
							{
								format: hdrFormat,
								// additive so each upsample LEVEL accumulates onto the level below
								// (progressive widening of the glow up the chain).
								blend: {
									color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
									alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
								}
							}
						]
					},
					primitive: { topology: 'triangle-list' }
				}),
				3500,
				'WebGPU bloom up pipeline'
			);

			const tonemapPipeline: any = await withTimeout(
				device.createRenderPipelineAsync({
					label: 'vestige tonemap pipeline',
					layout: 'auto',
					vertex: { module: tonemapModule, entryPoint: 'vs' },
					fragment: {
						module: tonemapModule,
						entryPoint: 'fs',
						targets: [{ format }]
					},
					primitive: { topology: 'triangle-list' }
				}),
				3500,
				'WebGPU tonemap pipeline'
			);

		const hdrSampler = device.createSampler({
			label: 'vestige hdr sampler',
			magFilter: 'linear',
			minFilter: 'linear'
		});

		const computeBindGroup = device.createBindGroup({
			label: 'vestige compute bind group',
			layout: computePipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: particleBuffer } },
				{ binding: 1, resource: { buffer: uniformBuffer } }
			]
		});

		const renderBindGroup = device.createBindGroup({
			label: 'vestige render bind group',
			layout: renderPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: particleBuffer } },
				{ binding: 1, resource: { buffer: uniformBuffer } }
			]
		});

		context.configure({
			device,
			format,
			alphaMode: 'opaque'
		});

		gpu = {
			device,
			context,
			format,
			hdrFormat,
			particleBuffer,
			uniformBuffer,
			computePipeline,
			renderPipeline,
			tonemapPipeline,
			bloomDownPipeline,
			bloomUpPipeline,
			bloomBGL,
			fadePipeline,
			trailTexture: null,
			trailView: null,
			trailPrimed: false,
			computeBindGroup,
			renderBindGroup,
			tonemapBindGroup: null,
			hdrTexture: null,
			hdrView: null,
			hdrMsaaTexture: null,
			hdrMsaaView: null,
			hdrSampler,
			bloomMips: [],
			bloomBindGroups: [],
			hdrWidth: 0,
			hdrHeight: 0,
			particleCount
		};

			device.lost.then((info: { message?: string }) => {
				if (disposed) return;
				if (gpu?.device !== device) return;
				console.warn('[launch] WebGPU device lost:', info?.message ?? 'unknown');
				bootFallback(1200);
			});

			return true;
		}

	// (re)create the HDR render target, the 4x MSAA target, and the bloom mip chain
	// when the canvas size changes. Size-guarded so steady-state allocates NOTHING
	// per frame (per-frame createTexture mid-burst is a hitch we must avoid).
	function ensureHdrTarget(canvasW: number, canvasH: number) {
		if (!gpu) return;
		// particle/HDR/bloom all run at HALF resolution to kill burst overdraw.
		const width = Math.max(1, Math.floor(canvasW / PARTICLE_SCALE));
		const height = Math.max(1, Math.floor(canvasH / PARTICLE_SCALE));
		if (gpu.hdrTexture && gpu.hdrWidth === width && gpu.hdrHeight === height) return;

		const globals = globalThis as unknown as { GPUTextureUsage: Record<string, number> };
		const tu = globals.GPUTextureUsage;

		// tear down old textures
		gpu.hdrTexture?.destroy?.();
		gpu.hdrMsaaTexture?.destroy?.();
		gpu.trailTexture?.destroy?.();
		for (const m of gpu.bloomMips) m.texture?.destroy?.();
		gpu.bloomMips = [];

		// single-sample HDR resolve target (sampled by bloom + tonemap)
		gpu.hdrTexture = gpu.device.createTexture({
			label: 'vestige hdr resolve',
			size: { width, height },
			format: gpu.hdrFormat,
			usage: tu.RENDER_ATTACHMENT | tu.TEXTURE_BINDING
		});
		gpu.hdrView = gpu.hdrTexture.createView();

		// TRAIL accumulation target — particles draw here additively ON TOP of the
		// faded previous frame, producing motion-trail comet tails. Bloom + tonemap
		// read this instead of a freshly-cleared hdr. Needs one clean clear after
		// (re)allocation, then load forever.
		gpu.trailTexture = gpu.device.createTexture({
			label: 'vestige trail accum',
			size: { width, height },
			format: gpu.hdrFormat,
			usage: tu.RENDER_ATTACHMENT | tu.TEXTURE_BINDING
		});
		gpu.trailView = gpu.trailTexture.createView();
		gpu.trailPrimed = false;

		// optional MSAA render target (particles render here, resolve into hdrTexture)
		if (USE_MSAA) {
			gpu.hdrMsaaTexture = gpu.device.createTexture({
				label: 'vestige hdr msaa',
				size: { width, height },
				format: gpu.hdrFormat,
				sampleCount: MSAA_SAMPLES,
				usage: tu.RENDER_ATTACHMENT
			});
			gpu.hdrMsaaView = gpu.hdrMsaaTexture.createView();
		} else {
			gpu.hdrMsaaTexture = null;
			gpu.hdrMsaaView = null;
		}

		// bloom mip chain (each half the previous), all rgba16float.
		let mw = width;
		let mh = height;
		for (let i = 0; i < BLOOM_LEVELS; i += 1) {
			mw = Math.max(1, Math.floor(mw / 2));
			mh = Math.max(1, Math.floor(mh / 2));
			const tex = gpu.device.createTexture({
				label: `vestige bloom mip ${i}`,
				size: { width: mw, height: mh },
				format: gpu.hdrFormat,
				usage: tu.RENDER_ATTACHMENT | tu.TEXTURE_BINDING
			});
			gpu.bloomMips.push({ texture: tex, view: tex.createView(), width: mw, height: mh });
		}

		// per-pass source bind groups: down chain reads hdr then each mip; up chain
		// reads the smaller mip. We build them fresh against the source views.
		const g = gpu;
		const makeSrcBG = (srcView: any) =>
			g.device.createBindGroup({
				layout: g.bloomBGL,
				entries: [
					{ binding: 0, resource: srcView },
					{ binding: 1, resource: g.hdrSampler }
				]
			});
		gpu.bloomBindGroups = [];
		// downsample sources: TRAIL -> mip0 -> mip1 -> ... (reads the level above).
		// The trail texture (accumulated orbs) is what bloom + tonemap sample.
		gpu.bloomBindGroups.push(makeSrcBG(gpu.trailView));
		for (let i = 0; i < BLOOM_LEVELS - 1; i += 1) {
			gpu.bloomBindGroups.push(makeSrcBG(gpu.bloomMips[i].view));
		}
		// upsample sources: read from the smaller mip going back up
		for (let i = BLOOM_LEVELS - 1; i > 0; i -= 1) {
			gpu.bloomBindGroups.push(makeSrcBG(gpu.bloomMips[i].view));
		}

		gpu.hdrWidth = width;
		gpu.hdrHeight = height;

		// tonemap reads trail (binding 0) + bloom mip0 (binding 2) + uniforms (3).
		gpu.tonemapBindGroup = gpu.device.createBindGroup({
			label: 'vestige tonemap bind group',
			layout: gpu.tonemapPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: gpu.trailView },
				{ binding: 1, resource: gpu.hdrSampler },
				{ binding: 2, resource: gpu.bloomMips[0].view },
				{ binding: 3, resource: { buffer: gpu.uniformBuffer } }
			]
		});
	}

	function drawWebGPU(now: number) {
		if (!gpu || disposed) return;
		frame = requestAnimationFrame(drawWebGPU);
		const delta = lastFrame ? Math.min((now - lastFrame) / 1000, 0.033) : 0.016;
		lastFrame = now;
		pointerActive *= 0.96;

		try {
			writeUniforms(now, delta);
			ensureHdrTarget(gpuCanvas!.width, gpuCanvas!.height);

			// Single command encoder for the whole frame (implicit hazard sync within
			// a submission; fewer submits = less driver overhead).
			const encoder = gpu.device.createCommandEncoder({ label: 'vestige frame' });
			const computePass = encoder.beginComputePass({ label: 'vestige compute pass' });
			computePass.setPipeline(gpu.computePipeline);
			computePass.setBindGroup(0, gpu.computeBindGroup);
			computePass.dispatchWorkgroups(Math.ceil(gpu.particleCount / WORKGROUP_SIZE));
			computePass.end();

			// PASS 1a: FADE the trail texture (dst *= trailDecay). loadOp 'load' keeps
			// last frame's light; only the very first frame after (re)alloc clears it.
			const fadePass = encoder.beginRenderPass({
				label: 'vestige trail fade',
				colorAttachments: [
					{
						view: gpu.trailView,
						clearValue: { r: 0, g: 0, b: 0, a: 1 },
						loadOp: gpu.trailPrimed ? 'load' : 'clear',
						storeOp: 'store'
					}
				]
			});
			gpu.trailPrimed = true;
			fadePass.setPipeline(gpu.fadePipeline);
			fadePass.setBlendConstant({ r: trailDecay, g: trailDecay, b: trailDecay, a: trailDecay });
			fadePass.draw(3);
			fadePass.end();

			// PASS 1b: draw the orbs ADDITIVELY on top of the faded trail. loadOp
			// 'load' is MANDATORY — 'clear' would erase the trail and kill the effect.
			const partPass = encoder.beginRenderPass({
				label: 'vestige particle pass',
				colorAttachments: [{ view: gpu.trailView, loadOp: 'load', storeOp: 'store' }]
			});
			partPass.setPipeline(gpu.renderPipeline);
			partPass.setBindGroup(0, gpu.renderBindGroup);
			partPass.draw(6, gpu.particleCount);
			partPass.end();

			// PASS 2: BLOOM. Downsample hdr -> mip0 -> mip1 ... then upsample back,
			// each into rgba16float so the glow stays colored.
			let bgIndex = 0;
			for (let i = 0; i < BLOOM_LEVELS; i += 1) {
				const dst = gpu.bloomMips[i];
				const pass = encoder.beginRenderPass({
					label: `bloom down ${i}`,
					colorAttachments: [
						{ view: dst.view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }
					]
				});
				pass.setPipeline(gpu.bloomDownPipeline);
				pass.setBindGroup(0, gpu.bloomBindGroups[bgIndex]);
				bgIndex += 1;
				pass.draw(3);
				pass.end();
			}
			// upsample: combine smaller mips back up into mip0 (additive load).
			for (let i = BLOOM_LEVELS - 1; i > 0; i -= 1) {
				const dst = gpu.bloomMips[i - 1];
				const pass = encoder.beginRenderPass({
					label: `bloom up ${i}`,
					colorAttachments: [
						{ view: dst.view, loadOp: 'load', storeOp: 'store' }
					]
				});
				pass.setPipeline(gpu.bloomUpPipeline);
				pass.setBindGroup(0, gpu.bloomBindGroups[bgIndex]);
				bgIndex += 1;
				pass.draw(3);
				pass.end();
			}

				// PASS 3: tonemap hdr + bloom onto the full-res swapchain.
				const tonePass = encoder.beginRenderPass({
				label: 'vestige tonemap pass',
				colorAttachments: [
					{
						view: gpu.context.getCurrentTexture().createView(),
						clearValue: { r: 0, g: 0, b: 0, a: 1 },
						loadOp: 'clear',
						storeOp: 'store'
					}
				]
			});
			tonePass.setPipeline(gpu.tonemapPipeline);
			tonePass.setBindGroup(0, gpu.tonemapBindGroup);
			tonePass.draw(3);
			tonePass.end();

				gpu.device.queue.submit([encoder.finish()]);
				if (mode !== 'webgpu') {
					mode = 'webgpu';
					cancelAnimationFrame(projectionFrame);
					projectionFrame = 0;
				}
			} catch (error) {
				console.warn('[launch] WebGPU frame failed, falling back:', error);
				bootFallback(1200);
			}
		}

	// ---------------------------------------------------------------------------
	//  CANVAS2D FALLBACK — NODE ONLY.
	//  Draws ONLY filled nodes (fillRect for squares/beads, filled triangles for
	//  diamonds/shards via path fill — NEVER stroke/arc/ellipse/moveTo-lineTo
	//  edges/rings). No connections, no lines, no rings, no scanlines.
	// ---------------------------------------------------------------------------
	type FallbackParticle = {
		hx: number;
		hy: number;
		x: number;
		y: number;
		vx: number;
		vy: number;
		kind: number;
		seed: number;
		delay: number;
		lane: number;
	};

	function hslNode(seed: number, energy: number): string {
		// same spectrum family as the GPU palette, expressed in HSL.
		const hue = 300 - seed * 180; // magenta(300) -> emerald(120) range
		const light = 56 + energy * 22;
		return `hsl(${hue}, 95%, ${Math.min(light, 82)}%)`;
	}

	function startFallback() {
		if (disposed || !fallbackCanvas) return;
		if (fallbackStarted && projectionFrame) return;
		fallbackStarted = true;
		const canvas = fallbackCanvas;
		const rawContext = canvas.getContext('2d');
		if (!rawContext) return;
		const ctx: CanvasRenderingContext2D = rawContext;
		const rand = random(seed ^ 0x1234abcd);
		const isMobile = window.innerWidth < 760;
		const count = isMobile ? 820 : 2600;
		const fallbackStartedAt = performance.now();

		// glyph anchors (node-only wordmark) in normalised [-1.3..1.3] x, [-0.5..0.5] y
		const anchors: Array<[number, number]> = [];
		if (!isMobile) {
			try {
				const sign = growSign();
				for (const path of sign.paths) {
					const m = path.d.match(/M([\d.-]+) ([\d.-]+)L([\d.-]+) ([\d.-]+)/);
					if (!m) continue;
					const x1 = (Number(m[1]) / sign.width - 0.5) * 2.6;
					const y1 = -(Number(m[2]) / sign.height - 0.5) * 0.9;
					const x2 = (Number(m[3]) / sign.width - 0.5) * 2.6;
					const y2 = -(Number(m[4]) / sign.height - 0.5) * 0.9;
					anchors.push([x1, y1], [(x1 + x2) / 2, (y1 + y2) / 2], [x2, y2]);
				}
			} catch {
				/* node band fallback below */
			}
		}

		const particles: FallbackParticle[] = Array.from({ length: count }, (_, i) => {
			const edge = Math.floor(rand() * 4);
			const te = rand();
			let hx = 0;
			let hy = 0;
			if (edge === 0) {
				hx = -1.25;
				hy = te * 2 - 1;
			} else if (edge === 1) {
				hx = 1.25;
				hy = te * 2 - 1;
			} else if (edge === 2) {
				hy = 1.1;
				hx = te * 2 - 1;
			} else {
				hy = -1.1;
				hx = te * 2 - 1;
			}
				const startA = i * 2.39996323;
				const startR = 0.12 + rand() * 0.48;
				return {
					hx,
					hy,
					x: isMobile ? Math.cos(startA) * startR : hx,
					y: isMobile ? Math.sin(startA * 0.92) * startR * 0.62 : hy,
				vx: 0,
				vy: 0,
				kind: Math.floor(rand() * 7),
				seed: rand(),
				delay: 0.1 * rand() + 0.08 * ((edge + te) / 4),
				lane: i % 14
			};
		});

		// node draw helpers — FILLED RECTS ONLY (with rotation for diamonds/shards).
		// Deliberately NO path APIs (no moveTo/lineTo/arc/ellipse/stroke) so the
		// fallback is provably node-only: every node is one or more filled boxes.
		function fillRot(px: number, py: number, hw: number, hh: number, angle: number) {
			ctx.save();
			ctx.translate(px, py);
			if (angle !== 0) ctx.rotate(angle);
			ctx.fillRect(-hw, -hh, hw * 2, hh * 2);
			ctx.restore();
		}
		function fillSquare(px: number, py: number, s: number) {
			ctx.fillRect(px - s, py - s, s * 2, s * 2);
		}
		function fillDiamond(px: number, py: number, s: number) {
			// a square rotated 45deg reads as a diamond — pure fill, no outline.
			fillRot(px, py, s * 0.82, s * 0.82, Math.PI / 4);
		}
		function fillShard(px: number, py: number, s: number) {
			ctx.fillRect(px - s * 0.4, py - s * 1.5, s * 0.8, s * 3);
		}
		function fillBead(px: number, py: number, s: number) {
			ctx.fillRect(px - s * 1.4, py - s * 0.6, s * 2.8, s * 1.2);
		}
		function fillStar(px: number, py: number, s: number) {
			// star core = two overlapping rotated squares (8-point), no outline.
			fillRot(px, py, s * 0.8, s * 0.8, 0);
			fillRot(px, py, s * 0.8, s * 0.8, Math.PI / 4);
			fillSquare(px, py, s * 0.4);
		}
		function fillCross(px: number, py: number, s: number) {
			ctx.fillRect(px - s * 1.5, py - s * 0.35, s * 3, s * 0.7);
			ctx.fillRect(px - s * 0.35, py - s * 1.5, s * 0.7, s * 3);
		}

		function brainTarget(p: FallbackParticle): [number, number] {
			const lobe = p.seed > 0.5 ? 1 : -1;
			const th = p.seed * Math.PI * 2;
			const r = 0.5 + p.seed * 0.4;
			return [Math.cos(th) * r * 0.55 + lobe * 0.34, Math.sin(th) * r * 0.7];
		}
		function graphTarget(p: FallbackParticle, t: number): [number, number] {
			const k = Math.floor(p.seed * 240);
			const th = 2.39996323 * k + t * 0.1;
			const r = 0.3 + (k % 30) / 30 * 0.7;
			return [Math.cos(th) * r, Math.sin(th) * r * 0.8];
		}
		function mobileGraphTarget(p: FallbackParticle, t: number): [number, number] {
			const k = Math.floor(p.seed * 84);
			const shell = k % 3;
			const a = k * 2.39996323 + t * 0.12;
			const b = Math.sin(k * 1.618 + t * 0.18);
			const r = 0.18 + shell * 0.24 + ((k * 17) % 23) / 23 * 0.16;
			const depth = 0.72 + b * 0.18;
			return [Math.cos(a) * r * depth, Math.sin(a * 0.92) * r * 0.72 * depth + b * 0.08];
		}
		function latticeTarget(p: FallbackParticle): [number, number] {
			const g = 6;
			const cx = Math.floor(p.seed * g);
			const cy = Math.floor((p.seed * 53) % 1 * g);
			return [(cx + 0.5) / g * 2 - 1, (cy + 0.5) / g * 1.6 - 0.8];
		}
		function archiveTarget(p: FallbackParticle, t: number): [number, number] {
			const side = p.seed > 0.5 ? 1 : -1;
			const row = (p.seed * 7 + t * 0.05) % 1;
			return [side * (0.7 + (p.lane % 5) * 0.12), row * 2 - 1];
		}
		function receiptTarget(p: FallbackParticle, t: number): [number, number] {
			const x = p.lane / 13 * 2 - 1;
			const travel = (p.seed + t * 0.1) % 1;
			return [x + Math.sin(travel * 18 + p.lane) * 0.05, travel * 2 - 1];
		}
		function glyphTarget(p: FallbackParticle): [number, number] {
			if (!anchors.length) return [(p.seed - 0.5) * 2.4, (p.lane / 14 - 0.5) * 0.8];
			const a = anchors[Math.floor(p.seed * anchors.length) % anchors.length];
			return [a[0], a[1]];
		}

		function formation(p: FallbackParticle, shape: number, t: number): [number, number] {
			switch (shape) {
				case 0:
					return glyphTarget(p);
				case 1:
					return brainTarget(p);
				case 2:
					if (isMobile) return mobileGraphTarget(p, t);
					return graphTarget(p, t);
				case 3:
					return latticeTarget(p);
				case 4:
					return archiveTarget(p, t);
				case 5:
					return receiptTarget(p, t);
				default:
					return [(p.seed - 0.5) * 3, (p.lane / 14 - 0.5) * 2];
			}
		}

		function drawGraphScaffold(t: number, cx: number, cy: number, span: number) {
			const nodeCount = isMobile ? 64 : 58;
			const points: Array<{ x: number; y: number; z: number }> = [];
			for (let i = 0; i < nodeCount; i += 1) {
				const a = i * 2.399963 + t * 0.18;
				const ring = 0.12 + (((i * 37) % 100) / 100) * 0.74;
				const z = Math.sin(i * 1.73 + t * 0.55);
				const scale = 1 / (1.55 - z * 0.22);
				points.push({
					x: cx + Math.cos(a) * ring * span * scale,
					y: cy + Math.sin(a * 0.82) * ring * span * 0.62 * scale,
					z
				});
			}

			ctx.save();
			ctx.globalCompositeOperation = 'lighter';
			ctx.lineWidth = isMobile ? 1.6 : 0.9;
			for (let i = 0; i < points.length; i += 1) {
				const a = points[i];
				const links = [
					points[(i + 5) % points.length],
					points[(i + 13) % points.length],
					points[(i + 29) % points.length]
				];
				for (const b of links) {
					if (Math.abs(a.z - b.z) > 1.25) continue;
					const alpha = isMobile ? 0.28 : 0.12;
					ctx.strokeStyle = `rgba(116, 225, 255, ${alpha})`;
					ctx.beginPath();
					ctx.moveTo(a.x, a.y);
					ctx.lineTo(b.x, b.y);
					ctx.stroke();
				}
			}
			for (const p of points) {
				const s = (isMobile ? 3.0 : 2.0) + (p.z + 1) * 1.4;
				ctx.fillStyle = p.z > 0 ? 'rgba(118, 245, 205, 0.95)' : 'rgba(133, 150, 255, 0.72)';
				ctx.fillRect(p.x - s, p.y - s, s * 2, s * 2);
			}
			ctx.restore();
		}

			let localShape = isMobile ? 2 : 0;
			let localPrev = 0;

			function draw(now: number) {
			if (disposed) return;
			projectionFrame = requestAnimationFrame(draw);
			const { width, height, dpr } = resizeCanvas(canvas, isMobile ? 1.5 : 2);
			const w = width / dpr;
			const h = height / dpr;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

			// solid, NON-circular wash (no radial-gradient circle). Subtle vertical
			// tint via a linear fill only.
			ctx.globalCompositeOperation = 'source-over';
			ctx.fillStyle = '#03040d';
			ctx.fillRect(0, 0, w, h);

			const t = (now - fallbackStartedAt) / 1000;
			const phase = (t % LOOP) / LOOP;
			if (!reducedMotion && localPrev > 0.9 && phase < 0.1) localShape = (localShape + 1) % 7;
			localPrev = phase;
			const mobileShapes = [2, 1, 3, 0];
			const shape = reducedMotion
				? 1
				: isMobile
					? mobileShapes[Math.floor(t / 7) % mobileShapes.length]
					: localShape;

			const wStream = isMobile ? 0 : 1 - Math.min(Math.max((phase - 0.16) / 0.06, 0), 1);
			const wHold = isMobile
				? 1
				: Math.min(Math.max((phase - 0.3) / 0.16, 0), 1) *
					(1 - Math.min(Math.max((phase - 0.78) / 0.08, 0), 1));
			const wDissolve = isMobile ? 0 : Math.min(Math.max((phase - 0.84) / 0.16, 0), 1);

			ctx.globalCompositeOperation = 'lighter';
			const span = Math.min(w, h) * (isMobile ? 0.36 : 0.42);
			const cx = w * 0.5;
			const cy = h * (isMobile ? 0.38 : 0.5);
			if (isMobile) drawGraphScaffold(t, cx, cy, span);
			if (isMobile) ctx.globalCompositeOperation = 'source-over';

			for (let i = 0; i < particles.length; i += 1) {
				const p = particles[i];
				const [fx, fy] = formation(p, shape, t);
				// blend home -> formation by beats
				let tx = fx;
				let ty = fy;
				if (wStream > 0.01) {
					tx = p.hx + (fx - p.hx) * (1 - wStream);
					ty = p.hy + (fy - p.hy) * (1 - wStream);
				}
				if (wDissolve > 0.01) {
					tx = tx + (p.hx - tx) * wDissolve;
					ty = ty + (p.hy - ty) * wDissolve;
				}
				// spring toward target
				p.vx += (tx - p.x) * 0.08 * (0.4 + wHold);
				p.vy += (ty - p.y) * 0.08 * (0.4 + wHold);
				// pointer repel
				const dx = p.x - pointerX;
				const dy = p.y - pointerY;
				const pd = Math.max(Math.hypot(dx, dy), 0.08);
				if (pointerActive > 0.01 && pd < 0.6) {
					p.vx += (dx / pd) * pointerActive * 0.01;
					p.vy += (dy / pd) * pointerActive * 0.01;
				}
				p.vx *= 0.86;
				p.vy *= 0.86;
				p.x += p.vx * (reducedMotion ? 0.2 : 1);
				p.y += p.vy * (reducedMotion ? 0.2 : 1);

				const speed = Math.hypot(p.vx, p.vy);
				const energy = Math.min(speed * 8 + wHold * 0.4 + 0.2, 1.5);
				const px = cx + p.x * span;
				const py = cy - p.y * span;
				const s = (isMobile ? 1.15 : 1.1) + p.seed * 1.25 + energy * (isMobile ? 1.65 : 1.8);
				ctx.fillStyle = isMobile
					? p.seed > 0.5
						? 'rgba(104, 246, 211, 0.78)'
						: 'rgba(134, 156, 255, 0.72)'
					: hslNode(p.seed, energy);
				ctx.globalAlpha = Math.min((isMobile ? 0.32 : 0.35) + energy * (isMobile ? 0.2 : 0.5), 0.82);
				switch (p.kind) {
					case 0:
						fillSquare(px, py, s);
						break;
					case 1:
						fillDiamond(px, py, s);
						break;
					case 2:
						fillSquare(px, py, s * 0.9);
						break;
					case 3:
						fillShard(px, py, s);
						break;
					case 4:
						fillStar(px, py, s);
						break;
					case 5:
						fillBead(px, py, s);
						break;
					default:
						fillCross(px, py, s);
						break;
					}
				}
				if (isMobile) drawGraphScaffold(t + 0.4, cx, cy, span);
				ctx.globalAlpha = 1;
				ctx.globalCompositeOperation = 'source-over';
				}

			fallbackDraw = draw;
			cancelAnimationFrame(projectionFrame);
			projectionFrame = 0;
			draw(performance.now());
		}

	function forceFallbackRequested() {
		return typeof location !== 'undefined' && /[?&]fallback=1/.test(location.search);
	}

	function canAttemptWebGPU() {
		if (typeof navigator === 'undefined' || typeof globalThis === 'undefined') return false;
		const nav = navigator as Navigator & { gpu?: any };
		const globals = globalThis as unknown as { GPUBufferUsage?: Record<string, number> };
		return Boolean(nav.gpu && globals.GPUBufferUsage && !forceFallbackRequested());
	}

	function scheduleWebGPUBoot(delay = 0) {
		if (disposed || !canAttemptWebGPU()) return;
		clearTimeout(webgpuRetryTimer);
		webgpuRetryTimer = setTimeout(() => {
			void tryBootWebGPU();
		}, delay);
	}

	async function tryBootWebGPU() {
		if (disposed || webgpuBooting || mode === 'webgpu' || forceFallbackRequested()) return;
		webgpuBooting = true;
		const token = ++webgpuBootToken;
		try {
			const ok = await bootWebGPU();
			if (disposed || token !== webgpuBootToken) return;
			if (!ok) {
				webgpuAttempt += 1;
				if (mode === 'booting') bootFallback();
				const delay = Math.min(900 + webgpuAttempt * 1200, 6500);
				scheduleWebGPUBoot(delay);
				return;
			}
			webgpuAttempt = 0;
			lastFrame = performance.now();
			frame = requestAnimationFrame(drawWebGPU);
		} catch (error) {
			if (disposed || token !== webgpuBootToken) return;
			console.warn('[launch] Raw WebGPU boot failed:', error);
			webgpuAttempt += 1;
			if (mode === 'booting') bootFallback();
			const delay = Math.min(1200 + webgpuAttempt * 1400, 8000);
			scheduleWebGPUBoot(delay);
		} finally {
			if (token === webgpuBootToken) webgpuBooting = false;
		}
	}

	function bootFallback(retryDelay = 0) {
		if (disposed) return;
		destroyGpu();
		mode = 'fallback';
		// only start the Canvas2D loop NOW (when WebGPU is NOT live) so the
		// fallback never leaks on top of the GPU canvas.
		startFallback();
		if (retryDelay > 0) scheduleWebGPUBoot(retryDelay);
	}

	function onPointerMove(event: PointerEvent) {
		pointerX = (event.clientX / Math.max(window.innerWidth, 1)) * 2 - 1;
		pointerY = -((event.clientY / Math.max(window.innerHeight, 1)) * 2 - 1);
		pointerActive = 1;
		pointerPressure = event.pressure || 0.72;
	}

	function onPointerLeave() {
		pointerActive = 0;
	}

	// Window event from the page on signup success — detonate the seeded-brain
	// supernova. Top-level so onMount (add) and onDestroy (remove) share the identity.
	function onSupernova(event: Event) {
		const detail = (event as CustomEvent<{ seed?: number }>).detail;
		const s = detail && typeof detail.seed === 'number' ? detail.seed : seed;
		triggerSupernova(s);
	}

		function onVisibilityChange() {
			if (document.visibilityState === 'hidden') {
				cancelAnimationFrame(frame);
				frame = 0;
			return;
		}
		if (gpu && mode === 'webgpu') {
			lastFrame = performance.now();
			if (!frame) frame = requestAnimationFrame(drawWebGPU);
			return;
			}
			scheduleWebGPUBoot(150);
		}

		function resetFallbackLoop() {
			cancelAnimationFrame(projectionFrame);
			projectionFrame = 0;
			fallbackStarted = false;
			fallbackDraw = null;
		}

		function kickViewport() {
			if (disposed) return;
			if (gpu && mode === 'webgpu') {
				lastFrame = performance.now();
				if (!frame) frame = requestAnimationFrame(drawWebGPU);
				return;
			}
			resetFallbackLoop();
			if (canAttemptWebGPU()) {
				mode = 'booting';
				scheduleWebGPUBoot(80);
			} else {
				bootFallback();
			}
		}

		function scheduleViewportKick(delayOrEvent: number | Event = 80) {
			const delay = typeof delayOrEvent === 'number' ? delayOrEvent : 80;
			clearTimeout(viewportKickTimer);
			viewportKickTimer = setTimeout(kickViewport, delay);
		}

		function onPageHide() {
			clearTimeout(webgpuRetryTimer);
			clearTimeout(viewportKickTimer);
			webgpuBootToken += 1;
			webgpuBooting = false;
			cancelAnimationFrame(projectionFrame);
			projectionFrame = 0;
			fallbackDraw = null;
			destroyGpu();
			mode = 'booting';
		}

		function onPageShow(event: PageTransitionEvent) {
			disposed = false;
			startedAt = performance.now();
			resetFallbackLoop();
			if (canAttemptWebGPU()) {
				mode = 'booting';
				scheduleWebGPUBoot(event.persisted ? 80 : 150);
			} else {
				bootFallback();
			}
		}

		onMount(() => {
			disposed = false;
			startedAt = performance.now();
		window.addEventListener('pointermove', onPointerMove, { passive: true });
		window.addEventListener('pointerdown', onPointerMove, { passive: true });
		window.addEventListener('pointerup', onPointerLeave, { passive: true });
			document.addEventListener('visibilitychange', onVisibilityChange);
			window.addEventListener('pagehide', onPageHide);
			window.addEventListener('pageshow', onPageShow);
			window.addEventListener('vestige:supernova', onSupernova);
			window.addEventListener('resize', scheduleViewportKick, { passive: true });
			window.addEventListener('orientationchange', scheduleViewportKick, { passive: true });
			window.addEventListener('focus', scheduleViewportKick, { passive: true });
			window.visualViewport?.addEventListener('resize', scheduleViewportKick, { passive: true });
			if (forceFallbackRequested() || !canAttemptWebGPU()) {
				bootFallback();
			} else {
				mode = 'booting';
				(async () => {
					await new Promise<void>((resolve) => {
						requestAnimationFrame(() => setTimeout(resolve, 0));
					});
					await sleep(0);
					scheduleWebGPUBoot(0);
				})();
			}

		return () => {
			disposed = true;
			clearTimeout(webgpuRetryTimer);
		};
	});

		onDestroy(() => {
			disposed = true;
			if (typeof window === 'undefined') return;
			clearTimeout(webgpuRetryTimer);
			clearTimeout(viewportKickTimer);
			cancelAnimationFrame(frame);
			cancelAnimationFrame(projectionFrame);
		syncTarget?.style.removeProperty('--burst');
		syncTarget?.style.removeProperty('--flash');
		window.removeEventListener('pointermove', onPointerMove);
		window.removeEventListener('pointerdown', onPointerMove);
		window.removeEventListener('pointerup', onPointerLeave);
			document.removeEventListener('visibilitychange', onVisibilityChange);
			window.removeEventListener('pagehide', onPageHide);
			window.removeEventListener('pageshow', onPageShow);
			window.removeEventListener('vestige:supernova', onSupernova);
			window.removeEventListener('resize', scheduleViewportKick);
			window.removeEventListener('orientationchange', scheduleViewportKick);
			window.removeEventListener('focus', scheduleViewportKick);
			window.visualViewport?.removeEventListener('resize', scheduleViewportKick);
			destroyGpu();
		});
</script>

<div class={`raw-vestige-engine ${className}`} data-mode={mode}>
	<canvas bind:this={gpuCanvas} class="engine-canvas gpu-canvas" aria-hidden="true"></canvas>
	<canvas bind:this={fallbackCanvas} class="engine-canvas fallback-canvas" aria-hidden="true"></canvas>
</div>

<style>
	.raw-vestige-engine {
		position: fixed;
		inset: 0;
		z-index: 0;
		overflow: hidden;
		/* solid, NON-circular background (no radial-gradient circle). */
		background: linear-gradient(160deg, #03040d 0%, #04061a 52%, #02030a 100%);
		opacity: 1;
		transition: opacity 280ms ease;
	}

	.raw-vestige-engine[data-mode='booting'] {
		opacity: 0;
	}

	.engine-canvas {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		display: block;
		opacity: 0;
		transition: opacity 520ms ease;
	}

	/* GPU canvas only shows in webgpu mode; fallback canvas only shows in
	   fallback mode. They are never both visible -> no fallback leak. */
	.raw-vestige-engine[data-mode='webgpu'] .gpu-canvas {
		opacity: 1;
	}

	.raw-vestige-engine[data-mode='fallback'] .fallback-canvas {
		opacity: 1;
	}
</style>
