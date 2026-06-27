<script lang="ts">
	// BackdropEngine — the "alive purple neural field" that breathes behind every
	// dashboard route. Direction #1 (verified universally-60fps): a curl-noise GPU
	// particle field, additively blended into a violet gradient palette, calm at
	// idle so the expensive event effects (decay plume, firewall lightning) hit
	// hard by contrast. Raw WebGPU/WGSL, zero Three.js. Degrades to a Canvas2D
	// field on pre-iOS-26 / no-WebGPU devices (load-bearing for the mobile launch).
	import { onDestroy, onMount } from 'svelte';
	import { browser } from '$app/environment';
	import {
		bootVestigeGpu,
		clampDpr,
		prefersReducedMotion,
		webgpuAvailable,
		type VestigeGpuHandle
	} from './vestigeGpu';
	import { onFieldEvent, fieldEventCode, type FieldEvent } from './fieldEvents';

	// The active reactive effect (a firewall catch / decay / birth). `code` is the
	// shader uniform; `age` counts up in seconds and the effect fades out by
	// EVENT_DURATION, after which code resets to 0. Pointer/origin in clip space.
	const EVENT_DURATION = 1.6;
	let evtCode = 0;
	let evtAge = 0;
	let evtX = 0;
	let evtY = 0;
	let evtIntensity = 1;
	function triggerFieldEvent(e: FieldEvent) {
		evtCode = fieldEventCode(e.kind);
		evtAge = 0;
		// map [0,1] origin to clip space [-1,1]; default center
		evtX = (e.x ?? 0.5) * 2 - 1;
		evtY = -((e.y ?? 0.5) * 2 - 1);
		evtIntensity = e.intensity ?? 1;
	}

	interface Props {
		/** particle budget at full tier; auto-scaled down for weak GPUs / mobile */
		count?: number;
		/** 0..1 ambient intensity — higher = brighter, faster drift */
		intensity?: number;
		class?: string;
	}
	let { count = 22000, intensity = 1, class: className = '' }: Props = $props();

	let gpuCanvas = $state<HTMLCanvasElement | undefined>(undefined);
	let fallbackCanvas = $state<HTMLCanvasElement | undefined>(undefined);
	let mode = $state<'booting' | 'webgpu' | 'fallback'>('booting');

	const WORKGROUP = 64;
	// Guard every browser-global behind `browser`: this component server-side
	// renders, and prefersReducedMotion / matchMedia don't exist on the server.
	const reduced = browser ? prefersReducedMotion() : false;

	// ---- WGSL: curl-noise particle field -------------------------------------
	// Particles advect through a divergence-free (curl-of-noise) flow so the field
	// looks fluid without a solver. Each particle wraps in a unit box and carries
	// an energy/seed for the palette. Cheap: pure per-particle compute, no
	// neighbour search — this is what keeps it 60fps everywhere.
	const COMPUTE_WGSL = /* wgsl */ `
struct Particle { pos: vec4f, vel: vec4f };
struct U {
	viewport: vec4f,   // w, h, dpr, reducedMotion
	params: vec4f,     // time, dt, intensity, count
	event: vec4f,      // code (0 none,1 firewall,2 decay,3 birth), fade(1->0), ox, oy
};
@group(0) @binding(0) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(1) var<uniform> u: U;

// hash + value noise (cheap, deterministic)
fn hash3(p: vec3f) -> f32 {
	let q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
	let r = q + dot(q, q.yxz + 33.33);
	return fract((r.x + r.y) * r.z);
}
fn noise3(p: vec3f) -> f32 {
	let i = floor(p); let f = fract(p);
	let w = f * f * (3.0 - 2.0 * f);
	let c000 = hash3(i + vec3f(0,0,0)); let c100 = hash3(i + vec3f(1,0,0));
	let c010 = hash3(i + vec3f(0,1,0)); let c110 = hash3(i + vec3f(1,1,0));
	let c001 = hash3(i + vec3f(0,0,1)); let c101 = hash3(i + vec3f(1,0,1));
	let c011 = hash3(i + vec3f(0,1,1)); let c111 = hash3(i + vec3f(1,1,1));
	let x00 = mix(c000, c100, w.x); let x10 = mix(c010, c110, w.x);
	let x01 = mix(c001, c101, w.x); let x11 = mix(c011, c111, w.x);
	return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}
fn potential(p: vec3f) -> vec3f {
	let t = u.params.x * 0.06;
	return vec3f(
		noise3(p * 1.4 + vec3f(0.0, t, 0.0)),
		noise3(p * 1.4 + vec3f(5.2, t, 1.3)),
		noise3(p * 1.4 + vec3f(2.7, t, 9.1))
	);
}
// curl of the potential field = divergence-free flow
fn curlFlow(p: vec3f) -> vec3f {
	let e = 0.08;
	let dx = vec3f(e, 0.0, 0.0); let dy = vec3f(0.0, e, 0.0); let dz = vec3f(0.0, 0.0, e);
	let px0 = potential(p - dx); let px1 = potential(p + dx);
	let py0 = potential(p - dy); let py1 = potential(p + dy);
	let pz0 = potential(p - dz); let pz1 = potential(p + dz);
	let x = (py1.z - py0.z) - (pz1.y - pz0.y);
	let y = (pz1.x - pz0.x) - (px1.z - px0.z);
	let z = (px1.y - px0.y) - (py1.x - py0.x);
	return vec3f(x, y, z) / (2.0 * e);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
	let i = gid.x;
	if (i >= u32(u.params.w)) { return; }
	var pr = parts[i];
	let rm = u.viewport.w;
	let speed = mix(1.0, 0.18, rm) * u.params.z;
	let flow = curlFlow(pr.pos.xyz * 1.1);
	pr.vel = vec4f(mix(pr.vel.xyz, flow * 0.6, 0.06), pr.vel.w);

	// --- reactive event impulse (scoped to the event origin) -----------------
	let ecode = u.event.x;
	let efade = u.event.y; // 1 at trigger -> 0 at end
	if (ecode > 0.5 && efade > 0.001) {
		let origin = vec3f(u.event.z, u.event.w, 0.0);
		let d = pr.pos.xyz - origin;
		let dist = length(d) + 0.0001;
		let dir = d / dist;
		let reach = exp(-dist * 1.4) * efade; // local falloff
		if (ecode < 1.5) {
			// FIREWALL: a hard outward shockwave + energy spike (the "block")
			pr.vel = vec4f(pr.vel.xyz + dir * reach * 0.9, pr.vel.w);
			pr.pos = vec4f(pr.pos.xyz, min(pr.pos.w + reach * 0.6, 1.0));
		} else if (ecode < 2.5) {
			// DECAY: pull inward + damp (a cold collapse, the Rac1 cascade fade)
			pr.vel = vec4f(pr.vel.xyz - dir * reach * 0.5, pr.vel.w * (1.0 - reach * 0.5));
			pr.pos = vec4f(pr.pos.xyz, max(pr.pos.w - reach * 0.4, 0.0));
		} else {
			// BIRTH: a gentle bright bloom outward
			pr.vel = vec4f(pr.vel.xyz + dir * reach * 0.4, pr.vel.w);
			pr.pos = vec4f(pr.pos.xyz, min(pr.pos.w + reach * 0.5, 1.0));
		}
	}

	pr.pos = vec4f(pr.pos.xyz + pr.vel.xyz * u.params.y * speed, pr.pos.w);
	// soft-wrap inside a unit box so the field is endless
	pr.pos = vec4f(fract(pr.pos.xyz * 0.5 + 0.5) * 2.0 - 1.0, pr.pos.w);
	parts[i] = pr;
}
`;

	// ---- WGSL: additive point render with violet palette ---------------------
	const RENDER_WGSL = /* wgsl */ `
struct Particle { pos: vec4f, vel: vec4f };
struct U { viewport: vec4f, params: vec4f, event: vec4f };
@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<uniform> u: U;

struct VsOut {
	@builtin(position) clip: vec4f,
	@location(0) quad: vec2f,
	@location(1) glow: f32,
	@location(2) seed: f32,
	@location(3) crimson: f32, // 0 normal violet .. 1 firewall crimson
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
	var corners = array<vec2f, 6>(
		vec2f(-1.0,-1.0), vec2f(1.0,-1.0), vec2f(-1.0,1.0),
		vec2f(-1.0,1.0),  vec2f(1.0,-1.0), vec2f(1.0,1.0)
	);
	let c = corners[vi];
	let p = parts[ii];
	let aspect = u.viewport.x / max(u.viewport.y, 1.0);
	// gentle parallax depth so the cloud reads volumetric
	let depth = 1.7 + p.pos.z * 0.5;
	let sizePx = (2.6 + p.pos.w * 3.4) / depth;
	var ndc = vec2f(p.pos.x / aspect, p.pos.y);
	ndc = ndc + c * vec2f(sizePx * 0.01 / aspect, sizePx * 0.01);
	let speed = length(p.vel.xyz);
	var out: VsOut;
	out.clip = vec4f(ndc, 0.0, 1.0);
	out.quad = c;
	out.glow = clamp(0.35 + speed * 1.4 + p.pos.w * 0.4, 0.2, 1.6);
	out.seed = p.pos.w;
	// FIREWALL (event code 1): particles near the origin flare crimson — the
	// "angry" block. Crimson is reserved ONLY for this; nothing else turns red.
	var crimson = 0.0;
	if (u.event.x > 0.5 && u.event.x < 1.5) {
		let d = length(p.pos.xy - vec2f(u.event.z, u.event.w));
		crimson = exp(-d * 1.6) * u.event.y;
	}
	out.crimson = clamp(crimson, 0.0, 1.0);
	return out;
}

// 3-stop violet ramp: deep indigo -> brand violet -> hot magenta highlight.
// Driven by energy, NEVER hue-cycled (that's what keeps it premium, not rainbow).
fn palette(e: f32) -> vec3f {
	let indigo  = vec3f(0.10, 0.05, 0.28);
	let violet  = vec3f(0.42, 0.20, 0.86);
	let magenta = vec3f(0.92, 0.32, 0.85);
	let t = clamp(e, 0.0, 1.0);
	let lo = mix(indigo, violet, smoothstep(0.0, 0.55, t));
	return mix(lo, magenta, smoothstep(0.55, 1.0, t));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
	let r = length(in.quad);
	if (r > 1.0) { discard; }
	// soft core + wider halo so each particle reads as a glowing synapse, not a dot
	let core = pow(1.0 - r, 2.6);
	let halo = (1.0 - r) * 0.5;
	let falloff = core + halo;
	let energy = clamp(in.glow * (0.55 + in.seed * 0.6), 0.0, 1.0);
	// brightness floor so the field is visible even where the flow is slow
	let lum = 0.55 + in.glow * 0.9;
	var col = palette(energy) * lum * falloff;
	// firewall flare: blend toward an angry crimson + extra glow at the catch
	let crimsonCol = vec3f(1.0, 0.12, 0.22) * (lum + in.crimson * 1.4) * falloff;
	col = mix(col, crimsonCol, in.crimson);
	return vec4f(col, falloff); // premultiplied additive
}
`;

	type Handle = {
		gpu: VestigeGpuHandle;
		partBuf: any;
		uBuf: any;
		computePipe: any;
		renderPipe: any;
		computeBind: any;
		renderBind: any;
		particleCount: number;
	};
	let handle: Handle | null = null;
	let raf = 0;
	let disposed = false;
	let startedAt = 0;
	let lastT = 0;

	// Tracks whether the tick() chain is currently live, so any caller can ask to
	// (re)start the loop without risking a second parallel chain. `tick` clears it
	// only when it actually stops (disposed); otherwise it keeps re-scheduling.
	let running = false;

	// Single rAF dispatcher: the ONLY place the draw loop is scheduled. Routing
	// through one tick() means a mode switch, a mid-frame fallback, or a
	// visibility change can never leave two self-scheduling chains racing.
	function tick() {
		if (disposed) { running = false; return; }
		if (mode === 'webgpu') drawWebgpu();
		else if (mode === 'fallback') drawFallback();
		raf = requestAnimationFrame(tick);
	}
	// Idempotent loop starter: starts the chain only if it isn't already running,
	// so a mid-frame bootFallback() (from drawWebgpu's catch) does NOT stack a
	// second chain on top of the one tick() is about to schedule itself.
	function startLoop() {
		if (disposed || running) return;
		running = true;
		cancelAnimationFrame(raf);
		raf = requestAnimationFrame(tick);
	}
	function stopLoop() {
		running = false;
		cancelAnimationFrame(raf);
	}

	// Destroy any GPU buffers we may have created. Safe to call with either the
	// committed `handle` or the in-flight boot locals — every early return,
	// catch, and device-loss path funnels through here so a buffer can never leak.
	function releaseBuffers(partBuf?: any, uBuf?: any) {
		try { (partBuf ?? handle?.partBuf)?.destroy?.(); } catch {}
		try { (uBuf ?? handle?.uBuf)?.destroy?.(); } catch {}
	}

	function pickCount(): number {
		const small = (globalThis as any).innerWidth < 760;
		let n = small ? Math.min(count, 12000) : count;
		if (reduced) n = Math.floor(n * 0.5);
		return Math.max(2000, n);
	}

	function buildParticles(n: number): Float32Array {
		// 2 vec4 per particle (pos.xyz + energy, vel.xyz + pad) = 8 floats
		const data = new Float32Array(n * 8);
		let s = 0x9e3779b9 >>> 0;
		const rnd = () => {
			s ^= s << 13; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
			return s / 0xffffffff;
		};
		for (let i = 0; i < n; i++) {
			const o = i * 8;
			data[o + 0] = rnd() * 2 - 1;
			data[o + 1] = rnd() * 2 - 1;
			data[o + 2] = rnd() * 2 - 1;
			data[o + 3] = rnd(); // energy seed
			data[o + 4] = 0; data[o + 5] = 0; data[o + 6] = 0; data[o + 7] = 0;
		}
		return data;
	}

	async function tryBootWebgpu() {
		// Bail if the watchdog already handed control to Canvas2D (or we were
		// disposed) before we even started — a late boot must never re-take over.
		if (mode !== 'booting' || disposed || !gpuCanvas) return;
		// Buffers tracked in locals so any early exit can release them before they
		// ever reach `handle` (otherwise they leak on every late-return path).
		let partBuf: any;
		let uBuf: any;
		const gpu = await bootVestigeGpu(gpuCanvas);
		// A 6s boot can resolve after the watchdog fired bootFallback(): if mode is
		// no longer 'booting' (or we were disposed) abort so we don't start a 2nd loop.
		if (disposed || mode !== 'booting') { releaseBuffers(partBuf, uBuf); return; }
		if (!gpu) { bootFallback(); return; }
		try {
			const device = gpu.device;
			const particleCount = pickCount();
			const partData = buildParticles(particleCount);
			partBuf = device.createBuffer({
				size: partData.byteLength,
				usage: 0x80 | 0x8, // STORAGE | COPY_DST
				mappedAtCreation: true
			});
			new Float32Array(partBuf.getMappedRange()).set(partData);
			partBuf.unmap();

			uBuf = device.createBuffer({ size: 48, usage: 0x40 | 0x8 }); // UNIFORM | COPY_DST (3x vec4)

			const computeMod = device.createShaderModule({ code: COMPUTE_WGSL });
			const renderMod = device.createShaderModule({ code: RENDER_WGSL });
			const computePipe = await device.createComputePipelineAsync({
				layout: 'auto',
				compute: { module: computeMod, entryPoint: 'main' }
			});
			if (disposed || mode !== 'booting') { releaseBuffers(partBuf, uBuf); return; }
			const renderPipe = await device.createRenderPipelineAsync({
				layout: 'auto',
				vertex: { module: renderMod, entryPoint: 'vs' },
				fragment: {
					module: renderMod,
					entryPoint: 'fs',
					targets: [
						{
							format: gpu.format,
							blend: {
								color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
								alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
							}
						}
					]
				},
				primitive: { topology: 'triangle-list' }
			});
			if (disposed || mode !== 'booting') { releaseBuffers(partBuf, uBuf); return; }
			const computeBind = device.createBindGroup({
				layout: computePipe.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: partBuf } },
					{ binding: 1, resource: { buffer: uBuf } }
				]
			});
			const renderBind = device.createBindGroup({
				layout: renderPipe.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: partBuf } },
					{ binding: 1, resource: { buffer: uBuf } }
				]
			});
			handle = { gpu, partBuf, uBuf, computePipe, renderPipe, computeBind, renderBind, particleCount };

			gpu.device.lost?.then?.(() => {
				if (disposed) return;
				releaseBuffers(); // destroy this handle's buffers before dropping it
				handle = null;
				bootFallback();
			});

			mode = 'webgpu';
			if (bootWatchdog) { clearTimeout(bootWatchdog); bootWatchdog = null; }
			resize();
			startedAt = performance.now();
			lastT = startedAt;
			startLoop();
		} catch {
			releaseBuffers(partBuf, uBuf); // pipelines/binds may have failed mid-build
			bootFallback();
		}
	}

	function resize() {
		const dpr = clampDpr(1.5);
		if (mode === 'webgpu' && gpuCanvas) {
			gpuCanvas.width = Math.floor(gpuCanvas.clientWidth * dpr);
			gpuCanvas.height = Math.floor(gpuCanvas.clientHeight * dpr);
			// Resizing the canvas drops the swapchain's configuration; re-apply it
			// (same premultiplied alpha bootVestigeGpu used) or the next frame
			// renders against a stale-sized / unconfigured context.
			if (handle) {
				try {
					handle.gpu.context.configure({
						device: handle.gpu.device,
						format: handle.gpu.format,
						alphaMode: 'premultiplied'
					});
				} catch {}
			}
		} else if (fallbackCanvas) {
			fallbackCanvas.width = Math.floor(fallbackCanvas.clientWidth * Math.min(dpr, 1.25));
			fallbackCanvas.height = Math.floor(fallbackCanvas.clientHeight * Math.min(dpr, 1.25));
		}
	}

	function drawWebgpu() {
		if (disposed || !handle || mode !== 'webgpu') return;
		try {
			const now = performance.now();
			const dt = Math.min((now - lastT) / 1000, 0.05);
			lastT = now;
			const t = (now - startedAt) / 1000;
			const { device, context } = handle.gpu;

			// advance any active reactive effect; clear it once it has faded out
			if (evtCode !== 0) {
				evtAge += dt;
				if (evtAge > EVENT_DURATION) evtCode = 0;
			}
			const evtNorm = evtCode === 0 ? 0 : Math.max(0, 1 - evtAge / EVENT_DURATION);

			const u = new Float32Array(12);
			u[0] = gpuCanvas!.width; u[1] = gpuCanvas!.height; u[2] = clampDpr(1.5); u[3] = reduced ? 1 : 0;
			u[4] = t; u[5] = dt; u[6] = intensity; u[7] = handle.particleCount;
			// event vec4: code, fade(1->0), origin handled below via x/y, intensity
			u[8] = evtCode; u[9] = evtNorm * evtIntensity; u[10] = evtX; u[11] = evtY;
			device.queue.writeBuffer(handle.uBuf, 0, u);

			const enc = device.createCommandEncoder();
			const cp = enc.beginComputePass();
			cp.setPipeline(handle.computePipe);
			cp.setBindGroup(0, handle.computeBind);
			cp.dispatchWorkgroups(Math.ceil(handle.particleCount / WORKGROUP));
			cp.end();

			const view = context.getCurrentTexture().createView();
			const rp = enc.beginRenderPass({
				colorAttachments: [
					{ view, clearValue: { r: 0.02, g: 0.012, b: 0.05, a: 1 }, loadOp: 'clear', storeOp: 'store' }
				]
			});
			rp.setPipeline(handle.renderPipe);
			rp.setBindGroup(0, handle.renderBind);
			rp.draw(6, handle.particleCount);
			rp.end();
			device.queue.submit([enc.finish()]);
		} catch (err) {
			// A single bad frame (e.g. device loss mid-submit) must not kill the
			// loop forever — drop to the resilient Canvas2D field instead.
			console.error('[BackdropEngine] WebGPU frame failed, falling back', err);
			bootFallback();
		}
		// NOTE: no rAF here — tick() owns scheduling.
	}

	// ---- Canvas2D fallback (no WebGPU: pre-iOS-26 iPhones, old GPUs) ----------
	let fbParts: { x: number; y: number; vx: number; vy: number; e: number }[] = [];
	function bootFallback() {
		if (disposed || !fallbackCanvas) return;
		if (mode === 'fallback' && fbParts.length) return; // idempotent: don't double-start
		if (bootWatchdog) { clearTimeout(bootWatchdog); bootWatchdog = null; }
		mode = 'fallback';
		resize();
		const n = (globalThis as any).innerWidth < 760 ? 520 : 1400;
		fbParts = Array.from({ length: n }, () => ({
			x: Math.random(), y: Math.random(),
			vx: 0, vy: 0, e: Math.random()
		}));
		startedAt = performance.now();
		fbLastT = 0;
		// startLoop() is a no-op if the chain is already live (e.g. called from
		// drawWebgpu's catch inside tick) — tick simply routes to drawFallback
		// next frame. When called cold (watchdog / no-WebGPU) it starts the chain.
		startLoop();
	}
	function fbField(x: number, y: number, t: number): [number, number] {
		// cheap pseudo-curl: orthogonal gradient of a sine field
		const a = Math.sin(x * 6.2 + t * 0.3) + Math.cos(y * 5.1 - t * 0.2);
		return [Math.cos(a * 2.0) * 0.0009, Math.sin(a * 2.0) * 0.0009];
	}
	let fbLastT = 0;
	function drawFallback() {
		if (disposed || !fallbackCanvas || mode !== 'fallback') return;
		const ctx = fallbackCanvas.getContext('2d');
		if (!ctx) return;
		const w = fallbackCanvas.width, h = fallbackCanvas.height;
		const nowMs = performance.now();
		const t = (nowMs - startedAt) / 1000;
		const dt = fbLastT ? Math.min((nowMs - fbLastT) / 1000, 0.05) : 0.016;
		fbLastT = nowMs;
		// advance any active reactive event (firewall flash also works here so
		// pre-iOS-26 / no-WebGPU users still see the catch)
		if (evtCode !== 0) { evtAge += dt; if (evtAge > EVENT_DURATION) evtCode = 0; }
		const fwFade = evtCode === 1 ? Math.max(0, 1 - evtAge / EVENT_DURATION) * evtIntensity : 0;
		const ox = (evtX + 1) * 0.5, oy = (1 - evtY) * 0.5; // back to [0,1]
		ctx.fillStyle = 'rgba(5,3,13,0.18)'; // trail fade on void
		ctx.fillRect(0, 0, w, h);
		ctx.globalCompositeOperation = 'lighter';
		// Normalise motion to a 60fps reference so the field drifts at the same
		// speed regardless of refresh rate (WebGPU advects by *dt; match it here).
		const fstep = dt * 60;
		for (const p of fbParts) {
			const [fx, fy] = fbField(p.x, p.y, t);
			// scale the flow force by the ambient `intensity` prop (was ignored)
			p.vx = p.vx * 0.92 + fx * (reduced ? 0.3 : 1) * intensity;
			p.vy = p.vy * 0.92 + fy * (reduced ? 0.3 : 1) * intensity;
			// firewall shockwave: push outward from the origin
			let crimson = 0;
			if (fwFade > 0) {
				const dx = p.x - ox, dy = p.y - oy;
				const dist = Math.hypot(dx, dy) + 1e-4;
				const reach = Math.exp(-dist * 2.2) * fwFade;
				p.vx += (dx / dist) * reach * 0.01 * fstep;
				p.vy += (dy / dist) * reach * 0.01 * fstep;
				crimson = Math.min(reach * 1.6, 1);
			}
			p.x += p.vx * fstep; p.y += p.vy * fstep;
			if (p.x < 0) p.x += 1; if (p.x > 1) p.x -= 1;
			if (p.y < 0) p.y += 1; if (p.y > 1) p.y -= 1;
			const sp = Math.hypot(p.vx, p.vy) * 220;
			const energy = Math.min(0.3 + sp + p.e * 0.4, 1);
			// violet ramp matching the WGSL palette, blended toward crimson on a catch
			const vr = 26 + energy * 209, vg = 13 + energy * 70, vb = 71 + energy * 146;
			const r = Math.floor(vr + (255 - vr) * crimson);
			const g = Math.floor(vg + (30 - vg) * crimson);
			const b = Math.floor(vb + (56 - vb) * crimson);
			ctx.fillStyle = `rgba(${r},${g},${b},${(0.5 + crimson * 0.4) * energy})`;
			ctx.fillRect(p.x * w, p.y * h, 2.2 + crimson * 1.5, 2.2 + crimson * 1.5);
		}
		ctx.globalCompositeOperation = 'source-over';
		// NOTE: no rAF here — tick() owns scheduling.
	}

	function onResize() { resize(); }
	function onVisibility() {
		if (document.hidden) {
			stopLoop();
		} else if (!disposed) {
			// Reset BOTH clocks so neither loop sees a huge dt jump after the tab
			// was backgrounded (fallback uses fbLastT, webgpu uses lastT).
			lastT = performance.now();
			fbLastT = performance.now();
			startLoop(); // idempotent — never stacks a second chain
		}
	}

	let unsubFieldEvents: (() => void) | null = null;
	let bootWatchdog: ReturnType<typeof setTimeout> | null = null;

	onMount(() => {
		disposed = false;
		window.addEventListener('resize', onResize);
		document.addEventListener('visibilitychange', onVisibility);
		unsubFieldEvents = onFieldEvent(triggerFieldEvent);
		if (webgpuAvailable()) {
			// defer one frame so the route paints before GPU boot work
			requestAnimationFrame(() => { if (!disposed) tryBootWebgpu(); });
			// Watchdog: if WebGPU hasn't actually started rendering within 6s (a
			// slow/contended GPU or a hung device request), fall back to Canvas2D
			// so the field is never stuck on the static gradient.
			bootWatchdog = setTimeout(() => {
				if (!disposed && mode === 'booting') bootFallback();
			}, 6000);
		} else {
			bootFallback();
		}
		return () => {};
	});

	onDestroy(() => {
		disposed = true;
		unsubFieldEvents?.();
		if (bootWatchdog) { clearTimeout(bootWatchdog); bootWatchdog = null; }
		if (!browser) return; // onDestroy also fires during SSR — no DOM there
		stopLoop();
		window.removeEventListener('resize', onResize);
		document.removeEventListener('visibilitychange', onVisibility);
		releaseBuffers(); // destroy GPU buffers so they don't outlive the component
		handle = null;
	});
</script>

<div class="backdrop-engine {className}" aria-hidden="true" data-mode={mode}>
	<canvas bind:this={gpuCanvas} class="backdrop-canvas" class:active={mode === 'webgpu'}></canvas>
	<canvas bind:this={fallbackCanvas} class="backdrop-canvas" class:active={mode === 'fallback'}></canvas>
</div>

<style>
	.backdrop-engine {
		position: fixed;
		inset: 0;
		z-index: 0;
		pointer-events: none;
		overflow: hidden;
		/* a static violet gradient sits under the canvas so first paint (and the
		   no-JS / pre-boot moment) is already on-brand, never a black void. */
		background:
			radial-gradient(120% 90% at 20% 0%, rgba(66, 32, 137, 0.22), transparent 60%),
			radial-gradient(100% 80% at 90% 100%, rgba(146, 51, 214, 0.16), transparent 55%),
			var(--color-void, #050510);
	}
	.backdrop-canvas {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		opacity: 0;
		transition: opacity 1.1s ease;
	}
	.backdrop-canvas.active {
		opacity: 1;
	}
</style>
