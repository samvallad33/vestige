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
	pr.pos = vec4f(pr.pos.xyz + pr.vel.xyz * u.params.y * speed, pr.pos.w);
	// soft-wrap inside a unit box so the field is endless
	pr.pos = vec4f(fract(pr.pos.xyz * 0.5 + 0.5) * 2.0 - 1.0, pr.pos.w);
	parts[i] = pr;
}
`;

	// ---- WGSL: additive point render with violet palette ---------------------
	const RENDER_WGSL = /* wgsl */ `
struct Particle { pos: vec4f, vel: vec4f };
struct U { viewport: vec4f, params: vec4f };
@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(1) var<uniform> u: U;

struct VsOut {
	@builtin(position) clip: vec4f,
	@location(0) quad: vec2f,
	@location(1) glow: f32,
	@location(2) seed: f32,
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
	let col = palette(energy) * lum * falloff;
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
		if (disposed || !gpuCanvas) return;
		const gpu = await bootVestigeGpu(gpuCanvas);
		if (disposed) return;
		if (!gpu) { bootFallback(); return; }
		try {
			const device = gpu.device;
			const particleCount = pickCount();
			const partData = buildParticles(particleCount);
			const partBuf = device.createBuffer({
				size: partData.byteLength,
				usage: 0x80 | 0x8, // STORAGE | COPY_DST
				mappedAtCreation: true
			});
			new Float32Array(partBuf.getMappedRange()).set(partData);
			partBuf.unmap();

			const uBuf = device.createBuffer({ size: 32, usage: 0x40 | 0x8 }); // UNIFORM | COPY_DST

			const computeMod = device.createShaderModule({ code: COMPUTE_WGSL });
			const renderMod = device.createShaderModule({ code: RENDER_WGSL });
			const computePipe = await device.createComputePipelineAsync({
				layout: 'auto',
				compute: { module: computeMod, entryPoint: 'main' }
			});
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
			if (disposed) return;
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
				handle = null;
				bootFallback();
			});

			mode = 'webgpu';
			resize();
			startedAt = performance.now();
			lastT = startedAt;
			raf = requestAnimationFrame(drawWebgpu);
		} catch {
			bootFallback();
		}
	}

	function resize() {
		const dpr = clampDpr(1.5);
		if (mode === 'webgpu' && gpuCanvas) {
			gpuCanvas.width = Math.floor(gpuCanvas.clientWidth * dpr);
			gpuCanvas.height = Math.floor(gpuCanvas.clientHeight * dpr);
		} else if (fallbackCanvas) {
			fallbackCanvas.width = Math.floor(fallbackCanvas.clientWidth * Math.min(dpr, 1.25));
			fallbackCanvas.height = Math.floor(fallbackCanvas.clientHeight * Math.min(dpr, 1.25));
		}
	}

	function drawWebgpu() {
		if (disposed || !handle || mode !== 'webgpu') return;
		const now = performance.now();
		const dt = Math.min((now - lastT) / 1000, 0.05);
		lastT = now;
		const t = (now - startedAt) / 1000;
		const { device, context } = handle.gpu;

		const u = new Float32Array(8);
		u[0] = gpuCanvas!.width; u[1] = gpuCanvas!.height; u[2] = clampDpr(1.5); u[3] = reduced ? 1 : 0;
		u[4] = t; u[5] = dt; u[6] = intensity; u[7] = handle.particleCount;
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
		raf = requestAnimationFrame(drawWebgpu);
	}

	// ---- Canvas2D fallback (no WebGPU: pre-iOS-26 iPhones, old GPUs) ----------
	let fbParts: { x: number; y: number; vx: number; vy: number; e: number }[] = [];
	function bootFallback() {
		if (disposed || !fallbackCanvas) return;
		mode = 'fallback';
		resize();
		const n = (globalThis as any).innerWidth < 760 ? 520 : 1400;
		fbParts = Array.from({ length: n }, () => ({
			x: Math.random(), y: Math.random(),
			vx: 0, vy: 0, e: Math.random()
		}));
		startedAt = performance.now();
		raf = requestAnimationFrame(drawFallback);
	}
	function fbField(x: number, y: number, t: number): [number, number] {
		// cheap pseudo-curl: orthogonal gradient of a sine field
		const a = Math.sin(x * 6.2 + t * 0.3) + Math.cos(y * 5.1 - t * 0.2);
		return [Math.cos(a * 2.0) * 0.0009, Math.sin(a * 2.0) * 0.0009];
	}
	function drawFallback() {
		if (disposed || !fallbackCanvas || mode !== 'fallback') return;
		const ctx = fallbackCanvas.getContext('2d');
		if (!ctx) return;
		const w = fallbackCanvas.width, h = fallbackCanvas.height;
		const t = (performance.now() - startedAt) / 1000;
		ctx.fillStyle = 'rgba(5,3,13,0.18)'; // trail fade on void
		ctx.fillRect(0, 0, w, h);
		ctx.globalCompositeOperation = 'lighter';
		for (const p of fbParts) {
			const [fx, fy] = fbField(p.x, p.y, t);
			p.vx = p.vx * 0.92 + fx * (reduced ? 0.3 : 1);
			p.vy = p.vy * 0.92 + fy * (reduced ? 0.3 : 1);
			p.x += p.vx; p.y += p.vy;
			if (p.x < 0) p.x += 1; if (p.x > 1) p.x -= 1;
			if (p.y < 0) p.y += 1; if (p.y > 1) p.y -= 1;
			const sp = Math.hypot(p.vx, p.vy) * 220;
			const energy = Math.min(0.3 + sp + p.e * 0.4, 1);
			// violet ramp matching the WGSL palette
			const r = Math.floor(26 + energy * 209);
			const g = Math.floor(13 + energy * 70);
			const b = Math.floor(71 + energy * 146);
			ctx.fillStyle = `rgba(${r},${g},${b},${0.5 * energy})`;
			ctx.fillRect(p.x * w, p.y * h, 2.2, 2.2);
		}
		ctx.globalCompositeOperation = 'source-over';
		raf = requestAnimationFrame(drawFallback);
	}

	function onResize() { resize(); }
	function onVisibility() {
		if (document.hidden) {
			cancelAnimationFrame(raf);
		} else if (!disposed) {
			lastT = performance.now();
			raf = requestAnimationFrame(mode === 'webgpu' ? drawWebgpu : drawFallback);
		}
	}

	onMount(() => {
		disposed = false;
		window.addEventListener('resize', onResize);
		document.addEventListener('visibilitychange', onVisibility);
		if (webgpuAvailable()) {
			// defer one frame so the route paints before GPU boot work
			requestAnimationFrame(() => { if (!disposed) tryBootWebgpu(); });
		} else {
			bootFallback();
		}
		return () => {};
	});

	onDestroy(() => {
		disposed = true;
		if (!browser) return; // onDestroy also fires during SSR — no DOM there
		cancelAnimationFrame(raf);
		window.removeEventListener('resize', onResize);
		document.removeEventListener('visibilitychange', onVisibility);
		try { handle?.partBuf?.destroy?.(); handle?.uBuf?.destroy?.(); } catch {}
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
