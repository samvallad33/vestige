<script lang="ts">
	/**
	 * AmbientField — the Phase 5 base coat. A single metric-bound WebGPU field
	 * mounted full-bleed BEHIND a quiet route's content, at low opacity. It is a
	 * base coat, not a hero: one cheap substrate, but its drive is REAL backend
	 * metrics so it legibly READS the data (a route with 129 endangered memories
	 * storms; 0 endangered is calm — the discipline test still passes).
	 *
	 * Guardrails honored:
	 *   - WebGPU absent / adapter null / context throw → renders nothing (the
	 *     page's own flat background shows through). Never a black canvas.
	 *   - prefers-reduced-motion → the drift freezes (params.reduced=1) but the
	 *     field still renders the metric snapshot (motion is not the information).
	 *   - Page-Visibility + IntersectionObserver gate the rAF loop.
	 *   - DPR clamped; mobile-first mote budget, scales up on desktop.
	 */
	import { onMount } from 'svelte';
	import { ambientFieldWGSL } from '$lib/observatory/ambient/ambient-field.wgsl';

	interface Props {
		/** 0..1 real actively-forgetting fraction — storm intensity. */
		endangered?: number;
		/** 0..1 real contradiction-pair fraction — rift intensity. */
		fracture?: number;
		/** 0..1 real due-for-review fraction — global pulse rate. */
		due?: number;
		/** Real memory count — mote density (how full the mind is). */
		count?: number;
		/** Route accent as [r,g,b] 0..1. Defaults to the brand cyan. */
		accent?: [number, number, number];
		/** Backdrop opacity (kept low — it is a base coat). */
		opacity?: number;
	}

	let {
		endangered = 0,
		fracture = 0,
		due = 0,
		count = 0,
		accent = [0.13, 0.78, 0.87], // #22C7DE brand cyan
		opacity = 0.5
	}: Props = $props();

	let canvas: HTMLCanvasElement | null = $state(null);
	let supported = $state(true);

	// Mote capacity: mobile-first budget, scaled up on wider viewports. Never
	// exceeds the real memory count (the field can't show more mind than exists).
	const CAP_MOBILE = 220;
	const CAP_DESKTOP = 520;

	onMount(() => {
		if (!canvas) return;
		let device: GPUDevice | null = null;
		let context: GPUCanvasContext | null = null;
		let pipeline: GPURenderPipeline | null = null;
		let bindGroup: GPUBindGroup | null = null;
		let uniformBuf: GPUBuffer | null = null;
		let raf = 0;
		let disposed = false;
		let visible = true;
		let onScreen = true;
		let startTs = 0;
		let elapsed = 0;
		let lastTs = 0;

		const reducedMedia = window.matchMedia('(prefers-reduced-motion: reduce)');
		const params = new Float32Array(12);

		const capacity = () =>
			Math.min(
				count > 0 ? count : CAP_DESKTOP,
				window.innerWidth < 640 ? CAP_MOBILE : CAP_DESKTOP
			);

		function writeParams(now: number) {
			const dpr = Math.min(window.devicePixelRatio || 1, window.innerWidth < 640 ? 2 : 1.5);
			const w = Math.max(1, Math.floor((canvas!.clientWidth || 1) * dpr));
			const h = Math.max(1, Math.floor((canvas!.clientHeight || 1) * dpr));
			if (canvas!.width !== w || canvas!.height !== h) {
				canvas!.width = w;
				canvas!.height = h;
			}
			params[0] = elapsed; // seconds
			params[1] = capacity();
			params[2] = Math.max(0, Math.min(1, endangered));
			params[3] = Math.max(0, Math.min(1, fracture));
			params[4] = Math.max(0, Math.min(1, due));
			params[5] = w / Math.max(1, h);
			params[6] = accent[0];
			params[7] = accent[1];
			params[8] = accent[2];
			params[9] = dpr;
			params[10] = reducedMedia.matches ? 1 : 0;
			params[11] = 0;
			void now;
		}

		async function boot() {
			const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
			if (!gpu) {
				supported = false;
				return;
			}
			let adapter: GPUAdapter | null = null;
			try {
				adapter = await gpu.requestAdapter();
			} catch {
				supported = false;
				return;
			}
			if (!adapter || disposed) {
				supported = false;
				return;
			}
			try {
				device = await adapter.requestDevice();
			} catch {
				supported = false;
				return;
			}
			if (disposed) {
				device?.destroy();
				return;
			}
			const ctx = canvas!.getContext('webgpu');
			if (!ctx) {
				supported = false;
				return;
			}
			context = ctx;
			const format = gpu.getPreferredCanvasFormat();
			context.configure({ device, format, alphaMode: 'premultiplied' });

			uniformBuf = device.createBuffer({
				label: 'ambient-params',
				size: params.byteLength,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
			const module = device.createShaderModule({ label: 'ambient-field', code: ambientFieldWGSL });
			pipeline = device.createRenderPipeline({
				label: 'ambient-field',
				layout: 'auto',
				vertex: { module, entryPoint: 'vs_main' },
				fragment: {
					module,
					entryPoint: 'fs_main',
					targets: [
						{
							format,
							// Additive-ish: motes accumulate glow over the transparent void.
							blend: {
								color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
								alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
							}
						}
					]
				},
				primitive: { topology: 'triangle-list' }
			});
			bindGroup = device.createBindGroup({
				label: 'ambient-bind',
				layout: pipeline.getBindGroupLayout(0),
				entries: [{ binding: 0, resource: { buffer: uniformBuf } }]
			});

			lastTs = 0;
			raf = requestAnimationFrame(frame);
		}

		function frame(ts: number) {
			if (disposed || !device || !context || !pipeline || !bindGroup || !uniformBuf) return;
			if (!visible || !onScreen) {
				// Paused: don't submit GPU work, but keep the rAF cheap so we
				// resume instantly when the route/tab comes back.
				raf = requestAnimationFrame(frame);
				return;
			}
			// Advance the ambient clock only when motion is allowed; a reduced-
			// motion viewer still sees the metric snapshot, just frozen.
			if (lastTs > 0 && !reducedMedia.matches) {
				elapsed += Math.min(ts - lastTs, 100) / 1000;
			}
			lastTs = ts;

			writeParams(ts);
			device.queue.writeBuffer(uniformBuf, 0, params);

			let view: GPUTextureView;
			try {
				view = context.getCurrentTexture().createView();
			} catch {
				raf = requestAnimationFrame(frame);
				return;
			}
			const encoder = device.createCommandEncoder({ label: 'ambient-frame' });
			const pass = encoder.beginRenderPass({
				colorAttachments: [
					{ view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }
				]
			});
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, bindGroup);
			pass.draw(6, Math.floor(capacity()));
			pass.end();
			device.queue.submit([encoder.finish()]);
			raf = requestAnimationFrame(frame);
		}

		const onVis = () => {
			visible = document.visibilityState === 'visible';
		};
		document.addEventListener('visibilitychange', onVis);

		const io = new IntersectionObserver(
			(entries) => {
				onScreen = entries.some((e) => e.isIntersecting);
			},
			{ threshold: 0 }
		);
		io.observe(canvas);

		void boot();
		void startTs;

		return () => {
			disposed = true;
			cancelAnimationFrame(raf);
			document.removeEventListener('visibilitychange', onVis);
			io.disconnect();
			uniformBuf?.destroy();
			device?.destroy();
		};
	});
</script>

<!-- The field sits behind page content (the parent positions it). When WebGPU
     is unavailable the canvas simply renders nothing and the page's own
     background shows through — never a black hole. -->
{#if supported}
	<canvas
		bind:this={canvas}
		class="pointer-events-none absolute inset-0 h-full w-full"
		style="opacity: {opacity}"
		aria-hidden="true"
	></canvas>
{/if}
