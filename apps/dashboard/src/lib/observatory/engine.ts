/**
 * Cognitive Observatory — WebGPU engine.
 *
 * Owns adapter/device/context lifecycle, the render loop, resize, and dispose.
 * Increment 3 scope: boot WebGPU, clear to void #05060a, DPR-clamped resize,
 * readable fallback when WebGPU is unavailable. Later increments register
 * pipelines on top of this shell (spec §2, §3.4).
 *
 * Hard rules (spec §6):
 *  - No Math.random()/Date.now()/performance.now() deciding simulation state
 *    (the DemoClock is the only sim clock; rAF timestamps only schedule).
 *  - No GPU readback in the frame loop.
 */

import { DemoClock } from './demo-clock';
import { PARAMS_FLOATS, PARAM_IDX, demoModeId, type DemoMode } from './types';
import { PostChain, SCENE_FORMAT } from './post/post-chain';
import { VOID_CLEAR_HDR } from './post/tone-reference';

/**
 * Void background — visual DNA §7. #05060a as the DISPLAY value (public API,
 * pre-post-stack). The HDR scene pass clears to VOID_CLEAR_HDR
 * (post/tone-reference.ts), whose tonemapped composite result is exactly
 * this color.
 */
export const VOID_CLEAR: GPUColor = {
	r: 0x05 / 255,
	g: 0x06 / 255,
	b: 0x0a / 255,
	a: 1
};

export interface EngineOptions {
	canvas: HTMLCanvasElement;
	demo: DemoMode;
	seed: string;
	/** Max devicePixelRatio to render at (spec: DPR clamp). */
	maxDpr?: number;
	/** Called once per completed frame with the loop frame index (telemetry). */
	onFrame?: (frame: number, fps: number) => void;
	/**
	 * Capture mode (?frame=N): freeze the simulation at this exact loop frame.
	 * Same URL + frame → identical pixels, for shareable stills (spec §4 Inc 9).
	 */
	freezeFrame?: number | null;
}

export type EngineStatus =
	| { state: 'booting' }
	| { state: 'running' }
	| { state: 'unsupported'; reason: string }
	| { state: 'error'; reason: string }
	| { state: 'disposed' };

/**
 * Frame hook contract for later increments: each registered pass gets the
 * encoder + the main scene pass (offscreen HDR target) once per frame, after
 * the clear. The post chain composites the scene to the swapchain afterwards.
 */
export interface FramePass {
	/** Encode compute work (before the render pass). */
	compute?(encoder: GPUCommandEncoder, frame: number): void;
	/** Encode draw calls inside the main render pass. */
	render?(pass: GPURenderPassEncoder, frame: number): void;
	/**
	 * Optional render-rate contract for quiet, information-dense organs.
	 *
	 * The engine still advances its deterministic clock at 60 Hz; this only
	 * decides how often an otherwise-settled scene pays for an HDR render and
	 * post chain. Passes that omit it retain the normal 60 fps behavior.
	 */
	targetFrameRate?(frame: number): number;
}

export class ObservatoryEngine {
	private canvas: HTMLCanvasElement;
	private device: GPUDevice | null = null;
	private context: GPUCanvasContext | null = null;
	private format: GPUTextureFormat = 'bgra8unorm';

	private clock: DemoClock;
	private demo: DemoMode;
	private freezeFrame: number | null;
	private rafId = 0;
	private running = false;
	private disposed = false;
	private maxDpr: number;
	private onFrame?: (frame: number, fps: number) => void;
	private lastRenderTs = Number.NEGATIVE_INFINITY;
	private visibilityListenerAttached = false;

	/** Per-frame uniform data (layout: types.PARAMS_FLOATS). */
	readonly params = new Float32Array(PARAMS_FLOATS);
	paramsBuffer: GPUBuffer | null = null;

	private passes: FramePass[] = [];
	private post: PostChain | null = null;
	private _status: EngineStatus = { state: 'booting' };
	private statusListeners = new Set<(s: EngineStatus) => void>();

	/**
	 * Live-event hook (v2.3 living field). Invoked once per frame AFTER p[0..11]
	 * are set but BEFORE the params buffer is written to the GPU, so the live
	 * bridge can drive lanes 12..15 (liveKind/liveStartFrame/liveEnergy/
	 * projectionDays) and mutate node buffers from the real backend WebSocket
	 * stream. `simFrame` is the monotonic (non-wrapping) sim frame so live
	 * event envelopes never pop at the 720-frame loop seam. Allocation-free by
	 * contract — the bridge drains a preallocated ring buffer here.
	 */
	private preFrameHook: ((simFrame: number) => void) | null = null;

	// fps estimate for telemetry only (never sim state)
	private lastRafTs = 0;
	private fpsEstimate = 0;

	// Fixed-timestep accumulator (gafferongames.com/post/fix_your_timestep):
	// wall clock ONLY schedules how many fixed 60Hz ticks to run — sim state
	// stays a pure function of the frame index, so a 120Hz ProMotion display
	// plays the same 12s loop as a 60Hz panel instead of double-speed.
	private accumulatorMs = 0;
	private static readonly FIXED_DT_MS = 1000 / 60;

	/**
	 * Paused (prefers-reduced-motion or the on-page control): the deterministic
	 * clock stops advancing so the ambient orbit + force-sim drift FREEZE, but
	 * the frame still renders and the live preFrameHook still runs — discrete
	 * event pulses (firewall, decay, dream) are information, not decoration, so
	 * they must land even when motion is reduced (WCAG-friendly).
	 */
	private paused = false;

	constructor(opts: EngineOptions) {
		this.canvas = opts.canvas;
		this.demo = opts.demo;
		this.maxDpr = opts.maxDpr ?? 2;
		this.onFrame = opts.onFrame;
		this.clock = new DemoClock({ seed: opts.seed });
		this.freezeFrame =
			typeof opts.freezeFrame === 'number' && Number.isFinite(opts.freezeFrame)
				? ((Math.floor(opts.freezeFrame) % this.clock.framesPerLoop) +
						this.clock.framesPerLoop) %
					this.clock.framesPerLoop
				: null;
		this.params[8] = 1; // brightness default — the void must never eat the field
		this.setCursorPreNdc(999, 999, 0, 0);
	}

	get status(): EngineStatus {
		return this._status;
	}

	get gpuDevice(): GPUDevice | null {
		return this.device;
	}

	get presentationFormat(): GPUTextureFormat {
		return this.format;
	}

	/**
	 * Format every FramePass render pipeline targets: the offscreen HDR scene
	 * texture (post stack input), NOT the swapchain.
	 */
	get sceneFormat(): GPUTextureFormat {
		return SCENE_FORMAT;
	}

	get demoClock(): DemoClock {
		return this.clock;
	}

	onStatus(cb: (s: EngineStatus) => void): () => void {
		this.statusListeners.add(cb);
		cb(this._status);
		return () => this.statusListeners.delete(cb);
	}

	private setStatus(s: EngineStatus) {
		this._status = s;
		for (const cb of this.statusListeners) cb(s);
	}

	/** Register a frame pass (later increments: sim, nodes, edges, path, post). */
	addPass(pass: FramePass): void {
		this.passes.push(pass);
	}

	/**
	 * Deregister a single frame pass and free its GPU resources — WITHOUT tearing
	 * down the device. This is the Spatial Palace primitive: one persistent engine
	 * outlives every route, and organs register/deregister their passes as the
	 * camera flies between regions. Calls the pass's optional `dispose()` (the
	 * RouteFramePass contract) so its buffers/textures are released. `dispose()`
	 * (the whole-engine teardown) remains the ONLY path that destroys the device.
	 * No-op if the pass was never registered.
	 */
	removePass(pass: FramePass): void {
		const i = this.passes.indexOf(pass);
		if (i === -1) return;
		this.passes.splice(i, 1);
		(pass as FramePass & { dispose?: () => void }).dispose?.();
	}

	/**
	 * Deregister ALL frame passes, disposing each, but keep the device/context/
	 * paramsBuffer/post stack alive. Used on a scene swap (fly into a new organ):
	 * clear the old organ's passes, then addPass the new organ's set + re-add the
	 * persistent nav/chrome passes. Does NOT destroy the device — see `dispose()`.
	 */
	clearPasses(): void {
		for (const pass of this.passes) {
			(pass as FramePass & { dispose?: () => void }).dispose?.();
		}
		this.passes.length = 0;
	}

	/**
	 * Register the per-frame live-event hook (v2.3). See `preFrameHook`. Pass
	 * null to detach (the field falls back to the calm deterministic loop).
	 */
	setPreFrameHook(hook: ((simFrame: number) => void) | null): void {
		this.preFrameHook = hook;
	}

	/** Monotonic sim frame (does NOT wrap at the loop period). */
	get totalFrames(): number {
		return this.clock.state.totalFrames;
	}

	/**
	 * Cursor lens for Parallax Engram text. x/y are in the text layer's
	 * pre-aspect-divide NDC space (the inverse of TextLayerPass.pickAt's
	 * screen-space transform), and velocity is in that same space.
	 */
	setCursorPreNdc(x: number, y: number, vx = 0, vy = 0): void {
		this.params[PARAM_IDX.cursorX] = Number.isFinite(x) ? x : 999;
		this.params[PARAM_IDX.cursorY] = Number.isFinite(y) ? y : 999;
		this.params[PARAM_IDX.cursorVx] = Number.isFinite(vx) ? vx : 0;
		this.params[PARAM_IDX.cursorVy] = Number.isFinite(vy) ? vy : 0;
	}

	/**
	 * Freeze/unfreeze the ambient motion (prefers-reduced-motion or the on-page
	 * pause control). Frozen = the clock stops advancing, so the orbit + force
	 * sim hold still; live event pulses (via the preFrameHook) still land.
	 */
	setPaused(paused: boolean): void {
		this.paused = paused;
		this.requestRender();
	}

	get isPaused(): boolean {
		return this.paused;
	}

	/**
	 * Makes the next scheduled frame render immediately. A quiet receipt volume
	 * calls this on selection, replay, or slicer input so interaction never waits
	 * for its low-rate settled cadence.
	 */
	requestRender(): void {
		this.lastRenderTs = Number.NEGATIVE_INFINITY;
	}

	/**
	 * Wall-clock now in ms. The ONE sanctioned wall-clock read (never for sim
	 * state — the DemoClock owns that). The live FSRS decay field legitimately
	 * needs real time to compute "days since last review"; that is a real
	 * external fact, not simulation state, so it does not break determinism.
	 */
	get wallNowMs(): number {
		return Date.now();
	}

	/** Boot WebGPU. Resolves true when running, false when unsupported/error. */
	async start(): Promise<boolean> {
		if (this.disposed) return false;

		const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
		if (!gpu) {
			this.setStatus({
				state: 'unsupported',
				reason: 'WebGPU is not available in this browser.'
			});
			return false;
		}

		let adapter: GPUAdapter | null = null;
		try {
			adapter = await gpu.requestAdapter();
		} catch (e) {
			this.setStatus({
				state: 'error',
				reason: e instanceof Error ? e.message : 'requestAdapter failed'
			});
			return false;
		}
		if (!adapter) {
			this.setStatus({
				state: 'unsupported',
				reason: 'No suitable GPU adapter found.'
			});
			return false;
		}

		try {
			this.device = await adapter.requestDevice();
		} catch (e) {
			this.setStatus({
				state: 'error',
				reason: e instanceof Error ? e.message : 'requestDevice failed'
			});
			return false;
		}
		if (this.disposed) {
			// disposed while awaiting — release the device we just got
			this.device?.destroy();
			this.device = null;
			return false;
		}

		this.device.lost.then((info) => {
			if (this.disposed || info.reason === 'destroyed') return;
			this.setStatus({ state: 'error', reason: `GPU device lost: ${info.message}` });
			this.stopLoop();
		});

		// Surface shader/pipeline validation errors loudly — a silent black
		// field is the worst failure mode an observatory can have.
		this.device.onuncapturederror = (ev: GPUUncapturedErrorEvent) => {
			console.error('[observatory] WebGPU error:', ev.error.message);
		};

		const context = this.canvas.getContext('webgpu');
		if (!context) {
			this.setStatus({ state: 'error', reason: 'Could not get webgpu canvas context.' });
			return false;
		}
		this.context = context;
		this.format = gpu.getPreferredCanvasFormat();
		this.configureContext();

		// Per-frame uniform buffer (written by writeBuffer each frame; no readback).
		this.paramsBuffer = this.device.createBuffer({
			label: 'observatory-params',
			size: this.params.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// Post stack (S1–S4): HDR scene → mip bloom → tonemap/grain/vignette.
		// Sampler + explicit layouts + 4 pipelines build once here; textures
		// are created on the first ensure() (resize or boot frame).
		this.post = new PostChain(this.device, this.paramsBuffer, this.format);

		this.setStatus({ state: 'running' });
		this.attachVisibilityListener();
		// ALWAYS start the loop, even if document.hidden. A genuinely hidden
		// tab never fires rAF (the browser starves it — zero battery cost), so
		// gating the initial start buys nothing; but embedded/headless/capture
		// environments report hidden=true WHILE still firing rAF, and gating
		// here rendered a permanently black canvas for all of them (verified:
		// in-app browser pane, Jul 14 2026). The visibilitychange handler still
		// stops/resumes the loop for real tab switches.
		this.resumeLoop();
		return true;
	}

	/** Resize the drawing buffer to the canvas' CSS size × clamped DPR. */
	resize(): void {
		if (!this.device || !this.context) return;
		const dpr = Math.min(window.devicePixelRatio || 1, this.maxDpr);
		const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
		const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
		if (this.canvas.width !== w || this.canvas.height !== h) {
			this.canvas.width = w;
			this.canvas.height = h;
			// context.configure picks up the new size on next getCurrentTexture
			this.configureContext();
			// HDR scene + bloom mip textures recreate on resize (idempotent).
			this.post?.ensure(w, h);
		}
	}

	private configureContext(): void {
		if (!this.device || !this.context) return;
		this.context.configure({
			device: this.device,
			format: this.format,
			alphaMode: 'opaque'
		});
	}

	private attachVisibilityListener(): void {
		if (this.visibilityListenerAttached || typeof document === 'undefined') return;
		document.addEventListener('visibilitychange', this.handleVisibilityChange);
		this.visibilityListenerAttached = true;
	}

	private handleVisibilityChange = () => {
		if (typeof document === 'undefined') return;
		if (document.hidden) {
			this.stopLoop();
			return;
		}
		this.resumeLoop();
	};

	private resumeLoop(): void {
		if (this.running || this.disposed || !this.device || !this.context || !this.paramsBuffer || !this.post) return;
		this.running = true;
		// Do not turn a tab's hidden time into a visual fast-forward or leave it
		// waiting for a prior settled-rate interval when it becomes visible again.
		this.lastRafTs = 0;
		this.accumulatorMs = 0;
		this.requestRender();
		this.rafId = requestAnimationFrame(this.frame);
	}

	private frameRateFor(frame: number): number {
		// A pass WITHOUT targetFrameRate implicitly demands the full 60 fps —
		// NodeRenderer, LivingField, and every shipped organ animate every frame
		// and never opted into throttling. Only when EVERY registered pass opts
		// into a quieter cadence may the engine settle below 60 (a quiet
		// instrument like the chrono shuttle must never drag the living field
		// down to its own idle rate — that bug shipped once, Jul 14 2026).
		let target = 0;
		for (const pass of this.passes) {
			const requested = pass.targetFrameRate?.(frame);
			if (typeof requested !== 'number' || !Number.isFinite(requested)) {
				target = 60;
				continue;
			}
			target = Math.max(target, requested);
		}
		return Math.max(1, Math.min(60, target || 60));
	}

	private frame = (ts: number) => {
		if (!this.running || !this.device || !this.context || !this.paramsBuffer || !this.post)
			return;

		// Wall delta feeds ONLY the fixed-timestep accumulator. The fps estimate
		// is computed at submit time from RENDERED-frame deltas (below) — the
		// rAF cadence here would report ~60/120 even while the settled-rate
		// governor presents at 10-12 fps, lying to the telemetry strip.
		let deltaMs = 0;
		if (this.lastRafTs > 0) deltaMs = ts - this.lastRafTs;
		this.lastRafTs = ts;

		// Fixed 60Hz timestep: advance the deterministic clock by however many
		// whole ticks of wall time elapsed (clamped so a background tab doesn't
		// fast-forward the story on return). The sequence of frames is identical
		// on every display; only the scheduling reads the wall clock.
		this.accumulatorMs += Math.min(deltaMs, 250);
		let ticked = false;
		// When paused, the clock is frozen (ambient/sim drift stops) — but we
		// keep draining the accumulator so it doesn't fast-forward on resume.
		while (this.accumulatorMs >= ObservatoryEngine.FIXED_DT_MS) {
			if (!this.paused) this.clock.tick();
			this.accumulatorMs -= ObservatoryEngine.FIXED_DT_MS;
			ticked = true;
		}
		// First rAF (deltaMs 0) still renders frame 0.
		void ticked;

		// Capture mode (?frame=N) pins every derived value to one loop frame.
		const state = this.clock.state;
		const frame = this.freezeFrame ?? state.frame;
		const targetFrameRate = this.frameRateFor(frame);
		const minFrameInterval = 1000 / targetFrameRate;
		if (ts - this.lastRenderTs < minFrameInterval) {
			this.rafId = requestAnimationFrame(this.frame);
			return;
		}
		const phase = frame / this.clock.framesPerLoop;

		// Per-frame params (layout must match WGSL Params; types.ts doc block).
		// EVERYTHING derives from the wrapped loop frame — never totalFrames —
		// so the 720-frame loop is periodic by construction: loop k is pixel-
		// identical to loop 1, recordings never pop at the seam, and a
		// ?frame=N still matches what a live viewer sees at that playhead
		// position forever.
		const p = this.params;
		p[0] = frame;
		p[1] = phase;
		// p[2] nodeCount, p[3] edgeCount, p[4] pathCount — set by graph upload (Inc 4+)
		// Breath: exactly 4 cycles per 720-frame loop (0.333 Hz ≈ spec §7.2's
		// ~0.32 Hz) — integer cycles/loop is what makes the seam invisible.
		p[5] = 0.5 + 0.5 * Math.sin(2 * Math.PI * 4 * phase);
		p[6] = this.canvas.width;
		p[7] = this.canvas.height;
		// p[8] brightness — set by the canvas component
		p[9] = demoModeId(this.demo);
		// Ambient seconds. LIVE mode uses monotonic totalFrames so slow ambient
		// motion (LivingField ring_spin/twinkle, msdf sway, organ orbits) never
		// snaps at the 720-frame loop seam — the 12-second pop every organ had.
		// CAPTURE mode (?frame=N) pins to the wrapped loop frame so stills stay
		// byte-stable. Choreography must keep keying off p[0]/p[1] (loop frame/
		// phase), never this lane — ambience is allowed off-loop, stories are not.
		p[10] = this.freezeFrame !== null ? frame / 60 : state.totalFrames / 60;
		// p[11] capture_mode — 1.0 when freezeFrame is active (capture mode).
		// When 1.0, the compute shader skips physics integration so the
		// storage-buffer state stays frozen at the initial upload values,
		// making same URL + frame → identical pixels (spec §4 Inc 9).
		p[11] = this.freezeFrame !== null ? 1.0 : 0.0;

		// v2.3 living field — let the live bridge drive lanes 12..15 and mutate
		// node buffers from the real backend event stream. Runs on the
		// monotonic sim frame so envelopes never pop at the loop seam. Lanes
		// 12..15 persist across frames in `this.params` (the loop above only
		// writes 0..11), so a settled field with no live events stays calm.
		this.preFrameHook?.(state.totalFrames);

		this.device.queue.writeBuffer(this.paramsBuffer, 0, p);

		let swapTex: GPUTexture;
		try {
			swapTex = this.context.getCurrentTexture();
		} catch {
			// canvas hidden/zero-sized this frame — try again next frame
			this.rafId = requestAnimationFrame(this.frame);
			return;
		}
		// No-op unless the size changed — covers the boot frame before any resize.
		this.post.ensure(swapTex.width, swapTex.height);
		const swapView = swapTex.createView();

		const encoder = this.device.createCommandEncoder({ label: 'observatory-frame' });

		// Passes receive the freeze-adjusted frame — same value as params.frame,
		// so capture mode pins hook-driven sim work too, not just uniforms.
		for (const pass of this.passes) pass.compute?.(encoder, frame);

		const render = encoder.beginRenderPass({
			label: 'observatory-main',
			colorAttachments: [
				{
					view: this.post.sceneView,
					clearValue: VOID_CLEAR_HDR,
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});
		for (const pass of this.passes) pass.render?.(render, frame);
		render.end();

		// Post stack: bloom pyramid + composite (tonemap/grain/vignette) to the
		// swapchain — same encoder, single submit, no readback.
		this.post.encode(encoder, swapView);

		this.device.queue.submit([encoder.finish()]);

		// Honest fps: measured across frames actually PRESENTED, so telemetry
		// shows the settled rate when quiet passes throttle the render.
		if (Number.isFinite(this.lastRenderTs) && ts > this.lastRenderTs) {
			this.fpsEstimate = Math.round(1000 / (ts - this.lastRenderTs));
		}
		this.lastRenderTs = ts;
		this.onFrame?.(frame, this.fpsEstimate);
		this.rafId = requestAnimationFrame(this.frame);
	};

	private stopLoop(): void {
		this.running = false;
		if (this.rafId !== 0) {
			cancelAnimationFrame(this.rafId);
			this.rafId = 0;
		}
	}

	dispose(): void {
		if (this.disposed) return;
		this.disposed = true;
		this.stopLoop();
		if (this.visibilityListenerAttached && typeof document !== 'undefined') {
			document.removeEventListener('visibilitychange', this.handleVisibilityChange);
			this.visibilityListenerAttached = false;
		}
		this.paramsBuffer?.destroy();
		this.paramsBuffer = null;
		this.post?.dispose();
		this.post = null;
		this.device?.destroy();
		this.device = null;
		this.context = null;
		this.passes = [];
		this.setStatus({ state: 'disposed' });
		this.statusListeners.clear();
	}
}
