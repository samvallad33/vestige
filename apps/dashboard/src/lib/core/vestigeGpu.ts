// VestigeGPU — reusable raw-WebGPU core, extracted from the proven launch engine
// (lib/launch/RawVestigeEngine.svelte). Same battle-tested boot path: 3-tier
// adapter fallback, adaptive device limits, device-loss recovery, and a hard
// guarantee that callers can fall back to Canvas2D when WebGPU is absent.
//
// This is intentionally NOT a framework. It is a thin, dependency-free handle
// around the bits every Vestige GPU surface needs, so the dashboard backdrop and
// (later) the GPU data-viz layers share one boot path instead of re-deriving it.
// No Three.js. No npm GPU deps. Everything lives on the bare metal.

/** Minimal shape of the global WebGPU entry point, kept `any` so the dashboard
 *  builds on toolchains whose lib.dom may predate the WebGPU types. */
type GpuApi = {
	requestAdapter: (opts?: Record<string, unknown>) => Promise<any>;
	getPreferredCanvasFormat: () => string;
};

function getGpu(): GpuApi | null {
	const g = (globalThis as any).navigator?.gpu;
	return g ?? null;
}

/** Is raw WebGPU even worth attempting? Cheap synchronous probe — the real
 *  decision still comes from whether `boot()` returns a device. */
export function webgpuAvailable(): boolean {
	return !!getGpu();
}

/** Race a promise against a timeout so a hung adapter/device request can never
 *  wedge the boot sequence (Safari/iOS have been seen to stall here). */
function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		const t = setTimeout(() => reject(new Error('gpu-timeout')), ms);
		p.then(
			(v) => {
				clearTimeout(t);
				resolve(v);
			},
			(e) => {
				clearTimeout(t);
				reject(e);
			}
		);
	});
}

/** Try progressively less demanding adapters: high-performance discrete first,
 *  then the default, then the compatibility tier (older integrated GPUs). This
 *  is the exact ladder the launch hero proved across machines. */
async function requestBestAdapter(gpu: GpuApi): Promise<any | null> {
	const attempts: Array<Record<string, unknown>> = [
		{ powerPreference: 'high-performance' },
		{},
		{ featureLevel: 'compatibility' }
	];
	for (const opts of attempts) {
		try {
			const adapter = await withTimeout(gpu.requestAdapter(opts), 2500);
			if (adapter) return adapter;
		} catch {
			// try the next, less demanding, tier
		}
	}
	return null;
}

/** Request a device that honours the adapter's own advertised limits (never
 *  exceed them), falling back to a default device if the limited request fails. */
async function requestBestDevice(adapter: any): Promise<any | null> {
	try {
		const limits: Record<string, number> = {};
		// Ask for exactly what the adapter advertises for the two limits the
		// backdrop cares about — storage buffer + max buffer — clamped to the
		// spec-default minimums so we never request below baseline.
		const want = (k: string, floor: number) => {
			const v = adapter?.limits?.[k];
			if (typeof v === 'number' && v > floor) limits[k] = v;
		};
		want('maxStorageBufferBindingSize', 134_217_728); // 128 MiB spec default
		want('maxBufferSize', 268_435_456);
		return await withTimeout(
			adapter.requestDevice({ requiredLimits: limits }),
			3500
		);
	} catch {
		try {
			return await withTimeout(adapter.requestDevice(), 3500);
		} catch {
			return null;
		}
	}
}

export type VestigeGpuHandle = {
	device: any;
	context: any;
	/** preferred 8-bit swapchain format for the visible canvas */
	format: string;
	/** HDR accumulation format — additive, unclamped, so glow exceeds 1.0 */
	hdrFormat: 'rgba16float';
	/** the adapter, kept so callers can feature-detect (subgroups, etc.) */
	adapter: any;
};

/** Boot WebGPU on a canvas. Returns a handle, or `null` when WebGPU is
 *  unavailable / boot failed — in which case the caller must use its Canvas2D
 *  fallback (load-bearing: pre-iOS-26 iPhones have NO WebGPU). */
export async function bootVestigeGpu(
	canvas: HTMLCanvasElement
): Promise<VestigeGpuHandle | null> {
	const gpu = getGpu();
	if (!gpu) return null;

	const adapter = await requestBestAdapter(gpu);
	if (!adapter) return null;

	const device = await requestBestDevice(adapter);
	if (!device) return null;

	const context = canvas.getContext('webgpu') as any;
	if (!context) return null;

	const format = gpu.getPreferredCanvasFormat();
	context.configure({ device, format, alphaMode: 'premultiplied' });

	return { device, context, format, hdrFormat: 'rgba16float', adapter };
}

/** Cap device-pixel-ratio so a full-viewport field never shades 4x the pixels
 *  on a retina/mobile screen — the single biggest thermal/throughput lever. */
export function clampDpr(max = 1.5): number {
	const dpr = (globalThis as any).devicePixelRatio ?? 1;
	return Math.min(Math.max(dpr, 1), max);
}

/** Has the user asked for reduced motion? Backdrops must honour this. */
export function prefersReducedMotion(): boolean {
	try {
		return (
			(globalThis as any).matchMedia?.('(prefers-reduced-motion: reduce)')
				?.matches ?? false
		);
	} catch {
		return false;
	}
}
