/**
 * Cognitive Observatory — post-processing chain (S1–S4).
 *
 * Owns EVERY GPU resource of the post stack; the engine holds exactly one
 * field (`post`). The scene renders into an offscreen HDR texture
 * (rgba16float — core WebGPU, no feature gate) instead of the swapchain, then
 * PostChain encodes, on the SAME encoder (single submit, zero readback):
 *
 *   1..N      threshold-free mip bloom: progressive 13-tap Jimenez downsample
 *             (Karis average on the first hop kills fireflies), then 9-tap
 *             tent upsample ACCUMULATED additively up the chain,
 *   final     composite → swapchain: scene + BLOOM_STRENGTH·bloom/mipCount →
 *             Khronos PBR Neutral → seeded grain → cos⁴ vignette.
 *
 * Bloom RADIUS varies with viewport size (mipCount grows with the canvas);
 * BRIGHTNESS does not — the composite divides by textureNumLevels, so the
 * up-chain's DC gain of exactly mipCount normalizes to 1.
 *
 * Zero per-frame uniforms/allocations: pipelines + explicit layouts build
 * once in the constructor; textures/views/bind groups rebuild only when
 * ensure() sees a new size; the shaders read sizes via textureDimensions /
 * textureNumLevels straight from the bound views.
 *
 * SUBRESOURCE RULE (load-bearing): the blur chain binds ONLY single-
 * subresource views (mipView[i]) — sampling mip i+1 while rendering mip i is
 * valid only because the subresources are disjoint. The full-mip view exists
 * ONLY in the composite bind group, where bloomTex is never an attachment.
 */

import { planBloomMips, type BloomMipPlan } from './mip-plan';
import { postWGSL } from './shaders/post.wgsl';

// Tuning constants — defined next to the WGSL they are interpolated into;
// re-exported here as the public constant surface of the post stack.
export {
	BLOOM_STRENGTH,
	BLOOM_CHROMATIC_TEXELS,
	GRAIN_AMP,
	VIGNETTE_LIFT,
	VIGNETTE_TAN
} from './shaders/post.wgsl';

/**
 * Format every FramePass render pipeline targets — the offscreen HDR scene
 * texture. rgba16float is core WebGPU: render-attachable, blendable, and
 * filterable with no feature gate.
 */
export const SCENE_FORMAT: GPUTextureFormat = 'rgba16float';

const ADDITIVE_BLEND: GPUBlendState = {
	color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
	alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
};

export class PostChain {
	private device: GPUDevice;
	private paramsBuffer: GPUBuffer;
	private samp: GPUSampler;

	private blurLayout: GPUBindGroupLayout;
	private compositeLayout: GPUBindGroupLayout;
	private pipeDownFirst: GPURenderPipeline;
	private pipeDown: GPURenderPipeline;
	private pipeUp: GPURenderPipeline;
	private pipeComposite: GPURenderPipeline;

	// Size-dependent resources — (re)built by ensure() only.
	private width = 0;
	private height = 0;
	private plan: BloomMipPlan | null = null;
	private sceneTex: GPUTexture | null = null;
	private _sceneView: GPUTextureView | null = null;
	private bloomTex: GPUTexture | null = null;
	private mipViews: GPUTextureView[] = [];
	private bloomFullView: GPUTextureView | null = null;
	private downBind: GPUBindGroup[] = [];
	private upBind: GPUBindGroup[] = [];
	private compositeBind: GPUBindGroup | null = null;

	constructor(
		device: GPUDevice,
		paramsBuffer: GPUBuffer,
		presentationFormat: GPUTextureFormat
	) {
		this.device = device;
		this.paramsBuffer = paramsBuffer;

		this.samp = device.createSampler({
			label: 'observatory-post-sampler',
			minFilter: 'linear',
			magFilter: 'linear',
			addressModeU: 'clamp-to-edge',
			addressModeV: 'clamp-to-edge'
		});

		const module = device.createShaderModule({
			label: 'observatory-post',
			code: postWGSL
		});

		// EXPLICIT layouts (WGSL trap #6 structurally dead): one blur layout
		// serves all three blur pipelines — explicit layouts may contain entries
		// an entry point ignores, so down/up bind groups are interchangeable.
		this.blurLayout = device.createBindGroupLayout({
			label: 'observatory-post-blur-layout',
			entries: [
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: 'float', viewDimension: '2d' }
				},
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } }
			]
		});
		this.compositeLayout = device.createBindGroupLayout({
			label: 'observatory-post-composite-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{
					binding: 3,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: 'float', viewDimension: '2d' }
				},
				{
					binding: 4,
					visibility: GPUShaderStage.FRAGMENT,
					texture: { sampleType: 'float', viewDimension: '2d' }
				}
			]
		});

		const blurPipeLayout = device.createPipelineLayout({
			label: 'observatory-post-blur-pipe-layout',
			bindGroupLayouts: [this.blurLayout]
		});
		const compositePipeLayout = device.createPipelineLayout({
			label: 'observatory-post-composite-pipe-layout',
			bindGroupLayouts: [this.compositeLayout]
		});

		const makePipe = (
			label: string,
			layout: GPUPipelineLayout,
			entryPoint: string,
			format: GPUTextureFormat,
			blend?: GPUBlendState
		): GPURenderPipeline =>
			device.createRenderPipeline({
				label,
				layout,
				vertex: { module, entryPoint: 'vs_fullscreen' },
				fragment: { module, entryPoint, targets: [{ format, blend }] },
				primitive: { topology: 'triangle-list' }
			});

		this.pipeDownFirst = makePipe(
			'observatory-post-down-karis',
			blurPipeLayout,
			'fs_downsample_karis',
			SCENE_FORMAT
		);
		this.pipeDown = makePipe(
			'observatory-post-down',
			blurPipeLayout,
			'fs_downsample',
			SCENE_FORMAT
		);
		this.pipeUp = makePipe(
			'observatory-post-up',
			blurPipeLayout,
			'fs_upsample_tent',
			SCENE_FORMAT,
			ADDITIVE_BLEND
		);
		// Composite targets the swapchain — presentation format comes from the
		// constructor arg; never hardcode bgra8unorm.
		this.pipeComposite = makePipe(
			'observatory-post-composite',
			compositePipeLayout,
			'fs_composite',
			presentationFormat
		);
	}

	/**
	 * The main scene pass' color attachment (offscreen HDR). ensure() always
	 * precedes use in the engine's frame loop.
	 */
	get sceneView(): GPUTextureView {
		if (!this._sceneView) {
			throw new Error('PostChain.ensure() must run before sceneView is used');
		}
		return this._sceneView;
	}

	/**
	 * Idempotent size-compare: recreates textures/views/bind groups iff the
	 * size changed (clamped ≥ 1). Called from engine.resize() AND from the
	 * frame loop with the swapchain texture's dims (covers the boot frame).
	 */
	ensure(width: number, height: number): void {
		const w = Math.max(1, Math.floor(width));
		const h = Math.max(1, Math.floor(height));
		if (w === this.width && h === this.height && this.sceneTex !== null) return;
		this.width = w;
		this.height = h;

		this.sceneTex?.destroy();
		this.bloomTex?.destroy();

		this.sceneTex = this.device.createTexture({
			label: 'observatory-scene-hdr',
			size: [w, h],
			format: SCENE_FORMAT,
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
		});
		this._sceneView = this.sceneTex.createView({ label: 'observatory-scene-hdr-view' });

		const plan = planBloomMips(w, h);
		this.plan = plan;
		this.bloomTex = this.device.createTexture({
			label: 'observatory-bloom-mips',
			size: [plan.baseW, plan.baseH],
			format: SCENE_FORMAT,
			mipLevelCount: plan.mipCount,
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
		});

		// SINGLE-subresource views for the blur chain (see header rule) …
		const bloomTex = this.bloomTex;
		this.mipViews = Array.from({ length: plan.mipCount }, (_, i) =>
			bloomTex.createView({
				label: `observatory-bloom-mip-${i}`,
				baseMipLevel: i,
				mipLevelCount: 1
			})
		);
		// … and the full-mip view for the composite ONLY (textureNumLevels).
		this.bloomFullView = bloomTex.createView({ label: 'observatory-bloom-full' });

		// Bind groups rebuild here only — zero per-frame allocations.
		const sceneView = this._sceneView;
		this.downBind = this.mipViews.map((_, i) =>
			this.device.createBindGroup({
				label: `observatory-bloom-down-bind-${i}`,
				layout: this.blurLayout,
				entries: [
					{ binding: 1, resource: i === 0 ? sceneView : this.mipViews[i - 1] },
					{ binding: 2, resource: this.samp }
				]
			})
		);
		this.upBind = [];
		for (let i = 0; i + 1 < plan.mipCount; i++) {
			this.upBind.push(
				this.device.createBindGroup({
					label: `observatory-bloom-up-bind-${i}`,
					layout: this.blurLayout,
					entries: [
						{ binding: 1, resource: this.mipViews[i + 1] },
						{ binding: 2, resource: this.samp }
					]
				})
			);
		}
		this.compositeBind = this.device.createBindGroup({
			label: 'observatory-post-composite-bind',
			layout: this.compositeLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.paramsBuffer } },
				{ binding: 2, resource: this.samp },
				{ binding: 3, resource: sceneView },
				{ binding: 4, resource: this.bloomFullView }
			]
		});
	}

	/**
	 * Encode the whole post stack: 2(N−1)+2 tiny fullscreen passes, each
	 * `setPipeline; setBindGroup; draw(3)`. Same encoder as the scene pass.
	 */
	encode(encoder: GPUCommandEncoder, swapchainView: GPUTextureView): void {
		const plan = this.plan;
		if (!plan || !this.compositeBind) return;
		const n = plan.mipCount;

		// Downsample: scene (full res) → mip 0 (Karis), then mip i−1 → i.
		for (let i = 0; i < n; i++) {
			const pass = encoder.beginRenderPass({
				label: `observatory-bloom-down-${i}`,
				colorAttachments: [{ view: this.mipViews[i], loadOp: 'clear', storeOp: 'store' }]
			});
			pass.setPipeline(i === 0 ? this.pipeDownFirst : this.pipeDown);
			pass.setBindGroup(0, this.downBind[i]);
			pass.draw(3);
			pass.end();
		}

		// Upsample: tent of mip i+1 accumulates ADDITIVELY onto the stored
		// downsample at mip i (loadOp 'load' + one/one blend). DC gain becomes
		// exactly n — normalized in the composite. Runs zero times when n = 1.
		for (let i = n - 2; i >= 0; i--) {
			const pass = encoder.beginRenderPass({
				label: `observatory-bloom-up-${i}`,
				colorAttachments: [{ view: this.mipViews[i], loadOp: 'load', storeOp: 'store' }]
			});
			pass.setPipeline(this.pipeUp);
			pass.setBindGroup(0, this.upBind[i]);
			pass.draw(3);
			pass.end();
		}

		// Composite to the swapchain: bloom-add → tonemap → grain → vignette.
		const pass = encoder.beginRenderPass({
			label: 'observatory-post-composite',
			colorAttachments: [{ view: swapchainView, loadOp: 'clear', storeOp: 'store' }]
		});
		pass.setPipeline(this.pipeComposite);
		pass.setBindGroup(0, this.compositeBind);
		pass.draw(3);
		pass.end();
	}

	dispose(): void {
		this.sceneTex?.destroy();
		this.bloomTex?.destroy();
		this.sceneTex = null;
		this.bloomTex = null;
		this._sceneView = null;
		this.bloomFullView = null;
		this.mipViews = [];
		this.downBind = [];
		this.upBind = [];
		this.compositeBind = null;
		this.plan = null;
		this.width = 0;
		this.height = 0;
	}
}
