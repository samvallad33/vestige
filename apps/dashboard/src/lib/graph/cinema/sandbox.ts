// Memory Cinema — the isolated WebGPU sandbox.
//
// Boots a SEPARATE WebGPU canvas + scene on Cinema launch. The legacy WebGL
// graph (nebula, grain, every current user's experience) is never touched —
// zero regression by construction. Inside the sandbox: the SemanticComputeStorm
// + selective MRT emissive bloom, driven by the CinemaDirector's beats.
//
// Everything here is dynamically imported (three/webgpu, three/tsl, storm.ts)
// so the heavy WebGPU bundle stays out of the main app. If WebGPU is
// unavailable, isSupported() returns false and the caller falls back to the
// camera-only flythrough on the existing canvas (captions still play).

import * as THREE from 'three';
import type { SemanticRole, SemanticComputeStorm } from './storm';

export function isWebGPUSupported(): boolean {
	return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

interface SandboxDeps {
	WebGPURenderer: new (params: object) => {
		init: () => Promise<void>;
		setSize: (w: number, h: number) => void;
		setPixelRatio: (r: number) => void;
		renderAsync: (scene: THREE.Scene, camera: THREE.Camera) => Promise<void>;
		computeAsync: (node: unknown) => Promise<void>;
		domElement: HTMLCanvasElement;
		dispose?: () => void;
	};
	PostProcessing: new (renderer: unknown) => { renderAsync: () => Promise<void>; outputNode: unknown };
	StormCtor: typeof SemanticComputeStorm;
	tsl: typeof import('three/tsl');
	bloomMod: { bloom: (node: unknown, strength?: number, radius?: number, threshold?: number) => unknown };
}

export class CinemaSandbox {
	private container: HTMLElement;
	private deps!: SandboxDeps;
	private renderer!: SandboxDeps['WebGPURenderer']['prototype'];
	private scene = new THREE.Scene();
	private camera: THREE.PerspectiveCamera;
	private storm!: SemanticComputeStorm;
	private post: { renderAsync: () => Promise<void> } | null = null;
	private booted = false;

	/** Camera target the director drives; mirrored into camera.lookAt each frame. */
	readonly target = new THREE.Vector3(0, 0, 0);

	constructor(container: HTMLElement) {
		this.container = container;
		this.camera = new THREE.PerspectiveCamera(
			60,
			container.clientWidth / Math.max(1, container.clientHeight),
			0.1,
			2000
		);
		this.camera.position.set(0, 18, 60);
		this.scene.background = new THREE.Color(0x02020a);
	}

	get cameraRef(): THREE.PerspectiveCamera {
		return this.camera;
	}

	/**
	 * Boot the WebGPU pipeline. Throws if WebGPU is unsupported or init fails —
	 * the caller treats a throw as "fall back to camera-only mode".
	 */
	async boot(): Promise<void> {
		if (this.booted) return;
		if (!isWebGPUSupported()) throw new Error('WebGPU not supported');

		// Dynamic imports keep three/webgpu out of the main bundle.
		const webgpu = (await import('three/webgpu')) as unknown as {
			WebGPURenderer: SandboxDeps['WebGPURenderer'];
			PostProcessing: SandboxDeps['PostProcessing'];
		};
		const tsl = (await import('three/tsl')) as typeof import('three/tsl');
		// bloom() lives in the TSL display helpers; import the node module.
		const bloomMod = (await import(
			'three/examples/jsm/tsl/display/BloomNode.js'
		)) as unknown as SandboxDeps['bloomMod'];
		const { SemanticComputeStorm } = await import('./storm');

		this.deps = {
			WebGPURenderer: webgpu.WebGPURenderer,
			PostProcessing: webgpu.PostProcessing,
			StormCtor: SemanticComputeStorm,
			tsl,
			bloomMod,
		};

		const renderer = new this.deps.WebGPURenderer({ antialias: true, alpha: false });
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.setSize(this.container.clientWidth, this.container.clientHeight);
		// CRITICAL FOOTGUN: WebGPU init is async. Must await before first render
		// or the canvas silently draws nothing.
		await renderer.init();
		this.container.appendChild(renderer.domElement);
		this.renderer = renderer;

		// The compute storm (150k GPU particles).
		this.storm = new this.deps.StormCtor(renderer, this.scene, {});

		// Selective MRT bloom: scene pass emits an emissive MRT; bloom only the
		// emissive channel so the storm blazes against the void without washing
		// the whole frame to grey. Falls back to a plain pass if MRT setup
		// throws on a given driver.
		try {
			const { pass, mrt, output, emissive } = this.deps.tsl as unknown as {
				pass: (s: THREE.Scene, c: THREE.Camera) => {
					setMRT: (m: unknown) => void;
					getTextureNode: (name: string) => unknown;
				};
				mrt: (cfg: Record<string, unknown>) => unknown;
				output: unknown;
				emissive: unknown;
			};
			const scenePass = pass(this.scene, this.camera);
			scenePass.setMRT(mrt({ output, emissive }));
			const outputTex = scenePass.getTextureNode('output');
			const emissiveTex = scenePass.getTextureNode('emissive');
			const bloomed = this.deps.bloomMod.bloom(emissiveTex, 1.1, 0.6, 0.0);
			const post = new this.deps.PostProcessing(renderer);
			(post as unknown as { outputNode: unknown }).outputNode = (
				outputTex as { add: (n: unknown) => unknown }
			).add(bloomed);
			this.post = post as unknown as { renderAsync: () => Promise<void> };
		} catch (e) {
			// MRT/bloom unavailable on this driver — render straight, no crash.
			console.warn('[cinema] selective bloom unavailable, rendering without MRT:', e);
			this.post = null;
		}

		this.booted = true;
	}

	/** Retarget the storm + look the camera at the beat (called by the director). */
	transitionTo(role: SemanticRole, worldPos: THREE.Vector3): void {
		if (!this.booted) return;
		this.storm.transitionTo(role, worldPos);
	}

	/** Render one frame. Camera is driven externally (director mutates position/target). */
	async render(deltaSeconds: number): Promise<void> {
		if (!this.booted) return;
		this.camera.lookAt(this.target);
		await this.storm.update(deltaSeconds);
		if (this.post) await this.post.renderAsync();
		else await this.renderer.renderAsync(this.scene, this.camera);
	}

	resize(): void {
		if (!this.booted) return;
		const w = this.container.clientWidth;
		const h = this.container.clientHeight;
		this.camera.aspect = w / Math.max(1, h);
		this.camera.updateProjectionMatrix();
		this.renderer.setSize(w, h);
	}

	dispose(): void {
		if (!this.booted) return;
		this.storm?.dispose();
		this.renderer?.dispose?.();
		if (this.renderer?.domElement?.parentNode) {
			this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
		}
		this.booted = false;
	}
}
