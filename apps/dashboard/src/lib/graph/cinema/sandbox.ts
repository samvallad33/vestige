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

// The storm lives at the world origin, permanently. The camera always looks here
// and is clamped to a safe distance band so the subject can never leave frame.
const ORIGIN = new THREE.Vector3(0, 0, 0);
// Keep the camera in a narrow, fairly FAR band so the contained storm always
// sits comfortably small and centered in frame (a closer camera makes the cloud
// fill — and spill past — the edges once the bloom halo is added).
const MIN_CAM_DIST = 30;
const MAX_CAM_DIST = 44;

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
	// Scene/camera are created in boot() from the three/webgpu module so every
	// object handed to the WebGPU renderer comes from the SAME Three.js instance
	// (avoids the "multiple instances of Three.js" incompatibility — the base
	// three import is used only for the shared Vector3 math type the director
	// mutates, which is identical across instances).
	private scene!: THREE.Scene;
	private camera!: THREE.PerspectiveCamera;
	private storm!: SemanticComputeStorm;
	private post: { renderAsync: () => Promise<void> } | null = null;
	private booted = false;

	/** Camera target the director drives; mirrored into camera.lookAt each frame. */
	readonly target = new THREE.Vector3(0, 0, 0);

	constructor(container: HTMLElement) {
		this.container = container;
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
			Scene: new () => THREE.Scene;
			PerspectiveCamera: new (fov: number, aspect: number, near: number, far: number) => THREE.PerspectiveCamera;
			Color: new (hex: number) => THREE.Color;
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

		// Fail loud if the dynamic import didn't yield the expected constructors,
		// instead of a cryptic "undefined is not a constructor" later.
		if (!webgpu.WebGPURenderer || !webgpu.Scene || !webgpu.PerspectiveCamera || !webgpu.Color) {
			throw new Error('[cinema] three/webgpu is missing expected exports');
		}

		// Build scene + camera from the SAME (webgpu) module instance the
		// renderer + storm use, so all objects are instance-compatible.
		const w = Math.max(1, this.container.clientWidth);
		const h = Math.max(1, this.container.clientHeight);
		this.scene = new webgpu.Scene();
		this.scene.background = new webgpu.Color(0x02020a);
		this.camera = new webgpu.PerspectiveCamera(60, w / h, 0.1, 2000);
		this.camera.position.set(0, 18, 60);

		const renderer = new this.deps.WebGPURenderer({ antialias: true, alpha: false });
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.setSize(w, h);
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
			if (typeof scenePass?.setMRT !== 'function' || typeof scenePass?.getTextureNode !== 'function') {
				throw new Error('three/tsl pass() API mismatch — setMRT/getTextureNode missing');
			}
			scenePass.setMRT(mrt({ output, emissive }));
			const outputTex = scenePass.getTextureNode('output');
			const emissiveTex = scenePass.getTextureNode('emissive');
			// Gentler bloom (strength 0.6, threshold 0.35) so it accents the bright
			// cores instead of washing the whole colored cloud to white.
			const bloomed = this.deps.bloomMod.bloom(emissiveTex, 0.6, 0.65, 0.35);
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

	/** Retarget the storm's MODE/ignition for a beat. The storm is permanently
	 * centered at the WORLD ORIGIN (see render) so it is always dead-center in
	 * frame — worldPos here only conveys which node, not where the storm sits. */
	transitionTo(role: SemanticRole, _worldPos: THREE.Vector3): void {
		if (!this.booted) return;
		this.storm.transitionTo(role, ORIGIN);
	}

	/** Render one frame. The storm is pinned to the origin and the camera always
	 * looks at the origin, so the storm CANNOT leave the frame. The director
	 * varies only the camera's orbital position/angle (set via cameraRef), and we
	 * clamp that to a safe distance band here as a final guarantee. */
	async render(deltaSeconds: number): Promise<void> {
		if (!this.booted) return;

		// Hard guarantee: clamp the camera into a distance band from origin so a
		// runaway director move can never push the subject out of view, then look
		// dead at the origin where the storm lives.
		const distToOrigin = this.camera.position.length();
		if (distToOrigin < MIN_CAM_DIST || distToOrigin > MAX_CAM_DIST || !Number.isFinite(distToOrigin)) {
			const d = Math.min(MAX_CAM_DIST, Math.max(MIN_CAM_DIST, distToOrigin || MAX_CAM_DIST));
			if (distToOrigin > 1e-3) this.camera.position.setLength(d);
			else this.camera.position.set(0, 12, d);
		}
		this.camera.lookAt(ORIGIN);

		// Size the containment sphere to the camera's VERTICAL FOV at the origin
		// (the limiting dimension on a landscape frame). 0.82 lets the storm fill
		// most of the frame; the storm's internal shell sits well inside this and
		// the hard boundary snap keeps the bloom halo from spilling past the edge.
		const dist = this.camera.position.length();
		const vfov = (this.camera.fov * Math.PI) / 180;
		const fitRadius = Math.tan(vfov / 2) * dist * 0.82;
		this.storm.setContainRadius(fitRadius);

		await this.storm.update(deltaSeconds);
		if (this.post) await this.post.renderAsync();
		else await this.renderer.renderAsync(this.scene, this.camera);
	}

	resize(): void {
		if (!this.booted) return;
		const w = Math.max(1, this.container.clientWidth);
		const h = Math.max(1, this.container.clientHeight);
		this.camera.aspect = w / h;
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
