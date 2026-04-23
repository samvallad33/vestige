import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

export interface SceneContext {
	scene: THREE.Scene;
	camera: THREE.PerspectiveCamera;
	renderer: THREE.WebGLRenderer;
	controls: OrbitControls;
	composer: EffectComposer;
	bloomPass: UnrealBloomPass;
	raycaster: THREE.Raycaster;
	mouse: THREE.Vector2;
	lights: {
		ambient: THREE.AmbientLight;
		point1: THREE.PointLight;
		point2: THREE.PointLight;
	};
	starfield: THREE.Points;
}

function createStarfield(): THREE.Points {
	// 2000 dim points distributed on a spherical shell at radius 600-1000.
	// Purely decorative depth cue — never intersects the graph area and
	// sits below the bloom threshold so it doesn't bloom.
	const starCount = 2000;
	const positions = new Float32Array(starCount * 3);
	const colors = new Float32Array(starCount * 3);
	for (let i = 0; i < starCount; i++) {
		const theta = Math.random() * Math.PI * 2;
		const phi = Math.acos(2 * Math.random() - 1);
		const r = 600 + Math.random() * 400;
		positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
		positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
		positions[i * 3 + 2] = r * Math.cos(phi);
		// Subtle colour variation — mostly cool white, some violet tint.
		const tint = Math.random();
		colors[i * 3] = 0.55 + tint * 0.25;
		colors[i * 3 + 1] = 0.55 + tint * 0.15;
		colors[i * 3 + 2] = 0.75 + tint * 0.25;
	}
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
	const mat = new THREE.PointsMaterial({
		size: 1.6,
		sizeAttenuation: true,
		vertexColors: true,
		transparent: true,
		opacity: 0.6,
		depthWrite: false,
		blending: THREE.AdditiveBlending,
	});
	return new THREE.Points(geo, mat);
}

export function createScene(container: HTMLDivElement): SceneContext {
	const scene = new THREE.Scene();
	// Darker-than-black background with a subtle colour cast. Combined with
	// the starfield and reduced fog, the void has depth instead of reading
	// as a broken shader canvas.
	scene.background = new THREE.Color(0x05050f);
	// Fog density reduced 0.008 → 0.0035 — the old value was killing every
	// edge and node past 50 units. Lighter colour blends into the background.
	scene.fog = new THREE.FogExp2(0x0a0a1a, 0.0035);

	const camera = new THREE.PerspectiveCamera(
		60,
		container.clientWidth / container.clientHeight,
		0.1,
		2000
	);
	camera.position.set(0, 30, 80);

	const renderer = new THREE.WebGLRenderer({
		antialias: true,
		alpha: true,
		powerPreference: 'high-performance',
	});
	renderer.setSize(container.clientWidth, container.clientHeight);
	renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.25;
	container.appendChild(renderer.domElement);

	const controls = new OrbitControls(camera, renderer.domElement);
	controls.enableDamping = true;
	controls.dampingFactor = 0.05;
	controls.rotateSpeed = 0.5;
	controls.zoomSpeed = 0.8;
	// Distance clamps — the camera starts at ~86 units from origin
	// (position.set(0, 30, 80)). The graph's force-directed layout seats
	// most nodes within a ~120-unit radius. 500 was dramatically out of
	// scale — the user could zoom out until every node was one pixel on
	// a black starfield (issue reported 2026-04-23). 180 keeps the full
	// graph in frame with nodes still readable; 12 prevents zooming inside
	// a node and losing orientation.
	controls.minDistance = 12;
	controls.maxDistance = 180;
	controls.autoRotate = true;
	controls.autoRotateSpeed = 0.3;

	const composer = new EffectComposer(renderer);
	composer.addPass(new RenderPass(scene, camera));
	// Bloom retuned for radial-gradient glow sprites (issue #31 fix):
	//   strength 0.8 → 0.55 — gentler, avoids the old blown-out look
	//   radius   0.4 → 0.6  — softer falloff, diffuses cleanly through glow
	//   threshold 0.85 → 0.2 — let mid-tones bloom instead of highlights only
	const bloomPass = new UnrealBloomPass(
		new THREE.Vector2(container.clientWidth, container.clientHeight),
		0.55,
		0.6,
		0.2
	);
	composer.addPass(bloomPass);

	const ambient = new THREE.AmbientLight(0x2a2a5a, 0.7);
	scene.add(ambient);

	const point1 = new THREE.PointLight(0x6366f1, 1.8, 240);
	point1.position.set(50, 50, 50);
	scene.add(point1);

	const point2 = new THREE.PointLight(0xa855f7, 1.2, 240);
	point2.position.set(-50, -30, -50);
	scene.add(point2);

	const starfield = createStarfield();
	scene.add(starfield);

	const raycaster = new THREE.Raycaster();
	raycaster.params.Points = { threshold: 2 };
	const mouse = new THREE.Vector2();

	return {
		scene,
		camera,
		renderer,
		controls,
		composer,
		bloomPass,
		raycaster,
		mouse,
		lights: { ambient, point1, point2 },
		starfield,
	};
}

export function resizeScene(ctx: SceneContext, container: HTMLDivElement) {
	const w = container.clientWidth;
	const h = container.clientHeight;
	ctx.camera.aspect = w / h;
	ctx.camera.updateProjectionMatrix();
	ctx.renderer.setSize(w, h);
	ctx.composer.setSize(w, h);
}

export function disposeScene(ctx: SceneContext) {
	ctx.scene.traverse((obj: THREE.Object3D) => {
		if (obj instanceof THREE.Mesh || obj instanceof THREE.InstancedMesh) {
			obj.geometry?.dispose();
			if (Array.isArray(obj.material)) {
				obj.material.forEach((m: THREE.Material) => m.dispose());
			} else if (obj.material) {
				(obj.material as THREE.Material).dispose();
			}
		}
	});
	ctx.renderer.dispose();
	ctx.composer.dispose();
}
