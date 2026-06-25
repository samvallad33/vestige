// Living Brain Neural Flow — a GPGPU particle organism.
//
// 65,536 particles simulated on the GPU via a TWO-variable GPUComputationRenderer
// (position + velocity, ping-ponged). Flow runs TANGENT to a two-lobe brain SDF
// so particles glide along the cortex instead of dispersing into a starfield.
// The cursor parts the field like liquid (speed-gated repulsion + signed swirl)
// and it heals back to the brain via a Hooke spring. Color is physical thin-film
// iridescence (spectral_zucconi6), not HSV. An entrance timeline inhales chaos
// into a dense, breathing, iridescent brain.
//
// Built from a researched spec (docs/launch/living-brain-hero-spec.json).
// WebGL2, ships everywhere. Degrades to 16,384 particles on low-end.

import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

export interface NeuralFlowOptions {
	seed?: number;
	reducedMotion?: boolean;
}

// ---- shared GLSL: brain SDF + curl-of-SDF-modulated-noise (divergence free) --
const GLSL_COMMON = /* glsl */ `
	vec3 hash3(vec3 p){
		p = vec3(dot(p,vec3(127.1,311.7,74.7)),
		         dot(p,vec3(269.5,183.3,246.1)),
		         dot(p,vec3(113.5,271.9,124.6)));
		return -1.0 + 2.0*fract(sin(p)*43758.5453123);
	}
	float snoise(vec3 p){
		vec3 i = floor(p); vec3 f = fract(p);
		vec3 u = f*f*(3.0-2.0*f);
		return mix(mix(mix(dot(hash3(i+vec3(0,0,0)),f-vec3(0,0,0)),
		                   dot(hash3(i+vec3(1,0,0)),f-vec3(1,0,0)),u.x),
		               mix(dot(hash3(i+vec3(0,1,0)),f-vec3(0,1,0)),
		                   dot(hash3(i+vec3(1,1,0)),f-vec3(1,1,0)),u.x),u.y),
		           mix(mix(dot(hash3(i+vec3(0,0,1)),f-vec3(0,0,1)),
		                   dot(hash3(i+vec3(1,0,1)),f-vec3(1,0,1)),u.x),
		               mix(dot(hash3(i+vec3(0,1,1)),f-vec3(0,1,1)),
		                   dot(hash3(i+vec3(1,1,1)),f-vec3(1,1,1)),u.x),u.y),u.z);
	}
	float sdEllipsoid(vec3 p, vec3 r){ float k0=length(p/r); float k1=length(p/(r*r)); return k0*(k0-1.0)/k1; }
	float brainSDF(vec3 p){
		float L = sdEllipsoid(p-vec3(-0.32,0.0,0.0), vec3(0.85,0.95,1.15));
		float R = sdEllipsoid(p-vec3( 0.32,0.0,0.0), vec3(0.85,0.95,1.15));
		float d = min(L,R);
		d += 0.06*sin(8.0*p.x)*sin(7.0*p.y)*sin(6.0*p.z); // sulci/gyri wrinkles
		return d;
	}
	vec3 sdfNormal(vec3 p){
		const vec2 e = vec2(1.0,-1.0)*0.004;
		return normalize(e.xyy*brainSDF(p+e.xyy)+e.yyx*brainSDF(p+e.yyx)+
		                 e.yxy*brainSDF(p+e.yxy)+e.xxx*brainSDF(p+e.xxx));
	}
	vec3 potential(vec3 p, float t){ return sdfNormal(p)*snoise(p*1.6 + vec3(0.0,0.0,t*0.15)); }
	vec3 curlNoise(vec3 p, float t){
		const float e=0.012; vec3 dx=vec3(e,0,0),dy=vec3(0,e,0),dz=vec3(0,0,e);
		float x=potential(p+dy,t).z-potential(p-dy,t).z-(potential(p+dz,t).y-potential(p-dz,t).y);
		float y=potential(p+dz,t).x-potential(p-dz,t).x-(potential(p+dx,t).z-potential(p-dx,t).z);
		float z=potential(p+dx,t).y-potential(p-dx,t).y-(potential(p+dy,t).x-potential(p-dy,t).x);
		return vec3(x,y,z)/(2.0*e);
	}
	// center-weighted reseed point on the brain shell from a uv hash
	vec3 shellPoint(vec2 uv, float seed){
		vec3 h = hash3(vec3(uv*97.0, seed));
		float rad = pow(fract(h.x*0.5+0.5), 1.5);
		float theta = (h.y*0.5+0.5)*6.2831853;
		float phi   = acos(2.0*(h.z*0.5+0.5)-1.0);
		vec3 p = vec3(sin(phi)*cos(theta), sin(phi)*sin(theta)*0.85, cos(phi));
		p *= mix(0.25, 1.25, rad);
		p.x += sign(h.x)*0.32; // bias toward a lobe
		p -= sdfNormal(p)*brainSDF(p)*0.85; // snap onto the shell
		return p;
	}
`;

// ---- VELOCITY shader: the 4-force liquid + brain physics --------------------
const VELOCITY_SHADER = /* glsl */ `
	uniform float uTime, uDt, uForm;
	uniform vec2 uMouse, uMouseVel;
	uniform float uMouseSpeed, uAspect, uCalm;

	${GLSL_COMMON}

	void main(){
		vec2 uv = gl_FragCoord.xy / resolution.xy;
		vec4 P = texture2D( texturePosition, uv );
		vec4 V = texture2D( textureVelocity, uv );
		vec3 pos = P.xyz; vec3 vel = V.xyz;

		vec3 accel = vec3(0.0);
		float pulse = 0.5 + 0.5*sin(uTime*2.0);

		// (4) ambient SDF-tangent brain flow + gentle shell restoring (loose, so the
		// cloud keeps visible structure instead of collapsing to a dense membrane)
		vec3 flow = curlNoise(pos, uTime) * (1.4 * (1.0 - uCalm*0.85));
		float restore = (1.8 + 1.0*pulse) * uForm;
		flow += -sdfNormal(pos) * brainSDF(pos) * restore;
		accel += flow;

		// (3) HEAL: spring back toward this particle's brain-home
		vec3 home = shellPoint(uv, 3.0);
		accel += (home - pos) * (0.012 * uForm);

		// (1)+(2) liquid cursor: speed-gated gaussian repulsion + signed swirl
		vec2 q = (pos.xy - uMouse); q.x *= uAspect;
		float fall = exp(-dot(q,q) / (0.10));
		vec2 rdir = normalize(pos.xy - uMouse + 1e-5);
		accel.xy += rdir * fall * 0.9 * uMouseSpeed;
		vec2 tangent = vec2(-rdir.y, rdir.x);
		float swirlSign = sign(uMouseVel.x*rdir.y - uMouseVel.y*rdir.x);
		accel.xy += tangent * swirlSign * fall * 0.9 * uMouseSpeed;

		vel = (vel + accel*uDt) * 0.82;
		float sp = length(vel);
		float maxSpeed = 3.0;
		if (sp > maxSpeed) vel *= maxSpeed/sp;

		gl_FragColor = vec4(vel, length(flow)); // .w carries curl magnitude for color
	}
`;

// ---- POSITION shader: integrate + life/reseed -------------------------------
const POSITION_SHADER = /* glsl */ `
	uniform float uTime, uDt, uForm;
	${GLSL_COMMON}
	void main(){
		vec2 uv = gl_FragCoord.xy / resolution.xy;
		vec4 P = texture2D( texturePosition, uv );
		vec4 V = texture2D( textureVelocity, uv );
		vec3 pos = P.xyz; float life = P.w;

		pos += V.xyz * uDt;
		life -= uDt * 0.025;
		// reseed if it died OR wandered too far from the brain (keeps the cloud bounded)
		if (life < 0.0 || brainSDF(pos) > 1.8) {
			pos = shellPoint(uv, floor(uTime*0.7));
			life = 2.0 + fract(sin(dot(uv,vec2(12.9,78.2)))*43758.5)*3.0;
		}
		gl_FragColor = vec4(pos, life);
	}
`;

const RENDER_VERT = /* glsl */ `
	uniform sampler2D texturePosition;
	uniform sampler2D textureVelocity;
	uniform float uSize, uForm;
	attribute vec2 aRef;
	varying float vCurl;
	varying float vSpeed;
	varying vec3 vViewN;
	varying vec3 vViewDir;
	${GLSL_COMMON}
	void main(){
		vec4 P = texture2D(texturePosition, aRef);
		vec4 V = texture2D(textureVelocity, aRef);
		vec3 pos = P.xyz;
		vCurl = V.w;
		vSpeed = length(V.xyz);
		vec3 n = sdfNormal(pos);
		vec4 mv = modelViewMatrix * vec4(pos, 1.0);
		vViewN = normalize(normalMatrix * n);
		vViewDir = normalize(-mv.xyz);
		float dist = -mv.z;
		gl_PointSize = uSize * (0.6 + 2.0*clamp(vSpeed*0.6,0.04,1.0)) * (300.0/dist) * uForm;
		gl_Position = projectionMatrix * mv;
	}
`;

const RENDER_FRAG = /* glsl */ `
	precision highp float;
	varying float vCurl;
	varying float vSpeed;
	varying vec3 vViewN;
	varying vec3 vViewDir;
	uniform float uTime;

	// verified spectral fit (Zucconi 6) wavelength[400..700] -> RGB
	vec3 bump3y(vec3 x, vec3 yo){ vec3 y = 1.0 - x*x; return max(y - yo, 0.0); }
	vec3 spectral_zucconi6(float w){
		float x = clamp((w-400.0)/300.0, 0.0, 1.0);
		const vec3 c1=vec3(3.54585104,2.93225262,2.41593945);
		const vec3 x1=vec3(0.69549072,0.49228336,0.27699880);
		const vec3 y1=vec3(0.02312639,0.15225084,0.52607955);
		const vec3 c2=vec3(3.90307140,3.21182957,3.96587128);
		const vec3 x2=vec3(0.11748627,0.86755042,0.66077860);
		const vec3 y2=vec3(0.84897130,0.88445281,0.73949448);
		return bump3y(c1*(x-x1), y1) + bump3y(c2*(x-x2), y2);
	}

	void main(){
		vec2 uv = gl_PointCoord - 0.5;
		float d2 = dot(uv,uv);
		if (d2 > 0.25) discard;
		// sharper core + soft halo so particles read as distinct glowing points
		float soft = exp(-d2 * 14.0) + 0.35*exp(-d2 * 4.0);

		float pulse = 0.5 + 0.5*sin(uTime*2.0);
		float cosTheta = abs(dot(vViewDir, vViewN));
		float thickness = 280.0 + vCurl*120.0 + 60.0*pulse;
		float w = clamp(2.0*1.35*thickness*cosTheta, 400.0, 700.0);
		vec3 irid = spectral_zucconi6(w);

		float fres = pow(1.0 - clamp(dot(vViewN, vViewDir), 0.0, 1.0), 2.0);
		vec3 indigo = vec3(0.20, 0.10, 0.55);
		// blend toward full iridescence more readily so the violet/cyan/emerald show
		vec3 col = mix(indigo, irid, 0.35 + 0.65*fres);
		float brightness = 0.45 + 1.9*fres + vSpeed*0.6;
		// MUCH lower per-particle energy: additive of 65k must stay see-through,
		// a glittering constellation, not a solid white ball.
		float energy = soft * (0.05 + 0.12*clamp(vSpeed,0.0,1.0));
		gl_FragColor = vec4(col * brightness, energy);
	}
`;

export class NeuralFlow {
	private renderer: THREE.WebGLRenderer;
	private scene = new THREE.Scene();
	private camera: THREE.PerspectiveCamera;
	private composer: EffectComposer;
	private gpu: GPUComputationRenderer;
	private posVar: ReturnType<GPUComputationRenderer['addVariable']>;
	private velVar: ReturnType<GPUComputationRenderer['addVariable']>;
	private points!: THREE.Points;
	private material!: THREE.ShaderMaterial;
	private clock = new THREE.Clock();
	private raf = 0;
	private host: HTMLElement;
	private size: number;
	private calm: number;
	private disposed = false;
	private mouse = new THREE.Vector2(0.5, 0.5);
	private pmouse = new THREE.Vector2(0.5, 0.5);
	private mouseVel = new THREE.Vector2(0, 0);
	private mouseSpeed = 0;
	private form = 0; // entrance progress 0..1

	constructor(host: HTMLElement, opts: NeuralFlowOptions = {}) {
		this.host = host;
		this.calm = opts.reducedMotion ? 1 : 0;
		const seed = (opts.seed ?? 1234) % 1000;

		// perf: drop to 128 on low-end
		const lowEnd = window.devicePixelRatio > 2.2 || navigator.hardwareConcurrency <= 4;
		this.size = lowEnd ? 128 : 256;

		const w = host.clientWidth || 1;
		const h = host.clientHeight || 1;

		this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance' });
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
		this.renderer.setSize(w, h);
		this.renderer.setClearColor(0x04050a, 0);
		host.appendChild(this.renderer.domElement);

		this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
		this.camera.position.set(0, 0, 6.4);

		this.gpu = new GPUComputationRenderer(this.size, this.size, this.renderer);
		const pos0 = this.gpu.createTexture();
		const vel0 = this.gpu.createTexture();
		this.fillInitial(pos0, vel0, seed);
		this.posVar = this.gpu.addVariable('texturePosition', POSITION_SHADER, pos0);
		this.velVar = this.gpu.addVariable('textureVelocity', VELOCITY_SHADER, vel0);
		this.gpu.setVariableDependencies(this.posVar, [this.posVar, this.velVar]);
		this.gpu.setVariableDependencies(this.velVar, [this.posVar, this.velVar]);

		const pu = this.posVar.material.uniforms;
		pu.uTime = { value: 0 }; pu.uDt = { value: 0 }; pu.uForm = { value: 0 };
		const vu = this.velVar.material.uniforms;
		vu.uTime = { value: 0 }; vu.uDt = { value: 0 }; vu.uForm = { value: 0 };
		vu.uMouse = { value: new THREE.Vector2(0.5, 0.5) };
		vu.uMouseVel = { value: new THREE.Vector2(0, 0) };
		vu.uMouseSpeed = { value: 0 };
		vu.uAspect = { value: w / h };
		vu.uCalm = { value: this.calm };

		const err = this.gpu.init();
		if (err) console.warn('[neuralflow] gpgpu init:', err);

		this.buildPoints();

		this.composer = new EffectComposer(this.renderer);
		this.composer.addPass(new RenderPass(this.scene, this.camera));
		const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.5, 0.5, 0.82);
		this.composer.addPass(bloom);

		this.onResize = this.onResize.bind(this);
		window.addEventListener('resize', this.onResize);

		// entrance: ease form 0->1 over ~1.6s
		this.clock.start();
		this.animate();
	}

	private fillInitial(pos: THREE.DataTexture, vel: THREE.DataTexture, seed: number) {
		const p = pos.image.data as Float32Array;
		const v = vel.image.data as Float32Array;
		let s = seed * 9301 + 49297;
		const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
		for (let i = 0; i < p.length; i += 4) {
			// start as a wide scattered cloud; the entrance pulls it into the brain
			const r = 3.5 + rnd() * 4.0;
			const th = rnd() * Math.PI * 2;
			const ph = Math.acos(2 * rnd() - 1);
			p[i] = r * Math.sin(ph) * Math.cos(th);
			p[i + 1] = r * Math.sin(ph) * Math.sin(th);
			p[i + 2] = r * Math.cos(ph);
			p[i + 3] = rnd() * 2; // life
			v[i] = 0; v[i + 1] = 0; v[i + 2] = 0; v[i + 3] = 0;
		}
	}

	private buildPoints() {
		const count = this.size * this.size;
		const refs = new Float32Array(count * 2);
		for (let i = 0; i < count; i++) {
			refs[i * 2] = (i % this.size) / this.size;
			refs[i * 2 + 1] = Math.floor(i / this.size) / this.size;
		}
		const geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
		geo.setAttribute('aRef', new THREE.BufferAttribute(refs, 2));

		this.material = new THREE.ShaderMaterial({
			uniforms: {
				texturePosition: { value: null },
				textureVelocity: { value: null },
				uSize: { value: (this.host.clientHeight || 800) * 0.0014 },
				uForm: { value: 0 },
				uTime: { value: 0 }
			},
			transparent: true,
			depthWrite: false,
			depthTest: true,
			blending: THREE.AdditiveBlending,
			vertexShader: RENDER_VERT,
			fragmentShader: RENDER_FRAG
		});

		this.points = new THREE.Points(geo, this.material);
		this.points.frustumCulled = false;
		this.scene.add(this.points);
	}

	setCursor(nx: number, ny: number) {
		// nx, ny in -1..1; store as 0..1 mouse + velocity
		const mx = nx * 0.5 + 0.5;
		const my = ny * 0.5 + 0.5;
		this.mouse.set(mx, my);
	}

	private onResize() {
		const w = this.host.clientWidth || 1;
		const h = this.host.clientHeight || 1;
		this.renderer.setSize(w, h);
		this.composer.setSize(w, h);
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
		(this.velVar.material.uniforms as any).uAspect.value = w / h;
		this.material.uniforms.uSize.value = h * 0.0016;
	}

	private animate = () => {
		if (this.disposed) return;
		this.raf = requestAnimationFrame(this.animate);
		const dt = Math.min(this.clock.getDelta(), 0.033);
		const t = this.clock.elapsedTime;

		// entrance: easeOutExpo to 1 over ~1.6s
		const target = 1;
		const ease = 1 - Math.pow(2, -10 * Math.min(t / 1.6, 1));
		this.form = this.form + (target * ease - this.form) * 0.2;

		// mouse velocity (EMA) + idle decay -> heal
		const dx = (this.mouse.x - this.pmouse.x);
		const dy = (this.mouse.y - this.pmouse.y);
		const raw = Math.hypot(dx, dy);
		this.mouseSpeed = this.mouseSpeed * 0.85 + raw * 12 * 0.15;
		this.mouseSpeed *= 0.92;
		this.mouseVel.set(dx, dy);
		this.pmouse.copy(this.mouse);

		const vu = this.velVar.material.uniforms as any;
		vu.uTime.value = t; vu.uDt.value = dt; vu.uForm.value = this.form;
		vu.uMouse.value.copy(this.mouse);
		vu.uMouseVel.value.copy(this.mouseVel);
		vu.uMouseSpeed.value = Math.min(this.mouseSpeed, 2.5);
		const pu = this.posVar.material.uniforms as any;
		pu.uTime.value = t; pu.uDt.value = dt; pu.uForm.value = this.form;

		this.gpu.compute();

		this.material.uniforms.texturePosition.value = this.gpu.getCurrentRenderTarget(this.posVar).texture;
		this.material.uniforms.textureVelocity.value = this.gpu.getCurrentRenderTarget(this.velVar).texture;
		this.material.uniforms.uForm.value = this.form;
		this.material.uniforms.uTime.value = t;

		this.points.rotation.y = t * 0.05 * (1 - this.calm);
		this.points.rotation.x = Math.sin(t * 0.04) * 0.12;

		this.composer.render();
	};

	dispose() {
		this.disposed = true;
		cancelAnimationFrame(this.raf);
		window.removeEventListener('resize', this.onResize);
		this.material?.dispose();
		this.points?.geometry.dispose();
		this.composer?.dispose();
		this.renderer?.dispose();
		if (this.renderer?.domElement?.parentNode === this.host) {
			this.host.removeChild(this.renderer.domElement);
		}
	}
}
