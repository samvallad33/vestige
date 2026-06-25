// Full-viewport Node Engine — the legendary landing cinematic.
//
// ~40k GPU particles (200x200 two-FBO GPGPU sim) running one looping 4-beat
// cinematic that OWNS the entire viewport:
//   STREAM   particles fly in from beyond all four screen edges toward center
//   EXPLODE  they slam together and a one-shot radial blast fills the frame
//   REFORM   they spring onto a shape (brain -> graph constellation -> lattice)
//   DISSOLVE they drift back out to the edge shell, seamlessly restarting
//
// Separate from Memory Cinema (src/lib/graph/cinema/*) which must NOT be touched.
// WebGL2 only, ships everywhere. Built from a researched spec
// (docs/launch/node-engine-spec.json).

import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

const LOOP = 18.0; // seconds per full cinematic loop

// ---- shared GLSL noise (simplex + curl) ------------------------------------
const NOISE = /* glsl */ `
	vec3 hash3(vec3 p){
		p = vec3(dot(p,vec3(127.1,311.7,74.7)),dot(p,vec3(269.5,183.3,246.1)),dot(p,vec3(113.5,271.9,124.6)));
		return -1.0+2.0*fract(sin(p)*43758.5453123);
	}
	float snoise(vec3 p){
		vec3 i=floor(p),f=fract(p),u=f*f*(3.0-2.0*f);
		return mix(mix(mix(dot(hash3(i+vec3(0,0,0)),f-vec3(0,0,0)),dot(hash3(i+vec3(1,0,0)),f-vec3(1,0,0)),u.x),
		               mix(dot(hash3(i+vec3(0,1,0)),f-vec3(0,1,0)),dot(hash3(i+vec3(1,1,0)),f-vec3(1,1,0)),u.x),u.y),
		           mix(mix(dot(hash3(i+vec3(0,0,1)),f-vec3(0,0,1)),dot(hash3(i+vec3(1,0,1)),f-vec3(1,0,1)),u.x),
		               mix(dot(hash3(i+vec3(0,1,1)),f-vec3(0,1,1)),dot(hash3(i+vec3(1,1,1)),f-vec3(1,1,1)),u.x),u.y),u.z);
	}
	vec3 snoiseVec3(vec3 p){ return vec3(snoise(p),snoise(p+vec3(17.1,9.2,3.3)),snoise(p+vec3(101.7,5.4,71.2))); }
	float h11(float p){ return fract(sin(p*127.1)*43758.5453); }
	float h1(float p){ return fract(sin(p*78.233)*12543.531); }

	#define TAU 6.28318530718

	// --- UNEXPECTED math shapes (researched, real parametric/ODE point clouds) ---

	// Aizawa strange attractor: chaotic "thinking" sculpture, toroidal shell + axial spike.
	vec3 shapeAizawa(float a, float b, float c, float S){
		vec3 p = (vec3(fract(a*43.0+0.13), fract(b*91.7+0.41), fract(c*57.3+0.77)) - 0.5) * 0.12;
		const float A=0.95, B=0.7, C=0.6, D=3.5, E=0.25, F=0.1, dt=0.01;
		int iters = 260 + int(a*140.0);
		for(int i=0;i<400;i++){
			if(i>=iters) break;
			float dx = (p.z - B)*p.x - D*p.y;
			float dy =  D*p.x + (p.z - B)*p.y;
			float dz =  C + A*p.z - p.z*p.z*p.z/3.0 - (p.x*p.x + p.y*p.y)*(1.0 + E*p.z) + F*p.z*p.x*p.x*p.x;
			p += vec3(dx,dy,dz)*dt;
		}
		// the Aizawa body is centered around ~(0,0,0.7); shift z to center it in frame
		p.z -= 0.7;
		return p.xzy * (0.46 * S); // swap y/z so the axial spike stands upright
	}

	// Hopf fibration: ~12 interlocking rings, every ring links every other once.
	vec3 shapeHopf(float a, float b, float c, float S){
		float ringId = floor(a * 12.0);
		float cosT = 2.0*fract(ringId*0.61803398875) - 1.0;
		float baseTheta = acos(clamp(cosT,-1.0,1.0));
		float basePhi = TAU * fract(ringId*0.7548776662);
		vec3 bp = vec3(sin(baseTheta)*cos(basePhi), sin(baseTheta)*sin(basePhi), cos(baseTheta));
		float t = TAU * b;
		float k = 1.0 / sqrt(2.0*(1.0 + bp.z) + 1e-4);
		vec4 P = vec4((1.0+bp.z)*cos(t), bp.x*sin(t)-bp.y*cos(t), bp.x*cos(t)+bp.y*sin(t), (1.0+bp.z)*sin(t)) * k;
		vec3 q = P.xyz / (1.0 - P.w + 1e-3);
		q += (vec3(fract(c*71.3),fract(c*131.7),fract(c*197.1))-0.5)*0.04;
		return q * (0.28 * S);
	}

	// (p,q) torus knot inflated into a volumetric tube — a glowing self-tied rope.
	vec3 shapeTorusKnot(float a, float b, float c, float S){
		const float P = 3.0, Q = 7.0, R = 1.0, rMinor = 0.45, rTube = 0.16;
		float t = a * TAU;
		float cq = cos(Q*t), sq = sin(Q*t);
		float ring = R + rMinor*cq;
		vec3 Cc = vec3(ring*cos(P*t), ring*sin(P*t), rMinor*sq);
		vec3 T = normalize(vec3(-rMinor*Q*sq*cos(P*t) - ring*P*sin(P*t),
		                        -rMinor*Q*sq*sin(P*t) + ring*P*cos(P*t), rMinor*Q*cq));
		vec3 N = normalize(cross(T, vec3(0.0,0.0,1.0)));
		vec3 Bn = cross(T, N);
		float ang = b * TAU; float rad = sqrt(c) * rTube;
		return (Cc + rad*(cos(ang)*N + sin(ang)*Bn)) * (0.55 * S);
	}

	// DNA double helix: two strands + base-pair rungs.
	vec3 shapeDNA(float a, float b, float c, float S){
		const float TWISTS = 6.0, HEIGHT = 2.0, RADIUS = 0.55;
		float yy = (a - 0.5) * HEIGHT;
		float ang = a * TWISTS * TAU;
		vec3 sA = vec3(cos(ang)*RADIUS, yy, sin(ang)*RADIUS);
		vec3 sB = vec3(cos(ang+3.14159)*RADIUS, yy, sin(ang+3.14159)*RADIUS);
		vec3 pos;
		if(b < 0.42) pos = sA; else if(b < 0.84) pos = sB; else pos = mix(sA, sB, fract(b*53.0));
		float rad = sqrt(c)*0.06; float ja = fract(c*131.7)*TAU, jb = fract(c*71.3)*3.14159;
		pos += rad * vec3(sin(jb)*cos(ja), cos(jb), sin(jb)*sin(ja));
		return pos * (0.62 * S);
	}

	// Spiral galaxy: 4 log-spiral arms + bright spherical core.
	vec3 shapeGalaxy(float a, float b, float c, float S){
		const float N_ARMS = 4.0, TWIST = 2.6;
		float arm = floor(a * N_ARMS);
		float t = sqrt(b); float r = t;
		float jit = (fract(a*97.13) - 0.5) * 0.35 * (1.0 - t);
		float ang = arm * TAU / N_ARMS + t * TWIST * TAU + jit;
		float yy = (fract(c*53.7) - 0.5) * 0.14 * exp(-2.5*r);
		vec3 pos = vec3(cos(ang)*r, yy, sin(ang)*r);
		if(b < 0.06){
			float rr = pow(fract(c*191.3), 0.5) * 0.12;
			float th = fract(a*311.7)*TAU, ph = acos(2.0*fract(c*131.1)-1.0);
			pos = vec3(rr*sin(ph)*cos(th), rr*cos(ph), rr*sin(ph)*sin(th));
		}
		// tilt the disc ~58deg around X so we see the spiral face, not edge-on
		float ca = 0.53, sa = 0.85;
		pos = vec3(pos.x, pos.y*ca - pos.z*sa, pos.y*sa + pos.z*ca);
		return pos * (1.15 * S);
	}

	// Gielis superformula supershape: a spiky alien crystal sea-urchin.
	float superR(float ang, float m, float n1, float n2, float n3){
		float t = m * ang * 0.25;
		float c1 = pow(abs(cos(t)), n2);
		float c2 = pow(abs(sin(t)), n3);
		return pow(c1 + c2 + 1e-6, -1.0/n1);
	}
	vec3 shapeSupershape(float a, float b, float c, float S){
		const float PI = 3.14159265359;
		float theta = (a*2.0 - 1.0) * PI;
		float phi = (b - 0.5) * PI;
		float m=7.0, n1=0.2, n2=1.7, n3=1.7;
		float r1 = superR(theta, m, n1, n2, n3);
		float r2 = superR(phi, m, n1, n2, n3);
		vec3 pos = vec3(r1*cos(theta)*r2*cos(phi), r1*sin(theta)*r2*cos(phi), r2*sin(phi));
		pos = normalize(pos) * pow(length(pos), 0.6);
		pos *= (1.0 + (c - 0.5)*0.04);
		return pos * (0.85 * S);
	}

	// TEXT beat: particles land where the launch-message mask alpha is high,
	// spelling VESTIGE / JULY 14TH / SIGN UP NOW. Strays relax up the alpha
	// gradient so the message reads crisp.
	vec3 shapeText(float a, float b, float c, float S){
		vec2 uv = vec2(a, b);
		for(int i=0;i<5;i++){
			float al = texture2D(uMask, uv).r;
			if(al > 0.5) break;
			float e = 1.0/256.0;
			float gx = texture2D(uMask, uv+vec2(e,0.0)).r - texture2D(uMask, uv-vec2(e,0.0)).r;
			float gy = texture2D(uMask, uv+vec2(0.0,e)).r - texture2D(uMask, uv-vec2(0.0,e)).r;
			uv += normalize(vec2(gx,gy) + 1e-5) * 0.025;
		}
		float depth = (c - 0.5) * 0.08; // thin slab so it isn't perfectly flat
		// map UV (0..1) to centered world; flip v (canvas is top-down); aspect-correct.
		// Scale so the wide message fits the frame width (not height).
		vec3 pos = vec3((uv.x - 0.5) * uMaskAspect, -(uv.y - 0.5), depth);
		return pos * (0.62 * S);
	}

	// Procedural shape targets computed from the per-particle seed. Computed in
	// shader (NOT sampled from a custom DataTexture, which renders black in GPGPU).
	// 0=BRAIN 1=GRAPH 2=LATTICE 3=AIZAWA 4=HOPF 5=KNOT 6=DNA 7=GALAXY 8=SUPERSHAPE
	vec3 shapeTarget(float seed, float shape, float S){
		float a = h11(seed*3.1), b = h1(seed*7.7), c = h11(seed*13.3+2.0);
		float ms = shape;
		if (ms > 2.5){
			if (ms < 3.5) return shapeAizawa(a,b,c,S);
			if (ms < 4.5) return shapeHopf(a,b,c,S);
			if (ms < 5.5) return shapeTorusKnot(a,b,c,S);
			if (ms < 6.5) return shapeDNA(a,b,c,S);
			if (ms < 7.5) return shapeGalaxy(a,b,c,S);
			return shapeSupershape(a,b,c,S);
		}
		if (ms < 0.5){
			// BRAIN: two lobes, SURFACE-weighted (r near 1) so it reads as a shell
			// with structure, not a fuzzy filled ball. Sulci ridges from noise.
			float r = mix(0.78, 1.0, pow(a, 0.5)); // most mass near the surface
			float theta = b*6.2831853, phi = acos(2.0*c-1.0);
			float lobe = (h11(seed*2.7) < 0.5) ? -1.0 : 1.0; // assign to a hemisphere
			float x = abs(sin(phi)*cos(theta)) * lobe;
			float y = cos(phi);
			float z = sin(phi)*sin(theta);
			// squash front-back, separate the two lobes, dimple the medial fissure
			vec3 p = vec3(x*0.78 + lobe*0.42, y*0.95, z*1.18);
			// sulci: fold the surface inward along noise ridges
			float ridges = snoise(p*3.4 + seed) * 0.12;
			p *= (1.0 + ridges);
			// flatten the bottom a touch (brain stem region)
			p.y *= (p.y < 0.0) ? 0.82 : 1.0;
			return p * r * S;
		} else if (ms < 1.5){
			// GRAPH CONSTELLATION: fibonacci-sphere node OR edge filament
			float M = 600.0;
			float k = floor(a*M);
			float yy = 1.0 - (k/(M-1.0))*2.0; float rr = sqrt(max(0.0,1.0-yy*yy));
			float t = 2.39996323*k;
			vec3 nodeA = vec3(cos(t)*rr, yy, sin(t)*rr)*S*1.2;
			if (b < 0.4) return nodeA + (snoiseVec3(vec3(seed))*0.06*S); // bright node
			float k2 = floor(c*M);
			float y2 = 1.0-(k2/(M-1.0))*2.0; float r2 = sqrt(max(0.0,1.0-y2*y2)); float t2=2.39996323*k2;
			vec3 nodeB = vec3(cos(t2)*r2, y2, sin(t2)*r2)*S*1.2;
			return mix(nodeA, nodeB, h1(seed*5.0)); // edge filament
		} else {
			// NEURAL LATTICE: hash-jittered 3D grid + struts
			float G = 6.0;
			vec3 cell = floor(vec3(a,b,c)*G);
			vec3 center = (cell+0.5)/G*2.0*S - S;
			float strut = h11(seed*17.0);
			if (strut < 0.4){
				float ax = floor(h1(seed*19.0)*3.0);
				float m = (h11(seed*23.0)-0.5)*(2.0*S/G);
				if (ax<0.5) center.x += m; else if (ax<1.5) center.y += m; else center.z += m;
			} else {
				center += (snoiseVec3(vec3(seed*2.0))*0.3*(2.0*S/G));
			}
			return center;
		}
	}
	vec3 curlNoise(vec3 p){
		const float e=0.1; vec3 dx=vec3(e,0,0),dy=vec3(0,e,0),dz=vec3(0,0,e);
		vec3 px0=snoiseVec3(p-dx),px1=snoiseVec3(p+dx);
		vec3 py0=snoiseVec3(p-dy),py1=snoiseVec3(p+dy);
		vec3 pz0=snoiseVec3(p-dz),pz1=snoiseVec3(p+dz);
		float x=py1.z-py0.z-pz1.y+pz0.y, y=pz1.x-pz0.x-px1.z+px0.z, z=px1.y-px0.y-py1.x+py0.x;
		return normalize(vec3(x,y,z)/(2.0*e)+1e-6);
	}
	float hash11(float p){ return fract(sin(p*127.1)*43758.5453); }
`;

const POSITION_SHADER = /* glsl */ `
	void main(){
		vec2 uv = gl_FragCoord.xy / resolution.xy;
		vec4 pT = texture2D(texturePosition, uv);
		vec3 vel = texture2D(textureVelocity, uv).xyz;
		// velocity is a per-frame delta (proven Codrops GPGPU pattern): add raw,
		// NOT vel*dt — otherwise particles crawl at 1/60th speed and never form.
		gl_FragColor = vec4(pT.xyz + vel, pT.w); // carry seed in .w
	}
`;

const VELOCITY_SHADER = /* glsl */ `
	uniform float uTime, uPhaseT, uDt, uBlastTime, uDebug, uShape, uScale, uMaskAspect;
	uniform sampler2D texOrigin;
	uniform sampler2D uMask;
	${NOISE}
	void main(){
		vec2 uv = gl_FragCoord.xy / resolution.xy;
		vec4 pT = texture2D(texturePosition, uv);
		vec3 pos = pT.xyz; float seed = pT.w;
		vec3 vel = texture2D(textureVelocity, uv).xyz;
		vec4 oT = texture2D(texOrigin, uv);
		vec3 originPos = oT.xyz; float perimT = oT.w;
		vec3 shapePos = shapeTarget(seed, uShape, uScale);

		float wStream   = 1.0 - smoothstep(0.18, 0.21, uPhaseT);
		float pulse     = smoothstep(0.200,0.215,uPhaseT) * (1.0 - smoothstep(0.215,0.245,uPhaseT));
		float wReform   = smoothstep(0.235,0.55,uPhaseT) * (1.0 - smoothstep(0.80,0.86,uPhaseT));
		float wDissolve = smoothstep(0.82,1.00,uPhaseT);

		// Velocity is a PER-FRAME delta (Codrops GPGPU pattern): forces are small,
		// position adds raw velocity, damping keeps it stable.

		// STREAM: staggered release toward center
		float delay = 0.10*hash11(seed*91.7) + 0.08*perimT;
		float act = smoothstep(0.0, 0.06, uPhaseT - delay);
		vec3 toCenter = -pos; float dC = length(toCenter);
		vec3 dirC = toCenter/max(dC,1e-4);
		float tConv = clamp(uPhaseT/0.20, 0.0, 1.0);
		float pull = pow(tConv, 3.0);
		vel += dirC * (0.004 + 0.02*pull) * act * wStream * smoothstep(0.0,6.0,dC);

		// EXPLODE: one-shot gaussian radial blast
		float age = uTime - uBlastTime;
		float gate = exp(-age*age*60.0);
		float r = max(length(pos),1e-4);
		vec3 outDir = pos/r;
		float falloff = 1.0/(1.0 + r*r*4.0);
		vel += outDir * 1.4 * falloff * gate;
		vel += snoiseVec3(pos*31.0) * 0.02 * gate;

		// REFORM/HOLD: spring onto shape (per-frame delta -> small k, critically damped).
		float wHold = smoothstep(0.235, 0.42, uPhaseT) * (1.0 - smoothstep(0.80, 0.86, uPhaseT));
		float localProg = smoothstep(delay, delay+0.30, (uPhaseT-0.235)/0.30);
		float kAttract = mix(0.0, 0.10, wHold) * localProg;
		vec3 toShape = shapePos - pos;
		vel += toShape * kAttract;

		// DISSOLVE: drift back to edge shell (== next-loop spawn -> seamless)
		vec3 toEdge = originPos - pos;
		vel += toEdge * (0.04 * wDissolve);
		vel += normalize(pos+1e-4) * 0.01 * wDissolve;

		// curl turbulence: strong on stream/blast/dissolve, NEAR ZERO once formed.
		float turb = (0.012*wStream + 0.006*gate + 0.014*wDissolve) * (1.0 - wHold*0.96) + 0.0002;
		vel += curlNoise(pos*0.30 + uTime*0.10) * turb;

		// damping: heavy on hold so the shape SETTLES crisp; lighter during motion.
		float damping = mix(0.90, 0.55, pulse) * mix(1.0, 0.82, wHold);
		vel *= damping;
		float v = length(vel); if (v > 0.6) vel *= 0.6/v;

		// DIAGNOSTIC: uDebug=1 ignores all phase logic and just springs to target.
		// If the brain forms with this, the pipeline works and phase logic is the bug.
		if (uDebug > 0.5) {
			// spring to procedural BRAIN with HARDCODED scale 5.0 (rules out uScale uniform).
			vec3 tgt = shapeTarget(seed, 0.0, 5.0);
			vec3 d = tgt - pos;
			vel = d * 0.12;
			vel *= 0.86;
		}

		gl_FragColor = vec4(vel, 1.0);
	}
`;

const RENDER_VERT = /* glsl */ `
	uniform sampler2D texturePosition;
	uniform sampler2D textureVelocity;
	uniform float uSize, uDpr, uPhaseT, uTime;
	attribute vec2 reference;
	varying float vSpeed; varying float vSeed;
	void main(){
		vec4 pT = texture2D(texturePosition, reference);
		vec3 pos = pT.xyz; vSeed = pT.w;
		vec3 vel = texture2D(textureVelocity, reference).xyz;
		vSpeed = length(vel);
		vec4 mv = modelViewMatrix * vec4(pos,1.0);
		float breathe = 1.0 + 0.05*sin(uTime*1.4)*step(0.55,uPhaseT)*step(uPhaseT,0.82);
		float stretch = 1.0 + clamp(vSpeed*0.6, 0.0, 2.5);
		gl_PointSize = uSize * uDpr * breathe * stretch / max(-mv.z, 0.1);
		gl_Position = projectionMatrix * mv;
	}
`;

const RENDER_FRAG = /* glsl */ `
	precision highp float;
	uniform vec3 uViolet, uCyan, uEmerald;
	uniform vec2 uTextCenter, uResolution;
	varying float vSpeed; varying float vSeed;
	void main(){
		vec2 q = gl_PointCoord - 0.5;
		float d2 = dot(q,q);
		if (d2 > 0.25) discard;
		// bright core + soft glowing halo so each particle reads as a luminous orb
		float a = exp(-d2*16.0) + 0.5*exp(-d2*4.0);

		// RICH 5-stop spectrum across the seed: magenta -> violet -> blue -> cyan -> emerald.
		// Each particle holds its hue (spatially separated) instead of averaging to one color.
		vec3 magenta = vec3(0.95, 0.25, 0.85);
		vec3 violet  = vec3(0.55, 0.35, 1.00);
		vec3 blue    = vec3(0.25, 0.45, 1.00);
		vec3 cyan    = vec3(0.15, 0.85, 0.95);
		vec3 emerald = vec3(0.25, 0.95, 0.55);
		float s = vSeed;
		vec3 col;
		if (s < 0.25)      col = mix(magenta, violet, s/0.25);
		else if (s < 0.50) col = mix(violet,  blue,   (s-0.25)/0.25);
		else if (s < 0.75) col = mix(blue,    cyan,   (s-0.50)/0.25);
		else               col = mix(cyan,    emerald,(s-0.75)/0.25);

		// fast particles flare WARM gold (energy), not blue-white — keeps color alive.
		vec3 gold = vec3(1.0, 0.75, 0.35);
		col = mix(col, gold, clamp(vSpeed*0.5, 0.0, 0.55));

		col *= 1.7; // luminous but not blown-out
		vec2 sUv = gl_FragCoord.xy / uResolution;
		float textMask = smoothstep(0.0, 0.20, length(sUv - uTextCenter));
		float alpha = a * (0.7 + 0.3*textMask);
		gl_FragColor = vec4(col * alpha, alpha);
	}
`;

export interface NodeEngineOptions {
	seed?: number;
	reducedMotion?: boolean;
	/** Debug: freeze the loop at a fixed phase in [0,1) to inspect a single beat. */
	forcePhase?: number;
	/** Debug: force a specific shape index 0..8 to inspect one shape. */
	forceShape?: number;
	/** Called when the active beat is/ isn't the particle TEXT message. */
	onTextBeat?: (isText: boolean) => void;
}

export class NodeEngine {
	private renderer: THREE.WebGLRenderer;
	private scene = new THREE.Scene();
	private camera: THREE.PerspectiveCamera;
	private composer: EffectComposer;
	private gpu: GPUComputationRenderer;
	private posVar!: ReturnType<GPUComputationRenderer['addVariable']>;
	private velVar!: ReturnType<GPUComputationRenderer['addVariable']>;
	private points!: THREE.Points;
	private material!: THREE.ShaderMaterial;
	private targets: THREE.DataTexture[] = [];
	private clock = new THREE.Clock();
	private raf = 0;
	private host: HTMLElement;
	private TEX: number;
	private N: number;
	private reduced: boolean;
	private disposed = false;
	private uBlastTime = -999;
	private prevPhase = 0;
	private shapeIndex = 0;
	private frustum = new THREE.Vector2(20, 12);
	private forcePhase: number | undefined;
	private forceShape: number | undefined;
	private rotAmt = 1;
	private onTextBeat: ((isText: boolean) => void) | undefined;
	private lastTextState = false;

	constructor(host: HTMLElement, opts: NodeEngineOptions = {}) {
		this.host = host;
		this.reduced = !!opts.reducedMotion;
		this.forcePhase = opts.forcePhase;
		this.forceShape = opts.forceShape;
		this.onTextBeat = opts.onTextBeat;
		const lowEnd = window.devicePixelRatio > 2.2 || (navigator.hardwareConcurrency || 8) <= 4;
		this.TEX = lowEnd ? 140 : 200;
		this.N = this.TEX * this.TEX;
		const seed = (opts.seed ?? 1234) % 100000;

		const w = host.clientWidth || 1, h = host.clientHeight || 1;
		this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance' });
		this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		this.renderer.setClearColor(0x05050f, 0);
		this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
		this.renderer.setSize(w, h);
		host.appendChild(this.renderer.domElement);

		this.camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 100);
		this.camera.position.set(0, 0, 14);
		this.computeFrustum();

		const originTex = this.buildOriginTexture(seed);
		this.gpu = new GPUComputationRenderer(this.TEX, this.TEX, this.renderer);
		const pos0 = this.gpu.createTexture();
		const vel0 = this.gpu.createTexture();
		// start everyone on the edge shell, with seed in .w
		(pos0.image.data as Float32Array).set(originTex.image.data as Float32Array);
		for (let i = 0; i < this.N; i++) (pos0.image.data as Float32Array)[i * 4 + 3] = (i / this.N);
		this.posVar = this.gpu.addVariable('texturePosition', POSITION_SHADER, pos0);
		this.velVar = this.gpu.addVariable('textureVelocity', VELOCITY_SHADER, vel0);
		this.gpu.setVariableDependencies(this.posVar, [this.posVar, this.velVar]);
		this.gpu.setVariableDependencies(this.velVar, [this.posVar, this.velVar]);

		const pu = this.posVar.material.uniforms; pu.uDt = { value: 0 };
		const vu = this.velVar.material.uniforms;
		vu.uTime = { value: 0 }; vu.uPhaseT = { value: 0 }; vu.uDt = { value: 0 }; vu.uBlastTime = { value: -999 };
		vu.texOrigin = { value: originTex };
		vu.uShape = { value: this.forceShape ?? 0 };
		vu.uScale = { value: this.frustum.y * 0.42 };
		vu.uDebug = { value: this.forcePhase !== undefined && this.forcePhase < 0 ? 1 : 0 };
		const mask = this.buildTextMask(['VESTIGE', 'JULY 14TH', 'SIGN UP NOW']);
		vu.uMask = { value: mask.texture };
		vu.uMaskAspect = { value: mask.aspect };

		const err = this.gpu.init();
		if (err) console.warn('[nodeEngine] gpgpu init:', err);

		this.buildPoints(seed);

		this.composer = new EffectComposer(this.renderer);
		this.composer.addPass(new RenderPass(this.scene, this.camera));
		const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.6, 0.5, 0.6);
		this.composer.addPass(bloom);

		this.onResize = this.onResize.bind(this);
		window.addEventListener('resize', this.onResize);
		this.animate();
	}

	private computeFrustum() {
		const vFOV = (this.camera.fov * Math.PI) / 180;
		const h = 2 * Math.tan(vFOV / 2) * Math.abs(this.camera.position.z);
		const w = h * this.camera.aspect;
		this.frustum.set(w, h);
	}

	// Build an alpha mask of the launch message (3 rows) on a canvas. The shader
	// samples it: particles land where text alpha is high, spelling the message.
	private buildTextMask(lines: string[]): { texture: THREE.Texture; aspect: number } {
		const W = 1024, H = 384;
		const cv = document.createElement('canvas');
		cv.width = W; cv.height = H;
		const ctx = cv.getContext('2d')!;
		ctx.fillStyle = '#000';
		ctx.fillRect(0, 0, W, H);
		ctx.fillStyle = '#fff';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		const rowH = H / lines.length;
		for (let i = 0; i < lines.length; i++) {
			// size each line to fill the width nicely
			const line = lines[i];
			let size = Math.floor(rowH * 0.78);
			ctx.font = `900 ${size}px Inter, Arial, sans-serif`;
			// shrink to fit if too wide
			while (ctx.measureText(line).width > W * 0.92 && size > 10) {
				size -= 2; ctx.font = `900 ${size}px Inter, Arial, sans-serif`;
			}
			ctx.fillText(line, W / 2, rowH * (i + 0.5));
		}
		const tex = new THREE.CanvasTexture(cv);
		tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
		tex.wrapS = THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		this.renderer.initTexture(tex);
		return { texture: tex, aspect: W / H };
	}

	// edge-spawn shell: each particle off-screen on one of 4 edges, perimeterT in .w
	private buildOriginTexture(seed: number): THREE.DataTexture {
		const data = new Float32Array(this.N * 4);
		let s = seed + 1;
		const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
		const hw = this.frustum.x * 0.5 * 1.18, hh = this.frustum.y * 0.5 * 1.18;
		const camZ = Math.abs(this.camera.position.z);
		for (let i = 0; i < this.N; i++) {
			const edge = Math.floor(rnd() * 4);
			const t = rnd();
			let x = 0, y = 0;
			if (edge === 0) { x = -hw; y = (t * 2 - 1) * hh; }
			else if (edge === 1) { x = hw; y = (t * 2 - 1) * hh; }
			else if (edge === 2) { y = hh; x = (t * 2 - 1) * hw; }
			else { y = -hh; x = (t * 2 - 1) * hw; }
			const z = (rnd() - 0.4) * camZ * 0.6;
			const perimT = (edge + t) / 4;
			data[i * 4] = x; data[i * 4 + 1] = y; data[i * 4 + 2] = z; data[i * 4 + 3] = perimT;
		}
		const tex = new THREE.DataTexture(data, this.TEX, this.TEX, THREE.RGBAFormat, THREE.FloatType);
		tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter;
		tex.wrapS = THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		this.renderer.initTexture(tex); // force GPU upload (three.js #15882)
		return tex;
	}

	private dataTex(fill: (i: number, out: Float32Array, o: number) => void): THREE.DataTexture {
		const data = new Float32Array(this.N * 4);
		for (let i = 0; i < this.N; i++) fill(i, data, i * 4);
		const tex = new THREE.DataTexture(data, this.TEX, this.TEX, THREE.RGBAFormat, THREE.FloatType);
		tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter;
		tex.wrapS = THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		// FORCE GPU upload now. A procedurally-built DataTexture passed as a custom
		// sampler2D into a GPUComputationRenderer pass reads BLACK until uploaded
		// (three.js #15882). initTexture uploads it before the first compute().
		this.renderer.initTexture(tex);
		return tex;
	}

	// three target shapes the cloud reforms into
	private buildTargets(seed: number) {
		let s = seed * 1.7 + 5;
		const rnd = () => { s = Math.sin(s) * 43758.5453; return s - Math.floor(s); };
		const scale = this.frustum.y * 0.42;

		// A: BRAIN — two squashed hemispheres + sulci wrinkle, volumetric fill
		const brain = this.dataTex((i, out, o) => {
			const r = Math.cbrt(rnd()); // volumetric (denser core)
			const theta = rnd() * Math.PI * 2;
			const phi = Math.acos(2 * rnd() - 1);
			let x = Math.sin(phi) * Math.cos(theta);
			let y = Math.cos(phi) * 0.92;
			let z = Math.sin(phi) * Math.sin(theta) * 1.18;
			x += Math.sign(x) * 0.34; // split into two lobes
			const wrinkle = 1 + 0.07 * Math.sin(8 * x) * Math.sin(7 * y);
			out[o] = x * r * scale * wrinkle;
			out[o + 1] = y * r * scale * wrinkle;
			out[o + 2] = z * r * scale * wrinkle;
			out[o + 3] = rnd() < 0.45 ? 1 : 0;
		});

		// B: GRAPH CONSTELLATION — Fibonacci-sphere nodes + edge filaments
		const M = 600, GA = Math.PI * (3 - Math.sqrt(5));
		const nodes: [number, number, number][] = [];
		for (let k = 0; k < M; k++) {
			const y = 1 - (k / (M - 1)) * 2, rr = Math.sqrt(1 - y * y), t = GA * k;
			nodes.push([Math.cos(t) * rr * scale * 1.2, y * scale * 1.2, Math.sin(t) * rr * scale * 1.2]);
		}
		const constellation = this.dataTex((i, out, o) => {
			if (rnd() < 0.35) {
				const n = nodes[Math.floor(rnd() * M)];
				out[o] = n[0] + (rnd() - 0.5) * 0.4; out[o + 1] = n[1] + (rnd() - 0.5) * 0.4; out[o + 2] = n[2] + (rnd() - 0.5) * 0.4;
				out[o + 3] = 1;
			} else {
				const a = nodes[Math.floor(rnd() * M)], b = nodes[Math.floor(rnd() * M)];
				const m = rnd();
				out[o] = a[0] + (b[0] - a[0]) * m; out[o + 1] = a[1] + (b[1] - a[1]) * m; out[o + 2] = a[2] + (b[2] - a[2]) * m;
				out[o + 3] = 0;
			}
		});

		// C: NEURAL LATTICE — hash-jittered 3D grid
		const G = 7, cell = (scale * 2) / G;
		const lattice = this.dataTex((i, out, o) => {
			const cx = Math.floor(rnd() * G), cy = Math.floor(rnd() * G), cz = Math.floor(rnd() * G);
			const onStrut = rnd() < 0.4;
			let x = (cx + 0.5) * cell - scale, y = (cy + 0.5) * cell - scale, z = (cz + 0.5) * cell - scale;
			if (onStrut) {
				const axis = Math.floor(rnd() * 3), m = rnd();
				if (axis === 0) x += (m - 0.5) * cell;
				else if (axis === 1) y += (m - 0.5) * cell;
				else z += (m - 0.5) * cell;
				out[o + 3] = 0;
			} else {
				x += (rnd() - 0.5) * 0.3 * cell; y += (rnd() - 0.5) * 0.3 * cell; z += (rnd() - 0.5) * 0.3 * cell;
				out[o + 3] = 1;
			}
			out[o] = x; out[o + 1] = y; out[o + 2] = z;
		});

		this.targets = [brain, constellation, lattice];
	}

	private buildPoints(seed: number) {
		const refs = new Float32Array(this.N * 2);
		// texel CENTERS (+0.5) so render samples exactly the texel the sim writes.
		// Using texel corners (i/TEX) reads a neighbor -> particles never move.
		for (let i = 0; i < this.N; i++) {
			refs[i * 2] = ((i % this.TEX) + 0.5) / this.TEX;
			refs[i * 2 + 1] = (Math.floor(i / this.TEX) + 0.5) / this.TEX;
		}
		const geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.N * 3), 3));
		geo.setAttribute('reference', new THREE.BufferAttribute(refs, 2));
		const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
		this.material = new THREE.ShaderMaterial({
			uniforms: {
				texturePosition: { value: null }, textureVelocity: { value: null },
				uSize: { value: 38 }, uDpr: { value: Math.min(window.devicePixelRatio, 2) },
				uPhaseT: { value: 0 }, uTime: { value: 0 },
				uViolet: { value: new THREE.Color(0x6366f1) }, uCyan: { value: new THREE.Color(0x22d3ee) },
				uEmerald: { value: new THREE.Color(0x34d399) },
				uTextCenter: { value: new THREE.Vector2(0.5, 0.46) }, uResolution: { value: new THREE.Vector2(w, h) }
			},
			transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
			vertexShader: RENDER_VERT, fragmentShader: RENDER_FRAG
		});
		this.points = new THREE.Points(geo, this.material);
		this.points.frustumCulled = false;
		// Push the whole cloud DOWN so shapes cut off near the top, leaving a clean
		// band at the very top of the viewport for the fixed launch sign.
		this.points.position.y = -this.frustum.y * 0.16;
		this.scene.add(this.points);
	}

	private onResize() {
		const w = this.host.clientWidth || 1, h = this.host.clientHeight || 1;
		this.renderer.setSize(w, h); this.composer.setSize(w, h);
		this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
		this.computeFrustum();
		(this.material.uniforms.uResolution.value as THREE.Vector2).set(w, h);
	}

	private animate = () => {
		if (this.disposed) return;
		this.raf = requestAnimationFrame(this.animate);
		const dt = Math.min(this.clock.getDelta(), 1 / 30);
		const t = this.clock.elapsedTime;

		let phase = (t % LOOP) / LOOP;
		if (this.reduced) phase = 0.66; // freeze in HOLD on the brain
		if (this.forcePhase !== undefined) phase = this.forcePhase; // debug freeze

		// loop wrap -> advance shape + fresh blast time (unless a shape is forced)
		if (this.forceShape === undefined && !this.reduced && this.prevPhase > 0.9 && phase < 0.1) {
			this.shapeIndex = (this.shapeIndex + 1) % 9;
			(this.velVar.material.uniforms as any).uShape.value = this.shapeIndex;
		}
		// fire the blast once when crossing into the SLAM window
		if (!this.reduced && this.prevPhase < 0.2 && phase >= 0.2) {
			this.uBlastTime = t;
		}
		this.prevPhase = phase;

		const vu = this.velVar.material.uniforms as any;
		vu.uTime.value = t; vu.uPhaseT.value = phase; vu.uDt.value = dt; vu.uBlastTime.value = this.uBlastTime;
		(this.posVar.material.uniforms as any).uDt.value = dt;
		this.gpu.compute();

		this.material.uniforms.texturePosition.value = this.gpu.getCurrentRenderTarget(this.posVar).texture;
		this.material.uniforms.textureVelocity.value = this.gpu.getCurrentRenderTarget(this.velVar).texture;
		this.material.uniforms.uPhaseT.value = phase;
		this.material.uniforms.uTime.value = t;

		// 3D rotation for MATH shapes (mesmerizing); but TEXT beats (even index)
		// must stay flat-on and upright so the message is readable. Ease the
		// rotation to 0 during text beats.
		// 3D rotation so every shape reads volumetrically (a static galaxy/knot
		// looks flat; a slowly turning one is mesmerizing).
		this.points.rotation.y = t * 0.18 * (this.reduced ? 0 : 1);
		this.points.rotation.x = Math.sin(t * 0.13) * 0.22 * (this.reduced ? 0 : 1);

		this.composer.render();
	};

	dispose() {
		this.disposed = true;
		cancelAnimationFrame(this.raf);
		window.removeEventListener('resize', this.onResize);
		this.material?.dispose();
		this.points?.geometry.dispose();
		this.targets.forEach((t) => t.dispose());
		this.composer?.dispose();
		this.renderer?.dispose();
		if (this.renderer?.domElement?.parentNode === this.host) this.host.removeChild(this.renderer.domElement);
	}
}
