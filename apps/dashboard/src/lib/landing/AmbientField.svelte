<script lang="ts">
	// Ambient outer field: god-ray glow radiating from the center (so the outer
	// frame looks CAUSED BY the storm, not decorated) + a parallax starfield that
	// fills the corners with quiet depth. Pure 2D canvas + CSS, engine-agnostic,
	// sits BEHIND the WebGPU/WebGL storm. ~one cheap canvas, holds 60fps.
	import { onMount, onDestroy } from 'svelte';

	interface Props {
		seed?: number;
		reducedMotion?: boolean;
	}
	let { seed = 1234, reducedMotion = false }: Props = $props();

	let canvas = $state<HTMLCanvasElement | undefined>(undefined);
	let raf = 0;
	let disposed = false;

	type Star = { x: number; y: number; z: number; r: number; tw: number; hue: number };
	let stars: Star[] = [];
	let w = 0, h = 0, dpr = 1;
	let pointer = { x: 0.5, y: 0.5 };

	function mulberry32(a: number) {
		return () => {
			a |= 0; a = (a + 0x6d2b79f5) | 0;
			let t = Math.imul(a ^ (a >>> 15), 1 | a);
			t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
			return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
		};
	}

	function build() {
		if (!canvas) return;
		dpr = Math.min(window.devicePixelRatio || 1, 2);
		w = canvas.clientWidth; h = canvas.clientHeight;
		canvas.width = Math.floor(w * dpr);
		canvas.height = Math.floor(h * dpr);
		const rnd = mulberry32(seed || 1);
		// density rises toward the EDGES (rejection sample away from center)
		const count = Math.floor((w * h) / 9000);
		stars = [];
		let guard = 0;
		while (stars.length < count && guard < count * 6) {
			guard++;
			const x = rnd(), y = rnd();
			const dx = x - 0.5, dy = y - 0.5;
			const edge = Math.min(1, Math.hypot(dx, dy) * 2); // 0 center -> 1 corner
			if (rnd() > edge * 0.85 + 0.06) continue; // keep more near edges
			const z = 0.2 + rnd() * 0.8; // depth -> parallax + size
			const hue = [262, 190, 152][Math.floor(rnd() * 3)]; // violet/cyan/emerald
			stars.push({ x, y, z, r: (0.4 + rnd() * 1.4) * z, tw: rnd() * Math.PI * 2, hue });
		}
	}

	function draw(t: number) {
		if (disposed || !canvas) return;
		raf = requestAnimationFrame(draw);
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, w, h);

		const cx = w / 2, cy = h * 0.46;
		const time = reducedMotion ? 0 : t * 0.001;

		// --- god-ray glow: violet/cyan/emerald light radiating from the storm ---
		const breathe = 0.85 + 0.15 * Math.sin(time * 0.6);
		const maxR = Math.hypot(w, h) * 0.62 * breathe;
		const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR);
		g.addColorStop(0.0, 'rgba(120, 90, 240, 0.22)');
		g.addColorStop(0.22, 'rgba(60, 120, 230, 0.14)');
		g.addColorStop(0.5, 'rgba(40, 180, 160, 0.07)');
		g.addColorStop(1.0, 'rgba(5, 6, 12, 0)');
		ctx.globalCompositeOperation = 'lighter';
		ctx.fillStyle = g;
		ctx.fillRect(0, 0, w, h);

		// faint rotating light shafts for the "rays" read
		if (!reducedMotion) {
			ctx.save();
			ctx.translate(cx, cy);
			ctx.rotate(time * 0.04);
			const shafts = 7;
			for (let i = 0; i < shafts; i++) {
				ctx.rotate((Math.PI * 2) / shafts);
				const sg = ctx.createLinearGradient(0, 0, maxR, 0);
				sg.addColorStop(0, 'rgba(90,140,235,0.05)');
				sg.addColorStop(1, 'rgba(5,6,12,0)');
				ctx.fillStyle = sg;
				ctx.beginPath();
				ctx.moveTo(0, 0);
				ctx.lineTo(maxR, -maxR * 0.06);
				ctx.lineTo(maxR, maxR * 0.06);
				ctx.closePath();
				ctx.fill();
			}
			ctx.restore();
		}

		// --- parallax starfield filling the corners ---
		const px = (pointer.x - 0.5), py = (pointer.y - 0.5);
		for (const s of stars) {
			const driftX = reducedMotion ? 0 : Math.sin(time * 0.05 * s.z + s.tw) * 6 * s.z;
			const sx = s.x * w + driftX - px * 40 * s.z;
			const sy = s.y * h - py * 40 * s.z;
			const tw = reducedMotion ? 0.7 : 0.45 + 0.55 * (0.5 + 0.5 * Math.sin(time * 1.5 + s.tw));
			ctx.beginPath();
			ctx.fillStyle = `hsla(${s.hue}, 80%, 75%, ${0.55 * tw})`;
			ctx.arc(sx, sy, s.r, 0, Math.PI * 2);
			ctx.fill();
		}
		ctx.globalCompositeOperation = 'source-over';
	}

	function onPointer(e: PointerEvent) {
		pointer.x = e.clientX / window.innerWidth;
		pointer.y = e.clientY / window.innerHeight;
	}
	function onResize() { build(); }

	onMount(() => {
		build();
		raf = requestAnimationFrame(draw);
		window.addEventListener('resize', onResize);
		if (!reducedMotion) window.addEventListener('pointermove', onPointer, { passive: true });
	});
	onDestroy(() => {
		disposed = true;
		cancelAnimationFrame(raf);
		window.removeEventListener('resize', onResize);
		window.removeEventListener('pointermove', onPointer);
	});
</script>

<canvas class="ambient" bind:this={canvas} aria-hidden="true"></canvas>

<style>
	.ambient {
		position: fixed;
		inset: 0;
		z-index: 0;
		width: 100vw;
		height: 100vh;
		height: 100svh;
		display: block;
		pointer-events: none;
	}
</style>
