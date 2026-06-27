// Grows real neural dendrites from text glyphs via space colonization, in the
// browser (canvas + fonts available), and returns SVG-ready path data.
// Runs once on mount, deterministic via a seeded RNG so the
// same text always grows the same neural sign.

export interface DendritePath {
	d: string;
	col: string;
	w: number;
	len: number;
	depth: number;
}
export interface DendriteSign {
	paths: DendritePath[];
	width: number;
	height: number;
}

export interface GrowSignOptions {
	fontPx?: number;
	targetW?: number;
	iters?: number;
}

// deterministic PRNG (mulberry32) so the sign is stable across loads
function rng(seed: number) {
	let a = seed >>> 0;
	return () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

interface Node {
	x: number;
	y: number;
	parent: number;
	w: number;
}

function growLine(text: string, fontPx: number, iters: number, rand: () => number) {
	const SEG = 2.2,
		ATTRACT = 14,
		KILL = 4,
		PAD = 40;
	const meas = document.createElement('canvas').getContext('2d')!;
	meas.font = `900 ${fontPx}px Inter, Arial, sans-serif`;
	const tw = Math.ceil(meas.measureText(text).width);
	const W = tw + PAD * 2;
	const H = Math.ceil(fontPx * 1.5);
	const cv = document.createElement('canvas');
	cv.width = W;
	cv.height = H;
	const ctx = cv.getContext('2d')!;
	ctx.font = `900 ${fontPx}px Inter, Arial, sans-serif`;
	ctx.fillStyle = '#fff';
	ctx.textBaseline = 'middle';
	ctx.fillText(text, PAD, H / 2);
	const data = ctx.getImageData(0, 0, W, H).data;
	const aAt = (x: number, y: number) =>
		x < 0 || y < 0 || x >= W || y >= H ? 0 : data[(y * W + x) * 4 + 3];

	const attractors: [number, number][] = [];
	const inside: [number, number][] = [];
	for (let y = 0; y < H; y += 2) {
		for (let x = 0; x < W; x += 2) {
			if (aAt(x, y) > 128) {
				inside.push([x, y]);
				attractors.push([x, y]);
				const edge =
					aAt(x + 2, y) <= 128 || aAt(x - 2, y) <= 128 || aAt(x, y + 2) <= 128 || aAt(x, y - 2) <= 128;
				if (edge && rand() < 0.22) {
					attractors.push([x + (rand() - 0.5) * 22, y + (rand() - 0.5) * 22]);
				}
			}
		}
	}
	if (!inside.length) return { nodes: [] as Node[], W, H };

	const bands: Record<number, [number, number]> = {};
	for (const [x, y] of inside) {
		const b = (x / 16) | 0;
		if (!bands[b] || y > bands[b][1]) bands[b] = [x, y];
	}
	const nodes: Node[] = [];
	for (const k in bands) nodes.push({ x: bands[k][0], y: bands[k][1], parent: -1, w: 1 });

	let live = attractors.slice();
	for (let it = 0; it < iters && live.length; it++) {
		const infCount = new Array(nodes.length).fill(0);
		const acc: [number, number][] = nodes.map(() => [0, 0]);
		for (const a of live) {
			let best = -1,
				bd = ATTRACT * ATTRACT;
			for (let i = 0; i < nodes.length; i++) {
				const dx = nodes[i].x - a[0],
					dy = nodes[i].y - a[1];
				const d = dx * dx + dy * dy;
				if (d < bd) {
					bd = d;
					best = i;
				}
			}
			if (best >= 0) {
				acc[best][0] += a[0] - nodes[best].x;
				acc[best][1] += a[1] - nodes[best].y;
				infCount[best]++;
			}
		}
		const fresh: Node[] = [];
		for (let i = 0; i < nodes.length; i++) {
			if (!infCount[i]) continue;
			const L = Math.hypot(acc[i][0], acc[i][1]) || 1;
			fresh.push({
				x: nodes[i].x + (SEG * acc[i][0]) / L + (rand() - 0.5) * 1.2,
				y: nodes[i].y + (SEG * acc[i][1]) / L + (rand() - 0.5) * 1.2,
				parent: i,
				w: 1
			});
		}
		if (!fresh.length) break;
		const before = nodes.length;
		for (const f of fresh) nodes.push(f);
		live = live.filter((a) => {
			for (let i = before; i < nodes.length; i++) {
				const dx = nodes[i].x - a[0],
					dy = nodes[i].y - a[1];
				if (dx * dx + dy * dy < KILL * KILL) return false;
			}
			return true;
		});
	}
	for (let i = nodes.length - 1; i > 0; i--) {
		if (nodes[i].parent >= 0) nodes[nodes[i].parent].w += nodes[i].w * 0.045;
	}
	return { nodes, W, H };
}

/** Grow the launch sign. Returns SVG-ready data centered to width 1000. */
export function growSign(options: GrowSignOptions = {}): DendriteSign {
	const rand = rng(0x5e57); // fixed seed -> stable sign
	const palette = ['#39ff9d', '#22d3ee', '#b388ff'];
	const specs = [
		{
			text: 'VESTIGE',
			px: options.fontPx ?? 150,
			targetW: options.targetW ?? 940,
			iters: options.iters ?? 600
		}
	];
	const VBW = 1000;
	const lines = specs.map((s) => {
		const res = growLine(s.text, s.px, s.iters, rand);
		const scale = s.targetW / Math.max(1, res.W);
		return { res, scale, scaledH: res.H * scale };
	});
	const VBH = lines.reduce((a, l) => a + l.scaledH, 0);

	const paths: DendritePath[] = [];
	let yCursor = 0;
	let colBase = 0;
	for (const line of lines) {
		const { res, scale, scaledH } = line;
		const offX = (VBW - res.W * scale) / 2;
		const offY = yCursor;
		for (let i = 0; i < res.nodes.length; i++) {
			const n = res.nodes[i];
			if (n.parent < 0) continue;
			const p = res.nodes[n.parent];
			const x1 = offX + n.x * scale,
				y1 = offY + n.y * scale;
			const x2 = offX + p.x * scale,
				y2 = offY + p.y * scale;
			const w = Math.max(0.5, Math.min(4.5, n.w * scale * 0.5));
			const len = Math.hypot(x2 - x1, y2 - y1);
			paths.push({
				d: `M${x1.toFixed(1)} ${y1.toFixed(1)}L${x2.toFixed(1)} ${y2.toFixed(1)}`,
				col: palette[(i + colBase) % 3],
				w: Math.round(w * 10) / 10,
				len: Math.round(len + 1),
				depth: Math.round((i / res.nodes.length) * 30)
			});
		}
		yCursor += scaledH;
		colBase += 1;
	}
	return { paths, width: VBW, height: Math.round(VBH) };
}
