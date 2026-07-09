import { base } from '$app/paths';
import type { MsdfGlyph } from './layout';

export type MsdfAtlasMetrics = {
	emSize?: number;
	lineHeight: number;
	ascender?: number;
	descender?: number;
	underlineY?: number;
	underlineThickness?: number;
};

export type LoadedMsdfAtlas = {
	atlas: {
		type?: string;
		distanceRange: number;
		distanceRangeMiddle?: number;
		size: number;
		width: number;
		height: number;
		yOrigin?: string;
	};
	metrics: MsdfAtlasMetrics;
	glyphs: MsdfGlyph[];
	glyphMap: Map<number, MsdfGlyph>;
	texture: GPUTexture;
	textureView: GPUTextureView;
	sampler: GPUSampler;
	dispose: () => void;
};

type RawAtlas = Omit<LoadedMsdfAtlas, 'glyphMap' | 'texture' | 'textureView' | 'sampler' | 'dispose'>;

export async function loadMsdfAtlas(device: GPUDevice): Promise<LoadedMsdfAtlas> {
	const jsonUrl = `${base}/msdf/jetbrains-mono.json`;
	const pngUrl = `${base}/msdf/jetbrains-mono.png`;

	const jsonResponse = await fetch(jsonUrl);
	if (!jsonResponse.ok) throw new Error(`MSDF atlas JSON failed: ${jsonResponse.status} ${jsonUrl}`);
	const raw = (await jsonResponse.json()) as RawAtlas;
	if (raw.atlas?.yOrigin !== 'bottom') {
		throw new Error(`MSDF atlas yOrigin must be bottom, got ${raw.atlas?.yOrigin ?? 'missing'}`);
	}

	const pngResponse = await fetch(pngUrl);
	if (!pngResponse.ok) throw new Error(`MSDF atlas PNG failed: ${pngResponse.status} ${pngUrl}`);
	const blob = await pngResponse.blob();
	const bitmap = await createImageBitmap(blob);
	const texture = device.createTexture({
		label: 'msdf-jetbrains-mono-rgba8unorm',
		size: [bitmap.width, bitmap.height, 1],
		format: 'rgba8unorm',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
	});
	device.queue.copyExternalImageToTexture(
		{ source: bitmap },
		{ texture },
		{ width: bitmap.width, height: bitmap.height }
	);
	bitmap.close?.();

	const sampler = device.createSampler({
		label: 'msdf-jetbrains-mono-linear-sampler',
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter: 'linear',
		addressModeU: 'clamp-to-edge',
		addressModeV: 'clamp-to-edge'
	});
	const textureView = texture.createView({ label: 'msdf-jetbrains-mono-view' });
	const glyphMap = new Map(raw.glyphs.map((glyph) => [glyph.unicode, glyph]));
	return {
		...raw,
		glyphMap,
		texture,
		textureView,
		sampler,
		dispose: () => texture.destroy()
	};
}
