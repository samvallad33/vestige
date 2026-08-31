/**
 * Loop export — "share your brain, not your memories."
 *
 * Turns one full deterministic Observatory loop (720 frames, 12s) into an
 * mp4 WITHOUT screen recording: the engine steps its clock frame by frame
 * offline, and each rendered frame is handed to the encoder at its exact
 * timestamp (Mediabunny CanvasSource → WebCodecs hardware encode →
 * Mp4OutputFormat → in-memory buffer). Because the loop clock is a pure
 * function of the frame index, every machine exports the byte-identical
 * clip of the same loop — a slow laptop just takes longer to finish, it
 * never drops a frame.
 *
 * Privacy contract: the export mount renders the pure canvas stage
 * (chrome 'none') — the field, the moments, the choreography — and no DOM
 * instruments. Memory text never enters the clip unless a future opt-in
 * explicitly draws it.
 */

import {
	Output,
	Mp4OutputFormat,
	BufferTarget,
	CanvasSource,
	QUALITY_HIGH
} from 'mediabunny';
import type { ObservatoryEngine } from '$lib/observatory/engine';

export interface LoopExportProgress {
	/** Frames rendered+encoded so far. */
	done: number;
	/** Total frames in the loop. */
	total: number;
	/** 'render' while stepping frames, 'finalize' while the muxer flushes. */
	stage: 'render' | 'finalize';
}

export interface LoopExportOptions {
	engine: ObservatoryEngine;
	/** Loop presentation rate. The clock is fixed-60Hz; keep the default. */
	fps?: number;
	onProgress?: (p: LoopExportProgress) => void;
	signal?: AbortSignal;
}

/** Whether this browser can encode the clip at all (WebCodecs H.264). */
export function loopExportSupported(): boolean {
	return typeof VideoEncoder !== 'undefined';
}

/**
 * Render one full loop to an mp4 buffer. The engine must be booted with its
 * graph uploaded; the caller owns mounting/unmounting the export stage.
 * Throws on abort — the engine is always handed back to the live loop.
 */
export async function exportLoopMp4({
	engine,
	fps = 60,
	onProgress,
	signal
}: LoopExportOptions): Promise<Uint8Array> {
	const totalFrames = engine.demoClock.framesPerLoop;
	const canvas = engine.canvasElement;

	const target = new BufferTarget();
	const output = new Output({
		format: new Mp4OutputFormat(),
		target
	});
	const source = new CanvasSource(canvas, {
		codec: 'avc',
		bitrate: QUALITY_HIGH
	});
	output.addVideoTrack(source, { frameRate: fps });
	await output.start();

	engine.beginExport();
	try {
		for (let i = 0; i < totalFrames; i++) {
			signal?.throwIfAborted();
			// Frame 0 renders without advancing — the clip opens exactly where
			// a live boot opens; every later frame ticks the clock once.
			await engine.renderExportFrame(i > 0);
			await source.add(i / fps, 1 / fps);
			onProgress?.({ done: i + 1, total: totalFrames, stage: 'render' });
		}
		onProgress?.({ done: totalFrames, total: totalFrames, stage: 'finalize' });
		await output.finalize();
	} finally {
		engine.endExport();
	}

	const buffer = target.buffer; // ArrayBuffer | null until finalized
	if (!buffer) throw new Error('export produced no buffer');
	return new Uint8Array(buffer);
}

/** Hand the finished clip to the user as a download. */
export function downloadClip(buffer: Uint8Array, filename: string): void {
	// Honest telemetry: the clip's real size, before the download hand-off
	// (embedded webviews sometimes swallow programmatic downloads silently).
	console.info(
		`[loop-export] ${filename}: ${(buffer.length / 1e6).toFixed(1)}MB, handing to browser download`
	);
	const blob = new Blob([buffer as unknown as BlobPart], { type: 'video/mp4' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	// Give the browser a beat to grab the blob before revoking.
	setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
