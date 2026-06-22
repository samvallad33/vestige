<script lang="ts">
	// Memory Cinema — the orchestration layer (Phase 4).
	//
	// Ties the three tiers together into one fullscreen experience:
	//   Tier 3 (always): BFS pathfinder + camera director drive the flythrough.
	//   Tier 2 (default): local structured captions from real node/edge data.
	//   Tier 1 (opt-in / when available): richer narration from a backend LLM
	//     endpoint, or the opt-in on-device model (lazy-loaded only on click).
	//   WebGPU storm renders if supported; otherwise camera + captions still play.
	//
	// Launches in an isolated overlay with its own canvas — the WebGL graph
	// underneath is never touched.

	import { onDestroy } from 'svelte';
	import * as THREE from 'three';
	import type { GraphNode, GraphEdge } from '$types';
	import { planCinemaPath, type CinemaPath, type CinemaBeat } from '$lib/graph/cinema/pathfinder';
	import { CinemaDirector } from '$lib/graph/cinema/director';
	import { resolveNarration, type CinemaNarration } from '$lib/graph/cinema/narrator';
	import type { SemanticRole } from '$lib/graph/cinema/storm';
	import type { CinemaSandbox } from '$lib/graph/cinema/sandbox';

	interface Props {
		nodes: GraphNode[];
		edges: GraphEdge[];
		centerId: string;
		/** Optional Tier-1 backend narration fetcher (passed when backend supports it). */
		fetchBackendNarration?: () => Promise<import('$lib/graph/cinema/narrator').BeatNarration[] | null>;
	}
	let { nodes, edges, centerId, fetchBackendNarration }: Props = $props();

	let open = $state(false);
	let stage = $state<'idle' | 'planning' | 'playing' | 'done'>('idle');
	let caption = $state('');
	let chip = $state('');
	let progress = $state(0);
	let beatIndex = $state(0);
	let totalBeats = $state(0);
	let narrationSource = $state<CinemaNarration['source'] | ''>('');
	let webgpuActive = $state(false);
	let voiceOn = $state(false);
	let localAiOn = $state(false);
	let statusLine = $state('');

	let canvasHost = $state<HTMLDivElement | undefined>(undefined);
	let sandbox: CinemaSandbox | null = null;
	let director: CinemaDirector | null = null;
	let path: CinemaPath | null = null;
	let narration: CinemaNarration | null = null;
	let rafId = 0;
	let lastFrame = 0;
	let typeTimer: ReturnType<typeof setInterval> | null = null;

	const reducedMotion =
		typeof window !== 'undefined' &&
		window.matchMedia('(prefers-reduced-motion: reduce)').matches;

	// Deterministic layout: spread path nodes on a gentle spiral so the camera
	// has distinct world positions to fly between (independent of the WebGL
	// graph's internal coordinates — keeps the sandbox isolated).
	function layoutPositions(p: CinemaPath): Map<string, THREE.Vector3> {
		const pos = new Map<string, THREE.Vector3>();
		const n = p.beats.length;
		for (let i = 0; i < n; i++) {
			const angle = (i / Math.max(1, n)) * Math.PI * 2 * 1.4;
			const radius = 22 + i * 6;
			pos.set(
				p.beats[i].nodeId,
				new THREE.Vector3(
					Math.cos(angle) * radius,
					(i % 2 === 0 ? 1 : -1) * (4 + i * 2),
					Math.sin(angle) * radius
				)
			);
		}
		return pos;
	}

	function roleFor(beat: CinemaBeat): SemanticRole {
		if (beat.kind === 'origin') return 'anchor';
		if (beat.kind === 'contradiction') return 'contradiction';
		return 'connection';
	}

	function speak(text: string) {
		if (!voiceOn || typeof speechSynthesis === 'undefined') return;
		try {
			speechSynthesis.cancel();
			const u = new SpeechSynthesisUtterance(text);
			u.rate = 0.98;
			u.pitch = 1.0;
			speechSynthesis.speak(u);
		} catch {
			/* voice unavailable — captions carry it */
		}
	}

	// Typewriter caption stream so text "arrives" with the camera.
	function streamCaption(text: string) {
		if (typeTimer) clearInterval(typeTimer);
		caption = '';
		if (reducedMotion) {
			caption = text;
			return;
		}
		let i = 0;
		typeTimer = setInterval(() => {
			caption = text.slice(0, ++i);
			if (i >= text.length && typeTimer) {
				clearInterval(typeTimer);
				typeTimer = null;
			}
		}, 18);
	}

	function onBeat(beat: CinemaBeat, index: number) {
		beatIndex = index + 1;
		const text = narration?.beats[index]?.text ?? beat.node.label ?? '';
		chip = narration?.beats[index]?.chip ?? '';
		streamCaption(text);
		speak(text);
		if (sandbox && webgpuActive) {
			const wp = currentPositions?.get(beat.nodeId);
			if (wp) sandbox.transitionTo(roleFor(beat), wp);
		}
	}

	let currentPositions: Map<string, THREE.Vector3> | null = null;

	async function launch() {
		open = true;
		stage = 'planning';
		statusLine = 'Planning a path through your memory…';
		caption = '';
		chip = '';
		progress = 0;
		beatIndex = 0;

		// Tier 3: plan the path (always works).
		path = planCinemaPath(nodes, edges, centerId, 7);
		totalBeats = path.beats.length;
		if (totalBeats === 0) {
			statusLine = 'Not enough memory to compose a tour yet.';
			stage = 'done';
			return;
		}
		currentPositions = layoutPositions(path);

		// Tiers 1/2: resolve narration (backend LLM → local captions).
		narration = await resolveNarration(path, localAiOn ? localAiFetcher() : fetchBackendNarration);
		narrationSource = narration.source;

		// Try the WebGPU storm; fall back silently to camera + captions.
		webgpuActive = false;
		if (canvasHost) {
			try {
				const { CinemaSandbox, isWebGPUSupported } = await import('$lib/graph/cinema/sandbox');
				if (isWebGPUSupported()) {
					sandbox = new CinemaSandbox(canvasHost);
					await sandbox.boot();
					webgpuActive = true;
				}
			} catch (e) {
				console.warn('[cinema] WebGPU sandbox unavailable, camera-only mode:', e);
				sandbox = null;
				webgpuActive = false;
			}
		}

		// Director drives the camera (sandbox camera if WebGPU, else a virtual one).
		const cam = sandbox?.cameraRef ?? new THREE.PerspectiveCamera(60, 1.6, 0.1, 2000);
		const target = sandbox?.target ?? new THREE.Vector3();
		director = new CinemaDirector(cam, target, currentPositions, path, {
			onBeat,
			onProgress: (t) => (progress = t),
			onComplete: () => {
				stage = 'done';
				statusLine = 'End of tour.';
			},
		}, { reducedMotion });

		stage = 'playing';
		statusLine = webgpuActive
			? 'Rendering 150k-particle semantic storm on WebGPU…'
			: 'Cinematic flythrough (captions mode).';
		lastFrame = performance.now();
		director.start();
		loop();
	}

	let renderFailures = 0;
	function loop() {
		rafId = requestAnimationFrame(loop);
		const now = performance.now();
		const dt = Math.min(0.05, (now - lastFrame) / 1000);
		lastFrame = now;
		// The camera director is the bulletproof core — it must advance every
		// frame regardless of whether the WebGPU render succeeds.
		try {
			director?.update(dt);
		} catch (e) {
			console.warn('[cinema] director error:', e);
		}
		if (sandbox && webgpuActive) {
			sandbox.render(dt).catch((e) => {
				// A render failure must never stall the tour. After a few
				// consecutive failures, drop to camera-only (captions still play).
				if (++renderFailures >= 3) {
					console.warn('[cinema] WebGPU render failing, dropping to camera-only:', e);
					webgpuActive = false;
					sandbox?.dispose();
					sandbox = null;
				}
			});
		}
	}

	function close() {
		cancelAnimationFrame(rafId);
		if (typeTimer) clearInterval(typeTimer);
		if (typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
		director?.stop();
		sandbox?.dispose();
		sandbox = null;
		director = null;
		open = false;
		stage = 'idle';
		webgpuActive = false;
	}

	// Opt-in on-device narration. Lazy-loads Transformers.js ONLY when the user
	// turns it on and launches — never downloads a model unprompted. Falls back
	// to local captions if the model isn't present (it isn't bundled).
	function localAiFetcher() {
		return async () => {
			try {
				statusLine = 'Loading on-device model (first run downloads weights)…';
				// Dynamic import via a computed specifier so TypeScript/Vite don't
				// try to resolve the (optional, un-bundled) package at build time.
				// Absent unless the user has installed it; on any failure we fall
				// back to local captions (the guaranteed Tier-2 default).
				const pkg = '@huggingface/transformers';
				const mod = await import(/* @vite-ignore */ pkg).catch(() => null);
				if (!mod || !path) return null;
				// On-device narration hook point. Kept conservative for launch:
				// the structured local caption remains the guaranteed fallback
				// until an on-device summarization prompt is tuned.
				return null;
			} catch {
				return null;
			}
		};
	}

	onDestroy(close);
</script>

<button
	class="cinema-launch glass rounded-full px-4 py-2 text-sm text-bright flex items-center gap-2 hover:scale-[1.03] transition"
	onclick={launch}
	aria-label="Start Memory Cinema — an AI-narrated flythrough of your memory"
>
	<span aria-hidden="true">🎬</span> Memory Cinema
</button>

{#if open}
	<div class="cinema-overlay" role="dialog" aria-modal="true" aria-label="Memory Cinema">
		<div class="cinema-canvas" bind:this={canvasHost}></div>

		<!-- Top bar: status + close -->
		<div class="cinema-top glass-subtle">
			<div class="flex items-center gap-2 text-xs text-dim">
				<span class="cinema-dot" class:active={stage === 'playing'}></span>
				<span>{statusLine}</span>
				{#if narrationSource}
					<span class="cinema-badge">{narrationSource === 'backend-llm' ? 'AI narration' : 'Live captions'}</span>
				{/if}
				{#if webgpuActive}<span class="cinema-badge cinema-badge-gpu">WebGPU</span>{/if}
			</div>
			<div class="flex items-center gap-2">
				<label class="cinema-toggle" title="Speak narration aloud">
					<input type="checkbox" bind:checked={voiceOn} /> Voice
				</label>
				<label class="cinema-toggle" title="Use an on-device model for narration (downloads weights on first use)">
					<input type="checkbox" bind:checked={localAiOn} /> Local AI
				</label>
				<button class="cinema-close" onclick={close} aria-label="Close Memory Cinema">✕</button>
			</div>
		</div>

		<!-- Bottom: narration captions + progress -->
		<div class="cinema-caption-wrap">
			{#if chip}<div class="cinema-chip">{chip}</div>{/if}
			<p class="cinema-caption">{caption}</p>
			<div class="cinema-progress" aria-hidden="true">
				<div class="cinema-progress-fill" style="width:{progress * 100}%"></div>
			</div>
			<div class="cinema-beatcount text-dim text-xs">
				{#if totalBeats > 0}Beat {beatIndex} / {totalBeats}{/if}
				{#if stage === 'done'}<button class="cinema-replay" onclick={launch}>↻ Replay</button>{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.cinema-overlay {
		position: fixed;
		inset: 0;
		z-index: 90;
		background: radial-gradient(circle at 50% 40%, #05050f 0%, #010108 100%);
		display: flex;
		flex-direction: column;
	}
	.cinema-canvas {
		position: absolute;
		inset: 0;
		z-index: 0;
	}
	.cinema-top {
		position: relative;
		z-index: 2;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding: max(0.75rem, env(safe-area-inset-top)) 1rem 0.75rem;
		flex-wrap: wrap;
	}
	.cinema-badge {
		font-size: 0.65rem;
		padding: 0.1rem 0.45rem;
		border-radius: 999px;
		border: 1px solid rgba(129, 140, 248, 0.4);
		color: var(--color-synapse-glow);
	}
	.cinema-badge-gpu {
		border-color: rgba(20, 232, 198, 0.5);
		color: #14e8c6;
	}
	.cinema-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--color-muted);
	}
	.cinema-dot.active {
		background: #14e8c6;
		box-shadow: 0 0 10px #14e8c6;
	}
	.cinema-toggle {
		font-size: 0.7rem;
		color: var(--color-dim);
		display: flex;
		align-items: center;
		gap: 0.3rem;
		cursor: pointer;
	}
	.cinema-close {
		background: transparent;
		border: 1px solid rgba(255, 255, 255, 0.15);
		color: var(--color-text);
		border-radius: 8px;
		width: 32px;
		height: 32px;
		cursor: pointer;
	}
	.cinema-caption-wrap {
		position: relative;
		z-index: 2;
		margin-top: auto;
		padding: 1rem 1.25rem max(1.25rem, env(safe-area-inset-bottom));
		max-width: 720px;
	}
	.cinema-chip {
		display: inline-block;
		font-size: 0.65rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-dream-glow);
		margin-bottom: 0.35rem;
	}
	.cinema-caption {
		font-size: clamp(1.05rem, 2.4vw, 1.6rem);
		line-height: 1.45;
		color: var(--color-bright);
		text-shadow: 0 2px 24px rgba(0, 0, 0, 0.9);
		min-height: 2.6em;
		margin: 0 0 0.75rem;
	}
	.cinema-progress {
		height: 3px;
		background: rgba(255, 255, 255, 0.1);
		border-radius: 3px;
		overflow: hidden;
	}
	.cinema-progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--color-synapse), var(--color-dream));
		transition: width 0.2s linear;
	}
	.cinema-beatcount {
		margin-top: 0.4rem;
		display: flex;
		gap: 0.75rem;
		align-items: center;
	}
	.cinema-replay {
		background: transparent;
		border: 1px solid rgba(129, 140, 248, 0.4);
		color: var(--color-synapse-glow);
		border-radius: 999px;
		padding: 0.15rem 0.7rem;
		cursor: pointer;
		font-size: 0.75rem;
	}
	@media (prefers-reduced-motion: reduce) {
		.cinema-progress-fill {
			transition: none;
		}
	}
</style>
