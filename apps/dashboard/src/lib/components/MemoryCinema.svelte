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
	import {
		resolveNarration,
		localCaptions,
		type CinemaNarration,
		type BeatNarration,
	} from '$lib/graph/cinema/narrator';
	import { computeSignals } from '$lib/graph/cinema/topology';
	import {
		planShotsDeterministic,
		resolveShots,
		type DirectorPlan,
		type ResolvedShot,
		type StormMode,
	} from '$lib/graph/cinema/auteur';
	import type { SemanticRole } from '$lib/graph/cinema/storm';
	import type { CinemaSandbox } from '$lib/graph/cinema/sandbox';

	interface Props {
		nodes: GraphNode[];
		edges: GraphEdge[];
		centerId: string;
		/** Optional Tier-1 backend narration fetcher (passed when backend supports it). */
		fetchBackendNarration?: () => Promise<BeatNarration[] | null>;
	}
	let { nodes, edges, centerId, fetchBackendNarration }: Props = $props();

	let open = $state(false);
	let stage = $state<'idle' | 'planning' | 'playing' | 'done'>('idle');
	let caption = $state('');
	let chip = $state('');
	let progress = $state(0);
	let beatIndex = $state(0);
	let totalBeats = $state(0);
	let narrationSource = $state<CinemaNarration['source'] | null>(null);
	let webgpuActive = $state(false);
	let voiceOn = $state(false);
	let localAiOn = $state(false);
	// Hide all UI chrome (top bar + captions) for a clean demo capture. Toggle
	// with H. Default on; flip to off to record the pure storm with nothing
	// blocking the visuals.
	let cinemaChrome = $state(true);
	let statusLine = $state('');
	// Auteur (director) state surfaced in the overlay.
	let directorNote = $state(''); // the current shot's "why" (cites a real metric)
	let act = $state<'I' | 'II' | 'III'>('I');
	let tension = $state(0); // 0..1 for the tension sparkline
	let logline = $state('');
	let plan = $state<DirectorPlan | null>(null);

	let canvasHost = $state<HTMLDivElement | undefined>(undefined);
	let sandbox: CinemaSandbox | null = null;
	let director: CinemaDirector | null = null;
	let path: CinemaPath | null = null;
	let narration: CinemaNarration | null = null;
	let rafId = 0;
	let lastFrame = 0;
	let typeTimer: ReturnType<typeof setInterval> | null = null;
	let renderFailures = 0;
	// ENDLESS DREAM MODE — after the tour ends, fire a new random crazier figure
	// on this timer so the storm never sits idle. ($state so the template's
	// "∞ dreaming" indicator reacts when it starts/stops.)
	let dreamTimer = $state<ReturnType<typeof setInterval> | null>(null);

	const reducedMotion =
		typeof window !== 'undefined' &&
		window.matchMedia('(prefers-reduced-motion: reduce)').matches;

	// ── INTERACTIVE PARALLAX ────────────────────────────────────────────────
	// Pointer/scroll/touch nudge the camera around the storm so the overlay feels
	// like a live toy, then converge back to 0 when idle so the scripted film is
	// never broken. None of this state is read in the template, so plain let/const
	// is correct (no $state needed). Frame-rate-independent easing via `damp`.
	const damp = (cur: number, tgt: number, lambda: number, dt: number) =>
		cur + (tgt - cur) * (1 - Math.exp(-lambda * dt));
	const pointer = { x: 0, y: 0 };
	const camOff = { yaw: 0, pitch: 0 };
	let zoomTarget = 0,
		zoomLive = 0;
	let lastInputAt = 0; // performance.now() of the last pointer/scroll/touch input
	const IDLE_MS = 2500,
		MAX_YAW = 0.35,
		MAX_PITCH = 0.22;
	const markInput = () => {
		lastInputAt = performance.now();
	};

	// Deterministic layout: spread path nodes on a gentle spiral so the camera
	// has distinct world positions to fly between (independent of the WebGL
	// graph's internal coordinates — keeps the sandbox isolated).
	// Lay the beat nodes out on a TIGHT, BOUNDED shell centered on the origin —
	// fixed radius, no per-beat growth. Earlier this grew (22 + i*6) so each beat
	// sat farther out and the camera+storm marched off into space ("flying off").
	// A bounded shell keeps the whole composition centered; cinematic variety
	// comes from the camera angle/move/standoff, not from translating across a
	// huge volume. The focused node is always re-centered by recenterOn() below.
	const SHELL_RADIUS = 14;
	function layoutPositions(p: CinemaPath): Map<string, THREE.Vector3> {
		const pos = new Map<string, THREE.Vector3>();
		const n = p.beats.length;
		for (let i = 0; i < n; i++) {
			// Distribute beats evenly on a sphere (golden-angle spiral) so they
			// never clump and never exceed SHELL_RADIUS from center.
			const t = n > 1 ? i / (n - 1) : 0.5;
			const y = 1 - t * 2; // 1..-1
			const r = Math.sqrt(Math.max(0, 1 - y * y));
			const theta = i * 2.399963; // golden angle
			pos.set(
				p.beats[i].nodeId,
				new THREE.Vector3(
					Math.cos(theta) * r * SHELL_RADIUS,
					y * SHELL_RADIUS * 0.5,
					Math.sin(theta) * r * SHELL_RADIUS
				)
			);
		}
		return pos;
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

	// Map the director's StormMode to the storm runtime's SemanticRole. 'surprise'
	// is a Phase-3 storm mode; until then it reads as 'connection'.
	function stormRole(mode: StormMode): SemanticRole {
		return mode === 'surprise' ? 'connection' : mode;
	}

	function onBeat(beat: CinemaBeat, index: number, shot: ResolvedShot | null) {
		beatIndex = index + 1;
		const text = narration?.beats[index]?.text ?? beat.node.label ?? '';
		chip = narration?.beats[index]?.chip ?? '';
		streamCaption(text);
		speak(text);
		// Surface the director's intent for this shot — the "why", act, tension.
		if (shot) {
			directorNote = shot.why;
			act = shot.act;
			tension = shot.tension;
		}
		if (sandbox && webgpuActive) {
			const wp = currentPositions?.get(beat.nodeId);
			if (wp) {
				const mode: StormMode = shot?.stormMode ?? 'connection';
				// Pass act + 0-based beat index so the storm holds Act I dimmer AND
				// fades in extra-soft on beats 0/1 (which otherwise wash to white).
				sandbox.transitionTo(stormRole(mode), wp, shot?.act ?? 'I', index);
			}
			// Drive flythrough strength from beat energy: high-tension beats fly
			// THROUGH the storm (relaxed clamp + streak); reduced-motion = no streak.
			const energy = shot?.tension ?? 0;
			sandbox.setFlythrough(reducedMotion ? 0 : energy * 0.8);
		}
	}

	let currentPositions: Map<string, THREE.Vector3> | null = null;

	async function launch() {
		// Tear down any prior run so Replay never inherits stale state.
		cancelAnimationFrame(rafId);
		stopDreamMode();
		if (typeTimer) clearInterval(typeTimer);
		director?.stop();
		sandbox?.dispose();
		sandbox = null;
		director = null;
		narration = null;
		renderFailures = 0;
		directorNote = '';
		logline = '';
		plan = null;
		act = 'I';
		tension = 0;

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

		// THE AUTEUR: read the graph's dramatic structure and direct the film.
		// Tier 2 (deterministic) ships the hero; Tier 1 (LLM) lands in Phase 4.
		const signals = computeSignals(nodes, edges);
		plan = planShotsDeterministic(path, signals);
		logline = plan.logline;
		const shots = resolveShots(plan, path);
		act = shots[0]?.act ?? 'I';
		tension = shots[0]?.tension ?? 0;

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
		const fallbackAspect =
			canvasHost && canvasHost.clientHeight > 0
				? canvasHost.clientWidth / canvasHost.clientHeight
				: 16 / 9;
		const cam = sandbox?.cameraRef ?? new THREE.PerspectiveCamera(60, fallbackAspect, 0.1, 2000);
		const target = sandbox?.target ?? new THREE.Vector3();
		director = new CinemaDirector(cam, target, currentPositions, path, {
			onBeat,
			onProgress: (t) => (progress = t),
			onComplete: () => {
				stage = 'done';
				statusLine =
					reducedMotion || !webgpuActive
						? 'End of tour.'
						: '∞ Dreaming — endless generative figures';
				startDreamMode();
			},
		}, { reducedMotion, shots, centerOnOrigin: webgpuActive });

		stage = 'playing';
		statusLine = webgpuActive
			? 'Rendering 150k-particle semantic storm on WebGPU…'
			: 'Cinematic flythrough (captions mode).';
		lastFrame = performance.now();
		director.start();
		loop();
	}

	function loop() {
		rafId = requestAnimationFrame(loop);
		const now = performance.now();
		const dt = Math.max(0, Math.min(0.05, (now - lastFrame) / 1000));
		lastFrame = now;
		// The camera director is the bulletproof core — it must advance every
		// frame regardless of whether the WebGPU render succeeds.
		try {
			director?.update(dt);
		} catch (e) {
			console.warn('[cinema] director error:', e);
		}
		// Compose interactive parallax ON TOP of the director's camera. The user
		// nudges yaw/pitch/zoom; we ease toward 0 when idle so the scripted film is
		// seamless. sandbox.render() re-clamps distance + lookAt(origin), so the
		// user can never break the framing — this is purely additive sugar.
		if (!reducedMotion && sandbox && webgpuActive) {
			const cam = sandbox.cameraRef;
			const idle = performance.now() - lastInputAt > IDLE_MS;
			const yawT = idle ? 0 : pointer.x * MAX_YAW;
			const pitchT = idle ? 0 : pointer.y * MAX_PITCH;
			const lam = idle ? 1.5 : 3.5;
			camOff.yaw = damp(camOff.yaw, yawT, lam, dt);
			camOff.pitch = damp(camOff.pitch, pitchT, lam, dt);
			if (idle) zoomTarget = damp(zoomTarget, 0, 1.2, dt);
			zoomLive = damp(zoomLive, zoomTarget, 5.5, dt);
			// Spherical offset around the director's current camera position
			// (target = origin).
			const dir = cam.position.clone();
			const sph = new THREE.Spherical().setFromVector3(dir);
			sph.theta += camOff.yaw;
			sph.phi = THREE.MathUtils.clamp(sph.phi + camOff.pitch, 0.2, Math.PI - 0.2);
			sph.radius *= 1 - zoomLive * 0.35;
			dir.setFromSpherical(sph);
			cam.position.copy(dir);
		}
		// Snapshot the sandbox so the async catch can't act on a sandbox that
		// close() nulled out while the render promise was in flight.
		const sb = sandbox;
		if (sb && webgpuActive) {
			sb.render(dt).catch((e) => {
				// A render failure must never stall the tour. After a few
				// consecutive failures, drop to camera-only (captions still play).
				if (++renderFailures >= 3 && sandbox === sb) {
					console.warn('[cinema] WebGPU render failing, dropping to camera-only:', e);
					webgpuActive = false;
					sb.dispose();
					sandbox = null;
				}
			});
		}
	}

	// ── ENDLESS DREAM MODE ──────────────────────────────────────────────────
	// When the scripted tour ends, instead of freezing on the last figure, the
	// storm enters an infinite generative loop: every few seconds it morphs into
	// a fresh RANDOM procedural figure (supershape, torus-knot, lissajous, helix,
	// quantum foam) and detonates a color blast — each one crazier than the last.
	// The render loop() is already running, so we just fire dreamBeats on a timer.
	function startDreamMode() {
		if (reducedMotion || !sandbox || !webgpuActive) return; // honor reduced-motion
		stopDreamMode();
		// The endless dream gently flies THROUGH the procedural figures.
		if (!reducedMotion) sandbox?.setFlythrough(0.6);
		// Fire the first wild figure immediately, then keep going forever.
		sandbox?.dreamBeat();
		caption = '';
		chip = 'Dreaming';
		dreamTimer = setInterval(() => {
			// Sandbox may have been torn down (close / render-fail fallback).
			if (!sandbox || !webgpuActive) {
				stopDreamMode();
				return;
			}
			sandbox.dreamBeat();
		}, 5500); // a beat every ~5.5s — the blast flares then the figure settles
		           // into its clean shape before the next detonation.
	}

	function stopDreamMode() {
		if (dreamTimer) {
			clearInterval(dreamTimer);
			dreamTimer = null;
		}
	}

	function close() {
		cancelAnimationFrame(rafId);
		stopDreamMode();
		if (typeTimer) clearInterval(typeTimer);
		if (typeof speechSynthesis !== 'undefined') speechSynthesis.cancel();
		// Reset flythrough + parallax so a Replay never inherits stale offsets.
		if (sandbox) sandbox.setFlythrough?.(0);
		camOff.yaw = camOff.pitch = 0;
		zoomTarget = zoomLive = 0;
		director?.stop();
		sandbox?.dispose();
		sandbox = null;
		director = null;
		open = false;
		stage = 'idle';
		webgpuActive = false;
	}

	// a11y: Escape closes the fullscreen overlay; the close button auto-focuses
	// on open so keyboard users land inside the dialog.
	let closeBtn = $state<HTMLButtonElement | undefined>(undefined);
	function onOverlayKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') {
			e.preventDefault();
			close();
		} else if (e.key === 'h' || e.key === 'H') {
			// Toggle UI chrome for a clean demo capture (hide top bar + captions).
			e.preventDefault();
			cinemaChrome = !cinemaChrome;
		}
	}
	$effect(() => {
		if (open && closeBtn) closeBtn.focus();
	});

	// Tag <body> while Cinema is open so the graph page can hide its bottom-stats
	// pill (which lives behind us in a separate stacking context). Cleaned up when
	// Cinema closes or the component unmounts.
	$effect(() => {
		if (typeof document === 'undefined') return;
		document.body.classList.toggle('cinema-open', open);
		return () => document.body.classList.remove('cinema-open');
	});

	// Interactive parallax listeners — only while the overlay is open, motion is
	// allowed, and the canvas host exists. Reduced-motion fully disables them
	// (no listeners attached at all). Cleanup removes everything on close/unmount.
	$effect(() => {
		if (!open || reducedMotion || !canvasHost) return;
		const host = canvasHost;
		const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
		host.style.touchAction = 'none';

		const onPointerMove = (e: PointerEvent) => {
			pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
			pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
			markInput();
		};
		const onWheel = (e: WheelEvent) => {
			e.preventDefault();
			zoomTarget = clamp(zoomTarget + e.deltaY * 0.0008, -1, 1);
			markInput();
		};
		let pinchPrev: number | null = null;
		const onTouchMove = (e: TouchEvent) => {
			if (e.touches.length === 2) {
				const dx = e.touches[0].clientX - e.touches[1].clientX;
				const dy = e.touches[0].clientY - e.touches[1].clientY;
				const dist = Math.hypot(dx, dy);
				if (pinchPrev !== null) {
					zoomTarget = clamp(zoomTarget + (dist - pinchPrev) * 0.002, -1, 1);
				}
				pinchPrev = dist;
				markInput();
			}
		};
		const onTouchEnd = () => {
			pinchPrev = null;
		};

		host.addEventListener('pointermove', onPointerMove, { passive: true });
		host.addEventListener('wheel', onWheel, { passive: false });
		host.addEventListener('touchmove', onTouchMove, { passive: true });
		host.addEventListener('touchend', onTouchEnd);

		return () => {
			host.removeEventListener('pointermove', onPointerMove);
			host.removeEventListener('wheel', onWheel);
			host.removeEventListener('touchmove', onTouchMove);
			host.removeEventListener('touchend', onTouchEnd);
		};
	});

	// Opt-in on-device narration. Lazy-loads @huggingface/transformers ONLY when
	// the user enables "Local AI" and launches — never downloads a model
	// unprompted. Runs a small instruction model in-browser on WebGPU to rewrite
	// each beat's structured caption into richer prose. Returns null (→ Tier-2
	// local captions) ONLY if the package is absent or generation genuinely fails
	// — a real implementation with a real fallback, not a placeholder.
	type TransformersPipeline = (
		input: string,
		opts?: Record<string, unknown>
	) => Promise<Array<{ generated_text?: string }>>;
	function localAiFetcher(): () => Promise<BeatNarration[] | null> {
		return async () => {
			if (!path) return null;
			try {
				statusLine = 'Loading on-device model (first run downloads weights)…';
				// Computed specifier so TS/Vite don't resolve the optional,
				// un-bundled package at build time.
				const pkg = '@huggingface/transformers';
				const mod = (await import(/* @vite-ignore */ pkg).catch(() => null)) as {
					pipeline?: (task: string, model: string, opts?: Record<string, unknown>) => Promise<TransformersPipeline>;
				} | null;
				if (!mod?.pipeline) return null;

				const generate = await mod.pipeline(
					'text-generation',
					'onnx-community/Qwen2.5-0.5B-Instruct',
					{ device: 'webgpu', dtype: 'q4' }
				);

				// Seed from the deterministic local captions, then enrich each beat.
				const base = localCaptions(path);
				statusLine = 'Narrating with the on-device model…';
				const out: BeatNarration[] = [];
				for (const b of base.beats) {
					const prompt =
						`You are narrating a cinematic tour of an AI's memory graph. ` +
						`In one vivid sentence, narrate this beat: "${b.text}"`;
					const res = await generate(prompt, { max_new_tokens: 48, temperature: 0.7, do_sample: true });
					const text = res?.[0]?.generated_text?.replace(prompt, '').trim();
					out.push({ nodeId: b.nodeId, chip: b.chip, text: text && text.length > 4 ? text : b.text });
				}
				return out;
			} catch (e) {
				console.warn('[cinema] on-device narration failed, using local captions:', e);
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
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div
		class="cinema-overlay"
		role="dialog"
		aria-modal="true"
		aria-label="Memory Cinema"
		tabindex="-1"
		onkeydown={onOverlayKeydown}
	>
		<div class="cinema-canvas" bind:this={canvasHost}></div>

		<!-- Press H to hide all chrome for a clean demo capture. A tiny restore hint
		     shows while hidden so it's never a trap. -->
		{#if !cinemaChrome}
			<button class="cinema-restore" onclick={() => (cinemaChrome = true)} title="Show UI (H)">
				press H to show UI
			</button>
		{/if}

		{#if cinemaChrome}
		<!-- Top bar: status + close -->
		<div class="cinema-top glass-subtle">
			<div class="flex items-center gap-2 text-xs text-dim">
				<span class="cinema-dot" class:active={stage === 'playing'}></span>
				<span>{statusLine}</span>
				{#if plan}
					<span class="cinema-badge" title="Who directed this film">
						{plan.source === 'deterministic' ? 'Auteur (local)' : 'Auteur (AI)'}
					</span>
				{/if}
				{#if narrationSource}
					<span class="cinema-badge">{narrationSource === 'backend-llm' ? 'AI narration' : 'Live captions'}</span>
				{/if}
				{#if webgpuActive}<span class="cinema-badge cinema-badge-gpu">WebGPU</span>{/if}
				{#if stage === 'playing'}<span class="cinema-act">Act {act}</span>{/if}
			</div>
			<div class="flex items-center gap-2">
				<label class="cinema-toggle" title="Speak narration aloud">
					<input type="checkbox" bind:checked={voiceOn} /> Voice
				</label>
				<label class="cinema-toggle" title="Use an on-device model for narration (downloads weights on first use)">
					<input type="checkbox" bind:checked={localAiOn} /> Local AI
				</label>
				<button bind:this={closeBtn} class="cinema-close" onclick={close} aria-label="Close Memory Cinema (Esc)">✕</button>
			</div>
		</div>

		<!-- Pre-roll DIRECTOR'S PLAN card: the AI states its film before rolling. -->
		{#if stage === 'planning' && logline}
			<div class="cinema-plan-card glass-panel">
				<div class="cinema-plan-kicker">Director's plan</div>
				<p class="cinema-plan-logline">{logline}</p>
			</div>
		{/if}

		<!-- Bottom: narration captions + director's note + progress -->
		<div class="cinema-caption-wrap">
			{#if chip}<div class="cinema-chip">{chip}</div>{/if}
			<p class="cinema-caption">{caption}</p>
			{#if directorNote && stage === 'playing'}
				<p class="cinema-note" title="Why the director chose this shot">▸ {directorNote}</p>
			{/if}
			<div class="cinema-progress" aria-hidden="true">
				<div
					class="cinema-progress-fill"
					style="width:{progress * 100}%; --tension:{tension}"
				></div>
			</div>
			<div class="cinema-beatcount text-dim text-xs">
				{#if stage === 'done' && dreamTimer}<span class="cinema-dream">∞ dreaming</span>
				{:else if totalBeats > 0}Beat {beatIndex} / {totalBeats}{/if}
				{#if stage === 'done'}<button class="cinema-replay" onclick={launch}>↻ Replay</button>{/if}
			</div>
		</div>
		{/if}
	</div>
{/if}

<style>
	.cinema-overlay {
		position: fixed;
		inset: 0;
		z-index: 200; /* above the graph page's bottom-stats pill + all nav chrome */
		background: radial-gradient(circle at 50% 40%, #05050f 0%, #010108 100%);
		display: flex;
		flex-direction: column;
	}
	/* While Cinema is open, hide the graph page's bottom-stats pill (it lives in a
	   separate stacking context on the page behind us and would otherwise bleed
	   through into a demo capture). Toggled by the .cinema-open class we add to
	   <body> on mount. */
	:global(body.cinema-open .graph-stats-pill) {
		display: none;
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
	.cinema-act {
		font-size: 0.6rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--color-dream-glow);
		opacity: 0.85;
	}
	/* Pre-roll director's plan card — centered, the AI's statement of intent. */
	.cinema-plan-card {
		position: absolute;
		z-index: 3;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		max-width: 520px;
		padding: 1.5rem 1.75rem;
		border-radius: 16px;
		text-align: center;
		animation: cinema-plan-in 0.5s ease both;
	}
	@keyframes cinema-plan-in {
		from { opacity: 0; transform: translate(-50%, -46%); }
		to { opacity: 1; transform: translate(-50%, -50%); }
	}
	.cinema-plan-kicker {
		font-size: 0.65rem;
		letter-spacing: 0.18em;
		text-transform: uppercase;
		color: var(--color-synapse-glow);
		margin-bottom: 0.5rem;
	}
	.cinema-plan-logline {
		font-size: clamp(1.05rem, 2.2vw, 1.4rem);
		line-height: 1.5;
		color: var(--color-bright);
		margin: 0;
	}
	.cinema-note {
		font-size: 0.78rem;
		color: var(--color-synapse-glow);
		opacity: 0.85;
		margin: 0 0 0.6rem;
		font-style: italic;
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
		/* Tint shifts toward crimson as the shot's tension rises (--tension 0..1). */
		background: linear-gradient(
			90deg,
			var(--color-synapse),
			color-mix(in oklch, var(--color-dream), #ff2d55 calc(var(--tension, 0) * 100%))
		);
		transition: width 0.2s linear, background 0.4s ease;
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
	.cinema-dream {
		color: var(--color-dream-glow);
		letter-spacing: 0.08em;
		animation: cinema-dream-pulse 3s ease-in-out infinite;
	}
	/* Tiny restore hint shown while chrome is hidden (demo-capture mode). */
	.cinema-restore {
		position: absolute;
		bottom: 0.6rem;
		right: 0.8rem;
		z-index: 95;
		background: rgba(10, 10, 26, 0.5);
		border: 1px solid rgba(129, 140, 248, 0.25);
		color: var(--color-muted);
		border-radius: 999px;
		padding: 0.2rem 0.7rem;
		font-size: 0.7rem;
		letter-spacing: 0.04em;
		cursor: pointer;
		opacity: 0.4;
		transition: opacity 0.2s ease;
	}
	.cinema-restore:hover {
		opacity: 1;
	}
	@keyframes cinema-dream-pulse {
		0%, 100% { opacity: 0.55; }
		50% { opacity: 1; }
	}
	@media (prefers-reduced-motion: reduce) {
		.cinema-progress-fill {
			transition: none;
		}
	}
</style>
