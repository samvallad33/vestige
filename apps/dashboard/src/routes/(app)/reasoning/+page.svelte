<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import RouteStage, { type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import {
		normalizeDeepReferenceResponse,
		type ReasoningScene,
		type NormalizedEvidence
	} from '$lib/observatory/reasoning/reasoning-scene';
	import { createReasoningTracePasses } from '$lib/observatory/reasoning/reasoning-trace-pass';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import type { RouteFramePass } from '$lib/observatory/RouteStage.svelte';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	// ── Observable Decision Trace ────────────────────────────────────────────
	// This route is now a full-bleed WebGPU "observable decision trace": one
	// query becomes a left-to-right causal beam whose evidence, contradictions,
	// supersessions and recommendation are STAGED as the decision forms. The old
	// DOM card-stack (header / form / answer / confidence / pipeline / evidence
	// grid) is gone — the only DOM is a visually-hidden native input + an
	// aria-live output region for keyboard + screen-reader access. Everything
	// visible is rendered in-canvas via the reasoning-trace pass.
	//
	// Honest framing: this is a ONE-SHOT deep_reference response replayed as a
	// trace — never a pretend token stream. Every bright object maps to a real
	// field of the response.

	let query = $state('');
	let loading = $state(false);
	let error: string | null = $state(null);
	let reasoningScene: ReasoningScene | null = $state(null);
	let selection: string | null = $state(null); // aria description of last pick
	let askInputEl: HTMLInputElement | null = $state(null);
	// Live viewport aspect → portrait entry affordance. Desktop (aspect>=0.85) is
	// byte-identical: the DOM overlay never mounts and the full canvas empty label
	// renders exactly as before. Only narrow/portrait phones get the tappable UI.
	let portrait = $state(false);
	// A visible portrait DOM input the user can TAP to ask (phones have no Cmd+K).
	let mobileInputEl: HTMLInputElement | null = $state(null);

	function syncPortrait() {
		if (typeof window === 'undefined') return;
		const vw = window.innerWidth;
		const vh = window.innerHeight;
		portrait = vw > 0 && vh > 0 && vw / vh < 0.85;
	}

	// Desktop keeps the exact original canvas guidance. Portrait blanks the canvas
	// label (the DOM overlay owns guidance) so the long line can't clip off-screen.
	const emptyLabel = $derived(
		portrait ? '' : 'ASK A QUESTION - PRESS CMD+K - WATCH THE DECISION FORM'
	);

	// Evidence galaxy behind the trace: every real deep_reference evidence memory
	// becomes a cell (trust = oxygen, contradictions scar) so the Theater is a
	// full-bleed field, not a thin beam on black. The beam/ribbon/nucleus trace
	// draws ON TOP. Rebuilds whenever a new decision trace lands.
	let evidenceField: LivingFieldPass | null = null;
	// Passive real memory pool shown at rest (pre-query), dim, so the stage is
	// alive without faking a decision trace. Replaced by real evidence on a query.
	let restPool: NormalizedEvidence[] = [];
	function buildEvidenceCells() {
		const s = reasoningScene;
		// Pre-query: dim memory-pool substrate (honest — no trace claimed).
		if (!s || (s.evidence ?? []).length === 0) {
			const data: FieldDatum[] = restPool.map((e, i) => ({
				id: e.id || `rest:${i}`,
				score: 0.25 + 0.4 * clampR(e.trust ?? 0.5),
				hue: FIELD_HUE.bridge,
				energy: 0.14 + 0.26 * clampR(e.trust ?? 0.5),
				metric2: clampR(e.trust ?? 0.5),
				kind: 'reasoning-rest',
				payload: e
			}));
			return layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.014, maxCellR: 0.045 });
		}
		const contradictionIds = new Set(
			(s.contradictions ?? []).flatMap((c) => [c.stronger?.id, c.weaker?.id]).filter(Boolean) as string[]
		);
		const data: FieldDatum[] = (s.evidence ?? []).map((e: NormalizedEvidence, i) => ({
			id: e.id || `evidence:${i}`,
			score: 0.4 + 0.6 * clampR(e.trust ?? 0.5),
			hue: contradictionIds.has(e.id) ? FIELD_HUE.scarlet : FIELD_HUE.forward,
			energy: 0.45 + 0.55 * clampR(e.trust ?? 0.5),
			metric2: clampR(e.trust ?? 0.5),
			scar: contradictionIds.has(e.id),
			selected: e.id === s.recommended?.memory_id,
			kind: 'reasoning-evidence',
			payload: e
		}));
		return layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.016, maxCellR: 0.06 });
	}
	function clampR(v: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0.5));
	}
	function createReasoningPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		const field = new LivingFieldPass(engine);
		// Keep portrait byte-for-byte at the verified dim 0.2. Desktop has a broad
		// reading well over the full trace, so the field can be richer around it
		// without washing out the MSDF query, gates, evidence, or receipt text.
		const vw = engine.params[6] || (typeof window !== 'undefined' ? window.innerWidth : 0);
		const vh = engine.params[7] || (typeof window !== 'undefined' ? window.innerHeight : 0);
		const aspect = vw > 0 && vh > 0 ? vw / vh : 1;
		const desktop = aspect >= 0.85;
		field.setIntensity(desktop ? 0.9 : 0.2);
		field.setReadingWell(
			desktop
				? { x: 0, y: 0.15, hw: 0.95, hh: 0.72, floor: 0.22, soft: 0.22 }
				: { x: 0, y: 0.15, hw: 0.95, hh: 0.72, floor: 0.06, soft: 0.22 }
		);
		evidenceField = field;
		field.setCells(buildEvidenceCells());
		const fieldWrapper: RouteFramePass = {
			compute: (encoder) => field.compute(encoder),
			render: (pass) => field.render(pass),
			dispose: () => {
				field.dispose();
				if (evidenceField === field) evidenceField = null;
			}
		};
		// Evidence field FIRST (behind), then the real beam/ribbon/nucleus trace.
		return [fieldWrapper, ...createReasoningTracePasses(engine, scene)];
	}
	$effect(() => {
		void reasoningScene?.evidence.length;
		evidenceField?.setCells(buildEvidenceCells());
	});

	const EXAMPLE_QUERIES = [
		'What port does the dev server use?',
		'Should I enable prefix caching with vLLM?',
		'How does FSRS-6 trust scoring work?',
		'Why did the benchmark score drop after the parser change?'
	];

	async function ask() {
		const q = query.trim();
		if (!q || loading) return;
		loading = true;
		error = null;
		reasoningScene = null;
		selection = null;
		try {
			const raw = (await api.deepReference(q, 20)) as Record<string, unknown>;
			reasoningScene = normalizeDeepReferenceResponse(raw);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			loading = false;
		}
	}

	/** Live-region summary of what the trace currently shows (screen readers). */
	const ariaSummary = $derived.by(() => {
		if (loading) return 'Reasoning in progress.';
		if (error) return `Error: ${error}`;
		if (!reasoningScene) return 'Ask a question to trace how Vestige forms a decision from memory.';
		const s = reasoningScene;
		const rec = s.recommended?.answer_preview ?? 'no recommendation';
		return (
			`Decision trace for "${query}". ${s.evidence.length} evidence memories, ` +
			`${s.contradictions.length} contradiction${s.contradictions.length === 1 ? '' : 's'}, ` +
			`${s.superseded.length} superseded. Recommendation: ${rec}. ` +
			`Confidence ${Math.round((s.recommended?.trust_score ?? 0) * 100)} percent.`
		);
	});

	function handleRoutePick(pick: RoutePick) {
		// Selection is surfaced to screen readers; the pass draws the in-canvas
		// receipt for the picked object (gate / evidence / recommendation).
		const p = pick.payload as { ariaLabel?: string; preview?: string } | undefined;
		selection = p?.ariaLabel ?? p?.preview ?? `${pick.kind} selected`;
	}

	function handleGlobalKey(e: KeyboardEvent) {
		if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
			e.preventDefault();
			askInputEl?.focus();
			askInputEl?.select();
		}
	}

	onMount(() => {
		askInputEl?.focus();
		syncPortrait();
		window.addEventListener('keydown', handleGlobalKey);
		window.addEventListener('resize', syncPortrait);
		window.addEventListener('orientationchange', syncPortrait);
		// Keep the Theater HONEST-EMPTY at rest (the DOM "ask a question to trace"
		// state), but light the field with a passive real memory pool so the stage
		// isn't black while it waits. A real query replaces the pool with the actual
		// evidence galaxy. No auto-run — the user (or a demo) drives the trace.
		void api.memories
			.list({ limit: '80' })
			.then((res) => {
				restPool = res.memories.map((m, i) => {
					const retention = clampR(m.retentionStrength);
					return {
						id: m.id || `rest:${i}`,
						trust: 0.35 + 0.4 * retention,
						date: '',
						role: 'supporting' as const,
						preview: ''
					};
				});
				evidenceField?.setCells(buildEvidenceCells());
			})
			.catch(() => {});
		return () => {
			window.removeEventListener('keydown', handleGlobalKey);
			window.removeEventListener('resize', syncPortrait);
			window.removeEventListener('orientationchange', syncPortrait);
		};
	});
</script>

<svelte:head>
	<title>Reasoning Theater · Vestige</title>
</svelte:head>

<RouteStage
	organ="reasoning"
	seed={`reasoning-trace:${query || 'empty'}`}
	scene={reasoningScene}
	passes={createReasoningPasses}
	loading={loading}
	error={error}
	{emptyLabel}
	onpick={handleRoutePick}
/>

<!--
  PORTRAIT ENTRY AFFORDANCE (phones have no Cmd+K). Desktop never mounts this —
  it's gated on the live viewport aspect, so the 1440px render is untouched. This
  is the ONE focal point at rest on a phone: a title, one line of guidance, and a
  real tappable input that runs the same ask() the canvas trace consumes.
-->
{#if portrait && !reasoningScene && !loading && !error}
	<div class="reasoning-mobile" role="search">
		<h1 class="rm-title">REASONING THEATER</h1>
		<p class="rm-sub">Ask your memory a question and watch the decision form from evidence.</p>
		<form
			class="rm-form"
			onsubmit={(e) => {
				e.preventDefault();
				query = mobileInputEl?.value ?? query;
				void ask();
				mobileInputEl?.blur();
			}}
		>
			<input
				bind:this={mobileInputEl}
				class="rm-input"
				type="search"
				enterkeyhint="search"
				autocomplete="off"
				spellcheck="false"
				placeholder="Ask your memory anything..."
				aria-label="Ask Vestige a question"
			/>
			<button class="rm-go" type="submit">ASK</button>
		</form>
		<ul class="rm-examples">
			{#each EXAMPLE_QUERIES.slice(0, 3) as q}
				<li>
					<button
						type="button"
						class="rm-chip"
						onclick={() => {
							query = q;
							if (mobileInputEl) mobileInputEl.value = q;
							void ask();
						}}>{q}</button
					>
				</li>
			{/each}
		</ul>
	</div>
{/if}

<!--
  The ONLY DOM: a visually-hidden native input (real keyboard + IME, Cmd+K
  focus, example-query datalist) and an aria-live output. The visible query
  echo, gates, evidence, and receipt are all rendered in-canvas by the trace
  pass. Nothing here paints a visible pixel — the field owns the surface.
-->
<form class="sr-only" onsubmit={(e) => { e.preventDefault(); void ask(); }}>
	<label for="reasoning-ask">Ask Vestige a question</label>
	<input
		id="reasoning-ask"
		bind:this={askInputEl}
		bind:value={query}
		list="reasoning-examples"
		autocomplete="off"
		spellcheck="false"
		placeholder="Ask your memory anything…"
	/>
	<datalist id="reasoning-examples">
		{#each EXAMPLE_QUERIES as q}
			<option value={q}></option>
		{/each}
	</datalist>
	<button type="submit">Trace decision</button>
</form>

<div class="sr-only" aria-live="polite" role="status">{ariaSummary}</div>
{#if selection}
	<div class="sr-only" aria-live="polite">{selection}</div>
{/if}

<style>
	/* Standard visually-hidden: keyboard + screen-reader reachable, zero pixels. */
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}

	/* Portrait entry affordance — the readable focal point on a phone. Sits in the
	   upper-center over the dim field, clear of the bottom nav FAB. Never shown on
	   desktop (the {#if portrait} gate keeps it unmounted there). */
	.reasoning-mobile {
		position: fixed;
		left: 0;
		right: 0;
		top: 16vh;
		z-index: 5;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.9rem;
		padding: 0 7vw;
		text-align: center;
		pointer-events: none;
	}
	.reasoning-mobile > * {
		pointer-events: auto;
	}
	.rm-title {
		margin: 0;
		font-size: clamp(1.4rem, 7vw, 2rem);
		font-weight: 600;
		letter-spacing: 0.14em;
		color: #e9ffb7;
		text-shadow: 0 0 24px rgba(0, 245, 212, 0.28);
	}
	.rm-sub {
		margin: 0;
		max-width: 30ch;
		font-size: clamp(0.85rem, 3.6vw, 1rem);
		line-height: 1.45;
		color: rgba(200, 230, 235, 0.72);
	}
	.rm-form {
		display: flex;
		width: 100%;
		max-width: 30rem;
		margin-top: 0.4rem;
		border: 1px solid rgba(0, 245, 212, 0.4);
		border-radius: 0.7rem;
		background: rgba(6, 14, 16, 0.66);
		-webkit-backdrop-filter: blur(6px);
		backdrop-filter: blur(6px);
		overflow: hidden;
	}
	.rm-input {
		flex: 1 1 auto;
		min-width: 0;
		padding: 0.85rem 0.95rem;
		border: 0;
		background: transparent;
		color: #eafcff;
		font-size: 1rem;
		outline: none;
	}
	.rm-input::placeholder {
		color: rgba(160, 190, 195, 0.55);
	}
	.rm-go {
		flex: 0 0 auto;
		padding: 0 1.2rem;
		border: 0;
		border-left: 1px solid rgba(0, 245, 212, 0.3);
		background: rgba(0, 245, 212, 0.14);
		color: #7ffff0;
		font-weight: 700;
		letter-spacing: 0.12em;
		cursor: pointer;
	}
	.rm-go:active {
		background: rgba(0, 245, 212, 0.28);
	}
	.rm-examples {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		width: 100%;
		max-width: 30rem;
		margin: 0.2rem 0 0;
		padding: 0;
		list-style: none;
	}
	.rm-chip {
		width: 100%;
		padding: 0.6rem 0.8rem;
		border: 1px solid rgba(120, 150, 155, 0.25);
		border-radius: 0.55rem;
		background: rgba(10, 18, 20, 0.5);
		color: rgba(190, 220, 225, 0.82);
		font-size: 0.86rem;
		text-align: left;
		cursor: pointer;
	}
	.rm-chip:active {
		border-color: rgba(0, 245, 212, 0.5);
		color: #d6fff8;
	}
</style>
