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

	// Evidence galaxy behind the trace: every real deep_reference evidence memory
	// becomes a cell (trust = oxygen, contradictions scar) so the Theater is a
	// full-bleed field, not a thin beam on black. The beam/ribbon/nucleus trace
	// draws ON TOP. Rebuilds whenever a new decision trace lands.
	let evidenceField: LivingFieldPass | null = null;
	function buildEvidenceCells() {
		const s = reasoningScene;
		if (!s) return [];
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
		window.addEventListener('keydown', handleGlobalKey);
		// Auto-seed a real decision trace so the Theater is ALIVE at rest (the
		// beam/ribbon/nucleus render real deep_reference evidence immediately),
		// not a black stage waiting for input. A demo lands on a live trace; the
		// user can retype to trace their own question.
		if (!query.trim()) {
			query = 'What is the Vestige dashboard direction?';
			void ask();
		}
		return () => window.removeEventListener('keydown', handleGlobalKey);
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
	emptyLabel="ASK A QUESTION - PRESS CMD+K - WATCH THE DECISION FORM"
	onpick={handleRoutePick}
/>

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
		placeholder="Ask a question…"
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
</style>
