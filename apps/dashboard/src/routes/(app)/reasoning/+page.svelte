<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type Receipt } from '$stores/api';
	import { base } from '$app/paths';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AmbientField from '$components/AmbientField.svelte';
	import {
		normalizeDeepReferenceResponse,
		type ReasoningScene,
		type NormalizedEvidence
	} from '$lib/observatory/reasoning/reasoning-scene';

	// Memory Replay is deliberately DOM-first: the value is the proof a person can read,
	// not a decorative simulation of thought.
	let query = $state('');
	let loading = $state(false);
	let error = $state<string | null>(null);
	let scene = $state<ReasoningScene | null>(null);
	let runId = $state<string | null>(null);
	let receiptId = $state<string | null>(null);
	let input: HTMLInputElement | null = $state(null);

	let receiptSeal = $state<Receipt | null>(null);
	const EXAMPLES = ['refund compliance exception', 'What port does the dev server use?', 'How does FSRS-6 trust scoring work?'];
	const confidence = $derived(Math.round((scene?.recommended?.trust_score ?? 0) * 100));
	const receiptUrl = $derived(receiptId ? `${base}/observatory?receipt=${encodeURIComponent(receiptId)}` : null);
	const blackBoxUrl = $derived(runId ? `${base}/blackbox?run=${encodeURIComponent(runId)}` : `${base}/blackbox`);

	function freshRunId() {
		return `run_replay_${Date.now().toString(36)}`;
	}

	function stringAt(raw: Record<string, unknown>, ...keys: string[]) {
		for (const key of keys) {
			const value = raw[key];
			if (typeof value === 'string' && value) return value;
		}
		return null;
	}

	async function runReplay() {
		const question = query.trim();
		if (!question || loading) return;
		loading = true;
		error = null;
		scene = null;
		receiptId = null;
		const nextRunId = freshRunId();
		runId = nextRunId;
		try {
			const raw = (await api.deepReference(question, 20, nextRunId)) as Record<string, unknown>;
			scene = normalizeDeepReferenceResponse(raw);
			runId = stringAt(raw, 'runId', 'run_id') ?? nextRunId;
			receiptId = stringAt(raw, 'receiptId', 'receipt_id');
			receiptSeal = null;
			if (receiptId) {
				try {
					receiptSeal = await api.receipts.get(receiptId);
				} catch {
					receiptSeal = null;
				}
			}
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unable to retrieve the supporting memory.';
		} finally {
			loading = false;
		}
	}

	function useExample(example: string) {
		query = example;
		void runReplay();
	}

	function reset() {
		query = '';
		scene = null;
		error = null;
		runId = null;
		receiptId = null;
		receiptSeal = null;
		input?.focus();
	}

	function labelFor(evidence: NormalizedEvidence) {
		if (evidence.role === 'primary') return 'Primary evidence';
		if (evidence.role === 'contradicting') return 'Conflicting evidence';
		if (evidence.role === 'superseded') return 'Older, superseded evidence';
		return 'Supporting evidence';
	}

	onMount(() => input?.focus());

	// Living base coat — real store vitals drive the ambient field (never
	// decorative randomness). One cheap fetch; zeros render a calm field.
	let ambient = $state({ endangered: 0, fracture: 0, due: 0, count: 0 });
	onMount(async () => {
		try {
			const [s, rd] = await Promise.all([api.stats(), api.retentionDistribution()]);
			const total = Math.max(1, s.totalMemories);
			ambient = {
				endangered: Math.min(1, (rd.endangered?.length ?? 0) / total),
				fracture: 0,
				due: Math.min(1, (s.dueForReview ?? 0) / total),
				count: s.totalMemories
			};
		} catch {
			/* field stays calm — never invents vitals */
		}
	});
</script>

<svelte:head>
	<title>Memory Replay · Vestige</title>
</svelte:head>

<main class="replay-shell" style="position: relative">
	<AmbientField {...ambient} accent={[0.36, 0.94, 0.65]} opacity={0.5} />
	<PageHeader
		icon="reasoning"
		title="Memory Replay"
		subtitle="See the exact memories Vestige retrieved before it makes a recommendation."
		accent="recall"
	>
		<span class="live-pill"><span></span> Live retrieval proof</span>
	</PageHeader>

	<section class="hero-card">
		<div class="hero-copy">
			<p class="eyebrow">MAKE AI DECISIONS AUDITABLE</p>
			<h1>Ask a question. See the memory behind the answer.</h1>
			<p>Vestige retrieves real, local memory, evaluates conflicts, and gives you a receipt for the run.</p>
		</div>
		<form
			class="ask-form"
			onsubmit={(event) => {
				event.preventDefault();
				void runReplay();
			}}
		>
			<label for="memory-question">Your question</label>
			<div class="ask-row">
				<input
					bind:this={input}
					bind:value={query}
					id="memory-question"
					type="search"
					placeholder="Ask Vestige something it should remember…"
					autocomplete="off"
				/>
				<button type="submit" disabled={!query.trim() || loading}>
					<Icon name="reasoning" size={16} /> {loading ? 'Retrieving…' : 'Run replay'}
				</button>
			</div>
			<div class="examples" aria-label="Example questions">
				<span>Try:</span>
				{#each EXAMPLES as example}
					<button type="button" onclick={() => useExample(example)}>{example}</button>
				{/each}
			</div>
		</form>
	</section>

	{#if error}
		<section class="notice error">
			<Icon name="close" size={18} />
			<div><strong>Replay could not complete.</strong><br />{error}</div>
			<button type="button" onclick={() => void runReplay()}>Try again</button>
		</section>
	{:else if loading}
		<section class="loading-card" aria-live="polite">
			<div class="pulse"></div><div><strong>Retrieving from your memory</strong><p>Creating a trace and receipt for this run…</p></div>
		</section>
	{:else if scene}
		<section class="proof-strip" aria-label="Retrieval proof summary">
			<div><strong>{scene.evidence.length}</strong><span>memories retrieved</span></div>
			<div><strong>{scene.contradictions.length}</strong><span>conflicts flagged</span></div>
			<div><strong>{confidence}%</strong><span>confidence</span></div>
			<div class:ready={Boolean(receiptId)}><strong>{receiptId ? 'Receipt ready' : 'Trace recorded'}</strong><span>{receiptId ? 'proof is linked to this run' : 'receipt availability pending'}</span></div>
		</section>

		<div class="results-grid">
			<section class="decision-card">
				<div class="section-kicker"><span class="dot"></span> Recommendation</div>
				{#if scene.recommended?.answer_preview}
					<h2>{scene.recommended.answer_preview}</h2>
				{:else}
					<h2>No reliable recommendation was produced.</h2>
				{/if}
				<p class="honesty"><strong>What this proves:</strong> the memory IDs below were retrieved in this run. It does not claim an answer changed.</p>
				<div class="action-row">
					<a href={blackBoxUrl}>Open this run in Black Box <span>→</span></a>
					{#if receiptUrl}<a class="secondary" href={receiptUrl}>Open exact receipt <span>→</span></a>{/if}
					<button class="quiet" type="button" onclick={reset}>New question</button>
				</div>
				{#if receiptSeal}
					<div class="receipt-seal">
						<strong>Receipt seal</strong>
						<span>{receiptSeal.retrieved.length} retrieved · {receiptSeal.suppressed.length} suppressed · trust floor {receiptSeal.trust_floor}</span>
					</div>
				{/if}
				{#if runId}<code class="run-id">RUN {runId}</code>{/if}
			</section>

			<aside class="method-card">
				<p class="section-kicker">Why this is different</p>
				<ol>
					<li><span>1</span><div><strong>Retrieve</strong><small>Find the real memories relevant to your question.</small></div></li>
					<li><span>2</span><div><strong>Evaluate</strong><small>Expose conflicts and older superseded context.</small></div></li>
					<li><span>3</span><div><strong>Prove</strong><small>Keep a run receipt anyone can inspect.</small></div></li>
				</ol>
			</aside>
		</div>

		<section class="evidence-panel">
			<header>
				<div><p class="section-kicker">Proven: retrieved in this run</p><h2>The evidence Vestige actually used</h2></div>
				<span>{scene.evidence.length} memory{scene.evidence.length === 1 ? '' : 'ies'}</span>
			</header>
			{#if scene.evidence.length}
				<div class="evidence-list">
					{#each scene.evidence as evidence (evidence.id)}
						<article class:primary={evidence.role === 'primary'} class:conflict={evidence.role === 'contradicting'}>
							<div class="evidence-top"><span>{labelFor(evidence)}</span><b>{Math.round(evidence.trust * 100)}% trust</b></div>
							<p>{evidence.preview || 'Memory content is unavailable in this response.'}</p>
							<a href={`${base}/memories?memory=${encodeURIComponent(evidence.id)}`}><code>{evidence.id}</code></a>
						</article>
					{/each}
				</div>
			{:else}
				<div class="empty-evidence">No memory was retrieved for this question. That result is visible rather than hidden.</div>
			{/if}
		</section>

		{#if scene.contradictions.length || scene.superseded.length}
			<section class="attribution-panel">
				<p class="section-kicker">Attributed: likely influence</p>
				<h2>Context Vestige weighed, but did not treat as decisive</h2>
				{#each scene.contradictions as conflict}
					<p><strong>Conflict flagged:</strong> {conflict.summary}</p>
				{/each}
				{#each scene.superseded as older}
					<p><strong>Superseded:</strong> {older.preview || older.id} <code>{older.id}</code></p>
				{/each}
			</section>
		{/if}
	{:else}
		<section class="empty-state">
			<div class="empty-icon"><Icon name="memories" size={28} /></div>
			<div><h2>Turn an AI answer into evidence you can inspect.</h2><p>Run a replay to reveal retrieved memory, confidence, conflicts, and the exact run receipt.</p></div>
		</section>
	{/if}
</main>

<style>
	:global(body) { background: #071012; }
	.replay-shell { min-height: 100%; max-width: 1180px; margin: 0 auto; padding: 2rem clamp(1rem, 3vw, 2.5rem) 4rem; color: #e8f5f4; }
	.live-pill { display: inline-flex; align-items: center; gap: .5rem; color: #91bab6; font: 600 .72rem/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .08em; text-transform: uppercase; }
	.live-pill span, .dot { width: .5rem; height: .5rem; border-radius: 999px; background: #00e7c8; box-shadow: 0 0 12px #00e7c8; }
	.hero-card, .decision-card, .method-card, .evidence-panel, .attribution-panel, .empty-state, .loading-card, .notice { border: 1px solid rgba(139, 192, 184, .17); background: linear-gradient(135deg, rgba(14, 33, 35, .96), rgba(7, 18, 21, .94)); box-shadow: 0 18px 50px rgba(0,0,0,.2); border-radius: 1.15rem; }
	.hero-card { margin-top: 1.5rem; padding: clamp(1.3rem, 3vw, 2.6rem); display: grid; gap: 2rem; grid-template-columns: minmax(0, .85fr) minmax(380px, 1.15fr); background: radial-gradient(circle at 100% 0%, rgba(0, 231, 200, .12), transparent 42%), linear-gradient(135deg, #10282a, #071315); }
	.eyebrow, .section-kicker { margin: 0; color: #55dbc8; font: 700 .68rem/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .13em; text-transform: uppercase; }
	.hero-copy h1 { max-width: 16ch; margin: .7rem 0 .8rem; font-size: clamp(1.7rem, 3.2vw, 2.65rem); line-height: 1.05; letter-spacing: -.045em; color: #f2fffd; }
	.hero-copy > p:last-child { max-width: 48ch; margin: 0; color: #a9c4c1; line-height: 1.55; }
	.ask-form { align-self: end; } .ask-form label { display: block; margin-bottom: .6rem; color: #b8d1cf; font-size: .78rem; font-weight: 700; }
	.ask-row { display: flex; gap: .65rem; } .ask-row input { min-width: 0; flex: 1; border: 1px solid rgba(120, 178, 170, .3); border-radius: .75rem; background: rgba(0,0,0,.24); padding: .9rem 1rem; color: #f4fffd; outline: none; font-size: .92rem; } .ask-row input:focus { border-color: #4ce3cb; box-shadow: 0 0 0 3px rgba(0,231,200,.13); }
	button, .action-row a { cursor: pointer; } .ask-row button { border: 0; border-radius: .75rem; padding: .9rem 1.05rem; display: inline-flex; align-items: center; gap: .45rem; background: #00cbb0; color: #03201d; font-weight: 800; white-space: nowrap; } .ask-row button:disabled { opacity: .45; cursor: not-allowed; }
	.examples { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; margin-top: .8rem; color: #7d9c99; font-size: .72rem; } .examples button { border: 1px solid rgba(134, 184, 177, .22); border-radius: 99px; background: transparent; padding: .35rem .6rem; color: #b9d5d1; font-size: .72rem; } .examples button:hover { border-color: #54ddc9; color: #effffd; }
	.proof-strip { display: grid; grid-template-columns: repeat(4, 1fr); margin: 1rem 0; overflow: hidden; border: 1px solid rgba(139, 192, 184, .16); border-radius: .9rem; background: rgba(9, 24, 26, .8); } .proof-strip > div { padding: 1rem 1.15rem; border-right: 1px solid rgba(139, 192, 184, .15); } .proof-strip > div:last-child { border: 0; } .proof-strip strong { display: block; color: #eafffb; font-size: 1.1rem; } .proof-strip span { display: block; margin-top: .2rem; color: #82a39f; font-size: .7rem; } .proof-strip .ready strong { color: #5be6cf; }
	.results-grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(250px, .65fr); gap: 1rem; } .decision-card, .method-card, .evidence-panel, .attribution-panel { padding: clamp(1.15rem, 2vw, 1.6rem); } .section-kicker { display: flex; align-items: center; gap: .5rem; } .decision-card h2 { max-width: 30ch; margin: 1rem 0; color: #f4fffd; font-size: clamp(1.3rem, 2.4vw, 1.85rem); line-height: 1.25; letter-spacing: -.025em; } .honesty { margin: 0; border-left: 2px solid #5ce3d0; padding-left: .85rem; color: #a8c6c1; font-size: .83rem; line-height: 1.5; }
	.action-row { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: 1.3rem; } .action-row a, .quiet { border: 1px solid rgba(102, 226, 205, .38); border-radius: .6rem; background: rgba(0, 225, 195, .11); padding: .6rem .75rem; color: #75f0dc; font-size: .78rem; font-weight: 700; text-decoration: none; } .action-row .secondary, .quiet { border-color: rgba(158, 190, 186, .25); background: transparent; color: #aec9c5; } .receipt-seal { margin-top: 1rem; display: grid; gap: .2rem; padding: .7rem .8rem; border: 1px solid rgba(34, 199, 222, .28); border-radius: .6rem; background: rgba(2, 12, 16, .7); color: #9fd9d0; font-size: .74rem; } .receipt-seal strong { color: #7ff3e6; font-size: .62rem; letter-spacing: .12em; text-transform: uppercase; } .run-id { display: block; margin-top: 1rem; color: #60817c; font-size: .66rem; overflow-wrap: anywhere; }
	.method-card { background: linear-gradient(180deg, rgba(10, 29, 31, .96), rgba(7, 17, 19, .96)); } .method-card ol { margin: 1.2rem 0 0; padding: 0; list-style: none; } .method-card li { display: flex; gap: .7rem; margin: 1rem 0; } .method-card li > span { display: grid; place-items: center; flex: 0 0 1.55rem; height: 1.55rem; border-radius: 50%; background: rgba(0,226,196,.12); color: #60e6d2; font-size: .72rem; font-weight: 800; } .method-card strong { display: block; font-size: .85rem; } .method-card small { display: block; margin-top: .18rem; color: #87a6a2; line-height: 1.35; }
	.evidence-panel, .attribution-panel { margin-top: 1rem; } .evidence-panel header { display: flex; align-items: start; justify-content: space-between; gap: 1rem; margin-bottom: 1rem; } .evidence-panel h2, .attribution-panel h2 { margin: .45rem 0 0; color: #effdfa; font-size: 1.13rem; } .evidence-panel header > span { border-radius: 99px; background: rgba(103, 157, 150, .12); padding: .35rem .55rem; color: #94b9b4; font-size: .7rem; white-space: nowrap; }
	.evidence-list { display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit, minmax(min(100%, 285px), 1fr)); } .evidence-list article { min-width: 0; border: 1px solid rgba(144, 191, 184, .17); border-radius: .75rem; background: rgba(1, 12, 13, .34); padding: 1rem; } .evidence-list article.primary { border-color: rgba(60, 225, 198, .46); box-shadow: inset 3px 0 #2ce0c4; } .evidence-list article.conflict { border-color: rgba(255, 115, 104, .42); } .evidence-top { display: flex; justify-content: space-between; gap: .5rem; color: #5ce2cf; font-size: .68rem; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; } .evidence-list article.conflict .evidence-top { color: #ff9c91; } .evidence-top b { color: #a3c3bf; font-weight: 600; } .evidence-list p { min-height: 3.2em; margin: .75rem 0; color: #d6e8e5; font-size: .86rem; line-height: 1.5; } .evidence-list code, .attribution-panel code { color: #6e9690; font-size: .65rem; overflow-wrap: anywhere; }
	.attribution-panel { border-color: rgba(234, 174, 80, .27); background: linear-gradient(135deg, rgba(49, 35, 14, .44), rgba(16, 23, 21, .94)); } .attribution-panel .section-kicker { color: #e9bc6e; } .attribution-panel p:not(.section-kicker) { margin: .8rem 0 0; color: #c4c8b5; font-size: .85rem; line-height: 1.5; }
	.empty-state, .loading-card, .notice { display: flex; align-items: center; gap: 1rem; margin-top: 1rem; padding: 1.35rem; } .empty-icon, .pulse { display: grid; place-items: center; flex: 0 0 3.2rem; height: 3.2rem; border-radius: .8rem; background: rgba(0,226,196,.1); color: #63e6d2; } .empty-state h2, .loading-card strong { margin: 0; color: #effdfa; font-size: 1rem; } .empty-state p, .loading-card p { margin: .35rem 0 0; color: #91b0ac; font-size: .83rem; line-height: 1.45; } .pulse { position: relative; } .pulse::after { content: ''; width: .72rem; height: .72rem; border-radius: 50%; background: #58e8d2; animation: pulse 1s infinite alternate; } .notice.error { border-color: rgba(255, 111, 100, .4); color: #f6b0a9; } .notice button { margin-left: auto; border: 1px solid currentColor; border-radius: .5rem; background: transparent; padding: .45rem .65rem; color: inherit; }
	@keyframes pulse { to { opacity: .35; transform: scale(.55); } }
	@media (max-width: 800px) { .hero-card, .results-grid { grid-template-columns: 1fr; } .proof-strip { grid-template-columns: 1fr 1fr; } .proof-strip > div:nth-child(2) { border-right: 0; } .proof-strip > div:nth-child(-n+2) { border-bottom: 1px solid rgba(139, 192, 184, .15); } }
	@media (max-width: 520px) { .replay-shell { padding-top: 1rem; } .ask-row { flex-direction: column; } .ask-row button { justify-content: center; } .proof-strip { grid-template-columns: 1fr; } .proof-strip > div { border-right: 0; border-bottom: 1px solid rgba(139, 192, 184, .15); } .proof-strip > div:last-child { border-bottom: 0; } }
</style>
