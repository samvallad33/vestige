<script lang="ts">
	/**
	 * /benchmark — the cross-model memory benchmark, rendered ON TOP of the real
	 * living Observatory WebGPU field. The field runs the `salience-rescue` demo:
	 * the causal-recovery axon reaching backward to rescue the truth, which is
	 * literally the thesis of this benchmark (memory reaches back to the cause a
	 * similarity search misses). The DOM overlay carries the data: the money-shot
	 * transcript, the four-model leaderboard, the reproduce command. The engine
	 * does the moving / glowing / alive part; this route is the instrument overlay.
	 */
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import ObservatoryStage from '$lib/observatory/ObservatoryStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';

	let engineRef: ObservatoryEngine | null = null;
	let mounted = $state(false);

	onMount(() => {
		mounted = true;
	});

	function handleReady(engine: ObservatoryEngine) {
		engineRef = engine;
		engine.demoClock.reset();
	}

	// The verified cross-model result (published from each run's SUMMARY.txt).
	type Arm = { v: number; cls: 'void' | 'low' | 'hi' | 'perfect' };
	const rows: { model: string; note: string; lab: string; anarchy: Arm; rag: Arm; vestige: Arm }[] = [
		{
			model: 'Kimi K2.7', note: 'open-weight coder', lab: 'Moonshot',
			anarchy: { v: 0, cls: 'void' }, rag: { v: 1, cls: 'low' }, vestige: { v: 5, cls: 'perfect' }
		},
		{
			model: 'GLM 5.2', note: 'open-weight', lab: 'Zhipu',
			anarchy: { v: 0, cls: 'void' }, rag: { v: 0, cls: 'void' }, vestige: { v: 5, cls: 'perfect' }
		},
		{
			model: 'DeepSeek V4', note: 'open-weight · flash', lab: 'DeepSeek',
			anarchy: { v: 0, cls: 'void' }, rag: { v: 1, cls: 'low' }, vestige: { v: 3, cls: 'hi' }
		},
		{
			model: 'Kimi K3', note: '2.8T · 1M context · newest', lab: 'Moonshot',
			anarchy: { v: 0, cls: 'void' }, rag: { v: 3, cls: 'low' }, vestige: { v: 5, cls: 'perfect' }
		}
	];
	const agg = { anarchy: 0, rag: 5, vestige: 18, total: 20 };

	const REPRO =
		"MASTER_SEED=2026 ARMS='anarchy rag sync' N_TRIALS=5 \\\n  bash run-experiment.sh";
	let copied = $state(false);
	async function copyRepro() {
		try {
			await navigator.clipboard.writeText(
				"MASTER_SEED=2026 ARMS='anarchy rag sync' N_TRIALS=5 bash run-experiment.sh"
			);
			copied = true;
			setTimeout(() => (copied = false), 1600);
		} catch {
			copied = false;
		}
	}
</script>

<svelte:head>
	<title>Prove or Void · Vestige memory benchmark</title>
</svelte:head>

<!-- The living WebGPU field: real Observatory engine, salience-rescue demo.
     chrome="none" + showSwitcher=false = pure field, no built-in HUD, so our
     benchmark overlay owns the foreground. live=true keeps it breathing. -->
<div class="bench-field">
	{#if mounted}
		<!-- No `live`: the WebGPU render loop still runs and the field breathes, but
		     `live` also renders the stage's own "Strongest memories" recall marquee
		     (an in-DOM HUD) which would collide with our benchmark overlay. Dropping
		     it leaves a pure, animated field for our foreground to own. -->
		<ObservatoryStage
			demo="salience-rescue"
			seed="vestige-benchmark-prove-or-void"
			chrome="none"
			showSwitcher={false}
			onready={handleReady}
		/>
	{/if}
</div>
<div class="bench-veil" aria-hidden="true"></div>

<main class="bench">
	<!-- ── HERO ── -->
	<section class="hero">
		<div class="eyebrow">Cross-model memory benchmark · July 2026</div>
		<h1>The model didn't need to be <span class="void-word">smarter</span>.<br />It needed to <span class="know-word">remember</span>.</h1>
		<p class="lede">
			Four frontier models fixed the same production bug. Without memory, none could ship a
			safe fix. With causal memory, they nearly always could. Same models. Same budget. The
			only variable was which past they could reach.
		</p>
		<div class="hero-stats">
			<div class="hstat void"><span class="n">0<i>/20</i></span><span class="l">No memory</span></div>
			<div class="hstat low"><span class="n">5<i>/20</i></span><span class="l">Similarity · RAG</span></div>
			<div class="hstat hi"><span class="n">18<i>/20</i></span><span class="l">Vestige · backfill</span></div>
		</div>
	</section>

	<!-- ── MONEY SHOT ── -->
	<section class="money">
		<div class="sec-eyebrow">The money shot · Kimi K3 · 2.8T params · 1M context</div>
		<h2>The largest AI model ever built chose a production key because it was first in the list.</h2>
		<div class="diff">
			<div class="diff-bar">
				<span class="d r"></span><span class="d y"></span><span class="d g"></span>
				<span class="path">trial 1 · correct key = k_yildun · absent from the repo</span>
				<span class="model">kimi-k3 · max reasoning</span>
			</div>
			<div class="diff-cols">
				<div class="col no-mem">
					<div class="col-head"><span class="k">No memory</span><span class="verdict">green · voids prod</span></div>
					<pre>&gt; reasoning
The failure is "no active signing key selected."
Both configs were blanked by the refactor.

The keyring's first entry is <b class="wrong">k_pavo</b>, and the config
comment "entry the platform shipped with" hints at
the first entry the platform shipped with.

<b class="wrong">k_pavo</b> as the first / shipped entry is the most
defensible single choice.

&gt; write · test passes · <span class="fail">prod replay: VOIDS 5 tokens</span></pre>
				</div>
				<div class="col with-mem">
					<div class="col-head"><span class="k">Vestige backfill</span><span class="verdict">fixed · prod-safe</span></div>
					<pre>&gt; vestige_backfill(failure)
The backfill surfaced a memory: a Q3 rotation
runbook. The live signer was migrated to <b class="right">k_yildun</b>
and the old material aged out.

So the active signing key should be <b class="right">k_yildun</b>.
Both services must select it.

&gt; write · test passes · <span class="ok">prod replay: verified</span></pre>
				</div>
			</div>
			<div class="diff-foot">
				<span class="axon">←</span> Backfill reaches <b>backward through time</b> to the cause. Similarity searches the goal line; the turnover happened in midfield, three weeks ago.
			</div>
		</div>
	</section>

	<!-- ── LEADERBOARD ── -->
	<section class="board-sec">
		<div class="sec-eyebrow">The leaderboard</div>
		<h2>Every model. Same seed. Shipped-correct <span class="dim">and</span> production-safe.</h2>
		<div class="board-scroll">
			<table class="board">
				<thead>
					<tr>
						<th>Model</th>
						<th>Lab</th>
						<th class="a-void">No memory</th>
						<th class="a-rag">RAG · similarity</th>
						<th class="a-vest">Vestige · backfill</th>
					</tr>
				</thead>
				<tbody>
					{#each rows as r (r.model)}
						<tr>
							<td><div class="mc"><span class="mn">{r.model}</span><span class="ml">{r.note}</span></div></td>
							<td class="dim">{r.lab}</td>
							<td><span class="score s-{r.anarchy.cls}">{r.anarchy.v}<i>/5</i></span></td>
							<td><span class="score s-{r.rag.cls}">{r.rag.v}<i>/5</i></span></td>
							<td><span class="score s-{r.vestige.cls}">{r.vestige.v}<i>/5</i></span></td>
						</tr>
					{/each}
					<tr class="agg">
						<td><div class="mc"><span class="mn">All four</span><span class="ml">20 trials each arm</span></div></td>
						<td class="dim">4 labs</td>
						<td><span class="score s-void">{agg.anarchy}<i>/20</i></span></td>
						<td><span class="score s-low">{agg.rag}<i>/20</i></span></td>
						<td><span class="score s-hi">{agg.vestige}<i>/20</i></span></td>
					</tr>
				</tbody>
			</table>
		</div>
		<div class="board-note">
			<span>Clean separation on Kimi K3: anarchy 0/5, Vestige 5/5 → <b class="p">p = (1/50)<sup>5</sup> = 3.2×10<sup>−9</sup></b></span>
			<span class="dim">GPT-5.6 Sol and Fable 5 rows land next.</span>
		</div>
	</section>

	<!-- ── REPRODUCE ── -->
	<section class="repro-sec">
		<div class="sec-eyebrow">Run it yourself</div>
		<h2>Every number here is one command away from being yours.</h2>
		<p class="lede">
			Deterministic. Fixed seed. The same fifty keys every model faces. The correct key is
			provably absent from the repo, so nobody can claim the agents were tuned to it.
		</p>
		<div class="repro">
			<div class="repro-bar"><span class="g"></span> silent-rotation · N=5 · 3 arms
				<button class="copy" onclick={copyRepro}>{copied ? 'copied ✓' : 'copy'}</button>
			</div>
			<pre>{REPRO}</pre>
		</div>
	</section>

	<!-- ── CTA ── -->
	<footer class="cta">
		<div class="sec-eyebrow">Vestige · local-first memory for AI agents</div>
		<h2>Give your agent the one thing a bigger context window can't buy.</h2>
		<a class="btn" href="https://github.com/samvallad33/vestige" target="_blank" rel="noreferrer">npm i -g vestige-mcp-server →</a>
		<div class="foot-meta">
			<span>One 25MB binary</span><span>No cloud</span><span>Your data never leaves your machine</span><span>AGPL-3.0</span>
		</div>
	</footer>
</main>

<style>
	.bench-field {
		position: fixed;
		inset: 0;
		z-index: 0;
	}
	/* readability veil over the living field so the data reads; the field still
	   glows through, especially at the top and edges. */
	.bench-veil {
		position: fixed;
		inset: 0;
		z-index: 1;
		pointer-events: none;
		background:
			radial-gradient(130% 90% at 50% -10%, transparent 38%, rgba(2, 3, 7, 0.62) 100%),
			linear-gradient(180deg, rgba(2, 3, 7, 0.28) 0%, rgba(2, 3, 7, 0.5) 24%, rgba(2, 3, 7, 0.6) 60%, rgba(2, 3, 7, 0.82) 100%);
	}

	.bench {
		position: relative;
		z-index: 2;
		max-width: 1100px;
		margin: 0 auto;
		padding: 0 clamp(20px, 5vw, 56px) 120px;
		color: #dcefff;
		font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
		line-height: 1.6;
	}

	.eyebrow,
	.sec-eyebrow {
		font-size: 12px;
		letter-spacing: 0.28em;
		text-transform: uppercase;
		color: #1bd6ff;
		font-weight: 500;
	}

	section,
	.cta {
		padding: clamp(64px, 11vh, 116px) 0 0;
	}
	.hero {
		min-height: 92svh;
		display: flex;
		flex-direction: column;
		justify-content: center;
		gap: 26px;
		padding-top: 0;
	}
	h1 {
		font-size: clamp(2.3rem, 6vw, 4.6rem);
		font-weight: 700;
		line-height: 1.02;
		letter-spacing: -0.02em;
		margin: 0;
		text-wrap: balance;
		text-shadow: 0 4px 40px rgba(2, 3, 7, 0.7);
	}
	.void-word { color: #ff3b30; }
	.know-word { color: #1bd6ff; text-shadow: 0 0 34px rgba(27, 214, 255, 0.5); }
	h2 {
		font-size: clamp(1.5rem, 3.3vw, 2.35rem);
		font-weight: 700;
		line-height: 1.08;
		letter-spacing: -0.01em;
		margin: 16px 0 0;
		max-width: 24ch;
		text-wrap: balance;
	}
	.lede {
		font-size: clamp(1.02rem, 2vw, 1.3rem);
		max-width: 44ch;
		line-height: 1.5;
		color: #cfe6f0;
		margin: 6px 0 0;
	}
	.dim { color: #7fa3b3; }

	.hero-stats {
		display: flex;
		gap: clamp(22px, 5vw, 52px);
		flex-wrap: wrap;
		margin-top: 12px;
	}
	.hstat { display: flex; flex-direction: column; gap: 3px; }
	.hstat .n {
		font-size: clamp(2rem, 4.4vw, 3rem);
		font-weight: 700;
		line-height: 1;
		font-variant-numeric: tabular-nums;
	}
	.hstat .n i { font-style: normal; font-size: 0.42em; color: #47606c; }
	.hstat .l {
		font-size: 11.5px;
		letter-spacing: 0.13em;
		text-transform: uppercase;
		color: #7fa3b3;
	}
	.hstat.void .n { color: #ff3b30; }
	.hstat.low .n { color: #ffb000; }
	.hstat.hi .n { color: #1bd6ff; text-shadow: 0 0 28px rgba(27, 214, 255, 0.4); }

	/* ── money shot ── */
	.diff {
		margin-top: 30px;
		border: 1px solid rgba(27, 214, 255, 0.16);
		border-radius: 16px;
		background: linear-gradient(180deg, rgba(8, 17, 21, 0.86), rgba(3, 6, 8, 0.9));
		box-shadow: 0 40px 120px -46px rgba(0, 0, 0, 0.92);
		overflow: hidden;
		-webkit-backdrop-filter: blur(3px);
		backdrop-filter: blur(3px);
	}
	.diff-bar {
		display: flex;
		align-items: center;
		gap: 9px;
		padding: 13px 18px;
		border-bottom: 1px solid rgba(120, 190, 210, 0.12);
		font-size: 12.5px;
		color: #7fa3b3;
	}
	.diff-bar .d { width: 10px; height: 10px; border-radius: 50%; }
	.diff-bar .d.r { background: #ff3b30; } .diff-bar .d.y { background: #ffb000; } .diff-bar .d.g { background: #29f2a9; }
	.diff-bar .path { margin-left: 6px; color: #47606c; }
	.diff-bar .model { margin-left: auto; color: #1bd6ff; }
	.diff-cols { display: grid; grid-template-columns: 1fr 1fr; }
	.col { padding: 22px clamp(18px, 2.4vw, 30px) 26px; }
	.col + .col { border-left: 1px solid rgba(120, 190, 210, 0.12); }
	.col-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 14px; }
	.col-head .k { font-size: 11.5px; letter-spacing: 0.16em; text-transform: uppercase; }
	.no-mem .col-head .k { color: #ff3b30; }
	.with-mem .col-head .k { color: #ff2df7; }
	.col-head .verdict { margin-left: auto; font-size: 11.5px; padding: 3px 10px; border-radius: 999px; }
	.no-mem .verdict { color: #ff3b30; border: 1px solid rgba(255, 59, 48, 0.4); }
	.with-mem .verdict { color: #29f2a9; border: 1px solid rgba(41, 242, 169, 0.4); }
	.col pre {
		margin: 0;
		font-size: 13px;
		line-height: 1.6;
		white-space: pre-wrap;
		color: #7fa3b3;
		font-family: inherit;
	}
	.col pre b.wrong { color: #ff3b30; font-weight: 700; }
	.col pre b.right { color: #00f5d4; font-weight: 700; }
	.col pre .fail { color: #ff3b30; }
	.col pre .ok { color: #29f2a9; }
	.diff-foot {
		padding: 16px clamp(18px, 2.4vw, 30px);
		border-top: 1px solid rgba(120, 190, 210, 0.12);
		font-size: 13px;
		color: #7fa3b3;
	}
	.diff-foot b { color: #dcefff; }
	.diff-foot .axon { color: #ff2df7; font-weight: 700; }

	/* ── leaderboard ── */
	.board-scroll {
		margin-top: 26px;
		overflow-x: auto;
		border: 1px solid rgba(120, 190, 210, 0.12);
		border-radius: 14px;
		background: rgba(4, 8, 10, 0.55);
		-webkit-backdrop-filter: blur(3px);
		backdrop-filter: blur(3px);
	}
	table.board { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: 620px; }
	table.board th, table.board td { text-align: left; padding: 15px 18px; }
	table.board thead th {
		font-size: 11px;
		letter-spacing: 0.13em;
		text-transform: uppercase;
		color: #7fa3b3;
		font-weight: 500;
		border-bottom: 1px solid rgba(120, 190, 210, 0.12);
	}
	thead th.a-void { color: #ff6a60; } thead th.a-rag { color: #ffb000; } thead th.a-vest { color: #1bd6ff; }
	table.board tbody tr { border-bottom: 1px solid rgba(120, 190, 210, 0.09); }
	table.board tbody tr:last-child { border-bottom: none; }
	.mc { display: flex; flex-direction: column; gap: 2px; }
	.mc .mn { font-weight: 700; font-size: 15px; color: #dcefff; }
	.mc .ml { font-size: 11px; color: #47606c; }
	.score { font-weight: 700; font-size: 1.3rem; }
	.score i { font-style: normal; color: #47606c; font-weight: 400; font-size: 0.72em; }
	.score.s-void { color: #ff3b30; }
	.score.s-low { color: #ffb000; }
	.score.s-hi { color: #1bd6ff; }
	.score.s-perfect { color: #e9ffb7; text-shadow: 0 0 18px rgba(233, 255, 183, 0.35); }
	tr.agg td { background: rgba(27, 214, 255, 0.05); }
	tr.agg .mn { color: #1bd6ff; }
	.board-note {
		margin-top: 15px;
		display: flex;
		gap: 22px;
		flex-wrap: wrap;
		font-size: 13px;
		color: #7fa3b3;
	}
	.board-note .p { color: #00f5d4; font-weight: 600; }

	/* ── reproduce ── */
	.repro {
		margin-top: 26px;
		border: 1px solid rgba(27, 214, 255, 0.16);
		border-radius: 14px;
		overflow: hidden;
		background: linear-gradient(180deg, rgba(8, 17, 21, 0.9), rgba(3, 6, 8, 0.92));
	}
	.repro-bar {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 11px 16px;
		border-bottom: 1px solid rgba(120, 190, 210, 0.12);
		font-size: 12px;
		color: #7fa3b3;
	}
	.repro-bar .g { width: 9px; height: 9px; border-radius: 50%; background: #29f2a9; box-shadow: 0 0 10px #29f2a9; }
	.repro-bar .copy {
		margin-left: auto;
		font-family: inherit;
		font-size: 11.5px;
		color: #00f5d4;
		background: transparent;
		border: 1px solid rgba(0, 245, 212, 0.3);
		border-radius: 999px;
		padding: 3px 12px;
		cursor: pointer;
		transition: background 0.15s ease;
	}
	.repro-bar .copy:hover { background: rgba(0, 245, 212, 0.1); }
	.repro pre {
		margin: 0;
		padding: 20px 18px;
		font-size: 13.5px;
		line-height: 1.7;
		color: #cfe6f0;
		overflow-x: auto;
		font-family: inherit;
	}

	/* ── cta ── */
	.cta { display: flex; flex-direction: column; gap: 20px; align-items: flex-start; border-top: 1px solid rgba(120, 190, 210, 0.12); margin-top: 90px; }
	.btn {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		font-weight: 700;
		font-size: 15px;
		padding: 14px 24px;
		border-radius: 999px;
		background: linear-gradient(120deg, #1bd6ff, #00f5d4);
		color: #041014;
		text-decoration: none;
		box-shadow: 0 0 44px -6px rgba(27, 214, 255, 0.6);
		transition: transform 0.15s ease, box-shadow 0.15s ease;
	}
	.btn:hover { transform: translateY(-2px); box-shadow: 0 0 60px -4px rgba(27, 214, 255, 0.85); }
	.foot-meta { margin-top: 22px; display: flex; gap: 20px; flex-wrap: wrap; font-size: 12.5px; color: #47606c; }

	@media (max-width: 820px) {
		.diff-cols { grid-template-columns: 1fr; }
		.col + .col { border-left: none; border-top: 1px solid rgba(120, 190, 210, 0.12); }
	}
	@media (prefers-reduced-motion: reduce) {
		.btn { transition: none; }
	}
</style>
