<script lang="ts">
	/**
	 * Guard — what the gate stopped, and why.
	 *
	 * Every light on the river is a real Operator receipt (GET /api/stops, the
	 * vestige_stops reader): a STOP in crimson, an action a permit let through
	 * in green, at the minute it happened, one lane per agent that tried it.
	 * Each receipt names the receipt before it (prev), so the record is a
	 * hash chain; picking one draws its links to its neighbours and reads it:
	 * the rule, what was attempted (as the gate saw it, after normalising),
	 * the permit, the gate version.
	 */
	import '$lib/trace/trace-organ.css';
	import { alive } from '$lib/trace/alive';
	import { onMount } from 'svelte';
	import OrganStage from '$lib/trace/OrganStage.svelte';
	import type { StageWords } from '$lib/trace/TraceStage.svelte';
	import { api } from '$lib/stores/api';
	import { typeColor } from '$lib/trace/trace-pass';
	import { headline, type TraceLink, type TraceMemory } from '$lib/trace/trace-data';

	interface Stop {
		digest?: string;
		prev?: string;
		decision?: string;
		rule?: string;
		source?: string;
		tool?: string;
		session?: string;
		cwd?: string;
		action_preview?: string;
		normalized?: string;
		transforms?: string[];
		detail?: string;
		gate_version?: string;
		ts_utc?: string;
		via?: string;
		commitment?: string;
		chain_break?: boolean;
		proof_frame?: ProofFrame;
		permit?: { status?: string; by?: string; granted?: number; expires?: number; single_use?: boolean; [k: string]: unknown };
	}

	interface ProofFrame {
		stop?: boolean;
		kind?: string;
		decision?: string;
		rule?: string;
		proof_handle?: string | null;
		receipt_digest?: string | null;
		tool_process_created?: boolean;
		chain_break?: boolean;
		local?: boolean;
		export_span?: boolean;
	}

	const WINDOWS = [
		{ key: '24h', label: '24 hours' },
		{ key: '7d', label: '7 days' },
		{ key: '30d', label: '30 days' }
	] as const;
	type WindowKey = (typeof WINDOWS)[number]['key'];
	/** Lanes: the daily agents keep fixed ones; any other source gets its own below them. */
	const SOURCE_LANE: Record<string, number> = { claude: 0.22, codex: 0.0, zcode: -0.22 };
	function laneOf(source: string | undefined): number {
		const s = source ?? '';
		if (s in SOURCE_LANE) return SOURCE_LANE[s];
		let h = 0;
		for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
		return -0.42 - (h % 6) * 0.08;
	}

	let windowKey = $state<WindowKey>('7d');
	/** 'stops': what was stopped or let through by a permit; 'all': every action the gate saw. */
	let scope = $state<'stops' | 'all'>('stops');
	let sawTotal = $state(0);
	const RIVER_MAX = 1100;
	function kindOf(decision?: string): string {
		switch (decision) {
			case 'STOP':
				return 'stop';
			case 'ALLOW':
			case 'PERMIT_GRANTED':
				return 'permit';
			case 'PASS':
				return 'pass';
			case 'GATE_ERROR':
				return 'correction';
			default:
				return decision?.startsWith('SHADOW') ? 'shadow' : 'pass';
		}
	}
	let stops = $state<Stop[]>([]);
	let summary = $state('');
	let loading = $state(true);
	let error = $state<string | null>(null);
	let ruleFilter = $state<string | null>(null);
	let pickedId = $state<string | null>(null);
	let titleEl = $state<HTMLElement | null>(null);

	const idOf = (s: Stop, i: number) => s.digest ?? `stop-${i}`;

	const shown = $derived(ruleFilter ? stops.filter((s) => (s.rule ?? s.commitment) === ruleFilter) : stops);
	const river = $derived(
		shown
			.map<TraceMemory | null>((s, i) => {
				if (!s.ts_utc) return null;
				const kind = kindOf(s.decision);
				return {
					id: idOf(s, i),
					type: kind,
					content: s.normalized || s.action_preview || s.detail || '',
					createdAt: s.ts_utc,
					retention: kind === 'stop' ? 1 : kind === 'pass' ? 0.3 : 0.75,
					tags: [],
					// Stops are held on one shield, first; what went through
					// stands on one shield per agent after it.
					lane: kind === 'stop' ? 1 : laneOf(s.source),
					laneLabel: kind === 'stop' ? 'stopped' : s.source || 'unknown agent'
				};
			})
			.filter((m): m is TraceMemory => m !== null)
			.sort((a, b) => Date.parse(a.createdAt) - Date.parse(b.createdAt))
			.slice(-RIVER_MAX)
	);
	const byId = $derived(new Map(shown.map((s, i) => [idOf(s, i), s])));
	const byDigest = $derived(new Map(stops.filter((s) => s.digest).map((s) => [s.digest!, s])));
	const picked = $derived(pickedId ? (byId.get(pickedId) ?? null) : null);
	const frame = $derived(picked ? proofFrame(picked) : null);

	/** The picked receipt's chain links: the one it names, and the one naming it. */
	const links = $derived.by<TraceLink[]>(() => {
		if (!picked?.digest) return [];
		const out: TraceLink[] = [];
		if (picked.prev && byId.has(picked.prev)) out.push({ source: picked.prev, target: picked.digest, type: 'closes' });
		const next = shown.find((s) => s.prev === picked.digest);
		if (next?.digest) out.push({ source: picked.digest, target: next.digest, type: 'closes' });
		return out;
	});

	const ruleCounts = $derived.by(() => {
		const c = new Map<string, { stop: number; allow: number }>();
		for (const s of stops) {
			if (kindOf(s.decision) === 'pass') continue;
			const r = s.rule ?? s.commitment ?? 'unnamed';
			const e = c.get(r) ?? { stop: 0, allow: 0 };
			if (s.decision === 'ALLOW') e.allow++;
			else e.stop++;
			c.set(r, e);
		}
		return [...c.entries()].sort((a, b) => b[1].stop + b[1].allow - (a[1].stop + a[1].allow));
	});
	const sourceCounts = $derived.by(() => {
		const c = new Map<string, number>();
		for (const s of stops) c.set(s.source ?? 'unknown', (c.get(s.source ?? 'unknown') ?? 0) + 1);
		return [...c.entries()].sort((a, b) => b[1] - a[1]);
	});
	const stopCount = $derived(stops.filter((s) => s.decision === 'STOP').length);
	const permitCount = $derived(stops.filter((s) => kindOf(s.decision) === 'permit').length);
	const shadowCount = $derived(stops.filter((s) => kindOf(s.decision) === 'shadow').length);

	async function load() {
		loading = true;
		error = null;
		pickedId = null;
		try {
			if (scope === 'all') {
				const res = await api.stops({ since: windowKey, decision: 'all', limit: 2000 });
				stops = (res.items ?? []) as Stop[];
				sawTotal = stops.length;
				summary = res.summary ?? '';
			} else {
				const res = await api.stops({ since: windowKey, decision: 'STOP', limit: 2000 });
				const permitted = await api.stops({ since: windowKey, decision: 'ALLOW', limit: 2000 }).catch(() => null);
				stops = [...((res.items ?? []) as Stop[]), ...(((permitted?.items ?? []) as Stop[]).filter((s) => s.via === 'permit'))];
				summary = res.summary ?? '';
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			stops = [];
		} finally {
			loading = false;
		}
	}

	function describe(id: string): StageWords | null {
		const s = byId.get(id);
		if (!s) return null;
		return {
			kicker: `${verb(s.decision)}${s.rule ? ` · ${s.rule}` : ''} · ${s.source ?? ''} · ${when(s.ts_utc)}`,
			text: s.normalized || s.action_preview || s.detail || '',
			tone: kindOf(s.decision)
		};
	}

	const labelIds = $derived.by(() => {
		if (pickedId) return [pickedId, ...links.map((l) => (l.source === pickedId ? l.target : l.source))];
		const newest = river.at(-1)?.id;
		return newest ? [newest] : [];
	});

	function proofFrame(stop: Stop): ProofFrame {
		if (stop.proof_frame) return stop.proof_frame;
		if (stop.decision !== 'STOP') return { stop: false, kind: 'not-stop', decision: stop.decision, local: true, export_span: false };
		const digest = stop.digest && /^[0-9a-f]{64}$/i.test(stop.digest) ? stop.digest : '';
		return {
			stop: true,
			kind: 'stop',
			rule: stop.commitment || stop.rule || '',
			proof_handle: digest ? `op-receipt-${digest.slice(0, 16)}` : null,
			receipt_digest: digest || null,
			tool_process_created: false,
			chain_break: Boolean(stop.chain_break),
			local: true,
			export_span: false
		};
	}

	function verb(decision?: string): string {
		const k = kindOf(decision);
		return k === 'stop' ? 'stopped' : k === 'permit' ? 'permitted' : k === 'shadow' ? 'would have stopped' : k === 'pass' ? 'passed' : 'gate error';
	}

	function setScope(next: 'stops' | 'all') {
		if (next === scope) return;
		scope = next;
		ruleFilter = null;
		void load();
	}

	function setWindow(k: WindowKey) {
		if (k === windowKey) return;
		windowKey = k;
		void load();
	}

	const rgb = (t: string) => {
		const [r, g, b] = typeColor(t);
		return `rgb(${Math.round(r * 255)} ${Math.round(g * 255)} ${Math.round(b * 255)})`;
	};
	function when(iso?: string): string {
		if (!iso) return '';
		const d = new Date(iso);
		if (!Number.isFinite(d.getTime())) return '';
		return new Date().toDateString() === d.toDateString()
			? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
			: d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
	}
	const short = (h?: string) => (h ? `${h.slice(0, 12)}…` : '—');

	onMount(() => void load());
</script>

<svelte:head>
	<title>Guard · Vestige</title>
</svelte:head>

<OrganStage
	memories={river}
	now={Date.now()}
	{links}
	selectedId={pickedId}
	origin={() => {
		const r = titleEl?.getBoundingClientRect();
		return r ? { x: ((r.left + 160) / innerWidth) * 2 - 1, y: 1 - ((r.top + r.height / 2) / innerHeight) * 2 } : undefined;
	}}
	{describe}
	{labelIds}
	onpick={(id) => (pickedId = id)}
	story="guard"
	seed="vestige-guard-v2"
	enter={false}
/>

<header class="tr-ask" class:tr-quiet={!!pickedId}>
	<p class="tr-eyebrow" use:alive={{ text: 'Guard', glow: true }}></p>
	<!-- svelte-ignore a11y_missing_content (alive renders the text and its aria-label) -->
	<h1 class="tr-title" bind:this={titleEl} use:alive={{ text: 'What the gate stopped, and why.', glow: true }}></h1>
	<p class="tr-hint">
		{#each WINDOWS as w (w.key)}
			<button type="button" class="tr-chip" aria-pressed={windowKey === w.key} onclick={() => setWindow(w.key)}>{w.label}</button>
		{/each}
		<span class="tr-hint-text">·</span>
		<button type="button" class="tr-chip" aria-pressed={scope === 'stops'} onclick={() => setScope('stops')}>stops</button>
		<button type="button" class="tr-chip" aria-pressed={scope === 'all'} onclick={() => setScope('all')}>everything it saw</button>
		<span class="tr-hint-text">every stop on one shield · what went through, one shield per agent</span>
	</p>
	{#if loading}
		<p class="tr-line">Reading the gate's receipts…</p>
	{:else if error}
		<p class="tr-line tr-err">{error}</p>
	{:else}
		<p class="tr-line">
			{#if scope === 'all'}
				The gate saw <strong use:alive={{ text: String(sawTotal.toLocaleString()), count: true }}></strong> actions{sawTotal >= 2000 ? ' (the newest 2,000)' : ''}: it stopped
				<strong use:alive={{ text: String(stopCount), count: true }}></strong>{shadowCount ? `, would have stopped ${shadowCount} more in shadow mode` : ''}{permitCount ? `, and ${permitCount} went through on a permit` : ''}.
			{:else}
				<strong use:alive={{ text: String(stopCount.toLocaleString()), count: true }}></strong> stops{permitCount ? ` and ${permitCount.toLocaleString()} actions a permit let through` : ''},
				every one a hash-chained receipt written outside the agent.
			{/if}
		</p>
		<p class="tr-types">
			<span><i style:background={rgb('stop')}></i>stopped</span>
			<span><i style:background={rgb('permit')}></i>permitted</span>
			{#if scope === 'all'}
				<span><i style:background={rgb('shadow')}></i>shadow</span>
				<span><i style:background={rgb('pass')}></i>passed</span>
			{/if}
			{#each sourceCounts.slice(0, 5) as [src, n] (src)}<span>{src} {n}</span>{/each}
			{#if sourceCounts.length > 5}<span>{sourceCounts.length - 5} more sources</span>{/if}
		</p>
	{/if}
</header>

<aside class="tr-lane" class:idle={!pickedId} aria-live="polite">
	{#if picked}
		<p class="tr-kicker" style:color={rgb(kindOf(picked.decision))}>
			{verb(picked.decision)}{picked.rule ?? picked.commitment ? ` · ${picked.rule ?? picked.commitment}` : ''} · {when(picked.ts_utc)}
		</p>
		<p class="tr-text" use:alive={{ text: headline(picked.normalized || picked.action_preview || '', 520) }}></p>
		{#if picked.detail}<p class="tr-note">What the gate saw: {picked.detail}</p>{/if}
		{#if picked.transforms?.length}<p class="tr-note">Normalised by {picked.transforms.join(', ')} before matching.</p>{/if}

		<div class="tr-block">
			<p class="tr-proof-kicker" use:alive={{ text: 'Receipt' }}></p>
			<p class="tr-proof-line">{picked.source ?? '?'} · {picked.tool ?? '?'} · gate {picked.gate_version ?? '?'}</p>
			<p class="tr-id">{picked.digest ?? '—'}</p>
			<p class="tr-note">
				chained to {short(picked.prev)}
				{#if picked.prev && byDigest.has(picked.prev)}
					<button type="button" class="tr-chip" onclick={() => (pickedId = picked.prev ?? null)}>step back</button>
				{/if}
			</p>
			{#if picked.permit}
				<p class="tr-proof-line">
					permit {picked.permit.status ?? ''}{picked.permit.single_use ? ', single use' : ''}{picked.permit.by ? `, granted by ${picked.permit.by}` : ''}
				</p>
				<p class="tr-note">
					{picked.permit.granted ? `granted ${when(new Date(picked.permit.granted * 1000).toISOString())}` : ''}{picked.permit.expires
						? ` · expires ${when(new Date(picked.permit.expires * 1000).toISOString())}`
						: ''}
				</p>
			{/if}
			{#if picked.cwd}<p class="tr-note">in {picked.cwd}</p>{/if}
		</div>
		{#if frame?.stop}
			<div class="tr-block" data-cinema-frame="stop">
				<p class="tr-proof-kicker">Proof frame</p>
				<p class="tr-proof-line">{frame.rule || 'rule unavailable'}</p>
				<p class="tr-id">{frame.proof_handle || 'receipt unavailable'}</p>
				<p class="tr-note">{frame.receipt_digest || '—'}</p>
				<p class="tr-note">The tool process was not created.</p>
				{#if frame.chain_break}
					<p class="tr-err">This step does not match the receipt before it.</p>
				{/if}
				<p class="tr-note">Local record. No span leaves this machine.</p>
			</div>
		{/if}
		<div class="tr-actions">
			<button type="button" class="tr-act quiet" onclick={() => (pickedId = null)}>Back to every rule</button>
		</div>
	{:else}
		<p class="tr-proof-kicker">{summary || 'Rules'}</p>
		{#if ruleFilter}
			<button type="button" class="tr-chip" onclick={() => (ruleFilter = null)}>show every rule</button>
		{/if}
		{#each ruleCounts as [rule, n] (rule)}
			<button type="button" class="tr-item" aria-current={ruleFilter === rule} onclick={() => (ruleFilter = ruleFilter === rule ? null : rule)}>
				<span class="tr-edge causal">{n.stop} stopped{n.allow ? ` · ${n.allow} permitted` : ''}</span>
				<span>{rule}</span>
			</button>
		{:else}
			<p class="tr-note">{loading ? '' : 'Nothing reached the gate in this window.'}</p>
		{/each}
		<p class="tr-note">Click a light to read its receipt; click a rule to see only its stops.</p>
	{/if}
</aside>
