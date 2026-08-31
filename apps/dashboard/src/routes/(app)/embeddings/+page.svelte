<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AmbientField from '$components/AmbientField.svelte';
	import type {
		EmbeddingProfile,
		EmbeddingProfileActionResponse,
		EmbeddingProfilesResponse
	} from '$types';

	type Action = 'install' | 'evaluate' | 'migrate' | 'activate' | 'rollback';

	const FALLBACK_PROFILES: EmbeddingProfile[] = [
		{
			id: 'nomic-v1.5-legacy-raw-256', name: 'Nomic Embed Text v1.5 legacy', modelId: 'nomic-ai/nomic-embed-text-v1.5',
			description: 'Vestige’s proven default. Kept intact while optional profiles are prepared beside it.',
			stage: 'active', installed: true, active: true, dimensions: 256, maxTokens: 8192,
			vectorBytes: 1024, hardware: 'Reported by the local preflight', localOnly: true,
			migration: { state: 'not_required' }, evaluation: { state: 'complete', metric: 'Current production baseline' }
		},
		{
			id: 'qwen3-0.6b-retrieval-v1-1024', name: 'Qwen3 Embedding 0.6B', modelId: 'Qwen/Qwen3-Embedding-0.6B',
			description: 'Preview catalog profile. The local preflight reports its actual artifact and hardware requirements.',
			stage: 'available', installed: false, active: false, dimensions: 1024, maxTokens: 8192,
			localOnly: true,
			migration: { state: 'not_started' }, evaluation: { state: 'not_run' }
		},
		{
			id: 'qwen3-4b-retrieval-v1-1024', name: 'Qwen3 Embedding 4B', modelId: 'Qwen/Qwen3-Embedding-4B',
			description: 'Preview catalog profile. The local preflight reports its actual artifact and hardware requirements.',
			stage: 'available', installed: false, active: false, dimensions: 1024, maxTokens: 8192,
			localOnly: true,
			migration: { state: 'not_started' }, evaluation: { state: 'not_run' }
		}
	];

	let profiles = $state<EmbeddingProfile[]>(FALLBACK_PROFILES);
	let activeProfileId = $state<string | null>('nomic-v1.5-legacy-raw-256');
	let rollbackProfileId = $state<string | null>(null);
	let available = $state<boolean | null>(null);
	let loading = $state(true);
	let refreshing = $state(false);
	let busy = $state<{ action: Action; profileId: string } | null>(null);
	let notice = $state('Checking the local embedding service…');
	let unavailableReason = $state<string | null>(null);
	let pending = $state<{ action: Action; profile: EmbeddingProfile } | null>(null);
	let cliCommand = $state<string | null>(null);

	const activeProfile = $derived(profiles.find((profile) => profile.id === activeProfileId) ?? profiles.find((profile) => profile.active) ?? null);
	const rollbackProfile = $derived(profiles.find((profile) => profile.id === rollbackProfileId) ?? null);

	onMount(() => void refresh());

	function isUnavailable(error: unknown) {
		return error instanceof Error && (/API 404|API 405|API 501|Failed to fetch|NetworkError/i.test(error.message));
	}

	function applySnapshot(snapshot: Partial<EmbeddingProfilesResponse | EmbeddingProfileActionResponse>) {
		if (Array.isArray(snapshot.profiles) && snapshot.profiles.length) profiles = snapshot.profiles;
		if ('activeProfileId' in snapshot && snapshot.activeProfileId !== undefined) activeProfileId = snapshot.activeProfileId ?? null;
		if ('rollbackProfileId' in snapshot && snapshot.rollbackProfileId !== undefined) rollbackProfileId = snapshot.rollbackProfileId ?? null;
		if ('available' in snapshot && snapshot.available !== undefined) available = snapshot.available ?? true;
	}

	async function refresh() {
		refreshing = true;
		if (loading) notice = 'Checking the local embedding service…';
		try {
			const state = await api.embeddings.profiles();
			applySnapshot(state);
			available = state.available ?? true;
			unavailableReason = null;
			notice = 'Live local profile state loaded. Every vector space remains isolated.';
		} catch (cause) {
			available = false;
			// The browser never surfaces raw server text here: it may contain an
			// implementation detail (such as an artifact path) that is not useful
			// to an operator and should remain local to server logs/receipts.
			unavailableReason = 'Embedding Profiles is not available from this Vestige server yet.';
			notice = 'The service has not exposed Embedding Profiles yet. This preview cannot change models or download files.';
		} finally {
			loading = false;
			refreshing = false;
		}
	}

	function nextAction(profile: EmbeddingProfile): Action | null {
		if (profile.active) return null;
		if (!profile.installed) return 'install';
		if (profile.evaluation?.state !== 'complete') return 'evaluate';
		if (!['complete', 'not_required'].includes(profile.migration?.state ?? 'not_started')) return 'migrate';
		return 'activate';
	}

	function requiresExplicitLocalCli(action: Action) {
		return action === 'install' || action === 'evaluate' || action === 'migrate';
	}

	function actionLabel(action: Action) {
		return ({ install: 'Show install command', evaluate: 'Show evaluation command', migrate: 'Show migration command', activate: 'Activate profile', rollback: 'Roll back now' } as const)[action];
	}

	function localCliCommand(action: Action, profile: EmbeddingProfile) {
		const artifactRoot = '<verified-artifact-directory>';
		switch (action) {
			case 'install': return `vestige embeddings install ${profile.id} --from ${artifactRoot} --yes`;
			case 'evaluate': return `vestige embeddings evaluate ${profile.id} --from ${artifactRoot}`;
			case 'migrate': return `vestige embeddings migrate --to ${profile.id} --from ${artifactRoot}${profile.migration?.id && profile.migration.state !== 'cancelled' ? ` --resume ${profile.migration.id}` : ''} --yes`;
			default: return null;
		}
	}

	function actionDetail(action: Action, profile: EmbeddingProfile) {
		switch (action) {
			case 'install': return `Verify and register local artifacts for ${profile.name}. No model becomes active.`;
			case 'evaluate': return `Run the local evaluation before ${profile.name} can create retrieval vectors.`;
			case 'migrate': return `Create a separate ${profile.name} vector index. Existing Nomic vectors remain unchanged.`;
			case 'activate': return `Atomically point retrieval to ${profile.name}. The prior profile stays rollback-ready.`;
			case 'rollback': return `Atomically restore ${profile.name}. No files or vectors will be deleted.`;
		}
	}

	function requestAction(action: Action, profile: EmbeddingProfile) {
		if (!available) {
			notice = 'No action was sent: this server does not yet provide the Embedding Profiles API.';
			return;
		}
		if (requiresExplicitLocalCli(action)) {
			cliCommand = localCliCommand(action, profile);
			notice = `No dashboard operation was sent. ${profile.name} requires this explicit local command so Vestige never accepts or stores artifact directories over HTTP.`;
			return;
		}
		pending = { action, profile };
	}

	async function confirmAction() {
		if (!pending) return;
		const { action, profile } = pending;
		pending = null;
		cliCommand = null;
		busy = { action, profileId: profile.id };
		notice = `${actionLabel(action)} requested for ${profile.name}. Waiting for the local receipt…`;
		try {
			let result: EmbeddingProfileActionResponse;
			switch (action) {
				case 'install': result = await api.embeddings.install(profile.id); break;
				case 'evaluate': result = await api.embeddings.evaluate(profile.id); break;
				case 'migrate': result = await api.embeddings.migrate(profile.id); break;
				case 'activate': result = await api.embeddings.activate(profile.id); break;
				case 'rollback': result = await api.embeddings.rollback(profile.id); break;
			}
			applySnapshot(result);
			notice = result.message ?? `${actionLabel(action)} accepted. Refreshing the receipt-backed profile state…`;
			await refresh();
		} catch (cause) {
			if (isUnavailable(cause)) {
				available = false;
				unavailableReason = 'Embedding Profiles is not available from this Vestige server yet.';
				notice = 'No model state changed: the Embedding Profiles API is unavailable on this server.';
			} else {
				notice = 'The local operation did not complete. Review the local server receipt before trying again.';
			}
		} finally {
			busy = null;
		}
	}

	function stageLabel(profile: EmbeddingProfile) {
		if (profile.active) return 'Active';
		if (profile.migration?.state === 'in_progress') return 'Migrating';
		if (profile.migration?.state === 'validating') return 'Validating index';
		if (profile.migration?.state === 'paused') return 'Migration paused';
		if (profile.migration?.state === 'cancelled') return 'Migration cancelled';
		if (profile.evaluation?.state === 'running') return 'Evaluating';
		return profile.stage.replaceAll('_', ' ');
	}

	function formatBytes(value?: number) {
		if (value == null) return 'reported at preflight';
		if (value === 0) return 'already bundled';
		const units = ['B', 'KB', 'MB', 'GB', 'TB'];
		let index = 0;
		let amount = value;
		while (amount >= 1000 && index < units.length - 1) { amount /= 1000; index += 1; }
		return `${amount >= 10 ? amount.toFixed(0) : amount.toFixed(1)} ${units[index]}`;
	}

	function progress(profile: EmbeddingProfile) {
		const migration = profile.migration;
		if (!migration?.total || migration.completed == null) return null;
		return Math.round((migration.completed / migration.total) * 100);
	}

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

<svelte:head><title>Embedding Profiles · Vestige</title></svelte:head>

<main class="embeddings-shell" style="position: relative">
	<AmbientField {...ambient} accent={[0.55, 0.78, 0.86]} opacity={0.5} />
	<PageHeader icon="embeddings" accent="synapse" title="Embedding Profiles" subtitle="Local vector spaces with a deliberate, receipt-backed path from install to rollback.">
		<button class="refresh" type="button" onclick={refresh} disabled={refreshing || busy !== null}>
			{refreshing ? 'Refreshing…' : 'Refresh local state'}
		</button>
	</PageHeader>

	<section class:unavailable={available === false} class="local-contract" aria-label="Embedding profile safety contract">
		<div class="contract-icon"><Icon name={available === false ? 'pulse' : 'embeddings'} size={20} /></div>
		<div><strong>{available === false ? 'Preview only — service unavailable' : 'Local-only, profile-isolated retrieval'}</strong><p>{available === false ? 'No controls can download a model, switch retrieval, or re-embed memories until the server advertises these endpoints.' : 'Install, evaluation, and migration run through an explicit local CLI command with a verified artifact directory. This dashboard only reads receipts and performs separately confirmed, fully-gated activation or rollback. Vestige never compares vectors across profiles.'}</p></div>
	</section>

	{#if unavailableReason}
		<p class="availability-note" role="status"><span>LOCAL API</span>{unavailableReason}</p>
	{/if}

	<section class="active-strip" aria-label="Current embedding profile">
		<div><p class="eyebrow">ACTIVE RETRIEVAL POINTER</p><strong>{activeProfile?.name ?? 'No profile active'}</strong><span>{activeProfile?.modelId ?? 'No local vector space selected'}</span></div>
		<div class="pointer-rule"><i></i><span>one active profile</span><i></i></div>
		<div class="rollback"><p class="eyebrow">ROLLBACK</p>{#if rollbackProfile}<strong>{rollbackProfile.name}</strong><button type="button" onclick={() => requestAction('rollback', rollbackProfile)} disabled={busy !== null || available === false}>Restore preserved profile</button>{:else}<strong>Nothing pending</strong><span>The current profile has no newer activation to undo.</span>{/if}</div>
	</section>

	<section class="profile-grid" aria-busy={loading}>
		{#each profiles as profile (profile.id)}
			{@const next = nextAction(profile)}
			{@const percent = progress(profile)}
			<article class:active={profile.active} class="profile-card">
				<header>
					<div><p class="eyebrow">{profile.active ? 'DEFAULT · PRESERVED' : 'OPTIONAL PROFILE'}</p><h2>{profile.name}</h2><code>{profile.modelId}</code></div>
					<span class:active={profile.active} class="stage">{stageLabel(profile)}</span>
				</header>
				<p class="description">{profile.description ?? 'A local profile with its own encoder and vector contract.'}</p>

				<dl class="tradeoffs">
					<div><dt>Encoder</dt><dd>{profile.dimensions.toLocaleString()} dimensions</dd></div>
					<div><dt>Context</dt><dd>{profile.maxTokens?.toLocaleString() ?? '—'} tokens</dd></div>
					<div><dt>Model disk</dt><dd>{formatBytes(profile.modelBytes)}</dd></div>
					<div><dt>Vector / memory</dt><dd>{formatBytes(profile.vectorBytes)}</dd></div>
					<div class="wide"><dt>Hardware path</dt><dd>{profile.hardware ?? 'Local hardware preflight required'}</dd></div>
				</dl>

				<div class="stages" aria-label={`${profile.name} staged workflow`}>
					<span class:done={profile.installed}>1 Install</span><i></i>
					<span class:done={profile.evaluation?.state === 'complete'}>2 Evaluate</span><i></i>
					<span class:done={['complete', 'not_required'].includes(profile.migration?.state ?? '')}>3 Isolate vectors</span><i></i>
					<span class:done={profile.active}>4 Activate</span>
				</div>

				{#if percent !== null}
					<div class="migration-progress"><div><span>Separate index migration</span><strong>{profile.migration?.completed?.toLocaleString()} / {profile.migration?.total?.toLocaleString()}</strong></div><progress value={percent} max="100">{percent}%</progress></div>
				{/if}
				{#if profile.evaluation?.state === 'complete'}
					<p class="evaluation"><Icon name="pulse" size={14} /> {profile.evaluation.metric ?? 'Local evaluation complete'}{#if profile.evaluation.score != null}: {profile.evaluation.score.toFixed(3)}{/if}{#if profile.evaluation.sampleSize != null} · {profile.evaluation.sampleSize.toLocaleString()} cases{/if}</p>
				{/if}

				<footer>
					{#if profile.lastReceipt?.summary}<p class="receipt"><span>RECEIPT</span>{profile.lastReceipt.summary}</p>{:else}<p class="receipt"><span>CONTRACT</span>{profile.active ? 'Default is unchanged until you explicitly activate another evaluated profile.' : 'No changes occur until the next explicit stage is confirmed.'}</p>{/if}
					{#if next}<button class="action" type="button" disabled={busy !== null || available === false} onclick={() => requestAction(next, profile)}>{busy?.profileId === profile.id ? `${stageLabel(profile)}…` : actionLabel(next)} <span>→</span></button>{:else if profile.active}<span class="active-state">● Retrieval is using this profile</span>{/if}
				</footer>
			</article>
		{/each}
	</section>

	<section class="operation-status" aria-live="polite"><p class="eyebrow">OPERATION STATUS</p><output>{notice}</output>{#if cliCommand}<p class="cli-instruction"><span>RUN LOCALLY AGAINST THIS VESTIGE DATA DIRECTORY</span><code>{cliCommand}</code><small>Replace the placeholder with the verified local artifact directory. This page never sends that path to the server.</small></p>{/if}</section>
</main>

{#if pending}
	<div class="confirmation-backdrop" role="presentation">
		<div class="confirmation" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
			<p class="eyebrow">EXPLICIT CONFIRMATION</p><h2 id="confirm-title">{actionLabel(pending.action)}?</h2><p>{actionDetail(pending.action, pending.profile)}</p>
			<div class="confirmation-contract"><Icon name="embeddings" size={17} /><span>Local only. This stage does not silently run any later stage.</span></div>
			<div class="confirmation-actions"><button type="button" class="cancel" onclick={() => (pending = null)}>Cancel</button><button type="button" class="confirm" onclick={confirmAction}>Confirm {actionLabel(pending.action)}</button></div>
		</div>
	</div>
{/if}

<style>
	.embeddings-shell{position:relative;z-index:2;min-height:100%;max-width:1280px;margin:0 auto;padding:2rem clamp(1rem,3vw,2.75rem) 5rem;color:#e9fbf8;overflow:auto;pointer-events:none}.embeddings-shell :global(button){pointer-events:auto}.refresh{border:1px solid rgba(104,234,212,.32);border-radius:.65rem;background:rgba(6,29,31,.82);padding:.58rem .78rem;color:#91f0df;font:700 .7rem ui-monospace,monospace;cursor:pointer}.refresh:hover:not(:disabled){border-color:#83f9e1;background:rgba(67,222,197,.13)}button:disabled{cursor:not-allowed;opacity:.53}.local-contract,.active-strip,.profile-card,.operation-status{border:1px solid rgba(119,214,197,.2);background:linear-gradient(135deg,rgba(8,30,34,.92),rgba(4,13,19,.89));box-shadow:0 18px 66px rgba(0,0,0,.22);backdrop-filter:blur(15px)}.local-contract{display:flex;align-items:flex-start;gap:.78rem;border-radius:1rem;padding:1rem 1.1rem;pointer-events:auto}.contract-icon{display:grid;place-items:center;flex:0 0 2.2rem;height:2.2rem;border-radius:.65rem;background:rgba(74,230,207,.13);color:#74ead5}.local-contract strong{font-size:.87rem}.local-contract p{max-width:85ch;margin:.25rem 0 0;color:#a7c7c2;font-size:.76rem;line-height:1.5}.local-contract.unavailable{border-color:rgba(245,177,89,.34)}.local-contract.unavailable .contract-icon{background:rgba(245,177,89,.12);color:#f4ba6f}.availability-note{display:flex;gap:.55rem;align-items:center;margin:.6rem .15rem;color:#e1a966;font:.68rem/1.4 ui-monospace,monospace}.availability-note span,.eyebrow,.receipt span,.cli-instruction span{color:#69e5d0;font:700 .62rem/1.2 ui-monospace,monospace;letter-spacing:.13em}.availability-note span{color:#e5a85e}.active-strip{display:grid;grid-template-columns:1fr minmax(160px,.62fr) 1fr;align-items:center;gap:1rem;margin-top:1rem;border-radius:1rem;padding:1rem 1.15rem;pointer-events:auto}.active-strip strong{display:block;margin:.32rem 0 .15rem;color:#dffdfa;font-size:.94rem}.active-strip span{color:#8baea8;font:.67rem/1.4 ui-monospace,monospace}.pointer-rule{display:flex;align-items:center;gap:.45rem;color:#67ddca;font:.58rem ui-monospace,monospace;text-align:center;white-space:nowrap}.pointer-rule i{display:block;flex:1;height:1px;background:linear-gradient(90deg,transparent,#42c9b5,transparent)}.rollback{text-align:right}.rollback button{margin-top:.38rem;border:0;background:none;color:#79e9d4;font:700 .67rem ui-monospace,monospace;cursor:pointer}.profile-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:1rem;pointer-events:auto}.profile-card{position:relative;display:flex;min-width:0;flex-direction:column;border-radius:1rem;padding:1.1rem;overflow:hidden}.profile-card.active{border-color:rgba(108,239,211,.48);box-shadow:0 0 0 1px rgba(86,231,204,.1),0 22px 72px rgba(11,178,152,.12)}.profile-card.active::before{position:absolute;inset:0;background:radial-gradient(circle at 87% 0,rgba(70,227,201,.14),transparent 43%);content:'';pointer-events:none}.profile-card header,.profile-card footer{position:relative;display:flex;gap:.75rem;justify-content:space-between}.profile-card h2{max-width:21ch;margin:.45rem 0 .28rem;font-size:1.05rem;line-height:1.17;letter-spacing:-.02em}.profile-card code{display:block;max-width:25ch;overflow:hidden;color:#7ca8a1;font:.61rem/1.35 ui-monospace,monospace;text-overflow:ellipsis;white-space:nowrap}.stage{height:max-content;border:1px solid rgba(133,183,176,.28);border-radius:99px;padding:.27rem .45rem;color:#a7c8c3;font:.58rem ui-monospace,monospace;text-transform:capitalize;white-space:nowrap}.stage.active{border-color:rgba(97,232,203,.44);background:rgba(52,214,184,.11);color:#76eed8}.description{min-height:3.5em;margin:.85rem 0;color:#a7c4c0;font-size:.75rem;line-height:1.53}.tradeoffs{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin:0;border-top:1px solid rgba(133,209,196,.15);padding-top:.8rem}.tradeoffs div{min-width:0}.tradeoffs .wide{grid-column:1/-1}.tradeoffs dt{color:#769b95;font:.58rem ui-monospace,monospace;text-transform:uppercase;letter-spacing:.07em}.tradeoffs dd{margin:.2rem 0 0;color:#d2ede8;font-size:.7rem;line-height:1.35}.stages{display:flex;align-items:center;gap:.32rem;margin:1.05rem 0 .8rem;color:#627d79;font:.57rem ui-monospace,monospace;white-space:nowrap}.stages span.done{color:#74e5d0}.stages i{height:1px;flex:1;background:rgba(105,168,158,.24)}.migration-progress{margin:-.12rem 0 .8rem}.migration-progress div{display:flex;justify-content:space-between;color:#9fc8c1;font:.62rem ui-monospace,monospace}.migration-progress progress{width:100%;height:.35rem;margin-top:.35rem;accent-color:#63dcc7}.evaluation{display:flex;align-items:center;gap:.32rem;margin:-.08rem 0 .8rem;color:#79ded0;font:.64rem/1.4 ui-monospace,monospace}.profile-card footer{align-items:flex-end;min-height:2.4rem;margin-top:auto;padding-top:.85rem;border-top:1px solid rgba(133,209,196,.15)}.receipt{max-width:60%;margin:0;color:#93b8b2;font-size:.63rem;line-height:1.4}.receipt span{display:block;margin-bottom:.16rem}.action{border:1px solid rgba(100,234,210,.42);border-radius:.58rem;background:linear-gradient(135deg,rgba(42,202,177,.21),rgba(19,106,104,.19));padding:.55rem .64rem;color:#c9fff3;font:700 .66rem ui-monospace,monospace;cursor:pointer;white-space:nowrap}.action:hover:not(:disabled){border-color:#8ff6df;box-shadow:0 0 20px rgba(71,224,197,.18)}.action span{margin-left:.3rem;color:#73e9d2}.active-state{color:#77ead5;font:.63rem ui-monospace,monospace;white-space:nowrap}.operation-status{margin-top:1rem;border-radius:1rem;padding:.85rem 1rem;pointer-events:auto}.operation-status output{display:block;margin-top:.35rem;color:#c3e4df;font-size:.75rem;line-height:1.45}.cli-instruction{display:grid;gap:.42rem;margin:.75rem 0 0;border-top:1px solid rgba(133,209,196,.15);padding-top:.75rem}.cli-instruction code{overflow:auto;border:1px solid rgba(105,229,208,.18);border-radius:.5rem;background:rgba(1,10,13,.7);padding:.55rem;color:#a7f4e5;font:.67rem/1.45 ui-monospace,monospace}.cli-instruction small{color:#86a9a3;font-size:.67rem;line-height:1.4}.confirmation-backdrop{position:fixed;z-index:50;inset:0;display:grid;place-items:center;background:rgba(1,7,10,.74);padding:1rem;backdrop-filter:blur(8px)}.confirmation{width:min(100%,460px);border:1px solid rgba(111,235,212,.36);border-radius:1rem;background:linear-gradient(145deg,#0a292e,#061419);padding:1.25rem;box-shadow:0 26px 90px rgba(0,0,0,.5)}.confirmation h2{margin:.52rem 0;font-size:1.25rem}.confirmation>p:not(.eyebrow){color:#b0d0cb;font-size:.82rem;line-height:1.5}.confirmation-contract{display:flex;gap:.5rem;align-items:center;margin:1rem 0;border-radius:.62rem;background:rgba(83,222,197,.09);padding:.68rem;color:#85e8d6;font:.68rem/1.4 ui-monospace,monospace}.confirmation-actions{display:flex;justify-content:flex-end;gap:.6rem}.confirmation-actions button{border-radius:.6rem;padding:.58rem .75rem;font:700 .68rem ui-monospace,monospace;cursor:pointer}.cancel{border:1px solid rgba(157,194,188,.25);background:transparent;color:#bad6d1}.confirm{border:1px solid rgba(112,239,212,.45);background:#38cdb4;color:#04211e}.confirm:hover{background:#6cebd5}@media(max-width:980px){.profile-grid{grid-template-columns:1fr 1fr}.profile-card:last-child{grid-column:1/-1}.active-strip{grid-template-columns:1fr 1fr}.pointer-rule{display:none}}@media(max-width:620px){.embeddings-shell{padding-top:1.25rem}.active-strip,.profile-grid{grid-template-columns:1fr}.rollback{text-align:left}.profile-card:last-child{grid-column:auto}.profile-card h2{font-size:1rem}.stages{font-size:.51rem;gap:.2rem}.profile-card footer{align-items:flex-start;flex-direction:column}.receipt{max-width:100%}.action{width:100%}.confirmation-actions{display:grid;grid-template-columns:1fr 1fr}}
</style>
