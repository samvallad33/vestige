<script lang="ts">
	import { onMount } from 'svelte';
	import LaunchEngineHost from '$lib/launch/LaunchEngineHost.svelte';
	import NeuralWordmark from '$lib/launch/NeuralWordmark.svelte';

	const heroSeed = 20260625;
	const GITHUB_URL = 'https://github.com/samvallad33/vestige';
	const LAUNCH_DATE = 'July 14, 2026';
	// only show the count once it's meaningful social proof (never "3 people").
	const COUNT_REVEAL_THRESHOLD = 25;
	const CODE_RE = /^[a-z2-9]{4,16}$/;

	type SubmitState = 'idle' | 'submitting' | 'success' | 'error';
	type WaitlistApi = typeof import('$lib/launch/waitlist');
	type PreHydrationResult =
		| { ok: true; referralCode: string; referrals: number; duplicate: boolean }
		| { ok: false; reason: 'invalid' | 'error'; message?: string };

	let mounted = $state(false);
	let prefersReducedMotion = $state(false);
	let revealed = $state(false);

	// .launch-shell ref → the engine writes --burst/--flash onto it each frame.
	let shell = $state<HTMLElement | undefined>(undefined);
	// pin the overlay to rest while the user is in the form (never disrupt a signup).
	let formInteracting = $state(false);
	let idleTimer: ReturnType<typeof setTimeout> | undefined;

	let email = $state('');
	let honeypot = $state(''); // spam trap
	let submitState = $state<SubmitState>('idle');
	let submitMessage = $state('');
	let waitlistCount = $state<number | null>(null); // real count from Supabase

	// referral loop: the code that brought THIS visitor in (?ref=), the code we
	// give THEM to share, and how many friends have joined from their link.
	let incomingRef = $state<string | undefined>(undefined);
	let myReferralCode = $state<string | null>(null);
	let myReferrals = $state<number>(0);
	let shareUrl = $derived(myReferralCode ? buildShareUrl(myReferralCode) : '');
	let copied = $state(false);
	let refTimer: ReturnType<typeof setInterval> | undefined;
	let waitlistApiPromise: Promise<WaitlistApi> | undefined;
	const supabaseUrl = (import.meta.env.VITE_SUPABASE_URL as string | undefined)?.trim() ?? '';
	const supabaseAnonKey = (import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined)?.trim() ?? '';
	const preHydrationWaitlistScript = createPreHydrationWaitlistScript(
		supabaseUrl,
		supabaseAnonKey
	);

	function loadWaitlistApi() {
		return (waitlistApiPromise ??= import('$lib/launch/waitlist'));
	}

	function createPreHydrationWaitlistScript(url: string, anonKey: string) {
		if (!url || !anonKey) return '';
		return `
(() => {
	if (window.__vestigeWaitlistPreHydration) return;
	window.__vestigeWaitlistPreHydration = true;
	const SUPABASE_URL = ${JSON.stringify(url)};
	const SUPABASE_KEY = ${JSON.stringify(anonKey)};
	const EMAIL_RE = /^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$/;
	const CODE_RE = /^[a-z2-9]{4,16}$/;
	function referralCodeFromUrl() {
		try {
			const raw = new URLSearchParams(window.location.search).get('ref');
			const code = raw && raw.trim().toLowerCase();
			return code && CODE_RE.test(code) ? code : null;
		} catch {
			return null;
		}
	}
	function setMessage(form, message, isError) {
		let node = form.querySelector('[data-prehydrate-message]');
		if (!node) {
			node = document.createElement('p');
			node.className = 'msg' + (isError ? ' error' : '');
			node.setAttribute('data-prehydrate-message', 'true');
			form.appendChild(node);
		}
		node.textContent = message;
		node.classList.toggle('error', Boolean(isError));
	}
	function emitResult(detail) {
		window.__vestigeWaitlistResult = detail;
		window.dispatchEvent(new CustomEvent('vestige:waitlist-result', { detail }));
	}
	document.addEventListener('submit', async (event) => {
		const target = event.target;
		const form = target && target.closest ? target.closest('form[data-waitlist-form]') : null;
		if (!form || window.__vestigeWaitlistHydrated) return;
		event.preventDefault();
		event.stopPropagation();
		if (window.__vestigeWaitlistSubmitting) return;
		const input = form.querySelector('input.email');
		const button = form.querySelector('button.join-btn');
		const email = String((input && input.value) || '').trim().toLowerCase();
		if (!EMAIL_RE.test(email)) {
			setMessage(form, 'Enter a valid email so we can send your invite.', true);
			return;
		}
		window.__vestigeWaitlistSubmitting = true;
		if (button) {
			button.disabled = true;
			button.textContent = 'Joining...';
		}
		try {
			const response = await fetch(SUPABASE_URL.replace(/\\/$/, '') + '/rest/v1/rpc/join_waitlist', {
				method: 'POST',
				headers: {
					apikey: SUPABASE_KEY,
					authorization: 'Bearer ' + SUPABASE_KEY,
					'content-type': 'application/json'
				},
				body: JSON.stringify({
					p_email: email,
					p_referred_by: referralCodeFromUrl(),
					p_referrer: document.referrer || null
				})
			});
			const payload = await response.json().catch(() => null);
			if (!response.ok) {
				throw new Error((payload && payload.message) || 'waitlist request failed');
			}
			const row = Array.isArray(payload) ? payload[0] : payload;
			if (!row || !row.referral_code) throw new Error('no referral code returned');
			const detail = {
				ok: true,
				referralCode: row.referral_code,
				referrals: Number(row.referrals || 0),
				duplicate: Boolean(row.duplicate)
			};
			try {
				localStorage.setItem('vestige_waitlisted', '1');
				localStorage.setItem('vestige_referral_code', detail.referralCode);
			} catch {}
			if (input) input.value = '';
			if (button) button.textContent = 'Joined';
			setMessage(
				form,
				detail.duplicate
					? "You're already on the list - invite coming before July 14."
					: "You're on the list. We'll email your invite before July 14.",
				false
			);
			emitResult(detail);
		} catch (error) {
			const detail = {
				ok: false,
				reason: String(error && error.message || '').match(/invalid email/i) ? 'invalid' : 'error',
				message: error && error.message ? error.message : undefined
			};
			setMessage(
				form,
				detail.reason === 'invalid'
					? 'Enter a valid email so we can send your invite.'
					: 'Could not save that email yet. Try again in a moment.',
				true
			);
			emitResult(detail);
			if (button) {
				button.disabled = false;
				button.textContent = 'Get early access';
			}
		} finally {
			window.__vestigeWaitlistSubmitting = false;
		}
	}, true);
})();
		`.trim();
	}

	function deferNonVisualWork(callback: () => void) {
		setTimeout(() => {
			const requestIdle = (window as Window & {
				requestIdleCallback?: (cb: () => void, options?: { timeout: number }) => number;
			}).requestIdleCallback;
			if (requestIdle) {
				requestIdle(callback, { timeout: 2600 });
			} else {
				callback();
			}
		}, 1400);
	}

	function getReferralCodeFromUrl(): string | undefined {
		try {
			const raw = new URLSearchParams(window.location.search).get('ref');
			const code = raw?.trim().toLowerCase();
			return code && CODE_RE.test(code) ? code : undefined;
		} catch {
			return undefined;
		}
	}

	function buildShareUrl(referralCode: string): string {
		const publicBase = (import.meta.env.VITE_PUBLIC_LAUNCH_URL as string | undefined)?.trim();
		const base = publicBase || `${window.location.origin}${window.location.pathname}`;
		const sep = base.includes('?') ? '&' : '?';
		return `${base}${sep}ref=${encodeURIComponent(referralCode)}`;
	}

	// Deterministic email -> uint32 seed (FNV-1a). Same email always freezes on the
	// same brain; different emails diverge visibly. Lowercased+trimmed so casing and
	// whitespace don't fork the brain. Returns a finite uint32 (never NaN).
	function seedFromEmail(value: string): number {
		const s = (value || '').trim().toLowerCase();
		let h = 0x811c9dc5; // FNV offset basis
		for (let i = 0; i < s.length; i += 1) {
			h ^= s.charCodeAt(i);
			h = Math.imul(h, 0x01000193); // FNV prime
		}
		return h >>> 0; // unsigned 32-bit
	}

	// Fire the WebGPU supernova: the hero decouples from its ambient loop and freezes
	// on THIS user's seeded brain. Buffered on window so a late-mounting engine (the
	// engine lazy-loads ~1.4s in) can still read it. No-op on the server.
	function detonateSupernova(seedEmail: string) {
		if (typeof window === 'undefined') return;
		const seed = seedFromEmail(seedEmail);
		(window as Window & { __vestigeSupernova?: number }).__vestigeSupernova = seed;
		window.dispatchEvent(new CustomEvent('vestige:supernova', { detail: { seed } }));
	}

	// Detonate FIRST (seed the brain + start the WebGPU collapse and the DOM --burst
	// implode), release the form-focus suppression so the form is visibly sucked in,
	// THEN swap in the success card ~700ms later so the form implodes before the card
	// replaces it. reduced-motion swaps instantly (gentle reveal, no implode wait).
	let supernovaTimer: ReturnType<typeof setTimeout> | undefined;
	function runSupernovaThenSucceed(seedEmail: string, commit: () => void) {
		clearTimeout(idleTimer);
		formInteracting = false; // stop the engine pinning --burst to 0 so the form imploads
		detonateSupernova(seedEmail);
		const delay = prefersReducedMotion ? 0 : 700;
		clearTimeout(supernovaTimer);
		supernovaTimer = setTimeout(commit, delay);
	}

	function lockBurst() {
		formInteracting = true;
		clearTimeout(idleTimer);
	}
	function unlockBurstSoon() {
		clearTimeout(idleTimer);
		idleTimer = setTimeout(() => (formInteracting = false), 1200); // grace window
	}

	function markSuccess(note?: string, code?: string, referrals?: number) {
		submitState = 'success';
		submitMessage = note ?? "You're on the list. We'll email your invite before July 14.";
		if (typeof waitlistCount === 'number') waitlistCount += 1;
		try {
			localStorage.setItem('vestige_waitlisted', '1');
			if (code) localStorage.setItem('vestige_referral_code', code);
		} catch {
			/* ignore */
		}
		if (code) {
			myReferralCode = code;
			if (typeof referrals === 'number') myReferrals = referrals;
			startReferralPolling(code);
		}
	}

	function applyPreHydrationResult(result: PreHydrationResult) {
		if (result.ok) {
			// In the pre-hydration path Svelte's `email` state may not yet be bound to
			// the input the user typed into, so read the input value directly for the
			// seed (falls back to `email`). seedFromEmail('') is still deterministic.
			const inputEl =
				typeof document !== 'undefined'
					? (document.querySelector('input[type="email"]') as HTMLInputElement | null)
					: null;
			const joinedEmail = inputEl?.value || email;
			myReferrals = result.referrals;
			runSupernovaThenSucceed(joinedEmail, () => {
				email = '';
				markSuccess(
					result.duplicate ? "You're already on the list — invite coming before July 14." : undefined,
					result.referralCode,
					result.referrals
				);
			});
			return;
		}
		submitState = 'error';
		submitMessage =
			result.reason === 'invalid'
				? 'Enter a valid email so we can send your invite.'
				: 'Could not save that email yet. Try again in a moment.';
	}

	onMount(() => {
		let cancelled = false;
		const waitlistWindow = window as Window & {
			__vestigeWaitlistHydrated?: boolean;
			__vestigeWaitlistResult?: PreHydrationResult;
		};
		const onPreHydrationResult = (event: Event) => {
			applyPreHydrationResult((event as CustomEvent<PreHydrationResult>).detail);
		};
		waitlistWindow.__vestigeWaitlistHydrated = true;
		window.addEventListener('vestige:waitlist-result', onPreHydrationResult);
		mounted = true;
		prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
		// Reveal the wordmark + form IMMEDIATELY — the overlay must never wait on the
		// WebGPU engine. The engine's first boot (60k-particle buffer build + WGSL
		// pipeline compilation) is heavy enough to block the main thread for seconds
		// on some browsers (notably Safari), and the old double-rAF reveal queued
		// BEHIND that work — so the signup form stayed invisible for ~5s. A single
		// rAF flips the CSS opacity transition on the very next frame, before the
		// engine starts (the engine boot is deferred below).
		requestAnimationFrame(() => (revealed = true));

		// who referred this visitor? (?ref=CODE) — attributed on signup.
		incomingRef = getReferralCodeFromUrl();

		// restore any locally-stored signup so refresh doesn't re-prompt, and
		// bring back their share link + live referral count.
		if (waitlistWindow.__vestigeWaitlistResult) {
			applyPreHydrationResult(waitlistWindow.__vestigeWaitlistResult);
		} else {
			try {
				if (localStorage.getItem('vestige_waitlisted') === '1') {
					submitState = 'success';
					submitMessage = "You're on the list. We'll email your invite before July 14.";
					const savedCode = localStorage.getItem('vestige_referral_code');
					if (savedCode) {
						myReferralCode = savedCode;
						startReferralPolling(savedCode);
					}
				}
			} catch {
				/* private mode — ignore */
			}
		}

		// Fetch social proof after the first visual frames. Supabase should never
		// compete with canvas/WebGPU boot on refresh.
		deferNonVisualWork(() => {
			loadWaitlistApi()
				.then(({ getWaitlistCount }) => getWaitlistCount())
				.then((n) => {
					if (!cancelled && typeof n === 'number') waitlistCount = n;
				});
		});

		return () => {
			cancelled = true;
			waitlistWindow.__vestigeWaitlistHydrated = false;
			window.removeEventListener('vestige:waitlist-result', onPreHydrationResult);
			clearInterval(refTimer);
			clearTimeout(supernovaTimer);
		};
	});

	// poll the referrer's "N friends joined" counter so a sharer watching the
	// page sees it tick up live (cheap RPC; stops on unmount).
	function startReferralPolling(code: string) {
		const tick = async () => {
			const { getReferralCount } = await loadWaitlistApi();
			const n = await getReferralCount(code);
			if (typeof n === 'number') myReferrals = n;
		};
		tick();
		clearInterval(refTimer);
		refTimer = setInterval(tick, 15000);
	}

	async function joinWaitlist(event: SubmitEvent) {
		event.preventDefault();
		if (submitState === 'submitting' || submitState === 'success') return;

		// honeypot: silently "succeed" for bots
		if (honeypot.trim()) {
			submitState = 'success';
			submitMessage = "You're on the list.";
			return;
		}

		if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim())) {
			submitState = 'error';
			submitMessage = 'Enter a valid email so we can send your invite.';
			return;
		}

		submitState = 'submitting';
		submitMessage = '';

		const { joinWaitlist: submitWaitlist, waitlistConfigured } = await loadWaitlistApi();

		// Not configured (no Supabase env): keep the page usable for the demo by
		// capturing locally. Set VITE_SUPABASE_URL + VITE_SUPABASE_ANON_KEY to go live.
		if (!waitlistConfigured) {
			const joinedEmail = email;
			runSupernovaThenSucceed(joinedEmail, () => {
				email = '';
				markSuccess();
			});
			return;
		}

		const referrer = typeof document !== 'undefined' ? document.referrer || undefined : undefined;
		const result = await submitWaitlist(email, { referredBy: incomingRef, referrer });

		if (result.ok) {
			const joinedEmail = email;
			myReferrals = result.referrals;
			// SUPERNOVA: detonate the seeded-brain hero first, implode the form, THEN
			// swap in the success card so converting itself is the spectacle.
			runSupernovaThenSucceed(joinedEmail, () => {
				email = '';
				markSuccess(
					result.duplicate ? "You're already on the list — invite coming before July 14." : undefined,
					result.referralCode,
					result.referrals
				);
			});
		} else if (result.reason === 'invalid') {
			submitState = 'error';
			submitMessage = 'Enter a valid email so we can send your invite.';
		} else {
			submitState = 'error';
			submitMessage = 'Could not save that email yet. Try again in a moment.';
		}
	}

	async function shareInvite() {
		// share the visitor's PERSONAL referral link when we have one (so their
		// friends are attributed back to them); fall back to the plain page URL.
		const url = shareUrl || 'https://samvallad33.github.io/vestige';
		const text =
			'I just joined the Vestige waitlist — local-first memory for AI agents, rendered as a live WebGPU particle brain. Launching July 14:';
		try {
			if (navigator.share) {
				await navigator.share({ title: 'Vestige', text, url });
			} else {
				await navigator.clipboard.writeText(`${text} ${url}`);
				submitMessage = 'Invite link copied — send it to a friend.';
			}
		} catch {
			/* user cancelled — ignore */
		}
	}

	// one-tap X/Twitter share — the highest-signal share for a dev launch. Opens
	// the compose window pre-filled with the personal referral link.
	function shareToX() {
		const url = shareUrl || 'https://samvallad33.github.io/vestige';
		const text =
			'I just joined the @vestige waitlist — local-first memory for AI agents, rendered as a live WebGPU particle brain (60k particles, zero libraries). Launching July 14:';
		const intent = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
		window.open(intent, '_blank', 'noopener,noreferrer');
	}

	async function copyShareLink() {
		const url = shareUrl || 'https://samvallad33.github.io/vestige';
		try {
			await navigator.clipboard.writeText(url);
			copied = true;
			setTimeout(() => (copied = false), 2000);
		} catch {
			submitMessage = url; // clipboard blocked — at least show the link
		}
	}
</script>

<svelte:head>
	<title>Vestige · memory that watches itself think · waitlist</title>
	{@html preHydrationWaitlistScript ? `<script>${preHydrationWaitlistScript}</script>` : ''}
	<meta
		name="description"
		content="Vestige: local-first memory for AI coding agents, rendered as a living WebGPU particle brain. Launching July 14. Join the waitlist."
	/>
	<meta property="og:title" content="Vestige · launching July 14" />
	<meta
		property="og:description"
		content="Local-first memory for AI agents. A raw WebGPU particle brain you can watch think. Join the waitlist."
	/>
</svelte:head>

<main class="launch-shell" bind:this={shell} aria-label="Vestige launch">
	<LaunchEngineHost
		seed={heroSeed}
		reducedMotion={prefersReducedMotion}
		syncTarget={shell}
		suppress={formInteracting}
	/>

	<!-- readability scrim: keeps the wordmark + form crisp over the particles,
	     bottom-weighted so the CTA always reads. Strongest on mobile portrait. -->
	<div class="scrim" aria-hidden="true"></div>

	<!-- top bar -->
	<header class="topbar" class:revealed>
		<span class="wm-tiny" aria-hidden="true">VESTIGE</span>
		<a class="ghost-link" href={GITHUB_URL} target="_blank" rel="noreferrer">
			<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" fill="currentColor">
				<path
					d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
				/>
			</svg>
			<span>Star on GitHub</span>
		</a>
	</header>

	<!-- waitlist card -->
	<section class="overlay" class:revealed aria-labelledby="vestige-heading">
		<!-- TOP: the grown-dendrite wordmark -->
		<div class="hero-block">
			<h1 id="vestige-heading" class="wordmark-wrap" aria-label="Vestige">
				<NeuralWordmark />
			</h1>
		</div>

		<!-- middle stays open so the 3D particle space is fully visible -->

		<!-- BOTTOM: tagline, copy, email capture, meta -->
		<!-- svelte-ignore a11y_no_static_element_interactions a11y_no_noninteractive_element_interactions -->
		<div
			class="cta-block"
			onfocusin={lockBurst}
			onpointerdown={lockBurst}
			onkeydown={lockBurst}
			onfocusout={unlockBurstSoon}
		>
			<p class="tagline">memory that watches itself <span class="wild">think</span></p>
			<p class="subline">
				Local-first memory for AI coding agents. A living particle brain you can
				watch remember, forget, and reform.
			</p>

			{#if submitState !== 'success'}
				<form class="join" data-waitlist-form onsubmit={joinWaitlist} novalidate>
					<div class="field">
						<input
							class="email"
							type="email"
							inputmode="email"
							autocomplete="email"
							placeholder="you@email.com"
							bind:value={email}
							aria-label="Email address"
							required
						/>
						<button class="join-btn" type="submit" disabled={submitState === 'submitting'}>
							{submitState === 'submitting' ? 'Joining…' : 'Get early access'}
						</button>
					</div>
					<!-- honeypot (hidden from humans) -->
					<input
						class="hp"
						bind:value={honeypot}
						tabindex="-1"
						autocomplete="off"
						aria-hidden="true"
						placeholder="Leave this empty"
					/>
					{#if submitMessage}
						<p class="msg" class:error={submitState === 'error'}>{submitMessage}</p>
					{/if}
				</form>
			{:else}
				<div class="success" role="status">
					<p class="msg ok">{submitMessage}</p>

					{#if myReferralCode}
						<p class="share-lead">
							Move up the list — every friend who joins from your link gets you
							earlier access.
						</p>

						<!-- the personal share link (the viral artifact) -->
						<div class="share-link">
							<input
								class="share-url"
								type="text"
								readonly
								value={shareUrl}
								aria-label="Your personal invite link"
								onfocus={(e) => (e.currentTarget as HTMLInputElement).select()}
							/>
							<button class="copy-btn" type="button" onclick={copyShareLink}>
								{copied ? 'Copied ✓' : 'Copy'}
							</button>
						</div>

						<div class="share-actions">
							<button class="x-btn" type="button" onclick={shareToX}>
								<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" fill="currentColor">
									<path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231 5.45-6.231Zm-1.161 17.52h1.833L7.084 4.126H5.117L17.083 19.77Z" />
								</svg>
								Share on X
							</button>
							<button class="share-btn" type="button" onclick={shareInvite}>
								Invite a friend
							</button>
						</div>

						{#if myReferrals > 0}
							<p class="ref-count" aria-live="polite">
								🔥 <strong>{myReferrals.toLocaleString()}</strong>
								{myReferrals === 1 ? 'friend has' : 'friends have'} joined from your link
							</p>
						{:else}
							<p class="ref-count muted">No referrals yet — share your link to climb.</p>
						{/if}
						{:else}
							<!-- unconfigured / local-capture fallback: still let them share -->
							<button class="share-btn" type="button" onclick={shareInvite}>
								Invite a friend
							</button>
						{/if}
					</div>
				{/if}

			<div class="meta">
				<span class="date">▲ Launching {LAUNCH_DATE}</span>
				{#if waitlistCount !== null && waitlistCount >= COUNT_REVEAL_THRESHOLD}
					<span class="dot" aria-hidden="true">·</span>
					<span class="count"><strong>{waitlistCount.toLocaleString()}</strong> on the waitlist</span>
				{/if}
				<span class="dot" aria-hidden="true">·</span>
				<a class="src" href={GITHUB_URL} target="_blank" rel="noreferrer">open source</a>
			</div>
		</div>

			<p class="scroll-hint" aria-hidden="true">raw WebGPU · 60k particles · zero libraries</p>
		</section>
	</main>

<style>
	:global(body) {
		margin: 0;
		background: #02030a;
	}
	:global(*) {
		box-sizing: border-box;
	}

	/* the engine writes --burst (0 rest .. 1 collapse) and --flash (razor spike)
	   onto .launch-shell each frame; the overlay implodes/explodes off them.
	   Static fallbacks so the first frame (pre-boot) is never NaN. */
		.launch-shell {
			--burst: 0;
			--flash: 0;
			position: relative;
			min-height: 100vh;
			min-height: 100svh;
		overflow: hidden;
		color: #eaf3ff;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
		background: #02030a;
			isolation: isolate;
			contain: layout; /* keep layout scoped without trapping fixed canvas paint on mobile */
		}
	@property --burst {
		syntax: '<number>';
		inherits: true;
		initial-value: 0;
	}
	@property --flash {
		syntax: '<number>';
		inherits: true;
		initial-value: 0;
	}

	/* readability scrim — subtle on desktop (center formation has dark margins),
	   stronger and bottom-weighted on mobile portrait where particles fill all. */
	.scrim {
		position: fixed;
		inset: 0;
		z-index: 2;
		pointer-events: none;
		background:
			radial-gradient(120% 80% at 50% 42%, transparent 38%, rgba(2, 3, 10, 0.55) 100%),
			linear-gradient(180deg, rgba(2, 3, 10, 0.35) 0%, transparent 26%, transparent 50%, rgba(2, 3, 10, 0.78) 100%);
	}

	/* ---- top bar ---- */
	.topbar {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		z-index: 4;
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: clamp(0.85rem, 2.5vw, 1.4rem) clamp(1rem, 4vw, 2.4rem);
		/* Visible by default; .revealed only animates the entrance (JS enhancement). */
		opacity: 1;
		transform: none;
		transition:
			opacity 0.8s ease,
			transform 0.8s ease;
		pointer-events: none;
	}
	.topbar.revealed {
		opacity: 1;
		transform: none;
	}
	.wm-tiny {
		font-weight: 800;
		letter-spacing: 0.34em;
		font-size: 0.8rem;
		color: rgba(225, 240, 255, 0.62);
		text-indent: 0.34em;
	}
	.ghost-link {
		pointer-events: auto;
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.85rem;
		border-radius: 999px;
		border: 1px solid rgba(150, 190, 255, 0.22);
		background: rgba(10, 16, 34, 0.4);
		backdrop-filter: blur(10px);
		color: #cfe2ff;
		text-decoration: none;
		font-size: 0.82rem;
		font-weight: 650;
		transition:
			border-color 0.25s ease,
			background 0.25s ease,
			transform 0.25s ease;
	}
	.ghost-link:hover {
		border-color: rgba(120, 200, 255, 0.6);
		background: rgba(20, 30, 60, 0.55);
		transform: translateY(-1px);
	}

	/* ---- overlay: wordmark pinned TOP, copy + CTA pinned BOTTOM, middle open
	   so the 3D particle space is fully visible ---- */
	.overlay {
		position: relative;
		z-index: 3;
		min-height: 100svh;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: space-between;
		padding: clamp(1.6rem, 5vh, 3.4rem) 1.2rem clamp(2rem, 6vh, 4rem);
		text-align: center;
		pointer-events: none;
		perspective: 1200px; /* so the implode's translateZ reads as real depth */
	}

	/* hero-block: reveal via OPACITY only (transform is reserved for the implode,
	   driven per-frame off --burst so it sucks toward the singularity then snaps
	   back, in sync with the particles). Wordmark is heaviest / deepest pull. */
	.hero-block {
		width: 100%;
		/* Visible by DEFAULT so the wordmark + form NEVER depend on JS succeeding.
		   The .revealed class (added on mount) only drives the entrance fade as a
		   progressive enhancement; if hydration ever fails or onMount doesn't run
		   (seen on some mobile browsers), the content must still be fully visible. */
		opacity: 1;
		transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.05s;
		transform: scale(calc(1 - var(--burst) * 0.16))
			translateZ(calc(var(--burst) * -560px))
			rotateZ(calc(var(--burst) * -6deg));
		transform-origin: 50% 50%;
		will-change: transform;
	}
	.overlay.revealed .hero-block {
		opacity: 1;
	}
	/* white-hot flash punch lives on the inner wrapper (filter), so transform and
	   blur never fight on the same node (Chrome 'Animating a blur'). */
	.hero-block .wordmark-wrap {
		filter: brightness(calc(1 + var(--flash) * 2))
			drop-shadow(0 0 calc(var(--flash) * 30px) rgba(160, 210, 255, 0.7));
	}

		/* VESTIGE is built from grown dendrites (the NeuralWordmark component) — the
		   letterforms themselves are organic neural branches, alive by their shape. */
		.wordmark-wrap {
			margin: 0;
			width: 100%;
			display: flex;
			justify-content: center;
		}
	.tagline {
		margin: 0 0 0;
		font-size: clamp(1rem, 2.8vw, 1.55rem);
		font-weight: 600;
		letter-spacing: 0.02em;
		color: #d4e6ff;
	}
	/* "think" in weird, wild, shifting colours — a psychedelic gradient that
	   flows AND hue-rotates so it never sits on one colour. */
	.wild {
		font-weight: 800;
		background: linear-gradient(
			92deg,
			#ff2d95,
			#ff9a3d,
			#f5ff3d,
			#3dff7a,
			#36f0ff,
			#a14bff,
			#ff2d95
		);
		background-size: 400% 100%;
		-webkit-background-clip: text;
		background-clip: text;
		color: transparent;
		filter: drop-shadow(0 0 14px rgba(255, 90, 200, 0.45));
		will-change: background-position, filter;
		animation:
			wild-flow 4s linear infinite,
			wild-hue 7s linear infinite;
	}
	@keyframes wild-flow {
		to {
			background-position: 400% 0;
		}
	}
	@keyframes wild-hue {
		to {
			filter: drop-shadow(0 0 14px rgba(255, 90, 200, 0.45)) hue-rotate(360deg);
		}
	}
	.subline {
		margin: 0.7rem auto 0;
		max-width: 42ch;
		font-size: clamp(0.85rem, 2vw, 1.02rem);
		line-height: 1.55;
		color: rgba(190, 212, 245, 0.78);
	}

	/* ---- CTA (pinned to the bottom of the viewport) ---- */
	/* cta-block: lighter, further from centre → sucks in faster + fades more, so
	   the parallax vs the wordmark sells gravitational collapse (not a flat shrink).
	   Reveal entrance is a one-shot keyframe so it never fights the per-frame
	   --burst writes (no transition on the consumed transform/opacity). */
	/* cta-block: reveal via OPACITY only (transitions once); the implode lives on
	   `transform` + the independent `translate` (var-driven, NO transition, so the
	   per-frame --burst writes never lag). Lighter pull than the wordmark. */
	.cta-block {
		margin-top: auto; /* push to the very bottom; middle stays open */
		pointer-events: auto;
		width: min(540px, 92vw);
		/* Visible by DEFAULT — the signup form must NEVER be hidden if JS fails to
		   run. .revealed only drives the entrance fade (progressive enhancement). */
		opacity: 1;
		transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.22s;
		transform: scale(calc(1 - var(--burst) * 0.1)) translateZ(calc(var(--burst) * -300px));
		translate: 0 calc(var(--burst) * 14px);
		will-change: transform;
	}
	.overlay.revealed .cta-block {
		opacity: 1;
	}
	/* the shockwave hits the tagline at the flash instant */
	.cta-block .tagline {
		filter: blur(calc(var(--flash) * 2.5px));
	}

	.join {
		margin: 1.3rem 0 0;
	}
	.field {
		display: flex;
		gap: 0.5rem;
		padding: 0.4rem;
		border-radius: 16px;
		border: 1px solid rgba(140, 190, 255, 0.28);
		background: rgba(8, 14, 32, 0.55);
		backdrop-filter: blur(16px);
		box-shadow:
			0 20px 60px rgba(0, 0, 0, 0.45),
			0 0 0 1px rgba(120, 200, 255, 0.05) inset;
		transition: border-color 0.25s ease;
	}
	.field:focus-within {
		border-color: rgba(90, 200, 255, 0.7);
	}
	.email {
		flex: 1;
		min-width: 0;
		border: 0;
		background: transparent;
		color: #fff;
		font: inherit;
		font-size: 1.02rem;
		padding: 0.75rem 0.9rem;
		outline: none;
	}
	.email::placeholder {
		color: rgba(180, 205, 240, 0.5);
	}
	.join-btn {
		flex: 0 0 auto;
		border: 0;
		border-radius: 12px;
		padding: 0.78rem 1.25rem;
		font: inherit;
		font-weight: 750;
		color: #04122b;
		cursor: pointer;
		background: linear-gradient(135deg, #5fd0ff, #7b9bff);
		box-shadow: 0 8px 26px rgba(70, 160, 255, 0.4);
		transition:
			transform 0.2s ease,
			box-shadow 0.2s ease,
			filter 0.2s ease;
		white-space: nowrap;
	}
	.join-btn:hover:not(:disabled) {
		transform: translateY(-1px);
		filter: brightness(1.08);
		box-shadow: 0 12px 34px rgba(70, 160, 255, 0.55);
	}
	.join-btn:disabled {
		opacity: 0.7;
		cursor: wait;
	}
	.hp {
		position: absolute;
		left: -9999px;
		width: 1px;
		height: 1px;
		opacity: 0;
	}

	.msg {
		margin: 0.7rem 0 0;
		font-size: 0.86rem;
		line-height: 1.4;
		color: #cfe2ff;
	}
	.msg.error {
		color: #ffb0b0;
	}
	.msg.ok {
		color: #9af5c8;
		font-weight: 650;
	}

	.success {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.85rem;
		padding: 1rem 1.1rem;
		border-radius: 16px;
		border: 1px solid rgba(90, 240, 180, 0.3);
		background: rgba(8, 26, 22, 0.5);
		backdrop-filter: blur(16px);
	}
	.share-lead {
		margin: 0;
		font-size: 0.86rem;
		line-height: 1.45;
		color: rgba(200, 230, 255, 0.82);
		max-width: 40ch;
	}
	/* the personal share link — the artifact people actually pass around */
	.share-link {
		display: flex;
		gap: 0.4rem;
		width: 100%;
		padding: 0.35rem;
		border-radius: 12px;
		border: 1px solid rgba(140, 190, 255, 0.28);
		background: rgba(6, 12, 28, 0.6);
	}
	.share-url {
		flex: 1;
		min-width: 0;
		border: 0;
		background: transparent;
		color: #bfe0ff;
		font: inherit;
		font-size: 0.82rem;
		padding: 0.5rem 0.6rem;
		outline: none;
		text-overflow: ellipsis;
	}
	.copy-btn {
		flex: 0 0 auto;
		border: 0;
		border-radius: 9px;
		padding: 0.5rem 0.9rem;
		font: inherit;
		font-weight: 700;
		font-size: 0.82rem;
		color: #04122b;
		cursor: pointer;
		background: linear-gradient(135deg, #5fd0ff, #7b9bff);
		transition: filter 0.2s ease;
	}
	.copy-btn:hover {
		filter: brightness(1.08);
	}
	.share-actions {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
		justify-content: center;
	}
	/* X share — the highest-signal share for a dev launch */
	.x-btn {
		display: inline-flex;
		align-items: center;
		gap: 0.45rem;
		border: 1px solid rgba(220, 235, 255, 0.45);
		border-radius: 12px;
		padding: 0.6rem 1.1rem;
		background: rgba(245, 248, 255, 0.95);
		color: #0a0f1a;
		font: inherit;
		font-weight: 700;
		cursor: pointer;
		transition:
			transform 0.2s ease,
			filter 0.2s ease;
	}
	.x-btn:hover {
		transform: translateY(-1px);
		filter: brightness(0.97);
	}
	.ref-count {
		margin: 0.1rem 0 0;
		font-size: 0.88rem;
		color: #9af5c8;
		font-weight: 600;
	}
	.ref-count strong {
		color: #eaffe9;
	}
	.ref-count.muted {
		color: rgba(190, 212, 245, 0.62);
		font-weight: 500;
	}
		.share-btn {
			border: 1px solid rgba(140, 190, 255, 0.3);
			border-radius: 12px;
			padding: 0.6rem 1.1rem;
		background: rgba(20, 30, 60, 0.5);
		color: #dceaff;
		font: inherit;
		font-weight: 650;
		cursor: pointer;
		transition:
				transform 0.2s ease,
				border-color 0.2s ease;
		}
		.share-btn:hover {
			transform: translateY(-1px);
			border-color: rgba(120, 200, 255, 0.65);
		}

	.meta {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: center;
		gap: 0.5rem;
		margin-top: 1rem;
		font-size: 0.84rem;
		color: rgba(190, 212, 245, 0.72);
	}
	.meta strong {
		color: #eaf3ff;
	}
	.date {
		color: #8fe9ff;
		font-weight: 650;
	}
	.dot {
		opacity: 0.4;
	}
	.src {
		color: #bcd4f5;
		text-decoration: underline;
		text-underline-offset: 2px;
	}

		.scroll-hint {
			margin: 0;
			font-size: 0.72rem;
			letter-spacing: 0.18em;
			text-transform: uppercase;
			color: rgba(150, 180, 220, 0.4);
		}

		@media (max-width: 760px) {
			/* stronger, taller scrim — particles fill the whole portrait screen */
			.scrim {
				background:
					radial-gradient(140% 60% at 50% 38%, transparent 22%, rgba(2, 3, 10, 0.6) 100%),
					linear-gradient(180deg, rgba(2, 3, 10, 0.5) 0%, rgba(2, 3, 10, 0.12) 30%, rgba(2, 3, 10, 0.55) 62%, rgba(2, 3, 10, 0.92) 100%);
		}
		/* dim only the ACTIVE engine canvas. The fallback canvas is rendered after
		   the GPU canvas, so forcing both visible can cover the real WebGPU show. */
		:global(.raw-vestige-engine[data-mode='webgpu'] .gpu-canvas),
		:global(.raw-vestige-engine[data-mode='fallback'] .fallback-canvas) {
			opacity: 0.82;
		}
		:global(.raw-vestige-engine[data-mode='webgpu'] .fallback-canvas),
		:global(.raw-vestige-engine[data-mode='fallback'] .gpu-canvas) {
			opacity: 0;
		}
		.overlay {
			justify-content: flex-end;
			gap: 1.4rem;
			padding-bottom: clamp(2rem, 7vh, 3.4rem);
		}
		.wordmark-wrap {
			width: 100%;
		}
		.tagline,
		.subline {
			text-shadow: 0 2px 14px rgba(2, 3, 10, 0.9);
		}
		.subline {
			max-width: 34ch;
		}
		.field {
			flex-direction: column;
		}
		.join-btn {
			width: 100%;
			padding: 0.95rem 1.25rem;
		}
		.email {
			padding: 0.95rem 0.9rem;
		}
		.meta {
			text-shadow: 0 1px 10px rgba(2, 3, 10, 0.9);
		}
		/* gentler implode on phones (smaller deltas, no per-frame DOM blur which is
		   expensive on mobile); the WebGPU background carries the spectacle. */
		.hero-block {
			transform: scale(calc(1 - var(--burst) * 0.08)) translateZ(calc(var(--burst) * -160px));
		}
		.cta-block {
			transform: scale(calc(1 - var(--burst) * 0.05));
			translate: 0 calc(var(--burst) * 8px);
		}
		.hero-block .wordmark-wrap,
		.cta-block .tagline {
			filter: none;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		/* hard OFF — a periodic full-overlay implode is a WCAG 2.3.3 vestibular
		   trigger; never distort for users who opt out. */
		.topbar,
		.hero-block,
		.cta-block {
			transition: none;
			transform: none !important;
			translate: none !important;
			filter: none !important;
			opacity: 1 !important;
		}
		.hero-block .wordmark-wrap,
		.cta-block .tagline {
			filter: none !important;
		}
		.wild {
			animation: none;
		}
	}
	</style>
