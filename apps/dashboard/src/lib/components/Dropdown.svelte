<script lang="ts" module>
	export interface DropdownOption {
		value: string;
		label: string;
		/** Optional color dot (e.g. node-type color) shown before the label. */
		color?: string;
		/** Optional small count/badge shown after the label. */
		badge?: string | number;
		/** Optional icon name from the Icon system. */
		icon?: IconName;
	}
</script>

<script lang="ts">
	import Icon, { type IconName } from './Icon.svelte';

	// ═══════════════════════════════════════════════════════════════════
	//  DROPDOWN — accessible, animated, themed select replacement.
	// ───────────────────────────────────────────────────────────────────
	//  Replaces the dead native <select>. Keyboard-navigable (↑/↓/Enter/Esc/
	//  Home/End/type-ahead), closes on outside click, animates open with a
	//  spring-ish scale, and themes to the cosmic palette. Makes filtering
	//  CLEAR: shows the current value, an icon, color dots, and counts.
	// ═══════════════════════════════════════════════════════════════════
	interface Props {
		options: DropdownOption[];
		value: string;
		label?: string;
		placeholder?: string;
		icon?: IconName;
		/** Width hint for the trigger; menu matches the trigger width. */
		class?: string;
		onChange?: (value: string) => void;
	}
	let {
		options,
		value = $bindable(),
		label,
		placeholder = 'Select…',
		icon,
		class: cls = '',
		onChange,
	}: Props = $props();

	let open = $state(false);
	let triggerEl = $state<HTMLButtonElement>();
	let menuEl = $state<HTMLDivElement>();
	let activeIndex = $state(-1);
	let typeAhead = '';
	let typeAheadTimer: ReturnType<typeof setTimeout>;

	let selected = $derived(options.find((o) => o.value === value));

	function select(opt: DropdownOption) {
		value = opt.value;
		onChange?.(opt.value);
		close();
		triggerEl?.focus();
	}

	function toggle() {
		open ? close() : openMenu();
	}

	function openMenu() {
		open = true;
		activeIndex = Math.max(0, options.findIndex((o) => o.value === value));
		requestAnimationFrame(() => {
			(menuEl?.querySelectorAll('[role="option"]')[activeIndex] as HTMLElement)?.scrollIntoView({
				block: 'nearest',
			});
		});
	}

	function close() {
		open = false;
		activeIndex = -1;
	}

	function onTriggerKey(e: KeyboardEvent) {
		if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			open ? move(1) : openMenu();
		} else if (e.key === 'ArrowUp') {
			e.preventDefault();
			open ? move(-1) : openMenu();
		} else if (e.key === 'Escape') {
			close();
		} else if (e.key === 'Home' && open) {
			e.preventDefault();
			activeIndex = 0;
		} else if (e.key === 'End' && open) {
			e.preventDefault();
			activeIndex = options.length - 1;
		} else if (open && e.key === 'Enter' && activeIndex >= 0) {
			e.preventDefault();
			select(options[activeIndex]);
		} else if (e.key.length === 1 && /\S/.test(e.key)) {
			// type-ahead
			if (!open) openMenu();
			clearTimeout(typeAheadTimer);
			typeAhead += e.key.toLowerCase();
			typeAheadTimer = setTimeout(() => (typeAhead = ''), 600);
			const idx = options.findIndex((o) => o.label.toLowerCase().startsWith(typeAhead));
			if (idx >= 0) activeIndex = idx;
		}
	}

	function move(delta: number) {
		const n = options.length;
		activeIndex = (activeIndex + delta + n) % n;
		requestAnimationFrame(() => {
			(menuEl?.querySelectorAll('[role="option"]')[activeIndex] as HTMLElement)?.scrollIntoView({
				block: 'nearest',
			});
		});
	}

	// Close on outside click while open.
	$effect(() => {
		if (!open) return;
		function onDocClick(e: MouseEvent) {
			if (!triggerEl?.contains(e.target as Node) && !menuEl?.contains(e.target as Node)) close();
		}
		document.addEventListener('click', onDocClick, true);
		return () => document.removeEventListener('click', onDocClick, true);
	});
</script>

<div class="dd {cls}">
	{#if label}
		<span class="dd-label">{label}</span>
	{/if}
	<button
		bind:this={triggerEl}
		type="button"
		class="dd-trigger"
		class:dd-open={open}
		aria-haspopup="listbox"
		aria-expanded={open}
		onclick={toggle}
		onkeydown={onTriggerKey}
	>
		{#if icon}
			<span class="dd-trigger-icon"><Icon name={icon} size={15} /></span>
		{/if}
		<span class="dd-value" class:dd-placeholder={!selected}>
			{#if selected?.color}
				<span class="dd-dot" style="background:{selected.color}"></span>
			{/if}
			{selected ? selected.label : placeholder}
		</span>
		<span class="dd-chevron" class:dd-chevron-open={open}><Icon name="chevron" size={14} /></span>
	</button>

	{#if open}
		<div
			bind:this={menuEl}
			class="dd-menu glass-panel"
			role="listbox"
			tabindex="-1"
			aria-label={label ?? placeholder}
		>
			{#each options as opt, i (opt.value)}
				<button
					type="button"
					role="option"
					aria-selected={opt.value === value}
					class="dd-option"
					class:dd-active={i === activeIndex}
					class:dd-selected={opt.value === value}
					onmouseenter={() => (activeIndex = i)}
					onclick={() => select(opt)}
				>
					{#if opt.icon}<span class="dd-opt-icon"><Icon name={opt.icon} size={15} /></span>{/if}
					{#if opt.color}<span class="dd-dot" style="background:{opt.color}"></span>{/if}
					<span class="dd-opt-label">{opt.label}</span>
					{#if opt.badge !== undefined}<span class="dd-badge">{opt.badge}</span>{/if}
					{#if opt.value === value}<span class="dd-check"><Icon name="sparkle" size={12} /></span>{/if}
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.dd {
		position: relative;
		display: inline-flex;
		flex-direction: column;
		gap: 0.3rem;
	}
	.dd-label {
		font-size: 0.65rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-muted);
		padding-left: 0.15rem;
	}
	.dd-trigger {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.55rem 0.7rem 0.55rem 0.8rem;
		min-width: 9rem;
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(99, 102, 241, 0.12);
		border-radius: 0.75rem;
		color: var(--color-text);
		font-size: 0.8rem;
		font-family: inherit;
		cursor: pointer;
		backdrop-filter: blur(8px);
		transition:
			border-color 0.2s ease,
			background 0.2s ease,
			box-shadow 0.2s ease;
	}
	.dd-trigger:hover {
		border-color: rgba(99, 102, 241, 0.3);
		background: rgba(255, 255, 255, 0.05);
	}
	.dd-trigger:focus-visible,
	.dd-trigger.dd-open {
		outline: none;
		border-color: rgba(99, 102, 241, 0.5);
		box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12);
	}
	.dd-trigger-icon {
		color: var(--color-synapse-glow);
		display: inline-flex;
	}
	.dd-value {
		flex: 1;
		text-align: left;
		display: inline-flex;
		align-items: center;
		gap: 0.45rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.dd-placeholder {
		color: var(--color-muted);
	}
	.dd-dot {
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 50%;
		flex-shrink: 0;
		box-shadow: 0 0 6px currentColor;
	}
	.dd-chevron {
		color: var(--color-dim);
		display: inline-flex;
		transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
	}
	.dd-chevron-open {
		transform: rotate(180deg);
		color: var(--color-synapse-glow);
	}

	.dd-menu {
		position: absolute;
		top: calc(100% + 0.4rem);
		left: 0;
		z-index: 60;
		min-width: 100%;
		max-height: 18rem;
		overflow-y: auto;
		padding: 0.35rem;
		border-radius: 0.85rem;
		transform-origin: top center;
	}
	@media not (prefers-reduced-motion: reduce) {
		.dd-menu {
			animation: dd-pop 0.18s cubic-bezier(0.34, 1.56, 0.64, 1);
		}
		@keyframes dd-pop {
			from {
				opacity: 0;
				transform: translateY(-6px) scale(0.96);
			}
			to {
				opacity: 1;
				transform: translateY(0) scale(1);
			}
		}
	}
	.dd-option {
		width: 100%;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.6rem;
		border: none;
		background: transparent;
		color: var(--color-dim);
		font-size: 0.8rem;
		font-family: inherit;
		text-align: left;
		border-radius: 0.6rem;
		cursor: pointer;
		transition: background 0.12s ease, color 0.12s ease;
	}
	.dd-active {
		background: rgba(99, 102, 241, 0.14);
		color: var(--color-text);
	}
	.dd-selected {
		color: var(--color-synapse-glow);
	}
	.dd-opt-icon {
		color: var(--color-dim);
		display: inline-flex;
	}
	.dd-active .dd-opt-icon {
		color: var(--color-synapse-glow);
	}
	.dd-opt-label {
		flex: 1;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.dd-badge {
		font-size: 0.65rem;
		font-variant-numeric: tabular-nums;
		color: var(--color-muted);
		background: rgba(255, 255, 255, 0.05);
		padding: 0.05rem 0.4rem;
		border-radius: 0.4rem;
	}
	.dd-check {
		color: var(--color-synapse-glow);
		display: inline-flex;
	}
</style>
