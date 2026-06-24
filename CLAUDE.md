# 🚀 MANDATE #0 — MAXIMUM AMBITION. NO HEDGING. (read this first, every time)

This is the standing default for ALL work on Vestige, before anything else below.

- **Assume maximum ambition.** Build the most exhaustive, jaw-dropping, correct
  version — not the safe "good enough" one. The ceiling is "nobody else has
  this," not "this works." Default to the winning move, not the cautious one.
- **Scour before you settle.** When unsure or stuck, research the real frontier
  (web, source, the actual running app) before acting. Never accept the first
  adequate answer when a deeper one exists. Verify, don't assume.
- **No hedging.** Lead with the best path and a clear recommendation. Forbidden:
  "this is probably too risky," "temper expectations," "good enough," "maybe try
  the easier one." Risks get their own honest section — never used to shrink the
  target.
- **Show proof.** Verify changes in the real running app and share the evidence
  (screenshots, test output, gate results) — don't claim done without it.
- **Protect what's flawless, detonate what isn't.** Treat finished, loved work as
  load-bearing (don't break it); push everything else past where any other dev
  would stop.

Origin: Sam, Jun 22 2026 — the overnight session that turned the dashboard +
Memory Cinema from "alive" into a category-of-one particle journey. The depth
only happened because the bar was set to maximum. Make that the default, not the
exception.

---

# Vestige Agent Guidance

This file is intentionally safe for the public repository. It gives coding
agents project-specific context without relying on private local files,
personal operating notes, or mandatory background hooks.

## Project Shape

Vestige is a local-first MCP memory server written in Rust, with a SvelteKit
dashboard embedded into the release binary. The core product promise is:

- user-owned memory stored locally by default
- MCP-native integration with coding agents
- retrieval and memory lifecycle behavior informed by cognitive science
- explicit tools for search, review, suppression, purge, graph exploration,
  contradiction inspection, and maintenance

## Working Rules

- Prefer source evidence over memory. Use `rg`, tests, and nearby code before
  making claims about behavior.
- Keep release changes scoped. Do not rewrite unrelated modules during a
  version/tag cleanup unless the release gate requires it.
- Preserve local-first behavior. Heavy models, Sanhedrin-style verifier hooks,
  and preflight automation must remain optional.
- Treat deletion semantics carefully. `purge` must remove content and
  embeddings, while retaining only content-free audit tombstones.
- Treat exact lookup semantics carefully. Env vars, paths, UUIDs, quoted
  strings, and code identifiers should not be distorted by semantic expansion.

## Common Checks

Run the narrowest check that covers the change, then run the release gates
before tagging:

```sh
cargo test --workspace --no-fail-fast
cargo clippy --workspace -- -D warnings
pnpm --filter @vestige/dashboard check
pnpm --filter @vestige/dashboard build
```

For documentation-only changes, at minimum run:

```sh
git diff --check
```

## Documentation

- User setup: `README.md`
- Claude-specific templates: `docs/CLAUDE-SETUP.md`
- Storage and sync behavior: `docs/STORAGE.md`
- Cognitive Sandwich and optional verifier hooks: `docs/COGNITIVE_SANDWICH.md`
- Release history: `CHANGELOG.md`

## Public-Repo Hygiene

Do not commit private absolute paths, local agent memory paths, unpublished
planning files, real credentials, personal operating notes, or private repo
locations. Example environment variables in docs must be empty placeholders or
obviously fake examples.
