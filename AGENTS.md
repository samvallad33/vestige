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

# ⚡ VESTIGE COGNITIVE PROTOCOL (MANDATORY FOR ALL AGENTS)

You are connected to Vestige (`vestige-mcp`), a deterministic local safety kernel and causal memory ledger (Strata). While the autonomic Reflex hooks (`~/.vestige/reflex.sock`) run automatically in the background, you are strictly required to drive the 6 Vestige MCP tools (`session`, `remember`, `recall`, `causal_walk`, `curate`, `precedent`) as your primary cognitive loop.

Never rely on raw context-window guessing, grep/rg fishing, or unrecorded state mutations when a Vestige tool applies.

## 1. Boot Sequence (First Action of Every Turn/Session)
- **MANDATORY FIRST CALL**: At the start of any task or session, before reading files or running shell commands, call `session(action="start")` to boot lifecycle state, check active intentions, and load projected rules.
- **Pre-Flight Handle Discovery**: Before modifying any module or debugging a subsystem, call `recall` (using exact handles or discovered_handles) to pull existing architectural invariants and prior decisions.

## 2. Debugging & Failure Recovery (causal_walk + prove)
Whenever a test fails, a build breaks, a runtime error appears, or an unexpected regression occurs:
- **STOP GUESSING**: Do NOT immediately edit code or run blind grep searches based on stack-trace text.
- **Step 1 — Trace the Root Cause**: Immediately call `causal_walk` pointed at the symptom or failing test (`from_handle` or `start_points`) to walk backward across recorded Strata edges (symptom -> failing test -> tool call -> git commit -> memory write) and surface forgotten lessons.
- **Step 2 — Prove the Cause**: When a suspect commit or historical mutation is identified, invoke `causal_walk(action="prove")` to run a deterministic bisect witness and verify the root cause before writing a fix.

## 3. Gated Writes & Retroactive Salience (remember + curate)
Whenever you make a non-trivial architectural decision, discover a root cause, or ship a verified fix:
- **Anchor Decisions to Code**: Call `remember(action="save", ...)` to perform a gated write into Strata with explicit codebase anchoring and inline Retroactive Salience Backfill (RSB) linking the fix to the failure.
- **Maintain the Descent (Zero Stale Noise)**: If a new change invalidates or replaces an older assumption, you MUST call `curate` to supersede, demote, or bridge the outdated memory node so future subagents never converge on stale beliefs.

## 4. Security & Operator Compliance (precedent)
- **Never Bypass the Gate**: Every shell command, git operation, and memory write is gated by the 114-rule Operator. If a tool call is blocked by Reflex or Operator, do not attempt workarounds or shell flags to bypass it.
- **Inspect the Receipt**: Immediately call `precedent` to inspect the Operator guard stop, read the BLAKE3/Ed25519 receipt, or run secret scanning before proposing a compliant alternative.

## 5. Hard Execution Constraints (Anti-Drift Checklist)
- **No Unanchored Fixes**: A bug fix is NOT complete until `causal_walk` has traced the cause and `remember` has anchored the resolution in Strata.
- **No Silent Supersession**: Never leave contradictory architectural facts active; always run `curate` when changing an invariant.
- **Tool Priority**: Always prefer `recall` and `causal_walk` over raw filesystem scanning when investigating "why" code was written a certain way.

---

# Vestige Agent Guidance

This file gives coding agents project-specific context without relying on private local files, personal operating notes, or mandatory background hooks.

## Project Shape

Vestige is the Causal Proof Engine and the operating system for AI agents. It
is written in Rust, with a SvelteKit dashboard embedded into the release
binary. Its kernel is Strata, an append-only, hash-chained, signed log stored
on the user's machine. `vestige-mcp` serves that log to agents over MCP. The
core product promise is:

- user-owned data, stored locally by default
- MCP-native integration with coding agents
- zero vectors, zero string matching and zero RAG in the engine (recall,
  ranking, pairing and explanation): a record is found by an exact handle (an
  id, a unique id prefix of 8 or more characters, or an exact tag) and linked
  only by an edge the log recorded
- every output carries its proof: a memory id, an edge path or a receipt
- a call the Strata log cannot honor says so with a reason code
  (`unavailable_in_4_0`, `similarity_disabled`, `pending_strata`) instead of
  returning zeros
- six advertised MCP tools: `session`, `remember`, `recall`, `causal_walk`,
  `curate`, and `precedent`. Inspect `tools/list` and
  `session(action="status", view="tools", tool="<name>")` for the installed
  action schemas. Runtime support is determined by the active store owner.

## Working Rules

- Prefer source evidence over memory. Use `rg`, tests, and nearby code before
  making claims about behavior.
- Keep release changes scoped. Do not rewrite unrelated modules during a
  version/tag cleanup unless the release gate requires it.
- Preserve local-first behavior. Heavy models, Sanhedrin-style verifier hooks,
  and preflight automation must remain optional.
- Treat deletion semantics carefully. A Strata log is append-only, so nothing
  is erased. `purge`, `memory` actions `purge` and `delete`, and
  `delete_knowledge` return `unavailable_in_4_0`. `suppress` hides a record
  from every read, keeps its bytes, and cannot be undone. Never describe
  either as erasure.
- Treat exact lookup semantics carefully. Do not add embeddings, vectors,
  fuzzy matching, or semantic similarity to recall, ranking, pairing, or
  explanation. A handle still resolves exactly: an id, a unique id prefix, or
  an exact, case-sensitive tag. Free text on `recall` and the memory hook is
  exact-token BM25 over stored text, not a meaning match. Env vars, paths,
  UUIDs and code identifiers must reach the log unchanged.

## Common Checks

Run the narrowest check that covers the change, then run the release gates
before tagging:

```sh
cargo check -p vestige-mcp
cargo test --workspace --no-fail-fast
cargo clippy --workspace -- -D warnings
pnpm --filter @vestige/dashboard check
pnpm --filter @vestige/dashboard build
```

On `feat/vestigeos-revival`, `cargo check -p vestige-mcp` is mandatory before
every commit that touches dashboard routes, handlers, or MCP tools — frontend
gates alone do not compile Rust and previously hid a missing `/api/backfill`
handler.

For documentation-only changes, at minimum run:

```sh
git diff --check
```

## Documentation

- User setup: `README.md`
- Codex setup and agent templates: `docs/integrations/codex.md`
- Cursor setup: `docs/integrations/cursor.md`
- Storage and backup behavior: `docs/STORAGE.md`
- Cognitive Sandwich and optional verifier hooks: `docs/COGNITIVE_SANDWICH.md`
- Release history: `CHANGELOG.md`

## Public-Repo Hygiene

Do not commit private absolute paths, local agent memory paths, unpublished
planning files, real credentials, personal operating notes, or private repo
locations. Example environment variables in docs must be empty placeholders or
obviously fake examples.
