<div align="center">

# VESTIGE v6.2
### The Ring-0 Autonomic Execution Kernel & World's First Causal Proof Engine for AI Agents

**Think before doing. Prove before saving. Memory is root cause, not search.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Binary Size](https://img.shields.io/badge/Binary_Size-~7.8MB-brightgreen.svg)](https://github.com/samvallad33/vestige)
[![Embeddings](https://img.shields.io/badge/Embeddings-0_Vectors-orange.svg)](#-why-we-ripped-out-vectors)
[![IPC Latency](https://img.shields.io/badge/IPC_Latency-<0.2ms-blueviolet.svg)](#-the-autonomic-dual-tier-architecture)
[![Gate Rules](https://img.shields.io/badge/Gate_Rules-114_Deterministic-red.svg)](#-114-deterministic-pre-execution-rules)

[**Quick Start**](#-quick-start-30-seconds) · [**Why We Ripped Out Vectors**](#-why-we-ripped-out-vectors) · [**S.T.R.A.T.A. Engine**](#️-strata--the-worlds-first-causal-proof-engine) · [**The 6 MCP Verbs**](#️-the-6-verb-mcp-surface) · [**Reflex Hooks**](#-the-4-autonomic-reflex-arcs-0-turn-cognitive-injection) · [**Vestige Operator**](#-vestige-operator)

</div>

---

## 🚨 The Industry Lie: Memory Is Not a Search Engine

Every other AI memory tool on the market is a passive scrapbook built on **unordered vector soup**. Every other AI security tool is a **stateless bouncer** with complete amnesia.

When an autonomous coding agent breaks a build at 3:00 PM because a configuration flag was silently flipped at 9:00 AM, **Vector RAG fails 100% of the time**:
1. The 3:00 PM stack trace (`ConnectionPoolExhaustedException`) shares **0% semantic similarity** with the 9:00 AM config edit (`max_idle_time = 0`).
2. Cosine similarity retrieves plausible-looking decoys instead of the causal ancestor.
3. The agent guesses, edits the wrong file, fails again, and enters a **$50,000 infinite retry loop**.

**Vestige v6.2 ends the guessing game.** We ripped out ONNX, deleted vector embeddings, deleted keyword search, and deleted SQLite. In their place is a **~7.8 MB, 100% offline-verified Rust execution kernel** that unifies pre-execution security gating, autonomic lifecycle hooks (`reflex.sock`), and backward causal credit assignment into a single closed-loop nervous system.

> **An Independent Study of 757 Agent Memory Systems ([Agent Memory Atlas](https://github.com/samvallad33/vestige)):**
> *"Every system in this atlas builds the ascent (extraction and retrieval). Almost none completes the descent (correction and forgetting)."*
> **Vestige is built for the descent.**

---

## ⚡ Quick Start (30 Seconds)

One command installs the `vestige-mcp` Rust daemon, wires the 4 autonomic `Reflex` hooks into your IDE (Claude Code, Cursor, Codex, Windsurf, Gemini CLI), and replays your last 30 days of agent transcripts against our **114-rule gate** to show every destructive command or failure loop Vestige would have stopped.

**Operator** is the guard for your agent's tool calls. `npx @vestige/init --guard` installs 102 STOP rules, OP-000 through OP-114, with OP-086–OP-097 and OP-100 reserved. It stops force pushes, dropped tables, recursive deletes, production deploys, credential reads and the rest of that closed list before they run, even when the command is disguised. Classification is deterministic: no model call and no network call. Operator Lite (numbered rules OP-000 through OP-025) is pinned beside the gate and judges every call too. An existing `~/.operator/gate/operator-gate.py` is left in place.

```bash
npx @vestige/init
```

Verify your live cryptographic store and 6-tool surface at any time:

```bash
vestige precedent --query verify_log
```

---

## 🏛️ S.T.R.A.T.A. — The World's First Causal Proof Engine

The 102-rule gate is in this repository and is what `--guard` installs. See [operator/README.md](operator/README.md). `board`, `letter`, `almost-forgot`, `onboard` and `briefing` load companion modules when those files sit next to the gate. [Buy Vestige Operator](https://payhip.com/b/d4xvu) is the owner's copy of those companions: $149 once, every later version included, and a 30-day refund.

Underneath Vestige sits **S.T.R.A.T.A.** (*Sequential Topological Reasoning for Autonomous Transaction Authorization*)—an append-only, cryptographic execution ledger that replaces fuzzy similarity with mathematical provenance:

* **Sequential (`BLAKE3` + `Ed25519`):** Every memory write, tool call, and gate stop is recorded as an immutable frame carrying the `BLAKE3` hash of the frame before it. Nightly segments are sealed with `Ed25519` signatures.
* **Topological (`causal_walk`):** Memories are indexed by exact handles (`id`, `tag`, `commit:<sha>`, hunk-level `touched` file edges, and interned stack-trace symbols). Free-text guessing is refused—and on refusal, **Handle Discovery** automatically parses your query to return exact candidate handles found in your words.
* **Reasoning (`FSRS-6` + `Retroactive Salience Backfill`):**
  * **FSRS-6 Decay (Precision Forgetting):** Unverified, unreferenced session noise automatically decays out of working memory so context windows stay razor-sharp.
  * **Retroactive Salience Backfill (Runtime Credit Assignment):** When a downstream failure or high-salience constraint lands, Vestige immediately walks the topological DAG backward on save, rescues the quiet upstream root-cause memory from FSRS-6 decay, boosts its salience, and writes a reversible cryptographic receipt.
* **Autonomous (`reflex.sock`):** A zero-lock Unix Domain Socket (`~/.vestige/reflex.sock`) bridges external shell hooks directly into the hot in-memory Strata graph in **<0.2ms**—eliminating `.serve.lock` contention across multiple concurrent IDE windows.
* **Transaction Authorization (`Operator` + `prove`):** Every agent action is treated as an untrusted transaction that must pass **114 pre-execution rules** before running and can be verified post-fix via deterministic git-bisect test witnesses (`causal_walk(action="prove")`).

---

## 🧠 The Autonomic Dual-Tier Architecture

Standard MCP servers wait passively for an LLM to remember to call a tool. Vestige operates like a human nervous system: **Reflex Hooks** act as the spinal cord (firing automatically in **0 LLM turns** and **<0.2ms**), while the **6 MCP Tools** act as the deliberate prefrontal cortex.

```text
                         ┌────────────────────────────────────────┐
                         │   Model Prefrontal Cortex (6 Verbs)    │
                         │   session · remember · recall ·        │
                         │   causal_walk · curate · precedent     │
                         └──────────────────┬─────────────────────┘
                                            │ stdio / JSON-RPC
                                            ▼
┌──────────────────────┐  Unix Domain Socket   ┌────────────────────────────────────────┐
│  Reflex Hooks Engine │ ◄───────────────────► │       vestige-mcp Server Daemon        │
│    (Spinal Cord)     │  ~/.vestige/reflex.sock│  ┌───────────────────────────────────┐ │
│                      │   (<0.2ms Zero-Lock)  │  │ In-Memory Strata Graph (Hot RAM)  │ │
│ • SessionStart       │                       │  │ • FSRS-6 scheduler state          │ │
│ • PreToolUse (114)   │                       │  │ • Exact Handle & Symbol Index     │ │
│ • PostToolUse (RSB)  │                       │  │ • Causal DAG & Backfill Engine    │ │
│ • PreCommit (Prove)  │                       │  └─────────────────┬─────────────────┘ │
└──────────────────────┘                                            │ Exclusive Write Lock
                                                                    ▼
                                                       ┌────────────────────────┐
                                                       │ Append-Only Strata Log │
                                                       │ (BLAKE3 + Ed25519)     │
                                                       └────────────────────────┘
```

---

## ⚡ The 4 Autonomic Reflex Arcs (0-Turn Cognitive Injection)

1. **`Reflex::SessionStart` (Active Handle Map Injection):**
   On boot, resume, or context compaction, Vestige queries the local Git index against the hot Strata graph and injects the **Top 5 Hot Entity Handles** (`id`, `tag`, `last_touched_file`, and FSRS-6 retrievability) plus pinned repo rules. Your agent starts Turn 1 already holding the exact handles it needs.
2. **`Reflex::PreToolUse` (114 Gate Rules + Landmine Interception):**
   Before `Bash`, `Edit`, or `Write` executes, Vestige runs **114 deterministic security rules** (catching obfuscated shell commands, `r''m`, `$IFS`, ANSI-C quoting, base64 pipes, destructive SQL/git, and multi-step credential exfiltration) **AND** checks if the target file is linked to an un-recalled `failure` node in Strata. If a landmine exists, Reflex warns the agent with the exact `handle` (`mem-xxxx`) *before* it repeats a mistake.
3. **`Reflex::PostToolUse` (Auto-Failure Capture + Inline Backfill):**
   Whenever a command exits with `exit_code != 0`, Reflex automatically records a signed `failure` frame in Strata, triggers **Retroactive Salience Backfill** across upstream touched nodes, surfaces any faded `forgotten_lesson` (`retrievability < 0.5`), and appends the exact `causal_walk` command directly to the tool output.
4. **`Reflex::PreCommit` (Proof-Carrying PR Enforcement):**
   When an agent runs `git commit` or `gh pr create` after a logged failure in the current trajectory, Reflex verifies that a cryptographically signed `prove` bisect witness exists for the fix.

---

## 🛠️ The 6-Verb MCP Surface

Most MCP servers drown agents in 18–30 overlapping tool schemas—burning ~2,500 prompt tokens per turn and confusing smaller models. In v6.2, Vestige compresses its entire capability set into **6 flat-JSON cognitive verbs** (while preserving internal backward-compatible aliases for all 18 legacy tool names):

| MCP Tool | Cognitive Intent | Key Actions & Capabilities |
| :--- | :--- | :--- |
| **`1. session`** | *Boot, Plan & Project* | `action: "start" \| "status" \| "set_intention" \| "check_intentions" \| "complete_intention" \| "project"`<br>Manages cross-session prospective intentions, store health/stats, and compiles lock-free `CLAUDE.md` / `AGENTS.md` context fences directly from RAM. |
| **`2. remember`** | *The Single Gated Write* | Flat parameters: `content`, `tags`, `repo_path`, `files`, `is_decision`.<br>Automatically anchors codebase records when `repo_path`/`files` are present, runs inline **Retroactive Salience Backfill** (returning `backfilled_handles`), and auto-updates `CLAUDE.md` when tagged `"pinned"`. |
| **`3. recall`** | *Exact-Handle Read* | Flat parameters: `handle`, `detail_level`, `query`.<br>Resolves exact IDs, prefixes, tags, or repo context (`get_context`). Returns structured `discovered_handles: [...]` if free text is passed so the agent self-corrects on the immediate next turn. |
| **`4. causal_walk`** | *Root Cause & Bisect Proof* | `action: "walk" \| "prove"`, `from_handle`, `good_ref`, `bad_ref`, `test_cmd`.<br>• `"walk"`: Auto-gathers GitHub issue/PR windows, walks hunk-level touched and git log -L ancestry backward, and auto-attaches faded `forgotten_lesson` nodes.<br>• `"prove"`: Executes a frozen-protocol git bisect against `test_cmd` and writes a signed Strata witness card. |
| **`5. curate`** | *Graph & State Mutation* | `action: "promote" \| "demote" \| "edit" \| "link" \| "bridge" \| "connect_identities" \| "suppress"`<br>Adjusts FSRS-6 stability weights, wires explicit causal links, bridges cross-project structural similarities (`ghostlink`), joins shared entity identities, or permanently suppresses invalid records with a signed audit receipt. |
| **`6. precedent`** | *Receipts & Integrity Audit* | `query: "stops" \| "receipt" \| "replay" \| "verify_log" \| "scan_secrets" \| "selftest"`<br>Inspects Operator gate stops, replays write receipts, scans for credential-shaped strings, and runs live BLAKE3 + Ed25519 verification (`strata-verify`) without stopping the server. |

---

## 🏎️ David vs. Goliath: Making Local 8B Models Overpower Cloud Giants

Cloud memory wrappers charge a ~6,900-token tax per turn to stuff noisy vector chunks into frontier models. Because Vestige v6.2 pushes execution safety, failure detection, and backward credit assignment into a sub-millisecond local Rust daemon:

* **Zero Context Stuffing:** Exact handles and FSRS-6 decay keep active context lean (<200 tokens of memory overhead), preventing small-model attention degradation.
* **Flat 6-Verb Schemas:** Eliminates tool-selection hallucinations in 8B–9B local coding models (Qwen3-Coder-8B, Llama-3.1-8B, Gemma 4).
* **$0.00 Cloud Token Tax:** A release build of `vestige-mcp` is verified in CI to never link `reqwest`—guaranteeing zero network calls, zero telemetry, and zero API fees.

Run the 3-lane ablation benchmark (Mode A: Baseline vs. Mode B: Forward-Only FSRS-6 vs. Mode C: Full Vestige v6.2 with RSB + Reflex) and generate an Ed25519-sealed proof log:

```bash
./benchmarks/run-all-benchmarks.sh --ablation
```

---

## 🖥️ Host-Boundary CLI & Local WebGPU Dashboard

Because `vestige-mcp` routes concurrent CLI and hook calls through `~/.vestige/reflex.sock`, you never hit `.serve.lock` collisions while an agent is active. The host CLI is stripped down to pure boundary operations:

```bash
vestige serve                         # Start the MCP server + reflex.sock IPC daemon
vestige backup <target-dir>           # Seal active Strata segment with Ed25519 & write owner-only backup
vestige dashboard                     # Launch the loopback-hardened local UI on http://127.0.0.1:3927
vestige update                        # Verify .sha256 release digests and update binary
```

---

## 🛡️ Vestige Operator

* **Vestige Core & Operator Lite (AGPL-3.0-only):** Free and open-source forever. Includes the full Strata engine, the 6 MCP tools, the 4 Reflex hooks, and the built-in pre-execution rule gate.
* **Vestige Operator ($149 One-Time License):** Adds custom repository law authoring (`vestige.toml`), team-wide proof-carrying PR enforcement (`actions/proof-carrying-pr`), multi-step exfiltration chain governors, and the weekly executive stop-board.

---

## 📚 Technical Reports & Documentation

* **TR-2026-01: Retroactive Salience Backfill & Silent Rotation Benchmark** — How inline runtime credit assignment + FSRS-6 solves delayed root-cause attribution.
* **Configuration & Security Model** — Loopback DNS-rebinding defenses, Sec-Fetch-Site enforcement, BLAKE3 frame chaining, and reflex.sock permissions (0o600).
* **Getting Started Guide & FAQ** — [GETTING-STARTED.md](docs/GETTING-STARTED.md) · [FAQ.md](docs/FAQ.md) · [UPGRADING.md](docs/UPGRADING.md)

Built from first principles in Rust by Sam Valladares.

**Zero vectors. Zero cloud tax. 100% causal proof.**
