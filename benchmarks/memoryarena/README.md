# MemoryArena harness

Vestige as a memory backend for
[MemoryArena](https://github.com/ZexueHe/MemoryArena) (arXiv:2602.16313), the
benchmark that scores memory inside an agent loop instead of as a retrieval
quiz.

**Status: preregistered, not run. There is no Vestige number on MemoryArena.**
The protocol, arms, metrics and decision rules are fixed in
[`docs/benchmarks/MEMORYARENA-PREREGISTRATION.md`](../../docs/benchmarks/MEMORYARENA-PREREGISTRATION.md).
Read it before running anything, and read
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) before quoting anything.

## Quick start (no MemoryArena checkout, no API key)

```sh
cargo build -p vestige-mcp
python3 benchmarks/memoryarena/smoke_test.py
```

The smoke test drives the adapter against the real binary and asserts the
properties the run depends on: a stored fact comes back for the right
question, a second task never sees the first task's memories, the hits carry a
real `semanticScore` (embeddings, not the keyword fallback), a 3,000-character
LaTeX-heavy prompt wraps without error, an upstream-shaped
`## Task ... ## solution ...` entry is retrievable by a later related task, and
every add and wrap landed in the sidecar log with byte counts.

## Files

| File | Purpose |
| --- | --- |
| `vestige_memory_system.py` | The adapter. `VestigeMemorySystem` (add_chunk / wrap_user_prompt over MCP stdio, one shared server, one scope per task, readiness guard, JSONL sidecar) and `NoMemorySystem` (the floor arm). Self-contained so it can be copied into a MemoryArena clone. |
| `install.py` | Installs the adapter into a pinned MemoryArena clone: copies the module, registers `vestige` and `none` in `memory/server.py`, copies the run configs. Anchored edits, idempotent, refuses an unpinned HEAD. |
| `configs/` | Run configs mirroring upstream's field for field: `math_vestige`, `phys_vestige`, `math_none`, `phys_none`, and `phys_bm25` (upstream ships the math one only). |
| `smoke_test.py` | The check above. Exit 1 on any failure. |
| `analyze.py` | The preregistered analysis: per-arm PS and SR with n, paired exact sign test against the reference arm, sidecar confound summary. |
| `MEMORYARENA.lock.json` | Pinned upstream code and dataset revisions, row counts, metric definitions. |
| `results/` | Checked-in run outputs, once a run exists. Empty until then. |

Standard library only. No `pip install` for anything in this directory; the
upstream environment is upstream's business.

## How the adapter fits upstream

MemoryArena's memory server (`memory/server.py`) constructs one backend object
per task and calls two methods: `add_chunk(chunk)` after each agent step with
the agent's own memory entry, and `wrap_user_prompt(prompt)` before each step,
expecting a string with a `<memory_context>` block followed by `User: <prompt>`.
`install.py` adds a `name == "vestige"` branch that constructs
`VestigeMemorySystem(user_id=req.user_id)`, so every task gets its own Vestige
scope while sharing one `vestige-mcp` process for the whole run.

Environment variables the adapter reads are documented at the top of
`vestige_memory_system.py`. The two you will set for a real run are
`VESTIGE_MCP_BINARY` (a release build) and `VESTIGE_ARENA_DATA_DIR` (a fresh
directory per task family).

## Running the benchmark

The exact procedure, arm order and analysis call are in section 8 of the
preregistration. Do not improvise on them; if something has to change, it is
written into the Amendments section of that file first.
