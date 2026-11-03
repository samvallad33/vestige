# Upgrading to Vestige v6.2.0

This guide covers historical transitions from v3 and v4 to the v6.2 Autonomic Execution Kernel.

## The 6 Cognitive Verbs

In v6.2, Vestige collapsed all 18 MCP tools into 6 distinct cognitive verbs:
1. `session`: Boot, intentions, status, health stats, and lock-free rule projection.
2. `remember`: Single gated write, codebase anchoring, and inline Retroactive Salience Backfill.
3. `recall`: Exact-handle retrieval with structured candidate discovery (`discovered_handles`).
4. `causal_walk`: Root-cause backwards walk, forgotten lesson surfacing, and deterministic bisect proof cards (`action="prove"`).
5. `curate`: Memory state mutations, promotions, demotions, edits, bridges, and identity connections.
6. `precedent`: Operator guard stops, receipts, replay, live BLAKE3/Ed25519 log verification, and secret scanning.

All legacy tool names remain supported internally via backwards-compatible dispatch aliases.

## Legacy Configuration Variables

The following environment variables configured the v3 SQLite engine and have no effect on a Strata log:

| Variable | Why it has no effect on a Strata log |
|----------|--------------------------------------|
| `FASTEMBED_CACHE_PATH`, `ORT_DYLIB_PATH`, `VESTIGE_DISABLE_VECTOR_SEARCH`, `HF_HOME` | There is no embedding model or vector index. |
| `VESTIGE_CONSOLIDATION_INTERVAL_HOURS` | The periodic consolidation pass is a no-op on Strata. |
| `VESTIGE_BACKFILL_AUTOFIRE`, `VESTIGE_FAILURE_FEEDBACK` | The v3 failure hooks do not run; use `causal_walk` and `reflex.sock`. |
| `VESTIGE_AUTO_CONSOLIDATE_MERGE` | Merging near duplicates needs embeddings, which Strata does not use. |
| `VESTIGE_DREAM_COMPILE_AUTOFIRE` | Replaced by live Strata FSRS re-derivation. |
| `VESTIGE_TRACE`, `VESTIGE_TRACE_RETENTION_DAYS` | Replaced by cryptographic receipts and DSSE witness bundles. |
| `VESTIGE_CLOUD_ENDPOINT`, `VESTIGE_CLOUD_SYNC_KEY`, `VESTIGE_CLOUD_ENCRYPTION_KEY` | Replaced by owner-only segment sealing via `vestige backup`. |

## Purged v3 CLI Subcommands

The following 8 legacy SQLite subcommands were stripped from the CLI in v6.2:
- `consolidate`
- `backfill`
- `ingest-git`
- `portable-export`
- `portable-import`
- `sync`
- `gc`
- `sandwich`

Their capabilities are now served lock-free through the 6 MCP tools or `vestige backup`.
