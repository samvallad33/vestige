# Vestige Strata for OpenClaw

The kernel of the Causal Proof Engine, for your OpenClaw agent: **Strata**,
Vestige 4.0's signed, append-only causal log. Zero vectors, zero RAG: every
write passes a gate, comes back with a receipt you can replay byte-for-byte,
and every answer carries its proof. Retention follows an FSRS-6 review
schedule. Your agent's history is a log you own, on your disk, that nobody
can silently rewrite.

## The tools

| Tool | What it does |
|---|---|
| `strata_ingest` | Store a memory. Nothing is merged by similarity; the write passes the log's gate and returns with a receipt. |
| `strata_recall` | Recall by exact handle — memory id, unique id prefix, or exact tag — plus its one-hop recorded causal edges. No similarity search. |
| `strata_causal_walk` | Investigate a failure from an explicit start point: a bounded backward walk over recorded causal edges (`closed_by`, `derived_from`, `evidence_of`, `touched`). Hypotheses anchored in what the log actually recorded, never a guess — it refuses with `needs_report` when there is no trail. |
| `strata_compose` | List never-composed memory pairs as leads: two live memories in one scope with no recorded edge between them. |
| `strata_verify` | Replay the log and verify the receipt chain and signing-key fingerprint. Read-only. |
| `strata_stats` | Store statistics and health. |

## Install

1. Install the Vestige CLI (4.0+):

```sh
brew install samvallad33/tap/vestige
# or grab the release archive for your machine — see
# https://github.com/samvallad33/vestige#install
```

2. Install this plugin — it finds `vestige` via `$VESTIGE_BIN`, your `PATH`,
   `~/.local/bin`, `~/.cargo/bin`, or Homebrew.

## Why exact handles instead of similarity search

Similarity search guesses. A Strata log records. When your agent recalls a
memory by handle, it gets the memory plus the causal edges the log actually
witnessed — and when a failure has no recorded trail, the causal walk says so
instead of confabulating a cause. Retrieval you can audit beats retrieval you
must trust.

## Local-first

The log lives in your Vestige data directory (`~/Library/Application
Support/com.vestige.core` on macOS by default). No account, no telemetry, no
cloud. The log is single-writer by design, so `strata_verify` refuses while a
Vestige server holds the store — that refusal is the integrity guarantee
working, not a bug.

Source: [samvallad33/vestige](https://github.com/samvallad33/vestige)
