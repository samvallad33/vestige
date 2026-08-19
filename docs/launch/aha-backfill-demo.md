# The 90-second Backfill demo

## The audience should leave with one idea

**Normal memory finds what sounds like the failure. Vestige surfaces an earlier, evidence-linked decision for review and future retrieval.**

Do not show the dashboard, a graph, or a tool catalog. The screen should show only a tiny incident timeline and the contrast between two answers.

## Run it

From a local Vestige checkout:

```bash
./scripts/aha-backfill-demo.sh timeout
./scripts/aha-backfill-demo.sh migration
./scripts/aha-backfill-demo.sh permissions
```

The script uses a temporary local database and removes it afterward. It does not touch a presenter's real memory store.

## Pitch-ready scenarios

| Scenario | Quiet earlier decision | Failure today | Shared evidence |
|---|---|---|---|
| `timeout` | Lowered `API_TIMEOUT` for faster cold starts | Auth service crashes under login load | `API_TIMEOUT` |
| `migration` | Approved `CUSTOMER_ID` as text for a partner import mapping | Supplier feed assigns order lines to two tenants | `CUSTOMER_ID` |
| `permissions` | Raised `PERMISSION_CACHE_TTL` to reduce lookups | Revoked users retain access | `PERMISSION_CACHE_TTL` |

Use `timeout` with technical founders, `migration` with data/platform teams, and `permissions` with B2B or security-sensitive product teams. Each is an intentionally fictional, isolated incident—not a claim about a customer's system.

## What to say

1. "An agent sped up cold starts by lowering a timeout. Three days later the auth service crashed during a login surge."
2. "A normal search sees another 500 error and calls that relevant. It resembles the incident, but it cannot explain it."
3. "Backfill starts at the failure and reaches backward through the shared incident key: `API_TIMEOUT`. It surfaces the quiet configuration decision that came first as a ranked candidate."
4. "That ranked candidate is then promoted, with an evidence link, for future retrieval."

Stop there. The value is not that Vestige has a clever search score. The value is that an agent can recover an evidence-backed candidate lesson from its own history.

## Proof boundaries

- The surfaced memory is a ranked **candidate cause**, not a claim of proven causality.
- The demo uses an explicit shared identifier (`API_TIMEOUT`) and backward time ordering.
- The `--contrast` output labels its baseline honestly: semantic hybrid search when embeddings are ready, otherwise keyword BM25.
- The CLI's default run promotes the candidate. For a preview-only demo, use `vestige backfill --contrast --no-promote` after seeding the three memories.

## What success looks like

The audience can repeat this sentence without prompting: **"It finds an earlier evidence-backed candidate, not just a similar error."**
