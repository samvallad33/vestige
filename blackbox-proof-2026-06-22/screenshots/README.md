# Proof Pack Screenshots

Captured with Playwright (`@playwright/test`, headless Chromium, 1440×1700 @2x)
from the **live** Vestige dashboard at `http://localhost:5173/dashboard`,
proxying to a real `vestige-mcp` server with real trace data.

| File | Tab | Shows |
|------|-----|-------|
| `black-box.png` | Black Box | spine header (WebSocket Connected), run picker (`proof`/`proof2`), timeline scrubber + colored ticks, current event detail, memory pulse, **event producers** (with honest `dream.patch`/`sanhedrin.veto` off-by-default states), receipts panel, full event log |
| `receipts.png` | Black Box → Receipts | a real `ReceiptCard`: receipt id, retrieved/suppressed/trust-floor, activation path, retrieved ids, "Open receipt in Cinema" |
| `memory-prs.png` | Memory PRs | killer line + quarantine-review note, Fast/Risk-Gated/Paranoid modes, status filters, PR rows, cognition diff, "Why this opened" signal (`sensitive_topic`), `Decided: promote` |
| `graph.png` | Graph | the live WebGL memory constellation + Memory Cinema button (unchanged) |

Re-capture: start the dev server (`pnpm --filter @vestige/dashboard dev`),
point its `/api` proxy at a running `vestige-mcp` with trace data, then run the
capture script (see PROOF.md "Reproduce").
