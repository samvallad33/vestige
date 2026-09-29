# From Sanhedrin receipts to 4.0 claim verdicts

> **Removed in 4.0.** Sanhedrin — the model-run verifier that read transcripts
> and vetoed a turn with `vestige.sanhedrin.receipt.v1` documents — is gone.
> Hard rule H3: no LLM decides. The state directory
> `~/.vestige/sanhedrin/` (latest.json, receipts/, appeals.jsonl,
> fail-open.jsonl) and its env knobs (`VESTIGE_SANHEDRIN_*`, including the
> staged-evidence overlay and the compatibility flags described here
> historically) belong to that retired pipeline. The companion schema
> [`SANHEDRIN_TEST_INTEGRITY_DELTAS.md`](SANHEDRIN_TEST_INTEGRITY_DELTAS.md)
> is kept as a historical archive. This page describes what replaced it.

## What replaces a model verdict: deterministic claim verdicts

4.0 keeps the *accountability* of Sanhedrin and removes the *judge*. Claims
about work are decided by the deterministic gate against the log, and the
decision is a signed frame like everything else:

- **`CLAIM` (frame kind 48)** — an agent's declared claim, receipt-cited: it
  names the receipts that allegedly support it.
- **`CLAIM_VERDICT` (frame kind 37)** — the deterministic verdict over that
  claim, computed by replaying the cited receipts at their decision points.
  No model runs; the same log at the same `as_of` yields the same verdict.

Every receipt payload carries the PARAMS hash that governed it and the
caller-supplied decision point, so a verdict is replayable bit for bit — see
[RECEIPTS.md](RECEIPTS.md) for the registry and
`strata-verify <dir>` for the verifier.

## The enforcement path that remains

Where a 3.x install ran Sanhedrin through a Stop hook, the 4.0 pattern is a
deterministic claim gate: a hook that fails closed (refuses the stop when
Vestige is unreachable), checks claims against live receipts and commands, and
records what it used. Fail-closed replaces fail-open: an unverifiable claim is
a veto with a receipt trail, never a silent pass.

<!-- TODO(comment): the claim-gate hook script itself is not in this tree at
this HEAD (hooks/ still carries the 3.x Sanhedrin stack). Wire or document the
claim-gate script before treating its fail-closed contract as shipped
behavior. -->

## Compatibility rules for old artifacts

- Existing `vestige.sanhedrin.receipt.v1` files remain readable as documents;
  nothing in 4.0 produces, validates, or appeals them.
- 4.0 keeps rendering unknown receipt schemas defensively wherever it renders
  receipts at all; treat a Sanhedrin document as historical evidence, not as a
  live verdict.
- Durable support for any claim now means: a receipt frame in the STRATA log
  whose verification passes — not a staged overlay, a transcript regex scan,
  or a model's opinion.
