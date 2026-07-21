# Withheld cells

These are NOT counted in any published table. They are here so the exclusions are auditable rather
than something you have to take on trust.

- `runB-trial-2-zep*` — my harness never flushed Zep's FalkorDB graph between trials. These agents
  were served the fact "k_antares was decommissioned in favor of k_nashira as the live signer", and
  neither key exists in that trial's corpus. Both are from the previous trial. This measures my
  harness, not Zep. Fixed with a per-trial graph namespace and a hard flush, then re-run clean in
  `runB-trial-3`.
- `runB-trial-1-mem0.json` — same bug class. mem0 persists to `~/.mem0` in addition to its configured
  store, and only the configured one was cleared. Fixed, then re-run clean in `runB-trial-2` and
  `runB-trial-3`.
- `runB-trial-3-hindsight-DEAD.json` — not a contamination case. The backend returned zero successful
  retrievals out of nine attempts (`memory_layer_alive: false`, `retrieval_err_total: 9`). An arm whose
  backend never answered is a missing measurement, not a retrieval loss.
