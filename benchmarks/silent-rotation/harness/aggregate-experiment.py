#!/usr/bin/env python3
"""Aggregate an N-trial Silent Rotation experiment into a statistical result.

Reads experiment-<seed>-<N>/trial-*/{anarchy,sync}.json and prints:
  - a per-trial table (correct key, anarchy verdict, sync verdict, backfill use)
  - aggregate: anarchy success X/N, sync success Y/N
  - a one-sided p-value for "sync succeeds more than anarchy" (exact, stdlib only)
  - the honest headline a senior engineer reads

"Success" = fleet_verdict == 'fixed_correctly' (green AND production-safe AND
right key). Anything else (merge conflict, still-red, green_but_voids_prod) is a
failure. This is the outcome that matters: shipped a correct, production-safe fix.
"""
import json
import math
import sys
from pathlib import Path


def is_success(v: dict) -> bool:
    return v.get("fleet_verdict") == "fixed_correctly"


def binom_p_at_least(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact via stdlib."""
    if k <= 0:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def main():
    expdir = Path(sys.argv[1])
    n = int(sys.argv[2])
    # Which arms to aggregate. Defaults to the 3-arm set; a 2-arm run (anarchy
    # sync) still works because a missing rag.json is tolerated per trial.
    import os as _os
    arms = _os.environ.get("ARMS", "anarchy rag sync").split()
    rows = []
    succ = {arm: 0 for arm in arms}
    tot_tok = {arm: 0 for arm in arms}
    tot_cost = {arm: 0.0 for arm in arms}
    # Per-arm MEMORY-LAYER HEALTH. Without this, an arm whose backend never
    # answered renders as a plain "0/5" -- visually identical to a real
    # retrieval loss. That is exactly how three dead competitor arms
    # (hindsight 19/19 errors, zep 22/22, mem0 14/15) were published as
    # substantive defeats in earlier runs. A trial only counts as a real
    # measurement if that arm's memory layer actually answered.
    dead_trials = {arm: 0 for arm in arms}    # memory layer never answered
    live_trials = {arm: 0 for arm in arms}    # memory layer answered at least once
    no_data_trials = {arm: 0 for arm in arms}  # arm produced no result JSON at all
    for t in range(1, n + 1):
        td = expdir / f"trial-{t}"
        try:
            man = json.load(open(td / "manifest.json"))
        except (FileNotFoundError, json.JSONDecodeError):
            rows.append((t, "?", {arm: "MISSING" for arm in arms}, "?"))
            continue
        verdicts = {}
        for arm in arms:
            try:
                d = json.load(open(td / f"{arm}.json"))
            except (FileNotFoundError, json.JSONDecodeError):
                # No JSON at all: the arm never produced a result (crashed before
                # writing, or the run was stopped before reaching it). Either way
                # this trial is NOT a measurement of that product.
                verdicts[arm] = "MISSING"
                if arm != "anarchy":
                    no_data_trials[arm] += 1
                continue
            verdicts[arm] = d.get("fleet_verdict", "?")
            succ[arm] += int(is_success(d))
            # anarchy has NO memory tool by design, so it is never "dead".
            if arm != "anarchy":
                if (d.get("memory_unavailable") is True
                        or d.get("memory_layer_alive") is False
                        or d.get("fleet_verdict") == "errored"):
                    dead_trials[arm] += 1
                    verdicts[arm] = "MEMORY_UNAVAILABLE"
                elif d.get("memory_layer_alive") is True:
                    live_trials[arm] += 1
            tot_tok[arm] += d.get("fleet_tokens", {}).get("total", 0) or 0
            tot_cost[arm] += d.get("fleet_cost_usd", 0) or 0
        # backfill count comes from the sync arm (the memory arm)
        bf = "?"
        try:
            s = json.load(open(td / "sync.json"))
            bf = f"{s.get('agents_used_vestige_backfill', '?')}"
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        rows.append((t, man.get("correct_kid", "?"), verdicts, bf))

    # Display labels per arm. Any arm not listed falls back to its uppercased
    # name (render is robust to new arms).
    LABEL = {"anarchy": "ANARCHY (no mem)", "rag": "RAG (similarity)",
             "sync": "SYNC (Vestige)", "mem0": "MEM0 (mem0ai)",
             "supermemory": "SUPERMEMORY", "hindsight": "HINDSIGHT",
             "zep": "ZEP (Graphiti KG)"}
    line = "=" * (24 + 26 * len(arms) + 10)
    print(line)
    print("  SILENT ROTATION — N-TRIAL RANDOMIZED BENCHMARK (per-trial)")
    print(line)
    hdr = f"  {'trial':<6} {'correct key':<14} "
    hdr += "".join(f"{LABEL.get(a, a.upper()):<26}" for a in arms)
    hdr += "backfill"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for (t, kid, verdicts, bf) in rows:
        rowstr = f"  {t:<6} {kid:<14} "
        rowstr += "".join(f"{verdicts.get(a, '?'):<26}" for a in arms)
        rowstr += bf
        print(rowstr)
    print(line)

    # Cost + tokens per arm (measured, from each trial's fleet_cost_usd / fleet_tokens).
    print(f"  {'':<20}" + "".join(f"{LABEL.get(a, a.upper()):>26}" for a in arms))
    print(f"  {'Total tokens':<20}" + "".join(f"{tot_tok[a]:>26,}" for a in arms))
    print(f"  {'Total cost (USD)':<20}" + "".join(f"{('$%.4f' % tot_cost[a]):>26}" for a in arms))
    grand_cost = sum(tot_cost.values())
    grand_tok = sum(tot_tok.values())
    print(f"  {'Experiment total':<20} {('$%.4f' % grand_cost)} ({grand_tok:,} tokens across {len(arms)} arms)")
    print(line)

    for arm in arms:
        # A score is only honest if that arm's memory layer actually answered.
        # Flag any arm with dead trials LOUDLY and inline -- never let a crash
        # masquerade as a retrieval loss in the published table.
        if arm != "anarchy" and (dead_trials[arm] or no_data_trials[arm]):
            bad = dead_trials[arm] + no_data_trials[arm]
            flag = (f"   <-- !! {bad}/{n} trials NOT A MEASUREMENT "
                    f"(dead={dead_trials[arm]} no-data={no_data_trials[arm]}; "
                    f"backend never answered -- do NOT publish this score)")
        else:
            flag = ""
        print(f"  {LABEL.get(arm, arm.upper()):<28} shipped correct+prod-safe: {succ[arm]}/{n}{flag}")
    # Explicit per-arm health line so a reader never has to open the raw JSON.
    print()
    print("  MEMORY-LAYER HEALTH (trials where the arm's backend actually answered):")
    for arm in arms:
        if arm == "anarchy":
            print(f"    {LABEL.get(arm, arm.upper()):<28} n/a (no memory tool by design)")
        else:
            bad = dead_trials[arm] + no_data_trials[arm]
            state = "OK" if bad == 0 and live_trials[arm] == n else (
                "UNUSABLE" if live_trials[arm] == 0 else "DEGRADED")
            print(f"    {LABEL.get(arm, arm.upper()):<28} live {live_trials[arm]}/{n}"
                  f"  dead {dead_trials[arm]}/{n}  no-data {no_data_trials[arm]}/{n}   [{state}]")
    anarchy_succ = succ.get("anarchy", 0)
    sync_succ = succ.get("sync", 0)

    # One-sided significance under the NULL that memory does nothing: a memoryless
    # agent's chance of blindly picking the correct key is 1/K where K = keyring
    # size (recovered from a trial manifest). With a large keyring this null is
    # tiny, so even one clean sync trial is significant, and a full sweep is
    # crushing. p0 = 1/K is a CONSERVATIVE bound (the fleet must also AGREE).
    keyring_size = 3
    try:
        m1 = json.load(open(expdir / "trial-1" / "manifest.json"))
        keyring_size = max(2, len(m1.get("key_names", [])) or 3)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    p0 = 1.0 / keyring_size
    pval = binom_p_at_least(sync_succ, n, p0)
    print(f"  keyring size (K): {keyring_size}  ->  blind-guess success = 1/{keyring_size} = {100.0/keyring_size:.1f}%")
    print(f"  p-value (sync successes vs blind-guess null p0=1/{keyring_size}): {pval:.4g}")
    if anarchy_succ == 0 and sync_succ == n and n >= 3:
        print(f"  => Clean separation: anarchy 0/{n}, sync {n}/{n}. "
              f"Under the null, that is p = (1/{keyring_size})^{n} = {p0**n:.2e}.")
    print(line)
    # The headline is now DERIVED, never asserted. It used to be four
    # unconditional print() calls with no reference to the data, so it fired
    # even when the numbers contradicted it -- it already sat directly under
    # "RAG 2/5" in the GLM summary, and it would have printed unchanged had
    # Vestige scored 0/5. A constant is not a finding.
    if anarchy_succ == 0 and sync_succ == n and n >= 3:
        print("  HEADLINE: With the correct key RANDOMIZED and absent from every file")
        print("  the agent can read, the memoryless fleet did not ship a")
        print("  production-safe fix in any trial; the Vestige-coordinated fleet did")
        print(f"  in all {n}. Same model, same corpus, same turn budget.")
    else:
        print(f"  RESULT: anarchy {anarchy_succ}/{n}, sync (Vestige) {sync_succ}/{n}.")
        print("  This run did NOT produce clean separation -- report the numbers as")
        print("  measured. Do not reuse the clean-separation headline.")
    # Fairness qualifiers that must travel WITH the numbers, always.
    print(line)
    print("  METHOD NOTES (must be published alongside any number above):")
    print("   - The Vestige arm additionally has a vestige_log WRITE tool for fleet")
    print("     coordination that no competitor arm is given. Transcripts show the")
    print("     key was recovered by the read path (backfill) before any log write.")
    print("   - Vestige's causal join follows a shared 'active_key' tag that the")
    print("     seed places on the cause and the failure. Competitor arms receive")
    print("     the corpus with tag metadata stripped.")
    print("   - Any arm whose memory layer returned zero successful retrievals is")
    print("     memory_unavailable, NOT a retrieval loss. Check memory_layer_alive")
    print("     in each arm's JSON before quoting its score.")
    print(line)


if __name__ == "__main__":
    main()
