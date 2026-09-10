#!/usr/bin/env python3
"""Exploratory paired task accounting over a verified frozen ledger.

No model calls. This does not certify provider billing or task evaluators.
"""
import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import random

import ledger


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(bundle, baseline, candidate, samples=2000, seed=0):
    ledger.require(type(samples) is int and 100 <= samples <= 100_000,
                   "samples must be between 100 and 100000")
    ledger.require(type(seed) is int, "seed must be an integer")
    root = Path(bundle)
    report = ledger.evaluate(root)  # Verifies contract, accountant, artifacts and events.
    contract = ledger.read_json(root / "contract.json")
    ledger.require(baseline != candidate and baseline in report["arms"] and candidate in report["arms"],
                   "select two distinct contract arms")
    outcomes = {}
    for line in (root / "events.jsonl").read_text().splitlines():
        if line.strip():
            event = ledger.parse_json(line)
            if event["kind"] == "outcome":
                outcomes[(event["arm"], event["case"], event["trial"])] = event["status"]
    costs = defaultdict(lambda: Decimal(0))
    unknown = set()
    durations = defaultdict(list)
    for event in report["cost_events"]:
        if event.get("case") is None:
            continue
        key = (event["arm"], event["case"], event["trial"])
        if event["usd"] is None:
            unknown.add(key)
        else:
            costs[key] += Decimal(event["usd"])
        durations[key].append(event.get("elapsed_ms"))
    pairs = []
    clusters = defaultdict(list)
    for case in contract["cases"]:
        for trial in range(contract["repetitions"]):
            pair = {"case": case["id"], "trial": trial, "state": case["state"],
                    "workload": case["workload"], "arms": {}}
            for arm in (baseline, candidate):
                key = (arm, case["id"], trial)
                timings = durations[key]
                pair["arms"][arm] = {"outcome": outcomes.get(key),
                             "task_attributed_usd": str(costs[key]) if key not in unknown and timings else None,
                             "summed_request_ms": sum(timings) if timings and None not in timings else None}
            pairs.append(pair)
            clusters[case["id"]].append(pair)
    complete = all(report["arms"][arm]["accounting_complete"] for arm in (baseline, candidate))
    costs_per_success = [report["arms"][arm]["usd_per_success"] for arm in (baseline, candidate)]
    savings = None
    if complete and all(value is not None for value in costs_per_success):
        native, treatment = map(Decimal, costs_per_success)
        if native > 0:
            savings = str(Decimal(1) - treatment / native)
    wins = losses = ties = missing = 0
    for pair in pairs:
        a, b = pair["arms"][baseline]["outcome"], pair["arms"][candidate]["outcome"]
        if a is None or b is None:
            missing += 1
        elif (a == "success") == (b == "success"):
            ties += 1
        elif b == "success":
            wins += 1
        else:
            losses += 1
    # Resample case clusters, retaining all repetitions and both arms together.
    # This is exploratory: no post-hoc interval becomes a confirmatory claim.
    interval = None
    if not missing and len(clusters) >= 2:
        rng = random.Random(seed)
        groups = list(clusters.values())
        deltas = []
        for _ in range(samples):
            selected = [pair for group in rng.choices(groups, k=len(groups)) for pair in group]
            deltas.append(sum((pair["arms"][candidate]["outcome"] == "success") -
                              (pair["arms"][baseline]["outcome"] == "success") for pair in selected) / len(selected))
        interval = {"low": percentile(deltas, .025), "high": percentile(deltas, .975),
                    "method": "exploratory paired case-cluster percentile bootstrap",
                    "samples": samples, "seed": seed, "case_clusters": len(clusters)}
    latency = {}
    for arm in (baseline, candidate):
        values = [pair["arms"][arm]["summed_request_ms"] for pair in pairs]
        known = [value for value in values if value is not None]
        latency[arm] = {"known_tasks": len(known), "missing_tasks": len(values) - len(known),
                        "p50_summed_request_ms": percentile(known, .5),
                        "p95_summed_request_ms": percentile(known, .95)}
    return {"version": "vestige-task-comparison/v1", "analysis_kind": "exploratory",
            "evidence_kind": report["evidence_kind"], "contract_sha256": report["contract_sha256"],
            "events_sha256": report["events_sha256"], "analyzer_sha256": ledger.digest(__file__),
            "baseline": baseline, "candidate": candidate, "accounting_complete": complete,
            "arms": {arm: report["arms"][arm] for arm in (baseline, candidate)},
            "cost_per_success_reduction_fraction": savings,
            "paired_success": {"candidate_wins": wins, "candidate_losses": losses,
                               "ties": ties, "missing_pairs": missing, "exploratory_interval": interval},
            "request_duration": latency, "pairs": pairs,
            "limitations": [
                "Synthetic evidence is an accounting test, never product savings evidence.",
                "Task-attributed pair costs exclude shared overhead; arm totals include it.",
                "Summed request duration is not task wall-clock latency and can include parallel calls.",
                "Bootstrap intervals are exploratory and unstable with few independent case clusters.",
                "No causal attribution, noninferiority, live provider qualification or invoice reconciliation is established.",
                "Unknown billing, missing outcomes or missing overhead prevent complete cost comparisons."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(compare(args.bundle, args.baseline, args.candidate, args.samples, args.seed), indent=2))


if __name__ == "__main__":
    main()
