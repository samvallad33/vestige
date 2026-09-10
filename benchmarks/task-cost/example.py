#!/usr/bin/env python3
"""Create an explicitly synthetic accounting fixture, never a savings benchmark."""
import argparse
import json
from pathlib import Path

from ledger import VERSION, digest, freeze, record


def create(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        path = root / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        return {"path": name, "sha256": digest(path)}

    stimulus = write("stimulus.json", {"task": "Synthetic accounting fixture; no developer task executed."})
    evaluator = write("evaluator.json", {"kind": "synthetic declared outcomes", "executes_tests": False})
    identity = {"model": "fixture-model-v1", "model_revision": "fixture-model-v1", "effort": "fixed",
                "agent_revision": "fixture-agent-v1", "tool_config_sha256": stimulus["sha256"],
                "memory_snapshot_sha256": stimulus["sha256"]}
    contract = {"version": VERSION, "evidence_kind": "synthetic", "currency": "USD", "repetitions": 1,
                "arms": {"native": identity, "candidate": identity},
                "cases": [{"id": case, "state": state, "workload": "accounting-only",
                           "prompt": stimulus, "source": stimulus, "history": stimulus, "evaluator": evaluator}
                          for case, state in (("cold-example", "cold"), ("warm-example", "warm"))],
                "prices": {"fixture": {"model": "fixture-model-v1", "effective_date": "2026-09-10",
                                       "source": "synthetic rates; not vendor prices",
                                       "usd_per_million_tokens": {"uncached_input": "2", "cached_input": "0.2",
                                                                 "output": "8"}}},
                "execution_policy": "Synthetic fixture only. No agents or provider requests.",
                "scoring_policy": "Declared fixture outcomes test accounting, not product quality."}
    write("contract.json", contract)
    freeze(root)
    events = []
    for arm in contract["arms"]:
        for case in contract["cases"]:
            prefix = f"{arm}-{case['id']}"
            response = write(prefix + "-response.json", {
                "id": prefix, "model": "fixture-model-v1",
                "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200},
                          "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 30}}})
            events.append({"id": prefix, "kind": "request", "arm": arm, "case": case["id"], "trial": 0,
                           "phase": "agent", "format": "openai_responses", "price_id": "fixture",
                           "request": stimulus, "response": response, "elapsed_ms": 100})
            events.append({"id": prefix + "-outcome", "kind": "outcome", "arm": arm,
                           "case": case["id"], "trial": 0,
                           "status": "success" if case["state"] == "cold" else "failure", "evidence": evaluator})
        coverage = write(arm + "-overhead.json", {"note": "Synthetic explicit overhead declaration.",
                                                  "unmeasured_categories": []})
        events.append({"id": arm + "-setup", "kind": "overhead", "arm": arm, "phase": "setup",
                       "usd": "0.01", "basis": "estimated", "evidence": coverage})
        events.append({"id": arm + "-coverage", "kind": "overhead_coverage", "arm": arm,
                       "status": "complete", "evidence": coverage})
    for event in events:
        record(root, event)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    create(parser.parse_args().directory)
