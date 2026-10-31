"""Prepare source-unresolved Causal-Drift-Eval and distinct local fixtures."""

import argparse
import json

from campaigns.reconcile_safe.prepare import prepare as prepare_reconciliation


def prepare(output):
    return prepare_reconciliation(output, benchmark="Causal-Drift-Eval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("benchmark", "state", "official_source_status", "sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
