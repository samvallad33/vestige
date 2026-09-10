#!/usr/bin/env python3
"""Write an exclusive local evaluation package from a verified task ledger."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import compare
import ledger
import qualification
import reconcile


def create(bundle, destination, baseline, candidate, *, billing_export=None):
    destination = Path(destination).resolve()
    ledger.require(not destination.exists(), "report destination already exists")
    result = compare.compare(bundle, baseline, candidate)
    screening = qualification.qualify(bundle, baseline, candidate)
    screening.pop("comparison", None)
    billing = reconcile.reconcile(bundle, billing_export) if billing_export else None
    # No prompts, memories, credentials, model responses or source archives are copied.
    # The package is a summary; reproduction requires the original controlled bundle.
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        root = Path(temp) / "package"
        root.mkdir()
        (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        (root / "qualification.json").write_text(json.dumps(screening, indent=2) + "\n")
        if billing is not None:
            (root / "reconciliation.json").write_text(
                json.dumps(billing, indent=2) + "\n"
            )
        rows = []
        for arm in (baseline, candidate):
            report = result["arms"][arm]
            rows.append(
                f"| {arm.replace('|', '/').replace(chr(10), ' ')} | {report['successes']}/{report['planned_tasks']} | "
                f"{report['total_usd']} | {report['usd_per_success']} | {report['accounting_complete']} |"
            )
        body = "\n".join(
            [
                "# Vestige company evaluation summary",
                "",
                f"Evidence: **{result['evidence_kind']}**. Analysis: **exploratory**.",
                "",
                "Synthetic evidence validates accounting only. It cannot demonstrate product savings.",
                "",
                "| Arm | Successful tasks | Total USD | USD per success | Accounting complete |",
                "|---|---:|---:|---:|---|",
                *rows,
                "",
                "Total USD includes failed attempts, retries and declared shared overhead. Null means unavailable.",
                "",
                "## Task wall-clock time",
                "",
                "Missing spans stay unavailable. These values are not summed request durations.",
                "",
                *[
                    f"- {arm}: p50={result['task_wall_time'][arm]['p50_ms']} ms; p95={result['task_wall_time'][arm]['p95_ms']} ms; missing={result['task_wall_time'][arm]['missing_tasks']}"
                    for arm in (baseline, candidate)
                ],
                "",
                f"Declared sample screening: **{screening['status']}** (exploratory; see qualification.json).",
                "Billing reconciliation: "
                + (
                    "see reconciliation.json; supplied evidence is not authenticated."
                    if billing
                    else "no billing export supplied."
                ),
                "",
                "## Evidence identity",
                "",
                f"- Contract SHA-256: `{result['contract_sha256']}`",
                f"- Events SHA-256: `{result['events_sha256']}`",
                f"- Analyzer SHA-256: `{result['analyzer_sha256']}`",
                "",
                "## Reproduction gate",
                "",
                "Retain the original frozen bundle in company-controlled storage. Run the ledger tests and",
                "compare.py against that bundle with the same arm selection. Verify the frozen model, agent,",
                "source, history, evaluator and tool identities before using any result. This summary does",
                "not include the underlying private artifacts and is not independently sufficient to reproduce them.",
                "",
                "## Remaining qualification",
                "",
                "Before a savings claim: qualify live provider usage and rates; independently score frozen",
                "developer tasks; preserve all missing/failed attempts; predeclare held-out quality bounds",
                "and sampling; measure task wall time and setup/maintenance overhead; check cache parity.",
                "",
                "A positive cost difference is not proof of causality or universal savings. See comparison.json",
                "for every paired outcome, missing category and exploratory interval.",
                "",
            ]
        )
        (root / "REPORT.md").write_text(body)
        manifest = {
            "version": "vestige-company-evaluation/v1",
            "files": {
                path.name: ledger.digest(path) for path in sorted(root.iterdir())
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Exclusive mkdir closes the check/create race; failures clean only this new directory.
        destination.mkdir()
        try:
            for path in root.iterdir():
                shutil.copyfile(path, destination / path.name)
        except BaseException:
            shutil.rmtree(destination)
            raise
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--billing-export", type=Path)
    args = parser.parse_args()
    create(
        args.bundle,
        args.destination,
        args.baseline,
        args.candidate,
        billing_export=args.billing_export,
    )
