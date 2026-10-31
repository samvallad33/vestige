"""Prepare an identity hold plus local stress fixtures; never an official score.

Bounded exact-name searches on 2026-10-31 found no primary benchmark repository,
paper or dataset for ReconcileSafe. This is not proof of nonexistence. The local
fixtures remain useful input contracts while the official identity is unresolved.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def local_fixtures():
    return [
        {
            "id": "agreeing-branches",
            "kind": "agreement_control",
            "visible": {
                "base": {"revision": "base", "service": "alpha", "timeout": 30},
                "branches": [
                    {"parent": "base", "timeout": 45},
                    {"parent": "base", "timeout": 45},
                ],
            },
            "gold": {
                "conflict": False,
                "reason": "both branches propose the same value",
            },
        },
        {
            "id": "divergent-branches",
            "kind": "branch_conflict",
            "visible": {
                "base": {"revision": "base", "service": "alpha", "timeout": 30},
                "branches": [
                    {"parent": "base", "timeout": 45},
                    {"parent": "base", "timeout": 60},
                ],
            },
            "gold": {"conflict": True, "conflicting_field": "timeout"},
        },
        {
            "id": "stale-write",
            "kind": "stale_concurrent_write",
            "visible": {
                "read_revision": "r1",
                "current_revision": "r2",
                "proposal": {"parent": "r1", "timeout": 90},
            },
            "gold": {"stale": True, "require_reconciliation_before_apply": True},
        },
    ]


def native_readiness(capability_evidence=None):
    """No configured flag or caller-supplied JSON can certify a native H1 route."""
    return {
        "state": "UNSUPPORTED_NATIVE_API"
        if capability_evidence is None
        else "UNVERIFIED_NATIVE_EVIDENCE",
        "requested_mechanisms": [
            "sheaf_cohomology_h1",
            "sheaf_laplacian",
            "branch_reconciliation",
        ],
        "official_score": None,
        "native_execution_verified": False,
        "reason": "requires a registered native API, exact source identity and verified execution receipts; no generic graph check is substituted",
    }


def prepare(output, *, benchmark="ReconcileSafe"):
    fixtures = local_fixtures()
    manifest = {
        "benchmark": benchmark,
        "official_source_status": "SOURCE_UNRESOLVED",
        "official_source_url": None,
        "official_commit": None,
        "official_license": None,
        "checked_date": "2026-10-31",
        "state": "PREPARED_SOURCE_AND_NATIVE_GATED",
        "research": {
            "method": "bounded exact-name public web searches; GitHub and arxiv domain queries",
            "queries": [
                f'"{benchmark}" benchmark',
                f'"{benchmark}" site:github.com',
                f'"{benchmark}" site:arxiv.org',
            ],
            "result": "no authoritative exact benchmark identity established; does not prove nonexistence",
        },
        "local_extension": {
            "name": "local-reconciliation-contract-fixtures-v1",
            "official_benchmark": False,
            "fixture_count": len(fixtures),
            "fixtures_sha256": digest(fixtures),
            "provider_calls": 0,
            "native_execution": native_readiness(),
        },
        "official_score": None,
        "claim_boundary": "fixture preparation is not native sheaf qualification or official benchmark execution",
    }
    manifest["sha256"] = digest(manifest)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, value in (
        ("manifest.json", manifest),
        ("local-fixtures-private.json", fixtures),
        (
            "runtime-inputs.json",
            [
                {"record_id": digest(row["id"])[:24], "observations": row["visible"]}
                for row in fixtures
            ],
        ),
    ):
        with (root / name).open("x", encoding="utf-8") as stream:
            os.chmod(root / name, 0o600)
            stream.write(canonical(value) + "\n")
    return manifest


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
