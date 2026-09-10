#!/usr/bin/env python3
"""Reconcile ledger request costs against a caller-supplied billing export.

Expected export: {source, currency:'USD', charges:[{usage_format, request_id, usd}]}.
This validates supplied evidence; it does not authenticate a vendor invoice.
"""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import ledger


def reconcile(bundle, billing_export):
    report = ledger.evaluate(bundle)
    billing = ledger.read_json(billing_export)
    ledger.require(
        billing.get("currency") == "USD"
        and isinstance(billing.get("source"), str)
        and billing["source"].strip(),
        "billing export requires USD currency and source provenance",
    )
    charges = {}
    for charge in billing.get("charges", []):
        key = (charge.get("usage_format"), charge.get("request_id"))
        ledger.require(
            all(isinstance(part, str) and part for part in key) and key not in charges,
            "invalid/duplicate billing request",
        )
        charges[key] = ledger.amount(charge["usd"])
    expected = {}
    unknown = []
    for event in report["cost_events"]:
        if "provider_response_id" not in event:
            continue
        key = (event["usage_format"], event["provider_response_id"])
        if event["usd"] is None or not key[1]:
            unknown.append(event["id"])
            continue
        ledger.require(key not in expected, "provider request appears twice")
        expected[key] = Decimal(event["usd"])
    missing = sorted(set(expected) - charges.keys())
    unmatched = sorted(set(charges) - expected.keys())
    differences = [
        {
            "usage_format": key[0],
            "request_id": key[1],
            "ledger_usd": str(expected[key]),
            "billing_usd": str(charges[key]),
            "delta_usd": str(charges[key] - expected[key]),
        }
        for key in sorted(expected.keys() & charges.keys())
        if expected[key] != charges[key]
    ]
    return {
        "version": "vestige-billing-reconciliation/v1",
        "billing_export_sha256": ledger.digest(billing_export),
        "contract_sha256": report["contract_sha256"],
        "events_sha256": report["events_sha256"],
        "matched_request_count": len(expected.keys() & charges.keys()),
        "matches_supplied_export": bool(expected)
        and not missing
        and not unmatched
        and not differences
        and not unknown,
        "missing_billing_requests": missing,
        "unmatched_billing_requests": unmatched,
        "differences": differences,
        "unknown_usage_events": unknown,
        "source": billing["source"],
        "boundary": "Exact comparison to caller-supplied request charges; no invoice authenticity, tax, discount or complete-account audit is implied.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("billing_export", type=Path)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.bundle, args.billing_export), indent=2))
